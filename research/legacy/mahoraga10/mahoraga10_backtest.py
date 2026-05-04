from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import mahoraga6_1 as m6
from mahoraga10_alpha import build_alpha_components, fit_alpha_model, precompute_alpha_path
from mahoraga10_config import Mahoraga10Config
from mahoraga10_policy import (
    FEATURE_COLS,
    apply_classifier,
    build_daily_context,
    build_policy_weekly,
    build_weekly_dataset,
    fit_best_classifier,
    iter_policy_candidates,
    weekly_to_daily_policy,
)
from mahoraga10_universe import members_at_date, union_universe
from mahoraga10_utils import bhy_qvalues, iter_grid, paired_ttest_pvalue, summarize_diff_strategy


def _load_folds(cfg: Mahoraga10Config, ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    folds = pd.DataFrame(m6.build_contiguous_folds(cfg, pd.DatetimeIndex(ohlcv["close"].index)))
    folds = folds[folds["fold"].isin(cfg.mode_folds())].copy().sort_values("fold")
    return folds


def _prepare_fold_invariants(
    ohlcv: Dict[str, pd.DataFrame],
    cfg: Mahoraga10Config,
    universe_schedule: Optional[pd.DataFrame],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> Dict[str, Any]:
    univ_master = union_universe(ohlcv, universe_schedule, list(cfg.universe_static))
    close = ohlcv["close"][univ_master].copy()
    high = ohlcv["high"][univ_master].copy()
    low = ohlcv["low"][univ_master].copy()
    rets = close.pct_change().fillna(0.0)
    idx = close.index
    qqq = m6.to_s(ohlcv["close"][cfg.bench_qqq].reindex(idx).ffill(), "QQQ")
    spy = m6.to_s(ohlcv["close"][cfg.bench_spy].reindex(idx).ffill(), "SPY")

    crisis_scale, crisis_state = m6.compute_crisis_gate(qqq, cfg)
    turb_scale = m6.compute_turbulence(close, ohlcv["volume"][univ_master], qqq, cfg)
    vix_series = None
    if cfg.bench_vix in ohlcv.get("close", pd.DataFrame()).columns:
        vix_series = m6.to_s(ohlcv["close"][cfg.bench_vix].reindex(idx).ffill(), "VIX")
    corr_rho, corr_scale_legacy, corr_state = m6.compute_corr_shield_series(
        rets, idx, cfg, univ_master,
        use_pit_universe=universe_schedule is not None and len(universe_schedule) > 0,
        universe_schedule=universe_schedule,
        vix=vix_series,
    )
    reb_dates = set(close.resample(cfg.rebalance_freq).last().index)
    alpha_fit = fit_alpha_model(close, qqq, cfg, train_start, train_end)
    components = build_alpha_components(close, qqq, cfg)
    return {
        "close": close,
        "high": high,
        "low": low,
        "rets": rets,
        "idx": idx,
        "qqq": qqq,
        "spy": spy,
        "crisis_scale": crisis_scale,
        "crisis_state": crisis_state,
        "turb_scale": turb_scale,
        "corr_rho": corr_rho,
        "corr_scale_legacy": corr_scale_legacy,
        "corr_state": corr_state,
        "reb_dates": reb_dates,
        "members_at": lambda dt: members_at_date(universe_schedule, dt, univ_master),
        "components": components,
        "alpha_fit": alpha_fit,
    }


def _apply_policy_to_alpha_cache(
    pre: Dict[str, Any],
    alpha_cache: Dict[str, Any],
    policy_daily: pd.DataFrame,
    cfg: Mahoraga10Config,
    costs: m6.CostsConfig,
    label: str,
) -> Dict[str, Any]:
    idx = pre["idx"]
    rets = pre["rets"]
    w_exec_1x = alpha_cache["weights_exec_1x"]
    gross_1x = alpha_cache["gross_1x"]

    realized = gross_1x.rolling(cfg.port_vol_window).std(ddof=1) * np.sqrt(cfg.trading_days)
    dyn_target = cfg.vol_target_ann * policy_daily["vol_mult"].reindex(idx).ffill().bfill()
    vol_scale = (dyn_target / realized.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(cfg.min_exposure, cfg.max_exposure)

    corr_secondary = pd.Series(1.0, index=idx, dtype=float)
    if cfg.use_corr_as_secondary_veto:
        hit = pre["corr_rho"].reindex(idx).fillna(0.0) >= cfg.corr_secondary_rho
        corr_secondary.loc[hit] = cfg.corr_secondary_scale

    gate_scale = policy_daily["gate_scale"].reindex(idx).ffill().bfill().clip(0.0, 1.0)
    exp_cap = policy_daily["exp_cap"].reindex(idx).ffill().bfill().clip(0.0, cfg.max_exposure)
    cap = (
        pre["crisis_scale"].reindex(idx).fillna(1.0)
        * pre["turb_scale"].reindex(idx).fillna(1.0)
        * corr_secondary
        * gate_scale
        * exp_cap
    ).clip(0.0, cfg.max_exposure)

    tgt_scale = pd.Series(np.minimum(vol_scale.values, cap.values), index=idx).clip(0.0, cfg.max_exposure)
    exec_scale = tgt_scale.shift(1).fillna(0.0)
    w_exec = w_exec_1x.mul(exec_scale, axis=0)
    turnover, tc = m6._costs(w_exec, costs)
    port_net = ((w_exec * rets).sum(axis=1) - tc).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    equity = cfg.capital_initial * (1.0 + port_net).cumprod()
    exposure = w_exec.abs().sum(axis=1).clip(0.0, cfg.max_exposure)

    qqq_r = pre["qqq"].pct_change().fillna(0.0) - costs.qqq_expense_ratio
    spy_r = pre["spy"].pct_change().fillna(0.0)
    qqq_eq = cfg.capital_initial * (1.0 + qqq_r).cumprod()
    spy_eq = cfg.capital_initial * (1.0 + spy_r).cumprod()
    return {
        "label": label,
        "returns_net": port_net,
        "equity": equity,
        "exposure": exposure,
        "turnover": turnover,
        "weights_scaled": w_exec,
        "total_scale": exec_scale,
        "total_scale_target": tgt_scale,
        "cap": cap,
        "vol_scale": vol_scale,
        "crisis_scale": pre["crisis_scale"],
        "turb_scale": pre["turb_scale"],
        "corr_rho": pre["corr_rho"],
        "corr_state": pre["corr_state"],
        "scores": alpha_cache["scores"],
        "alpha_mix": policy_daily["alpha_mix"].reindex(idx).ffill().bfill(),
        "stop_hits": alpha_cache["stop_hits"],
        "bench": {"QQQ_r": qqq_r, "QQQ_eq": qqq_eq, "SPY_r": spy_r, "SPY_eq": spy_eq},
    }


def _candidate_metrics(
    bt: Dict[str, Any],
    base_bt: Dict[str, Any],
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    policy_daily: pd.DataFrame,
    cfg: Mahoraga10Config,
) -> Dict[str, float]:
    r_new = bt["returns_net"].loc[val_start:val_end]
    r_base = base_bt["returns_net"].loc[val_start:val_end]
    eq_new = cfg.capital_initial * (1.0 + r_new).cumprod()
    eq_base = cfg.capital_initial * (1.0 + r_base).cumprod()
    s_new = m6.summarize(r_new, eq_new, bt["exposure"].loc[r_new.index], bt["turnover"].loc[r_new.index], cfg, "VAL_NEW")
    s_base = m6.summarize(r_base, eq_base, base_bt["exposure"].loc[r_base.index], base_bt["turnover"].loc[r_base.index], cfg, "VAL_BASE")
    diff_stats = summarize_diff_strategy(r_new, r_base, bt["exposure"].loc[r_new.index], policy_daily.loc[r_new.index, "is_intervening"])
    utility = (
        1.4 * (s_new["Sharpe"] - s_base["Sharpe"])
        + 0.20 * (s_new["CAGR"] - s_base["CAGR"])
        - 0.35 * max(0.0, abs(s_new["MaxDD"]) - abs(s_base["MaxDD"]))
        - 0.15 * max(0.0, 0.35 - diff_stats["avg_exp"])
        - 0.20 * max(0.0, diff_stats["intervention_rate"] - 0.45)
    )
    return {
        "val_sharpe": float(s_new["Sharpe"]),
        "val_cagr": float(s_new["CAGR"]),
        "val_maxdd": float(s_new["MaxDD"]),
        "base_val_sharpe": float(s_base["Sharpe"]),
        "base_val_cagr": float(s_base["CAGR"]),
        "base_val_maxdd": float(s_base["MaxDD"]),
        "val_pvalue": float(diff_stats["pvalue"]),
        "intervention_rate": float(diff_stats["intervention_rate"]),
        "avg_exp": float(diff_stats["avg_exp"]),
        "utility": float(utility),
    }


def _rank_candidates(rows: List[Dict[str, Any]], cfg: Mahoraga10Config) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df = df.sort_values(["utility", "val_sharpe", "val_cagr"], ascending=[False, False, False]).reset_index(drop=True)
    df["val_qvalue"] = bhy_qvalues(df["val_pvalue"].to_numpy(), alpha=cfg.bhy_alpha)
    return df


def _run_single_fold(
    fold_row: pd.Series,
    ohlcv: Dict[str, pd.DataFrame],
    cfg: Mahoraga10Config,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    fold_n = int(fold_row["fold"])
    train_start = pd.Timestamp(fold_row["train_start"])
    train_end = pd.Timestamp(fold_row["train_end"])
    test_start = pd.Timestamp(fold_row["test_start"])
    test_end = pd.Timestamp(fold_row["test_end"])

    cfg_fold = deepcopy(cfg)
    qqq_full = m6.to_s(ohlcv["close"][cfg.bench_qqq].ffill(), "QQQ")
    dd_thr, vol_thr = m6.calibrate_crisis_thresholds(qqq_full, str(train_start.date()), str(train_end.date()), cfg_fold)
    cfg_fold.crisis_dd_thr = dd_thr
    cfg_fold.crisis_vol_zscore_thr = vol_thr

    pre = _prepare_fold_invariants(ohlcv, cfg_fold, universe_schedule, train_start, train_end)

    # Baseline remains the unmodified 6.1 reference.
    base_cfg = deepcopy(cfg_fold)
    base_bt = m6.backtest(ohlcv, base_cfg, costs, label=f"FOLD{fold_n}_BASE", universe_schedule=universe_schedule)

    daily_ctx = build_daily_context(base_bt, ohlcv, cfg_fold, pre)
    weekly_df = build_weekly_dataset(daily_ctx, cfg_fold, train_end)
    train_weekly = weekly_df.loc[:train_end].copy()
    frag_fit = fit_best_classifier(train_weekly, "fragility_y", cfg_fold, cfg_fold.outer_parallel)
    recv_fit = fit_best_classifier(train_weekly, "recovery_y", cfg_fold, cfg_fold.outer_parallel)
    weekly_df["fragility_p"] = apply_classifier(frag_fit, weekly_df, "fragility_p")
    weekly_df["recovery_p"] = apply_classifier(recv_fit, weekly_df, "recovery_p")

    cut = int(len(train_weekly) * (1.0 - cfg_fold.inner_val_frac))
    cut = max(cfg_fold.min_train_weeks, cut)
    cut = min(cut, len(train_weekly) - 1)
    val_start = train_weekly.index[cut]

    alpha_caches: List[Dict[str, Any]] = []
    for alpha_params in iter_grid(cfg_fold.alpha_grid()):
        cache = precompute_alpha_path(pre, pre["components"], pre["alpha_fit"], alpha_params, cfg_fold)
        alpha_caches.append(cache)

    candidate_rows: List[Dict[str, Any]] = []
    for alpha_cache in alpha_caches:
        alpha_mix_base = float(alpha_cache["alpha_params"]["alpha_mix_base"])
        for policy_params in iter_policy_candidates(cfg_fold):
            policy_weekly = build_policy_weekly(weekly_df, alpha_mix_base, policy_params, cfg_fold)
            policy_daily = weekly_to_daily_policy(policy_weekly, pre["idx"])
            bt = _apply_policy_to_alpha_cache(pre, alpha_cache, policy_daily, cfg_fold, costs, label=f"FOLD{fold_n}_CAND")
            metrics = _candidate_metrics(bt, base_bt, val_start, train_end, policy_daily, cfg_fold)
            candidate_rows.append({**alpha_cache["alpha_params"], **policy_params, **metrics})

    calib_df = _rank_candidates(candidate_rows, cfg_fold)
    if len(calib_df) == 0:
        best_params = {k: float(v[0]) for k, v in cfg_fold.alpha_grid().items() | cfg_fold.policy_grid().items()}
        best_row = {"val_pvalue": 1.0, "val_qvalue": 1.0, "utility": 0.0}
    else:
        best_row = calib_df.iloc[0].to_dict()
        best_params = {k: float(best_row[k]) for k in list(cfg_fold.alpha_grid().keys()) + list(cfg_fold.policy_grid().keys())}

    best_alpha = precompute_alpha_path(pre, pre["components"], pre["alpha_fit"], {k: best_params[k] for k in cfg_fold.alpha_grid().keys()}, cfg_fold)
    best_policy_weekly = build_policy_weekly(weekly_df, best_params["alpha_mix_base"], {k: best_params[k] for k in cfg_fold.policy_grid().keys()}, cfg_fold)
    best_policy_daily = weekly_to_daily_policy(best_policy_weekly, pre["idx"])
    m10_bt = _apply_policy_to_alpha_cache(pre, best_alpha, best_policy_daily, cfg_fold, costs, label=f"FOLD{fold_n}_M10")

    base_r = base_bt["returns_net"].loc[test_start:test_end]
    m10_r = m10_bt["returns_net"].loc[test_start:test_end]
    base_eq = cfg_fold.capital_initial * (1.0 + base_r).cumprod()
    m10_eq = cfg_fold.capital_initial * (1.0 + m10_r).cumprod()
    s_base = m6.summarize(base_r, base_eq, base_bt["exposure"].loc[base_r.index], base_bt["turnover"].loc[base_r.index], cfg_fold, f"FOLD{fold_n}_BASE")
    s_m10 = m6.summarize(m10_r, m10_eq, m10_bt["exposure"].loc[m10_r.index], m10_bt["turnover"].loc[m10_r.index], cfg_fold, f"FOLD{fold_n}_M10")
    test_p = paired_ttest_pvalue(m10_r - base_r, alternative="greater")

    row = {
        "fold": fold_n,
        "train": f"{train_start.date()}→{train_end.date()}",
        "test": f"{test_start.date()}→{test_end.date()}",
        "BASE_CAGR%": round(s_base["CAGR"] * 100, 2),
        "BASE_Sharpe": round(s_base["Sharpe"], 4),
        "BASE_MaxDD%": round(s_base["MaxDD"] * 100, 2),
        "M10_CAGR%": round(s_m10["CAGR"] * 100, 2),
        "M10_Sharpe": round(s_m10["Sharpe"], 4),
        "M10_MaxDD%": round(s_m10["MaxDD"] * 100, 2),
        "FragilityModel": frag_fit.get("name", "neutral"),
        "RecoveryModel": recv_fit.get("name", "neutral"),
        "InterventionRate": round(float(best_policy_daily.loc[test_start:test_end, "is_intervening"].mean()), 4),
        "MeanGate": round(float(best_policy_daily.loc[test_start:test_end, "gate_scale"].mean()), 4),
        "MeanAlphaMix": round(float(best_policy_daily.loc[test_start:test_end, "alpha_mix"].mean()), 4),
        "Val_pvalue": round(float(best_row.get("val_pvalue", 1.0)), 6),
        "Val_qvalue": round(float(best_row.get("val_qvalue", 1.0)), 6),
        "Test_pvalue": round(float(test_p), 6),
    }
    return {
        "fold": fold_n,
        "fold_row": row,
        "base_bt": base_bt,
        "m10_bt": m10_bt,
        "weekly_df": weekly_df,
        "policy_weekly": best_policy_weekly,
        "policy_daily": best_policy_daily,
        "calibration_df": calib_df,
        "best_params": best_params,
        "frag_fit": frag_fit,
        "recv_fit": recv_fit,
    }


def stitch_oos_path(results: List[Dict[str, Any]], key: str) -> Dict[str, pd.Series]:
    rets, exp, to = [], [], []
    for r in sorted(results, key=lambda x: x["fold"]):
        bt = r[key]
        t0, t1 = [pd.Timestamp(x) for x in r["fold_row"]["test"].split("→")]
        rr = bt["returns_net"].loc[t0:t1]
        rets.append(rr)
        exp.append(bt["exposure"].loc[rr.index])
        to.append(bt["turnover"].loc[rr.index])
    r = pd.concat(rets).sort_index() if rets else pd.Series(dtype=float)
    e = pd.concat(exp).sort_index() if exp else pd.Series(dtype=float)
    t = pd.concat(to).sort_index() if to else pd.Series(dtype=float)
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return {"returns": r, "exposure": e, "turnover": t, "equity": eq}


def run_walk_forward_mahoraga10(ohlcv: Dict[str, pd.DataFrame], cfg: Mahoraga10Config, costs: m6.CostsConfig, universe_schedule: Optional[pd.DataFrame]) -> Dict[str, Any]:
    fold_df = _load_folds(cfg, ohlcv)
    jobs = [delayed(_run_single_fold)(row, ohlcv, cfg, costs, universe_schedule) for _, row in fold_df.iterrows()]
    if cfg.outer_parallel and len(jobs) > 1:
        results = Parallel(n_jobs=min(cfg.max_outer_jobs, len(jobs)), backend=cfg.outer_backend)(jobs)
    else:
        results = [_run_single_fold(row, ohlcv, cfg, costs, universe_schedule) for _, row in fold_df.iterrows()]
    results = sorted(results, key=lambda x: x["fold"])
    test_p = [r["fold_row"]["Test_pvalue"] for r in results]
    test_q = bhy_qvalues(test_p, alpha=cfg.bhy_alpha)
    for r, qv in zip(results, test_q):
        r["fold_row"]["Test_qvalue"] = round(float(qv), 6)
    return {
        "results": results,
        "stitched_base": stitch_oos_path(results, "base_bt"),
        "stitched_m10": stitch_oos_path(results, "m10_bt"),
    }
