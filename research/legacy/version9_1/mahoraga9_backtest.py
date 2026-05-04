from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import mahoraga6_1 as m6
from mahoraga9_alpha import build_score_bundle, fit_alpha_configs
from mahoraga9_calibration import candidate_metrics, iter_policy_candidates, rank_candidates
from mahoraga9_config import Mahoraga9Config
from mahoraga9_features import build_daily_context, build_weekly_context
from mahoraga9_hawkes import build_hawkes_signals
from mahoraga9_meta_labels import build_labels
from mahoraga9_meta_models import apply_model, fit_best_model
from mahoraga9_risk import build_policy_table, weekly_to_daily_policy
from mahoraga9_utils import bhy_qvalues, paired_ttest_pvalue, time_split_index


def _load_folds(cfg: Mahoraga9Config, ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    trading_idx = pd.DatetimeIndex(ohlcv["close"].index)
    folds = m6.build_contiguous_folds(cfg, trading_idx)
    df = pd.DataFrame(folds)
    return df[df["fold"].isin(cfg.mode_folds())].copy().sort_values("fold")


def _union_universe(ohlcv: Dict[str, pd.DataFrame], universe_schedule: Optional[pd.DataFrame], cfg: Mahoraga9Config) -> List[str]:
    if universe_schedule is not None and len(universe_schedule) > 0:
        members = set()
        for s in universe_schedule["members"]:
            members |= set(json.loads(s))
        return sorted([t for t in members if t in ohlcv["close"].columns])
    return sorted([t for t in cfg.universe_static if t in ohlcv["close"].columns])


def _prepare_invariants(ohlcv: Dict[str, pd.DataFrame], cfg: Mahoraga9Config, universe_schedule: Optional[pd.DataFrame]) -> Dict[str, Any]:
    univ_master = _union_universe(ohlcv, universe_schedule, cfg)
    close = ohlcv["close"][univ_master].copy()
    high = ohlcv["high"][univ_master].copy()
    low = ohlcv["low"][univ_master].copy()
    idx = close.index
    rets = close.pct_change().fillna(0.0)
    qqq = m6.to_s(ohlcv["close"][cfg.bench_qqq].reindex(idx).ffill(), "QQQ")
    spy = m6.to_s(ohlcv["close"][cfg.bench_spy].reindex(idx).ffill(), "SPY")
    crisis_scale, crisis_state = m6.compute_crisis_gate(qqq, cfg)
    turb_scale = m6.compute_turbulence(close, ohlcv["volume"][univ_master], qqq, cfg)
    vix_series = None
    if cfg.bench_vix in ohlcv.get("close", pd.DataFrame()).columns:
        vix_series = m6.to_s(ohlcv["close"][cfg.bench_vix].reindex(idx).ffill(), "VIX")
    corr_rho, corr_scale_legacy, corr_state = m6.compute_corr_shield_series(
        rets, idx, cfg, univ_master, universe_schedule is not None and len(universe_schedule) > 0,
        universe_schedule=universe_schedule, vix=vix_series
    )
    reb_dates = set(close.resample(cfg.rebalance_freq).last().index)
    return {
        "univ_master": univ_master,
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
        "universe_schedule": universe_schedule,
    }


def _members_at_date(universe_schedule: Optional[pd.DataFrame], dt: pd.Timestamp, univ_master: List[str]) -> List[str]:
    if universe_schedule is None or len(universe_schedule) == 0:
        return univ_master
    members = m6.get_universe_at_date(universe_schedule, dt)
    return [t for t in members if t in univ_master]


def _custom_backtest(
    pre: Dict[str, Any],
    raw_scores: pd.DataFrame,
    resid_scores: pd.DataFrame,
    alpha_mix_daily: pd.Series,
    policy_daily: pd.DataFrame,
    cfg: Mahoraga9Config,
    costs: m6.CostsConfig,
    label: str,
) -> Dict[str, Any]:
    idx = pre["idx"]
    close = pre["close"]
    high = pre["high"]
    low = pre["low"]
    rets = pre["rets"]
    univ_master = pre["univ_master"]
    reb_dates = pre["reb_dates"]

    alpha_mix = alpha_mix_daily.reindex(idx).ffill().bfill().clip(cfg.alpha_mix_min, cfg.alpha_mix_max)
    mix_arr = alpha_mix.to_numpy(dtype=float).reshape(-1, 1)
    mixed_scores = raw_scores.reindex(idx, columns=univ_master).fillna(0.0).to_numpy(dtype=float) * (1.0 - mix_arr) + resid_scores.reindex(idx, columns=univ_master).fillna(0.0).to_numpy(dtype=float) * mix_arr
    mixed_scores_df = pd.DataFrame(mixed_scores, index=idx, columns=univ_master)

    w = pd.DataFrame(0.0, index=idx, columns=univ_master)
    last_w = pd.Series(0.0, index=univ_master)
    for dt in idx:
        if dt in reb_dates:
            members = _members_at_date(pre["universe_schedule"], dt, univ_master)
            if not members:
                last_w = pd.Series(0.0, index=univ_master)
            else:
                row = mixed_scores_df.loc[dt, members]
                sel_names = row.nlargest(cfg.top_k).index.tolist()
                names = [n for n in sel_names if row.get(n, 0.0) > 0.0]
                if not names:
                    last_w = pd.Series(0.0, index=univ_master)
                elif len(names) == 1:
                    last_w = pd.Series(0.0, index=univ_master)
                    last_w[names[0]] = 1.0
                else:
                    lb = rets.loc[:dt, names].tail(cfg.hrp_window).dropna(how="any")
                    if len(lb) < 60:
                        lb = rets.loc[:dt, names].dropna(how="any")
                    ww = m6.hrp_weights(lb).reindex(names, fill_value=0.0) if len(lb) else pd.Series(1.0 / len(names), index=names)
                    if ww.sum() > 0:
                        ww = ww.clip(upper=cfg.weight_cap)
                        ww = ww / ww.sum()
                    last_w = pd.Series(0.0, index=univ_master)
                    last_w[names] = ww.reindex(names, fill_value=0.0).values
        w.loc[dt] = last_w.values

    w_stop, stop_hits = m6.apply_chandelier(w, close, high, low, cfg)
    w_exec_1x = w_stop.shift(1).fillna(0.0)
    gross_1x = (w_exec_1x * rets).sum(axis=1)

    realized = gross_1x.rolling(cfg.port_vol_window).std(ddof=1) * np.sqrt(cfg.trading_days)
    dyn_target = cfg.vol_target_ann * policy_daily["vol_mult"].reindex(idx).ffill().bfill().clip(0.5, 1.2)
    vol_scale = (dyn_target / realized.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(cfg.min_exposure, cfg.max_exposure)

    corr_secondary = pd.Series(1.0, index=idx, dtype=float)
    if cfg.use_corr_as_secondary_veto:
        corr_hit = pre["corr_rho"].reindex(idx).fillna(0.0) >= cfg.corr_secondary_rho
        if cfg.corr_use_vix_confirm and cfg.bench_vix in close.columns:
            pass
        corr_secondary.loc[corr_hit] = cfg.corr_secondary_scale

    gate_scale = policy_daily["gate_scale"].reindex(idx).ffill().bfill().clip(0.0, 1.0)
    exp_cap = policy_daily["exp_cap"].reindex(idx).ffill().bfill().clip(0.0, cfg.max_exposure)
    cap = (pre["crisis_scale"].reindex(idx).fillna(1.0) * pre["turb_scale"].reindex(idx).fillna(1.0) * corr_secondary * gate_scale * exp_cap).clip(0.0, cfg.max_exposure)
    tgt_sc = pd.Series(np.minimum(vol_scale.values, cap.values), index=idx).clip(0.0, cfg.max_exposure)
    exec_sc = tgt_sc.shift(1).fillna(0.0)
    w_exec = w_exec_1x.mul(exec_sc, axis=0)

    to, tc = m6._costs(w_exec, costs)
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
        "turnover": to,
        "weights_scaled": w_exec,
        "total_scale": exec_sc,
        "total_scale_target": tgt_sc,
        "cap": cap,
        "turb_scale": pre["turb_scale"],
        "crisis_scale": pre["crisis_scale"],
        "crisis_state": pre["crisis_state"],
        "vol_scale": vol_scale,
        "external_scale": gate_scale,
        "corr_scale": corr_secondary,
        "corr_rho": pre["corr_rho"],
        "corr_state": pre["corr_state"],
        "stop_hits": stop_hits,
        "scores": mixed_scores_df,
        "alpha_mix": alpha_mix,
        "bench": {"QQQ_r": qqq_r, "QQQ_eq": qqq_eq, "SPY_r": spy_r, "SPY_eq": spy_eq},
    }


def _build_weekly_dataset(base_bt: Dict[str, Any], ohlcv: Dict[str, pd.DataFrame], cfg: Mahoraga9Config, train_end: pd.Timestamp) -> pd.DataFrame:
    daily_ctx = build_daily_context(base_bt, ohlcv, cfg)
    weekly_ctx = build_weekly_context(daily_ctx, cfg)
    train_index = weekly_ctx.loc[:train_end].index
    hawkes = build_hawkes_signals(weekly_ctx, cfg, train_index)
    labels = build_labels(weekly_ctx.join(hawkes), cfg, train_index)
    return weekly_ctx.join(hawkes).join(labels)


def _run_single_fold_v9(fold_row: pd.Series, ohlcv: Dict[str, pd.DataFrame], cfg: Mahoraga9Config, costs: m6.CostsConfig, universe_schedule: Optional[pd.DataFrame]) -> Dict[str, Any]:
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

    alpha_cfgs = fit_alpha_configs(ohlcv, cfg_fold, train_start, train_end)
    base_cfg = alpha_cfgs["raw"]
    base_bt = m6.backtest(ohlcv, base_cfg, costs, label=f"FOLD{fold_n}_BASE", universe_schedule=universe_schedule)

    score_bundle = build_score_bundle(ohlcv, alpha_cfgs["raw"], alpha_cfgs["resid"])
    pre = _prepare_invariants(ohlcv, base_cfg, universe_schedule)

    weekly_df = _build_weekly_dataset(base_bt, ohlcv, cfg_fold, train_end)
    train_weekly = weekly_df.loc[:train_end].copy()
    frag_fit = fit_best_model(train_weekly, "fragility_y", cfg_fold)
    recv_fit = fit_best_model(train_weekly, "recovery_y", cfg_fold)
    weekly_df["fragility_p"] = apply_model(frag_fit, weekly_df, "fragility_p")
    weekly_df["recovery_p"] = apply_model(recv_fit, weekly_df, "recovery_p")

    train_only = weekly_df.loc[:train_end].copy()
    cut = time_split_index(train_only.index, cfg_fold.inner_val_frac, cfg_fold.min_train_weeks)
    val_start = train_only.index[cut]

    candidate_rows: List[Dict[str, Any]] = []
    for params in iter_policy_candidates(cfg_fold):
        pol_weekly = build_policy_table(weekly_df, params, cfg_fold)
        pol_daily = weekly_to_daily_policy(pol_weekly, pre["idx"])
        bt = _custom_backtest(
            pre,
            score_bundle["raw_scores"],
            score_bundle["resid_scores"],
            pol_daily["alpha_mix"],
            pol_daily,
            cfg_fold,
            costs,
            label=f"FOLD{fold_n}_CAND",
        )
        val_r = bt["returns_net"].loc[val_start:train_end]
        base_val_r = base_bt["returns_net"].loc[val_start:train_end]
        metrics = candidate_metrics(val_r, base_val_r, bt["exposure"].loc[val_r.index], pol_daily.loc[val_r.index], cfg_fold, fold_n)
        candidate_rows.append({**params, **metrics})

    calib_df = rank_candidates(candidate_rows, cfg_fold, fold_n)
    if len(calib_df) == 0:
        best_params = {k: float(v[0]) for k, v in cfg_fold.candidate_grid().items()}
        best_row = {"val_pvalue": 1.0, "val_qvalue": 1.0, "utility": 0.0}
    else:
        best_row = calib_df.iloc[0].to_dict()
        best_params = {k: float(best_row[k]) for k in cfg_fold.candidate_grid().keys()}

    policy_weekly = build_policy_table(weekly_df, best_params, cfg_fold)
    policy_daily = weekly_to_daily_policy(policy_weekly, pre["idx"])
    v9_bt = _custom_backtest(
        pre,
        score_bundle["raw_scores"],
        score_bundle["resid_scores"],
        policy_daily["alpha_mix"],
        policy_daily,
        cfg_fold,
        costs,
        label=f"FOLD{fold_n}_V9_1",
    )

    base_r = base_bt["returns_net"].loc[test_start:test_end]
    v9_r = v9_bt["returns_net"].loc[test_start:test_end]
    base_eq = cfg_fold.capital_initial * (1.0 + base_r).cumprod()
    v9_eq = cfg_fold.capital_initial * (1.0 + v9_r).cumprod()
    s_base = m6.summarize(base_r, base_eq, base_bt["exposure"].loc[base_r.index], base_bt["turnover"].loc[base_r.index], cfg_fold, f"FOLD{fold_n}_BASE")
    s_v9 = m6.summarize(v9_r, v9_eq, v9_bt["exposure"].loc[v9_r.index], v9_bt["turnover"].loc[v9_r.index], cfg_fold, f"FOLD{fold_n}_V9_1")
    test_p = paired_ttest_pvalue(v9_r - base_r, alternative="greater")

    row = {
        "fold": fold_n,
        "train": f"{train_start.date()}→{train_end.date()}",
        "test": f"{test_start.date()}→{test_end.date()}",
        "BASE_CAGR%": round(s_base["CAGR"] * 100, 2),
        "BASE_Sharpe": round(s_base["Sharpe"], 4),
        "BASE_MaxDD%": round(s_base["MaxDD"] * 100, 2),
        "V9_CAGR%": round(s_v9["CAGR"] * 100, 2),
        "V9_Sharpe": round(s_v9["Sharpe"], 4),
        "V9_MaxDD%": round(s_v9["MaxDD"] * 100, 2),
        "FragilityModel": frag_fit.get("name", "neutral"),
        "RecoveryModel": recv_fit.get("name", "neutral"),
        "InterventionRate": round(float(policy_daily.loc[test_start:test_end, "is_intervening"].mean()), 4),
        "MeanGate": round(float(policy_daily.loc[test_start:test_end, "gate_scale"].mean()), 4),
        "MeanAlphaMix": round(float(policy_daily.loc[test_start:test_end, "alpha_mix"].mean()), 4),
        "Val_pvalue": round(float(best_row.get("val_pvalue", 1.0)), 6),
        "Val_qvalue": round(float(best_row.get("val_qvalue", 1.0)), 6),
        "Test_pvalue": round(float(test_p), 6),
    }
    return {
        "fold_row": row,
        "fold": fold_n,
        "base_bt": base_bt,
        "v9_bt": v9_bt,
        "weekly_df": weekly_df,
        "policy_weekly": policy_weekly,
        "policy_daily": policy_daily,
        "calibration_df": calib_df,
        "best_policy": best_params,
        "frag_fit": frag_fit,
        "recv_fit": recv_fit,
    }


def stitch_oos_path(results: List[Dict[str, Any]], key: str) -> Dict[str, pd.Series]:
    rets, exp, to = [], [], []
    for r in sorted(results, key=lambda x: x["fold"]):
        bt = r[key]
        row = r["fold_row"]
        test = row["test"].split("→")
        t0, t1 = pd.Timestamp(test[0]), pd.Timestamp(test[1])
        rr = bt["returns_net"].loc[t0:t1]
        rets.append(rr)
        exp.append(bt["exposure"].loc[rr.index])
        to.append(bt["turnover"].loc[rr.index])
    r = pd.concat(rets).sort_index() if rets else pd.Series(dtype=float)
    e = pd.concat(exp).sort_index() if exp else pd.Series(dtype=float)
    t = pd.concat(to).sort_index() if to else pd.Series(dtype=float)
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return {"returns": r, "exposure": e, "turnover": t, "equity": eq}


def run_walk_forward_v9(ohlcv: Dict[str, pd.DataFrame], cfg: Mahoraga9Config, costs: m6.CostsConfig, universe_schedule: Optional[pd.DataFrame]) -> Dict[str, Any]:
    fold_df = _load_folds(cfg, ohlcv)
    jobs = [delayed(_run_single_fold_v9)(row, ohlcv, cfg, costs, universe_schedule) for _, row in fold_df.iterrows()]
    if cfg.outer_parallel and len(jobs) > 1:
        results = Parallel(n_jobs=min(cfg.max_outer_jobs, len(jobs)), backend=cfg.outer_backend)(jobs)
    else:
        results = [_run_single_fold_v9(row, ohlcv, cfg, costs, universe_schedule) for _, row in fold_df.iterrows()]
    results = sorted(results, key=lambda x: x["fold"])
    test_p = [r["fold_row"]["Test_pvalue"] for r in results]
    test_q = bhy_qvalues(test_p)
    for r, qv in zip(results, test_q):
        r["fold_row"]["Test_qvalue"] = round(float(qv), 6)
    return {
        "results": results,
        "stitched_base": stitch_oos_path(results, "base_bt"),
        "stitched_v9": stitch_oos_path(results, "v9_bt"),
    }
