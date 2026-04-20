from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import mahoraga6_1 as m6
from mahoraga11_alpha import build_alpha_components, fit_alpha_model, precompute_engine_path
from mahoraga11_config import Mahoraga11Config
from mahoraga11_router import (
    build_daily_context,
    build_weekly_dataset,
    build_router_weekly,
    fit_best_classifier,
    apply_classifier,
    iter_router_candidates,
    weekly_to_daily_router,
)
from mahoraga11_universe import members_at_date, union_universe
from mahoraga11_utils import bhy_qvalues, ensure_dir, iter_grid, paired_ttest_pvalue


def _load_folds(cfg: Mahoraga11Config, ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    folds = pd.DataFrame(m6.build_contiguous_folds(cfg, pd.DatetimeIndex(ohlcv["close"].index)))
    folds = folds[folds["fold"].isin(cfg.mode_folds())].copy().sort_values("fold")
    return folds


def _prepare_fold_cfg(cfg_base: Mahoraga11Config, ohlcv: Dict[str, pd.DataFrame], train_start: pd.Timestamp, train_end: pd.Timestamp, universe_schedule: Optional[pd.DataFrame]) -> Mahoraga11Config:
    cfg = deepcopy(cfg_base)
    qqq_full = m6.to_s(ohlcv["close"][cfg.bench_qqq].ffill(), "QQQ")
    dd_thr, vol_thr = m6.calibrate_crisis_thresholds(qqq_full, str(train_start.date()), str(train_end.date()), cfg)
    cfg.crisis_dd_thr = dd_thr
    cfg.crisis_vol_zscore_thr = vol_thr
    final_train_tickers = m6.get_training_universe(str(train_end.date()), universe_schedule, cfg.universe_static, list(ohlcv["close"].columns))
    close_univ = ohlcv["close"][final_train_tickers]
    wt, wm, wr = m6.fit_ic_weights(close_univ, qqq_full.loc[train_start:train_end], cfg, str(train_start.date()), str(train_end.date()))
    cfg.w_trend, cfg.w_mom, cfg.w_rel = wt, wm, wr
    return cfg


def _prepare_fold_invariants(ohlcv: Dict[str, pd.DataFrame], cfg: Mahoraga11Config, universe_schedule: Optional[pd.DataFrame], train_start: pd.Timestamp, train_end: pd.Timestamp) -> Dict[str, Any]:
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


def _build_baseline_bt(pre: Dict[str, Any], cfg: Mahoraga11Config, costs: m6.CostsConfig) -> Dict[str, Any]:
    score = m6.compute_scores(pre["close"], pre["qqq"], cfg)
    close = pre["close"]
    high = pre["high"]
    low = pre["low"]
    rets = pre["rets"]
    idx = pre["idx"]

    w = pd.DataFrame(0.0, index=idx, columns=close.columns)
    last_w = pd.Series(0.0, index=close.columns)
    for dt in idx:
        if dt in pre["reb_dates"]:
            members = pre["members_at"](dt)
            members = [m for m in members if m in close.columns]
            if members:
                row = score.loc[dt, members]
                chosen = [n for n in row.nlargest(cfg.top_k).index.tolist() if row.get(n, 0.0) > 0.0]
                if len(chosen) == 1:
                    last_w = pd.Series(0.0, index=close.columns)
                    last_w[chosen[0]] = 1.0
                elif len(chosen) >= 2:
                    hist = rets.loc[:dt, chosen].tail(cfg.hrp_window).dropna(how="any")
                    if len(hist) < 60:
                        hist = rets.loc[:dt, chosen].dropna(how="any")
                    ww = m6.hrp_weights(hist).reindex(chosen, fill_value=0.0) if len(hist) else pd.Series(1.0 / len(chosen), index=chosen)
                    ww = ww.clip(upper=cfg.weight_cap)
                    ww = ww / ww.sum() if ww.sum() > 0 else pd.Series(1.0 / len(chosen), index=chosen)
                    last_w = pd.Series(0.0, index=close.columns)
                    last_w[chosen] = ww.reindex(chosen).values
                else:
                    last_w = pd.Series(0.0, index=close.columns)
            else:
                last_w = pd.Series(0.0, index=close.columns)
        w.loc[dt] = last_w.values

    w_stop, _ = m6.apply_chandelier(w, close, high, low, cfg)
    w_exec_1x = w_stop.shift(1).fillna(0.0)
    gross_1x = (w_exec_1x * rets).sum(axis=1)
    vol_scale = m6.vol_target_scale(gross_1x, cfg)

    corr_secondary = pd.Series(1.0, index=idx, dtype=float)
    if cfg.use_corr_as_secondary_veto:
        hit = pre["corr_rho"].reindex(idx).fillna(0.0) >= cfg.corr_secondary_rho
        corr_secondary.loc[hit] = cfg.corr_secondary_scale
    cap = (pre["crisis_scale"].reindex(idx).fillna(1.0) * pre["turb_scale"].reindex(idx).fillna(1.0) * corr_secondary).clip(0.0, cfg.max_exposure)
    tgt_scale = pd.Series(np.minimum(vol_scale.values, cap.values), index=idx).clip(0.0, cfg.max_exposure)
    exec_scale = tgt_scale.shift(1).fillna(0.0)
    w_exec = w_exec_1x.mul(exec_scale, axis=0)
    turnover, tc = m6._costs(w_exec, costs)
    port_net = ((w_exec * rets).sum(axis=1) - tc).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    equity = cfg.capital_initial * (1.0 + port_net).cumprod()
    exposure = w_exec.abs().sum(axis=1).clip(0.0, cfg.max_exposure)
    return {
        "label": "BASELINE",
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
        "scores": score,
        "bench": {
            "QQQ_r": pre["qqq"].pct_change().fillna(0.0) - costs.qqq_expense_ratio,
            "QQQ_eq": cfg.capital_initial * (1.0 + pre["qqq"].pct_change().fillna(0.0) - costs.qqq_expense_ratio).cumprod(),
            "SPY_r": pre["spy"].pct_change().fillna(0.0),
            "SPY_eq": cfg.capital_initial * (1.0 + pre["spy"].pct_change().fillna(0.0)).cumprod(),
        },
    }


def _apply_router_to_engine_cache(pre: Dict[str, Any], ceiling_cache: Dict[str, Any], floor_cache: Dict[str, Any], router_daily: pd.DataFrame, cfg: Mahoraga11Config, costs: m6.CostsConfig, label: str) -> Dict[str, Any]:
    idx = pre["idx"]
    rets = pre["rets"]
    floor_blend = router_daily["floor_blend"].reindex(idx).ffill().bfill().clip(0.0, 1.0)

    w_exec_1x = ceiling_cache["weights_exec_1x"].mul(1.0 - floor_blend, axis=0) + floor_cache["weights_exec_1x"].mul(floor_blend, axis=0)
    gross_1x = (w_exec_1x * rets).sum(axis=1)

    realized = gross_1x.rolling(cfg.port_vol_window).std(ddof=1) * np.sqrt(cfg.trading_days)
    dyn_target = cfg.vol_target_ann * router_daily["vol_mult"].reindex(idx).ffill().bfill()
    vol_scale = (dyn_target / realized.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(cfg.min_exposure, cfg.max_exposure)

    corr_secondary = pd.Series(1.0, index=idx, dtype=float)
    if cfg.use_corr_as_secondary_veto:
        hit = pre["corr_rho"].reindex(idx).fillna(0.0) >= cfg.corr_secondary_rho
        corr_secondary.loc[hit] = cfg.corr_secondary_scale

    gate_scale = router_daily["gate_scale"].reindex(idx).ffill().bfill().clip(0.0, 1.0)
    exp_cap = router_daily["exp_cap"].reindex(idx).ffill().bfill().clip(0.0, cfg.max_exposure)
    cap = (pre["crisis_scale"].reindex(idx).fillna(1.0) * pre["turb_scale"].reindex(idx).fillna(1.0) * corr_secondary * gate_scale * exp_cap).clip(0.0, cfg.max_exposure)

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
        "scores": {"ceiling": ceiling_cache["scores"], "floor": floor_cache["scores"]},
        "floor_blend": floor_blend,
        "router_mode": router_daily["mode"].reindex(idx).ffill().bfill(),
        "bench": {"QQQ_r": qqq_r, "QQQ_eq": qqq_eq, "SPY_r": spy_r, "SPY_eq": spy_eq},
    }


def _candidate_metrics(bt: Dict[str, Any], base_bt: Dict[str, Any], val_start: pd.Timestamp, val_end: pd.Timestamp, router_daily: pd.DataFrame, cfg: Mahoraga11Config) -> Dict[str, float]:
    r_new = bt["returns_net"].loc[val_start:val_end]
    r_base = base_bt["returns_net"].loc[val_start:val_end]
    eq_new = cfg.capital_initial * (1.0 + r_new).cumprod()
    eq_base = cfg.capital_initial * (1.0 + r_base).cumprod()
    s_new = m6.summarize(r_new, eq_new, bt["exposure"].loc[r_new.index], bt["turnover"].loc[r_new.index], cfg, "VAL_NEW")
    s_base = m6.summarize(r_base, eq_base, base_bt["exposure"].loc[r_base.index], base_bt["turnover"].loc[r_base.index], cfg, "VAL_BASE")
    pval = paired_ttest_pvalue(r_new - r_base, alternative="greater")
    intervention = float(router_daily["is_intervening"].reindex(r_new.index).fillna(0.0).mean())
    floor_blend = float(router_daily["floor_blend"].reindex(r_new.index).fillna(0.0).mean())
    utility = (
        0.55 * (s_new["Sharpe"] - s_base["Sharpe"])
        + 0.18 * (s_new["CAGR"] - s_base["CAGR"])
        + 0.20 * (abs(s_base["MaxDD"]) - abs(s_new["MaxDD"]))
        - 0.10 * max(0.0, intervention - 0.35)
        - 0.06 * max(0.0, floor_blend - 0.60)
    )
    return {
        "utility": float(utility),
        "val_sharpe": float(s_new["Sharpe"]),
        "val_cagr": float(s_new["CAGR"]),
        "val_maxdd": float(s_new["MaxDD"]),
        "val_pvalue": float(pval),
        "intervention_rate": intervention,
        "mean_floor_blend": floor_blend,
    }


def _stitch(results: List[Dict[str, Any]], key: str) -> Dict[str, pd.Series]:
    bt_list = [r[key] for r in results]
    ret = pd.concat([b["returns_net"] for b in bt_list]).sort_index()
    exp = pd.concat([b["exposure"] for b in bt_list]).sort_index().reindex(ret.index).fillna(0.0)
    to = pd.concat([b["turnover"] for b in bt_list]).sort_index().reindex(ret.index).fillna(0.0)
    return {"returns": ret, "exposure": exp, "turnover": to}


def _run_single_fold(row: pd.Series, ohlcv: Dict[str, pd.DataFrame], cfg_base: Mahoraga11Config, costs: m6.CostsConfig, universe_schedule: Optional[pd.DataFrame]) -> Dict[str, Any]:
    fold_n = int(row["fold"])
    train_start = pd.Timestamp(row["train_start"])
    train_end = pd.Timestamp(row["train_end"])
    test_start = pd.Timestamp(row["test_start"])
    test_end = pd.Timestamp(row["test_end"])

    cfg_fold = _prepare_fold_cfg(cfg_base, ohlcv, train_start, train_end, universe_schedule)
    pre = _prepare_fold_invariants(ohlcv, cfg_fold, universe_schedule, train_start, train_end)
    base_bt = _build_baseline_bt(pre, cfg_fold, costs)

    daily_ctx = build_daily_context(base_bt, ohlcv, cfg_fold, pre)
    weekly_df = build_weekly_dataset(daily_ctx, cfg_fold, train_end)

    train_weekly = weekly_df.loc[:train_end].copy()
    structural_fit = fit_best_classifier(train_weekly, "structural_y", cfg_fold, cfg_fold.outer_parallel)
    fast_fit = fit_best_classifier(train_weekly, "fast_fragility_y", cfg_fold, cfg_fold.outer_parallel)
    recovery_fit = fit_best_classifier(train_weekly, "recovery_y", cfg_fold, cfg_fold.outer_parallel)

    weekly_df["structural_p"] = apply_classifier(structural_fit, weekly_df, "structural_p")
    weekly_df["fast_fragility_p"] = apply_classifier(fast_fit, weekly_df, "fast_fragility_p")
    weekly_df["recovery_p"] = apply_classifier(recovery_fit, weekly_df, "recovery_p")

    # Precompute engine paths once per unique engine candidate.
    engine_cands = list(iter_grid(cfg_fold.engine_grid()))
    engine_cache: Dict[Tuple[float, float, float], Dict[str, Any]] = {}
    for eng in engine_cands:
        for mix, beta_pen in [(eng["ceiling_mix"], eng["ceiling_beta_penalty"]), (eng["floor_mix"], eng["floor_beta_penalty"])]:
            key = (round(mix, 6), round(beta_pen, 6), round(eng["raw_rel_boost"], 6))
            if key not in engine_cache:
                engine_cache[key] = precompute_engine_path(pre, pre["components"], pre["alpha_fit"], mix, beta_pen, eng["raw_rel_boost"], cfg_fold)

    cut = int(round(len(train_weekly) * (1.0 - cfg_fold.inner_val_frac)))
    cut = max(cfg_fold.min_train_weeks, cut)
    cut = min(cut, len(train_weekly) - 1)
    val_start = train_weekly.index[cut]
    val_end = train_end

    leaderboard = []
    for eng in engine_cands:
        ceil_key = (round(eng["ceiling_mix"], 6), round(eng["ceiling_beta_penalty"], 6), round(eng["raw_rel_boost"], 6))
        floor_key = (round(eng["floor_mix"], 6), round(eng["floor_beta_penalty"], 6), round(eng["raw_rel_boost"], 6))
        ceil_cache = engine_cache[ceil_key]
        floor_cache = engine_cache[floor_key]

        for router_params in iter_router_candidates(cfg_fold):
            router_weekly = build_router_weekly(weekly_df.loc[:train_end], router_params, cfg_fold)
            router_daily = weekly_to_daily_router(router_weekly, pre["idx"])
            bt = _apply_router_to_engine_cache(pre, ceil_cache, floor_cache, router_daily, cfg_fold, costs, label="M11_VAL")
            met = _candidate_metrics(bt, base_bt, val_start, val_end, router_daily, cfg_fold)
            leaderboard.append({**eng, **router_params, **met})

    calib_df = pd.DataFrame(leaderboard).sort_values(["utility", "val_sharpe"], ascending=[False, False]).reset_index(drop=True)
    calib_df["val_qvalue"] = bhy_qvalues(calib_df["val_pvalue"].values, alpha=cfg_fold.bhy_alpha) if len(calib_df) else []
    best = calib_df.iloc[0].to_dict() if len(calib_df) else {
        "ceiling_mix": 0.10,
        "floor_mix": 0.50,
        "ceiling_beta_penalty": 0.00,
        "floor_beta_penalty": 0.06,
        "raw_rel_boost": 1.00,
        "structural_prob_thr": 0.65,
        "fast_prob_thr": 0.65,
        "recovery_prob_thr": 0.60,
        "hawkes_weight": 0.20,
        "floor_blend_max": 0.65,
        "gate_floor": 0.80,
        "vol_mult_stress": 0.88,
        "vol_mult_recovery": 1.05,
        "max_exp_stress": 0.80,
        "max_exp_recovery": 1.05,
    }

    ceil_key = (round(float(best["ceiling_mix"]), 6), round(float(best["ceiling_beta_penalty"]), 6), round(float(best["raw_rel_boost"]), 6))
    floor_key = (round(float(best["floor_mix"]), 6), round(float(best["floor_beta_penalty"]), 6), round(float(best["raw_rel_boost"]), 6))
    ceil_cache = engine_cache[ceil_key]
    floor_cache = engine_cache[floor_key]

    router_params = {k: float(best[k]) for k in cfg_fold.router_grid().keys()}
    router_weekly = build_router_weekly(weekly_df.loc[test_start:test_end], router_params, cfg_fold)
    router_daily = weekly_to_daily_router(router_weekly, pre["idx"])
    m11_bt = _apply_router_to_engine_cache(pre, ceil_cache, floor_cache, router_daily, cfg_fold, costs, label=f"M11_{fold_n}")

    rb = base_bt["returns_net"].loc[test_start:test_end]
    qb = cfg_fold.capital_initial * (1.0 + rb).cumprod()
    eb = base_bt["exposure"].loc[test_start:test_end]
    tb = base_bt["turnover"].loc[test_start:test_end]
    sb = m6.summarize(rb, qb, eb, tb, cfg_fold, f"BASE_{fold_n}")

    rm = m11_bt["returns_net"].loc[test_start:test_end]
    qm = cfg_fold.capital_initial * (1.0 + rm).cumprod()
    em = m11_bt["exposure"].loc[test_start:test_end]
    tm = m11_bt["turnover"].loc[test_start:test_end]
    sm = m6.summarize(rm, qm, em, tm, cfg_fold, f"M11_{fold_n}")
    test_p = paired_ttest_pvalue(rm - rb, alternative="greater")

    fold_row = {
        "fold": fold_n,
        "train": f"{train_start.date()}→{train_end.date()}",
        "test": f"{test_start.date()}→{test_end.date()}",
        "BASE_CAGR%": round(sb["CAGR"] * 100, 2),
        "BASE_Sharpe": round(sb["Sharpe"], 4),
        "BASE_MaxDD%": round(sb["MaxDD"] * 100, 2),
        "M11_CAGR%": round(sm["CAGR"] * 100, 2),
        "M11_Sharpe": round(sm["Sharpe"], 4),
        "M11_MaxDD%": round(sm["MaxDD"] * 100, 2),
        "StructuralModel": structural_fit["name"],
        "FastFragilityModel": fast_fit["name"],
        "RecoveryModel": recovery_fit["name"],
        "InterventionRate": round(float(router_daily["is_intervening"].loc[test_start:test_end].mean()), 4),
        "MeanGate": round(float(router_daily["gate_scale"].loc[test_start:test_end].mean()), 4),
        "MeanFloorBlend": round(float(router_daily["floor_blend"].loc[test_start:test_end].mean()), 4),
        "Val_pvalue": round(float(best.get("val_pvalue", 1.0)), 6),
        "Val_qvalue": round(float(best.get("val_qvalue", 1.0)), 6),
        "Test_pvalue": round(float(test_p), 6),
        "Test_qvalue": 1.0,
    }

    return {
        "fold": fold_n,
        "base_bt": base_bt,
        "m11_bt": m11_bt,
        "fold_row": fold_row,
        "calibration_df": calib_df,
        "router_weekly": router_weekly.assign(fold=fold_n),
        "best_params": best,
    }


def run_walk_forward_mahoraga11(ohlcv: Dict[str, pd.DataFrame], cfg: Mahoraga11Config, costs: m6.CostsConfig, universe_schedule: Optional[pd.DataFrame]) -> Dict[str, Any]:
    folds = _load_folds(cfg, ohlcv)
    tasks = [row for _, row in folds.iterrows()]
    use_parallel = cfg.outer_parallel and len(tasks) > 1
    if use_parallel:
        results = Parallel(n_jobs=min(cfg.max_outer_jobs, len(tasks)), backend=cfg.outer_backend)(
            delayed(_run_single_fold)(row, ohlcv, cfg, costs, universe_schedule) for row in tasks
        )
    else:
        results = [_run_single_fold(row, ohlcv, cfg, costs, universe_schedule) for row in tasks]

    results = sorted(results, key=lambda x: x["fold"])
    fold_df = pd.DataFrame([r["fold_row"] for r in results]).sort_values("fold")
    if len(fold_df):
        fold_df["Test_qvalue"] = bhy_qvalues(fold_df["Test_pvalue"].values, alpha=cfg.bhy_alpha)

    stitched_base = _stitch(results, "base_bt")
    stitched_m11 = _stitch(results, "m11_bt")
    return {
        "results": results,
        "fold_df": fold_df,
        "stitched_base": stitched_base,
        "stitched_m11": stitched_m11,
    }
