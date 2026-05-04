from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import mahoraga6_1 as m6
from mahoraga9_alpha import fit_alpha_configs
from mahoraga9_calibration import calibrate_policy
from mahoraga9_config import Mahoraga9Config
from mahoraga9_features import build_daily_context, build_weekly_context
from mahoraga9_hawkes import build_hawkes_signals
from mahoraga9_meta_labels import build_labels
from mahoraga9_meta_models import apply_model, fit_best_model
from mahoraga9_risk import build_policy_table, weekly_to_daily_gate


def _load_folds(cfg: Mahoraga9Config, ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    trading_idx = pd.DatetimeIndex(ohlcv["close"].index)
    folds = m6.build_contiguous_folds(cfg, trading_idx)
    df = pd.DataFrame(folds)
    return df[df["fold"].isin(cfg.mode_folds())].copy().sort_values("fold")


def _weekly_dataset(base_bt: Dict[str, Any], ohlcv: Dict[str, pd.DataFrame], cfg: Mahoraga9Config) -> pd.DataFrame:
    daily_ctx = build_daily_context(base_bt, ohlcv, cfg)
    weekly_ctx = build_weekly_context(daily_ctx, cfg)
    hawkes = build_hawkes_signals(weekly_ctx, cfg)
    labels = build_labels(weekly_ctx.join(hawkes), cfg)
    return weekly_ctx.join(hawkes).join(labels)


def _run_single_fold_v9(fold_row: pd.Series, ohlcv: Dict[str, pd.DataFrame], cfg: Mahoraga9Config, costs: m6.CostsConfig, universe_schedule: Optional[pd.DataFrame]) -> Dict[str, Any]:
    fold_n = int(fold_row["fold"])
    train_start, train_end = fold_row["train_start"], fold_row["train_end"]
    test_start, test_end = fold_row["test_start"], fold_row["test_end"]

    cfg_fold = deepcopy(cfg)
    qqq_full = m6.to_s(ohlcv["close"][cfg.bench_qqq].ffill(), "QQQ")
    dd_thr, vol_thr = m6.calibrate_crisis_thresholds(qqq_full, train_start, train_end, cfg_fold)
    cfg_fold.crisis_dd_thr = dd_thr
    cfg_fold.crisis_vol_zscore_thr = vol_thr

    alpha_cfgs = fit_alpha_configs(ohlcv, cfg_fold, train_start, train_end)
    base_cfg = alpha_cfgs["raw"]
    base_bt = m6.backtest(ohlcv, base_cfg, costs, label=f"FOLD{fold_n}_BASE", universe_schedule=universe_schedule)

    weekly_df = _weekly_dataset(base_bt, ohlcv, cfg_fold)

    train_weekly = weekly_df.loc[:test_start].iloc[:-1].copy()
    frag_fit = fit_best_model(train_weekly, "fragility_y", cfg_fold)
    recv_fit = fit_best_model(train_weekly, "recovery_y", cfg_fold)

    weekly_df["fragility_p"] = apply_model(frag_fit, weekly_df, "fragility_p")
    weekly_df["recovery_p"] = apply_model(recv_fit, weekly_df, "recovery_p")

    pre_test = weekly_df.loc[:test_start].iloc[:-1].copy()
    best_params, calib = calibrate_policy(
        pre_test.dropna(subset=["fragility_p", "recovery_p"]),
        cfg_fold,
    )
    policy_weekly = build_policy_table(weekly_df, best_params, cfg_fold)
    policy_daily = weekly_to_daily_gate(policy_weekly, base_bt["returns_net"].index)

    v9_bt = m6.backtest(
        ohlcv,
        base_cfg,
        costs,
        label=f"FOLD{fold_n}_V9",
        universe_schedule=universe_schedule,
        external_scale=policy_daily["gate_scale"] * policy_daily["vol_mult"] * policy_daily["exp_cap"],
    )

    base_r = base_bt["returns_net"].loc[test_start:test_end]
    v9_r = v9_bt["returns_net"].loc[test_start:test_end]
    base_eq = cfg_fold.capital_initial * (1.0 + base_r).cumprod()
    v9_eq = cfg_fold.capital_initial * (1.0 + v9_r).cumprod()
    s_base = m6.summarize(base_r, base_eq, base_bt["exposure"].loc[base_r.index], base_bt["turnover"].loc[base_r.index], cfg_fold, f"FOLD{fold_n}_BASE")
    s_v9 = m6.summarize(v9_r, v9_eq, v9_bt["exposure"].loc[v9_r.index], v9_bt["turnover"].loc[v9_r.index], cfg_fold, f"FOLD{fold_n}_V9")

    row = {
        "fold": fold_n,
        "train": f"{pd.Timestamp(train_start).date()}→{pd.Timestamp(train_end).date()}",
        "test": f"{pd.Timestamp(test_start).date()}→{pd.Timestamp(test_end).date()}",
        "BASE_CAGR%": round(s_base["CAGR"] * 100, 2),
        "BASE_Sharpe": round(s_base["Sharpe"], 4),
        "BASE_MaxDD%": round(s_base["MaxDD"] * 100, 2),
        "V9_CAGR%": round(s_v9["CAGR"] * 100, 2),
        "V9_Sharpe": round(s_v9["Sharpe"], 4),
        "V9_MaxDD%": round(s_v9["MaxDD"] * 100, 2),
        "FragilityModel": frag_fit.get("name", "neutral"),
        "RecoveryModel": recv_fit.get("name", "neutral"),
        "InterventionRate": round(float((policy_daily["gate_scale"] != 1.0).mean()), 4),
        "MeanGate": round(float(policy_daily["gate_scale"].mean()), 4),
        "MeanAlphaMix": round(float(policy_daily["alpha_mix"].mean()), 4),
    }
    return {
        "fold_row": row,
        "fold": fold_n,
        "base_bt": base_bt,
        "v9_bt": v9_bt,
        "weekly_df": weekly_df,
        "policy_weekly": policy_weekly,
        "policy_daily": policy_daily,
        "calibration_df": calib,
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
    return {
        "results": sorted(results, key=lambda x: x["fold"]),
        "stitched_base": stitch_oos_path(results, "base_bt"),
        "stitched_v9": stitch_oos_path(results, "v9_bt"),
    }
