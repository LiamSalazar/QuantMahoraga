from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import pandas as pd

try:
    import mahoraga6_1 as m6
    import mahoraga7_1 as h7
except Exception:
    import mahoraga6_1 as m6  # type: ignore
    import mahoraga7_1 as h7  # type: ignore

from mahoraga8_config import Mahoraga8Config
from mahoraga8_regime import build_regime_table
from mahoraga8_policy import build_policy_table
from mahoraga8_core import run_adaptive_core_backtest
from mahoraga8_calibration import calibrate_mahoraga8


def _get_fold_cfg(
    ohlcv: Dict[str, pd.DataFrame],
    base_cfg: Mahoraga8Config,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
    fold_row: pd.Series,
) -> Mahoraga8Config:
    cfg = deepcopy(base_cfg)
    cfg.weight_cap = float(fold_row["best_weight_cap"])
    cfg.k_atr = float(fold_row["best_k_atr"])
    cfg.turb_zscore_thr = float(fold_row["best_turb_zscore_thr"])
    cfg.turb_scale_min = float(fold_row["best_turb_scale_min"])
    cfg.vol_target_ann = float(fold_row["best_vol_target_ann"])

    train_start, train_end = h7._parse_range(fold_row["train"])
    qqq_full = m6.to_s(ohlcv["close"][cfg.bench_qqq].ffill(), "QQQ")
    dd_thr, vol_thr = m6.calibrate_crisis_thresholds(qqq_full, train_start, train_end, cfg)
    cfg.crisis_dd_thr = dd_thr
    cfg.crisis_vol_zscore_thr = vol_thr

    final_train_tickers = m6.get_training_universe(
        train_end, universe_schedule, cfg.universe_static, list(ohlcv["close"].columns)
    )
    close_univ = ohlcv["close"][final_train_tickers]
    wt, wm, wr = m6.fit_ic_weights(close_univ, qqq_full.loc[train_start:train_end], cfg, train_start, train_end)
    cfg.w_trend, cfg.w_mom, cfg.w_rel = wt, wm, wr
    return cfg


def _run_single_fold_h8(
    fold_n: int,
    baseline_row: pd.Series,
    ohlcv: Dict[str, pd.DataFrame],
    cfg_base: Mahoraga8Config,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    train_start, train_end = h7._parse_range(baseline_row["train"])
    test_start, test_end = h7._parse_range(baseline_row["test"])

    cfg_fold = _get_fold_cfg(ohlcv, cfg_base, costs, universe_schedule, baseline_row)
    print(f"  [IC] trend={cfg_fold.w_trend:.3f} mom={cfg_fold.w_mom:.3f} rel={cfg_fold.w_rel:.3f}")
    print(f"  [fold {fold_n}] Calibrating {cfg_fold.variant} on train via inner validation …")

    feat_full = h7._build_context_table(ohlcv, cfg_fold, universe_schedule)
    best_params, calib_df = calibrate_mahoraga8(feat_full, ohlcv, cfg_fold, costs, universe_schedule, train_start, train_end)

    hawkes_df, _ = h7._build_hawkes_signals(
        feat_full,
        best_params["stress_q"], best_params["recovery_q"], best_params["decay"],
        best_params["stress_scale"], best_params["recovery_scale"],
        feat_full.loc[train_start:train_end],
    )

    base_bt = m6.backtest(ohlcv, cfg_fold, costs, label=f"BASE_{fold_n}", universe_schedule=universe_schedule)
    train_idx = hawkes_df.index[(hawkes_df.index >= pd.Timestamp(train_start)) & (hawkes_df.index <= pd.Timestamp(train_end))]
    regime_table = build_regime_table(
        hawkes_df, base_bt, base_bt["bench"]["QQQ_r"], train_idx, cfg_fold,
        hawkes_urgency_weight=float(best_params["hawkes_urgency_weight"]),
        hawkes_panic_boost=float(best_params["hawkes_panic_boost"]),
        hawkes_recovery_boost=float(best_params["hawkes_recovery_boost"]),
    )
    policy_table = build_policy_table(
        regime_table, cfg_fold,
        state_map=str(best_params["state_map"]),
        risk_budget_blend=float(best_params["risk_budget_blend"]),
        exposure_cap_mult=float(best_params["exposure_cap_mult"]),
        vol_target_shift=float(best_params["vol_target_shift"]),
    )

    ov_bt = run_adaptive_core_backtest(
        ohlcv, cfg_fold, costs, universe_schedule, regime_table, policy_table,
        label=f"H8_2HM_{fold_n}",
    )

    rb = base_bt["returns_net"].loc[test_start:test_end]
    qb = base_bt["equity"].loc[test_start:test_end]
    eb = base_bt["exposure"].loc[test_start:test_end]
    tb = base_bt["turnover"].loc[test_start:test_end]
    sb = m6.summarize(rb, qb, eb, tb, cfg_fold, f"BASE_{fold_n}")

    ro = ov_bt["returns_net"].loc[test_start:test_end]
    qo = ov_bt["equity"].loc[test_start:test_end]
    eo = ov_bt["exposure"].loc[test_start:test_end]
    to = ov_bt["turnover"].loc[test_start:test_end]
    so = m6.summarize(ro, qo, eo, to, cfg_fold, f"H8_2HM_{fold_n}")

    regime_test = policy_table["active_regime_state"].reindex(ro.index).ffill().fillna("NORMAL")
    qqq_future = base_bt["bench"]["QQQ_r"].resample(cfg_fold.decision_freq).sum(min_count=1).reindex(policy_table.index).fillna(0.0)
    pos = qqq_future > 0
    captured = ((policy_table["active_regime_state"].isin(["RECOVERY", "NORMAL"])) & pos).sum()
    missed = pos.sum() - captured
    recovery_capture_rate = float(captured / max(int(pos.sum()), 1)) if pos.sum() > 0 else 0.0

    print(f"  [fold {fold_n}] BASE Sharpe={sb['Sharpe']:.3f} | 8.2HM Sharpe={so['Sharpe']:.3f} | Δ={so['Sharpe'] - sb['Sharpe']:+.3f}")

    return {
        "fold": fold_n,
        "base_bt": base_bt,
        "ov_bt": ov_bt,
        "policy_table": policy_table,
        "regime_table": regime_table,
        "calib_df": calib_df,
        "best_params": best_params,
        "fold_row": {
            "fold": fold_n,
            "train": f"{train_start}→{train_end}",
            "test": f"{test_start}→{test_end}",
            "BASE_CAGR%": round(sb["CAGR"] * 100, 2),
            "BASE_Sharpe": round(sb["Sharpe"], 4),
            "BASE_MaxDD%": round(sb["MaxDD"] * 100, 2),
            "H8_CAGR%": round(so["CAGR"] * 100, 2),
            "H8_Sharpe": round(so["Sharpe"], 4),
            "H8_MaxDD%": round(so["MaxDD"] * 100, 2),
            "H8_CVaR5%": round(so["CVaR_5"] * 100, 2),
            "DeltaSharpe": round(so["Sharpe"] - sb["Sharpe"], 4),
            "MeanRiskBudget": round(float(policy_table["risk_budget_cap"].mean()), 4),
            "InterventionRate": round(float((policy_table["active_regime_state"] != "NORMAL").mean()), 4),
            "MissedReboundQQQ": round(float(missed), 2),
            "RecoveryCaptureRate": round(recovery_capture_rate, 4),
            "PanicRate": round(float((regime_test == "PANIC").mean()), 4),
            "StressRate": round(float((regime_test == "STRESS").mean()), 4),
        },
    }


def stitch_oos_path(results: list[Dict[str, Any]], key: str) -> pd.Series:
    parts = []
    for r in results:
        bt = r[key]
        test_str = r["fold_row"]["test"]
        test_start, test_end = h7._parse_range(test_str)
        parts.append(bt["returns_net"].loc[test_start:test_end])
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).sort_index()
