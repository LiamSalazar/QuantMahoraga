from __future__ import annotations

from itertools import product as iproduct
from typing import Any, Dict, Optional, Tuple

import numpy as np
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


def _future_window_return(returns_daily: pd.Series, weekly_idx: pd.DatetimeIndex, horizon_weeks: int) -> pd.Series:
    idx = returns_daily.index
    out = pd.Series(np.nan, index=weekly_idx, dtype=float)
    if len(idx) == 0:
        return out
    step = max(1, horizon_weeks * 5)
    r_arr = returns_daily.to_numpy(dtype=float)
    pos_arr = idx.searchsorted(weekly_idx.values, side="left")
    n = len(idx)
    log1r = np.log1p(np.clip(r_arr, -0.9999, None))
    cumlog = np.concatenate([[0.0], np.cumsum(log1r)])
    vals = np.empty(len(weekly_idx), dtype=float)
    vals[:] = np.nan
    for i, pos in enumerate(pos_arr):
        if pos >= n:
            continue
        end_i = min(n - 1, pos + step)
        if end_i <= pos:
            continue
        vals[i] = np.expm1(cumlog[end_i + 1] - cumlog[pos + 1])
    return pd.Series(vals, index=weekly_idx, dtype=float)


def _panic_summary(ov_r: pd.Series, panic_state: pd.Series, cfg: Mahoraga8Config) -> Dict[str, float]:
    panic_r = ov_r.reindex(panic_state.index).fillna(0.0)
    panic_mask = panic_state.astype(str).isin(["PANIC"])
    pr = panic_r.loc[panic_mask]
    if len(pr) < 3:
        return {"panic_sharpe": 0.0, "panic_days": float(panic_mask.sum())}
    std = float(pr.std(ddof=1))
    sharpe = float(pr.mean() / std * np.sqrt(cfg.trading_days)) if std > 1e-12 else 0.0
    return {"panic_sharpe": sharpe, "panic_days": float(len(pr))}


def _stress_sharpe_from_policy(ov_r: pd.Series, state_series: pd.Series, cfg: Mahoraga8Config) -> float:
    mask = state_series.astype(str).isin(["STRESS"])
    r = ov_r.reindex(state_series.index).fillna(0.0).loc[mask]
    if len(r) < 3:
        return 0.0
    std = float(r.std(ddof=1))
    return float(r.mean() / std * np.sqrt(cfg.trading_days)) if std > 1e-12 else 0.0


def _forward_positive_capture(policy_weekly: pd.DataFrame, qqq_future_window: pd.Series) -> Tuple[float, float, float]:
    fw = qqq_future_window.reindex(policy_weekly.index).fillna(0.0)
    pos = fw > 0
    if pos.sum() == 0:
        return 0.0, 0.0, 0.0
    captured = (policy_weekly["active_regime_state"].isin(["RECOVERY", "NORMAL"]) & pos).sum()
    missed = pos.sum() - captured
    rate = float(captured / max(int(pos.sum()), 1))
    return float(missed), float(captured), rate


def _score_candidate(
    base_sum: Dict[str, float],
    ov_sum: Dict[str, float],
    panic_sharpe: float,
    stress_sharpe: float,
    missed_rebound: float,
    recovery_capture_rate: float,
    turnover_ann: float,
    worst_fold_proxy: float,
    intervention_rate: float,
    cfg: Mahoraga8Config,
) -> float:
    delta_sharpe = float(ov_sum["Sharpe"] - base_sum["Sharpe"])
    delta_cagr = float(ov_sum["CAGR"] - base_sum["CAGR"])
    delta_dd = float(base_sum["MaxDD"] - ov_sum["MaxDD"])
    delta_cvar = float(base_sum["CVaR_5"] - ov_sum["CVaR_5"])
    pen_int = max(0.0, float(intervention_rate) - float(cfg.target_intervention_rate))
    return float(
        cfg.score_w_sharpe * delta_sharpe +
        cfg.score_w_cagr * delta_cagr +
        cfg.score_w_dd * delta_dd +
        cfg.score_w_cvar * delta_cvar +
        cfg.score_w_panic * panic_sharpe +
        cfg.score_w_stress * stress_sharpe +
        cfg.score_w_recovery_capture * recovery_capture_rate
        - cfg.score_pen_missed_rebound * missed_rebound / 100.0
        - cfg.score_pen_turnover * turnover_ann / 100.0
        - cfg.score_pen_worst_fold * max(0.0, -worst_fold_proxy)
        - cfg.score_pen_intervention * pen_int
    )


def _calibrate_stage1_hawkes(
    feat_full: pd.DataFrame,
    train_start: str,
    inner_train_end: str,
    inner_val_start: str,
    inner_val_end: str,
    cfg_fold: Mahoraga8Config,
    crisis_state_weekly: pd.Series,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows = []
    best_score = -np.inf
    best = None
    overrides = cfg_fold.mode_overrides()
    for stress_q, recovery_q, decay, stress_scale, recovery_scale in iproduct(
        overrides["stress_q_grid"],
        overrides["recovery_q_grid"],
        overrides["hawkes_decay_grid"],
        overrides["stress_scale_grid"],
        overrides["recovery_scale_grid"],
    ):
        hawkes_df, _ = h7._build_hawkes_signals(
            feat_full, stress_q, recovery_q, decay, stress_scale, recovery_scale,
            feat_full.loc[train_start:inner_train_end],
        )
        score = h7._diagnostic_alignment_score(hawkes_df.loc[inner_val_start:inner_val_end], crisis_state_weekly)
        row = {
            "stress_q": stress_q,
            "recovery_q": recovery_q,
            "decay": decay,
            "stress_scale": stress_scale,
            "recovery_scale": recovery_scale,
            "score": score,
        }
        rows.append(row)
        if score > best_score:
            best_score = score
            best = row.copy()
    return best, pd.DataFrame(rows).sort_values("score", ascending=False)


def calibrate_mahoraga8(
    feat_full: pd.DataFrame,
    ohlcv: Dict[str, pd.DataFrame],
    cfg_fold: Mahoraga8Config,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
    train_start: str,
    train_end: str,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    feat_train = feat_full.loc[train_start:train_end].copy()
    overrides = cfg_fold.mode_overrides()
    if len(feat_train) < cfg_fold.min_train_weeks:
        return {
            "stress_q": overrides["stress_q_grid"][0],
            "recovery_q": overrides["recovery_q_grid"][0],
            "decay": overrides["hawkes_decay_grid"][0],
            "stress_scale": overrides["stress_scale_grid"][0],
            "recovery_scale": overrides["recovery_scale_grid"][0],
            "state_map": overrides["state_map_grid"][0],
            "risk_budget_blend": overrides["risk_budget_blend_grid"][0],
            "exposure_cap_mult": overrides["exposure_cap_mult_grid"][0],
            "vol_target_shift": overrides["vol_target_shift_grid"][0],
            "hawkes_urgency_weight": overrides["hawkes_urgency_weight_grid"][0],
            "hawkes_panic_boost": overrides["hawkes_panic_boost_grid"][0],
            "hawkes_recovery_boost": overrides["hawkes_recovery_boost_grid"][0],
        }, pd.DataFrame()

    split_n = max(int(len(feat_train) * (1.0 - cfg_fold.inner_val_frac)), cfg_fold.min_train_weeks)
    split_n = min(split_n, len(feat_train) - max(10, len(feat_train) // 5))
    inner_train_end = str(feat_train.index[split_n - 1].date())
    inner_val_start = str(feat_train.index[split_n].date())
    inner_val_end = train_end

    base_bt = m6.backtest(ohlcv, cfg_fold, costs, label="BASE_INNER", universe_schedule=universe_schedule)
    base_r = base_bt["returns_net"].loc[inner_val_start:inner_val_end]
    base_eq = base_bt["equity"].loc[inner_val_start:inner_val_end]
    base_exp = base_bt["exposure"].loc[inner_val_start:inner_val_end]
    base_to = base_bt["turnover"].loc[inner_val_start:inner_val_end]
    base_sum = m6.summarize(base_r, base_eq, base_exp, base_to, cfg_fold, "BASE_INNER")
    crisis_state_weekly = base_bt["crisis_state"].resample(cfg_fold.decision_freq).last().reindex(feat_full.index).ffill().fillna(0.0)

    s1_best, s1_df = _calibrate_stage1_hawkes(feat_full, train_start, inner_train_end, inner_val_start, inner_val_end, cfg_fold, crisis_state_weekly)
    hawkes_df_fixed, _ = h7._build_hawkes_signals(
        feat_full,
        s1_best["stress_q"], s1_best["recovery_q"], s1_best["decay"],
        s1_best["stress_scale"], s1_best["recovery_scale"],
        feat_full.loc[train_start:inner_train_end],
    )

    wk_idx = hawkes_df_fixed.index
    inner_train_idx = wk_idx[(wk_idx >= pd.Timestamp(train_start)) & (wk_idx <= pd.Timestamp(inner_train_end))]
    inner_val_idx = wk_idx[(wk_idx >= pd.Timestamp(inner_val_start)) & (wk_idx <= pd.Timestamp(inner_val_end))]

    rows = []
    best_score = -np.inf
    best = None

    for state_map, risk_budget_blend, exposure_cap_mult, vol_target_shift, hawkes_urgency_weight, hawkes_panic_boost, hawkes_recovery_boost in iproduct(
        overrides["state_map_grid"],
        overrides["risk_budget_blend_grid"],
        overrides["exposure_cap_mult_grid"],
        overrides["vol_target_shift_grid"],
        overrides["hawkes_urgency_weight_grid"],
        overrides["hawkes_panic_boost_grid"],
        overrides["hawkes_recovery_boost_grid"],
    ):
        regime_table = build_regime_table(
            hawkes_df_fixed, base_bt, base_bt["bench"]["QQQ_r"], inner_train_idx, cfg_fold,
            hawkes_urgency_weight=float(hawkes_urgency_weight),
            hawkes_panic_boost=float(hawkes_panic_boost),
            hawkes_recovery_boost=float(hawkes_recovery_boost),
        )
        policy_table = build_policy_table(
            regime_table,
            cfg_fold,
            state_map=str(state_map),
            risk_budget_blend=float(risk_budget_blend),
            exposure_cap_mult=float(exposure_cap_mult),
            vol_target_shift=float(vol_target_shift),
        )
        ov_bt = run_adaptive_core_backtest(
            ohlcv, cfg_fold, costs, universe_schedule,
            regime_table, policy_table, label="H8_2HM_INNER",
        )
        ov_r = ov_bt["returns_net"].loc[inner_val_start:inner_val_end]
        ov_eq = ov_bt["equity"].loc[inner_val_start:inner_val_end]
        ov_exp = ov_bt["exposure"].loc[inner_val_start:inner_val_end]
        ov_to = ov_bt["turnover"].loc[inner_val_start:inner_val_end]
        ov_sum = m6.summarize(ov_r, ov_eq, ov_exp, ov_to, cfg_fold, "H8_2HM_INNER")

        qqq_future = _future_window_return(base_bt["bench"]["QQQ_r"], regime_table.index, cfg_fold.conformal_horizon_weeks)
        missed_rebound, _captured, recovery_capture_rate = _forward_positive_capture(policy_table.loc[inner_val_idx], qqq_future.loc[inner_val_idx])
        state_val = policy_table["active_regime_state"].reindex(ov_r.index).ffill().fillna("NORMAL")
        panic_metrics = _panic_summary(ov_r, state_val, cfg_fold)
        stress_sharpe = _stress_sharpe_from_policy(ov_r, state_val, cfg_fold)
        turnover_ann = float(ov_to.mean() * cfg_fold.trading_days) if len(ov_to) else 0.0
        intervention_rate = float((policy_table.loc[inner_val_idx, "active_regime_state"] != "NORMAL").mean()) if len(inner_val_idx) else 0.0
        worst_fold_proxy = min(0.0, ov_sum["Sharpe"] - base_sum["Sharpe"])

        score = _score_candidate(
            base_sum, ov_sum, panic_metrics["panic_sharpe"], stress_sharpe,
            missed_rebound, recovery_capture_rate, turnover_ann, worst_fold_proxy,
            intervention_rate, cfg_fold,
        )

        row = {
            **s1_best,
            "state_map": state_map,
            "risk_budget_blend": risk_budget_blend,
            "exposure_cap_mult": exposure_cap_mult,
            "vol_target_shift": vol_target_shift,
            "hawkes_urgency_weight": hawkes_urgency_weight,
            "hawkes_panic_boost": hawkes_panic_boost,
            "hawkes_recovery_boost": hawkes_recovery_boost,
            "score": score,
            "base_sharpe": base_sum["Sharpe"],
            "ov_sharpe": ov_sum["Sharpe"],
            "ov_cagr": ov_sum["CAGR"],
            "ov_maxdd": ov_sum["MaxDD"],
            "ov_cvar": ov_sum["CVaR_5"],
            "missed_rebound": missed_rebound,
            "recovery_capture_rate": recovery_capture_rate,
            "panic_sharpe": panic_metrics["panic_sharpe"],
            "stress_sharpe": stress_sharpe,
            "turnover_ann": turnover_ann,
            "intervention_rate": intervention_rate,
        }
        rows.append(row)
        if score > best_score:
            best_score = score
            best = row.copy()

    calib_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    if len(s1_df):
        s1 = s1_df.copy()
        s1["stage"] = "hawkes_only"
        calib_df["stage"] = "markov_hawkes_fusion"
        calib_df = pd.concat([s1, calib_df], ignore_index=True, sort=False)
    return best, calib_df
