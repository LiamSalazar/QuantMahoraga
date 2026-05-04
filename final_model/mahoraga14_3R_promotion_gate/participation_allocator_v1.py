from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from mahoraga14_config import Mahoraga14Config


def _series(x: Any, index: pd.Index, default: float = 0.0) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.Series(x, index=index, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(float(x), index=index, dtype=float)


def _quantile(series: pd.Series, q: float, default: float) -> float:
    s = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return float(default)
    return float(s.quantile(q))


def _bounds(low: float, high: float, span: float = 0.05) -> tuple[float, float]:
    lo = float(low) if np.isfinite(low) else 0.0
    hi = float(high) if np.isfinite(high) else lo + span
    if hi <= lo:
        hi = lo + max(span, abs(lo) * 0.05 + 1e-4)
    return lo, hi


def _ramp(series: pd.Series, low: float, high: float) -> pd.Series:
    lo, hi = _bounds(low, high)
    denom = max(hi - lo, np.finfo(float).eps)
    s = pd.Series(series, dtype=float)
    return ((s - lo) / denom).clip(0.0, 1.0)


def _reverse_ramp(series: pd.Series, low: float, high: float) -> pd.Series:
    return 1.0 - _ramp(series, low, high)


def _augment_weekly_frame(weekly_df: pd.DataFrame) -> pd.DataFrame:
    out = weekly_df.copy()
    idx = out.index
    defaults = {
        "continuation_trigger_score": 0.0,
        "continuation_pressure_score": 0.0,
        "continuation_break_risk_p": 1.0,
        "continuation_benchmark_score": 0.0,
        "continuation_support_score": 0.0,
        "continuation_structural_health_score": 0.0,
        "breadth_63": 0.0,
        "breadth_rebound_4w": 0.0,
        "qqq_ret_2w": 0.0,
        "qqq_rebound_2w": 0.0,
        "qqq_trend_state_4w": 0.0,
        "qqq_slope_2w": 0.0,
        "qqq_slope_4w": 0.0,
        "qqq_vol_4w": 0.0,
        "avg_corr_21": 0.0,
        "corr_persist_4w": 0.0,
        "corr_release_4w": 0.0,
        "base_ret_2w": 0.0,
        "base_ret_4w": 0.0,
        "base_eff_4w": 0.0,
        "base_realized_vol_4w": 0.0,
        "base_scale_gap": 0.0,
        "base_slope_2w": 0.0,
        "base_slope_4w": 0.0,
        "base_trend_state_4w": 0.0,
        "base_minus_qqq_4w": 0.0,
        "stop_density_4w": 0.0,
        "loss_share_4w": 0.0,
        "down_up_ratio_4w": 0.0,
        "structural_p": 0.0,
        "base_dd": 0.0,
    }
    for col, default in defaults.items():
        out[col] = _series(out[col], idx, default) if col in out.columns else pd.Series(default, index=idx, dtype=float)
    out["underparticipation_gap"] = (out["qqq_ret_2w"].rolling(2, min_periods=1).sum() - out["base_ret_4w"]).fillna(0.0)
    out["benchmark_weakness"] = (-out["qqq_ret_2w"]).clip(lower=0.0)
    out["breadth_weakness"] = (-out["breadth_rebound_4w"]).clip(lower=0.0)
    return out


def fit_participation_allocator_v1(train_weekly: pd.DataFrame, cfg: Mahoraga14Config) -> Dict[str, Any]:
    df = _augment_weekly_frame(train_weekly)
    thresholds = {
        "trigger_low": _quantile(df["continuation_trigger_score"], 0.45, 0.40),
        "trigger_high": _quantile(df["continuation_trigger_score"], 0.82, 0.72),
        "pressure_low": _quantile(df["continuation_pressure_score"], 0.45, 0.40),
        "pressure_high": _quantile(df["continuation_pressure_score"], 0.82, 0.72),
        "benchmark_low": _quantile(df["qqq_ret_2w"], 0.45, 0.00),
        "benchmark_high": _quantile(df["qqq_ret_2w"], 0.82, 0.04),
        "qqq_rebound_low": _quantile(df["qqq_rebound_2w"], 0.45, 0.01),
        "qqq_rebound_high": _quantile(df["qqq_rebound_2w"], 0.82, 0.05),
        "qqq_trend_low": _quantile(df["qqq_trend_state_4w"], 0.40, 0.50),
        "qqq_trend_high": _quantile(df["qqq_trend_state_4w"], 0.80, 0.90),
        "qqq_slope_low": _quantile(df["qqq_slope_4w"], 0.45, 0.00),
        "qqq_slope_high": _quantile(df["qqq_slope_4w"], 0.82, 0.03),
        "breadth_low": _quantile(df["breadth_63"], 0.40, 0.48),
        "breadth_high": _quantile(df["breadth_63"], 0.80, 0.62),
        "breadth_rebound_low": _quantile(df["breadth_rebound_4w"], 0.45, 0.00),
        "breadth_rebound_high": _quantile(df["breadth_rebound_4w"], 0.82, 0.03),
        "corr_release_low": _quantile(df["corr_release_4w"], 0.45, 0.00),
        "corr_release_high": _quantile(df["corr_release_4w"], 0.82, 0.03),
        "avg_corr_low": _quantile(df["avg_corr_21"], 0.30, 0.45),
        "avg_corr_high": _quantile(df["avg_corr_21"], 0.75, 0.70),
        "base_ret_low": _quantile(df["base_ret_4w"], 0.45, 0.00),
        "base_ret_high": _quantile(df["base_ret_4w"], 0.82, 0.05),
        "base_eff_low": _quantile(df["base_eff_4w"], 0.40, 0.45),
        "base_eff_high": _quantile(df["base_eff_4w"], 0.80, 0.72),
        "base_trend_low": _quantile(df["base_trend_state_4w"], 0.40, 0.45),
        "base_trend_high": _quantile(df["base_trend_state_4w"], 0.80, 0.85),
        "base_slope_low": _quantile(df["base_slope_4w"], 0.45, 0.00),
        "base_slope_high": _quantile(df["base_slope_4w"], 0.82, 0.03),
        "break_risk_soft": _quantile(df["continuation_break_risk_p"], 0.55, 0.35),
        "break_risk_hard": _quantile(df["continuation_break_risk_p"], 0.82, 0.60),
        "fragility_soft": _quantile(df["stop_density_4w"] + df["loss_share_4w"], 0.55, 0.55),
        "fragility_hard": _quantile(df["stop_density_4w"] + df["loss_share_4w"], 0.82, 0.80),
        "stop_density_soft": _quantile(df["stop_density_4w"], 0.55, 0.02),
        "stop_density_hard": _quantile(df["stop_density_4w"], 0.82, 0.04),
        "loss_share_soft": _quantile(df["loss_share_4w"], 0.55, 0.55),
        "loss_share_hard": _quantile(df["loss_share_4w"], 0.82, 0.70),
        "scale_gap_soft": _quantile(df["base_scale_gap"], 0.55, 0.20),
        "scale_gap_hard": _quantile(df["base_scale_gap"], 0.82, 0.45),
        "base_vol_soft": _quantile(df["base_realized_vol_4w"], 0.55, 0.24),
        "base_vol_hard": _quantile(df["base_realized_vol_4w"], 0.82, 0.36),
        "qqq_vol_soft": _quantile(df["qqq_vol_4w"], 0.55, 0.24),
        "qqq_vol_hard": _quantile(df["qqq_vol_4w"], 0.82, 0.36),
        "underpart_low": _quantile(df["underparticipation_gap"], 0.55, 0.00),
        "underpart_high": _quantile(df["underparticipation_gap"], 0.85, 0.06),
        "base_minus_qqq_low": _quantile(df["base_minus_qqq_4w"], 0.20, -0.04),
        "base_minus_qqq_high": _quantile(df["base_minus_qqq_4w"], 0.55, 0.00),
        "benchmark_weak_soft": _quantile(df["benchmark_weakness"], 0.55, 0.01),
        "benchmark_weak_hard": _quantile(df["benchmark_weakness"], 0.82, 0.04),
        "structural_soft": _quantile(df["structural_p"], 0.55, 0.40),
        "structural_hard": _quantile(df["structural_p"], 0.82, 0.65),
    }
    return {"name": "PARTICIPATION_ALLOCATOR_V1", "thresholds": thresholds}


def apply_participation_allocator_v1(
    model_info: Dict[str, Any],
    weekly_df: pd.DataFrame,
    cfg: Mahoraga14Config,
) -> pd.DataFrame:
    df = _augment_weekly_frame(weekly_df)
    th = dict(model_info.get("thresholds", {}))

    continuation_score = (
        0.40 * _ramp(df["continuation_pressure_score"], th["pressure_low"], th["pressure_high"])
        + 0.25 * _ramp(df["continuation_trigger_score"], th["trigger_low"], th["trigger_high"])
        + 0.15 * df["continuation_benchmark_score"].clip(0.0, 1.0)
        + 0.10 * df["continuation_support_score"].clip(0.0, 1.0)
        + 0.10 * df["continuation_structural_health_score"].clip(0.0, 1.0)
    ).clip(0.0, 1.0)

    benchmark_strength = (
        0.35 * _ramp(df["qqq_ret_2w"], th["benchmark_low"], th["benchmark_high"])
        + 0.25 * _ramp(df["qqq_rebound_2w"], th["qqq_rebound_low"], th["qqq_rebound_high"])
        + 0.20 * _ramp(df["qqq_trend_state_4w"], th["qqq_trend_low"], th["qqq_trend_high"])
        + 0.20 * _ramp(df["qqq_slope_4w"], th["qqq_slope_low"], th["qqq_slope_high"])
    ).clip(0.0, 1.0)

    persistence_strength = (
        0.30 * _ramp(df["base_ret_4w"], th["base_ret_low"], th["base_ret_high"])
        + 0.25 * _ramp(df["base_eff_4w"], th["base_eff_low"], th["base_eff_high"])
        + 0.20 * _ramp(df["base_trend_state_4w"], th["base_trend_low"], th["base_trend_high"])
        + 0.15 * _ramp(df["base_slope_4w"], th["base_slope_low"], th["base_slope_high"])
        + 0.10 * _reverse_ramp(df["base_realized_vol_4w"], th["base_vol_soft"], th["base_vol_hard"])
    ).clip(0.0, 1.0)

    breadth_health = (
        0.35 * _ramp(df["breadth_63"], th["breadth_low"], th["breadth_high"])
        + 0.30 * _ramp(df["breadth_rebound_4w"], th["breadth_rebound_low"], th["breadth_rebound_high"])
        + 0.20 * _ramp(df["corr_release_4w"], th["corr_release_low"], th["corr_release_high"])
        + 0.15 * _reverse_ramp(df["avg_corr_21"], th["avg_corr_low"], th["avg_corr_high"])
    ).clip(0.0, 1.0)

    fragility_score = (
        0.24 * _ramp(df["stop_density_4w"], th["stop_density_soft"], th["stop_density_hard"])
        + 0.18 * _ramp(df["loss_share_4w"], th["loss_share_soft"], th["loss_share_hard"])
        + 0.14 * _ramp(df["base_scale_gap"], th["scale_gap_soft"], th["scale_gap_hard"])
        + 0.18 * _ramp(df["continuation_break_risk_p"], th["break_risk_soft"], th["break_risk_hard"])
        + 0.12 * _ramp(df["qqq_vol_4w"], th["qqq_vol_soft"], th["qqq_vol_hard"])
        + 0.14 * _ramp(df["structural_p"], th["structural_soft"], th["structural_hard"])
    ).clip(0.0, 1.0)

    benchmark_weakness = (
        0.50 * _ramp(df["benchmark_weakness"], th["benchmark_weak_soft"], th["benchmark_weak_hard"])
        + 0.25 * _reverse_ramp(df["qqq_trend_state_4w"], th["qqq_trend_low"], th["qqq_trend_high"])
        + 0.25 * _ramp((-df["breadth_rebound_4w"]).clip(lower=0.0), 0.0, max(0.02, th["breadth_rebound_high"]))
    ).clip(0.0, 1.0)

    bull_core = (
        0.32 * continuation_score
        + 0.24 * benchmark_strength
        + 0.24 * persistence_strength
        + 0.20 * breadth_health
    ).clip(0.0, 1.0)

    backoff_core = (
        0.42 * fragility_score
        + 0.28 * benchmark_weakness
        + 0.18 * _ramp(df["continuation_break_risk_p"], th["break_risk_soft"], th["break_risk_hard"])
        + 0.12 * _ramp(df["structural_p"], th["structural_soft"], th["structural_hard"])
    ).clip(0.0, 1.0)

    underparticipation_score = (
        0.60 * _ramp(df["underparticipation_gap"], th["underpart_low"], th["underpart_high"])
        + 0.40 * _reverse_ramp(df["base_minus_qqq_4w"], th["base_minus_qqq_low"], th["base_minus_qqq_high"])
    ).clip(0.0, 1.0)

    bull_score = (bull_core - 0.60 * backoff_core).clip(0.0, 1.0)
    target_budget = (
        float(cfg.participation_long_budget_base)
        + (float(cfg.participation_long_budget_ceiling) - float(cfg.participation_long_budget_base)) * bull_score
        + 0.10 * underparticipation_score
        - 0.02 * fragility_score
    ).clip(float(cfg.participation_long_budget_floor), float(cfg.participation_long_budget_ceiling))

    gate_scale_adj = 1.0 + (float(cfg.participation_gate_max) - 1.0) * (0.65 * bull_score + 0.35 * benchmark_strength)
    gate_scale_adj = gate_scale_adj.clip(1.0, float(cfg.participation_gate_max))
    vol_mult_adj = 1.0 + (float(cfg.participation_vol_mult_max) - 1.0) * (0.55 * bull_score + 0.45 * persistence_strength)
    vol_mult_adj = vol_mult_adj.clip(1.0, float(cfg.participation_vol_mult_max))
    exp_cap_adj = 1.0 + (float(cfg.participation_exp_cap_max) - 1.0) * (0.55 * bull_score + 0.25 * underparticipation_score + 0.20 * benchmark_strength)
    exp_cap_adj = exp_cap_adj.clip(1.0, float(cfg.participation_exp_cap_max))
    leader_blend = float(cfg.participation_leader_blend_max) * (
        0.45 * bull_score + 0.30 * underparticipation_score + 0.25 * benchmark_strength
    ).clip(0.0, 1.0)

    state = pd.Series("NEUTRAL", index=df.index, dtype=object)
    state.loc[bull_score >= 0.72] = "HIGH_PARTICIPATION"
    state.loc[(bull_score >= 0.50) & (bull_score < 0.72)] = "BULL_PARTICIPATION"
    state.loc[backoff_core >= 0.70] = "BACKOFF"
    state.loc[(backoff_core >= 0.82) | (fragility_score >= 0.82)] = "HARD_BACKOFF"

    return pd.DataFrame(
        {
            "continuation_allocator_score": continuation_score,
            "benchmark_strength_score": benchmark_strength,
            "persistence_strength_score": persistence_strength,
            "breadth_health_score": breadth_health,
            "fragility_score": fragility_score,
            "benchmark_weakness_score": benchmark_weakness,
            "bull_score_raw": bull_core,
            "backoff_score_raw": backoff_core,
            "underparticipation_score": underparticipation_score,
            "long_budget_raw": target_budget,
            "gate_scale_adjustment_raw": gate_scale_adj,
            "vol_mult_adjustment_raw": vol_mult_adj,
            "exp_cap_adjustment_raw": exp_cap_adj,
            "leader_blend_raw": leader_blend.clip(0.0, float(cfg.participation_leader_blend_max)),
            "participation_state_raw": state,
        },
        index=df.index,
    )
