from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mahoraga14_config import Mahoraga14Config


def _series(x: Any, index: pd.Index, default: float = 0.0) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.Series(x, index=index, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(float(x), index=index, dtype=float)


def apply_conviction_amplifier_layer(
    allocator_weekly_raw: pd.DataFrame,
    weekly_df: pd.DataFrame,
    cfg: Mahoraga14Config,
) -> pd.DataFrame:
    idx = allocator_weekly_raw.index
    out = allocator_weekly_raw.copy()
    continuation_score = _series(out.get("continuation_allocator_score", 0.0), idx, 0.0).clip(0.0, 1.0)
    bull_score = _series(out.get("bull_score_raw", 0.0), idx, 0.0).clip(0.0, 1.0)
    participation_pressure = _series(out.get("participation_pressure_score", 0.0), idx, 0.0).clip(0.0, 1.0)
    benchmark_strength = _series(out.get("benchmark_strength_score", 0.0), idx, 0.0).clip(0.0, 1.0)
    breadth_health = _series(out.get("breadth_health_score", 0.0), idx, 0.0).clip(0.0, 1.0)
    volatility_health = _series(out.get("volatility_health_score", 0.0), idx, 0.0).clip(0.0, 1.0)
    fragility = _series(out.get("fragility_score", 0.0), idx, 0.0).clip(0.0, 1.0)
    backoff_score = _series(out.get("backoff_score_raw", 0.0), idx, 0.0).clip(0.0, 1.0)
    leader_opportunity = _series(out.get("leader_opportunity_score", 0.0), idx, 0.0).clip(0.0, 1.0)
    residual_relief = _series(out.get("residual_relief_score", 0.0), idx, 0.0).clip(0.0, 1.0)
    cash_drag_pressure = _series(out.get("cash_drag_pressure_score", 0.0), idx, 0.0).clip(0.0, 1.0)
    break_risk = _series(weekly_df.get("continuation_break_risk_p", 0.0), idx, 0.0).clip(0.0, 1.0)
    structural_p = _series(weekly_df.get("structural_p", 0.0), idx, 0.0).clip(0.0, 1.0)

    healthy_regime = (
        0.22 * continuation_score
        + 0.22 * benchmark_strength
        + 0.18 * bull_score
        + 0.14 * breadth_health
        + 0.12 * volatility_health
        + 0.12 * (1.0 - fragility)
    ).clip(0.0, 1.0)

    amplifier_score = (
        0.40 * participation_pressure
        + 0.18 * leader_opportunity
        + 0.16 * residual_relief
        + 0.14 * cash_drag_pressure
        + 0.12 * healthy_regime
        - 0.30 * backoff_score
    ).clip(0.0, 1.0)

    activation = (
        (participation_pressure >= float(cfg.conviction_activation_threshold))
        & (continuation_score >= 0.45)
        & (benchmark_strength >= 0.45)
        & (fragility <= 0.55)
        & (break_risk <= 0.55)
        & (structural_p <= 0.68)
    )

    live_score = amplifier_score.where(activation, amplifier_score * 0.25)
    budget_boost = float(cfg.conviction_max_budget_boost) * live_score
    gate_boost = float(cfg.conviction_gate_boost_max) * live_score
    vol_boost = float(cfg.conviction_vol_boost_max) * live_score
    exp_boost = float(cfg.conviction_exp_boost_max) * live_score
    leader_boost = float(cfg.conviction_leader_boost_max) * live_score
    conviction_boost = (float(cfg.conviction_weight_scale_max) - 1.0) * live_score

    out["conviction_regime_score"] = healthy_regime
    out["conviction_amplifier_score"] = live_score
    out["conviction_activation"] = activation.astype(int)

    out["long_budget_raw"] = (out["long_budget_raw"] + budget_boost).clip(
        float(cfg.participation_long_budget_floor),
        float(cfg.participation_long_budget_ceiling),
    )
    out["gate_scale_adjustment_raw"] = (out["gate_scale_adjustment_raw"] + gate_boost).clip(1.0, float(cfg.participation_gate_max))
    out["vol_mult_adjustment_raw"] = (out["vol_mult_adjustment_raw"] + vol_boost).clip(1.0, float(cfg.participation_vol_mult_max))
    out["exp_cap_adjustment_raw"] = (out["exp_cap_adjustment_raw"] + exp_boost).clip(1.0, float(cfg.participation_exp_cap_max))
    out["leader_blend_raw"] = (out["leader_blend_raw"] + leader_boost).clip(0.0, float(cfg.participation_leader_blend_max))
    out["conviction_multiplier_raw"] = (
        _series(out.get("conviction_multiplier_raw", 1.0), idx, 1.0) + conviction_boost
    ).clip(1.0, float(cfg.conviction_weight_scale_max))
    out["leader_multiplier_raw"] = (
        _series(out.get("leader_multiplier_raw", 1.0), idx, 1.0) + leader_boost
    ).clip(1.0, 1.0 + float(cfg.conviction_leader_boost_max))

    target_floor = float(cfg.participation_allocator_cash_target_floor)
    current_cash_target = _series(out.get("cash_budget_target_raw", target_floor), idx, target_floor)
    desired_cash_target = (
        current_cash_target + 0.04 * live_score + 0.02 * cash_drag_pressure - 0.03 * backoff_score
    ).clip(target_floor, float(cfg.participation_allocator_cash_target_ceiling))
    out["cash_budget_target_raw"] = desired_cash_target.clip(
        float(cfg.participation_allocator_cash_target_floor),
        float(cfg.participation_long_budget_ceiling),
    )

    state = pd.Series(out.get("participation_state_raw", "NEUTRAL"), index=idx, dtype=object)
    state.loc[activation & (live_score >= 0.72)] = "AMPLIFIED_HIGH_PARTICIPATION"
    state.loc[activation & (live_score >= 0.50) & (live_score < 0.72)] = "AMPLIFIED_BULL_PARTICIPATION"
    out["participation_state_raw"] = state
    return out
