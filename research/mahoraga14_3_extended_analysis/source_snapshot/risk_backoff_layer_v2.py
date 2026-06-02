from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mahoraga14_config import Mahoraga14Config


def _series(x: Any, index: pd.Index, default: float = 0.0) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.Series(x, index=index, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(float(x), index=index, dtype=float)


def apply_risk_backoff_layer_v2(
    allocator_weekly: pd.DataFrame,
    weekly_df: pd.DataFrame,
    cfg: Mahoraga14Config,
) -> pd.DataFrame:
    idx = allocator_weekly.index
    out = allocator_weekly.copy()
    break_risk = _series(weekly_df.get("continuation_break_risk_p", 0.0), idx, 0.0).clip(0.0, 1.0)
    fragility = _series(out.get("fragility_score", 0.0), idx, 0.0).clip(0.0, 1.0)
    benchmark_weakness = _series(out.get("benchmark_weakness_score", 0.0), idx, 0.0).clip(0.0, 1.0)
    breadth_health = _series(out.get("breadth_health_score", 0.0), idx, 0.0).clip(0.0, 1.0)
    continuation_score = _series(out.get("continuation_allocator_score", 0.0), idx, 0.0).clip(0.0, 1.0)
    benchmark_strength = _series(out.get("benchmark_strength_score", 0.0), idx, 0.0).clip(0.0, 1.0)
    structural_p = _series(weekly_df.get("structural_p", 0.0), idx, 0.0).clip(0.0, 1.0)
    leader_opportunity = _series(out.get("leader_opportunity_score", 0.0), idx, 0.0).clip(0.0, 1.0)
    conviction_score = _series(out.get("conviction_amplifier_score", 0.0), idx, 0.0).clip(0.0, 1.0)

    backoff_score = (
        0.30 * break_risk
        + 0.24 * fragility
        + 0.18 * benchmark_weakness
        + 0.14 * structural_p
        + 0.08 * (1.0 - breadth_health)
        + 0.06 * (1.0 - continuation_score)
    ).clip(0.0, 1.0)

    hard_guard = (
        (break_risk >= float(cfg.risk_backoff_hard_break_risk))
        | (fragility >= float(cfg.risk_backoff_hard_fragility))
        | (benchmark_weakness >= float(cfg.risk_backoff_hard_benchmark_weakness))
        | ((continuation_score <= 0.18) & (benchmark_strength <= 0.20) & (structural_p >= 0.60))
    )

    soft_scale = (1.0 - 0.68 * backoff_score).clip(0.35, 1.0)
    leader_soft_scale = (1.0 - 0.78 * backoff_score).clip(0.15, 1.0)
    conviction_soft_scale = (1.0 - 0.72 * backoff_score).clip(0.25, 1.0)

    out["risk_backoff_score"] = backoff_score
    out["risk_backoff_hard_guard"] = hard_guard.astype(int)

    out["long_budget"] = (
        float(cfg.risk_backoff_budget_floor)
        + (out["long_budget_raw"] - float(cfg.risk_backoff_budget_floor)) * soft_scale
    ).clip(float(cfg.risk_backoff_budget_floor), float(cfg.participation_long_budget_ceiling))
    out["gate_scale_adjustment"] = (
        1.0 + (out["gate_scale_adjustment_raw"] - 1.0) * soft_scale
    ).clip(float(cfg.risk_backoff_gate_floor), float(cfg.participation_gate_max))
    out["vol_mult_adjustment"] = (
        1.0 + (out["vol_mult_adjustment_raw"] - 1.0) * soft_scale
    ).clip(float(cfg.risk_backoff_vol_floor), float(cfg.participation_vol_mult_max))
    out["exp_cap_adjustment"] = (
        1.0 + (out["exp_cap_adjustment_raw"] - 1.0) * soft_scale
    ).clip(float(cfg.risk_backoff_exp_floor), float(cfg.participation_exp_cap_max))
    out["leader_blend"] = (out["leader_blend_raw"] * leader_soft_scale).clip(0.0, float(cfg.participation_leader_blend_max))
    out["conviction_multiplier"] = (
        _series(out.get("conviction_multiplier_raw", 1.0), idx, 1.0).sub(1.0).mul(conviction_soft_scale).add(1.0)
    ).clip(1.0, float(cfg.conviction_weight_scale_max))
    out["leader_multiplier"] = (
        _series(out.get("leader_multiplier_raw", 1.0), idx, 1.0).sub(1.0).mul(leader_soft_scale).add(1.0)
    ).clip(1.0, 1.0 + float(cfg.conviction_leader_boost_max))
    out["cash_budget_target"] = (
        _series(out.get("cash_budget_target_raw", cfg.participation_allocator_cash_target_floor), idx, cfg.participation_allocator_cash_target_floor)
        - 0.06 * backoff_score
    ).clip(float(cfg.risk_backoff_budget_floor), float(cfg.participation_long_budget_ceiling))

    if hard_guard.any():
        out.loc[hard_guard, "long_budget"] = np.minimum(out.loc[hard_guard, "long_budget"], float(cfg.risk_backoff_hard_budget))
        out.loc[hard_guard, "gate_scale_adjustment"] = np.minimum(out.loc[hard_guard, "gate_scale_adjustment"], float(cfg.risk_backoff_gate_floor))
        out.loc[hard_guard, "vol_mult_adjustment"] = np.minimum(out.loc[hard_guard, "vol_mult_adjustment"], float(cfg.risk_backoff_vol_floor))
        out.loc[hard_guard, "exp_cap_adjustment"] = np.minimum(out.loc[hard_guard, "exp_cap_adjustment"], float(cfg.risk_backoff_exp_floor))
        out.loc[hard_guard, "leader_blend"] = 0.0
        out.loc[hard_guard, "conviction_multiplier"] = 1.0
        out.loc[hard_guard, "leader_multiplier"] = 1.0
        out.loc[hard_guard, "cash_budget_target"] = np.minimum(out.loc[hard_guard, "cash_budget_target"], float(cfg.risk_backoff_hard_budget))

    state = pd.Series("NEUTRAL", index=idx, dtype=object)
    state.loc[(out["long_budget"] >= 0.96) & (conviction_score >= 0.55)] = "HIGH_PARTICIPATION"
    state.loc[(out["long_budget"] >= 0.88) & (out["long_budget"] < 0.96)] = "BULL_PARTICIPATION"
    state.loc[(backoff_score >= 0.55) | (out["long_budget"] <= 0.72)] = "BACKOFF"
    state.loc[hard_guard] = "HARD_BACKOFF"
    state.loc[(leader_opportunity >= 0.60) & (state == "BULL_PARTICIPATION")] = "LEADER_BULL_PARTICIPATION"
    out["participation_state"] = state
    return out
