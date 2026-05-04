from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mahoraga14_config import Mahoraga14Config


def _series(x: Any, index: pd.Index, default: float = 0.0) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.Series(x, index=index, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(float(x), index=index, dtype=float)


def apply_risk_backoff_layer(
    allocator_weekly: pd.DataFrame,
    weekly_df: pd.DataFrame,
    cfg: Mahoraga14Config,
) -> pd.DataFrame:
    idx = allocator_weekly.index
    out = allocator_weekly.copy()
    break_risk = _series(weekly_df.get("continuation_break_risk_p", 0.0), idx, 0.0)
    fragility = _series(out.get("fragility_score", 0.0), idx, 0.0)
    benchmark_weakness = _series(out.get("benchmark_weakness_score", 0.0), idx, 0.0)
    breadth_health = _series(out.get("breadth_health_score", 0.0), idx, 0.0)
    structural_p = _series(weekly_df.get("structural_p", 0.0), idx, 0.0)
    continuation_pressure = _series(weekly_df.get("continuation_pressure_score", 0.0), idx, 0.0)
    benchmark_score = _series(weekly_df.get("continuation_benchmark_score", 0.0), idx, 0.0)

    backoff_score = (
        0.35 * break_risk.clip(0.0, 1.0)
        + 0.25 * fragility.clip(0.0, 1.0)
        + 0.18 * benchmark_weakness.clip(0.0, 1.0)
        + 0.12 * structural_p.clip(0.0, 1.0)
        + 0.10 * (1.0 - breadth_health.clip(0.0, 1.0))
    ).clip(0.0, 1.0)

    hard_guard = (
        (break_risk >= float(cfg.risk_backoff_hard_break_risk))
        | (fragility >= float(cfg.risk_backoff_hard_fragility))
        | (benchmark_weakness >= float(cfg.risk_backoff_hard_benchmark_weakness))
        | ((continuation_pressure <= 0.15) & (benchmark_score <= 0.20) & (structural_p >= 0.60))
    )

    soft_scale = (1.0 - 0.55 * backoff_score).clip(0.45, 1.0)
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
    out["leader_blend"] = (out["leader_blend_raw"] * (1.0 - 0.70 * backoff_score)).clip(0.0, float(cfg.participation_leader_blend_max))

    if hard_guard.any():
        out.loc[hard_guard, "long_budget"] = np.minimum(out.loc[hard_guard, "long_budget"], float(cfg.risk_backoff_hard_budget))
        out.loc[hard_guard, "gate_scale_adjustment"] = np.minimum(out.loc[hard_guard, "gate_scale_adjustment"], float(cfg.risk_backoff_gate_floor))
        out.loc[hard_guard, "vol_mult_adjustment"] = np.minimum(out.loc[hard_guard, "vol_mult_adjustment"], float(cfg.risk_backoff_vol_floor))
        out.loc[hard_guard, "exp_cap_adjustment"] = np.minimum(out.loc[hard_guard, "exp_cap_adjustment"], float(cfg.risk_backoff_exp_floor))
        out.loc[hard_guard, "leader_blend"] = 0.0

    state = pd.Series("NEUTRAL", index=idx, dtype=object)
    state.loc[out["long_budget"] >= 0.92] = "HIGH_PARTICIPATION"
    state.loc[(out["long_budget"] >= 0.82) & (out["long_budget"] < 0.92)] = "BULL_PARTICIPATION"
    state.loc[(backoff_score >= 0.55) | (out["long_budget"] <= 0.60)] = "BACKOFF"
    state.loc[hard_guard] = "HARD_BACKOFF"
    out["participation_state"] = state
    return out
