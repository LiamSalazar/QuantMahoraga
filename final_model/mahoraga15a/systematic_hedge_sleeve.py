from __future__ import annotations

import numpy as np
import pandas as pd

from mahoraga15a_config import Mahoraga15AConfig
from mahoraga15a_utils import clip01


def _solve_non_negative_ridge(a: np.ndarray, gap: np.ndarray, ridge: float) -> np.ndarray:
    gram = a.T @ a + float(ridge) * np.eye(a.shape[1], dtype=float)
    rhs = a.T @ gap
    try:
        sol = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(gram) @ rhs
    return np.clip(sol, 0.0, None)


def _cap_mix(weights: np.ndarray, max_share: float) -> np.ndarray:
    total = float(weights.sum())
    if total <= 1e-12:
        return np.zeros_like(weights)
    mix = weights / total
    lead = int(np.argmax(mix))
    lag = 1 - lead
    if mix[lead] <= max_share:
        return weights
    excess = weights[lead] - max_share * total
    weights = weights.copy()
    weights[lead] -= excess
    weights[lag] += excess
    return np.clip(weights, 0.0, None)


def build_raw_hedge_plan(
    state: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: Mahoraga15AConfig,
) -> pd.DataFrame:
    rows = []
    for dt, row in state.join(controls).iterrows():
        long_mult_target = float(row["long_multiplier_target"])
        long_beta_qqq_scaled = float(row["long_beta_qqq"] * long_mult_target)
        long_beta_spy_scaled = float(row["long_beta_spy"] * long_mult_target)
        gap = np.array(
            [
                max(0.0, long_beta_qqq_scaled - float(row["target_beta_qqq"])),
                max(0.0, long_beta_spy_scaled - float(row["target_beta_spy"])),
            ],
            dtype=float,
        )
        beta_gap_score = float(
            clip01(
                0.55 * (gap[0] / max(1e-6, cfg.target_beta_qqq_high))
                + 0.45 * (gap[1] / max(1e-6, cfg.target_beta_spy_high))
            ).iloc[0]
        )

        directional_permission = float(
            clip01(
                cfg.hedge_directional_floor
                + 0.45 * row["benchmark_weakness"]
                + 0.30 * row["bear_persistence"]
                + 0.25 * row["break_risk"]
                - 0.35 * row["continuation_relief"]
            ).iloc[0]
        )
        fragility_permission = float(
            clip01(
                0.50 * row["structural_fragility"] + 0.30 * row["stress_intensity"] + 0.20 * row["realized_vol_pressure"]
            ).iloc[0]
        )
        hedge_permission = float(
            clip01(
                cfg.hedge_permission_floor
                + 0.45 * beta_gap_score
                + 0.30 * fragility_permission
                + 0.25 * directional_permission
            ).iloc[0]
        )

        a = np.array(
            [
                [1.0, float(row["spy_beta_qqq"])],
                [float(row["qqq_beta_spy"]), 1.0],
            ],
            dtype=float,
        )
        raw = _solve_non_negative_ridge(a, gap, cfg.hedge_solver_ridge)
        raw = _cap_mix(raw, cfg.hedge_max_single_name_share)
        raw_budget_unscaled = float(raw.sum())
        if raw_budget_unscaled > 1e-12:
            raw = raw * (hedge_permission * raw_budget_unscaled / raw_budget_unscaled)
        raw_budget = float(raw.sum())

        projected_beta_qqq = long_beta_qqq_scaled - (raw[0] + raw[1] * float(row["spy_beta_qqq"]))
        projected_beta_spy = long_beta_spy_scaled - (raw[0] * float(row["qqq_beta_spy"]) + raw[1])
        rows.append(
            {
                "Date": dt,
                "raw_short_qqq": float(raw[0]),
                "raw_short_spy": float(raw[1]),
                "raw_short_budget": raw_budget,
                "raw_projected_beta_qqq": projected_beta_qqq,
                "raw_projected_beta_spy": projected_beta_spy,
                "raw_long_beta_qqq_scaled": long_beta_qqq_scaled,
                "raw_long_beta_spy_scaled": long_beta_spy_scaled,
                "raw_beta_gap_qqq": float(gap[0]),
                "raw_beta_gap_spy": float(gap[1]),
                "hedge_directional_permission": directional_permission,
                "hedge_fragility_permission": fragility_permission,
                "hedge_permission": hedge_permission,
            }
        )
    return pd.DataFrame(rows).set_index("Date")
