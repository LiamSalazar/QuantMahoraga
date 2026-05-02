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


def _crisis_floor_size(crisis_activation: float, short_cap_dynamic: float, cfg: Mahoraga15AConfig) -> float:
    if crisis_activation <= cfg.hedge_crisis_onset:
        return 0.0
    normalized = (crisis_activation - cfg.hedge_crisis_onset) / max(1e-6, 1.0 - cfg.hedge_crisis_onset)
    return float(short_cap_dynamic * cfg.hedge_overlay_floor_scale * normalized ** cfg.hedge_crisis_floor_power)


def build_raw_hedge_plan(
    state: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: Mahoraga15AConfig,
) -> pd.DataFrame:
    rows = []
    merged = state.copy()
    for col in controls.columns:
        merged[col] = controls[col]
    for dt, row in merged.iterrows():
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

        crisis_activation = float(row["crisis_activation"])
        crisis_transition = float(row["crisis_transition"])
        crisis_persistence = float(row["crisis_persistence"])
        continuation_relief = float(row["continuation_relief"])
        benchmark_weakness = float(row["benchmark_weakness"])
        fragility_permission = float(
            clip01(
                0.45 * row["structural_fragility"]
                + 0.25 * row["realized_vol_pressure"]
                + 0.15 * crisis_activation
                + 0.15 * row["break_risk"]
            ).iloc[0]
        )
        directional_permission = float(
            clip01(
                cfg.hedge_directional_floor
                + 0.40 * benchmark_weakness
                + 0.35 * crisis_persistence
                + 0.25 * crisis_transition
                - 0.40 * continuation_relief
            ).iloc[0]
        )
        hedge_permission = float(
            clip01(
                0.20 * beta_gap_score
                + 0.35 * crisis_activation
                + 0.25 * fragility_permission
                + 0.20 * directional_permission
            ).iloc[0]
        )

        a = np.array(
            [
                [1.0, float(row["spy_beta_qqq"])],
                [float(row["qqq_beta_spy"]), 1.0],
            ],
            dtype=float,
        )
        beta_raw = _solve_non_negative_ridge(a, gap, cfg.hedge_solver_ridge)
        beta_raw = _cap_mix(beta_raw, cfg.hedge_max_single_name_share)
        beta_budget = float(beta_raw.sum()) * (cfg.hedge_overlay_beta_weight * hedge_permission)

        crisis_floor_budget = _crisis_floor_size(crisis_activation, float(row["short_cap_dynamic"]), cfg)
        overlay_floor_budget = crisis_floor_budget * clip01(
            cfg.hedge_overlay_crisis_weight * directional_permission
            + (1.0 - cfg.hedge_overlay_crisis_weight) * crisis_transition
        ).iloc[0]

        qqq_overlay_share = float(row.get("qqq_overlay_share", 0.55))
        qqq_overlay_share = float(np.clip(qqq_overlay_share, 0.20, 0.80))
        overlay_mix = np.array([qqq_overlay_share, 1.0 - qqq_overlay_share], dtype=float)

        if beta_raw.sum() > 1e-12:
            beta_mix = beta_raw / beta_raw.sum()
        else:
            beta_mix = overlay_mix

        raw_budget = max(beta_budget, overlay_floor_budget)
        if overlay_floor_budget > beta_budget:
            mix = 0.65 * overlay_mix + 0.35 * beta_mix
        else:
            mix = 0.75 * beta_mix + 0.25 * overlay_mix
        mix = np.clip(mix, 0.0, None)
        mix = mix / mix.sum() if mix.sum() > 1e-12 else np.array([0.5, 0.5], dtype=float)

        raw = raw_budget * mix
        raw = _cap_mix(raw, cfg.hedge_max_single_name_share)
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
                "raw_beta_budget": beta_budget,
                "raw_crisis_floor_budget": overlay_floor_budget,
                "hedge_directional_permission": directional_permission,
                "hedge_fragility_permission": fragility_permission,
                "hedge_permission": hedge_permission,
            }
        )
    return pd.DataFrame(rows).set_index("Date")
