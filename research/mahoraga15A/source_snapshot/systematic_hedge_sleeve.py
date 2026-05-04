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


def _floor_budget(activation: float, onset: float, cap: float, scale: float, power: float) -> float:
    if activation <= onset:
        return 0.0
    normalized = (activation - onset) / max(1e-6, 1.0 - onset)
    return float(cap * scale * normalized ** power)


def _mix_from_overlay(beta_raw: np.ndarray, overlay_share: float) -> np.ndarray:
    overlay_share = float(np.clip(overlay_share, 0.20, 0.80))
    overlay_mix = np.array([overlay_share, 1.0 - overlay_share], dtype=float)
    if beta_raw.sum() > 1e-12:
        beta_mix = beta_raw / beta_raw.sum()
    else:
        beta_mix = overlay_mix
    mix = 0.70 * beta_mix + 0.30 * overlay_mix
    mix = np.clip(mix, 0.0, None)
    return mix / mix.sum() if mix.sum() > 1e-12 else np.array([0.5, 0.5], dtype=float)


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

        a = np.array(
            [
                [1.0, float(row["spy_beta_qqq"])],
                [float(row["qqq_beta_spy"]), 1.0],
            ],
            dtype=float,
        )
        base_beta_raw = _cap_mix(_solve_non_negative_ridge(a, gap, cfg.hedge_solver_ridge), cfg.hedge_max_single_name_share)
        crash_share = float(clip01(0.55 * row["crash_activation"] + 0.25 * row["crash_transition"] + 0.20 * row["hawkes_stress"]).iloc[0])
        bear_share = float(clip01(0.50 * row["bear_activation"] + 0.30 * row["bear_transition"] + 0.20 * row["benchmark_weakness"]).iloc[0])
        share_norm = max(1e-6, crash_share + bear_share)
        crash_beta_raw = base_beta_raw * (crash_share / share_norm)
        bear_beta_raw = base_beta_raw * (bear_share / share_norm)

        crash_beta_budget = float(crash_beta_raw.sum()) * float(
            clip01(cfg.crash_beta_weight * row["crash_permission"] + 0.25 * beta_gap_score + 0.20 * row["crash_transition"]).iloc[0]
        )
        bear_beta_budget = float(bear_beta_raw.sum()) * float(
            clip01(cfg.bear_beta_weight * row["bear_permission"] + 0.20 * beta_gap_score + 0.20 * row["bear_transition"]).iloc[0]
        )

        crash_floor = _floor_budget(
            activation=float(row["crash_activation"]),
            onset=cfg.crash_floor_activation,
            cap=float(row["crash_cap_dynamic"]),
            scale=cfg.crash_overlay_floor_scale,
            power=cfg.crash_floor_power,
        )
        bear_floor = _floor_budget(
            activation=float(row["bear_activation"]),
            onset=cfg.bear_floor_activation,
            cap=float(row["bear_cap_dynamic"]),
            scale=cfg.bear_overlay_floor_scale,
            power=cfg.bear_floor_power,
        )
        crash_overlay_budget = crash_floor * float(
            clip01(cfg.crash_overlay_weight * row["crash_transition"] + (1.0 - cfg.crash_overlay_weight) * row["crash_risk_score"]).iloc[0]
        )
        bear_overlay_budget = bear_floor * float(
            clip01(cfg.bear_overlay_weight * row["bear_transition"] + (1.0 - cfg.bear_overlay_weight) * row["bear_risk_score"]).iloc[0]
        )

        crash_budget = max(crash_beta_budget, crash_overlay_budget)
        bear_budget = max(bear_beta_budget, bear_overlay_budget)
        crash_mix = _mix_from_overlay(crash_beta_raw, float(row.get("qqq_crash_share", 0.60)))
        bear_mix = _mix_from_overlay(bear_beta_raw, float(row.get("qqq_bear_share", 0.55)))
        raw_crash = _cap_mix(crash_budget * crash_mix, cfg.hedge_max_single_name_share)
        raw_bear = _cap_mix(bear_budget * bear_mix, cfg.hedge_max_single_name_share)
        raw_total = raw_crash + raw_bear

        projected_beta_qqq = long_beta_qqq_scaled - (raw_total[0] + raw_total[1] * float(row["spy_beta_qqq"]))
        projected_beta_spy = long_beta_spy_scaled - (raw_total[0] * float(row["qqq_beta_spy"]) + raw_total[1])
        rows.append(
            {
                "Date": dt,
                "raw_crash_short_qqq": float(raw_crash[0]),
                "raw_crash_short_spy": float(raw_crash[1]),
                "raw_crash_short_budget": float(raw_crash.sum()),
                "raw_bear_short_qqq": float(raw_bear[0]),
                "raw_bear_short_spy": float(raw_bear[1]),
                "raw_bear_short_budget": float(raw_bear.sum()),
                "raw_short_qqq": float(raw_total[0]),
                "raw_short_spy": float(raw_total[1]),
                "raw_short_budget": float(raw_total.sum()),
                "raw_projected_beta_qqq": projected_beta_qqq,
                "raw_projected_beta_spy": projected_beta_spy,
                "raw_long_beta_qqq_scaled": long_beta_qqq_scaled,
                "raw_long_beta_spy_scaled": long_beta_spy_scaled,
                "raw_beta_gap_qqq": float(gap[0]),
                "raw_beta_gap_spy": float(gap[1]),
                "raw_beta_budget_crash": crash_beta_budget,
                "raw_beta_budget_bear": bear_beta_budget,
                "raw_overlay_budget_crash": crash_overlay_budget,
                "raw_overlay_budget_bear": bear_overlay_budget,
                "crash_permission": float(row["crash_permission"]),
                "bear_permission": float(row["bear_permission"]),
                "hedge_permission": float(row["hedge_permission"]),
            }
        )
    return pd.DataFrame(rows).set_index("Date")
