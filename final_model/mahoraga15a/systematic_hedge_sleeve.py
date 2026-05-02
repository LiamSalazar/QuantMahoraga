from __future__ import annotations

import numpy as np
import pandas as pd

from mahoraga15a_config import Mahoraga15AConfig


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
        a = np.array(
            [
                [1.0, float(row["spy_beta_qqq"])],
                [float(row["qqq_beta_spy"]), 1.0],
            ],
            dtype=float,
        )
        raw = _solve_non_negative_ridge(a, gap, cfg.hedge_solver_ridge)
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
            }
        )
    return pd.DataFrame(rows).set_index("Date")
