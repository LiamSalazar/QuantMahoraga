from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from mahoraga11_config import Mahoraga11Config
from mahoraga11_utils import iter_grid, sigmoid


def iter_policy_candidates(cfg: Mahoraga11Config):
    yield from iter_grid(cfg.policy_grid())


def _normalize_signal(series: pd.Series) -> pd.Series:
    s = pd.Series(series, index=series.index, dtype=float).fillna(0.0)
    z = (s - s.rolling(26, min_periods=4).mean()) / s.rolling(26, min_periods=4).std(ddof=1).replace(0.0, np.nan)
    return pd.Series(sigmoid(z.fillna(0.0)), index=s.index)


def build_override_weekly(
    weekly_df: pd.DataFrame,
    policy_params: Dict[str, float],
    cfg: Mahoraga11Config,
) -> pd.DataFrame:
    out = weekly_df.copy()
    hawkes_weight = float(policy_params["hawkes_weight"])

    stress_hawkes = _normalize_signal(out.get("transition_hawkes_stress", pd.Series(0.0, index=out.index)))
    recovery_hawkes = _normalize_signal(out.get("transition_hawkes_recovery", pd.Series(0.0, index=out.index)))

    structural_path = pd.Series(
        sigmoid(
            2.2 * (-out["base_dd"])
            + 1.4 * out["loss_share_4w"]
            + 1.0 * out["stop_density_4w"]
            + 0.9 * out["corr_persist_4w"]
            + 0.8 * (-out["base_minus_qqq_4w"])
            - 1.4 * out["base_rebound_4w"]
            - 0.8 * out["base_eff_4w"]
        ),
        index=out.index,
    )
    transition_path = pd.Series(
        sigmoid(
            2.0 * (-out["base_ret_1w"])
            + 1.3 * (-out["base_dd_change_2w"])
            + 1.0 * out["loss_share_2w"]
            + 1.1 * out["stop_density_2w"]
            + 0.8 * out["base_scale_gap"]
        ),
        index=out.index,
    )
    recovery_path = pd.Series(
        sigmoid(
            1.8 * out["base_rebound_2w"]
            + 1.2 * out["base_rebound_4w"]
            + 1.0 * out["base_dd_change_2w"]
            + 0.8 * out["qqq_rebound_2w"]
            + 0.8 * out["breadth_63"]
            - 1.0 * out["corr_persist_4w"]
        ),
        index=out.index,
    )

    out["structural_path_score"] = structural_path.clip(0.0, 1.0)
    out["transition_path_score"] = transition_path.clip(0.0, 1.0)
    out["recovery_path_score"] = recovery_path.clip(0.0, 1.0)
    out["stress_hawkes_norm"] = stress_hawkes
    out["recovery_hawkes_norm"] = recovery_hawkes

    out["structural_score"] = (0.72 * out["structural_p"] + 0.28 * out["structural_path_score"] + 0.10 * hawkes_weight * stress_hawkes).clip(0.0, 1.0)
    out["transition_score"] = (0.60 * out["transition_p"] + 0.25 * out["transition_path_score"] + 0.15 * hawkes_weight * stress_hawkes).clip(0.0, 1.0)
    out["recovery_score"] = (0.60 * out["recovery_p"] + 0.25 * out["recovery_path_score"] + 0.15 * hawkes_weight * recovery_hawkes).clip(0.0, 1.0)

    structural_enter = float(policy_params["structural_enter_thr"])
    structural_exit = max(0.45, structural_enter - 0.12)
    transition_enter = float(policy_params["transition_enter_thr"])
    recovery_enter = float(policy_params["recovery_enter_thr"])

    out["override_type"] = "BASELINE"
    out["override_detail"] = "BASELINE"
    out["defense_blend"] = 0.0
    out["gate_scale"] = 1.0
    out["vol_mult"] = 1.0
    out["exp_cap"] = 1.0
    out["is_override"] = 0.0
    out["is_structural_override"] = 0.0
    out["is_transition_override"] = 0.0
    out["is_recovery_lift"] = 0.0

    structural_on = False
    transition_cooldown = 0
    for dt in out.index:
        row = out.loc[dt]
        structural_guard = (row["base_dd"] <= -0.05) or (row["stop_density_4w"] >= 0.03) or (row["base_eff_4w"] <= 0.30)
        transition_guard = (row["base_ret_1w"] <= -0.02) or (row["base_dd_change_2w"] <= -0.02)
        recovery_guard = (row["base_rebound_2w"] >= 0.02) or (row["base_dd_change_2w"] >= 0.01)

        if structural_on:
            structural_exit_ok = (row["structural_score"] <= structural_exit and recovery_guard) or (
                row["structural_score"] <= structural_exit - 0.05
            )
            if structural_exit_ok:
                structural_on = False

        if not structural_on and row["structural_score"] >= structural_enter and structural_guard:
            structural_on = True

        if structural_on:
            out.loc[dt, "override_type"] = "STRUCTURAL_DEFENSE"
            out.loc[dt, "override_detail"] = "STRUCTURAL_DEFENSE"
            out.loc[dt, "defense_blend"] = float(policy_params["structural_blend"])
            out.loc[dt, "gate_scale"] = float(policy_params["structural_gate"])
            out.loc[dt, "vol_mult"] = min(float(policy_params["structural_gate"]) + 0.05, 1.0)
            out.loc[dt, "exp_cap"] = float(policy_params["structural_exp_cap"])
            out.loc[dt, "is_override"] = 1.0
            out.loc[dt, "is_structural_override"] = 1.0
            transition_cooldown = 2
            continue

        if row["transition_score"] >= transition_enter and transition_guard:
            out.loc[dt, "override_type"] = "TRANSITION_RECOVERY"
            out.loc[dt, "override_detail"] = "TRANSITION_STRESS"
            out.loc[dt, "defense_blend"] = float(policy_params["transition_blend"])
            out.loc[dt, "gate_scale"] = float(policy_params["transition_gate"])
            out.loc[dt, "vol_mult"] = float(policy_params["transition_vol_mult"])
            out.loc[dt, "exp_cap"] = float(policy_params["transition_exp_cap"])
            out.loc[dt, "is_override"] = 1.0
            out.loc[dt, "is_transition_override"] = 1.0
            transition_cooldown = 2
            continue

        if row["recovery_score"] >= recovery_enter and (transition_cooldown > 0 or recovery_guard):
            out.loc[dt, "override_type"] = "TRANSITION_RECOVERY"
            out.loc[dt, "override_detail"] = "RECOVERY_LIFT"
            out.loc[dt, "defense_blend"] = 0.0
            out.loc[dt, "gate_scale"] = 1.0
            out.loc[dt, "vol_mult"] = float(policy_params["recovery_vol_mult"])
            out.loc[dt, "exp_cap"] = float(policy_params["recovery_exp_cap"])
            out.loc[dt, "is_override"] = 1.0
            out.loc[dt, "is_transition_override"] = 1.0
            out.loc[dt, "is_recovery_lift"] = 1.0

        transition_cooldown = max(0, transition_cooldown - 1)

    return out


def weekly_to_daily_override(override_weekly: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    cols = [
        "override_type",
        "override_detail",
        "defense_blend",
        "gate_scale",
        "vol_mult",
        "exp_cap",
        "is_override",
        "is_structural_override",
        "is_transition_override",
        "is_recovery_lift",
        "structural_score",
        "transition_score",
        "recovery_score",
        "structural_p",
        "transition_p",
        "recovery_p",
    ]
    out = override_weekly[cols].reindex(idx).ffill()
    out["defense_blend"] = out["defense_blend"].fillna(0.0).clip(0.0, 1.0)
    out["gate_scale"] = out["gate_scale"].fillna(1.0).clip(0.0, 1.0)
    out["vol_mult"] = out["vol_mult"].fillna(1.0)
    out["exp_cap"] = out["exp_cap"].fillna(1.0)
    out["is_override"] = out["is_override"].fillna(0.0)
    out["is_structural_override"] = out["is_structural_override"].fillna(0.0)
    out["is_transition_override"] = out["is_transition_override"].fillna(0.0)
    out["is_recovery_lift"] = out["is_recovery_lift"].fillna(0.0)
    out["override_type"] = out["override_type"].fillna("BASELINE")
    out["override_detail"] = out["override_detail"].fillna("BASELINE")
    for col in ["structural_score", "transition_score", "recovery_score", "structural_p", "transition_p", "recovery_p"]:
        out[col] = out[col].fillna(0.0)
    return out
