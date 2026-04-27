from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from continuation_v2_model import evaluate_continuation_v2_context
from mahoraga13_config import Mahoraga13Config
from mahoraga13_utils import iter_grid, sigmoid


def iter_policy_candidates(cfg: Mahoraga13Config):
    yield from iter_grid(cfg.policy_grid())


def _normalize_signal(series: pd.Series) -> pd.Series:
    s = pd.Series(series, index=series.index, dtype=float).fillna(0.0)
    z = (s - s.rolling(26, min_periods=4).mean()) / s.rolling(26, min_periods=4).std(ddof=1).replace(0.0, np.nan)
    return pd.Series(sigmoid(z.fillna(0.0)), index=s.index)


def _variant_flags(variant: str) -> Dict[str, bool]:
    name = str(variant).strip().upper()
    if name == "BASELINE":
        return {"allow_structural": False, "allow_continuation_v2": False}
    if name in {"STRUCTURAL_ONLY", "STRUCTURAL_DEFENSE_ONLY"}:
        return {"allow_structural": True, "allow_continuation_v2": False}
    if name == "CONTINUATION_V2_ONLY":
        return {"allow_structural": False, "allow_continuation_v2": True}
    if name in {"STRUCTURAL_CONTINUATION_V2", "STRUCTURAL_DEFENSE_PLUS_CONTINUATION_V2"}:
        return {"allow_structural": True, "allow_continuation_v2": True}
    raise ValueError(f"Unknown override variant: {variant}")


def build_override_weekly(
    weekly_df: pd.DataFrame,
    policy_params: Dict[str, float],
    cfg: Mahoraga13Config,
    variant: str = "STRUCTURAL_DEFENSE_ONLY",
    continuation_v2_info: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    out = weekly_df.copy()
    flags = _variant_flags(variant)
    hawkes_weight = float(policy_params["hawkes_weight"])
    continuation_v2_info = continuation_v2_info or {}
    guard_thresholds = continuation_v2_info.get("guard_thresholds", {})

    stress_hawkes = _normalize_signal(out.get("transition_hawkes_stress", pd.Series(0.0, index=out.index)))
    recovery_hawkes = _normalize_signal(out.get("transition_hawkes_recovery", pd.Series(0.0, index=out.index)))
    hawkes_release = _normalize_signal(
        out.get("transition_hawkes_recovery", pd.Series(0.0, index=out.index))
        - 0.5 * out.get("transition_hawkes_stress", pd.Series(0.0, index=out.index))
    )
    continuation_p = pd.Series(
        out.get("continuation_v2_p", pd.Series(0.0, index=out.index)),
        index=out.index,
        dtype=float,
    ).fillna(0.0)

    if guard_thresholds:
        context = evaluate_continuation_v2_context(out, guard_thresholds)
    else:
        context = pd.DataFrame(
            {
                "compression_valid": 0.0,
                "pause_valid": 0.0,
                "support_valid": 0.0,
                "benchmark_valid": 0.0,
                "stress_ok": 0.0,
            },
            index=out.index,
        )

    structural_path = pd.Series(
        sigmoid(
            2.4 * (-out["base_dd"])
            + 1.1 * out["base_dd_duration_4w"]
            + 1.0 * (-out["base_dd_velocity_4w"])
            + 1.2 * out["loss_share_4w"]
            + 1.0 * out["stop_density_4w"]
            + 0.9 * out["corr_persist_4w"]
            + 0.8 * (-out["base_minus_qqq_4w"])
            - 1.4 * out["base_rebound_4w"]
            - 1.0 * out["base_eff_4w"]
        ),
        index=out.index,
    ).clip(0.0, 1.0)

    compression_path = pd.concat(
        [
            _normalize_signal(1.0 - out.get("path_efficiency_2w", pd.Series(0.0, index=out.index))).rename("path_efficiency_2w"),
            _normalize_signal(1.0 - out.get("path_efficiency_4w", pd.Series(0.0, index=out.index))).rename("path_efficiency_4w"),
            _normalize_signal(out.get("swing_amplitude_vs_net_2w", pd.Series(0.0, index=out.index))).rename("swing_amplitude_vs_net_2w"),
            _normalize_signal(out.get("swing_amplitude_vs_net_4w", pd.Series(0.0, index=out.index))).rename("swing_amplitude_vs_net_4w"),
            _normalize_signal(out.get("compression_score_2w", pd.Series(0.0, index=out.index))).rename("compression_score_2w"),
            _normalize_signal(out.get("compression_score_4w", pd.Series(0.0, index=out.index))).rename("compression_score_4w"),
        ],
        axis=1,
    ).mean(axis=1).clip(0.0, 1.0)
    support_path = pd.concat(
        [
            _normalize_signal(out.get("base_ret_1w", pd.Series(0.0, index=out.index))).rename("base_ret_1w"),
            _normalize_signal(out.get("base_ret_2w", pd.Series(0.0, index=out.index))).rename("base_ret_2w"),
            _normalize_signal(out.get("stop_release_2w", pd.Series(0.0, index=out.index))).rename("stop_release_2w"),
            _normalize_signal(out.get("breadth_rebound_4w", pd.Series(0.0, index=out.index))).rename("breadth_rebound_4w"),
            _normalize_signal(out.get("scale_recovery_2w", pd.Series(0.0, index=out.index))).rename("scale_recovery_2w"),
            _normalize_signal(out.get("corr_release_4w", pd.Series(0.0, index=out.index))).rename("corr_release_4w"),
            _normalize_signal(out.get("qqq_ret_2w", pd.Series(0.0, index=out.index))).rename("qqq_ret_2w"),
            _normalize_signal(out.get("qqq_rebound_2w", pd.Series(0.0, index=out.index))).rename("qqq_rebound_2w"),
            hawkes_release.rename("hawkes_release"),
        ],
        axis=1,
    ).mean(axis=1).clip(0.0, 1.0)
    continuation_path = (0.55 * compression_path + 0.45 * support_path).clip(0.0, 1.0)

    out["structural_path_score"] = structural_path
    out["continuation_v2_path_score"] = continuation_path
    out["stress_hawkes_norm"] = stress_hawkes
    out["recovery_hawkes_norm"] = recovery_hawkes
    out["continuation_v2_p"] = continuation_p
    out["continuation_compression_valid"] = pd.Series(context["compression_valid"], index=out.index, dtype=float).fillna(0.0)
    out["continuation_pause_valid"] = pd.Series(context["pause_valid"], index=out.index, dtype=float).fillna(0.0)
    out["continuation_support_valid"] = pd.Series(context["support_valid"], index=out.index, dtype=float).fillna(0.0)
    out["continuation_benchmark_valid"] = pd.Series(context["benchmark_valid"], index=out.index, dtype=float).fillna(0.0)
    out["continuation_stress_ok"] = pd.Series(context["stress_ok"], index=out.index, dtype=float).fillna(0.0)
    out["structural_score"] = (
        0.74 * pd.Series(out.get("structural_p", 0.0), index=out.index, dtype=float).fillna(0.0)
        + 0.22 * out["structural_path_score"]
        + 0.04 * hawkes_weight * stress_hawkes
    ).clip(0.0, 1.0)
    out["continuation_v2_score"] = (
        0.78 * continuation_p
        + 0.22 * continuation_path
    ).clip(0.0, 1.0)

    structural_enter = float(policy_params["structural_enter_thr"])
    structural_exit = max(0.46, structural_enter - 0.12)
    continuation_enter = float(continuation_v2_info.get("entry_threshold", 1.01))

    out["override_type"] = "BASELINE"
    out["override_detail"] = "BASELINE"
    out["defense_blend"] = 0.0
    out["gate_scale"] = 1.0
    out["vol_mult"] = 1.0
    out["exp_cap"] = 1.0
    out["is_override"] = 0.0
    out["is_structural_override"] = 0.0
    out["is_continuation_v2"] = 0.0
    out["is_continuation_reentry"] = 0.0
    out["continuation_guard_pass"] = 0.0

    structural_on = False
    for dt in out.index:
        row = out.loc[dt]
        structural_guard = (
            (row["base_dd"] <= -0.05)
            or (row["base_dd_duration_4w"] >= 2.5)
            or (row["stop_density_4w"] >= 0.03)
            or (row["base_eff_4w"] <= 0.30)
        )

        if structural_on:
            structural_exit_ok = (
                (row["structural_score"] <= structural_exit and row["base_rebound_4w"] > 0.0)
                or (row["structural_score"] <= structural_exit - 0.05)
            )
            if structural_exit_ok:
                structural_on = False

        if flags["allow_structural"] and not structural_on and row["structural_score"] >= structural_enter and structural_guard:
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
            continue

        continuation_guard = (
            flags["allow_continuation_v2"]
            and bool(row["continuation_compression_valid"] > 0.0)
            and bool(row["continuation_pause_valid"] > 0.0)
            and bool(row["continuation_support_valid"] > 0.0)
            and bool(row["continuation_benchmark_valid"] > 0.0)
            and bool(row["continuation_stress_ok"] > 0.0)
            and (row["continuation_v2_score"] >= continuation_enter)
            and (row["structural_score"] <= structural_enter - cfg.continuation_v2_structural_margin)
        )
        out.loc[dt, "continuation_guard_pass"] = float(continuation_guard)
        if continuation_guard:
            out.loc[dt, "override_type"] = "CONTINUATION_LIFT"
            out.loc[dt, "override_detail"] = "CONTINUATION_LIFT"
            out.loc[dt, "defense_blend"] = 0.0
            out.loc[dt, "gate_scale"] = float(cfg.continuation_v2_gate)
            out.loc[dt, "vol_mult"] = float(cfg.continuation_v2_vol_mult)
            out.loc[dt, "exp_cap"] = float(cfg.continuation_v2_exp_cap)
            out.loc[dt, "is_override"] = 1.0
            out.loc[dt, "is_continuation_v2"] = 1.0
            out.loc[dt, "is_continuation_reentry"] = 1.0

    return out


def weekly_to_daily_override(override_weekly: pd.DataFrame, idx: pd.DatetimeIndex, cfg: Mahoraga13Config) -> pd.DataFrame:
    cols = [
        "override_type",
        "override_detail",
        "defense_blend",
        "gate_scale",
        "vol_mult",
        "exp_cap",
        "is_override",
        "is_structural_override",
        "is_continuation_v2",
        "is_continuation_reentry",
        "structural_score",
        "continuation_v2_score",
        "structural_p",
        "continuation_v2_p",
        "continuation_compression_valid",
        "continuation_pause_valid",
        "continuation_support_valid",
        "continuation_benchmark_valid",
        "continuation_stress_ok",
        "continuation_guard_pass",
    ]
    out = override_weekly[cols].reindex(idx).ffill()
    out["defense_blend"] = out["defense_blend"].fillna(0.0).clip(0.0, 1.0)
    out["gate_scale"] = out["gate_scale"].fillna(1.0).clip(0.0, cfg.max_gate_scale)
    out["vol_mult"] = out["vol_mult"].fillna(1.0)
    out["exp_cap"] = out["exp_cap"].fillna(1.0)
    out["is_override"] = out["is_override"].fillna(0.0)
    out["is_structural_override"] = out["is_structural_override"].fillna(0.0)
    out["is_continuation_v2"] = out["is_continuation_v2"].fillna(0.0)
    out["is_continuation_reentry"] = out["is_continuation_reentry"].fillna(0.0)
    out["override_type"] = out["override_type"].fillna("BASELINE")
    out["override_detail"] = out["override_detail"].fillna("BASELINE")
    for col in [
        "structural_score",
        "continuation_v2_score",
        "structural_p",
        "continuation_v2_p",
        "continuation_compression_valid",
        "continuation_pause_valid",
        "continuation_support_valid",
        "continuation_benchmark_valid",
        "continuation_stress_ok",
        "continuation_guard_pass",
    ]:
        out[col] = out[col].fillna(0.0)
    return out
