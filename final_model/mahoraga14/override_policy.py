from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from mahoraga14_config import Mahoraga14Config
from mahoraga14_utils import iter_grid, sigmoid


def iter_policy_candidates(cfg: Mahoraga14Config):
    yield from iter_grid(cfg.policy_grid())


def _normalize_signal(series: pd.Series) -> pd.Series:
    s = pd.Series(series, index=series.index, dtype=float).fillna(0.0)
    z = (s - s.rolling(26, min_periods=4).mean()) / s.rolling(26, min_periods=4).std(ddof=1).replace(0.0, np.nan)
    return pd.Series(sigmoid(z.fillna(0.0)), index=s.index)


def _variant_flags(variant: str) -> Dict[str, bool]:
    name = str(variant).strip().upper()
    if name in {"BASELINE", "BASE_ALPHA", "BASE_ALPHA_V2"}:
        return {"allow_structural": False, "allow_continuation": False}
    if name in {"STRUCTURAL_ONLY", "STRUCTURAL_DEFENSE_ONLY"}:
        return {"allow_structural": True, "allow_continuation": False}
    if name in {"CONTINUATION_PRESSURE_V2_ONLY", "CONTINUATION_ONLY"}:
        return {"allow_structural": False, "allow_continuation": True}
    if name in {
        "STRUCTURAL_CONTINUATION_PRESSURE_V2",
        "STRUCTURAL_DEFENSE_PLUS_CONTINUATION_PRESSURE_V2",
        "STRUCTURAL_DEFENSE_PLUS_CONTINUATION",
    }:
        return {"allow_structural": True, "allow_continuation": True}
    raise ValueError(f"Unknown override variant: {variant}")


def _ramp_value(value: float, low: float, high: float) -> float:
    lo = float(low)
    hi = float(high)
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = lo + 0.05
    if hi <= lo:
        hi = lo + max(0.05, abs(lo) * 0.05 + 1e-4)
    return float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0))


def build_override_weekly(
    weekly_df: pd.DataFrame,
    policy_params: Dict[str, float],
    cfg: Mahoraga14Config,
    variant: str = "STRUCTURAL_DEFENSE_ONLY",
    continuation_info: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    out = weekly_df.copy()
    flags = _variant_flags(variant)
    hawkes_weight = float(policy_params["hawkes_weight"])
    continuation_info = continuation_info or {}

    stress_hawkes = _normalize_signal(out.get("transition_hawkes_stress", pd.Series(0.0, index=out.index)))
    recovery_hawkes = _normalize_signal(out.get("transition_hawkes_recovery", pd.Series(0.0, index=out.index)))

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

    structural_p = pd.Series(out.get("structural_p", pd.Series(0.0, index=out.index)), index=out.index, dtype=float).fillna(0.0)
    trigger_p = pd.Series(out.get("continuation_trigger_p", pd.Series(0.0, index=out.index)), index=out.index, dtype=float).fillna(0.0)
    pressure_p = pd.Series(out.get("continuation_pressure_p", pd.Series(0.0, index=out.index)), index=out.index, dtype=float).fillna(0.0)
    break_risk_p = pd.Series(out.get("continuation_break_risk_p", pd.Series(1.0, index=out.index)), index=out.index, dtype=float).fillna(1.0)
    trigger_score = pd.Series(out.get("continuation_trigger_score", trigger_p), index=out.index, dtype=float).fillna(0.0)
    pressure_score = pd.Series(out.get("continuation_pressure_score", pressure_p), index=out.index, dtype=float).fillna(0.0)
    trigger_context = pd.Series(out.get("continuation_trigger_context", pd.Series(0.0, index=out.index)), index=out.index, dtype=float).fillna(0.0)
    pressure_context = pd.Series(out.get("continuation_pressure_context", pd.Series(0.0, index=out.index)), index=out.index, dtype=float).fillna(0.0)
    compression_score = pd.Series(out.get("continuation_compression_score", pd.Series(0.0, index=out.index)), index=out.index, dtype=float).fillna(0.0)
    pause_score = pd.Series(out.get("continuation_pause_score", pd.Series(0.0, index=out.index)), index=out.index, dtype=float).fillna(0.0)
    support_score = pd.Series(out.get("continuation_support_score", pd.Series(0.0, index=out.index)), index=out.index, dtype=float).fillna(0.0)
    benchmark_score = pd.Series(out.get("continuation_benchmark_score", pd.Series(0.0, index=out.index)), index=out.index, dtype=float).fillna(0.0)
    structural_health = pd.Series(out.get("continuation_structural_health_score", pd.Series(0.0, index=out.index)), index=out.index, dtype=float).fillna(0.0)

    out["stress_hawkes_norm"] = stress_hawkes
    out["recovery_hawkes_norm"] = recovery_hawkes
    out["structural_p"] = structural_p
    out["structural_path_score"] = structural_path
    out["structural_score"] = (0.74 * structural_p + 0.22 * structural_path + 0.04 * hawkes_weight * stress_hawkes).clip(0.0, 1.0)
    out["continuation_trigger_p"] = trigger_p
    out["continuation_pressure_p"] = pressure_p
    out["continuation_break_risk_p"] = break_risk_p
    out["continuation_trigger_score"] = trigger_score
    out["continuation_pressure_score"] = pressure_score
    out["continuation_trigger_context"] = trigger_context
    out["continuation_pressure_context"] = pressure_context
    out["continuation_compression_score"] = compression_score
    out["continuation_pause_score"] = pause_score
    out["continuation_support_score"] = support_score
    out["continuation_benchmark_score"] = benchmark_score
    out["continuation_structural_health_score"] = structural_health
    out["continuation_compression_valid"] = pd.Series(out.get("continuation_compression_valid", 0.0), index=out.index, dtype=float).fillna(0.0)
    out["continuation_pause_valid"] = pd.Series(out.get("continuation_pause_valid", 0.0), index=out.index, dtype=float).fillna(0.0)
    out["continuation_benchmark_valid"] = pd.Series(out.get("continuation_benchmark_valid", 0.0), index=out.index, dtype=float).fillna(0.0)
    out["continuation_trigger_valid"] = pd.Series(out.get("continuation_trigger_valid", 0.0), index=out.index, dtype=float).fillna(0.0)
    out["continuation_structural_low"] = pd.Series(out.get("continuation_structural_low", 0.0), index=out.index, dtype=float).fillna(0.0)

    structural_enter = float(policy_params["structural_enter_thr"])
    structural_exit = max(0.46, structural_enter - 0.12)
    trigger_enter = float(continuation_info.get("trigger_enter", 1.01))
    trigger_ceiling = float(continuation_info.get("trigger_ceiling", max(trigger_enter + 0.05, 1.0)))
    pressure_enter = float(continuation_info.get("pressure_enter", 1.01))
    pressure_ceiling = float(continuation_info.get("pressure_ceiling", max(pressure_enter + 0.05, 1.0)))
    break_risk_floor = float(continuation_info.get("break_risk_floor", 0.0))
    break_risk_cap = float(continuation_info.get("break_risk_cap", 0.0))

    headroom_span = max(cfg.continuation_structural_margin + 0.04, 0.08)
    out["continuation_structural_headroom"] = ((structural_enter - out["structural_score"]) / headroom_span).clip(0.0, 1.0)
    out["continuation_break_headroom"] = (
        (break_risk_cap - break_risk_p) / max(0.03, break_risk_cap - break_risk_floor)
    ).clip(0.0, 1.0)
    out["continuation_pressure"] = (
        0.40 * pressure_score
        + 0.20 * trigger_score
        + 0.15 * benchmark_score
        + 0.15 * out["continuation_structural_headroom"]
        + 0.10 * out["continuation_break_headroom"]
    ).clip(0.0, 1.0)

    out["override_type"] = "BASELINE"
    out["override_detail"] = "BASELINE"
    out["defense_blend"] = 0.0
    out["gate_scale"] = 1.0
    out["vol_mult"] = 1.0
    out["exp_cap"] = 1.0
    out["is_override"] = 0.0
    out["is_structural_override"] = 0.0
    out["is_continuation_lift"] = 0.0
    out["is_continuation_activation"] = 0.0
    out["continuation_guard_pass"] = 0.0
    out["continuation_soft_intensity"] = 0.0

    structural_on = False
    continuation_on = False
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
            continuation_on = False
            out.loc[dt, "override_type"] = "STRUCTURAL_DEFENSE"
            out.loc[dt, "override_detail"] = "STRUCTURAL_DEFENSE"
            out.loc[dt, "defense_blend"] = float(policy_params["structural_blend"])
            out.loc[dt, "gate_scale"] = float(policy_params["structural_gate"])
            out.loc[dt, "vol_mult"] = min(float(policy_params["structural_gate"]) + 0.05, 1.0)
            out.loc[dt, "exp_cap"] = float(policy_params["structural_exp_cap"])
            out.loc[dt, "is_override"] = 1.0
            out.loc[dt, "is_structural_override"] = 1.0
            out.loc[dt, "continuation_structural_headroom"] = 0.0
            out.loc[dt, "continuation_break_headroom"] = 0.0
            out.loc[dt, "continuation_pressure"] = 0.0
            continue

        soft_candidate = (
            flags["allow_continuation"]
            and bool(row["continuation_structural_low"] > 0.0)
            and bool((row["continuation_compression_valid"] > 0.0) or (row["continuation_pause_valid"] > 0.0))
            and bool(row["continuation_benchmark_valid"] > 0.0)
            and bool(row["continuation_trigger_valid"] > 0.0)
            and bool(row["structural_score"] <= structural_enter - cfg.continuation_structural_margin)
            and bool(row["continuation_break_risk_p"] <= break_risk_cap)
        )
        hard_guard = (
            soft_candidate
            and (row["continuation_trigger_score"] >= trigger_enter)
            and (row["continuation_pressure_score"] >= pressure_enter)
        )

        if continuation_on:
            sustain_ok = (
                flags["allow_continuation"]
                and (row["structural_score"] <= structural_enter - 0.02)
                and (row["continuation_pressure_score"] >= max(0.0, pressure_enter - 0.06))
                and (row["continuation_break_risk_p"] <= break_risk_cap + 0.03)
                and (row["continuation_benchmark_score"] >= 0.30)
            )
            continuation_on = bool(sustain_ok)

        activated_now = False
        if not continuation_on and hard_guard:
            continuation_on = True
            activated_now = True

        out.loc[dt, "continuation_guard_pass"] = float(hard_guard)

        if continuation_on:
            trigger_intensity = _ramp_value(row["continuation_trigger_score"], trigger_enter, trigger_ceiling)
            pressure_intensity = _ramp_value(row["continuation_pressure_score"], pressure_enter, pressure_ceiling)
            soft_intensity = float(
                np.mean(
                    [
                        trigger_intensity,
                        pressure_intensity,
                        float(row["continuation_break_headroom"]),
                        float(row["continuation_structural_headroom"]),
                        float(row["continuation_benchmark_score"]),
                    ]
                )
            )
            soft_intensity = float(np.clip(soft_intensity, 0.0, 1.0))
            out.loc[dt, "continuation_soft_intensity"] = soft_intensity
            out.loc[dt, "override_type"] = "CONTINUATION_LIFT"
            out.loc[dt, "override_detail"] = "CONTINUATION_LIFT"
            out.loc[dt, "defense_blend"] = 0.0
            out.loc[dt, "gate_scale"] = 1.0 + max(0.0, float(cfg.continuation_gate) - 1.0) * soft_intensity
            out.loc[dt, "vol_mult"] = 1.0 + max(0.0, float(cfg.continuation_vol_mult) - 1.0) * soft_intensity
            out.loc[dt, "exp_cap"] = 1.0 + max(0.0, float(cfg.continuation_exp_cap) - 1.0) * soft_intensity
            out.loc[dt, "is_override"] = 1.0
            out.loc[dt, "is_continuation_lift"] = 1.0
            out.loc[dt, "is_continuation_activation"] = float(activated_now)

    return out


def weekly_to_daily_override(override_weekly: pd.DataFrame, idx: pd.DatetimeIndex, cfg: Mahoraga14Config) -> pd.DataFrame:
    cols = [
        "override_type",
        "override_detail",
        "defense_blend",
        "gate_scale",
        "vol_mult",
        "exp_cap",
        "is_override",
        "is_structural_override",
        "is_continuation_lift",
        "is_continuation_activation",
        "structural_score",
        "structural_p",
        "continuation_trigger_p",
        "continuation_pressure_p",
        "continuation_break_risk_p",
        "continuation_trigger_score",
        "continuation_pressure_score",
        "continuation_pressure",
        "continuation_structural_headroom",
        "continuation_break_headroom",
        "continuation_soft_intensity",
        "continuation_trigger_context",
        "continuation_pressure_context",
        "continuation_compression_score",
        "continuation_pause_score",
        "continuation_support_score",
        "continuation_benchmark_score",
        "continuation_structural_health_score",
        "continuation_compression_valid",
        "continuation_pause_valid",
        "continuation_benchmark_valid",
        "continuation_trigger_valid",
        "continuation_structural_low",
        "continuation_guard_pass",
    ]
    out = override_weekly[cols].reindex(idx).ffill()
    activation_event = override_weekly["is_continuation_activation"].reindex(idx).fillna(0.0)
    out["defense_blend"] = out["defense_blend"].fillna(0.0).clip(0.0, 1.0)
    out["gate_scale"] = out["gate_scale"].fillna(1.0).clip(0.0, cfg.max_gate_scale)
    out["vol_mult"] = out["vol_mult"].fillna(1.0)
    out["exp_cap"] = out["exp_cap"].fillna(1.0)
    out["is_override"] = out["is_override"].fillna(0.0)
    out["is_structural_override"] = out["is_structural_override"].fillna(0.0)
    out["is_continuation_lift"] = out["is_continuation_lift"].fillna(0.0)
    out["is_continuation_activation"] = activation_event
    out["override_type"] = out["override_type"].fillna("BASELINE")
    out["override_detail"] = out["override_detail"].fillna("BASELINE")
    for col in [
        "structural_score",
        "structural_p",
        "continuation_trigger_p",
        "continuation_pressure_p",
        "continuation_break_risk_p",
        "continuation_trigger_score",
        "continuation_pressure_score",
        "continuation_pressure",
        "continuation_structural_headroom",
        "continuation_break_headroom",
        "continuation_soft_intensity",
        "continuation_trigger_context",
        "continuation_pressure_context",
        "continuation_compression_score",
        "continuation_pause_score",
        "continuation_support_score",
        "continuation_benchmark_score",
        "continuation_structural_health_score",
        "continuation_compression_valid",
        "continuation_pause_valid",
        "continuation_benchmark_valid",
        "continuation_trigger_valid",
        "continuation_structural_low",
        "continuation_guard_pass",
    ]:
        out[col] = out[col].fillna(0.0)
    return out
