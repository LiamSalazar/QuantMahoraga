from __future__ import annotations

import numpy as np
import pandas as pd

from mahoraga15a_config import Mahoraga15AConfig
from mahoraga15a_utils import clip01


def build_allocator_controls(state: pd.DataFrame, cfg: Mahoraga15AConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=state.index)

    stress_core = (
        cfg.allocator_w_fragility * state["structural_fragility"]
        + cfg.allocator_w_break_risk * state["break_risk"]
        + cfg.allocator_w_benchmark_weakness * state["benchmark_weakness"]
        + cfg.allocator_w_drawdown * state["drawdown_pressure"]
        + cfg.allocator_w_corr_pressure * state["corr_pressure"]
        + cfg.allocator_w_exposure_pressure * state["exposure_pressure"]
        + cfg.allocator_w_realized_vol * state["realized_vol_pressure"]
        + cfg.allocator_w_bear_persistence * state["bear_persistence"]
        + cfg.allocator_w_transition_shock * state["transition_shock"]
        - cfg.allocator_w_continuation_relief * state["continuation_relief"]
    )
    out["stress_intensity"] = clip01(stress_core)
    out["directional_bear"] = clip01(0.40 * state["benchmark_weakness"] + 0.35 * state["bear_persistence"] + 0.25 * state["trend_weakness"])
    out["crash_activation"] = clip01(state["crash_activation"])
    out["bear_activation"] = clip01(state["bear_activation"])
    out["crash_transition"] = clip01(0.50 * state["transition_shock"] + 0.25 * state["hawkes_stress"] + 0.15 * state["corr_spike"] + 0.10 * state["break_risk_jump"])
    out["bear_transition"] = clip01(0.55 * state["bear_persistence"] + 0.25 * state["benchmark_weakness"] + 0.20 * state["trend_weakness"])
    out["crisis_persistence"] = clip01(0.50 * state["bear_persistence"] + 0.30 * state["drawdown_pressure"] + 0.20 * state["benchmark_weakness"])
    out["combined_risk"] = clip01(
        0.42 * out["crash_activation"]
        + 0.38 * out["bear_activation"]
        + 0.12 * state["realized_vol_pressure"]
        + 0.08 * state["drawdown_pressure"]
    )

    beta_target_mix = clip01(0.38 * out["crash_activation"] + 0.42 * out["bear_activation"] + 0.20 * out["directional_bear"])
    out["target_beta_qqq"] = cfg.target_beta_qqq_high - (cfg.target_beta_qqq_high - cfg.target_beta_qqq_low) * beta_target_mix
    out["target_beta_spy"] = cfg.target_beta_spy_high - (cfg.target_beta_spy_high - cfg.target_beta_spy_low) * beta_target_mix
    out["beta_gap_qqq"] = np.maximum(0.0, state["long_beta_qqq"] - out["target_beta_qqq"])
    out["beta_gap_spy"] = np.maximum(0.0, state["long_beta_spy"] - out["target_beta_spy"])
    out["beta_gap_score"] = clip01(
        0.55 * clip01(out["beta_gap_qqq"] / max(1e-6, cfg.target_beta_qqq_high))
        + 0.45 * clip01(out["beta_gap_spy"] / max(1e-6, cfg.target_beta_spy_high))
    )

    out["crash_permission"] = clip01(
        0.36 * out["crash_activation"]
        + 0.24 * out["crash_transition"]
        + 0.14 * state["corr_spike"]
        + 0.14 * state["fragility_jump"]
        + 0.12 * out["beta_gap_score"]
    )
    out["bear_permission"] = clip01(
        0.34 * out["bear_activation"]
        + 0.26 * out["bear_transition"]
        + 0.16 * state["structural_fragility"]
        + 0.14 * state["break_risk"]
        + 0.10 * out["beta_gap_score"]
    )
    out["hedge_permission"] = clip01(0.50 * out["crash_permission"] + 0.50 * out["bear_permission"])

    long_de_risk_pressure = clip01(
        0.28 * out["crash_activation"]
        + 0.25 * out["bear_activation"]
        + 0.18 * state["realized_vol_pressure"]
        + 0.12 * state["drawdown_pressure"]
        + 0.10 * out["beta_gap_score"]
        + 0.07 * state["corr_pressure"]
    )
    out["long_multiplier_target"] = (
        cfg.allocator_long_multiplier_ceiling
        - (cfg.allocator_long_multiplier_ceiling - cfg.allocator_long_multiplier_floor) * long_de_risk_pressure
        + 0.03 * state["continuation_relief"]
    ).clip(cfg.allocator_long_multiplier_floor, cfg.allocator_long_multiplier_ceiling)

    crash_regime_mix = clip01(
        0.42 * out["crash_activation"]
        + 0.24 * out["crash_transition"]
        + 0.14 * state["hawkes_stress"]
        + 0.10 * state["corr_spike"]
        + 0.10 * state["break_risk_jump"]
    )
    out["crash_cap_dynamic"] = (
        cfg.allocator_crash_budget_benign_cap
        + (cfg.allocator_crash_budget_crisis_cap - cfg.allocator_crash_budget_benign_cap) * crash_regime_mix
    ).clip(cfg.allocator_crash_budget_benign_cap, cfg.allocator_crash_budget_crisis_cap)
    out["crash_cap_dynamic"] = np.maximum(out["crash_cap_dynamic"], cfg.allocator_crash_overlay_floor * out["crash_activation"])

    bear_regime_mix = clip01(
        0.38 * out["bear_activation"]
        + 0.24 * out["bear_transition"]
        + 0.18 * state["benchmark_weakness"]
        + 0.10 * state["structural_fragility"]
        + 0.10 * state["continuation_failure"]
    )
    out["bear_cap_dynamic"] = (
        cfg.allocator_bear_budget_benign_cap
        + (cfg.allocator_bear_budget_crisis_cap - cfg.allocator_bear_budget_benign_cap) * bear_regime_mix
    ).clip(cfg.allocator_bear_budget_benign_cap, cfg.allocator_bear_budget_crisis_cap)
    out["bear_cap_dynamic"] = np.maximum(out["bear_cap_dynamic"], cfg.allocator_bear_overlay_floor * out["bear_activation"])
    out["total_short_cap_dynamic"] = np.minimum(cfg.allocator_total_short_budget_cap, out["crash_cap_dynamic"] + out["bear_cap_dynamic"])

    out["cash_target"] = (
        cfg.allocator_cash_floor
        + 0.035 * out["crash_activation"]
        + 0.030 * out["bear_activation"]
        + 0.020 * state["realized_vol_pressure"]
        + 0.010 * state["corr_pressure"]
        - 0.020 * state["continuation_relief"]
    ).clip(cfg.allocator_cash_floor, cfg.allocator_cash_target_ceiling)

    out["net_exposure_floor_dynamic"] = np.select(
        [out["combined_risk"] < 0.35, out["combined_risk"] < 0.70],
        [cfg.allocator_net_exposure_floor_benign, cfg.allocator_net_exposure_floor_stress],
        default=cfg.allocator_net_exposure_floor_crisis,
    )
    out["net_exposure_floor_dynamic"] = np.minimum(
        out["net_exposure_floor_dynamic"],
        cfg.allocator_net_exposure_floor_benign - 0.42 * out["combined_risk"],
    )

    out["crash_reaction_multiplier"] = (
        cfg.allocator_crash_speed_floor
        + (cfg.allocator_crash_speed_ceiling - cfg.allocator_crash_speed_floor)
        * clip01(0.55 * out["crash_transition"] + 0.25 * state["hawkes_stress"] + 0.20 * state["break_risk_jump"])
    ).clip(cfg.allocator_crash_speed_floor, cfg.allocator_crash_speed_ceiling)
    out["bear_reaction_multiplier"] = (
        cfg.allocator_bear_speed_floor
        + (cfg.allocator_bear_speed_ceiling - cfg.allocator_bear_speed_floor)
        * clip01(0.50 * out["bear_transition"] + 0.25 * state["structural_fragility"] + 0.25 * state["benchmark_weakness"])
    ).clip(cfg.allocator_bear_speed_floor, cfg.allocator_bear_speed_ceiling)

    out["crash_up_speed"] = (
        cfg.allocator_crash_up_speed_base
        + cfg.allocator_crash_up_speed_hawkes * state["hawkes_stress"]
        + cfg.allocator_crash_up_speed_break * state["break_risk_jump"]
    ).clip(0.08, 0.98)
    out["crash_down_speed"] = (
        cfg.allocator_crash_down_speed_base
        + cfg.allocator_crash_down_speed_recovery * state["hawkes_recovery"]
    ).clip(0.08, 0.95)
    out["bear_up_speed"] = (
        cfg.allocator_bear_up_speed_base
        + cfg.allocator_bear_up_speed_persistence * out["bear_transition"]
        + cfg.allocator_bear_up_speed_fragility * state["structural_fragility"]
    ).clip(0.05, 0.75)
    out["bear_down_speed"] = (
        cfg.allocator_bear_down_speed_base
        + cfg.allocator_bear_down_speed_recovery * state["hawkes_recovery"]
        + cfg.allocator_bear_down_speed_continuation * state["continuation_relief"]
    ).clip(0.04, 0.60)
    return out


def _next_budget(
    raw_budget: float,
    prev_budget: float,
    prev_velocity: float,
    up_speed: float,
    down_speed: float,
    reaction_multiplier: float,
    shock_kick: float,
    release_decay: float,
    activation: float,
    trigger_signal: float,
    continuation_relief: float,
    velocity_decay: float,
    step_cap: float,
) -> tuple[float, float, float]:
    error = raw_budget - prev_budget
    if error >= 0.0:
        desired_step = up_speed * reaction_multiplier * error + shock_kick * trigger_signal * max(raw_budget, prev_budget)
    else:
        desired_step = (down_speed / max(1.0, 0.5 * reaction_multiplier)) * error
        if activation < 0.35:
            desired_step -= (1.0 - activation) * (1.0 + continuation_relief) * release_decay * prev_budget
    velocity = velocity_decay * prev_velocity + desired_step
    velocity = float(np.clip(velocity, -step_cap, step_cap))
    budget = max(0.0, prev_budget + velocity)
    return budget, velocity, error


def finalize_allocator_trace(
    state: pd.DataFrame,
    hedge_plan: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: Mahoraga15AConfig,
) -> pd.DataFrame:
    out = state.copy()
    for frame in (controls, hedge_plan):
        for col in frame.columns:
            out[col] = frame[col]

    prev_long_mult = 1.0
    prev_crash_budget = 0.0
    prev_bear_budget = 0.0
    prev_crash_velocity = 0.0
    prev_bear_velocity = 0.0
    prev_crash_mix = np.array([0.5, 0.5], dtype=float)
    prev_bear_mix = np.array([0.5, 0.5], dtype=float)
    final_rows = []

    for dt, row in out.iterrows():
        long_mult_target = float(row["long_multiplier_target"])
        long_mult_step = cfg.allocator_long_speed * (long_mult_target - prev_long_mult)
        long_mult_step = float(np.clip(long_mult_step, -cfg.allocator_long_step_cap, cfg.allocator_long_step_cap))
        long_mult = float(np.clip(prev_long_mult + long_mult_step, cfg.allocator_long_multiplier_floor, cfg.allocator_long_multiplier_ceiling))

        crash_budget_pre, crash_velocity, crash_error = _next_budget(
            raw_budget=float(row["raw_crash_short_budget"]),
            prev_budget=prev_crash_budget,
            prev_velocity=prev_crash_velocity,
            up_speed=float(row["crash_up_speed"]),
            down_speed=float(row["crash_down_speed"]),
            reaction_multiplier=float(row["crash_reaction_multiplier"]),
            shock_kick=cfg.allocator_crash_shock_kick,
            release_decay=cfg.allocator_crash_release_decay,
            activation=float(row["crash_activation"]),
            trigger_signal=float(row["crash_transition"]),
            continuation_relief=float(row["continuation_relief"]),
            velocity_decay=cfg.allocator_crash_velocity_decay,
            step_cap=cfg.allocator_crash_step_cap,
        )
        bear_budget_pre, bear_velocity, bear_error = _next_budget(
            raw_budget=float(row["raw_bear_short_budget"]),
            prev_budget=prev_bear_budget,
            prev_velocity=prev_bear_velocity,
            up_speed=float(row["bear_up_speed"]),
            down_speed=float(row["bear_down_speed"]),
            reaction_multiplier=float(row["bear_reaction_multiplier"]),
            shock_kick=cfg.allocator_bear_shock_kick,
            release_decay=cfg.allocator_bear_release_decay,
            activation=float(row["bear_activation"]),
            trigger_signal=float(row["bear_transition"]),
            continuation_relief=float(row["continuation_relief"]),
            velocity_decay=cfg.allocator_bear_velocity_decay,
            step_cap=cfg.allocator_bear_step_cap,
        )

        gross_long_native = float(row["gross_long_native"])
        long_budget = max(0.0, gross_long_native * long_mult)
        cash_target = float(row["cash_target"])
        max_total_by_cash = max(0.0, 1.0 - cash_target - long_budget)
        max_total_by_net = max(0.0, long_budget - float(row["net_exposure_floor_dynamic"]))
        max_total_short = min(float(row["total_short_cap_dynamic"]), max_total_by_cash, max_total_by_net)

        crash_budget = float(np.clip(crash_budget_pre, 0.0, float(row["crash_cap_dynamic"])))
        bear_budget = float(np.clip(bear_budget_pre, 0.0, float(row["bear_cap_dynamic"])))
        total_budget = crash_budget + bear_budget
        if total_budget > max_total_short and total_budget > 1e-12:
            scale = max_total_short / total_budget
            crash_budget *= scale
            bear_budget *= scale
            total_budget = crash_budget + bear_budget

        raw_crash_total = max(0.0, float(row["raw_crash_short_qqq"]) + float(row["raw_crash_short_spy"]))
        raw_bear_total = max(0.0, float(row["raw_bear_short_qqq"]) + float(row["raw_bear_short_spy"]))
        if raw_crash_total > 1e-12:
            crash_mix = np.array([float(row["raw_crash_short_qqq"]), float(row["raw_crash_short_spy"])], dtype=float) / raw_crash_total
            prev_crash_mix = crash_mix
        else:
            crash_mix = prev_crash_mix
        if raw_bear_total > 1e-12:
            bear_mix = np.array([float(row["raw_bear_short_qqq"]), float(row["raw_bear_short_spy"])], dtype=float) / raw_bear_total
            prev_bear_mix = bear_mix
        else:
            bear_mix = prev_bear_mix

        crash_qqq = float(crash_budget * crash_mix[0])
        crash_spy = float(crash_budget * crash_mix[1])
        bear_qqq = float(bear_budget * bear_mix[0])
        bear_spy = float(bear_budget * bear_mix[1])
        qqq_short_budget = crash_qqq + bear_qqq
        spy_short_budget = crash_spy + bear_spy
        predicted_beta_qqq = float(row["long_beta_qqq"] * long_mult - (qqq_short_budget + spy_short_budget * row["spy_beta_qqq"]))
        predicted_beta_spy = float(row["long_beta_spy"] * long_mult - (qqq_short_budget * row["qqq_beta_spy"] + spy_short_budget))
        cash_buffer = max(0.0, 1.0 - long_budget - total_budget)
        net_exposure = long_budget - total_budget
        gross_exposure = long_budget + total_budget

        final_rows.append(
            {
                "Date": dt,
                "long_multiplier": long_mult,
                "long_budget": long_budget,
                "crash_short_budget": crash_budget,
                "bear_short_budget": bear_budget,
                "systematic_short_budget": total_budget,
                "cash_buffer": cash_buffer,
                "net_exposure": net_exposure,
                "gross_exposure": gross_exposure,
                "crash_qqq_short_budget": crash_qqq,
                "crash_spy_short_budget": crash_spy,
                "bear_qqq_short_budget": bear_qqq,
                "bear_spy_short_budget": bear_spy,
                "qqq_short_budget": qqq_short_budget,
                "spy_short_budget": spy_short_budget,
                "predicted_beta_qqq": predicted_beta_qqq,
                "predicted_beta_spy": predicted_beta_spy,
                "crash_speed_applied": float(row["crash_up_speed"] if crash_error >= 0.0 else row["crash_down_speed"]),
                "bear_speed_applied": float(row["bear_up_speed"] if bear_error >= 0.0 else row["bear_down_speed"]),
                "crash_reaction_multiplier_applied": float(row["crash_reaction_multiplier"]),
                "bear_reaction_multiplier_applied": float(row["bear_reaction_multiplier"]),
                "long_step_applied": long_mult_step,
                "crash_step_applied": crash_velocity,
                "bear_step_applied": bear_velocity,
                "crash_velocity_state": crash_velocity,
                "bear_velocity_state": bear_velocity,
            }
        )
        prev_long_mult = long_mult
        prev_crash_budget = crash_budget
        prev_bear_budget = bear_budget
        prev_crash_velocity = crash_velocity
        prev_bear_velocity = bear_velocity

    final_df = pd.DataFrame(final_rows).set_index("Date")
    return pd.concat([out, final_df], axis=1)
