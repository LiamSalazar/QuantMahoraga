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
        - cfg.allocator_w_continuation_relief * state["continuation_relief"]
    )
    out["stress_intensity"] = clip01(stress_core)
    out["directional_bear"] = clip01(0.60 * state["benchmark_weakness"] + 0.40 * state["bear_persistence"])

    beta_target_mix = clip01(0.60 * out["stress_intensity"] + 0.40 * out["directional_bear"])
    out["target_beta_qqq"] = cfg.target_beta_qqq_high - (cfg.target_beta_qqq_high - cfg.target_beta_qqq_low) * beta_target_mix
    out["target_beta_spy"] = cfg.target_beta_spy_high - (cfg.target_beta_spy_high - cfg.target_beta_spy_low) * beta_target_mix
    out["beta_gap_qqq"] = np.maximum(0.0, state["long_beta_qqq"] - out["target_beta_qqq"])
    out["beta_gap_spy"] = np.maximum(0.0, state["long_beta_spy"] - out["target_beta_spy"])
    out["beta_gap_score"] = clip01(
        0.55 * clip01(out["beta_gap_qqq"] / max(1e-6, cfg.target_beta_qqq_high))
        + 0.45 * clip01(out["beta_gap_spy"] / max(1e-6, cfg.target_beta_spy_high))
    )

    de_risk_pressure = clip01(
        0.40 * out["stress_intensity"]
        + 0.25 * state["realized_vol_pressure"]
        + 0.20 * out["beta_gap_score"]
        + 0.15 * state["drawdown_pressure"]
    )
    out["long_multiplier_target"] = (
        cfg.allocator_long_multiplier_ceiling
        - (cfg.allocator_long_multiplier_ceiling - cfg.allocator_long_multiplier_floor) * de_risk_pressure
        + 0.05 * state["continuation_relief"]
    ).clip(cfg.allocator_long_multiplier_floor, cfg.allocator_long_multiplier_ceiling)

    short_regime_mix = clip01(
        0.45 * out["stress_intensity"]
        + 0.25 * out["directional_bear"]
        + 0.20 * state["break_risk"]
        + 0.10 * out["beta_gap_score"]
    )
    out["short_cap_dynamic"] = (
        cfg.allocator_short_budget_benign_cap
        + (cfg.allocator_short_budget_crisis_cap - cfg.allocator_short_budget_benign_cap) * short_regime_mix
    ).clip(cfg.allocator_short_budget_benign_cap, cfg.allocator_short_budget_crisis_cap)
    out["short_cap_regime_band"] = np.select(
        [short_regime_mix < 0.35, short_regime_mix < 0.70],
        [cfg.allocator_short_budget_benign_cap, cfg.allocator_short_budget_stress_cap],
        default=cfg.allocator_short_budget_crisis_cap,
    )
    out["short_cap_dynamic"] = np.minimum(out["short_cap_dynamic"], out["short_cap_regime_band"])

    out["cash_target"] = (
        cfg.allocator_cash_floor
        + 0.10 * out["stress_intensity"]
        + 0.05 * state["break_risk"]
        + 0.03 * state["realized_vol_pressure"]
        - 0.03 * state["continuation_relief"]
    ).clip(cfg.allocator_cash_floor, cfg.allocator_cash_target_ceiling)

    out["short_up_speed"] = (
        cfg.allocator_short_up_speed_base
        + cfg.allocator_short_up_speed_hawkes * state["hawkes_stress"]
        + cfg.allocator_short_up_speed_break * state["break_risk"]
    ).clip(0.06, 0.92)
    out["short_down_speed"] = (
        cfg.allocator_short_down_speed_base
        + cfg.allocator_short_down_speed_recovery * state["hawkes_recovery"]
        + cfg.allocator_short_down_speed_continuation * state["continuation_relief"]
    ).clip(0.04, 0.75)
    return out


def finalize_allocator_trace(
    state: pd.DataFrame,
    hedge_plan: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: Mahoraga15AConfig,
) -> pd.DataFrame:
    out = pd.concat([state, controls, hedge_plan], axis=1)
    prev_short = 0.0
    prev_long_mult = 1.0
    prev_mix = np.array([0.5, 0.5], dtype=float)

    final_rows = []
    for dt, row in out.iterrows():
        long_mult_target = float(row["long_multiplier_target"])
        long_mult_step = cfg.allocator_long_speed * (long_mult_target - prev_long_mult)
        long_mult_step = float(np.clip(long_mult_step, -cfg.allocator_long_step_cap, cfg.allocator_long_step_cap))
        long_mult = float(np.clip(prev_long_mult + long_mult_step, cfg.allocator_long_multiplier_floor, cfg.allocator_long_multiplier_ceiling))

        raw_short = float(row["raw_short_budget"])
        if raw_short > prev_short + cfg.allocator_hysteresis_band:
            short_speed = float(row["short_up_speed"])
            desired_step = short_speed * (raw_short - prev_short)
        elif raw_short < prev_short - cfg.allocator_hysteresis_band:
            short_speed = float(row["short_down_speed"])
            desired_step = short_speed * (raw_short - prev_short)
        else:
            short_speed = 0.0
            desired_step = 0.0
        desired_step = float(np.clip(desired_step, -cfg.allocator_short_step_cap, cfg.allocator_short_step_cap))
        short_target = max(0.0, prev_short + desired_step)

        gross_long_native = float(row["gross_long_native"])
        long_budget = max(0.0, gross_long_native * long_mult)
        cash_target = float(row["cash_target"])
        max_short_by_cap = float(row["short_cap_dynamic"])
        max_short_by_cash = max(0.0, 1.0 - cash_target - long_budget)
        max_short_by_net = max(0.0, long_budget - cfg.allocator_net_exposure_min)
        short_budget = float(np.clip(short_target, 0.0, min(max_short_by_cap, max_short_by_cash, max_short_by_net)))

        raw_qqq = float(row["raw_short_qqq"])
        raw_spy = float(row["raw_short_spy"])
        raw_total = max(0.0, raw_qqq + raw_spy)
        if raw_total > 1e-12:
            mix = np.array([raw_qqq, raw_spy], dtype=float) / raw_total
            prev_mix = mix
        else:
            mix = prev_mix

        final_qqq = float(short_budget * mix[0])
        final_spy = float(short_budget * mix[1])
        predicted_beta_qqq = float(row["long_beta_qqq"] * long_mult - (final_qqq + final_spy * row["spy_beta_qqq"]))
        predicted_beta_spy = float(row["long_beta_spy"] * long_mult - (final_qqq * row["qqq_beta_spy"] + final_spy))
        cash_buffer = max(0.0, 1.0 - long_budget - short_budget)
        net_exposure = long_budget - short_budget
        gross_exposure = long_budget + short_budget

        final_rows.append(
            {
                "Date": dt,
                "long_multiplier": long_mult,
                "long_budget": long_budget,
                "systematic_short_budget": short_budget,
                "cash_buffer": cash_buffer,
                "net_exposure": net_exposure,
                "gross_exposure": gross_exposure,
                "qqq_short_budget": final_qqq,
                "spy_short_budget": final_spy,
                "predicted_beta_qqq": predicted_beta_qqq,
                "predicted_beta_spy": predicted_beta_spy,
                "short_speed_applied": short_speed,
                "long_step_applied": long_mult_step,
                "short_step_applied": desired_step,
            }
        )
        prev_short = short_budget
        prev_long_mult = long_mult

    final_df = pd.DataFrame(final_rows).set_index("Date")
    return pd.concat([out, final_df], axis=1)
