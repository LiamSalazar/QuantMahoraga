from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from mahoraga15a_config import Mahoraga15AConfig
from mahoraga15a_utils import clip01


def build_allocator_controls(state: pd.DataFrame, cfg: Mahoraga15AConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=state.index)
    stress = (
        cfg.allocator_w_fragility * state["structural_fragility"]
        + cfg.allocator_w_break_risk * state["break_risk"]
        + cfg.allocator_w_benchmark_weakness * state["benchmark_weakness"]
        + cfg.allocator_w_drawdown * state["drawdown_pressure"]
        + cfg.allocator_w_corr_pressure * state["corr_pressure"]
        + cfg.allocator_w_exposure_pressure * state["exposure_pressure"]
        - cfg.allocator_w_continuation_relief * state["continuation_relief"]
    )
    out["stress_intensity"] = clip01(stress)
    out["target_beta_qqq"] = cfg.target_beta_qqq_high - (cfg.target_beta_qqq_high - cfg.target_beta_qqq_low) * out["stress_intensity"]
    out["target_beta_spy"] = cfg.target_beta_spy_high - (cfg.target_beta_spy_high - cfg.target_beta_spy_low) * out["stress_intensity"]
    out["long_multiplier_target"] = (
        cfg.allocator_long_multiplier_ceiling
        - (cfg.allocator_long_multiplier_ceiling - cfg.allocator_long_multiplier_floor) * out["stress_intensity"]
        + 0.08 * state["continuation_relief"]
    ).clip(cfg.allocator_long_multiplier_floor, cfg.allocator_long_multiplier_ceiling)
    out["short_cap_dynamic"] = (
        cfg.allocator_short_budget_benign_cap
        + (cfg.allocator_short_budget_crisis_cap - cfg.allocator_short_budget_benign_cap)
        * clip01(0.65 * out["stress_intensity"] + 0.35 * state["break_risk"])
    ).clip(cfg.allocator_short_budget_benign_cap, cfg.allocator_short_budget_crisis_cap)
    out["short_up_speed"] = (
        cfg.allocator_short_up_speed_base
        + cfg.allocator_short_up_speed_hawkes * state["hawkes_stress"]
        + cfg.allocator_short_up_speed_break * state["break_risk"]
    ).clip(0.05, 0.90)
    out["short_down_speed"] = (
        cfg.allocator_short_down_speed_base
        + cfg.allocator_short_down_speed_recovery * state["hawkes_recovery"]
        + cfg.allocator_short_down_speed_continuation * state["continuation_relief"]
    ).clip(0.03, 0.60)
    out["cash_floor"] = cfg.allocator_cash_floor
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
        long_mult = prev_long_mult + cfg.allocator_long_speed * (long_mult_target - prev_long_mult)
        long_mult = float(np.clip(long_mult, cfg.allocator_long_multiplier_floor, cfg.allocator_long_multiplier_ceiling))

        raw_short = float(row["raw_short_budget"])
        if raw_short > prev_short + cfg.allocator_hysteresis_band:
            short_speed = float(row["short_up_speed"])
            short_target = prev_short + short_speed * (raw_short - prev_short)
        elif raw_short < prev_short - cfg.allocator_hysteresis_band:
            short_speed = float(row["short_down_speed"])
            short_target = prev_short + short_speed * (raw_short - prev_short)
        else:
            short_speed = 0.0
            short_target = prev_short

        gross_long_native = float(row["gross_long_native"])
        long_budget = gross_long_native * long_mult
        max_short_by_cap = float(row["short_cap_dynamic"])
        max_short_by_cash = max(0.0, 1.0 - float(row["cash_floor"]) - long_budget)
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
        final_beta_qqq = float(row["long_beta_qqq"] * long_mult - (final_qqq + final_spy * row["spy_beta_qqq"]))
        final_beta_spy = float(row["long_beta_spy"] * long_mult - (final_qqq * row["qqq_beta_spy"] + final_spy))
        cash_buffer = max(0.0, 1.0 - long_budget - short_budget)

        final_rows.append(
            {
                "Date": dt,
                "long_multiplier": long_mult,
                "long_budget": long_budget,
                "systematic_short_budget": short_budget,
                "cash_buffer": cash_buffer,
                "qqq_short_budget": final_qqq,
                "spy_short_budget": final_spy,
                "predicted_beta_qqq": final_beta_qqq,
                "predicted_beta_spy": final_beta_spy,
                "short_speed_applied": short_speed,
            }
        )
        prev_short = short_budget
        prev_long_mult = long_mult

    final_df = pd.DataFrame(final_rows).set_index("Date")
    return pd.concat([out, final_df], axis=1)
