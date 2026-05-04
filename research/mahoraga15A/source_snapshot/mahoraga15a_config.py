from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
_M14_DIR = _PARENT_DIR / "mahoraga14"
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_PARENT_DIR) not in sys.path:
    sys.path.append(str(_PARENT_DIR))
if str(_M14_DIR) not in sys.path:
    sys.path.append(str(_M14_DIR))

from mahoraga14_config import Mahoraga14Config


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class Mahoraga15AConfig(Mahoraga14Config):
    variant: str = "15A3"
    label: str = "MAHORAGA_15A3"
    outputs_dir: str = field(default_factory=lambda: str(_project_root() / "mahoraga15a3_outputs"))
    plots_dir: str = field(default_factory=lambda: str(_project_root() / "mahoraga15a3_outputs" / "figures"))

    official_long_variant_key: str = "CONTINUATION_PRESSURE_V2_ONLY"
    official_long_label: str = "MAHORAGA_14_1_LONG_ONLY"
    delevered_label: str = "DELEVERED_CONTROL"
    ls_label: str = "MAHORAGA_15A3_LS"
    crash_hedge_label: str = "CRASH_HEDGE_SLEEVE"
    bear_hedge_label: str = "BEAR_HEDGE_SLEEVE"
    allocator_label: str = "IMPACT_ALLOCATOR_V3"
    future_idio_short_label: str = "SPARSE_IDIO_SHORT_SLEEVE"
    require_official_long_guardrail: bool = True
    freeze_rebuild_tol: float = 1e-10

    hedge_beta_window: int = 63
    hedge_beta_min_obs: int = 42
    hedge_ridge_alpha: float = 4.0
    hedge_solver_ridge: float = 0.10
    hedge_max_single_name_share: float = 0.80
    hedge_beta_sanity_min: float = 0.35
    hedge_beta_sanity_max: float = 1.45

    crash_floor_activation: float = 0.70
    crash_floor_power: float = 1.25
    crash_overlay_floor_scale: float = 1.00
    crash_beta_weight: float = 0.40
    crash_overlay_weight: float = 0.60
    bear_floor_activation: float = 0.52
    bear_floor_power: float = 1.10
    bear_overlay_floor_scale: float = 0.85
    bear_beta_weight: float = 0.55
    bear_overlay_weight: float = 0.45

    allocator_norm_window: int = 126
    allocator_long_multiplier_floor: float = 0.88
    allocator_long_multiplier_ceiling: float = 1.00
    allocator_cash_floor: float = 0.03
    allocator_cash_target_ceiling: float = 0.14
    allocator_net_exposure_floor_benign: float = 0.30
    allocator_net_exposure_floor_stress: float = 0.12
    allocator_net_exposure_floor_crisis: float = -0.12
    allocator_total_short_budget_cap: float = 0.35
    allocator_hysteresis_band: float = 0.005
    allocator_long_speed: float = 0.20
    allocator_long_step_cap: float = 0.035

    allocator_crash_budget_benign_cap: float = 0.00
    allocator_crash_budget_stress_cap: float = 0.08
    allocator_crash_budget_crisis_cap: float = 0.22
    allocator_bear_budget_benign_cap: float = 0.02
    allocator_bear_budget_stress_cap: float = 0.12
    allocator_bear_budget_crisis_cap: float = 0.18
    allocator_crash_step_cap: float = 0.085
    allocator_bear_step_cap: float = 0.045
    allocator_crash_velocity_decay: float = 0.40
    allocator_bear_velocity_decay: float = 0.72
    allocator_crash_shock_kick: float = 0.110
    allocator_bear_shock_kick: float = 0.030
    allocator_crash_release_decay: float = 0.95
    allocator_bear_release_decay: float = 0.72
    allocator_crash_speed_floor: float = 0.95
    allocator_crash_speed_ceiling: float = 2.40
    allocator_bear_speed_floor: float = 0.70
    allocator_bear_speed_ceiling: float = 1.55
    allocator_crash_up_speed_base: float = 0.18
    allocator_crash_up_speed_hawkes: float = 0.45
    allocator_crash_up_speed_break: float = 0.28
    allocator_crash_down_speed_base: float = 0.18
    allocator_crash_down_speed_recovery: float = 0.32
    allocator_bear_up_speed_base: float = 0.10
    allocator_bear_up_speed_persistence: float = 0.18
    allocator_bear_up_speed_fragility: float = 0.10
    allocator_bear_down_speed_base: float = 0.06
    allocator_bear_down_speed_recovery: float = 0.16
    allocator_bear_down_speed_continuation: float = 0.12

    target_beta_qqq_low: float = 0.12
    target_beta_qqq_high: float = 0.34
    target_beta_spy_low: float = 0.08
    target_beta_spy_high: float = 0.28

    allocator_w_fragility: float = 0.18
    allocator_w_break_risk: float = 0.16
    allocator_w_benchmark_weakness: float = 0.13
    allocator_w_drawdown: float = 0.10
    allocator_w_corr_pressure: float = 0.08
    allocator_w_exposure_pressure: float = 0.03
    allocator_w_realized_vol: float = 0.11
    allocator_w_bear_persistence: float = 0.13
    allocator_w_transition_shock: float = 0.14
    allocator_w_continuation_relief: float = 0.12
    allocator_crash_activation_threshold: float = 0.62
    allocator_crash_activation_slope: float = 8.0
    allocator_bear_activation_threshold: float = 0.52
    allocator_bear_activation_slope: float = 6.0
    allocator_crash_overlay_floor: float = 0.10
    allocator_bear_overlay_floor: float = 0.04

    tsmom_micro_window: int = 5
    tsmom_fast_window: int = 21
    tsmom_slow_window: int = 63

    stress_extra_slippage: float = 0.0005
    stress_hedge_ratio_under_mult: float = 0.75
    stress_hedge_ratio_over_mult: float = 1.25
    stress_reaction_slower_mult: float = 0.55
    stress_reaction_faster_mult: float = 1.45
    stress_allocator_cap_mult: float = 0.90
    stress_allocator_long_mult: float = 0.96
    stress_delay_rebalances: int = 1
    stress_delay_rebalances_2: int = 2
    stress_crash_lead_rebalances: int = -1
    stress_crash_lag_rebalances: int = 1
    stress_crash_lag_rebalances_2: int = 2
    stress_bear_lead_rebalances: int = -1
    stress_bear_lag_rebalances: int = 1
    stress_bear_lag_rebalances_2: int = 2

    mc_stationary_samples: int = 250
    mc_friction_samples: int = 250
    mc_block_size: int = 15
    mc_seed: int = 42
    mc_material_cagr_gap: float = 0.02
    mc_short_cap_multipliers: tuple = (0.90, 1.00, 1.10)
    mc_long_multipliers: tuple = (0.98, 1.00, 1.02)
    mc_speed_multipliers: tuple = (0.75, 1.00, 1.25)
    mc_hedge_ratio_multipliers: tuple = (0.85, 1.00, 1.15)

    success_beta_reduction_min: float = 0.10
    success_cagr_drop_max_pct: float = 2.00
    success_gross_short_min: float = 0.05
    success_crisis_gross_short_min: float = 0.08
    success_crash_window_min: float = 0.03
    success_bear_window_min: float = 0.03
    success_visible_sharpe_delta: float = 0.00
    success_visible_sortino_delta: float = 0.00
    fast_fail_similarity_sharpe_tol: float = 0.003
    fast_fail_similarity_cagr_tol_pct: float = 0.50
    fast_fail_similarity_beta_tol: float = 0.03
    fast_fail_sensitivity_tol: float = 5e-4
    fast_fail_timing_sharpe_tol: float = 0.01
    fast_fail_timing_cagr_tol_pct: float = 0.75
    fast_fail_timing_gross_short_tol: float = 0.01

    report_name: str = "final_report_fast_ls.md"
