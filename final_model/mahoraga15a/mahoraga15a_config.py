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
    variant: str = "15A1"
    label: str = "MAHORAGA_15A1"
    outputs_dir: str = field(default_factory=lambda: str(_project_root() / "mahoraga15a1_outputs"))
    plots_dir: str = field(default_factory=lambda: str(_project_root() / "mahoraga15a1_outputs" / "figures"))

    official_long_variant_key: str = "CONTINUATION_PRESSURE_V2_ONLY"
    official_long_label: str = "MAHORAGA_14_1_LONG_ONLY"
    delevered_label: str = "DELEVERED_CONTROL"
    ls_label: str = "MAHORAGA_15A1_LS"
    future_idio_short_label: str = "SPARSE_IDIO_SHORT_SLEEVE"
    require_official_long_guardrail: bool = True
    freeze_rebuild_tol: float = 1e-10

    hedge_beta_window: int = 63
    hedge_beta_min_obs: int = 42
    hedge_ridge_alpha: float = 4.0
    hedge_solver_ridge: float = 0.10
    hedge_turnover_penalty: float = 0.03
    hedge_max_single_name_share: float = 0.80
    hedge_beta_sanity_min: float = 0.35
    hedge_beta_sanity_max: float = 1.45
    hedge_directional_floor: float = 0.08
    hedge_permission_floor: float = 0.05

    allocator_norm_window: int = 126
    allocator_long_multiplier_floor: float = 0.74
    allocator_long_multiplier_ceiling: float = 1.00
    allocator_short_budget_benign_cap: float = 0.05
    allocator_short_budget_stress_cap: float = 0.15
    allocator_short_budget_crisis_cap: float = 0.35
    allocator_cash_floor: float = 0.05
    allocator_cash_target_ceiling: float = 0.32
    allocator_net_exposure_min: float = 0.25
    allocator_hysteresis_band: float = 0.005
    allocator_long_speed: float = 0.18
    allocator_long_step_cap: float = 0.055
    allocator_short_step_cap: float = 0.040
    allocator_short_up_speed_base: float = 0.14
    allocator_short_up_speed_hawkes: float = 0.28
    allocator_short_up_speed_break: float = 0.20
    allocator_short_down_speed_base: float = 0.08
    allocator_short_down_speed_recovery: float = 0.20
    allocator_short_down_speed_continuation: float = 0.10

    target_beta_qqq_low: float = 0.16
    target_beta_qqq_high: float = 0.34
    target_beta_spy_low: float = 0.12
    target_beta_spy_high: float = 0.28

    allocator_w_fragility: float = 0.24
    allocator_w_break_risk: float = 0.18
    allocator_w_benchmark_weakness: float = 0.16
    allocator_w_drawdown: float = 0.14
    allocator_w_corr_pressure: float = 0.08
    allocator_w_exposure_pressure: float = 0.06
    allocator_w_realized_vol: float = 0.08
    allocator_w_bear_persistence: float = 0.10
    allocator_w_continuation_relief: float = 0.18

    tsmom_fast_window: int = 21
    tsmom_slow_window: int = 63

    stress_extra_slippage: float = 0.0005
    stress_hedge_ratio_under_mult: float = 0.75
    stress_hedge_ratio_over_mult: float = 1.25
    stress_reaction_slower_mult: float = 0.50
    stress_reaction_faster_mult: float = 1.50
    stress_allocator_cap_mult: float = 0.90
    stress_allocator_long_mult: float = 0.95
    stress_delay_rebalances: int = 1

    mc_stationary_samples: int = 250
    mc_friction_samples: int = 250
    mc_block_size: int = 15
    mc_seed: int = 42
    mc_material_cagr_gap: float = 0.02
    mc_short_cap_multipliers: tuple = (0.90, 1.00, 1.10)
    mc_long_multipliers: tuple = (0.96, 1.00, 1.04)
    mc_speed_multipliers: tuple = (0.75, 1.00, 1.25)
    mc_hedge_ratio_multipliers: tuple = (0.85, 1.00, 1.15)

    success_beta_reduction_min: float = 0.10
    success_cagr_drop_max_pct: float = 1.50
    success_gross_short_min: float = 0.03
    success_crisis_gross_short_min: float = 0.08
    success_visible_sharpe_delta: float = 0.01
    success_visible_sortino_delta: float = 0.02
    fast_fail_similarity_sharpe_tol: float = 0.003
    fast_fail_similarity_cagr_tol_pct: float = 0.50
    fast_fail_similarity_beta_tol: float = 0.03
    fast_fail_sensitivity_tol: float = 1e-4

    report_name: str = "final_report_fast_ls.md"
