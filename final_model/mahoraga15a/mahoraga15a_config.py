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
    variant: str = "15A"
    label: str = "MAHORAGA_15A"
    outputs_dir: str = field(default_factory=lambda: str(_project_root() / "mahoraga15a_outputs"))
    plots_dir: str = field(default_factory=lambda: str(_project_root() / "mahoraga15a_outputs" / "figures"))

    official_long_variant_key: str = "CONTINUATION_PRESSURE_V2_ONLY"
    official_long_label: str = "MAHORAGA_14_1_LONG_ONLY"
    ls_label: str = "MAHORAGA_15A_LS"
    future_idio_short_label: str = "SPARSE_IDIO_SHORT_SLEEVE"
    require_official_long_guardrail: bool = True
    freeze_rebuild_tol: float = 1e-10

    hedge_beta_window: int = 63
    hedge_beta_min_obs: int = 42
    hedge_ridge_alpha: float = 4.0
    hedge_solver_ridge: float = 0.10
    hedge_turnover_penalty: float = 0.03
    hedge_max_single_name_share: float = 0.80

    allocator_norm_window: int = 126
    allocator_long_multiplier_floor: float = 0.72
    allocator_long_multiplier_ceiling: float = 1.00
    allocator_short_budget_benign_cap: float = 0.05
    allocator_short_budget_crisis_cap: float = 0.35
    allocator_cash_floor: float = 0.05
    allocator_net_exposure_min: float = 0.25
    allocator_hysteresis_band: float = 0.005
    allocator_long_speed: float = 0.16
    allocator_short_up_speed_base: float = 0.08
    allocator_short_up_speed_hawkes: float = 0.24
    allocator_short_up_speed_break: float = 0.18
    allocator_short_down_speed_base: float = 0.05
    allocator_short_down_speed_recovery: float = 0.14
    allocator_short_down_speed_continuation: float = 0.08

    target_beta_qqq_low: float = 0.20
    target_beta_qqq_high: float = 0.35
    target_beta_spy_low: float = 0.15
    target_beta_spy_high: float = 0.30

    allocator_w_fragility: float = 0.30
    allocator_w_break_risk: float = 0.20
    allocator_w_benchmark_weakness: float = 0.18
    allocator_w_drawdown: float = 0.16
    allocator_w_corr_pressure: float = 0.10
    allocator_w_exposure_pressure: float = 0.06
    allocator_w_continuation_relief: float = 0.24

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

    report_name: str = "final_report_fast_ls.md"
