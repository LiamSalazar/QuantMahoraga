from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import mahoraga6_1 as m6


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class Mahoraga14Config(m6.Mahoraga6Config):
    variant: str = "14.3"
    label: str = "MAHORAGA_14_3"
    cache_dir: str = field(default_factory=lambda: str(_project_root() / "data_cache"))
    outputs_dir: str = field(default_factory=lambda: str(_project_root() / "mahoraga14_3_outputs"))
    plots_dir: str = field(default_factory=lambda: str(_project_root() / "mahoraga14_3_plots"))
    official_baseline_label: str = "BASE_ALPHA_V2"
    historical_benchmark_label: str = "LEGACY"
    model_label: str = "M14_3_LONG_PARTICIPATION"
    main_variant_key: str = "STRUCTURAL_DEFENSE_ONLY"
    continuation_variant_key: str = "CONTINUATION_PRESSURE_V2_ONLY"
    combo_variant_key: str = "MAHORAGA14_3_LONG_PARTICIPATION"
    full_primary_variant_key: str = "MAHORAGA14_3_LONG_PARTICIPATION"
    primary_variant_key: str = "MAHORAGA14_3_LONG_PARTICIPATION"
    control_variant_key: str = "CONTINUATION_PRESSURE_V2_ONLY"

    weight_cap: float = 0.60
    k_atr: float = 3.0
    turb_zscore_thr: float = 1.5
    turb_scale_min: float = 0.30
    vol_target_ann: float = 0.30

    run_mode: str = "FAST"
    make_plots_flag: bool = False
    decision_freq: str = "W-FRI"
    outer_parallel: bool = True
    outer_backend: str = "loky"
    max_outer_jobs: int = 2

    fast_folds: Tuple[int, ...] = (1, 2, 3, 4, 5)
    full_folds: Tuple[int, ...] = (1, 2, 3, 4, 5)
    floor_folds: Tuple[int, ...] = (3, 5)
    ceiling_folds: Tuple[int, ...] = (1, 2, 4)

    inner_val_frac: float = 0.30
    min_train_weeks: int = 78
    bhy_alpha: float = 0.05
    top_support_candidates: int = 5

    ceiling_base_sharpe_tol: float = -0.03
    ceiling_main_sharpe_tol: float = -0.002
    ceiling_override_rate_cap: float = 0.16
    floor_main_sharpe_floor: float = 0.0
    floor_override_rate_cap: float = 0.42
    max_gate_scale: float = 1.42

    hard_fold4_sharpe_tol: float = -0.03
    hard_ceiling_mean_sharpe_tol: float = -0.03
    hard_fold3_vs_main_tol: float = -0.02
    hard_fold5_sharpe_min_delta: float = 0.0
    hard_reentry_rate_min: float = 0.01
    hard_override_rate_abs_cap: float = 0.30
    hard_override_rate_buffer: float = 0.08
    promising_stitched_sharpe_delta: float = 0.01

    residual_beta_window: int = 63
    residual_beta_min_obs: int = 42
    residual_ridge_alpha: float = 8.0
    residual_return_clip: float = 0.25
    residual_spans_short: Tuple[int, ...] = (21, 42)
    residual_spans_long: Tuple[int, ...] = (63, 126)
    residual_mom_windows: Tuple[int, ...] = (21, 63, 126)
    residual_burn_in: int = 126
    raw_weight_shrink: float = 0.70
    resid_weight_shrink: float = 0.60
    tech_factor_shrink: float = 0.15

    use_hawkes: bool = True
    hawkes_event_q_low: float = 0.25
    hawkes_event_q_high: float = 0.75
    hawkes_decay: float = 0.70

    use_corr_as_secondary_veto: bool = True
    corr_secondary_rho: float = 0.90
    corr_secondary_scale: float = 0.80

    enable_rf_challenger: bool = True
    rf_n_estimators: int = 250
    rf_max_depth: int = 4
    rf_min_samples_leaf: int = 10
    rf_random_state: int = 42

    continuation_target_rate: float = 0.09
    continuation_min_rate: float = 0.02
    continuation_rate_cap: float = 0.18
    continuation_trigger_floor_quantile: float = 0.60
    continuation_trigger_ceiling_quantile: float = 0.85
    continuation_pressure_floor_quantile: float = 0.62
    continuation_pressure_ceiling_quantile: float = 0.86
    continuation_break_risk_floor_quantile: float = 0.20
    continuation_break_risk_cap_quantile: float = 0.60
    continuation_structural_margin: float = 0.06
    continuation_gate: float = 1.05
    continuation_vol_mult: float = 1.05
    continuation_exp_cap: float = 1.07

    participation_long_budget_floor: float = 0.78
    participation_long_budget_base: float = 0.94
    participation_long_budget_ceiling: float = 1.00
    participation_gate_max: float = 1.42
    participation_vol_mult_max: float = 1.32
    participation_exp_cap_max: float = 1.48
    participation_leader_blend_max: float = 0.78
    participation_mix_lift: float = 0.26
    participation_beta_penalty_mult: float = 0.08
    participation_raw_rel_boost_lift: float = 0.36
    participation_cash_redeploy_scale_cap: float = 1.85
    participation_max_name_weight: float = 0.82
    participation_redeploy_headroom_floor: float = 0.04
    participation_allocator_min_conviction: float = 0.55
    participation_allocator_high_conviction: float = 0.80
    participation_allocator_cash_target_floor: float = 0.92
    participation_allocator_cash_target_ceiling: float = 0.99
    conviction_activation_threshold: float = 0.62
    conviction_max_budget_boost: float = 0.08
    conviction_gate_boost_max: float = 0.12
    conviction_vol_boost_max: float = 0.08
    conviction_exp_boost_max: float = 0.15
    conviction_leader_boost_max: float = 0.16
    conviction_weight_scale_max: float = 1.35
    leader_top_k: int = 12
    leader_return_lookback: int = 21
    leader_gap_floor: float = 0.0025
    leader_tilt_strength_max: float = 0.22
    leader_top_weight_floor: float = 0.55
    risk_backoff_budget_floor: float = 0.70
    risk_backoff_hard_budget: float = 0.45
    risk_backoff_gate_floor: float = 0.92
    risk_backoff_vol_floor: float = 0.93
    risk_backoff_exp_floor: float = 0.88
    risk_backoff_hard_break_risk: float = 0.82
    risk_backoff_hard_fragility: float = 0.82
    risk_backoff_hard_benchmark_weakness: float = 0.80
    bull_window_min_days: int = 63
    bull_window_min_return: float = 0.10
    bull_window_max_drawdown: float = -0.12
    bull_window_min_efficiency: float = 0.45
    diagnostics_bootstrap_samples: int = 300
    diagnostics_bootstrap_block: int = 20
    diagnostics_allocator_tighter_mult: float = 0.85
    diagnostics_allocator_looser_mult: float = 1.15
    diagnostics_bull_weaker_mult: float = 0.80
    diagnostics_bull_stronger_mult: float = 1.20
    diagnostics_conviction_weaker_mult: float = 0.80
    diagnostics_conviction_stronger_mult: float = 1.20
    diagnostics_leader_weaker_mult: float = 0.75
    diagnostics_leader_stronger_mult: float = 1.25

    def mode_folds(self) -> Tuple[int, ...]:
        return self.fast_folds if self.run_mode.upper() == "FAST" else self.full_folds

    def engine_grid(self) -> Dict[str, Tuple[float, ...]]:
        if self.run_mode.upper() == "FAST":
            return {
                "base_mix": (0.18, 0.30),
                "defense_mix": (0.45, 0.60),
                "base_beta_penalty": (0.00,),
                "defense_beta_penalty": (0.05,),
                "raw_rel_boost": (1.00, 1.10),
            }
        # FULL 14.1 is an audit mode, not a combinatorial discovery mode.
        # Keeping calibration on the same focused grid as FAST preserves
        # comparability and prevents the prior 78k-combos-per-fold blow-up.
        return {
            "base_mix": (0.18, 0.30),
            "defense_mix": (0.45, 0.60),
            "base_beta_penalty": (0.00,),
            "defense_beta_penalty": (0.05,),
            "raw_rel_boost": (1.00, 1.10),
        }

    def policy_grid(self) -> Dict[str, Tuple[float, ...]]:
        if self.run_mode.upper() == "FAST":
            return {
                "structural_enter_thr": (0.76, 0.82),
                "hawkes_weight": (0.10, 0.18),
                "structural_blend": (0.30, 0.45),
                "structural_gate": (0.88,),
                "structural_exp_cap": (0.80, 0.86),
            }
        return {
            "structural_enter_thr": (0.76, 0.82),
            "hawkes_weight": (0.10, 0.18),
            "structural_blend": (0.30, 0.45),
            "structural_gate": (0.88,),
            "structural_exp_cap": (0.80, 0.86),
        }
