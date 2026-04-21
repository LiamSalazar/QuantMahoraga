from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import mahoraga6_1 as m6


@dataclass
class Mahoraga12Config(m6.Mahoraga6Config):
    variant: str = "12.0"
    label: str = "MAHORAGA_12"
    outputs_dir: str = "mahoraga12_outputs"
    plots_dir: str = "mahoraga12_plots"

    weight_cap: float = 0.60
    k_atr: float = 3.0
    turb_zscore_thr: float = 1.5
    turb_scale_min: float = 0.30
    vol_target_ann: float = 0.30

    run_mode: str = "FAST"  # FAST / FULL
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

    # Guard rails for candidate selection.
    ceiling_base_sharpe_tol: float = -0.03
    ceiling_model_sharpe_tol: float = -0.002
    ceiling_override_rate_cap: float = 0.16
    floor_model_sharpe_floor: float = 0.0
    floor_override_rate_cap: float = 0.42
    max_gate_scale: float = 1.10
    recovery_memory_weeks: int = 3

    # Alpha design.
    residual_beta_window: int = 63
    residual_spans_short: Tuple[int, ...] = (21, 42)
    residual_spans_long: Tuple[int, ...] = (63, 126)
    residual_mom_windows: Tuple[int, ...] = (21, 63, 126)
    residual_burn_in: int = 126
    raw_weight_shrink: float = 0.70
    resid_weight_shrink: float = 0.60

    # Hawkes only as a transition urgency signal.
    use_hawkes: bool = True
    hawkes_event_q_low: float = 0.25
    hawkes_event_q_high: float = 0.75
    hawkes_decay: float = 0.70

    # Secondary veto only.
    use_corr_as_secondary_veto: bool = True
    corr_secondary_rho: float = 0.90
    corr_secondary_scale: float = 0.80

    # Classifiers.
    enable_rf_challenger: bool = True
    rf_n_estimators: int = 250
    rf_max_depth: int = 4
    rf_min_samples_leaf: int = 10
    rf_random_state: int = 42

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
        return {
            "base_mix": (0.14, 0.22, 0.30, 0.38),
            "defense_mix": (0.40, 0.50, 0.62),
            "base_beta_penalty": (0.00, 0.01, 0.02),
            "defense_beta_penalty": (0.04, 0.06, 0.08),
            "raw_rel_boost": (1.00, 1.08, 1.16),
        }

    def policy_grid(self) -> Dict[str, Tuple[float, ...]]:
        if self.run_mode.upper() == "FAST":
            return {
                "structural_enter_thr": (0.76, 0.82),
                "transition_enter_thr": (0.68,),
                "recovery_enter_thr": (0.58, 0.64),
                "hawkes_weight": (0.10, 0.18),
                "structural_blend": (0.30, 0.45),
                "transition_blend": (0.08,),
                "structural_gate": (0.88,),
                "transition_gate": (0.90,),
                "recovery_gate": (1.00, 1.05),
                "transition_vol_mult": (0.92,),
                "recovery_vol_mult": (1.06,),
                "structural_exp_cap": (0.80,),
                "transition_exp_cap": (0.90,),
                "recovery_exp_cap": (1.06,),
            }
        return {
            "structural_enter_thr": (0.70, 0.76, 0.82),
            "transition_enter_thr": (0.62, 0.68, 0.74),
            "recovery_enter_thr": (0.54, 0.60, 0.66),
            "hawkes_weight": (0.08, 0.14, 0.20),
            "structural_blend": (0.25, 0.35, 0.50),
            "transition_blend": (0.05, 0.08, 0.12),
            "structural_gate": (0.80, 0.86, 0.92),
            "transition_gate": (0.88, 0.92, 0.96),
            "recovery_gate": (1.00, 1.04, 1.08),
            "transition_vol_mult": (0.88, 0.92, 0.98),
            "recovery_vol_mult": (1.02, 1.06, 1.10),
            "structural_exp_cap": (0.76, 0.82, 0.88),
            "transition_exp_cap": (0.86, 0.92, 0.98),
            "recovery_exp_cap": (1.02, 1.06, 1.10),
        }
