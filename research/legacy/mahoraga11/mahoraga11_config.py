from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import mahoraga6_1 as m6


@dataclass
class Mahoraga11Config(m6.Mahoraga6Config):
    variant: str = "11.1"
    label: str = "MAHORAGA_11"
    outputs_dir: str = "mahoraga11_outputs"
    plots_dir: str = "mahoraga11_plots"

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

    # Alpha design.
    residual_beta_window: int = 63
    residual_spans_short: Tuple[int, ...] = (21, 42)
    residual_spans_long: Tuple[int, ...] = (63, 126)
    residual_mom_windows: Tuple[int, ...] = (21, 63, 126)
    residual_burn_in: int = 126
    raw_weight_shrink: float = 0.70
    resid_weight_shrink: float = 0.60

    # Router / transition.
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
                "base_mix": (0.18, 0.28),
                "defense_mix": (0.50, 0.65),
                "base_beta_penalty": (0.00,),
                "defense_beta_penalty": (0.06,),
                "raw_rel_boost": (1.00, 1.08),
            }
        return {
            "base_mix": (0.12, 0.20, 0.28),
            "defense_mix": (0.45, 0.55, 0.70),
            "base_beta_penalty": (0.00, 0.02),
            "defense_beta_penalty": (0.04, 0.06, 0.08),
            "raw_rel_boost": (1.00, 1.08, 1.15),
        }

    def policy_grid(self) -> Dict[str, Tuple[float, ...]]:
        if self.run_mode.upper() == "FAST":
            return {
                "structural_enter_thr": (0.74, 0.80),
                "transition_enter_thr": (0.66,),
                "recovery_enter_thr": (0.60,),
                "hawkes_weight": (0.12, 0.20),
                "structural_blend": (0.35, 0.50),
                "transition_blend": (0.10,),
                "structural_gate": (0.82, 0.90),
                "transition_gate": (0.90,),
                "transition_vol_mult": (0.92,),
                "recovery_vol_mult": (1.02,),
                "structural_exp_cap": (0.78, 0.86),
                "transition_exp_cap": (0.90,),
                "recovery_exp_cap": (1.02,),
            }
        return {
            "structural_enter_thr": (0.68, 0.74, 0.82),
            "transition_enter_thr": (0.60, 0.68, 0.74),
            "recovery_enter_thr": (0.55, 0.62, 0.70),
            "hawkes_weight": (0.10, 0.18, 0.26),
            "structural_blend": (0.30, 0.40, 0.55),
            "transition_blend": (0.05, 0.10, 0.18),
            "structural_gate": (0.78, 0.86, 0.92),
            "transition_gate": (0.88, 0.92, 0.96),
            "transition_vol_mult": (0.88, 0.94, 1.00),
            "recovery_vol_mult": (1.00, 1.03, 1.06),
            "structural_exp_cap": (0.72, 0.80, 0.88),
            "transition_exp_cap": (0.86, 0.92, 0.98),
            "recovery_exp_cap": (1.00, 1.03, 1.06),
        }

    def router_grid(self) -> Dict[str, Tuple[float, ...]]:
        return self.policy_grid()
