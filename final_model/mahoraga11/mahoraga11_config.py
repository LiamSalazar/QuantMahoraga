from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import mahoraga6_1 as m6


@dataclass
class Mahoraga11Config(m6.Mahoraga6Config):
    variant: str = "11.0"
    label: str = "MAHORAGA_11"
    outputs_dir: str = "mahoraga11_outputs"
    plots_dir: str = "mahoraga11_plots"

    # Use the stable 6.1 selection as the floor, then improve from there.
    weight_cap: float = 0.55
    k_atr: float = 3.0
    turb_zscore_thr: float = 1.5
    turb_scale_min: float = 0.40
    vol_target_ann: float = 0.25

    run_mode: str = "FAST"  # FAST / FULL
    make_plots_flag: bool = False
    decision_freq: str = "W-FRI"
    outer_parallel: bool = True
    outer_backend: str = "loky"
    max_outer_jobs: int = 3

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
                "ceiling_mix": (0.10, 0.20),
                "floor_mix": (0.45, 0.60),
                "ceiling_beta_penalty": (0.00, 0.03),
                "floor_beta_penalty": (0.05, 0.08),
                "raw_rel_boost": (1.00, 1.10),
            }
        return {
            "ceiling_mix": (0.05, 0.10, 0.20),
            "floor_mix": (0.40, 0.50, 0.60),
            "ceiling_beta_penalty": (0.00, 0.02, 0.04),
            "floor_beta_penalty": (0.04, 0.06, 0.08),
            "raw_rel_boost": (1.00, 1.10, 1.20),
        }

    def router_grid(self) -> Dict[str, Tuple[float, ...]]:
        if self.run_mode.upper() == "FAST":
            return {
                "structural_prob_thr": (0.60, 0.70),
                "fast_prob_thr": (0.58, 0.68),
                "recovery_prob_thr": (0.55, 0.65),
                "hawkes_weight": (0.15, 0.30),
                "floor_blend_max": (0.55, 0.75),
                "gate_floor": (0.72, 0.85),
                "vol_mult_stress": (0.82, 0.92),
                "vol_mult_recovery": (1.00, 1.06),
                "max_exp_stress": (0.72, 0.86),
                "max_exp_recovery": (1.00, 1.08),
            }
        return {
            "structural_prob_thr": (0.55, 0.65, 0.75),
            "fast_prob_thr": (0.55, 0.65, 0.75),
            "recovery_prob_thr": (0.50, 0.60, 0.70),
            "hawkes_weight": (0.10, 0.20, 0.30),
            "floor_blend_max": (0.50, 0.65, 0.80),
            "gate_floor": (0.68, 0.80, 0.90),
            "vol_mult_stress": (0.78, 0.88, 0.95),
            "vol_mult_recovery": (1.00, 1.05, 1.10),
            "max_exp_stress": (0.68, 0.80, 0.90),
            "max_exp_recovery": (1.00, 1.05, 1.10),
        }
