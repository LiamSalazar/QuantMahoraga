from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import mahoraga6_1 as m6


@dataclass
class Mahoraga10Config(m6.Mahoraga6Config):
    variant: str = "10.0"
    label: str = "MAHORAGA_10"
    outputs_dir: str = "mahoraga10_outputs"
    plots_dir: str = "mahoraga10_plots"

    run_mode: str = "FAST"  # FAST / FULL
    outer_parallel: bool = True
    outer_backend: str = "loky"
    max_outer_jobs: int = 3
    decision_freq: str = "W-FRI"

    # Evaluate all folds in both modes.
    fast_folds: Tuple[int, ...] = (1, 2, 3, 4, 5)
    full_folds: Tuple[int, ...] = (1, 2, 3, 4, 5)

    # Alpha core.
    residual_beta_window: int = 63
    residual_spans_short: Tuple[int, ...] = (21, 42)
    residual_spans_long: Tuple[int, ...] = (63, 126)
    residual_mom_windows: Tuple[int, ...] = (21, 63, 126)
    residual_burn_in: int = 126
    raw_weight_shrink: float = 0.65
    resid_weight_shrink: float = 0.55
    max_alpha_mix_delta: float = 0.15

    # Risk / transition layer.
    inner_val_frac: float = 0.30
    min_train_weeks: int = 78
    weekly_horizon_weeks: int = 2
    use_hawkes: bool = True
    hawkes_event_q_low: float = 0.25
    hawkes_event_q_high: float = 0.75

    # Correlation shield is secondary only.
    use_corr_as_secondary_veto: bool = True
    corr_secondary_rho: float = 0.90
    corr_secondary_scale: float = 0.80

    # Model selection and audit.
    bhy_alpha: float = 0.05
    floor_folds: Tuple[int, ...] = (3, 5)
    ceiling_folds: Tuple[int, ...] = (1, 2, 4)

    # Meta-models.
    enable_rf_challenger: bool = True
    rf_n_estimators: int = 200
    rf_max_depth: int = 4
    rf_min_samples_leaf: int = 12
    rf_random_state: int = 42

    def mode_folds(self) -> Tuple[int, ...]:
        return self.fast_folds if self.run_mode.upper() == "FAST" else self.full_folds

    def alpha_grid(self) -> Dict[str, Tuple[float, ...]]:
        if self.run_mode.upper() == "FAST":
            return {
                "alpha_mix_base": (0.20, 0.35, 0.50),
                "beta_penalty": (0.00, 0.05),
                "raw_rel_boost": (1.00, 1.15),
            }
        return {
            "alpha_mix_base": (0.15, 0.25, 0.35, 0.50),
            "beta_penalty": (0.00, 0.03, 0.06),
            "raw_rel_boost": (1.00, 1.10, 1.20),
        }

    def policy_grid(self) -> Dict[str, Tuple[float, ...]]:
        if self.run_mode.upper() == "FAST":
            return {
                "fragility_prob_thr": (0.60, 0.70),
                "recovery_prob_thr": (0.55, 0.65),
                "hawkes_weight": (0.15, 0.30),
                "gate_floor": (0.70, 0.85),
                "vol_mult_stress": (0.80, 0.90),
                "vol_mult_recovery": (1.00, 1.05),
                "max_exp_stress": (0.70, 0.85),
                "max_exp_recovery": (1.00, 1.10),
                "alpha_tilt": (0.00, 0.08),
            }
        return {
            "fragility_prob_thr": (0.55, 0.65, 0.75),
            "recovery_prob_thr": (0.50, 0.60, 0.70),
            "hawkes_weight": (0.10, 0.20, 0.30),
            "gate_floor": (0.65, 0.80, 0.90),
            "vol_mult_stress": (0.75, 0.85, 0.95),
            "vol_mult_recovery": (1.00, 1.05, 1.10),
            "max_exp_stress": (0.65, 0.80, 0.90),
            "max_exp_recovery": (1.00, 1.05, 1.10),
            "alpha_tilt": (0.00, 0.05, 0.10),
        }
