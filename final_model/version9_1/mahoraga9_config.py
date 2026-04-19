from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import mahoraga6_1 as m6


@dataclass
class Mahoraga9Config(m6.Mahoraga6Config):
    variant: str = "9.1"
    label: str = "MAHORAGA_9_1"
    outputs_dir: str = "version9_1_outputs"
    plots_dir: str = "version9_1_plots"

    run_mode: str = "FAST"  # FAST / FULL
    make_plots_flag: bool = False
    outer_parallel: bool = True
    outer_backend: str = "loky"
    max_outer_jobs: int = 5

    # FAST and FULL both evaluate all 5 folds.
    fast_folds: Tuple[int, ...] = (1, 2, 3, 4, 5)
    full_folds: Tuple[int, ...] = (1, 2, 3, 4, 5)

    decision_freq: str = "W-FRI"
    fragility_horizon_weeks: int = 2
    recovery_horizon_weeks: int = 2
    inner_val_frac: float = 0.30
    min_train_weeks: int = 104

    # Alpha redesign
    resid_beta_window: int = 63
    resid_mom_windows: Tuple[int, ...] = (21, 63)
    resid_trend_fast: int = 21
    resid_trend_slow: int = 84
    alpha_mix_min: float = 0.0
    alpha_mix_max: float = 0.55
    alpha_mix_default: float = 0.15
    residual_rel_boost: float = 1.10
    residual_trend_mult: float = 0.90
    residual_mom_mult: float = 1.10

    alpha_mix_base_grid_fast: Tuple[float, ...] = (0.00, 0.15, 0.30)
    alpha_mix_base_grid_full: Tuple[float, ...] = (0.00, 0.10, 0.20, 0.30, 0.40)
    alpha_tilt_grid_fast: Tuple[float, ...] = (0.00, 0.10)
    alpha_tilt_grid_full: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.15)

    # Hawkes fast-transition layer
    hawkes_decay: float = 0.80
    hawkes_stress_q: float = 0.85
    hawkes_recovery_q: float = 0.80
    hawkes_stress_scale: float = 1.00
    hawkes_recovery_scale: float = 1.00
    hawkes_weight_grid_fast: Tuple[float, ...] = (0.00, 0.15)
    hawkes_weight_grid_full: Tuple[float, ...] = (0.00, 0.10, 0.20, 0.30)

    # Meta models
    meta_model_candidates_fast: Tuple[str, ...] = ("logit", "rf")
    meta_model_candidates_full: Tuple[str, ...] = ("logit", "rf")
    rf_n_estimators_fast: int = 250
    rf_n_estimators_full: int = 500
    rf_max_depth_fast: int = 4
    rf_max_depth_full: int = 5
    rf_min_samples_leaf: int = 25
    logit_c_grid_fast: Tuple[float, ...] = (0.50, 1.00)
    logit_c_grid_full: Tuple[float, ...] = (0.10, 0.50, 1.00, 2.00)

    # Sparse adaptive policy
    fragility_prob_thr_grid_fast: Tuple[float, ...] = (0.70, 0.80)
    fragility_prob_thr_grid_full: Tuple[float, ...] = (0.65, 0.75, 0.85)
    recovery_prob_thr_grid_fast: Tuple[float, ...] = (0.65, 0.75)
    recovery_prob_thr_grid_full: Tuple[float, ...] = (0.60, 0.70, 0.80)
    risk_floor_grid_fast: Tuple[float, ...] = (1.00, 0.90)
    risk_floor_grid_full: Tuple[float, ...] = (1.00, 0.95, 0.90, 0.85)

    # Secondary correlation veto only
    use_corr_as_secondary_veto: bool = True
    corr_secondary_rho: float = 0.90
    corr_secondary_scale: float = 0.85
    corr_use_vix_confirm: bool = True
    corr_vix_confirm_level: float = 24.0

    # Selection utility
    score_w_sharpe: float = 0.42
    score_w_cagr: float = 0.08
    score_w_alpha: float = 0.12
    score_w_recovery: float = 0.06
    score_pen_maxdd: float = 0.16
    score_pen_cvar: float = 0.08
    score_pen_intervention: float = 0.05
    score_pen_exposure_collapse: float = 0.05
    score_pen_qvalue: float = 0.06
    max_allowed_intervention_rate: float = 0.35
    max_allowed_exposure_collapse: float = 0.20
    max_allowed_fold_dd_worsening: float = 0.02

    # FAST and FULL report summaries
    floor_folds: Tuple[int, ...] = (3, 5)
    ceiling_folds: Tuple[int, ...] = (1, 2, 4)

    def mode_folds(self) -> Tuple[int, ...]:
        return self.fast_folds if self.run_mode.upper() == "FAST" else self.full_folds

    def alpha_mix_base_grid(self) -> Tuple[float, ...]:
        return self.alpha_mix_base_grid_fast if self.run_mode.upper() == "FAST" else self.alpha_mix_base_grid_full

    def alpha_tilt_grid(self) -> Tuple[float, ...]:
        return self.alpha_tilt_grid_fast if self.run_mode.upper() == "FAST" else self.alpha_tilt_grid_full

    def hawkes_weight_grid(self) -> Tuple[float, ...]:
        return self.hawkes_weight_grid_fast if self.run_mode.upper() == "FAST" else self.hawkes_weight_grid_full

    def fragility_prob_thr_grid(self) -> Tuple[float, ...]:
        return self.fragility_prob_thr_grid_fast if self.run_mode.upper() == "FAST" else self.fragility_prob_thr_grid_full

    def recovery_prob_thr_grid(self) -> Tuple[float, ...]:
        return self.recovery_prob_thr_grid_fast if self.run_mode.upper() == "FAST" else self.recovery_prob_thr_grid_full

    def risk_floor_grid(self) -> Tuple[float, ...]:
        return self.risk_floor_grid_fast if self.run_mode.upper() == "FAST" else self.risk_floor_grid_full

    def meta_model_candidates(self) -> Tuple[str, ...]:
        return self.meta_model_candidates_fast if self.run_mode.upper() == "FAST" else self.meta_model_candidates_full

    def rf_n_estimators(self) -> int:
        return self.rf_n_estimators_fast if self.run_mode.upper() == "FAST" else self.rf_n_estimators_full

    def rf_max_depth(self) -> int:
        return self.rf_max_depth_fast if self.run_mode.upper() == "FAST" else self.rf_max_depth_full

    def logit_c_grid(self) -> Tuple[float, ...]:
        return self.logit_c_grid_fast if self.run_mode.upper() == "FAST" else self.logit_c_grid_full

    def candidate_grid(self) -> Dict[str, Tuple[float, ...]]:
        return {
            "alpha_mix_base": self.alpha_mix_base_grid(),
            "alpha_tilt": self.alpha_tilt_grid(),
            "fragility_prob_thr": self.fragility_prob_thr_grid(),
            "recovery_prob_thr": self.recovery_prob_thr_grid(),
            "hawkes_weight": self.hawkes_weight_grid(),
            "risk_floor": self.risk_floor_grid(),
        }
