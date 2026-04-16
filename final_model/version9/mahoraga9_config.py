from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import mahoraga6_1 as m6


@dataclass
class Mahoraga9Config(m6.Mahoraga6Config):
    variant: str = "9.0"
    label: str = "MAHORAGA_9"
    outputs_dir: str = "version9_outputs"
    plots_dir: str = "version9_plots"

    run_mode: str = "FAST"  # FAST / FULL
    make_plots_flag: bool = False
    outer_parallel: bool = True
    outer_backend: str = "loky"
    max_outer_jobs: int = 5
    fast_folds: Tuple[int, ...] = (3, 5)

    # meta horizon / cadence
    decision_freq: str = "W-FRI"
    fragility_horizon_weeks: int = 2
    recovery_horizon_weeks: int = 2
    inner_val_frac: float = 0.30
    min_train_weeks: int = 104

    # alpha redesign
    resid_beta_window: int = 63
    resid_mom_window: int = 63
    resid_trend_fast: int = 42
    resid_trend_slow: int = 126
    alpha_mix_min: float = 0.20
    alpha_mix_max: float = 0.80
    alpha_mix_default: float = 0.50
    residual_rel_boost: float = 1.30
    residual_trend_mult: float = 0.80
    residual_mom_mult: float = 0.85

    # Hawkes-like transition signal
    hawkes_decay: float = 0.80
    hawkes_stress_q: float = 0.85
    hawkes_recovery_q: float = 0.80
    hawkes_stress_scale: float = 1.00
    hawkes_recovery_scale: float = 1.00

    # classifiers
    meta_model_candidates: Tuple[str, ...] = ("logit", "rf")
    rf_n_estimators: int = 250
    rf_max_depth: int = 4
    rf_min_samples_leaf: int = 25
    logit_c_grid: Tuple[float, ...] = (0.10, 0.50, 1.00, 2.00)

    # cheap candidate sweep over policy mapping
    fragility_gate_low_grid: Tuple[float, ...] = (0.55, 0.65, 0.75)
    fragility_prob_thr_grid: Tuple[float, ...] = (0.50, 0.60, 0.70)
    recovery_prob_thr_grid: Tuple[float, ...] = (0.50, 0.60, 0.70)
    hawkes_weight_grid: Tuple[float, ...] = (0.20, 0.35, 0.50)
    vol_target_floor_grid: Tuple[float, ...] = (0.70, 0.80, 0.90)
    exposure_floor_grid: Tuple[float, ...] = (0.50, 0.60, 0.70)
    exposure_ceiling_grid: Tuple[float, ...] = (0.95, 1.00)

    # corr shield downgraded from primary execution role
    use_corr_as_secondary_veto: bool = True
    corr_secondary_rho: float = 0.90
    corr_secondary_scale: float = 0.80
    corr_use_vix_confirm: bool = True
    corr_vix_confirm_level: float = 24.0

    # selection utility
    score_w_sharpe: float = 0.45
    score_w_cagr: float = 0.10
    score_w_alpha: float = 0.15
    score_w_recovery: float = 0.08
    score_pen_maxdd: float = 0.18
    score_pen_cvar: float = 0.10
    score_pen_intervention: float = 0.06
    score_pen_dispersion: float = 0.10
    max_allowed_intervention_rate: float = 0.35
    max_allowed_exposure_collapse: float = 0.35
    max_allowed_fold_dd_worsening: float = 0.015

    # runtime shortcuts
    fast_grid_take_first: int = 2

    def mode_folds(self) -> Tuple[int, ...]:
        return self.fast_folds if self.run_mode.upper() == "FAST" else tuple(range(1, 6))

    def candidate_grid(self) -> Dict[str, Tuple[float, ...]]:
        grid = {
            "fragility_gate_low": self.fragility_gate_low_grid,
            "fragility_prob_thr": self.fragility_prob_thr_grid,
            "recovery_prob_thr": self.recovery_prob_thr_grid,
            "hawkes_weight": self.hawkes_weight_grid,
            "vol_target_floor": self.vol_target_floor_grid,
            "exposure_floor": self.exposure_floor_grid,
            "exposure_ceiling": self.exposure_ceiling_grid,
        }
        if self.run_mode.upper() == "FAST":
            return {k: tuple(v[: self.fast_grid_take_first]) for k, v in grid.items()}
        return grid
