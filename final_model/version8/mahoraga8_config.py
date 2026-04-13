from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

try:
    import mahoraga7_1 as h7
except Exception:
    import mahoraga7_1 as h7  # type: ignore


@dataclass
class Mahoraga8Config(h7.Mahoraga7Config):
    variant: str = "8"
    outputs_dir: str = "mahoraga8_outputs"
    plots_dir: str = "mahoraga8_plots"
    label: str = "MAHORAGA_8"

    baseline_outputs_dir: str = "mahoraga6_1_outputs"
    baseline_folds_csv: str = ""

    run_mode: str = "FAST"  # SMOKE / FAST / FULL
    make_plots_flag: bool = False
    decision_freq: str = "W-FRI"
    outer_parallel: bool = True
    outer_backend: str = "auto"
    max_outer_jobs: int = 5
    smoke_folds: Tuple[int, ...] = (5,)
    fast_folds: Tuple[int, ...] = (3, 5)

    stress_q_grid: Tuple[float, ...] = (0.85, 0.90)
    recovery_q_grid: Tuple[float, ...] = (0.80, 0.85)
    hawkes_decay_grid: Tuple[float, ...] = (0.65, 0.80)
    stress_scale_grid: Tuple[float, ...] = (0.8, 1.0)
    recovery_scale_grid: Tuple[float, ...] = (0.8, 1.0)

    cp_hazard: float = 1.0 / 52.0
    cp_max_run_length: int = 104
    cp_prior_kappa: float = 1.0
    cp_prior_alpha: float = 1.0
    cp_prior_beta: float = 1.0
    cp_prob_quantile: float = 0.90
    cp_severity_quantile: float = 0.80
    cp_transition_smoothing: int = 2

    conformal_horizon_weeks: int = 2
    conformal_alpha: float = 0.90
    conformal_l2: float = 1e-3
    conformal_budget_low_q: float = 0.55
    conformal_budget_high_q: float = 0.90
    conformal_min_exposure: float = 0.25

    top_k_normal: int = 3
    top_k_stress: int = 2
    top_k_panic: int = 1
    top_k_recovery: int = 3

    weight_cap_normal: float = 0.60
    weight_cap_stress: float = 0.52
    weight_cap_panic: float = 0.45
    weight_cap_recovery: float = 0.62

    vol_target_normal: float = 0.30
    vol_target_stress: float = 0.22
    vol_target_panic: float = 0.12
    vol_target_recovery: float = 0.26

    max_exposure_normal: float = 1.00
    max_exposure_stress: float = 0.75
    max_exposure_panic: float = 0.35
    max_exposure_recovery: float = 0.90

    k_atr_normal: float = 2.5
    k_atr_stress: float = 2.3
    k_atr_panic: float = 2.0
    k_atr_recovery: float = 2.4

    rel_tilt_normal: float = 0.60
    rel_tilt_stress: float = 0.55
    rel_tilt_panic: float = 0.55
    rel_tilt_recovery: float = 0.60

    state_map_grid: Tuple[str, ...] = ("default", "defensive_plus", "recovery_plus")
    risk_budget_blend_grid: Tuple[float, ...] = (0.50, 0.75, 1.00)
    exposure_cap_mult_grid: Tuple[float, ...] = (0.90, 1.00, 1.10)
    top_k_shift_grid: Tuple[int, ...] = (0, 1)
    vol_target_shift_grid: Tuple[float, ...] = (0.00, 0.03)

    score_w_sharpe: float = 0.46
    score_w_dd: float = 0.22
    score_w_cagr: float = 0.10
    score_w_cvar: float = 0.14
    score_w_panic: float = 0.14
    score_w_stress: float = 0.10
    score_w_recovery_capture: float = 0.08
    score_pen_missed_rebound: float = 0.16
    score_pen_turnover: float = 0.03
    score_pen_worst_fold: float = 0.18
    score_pen_tail: float = 0.10

    utility_dd_penalty: float = 0.40
    inner_val_frac: float = 0.30
    min_train_weeks: int = 80

    def mode_folds(self) -> Tuple[int, ...]:
        mode = self.run_mode.upper()
        if mode == 'SMOKE':
            return self.smoke_folds
        if mode == 'FAST':
            return self.fast_folds
        return tuple(range(1, 6))

    def mode_overrides(self) -> Dict[str, Tuple]:
        mode = self.run_mode.upper()
        if mode == 'SMOKE':
            return {
                'stress_q_grid': (self.stress_q_grid[0],),
                'recovery_q_grid': (self.recovery_q_grid[0],),
                'hawkes_decay_grid': (self.hawkes_decay_grid[0],),
                'stress_scale_grid': (self.stress_scale_grid[0],),
                'recovery_scale_grid': (self.recovery_scale_grid[0],),
                'state_map_grid': ('default',),
                'risk_budget_blend_grid': (0.75,),
                'exposure_cap_mult_grid': (1.0,),
                'top_k_shift_grid': (0,),
                'vol_target_shift_grid': (0.0,),
            }
        if mode == 'FAST':
            return {
                'stress_q_grid': self.stress_q_grid,
                'recovery_q_grid': self.recovery_q_grid,
                'hawkes_decay_grid': self.hawkes_decay_grid,
                'stress_scale_grid': self.stress_scale_grid,
                'recovery_scale_grid': self.recovery_scale_grid,
                'state_map_grid': ('default', 'defensive_plus'),
                'risk_budget_blend_grid': (0.50, 0.75, 1.00),
                'exposure_cap_mult_grid': (0.90, 1.00),
                'top_k_shift_grid': (0, 1),
                'vol_target_shift_grid': (0.00, 0.03),
            }
        return {
            'stress_q_grid': self.stress_q_grid,
            'recovery_q_grid': self.recovery_q_grid,
            'hawkes_decay_grid': self.hawkes_decay_grid,
            'stress_scale_grid': self.stress_scale_grid,
            'recovery_scale_grid': self.recovery_scale_grid,
            'state_map_grid': self.state_map_grid,
            'risk_budget_blend_grid': self.risk_budget_blend_grid,
            'exposure_cap_mult_grid': self.exposure_cap_mult_grid,
            'top_k_shift_grid': self.top_k_shift_grid,
            'vol_target_shift_grid': self.vol_target_shift_grid,
        }

    def regime_defaults(self) -> Dict[str, Dict[str, float]]:
        return {
            'NORMAL': {'top_k': float(self.top_k_normal), 'weight_cap': self.weight_cap_normal, 'vol_target': self.vol_target_normal, 'max_exposure': self.max_exposure_normal, 'k_atr': self.k_atr_normal, 'rel_tilt': self.rel_tilt_normal},
            'STRESS': {'top_k': float(self.top_k_stress), 'weight_cap': self.weight_cap_stress, 'vol_target': self.vol_target_stress, 'max_exposure': self.max_exposure_stress, 'k_atr': self.k_atr_stress, 'rel_tilt': self.rel_tilt_stress},
            'PANIC': {'top_k': float(self.top_k_panic), 'weight_cap': self.weight_cap_panic, 'vol_target': self.vol_target_panic, 'max_exposure': self.max_exposure_panic, 'k_atr': self.k_atr_panic, 'rel_tilt': self.rel_tilt_panic},
            'RECOVERY': {'top_k': float(self.top_k_recovery), 'weight_cap': self.weight_cap_recovery, 'vol_target': self.vol_target_recovery, 'max_exposure': self.max_exposure_recovery, 'k_atr': self.k_atr_recovery, 'rel_tilt': self.rel_tilt_recovery},
        }
