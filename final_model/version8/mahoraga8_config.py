from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

try:
    import mahoraga7_1 as h7
except Exception:
    import mahoraga7_1 as h7  # type: ignore


@dataclass
class Mahoraga8Config(h7.Mahoraga7Config):
    variant: str = "8.1L"
    outputs_dir: str = "mahoraga8_1lite_outputs"
    plots_dir: str = "mahoraga8_1lite_plots"
    label: str = "MAHORAGA_8_1_LITE"

    baseline_outputs_dir: str = "mahoraga6_1_outputs"
    baseline_folds_csv: str = ""

    run_mode: str = "FAST"  # SMOKE / FAST / FULL
    make_plots_flag: bool = False
    decision_freq: str = "W-FRI"
    outer_parallel: bool = False  # safer by default while iterating
    outer_backend: str = "auto"
    max_outer_jobs: int = 5
    smoke_folds: Tuple[int, ...] = (5,)
    fast_folds: Tuple[int, ...] = (3, 5)

    # Stage-1 Hawkes only
    stress_q_grid: Tuple[float, ...] = (0.85, 0.90)
    recovery_q_grid: Tuple[float, ...] = (0.80, 0.85)
    hawkes_decay_grid: Tuple[float, ...] = (0.65, 0.80)
    stress_scale_grid: Tuple[float, ...] = (0.8, 1.0)
    recovery_scale_grid: Tuple[float, ...] = (0.8, 1.0)

    # Regime / BOCPD-lite / conformal
    cp_hazard: float = 1.0 / 52.0
    cp_max_run_length: int = 104
    cp_prior_kappa: float = 1.0
    cp_prior_alpha: float = 1.0
    cp_prior_beta: float = 1.0
    cp_prob_quantile: float = 0.92
    cp_severity_quantile: float = 0.82
    cp_transition_smoothing: int = 2
    stress_entry_quantile: float = 0.80
    stress_exit_quantile: float = 0.65
    panic_entry_quantile: float = 0.93
    panic_exit_quantile: float = 0.80
    recovery_entry_quantile: float = 0.35
    stress_min_persistence: int = 2
    panic_min_persistence: int = 2
    recovery_min_persistence: int = 2

    conformal_horizon_weeks: int = 2
    conformal_alpha: float = 0.90
    conformal_l2: float = 1e-3
    conformal_budget_low_q: float = 0.60
    conformal_budget_high_q: float = 0.92
    conformal_min_exposure: float = 0.25

    # Lite core control: only exposure and vol_target adapt
    integrated_core_mode: str = "lite"
    allow_adaptive_selection: bool = False
    allow_adaptive_top_k: bool = False
    allow_adaptive_weight_cap: bool = False
    allow_adaptive_rel_tilt: bool = False
    allow_adaptive_vol_target: bool = True
    allow_adaptive_max_exposure: bool = True

    vol_target_normal: float = 0.30
    vol_target_stress: float = 0.24
    vol_target_panic: float = 0.16
    vol_target_recovery: float = 0.28

    max_exposure_normal: float = 1.00
    max_exposure_stress: float = 0.82
    max_exposure_panic: float = 0.45
    max_exposure_recovery: float = 0.92

    # Calibration search — lite only (no adaptive top-k / weight-cap)
    state_map_grid: Tuple[str, ...] = ("default", "defensive_plus")
    risk_budget_blend_grid: Tuple[float, ...] = (0.55, 0.75, 0.95)
    exposure_cap_mult_grid: Tuple[float, ...] = (0.90, 1.00)
    vol_target_shift_grid: Tuple[float, ...] = (-0.02, 0.00, 0.02)

    # retained only for compatibility with prior code paths, but fixed in lite mode
    top_k_shift_grid: Tuple[int, ...] = (0,)

    score_w_sharpe: float = 0.32
    score_w_dd: float = 0.28
    score_w_cagr: float = 0.05
    score_w_cvar: float = 0.18
    score_w_panic: float = 0.22
    score_w_stress: float = 0.18
    score_w_recovery_capture: float = 0.04
    score_pen_missed_rebound: float = 0.08
    score_pen_turnover: float = 0.01
    score_pen_worst_fold: float = 0.28
    score_pen_tail: float = 0.14
    score_pen_regime_degrade: float = 0.35

    utility_dd_penalty: float = 0.40
    inner_val_frac: float = 0.30
    min_train_weeks: int = 80

    def mode_folds(self) -> Tuple[int, ...]:
        mode = self.run_mode.upper()
        if mode == "SMOKE":
            return self.smoke_folds
        if mode == "FAST":
            return self.fast_folds
        return tuple(range(1, 6))

    def mode_overrides(self) -> Dict[str, Tuple]:
        mode = self.run_mode.upper()
        if mode == "SMOKE":
            return {
                "stress_q_grid": (self.stress_q_grid[0],),
                "recovery_q_grid": (self.recovery_q_grid[0],),
                "hawkes_decay_grid": (self.hawkes_decay_grid[0],),
                "stress_scale_grid": (self.stress_scale_grid[0],),
                "recovery_scale_grid": (self.recovery_scale_grid[0],),
                "state_map_grid": ("default",),
                "risk_budget_blend_grid": (0.75,),
                "exposure_cap_mult_grid": (1.0,),
                "top_k_shift_grid": (0,),
                "vol_target_shift_grid": (0.0,),
            }
        if mode == "FAST":
            return {
                "stress_q_grid": self.stress_q_grid,
                "recovery_q_grid": self.recovery_q_grid,
                "hawkes_decay_grid": self.hawkes_decay_grid,
                "stress_scale_grid": self.stress_scale_grid,
                "recovery_scale_grid": self.recovery_scale_grid,
                "state_map_grid": self.state_map_grid,
                "risk_budget_blend_grid": self.risk_budget_blend_grid,
                "exposure_cap_mult_grid": self.exposure_cap_mult_grid,
                "top_k_shift_grid": (0,),
                "vol_target_shift_grid": self.vol_target_shift_grid,
            }
        return {
            "stress_q_grid": self.stress_q_grid,
            "recovery_q_grid": self.recovery_q_grid,
            "hawkes_decay_grid": self.hawkes_decay_grid,
            "stress_scale_grid": self.stress_scale_grid,
            "recovery_scale_grid": self.recovery_scale_grid,
            "state_map_grid": self.state_map_grid,
            "risk_budget_blend_grid": self.risk_budget_blend_grid,
            "exposure_cap_mult_grid": self.exposure_cap_mult_grid,
            "top_k_shift_grid": (0,),
            "vol_target_shift_grid": self.vol_target_shift_grid,
        }
