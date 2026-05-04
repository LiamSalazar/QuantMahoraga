from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

try:
    import mahoraga7_1 as h7
except Exception:
    import mahoraga7_1 as h7  # type: ignore


@dataclass
class Mahoraga8Config(h7.Mahoraga7Config):
    """
    Mahoraga 8.2 HM
    Hawkes + Markov regime fusion with 6.1-like frozen base selection and
    adaptive risk/exposure only.
    """

    
    variant: str = "8.2HM"
    outputs_dir: str = "mahoraga8_2hm_outputs"
    plots_dir: str = "mahoraga8_2hm_plots"
    label: str = "MAHORAGA_8_2HM"

    baseline_outputs_dir: str = "mahoraga6_1_outputs"
    baseline_folds_csv: str = ""

    # runtime
    run_mode: str = "FAST"  # SMOKE / FAST / FULL
    make_plots_flag: bool = False
    decision_freq: str = "W-FRI"
    outer_parallel: bool = True
    outer_backend: str = "auto"
    max_outer_jobs: int = 5
    smoke_folds: Tuple[int, ...] = (5,)
    fast_folds: Tuple[int, ...] = (3, 5)

    # hawkes stage-1
    stress_q_grid: Tuple[float, ...] = (0.85, 0.90)
    recovery_q_grid: Tuple[float, ...] = (0.80, 0.85)
    hawkes_decay_grid: Tuple[float, ...] = (0.65, 0.80)
    stress_scale_grid: Tuple[float, ...] = (0.8, 1.0)
    recovery_scale_grid: Tuple[float, ...] = (0.8, 1.0)

    # Markov-lite filter
    markov_state_count: int = 4  # NORMAL / STRESS / PANIC / RECOVERY
    markov_p_stay_normal: float = 0.94
    markov_p_stay_stress: float = 0.90
    markov_p_stay_panic: float = 0.90
    markov_p_stay_recovery: float = 0.90
    markov_emission_scale: float = 2.0
    markov_feature_scale: float = 1.0

    # entry / exit / persistence
    panic_entry_quantile: float = 0.90
    panic_exit_quantile: float = 0.75
    stress_entry_quantile: float = 0.72
    stress_exit_quantile: float = 0.55
    recovery_entry_quantile: float = 0.70
    recovery_exit_quantile: float = 0.45

    panic_min_persistence: int = 2
    stress_min_persistence: int = 2
    recovery_min_persistence: int = 2

    # hawkes urgency fusion
    hawkes_urgency_weight_grid: Tuple[float, ...] = (0.35, 0.50, 0.65)
    hawkes_panic_boost_grid: Tuple[float, ...] = (0.00, 0.10, 0.20)
    hawkes_recovery_boost_grid: Tuple[float, ...] = (0.00, 0.08, 0.15)

    # conformal / risk budget
    conformal_horizon_weeks: int = 2
    conformal_alpha: float = 0.90
    conformal_l2: float = 1e-3
    conformal_budget_low_q: float = 0.55
    conformal_budget_high_q: float = 0.90
    conformal_min_exposure: float = 0.25

    # H8.2 keeps base selection fixed
    allow_adaptive_selection: bool = False
    allow_adaptive_top_k: bool = False
    allow_adaptive_weight_cap: bool = False
    allow_adaptive_rel_tilt: bool = False
    allow_adaptive_vol_target: bool = True
    allow_adaptive_max_exposure: bool = True

    # adaptive risk by regime (multipliers / caps)
    vol_target_normal: float = 1.00   # multiplier vs base vol_target_ann
    vol_target_stress: float = 0.85
    vol_target_panic: float = 0.65
    vol_target_recovery: float = 0.95

    max_exposure_normal: float = 1.00
    max_exposure_stress: float = 0.82
    max_exposure_panic: float = 0.55
    max_exposure_recovery: float = 0.92

    # stage-2 search
    state_map_grid: Tuple[str, ...] = ("default", "defensive_plus", "balanced")
    risk_budget_blend_grid: Tuple[float, ...] = (0.50, 0.70, 0.85)
    exposure_cap_mult_grid: Tuple[float, ...] = (0.90, 1.00, 1.10)
    vol_target_shift_grid: Tuple[float, ...] = (-0.03, 0.00, 0.03)

    # scoring
    score_w_sharpe: float = 0.38
    score_w_dd: float = 0.26
    score_w_cagr: float = 0.08
    score_w_cvar: float = 0.14
    score_w_panic: float = 0.18
    score_w_stress: float = 0.14
    score_w_recovery_capture: float = 0.06
    score_pen_missed_rebound: float = 0.10
    score_pen_turnover: float = 0.02
    score_pen_worst_fold: float = 0.25
    score_pen_tail: float = 0.10
    score_pen_intervention: float = 0.10
    target_intervention_rate: float = 0.35

    utility_dd_penalty: float = 0.40
    inner_val_frac: float = 0.30
    min_train_weeks: int = 80

    # regime / BOCPD-lite
    cp_hazard: float = 1.0 / 52.0
    cp_max_run_length: int = 104
    cp_prior_kappa: float = 1.0
    cp_prior_alpha: float = 1.0
    cp_prior_beta: float = 1.0
    cp_prob_quantile: float = 0.90
    cp_severity_quantile: float = 0.80
    cp_transition_smoothing: int = 2

    # Markov-lite filter
    markov_state_count: int = 4
    markov_p_stay_normal: float = 0.94
    markov_p_stay_stress: float = 0.90
    markov_p_stay_panic: float = 0.90
    markov_p_stay_recovery: float = 0.90
    markov_emission_scale: float = 2.0
    markov_feature_scale: float = 1.0

    # entry / exit / persistence
    panic_entry_quantile: float = 0.90
    panic_exit_quantile: float = 0.75
    stress_entry_quantile: float = 0.72
    stress_exit_quantile: float = 0.55
    recovery_entry_quantile: float = 0.70
    recovery_exit_quantile: float = 0.45

    panic_min_persistence: int = 2
    stress_min_persistence: int = 2
    recovery_min_persistence: int = 2

    # conformal / risk budget
    conformal_horizon_weeks: int = 2
    conformal_alpha: float = 0.90
    conformal_l2: float = 1e-3
    conformal_budget_low_q: float = 0.55
    conformal_budget_high_q: float = 0.90
    conformal_min_exposure: float = 0.25

    # H8.2 HM adaptive risk only
    vol_target_normal: float = 1.00
    vol_target_stress: float = 0.85
    vol_target_panic: float = 0.65
    vol_target_recovery: float = 0.95

    max_exposure_normal: float = 1.00
    max_exposure_stress: float = 0.82
    max_exposure_panic: float = 0.55
    max_exposure_recovery: float = 0.92

    # stage-2 search
    state_map_grid: tuple[str, ...] = ("default", "defensive_plus", "balanced")
    risk_budget_blend_grid: tuple[float, ...] = (0.50, 0.70, 0.85)
    exposure_cap_mult_grid: tuple[float, ...] = (0.90, 1.00, 1.10)
    vol_target_shift_grid: tuple[float, ...] = (-0.03, 0.00, 0.03)

    hawkes_urgency_weight_grid: tuple[float, ...] = (0.35, 0.50, 0.65)
    hawkes_panic_boost_grid: tuple[float, ...] = (0.00, 0.10, 0.20)
    hawkes_recovery_boost_grid: tuple[float, ...] = (0.00, 0.08, 0.15)

    # scoring
    score_w_sharpe: float = 0.38
    score_w_dd: float = 0.26
    score_w_cagr: float = 0.08
    score_w_cvar: float = 0.14
    score_w_panic: float = 0.18
    score_w_stress: float = 0.14
    score_w_recovery_capture: float = 0.06
    score_pen_missed_rebound: float = 0.10
    score_pen_turnover: float = 0.02
    score_pen_worst_fold: float = 0.25
    score_pen_tail: float = 0.10
    score_pen_intervention: float = 0.10
    target_intervention_rate: float = 0.35

    # split
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
                "risk_budget_blend_grid": (0.70,),
                "exposure_cap_mult_grid": (1.0,),
                "vol_target_shift_grid": (0.0,),
                "hawkes_urgency_weight_grid": (0.50,),
                "hawkes_panic_boost_grid": (0.10,),
                "hawkes_recovery_boost_grid": (0.08,),
            }
        if mode == "FAST":
            return {
                "stress_q_grid": self.stress_q_grid,
                "recovery_q_grid": self.recovery_q_grid,
                "hawkes_decay_grid": self.hawkes_decay_grid,
                "stress_scale_grid": self.stress_scale_grid,
                "recovery_scale_grid": self.recovery_scale_grid,
                "state_map_grid": ("default", "balanced"),
                "risk_budget_blend_grid": (0.50, 0.70, 0.85),
                "exposure_cap_mult_grid": (0.90, 1.00, 1.10),
                "vol_target_shift_grid": (-0.03, 0.00, 0.03),
                "hawkes_urgency_weight_grid": self.hawkes_urgency_weight_grid,
                "hawkes_panic_boost_grid": self.hawkes_panic_boost_grid,
                "hawkes_recovery_boost_grid": self.hawkes_recovery_boost_grid,
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
            "vol_target_shift_grid": self.vol_target_shift_grid,
            "hawkes_urgency_weight_grid": self.hawkes_urgency_weight_grid,
            "hawkes_panic_boost_grid": self.hawkes_panic_boost_grid,
            "hawkes_recovery_boost_grid": self.hawkes_recovery_boost_grid,
        }

    def regime_defaults(self) -> Dict[str, Dict[str, float]]:
        return {
            "NORMAL": {
                "vol_target_mult": self.vol_target_normal,
                "max_exposure": self.max_exposure_normal,
            },
            "STRESS": {
                "vol_target_mult": self.vol_target_stress,
                "max_exposure": self.max_exposure_stress,
            },
            "PANIC": {
                "vol_target_mult": self.vol_target_panic,
                "max_exposure": self.max_exposure_panic,
            },
            "RECOVERY": {
                "vol_target_mult": self.vol_target_recovery,
                "max_exposure": self.max_exposure_recovery,
            },
        }
