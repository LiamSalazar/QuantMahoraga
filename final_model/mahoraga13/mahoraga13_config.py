from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import mahoraga6_1 as m6


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class Mahoraga13Config(m6.Mahoraga6Config):
    variant: str = "13.0"
    label: str = "MAHORAGA_13"
    cache_dir: str = field(default_factory=lambda: str(_project_root() / "data_cache"))
    outputs_dir: str = field(default_factory=lambda: str(_project_root() / "mahoraga13_outputs"))
    plots_dir: str = field(default_factory=lambda: str(_project_root() / "mahoraga13_plots"))
    official_baseline_label: str = "BASE_ALPHA"
    historical_benchmark_label: str = "LEGACY"
    model_label: str = "M13_MAIN"
    main_variant_key: str = "STRUCTURAL_DEFENSE_ONLY"
    continuation_variant_key: str = "CONTINUATION_V2_ONLY"
    combo_variant_key: str = "STRUCTURAL_DEFENSE_PLUS_CONTINUATION_V2"

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
    ceiling_main_sharpe_tol: float = -0.002
    ceiling_override_rate_cap: float = 0.16
    floor_main_sharpe_floor: float = 0.0
    floor_override_rate_cap: float = 0.42
    max_gate_scale: float = 1.10

    # Experimental continuation_v2 guard rails.
    continuation_v2_entry_floor: float = 0.10
    continuation_v2_entry_cap: float = 0.40
    continuation_v2_target_rate: float = 0.05
    continuation_v2_min_rate: float = 0.01
    continuation_v2_rate_cap: float = 0.10
    continuation_v2_structural_margin: float = 0.06
    continuation_v2_gate: float = 1.03
    continuation_v2_vol_mult: float = 1.04
    continuation_v2_exp_cap: float = 1.05

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
                "hawkes_weight": (0.10, 0.18),
                "structural_blend": (0.30, 0.45),
                "structural_gate": (0.88,),
                "structural_exp_cap": (0.80, 0.86),
            }
        return {
            "structural_enter_thr": (0.70, 0.76, 0.82),
            "hawkes_weight": (0.08, 0.14, 0.20),
            "structural_blend": (0.25, 0.35, 0.50),
            "structural_gate": (0.80, 0.86, 0.92),
            "structural_exp_cap": (0.76, 0.82, 0.88),
        }

