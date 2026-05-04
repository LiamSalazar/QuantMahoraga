from __future__ import annotations

from typing import Dict, List

from mahoraga15a_config import Mahoraga15AConfig


def build_sparse_idio_short_interface(cfg: Mahoraga15AConfig) -> Dict[str, List[str]]:
    return {
        "label": [cfg.future_idio_short_label],
        "status": ["NOT_IMPLEMENTED_15A3"],
        "module_path": ["final_model/mahoraga15a/idio_short_placeholder.py"],
        "allocator_hook": ["idio_short_budget_target_after_crash_and_bear_sleeves"],
        "candidate_signals": [
            "residual_breakdown",
            "earnings_gap_failure",
            "relative_weakness_persistence",
            "post-crowding_reversal",
        ],
        "mandatory_filters": [
            "borrow_available",
            "borrow_cost_cap",
            "adv_liquidity_floor",
            "spread_slippage_cap",
            "shortable_history_min_obs",
        ],
    }
