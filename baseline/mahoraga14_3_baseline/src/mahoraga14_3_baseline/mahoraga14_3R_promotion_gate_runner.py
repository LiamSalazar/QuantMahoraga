from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import mahoraga6_1 as m6

from mahoraga14_backtest import run_walk_forward_mahoraga14
from mahoraga14_config import Mahoraga14Config
from mahoraga14_data import load_inputs
from mahoraga14_utils import ensure_dir
from promotion_gate_suite import save_promotion_gate_outputs


def run_mahoraga14_3R_promotion_gate(
    make_plots_flag: bool = False,
    run_mode: str = "FAST",
) -> Dict[str, Any]:
    cfg = Mahoraga14Config()
    cfg.run_mode = run_mode.upper()
    cfg.make_plots_flag = make_plots_flag

    if cfg.run_mode == "FAST":
        cfg.max_outer_jobs = min(cfg.max_outer_jobs, 3)
    else:
        cfg.max_outer_jobs = max(2, cfg.max_outer_jobs)

    ensure_dir(cfg.cache_dir)
    ensure_dir(cfg.outputs_dir)

    costs = m6.CostsConfig()
    ohlcv, universe_schedule, ff, universe_snaps = load_inputs(cfg)
    wf = run_walk_forward_mahoraga14(ohlcv, cfg, costs, universe_schedule)
    artifacts = save_promotion_gate_outputs(wf, cfg)

    decision_path = Path(cfg.outputs_dir) / "promotion_gate_decision.md"
    if decision_path.exists():
        print(decision_path.read_text(encoding="utf-8"))

    return {
        "wf": wf,
        "artifacts": artifacts,
        "cfg": cfg,
        "ff": ff,
        "universe_schedule": universe_schedule,
        "universe_snaps": universe_snaps,
    }


if __name__ == "__main__":
    run_mahoraga14_3R_promotion_gate(make_plots_flag=False, run_mode="FAST")
