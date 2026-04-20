from __future__ import annotations

from typing import Any, Dict

import mahoraga6_1 as m6

from mahoraga11_backtest import run_walk_forward_mahoraga11
from mahoraga11_config import Mahoraga11Config
from mahoraga11_data import load_inputs
from mahoraga11_reporting import build_final_report_text, save_outputs
from mahoraga11_utils import ensure_dir


def run_mahoraga11(make_plots_flag: bool = False, run_mode: str = "FAST") -> Dict[str, Any]:
    cfg = Mahoraga11Config()
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
    wf = run_walk_forward_mahoraga11(ohlcv, cfg, costs, universe_schedule)
    artifacts = save_outputs(wf, cfg, ff=ff)
    print(build_final_report_text(wf, cfg))
    return {"wf": wf, "artifacts": artifacts, "cfg": cfg, "ff": ff, "universe_schedule": universe_schedule, "universe_snaps": universe_snaps}


if __name__ == "__main__":
    run_mahoraga11(make_plots_flag=False, run_mode="FAST")
