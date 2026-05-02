from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict


_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
_M14_DIR = _PARENT_DIR / "mahoraga14"
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_PARENT_DIR) not in sys.path:
    sys.path.append(str(_PARENT_DIR))
if str(_M14_DIR) not in sys.path:
    sys.path.append(str(_M14_DIR))

import mahoraga6_1 as m6
from backtest_executor import run_walk_forward_mahoraga15a
from mahoraga14_data import load_inputs
from mahoraga15a_config import Mahoraga15AConfig
from mahoraga15a_reporting import save_outputs
from mahoraga15a_utils import ensure_dir


def run_mahoraga15a(run_mode: str = "FAST") -> Dict[str, Any]:
    cfg = Mahoraga15AConfig()
    cfg.run_mode = run_mode.upper()
    ensure_dir(cfg.outputs_dir)
    ensure_dir(cfg.plots_dir)
    costs = m6.CostsConfig()
    ohlcv, universe_schedule, ff, universe_snaps = load_inputs(cfg)
    wf = run_walk_forward_mahoraga15a(ohlcv, cfg, costs, universe_schedule)
    artifacts = save_outputs(wf, cfg, costs)
    return {
        "wf": wf,
        "artifacts": artifacts,
        "cfg": cfg,
        "costs": costs,
        "ff": ff,
        "universe_schedule": universe_schedule,
        "universe_snaps": universe_snaps,
    }


if __name__ == "__main__":
    run_mahoraga15a(run_mode="FAST")
