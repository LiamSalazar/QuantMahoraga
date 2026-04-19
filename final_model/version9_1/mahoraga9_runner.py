from __future__ import annotations

from typing import Any, Dict

import mahoraga6_1 as m6

from mahoraga9_backtest import run_walk_forward_v9
from mahoraga9_config import Mahoraga9Config
from mahoraga9_data import load_market_data
from mahoraga9_reporting import save_outputs_v9
from mahoraga9_universe import build_universe
from mahoraga9_utils import ensure_dir


def run_mahoraga9(make_plots_flag: bool = False, run_mode: str = "FAST") -> Dict[str, Any]:
    cfg = Mahoraga9Config()
    cfg.make_plots_flag = make_plots_flag
    cfg.run_mode = run_mode.upper()
    costs = m6.CostsConfig()

    ensure_dir(cfg.cache_dir)
    ensure_dir(cfg.outputs_dir)
    ensure_dir(cfg.plots_dir)

    print("=" * 80)
    print("  MAHORAGA 9.1 — alpha/risk rewrite")
    print("=" * 80)

    print("\n[1] Loading market data …")
    ohlcv = load_market_data(cfg)

    print("\n[2] Building canonical universe …")
    universe_schedule, asset_registry, data_quality_report = build_universe(ohlcv, cfg)
    print(f"  [universe] {len(universe_schedule)} reconstitution dates built")

    print(f"\n[3] Running walk-forward in {cfg.run_mode} mode …")
    wf = run_walk_forward_v9(ohlcv, cfg, costs, universe_schedule)

    print("\n[4] Saving outputs …")
    saved = save_outputs_v9(cfg, wf)
    return {
        "cfg": cfg,
        "wf": wf,
        "ohlcv": ohlcv,
        "universe_schedule": universe_schedule,
        "asset_registry": asset_registry,
        "data_quality_report": data_quality_report,
        "saved": saved,
    }


if __name__ == "__main__":
    run_mahoraga9(make_plots_flag=False, run_mode="FAST")
