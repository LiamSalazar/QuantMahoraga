from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    import mahoraga6_1 as m6
    import mahoraga7_1 as h7
except Exception:
    import mahoraga6_1 as m6  # type: ignore
    import mahoraga7_1 as h7  # type: ignore

from mahoraga8_config import Mahoraga8Config
from mahoraga8_backtest import _run_single_fold_h8, stitch_oos_path
from mahoraga8_reporting import save_outputs_h8


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def run_walk_forward_h8(ohlcv: Dict[str, Any], cfg: Mahoraga8Config, costs: m6.CostsConfig, universe_schedule: Optional[Any]) -> Dict[str, Any]:
    baseline_df = h7._load_baseline_folds(cfg)
    target_folds = set(cfg.mode_folds())
    baseline_df = baseline_df[baseline_df['fold'].isin(target_folds)].copy().sort_values('fold')
    results = []
    for _, row in baseline_df.iterrows():
        fold_n = int(row['fold'])
        results.append(_run_single_fold_h8(fold_n, row, ohlcv, cfg, costs, universe_schedule))
    return {'results': results, 'stitched_base': stitch_oos_path(results, 'base_bt'), 'stitched_h8': stitch_oos_path(results, 'ov_bt')}


def run_mahoraga8(make_plots_flag: bool = False, run_mode: str = 'FAST') -> Dict[str, Any]:
    print('=' * 80)
    print('  MAHORAGA 8 — Integrated regime-aware core')
    print('=' * 80)
    cfg = Mahoraga8Config()
    cfg.make_plots_flag = make_plots_flag
    cfg.run_mode = run_mode.upper()
    costs = m6.CostsConfig()
    ucfg = m6.UniverseConfig()
    _ensure_dir(cfg.cache_dir)
    _ensure_dir(cfg.outputs_dir)
    print('\n[1] Downloading data …')
    equity_tickers = sorted(set(list(cfg.universe_static)))
    bench_tickers = [cfg.bench_qqq, cfg.bench_spy, cfg.bench_vix]
    all_tickers = sorted(set(equity_tickers + bench_tickers))
    ohlcv = m6.download_ohlcv(all_tickers, cfg.data_start, cfg.data_end, cfg.cache_dir)
    print('\n[2] Canonical universe engine …')
    asset_registry = m6.build_asset_registry(equity_tickers, cfg, bench_tickers)
    data_quality_report = m6.compute_data_quality_report(ohlcv, equity_tickers, cfg)
    clean_equity = m6.filter_equity_candidates([t for t in equity_tickers if t in ohlcv['close'].columns], asset_registry, data_quality_report, cfg)
    universe_schedule, _ = m6.build_canonical_universe_schedule(ohlcv['close'], ohlcv['volume'], ucfg, clean_equity, cfg.data_start, cfg.data_end, registry_df=asset_registry, quality_df=data_quality_report)
    print(f"  [universe] {len(universe_schedule)} reconstitution dates built")
    print(f"\n[3] Walk-forward {cfg.variant} …")
    wf = run_walk_forward_h8(ohlcv, cfg, costs, universe_schedule)
    save_outputs_h8(cfg, wf)
    return {'cfg': cfg, 'wf': wf, 'ohlcv': ohlcv}


if __name__ == '__main__':
    run_mahoraga8(make_plots_flag=False, run_mode='FAST')
