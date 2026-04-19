from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

import mahoraga6_1 as m6
from mahoraga9_config import Mahoraga9Config


def normalize_universe_schedule(universe_schedule: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if universe_schedule is None or len(universe_schedule) == 0:
        return universe_schedule
    if isinstance(universe_schedule.index, pd.DatetimeIndex):
        return universe_schedule.sort_index()
    us = universe_schedule.copy()
    for c in ["date", "as_of", "rebalance_date", "effective_date", "timestamp"]:
        if c in us.columns:
            us[c] = pd.to_datetime(us[c], errors="coerce")
            us = us.dropna(subset=[c]).sort_values(c).set_index(c)
            return us
    return universe_schedule


def build_universe(ohlcv: dict[str, pd.DataFrame], cfg: Mahoraga9Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ucfg = m6.UniverseConfig()
    equity_tickers = sorted(set(list(cfg.universe_static)))
    bench_tickers = [cfg.bench_qqq, cfg.bench_spy, cfg.bench_vix]
    asset_registry = m6.build_asset_registry(equity_tickers, cfg, bench_tickers)
    data_quality_report = m6.compute_data_quality_report(ohlcv, equity_tickers, cfg)
    clean_equity = m6.filter_equity_candidates(
        [t for t in equity_tickers if t in ohlcv["close"].columns],
        asset_registry,
        data_quality_report,
        cfg,
    )
    universe_schedule, _ = m6.build_canonical_universe_schedule(
        ohlcv["close"],
        ohlcv["volume"],
        ucfg,
        clean_equity,
        cfg.data_start,
        cfg.data_end,
        registry_df=asset_registry,
        quality_df=data_quality_report,
    )
    return normalize_universe_schedule(universe_schedule), asset_registry, data_quality_report
