from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

import mahoraga6_1 as m6
from mahoraga9_config import Mahoraga9Config


def load_market_data(cfg: Mahoraga9Config) -> Dict[str, pd.DataFrame]:
    equity_tickers = sorted(set(list(cfg.universe_static)))
    bench_tickers = [cfg.bench_qqq, cfg.bench_spy, cfg.bench_vix]
    all_tickers = sorted(set(equity_tickers + bench_tickers))
    return m6.download_ohlcv(all_tickers, cfg.data_start, cfg.data_end, cfg.cache_dir)


def load_ff_factors(cfg: Mahoraga9Config) -> Optional[pd.DataFrame]:
    return m6.load_ff_factors(cfg.cache_dir)
