from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd

import mahoraga6_1 as m6
from mahoraga12_config import Mahoraga12Config


def load_inputs(cfg: Mahoraga12Config) -> Tuple[Dict[str, pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[list]]:
    ucfg = m6.UniverseConfig()
    tickers = sorted(set(list(cfg.universe_static) + [cfg.bench_qqq, cfg.bench_spy, cfg.bench_vix]))
    ohlcv = m6.download_ohlcv(tickers, cfg.data_start, cfg.data_end, cfg.cache_dir)

    universe_schedule = None
    universe_snaps = None
    if cfg.use_canonical_universe:
        close_eq = ohlcv["close"][[t for t in cfg.universe_static if t in ohlcv["close"].columns]].copy()
        vol_eq = ohlcv["volume"][close_eq.columns].copy()
        universe_schedule, universe_snaps = m6.build_canonical_universe_schedule(
            close_eq,
            vol_eq,
            ucfg,
            list(close_eq.columns),
            cfg.data_start,
            cfg.data_end,
        )
    ff = m6.load_ff_factors(cfg.cache_dir)
    return ohlcv, universe_schedule, ff, universe_snaps
