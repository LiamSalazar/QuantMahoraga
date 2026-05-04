from __future__ import annotations

from copy import deepcopy
from typing import Dict, Tuple

import mahoraga6_1 as m6
from mahoraga9_config import Mahoraga9Config


def fit_alpha_configs(ohlcv: dict, cfg_fold: Mahoraga9Config, train_start, train_end) -> Dict[str, Mahoraga9Config]:
    qqq = m6.to_s(ohlcv["close"][cfg_fold.bench_qqq].ffill(), "QQQ")
    close = ohlcv["close"][list(cfg_fold.universe_static)]

    cfg_raw = deepcopy(cfg_fold)
    wt, wm, wr = m6.fit_ic_weights(close, qqq.loc[train_start:train_end], cfg_raw, train_start, train_end)
    cfg_raw.w_trend, cfg_raw.w_mom, cfg_raw.w_rel = wt, wm, wr

    cfg_resid = deepcopy(cfg_raw)
    cfg_resid.w_trend = max(0.0, cfg_raw.w_trend * cfg_fold.residual_trend_mult)
    cfg_resid.w_mom = max(0.0, cfg_raw.w_mom * cfg_fold.residual_mom_mult)
    cfg_resid.w_rel = max(0.0, cfg_raw.w_rel * cfg_fold.residual_rel_boost)
    s = cfg_resid.w_trend + cfg_resid.w_mom + cfg_resid.w_rel
    if s > 0:
        cfg_resid.w_trend /= s
        cfg_resid.w_mom /= s
        cfg_resid.w_rel /= s
    return {"raw": cfg_raw, "resid": cfg_resid}
