from __future__ import annotations

from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import mahoraga6_1 as m6
from mahoraga9_config import Mahoraga9Config


def _residual_price(price: pd.Series, bench: pd.Series, beta_window: int) -> pd.Series:
    r = price.pct_change().fillna(0.0)
    b = bench.pct_change().fillna(0.0)
    beta = r.rolling(beta_window).cov(b) / b.rolling(beta_window).var().replace(0.0, np.nan)
    beta = beta.shift(1).replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
    resid_r = (r - beta * b).clip(-0.30, 0.30)
    resid_px = (1.0 + resid_r).cumprod()
    return resid_px


def _trend_unit(price: pd.Series, fast: int, slow: int) -> pd.Series:
    ema_f = price.ewm(span=fast, adjust=False).mean().shift(1)
    ema_s = price.ewm(span=slow, adjust=False).mean().shift(1)
    sig = ((ema_f > ema_s) & price.notna()).astype(float)
    return sig.fillna(0.0)


def _mom_unit(price: pd.Series, windows: Tuple[int, ...]) -> pd.Series:
    raw = sum((price / price.shift(w) - 1.0).shift(1) for w in windows) / max(len(windows), 1)
    return ((raw.clip(-1.0, 1.0) + 1.0) / 2.0).fillna(0.0)


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


def build_score_bundle(ohlcv: Dict[str, pd.DataFrame], cfg_raw: Mahoraga9Config, cfg_resid: Mahoraga9Config) -> Dict[str, pd.DataFrame]:
    close = ohlcv["close"].reindex(columns=[c for c in cfg_raw.universe_static if c in ohlcv["close"].columns]).copy()
    qqq = ohlcv["close"][cfg_raw.bench_qqq].reindex(close.index).ffill()
    raw_scores = m6.compute_scores(close, qqq, cfg_raw)

    resid_scores = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    rel_base = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    qqq_unit_rel = qqq
    for t in close.columns:
        p = close[t].ffill()
        resid_px = _residual_price(p, qqq, cfg_raw.resid_beta_window)
        resid_tr = _trend_unit(resid_px, cfg_raw.resid_trend_fast, cfg_raw.resid_trend_slow)
        resid_mo = _mom_unit(resid_px, cfg_raw.resid_mom_windows)
        rel_sig = m6._rel(p, qqq_unit_rel, cfg_raw).fillna(0.0)
        resid_scores[t] = (
            cfg_resid.w_trend * resid_tr +
            cfg_resid.w_mom * resid_mo +
            cfg_resid.w_rel * rel_sig
        ).fillna(0.0)
        rel_base[t] = rel_sig
    resid_scores.iloc[:cfg_raw.burn_in] = 0.0
    return {
        "raw_scores": raw_scores.fillna(0.0),
        "resid_scores": resid_scores.fillna(0.0),
        "rel_scores": rel_base.fillna(0.0),
    }
