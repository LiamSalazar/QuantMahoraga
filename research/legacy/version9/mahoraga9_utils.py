from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_series(x, name: str = "x") -> pd.Series:
    if isinstance(x, pd.Series):
        return x.rename(name)
    return pd.Series(x, name=name)


def annualize(r: pd.Series, td: int = 252) -> float:
    r = safe_series(r).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) == 0:
        return 0.0
    return float((1.0 + r).prod() ** (td / len(r)) - 1.0)


def sharpe(r: pd.Series, td: int = 252) -> float:
    r = safe_series(r).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    return float(r.mean() / sd * np.sqrt(td)) if sd > 1e-12 else 0.0


def max_dd(r: pd.Series) -> float:
    r = safe_series(r).fillna(0.0)
    eq = (1.0 + r).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(dd.min()) if len(dd) else 0.0


def cvar5(r: pd.Series) -> float:
    r = safe_series(r).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) == 0:
        return 0.0
    q = float(r.quantile(0.05))
    tail = r[r <= q]
    return float(tail.mean()) if len(tail) else q


def intervention_rate(scale: pd.Series, tol: float = 1e-6) -> float:
    s = safe_series(scale).fillna(1.0)
    return float((np.abs(s - 1.0) > tol).mean()) if len(s) else 0.0


def weekly_last(s: pd.Series, freq: str = "W-FRI") -> pd.Series:
    s = safe_series(s).sort_index()
    return s.resample(freq).last().dropna()


def zscore_train(s: pd.Series, train_idx: pd.Index) -> pd.Series:
    tr = s.reindex(train_idx).dropna()
    mu = float(tr.mean()) if len(tr) else 0.0
    sd = float(tr.std(ddof=1)) if len(tr) > 1 else 1.0
    if not np.isfinite(sd) or sd <= 1e-8:
        sd = 1.0
    out = (s - mu) / sd
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def rank_to_unit_interval(s: pd.Series) -> pd.Series:
    s = safe_series(s)
    if len(s) == 0:
        return s.astype(float)
    r = s.rank(method="average")
    return ((r - 1.0) / max(len(r) - 1.0, 1.0)).astype(float)


def clip_prob(x: Iterable[float] | pd.Series, lo: float = 1e-4, hi: float = 1 - 1e-4):
    if isinstance(x, pd.Series):
        return x.clip(lo, hi)
    arr = np.asarray(list(x), dtype=float)
    return np.clip(arr, lo, hi)
