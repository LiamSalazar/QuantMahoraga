from __future__ import annotations

import os
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy import stats


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


def paired_ttest_pvalue(diff: pd.Series, alternative: str = "greater") -> float:
    x = safe_series(diff).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 8:
        return 1.0
    stat, p_two = stats.ttest_1samp(x.to_numpy(dtype=float), popmean=0.0, nan_policy="omit")
    if not np.isfinite(p_two):
        return 1.0
    if alternative == "greater":
        if stat >= 0:
            return float(p_two / 2.0)
        return float(1.0 - p_two / 2.0)
    if alternative == "less":
        if stat <= 0:
            return float(p_two / 2.0)
        return float(1.0 - p_two / 2.0)
    return float(p_two)


def bhy_qvalues(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    if p.size == 0:
        return np.asarray([], dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    m = p.size
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(p)
    ranked = p[order]
    raw = ranked * m * c_m / np.arange(1, m + 1)
    q = np.minimum.accumulate(raw[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def differential_alpha_proxy(model_r: pd.Series, base_r: pd.Series, td: int = 252) -> float:
    diff = safe_series(model_r).reindex(base_r.index).fillna(0.0) - safe_series(base_r).fillna(0.0)
    return float(diff.mean() * td) if len(diff) else 0.0


def time_split_index(index: pd.Index, inner_val_frac: float, min_train_points: int) -> int:
    n = len(index)
    if n <= max(min_train_points + 8, 16):
        return max(int(n * 0.7), min(n - 8, min_train_points))
    cut = int(round(n * (1.0 - inner_val_frac)))
    cut = max(cut, min_train_points)
    cut = min(cut, n - 8)
    return int(cut)
