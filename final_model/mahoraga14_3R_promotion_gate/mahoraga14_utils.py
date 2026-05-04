from __future__ import annotations

import itertools
import os
from typing import Dict, Iterable, Iterator

import numpy as np
import pandas as pd
from scipy import stats


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def time_split_index(index: pd.Index, val_frac: float, min_train: int) -> int:
    n = len(index)
    if n <= min_train + 1:
        return max(1, n - 1)
    cut = int(round(n * (1.0 - val_frac)))
    cut = max(min_train, cut)
    cut = min(cut, n - 1)
    return cut


def paired_ttest_pvalue(diff: pd.Series, alternative: str = "greater") -> float:
    x = pd.Series(diff).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 10:
        return 1.0
    t_stat, p_two = stats.ttest_1samp(x.values, 0.0, nan_policy="omit")
    if not np.isfinite(t_stat) or not np.isfinite(p_two):
        return 1.0
    if alternative == "greater":
        return float(p_two / 2.0) if t_stat > 0 else float(1.0 - p_two / 2.0)
    if alternative == "less":
        return float(p_two / 2.0) if t_stat < 0 else float(1.0 - p_two / 2.0)
    return float(p_two)


def bhy_qvalues(p_values: Iterable[float], alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    if len(p) == 0:
        return np.array([], dtype=float)
    p = np.clip(np.where(np.isfinite(p), p, 1.0), 0.0, 1.0)
    m = len(p)
    c_m = np.sum(1.0 / np.arange(1, m + 1))
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    q_raw = p[order] * (m * c_m) / ranks
    q_sorted = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty(m, dtype=float)
    q[order] = q_sorted
    return q


def iter_grid(grid: Dict[str, Iterable[float]]) -> Iterator[Dict[str, float]]:
    keys = list(grid.keys())
    vals = [tuple(v) for v in grid.values()]
    for combo in itertools.product(*vals):
        yield {k: float(v) for k, v in zip(keys, combo)}


def cross_sectional_z(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=1).replace(0.0, np.nan)
    out = df.sub(mu, axis=0).div(sd, axis=0)
    return out.clip(-clip, clip).fillna(0.0)


def sigmoid(x) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def clip01(s: pd.Series) -> pd.Series:
    return pd.Series(np.clip(pd.Series(s).astype(float).values, 0.0, 1.0), index=pd.Index(s.index))

