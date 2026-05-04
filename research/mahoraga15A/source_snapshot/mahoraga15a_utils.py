from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy import stats


_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
_M14_DIR = _PARENT_DIR / "mahoraga14"
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_PARENT_DIR) not in sys.path:
    sys.path.append(str(_PARENT_DIR))
if str(_M14_DIR) not in sys.path:
    sys.path.append(str(_M14_DIR))

import mahoraga6_1 as m6


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clip01(values) -> pd.Series:
    s = pd.Series(values, dtype=float)
    return pd.Series(np.clip(s.values, 0.0, 1.0), index=s.index)


def sigmoid(values) -> pd.Series:
    s = pd.Series(values, dtype=float)
    x = np.clip(s.values, -40.0, 40.0)
    return pd.Series(1.0 / (1.0 + np.exp(-x)), index=s.index)


def rolling_zscore(values: pd.Series, window: int, min_periods: int = 20) -> pd.Series:
    s = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan)
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=1).replace(0.0, np.nan)
    return ((s - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def rolling_score(values: pd.Series, window: int, invert: bool = False) -> pd.Series:
    z = rolling_zscore(values, window=window)
    if invert:
        z = -z
    return clip01(sigmoid(z))


def drawdown_series(equity: pd.Series) -> pd.Series:
    eq = pd.Series(equity, dtype=float).ffill().fillna(0.0)
    if len(eq) == 0:
        return pd.Series(dtype=float)
    return eq / eq.cummax() - 1.0


def drawdown_duration(equity: pd.Series) -> pd.Series:
    eq = pd.Series(equity, dtype=float).ffill().fillna(0.0)
    if len(eq) == 0:
        return pd.Series(dtype=float)
    at_high = eq >= eq.cummax() - 1e-12
    groups = at_high.astype(int).cumsum()
    return (~at_high).groupby(groups).cumsum().astype(float)


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


def alpha_nw(strategy_r: pd.Series, bench_r: pd.Series, cfg, label: str) -> Dict[str, float]:
    res = m6.alpha_test_nw(strategy_r, bench_r, cfg, label=label)
    if "error" in res:
        return {"alpha_ann": np.nan, "t_alpha": np.nan, "p_alpha": np.nan, "beta": np.nan, "R2": np.nan}
    return {
        "alpha_ann": float(res.get("alpha_ann", np.nan)),
        "t_alpha": float(res.get("t_alpha", np.nan)),
        "p_alpha": float(res.get("p_alpha", np.nan)),
        "beta": float(res.get("beta", np.nan)),
        "R2": float(res.get("R2", np.nan)),
    }


def beta(strategy_r: pd.Series, bench_r: pd.Series) -> float:
    s = pd.Series(strategy_r, dtype=float).dropna()
    b = pd.Series(bench_r, dtype=float).reindex(s.index).fillna(0.0)
    if len(s) < 2 or float(b.var()) == 0.0:
        return 0.0
    return float(s.cov(b) / b.var())


def capture_ratio(strategy_r: pd.Series, bench_r: pd.Series, upside: bool) -> float:
    s = pd.Series(strategy_r, dtype=float).dropna()
    b = pd.Series(bench_r, dtype=float).reindex(s.index).fillna(0.0)
    mask = b > 0.0 if upside else b < 0.0
    denom = float(b.loc[mask].sum())
    if not mask.any() or abs(denom) < 1e-12:
        return 0.0
    return float(s.loc[mask].sum() / denom)


def return_per_exposure(r: pd.Series, exposure: pd.Series) -> float:
    s = pd.Series(r, dtype=float)
    exp = pd.Series(exposure, dtype=float).reindex(s.index).fillna(0.0)
    denom = float(exp.mean())
    if denom <= 1e-12:
        return 0.0
    return float(s.mean() / denom)


def rolling_ridge_beta(
    y: pd.Series,
    x: pd.Series,
    window: int,
    min_obs: int,
    ridge_alpha: float,
) -> pd.Series:
    y_s = pd.Series(y, dtype=float).fillna(0.0)
    x_s = pd.Series(x, dtype=float).reindex(y_s.index).fillna(0.0)
    out = pd.Series(np.nan, index=y_s.index, dtype=float)
    x_vals = x_s.to_numpy(dtype=float)
    y_vals = y_s.to_numpy(dtype=float)

    for i in range(len(out)):
        start = max(0, i - window)
        hist = i - start
        if hist < min_obs:
            continue
        x_hist = x_vals[start:i]
        y_hist = y_vals[start:i]
        x_mean = float(np.nanmean(x_hist))
        y_mean = float(np.nanmean(y_hist))
        x_c = np.nan_to_num(x_hist - x_mean, nan=0.0)
        y_c = np.nan_to_num(y_hist - y_mean, nan=0.0)

        x_std = float(np.nanstd(x_c, ddof=1))
        if not np.isfinite(x_std) or x_std <= 1e-12:
            continue

        x_z = x_c / x_std
        denom = float(np.dot(x_z, x_z) + float(ridge_alpha))
        if denom <= 1e-12:
            continue
        coef_z = float(np.dot(x_z, y_c) / denom)
        out.iloc[i] = coef_z / x_std

    fallback = pd.Series(y_s.rolling(window, min_periods=min_obs).cov(x_s), index=y_s.index)
    denom = x_s.rolling(window, min_periods=min_obs).var().replace(0.0, np.nan)
    fallback = (fallback / denom).replace([np.inf, -np.inf], np.nan)
    return out.fillna(fallback).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


def rolling_portfolio_metrics(r: pd.Series, bench_r: pd.Series, window: int, cfg, label: str) -> pd.DataFrame:
    s = pd.Series(r, dtype=float).fillna(0.0)
    b = pd.Series(bench_r, dtype=float).reindex(s.index).fillna(0.0)
    rows = []
    for i in range(len(s)):
        start = max(0, i - window)
        if i - start < max(20, window // 2):
            rows.append((np.nan, np.nan))
            continue
        sub_s = s.iloc[start:i]
        sub_b = b.iloc[start:i]
        rows.append((beta(sub_s, sub_b), alpha_nw(sub_s, sub_b, cfg, f"{label}_{i}")["alpha_ann"]))
    return pd.DataFrame(rows, index=s.index, columns=["beta", "alpha_ann"])


def summarize_object(obj: Dict[str, pd.Series], cfg, label: str) -> Dict[str, float]:
    return m6.summarize(obj["returns"], obj["equity"], obj["exposure"], obj["turnover"], cfg, label)


def stationary_bootstrap(values: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    if n == 0:
        return np.array([], dtype=float)
    p = 1.0 / max(1, int(block))
    out = np.empty(n, dtype=float)
    idx = int(rng.integers(0, n))
    for i in range(n):
        if i == 0 or rng.random() < p:
            idx = int(rng.integers(0, n))
        else:
            idx = (idx + 1) % n
        out[i] = values[idx]
    return out


def build_weight_backtest(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cfg,
    costs,
    label: str,
) -> Dict[str, pd.Series]:
    w = weights.reindex(asset_returns.index).fillna(0.0)
    ret = asset_returns.reindex(w.index).fillna(0.0)
    turnover, tc = m6._costs(w, costs)
    gross = w.mul(ret, axis=0).sum(axis=1).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    net = (gross - tc).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    equity = cfg.capital_initial * (1.0 + net).cumprod()
    gross_long = w.clip(lower=0.0).sum(axis=1)
    gross_short = (-w.clip(upper=0.0)).sum(axis=1)
    net_exposure = w.sum(axis=1)
    gross_exposure = gross_long + gross_short
    return {
        "label": label,
        "weights": w,
        "returns": net,
        "gross_returns": gross,
        "transaction_cost": tc,
        "turnover": turnover,
        "equity": equity,
        "gross_long": gross_long,
        "gross_short": gross_short,
        "net_exposure": net_exposure,
        "gross_exposure": gross_exposure,
        "exposure": gross_exposure,
    }


def stitch_objects(
    fold_objects: Iterable[Dict[str, pd.Series]],
    fold_meta: Iterable[Tuple[int, pd.Timestamp, pd.Timestamp]],
    cfg,
    label: str,
) -> Dict[str, pd.Series]:
    frames = []
    trace_rows = []
    for obj, (fold, start, end) in zip(fold_objects, fold_meta):
        idx = pd.DatetimeIndex(obj["returns"].loc[start:end].index)
        frame = pd.DataFrame(
            {
                "returns": obj["returns"].loc[idx],
                "gross_returns": obj["gross_returns"].loc[idx],
                "transaction_cost": obj["transaction_cost"].loc[idx],
                "turnover": obj["turnover"].loc[idx],
                "gross_long": obj["gross_long"].loc[idx],
                "gross_short": obj["gross_short"].loc[idx],
                "net_exposure": obj["net_exposure"].loc[idx],
                "gross_exposure": obj["gross_exposure"].loc[idx],
            }
        )
        frames.append(frame)
        trace_rows.append(
            {
                "Label": label,
                "Fold": int(fold),
                "WindowType": "TEST_ONLY",
                "SliceStart": idx.min() if len(idx) else pd.NaT,
                "SliceEnd": idx.max() if len(idx) else pd.NaT,
                "Rows": int(len(idx)),
            }
        )
    stitched = pd.concat(frames).sort_index() if frames else pd.DataFrame()
    equity = cfg.capital_initial * (1.0 + stitched.get("returns", pd.Series(dtype=float))).cumprod() if len(stitched) else pd.Series(dtype=float)
    return {
        "label": label,
        "returns": stitched.get("returns", pd.Series(dtype=float)),
        "gross_returns": stitched.get("gross_returns", pd.Series(dtype=float)),
        "transaction_cost": stitched.get("transaction_cost", pd.Series(dtype=float)),
        "turnover": stitched.get("turnover", pd.Series(dtype=float)),
        "gross_long": stitched.get("gross_long", pd.Series(dtype=float)),
        "gross_short": stitched.get("gross_short", pd.Series(dtype=float)),
        "net_exposure": stitched.get("net_exposure", pd.Series(dtype=float)),
        "gross_exposure": stitched.get("gross_exposure", pd.Series(dtype=float)),
        "equity": equity,
        "exposure": stitched.get("gross_exposure", pd.Series(dtype=float)),
        "trace": pd.DataFrame(trace_rows),
    }
