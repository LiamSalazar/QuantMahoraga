from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from mahoraga9_config import Mahoraga9Config


def _rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var().replace(0.0, np.nan)
    beta = cov / var
    return beta.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)


def _avg_pairwise_corr(rets: pd.DataFrame, window: int, idx: pd.DatetimeIndex) -> pd.Series:
    cols = list(rets.columns)
    n = len(cols)
    if n <= 1:
        return pd.Series(0.0, index=idx, dtype=float)
    rolling_corr = rets.rolling(window).corr()
    eye = pd.DataFrame(np.eye(n, dtype=bool), index=cols, columns=cols)

    def _one_date(df: pd.DataFrame) -> float:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return np.nan
        mat = df.reindex(index=cols, columns=cols)
        vals = mat.mask(eye).stack(dropna=True)
        return float(vals.mean()) if len(vals) else np.nan

    out = rolling_corr.groupby(level=0, sort=False).apply(_one_date)
    out.index = pd.DatetimeIndex(out.index)
    return out.reindex(idx).ffill().fillna(0.0)


def build_daily_context(base_bt: Dict[str, Any], ohlcv: Dict[str, pd.DataFrame], cfg: Mahoraga9Config) -> pd.DataFrame:
    idx = base_bt["returns_net"].index
    qqq = ohlcv["close"][cfg.bench_qqq].reindex(idx).ffill()
    vix = ohlcv["close"][cfg.bench_vix].reindex(idx).ffill() if cfg.bench_vix in ohlcv["close"].columns else pd.Series(np.nan, index=idx)
    close = ohlcv["close"].reindex(idx)
    rets = close.pct_change().replace([np.inf, -np.inf], np.nan)
    qqq_r = qqq.pct_change().fillna(0.0)
    base_r = base_bt["returns_net"].reindex(idx).fillna(0.0)

    avg_corr_21 = _avg_pairwise_corr(rets, 21, idx)
    avg_corr_63 = _avg_pairwise_corr(rets, 63, idx)
    breadth_63 = (rets.rolling(63).mean() > 0).mean(axis=1)
    xs_disp_5 = rets.rolling(5).std(ddof=1).mean(axis=1)
    xs_disp_21 = rets.rolling(21).std(ddof=1).mean(axis=1)
    qqq_ret_5 = qqq.pct_change(5)
    qqq_ret_21 = qqq.pct_change(21)
    qqq_eq = (1.0 + qqq_r.fillna(0.0)).cumprod()
    qqq_dd = qqq_eq / qqq_eq.cummax() - 1.0
    qqq_vol_21 = qqq_r.rolling(21).std(ddof=1) * np.sqrt(cfg.trading_days)
    vix_z_63 = (vix - vix.rolling(63).mean()) / vix.rolling(63).std(ddof=1).replace(0.0, np.nan)
    beta_63 = _rolling_beta(base_r, qqq_r, cfg.resid_beta_window)

    out = pd.DataFrame(index=idx)
    out["base_r"] = base_r
    out["qqq_r"] = qqq_r.fillna(0.0)
    out["base_exposure"] = base_bt.get("exposure", pd.Series(0.0, index=idx)).reindex(idx).fillna(0.0)
    out["base_turnover"] = base_bt.get("turnover", pd.Series(0.0, index=idx)).reindex(idx).fillna(0.0)
    out["crisis_scale"] = base_bt.get("crisis_scale", pd.Series(1.0, index=idx)).reindex(idx).fillna(1.0)
    out["turb_scale"] = base_bt.get("turb_scale", pd.Series(1.0, index=idx)).reindex(idx).fillna(1.0)
    out["corr_scale"] = base_bt.get("corr_scale", pd.Series(1.0, index=idx)).reindex(idx).fillna(1.0)
    out["avg_corr_21"] = avg_corr_21
    out["avg_corr_63"] = avg_corr_63
    out["breadth_63"] = breadth_63.fillna(0.0)
    out["xs_disp_5"] = xs_disp_5.fillna(0.0)
    out["xs_disp_21"] = xs_disp_21.fillna(0.0)
    out["qqq_ret_5"] = qqq_ret_5.fillna(0.0)
    out["qqq_ret_21"] = qqq_ret_21.fillna(0.0)
    out["qqq_drawdown"] = qqq_dd.fillna(0.0)
    out["qqq_vol_21"] = qqq_vol_21.ffill().fillna(0.0)
    out["vix_level"] = vix.ffill().fillna(0.0)
    out["vix_z_63"] = vix_z_63.fillna(0.0)
    out["beta_63"] = beta_63.fillna(1.0)
    return out


def build_weekly_context(daily_ctx: pd.DataFrame, cfg: Mahoraga9Config) -> pd.DataFrame:
    idx = daily_ctx.index.to_series().resample(cfg.decision_freq).last().dropna().index
    out = pd.DataFrame(index=idx)
    for c in daily_ctx.columns:
        out[c] = daily_ctx[c].resample(cfg.decision_freq).last().reindex(idx).ffill()
    return out
