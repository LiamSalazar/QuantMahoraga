from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from mahoraga11_config import Mahoraga11Config


def _avg_pairwise_corr(rets: pd.DataFrame, window: int, idx: pd.DatetimeIndex) -> pd.Series:
    cols = list(rets.columns)
    if len(cols) <= 1:
        return pd.Series(0.0, index=idx)
    rolling_corr = rets.rolling(window).corr()
    eye = pd.DataFrame(np.eye(len(cols), dtype=bool), index=cols, columns=cols)
    vals = []
    dates = []
    for dt, mat in rolling_corr.groupby(level=0):
        block = mat.droplevel(0).reindex(index=cols, columns=cols)
        s = block.mask(eye).stack(dropna=True)
        vals.append(float(s.mean()) if len(s) else np.nan)
        dates.append(dt)
    return pd.Series(vals, index=pd.DatetimeIndex(dates)).reindex(idx).ffill().fillna(0.0)


def _rolling_efficiency(r: pd.Series, window: int) -> pd.Series:
    num = r.rolling(window).sum().abs()
    den = r.abs().rolling(window).sum().replace(0.0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rebound_from_trough(eq: pd.Series, window: int) -> pd.Series:
    trough = eq.rolling(window).min().replace(0.0, np.nan)
    return (eq / trough - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _loss_share(r: pd.Series, window: int) -> pd.Series:
    return (r < 0.0).rolling(window).mean().fillna(0.0)


def _down_up_ratio(r: pd.Series, window: int) -> pd.Series:
    downside = r.clip(upper=0.0).abs().rolling(window).mean()
    upside = r.clip(lower=0.0).rolling(window).mean().replace(0.0, np.nan)
    return (downside / upside).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_candidate_daily_context(
    base_bt: Dict[str, Any],
    base_cache: Dict[str, Any],
    ohlcv: Dict[str, pd.DataFrame],
    pre: Dict[str, Any],
    cfg: Mahoraga11Config,
) -> pd.DataFrame:
    idx = pre["idx"]
    qqq = pre["qqq"].reindex(idx).ffill()
    qqq_r = qqq.pct_change().fillna(0.0)
    qqq_eq = (1.0 + qqq_r).cumprod()
    qqq_dd = qqq_eq / qqq_eq.cummax() - 1.0
    vix = (
        ohlcv["close"][cfg.bench_vix].reindex(idx).ffill()
        if cfg.bench_vix in ohlcv["close"].columns
        else pd.Series(np.nan, index=idx)
    )
    rets = pre["rets"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    base_r = base_bt["returns_net"].reindex(idx).fillna(0.0)
    base_eq = (1.0 + base_r).cumprod()
    base_dd = base_eq / base_eq.cummax() - 1.0

    stop_active = base_cache.get("stop_active_share", pd.Series(0.0, index=idx)).reindex(idx).fillna(0.0)
    new_stop = base_cache.get("new_stop_share", pd.Series(0.0, index=idx)).reindex(idx).fillna(0.0)
    turnover_1x = base_cache.get("turnover_1x", pd.Series(0.0, index=idx)).reindex(idx).fillna(0.0)
    total_scale = base_bt.get("total_scale_target", pd.Series(1.0, index=idx)).reindex(idx).fillna(1.0)

    out = pd.DataFrame(index=idx)
    out["base_r"] = base_r
    out["base_ret_5d"] = base_eq.pct_change(5).fillna(0.0)
    out["base_ret_10d"] = base_eq.pct_change(10).fillna(0.0)
    out["base_ret_20d"] = base_eq.pct_change(20).fillna(0.0)
    out["base_dd"] = base_dd.fillna(0.0)
    out["base_dd_change_5d"] = base_dd.diff(5).fillna(0.0)
    out["base_dd_worsen_10d"] = (base_dd - base_dd.rolling(10).max()).fillna(0.0)
    out["base_rebound_10d"] = _rebound_from_trough(base_eq, 10)
    out["base_rebound_20d"] = _rebound_from_trough(base_eq, 20)
    out["base_eff_10d"] = _rolling_efficiency(base_r, 10)
    out["base_eff_20d"] = _rolling_efficiency(base_r, 20)
    out["base_loss_share_10d"] = _loss_share(base_r, 10)
    out["base_loss_share_20d"] = _loss_share(base_r, 20)
    out["base_down_up_ratio_20d"] = _down_up_ratio(base_r, 20)
    out["stop_active_10d"] = stop_active.rolling(10).mean().fillna(0.0)
    out["new_stop_10d"] = new_stop.rolling(10).mean().fillna(0.0)
    out["turnover_1x_10d"] = turnover_1x.rolling(10).mean().fillna(0.0)
    out["base_exposure"] = base_bt.get("exposure", pd.Series(0.0, index=idx)).reindex(idx).fillna(0.0)
    out["base_turnover"] = base_bt.get("turnover", pd.Series(0.0, index=idx)).reindex(idx).fillna(0.0)
    out["base_scale_gap"] = (1.0 - total_scale).clip(0.0, 1.0)

    out["avg_corr_21"] = _avg_pairwise_corr(rets, 21, idx)
    out["avg_corr_63"] = _avg_pairwise_corr(rets, 63, idx)
    out["corr_persist_21"] = (pre["corr_rho"].reindex(idx).fillna(0.0) >= 0.85).rolling(21).mean().fillna(0.0)
    out["breadth_63"] = (rets.rolling(63).mean() > 0).mean(axis=1).fillna(0.0)
    out["xs_disp_21"] = rets.rolling(21).std(ddof=1).mean(axis=1).fillna(0.0)
    out["qqq_ret_2w"] = qqq.pct_change(10).fillna(0.0)
    out["qqq_drawdown"] = qqq_dd.fillna(0.0)
    out["qqq_rebound_10d"] = _rebound_from_trough(qqq_eq, 10)
    out["qqq_eff_10d"] = _rolling_efficiency(qqq_r, 10)
    out["qqq_vol_21"] = (qqq_r.rolling(21).std(ddof=1) * np.sqrt(cfg.trading_days)).fillna(0.0)
    out["vix_z_63"] = ((vix - vix.rolling(63).mean()) / vix.rolling(63).std(ddof=1).replace(0.0, np.nan)).fillna(0.0)
    out["crisis_scale"] = base_bt.get("crisis_scale", pd.Series(1.0, index=idx)).reindex(idx).fillna(1.0)
    out["turb_scale"] = base_bt.get("turb_scale", pd.Series(1.0, index=idx)).reindex(idx).fillna(1.0)
    out["corr_rho"] = pre["corr_rho"].reindex(idx).fillna(0.0)
    return out


def build_weekly_path_dataset(daily_ctx: pd.DataFrame, cfg: Mahoraga11Config) -> pd.DataFrame:
    weekly_idx = daily_ctx.resample(cfg.decision_freq).last().dropna().index
    weekly = pd.DataFrame(index=weekly_idx)
    for col in daily_ctx.columns:
        if col == "base_r":
            weekly[col] = (1.0 + daily_ctx[col]).resample(cfg.decision_freq).prod().reindex(weekly_idx).fillna(1.0) - 1.0
        else:
            weekly[col] = daily_ctx[col].resample(cfg.decision_freq).last().reindex(weekly_idx).ffill().fillna(0.0)

    weekly["base_eq"] = (1.0 + weekly["base_r"].fillna(0.0)).cumprod()
    weekly["base_ret_1w"] = weekly["base_r"].fillna(0.0)
    weekly["base_ret_2w"] = weekly["base_eq"].pct_change(2).fillna(0.0)
    weekly["base_ret_4w"] = weekly["base_eq"].pct_change(4).fillna(0.0)
    weekly["base_dd_change_2w"] = weekly["base_dd"].diff(2).fillna(0.0)
    weekly["base_rebound_2w"] = weekly["base_rebound_10d"].rolling(2).mean().fillna(0.0)
    weekly["base_rebound_4w"] = weekly["base_rebound_20d"].rolling(4).mean().fillna(0.0)
    weekly["base_eff_2w"] = weekly["base_eff_10d"].rolling(2).mean().fillna(0.0)
    weekly["base_eff_4w"] = weekly["base_eff_20d"].rolling(4).mean().fillna(0.0)
    weekly["loss_share_2w"] = weekly["base_loss_share_10d"].rolling(2).mean().fillna(0.0)
    weekly["loss_share_4w"] = weekly["base_loss_share_20d"].rolling(4).mean().fillna(0.0)
    weekly["down_up_ratio_4w"] = weekly["base_down_up_ratio_20d"].rolling(4).mean().fillna(0.0)
    weekly["stop_density_2w"] = weekly["new_stop_10d"].rolling(2).mean().fillna(0.0)
    weekly["stop_density_4w"] = weekly["stop_active_10d"].rolling(4).mean().fillna(0.0)
    weekly["turnover_2w"] = weekly["base_turnover"].rolling(2).mean().fillna(0.0)
    weekly["base_scale_gap"] = weekly["base_scale_gap"].rolling(2).mean().fillna(0.0)
    weekly["corr_persist_4w"] = weekly["corr_persist_21"].rolling(4).mean().fillna(0.0)
    weekly["qqq_rebound_2w"] = weekly["qqq_rebound_10d"].rolling(2).mean().fillna(0.0)
    weekly["base_minus_qqq_4w"] = weekly["base_ret_4w"] - weekly["qqq_ret_2w"].rolling(2).sum().fillna(0.0)
    return weekly
