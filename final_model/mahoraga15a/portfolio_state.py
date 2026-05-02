from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from mahoraga15a_config import Mahoraga15AConfig
from mahoraga15a_utils import clip01, drawdown_duration, drawdown_series, rolling_ridge_beta, rolling_score


def build_shared_state(
    result: Dict[str, Any],
    long_book_fold: Dict[str, Any],
    cfg: Mahoraga15AConfig,
) -> pd.DataFrame:
    idx = pd.DatetimeIndex(long_book_fold["returns"].index)
    weights = long_book_fold["weights"].reindex(idx).fillna(0.0)
    asset_rets = long_book_fold["asset_returns"].reindex(idx).fillna(0.0)
    qqq_r = asset_rets["QQQ"].reindex(idx).fillna(0.0)
    spy_r = asset_rets["SPY"].reindex(idx).fillna(0.0)

    long_gross_r = weights.mul(asset_rets[weights.columns], axis=0).sum(axis=1).fillna(0.0)
    gross_long_native = long_book_fold["gross_long"].reindex(idx).fillna(0.0)
    long_unit_gross_r = long_gross_r.divide(gross_long_native.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    long_eq = (1.0 + long_book_fold["returns"].reindex(idx).fillna(0.0)).cumprod()
    long_dd = drawdown_series(long_eq)
    long_dd_duration = drawdown_duration(long_eq)
    qqq_eq = (1.0 + qqq_r).cumprod()
    spy_eq = (1.0 + spy_r).cumprod()
    qqq_dd = drawdown_series(qqq_eq)
    spy_dd = drawdown_series(spy_eq)
    qqq_vol_21 = qqq_r.rolling(21, min_periods=10).std(ddof=1).fillna(0.0) * np.sqrt(cfg.trading_days)
    spy_vol_21 = spy_r.rolling(21, min_periods=10).std(ddof=1).fillna(0.0) * np.sqrt(cfg.trading_days)
    long_vol_21 = long_book_fold["returns"].reindex(idx).fillna(0.0).rolling(21, min_periods=10).std(ddof=1).fillna(0.0) * np.sqrt(cfg.trading_days)

    override_daily = result["variant_runs"][cfg.official_long_variant_key]["override_daily"].reindex(idx).ffill()
    override_weekly = result["variant_runs"][cfg.official_long_variant_key]["override_weekly"].copy()
    weekly_daily = override_weekly.reindex(idx).ffill()

    out = pd.DataFrame(index=idx)
    out["long_return_native"] = long_book_fold["returns"].reindex(idx).fillna(0.0)
    out["long_return_gross_native"] = long_gross_r
    out["long_return_unit_gross"] = long_unit_gross_r
    out["gross_long_native"] = gross_long_native
    out["turnover_long_native"] = long_book_fold["turnover"].reindex(idx).fillna(0.0)
    out["long_drawdown"] = long_dd.reindex(idx).fillna(0.0)
    out["long_drawdown_duration"] = long_dd_duration.reindex(idx).fillna(0.0)
    out["long_realized_vol_21"] = long_vol_21
    out["long_beta_qqq"] = rolling_ridge_beta(long_gross_r, qqq_r, cfg.hedge_beta_window, cfg.hedge_beta_min_obs, cfg.hedge_ridge_alpha)
    out["long_beta_spy"] = rolling_ridge_beta(long_gross_r, spy_r, cfg.hedge_beta_window, cfg.hedge_beta_min_obs, cfg.hedge_ridge_alpha)
    out["qqq_beta_spy"] = rolling_ridge_beta(qqq_r, spy_r, cfg.hedge_beta_window, cfg.hedge_beta_min_obs, cfg.hedge_ridge_alpha)
    out["spy_beta_qqq"] = rolling_ridge_beta(spy_r, qqq_r, cfg.hedge_beta_window, cfg.hedge_beta_min_obs, cfg.hedge_ridge_alpha)
    out["qqq_return_20d"] = qqq_eq.pct_change(20).fillna(0.0)
    out["qqq_return_60d"] = qqq_eq.pct_change(cfg.tsmom_slow_window).fillna(0.0)
    out["spy_return_20d"] = spy_eq.pct_change(20).fillna(0.0)
    out["spy_return_60d"] = spy_eq.pct_change(cfg.tsmom_slow_window).fillna(0.0)
    out["qqq_drawdown"] = qqq_dd.reindex(idx).fillna(0.0)
    out["spy_drawdown"] = spy_dd.reindex(idx).fillna(0.0)
    out["qqq_vol_21"] = qqq_vol_21
    out["spy_vol_21"] = spy_vol_21
    out["corr_rho"] = pd.Series(result["stress_pre"]["corr_rho"], index=result["stress_pre"]["idx"], dtype=float).reindex(idx).fillna(0.0)
    out["structural_fragility"] = override_daily.get("structural_score", pd.Series(0.0, index=idx)).fillna(0.0)
    out["continuation_pressure"] = override_daily.get("continuation_pressure", pd.Series(0.0, index=idx)).fillna(0.0)
    out["break_risk"] = override_daily.get("continuation_break_risk_p", pd.Series(0.0, index=idx)).fillna(0.0)
    out["continuation_benchmark_score"] = override_daily.get("continuation_benchmark_score", pd.Series(0.0, index=idx)).fillna(0.0)
    out["continuation_support_score"] = override_daily.get("continuation_support_score", pd.Series(0.0, index=idx)).fillna(0.0)
    out["compression_score"] = override_daily.get("continuation_compression_score", pd.Series(0.0, index=idx)).fillna(0.0)
    out["hawkes_stress"] = weekly_daily.get("stress_hawkes_norm", pd.Series(0.0, index=idx)).fillna(0.0)
    out["hawkes_recovery"] = weekly_daily.get("recovery_hawkes_norm", pd.Series(0.0, index=idx)).fillna(0.0)

    qqq_weak = 0.45 * rolling_score(-out["qqq_return_20d"], cfg.allocator_norm_window) + 0.35 * clip01(-out["qqq_drawdown"] / 0.18) + 0.20 * rolling_score(out["qqq_vol_21"], cfg.allocator_norm_window)
    spy_weak = 0.50 * rolling_score(-out["spy_return_20d"], cfg.allocator_norm_window) + 0.30 * clip01(-out["spy_drawdown"] / 0.18) + 0.20 * rolling_score(out["spy_vol_21"], cfg.allocator_norm_window)
    bear_persistence = 0.55 * rolling_score(-out["qqq_return_60d"], cfg.allocator_norm_window) + 0.45 * rolling_score(-out["spy_return_60d"], cfg.allocator_norm_window)
    out["benchmark_weakness"] = clip01(0.60 * qqq_weak + 0.40 * spy_weak)
    out["bear_persistence"] = clip01(bear_persistence)
    out["drawdown_pressure"] = clip01(0.70 * clip01(-out["long_drawdown"] / 0.20) + 0.30 * rolling_score(out["long_drawdown_duration"], cfg.allocator_norm_window))
    out["corr_pressure"] = clip01(0.65 * rolling_score(out["corr_rho"], cfg.allocator_norm_window) + 0.35 * rolling_score(out["compression_score"], cfg.allocator_norm_window))
    out["exposure_pressure"] = clip01(rolling_score(out["gross_long_native"], cfg.allocator_norm_window))
    out["realized_vol_pressure"] = clip01(0.55 * rolling_score(out["long_realized_vol_21"], cfg.allocator_norm_window) + 0.45 * clip01(out["long_realized_vol_21"] / max(1e-6, cfg.vol_target_ann) - 1.0))
    out["continuation_relief"] = clip01(out["continuation_pressure"] * (1.0 - out["structural_fragility"]) * (1.0 - out["benchmark_weakness"]) * (1.0 - out["bear_persistence"]))
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
