from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import mahoraga6_1 as m6
except Exception:
    import mahoraga6_1 as m6  # type: ignore

from mahoraga8_config import Mahoraga8Config


def _dynamic_vol_target_scale(gross_1x: pd.Series, target_ann_daily: pd.Series, cfg: Mahoraga8Config) -> pd.Series:
    if not getattr(cfg, 'vol_target_on', True):
        return pd.Series(1.0, index=gross_1x.index)
    realized = gross_1x.rolling(cfg.port_vol_window).std(ddof=1) * np.sqrt(cfg.trading_days)
    realized = realized.replace(0.0, np.nan)
    sc = (target_ann_daily / realized).clip(lower=cfg.min_exposure, upper=cfg.max_exposure)
    return sc.replace([np.inf, -np.inf], np.nan).fillna(1.0)


def _compute_scores_base(close: pd.DataFrame, qqq: pd.Series, cfg: Mahoraga8Config, rel_tilt: float) -> pd.DataFrame:
    cfg_rel = deepcopy(cfg)
    cfg_rel.w_rel = max(0.0, min(1.0, float(cfg.w_rel) * rel_tilt))
    rem = max(1e-8, cfg_rel.w_trend + cfg_rel.w_mom + cfg_rel.w_rel)
    cfg_rel.w_trend /= rem
    cfg_rel.w_mom /= rem
    cfg_rel.w_rel /= rem
    return m6.compute_scores(close, qqq, cfg_rel)


def _get_pit_members(dt: pd.Timestamp, universe_schedule: Optional[pd.DataFrame], fallback: list[str]) -> list[str]:
    if universe_schedule is None or len(universe_schedule) == 0:
        return fallback
    valid = universe_schedule.index[universe_schedule.index <= dt]
    if len(valid) == 0:
        return fallback
    members = universe_schedule.loc[valid[-1], 'members']
    return list(members) if isinstance(members, (list, tuple)) else fallback


def _select_assets_adaptive(dt: pd.Timestamp, sc_use: pd.DataFrame, rets: pd.DataFrame, univ_master: list[str], universe_schedule: Optional[pd.DataFrame], top_k: int, weight_cap: float, cfg: Mahoraga8Config) -> np.ndarray:
    pit_members = _get_pit_members(dt, universe_schedule, univ_master)
    if dt not in sc_use.index:
        return np.zeros(len(univ_master), dtype=float)
    if universe_schedule is not None:
        pit_scores = sc_use.loc[dt, pit_members]
        sel_names = pit_scores.nlargest(max(1, int(top_k))).index.tolist()
        names = [n for n in sel_names if pit_scores.get(n, 0) > 0]
    else:
        sc = sc_use.loc[dt]
        names = sc.nlargest(max(1, int(top_k)))
        names = [n for n in names.index.tolist() if sc.get(n, 0) > 0]
    if not names:
        return np.zeros(len(univ_master), dtype=float)
    if len(names) == 1:
        arr = np.zeros(len(univ_master), dtype=float)
        arr[univ_master.index(names[0])] = 1.0
        return arr
    lb = rets.loc[:dt].tail(cfg.hrp_window)[names].dropna()
    if len(lb) < 60:
        lb = rets.loc[:dt][names].dropna()
    ww = m6.hrp_weights(lb).reindex(names, fill_value=0.0)
    if ww.sum() > 0:
        ww = ww.clip(upper=weight_cap)
        ww = ww / ww.sum()
    arr = np.zeros(len(univ_master), dtype=float)
    for n in names:
        arr[univ_master.index(n)] = float(ww.get(n, 0.0))
    return arr


def _build_weights_adaptive(close: pd.DataFrame, rets: pd.DataFrame, qqq: pd.Series, rebal_set: set, cfg: Mahoraga8Config, universe_schedule: Optional[pd.DataFrame], policy_table: pd.DataFrame) -> pd.DataFrame:
    idx = rets.index
    univ_master = list(rets.columns)
    rel_cache: Dict[float, pd.DataFrame] = {}
    score_base = m6.compute_scores(close, qqq, cfg)
    rebal_weights: Dict[pd.Timestamp, np.ndarray] = {}
    last_w = np.zeros(len(univ_master), dtype=float)
    policy_daily = policy_table.reindex(idx).ffill().bfill()
    for dt in idx:
        if dt not in rebal_set:
            continue
        row = policy_daily.loc[dt]
        rel_tilt = float(row.get('active_rel_tilt', cfg.rel_tilt_normal))
        if rel_tilt not in rel_cache:
            rel_cache[rel_tilt] = _compute_scores_base(close, qqq, cfg, rel_tilt)
        state = str(row.get('active_regime_state', 'NORMAL'))
        sc_use = rel_cache[rel_tilt] if state in ('RECOVERY', 'NORMAL') else score_base
        top_k = int(max(1, round(float(row.get('active_top_k', cfg.top_k)))))
        weight_cap = float(row.get('active_weight_cap', cfg.weight_cap))
        last_w = _select_assets_adaptive(dt, sc_use, rets, univ_master, universe_schedule, top_k, weight_cap, cfg)
        rebal_weights[dt] = last_w.copy()
    if rebal_weights:
        w_sparse = pd.DataFrame(rebal_weights, index=univ_master).T
        w_sparse.index = pd.DatetimeIndex(w_sparse.index)
        w = w_sparse.reindex(idx).ffill().fillna(0.0)
    else:
        w = pd.DataFrame(0.0, index=idx, columns=univ_master, dtype=float)
    return w


def _compute_risk_scaled_exposure(gross_1x: pd.Series, crisis_scale: pd.Series, turb_scale: pd.Series, corr_scale: pd.Series, policy_daily: pd.DataFrame, cfg: Mahoraga8Config) -> Tuple[pd.Series, pd.Series]:
    max_exp = policy_daily['active_max_exposure'].reindex(gross_1x.index).ffill().fillna(cfg.max_exposure)
    risk_budget_cap = policy_daily['risk_budget_cap'].reindex(gross_1x.index).ffill().fillna(1.0)
    target_vol = policy_daily['active_vol_target'].reindex(gross_1x.index).ffill().fillna(cfg.vol_target_ann)
    vol_sc = _dynamic_vol_target_scale(gross_1x, target_vol, cfg)
    total_scale = np.minimum.reduce([
        crisis_scale.reindex(gross_1x.index).ffill().fillna(1.0).to_numpy(dtype=float),
        turb_scale.reindex(gross_1x.index).ffill().fillna(1.0).to_numpy(dtype=float),
        corr_scale.reindex(gross_1x.index).ffill().fillna(1.0).to_numpy(dtype=float),
        max_exp.to_numpy(dtype=float),
        risk_budget_cap.to_numpy(dtype=float),
        vol_sc.to_numpy(dtype=float),
    ])
    total_scale = pd.Series(np.clip(total_scale, cfg.min_exposure, cfg.max_exposure), index=gross_1x.index)
    return total_scale, vol_sc


def run_adaptive_core_backtest(ohlcv: Dict[str, pd.DataFrame], cfg: Mahoraga8Config, costs: m6.CostsConfig, universe_schedule: Optional[pd.DataFrame], regime_table: pd.DataFrame, policy_table: pd.DataFrame, label: str) -> Dict[str, Any]:
    close = ohlcv['close'][sorted(set(cfg.universe_static).intersection(ohlcv['close'].columns))].copy()
    high = ohlcv['high'][close.columns].copy()
    low = ohlcv['low'][close.columns].copy()
    rets = close.pct_change().fillna(0.0)
    qqq = m6.to_s(ohlcv['close'][cfg.bench_qqq].ffill(), 'QQQ')
    rebal_dates = m6.get_rebalance_dates(close.index, cfg.rebalance_freq)
    rebal_set = set(pd.DatetimeIndex(rebal_dates))
    ctx_crisis = m6.build_crisis_state(close, qqq, cfg)
    crisis_scale = m6.crisis_scale_series(ctx_crisis['state'], cfg)
    turb = m6.turbulence_series(close, qqq, cfg)
    turb_scale = m6.turbulence_scale_series(turb['zscore'], cfg)
    corr = m6.correlation_shield_series(rets, qqq.pct_change().fillna(0.0), cfg)
    corr_scale = corr['corr_scale']
    w_target = _build_weights_adaptive(close, rets, qqq, rebal_set, cfg, universe_schedule, policy_table)
    w_after_stops, _ = m6.apply_chandelier(w_target, close, high, low, cfg)
    w_exec_1x = w_after_stops.shift(1).fillna(0.0)
    policy_daily = policy_table.reindex(rets.index).ffill().bfill()
    gross_1x = (w_exec_1x * rets).sum(axis=1)
    total_scale, vol_sc = _compute_risk_scaled_exposure(gross_1x, crisis_scale, turb_scale, corr_scale, policy_daily, cfg)
    w_exec = w_exec_1x.mul(total_scale, axis=0)
    port_gross = (w_exec * rets).sum(axis=1)
    turnover = w_exec.diff().abs().sum(axis=1).fillna(0.0)
    tc = float(costs.commission) + (float(costs.slippage) if getattr(costs, 'apply_slippage', True) else 0.0)
    port_net = (port_gross - tc * turnover).fillna(0.0)
    equity = cfg.capital_initial * (1.0 + port_net).cumprod()
    exposure = w_exec.abs().sum(axis=1).clip(0.0, 1.0)
    return {
        'label': label,
        'weights_target': w_target,
        'weights_after_stops': w_after_stops,
        'weights_exec': w_exec,
        'returns_gross': port_gross,
        'returns_net': port_net,
        'equity': equity,
        'exposure': exposure,
        'turnover': turnover,
        'risk_budget_applied': policy_daily['risk_budget_cap'].reindex(exposure.index).ffill().fillna(1.0),
        'state_series': policy_daily['active_regime_state'].reindex(exposure.index).ffill().fillna('NORMAL'),
        'policy_daily': policy_daily,
        'bench': {'QQQ_r': qqq.pct_change().fillna(0.0)},
        'crisis_state': ctx_crisis['state'],
        'crisis_scale': crisis_scale,
        'turb_scale': turb_scale,
        'corr_scale': corr_scale,
        'total_scale_target': total_scale,
    }
