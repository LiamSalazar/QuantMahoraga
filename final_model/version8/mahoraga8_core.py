from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import mahoraga6_1 as m6
    import mahoraga7_1 as h7
except Exception:
    import mahoraga6_1 as m6  # type: ignore
    import mahoraga7_1 as h7  # type: ignore

from mahoraga8_config import Mahoraga8Config


def _dynamic_vol_target_scale(gross_1x: pd.Series, target_ann_daily: pd.Series, cfg: Mahoraga8Config) -> pd.Series:
    if not getattr(cfg, 'vol_target_on', True):
        return pd.Series(1.0, index=gross_1x.index)
    realized = gross_1x.rolling(cfg.port_vol_window).std(ddof=1) * np.sqrt(cfg.trading_days)
    realized = realized.replace(0.0, np.nan)
    sc = (target_ann_daily / realized).clip(lower=cfg.min_exposure, upper=cfg.max_exposure)
    return sc.replace([np.inf, -np.inf], np.nan).fillna(1.0)


def _compute_weights_base_like(
    ctx: Dict[str, Any],
    cfg: Mahoraga8Config,
    universe_schedule: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    H8.1 Lite:
    keep base selection frozen. We still reuse h7._build_weights_sparse
    because it is fast and aligned with the local stack, but we force
    BASELINE actions everywhere so no REL_TILT path is used.
    """
    score_base = ctx.get("score_base", None)
    if score_base is None:
        score_base = m6.compute_scores(ctx["close"], ctx["qqq"], cfg)

    action_daily = pd.Series("BASELINE", index=ctx["idx"], dtype=object)

    # score_rel is irrelevant when action is always BASELINE, but we pass
    # score_base to satisfy the interface safely.
    return h7._build_weights_sparse(
        ctx=ctx,
        cfg=cfg,
        universe_schedule=universe_schedule,
        score_rel=score_base,
        action_daily=action_daily,
    )


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


def run_adaptive_core_backtest(
    ohlcv: Dict[str, pd.DataFrame],
    cfg: Mahoraga8Config,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
    regime_table: pd.DataFrame,
    policy_table: pd.DataFrame,
    label: str,
) -> Dict[str, Any]:
    ctx = h7._precompute_overlay_context(ohlcv, cfg, costs, universe_schedule)

    close = ctx['close']
    high = ctx['high']
    low = ctx['low']
    rets = ctx['rets']

    crisis_scale = ctx['crisis_scale']
    turb_scale = ctx['turb_scale']
    corr_scale = ctx['corr_scale']
    crisis_state = ctx['crisis_state']
    corr_state = ctx['corr_state']
    corr_rho = ctx['corr_rho']

    w_target = _compute_weights_base_like(ctx, cfg, universe_schedule)
    w_after_stops, stop_hits = m6.apply_chandelier(w_target, close, high, low, cfg)
    w_exec_1x = w_after_stops.shift(1).fillna(0.0)

    policy_daily = policy_table.reindex(rets.index).ffill().bfill()
    gross_1x = (w_exec_1x * rets).sum(axis=1)
    total_scale, vol_sc = _compute_risk_scaled_exposure(
        gross_1x=gross_1x,
        crisis_scale=crisis_scale,
        turb_scale=turb_scale,
        corr_scale=corr_scale,
        policy_daily=policy_daily,
        cfg=cfg,
    )

    w_exec = w_exec_1x.mul(total_scale, axis=0)
    to, tc = m6._costs(w_exec, costs)
    port_gross = (w_exec * rets).sum(axis=1)
    port_net = (port_gross - tc).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    equity = cfg.capital_initial * (1.0 + port_net).cumprod()
    exposure = w_exec.abs().sum(axis=1).clip(0.0, cfg.max_exposure)

    qqq_r = ctx['qqq_r']
    qqq_eq = cfg.capital_initial * (1.0 + qqq_r).cumprod()

    return {
        'label': label,
        'weights_target': w_target,
        'weights_after_stops': w_after_stops,
        'weights_exec': w_exec,
        'returns_gross': port_gross,
        'returns_net': port_net,
        'equity': equity,
        'exposure': exposure,
        'turnover': to,
        'risk_budget_applied': policy_daily['risk_budget_cap'].reindex(exposure.index).ffill().fillna(1.0),
        'state_series': policy_daily['active_regime_state'].reindex(exposure.index).ffill().fillna('NORMAL'),
        'policy_daily': policy_daily,
        'bench': {'QQQ_r': qqq_r, 'QQQ_eq': qqq_eq},
        'crisis_state': crisis_state,
        'crisis_scale': crisis_scale,
        'turb_scale': turb_scale,
        'corr_scale': corr_scale,
        'corr_rho': corr_rho,
        'corr_state': corr_state,
        'vol_scale': vol_sc,
        'total_scale_target': total_scale,
        'stop_hits': stop_hits,
    }
