from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from mahoraga8_config import Mahoraga8Config


def _state_params(cfg: Mahoraga8Config, state_map: str) -> Dict[str, Dict[str, float]]:
    p = cfg.regime_defaults()
    if state_map == 'defensive_plus':
        p['STRESS']['max_exposure'] *= 0.90
        p['PANIC']['max_exposure'] *= 0.85
        p['STRESS']['vol_target'] = max(0.10, p['STRESS']['vol_target'] - 0.02)
        p['PANIC']['vol_target'] = max(0.08, p['PANIC']['vol_target'] - 0.02)
    elif state_map == 'recovery_plus':
        p['RECOVERY']['max_exposure'] = min(1.0, p['RECOVERY']['max_exposure'] + 0.05)
        p['RECOVERY']['vol_target'] = min(0.35, p['RECOVERY']['vol_target'] + 0.02)
        p['RECOVERY']['weight_cap'] = min(0.70, p['RECOVERY']['weight_cap'] + 0.03)
    return p


def _apply_transition_smoothing(policy: pd.DataFrame, cfg: Mahoraga8Config) -> pd.DataFrame:
    if cfg.cp_transition_smoothing <= 1:
        return policy

    out = policy.copy()

    smooth_cols = [
        "active_top_k",
        "active_weight_cap",
        "active_vol_target",
        "active_max_exposure",
        "active_k_atr",
        "active_rel_tilt",
        "risk_budget_cap",
        "active_regime_confidence",
    ]
    smooth_cols = [c for c in smooth_cols if c in out.columns]

    for c in smooth_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").rolling(
            cfg.cp_transition_smoothing,
            min_periods=1
        ).mean()

    if "active_top_k" in out.columns:
        out["active_top_k"] = out["active_top_k"].round().clip(lower=1).astype(int)

    for c in ["active_regime_state", "active_defense_mode", "active_reentry_mode"]:
        if c in out.columns:
            out[c] = out[c].ffill().bfill()

    return out


def _apply_risk_budget_caps(policy: pd.DataFrame) -> pd.DataFrame:
    out = policy.copy()
    out['active_max_exposure'] = np.minimum(out['active_max_exposure'], out['risk_budget_cap'])
    out['active_vol_target'] = out['active_vol_target'] * np.clip(out['risk_budget_cap'], 0.3, 1.0)
    return out


def build_policy_table(regime_table: pd.DataFrame, cfg: Mahoraga8Config, state_map: str = 'default', risk_budget_blend: float = 0.75, exposure_cap_mult: float = 1.0, top_k_shift: int = 0, vol_target_shift: float = 0.0) -> pd.DataFrame:
    state_params = _state_params(cfg, state_map)
    rows = []
    for _, row in regime_table.iterrows():
        state = str(row.get('regime_state', 'NORMAL'))
        state = state if state in state_params else 'NORMAL'
        rp = state_params[state]
        rb = float(row.get('risk_budget', 1.0))
        rb_cap = float(np.clip((1.0 - risk_budget_blend) + risk_budget_blend * rb, cfg.conformal_min_exposure, 1.0))
        rows.append({
            'active_top_k': int(max(1, round(rp['top_k'] + top_k_shift))),
            'active_weight_cap': float(np.clip(rp['weight_cap'], 0.10, 1.0)),
            'active_vol_target': float(max(0.08, rp['vol_target'] + vol_target_shift)),
            'active_max_exposure': float(np.clip(rp['max_exposure'] * exposure_cap_mult, cfg.conformal_min_exposure, 1.0)),
            'active_k_atr': float(max(1.0, rp['k_atr'])),
            'active_rel_tilt': float(np.clip(rp['rel_tilt'], 0.50, 0.70)),
            'risk_budget_cap': rb_cap,
            'active_regime_state': state,
            'active_regime_confidence': float(row.get('regime_confidence', 0.0)),
            'active_defense_mode': 'DEFENSIVE' if state in ('STRESS', 'PANIC') else 'NORMAL',
            'active_reentry_mode': 'FAST' if state == 'RECOVERY' else 'NORMAL',
        })
    policy = pd.DataFrame(rows, index=regime_table.index)
    policy = _apply_transition_smoothing(policy, cfg)
    policy = _apply_risk_budget_caps(policy)
    return policy
