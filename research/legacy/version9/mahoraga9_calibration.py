from __future__ import annotations

from itertools import product as iproduct
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from mahoraga9_config import Mahoraga9Config
from mahoraga9_risk import build_policy_table
from mahoraga9_utils import annualize, cvar5, intervention_rate, max_dd, sharpe


def _cheap_objective(base_weekly_r: pd.Series, policy_weekly: pd.DataFrame, frag_p: pd.Series, cfg: Mahoraga9Config) -> Dict[str, float]:
    scaled_r = base_weekly_r.reindex(policy_weekly.index).fillna(0.0) * policy_weekly["gate_scale"] * policy_weekly["vol_mult"]
    sh = sharpe(scaled_r, td=52)
    cg = annualize(scaled_r, td=52)
    dd = max_dd(scaled_r)
    cv = cvar5(scaled_r)
    ir = intervention_rate(policy_weekly["gate_scale"])
    alpha_proxy = float((scaled_r - base_weekly_r.reindex(policy_weekly.index).fillna(0.0)).mean() * 52.0)
    util = (
        cfg.score_w_sharpe * sh
        + cfg.score_w_cagr * cg
        + cfg.score_w_alpha * alpha_proxy
        - cfg.score_pen_maxdd * abs(dd)
        - cfg.score_pen_cvar * abs(cv)
        - cfg.score_pen_intervention * max(0.0, ir - cfg.max_allowed_intervention_rate)
    )
    return {
        "utility": float(util),
        "sharpe": float(sh),
        "cagr": float(cg),
        "maxdd": float(dd),
        "cvar5": float(cv),
        "intervention_rate": float(ir),
    }


def calibrate_policy(weekly_df: pd.DataFrame, cfg: Mahoraga9Config) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows = []
    keys = list(cfg.candidate_grid().keys())
    values = [cfg.candidate_grid()[k] for k in keys]
    for combo in iproduct(*values):
        params = {k: float(v) for k, v in zip(keys, combo)}
        pol = build_policy_table(weekly_df, params, cfg)
        obj = _cheap_objective(weekly_df["base_r"], pol, weekly_df["fragility_p"], cfg)
        rows.append({**params, **obj})
    calib = pd.DataFrame(rows).sort_values(["utility", "sharpe"], ascending=[False, False]).reset_index(drop=True)
    best = calib.iloc[0].to_dict() if len(calib) else {k: float(v[0]) for k, v in cfg.candidate_grid().items()}
    best_params = {k: float(best[k]) for k in keys}
    return best_params, calib
