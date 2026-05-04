from __future__ import annotations

from itertools import product as iproduct
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from mahoraga9_config import Mahoraga9Config
from mahoraga9_utils import annualize, bhy_qvalues, cvar5, differential_alpha_proxy, intervention_rate, max_dd, paired_ttest_pvalue, sharpe


def iter_policy_candidates(cfg: Mahoraga9Config) -> List[Dict[str, float]]:
    grid = cfg.candidate_grid()
    keys = list(grid.keys())
    out = []
    for combo in iproduct(*[grid[k] for k in keys]):
        out.append({k: float(v) for k, v in zip(keys, combo)})
    return out


def candidate_metrics(model_r: pd.Series, base_r: pd.Series, exposure: pd.Series, policy_daily: pd.DataFrame, cfg: Mahoraga9Config, fold_n: int) -> Dict[str, float]:
    model_r = model_r.fillna(0.0)
    base_r = base_r.reindex(model_r.index).fillna(0.0)
    dd_model = max_dd(model_r)
    dd_base = max_dd(base_r)
    ir = intervention_rate(policy_daily["is_intervening"])
    exp_mean = float(exposure.reindex(model_r.index).fillna(0.0).mean()) if len(exposure) else 0.0
    exp_collapse = max(0.0, 1.0 - exp_mean / max(float(exposure.reindex(model_r.index).fillna(0.0).mean()), 1e-8))
    diff = model_r - base_r
    p_value = paired_ttest_pvalue(diff, alternative="greater")

    utility = (
        cfg.score_w_sharpe * sharpe(model_r)
        + cfg.score_w_cagr * annualize(model_r)
        + cfg.score_w_alpha * differential_alpha_proxy(model_r, base_r)
        - cfg.score_pen_maxdd * abs(dd_model)
        - cfg.score_pen_cvar * abs(cvar5(model_r))
        - cfg.score_pen_intervention * max(0.0, ir - cfg.max_allowed_intervention_rate)
        - cfg.score_pen_exposure_collapse * exp_collapse
    )
    if fold_n in cfg.floor_folds and dd_model - dd_base > cfg.max_allowed_fold_dd_worsening:
        utility -= 0.25
    return {
        "utility_pre_q": float(utility),
        "val_sharpe": float(sharpe(model_r)),
        "val_cagr": float(annualize(model_r)),
        "val_maxdd": float(dd_model),
        "val_cvar5": float(cvar5(model_r)),
        "val_alpha_proxy": float(differential_alpha_proxy(model_r, base_r)),
        "val_pvalue": float(p_value),
        "intervention_rate": float(ir),
        "mean_exposure": float(exp_mean),
        "dd_delta": float(dd_model - dd_base),
    }


def rank_candidates(rows: Iterable[Dict[str, float]], cfg: Mahoraga9Config, fold_n: int) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if len(df) == 0:
        return df
    q = bhy_qvalues(df["val_pvalue"].fillna(1.0).tolist())
    df["val_qvalue"] = q
    df["utility"] = df["utility_pre_q"] - cfg.score_pen_qvalue * df["val_qvalue"]
    df = df.sort_values(["utility", "val_sharpe", "val_cagr"], ascending=[False, False, False]).reset_index(drop=True)
    return df
