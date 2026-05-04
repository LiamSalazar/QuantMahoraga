from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from mahoraga14_config import Mahoraga14Config


def derive_participation_engine_params(selected_candidate: Dict[str, Any], cfg: Mahoraga14Config) -> Dict[str, float]:
    base_mix = float(selected_candidate.get("base_mix", 0.30))
    base_beta_penalty = float(selected_candidate.get("base_beta_penalty", 0.00))
    raw_rel_boost = float(selected_candidate.get("raw_rel_boost", 1.00))
    return {
        "mix": float(np.clip(base_mix + cfg.participation_mix_lift, 0.18, 0.55)),
        "beta_penalty": float(np.clip(base_beta_penalty * cfg.participation_beta_penalty_mult, 0.0, base_beta_penalty + 0.02)),
        "raw_rel_boost": float(np.clip(raw_rel_boost + cfg.participation_raw_rel_boost_lift, 1.00, 1.40)),
    }


def _rebudget_row(
    row: pd.Series,
    target_budget: float,
    cfg: Mahoraga14Config,
) -> Tuple[pd.Series, float]:
    clean = pd.Series(row, dtype=float).clip(lower=0.0).fillna(0.0)
    gross = float(clean.sum())
    if gross <= 1e-12 or target_budget <= 1e-12:
        return pd.Series(0.0, index=clean.index, dtype=float), gross

    target = float(np.clip(target_budget, 0.0, 1.0))
    scale = min(float(cfg.participation_cash_redeploy_scale_cap), target / gross if gross > 0 else 0.0)
    scaled = clean * scale
    capped = scaled.clip(upper=float(cfg.participation_max_name_weight))
    residual = max(0.0, target - float(capped.sum()))

    if residual > 1e-10:
        headroom = (float(cfg.participation_max_name_weight) - capped).clip(lower=0.0)
        headroom = headroom[headroom > float(cfg.participation_redeploy_headroom_floor)]
        if len(headroom):
            alloc = residual * headroom / float(headroom.sum())
            capped.loc[headroom.index] = capped.loc[headroom.index] + alloc.values

    total = float(capped.sum())
    if total > target and total > 0.0:
        capped = capped * (target / total)
    return capped.clip(lower=0.0), gross


def apply_bull_participation_layer(
    base_weights_exec_1x: pd.DataFrame,
    participation_weights_exec_1x: pd.DataFrame,
    allocator_daily: pd.DataFrame,
    cfg: Mahoraga14Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = base_weights_exec_1x.index
    cols = base_weights_exec_1x.columns
    leader_blend = pd.Series(allocator_daily.get("leader_blend", 0.0), index=idx, dtype=float).fillna(0.0).clip(0.0, float(cfg.participation_leader_blend_max))
    long_budget = pd.Series(allocator_daily.get("long_budget", 0.0), index=idx, dtype=float).fillna(0.0).clip(0.0, 1.0)

    blended = base_weights_exec_1x.mul(1.0 - leader_blend, axis=0) + participation_weights_exec_1x.mul(leader_blend, axis=0)
    blended = blended.reindex(index=idx, columns=cols).fillna(0.0).clip(lower=0.0)

    final_weights = pd.DataFrame(0.0, index=idx, columns=cols)
    diag_rows = []
    for dt in idx:
        row = blended.loc[dt]
        budget = float(long_budget.loc[dt])
        reweighted, gross_before = _rebudget_row(row, budget, cfg)
        final_weights.loc[dt] = reweighted.values
        gross_after = float(reweighted.sum())
        diag_rows.append(
            {
                "Date": dt,
                "leader_blend": float(leader_blend.loc[dt]),
                "target_long_budget": budget,
                "gross_before_budget": gross_before,
                "gross_after_budget": gross_after,
                "cash_drag_before": max(0.0, 1.0 - gross_before),
                "cash_drag_after": max(0.0, 1.0 - gross_after),
                "cash_redeployed": max(0.0, gross_after - gross_before),
            }
        )
    diagnostics = pd.DataFrame(diag_rows).set_index("Date")
    return final_weights, diagnostics
