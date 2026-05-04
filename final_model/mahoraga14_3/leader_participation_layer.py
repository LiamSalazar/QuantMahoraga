from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from mahoraga14_config import Mahoraga14Config


def derive_leader_participation_engine_params(selected_candidate: Dict[str, Any], cfg: Mahoraga14Config) -> Dict[str, float]:
    base_mix = float(selected_candidate.get("base_mix", 0.30))
    base_beta_penalty = float(selected_candidate.get("base_beta_penalty", 0.00))
    raw_rel_boost = float(selected_candidate.get("raw_rel_boost", 1.00))
    return {
        "mix": float(np.clip(base_mix + cfg.participation_mix_lift, 0.18, 0.60)),
        "beta_penalty": float(np.clip(base_beta_penalty * cfg.participation_beta_penalty_mult, 0.0, base_beta_penalty + 0.01)),
        "raw_rel_boost": float(np.clip(raw_rel_boost + cfg.participation_raw_rel_boost_lift, 1.00, 1.50)),
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
    scale = min(float(cfg.participation_cash_redeploy_scale_cap), target / gross if gross > 0.0 else 0.0)
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


def _leader_signal(
    base_row: pd.Series,
    participation_row: pd.Series,
    trailing_ret_row: pd.Series,
) -> pd.Series:
    gap = (participation_row - base_row).clip(lower=0.0)
    trailing = pd.Series(trailing_ret_row, index=base_row.index, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    trailing = trailing.rank(pct=True).fillna(0.0)
    participation_rank = participation_row.rank(pct=True).fillna(0.0)
    score = 0.45 * participation_rank + 0.35 * trailing + 0.20 * gap.rank(pct=True).fillna(0.0)
    return score.clip(lower=0.0)


def apply_leader_participation_layer(
    base_weights_exec_1x: pd.DataFrame,
    participation_weights_exec_1x: pd.DataFrame,
    allocator_daily: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cfg: Mahoraga14Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = base_weights_exec_1x.index
    cols = base_weights_exec_1x.columns
    leader_blend = pd.Series(allocator_daily.get("leader_blend", 0.0), index=idx, dtype=float).fillna(0.0).clip(0.0, float(cfg.participation_leader_blend_max))
    long_budget = pd.Series(allocator_daily.get("long_budget", cfg.participation_long_budget_base), index=idx, dtype=float).fillna(float(cfg.participation_long_budget_base)).clip(0.0, 1.0)
    cash_budget_target = pd.Series(
        allocator_daily.get("cash_budget_target", allocator_daily.get("long_budget", cfg.participation_long_budget_base)),
        index=idx,
        dtype=float,
    ).fillna(float(cfg.participation_long_budget_base)).clip(0.0, 1.0)
    conviction_multiplier = pd.Series(allocator_daily.get("conviction_multiplier", 1.0), index=idx, dtype=float).fillna(1.0).clip(1.0, float(cfg.conviction_weight_scale_max))
    leader_multiplier = pd.Series(allocator_daily.get("leader_multiplier", 1.0), index=idx, dtype=float).fillna(1.0).clip(1.0, 1.0 + float(cfg.conviction_leader_boost_max))

    trailing_returns = (
        (1.0 + asset_returns.reindex(idx).fillna(0.0))
        .rolling(int(cfg.leader_return_lookback), min_periods=5)
        .apply(np.prod, raw=True)
        .sub(1.0)
        .fillna(0.0)
    )

    final_weights = pd.DataFrame(0.0, index=idx, columns=cols)
    diag_rows = []
    top_k = max(1, int(cfg.leader_top_k))
    for dt in idx:
        base_row = base_weights_exec_1x.loc[dt].reindex(cols).fillna(0.0).clip(lower=0.0)
        participation_row = participation_weights_exec_1x.loc[dt].reindex(cols).fillna(0.0).clip(lower=0.0)
        blend = float(leader_blend.loc[dt])
        blended = base_row.mul(1.0 - blend) + participation_row.mul(blend)
        blended = blended.clip(lower=0.0)

        conviction = float(conviction_multiplier.loc[dt])
        if conviction > 1.0:
            blended = blended * conviction

        leader_signal = _leader_signal(base_row, participation_row, trailing_returns.loc[dt].reindex(cols).fillna(0.0))
        leader_gap = (participation_row - base_row).clip(lower=0.0)
        eligible_mask = (participation_row >= float(cfg.leader_gap_floor)) | (leader_gap >= float(cfg.leader_gap_floor))
        signal_pool = leader_signal[eligible_mask]
        signal_top = signal_pool.nlargest(min(top_k, len(signal_pool)))
        top_names = list(signal_top.index)
        top_share_before = float(blended.loc[top_names].sum()) if top_names else 0.0
        gap_share = float(leader_gap.loc[top_names].sum()) if top_names else 0.0

        if top_names and blend > 0.0:
            extra_strength = float(cfg.leader_tilt_strength_max) * blend * (0.10 + max(0.0, float(leader_multiplier.loc[dt]) - 1.0))
            signal_mass = float(signal_top.sum())
            if signal_mass > 1e-12:
                bump = extra_strength * signal_top / signal_mass
                blended.loc[top_names] = blended.loc[top_names] + bump.values

        target_budget = max(float(long_budget.loc[dt]), float(cash_budget_target.loc[dt]))
        reweighted, gross_before = _rebudget_row(blended, target_budget, cfg)
        gross_after = float(reweighted.sum())
        leader_active_weight = float(reweighted.loc[top_names].sum()) if top_names else 0.0
        final_weights.loc[dt] = reweighted.values
        diag_rows.append(
            {
                "Date": dt,
                "leader_blend": blend,
                "conviction_multiplier": conviction,
                "leader_multiplier": float(leader_multiplier.loc[dt]),
                "target_long_budget": float(long_budget.loc[dt]),
                "cash_budget_target": float(cash_budget_target.loc[dt]),
                "gross_before_budget": gross_before,
                "gross_after_budget": gross_after,
                "cash_drag_before": max(0.0, 1.0 - gross_before),
                "cash_drag_after": max(0.0, 1.0 - gross_after),
                "cash_redeployed": max(0.0, gross_after - gross_before),
                "leader_top_share_before": top_share_before,
                "leader_active_weight": leader_active_weight,
                "leader_gap_share": gap_share,
                "leader_names_count": len(top_names),
            }
        )
    diagnostics = pd.DataFrame(diag_rows).set_index("Date")
    return final_weights, diagnostics
