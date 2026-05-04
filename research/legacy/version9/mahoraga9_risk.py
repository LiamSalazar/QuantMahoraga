from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from mahoraga9_config import Mahoraga9Config


def build_policy_table(weekly_df: pd.DataFrame, params: Dict[str, float], cfg: Mahoraga9Config) -> pd.DataFrame:
    out = pd.DataFrame(index=weekly_df.index)
    frag_p = weekly_df["fragility_p"].clip(0.0, 1.0)
    recv_p = weekly_df["recovery_p"].clip(0.0, 1.0)
    stress_i = weekly_df["stress_intensity"].clip(lower=0.0)
    recv_i = weekly_df["recovery_intensity"].clip(lower=0.0)
    hawkes_blend = params["hawkes_weight"] * stress_i.rank(pct=True) - params["hawkes_weight"] * recv_i.rank(pct=True)

    gate = np.where(frag_p >= params["fragility_prob_thr"], params["fragility_gate_low"], 1.0)
    gate = np.where(recv_p >= params["recovery_prob_thr"], np.maximum(gate, 0.95), gate)
    gate = np.clip(gate - 0.10 * hawkes_blend, params["fragility_gate_low"], 1.05)

    alpha_mix = cfg.alpha_mix_default + 0.25 * (recv_p - frag_p)
    alpha_mix = alpha_mix.clip(cfg.alpha_mix_min, cfg.alpha_mix_max)

    vol_mult = 1.0 - 0.30 * frag_p + 0.15 * recv_p - 0.10 * stress_i.rank(pct=True)
    vol_mult = vol_mult.clip(params["vol_target_floor"], 1.05)

    exp_cap = 1.0 - 0.35 * frag_p + 0.20 * recv_p
    exp_cap = exp_cap.clip(params["exposure_floor"], params["exposure_ceiling"])

    if cfg.use_corr_as_secondary_veto:
        corr_hit = weekly_df["avg_corr_21"] >= cfg.corr_secondary_rho
        if cfg.corr_use_vix_confirm:
            corr_hit = corr_hit & (weekly_df["vix_level"] >= cfg.corr_vix_confirm_level)
        exp_cap = pd.Series(exp_cap, index=weekly_df.index)
        exp_cap.loc[corr_hit] = np.minimum(exp_cap.loc[corr_hit], cfg.corr_secondary_scale)

    out["gate_scale"] = pd.Series(gate, index=weekly_df.index).astype(float)
    out["alpha_mix"] = pd.Series(alpha_mix, index=weekly_df.index).astype(float)
    out["vol_mult"] = pd.Series(vol_mult, index=weekly_df.index).astype(float)
    out["exp_cap"] = pd.Series(exp_cap, index=weekly_df.index).astype(float)
    return out


def weekly_to_daily_gate(policy_weekly: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    cols = ["gate_scale", "alpha_mix", "vol_mult", "exp_cap"]
    out = policy_weekly[cols].reindex(daily_index).ffill().bfill()
    out.index = daily_index
    return out
