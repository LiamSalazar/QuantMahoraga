from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from mahoraga9_config import Mahoraga9Config


def build_policy_table(weekly_df: pd.DataFrame, params: Dict[str, float], cfg: Mahoraga9Config) -> pd.DataFrame:
    out = pd.DataFrame(index=weekly_df.index)
    frag_p = weekly_df["fragility_p"].clip(0.0, 1.0)
    recv_p = weekly_df["recovery_p"].clip(0.0, 1.0)
    stress_u = weekly_df["stress_intensity"].rank(pct=True).fillna(0.0)
    recv_u = weekly_df["recovery_intensity"].rank(pct=True).fillna(0.0)

    hawkes_weight = float(params["hawkes_weight"])
    frag_score = (frag_p + hawkes_weight * stress_u).clip(0.0, 1.5)
    recv_score = (recv_p + hawkes_weight * recv_u).clip(0.0, 1.5)

    risk_floor = float(params["risk_floor"])
    frag_thr = float(params["fragility_prob_thr"])
    recv_thr = float(params["recovery_prob_thr"])

    frag_sig = frag_score >= frag_thr
    recv_sig = (recv_score >= recv_thr) & (~frag_sig)

    gate_scale = pd.Series(1.0, index=weekly_df.index, dtype=float)
    vol_mult = pd.Series(1.0, index=weekly_df.index, dtype=float)
    exp_cap = pd.Series(1.0, index=weekly_df.index, dtype=float)

    gate_scale.loc[frag_sig] = risk_floor
    vol_mult.loc[frag_sig] = risk_floor
    exp_cap.loc[frag_sig] = risk_floor

    if cfg.use_corr_as_secondary_veto:
        corr_hit = weekly_df["avg_corr_21"] >= cfg.corr_secondary_rho
        if cfg.corr_use_vix_confirm:
            corr_hit = corr_hit & (weekly_df["vix_level"] >= cfg.corr_vix_confirm_level)
        exp_cap.loc[corr_hit] = np.minimum(exp_cap.loc[corr_hit], cfg.corr_secondary_scale)

    alpha_mix = pd.Series(float(params["alpha_mix_base"]), index=weekly_df.index, dtype=float)
    tilt = float(params["alpha_tilt"])
    alpha_mix = alpha_mix + tilt * (recv_score - frag_score)
    alpha_mix = alpha_mix.clip(cfg.alpha_mix_min, cfg.alpha_mix_max)

    out["gate_scale"] = gate_scale.astype(float)
    out["vol_mult"] = vol_mult.astype(float)
    out["exp_cap"] = exp_cap.astype(float)
    out["alpha_mix"] = alpha_mix.astype(float)
    out["frag_score"] = frag_score.astype(float)
    out["recv_score"] = recv_score.astype(float)
    out["is_intervening"] = ((np.abs(out["gate_scale"] - 1.0) > 1e-8) | (np.abs(out["vol_mult"] - 1.0) > 1e-8) | (np.abs(out["exp_cap"] - 1.0) > 1e-8)).astype(int)
    return out


def weekly_to_daily_policy(policy_weekly: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    cols = ["gate_scale", "vol_mult", "exp_cap", "alpha_mix", "is_intervening", "frag_score", "recv_score"]
    out = policy_weekly[cols].reindex(daily_index).ffill().bfill()
    out.index = daily_index
    return out
