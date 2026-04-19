from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from mahoraga10_config import Mahoraga10Config


def build_hawkes_features(weekly_df: pd.DataFrame, decay: float, cfg: Mahoraga10Config) -> pd.DataFrame:
    out = pd.DataFrame(index=weekly_df.index)
    train_idx = weekly_df.index
    q_low = float(weekly_df["qqq_ret_2w"].quantile(cfg.hawkes_event_q_low)) if len(weekly_df) else -0.02
    q_high = float(weekly_df["qqq_ret_2w"].quantile(cfg.hawkes_event_q_high)) if len(weekly_df) else 0.02

    stress_event = (
        (weekly_df["qqq_ret_2w"] <= q_low)
        | (weekly_df["qqq_drawdown"] <= weekly_df["qqq_drawdown"].quantile(0.25))
        | (weekly_df["vix_z_63"] >= 1.0)
        | (weekly_df["breadth_63"] <= weekly_df["breadth_63"].quantile(0.25))
    ).astype(float)
    recovery_event = (
        (weekly_df["qqq_ret_2w"] >= q_high)
        & (weekly_df["breadth_63"] >= weekly_df["breadth_63"].quantile(0.55))
        & (weekly_df["qqq_drawdown"] > weekly_df["qqq_drawdown"].shift(1).fillna(method="bfill"))
    ).astype(float)

    stress_i = np.zeros(len(weekly_df), dtype=float)
    recv_i = np.zeros(len(weekly_df), dtype=float)
    for i in range(1, len(weekly_df)):
        stress_i[i] = decay * stress_i[i - 1] + float(stress_event.iloc[i - 1])
        recv_i[i] = decay * recv_i[i - 1] + float(recovery_event.iloc[i - 1])
    out["stress_event"] = stress_event
    out["recovery_event"] = recovery_event
    out["stress_intensity"] = pd.Series(stress_i, index=weekly_df.index).clip(0.0, 5.0)
    out["recovery_intensity"] = pd.Series(recv_i, index=weekly_df.index).clip(0.0, 5.0)
    return out
