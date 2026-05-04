from __future__ import annotations

import numpy as np
import pandas as pd

from mahoraga9_config import Mahoraga9Config
from mahoraga9_utils import zscore_train


def _exp_intensity(events: pd.Series, decay: float) -> pd.Series:
    x = events.astype(float).fillna(0.0).to_numpy()
    out = np.zeros(len(x), dtype=float)
    for i in range(1, len(x)):
        out[i] = decay * out[i - 1] + x[i - 1]
    return pd.Series(out, index=events.index, dtype=float)


def build_hawkes_signals(weekly_ctx: pd.DataFrame, cfg: Mahoraga9Config, train_index: pd.Index) -> pd.DataFrame:
    stress_score = (
        0.35 * zscore_train(weekly_ctx["vix_z_63"], train_index)
        + 0.25 * zscore_train(-weekly_ctx["qqq_ret_5"], train_index)
        + 0.20 * zscore_train(-weekly_ctx["breadth_63"], train_index)
        + 0.20 * zscore_train(weekly_ctx["avg_corr_21"], train_index)
    )
    recovery_score = (
        0.35 * zscore_train(weekly_ctx["qqq_ret_5"], train_index)
        + 0.25 * zscore_train(weekly_ctx["breadth_63"], train_index)
        + 0.20 * zscore_train(-weekly_ctx["vix_z_63"], train_index)
        + 0.20 * zscore_train(-weekly_ctx["qqq_drawdown"], train_index)
    )
    stress_base = stress_score.reindex(train_index).dropna()
    recovery_base = recovery_score.reindex(train_index).dropna()
    stress_thr = float(stress_base.quantile(cfg.hawkes_stress_q)) if len(stress_base) else 0.0
    recovery_thr = float(recovery_base.quantile(cfg.hawkes_recovery_q)) if len(recovery_base) else 0.0
    stress_evt = (stress_score >= stress_thr).astype(float) * cfg.hawkes_stress_scale
    recovery_evt = (recovery_score >= recovery_thr).astype(float) * cfg.hawkes_recovery_scale

    out = pd.DataFrame(index=weekly_ctx.index)
    out["stress_score"] = stress_score.fillna(0.0)
    out["recovery_score"] = recovery_score.fillna(0.0)
    out["stress_event"] = stress_evt
    out["recovery_event"] = recovery_evt
    out["stress_intensity"] = _exp_intensity(stress_evt, cfg.hawkes_decay)
    out["recovery_intensity"] = _exp_intensity(recovery_evt, cfg.hawkes_decay)
    return out
