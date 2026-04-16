from __future__ import annotations

import numpy as np
import pandas as pd

from mahoraga9_config import Mahoraga9Config


def _exp_intensity(events: pd.Series, decay: float) -> pd.Series:
    x = events.astype(float).fillna(0.0).to_numpy()
    out = np.zeros(len(x), dtype=float)
    for i in range(1, len(x)):
        out[i] = decay * out[i - 1] + x[i - 1]
    return pd.Series(out, index=events.index, dtype=float)


def build_hawkes_signals(weekly_ctx: pd.DataFrame, cfg: Mahoraga9Config) -> pd.DataFrame:
    stress_score = (
        0.35 * weekly_ctx["vix_z_63"]
        + 0.25 * (-weekly_ctx["qqq_ret_5"])
        + 0.20 * (-weekly_ctx["breadth_63"])
        + 0.20 * weekly_ctx["avg_corr_21"]
    )
    recovery_score = (
        0.35 * weekly_ctx["qqq_ret_5"]
        + 0.25 * weekly_ctx["breadth_63"]
        + 0.20 * (-weekly_ctx["vix_z_63"])
        + 0.20 * (-weekly_ctx["qqq_drawdown"])
    )
    stress_thr = float(stress_score.quantile(cfg.hawkes_stress_q))
    recovery_thr = float(recovery_score.quantile(cfg.hawkes_recovery_q))
    stress_evt = (stress_score >= stress_thr).astype(float) * cfg.hawkes_stress_scale
    recovery_evt = (recovery_score >= recovery_thr).astype(float) * cfg.hawkes_recovery_scale

    out = pd.DataFrame(index=weekly_ctx.index)
    out["stress_score"] = stress_score
    out["recovery_score"] = recovery_score
    out["stress_event"] = stress_evt
    out["recovery_event"] = recovery_evt
    out["stress_intensity"] = _exp_intensity(stress_evt, cfg.hawkes_decay)
    out["recovery_intensity"] = _exp_intensity(recovery_evt, cfg.hawkes_decay)
    return out
