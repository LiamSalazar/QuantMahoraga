from __future__ import annotations

import numpy as np
import pandas as pd

from mahoraga9_config import Mahoraga9Config


def _future_return(weekly_r: pd.Series, horizon_weeks: int) -> pd.Series:
    vals = []
    arr = weekly_r.fillna(0.0).to_numpy(dtype=float)
    n = len(arr)
    for i in range(n):
        j = min(n, i + horizon_weeks + 1)
        if j <= i + 1:
            vals.append(np.nan)
        else:
            vals.append(float(np.prod(1.0 + arr[i + 1:j]) - 1.0))
    return pd.Series(vals, index=weekly_r.index, dtype=float)


def build_labels(weekly_ctx: pd.DataFrame, cfg: Mahoraga9Config) -> pd.DataFrame:
    base_r = weekly_ctx["base_r"].fillna(0.0)
    fut_frag = _future_return(base_r, cfg.fragility_horizon_weeks)
    fut_recv = _future_return(base_r, cfg.recovery_horizon_weeks)
    draw_proxy = weekly_ctx["qqq_drawdown"].fillna(0.0)
    stress_proxy = 0.5 * weekly_ctx["avg_corr_21"] + 0.5 * weekly_ctx["vix_z_63"]

    frag_thr = float(fut_frag.quantile(0.30)) if fut_frag.notna().any() else 0.0
    recv_thr = float(fut_recv.quantile(0.70)) if fut_recv.notna().any() else 0.0

    out = pd.DataFrame(index=weekly_ctx.index)
    out["fragility_y"] = ((fut_frag <= frag_thr) | (stress_proxy >= float(stress_proxy.quantile(0.75)))).astype(int)
    out["recovery_y"] = ((fut_recv >= recv_thr) & (draw_proxy <= float(draw_proxy.quantile(0.50)))).astype(int)
    out["fwd_fragility_ret"] = fut_frag
    out["fwd_recovery_ret"] = fut_recv
    return out
