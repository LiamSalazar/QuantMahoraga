from __future__ import annotations

import polars as pl

from .config import OFFICIAL_CANDIDATE_ID
from .date_utils import parse_date


def build_fact_path_recursive(sources: dict[str, pl.DataFrame], run_id: str) -> pl.DataFrame:
    active = sources.get("active_return", pl.DataFrame())
    decision = sources.get("decision", pl.DataFrame())
    if active.is_empty() or "CumOfficial" not in active.columns:
        return pl.DataFrame()
    out = active.with_columns(
        date_value=parse_date("Date"),
        candidate_id=pl.lit(OFFICIAL_CANDIDATE_ID),
        equity=1.0 + pl.col("CumOfficial").cast(pl.Float64, strict=False),
        run_id=pl.lit(run_id),
        demo_mode=pl.lit(False),
    ).select("candidate_id", "date_value", "equity", "run_id", "demo_mode")
    if not decision.is_empty() and {"date", "fold"}.issubset(decision.columns):
        folds = decision.select(
            parse_date("date").alias("date_value"),
            pl.col("fold").cast(pl.Int32),
        ).unique(subset=["date_value"], keep="first")
        out = out.join(folds, on="date_value", how="left")
    else:
        out = out.with_columns(fold=pl.lit(None, dtype=pl.Int32))
    rows = []
    peak = None
    duration = 0
    episode = 0
    in_drawdown = False
    for row in out.sort("date_value").iter_rows(named=True):
        equity = float(row["equity"]) if row["equity"] is not None else 1.0
        if peak is None or equity >= peak:
            peak = equity
            recovery_days = duration if in_drawdown else 0
            duration = 0
            state_entry = False
            state_exit = in_drawdown
            in_drawdown = False
        else:
            if not in_drawdown:
                episode += 1
                state_entry = True
                duration = 1
            else:
                state_entry = False
                duration += 1
            state_exit = False
            recovery_days = None
            in_drawdown = True
        drawdown = equity / peak - 1.0 if peak else 0.0
        rows.append(
            {
                **row,
                "rolling_peak": peak,
                "drawdown": drawdown,
                "drawdown_duration": duration,
                "recovery_days": recovery_days,
                "state_entry": state_entry,
                "state_exit": state_exit,
                "path_episode_id": f"dd_{episode:04d}" if in_drawdown or state_exit else "peak",
            }
        )
    return pl.DataFrame(rows)
