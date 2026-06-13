from __future__ import annotations

import polars as pl

from .date_utils import parse_date


def build_fact_signal_daily(sources: dict[str, pl.DataFrame], run_id: str) -> pl.DataFrame:
    position = sources.get("position", pl.DataFrame())
    if position.is_empty():
        return pl.DataFrame()
    return position.with_columns(
        date_value=parse_date("date"),
        trend=pl.col("raw_trend").cast(pl.Float64, strict=False),
        momentum=pl.col("raw_momentum").cast(pl.Float64, strict=False),
        residual_trend=pl.col("residual_score").cast(pl.Float64, strict=False),
        residual_momentum=pl.col("residual_score").cast(pl.Float64, strict=False),
        final_score=pl.col("base_score").cast(pl.Float64, strict=False),
        selected_flag=pl.col("selected_flag").cast(pl.Boolean, strict=False),
        run_id=pl.lit(run_id),
        demo_mode=pl.lit(False),
    ).select(
        "date_value",
        "ticker",
        "candidate_id",
        "fold",
        "universe_id",
        "trend",
        "momentum",
        "relative_momentum",
        "residual_trend",
        "residual_momentum",
        "beta_drag",
        "final_score",
        "rank",
        "selected_flag",
        "run_id",
        "demo_mode",
    )
