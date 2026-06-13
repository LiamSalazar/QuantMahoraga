from __future__ import annotations

import polars as pl

from .date_utils import parse_date


def build_fact_position_daily(sources: dict[str, pl.DataFrame], run_id: str) -> pl.DataFrame:
    position = sources.get("position", pl.DataFrame())
    if position.is_empty():
        return pl.DataFrame()
    return position.with_columns(
        date_value=parse_date("date"),
        target_weight=pl.col("base_weight").cast(pl.Float64, strict=False),
        weight_after_stop=pl.when(pl.col("stop_flag").cast(pl.Int8, strict=False) == 1).then(0.0).otherwise(pl.col("leader_adjusted_weight").cast(pl.Float64, strict=False)),
        weight_exec_1x=pl.col("leader_adjusted_weight").cast(pl.Float64, strict=False),
        stop_active=pl.col("stop_flag").cast(pl.Boolean, strict=False),
        leader_flag=pl.col("leader_flag").cast(pl.Boolean, strict=False),
        selected_flag=pl.col("selected_flag").cast(pl.Boolean, strict=False),
        final_score=pl.col("base_score").cast(pl.Float64, strict=False),
        run_id=pl.lit(run_id),
        demo_mode=pl.lit(False),
    ).select(
        "date_value",
        "candidate_id",
        "fold",
        "universe_id",
        "ticker",
        "target_weight",
        "weight_after_stop",
        "weight_exec_1x",
        "final_weight",
        "pnl_contribution",
        "stop_active",
        "leader_flag",
        "selected_flag",
        "final_score",
        "rank",
        "run_id",
        "demo_mode",
    )
