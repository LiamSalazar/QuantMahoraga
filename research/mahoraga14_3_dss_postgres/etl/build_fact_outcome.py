from __future__ import annotations

import polars as pl

from .date_utils import parse_date


def build_fact_outcome(sources: dict[str, pl.DataFrame], run_id: str) -> pl.DataFrame:
    outcome = sources.get("outcome", pl.DataFrame())
    if outcome.is_empty():
        return pl.DataFrame()
    return outcome.with_columns(
        decision_date=parse_date("decision_date"),
        alpha_vs_qqq=pl.col("realized_alpha_vs_qqq").cast(pl.Float64, strict=False),
        alpha_vs_spy=pl.col("realized_alpha_vs_spy").cast(pl.Float64, strict=False),
        qqq_return=(pl.col("realized_return").cast(pl.Float64, strict=False) - pl.col("realized_alpha_vs_qqq").cast(pl.Float64, strict=False)),
        spy_return=(pl.col("realized_return").cast(pl.Float64, strict=False) - pl.col("realized_alpha_vs_spy").cast(pl.Float64, strict=False)),
        helped_flag=pl.col("decision_helped_flag_vs_qqq").cast(pl.Boolean, strict=False),
        drawdown_change=pl.col("realized_drawdown_change").cast(pl.Float64, strict=False),
        exposure_adjusted_outcome=(
            pl.col("realized_return").cast(pl.Float64, strict=False) / pl.when(pl.col("realized_exposure").cast(pl.Float64, strict=False).abs() > 1e-9).then(pl.col("realized_exposure").cast(pl.Float64, strict=False)).otherwise(None)
        ),
        run_id=pl.lit(run_id),
        demo_mode=pl.lit(False),
    ).select(
        "decision_date",
        "candidate_id",
        "fold",
        "universe_id",
        "horizon",
        "realized_return",
        "qqq_return",
        "spy_return",
        "alpha_vs_qqq",
        "alpha_vs_spy",
        "helped_flag",
        "drawdown_change",
        "exposure_adjusted_outcome",
        "realized_turnover",
        "realized_exposure",
        "run_id",
        "demo_mode",
    )
