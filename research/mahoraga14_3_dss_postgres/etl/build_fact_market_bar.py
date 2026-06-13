from __future__ import annotations

import polars as pl

from .date_utils import parse_date


def build_fact_market_bar(sources: dict[str, pl.DataFrame], run_id: str) -> pl.DataFrame:
    market = sources.get("market_context", pl.DataFrame())
    if market.is_empty():
        return pl.DataFrame()
    qqq = market.select(
        parse_date("date").alias("date_value"),
        pl.lit("QQQ").alias("ticker"),
        pl.col("qqq_return").cast(pl.Float64, strict=False).alias("close_return"),
        pl.col("qqq_drawdown").cast(pl.Float64, strict=False).alias("drawdown"),
        pl.col("qqq_vol").cast(pl.Float64, strict=False).alias("realized_vol"),
        pl.col("benchmark_strength").cast(pl.Float64, strict=False),
        pl.col("benchmark_weakness").cast(pl.Float64, strict=False),
        pl.col("market_regime_proxy").cast(pl.Utf8).alias("market_regime"),
        pl.lit(run_id).alias("run_id"),
        pl.lit(False).alias("demo_mode"),
    )
    spy = market.select(
        parse_date("date").alias("date_value"),
        pl.lit("SPY").alias("ticker"),
        pl.col("spy_return").cast(pl.Float64, strict=False).alias("close_return"),
        pl.col("spy_drawdown").cast(pl.Float64, strict=False).alias("drawdown"),
        pl.lit(None, dtype=pl.Float64).alias("realized_vol"),
        pl.col("benchmark_strength").cast(pl.Float64, strict=False),
        pl.col("benchmark_weakness").cast(pl.Float64, strict=False),
        pl.col("market_regime_proxy").cast(pl.Utf8).alias("market_regime"),
        pl.lit(run_id).alias("run_id"),
        pl.lit(False).alias("demo_mode"),
    )
    return pl.concat([qqq, spy], how="diagonal_relaxed")
