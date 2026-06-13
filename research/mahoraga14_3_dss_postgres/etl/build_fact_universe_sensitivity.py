from __future__ import annotations

import polars as pl


def build_fact_universe_sensitivity(sources: dict[str, pl.DataFrame], run_id: str) -> pl.DataFrame:
    universe = sources.get("universe_summary", pl.DataFrame())
    if universe.is_empty():
        return pl.DataFrame()
    candidate_col = "candidate_id" if "candidate_id" in universe.columns else "CandidateId"
    return universe.select(
        pl.col(candidate_col).cast(pl.Utf8).alias("candidate_id"),
        pl.col("universe_id").cast(pl.Utf8),
        pl.col("proposed_count").cast(pl.Int32, strict=False),
        pl.col("usable_count").cast(pl.Int32, strict=False),
        pl.col("mean_coverage_ratio").cast(pl.Float64, strict=False),
        pl.col("CAGR").cast(pl.Float64, strict=False).alias("cagr"),
        pl.col("Sharpe").cast(pl.Float64, strict=False).alias("sharpe"),
        pl.col("MaxDD").cast(pl.Float64, strict=False).alias("maxdd"),
        pl.col("run_status").cast(pl.Utf8),
        pl.lit(run_id).alias("run_id"),
        pl.lit(False).alias("demo_mode"),
    )
