from __future__ import annotations

import polars as pl


SURFACE_METRICS = ["CAGR", "Sharpe", "Sortino", "MaxDD", "AlphaNW_QQQ", "AlphaNW_SPY", "AvgExposure", "AvgTurnover"]


def _surface_from_summary(df: pl.DataFrame, source_artifact: str, run_id: str) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame()
    candidate_col = "candidate_id" if "candidate_id" in df.columns else "CandidateId"
    base_cols = [
        pl.col(candidate_col).cast(pl.Utf8).alias("candidate_id"),
        pl.col("universe_id").cast(pl.Utf8),
        pl.col("sweep_role").cast(pl.Utf8),
        pl.col("budget_multiplier").cast(pl.Float64, strict=False),
        pl.col("conviction_multiplier").cast(pl.Float64, strict=False),
        pl.col("leader_multiplier").cast(pl.Float64, strict=False),
        pl.col("backoff_strength").cast(pl.Float64, strict=False),
        pl.col("Sharpe").cast(pl.Float64, strict=False).alias("_sharpe"),
        pl.col("CAGR").cast(pl.Float64, strict=False).alias("_cagr"),
        pl.col("MaxDD").cast(pl.Float64, strict=False).alias("_maxdd"),
    ]
    narrow = df.select(base_cols + [pl.col(metric).cast(pl.Float64, strict=False) for metric in SURFACE_METRICS if metric in df.columns]).melt(
        id_vars=["candidate_id", "universe_id", "sweep_role", "budget_multiplier", "conviction_multiplier", "leader_multiplier", "backoff_strength", "_sharpe", "_cagr", "_maxdd"],
        variable_name="metric_name",
        value_name="metric_value",
    )
    return narrow.with_columns(
        fold=pl.lit(None, dtype=pl.Int32),
        regime=pl.lit(None, dtype=pl.Utf8),
        robust_score=((pl.col("_sharpe") / 2.0) + (pl.col("_cagr") / 60.0) - (pl.col("_maxdd").abs() / 50.0)).clip(-2, 2),
        source_artifact=pl.lit(source_artifact),
        run_id=pl.lit(run_id),
        demo_mode=pl.lit(False),
    ).drop(["_sharpe", "_cagr", "_maxdd"])


def build_fact_robustness_surface(sources: dict[str, pl.DataFrame], run_id: str) -> pl.DataFrame:
    frames = [
        _surface_from_summary(sources.get("extended_summary", pl.DataFrame()), "extended_multiplier_summary.csv", run_id),
        _surface_from_summary(sources.get("universe_summary", pl.DataFrame()), "universe_robustness_summary.csv", run_id),
    ]
    frames = [frame for frame in frames if not frame.is_empty()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
