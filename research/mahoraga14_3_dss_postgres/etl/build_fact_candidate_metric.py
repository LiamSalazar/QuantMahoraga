from __future__ import annotations

import polars as pl


def _summary_to_metric(df: pl.DataFrame, metric_set: str, run_id: str) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame()
    candidate_col = "candidate_id" if "candidate_id" in df.columns else "CandidateId"
    keep = {
        "candidate_id": pl.col(candidate_col).cast(pl.Utf8),
        "universe_id": pl.col("universe_id").cast(pl.Utf8),
        "sweep_role": pl.col("sweep_role").cast(pl.Utf8),
        "metric_set": pl.lit(metric_set),
        "cagr": pl.col("CAGR").cast(pl.Float64, strict=False),
        "sharpe": pl.col("Sharpe").cast(pl.Float64, strict=False),
        "sortino": pl.col("Sortino").cast(pl.Float64, strict=False),
        "maxdd": pl.col("MaxDD").cast(pl.Float64, strict=False),
        "alpha_qqq": pl.col("AlphaNW_QQQ").cast(pl.Float64, strict=False),
        "alpha_spy": pl.col("AlphaNW_SPY").cast(pl.Float64, strict=False),
        "beta_qqq": pl.col("BetaQQQ").cast(pl.Float64, strict=False) if "BetaQQQ" in df.columns else pl.lit(None, dtype=pl.Float64),
        "beta_spy": pl.col("BetaSPY").cast(pl.Float64, strict=False) if "BetaSPY" in df.columns else pl.lit(None, dtype=pl.Float64),
        "avg_exposure": pl.col("AvgExposure").cast(pl.Float64, strict=False),
        "avg_turnover": pl.col("AvgTurnover").cast(pl.Float64, strict=False),
        "return_per_exposure": pl.col("ReturnPerExposure").cast(pl.Float64, strict=False),
        "robust_region_flag": pl.col("robust_region_flag").cast(pl.Boolean, strict=False),
        "run_id": pl.lit(run_id),
        "demo_mode": pl.lit(False),
    }
    return df.select(**keep)


def build_fact_candidate_metric(sources: dict[str, pl.DataFrame], run_id: str) -> pl.DataFrame:
    frames = [
        _summary_to_metric(sources.get("extended_summary", pl.DataFrame()), "extended_multiplier", run_id),
        _summary_to_metric(sources.get("universe_summary", pl.DataFrame()), "universe_robustness", run_id),
    ]
    frames = [frame for frame in frames if not frame.is_empty()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
