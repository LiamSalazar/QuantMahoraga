from __future__ import annotations

import polars as pl

from .date_utils import parse_date


def build_fact_decision_state(sources: dict[str, pl.DataFrame], run_id: str) -> pl.DataFrame:
    decision = sources.get("decision", pl.DataFrame())
    if decision.is_empty():
        return pl.DataFrame()
    market = sources.get("market_context", pl.DataFrame())
    active = sources.get("active_return", pl.DataFrame())
    out = decision.with_columns(
        date_value=parse_date("date"),
        regime=pl.col("participation_state").cast(pl.Utf8),
        continuation_trigger_flag=(pl.col("continuation_trigger_p").cast(pl.Float64, strict=False) >= 0.5),
        continuation_pressure_flag=(pl.col("continuation_pressure_p").cast(pl.Float64, strict=False) >= 0.5),
        structural_flag=(pl.col("structural_p").cast(pl.Float64, strict=False) >= 0.5),
        backoff_flag=(pl.col("backoff_strength_applied").cast(pl.Float64, strict=False) > 0),
        hard_backoff_flag=pl.col("hard_backoff_flag").cast(pl.Boolean, strict=False),
        gross_exposure=pl.col("expected_exposure").cast(pl.Float64, strict=False),
        turnover=pl.col("expected_turnover").cast(pl.Float64, strict=False),
        run_id=pl.lit(run_id),
        demo_mode=pl.lit(False),
    )
    if not market.is_empty() and {"date", "qqq_return", "qqq_drawdown"}.issubset(market.columns):
        market_small = market.select(
            parse_date("date").alias("date_value"),
            pl.col("qqq_return").cast(pl.Float64, strict=False).alias("benchmark_return"),
            pl.col("qqq_drawdown").cast(pl.Float64, strict=False).alias("drawdown"),
        ).unique(subset=["date_value"], keep="first")
        out = out.join(market_small, on="date_value", how="left")
    if not active.is_empty() and {"Date", "OfficialReturn"}.issubset(active.columns):
        active_small = active.select(
            parse_date("Date").alias("date_value"),
            pl.col("OfficialReturn").cast(pl.Float64, strict=False).alias("net_return"),
        ).unique(subset=["date_value"], keep="first")
        out = out.join(active_small, on="date_value", how="left")
    for column in ["benchmark_return", "drawdown", "net_return"]:
        if column not in out.columns:
            out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    keep = [
        "date_value",
        "candidate_id",
        "fold",
        "universe_id",
        "regime",
        "participation_state",
        "continuation_trigger_flag",
        "continuation_pressure_flag",
        "structural_flag",
        "backoff_flag",
        "hard_backoff_flag",
        "leader_blend",
        "gross_exposure",
        "net_return",
        "benchmark_return",
        "turnover",
        "drawdown",
        "long_budget",
        "gate_scale",
        "vol_mult",
        "exp_cap",
        "expected_exposure",
        "expected_turnover",
        "run_id",
        "demo_mode",
    ]
    return out.select([column for column in keep if column in out.columns])
