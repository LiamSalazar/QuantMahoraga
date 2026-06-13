from __future__ import annotations

import polars as pl

from .date_utils import parse_date


def build_fact_module_trace(sources: dict[str, pl.DataFrame], run_id: str) -> pl.DataFrame:
    trace = sources.get("module_trace", pl.DataFrame())
    if trace.is_empty():
        return pl.DataFrame()
    return trace.with_columns(
        date_value=parse_date("date"),
        module_active=pl.col("threshold_crossed").cast(pl.Boolean, strict=False),
        raw_value=pl.col("signal_strength").cast(pl.Float64, strict=False),
        intensity_score=pl.col("signal_strength").cast(pl.Float64, strict=False),
        probability=pl.when((pl.col("signal_strength") >= 0) & (pl.col("signal_strength") <= 1)).then(pl.col("signal_strength")).otherwise(None),
        state_label=pl.col("branch_taken").cast(pl.Utf8),
        effect_on_budget=pl.lit(None, dtype=pl.Float64),
        effect_on_exposure=pl.lit(None, dtype=pl.Float64),
        effect_on_blend=pl.lit(None, dtype=pl.Float64),
        input_summary=pl.col("main_inputs_summary_json").fill_null("{}"),
        output_summary=pl.col("main_outputs_summary_json").fill_null("{}"),
        run_id=pl.lit(run_id),
        demo_mode=pl.lit(False),
    ).select(
        "date_value",
        "candidate_id",
        "fold",
        "universe_id",
        "module_name",
        "module_active",
        "raw_value",
        "intensity_score",
        "probability",
        "state_label",
        "effect_on_budget",
        "effect_on_exposure",
        "effect_on_blend",
        "input_summary",
        "output_summary",
        "run_id",
        "demo_mode",
    )
