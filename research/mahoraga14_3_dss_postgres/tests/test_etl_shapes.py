from __future__ import annotations

import json

import polars as pl

from etl.paths import get_paths
from etl.validate_outputs import validate


def test_parquet_outputs_validate() -> None:
    report = validate()
    assert report["passed"], report


def test_core_fact_shapes() -> None:
    paths = get_paths()
    position = pl.read_parquet(paths.parquet_root / "facts" / "fact_position_daily.parquet")
    outcome = pl.read_parquet(paths.parquet_root / "facts" / "fact_outcome.parquet")
    whatif = pl.read_parquet(paths.parquet_root / "facts" / "fact_whatif.parquet")
    assert {"date_value", "candidate_id", "fold", "universe_id", "ticker", "final_weight"}.issubset(position.columns)
    assert {"decision_date", "horizon", "alpha_vs_qqq", "helped_flag"}.issubset(outcome.columns)
    assert {"scenario_id", "budget_multiplier", "cost_bps", "demo_mode"}.issubset(whatif.columns)
    assert position.height > 100_000
    assert outcome.get_column("horizon").n_unique() >= 3
    assert whatif.filter(pl.col("source_artifact") == "demo_synthetic_whatif_grid").filter(~pl.col("demo_mode")).height == 0


def test_pipeline_summary_reports_demo_rows() -> None:
    paths = get_paths()
    summary = json.loads((paths.reports_root / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary["baseline_modified"] is False
    assert summary["real_rows_written_estimate"] > summary["demo_rows_written"]
    assert summary["validation_passed"] is True
