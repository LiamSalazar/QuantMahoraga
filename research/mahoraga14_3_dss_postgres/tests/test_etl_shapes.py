from __future__ import annotations

import json

import polars as pl

from etl.build_fact_candidate_metric import build_fact_candidate_metric
from etl.lfs_guard import assert_not_lfs_pointer
from etl.load_postgres import LOAD_ORDER, _parquet_for
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


def test_load_order_includes_minimum_oltp_tables() -> None:
    assert LOAD_ORDER[:4] == [
        "oltp.research_run",
        "oltp.data_snapshot",
        "oltp.artifact_inventory",
        "oltp.candidate_grid",
    ]


def test_parquet_for_resolves_oltp_dimensions_and_facts() -> None:
    paths = get_paths()
    assert _parquet_for(paths, "oltp.research_run") == paths.parquet_root / "oltp" / "research_run.parquet"
    assert _parquet_for(paths, "dw.dim_candidate") == paths.parquet_root / "dimensions" / "dim_candidate.parquet"
    assert _parquet_for(paths, "dw.fact_outcome") == paths.parquet_root / "facts" / "fact_outcome.parquet"


def test_fact_candidate_metric_preserves_null_sweep_role_when_not_applicable() -> None:
    universe_summary = pl.DataFrame(
        [
            {
                "candidate_id": "B1.05_C1.10_L1.10_R1.05",
                "universe_id": "negative_control_nontech",
                "sweep_role": None,
                "CAGR": None,
                "Sharpe": None,
                "Sortino": None,
                "MaxDD": None,
                "AlphaNW_QQQ": None,
                "AlphaNW_SPY": None,
                "AvgExposure": None,
                "AvgTurnover": None,
                "ReturnPerExposure": None,
                "robust_region_flag": None,
            }
        ]
    )
    fact = build_fact_candidate_metric({"universe_summary": universe_summary}, "test_run")
    assert fact.height == 1
    assert fact.item(0, "metric_set") == "universe_robustness"
    assert fact.item(0, "sweep_role") is None


def test_lfs_guard_reports_actionable_message(tmp_path) -> None:
    pointer = tmp_path / "data.csv"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
    try:
        assert_not_lfs_pointer(pointer)
    except RuntimeError as exc:
        message = str(exc)
        assert "git lfs install && git lfs pull" in message
        assert str(pointer) in message
    else:
        raise AssertionError("Expected Git LFS pointer detection to fail")
