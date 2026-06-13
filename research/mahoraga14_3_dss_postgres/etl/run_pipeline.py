from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import polars as pl

from .build_dimensions import build_dimensions
from .build_fact_candidate_metric import build_fact_candidate_metric
from .build_fact_cost_sensitivity import build_fact_cost_sensitivity
from .build_fact_decision_state import build_fact_decision_state
from .build_fact_market_bar import build_fact_market_bar
from .build_fact_module_trace import build_fact_module_trace
from .build_fact_outcome import build_fact_outcome
from .build_fact_path_recursive import build_fact_path_recursive
from .build_fact_position_daily import build_fact_position_daily
from .build_fact_robustness_surface import build_fact_robustness_surface
from .build_fact_signal_daily import build_fact_signal_daily
from .build_fact_universe_sensitivity import build_fact_universe_sensitivity
from .build_fact_whatif import build_fact_whatif
from .config import BASELINE_REFERENCE, OFFICIAL_CANDIDATE_ID, PHASE, RuntimeConfig, make_config
from .discover_artifacts import discover
from .extract_existing_cubes import load_sources
from .load_postgres import load_all
from .paths import DssPaths, ensure_output_dirs, get_paths
from .refresh_views import refresh
from .validate_outputs import validate
from .write_parquet import write_tables


def _build_fact_data_quality(validation_report: dict, run_id: str) -> pl.DataFrame:
    rows = []
    for check in validation_report.get("checks", []):
        rows.append(
            {
                "run_id": run_id,
                "table_name": check.get("table_name", "unknown"),
                "check_name": check.get("check_name"),
                "passed": bool(check.get("passed")),
                "observed_value": str(check.get("observed_value")),
                "expected_value": str(check.get("expected_value")),
                "severity": check.get("severity", "info"),
                "detail": json.dumps({key: value for key, value in check.items() if key not in {"table_name", "check_name", "passed", "observed_value", "expected_value", "severity"}}),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _build_oltp_tables(inventory: pl.DataFrame, candidate_metric: pl.DataFrame, config: RuntimeConfig) -> dict[str, pl.DataFrame]:
    started_at = datetime.now(timezone.utc).isoformat()
    research_run = pl.DataFrame(
        [
            {
                "run_id": config.run_id,
                "phase": PHASE,
                "baseline_reference": BASELINE_REFERENCE,
                "official_candidate_id": OFFICIAL_CANDIDATE_ID,
                "profile": config.profile,
                "source_mode": config.mode,
                "demo_mode": config.demo_mode,
                "started_at": started_at,
                "completed_at": started_at,
                "status": "COMPLETED_PARQUET" if config.mode != "postgres" else "COMPLETED_POSTGRES_LOAD_PENDING",
                "notes": "Reads frozen baseline and extended-analysis artifacts; does not recalibrate the model.",
            }
        ]
    )
    snapshot = pl.DataFrame(
        [
            {
                "run_id": config.run_id,
                "snapshot_name": "baseline_and_extended_outputs",
                "source_root": "baseline/mahoraga14_3_baseline + research/mahoraga14_3_extended_analysis",
                "artifact_count": inventory.height,
                "row_count": int(inventory.get_column("row_count").drop_nulls().sum()) if "row_count" in inventory.columns else 0,
                "content_hash": None,
                "created_at": started_at,
            }
        ]
    )
    candidate_grid = pl.DataFrame()
    if not candidate_metric.is_empty():
        candidate_grid = candidate_metric.select(
            "run_id",
            "candidate_id",
            "universe_id",
            "sweep_role",
            pl.lit(None, dtype=pl.Float64).alias("budget_multiplier"),
            pl.lit(None, dtype=pl.Float64).alias("conviction_multiplier"),
            pl.lit(None, dtype=pl.Float64).alias("leader_multiplier"),
            pl.lit(None, dtype=pl.Float64).alias("backoff_strength"),
            pl.col("metric_set").alias("source_artifact"),
            "demo_mode",
        ).unique(subset=["run_id", "candidate_id", "universe_id", "sweep_role"])
    return {
        "research_run": research_run,
        "data_snapshot": snapshot,
        "artifact_inventory": inventory,
        "candidate_grid": candidate_grid,
    }


def build_all(config: RuntimeConfig, paths: DssPaths | None = None) -> tuple[dict[str, pl.DataFrame], dict[str, int], dict]:
    paths = ensure_output_dirs(paths or get_paths())
    sources = load_sources(paths)
    inventory = discover(paths, run_id=config.run_id)

    fact_whatif = build_fact_whatif(sources, config)
    dimensions = build_dimensions(sources, fact_whatif)
    facts = {
        "fact_market_bar": build_fact_market_bar(sources, config.run_id),
        "fact_signal_daily": build_fact_signal_daily(sources, config.run_id),
        "fact_decision_state": build_fact_decision_state(sources, config.run_id),
        "fact_position_daily": build_fact_position_daily(sources, config.run_id),
        "fact_module_trace": build_fact_module_trace(sources, config.run_id),
        "fact_outcome": build_fact_outcome(sources, config.run_id),
        "fact_candidate_metric": build_fact_candidate_metric(sources, config.run_id),
        "fact_robustness_surface": build_fact_robustness_surface(sources, config.run_id),
        "fact_cost_sensitivity": build_fact_cost_sensitivity(sources, config.run_id),
        "fact_universe_sensitivity": build_fact_universe_sensitivity(sources, config.run_id),
        "fact_whatif": fact_whatif,
        "fact_path_recursive": build_fact_path_recursive(sources, config.run_id),
    }
    oltp = _build_oltp_tables(inventory, facts["fact_candidate_metric"], config)
    tables = {**oltp, **dimensions, **facts}
    row_counts = write_tables(tables, paths, config.run_id)
    validation_report = validate(paths)
    quality = _build_fact_data_quality(validation_report, config.run_id)
    if not quality.is_empty():
        tables["fact_data_quality"] = quality
        row_counts.update(write_tables({"fact_data_quality": quality}, paths, config.run_id))
    real_rows = sum(count for name, count in row_counts.items() if name != "fact_whatif")
    whatif = facts["fact_whatif"]
    demo_rows = int(whatif.filter(pl.col("demo_mode")).height) if not whatif.is_empty() and "demo_mode" in whatif.columns else 0
    real_whatif = int(whatif.filter(~pl.col("demo_mode")).height) if not whatif.is_empty() and "demo_mode" in whatif.columns else 0
    summary = {
        "run_id": config.run_id,
        "profile": config.profile,
        "mode": config.mode,
        "baseline_modified": False,
        "total_rows_written": sum(row_counts.values()),
        "real_rows_written_estimate": real_rows + real_whatif,
        "demo_rows_written": demo_rows,
        "expected_real_min_rows_for_profile": config.row_target["expected_real_min_rows"],
        "real_row_target_met": (real_rows + real_whatif) >= config.row_target["expected_real_min_rows"],
        "row_counts": row_counts,
        "validation_passed": validation_report["passed"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (paths.reports_root / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return tables, row_counts, summary


def run(config: RuntimeConfig, skip_postgres: bool = False, truncate: bool = False) -> dict:
    paths = ensure_output_dirs()
    _, _, summary = build_all(config, paths)
    if config.mode == "postgres" and not skip_postgres:
        load_counts = load_all(config, paths, bootstrap=True, truncate=truncate)
        refresh(config.database_url)
        summary["postgres_load_counts"] = load_counts
        (paths.reports_root / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Mahoraga Quant DSS parquet/Postgres artifacts.")
    parser.add_argument("--mode", choices=["parquet", "postgres", "demo"], default="parquet")
    parser.add_argument("--profile", choices=["small", "standard", "competition"], default="small")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--skip-postgres", action="store_true")
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--no-demo-grid", action="store_true", help="Disable explicitly flagged synthetic what-if rows.")
    args = parser.parse_args()
    config = make_config(
        profile=args.profile,
        mode=args.mode,
        database_url=args.database_url,
        include_demo_grid=not args.no_demo_grid,
    )
    summary = run(config, skip_postgres=args.skip_postgres, truncate=args.truncate)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
