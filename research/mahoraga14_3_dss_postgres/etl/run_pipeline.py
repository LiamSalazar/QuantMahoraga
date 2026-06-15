from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

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
from .control_plane import ensure_control_plane, finish_pipeline_run, publish_run, stage_timer, start_pipeline_run
from .data_contracts import persist_contract_results, validate_all_contracts, write_contract_report
from .discover_artifacts import discover
from .extract_existing_cubes import load_sources
from .load_postgres import load_all
from .pending_outcomes import detect_pending_outcomes, persist_pending_outcomes
from .paths import DssPaths, ensure_output_dirs, get_paths
from .refresh_views import refresh
from .source_manifest import diff_manifests, load_previous_manifest, persist_manifest, scan_sources, write_manifest_report
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


def _effective_parallelism(requested: int | None = None) -> int:
    if requested and requested > 0:
        return requested
    raw = os.getenv("DSS_PARALLELISM", "1").strip().lower()
    if raw == "auto":
        return max(1, min(8, (os.cpu_count() or 2) // 2 or 1))
    if raw.isdigit():
        return max(1, int(raw))
    return 1


def _build_facts(sources: dict[str, pl.DataFrame], config: RuntimeConfig, fact_whatif: pl.DataFrame, parallelism: int) -> dict[str, pl.DataFrame]:
    builders = _fact_builders(sources, config, fact_whatif)
    if parallelism <= 1:
        facts = {name: build() for name, build in builders.items()}
    else:
        facts = {}
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            future_to_name = {executor.submit(build): name for name, build in builders.items()}
            for future in as_completed(future_to_name):
                facts[future_to_name[future]] = future.result()
        facts = {name: facts[name] for name in builders}
    facts["fact_whatif"] = fact_whatif
    return facts


def _fact_builders(sources: dict[str, pl.DataFrame], config: RuntimeConfig, fact_whatif: pl.DataFrame) -> dict[str, Any]:
    return {
        "fact_market_bar": lambda: build_fact_market_bar(sources, config.run_id),
        "fact_signal_daily": lambda: build_fact_signal_daily(sources, config.run_id),
        "fact_decision_state": lambda: build_fact_decision_state(sources, config.run_id),
        "fact_position_daily": lambda: build_fact_position_daily(sources, config.run_id),
        "fact_module_trace": lambda: build_fact_module_trace(sources, config.run_id),
        "fact_outcome": lambda: build_fact_outcome(sources, config.run_id),
        "fact_candidate_metric": lambda: build_fact_candidate_metric(sources, config.run_id),
        "fact_robustness_surface": lambda: build_fact_robustness_surface(sources, config.run_id),
        "fact_cost_sensitivity": lambda: build_fact_cost_sensitivity(sources, config.run_id),
        "fact_universe_sensitivity": lambda: build_fact_universe_sensitivity(sources, config.run_id),
        "fact_path_recursive": lambda: build_fact_path_recursive(sources, config.run_id),
        "fact_whatif": lambda: fact_whatif,
    }


def build_all(config: RuntimeConfig, paths: DssPaths | None = None, parallelism: int | None = None) -> tuple[dict[str, pl.DataFrame], dict[str, int], dict]:
    paths = ensure_output_dirs(paths or get_paths())
    sources = load_sources(paths)
    inventory = discover(paths, run_id=config.run_id)

    fact_whatif = build_fact_whatif(sources, config)
    dimensions = build_dimensions(sources, fact_whatif)
    facts = _build_facts(sources, config, fact_whatif, _effective_parallelism(parallelism))
    oltp = _build_oltp_tables(inventory, facts["fact_candidate_metric"], config)
    tables = {**oltp, **dimensions, **facts}
    row_counts = write_tables(tables, paths, config.run_id)
    validation_report = validate(paths)
    contract_results = validate_all_contracts(tables)
    write_contract_report(contract_results, config.run_id, paths)
    quality = _build_fact_data_quality(validation_report, config.run_id)
    if not quality.is_empty():
        tables["fact_data_quality"] = quality
        row_counts.update(write_tables({"fact_data_quality": quality}, paths, config.run_id))
    pending_outcomes = detect_pending_outcomes(facts["fact_decision_state"], facts["fact_outcome"], config.run_id)
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
        "data_contracts_passed": all(result.passed for result in contract_results),
        "pending_outcomes": {
            "rows": pending_outcomes.height,
            "ready": int(pending_outcomes.filter(pl.col("status") == "ready").height) if not pending_outcomes.is_empty() else 0,
            "pending": int(pending_outcomes.filter(pl.col("status") == "pending").height) if not pending_outcomes.is_empty() else 0,
            "computed": int(pending_outcomes.filter(pl.col("status") == "computed").height) if not pending_outcomes.is_empty() else 0,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (paths.reports_root / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return tables, row_counts, summary


def build_selected(
    config: RuntimeConfig,
    paths: DssPaths | None,
    selected_tables: list[str],
    selected_partitions: dict[str, list[str]] | None = None,
    parallelism: int | None = None,
) -> tuple[dict[str, pl.DataFrame], dict[str, int], dict, str | None]:
    paths = ensure_output_dirs(paths or get_paths())
    selected = [table for table in selected_tables if table != "all"]
    if not selected:
        return {}, {}, {"run_id": config.run_id, "total_rows_written": 0, "row_counts": {}}, None

    sources = load_sources(paths)
    fact_whatif = build_fact_whatif(sources, config) if "fact_whatif" in selected or any(table.startswith("dim_") for table in selected) else pl.DataFrame()
    builders = _fact_builders(sources, config, fact_whatif)
    unsupported = [table for table in selected if table not in builders]
    if unsupported:
        return {}, {}, {}, f"Selected build has no safe builder for: {', '.join(sorted(unsupported))}"

    max_workers = _effective_parallelism(parallelism)
    if max_workers <= 1 or len(selected) <= 1:
        tables = {table: builders[table]() for table in selected}
    else:
        tables = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_table = {executor.submit(builders[table]): table for table in selected}
            for future in as_completed(future_to_table):
                tables[future_to_table[future]] = future.result()
        tables = {table: tables[table] for table in selected}

    row_counts = write_tables(tables, paths, config.run_id)
    contract_results = validate_all_contracts(tables)
    write_contract_report(contract_results, config.run_id, paths)
    summary = {
        "run_id": config.run_id,
        "profile": config.profile,
        "mode": config.mode,
        "selected_tables": selected,
        "selected_partitions": selected_partitions or {},
        "total_rows_written": sum(row_counts.values()),
        "row_counts": row_counts,
        "data_contracts_passed": all(result.passed for result in contract_results),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return tables, row_counts, summary, None


def run(config: RuntimeConfig, skip_postgres: bool = False, truncate: bool = False) -> dict:
    paths = ensure_output_dirs()
    database_url = config.database_url if config.mode == "postgres" and not skip_postgres else None
    if database_url:
        ensure_control_plane(database_url, paths)
    previous_manifest = load_previous_manifest(database_url)
    current_manifest = scan_sources(config=config)
    source_diff = diff_manifests(current_manifest, previous_manifest)
    write_manifest_report(current_manifest, source_diff, config.run_id, paths)
    if database_url:
        persist_manifest(database_url, current_manifest, config.run_id)
        start_pipeline_run(
            database_url,
            run_id=config.run_id,
            strategy="full_refresh",
            profile=config.profile,
            mode=config.mode,
            changed_sources_count=source_diff.changed_sources_count,
            changed_partitions_count=0,
        )
    try:
        with stage_timer(database_url, config.run_id, "build_parquet") as metrics:
            tables, row_counts, summary = build_all(config, paths)
            metrics["rows_written"] = sum(row_counts.values())
        if database_url:
            contract_results = validate_all_contracts(tables)
            persist_contract_results(database_url, config.run_id, contract_results)
            if any(not result.passed for result in contract_results):
                finish_pipeline_run(database_url, run_id=config.run_id, status="FAILED_VALIDATION", validation_status="FAIL", published=False)
                raise RuntimeError("Data contract validation failed; publish was blocked.")
            pending = detect_pending_outcomes(tables.get("fact_decision_state", pl.DataFrame()), tables.get("fact_outcome", pl.DataFrame()), config.run_id)
            persist_pending_outcomes(database_url, pending)
        if config.mode == "postgres" and not skip_postgres:
            with stage_timer(database_url, config.run_id, "postgres_load") as metrics:
                load_counts = load_all(config, paths, bootstrap=True, truncate=truncate)
                metrics["rows_written"] = sum(load_counts.values())
            with stage_timer(database_url, config.run_id, "mart_refresh"):
                refresh(config.database_url, run_id=config.run_id)
            summary["postgres_load_counts"] = load_counts
            publish_run(database_url, config.run_id)
            summary["published"] = True
            (paths.reports_root / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if database_url:
            finish_pipeline_run(
                database_url,
                run_id=config.run_id,
                status="COMPLETED",
                total_rows_processed=int(summary.get("total_rows_written") or 0),
                total_rows_loaded=sum(summary.get("postgres_load_counts", {}).values()) if summary.get("postgres_load_counts") else 0,
                validation_status="PASS" if summary.get("validation_passed") and summary.get("data_contracts_passed", True) else "FAIL",
                published=bool(summary.get("published")),
            )
        return summary
    except Exception as exc:
        if database_url:
            finish_pipeline_run(database_url, run_id=config.run_id, status="FAILED", error_message=str(exc), published=False)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Mahoraga Quant DSS parquet/Postgres artifacts.")
    parser.add_argument("--mode", choices=["parquet", "postgres", "demo"], default="parquet")
    parser.add_argument("--profile", choices=["small", "standard", "competition"], default="small")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--skip-postgres", action="store_true")
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--parallelism", type=int, default=None)
    parser.add_argument("--no-demo-grid", action="store_true", help="Disable explicitly flagged synthetic what-if rows.")
    args = parser.parse_args()
    config = make_config(
        profile=args.profile,
        mode=args.mode,
        database_url=args.database_url,
        include_demo_grid=not args.no_demo_grid,
    )
    if args.parallelism:
        os.environ["DSS_PARALLELISM"] = str(args.parallelism)
    summary = run(config, skip_postgres=args.skip_postgres, truncate=args.truncate)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
