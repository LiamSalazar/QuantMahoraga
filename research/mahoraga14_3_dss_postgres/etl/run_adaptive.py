from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import polars as pl

from .adaptive_planner import build_execution_plan
from .config import make_config
from .control_plane import (
    ensure_control_plane,
    finish_pipeline_run,
    log_cache_invalidations,
    publish_run,
    stage_timer,
    start_pipeline_run,
    write_control_json,
)
from .data_contracts import persist_contract_results, validate_all_contracts, write_contract_report
from .incremental import PARTITION_SPECS, analyze_affected_tables, load_incremental_partitions
from .mart_dependencies import cache_for_marts
from .partitioned_parquet import PARTITION_COLUMNS, write_partition_manifest, write_partitioned_table
from .paths import ensure_output_dirs
from .pending_outcomes import detect_pending_outcomes, persist_pending_outcomes
from .refresh_views import refresh
from .run_pipeline import build_all, run as run_standard_pipeline
from .source_manifest import diff_manifests, load_previous_manifest, persist_manifest, scan_sources, write_manifest_report


def _incremental_safe(tables: list[str]) -> tuple[bool, str]:
    unsupported = sorted(table for table in tables if table not in PARTITION_SPECS)
    if unsupported:
        return False, f"Unsupported incremental tables: {', '.join(unsupported)}."
    return True, "All affected facts have supported logical partition specs."


def _write_partitioned_outputs(tables: dict[str, pl.DataFrame], run_id: str, database_url: str | None) -> None:
    entries = []
    root = ensure_output_dirs().outputs_root / "parquet_partitioned"
    for table_name, df in tables.items():
        if table_name in PARTITION_COLUMNS:
            entries.extend(write_partitioned_table(df, table_name, root=root))
    write_partition_manifest(run_id, entries, database_url=database_url)


def run_adaptive(args: argparse.Namespace) -> dict:
    paths = ensure_output_dirs()
    config = make_config(
        profile=args.profile,
        mode=args.mode,
        database_url=args.database_url,
        include_demo_grid=not args.no_demo_grid,
    )
    database_url = config.database_url if config.mode == "postgres" else None
    previous_manifest = load_previous_manifest(None if args.dry_run else database_url)
    current_manifest = scan_sources(config=config)
    source_diff = diff_manifests(current_manifest, previous_manifest)
    plan = build_execution_plan(
        config=config,
        source_diff=source_diff,
        previous_manifest_available=bool(previous_manifest),
        requested_strategy=args.strategy,
        parallelism=args.parallelism,
        dry_run=args.dry_run,
    )
    write_manifest_report(current_manifest, source_diff, config.run_id, paths)
    write_control_json(paths, f"execution_plan_{config.run_id}.json", plan.to_dict())
    if args.dry_run:
        payload = {"run_id": config.run_id, "plan": plan.to_dict()}
        print(json.dumps(payload, indent=2))
        return payload

    if not database_url:
        raise RuntimeError("DATABASE_URL is required for adaptive Postgres execution")
    ensure_control_plane(database_url, paths)
    persist_manifest(database_url, current_manifest, config.run_id)
    start_pipeline_run(
        database_url,
        run_id=config.run_id,
        strategy=plan.strategy,
        profile=config.profile,
        mode=config.mode,
        changed_sources_count=source_diff.changed_sources_count,
        changed_partitions_count=sum(len(parts) for parts in plan.affected_partitions.values()),
    )
    try:
        if plan.strategy in {"full_refresh", "backfill"}:
            summary = run_standard_pipeline(config, skip_postgres=False, truncate=not args.no_truncate)
            summary["adaptive_plan"] = plan.to_dict()
            return summary

        if plan.strategy in {"no_data_refresh", "querylog_only", "mart_only"}:
            with stage_timer(database_url, config.run_id, "mart_refresh_dependency"):
                refreshed = refresh(database_url, strategy="dependency", changed_tables=plan.affected_tables or ["query_logs"], run_id=config.run_id)
            endpoints = cache_for_marts(refreshed)
            log_cache_invalidations(database_url, config.run_id, endpoints, f"{plan.strategy}:{plan.reason}")
            publish_run(database_url, config.run_id)
            finish_pipeline_run(database_url, run_id=config.run_id, status="COMPLETED", validation_status="PASS", published=True)
            return {"run_id": config.run_id, "plan": plan.to_dict(), "refreshed_marts": refreshed}

        safe, reason = _incremental_safe(plan.affected_tables)
        if plan.strategy in {"incremental_partition_refresh", "whatif_only"} and not safe:
            fallback_plan = {**plan.to_dict(), "strategy": "full_refresh", "fallback_reason": reason}
            write_control_json(paths, f"execution_plan_{config.run_id}_fallback.json", fallback_plan)
            summary = run_standard_pipeline(config, skip_postgres=False, truncate=not args.no_truncate)
            summary["adaptive_plan"] = fallback_plan
            return summary

        with stage_timer(database_url, config.run_id, "build_incremental_staging") as metrics:
            tables, row_counts, summary = build_all(config, paths, parallelism=plan.parallelism)
            _write_partitioned_outputs({name: tables[name] for name in plan.affected_tables if name in tables}, config.run_id, database_url)
            metrics["rows_written"] = sum(row_counts.get(name, 0) for name in plan.affected_tables)

        contract_results = validate_all_contracts({name: tables[name] for name in plan.affected_tables if name in tables})
        write_contract_report(contract_results, config.run_id, paths)
        persist_contract_results(database_url, config.run_id, contract_results)
        if any(not result.passed for result in contract_results):
            finish_pipeline_run(database_url, run_id=config.run_id, status="FAILED_VALIDATION", validation_status="FAIL", published=False)
            raise RuntimeError("Data contract validation failed; incremental publish was blocked.")

        with stage_timer(database_url, config.run_id, "incremental_partition_load") as metrics:
            load_counts = load_incremental_partitions(database_url, plan, paths)
            metrics["rows_written"] = sum(load_counts.values())
        analyze_affected_tables(database_url, plan)
        pending = detect_pending_outcomes(tables.get("fact_decision_state", pl.DataFrame()), tables.get("fact_outcome", pl.DataFrame()), config.run_id)
        persist_pending_outcomes(database_url, pending)

        with stage_timer(database_url, config.run_id, "mart_refresh_dependency"):
            refreshed = refresh(database_url, strategy="dependency", changed_tables=plan.affected_tables, run_id=config.run_id)
        log_cache_invalidations(database_url, config.run_id, plan.cache_to_invalidate, plan.reason)
        publish_run(database_url, config.run_id)
        finish_pipeline_run(
            database_url,
            run_id=config.run_id,
            status="COMPLETED",
            total_rows_processed=int(summary.get("total_rows_written") or 0),
            total_rows_loaded=sum(load_counts.values()),
            validation_status="PASS",
            published=True,
        )
        return {
            "run_id": config.run_id,
            "plan": plan.to_dict(),
            "incremental_load_counts": load_counts,
            "refreshed_marts": refreshed,
            "cache_invalidated": plan.cache_to_invalidate,
        }
    except Exception as exc:
        finish_pipeline_run(database_url, run_id=config.run_id, status="FAILED", error_message=str(exc), published=False)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Mahoraga DSS Postgres pipeline runner.")
    parser.add_argument("--strategy", choices=["auto", "full", "incremental", "backfill", "dry-run"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", choices=["postgres", "parquet", "demo"], default="postgres")
    parser.add_argument("--profile", choices=["small", "standard", "competition"], default="standard")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--parallelism", type=int, default=None)
    parser.add_argument("--no-truncate", action="store_true", help="Do not truncate on full-refresh fallback.")
    parser.add_argument("--no-demo-grid", action="store_true", help="Disable explicitly flagged synthetic what-if rows.")
    args = parser.parse_args()
    result = run_adaptive(args)
    if not args.dry_run:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
