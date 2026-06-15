from __future__ import annotations

import os
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from .mart_dependencies import ALL_MARTS, cache_for_marts, marts_for_tables
from .partition_rules import SOURCE_PARTITION_COLUMNS, partition_label
from .paths import get_paths
from .source_manifest import SourceDiff

SOURCE_TABLE_HINTS = {
    "position_cube": ["fact_position_daily"],
    "module_trace_cube": ["fact_module_trace"],
    "outcome_cube": ["fact_outcome"],
    "decision_date_cube": ["fact_decision_state", "fact_path_recursive"],
    "market_context_cube": ["fact_market_bar"],
    "extended_multiplier_summary": ["fact_candidate_metric", "fact_whatif"],
    "extended_multiplier_fold_summary": ["fact_candidate_metric"],
    "universe_robustness_summary": ["fact_universe_sensitivity", "fact_candidate_metric"],
    "active_return_vs_qqq_official": ["fact_path_recursive"],
    "fold_summary_official": ["fact_candidate_metric"],
    "cost_sensitivity_official": ["fact_cost_sensitivity"],
    "slippage_sensitivity_official": ["fact_cost_sensitivity"],
}


@dataclass(frozen=True)
class ExecutionPlan:
    strategy: str
    reason: str
    estimated_rows: int
    estimation_source: str
    scale_class: str
    parallelism: int
    changed_sources: list[str] = field(default_factory=list)
    affected_tables: list[str] = field(default_factory=list)
    affected_partitions: dict[str, list[str]] = field(default_factory=dict)
    facts_to_build: list[str] = field(default_factory=list)
    dimensions_to_build: list[str] = field(default_factory=list)
    marts_to_refresh: list[str] = field(default_factory=list)
    cache_to_invalidate: list[str] = field(default_factory=list)
    validation_level: str = "standard"
    publish_mode: str = "direct"
    fallback_allowed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scale_class(rows: int) -> str:
    if rows < 1_000_000:
        return "small"
    if rows < 10_000_000:
        return "medium"
    return "large"


def _pipeline_summary_rows() -> tuple[int | None, str | None]:
    path = get_paths().reports_root / "pipeline_summary.json"
    if not path.exists():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    for key in ["total_rows_written", "real_rows_written_estimate"]:
        value = payload.get(key)
        if isinstance(value, int) and value > 0:
            return value, f"latest_pipeline_summary.{key}"
    return None, None


def _postgres_rows(database_url: str | None) -> tuple[int | None, str | None, dict[str, int]]:
    if not database_url:
        return None, None, {}
    try:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(database_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT schemaname || '.' || relname AS table_name, n_live_tup::bigint AS row_count
                    FROM pg_stat_user_tables
                    WHERE schemaname IN ('dw', 'mart', 'oltp')
                    """
                )
                rows = {row["table_name"]: int(row["row_count"] or 0) for row in cur.fetchall()}
        fact_rows = sum(count for table, count in rows.items() if table.startswith("dw.fact_"))
        dw_rows = sum(count for table, count in rows.items() if table.startswith("dw."))
        if fact_rows > 0:
            return fact_rows, "postgres.dw.fact_rows", rows
        if dw_rows > 0:
            return dw_rows, "postgres.dw_rows", rows
        return None, None, rows
    except Exception:
        return None, None, {}


def estimate_scale(config: Any, source_diff: SourceDiff, table_stats: dict[str, Any] | None = None) -> tuple[int, str, dict[str, int]]:
    table_stats = table_stats or {}
    if isinstance(table_stats.get("estimated_rows"), int) and table_stats["estimated_rows"] > 0:
        rows = int(table_stats["estimated_rows"])
        return rows, str(table_stats.get("estimation_source") or "table_stats.estimated_rows"), {}

    summary_rows, summary_source = _pipeline_summary_rows()
    pg_rows, pg_source, pg_table_stats = _postgres_rows(table_stats.get("database_url"))
    source_rows = source_diff.total_row_count
    target_rows = int(getattr(config, "row_target", {}).get("expected_real_min_rows") or 0)

    candidates = [
        (summary_rows, summary_source),
        (pg_rows, pg_source),
        (source_rows if source_rows > 0 else None, "source_manifest.total_row_count"),
        (target_rows if target_rows > 0 else None, "profile.expected_real_min_rows"),
    ]
    for rows, source in candidates:
        if rows:
            return int(rows), str(source), pg_table_stats
    return 0, "no_estimate_available", pg_table_stats


def _parallelism(user_value: int | None = None) -> int:
    if user_value and user_value > 0:
        return user_value
    raw = os.getenv("DSS_PARALLELISM", "auto").strip().lower()
    if raw == "1":
        return 1
    if raw.isdigit():
        return max(1, int(raw))
    cpus = os.cpu_count() or 2
    return max(1, min(8, cpus // 2 or 1))


def _tables_from_sources(diff: SourceDiff) -> list[str]:
    tables: list[str] = []
    for source in diff.changed_sources:
        name = source.source_name
        for token, affected in SOURCE_TABLE_HINTS.items():
            if token in name or token in source.source_path:
                tables.extend(affected)
    return sorted(set(tables))


def _scan_source_partitions(path: Path, table_name: str) -> list[str]:
    mapping = SOURCE_PARTITION_COLUMNS.get(table_name)
    if not mapping or not path.exists() or path.suffix.lower() not in {".parquet", ".csv"}:
        return []
    try:
        lf = pl.scan_parquet(path) if path.suffix.lower() == ".parquet" else pl.scan_csv(path, infer_schema_length=1000, ignore_errors=True)
        schema = set(lf.collect_schema().names())
        exprs = []
        aliases = []
        for out_col, source_col in mapping.items():
            if source_col not in schema:
                return []
            if out_col == "year":
                exprs.append(pl.col(source_col).cast(pl.Date, strict=False).dt.year().alias("year"))
            else:
                exprs.append(pl.col(source_col).alias(out_col))
            aliases.append(out_col)
        frame = lf.select(exprs).unique().collect()
        labels = []
        for row in frame.to_dicts():
            clean = {key: row.get(key) for key in aliases}
            if any(value is None for value in clean.values()):
                return []
            labels.append(partition_label(clean))
        return sorted(set(labels))
    except Exception:
        return []


def _source_matches_table(source_name: str, source_path: str, table_name: str) -> bool:
    for token, affected in SOURCE_TABLE_HINTS.items():
        if table_name in affected and (token in source_name or token in source_path):
            return True
    return False


def derive_affected_partitions(source_diff: SourceDiff, affected_tables: list[str]) -> dict[str, list[str]]:
    partitions: dict[str, list[str]] = {}
    for table_name in affected_tables:
        labels: list[str] = []
        for source in source_diff.changed_sources:
            if not _source_matches_table(source.source_name, source.source_path, table_name):
                continue
            labels.extend(_scan_source_partitions(Path(source.source_path), table_name))
            if not labels:
                inferred = _partition_from_manifest(source, table_name)
                if inferred:
                    labels.append(inferred)
        partitions[table_name] = sorted(set(labels)) if labels else ["ALL"]
    return partitions


def _partition_from_manifest(source: Any, table_name: str) -> str | None:
    if table_name == "fact_outcome" and source.horizon is not None and source.fold is not None and source.candidate_id and source.universe_id:
        return partition_label({"horizon": source.horizon, "fold": source.fold, "candidate_id": source.candidate_id, "universe_id": source.universe_id})
    if table_name == "fact_module_trace" and source.module_name and source.fold is not None and source.candidate_id and source.universe_id:
        return partition_label({"module_name": source.module_name, "fold": source.fold, "candidate_id": source.candidate_id, "universe_id": source.universe_id})
    if table_name in {"fact_position_daily", "fact_signal_daily"} and source.min_date and source.fold is not None and source.candidate_id and source.universe_id:
        return partition_label({"year": int(source.min_date[:4]), "fold": source.fold, "candidate_id": source.candidate_id, "universe_id": source.universe_id})
    if table_name in {"fact_market_bar"} and source.min_date:
        return partition_label({"year": int(source.min_date[:4])})
    if table_name == "fact_path_recursive" and source.min_date and source.candidate_id and source.fold is not None:
        return partition_label({"year": int(source.min_date[:4]), "candidate_id": source.candidate_id, "fold": source.fold})
    return None


def _strategy_from_tables(tables: list[str]) -> str | None:
    if not tables:
        return "no_data_refresh"
    if tables == ["fact_whatif"]:
        return "whatif_only"
    if tables == ["query_logs"]:
        return "querylog_only"
    return None


def build_execution_plan(
    *,
    config: Any,
    source_diff: SourceDiff,
    previous_manifest_available: bool,
    requested_strategy: str = "auto",
    table_stats: dict[str, Any] | None = None,
    parallelism: int | None = None,
    dry_run: bool = False,
) -> ExecutionPlan:
    estimated_rows, estimation_source, _ = estimate_scale(config, source_diff, table_stats)
    scale = _scale_class(estimated_rows)
    affected_tables = _tables_from_sources(source_diff)
    affected_partitions = derive_affected_partitions(source_diff, affected_tables)
    marts = marts_for_tables(affected_tables)
    cache = cache_for_marts(marts)
    changed_sources = [entry.source_name for entry in source_diff.changed_sources]
    requested = requested_strategy.replace("-", "_")

    if dry_run or requested == "dry_run":
        return ExecutionPlan(
            strategy="dry_run",
            reason="Dry run requested; no database changes will be made.",
            estimated_rows=estimated_rows,
            estimation_source=estimation_source,
            scale_class=scale,
            parallelism=_parallelism(parallelism),
            changed_sources=changed_sources,
            affected_tables=affected_tables,
            affected_partitions=affected_partitions,
            facts_to_build=affected_tables,
            dimensions_to_build=["all"] if affected_tables else [],
            marts_to_refresh=marts,
            cache_to_invalidate=cache,
            validation_level="strict" if scale != "small" else "standard",
            publish_mode="staged",
        )

    if requested == "full":
        strategy = "full_refresh"
        reason = "Full refresh explicitly requested."
    elif requested in {"incremental", "incremental_partition_refresh"} and not affected_tables and previous_manifest_available:
        strategy = "no_data_refresh"
        reason = "Incremental refresh requested, but no data-bearing source changed."
    elif requested in {"incremental", "incremental_partition_refresh"}:
        strategy = "incremental_partition_refresh"
        reason = "Incremental refresh explicitly requested."
    elif requested == "backfill":
        strategy = "backfill"
        reason = "Backfill explicitly requested."
    elif requested == "pending_outcomes":
        strategy = "pending_outcomes"
        reason = "Pending-outcomes maintenance explicitly requested."
        affected_tables = ["fact_outcome"]
        affected_partitions = {"fact_outcome": ["ALL"]}
        marts = marts_for_tables(affected_tables)
        cache = cache_for_marts(marts)
    elif not previous_manifest_available:
        strategy = "full_refresh"
        reason = "No previous source manifest was available."
    elif not affected_tables:
        strategy = "no_data_refresh"
        reason = "No data-bearing source changed."
    elif (single := _strategy_from_tables(affected_tables)) in {"whatif_only", "querylog_only"}:
        strategy = single
        reason = f"Only {affected_tables[0]} changed."
    elif source_diff.changed_ratio > 0.60:
        strategy = "full_refresh"
        reason = f"Changed row estimate is {source_diff.changed_ratio:.1%}, above 60%."
    elif scale == "small" and source_diff.changed_ratio > 0.20:
        strategy = "full_refresh"
        reason = "Dataset is small and a broad change is cheaper as a full refresh."
    else:
        strategy = "incremental_partition_refresh"
        reason = "Changed rows/partitions are bounded enough for partition refresh."

    if affected_partitions and any(parts == ["ALL"] for parts in affected_partitions.values()) and strategy == "incremental_partition_refresh":
        reason = f"{reason} Some partitions could not be inferred exactly and are marked ALL."

    if strategy == "no_data_refresh" and marts == ["mart.mv_query_performance"]:
        strategy = "querylog_only"
    if strategy in {"full_refresh", "backfill"}:
        marts = ALL_MARTS
        cache = cache_for_marts(marts)
        affected_tables = affected_tables or ["all"]
        affected_partitions = affected_partitions or {}

    validation_level = "strict" if scale in {"medium", "large"} or strategy != "no_data_refresh" else "standard"
    publish_mode = "staged" if strategy in {"incremental_partition_refresh", "backfill"} or scale != "small" else "direct"

    return ExecutionPlan(
        strategy=strategy,
        reason=reason,
        estimated_rows=estimated_rows,
        estimation_source=estimation_source,
        scale_class=scale,
        parallelism=_parallelism(parallelism),
        changed_sources=changed_sources,
        affected_tables=affected_tables,
        affected_partitions=affected_partitions,
        facts_to_build=affected_tables if strategy != "no_data_refresh" else [],
        dimensions_to_build=["all"] if strategy in {"full_refresh", "backfill"} else [],
        marts_to_refresh=marts,
        cache_to_invalidate=cache,
        validation_level=validation_level,
        publish_mode=publish_mode,
    )
