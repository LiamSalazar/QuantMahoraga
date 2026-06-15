from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from .mart_dependencies import ALL_MARTS, cache_for_marts, marts_for_tables
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
    estimated_rows = source_diff.total_row_count
    scale = _scale_class(estimated_rows)
    affected_tables = _tables_from_sources(source_diff)
    marts = marts_for_tables(affected_tables)
    cache = cache_for_marts(marts)
    changed_sources = [entry.source_name for entry in source_diff.changed_sources]
    requested = requested_strategy.replace("-", "_")

    if dry_run or requested == "dry_run":
        return ExecutionPlan(
            strategy="dry_run",
            reason="Dry run requested; no database changes will be made.",
            estimated_rows=estimated_rows,
            scale_class=scale,
            parallelism=_parallelism(parallelism),
            changed_sources=changed_sources,
            affected_tables=affected_tables,
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

    if strategy == "no_data_refresh" and marts == ["mart.mv_query_performance"]:
        strategy = "querylog_only"
    if strategy in {"full_refresh", "backfill"}:
        marts = ALL_MARTS
        cache = cache_for_marts(marts)
        affected_tables = affected_tables or ["all"]

    validation_level = "strict" if scale in {"medium", "large"} or strategy != "no_data_refresh" else "standard"
    publish_mode = "staged" if strategy in {"incremental_partition_refresh", "backfill"} or scale != "small" else "direct"

    return ExecutionPlan(
        strategy=strategy,
        reason=reason,
        estimated_rows=estimated_rows,
        scale_class=scale,
        parallelism=_parallelism(parallelism),
        changed_sources=changed_sources,
        affected_tables=affected_tables,
        facts_to_build=affected_tables if strategy != "no_data_refresh" else [],
        dimensions_to_build=["all"] if strategy in {"full_refresh", "backfill"} else [],
        marts_to_refresh=marts,
        cache_to_invalidate=cache,
        validation_level=validation_level,
        publish_mode=publish_mode,
    )
