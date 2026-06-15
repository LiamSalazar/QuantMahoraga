# Scalability And Operations

This iteration keeps the DSS contract intact: same official baseline, same
facts/marts/endpoints, same Postgres backend, and the same Parquet staging
layout under `outputs/parquet/`. It adds an operations layer around the current
batch pipeline so the system can reason about changes, refresh scope, data
quality, publication, cache invalidation hooks, and benchmark evidence.

## Current Batch Path

The standard path remains valid:

```bash
python -m etl.run_pipeline --mode postgres --profile standard --truncate
python -m etl.validate_postgres
python -m scripts.smoke_postgres
```

It still reads frozen baseline and extended audit artifacts, builds Polars
frames, writes table-level Parquet, loads Postgres by `COPY`, refreshes marts,
and serves the existing FastAPI/React DSS.

## Control Plane

`sql/011_create_control_plane.sql` adds idempotent OLTP tables:

- `oltp.pipeline_run`
- `oltp.pipeline_stage_log`
- `oltp.source_manifest`
- `oltp.partition_manifest`
- `oltp.mart_refresh_log`
- `oltp.data_quality_check`
- `oltp.cache_invalidation_log`
- `oltp.publish_log`
- `oltp.pending_outcome`
- `oltp.schema_migration_log`

These tables do not replace existing OLTP/DW/mart tables. They record what
happened around a run: source changes, stage duration, validation results,
partitions written/loaded, mart refreshes, cache invalidation intent, publish
state, and pending outcomes.

`SCHEMA_VERSION` is exposed as optional metadata on `/health`.

## Source Manifest

`etl/source_manifest.py` scans the expected baseline and extended artifacts,
computes a source hash, row count, date bounds, and simple dimensional metadata
when columns are available. Small files use a full SHA-256 hash. Large files use
a configurable metadata/sample hash to avoid reading the whole artifact just to
detect changes.

Reports are written to:

```text
outputs/control/source_manifest_<run_id>.json
```

When Postgres is available, the latest manifest is upserted into
`oltp.source_manifest`.

## Adaptive Planner

`etl/adaptive_planner.py` produces an `ExecutionPlan` with:

- strategy
- reason
- estimated rows
- scale class
- parallelism
- changed sources
- affected facts and partitions
- marts to refresh
- cache endpoints to invalidate
- validation level
- publish mode

Rules are intentionally conservative:

- explicit `full` means full refresh;
- no previous manifest means full refresh;
- changed rows above 60% means full refresh;
- small data can use full refresh when broad changes are cheaper;
- medium/large data prefers partition refresh when change ratio is bounded;
- fact-specific changes map to dependent marts;
- unsupported incremental tables fall back to full refresh with a written reason.

Dry run:

```bash
python -m etl.run_adaptive --strategy auto --dry-run
```

Execution plan JSON:

```text
outputs/control/execution_plan_<run_id>.json
```

## Incremental Pipeline

`etl/incremental.py` implements real partition replacement for a safe subset of
facts using:

1. rebuild staged Parquet for the current artifacts;
2. derive logical partitions from the staged frame;
3. `DELETE` only the affected logical partition in Postgres;
4. `COPY` replacement rows from Parquet/CSV staging;
5. `ANALYZE` affected tables;
6. refresh only dependent marts.

Supported logical partitions:

- `fact_position_daily`: year + fold + candidate + universe
- `fact_signal_daily`: year + fold + candidate + universe
- `fact_market_bar`: year
- `fact_outcome`: horizon + fold + candidate + universe
- `fact_module_trace`: module + fold + candidate + universe
- `fact_whatif`: scenario + fold + horizon + demo mode
- `fact_path_recursive`: year + candidate + fold

If the planner cannot prove that all affected tables are supported, adaptive
execution falls back to full refresh. This protects official DSS results.

## Partitioned Parquet

The existing table-level Parquet staging remains unchanged. Optional partitioned
staging is added under:

```text
outputs/parquet_partitioned/
```

Examples:

```text
fact_position_daily/year=2020/fold=2/part-000.parquet
fact_outcome/horizon=20/fold=4/part-000.parquet
fact_module_trace/module_name=risk_backoff_layer_v2/fold=3/part-000.parquet
```

Partition manifests are written to `outputs/control/partition_manifest_<run_id>.json`
and `oltp.partition_manifest`.

## Data Contracts

`etl/data_contracts.py` validates critical tables:

- `fact_position_daily`
- `fact_outcome`
- `fact_module_trace`
- `fact_decision_state`
- `fact_whatif`
- `dim_candidate`
- `dim_asset`
- `dim_date`

Contracts define grain, required columns, null rules, allowed folds/horizons,
duplicate keys, and simulated-row separation for what-if rows. Reports are
written as JSON and Markdown in `outputs/control/`.

Errors block adaptive incremental publish. The standard pipeline records
contract results without changing official table semantics.

## Mart Dependencies

`etl/mart_dependencies.py` maps facts to materialized views. `etl.refresh_views`
now supports:

```bash
python -m etl.refresh_views --strategy full
python -m etl.refresh_views --strategy dependency --changed-tables fact_outcome,fact_position_daily
python -m etl.refresh_views --strategy fast
```

The default remains full refresh. Each refreshed mart is logged in
`oltp.mart_refresh_log` when a run id is provided.

## Cache Hooks

No mandatory cache is introduced. That avoids stale payload risk. Adaptive runs
write invalidation intent to `oltp.cache_invalidation_log`, using fact/mart
dependencies:

- `fact_whatif` invalidates `/whatif/grid`;
- `fact_position_daily` invalidates ticker/replay/regime screens;
- `fact_outcome` invalidates distributions/replay/module screens;
- query telemetry invalidates execution evidence.

Future API cache can key by endpoint + filters + active run id and consume this
log.

## Publish And Rollback

The DSS does not currently filter by active run id, so publish is deliberately
lightweight. Successful runs insert `oltp.publish_log`; failed validation does
not publish or invalidate cache. `previous_active_run_id` gives a rollback
anchor for future active-run gating without changing existing endpoints now.

## Pending Outcomes

`etl/pending_outcomes.py` calculates maturity dates conservatively from dates
available in the dataset. It records pending, ready, and computed outcome states
in `oltp.pending_outcome` without modifying `dw.fact_outcome`.

## Domain-Aware Temperature

Temperature is based on analytical function, not recency.

Hot read models:

- scorecards
- fold performance
- benchmark comparison
- best/official/worst
- robustness compare
- ticker contribution summaries
- regime summaries
- module effectiveness
- what-if grid
- execution evidence

Warm drill-through:

- `fact_position_daily`
- `fact_module_trace`
- `fact_outcome`
- `fact_decision_state`
- `fact_path_recursive`
- decision casebook rows

Cold reproducibility/archive:

- raw artifacts
- old manifests
- source snapshots
- full Parquet staging
- failed-run diagnostics
- historical benchmark reports

The planner uses this metadata for cache/prewarm intent and to avoid refreshing
unrelated hot marts when a warm fact partition changes.

## Benchmarks

Representative query suite:

```bash
python -m scripts.benchmark_queries --smoke
python -m scripts.benchmark_queries
```

Outputs:

- `outputs/benchmarks/query_benchmark_summary.csv`
- `outputs/benchmarks/query_benchmark_report.md`
- `outputs/benchmarks/query_plans/*.json`

Partition pruning evidence:

```bash
python -m scripts.partition_pruning_demo
```

Output:

- `outputs/benchmarks/partition_pruning_report.md`

Local API load test:

```bash
python -m scripts.load_test_api --base-url http://127.0.0.1:8002 --concurrency 20 --requests 1000
```

Outputs:

- `outputs/benchmarks/api_load_test_summary.md`
- `outputs/benchmarks/api_load_test_results.csv`

These scripts provide engineering evidence only. They are not claims about
production user capacity.

## Scale Fixtures

Synthetic scale fixtures are benchmark-only:

```bash
python -m scripts.generate_scale_fixture --target-rows 4000000
python -m scripts.generate_scale_fixture --target-rows 40000000
```

They write under `outputs/scale_fixtures/`, include `benchmark_mode=true`, and
must not be loaded into the standard DSS or presented as quant evidence.

## Scaling Path

At around 400k-500k rows, full refresh remains acceptable. At millions of rows,
the adaptive layer reduces recomputation by detecting changed artifacts,
refreshing logical partitions, refreshing dependent marts only, and keeping
frontend endpoints on hot marts rather than raw facts. At tens of millions of
rows, the same design requires stronger production hardening: physical
partition lifecycle automation, richer row-level source lineage, more
concurrent-safe materialized view refreshes, and active-run gating for reads.

Implemented now:

- control plane tables and logging hooks;
- source manifest and diff;
- adaptive dry-run planner;
- data contracts;
- partitioned Parquet writer;
- incremental partition replacement for supported facts;
- dependency-based mart refresh;
- publish and cache invalidation logs;
- pending outcomes table;
- benchmark/query/load-test scripts;
- domain-aware temperature metadata.

Future hardening:

- attach/detach physical partition publication;
- active-run filtering in marts/endpoints;
- durable API cache keyed by active run;
- unique indexes for more concurrent mart refreshes;
- richer cost model using historical p95 and refresh duration;
- automated rollback execution instead of rollback metadata only.
