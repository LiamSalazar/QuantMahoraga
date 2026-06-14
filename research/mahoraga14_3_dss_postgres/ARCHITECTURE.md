# Architecture

```mermaid
flowchart LR
  A["Frozen baseline outputs"] --> C["Polars ETL"]
  B["Extended audit cubes"] --> C
  C --> D["Parquet staging"]
  D --> E["Postgres OLTP/DW"]
  E --> F["Materialized marts"]
  D --> G["FastAPI parquet backend"]
  F --> H["FastAPI postgres backend"]
  G --> I["React DSS"]
  H --> I
```

## OLTP

The `oltp` schema models research operations, not trading operations:

- research runs
- data snapshots
- artifact inventory
- candidate grids
- simulation jobs and statuses
- cube builds
- DSS query logs
- audit cases
- what-if requests

It exists to show operational lineage for the laboratory and DSS.

The ETL writes OLTP Parquet tables under `outputs/parquet/oltp/` and the Postgres loader copies at least `research_run`, `data_snapshot`, `artifact_inventory`, and `candidate_grid` before DW dimensions and facts. Missing OLTP Parquet files are reported as 0-row skipped loads rather than hard failures.

## DW

The `dw` schema separates dimensions and facts. Facts are intentionally aligned with Mahoraga research questions: decisions, positions, modules, outcomes, robustness, what-if scenarios, universe stress, costs, and drawdown paths.

## Marts

The `mart` schema contains materialized views optimized for the API:

- `mv_scorecard_candidate`
- `mv_performance_by_fold`
- `mv_robustness_surface`
- `mv_decision_outcome`
- `mv_module_effectiveness`
- `mv_ticker_contribution`
- `mv_regime_behavior`
- `mv_whatif_grid`
- `mv_drawdown_replay`
- `mv_decision_replay`
- `mv_query_performance`

Materialized views are created after base tables and refreshed after data loads. `mart.mv_query_performance` is the historical/materialized summary; the API endpoint `/query/performance` also reads recent aggregates directly from `oltp.dss_query_log` so it can show fresh request logs before the next refresh.

## Nullable Scope Fields

Some facts intentionally combine metrics from different scopes. For `dw.fact_candidate_metric` and `dw.fact_robustness_surface`, `sweep_role` is nullable when the metric set does not have an applicable sweep role, for example universe robustness rows. Uniqueness for `fact_candidate_metric` uses `COALESCE(sweep_role, '__not_applicable__')` rather than a primary key on a nullable column.

## Partitioning And Indexes

Partitioned facts:

- `fact_market_bar`: range partitioned by date.
- `fact_position_daily`: range partitioned by date.
- `fact_outcome`: list partitioned by horizon.
- `fact_module_trace`: list partitioned by module.

Indexes target DSS filters:

- candidate + fold + date
- candidate + horizon + decision date
- asset + date
- module + candidate + fold + date
- regime + candidate + fold
- BRIN temporal indexes for large date-ordered facts

`sql/010_sample_queries.sql` includes `EXPLAIN (ANALYZE, BUFFERS)` examples for the main marts.

## Stack

- Python for orchestration and API.
- Polars for ETL transforms.
- Parquet for local staging and Windows dev mode.
- Postgres for final OLTP/DW/mart execution.
- FastAPI for validated DSS endpoints.
- React/Vite/Recharts for interactive UI.
