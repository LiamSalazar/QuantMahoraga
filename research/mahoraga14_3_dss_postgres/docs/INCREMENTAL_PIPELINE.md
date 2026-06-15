# Incremental Pipeline

The incremental path is an operational layer around the existing DSS contract.
It does not change official metrics, final table names, materialized view names,
or frontend payload semantics.

## Execution Flow

1. Build a source manifest and diff it against the previous manifest.
2. Map changed sources to affected facts.
3. Derive logical affected partitions from changed source files.
4. Build only selected facts when a safe builder exists.
5. Write table-level Parquet for those selected facts.
6. Optionally write affected partitioned Parquet under
   `outputs/parquet_partitioned/`.
7. Validate data contracts for selected facts.
8. Replace affected logical partitions in Postgres with `DELETE` + `COPY`.
9. Analyze affected tables.
10. Refresh dependent marts only.
11. Record cache invalidation intent.
12. Publish the run to `oltp.active_dss_run`.

## Partial Build Support

Implemented selected builders:

- `fact_whatif`
- `fact_outcome`
- `fact_position_daily`
- `fact_module_trace`
- `fact_signal_daily`
- `fact_market_bar`
- `fact_decision_state`
- `fact_path_recursive`
- `fact_candidate_metric`
- `fact_robustness_surface`
- `fact_cost_sensitivity`
- `fact_universe_sensitivity`

The selected build avoids rebuilding all DSS facts. A selected fact may still
read its source cube and then filter/load only affected logical partitions. This
is intentional: correctness is prioritized over aggressive source-level slicing.

## Logical Partitions

- `fact_position_daily`: `year/fold/candidate_id/universe_id`
- `fact_signal_daily`: `year/fold/candidate_id/universe_id`
- `fact_market_bar`: `year`
- `fact_outcome`: `horizon/fold/candidate_id/universe_id`
- `fact_module_trace`: `module_name/fold/candidate_id/universe_id`
- `fact_whatif`: `scenario_id/fold/horizon/demo_mode`
- `fact_path_recursive`: `year/candidate_id/fold`

`adaptive_planner` scans changed source files for distinct partition keys. If it
cannot infer a safe partition set, it emits `ALL` for that table. `ALL` is only
accepted for safe broad replacement such as `fact_whatif`; otherwise the runner
falls back to full refresh.

## Incremental Facts

End-to-end incremental replacement is implemented for the supported facts above.
The main defended routes are:

- `fact_whatif`: broad replacement is allowed; refreshes `mart.mv_whatif_grid`.
- `fact_outcome`: partition replacement by horizon/fold/candidate/universe;
  refreshes outcome-dependent marts.
- `fact_position_daily`: partition replacement by year/fold/candidate/universe;
  refreshes ticker/replay/regime marts.
- `fact_module_trace`: partition replacement by module/fold/candidate/universe;
  refreshes module/replay marts.

If a changed source also affects unsupported or non-partition-safe tables, the
adaptive runner writes an explicit fallback plan and runs a full refresh.

## Publish And Rollback

Successful publish writes:

- `oltp.publish_log`
- `oltp.active_dss_run`

`active_dss_run` is a single-row operational marker used by reporting and future
cache keys. Current DSS queries are not globally gated by active run id yet,
because that would change a broad read contract. Rollback is therefore a
recorded previous-active-run anchor, not an automated data rewind.

The `pending-outcomes` maintenance strategy updates `oltp.pending_outcome` but
does not publish a new active DSS run, because it does not load replacement fact
rows.

## Scale Validation Status

The incremental/adaptive architecture was validated without changing the
standard DSS contract. The standard Postgres pipeline, `validate_postgres`,
`smoke_postgres`, and critical data contracts passed for the real DSS dataset.
Those checks are separate from benchmark-only fixture validation.

Scale fixture validation covered Parquet generation and row-count/read behavior:

- 4M benchmark fixture: 4 Parquet files, 1,000,000 rows each, 4,000,000 rows
  total, with manifest flags `benchmark_mode=true` and
  `not_research_evidence=true`.
- 40M benchmark fixture: `outputs/scale_fixtures/scale_40m`, generated in
  approximately 1m11s, observed size around 42 MB, 40 Parquet files,
  1,000,000 rows each, 40,000,000 rows total.

The 4M and 40M fixtures are not quant evidence, were not loaded into the
standard DSS, and must not be interpreted as official Mahoraga results.

## Current Limits

- Materialized view refresh is concurrent only where Postgres exposes a valid
  unique index; otherwise it falls back to normal refresh and logs the method.
- Source-level partial reads are conservative. Selected facts may still build
  the full selected fact before loading only affected partitions.
- Active-run gating is prepared but not enforced across all API queries.
- Cache invalidation is logged as a safe hook; no risky stale-response cache is
  enabled by default.

## Operational Commands

Quick validation environment:

```bash
cd ~/QuantMahoraga/research/mahoraga14_3_dss_postgres
source .venv/bin/activate

export DSS_BACKEND=postgres
export DATABASE_URL="postgresql:///mahoraga_dss"
```

Backend/API:

```bash
cd ~/QuantMahoraga/research/mahoraga14_3_dss_postgres
source .venv/bin/activate

export DSS_BACKEND=postgres
export DATABASE_URL="postgresql:///mahoraga_dss"

uvicorn api.main:app --reload --host 127.0.0.1 --port 8002
```

Frontend:

```bash
cd ~/QuantMahoraga/research/mahoraga14_3_dss_postgres/frontend

npm install
npm run build

VITE_API_BASE="http://127.0.0.1:8002" npm run dev
```

App URL:

```text
http://127.0.0.1:5174
```
