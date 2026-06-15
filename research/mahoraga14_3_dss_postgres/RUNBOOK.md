# Runbook

## Install

```bash
cd ~/QuantMahoraga/research/mahoraga14_3_dss_postgres
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Build Parquet

```bash
python -m etl.run_pipeline --mode parquet --profile standard
python -m etl.validate_outputs
```

Outputs:

- `outputs/parquet/oltp/*.parquet`
- `outputs/parquet/dimensions/*.parquet`
- `outputs/parquet/facts/*.parquet`
- `outputs/reports/pipeline_summary.json`
- `outputs/reports/validation_report.json`

## Load Postgres

Local socket:

```bash
export DSS_BACKEND=postgres
export DATABASE_URL="postgresql:///mahoraga_dss"
python -m etl.run_pipeline --mode postgres --profile standard --truncate
python -m etl.refresh_views
python -m etl.validate_postgres
python -m scripts.smoke_postgres
```

## Adaptive Pipeline

Plan without modifying Postgres:

```bash
export DSS_BACKEND=postgres
export DATABASE_URL="postgresql:///mahoraga_dss"
python -m etl.run_adaptive --strategy auto --dry-run
```

Run adaptive execution:

```bash
python -m etl.run_adaptive --strategy auto
```

Maintain pending outcome state:

```bash
python -m etl.run_adaptive --strategy pending-outcomes
```

Full refresh remains the fallback when no previous manifest exists, when a
large portion of rows changed, or when the affected tables are not supported by
the incremental partition loader. Incremental partition refresh is currently
supported for selected facts documented in
`docs/SCALABILITY_AND_OPERATIONS.md`.

Refresh dependent marts manually:

```bash
python -m etl.refresh_views --strategy dependency --changed-tables fact_outcome,fact_position_daily
```

Operational reports are written to `outputs/control/`.

Validate the scalable layer:

```bash
make validate-scalable
```

## Engineering Benchmarks

```bash
python -m scripts.benchmark_queries --smoke
python -m scripts.partition_pruning_demo
python -m scripts.load_test_api --base-url http://127.0.0.1:8002 --concurrency 20 --requests 1000
```

Benchmark-only scale fixtures:

```bash
python -m scripts.generate_scale_fixture --target-rows 4000000
```

Fixtures are written under `outputs/scale_fixtures/` with
`benchmark_mode=true`. Do not load them into the standard DSS or present them as
research evidence.

TCP:

```bash
export DSS_BACKEND=postgres
export DATABASE_URL="postgresql://<user>:<password>@127.0.0.1:5432/mahoraga_dss"
```

Docker optional:

```bash
docker compose -f docker-compose.postgres.yml up -d
export DATABASE_URL="postgresql://<user>:<password>@127.0.0.1:5432/mahoraga_dss"
python -m etl.run_pipeline --mode postgres --profile standard --truncate
```

Do not store real passwords in tracked files.

## Validate OLTP, DW, Marts

OLTP should be non-empty for:

- `oltp.research_run`
- `oltp.data_snapshot`
- `oltp.artifact_inventory`
- `oltp.candidate_grid`

DW should be non-empty for core dimensions and facts:

- `dw.dim_candidate`
- `dw.dim_universe`
- `dw.fact_decision_state`
- `dw.fact_position_daily`
- `dw.fact_module_trace`
- `dw.fact_outcome`
- `dw.fact_candidate_metric`
- `dw.fact_whatif`

Marts should be refreshable and queryable:

- `mart.mv_scorecard_candidate`
- `mart.mv_decision_replay`
- `mart.mv_module_effectiveness`
- `mart.mv_ticker_contribution`
- `mart.mv_regime_behavior`
- `mart.mv_whatif_grid`
- `mart.mv_query_performance`

Manual SQL:

```sql
SELECT COUNT(*) FROM oltp.research_run;
SELECT COUNT(*) FROM oltp.data_snapshot;
SELECT COUNT(*) FROM oltp.artifact_inventory;
SELECT COUNT(*) FROM oltp.candidate_grid;
SELECT COUNT(*) FROM dw.fact_decision_state;
SELECT COUNT(*) FROM dw.fact_position_daily;
SELECT COUNT(*) FROM dw.fact_module_trace;
SELECT COUNT(*) FROM dw.fact_outcome;
SELECT COUNT(*) FROM dw.fact_candidate_metric;
SELECT COUNT(*) FROM dw.fact_whatif;
SELECT COUNT(*) FROM mart.mv_scorecard_candidate;
SELECT COUNT(*) FROM mart.mv_decision_replay;
SELECT COUNT(*) FROM mart.mv_module_effectiveness;
SELECT COUNT(*) FROM mart.mv_ticker_contribution;
SELECT COUNT(*) FROM mart.mv_regime_behavior;
SELECT COUNT(*) FROM mart.mv_whatif_grid;
```

## Start API And Frontend

```bash
export DSS_BACKEND=postgres
export DATABASE_URL="postgresql:///mahoraga_dss"
uvicorn api.main:app --reload --host 127.0.0.1 --port 8002
```

```bash
cd ~/QuantMahoraga/research/mahoraga14_3_dss_postgres/frontend
npm install
VITE_API_BASE="http://127.0.0.1:8002" npm run dev
```

Build check:

```bash
cd ~/QuantMahoraga/research/mahoraga14_3_dss_postgres/frontend
npm run build
```

Vite 7 requires Node `20.19+` or `22.12+`. Use `nvm use 20` or `nvm use 22` if an older Node is active.

## Frontend Research UX Checks

Open `http://127.0.0.1:5174` with `VITE_API_BASE="http://127.0.0.1:8002"` and verify:

- `Command Center`: shows `Official Baseline — Mahoraga 14.3R ROBUST_MAIN`, secondary candidate ID `B1.05_C1.10_L1.10_R1.05`, backend, real rows, simulated what-if rows, marts, query-log status, metrics, best/official/worst observed candidates, benchmark comparison, and research-question cards.
- `Baseline Evidence`: shows stitched comparison, Newey-West alpha/beta, fold summary, p/q values, and cost/slippage sensitivity from official baseline outputs.
- `Robustness Lab`: shows official marker context, sensitivity ranking, Pareto trade-off, plateau radius, candidate ranking, and worst-fold damage from extended robustness outputs.
- `What-if & Stress`: observed/audited scenarios are separate from `demo_mode=true` simulated what-if rows; sliders use `Apply scenario`.
- `Decision Replay`: shows date/fold/candidate/regime/exposure state, weights, modules, outcomes, and a professional timeline empty state when a timeline is not materialized.
- `Module Attribution`: shows activation/helped/outcome evidence by module and horizon.
- `Ticker Contribution`: shows positive/negative contribution, selection rate, leader flag rate, average weight, and concentration.
- `Regime Analysis`: shows return, benchmark, exposure, drawdown, backoff, continuation, and leader blend by regime.
- `OLAP Explorer`: uses guided presets for slice/dice/roll-up/drill-down/pivot operations; no free-text filters are required.
- `Data Engineering`: shows active backend, latest run, OLTP/DW/mart row counts, real/simulated rows, validation status, available marts, and query performance.

Useful endpoint smoke checks:

```bash
curl -sS http://127.0.0.1:8002/data/health-summary | python -m json.tool | head
curl -sS http://127.0.0.1:8002/research/command-center | python -m json.tool | head
curl -sS http://127.0.0.1:8002/research/baseline-evidence | python -m json.tool | head
curl -sS http://127.0.0.1:8002/research/extended-summary | python -m json.tool | head
curl -sS http://127.0.0.1:8002/research/best-official-worst | python -m json.tool | head
```

## Windows Without Postgres

```powershell
cd D:\QuantMahoraga\research\mahoraga14_3_dss_postgres
python -m pip install -r requirements.txt
$env:DSS_BACKEND="parquet"
python -m etl.run_pipeline --mode parquet --profile small
uvicorn api.main:app --host 127.0.0.1 --port 8002
```

## Real Vs Demo Rows

`pipeline_summary.json` reports:

- `real_rows_written_estimate`: rows derived from current real artifacts.
- `demo_rows_written`: rows explicitly flagged with `demo_mode=true`.
- `total_rows_written`: real estimate plus demo rows and metadata tables.
- `expected_real_min_rows_for_profile`: target for the selected profile.
- `real_row_target_met`: true only when real artifact-derived rows meet the target.

The current `standard` profile produces about 494k real rows. It does not meet the 4M real-row target with the artifacts currently available.

In the frontend, this is displayed as audited Postgres/parquet artifacts plus flagged simulated what-if rows. Do not present `demo_mode=true` as a global system label. Only what-if rows and explicitly flagged scenario rows are simulated.

## Git LFS Pointers

If a data file is a Git LFS pointer, the ETL stops with the affected path and this instruction:

```bash
git lfs install && git lfs pull
```
