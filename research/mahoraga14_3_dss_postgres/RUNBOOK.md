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

## Git LFS Pointers

If a data file is a Git LFS pointer, the ETL stops with the affected path and this instruction:

```bash
git lfs install && git lfs pull
```
