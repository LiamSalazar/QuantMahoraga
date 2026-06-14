# Mahoraga Quant DSS Postgres

Decision Support System over frozen Mahoraga14_3 baseline outputs and extended-analysis audit artifacts.

This layer does not modify `baseline/mahoraga14_3_baseline`, recalibrate official results, or mark synthetic rows as real. It reads existing artifacts, writes Parquet staging, loads a real Postgres OLTP/DW/mart model, exposes FastAPI endpoints, and serves the React DSS.

## Linux With Local Postgres Socket

```bash
cd ~/QuantMahoraga/research/mahoraga14_3_dss_postgres
source .venv/bin/activate

export DSS_BACKEND=postgres
export DATABASE_URL="postgresql:///mahoraga_dss"

python -m etl.run_pipeline --mode parquet --profile standard
python -m etl.run_pipeline --mode postgres --profile standard --truncate
python -m etl.refresh_views
python -m etl.validate_outputs
python -m etl.validate_postgres
python -m scripts.smoke_postgres

uvicorn api.main:app --reload --host 127.0.0.1 --port 8002
```

Frontend:

```bash
cd ~/QuantMahoraga/research/mahoraga14_3_dss_postgres/frontend
npm install
VITE_API_BASE="http://127.0.0.1:8002" npm run dev
```

## Other Modes

Windows without Postgres:

```powershell
cd D:\QuantMahoraga\research\mahoraga14_3_dss_postgres
python -m pip install -r requirements.txt
$env:DSS_BACKEND="parquet"
python -m etl.run_pipeline --mode parquet --profile small
uvicorn api.main:app --host 127.0.0.1 --port 8002
```

Docker Postgres:

```bash
docker compose -f docker-compose.postgres.yml up -d
export DSS_BACKEND=postgres
export DATABASE_URL="postgresql://<user>:<password>@127.0.0.1:5432/mahoraga_dss"
python -m etl.run_pipeline --mode postgres --profile standard --truncate
```

TCP Postgres without Docker:

```bash
export DSS_BACKEND=postgres
export DATABASE_URL="postgresql://<user>:<password>@127.0.0.1:5432/mahoraga_dss"
```

Do not commit real passwords. Keep credentials in the shell, a local untracked `.env`, or a secret manager.

## Current Validated Counts

Latest Linux/Postgres standard run:

- `total_rows_written`: `496,967`
- `real_rows_written_estimate`: `494,467`
- `demo_rows_written`: `2,500`
- `expected_real_min_rows_for_profile`: `4,000,000`
- `real_row_target_met`: `false`
- `validation_passed`: `true`

The available real artifacts currently produce about 494k real rows. The architecture can scale with more real candidates, universes, horizons, module traces, decisions, positions, and outcomes, but this iteration does not invent rows or inflate official results. Extended what-if/demo rows remain flagged with `demo_mode=true`.

## API Endpoints

With `DSS_BACKEND=postgres`, these endpoints are validated:

- `/health`
- `/metadata/options`
- `/overview`
- `/scorecard`
- `/robustness/surface`
- `/whatif/grid`
- `/decision/replay`
- `/slice`
- `/drilldown`
- `/module/effectiveness`
- `/ticker/contribution`
- `/regime/behavior`
- `/fold/performance`
- `/candidate/compare`
- `/query/performance`

`/query/performance` reads recent logs directly from `oltp.dss_query_log`; `mart.mv_query_performance` remains the refreshable historical summary.

## Data Checks

```bash
python -m etl.validate_outputs
python -m etl.validate_postgres
python -m scripts.smoke_postgres
```

The smoke test prints JSON with `passed`, `checked_tables`, `row_counts`, `checked_views`, and `failures`, and exits with code 1 on critical failure.
