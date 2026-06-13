# Runbook

## Install

```powershell
cd D:\QuantMahoraga\research\mahoraga14_3_dss_postgres
python -m pip install -r requirements.txt
```

## Discover Artifacts

```powershell
python -m etl.discover_artifacts
```

Output: `outputs/reports/artifact_inventory.csv`.

## Build Parquet Dev Mode

```powershell
python -m etl.run_pipeline --mode parquet --profile small
```

Outputs:

- `outputs/parquet/dimensions/*.parquet`
- `outputs/parquet/facts/*.parquet`
- `outputs/parquet/oltp/*.parquet`
- `outputs/reports/pipeline_summary.json`
- `outputs/reports/validation_report.json`

## Build Without Synthetic What-if Grid

```powershell
python -m etl.run_pipeline --mode parquet --profile small --no-demo-grid
```

## Optional Docker Postgres

```powershell
docker compose -f docker-compose.postgres.yml up -d
$env:DATABASE_URL="postgresql://mahoraga:mahoraga@127.0.0.1:5432/mahoraga_dss"
```

## Real Postgres Load

```powershell
$env:DATABASE_URL="postgresql://mahoraga:mahoraga@127.0.0.1:5432/mahoraga_dss"
python -m etl.run_pipeline --mode postgres --profile standard
```

Separate load/refresh commands:

```powershell
python -m etl.load_postgres --truncate
python -m etl.refresh_views
```

## Validate

```powershell
python -m etl.validate_outputs
python -m pytest
```

## Start API

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8010
```

Postgres API:

```powershell
$env:DSS_BACKEND="postgres"
$env:DATABASE_URL="postgresql://mahoraga:mahoraga@127.0.0.1:5432/mahoraga_dss"
uvicorn api.main:app --host 127.0.0.1 --port 8010
```

## Start Frontend

```powershell
cd D:\QuantMahoraga\research\mahoraga14_3_dss_postgres\frontend
npm.cmd install
$env:VITE_API_BASE="http://127.0.0.1:8010"
npm.cmd run dev
```

Open `http://127.0.0.1:5174`.

## Competition Profile

```powershell
python -m etl.run_pipeline --mode postgres --profile competition
```

The profile expands the explicitly flagged demo what-if grid and loads all available real facts. It reports whether real artifact-derived rows meet the competition target. It does not invent official performance when the repository lacks enough granular source artifacts.

