# Mahoraga Quant DSS / Research Intelligence Layer

This layer turns frozen Mahoraga14_3 baseline outputs and extended-analysis audit artifacts into a Decision Support System for research exploration.

It does not modify `baseline/mahoraga14_3_baseline`, recalibrate the official candidate, run live trading, or replace the official CSV/Parquet evidence. It reads existing artifacts, builds Parquet staging tables, can load a real Postgres OLTP/DW/mart model, serves FastAPI endpoints, and provides a dark interactive web DSS.

## What It Answers

- Robustness by candidate, fold, universe, regime, and multiplier neighborhood.
- What-if behavior for budget, conviction, leader, backoff, costs, and slippage.
- Decision replay by date, fold, candidate, module, ticker, and future horizon.
- Slice, dice, drill-down, roll-up, and pivot-style guided cube exploration.
- Module attribution, ticker contribution, regime behavior, drawdown replay, and query performance.

## Modes

| Mode | Command | Purpose |
|---|---|---|
| A. Postgres real | `python -m etl.run_pipeline --mode postgres --profile standard` | Linux/final presentation with real `DATABASE_URL`, SQL schemas, COPY load, and materialized views. |
| B. Docker optional | `docker compose -f docker-compose.postgres.yml up -d` | Local Postgres if Docker Desktop or Linux Docker is available. |
| C. Dev without Postgres | `python -m etl.run_pipeline --mode parquet --profile small` | Windows-friendly API/frontend development from Parquet plus explicitly flagged demo what-if rows. |

## Current Local Build

The latest local `small` parquet run produced:

- Total rows written: `492,687`.
- Real artifact-derived row estimate: `492,327`.
- Explicit demo what-if rows: `360`.
- Validation: passed.

The current repository artifacts do not contain enough granular candidate/fold/ticker rows to honestly produce a 4M+ real fact set. The `standard` and `competition` profiles are configured for larger what-if grids and real Postgres loading, but they will still report whether the real-row target is met instead of inventing official performance.

## Run on Windows Without Postgres

```powershell
cd D:\QuantMahoraga\research\mahoraga14_3_dss_postgres
python -m pip install -r requirements.txt
python -m etl.run_pipeline --mode parquet --profile small
uvicorn api.main:app --host 127.0.0.1 --port 8010
```

In another terminal:

```powershell
cd D:\QuantMahoraga\research\mahoraga14_3_dss_postgres\frontend
npm.cmd install
$env:VITE_API_BASE="http://127.0.0.1:8010"
npm.cmd run dev
```

Open `http://127.0.0.1:5174`.

## Run on Linux With Postgres

```bash
cd /path/to/QuantMahoraga/research/mahoraga14_3_dss_postgres
python -m pip install -r requirements.txt
export DATABASE_URL="postgresql://mahoraga:mahoraga@127.0.0.1:5432/mahoraga_dss"
python -m etl.run_pipeline --mode postgres --profile standard
uvicorn api.main:app --host 127.0.0.1 --port 8010
```

For Docker:

```bash
docker compose -f docker-compose.postgres.yml up -d
export DATABASE_URL="postgresql://mahoraga:mahoraga@127.0.0.1:5432/mahoraga_dss"
python -m etl.run_pipeline --mode postgres --profile standard
```

## API

Endpoints:

- `GET /health`
- `GET /metadata/options`
- `GET /overview`
- `GET /scorecard`
- `GET /robustness/surface`
- `GET /whatif/grid`
- `GET /decision/replay`
- `GET /slice`
- `GET /drilldown`
- `GET /module/effectiveness`
- `GET /ticker/contribution`
- `GET /regime/behavior`
- `GET /fold/performance`
- `GET /candidate/compare`
- `GET /query/performance`

Filters are validated against real option lists from dimensions. The frontend uses only guided controls: selects, sliders, toggles/checkboxes, date pickers, segmented controls, and table sorting/paging.

## Tests

```powershell
cd D:\QuantMahoraga\research\mahoraga14_3_dss_postgres
python -m pytest
```

Latest local result: `7 passed`.

