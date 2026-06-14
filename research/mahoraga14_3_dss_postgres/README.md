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
- `/data/health-summary`
- `/metadata/options`
- `/labels/candidates`
- `/research/command-center`
- `/research/baseline-evidence`
- `/research/extended-summary`
- `/research/best-official-worst`
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

## Frontend Research UX

The React frontend is organized as a lazy-loaded Mahoraga Quant Research Command Center. Startup loads only `/data/health-summary`, cached `/metadata/options`, and the active Command Center research aggregate. Other views query their data only when opened.

- `Command Center`: answers what Mahoraga is, what the official frozen baseline is, and what evidence supports it. Uses `/research/command-center`, `mart.mv_scorecard_candidate`, `mart.mv_drawdown_replay`, official baseline CSVs, and extended robustness outputs.
- `Baseline Evidence`: formal baseline evidence from `stitched_comparison_official.csv`, fold summary, Newey-West alpha, p/q values, cost/slippage sensitivity, exposure, turnover, and return per exposure.
- `Robustness Lab`: observed multiplier robustness, sensitivity tornado, Pareto trade-off, plateau radius, and worst-fold damage from `fact_robustness_surface` and extended multiplier audit CSVs.
- `What-if & Stress`: separates observed/audited scenarios from `demo_mode=true` simulated what-if rows. Sliders use an explicit Apply action and simulated rows are labeled `Simulated what-if · not official performance`.
- `Decision Replay`: reconstructs date-level decisions from `fact_decision_state`, `fact_position_daily`, `fact_module_trace`, `fact_outcome`, and market bars.
- `Module Attribution`: activation/helped/alpha diagnostics by module and horizon from `mart.mv_module_effectiveness`.
- `Ticker Contribution`: contribution, selection, leadership, concentration, and average weight from `mart.mv_ticker_contribution`.
- `Regime Analysis`: return, benchmark, exposure, drawdown, backoff, continuation, and leader blend by regime from `mart.mv_regime_behavior`.
- `OLAP Explorer`: guided presets for slice, dice, roll-up, drill-down, and pivot-style operations through `/slice`; it does not expose free-text filters.
- `Data Engineering`: active backend, latest run, row origin counts, marts, validation status, and query performance from `/data/health-summary` and `/query/performance`.

Candidate labels are presentation-safe:

- `B1.05_C1.10_L1.10_R1.05` is shown as `Official Baseline — Mahoraga 14.3R ROBUST_MAIN`, with the raw ID only as secondary detail.
- Sweep candidates are shown as multiplier labels, for example `Budget 0.90 / Conviction 1.10 / Leader 1.10 / Backoff 1.05`.
- Controlled extremes are shown as `Extreme: pro-risk`, `Extreme: pro-defense`, or explicit stress cases.

The UI does not mark the whole system as demo. It reports `Postgres · audited artifacts + flagged simulated what-if`, along with `real_rows`, `simulated_rows`, and row-level `demo_mode`/origin where relevant.

If a chart would be misleading, the frontend renders KPI/table evidence or a professional empty state instead of a single-point scatter, empty heatmap, or blank timeline.

## Data Checks

```bash
python -m etl.validate_outputs
python -m etl.validate_postgres
python -m scripts.smoke_postgres
```

The smoke test prints JSON with `passed`, `checked_tables`, `row_counts`, `checked_views`, and `failures`, and exits with code 1 on critical failure.

## Frontend Validation

```bash
cd ~/QuantMahoraga/research/mahoraga14_3_dss_postgres
source .venv/bin/activate
export DSS_BACKEND=postgres
export DATABASE_URL="postgresql:///mahoraga_dss"
uvicorn api.main:app --reload --host 127.0.0.1 --port 8002
```

```bash
cd ~/QuantMahoraga/research/mahoraga14_3_dss_postgres/frontend
npm install
npm run build
VITE_API_BASE="http://127.0.0.1:8002" npm run dev
```

Vite 7 requires Node `20.19+` or `22.12+`. If the local Node is older, switch with `nvm` before running `npm run build` or `npm run dev`.
