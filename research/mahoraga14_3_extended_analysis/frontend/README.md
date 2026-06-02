# Mahoraga Extended Analysis Frontend

Minimal React/TypeScript interface for the extended analysis outputs.

## Install

```powershell
cd D:\QuantMahoraga\research\mahoraga14_3_extended_analysis\frontend
npm install
```

## Run

Start the API first:

```powershell
cd D:\QuantMahoraga
python .\research\mahoraga14_3_extended_analysis\run_api.py
```

Then run the frontend:

```powershell
npm run dev
```

Default URL: `http://127.0.0.1:5173`

The frontend defaults to `http://127.0.0.1:8000` for API calls. If the API prints a different port, set `VITE_API_BASE`, for example:

```powershell
$env:VITE_API_BASE="http://127.0.0.1:8001"
npm run dev
```

## Views

- Baseline Overview: official metrics, robustness summary, universe snapshot and generated figures.
- Multiplier Robustness: candidate table, filters, plateau radius, sensitivity ranking and robustness figures.
- Decision Audit Explorer: filtered access to decision, position, module trace and market context cube data.

## API Endpoints Consumed

- `/summary/baseline`
- `/robustness/multipliers`
- `/robustness/plateau`
- `/decisions`
- `/positions`
- `/module-trace`
- `/market-context`
- `/universes/summary`

## Limitations

The frontend is an exploration layer over materialized outputs. It does not run analysis, does not read Parquet directly, and does not replace the CSV/Parquet audit artifacts.
