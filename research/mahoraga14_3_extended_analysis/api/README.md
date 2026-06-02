# Mahoraga Extended Analysis API

Run from the repository root:

```powershell
python .\research\mahoraga14_3_extended_analysis\run_api.py
```

Default URL: `http://127.0.0.1:8000`. If that port is occupied, `run_api.py` chooses the next free port in `8000-8019` and prints the chosen URL.

## Endpoints

- `GET /health`
- `GET /summary/baseline`
- `GET /robustness/multipliers`
- `GET /robustness/plateau`
- `GET /decisions`
- `GET /positions`
- `GET /module-trace`
- `GET /market-context`
- `GET /universes/summary`

Large cube endpoints accept filters such as `date_start`, `date_end`, `fold`, `candidate_id`, `universe_id`, and `limit`. `positions` also accepts `ticker` and `selected_only`; `module-trace` accepts `module_name`.

The API reads materialized CSV/Parquet outputs. It does not recompute the analysis.
