# Implementation Report

- run_id: `ext14_3_20260602T082114Z0000`

## Structure created

- `README.md`, `requirements_extended.txt`, `run_extended_analysis.py`, `run_api.py`
- `src/extended_analysis/` analysis package
- `api/` FastAPI app
- `frontend/` React/TypeScript/Tailwind app
- `outputs/extended_multiplier_robustness/`
- `outputs/universe_robustness/`
- `outputs/audit_cube/`
- `outputs/figures/`
- `outputs/reports/`
- `source_snapshot/` copied from frozen baseline source

## What executed

- Estimated candidate evaluations before run: one-dimensional 18, two-dimensional selected after sensitivity, extremes 4.
- Actual multiplier candidate rows: 42
- Universe rows: 15
- Decision cube rows: 13770
- Position cube rows: 165240
- Module trace cube rows: 96390
- Outcome cube rows: 41310
- Market context rows: 2295

## Timings

```
                         stage    seconds
             base_walk_forward   1.254203
extended_multiplier_robustness   9.206534
           universe_robustness  10.459717
                    audit_cube 455.079047
                         total 475.999522
```

## Cache and fallback

- base walk-forward loaded from research cache

## Skipped or limited

- alternate universe walk-forward aborted by compute budget after materializing coverage: negative_control_nontech

## Reproduction

```powershell
cd D:\QuantMahoraga
python .\research\mahoraga14_3_extended_analysis\run_extended_analysis.py
python .\research\mahoraga14_3_extended_analysis\run_api.py
```

Frontend:

```powershell
cd D:\QuantMahoraga\research\mahoraga14_3_extended_analysis\frontend
npm install
npm run dev
```

## Baseline safety

The implementation writes only inside `research/mahoraga14_3_extended_analysis`. The official `baseline/` package is treated as read-only.

## Post-run verification

- Python entrypoints compiled with `py_compile`.
- API started successfully on `http://127.0.0.1:8001` because `8000` was already occupied by another local service.
- Checked `/health`, `/summary/baseline`, `/robustness/plateau`, `/decisions`, and `/positions`.
- Frontend dependencies installed with a research-local npm cache because `npm` was not in PATH.
- Frontend production build succeeded with `npm run build`.
- Frontend dev server started on `http://127.0.0.1:5173` with `VITE_API_BASE=http://127.0.0.1:8001`.
- Browser verification confirmed Overview, Multiplier Robustness, Decision Audit, desktop image loading, and a basic mobile width check.
