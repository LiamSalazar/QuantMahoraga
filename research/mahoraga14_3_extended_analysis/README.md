# Mahoraga 14.3 Extended Analysis

Independent research phase for extended robustness, universe dependence and decision auditability of the frozen official Mahoraga14_3 baseline.

This directory is intentionally separate from `baseline/`. It does not define a new baseline, does not reoptimize the official candidate, and does not modify official outputs.

## Plan

1. Build the phase under `research/mahoraga14_3_extended_analysis`.
2. Use a copied `source_snapshot/` of the frozen 14.3 baseline source for read-only reproducibility.
3. Run or load a research-local walk-forward cache for `base_universe_12`.
4. Evaluate extended multiplier robustness with 1D sweeps, sensitivity ranking, a focused 2D sweep on the two most sensitive axes, and controlled extremes.
5. Run lightweight universe robustness: official candidate plus two controlled extremes.
6. Build Parquet audit cubes for representative candidates only.
7. Serve materialized outputs through FastAPI.
8. Explore results through a minimal React/TypeScript/Tailwind frontend.

## Run Analysis

```powershell
cd D:\QuantMahoraga
python .\research\mahoraga14_3_extended_analysis\run_extended_analysis.py
```

Force recomputation:

```powershell
python .\research\mahoraga14_3_extended_analysis\run_extended_analysis.py --force
```

Skip alternate universes:

```powershell
python .\research\mahoraga14_3_extended_analysis\run_extended_analysis.py --skip-universes
```

Limit uncached alternate-universe walk-forwards:

```powershell
python .\research\mahoraga14_3_extended_analysis\run_extended_analysis.py --max-new-universe-runs 0
```

This leaves coverage audits in place and marks remaining uncached universes as `ABORTED_COMPUTE_BUDGET`.

## Run API

```powershell
cd D:\QuantMahoraga
python .\research\mahoraga14_3_extended_analysis\run_api.py
```

API URL: `http://127.0.0.1:8000` by default. If that port is already occupied, `run_api.py` chooses the next free port in `8000-8019` and prints it.

## Run Frontend

```powershell
cd D:\QuantMahoraga\research\mahoraga14_3_extended_analysis\frontend
npm install
npm run dev
```

Frontend URL: `http://127.0.0.1:5173`

If the API did not start on port `8000`, set `VITE_API_BASE` before `npm run dev`.

## Main Outputs

- `outputs/extended_multiplier_robustness/extended_multiplier_summary.csv`
- `outputs/extended_multiplier_robustness/plateau_radius_report.md`
- `outputs/universe_robustness/universe_coverage_audit.csv`
- `outputs/universe_robustness/universe_robustness_summary.csv`
- `outputs/audit_cube/*.parquet`
- `outputs/audit_cube/cube_dictionary.md`
- `outputs/audit_cube/cube_lineage.md`
- `outputs/reports/final_extended_analysis_report.md`
- `outputs/reports/implementation_report.md`

## Methodological Limits

The extended multiplier phase samples a disciplined region; it does not prove global optimality. Alternate universe tests are portability stresses and can reflect composition, data coverage and canonical universe mechanics. The granular audit cube uses exposed snapshot traces; nullable fields are kept explicit rather than imputed.
