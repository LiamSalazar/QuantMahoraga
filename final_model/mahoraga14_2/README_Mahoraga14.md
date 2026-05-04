# Mahoraga 14_2

Mahoraga 14_2 is a **separate long-only thesis** focused on bull participation.

## Architecture

1. `BASE_ALPHA_V2`
   - frozen 14.1 long-only baseline and internal control anchor.
2. `PARTICIPATION_ALLOCATOR_V1`
   - weekly allocator that scores continuation, persistence, benchmark strength, breadth, fragility, break-risk, realized volatility and trend state.
3. `BULL_PARTICIPATION_LAYER`
   - blends the baseline long book with a leader-participation engine and redeploys cash drag toward a target long budget without adding shorts or hedges.
4. `RISK_BACKOFF_LAYER`
   - cuts long budget / blend / caps back when fragility, break-risk or benchmark weakness rise.
5. `FAST_FAIL_DIAGNOSTICS_14_2`
   - compares primarily against `QQQ` and `SPY`, plus `Mahoraga 14.1` as control, and writes explicit fail-fast artifacts.

## Files

- `mahoraga14_config.py`: 14_2 configuration, labels, outputs and participation/backoff parameters.
- `path_structure_features.py`: extended weekly context with QQQ/book slope, trend-state and realized-vol inputs for the allocator.
- `participation_allocator_v1.py`: bull participation allocator.
- `bull_participation_layer.py`: leader-blend + cash-drag redeployment layer.
- `risk_backoff_layer.py`: explicit backoff logic.
- `backtest_executor.py`: fold execution and integration of the new primary thesis.
- `fast_fail_diagnostics_14_2.py`: FAST outputs, scorecards, active-return and fail-fast report.
- `mahoraga14_runner.py`: main runner exposing `run_mahoraga14_2`.
- `mahoraga14_2_runner.py`: direct FAST entrypoint.

## Run

```bash
python mahoraga14_2_runner.py
```

## Outputs

FAST writes to `D:\QuantMahoraga\mahoraga14_2_outputs` and includes:

- `stitched_comparison_fast_14_2.csv`
- `bull_window_scorecard_fast.csv`
- `active_return_vs_qqq_fast.csv`
- `active_return_vs_qqq_curve.csv`
- `upside_participation_decomposition_fast.csv`
- `allocator_cash_drag_fast.csv`
- `leader_miss_analysis_fast.csv`
- `candidate_audit_fast_14_2.csv`
- `final_report_fast_14_2.md`

## Current status

The current 14_2 thesis is implemented and auditable, but the latest FAST run marks it `FAIL_FAST`; inspect the FAST report before promoting or merging.
