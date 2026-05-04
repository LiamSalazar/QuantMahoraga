# Mahoraga 14_3

Mahoraga 14_3 is a **separate long-only thesis** focused on bull participation, conviction translation and leader participation.

## Architecture

1. `BASE_ALPHA_V2`
   - frozen 14.1 long-only baseline and internal control anchor.
2. `PARTICIPATION_ALLOCATOR_V2`
   - weekly allocator that scores continuation, persistence, benchmark strength, breadth, realized volatility, cash drag pressure and leader opportunity.
3. `CONVICTION_AMPLIFIER_LAYER`
   - converts healthy bull conviction into meaningful lifts to long budget, gate, caps and effective weight scale.
4. `LEADER_PARTICIPATION_LAYER`
   - blends the baseline long book with a leader-participation engine, tilts toward conditional leaders and redeploys cash drag without adding shorts or hedges.
5. `RISK_BACKOFF_LAYER_V2`
   - cuts budget / conviction / leader participation back when fragility, break-risk or benchmark weakness rise.
6. `FAST_FAIL_DIAGNOSTICS_14_3`
   - compares primarily against `QQQ` and `SPY`, plus `Mahoraga 14.1` as control, and writes explicit fail-fast artifacts.

## Files

- `mahoraga14_config.py`: 14_3 configuration, labels, outputs and participation/backoff parameters.
- `path_structure_features.py`: weekly market/book context used by continuation and allocator.
- `participation_allocator_v2.py`: stronger allocator with explicit cash-drag and leader-opportunity scores.
- `conviction_amplifier_layer.py`: explicit conviction translation layer.
- `leader_participation_layer.py`: leader blend + conviction weight scaling + cash-drag redeployment layer.
- `risk_backoff_layer_v2.py`: explicit backoff logic.
- `backtest_executor.py`: fold execution and integration of the new primary thesis.
- `fast_fail_diagnostics_14_3.py`: FAST outputs, scorecards, continuation diagnostic, active-return and fail-fast report.
- `mahoraga14_runner.py`: main runner exposing `run_mahoraga14_3`.
- `mahoraga14_3_runner.py`: direct FAST entrypoint.

## Run

```bash
python mahoraga14_3_runner.py
```

## Outputs

FAST writes to `D:\QuantMahoraga\mahoraga14_3_outputs` and includes:

- `stitched_comparison_fast_14_3.csv`
- `bull_window_scorecard_fast_14_3.csv`
- `active_return_vs_qqq_fast_14_3.csv`
- `active_return_vs_qqq_curve_14_3.csv`
- `upside_participation_decomposition_fast_14_3.csv`
- `allocator_cash_drag_fast_14_3.csv`
- `leader_miss_analysis_fast_14_3.csv`
- `continuation_diagnostic_fast_14_3.csv`
- `candidate_audit_fast_14_3.csv`
- `final_report_fast_14_3.md`

## Current status

The current 14_3 thesis is implemented and auditable; inspect the FAST report before promoting or merging.
