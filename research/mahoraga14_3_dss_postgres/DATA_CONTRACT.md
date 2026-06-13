# Data Contract

## Inputs

Required read-only sources:

| Source | Role |
|---|---|
| `baseline/mahoraga14_3_baseline/outputs/active_return_vs_qqq_official.csv` | Official daily equity, benchmark return, active return, and drawdown replay base. |
| `baseline/mahoraga14_3_baseline/outputs/fold_summary_official.csv` | Official fold intervals and benchmark/control metrics. |
| `baseline/mahoraga14_3_baseline/outputs/stitched_comparison_official.csv` | Official scorecard context. |
| `baseline/mahoraga14_3_baseline/audit/allocator_cash_drag_official.csv` | Official allocator and participation state evidence. |
| `research/mahoraga14_3_extended_analysis/outputs/audit_cube/*.parquet` | Decision, position, module trace, outcome, and market context cubes. |
| `research/mahoraga14_3_extended_analysis/outputs/extended_multiplier_robustness/*.csv` | Candidate robustness and fold summaries. |
| `research/mahoraga14_3_extended_analysis/outputs/universe_robustness/*.csv` | Universe stress summaries and coverage. |

## Missing Artifacts

The artifact inventory records missing files with `exists_flag=false`. Required missing artifacts fail validation. Optional artifacts are documented but do not block the dev mode.

No missing source is backfilled with false official results. Synthetic rows are limited to `fact_whatif`, tagged `demo_mode=true`, and marked with `source_artifact='demo_synthetic_whatif_grid'`.

## Dimensions

- `dim_date`
- `dim_asset`
- `dim_candidate`
- `dim_universe`
- `dim_fold`
- `dim_module`
- `dim_regime`
- `dim_horizon`
- `dim_scenario`
- `dim_metric`

## Facts

- `fact_market_bar`
- `fact_signal_daily`
- `fact_decision_state`
- `fact_position_daily`
- `fact_module_trace`
- `fact_outcome`
- `fact_candidate_metric`
- `fact_robustness_surface`
- `fact_cost_sensitivity`
- `fact_universe_sensitivity`
- `fact_whatif`
- `fact_path_recursive`
- `fact_data_quality`

## Critical Grains

| Fact | Grain |
|---|---|
| `fact_decision_state` | date x candidate x fold x universe x run |
| `fact_position_daily` | date x candidate x fold x universe x ticker x run |
| `fact_module_trace` | date x candidate x fold x universe x module x run |
| `fact_outcome` | decision date x candidate x fold x universe x horizon x run |
| `fact_whatif` | scenario x candidate x fold x universe x horizon x cost/slippage x run |
| `fact_path_recursive` | candidate x date x run |

## Validations

- Required Parquet tables exist and have rows.
- Duplicate keys are checked for decision, position, and outcome grains.
- Demo what-if rows must be explicitly flagged.
- Validation outputs are written to `outputs/reports/validation_report.json` and `fact_data_quality`.

