# Mahoraga 14_3R

Mahoraga 14_3R is an **acceptance and robustness phase** over the frozen `Mahoraga14_3_LONG_PARTICIPATION` architecture.

## Scope

- No new thesis.
- No new layers.
- No short logic.
- No free rewrite.
- Only parameter freeze, local stability, robustness and model-selection discipline.

## Frozen architecture under review

1. `BASE_ALPHA_V2`
2. `PARTICIPATION_ALLOCATOR_V2`
3. `CONVICTION_AMPLIFIER_LAYER`
4. `LEADER_PARTICIPATION_LAYER`
5. `RISK_BACKOFF_LAYER_V2`
6. continuation as a quality-filter diagnostic

## Acceptance-specific files

- `acceptance_suite_14_3R.py`: parameter inventory, local plateau search, leave-one-window-out, continuation acceptance, robustness, conservative model-selection guard and final decision outputs.
- `mahoraga14_3R_runner.py`: direct acceptance-suite entrypoint.
- `mahoraga14_config.py`: frozen 14_3 architecture plus 14_3R acceptance neighborhood settings.

## Core reused from 14_3

- `participation_allocator_v2.py`
- `conviction_amplifier_layer.py`
- `leader_participation_layer.py`
- `risk_backoff_layer_v2.py`
- `backtest_executor.py`
- `fast_fail_diagnostics_14_3.py`

## Run

```bash
python mahoraga14_3R_runner.py
```

## Acceptance outputs

Artifacts are written to `D:\QuantMahoraga\mahoraga14_3R_outputs`:

- `stitched_comparison_acceptance_14_3R.csv`
- `priority_window_acceptance_14_3R.csv`
- `local_stability_grid_14_3R.csv`
- `leave_one_window_out_14_3R.csv`
- `continuation_acceptance_14_3R.csv`
- `acceptance_robustness_suite_14_3R.csv`
- `acceptance_bootstrap_summary_14_3R.csv`
- `model_selection_guard_14_3R.md`
- `acceptance_decision_14_3R.md`
- `final_report_acceptance_14_3R.md`

## Decision rule

14_3R promotes `Mahoraga14_3_LONG_PARTICIPATION` only if the stitched edge remains material, the priority bull windows are acceptable, the local neighborhood is stable, the leave-one-window-out checks do not collapse, and the robustness / selection guard does not flag fragility.
