# Mahoraga 13

Mahoraga 13 is a **strict consolidation** on top of Mahoraga 12.

## Design

The package now has a strict hierarchy:

1. **Base Alpha Engine**
   - builds the improved baseline that governs by default.
2. **Path Structure Features**
   - describes drawdown shape, rebound quality, stop pressure, chop, correlation persistence and market context.
3. **Exceptional Overrides Only**
   - `STRUCTURAL_DEFENSE_ONLY` in the main branch
   - `CONTINUATION_V2_ONLY` and `STRUCTURAL_DEFENSE + CONTINUATION_V2` only as experimental branches
   - continuation uses a smooth `CONTINUATION_LIFT` driven by path-state / continuation-pressure after valid compression / pause conditions
4. **Backtest Executor**
   - calibrates compact FAST grids and wider FULL grids without nested parallelism.

## Core principles

- The improved baseline is the default state.
- Overrides are exceptional and auditable.
- No Markov layer.
- Hawkes is used only as a fast transition signal.
- FAST and FULL both evaluate folds 1 to 5.
- FULL reports p-values, BHY q-values and candidate audit artifacts.
- Invariants are separated into:
  - global
  - per fold
  - per candidate

## Files

- `mahoraga13_config.py`: configuration and compact/wide grids.
- `mahoraga13_data.py`: OHLCV and PIT universe loading.
- `mahoraga13_universe.py`: universe helpers.
- `base_alpha_engine.py`: improved alpha baseline and cached engine paths.
- `path_structure_features.py`: daily/weekly path descriptors.
- `structural_defense_model.py`: structural override model.
- `transition_recovery_model.py`: Hawkes transition signal and historical transition/recovery references.
- `continuation_v2_model.py`: isolated continuation-after-compression label and model selection.
- `override_policy.py`: baseline-first override state machine with structural priority and auditable continuation pressure.
- `backtest_executor.py`: walk-forward execution, calibration and stitching.
- `fast_report.py`: compact FAST artifacts.
- `full_report.py`: audit-heavy FULL artifacts.
- `mahoraga13_backtest.py`: compatibility wrapper to the executor.
- `mahoraga13_reporting.py`: compatibility wrapper to FAST/FULL reporting.
- `mahoraga13_runner.py`: entrypoint.

## Run

```bash
python mahoraga13_runner.py
```

## Notes

- `mahoraga6_1.py` is bundled locally so the package remains self-contained.
- Outputs are generated only at runtime.
- Expensive alpha path work is cached once per unique engine candidate.
- Policy sweeps reuse candidate-specific probabilities and path features instead of recomputing the full alpha path.
- FAST only fits continuation labels when the continuation branch is actually needed, so main-branch calibration stays compact.


