# Mahoraga 11

Mahoraga 11 is a **baseline-first, path-structure-aware** redesign of the line.

## Design

The package now has a strict hierarchy:

1. **Base Alpha Engine**
   - builds the improved baseline that governs by default.
2. **Path Structure Features**
   - describes drawdown shape, rebound quality, stop pressure, chop, correlation persistence and market context.
3. **Exceptional Overrides Only**
   - `STRUCTURAL_DEFENSE`
   - `TRANSITION_RECOVERY`
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

- `mahoraga11_config.py`: configuration and compact/wide grids.
- `mahoraga11_data.py`: OHLCV and PIT universe loading.
- `mahoraga11_universe.py`: universe helpers.
- `base_alpha_engine.py`: improved alpha baseline and cached engine paths.
- `path_structure_features.py`: daily/weekly path descriptors.
- `structural_defense_model.py`: structural override model.
- `transition_recovery_model.py`: fast transition/recovery model plus Hawkes transition signal.
- `override_policy.py`: baseline-first override state machine.
- `backtest_executor.py`: walk-forward execution, calibration and stitching.
- `fast_report.py`: compact FAST artifacts.
- `full_report.py`: audit-heavy FULL artifacts.
- `mahoraga11_backtest.py`: compatibility wrapper to the executor.
- `mahoraga11_reporting.py`: compatibility wrapper to FAST/FULL reporting.
- `mahoraga11_runner.py`: entrypoint.

## Run

```bash
python mahoraga11_runner.py
```

## Notes

- `mahoraga6_1.py` is bundled locally so the package remains self-contained.
- Outputs are generated only at runtime.
- Expensive alpha path work is cached once per unique engine candidate.
- Policy sweeps reuse candidate-specific probabilities and path features instead of recomputing the full alpha path.
