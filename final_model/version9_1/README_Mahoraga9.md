# Mahoraga 9.1

Mahoraga 9.1 is a controlled rewrite of the previous V9 attempt.

## Research objective
- Keep folds 3 and 5 acceptable and stable. They define the floor.
- Preserve and, when justified, improve folds 1, 2 and 4. They define the ceiling.
- Avoid permanent intervention.
- Report p-value and q-value in FAST and FULL mode.

## Key architectural changes vs the previous V9
- FAST now evaluates all 5 folds.
- The alpha core is no longer a pure 6.1 overlay. A blended raw/residual score is used.
- Hawkes remains only as a fast transition feature.
- The adaptive layer can stay neutral; it is not forced to intervene.
- Selection is scored with floor/ceiling aware utility and multiple-testing control.

## Runtime modes
- FAST
  - all 5 folds
  - compact candidate grid
  - compact reports
  - intended to validate or discard quickly
- FULL
  - all 5 folds
  - wider candidate grid
  - full audit artifacts

## Package layout
- `mahoraga9_config.py`: configuration and grids
- `mahoraga9_utils.py`: metrics, significance, BHY q-values, file helpers
- `mahoraga9_data.py`: market data wrappers
- `mahoraga9_universe.py`: point-in-time universe wrappers
- `mahoraga9_alpha.py`: raw and residual alpha scores
- `mahoraga9_features.py`: daily/weekly market context features
- `mahoraga9_hawkes.py`: transition urgency features
- `mahoraga9_meta_labels.py`: forward labels for fragility and recovery
- `mahoraga9_meta_models.py`: logistic and random-forest challengers
- `mahoraga9_risk.py`: sparse adaptive policy mapping
- `mahoraga9_calibration.py`: candidate generation, ranking and utility
- `mahoraga9_backtest.py`: custom fold execution and stitched OOS paths
- `mahoraga9_reporting.py`: text/CSV outputs
- `mahoraga9_runner.py`: end-to-end entry point

## Notes
- Outputs are created at runtime and are not part of the source package.
- The package ships with the legacy `mahoraga6_1.py` only as a dependency source for the canonical universe, statistical helpers and the legacy baseline.
