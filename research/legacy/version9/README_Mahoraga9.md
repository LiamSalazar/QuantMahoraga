# Mahoraga 9

Mahoraga 9 is a full rebuild from the 6.1 research line.

## Design goals
- Re-run the complete workflow from scratch.
- Improve fragile folds without turning the engine into a permanently defensive system.
- Preserve walk-forward discipline and point-in-time universe construction.
- Keep the codebase fast, auditable, and easy to extend.

## Core ideas
- New execution stack over the 6.1 legacy backtest engine.
- No Markov execution layer.
- Hawkes-like transition urgency kept as a fast signal.
- Dual adaptive controllers:
  - fragility controller
  - recovery controller
- Risk adapts through external scaling and policy mapping.
- Alpha adaptation hooks are prepared through `alpha_mix`, even if the first executable path focuses on the adaptive risk gate.

## Runtime modes
- `FAST`
  - Focus folds: 3 and 5
  - Compact grids
  - Minimal outputs required to validate or discard quickly
- `FULL`
  - All folds
  - Full audit outputs
  - Larger calibration sweep

## Package layout
- `mahoraga9_config.py`: configuration and grids
- `mahoraga9_data.py`: market data loading wrappers
- `mahoraga9_universe.py`: point-in-time universe wrappers
- `mahoraga9_features.py`: market and portfolio context features
- `mahoraga9_alpha.py`: alpha configuration helpers
- `mahoraga9_hawkes.py`: stress / recovery urgency features
- `mahoraga9_meta_labels.py`: forward-looking economic labels
- `mahoraga9_meta_models.py`: logistic / random-forest meta-models
- `mahoraga9_risk.py`: policy mapping and daily gate generation
- `mahoraga9_calibration.py`: cheap candidate sweep on validation
- `mahoraga9_backtest.py`: fold execution logic
- `mahoraga9_reporting.py`: CSV and text outputs
- `mahoraga9_runner.py`: end-to-end entry point

## Outputs
Outputs are generated at runtime and are not part of the source package.
The default directories are:
- `version9_outputs/`
- `version9_plots/`

## Notes
This initial V9 package is executable and structured for research iteration.
The current executable path already supports:
- full data/universe pipeline
- adaptive fragility/recovery gating
- FAST/FULL modes
- fold-level calibration and reporting

The next natural extension is a tighter execution path for `alpha_mix`, where
weights or score families are blended directly inside the portfolio engine.
