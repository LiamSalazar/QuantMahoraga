# Mahoraga 8

Mahoraga 8 is an integrated regime-aware core.

Unlike the 7.x family, it does **not** treat Mahoraga 6.1 as an untouchable
frozen execution core with an overlay layered on top.

## Required local files
This folder must already contain:
- `mahoraga6_1.py`
- `mahoraga7_1.py`

## New files
- `mahoraga8_config.py`
- `mahoraga8_regime.py`
- `mahoraga8_policy.py`
- `mahoraga8_core.py`
- `mahoraga8_calibration.py`
- `mahoraga8_backtest.py`
- `mahoraga8_reporting.py`
- `mahoraga8_runner.py`

## Modes
- `SMOKE`: quickest sanity check
- `FAST`: useful research pass
- `FULL`: all folds and full search

## Run
```bash
python mahoraga8_runner.py
```
