# Mahoraga 8.1 Lite

Mahoraga 8.1 Lite keeps **selection frozen like 6.1** and only adapts:
- max exposure
- vol target
- risk budget
- regime persistence / hysteresis

This is the disciplined follow-up to H8 after the first integrated-core version
proved too intrusive.

## Files you should replace in version8/
- mahoraga8_config.py
- mahoraga8_regime.py
- mahoraga8_policy.py
- mahoraga8_core.py
- mahoraga8_calibration.py
- mahoraga8_backtest.py
- mahoraga8_runner.py
- mahoraga8_reporting.py

## Required local files
This folder must also contain your local compatible copies of:
- mahoraga6_1.py
- mahoraga7_1.py

## Run
```bash
python mahoraga8_runner.py
```
