# Mahoraga 8.2 HM

Mahoraga 8.2 HM = **Hawkes + Markov regime fusion**.

## Core principle
- Keep 6.1-like base selection frozen.
- Use Hawkes for **transition urgency / fast contagion**.
- Use Markov-lite filtering for **persistent regime state**.
- Adapt only:
  - total risk budget
  - max exposure
  - vol target multiplier

## Required local files
This folder must already contain:
- `mahoraga6_1.py`
- `mahoraga7_1.py`

## Modes
- `SMOKE`
- `FAST`
- `FULL`

## Run
```bash
python mahoraga8_runner.py
```


## Optimization notes
- The expensive fold-invariant core is cached once per fold.
- Calibration is split into Stage 2A (regime fusion) and Stage 2B (cheap policy sweep).
- Candidate scoring during calibration uses a fast summary path.
