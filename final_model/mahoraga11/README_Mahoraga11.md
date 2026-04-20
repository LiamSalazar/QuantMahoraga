# Mahoraga 11

Mahoraga 11 is an **alpha-first, multi-engine** redesign of the Mahoraga line.

## Design

Mahoraga 11 separates the system into four layers:

1. **Ceiling Engine**
   - preserves the directional alpha that historically worked best in folds 1, 2, and 4.
2. **Floor Engine**
   - emphasizes residual / idiosyncratic alpha and stronger beta control for fold-3-like stress.
3. **Transition Engine**
   - uses Hawkes-style transition urgency and dual ML fragility/recovery probabilities.
4. **Risk Executor**
   - applies bounded scaling through exposure caps, vol-target multipliers, and execution costs.

## Core principles

- FAST and FULL both evaluate all 5 folds.
- FAST is computationally lighter, not methodologically incomplete.
- No frozen baseline overlay design: the alpha core itself is modified.
- Hawkes is used as a **transition signal**, not as a global regime oracle.
- p-values and BHY q-values are reported in both modes.
- Expensive alpha path computation is separated from the cheaper router sweep.

## Files

- `mahoraga11_config.py`: configuration and grids.
- `mahoraga11_data.py`: OHLCV / PIT universe loading.
- `mahoraga11_universe.py`: universe helpers.
- `mahoraga11_alpha.py`: raw/residual alpha engines and cached 1x paths.
- `mahoraga11_hawkes.py`: Hawkes-style transition features.
- `mahoraga11_router.py`: weekly features, dual ML models, router policy.
- `mahoraga11_backtest.py`: walk-forward, calibration, stitching.
- `mahoraga11_reporting.py`: FAST/FULL outputs.
- `mahoraga11_runner.py`: entrypoint.

## Run

```bash
python mahoraga11_runner.py
```

## Notes

- `mahoraga6_1.py` is bundled locally so the package is self-contained.
- Outputs are generated at runtime only.
- The code is structured to minimize recomputation:
  - fold invariants are computed once per fold;
  - alpha engine paths are cached per unique engine candidate;
  - router sweeps reuse ML probabilities and Hawkes features.
