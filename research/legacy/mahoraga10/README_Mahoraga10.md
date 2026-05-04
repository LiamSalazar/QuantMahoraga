# Mahoraga 10

Mahoraga 10 is an **alpha-first** research stack built on top of the stable
audit and walk-forward infrastructure from Mahoraga 6.1.

## Design goals
- Preserve the methodological floor established by 6.1.
- Raise the ceiling through a new alpha core instead of relying on a dominant overlay.
- Keep Hawkes as a transition-urgency layer, not as the primary alpha engine.
- Maintain full auditability, significance reporting, and reproducibility.
- Separate expensive computations from cheap policy sweeps.

## Core ideas
1. **New alpha core**
   - Raw directional alpha: trend, momentum, relative strength.
   - Residual alpha: residual trend and residual momentum after rolling beta neutralisation versus QQQ.
   - Controlled beta penalty to avoid simply owning the common tech factor.

2. **Minimal adaptive policy**
   - Fragility and recovery models operate on weekly context.
   - Policy adjusts only exposure, volatility target, and a small alpha tilt.
   - Intervention is optional, not permanent.

3. **Hawkes transition features**
   - Stress and recovery intensities are built from weekly market events.
   - Hawkes affects urgency, not the ranking engine.

4. **FAST and FULL**
   - Both modes evaluate the 5 folds.
   - FAST uses compact grids and a reduced audit set.
   - FULL runs exhaustive reporting and robustness analysis.

## Optimisation strategy
- Global market / universe invariants are computed once per run.
- Fold-local invariants are computed once per fold.
- Alpha candidates are precomputed once per fold.
- Policy candidates are evaluated cheaply on top of the precomputed alpha caches.
- Random forests use `n_jobs=1` when outer fold parallelism is active.

## Outputs
### FAST
- final_report_fast.txt
- walk_forward_folds_fast.csv
- candidate_leaderboard_fast.csv
- stitched_comparison_fast.csv
- alpha_nw_fast.csv
- sharpe_ci_fast.csv

### FULL
- final_report.txt
- comparison_full.csv
- comparison_oos.csv
- walk_forward_folds.csv
- candidate_leaderboard.csv
- alpha_nw.csv
- sharpe_ci.csv
- stress_oos.csv
- regime_oos.csv
- local_sensitivity.csv

## Execution
```bash
python mahoraga10_runner.py
```
