# Acceptance Decision

## Decision: KEEP_AS_EXPERIMENTAL_BRANCH

## Why
- stitched CAGR delta vs control: 2.6805
- stitched Sharpe delta vs control: 0.1311
- stitched Sortino delta vs control: 0.2150
- stitched MaxDD delta vs control: 3.6247
- base-current local robust flag: 1
- any priority window fail: True
- leave-one-window-out fail: False
- conservative model-selection guard p-value: 0.1700
- bootstrap DeltaSharpe p25: 0.0745

## Institutional conclusion
- Mahoraga14_3_LONG_PARTICIPATION is promising and robust enough to keep active, but it should NOT replace Mahoraga14_1_LONG_ONLY_CONTROL as the official baseline yet.