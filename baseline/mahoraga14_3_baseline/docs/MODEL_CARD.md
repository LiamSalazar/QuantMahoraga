# Model Card

- official model: `MAHORAGA14_3_BASELINE_OFFICIAL`
- frozen reference: `Mahoraga14_3R / ROBUST_MAIN / B1.05_C1.10_L1.10_R1.05`
- architecture: BASE_ALPHA_V2 + PARTICIPATION_ALLOCATOR_V2 + CONVICTION_AMPLIFIER_LAYER + LEADER_PARTICIPATION_LAYER + RISK_BACKOFF_LAYER_V2 + continuation quality filter
- stitched CAGR: 32.5518%
- stitched Sharpe: 1.4826
- stitched Sortino: 2.5280
- stitched MaxDD: -16.1997%
- intended use: institutional long-only benchmark replacement and future long-side research starting point.
- excluded uses: short-side deployment, new-thesis discovery, or uncontrolled parameter search inside the official baseline package.