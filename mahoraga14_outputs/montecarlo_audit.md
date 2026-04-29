# Monte Carlo audit

Methods:
- stationary_block_bootstrap: stationary bootstrap on stitched TEST_ONLY daily returns.
- friction_multiplier_mc: multiplies realized stitched transaction-cost series.
- local_param_neighborhood: local perturbation around the final continuation candidate.

Probability thresholds:
- CAGR < 0
- Sharpe < 1
- MaxDD < -25%