# Acceptance Robustness Notes

- worst scenario Sharpe delta: -0.1832
- worst scenario CAGR delta: -4.9724 pts
- bootstrap DeltaSharpe p25/p50/p75: 0.0745 / 0.1313 / 0.2010
- bootstrap DeltaCAGR p25/p50/p75: 0.0114 / 0.0271 / 0.0453
- interpretation: acceptance requires degradation bands to stay inside a reasonable range, not just a good base-case backtest.