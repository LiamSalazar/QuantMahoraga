# MaxDD audit

Method:
1. Rebuild stitched equity from stitched net returns only.
2. Compute drawdown as `equity / equity.cummax() - 1`.
3. Confirm stored stitched equity matches rebuilt equity within tolerance.

- LEGACY: stored=-21.11% rebuilt=-21.11% peak=2018-09-04 00:00:00 trough=2019-10-02 00:00:00 diff_bps=0.0000
- QQQ: stored=-35.24% rebuilt=-35.24% peak=2021-11-19 00:00:00 trough=2022-11-03 00:00:00 diff_bps=0.0000
- SPY: stored=-33.72% rebuilt=-33.72% peak=2020-02-19 00:00:00 trough=2020-03-23 00:00:00 diff_bps=0.0000
- BASE_ALPHA_V2: stored=-20.24% rebuilt=-20.24% peak=2020-09-01 00:00:00 trough=2020-11-10 00:00:00 diff_bps=0.0000
- BASE_ALPHA_V2 + CONTINUATION_PRESSURE_V2_ONLY: stored=-20.24% rebuilt=-20.24% peak=2020-09-01 00:00:00 trough=2020-11-10 00:00:00 diff_bps=0.0000
- BASE_ALPHA_V2 + STRUCTURAL_DEFENSE_ONLY + CONTINUATION_PRESSURE_V2: stored=-20.27% rebuilt=-20.27% peak=2020-09-01 00:00:00 trough=2020-11-10 00:00:00 diff_bps=0.0000
- BASE_ALPHA_V2 + STRUCTURAL_DEFENSE_ONLY: stored=-20.27% rebuilt=-20.27% peak=2020-09-01 00:00:00 trough=2020-11-10 00:00:00 diff_bps=0.0000