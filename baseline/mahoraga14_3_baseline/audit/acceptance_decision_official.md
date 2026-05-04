# Acceptance Decision Official

- official baseline: `MAHORAGA14_3_BASELINE_OFFICIAL`
- frozen candidate id: `B1.05_C1.10_L1.10_R1.05`
- replaces baseline: `Mahoraga14_1_LONG_ONLY_CONTROL`
- promotion decision carried forward from promotion gate: `ROBUST_MAIN`
- priority windows status: 2017_2018=PASS, 2020_2021=PASS, 2023_2024=PASS

## Priority detail
```
            CandidateId    GateRole    Window Source      Start        End  CandidateReturn  ControlReturn  QQQReturn  SPYReturn  DeltaReturn_vs_Control  DeltaReturn_vs_QQQ  DeltaReturn_vs_SPY  SharpeLocal  SortinoLocal  MaxDDLocal  BetaQQQLocal  BetaSPYLocal  UpsideCaptureQQQLocal  ExposureLocal                        Variant GateStatus                                             GateReason
B1.05_C1.10_L1.10_R1.05 ROBUST_MAIN 2017_2018 MANUAL 2017-01-03 2018-12-31         0.292992       0.221252   0.319696   0.161447                0.071740           -0.026704            0.131545     0.829399      1.254508   -0.137116      0.493916      0.543326               0.658609       0.644500 MAHORAGA14_3_BASELINE_OFFICIAL       PASS beats_or_matches_control_with_acceptable_local_quality
B1.05_C1.10_L1.10_R1.05 ROBUST_MAIN 2020_2021 MANUAL 2020-04-01 2021-12-31         0.907443       0.688098   1.099695   0.891030                0.219345           -0.192252            0.016413     1.635316      2.712046   -0.161997      0.786578      0.703480               0.892549       0.775296 MAHORAGA14_3_BASELINE_OFFICIAL       PASS                       maintains_improvement_vs_control
B1.05_C1.10_L1.10_R1.05 ROBUST_MAIN 2023_2024 MANUAL 2023-01-01 2024-12-31         2.071707       1.747975   0.936927   0.575765                0.323732            1.134780            1.495942     2.469960      5.093577   -0.144512      0.959330      1.153424               1.077780       0.736797 MAHORAGA14_3_BASELINE_OFFICIAL       PASS                             keeps_recent_bull_strength
```