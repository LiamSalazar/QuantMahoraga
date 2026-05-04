# Promotion Gate Decision

## Decision: PROMOTE_TO_OFFICIAL_BASELINE

- selected gate candidate: `B1.05_C1.10_L1.10_R1.05` (ROBUST_MAIN)
- delta CAGR vs control: 7.8676 pts
- delta Sharpe vs control: 0.2889
- delta Sortino vs control: 0.5216
- delta MaxDD vs control: 4.0355 pts
- priority windows: 2017_2018=PASS, 2020_2021=PASS, 2023_2024=PASS
- worst fold Sharpe delta vs control: 0.1194
- severe fold damage count: 0

## Gate scoreboard
```
            CandidateId            GateRole  DeltaCAGR_vs_Control  DeltaSharpe_vs_Control  DeltaSortino_vs_Control  DeltaMaxDD_vs_Control  DeltaAlphaNW_QQQ_vs_Control  DeltaAlphaNW_SPY_vs_Control Window2017_2018 Window2020_2021 Window2023_2024  WorstFoldSharpeDelta  WorstFoldCAGRDelta  SevereFoldDamageCount  EligibleForPromotion  PromotionScore
B1.05_C1.10_L1.10_R1.05         ROBUST_MAIN              7.867569                0.288873                 0.521626               4.035467                     0.065308                     0.067572            PASS            PASS            PASS              0.119377            1.739754                      0                     1        3.973398
B1.05_C1.10_L1.10_R1.00    NEIGHBOR_BACKOFF              7.686254                0.274970                 0.494320               3.868420                     0.063104                     0.065526            PASS            PASS            PASS              0.102862            1.485437                      0                     1        3.848655
B1.05_C1.00_L1.10_R1.05 NEIGHBOR_CONVICTION              7.179648                0.269044                 0.482298               4.042599                     0.059239                     0.061180            PASS            PASS            PASS              0.096300            1.354543                      0                     1        3.680504
B1.05_C1.10_L1.00_R1.05     NEIGHBOR_LEADER              6.707316                0.253739                 0.452063               3.911832                     0.055088                     0.056840            PASS            PASS            PASS              0.083637            0.988374                      0                     1        3.461162
B1.00_C1.00_L1.00_R1.00        BASE_CURRENT              2.680545                0.131064                 0.214988               3.624696                     0.020009                     0.019847            FAIL            PASS            PASS             -0.043692           -3.394005                      0                     0        1.101508
```

- institutional conclusion: the selected 14.3R candidate is strong enough to replace the official long-only control.