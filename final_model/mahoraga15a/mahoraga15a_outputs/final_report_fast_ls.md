# Mahoraga15A FAST

## 1. Stitched comparison
                Variant  CAGR%  Sharpe  Sortino  MaxDD%  AvgExposure  GrossLong  GrossShort  NetExposure  AvgTurnover  ReturnPerExposure  BetaQQQ  BetaSPY  AlphaNW_QQQ  AlphaNW_SPY  UpsideCaptureQQQ  DownsideCaptureQQQ
                 LEGACY  23.07  1.1348   1.9157  -21.11       0.6069     0.6069         0.0       0.6069       0.0331           0.001489   0.4606   0.4375     0.140014     0.172927            0.6804              0.6060
                    QQQ  20.14  0.9178   1.4556  -35.24       1.0000     1.0000         0.0       1.0000       0.0000           0.000832   1.0000   1.1604     0.000000     0.029430            1.0000              1.0000
                    SPY  14.88  0.8463   1.3087  -33.72       1.0000     1.0000         0.0       1.0000       0.0000           0.000618   0.7525   1.0000    -0.002081     0.000000            0.7034              0.6962
MAHORAGA_14_1_LONG_ONLY  24.68  1.1937   2.0063  -20.24       0.6316     0.6316         0.0       0.6316       0.0345           0.001514   0.4853   0.4683     0.149353     0.182964            0.7078              0.6268
        MAHORAGA_15A_LS  21.84  1.1977   2.0105  -18.26       0.5560     0.5560         0.0       0.5560       0.0305           0.001523   0.4243   0.4087     0.132438     0.161470            0.6224              0.5499

## 2. Hedge contribution vs frozen long book
 Sharpe  Sortino  MaxDD%  BetaQQQ  BetaSPY
  0.004   0.0042    1.98   -0.061  -0.0596

## 3. Pairwise p-values / q-values
                 Target               Reference                                 Comparison  p_value  q_value
MAHORAGA_14_1_LONG_ONLY                  LEGACY          MAHORAGA_14_1_LONG_ONLY_vs_LEGACY 0.255017      1.0
MAHORAGA_14_1_LONG_ONLY                     QQQ             MAHORAGA_14_1_LONG_ONLY_vs_QQQ 0.323094      1.0
MAHORAGA_14_1_LONG_ONLY                     SPY             MAHORAGA_14_1_LONG_ONLY_vs_SPY 0.107110      1.0
        MAHORAGA_15A_LS MAHORAGA_14_1_LONG_ONLY MAHORAGA_15A_LS_vs_MAHORAGA_14_1_LONG_ONLY 0.999689      1.0
        MAHORAGA_15A_LS                  LEGACY                  MAHORAGA_15A_LS_vs_LEGACY 0.758495      1.0
        MAHORAGA_15A_LS                     QQQ                     MAHORAGA_15A_LS_vs_QQQ 0.477819      1.0
        MAHORAGA_15A_LS                     SPY                     MAHORAGA_15A_LS_vs_SPY 0.185821      1.0

## 4. Stress suite
                Variant                      Scenario                                     ScenarioNote  CAGR%  Sharpe  Sortino  MaxDD%  BetaQQQ  BetaSPY  AlphaNW_QQQ  AlphaNW_SPY  DeltaCAGR_vs_LSBase%  DeltaSharpe_vs_LSBase  DeltaSortino_vs_LSBase  DeltaMaxDD_vs_LSBase%  DeltaCAGR_vs_LongOnly%  DeltaSharpe_vs_LongOnly  DeltaSortino_vs_LongOnly  DeltaMaxDD_vs_LongOnly%
MAHORAGA_14_1_LONG_ONLY            BASELINE_LONG_ONLY         official frozen 14.1 long-only reference  24.68  1.1937   2.0063  -20.24   0.4853   0.4683     0.149353     0.182964                  2.85                -0.0040                 -0.0042                  -1.97                    0.00                   0.0000                    0.0000                     0.00
        MAHORAGA_15A_LS                  BASELINE_15A              Mahoraga15A unstressed stitched OOS  21.84  1.1977   2.0105  -18.26   0.4243   0.4087     0.132438     0.161470                  0.00                 0.0000                  0.0000                   0.00                   -2.85                   0.0040                    0.0042                     1.97
        MAHORAGA_15A_LS                  COST_PLUS_25                        commission/slippage x1.25  21.53  1.1838   1.9861  -18.29   0.4243   0.4087     0.129617     0.158577                 -0.30                -0.0140                 -0.0244                  -0.03                   -3.15                  -0.0100                   -0.0203                     1.94
        MAHORAGA_15A_LS                  COST_PLUS_50                        commission/slippage x1.50  21.23  1.1698   1.9617  -18.32   0.4242   0.4087     0.126802     0.155692                 -0.61                -0.0280                 -0.0488                  -0.06                   -3.46                  -0.0239                   -0.0447                     1.91
        MAHORAGA_15A_LS                 COST_PLUS_100                        commission/slippage x2.00  20.62  1.1418   1.9129  -18.38   0.4242   0.4086     0.121194     0.149941                 -1.21                -0.0560                 -0.0976                  -0.12                   -4.06                  -0.0520                   -0.0935                     1.85
        MAHORAGA_15A_LS                EXTRA_SLIPPAGE                             extra slippage +5bps  21.37  1.1762   1.9729  -18.31   0.4243   0.4087     0.128100     0.157023                 -0.47                -0.0215                 -0.0376                  -0.05                   -3.32                  -0.0175                   -0.0334                     1.93
        MAHORAGA_15A_LS   EXECUTION_DELAY_1_REBALANCE delay long budget and hedge one weekly rebalance  21.57  1.1850   1.9841  -18.26   0.4253   0.4100     0.129775     0.158755                 -0.26                -0.0127                 -0.0264                   0.00                   -3.11                  -0.0087                   -0.0222                     1.98
        MAHORAGA_15A_LS    HEDGE_RATIO_UNDERESTIMATED                   systematic hedge scaled to 75%  21.84  1.1977   2.0105  -18.26   0.4243   0.4087     0.132438     0.161470                  0.00                 0.0000                  0.0000                   0.00                   -2.85                   0.0040                    0.0042                     1.97
        MAHORAGA_15A_LS     HEDGE_RATIO_OVERESTIMATED                  systematic hedge scaled to 125%  21.84  1.1977   2.0105  -18.26   0.4243   0.4087     0.132438     0.161470                  0.00                 0.0000                  0.0000                   0.00                   -2.85                   0.0040                    0.0042                     1.97
        MAHORAGA_15A_LS               REACTION_SLOWER                          allocator reacts slower  21.84  1.1977   2.0105  -18.26   0.4243   0.4087     0.132438     0.161470                  0.00                 0.0000                  0.0000                   0.00                   -2.85                   0.0040                    0.0042                     1.97
        MAHORAGA_15A_LS               REACTION_FASTER                          allocator reacts faster  21.84  1.1977   2.0105  -18.26   0.4243   0.4087     0.132438     0.161470                  0.00                 0.0000                  0.0000                   0.00                   -2.85                   0.0040                    0.0042                     1.97
        MAHORAGA_15A_LS ALLOCATOR_GATE_VOL_CAP_STRESS   tighter allocator caps and lighter long budget  20.73  1.1977   2.0105  -17.41   0.4031   0.3883     0.125419     0.152812                 -1.11                 0.0000                  0.0000                   0.85                   -3.96                   0.0040                    0.0042                     2.82
        MAHORAGA_15A_LS         EMPIRICAL_PATH_STRESS     inject worst empirical block into later path  20.18  1.1087   1.8247  -18.26   0.4259   0.4095     0.117149     0.146050                 -1.66                -0.0890                 -0.1858                   0.00                   -4.51                  -0.0850                   -0.1816                     1.97

## 5. Monte Carlo / bootstrap
                    Method  Samples  MeanCAGR%  P5_CAGR%  P25_CAGR%  P50_CAGR%  P75_CAGR%  P95_CAGR%  MeanSharpe  P5_Sharpe  P25_Sharpe  P50_Sharpe  P75_Sharpe  P95_Sharpe  MeanMaxDD%  P5_MaxDD%  P25_MaxDD%  P50_MaxDD%  P75_MaxDD%  P95_MaxDD%  Prob_Sharpe_lt_Baseline  Prob_MaxDD_worse_Baseline  Prob_CAGR_materially_worse_Baseline
stationary_block_bootstrap      250      21.45      9.48      16.11      21.26      26.25      34.19      1.1718     0.6203      0.9504      1.1801      1.3937      1.6946      -23.02     -33.13      -25.95      -22.32      -19.26      -16.15                    0.516                      0.688                                0.576
    friction_multiplier_mc      250      21.80     21.20      21.56      21.82      22.04      22.41      1.1962     1.1685      1.1853      1.1969      1.2069      1.2240      -18.26     -18.32      -18.29      -18.26      -18.24      -18.20                    0.392                      0.000                                1.000
  local_param_neighborhood       18      21.84     21.17      21.17      21.84      22.50      22.50      1.1977     1.1977      1.1977      1.1977      1.1977      1.1977      -18.26     -18.77      -18.77      -18.26      -17.75      -17.75                    0.000                      0.000                                1.000

## 6. Audit / success checks
                       Metric     Value  Threshold  Passed                                        Detail
                  SharpeDelta  0.004000       0.00    True                  Sharpe stitched must improve
                 SortinoDelta  0.004200       0.00    True                 Sortino stitched must improve
             BetaQQQReduction  0.061000       0.05    True              Beta vs QQQ must go down clearly
             BetaSPYReduction  0.059600       0.05    True              Beta vs SPY must go down clearly
                  MaxDDDelta%  1.980000       1.00   False MaxDD should improve or not worsen materially
                   CAGRDelta% -2.840000      -2.00   False                  CAGR should not be destroyed
            AlphaNW_QQQ_Delta -0.016915      -0.02    True  Alpha NW vs QQQ should not worsen materially
            WorstStressSharpe  1.108700       0.00    True                         EMPIRICAL_PATH_STRESS
HedgeImpactReal_AvgGrossShort  0.000000       0.03   False         short sleeve should not be decorative
   Prob_Sharpe_lt_Baseline_MC  0.000000       0.50    True                 local neighborhood robustness