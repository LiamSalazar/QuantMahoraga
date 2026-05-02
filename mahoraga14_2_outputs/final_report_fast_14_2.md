# Mahoraga14_2 FAST

## Thesis status: FAIL_FAST

FAIL-FAST reasons:
- improvement_microscopic_without_real_participation_shift
- maxdd_materially_worse
- bull_windows_not_improved_vs_control

## Core comparison
```
                        Variant      CAGR   Sharpe  Sortino      MaxDD  AvgExposure  AvgTurnover  ReturnPerExposure  BetaQQQ  BetaSPY  UpsideCaptureQQQ  DownsideCaptureQQQ  UpsideCaptureSPY  DownsideCaptureSPY  AlphaNW_QQQ  AlphaNW_SPY  AlphaNW_QQQ_t  AlphaNW_SPY_t  AlphaNW_QQQ_p  AlphaNW_SPY_p  p_value  q_value  p_value_vs_SPY  q_value_vs_SPY  p_value_vs_Control  q_value_vs_Control
                            QQQ 20.141212 0.917784 1.455634 -35.241581     1.000000     0.000000           0.000832 1.000000 1.160418          1.000000            1.000000          1.237785            1.216410     0.000000     0.029430          7.547          1.052       0.000000       0.292680      NaN      NaN             NaN             NaN                 NaN                 NaN
                            SPY 14.879523 0.846307 1.308745 -33.717264     1.000000     0.000000           0.000618 0.752495 1.000000          0.703416            0.696234          1.000000            1.000000    -0.002081     0.000000         -0.088         -2.626       0.929869       0.008647      NaN      NaN             NaN             NaN                 NaN                 NaN
                  BASE_ALPHA_V2 24.516345 1.188725 1.996479 -20.235195     0.630308     0.034496           0.001508 0.484404 0.467471          0.706102            0.626041          0.818593            0.685596     0.147931     0.181438          2.473          2.701       0.013389       0.006913      NaN      NaN             NaN             NaN                 NaN                 NaN
 MAHORAGA14_1_LONG_ONLY_CONTROL 24.684280 1.193715 2.006328 -20.235195     0.631606     0.034535           0.001514 0.485257 0.468282          0.707750            0.626751          0.820529            0.686239     0.149353     0.182964          2.490          2.716       0.012777       0.006617 0.323094      1.0        0.107110        0.657441                 NaN                 NaN
MAHORAGA14_2_LONG_PARTICIPATION 24.190078 1.238955 2.075183 -25.478719     0.596399     0.044395           0.001561 0.458649 0.448055          0.672294            0.590376          0.774273            0.639056     0.148428     0.179145          2.673          2.887       0.007526       0.003886 0.353864      1.0        0.115172        0.657441            0.630481                 1.0
                         LEGACY 23.068204 1.134758 1.915726 -21.105963     0.606927     0.033096           0.001489 0.460640 0.437465          0.680356            0.605989          0.783143            0.657669     0.140014     0.172927          2.306          2.552       0.021132       0.010715      NaN      NaN             NaN             NaN                 NaN                 NaN
```

## Primary vs control deltas
- CAGR delta: -0.49 pts
- Sharpe delta: 0.0452
- Sortino delta: 0.0689
- MaxDD delta: -5.24 pts
- AvgExposure delta: -0.0352
- UpsideCaptureQQQ delta: -0.0355

## Bull windows
```
   Window Source      Start        End  Mahoraga14_2Return  Mahoraga14_1ControlReturn  QQQReturn  SPYReturn  DeltaReturn_vs_QQQ  DeltaReturn_vs_SPY  DeltaReturn_vs_Control  SharpeLocal  SortinoLocal  MaxDDLocal  BetaQQQLocal  BetaSPYLocal  UpsideCaptureQQQLocal  UpsideCaptureSPYLocal  ExposureLocal  ControlExposureLocal
2017_2018 MANUAL 2017-01-03 2018-12-31            0.202475                   0.221252   0.319696   0.161447           -0.117221            0.041028               -0.018777     0.649476      0.974332   -0.129081      0.455588      0.501892               0.592134               0.688979       0.591074              0.584997
2020_2021 MANUAL 2020-04-01 2021-12-31            0.574349                   0.688098   1.099695   0.891030           -0.525346           -0.316681               -0.113749     1.264464      2.041827   -0.254787      0.667871      0.592847               0.758190               0.755954       0.699792              0.742419
2023_2024 MANUAL 2023-01-01 2024-12-31            1.534001                   1.747975   0.936927   0.575765            0.597074            0.958236               -0.213973     2.171959      4.341571   -0.143182      0.910796      1.092369               0.986754               1.310157       0.689418              0.765954
```

## Upside participation decomposition
```
   Window      Start        End  WindowReturnPrimary1x  WindowReturnControl1x  WindowReturnQQQ  EstimatedCashDragContribution  EstimatedBetaResidualReliefContribution  EstimatedDefenseDragContribution  EstimatedAllocatorRestrictionContribution  EstimatedUnderExposureContribution  AvgLongBudget  AvgLeaderBlend  AvgGateAdj  AvgVolAdj  AvgExpCapAdj  AvgCashDragBefore  AvgCashDragAfter
2017_2018 2017-01-03 2018-12-31               0.423975               0.427265         0.319696                      -0.007600                                -0.003093                         -0.000632                                  -0.000504                           -0.090044       0.828071        0.130859    1.066434   1.051117      1.070306           0.129134          0.172826
2020_2021 2020-04-01 2021-12-31               0.784307               0.864061         0.922807                       0.019969                                 0.003328                         -0.002096                                  -0.006733                           -0.217939       1.710880        0.373706    2.201558   2.135476      2.212899           0.262034          0.374282
2023_2024 2023-01-01 2024-12-31               1.663144               1.969170         0.880782                       0.021212                                -0.036590                          0.004125                                  -0.004503                           -0.120429       1.618912        0.278742    2.140650   2.102060      2.138959           0.166450          0.381088
```

## Stress suite
```
                        Variant                    Scenario                                   ScenarioNote  BaseCAGR%  StressCAGR%  DeltaCAGR%  BaseSharpe  StressSharpe  DeltaSharpe  BaseMaxDD%  StressMaxDD%  DeltaMaxDD%  BaseAlphaNW_QQQ  StressAlphaNW_QQQ  BaseAlphaNW_SPY  StressAlphaNW_SPY
MAHORAGA14_2_LONG_PARTICIPATION                    BASELINE                        unstressed stitched OOS    24.1901      24.1901      0.0000    1.238955      1.238955     0.000000    -25.4787      -25.4787       0.0000         0.148428           0.148428         0.179145           0.179145
MAHORAGA14_2_LONG_PARTICIPATION                COST_PLUS_25                commission/slippage x1.25 proxy    24.1901      23.7404     -0.4497    1.238955      1.219946    -0.019009    -25.4787      -25.5315      -0.0527         0.148428           0.144281         0.179145           0.174885
MAHORAGA14_2_LONG_PARTICIPATION                COST_PLUS_50                commission/slippage x1.50 proxy    24.1901      23.2923     -0.8978    1.238955      1.200919    -0.038035    -25.4787      -25.5842      -0.1055         0.148428           0.140149         0.179145           0.170641
MAHORAGA14_2_LONG_PARTICIPATION               COST_PLUS_100                commission/slippage x2.00 proxy    24.1901      22.4008     -1.7893    1.238955      1.162817    -0.076137    -25.4787      -25.6895      -0.2108         0.148428           0.131929         0.179145           0.162198
MAHORAGA14_2_LONG_PARTICIPATION          SLIPPAGE_PLUS_5BPS              extra slippage +5bps via turnover    24.1901      23.4989     -0.6912    1.238955      1.209703    -0.029252    -25.4787      -25.5599      -0.0811         0.148428           0.142054         0.179145           0.172598
MAHORAGA14_2_LONG_PARTICIPATION           ALLOCATOR_TIGHTER           allocator budgets/caps tighter proxy    24.1901      20.4944     -3.6957    1.238955      1.238955    -0.000000    -25.4787      -22.0174       3.4613         0.148428           0.124839         0.179145           0.150364
MAHORAGA14_2_LONG_PARTICIPATION            ALLOCATOR_LOOSER            allocator budgets/caps looser proxy    24.1901      27.0111      2.8210    1.238955      1.234796    -0.004159    -25.4787      -27.7564      -2.2777         0.148428           0.165742         0.179145           0.200766
MAHORAGA14_2_LONG_PARTICIPATION   BULL_PARTICIPATION_WEAKER                bull participation weaker proxy    24.1901      19.3597     -4.8304    1.238955      1.048430    -0.190525    -25.4787      -25.7914      -0.3127         0.148428           0.104777         0.179145           0.133723
MAHORAGA14_2_LONG_PARTICIPATION BULL_PARTICIPATION_STRONGER              bull participation stronger proxy    24.1901      29.2103      5.0203    1.238955      1.420386     0.181431    -25.4787      -25.1650       0.3137         0.148428           0.193797         0.179145           0.226379
MAHORAGA14_2_LONG_PARTICIPATION       EMPIRICAL_PATH_STRESS replay worst empirical blocks on later windows    24.1901      20.4769     -3.7132    1.238955      1.061529    -0.177426    -25.4787      -25.4787      -0.0000         0.148428           0.115066         0.179145           0.144660
```

## Robustness samples
```
                        Variant                     Method  SampleId     CAGR   Sharpe     MaxDD  BudgetMultiplier  LeaderMultiplier
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap         0 0.349784 1.771101 -0.175245               NaN               NaN
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap         1 0.104051 0.647085 -0.419258               NaN               NaN
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap         2 0.354794 1.718079 -0.309711               NaN               NaN
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap         3 0.124773 0.702788 -0.294788               NaN               NaN
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap         4 0.178322 0.973989 -0.336917               NaN               NaN
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap         5 0.361277 1.707891 -0.157952               NaN               NaN
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap         6 0.344972 1.716834 -0.235697               NaN               NaN
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap         7 0.241573 1.224659 -0.270607               NaN               NaN
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap         8 0.271535 1.425687 -0.177896               NaN               NaN
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap         9 0.224393 1.217664 -0.181740               NaN               NaN
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap        10 0.272347 1.392288 -0.293096               NaN               NaN
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap        11 0.240069 1.251676 -0.171802               NaN               NaN
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap        12 0.242340 1.154466 -0.266994               NaN               NaN
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap        13 0.233273 1.168402 -0.391238               NaN               NaN
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap        14 0.166243 0.858913 -0.316907               NaN               NaN
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap        15 0.142291 0.821751 -0.245168               NaN               NaN
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap        16 0.622260 2.482325 -0.271222               NaN               NaN
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap        17 0.194335 1.051407 -0.254787               NaN               NaN
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap        18 0.210211 1.184619 -0.231564               NaN               NaN
MAHORAGA14_2_LONG_PARTICIPATION stationary_block_bootstrap        19 0.272824 1.356243 -0.253509               NaN               NaN
```