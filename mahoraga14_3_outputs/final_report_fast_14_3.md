# Mahoraga14_3 FAST

## Thesis status: FAIL_FAST

FAIL-FAST reasons:
- improvement_microscopic_without_real_participation_shift
- priority_bull_windows_not_improved

## Core comparison
```
                        Variant      CAGR   Sharpe  Sortino      MaxDD  AvgExposure  AvgTurnover  ReturnPerExposure  BetaQQQ  BetaSPY  UpsideCaptureQQQ  DownsideCaptureQQQ  UpsideCaptureSPY  DownsideCaptureSPY  AlphaNW_QQQ  AlphaNW_SPY  AlphaNW_QQQ_t  AlphaNW_SPY_t  AlphaNW_QQQ_p  AlphaNW_SPY_p  p_value  q_value  p_value_vs_SPY  q_value_vs_SPY  p_value_vs_Control  q_value_vs_Control
                            QQQ 20.141212 0.917784 1.455634 -35.241581     1.000000     0.000000           0.000832 1.000000 1.160418          1.000000            1.000000          1.237785            1.216410     0.000000     0.029430          7.547          1.052       0.000000       0.292680      NaN      NaN             NaN             NaN                 NaN                 NaN
                            SPY 14.879523 0.846307 1.308745 -33.717264     1.000000     0.000000           0.000618 0.752495 1.000000          0.703416            0.696234          1.000000            1.000000    -0.002081     0.000000         -0.088         -2.626       0.929869       0.008647      NaN      NaN             NaN             NaN                 NaN                 NaN
                  BASE_ALPHA_V2 24.516345 1.188725 1.996479 -20.235195     0.630308     0.034496           0.001508 0.484404 0.467471          0.706102            0.626041          0.818593            0.685596     0.147931     0.181438          2.473          2.701       0.013389       0.006913      NaN      NaN             NaN             NaN                 NaN                 NaN
 MAHORAGA14_1_LONG_ONLY_CONTROL 24.684280 1.193715 2.006328 -20.235195     0.631606     0.034535           0.001514 0.485257 0.468282          0.707750            0.626751          0.820529            0.686239     0.149353     0.182964          2.490          2.716       0.012777       0.006617 0.323094 0.737732        0.107110        0.611417                 NaN                 NaN
MAHORAGA14_3_LONG_PARTICIPATION 27.364825 1.324779 2.221316 -16.610499     0.630307     0.046093           0.001646 0.500291 0.492504          0.723746            0.627815          0.834971            0.679573     0.169362     0.202811          3.012          3.198       0.002600       0.001383 0.214475 0.612146        0.054203        0.611417            0.179467            0.612146
                         LEGACY 23.068204 1.134758 1.915726 -21.105963     0.606927     0.033096           0.001489 0.460640 0.437465          0.680356            0.605989          0.783143            0.657669     0.140014     0.172927          2.306          2.552       0.021132       0.010715      NaN      NaN             NaN             NaN                 NaN                 NaN
```

## Primary vs control deltas
- CAGR delta: 2.68 pts
- Sharpe delta: 0.1311
- Sortino delta: 0.2150
- MaxDD delta: 3.62 pts
- AvgExposure delta: -0.0013
- UpsideCaptureQQQ delta: 0.0160

## Bull windows
```
   Window Source      Start        End  Mahoraga14_3Return  Mahoraga14_1ControlReturn  QQQReturn  SPYReturn  DeltaReturn_vs_QQQ  DeltaReturn_vs_SPY  DeltaReturn_vs_Control  SharpeLocal  SortinoLocal  MaxDDLocal  BetaQQQLocal  BetaSPYLocal  UpsideCaptureQQQLocal  UpsideCaptureSPYLocal  ExposureLocal  ControlExposureLocal
2017_2018 MANUAL 2017-01-03 2018-12-31            0.215379                   0.221252   0.319696   0.161447           -0.104317            0.053932               -0.005873     0.666071      0.993378   -0.138461      0.480510      0.528758               0.623047               0.725589       0.621547              0.584997
2020_2021 MANUAL 2020-04-01 2021-12-31            0.741991                   0.688098   1.099695   0.891030           -0.357704           -0.149039                0.053893     1.464074      2.388970   -0.166105      0.762416      0.682441               0.847602               0.865407       0.747305              0.742419
2023_2024 MANUAL 2023-01-01 2024-12-31            1.756526                   1.747975   0.936927   0.575765            0.819599            1.180761                0.008551     2.319570      4.689841   -0.143586      0.926900      1.114536               1.025637               1.357549       0.711075              0.765954
```

## Continuation diagnostic
```
 Segment  Fold                         Variant  Activations  ActivationRate  LiftRate  HitRate1W  HitRate4W  NoActHitRate4W  MeanRet1W  MeanRet4W  MeanExcessVsQQQ4W  MeanExcessVsSPY4W  MeanPostDD4W  EdgeVsNoActivation4W  MeanTriggerScore  MeanPressureScore  MeanBreakRisk  MeanBenchmarkScore  PressureEntry  BreakRiskCap  ContinuationUseful
  FOLD_1     1  MAHORAGA14_1_LONG_ONLY_CONTROL            0        0.000000  0.000000   0.000000   0.000000        0.451923   0.000000   0.000000           0.000000           0.000000      0.000000              0.000000          0.000000           0.000000       0.000000            0.000000       0.402251      0.598197                   0
  FOLD_1     1 MAHORAGA14_3_LONG_PARTICIPATION            0        0.000000  0.000000   0.000000   0.000000        0.461538   0.000000   0.000000           0.000000           0.000000      0.000000              0.000000          0.000000           0.000000       0.000000            0.000000       0.402251      0.598197                   0
  FOLD_2     2  MAHORAGA14_1_LONG_ONLY_CONTROL            5        0.048077  0.176238   0.600000   0.800000        0.606061   0.018448   0.096339           0.044490           0.066216     -0.000364              0.077863          0.588614           0.708311       0.495091            0.741008       0.455670      0.586152                   1
  FOLD_2     2 MAHORAGA14_3_LONG_PARTICIPATION            5        0.048077  0.176238   0.800000   0.800000        0.686869   0.018326   0.094112           0.042263           0.063989     -0.009106              0.067178          0.588614           0.708311       0.495091            0.741008       0.455670      0.586152                   1
  FOLD_3     3  MAHORAGA14_1_LONG_ONLY_CONTROL            3        0.028846  0.039761   0.333333   0.666667        0.475248  -0.000749   0.029424           0.004864           0.004323     -0.023739              0.018904          0.536729           0.594654       0.409417            0.807936       0.296628      0.595992                   1
  FOLD_3     3 MAHORAGA14_3_LONG_PARTICIPATION            3        0.028846  0.039761   0.333333   0.666667        0.465347  -0.000640   0.031457           0.006897           0.006356     -0.022512              0.022691          0.536729           0.594654       0.409417            0.807936       0.296628      0.595992                   1
  FOLD_4     4  MAHORAGA14_1_LONG_ONLY_CONTROL            0        0.000000  0.000000   0.000000   0.000000        0.666667   0.000000   0.000000           0.000000           0.000000      0.000000              0.000000          0.000000           0.000000       0.000000            0.000000       0.369439      0.569829                   0
  FOLD_4     4 MAHORAGA14_3_LONG_PARTICIPATION            0        0.000000  0.000000   0.000000   0.000000        0.705128   0.000000   0.000000           0.000000           0.000000      0.000000              0.000000          0.000000           0.000000       0.000000            0.000000       0.369439      0.569829                   0
  FOLD_5     5  MAHORAGA14_1_LONG_ONLY_CONTROL            2        0.023256  0.024331   0.000000   0.500000        0.452381  -0.003436   0.090658           0.074792           0.079808     -0.003021              0.085218          0.625953           0.564734       0.436275            0.777953       0.449896      0.574338                   1
  FOLD_5     5 MAHORAGA14_3_LONG_PARTICIPATION            2        0.023256  0.024331   0.000000   0.500000        0.464286  -0.003058   0.081099           0.065234           0.070250     -0.002774              0.072693          0.625953           0.564734       0.436275            0.777953       0.449896      0.574338                   1
STITCHED     0  MAHORAGA14_1_LONG_ONLY_CONTROL           10        0.036777  0.101674   0.333333   0.583333        0.530410   0.006927   0.062607           0.032219           0.041972     -0.006590              0.051372          0.483764           0.537915       0.381355            0.640396       0.408534      0.586243                   1
STITCHED     0 MAHORAGA14_3_LONG_PARTICIPATION           10        0.036777  0.101674   0.416667   0.583333        0.556331   0.006966   0.060594           0.030206           0.039959     -0.009885              0.045779          0.483764           0.537915       0.381355            0.640396       0.408534      0.586243                   1
```

## Upside participation decomposition
```
   Window      Start        End  WindowReturnPrimary1x  WindowReturnControl1x  WindowReturnQQQ  EstimatedCashDragContribution  EstimatedBetaResidualReliefContribution  EstimatedDefenseDragContribution  EstimatedNameSelectionContribution  EstimatedLeaderUnderweightContribution  EstimatedAllocatorRestrictionContribution  EstimatedUnderExposureContribution  AvgLongBudget  AvgLeaderBlend  AvgConvictionMultiplier  AvgLeaderMultiplier  AvgGateAdj  AvgVolAdj  AvgExpCapAdj  AvgCashDragBefore  AvgCashDragAfter  AvgLeaderActiveWeight  AvgLeaderTopShareBefore
2017_2018 2017-01-03 2018-12-31               0.436219               0.427265         0.319696                      -0.012408                                -0.012276                         -0.000930                           -0.048590                               -0.303521                                  -0.000052                           -0.079507       0.843133        0.192697                 1.118402             1.049367    1.102158   1.096261      1.109820           0.103265          0.136835               0.863165                 0.979943
2020_2021 2020-04-01 2021-12-31               1.004166               0.864061         0.922807                       0.024507                                 0.004977                         -0.002826                           -0.093656                               -0.478885                                  -0.009027                           -0.089751       0.865958        0.242685                 1.130100             1.062194    1.136739   1.101359      1.138528           0.119193          0.124302               0.875698                 0.991751
2023_2024 2023-01-01 2024-12-31               1.914065               1.969170         0.880782                       0.045191                                -0.045903                          0.006729                           -0.139224                               -0.500832                                   0.005101                           -0.067633       0.811832        0.183740                 1.104375             1.046707    1.094606   1.078714      1.093845           0.074114          0.170622               0.829378                 1.017283
```

## Stress suite
```
                        Variant                      Scenario                                   ScenarioNote  BaseCAGR%  StressCAGR%  DeltaCAGR%  BaseSharpe  StressSharpe  DeltaSharpe  BaseMaxDD%  StressMaxDD%  DeltaMaxDD%  BaseAlphaNW_QQQ  StressAlphaNW_QQQ  BaseAlphaNW_SPY  StressAlphaNW_SPY
MAHORAGA14_3_LONG_PARTICIPATION                      BASELINE                        unstressed stitched OOS    27.3648      27.3648      0.0000    1.324779      1.324779     0.000000    -16.6105      -16.6105       0.0000         0.169362           0.169362         0.202811           0.202811
MAHORAGA14_3_LONG_PARTICIPATION                  COST_PLUS_25                commission/slippage x1.25 proxy    27.3648      26.8859     -0.4789    1.324779      1.305801    -0.018978    -16.6105      -16.7096      -0.0991         0.169362           0.164974         0.202811           0.198299
MAHORAGA14_3_LONG_PARTICIPATION                  COST_PLUS_50                commission/slippage x1.50 proxy    27.3648      26.4087     -0.9561    1.324779      1.286806    -0.037972    -16.6105      -16.8086      -0.1981         0.169362           0.160603         0.202811           0.193804
MAHORAGA14_3_LONG_PARTICIPATION                 COST_PLUS_100                commission/slippage x2.00 proxy    27.3648      25.4596     -1.9052    1.324779      1.248771    -0.076008    -16.6105      -17.0063      -0.3958         0.169362           0.151910         0.202811           0.184864
MAHORAGA14_3_LONG_PARTICIPATION            SLIPPAGE_PLUS_5BPS              extra slippage +5bps via turnover    27.3648      26.6288     -0.7361    1.324779      1.295575    -0.029204    -16.6105      -16.7629      -0.1524         0.169362           0.162618         0.202811           0.195876
MAHORAGA14_3_LONG_PARTICIPATION             ALLOCATOR_TIGHTER           allocator budgets/caps tighter proxy    27.3648      23.1322     -4.2326    1.324779      1.324779     0.000000    -16.6105      -14.1684       2.4421         0.169362           0.142245         0.202811           0.169961
MAHORAGA14_3_LONG_PARTICIPATION              ALLOCATOR_LOOSER            allocator budgets/caps looser proxy    27.3648      30.5631      3.1982    1.324779      1.319859    -0.004919    -16.6105      -17.9571      -1.3466         0.169362           0.189094         0.202811           0.227258
MAHORAGA14_3_LONG_PARTICIPATION             CONVICTION_WEAKER          conviction amplification weaker proxy    27.3648      24.0985     -3.2663    1.324779      1.205718    -0.119061    -16.6105      -16.9279      -0.3174         0.169362           0.139923         0.202811           0.172178
MAHORAGA14_3_LONG_PARTICIPATION           CONVICTION_STRONGER        conviction amplification stronger proxy    27.3648      30.7152      3.3503    1.324779      1.440656     0.115878    -16.6105      -16.2920       0.3185         0.169362           0.199557         0.202811           0.234240
MAHORAGA14_3_LONG_PARTICIPATION   LEADER_PARTICIPATION_WEAKER              leader participation weaker proxy    27.3648      22.3924     -4.9724    1.324779      1.141570    -0.183209    -16.6105      -17.2612      -0.6507         0.169362           0.124783         0.202811           0.156359
MAHORAGA14_3_LONG_PARTICIPATION LEADER_PARTICIPATION_STRONGER            leader participation stronger proxy    27.3648      32.9017      5.5369    1.324779      1.512218     0.187439    -16.6105      -15.9178       0.6927         0.169362           0.218998         0.202811           0.254560
MAHORAGA14_3_LONG_PARTICIPATION         EMPIRICAL_PATH_STRESS replay worst empirical blocks on later windows    27.3648      23.7957     -3.5692    1.324779      1.159959    -0.164820    -16.6105      -18.9923      -2.3818         0.169362           0.137490         0.202811           0.169978
```

## Robustness samples
```
                        Variant                     Method  SampleId     CAGR   Sharpe     MaxDD  BudgetMultiplier  ConvictionMultiplier  LeaderMultiplier
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap         0 0.387153 1.856535 -0.179119               NaN                   NaN               NaN
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap         1 0.113333 0.672270 -0.433986               NaN                   NaN               NaN
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap         2 0.363649 1.700797 -0.344805               NaN                   NaN               NaN
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap         3 0.144838 0.778606 -0.286989               NaN                   NaN               NaN
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap         4 0.206177 1.062116 -0.253379               NaN                   NaN               NaN
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap         5 0.402943 1.813728 -0.141590               NaN                   NaN               NaN
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap         6 0.365473 1.727746 -0.219129               NaN                   NaN               NaN
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap         7 0.286532 1.361276 -0.259820               NaN                   NaN               NaN
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap         8 0.306536 1.528220 -0.157666               NaN                   NaN               NaN
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap         9 0.257655 1.320383 -0.188115               NaN                   NaN               NaN
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap        10 0.291090 1.422575 -0.307311               NaN                   NaN               NaN
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap        11 0.265642 1.309301 -0.211490               NaN                   NaN               NaN
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap        12 0.274903 1.225779 -0.205132               NaN                   NaN               NaN
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap        13 0.272697 1.283528 -0.323712               NaN                   NaN               NaN
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap        14 0.199634 0.969616 -0.295719               NaN                   NaN               NaN
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap        15 0.126204 0.722884 -0.274021               NaN                   NaN               NaN
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap        16 0.664989 2.521977 -0.192705               NaN                   NaN               NaN
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap        17 0.235968 1.182396 -0.179322               NaN                   NaN               NaN
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap        18 0.215561 1.157864 -0.218509               NaN                   NaN               NaN
MAHORAGA14_3_LONG_PARTICIPATION stationary_block_bootstrap        19 0.311530 1.471601 -0.200583               NaN                   NaN               NaN
```