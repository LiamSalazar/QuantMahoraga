# Mahoraga14_3R Acceptance Report

## Final decision: KEEP_AS_EXPERIMENTAL_BRANCH

## Stitched comparison
```
                        Variant      CAGR   Sharpe  Sortino      MaxDD  AvgExposure  AvgTurnover  ReturnPerExposure  BetaQQQ  BetaSPY  UpsideCaptureQQQ  DownsideCaptureQQQ  UpsideCaptureSPY  DownsideCaptureSPY  AlphaNW_QQQ  AlphaNW_SPY  AlphaNW_QQQ_t  AlphaNW_SPY_t  AlphaNW_QQQ_p  AlphaNW_SPY_p  p_value  q_value  p_value_vs_SPY  q_value_vs_SPY  p_value_vs_Control  q_value_vs_Control
                            QQQ 20.141212 0.917784 1.455634 -35.241581     1.000000     0.000000           0.000832 1.000000 1.160418          1.000000            1.000000          1.237785            1.216410     0.000000     0.029430          7.547          1.052       0.000000       0.292680      NaN      NaN             NaN             NaN                 NaN                 NaN
                            SPY 14.879523 0.846307 1.308745 -33.717264     1.000000     0.000000           0.000618 0.752495 1.000000          0.703416            0.696234          1.000000            1.000000    -0.002081     0.000000         -0.088         -2.626       0.929869       0.008647      NaN      NaN             NaN             NaN                 NaN                 NaN
                  BASE_ALPHA_V2 24.516345 1.188725 1.996479 -20.235195     0.630308     0.034496           0.001508 0.484404 0.467471          0.706102            0.626041          0.818593            0.685596     0.147931     0.181438          2.473          2.701       0.013389       0.006913      NaN      NaN             NaN             NaN                 NaN                 NaN
 MAHORAGA14_1_LONG_ONLY_CONTROL 24.684280 1.193715 2.006328 -20.235195     0.631606     0.034535           0.001514 0.485257 0.468282          0.707750            0.626751          0.820529            0.686239     0.149353     0.182964          2.490          2.716       0.012777       0.006617 0.323094 0.737732        0.107110        0.611417                 NaN                 NaN
MAHORAGA14_3_LONG_PARTICIPATION 27.364825 1.324779 2.221316 -16.610499     0.630307     0.046093           0.001646 0.500291 0.492504          0.723746            0.627815          0.834971            0.679573     0.169362     0.202811          3.012          3.198       0.002600       0.001383 0.214475 0.612146        0.054203        0.611417            0.179467            0.612146
                         LEGACY 23.068204 1.134758 1.915726 -21.105963     0.606927     0.033096           0.001489 0.460640 0.437465          0.680356            0.605989          0.783143            0.657669     0.140014     0.172927          2.306          2.552       0.021132       0.010715      NaN      NaN             NaN             NaN                 NaN                 NaN
```

## Priority windows
```
            CandidateId    Window      Start        End  CandidateReturn  ControlReturn  QQQReturn  SPYReturn  DeltaReturn_vs_Control  DeltaReturn_vs_QQQ  DeltaReturn_vs_SPY  SharpeLocal  SortinoLocal  MaxDDLocal  BetaQQQLocal  ExposureLocal VsControlStatus          VsQQQStatus      CompensationStatus     PriorityDecision
B1.00_C1.00_L1.00_R1.00 2017_2018 2017-01-03 2018-12-31         0.215379       0.221252   0.319696   0.161447               -0.005873           -0.104317            0.053932     0.666071      0.993378   -0.138461      0.480510       0.621547         WORSENS MATERIALLY_BELOW_QQQ          NOT_ACCEPTABLE FAIL_PRIORITY_WINDOW
B1.00_C1.00_L1.00_R1.00 2020_2021 2020-04-01 2021-12-31         0.741991       0.688098   1.099695   0.891030                0.053893           -0.357704           -0.149039     1.464074      2.388970   -0.166105      0.762416       0.747305        IMPROVES MATERIALLY_BELOW_QQQ ACCEPTABLE_COMPENSATION    PASS_OR_TOLERABLE
B1.00_C1.00_L1.00_R1.00 2023_2024 2023-01-01 2024-12-31         1.756526       1.747975   0.936927   0.575765                0.008551            0.819599            1.180761     2.319570      4.689841   -0.143586      0.926900       0.711075        IMPROVES            BEATS_QQQ ACCEPTABLE_COMPENSATION    PASS_OR_TOLERABLE
```

## Local stability
- base-current robust score: 0.5148
- base-current robust flag: 1

## Leave-one-window-out
```
ExcludedWindow     SelectedCandidateId  SelectedBudgetMultiplier  SelectedConvictionMultiplier  SelectedLeaderMultiplier  SelectedBackoffStrength  SelectionScoreNet  BaseCurrentRank  EvalCandidateReturn  EvalControlReturn  EvalQQQReturn  EvalDelta_vs_Control  EvalDelta_vs_QQQ  EvalSharpeLocal  EvalSortinoLocal  EvalMaxDDLocal  EvalExposureLocal  BaseCurrentEvalDelta_vs_Control  BaseCurrentEvalDelta_vs_QQQ
     2017_2018 B1.05_C1.10_L1.10_R1.05                      1.05                           1.1                       1.1                     1.05           0.995062               40             0.292992           0.221252       0.319696              0.071740         -0.026704         0.829399          1.254508       -0.137116           0.644500                        -0.005873                    -0.104317
     2020_2021 B1.05_C1.10_L1.10_R1.05                      1.05                           1.1                       1.1                     1.05           0.995062               40             0.907443           0.688098       1.099695              0.219345         -0.192252         1.635316          2.712046       -0.161997           0.775296                         0.053893                    -0.357704
     2023_2024 B1.05_C1.10_L1.10_R1.05                      1.05                           1.1                       1.1                     1.05           0.995062               40             2.071707           1.747975       0.936927              0.323732          1.134780         2.469960          5.093577       -0.144512           0.736797                         0.008551                     0.819599
```

## Continuation acceptance
```
 Segment  Fold                         Variant  Activations  ActivationRate  LiftRate  HitRate1W  HitRate4W  NoActHitRate4W  MeanRet1W  MeanRet4W  MeanExcessVsQQQ4W  MeanExcessVsSPY4W  MeanPostDD4W  EdgeVsNoActivation4W  MeanTriggerScore  MeanPressureScore  MeanBreakRisk  MeanBenchmarkScore  PressureEntry  BreakRiskCap  ContinuationUseful                                  FinalRoleDecision      Interpretation
STITCHED     0 MAHORAGA14_3_LONG_PARTICIPATION           10        0.036777  0.101674   0.416667   0.583333        0.556331   0.006966   0.060594           0.030206           0.039959     -0.009885              0.045779          0.483764           0.537915       0.381355            0.640396       0.408534      0.586243                   1 A. continuation se mantiene como filtro de calidad positive_local_edge
```

## Acceptance robustness
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

## Bootstrap summary
```
         Metric       p05       p25       p50       p75       p95      mean
  CandidateCAGR  0.165603  0.221610  0.267482  0.320655  0.399491  0.273377
CandidateSharpe  0.878318  1.112721  1.310472  1.524074  1.813843  1.317130
 CandidateMaxDD -0.309486 -0.255323 -0.212656 -0.187253 -0.155758 -0.224950
      DeltaCAGR -0.015639  0.011423  0.027071  0.045269  0.064573  0.027325
    DeltaSharpe -0.010441  0.074480  0.131321  0.200962  0.289431  0.136692
     DeltaMaxDD -0.032336  0.002823  0.022701  0.045862  0.094810  0.027262
```