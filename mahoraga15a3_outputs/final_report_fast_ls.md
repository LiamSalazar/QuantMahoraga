# Mahoraga15A3 FAST

## Thesis status: THESIS FAILED FAST

## 1. Stitched comparison
| Variant | CAGR% | Sharpe | Sortino | MaxDD% | AvgExposure | GrossLong | GrossShort | NetExposure | AvgTurnover | ReturnPerExposure | BetaQQQ | BetaSPY | AlphaNW_QQQ | AlphaNW_SPY | UpsideCaptureQQQ | DownsideCaptureQQQ | UpsideCaptureSPY | DownsideCaptureSPY |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAHORAGA_14_1_LONG_ONLY | 24.68 | 1.1937 | 2.0063 | -20.24 | 0.6316 | 0.6316 | 0.0 | 0.6316 | 0.0345 | 0.001514 | 0.4853 | 0.4683 | 0.149353 | 0.182964 | 0.7078 | 0.6268 | 0.8205 | 0.6862 |
| DELEVERED_CONTROL | 22.54 | 1.2236 | 2.0497 | -15.49 | 0.545 | 0.545 | 0.0 | 0.545 | 0.0333 | 0.001597 | 0.4031 | 0.3849 | 0.144264 | 0.17274 | 0.6078 | 0.5275 | 0.7032 | 0.5732 |
| MAHORAGA_15A3_LS | 23.39 | 1.218 | 2.0534 | -17.78 | 0.6632 | 0.6041 | 0.0591 | 0.545 | 0.0392 | 0.001362 | 0.4186 | 0.3973 | 0.150118 | 0.180292 | 0.6296 | 0.5459 | 0.7234 | 0.5871 |
| QQQ | 20.14 | 0.9178 | 1.4556 | -35.24 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | 0.000832 | 1.0 | 1.1604 | 0.0 | 0.02943 | 1.0 | 1.0 | 1.2378 | 1.2164 |
| SPY | 14.88 | 0.8463 | 1.3087 | -33.72 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | 0.000618 | 0.7525 | 1.0 | -0.002081 | 0.0 | 0.7034 | 0.6962 | 1.0 | 1.0 |

## 2. Delevered control check
| CAGR% | Sharpe | Sortino | MaxDD% | AvgExposure | GrossLong | GrossShort | NetExposure | AvgTurnover | ReturnPerExposure | BetaQQQ | BetaSPY | AlphaNW_QQQ | AlphaNW_SPY | UpsideCaptureQQQ | DownsideCaptureQQQ | UpsideCaptureSPY | DownsideCaptureSPY | DeltaSharpe_vs_Long | DeltaSortino_vs_Long | DeltaCAGR_vs_Long% | DeltaBetaQQQ_vs_Long | DeltaBetaSPY_vs_Long | DeltaSharpe_vs_Delevered | DeltaSortino_vs_Delevered | DeltaCAGR_vs_Delevered% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 24.68 | 1.1937 | 2.0063 | -20.24 | 0.6316 | 0.6316 | 0.0 | 0.6316 | 0.0345 | 0.001514 | 0.4853 | 0.4683 | 0.149353 | 0.182964 | 0.7078 | 0.6268 | 0.8205 | 0.6862 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -0.0299 | -0.0434 | 2.14 |
| 22.54 | 1.2236 | 2.0497 | -15.49 | 0.545 | 0.545 | 0.0 | 0.545 | 0.0333 | 0.001597 | 0.4031 | 0.3849 | 0.144264 | 0.17274 | 0.6078 | 0.5275 | 0.7032 | 0.5732 | 0.0299 | 0.0434 | -2.14 | -0.0822 | -0.0834 | 0.0 | 0.0 | 0.0 |
| 23.39 | 1.218 | 2.0534 | -17.78 | 0.6632 | 0.6041 | 0.0591 | 0.545 | 0.0392 | 0.001362 | 0.4186 | 0.3973 | 0.150118 | 0.180292 | 0.6296 | 0.5459 | 0.7234 | 0.5871 | 0.0243 | 0.0471 | -1.29 | -0.0667 | -0.071 | -0.0056 | 0.0037 | 0.85 |

## 3. Pairwise p-values / q-values
| Target | Reference | Comparison | p_value | q_value |
| --- | --- | --- | --- | --- |
| DELEVERED_CONTROL | MAHORAGA_14_1_LONG_ONLY | DELEVERED_CONTROL_vs_MAHORAGA_14_1_LONG_ONLY | 0.959813 | 1.0 |
| MAHORAGA_15A3_LS | MAHORAGA_14_1_LONG_ONLY | MAHORAGA_15A3_LS_vs_MAHORAGA_14_1_LONG_ONLY | 0.950209 | 1.0 |
| MAHORAGA_15A3_LS | DELEVERED_CONTROL | MAHORAGA_15A3_LS_vs_DELEVERED_CONTROL | 0.106703 | 1.0 |
| MAHORAGA_14_1_LONG_ONLY | QQQ | MAHORAGA_14_1_LONG_ONLY_vs_QQQ | 0.323094 | 1.0 |
| MAHORAGA_14_1_LONG_ONLY | SPY | MAHORAGA_14_1_LONG_ONLY_vs_SPY | 0.10711 | 1.0 |
| DELEVERED_CONTROL | QQQ | DELEVERED_CONTROL_vs_QQQ | 0.443816 | 1.0 |
| DELEVERED_CONTROL | SPY | DELEVERED_CONTROL_vs_SPY | 0.168385 | 1.0 |
| MAHORAGA_15A3_LS | QQQ | MAHORAGA_15A3_LS_vs_QQQ | 0.397345 | 1.0 |
| MAHORAGA_15A3_LS | SPY | MAHORAGA_15A3_LS_vs_SPY | 0.144409 | 1.0 |

## 4. PnL attribution by sleeve
| SegmentType | Segment | Start | End | Days | PnLTotalPctInit | PnLLongPctInit | PnLCrashHedgePctInit | PnLBearHedgePctInit | PnLSystematicHedgePctInit | PnLInteractionPctInit | PnLCashDragPctInit | AvgGrossShort | AvgCrashShort | AvgBearShort | AvgNetExposure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STITCHED | FULL_OOS | 2017-01-03 | 2026-02-19 | 2295 | 5.782994 | 5.927251 | -0.054977 | -0.094572 | -0.149549 | 0.005292 | -1.321815 | 0.0591 | 0.0278 | 0.0313 | 0.545 |
| FOLD | FOLD_1 | 2017-01-03 | 2018-12-31 | 502 | 0.226391 | 0.20987 | 0.005529 | 0.010449 | 0.015978 | 0.000543 | -0.035471 | 0.0858 | 0.0401 | 0.0457 | 0.4746 |
| FOLD | FOLD_2 | 2019-01-02 | 2020-12-31 | 505 | 0.740877 | 0.744628 | -0.003621 | -0.000561 | -0.004182 | 0.000431 | -0.380921 | 0.0451 | 0.0223 | 0.0228 | 0.5568 |
| FOLD | FOLD_3 | 2021-01-04 | 2022-12-30 | 503 | 0.476612 | 0.45658 | 0.011328 | 0.007846 | 0.019174 | 0.000858 | 0.156934 | 0.0447 | 0.0209 | 0.0238 | 0.5081 |
| FOLD | FOLD_4 | 2023-01-03 | 2024-06-28 | 374 | 3.778928 | 3.91337 | -0.060318 | -0.07579 | -0.136108 | 0.001666 | -0.454371 | 0.0761 | 0.0342 | 0.0419 | 0.741 |
| FOLD | FOLD_5 | 2024-07-01 | 2026-02-19 | 411 | 0.560185 | 0.602803 | -0.007895 | -0.036516 | -0.044411 | 0.001793 | -0.607987 | 0.0458 | 0.0222 | 0.0236 | 0.4836 |
| YEAR | 2017 | 2017-01-03 | 2017-12-29 | 251 | 0.01844 | 0.046612 | -0.014577 | -0.013857 | -0.028434 | 0.000262 | -0.030025 | 0.1142 | 0.0535 | 0.0606 | 0.5598 |
| YEAR | 2018 | 2018-01-02 | 2018-12-31 | 251 | 0.207951 | 0.163258 | 0.020106 | 0.024306 | 0.044412 | 0.000281 | -0.005446 | 0.0574 | 0.0266 | 0.0307 | 0.3893 |
| YEAR | 2019 | 2019-01-02 | 2019-12-31 | 252 | 0.244095 | 0.252422 | -0.004643 | -0.003792 | -0.008434 | 0.000108 | -0.263662 | 0.0547 | 0.0275 | 0.0272 | 0.638 |
| YEAR | 2020 | 2020-01-02 | 2020-12-31 | 253 | 0.496782 | 0.492206 | 0.001022 | 0.003231 | 0.004253 | 0.000323 | -0.117259 | 0.0356 | 0.017 | 0.0185 | 0.4759 |
| YEAR | 2021 | 2021-01-04 | 2021-12-31 | 252 | 0.570283 | 0.626969 | -0.029538 | -0.027378 | -0.056917 | 0.000231 | -0.175986 | 0.0353 | 0.0188 | 0.0164 | 0.837 |
| YEAR | 2022 | 2022-01-03 | 2022-12-30 | 251 | -0.093671 | -0.170389 | 0.040867 | 0.035224 | 0.076091 | 0.000627 | 0.33292 | 0.0542 | 0.023 | 0.0312 | 0.1778 |
| YEAR | 2023 | 2023-01-03 | 2023-12-29 | 250 | 1.496019 | 1.606114 | -0.046812 | -0.064378 | -0.11119 | 0.001095 | -0.29454 | 0.0672 | 0.03 | 0.0372 | 0.7642 |
| YEAR | 2024 | 2024-01-02 | 2024-12-31 | 252 | 2.379674 | 2.312151 | 0.036265 | 0.030415 | 0.06668 | 0.000843 | -0.138512 | 0.0605 | 0.0276 | 0.033 | 0.5744 |
| YEAR | 2025 | 2025-01-02 | 2025-12-31 | 250 | 0.576672 | 0.694703 | -0.04786 | -0.071306 | -0.119165 | 0.001135 | -0.618973 | 0.052 | 0.0257 | 0.0264 | 0.4664 |
| YEAR | 2026 | 2026-01-02 | 2026-02-19 | 33 | -0.113252 | -0.096795 | -0.009806 | -0.007038 | -0.016844 | 0.000387 | -0.010333 | 0.0678 | 0.0318 | 0.0359 | 0.713 |
| CRISIS | CRISIS_2020_CRASH | 2020-02-19 | 2020-04-30 | 51 | -0.137593 | -0.137593 | 0.0 | 0.0 | 0.0 | 0.0 | 0.007753 | 0.0 | 0.0 | 0.0 | 0.0747 |
| CRISIS | CRISIS_2022_TECH_BEAR | 2022-01-03 | 2022-12-30 | 251 | -0.093671 | -0.170389 | 0.040867 | 0.035224 | 0.076091 | 0.000627 | 0.33292 | 0.0542 | 0.023 | 0.0312 | 0.1778 |
| CRISIS | AUTO_QQQ_STRESS_1 | 2018-09-25 | 2018-12-24 | 63 | -0.100711 | -0.147571 | 0.022692 | 0.024026 | 0.046718 | 0.000142 | 0.200239 | 0.1445 | 0.0631 | 0.0815 | 0.2129 |
| CRISIS | AUTO_QQQ_STRESS_2 | 2025-01-07 | 2025-04-08 | 63 | -0.364939 | -0.380832 | 0.006699 | 0.008748 | 0.015446 | 0.000446 | 1.025557 | 0.0554 | 0.0286 | 0.0268 | 0.3751 |

## 5. Crisis window scorecard
| Window | Start | End | LS_Return% | LS_Sharpe | LS_Sortino | LS_MaxDD% | LS_BetaQQQ | LS_BetaSPY | LS_GrossShort | CrashHedgePnLPctInit | BearHedgePnLPctInit | LongOnly_Return% | LongOnly_Sharpe | LongOnly_Sortino | LongOnly_MaxDD% | Delevered_Return% | Delevered_Sharpe | Delevered_Sortino | Delevered_MaxDD% | DeltaReturn_vs_LongOnly% | DeltaReturn_vs_Delevered% | DeltaSharpe_vs_LongOnly | DeltaSharpe_vs_Delevered | DeltaSortino_vs_LongOnly | DeltaSortino_vs_Delevered | DeltaMaxDD_vs_LongOnly% | DeltaMaxDD_vs_Delevered% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CRISIS_2020_CRASH | 2020-02-19 | 2020-04-30 | -8.07 | -2.393 | -2.6298 | -10.67 | 0.0384 | 0.0274 | 0.0 | 0.0 | 0.0 | -8.49 | -2.3985 | -2.6326 | -11.19 | -8.07 | -2.393 | -2.6298 | -10.67 | 0.42 | 0.0 | 0.0055 | 0.0 | 0.0028 | 0.0 | 0.52 | 0.0 |
| CRISIS_2022_TECH_BEAR | 2022-01-03 | 2022-12-30 | -3.69 | -0.331 | -0.5019 | -7.35 | 0.1887 | 0.2382 | 0.0542 | 0.040867 | 0.035224 | -7.08 | -0.5509 | -0.8504 | -10.75 | -3.67 | -0.3703 | -0.5479 | -6.95 | 3.39 | -0.02 | 0.2199 | 0.0393 | 0.3485 | 0.046 | 3.4 | -0.4 |
| AUTO_QQQ_STRESS_1 | 2018-09-25 | 2018-12-24 | -7.64 | -3.6625 | -4.7559 | -8.39 | 0.1583 | 0.1846 | 0.1445 | 0.022692 | 0.024026 | -11.76 | -4.0959 | -5.5644 | -12.58 | -6.97 | -3.904 | -4.7443 | -7.75 | 4.12 | -0.67 | 0.4334 | 0.2415 | 0.8085 | -0.0116 | 4.19 | -0.64 |
| AUTO_QQQ_STRESS_2 | 2025-01-07 | 2025-04-08 | -5.69 | -1.7443 | -2.3259 | -10.29 | 0.1862 | 0.1702 | 0.0554 | 0.006699 | 0.008748 | -6.26 | -1.7439 | -2.3848 | -12.01 | -5.69 | -1.8198 | -2.3606 | -9.73 | 0.57 | 0.0 | -0.0004 | 0.0755 | 0.0589 | 0.0347 | 1.72 | -0.56 |

## 6. Crash vs bear activity
| SegmentType | Segment | Start | End | AvgTotalShort | AvgCrashShort | AvgBearShort | CrashShareOfShort | BearShareOfShort | CrashActiveDayPct | BearActiveDayPct | AvgCrashQQQShort | AvgCrashSPYShort | AvgBearQQQShort | AvgBearSPYShort | CrashPnLPctInit | BearPnLPctInit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STITCHED | FULL_OOS | 2017-01-03 | 2026-02-19 | 0.0591 | 0.0278 | 0.0313 | 0.4703 | 0.5297 | 0.5142 | 0.5333 | 0.016 | 0.0118 | 0.0177 | 0.0136 | -0.054977 | -0.094572 |
| FOLD | FOLD_1 | 2017-01-03 | 2018-12-31 | 0.0858 | 0.0401 | 0.0457 | 0.4673 | 0.5327 | 0.6992 | 0.7191 | 0.0248 | 0.0153 | 0.0279 | 0.0178 | 0.005529 | 0.010449 |
| FOLD | FOLD_2 | 2019-01-02 | 2020-12-31 | 0.0451 | 0.0223 | 0.0228 | 0.4937 | 0.5063 | 0.4218 | 0.4475 | 0.0121 | 0.0102 | 0.0122 | 0.0107 | -0.003621 | -0.000561 |
| FOLD | FOLD_3 | 2021-01-04 | 2022-12-30 | 0.0447 | 0.0209 | 0.0238 | 0.4677 | 0.5323 | 0.4632 | 0.501 | 0.0113 | 0.0096 | 0.0122 | 0.0116 | 0.011328 | 0.007846 |
| FOLD | FOLD_4 | 2023-01-03 | 2024-06-28 | 0.0761 | 0.0342 | 0.0419 | 0.4491 | 0.5509 | 0.6658 | 0.6658 | 0.0196 | 0.0145 | 0.0238 | 0.0181 | -0.060318 | -0.07579 |
| FOLD | FOLD_5 | 2024-07-01 | 2026-02-19 | 0.0458 | 0.0222 | 0.0236 | 0.4842 | 0.5158 | 0.326 | 0.3309 | 0.0123 | 0.0099 | 0.013 | 0.0107 | -0.007895 | -0.036516 |
| YEAR | 2017 | 2017-01-03 | 2017-12-29 | 0.1142 | 0.0535 | 0.0606 | 0.4689 | 0.5311 | 0.9442 | 0.9442 | 0.0351 | 0.0185 | 0.0393 | 0.0213 | -0.014577 | -0.013857 |
| YEAR | 2018 | 2018-01-02 | 2018-12-31 | 0.0574 | 0.0266 | 0.0307 | 0.4642 | 0.5358 | 0.4542 | 0.494 | 0.0146 | 0.0121 | 0.0165 | 0.0143 | 0.020106 | 0.024306 |
| YEAR | 2019 | 2019-01-02 | 2019-12-31 | 0.0547 | 0.0275 | 0.0272 | 0.5031 | 0.4969 | 0.4722 | 0.4881 | 0.0141 | 0.0135 | 0.0137 | 0.0135 | -0.004643 | -0.003792 |
| YEAR | 2020 | 2020-01-02 | 2020-12-31 | 0.0356 | 0.017 | 0.0185 | 0.4791 | 0.5209 | 0.3715 | 0.4071 | 0.0102 | 0.0069 | 0.0106 | 0.0079 | 0.001022 | 0.003231 |
| YEAR | 2021 | 2021-01-04 | 2021-12-31 | 0.0353 | 0.0188 | 0.0164 | 0.534 | 0.466 | 0.4405 | 0.4405 | 0.0107 | 0.0082 | 0.0093 | 0.0071 | -0.029538 | -0.027378 |
| YEAR | 2022 | 2022-01-03 | 2022-12-30 | 0.0542 | 0.023 | 0.0312 | 0.4244 | 0.5756 | 0.4861 | 0.5618 | 0.0119 | 0.0111 | 0.0152 | 0.016 | 0.040867 | 0.035224 |
| YEAR | 2023 | 2023-01-03 | 2023-12-29 | 0.0672 | 0.03 | 0.0372 | 0.4458 | 0.5542 | 0.648 | 0.648 | 0.0172 | 0.0127 | 0.021 | 0.0162 | -0.046812 | -0.064378 |
| YEAR | 2024 | 2024-01-02 | 2024-12-31 | 0.0605 | 0.0276 | 0.033 | 0.4553 | 0.5447 | 0.4802 | 0.4762 | 0.016 | 0.0115 | 0.019 | 0.014 | 0.036265 | 0.030415 |
| YEAR | 2025 | 2025-01-02 | 2025-12-31 | 0.052 | 0.0257 | 0.0264 | 0.4932 | 0.5068 | 0.312 | 0.324 | 0.0139 | 0.0117 | 0.0142 | 0.0122 | -0.04786 | -0.071306 |
| YEAR | 2026 | 2026-01-02 | 2026-02-19 | 0.0678 | 0.0318 | 0.0359 | 0.47 | 0.53 | 0.6667 | 0.6667 | 0.0173 | 0.0146 | 0.0194 | 0.0165 | -0.009806 | -0.007038 |
| CRISIS | CRISIS_2020_CRASH | 2020-02-19 | 2020-04-30 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| CRISIS | CRISIS_2022_TECH_BEAR | 2022-01-03 | 2022-12-30 | 0.0542 | 0.023 | 0.0312 | 0.4244 | 0.5756 | 0.4861 | 0.5618 | 0.0119 | 0.0111 | 0.0152 | 0.016 | 0.040867 | 0.035224 |
| CRISIS | AUTO_QQQ_STRESS_1 | 2018-09-25 | 2018-12-24 | 0.1445 | 0.0631 | 0.0815 | 0.4362 | 0.5638 | 0.873 | 0.873 | 0.0338 | 0.0293 | 0.0421 | 0.0394 | 0.022692 | 0.024026 |
| CRISIS | AUTO_QQQ_STRESS_2 | 2025-01-07 | 2025-04-08 | 0.0554 | 0.0286 | 0.0268 | 0.5168 | 0.4832 | 0.4286 | 0.4286 | 0.0147 | 0.0139 | 0.0136 | 0.0132 | 0.006699 | 0.008748 |

## 7. Short activity summary
| SegmentType | Segment | Start | End | AvgGrossShort | AvgCrashShort | AvgBearShort | MaxGrossShort | CrashActiveDayPct | BearActiveDayPct | AvgLongBudget | AvgNetExposure | AvgCashBuffer | AvgTurnover | HedgePnLPctInit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STITCHED | FULL_OOS | 2017-01-03 | 2026-02-19 | 0.0591 | 0.0278 | 0.0313 | 0.297 | 0.5142 | 0.5333 | 0.6041 | 0.545 | 0.3368 | 0.0392 | -0.149549 |
| FOLD | FOLD_1 | 2017-01-03 | 2018-12-31 | 0.0858 | 0.0401 | 0.0457 | 0.2871 | 0.6992 | 0.7191 | 0.5603 | 0.4746 | 0.3539 | 0.0477 | 0.015978 |
| FOLD | FOLD_2 | 2019-01-02 | 2020-12-31 | 0.0451 | 0.0223 | 0.0228 | 0.2903 | 0.4218 | 0.4475 | 0.6019 | 0.5568 | 0.353 | 0.0298 | -0.004182 |
| FOLD | FOLD_3 | 2021-01-04 | 2022-12-30 | 0.0447 | 0.0209 | 0.0238 | 0.297 | 0.4632 | 0.501 | 0.5528 | 0.5081 | 0.4025 | 0.0394 | 0.019174 |
| FOLD | FOLD_4 | 2023-01-03 | 2024-06-28 | 0.0761 | 0.0342 | 0.0419 | 0.2837 | 0.6658 | 0.6658 | 0.8171 | 0.741 | 0.1068 | 0.0416 | -0.136108 |
| FOLD | FOLD_5 | 2024-07-01 | 2026-02-19 | 0.0458 | 0.0222 | 0.0236 | 0.2904 | 0.326 | 0.3309 | 0.5294 | 0.4836 | 0.4248 | 0.0376 | -0.044411 |
| YEAR | 2017 | 2017-01-03 | 2017-12-29 | 0.1142 | 0.0535 | 0.0606 | 0.2389 | 0.9442 | 0.9442 | 0.674 | 0.5598 | 0.2118 | 0.0667 | -0.028434 |
| YEAR | 2018 | 2018-01-02 | 2018-12-31 | 0.0574 | 0.0266 | 0.0307 | 0.2871 | 0.4542 | 0.494 | 0.4467 | 0.3893 | 0.4959 | 0.0288 | 0.044412 |
| YEAR | 2019 | 2019-01-02 | 2019-12-31 | 0.0547 | 0.0275 | 0.0272 | 0.2903 | 0.4722 | 0.4881 | 0.6927 | 0.638 | 0.2526 | 0.0389 | -0.008434 |
| YEAR | 2020 | 2020-01-02 | 2020-12-31 | 0.0356 | 0.017 | 0.0185 | 0.2292 | 0.3715 | 0.4071 | 0.5114 | 0.4759 | 0.453 | 0.0208 | 0.004253 |
| YEAR | 2021 | 2021-01-04 | 2021-12-31 | 0.0353 | 0.0188 | 0.0164 | 0.2528 | 0.4405 | 0.4405 | 0.8723 | 0.837 | 0.0924 | 0.0562 | -0.056917 |
| YEAR | 2022 | 2022-01-03 | 2022-12-30 | 0.0542 | 0.023 | 0.0312 | 0.297 | 0.4861 | 0.5618 | 0.232 | 0.1778 | 0.7138 | 0.0225 | 0.076091 |
| YEAR | 2023 | 2023-01-03 | 2023-12-29 | 0.0672 | 0.03 | 0.0372 | 0.2837 | 0.648 | 0.648 | 0.8314 | 0.7642 | 0.1014 | 0.0449 | -0.11119 |
| YEAR | 2024 | 2024-01-02 | 2024-12-31 | 0.0605 | 0.0276 | 0.033 | 0.2904 | 0.4802 | 0.4762 | 0.6349 | 0.5744 | 0.3046 | 0.0355 | 0.06668 |
| YEAR | 2025 | 2025-01-02 | 2025-12-31 | 0.052 | 0.0257 | 0.0264 | 0.2857 | 0.312 | 0.324 | 0.5184 | 0.4664 | 0.4296 | 0.034 | -0.119165 |
| YEAR | 2026 | 2026-01-02 | 2026-02-19 | 0.0678 | 0.0318 | 0.0359 | 0.2438 | 0.6667 | 0.6667 | 0.7807 | 0.713 | 0.1515 | 0.0716 | -0.016844 |
| CRISIS | CRISIS_2020_CRASH | 2020-02-19 | 2020-04-30 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0747 | 0.0747 | 0.9253 | 0.0094 | 0.0 |
| CRISIS | CRISIS_2022_TECH_BEAR | 2022-01-03 | 2022-12-30 | 0.0542 | 0.023 | 0.0312 | 0.297 | 0.4861 | 0.5618 | 0.232 | 0.1778 | 0.7138 | 0.0225 | 0.076091 |
| CRISIS | AUTO_QQQ_STRESS_1 | 2018-09-25 | 2018-12-24 | 0.1445 | 0.0631 | 0.0815 | 0.2871 | 0.873 | 0.873 | 0.3575 | 0.2129 | 0.498 | 0.0419 | 0.046718 |
| CRISIS | AUTO_QQQ_STRESS_2 | 2025-01-07 | 2025-04-08 | 0.0554 | 0.0286 | 0.0268 | 0.2621 | 0.4286 | 0.4286 | 0.4305 | 0.3751 | 0.5141 | 0.0556 | 0.015446 |

## 8. Timing sensitivity
| Variant | Scenario | ScenarioNote | CAGR% | Sharpe | Sortino | MaxDD% | GrossShort | BetaQQQ | BetaSPY | AlphaNW_QQQ | AlphaNW_SPY | DeltaCAGR_vs_LSBase% | DeltaSharpe_vs_LSBase | DeltaSortino_vs_LSBase | DeltaMaxDD_vs_LSBase% | DeltaSharpe_vs_Delevered | DeltaSortino_vs_Delevered | DeltaCAGR_vs_Delevered% | DeltaSharpe_vs_LongOnly | DeltaSortino_vs_LongOnly | DeltaCAGR_vs_LongOnly% | ScenarioType | ShiftRebalances | DeltaGrossShort_vs_LSBase | DeltaBetaQQQ_vs_LSBase | DeltaBetaSPY_vs_LSBase | CrisisGrossShortAvg | CrashGrossShortAvg | BearGrossShortAvg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAHORAGA_15A3_LS | BASELINE_15A3 | Mahoraga15A3 unstressed stitched OOS | 23.39 | 1.218 | 2.0534 | -17.78 | 0.0591 | 0.4186 | 0.3973 | 0.150118 | 0.180292 | 0.0 | 0.0 | 0.0 | 0.0 | -0.0056 | 0.0037 | 0.85 | 0.0243 | 0.0471 | -1.29 | BASELINE_TIMING | 0 | 0.0 | 0.0 | 0.0 | 0.0635 | 0.0287 | 0.0349 |
| MAHORAGA_15A3_LS | EXECUTION_DELAY_1_REBALANCE | delay long budget and both hedges one weekly rebalance | 24.63 | 1.2328 | 2.104 | -17.11 | 0.0586 | 0.4213 | 0.4027 | 0.162489 | 0.192666 | 1.24 | 0.0149 | 0.0506 | 0.66 | 0.0092 | 0.0543 | 2.09 | 0.0391 | 0.0977 | -0.05 | EXECUTION_DELAY | 1 | -0.0005 | 0.0027 | 0.0054 | 0.0637 | 0.0 | 0.0 |
| MAHORAGA_15A3_LS | EXECUTION_DELAY_2_REBALANCES | delay long budget and both hedges two weekly rebalances | 25.97 | 1.2922 | 2.234 | -20.97 | 0.0582 | 0.4116 | 0.3918 | 0.177205 | 0.207355 | 2.58 | 0.0742 | 0.1806 | -3.19 | 0.0686 | 0.1843 | 3.43 | 0.0985 | 0.2277 | 1.29 | EXECUTION_DELAY | 2 | -0.0009 | -0.007 | -0.0055 | 0.0617 | 0.0 | 0.0 |
| MAHORAGA_15A3_LS | REACTION_SLOWER | allocator reacts slower | 23.38 | 1.2172 | 2.0519 | -17.78 | 0.0589 | 0.4188 | 0.3975 | 0.149927 | 0.180104 | -0.02 | -0.0008 | -0.0015 | 0.0 | -0.0064 | 0.0022 | 0.84 | 0.0235 | 0.0456 | -1.31 | REACTION_SPEED | 0 | -0.0002 | 0.0002 | 0.0002 | 0.0635 | 0.0 | 0.0 |
| MAHORAGA_15A3_LS | REACTION_FASTER | allocator reacts faster | 23.42 | 1.2193 | 2.0557 | -17.78 | 0.0592 | 0.4186 | 0.3973 | 0.150383 | 0.180569 | 0.03 | 0.0013 | 0.0023 | 0.0 | -0.0043 | 0.0059 | 0.88 | 0.0256 | 0.0493 | -1.26 | REACTION_SPEED | 0 | 0.0001 | 0.0 | 0.0 | 0.0634 | 0.0 | 0.0 |
| MAHORAGA_15A3_LS | CRASH_LEAD_1_REBALANCE | shift crash hedge one rebalance earlier | 23.4 | 1.2194 | 2.0627 | -19.45 | 0.0589 | 0.421 | 0.4002 | 0.149588 | 0.179805 | 0.01 | 0.0014 | 0.0093 | -1.67 | -0.0042 | 0.013 | 0.86 | 0.0257 | 0.0564 | -1.28 | CRASH_LEAD_LAG | -1 | -0.0002 | 0.0024 | 0.0029 | 0.0608 | 0.0 | 0.0 |
| MAHORAGA_15A3_LS | CRASH_LAG_1_REBALANCE | shift crash hedge one rebalance later | 22.71 | 1.1857 | 1.9997 | -18.68 | 0.0588 | 0.4204 | 0.3995 | 0.143398 | 0.173441 | -0.69 | -0.0323 | -0.0537 | -0.9 | -0.0379 | -0.05 | 0.17 | -0.008 | -0.0066 | -1.98 | CRASH_LEAD_LAG | 1 | -0.0003 | 0.0018 | 0.0022 | 0.0642 | 0.0 | 0.0 |
| MAHORAGA_15A3_LS | CRASH_LAG_2_REBALANCES | shift crash hedge two rebalances later | 22.96 | 1.1967 | 2.0206 | -18.33 | 0.0585 | 0.4208 | 0.4001 | 0.145612 | 0.175703 | -0.44 | -0.0213 | -0.0328 | -0.55 | -0.0269 | -0.0292 | 0.42 | 0.003 | 0.0142 | -1.73 | CRASH_LEAD_LAG | 2 | -0.0006 | 0.0022 | 0.0028 | 0.0636 | 0.0 | 0.0 |
| MAHORAGA_15A3_LS | BEAR_LEAD_1_REBALANCE | shift bear hedge one rebalance earlier | 23.61 | 1.2277 | 2.0779 | -19.69 | 0.0586 | 0.422 | 0.4018 | 0.151325 | 0.181552 | 0.22 | 0.0097 | 0.0245 | -1.92 | 0.0041 | 0.0281 | 1.07 | 0.034 | 0.0715 | -1.07 | BEAR_LEAD_LAG | -1 | -0.0005 | 0.0034 | 0.0045 | 0.0614 | 0.0 | 0.0 |
| MAHORAGA_15A3_LS | BEAR_LAG_1_REBALANCE | shift bear hedge one rebalance later | 22.71 | 1.1849 | 2.0004 | -18.77 | 0.0589 | 0.4209 | 0.4003 | 0.143274 | 0.173296 | -0.69 | -0.0331 | -0.053 | -0.99 | -0.0387 | -0.0493 | 0.17 | -0.0088 | -0.0059 | -1.98 | BEAR_LEAD_LAG | 1 | -0.0002 | 0.0023 | 0.003 | 0.0631 | 0.0 | 0.0 |
| MAHORAGA_15A3_LS | BEAR_LAG_2_REBALANCES | shift bear hedge two rebalances later | 22.8 | 1.1898 | 2.0111 | -18.46 | 0.0589 | 0.4209 | 0.4003 | 0.144121 | 0.174162 | -0.59 | -0.0282 | -0.0423 | -0.68 | -0.0338 | -0.0387 | 0.26 | -0.0039 | 0.0047 | -1.88 | BEAR_LEAD_LAG | 2 | -0.0002 | 0.0023 | 0.003 | 0.0617 | 0.0 | 0.0 |

## 9. Hedge effectiveness / fail-fast
| Section | Metric | Value | Threshold | Passed | Detail |
| --- | --- | --- | --- | --- | --- |
| FAST_FAIL | GrossShortStitched | 0.0591 | 0.05 | True | GrossShort stitched must be real, not decorative |
| FAST_FAIL | GrossShortCrisisAvg | 0.063525 | 0.08 | False | crisis windows should show clearly higher short activity |
| FAST_FAIL | CrashHedgeActiveInCrashWindow | 0.0 | 0.03 | False | crash hedge must appear in crash windows |
| FAST_FAIL | BearHedgeActiveInBearWindow | 0.0312 | 0.03 | True | bear hedge must appear in bear windows |
| FAST_FAIL | ReactionStressMoves | 0.0013 | 0.0005 | True | reaction slower/faster must move the system materially |
| FAST_FAIL | HedgeRatioStressMoves | 0.0054 | 0.0005 | True | hedge-ratio stress must move the system materially |
| FAST_FAIL | TimingDelaySharpeImprovement | 0.0742 | 0.01 | False | delay +1/+2 should not improve Sharpe materially over baseline timing |
| FAST_FAIL | TimingDelayCAGRImprovementPct | 2.58 | 0.75 | False | delay +1/+2 should not improve CAGR materially over baseline timing |
| FAST_FAIL | CrashLeadLagSensitivityMoves | 0.0323 | 0.0005 | True | crash hedge lead/lag should move if crash sleeve is real |
| FAST_FAIL | BearLeadLagSensitivityMoves | 0.0331 | 0.0005 | True | bear hedge lead/lag should move if bear sleeve is real |
| FAST_FAIL | LSNotSameAsDeleveredControl | 1.0 | 1.0 | True | LS should not be almost identical to delevered control |
| FAST_FAIL | HedgePnLRelevant | 0.149549 | 0.0025 | True | combined hedge PnL should be non-trivial |
| FAST_FAIL | MicroscopicGainWithCAGRLoss | 1.0 | 1.0 | True | small risk gain cannot justify a large CAGR loss |
| FAST_FAIL | CrisisSeparationVsLongOnly | 1.0 | 1.0 | True | crisis windows should separate LS from long-only |
| FAST_FAIL | CrisisSeparationVsDelevered | 1.0 | 1.0 | True | crisis windows should separate LS from delevered control |
| SUCCESS_CHECK | SharpeVisibleDelta_vs_Long | 0.02429999999999999 | 0.0 | True | Sharpe stitched should not fall below long-only |
| SUCCESS_CHECK | SortinoVisibleDelta_vs_Long | 0.04709999999999992 | 0.0 | True | Sortino stitched should not fall below long-only |
| SUCCESS_CHECK | SharpeDelta_vs_Delevered | -0.005600000000000049 | 0.0 | False | LS should beat delevered control in Sharpe |
| SUCCESS_CHECK | SortinoDelta_vs_Delevered | 0.0036999999999998145 | 0.0 | True | LS should beat delevered control in Sortino |
| SUCCESS_CHECK | BetaQQQReduction_vs_Long | 0.06669999999999998 | 0.1 | False | beta vs QQQ should fall clearly |
| SUCCESS_CHECK | BetaSPYReduction_vs_Long | 0.07100000000000001 | 0.1 | False | beta vs SPY should fall clearly |
| SUCCESS_CHECK | CAGRDelta_vs_Long% | -1.2899999999999991 | -2.0 | True | CAGR should not be materially destroyed |
| SUCCESS_CHECK | AlphaNW_QQQ_Delta_vs_Long | 0.0007649999999999879 | -0.02 | True | benchmark-adjusted alpha vs QQQ should not worsen materially |
| SUCCESS_CHECK | AlphaNW_SPY_Delta_vs_Long | -0.00267199999999998 | -0.02 | True | benchmark-adjusted alpha vs SPY should not worsen materially |

## 10. Stress suite
| Variant | Scenario | ScenarioNote | CAGR% | Sharpe | Sortino | MaxDD% | GrossShort | BetaQQQ | BetaSPY | AlphaNW_QQQ | AlphaNW_SPY | DeltaCAGR_vs_LSBase% | DeltaSharpe_vs_LSBase | DeltaSortino_vs_LSBase | DeltaMaxDD_vs_LSBase% | DeltaSharpe_vs_Delevered | DeltaSortino_vs_Delevered | DeltaCAGR_vs_Delevered% | DeltaSharpe_vs_LongOnly | DeltaSortino_vs_LongOnly | DeltaCAGR_vs_LongOnly% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAHORAGA_14_1_LONG_ONLY | BASELINE_LONG_ONLY | official frozen 14.1 long-only reference | 24.68 | 1.1937 | 2.0063 | -20.24 | 0.0 | 0.4853 | 0.4683 | 0.149353 | 0.182964 | 1.29 | -0.0243 | -0.0471 | -2.46 | -0.0299 | -0.0434 | 2.14 | 0.0 | 0.0 | 0.0 |
| DELEVERED_CONTROL | BASELINE_DELEVERED_CONTROL | path-matched delevered control without real short | 22.54 | 1.2236 | 2.0497 | -15.49 | 0.0 | 0.4031 | 0.3849 | 0.144264 | 0.17274 | -0.85 | 0.0056 | -0.0037 | 2.28 | 0.0 | 0.0 | 0.0 | 0.0299 | 0.0434 | -2.14 |
| MAHORAGA_15A3_LS | BASELINE_15A3 | Mahoraga15A3 unstressed stitched OOS | 23.39 | 1.218 | 2.0534 | -17.78 | 0.0591 | 0.4186 | 0.3973 | 0.150118 | 0.180292 | 0.0 | 0.0 | 0.0 | 0.0 | -0.0056 | 0.0037 | 0.85 | 0.0243 | 0.0471 | -1.29 |
| MAHORAGA_15A3_LS | COST_PLUS_25 | commission/slippage x1.25 | 23.0 | 1.2009 | 2.0235 | -17.83 | 0.0591 | 0.4186 | 0.3973 | 0.146445 | 0.176524 | -0.39 | -0.0171 | -0.03 | -0.05 | -0.0227 | -0.0263 | 0.46 | 0.0072 | 0.0171 | -1.69 |
| MAHORAGA_15A3_LS | COST_PLUS_50 | commission/slippage x1.50 | 22.61 | 1.1838 | 1.9935 | -17.88 | 0.0591 | 0.4185 | 0.3972 | 0.142783 | 0.172767 | -0.79 | -0.0342 | -0.0599 | -0.1 | -0.0398 | -0.0562 | 0.07 | -0.0099 | -0.0128 | -2.08 |
| MAHORAGA_15A3_LS | COST_PLUS_100 | commission/slippage x2.00 | 21.82 | 1.1495 | 1.9336 | -17.97 | 0.0591 | 0.4185 | 0.3971 | 0.135495 | 0.165291 | -1.57 | -0.0685 | -0.1198 | -0.2 | -0.0741 | -0.1161 | -0.72 | -0.0442 | -0.0727 | -2.86 |
| MAHORAGA_15A3_LS | EXTRA_SLIPPAGE | extra slippage +5bps | 22.79 | 1.1917 | 2.0073 | -17.85 | 0.0591 | 0.4185 | 0.3972 | 0.144472 | 0.1745 | -0.61 | -0.0263 | -0.0461 | -0.08 | -0.0319 | -0.0424 | 0.25 | -0.002 | 0.001 | -1.9 |
| MAHORAGA_15A3_LS | EXECUTION_DELAY_1_REBALANCE | delay long budget and both hedges one weekly rebalance | 24.63 | 1.2328 | 2.104 | -17.11 | 0.0586 | 0.4213 | 0.4027 | 0.162489 | 0.192666 | 1.24 | 0.0149 | 0.0506 | 0.66 | 0.0092 | 0.0543 | 2.09 | 0.0391 | 0.0977 | -0.05 |
| MAHORAGA_15A3_LS | EXECUTION_DELAY_2_REBALANCES | delay long budget and both hedges two weekly rebalances | 25.97 | 1.2922 | 2.234 | -20.97 | 0.0582 | 0.4116 | 0.3918 | 0.177205 | 0.207355 | 2.58 | 0.0742 | 0.1806 | -3.19 | 0.0686 | 0.1843 | 3.43 | 0.0985 | 0.2277 | 1.29 |
| MAHORAGA_15A3_LS | HEDGE_RATIO_UNDERESTIMATED | hedges scaled to 75% | 23.28 | 1.2126 | 2.0443 | -17.85 | 0.0577 | 0.4202 | 0.399 | 0.148745 | 0.17896 | -0.11 | -0.0054 | -0.0091 | -0.08 | -0.011 | -0.0055 | 0.74 | 0.0189 | 0.0379 | -1.4 |
| MAHORAGA_15A3_LS | HEDGE_RATIO_OVERESTIMATED | hedges scaled to 125% | 23.47 | 1.2217 | 2.0595 | -17.78 | 0.0598 | 0.4179 | 0.3965 | 0.150998 | 0.181165 | 0.08 | 0.0037 | 0.0061 | 0.0 | -0.0019 | 0.0098 | 0.93 | 0.0279 | 0.0532 | -1.21 |
| MAHORAGA_15A3_LS | REACTION_SLOWER | allocator reacts slower | 23.38 | 1.2172 | 2.0519 | -17.78 | 0.0589 | 0.4188 | 0.3975 | 0.149927 | 0.180104 | -0.02 | -0.0008 | -0.0015 | 0.0 | -0.0064 | 0.0022 | 0.84 | 0.0235 | 0.0456 | -1.31 |
| MAHORAGA_15A3_LS | REACTION_FASTER | allocator reacts faster | 23.42 | 1.2193 | 2.0557 | -17.78 | 0.0592 | 0.4186 | 0.3973 | 0.150383 | 0.180569 | 0.03 | 0.0013 | 0.0023 | 0.0 | -0.0043 | 0.0059 | 0.88 | 0.0256 | 0.0493 | -1.26 |
| MAHORAGA_15A3_LS | CRASH_LEAD_1_REBALANCE | shift crash hedge one rebalance earlier | 23.4 | 1.2194 | 2.0627 | -19.45 | 0.0589 | 0.421 | 0.4002 | 0.149588 | 0.179805 | 0.01 | 0.0014 | 0.0093 | -1.67 | -0.0042 | 0.013 | 0.86 | 0.0257 | 0.0564 | -1.28 |
| MAHORAGA_15A3_LS | CRASH_LAG_1_REBALANCE | shift crash hedge one rebalance later | 22.71 | 1.1857 | 1.9997 | -18.68 | 0.0588 | 0.4204 | 0.3995 | 0.143398 | 0.173441 | -0.69 | -0.0323 | -0.0537 | -0.9 | -0.0379 | -0.05 | 0.17 | -0.008 | -0.0066 | -1.98 |
| MAHORAGA_15A3_LS | CRASH_LAG_2_REBALANCES | shift crash hedge two rebalances later | 22.96 | 1.1967 | 2.0206 | -18.33 | 0.0585 | 0.4208 | 0.4001 | 0.145612 | 0.175703 | -0.44 | -0.0213 | -0.0328 | -0.55 | -0.0269 | -0.0292 | 0.42 | 0.003 | 0.0142 | -1.73 |
| MAHORAGA_15A3_LS | BEAR_LEAD_1_REBALANCE | shift bear hedge one rebalance earlier | 23.61 | 1.2277 | 2.0779 | -19.69 | 0.0586 | 0.422 | 0.4018 | 0.151325 | 0.181552 | 0.22 | 0.0097 | 0.0245 | -1.92 | 0.0041 | 0.0281 | 1.07 | 0.034 | 0.0715 | -1.07 |
| MAHORAGA_15A3_LS | BEAR_LAG_1_REBALANCE | shift bear hedge one rebalance later | 22.71 | 1.1849 | 2.0004 | -18.77 | 0.0589 | 0.4209 | 0.4003 | 0.143274 | 0.173296 | -0.69 | -0.0331 | -0.053 | -0.99 | -0.0387 | -0.0493 | 0.17 | -0.0088 | -0.0059 | -1.98 |
| MAHORAGA_15A3_LS | BEAR_LAG_2_REBALANCES | shift bear hedge two rebalances later | 22.8 | 1.1898 | 2.0111 | -18.46 | 0.0589 | 0.4209 | 0.4003 | 0.144121 | 0.174162 | -0.59 | -0.0282 | -0.0423 | -0.68 | -0.0338 | -0.0387 | 0.26 | -0.0039 | 0.0047 | -1.88 |
| MAHORAGA_15A3_LS | ALLOCATOR_GATE_VOL_CAP_STRESS | tighter caps and lighter long budget | 22.14 | 1.2134 | 2.0495 | -17.06 | 0.0678 | 0.3976 | 0.377 | 0.141554 | 0.170048 | -1.25 | -0.0046 | -0.0039 | 0.72 | -0.0102 | -0.0002 | -0.4 | 0.0197 | 0.0432 | -2.55 |
| MAHORAGA_15A3_LS | EMPIRICAL_PATH_STRESS | inject worst empirical block into later path | 21.71 | 1.1312 | 1.8748 | -17.78 | 0.0591 | 0.4197 | 0.3978 | 0.134797 | 0.16475 | -1.68 | -0.0868 | -0.1786 | 0.0 | -0.0924 | -0.1749 | -0.83 | -0.0625 | -0.1315 | -2.97 |

## 11. Monte Carlo / bootstrap
| Method | Samples | MeanCAGR% | P5_CAGR% | P50_CAGR% | P95_CAGR% | MeanSharpe | P5_Sharpe | P50_Sharpe | P95_Sharpe | MeanSortino | P5_Sortino | P50_Sortino | P95_Sortino | MeanMaxDD% | P5_MaxDD% | P50_MaxDD% | P95_MaxDD% | Prob_Sharpe_lt_Baseline | Prob_MaxDD_worse_Baseline | Prob_CAGR_materially_worse_Baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stationary_block_bootstrap | 250 | 22.99 | 10.58 | 22.96 | 36.36 | 1.1916 | 0.6622 | 1.2098 | 1.7143 | 2.0237 | 1.0683 | 2.0594 | 3.0217 | -23.41 | -33.44 | -22.55 | -16.29 | 0.48 | 0.712 | 0.48 |
| friction_multiplier_mc | 250 | 23.35 | 22.57 | 23.37 | 24.14 | 1.2161 | 1.1822 | 1.2169 | 1.25 | 2.0501 | 1.9908 | 2.0515 | 2.1097 | -17.78 | -17.88 | -17.78 | -17.69 | 0.148 | 0.0 | 0.088 |
| local_param_neighborhood | 81 | 23.35 | 22.61 | 23.39 | 24.07 | 1.2163 | 1.2095 | 1.2166 | 1.2232 | 2.0504 | 2.0397 | 2.0505 | 2.0617 | -17.84 | -18.36 | -17.82 | -17.31 | 0.0 | 0.0 | 0.1358 |

## 12. Explicit failure flags
| Metric | Value | Threshold | Detail |
| --- | --- | --- | --- |
| GrossShortCrisisAvg | 0.063525 | 0.08 | crisis windows should show clearly higher short activity |
| CrashHedgeActiveInCrashWindow | 0.0 | 0.03 | crash hedge must appear in crash windows |
| TimingDelaySharpeImprovement | 0.0742 | 0.01 | delay +1/+2 should not improve Sharpe materially over baseline timing |
| TimingDelayCAGRImprovementPct | 2.58 | 0.75 | delay +1/+2 should not improve CAGR materially over baseline timing |