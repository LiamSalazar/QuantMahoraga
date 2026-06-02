# Plateau Radius Report

## Formal definitions

Let the official multiplier vector be m0 = (1.05, 1.10, 1.10, 1.05). For any sampled candidate m, define relative perturbation d(m,m0)=max_i |m_i/m0_i - 1|.

`distance_to_decay` is min d(m,m0) over sampled candidates where at least one decay condition holds: Sharpe drop > 10%, CAGR drop > 10%, MaxDD worsening > 5 percentage points, or severe_fold_damage_count > 0.

`plateau_radius` is computed per axis by holding the other three multipliers at the official values and taking the sampled interval around the official value where all robustness conditions hold.

`robust_region_share_extended` is the count of sampled candidates satisfying all robustness conditions divided by the total sampled candidate count.

The 10% relative Sharpe/CAGR and 5 percentage point MaxDD thresholds are not new optimization criteria; they are stress-audit tolerances requested for this phase and interpreted as degradation boundaries, not promotion rules.

## Results

- distance_to_decay: 0.0476
- robust_region_share_extended: 64.29%
- sampled candidates: 42

## Plateau by axis

```
                 axis  official_value  robust_min_sampled_value  robust_max_sampled_value  plateau_radius_relative  plateau_radius_absolute_low  plateau_radius_absolute_high
    budget_multiplier            1.05                      1.05                      1.15                 0.000000                         0.00                          0.10
conviction_multiplier            1.10                      0.90                      1.30                 0.181818                         0.20                          0.20
    leader_multiplier            1.10                      0.90                      1.30                 0.181818                         0.20                          0.20
     backoff_strength            1.05                      0.90                      1.20                 0.142857                         0.15                          0.15
```

## Sensitivity ranking

```
                 axis  sensitivity_score  mean_sensitivity_score      worst_candidate_id  worst_sharpe_drop  worst_cagr_drop  worst_maxdd_worsening  worst_severe_fold_damage_count  official_value             sampled_values
    budget_multiplier           5.567113                1.824686 B0.90_C1.10_L1.10_R1.05           0.189925         0.294192               0.414981                               5            1.05 0.9,0.95,1.0,1.05,1.1,1.15
    leader_multiplier           0.177613                0.065420 B1.05_C1.10_L0.90_R1.05           0.050215         0.073997               0.267008                               0            1.10        0.9,1.0,1.1,1.2,1.3
     backoff_strength           0.150283                0.049660 B1.05_C1.10_L1.10_R0.90           0.029285         0.018538               0.512302                               0            1.05       0.9,1.0,1.05,1.1,1.2
conviction_multiplier           0.075705                0.028627 B1.05_C0.90_L1.10_R1.05           0.029373         0.045196               0.005678                               0            1.10        0.9,1.0,1.1,1.2,1.3
```

## Interpretation guardrails

- These metrics describe stability in the sampled perturbation set only.
- A large plateau does not prove global optimality.
- A small distance_to_decay does not automatically invalidate the baseline; it identifies where additional audit attention is needed.
- Worst-fold degradation is treated as a first-class risk because stitched performance can hide fold-local fragility.