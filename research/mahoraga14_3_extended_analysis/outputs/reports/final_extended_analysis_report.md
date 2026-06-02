# Final Extended Analysis Report

- run_id: `ext14_3_20260602T082114Z0000`
- baseline reference: `Mahoraga14_3R / ROBUST_MAIN / B1.05_C1.10_L1.10_R1.05`

## Objective

This research phase audits the frozen Mahoraga14_3 baseline for extended multiplier robustness, universe dependence, and granular decision traceability. It does not define a new baseline and does not reoptimize the official candidate.

## Theoretical foundation

The robustness analysis treats parameter perturbation as local stability testing rather than search. The core degradation checks combine risk-adjusted return, growth, drawdown control, benchmark-adjusted alpha and fold-local damage because financial backtests can look strong in stitched aggregate while hiding path-local instability.

Newey-West alpha is retained because daily strategy returns can exhibit serial correlation and heteroskedasticity. Fold degradation is explicitly measured because walk-forward validation is the relevant out-of-sample unit for this architecture.

## Computational design

- Baseline walk-forward is run or loaded inside this research phase.
- One-dimensional sweeps identify sensitive axes.
- Only the two most sensitive axes receive a two-dimensional sweep.
- Controlled extremes test boundary behavior without becoming optimization candidates.
- Full granular cubes are limited to representative candidates for auditability and usability.

## Multiplier robustness

- Official CAGR: 32.5518%
- Official Sharpe: 1.4826
- Official Sortino: 2.5280
- Official MaxDD: -16.1997%
- distance_to_decay: 0.0476
- robust_region_share_extended: 64.29%
- sampled candidates: 42

## Sensitivity ranking

```
                 axis  sensitivity_score  mean_sensitivity_score      worst_candidate_id  worst_sharpe_drop  worst_cagr_drop  worst_maxdd_worsening  worst_severe_fold_damage_count  official_value             sampled_values
    budget_multiplier           5.567113                1.824686 B0.90_C1.10_L1.10_R1.05           0.189925         0.294192               0.414981                               5            1.05 0.9,0.95,1.0,1.05,1.1,1.15
    leader_multiplier           0.177613                0.065420 B1.05_C1.10_L0.90_R1.05           0.050215         0.073997               0.267008                               0            1.10        0.9,1.0,1.1,1.2,1.3
     backoff_strength           0.150283                0.049660 B1.05_C1.10_L1.10_R0.90           0.029285         0.018538               0.512302                               0            1.05       0.9,1.0,1.05,1.1,1.2
conviction_multiplier           0.075705                0.028627 B1.05_C0.90_L1.10_R1.05           0.029373         0.045196               0.005678                               0            1.10        0.9,1.0,1.1,1.2,1.3
```

## Plateau radius

```
                 axis  official_value  robust_min_sampled_value  robust_max_sampled_value  plateau_radius_relative  plateau_radius_absolute_low  plateau_radius_absolute_high
    budget_multiplier            1.05                      1.05                      1.15                 0.000000                         0.00                          0.10
conviction_multiplier            1.10                      0.90                      1.30                 0.181818                         0.20                          0.20
    leader_multiplier            1.10                      0.90                      1.30                 0.181818                         0.20                          0.20
     backoff_strength            1.05                      0.90                      1.20                 0.142857                         0.15                          0.15
```

## Worst-fold degradation

```
            CandidateId                                   sweep_role  worst_fold_sharpe_delta_vs_official  worst_fold_cagr_delta_vs_official  max_fold_maxdd_worsening_vs_official  severe_fold_damage_count
        EXTREME_all-low                           CONTROLLED_EXTREME                            -0.479528                         -33.120994                              2.919118                         5
    EXTREME_pro-defense                           CONTROLLED_EXTREME                            -0.400291                         -33.532032                              1.249386                         5
B0.90_C1.10_L0.90_R1.05 TWO_DIM_budget_multiplier__leader_multiplier                            -0.379352                         -30.481606                              1.928401                         5
B0.90_C1.10_L1.00_R1.05 TWO_DIM_budget_multiplier__leader_multiplier                            -0.335199                         -27.853381                              1.666171                         5
B0.90_C1.10_L1.10_R1.05                    ONE_DIM_budget_multiplier                            -0.295997                         -25.337204                              1.451006                         5
B0.95_C1.10_L0.90_R1.05 TWO_DIM_budget_multiplier__leader_multiplier                            -0.270033                         -22.573698                              1.394845                         4
B0.90_C1.10_L1.20_R1.05 TWO_DIM_budget_multiplier__leader_multiplier                            -0.257292                         -22.786623                              1.235343                         3
B0.95_C1.10_L1.00_R1.05 TWO_DIM_budget_multiplier__leader_multiplier                            -0.227847                         -19.831280                              1.131171                         2
B0.90_C1.10_L1.30_R1.05 TWO_DIM_budget_multiplier__leader_multiplier                            -0.219075                         -20.201176                              1.019180                         2
B0.95_C1.10_L1.10_R1.05                    ONE_DIM_budget_multiplier                            -0.190470                         -17.205974                              0.914855                         2
```

## Universe robustness

```
          universe_id      CAGR   Sharpe  Sortino      MaxDD  AlphaNW_QQQ  AlphaNW_SPY  usable_count run_status
     base_universe_12 32.551849 1.482588 2.527954 -16.199729     0.214661     0.250536            12         OK
              tech_20 29.957619 1.380757 2.380046 -19.482867     0.190694     0.225178            20         OK
      tech_plus_semis 30.917935 1.371710 2.326986 -19.694647     0.199619     0.236136            23         OK
wider_largecap_growth 24.932976 1.203062 2.054226 -21.166835     0.153920     0.185237            24         OK
```

## Limitations

- Extended samples do not prove global parameter stability outside sampled ranges.
- Alternate universes mix economic composition with data coverage and canonical schedule effects.
- The decision audit cube exposes the stable fields available in the frozen snapshot; nullable fields indicate unavailable primitives rather than inferred values.
- Candidate perturbation is applied through the frozen multiplier layer, so granular module traces remain anchored to the official policy path.

## Open risks

- Universe robustness can be sensitive to ticker seasoning and corporate history.
- Large controlled extremes may test unrealistic operating regimes.
- Future data could move both fold-local and universe-local conclusions.

## Conclusion

The generated outputs should be read as an independent audit layer: they clarify where the official point remains stable, where degradation first appears, and which decisions can be reconstructed date by date. They do not alter the official freeze.