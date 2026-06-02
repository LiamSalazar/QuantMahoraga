# Universe Robustness Report

## Methodology

Each alternate universe is treated as an input-universe stress, not a new baseline search. The official multiplier vector is evaluated first; at most two controlled extremes are added to test whether risk-seeking or defensive perturbations explain major degradation.

The negative control is non-technology by design and is interpreted only as a technical sanity check, not as evidence for or against an economic technology edge.

## Coverage summary

```
             universe_id  proposed  usable  mean_coverage
        base_universe_12        12      12       0.952753
negative_control_nontech        16      16       1.000000
                 tech_20        20      20       0.953903
         tech_plus_semis        23      23       0.975349
   wider_largecap_growth        24      24       0.932037
```

## Official candidate by universe

```
          universe_id      CAGR   Sharpe  Sortino      MaxDD  AlphaNW_QQQ  AlphaNW_SPY run_status
     base_universe_12 32.551849 1.482588 2.527954 -16.199729     0.214661     0.250536         OK
              tech_20 29.957619 1.380757 2.380046 -19.482867     0.190694     0.225178         OK
      tech_plus_semis 30.917935 1.371710 2.326986 -19.694647     0.199619     0.236136         OK
wider_largecap_growth 24.932976 1.203062 2.054226 -21.166835     0.153920     0.185237         OK
```

## Run metadata

```
             universe_id  coverage_cached  wf_cached  seconds                 status                                                      reason
        base_universe_12             True       True 1.234739                     ok                                                         NaN
                 tech_20             True       True 1.445576                     ok                                                         NaN
         tech_plus_semis             True       True 1.335681                     ok                                                         NaN
   wider_largecap_growth             True       True 1.240807                     ok                                                         NaN
negative_control_nontech             True      False 0.066849 aborted_compute_budget max_new_universe_runs=0 reached after observed high runtime
```

## Interpretation

- A collapse in the negative control is expected and should not be read as a failed technology edge.
- Changes across tech universes can arise from composition, seasoning, volatility, and canonical schedule membership, not only from model failure.
- Alternate universe runs reuse the same policy-layer architecture and official multipliers; they are not reoptimized for each universe.
- Coverage gaps are first-class limitations and are not hidden by forced backfills.