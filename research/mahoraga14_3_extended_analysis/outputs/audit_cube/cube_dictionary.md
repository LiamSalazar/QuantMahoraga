# Audit Cube Dictionary

## Design decision: representative granular subset

The full granular `position_cube` and `module_trace_cube` are limited to 6 representative candidates: B1.05_C1.10_L1.10_R1.05, EXTREME_pro-risk, EXTREME_pro-defense, B0.90_C1.10_L1.10_R1.05, B1.05_C1.10_L0.90_R1.05, B1.05_C0.90_L1.10_R1.05.

This is intentional. The extended multiplier sweep is an audit sample, not a production portfolio catalogue. Building ticker-date-module traces for every perturbation would make the frontend slower, increase storage, and add little audit value beyond the candidates that define the official point, controlled extremes, and most sensitive directions.

## Common metadata

All cube tables include `run_id`, `analysis_phase`, `candidate_id`, `universe_id`, `baseline_reference`, and `generated_at`.

## decision_date_cube.parquet

One row per date/fold/candidate/universe. It records allocator state, participation state, gate/vol/exp controls, continuation/backoff/leader signals, expected exposure and turnover.

## position_cube.parquet

One row per date/ticker/fold/candidate/universe for representative candidates. It records scores, ranks, weights, stop flags, forward returns, and PnL contribution. Some component-level raw fields are nullable when the frozen snapshot exposes only composite scores at that layer.

## module_trace_cube.parquet

One row per date/module/candidate/fold/universe. JSON summaries capture structured main inputs/outputs for audit without forcing a brittle schema for every module internals.

## outcome_cube.parquet

One row per decision date/horizon/candidate/fold/universe. `decision_helped_flag_vs_qqq` equals 1 when realized system return over the horizon exceeds QQQ. `decision_helped_flag_vs_control` equals 1 when realized system return exceeds the official historical control.

## market_context_cube.parquet

One row per date with QQQ/SPY returns, drawdowns, realized vol, breadth and benchmark strength/weakness proxies from the frozen allocator path.

## Limitations

- Cubes inherit the granularity exposed by the frozen 14.3R snapshot.
- Candidate perturbations alter returns/exposure through the frozen multiplier layer; module-state traces remain anchored to the frozen policy path.
- Nullable fields are explicit rather than imputed when the source module does not expose a stable primitive.