# Mahoraga 14_3R Promotion Gate

Mahoraga 14_3R promotion gate is a **final, fast, disciplined promotion check** over the frozen long-only architecture.

## Scope

- No new thesis
- No new layers
- No shorts
- No rediscovery grid
- Only a small candidate gate over already accepted knobs

## Candidate set

- `B1.00_C1.00_L1.00_R1.00`
- `B1.05_C1.10_L1.10_R1.05`
- `B1.05_C1.10_L1.10_R1.00`
- `B1.05_C1.00_L1.10_R1.05`
- `B1.05_C1.10_L1.00_R1.05`

## Architecture under gate

1. `BASE_ALPHA_V2`
2. `PARTICIPATION_ALLOCATOR_V2`
3. `CONVICTION_AMPLIFIER_LAYER`
4. `LEADER_PARTICIPATION_LAYER`
5. `RISK_BACKOFF_LAYER_V2`
6. continuation as a quality filter

## Files

- `promotion_gate_suite.py`: stitched, folds, bull windows, priority gate, active return, alpha/pq and promotion decision.
- `mahoraga14_3R_promotion_gate_runner.py`: direct FAST runner for the promotion gate.
- `mahoraga14_config.py`: outputs and labels for the promotion-gate phase.

## Run

```bash
python mahoraga14_3R_promotion_gate_runner.py
```

## Outputs

Artifacts are written to `D:\QuantMahoraga\mahoraga14_3R_promotion_gate_outputs`:

- `stitched_comparison_promotion_gate.csv`
- `fold_summary_promotion_gate.csv`
- `bull_window_scorecard_promotion_gate.csv`
- `priority_window_acceptance_promotion_gate.csv`
- `active_return_vs_qqq_promotion_gate.csv`
- `alpha_nw_promotion_gate.csv`
- `pvalue_qvalue_promotion_gate.csv`
- `promotion_gate_decision.md`
- `final_report_promotion_gate.md`
