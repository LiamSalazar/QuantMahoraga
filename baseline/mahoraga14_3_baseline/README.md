# Mahoraga14_3 Baseline

Frozen, self-contained official long-only baseline.

## Official Freeze

- official baseline: `MAHORAGA14_3_BASELINE_OFFICIAL`
- frozen reference: `Mahoraga14_3R / ROBUST_MAIN / B1.05_C1.10_L1.10_R1.05`
- replaced baseline: `Mahoraga14_1_LONG_ONLY_CONTROL`

## Local Structure

- `src/`: self-contained baseline source code.
- `config/`: parameter freeze files.
- `outputs/`: official performance, sensitivity, and figure outputs.
- `audit/`: acceptance, robustness, continuation, and diagnostic artifacts.
- `paper_pack/`: paper-ready tables, figures, and supported claims.
- `docs/`: freeze notes, decision flow, model card, and robustness documentation.
- `manifests/`: manifests and provenance files.
- `scripts/`: reproducible runners from the repository root.
- `tests/`: minimal import, pathing, and freeze tests.

## How to Run

```powershell
cd D:\QuantMahoraga
python .\baseline\mahoraga14_3_baseline\scripts\run_official_baseline.py
```

## What This Folder Is Not

- It is not a discovery branch.
- It is not the research workspace.
- It does not include short sleeves or hedge sleeves.
- It does not redefine the scientific thesis; it only freezes the promoted baseline.
