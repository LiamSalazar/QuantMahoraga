# Mahoraga14_3 Baseline

Baseline long-only oficial, congelado y autocontenido.

## Freeze oficial

- baseline oficial: `MAHORAGA14_3_BASELINE_OFFICIAL`
- referencia congelada: `Mahoraga14_3R / ROBUST_MAIN / B1.05_C1.10_L1.10_R1.05`
- baseline reemplazado: `Mahoraga14_1_LONG_ONLY_CONTROL`

## Estructura local

- `src/`: código autocontenido del baseline.
- `config/`: freeze de parámetros.
- `outputs/`: performance, sensibilidad y figuras oficiales.
- `audit/`: aceptación, robustez, continuidad y diagnósticos.
- `paper_pack/`: tablas, figuras y claims listos para paper.
- `docs/`: freeze, decision flow, model card y robustez.
- `manifests/`: manifests y provenance.
- `scripts/`: runners reproducibles desde la raíz del repo.
- `tests/`: pruebas mínimas de imports, pathing y freeze.

## Cómo correr

```powershell
cd D:\QuantMahoraga
python .\baseline\mahoraga14_3_baseline\scripts\run_official_baseline.py
```

## Qué no es esta carpeta

- no es una rama de discovery
- no es la zona de research
- no incluye shorts ni sleeves de hedge
- no redefine la tesis científica; solo congela la promovida
