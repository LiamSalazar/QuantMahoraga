# QuantMahoraga

Repositorio institucional organizado en tres capas:

- `baseline/`: baseline oficial congelado y reproducible.
- `research/`: ramas históricas y experimentales archivadas/documentadas.
- `shared/`: infraestructura compartida de pathing y futuros módulos comunes.

## Baseline oficial

La carpeta oficial es [baseline/mahoraga14_3_baseline](/D:/QuantMahoraga/baseline/mahoraga14_3_baseline) y congela:

- `Mahoraga14_3R / ROBUST_MAIN / B1.05_C1.10_L1.10_R1.05`
- arquitectura long-only de `14.3R`
- baseline histórico retenido como control documental: `Mahoraga14_1_LONG_ONLY_CONTROL`

## Cómo correr el baseline oficial

```powershell
cd D:\QuantMahoraga
python .\baseline\mahoraga14_3_baseline\scripts\run_official_baseline.py
```

## Estructura rápida

- [baseline/mahoraga14_3_baseline](/D:/QuantMahoraga/baseline/mahoraga14_3_baseline): paquete oficial, outputs, audit, paper pack, docs y manifests.
- [research](/D:/QuantMahoraga/research): archivo curado de ramas no oficiales, con `README.md` y `status.md` por carpeta.
- [shared/pathing](/D:/QuantMahoraga/shared/pathing): resolución central de rutas.
- [docs](/D:/QuantMahoraga/docs): overview, metodología y governance.

## Regla operativa

No se hace discovery dentro del baseline oficial. Cualquier cambio nuevo debe nacer en `research/` y pasar por acceptance/promotion antes de contaminar `baseline/`.
