# Outline del paper

## Titulo propuesto
Mahoraga 14.1: consolidacion auditada de un sistema long-only con alpha residualizado, overrides estructurales y continuation lift bajo validacion walk-forward

## Tesis central
Mahoraga 14.1 no introduce una arquitectura nueva; formaliza una consolidacion auditada de Mahoraga 14 cuyo objetivo es verificar que el stitched OOS, el MaxDD, el stress suite y el Monte Carlo del modo FULL sean metodologicamente correctos y produzcan evidencia util para analisis tecnico y paper.

## Pregunta de investigacion
Determinar si la implementacion final de Mahoraga 14.1 conserva una mejora stitched verificable sobre la base fuerte `BASE_ALPHA_V2`, si esa mejora resiste una auditoria FULL de integridad y si los resultados frente a `LEGACY`, `QQQ` y `SPY` siguen siendo consistentes una vez corregido el riesgo historico de stitched OOS mal armado.

## Hipotesis que sobreviven en 14.1
1. `BASE_ALPHA_V2` es la base fuerte y el benchmark interno correcto para la comparacion principal.
2. `CONTINUATION_PRESSURE_V2_ONLY` produce una mejora stitched pequena pero real sobre `BASE_ALPHA_V2`.
3. `STRUCTURAL_DEFENSE_ONLY` ayuda mas al suelo que al techo, pero no domina como rama principal agregada.
4. El alpha Newey-West stitched frente a `QQQ` y `SPY` permanece positivo.
5. La utilidad de 14.1 depende tanto de la auditoria del pipeline como del nivel puntual de desempeno.

## Hipotesis descartadas o no promovidas en 14.1
1. `STRUCTURAL_DEFENSE_ONLY` como rama principal agregada.
2. Un modo FULL combinatorio amplio como motor de descubrimiento.
3. La inclusion operativa de clasificadores standalone de transicion/recuperacion, capa Markov, ML RegimeGate, long-short o tecnicas de deep learning/RL.
4. La presentacion de Hawkes como un modelo puntual estimado; en la implementacion final solo sobrevive una senal Hawkes-style de intensidad decaida.

## Estructura
1. **Titulo**
   - version, objetivo y alcance auditado.
2. **Resumen**
   - objetivo, arquitectura final, protocolo FULL, hallazgos principales y limites.
3. **Introduccion**
   - problema, necesidad de evidencia OOS confiable y riesgo historico de stitched defectuoso.
4. **Evolucion del proyecto y antecedentes de investigacion**
   - paso desde la base historica modularizada hacia Mahoraga 14 y consolidacion auditada 14.1.
   - modulos que sobreviven y modulos que quedan fuera de la descripcion final.
5. **Datos, universo y preparacion de informacion**
   - datos OHLCV, benchmarks, frecuencia diaria/semanal, universo canonico PIT por `LiqSizeProxy`, limites de sesgo de supervivencia y nota sobre factores FF cargados pero no usados en FULL.
6. **Arquitectura del sistema**
   - motor base, rutas primaria/defensiva, overrides semanales, ejecucion diaria y capa de auditoria FULL.
7. **Metodologia**
   - calibracion FAST/FULL dirigida, seleccion de candidatos, comparaciones, p-values/q-values y alpha HAC.
8. **Construccion del motor de alpha**
   - `BASE_ALPHA_V2`: core legado, capa adaptiva residualizada, residualizacion sobre `QQQ`, `SPY`, `TECH`, `PC1`, mezcla final y score estandarizado.
9. **Construccion de cartera y ejecucion**
   - seleccion top-3, HRP, rebalanceo semanal, stops, shift de ejecucion, costos, reconstruccion de equity y exposicion.
10. **Gestion de riesgo y reglas de salida**
   - objetivo de volatilidad, `crisis_scale`, `turb_scale`, veto secundario de correlacion, `gate_scale`, `vol_mult`, `exp_cap`, drawdown.
11. **Overrides y capa superior**
   - `STRUCTURAL_DEFENSE_ONLY`, `CONTINUATION_PRESSURE_V2_ONLY`, combinacion y rol real de `break-risk`.
12. **Backtesting y protocolo walk-forward**
   - cinco folds expandentes, embargo, ventanas de validacion/test y benchmarks obligatorios.
13. **Auditoria FULL: stitched, MaxDD, stress tests y Monte Carlo**
   - trazabilidad por fold, integridad de stitched, recomputacion independiente de MaxDD, stress suite y stationary block bootstrap.
14. **Resultados**
   - resultados stitched, comparacion contra `LEGACY`, `QQQ`, `SPY`, alpha Newey-West, piso/techo, event study y sensibilidad.
15. **Limitaciones**
   - universo por `LiqSizeProxy`, numero bajo de activaciones continuation, q-values no concluyentes, stress path no exhaustivo.
16. **Conclusiones**
   - 14.1 como consolidacion auditada, no como expansion de arquitectura.

## Tablas clave
1. Comparacion stitched final de `LEGACY`, `QQQ`, `SPY`, `BASE`, `CONT` y `COMBO`.
2. Resumen de seleccion piso/techo y estatus de ramas.
3. Auditoria stitched y MaxDD.
4. Stress suite del candidato primario.
5. Monte Carlo/bootstrap del candidato primario.
6. Event study de continuation.

## Figuras esenciales
1. `equity_curve_stitched_full`
2. `drawdown_curve_stitched_full`
3. `fold_heatmap_full`
4. `continuation_event_study_full`
5. `montecarlo_distribution_full`

## Base documental y empirica a citar
1. Codigo final de `base_alpha_engine.py`, `override_policy.py`, `continuation_v2_model.py`, `structural_defense_model.py`, `backtest_executor.py`, `full_report.py`, `mahoraga14_config.py`, `mahoraga14_data.py`, `mahoraga6_1.py`, `README_Mahoraga14.md`.
2. Outputs auditados FULL en `mahoraga14_outputs/`.
