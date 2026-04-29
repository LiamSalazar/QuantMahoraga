# Evidencia utilizada para el paper

## Codigo
1. `D:\QuantMahoraga\final_model\mahoraga14\mahoraga14_runner.py`
2. `D:\QuantMahoraga\final_model\mahoraga14\mahoraga14_config.py`
3. `D:\QuantMahoraga\final_model\mahoraga14\mahoraga14_data.py`
4. `D:\QuantMahoraga\final_model\mahoraga14\mahoraga14_universe.py`
5. `D:\QuantMahoraga\final_model\mahoraga14\base_alpha_engine.py`
6. `D:\QuantMahoraga\final_model\mahoraga14\backtest_executor.py`
7. `D:\QuantMahoraga\final_model\mahoraga14\override_policy.py`
8. `D:\QuantMahoraga\final_model\mahoraga14\continuation_v2_model.py`
9. `D:\QuantMahoraga\final_model\mahoraga14\structural_defense_model.py`
10. `D:\QuantMahoraga\final_model\mahoraga14\transition_recovery_model.py`
11. `D:\QuantMahoraga\final_model\mahoraga14\full_report.py`
12. `D:\QuantMahoraga\final_model\mahoraga14\fast_report.py`
13. `D:\QuantMahoraga\final_model\mahoraga14\mahoraga6_1.py`
14. `D:\QuantMahoraga\final_model\mahoraga14\README_Mahoraga14.md`

## Outputs FULL auditados
1. `D:\QuantMahoraga\mahoraga14_outputs\final_report_full.md`
2. `D:\QuantMahoraga\mahoraga14_outputs\stitched_comparison_full.csv`
3. `D:\QuantMahoraga\mahoraga14_outputs\fold_summary_full.csv`
4. `D:\QuantMahoraga\mahoraga14_outputs\floor_ceiling_summary_full.csv`
5. `D:\QuantMahoraga\mahoraga14_outputs\pvalue_qvalue_full.csv`
6. `D:\QuantMahoraga\mahoraga14_outputs\alpha_nw_full.csv`
7. `D:\QuantMahoraga\mahoraga14_outputs\selected_candidates_full.csv`
8. `D:\QuantMahoraga\mahoraga14_outputs\selected_candidate_audit_full.csv`
9. `D:\QuantMahoraga\mahoraga14_outputs\selected_config_support_full.csv`
10. `D:\QuantMahoraga\mahoraga14_outputs\stitched_full_trace.csv`
11. `D:\QuantMahoraga\mahoraga14_outputs\stitched_integrity_checks.csv`
12. `D:\QuantMahoraga\mahoraga14_outputs\equity_stitched_full.csv`
13. `D:\QuantMahoraga\mahoraga14_outputs\drawdown_stitched_full.csv`
14. `D:\QuantMahoraga\mahoraga14_outputs\maxdd_audit_full.csv`
15. `D:\QuantMahoraga\mahoraga14_outputs\maxdd_audit_notes.md`
16. `D:\QuantMahoraga\mahoraga14_outputs\stress_suite_full.csv`
17. `D:\QuantMahoraga\mahoraga14_outputs\stress_suite_audit.md`
18. `D:\QuantMahoraga\mahoraga14_outputs\stress_paths_trace.csv`
19. `D:\QuantMahoraga\mahoraga14_outputs\montecarlo_summary_full.csv`
20. `D:\QuantMahoraga\mahoraga14_outputs\montecarlo_percentiles_full.csv`
21. `D:\QuantMahoraga\mahoraga14_outputs\montecarlo_audit.md`
22. `D:\QuantMahoraga\mahoraga14_outputs\continuation_event_study_full.csv`
23. `D:\QuantMahoraga\mahoraga14_outputs\continuation_calibration_full.csv`
24. `D:\QuantMahoraga\mahoraga14_outputs\override_usage_full.csv`
25. `D:\QuantMahoraga\mahoraga14_outputs\continuation_usage_full.csv`
26. `D:\QuantMahoraga\mahoraga14_outputs\full_readiness_checklist.csv`
27. `D:\QuantMahoraga\mahoraga14_outputs\full_readiness_summary.md`

## Figuras auditadas reutilizadas en el manuscrito
1. `D:\QuantMahoraga\mahoraga14_outputs\figures\equity_curve_stitched_full.png`
2. `D:\QuantMahoraga\mahoraga14_outputs\figures\drawdown_curve_stitched_full.png`
3. `D:\QuantMahoraga\mahoraga14_outputs\figures\fold_heatmap_full.png`
4. `D:\QuantMahoraga\mahoraga14_outputs\figures\continuation_event_study_full.png`
5. `D:\QuantMahoraga\mahoraga14_outputs\figures\montecarlo_distribution_full.png`

## Modulos explicitamente excluidos de la descripcion final del sistema
1. Clasificadores standalone `fit_transition_model` y `fit_recovery_model` de `transition_recovery_model.py`.
2. Capa Markov.
3. `ML RegimeGate` historico de `mahoraga6_1.py`.
4. Ramas long-short.
5. Deep learning, reinforcement learning y componentes no ejecutados por el pipeline `mahoraga14_runner.py -> backtest_executor.py -> full_report.py`.
