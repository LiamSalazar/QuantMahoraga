# Questions Answered

| Question | DSS Surface | Evidence |
|---|---|---|
| Que tan robusto es el candidato oficial? | Robustness Surface, Scorecard | `fact_robustness_surface`, `fact_candidate_metric` |
| Que pasa en cada fold? | Overview, Fold Performance | `fact_outcome`, `dim_fold` |
| Que modulos ayudan? | Module Lab | `fact_module_trace`, `fact_outcome` |
| Que tickers explican el resultado? | Ticker Contribution | `fact_position_daily` |
| Que decisiones explican drawdowns? | Decision Replay, Drawdown Replay | `fact_decision_state`, `fact_path_recursive`, `fact_position_daily` |
| Que pasa si cambian costos? | What-if Lab | `fact_whatif`, `fact_cost_sensitivity` |
| Que pasa si cambian multiplicadores? | What-if Lab, Robustness Surface | `fact_whatif`, `fact_robustness_surface` |
| Que pasa por regimen? | Regime Lab | `fact_decision_state`, `fact_outcome` |
| Como se degrada al cambiar universo? | Candidate Compare | `fact_universe_sensitivity`, `fact_candidate_metric` |
| Que escenarios mantienen buen tradeoff? | What-if Lab | `fact_whatif` |

The query registry in `api/query_registry.py` exposes these questions to the API through `GET /metadata/questions`.

