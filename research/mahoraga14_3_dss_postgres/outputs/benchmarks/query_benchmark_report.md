# Query Benchmark Report

| Query | Planning ms | Execution ms | Scans | Relations |
| --- | ---: | ---: | --- | --- |
| command_center_scorecard | 0.499 | 0.191 | Seq Scan | mv_scorecard_candidate |
| decision_replay | 2.354 | 1.212 | Index Scan | fact_decision_state,fact_outcome_h20 |
| whatif_grid | 0.466 | 1.543 | Seq Scan | mv_whatif_grid |
