# Query Benchmark Report

| Query | Planning ms | Execution ms | Scans | Relations |
| --- | ---: | ---: | --- | --- |
| command_center_scorecard | 0.609 | 0.174 | Seq Scan | mv_scorecard_candidate |
| decision_replay | 2.699 | 1.224 | Index Scan | fact_decision_state,fact_outcome_h20 |
| whatif_grid | 0.491 | 2.222 | Seq Scan | mv_whatif_grid |
