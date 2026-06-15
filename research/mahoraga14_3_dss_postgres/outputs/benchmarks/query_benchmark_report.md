# Query Benchmark Report

| Query | Planning ms | Execution ms | Scans | Relations |
| --- | ---: | ---: | --- | --- |
| command_center_scorecard | 0.618 | 0.182 | Seq Scan | mv_scorecard_candidate |
| decision_replay | 2.646 | 1.274 | Index Scan | fact_decision_state,fact_outcome_h20 |
| whatif_grid | 0.517 | 1.858 | Seq Scan | mv_whatif_grid |
