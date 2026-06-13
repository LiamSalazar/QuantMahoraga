EXPLAIN (ANALYZE, BUFFERS)
SELECT candidate_id, universe_id, sharpe, cagr, maxdd, any_robust_region
FROM mart.mv_scorecard_candidate
ORDER BY sharpe DESC
LIMIT 20;

EXPLAIN (ANALYZE, BUFFERS)
SELECT fold, horizon, avg_alpha_vs_qqq, helped_rate, observations
FROM mart.mv_performance_by_fold
WHERE candidate_id = 'B1.05_C1.10_L1.10_R1.05'
ORDER BY fold, horizon;

EXPLAIN (ANALYZE, BUFFERS)
SELECT budget_multiplier, conviction_multiplier, metric_value, robust_score
FROM mart.mv_robustness_surface
WHERE metric_name = 'Sharpe'
  AND universe_id = 'base_universe_12'
ORDER BY budget_multiplier, conviction_multiplier;

EXPLAIN (ANALYZE, BUFFERS)
SELECT date_value, participation_state, long_budget, expected_exposure, horizon, alpha_vs_qqq
FROM mart.mv_decision_replay
WHERE candidate_id = 'B1.05_C1.10_L1.10_R1.05'
  AND fold = 1
  AND date_value BETWEEN DATE '2017-01-01' AND DATE '2017-03-31'
ORDER BY date_value, horizon;

EXPLAIN (ANALYZE, BUFFERS)
SELECT ticker, total_pnl_contribution, selection_rate, leader_flag_rate
FROM mart.mv_ticker_contribution
WHERE candidate_id = 'B1.05_C1.10_L1.10_R1.05'
ORDER BY total_pnl_contribution DESC
LIMIT 15;
