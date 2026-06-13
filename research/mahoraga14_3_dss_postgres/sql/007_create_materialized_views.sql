CREATE MATERIALIZED VIEW IF NOT EXISTS mart.mv_scorecard_candidate AS
SELECT
    candidate_id,
    universe_id,
    bool_or(robust_region_flag) AS any_robust_region,
    max(cagr) AS cagr,
    max(sharpe) AS sharpe,
    max(sortino) AS sortino,
    min(maxdd) AS maxdd,
    max(alpha_qqq) AS alpha_qqq,
    max(alpha_spy) AS alpha_spy,
    avg(avg_exposure) AS avg_exposure,
    avg(avg_turnover) AS avg_turnover,
    max(return_per_exposure) AS return_per_exposure,
    bool_or(demo_mode) AS demo_mode,
    max(run_id) AS run_id
FROM dw.fact_candidate_metric
GROUP BY candidate_id, universe_id;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart.mv_performance_by_fold AS
SELECT
    candidate_id,
    universe_id,
    fold,
    avg(realized_return) AS avg_realized_return,
    avg(alpha_vs_qqq) AS avg_alpha_vs_qqq,
    avg(alpha_vs_spy) AS avg_alpha_vs_spy,
    avg(realized_exposure) AS avg_exposure,
    avg(realized_turnover) AS avg_turnover,
    avg(helped_flag::int) AS helped_rate,
    count(*) AS observations,
    bool_or(demo_mode) AS demo_mode
FROM dw.fact_outcome
GROUP BY candidate_id, universe_id, fold;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart.mv_robustness_surface AS
SELECT
    candidate_id,
    universe_id,
    sweep_role,
    budget_multiplier,
    conviction_multiplier,
    leader_multiplier,
    backoff_strength,
    metric_name,
    fold,
    regime,
    avg(metric_value) AS metric_value,
    avg(robust_score) AS robust_score,
    bool_or(demo_mode) AS demo_mode
FROM dw.fact_robustness_surface
GROUP BY candidate_id, universe_id, sweep_role, budget_multiplier, conviction_multiplier,
         leader_multiplier, backoff_strength, metric_name, fold, regime;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart.mv_decision_outcome AS
SELECT
    d.date_value,
    d.candidate_id,
    d.fold,
    d.universe_id,
    d.regime,
    d.participation_state,
    d.long_budget,
    d.leader_blend,
    d.expected_exposure,
    d.expected_turnover,
    d.hard_backoff_flag,
    o.horizon,
    o.realized_return,
    o.alpha_vs_qqq,
    o.alpha_vs_spy,
    o.helped_flag,
    o.drawdown_change,
    (d.demo_mode OR o.demo_mode) AS demo_mode
FROM dw.fact_decision_state d
LEFT JOIN dw.fact_outcome o
  ON o.decision_date = d.date_value
 AND o.candidate_id = d.candidate_id
 AND o.fold = d.fold
 AND o.universe_id = d.universe_id;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart.mv_module_effectiveness AS
SELECT
    m.module_name,
    m.candidate_id,
    m.fold,
    m.universe_id,
    o.horizon,
    avg(m.module_active::int) AS activation_rate,
    avg(o.helped_flag::int) AS helped_rate,
    avg(o.alpha_vs_qqq) AS avg_alpha_vs_qqq,
    avg(o.drawdown_change) AS avg_drawdown_change,
    avg(m.effect_on_exposure) AS avg_exposure_effect,
    count(*) AS observations,
    bool_or(m.demo_mode OR coalesce(o.demo_mode, false)) AS demo_mode
FROM dw.fact_module_trace m
LEFT JOIN dw.fact_outcome o
  ON o.decision_date = m.date_value
 AND o.candidate_id = m.candidate_id
 AND o.fold = m.fold
 AND o.universe_id = m.universe_id
GROUP BY m.module_name, m.candidate_id, m.fold, m.universe_id, o.horizon;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart.mv_ticker_contribution AS
SELECT
    candidate_id,
    fold,
    universe_id,
    ticker,
    sum(pnl_contribution) AS total_pnl_contribution,
    avg(final_weight) AS avg_final_weight,
    avg(selected_flag::int) AS selection_rate,
    avg(leader_flag::int) AS leader_flag_rate,
    min(pnl_contribution) AS worst_daily_contribution,
    count(*) AS observations,
    bool_or(demo_mode) AS demo_mode
FROM dw.fact_position_daily
GROUP BY candidate_id, fold, universe_id, ticker;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart.mv_regime_behavior AS
SELECT
    regime,
    candidate_id,
    fold,
    universe_id,
    avg(net_return) AS avg_net_return,
    avg(benchmark_return) AS avg_benchmark_return,
    avg(expected_exposure) AS avg_exposure,
    avg(expected_turnover) AS avg_turnover,
    avg(drawdown) AS avg_drawdown,
    avg(backoff_flag::int) AS backoff_activation_rate,
    avg(continuation_trigger_flag::int) AS continuation_activation_rate,
    avg(leader_blend) AS avg_leader_blend,
    count(*) AS observations,
    bool_or(demo_mode) AS demo_mode
FROM dw.fact_decision_state
GROUP BY regime, candidate_id, fold, universe_id;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart.mv_whatif_grid AS
SELECT *
FROM dw.fact_whatif;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart.mv_drawdown_replay AS
SELECT *
FROM dw.fact_path_recursive;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart.mv_decision_replay AS
SELECT
    d.*,
    o.horizon,
    o.realized_return,
    o.alpha_vs_qqq,
    o.alpha_vs_spy,
    o.helped_flag
FROM dw.fact_decision_state d
LEFT JOIN dw.fact_outcome o
  ON o.decision_date = d.date_value
 AND o.candidate_id = d.candidate_id
 AND o.fold = d.fold
 AND o.universe_id = d.universe_id;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart.mv_query_performance AS
SELECT
    endpoint,
    backend,
    source_relation,
    used_materialized_view,
    count(*) AS query_count,
    avg(elapsed_ms) AS avg_elapsed_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY elapsed_ms) AS p95_elapsed_ms,
    avg(rows_returned) AS avg_rows_returned,
    max(created_at) AS last_seen_at
FROM oltp.dss_query_log
GROUP BY endpoint, backend, source_relation, used_materialized_view;

CREATE UNIQUE INDEX IF NOT EXISTS uix_mv_scorecard_candidate
    ON mart.mv_scorecard_candidate (candidate_id, universe_id);
CREATE INDEX IF NOT EXISTS idx_mv_performance_by_fold
    ON mart.mv_performance_by_fold (candidate_id, universe_id, fold);
CREATE INDEX IF NOT EXISTS idx_mv_robustness_surface
    ON mart.mv_robustness_surface (metric_name, universe_id, fold, regime);
CREATE INDEX IF NOT EXISTS idx_mv_decision_outcome
    ON mart.mv_decision_outcome (candidate_id, fold, universe_id, date_value, horizon);
CREATE INDEX IF NOT EXISTS idx_mv_module_effectiveness
    ON mart.mv_module_effectiveness (module_name, candidate_id, universe_id, horizon);
CREATE INDEX IF NOT EXISTS idx_mv_ticker_contribution
    ON mart.mv_ticker_contribution (ticker, candidate_id, fold, universe_id);
CREATE INDEX IF NOT EXISTS idx_mv_regime_behavior
    ON mart.mv_regime_behavior (regime, candidate_id, fold, universe_id);
CREATE INDEX IF NOT EXISTS idx_mv_whatif_grid
    ON mart.mv_whatif_grid (universe_id, fold, horizon, budget_multiplier, conviction_multiplier);
CREATE INDEX IF NOT EXISTS idx_mv_drawdown_replay
    ON mart.mv_drawdown_replay (candidate_id, fold, date_value);
