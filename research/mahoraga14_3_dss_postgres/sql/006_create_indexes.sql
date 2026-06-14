CREATE INDEX IF NOT EXISTS idx_fact_signal_candidate_fold_date
    ON dw.fact_signal_daily (candidate_id, fold, date_value);

CREATE INDEX IF NOT EXISTS idx_fact_decision_candidate_fold_date
    ON dw.fact_decision_state (candidate_id, fold, date_value);

CREATE INDEX IF NOT EXISTS idx_fact_outcome_candidate_horizon_date
    ON dw.fact_outcome (candidate_id, horizon, decision_date);

CREATE INDEX IF NOT EXISTS idx_fact_position_asset_date
    ON dw.fact_position_daily (ticker, date_value);

CREATE INDEX IF NOT EXISTS idx_fact_position_candidate_fold_date
    ON dw.fact_position_daily (candidate_id, fold, date_value);

CREATE INDEX IF NOT EXISTS idx_fact_position_replay_lookup
    ON dw.fact_position_daily (candidate_id, universe_id, fold, date_value, ticker);

CREATE INDEX IF NOT EXISTS idx_fact_module_candidate_fold_date
    ON dw.fact_module_trace (module_name, candidate_id, fold, date_value);

CREATE INDEX IF NOT EXISTS idx_fact_module_replay_lookup
    ON dw.fact_module_trace (candidate_id, universe_id, fold, date_value);

CREATE INDEX IF NOT EXISTS idx_fact_outcome_replay_lookup
    ON dw.fact_outcome (candidate_id, universe_id, fold, decision_date, horizon);

CREATE INDEX IF NOT EXISTS idx_fact_decision_regime_candidate_fold
    ON dw.fact_decision_state (regime, candidate_id, fold);

CREATE INDEX IF NOT EXISTS idx_fact_market_bar_date_brin
    ON dw.fact_market_bar USING BRIN (date_value);

CREATE INDEX IF NOT EXISTS idx_fact_position_date_brin
    ON dw.fact_position_daily USING BRIN (date_value);

CREATE INDEX IF NOT EXISTS idx_fact_path_date_brin
    ON dw.fact_path_recursive USING BRIN (date_value);

CREATE INDEX IF NOT EXISTS idx_fact_robustness_grid
    ON dw.fact_robustness_surface (metric_name, universe_id, fold, regime);

CREATE INDEX IF NOT EXISTS idx_fact_whatif_grid
    ON dw.fact_whatif (universe_id, fold, horizon, cost_bps, slippage_bps);

CREATE INDEX IF NOT EXISTS idx_query_log_endpoint_created
    ON oltp.dss_query_log (endpoint, created_at DESC);
