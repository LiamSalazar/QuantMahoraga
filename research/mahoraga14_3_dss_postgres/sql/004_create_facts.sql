CREATE TABLE IF NOT EXISTS dw.fact_market_bar (
    date_value DATE NOT NULL,
    ticker TEXT NOT NULL,
    close_return DOUBLE PRECISION,
    drawdown DOUBLE PRECISION,
    realized_vol DOUBLE PRECISION,
    benchmark_strength DOUBLE PRECISION,
    benchmark_weakness DOUBLE PRECISION,
    market_regime TEXT,
    run_id TEXT NOT NULL,
    demo_mode BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (date_value, ticker, run_id)
) PARTITION BY RANGE (date_value);

CREATE TABLE IF NOT EXISTS dw.fact_signal_daily (
    date_value DATE NOT NULL,
    ticker TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    fold INTEGER NOT NULL,
    universe_id TEXT NOT NULL,
    trend DOUBLE PRECISION,
    momentum DOUBLE PRECISION,
    relative_momentum DOUBLE PRECISION,
    residual_trend DOUBLE PRECISION,
    residual_momentum DOUBLE PRECISION,
    beta_drag DOUBLE PRECISION,
    final_score DOUBLE PRECISION,
    rank DOUBLE PRECISION,
    selected_flag BOOLEAN,
    run_id TEXT NOT NULL,
    demo_mode BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (date_value, ticker, candidate_id, fold, universe_id, run_id)
);

CREATE TABLE IF NOT EXISTS dw.fact_decision_state (
    date_value DATE NOT NULL,
    candidate_id TEXT NOT NULL,
    fold INTEGER NOT NULL,
    universe_id TEXT NOT NULL,
    regime TEXT,
    participation_state TEXT,
    continuation_trigger_flag BOOLEAN,
    continuation_pressure_flag BOOLEAN,
    structural_flag BOOLEAN,
    backoff_flag BOOLEAN,
    hard_backoff_flag BOOLEAN,
    leader_blend DOUBLE PRECISION,
    gross_exposure DOUBLE PRECISION,
    net_return DOUBLE PRECISION,
    benchmark_return DOUBLE PRECISION,
    turnover DOUBLE PRECISION,
    drawdown DOUBLE PRECISION,
    long_budget DOUBLE PRECISION,
    gate_scale DOUBLE PRECISION,
    vol_mult DOUBLE PRECISION,
    exp_cap DOUBLE PRECISION,
    expected_exposure DOUBLE PRECISION,
    expected_turnover DOUBLE PRECISION,
    run_id TEXT NOT NULL,
    demo_mode BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (date_value, candidate_id, fold, universe_id, run_id)
);

CREATE TABLE IF NOT EXISTS dw.fact_position_daily (
    date_value DATE NOT NULL,
    candidate_id TEXT NOT NULL,
    fold INTEGER NOT NULL,
    universe_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    target_weight DOUBLE PRECISION,
    weight_after_stop DOUBLE PRECISION,
    weight_exec_1x DOUBLE PRECISION,
    final_weight DOUBLE PRECISION,
    pnl_contribution DOUBLE PRECISION,
    stop_active BOOLEAN,
    leader_flag BOOLEAN,
    selected_flag BOOLEAN,
    final_score DOUBLE PRECISION,
    rank DOUBLE PRECISION,
    run_id TEXT NOT NULL,
    demo_mode BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (date_value, candidate_id, fold, universe_id, ticker, run_id)
) PARTITION BY RANGE (date_value);

CREATE TABLE IF NOT EXISTS dw.fact_module_trace (
    date_value DATE NOT NULL,
    candidate_id TEXT NOT NULL,
    fold INTEGER NOT NULL,
    universe_id TEXT NOT NULL,
    module_name TEXT NOT NULL,
    module_active BOOLEAN,
    raw_value DOUBLE PRECISION,
    intensity_score DOUBLE PRECISION,
    probability DOUBLE PRECISION,
    state_label TEXT,
    effect_on_budget DOUBLE PRECISION,
    effect_on_exposure DOUBLE PRECISION,
    effect_on_blend DOUBLE PRECISION,
    input_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    run_id TEXT NOT NULL,
    demo_mode BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (date_value, candidate_id, fold, universe_id, module_name, run_id)
) PARTITION BY LIST (module_name);

CREATE TABLE IF NOT EXISTS dw.fact_outcome (
    decision_date DATE NOT NULL,
    candidate_id TEXT NOT NULL,
    fold INTEGER NOT NULL,
    universe_id TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    realized_return DOUBLE PRECISION,
    qqq_return DOUBLE PRECISION,
    spy_return DOUBLE PRECISION,
    alpha_vs_qqq DOUBLE PRECISION,
    alpha_vs_spy DOUBLE PRECISION,
    helped_flag BOOLEAN,
    drawdown_change DOUBLE PRECISION,
    exposure_adjusted_outcome DOUBLE PRECISION,
    realized_turnover DOUBLE PRECISION,
    realized_exposure DOUBLE PRECISION,
    run_id TEXT NOT NULL,
    demo_mode BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (decision_date, candidate_id, fold, universe_id, horizon, run_id)
) PARTITION BY LIST (horizon);

CREATE TABLE IF NOT EXISTS dw.fact_candidate_metric (
    candidate_id TEXT NOT NULL,
    universe_id TEXT NOT NULL,
    sweep_role TEXT,
    metric_set TEXT NOT NULL,
    cagr DOUBLE PRECISION,
    sharpe DOUBLE PRECISION,
    sortino DOUBLE PRECISION,
    maxdd DOUBLE PRECISION,
    alpha_qqq DOUBLE PRECISION,
    alpha_spy DOUBLE PRECISION,
    beta_qqq DOUBLE PRECISION,
    beta_spy DOUBLE PRECISION,
    avg_exposure DOUBLE PRECISION,
    avg_turnover DOUBLE PRECISION,
    return_per_exposure DOUBLE PRECISION,
    robust_region_flag BOOLEAN,
    run_id TEXT NOT NULL,
    demo_mode BOOLEAN NOT NULL DEFAULT false
);

ALTER TABLE dw.fact_candidate_metric
    DROP CONSTRAINT IF EXISTS fact_candidate_metric_pkey;

ALTER TABLE dw.fact_candidate_metric
    ALTER COLUMN sweep_role DROP NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uix_fact_candidate_metric_scope
    ON dw.fact_candidate_metric (
        candidate_id,
        universe_id,
        COALESCE(sweep_role, '__not_applicable__'),
        metric_set,
        run_id
    );

CREATE TABLE IF NOT EXISTS dw.fact_robustness_surface (
    candidate_id TEXT NOT NULL,
    universe_id TEXT NOT NULL,
    sweep_role TEXT,
    budget_multiplier DOUBLE PRECISION,
    conviction_multiplier DOUBLE PRECISION,
    leader_multiplier DOUBLE PRECISION,
    backoff_strength DOUBLE PRECISION,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    fold INTEGER,
    regime TEXT,
    robust_score DOUBLE PRECISION,
    source_artifact TEXT NOT NULL,
    run_id TEXT NOT NULL,
    demo_mode BOOLEAN NOT NULL DEFAULT false
);

ALTER TABLE dw.fact_robustness_surface
    ALTER COLUMN sweep_role DROP NOT NULL;

CREATE TABLE IF NOT EXISTS dw.fact_cost_sensitivity (
    candidate_id TEXT NOT NULL,
    universe_id TEXT NOT NULL,
    cost_bps DOUBLE PRECISION NOT NULL,
    slippage_bps DOUBLE PRECISION NOT NULL,
    cagr DOUBLE PRECISION,
    sharpe DOUBLE PRECISION,
    maxdd DOUBLE PRECISION,
    alpha DOUBLE PRECISION,
    run_id TEXT NOT NULL,
    demo_mode BOOLEAN NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS dw.fact_universe_sensitivity (
    candidate_id TEXT NOT NULL,
    universe_id TEXT NOT NULL,
    proposed_count INTEGER,
    usable_count INTEGER,
    mean_coverage_ratio DOUBLE PRECISION,
    cagr DOUBLE PRECISION,
    sharpe DOUBLE PRECISION,
    maxdd DOUBLE PRECISION,
    run_status TEXT,
    run_id TEXT NOT NULL,
    demo_mode BOOLEAN NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS dw.fact_whatif (
    scenario_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    budget_multiplier DOUBLE PRECISION NOT NULL,
    conviction_multiplier DOUBLE PRECISION NOT NULL,
    leader_multiplier DOUBLE PRECISION NOT NULL,
    backoff_strength DOUBLE PRECISION NOT NULL,
    cost_bps DOUBLE PRECISION NOT NULL,
    slippage_bps DOUBLE PRECISION NOT NULL,
    fold INTEGER,
    universe_id TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    cagr DOUBLE PRECISION,
    sharpe DOUBLE PRECISION,
    sortino DOUBLE PRECISION,
    maxdd DOUBLE PRECISION,
    alpha DOUBLE PRECISION,
    beta DOUBLE PRECISION,
    turnover DOUBLE PRECISION,
    avg_exposure DOUBLE PRECISION,
    rank INTEGER,
    robust_score DOUBLE PRECISION,
    source_artifact TEXT NOT NULL,
    run_id TEXT NOT NULL,
    demo_mode BOOLEAN NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS dw.fact_path_recursive (
    candidate_id TEXT NOT NULL,
    fold INTEGER,
    date_value DATE NOT NULL,
    equity DOUBLE PRECISION,
    rolling_peak DOUBLE PRECISION,
    drawdown DOUBLE PRECISION,
    drawdown_duration INTEGER,
    recovery_days INTEGER,
    state_entry BOOLEAN,
    state_exit BOOLEAN,
    path_episode_id TEXT,
    run_id TEXT NOT NULL,
    demo_mode BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (candidate_id, date_value, run_id)
);

CREATE TABLE IF NOT EXISTS dw.fact_data_quality (
    run_id TEXT NOT NULL,
    table_name TEXT NOT NULL,
    check_name TEXT NOT NULL,
    passed BOOLEAN NOT NULL,
    observed_value TEXT,
    expected_value TEXT,
    severity TEXT NOT NULL,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
