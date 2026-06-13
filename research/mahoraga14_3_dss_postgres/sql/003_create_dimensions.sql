CREATE TABLE IF NOT EXISTS dw.dim_date (
    date_key INTEGER PRIMARY KEY,
    date_value DATE NOT NULL UNIQUE,
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    month INTEGER NOT NULL,
    month_name TEXT NOT NULL,
    week INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL,
    is_month_end BOOLEAN NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS dw.dim_asset (
    asset_id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL UNIQUE,
    asset_name TEXT,
    asset_class TEXT NOT NULL DEFAULT 'equity',
    sector TEXT,
    source_universe TEXT,
    demo_mode BOOLEAN NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS dw.dim_candidate (
    candidate_sk BIGSERIAL PRIMARY KEY,
    candidate_id TEXT NOT NULL UNIQUE,
    candidate_label TEXT NOT NULL,
    family TEXT NOT NULL DEFAULT 'official_or_extended',
    budget_multiplier DOUBLE PRECISION,
    conviction_multiplier DOUBLE PRECISION,
    leader_multiplier DOUBLE PRECISION,
    backoff_strength DOUBLE PRECISION,
    is_official BOOLEAN NOT NULL DEFAULT false,
    demo_mode BOOLEAN NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS dw.dim_universe (
    universe_sk BIGSERIAL PRIMARY KEY,
    universe_id TEXT NOT NULL UNIQUE,
    proposed_count INTEGER,
    usable_count INTEGER,
    usable_tickers TEXT,
    missing_tickers TEXT,
    effective_start DATE,
    effective_end DATE,
    mean_coverage_ratio DOUBLE PRECISION,
    demo_mode BOOLEAN NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS dw.dim_fold (
    fold_sk BIGSERIAL PRIMARY KEY,
    fold INTEGER NOT NULL UNIQUE,
    test_start DATE,
    test_end DATE,
    label TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dw.dim_module (
    module_sk BIGSERIAL PRIMARY KEY,
    module_name TEXT NOT NULL UNIQUE,
    module_family TEXT NOT NULL,
    display_order INTEGER NOT NULL DEFAULT 999
);

CREATE TABLE IF NOT EXISTS dw.dim_regime (
    regime_sk BIGSERIAL PRIMARY KEY,
    regime_name TEXT NOT NULL UNIQUE,
    regime_family TEXT NOT NULL DEFAULT 'participation_state',
    description TEXT
);

CREATE TABLE IF NOT EXISTS dw.dim_horizon (
    horizon_sk BIGSERIAL PRIMARY KEY,
    horizon INTEGER NOT NULL UNIQUE,
    horizon_label TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dw.dim_scenario (
    scenario_sk BIGSERIAL PRIMARY KEY,
    scenario_id TEXT NOT NULL UNIQUE,
    scenario_family TEXT NOT NULL,
    budget_multiplier DOUBLE PRECISION,
    conviction_multiplier DOUBLE PRECISION,
    leader_multiplier DOUBLE PRECISION,
    backoff_strength DOUBLE PRECISION,
    cost_bps DOUBLE PRECISION,
    slippage_bps DOUBLE PRECISION,
    demo_mode BOOLEAN NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS dw.dim_metric (
    metric_sk BIGSERIAL PRIMARY KEY,
    metric_name TEXT NOT NULL UNIQUE,
    metric_family TEXT NOT NULL,
    display_name TEXT NOT NULL,
    higher_is_better BOOLEAN NOT NULL DEFAULT true,
    unit TEXT
);
