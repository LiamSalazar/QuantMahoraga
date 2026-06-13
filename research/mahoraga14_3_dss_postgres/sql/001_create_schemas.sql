CREATE SCHEMA IF NOT EXISTS oltp;
CREATE SCHEMA IF NOT EXISTS dw;
CREATE SCHEMA IF NOT EXISTS mart;
CREATE SCHEMA IF NOT EXISTS audit;

CREATE TABLE IF NOT EXISTS audit.validation_result (
    validation_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    check_name TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'error')),
    passed BOOLEAN NOT NULL,
    observed_value TEXT,
    expected_value TEXT,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit.load_event (
    load_event_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    table_name TEXT NOT NULL,
    source_path TEXT,
    row_count BIGINT NOT NULL DEFAULT 0,
    mode TEXT NOT NULL,
    demo_mode BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
