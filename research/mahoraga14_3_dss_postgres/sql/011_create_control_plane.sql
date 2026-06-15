CREATE TABLE IF NOT EXISTS oltp.schema_migration_log (
    migration_id BIGSERIAL PRIMARY KEY,
    schema_version TEXT NOT NULL,
    migration_name TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    checksum TEXT,
    UNIQUE (schema_version, migration_name)
);

INSERT INTO oltp.schema_migration_log (schema_version, migration_name)
VALUES ('dss_postgres_v1.1', '011_create_control_plane.sql')
ON CONFLICT (schema_version, migration_name) DO NOTHING;

CREATE TABLE IF NOT EXISTS oltp.pipeline_run (
    run_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    profile TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'STARTED',
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    total_rows_processed BIGINT NOT NULL DEFAULT 0,
    total_rows_loaded BIGINT NOT NULL DEFAULT 0,
    changed_sources_count INTEGER NOT NULL DEFAULT 0,
    changed_partitions_count INTEGER NOT NULL DEFAULT 0,
    validation_status TEXT,
    published BOOLEAN NOT NULL DEFAULT false,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS oltp.pipeline_stage_log (
    pipeline_stage_log_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    stage_name TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    duration_ms BIGINT,
    rows_read BIGINT,
    rows_written BIGINT,
    output_bytes BIGINT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS oltp.source_manifest (
    source_name TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    row_count BIGINT,
    min_date DATE,
    max_date DATE,
    candidate_id TEXT,
    universe_id TEXT,
    fold INTEGER,
    horizon INTEGER,
    module_name TEXT,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_loaded_run_id TEXT,
    PRIMARY KEY (source_name, source_path)
);

CREATE TABLE IF NOT EXISTS oltp.partition_manifest (
    partition_manifest_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    table_name TEXT NOT NULL,
    partition_key TEXT NOT NULL,
    partition_type TEXT NOT NULL,
    row_count BIGINT NOT NULL DEFAULT 0,
    min_date DATE,
    max_date DATE,
    source_hash TEXT,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    status TEXT NOT NULL DEFAULT 'loaded'
);

CREATE TABLE IF NOT EXISTS oltp.mart_refresh_log (
    mart_refresh_log_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    mart_name TEXT NOT NULL,
    refresh_strategy TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    duration_ms BIGINT,
    rows_after_refresh BIGINT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS oltp.data_quality_check (
    data_quality_check_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    table_name TEXT NOT NULL,
    check_name TEXT NOT NULL,
    status TEXT NOT NULL,
    severity TEXT NOT NULL,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    observed_value TEXT,
    expected_value TEXT,
    details JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS oltp.cache_invalidation_log (
    cache_invalidation_log_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    endpoint_pattern TEXT NOT NULL,
    reason TEXT NOT NULL,
    invalidated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS oltp.publish_log (
    publish_log_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    previous_active_run_id TEXT,
    new_active_run_id TEXT NOT NULL,
    status TEXT NOT NULL,
    published_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    rollback_available BOOLEAN NOT NULL DEFAULT true
);

CREATE TABLE IF NOT EXISTS oltp.pending_outcome (
    pending_outcome_id BIGSERIAL PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    universe_id TEXT NOT NULL,
    fold INTEGER NOT NULL,
    decision_date DATE NOT NULL,
    horizon INTEGER NOT NULL,
    maturity_date DATE NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    computed_at TIMESTAMPTZ,
    run_id TEXT,
    UNIQUE (candidate_id, universe_id, fold, decision_date, horizon)
);

CREATE INDEX IF NOT EXISTS idx_pipeline_stage_log_run_stage
    ON oltp.pipeline_stage_log (run_id, stage_name);
CREATE INDEX IF NOT EXISTS idx_source_manifest_last_seen
    ON oltp.source_manifest (last_seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_partition_manifest_table_key
    ON oltp.partition_manifest (table_name, partition_key, loaded_at DESC);
CREATE INDEX IF NOT EXISTS idx_mart_refresh_log_run
    ON oltp.mart_refresh_log (run_id, mart_name);
CREATE INDEX IF NOT EXISTS idx_data_quality_check_run
    ON oltp.data_quality_check (run_id, severity, status);
CREATE INDEX IF NOT EXISTS idx_pending_outcome_status_maturity
    ON oltp.pending_outcome (status, maturity_date);
