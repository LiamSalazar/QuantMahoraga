CREATE TABLE IF NOT EXISTS oltp.research_run (
    research_run_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    phase TEXT NOT NULL,
    baseline_reference TEXT NOT NULL,
    official_candidate_id TEXT NOT NULL,
    profile TEXT NOT NULL,
    source_mode TEXT NOT NULL,
    demo_mode BOOLEAN NOT NULL DEFAULT false,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'STARTED',
    notes TEXT
);

CREATE TABLE IF NOT EXISTS oltp.data_snapshot (
    data_snapshot_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES oltp.research_run(run_id),
    snapshot_name TEXT NOT NULL,
    source_root TEXT NOT NULL,
    artifact_count INTEGER NOT NULL DEFAULT 0,
    row_count BIGINT NOT NULL DEFAULT 0,
    content_hash TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS oltp.artifact_inventory (
    artifact_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES oltp.research_run(run_id),
    artifact_role TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    storage_format TEXT NOT NULL,
    exists_flag BOOLEAN NOT NULL,
    row_count BIGINT,
    column_count INTEGER,
    size_bytes BIGINT,
    required_flag BOOLEAN NOT NULL DEFAULT false,
    demo_mode BOOLEAN NOT NULL DEFAULT false,
    schema_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    phase TEXT,
    discovered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, relative_path)
);

ALTER TABLE oltp.artifact_inventory
    ADD COLUMN IF NOT EXISTS phase TEXT;

CREATE TABLE IF NOT EXISTS oltp.candidate_grid (
    candidate_grid_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES oltp.research_run(run_id),
    candidate_id TEXT NOT NULL,
    universe_id TEXT NOT NULL,
    sweep_role TEXT,
    budget_multiplier DOUBLE PRECISION,
    conviction_multiplier DOUBLE PRECISION,
    leader_multiplier DOUBLE PRECISION,
    backoff_strength DOUBLE PRECISION,
    source_artifact TEXT NOT NULL,
    demo_mode BOOLEAN NOT NULL DEFAULT false
);

ALTER TABLE oltp.candidate_grid
    DROP CONSTRAINT IF EXISTS candidate_grid_run_id_candidate_id_universe_id_sweep_role_key;

ALTER TABLE oltp.candidate_grid
    ALTER COLUMN sweep_role DROP NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uix_candidate_grid_scope
    ON oltp.candidate_grid (
        run_id,
        candidate_id,
        universe_id,
        COALESCE(sweep_role, '__not_applicable__')
    );

CREATE TABLE IF NOT EXISTS oltp.simulation_job (
    simulation_job_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES oltp.research_run(run_id),
    scenario_id TEXT NOT NULL,
    requested_by TEXT NOT NULL DEFAULT 'dss',
    profile TEXT NOT NULL,
    request_json JSONB NOT NULL,
    demo_mode BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS oltp.simulation_job_status (
    simulation_job_status_id BIGSERIAL PRIMARY KEY,
    simulation_job_id BIGINT NOT NULL REFERENCES oltp.simulation_job(simulation_job_id),
    status TEXT NOT NULL,
    message TEXT,
    rows_written BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS oltp.cube_build (
    cube_build_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES oltp.research_run(run_id),
    cube_name TEXT NOT NULL,
    source_tables TEXT[] NOT NULL DEFAULT '{}',
    row_count BIGINT NOT NULL DEFAULT 0,
    duration_ms INTEGER,
    status TEXT NOT NULL DEFAULT 'STARTED',
    demo_mode BOOLEAN NOT NULL DEFAULT false,
    built_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS oltp.dss_query_log (
    query_log_id BIGSERIAL PRIMARY KEY,
    query_id TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    backend TEXT NOT NULL,
    source_relation TEXT,
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    rows_returned BIGINT NOT NULL DEFAULT 0,
    elapsed_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    used_materialized_view BOOLEAN NOT NULL DEFAULT false,
    scanned_rows BIGINT,
    demo_mode BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS oltp.audit_case (
    audit_case_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES oltp.research_run(run_id),
    case_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    fold INTEGER,
    decision_date DATE,
    universe_id TEXT,
    case_type TEXT NOT NULL,
    case_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, case_id)
);

CREATE TABLE IF NOT EXISTS oltp.whatif_request (
    whatif_request_id BIGSERIAL PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL REFERENCES oltp.research_run(run_id),
    candidate_id TEXT NOT NULL,
    fold INTEGER,
    universe_id TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    budget_multiplier DOUBLE PRECISION NOT NULL,
    conviction_multiplier DOUBLE PRECISION NOT NULL,
    leader_multiplier DOUBLE PRECISION NOT NULL,
    backoff_strength DOUBLE PRECISION NOT NULL,
    cost_bps DOUBLE PRECISION NOT NULL,
    slippage_bps DOUBLE PRECISION NOT NULL,
    result_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    demo_mode BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
