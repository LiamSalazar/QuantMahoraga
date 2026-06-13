INSERT INTO audit.validation_result (run_id, check_name, severity, passed, observed_value, expected_value, detail)
SELECT
    coalesce(max(run_id), 'unknown'),
    'fact_position_daily_has_rows',
    'error',
    count(*) > 0,
    count(*)::text,
    '> 0',
    '{}'::jsonb
FROM dw.fact_position_daily;

INSERT INTO audit.validation_result (run_id, check_name, severity, passed, observed_value, expected_value, detail)
SELECT
    coalesce(max(run_id), 'unknown'),
    'fact_outcome_has_expected_horizons',
    'error',
    count(DISTINCT horizon) >= 3,
    count(DISTINCT horizon)::text,
    '>= 3',
    jsonb_build_object('horizons', jsonb_agg(DISTINCT horizon))
FROM dw.fact_outcome;

INSERT INTO audit.validation_result (run_id, check_name, severity, passed, observed_value, expected_value, detail)
SELECT
    coalesce(max(run_id), 'unknown'),
    'position_duplicate_key_check',
    'error',
    count(*) = 0,
    count(*)::text,
    '0 duplicate keys',
    '{}'::jsonb
FROM (
    SELECT date_value, candidate_id, fold, universe_id, ticker, run_id
    FROM dw.fact_position_daily
    GROUP BY date_value, candidate_id, fold, universe_id, ticker, run_id
    HAVING count(*) > 1
) dupes;

INSERT INTO audit.validation_result (run_id, check_name, severity, passed, observed_value, expected_value, detail)
SELECT
    coalesce(max(run_id), 'unknown'),
    'demo_rows_are_flagged',
    'warning',
    count(*) FILTER (WHERE demo_mode IS NULL) = 0,
    count(*) FILTER (WHERE demo_mode IS NULL)::text,
    '0 unflagged demo values',
    '{}'::jsonb
FROM dw.fact_whatif;

SELECT * FROM audit.validation_result ORDER BY created_at DESC LIMIT 50;
