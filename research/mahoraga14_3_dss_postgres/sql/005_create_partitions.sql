DO $$
DECLARE
    y INTEGER;
BEGIN
    FOR y IN 2016..2027 LOOP
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS dw.fact_market_bar_%s PARTITION OF dw.fact_market_bar FOR VALUES FROM (%L) TO (%L)',
            y, make_date(y, 1, 1), make_date(y + 1, 1, 1)
        );
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS dw.fact_position_daily_%s PARTITION OF dw.fact_position_daily FOR VALUES FROM (%L) TO (%L)',
            y, make_date(y, 1, 1), make_date(y + 1, 1, 1)
        );
    END LOOP;
END $$;

CREATE TABLE IF NOT EXISTS dw.fact_market_bar_default PARTITION OF dw.fact_market_bar DEFAULT;
CREATE TABLE IF NOT EXISTS dw.fact_position_daily_default PARTITION OF dw.fact_position_daily DEFAULT;

CREATE TABLE IF NOT EXISTS dw.fact_outcome_h1 PARTITION OF dw.fact_outcome FOR VALUES IN (1);
CREATE TABLE IF NOT EXISTS dw.fact_outcome_h5 PARTITION OF dw.fact_outcome FOR VALUES IN (5);
CREATE TABLE IF NOT EXISTS dw.fact_outcome_h20 PARTITION OF dw.fact_outcome FOR VALUES IN (20);
CREATE TABLE IF NOT EXISTS dw.fact_outcome_h60 PARTITION OF dw.fact_outcome FOR VALUES IN (60);
CREATE TABLE IF NOT EXISTS dw.fact_outcome_default PARTITION OF dw.fact_outcome DEFAULT;

CREATE TABLE IF NOT EXISTS dw.fact_module_trace_base_alpha PARTITION OF dw.fact_module_trace FOR VALUES IN ('BASE_ALPHA_V2');
CREATE TABLE IF NOT EXISTS dw.fact_module_trace_continuation PARTITION OF dw.fact_module_trace FOR VALUES IN ('continuation_v2_model');
CREATE TABLE IF NOT EXISTS dw.fact_module_trace_structural PARTITION OF dw.fact_module_trace FOR VALUES IN ('structural_defense_model');
CREATE TABLE IF NOT EXISTS dw.fact_module_trace_allocator PARTITION OF dw.fact_module_trace FOR VALUES IN ('participation_allocator_v2');
CREATE TABLE IF NOT EXISTS dw.fact_module_trace_conviction PARTITION OF dw.fact_module_trace FOR VALUES IN ('conviction_amplifier_layer');
CREATE TABLE IF NOT EXISTS dw.fact_module_trace_backoff PARTITION OF dw.fact_module_trace FOR VALUES IN ('risk_backoff_layer_v2');
CREATE TABLE IF NOT EXISTS dw.fact_module_trace_leader PARTITION OF dw.fact_module_trace FOR VALUES IN ('leader_participation_layer');
CREATE TABLE IF NOT EXISTS dw.fact_module_trace_default PARTITION OF dw.fact_module_trace DEFAULT;
