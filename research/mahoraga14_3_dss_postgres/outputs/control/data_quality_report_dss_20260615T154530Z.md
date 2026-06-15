# Data Quality Report dss_20260615T154530Z

Passed: `True`

## fact_position_daily
- passed: `True`
- errors: `0`
- warnings: `0`
- ERROR required_columns_present: PASS ([] vs [])
- ERROR date_value_not_null: PASS (0 vs 0)
- ERROR candidate_id_not_null: PASS (0 vs 0)
- ERROR fold_not_null: PASS (0 vs 0)
- ERROR universe_id_not_null: PASS (0 vs 0)
- ERROR ticker_not_null: PASS (0 vs 0)
- ERROR run_id_not_null: PASS (0 vs 0)
- ERROR fold_allowed_values: PASS (0 vs 0)
- ERROR duplicate_grain_check: PASS (0 vs 0)
- ERROR demo_mode_not_null: PASS (0 vs 0)

## fact_outcome
- passed: `True`
- errors: `0`
- warnings: `0`
- ERROR required_columns_present: PASS ([] vs [])
- ERROR decision_date_not_null: PASS (0 vs 0)
- ERROR candidate_id_not_null: PASS (0 vs 0)
- ERROR universe_id_not_null: PASS (0 vs 0)
- ERROR fold_not_null: PASS (0 vs 0)
- ERROR horizon_not_null: PASS (0 vs 0)
- ERROR run_id_not_null: PASS (0 vs 0)
- ERROR horizon_allowed_values: PASS (0 vs 0)
- ERROR fold_allowed_values: PASS (0 vs 0)
- ERROR duplicate_grain_check: PASS (0 vs 0)
- ERROR demo_mode_not_null: PASS (0 vs 0)

## fact_module_trace
- passed: `True`
- errors: `0`
- warnings: `0`
- ERROR required_columns_present: PASS ([] vs [])
- ERROR date_value_not_null: PASS (0 vs 0)
- ERROR candidate_id_not_null: PASS (0 vs 0)
- ERROR universe_id_not_null: PASS (0 vs 0)
- ERROR fold_not_null: PASS (0 vs 0)
- ERROR module_name_not_null: PASS (0 vs 0)
- ERROR run_id_not_null: PASS (0 vs 0)
- ERROR fold_allowed_values: PASS (0 vs 0)
- ERROR duplicate_grain_check: PASS (0 vs 0)
- ERROR demo_mode_not_null: PASS (0 vs 0)

## fact_decision_state
- passed: `True`
- errors: `0`
- warnings: `0`
- ERROR required_columns_present: PASS ([] vs [])
- ERROR date_value_not_null: PASS (0 vs 0)
- ERROR candidate_id_not_null: PASS (0 vs 0)
- ERROR universe_id_not_null: PASS (0 vs 0)
- ERROR fold_not_null: PASS (0 vs 0)
- ERROR run_id_not_null: PASS (0 vs 0)
- ERROR fold_allowed_values: PASS (0 vs 0)
- ERROR duplicate_grain_check: PASS (0 vs 0)
- ERROR demo_mode_not_null: PASS (0 vs 0)

## fact_whatif
- passed: `True`
- errors: `0`
- warnings: `0`
- ERROR required_columns_present: PASS ([] vs [])
- ERROR scenario_id_not_null: PASS (0 vs 0)
- ERROR candidate_id_not_null: PASS (0 vs 0)
- ERROR universe_id_not_null: PASS (0 vs 0)
- ERROR horizon_not_null: PASS (0 vs 0)
- ERROR source_artifact_not_null: PASS (0 vs 0)
- ERROR run_id_not_null: PASS (0 vs 0)
- ERROR horizon_allowed_values: PASS (0 vs 0)
- ERROR duplicate_grain_check: PASS (0 vs 0)
- ERROR synthetic_rows_are_flagged: PASS (0 vs 0)

## dim_candidate
- passed: `True`
- errors: `0`
- warnings: `0`
- ERROR required_columns_present: PASS ([] vs [])
- ERROR candidate_id_not_null: PASS (0 vs 0)
- ERROR candidate_label_not_null: PASS (0 vs 0)
- ERROR duplicate_grain_check: PASS (0 vs 0)

## dim_asset
- passed: `True`
- errors: `0`
- warnings: `0`
- ERROR required_columns_present: PASS ([] vs [])
- ERROR ticker_not_null: PASS (0 vs 0)
- ERROR duplicate_grain_check: PASS (0 vs 0)

## dim_date
- passed: `True`
- errors: `0`
- warnings: `0`
- ERROR required_columns_present: PASS ([] vs [])
- ERROR date_key_not_null: PASS (0 vs 0)
- ERROR date_value_not_null: PASS (0 vs 0)
- ERROR duplicate_grain_check: PASS (0 vs 0)
