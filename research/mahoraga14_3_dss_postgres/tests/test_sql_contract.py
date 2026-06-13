from __future__ import annotations

from etl.paths import get_paths


def test_sql_contract_contains_required_schemas_and_marts() -> None:
    sql_root = get_paths().sql_root
    schema_sql = (sql_root / "001_create_schemas.sql").read_text(encoding="utf-8")
    facts_sql = (sql_root / "004_create_facts.sql").read_text(encoding="utf-8")
    marts_sql = (sql_root / "007_create_materialized_views.sql").read_text(encoding="utf-8")
    for schema in ["oltp", "dw", "mart", "audit"]:
        assert f"CREATE SCHEMA IF NOT EXISTS {schema}" in schema_sql
    for fact in ["fact_decision_state", "fact_position_daily", "fact_module_trace", "fact_outcome", "fact_whatif"]:
        assert f"dw.{fact}" in facts_sql
    for mart in ["mv_scorecard_candidate", "mv_robustness_surface", "mv_decision_replay", "mv_query_performance"]:
        assert f"mart.{mart}" in marts_sql
