from __future__ import annotations

FACT_TO_MARTS: dict[str, list[str]] = {
    "fact_outcome": [
        "mart.mv_decision_outcome",
        "mart.mv_decision_replay",
        "mart.mv_module_effectiveness",
        "mart.mv_performance_by_fold",
    ],
    "fact_position_daily": [
        "mart.mv_ticker_contribution",
        "mart.mv_decision_replay",
    ],
    "fact_decision_state": [
        "mart.mv_decision_outcome",
        "mart.mv_decision_replay",
        "mart.mv_regime_behavior",
    ],
    "fact_module_trace": [
        "mart.mv_module_effectiveness",
        "mart.mv_decision_replay",
    ],
    "fact_whatif": ["mart.mv_whatif_grid"],
    "fact_candidate_metric": ["mart.mv_scorecard_candidate"],
    "fact_robustness_surface": ["mart.mv_robustness_surface"],
    "fact_path_recursive": ["mart.mv_drawdown_replay"],
    "query_logs": ["mart.mv_query_performance"],
}

MART_TO_CACHE: dict[str, list[str]] = {
    "mart.mv_scorecard_candidate": ["/research/command-center", "/scorecard", "/research/robustness-compare"],
    "mart.mv_performance_by_fold": ["/fold/performance", "/research/olap-preset"],
    "mart.mv_decision_outcome": ["/research/distributions", "/research/cohorts", "/research/decision-casebook"],
    "mart.mv_decision_replay": ["/decision/replay"],
    "mart.mv_module_effectiveness": ["/module/effectiveness"],
    "mart.mv_ticker_contribution": ["/ticker/contribution"],
    "mart.mv_regime_behavior": ["/regime/behavior"],
    "mart.mv_whatif_grid": ["/whatif/grid", "/research/whatif-reference"],
    "mart.mv_query_performance": ["/data/execution-evidence", "/query/performance"],
}

ALL_MARTS = [
    "mart.mv_scorecard_candidate",
    "mart.mv_performance_by_fold",
    "mart.mv_robustness_surface",
    "mart.mv_decision_outcome",
    "mart.mv_module_effectiveness",
    "mart.mv_ticker_contribution",
    "mart.mv_regime_behavior",
    "mart.mv_whatif_grid",
    "mart.mv_drawdown_replay",
    "mart.mv_decision_replay",
    "mart.mv_query_performance",
]

TABLE_TEMPERATURE: dict[str, str] = {
    "mart.mv_scorecard_candidate": "hot",
    "mart.mv_performance_by_fold": "hot",
    "mart.mv_robustness_surface": "hot",
    "mart.mv_module_effectiveness": "hot",
    "mart.mv_ticker_contribution": "hot",
    "mart.mv_regime_behavior": "hot",
    "mart.mv_whatif_grid": "hot",
    "mart.mv_query_performance": "hot",
    "dw.fact_position_daily": "warm",
    "dw.fact_module_trace": "warm",
    "dw.fact_outcome": "warm",
    "dw.fact_decision_state": "warm",
    "dw.fact_path_recursive": "warm",
    "outputs/parquet": "cold",
    "outputs/control": "cold",
    "outputs/benchmarks": "cold",
}


def marts_for_tables(changed_tables: list[str]) -> list[str]:
    marts: list[str] = []
    for table in changed_tables:
        marts.extend(FACT_TO_MARTS.get(table, []))
    return sorted(set(marts), key=lambda name: ALL_MARTS.index(name) if name in ALL_MARTS else 999)


def cache_for_marts(marts: list[str]) -> list[str]:
    endpoints: list[str] = []
    for mart in marts:
        endpoints.extend(MART_TO_CACHE.get(mart, []))
    return sorted(set(endpoints))


def temperature_for(name: str) -> str:
    return TABLE_TEMPERATURE.get(name, "warm")
