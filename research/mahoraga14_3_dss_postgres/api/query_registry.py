from __future__ import annotations


QUESTIONS = [
    {
        "id": "candidate_robust_by_fold",
        "question": "Que candidato es mas robusto por fold?",
        "endpoint": "/fold/performance",
        "facts": ["fact_outcome", "fact_candidate_metric"],
        "operations": ["roll-up", "slice"],
    },
    {
        "id": "official_grid_location",
        "question": "Donde se ubica el candidato oficial dentro del grid?",
        "endpoint": "/robustness/surface",
        "facts": ["fact_robustness_surface"],
        "operations": ["slice", "dice"],
    },
    {
        "id": "modules_reduce_drawdown",
        "question": "Que modulos reducen drawdown?",
        "endpoint": "/module/effectiveness",
        "facts": ["fact_module_trace", "fact_outcome"],
        "operations": ["drill-down", "roll-up"],
    },
    {
        "id": "leader_participation_help",
        "question": "Cuando leader participation ayuda?",
        "endpoint": "/decision/replay",
        "facts": ["fact_position_daily", "fact_outcome"],
        "operations": ["decision replay", "drill-down"],
    },
    {
        "id": "ticker_risk_adjusted_contribution",
        "question": "Que tickers aportan mas retorno ajustado por riesgo?",
        "endpoint": "/ticker/contribution",
        "facts": ["fact_position_daily"],
        "operations": ["roll-up", "drill-down"],
    },
    {
        "id": "cost_change",
        "question": "Como cambia el resultado al subir costos?",
        "endpoint": "/whatif/grid",
        "facts": ["fact_whatif", "fact_cost_sensitivity"],
        "operations": ["what-if", "slice"],
    },
    {
        "id": "market_regime",
        "question": "Que pasa por regimen de mercado?",
        "endpoint": "/regime/behavior",
        "facts": ["fact_decision_state", "fact_outcome"],
        "operations": ["slice", "roll-up"],
    },
    {
        "id": "worst_drawdown_replay",
        "question": "Que decisiones explican el peor drawdown?",
        "endpoint": "/decision/replay",
        "facts": ["fact_path_recursive", "fact_decision_state", "fact_position_daily"],
        "operations": ["decision replay", "drill-down"],
    },
    {
        "id": "cagr_maxdd_tradeoff",
        "question": "Que combinacion tiene mejor tradeoff CAGR/MaxDD?",
        "endpoint": "/whatif/grid",
        "facts": ["fact_whatif"],
        "operations": ["what-if", "pareto"],
    },
    {
        "id": "universe_stability",
        "question": "Que tan estable es Mahoraga por universo?",
        "endpoint": "/candidate/compare",
        "facts": ["fact_universe_sensitivity", "fact_candidate_metric"],
        "operations": ["dice", "roll-up"],
    },
]


def registry() -> dict:
    return {"questions": QUESTIONS}
