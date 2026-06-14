from __future__ import annotations


QUESTIONS = [
    {"id": "fold-best-performance", "question": "Which fold contributes most to official performance?", "endpoint": "/research/olap-preset", "facts": ["mart.mv_performance_by_fold"], "operations": ["roll-up"]},
    {"id": "fold-worst-drawdown", "question": "Which fold carries the worst drawdown?", "endpoint": "/research/olap-preset", "facts": ["mart.mv_performance_by_fold"], "operations": ["slice"]},
    {"id": "sharpe-stable-folds", "question": "Is Sharpe stable across folds?", "endpoint": "/research/olap-preset", "facts": ["mart.mv_performance_by_fold"], "operations": ["roll-up"]},
    {"id": "candidate-cagr-maxdd", "question": "Which candidate has the best CAGR/MaxDD tradeoff?", "endpoint": "/research/robustness-compare", "facts": ["mart.mv_scorecard_candidate"], "operations": ["pivot"]},
    {"id": "candidate-best-sharpe", "question": "Which candidate has the best Sharpe?", "endpoint": "/research/robustness-compare", "facts": ["mart.mv_scorecard_candidate"], "operations": ["roll-up"]},
    {"id": "candidate-severe-fold-damage", "question": "Which candidate has severe fold damage?", "endpoint": "/research/extended-summary", "facts": ["extended_multiplier_fold_summary.csv"], "operations": ["dice"]},
    {"id": "axis-degrades-most", "question": "Which multiplier axis degrades the model most?", "endpoint": "/research/extended-summary", "facts": ["sensitivity_ranking.csv"], "operations": ["roll-up"]},
    {"id": "module-helps-horizon", "question": "Which module helps most by horizon?", "endpoint": "/module/effectiveness", "facts": ["mart.mv_module_effectiveness"], "operations": ["pivot"]},
    {"id": "module-active-low-value", "question": "Which module activates often but adds little?", "endpoint": "/module/effectiveness", "facts": ["mart.mv_module_effectiveness"], "operations": ["dice"]},
    {"id": "module-better-outcomes", "question": "Which module coincides with better outcomes?", "endpoint": "/module/effectiveness", "facts": ["mart.mv_module_effectiveness"], "operations": ["roll-up"]},
    {"id": "ticker-top-contribution", "question": "Which tickers contribute most?", "endpoint": "/ticker/contribution", "facts": ["mart.mv_ticker_contribution"], "operations": ["drill-down"]},
    {"id": "ticker-largest-drags", "question": "Which tickers drag most?", "endpoint": "/ticker/contribution", "facts": ["mart.mv_ticker_contribution"], "operations": ["drill-down"]},
    {"id": "ticker-selection-low-contribution", "question": "Which tickers are frequently selected but low contribution?", "endpoint": "/ticker/contribution", "facts": ["mart.mv_ticker_contribution"], "operations": ["dice"]},
    {"id": "ticker-frequent-leaders", "question": "Which tickers are frequent leaders?", "endpoint": "/ticker/contribution", "facts": ["mart.mv_ticker_contribution"], "operations": ["roll-up"]},
    {"id": "regime-best-alpha", "question": "Which regime has the best alpha proxy?", "endpoint": "/regime/behavior", "facts": ["mart.mv_regime_behavior"], "operations": ["slice"]},
    {"id": "regime-exposure-concentration", "question": "Where is exposure concentrated?", "endpoint": "/regime/behavior", "facts": ["mart.mv_regime_behavior"], "operations": ["slice"]},
    {"id": "regime-backoff-most", "question": "Where does backoff activate most?", "endpoint": "/regime/behavior", "facts": ["mart.mv_regime_behavior"], "operations": ["slice"]},
    {"id": "regime-weakest-outcome", "question": "Which regime has weakest average outcome?", "endpoint": "/regime/behavior", "facts": ["mart.mv_regime_behavior"], "operations": ["slice"]},
    {"id": "decision-best-20d", "question": "Best decisions by 20d outcome.", "endpoint": "/research/decision-casebook", "facts": ["mart.mv_decision_outcome"], "operations": ["drill-through"]},
    {"id": "decision-worst-20d", "question": "Worst decisions by 20d outcome.", "endpoint": "/research/decision-casebook", "facts": ["mart.mv_decision_outcome"], "operations": ["drill-through"]},
    {"id": "decision-high-exposure-bad", "question": "High exposure with bad outcome.", "endpoint": "/research/decision-casebook", "facts": ["dw.fact_decision_state", "dw.fact_outcome"], "operations": ["dice"]},
    {"id": "decision-backoff-positive", "question": "Backoff decisions with positive outcome.", "endpoint": "/research/decision-casebook", "facts": ["dw.fact_decision_state", "dw.fact_outcome"], "operations": ["slice"]},
    {"id": "outcome-percentiles-horizon", "question": "Outcome percentiles by horizon.", "endpoint": "/research/distributions", "facts": ["dw.fact_outcome"], "operations": ["roll-up"]},
    {"id": "exposure-buckets-outcome", "question": "Exposure buckets vs outcome.", "endpoint": "/research/distributions", "facts": ["dw.fact_decision_state", "dw.fact_outcome"], "operations": ["dice"]},
    {"id": "turnover-buckets-outcome", "question": "Turnover buckets vs outcome.", "endpoint": "/research/cohorts", "facts": ["dw.fact_decision_state", "dw.fact_outcome"], "operations": ["dice"]},
    {"id": "drawdown-distribution-regime", "question": "Drawdown distribution by regime/fold.", "endpoint": "/research/cohorts", "facts": ["dw.fact_decision_state"], "operations": ["roll-up"]},
    {"id": "engineering-slowest-endpoint", "question": "Which endpoint is slowest?", "endpoint": "/data/execution-evidence", "facts": ["oltp.dss_query_log"], "operations": ["roll-up"]},
    {"id": "engineering-highest-p95", "question": "Which endpoint has highest p95?", "endpoint": "/data/execution-evidence", "facts": ["oltp.dss_query_log"], "operations": ["roll-up"]},
    {"id": "engineering-source-most-used", "question": "Which source relation is used most?", "endpoint": "/data/execution-evidence", "facts": ["oltp.dss_query_log"], "operations": ["roll-up"]},
    {"id": "engineering-useful-mart", "question": "Which mart supports the most useful DSS view?", "endpoint": "/data/execution-evidence", "facts": ["mart materialized views"], "operations": ["drill-through"]},
]


def registry() -> dict:
    return {"questions": QUESTIONS}
