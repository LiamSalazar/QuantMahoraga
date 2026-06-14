from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .db import create_backend
from .query_registry import registry
from .research_artifacts import (
    OFFICIAL_CANDIDATE_ID,
    OFFICIAL_UNIVERSE_ID,
    baseline_evidence,
    best_official_worst_from_extended,
    extended_summary,
    pipeline_summary,
)
from .schemas import HealthResponse
from .settings import load_settings

settings = load_settings()
backend = create_backend(settings)

app = FastAPI(title="Mahoraga Quant DSS API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _options() -> dict:
    return backend.options()


def _validate(value, option_key: str, label: str):
    if value is None:
        return value
    opts = _options().get(option_key, [])
    if value not in opts:
        raise HTTPException(status_code=422, detail=f"Invalid {label}: {value!r}.")
    return value


def _validate_many(values: list[str] | None, option_key: str, label: str) -> list[str] | None:
    if not values:
        return values
    opts = set(_options().get(option_key, []))
    bad = [value for value in values if value not in opts]
    if bad:
        raise HTTPException(status_code=422, detail=f"Invalid {label}: {bad!r}.")
    return values


def _timed(endpoint: str, source: str, fn):
    timed = getattr(backend, "timed", None)
    return timed(endpoint, source, fn) if callable(timed) else fn()


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return {
        "ok": True,
        "backend": backend.backend_name,
        "profile": settings.profile,
        "demo_mode": backend.demo_mode(),
        "row_counts": backend.row_counts(),
    }


@app.get("/data/health-summary")
def health_summary() -> dict:
    counts = backend.row_counts()
    summary = pipeline_summary()
    query_perf = backend.query_performance() if hasattr(backend, "query_performance") else {"count": 0, "rows": []}
    mart_rows = sum(count for table, count in counts.items() if str(table).startswith("mart.") or str(table).startswith("mv_"))
    dw_rows = sum(count for table, count in counts.items() if str(table).startswith("dw.") or str(table).startswith("fact_") or str(table).startswith("dim_"))
    oltp_rows = sum(count for table, count in counts.items() if str(table).startswith("oltp.") or table in {"research_run", "data_snapshot", "artifact_inventory", "candidate_grid"})
    total_rows = int(summary.get("total_rows_written") or sum(counts.values()))
    real_rows = int(summary.get("real_rows_written_estimate") or max(0, total_rows - int(summary.get("demo_rows_written") or 0)))
    simulated_rows = int(summary.get("demo_rows_written") or 0)
    return {
        "ok": True,
        "backend": backend.backend_name,
        "profile": settings.profile,
        "row_counts": counts,
        "logical_counts": {"oltp_rows": oltp_rows, "dw_rows": dw_rows, "mart_rows": mart_rows, "total_rows": total_rows},
        "real_rows": real_rows,
        "simulated_rows": simulated_rows,
        "contains_simulated_whatif": simulated_rows > 0 or bool(backend.demo_mode()),
        "validation_passed": summary.get("validation_passed"),
        "latest_run_id": summary.get("run_id") or summary.get("latest_run_id"),
        "query_logs_active": query_perf.get("count", 0) > 0,
        "query_log_count": query_perf.get("count", 0),
        "marts_available": sorted([table for table in counts if str(table).startswith("mart.") or str(table).startswith("mv_")]),
        "row_origin_note": "Postgres/parquet audited artifacts plus flagged simulated what-if rows.",
    }


@app.get("/data/execution-evidence")
def execution_evidence() -> dict:
    evidence = backend.execution_evidence() if hasattr(backend, "execution_evidence") else {"row_counts": backend.row_counts(), "query_performance": [], "source_usage": []}
    counts = evidence.get("row_counts", {})
    perf_rows = evidence.get("query_performance", [])
    source_rows = evidence.get("source_usage", [])
    summary = pipeline_summary()
    materialized_views = [name for name in counts if str(name).startswith("mart.mv_")]
    return {
        **evidence,
        "latest_run_id": summary.get("run_id") or summary.get("latest_run_id"),
        "validation_status": summary.get("validation_passed"),
        "oltp_tables_loaded": sum(1 for name, count in counts.items() if str(name).startswith("oltp.") and count > 0),
        "dw_facts_loaded": sum(1 for name, count in counts.items() if str(name).startswith("dw.fact_") and count > 0),
        "marts_loaded": sum(1 for name, count in counts.items() if str(name).startswith("mart.") and count > 0),
        "materialized_views_count": len(materialized_views),
        "real_rows": int(summary.get("real_rows_written_estimate") or 0),
        "simulated_whatif_rows": int(summary.get("demo_rows_written") or 0),
        "query_logs_active": bool(perf_rows),
        "slowest_endpoint": max(perf_rows, key=lambda row: float(row.get("avg_elapsed_ms") or 0), default=None),
        "highest_p95_endpoint": max(perf_rows, key=lambda row: float(row.get("p95_elapsed_ms") or 0), default=None),
        "fastest_endpoint": min(perf_rows, key=lambda row: float(row.get("avg_elapsed_ms") or 10**9), default=None),
        "most_used_endpoint": max(perf_rows, key=lambda row: int(row.get("query_count") or 0), default=None),
        "most_used_source_relation": max(source_rows, key=lambda row: int(row.get("query_count") or 0), default=None),
    }


@app.get("/labels/candidates")
def labels_candidates() -> dict:
    candidates = _options().get("candidates", [])
    rows = []
    for candidate_id in candidates:
        role = "Official baseline" if candidate_id == OFFICIAL_CANDIDATE_ID else "Observed/audited scenario"
        if str(candidate_id).startswith("EXTREME_pro-risk"):
            role = "Extreme: pro-risk"
        elif str(candidate_id).startswith("EXTREME_pro-defense"):
            role = "Extreme: pro-defense"
        elif str(candidate_id).startswith("EXTREME"):
            role = "Extreme stress case"
        rows.append({"candidate_id": candidate_id, "role": role, "is_official": candidate_id == OFFICIAL_CANDIDATE_ID})
    return {"official_candidate_id": OFFICIAL_CANDIDATE_ID, "default_universe_id": OFFICIAL_UNIVERSE_ID, "rows": rows}


@app.get("/metadata/options")
def metadata_options() -> dict:
    return _timed("/metadata/options", "dimensions", backend.options)


@app.get("/metadata/questions")
def metadata_questions() -> dict:
    return registry()


@app.get("/research/baseline-evidence")
def research_baseline_evidence() -> dict:
    return baseline_evidence()


@app.get("/research/extended-summary")
def research_extended_summary() -> dict:
    return extended_summary()


@app.get("/research/best-official-worst")
def research_best_official_worst(universe_id: str | None = OFFICIAL_UNIVERSE_ID) -> dict:
    _validate(universe_id, "universes", "universe")
    packaged = best_official_worst_from_extended()
    if packaged.get("rows"):
        return packaged
    rows = backend.scorecard(None, universe_id, 500).get("rows", [])
    official = next((row for row in rows if row.get("candidate_id") == OFFICIAL_CANDIDATE_ID), None)
    scored = [row for row in rows if row.get("sharpe") is not None]
    best = max(scored, key=lambda row: float(row.get("sharpe") or -999999), default=None)
    worst = min(scored, key=lambda row: float(row.get("sharpe") or 999999), default=None)
    return {"best": best, "official": official, "worst": worst, "rows": [row for row in [best, official, worst] if row]}


@app.get("/research/command-center")
def research_command_center(
    candidate_id: str = OFFICIAL_CANDIDATE_ID,
    universe_id: str = OFFICIAL_UNIVERSE_ID,
    fold: int | None = None,
) -> dict:
    _validate(candidate_id, "candidates", "candidate")
    _validate(universe_id, "universes", "universe")
    if fold is not None:
        _validate(fold, "folds", "fold")
    evidence = baseline_evidence()
    extended = extended_summary()
    overview_payload = backend.overview(candidate_id, fold, universe_id, "QQQ", None, None)
    return {
        "identity": {
            "official_candidate_id": OFFICIAL_CANDIDATE_ID,
            "official_universe_id": OFFICIAL_UNIVERSE_ID,
            "status": "Frozen · promoted · audited",
            "backend": backend.backend_name,
            "data_badge": f"{backend.backend_name.title()} · audited artifacts + flagged simulated what-if",
        },
        "health": health_summary(),
        "overview": overview_payload,
        "baseline_comparison": evidence.get("stitched_comparison", []),
        "best_official_worst": best_official_worst_from_extended(),
        "research_questions": registry().get("questions", []),
        "top_wins_drags": backend.top_wins_drags(candidate_id, universe_id) if hasattr(backend, "top_wins_drags") else {},
        "sensitivity_ranking": extended.get("sensitivity_ranking", []),
        "plateau_radius": extended.get("plateau_radius", []),
        "universe_robustness": extended.get("universe_robustness", []),
        "sources": ["mart.mv_scorecard_candidate", "mart.mv_drawdown_replay", "baseline official outputs", "extended analysis outputs"],
    }


@app.get("/research/top-wins-drags")
def research_top_wins_drags(candidate_id: str = OFFICIAL_CANDIDATE_ID, universe_id: str = OFFICIAL_UNIVERSE_ID) -> dict:
    _validate(candidate_id, "candidates", "candidate")
    _validate(universe_id, "universes", "universe")
    if not hasattr(backend, "top_wins_drags"):
        return {"folds": [], "tickers": [], "modules": [], "regimes": [], "candidates": []}
    return _timed("/research/top-wins-drags", "mart.mv_performance_by_fold+mart.mv_ticker_contribution", lambda: backend.top_wins_drags(candidate_id, universe_id))


@app.get("/research/decision-casebook")
def research_decision_casebook(candidate_id: str = OFFICIAL_CANDIDATE_ID, universe_id: str = OFFICIAL_UNIVERSE_ID, fold: int | None = None) -> dict:
    _validate(candidate_id, "candidates", "candidate")
    _validate(universe_id, "universes", "universe")
    if fold is not None:
        _validate(fold, "folds", "fold")
    if not hasattr(backend, "decision_casebook"):
        return {"count": 0, "rows": []}
    return _timed("/research/decision-casebook", "dw.fact_decision_state+dw.fact_outcome", lambda: backend.decision_casebook(candidate_id, universe_id, fold))


@app.get("/research/robustness-compare")
def research_robustness_compare(universe_id: str = OFFICIAL_UNIVERSE_ID) -> dict:
    _validate(universe_id, "universes", "universe")
    if not hasattr(backend, "robustness_compare"):
        return {"count": 0, "rows": []}
    return _timed("/research/robustness-compare", "mart.mv_scorecard_candidate", lambda: backend.robustness_compare(universe_id))


@app.get("/research/distributions")
def research_distributions(
    candidate_id: str = OFFICIAL_CANDIDATE_ID,
    universe_id: str = OFFICIAL_UNIVERSE_ID,
    fold: int | None = None,
) -> dict:
    _validate(candidate_id, "candidates", "candidate")
    _validate(universe_id, "universes", "universe")
    if fold is not None:
        _validate(fold, "folds", "fold")
    if not hasattr(backend, "research_distributions"):
        return {"outcome_percentiles": [], "decision_percentiles": [], "exposure_buckets": []}
    return _timed("/research/distributions", "dw.fact_outcome+dw.fact_decision_state", lambda: backend.research_distributions(candidate_id, universe_id, fold))


@app.get("/research/cohorts")
def research_cohorts(
    candidate_id: str = OFFICIAL_CANDIDATE_ID,
    universe_id: str = OFFICIAL_UNIVERSE_ID,
    fold: int | None = None,
) -> dict:
    _validate(candidate_id, "candidates", "candidate")
    _validate(universe_id, "universes", "universe")
    if fold is not None:
        _validate(fold, "folds", "fold")
    if not hasattr(backend, "research_cohorts"):
        return {"count": 0, "rows": []}
    return _timed("/research/cohorts", "dw.fact_decision_state+dw.fact_outcome", lambda: backend.research_cohorts(candidate_id, universe_id, fold))


@app.get("/research/whatif-reference")
def research_whatif_reference(
    candidate_id: str = OFFICIAL_CANDIDATE_ID,
    universe_id: str = OFFICIAL_UNIVERSE_ID,
    fold: int | None = 1,
    horizon: int = 20,
    cost_bps: float | None = 5.0,
    slippage_bps: float | None = 2.0,
) -> dict:
    _validate(candidate_id, "candidates", "candidate")
    _validate(universe_id, "universes", "universe")
    if fold is not None:
        _validate(fold, "folds", "fold")
    _validate(horizon, "horizons", "horizon")
    if not hasattr(backend, "whatif_reference"):
        return {"official": None, "best_simulated": None, "best_observed": None}
    return _timed("/research/whatif-reference", "mart.mv_scorecard_candidate+mart.mv_whatif_grid", lambda: backend.whatif_reference(candidate_id, universe_id, fold, horizon, cost_bps, slippage_bps))


@app.get("/research/olap-preset")
def research_olap_preset(
    preset_id: str,
    candidate_id: str = OFFICIAL_CANDIDATE_ID,
    universe_id: str = OFFICIAL_UNIVERSE_ID,
    fold: int | None = None,
    limit: int = Query(500, ge=1, le=2000),
) -> dict:
    _validate(candidate_id, "candidates", "candidate")
    _validate(universe_id, "universes", "universe")
    if fold is not None:
        _validate(fold, "folds", "fold")
    if not hasattr(backend, "olap_preset"):
        return {"preset_id": preset_id, "count": 0, "rows": []}
    return _timed("/research/olap-preset", "guided_olap_marts", lambda: backend.olap_preset(preset_id, candidate_id, universe_id, fold, limit))


@app.get("/overview")
def overview(
    candidate_id: str = "B1.05_C1.10_L1.10_R1.05",
    fold: int | None = None,
    universe_id: str = "base_universe_12",
    benchmark: str = "QQQ",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    _validate(candidate_id, "candidates", "candidate")
    _validate(universe_id, "universes", "universe")
    if fold is not None:
        _validate(fold, "folds", "fold")
    _validate(benchmark, "benchmarks", "benchmark")
    return _timed("/overview", "fact_path_recursive+fact_decision_state", lambda: backend.overview(candidate_id, fold, universe_id, benchmark, start_date, end_date))


@app.get("/scorecard")
def scorecard(
    candidate_id: str | None = None,
    universe_id: str | None = None,
    limit: int = Query(200, ge=1, le=5000),
) -> dict:
    _validate(candidate_id, "candidates", "candidate")
    _validate(universe_id, "universes", "universe")
    return _timed("/scorecard", "fact_candidate_metric", lambda: backend.scorecard(candidate_id, universe_id, limit))


@app.get("/robustness/surface")
def robustness_surface(
    metric: str = "Sharpe",
    fold: int | None = None,
    universe_id: str | None = "base_universe_12",
    regime: str | None = None,
    limit: int = Query(5000, ge=1, le=5000),
) -> dict:
    _validate(metric, "metrics", "metric")
    if fold is not None:
        _validate(fold, "folds", "fold")
    _validate(universe_id, "universes", "universe")
    _validate(regime, "regimes", "regime")
    return _timed("/robustness/surface", "fact_robustness_surface", lambda: backend.robustness_surface(metric, fold, universe_id, regime, limit))


@app.get("/whatif/grid")
def whatif_grid(
    candidate_id: str = "B1.05_C1.10_L1.10_R1.05",
    fold: int | None = 1,
    universe_id: str = "base_universe_12",
    horizon: int = 20,
    cost_bps: float | None = 5.0,
    slippage_bps: float | None = 2.0,
    limit: int = Query(5000, ge=1, le=5000),
) -> dict:
    _validate(candidate_id, "candidates", "candidate")
    if fold is not None:
        _validate(fold, "folds", "fold")
    _validate(universe_id, "universes", "universe")
    _validate(horizon, "horizons", "horizon")
    return _timed("/whatif/grid", "fact_whatif", lambda: backend.whatif_grid(candidate_id, fold, universe_id, horizon, cost_bps, slippage_bps, limit))


@app.get("/decision/replay")
def decision_replay(
    candidate_id: str = "B1.05_C1.10_L1.10_R1.05",
    fold: int | None = None,
    universe_id: str = "base_universe_12",
    date: str | None = None,
    ticker: str | None = None,
) -> dict:
    _validate(candidate_id, "candidates", "candidate")
    if fold is not None:
        _validate(fold, "folds", "fold")
    _validate(universe_id, "universes", "universe")
    _validate(ticker, "tickers", "ticker")
    return _timed("/decision/replay", "fact_decision_state+fact_position_daily+fact_module_trace+fact_outcome", lambda: backend.decision_replay(candidate_id, fold, universe_id, date, ticker))


ALLOWED_DIMS = {"candidate_id", "fold", "universe_id", "ticker", "module_name", "regime", "participation_state", "horizon", "date_value", "decision_date"}
ALLOWED_MEASURES = {"return", "alpha", "drawdown", "exposure", "turnover", "helped_rate"}
ALLOWED_OPS = {"slice", "dice", "roll-up", "drill-down", "pivot"}


@app.get("/slice")
def slice_query(
    dimensions: Annotated[list[str], Query()] = ["candidate_id", "fold"],
    measure: str = "alpha",
    operation: str = "slice",
    candidate_id: str | None = None,
    fold: int | None = None,
    universe_id: str | None = None,
    module: str | None = None,
    ticker: str | None = None,
    regime: str | None = None,
    horizon: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = Query(500, ge=1, le=5000),
) -> dict:
    bad_dims = [dim for dim in dimensions if dim not in ALLOWED_DIMS]
    if bad_dims:
        raise HTTPException(status_code=422, detail=f"Invalid dimensions: {bad_dims!r}.")
    if measure not in ALLOWED_MEASURES:
        raise HTTPException(status_code=422, detail=f"Invalid measure: {measure!r}.")
    if operation not in ALLOWED_OPS:
        raise HTTPException(status_code=422, detail=f"Invalid operation: {operation!r}.")
    _validate(candidate_id, "candidates", "candidate")
    if fold is not None:
        _validate(fold, "folds", "fold")
    _validate(universe_id, "universes", "universe")
    _validate(module, "modules", "module")
    _validate(ticker, "tickers", "ticker")
    _validate(regime, "regimes", "regime")
    if horizon is not None:
        _validate(horizon, "horizons", "horizon")
    return _timed(
        "/slice",
        "dynamic_fact",
        lambda: backend.slice_query(dimensions, measure, operation, candidate_id, fold, universe_id, module, ticker, regime, horizon, start_date, end_date, limit),
    )


@app.get("/drilldown")
def drilldown(
    base_dimensions: Annotated[list[str], Query()] = ["candidate_id"],
    next_dimension: str = "fold",
    measure: str = "alpha",
    candidate_id: str | None = None,
    universe_id: str | None = None,
) -> dict:
    dimensions = [*base_dimensions, next_dimension]
    return slice_query(dimensions, measure, "drill-down", candidate_id, None, universe_id, None, None, None, None, None, None, 500)


@app.get("/module/effectiveness")
def module_effectiveness(
    candidate_id: str = "B1.05_C1.10_L1.10_R1.05",
    universe_id: str = "base_universe_12",
    fold: int | None = None,
) -> dict:
    _validate(candidate_id, "candidates", "candidate")
    _validate(universe_id, "universes", "universe")
    if fold is not None:
        _validate(fold, "folds", "fold")
    return _timed("/module/effectiveness", "fact_module_trace+fact_outcome", lambda: backend.module_effectiveness(candidate_id, universe_id, fold))


@app.get("/ticker/contribution")
def ticker_contribution(
    candidate_id: str = "B1.05_C1.10_L1.10_R1.05",
    universe_id: str = "base_universe_12",
    fold: int | None = None,
    limit: int = Query(200, ge=1, le=1000),
) -> dict:
    _validate(candidate_id, "candidates", "candidate")
    _validate(universe_id, "universes", "universe")
    if fold is not None:
        _validate(fold, "folds", "fold")
    return _timed("/ticker/contribution", "fact_position_daily", lambda: backend.ticker_contribution(candidate_id, universe_id, fold, limit))


@app.get("/regime/behavior")
def regime_behavior(
    candidate_id: str = "B1.05_C1.10_L1.10_R1.05",
    universe_id: str = "base_universe_12",
    fold: int | None = None,
) -> dict:
    _validate(candidate_id, "candidates", "candidate")
    _validate(universe_id, "universes", "universe")
    if fold is not None:
        _validate(fold, "folds", "fold")
    return _timed("/regime/behavior", "fact_decision_state", lambda: backend.regime_behavior(candidate_id, universe_id, fold))


@app.get("/fold/performance")
def fold_performance(candidate_id: str | None = None, universe_id: str | None = None) -> dict:
    _validate(candidate_id, "candidates", "candidate")
    _validate(universe_id, "universes", "universe")
    return _timed("/fold/performance", "fact_outcome", lambda: backend.fold_performance(candidate_id, universe_id))


@app.get("/candidate/compare")
def candidate_compare(
    candidates: Annotated[list[str] | None, Query()] = None,
    universe_id: str | None = "base_universe_12",
) -> dict:
    _validate_many(candidates, "candidates", "candidate")
    _validate(universe_id, "universes", "universe")
    return _timed("/candidate/compare", "fact_candidate_metric", lambda: backend.candidate_compare(candidates, universe_id))


@app.get("/query/performance")
def query_performance() -> dict:
    return backend.query_performance() if hasattr(backend, "query_performance") else {"count": 0, "rows": []}
