from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


PHASE_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PHASE_ROOT / "outputs"
OFFICIAL_CANDIDATE_ID = "B1.05_C1.10_L1.10_R1.05"
OFFICIAL_UNIVERSE_ID = "base_universe_12"
NOT_AVAILABLE = "Not available in current cube"
SRC_ROOT = PHASE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from extended_analysis import metric_registry as registry

app = FastAPI(title="Mahoraga 14.3 Extended Analysis API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

figures_dir = OUTPUTS / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
app.mount("/figures", StaticFiles(directory=str(figures_dir)), name="figures")


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@lru_cache(maxsize=32)
def cached_csv(name: str) -> pd.DataFrame:
    return _csv(OUTPUTS / name)


@lru_cache(maxsize=16)
def cached_parquet(name: str) -> pd.DataFrame:
    return _parquet(OUTPUTS / name)


def records(df: pd.DataFrame, limit: int = 500) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"count": 0, "rows": []}
    out = df.head(max(1, min(limit, 5000))).copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    rows = [clean_record(row) for row in out.to_dict("records")]
    return {"count": int(len(df)), "rows": rows}


def clean_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_value(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return clean_value(value.item())
    except AttributeError:
        return value
    except ValueError:
        return value


def clean_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: clean_value(value) for key, value in row.items()}


def first_record(df: pd.DataFrame) -> Dict[str, Any] | None:
    if df.empty:
        return None
    out = df.head(1).copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return clean_record(out.iloc[0].where(pd.notna(out.iloc[0]), None).to_dict())


def unique_values(df: pd.DataFrame, column: str, limit: int = 500) -> List[Any]:
    if df.empty or column not in df.columns:
        return []
    values = df[column].dropna().drop_duplicates()
    try:
        values = values.sort_values()
    except TypeError:
        values = values.astype(str).sort_values()
    return [clean_value(value) for value in values.head(limit).tolist()]


def file_rows(path: Path) -> int:
    if not path.exists():
        return 0
    if path.suffix == ".parquet":
        return int(len(_parquet(path)))
    if path.suffix == ".csv":
        return int(len(_csv(path)))
    return 0


def plateau_report_metrics() -> Dict[str, Any]:
    path = OUTPUTS / "extended_multiplier_robustness" / "plateau_radius_report.md"
    if not path.exists():
        return {"distance_to_decay": None, "robust_region_share_extended": None, "sampled_candidates": None}
    text = path.read_text(encoding="utf-8")

    def number_after(label: str) -> Optional[float]:
        match = re.search(rf"{re.escape(label)}:\s*([0-9.\-]+)%?", text)
        if not match:
            return None
        value = float(match.group(1))
        if "%" in match.group(0):
            value = value / 100.0
        return value

    sampled = number_after("sampled candidates")
    return {
        "distance_to_decay": number_after("distance_to_decay"),
        "robust_region_share_extended": number_after("robust_region_share_extended"),
        "sampled_candidates": int(sampled) if sampled is not None else None,
    }


def pct_mean(series: pd.Series) -> Optional[float]:
    if series.empty:
        return None
    return float(pd.to_numeric(series, errors="coerce").mean())


def candidate_summary() -> pd.DataFrame:
    return cached_csv("extended_multiplier_robustness/extended_multiplier_summary.csv")


def decision_cube() -> pd.DataFrame:
    return cached_parquet("audit_cube/decision_date_cube.parquet")


def position_cube() -> pd.DataFrame:
    return cached_parquet("audit_cube/position_cube.parquet")


def trace_cube() -> pd.DataFrame:
    return cached_parquet("audit_cube/module_trace_cube.parquet")


def outcome_cube() -> pd.DataFrame:
    return cached_parquet("audit_cube/outcome_cube.parquet")


def market_cube() -> pd.DataFrame:
    return cached_parquet("audit_cube/market_context_cube.parquet")


def selected_columns(df: pd.DataFrame, columns: List[str], limit: int = 500) -> Dict[str, Any]:
    return records(df[[col for col in columns if col in df.columns]], limit)


def case_keys_from_df(df: pd.DataFrame, date_col: str = "date", limit: int = 40) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    rename = {date_col: "date"} if date_col != "date" and date_col in df.columns else {}
    out = df.rename(columns=rename)
    columns = ["date", "fold", "candidate_id", "universe_id"]
    available = [col for col in columns if col in out.columns]
    if not available:
        return []
    out = out[available].drop_duplicates()
    if "date" in out.columns:
        out = out.sort_values("date")
    return records(out, limit)["rows"]


def case_key_count(df: pd.DataFrame, date_col: str = "date") -> int:
    if df.empty:
        return 0
    rename = {date_col: "date"} if date_col != "date" and date_col in df.columns else {}
    out = df.rename(columns=rename)
    columns = ["date", "fold", "candidate_id", "universe_id"]
    available = [col for col in columns if col in out.columns]
    if not available:
        return 0
    return int(len(out[available].drop_duplicates()))


def bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(dtype=bool)
    return pd.to_numeric(df[column], errors="coerce").fillna(0).astype(bool)


def apply_common_filters(
    df: pd.DataFrame,
    date_col: str,
    date_start: Optional[str],
    date_end: Optional[str],
    fold: Optional[int],
    candidate_id: Optional[str],
    universe_id: Optional[str],
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if date_col in out.columns and (date_start or date_end):
        dates = pd.to_datetime(out[date_col], errors="coerce")
        if date_start:
            out = out[dates >= pd.Timestamp(date_start)]
            dates = pd.to_datetime(out[date_col], errors="coerce")
        if date_end:
            out = out[dates <= pd.Timestamp(date_end)]
    if fold is not None and "fold" in out.columns:
        out = out[out["fold"] == int(fold)]
    if candidate_id and "candidate_id" in out.columns:
        out = out[out["candidate_id"] == candidate_id]
    if universe_id and "universe_id" in out.columns:
        out = out[out["universe_id"] == universe_id]
    return out


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "phase_root": str(PHASE_ROOT),
        "outputs_exist": OUTPUTS.exists(),
    }


@app.get("/summary/baseline")
def baseline_summary() -> Dict[str, Any]:
    summary = cached_csv("extended_multiplier_robustness/extended_multiplier_summary.csv")
    universe = cached_csv("universe_robustness/universe_robustness_summary.csv")
    official = summary[summary["CandidateId"] == "B1.05_C1.10_L1.10_R1.05"].head(1) if not summary.empty else pd.DataFrame()
    return {
        "official": records(official, 1)["rows"][0] if len(official) else None,
        "robust_region_share_extended": float(summary["robust_region_flag"].mean()) if "robust_region_flag" in summary.columns and len(summary) else None,
        "sampled_candidates": int(len(summary)),
        "universe_runs": int(len(universe)),
        "figures": {
            "heatmap": "/figures/extended_multiplier_heatmap.png",
            "one_dimensional": "/figures/multiplier_1d_degradation.png",
            "universe": "/figures/universe_robustness_comparison.png",
        },
    }


@app.get("/robustness/multipliers")
def multipliers(
    axis: Optional[str] = None,
    candidate_id: Optional[str] = None,
    robust_only: bool = False,
    limit: int = Query(500, ge=1, le=5000),
) -> Dict[str, Any]:
    df = cached_csv("extended_multiplier_robustness/extended_multiplier_summary.csv")
    if axis:
        df = df[df["sweep_role"].astype(str).str.contains(axis, regex=False)]
    if candidate_id:
        df = df[df["CandidateId"] == candidate_id]
    if robust_only and "robust_region_flag" in df.columns:
        df = df[df["robust_region_flag"] == 1]
    return records(df, limit)


@app.get("/robustness/plateau")
def plateau() -> Dict[str, Any]:
    plateau_df = cached_csv("extended_multiplier_robustness/plateau_radius_by_axis.csv")
    sensitivity = cached_csv("extended_multiplier_robustness/sensitivity_ranking.csv")
    report = OUTPUTS / "extended_multiplier_robustness" / "plateau_radius_report.md"
    return {
        "plateau": records(plateau_df, 100)["rows"],
        "sensitivity": records(sensitivity, 100)["rows"],
        "report": report.read_text(encoding="utf-8") if report.exists() else "",
    }


@app.get("/decisions")
def decisions(
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    fold: Optional[int] = None,
    candidate_id: Optional[str] = None,
    universe_id: Optional[str] = None,
    limit: int = Query(500, ge=1, le=5000),
) -> Dict[str, Any]:
    df = cached_parquet("audit_cube/decision_date_cube.parquet")
    df = apply_common_filters(df, "date", date_start, date_end, fold, candidate_id, universe_id)
    return records(df, limit)


@app.get("/positions")
def positions(
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    fold: Optional[int] = None,
    candidate_id: Optional[str] = None,
    universe_id: Optional[str] = None,
    ticker: Optional[str] = None,
    selected_only: bool = False,
    limit: int = Query(500, ge=1, le=5000),
) -> Dict[str, Any]:
    df = cached_parquet("audit_cube/position_cube.parquet")
    df = apply_common_filters(df, "date", date_start, date_end, fold, candidate_id, universe_id)
    if ticker and "ticker" in df.columns:
        df = df[df["ticker"] == ticker.upper()]
    if selected_only and "selected_flag" in df.columns:
        df = df[df["selected_flag"] == 1]
    return records(df, limit)


@app.get("/module-trace")
def module_trace(
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    fold: Optional[int] = None,
    candidate_id: Optional[str] = None,
    universe_id: Optional[str] = None,
    module_name: Optional[str] = None,
    limit: int = Query(500, ge=1, le=5000),
) -> Dict[str, Any]:
    df = cached_parquet("audit_cube/module_trace_cube.parquet")
    df = apply_common_filters(df, "date", date_start, date_end, fold, candidate_id, universe_id)
    if module_name and "module_name" in df.columns:
        df = df[df["module_name"] == module_name]
    return records(df, limit)


@app.get("/market-context")
def market_context(
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    limit: int = Query(500, ge=1, le=5000),
) -> Dict[str, Any]:
    df = cached_parquet("audit_cube/market_context_cube.parquet")
    df = apply_common_filters(df, "date", date_start, date_end, None, None, None)
    return records(df, limit)


@app.get("/outcomes")
def outcomes(
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    fold: Optional[int] = None,
    candidate_id: Optional[str] = None,
    universe_id: Optional[str] = None,
    horizon: Optional[int] = None,
    limit: int = Query(500, ge=1, le=5000),
) -> Dict[str, Any]:
    df = cached_parquet("audit_cube/outcome_cube.parquet")
    df = apply_common_filters(df, "decision_date", date_start, date_end, fold, candidate_id, universe_id)
    if horizon is not None and "horizon" in df.columns:
        df = df[pd.to_numeric(df["horizon"], errors="coerce") == horizon]
    return records(df, limit)


@app.get("/universes/summary")
def universes_summary(limit: int = Query(500, ge=1, le=5000)) -> Dict[str, Any]:
    df = cached_csv("universe_robustness/universe_robustness_summary.csv")
    return records(df, limit)


@app.get("/metadata/options")
def metadata_options() -> Dict[str, Any]:
    decisions_df = decision_cube()
    positions_df = position_cube()
    traces_df = trace_cube()
    outcomes_df = outcome_cube()
    universes_df = cached_csv("universe_robustness/universe_robustness_summary.csv")

    candidates = set(unique_values(decisions_df, "candidate_id"))
    candidates.update(unique_values(candidate_summary(), "candidate_id"))
    universes = set(unique_values(decisions_df, "universe_id"))
    universes.update(unique_values(universes_df, "universe_id"))

    return {
        "candidates": sorted(candidates),
        "universes": sorted(universes),
        "folds": unique_values(decisions_df, "fold"),
        "tickers": unique_values(positions_df, "ticker"),
        "modules": unique_values(traces_df, "module_name"),
        "horizons": unique_values(outcomes_df, "horizon"),
    }


@app.get("/dss/scorecard")
def dss_scorecard() -> Dict[str, Any]:
    return registry.build_scorecard()


@app.get("/dss/research-questions")
def dss_research_questions() -> Dict[str, Any]:
    return registry.research_questions()


@app.get("/dss/candidates")
def dss_candidates() -> Dict[str, Any]:
    return registry.candidate_metadata()


@app.get("/dss/folds")
def dss_folds() -> Dict[str, Any]:
    return registry.fold_summaries()


@app.get("/dss/model-diagnostics")
def dss_model_diagnostics() -> Dict[str, Any]:
    return registry.model_diagnostics()


@app.get("/dss/performance-risk")
def dss_performance_risk() -> Dict[str, Any]:
    return registry.performance_risk()


@app.get("/dss/decision-cases")
def dss_decision_cases(
    preset_id: str = "official-baseline",
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    fold: Optional[int] = None,
    candidate_id: Optional[str] = None,
    universe_id: Optional[str] = None,
    limit: int = Query(120, ge=1, le=500),
) -> Dict[str, Any]:
    return registry.decision_cases(
        preset_id,
        date_start=date_start,
        date_end=date_end,
        fold=fold,
        candidate_id=candidate_id,
        universe_id=universe_id,
        limit=limit,
    )


@app.get("/dss/cube-operations")
def dss_cube_operations() -> Dict[str, Any]:
    return registry.cube_operations()


@app.get("/dss/overview")
def dss_overview() -> Dict[str, Any]:
    summary = candidate_summary()
    universe = cached_csv("universe_robustness/universe_robustness_summary.csv")
    sensitivity = cached_csv("extended_multiplier_robustness/sensitivity_ranking.csv")
    plateau_metrics = plateau_report_metrics()

    official_df = summary[summary["CandidateId"] == OFFICIAL_CANDIDATE_ID] if "CandidateId" in summary.columns else pd.DataFrame()
    official = first_record(official_df)
    robust_share = plateau_metrics["robust_region_share_extended"]
    if robust_share is None and "robust_region_flag" in summary.columns and len(summary):
        robust_share = float(pd.to_numeric(summary["robust_region_flag"], errors="coerce").mean())

    sensitivity_row = first_record(sensitivity.head(1)) if not sensitivity.empty else None
    main_sensitivity = sensitivity_row.get("axis") if sensitivity_row else NOT_AVAILABLE

    official_universe = universe[universe["candidate_id"] == OFFICIAL_CANDIDATE_ID] if "candidate_id" in universe.columns else pd.DataFrame()
    universe_ok = official_universe[official_universe.get("run_status", "") == "OK"] if not official_universe.empty and "run_status" in official_universe.columns else official_universe
    best_universe = first_record(universe_ok.sort_values("Sharpe", ascending=False).head(1)) if "Sharpe" in universe_ok.columns and not universe_ok.empty else None

    audit_dir = OUTPUTS / "audit_cube"
    audit_files = [
        "decision_date_cube.parquet",
        "position_cube.parquet",
        "module_trace_cube.parquet",
        "outcome_cube.parquet",
        "market_context_cube.parquet",
        "backoff_audit.csv",
        "continuation_activation_audit.csv",
        "leader_participation_audit.csv",
    ]
    artifacts = [
        {
            "file": name,
            "path": f"outputs/audit_cube/{name}",
            "rows": file_rows(audit_dir / name),
            "available": (audit_dir / name).exists(),
        }
        for name in audit_files
    ]

    narrative = [
        "The official baseline remains the frozen research reference; this DSS reads existing extended-analysis artifacts only.",
        "The extended robustness sample does not show global parameter fragility across every axis.",
        "The main documented sensitivity is budget underdeployment, so lower long-budget participation receives special review.",
        "Universe robustness is strongest in the original base universe and remains positive across completed technology/growth universe stresses.",
    ]

    return {
        "official_candidate_id": OFFICIAL_CANDIDATE_ID,
        "official_universe_id": OFFICIAL_UNIVERSE_ID,
        "official_metrics": official,
        "robustness_summary": {
            "robust_region_share_extended": robust_share,
            "distance_to_decay": plateau_metrics["distance_to_decay"],
            "sampled_candidates": plateau_metrics["sampled_candidates"] or int(len(summary)),
            "most_sensitive_axis": main_sensitivity,
        },
        "main_sensitivity": sensitivity_row,
        "universe_summary": records(official_universe, 20)["rows"],
        "best_universe": best_universe,
        "artifacts": artifacts,
        "narrative": narrative,
    }


@app.get("/dss/robustness/budget")
def dss_robustness_budget() -> Dict[str, Any]:
    summary = candidate_summary()
    if summary.empty:
        return {"rows": [], "interpretation": [NOT_AVAILABLE]}

    budget = summary[
        (summary["CandidateId"] == OFFICIAL_CANDIDATE_ID)
        | (summary["sweep_role"].astype(str).str.contains("budget_multiplier", regex=False))
    ].copy()
    if "budget_multiplier" in budget.columns:
        budget = budget.sort_values("budget_multiplier")
    keep = [
        "CandidateId",
        "candidate_id",
        "sweep_role",
        "budget_multiplier",
        "CAGR",
        "Sharpe",
        "Sortino",
        "MaxDD",
        "AlphaNW_QQQ",
        "AlphaNW_SPY",
        "robust_region_flag",
        "severe_fold_damage_count",
        "worst_fold_sharpe_delta_vs_official",
        "worst_fold_cagr_delta_vs_official",
        "max_fold_maxdd_worsening_vs_official",
    ]
    rows = selected_columns(budget, keep, 50)["rows"]

    lower = budget[pd.to_numeric(budget.get("budget_multiplier"), errors="coerce") < 1.05]
    higher = budget[pd.to_numeric(budget.get("budget_multiplier"), errors="coerce") > 1.05]
    lower_damage = int(pd.to_numeric(lower.get("severe_fold_damage_count"), errors="coerce").fillna(0).sum()) if not lower.empty else 0
    higher_damage = int(pd.to_numeric(higher.get("severe_fold_damage_count"), errors="coerce").fillna(0).sum()) if not higher.empty else 0

    interpretation = [
        "Budget-axis rows compare the official candidate against sampled one-dimensional long-budget perturbations.",
        f"Lower-budget samples record {lower_damage} severe fold-damage flags across the displayed rows.",
        f"Above-official budget samples record {higher_damage} severe fold-damage flags across the displayed rows.",
        "The documented sensitivity is asymmetric: reducing budget damages the model more than the moderate upward perturbations sampled here.",
    ]

    return {"rows": rows, "interpretation": interpretation}


@app.get("/dss/robustness/plateau")
def dss_robustness_plateau() -> Dict[str, Any]:
    plateau_df = cached_csv("extended_multiplier_robustness/plateau_radius_by_axis.csv")
    sensitivity = cached_csv("extended_multiplier_robustness/sensitivity_ranking.csv")
    summary = candidate_summary()

    interpretation_by_axis = {
        "budget_multiplier": "Budget has asymmetric sampled robustness; the robust sampled interval starts at the official value and extends upward.",
        "conviction_multiplier": "Conviction has broader sampled tolerance around the official point.",
        "leader_multiplier": "Leader participation has broader sampled tolerance around the official point.",
        "backoff_strength": "Backoff strength has moderate sampled tolerance around the official point.",
    }
    plateau_rows = []
    for row in records(plateau_df, 100)["rows"]:
        axis = row.get("axis")
        row["interpretation"] = interpretation_by_axis.get(str(axis), "No deterministic interpretation available for this axis.")
        plateau_rows.append(row)

    worst = pd.DataFrame()
    if not summary.empty and "severe_fold_damage_count" in summary.columns:
        worst = summary.copy()
        worst["severe_fold_damage_count"] = pd.to_numeric(worst["severe_fold_damage_count"], errors="coerce").fillna(0)
        worst = worst.sort_values(["severe_fold_damage_count", "worst_fold_cagr_delta_vs_official"], ascending=[False, True])
    worst_cols = [
        "CandidateId",
        "sweep_role",
        "CAGR",
        "Sharpe",
        "MaxDD",
        "severe_fold_damage_count",
        "worst_fold_sharpe_delta_vs_official",
        "worst_fold_cagr_delta_vs_official",
        "max_fold_maxdd_worsening_vs_official",
    ]

    return {
        "plateau": plateau_rows,
        "sensitivity": records(sensitivity, 100)["rows"],
        "worst_fold_degradation": selected_columns(worst, worst_cols, 15)["rows"],
        "interpretation": [
            "Plateau radius is computed from sampled one-dimensional perturbations around the official point.",
            "Fold-local damage is separated from stitched aggregate metrics because a strong total series can hide weak walk-forward segments.",
        ],
    }


@app.get("/dss/presets")
def dss_presets() -> Dict[str, Any]:
    decisions_df = decision_cube()
    positions_df = position_cube()
    traces_df = trace_cube()
    outcomes_df = outcome_cube()
    summary = candidate_summary()

    def preset(pid: str, label: str, description: str, df: pd.DataFrame, date_col: str = "date") -> Dict[str, Any]:
        return {
            "id": pid,
            "label": label,
            "description": description,
            "count": case_key_count(df, date_col=date_col),
            "sample_decisions": case_keys_from_df(df, date_col=date_col, limit=30),
        }

    presets: List[Dict[str, Any]] = []
    official = decisions_df[decisions_df["candidate_id"] == OFFICIAL_CANDIDATE_ID] if "candidate_id" in decisions_df.columns else pd.DataFrame()
    presets.append(preset("official-baseline", "Official baseline decisions", "Decision dates for the frozen official candidate.", official))

    hard_backoff = decisions_df[bool_series(decisions_df, "hard_backoff_flag")] if "hard_backoff_flag" in decisions_df.columns else pd.DataFrame()
    presets.append(preset("hard-backoff", "Hard backoff dates", "Decision dates where the hard backoff flag is active.", hard_backoff))

    if "structural_p" in decisions_df.columns:
        structural_values = pd.to_numeric(decisions_df["structural_p"], errors="coerce")
        structural_threshold = structural_values.quantile(0.75)
        high_structural = decisions_df[structural_values >= structural_threshold]
    else:
        high_structural = pd.DataFrame()
    presets.append(preset("high-structural-risk", "High structural risk dates", "Top-quartile structural probability decision dates.", high_structural))

    if "long_budget" in decisions_df.columns:
        long_budget = pd.to_numeric(decisions_df["long_budget"], errors="coerce")
        high_budget = decisions_df[long_budget >= long_budget.quantile(0.75)]
        low_budget = decisions_df[long_budget <= long_budget.quantile(0.25)]
    else:
        high_budget = low_budget = pd.DataFrame()
    presets.append(preset("high-long-budget", "High long-budget dates", "Top-quartile long-budget decision dates.", high_budget))
    presets.append(preset("low-long-budget", "Low long-budget dates", "Bottom-quartile long-budget decision dates.", low_budget))

    presets.append(preset("fold-1", "Fold 1", "Decision dates in walk-forward fold 1.", decisions_df[decisions_df.get("fold") == 1] if "fold" in decisions_df.columns else pd.DataFrame()))
    presets.append(preset("fold-5", "Fold 5", "Decision dates in walk-forward fold 5.", decisions_df[decisions_df.get("fold") == 5] if "fold" in decisions_df.columns else pd.DataFrame()))

    nvda = pd.DataFrame()
    if not positions_df.empty and {"ticker", "selected_flag"}.issubset(positions_df.columns):
        nvda = positions_df[(positions_df["ticker"] == "NVDA") & bool_series(positions_df, "selected_flag")]
    presets.append(preset("nvda-selected", "NVDA selected", "Decision dates where NVDA is selected.", nvda))

    leader = pd.DataFrame()
    if not positions_df.empty and {"leader_flag", "selected_flag"}.issubset(positions_df.columns):
        leader = positions_df[bool_series(positions_df, "leader_flag") & bool_series(positions_df, "selected_flag")]
    presets.append(preset("leader-active", "Leader participation active", "Decision dates with selected leader-flagged positions.", leader))

    continuation = decisions_df[pd.to_numeric(decisions_df.get("continuation_trigger_p"), errors="coerce") >= 0.5] if "continuation_trigger_p" in decisions_df.columns else pd.DataFrame()
    presets.append(preset("continuation-active", "Continuation active", "Decision dates with continuation trigger probability at or above 0.50.", continuation))

    structural_defense = pd.DataFrame()
    if not traces_df.empty and {"module_name", "branch_taken"}.issubset(traces_df.columns):
        structural_defense = traces_df[
            (traces_df["module_name"] == "structural_defense_model")
            & ~traces_df["branch_taken"].astype(str).str.contains("no_defense", case=False, regex=False)
        ]
    presets.append(preset("structural-defense-active", "Structural defense active", "Decision dates where structural defense takes an active branch.", structural_defense))

    beat_20d = pd.DataFrame()
    fail_20d = pd.DataFrame()
    if not outcomes_df.empty and {"horizon", "decision_helped_flag_vs_qqq"}.issubset(outcomes_df.columns):
        twenty = outcomes_df[pd.to_numeric(outcomes_df["horizon"], errors="coerce") == 20]
        beat_20d = twenty[bool_series(twenty, "decision_helped_flag_vs_qqq")]
        fail_20d = twenty[~bool_series(twenty, "decision_helped_flag_vs_qqq")]
    presets.append(preset("beat-qqq-20d", "Decisions that beat QQQ after 20d", "Decision dates with positive 20-day helped flag versus QQQ.", beat_20d, date_col="decision_date"))
    presets.append(preset("failed-qqq-20d", "Decisions that failed vs QQQ after 20d", "Decision dates without positive 20-day helped flag versus QQQ.", fail_20d, date_col="decision_date"))

    budget_sensitive = pd.DataFrame()
    if not summary.empty and {"sweep_role", "candidate_id", "robust_region_flag"}.issubset(summary.columns):
        sensitive_ids = summary[
            summary["sweep_role"].astype(str).str.contains("budget_multiplier", regex=False)
            & (pd.to_numeric(summary["robust_region_flag"], errors="coerce").fillna(0) == 0)
        ]["candidate_id"].dropna().unique()
        budget_sensitive = decisions_df[decisions_df["candidate_id"].isin(sensitive_ids)] if "candidate_id" in decisions_df.columns else pd.DataFrame()
    presets.append(preset("budget-sensitive", "Budget-sensitive candidate cases", "Decision dates from budget-axis candidates that fail robustness conditions.", budget_sensitive))

    return {"presets": presets}


@app.get("/dss/decision-detail")
def dss_decision_detail(
    date: Optional[str] = None,
    fold: Optional[int] = None,
    candidate_id: str = OFFICIAL_CANDIDATE_ID,
    universe_id: str = OFFICIAL_UNIVERSE_ID,
) -> Dict[str, Any]:
    decisions_df = decision_cube()
    if decisions_df.empty:
        return {"decision": None, "positions": [], "modules": [], "outcomes": [], "market_context": None, "interpretation": [NOT_AVAILABLE]}

    selected = decisions_df.copy()
    if "candidate_id" in selected.columns:
        selected = selected[selected["candidate_id"] == candidate_id]
    if "universe_id" in selected.columns:
        selected = selected[selected["universe_id"] == universe_id]
    if date and "date" in selected.columns:
        selected = selected[pd.to_datetime(selected["date"], errors="coerce").dt.strftime("%Y-%m-%d") == date]
    if fold is not None and "fold" in selected.columns:
        selected = selected[selected["fold"] == fold]
    if selected.empty and date:
        selected = decisions_df[pd.to_datetime(decisions_df["date"], errors="coerce").dt.strftime("%Y-%m-%d") == date] if "date" in decisions_df.columns else pd.DataFrame()
    if selected.empty:
        selected = decisions_df.head(1)

    decision = first_record(selected)
    if not decision:
        return {"decision": None, "positions": [], "modules": [], "outcomes": [], "market_context": None, "interpretation": [NOT_AVAILABLE]}

    ddate = pd.Timestamp(decision.get("date")).strftime("%Y-%m-%d") if decision.get("date") else ""
    dfold = decision.get("fold")
    dcandidate = str(decision.get("candidate_id"))
    duniverse = str(decision.get("universe_id"))

    positions_df = position_cube()
    pos = apply_common_filters(positions_df, "date", ddate, ddate, int(dfold) if dfold is not None else None, dcandidate, duniverse)
    if "selected_flag" in pos.columns:
        pos = pos[bool_series(pos, "selected_flag")]
    if "final_weight" in pos.columns:
        pos = pos.sort_values("final_weight", ascending=False)

    traces_df = trace_cube()
    traces = apply_common_filters(traces_df, "date", ddate, ddate, int(dfold) if dfold is not None else None, dcandidate, duniverse)

    outcomes_df = outcome_cube()
    outcomes = outcomes_df.copy()
    if "decision_date" in outcomes.columns:
        outcomes = outcomes[pd.to_datetime(outcomes["decision_date"], errors="coerce").dt.strftime("%Y-%m-%d") == ddate]
    if "fold" in outcomes.columns and dfold is not None:
        outcomes = outcomes[outcomes["fold"] == int(dfold)]
    if "candidate_id" in outcomes.columns:
        outcomes = outcomes[outcomes["candidate_id"] == dcandidate]
    if "universe_id" in outcomes.columns:
        outcomes = outcomes[outcomes["universe_id"] == duniverse]
    if "horizon" in outcomes.columns:
        outcomes = outcomes.sort_values("horizon")

    market_df = market_cube()
    market = market_df[pd.to_datetime(market_df["date"], errors="coerce").dt.strftime("%Y-%m-%d") == ddate] if not market_df.empty and "date" in market_df.columns else pd.DataFrame()

    interpretation: List[str] = []
    participation = decision.get("participation_state") or NOT_AVAILABLE
    interpretation.append(f"The model was in {participation}.")
    long_budget = decision.get("long_budget")
    interpretation.append(f"Long budget was {long_budget:.4f}." if isinstance(long_budget, (float, int)) else f"Long budget: {NOT_AVAILABLE}.")
    hard_backoff = bool(decision.get("hard_backoff_flag"))
    interpretation.append("Hard backoff was active on this date." if hard_backoff else "No hard backoff was active on this date.")
    continuation_p = decision.get("continuation_trigger_p")
    if isinstance(continuation_p, (float, int)):
        interpretation.append("Continuation probability was supportive." if continuation_p >= 0.5 else "Continuation probability was below 0.50.")
    else:
        interpretation.append(f"Continuation probability: {NOT_AVAILABLE}.")
    structural_p = decision.get("structural_p")
    if isinstance(structural_p, (float, int)):
        interpretation.append("Structural risk was elevated." if structural_p >= 0.5 else "Structural risk was not elevated by the 0.50 threshold.")
    else:
        interpretation.append(f"Structural risk: {NOT_AVAILABLE}.")

    twenty = outcomes[pd.to_numeric(outcomes.get("horizon"), errors="coerce") == 20] if not outcomes.empty and "horizon" in outcomes.columns else pd.DataFrame()
    twenty_row = first_record(twenty)
    if twenty_row and "decision_helped_flag_vs_qqq" in twenty_row:
        helped = bool(twenty_row.get("decision_helped_flag_vs_qqq"))
        interpretation.append("The decision beat QQQ over the 20-day horizon." if helped else "The decision failed versus QQQ over the 20-day horizon.")
        if isinstance(long_budget, (float, int)) and long_budget >= 0.75:
            interpretation.append("This case supports the participation thesis in the selected horizon." if helped else "This case should be reviewed because high participation was followed by weak relative outcome.")
    else:
        interpretation.append(f"20-day QQQ helped flag: {NOT_AVAILABLE}.")

    def outcome_chip(horizon: int, column: str, label: str) -> Dict[str, Any]:
        if outcomes.empty or "horizon" not in outcomes.columns or column not in outcomes.columns:
            return {"label": label, "value": None, "status": "unknown"}
        subset = outcomes[pd.to_numeric(outcomes["horizon"], errors="coerce") == horizon]
        row = first_record(subset)
        value = row.get(column) if row else None
        if value is None:
            return {"label": label, "value": None, "status": "unknown"}
        passed = bool(value)
        return {"label": label, "value": passed, "status": "positive" if passed else "negative"}

    comparison_chips = [
        outcome_chip(1, "decision_helped_flag_vs_qqq", "Beat QQQ at 1d?"),
        outcome_chip(5, "decision_helped_flag_vs_qqq", "Beat QQQ at 5d?"),
        outcome_chip(20, "decision_helped_flag_vs_qqq", "Beat QQQ at 20d?"),
        outcome_chip(20, "decision_helped_flag_vs_control", "Beat control?"),
        {"label": "Hard backoff?", "value": hard_backoff, "status": "active" if hard_backoff else "inactive"},
        {"label": "Leader active?", "value": bool(pd.to_numeric(pos.get("leader_flag"), errors="coerce").fillna(0).sum() > 0) if not pos.empty and "leader_flag" in pos.columns else None, "status": "active" if not pos.empty and "leader_flag" in pos.columns and pd.to_numeric(pos.get("leader_flag"), errors="coerce").fillna(0).sum() > 0 else "inactive"},
        {"label": "Continuation active?", "value": bool(isinstance(continuation_p, (float, int)) and continuation_p >= 0.5), "status": "active" if isinstance(continuation_p, (float, int)) and continuation_p >= 0.5 else "inactive"},
    ]

    return {
        "decision": decision,
        "positions": records(pos, 20)["rows"],
        "modules": records(traces, 30)["rows"],
        "outcomes": records(outcomes, 20)["rows"],
        "market_context": first_record(market),
        "interpretation": interpretation,
        "comparison_chips": comparison_chips,
        "data_sources": [
            "research/mahoraga14_3_extended_analysis/outputs/audit_cube/decision_date_cube.parquet",
            "research/mahoraga14_3_extended_analysis/outputs/audit_cube/position_cube.parquet",
            "research/mahoraga14_3_extended_analysis/outputs/audit_cube/module_trace_cube.parquet",
            "research/mahoraga14_3_extended_analysis/outputs/audit_cube/outcome_cube.parquet",
            "research/mahoraga14_3_extended_analysis/outputs/audit_cube/market_context_cube.parquet",
        ],
    }


@app.get("/dss/module-effectiveness")
def dss_module_effectiveness(
    candidate_id: str = OFFICIAL_CANDIDATE_ID,
    universe_id: str = OFFICIAL_UNIVERSE_ID,
) -> Dict[str, Any]:
    outcomes = outcome_cube()
    decisions_df = decision_cube()
    positions_df = position_cube()
    traces_df = trace_cube()

    if "candidate_id" in outcomes.columns:
        outcomes = outcomes[outcomes["candidate_id"] == candidate_id]
    if "universe_id" in outcomes.columns:
        outcomes = outcomes[outcomes["universe_id"] == universe_id]

    def flag_summary(flag_col: str, label: str) -> List[Dict[str, Any]]:
        if outcomes.empty or flag_col not in outcomes.columns or "horizon" not in outcomes.columns:
            return []
        grouped = outcomes.groupby("horizon").agg(
            count=(flag_col, "count"),
            helped_rate=(flag_col, "mean"),
            avg_alpha_vs_qqq=("realized_alpha_vs_qqq", "mean"),
        ).reset_index()
        grouped.insert(0, "module", label)
        return records(grouped, 20)["rows"]

    continuation = flag_summary("continuation_helped_flag", "Continuation")
    leader = flag_summary("leader_helped_flag", "Leader participation")
    backoff = flag_summary("backoff_helped_flag", "Backoff")

    dec_filtered = decisions_df
    if not dec_filtered.empty and "candidate_id" in dec_filtered.columns:
        dec_filtered = dec_filtered[dec_filtered["candidate_id"] == candidate_id]
    if not dec_filtered.empty and "universe_id" in dec_filtered.columns:
        dec_filtered = dec_filtered[dec_filtered["universe_id"] == universe_id]
    backoff_counts = {
        "backoff_count": int((pd.to_numeric(dec_filtered.get("backoff_strength_applied"), errors="coerce").fillna(0) > 0).sum()) if "backoff_strength_applied" in dec_filtered.columns else None,
        "hard_backoff_count": int(bool_series(dec_filtered, "hard_backoff_flag").sum()) if "hard_backoff_flag" in dec_filtered.columns else None,
    }

    leader_positions = positions_df
    if not leader_positions.empty and "candidate_id" in leader_positions.columns:
        leader_positions = leader_positions[leader_positions["candidate_id"] == candidate_id]
    if not leader_positions.empty and "universe_id" in leader_positions.columns:
        leader_positions = leader_positions[leader_positions["universe_id"] == universe_id]
    if not leader_positions.empty and {"leader_flag", "selected_flag"}.issubset(leader_positions.columns):
        leader_positions = leader_positions[bool_series(leader_positions, "leader_flag") & bool_series(leader_positions, "selected_flag")]
    top_leaders = pd.DataFrame()
    if not leader_positions.empty and "ticker" in leader_positions.columns:
        top_leaders = leader_positions.groupby("ticker").agg(
            selected_frequency=("selected_flag", "sum"),
            pnl_contribution=("pnl_contribution", "sum"),
            mean_final_weight=("final_weight", "mean"),
        ).reset_index().sort_values("selected_frequency", ascending=False)

    module_states = pd.DataFrame()
    if not traces_df.empty and {"module_name", "branch_taken"}.issubset(traces_df.columns):
        if "candidate_id" in traces_df.columns:
            traces_df = traces_df[traces_df["candidate_id"] == candidate_id]
        if "universe_id" in traces_df.columns:
            traces_df = traces_df[traces_df["universe_id"] == universe_id]
        module_states = traces_df.groupby(["module_name", "branch_taken"]).agg(
            observations=("branch_taken", "count"),
            mean_signal_strength=("signal_strength", "mean"),
            threshold_cross_rate=("threshold_crossed", "mean"),
        ).reset_index().sort_values("observations", ascending=False)

    fold_behavior = pd.DataFrame()
    if not outcomes.empty and {"fold", "horizon"}.issubset(outcomes.columns):
        fold_behavior = outcomes.groupby(["fold", "horizon"]).agg(
            count=("horizon", "count"),
            helped_rate=("decision_helped_flag_vs_qqq", "mean"),
            avg_alpha_vs_qqq=("realized_alpha_vs_qqq", "mean"),
            avg_exposure=("realized_exposure", "mean"),
        ).reset_index().sort_values(["fold", "horizon"])

    def rate_at(rows: List[Dict[str, Any]], horizon: int) -> Optional[float]:
        for row in rows:
            if int(row.get("horizon", -1)) == horizon and row.get("helped_rate") is not None:
                return float(row["helped_rate"])
        return None

    continuation_1d = rate_at(continuation, 1)
    continuation_20d = rate_at(continuation, 20)
    leader_1d = rate_at(leader, 1)
    leader_20d = rate_at(leader, 20)
    backoff_1d = rate_at(backoff, 1)
    backoff_20d = rate_at(backoff, 20)
    weakest_fold = first_record(fold_behavior.sort_values("helped_rate").head(1)) if not fold_behavior.empty and "helped_rate" in fold_behavior.columns else None
    explanations = {
        "continuation": [
            "Continuation helped rates are grouped directly from outcome_cube by horizon.",
            "Continuation appears more useful at longer horizons in current audit artifacts." if continuation_20d is not None and continuation_1d is not None and continuation_20d > continuation_1d else "Current continuation helped rates do not improve from 1d to 20d.",
        ],
        "leader": [
            "Leader helped rates are grouped directly from outcome_cube by horizon.",
            "Leader participation appears more useful at longer horizons in current audit artifacts." if leader_20d is not None and leader_1d is not None and leader_20d > leader_1d else "Current leader helped rates do not improve from 1d to 20d.",
            "Leader participation should be read together with technology/growth leadership concentration and ticker contribution.",
        ],
        "backoff": [
            f"Backoff count: {backoff_counts.get('backoff_count')}.",
            f"Hard backoff count: {backoff_counts.get('hard_backoff_count')}.",
            "Backoff helped rate improves at longer horizon in current audit artifacts." if backoff_20d is not None and backoff_1d is not None and backoff_20d > backoff_1d else "Current backoff helped rates do not show clear improvement from 1d to 20d.",
            "Helped rate is ex-post association, not causal proof.",
        ],
        "tickers": [
            "Ticker contribution is influenced by selected frequency, weight and realized returns.",
            "High contribution should be read together with selected frequency and mean weight.",
        ],
        "folds": [
            f"Weakest helped-rate cell is fold {weakest_fold.get('fold')} at horizon {weakest_fold.get('horizon')}d." if weakest_fold else NOT_AVAILABLE,
            "Fold weakness can be short-horizon or long-horizon, so both horizon and alpha columns matter.",
        ],
    }

    return {
        "continuation": continuation,
        "leader": leader,
        "backoff": backoff,
        "backoff_counts": backoff_counts,
        "top_leader_tickers": records(top_leaders, 12)["rows"],
        "module_states": records(module_states, 30)["rows"],
        "fold_behavior": records(fold_behavior, 60)["rows"],
        "interpretation": [
            "Helped rates are simple deterministic means of helped-flag columns in the outcome cube.",
            "Module activation counts are read from module trace rows and do not infer hidden model logic.",
        ],
        "explanations": explanations,
    }


@app.get("/dss/ticker-contribution")
def dss_ticker_contribution(
    candidate_id: str = OFFICIAL_CANDIDATE_ID,
    universe_id: str = OFFICIAL_UNIVERSE_ID,
) -> Dict[str, Any]:
    positions_df = position_cube()
    if positions_df.empty:
        return {"top_positive": [], "top_negative": [], "all": []}
    if "candidate_id" in positions_df.columns:
        positions_df = positions_df[positions_df["candidate_id"] == candidate_id]
    if "universe_id" in positions_df.columns:
        positions_df = positions_df[positions_df["universe_id"] == universe_id]

    grouped = positions_df.groupby("ticker").agg(
        selected_frequency=("selected_flag", "sum"),
        leader_frequency=("leader_flag", "sum"),
        total_pnl_contribution=("pnl_contribution", "sum"),
        mean_final_weight=("final_weight", "mean"),
        mean_base_score=("base_score", "mean"),
    ).reset_index()
    positive = grouped.sort_values("total_pnl_contribution", ascending=False)
    negative = grouped.sort_values("total_pnl_contribution", ascending=True)

    return {
        "top_positive": records(positive, 10)["rows"],
        "top_negative": records(negative, 10)["rows"],
        "all": records(grouped.sort_values("selected_frequency", ascending=False), 100)["rows"],
    }


@app.get("/dss/data-cubes")
def dss_data_cubes() -> Dict[str, Any]:
    cube_names = [
        "decision_date_cube",
        "position_cube",
        "module_trace_cube",
        "outcome_cube",
        "market_context_cube",
    ]
    grains = {
        "decision_date_cube": "date x fold x candidate x universe",
        "position_cube": "date x ticker x fold x candidate x universe",
        "module_trace_cube": "date x module x fold x candidate x universe",
        "outcome_cube": "decision_date x horizon x fold x candidate x universe",
        "market_context_cube": "date",
    }
    files = []
    schemas = {}
    for name in cube_names:
        path = OUTPUTS / "audit_cube" / f"{name}.parquet"
        df = cached_parquet(f"audit_cube/{name}.parquet")
        files.append(
            {
                "cube": name,
                "file": f"{name}.parquet",
                "path": f"research/mahoraga14_3_extended_analysis/outputs/audit_cube/{name}.parquet",
                "rows": int(len(df)),
                "columns": int(len(df.columns)) if not df.empty else 0,
                "grain": grains[name],
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
        schemas[name] = list(df.columns)

    operation_data = registry.cube_operations()
    return {
        "problem": operation_data["problem"],
        "evidence_chain": operation_data["evidence_chain"],
        "analytical_axes": operation_data["analytical_axes"],
        "operations": operation_data["operations"],
        "files": files,
        "schemas": schemas,
        "logical_dimensions": ["date", "fold", "candidate", "universe", "ticker", "module", "horizon", "market regime"],
        "relationships": [
            "decision_date_cube joins to position_cube on date, fold, candidate_id and universe_id.",
            "decision_date_cube joins to module_trace_cube on date, fold, candidate_id and universe_id.",
            "decision_date_cube joins to outcome_cube on date = decision_date, fold, candidate_id and universe_id.",
            "market_context_cube joins to decision_date_cube on date.",
        ],
        "sample_queries": [
            'decision[decision["hard_backoff_flag"] == True]',
            'positions[(positions["selected_flag"] == True) & (positions["final_weight"] > 0)]',
            'decision.merge(outcome, left_on=["date", "fold", "candidate_id", "universe_id"], right_on=["decision_date", "fold", "candidate_id", "universe_id"], how="left")',
        ],
    }
