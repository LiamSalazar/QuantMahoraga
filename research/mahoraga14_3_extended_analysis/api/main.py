from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


PHASE_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PHASE_ROOT / "outputs"

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
    return {"count": int(len(df)), "rows": out.where(pd.notna(out), None).to_dict("records")}


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


@app.get("/universes/summary")
def universes_summary(limit: int = Query(500, ge=1, le=5000)) -> Dict[str, Any]:
    df = cached_csv("universe_robustness/universe_robustness_summary.csv")
    return records(df, limit)
