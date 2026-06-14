from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


DSS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_OUTPUTS = REPO_ROOT / "baseline" / "mahoraga14_3_baseline" / "outputs"
BASELINE_AUDIT = REPO_ROOT / "baseline" / "mahoraga14_3_baseline" / "audit"
EXT_OUTPUTS = REPO_ROOT / "research" / "mahoraga14_3_extended_analysis" / "outputs"
OFFICIAL_CANDIDATE_ID = "B1.05_C1.10_L1.10_R1.05"
OFFICIAL_UNIVERSE_ID = "base_universe_12"


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): _clean(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _coerce(value: str) -> Any:
    text = value.strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(mark in text for mark in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def read_csv_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{key: _coerce(value) for key, value in row.items()} for row in reader]
    return rows[:limit] if limit else rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def baseline_evidence() -> dict[str, Any]:
    return {
        "official_candidate_id": OFFICIAL_CANDIDATE_ID,
        "official_universe_id": OFFICIAL_UNIVERSE_ID,
        "stitched_comparison": read_csv_rows(BASELINE_OUTPUTS / "stitched_comparison_official.csv"),
        "fold_summary": read_csv_rows(BASELINE_OUTPUTS / "fold_summary_official.csv"),
        "alpha_newey_west": read_csv_rows(BASELINE_OUTPUTS / "alpha_nw_official.csv"),
        "pvalue_qvalue": read_csv_rows(BASELINE_OUTPUTS / "pvalue_qvalue_official.csv"),
        "cost_sensitivity": read_csv_rows(BASELINE_OUTPUTS / "cost_sensitivity_official.csv"),
        "slippage_sensitivity": read_csv_rows(BASELINE_OUTPUTS / "slippage_sensitivity_official.csv"),
        "exposure_summary": read_csv_rows(BASELINE_OUTPUTS / "exposure_summary_official.csv"),
        "turnover_summary": read_csv_rows(BASELINE_OUTPUTS / "turnover_summary_official.csv"),
        "return_per_exposure": read_csv_rows(BASELINE_OUTPUTS / "return_per_exposure_official.csv"),
        "priority_windows": read_csv_rows(BASELINE_OUTPUTS / "priority_window_acceptance_official.csv"),
        "audit": {
            "acceptance_robustness": read_csv_rows(BASELINE_AUDIT / "acceptance_robustness_summary_official.csv"),
            "bootstrap": read_csv_rows(BASELINE_AUDIT / "bootstrap_summary_official.csv"),
            "continuation": read_csv_rows(BASELINE_AUDIT / "continuation_diagnostic_official.csv"),
            "leader_miss": read_csv_rows(BASELINE_AUDIT / "leader_miss_analysis_official.csv"),
            "cash_drag": read_csv_rows(BASELINE_AUDIT / "allocator_cash_drag_official.csv"),
        },
        "sources": [
            "baseline/mahoraga14_3_baseline/outputs",
            "baseline/mahoraga14_3_baseline/audit",
        ],
    }


def extended_summary() -> dict[str, Any]:
    robustness_root = EXT_OUTPUTS / "extended_multiplier_robustness"
    universe_root = EXT_OUTPUTS / "universe_robustness"
    audit_root = EXT_OUTPUTS / "audit_cube"
    summary = read_csv_rows(robustness_root / "extended_multiplier_summary.csv")
    return {
        "run": read_json(audit_root / "representative_candidates.json"),
        "extended_multiplier_summary": summary,
        "one_dimensional_sweeps": read_csv_rows(robustness_root / "one_dimensional_sweeps.csv"),
        "two_dimensional_sweeps": read_csv_rows(robustness_root / "two_dimensional_sweeps.csv"),
        "extreme_cases": read_csv_rows(robustness_root / "extreme_cases.csv"),
        "fold_summary": read_csv_rows(robustness_root / "extended_multiplier_fold_summary.csv"),
        "sensitivity_ranking": read_csv_rows(robustness_root / "sensitivity_ranking.csv"),
        "plateau_radius": read_csv_rows(robustness_root / "plateau_radius_by_axis.csv"),
        "universe_robustness": read_csv_rows(universe_root / "universe_robustness_summary.csv"),
        "universe_coverage": read_csv_rows(universe_root / "universe_coverage_audit.csv", 500),
        "audit_cube": {
            "backoff": read_csv_rows(audit_root / "backoff_audit.csv"),
            "continuation": read_csv_rows(audit_root / "continuation_activation_audit.csv"),
            "leader": read_csv_rows(audit_root / "leader_participation_audit.csv"),
            "structural_defense": read_csv_rows(audit_root / "structural_defense_audit.csv"),
            "top_decision_drivers": read_csv_rows(audit_root / "top_decision_drivers.csv"),
            "stop_loss": read_csv_rows(audit_root / "stop_loss_audit.csv", 500),
        },
        "sources": [
            "research/mahoraga14_3_extended_analysis/outputs/extended_multiplier_robustness",
            "research/mahoraga14_3_extended_analysis/outputs/universe_robustness",
            "research/mahoraga14_3_extended_analysis/outputs/audit_cube",
        ],
    }


def best_official_worst_from_extended() -> dict[str, Any]:
    rows = read_csv_rows(EXT_OUTPUTS / "extended_multiplier_robustness" / "extended_multiplier_summary.csv")
    if not rows:
        return {"best": None, "official": None, "worst": None, "rows": []}
    official = next((row for row in rows if row.get("candidate_id") == OFFICIAL_CANDIDATE_ID or row.get("CandidateId") == OFFICIAL_CANDIDATE_ID), None)
    scored = [row for row in rows if isinstance(row.get("Sharpe"), (int, float))]
    best = max(scored, key=lambda row: float(row.get("Sharpe") or -999999), default=None)
    worst = min(scored, key=lambda row: float(row.get("Sharpe") or 999999), default=None)
    for label, row in [("Best observed in sweep", best), ("Official baseline", official), ("Worst observed in sweep", worst)]:
        if row is not None:
            row["research_role"] = label
            row["row_origin"] = "observed/audited scenario"
            row["demo_mode"] = False
    return {"best": _clean(best), "official": _clean(official), "worst": _clean(worst), "rows": _clean([row for row in [best, official, worst] if row])}


def pipeline_summary() -> dict[str, Any]:
    return read_json(DSS_ROOT / "outputs" / "reports" / "pipeline_summary.json")
