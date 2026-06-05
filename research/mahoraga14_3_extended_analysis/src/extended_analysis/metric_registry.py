from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


OFFICIAL_CANDIDATE_ID = "B1.05_C1.10_L1.10_R1.05"
OFFICIAL_VARIANT = "MAHORAGA14_3_BASELINE_OFFICIAL"
OFFICIAL_UNIVERSE_ID = "base_universe_12"
NOT_AVAILABLE = "Not available in current outputs"

PHASE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
BASELINE_ROOT = REPO_ROOT / "baseline" / "mahoraga14_3_baseline"
BASELINE_OUTPUTS = BASELINE_ROOT / "outputs"
BASELINE_AUDIT = BASELINE_ROOT / "audit"
BASELINE_CONFIG = BASELINE_ROOT / "config"
BASELINE_SRC = BASELINE_ROOT / "src" / "mahoraga14_3_baseline"
OUTPUTS = PHASE_ROOT / "outputs"
EXT_MULT = OUTPUTS / "extended_multiplier_robustness"
UNIVERSE = OUTPUTS / "universe_robustness"
AUDIT_CUBE = OUTPUTS / "audit_cube"
REPORTS = OUTPUTS / "reports"

CATEGORIES = [
    "Performance",
    "Risk",
    "Benchmark and Statistical Evidence",
    "Portfolio and Execution Diagnostics",
    "ML / Signal Diagnostics",
    "Robustness",
    "Fold Validation",
]

DISPLAY_HINTS = {
    "percent_points",
    "ratio_percent",
    "decimal",
    "decimal4",
    "integer",
    "text",
}


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


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
    return {str(key): clean_value(value) for key, value in row.items()}


def records(df: pd.DataFrame, limit: int = 500) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.head(max(1, min(limit, 5000))).copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return [clean_record(row) for row in out.to_dict("records")]


def first_record(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    rows = records(df, 1)
    return rows[0] if rows else None


@lru_cache(maxsize=128)
def read_csv(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@lru_cache(maxsize=32)
def read_parquet(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def csv(path: Path) -> pd.DataFrame:
    return read_csv(str(path))


def parquet(path: Path) -> pd.DataFrame:
    return read_parquet(str(path))


def numeric(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def row_value(row: Optional[Dict[str, Any]], key: str) -> Any:
    if not row:
        return None
    return row.get(key)


def official_row(df: pd.DataFrame, variant_col: str = "Variant") -> Optional[Dict[str, Any]]:
    if df.empty:
        return None
    if "CandidateId" in df.columns:
        candidate = df[df["CandidateId"].astype(str) == OFFICIAL_CANDIDATE_ID]
        if not candidate.empty:
            return first_record(candidate)
    if "candidate_id" in df.columns:
        candidate = df[df["candidate_id"].astype(str) == OFFICIAL_CANDIDATE_ID]
        if not candidate.empty:
            return first_record(candidate)
    if variant_col in df.columns:
        variant = df[df[variant_col].astype(str) == OFFICIAL_VARIANT]
        if not variant.empty:
            return first_record(variant)
    return first_record(df)


def display_value(value: Any, hint: str = "decimal") -> str:
    if value is None:
        return NOT_AVAILABLE
    if hint == "text":
        return str(value)
    n = numeric(value)
    if n is None:
        return str(value)
    if hint == "percent_points":
        return f"{n:.2f}%"
    if hint == "ratio_percent":
        return f"{n * 100:.2f}%"
    if hint == "integer":
        return f"{int(round(n))}"
    if hint == "decimal4":
        return f"{n:.4f}"
    return f"{n:.3f}"


def metric(
    metric_name: str,
    value: Any,
    category: str,
    source_file: Path,
    interpretation: str,
    *,
    source_section: Optional[str] = None,
    limitation: Optional[str] = None,
    display_hint: str = "decimal",
) -> Dict[str, Any]:
    if display_hint not in DISPLAY_HINTS:
        display_hint = "decimal"
    return {
        "metric_name": metric_name,
        "value": clean_value(value),
        "display_value": display_value(value, display_hint),
        "category": category,
        "source_file": rel_path(source_file),
        "source_section": source_section,
        "interpretation": interpretation,
        "limitation": limitation,
    }


def metric_not_available(metric_name: str, category: str, reason: str) -> Dict[str, Any]:
    return {
        "metric_name": metric_name,
        "value": None,
        "display_value": NOT_AVAILABLE,
        "category": category,
        "source_file": None,
        "source_section": None,
        "interpretation": NOT_AVAILABLE,
        "limitation": reason,
    }


def normalize_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def active_return_frame() -> pd.DataFrame:
    return csv(BASELINE_OUTPUTS / "active_return_vs_qqq_official.csv")


def official_daily_stats() -> Dict[str, Any]:
    df = active_return_frame()
    if df.empty or "OfficialReturn" not in df.columns:
        return {}
    returns = pd.to_numeric(df["OfficialReturn"], errors="coerce").dropna()
    if returns.empty:
        return {}
    downside = returns[returns < 0]
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    in_dd = drawdown < 0
    max_duration = 0
    current = 0
    for flag in in_dd.tolist():
        current = current + 1 if flag else 0
        max_duration = max(max_duration, current)

    stats = {
        "annualized_volatility": float(returns.std() * math.sqrt(252)),
        "downside_volatility": float(downside.std() * math.sqrt(252)) if not downside.empty else None,
        "drawdown_duration": max_duration,
        "benchmark_correlation_qqq": float(returns.corr(pd.to_numeric(df["QQQReturn"], errors="coerce"))) if "QQQReturn" in df.columns else None,
        "benchmark_correlation_spy": float(returns.corr(pd.to_numeric(df["SPYReturn"], errors="coerce"))) if "SPYReturn" in df.columns else None,
    }
    if "SPYReturn" in df.columns:
        active_vs_spy = (1.0 + returns - pd.to_numeric(df["SPYReturn"], errors="coerce").fillna(0.0)).prod() - 1.0
        stats["active_return_vs_spy_final"] = float(active_vs_spy)
    if "CumOfficial" in df.columns:
        stats["total_return"] = float(pd.to_numeric(df["CumOfficial"], errors="coerce").dropna().iloc[-1])
        stats["equity_final"] = stats["total_return"] + 1.0
    if "CumActiveReturn_vs_QQQ" in df.columns:
        stats["active_return_vs_qqq_final"] = float(pd.to_numeric(df["CumActiveReturn_vs_QQQ"], errors="coerce").dropna().iloc[-1])
    return stats


def parse_plateau_report() -> Dict[str, Any]:
    path = EXT_MULT / "plateau_radius_report.md"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")

    def number_after(label: str) -> Optional[float]:
        match = re.search(rf"{re.escape(label)}:\s*([0-9.\-]+)%?", text, flags=re.IGNORECASE)
        if not match:
            return None
        value = float(match.group(1))
        return value / 100.0 if "%" in match.group(0) else value

    return {
        "distance_to_decay": number_after("distance_to_decay"),
        "robust_region_share_extended": number_after("robust_region_share_extended"),
        "sampled_candidates": number_after("sampled candidates"),
    }


def source_inventory() -> List[Dict[str, Any]]:
    groups = {
        "official_baseline_outputs": BASELINE_OUTPUTS,
        "official_baseline_audit": BASELINE_AUDIT,
        "official_baseline_paper_pack": BASELINE_ROOT / "paper_pack",
        "official_baseline_manifests": BASELINE_ROOT / "manifests",
        "official_baseline_docs": BASELINE_ROOT / "docs",
        "extended_multiplier_robustness": EXT_MULT,
        "universe_robustness": UNIVERSE,
        "audit_cube": AUDIT_CUBE,
        "extended_reports": REPORTS,
        "extended_manifests": OUTPUTS / "manifests",
    }
    out: List[Dict[str, Any]] = []
    for group, folder in groups.items():
        if not folder.exists():
            out.append({"group": group, "path": rel_path(folder), "available": False, "files": 0})
            continue
        files = [p for p in folder.iterdir() if p.is_file()]
        out.append(
            {
                "group": group,
                "path": rel_path(folder),
                "available": True,
                "files": len(files),
                "metric_files": [rel_path(p) for p in files if p.suffix.lower() in {".csv", ".json", ".md", ".parquet"}],
            }
        )
    return out


def build_scorecard() -> Dict[str, Any]:
    stitched_path = BASELINE_OUTPUTS / "stitched_comparison_official.csv"
    alpha_path = BASELINE_OUTPUTS / "alpha_nw_official.csv"
    pvalue_path = BASELINE_OUTPUTS / "pvalue_qvalue_official.csv"
    rpe_path = BASELINE_OUTPUTS / "return_per_exposure_official.csv"
    exposure_path = BASELINE_OUTPUTS / "exposure_summary_official.csv"
    turnover_path = BASELINE_OUTPUTS / "turnover_summary_official.csv"
    fold_path = BASELINE_OUTPUTS / "fold_summary_official.csv"
    cost_path = BASELINE_OUTPUTS / "cost_sensitivity_official.csv"
    slip_path = BASELINE_OUTPUTS / "slippage_sensitivity_official.csv"
    continuation_path = BASELINE_AUDIT / "continuation_diagnostic_official.csv"
    cash_drag_path = BASELINE_AUDIT / "allocator_cash_drag_official.csv"
    config_path = BASELINE_SRC / "mahoraga14_config.py"
    costs_path = BASELINE_SRC / "mahoraga6_1.py"
    ext_path = EXT_MULT / "extended_multiplier_summary.csv"
    sensitivity_path = EXT_MULT / "sensitivity_ranking.csv"
    plateau_path = EXT_MULT / "plateau_radius_by_axis.csv"
    universe_path = UNIVERSE / "universe_robustness_summary.csv"
    position_path = AUDIT_CUBE / "position_cube.parquet"

    stitched = csv(stitched_path)
    official = official_row(stitched)
    alpha = csv(alpha_path)
    pvals = csv(pvalue_path)
    rpe = official_row(csv(rpe_path))
    exposure = official_row(csv(exposure_path))
    turnover = official_row(csv(turnover_path))
    folds = csv(fold_path)
    official_folds = folds[folds["Variant"].astype(str) == OFFICIAL_VARIANT] if not folds.empty and "Variant" in folds.columns else pd.DataFrame()
    ext_summary = csv(ext_path)
    ext_official = official_row(ext_summary)
    sensitivity = csv(sensitivity_path)
    plateau = csv(plateau_path)
    universe = csv(universe_path)
    daily = official_daily_stats()
    plateau_report = parse_plateau_report()

    metrics: List[Dict[str, Any]] = []

    def add(name: str, value: Any, category: str, source: Path, text: str, **kwargs: Any) -> None:
        metrics.append(metric(name, value, category, source, text, **kwargs))

    add("total_return", daily.get("total_return"), "Performance", BASELINE_OUTPUTS / "active_return_vs_qqq_official.csv", "Final compounded official return over the official stitched period.", display_hint="ratio_percent")
    add("equity_final", daily.get("equity_final"), "Performance", BASELINE_OUTPUTS / "active_return_vs_qqq_official.csv", "Final equity multiple when the starting equity is 1.0.", display_hint="decimal")
    add("CAGR", row_value(official, "CAGR"), "Performance", stitched_path, "Official annualized compound growth rate.", display_hint="percent_points")
    add("active_return_vs_QQQ_final", daily.get("active_return_vs_qqq_final") or row_value(ext_official, "active_return_vs_QQQ_final"), "Performance", BASELINE_OUTPUTS / "active_return_vs_qqq_official.csv", "Final compounded active-return series versus QQQ.", display_hint="ratio_percent")
    add("active_return_vs_SPY_final", daily.get("active_return_vs_spy_final"), "Performance", BASELINE_OUTPUTS / "active_return_vs_qqq_official.csv", "Derived from official daily returns and SPY daily returns in the official active-return file.", display_hint="ratio_percent")
    add("CAGR_vs_QQQ", _reference_value(pvals, "QQQ", "CAGR_Delta"), "Performance", pvalue_path, "Official CAGR difference versus QQQ from the pairwise test table.", display_hint="percent_points")
    add("CAGR_vs_SPY", _reference_value(pvals, "SPY", "CAGR_Delta"), "Performance", pvalue_path, "Official CAGR difference versus SPY from the pairwise test table.", display_hint="percent_points")
    add("return_per_exposure", row_value(rpe, "ReturnPerExposure"), "Performance", rpe_path, "Total return normalized by average exposure.", display_hint="decimal4")

    cagr_decimal = numeric(row_value(official, "CAGR"))
    maxdd_percent = numeric(row_value(official, "MaxDD"))
    calmar = (cagr_decimal / 100.0) / abs(maxdd_percent / 100.0) if cagr_decimal is not None and maxdd_percent not in (None, 0) else None
    add("Sharpe", row_value(official, "Sharpe"), "Risk", stitched_path, "Official volatility-adjusted return.", display_hint="decimal")
    add("Sortino", row_value(official, "Sortino"), "Risk", stitched_path, "Official downside-risk-adjusted return.", display_hint="decimal")
    add("MaxDD", row_value(official, "MaxDD"), "Risk", stitched_path, "Worst official stitched drawdown.", display_hint="percent_points")
    add("annualized_volatility", daily.get("annualized_volatility"), "Risk", BASELINE_OUTPUTS / "active_return_vs_qqq_official.csv", "Derived from official daily return standard deviation.", display_hint="ratio_percent")
    add("downside_volatility", daily.get("downside_volatility"), "Risk", BASELINE_OUTPUTS / "active_return_vs_qqq_official.csv", "Derived from official negative daily returns.", display_hint="ratio_percent")
    add("Calmar_ratio", calmar, "Risk", stitched_path, "CAGR divided by absolute MaxDD.", limitation="Derived from official CAGR and MaxDD, not separately reported.", display_hint="decimal")
    add("worst_fold_MaxDD", _min_numeric(official_folds, "MaxDD"), "Risk", fold_path, "Most negative MaxDD among official folds.", display_hint="percent_points")
    add("worst_fold_Sharpe", _min_numeric(official_folds, "Sharpe"), "Risk", fold_path, "Lowest official fold Sharpe.", display_hint="decimal")
    add("severe_fold_damage_count", row_value(ext_official, "severe_fold_damage_count"), "Risk", ext_path, "Extended robustness severe fold damage count for the official reference candidate.", display_hint="integer")
    add("drawdown_duration_days", daily.get("drawdown_duration"), "Risk", BASELINE_OUTPUTS / "active_return_vs_qqq_official.csv", "Longest stitched drawdown spell derived from official daily returns.", display_hint="integer")

    for benchmark in ("QQQ", "SPY"):
        bench = alpha[alpha["Benchmark"].astype(str) == benchmark] if not alpha.empty and "Benchmark" in alpha.columns else pd.DataFrame()
        bench = bench[bench["Variant"].astype(str) == OFFICIAL_VARIANT] if not bench.empty and "Variant" in bench.columns else bench
        row = first_record(bench)
        suffix = benchmark
        add(f"AlphaNW_{suffix}", row_value(row, "alpha_ann"), "Benchmark and Statistical Evidence", alpha_path, f"Newey-West annualized alpha versus {benchmark}.", display_hint="decimal4")
        add(f"t_alpha_{suffix}", row_value(row, "t_alpha"), "Benchmark and Statistical Evidence", alpha_path, f"Newey-West t-statistic versus {benchmark}.", display_hint="decimal")
        add(f"p_alpha_{suffix}", row_value(row, "p_alpha"), "Benchmark and Statistical Evidence", alpha_path, f"Newey-West p-value versus {benchmark}.", display_hint="decimal4")
        add(f"beta_vs_{suffix}", row_value(row, "beta"), "Benchmark and Statistical Evidence", alpha_path, f"Regression beta versus {benchmark}.", display_hint="decimal")
        add(f"R2_vs_{suffix}", row_value(row, "R2"), "Benchmark and Statistical Evidence", alpha_path, f"Regression R-squared versus {benchmark}.", display_hint="decimal")
        add(f"p_value_vs_{suffix}", _reference_value(pvals, benchmark, "p_value"), "Benchmark and Statistical Evidence", pvalue_path, f"Pairwise p-value for official versus {benchmark}.", display_hint="decimal4")
        add(f"q_value_vs_{suffix}", _reference_value(pvals, benchmark, "q_value"), "Benchmark and Statistical Evidence", pvalue_path, f"Adjusted q-value for official versus {benchmark}.", display_hint="decimal4")
    add("upside_capture_QQQ", row_value(official, "UpsideCaptureQQQ"), "Benchmark and Statistical Evidence", stitched_path, "Official upside capture relative to QQQ up periods.", display_hint="ratio_percent")
    add("downside_capture_QQQ", row_value(official, "DownsideCaptureQQQ"), "Benchmark and Statistical Evidence", stitched_path, "Official downside capture relative to QQQ down periods.", display_hint="ratio_percent")
    add("benchmark_correlation_QQQ", daily.get("benchmark_correlation_qqq"), "Benchmark and Statistical Evidence", BASELINE_OUTPUTS / "active_return_vs_qqq_official.csv", "Correlation between official daily returns and QQQ daily returns.", display_hint="decimal")
    add("benchmark_correlation_SPY", daily.get("benchmark_correlation_spy"), "Benchmark and Statistical Evidence", BASELINE_OUTPUTS / "active_return_vs_qqq_official.csv", "Correlation between official daily returns and SPY daily returns.", display_hint="decimal")

    add("average_exposure", row_value(official, "AvgExposure") or row_value(exposure, "mean"), "Portfolio and Execution Diagnostics", stitched_path, "Average official invested exposure.", display_hint="ratio_percent")
    add("average_turnover", row_value(official, "AvgTurnover") or row_value(turnover, "mean"), "Portfolio and Execution Diagnostics", stitched_path, "Average official turnover per decision row.", display_hint="ratio_percent")
    avg_turnover = numeric(row_value(official, "AvgTurnover") or row_value(turnover, "mean"))
    add("annualized_turnover_estimate", avg_turnover * 252 if avg_turnover is not None else None, "Portfolio and Execution Diagnostics", turnover_path, "Daily average turnover annualized by 252 trading days.", limitation="Derived estimate from reported average turnover.", display_hint="decimal")
    config_text = config_path.read_text(encoding="utf-8", errors="replace") if config_path.exists() else ""
    costs_text = costs_path.read_text(encoding="utf-8", errors="replace") if costs_path.exists() else ""
    add("rebalance_frequency", _regex_value(config_text, r"decision_freq:\s*str\s*=\s*\"([^\"]+)\""), "Portfolio and Execution Diagnostics", config_path, "Official decision frequency from the frozen configuration.", display_hint="text")
    add("average_selected_positions", _average_selected_positions(), "Portfolio and Execution Diagnostics", position_path, "Average selected tickers per date for the official candidate in the audit cube.", display_hint="decimal")
    add("commission_assumption", _regex_float(costs_text, r"commission:\s*float\s*=\s*([0-9.]+)"), "Portfolio and Execution Diagnostics", costs_path, "Default commission assumption in CostsConfig.", display_hint="ratio_percent")
    add("slippage_assumption", _regex_float(costs_text, r"slippage:\s*float\s*=\s*([0-9.]+)"), "Portfolio and Execution Diagnostics", costs_path, "Default slippage assumption in CostsConfig.", display_hint="ratio_percent")
    add("cost_stress_max_CAGR_delta", _min_numeric(csv(cost_path), "DeltaCAGR_vs_Base"), "Portfolio and Execution Diagnostics", cost_path, "Largest reported CAGR degradation across official cost stress rows.", display_hint="percent_points")
    add("slippage_plus_5bps_CAGR_delta", _scenario_value(csv(slip_path), "SLIPPAGE_PLUS_5BPS", "DeltaCAGR_vs_Base"), "Portfolio and Execution Diagnostics", slip_path, "CAGR change under the official slippage stress scenario.", display_hint="percent_points")
    cash = csv(cash_drag_path)
    add("cash_drag_before_mean", _mean_numeric(cash, "cash_drag_before"), "Portfolio and Execution Diagnostics", cash_drag_path, "Mean pre-redeployment cash drag from official allocator audit.", display_hint="ratio_percent")
    add("cash_drag_after_mean", _mean_numeric(cash, "cash_drag_after"), "Portfolio and Execution Diagnostics", cash_drag_path, "Mean post-redeployment cash drag from official allocator audit.", display_hint="ratio_percent")

    cont = csv(continuation_path)
    stitched_cont = cont[cont["Segment"].astype(str) == "STITCHED"] if not cont.empty and "Segment" in cont.columns else pd.DataFrame()
    cont_row = first_record(stitched_cont)
    add("continuation_activation_rate", row_value(cont_row, "ActivationRate"), "ML / Signal Diagnostics", continuation_path, "Stitched continuation activation rate from the official diagnostic.", display_hint="ratio_percent")
    add("continuation_hit_rate_1W", row_value(cont_row, "HitRate1W"), "ML / Signal Diagnostics", continuation_path, "Official continuation one-week hit rate when activated.", display_hint="ratio_percent")
    add("continuation_hit_rate_4W", row_value(cont_row, "HitRate4W"), "ML / Signal Diagnostics", continuation_path, "Official continuation four-week hit rate when activated.", display_hint="ratio_percent")
    add("continuation_edge_vs_no_activation_4W", row_value(cont_row, "EdgeVsNoActivation4W"), "ML / Signal Diagnostics", continuation_path, "Four-week edge of continuation activations versus no-activation cases.", display_hint="ratio_percent")
    for audit_name, col, label in [
        ("continuation_activation_audit.csv", "continuation_help_rate", "continuation_helped_rate"),
        ("leader_participation_audit.csv", "leader_help_rate", "leader_helped_rate"),
        ("backoff_audit.csv", "backoff_help_rate", "backoff_helped_rate"),
    ]:
        audit_path = AUDIT_CUBE / audit_name
        audit = csv(audit_path)
        audit = audit[audit["candidate_id"].astype(str) == OFFICIAL_CANDIDATE_ID] if not audit.empty and "candidate_id" in audit.columns else pd.DataFrame()
        for horizon in (1, 5, 20):
            value = _horizon_value(audit, horizon, col)
            add(f"{label}_{horizon}d", value, "ML / Signal Diagnostics", audit_path, f"{label.replace('_', ' ').title()} at {horizon} trading days.", display_hint="ratio_percent")

    robust_share = plateau_report.get("robust_region_share_extended")
    if robust_share is None and not ext_summary.empty and "robust_region_flag" in ext_summary.columns:
        robust_share = float(pd.to_numeric(ext_summary["robust_region_flag"], errors="coerce").mean())
    sensitivity_row = first_record(sensitivity.sort_values("sensitivity_score", ascending=False)) if not sensitivity.empty and "sensitivity_score" in sensitivity.columns else first_record(sensitivity)
    add("robust_region_share_extended", robust_share, "Robustness", ext_path, "Fraction of sampled extended multiplier candidates that remained in the robust region.", display_hint="ratio_percent")
    add("distance_to_decay", plateau_report.get("distance_to_decay"), "Robustness", EXT_MULT / "plateau_radius_report.md", "Minimum documented perturbation distance to robustness decay.", display_hint="decimal4")
    add("sampled_candidates", len(ext_summary) if not ext_summary.empty else plateau_report.get("sampled_candidates"), "Robustness", ext_path, "Number of sampled multiplier candidates in the extended robustness table.", display_hint="integer")
    add("most_sensitive_axis", row_value(sensitivity_row, "axis"), "Robustness", sensitivity_path, "Highest sensitivity axis in the extended ranking.", display_hint="text")
    add("most_sensitive_axis_score", row_value(sensitivity_row, "sensitivity_score"), "Robustness", sensitivity_path, "Sensitivity score of the highest-ranked axis.", display_hint="decimal")
    widest = first_record(plateau.sort_values("plateau_radius_relative", ascending=False)) if not plateau.empty and "plateau_radius_relative" in plateau.columns else None
    add("widest_plateau_axis", row_value(widest, "axis"), "Robustness", plateau_path, "Axis with the widest sampled robust plateau.", display_hint="text")
    add("widest_plateau_radius", row_value(widest, "plateau_radius_relative"), "Robustness", plateau_path, "Relative plateau radius of the widest sampled robust axis.", display_hint="ratio_percent")
    official_universe = universe[universe["candidate_id"].astype(str) == OFFICIAL_CANDIDATE_ID] if not universe.empty and "candidate_id" in universe.columns else pd.DataFrame()
    ok_universes = official_universe[official_universe["run_status"].astype(str).str.upper() == "OK"] if not official_universe.empty and "run_status" in official_universe.columns else official_universe
    add("universe_robustness_completed_runs", len(ok_universes), "Robustness", universe_path, "Completed universe robustness runs for the official candidate.", display_hint="integer")
    add("negative_control_status", _negative_control_status(universe), "Robustness", universe_path, "Recorded status for the non-technology negative control universe.", display_hint="text")

    add("official_fold_count", len(official_folds), "Fold Validation", fold_path, "Number of official walk-forward folds with official metrics.", display_hint="integer")
    add("weakest_fold_by_Sharpe", _fold_id_for_min(official_folds, "Sharpe"), "Fold Validation", fold_path, "Fold with the lowest official Sharpe.", display_hint="text")
    add("weakest_fold_by_CAGR", _fold_id_for_min(official_folds, "CAGR"), "Fold Validation", fold_path, "Fold with the lowest official CAGR.", display_hint="text")
    add("worst_fold_alpha_QQQ", _min_numeric(official_folds, "AlphaNW_QQQ"), "Fold Validation", fold_path, "Lowest fold-level Newey-West alpha versus QQQ.", display_hint="decimal4")
    add("worst_fold_exposure", _min_numeric(official_folds, "Exposure"), "Fold Validation", fold_path, "Lowest fold-level average exposure.", display_hint="ratio_percent")

    unavailable = [
        metric_not_available("accuracy", "ML / Signal Diagnostics", "No accuracy output file is present in the official baseline or extended outputs."),
        metric_not_available("precision", "ML / Signal Diagnostics", "No precision output file is present in the current artifacts."),
        metric_not_available("recall", "ML / Signal Diagnostics", "No recall output file is present in the current artifacts."),
        metric_not_available("F1", "ML / Signal Diagnostics", "No F1 output file is present in the current artifacts."),
        metric_not_available("AUC", "ML / Signal Diagnostics", "Source code can compute AUC, but no materialized official AUC diagnostic file is present."),
        metric_not_available("Brier_score", "ML / Signal Diagnostics", "No Brier score output file is present in the current artifacts."),
        metric_not_available("confusion_matrix", "ML / Signal Diagnostics", "No confusion-matrix output file is present in the current artifacts."),
        metric_not_available("calibration_metrics", "ML / Signal Diagnostics", "No materialized calibration report is present in the current official outputs."),
        metric_not_available("concentration_metrics", "Portfolio and Execution Diagnostics", "No explicit concentration metric output is present; position weights are available for drill-down."),
    ]

    all_metrics = metrics + unavailable
    grouped = {
        category: [m for m in all_metrics if m["category"] == category]
        for category in CATEGORIES
    }
    return {
        "identity": official_identity(),
        "nomenclature": candidate_nomenclature(),
        "categories": grouped,
        "metrics": all_metrics,
        "unavailable_metrics": unavailable,
        "sources_discovered": source_inventory(),
        "summary": {
            "official_candidate_id": OFFICIAL_CANDIDATE_ID,
            "official_universe_id": OFFICIAL_UNIVERSE_ID,
            "metric_count": len(metrics),
            "unavailable_count": len(unavailable),
            "source_count": sum(1 for source in source_inventory() if source["available"]),
        },
    }


def _reference_value(df: pd.DataFrame, reference: str, column: str) -> Any:
    if df.empty or "Reference" not in df.columns or column not in df.columns:
        return None
    row = df[df["Reference"].astype(str) == reference]
    return row[column].iloc[0] if not row.empty else None


def _scenario_value(df: pd.DataFrame, scenario: str, column: str) -> Any:
    if df.empty or "Scenario" not in df.columns or column not in df.columns:
        return None
    row = df[df["Scenario"].astype(str) == scenario]
    return row[column].iloc[0] if not row.empty else None


def _horizon_value(df: pd.DataFrame, horizon: int, column: str) -> Any:
    if df.empty or "horizon" not in df.columns or column not in df.columns:
        return None
    row = df[pd.to_numeric(df["horizon"], errors="coerce") == horizon]
    return row[column].iloc[0] if not row.empty else None


def _min_numeric(df: pd.DataFrame, column: str) -> Any:
    if df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.min()) if not values.empty else None


def _mean_numeric(df: pd.DataFrame, column: str) -> Any:
    if df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _regex_value(text: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _regex_float(text: str, pattern: str) -> Optional[float]:
    value = _regex_value(text, pattern)
    return float(value) if value is not None else None


def _average_selected_positions() -> Optional[float]:
    df = parquet(AUDIT_CUBE / "position_cube.parquet")
    if df.empty or "selected_flag" not in df.columns:
        return None
    if "candidate_id" in df.columns:
        df = df[df["candidate_id"].astype(str) == OFFICIAL_CANDIDATE_ID]
    if "universe_id" in df.columns:
        df = df[df["universe_id"].astype(str) == OFFICIAL_UNIVERSE_ID]
    if df.empty:
        return None
    grouped = df.groupby(["date", "fold"], dropna=False)["selected_flag"].sum()
    return float(grouped.mean()) if not grouped.empty else None


def _negative_control_status(df: pd.DataFrame) -> Optional[str]:
    if df.empty or "universe_id" not in df.columns:
        return None
    row = df[df["universe_id"].astype(str) == "negative_control_nontech"]
    if row.empty:
        return None
    if "run_status" in row.columns:
        return str(row["run_status"].iloc[0])
    return "Coverage available; metrics not available"


def _fold_id_for_min(df: pd.DataFrame, column: str) -> Optional[str]:
    if df.empty or column not in df.columns or "Fold" not in df.columns:
        return None
    work = df.copy()
    work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=[column])
    if work.empty:
        return None
    row = work.sort_values(column).iloc[0]
    return f"Fold {int(row['Fold'])}"


def official_identity() -> Dict[str, Any]:
    return {
        "badge": "OFFICIAL BASELINE",
        "short_label": "Official baseline",
        "technical_id": OFFICIAL_CANDIDATE_ID,
        "candidate_role": "Frozen baseline reference",
        "universe": OFFICIAL_UNIVERSE_ID,
        "source": "official baseline + extended analysis",
        "variant_label": OFFICIAL_VARIANT,
    }


def candidate_nomenclature() -> List[Dict[str, str]]:
    return [
        {"code": "B", "parameter": "budget multiplier", "official_value": "1.05", "meaning": "Budget controls how much long exposure the system can express."},
        {"code": "C", "parameter": "conviction multiplier", "official_value": "1.10", "meaning": "Conviction controls signal expression strength."},
        {"code": "L", "parameter": "leader participation multiplier", "official_value": "1.10", "meaning": "Leader controls conditional participation in market leaders."},
        {"code": "R", "parameter": "risk backoff strength", "official_value": "1.05", "meaning": "R controls risk backoff strength."},
    ]


def candidate_metadata() -> Dict[str, Any]:
    summary = csv(EXT_MULT / "extended_multiplier_summary.csv")
    representatives = _representative_candidates()
    if summary.empty:
        return {"official": official_identity(), "candidates": [], "families": _candidate_families(), "representative_candidates": representatives}
    rows = []
    seen: set[str] = set()
    for row in records(summary.sort_values(["sweep_role", "CandidateId"]), 500):
        candidate_id = str(row.get("candidate_id") or row.get("CandidateId"))
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        rows.append(_candidate_row(row, representative=candidate_id in representatives))
    return {
        "official": official_identity(),
        "nomenclature": candidate_nomenclature(),
        "families": _candidate_families(),
        "representative_candidates": representatives,
        "candidates": rows,
    }


def _representative_candidates() -> List[str]:
    path = AUDIT_CUBE / "representative_candidates.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in data] if isinstance(data, list) else []


def _candidate_row(row: Dict[str, Any], representative: bool) -> Dict[str, Any]:
    candidate_id = str(row.get("candidate_id") or row.get("CandidateId"))
    sweep_role = str(row.get("sweep_role") or "")
    robust = bool(numeric(row.get("robust_region_flag")) == 1)
    changed_axes = _changed_axes(row)
    label, role, interpretation = _candidate_label_role(candidate_id, sweep_role, row, robust)
    return {
        "label": label,
        "technical_id": candidate_id,
        "role": role,
        "changed_axes": changed_axes,
        "values_changed": {axis: row.get(axis) for axis in changed_axes},
        "robust_flag": robust,
        "representative_cube_candidate": representative,
        "sweep_role": sweep_role,
        "interpretation": interpretation,
        "CAGR": row.get("CAGR"),
        "Sharpe": row.get("Sharpe"),
        "MaxDD": row.get("MaxDD"),
        "severe_fold_damage_count": row.get("severe_fold_damage_count"),
    }


def _changed_axes(row: Dict[str, Any]) -> List[str]:
    official = {
        "budget_multiplier": 1.05,
        "conviction_multiplier": 1.10,
        "leader_multiplier": 1.10,
        "backoff_strength": 1.05,
    }
    axes = []
    for axis, official_value in official.items():
        value = numeric(row.get(axis))
        if value is not None and abs(value - official_value) > 1e-9:
            axes.append(axis)
    return axes


def _candidate_label_role(candidate_id: str, sweep_role: str, row: Dict[str, Any], robust: bool) -> tuple[str, str, str]:
    if candidate_id == OFFICIAL_CANDIDATE_ID:
        return "Official baseline", "Official baseline", "This is the promoted long-only baseline used as the frozen reference point."
    if candidate_id == "EXTREME_pro-risk":
        return "Pro-risk extreme", "Pro-risk extreme", "Tests a higher-budget, higher-conviction, higher-leader and lower-backoff configuration."
    if candidate_id == "EXTREME_pro-defense":
        return "Pro-defense extreme", "Pro-defense extreme", "Tests a lower-budget, lower-conviction, lower-leader and stronger-backoff configuration."
    if candidate_id == "EXTREME_all-high":
        return "All-high extreme", "All-high", "Tests all sampled multipliers above the official values."
    if candidate_id == "EXTREME_all-low":
        return "All-low extreme", "All-low", "Tests all sampled multipliers below the official values."
    axes = _changed_axes(row)
    if sweep_role.startswith("TWO_DIM"):
        label = "Two-axis stress"
        return label, "Two-dimensional sensitive-axis candidate", "Tests joint movement across the two sampled sensitive axes."
    if "budget_multiplier" in sweep_role:
        value = numeric(row.get("budget_multiplier"))
        label = "Low-budget stress" if value is not None and value < 1.05 else "High-budget stress"
        return label, "One-dimensional budget sweep", "Tests whether the model remains robust when long-budget participation changes."
    if "conviction_multiplier" in sweep_role:
        return "Conviction stress", "One-dimensional conviction sweep", "Tests whether signal expression strength changes model behavior."
    if "leader_multiplier" in sweep_role:
        return "Leader stress", "One-dimensional leader sweep", "Tests whether conditional leader participation changes model behavior."
    if "backoff_strength" in sweep_role:
        return "Backoff stress", "One-dimensional backoff sweep", "Tests whether defensive backoff strength changes model behavior."
    if not robust:
        return "Non-robust candidate", "Non-robust candidate", "This sampled candidate failed the recorded robustness flag."
    if axes:
        return "Representative candidate", "Representative cube candidate", "This candidate is included in the granular audit cube subset."
    return "Controlled candidate", "Controlled extreme", "This candidate is part of the controlled robustness sample."


def _candidate_families() -> List[Dict[str, str]]:
    return [
        {"family": "Official baseline", "meaning": "Frozen reference candidate for the DSS."},
        {"family": "One-dimensional budget sweep", "meaning": "Only budget changes while conviction, leader and backoff stay fixed."},
        {"family": "One-dimensional conviction sweep", "meaning": "Only conviction changes while other multipliers stay fixed."},
        {"family": "One-dimensional leader sweep", "meaning": "Only leader participation changes while other multipliers stay fixed."},
        {"family": "One-dimensional backoff sweep", "meaning": "Only risk backoff strength changes while other multipliers stay fixed."},
        {"family": "Two-dimensional sensitive-axis candidate", "meaning": "Two sensitive axes move together in the sampled grid."},
        {"family": "Controlled extreme", "meaning": "A deliberate stress case, not a proposed baseline."},
        {"family": "Pro-risk extreme", "meaning": "Higher participation and signal expression with lower defense."},
        {"family": "Pro-defense extreme", "meaning": "Lower participation and stronger defense."},
        {"family": "All-high", "meaning": "All sampled multipliers are above the official value."},
        {"family": "All-low", "meaning": "All sampled multipliers are below the official value."},
        {"family": "Non-robust candidate", "meaning": "Candidate failed the robust-region flag in current outputs."},
        {"family": "Representative cube candidate", "meaning": "Candidate has granular decision/position/module/outcome traces."},
    ]


def fold_summaries() -> Dict[str, Any]:
    fold_path = BASELINE_OUTPUTS / "fold_summary_official.csv"
    folds = csv(fold_path)
    if folds.empty:
        return {"folds": [], "sources": [rel_path(fold_path)], "interpretation": [NOT_AVAILABLE]}
    official = folds[folds["Variant"].astype(str) == OFFICIAL_VARIANT].copy()
    qqq = folds[folds["Variant"].astype(str) == "QQQ"].copy()
    spy = folds[folds["Variant"].astype(str) == "SPY"].copy()
    control = folds[folds["Variant"].astype(str).str.contains("CONTROL", na=False)].copy()
    outcome = parquet(AUDIT_CUBE / "outcome_cube.parquet")
    outcome_agg = pd.DataFrame()
    if not outcome.empty:
        out = outcome[outcome["candidate_id"].astype(str) == OFFICIAL_CANDIDATE_ID] if "candidate_id" in outcome.columns else outcome
        if {"fold", "horizon"}.issubset(out.columns):
            outcome_agg = out.groupby(["fold", "horizon"]).agg(
                helped_rate_vs_qqq=("decision_helped_flag_vs_qqq", "mean"),
                alpha_vs_qqq=("realized_alpha_vs_qqq", "mean"),
                exposure=("realized_exposure", "mean"),
            ).reset_index()
    rows = []
    for row in records(official.sort_values("Fold"), 20):
        fold = int(row["Fold"])
        qqq_row = first_record(qqq[qqq["Fold"] == fold])
        spy_row = first_record(spy[spy["Fold"] == fold])
        control_row = first_record(control[control["Fold"] == fold])
        weak_spots = []
        if numeric(row.get("CAGR")) is not None and qqq_row and numeric(qqq_row.get("CAGR")) is not None and numeric(row.get("CAGR")) < numeric(qqq_row.get("CAGR")):
            weak_spots.append("CAGR below QQQ")
        if spy_row and numeric(row.get("CAGR")) is not None and numeric(spy_row.get("CAGR")) is not None and numeric(row.get("CAGR")) < numeric(spy_row.get("CAGR")):
            weak_spots.append("CAGR below SPY")
        if control_row and numeric(row.get("CAGR")) is not None and numeric(control_row.get("CAGR")) is not None and numeric(row.get("CAGR")) < numeric(control_row.get("CAGR")):
            weak_spots.append("CAGR below control")
        horizon_rows = outcome_agg[outcome_agg["fold"] == fold] if not outcome_agg.empty else pd.DataFrame()
        row["outcome_by_horizon"] = records(horizon_rows, 10)
        row["weak_spots"] = weak_spots or ["No benchmark/control CAGR weakness in this fold."]
        row["interpretation"] = _fold_interpretation(row, qqq_row, spy_row, control_row)
        rows.append(row)
    weakest_sharpe = min(rows, key=lambda r: numeric(r.get("Sharpe")) or 999)
    return {
        "folds": rows,
        "sources": [rel_path(fold_path), rel_path(AUDIT_CUBE / "outcome_cube.parquet")],
        "interpretation": [
            f"Weakest official Sharpe is Fold {weakest_sharpe.get('Fold')} at {display_value(weakest_sharpe.get('Sharpe'), 'decimal')}.",
            "Fold rows compare official performance to benchmark and control rows from the official fold summary.",
        ],
    }


def _fold_interpretation(row: Dict[str, Any], qqq_row: Optional[Dict[str, Any]], spy_row: Optional[Dict[str, Any]], control_row: Optional[Dict[str, Any]]) -> str:
    cagr = numeric(row.get("CAGR"))
    sharpe = numeric(row.get("Sharpe"))
    maxdd = numeric(row.get("MaxDD"))
    parts = [f"Fold {row.get('Fold')} official CAGR is {display_value(cagr, 'percent_points')} with Sharpe {display_value(sharpe, 'decimal')} and MaxDD {display_value(maxdd, 'percent_points')}."]
    if control_row and cagr is not None and numeric(control_row.get("CAGR")) is not None:
        parts.append("It beats the historical control on CAGR." if cagr >= numeric(control_row.get("CAGR")) else "It trails the historical control on CAGR.")
    if qqq_row and cagr is not None and numeric(qqq_row.get("CAGR")) is not None:
        parts.append("It beats QQQ on CAGR." if cagr >= numeric(qqq_row.get("CAGR")) else "It trails QQQ on CAGR.")
    if spy_row and cagr is not None and numeric(spy_row.get("CAGR")) is not None:
        parts.append("It beats SPY on CAGR." if cagr >= numeric(spy_row.get("CAGR")) else "It trails SPY on CAGR.")
    return " ".join(parts)


def model_diagnostics() -> Dict[str, Any]:
    scorecard = build_scorecard()
    diagnostics = scorecard["categories"].get("ML / Signal Diagnostics", [])
    unavailable = [m for m in diagnostics if m["value"] is None]
    available = [m for m in diagnostics if m["value"] is not None]
    return {
        "available": available,
        "unavailable": unavailable,
        "interpretation": [
            "In financial systems, raw classification accuracy alone can be misleading. A signal with near-50% accuracy may still add value if payoff asymmetry, exposure timing, drawdown control, or position sizing improves portfolio-level outcomes.",
            "These diagnostics should be interpreted together with alpha, drawdown, exposure, turnover, robustness and fold behavior.",
        ],
        "sources": sorted({m["source_file"] for m in available if m.get("source_file")}),
    }


def performance_risk() -> Dict[str, Any]:
    scorecard = build_scorecard()
    active = active_return_frame()
    time_series = []
    if not active.empty:
        cols = ["Date", "OfficialReturn", "CumOfficial", "QQQReturn", "SPYReturn", "CumActiveReturn_vs_QQQ"]
        available = [col for col in cols if col in active.columns]
        sample = active[available].copy()
        if len(sample) > 260:
            sample = sample.iloc[:: max(1, len(sample) // 260)]
        time_series = records(sample, 300)
    return {
        "performance": scorecard["categories"].get("Performance", []),
        "risk": scorecard["categories"].get("Risk", []),
        "time_series": time_series,
        "sources": [rel_path(BASELINE_OUTPUTS / "stitched_comparison_official.csv"), rel_path(BASELINE_OUTPUTS / "active_return_vs_qqq_official.csv")],
    }


def research_questions() -> Dict[str, Any]:
    score = build_scorecard()
    metrics_by_name = {m["metric_name"]: m for m in score["metrics"]}
    summary = csv(EXT_MULT / "extended_multiplier_summary.csv")
    sensitivity = csv(EXT_MULT / "sensitivity_ranking.csv")
    plateau = csv(EXT_MULT / "plateau_radius_by_axis.csv")
    universe = csv(UNIVERSE / "universe_robustness_summary.csv")
    fold_data = fold_summaries()
    ticker = ticker_contribution()
    module = module_effectiveness_summary()

    def m(name: str) -> Any:
        return metrics_by_name.get(name, {}).get("value")

    budget = summary[summary["sweep_role"].astype(str).str.contains("budget_multiplier", regex=False)] if not summary.empty and "sweep_role" in summary.columns else pd.DataFrame()
    low_budget = budget[pd.to_numeric(budget.get("budget_multiplier"), errors="coerce") < 1.05] if not budget.empty else pd.DataFrame()
    high_budget = budget[pd.to_numeric(budget.get("budget_multiplier"), errors="coerce") > 1.05] if not budget.empty else pd.DataFrame()
    low_damage = int(pd.to_numeric(low_budget.get("severe_fold_damage_count"), errors="coerce").fillna(0).sum()) if not low_budget.empty else 0
    high_damage = int(pd.to_numeric(high_budget.get("severe_fold_damage_count"), errors="coerce").fillna(0).sum()) if not high_budget.empty else 0
    most_sensitive = first_record(sensitivity.sort_values("sensitivity_score", ascending=False)) if not sensitivity.empty and "sensitivity_score" in sensitivity.columns else None
    widest = first_record(plateau.sort_values("plateau_radius_relative", ascending=False)) if not plateau.empty and "plateau_radius_relative" in plateau.columns else None
    damaged = summary[pd.to_numeric(summary.get("severe_fold_damage_count"), errors="coerce").fillna(0) > 0] if not summary.empty and "severe_fold_damage_count" in summary.columns else pd.DataFrame()
    official_universe = universe[universe["candidate_id"].astype(str) == OFFICIAL_CANDIDATE_ID] if not universe.empty and "candidate_id" in universe.columns else pd.DataFrame()

    questions = [
        _question(
            "spike",
            "Is the official candidate a narrow parameter spike?",
            [rel_path(EXT_MULT / "extended_multiplier_summary.csv"), rel_path(EXT_MULT / "plateau_radius_by_axis.csv"), rel_path(EXT_MULT / "sensitivity_ranking.csv")],
            ["Read robust-region share, plateau radius and sensitivity ranking.", "Check whether degradation is global or concentrated by axis."],
            {"robust_region_share_extended": display_value(m("robust_region_share_extended"), "ratio_percent"), "distance_to_decay": display_value(m("distance_to_decay"), "decimal4"), "sampled_candidates": m("sampled_candidates"), "most_sensitive_axis": m("most_sensitive_axis")},
            "The sampled evidence does not support global parameter fragility. The main weakness is localized around budget underdeployment." if numeric(m("robust_region_share_extended")) and numeric(m("robust_region_share_extended")) >= 0.5 else "Current sampled evidence is too weak to reject parameter fragility.",
            "Moderate",
            "Applies only to sampled perturbations, not every possible multiplier value.",
        ),
        _question(
            "budget-localized",
            "Is budget sensitivity fatal or localized?",
            [rel_path(EXT_MULT / "extended_multiplier_summary.csv"), rel_path(EXT_MULT / "sensitivity_ranking.csv"), rel_path(EXT_MULT / "plateau_radius_by_axis.csv")],
            ["Compare low-budget and high-budget one-dimensional candidates.", "Review severe fold damage and robust-region flags."],
            {"low_budget_severe_damage": low_damage, "high_budget_severe_damage": high_damage, "budget_plateau_low": _axis_value(plateau, "budget_multiplier", "robust_min_sampled_value"), "budget_plateau_high": _axis_value(plateau, "budget_multiplier", "robust_max_sampled_value")},
            "Budget sensitivity is asymmetric and localized around underdeployment; sampled upward budget perturbations do not show the same fold damage." if low_damage > high_damage else "Budget sensitivity is present, but the current damage asymmetry is not strong in the sampled rows.",
            "Moderate",
            "The conclusion depends on the current one-dimensional and focused two-dimensional sweep sample.",
        ),
        _question(
            "budget-reduced",
            "What happens when budget is reduced?",
            [rel_path(EXT_MULT / "extended_multiplier_summary.csv")],
            ["Filter one-dimensional budget candidates below the official value.", "Compare CAGR, Sharpe, MaxDD and severe fold damage."],
            {"rows": records(low_budget[["CandidateId", "budget_multiplier", "CAGR", "Sharpe", "MaxDD", "severe_fold_damage_count", "robust_region_flag"]] if not low_budget.empty else low_budget, 10)},
            "Reduced-budget samples show lower performance and more severe fold damage than the official candidate.",
            "Moderate",
            "This is a sampled long-budget stress, not a complete search over all lower budgets.",
        ),
        _question(
            "budget-increased",
            "What happens when budget is increased?",
            [rel_path(EXT_MULT / "extended_multiplier_summary.csv")],
            ["Filter one-dimensional budget candidates above the official value.", "Compare aggregate metrics and fold damage."],
            {"rows": records(high_budget[["CandidateId", "budget_multiplier", "CAGR", "Sharpe", "MaxDD", "severe_fold_damage_count", "robust_region_flag"]] if not high_budget.empty else high_budget, 10)},
            "Moderate upward budget samples remain robust in the current outputs, so the documented weakness is not symmetric.",
            "Moderate",
            "Higher values outside the sampled range are not covered.",
        ),
        _question(
            "most-sensitive",
            "Which multiplier is most sensitive?",
            [rel_path(EXT_MULT / "sensitivity_ranking.csv")],
            ["Sort sensitivity ranking by sensitivity score."],
            most_sensitive or {},
            f"The most sensitive multiplier is {row_value(most_sensitive, 'axis')}." if most_sensitive else NOT_AVAILABLE,
            "Strong" if most_sensitive else "Not available",
            "Sensitivity score is defined by the current extended-analysis procedure.",
        ),
        _question(
            "widest-region",
            "Which parameters have the widest robust region?",
            [rel_path(EXT_MULT / "plateau_radius_by_axis.csv")],
            ["Sort plateau radius by relative radius."],
            {"widest_axis": row_value(widest, "axis"), "widest_radius": row_value(widest, "plateau_radius_relative"), "plateau_rows": records(plateau, 10)},
            f"The widest sampled robust region is {row_value(widest, 'axis')}." if widest else NOT_AVAILABLE,
            "Strong" if widest else "Not available",
            "Plateau radius is sampled, not continuous.",
        ),
        _question(
            "fold-damage",
            "Which candidates caused fold-level damage?",
            [rel_path(EXT_MULT / "extended_multiplier_summary.csv")],
            ["Filter candidates where severe_fold_damage_count is positive.", "Sort by severe fold damage count."],
            {"damaged_candidates": records(damaged.sort_values("severe_fold_damage_count", ascending=False) if not damaged.empty else damaged, 12)},
            "The fold-damage evidence is concentrated in low-budget and all-low/pro-defense stress candidates." if not damaged.empty else "No severe fold damage is recorded in current robustness rows.",
            "Strong" if not damaged.empty else "Limited",
            "Damage counts are thresholded summaries, not full fold narratives by themselves.",
        ),
        _module_question("continuation", "Does continuation help, and at what horizon?", module),
        _module_question("leader", "Does leader participation help, and at what horizon?", module),
        _module_question("backoff", "Does backoff help during fragile regimes?", module),
        _question(
            "tickers",
            "Which tickers contributed most?",
            [rel_path(AUDIT_CUBE / "position_cube.parquet")],
            ["Aggregate selected position rows by ticker.", "Sort total PnL contribution descending."],
            {"top_positive": ticker.get("top_positive", [])[:5], "top_negative": ticker.get("top_negative", [])[:5]},
            f"The largest positive ticker contribution is {ticker.get('top_positive', [{}])[0].get('ticker') if ticker.get('top_positive') else NOT_AVAILABLE}.",
            "Moderate",
            "Contribution is influenced by frequency, weight and realized returns, so it should be read with selected frequency and mean weight.",
        ),
        _question(
            "weak-folds",
            "Which folds were weaker?",
            [rel_path(BASELINE_OUTPUTS / "fold_summary_official.csv"), rel_path(AUDIT_CUBE / "outcome_cube.parquet")],
            ["Use official fold summary for economic metrics.", "Use outcome cube for helped-rate behavior by horizon."],
            {"folds": fold_data.get("folds", [])},
            fold_data.get("interpretation", [NOT_AVAILABLE])[0],
            "Strong",
            "Fold weakness is descriptive and does not identify one causal module unless cube joins support it.",
        ),
        _question(
            "generalization",
            "Does the model generalize to nearby universes?",
            [rel_path(UNIVERSE / "universe_robustness_summary.csv"), rel_path(UNIVERSE / "universe_coverage_audit.csv")],
            ["Filter official candidate rows by universe.", "Compare run status, usable tickers, CAGR, Sharpe and MaxDD."],
            {"official_universe_rows": records(official_universe, 20)},
            "The completed nearby technology/growth universe runs remain positive but generally degrade away from the original universe." if not official_universe.empty else NOT_AVAILABLE,
            "Moderate",
            "The negative-control universe has coverage evidence but not completed walk-forward metrics in the current outputs.",
        ),
        _question(
            "baseline-support",
            "What metrics support the official baseline?",
            [rel_path(BASELINE_OUTPUTS / "stitched_comparison_official.csv"), rel_path(BASELINE_OUTPUTS / "alpha_nw_official.csv"), rel_path(BASELINE_OUTPUTS / "pvalue_qvalue_official.csv"), rel_path(BASELINE_OUTPUTS / "fold_summary_official.csv")],
            ["Read stitched performance.", "Read Newey-West alpha/beta.", "Read p/q values and fold behavior."],
            {"CAGR": display_value(m("CAGR"), "percent_points"), "Sharpe": display_value(m("Sharpe"), "decimal"), "MaxDD": display_value(m("MaxDD"), "percent_points"), "AlphaNW_QQQ": display_value(m("AlphaNW_QQQ"), "decimal4"), "q_value_vs_SPY": display_value(m("q_value_vs_SPY"), "decimal4")},
            "The official baseline is supported by strong stitched performance, positive Newey-West alpha, acceptable drawdown, and fold-level evidence versus the historical control.",
            "Strong",
            "Support is historical and research-specific; it is not a guarantee of future performance.",
        ),
        _question(
            "limitations",
            "What limitations remain?",
            [rel_path(REPORTS / "final_extended_analysis_report.md"), rel_path(AUDIT_CUBE / "cube_dictionary.md")],
            ["Read current availability and materialized artifact limits.", "Do not infer metrics that are absent."],
            {"not_available": [m["metric_name"] for m in score["unavailable_metrics"]]},
            "The DSS can support conclusions present in existing artifacts, but it cannot prove global optimality, future performance, full universe generalization, or causal module effects.",
            "Strong",
            "All interpretations are deterministic readings of current artifacts only.",
        ),
    ]
    return {"questions": questions}


def _question(
    qid: str,
    question: str,
    sources: List[str],
    methodology: List[str],
    evidence: Dict[str, Any],
    conclusion: str,
    confidence: str,
    limitation: str,
) -> Dict[str, Any]:
    return {
        "id": qid,
        "question": question,
        "data_sources_used": sources,
        "methodology": methodology,
        "evidence_values": clean_value(evidence),
        "conclusion": conclusion,
        "confidence_level": confidence,
        "limitations": limitation,
    }


def _axis_value(df: pd.DataFrame, axis: str, column: str) -> Any:
    if df.empty or "axis" not in df.columns or column not in df.columns:
        return None
    row = df[df["axis"].astype(str) == axis]
    return row[column].iloc[0] if not row.empty else None


def _module_question(qid: str, question: str, module: Dict[str, Any]) -> Dict[str, Any]:
    rows = module.get(qid, [])
    rates = {int(row["horizon"]): numeric(row.get("helped_rate")) for row in rows if numeric(row.get("horizon")) is not None}
    one = rates.get(1)
    twenty = rates.get(20)
    if twenty is not None and one is not None:
        conclusion = f"{question.split(',')[0]} appears more useful at longer horizons in current audit artifacts." if twenty > one else "Current helped rates do not improve from 1d to 20d."
        confidence = "Moderate"
    else:
        conclusion = NOT_AVAILABLE
        confidence = "Not available"
    source_map = {
        "continuation": "continuation_activation_audit.csv",
        "leader": "leader_participation_audit.csv",
        "backoff": "backoff_audit.csv",
    }
    return _question(
        qid,
        question,
        [rel_path(AUDIT_CUBE / source_map[qid]), rel_path(AUDIT_CUBE / "outcome_cube.parquet")],
        ["Group helped flags by horizon.", "Compare short horizon to 20-day horizon."],
        {"rows": rows, "rate_1d": one, "rate_20d": twenty},
        conclusion,
        confidence,
        "Helped rate is ex-post association, not causal proof.",
    )


def ticker_contribution() -> Dict[str, Any]:
    df = parquet(AUDIT_CUBE / "position_cube.parquet")
    if df.empty:
        return {"top_positive": [], "top_negative": [], "all": []}
    if "candidate_id" in df.columns:
        df = df[df["candidate_id"].astype(str) == OFFICIAL_CANDIDATE_ID]
    if "universe_id" in df.columns:
        df = df[df["universe_id"].astype(str) == OFFICIAL_UNIVERSE_ID]
    if df.empty or "ticker" not in df.columns:
        return {"top_positive": [], "top_negative": [], "all": []}
    grouped = df.groupby("ticker").agg(
        selected_frequency=("selected_flag", "sum"),
        leader_frequency=("leader_flag", "sum"),
        total_pnl_contribution=("pnl_contribution", "sum"),
        mean_final_weight=("final_weight", "mean"),
        mean_base_score=("base_score", "mean"),
    ).reset_index()
    return {
        "top_positive": records(grouped.sort_values("total_pnl_contribution", ascending=False), 10),
        "top_negative": records(grouped.sort_values("total_pnl_contribution", ascending=True), 10),
        "all": records(grouped.sort_values("selected_frequency", ascending=False), 100),
    }


def module_effectiveness_summary() -> Dict[str, Any]:
    outcome = parquet(AUDIT_CUBE / "outcome_cube.parquet")
    decision = parquet(AUDIT_CUBE / "decision_date_cube.parquet")
    if not outcome.empty and "candidate_id" in outcome.columns:
        outcome = outcome[outcome["candidate_id"].astype(str) == OFFICIAL_CANDIDATE_ID]
    if not outcome.empty and "universe_id" in outcome.columns:
        outcome = outcome[outcome["universe_id"].astype(str) == OFFICIAL_UNIVERSE_ID]

    def flag_summary(flag_col: str, module_name: str) -> List[Dict[str, Any]]:
        if outcome.empty or flag_col not in outcome.columns or "horizon" not in outcome.columns:
            return []
        grouped = outcome.groupby("horizon").agg(
            count=(flag_col, "count"),
            helped_rate=(flag_col, "mean"),
            avg_alpha_vs_qqq=("realized_alpha_vs_qqq", "mean"),
        ).reset_index()
        grouped.insert(0, "module", module_name)
        return records(grouped.sort_values("horizon"), 20)

    if not decision.empty and "candidate_id" in decision.columns:
        decision = decision[decision["candidate_id"].astype(str) == OFFICIAL_CANDIDATE_ID]
    if not decision.empty and "universe_id" in decision.columns:
        decision = decision[decision["universe_id"].astype(str) == OFFICIAL_UNIVERSE_ID]
    backoff_counts = {
        "backoff_count": int((pd.to_numeric(decision.get("backoff_strength_applied"), errors="coerce").fillna(0) > 0).sum()) if not decision.empty and "backoff_strength_applied" in decision.columns else None,
        "hard_backoff_count": int(pd.to_numeric(decision.get("hard_backoff_flag"), errors="coerce").fillna(0).sum()) if not decision.empty and "hard_backoff_flag" in decision.columns else None,
    }
    return {
        "continuation": flag_summary("continuation_helped_flag", "Continuation"),
        "leader": flag_summary("leader_helped_flag", "Leader participation"),
        "backoff": flag_summary("backoff_helped_flag", "Backoff"),
        "backoff_counts": backoff_counts,
    }


def cube_operations() -> Dict[str, Any]:
    operations = [
        {"operation": "slice", "meaning": "Filter one analytical axis.", "tables_used": "Any cube with the axis.", "example_conclusion": "Official candidate behavior can be isolated without changing artifacts."},
        {"operation": "dice", "meaning": "Filter several axes together.", "tables_used": "decision_date_cube, position_cube, outcome_cube.", "example_conclusion": "Fold 3 NVDA 20d cases can be reviewed as a focused evidence set."},
        {"operation": "drill-down", "meaning": "Move from aggregate metrics to dates, tickers, modules and outcomes.", "tables_used": "decision_date_cube, position_cube, module_trace_cube, outcome_cube.", "example_conclusion": "A weak fold can be explained through concrete decision cases."},
        {"operation": "roll-up", "meaning": "Aggregate rows by fold, module, ticker, horizon or regime.", "tables_used": "Any cube.", "example_conclusion": "Ticker contribution can be summarized across selected dates."},
        {"operation": "decision-to-outcome join", "meaning": "Join decision state to realized future performance.", "tables_used": "decision_date_cube and outcome_cube.", "example_conclusion": "High participation can be checked against 20d alpha versus QQQ."},
        {"operation": "module attribution", "meaning": "Group branch traces by module and threshold state.", "tables_used": "module_trace_cube and outcome_cube.", "example_conclusion": "Continuation or backoff states can be associated with helped rates."},
        {"operation": "regime audit", "meaning": "Compare decisions under benchmark and market context.", "tables_used": "decision_date_cube, market_context_cube and outcome_cube.", "example_conclusion": "Hard backoff cases can be reviewed during weak breadth or drawdown regimes."},
        {"operation": "ticker contribution audit", "meaning": "Aggregate selected positions by ticker.", "tables_used": "position_cube.", "example_conclusion": "High contribution should be read with frequency and mean weight."},
        {"operation": "fold weakness audit", "meaning": "Aggregate outcomes by fold and inspect weak cases.", "tables_used": "fold_summary_official.csv and outcome_cube.", "example_conclusion": "Weakness can be separated by short and long horizon behavior."},
        {"operation": "budget sensitivity audit", "meaning": "Compare official and budget-axis candidates.", "tables_used": "extended_multiplier_summary.csv and plateau_radius_by_axis.csv.", "example_conclusion": "Low budget underdeployment is the main sampled local sensitivity."},
    ]
    return {
        "problem": "Aggregate backtests show whether the strategy performed well, but they do not explain why. The DSS links decisions, selected positions, active modules, future outcomes and market context through shared analytical axes.",
        "evidence_chain": ["Decision state", "Selected positions", "Active modules", "Future outcome", "Market context", "Research conclusion"],
        "analytical_axes": [
            {"axis": "date", "meaning": "Trading decision date."},
            {"axis": "fold", "meaning": "Walk-forward test segment."},
            {"axis": "candidate", "meaning": "Multiplier candidate or official reference."},
            {"axis": "universe", "meaning": "Eligible ticker set."},
            {"axis": "ticker", "meaning": "Security selected or evaluated."},
            {"axis": "module", "meaning": "Policy or signal module state."},
            {"axis": "horizon", "meaning": "Forward outcome window."},
            {"axis": "market regime", "meaning": "Benchmark and breadth context around the decision."},
        ],
        "operations": operations,
    }


def decision_cases(
    preset_id: str = "official-baseline",
    *,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    fold: Optional[int] = None,
    candidate_id: Optional[str] = None,
    universe_id: Optional[str] = None,
    limit: int = 120,
) -> Dict[str, Any]:
    presets = _preset_definitions()
    decision = parquet(AUDIT_CUBE / "decision_date_cube.parquet")
    outcome = parquet(AUDIT_CUBE / "outcome_cube.parquet")
    market = parquet(AUDIT_CUBE / "market_context_cube.parquet")
    if decision.empty:
        return {"presets": presets, "active_preset": preset_id, "count": 0, "cases": [], "result_text": "Showing 0 cases matching current filters."}
    filtered = _filter_preset(decision, outcome, preset_id)
    filtered = _apply_decision_filters(filtered, date_start, date_end, fold, candidate_id, universe_id)
    count = int(len(filtered.drop_duplicates(subset=[col for col in ["date", "fold", "candidate_id", "universe_id"] if col in filtered.columns])))
    filtered = filtered.sort_values("date").head(max(1, min(limit, 500))).copy()
    cases = _case_timeline(filtered, outcome, market)
    return {
        "presets": [{**preset, "count": _preset_count(decision, outcome, preset["id"])} for preset in presets],
        "active_preset": preset_id,
        "count": count,
        "cases": cases,
        "result_text": f"Showing {count} cases matching current filters.",
        "explanation": next((preset["selected_explanation"] for preset in presets if preset["id"] == preset_id), ""),
    }


def _preset_definitions() -> List[Dict[str, Any]]:
    return [
        {"id": "official-baseline", "title": "Official baseline decisions", "what_it_means": "Decision dates for the frozen official reference candidate.", "research_question": "What does the promoted baseline do date by date?", "tables_used": ["decision_date_cube", "outcome_cube", "position_cube", "module_trace_cube"], "expected_interpretation": "Use as the neutral audit path before comparing stresses.", "selected_explanation": "You are reviewing official baseline decisions. These cases show the frozen reference behavior across folds."},
        {"id": "hard-backoff", "title": "Hard backoff dates", "what_it_means": "Dates where the hard defensive guard was active.", "research_question": "Did defensive reduction help during fragile states?", "tables_used": ["decision_date_cube", "outcome_cube", "market_context_cube"], "expected_interpretation": "Review whether risk-reduction states were followed by better or worse outcomes.", "selected_explanation": "You are reviewing hard backoff dates. These cases test whether risk-reduction states were followed by better or worse outcomes."},
        {"id": "low-long-budget", "title": "Low long-budget dates", "what_it_means": "Bottom-quartile long-budget cases.", "research_question": "What happens when participation is reduced?", "tables_used": ["decision_date_cube", "outcome_cube"], "expected_interpretation": "Look for underdeployment or drawdown-control evidence.", "selected_explanation": "You are reviewing low-budget cases. These cases test whether reduced participation avoided risk or missed returns."},
        {"id": "high-long-budget", "title": "High long-budget dates", "what_it_means": "Top-quartile long-budget cases.", "research_question": "What happens when participation is high?", "tables_used": ["decision_date_cube", "outcome_cube", "position_cube"], "expected_interpretation": "Check whether high participation was rewarded at longer horizons.", "selected_explanation": "You are reviewing high-budget cases. These cases test whether high participation was followed by stronger relative outcomes."},
        {"id": "continuation-active", "title": "Continuation active", "what_it_means": "Continuation probability at or above 0.50.", "research_question": "Does continuation help, and at what horizon?", "tables_used": ["decision_date_cube", "outcome_cube", "module_trace_cube"], "expected_interpretation": "Compare 1d, 5d and 20d helped chips.", "selected_explanation": "You are reviewing continuation-active cases. These cases test whether continuation states improved future outcomes."},
        {"id": "leader-active", "title": "Leader active", "what_it_means": "Selected positions include leader-flagged names.", "research_question": "Does leader participation help?", "tables_used": ["position_cube", "outcome_cube"], "expected_interpretation": "Read leader activity with ticker contribution and horizon outcomes.", "selected_explanation": "You are reviewing leader-active cases. These cases test whether leader participation supported outcomes."},
        {"id": "beat-qqq-20d", "title": "Beat QQQ at 20d", "what_it_means": "20-day outcome flag versus QQQ is positive.", "research_question": "Which decisions worked at the research horizon?", "tables_used": ["outcome_cube", "decision_date_cube"], "expected_interpretation": "Use successful cases to inspect common states and modules.", "selected_explanation": "You are reviewing decisions that beat QQQ after 20 trading days."},
        {"id": "failed-qqq-20d", "title": "Failed QQQ at 20d", "what_it_means": "20-day outcome flag versus QQQ is not positive.", "research_question": "Which decisions need scrutiny?", "tables_used": ["outcome_cube", "decision_date_cube"], "expected_interpretation": "Use weak cases to inspect fold, budget, regime and module context.", "selected_explanation": "You are reviewing decisions that did not beat QQQ after 20 trading days."},
    ]


def _filter_preset(decision: pd.DataFrame, outcome: pd.DataFrame, preset_id: str) -> pd.DataFrame:
    if preset_id == "hard-backoff" and "hard_backoff_flag" in decision.columns:
        return decision[pd.to_numeric(decision["hard_backoff_flag"], errors="coerce").fillna(0) > 0]
    if preset_id == "low-long-budget" and "long_budget" in decision.columns:
        values = pd.to_numeric(decision["long_budget"], errors="coerce")
        return decision[values <= values.quantile(0.25)]
    if preset_id == "high-long-budget" and "long_budget" in decision.columns:
        values = pd.to_numeric(decision["long_budget"], errors="coerce")
        return decision[values >= values.quantile(0.75)]
    if preset_id == "continuation-active" and "continuation_trigger_p" in decision.columns:
        return decision[pd.to_numeric(decision["continuation_trigger_p"], errors="coerce").fillna(0) >= 0.5]
    if preset_id == "leader-active":
        position = parquet(AUDIT_CUBE / "position_cube.parquet")
        if position.empty or not {"date", "fold", "candidate_id", "universe_id", "leader_flag", "selected_flag"}.issubset(position.columns):
            return decision.iloc[0:0]
        pos = position[(pd.to_numeric(position["leader_flag"], errors="coerce").fillna(0) > 0) & (pd.to_numeric(position["selected_flag"], errors="coerce").fillna(0) > 0)]
        keys = pos[["date", "fold", "candidate_id", "universe_id"]].drop_duplicates()
        return decision.merge(keys, on=["date", "fold", "candidate_id", "universe_id"], how="inner")
    if preset_id in {"beat-qqq-20d", "failed-qqq-20d"} and not outcome.empty:
        twenty = outcome[pd.to_numeric(outcome.get("horizon"), errors="coerce") == 20].copy()
        flag = pd.to_numeric(twenty.get("decision_helped_flag_vs_qqq"), errors="coerce").fillna(0) > 0
        twenty = twenty[flag if preset_id == "beat-qqq-20d" else ~flag]
        keys = twenty.rename(columns={"decision_date": "date"})[["date", "fold", "candidate_id", "universe_id"]].drop_duplicates()
        return decision.merge(keys, on=["date", "fold", "candidate_id", "universe_id"], how="inner")
    if "candidate_id" in decision.columns:
        return decision[decision["candidate_id"].astype(str) == OFFICIAL_CANDIDATE_ID]
    return decision


def _preset_count(decision: pd.DataFrame, outcome: pd.DataFrame, preset_id: str) -> int:
    return int(len(_filter_preset(decision, outcome, preset_id)))


def _apply_decision_filters(
    df: pd.DataFrame,
    date_start: Optional[str],
    date_end: Optional[str],
    fold: Optional[int],
    candidate_id: Optional[str],
    universe_id: Optional[str],
) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns and (date_start or date_end):
        dates = pd.to_datetime(out["date"], errors="coerce")
        if date_start:
            out = out[dates >= pd.Timestamp(date_start)]
            dates = pd.to_datetime(out["date"], errors="coerce")
        if date_end:
            out = out[dates <= pd.Timestamp(date_end)]
    if fold is not None and "fold" in out.columns:
        out = out[out["fold"] == int(fold)]
    if candidate_id and "candidate_id" in out.columns:
        out = out[out["candidate_id"].astype(str) == candidate_id]
    if universe_id and "universe_id" in out.columns:
        out = out[out["universe_id"].astype(str) == universe_id]
    return out


def _case_timeline(decision: pd.DataFrame, outcome: pd.DataFrame, market: pd.DataFrame) -> List[Dict[str, Any]]:
    if decision.empty:
        return []
    out = decision.copy()
    out["date_key"] = normalize_date_series(out["date"])
    if not market.empty and "date" in market.columns:
        m = market.copy()
        m["date_key"] = normalize_date_series(m["date"])
        keep = ["date_key", "market_regime_proxy", "qqq_drawdown", "breadth"]
        out = out.merge(m[[col for col in keep if col in m.columns]].drop_duplicates("date_key"), on="date_key", how="left")
    if not outcome.empty:
        oc = outcome.copy()
        oc = oc[pd.to_numeric(oc.get("horizon"), errors="coerce") == 20]
        oc["date_key"] = normalize_date_series(oc["decision_date"])
        keep = ["date_key", "fold", "candidate_id", "universe_id", "realized_alpha_vs_qqq", "decision_helped_flag_vs_qqq", "decision_helped_flag_vs_control"]
        out = out.merge(oc[[col for col in keep if col in oc.columns]], on=["date_key", "fold", "candidate_id", "universe_id"], how="left")
    candidates = {row["technical_id"]: row for row in candidate_metadata().get("candidates", [])}
    rows = []
    for row in records(out, 500):
        cid = str(row.get("candidate_id"))
        meta = candidates.get(cid, {})
        rows.append(
            {
                "date": row.get("date_key") or row.get("date"),
                "fold": row.get("fold"),
                "candidate_id": cid,
                "candidate_label": meta.get("label", cid),
                "universe_id": row.get("universe_id"),
                "participation_state": row.get("participation_state"),
                "long_budget": row.get("long_budget"),
                "market_regime": row.get("market_regime_proxy"),
                "outcome_20d_vs_qqq": row.get("realized_alpha_vs_qqq"),
                "beat_qqq_20d": row.get("decision_helped_flag_vs_qqq"),
                "beat_control_20d": row.get("decision_helped_flag_vs_control"),
                "key_module_state": _key_module_state(row.get("date"), row.get("fold"), cid, row.get("universe_id")),
            }
        )
    return rows


def _key_module_state(date_value: Any, fold: Any, candidate_id: str, universe_id: Any) -> str:
    trace = parquet(AUDIT_CUBE / "module_trace_cube.parquet")
    if trace.empty:
        return NOT_AVAILABLE
    date_key = pd.Timestamp(date_value).strftime("%Y-%m-%d") if date_value is not None else None
    work = trace.copy()
    work["date_key"] = normalize_date_series(work["date"])
    work = work[
        (work["date_key"] == date_key)
        & (work["candidate_id"].astype(str) == str(candidate_id))
        & (work["universe_id"].astype(str) == str(universe_id))
    ]
    if fold is not None and "fold" in work.columns:
        work = work[work["fold"] == int(fold)]
    if work.empty:
        return NOT_AVAILABLE
    active = work[pd.to_numeric(work.get("threshold_crossed"), errors="coerce").fillna(0) > 0]
    selected = active if not active.empty else work
    bits = [f"{row.module_name}:{row.branch_taken}" for row in selected.head(2).itertuples()]
    return ", ".join(bits)

