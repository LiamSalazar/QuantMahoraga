from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import acceptance_suite_14_3R as acc
import fast_fail_diagnostics_14_3 as d14
import mahoraga6_1 as m6
import promotion_gate_suite as pg
from mahoraga14_config import Mahoraga14Config
from mahoraga14_utils import ensure_dir


OFFICIAL_VARIANT_LABEL = "MAHORAGA14_3_BASELINE_OFFICIAL"


def _ensure_dirs(cfg: Mahoraga14Config) -> Dict[str, Path]:
    roots = {
        "outputs": Path(cfg.outputs_dir),
        "audit": Path(cfg.audit_dir),
        "paper_pack": Path(cfg.paper_pack_dir),
        "docs": Path(cfg.docs_dir),
        "manifests": Path(cfg.manifests_dir),
        "plots": Path(cfg.plots_dir),
        "config": Path(cfg.config_dir),
    }
    for root in roots.values():
        ensure_dir(str(root))
    return roots


def _official_candidate_payload(
    wf: Dict[str, Any],
    cfg: Mahoraga14Config,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    candidate_objects, context, specs = pg._candidate_objects(wf, cfg)
    return candidate_objects[cfg.official_candidate_id], context, specs, candidate_objects


def _filtered_gate_specs(specs: List[Dict[str, Any]], cfg: Mahoraga14Config) -> List[Dict[str, Any]]:
    return [spec for spec in specs if str(spec["CandidateId"]) == cfg.official_candidate_id]


def _official_stitched_df(
    candidate_objects: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
    specs: List[Dict[str, Any]],
    cfg: Mahoraga14Config,
) -> pd.DataFrame:
    df = pg._stitched_metrics_df(candidate_objects, context, specs, cfg)
    out = df[df["Variant"].isin(["QQQ", "SPY", "MAHORAGA14_1_LONG_ONLY_CONTROL", cfg.official_candidate_id])].copy()
    out.loc[out["Variant"] == cfg.official_candidate_id, "Variant"] = OFFICIAL_VARIANT_LABEL
    out.loc[out["Variant"] == OFFICIAL_VARIANT_LABEL, "CandidateId"] = cfg.official_candidate_id
    return out.reset_index(drop=True)


def _official_fold_df(
    wf: Dict[str, Any],
    candidate_objects: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
    specs: List[Dict[str, Any]],
    cfg: Mahoraga14Config,
) -> pd.DataFrame:
    df = pg._fold_summary_df(wf, candidate_objects, context, specs, cfg)
    out = df[df["Variant"].isin(["QQQ", "SPY", "MAHORAGA14_1_LONG_ONLY_CONTROL", cfg.official_candidate_id])].copy()
    out.loc[out["Variant"] == cfg.official_candidate_id, "Variant"] = OFFICIAL_VARIANT_LABEL
    out.loc[out["Variant"] == OFFICIAL_VARIANT_LABEL, "CandidateId"] = cfg.official_candidate_id
    return out.reset_index(drop=True)


def _official_bull_window_df(
    candidate_objects: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
    specs: List[Dict[str, Any]],
    cfg: Mahoraga14Config,
) -> pd.DataFrame:
    df = pg._bull_window_scorecard_df(candidate_objects, context, specs, cfg)
    out = df[df["CandidateId"] == cfg.official_candidate_id].copy()
    out["Variant"] = OFFICIAL_VARIANT_LABEL
    return out.reset_index(drop=True)


def _official_priority_df(bull_df: pd.DataFrame, cfg: Mahoraga14Config) -> pd.DataFrame:
    df = pg._priority_window_df(bull_df, cfg).copy()
    df["Variant"] = OFFICIAL_VARIANT_LABEL
    return df.reset_index(drop=True)


def _official_active_df(
    candidate_objects: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
    specs: List[Dict[str, Any]],
    cfg: Mahoraga14Config,
) -> pd.DataFrame:
    df = pg._active_return_df(candidate_objects, context, specs).copy()
    keep_cols = [
        "Date",
        "QQQReturn",
        "SPYReturn",
        "ControlReturn",
        "CumControl",
        f"{cfg.official_candidate_id}_Return",
        f"{cfg.official_candidate_id}_CumReturn",
        f"{cfg.official_candidate_id}_Active_vs_QQQ",
        f"{cfg.official_candidate_id}_CumActive_vs_QQQ",
    ]
    out = df[keep_cols].copy()
    return out.rename(
        columns={
            f"{cfg.official_candidate_id}_Return": "OfficialReturn",
            f"{cfg.official_candidate_id}_CumReturn": "CumOfficial",
            f"{cfg.official_candidate_id}_Active_vs_QQQ": "ActiveReturn_vs_QQQ",
            f"{cfg.official_candidate_id}_CumActive_vs_QQQ": "CumActiveReturn_vs_QQQ",
        }
    )


def _official_alpha_df(
    candidate_objects: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
    specs: List[Dict[str, Any]],
    cfg: Mahoraga14Config,
) -> pd.DataFrame:
    df = pg._alpha_nw_df(candidate_objects, context, specs, cfg)
    out = df[df["Variant"].isin(["MAHORAGA14_1_LONG_ONLY_CONTROL", cfg.official_candidate_id])].copy()
    out.loc[out["Variant"] == cfg.official_candidate_id, "Variant"] = OFFICIAL_VARIANT_LABEL
    return out.reset_index(drop=True)


def _official_pq_df(
    stitched_df: pd.DataFrame,
    candidate_objects: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
    specs: List[Dict[str, Any]],
    cfg: Mahoraga14Config,
) -> pd.DataFrame:
    raw_df = pg._stitched_metrics_df(candidate_objects, context, specs, cfg)
    df = pg._pvalue_qvalue_df(raw_df, candidate_objects, context, specs).copy()
    out = df[df["Variant"] == cfg.official_candidate_id].copy()
    out["Variant"] = OFFICIAL_VARIANT_LABEL
    return out.reset_index(drop=True)


def _official_cost_slippage_df(
    official_obj: Dict[str, Any],
    context: Dict[str, Any],
    cfg: Mahoraga14Config,
) -> pd.DataFrame:
    returns = pd.Series(official_obj["returns"], dtype=float)
    gross = pd.Series(official_obj["gross_returns"], dtype=float).reindex(returns.index).fillna(returns)
    tc = pd.Series(official_obj["transaction_cost"], dtype=float).reindex(returns.index).fillna(0.0)
    exposure = pd.Series(official_obj["exposure"], dtype=float).reindex(returns.index).fillna(0.0)
    turnover = pd.Series(official_obj["turnover"], dtype=float).reindex(returns.index).fillna(0.0)
    qqq_obj = context["qqq"]
    spy_obj = context["spy"]

    scenarios = {
        "BASELINE": gross - tc.abs(),
        "COST_PLUS_25": gross - tc.abs() * 1.25,
        "COST_PLUS_50": gross - tc.abs() * 1.50,
        "COST_PLUS_100": gross - tc.abs() * 2.00,
        "SLIPPAGE_PLUS_5BPS": (gross - tc.abs()) - turnover * 0.0005,
    }
    rows: List[Dict[str, Any]] = []
    base_metrics = d14._metrics_row(OFFICIAL_VARIANT_LABEL, official_obj, qqq_obj, spy_obj, cfg)
    for scenario, scenario_returns in scenarios.items():
        obj = {
            "returns": scenario_returns,
            "gross_returns": gross,
            "transaction_cost": tc,
            "exposure": exposure,
            "turnover": turnover,
            "equity": cfg.capital_initial * (1.0 + pd.Series(scenario_returns, dtype=float)).cumprod(),
        }
        metrics = d14._metrics_row(f"{OFFICIAL_VARIANT_LABEL}_{scenario}", obj, qqq_obj, spy_obj, cfg)
        rows.append(
            {
                "Variant": OFFICIAL_VARIANT_LABEL,
                "Scenario": scenario,
                "CAGR": float(metrics["CAGR"]) * 100.0,
                "Sharpe": float(metrics["Sharpe"]),
                "Sortino": float(metrics["Sortino"]),
                "MaxDD": float(metrics["MaxDD"]) * 100.0,
                "AlphaNW_QQQ": float(metrics["AlphaNW_QQQ"]),
                "AlphaNW_SPY": float(metrics["AlphaNW_SPY"]),
                "DeltaCAGR_vs_Base": float(metrics["CAGR"] - base_metrics["CAGR"]) * 100.0,
                "DeltaSharpe_vs_Base": float(metrics["Sharpe"] - base_metrics["Sharpe"]),
                "DeltaSortino_vs_Base": float(metrics["Sortino"] - base_metrics["Sortino"]),
                "DeltaMaxDD_vs_Base": float(metrics["MaxDD"] - base_metrics["MaxDD"]) * 100.0,
            }
        )
    return pd.DataFrame(rows).round(8)


def _official_allocator_cash_drag(
    wf: Dict[str, Any],
    cfg: Mahoraga14Config,
) -> pd.DataFrame:
    df = d14._build_allocator_cash_drag(wf, cfg).copy()
    idx = pd.to_datetime(df["Date"])
    cash_target = pd.Series(df.get("cash_budget_target", cfg.participation_allocator_cash_target_floor), index=idx, dtype=float)
    budget_scale = np.minimum(1.0, cash_target * float(cfg.official_budget_multiplier)) / cash_target.replace(0.0, np.nan)
    df["OfficialCandidateId"] = cfg.official_candidate_id
    df["OfficialBudgetMultiplier"] = float(cfg.official_budget_multiplier)
    df["OfficialConvictionMultiplier"] = float(cfg.official_conviction_multiplier)
    df["OfficialLeaderMultiplier"] = float(cfg.official_leader_multiplier)
    df["OfficialBackoffStrength"] = float(cfg.official_backoff_strength)
    df["OfficialBudgetScaleEstimate"] = pd.Series(budget_scale, index=idx).fillna(1.0).values
    return df.round(8)


def _official_architecture_diag(df: pd.DataFrame, cfg: Mahoraga14Config) -> pd.DataFrame:
    out = df.copy()
    out["OfficialCandidateId"] = cfg.official_candidate_id
    out["DiagnosticScope"] = "ARCHITECTURE_LEVEL_WITH_FROZEN_PROMOTION_MULTIPLIERS"
    return out


def _official_turnover_summary(
    official_obj: Dict[str, Any],
    control_obj: Dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for label, obj in ((OFFICIAL_VARIANT_LABEL, official_obj), ("MAHORAGA14_1_LONG_ONLY_CONTROL", control_obj)):
        s = pd.Series(obj["turnover"], dtype=float)
        rows.append(
            {
                "Variant": label,
                "mean": float(s.mean()),
                "median": float(s.median()),
                "p05": float(s.quantile(0.05)),
                "p95": float(s.quantile(0.95)),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows).round(8)


def _official_exposure_summary(
    official_obj: Dict[str, Any],
    control_obj: Dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for label, obj in ((OFFICIAL_VARIANT_LABEL, official_obj), ("MAHORAGA14_1_LONG_ONLY_CONTROL", control_obj)):
        s = pd.Series(obj["exposure"], dtype=float)
        rows.append(
            {
                "Variant": label,
                "mean": float(s.mean()),
                "median": float(s.median()),
                "p05": float(s.quantile(0.05)),
                "p95": float(s.quantile(0.95)),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows).round(8)


def _official_return_per_exposure(
    official_obj: Dict[str, Any],
    control_obj: Dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for label, obj in ((OFFICIAL_VARIANT_LABEL, official_obj), ("MAHORAGA14_1_LONG_ONLY_CONTROL", control_obj)):
        total_ret = float(np.prod(1.0 + pd.Series(obj["returns"], dtype=float).values) - 1.0)
        rows.append(
            {
                "Variant": label,
                "ReturnPerExposure": float(d14._return_per_exposure(pd.Series(obj["returns"], dtype=float), pd.Series(obj["exposure"], dtype=float))),
                "TotalReturn": total_ret,
                "AvgExposure": float(pd.Series(obj["exposure"], dtype=float).mean()),
                "Observations": int(len(pd.Series(obj["returns"], dtype=float))),
            }
        )
    return pd.DataFrame(rows).round(8)


def _official_model_selection_guard(
    family_df: pd.DataFrame,
    candidate_objects: Dict[str, Dict[str, Any]],
    control_obj: Dict[str, Any],
    cfg: Mahoraga14Config,
) -> Tuple[pd.DataFrame, str]:
    candidate_ids = list(family_df["CandidateId"])
    excess = np.column_stack(
        [
            pd.Series(candidate_objects[cid]["returns"], dtype=float).reindex(control_obj["returns"].index).fillna(0.0).values
            - pd.Series(control_obj["returns"], dtype=float).values
            for cid in candidate_ids
        ]
    )
    observed_mean = excess.mean(axis=0)
    centered = excess - observed_mean
    indices = acc._stationary_bootstrap_indices(
        len(control_obj["returns"]),
        cfg.acceptance_bootstrap_block,
        cfg.acceptance_model_selection_bootstrap_samples,
        seed=41,
    )
    boot_max = []
    official_idx = candidate_ids.index(cfg.official_candidate_id)
    official_boot = []
    for idx in indices:
        sample = centered[idx, :]
        boot_mean = sample.mean(axis=0)
        boot_max.append(float(boot_mean.max()))
        official_boot.append(float(boot_mean[official_idx]))
    family_p = float(np.mean(np.asarray(boot_max) >= float(observed_mean.max())))
    official_p = float(np.mean(np.asarray(official_boot) >= float(observed_mean[official_idx])))
    out = family_df[["CandidateId", "RobustCandidate", "RobustScore"]].copy()
    out["MeanDailyExcess_vs_Control"] = observed_mean
    out["FamilyRealityCheckPValue"] = family_p
    out["OfficialCandidatePValue"] = official_p
    md = "\n".join(
        [
            "# Model Selection Guard",
            "",
            "- This baseline freeze inherits multiple prior iterations, so model selection is documented conservatively.",
            f"- Local family reality-check p-value: {family_p:.4f}",
            f"- Official candidate `{cfg.official_candidate_id}` centered bootstrap p-value: {official_p:.4f}",
            "- Interpretation: the selected baseline is defended as the best robust point in the accepted local plateau, not as a global optimum claim.",
        ]
    )
    return out.round(8), md


def _official_local_stability_md(family_df: pd.DataFrame, cfg: Mahoraga14Config) -> str:
    selected = family_df[family_df["CandidateId"] == cfg.official_candidate_id].iloc[0]
    base_id = acc._candidate_id(1.0, 1.0, 1.0, 1.0)
    base_row = family_df[family_df["CandidateId"] == base_id].iloc[0]
    plateau_share = float(family_df["RobustCandidate"].sum() / max(1, len(family_df)))
    lines = [
        "# Local Stability Summary",
        "",
        f"- base-current candidate: `{base_id}` with robust score {float(base_row['RobustScore']):.4f}",
        f"- official promoted candidate: `{cfg.official_candidate_id}` with robust score {float(selected['RobustScore']):.4f}",
        f"- official promoted candidate robust flag: {int(selected['RobustCandidate'])}",
        f"- official promoted candidate local rank by robust score: {int(family_df.index[family_df['CandidateId'] == cfg.official_candidate_id][0] + 1)}",
        f"- plateau share in accepted local neighborhood: {plateau_share:.2%}",
        "- conclusion: the official baseline is promoted from inside a robust local plateau rather than from a fragile single-point optimum.",
    ]
    return "\n".join(lines)


def _official_loo_md(loo_df: pd.DataFrame, cfg: Mahoraga14Config) -> str:
    chosen_all = (loo_df["SelectedCandidateId"] == cfg.official_candidate_id).all()
    lines = [
        "# Leave-One-Window-Out Summary",
        "",
        f"- official candidate selected in every exclusion experiment: {int(bool(chosen_all))}",
        "",
    ]
    for _, row in loo_df.iterrows():
        lines.extend(
            [
                f"## Excluded {row['ExcludedWindow']}",
                f"- selected candidate: `{row['SelectedCandidateId']}`",
                f"- excluded-window delta vs control: {float(row['EvalDelta_vs_Control']):.4f}",
                f"- excluded-window delta vs QQQ: {float(row['EvalDelta_vs_QQQ']):.4f}",
                "",
            ]
        )
    return "\n".join(lines)


def _acceptance_decision_md(
    decision_df: pd.DataFrame,
    priority_df: pd.DataFrame,
    cfg: Mahoraga14Config,
) -> str:
    selected = decision_df.iloc[0]
    lines = [
        "# Acceptance Decision Official",
        "",
        f"- official baseline: `{OFFICIAL_VARIANT_LABEL}`",
        f"- frozen candidate id: `{cfg.official_candidate_id}`",
        f"- replaces baseline: `{cfg.replaced_baseline}`",
        f"- promotion decision carried forward from promotion gate: `{selected['GateRole']}`",
        f"- priority windows status: 2017_2018={selected['Window2017_2018']}, 2020_2021={selected['Window2020_2021']}, 2023_2024={selected['Window2023_2024']}",
        "",
        "## Priority detail",
        f"```\n{priority_df.to_string(index=False)}\n```",
    ]
    return "\n".join(lines)


def _plot_equity_curve(active_df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(10, 5.5))
    dates = pd.to_datetime(active_df["Date"])
    plt.plot(dates, active_df["CumOfficial"], label=OFFICIAL_VARIANT_LABEL, linewidth=1.8)
    plt.plot(dates, active_df["CumControl"], label="MAHORAGA14_1_LONG_ONLY_CONTROL", linewidth=1.2)
    plt.plot(dates, (1.0 + active_df["QQQReturn"]).cumprod() - 1.0, label="QQQ", linewidth=1.2)
    plt.plot(dates, (1.0 + active_df["SPYReturn"]).cumprod() - 1.0, label="SPY", linewidth=1.2)
    plt.legend()
    plt.title("Official Baseline Equity Curve")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _plot_active_return(active_df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(10, 5.5))
    plt.plot(pd.to_datetime(active_df["Date"]), active_df["CumActiveReturn_vs_QQQ"], label="Active vs QQQ", linewidth=1.8)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.legend()
    plt.title("Official Baseline Active Return vs QQQ")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _plot_bull_windows(priority_df: pd.DataFrame, out_png: Path) -> None:
    names = list(priority_df["Window"])
    x = np.arange(len(names))
    width = 0.25
    plt.figure(figsize=(9, 5.5))
    plt.bar(x - width, priority_df["CandidateReturn"], width=width, label=OFFICIAL_VARIANT_LABEL)
    plt.bar(x, priority_df["ControlReturn"], width=width, label="14.1 Control")
    plt.bar(x + width, priority_df["QQQReturn"], width=width, label="QQQ")
    plt.xticks(x, names)
    plt.title("Official Priority Bull Windows")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _plot_fold_heatmap(fold_df: pd.DataFrame, out_png: Path) -> None:
    cand = fold_df[fold_df["Variant"] == OFFICIAL_VARIANT_LABEL].set_index("Fold")
    ctrl = fold_df[fold_df["Variant"] == "MAHORAGA14_1_LONG_ONLY_CONTROL"].set_index("Fold")
    delta = pd.DataFrame(
        {
            "CAGR": cand["CAGR"] - ctrl["CAGR"],
            "Sharpe": cand["Sharpe"] - ctrl["Sharpe"],
            "Sortino": cand["Sortino"] - ctrl["Sortino"],
            "MaxDD": cand["MaxDD"] - ctrl["MaxDD"],
        }
    )
    plt.figure(figsize=(7.5, 4.5))
    plt.imshow(delta.values, aspect="auto", origin="lower", cmap="RdYlGn")
    plt.xticks(range(len(delta.columns)), list(delta.columns))
    plt.yticks(range(len(delta.index)), [str(x) for x in delta.index])
    plt.colorbar(label="Candidate minus control")
    plt.title("Official Baseline Fold Delta Heatmap")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _summary_stats_for_paper(
    stitched_df: pd.DataFrame,
    priority_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
) -> pd.DataFrame:
    official = stitched_df[stitched_df["Variant"] == OFFICIAL_VARIANT_LABEL].copy()
    control = stitched_df[stitched_df["Variant"] == "MAHORAGA14_1_LONG_ONLY_CONTROL"].copy()
    priority = priority_df[["Window", "CandidateReturn", "ControlReturn", "QQQReturn", "GateStatus"]].copy()
    priority["Variant"] = OFFICIAL_VARIANT_LABEL
    alpha = alpha_df[alpha_df["Variant"] == OFFICIAL_VARIANT_LABEL].copy()
    official["Section"] = "stitched"
    control["Section"] = "stitched_control"
    priority["Section"] = "priority_windows"
    alpha["Section"] = "alpha"
    return pd.concat([official, control, priority, alpha], ignore_index=True, sort=False).round(8)


def _table_manifest() -> pd.DataFrame:
    rows = [
        ("stitched_comparison_official.csv", "Main stitched OOS comparison for baseline, control and benchmarks"),
        ("fold_summary_official.csv", "Fold-by-fold metrics for official baseline and controls"),
        ("bull_window_scorecard_official.csv", "Automatic and manual bull-window scorecard for the official candidate"),
        ("priority_window_acceptance_official.csv", "Priority bull-window gate classification"),
        ("alpha_nw_official.csv", "Newey-West alpha table versus QQQ and SPY"),
        ("pvalue_qvalue_official.csv", "Paired p/q-value comparisons versus control and benchmarks"),
        ("acceptance_robustness_summary_official.csv", "Stress and robustness summary for the official candidate"),
        ("bootstrap_summary_official.csv", "Stationary block bootstrap summary versus control"),
    ]
    return pd.DataFrame(rows, columns=["table_file", "description"])


def _figure_manifest() -> pd.DataFrame:
    rows = [
        ("equity_curve_official.png", "Official baseline cumulative equity versus control and benchmarks"),
        ("active_return_vs_qqq_official.png", "Official baseline cumulative active return versus QQQ"),
        ("bull_window_scorecard_official.png", "Priority bull-window bar chart"),
        ("fold_heatmap_official.png", "Fold delta heatmap versus control"),
        ("robustness_distribution_official.png", "Bootstrap robustness distribution"),
        ("local_stability_heatmap_official.png", "Local stability heatmap from the accepted plateau"),
    ]
    return pd.DataFrame(rows, columns=["figure_file", "description"])


def _claims_supported_md() -> str:
    return "\n".join(
        [
            "# Claims Supported By Outputs",
            "",
            "- The official baseline outperforms the historical 14.1 control on stitched CAGR, Sharpe, Sortino and benchmark-adjusted alpha: `outputs/stitched_comparison_official.csv`.",
            "- The promoted candidate passes all three priority bull windows and repairs the prior 2017_2018 failure: `outputs/priority_window_acceptance_official.csv`.",
            "- Performance does not rely on a single fold collapse: `outputs/fold_summary_official.csv` and `outputs/fold_heatmap_official.png`.",
            "- Promotion was made from within a local plateau and documented with conservative selection controls: `audit/local_stability_summary_official.md` and `audit/model_selection_guard_official.md`.",
            "- Continuation remains a quality filter rather than the sole participation lever: `audit/continuation_acceptance_official.md` and `audit/continuation_diagnostic_official.csv`.",
        ]
    )


def _references_needed_md() -> str:
    return "\n".join(
        [
            "# References Needed",
            "",
            "- Time-series momentum / regime persistence literature supporting continuation as a quality filter.",
            "- Volatility-managed portfolios literature supporting state-dependent exposure control.",
            "- Residual momentum / leader participation literature supporting conditional leader inclusion.",
            "- Model selection / White reality check / SPA references for conservative promotion discipline.",
            "- Benchmark-adjusted alpha estimation references for Newey-West style inference.",
        ]
    )


def _module_interface_map_md() -> str:
    return "\n".join(
        [
            "# Module Interface Map",
            "",
            "## BASE_ALPHA_V2",
            "- inputs: universe OHLCV, fold schedule, shared precomputed state",
            "- outputs: base long-book weights and stitched baseline traces",
            "",
            "## PARTICIPATION_ALLOCATOR_V2",
            "- inputs: weekly continuation, breadth, volatility, benchmark and structural context",
            "- outputs: long budget, gate scale, vol multiplier, exp cap, leader blend, conviction multiplier",
            "",
            "## CONVICTION_AMPLIFIER_LAYER",
            "- inputs: allocator raw state plus healthy-regime context",
            "- outputs: amplified budget/gate/vol/exp-cap translations",
            "",
            "## LEADER_PARTICIPATION_LAYER",
            "- inputs: base long book, allocator state, leader opportunity state",
            "- outputs: conditional leader blend and redeployed cash-drag participation",
            "",
            "## RISK_BACKOFF_LAYER_V2",
            "- inputs: fragility, break-risk, benchmark weakness, continuation quality",
            "- outputs: clipped budget / conviction / leader participation under stress",
            "",
            "## continuation as quality filter",
            "- inputs: continuation trigger / pressure / break-risk models",
            "- outputs: quality signal used to allow or restrain participation; not a separate thesis in the official baseline",
        ]
    )


def _component_audit_md() -> str:
    return "\n".join(
        [
            "# Component Audit",
            "",
            "## BASE_ALPHA_V2",
            "- does: generates the long-only stock-selection core inherited from the frozen 14.1 baseline.",
            "- does not: decide final bull participation on its own.",
            "",
            "## PARTICIPATION_ALLOCATOR_V2",
            "- does: converts healthy regime evidence into budget/cap participation controls.",
            "- does not: invent new alpha or override the book unconditionally.",
            "",
            "## CONVICTION_AMPLIFIER_LAYER",
            "- does: scale the translation from conviction to effective participation.",
            "- does not: bypass risk backoff or create a new model family.",
            "",
            "## LEADER_PARTICIPATION_LAYER",
            "- does: conditionally lift exposure toward leaders and reduce cash drag.",
            "- does not: add shorts, hedges or permanent tech concentration.",
            "",
            "## RISK_BACKOFF_LAYER_V2",
            "- does: harden the system when fragility, break-risk or benchmark weakness rise.",
            "- does not: pursue upside participation.",
            "",
            "## continuation",
            "- does: act as a quality filter with positive local edge.",
            "- does not: bear sole responsibility for bull participation in the official baseline.",
        ]
    )


def _decision_flow_md() -> str:
    return "\n".join(
        [
            "# Decision Flow",
            "",
            "1. `BASE_ALPHA_V2` proposes the long-only book.",
            "2. `PARTICIPATION_ALLOCATOR_V2` reads market/book state and sets participation budget and caps.",
            "3. `CONVICTION_AMPLIFIER_LAYER` increases translation strength in healthy bull regimes.",
            "4. `LEADER_PARTICIPATION_LAYER` conditionally increases leader participation and reduces cash drag.",
            "5. `RISK_BACKOFF_LAYER_V2` clips the system when the regime deteriorates.",
            "6. continuation remains a quality filter that modulates participation but does not define the thesis.",
            "7. the promoted official freeze applies the accepted robust multipliers `B1.05_C1.10_L1.10_R1.05` over the frozen 14.3R architecture.",
        ]
    )


def _robustness_and_selection_md() -> str:
    return "\n".join(
        [
            "# Robustness And Selection",
            "",
            "- Local stability: the selected official candidate comes from a plateau, not a single narrow point.",
            "- Leave-one-window-out: the robust-main candidate remains selected when each priority window is excluded in turn.",
            "- Bootstrap: stationary block bootstrap keeps median and interquartile performance above control.",
            "- Conservative selection: promotion documents a family-level reality-check style guard instead of claiming global optimality.",
            "- Stress sensitivity: costs, slippage and parameter-neighborhood shocks were inspected before promotion.",
        ]
    )


def _overfitting_risk_md() -> str:
    return "\n".join(
        [
            "# Overfitting Risk Notes",
            "",
            "- The baseline is defendable, not proven immune to overfitting.",
            "- Prior iteration history exists, so promotion relies on conservative local selection and explicit governance.",
            "- Diagnostics are strongest for relative robustness and auditability, not for universal future guarantees.",
            "- Some implementation diagnostics inherit architecture-level traces from the promoted 14.3R family rather than a fresh discovery cycle; this is intentional to avoid re-opening tuning.",
        ]
    )


def _model_card_md(stitched_df: pd.DataFrame, cfg: Mahoraga14Config) -> str:
    official = stitched_df[stitched_df["Variant"] == OFFICIAL_VARIANT_LABEL].iloc[0]
    return "\n".join(
        [
            "# Model Card",
            "",
            f"- official model: `{OFFICIAL_VARIANT_LABEL}`",
            f"- frozen reference: `{cfg.official_reference_version}`",
            "- architecture: BASE_ALPHA_V2 + PARTICIPATION_ALLOCATOR_V2 + CONVICTION_AMPLIFIER_LAYER + LEADER_PARTICIPATION_LAYER + RISK_BACKOFF_LAYER_V2 + continuation quality filter",
            f"- stitched CAGR: {float(official['CAGR']):.4f}%",
            f"- stitched Sharpe: {float(official['Sharpe']):.4f}",
            f"- stitched Sortino: {float(official['Sortino']):.4f}",
            f"- stitched MaxDD: {float(official['MaxDD']):.4f}%",
            "- intended use: institutional long-only benchmark replacement and future long-side research starting point.",
            "- excluded uses: short-side deployment, new-thesis discovery, or uncontrolled parameter search inside the official baseline package.",
        ]
    )


def _baseline_freeze_md(cfg: Mahoraga14Config) -> str:
    return "\n".join(
        [
            "# BASELINE FREEZE",
            "",
            f"- official baseline package: `{cfg.baseline_name}`",
            f"- official candidate id: `{cfg.official_candidate_id}`",
            f"- architecture frozen from: `{cfg.official_reference_version}`",
            f"- replaces: `{cfg.replaced_baseline}`",
            "- status: official long-only institutional baseline",
            "- out of scope: short sleeves, new signals, new ML families, discovery grids",
        ]
    )


def _baseline_decision_md() -> str:
    return "\n".join(
        [
            "# BASELINE DECISION",
            "",
            "- The promoted institutional baseline is `Mahoraga14_3R / ROBUST_MAIN / B1.05_C1.10_L1.10_R1.05`.",
            "- `Mahoraga14_1_LONG_ONLY_CONTROL` is retained as a historical and documentary control.",
            "- `mahoraga14_2`, `mahoraga14_3`, `mahoraga14_3R` and other branches remain research artifacts and are not official baselines.",
        ]
    )


def _repo_baseline_manifest(
    cfg: Mahoraga14Config,
    outputs_root: Path,
    audit_root: Path,
) -> Dict[str, Any]:
    return {
        "baseline_name": cfg.baseline_name,
        "official_variant_label": OFFICIAL_VARIANT_LABEL,
        "official_candidate_id": cfg.official_candidate_id,
        "official_reference_version": cfg.official_reference_version,
        "replaced_baseline": cfg.replaced_baseline,
        "official_knobs": {
            "budget_multiplier": cfg.official_budget_multiplier,
            "conviction_multiplier": cfg.official_conviction_multiplier,
            "leader_multiplier": cfg.official_leader_multiplier,
            "backoff_strength": cfg.official_backoff_strength,
        },
        "outputs_root": str(outputs_root),
        "audit_root": str(audit_root),
    }


def _write_manifest_csv(root: Path, output_root: Path) -> None:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                    "size_bytes": path.stat().st_size,
                    "section": path.relative_to(root).parts[0] if len(path.relative_to(root).parts) else "",
                }
            )
    pd.DataFrame(files).to_csv(output_root / "output_manifest.csv", index=False)


def _write_file_manifest(package_root: Path, manifest_root: Path) -> None:
    rows = []
    for path in sorted(package_root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(package_root)).replace("\\", "/"),
                    "size_bytes": path.stat().st_size,
                    "suffix": path.suffix,
                }
            )
    pd.DataFrame(rows).to_csv(manifest_root / "file_manifest.csv", index=False)


def save_official_baseline_outputs(
    wf: Dict[str, Any],
    cfg: Mahoraga14Config,
) -> Dict[str, Any]:
    roots = _ensure_dirs(cfg)
    official_obj, context, specs, candidate_objects = _official_candidate_payload(wf, cfg)
    gate_specs = _filtered_gate_specs(specs, cfg)

    stitched_df = _official_stitched_df(candidate_objects, context, gate_specs, cfg)
    fold_df = _official_fold_df(wf, candidate_objects, context, gate_specs, cfg)
    bull_df = _official_bull_window_df(candidate_objects, context, gate_specs, cfg)
    priority_df = _official_priority_df(bull_df, cfg)
    active_df = _official_active_df(candidate_objects, context, gate_specs, cfg)
    alpha_df = _official_alpha_df(candidate_objects, context, gate_specs, cfg)
    pq_df = _official_pq_df(stitched_df, candidate_objects, context, gate_specs, cfg)

    cost_slippage_df = _official_cost_slippage_df(official_obj, context, cfg)
    cost_df = cost_slippage_df[cost_slippage_df["Scenario"].isin(["BASELINE", "COST_PLUS_25", "COST_PLUS_50", "COST_PLUS_100"])].copy()
    slip_df = cost_slippage_df[cost_slippage_df["Scenario"].isin(["BASELINE", "SLIPPAGE_PLUS_5BPS"])].copy()
    turnover_df = _official_turnover_summary(official_obj, context["control"])
    exposure_df = _official_exposure_summary(official_obj, context["control"])
    rpe_df = _official_return_per_exposure(official_obj, context["control"])

    family_df, family_objects, family_windows, base_context = acc._candidate_family(wf, cfg)
    loo_df = acc._leave_one_window_out(family_df, family_objects, family_windows, base_context, cfg)
    continuation_df = acc._continuation_acceptance_table(base_context["continuation_diag"])
    bootstrap_summary_df, bootstrap_samples_df = acc._bootstrap_summary(official_obj, base_context["control"], cfg)
    model_guard_df, model_guard_md = _official_model_selection_guard(family_df, family_objects, base_context["control"], cfg)

    decision_gate_df = pg._decision_table(
        pg._stitched_metrics_df(candidate_objects, context, specs, cfg),
        pg._fold_summary_df(wf, candidate_objects, context, specs, cfg),
        pg._priority_window_df(pg._bull_window_scorecard_df(candidate_objects, context, specs, cfg), cfg),
        specs,
    )

    continuation_diag_df = d14._build_continuation_diagnostic(wf, cfg)
    arch_scorecard = d14._build_bull_window_scorecard(wf, cfg)
    upside_df = _official_architecture_diag(d14._build_upside_participation_decomposition(wf, cfg, arch_scorecard), cfg)
    leader_df = _official_architecture_diag(d14._build_leader_miss_analysis(wf, cfg, arch_scorecard), cfg)
    allocator_cash_drag_df = _official_allocator_cash_drag(wf, cfg)
    acceptance_robustness_df = cost_slippage_df.copy()

    stitched_df.to_csv(roots["outputs"] / "stitched_comparison_official.csv", index=False)
    fold_df.to_csv(roots["outputs"] / "fold_summary_official.csv", index=False)
    bull_df.to_csv(roots["outputs"] / "bull_window_scorecard_official.csv", index=False)
    priority_df.to_csv(roots["outputs"] / "priority_window_acceptance_official.csv", index=False)
    active_df.to_csv(roots["outputs"] / "active_return_vs_qqq_official.csv", index=False)
    alpha_df.to_csv(roots["outputs"] / "alpha_nw_official.csv", index=False)
    pq_df.to_csv(roots["outputs"] / "pvalue_qvalue_official.csv", index=False)
    turnover_df.to_csv(roots["outputs"] / "turnover_summary_official.csv", index=False)
    exposure_df.to_csv(roots["outputs"] / "exposure_summary_official.csv", index=False)
    rpe_df.to_csv(roots["outputs"] / "return_per_exposure_official.csv", index=False)
    cost_df.to_csv(roots["outputs"] / "cost_sensitivity_official.csv", index=False)
    slip_df.to_csv(roots["outputs"] / "slippage_sensitivity_official.csv", index=False)

    continuation_diag_df.to_csv(roots["audit"] / "continuation_diagnostic_official.csv", index=False)
    upside_df.to_csv(roots["audit"] / "upside_participation_decomposition_official.csv", index=False)
    leader_df.to_csv(roots["audit"] / "leader_miss_analysis_official.csv", index=False)
    allocator_cash_drag_df.to_csv(roots["audit"] / "allocator_cash_drag_official.csv", index=False)
    acceptance_robustness_df.to_csv(roots["audit"] / "acceptance_robustness_summary_official.csv", index=False)
    bootstrap_summary_df.to_csv(roots["audit"] / "bootstrap_summary_official.csv", index=False)

    (roots["audit"] / "acceptance_decision_official.md").write_text(_acceptance_decision_md(decision_gate_df, priority_df, cfg), encoding="utf-8")
    (roots["audit"] / "local_stability_summary_official.md").write_text(_official_local_stability_md(family_df, cfg), encoding="utf-8")
    (roots["audit"] / "leave_one_window_out_summary_official.md").write_text(_official_loo_md(loo_df, cfg), encoding="utf-8")
    (roots["audit"] / "continuation_acceptance_official.md").write_text(acc._continuation_acceptance_notes_md(continuation_df), encoding="utf-8")
    (roots["audit"] / "model_selection_guard_official.md").write_text(model_guard_md, encoding="utf-8")

    (roots["docs"] / "BASELINE_FREEZE.md").write_text(_baseline_freeze_md(cfg), encoding="utf-8")
    (roots["docs"] / "BASELINE_DECISION.md").write_text(_baseline_decision_md(), encoding="utf-8")
    (roots["docs"] / "MODEL_CARD.md").write_text(_model_card_md(stitched_df, cfg), encoding="utf-8")
    (roots["docs"] / "DECISION_FLOW.md").write_text(_decision_flow_md(), encoding="utf-8")
    (roots["docs"] / "COMPONENT_AUDIT.md").write_text(_component_audit_md(), encoding="utf-8")
    (roots["docs"] / "MODULE_INTERFACE_MAP.md").write_text(_module_interface_map_md(), encoding="utf-8")
    (roots["docs"] / "ROBUSTNESS_AND_SELECTION.md").write_text(_robustness_and_selection_md(), encoding="utf-8")
    (roots["docs"] / "OVERFITTING_RISK_NOTES.md").write_text(_overfitting_risk_md(), encoding="utf-8")

    parameter_freeze = pd.DataFrame(
        [
            {"parameter": "official_candidate_id", "value": cfg.official_candidate_id, "role": "identity"},
            {"parameter": "budget_multiplier", "value": cfg.official_budget_multiplier, "role": "official_knob"},
            {"parameter": "conviction_multiplier", "value": cfg.official_conviction_multiplier, "role": "official_knob"},
            {"parameter": "leader_multiplier", "value": cfg.official_leader_multiplier, "role": "official_knob"},
            {"parameter": "backoff_strength", "value": cfg.official_backoff_strength, "role": "official_knob"},
            {"parameter": "primary_variant_key", "value": cfg.primary_variant_key, "role": "frozen_architecture"},
            {"parameter": "control_variant_key", "value": cfg.control_variant_key, "role": "historical_control"},
        ]
    )
    parameter_freeze.to_csv(Path(cfg.config_dir) / "PARAMETER_FREEZE.csv", index=False)

    summary_stats_df = _summary_stats_for_paper(stitched_df, priority_df, alpha_df)
    summary_stats_df.to_csv(roots["paper_pack"] / "summary_stats_for_paper.csv", index=False)
    _table_manifest().to_csv(roots["paper_pack"] / "table_manifest.csv", index=False)
    _figure_manifest().to_csv(roots["paper_pack"] / "figure_manifest.csv", index=False)
    (roots["paper_pack"] / "references_needed.md").write_text(_references_needed_md(), encoding="utf-8")
    (roots["paper_pack"] / "claims_supported_by_outputs.md").write_text(_claims_supported_md(), encoding="utf-8")

    _plot_equity_curve(active_df, roots["outputs"] / "equity_curve_official.png")
    _plot_active_return(active_df, roots["outputs"] / "active_return_vs_qqq_official.png")
    _plot_bull_windows(priority_df, roots["outputs"] / "bull_window_scorecard_official.png")
    _plot_fold_heatmap(fold_df, roots["outputs"] / "fold_heatmap_official.png")
    acc._plot_robustness_distribution(bootstrap_samples_df, roots["outputs"] / "robustness_distribution_official.png")
    acc._plot_local_stability_heatmap(family_df, roots["outputs"] / "local_stability_heatmap_official.png")

    baseline_manifest = _repo_baseline_manifest(cfg, roots["outputs"], roots["audit"])
    (roots["manifests"] / "baseline_manifest.json").write_text(json.dumps(baseline_manifest, indent=2), encoding="utf-8")
    _write_manifest_csv(roots["outputs"], roots["manifests"])
    _write_file_manifest(Path(cfg.docs_dir).parent, roots["manifests"])
    (roots["manifests"] / "provenance_notes.md").write_text(
        "\n".join(
            [
                "# Provenance Notes",
                "",
                "- Official package created from the promoted Mahoraga14_3R robust-main freeze.",
                "- Historical source lineage: `final_model/mahoraga14_3R_promotion_gate` -> `baseline/mahoraga14_3_baseline`.",
                "- Historical control retained for documentation: `Mahoraga14_1_LONG_ONLY_CONTROL`.",
                "- Research branches remain archived under `research/` and are not runtime dependencies of the official baseline.",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "stitched_df": stitched_df,
        "fold_df": fold_df,
        "bull_df": bull_df,
        "priority_df": priority_df,
        "alpha_df": alpha_df,
        "pq_df": pq_df,
        "decision_df": decision_gate_df,
        "family_df": family_df,
        "bootstrap_summary_df": bootstrap_summary_df,
    }
