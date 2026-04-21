from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

import mahoraga6_1 as m6
from mahoraga12_config import Mahoraga12Config
from mahoraga12_utils import bhy_qvalues, ensure_dir, paired_ttest_pvalue


ABLATION_VARIANTS = [
    "BASE_ALPHA",
    "STRUCTURAL_DEFENSE_ONLY",
    "TRANSITION_ONLY",
    "FULL_OVERRIDES",
]

ABLATION_LABELS = {
    "BASE_ALPHA": "BASE_ALPHA",
    "STRUCTURAL_DEFENSE_ONLY": "BASE_ALPHA + STRUCTURAL_DEFENSE_ONLY",
    "TRANSITION_ONLY": "BASE_ALPHA + TRANSITION_ONLY",
    "FULL_OVERRIDES": "BASE_ALPHA + FULL_OVERRIDES",
}


def _summary_from_stitched(stitched: Dict[str, Any], cfg: Mahoraga12Config, label: str) -> Dict[str, float]:
    return m6.summarize(stitched["returns"], stitched["equity"], stitched["exposure"], stitched["turnover"], cfg, label)


def _summary_from_window(bt: Dict[str, Any], start: pd.Timestamp, end: pd.Timestamp, cfg: Mahoraga12Config, label: str) -> Dict[str, float]:
    returns = bt["returns_net"].loc[start:end]
    equity = cfg.capital_initial * (1.0 + returns).cumprod()
    exposure = bt["exposure"].loc[start:end]
    turnover = bt["turnover"].loc[start:end]
    return m6.summarize(returns, equity, exposure, turnover, cfg, label)


def _build_primary_tests(wf: Dict[str, Any], cfg: Mahoraga12Config) -> pd.DataFrame:
    base_r = wf["stitched_base"]["returns"]
    model_r = wf["stitched_m12"]["returns"]
    legacy_r = wf["stitched_legacy"]["returns"]
    rows = [
        {"Comparison": "BASE_ALPHA_vs_LEGACY", "p_value": paired_ttest_pvalue(base_r - legacy_r, alternative="greater")},
        {"Comparison": "M12_vs_BASE_ALPHA", "p_value": paired_ttest_pvalue(model_r - base_r, alternative="greater")},
        {"Comparison": "M12_vs_LEGACY", "p_value": paired_ttest_pvalue(model_r - legacy_r, alternative="greater")},
    ]
    tests = pd.DataFrame(rows)
    tests["q_value"] = bhy_qvalues(tests["p_value"].values, alpha=cfg.bhy_alpha)
    return tests


def _build_ablation_fast_df(wf: Dict[str, Any], cfg: Mahoraga12Config) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for result in wf["results"]:
        fold = int(result["fold"])
        start = pd.Timestamp(result["test_start"])
        end = pd.Timestamp(result["test_end"])
        base_r = result["variant_bts"]["BASE_ALPHA"]["returns_net"].loc[start:end]
        pvals = []
        for variant in ABLATION_VARIANTS[1:]:
            diff = result["variant_bts"][variant]["returns_net"].loc[start:end] - base_r
            pvals.append(paired_ttest_pvalue(diff, alternative="greater"))
        qvals = bhy_qvalues(pvals, alpha=cfg.bhy_alpha)
        pq_map = {"BASE_ALPHA": (1.0, 1.0)}
        for variant, p_val, q_val in zip(ABLATION_VARIANTS[1:], pvals, qvals):
            pq_map[variant] = (float(p_val), float(q_val))

        for variant in ABLATION_VARIANTS:
            bt = result["variant_bts"][variant]
            summary = _summary_from_window(bt, start, end, cfg, f"{variant}_FOLD_{fold}")
            p_val, q_val = pq_map[variant]
            rows.append(
                {
                    "Segment": f"FOLD_{fold}",
                    "Fold": fold,
                    "Variant": ABLATION_LABELS[variant],
                    "CAGR%": round(summary["CAGR"] * 100.0, 2),
                    "Sharpe": round(summary["Sharpe"], 4),
                    "MaxDD%": round(summary["MaxDD"] * 100.0, 2),
                    "AvgExposure": round(float(bt["exposure"].loc[start:end].mean()), 4),
                    "AvgTurnover": round(float(bt["turnover"].loc[start:end].mean()), 4),
                    "PValue_vs_BASE_ALPHA": round(p_val, 6),
                    "QValue_vs_BASE_ALPHA": round(q_val, 6),
                }
            )

    stitched_variants = wf["stitched_variants"]
    base_r = stitched_variants["BASE_ALPHA"]["returns"]
    pvals = []
    for variant in ABLATION_VARIANTS[1:]:
        diff = stitched_variants[variant]["returns"] - base_r
        pvals.append(paired_ttest_pvalue(diff, alternative="greater"))
    qvals = bhy_qvalues(pvals, alpha=cfg.bhy_alpha)
    pq_map = {"BASE_ALPHA": (1.0, 1.0)}
    for variant, p_val, q_val in zip(ABLATION_VARIANTS[1:], pvals, qvals):
        pq_map[variant] = (float(p_val), float(q_val))

    for variant in ABLATION_VARIANTS:
        stitched = stitched_variants[variant]
        summary = _summary_from_stitched(stitched, cfg, variant)
        p_val, q_val = pq_map[variant]
        rows.append(
            {
                "Segment": "STITCHED",
                "Fold": 0,
                "Variant": ABLATION_LABELS[variant],
                "CAGR%": round(summary["CAGR"] * 100.0, 2),
                "Sharpe": round(summary["Sharpe"], 4),
                "MaxDD%": round(summary["MaxDD"] * 100.0, 2),
                "AvgExposure": round(float(stitched["exposure"].mean()), 4),
                "AvgTurnover": round(float(stitched["turnover"].mean()), 4),
                "PValue_vs_BASE_ALPHA": round(p_val, 6),
                "QValue_vs_BASE_ALPHA": round(q_val, 6),
            }
        )

    return pd.DataFrame(rows)


def _build_continuation_compare_df(wf: Dict[str, Any], cfg: Mahoraga12Config) -> pd.DataFrame:
    segments: List[Dict[str, Any]] = []
    comparison_pvals: List[float] = []

    for result in wf["results"]:
        fold = int(result["fold"])
        start = pd.Timestamp(result["test_start"])
        end = pd.Timestamp(result["test_end"])
        current_bt = result["variant_bts"]["FULL_OVERRIDES_CURRENT"]
        official_bt = result["variant_bts"]["FULL_OVERRIDES"]
        p_val = paired_ttest_pvalue(
            official_bt["returns_net"].loc[start:end] - current_bt["returns_net"].loc[start:end],
            alternative="greater",
        )
        comparison_pvals.append(p_val)
        segments.append({"Segment": f"FOLD_{fold}", "Fold": fold, "current_bt": current_bt, "official_bt": official_bt, "p_value": p_val, "start": start, "end": end})

    stitched_current = wf["stitched_m12_current"]
    stitched_official = wf["stitched_m12"]
    stitched_p = paired_ttest_pvalue(stitched_official["returns"] - stitched_current["returns"], alternative="greater")
    comparison_pvals.append(stitched_p)
    segments.append({"Segment": "STITCHED", "Fold": 0, "current_bt": stitched_current, "official_bt": stitched_official, "p_value": stitched_p, "start": None, "end": None})

    qvals = bhy_qvalues(comparison_pvals, alpha=cfg.bhy_alpha)
    rows: List[Dict[str, Any]] = []
    for segment, q_val in zip(segments, qvals):
        if segment["Segment"] == "STITCHED":
            current_summary = _summary_from_stitched(segment["current_bt"], cfg, cfg.current_model_label)
            official_summary = _summary_from_stitched(segment["official_bt"], cfg, cfg.model_label)
        else:
            current_summary = _summary_from_window(segment["current_bt"], segment["start"], segment["end"], cfg, cfg.current_model_label)
            official_summary = _summary_from_window(segment["official_bt"], segment["start"], segment["end"], cfg, cfg.model_label)

        rows.append(
            {
                "Segment": segment["Segment"],
                "Fold": segment["Fold"],
                "Variant": cfg.current_model_label,
                "CAGR%": round(current_summary["CAGR"] * 100.0, 2),
                "Sharpe": round(current_summary["Sharpe"], 4),
                "MaxDD%": round(current_summary["MaxDD"] * 100.0, 2),
                "PValue_vs_Current": 1.0,
                "QValue_vs_Current": 1.0,
            }
        )
        rows.append(
            {
                "Segment": segment["Segment"],
                "Fold": segment["Fold"],
                "Variant": f"{cfg.model_label}_WITH_CONTINUATION",
                "CAGR%": round(official_summary["CAGR"] * 100.0, 2),
                "Sharpe": round(official_summary["Sharpe"], 4),
                "MaxDD%": round(official_summary["MaxDD"] * 100.0, 2),
                "PValue_vs_Current": round(float(segment["p_value"]), 6),
                "QValue_vs_Current": round(float(q_val), 6),
            }
        )
    return pd.DataFrame(rows)


def build_fast_report_text(wf: Dict[str, Any], cfg: Mahoraga12Config) -> str:
    fold_df = wf["fold_df"].copy().sort_values("fold")
    tests_df = _build_primary_tests(wf, cfg)
    ablation_df = _build_ablation_fast_df(wf, cfg)
    continuation_df = _build_continuation_compare_df(wf, cfg)
    trace_df = wf["stitched_test_trace"].copy()

    s_base = _summary_from_stitched(wf["stitched_base"], cfg, cfg.official_baseline_label)
    s_model = _summary_from_stitched(wf["stitched_m12"], cfg, cfg.model_label)
    s_legacy = _summary_from_stitched(wf["stitched_legacy"], cfg, cfg.historical_benchmark_label)

    floor_mean = float(fold_df.loc[fold_df["fold"].isin(cfg.floor_folds), "M12_Sharpe"].mean()) if len(fold_df) else 0.0
    ceil_mean = float(fold_df.loc[fold_df["fold"].isin(cfg.ceiling_folds), "M12_Sharpe"].mean()) if len(fold_df) else 0.0

    lines = [
        "MAHORAGA 12 — FAST REPORT",
        "=" * 78,
        "",
        f"OFFICIAL BASELINE: {cfg.official_baseline_label}",
        f"HISTORICAL BENCHMARK: {cfg.historical_benchmark_label}",
        "",
        "STITCHED TEST-ONLY OOS COMPARISON",
        f"  BASE_ALPHA CAGR={s_base['CAGR']*100:.2f}%  Sharpe={s_base['Sharpe']:.3f}  MaxDD={s_base['MaxDD']*100:.2f}%",
        f"  M12        CAGR={s_model['CAGR']*100:.2f}%  Sharpe={s_model['Sharpe']:.3f}  MaxDD={s_model['MaxDD']*100:.2f}%",
        f"  LEGACY     CAGR={s_legacy['CAGR']*100:.2f}%  Sharpe={s_legacy['Sharpe']:.3f}  MaxDD={s_legacy['MaxDD']*100:.2f}%",
        "",
        "STITCHED HYPOTHESIS TESTS",
        tests_df.to_string(index=False),
        "",
        "STITCHED TEST WINDOW TRACE",
        trace_df.to_string(index=False),
        "",
        "FLOOR / CEILING SUMMARY",
        f"  Floor folds {cfg.floor_folds}: mean M12 Sharpe={floor_mean:.4f}",
        f"  Ceiling folds {cfg.ceiling_folds}: mean M12 Sharpe={ceil_mean:.4f}",
        "",
        "ABLATION (FOLDS + STITCHED)",
        ablation_df.to_string(index=False),
        "",
        "CURRENT FULL OVERRIDES VS CONTINUATION-ENHANCED MODEL",
        continuation_df.to_string(index=False),
        "",
        "FOLD SUMMARY",
        fold_df.to_string(index=False),
    ]
    return "\n".join(lines)


def save_fast_outputs(wf: Dict[str, Any], cfg: Mahoraga12Config) -> Dict[str, pd.DataFrame]:
    ensure_dir(cfg.outputs_dir)
    fold_df = wf["fold_df"].copy().sort_values("fold")
    selected_df = wf.get("selected_df", pd.DataFrame()).copy()
    support_df = wf.get("support_df", pd.DataFrame()).copy()

    comparison = pd.DataFrame(
        [
            m6._fmt(_summary_from_stitched(wf["stitched_base"], cfg, cfg.official_baseline_label)),
            m6._fmt(_summary_from_stitched(wf["stitched_m12"], cfg, cfg.model_label)),
            m6._fmt(_summary_from_stitched(wf["stitched_legacy"], cfg, cfg.historical_benchmark_label)),
        ]
    )
    comparison.to_csv(f"{cfg.outputs_dir}/stitched_comparison_fast.csv", index=False)
    fold_df.to_csv(f"{cfg.outputs_dir}/walk_forward_folds_fast.csv", index=False)
    selected_df.to_csv(f"{cfg.outputs_dir}/selected_candidates_fast.csv", index=False)
    support_df.to_csv(f"{cfg.outputs_dir}/selected_config_support_fast.csv", index=False)

    leaderboard = (
        pd.concat(
            [r["calibration_df"].assign(fold=r["fold"]) for r in wf["results"] if len(r.get("calibration_df", []))],
            axis=0,
            ignore_index=True,
        )
        if wf["results"]
        else pd.DataFrame()
    )
    leaderboard.to_csv(f"{cfg.outputs_dir}/candidate_leaderboard_fast.csv", index=False)

    override_weekly = pd.concat([r["override_weekly"] for r in wf["results"]], axis=0, ignore_index=False) if wf["results"] else pd.DataFrame()
    override_weekly.to_csv(f"{cfg.outputs_dir}/override_weekly_fast.csv", index=True)

    tests_df = _build_primary_tests(wf, cfg)
    tests_df.to_csv(f"{cfg.outputs_dir}/stitched_tests_fast.csv", index=False)

    trace_df = wf["stitched_test_trace"].copy()
    trace_df.to_csv(f"{cfg.outputs_dir}/stitched_test_trace_fast.csv", index=False)

    ablation_df = _build_ablation_fast_df(wf, cfg)
    ablation_df.to_csv(f"{cfg.outputs_dir}/ablation_fast.csv", index=False)

    continuation_df = _build_continuation_compare_df(wf, cfg)
    continuation_df.to_csv(f"{cfg.outputs_dir}/continuation_comparison_fast.csv", index=False)

    base_r = wf["stitched_base"]["returns"]
    model_r = wf["stitched_m12"]["returns"]
    legacy_r = wf["stitched_legacy"]["returns"]
    alpha_nw = pd.DataFrame(
        [
            m6.alpha_test_nw(base_r, legacy_r, cfg, label="BASE_ALPHA_vs_LEGACY", conditional=False, exposure=wf["stitched_base"]["exposure"]),
            m6.alpha_test_nw(model_r, base_r, cfg, label="M12_vs_BASE_ALPHA", conditional=False, exposure=wf["stitched_m12"]["exposure"]),
            m6.alpha_test_nw(model_r, legacy_r, cfg, label="M12_vs_LEGACY", conditional=False, exposure=wf["stitched_m12"]["exposure"]),
        ]
    )
    alpha_nw.to_csv(f"{cfg.outputs_dir}/alpha_nw_fast.csv", index=False)

    sharpe_ci = pd.DataFrame(
        [
            dict(Label=cfg.official_baseline_label, **m6.asymptotic_sharpe_ci(base_r, cfg)),
            dict(Label=cfg.model_label, **m6.asymptotic_sharpe_ci(model_r, cfg)),
            dict(Label=cfg.historical_benchmark_label, **m6.asymptotic_sharpe_ci(legacy_r, cfg)),
        ]
    )
    sharpe_ci.to_csv(f"{cfg.outputs_dir}/sharpe_ci_fast.csv", index=False)

    with open(f"{cfg.outputs_dir}/final_report_fast.txt", "w", encoding="utf-8") as f:
        f.write(build_fast_report_text(wf, cfg))

    return {
        "fold_df": fold_df,
        "selected_df": selected_df,
        "support_df": support_df,
        "comparison": comparison,
        "leaderboard": leaderboard,
        "override_weekly": override_weekly,
        "stitched_tests": tests_df,
        "stitched_trace": trace_df,
        "ablation": ablation_df,
        "continuation_compare": continuation_df,
        "alpha_nw": alpha_nw,
        "sharpe_ci": sharpe_ci,
    }
