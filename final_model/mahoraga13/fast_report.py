from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

import mahoraga6_1 as m6
from mahoraga13_config import Mahoraga13Config
from mahoraga13_utils import bhy_qvalues, ensure_dir, paired_ttest_pvalue


def _variant_keys(cfg: Mahoraga13Config) -> List[str]:
    return [
        cfg.official_baseline_label,
        cfg.main_variant_key,
        cfg.continuation_variant_key,
        cfg.combo_variant_key,
    ]


def _variant_label_map(cfg: Mahoraga13Config) -> Dict[str, str]:
    return {
        cfg.official_baseline_label: "BASE_ALPHA",
        cfg.main_variant_key: "BASE_ALPHA + STRUCTURAL_DEFENSE_ONLY",
        cfg.continuation_variant_key: "BASE_ALPHA + CONTINUATION_V2_ONLY",
        cfg.combo_variant_key: "BASE_ALPHA + STRUCTURAL_DEFENSE + CONTINUATION_V2",
        cfg.historical_benchmark_label: "LEGACY",
    }


def _summary_from_stitched(stitched: Dict[str, Any], cfg: Mahoraga13Config, label: str) -> Dict[str, float]:
    return m6.summarize(stitched["returns"], stitched["equity"], stitched["exposure"], stitched["turnover"], cfg, label)


def _summary_from_window(bt: Dict[str, Any], start: pd.Timestamp, end: pd.Timestamp, cfg: Mahoraga13Config, label: str) -> Dict[str, float]:
    returns = bt["returns_net"].loc[start:end]
    equity = cfg.capital_initial * (1.0 + returns).cumprod()
    exposure = bt["exposure"].loc[start:end]
    turnover = bt["turnover"].loc[start:end]
    return m6.summarize(returns, equity, exposure, turnover, cfg, label)


def _build_pvalue_qvalue_fast(wf: Dict[str, Any], cfg: Mahoraga13Config) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    labels = _variant_label_map(cfg)

    for result in wf["results"]:
        fold = int(result["fold"])
        start = pd.Timestamp(result["test_start"])
        end = pd.Timestamp(result["test_end"])
        base_r = result["variant_bts"][cfg.official_baseline_label]["returns_net"].loc[start:end]
        legacy_r = result["legacy_bt"]["returns_net"].loc[start:end]
        rows.append(
            {
                "Segment": f"FOLD_{fold}",
                "Fold": fold,
                "Comparison": "BASE_ALPHA_vs_LEGACY",
                "Reference": labels[cfg.historical_benchmark_label],
                "Target": labels[cfg.official_baseline_label],
                "p_value": paired_ttest_pvalue(base_r - legacy_r, alternative="greater"),
            }
        )
        for variant in [cfg.main_variant_key, cfg.continuation_variant_key, cfg.combo_variant_key]:
            variant_r = result["variant_bts"][variant]["returns_net"].loc[start:end]
            rows.append(
                {
                    "Segment": f"FOLD_{fold}",
                    "Fold": fold,
                    "Comparison": f"{variant}_vs_BASE_ALPHA",
                    "Reference": labels[cfg.official_baseline_label],
                    "Target": labels[variant],
                    "p_value": paired_ttest_pvalue(variant_r - base_r, alternative="greater"),
                }
            )

    base_stitched = wf["stitched_base"]["returns"]
    legacy_stitched = wf["stitched_legacy"]["returns"]
    rows.append(
        {
            "Segment": "STITCHED",
            "Fold": 0,
            "Comparison": "BASE_ALPHA_vs_LEGACY",
            "Reference": labels[cfg.historical_benchmark_label],
            "Target": labels[cfg.official_baseline_label],
            "p_value": paired_ttest_pvalue(base_stitched - legacy_stitched, alternative="greater"),
        }
    )
    for variant in [cfg.main_variant_key, cfg.continuation_variant_key, cfg.combo_variant_key]:
        rows.append(
            {
                "Segment": "STITCHED",
                "Fold": 0,
                "Comparison": f"{variant}_vs_BASE_ALPHA",
                "Reference": labels[cfg.official_baseline_label],
                "Target": labels[variant],
                "p_value": paired_ttest_pvalue(
                    wf["stitched_variants"][variant]["returns"] - base_stitched,
                    alternative="greater",
                ),
            }
        )

    df = pd.DataFrame(rows)
    df["q_value"] = bhy_qvalues(df["p_value"].values, alpha=cfg.bhy_alpha) if len(df) else []
    df["p_value"] = df["p_value"].round(6)
    df["q_value"] = df["q_value"].round(6)
    return df


def _build_ablation_fast_df(wf: Dict[str, Any], cfg: Mahoraga13Config) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    labels = _variant_label_map(cfg)
    pq = _build_pvalue_qvalue_fast(wf, cfg)
    pq_map = {
        (row["Segment"], row["Comparison"]): (float(row["p_value"]), float(row["q_value"]))
        for _, row in pq.iterrows()
    }

    for result in wf["results"]:
        fold = int(result["fold"])
        start = pd.Timestamp(result["test_start"])
        end = pd.Timestamp(result["test_end"])
        segment = f"FOLD_{fold}"
        for variant in _variant_keys(cfg):
            bt = result["variant_bts"][variant]
            summary = _summary_from_window(bt, start, end, cfg, variant)
            p_val, q_val = (1.0, 1.0) if variant == cfg.official_baseline_label else pq_map[(segment, f"{variant}_vs_BASE_ALPHA")]
            rows.append(
                {
                    "Segment": segment,
                    "Fold": fold,
                    "Variant": labels[variant],
                    "CAGR%": round(summary["CAGR"] * 100.0, 2),
                    "Sharpe": round(summary["Sharpe"], 4),
                    "MaxDD%": round(summary["MaxDD"] * 100.0, 2),
                    "AvgExposure": round(float(bt["exposure"].loc[start:end].mean()), 4),
                    "AvgTurnover": round(float(bt["turnover"].loc[start:end].mean()), 4),
                    "PValue_vs_BASE_ALPHA": round(p_val, 6),
                    "QValue_vs_BASE_ALPHA": round(q_val, 6),
                }
            )

    for variant in _variant_keys(cfg):
        stitched = wf["stitched_variants"][variant]
        summary = _summary_from_stitched(stitched, cfg, variant)
        p_val, q_val = (1.0, 1.0) if variant == cfg.official_baseline_label else pq_map[("STITCHED", f"{variant}_vs_BASE_ALPHA")]
        rows.append(
            {
                "Segment": "STITCHED",
                "Fold": 0,
                "Variant": labels[variant],
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


def _build_override_usage_fast(wf: Dict[str, Any], cfg: Mahoraga13Config) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    labels = _variant_label_map(cfg)

    def summarize(frame: pd.DataFrame, segment: str, fold: int, variant: str) -> Dict[str, Any]:
        if len(frame) == 0:
            return {
                "Segment": segment,
                "Fold": fold,
                "Variant": labels[variant],
                "OverrideRate": 0.0,
                "StructuralRate": 0.0,
                "ContinuationV2Rate": 0.0,
                "MeanDefenseBlend": 0.0,
                "MeanGate": 1.0,
                "MeanVolMult": 1.0,
                "MeanExpCap": 1.0,
            }
        return {
            "Segment": segment,
            "Fold": fold,
            "Variant": labels[variant],
            "OverrideRate": round(float(frame["is_override"].mean()), 4),
            "StructuralRate": round(float(frame["is_structural_override"].mean()), 4),
            "ContinuationV2Rate": round(float(frame["is_continuation_v2"].mean()), 4),
            "MeanDefenseBlend": round(float(frame["defense_blend"].mean()), 4),
            "MeanGate": round(float(frame["gate_scale"].mean()), 4),
            "MeanVolMult": round(float(frame["vol_mult"].mean()), 4),
            "MeanExpCap": round(float(frame["exp_cap"].mean()), 4),
        }

    for result in wf["results"]:
        fold = int(result["fold"])
        rows.append(summarize(pd.DataFrame(), f"FOLD_{fold}", fold, cfg.official_baseline_label))
        for variant in [cfg.main_variant_key, cfg.continuation_variant_key, cfg.combo_variant_key]:
            frame = result["variant_runs"][variant]["override_daily"].loc[result["test_start"]:result["test_end"]]
            rows.append(summarize(frame, f"FOLD_{fold}", fold, variant))

    rows.append(summarize(pd.DataFrame(), "STITCHED", 0, cfg.official_baseline_label))
    for variant in [cfg.main_variant_key, cfg.continuation_variant_key, cfg.combo_variant_key]:
        rows.append(summarize(wf["stitched_override_daily"][variant], "STITCHED", 0, variant))
    return pd.DataFrame(rows)


def _build_floor_ceiling_summary_fast(wf: Dict[str, Any], cfg: Mahoraga13Config) -> pd.DataFrame:
    fold_df = wf["fold_df"].copy().sort_values("fold")
    labels = _variant_label_map(cfg)
    sharpe_cols = {
        cfg.historical_benchmark_label: "LEGACY_Sharpe",
        cfg.official_baseline_label: "BASE_Sharpe",
        cfg.main_variant_key: "MAIN_Sharpe",
        cfg.continuation_variant_key: "CONT_V2_Sharpe",
        cfg.combo_variant_key: "COMBO_Sharpe",
    }
    stitched_map = {
        cfg.historical_benchmark_label: wf["stitched_legacy"],
        **wf["stitched_variants"],
    }
    base_ceiling = float(fold_df.loc[fold_df["fold"].isin(cfg.ceiling_folds), "BASE_Sharpe"].mean()) if len(fold_df) else 0.0
    base_floor = float(fold_df.loc[fold_df["fold"].isin(cfg.floor_folds), "BASE_Sharpe"].mean()) if len(fold_df) else 0.0
    fold5_base = float(fold_df.loc[fold_df["fold"] == 5, "BASE_Sharpe"].iloc[0]) if (fold_df["fold"] == 5).any() else 0.0
    rows: List[Dict[str, Any]] = []

    for variant, col in sharpe_cols.items():
        ceiling_mean = float(fold_df.loc[fold_df["fold"].isin(cfg.ceiling_folds), col].mean()) if len(fold_df) else 0.0
        floor_mean = float(fold_df.loc[fold_df["fold"].isin(cfg.floor_folds), col].mean()) if len(fold_df) else 0.0
        fold5 = float(fold_df.loc[fold_df["fold"] == 5, col].iloc[0]) if (fold_df["fold"] == 5).any() else 0.0
        stitched_sharpe = _summary_from_stitched(stitched_map[variant], cfg, variant)["Sharpe"]
        rows.append(
            {
                "Variant": labels[variant],
                "CeilingMeanSharpe": round(ceiling_mean, 4),
                "CeilingDeltaVsBase": round(ceiling_mean - base_ceiling, 4),
                "FloorMeanSharpe": round(floor_mean, 4),
                "FloorDeltaVsBase": round(floor_mean - base_floor, 4),
                "Fold5Sharpe": round(fold5, 4),
                "Fold5DeltaVsBase": round(fold5 - fold5_base, 4),
                "StitchedSharpe": round(stitched_sharpe, 4),
            }
        )
    return pd.DataFrame(rows)


def _build_stitched_comparison_fast(wf: Dict[str, Any], cfg: Mahoraga13Config) -> pd.DataFrame:
    labels = _variant_label_map(cfg)
    rows = []
    stitched_map = {
        cfg.historical_benchmark_label: wf["stitched_legacy"],
        **wf["stitched_variants"],
    }
    order = [
        cfg.historical_benchmark_label,
        cfg.official_baseline_label,
        cfg.main_variant_key,
        cfg.continuation_variant_key,
        cfg.combo_variant_key,
    ]
    for variant in order:
        summary = _summary_from_stitched(stitched_map[variant], cfg, variant)
        rows.append(
            {
                "Variant": labels[variant],
                "CAGR%": round(summary["CAGR"] * 100.0, 2),
                "Sharpe": round(summary["Sharpe"], 4),
                "MaxDD%": round(summary["MaxDD"] * 100.0, 2),
                "AvgExposure": round(float(stitched_map[variant]["exposure"].mean()), 4),
                "AvgTurnover": round(float(stitched_map[variant]["turnover"].mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def build_fast_report_text(wf: Dict[str, Any], cfg: Mahoraga13Config) -> str:
    comparison_df = _build_stitched_comparison_fast(wf, cfg)
    fold_df = wf["fold_df"].copy().sort_values("fold")
    floor_ceiling_df = _build_floor_ceiling_summary_fast(wf, cfg)
    ablation_df = _build_ablation_fast_df(wf, cfg)
    override_df = _build_override_usage_fast(wf, cfg)
    pq_df = _build_pvalue_qvalue_fast(wf, cfg)
    stitched_pq = pq_df[pq_df["Segment"] == "STITCHED"].copy()
    stitched_override = override_df[override_df["Segment"] == "STITCHED"].copy()
    trace_df = wf["stitched_test_trace"].copy()

    lines = [
        "MAHORAGA 13 — FAST REPORT",
        "=" * 78,
        "",
        f"OFFICIAL BASELINE: {cfg.official_baseline_label}",
        f"HISTORICAL BENCHMARK: {cfg.historical_benchmark_label}",
        f"MAIN BRANCH: {cfg.main_variant_key}",
        f"EXPERIMENTAL BRANCHES: {cfg.continuation_variant_key}, {cfg.combo_variant_key}",
        "",
        "STITCHED COMPARISON",
        comparison_df.to_string(index=False),
        "",
        "STITCHED P-VALUE / Q-VALUE",
        stitched_pq.to_string(index=False),
        "",
        "FLOOR / CEILING SUMMARY",
        floor_ceiling_df.to_string(index=False),
        "",
        "STITCHED OVERRIDE USAGE",
        stitched_override.to_string(index=False),
        "",
        "ABLATION (FOLDS + STITCHED)",
        ablation_df.to_string(index=False),
        "",
        "FOLD SUMMARY",
        fold_df.to_string(index=False),
        "",
        "STITCHED TEST WINDOW TRACE",
        trace_df.to_string(index=False),
    ]
    return "\n".join(lines)


def save_fast_outputs(wf: Dict[str, Any], cfg: Mahoraga13Config) -> Dict[str, pd.DataFrame]:
    ensure_dir(cfg.outputs_dir)
    comparison_df = _build_stitched_comparison_fast(wf, cfg)
    fold_df = wf["fold_df"].copy().sort_values("fold")
    floor_ceiling_df = _build_floor_ceiling_summary_fast(wf, cfg)
    ablation_df = _build_ablation_fast_df(wf, cfg)
    override_df = _build_override_usage_fast(wf, cfg)
    pq_df = _build_pvalue_qvalue_fast(wf, cfg)
    selected_df = wf.get("selected_df", pd.DataFrame()).copy()
    support_df = wf.get("support_df", pd.DataFrame()).copy()

    comparison_df.to_csv(f"{cfg.outputs_dir}/stitched_comparison_fast.csv", index=False)
    fold_df.to_csv(f"{cfg.outputs_dir}/fold_summary_fast.csv", index=False)
    floor_ceiling_df.to_csv(f"{cfg.outputs_dir}/floor_ceiling_summary_fast.csv", index=False)
    ablation_df.to_csv(f"{cfg.outputs_dir}/ablation_fast.csv", index=False)
    override_df.to_csv(f"{cfg.outputs_dir}/override_usage_fast.csv", index=False)
    selected_df.to_csv(f"{cfg.outputs_dir}/selected_candidates_fast.csv", index=False)
    support_df.to_csv(f"{cfg.outputs_dir}/selected_config_support_fast.csv", index=False)
    pq_df.to_csv(f"{cfg.outputs_dir}/pvalue_qvalue_fast.csv", index=False)

    with open(f"{cfg.outputs_dir}/final_report_fast.txt", "w", encoding="utf-8") as f:
        f.write(build_fast_report_text(wf, cfg))

    return {
        "comparison": comparison_df,
        "fold_df": fold_df,
        "floor_ceiling": floor_ceiling_df,
        "ablation": ablation_df,
        "override_usage": override_df,
        "selected_df": selected_df,
        "support_df": support_df,
        "pvalue_qvalue": pq_df,
    }
