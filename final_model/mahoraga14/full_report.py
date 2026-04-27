from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from fast_report import (
    _build_ablation_fast_df,
    _build_continuation_usage_fast,
    _build_floor_ceiling_summary_fast,
    _build_override_usage_fast,
    _build_pvalue_qvalue_fast,
    _build_stitched_comparison_fast,
    _build_stitched_test_only_fast,
)
from mahoraga14_config import Mahoraga14Config
from mahoraga14_utils import ensure_dir


def build_full_report_text(wf: Dict[str, Any], cfg: Mahoraga14Config) -> str:
    comparison_df = _build_stitched_test_only_fast(wf, cfg)
    fold_df = wf["fold_df"].copy().sort_values("fold")
    floor_ceiling_df = _build_floor_ceiling_summary_fast(wf, cfg)
    ablation_df = _build_ablation_fast_df(wf, cfg)
    override_df = _build_override_usage_fast(wf, cfg)
    continuation_df = _build_continuation_usage_fast(wf, cfg)
    pq_df = _build_pvalue_qvalue_fast(wf, cfg)

    lines = [
        "MAHORAGA 13 — FULL REPORT",
        "=" * 78,
        "",
        f"OFFICIAL BASELINE: {cfg.official_baseline_label}",
        f"HISTORICAL BENCHMARK: {cfg.historical_benchmark_label}",
        f"MAIN BRANCH: {cfg.main_variant_key}",
        f"EXPERIMENTAL BRANCHES: {cfg.continuation_variant_key}, {cfg.combo_variant_key}",
        "",
        "STITCHED TEST-ONLY",
        comparison_df.to_string(index=False),
        "",
        "P-VALUE / Q-VALUE",
        pq_df.to_string(index=False),
        "",
        "FLOOR / CEILING SUMMARY",
        floor_ceiling_df.to_string(index=False),
        "",
        "OVERRIDE USAGE",
        override_df.to_string(index=False),
        "",
        "CONTINUATION USAGE",
        continuation_df.to_string(index=False),
        "",
        "ABLATION",
        ablation_df.to_string(index=False),
        "",
        "FOLD SUMMARY",
        fold_df.to_string(index=False),
    ]
    return "\n".join(lines)


def save_full_outputs(wf: Dict[str, Any], cfg: Mahoraga14Config, ff=None) -> Dict[str, pd.DataFrame]:
    ensure_dir(cfg.outputs_dir)
    comparison_df = _build_stitched_comparison_fast(wf, cfg)
    stitched_test_only_df = _build_stitched_test_only_fast(wf, cfg)
    fold_df = wf["fold_df"].copy().sort_values("fold")
    floor_ceiling_df = _build_floor_ceiling_summary_fast(wf, cfg)
    ablation_df = _build_ablation_fast_df(wf, cfg)
    override_df = _build_override_usage_fast(wf, cfg)
    continuation_df = _build_continuation_usage_fast(wf, cfg)
    pq_df = _build_pvalue_qvalue_fast(wf, cfg)
    selected_df = wf.get("selected_df", pd.DataFrame()).copy()
    support_df = wf.get("support_df", pd.DataFrame()).copy()

    comparison_df.to_csv(f"{cfg.outputs_dir}/stitched_comparison_full.csv", index=False)
    stitched_test_only_df.to_csv(f"{cfg.outputs_dir}/stitched_test_only_full.csv", index=False)
    fold_df.to_csv(f"{cfg.outputs_dir}/fold_summary_full.csv", index=False)
    floor_ceiling_df.to_csv(f"{cfg.outputs_dir}/floor_ceiling_summary_full.csv", index=False)
    ablation_df.to_csv(f"{cfg.outputs_dir}/ablation_full.csv", index=False)
    override_df.to_csv(f"{cfg.outputs_dir}/override_usage_full.csv", index=False)
    continuation_df.to_csv(f"{cfg.outputs_dir}/continuation_usage_full.csv", index=False)
    selected_df.to_csv(f"{cfg.outputs_dir}/selected_candidates_full.csv", index=False)
    support_df.to_csv(f"{cfg.outputs_dir}/selected_config_support_full.csv", index=False)
    pq_df.to_csv(f"{cfg.outputs_dir}/pvalue_qvalue_full.csv", index=False)

    with open(f"{cfg.outputs_dir}/final_report_full.txt", "w", encoding="utf-8") as f:
        f.write(build_full_report_text(wf, cfg))

    return {
        "comparison": comparison_df,
        "stitched_test_only": stitched_test_only_df,
        "fold_df": fold_df,
        "floor_ceiling": floor_ceiling_df,
        "ablation": ablation_df,
        "override_usage": override_df,
        "continuation_usage": continuation_df,
        "selected_df": selected_df,
        "support_df": support_df,
        "pvalue_qvalue": pq_df,
    }
