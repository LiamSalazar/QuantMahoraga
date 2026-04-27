from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from fast_report import (
    _build_ablation_fast_df,
    _build_alpha_nw_fast,
    _build_continuation_event_study_fast,
    _build_continuation_usage_fast,
    _build_floor_ceiling_summary_fast,
    _build_fold_summary_fast,
    _build_override_usage_fast,
    _build_pvalue_qvalue_fast,
    _build_stitched_comparison_fast,
    build_fast_report_text,
)
from mahoraga14_config import Mahoraga14Config
from mahoraga14_utils import ensure_dir


def build_full_report_text(wf: Dict[str, Any], cfg: Mahoraga14Config) -> str:
    return build_fast_report_text(wf, cfg).replace("FAST REPORT", "FULL REPORT")


def save_full_outputs(wf: Dict[str, Any], cfg: Mahoraga14Config, ff=None) -> Dict[str, pd.DataFrame]:
    ensure_dir(cfg.outputs_dir)
    comparison_df = _build_stitched_comparison_fast(wf, cfg)
    fold_df = _build_fold_summary_fast(wf)
    floor_ceiling_df = _build_floor_ceiling_summary_fast(wf, cfg)
    ablation_df = _build_ablation_fast_df(wf, cfg)
    override_df = _build_override_usage_fast(wf, cfg)
    continuation_df = _build_continuation_usage_fast(wf, cfg)
    event_df = _build_continuation_event_study_fast(wf, cfg)
    selected_df = wf.get("selected_df", pd.DataFrame()).copy()
    support_df = wf.get("support_df", pd.DataFrame()).copy()
    pq_df = _build_pvalue_qvalue_fast(wf, cfg)
    alpha_nw_df = _build_alpha_nw_fast(wf, cfg)

    comparison_df.to_csv(f"{cfg.outputs_dir}/stitched_comparison_full.csv", index=False)
    fold_df.to_csv(f"{cfg.outputs_dir}/fold_summary_full.csv", index=False)
    floor_ceiling_df.to_csv(f"{cfg.outputs_dir}/floor_ceiling_summary_full.csv", index=False)
    ablation_df.to_csv(f"{cfg.outputs_dir}/ablation_full.csv", index=False)
    override_df.to_csv(f"{cfg.outputs_dir}/override_usage_full.csv", index=False)
    continuation_df.to_csv(f"{cfg.outputs_dir}/continuation_usage_full.csv", index=False)
    event_df.to_csv(f"{cfg.outputs_dir}/continuation_event_study_full.csv", index=False)
    selected_df.to_csv(f"{cfg.outputs_dir}/selected_candidates_full.csv", index=False)
    support_df.to_csv(f"{cfg.outputs_dir}/selected_config_support_full.csv", index=False)
    pq_df.to_csv(f"{cfg.outputs_dir}/pvalue_qvalue_full.csv", index=False)
    alpha_nw_df.to_csv(f"{cfg.outputs_dir}/alpha_nw_full.csv", index=False)

    with open(f"{cfg.outputs_dir}/final_report_full.txt", "w", encoding="utf-8") as f:
        f.write(build_full_report_text(wf, cfg))

    return {
        "stitched_comparison_full": comparison_df,
        "fold_summary_full": fold_df,
        "floor_ceiling_summary_full": floor_ceiling_df,
        "ablation_full": ablation_df,
        "override_usage_full": override_df,
        "continuation_usage_full": continuation_df,
        "continuation_event_study_full": event_df,
        "selected_candidates_full": selected_df,
        "selected_config_support_full": support_df,
        "pvalue_qvalue_full": pq_df,
        "alpha_nw_full": alpha_nw_df,
    }
