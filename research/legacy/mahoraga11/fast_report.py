from __future__ import annotations

from typing import Any, Dict

import pandas as pd

import mahoraga6_1 as m6
from mahoraga11_config import Mahoraga11Config
from mahoraga11_utils import ensure_dir, paired_ttest_pvalue


def build_fast_report_text(wf: Dict[str, Any], cfg: Mahoraga11Config) -> str:
    fold_df = wf["fold_df"].copy().sort_values("fold")

    legacy_r = wf["stitched_legacy"]["returns"]
    base_r = wf["stitched_base"]["returns"]
    model_r = wf["stitched_m11"]["returns"]
    legacy_eq = cfg.capital_initial * (1.0 + legacy_r).cumprod()
    base_eq = cfg.capital_initial * (1.0 + base_r).cumprod()
    model_eq = cfg.capital_initial * (1.0 + model_r).cumprod()

    s_legacy = m6.summarize(legacy_r, legacy_eq, wf["stitched_legacy"]["exposure"], wf["stitched_legacy"]["turnover"], cfg, "LEGACY")
    s_base = m6.summarize(base_r, base_eq, wf["stitched_base"]["exposure"], wf["stitched_base"]["turnover"], cfg, "BASE_ALPHA")
    s_model = m6.summarize(model_r, model_eq, wf["stitched_m11"]["exposure"], wf["stitched_m11"]["turnover"], cfg, "M11")

    floor_mean = float(fold_df.loc[fold_df["fold"].isin(cfg.floor_folds), "M11_Sharpe"].mean()) if len(fold_df) else 0.0
    ceil_mean = float(fold_df.loc[fold_df["fold"].isin(cfg.ceiling_folds), "M11_Sharpe"].mean()) if len(fold_df) else 0.0
    q_max = float(fold_df["Model_vs_Base_Test_qvalue"].max()) if "Model_vs_Base_Test_qvalue" in fold_df.columns and len(fold_df) else 1.0

    lines = [
        "MAHORAGA 11 — FAST REPORT",
        "=" * 78,
        "",
        "STITCHED OOS COMPARISON",
        f"  LEGACY    CAGR={s_legacy['CAGR']*100:.2f}%  Sharpe={s_legacy['Sharpe']:.3f}  MaxDD={s_legacy['MaxDD']*100:.2f}%",
        f"  BASE_ALPHA CAGR={s_base['CAGR']*100:.2f}%  Sharpe={s_base['Sharpe']:.3f}  MaxDD={s_base['MaxDD']*100:.2f}%",
        f"  M11       CAGR={s_model['CAGR']*100:.2f}%  Sharpe={s_model['Sharpe']:.3f}  MaxDD={s_model['MaxDD']*100:.2f}%",
        f"  Base vs Legacy p-value={paired_ttest_pvalue(base_r - legacy_r, alternative='greater'):.6f}",
        f"  M11  vs Base   p-value={paired_ttest_pvalue(model_r - base_r, alternative='greater'):.6f}",
        f"  M11  vs Legacy p-value={paired_ttest_pvalue(model_r - legacy_r, alternative='greater'):.6f}",
        f"  Fold-level BHY max q-value={q_max:.6f}",
        "",
        "FLOOR / CEILING SUMMARY",
        f"  Floor folds {cfg.floor_folds}: mean M11 Sharpe={floor_mean:.4f}",
        f"  Ceiling folds {cfg.ceiling_folds}: mean M11 Sharpe={ceil_mean:.4f}",
        "",
        "FOLD SUMMARY",
        fold_df.to_string(index=False),
    ]
    return "\n".join(lines)


def save_fast_outputs(wf: Dict[str, Any], cfg: Mahoraga11Config) -> Dict[str, pd.DataFrame]:
    ensure_dir(cfg.outputs_dir)
    fold_df = wf["fold_df"].copy().sort_values("fold")
    selected_df = wf.get("selected_df", pd.DataFrame()).copy()

    legacy_r = wf["stitched_legacy"]["returns"]
    base_r = wf["stitched_base"]["returns"]
    model_r = wf["stitched_m11"]["returns"]
    legacy_eq = cfg.capital_initial * (1.0 + legacy_r).cumprod()
    base_eq = cfg.capital_initial * (1.0 + base_r).cumprod()
    model_eq = cfg.capital_initial * (1.0 + model_r).cumprod()

    comparison = pd.DataFrame([
        m6._fmt(m6.summarize(legacy_r, legacy_eq, wf["stitched_legacy"]["exposure"], wf["stitched_legacy"]["turnover"], cfg, "LEGACY")),
        m6._fmt(m6.summarize(base_r, base_eq, wf["stitched_base"]["exposure"], wf["stitched_base"]["turnover"], cfg, "BASE_ALPHA")),
        m6._fmt(m6.summarize(model_r, model_eq, wf["stitched_m11"]["exposure"], wf["stitched_m11"]["turnover"], cfg, "M11")),
    ])
    comparison.to_csv(f"{cfg.outputs_dir}/stitched_comparison_fast.csv", index=False)
    fold_df.to_csv(f"{cfg.outputs_dir}/walk_forward_folds_fast.csv", index=False)
    selected_df.to_csv(f"{cfg.outputs_dir}/selected_candidates_fast.csv", index=False)

    leaderboard = pd.concat(
        [r["calibration_df"].assign(fold=r["fold"]) for r in wf["results"] if len(r.get("calibration_df", []))],
        axis=0,
        ignore_index=True,
    ) if wf["results"] else pd.DataFrame()
    leaderboard.to_csv(f"{cfg.outputs_dir}/candidate_leaderboard_fast.csv", index=False)

    override_weekly = pd.concat([r["override_weekly"] for r in wf["results"]], axis=0, ignore_index=False) if wf["results"] else pd.DataFrame()
    override_weekly.to_csv(f"{cfg.outputs_dir}/override_weekly_fast.csv", index=True)

    alpha_nw = pd.DataFrame([
        m6.alpha_test_nw(base_r, legacy_r, cfg, label="BASE_vs_LEGACY", conditional=False, exposure=wf["stitched_base"]["exposure"]),
        m6.alpha_test_nw(model_r, base_r, cfg, label="M11_vs_BASE", conditional=False, exposure=wf["stitched_m11"]["exposure"]),
        m6.alpha_test_nw(model_r, legacy_r, cfg, label="M11_vs_LEGACY", conditional=False, exposure=wf["stitched_m11"]["exposure"]),
    ])
    alpha_nw.to_csv(f"{cfg.outputs_dir}/alpha_nw_fast.csv", index=False)

    sharpe_ci = pd.DataFrame([
        dict(Label="LEGACY", **m6.asymptotic_sharpe_ci(legacy_r, cfg)),
        dict(Label="BASE_ALPHA", **m6.asymptotic_sharpe_ci(base_r, cfg)),
        dict(Label="M11", **m6.asymptotic_sharpe_ci(model_r, cfg)),
    ])
    sharpe_ci.to_csv(f"{cfg.outputs_dir}/sharpe_ci_fast.csv", index=False)

    with open(f"{cfg.outputs_dir}/final_report_fast.txt", "w", encoding="utf-8") as f:
        f.write(build_fast_report_text(wf, cfg))

    return {
        "fold_df": fold_df,
        "selected_df": selected_df,
        "comparison": comparison,
        "leaderboard": leaderboard,
        "override_weekly": override_weekly,
        "alpha_nw": alpha_nw,
        "sharpe_ci": sharpe_ci,
    }
