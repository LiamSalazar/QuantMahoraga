from __future__ import annotations

from typing import Any, Dict

import pandas as pd

import mahoraga6_1 as m6
from mahoraga11_config import Mahoraga11Config
from mahoraga11_utils import ensure_dir, paired_ttest_pvalue


def build_fold_summary(wf: Dict[str, Any]) -> pd.DataFrame:
    return wf["fold_df"].copy().sort_values("fold")


def build_final_report_text(wf: Dict[str, Any], cfg: Mahoraga11Config) -> str:
    fold_df = build_fold_summary(wf)
    base_r = wf["stitched_base"]["returns"]
    m11_r = wf["stitched_m11"]["returns"]
    base_eq = cfg.capital_initial * (1.0 + base_r).cumprod()
    m11_eq = cfg.capital_initial * (1.0 + m11_r).cumprod()
    s_base = m6.summarize(base_r, base_eq, wf["stitched_base"]["exposure"], wf["stitched_base"]["turnover"], cfg, "BASELINE")
    s_m11 = m6.summarize(m11_r, m11_eq, wf["stitched_m11"]["exposure"], wf["stitched_m11"]["turnover"], cfg, "M11")
    stitched_p = paired_ttest_pvalue(m11_r - base_r, alternative="greater")
    q_max = float(fold_df["Test_qvalue"].max()) if "Test_qvalue" in fold_df.columns and len(fold_df) else 1.0

    floor_mean = float(fold_df.loc[fold_df["fold"].isin(cfg.floor_folds), "M11_Sharpe"].mean()) if len(fold_df) else 0.0
    ceil_mean = float(fold_df.loc[fold_df["fold"].isin(cfg.ceiling_folds), "M11_Sharpe"].mean()) if len(fold_df) else 0.0

    lines = []
    lines.append("MAHORAGA 11 — FINAL REPORT")
    lines.append("=" * 78)
    lines.append("")
    lines.append("OOS COMPARISON")
    lines.append(f"  BASELINE CAGR={s_base['CAGR']*100:.2f}%  Sharpe={s_base['Sharpe']:.3f}  MaxDD={s_base['MaxDD']*100:.2f}%")
    lines.append(f"  M11      CAGR={s_m11['CAGR']*100:.2f}%  Sharpe={s_m11['Sharpe']:.3f}  MaxDD={s_m11['MaxDD']*100:.2f}%")
    lines.append(f"  Stitched diff p-value={stitched_p:.6f}")
    lines.append(f"  Fold-level BHY max q-value={q_max:.6f}")
    lines.append("")
    lines.append("FLOOR / CEILING SUMMARY")
    lines.append(f"  Floor folds {cfg.floor_folds}: mean M11 Sharpe={floor_mean:.4f}")
    lines.append(f"  Ceiling folds {cfg.ceiling_folds}: mean M11 Sharpe={ceil_mean:.4f}")
    lines.append("")
    lines.append("FOLD SUMMARY")
    lines.append(fold_df.to_string(index=False))
    return "
".join(lines)


def save_outputs(wf: Dict[str, Any], cfg: Mahoraga11Config, ff=None) -> Dict[str, pd.DataFrame]:
    ensure_dir(cfg.outputs_dir)
    fold_df = build_fold_summary(wf)
    base_r = wf["stitched_base"]["returns"]
    m11_r = wf["stitched_m11"]["returns"]
    base_eq = cfg.capital_initial * (1.0 + base_r).cumprod()
    m11_eq = cfg.capital_initial * (1.0 + m11_r).cumprod()

    comparison_oos = pd.DataFrame([
        m6._fmt(m6.summarize(base_r, base_eq, wf["stitched_base"]["exposure"], wf["stitched_base"]["turnover"], cfg, "BASELINE")),
        m6._fmt(m6.summarize(m11_r, m11_eq, wf["stitched_m11"]["exposure"], wf["stitched_m11"]["turnover"], cfg, "M11")),
    ])
    comparison_oos.to_csv(f"{cfg.outputs_dir}/{'comparison_oos.csv' if cfg.run_mode.upper() == 'FULL' else 'stitched_comparison_fast.csv'}", index=False)
    fold_df.to_csv(f"{cfg.outputs_dir}/{'walk_forward_folds.csv' if cfg.run_mode.upper() == 'FULL' else 'walk_forward_folds_fast.csv'}", index=False)

    leaderboard = pd.concat(
        [r["calibration_df"].assign(fold=r["fold"]) for r in wf["results"] if len(r.get("calibration_df", []))],
        axis=0,
        ignore_index=True,
    ) if wf["results"] else pd.DataFrame()
    leaderboard.to_csv(f"{cfg.outputs_dir}/{'candidate_leaderboard.csv' if cfg.run_mode.upper() == 'FULL' else 'candidate_leaderboard_fast.csv'}", index=False)

    router_art = pd.concat([r["router_weekly"] for r in wf["results"]], axis=0, ignore_index=False) if wf["results"] else pd.DataFrame()
    router_art.to_csv(f"{cfg.outputs_dir}/{'router_weekly.csv' if cfg.run_mode.upper() == 'FULL' else 'router_weekly_fast.csv'}", index=True)

    alpha_rows = []
    for lbl, ret, exp in [
        ("BASE_OOS", base_r, wf["stitched_base"]["exposure"]),
        ("M11_OOS", m11_r, wf["stitched_m11"]["exposure"]),
    ]:
        alpha_rows.append(m6.alpha_test_nw(ret, wf["stitched_base"]["returns"], cfg, label=lbl, conditional=False, exposure=exp))
    alpha_nw = pd.DataFrame(alpha_rows)
    alpha_nw.to_csv(f"{cfg.outputs_dir}/{'alpha_nw.csv' if cfg.run_mode.upper() == 'FULL' else 'alpha_nw_fast.csv'}", index=False)

    sharpe_ci = pd.DataFrame([
        dict(Label="BASE_OOS", **m6.asymptotic_sharpe_ci(base_r, cfg)),
        dict(Label="M11_OOS", **m6.asymptotic_sharpe_ci(m11_r, cfg)),
    ])
    sharpe_ci.to_csv(f"{cfg.outputs_dir}/{'sharpe_ci.csv' if cfg.run_mode.upper() == 'FULL' else 'sharpe_ci_fast.csv'}", index=False)

    if cfg.run_mode.upper() == "FULL":
        stress = m6.stress_report(m11_r, wf["stitched_m11"]["exposure"], m6.STRESS_EPISODES, cfg, r_bench=wf["stitched_base"]["returns"])
        stress.to_csv(f"{cfg.outputs_dir}/stress_oos.csv", index=False)
        local_cols = [
            "fold", "ceiling_mix", "floor_mix", "ceiling_beta_penalty", "floor_beta_penalty", "raw_rel_boost",
            "structural_prob_thr", "fast_prob_thr", "recovery_prob_thr", "hawkes_weight", "floor_blend_max",
            "gate_floor", "vol_mult_stress", "max_exp_stress", "utility", "val_sharpe", "val_cagr", "val_maxdd",
            "val_pvalue", "val_qvalue",
        ]
        local = leaderboard[[c for c in local_cols if c in leaderboard.columns]].copy()
        local.to_csv(f"{cfg.outputs_dir}/local_sensitivity.csv", index=False)

    report_name = "final_report.txt" if cfg.run_mode.upper() == "FULL" else "final_report_fast.txt"
    with open(f"{cfg.outputs_dir}/{report_name}", "w", encoding="utf-8") as f:
        f.write(build_final_report_text(wf, cfg))

    return {
        "fold_df": fold_df,
        "comparison_oos": comparison_oos,
        "leaderboard": leaderboard,
        "router_art": router_art,
        "alpha_nw": alpha_nw,
        "sharpe_ci": sharpe_ci,
    }
