from __future__ import annotations

from typing import Any, Dict

import pandas as pd

import mahoraga6_1 as m6
from mahoraga10_config import Mahoraga10Config
from mahoraga10_utils import ensure_dir, paired_ttest_pvalue


def build_fold_summary(wf: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([r["fold_row"] for r in wf["results"]]).sort_values("fold")


def build_final_report_text(wf: Dict[str, Any], cfg: Mahoraga10Config) -> str:
    fold_df = build_fold_summary(wf)
    base_r = wf["stitched_base"]["returns"]
    m10_r = wf["stitched_m10"]["returns"]
    base_eq = cfg.capital_initial * (1.0 + base_r).cumprod()
    m10_eq = cfg.capital_initial * (1.0 + m10_r).cumprod()
    s_base = m6.summarize(base_r, base_eq, wf["stitched_base"]["exposure"], wf["stitched_base"]["turnover"], cfg, "BASELINE")
    s_m10 = m6.summarize(m10_r, m10_eq, wf["stitched_m10"]["exposure"], wf["stitched_m10"]["turnover"], cfg, "M10")
    stitched_p = paired_ttest_pvalue(m10_r - base_r, alternative="greater")
    q_max = float(fold_df["Test_qvalue"].max()) if "Test_qvalue" in fold_df.columns and len(fold_df) else 1.0

    floor_mean = float(fold_df.loc[fold_df["fold"].isin(cfg.floor_folds), "M10_Sharpe"].mean()) if len(fold_df) else 0.0
    ceil_mean = float(fold_df.loc[fold_df["fold"].isin(cfg.ceiling_folds), "M10_Sharpe"].mean()) if len(fold_df) else 0.0

    lines = []
    lines.append("MAHORAGA 10 — FINAL REPORT")
    lines.append("=" * 78)
    lines.append("")
    lines.append("OOS COMPARISON")
    lines.append(f"  BASELINE CAGR={s_base['CAGR']*100:.2f}%  Sharpe={s_base['Sharpe']:.3f}  MaxDD={s_base['MaxDD']*100:.2f}%")
    lines.append(f"  M10      CAGR={s_m10['CAGR']*100:.2f}%  Sharpe={s_m10['Sharpe']:.3f}  MaxDD={s_m10['MaxDD']*100:.2f}%")
    lines.append(f"  Stitched diff p-value={stitched_p:.6f}")
    lines.append(f"  Fold-level BHY max q-value={q_max:.6f}")
    lines.append("")
    lines.append("FLOOR / CEILING SUMMARY")
    lines.append(f"  Floor folds {cfg.floor_folds}: mean M10 Sharpe={floor_mean:.4f}")
    lines.append(f"  Ceiling folds {cfg.ceiling_folds}: mean M10 Sharpe={ceil_mean:.4f}")
    lines.append("")
    lines.append("FOLD SUMMARY")
    lines.append(fold_df.to_string(index=False))
    return "\n".join(lines)


def save_outputs(wf: Dict[str, Any], cfg: Mahoraga10Config, ff=None) -> Dict[str, pd.DataFrame]:
    ensure_dir(cfg.outputs_dir)
    fold_df = build_fold_summary(wf)
    base_r = wf["stitched_base"]["returns"]
    m10_r = wf["stitched_m10"]["returns"]
    base_eq = cfg.capital_initial * (1.0 + base_r).cumprod()
    m10_eq = cfg.capital_initial * (1.0 + m10_r).cumprod()

    comparison_oos = pd.DataFrame([
        m6._fmt(m6.summarize(base_r, base_eq, wf["stitched_base"]["exposure"], wf["stitched_base"]["turnover"], cfg, "BASELINE")),
        m6._fmt(m6.summarize(m10_r, m10_eq, wf["stitched_m10"]["exposure"], wf["stitched_m10"]["turnover"], cfg, "M10")),
    ])
    comparison_oos.to_csv(f"{cfg.outputs_dir}/{'comparison_oos.csv' if cfg.run_mode.upper() == 'FULL' else 'stitched_comparison_fast.csv'}", index=False)
    fold_df.to_csv(f"{cfg.outputs_dir}/{'walk_forward_folds.csv' if cfg.run_mode.upper() == 'FULL' else 'walk_forward_folds_fast.csv'}", index=False)

    leaderboard = pd.concat(
        [r["calibration_df"].assign(fold=r["fold"]) for r in wf["results"] if len(r.get("calibration_df", []))],
        axis=0,
        ignore_index=True,
    ) if wf["results"] else pd.DataFrame()
    leaderboard.to_csv(f"{cfg.outputs_dir}/{'candidate_leaderboard.csv' if cfg.run_mode.upper() == 'FULL' else 'candidate_leaderboard_fast.csv'}", index=False)

    alpha_rows = []
    for lbl, ret, exp in [
        ("BASE_OOS", base_r, wf["stitched_base"]["exposure"]),
        ("M10_OOS", m10_r, wf["stitched_m10"]["exposure"]),
    ]:
        alpha_rows.append(m6.alpha_test_nw(ret, wf["stitched_base"]["returns"], cfg, label=lbl, conditional=False, exposure=exp))
    alpha_nw = pd.DataFrame(alpha_rows)
    alpha_nw.to_csv(f"{cfg.outputs_dir}/{'alpha_nw.csv' if cfg.run_mode.upper() == 'FULL' else 'alpha_nw_fast.csv'}", index=False)

    sharpe_ci = pd.DataFrame([
        dict(Label="BASE_OOS", **m6.asymptotic_sharpe_ci(base_r, cfg)),
        dict(Label="M10_OOS", **m6.asymptotic_sharpe_ci(m10_r, cfg)),
    ])
    sharpe_ci.to_csv(f"{cfg.outputs_dir}/{'sharpe_ci.csv' if cfg.run_mode.upper() == 'FULL' else 'sharpe_ci_fast.csv'}", index=False)

    if cfg.run_mode.upper() == "FULL":
        stress = m6.stress_report(m10_r, wf["stitched_m10"]["exposure"], m6.STRESS_EPISODES, cfg, r_bench=wf["stitched_base"]["returns"])
        stress.to_csv(f"{cfg.outputs_dir}/stress_oos.csv", index=False)
        local_cols = [
            "fold", "alpha_mix_base", "beta_penalty", "raw_rel_boost",
            "fragility_prob_thr", "recovery_prob_thr", "hawkes_weight", "gate_floor",
            "vol_mult_stress", "max_exp_stress", "alpha_tilt", "utility",
            "val_sharpe", "val_cagr", "val_maxdd", "val_pvalue", "val_qvalue",
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
        "alpha_nw": alpha_nw,
        "sharpe_ci": sharpe_ci,
    }
