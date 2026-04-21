from __future__ import annotations

from typing import Any, Dict

import pandas as pd

import mahoraga6_1 as m6
from mahoraga12_config import Mahoraga12Config
from mahoraga12_utils import ensure_dir, paired_ttest_pvalue


def _stitched_tests(wf: Dict[str, Any], cfg: Mahoraga12Config) -> pd.DataFrame:
    legacy_r = wf["stitched_legacy"]["returns"]
    base_r = wf["stitched_base"]["returns"]
    model_r = wf["stitched_m12"]["returns"]
    rows = [
        {
            "Comparison": "BASE_vs_LEGACY",
            "p_value": paired_ttest_pvalue(base_r - legacy_r, alternative="greater"),
            "q_value": float(wf["fold_df"]["Base_vs_Legacy_Test_qvalue"].max()) if len(wf["fold_df"]) else 1.0,
        },
        {
            "Comparison": "M12_vs_BASE",
            "p_value": paired_ttest_pvalue(model_r - base_r, alternative="greater"),
            "q_value": float(wf["fold_df"]["Model_vs_Base_Test_qvalue"].max()) if len(wf["fold_df"]) else 1.0,
        },
        {
            "Comparison": "M12_vs_LEGACY",
            "p_value": paired_ttest_pvalue(model_r - legacy_r, alternative="greater"),
            "q_value": float(wf["fold_df"]["Model_vs_Legacy_Test_qvalue"].max()) if len(wf["fold_df"]) else 1.0,
        },
    ]
    return pd.DataFrame(rows)


def build_full_report_text(wf: Dict[str, Any], cfg: Mahoraga12Config) -> str:
    fold_df = wf["fold_df"].copy().sort_values("fold")
    selected_df = wf.get("selected_df", pd.DataFrame()).copy()
    tests_df = _stitched_tests(wf, cfg)

    legacy_r = wf["stitched_legacy"]["returns"]
    base_r = wf["stitched_base"]["returns"]
    model_r = wf["stitched_m12"]["returns"]
    legacy_eq = cfg.capital_initial * (1.0 + legacy_r).cumprod()
    base_eq = cfg.capital_initial * (1.0 + base_r).cumprod()
    model_eq = cfg.capital_initial * (1.0 + model_r).cumprod()

    s_legacy = m6.summarize(legacy_r, legacy_eq, wf["stitched_legacy"]["exposure"], wf["stitched_legacy"]["turnover"], cfg, "LEGACY")
    s_base = m6.summarize(base_r, base_eq, wf["stitched_base"]["exposure"], wf["stitched_base"]["turnover"], cfg, "BASE_ALPHA")
    s_model = m6.summarize(model_r, model_eq, wf["stitched_m12"]["exposure"], wf["stitched_m12"]["turnover"], cfg, "M12")

    lines = [
        "MAHORAGA 12 — FULL REPORT",
        "=" * 78,
        "",
        "STITCHED OOS COMPARISON",
        f"  LEGACY     CAGR={s_legacy['CAGR']*100:.2f}%  Sharpe={s_legacy['Sharpe']:.3f}  MaxDD={s_legacy['MaxDD']*100:.2f}%",
        f"  BASE_ALPHA CAGR={s_base['CAGR']*100:.2f}%  Sharpe={s_base['Sharpe']:.3f}  MaxDD={s_base['MaxDD']*100:.2f}%",
        f"  M12        CAGR={s_model['CAGR']*100:.2f}%  Sharpe={s_model['Sharpe']:.3f}  MaxDD={s_model['MaxDD']*100:.2f}%",
        "",
        "STITCHED HYPOTHESIS TESTS",
        tests_df.to_string(index=False),
        "",
        "FOLD SUMMARY",
        fold_df.to_string(index=False),
        "",
        "SELECTED CANDIDATES",
        selected_df.to_string(index=False) if len(selected_df) else "No selected candidate rows.",
    ]
    return "\n".join(lines)


def save_full_outputs(wf: Dict[str, Any], cfg: Mahoraga12Config, ff=None) -> Dict[str, pd.DataFrame]:
    ensure_dir(cfg.outputs_dir)
    fold_df = wf["fold_df"].copy().sort_values("fold")
    selected_df = wf.get("selected_df", pd.DataFrame()).copy()
    support_df = wf.get("support_df", pd.DataFrame()).copy()

    legacy_r = wf["stitched_legacy"]["returns"]
    base_r = wf["stitched_base"]["returns"]
    model_r = wf["stitched_m12"]["returns"]
    legacy_eq = cfg.capital_initial * (1.0 + legacy_r).cumprod()
    base_eq = cfg.capital_initial * (1.0 + base_r).cumprod()
    model_eq = cfg.capital_initial * (1.0 + model_r).cumprod()

    comparison = pd.DataFrame([
        m6._fmt(m6.summarize(legacy_r, legacy_eq, wf["stitched_legacy"]["exposure"], wf["stitched_legacy"]["turnover"], cfg, "LEGACY")),
        m6._fmt(m6.summarize(base_r, base_eq, wf["stitched_base"]["exposure"], wf["stitched_base"]["turnover"], cfg, "BASE_ALPHA")),
        m6._fmt(m6.summarize(model_r, model_eq, wf["stitched_m12"]["exposure"], wf["stitched_m12"]["turnover"], cfg, "M12")),
    ])
    comparison.to_csv(f"{cfg.outputs_dir}/comparison_oos.csv", index=False)
    comparison.to_csv(f"{cfg.outputs_dir}/comparison_full.csv", index=False)
    fold_df.to_csv(f"{cfg.outputs_dir}/walk_forward_folds.csv", index=False)
    selected_df.to_csv(f"{cfg.outputs_dir}/selected_candidates.csv", index=False)
    support_df.to_csv(f"{cfg.outputs_dir}/selected_config_support.csv", index=False)

    leaderboard = pd.concat(
        [r["calibration_df"].assign(fold=r["fold"]) for r in wf["results"] if len(r.get("calibration_df", []))],
        axis=0,
        ignore_index=True,
    ) if wf["results"] else pd.DataFrame()
    leaderboard.to_csv(f"{cfg.outputs_dir}/candidate_leaderboard.csv", index=False)

    override_weekly = pd.concat([r["override_weekly"] for r in wf["results"]], axis=0, ignore_index=False) if wf["results"] else pd.DataFrame()
    override_weekly.to_csv(f"{cfg.outputs_dir}/override_weekly.csv", index=True)

    override_diag = pd.DataFrame()
    if len(override_weekly):
        override_diag = (
            override_weekly.reset_index()
            .groupby(["fold", "override_detail"], as_index=False)
            .agg(
                Weeks=("override_type", "size"),
                MeanGate=("gate_scale", "mean"),
                MeanBlend=("defense_blend", "mean"),
                MeanStructuralScore=("structural_score", "mean"),
                MeanTransitionScore=("transition_score", "mean"),
                MeanRecoveryScore=("recovery_score", "mean"),
            )
        )
    override_diag.to_csv(f"{cfg.outputs_dir}/override_diagnostics.csv", index=False)

    alpha_nw = pd.DataFrame([
        m6.alpha_test_nw(base_r, legacy_r, cfg, label="BASE_vs_LEGACY", conditional=False, exposure=wf["stitched_base"]["exposure"]),
        m6.alpha_test_nw(model_r, base_r, cfg, label="M12_vs_BASE", conditional=False, exposure=wf["stitched_m12"]["exposure"]),
        m6.alpha_test_nw(model_r, legacy_r, cfg, label="M12_vs_LEGACY", conditional=False, exposure=wf["stitched_m12"]["exposure"]),
    ])
    alpha_nw.to_csv(f"{cfg.outputs_dir}/alpha_nw.csv", index=False)

    sharpe_ci = pd.DataFrame([
        dict(Label="LEGACY", **m6.asymptotic_sharpe_ci(legacy_r, cfg)),
        dict(Label="BASE_ALPHA", **m6.asymptotic_sharpe_ci(base_r, cfg)),
        dict(Label="M12", **m6.asymptotic_sharpe_ci(model_r, cfg)),
    ])
    sharpe_ci.to_csv(f"{cfg.outputs_dir}/sharpe_ci.csv", index=False)

    stitched_tests = _stitched_tests(wf, cfg)
    stitched_tests.to_csv(f"{cfg.outputs_dir}/stitched_tests.csv", index=False)

    stress = m6.stress_report(model_r, wf["stitched_m12"]["exposure"], m6.STRESS_EPISODES, cfg, r_bench=base_r)
    stress.to_csv(f"{cfg.outputs_dir}/stress_oos.csv", index=False)

    local_cols = [
        "fold",
        "candidate_id",
        "base_mix",
        "defense_mix",
        "base_beta_penalty",
        "defense_beta_penalty",
        "raw_rel_boost",
        "structural_enter_thr",
        "transition_enter_thr",
        "recovery_enter_thr",
        "hawkes_weight",
        "structural_blend",
        "transition_blend",
        "structural_gate",
        "transition_gate",
        "recovery_gate",
        "transition_vol_mult",
        "recovery_vol_mult",
        "structural_exp_cap",
        "transition_exp_cap",
        "recovery_exp_cap",
        "utility",
        "val_base_vs_legacy_sharpe",
        "val_model_vs_base_sharpe",
        "base_vs_legacy_val_pvalue",
        "base_vs_legacy_val_qvalue",
        "model_vs_base_val_pvalue",
        "model_vs_base_val_qvalue",
    ]
    local_sensitivity = leaderboard[[c for c in local_cols if c in leaderboard.columns]].copy()
    local_sensitivity.to_csv(f"{cfg.outputs_dir}/local_sensitivity.csv", index=False)

    if ff is not None:
        ff_rows = [
            m6.factor_attribution(base_r, ff, cfg, label="BASE_ALPHA"),
            m6.factor_attribution(model_r, ff, cfg, label="M12"),
        ]
        pd.DataFrame([row for row in ff_rows if row is not None]).to_csv(f"{cfg.outputs_dir}/ff_attribution.csv", index=False)

    with open(f"{cfg.outputs_dir}/final_report.txt", "w", encoding="utf-8") as f:
        f.write(build_full_report_text(wf, cfg))

    return {
        "fold_df": fold_df,
        "selected_df": selected_df,
        "support_df": support_df,
        "comparison": comparison,
        "leaderboard": leaderboard,
        "override_weekly": override_weekly,
        "override_diag": override_diag,
        "alpha_nw": alpha_nw,
        "sharpe_ci": sharpe_ci,
        "stitched_tests": stitched_tests,
        "stress": stress,
        "local_sensitivity": local_sensitivity,
    }
