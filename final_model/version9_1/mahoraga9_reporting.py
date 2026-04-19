from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

import mahoraga6_1 as m6
from mahoraga9_config import Mahoraga9Config
from mahoraga9_utils import ensure_dir, paired_ttest_pvalue, bhy_qvalues


def _fold_df(wf: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([r["fold_row"] for r in wf["results"]]).sort_values("fold")


def _summary_from_stitched(stitched: Dict[str, pd.Series], cfg: Mahoraga9Config, label: str) -> Dict[str, float]:
    return m6.summarize(
        stitched["returns"],
        stitched["equity"] * cfg.capital_initial,
        stitched["exposure"],
        stitched["turnover"],
        cfg,
        label,
    )


def save_outputs_v9(cfg: Mahoraga9Config, wf: Dict[str, Any]) -> Dict[str, Path]:
    ensure_dir(cfg.outputs_dir)
    out_dir = Path(cfg.outputs_dir)
    files: Dict[str, Path] = {}

    fold_df = _fold_df(wf)
    fold_name = "walk_forward_folds_fast.csv" if cfg.run_mode.upper() == "FAST" else "walk_forward_folds.csv"
    fold_df.to_csv(out_dir / fold_name, index=False)
    files[fold_name] = out_dir / fold_name

    all_calib = []
    for r in wf["results"]:
        df = r["calibration_df"].copy()
        df["fold"] = r["fold"]
        all_calib.append(df)
    calib_df = pd.concat(all_calib).sort_values(["fold", "utility"], ascending=[True, False]) if all_calib else pd.DataFrame()
    calib_name = "candidate_leaderboard_fast.csv" if cfg.run_mode.upper() == "FAST" else "candidate_leaderboard.csv"
    calib_df.to_csv(out_dir / calib_name, index=False)
    files[calib_name] = out_dir / calib_name

    policy_rows = pd.DataFrame([
        {"fold": r["fold"], **r["best_policy"], "FragilityModel": r["frag_fit"].get("name", "neutral"), "RecoveryModel": r["recv_fit"].get("name", "neutral")}
        for r in wf["results"]
    ])
    policy_name = "best_params_by_fold_fast.csv" if cfg.run_mode.upper() == "FAST" else "best_params_by_fold.csv"
    policy_rows.to_csv(out_dir / policy_name, index=False)
    files[policy_name] = out_dir / policy_name

    base_s = _summary_from_stitched(wf["stitched_base"], cfg, "BASELINE_6_1")
    v9_s = _summary_from_stitched(wf["stitched_v9"], cfg, "V9_1")

    stitched_diff = wf["stitched_v9"]["returns"].reindex(wf["stitched_base"]["returns"].index).fillna(0.0) - wf["stitched_base"]["returns"].fillna(0.0)
    stitched_p = paired_ttest_pvalue(stitched_diff, alternative="greater")
    fold_p = fold_df["Test_pvalue"].astype(float).tolist() if len(fold_df) else []
    fold_q = bhy_qvalues(fold_p) if fold_p else []
    if len(fold_df):
        fold_df = fold_df.copy()
        fold_df["Test_qvalue"] = fold_q
        fold_df.to_csv(out_dir / fold_name, index=False)

    floor_mask = fold_df["fold"].isin(cfg.floor_folds) if len(fold_df) else pd.Series(dtype=bool)
    ceil_mask = fold_df["fold"].isin(cfg.ceiling_folds) if len(fold_df) else pd.Series(dtype=bool)
    floor_mean = float(fold_df.loc[floor_mask, "V9_Sharpe"].mean()) if len(fold_df.loc[floor_mask]) else 0.0
    ceil_mean = float(fold_df.loc[ceil_mask, "V9_Sharpe"].mean()) if len(fold_df.loc[ceil_mask]) else 0.0

    report_name = "final_report_fast.txt" if cfg.run_mode.upper() == "FAST" else "final_report.txt"
    report_path = out_dir / report_name
    with report_path.open("w", encoding="utf-8") as f:
        f.write("MAHORAGA 9.1 — FINAL REPORT\n")
        f.write("=" * 78 + "\n\n")
        f.write("OOS COMPARISON\n")
        f.write(f"  BASELINE CAGR={base_s['CAGR']*100:.2f}%  Sharpe={base_s['Sharpe']:.3f}  MaxDD={base_s['MaxDD']*100:.2f}%\\n")
        f.write(f"  V9.1     CAGR={v9_s['CAGR']*100:.2f}%  Sharpe={v9_s['Sharpe']:.3f}  MaxDD={v9_s['MaxDD']*100:.2f}%\\n")
        f.write(f"  Stitched diff p-value={stitched_p:.6f}\\n")
        if len(fold_q):
            f.write(f"  Fold-level BHY max q-value={max(fold_q):.6f}\\n")
        f.write("\nFLOOR / CEILING SUMMARY\n")
        f.write(f"  Floor folds {cfg.floor_folds}: mean V9 Sharpe={floor_mean:.4f}\\n")
        f.write(f"  Ceiling folds {cfg.ceiling_folds}: mean V9 Sharpe={ceil_mean:.4f}\\n")
        f.write("\nFOLD SUMMARY\n")
        f.write(fold_df.to_string(index=False))
        f.write("\n")
    files[report_name] = report_path

    model_rows = []
    for r in wf["results"]:
        model_rows.append(
            {
                "fold": r["fold"],
                "fragility_model": r["frag_fit"].get("name", "neutral"),
                "fragility_auc_val": r["frag_fit"].get("auc_val"),
                "fragility_brier_val": r["frag_fit"].get("brier_val"),
                "recovery_model": r["recv_fit"].get("name", "neutral"),
                "recovery_auc_val": r["recv_fit"].get("auc_val"),
                "recovery_brier_val": r["recv_fit"].get("brier_val"),
            }
        )
    model_df = pd.DataFrame(model_rows)
    model_name = "meta_diagnostics_fast.csv" if cfg.run_mode.upper() == "FAST" else "meta_model_diagnostics.csv"
    model_df.to_csv(out_dir / model_name, index=False)
    files[model_name] = out_dir / model_name
    return files
