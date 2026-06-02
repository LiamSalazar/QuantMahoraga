from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import acceptance_suite_14_3R as acc
import fast_fail_diagnostics_14_3 as d14
import mahoraga6_1 as m6
from mahoraga14_config import Mahoraga14Config
from mahoraga14_utils import bhy_qvalues, ensure_dir, paired_ttest_pvalue


def _candidate_specs() -> List[Dict[str, Any]]:
    rows = [
        ("B1.00_C1.00_L1.00_R1.00", "BASE_CURRENT"),
        ("B1.05_C1.10_L1.10_R1.05", "ROBUST_MAIN"),
        ("B1.05_C1.10_L1.10_R1.00", "NEIGHBOR_BACKOFF"),
        ("B1.05_C1.00_L1.10_R1.05", "NEIGHBOR_CONVICTION"),
        ("B1.05_C1.10_L1.00_R1.05", "NEIGHBOR_LEADER"),
    ]
    out: List[Dict[str, Any]] = []
    for candidate_id, role in rows:
        budget, conviction, leader, backoff = _parse_candidate_id(candidate_id)
        out.append(
            {
                "CandidateId": candidate_id,
                "GateRole": role,
                "budget_multiplier": budget,
                "conviction_multiplier": conviction,
                "leader_multiplier": leader,
                "backoff_strength": backoff,
            }
        )
    return out


def _parse_candidate_id(candidate_id: str) -> Tuple[float, float, float, float]:
    parts = candidate_id.split("_")
    return float(parts[0][1:]), float(parts[1][1:]), float(parts[2][1:]), float(parts[3][1:])


def _stitched_context(wf: Dict[str, Any], cfg: Mahoraga14Config) -> Dict[str, Any]:
    return acc._stitched_base_context(wf, cfg)


def _candidate_objects(
    wf: Dict[str, Any],
    cfg: Mahoraga14Config,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    context = _stitched_context(wf, cfg)
    specs = _candidate_specs()
    out: Dict[str, Dict[str, Any]] = {}
    for spec in specs:
        cid = str(spec["CandidateId"])
        out[cid] = acc._apply_frozen_knobs(
            context["primary"],
            context["allocator"],
            context["leader_diag"],
            context["qqq"],
            cfg,
            float(spec["budget_multiplier"]),
            float(spec["conviction_multiplier"]),
            float(spec["leader_multiplier"]),
            float(spec["backoff_strength"]),
        )
    return out, context, specs


def _stitched_metrics_df(
    candidate_objects: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
    specs: List[Dict[str, Any]],
    cfg: Mahoraga14Config,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    controls = [
        {"Variant": "QQQ", "GateRole": "BENCHMARK", "CandidateId": ""},
        {"Variant": "SPY", "GateRole": "BENCHMARK", "CandidateId": ""},
        {"Variant": "MAHORAGA14_1_LONG_ONLY_CONTROL", "GateRole": "OFFICIAL_CONTROL", "CandidateId": ""},
    ]
    obj_map = {
        "QQQ": context["qqq"],
        "SPY": context["spy"],
        "MAHORAGA14_1_LONG_ONLY_CONTROL": context["control"],
    }
    for item in controls:
        metrics = d14._metrics_row(item["Variant"], obj_map[item["Variant"]], context["qqq"], context["spy"], cfg)
        rows.append(
            {
                "Variant": item["Variant"],
                "GateRole": item["GateRole"],
                "CandidateId": item["CandidateId"],
                "CAGR": float(metrics["CAGR"]) * 100.0,
                "Sharpe": float(metrics["Sharpe"]),
                "Sortino": float(metrics["Sortino"]),
                "MaxDD": float(metrics["MaxDD"]) * 100.0,
                "BetaQQQ": float(metrics["BetaQQQ"]),
                "BetaSPY": float(metrics["BetaSPY"]),
                "AlphaNW_QQQ": float(metrics["AlphaNW_QQQ"]),
                "AlphaNW_SPY": float(metrics["AlphaNW_SPY"]),
                "UpsideCaptureQQQ": float(metrics["UpsideCaptureQQQ"]),
                "DownsideCaptureQQQ": float(metrics["DownsideCaptureQQQ"]),
                "AvgExposure": float(metrics["AvgExposure"]),
                "AvgTurnover": float(metrics["AvgTurnover"]),
            }
        )
    for spec in specs:
        cid = str(spec["CandidateId"])
        metrics = d14._metrics_row(cid, candidate_objects[cid], context["qqq"], context["spy"], cfg)
        rows.append(
            {
                "Variant": cid,
                "GateRole": str(spec["GateRole"]),
                "CandidateId": cid,
                "CAGR": float(metrics["CAGR"]) * 100.0,
                "Sharpe": float(metrics["Sharpe"]),
                "Sortino": float(metrics["Sortino"]),
                "MaxDD": float(metrics["MaxDD"]) * 100.0,
                "BetaQQQ": float(metrics["BetaQQQ"]),
                "BetaSPY": float(metrics["BetaSPY"]),
                "AlphaNW_QQQ": float(metrics["AlphaNW_QQQ"]),
                "AlphaNW_SPY": float(metrics["AlphaNW_SPY"]),
                "UpsideCaptureQQQ": float(metrics["UpsideCaptureQQQ"]),
                "DownsideCaptureQQQ": float(metrics["DownsideCaptureQQQ"]),
                "AvgExposure": float(metrics["AvgExposure"]),
                "AvgTurnover": float(metrics["AvgTurnover"]),
            }
        )
    return pd.DataFrame(rows).round(8)


def _fold_summary_df(
    wf: Dict[str, Any],
    candidate_objects: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
    specs: List[Dict[str, Any]],
    cfg: Mahoraga14Config,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    fold_map = [
        ("QQQ", "BENCHMARK", context["qqq"]),
        ("SPY", "BENCHMARK", context["spy"]),
        ("MAHORAGA14_1_LONG_ONLY_CONTROL", "OFFICIAL_CONTROL", context["control"]),
    ]
    for result in wf["results"]:
        fold_n = int(result["fold"])
        start = pd.Timestamp(result["test_start"])
        end = pd.Timestamp(result["test_end"])
        qqq_slice = acc._slice_object(context["qqq"], start, end, cfg)
        spy_slice = acc._slice_object(context["spy"], start, end, cfg)
        for variant, role, obj in fold_map:
            sliced = acc._slice_object(obj, start, end, cfg)
            metrics = d14._metrics_row(variant, sliced, qqq_slice, spy_slice, cfg)
            rows.append(
                {
                    "Fold": fold_n,
                    "TestStart": start,
                    "TestEnd": end,
                    "Variant": variant,
                    "GateRole": role,
                    "CandidateId": "",
                    "CAGR": float(metrics["CAGR"]) * 100.0,
                    "Sharpe": float(metrics["Sharpe"]),
                    "Sortino": float(metrics["Sortino"]),
                    "MaxDD": float(metrics["MaxDD"]) * 100.0,
                    "BetaQQQ": float(metrics["BetaQQQ"]),
                    "BetaSPY": float(metrics["BetaSPY"]),
                    "AlphaNW_QQQ": float(metrics["AlphaNW_QQQ"]),
                    "AlphaNW_SPY": float(metrics["AlphaNW_SPY"]),
                    "UpsideCaptureQQQ": float(metrics["UpsideCaptureQQQ"]),
                    "DownsideCaptureQQQ": float(metrics["DownsideCaptureQQQ"]),
                    "Exposure": float(metrics["AvgExposure"]),
                    "Turnover": float(metrics["AvgTurnover"]),
                }
            )
        for spec in specs:
            cid = str(spec["CandidateId"])
            sliced = acc._slice_object(candidate_objects[cid], start, end, cfg)
            metrics = d14._metrics_row(cid, sliced, qqq_slice, spy_slice, cfg)
            rows.append(
                {
                    "Fold": fold_n,
                    "TestStart": start,
                    "TestEnd": end,
                    "Variant": cid,
                    "GateRole": str(spec["GateRole"]),
                    "CandidateId": cid,
                    "CAGR": float(metrics["CAGR"]) * 100.0,
                    "Sharpe": float(metrics["Sharpe"]),
                    "Sortino": float(metrics["Sortino"]),
                    "MaxDD": float(metrics["MaxDD"]) * 100.0,
                    "BetaQQQ": float(metrics["BetaQQQ"]),
                    "BetaSPY": float(metrics["BetaSPY"]),
                    "AlphaNW_QQQ": float(metrics["AlphaNW_QQQ"]),
                    "AlphaNW_SPY": float(metrics["AlphaNW_SPY"]),
                    "UpsideCaptureQQQ": float(metrics["UpsideCaptureQQQ"]),
                    "DownsideCaptureQQQ": float(metrics["DownsideCaptureQQQ"]),
                    "Exposure": float(metrics["AvgExposure"]),
                    "Turnover": float(metrics["AvgTurnover"]),
                }
            )
    return pd.DataFrame(rows).round(8)


def _bull_window_scorecard_df(
    candidate_objects: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
    specs: List[Dict[str, Any]],
    cfg: Mahoraga14Config,
) -> pd.DataFrame:
    windows = d14._window_specs(pd.DatetimeIndex(context["qqq"]["returns"].index), context["qqq"]["returns"], cfg)
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        cid = str(spec["CandidateId"])
        candidate_obj = candidate_objects[cid]
        for window_name, start, end, source in windows:
            cand_slice = acc._slice_object(candidate_obj, start, end, cfg)
            ctrl_slice = acc._slice_object(context["control"], start, end, cfg)
            qqq_slice = acc._slice_object(context["qqq"], start, end, cfg)
            spy_slice = acc._slice_object(context["spy"], start, end, cfg)
            cand_vs_qqq = d14._window_summary(cand_slice, qqq_slice, cfg, str(window_name))
            cand_vs_spy = d14._window_summary(cand_slice, spy_slice, cfg, str(window_name))
            ctrl_vs_qqq = d14._window_summary(ctrl_slice, qqq_slice, cfg, str(window_name))
            qqq_return = float(np.prod(1.0 + pd.Series(qqq_slice["returns"], dtype=float).values) - 1.0)
            spy_return = float(np.prod(1.0 + pd.Series(spy_slice["returns"], dtype=float).values) - 1.0)
            rows.append(
                {
                    "CandidateId": cid,
                    "GateRole": str(spec["GateRole"]),
                    "Window": str(window_name),
                    "Source": str(source),
                    "Start": start,
                    "End": end,
                    "CandidateReturn": float(cand_vs_qqq["Return"]),
                    "ControlReturn": float(ctrl_vs_qqq["Return"]),
                    "QQQReturn": qqq_return,
                    "SPYReturn": spy_return,
                    "DeltaReturn_vs_Control": float(cand_vs_qqq["Return"] - ctrl_vs_qqq["Return"]),
                    "DeltaReturn_vs_QQQ": float(cand_vs_qqq["Return"] - qqq_return),
                    "DeltaReturn_vs_SPY": float(cand_vs_spy["Return"] - spy_return),
                    "SharpeLocal": float(cand_vs_qqq["Sharpe"]),
                    "SortinoLocal": float(cand_vs_qqq["Sortino"]),
                    "MaxDDLocal": float(cand_vs_qqq["MaxDD"]),
                    "BetaQQQLocal": float(cand_vs_qqq["Beta"]),
                    "BetaSPYLocal": float(cand_vs_spy["Beta"]),
                    "UpsideCaptureQQQLocal": float(cand_vs_qqq["UpsideCapture"]),
                    "ExposureLocal": float(cand_vs_qqq["Exposure"]),
                }
            )
    return pd.DataFrame(rows).round(8)


def _priority_gate_status(row: pd.Series) -> Tuple[str, str]:
    window = str(row["Window"])
    delta_control = float(row["DeltaReturn_vs_Control"])
    delta_qqq = float(row["DeltaReturn_vs_QQQ"])
    sharpe = float(row["SharpeLocal"])
    maxdd = float(row["MaxDDLocal"])
    if window == "2017_2018":
        if delta_control >= 0.0 and sharpe >= 0.75:
            return "PASS", "beats_or_matches_control_with_acceptable_local_quality"
        if delta_control >= -0.02 and delta_qqq >= -0.08 and sharpe >= 0.70 and maxdd >= -0.18:
            return "TOLERABLE", "no_longer_clearly_unacceptable"
        return "FAIL", "still_clearly_weaker_than_control_or_too_far_below_qqq"
    if window == "2020_2021":
        if delta_control >= 0.0:
            return "PASS", "maintains_improvement_vs_control"
        if delta_control >= -0.01 and sharpe >= 1.25:
            return "TOLERABLE", "slightly_weaker_but_not_structurally_broken"
        return "FAIL", "deteriorates_vs_control_in_priority_bull_window"
    if window == "2023_2024":
        if delta_control >= 0.0 and delta_qqq >= 0.0:
            return "PASS", "keeps_recent_bull_strength"
        if delta_control >= -0.02 and sharpe >= 1.75:
            return "TOLERABLE", "still_strong_even_if_not_best"
        return "FAIL", "breaks_recent_bull_strength"
    return "FAIL", "unknown_window"


def _priority_window_df(scorecard_df: pd.DataFrame, cfg: Mahoraga14Config) -> pd.DataFrame:
    out = scorecard_df[scorecard_df["Window"].isin(("2017_2018", "2020_2021", "2023_2024"))].copy()
    statuses = out.apply(_priority_gate_status, axis=1, result_type="expand")
    out["GateStatus"] = statuses[0].values
    out["GateReason"] = statuses[1].values
    return out.reset_index(drop=True).round(8)


def _active_return_df(
    candidate_objects: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
    specs: List[Dict[str, Any]],
) -> pd.DataFrame:
    idx = pd.DatetimeIndex(context["qqq"]["returns"].index)
    df = pd.DataFrame(index=idx)
    df["QQQReturn"] = pd.Series(context["qqq"]["returns"], dtype=float).reindex(idx).fillna(0.0)
    df["SPYReturn"] = pd.Series(context["spy"]["returns"], dtype=float).reindex(idx).fillna(0.0)
    df["ControlReturn"] = pd.Series(context["control"]["returns"], dtype=float).reindex(idx).fillna(0.0)
    df["CumControl"] = (1.0 + df["ControlReturn"]).cumprod() - 1.0
    for spec in specs:
        cid = str(spec["CandidateId"])
        ret = pd.Series(candidate_objects[cid]["returns"], dtype=float).reindex(idx).fillna(0.0)
        df[f"{cid}_Return"] = ret
        df[f"{cid}_CumReturn"] = (1.0 + ret).cumprod() - 1.0
        active = ret - df["QQQReturn"]
        df[f"{cid}_Active_vs_QQQ"] = active
        df[f"{cid}_CumActive_vs_QQQ"] = (1.0 + active).cumprod() - 1.0
    return df.reset_index(names="Date").round(8)


def _alpha_nw_df(
    candidate_objects: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
    specs: List[Dict[str, Any]],
    cfg: Mahoraga14Config,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    variants = [("MAHORAGA14_1_LONG_ONLY_CONTROL", "OFFICIAL_CONTROL", context["control"])] + [
        (str(spec["CandidateId"]), str(spec["GateRole"]), candidate_objects[str(spec["CandidateId"])]) for spec in specs
    ]
    for variant, role, obj in variants:
        for bench_label, bench_obj in (("QQQ", context["qqq"]), ("SPY", context["spy"])):
            res = m6.alpha_test_nw(obj["returns"], bench_obj["returns"], cfg, label=f"{variant}_{bench_label}")
            rows.append(
                {
                    "Variant": variant,
                    "GateRole": role,
                    "Benchmark": bench_label,
                    "alpha_ann": float(res.get("alpha_ann", np.nan)),
                    "t_alpha": float(res.get("t_alpha", np.nan)),
                    "p_alpha": float(res.get("p_alpha", np.nan)),
                    "beta": float(res.get("beta", np.nan)),
                    "R2": float(res.get("R2", np.nan)),
                }
            )
    return pd.DataFrame(rows).round(8)


def _pvalue_qvalue_df(
    stitched_df: pd.DataFrame,
    candidate_objects: Dict[str, Dict[str, Any]],
    context: Dict[str, Any],
    specs: List[Dict[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    refs = {
        "MAHORAGA14_1_LONG_ONLY_CONTROL": context["control"],
        "QQQ": context["qqq"],
        "SPY": context["spy"],
    }
    for spec in specs:
        cid = str(spec["CandidateId"])
        for ref_label, ref_obj in refs.items():
            pval = paired_ttest_pvalue(
                pd.Series(candidate_objects[cid]["returns"], dtype=float) - pd.Series(ref_obj["returns"], dtype=float),
                alternative="greater",
            )
            rows.append(
                {
                    "Variant": cid,
                    "GateRole": str(spec["GateRole"]),
                    "Reference": ref_label,
                    "p_value": float(pval),
                }
            )
    out = pd.DataFrame(rows)
    out["q_value"] = bhy_qvalues(out["p_value"].values, alpha=0.05) if len(out) else []
    metric_map = stitched_df.set_index("Variant")[["CAGR", "Sharpe", "Sortino", "MaxDD", "AlphaNW_QQQ", "AlphaNW_SPY"]].to_dict("index")
    out["CAGR_Delta"] = [metric_map[row["Variant"]]["CAGR"] - metric_map[row["Reference"]]["CAGR"] for _, row in out.iterrows()]
    out["Sharpe_Delta"] = [metric_map[row["Variant"]]["Sharpe"] - metric_map[row["Reference"]]["Sharpe"] for _, row in out.iterrows()]
    out["Sortino_Delta"] = [metric_map[row["Variant"]]["Sortino"] - metric_map[row["Reference"]]["Sortino"] for _, row in out.iterrows()]
    out["MaxDD_Delta"] = [metric_map[row["Variant"]]["MaxDD"] - metric_map[row["Reference"]]["MaxDD"] for _, row in out.iterrows()]
    return out.round(8)


def _decision_table(
    stitched_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    priority_df: pd.DataFrame,
    specs: List[Dict[str, Any]],
) -> pd.DataFrame:
    control = stitched_df[stitched_df["Variant"] == "MAHORAGA14_1_LONG_ONLY_CONTROL"].iloc[0]
    control_folds = fold_df[fold_df["Variant"] == "MAHORAGA14_1_LONG_ONLY_CONTROL"][["Fold", "CAGR", "Sharpe", "Sortino", "MaxDD"]].rename(
        columns={"CAGR": "ControlCAGR", "Sharpe": "ControlSharpe", "Sortino": "ControlSortino", "MaxDD": "ControlMaxDD"}
    )
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        cid = str(spec["CandidateId"])
        stitched = stitched_df[stitched_df["Variant"] == cid].iloc[0]
        candidate_priority = priority_df[priority_df["CandidateId"] == cid].copy()
        priority_map = candidate_priority.set_index("Window")["GateStatus"].to_dict()
        candidate_folds = fold_df[fold_df["Variant"] == cid].merge(control_folds, on="Fold", how="left")
        worst_fold_sharpe_delta = float((candidate_folds["Sharpe"] - candidate_folds["ControlSharpe"]).min())
        worst_fold_cagr_delta = float((candidate_folds["CAGR"] - candidate_folds["ControlCAGR"]).min())
        severe_fold_damage = int(
            (
                ((candidate_folds["Sharpe"] - candidate_folds["ControlSharpe"]) < -0.25)
                | ((candidate_folds["CAGR"] - candidate_folds["ControlCAGR"]) < -6.0)
                | ((candidate_folds["MaxDD"] - candidate_folds["ControlMaxDD"]) < -5.0)
            ).sum()
        )
        eligible = (
            (float(stitched["CAGR"] - control["CAGR"]) >= 1.00)
            and (float(stitched["Sharpe"] - control["Sharpe"]) >= 0.05)
            and (float(stitched["Sortino"] - control["Sortino"]) >= 0.05)
            and (float(stitched["MaxDD"] - control["MaxDD"]) >= -1.00)
            and (float(stitched["AlphaNW_QQQ"] - control["AlphaNW_QQQ"]) >= 0.0)
            and (float(stitched["AlphaNW_SPY"] - control["AlphaNW_SPY"]) >= 0.0)
            and (priority_map.get("2017_2018", "FAIL") != "FAIL")
            and (priority_map.get("2020_2021", "FAIL") != "FAIL")
            and (priority_map.get("2023_2024", "FAIL") != "FAIL")
            and (worst_fold_sharpe_delta >= -0.20)
            and (severe_fold_damage == 0)
        )
        priority_penalty = int((candidate_priority["GateStatus"] == "FAIL").sum())
        score = (
            2.0 * float(stitched["Sharpe"] - control["Sharpe"])
            + 0.25 * float(stitched["CAGR"] - control["CAGR"])
            + 0.50 * float(stitched["Sortino"] - control["Sortino"])
            + 0.10 * float(stitched["MaxDD"] - control["MaxDD"])
            + 5.0 * float(stitched["AlphaNW_QQQ"] - control["AlphaNW_QQQ"])
            + 5.0 * float(stitched["AlphaNW_SPY"] - control["AlphaNW_SPY"])
            - 0.50 * float(priority_penalty)
            + 0.10 * float(priority_map.get("2017_2018", "FAIL") == "PASS")
        )
        rows.append(
            {
                "CandidateId": cid,
                "GateRole": str(spec["GateRole"]),
                "DeltaCAGR_vs_Control": float(stitched["CAGR"] - control["CAGR"]),
                "DeltaSharpe_vs_Control": float(stitched["Sharpe"] - control["Sharpe"]),
                "DeltaSortino_vs_Control": float(stitched["Sortino"] - control["Sortino"]),
                "DeltaMaxDD_vs_Control": float(stitched["MaxDD"] - control["MaxDD"]),
                "DeltaAlphaNW_QQQ_vs_Control": float(stitched["AlphaNW_QQQ"] - control["AlphaNW_QQQ"]),
                "DeltaAlphaNW_SPY_vs_Control": float(stitched["AlphaNW_SPY"] - control["AlphaNW_SPY"]),
                "Window2017_2018": str(priority_map.get("2017_2018", "FAIL")),
                "Window2020_2021": str(priority_map.get("2020_2021", "FAIL")),
                "Window2023_2024": str(priority_map.get("2023_2024", "FAIL")),
                "WorstFoldSharpeDelta": worst_fold_sharpe_delta,
                "WorstFoldCAGRDelta": worst_fold_cagr_delta,
                "SevereFoldDamageCount": severe_fold_damage,
                "EligibleForPromotion": int(bool(eligible)),
                "PromotionScore": score,
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["EligibleForPromotion", "PromotionScore", "DeltaSharpe_vs_Control", "DeltaCAGR_vs_Control"],
        ascending=[False, False, False, False],
    )
    return out.round(8).reset_index(drop=True)


def _active_summary_md(active_df: pd.DataFrame, specs: List[Dict[str, Any]]) -> str:
    lines = []
    for spec in specs:
        cid = str(spec["CandidateId"])
        cum_col = f"{cid}_CumActive_vs_QQQ"
        lines.append(
            f"- `{cid}` final cum active vs QQQ: {float(active_df[cum_col].iloc[-1]):.4f}; worst trough: {float(active_df[cum_col].min()):.4f}"
        )
    return "\n".join(lines)


def _promotion_decision_md(
    decision: str,
    selected: pd.Series,
    decision_df: pd.DataFrame,
) -> str:
    lines = [
        "# Promotion Gate Decision",
        "",
        f"## Decision: {decision}",
        "",
        f"- selected gate candidate: `{selected['CandidateId']}` ({selected['GateRole']})",
        f"- delta CAGR vs control: {float(selected['DeltaCAGR_vs_Control']):.4f} pts",
        f"- delta Sharpe vs control: {float(selected['DeltaSharpe_vs_Control']):.4f}",
        f"- delta Sortino vs control: {float(selected['DeltaSortino_vs_Control']):.4f}",
        f"- delta MaxDD vs control: {float(selected['DeltaMaxDD_vs_Control']):.4f} pts",
        f"- priority windows: 2017_2018={selected['Window2017_2018']}, 2020_2021={selected['Window2020_2021']}, 2023_2024={selected['Window2023_2024']}",
        f"- worst fold Sharpe delta vs control: {float(selected['WorstFoldSharpeDelta']):.4f}",
        f"- severe fold damage count: {int(selected['SevereFoldDamageCount'])}",
        "",
        "## Gate scoreboard",
        f"```\n{decision_df.to_string(index=False)}\n```",
    ]
    if decision == "PROMOTE_TO_OFFICIAL_BASELINE":
        lines.append("")
        lines.append("- institutional conclusion: the selected 14.3R candidate is strong enough to replace the official long-only control.")
    else:
        lines.append("")
        lines.append("- institutional conclusion: Mahoraga14_3R should remain a shadow baseline / experimental branch and should NOT replace Mahoraga14_1_LONG_ONLY_CONTROL yet.")
    return "\n".join(lines)


def _final_report_md(
    stitched_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    bull_df: pd.DataFrame,
    priority_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    pq_df: pd.DataFrame,
    decision: str,
    selected: pd.Series,
    specs: List[Dict[str, Any]],
    active_df: pd.DataFrame,
) -> str:
    lines = [
        "# Mahoraga14_3R Promotion Gate",
        "",
        "## Candidate set",
        *[f"- `{spec['CandidateId']}` ({spec['GateRole']})" for spec in specs],
        "",
        "## Stitched OOS",
        f"```\n{stitched_df.to_string(index=False)}\n```",
        "",
        "## Folds",
        f"```\n{fold_df.to_string(index=False)}\n```",
        "",
        "## Bull windows",
        f"```\n{bull_df.to_string(index=False)}\n```",
        "",
        "## Priority windows",
        f"```\n{priority_df.to_string(index=False)}\n```",
        "",
        "## Active return vs QQQ",
        _active_summary_md(active_df, specs),
        "",
        "## Alpha vs QQQ/SPY",
        f"```\n{alpha_df.to_string(index=False)}\n```",
        "",
        "## p-values / q-values",
        f"```\n{pq_df.to_string(index=False)}\n```",
        "",
        "## Final decision",
        f"- decision: {decision}",
        f"- selected candidate: `{selected['CandidateId']}` ({selected['GateRole']})",
    ]
    return "\n".join(lines)


def save_promotion_gate_outputs(
    wf: Dict[str, Any],
    cfg: Mahoraga14Config,
) -> Dict[str, Any]:
    ensure_dir(cfg.outputs_dir)
    out_dir = Path(cfg.outputs_dir)

    candidate_objects, context, specs = _candidate_objects(wf, cfg)
    stitched_df = _stitched_metrics_df(candidate_objects, context, specs, cfg)
    fold_df = _fold_summary_df(wf, candidate_objects, context, specs, cfg)
    bull_df = _bull_window_scorecard_df(candidate_objects, context, specs, cfg)
    priority_df = _priority_window_df(bull_df, cfg)
    active_df = _active_return_df(candidate_objects, context, specs)
    alpha_df = _alpha_nw_df(candidate_objects, context, specs, cfg)
    pq_df = _pvalue_qvalue_df(stitched_df, candidate_objects, context, specs)
    decision_df = _decision_table(stitched_df, fold_df, priority_df, specs)
    selected = decision_df.iloc[0]
    decision = "PROMOTE_TO_OFFICIAL_BASELINE" if int(selected["EligibleForPromotion"]) == 1 else "KEEP_AS_SHADOW_BASELINE"

    stitched_df.to_csv(out_dir / "stitched_comparison_promotion_gate.csv", index=False)
    fold_df.to_csv(out_dir / "fold_summary_promotion_gate.csv", index=False)
    bull_df.to_csv(out_dir / "bull_window_scorecard_promotion_gate.csv", index=False)
    priority_df.to_csv(out_dir / "priority_window_acceptance_promotion_gate.csv", index=False)
    active_df.to_csv(out_dir / "active_return_vs_qqq_promotion_gate.csv", index=False)
    alpha_df.to_csv(out_dir / "alpha_nw_promotion_gate.csv", index=False)
    pq_df.to_csv(out_dir / "pvalue_qvalue_promotion_gate.csv", index=False)

    decision_md = _promotion_decision_md(decision, selected, decision_df)
    final_md = _final_report_md(stitched_df, fold_df, bull_df, priority_df, alpha_df, pq_df, decision, selected, specs, active_df)
    (out_dir / "promotion_gate_decision.md").write_text(decision_md, encoding="utf-8")
    (out_dir / "final_report_promotion_gate.md").write_text(final_md, encoding="utf-8")

    return {
        "decision": decision,
        "selected_candidate": selected.to_dict(),
        "stitched_df": stitched_df,
        "fold_df": fold_df,
        "bull_df": bull_df,
        "priority_df": priority_df,
        "active_df": active_df,
        "alpha_df": alpha_df,
        "pq_df": pq_df,
        "decision_df": decision_df,
    }
