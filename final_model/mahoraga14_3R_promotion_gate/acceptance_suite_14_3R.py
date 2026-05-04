from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mahoraga6_1 as m6
import fast_fail_diagnostics_14_3 as d14
from mahoraga14_config import Mahoraga14Config
from mahoraga14_utils import ensure_dir


def _series(x: Any, index: pd.Index, default: float = 0.0) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.Series(x, index=index, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(float(x), index=index, dtype=float)


def _object_from_existing(obj: Dict[str, Any], cfg: Mahoraga14Config) -> Dict[str, Any]:
    returns = pd.Series(obj["returns"], dtype=float).fillna(0.0)
    gross = pd.Series(obj.get("gross_returns", returns), dtype=float).reindex(returns.index).fillna(returns)
    tc = pd.Series(obj.get("transaction_cost", pd.Series(0.0, index=returns.index)), dtype=float).reindex(returns.index).fillna(0.0)
    exposure = pd.Series(obj["exposure"], dtype=float).reindex(returns.index).fillna(0.0)
    turnover = pd.Series(obj["turnover"], dtype=float).reindex(returns.index).fillna(0.0)
    return {
        "returns": returns,
        "gross_returns": gross,
        "transaction_cost": tc,
        "exposure": exposure,
        "turnover": turnover,
        "equity": cfg.capital_initial * (1.0 + returns).cumprod(),
    }


def _slice_object(obj: Dict[str, Any], start: pd.Timestamp, end: pd.Timestamp, cfg: Mahoraga14Config) -> Dict[str, Any]:
    returns = pd.Series(obj["returns"], dtype=float).loc[start:end]
    gross = pd.Series(obj.get("gross_returns", returns), dtype=float).reindex(returns.index).fillna(returns)
    tc = pd.Series(obj.get("transaction_cost", pd.Series(0.0, index=returns.index)), dtype=float).reindex(returns.index).fillna(0.0)
    exposure = pd.Series(obj["exposure"], dtype=float).reindex(returns.index).fillna(0.0)
    turnover = pd.Series(obj["turnover"], dtype=float).reindex(returns.index).fillna(0.0)
    return {
        "returns": returns,
        "gross_returns": gross,
        "transaction_cost": tc,
        "exposure": exposure,
        "turnover": turnover,
        "equity": cfg.capital_initial * (1.0 + returns).cumprod(),
    }


def _stitched_base_context(wf: Dict[str, Any], cfg: Mahoraga14Config) -> Dict[str, Any]:
    primary = _object_from_existing(wf["stitched_variants"][cfg.primary_variant_key], cfg)
    control = _object_from_existing(wf["stitched_variants"][cfg.control_variant_key], cfg)
    qqq = _object_from_existing(wf["stitched_benchmarks"]["QQQ"], cfg)
    spy = _object_from_existing(wf["stitched_benchmarks"]["SPY"], cfg)
    allocator = d14._stitch_variant_frame(wf["results"], cfg.primary_variant_key, "allocator_daily").reindex(primary["returns"].index).ffill()
    leader_diag = d14._stitch_variant_frame(wf["results"], cfg.primary_variant_key, "bull_diagnostics").reindex(primary["returns"].index).ffill()
    scorecard = d14._build_bull_window_scorecard(wf, cfg)
    continuation_diag = d14._build_continuation_diagnostic(wf, cfg)
    return {
        "primary": primary,
        "control": control,
        "qqq": qqq,
        "spy": spy,
        "allocator": allocator,
        "leader_diag": leader_diag,
        "scorecard": scorecard,
        "continuation_diag": continuation_diag,
    }


def _candidate_id(
    budget_mult: float,
    conviction_mult: float,
    leader_mult: float,
    backoff_mult: float,
) -> str:
    return f"B{budget_mult:.2f}_C{conviction_mult:.2f}_L{leader_mult:.2f}_R{backoff_mult:.2f}"


def _knob_grid(cfg: Mahoraga14Config) -> List[Dict[str, float]]:
    rows = []
    for budget_mult in cfg.acceptance_budget_multipliers:
        for conviction_mult in cfg.acceptance_conviction_multipliers:
            for leader_mult in cfg.acceptance_leader_multipliers:
                for backoff_mult in cfg.acceptance_backoff_multipliers:
                    rows.append(
                        {
                            "candidate_id": _candidate_id(budget_mult, conviction_mult, leader_mult, backoff_mult),
                            "budget_multiplier": float(budget_mult),
                            "conviction_multiplier": float(conviction_mult),
                            "leader_multiplier": float(leader_mult),
                            "backoff_strength": float(backoff_mult),
                        }
                    )
    return rows


def _apply_frozen_knobs(
    base_obj: Dict[str, Any],
    allocator: pd.DataFrame,
    leader_diag: pd.DataFrame,
    qqq_obj: Dict[str, Any],
    cfg: Mahoraga14Config,
    budget_multiplier: float,
    conviction_multiplier: float,
    leader_multiplier: float,
    backoff_strength: float,
) -> Dict[str, Any]:
    idx = pd.DatetimeIndex(base_obj["returns"].index)
    returns = pd.Series(base_obj["returns"], dtype=float).reindex(idx).fillna(0.0)
    gross = pd.Series(base_obj["gross_returns"], dtype=float).reindex(idx).fillna(returns)
    tc = pd.Series(base_obj["transaction_cost"], dtype=float).reindex(idx).fillna(0.0)
    exposure = pd.Series(base_obj["exposure"], dtype=float).reindex(idx).fillna(0.0)
    turnover = pd.Series(base_obj["turnover"], dtype=float).reindex(idx).fillna(0.0)
    qqq_r = pd.Series(qqq_obj["returns"], dtype=float).reindex(idx).fillna(0.0)

    cash_target_default = float(cfg.participation_allocator_cash_target_floor)
    budget = _series(allocator.get("long_budget", cfg.participation_long_budget_base), idx, cfg.participation_long_budget_base).clip(0.0, 1.0)
    cash_target = _series(allocator.get("cash_budget_target", budget), idx, cash_target_default).clip(0.0, 1.0)
    conviction = _series(allocator.get("conviction_multiplier", 1.0), idx, 1.0).clip(1.0, float(cfg.conviction_weight_scale_max))
    leader_blend = _series(allocator.get("leader_blend", 0.0), idx, 0.0).clip(0.0, float(cfg.participation_leader_blend_max))
    backoff = _series(allocator.get("risk_backoff_score", 0.0), idx, 0.0).clip(0.0, 1.0)
    leader_active = _series(leader_diag.get("leader_active_weight", 0.0), idx, 0.0).clip(0.0, 1.0)

    budget_scale = (np.minimum(1.0, cash_target * float(budget_multiplier)) / cash_target.replace(0.0, np.nan)).clip(0.90, 1.10).fillna(1.0)
    conviction_signal = ((conviction - 1.0) / max(1e-6, float(cfg.conviction_weight_scale_max) - 1.0)).clip(0.0, 1.0)
    leader_signal = (0.60 * leader_blend + 0.40 * leader_active).clip(0.0, 1.0)
    backoff_signal = backoff.clip(0.0, 1.0)
    qqq_up = (qqq_r > 0.0).astype(float)
    qqq_down = (qqq_r < 0.0).astype(float)

    upside_scale = (
        1.0
        + 0.75 * (budget_scale - 1.0)
        + 0.18 * (float(conviction_multiplier) - 1.0) * conviction_signal
        + 0.16 * (float(leader_multiplier) - 1.0) * leader_signal
        - 0.22 * (float(backoff_strength) - 1.0) * backoff_signal
        + 0.05 * (float(leader_multiplier) - 1.0) * leader_signal * qqq_up
    ).clip(0.80, 1.25)
    downside_scale = (
        1.0
        + 0.40 * (budget_scale - 1.0)
        + 0.06 * (float(conviction_multiplier) - 1.0) * conviction_signal
        + 0.05 * (float(leader_multiplier) - 1.0) * leader_signal
        - 0.30 * (float(backoff_strength) - 1.0) * backoff_signal
        - 0.05 * (float(backoff_strength) - 1.0) * backoff_signal * qqq_down
    ).clip(0.75, 1.20)

    gross_new = gross.clip(lower=0.0) * upside_scale + gross.clip(upper=0.0) * downside_scale
    cost_scale = 1.0 + 0.35 * abs(float(budget_multiplier) - 1.0) + 0.25 * abs(float(conviction_multiplier) - 1.0) + 0.25 * abs(float(leader_multiplier) - 1.0) + 0.15 * abs(float(backoff_strength) - 1.0)
    returns_new = gross_new - tc.abs() * cost_scale

    exposure_scale = (
        1.0
        + 0.65 * (budget_scale - 1.0)
        + 0.10 * (float(conviction_multiplier) - 1.0) * conviction_signal
        + 0.10 * (float(leader_multiplier) - 1.0) * leader_signal
        - 0.20 * (float(backoff_strength) - 1.0) * backoff_signal
    ).clip(0.82, 1.18)
    turnover_scale = 1.0 + 0.40 * abs(float(budget_multiplier) - 1.0) + 0.25 * abs(float(conviction_multiplier) - 1.0) + 0.25 * abs(float(leader_multiplier) - 1.0) + 0.15 * abs(float(backoff_strength) - 1.0)
    exposure_new = (exposure * exposure_scale).clip(0.0, 1.20)
    turnover_new = turnover * turnover_scale

    return {
        "returns": returns_new,
        "gross_returns": gross_new,
        "transaction_cost": tc.abs() * cost_scale,
        "exposure": exposure_new,
        "turnover": turnover_new,
        "equity": cfg.capital_initial * (1.0 + returns_new).cumprod(),
        "budget_signal_mean": float(budget_scale.mean()),
        "conviction_signal_mean": float(conviction_signal.mean()),
        "leader_signal_mean": float(leader_signal.mean()),
        "backoff_signal_mean": float(backoff_signal.mean()),
    }


def _priority_windows(scorecard_df: pd.DataFrame, cfg: Mahoraga14Config) -> pd.DataFrame:
    out = scorecard_df.copy()
    return out[out["Window"].isin(list(cfg.acceptance_priority_windows))].copy().reset_index(drop=True)


def _window_metrics_for_object(
    candidate_obj: Dict[str, Any],
    control_obj: Dict[str, Any],
    qqq_obj: Dict[str, Any],
    spy_obj: Dict[str, Any],
    cfg: Mahoraga14Config,
    priority_specs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, spec in priority_specs.iterrows():
        start = pd.Timestamp(spec["Start"])
        end = pd.Timestamp(spec["End"])
        cand_slice = _slice_object(candidate_obj, start, end, cfg)
        ctrl_slice = _slice_object(control_obj, start, end, cfg)
        qqq_slice = _slice_object(qqq_obj, start, end, cfg)
        spy_slice = _slice_object(spy_obj, start, end, cfg)
        cand_vs_qqq = d14._window_summary(cand_slice, qqq_slice, cfg, str(spec["Window"]))
        cand_vs_spy = d14._window_summary(cand_slice, spy_slice, cfg, str(spec["Window"]))
        ctrl_vs_qqq = d14._window_summary(ctrl_slice, qqq_slice, cfg, str(spec["Window"]))
        qqq_ret = float(np.prod(1.0 + qqq_slice["returns"].values) - 1.0)
        spy_ret = float(np.prod(1.0 + spy_slice["returns"].values) - 1.0)
        rows.append(
            {
                "Window": str(spec["Window"]),
                "Start": start,
                "End": end,
                "CandidateReturn": cand_vs_qqq["Return"],
                "ControlReturn": ctrl_vs_qqq["Return"],
                "QQQReturn": qqq_ret,
                "SPYReturn": spy_ret,
                "DeltaReturn_vs_Control": cand_vs_qqq["Return"] - ctrl_vs_qqq["Return"],
                "DeltaReturn_vs_QQQ": cand_vs_qqq["Return"] - qqq_ret,
                "DeltaReturn_vs_SPY": cand_vs_spy["Return"] - spy_ret,
                "SharpeLocal": cand_vs_qqq["Sharpe"],
                "SortinoLocal": cand_vs_qqq["Sortino"],
                "MaxDDLocal": cand_vs_qqq["MaxDD"],
                "BetaQQQLocal": cand_vs_qqq["Beta"],
                "ExposureLocal": cand_vs_qqq["Exposure"],
            }
        )
    return pd.DataFrame(rows)


def _candidate_metrics_row(
    candidate_id: str,
    knobs: Dict[str, float],
    candidate_obj: Dict[str, Any],
    control_obj: Dict[str, Any],
    qqq_obj: Dict[str, Any],
    spy_obj: Dict[str, Any],
    cfg: Mahoraga14Config,
    priority_specs: pd.DataFrame,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    metrics = d14._metrics_row(candidate_id, candidate_obj, qqq_obj, spy_obj, cfg)
    control_metrics = d14._metrics_row("CONTROL", control_obj, qqq_obj, spy_obj, cfg)
    windows = _window_metrics_for_object(candidate_obj, control_obj, qqq_obj, spy_obj, cfg, priority_specs)
    priority_avg_delta_control = float(windows["DeltaReturn_vs_Control"].mean()) if len(windows) else 0.0
    priority_min_delta_control = float(windows["DeltaReturn_vs_Control"].min()) if len(windows) else 0.0
    priority_avg_delta_qqq = float(windows["DeltaReturn_vs_QQQ"].mean()) if len(windows) else 0.0
    bull_window_score = priority_avg_delta_control - 0.50 * abs(min(0.0, priority_min_delta_control)) + 0.25 * priority_avg_delta_qqq
    row = {
        "CandidateId": candidate_id,
        **knobs,
        "CAGR": float(metrics["CAGR"]),
        "Sharpe": float(metrics["Sharpe"]),
        "Sortino": float(metrics["Sortino"]),
        "MaxDD": float(metrics["MaxDD"]),
        "AlphaNW_QQQ": float(metrics["AlphaNW_QQQ"]),
        "AlphaNW_SPY": float(metrics["AlphaNW_SPY"]),
        "UpsideCaptureQQQ": float(metrics["UpsideCaptureQQQ"]),
        "DeltaCAGR_vs_Control": float(metrics["CAGR"] - control_metrics["CAGR"]),
        "DeltaSharpe_vs_Control": float(metrics["Sharpe"] - control_metrics["Sharpe"]),
        "DeltaSortino_vs_Control": float(metrics["Sortino"] - control_metrics["Sortino"]),
        "DeltaMaxDD_vs_Control": float(metrics["MaxDD"] - control_metrics["MaxDD"]),
        "DeltaAlphaNW_QQQ_vs_Control": float(metrics["AlphaNW_QQQ"] - control_metrics["AlphaNW_QQQ"]),
        "DeltaAlphaNW_SPY_vs_Control": float(metrics["AlphaNW_SPY"] - control_metrics["AlphaNW_SPY"]),
        "DeltaUpsideCaptureQQQ_vs_Control": float(metrics["UpsideCaptureQQQ"] - control_metrics["UpsideCaptureQQQ"]),
        "PriorityAvgDelta_vs_Control": priority_avg_delta_control,
        "PriorityMinDelta_vs_Control": priority_min_delta_control,
        "PriorityAvgDelta_vs_QQQ": priority_avg_delta_qqq,
        "BullWindowScore": bull_window_score,
        "AvgExposure": float(pd.Series(candidate_obj["exposure"], dtype=float).mean()),
        "AvgTurnover": float(pd.Series(candidate_obj["turnover"], dtype=float).mean()),
    }
    return row, windows


def _candidate_family(
    wf: Dict[str, Any],
    cfg: Mahoraga14Config,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], Dict[str, pd.DataFrame], Dict[str, Any]]:
    context = _stitched_base_context(wf, cfg)
    primary = context["primary"]
    control = context["control"]
    qqq = context["qqq"]
    spy = context["spy"]
    allocator = context["allocator"]
    leader_diag = context["leader_diag"]
    priority_specs = _priority_windows(context["scorecard"], cfg)

    rows: List[Dict[str, Any]] = []
    objects: Dict[str, Dict[str, Any]] = {}
    windows_map: Dict[str, pd.DataFrame] = {}
    for knobs in _knob_grid(cfg):
        candidate_id = str(knobs["candidate_id"])
        obj = _apply_frozen_knobs(
            primary,
            allocator,
            leader_diag,
            qqq,
            cfg,
            knobs["budget_multiplier"],
            knobs["conviction_multiplier"],
            knobs["leader_multiplier"],
            knobs["backoff_strength"],
        )
        row, windows = _candidate_metrics_row(candidate_id, knobs, obj, control, qqq, spy, cfg, priority_specs)
        row["BudgetSignalMean"] = obj["budget_signal_mean"]
        row["ConvictionSignalMean"] = obj["conviction_signal_mean"]
        row["LeaderSignalMean"] = obj["leader_signal_mean"]
        row["BackoffSignalMean"] = obj["backoff_signal_mean"]
        rows.append(row)
        objects[candidate_id] = obj
        windows_map[candidate_id] = windows

    grid = pd.DataFrame(rows)
    higher_better = [
        "CAGR",
        "Sharpe",
        "Sortino",
        "MaxDD",
        "AlphaNW_QQQ",
        "AlphaNW_SPY",
        "UpsideCaptureQQQ",
        "PriorityAvgDelta_vs_Control",
        "PriorityMinDelta_vs_Control",
        "BullWindowScore",
    ]
    for col in higher_better:
        grid[f"{col}_PctRank"] = grid[col].rank(pct=True, method="average")
    grid["ConservativeScore"] = grid[[f"{c}_PctRank" for c in higher_better]].mean(axis=1)
    grid["FragilityPenalty"] = (
        0.10 * (grid["PriorityMinDelta_vs_Control"] < -0.01).astype(float)
        + 0.08 * (grid["DeltaSharpe_vs_Control"] < 0.0).astype(float)
        + 0.08 * (grid["DeltaCAGR_vs_Control"] < 0.0).astype(float)
        + 0.08 * (grid["DeltaMaxDD_vs_Control"] < -2.0).astype(float)
    )
    grid["RobustScore"] = grid["ConservativeScore"] - grid["FragilityPenalty"]
    grid["RobustCandidate"] = (
        (grid["DeltaCAGR_vs_Control"] >= 0.0)
        & (grid["DeltaSharpe_vs_Control"] >= 0.0)
        & (grid["DeltaSortino_vs_Control"] >= 0.0)
        & (grid["DeltaMaxDD_vs_Control"] >= -2.0)
        & (grid["PriorityMinDelta_vs_Control"] >= -0.01)
        & (grid["DeltaAlphaNW_QQQ_vs_Control"] >= -1e-9)
    ).astype(int)
    grid["BaseCurrentCandidate"] = (grid["CandidateId"] == _candidate_id(1.0, 1.0, 1.0, 1.0)).astype(int)
    grid = grid.sort_values(["RobustCandidate", "RobustScore", "ConservativeScore"], ascending=[False, False, False]).reset_index(drop=True)
    return grid.round(8), objects, windows_map, context


def _parameter_inventory(wf: Dict[str, Any], cfg: Mahoraga14Config) -> pd.DataFrame:
    selected = wf.get("selected_df", pd.DataFrame()).copy()
    rows = [
        {"Parameter": "base_mix", "CurrentValue": float(selected["base_mix"].median()) if "base_mix" in selected else np.nan, "ParameterType": "structural", "KnobGroup": "frozen_engine", "AcceptanceRole": "frozen", "Notes": "selected by 14.3 WFO; not retuned in 14.3R"},
        {"Parameter": "defense_mix", "CurrentValue": float(selected["defense_mix"].median()) if "defense_mix" in selected else np.nan, "ParameterType": "structural", "KnobGroup": "frozen_engine", "AcceptanceRole": "frozen", "Notes": "selected by 14.3 WFO; not retuned in 14.3R"},
        {"Parameter": "raw_rel_boost", "CurrentValue": float(selected["raw_rel_boost"].median()) if "raw_rel_boost" in selected else np.nan, "ParameterType": "structural", "KnobGroup": "frozen_engine", "AcceptanceRole": "frozen", "Notes": "engine boost inherited from 14.3"},
        {"Parameter": "structural_enter_thr", "CurrentValue": float(selected["structural_enter_thr"].median()) if "structural_enter_thr" in selected else np.nan, "ParameterType": "structural", "KnobGroup": "frozen_override", "AcceptanceRole": "frozen", "Notes": "structural defense threshold remains fixed"},
        {"Parameter": "hawkes_weight", "CurrentValue": float(selected["hawkes_weight"].median()) if "hawkes_weight" in selected else np.nan, "ParameterType": "structural", "KnobGroup": "frozen_override", "AcceptanceRole": "frozen", "Notes": "continuation/transition context frozen"},
        {"Parameter": "structural_blend", "CurrentValue": float(selected["structural_blend"].median()) if "structural_blend" in selected else np.nan, "ParameterType": "structural", "KnobGroup": "frozen_override", "AcceptanceRole": "frozen", "Notes": "structural defense blend frozen"},
        {"Parameter": "participation_long_budget_base", "CurrentValue": float(cfg.participation_long_budget_base), "ParameterType": "scale", "KnobGroup": "budget_multiplier", "AcceptanceRole": "primary_knob", "Notes": "budget center"},
        {"Parameter": "participation_long_budget_floor", "CurrentValue": float(cfg.participation_long_budget_floor), "ParameterType": "scale", "KnobGroup": "budget_multiplier", "AcceptanceRole": "primary_knob", "Notes": "budget lower bound"},
        {"Parameter": "participation_long_budget_ceiling", "CurrentValue": float(cfg.participation_long_budget_ceiling), "ParameterType": "scale", "KnobGroup": "budget_multiplier", "AcceptanceRole": "primary_knob", "Notes": "budget upper bound"},
        {"Parameter": "participation_allocator_cash_target_floor", "CurrentValue": float(cfg.participation_allocator_cash_target_floor), "ParameterType": "scale", "KnobGroup": "budget_multiplier", "AcceptanceRole": "primary_knob", "Notes": "cash drag redeployment floor"},
        {"Parameter": "participation_allocator_cash_target_ceiling", "CurrentValue": float(cfg.participation_allocator_cash_target_ceiling), "ParameterType": "scale", "KnobGroup": "budget_multiplier", "AcceptanceRole": "primary_knob", "Notes": "cash drag redeployment ceiling"},
        {"Parameter": "conviction_max_budget_boost", "CurrentValue": float(cfg.conviction_max_budget_boost), "ParameterType": "scale", "KnobGroup": "conviction_multiplier", "AcceptanceRole": "primary_knob", "Notes": "budget lift from conviction"},
        {"Parameter": "conviction_gate_boost_max", "CurrentValue": float(cfg.conviction_gate_boost_max), "ParameterType": "scale", "KnobGroup": "conviction_multiplier", "AcceptanceRole": "primary_knob", "Notes": "gate lift from conviction"},
        {"Parameter": "conviction_exp_boost_max", "CurrentValue": float(cfg.conviction_exp_boost_max), "ParameterType": "scale", "KnobGroup": "conviction_multiplier", "AcceptanceRole": "primary_knob", "Notes": "exp cap lift from conviction"},
        {"Parameter": "conviction_weight_scale_max", "CurrentValue": float(cfg.conviction_weight_scale_max), "ParameterType": "scale", "KnobGroup": "conviction_multiplier", "AcceptanceRole": "primary_knob", "Notes": "effective weight amplification"},
        {"Parameter": "participation_leader_blend_max", "CurrentValue": float(cfg.participation_leader_blend_max), "ParameterType": "scale", "KnobGroup": "leader_multiplier", "AcceptanceRole": "primary_knob", "Notes": "max blend toward leader engine"},
        {"Parameter": "conviction_leader_boost_max", "CurrentValue": float(cfg.conviction_leader_boost_max), "ParameterType": "scale", "KnobGroup": "leader_multiplier", "AcceptanceRole": "primary_knob", "Notes": "leader amplification"},
        {"Parameter": "leader_tilt_strength_max", "CurrentValue": float(cfg.leader_tilt_strength_max), "ParameterType": "scale", "KnobGroup": "leader_multiplier", "AcceptanceRole": "primary_knob", "Notes": "leader tilt strength"},
        {"Parameter": "risk_backoff_budget_floor", "CurrentValue": float(cfg.risk_backoff_budget_floor), "ParameterType": "scale", "KnobGroup": "backoff_strength", "AcceptanceRole": "primary_knob", "Notes": "budget floor under stress"},
        {"Parameter": "risk_backoff_gate_floor", "CurrentValue": float(cfg.risk_backoff_gate_floor), "ParameterType": "scale", "KnobGroup": "backoff_strength", "AcceptanceRole": "primary_knob", "Notes": "gate floor under stress"},
        {"Parameter": "risk_backoff_exp_floor", "CurrentValue": float(cfg.risk_backoff_exp_floor), "ParameterType": "scale", "KnobGroup": "backoff_strength", "AcceptanceRole": "primary_knob", "Notes": "exp cap floor under stress"},
        {"Parameter": "risk_backoff_hard_break_risk", "CurrentValue": float(cfg.risk_backoff_hard_break_risk), "ParameterType": "scale", "KnobGroup": "backoff_strength", "AcceptanceRole": "primary_knob", "Notes": "hard break-risk trigger"},
        {"Parameter": "risk_backoff_hard_fragility", "CurrentValue": float(cfg.risk_backoff_hard_fragility), "ParameterType": "scale", "KnobGroup": "backoff_strength", "AcceptanceRole": "primary_knob", "Notes": "hard fragility trigger"},
        {"Parameter": "continuation_target_rate", "CurrentValue": float(cfg.continuation_target_rate), "ParameterType": "structural", "KnobGroup": "continuation_filter", "AcceptanceRole": "diagnostic_only", "Notes": "continuation stays diagnostic; not tuned in 14.3R"},
        {"Parameter": "continuation_pressure_floor_quantile", "CurrentValue": float(cfg.continuation_pressure_floor_quantile), "ParameterType": "structural", "KnobGroup": "continuation_filter", "AcceptanceRole": "diagnostic_only", "Notes": "continuation calibration inherited from 14.3"},
    ]
    return pd.DataFrame(rows)


def _parameter_freeze_plan_md(inventory_df: pd.DataFrame) -> str:
    lines = [
        "# Mahoraga14_3R Parameter Freeze Plan",
        "",
        "## Institutional freeze rule",
        "- Freeze the 14.3 architecture exactly as implemented.",
        "- Reduce acceptance review to four interpretable scale knobs.",
        "- Do not retune structural engine / ML / continuation model families inside 14.3R.",
        "",
        "## Primary knobs",
        "- `budget_multiplier`: scales effective long budget and cash redeployment around the frozen 14.3 schedule.",
        "- `conviction_multiplier`: scales conviction translation already present in 14.3.",
        "- `leader_multiplier`: scales leader participation already present in 14.3.",
        "- `backoff_strength`: scales how aggressively 14.3 backs off under fragility/break-risk.",
        "",
        "## Frozen parameters",
        "- Engine mixture, structural policy, continuation model family and all short logic remain frozen.",
        "- Continuation is reviewed only as a quality filter; it is not promoted to a new tuning axis in 14.3R.",
        "",
        "## Selection discipline",
        "- Acceptance prefers a plateau candidate over a point-optimal candidate.",
        "- If the current 14.3 point is not on a stable plateau, 14.1 remains the institutional baseline.",
    ]
    return "\n".join(lines)


def _priority_acceptance_table(
    base_candidate_id: str,
    windows_df: pd.DataFrame,
    cfg: Mahoraga14Config,
) -> pd.DataFrame:
    out = windows_df.copy()
    out["CandidateId"] = base_candidate_id
    out["VsControlStatus"] = np.where(out["DeltaReturn_vs_Control"] > 0.005, "IMPROVES", np.where(out["DeltaReturn_vs_Control"] < -0.005, "WORSENS", "NEAR_FLAT"))
    out["VsQQQStatus"] = np.where(out["DeltaReturn_vs_QQQ"] > 0.0, "BEATS_QQQ", np.where(out["DeltaReturn_vs_QQQ"] < -0.05, "MATERIALLY_BELOW_QQQ", "MODERATELY_BELOW_QQQ"))
    out["CompensationStatus"] = np.where(
        (out["DeltaReturn_vs_Control"] >= -0.005) & (out["SharpeLocal"] >= 1.0) & (out["MaxDDLocal"] >= -0.18),
        "ACCEPTABLE_COMPENSATION",
        "NOT_ACCEPTABLE",
    )
    out["PriorityDecision"] = np.where(
        (out["VsControlStatus"] == "IMPROVES") | ((out["VsControlStatus"] == "NEAR_FLAT") & (out["CompensationStatus"] == "ACCEPTABLE_COMPENSATION")),
        "PASS_OR_TOLERABLE",
        "FAIL_PRIORITY_WINDOW",
    )
    cols = [
        "CandidateId",
        "Window",
        "Start",
        "End",
        "CandidateReturn",
        "ControlReturn",
        "QQQReturn",
        "SPYReturn",
        "DeltaReturn_vs_Control",
        "DeltaReturn_vs_QQQ",
        "DeltaReturn_vs_SPY",
        "SharpeLocal",
        "SortinoLocal",
        "MaxDDLocal",
        "BetaQQQLocal",
        "ExposureLocal",
        "VsControlStatus",
        "VsQQQStatus",
        "CompensationStatus",
        "PriorityDecision",
    ]
    return out[cols].round(8)


def _priority_acceptance_notes_md(priority_df: pd.DataFrame) -> str:
    lines = ["# Priority Window Acceptance", ""]
    for _, row in priority_df.iterrows():
        lines.append(f"## {row['Window']}")
        lines.append(f"- vs control: {row['VsControlStatus']} ({row['DeltaReturn_vs_Control']:.4f})")
        lines.append(f"- vs QQQ: {row['VsQQQStatus']} ({row['DeltaReturn_vs_QQQ']:.4f})")
        lines.append(f"- compensation: {row['CompensationStatus']}")
        lines.append(f"- decision: {row['PriorityDecision']}")
        lines.append("")
    return "\n".join(lines)


def _selection_score_from_subset(
    family_df: pd.DataFrame,
    candidate_windows: Dict[str, pd.DataFrame],
    excluded_window: str | None,
) -> pd.DataFrame:
    grid = family_df.copy()
    if excluded_window is None:
        window_cols = ["PriorityAvgDelta_vs_Control", "PriorityMinDelta_vs_Control", "BullWindowScore"]
        base = grid.copy()
    else:
        rows = []
        for _, row in grid.iterrows():
            windows = candidate_windows[str(row["CandidateId"])]
            windows = windows[windows["Window"] != excluded_window].copy()
            priority_avg = float(windows["DeltaReturn_vs_Control"].mean()) if len(windows) else 0.0
            priority_min = float(windows["DeltaReturn_vs_Control"].min()) if len(windows) else 0.0
            bull_score = priority_avg - 0.50 * abs(min(0.0, priority_min)) + 0.25 * (float(windows["DeltaReturn_vs_QQQ"].mean()) if len(windows) else 0.0)
            payload = row.to_dict()
            payload["PriorityAvgDelta_vs_Control_ex"] = priority_avg
            payload["PriorityMinDelta_vs_Control_ex"] = priority_min
            payload["BullWindowScore_ex"] = bull_score
            rows.append(payload)
        base = pd.DataFrame(rows)
        window_cols = ["PriorityAvgDelta_vs_Control_ex", "PriorityMinDelta_vs_Control_ex", "BullWindowScore_ex"]

    metrics = [
        "CAGR",
        "Sharpe",
        "Sortino",
        "MaxDD",
        "AlphaNW_QQQ",
        "AlphaNW_SPY",
        "UpsideCaptureQQQ",
        *window_cols,
    ]
    for col in metrics:
        base[f"{col}_SelRank"] = base[col].rank(pct=True, method="average")
    base["SelectionScore"] = base[[f"{c}_SelRank" for c in metrics]].mean(axis=1)
    base["SelectionPenalty"] = (
        0.10 * (base[window_cols[1]] < -0.01).astype(float)
        + 0.08 * (base["DeltaSharpe_vs_Control"] < 0.0).astype(float)
        + 0.08 * (base["DeltaCAGR_vs_Control"] < 0.0).astype(float)
    )
    base["SelectionScoreNet"] = base["SelectionScore"] - base["SelectionPenalty"]
    return base.sort_values(["RobustCandidate", "SelectionScoreNet"], ascending=[False, False]).reset_index(drop=True)


def _leave_one_window_out(
    family_df: pd.DataFrame,
    candidate_objects: Dict[str, Dict[str, Any]],
    candidate_windows: Dict[str, pd.DataFrame],
    context: Dict[str, Any],
    cfg: Mahoraga14Config,
) -> pd.DataFrame:
    rows = []
    base_candidate_id = _candidate_id(1.0, 1.0, 1.0, 1.0)
    for window_name in cfg.acceptance_priority_windows:
        selection_df = _selection_score_from_subset(family_df, candidate_windows, window_name)
        chosen = selection_df.iloc[0]
        candidate_id = str(chosen["CandidateId"])
        excluded_metrics = candidate_windows[candidate_id].set_index("Window").loc[window_name]
        base_excluded = candidate_windows[base_candidate_id].set_index("Window").loc[window_name]
        rows.append(
            {
                "ExcludedWindow": window_name,
                "SelectedCandidateId": candidate_id,
                "SelectedBudgetMultiplier": float(chosen["budget_multiplier"]),
                "SelectedConvictionMultiplier": float(chosen["conviction_multiplier"]),
                "SelectedLeaderMultiplier": float(chosen["leader_multiplier"]),
                "SelectedBackoffStrength": float(chosen["backoff_strength"]),
                "SelectionScoreNet": float(chosen["SelectionScoreNet"]),
                "BaseCurrentRank": int(selection_df.index[selection_df["CandidateId"] == base_candidate_id][0] + 1),
                "EvalCandidateReturn": float(excluded_metrics["CandidateReturn"]),
                "EvalControlReturn": float(excluded_metrics["ControlReturn"]),
                "EvalQQQReturn": float(excluded_metrics["QQQReturn"]),
                "EvalDelta_vs_Control": float(excluded_metrics["DeltaReturn_vs_Control"]),
                "EvalDelta_vs_QQQ": float(excluded_metrics["DeltaReturn_vs_QQQ"]),
                "EvalSharpeLocal": float(excluded_metrics["SharpeLocal"]),
                "EvalSortinoLocal": float(excluded_metrics["SortinoLocal"]),
                "EvalMaxDDLocal": float(excluded_metrics["MaxDDLocal"]),
                "EvalExposureLocal": float(excluded_metrics["ExposureLocal"]),
                "BaseCurrentEvalDelta_vs_Control": float(base_excluded["DeltaReturn_vs_Control"]),
                "BaseCurrentEvalDelta_vs_QQQ": float(base_excluded["DeltaReturn_vs_QQQ"]),
            }
        )
    return pd.DataFrame(rows).round(8)


def _leave_one_window_out_md(df: pd.DataFrame) -> str:
    lines = ["# Leave-One-Window-Out Summary", ""]
    for _, row in df.iterrows():
        lines.append(f"## Excluded {row['ExcludedWindow']}")
        lines.append(f"- selected candidate: {row['SelectedCandidateId']}")
        lines.append(f"- selection score: {row['SelectionScoreNet']:.4f}")
        lines.append(f"- excluded-window delta vs control: {row['EvalDelta_vs_Control']:.4f}")
        lines.append(f"- excluded-window delta vs QQQ: {row['EvalDelta_vs_QQQ']:.4f}")
        lines.append(f"- base-current rank without that window: {int(row['BaseCurrentRank'])}")
        lines.append("")
    return "\n".join(lines)


def _continuation_acceptance_table(continuation_diag_df: pd.DataFrame) -> pd.DataFrame:
    stitched = continuation_diag_df[continuation_diag_df["Segment"] == "STITCHED"].copy()
    primary = stitched[stitched["Variant"] == "MAHORAGA14_3_LONG_PARTICIPATION"].copy()
    if len(primary) == 0:
        return pd.DataFrame()
    out = primary.copy()
    role = "A. continuation se mantiene como filtro de calidad"
    if float(out["EdgeVsNoActivation4W"].iloc[0]) <= 0.0 or float(out["HitRate4W"].iloc[0]) <= float(out["NoActHitRate4W"].iloc[0]):
        role = "C. continuation no aporta lo suficiente y debe neutralizarse en futuras versiones"
    elif float(out["ActivationRate"].iloc[0]) < 0.03:
        role = "B. continuation se reduce de peso"
    out["FinalRoleDecision"] = role
    out["Interpretation"] = np.where(
        (out["HitRate4W"] > out["NoActHitRate4W"]) & (out["EdgeVsNoActivation4W"] > 0.0),
        "positive_local_edge",
        "weak_or_negative_edge",
    )
    return out.round(8)


def _continuation_acceptance_notes_md(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "# Continuation Acceptance\n\nNo continuation diagnostic available."
    row = df.iloc[0]
    lines = [
        "# Continuation Acceptance",
        "",
        f"- activation rate: {row['ActivationRate']:.4f}",
        f"- hit rate 4W: {row['HitRate4W']:.4f}",
        f"- no-activation hit rate 4W: {row['NoActHitRate4W']:.4f}",
        f"- edge vs no activation 4W: {row['EdgeVsNoActivation4W']:.4f}",
        f"- role decision: {row['FinalRoleDecision']}",
        "- conclusion: continuation still looks more useful as a quality filter than as the main participation lever.",
    ]
    return "\n".join(lines)


def _stationary_bootstrap_indices(length: int, block: int, samples: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    out: List[np.ndarray] = []
    p = 1.0 / max(1, int(block))
    for _ in range(int(samples)):
        idx = np.empty(length, dtype=int)
        cur = int(rng.integers(0, length))
        for i in range(length):
            if i == 0 or rng.random() < p:
                cur = int(rng.integers(0, length))
            else:
                cur = (cur + 1) % length
            idx[i] = cur
        out.append(idx)
    return out


def _model_selection_guard(
    family_df: pd.DataFrame,
    candidate_objects: Dict[str, Dict[str, Any]],
    control_obj: Dict[str, Any],
    cfg: Mahoraga14Config,
) -> Tuple[pd.DataFrame, str]:
    candidate_ids = list(family_df["CandidateId"])
    excess = np.column_stack(
        [
            pd.Series(candidate_objects[cid]["returns"], dtype=float).reindex(control_obj["returns"].index).fillna(0.0).values
            - pd.Series(control_obj["returns"], dtype=float).values
            for cid in candidate_ids
        ]
    )
    observed_mean = excess.mean(axis=0)
    centered = excess - observed_mean
    indices = _stationary_bootstrap_indices(len(control_obj["returns"]), cfg.acceptance_bootstrap_block, cfg.acceptance_model_selection_bootstrap_samples, seed=41)
    base_candidate_id = _candidate_id(1.0, 1.0, 1.0, 1.0)
    base_idx = candidate_ids.index(base_candidate_id)
    boot_max = []
    boot_base = []
    for idx in indices:
        sample = centered[idx, :]
        boot_mean = sample.mean(axis=0)
        boot_max.append(float(boot_mean.max()))
        boot_base.append(float(boot_mean[base_idx]))
    reality_check_p = float(np.mean(np.asarray(boot_max) >= float(observed_mean.max())))
    base_candidate_p = float(np.mean(np.asarray(boot_base) >= float(observed_mean[base_idx])))
    out = family_df[["CandidateId", "budget_multiplier", "conviction_multiplier", "leader_multiplier", "backoff_strength", "RobustCandidate", "RobustScore"]].copy()
    out["MeanDailyExcess_vs_Control"] = observed_mean
    out["MeanExcessRank"] = out["MeanDailyExcess_vs_Control"].rank(pct=True, method="average")
    out["RealityCheckFamilyPValue"] = reality_check_p
    out["BaseCandidatePValue"] = base_candidate_p
    selected_candidate = str(out.sort_values(["RobustCandidate", "RobustScore", "MeanDailyExcess_vs_Control"], ascending=[False, False, False]).iloc[0]["CandidateId"])
    md_lines = [
        "# Model Selection Guard",
        "",
        "- Multiple Mahoraga 14 variants were explored before 14.3; acceptance therefore uses a conservative guard.",
        "- 14.3R only examines a local neighborhood around the frozen 14.3 point, not a new discovery grid.",
        f"- White/Reality-Check style family bootstrap p-value (local family max excess vs control): {reality_check_p:.4f}",
        f"- Base-current candidate bootstrap p-value vs centered null: {base_candidate_p:.4f}",
        f"- Highest robust local candidate under the acceptance family: `{selected_candidate}`.",
        "- Decision discipline: a candidate is not promoted to institutional baseline solely because it is the best point in the local family.",
    ]
    return out.round(8), "\n".join(md_lines)


def _bootstrap_summary_samples(
    candidate_obj: Dict[str, Any],
    control_obj: Dict[str, Any],
    cfg: Mahoraga14Config,
    seed: int = 77,
) -> pd.DataFrame:
    cand_r = pd.Series(candidate_obj["returns"], dtype=float)
    ctrl_r = pd.Series(control_obj["returns"], dtype=float).reindex(cand_r.index).fillna(0.0)
    cand_exp = pd.Series(candidate_obj["exposure"], dtype=float).reindex(cand_r.index).fillna(0.0)
    ctrl_exp = pd.Series(control_obj["exposure"], dtype=float).reindex(cand_r.index).fillna(0.0)
    cand_to = pd.Series(candidate_obj["turnover"], dtype=float).reindex(cand_r.index).fillna(0.0)
    ctrl_to = pd.Series(control_obj["turnover"], dtype=float).reindex(cand_r.index).fillna(0.0)
    indices = _stationary_bootstrap_indices(len(cand_r), cfg.acceptance_bootstrap_block, cfg.acceptance_bootstrap_samples, seed=seed)
    rows = []
    for sample_id, idx in enumerate(indices):
        c_r = pd.Series(cand_r.values[idx], index=cand_r.index)
        k_r = pd.Series(ctrl_r.values[idx], index=ctrl_r.index)
        c_obj = {
            "returns": c_r,
            "equity": cfg.capital_initial * (1.0 + c_r).cumprod(),
            "exposure": pd.Series(cand_exp.values[idx], index=cand_exp.index),
            "turnover": pd.Series(cand_to.values[idx], index=cand_to.index),
        }
        k_obj = {
            "returns": k_r,
            "equity": cfg.capital_initial * (1.0 + k_r).cumprod(),
            "exposure": pd.Series(ctrl_exp.values[idx], index=ctrl_exp.index),
            "turnover": pd.Series(ctrl_to.values[idx], index=ctrl_to.index),
        }
        c_sum = m6.summarize(c_obj["returns"], c_obj["equity"], c_obj["exposure"], c_obj["turnover"], cfg, "boot_cand")
        k_sum = m6.summarize(k_obj["returns"], k_obj["equity"], k_obj["exposure"], k_obj["turnover"], cfg, "boot_ctrl")
        rows.append(
            {
                "SampleId": sample_id,
                "CandidateCAGR": float(c_sum["CAGR"]),
                "CandidateSharpe": float(c_sum["Sharpe"]),
                "CandidateMaxDD": float(c_sum["MaxDD"]),
                "ControlCAGR": float(k_sum["CAGR"]),
                "ControlSharpe": float(k_sum["Sharpe"]),
                "ControlMaxDD": float(k_sum["MaxDD"]),
                "DeltaCAGR": float(c_sum["CAGR"] - k_sum["CAGR"]),
                "DeltaSharpe": float(c_sum["Sharpe"] - k_sum["Sharpe"]),
                "DeltaMaxDD": float(c_sum["MaxDD"] - k_sum["MaxDD"]),
            }
        )
    return pd.DataFrame(rows).round(8)


def _bootstrap_summary(
    candidate_obj: Dict[str, Any],
    control_obj: Dict[str, Any],
    cfg: Mahoraga14Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    samples = _bootstrap_summary_samples(candidate_obj, control_obj, cfg)
    metrics = ["CandidateCAGR", "CandidateSharpe", "CandidateMaxDD", "DeltaCAGR", "DeltaSharpe", "DeltaMaxDD"]
    rows = []
    for metric in metrics:
        s = samples[metric]
        rows.append(
            {
                "Metric": metric,
                "p05": float(s.quantile(0.05)),
                "p25": float(s.quantile(0.25)),
                "p50": float(s.quantile(0.50)),
                "p75": float(s.quantile(0.75)),
                "p95": float(s.quantile(0.95)),
                "mean": float(s.mean()),
            }
        )
    return pd.DataFrame(rows).round(8), samples


def _robustness_notes_md(stress_df: pd.DataFrame, bootstrap_summary_df: pd.DataFrame) -> str:
    lines = ["# Acceptance Robustness Notes", ""]
    non_base = stress_df[stress_df["Scenario"] != "BASELINE"].copy()
    if len(non_base):
        lines.append(f"- worst scenario Sharpe delta: {float(non_base['DeltaSharpe'].min()):.4f}")
        lines.append(f"- worst scenario CAGR delta: {float(non_base['DeltaCAGR%'].min()):.4f} pts")
    if len(bootstrap_summary_df):
        delta_sharpe = bootstrap_summary_df[bootstrap_summary_df["Metric"] == "DeltaSharpe"].iloc[0]
        delta_cagr = bootstrap_summary_df[bootstrap_summary_df["Metric"] == "DeltaCAGR"].iloc[0]
        lines.append(f"- bootstrap DeltaSharpe p25/p50/p75: {delta_sharpe['p25']:.4f} / {delta_sharpe['p50']:.4f} / {delta_sharpe['p75']:.4f}")
        lines.append(f"- bootstrap DeltaCAGR p25/p50/p75: {delta_cagr['p25']:.4f} / {delta_cagr['p50']:.4f} / {delta_cagr['p75']:.4f}")
    lines.append("- interpretation: acceptance requires degradation bands to stay inside a reasonable range, not just a good base-case backtest.")
    return "\n".join(lines)


def _decision(
    comparison_df: pd.DataFrame,
    priority_df: pd.DataFrame,
    family_df: pd.DataFrame,
    loo_df: pd.DataFrame,
    model_guard_df: pd.DataFrame,
    bootstrap_summary_df: pd.DataFrame,
) -> Tuple[str, str]:
    primary = comparison_df[comparison_df["Variant"] == "MAHORAGA14_3_LONG_PARTICIPATION"].iloc[0]
    control = comparison_df[comparison_df["Variant"] == "MAHORAGA14_1_LONG_ONLY_CONTROL"].iloc[0]
    base_id = _candidate_id(1.0, 1.0, 1.0, 1.0)
    base_row = family_df[family_df["CandidateId"] == base_id].iloc[0]
    base_p = float(model_guard_df["BaseCandidatePValue"].iloc[0]) if len(model_guard_df) else 1.0
    priority_fail = bool((priority_df["PriorityDecision"] == "FAIL_PRIORITY_WINDOW").any())
    loo_fail = bool((loo_df["EvalDelta_vs_Control"] < -0.01).any()) if len(loo_df) else True
    bootstrap_delta_sharpe_p25 = float(bootstrap_summary_df[bootstrap_summary_df["Metric"] == "DeltaSharpe"]["p25"].iloc[0]) if len(bootstrap_summary_df) else -np.inf

    accept = (
        float(primary["CAGR"] - control["CAGR"]) > 0.50
        and float(primary["Sharpe"] - control["Sharpe"]) > 0.03
        and float(primary["Sortino"] - control["Sortino"]) > 0.03
        and float(primary["MaxDD"] - control["MaxDD"]) >= -0.50
        and int(base_row["RobustCandidate"]) == 1
        and not priority_fail
        and not loo_fail
        and base_p <= 0.10
        and bootstrap_delta_sharpe_p25 >= 0.0
    )
    if accept:
        decision = "ACCEPT_AS_NEW_BASELINE"
    elif float(primary["CAGR"] - control["CAGR"]) > 0.0 and float(primary["Sharpe"] - control["Sharpe"]) > 0.0 and int(base_row["RobustCandidate"]) == 1:
        decision = "KEEP_AS_EXPERIMENTAL_BRANCH"
    else:
        decision = "REJECT_AND_KEEP_14_1"

    lines = [
        "# Acceptance Decision",
        "",
        f"## Decision: {decision}",
        "",
        "## Why",
        f"- stitched CAGR delta vs control: {float(primary['CAGR'] - control['CAGR']):.4f}",
        f"- stitched Sharpe delta vs control: {float(primary['Sharpe'] - control['Sharpe']):.4f}",
        f"- stitched Sortino delta vs control: {float(primary['Sortino'] - control['Sortino']):.4f}",
        f"- stitched MaxDD delta vs control: {float(primary['MaxDD'] - control['MaxDD']):.4f}",
        f"- base-current local robust flag: {int(base_row['RobustCandidate'])}",
        f"- any priority window fail: {priority_fail}",
        f"- leave-one-window-out fail: {loo_fail}",
        f"- conservative model-selection guard p-value: {base_p:.4f}",
        f"- bootstrap DeltaSharpe p25: {bootstrap_delta_sharpe_p25:.4f}",
        "",
        "## Institutional conclusion",
    ]
    if decision == "ACCEPT_AS_NEW_BASELINE":
        lines.append("- Mahoraga14_3_LONG_PARTICIPATION is accepted as the new long-only institutional baseline.")
    elif decision == "KEEP_AS_EXPERIMENTAL_BRANCH":
        lines.append("- Mahoraga14_3_LONG_PARTICIPATION is promising and robust enough to keep active, but it should NOT replace Mahoraga14_1_LONG_ONLY_CONTROL as the official baseline yet.")
    else:
        lines.append("- Mahoraga14_3_LONG_PARTICIPATION should NOT replace Mahoraga14_1_LONG_ONLY_CONTROL and should be treated as rejected for baseline promotion.")
    return decision, "\n".join(lines)


def _plot_local_stability_heatmap(family_df: pd.DataFrame, out_png: Path) -> None:
    pivot = (
        family_df.groupby(["budget_multiplier", "leader_multiplier"])["RobustScore"]
        .mean()
        .reset_index()
        .pivot(index="budget_multiplier", columns="leader_multiplier", values="RobustScore")
        .sort_index()
    )
    plt.figure(figsize=(7.5, 5.5))
    plt.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    plt.xticks(range(len(pivot.columns)), [f"{x:.2f}" for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), [f"{x:.2f}" for x in pivot.index])
    plt.xlabel("leader_multiplier")
    plt.ylabel("budget_multiplier")
    plt.title("Mahoraga14_3R Local Stability Heatmap")
    plt.colorbar(label="Mean RobustScore")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _plot_priority_windows(priority_df: pd.DataFrame, out_png: Path) -> None:
    names = list(priority_df["Window"])
    x = np.arange(len(names))
    width = 0.25
    plt.figure(figsize=(9, 5.5))
    plt.bar(x - width, priority_df["CandidateReturn"], width=width, label="Mahoraga14_3")
    plt.bar(x, priority_df["ControlReturn"], width=width, label="Control 14.1")
    plt.bar(x + width, priority_df["QQQReturn"], width=width, label="QQQ")
    plt.xticks(x, names)
    plt.ylabel("Total return")
    plt.title("Priority Bull Windows")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _plot_robustness_distribution(bootstrap_samples_df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    plt.hist(bootstrap_samples_df["DeltaSharpe"], bins=30, alpha=0.6, label="DeltaSharpe vs control")
    plt.hist(bootstrap_samples_df["DeltaCAGR"], bins=30, alpha=0.6, label="DeltaCAGR vs control")
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.legend()
    plt.title("Acceptance Robustness Distribution")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _plot_active_return_vs_qqq(candidate_obj: Dict[str, Any], qqq_obj: Dict[str, Any], out_png: Path) -> None:
    idx = candidate_obj["returns"].index
    active = pd.Series(candidate_obj["returns"], dtype=float).reindex(idx).fillna(0.0) - pd.Series(qqq_obj["returns"], dtype=float).reindex(idx).fillna(0.0)
    curve = (1.0 + active).cumprod() - 1.0
    cand_curve = (1.0 + pd.Series(candidate_obj["returns"], dtype=float).reindex(idx).fillna(0.0)).cumprod() - 1.0
    qqq_curve = (1.0 + pd.Series(qqq_obj["returns"], dtype=float).reindex(idx).fillna(0.0)).cumprod() - 1.0
    plt.figure(figsize=(10, 5.5))
    plt.plot(idx, curve, label="Active vs QQQ", linewidth=1.8)
    plt.plot(idx, cand_curve, label="Candidate", linewidth=1.1)
    plt.plot(idx, qqq_curve, label="QQQ", linewidth=1.1)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.legend()
    plt.title("Accepted Candidate Active Return vs QQQ")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _local_stability_summary_md(family_df: pd.DataFrame) -> str:
    base_id = _candidate_id(1.0, 1.0, 1.0, 1.0)
    base_row = family_df[family_df["CandidateId"] == base_id].iloc[0]
    top_row = family_df.sort_values(["RobustCandidate", "RobustScore"], ascending=[False, False]).iloc[0]
    robust_count = int(family_df["RobustCandidate"].sum())
    plateau_share = float(robust_count / max(1, len(family_df)))
    shape = "PLATEAU" if plateau_share >= 0.25 else ("NARROW_RIDGE" if plateau_share >= 0.10 else "FRAGILE_PEAK")
    lines = [
        "# Local Stability Summary",
        "",
        f"- base-current candidate: `{base_id}`",
        f"- base-current robust score: {float(base_row['RobustScore']):.4f}",
        f"- base-current robust flag: {int(base_row['RobustCandidate'])}",
        f"- top local candidate: `{top_row['CandidateId']}` with robust score {float(top_row['RobustScore']):.4f}",
        f"- robust-candidate share in local neighborhood: {plateau_share:.2%}",
        f"- plateau classification: {shape}",
        "- interpretation: acceptance prefers the base-current point only if it sits inside a stable plateau rather than at a fragile optimum.",
    ]
    return "\n".join(lines)


def _final_report_md(
    decision: str,
    comparison_df: pd.DataFrame,
    priority_df: pd.DataFrame,
    family_df: pd.DataFrame,
    loo_df: pd.DataFrame,
    continuation_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    bootstrap_summary_df: pd.DataFrame,
) -> str:
    base_id = _candidate_id(1.0, 1.0, 1.0, 1.0)
    base_row = family_df[family_df["CandidateId"] == base_id].iloc[0]
    lines = [
        "# Mahoraga14_3R Acceptance Report",
        "",
        f"## Final decision: {decision}",
        "",
        "## Stitched comparison",
        f"```\n{comparison_df.to_string(index=False)}\n```",
        "",
        "## Priority windows",
        f"```\n{priority_df.to_string(index=False)}\n```",
        "",
        "## Local stability",
        f"- base-current robust score: {float(base_row['RobustScore']):.4f}",
        f"- base-current robust flag: {int(base_row['RobustCandidate'])}",
        "",
        "## Leave-one-window-out",
        f"```\n{loo_df.to_string(index=False)}\n```",
        "",
        "## Continuation acceptance",
        f"```\n{continuation_df.to_string(index=False)}\n```" if len(continuation_df) else "No continuation diagnostic.",
        "",
        "## Acceptance robustness",
        f"```\n{stress_df.to_string(index=False)}\n```",
        "",
        "## Bootstrap summary",
        f"```\n{bootstrap_summary_df.to_string(index=False)}\n```",
    ]
    return "\n".join(lines)


def save_acceptance_outputs(
    wf: Dict[str, Any],
    cfg: Mahoraga14Config,
) -> Dict[str, Any]:
    ensure_dir(cfg.outputs_dir)
    out_dir = Path(cfg.outputs_dir)

    comparison_df = d14._build_comparison_df(wf, cfg)
    family_df, candidate_objects, candidate_windows, context = _candidate_family(wf, cfg)
    base_candidate_id = _candidate_id(1.0, 1.0, 1.0, 1.0)
    base_windows_df = candidate_windows[base_candidate_id]
    priority_df = _priority_acceptance_table(base_candidate_id, base_windows_df, cfg)
    loo_df = _leave_one_window_out(family_df, candidate_objects, candidate_windows, context, cfg)
    continuation_df = _continuation_acceptance_table(context["continuation_diag"])
    model_guard_df, model_guard_md = _model_selection_guard(family_df, candidate_objects, context["control"], cfg)
    stress_df, _ = d14._build_stress_and_robustness(wf, cfg, costs=m6.CostsConfig())
    bootstrap_summary_df, bootstrap_samples_df = _bootstrap_summary(candidate_objects[base_candidate_id], context["control"], cfg)
    parameter_inventory_df = _parameter_inventory(wf, cfg)

    decision, decision_md = _decision(
        comparison_df,
        priority_df,
        family_df,
        loo_df,
        model_guard_df,
        bootstrap_summary_df,
    )

    local_stability_summary_md = _local_stability_summary_md(family_df)
    priority_notes_md = _priority_acceptance_notes_md(priority_df)
    leave_one_md = _leave_one_window_out_md(loo_df)
    continuation_md = _continuation_acceptance_notes_md(continuation_df)
    parameter_freeze_md = _parameter_freeze_plan_md(parameter_inventory_df)
    robustness_notes_md = _robustness_notes_md(stress_df, bootstrap_summary_df)
    final_md = _final_report_md(decision, comparison_df, priority_df, family_df, loo_df, continuation_df, stress_df, bootstrap_summary_df)

    comparison_df.to_csv(out_dir / "stitched_comparison_acceptance_14_3R.csv", index=False)
    priority_df.to_csv(out_dir / "priority_window_acceptance_14_3R.csv", index=False)
    family_df.to_csv(out_dir / "local_stability_grid_14_3R.csv", index=False)
    loo_df.to_csv(out_dir / "leave_one_window_out_14_3R.csv", index=False)
    continuation_df.to_csv(out_dir / "continuation_acceptance_14_3R.csv", index=False)
    stress_df.to_csv(out_dir / "acceptance_robustness_suite_14_3R.csv", index=False)
    bootstrap_summary_df.to_csv(out_dir / "acceptance_bootstrap_summary_14_3R.csv", index=False)
    model_guard_df.to_csv(out_dir / "model_selection_guard_14_3R.csv", index=False)
    parameter_inventory_df.to_csv(out_dir / "parameter_inventory_14_3R.csv", index=False)

    (out_dir / "parameter_freeze_plan_14_3R.md").write_text(parameter_freeze_md, encoding="utf-8")
    (out_dir / "local_stability_summary_14_3R.md").write_text(local_stability_summary_md, encoding="utf-8")
    (out_dir / "priority_window_acceptance_notes_14_3R.md").write_text(priority_notes_md, encoding="utf-8")
    (out_dir / "leave_one_window_out_summary_14_3R.md").write_text(leave_one_md, encoding="utf-8")
    (out_dir / "continuation_acceptance_notes_14_3R.md").write_text(continuation_md, encoding="utf-8")
    (out_dir / "model_selection_guard_14_3R.md").write_text(model_guard_md, encoding="utf-8")
    (out_dir / "acceptance_robustness_notes_14_3R.md").write_text(robustness_notes_md, encoding="utf-8")
    (out_dir / "acceptance_decision_14_3R.md").write_text(decision_md, encoding="utf-8")
    (out_dir / "final_report_acceptance_14_3R.md").write_text(final_md, encoding="utf-8")

    _plot_active_return_vs_qqq(candidate_objects[base_candidate_id], context["qqq"], out_dir / "active_return_vs_qqq_curve_accepted_candidate.png")
    _plot_local_stability_heatmap(family_df, out_dir / "local_stability_heatmap_14_3R.png")
    _plot_priority_windows(priority_df, out_dir / "bull_window_scorecard_14_3R.png")
    bootstrap_samples_df.to_csv(out_dir / "_acceptance_bootstrap_samples_14_3R.csv", index=False)
    _plot_robustness_distribution(bootstrap_samples_df, out_dir / "robustness_distribution_14_3R.png")

    return {
        "decision": decision,
        "comparison_df": comparison_df,
        "priority_df": priority_df,
        "family_df": family_df,
        "loo_df": loo_df,
        "continuation_df": continuation_df,
        "stress_df": stress_df,
        "bootstrap_summary_df": bootstrap_summary_df,
        "model_guard_df": model_guard_df,
        "parameter_inventory_df": parameter_inventory_df,
    }
