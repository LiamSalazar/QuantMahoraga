from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mahoraga6_1 as m6
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
    _variant_label_map,
)
from mahoraga14_config import Mahoraga14Config
from mahoraga14_utils import ensure_dir


def _full_variant_order(cfg: Mahoraga14Config) -> List[str]:
    return [
        cfg.historical_benchmark_label,
        "QQQ",
        "SPY",
        cfg.official_baseline_label,
        cfg.continuation_variant_key,
        cfg.combo_variant_key,
        cfg.main_variant_key,
    ]


def _audited_candidate_keys(cfg: Mahoraga14Config) -> List[str]:
    return [
        cfg.official_baseline_label,
        cfg.continuation_variant_key,
        cfg.combo_variant_key,
        cfg.main_variant_key,
    ]


def _candidate_role(key: str, cfg: Mahoraga14Config) -> str:
    if key == cfg.official_baseline_label:
        return "BASELINE"
    if key == cfg.full_primary_variant_key:
        return "PRIMARY"
    if key == cfg.main_variant_key:
        return "AUXILIARY_REFERENCE"
    if key == cfg.combo_variant_key:
        return "SECONDARY_CANDIDATE"
    return "CANDIDATE"


def _stitched_map(wf: Dict[str, Any], cfg: Mahoraga14Config) -> Dict[str, Dict[str, Any]]:
    return {
        cfg.historical_benchmark_label: wf["stitched_legacy"],
        "QQQ": wf["stitched_benchmarks"]["QQQ"],
        "SPY": wf["stitched_benchmarks"]["SPY"],
        **wf["stitched_variants"],
    }


def _summary_from_object(obj: Dict[str, Any], cfg: Mahoraga14Config, label: str) -> Dict[str, float]:
    return m6.summarize(obj["returns"], obj["equity"], obj["exposure"], obj["turnover"], cfg, label)


def _alpha_nw(strategy_r: pd.Series, bench_r: pd.Series, cfg: Mahoraga14Config, label: str) -> Dict[str, float]:
    out = m6.alpha_test_nw(strategy_r, bench_r, cfg, label=label)
    if "error" in out:
        return {"alpha_ann": np.nan, "t_alpha": np.nan, "p_alpha": np.nan, "beta": np.nan, "R2": np.nan}
    return {
        "alpha_ann": float(out.get("alpha_ann", np.nan)),
        "t_alpha": float(out.get("t_alpha", np.nan)),
        "p_alpha": float(out.get("p_alpha", np.nan)),
        "beta": float(out.get("beta", np.nan)),
        "R2": float(out.get("R2", np.nan)),
    }


def _full_comparison_df(wf: Dict[str, Any], cfg: Mahoraga14Config) -> pd.DataFrame:
    labels = _variant_label_map(cfg)
    df = _build_stitched_comparison_fast(wf, cfg).copy()
    order = {labels[key]: i for i, key in enumerate(_full_variant_order(cfg))}
    df["__sort"] = df["Variant"].map(order)
    df = df.sort_values(["__sort", "Variant"]).drop(columns="__sort").reset_index(drop=True)
    return df


def _drawdown_df_from_object(label: str, obj: Dict[str, Any]) -> pd.DataFrame:
    eq = pd.Series(obj["equity"], dtype=float).dropna()
    if len(eq) == 0:
        return pd.DataFrame(columns=["Date", "Variant", "Equity", "CumMax", "Drawdown"])
    cummax = eq.cummax()
    dd = eq / cummax - 1.0
    return pd.DataFrame(
        {
            "Date": eq.index,
            "Variant": label,
            "Equity": eq.values,
            "CumMax": cummax.values,
            "Drawdown": dd.values,
        }
    )


def _equity_df_from_object(label: str, obj: Dict[str, Any]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(obj["returns"].index)
    return pd.DataFrame(
        {
            "Date": idx,
            "Variant": label,
            "ReturnNet": pd.Series(obj["returns"]).reindex(idx).fillna(0.0).values,
            "ReturnGross": pd.Series(obj.get("gross_returns", obj["returns"])).reindex(idx).fillna(0.0).values,
            "TransactionCost": pd.Series(obj.get("transaction_cost", pd.Series(0.0, index=idx))).reindex(idx).fillna(0.0).values,
            "Exposure": pd.Series(obj["exposure"]).reindex(idx).fillna(0.0).values,
            "Turnover": pd.Series(obj["turnover"]).reindex(idx).fillna(0.0).values,
            "Equity": pd.Series(obj["equity"]).reindex(idx).ffill().values,
        }
    )


def _expected_oos_index(wf: Dict[str, Any]) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(wf["calendar_index"])
    parts = []
    for row in wf["folds"].sort_values("fold").itertuples(index=False):
        parts.append(idx[(idx >= pd.Timestamp(row.test_start)) & (idx <= pd.Timestamp(row.test_end))])
    if not parts:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(np.concatenate([p.values for p in parts]))


def _build_stitched_full_trace(wf: Dict[str, Any], cfg: Mahoraga14Config) -> pd.DataFrame:
    labels = _variant_label_map(cfg)
    calendar_idx = pd.DatetimeIndex(wf["calendar_index"])
    calendar_pos = {dt: i for i, dt in enumerate(calendar_idx)}
    fold_df = wf["folds"].set_index("fold")
    expected_rows = fold_df["actual_test_days"].to_dict()
    effective_bounds: Dict[int, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for fold, row in fold_df.iterrows():
        window_idx = calendar_idx[(calendar_idx >= pd.Timestamp(row["test_start"])) & (calendar_idx <= pd.Timestamp(row["test_end"]))]
        if len(window_idx):
            effective_bounds[int(fold)] = (pd.Timestamp(window_idx[0]), pd.Timestamp(window_idx[-1]))
        else:
            effective_bounds[int(fold)] = (pd.NaT, pd.NaT)
    rows: List[Dict[str, Any]] = []
    for key in _full_variant_order(cfg):
        trace = wf["stitched_traces"][key].copy()
        trace = trace.sort_values("Fold").reset_index(drop=True)
        prev_end = None
        for rec in trace.to_dict("records"):
            fold = int(rec["Fold"])
            slice_start = pd.Timestamp(rec["SliceStart"])
            slice_end = pd.Timestamp(rec["SliceEnd"])
            effective_start, effective_end = effective_bounds.get(fold, (pd.NaT, pd.NaT))
            gap_days = 0
            overlap = False
            if prev_end is not None:
                gap_days = int(calendar_pos.get(slice_start, -1) - calendar_pos.get(prev_end, -1) - 1)
                overlap = gap_days < 0
            rows.append(
                {
                    **rec,
                    "Variant": labels[key],
                    "ExpectedRows": int(expected_rows.get(fold, 0)),
                    "RowsMatchExpected": bool(int(rec["Rows"]) == int(expected_rows.get(fold, 0))),
                    "EffectiveCalendarStart": effective_start,
                    "EffectiveCalendarEnd": effective_end,
                    "SliceMatchesEffectiveCalendar": bool(slice_start == effective_start and slice_end == effective_end),
                    "RequestedBoundsFullyAvailable": bool(
                        pd.Timestamp(rec["RequestedStart"]) == effective_start and pd.Timestamp(rec["RequestedEnd"]) == effective_end
                    ),
                    "GapTradingDaysFromPrevFold": max(0, gap_days),
                    "OverlapWithPrevFold": overlap,
                }
            )
            prev_end = slice_end
    return pd.DataFrame(rows)


def _integrity_check_rows(
    variant: str,
    obj: Dict[str, Any],
    trace_df: pd.DataFrame,
    expected_idx: pd.DatetimeIndex,
    cfg: Mahoraga14Config,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idx = pd.DatetimeIndex(obj["returns"].index)
    eq = pd.Series(obj["equity"], dtype=float).reindex(idx).ffill()
    rebuilt_eq = cfg.capital_initial * (1.0 + pd.Series(obj["returns"]).reindex(idx).fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1.0 if len(eq) else pd.Series(dtype=float)

    def add(check: str, passed: bool, detail: str) -> None:
        rows.append({"Variant": variant, "Check": check, "Passed": bool(passed), "Detail": detail})

    add("stitched_only_test_windows", bool((trace_df["WindowType"] == "TEST_ONLY").all()), f"window_types={sorted(trace_df['WindowType'].unique().tolist())}")
    add("trace_rows_match_expected", bool(trace_df["RowsMatchExpected"].all()), f"rows={trace_df['Rows'].tolist()} expected={trace_df['ExpectedRows'].tolist()}")
    add(
        "trace_matches_effective_oos_calendar",
        bool(trace_df["SliceMatchesEffectiveCalendar"].all()),
        f"effective_bounds={trace_df[['EffectiveCalendarStart', 'EffectiveCalendarEnd']].astype(str).agg('->'.join, axis=1).tolist()}",
    )
    add(
        "requested_bounds_fully_available",
        bool(trace_df["RequestedBoundsFullyAvailable"].all()),
        f"requested_vs_effective={trace_df[['RequestedStart', 'RequestedEnd', 'EffectiveCalendarStart', 'EffectiveCalendarEnd']].astype(str).agg(' | '.join, axis=1).tolist()}",
    )
    add("trace_no_overlap", bool((~trace_df["OverlapWithPrevFold"]).all()), f"overlap_flags={trace_df['OverlapWithPrevFold'].tolist()}")
    add("trace_no_gaps", bool((trace_df["GapTradingDaysFromPrevFold"] <= 0).all()), f"gap_days={trace_df['GapTradingDaysFromPrevFold'].tolist()}")
    add("index_monotonic_increasing", bool(idx.is_monotonic_increasing), f"n={len(idx)}")
    add("index_unique", bool(not idx.has_duplicates), f"duplicates={int(idx.duplicated().sum())}")
    add("index_matches_expected_oos_calendar", bool(idx.equals(expected_idx)), f"actual={len(idx)} expected={len(expected_idx)}")
    add("exposure_aligned", bool(pd.DatetimeIndex(obj["exposure"].index).equals(idx)), f"len={len(obj['exposure'])}")
    add("turnover_aligned", bool(pd.DatetimeIndex(obj["turnover"].index).equals(idx)), f"len={len(obj['turnover'])}")
    add(
        "costs_aligned",
        bool(pd.DatetimeIndex(pd.Series(obj.get("transaction_cost", pd.Series(dtype=float))).index).equals(idx)),
        f"len={len(pd.Series(obj.get('transaction_cost', pd.Series(dtype=float))))}",
    )
    add("equity_rebuild_matches", bool(float((eq - rebuilt_eq).abs().max()) <= 1e-8 if len(eq) else True), f"max_abs_diff={float((eq - rebuilt_eq).abs().max()) if len(eq) else 0.0:.3e}")
    add("drawdown_reconstructible", bool(np.isfinite(dd).all() if len(dd) else True), f"min_dd={float(dd.min()) if len(dd) else 0.0:.6f}")
    return rows


def _build_stitched_integrity_checks(wf: Dict[str, Any], cfg: Mahoraga14Config, trace_df: pd.DataFrame) -> pd.DataFrame:
    labels = _variant_label_map(cfg)
    objects = _stitched_map(wf, cfg)
    expected_idx = _expected_oos_index(wf)
    rows: List[Dict[str, Any]] = []
    for key in _full_variant_order(cfg):
        variant = labels[key]
        rows.extend(_integrity_check_rows(variant, objects[key], trace_df[trace_df["Variant"] == variant].copy(), expected_idx, cfg))
    return pd.DataFrame(rows)


def _maxdd_details(eq: pd.Series) -> Dict[str, Any]:
    s = pd.Series(eq, dtype=float).dropna()
    if len(s) == 0:
        return {"maxdd": np.nan, "peak_date": pd.NaT, "trough_date": pd.NaT, "recovery_date": pd.NaT}
    cummax = s.cummax()
    dd = s / cummax - 1.0
    trough = dd.idxmin()
    peak = s.loc[:trough].idxmax()
    recovery_mask = s.loc[trough:] >= s.loc[peak]
    recovery = recovery_mask[recovery_mask].index[0] if recovery_mask.any() else pd.NaT
    return {"maxdd": float(dd.min()), "peak_date": peak, "trough_date": trough, "recovery_date": recovery}


def _build_maxdd_audit(wf: Dict[str, Any], cfg: Mahoraga14Config) -> Tuple[pd.DataFrame, str]:
    labels = _variant_label_map(cfg)
    objects = _stitched_map(wf, cfg)
    rows = []
    notes = [
        "# MaxDD audit",
        "",
        "Method:",
        "1. Rebuild stitched equity from stitched net returns only.",
        "2. Compute drawdown as `equity / equity.cummax() - 1`.",
        "3. Confirm stored stitched equity matches rebuilt equity within tolerance.",
        "",
    ]
    for key in _full_variant_order(cfg):
        label = labels[key]
        obj = objects[key]
        idx = pd.DatetimeIndex(obj["returns"].index)
        rebuilt_eq = cfg.capital_initial * (1.0 + pd.Series(obj["returns"]).reindex(idx).fillna(0.0)).cumprod()
        stored_eq = pd.Series(obj["equity"]).reindex(idx).ffill()
        stored_details = _maxdd_details(stored_eq)
        rebuilt_details = _maxdd_details(rebuilt_eq)
        eq_diff = float((stored_eq - rebuilt_eq).abs().max()) if len(idx) else 0.0
        dd_diff = abs(float(stored_details["maxdd"]) - float(rebuilt_details["maxdd"])) if len(idx) else 0.0
        rows.append(
            {
                "Variant": label,
                "StoredMaxDD%": round(float(stored_details["maxdd"]) * 100.0, 4),
                "RebuiltMaxDD%": round(float(rebuilt_details["maxdd"]) * 100.0, 4),
                "MaxDDAuditDiffBps": round(dd_diff * 10000.0, 4),
                "EquityAuditMaxAbsDiff": eq_diff,
                "PeakDate": stored_details["peak_date"],
                "TroughDate": stored_details["trough_date"],
                "RecoveryDate": stored_details["recovery_date"],
            }
        )
        notes.append(
            f"- {label}: stored={stored_details['maxdd']*100:.2f}% rebuilt={rebuilt_details['maxdd']*100:.2f}% "
            f"peak={stored_details['peak_date']} trough={stored_details['trough_date']} diff_bps={dd_diff*10000:.4f}"
        )
    return pd.DataFrame(rows), "\n".join(notes)


def _neutralize_pre_test_override(override_daily: pd.DataFrame, test_start: pd.Timestamp) -> pd.DataFrame:
    out = override_daily.copy()
    mask = out.index < test_start
    if not mask.any():
        return out
    out.loc[mask, "override_type"] = "BASELINE"
    out.loc[mask, "override_detail"] = "BASELINE"
    out.loc[mask, "defense_blend"] = 0.0
    out.loc[mask, "gate_scale"] = 1.0
    out.loc[mask, "vol_mult"] = 1.0
    out.loc[mask, "exp_cap"] = 1.0
    out.loc[mask, "is_override"] = 0.0
    out.loc[mask, "is_structural_override"] = 0.0
    if "is_continuation_lift" in out.columns:
        out.loc[mask, "is_continuation_lift"] = 0.0
    if "is_continuation_activation" in out.columns:
        out.loc[mask, "is_continuation_activation"] = 0.0
    return out


def _blend_daily_weights(base_weights: pd.DataFrame, defense_weights: pd.DataFrame, defense_blend: pd.Series) -> pd.DataFrame:
    blend = pd.Series(defense_blend, index=base_weights.index, dtype=float).reindex(base_weights.index).fillna(0.0).clip(0.0, 1.0)
    return base_weights.mul(1.0 - blend, axis=0) + defense_weights.mul(blend, axis=0)


def _delay_one_rebalance(weights_exec_1x: pd.DataFrame) -> pd.DataFrame:
    if len(weights_exec_1x) == 0:
        return weights_exec_1x.copy()
    change_mask = weights_exec_1x.ne(weights_exec_1x.shift()).any(axis=1).to_numpy()
    starts = np.flatnonzero(change_mask)
    if len(starts) == 0 or starts[0] != 0:
        starts = np.r_[0, starts]
    ends = np.r_[starts[1:], len(weights_exec_1x)]
    out = weights_exec_1x.copy()
    for g in range(1, len(starts)):
        out.iloc[starts[g] : ends[g]] = weights_exec_1x.iloc[starts[g - 1]].values
    return out


def _default_override_daily(idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "defense_blend": 0.0,
            "gate_scale": 1.0,
            "vol_mult": 1.0,
            "exp_cap": 1.0,
        },
        index=idx,
    )


def _prepare_variant_fold_inputs(
    result: Dict[str, Any],
    variant_key: str,
    policy_override: Optional[Dict[str, float]] = None,
    continuation_override: Optional[Dict[str, float]] = None,
    cfg_override: Optional[Dict[str, float]] = None,
) -> Tuple[Mahoraga14Config, pd.DataFrame, pd.DataFrame]:
    cfg_fold = deepcopy(result["cfg_fold"])
    if cfg_override:
        for key, value in cfg_override.items():
            setattr(cfg_fold, key, float(value))
    idx = pd.DatetimeIndex(result["stress_pre"]["idx"])

    if variant_key == cfg_fold.official_baseline_label:
        return cfg_fold, result["base_weights_exec_1x"].copy(), _default_override_daily(idx)

    from override_policy import build_override_weekly, weekly_to_daily_override

    policy_params = {k: float(v) for k, v in result["policy_params"].items()}
    if policy_override:
        policy_params.update({k: float(v) for k, v in policy_override.items()})
    continuation_info = deepcopy(result["continuation_fit"]) if variant_key in {cfg_fold.continuation_variant_key, cfg_fold.combo_variant_key} else None
    if continuation_info and continuation_override:
        continuation_info.update({k: float(v) for k, v in continuation_override.items()})

    weekly_full = result["weekly_full_all"].loc[: result["test_end"]].copy()
    override_weekly_full = build_override_weekly(weekly_full, policy_params, cfg_fold, variant=variant_key, continuation_info=continuation_info)
    override_daily = weekly_to_daily_override(override_weekly_full, idx, cfg_fold)
    override_daily = _neutralize_pre_test_override(override_daily, pd.Timestamp(result["test_start"]))
    weights_exec_1x = _blend_daily_weights(result["base_weights_exec_1x"], result["defense_weights_exec_1x"], override_daily["defense_blend"])
    return cfg_fold, weights_exec_1x, override_daily


def _stitch_fold_backtests(
    fold_backtests: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    cfg: Mahoraga14Config,
    label: str,
) -> Dict[str, Any]:
    frames = []
    for result, bt in zip(results, fold_backtests):
        start = pd.Timestamp(result["test_start"])
        end = pd.Timestamp(result["test_end"])
        r = bt["returns_net"].loc[start:end]
        frames.append(
            pd.DataFrame(
                {
                    "ReturnNet": r,
                    "ReturnGross": bt.get("returns_gross", r).loc[start:end],
                    "TransactionCost": bt.get("transaction_cost", pd.Series(0.0, index=r.index)).reindex(r.index).fillna(0.0),
                    "Exposure": bt["exposure"].loc[start:end],
                    "Turnover": bt["turnover"].loc[start:end],
                }
            )
        )
    stitched = pd.concat(frames).sort_index() if frames else pd.DataFrame()
    returns = stitched.get("ReturnNet", pd.Series(dtype=float))
    equity = cfg.capital_initial * (1.0 + returns).cumprod() if len(returns) else pd.Series(dtype=float)
    return {
        "label": label,
        "returns": returns,
        "gross_returns": stitched.get("ReturnGross", pd.Series(dtype=float)),
        "transaction_cost": stitched.get("TransactionCost", pd.Series(dtype=float)),
        "exposure": stitched.get("Exposure", pd.Series(dtype=float)),
        "turnover": stitched.get("Turnover", pd.Series(dtype=float)),
        "equity": equity,
    }


def _scenario_row(
    variant_label: str,
    scenario: str,
    note: str,
    base_obj: Dict[str, Any],
    stressed_obj: Dict[str, Any],
    qqq_obj: Dict[str, Any],
    spy_obj: Dict[str, Any],
    cfg: Mahoraga14Config,
) -> Dict[str, Any]:
    base_summary = _summary_from_object(base_obj, cfg, variant_label)
    stress_summary = _summary_from_object(stressed_obj, cfg, f"{variant_label}_{scenario}")
    base_alpha_qqq = _alpha_nw(base_obj["returns"], qqq_obj["returns"], cfg, f"{variant_label}_BASE_QQQ")
    base_alpha_spy = _alpha_nw(base_obj["returns"], spy_obj["returns"], cfg, f"{variant_label}_BASE_SPY")
    stress_alpha_qqq = _alpha_nw(stressed_obj["returns"], qqq_obj["returns"], cfg, f"{variant_label}_{scenario}_QQQ")
    stress_alpha_spy = _alpha_nw(stressed_obj["returns"], spy_obj["returns"], cfg, f"{variant_label}_{scenario}_SPY")
    return {
        "Variant": variant_label,
        "Scenario": scenario,
        "ScenarioNote": note,
        "BaseCAGR%": round(base_summary["CAGR"] * 100.0, 2),
        "StressCAGR%": round(stress_summary["CAGR"] * 100.0, 2),
        "DeltaCAGR%": round((stress_summary["CAGR"] - base_summary["CAGR"]) * 100.0, 2),
        "BaseSharpe": round(base_summary["Sharpe"], 4),
        "StressSharpe": round(stress_summary["Sharpe"], 4),
        "DeltaSharpe": round(stress_summary["Sharpe"] - base_summary["Sharpe"], 4),
        "BaseMaxDD%": round(base_summary["MaxDD"] * 100.0, 2),
        "StressMaxDD%": round(stress_summary["MaxDD"] * 100.0, 2),
        "DeltaMaxDD%": round((stress_summary["MaxDD"] - base_summary["MaxDD"]) * 100.0, 2),
        "BaseAlphaNW_QQQ": round(base_alpha_qqq["alpha_ann"], 6) if np.isfinite(base_alpha_qqq["alpha_ann"]) else np.nan,
        "StressAlphaNW_QQQ": round(stress_alpha_qqq["alpha_ann"], 6) if np.isfinite(stress_alpha_qqq["alpha_ann"]) else np.nan,
        "DeltaAlphaNW_QQQ": round(stress_alpha_qqq["alpha_ann"] - base_alpha_qqq["alpha_ann"], 6)
        if np.isfinite(base_alpha_qqq["alpha_ann"]) and np.isfinite(stress_alpha_qqq["alpha_ann"])
        else np.nan,
        "BaseAlphaNW_SPY": round(base_alpha_spy["alpha_ann"], 6) if np.isfinite(base_alpha_spy["alpha_ann"]) else np.nan,
        "StressAlphaNW_SPY": round(stress_alpha_spy["alpha_ann"], 6) if np.isfinite(stress_alpha_spy["alpha_ann"]) else np.nan,
        "DeltaAlphaNW_SPY": round(stress_alpha_spy["alpha_ann"] - base_alpha_spy["alpha_ann"], 6)
        if np.isfinite(base_alpha_spy["alpha_ann"]) and np.isfinite(stress_alpha_spy["alpha_ann"])
        else np.nan,
    }


def _run_stress_scenario(
    wf: Dict[str, Any],
    cfg: Mahoraga14Config,
    costs: m6.CostsConfig,
    variant_key: str,
    scenario: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    from base_alpha_engine import backtest_from_1x_weights

    fold_bts = []
    note = ""
    for result in wf["results"]:
        cfg_override: Dict[str, float] = {}
        policy_override: Dict[str, float] = {}
        continuation_override: Dict[str, float] = {}
        scenario_costs = deepcopy(costs)

        if scenario == "COST_PLUS_25":
            scenario_costs.commission *= 1.25
            scenario_costs.slippage *= 1.25
            note = "commission/slippage x1.25"
        elif scenario == "COST_PLUS_50":
            scenario_costs.commission *= 1.50
            scenario_costs.slippage *= 1.50
            note = "commission/slippage x1.50"
        elif scenario == "COST_PLUS_100":
            scenario_costs.commission *= 2.00
            scenario_costs.slippage *= 2.00
            note = "commission/slippage x2.00"
        elif scenario == "SLIPPAGE_PLUS_5BPS":
            scenario_costs.slippage += 0.0005
            note = "extra slippage +5bps"
        elif scenario == "CONTINUATION_FALSE_POSITIVES_EXTRA":
            if variant_key not in {cfg.continuation_variant_key, cfg.combo_variant_key}:
                return None, "not_applicable"
            continuation_override = {
                "trigger_enter": max(0.05, float(result["continuation_fit"].get("trigger_enter", 0.25)) - 0.05),
                "pressure_enter": max(0.05, float(result["continuation_fit"].get("pressure_enter", 0.25)) - 0.05),
                "break_risk_cap": min(0.99, float(result["continuation_fit"].get("break_risk_cap", 0.60)) + 0.04),
            }
            note = "lower trigger/pressure gates, higher break-risk cap"
        elif scenario == "STRUCTURAL_FALSE_POSITIVES_EXTRA":
            if variant_key not in {cfg.main_variant_key, cfg.combo_variant_key}:
                return None, "not_applicable"
            policy_override = {
                "structural_enter_thr": max(0.55, float(result["policy_params"]["structural_enter_thr"]) - 0.05),
            }
            note = "lower structural enter threshold by 5pts"
        elif scenario == "CONTINUATION_LIFT_EFFICACY_MINUS_50":
            if variant_key not in {cfg.continuation_variant_key, cfg.combo_variant_key}:
                return None, "not_applicable"
            cfg_override = {
                "continuation_gate": 1.0 + 0.5 * max(0.0, float(result["cfg_fold"].continuation_gate) - 1.0),
                "continuation_vol_mult": 1.0 + 0.5 * max(0.0, float(result["cfg_fold"].continuation_vol_mult) - 1.0),
                "continuation_exp_cap": 1.0 + 0.5 * max(0.0, float(result["cfg_fold"].continuation_exp_cap) - 1.0),
            }
            note = "halve continuation lift amplitude"
        elif scenario == "GATE_VOL_CAP_STRESS":
            note = "gate*0.95 vol_mult*0.95 exp_cap*0.90"
        elif scenario == "EXECUTION_DELAY_1_REBALANCE":
            note = "shift one full rebalance block"

        cfg_fold, weights_exec_1x, override_daily = _prepare_variant_fold_inputs(
            result,
            variant_key,
            policy_override=policy_override,
            continuation_override=continuation_override,
            cfg_override=cfg_override,
        )
        if scenario == "GATE_VOL_CAP_STRESS":
            override_daily["gate_scale"] = override_daily["gate_scale"].mul(0.95).clip(0.0, cfg_fold.max_gate_scale)
            override_daily["vol_mult"] = override_daily["vol_mult"].mul(0.95).clip(lower=0.0)
            override_daily["exp_cap"] = override_daily["exp_cap"].mul(0.90).clip(0.0, cfg_fold.max_exposure)
        if scenario == "EXECUTION_DELAY_1_REBALANCE":
            weights_exec_1x = _delay_one_rebalance(weights_exec_1x)

        bt = backtest_from_1x_weights(
            result["stress_pre"],
            weights_exec_1x,
            override_daily["gate_scale"],
            override_daily["vol_mult"],
            override_daily["exp_cap"],
            cfg_fold,
            scenario_costs,
            label=f"{variant_key}_{scenario}_{int(result['fold'])}",
        )
        fold_bts.append(bt)

    return _stitch_fold_backtests(fold_bts, wf["results"], cfg, f"{variant_key}_{scenario}"), note


def _apply_empirical_path_stress(obj: Dict[str, Any], label: str, block: int = 10, injections: int = 2) -> Tuple[Dict[str, Any], pd.DataFrame]:
    returns = pd.Series(obj["returns"], dtype=float).copy()
    if len(returns) < block * 4:
        return obj, pd.DataFrame(columns=["Variant", "SourceStart", "SourceEnd", "TargetStart", "TargetEnd", "BlockDays"])

    roll = (1.0 + returns).rolling(block).apply(np.prod, raw=True) - 1.0
    roll = roll.dropna().sort_values()
    source_ends: List[int] = []
    for dt in roll.index:
        pos = returns.index.get_loc(dt)
        if all(abs(pos - prev) >= block for prev in source_ends):
            source_ends.append(int(pos))
        if len(source_ends) >= injections:
            break
    if not source_ends:
        return obj, pd.DataFrame(columns=["Variant", "SourceStart", "SourceEnd", "TargetStart", "TargetEnd", "BlockDays"])

    target_starts = np.linspace(block, max(block, len(returns) - block - 1), len(source_ends) + 2, dtype=int)[1:-1]
    stressed = returns.copy()
    trace_rows = []
    for src_end, tgt_start in zip(source_ends, target_starts):
        src_start = src_end - block + 1
        tgt_end = min(len(stressed), tgt_start + block)
        src_block = returns.iloc[src_start : src_start + (tgt_end - tgt_start)].values
        tgt_block = stressed.iloc[tgt_start:tgt_end].values
        compounded = np.clip((1.0 + tgt_block) * (1.0 + src_block) - 1.0, -0.95, None)
        stressed.iloc[tgt_start:tgt_end] = compounded
        trace_rows.append(
            {
                "Variant": label,
                "SourceStart": returns.index[src_start],
                "SourceEnd": returns.index[src_start + len(src_block) - 1],
                "TargetStart": returns.index[tgt_start],
                "TargetEnd": returns.index[tgt_start + len(src_block) - 1],
                "BlockDays": int(len(src_block)),
            }
        )
    eq = 100_000.0 * (1.0 + stressed).cumprod()
    stressed_obj = {
        "returns": stressed,
        "gross_returns": stressed + pd.Series(obj.get("transaction_cost", pd.Series(0.0, index=returns.index)), index=returns.index, dtype=float),
        "transaction_cost": pd.Series(obj.get("transaction_cost", pd.Series(0.0, index=returns.index)), index=returns.index, dtype=float),
        "exposure": pd.Series(obj["exposure"]).reindex(stressed.index).fillna(0.0),
        "turnover": pd.Series(obj["turnover"]).reindex(stressed.index).fillna(0.0),
        "equity": eq,
    }
    return stressed_obj, pd.DataFrame(trace_rows)


def _build_stress_suite_full(
    wf: Dict[str, Any],
    cfg: Mahoraga14Config,
    costs: m6.CostsConfig,
) -> Tuple[pd.DataFrame, str, pd.DataFrame]:
    labels = _variant_label_map(cfg)
    stitched = _stitched_map(wf, cfg)
    qqq_obj = stitched["QQQ"]
    spy_obj = stitched["SPY"]
    scenarios = [
        "COST_PLUS_25",
        "COST_PLUS_50",
        "COST_PLUS_100",
        "SLIPPAGE_PLUS_5BPS",
        "EXECUTION_DELAY_1_REBALANCE",
        "CONTINUATION_FALSE_POSITIVES_EXTRA",
        "STRUCTURAL_FALSE_POSITIVES_EXTRA",
        "CONTINUATION_LIFT_EFFICACY_MINUS_50",
        "GATE_VOL_CAP_STRESS",
    ]

    rows: List[Dict[str, Any]] = []
    path_traces = []
    notes = [
        "# Stress suite audit",
        "",
        "All non-path stresses recompute fold-level returns/equity from the same TEST_ONLY windows.",
        "Benchmark comparisons keep the original stitched QQQ/SPY paths unchanged.",
        "",
    ]

    for key in _audited_candidate_keys(cfg):
        label = labels[key]
        base_obj = stitched[key]
        rows.append(_scenario_row(label, "BASELINE", "unstressed stitched OOS", base_obj, base_obj, qqq_obj, spy_obj, cfg))
        for scenario in scenarios:
            stressed_obj, note = _run_stress_scenario(wf, cfg, costs, key, scenario)
            if stressed_obj is None:
                continue
            rows.append(_scenario_row(label, scenario, note, base_obj, stressed_obj, qqq_obj, spy_obj, cfg))
            notes.append(f"- {label} | {scenario}: {note}")

        path_obj, path_trace = _apply_empirical_path_stress(base_obj, label)
        rows.append(_scenario_row(label, "EMPIRICAL_TAIL_BLOCK_PATH_STRESS", "replay worst empirical 10d blocks on later windows", base_obj, path_obj, qqq_obj, spy_obj, cfg))
        if len(path_trace):
            path_traces.append(path_trace)
    return pd.DataFrame(rows), "\n".join(notes), (pd.concat(path_traces, ignore_index=True) if path_traces else pd.DataFrame())


def _series_metrics_from_returns(r: pd.Series, cfg: Mahoraga14Config) -> Dict[str, float]:
    s = pd.Series(r, dtype=float).fillna(0.0)
    eq = cfg.capital_initial * (1.0 + s).cumprod()
    summary = m6.summarize(s, eq, pd.Series(1.0, index=s.index), pd.Series(0.0, index=s.index), cfg, "mc")
    return {"CAGR": float(summary["CAGR"]), "Sharpe": float(summary["Sharpe"]), "MaxDD": float(summary["MaxDD"])}


def _stationary_bootstrap(values: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    if n == 0:
        return np.array([], dtype=float)
    p = 1.0 / max(1, int(block))
    out = np.empty(n, dtype=float)
    idx = int(rng.integers(0, n))
    for i in range(n):
        if i == 0 or rng.random() < p:
            idx = int(rng.integers(0, n))
        else:
            idx = (idx + 1) % n
        out[i] = values[idx]
    return out


def _run_stationary_bootstrap_mc(
    obj: Dict[str, Any],
    cfg: Mahoraga14Config,
    variant_label: str,
    method: str,
    n_samples: int,
    block: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = pd.Series(obj["returns"], dtype=float).fillna(0.0).values
    rows = []
    for sample_id in range(n_samples):
        sample = _stationary_bootstrap(values, block=block, rng=rng)
        metrics = _series_metrics_from_returns(pd.Series(sample), cfg)
        rows.append({"Variant": variant_label, "Method": method, "SampleId": sample_id, **metrics})
    return pd.DataFrame(rows)


def _run_friction_mc(
    obj: Dict[str, Any],
    cfg: Mahoraga14Config,
    variant_label: str,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gross = pd.Series(obj.get("gross_returns", obj["returns"]), dtype=float).fillna(0.0).values
    tc = pd.Series(obj.get("transaction_cost", pd.Series(0.0, index=obj["returns"].index)), dtype=float).fillna(0.0).abs().values
    rows = []
    for sample_id in range(n_samples):
        mult = float(np.clip(rng.normal(loc=1.0, scale=0.30), 0.50, 2.25))
        stressed = gross - tc * mult
        metrics = _series_metrics_from_returns(pd.Series(stressed), cfg)
        rows.append({"Variant": variant_label, "Method": "friction_multiplier_mc", "SampleId": sample_id, "FrictionMultiplier": mult, **metrics})
    return pd.DataFrame(rows)


def _run_local_parameter_neighborhood(
    wf: Dict[str, Any],
    cfg: Mahoraga14Config,
    costs: m6.CostsConfig,
) -> pd.DataFrame:
    from base_alpha_engine import backtest_from_1x_weights

    primary = cfg.full_primary_variant_key
    labels = _variant_label_map(cfg)
    rows = []
    sample_id = 0
    for structural_shift in (-0.03, 0.0, 0.03):
        for blend_shift in (-0.05, 0.0, 0.05):
            for lift_mult in (0.75, 1.0):
                fold_bts = []
                for result in wf["results"]:
                    policy_override = {
                        "structural_enter_thr": np.clip(float(result["policy_params"]["structural_enter_thr"]) + structural_shift, 0.55, 0.95),
                        "structural_blend": np.clip(float(result["policy_params"]["structural_blend"]) + blend_shift, 0.0, 1.0),
                    }
                    cfg_override = {
                        "continuation_gate": 1.0 + lift_mult * max(0.0, float(result["cfg_fold"].continuation_gate) - 1.0),
                        "continuation_vol_mult": 1.0 + lift_mult * max(0.0, float(result["cfg_fold"].continuation_vol_mult) - 1.0),
                        "continuation_exp_cap": 1.0 + lift_mult * max(0.0, float(result["cfg_fold"].continuation_exp_cap) - 1.0),
                    }
                    cfg_fold, weights_exec_1x, override_daily = _prepare_variant_fold_inputs(
                        result,
                        primary,
                        policy_override=policy_override,
                        cfg_override=cfg_override,
                    )
                    bt = backtest_from_1x_weights(
                        result["stress_pre"],
                        weights_exec_1x,
                        override_daily["gate_scale"],
                        override_daily["vol_mult"],
                        override_daily["exp_cap"],
                        cfg_fold,
                        costs,
                        label=f"{primary}_LOCAL_{sample_id}_{int(result['fold'])}",
                    )
                    fold_bts.append(bt)
                stitched_obj = _stitch_fold_backtests(fold_bts, wf["results"], cfg, f"{primary}_LOCAL_{sample_id}")
                metrics = _series_metrics_from_returns(stitched_obj["returns"], cfg)
                rows.append(
                    {
                        "Variant": labels[primary],
                        "Method": "local_param_neighborhood",
                        "SampleId": sample_id,
                        "StructuralEnterShift": structural_shift,
                        "StructuralBlendShift": blend_shift,
                        "ContinuationLiftMultiplier": lift_mult,
                        **metrics,
                    }
                )
                sample_id += 1
    return pd.DataFrame(rows)


def _summarize_mc_results(samples: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    if len(samples) == 0:
        empty_summary = pd.DataFrame(columns=["Variant", "Method", "Samples", "MeanCAGR%", "MeanSharpe", "MeanMaxDD%", "Prob_CAGR_lt_0", "Prob_Sharpe_lt_1", "Prob_MaxDD_lt_25"])
        empty_pct = pd.DataFrame(columns=["Variant", "Method", "Metric", "P5", "P25", "P50", "P75", "P95"])
        return empty_summary, empty_pct, "# Monte Carlo audit\n\nNo Monte Carlo samples were produced."

    summary_rows = []
    pct_rows = []
    for (variant, method), sub in samples.groupby(["Variant", "Method"], sort=False):
        summary_rows.append(
            {
                "Variant": variant,
                "Method": method,
                "Samples": int(len(sub)),
                "MeanCAGR%": round(float(sub["CAGR"].mean()) * 100.0, 2),
                "MeanSharpe": round(float(sub["Sharpe"].mean()), 4),
                "MeanMaxDD%": round(float(sub["MaxDD"].mean()) * 100.0, 2),
                "Prob_CAGR_lt_0": round(float((sub["CAGR"] < 0.0).mean()), 4),
                "Prob_Sharpe_lt_1": round(float((sub["Sharpe"] < 1.0).mean()), 4),
                "Prob_MaxDD_lt_25": round(float((sub["MaxDD"] < -0.25).mean()), 4),
            }
        )
        for metric in ["CAGR", "Sharpe", "MaxDD"]:
            pct_rows.append(
                {
                    "Variant": variant,
                    "Method": method,
                    "Metric": metric,
                    "P5": round(float(sub[metric].quantile(0.05)), 6),
                    "P25": round(float(sub[metric].quantile(0.25)), 6),
                    "P50": round(float(sub[metric].quantile(0.50)), 6),
                    "P75": round(float(sub[metric].quantile(0.75)), 6),
                    "P95": round(float(sub[metric].quantile(0.95)), 6),
                }
            )

    notes = [
        "# Monte Carlo audit",
        "",
        "Methods:",
        "- stationary_block_bootstrap: stationary bootstrap on stitched TEST_ONLY daily returns.",
        "- friction_multiplier_mc: multiplies realized stitched transaction-cost series.",
        "- local_param_neighborhood: local perturbation around the final continuation candidate.",
        "",
        "Probability thresholds:",
        "- CAGR < 0",
        "- Sharpe < 1",
        "- MaxDD < -25%",
    ]
    return pd.DataFrame(summary_rows), pd.DataFrame(pct_rows), "\n".join(notes)


def _build_selected_candidate_audit(
    cfg: Mahoraga14Config,
    comparison_df: pd.DataFrame,
    pvalue_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    mc_pct_df: pd.DataFrame,
) -> pd.DataFrame:
    labels = _variant_label_map(cfg)
    stitched_tests = pvalue_df[pvalue_df["Segment"] == "STITCHED"].copy()
    comparison = comparison_df.set_index("Variant")
    rows = []
    for key in _audited_candidate_keys(cfg):
        label = labels[key]
        ref = labels[cfg.official_baseline_label]
        pq = stitched_tests[(stitched_tests["Target"] == label) & (stitched_tests["Reference"] == ref)].copy()
        worst_stress = stress_df[stress_df["Variant"] == label].sort_values("StressSharpe").head(1)
        mc_p50_cagr = mc_pct_df[(mc_pct_df["Variant"] == label) & (mc_pct_df["Metric"] == "CAGR")]["P50"]
        mc_p5_dd = mc_pct_df[(mc_pct_df["Variant"] == label) & (mc_pct_df["Metric"] == "MaxDD")]["P5"]
        rows.append(
            {
                "Variant": label,
                "Role": _candidate_role(key, cfg),
                "StitchedCAGR%": comparison.at[label, "CAGR%"],
                "StitchedSharpe": comparison.at[label, "Sharpe"],
                "StitchedMaxDD%": comparison.at[label, "MaxDD%"],
                "p_value_vs_BASE": float(pq["p_value"].iloc[0]) if len(pq) else np.nan,
                "q_value_vs_BASE": float(pq["q_value"].iloc[0]) if len(pq) else np.nan,
                "WorstStressScenario": worst_stress["Scenario"].iloc[0] if len(worst_stress) else "n/a",
                "WorstStressSharpe": float(worst_stress["StressSharpe"].iloc[0]) if len(worst_stress) else np.nan,
                "MC_P50_CAGR": float(mc_p50_cagr.iloc[0]) if len(mc_p50_cagr) else np.nan,
                "MC_P5_MaxDD": float(mc_p5_dd.iloc[0]) if len(mc_p5_dd) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _figure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 320,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#4c566a",
            "axes.labelcolor": "#2e3440",
            "xtick.color": "#2e3440",
            "ytick.color": "#2e3440",
            "grid.color": "#d8dee9",
            "font.size": 10,
            "axes.titleweight": "bold",
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _generate_figures(
    wf: Dict[str, Any],
    cfg: Mahoraga14Config,
    mc_samples: pd.DataFrame,
) -> Dict[str, str]:
    labels = _variant_label_map(cfg)
    stitched = _stitched_map(wf, cfg)
    figures_dir = Path(cfg.outputs_dir) / "figures"
    ensure_dir(str(figures_dir))
    _figure_style()

    primary_key = cfg.full_primary_variant_key
    primary_label = labels[primary_key]
    paths: Dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for key, color in [
        (cfg.official_baseline_label, "#355070"),
        (primary_key, "#bc4749"),
        (cfg.historical_benchmark_label, "#6c757d"),
        ("QQQ", "#1b9aaa"),
        ("SPY", "#718355"),
    ]:
        obj = stitched[key]
        ax.plot(obj["equity"].index, obj["equity"].values, label=labels[key], lw=2.0 if key in {cfg.official_baseline_label, primary_key} else 1.5, color=color)
    ax.set_title("Stitched OOS equity")
    ax.set_ylabel("Equity")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, alpha=0.35)
    path = figures_dir / "equity_curve_stitched_full.png"
    _save_figure(fig, path)
    paths["equity_curve_stitched_full"] = str(path)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for key, color in [
        (primary_key, "#bc4749"),
        (cfg.official_baseline_label, "#355070"),
        (cfg.historical_benchmark_label, "#6c757d"),
        ("QQQ", "#1b9aaa"),
    ]:
        eq = pd.Series(stitched[key]["equity"], dtype=float)
        dd = eq / eq.cummax() - 1.0
        ax.plot(dd.index, dd.values * 100.0, label=labels[key], lw=2.0 if key in {cfg.official_baseline_label, primary_key} else 1.5, color=color)
    ax.set_title("Stitched OOS drawdown")
    ax.set_ylabel("Drawdown %")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, alpha=0.35)
    path = figures_dir / "drawdown_curve_stitched_full.png"
    _save_figure(fig, path)
    paths["drawdown_curve_stitched_full"] = str(path)

    fold_df = wf["fold_df"].copy().sort_values("fold")
    alpha_df = _build_alpha_nw_fast(wf, cfg)
    alpha_df = alpha_df[(alpha_df["Segment"].str.startswith("FOLD_")) & (alpha_df["Benchmark"] == "QQQ")]
    fold_map = {
        labels[cfg.official_baseline_label]: ("BASE_CAGR%", "BASE_Sharpe", "BASE_MaxDD%"),
        labels[cfg.continuation_variant_key]: ("CONT_CAGR%", "CONT_Sharpe", "CONT_MaxDD%"),
        labels[cfg.combo_variant_key]: ("COMBO_CAGR%", "COMBO_Sharpe", "COMBO_MaxDD%"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("CAGR", lambda variant: np.asarray(fold_df[list(fold_map[variant])[0]], dtype=float)),
        ("Sharpe", lambda variant: np.asarray(fold_df[list(fold_map[variant])[1]], dtype=float)),
        ("MaxDD", lambda variant: np.asarray(fold_df[list(fold_map[variant])[2]], dtype=float)),
        ("AlphaNW_QQQ", lambda variant: alpha_df[alpha_df["Variant"] == variant].sort_values("Fold")["alpha_ann"].to_numpy(dtype=float)),
    ]
    variants = list(fold_map.keys())
    for ax, (title, getter) in zip(axes.flatten(), metrics):
        mat = np.vstack([getter(variant) for variant in variants])
        im = ax.imshow(mat, aspect="auto", cmap="Blues")
        ax.set_title(title)
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(variants)
        ax.set_xticks(range(len(fold_df)))
        ax.set_xticklabels(fold_df["fold"].tolist())
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = figures_dir / "fold_heatmap_full.png"
    _save_figure(fig, path)
    paths["fold_heatmap_full"] = str(path)

    event_df = _build_continuation_event_study_fast(wf, cfg)
    event_df = event_df[(event_df["Segment"] == "STITCHED") & (event_df["Variant"] == primary_label)]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if len(event_df):
        x = np.arange(3)
        act = [float(event_df["MeanRet1W"].iloc[0]), float(event_df["MeanRet2W"].iloc[0]), float(event_df["MeanRet4W"].iloc[0])]
        noact = [float(event_df["NoActMeanRet1W"].iloc[0]), float(event_df["NoActMeanRet2W"].iloc[0]), float(event_df["NoActMeanRet4W"].iloc[0])]
        w = 0.36
        ax.bar(x - w / 2.0, np.array(act) * 100.0, width=w, label="Activation", color="#bc4749")
        ax.bar(x + w / 2.0, np.array(noact) * 100.0, width=w, label="No activation", color="#355070")
        ax.set_xticks(x)
        ax.set_xticklabels(["1W", "2W", "4W"])
    ax.set_title("Continuation event study")
    ax.set_ylabel("Post-activation return %")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.35)
    path = figures_dir / "continuation_event_study_full.png"
    _save_figure(fig, path)
    paths["continuation_event_study_full"] = str(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    sub = mc_samples[(mc_samples["Variant"] == primary_label) & (mc_samples["Method"] == "stationary_block_bootstrap")]
    if len(sub):
        axes[0].hist(sub["CAGR"] * 100.0, bins=30, color="#355070", alpha=0.85)
        axes[0].set_title("Bootstrap CAGR %")
        axes[1].hist(sub["MaxDD"] * 100.0, bins=30, color="#bc4749", alpha=0.85)
        axes[1].set_title("Bootstrap MaxDD %")
    for ax in axes:
        ax.grid(True, alpha=0.35)
    path = figures_dir / "montecarlo_distribution_full.png"
    _save_figure(fig, path)
    paths["montecarlo_distribution_full"] = str(path)

    return paths


def _build_readiness_checklist(
    cfg: Mahoraga14Config,
    comparison_df: pd.DataFrame,
    alpha_nw_df: pd.DataFrame,
    integrity_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    mc_summary_df: pd.DataFrame,
    figure_paths: Dict[str, str],
    output_paths: List[Path],
) -> Tuple[pd.DataFrame, str]:
    required_outputs = [
        "stitched_comparison_full.csv",
        "fold_summary_full.csv",
        "floor_ceiling_summary_full.csv",
        "pvalue_qvalue_full.csv",
        "alpha_nw_full.csv",
        "stress_suite_full.csv",
        "montecarlo_summary_full.csv",
        "continuation_event_study_full.csv",
        "selected_candidate_audit_full.csv",
        "stitched_integrity_checks.csv",
        "maxdd_audit_full.csv",
        "stitched_full_trace.csv",
        "equity_stitched_full.csv",
        "drawdown_stitched_full.csv",
    ]
    existing = {path.name for path in output_paths if path.exists()}
    labels = set(comparison_df["Variant"].tolist())
    rows = [
        {"Item": "stitched_no_overlap", "Passed": bool((integrity_df[integrity_df["Check"] == "trace_no_overlap"]["Passed"]).all()), "Detail": "trace_no_overlap"},
        {"Item": "stitched_test_only", "Passed": bool((integrity_df[integrity_df["Check"] == "stitched_only_test_windows"]["Passed"]).all()), "Detail": "TEST_ONLY windows only"},
        {"Item": "drawdown_rebuilt", "Passed": bool((integrity_df[integrity_df["Check"] == "drawdown_reconstructible"]["Passed"]).all()), "Detail": "drawdown reconstructible from stitched equity"},
        {"Item": "stress_suite_executed", "Passed": bool(len(stress_df) > 0), "Detail": f"rows={len(stress_df)}"},
        {"Item": "monte_carlo_executed", "Passed": bool(len(mc_summary_df) > 0), "Detail": f"rows={len(mc_summary_df)}"},
        {"Item": "outputs_generated", "Passed": bool(set(required_outputs).issubset(existing)), "Detail": f"missing={sorted(set(required_outputs) - existing)}"},
        {"Item": "figures_generated", "Passed": bool(len(figure_paths) >= 5 and all(Path(p).exists() for p in figure_paths.values())), "Detail": f"figures={sorted(figure_paths.keys())}"},
        {"Item": "benchmarks_included", "Passed": bool({"LEGACY", "QQQ", "SPY"}.issubset(labels)), "Detail": "LEGACY/QQQ/SPY in stitched comparison"},
        {"Item": "alpha_nw_included", "Passed": bool(len(alpha_nw_df) > 0), "Detail": f"rows={len(alpha_nw_df)}"},
    ]
    checklist = pd.DataFrame(rows)
    ready = bool(checklist["Passed"].all())
    summary = [
        "# FULL readiness",
        "",
        f"Ready: {'YES' if ready else 'NO'}",
        "",
    ]
    for row in checklist.to_dict("records"):
        status = "PASS" if row["Passed"] else "FAIL"
        summary.append(f"- {status} | {row['Item']}: {row['Detail']}")
    return checklist, "\n".join(summary)


def build_full_report_text(wf: Dict[str, Any], cfg: Mahoraga14Config) -> str:
    comparison_df = _full_comparison_df(wf, cfg)
    fold_df = _build_fold_summary_fast(wf)
    primary_label = _variant_label_map(cfg)[cfg.full_primary_variant_key]
    lines = [
        "MAHORAGA 14.1 — FULL REPORT",
        "=" * 78,
        "",
        f"Primary audited candidate: {primary_label}",
        "FULL is configured as a directed audit mode aligned with FAST calibration.",
        "",
        "STITCHED COMPARISON",
        comparison_df.to_string(index=False),
        "",
        "FOLD SUMMARY",
        fold_df.to_string(index=False),
        "",
        "Detailed audit artifacts are written under the FULL outputs directory.",
    ]
    return "\n".join(lines)


def _build_final_report_full_text(
    cfg: Mahoraga14Config,
    comparison_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    pvalue_df: pd.DataFrame,
    alpha_nw_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    mc_summary_df: pd.DataFrame,
    event_df: pd.DataFrame,
    integrity_df: pd.DataFrame,
    maxdd_df: pd.DataFrame,
    selected_audit_df: pd.DataFrame,
    readiness_summary: str,
) -> str:
    primary_label = _variant_label_map(cfg)[cfg.full_primary_variant_key]
    stitched_tests = pvalue_df[pvalue_df["Segment"] == "STITCHED"].copy()
    lines = [
        "# Mahoraga 14.1 FULL audit",
        "",
        f"Primary candidate: **{primary_label}**",
        "",
        "## 1. Candidate description",
        "FULL 14.1 is a directed audit run. Calibration is intentionally kept aligned with FAST; the heavy work is the stitched/MaxDD/stress/Monte Carlo audit layer.",
        "",
        "## 2. Stitched OOS FULL",
        comparison_df.to_string(index=False),
        "",
        "## 3. Comparison vs LEGACY / QQQ / SPY",
        stitched_tests.to_string(index=False),
        "",
        "## 4. Alpha NW",
        alpha_nw_df.to_string(index=False),
        "",
        "## 5. p-values / q-values",
        pvalue_df.to_string(index=False),
        "",
        "## 6. Stress suite",
        stress_df.to_string(index=False),
        "",
        "## 7. Monte Carlo / bootstrap",
        mc_summary_df.to_string(index=False),
        "",
        "## 8. Continuation event study",
        event_df.to_string(index=False),
        "",
        "## 9. Stitched / MaxDD audit",
        integrity_df.to_string(index=False),
        "",
        maxdd_df.to_string(index=False),
        "",
        "## 10. Candidate audit",
        selected_audit_df.to_string(index=False),
        "",
        "## Fold summary",
        fold_df.to_string(index=False),
        "",
        "## Readiness",
        readiness_summary,
        "",
        "## Methodological conclusion",
        "The stitched OOS path is now traceable and auditable. Remaining uncertainty, if any, is captured explicitly in the readiness and audit files rather than hidden in aggregate metrics.",
    ]
    return "\n".join(lines)


def save_full_outputs(
    wf: Dict[str, Any],
    cfg: Mahoraga14Config,
    ff=None,
    costs: Optional[m6.CostsConfig] = None,
) -> Dict[str, pd.DataFrame]:
    ensure_dir(cfg.outputs_dir)
    labels = _variant_label_map(cfg)
    costs = costs or m6.CostsConfig()

    comparison_df = _full_comparison_df(wf, cfg)
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
    continuation_calibration_df = wf.get("continuation_calibration_df", pd.DataFrame()).copy()

    stitched = _stitched_map(wf, cfg)
    trace_df = _build_stitched_full_trace(wf, cfg)
    integrity_df = _build_stitched_integrity_checks(wf, cfg, trace_df)
    equity_df = pd.concat([_equity_df_from_object(labels[key], stitched[key]) for key in _full_variant_order(cfg)], ignore_index=True)
    drawdown_df = pd.concat([_drawdown_df_from_object(labels[key], stitched[key]) for key in _full_variant_order(cfg)], ignore_index=True)
    maxdd_df, maxdd_notes = _build_maxdd_audit(wf, cfg)
    stress_df, stress_notes, stress_paths_df = _build_stress_suite_full(wf, cfg, costs)

    mc_samples = []
    for key in [cfg.official_baseline_label, cfg.continuation_variant_key, cfg.combo_variant_key]:
        mc_samples.append(_run_stationary_bootstrap_mc(stitched[key], cfg, labels[key], "stationary_block_bootstrap", n_samples=400, block=20, seed=42 + len(mc_samples)))
        mc_samples.append(_run_friction_mc(stitched[key], cfg, labels[key], n_samples=300, seed=142 + len(mc_samples)))
    mc_samples.append(_run_local_parameter_neighborhood(wf, cfg, costs))
    mc_samples_df = pd.concat(mc_samples, ignore_index=True) if mc_samples else pd.DataFrame()
    mc_summary_df, mc_pct_df, mc_notes = _summarize_mc_results(mc_samples_df)

    selected_audit_df = _build_selected_candidate_audit(cfg, comparison_df, pq_df, stress_df, mc_pct_df)

    comparison_df.to_csv(Path(cfg.outputs_dir) / "stitched_comparison_full.csv", index=False)
    fold_df.to_csv(Path(cfg.outputs_dir) / "fold_summary_full.csv", index=False)
    floor_ceiling_df.to_csv(Path(cfg.outputs_dir) / "floor_ceiling_summary_full.csv", index=False)
    ablation_df.to_csv(Path(cfg.outputs_dir) / "ablation_full.csv", index=False)
    override_df.to_csv(Path(cfg.outputs_dir) / "override_usage_full.csv", index=False)
    continuation_df.to_csv(Path(cfg.outputs_dir) / "continuation_usage_full.csv", index=False)
    event_df.to_csv(Path(cfg.outputs_dir) / "continuation_event_study_full.csv", index=False)
    selected_df.to_csv(Path(cfg.outputs_dir) / "selected_candidates_full.csv", index=False)
    support_df.to_csv(Path(cfg.outputs_dir) / "selected_config_support_full.csv", index=False)
    pq_df.to_csv(Path(cfg.outputs_dir) / "pvalue_qvalue_full.csv", index=False)
    alpha_nw_df.to_csv(Path(cfg.outputs_dir) / "alpha_nw_full.csv", index=False)
    continuation_calibration_df.to_csv(Path(cfg.outputs_dir) / "continuation_calibration_full.csv", index=False)
    trace_df.to_csv(Path(cfg.outputs_dir) / "stitched_full_trace.csv", index=False)
    integrity_df.to_csv(Path(cfg.outputs_dir) / "stitched_integrity_checks.csv", index=False)
    equity_df.to_csv(Path(cfg.outputs_dir) / "equity_stitched_full.csv", index=False)
    drawdown_df.to_csv(Path(cfg.outputs_dir) / "drawdown_stitched_full.csv", index=False)
    maxdd_df.to_csv(Path(cfg.outputs_dir) / "maxdd_audit_full.csv", index=False)
    stress_df.to_csv(Path(cfg.outputs_dir) / "stress_suite_full.csv", index=False)
    mc_summary_df.to_csv(Path(cfg.outputs_dir) / "montecarlo_summary_full.csv", index=False)
    mc_pct_df.to_csv(Path(cfg.outputs_dir) / "montecarlo_percentiles_full.csv", index=False)
    selected_audit_df.to_csv(Path(cfg.outputs_dir) / "selected_candidate_audit_full.csv", index=False)

    with open(Path(cfg.outputs_dir) / "maxdd_audit_notes.md", "w", encoding="utf-8") as f:
        f.write(maxdd_notes)
    with open(Path(cfg.outputs_dir) / "stress_suite_audit.md", "w", encoding="utf-8") as f:
        f.write(stress_notes)
    with open(Path(cfg.outputs_dir) / "montecarlo_audit.md", "w", encoding="utf-8") as f:
        f.write(mc_notes)
    if len(stress_paths_df):
        stress_paths_df.to_csv(Path(cfg.outputs_dir) / "stress_paths_trace.csv", index=False)

    figure_paths = _generate_figures(wf, cfg, mc_samples_df)
    output_paths = list(Path(cfg.outputs_dir).glob("*"))
    checklist_df, readiness_summary = _build_readiness_checklist(
        cfg,
        comparison_df,
        alpha_nw_df,
        integrity_df,
        stress_df,
        mc_summary_df,
        figure_paths,
        output_paths,
    )
    checklist_df.to_csv(Path(cfg.outputs_dir) / "full_readiness_checklist.csv", index=False)
    with open(Path(cfg.outputs_dir) / "full_readiness_summary.md", "w", encoding="utf-8") as f:
        f.write(readiness_summary)

    final_report = _build_final_report_full_text(
        cfg,
        comparison_df,
        fold_df,
        pq_df,
        alpha_nw_df,
        stress_df,
        mc_summary_df,
        event_df,
        integrity_df,
        maxdd_df,
        selected_audit_df,
        readiness_summary,
    )
    with open(Path(cfg.outputs_dir) / "final_report_full.md", "w", encoding="utf-8") as f:
        f.write(final_report)
    with open(Path(cfg.outputs_dir) / "final_report_full.txt", "w", encoding="utf-8") as f:
        f.write(final_report)

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
        "continuation_calibration_full": continuation_calibration_df,
        "stitched_full_trace": trace_df,
        "stitched_integrity_checks": integrity_df,
        "equity_stitched_full": equity_df,
        "drawdown_stitched_full": drawdown_df,
        "maxdd_audit_full": maxdd_df,
        "stress_suite_full": stress_df,
        "montecarlo_summary_full": mc_summary_df,
        "montecarlo_percentiles_full": mc_pct_df,
        "selected_candidate_audit_full": selected_audit_df,
        "full_readiness_checklist": checklist_df,
    }
