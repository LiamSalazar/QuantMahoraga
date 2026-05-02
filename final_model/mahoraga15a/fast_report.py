from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_executor import rebuild_ls_fold
from idio_short_placeholder import build_sparse_idio_short_interface
from mahoraga15a_config import Mahoraga15AConfig
from mahoraga15a_utils import (
    alpha_nw,
    beta,
    bhy_qvalues,
    capture_ratio,
    ensure_dir,
    paired_ttest_pvalue,
    return_per_exposure,
    rolling_ridge_beta,
    stationary_bootstrap,
    stitch_objects,
    summarize_object,
)


def _objects_map(wf: Dict[str, Any], cfg: Mahoraga15AConfig) -> Dict[str, Dict[str, Any]]:
    return {
        "QQQ": wf["stitched_benchmarks"]["QQQ"],
        "SPY": wf["stitched_benchmarks"]["SPY"],
        cfg.official_long_label: wf["frozen_long"],
        cfg.delevered_label: wf["stitched_delevered_control"],
        cfg.ls_label: wf["stitched_ls"],
    }


def _series(obj: Dict[str, Any], key: str, fallback: float = 0.0) -> pd.Series:
    if key in obj:
        return pd.Series(obj[key], dtype=float)
    idx = pd.Index(obj["returns"].index)
    return pd.Series(fallback, index=idx, dtype=float)


def _slice_object(obj: Dict[str, Any], start: pd.Timestamp, end: pd.Timestamp, cfg: Mahoraga15AConfig, label: str) -> Dict[str, Any]:
    returns = pd.Series(obj["returns"], dtype=float).loc[start:end].fillna(0.0)
    if len(returns) == 0:
        return {
            "label": label,
            "returns": returns,
            "equity": pd.Series(dtype=float),
            "exposure": pd.Series(dtype=float),
            "turnover": pd.Series(dtype=float),
        }
    exposure = _series(obj, "exposure", 0.0).reindex(returns.index).fillna(0.0)
    turnover = _series(obj, "turnover", 0.0).reindex(returns.index).fillna(0.0)
    equity = cfg.capital_initial * (1.0 + returns).cumprod()
    return {
        "label": label,
        "returns": returns,
        "equity": equity,
        "exposure": exposure,
        "turnover": turnover,
        "gross_long": _series(obj, "gross_long", 0.0).reindex(returns.index).fillna(0.0),
        "gross_short": _series(obj, "gross_short", 0.0).reindex(returns.index).fillna(0.0),
        "net_exposure": _series(obj, "net_exposure", 0.0).reindex(returns.index).fillna(0.0),
    }


def _metrics_row(label: str, obj: Dict[str, Any], qqq_obj: Dict[str, Any], spy_obj: Dict[str, Any], cfg: Mahoraga15AConfig) -> Dict[str, Any]:
    summary = summarize_object(obj, cfg, label)
    alpha_qqq = alpha_nw(obj["returns"], qqq_obj["returns"], cfg, f"{label}_QQQ")
    alpha_spy = alpha_nw(obj["returns"], spy_obj["returns"], cfg, f"{label}_SPY")
    gross_long = _series(obj, "gross_long", fallback=float(summary["AvgExposure"]))
    gross_short = _series(obj, "gross_short", fallback=0.0)
    net_exp = _series(obj, "net_exposure", fallback=float(summary["AvgExposure"]))
    exposure = _series(obj, "exposure", fallback=float(summary["AvgExposure"]))
    turnover = _series(obj, "turnover", fallback=0.0)
    return {
        "Variant": label,
        "CAGR%": round(summary["CAGR"] * 100.0, 2),
        "Sharpe": round(summary["Sharpe"], 4),
        "Sortino": round(summary["Sortino"], 4),
        "MaxDD%": round(summary["MaxDD"] * 100.0, 2),
        "AvgExposure": round(float(exposure.mean()), 4),
        "GrossLong": round(float(gross_long.mean()), 4),
        "GrossShort": round(float(gross_short.mean()), 4),
        "NetExposure": round(float(net_exp.mean()), 4),
        "AvgTurnover": round(float(turnover.mean()), 4),
        "ReturnPerExposure": round(return_per_exposure(obj["returns"], exposure), 6),
        "BetaQQQ": round(beta(obj["returns"], qqq_obj["returns"]), 4),
        "BetaSPY": round(beta(obj["returns"], spy_obj["returns"]), 4),
        "AlphaNW_QQQ": round(alpha_qqq["alpha_ann"], 6) if np.isfinite(alpha_qqq["alpha_ann"]) else np.nan,
        "AlphaNW_SPY": round(alpha_spy["alpha_ann"], 6) if np.isfinite(alpha_spy["alpha_ann"]) else np.nan,
        "UpsideCaptureQQQ": round(capture_ratio(obj["returns"], qqq_obj["returns"], upside=True), 4),
        "DownsideCaptureQQQ": round(capture_ratio(obj["returns"], qqq_obj["returns"], upside=False), 4),
        "UpsideCaptureSPY": round(capture_ratio(obj["returns"], spy_obj["returns"], upside=True), 4),
        "DownsideCaptureSPY": round(capture_ratio(obj["returns"], spy_obj["returns"], upside=False), 4),
    }


def build_stitched_comparison_fast(wf: Dict[str, Any], cfg: Mahoraga15AConfig) -> pd.DataFrame:
    objects = _objects_map(wf, cfg)
    qqq_obj = objects["QQQ"]
    spy_obj = objects["SPY"]
    order = [cfg.official_long_label, cfg.delevered_label, cfg.ls_label, "QQQ", "SPY"]
    return pd.DataFrame([_metrics_row(key, objects[key], qqq_obj, spy_obj, cfg) for key in order])


def build_pairwise_pq_fast(wf: Dict[str, Any], cfg: Mahoraga15AConfig) -> pd.DataFrame:
    objects = _objects_map(wf, cfg)
    pairs = [
        (cfg.delevered_label, cfg.official_long_label),
        (cfg.ls_label, cfg.official_long_label),
        (cfg.ls_label, cfg.delevered_label),
        (cfg.official_long_label, "QQQ"),
        (cfg.official_long_label, "SPY"),
        (cfg.delevered_label, "QQQ"),
        (cfg.delevered_label, "SPY"),
        (cfg.ls_label, "QQQ"),
        (cfg.ls_label, "SPY"),
    ]
    rows = []
    for target, reference in pairs:
        rows.append(
            {
                "Target": target,
                "Reference": reference,
                "Comparison": f"{target}_vs_{reference}",
                "p_value": paired_ttest_pvalue(objects[target]["returns"] - objects[reference]["returns"], alternative="greater"),
            }
        )
    df = pd.DataFrame(rows)
    df["q_value"] = bhy_qvalues(df["p_value"].values, alpha=cfg.bhy_alpha)
    df["p_value"] = df["p_value"].round(6)
    df["q_value"] = df["q_value"].round(6)
    return df


def build_delevered_control_fast(comparison_df: pd.DataFrame, cfg: Mahoraga15AConfig) -> pd.DataFrame:
    comp = comparison_df.set_index("Variant")
    order = [cfg.official_long_label, cfg.delevered_label, cfg.ls_label]
    rows = []
    for label in order:
        row = comp.loc[label].to_dict()
        row["DeltaSharpe_vs_Long"] = round(float(comp.loc[label, "Sharpe"] - comp.loc[cfg.official_long_label, "Sharpe"]), 4)
        row["DeltaSortino_vs_Long"] = round(float(comp.loc[label, "Sortino"] - comp.loc[cfg.official_long_label, "Sortino"]), 4)
        row["DeltaCAGR_vs_Long%"] = round(float(comp.loc[label, "CAGR%"] - comp.loc[cfg.official_long_label, "CAGR%"]), 2)
        row["DeltaBetaQQQ_vs_Long"] = round(float(comp.loc[label, "BetaQQQ"] - comp.loc[cfg.official_long_label, "BetaQQQ"]), 4)
        row["DeltaBetaSPY_vs_Long"] = round(float(comp.loc[label, "BetaSPY"] - comp.loc[cfg.official_long_label, "BetaSPY"]), 4)
        if label != cfg.delevered_label:
            row["DeltaSharpe_vs_Delevered"] = round(float(comp.loc[label, "Sharpe"] - comp.loc[cfg.delevered_label, "Sharpe"]), 4)
            row["DeltaSortino_vs_Delevered"] = round(float(comp.loc[label, "Sortino"] - comp.loc[cfg.delevered_label, "Sortino"]), 4)
            row["DeltaCAGR_vs_Delevered%"] = round(float(comp.loc[label, "CAGR%"] - comp.loc[cfg.delevered_label, "CAGR%"]), 2)
        else:
            row["DeltaSharpe_vs_Delevered"] = 0.0
            row["DeltaSortino_vs_Delevered"] = 0.0
            row["DeltaCAGR_vs_Delevered%"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_beta_decomposition_fast(wf: Dict[str, Any], cfg: Mahoraga15AConfig) -> pd.DataFrame:
    allocator = wf["allocator_trace"].copy()
    ls_obj = wf["stitched_ls"]
    long_obj = wf["frozen_long"]
    control_obj = wf["stitched_delevered_control"]
    qqq_r = wf["stitched_benchmarks"]["QQQ"]["returns"].reindex(allocator.index).fillna(0.0)
    spy_r = wf["stitched_benchmarks"]["SPY"]["returns"].reindex(allocator.index).fillna(0.0)
    allocator["observed_beta_qqq_long"] = rolling_ridge_beta(long_obj["returns"].reindex(allocator.index).fillna(0.0), qqq_r, cfg.hedge_beta_window, cfg.hedge_beta_min_obs, cfg.hedge_ridge_alpha)
    allocator["observed_beta_spy_long"] = rolling_ridge_beta(long_obj["returns"].reindex(allocator.index).fillna(0.0), spy_r, cfg.hedge_beta_window, cfg.hedge_beta_min_obs, cfg.hedge_ridge_alpha)
    allocator["observed_beta_qqq_delevered"] = rolling_ridge_beta(control_obj["returns"].reindex(allocator.index).fillna(0.0), qqq_r, cfg.hedge_beta_window, cfg.hedge_beta_min_obs, cfg.hedge_ridge_alpha)
    allocator["observed_beta_spy_delevered"] = rolling_ridge_beta(control_obj["returns"].reindex(allocator.index).fillna(0.0), spy_r, cfg.hedge_beta_window, cfg.hedge_beta_min_obs, cfg.hedge_ridge_alpha)
    allocator["observed_beta_qqq_ls"] = rolling_ridge_beta(ls_obj["returns"].reindex(allocator.index).fillna(0.0), qqq_r, cfg.hedge_beta_window, cfg.hedge_beta_min_obs, cfg.hedge_ridge_alpha)
    allocator["observed_beta_spy_ls"] = rolling_ridge_beta(ls_obj["returns"].reindex(allocator.index).fillna(0.0), spy_r, cfg.hedge_beta_window, cfg.hedge_beta_min_obs, cfg.hedge_ridge_alpha)
    allocator = allocator.reset_index().rename(columns={"index": "Date"})
    keep = [
        "Date",
        "fold",
        "long_beta_qqq",
        "long_beta_spy",
        "target_beta_qqq",
        "target_beta_spy",
        "beta_gap_qqq",
        "beta_gap_spy",
        "predicted_beta_qqq",
        "predicted_beta_spy",
        "observed_beta_qqq_long",
        "observed_beta_spy_long",
        "observed_beta_qqq_delevered",
        "observed_beta_spy_delevered",
        "observed_beta_qqq_ls",
        "observed_beta_spy_ls",
        "qqq_short_budget",
        "spy_short_budget",
    ]
    return allocator[[c for c in keep if c in allocator.columns]]


def build_exposure_trace_fast(wf: Dict[str, Any], cfg: Mahoraga15AConfig) -> pd.DataFrame:
    ls = wf["stitched_ls"]
    long_only = wf["frozen_long"]
    control = wf["stitched_delevered_control"]
    allocator = wf["allocator_trace"]
    idx = pd.DatetimeIndex(ls["returns"].index)
    return pd.DataFrame(
        {
            "Date": idx,
            "LongOnlyGrossLong": _series(long_only, "gross_long", 0.0).reindex(idx).fillna(0.0).values,
            "DeleveredGrossLong": _series(control, "gross_long", 0.0).reindex(idx).fillna(0.0).values,
            "LS_GrossLong": _series(ls, "gross_long", 0.0).reindex(idx).fillna(0.0).values,
            "LS_GrossShort": _series(ls, "gross_short", 0.0).reindex(idx).fillna(0.0).values,
            "LS_NetExposure": _series(ls, "net_exposure", 0.0).reindex(idx).fillna(0.0).values,
            "LS_GrossExposure": _series(ls, "gross_exposure", 0.0).reindex(idx).fillna(0.0).values,
            "CashBuffer": allocator["cash_buffer"].reindex(idx).ffill().fillna(0.0).values,
        }
    )


def _window_overlap(a_start: pd.Timestamp, a_end: pd.Timestamp, b_start: pd.Timestamp, b_end: pd.Timestamp) -> bool:
    return not (a_end < b_start or b_end < a_start)


def build_crisis_windows(wf: Dict[str, Any], cfg: Mahoraga15AConfig) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    idx = pd.DatetimeIndex(wf["stitched_ls"]["returns"].index)
    if len(idx) == 0:
        return []
    windows = [
        ("CRISIS_2020_CRASH", pd.Timestamp("2020-02-19"), pd.Timestamp("2020-04-30")),
        ("CRISIS_2022_TECH_BEAR", pd.Timestamp("2022-01-03"), pd.Timestamp("2022-12-30")),
    ]
    windows = [(name, max(start, idx.min()), min(end, idx.max())) for name, start, end in windows if start <= idx.max() and end >= idx.min()]
    qqq = pd.Series(wf["stitched_benchmarks"]["QQQ"]["returns"], dtype=float).reindex(idx).fillna(0.0)
    lookback = min(63, max(21, len(idx) // 8))
    rolling = (1.0 + qqq).rolling(lookback, min_periods=lookback).apply(np.prod, raw=True) - 1.0
    extra = []
    for dt in rolling.dropna().sort_values().index:
        end = pd.Timestamp(dt)
        pos = idx.get_indexer([end])[0]
        start = idx[max(0, pos - lookback + 1)]
        if any(_window_overlap(start, end, s, e) for _, s, e in windows + extra):
            continue
        extra.append((f"AUTO_QQQ_STRESS_{len(extra) + 1}", start, end))
        if len(extra) >= 2:
            break
    return windows + extra


def _pnl_frame(wf: Dict[str, Any], cfg: Mahoraga15AConfig) -> pd.DataFrame:
    ls = wf["stitched_ls"]
    idx = pd.DatetimeIndex(ls["returns"].index)
    eq_prev = pd.Series(ls["equity"], dtype=float).shift(1).reindex(idx)
    if len(eq_prev):
        eq_prev.iloc[0] = cfg.capital_initial
    eq_prev = eq_prev.ffill().fillna(cfg.capital_initial)
    state = wf["state_trace"].reindex(idx).fillna(0.0)
    long_r = _series(ls, "long_net_contribution", 0.0).reindex(idx).fillna(0.0)
    short_r = _series(ls, "short_net_contribution", 0.0).reindex(idx).fillna(0.0)
    total_r = pd.Series(ls["returns"], dtype=float).reindex(idx).fillna(0.0)
    interaction_r = total_r - long_r - short_r
    cash_drag_r = -_series(ls, "cash_buffer", 0.0).reindex(idx).fillna(0.0) * state["long_return_unit_gross"].reindex(idx).fillna(0.0)

    out = pd.DataFrame(index=idx)
    out["equity_prev"] = eq_prev
    out["return_total"] = total_r
    out["return_long"] = long_r
    out["return_systematic_hedge"] = short_r
    out["return_interaction"] = interaction_r
    out["return_cash_drag"] = cash_drag_r
    for col in ["return_total", "return_long", "return_systematic_hedge", "return_interaction", "return_cash_drag"]:
        out[col.replace("return_", "pnl_")] = out["equity_prev"] * out[col]
    out["gross_short"] = _series(ls, "gross_short", 0.0).reindex(idx).fillna(0.0)
    out["net_exposure"] = _series(ls, "net_exposure", 0.0).reindex(idx).fillna(0.0)
    return out


def _pnl_segment_row(segment_type: str, segment: str, start: pd.Timestamp, end: pd.Timestamp, pnl_df: pd.DataFrame, cfg: Mahoraga15AConfig) -> Dict[str, Any]:
    sub = pnl_df.loc[start:end]
    if len(sub) == 0:
        return {}
    return {
        "SegmentType": segment_type,
        "Segment": segment,
        "Start": pd.Timestamp(sub.index.min()),
        "End": pd.Timestamp(sub.index.max()),
        "Days": int(len(sub)),
        "PnLTotalPctInit": round(float(sub["pnl_total"].sum() / cfg.capital_initial), 6),
        "PnLLongPctInit": round(float(sub["pnl_long"].sum() / cfg.capital_initial), 6),
        "PnLSystematicHedgePctInit": round(float(sub["pnl_systematic_hedge"].sum() / cfg.capital_initial), 6),
        "PnLInteractionPctInit": round(float(sub["pnl_interaction"].sum() / cfg.capital_initial), 6),
        "PnLCashDragPctInit": round(float(sub["pnl_cash_drag"].sum() / cfg.capital_initial), 6),
        "AvgGrossShort": round(float(sub["gross_short"].mean()), 4),
        "AvgNetExposure": round(float(sub["net_exposure"].mean()), 4),
    }


def build_pnl_attribution_fast(wf: Dict[str, Any], cfg: Mahoraga15AConfig) -> pd.DataFrame:
    pnl_df = _pnl_frame(wf, cfg)
    rows = [_pnl_segment_row("STITCHED", "FULL_OOS", pnl_df.index.min(), pnl_df.index.max(), pnl_df, cfg)]
    for payload in wf["ls_fold_payloads"]:
        rows.append(_pnl_segment_row("FOLD", f"FOLD_{int(payload['fold'])}", payload["test_start"], payload["test_end"], pnl_df, cfg))
    for year, sub in pnl_df.groupby(pnl_df.index.year):
        rows.append(_pnl_segment_row("YEAR", str(int(year)), pd.Timestamp(sub.index.min()), pd.Timestamp(sub.index.max()), pnl_df, cfg))
    for name, start, end in build_crisis_windows(wf, cfg):
        rows.append(_pnl_segment_row("CRISIS", name, start, end, pnl_df, cfg))
    return pd.DataFrame([row for row in rows if row])


def _local_window_metrics(obj: Dict[str, Any], bench_q: Dict[str, Any], bench_s: Dict[str, Any], start: pd.Timestamp, end: pd.Timestamp, cfg: Mahoraga15AConfig, label: str) -> Dict[str, Any]:
    sub = _slice_object(obj, start, end, cfg, label)
    if len(sub["returns"]) == 0:
        return {}
    summ = summarize_object(sub, cfg, label)
    qqq_r = pd.Series(bench_q["returns"], dtype=float).reindex(sub["returns"].index).fillna(0.0)
    spy_r = pd.Series(bench_s["returns"], dtype=float).reindex(sub["returns"].index).fillna(0.0)
    return {
        "Return%": round(float((1.0 + sub["returns"]).prod() - 1.0) * 100.0, 2),
        "Sharpe": round(float(summ["Sharpe"]), 4),
        "Sortino": round(float(summ["Sortino"]), 4),
        "MaxDD%": round(float(summ["MaxDD"]) * 100.0, 2),
        "BetaQQQ": round(beta(sub["returns"], qqq_r), 4),
        "BetaSPY": round(beta(sub["returns"], spy_r), 4),
        "GrossShort": round(float(sub["gross_short"].mean()), 4),
    }


def build_crisis_window_scorecard_fast(wf: Dict[str, Any], cfg: Mahoraga15AConfig, pnl_attr_df: pd.DataFrame) -> pd.DataFrame:
    bench_q = wf["stitched_benchmarks"]["QQQ"]
    bench_s = wf["stitched_benchmarks"]["SPY"]
    long_obj = wf["frozen_long"]
    control_obj = wf["stitched_delevered_control"]
    ls_obj = wf["stitched_ls"]
    hedge_map = pnl_attr_df[pnl_attr_df["SegmentType"] == "CRISIS"].set_index("Segment") if len(pnl_attr_df) else pd.DataFrame()
    rows = []
    for name, start, end in build_crisis_windows(wf, cfg):
        ls_m = _local_window_metrics(ls_obj, bench_q, bench_s, start, end, cfg, f"{cfg.ls_label}_{name}")
        long_m = _local_window_metrics(long_obj, bench_q, bench_s, start, end, cfg, f"{cfg.official_long_label}_{name}")
        control_m = _local_window_metrics(control_obj, bench_q, bench_s, start, end, cfg, f"{cfg.delevered_label}_{name}")
        if not ls_m:
            continue
        hedge_pnl = float(hedge_map.loc[name, "PnLSystematicHedgePctInit"]) if name in hedge_map.index else np.nan
        rows.append(
            {
                "Window": name,
                "Start": start,
                "End": end,
                "LS_Return%": ls_m["Return%"],
                "LS_Sharpe": ls_m["Sharpe"],
                "LS_Sortino": ls_m["Sortino"],
                "LS_MaxDD%": ls_m["MaxDD%"],
                "LS_BetaQQQ": ls_m["BetaQQQ"],
                "LS_BetaSPY": ls_m["BetaSPY"],
                "LS_GrossShort": ls_m["GrossShort"],
                "HedgePnLPctInit": round(hedge_pnl, 6) if np.isfinite(hedge_pnl) else np.nan,
                "LongOnly_Return%": long_m["Return%"],
                "LongOnly_Sharpe": long_m["Sharpe"],
                "LongOnly_Sortino": long_m["Sortino"],
                "LongOnly_MaxDD%": long_m["MaxDD%"],
                "Delevered_Return%": control_m["Return%"],
                "Delevered_Sharpe": control_m["Sharpe"],
                "Delevered_Sortino": control_m["Sortino"],
                "Delevered_MaxDD%": control_m["MaxDD%"],
                "DeltaReturn_vs_LongOnly%": round(float(ls_m["Return%"] - long_m["Return%"]), 2),
                "DeltaReturn_vs_Delevered%": round(float(ls_m["Return%"] - control_m["Return%"]), 2),
                "DeltaSharpe_vs_LongOnly": round(float(ls_m["Sharpe"] - long_m["Sharpe"]), 4),
                "DeltaSharpe_vs_Delevered": round(float(ls_m["Sharpe"] - control_m["Sharpe"]), 4),
                "DeltaSortino_vs_LongOnly": round(float(ls_m["Sortino"] - long_m["Sortino"]), 4),
                "DeltaSortino_vs_Delevered": round(float(ls_m["Sortino"] - control_m["Sortino"]), 4),
                "DeltaMaxDD_vs_LongOnly%": round(float(ls_m["MaxDD%"] - long_m["MaxDD%"]), 2),
                "DeltaMaxDD_vs_Delevered%": round(float(ls_m["MaxDD%"] - control_m["MaxDD%"]), 2),
            }
        )
    return pd.DataFrame(rows)


def build_allocator_response_fast(wf: Dict[str, Any], cfg: Mahoraga15AConfig) -> pd.DataFrame:
    trace = wf["allocator_trace"].copy().reset_index().rename(columns={"index": "Date"})
    trace["ShortActive"] = (trace["systematic_short_budget"] > 1e-6).astype(int)
    trace["ShortRegime"] = np.select(
        [trace["systematic_short_budget"] <= 0.05, trace["systematic_short_budget"] <= 0.15],
        ["BENIGN_0_5", "STRESS_5_15"],
        default="CRISIS_15_35",
    )
    keep = [
        "Date",
        "fold",
        "structural_fragility",
        "continuation_pressure",
        "break_risk",
        "continuation_relief",
        "benchmark_weakness",
        "bear_persistence",
        "drawdown_pressure",
        "corr_pressure",
        "realized_vol_pressure",
        "stress_intensity",
        "directional_bear",
        "crisis_activation",
        "crisis_transition",
        "crisis_persistence",
        "transition_shock",
        "target_beta_qqq",
        "target_beta_spy",
        "beta_gap_qqq",
        "beta_gap_spy",
        "hedge_permission",
        "reaction_multiplier",
        "long_budget",
        "crisis_short_budget",
        "systematic_short_budget",
        "cash_buffer",
        "net_exposure",
        "gross_exposure",
        "short_cap_dynamic",
        "net_exposure_floor_dynamic",
        "qqq_short_budget",
        "spy_short_budget",
        "predicted_beta_qqq",
        "predicted_beta_spy",
        "short_speed_applied",
        "reaction_multiplier_applied",
        "long_step_applied",
        "short_step_applied",
        "short_velocity_state",
        "ShortActive",
        "ShortRegime",
    ]
    return trace[[c for c in keep if c in trace.columns]]


def build_short_activity_summary_fast(wf: Dict[str, Any], cfg: Mahoraga15AConfig, pnl_attr_df: pd.DataFrame) -> pd.DataFrame:
    allocator = wf["allocator_trace"]
    ls = wf["stitched_ls"]
    hedge_pnl_map = pnl_attr_df.set_index(["SegmentType", "Segment"]) if len(pnl_attr_df) else pd.DataFrame()

    def segment_row(segment_type: str, segment: str, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Any]:
        sub = allocator.loc[start:end]
        if len(sub) == 0:
            return {}
        key = (segment_type, segment)
        hedge_pnl = hedge_pnl_map.loc[key, "PnLSystematicHedgePctInit"] if key in hedge_pnl_map.index else np.nan
        return {
            "SegmentType": segment_type,
            "Segment": segment,
            "Start": pd.Timestamp(sub.index.min()),
            "End": pd.Timestamp(sub.index.max()),
            "AvgGrossShort": round(float(sub["systematic_short_budget"].mean()), 4),
            "MaxGrossShort": round(float(sub["systematic_short_budget"].max()), 4),
            "ActiveDayPct": round(float((sub["systematic_short_budget"] > 1e-6).mean()), 4),
            "AvgQQQShort": round(float(sub["qqq_short_budget"].mean()), 4),
            "AvgSPYShort": round(float(sub["spy_short_budget"].mean()), 4),
            "AvgLongBudget": round(float(sub["long_budget"].mean()), 4),
            "AvgNetExposure": round(float(sub["net_exposure"].mean()), 4),
            "AvgCashBuffer": round(float(sub["cash_buffer"].mean()), 4),
            "AvgTurnover": round(float(_series(ls, "turnover", 0.0).reindex(sub.index).fillna(0.0).mean()), 4),
            "HedgePnLPctInit": round(float(hedge_pnl), 6) if np.isfinite(hedge_pnl) else np.nan,
        }

    rows = [segment_row("STITCHED", "FULL_OOS", allocator.index.min(), allocator.index.max())]
    for payload in wf["ls_fold_payloads"]:
        rows.append(segment_row("FOLD", f"FOLD_{int(payload['fold'])}", payload["test_start"], payload["test_end"]))
    for year, sub in allocator.groupby(allocator.index.year):
        rows.append(segment_row("YEAR", str(int(year)), pd.Timestamp(sub.index.min()), pd.Timestamp(sub.index.max())))
    for name, start, end in build_crisis_windows(wf, cfg):
        rows.append(segment_row("CRISIS", name, start, end))
    return pd.DataFrame([row for row in rows if row])


def _apply_empirical_path_stress(obj: Dict[str, Any], cfg: Mahoraga15AConfig) -> Dict[str, Any]:
    returns = pd.Series(obj["returns"], dtype=float).copy()
    block = max(10, cfg.mc_block_size)
    if len(returns) < block * 4:
        return obj
    roll = (1.0 + returns).rolling(block).apply(np.prod, raw=True) - 1.0
    source_end = int(returns.index.get_loc(roll.dropna().sort_values().index[0]))
    src_start = max(0, source_end - block + 1)
    tgt_start = max(block, len(returns) // 2)
    tgt_end = min(len(returns), tgt_start + block)
    stressed = returns.copy()
    src_block = returns.iloc[src_start : src_start + (tgt_end - tgt_start)].values
    tgt_block = stressed.iloc[tgt_start:tgt_end].values
    stressed.iloc[tgt_start:tgt_end] = np.clip((1.0 + tgt_block) * (1.0 + src_block) - 1.0, -0.95, None)
    tc = _series(obj, "transaction_cost", 0.0).reindex(stressed.index).fillna(0.0)
    eq = cfg.capital_initial * (1.0 + stressed).cumprod()
    out = dict(obj)
    out["returns"] = stressed
    out["gross_returns"] = stressed + tc
    out["transaction_cost"] = tc
    out["equity"] = eq
    return out


def _stress_metrics_row(
    variant: str,
    scenario: str,
    note: str,
    obj: Dict[str, Any],
    ls_base: Dict[str, Any],
    long_base: Dict[str, Any],
    control_base: Dict[str, Any],
    qqq_obj: Dict[str, Any],
    spy_obj: Dict[str, Any],
    cfg: Mahoraga15AConfig,
) -> Dict[str, Any]:
    summary = summarize_object(obj, cfg, f"{variant}_{scenario}")
    ls_summary = summarize_object(ls_base, cfg, cfg.ls_label)
    long_summary = summarize_object(long_base, cfg, cfg.official_long_label)
    control_summary = summarize_object(control_base, cfg, cfg.delevered_label)
    alpha_qqq = alpha_nw(obj["returns"], qqq_obj["returns"], cfg, f"{variant}_{scenario}_QQQ")
    alpha_spy = alpha_nw(obj["returns"], spy_obj["returns"], cfg, f"{variant}_{scenario}_SPY")
    return {
        "Variant": variant,
        "Scenario": scenario,
        "ScenarioNote": note,
        "CAGR%": round(summary["CAGR"] * 100.0, 2),
        "Sharpe": round(summary["Sharpe"], 4),
        "Sortino": round(summary["Sortino"], 4),
        "MaxDD%": round(summary["MaxDD"] * 100.0, 2),
        "GrossShort": round(float(_series(obj, "gross_short", 0.0).mean()), 4),
        "BetaQQQ": round(beta(obj["returns"], qqq_obj["returns"]), 4),
        "BetaSPY": round(beta(obj["returns"], spy_obj["returns"]), 4),
        "AlphaNW_QQQ": round(alpha_qqq["alpha_ann"], 6) if np.isfinite(alpha_qqq["alpha_ann"]) else np.nan,
        "AlphaNW_SPY": round(alpha_spy["alpha_ann"], 6) if np.isfinite(alpha_spy["alpha_ann"]) else np.nan,
        "DeltaCAGR_vs_LSBase%": round((summary["CAGR"] - ls_summary["CAGR"]) * 100.0, 2),
        "DeltaSharpe_vs_LSBase": round(summary["Sharpe"] - ls_summary["Sharpe"], 4),
        "DeltaSortino_vs_LSBase": round(summary["Sortino"] - ls_summary["Sortino"], 4),
        "DeltaMaxDD_vs_LSBase%": round((summary["MaxDD"] - ls_summary["MaxDD"]) * 100.0, 2),
        "DeltaSharpe_vs_Delevered": round(summary["Sharpe"] - control_summary["Sharpe"], 4),
        "DeltaSortino_vs_Delevered": round(summary["Sortino"] - control_summary["Sortino"], 4),
        "DeltaCAGR_vs_Delevered%": round((summary["CAGR"] - control_summary["CAGR"]) * 100.0, 2),
        "DeltaSharpe_vs_LongOnly": round(summary["Sharpe"] - long_summary["Sharpe"], 4),
        "DeltaSortino_vs_LongOnly": round(summary["Sortino"] - long_summary["Sortino"], 4),
        "DeltaCAGR_vs_LongOnly%": round((summary["CAGR"] - long_summary["CAGR"]) * 100.0, 2),
    }


def _stitch_scenario_ls(
    wf: Dict[str, Any],
    cfg: Mahoraga15AConfig,
    costs,
    override: Dict[str, float],
    label_suffix: str,
):
    fold_meta = [(p["fold"], p["test_start"], p["test_end"]) for p in wf["ls_fold_payloads"]]
    fold_objs = []
    for payload in wf["ls_fold_payloads"]:
        scenario_costs = deepcopy(costs)
        if "cost_mult" in override:
            scenario_costs.commission *= float(override["cost_mult"])
            scenario_costs.slippage *= float(override["cost_mult"])
        if "extra_slippage" in override:
            scenario_costs.slippage += float(override["extra_slippage"])
        stressed = rebuild_ls_fold(payload, cfg, scenario_costs, override=override)
        fold_objs.append(stressed["ls_obj"])
    return stitch_objects(fold_objs, fold_meta, cfg, f"{cfg.ls_label}_{label_suffix}")


def build_stress_suite_fast(wf: Dict[str, Any], cfg: Mahoraga15AConfig, costs) -> pd.DataFrame:
    qqq_obj = wf["stitched_benchmarks"]["QQQ"]
    spy_obj = wf["stitched_benchmarks"]["SPY"]
    long_base = wf["frozen_long"]
    control_base = wf["stitched_delevered_control"]
    ls_base = wf["stitched_ls"]
    rows = [
        _stress_metrics_row(cfg.official_long_label, "BASELINE_LONG_ONLY", "official frozen 14.1 long-only reference", long_base, ls_base, long_base, control_base, qqq_obj, spy_obj, cfg),
        _stress_metrics_row(cfg.delevered_label, "BASELINE_DELEVERED_CONTROL", "path-matched delevered control without real short", control_base, ls_base, long_base, control_base, qqq_obj, spy_obj, cfg),
        _stress_metrics_row(cfg.ls_label, "BASELINE_15A2", "Mahoraga15A2 unstressed stitched OOS", ls_base, ls_base, long_base, control_base, qqq_obj, spy_obj, cfg),
    ]
    scenarios = [
        ("COST_PLUS_25", {"cost_mult": 1.25}, "commission/slippage x1.25"),
        ("COST_PLUS_50", {"cost_mult": 1.50}, "commission/slippage x1.50"),
        ("COST_PLUS_100", {"cost_mult": 2.00}, "commission/slippage x2.00"),
        ("EXTRA_SLIPPAGE", {"extra_slippage": cfg.stress_extra_slippage}, "extra slippage +5bps"),
        ("EXECUTION_DELAY_1_REBALANCE", {"delay_days": 5 * cfg.stress_delay_rebalances}, "delay long budget and hedge one weekly rebalance"),
        ("EXECUTION_DELAY_2_REBALANCES", {"delay_days": 5 * cfg.stress_delay_rebalances_2}, "delay long budget and hedge two weekly rebalances"),
        ("HEDGE_RATIO_UNDERESTIMATED", {"hedge_ratio_mult": cfg.stress_hedge_ratio_under_mult}, "systematic hedge scaled to 75%"),
        ("HEDGE_RATIO_OVERESTIMATED", {"hedge_ratio_mult": cfg.stress_hedge_ratio_over_mult}, "systematic hedge scaled to 125%"),
        ("REACTION_SLOWER", {"up_speed_mult": cfg.stress_reaction_slower_mult, "down_speed_mult": cfg.stress_reaction_slower_mult}, "allocator reacts slower"),
        ("REACTION_FASTER", {"up_speed_mult": cfg.stress_reaction_faster_mult, "down_speed_mult": cfg.stress_reaction_faster_mult}, "allocator reacts faster"),
        ("HEDGE_LEAD_1_REBALANCE", {"hedge_shift_days": 5 * cfg.stress_hedge_lead_rebalances}, "shift hedge one rebalance earlier"),
        ("HEDGE_LAG_1_REBALANCE", {"hedge_shift_days": 5 * cfg.stress_hedge_lag_rebalances}, "shift hedge one rebalance later"),
        ("HEDGE_LAG_2_REBALANCES", {"hedge_shift_days": 5 * cfg.stress_hedge_lag_rebalances_2}, "shift hedge two rebalances later"),
        ("ALLOCATOR_GATE_VOL_CAP_STRESS", {"short_cap_mult": cfg.stress_allocator_cap_mult, "long_multiplier_mult": cfg.stress_allocator_long_mult}, "tighter allocator caps and lighter long budget"),
    ]
    for scenario, override, note in scenarios:
        stitched = _stitch_scenario_ls(wf, cfg, costs, override, scenario)
        rows.append(_stress_metrics_row(cfg.ls_label, scenario, note, stitched, ls_base, long_base, control_base, qqq_obj, spy_obj, cfg))
    rows.append(
        _stress_metrics_row(
            cfg.ls_label,
            "EMPIRICAL_PATH_STRESS",
            "inject worst empirical block into later path",
            _apply_empirical_path_stress(ls_base, cfg),
            ls_base,
            long_base,
            control_base,
            qqq_obj,
            spy_obj,
            cfg,
        )
    )
    return pd.DataFrame(rows)


def build_timing_sensitivity_fast(wf: Dict[str, Any], cfg: Mahoraga15AConfig, stress_df: pd.DataFrame) -> pd.DataFrame:
    scenarios = [
        "BASELINE_15A2",
        "EXECUTION_DELAY_1_REBALANCE",
        "EXECUTION_DELAY_2_REBALANCES",
        "REACTION_SLOWER",
        "REACTION_FASTER",
        "HEDGE_LEAD_1_REBALANCE",
        "HEDGE_LAG_1_REBALANCE",
        "HEDGE_LAG_2_REBALANCES",
    ]
    timing_df = stress_df[(stress_df["Variant"] == cfg.ls_label) & (stress_df["Scenario"].isin(scenarios))].copy()
    if len(timing_df) == 0:
        return timing_df
    baseline = timing_df.loc[timing_df["Scenario"] == "BASELINE_15A2"].iloc[0]
    crisis_windows = build_crisis_windows(wf, cfg)
    ls_base = wf["stitched_ls"]
    rows = []
    for scenario in scenarios:
        row = timing_df.loc[timing_df["Scenario"] == scenario]
        if len(row) == 0:
            continue
        rec = row.iloc[0].to_dict()
        if scenario == "BASELINE_15A2":
            scenario_obj = ls_base
            scenario_type = "BASELINE_TIMING"
            shift_rebalances = 0
        elif scenario.startswith("EXECUTION_DELAY_1"):
            scenario_obj = _stitch_scenario_ls(wf, cfg, wf["costs"], {"delay_days": 5 * cfg.stress_delay_rebalances}, scenario)
            scenario_type = "EXECUTION_DELAY"
            shift_rebalances = cfg.stress_delay_rebalances
        elif scenario.startswith("EXECUTION_DELAY_2"):
            scenario_obj = _stitch_scenario_ls(wf, cfg, wf["costs"], {"delay_days": 5 * cfg.stress_delay_rebalances_2}, scenario)
            scenario_type = "EXECUTION_DELAY"
            shift_rebalances = cfg.stress_delay_rebalances_2
        elif scenario == "REACTION_SLOWER":
            scenario_obj = _stitch_scenario_ls(
                wf,
                cfg,
                wf["costs"],
                {"up_speed_mult": cfg.stress_reaction_slower_mult, "down_speed_mult": cfg.stress_reaction_slower_mult},
                scenario,
            )
            scenario_type = "REACTION_SPEED"
            shift_rebalances = 0
        elif scenario == "REACTION_FASTER":
            scenario_obj = _stitch_scenario_ls(
                wf,
                cfg,
                wf["costs"],
                {"up_speed_mult": cfg.stress_reaction_faster_mult, "down_speed_mult": cfg.stress_reaction_faster_mult},
                scenario,
            )
            scenario_type = "REACTION_SPEED"
            shift_rebalances = 0
        elif scenario == "HEDGE_LEAD_1_REBALANCE":
            scenario_obj = _stitch_scenario_ls(wf, cfg, wf["costs"], {"hedge_shift_days": 5 * cfg.stress_hedge_lead_rebalances}, scenario)
            scenario_type = "HEDGE_LEAD_LAG"
            shift_rebalances = cfg.stress_hedge_lead_rebalances
        elif scenario == "HEDGE_LAG_1_REBALANCE":
            scenario_obj = _stitch_scenario_ls(wf, cfg, wf["costs"], {"hedge_shift_days": 5 * cfg.stress_hedge_lag_rebalances}, scenario)
            scenario_type = "HEDGE_LEAD_LAG"
            shift_rebalances = cfg.stress_hedge_lag_rebalances
        else:
            scenario_obj = _stitch_scenario_ls(wf, cfg, wf["costs"], {"hedge_shift_days": 5 * cfg.stress_hedge_lag_rebalances_2}, scenario)
            scenario_type = "HEDGE_LEAD_LAG"
            shift_rebalances = cfg.stress_hedge_lag_rebalances_2

        crisis_short_values = []
        for _, start, end in crisis_windows:
            crisis_short_values.append(float(_series(scenario_obj, "gross_short", 0.0).loc[start:end].mean()))
        avg_crisis_short = float(np.nanmean(crisis_short_values)) if crisis_short_values else 0.0
        rec["ScenarioType"] = scenario_type
        rec["ShiftRebalances"] = shift_rebalances
        rec["DeltaGrossShort_vs_LSBase"] = round(float(rec["GrossShort"] - baseline["GrossShort"]), 4)
        rec["DeltaBetaQQQ_vs_LSBase"] = round(float(rec["BetaQQQ"] - baseline["BetaQQQ"]), 4)
        rec["DeltaBetaSPY_vs_LSBase"] = round(float(rec["BetaSPY"] - baseline["BetaSPY"]), 4)
        rec["CrisisGrossShortAvg"] = round(avg_crisis_short, 4)
        rows.append(rec)
    return pd.DataFrame(rows)


def build_montecarlo_summary_fast(wf: Dict[str, Any], cfg: Mahoraga15AConfig, costs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ls_obj = wf["stitched_ls"]
    long_obj = wf["frozen_long"]
    baseline = summarize_object(long_obj, cfg, cfg.official_long_label)
    rng = np.random.default_rng(cfg.mc_seed)
    samples: List[Dict[str, Any]] = []

    values = pd.Series(ls_obj["returns"], dtype=float).fillna(0.0).values
    for sample_id in range(cfg.mc_stationary_samples):
        sample = stationary_bootstrap(values, cfg.mc_block_size, rng)
        eq = cfg.capital_initial * (1.0 + pd.Series(sample)).cumprod()
        obj = {"returns": pd.Series(sample), "equity": eq, "exposure": pd.Series(1.0, index=eq.index), "turnover": pd.Series(0.0, index=eq.index)}
        summary = summarize_object(obj, cfg, f"stationary_{sample_id}")
        samples.append({"Method": "stationary_block_bootstrap", "SampleId": sample_id, "CAGR": summary["CAGR"], "Sharpe": summary["Sharpe"], "Sortino": summary["Sortino"], "MaxDD": summary["MaxDD"]})

    gross = pd.Series(ls_obj["gross_returns"], dtype=float).fillna(0.0).values
    tc = pd.Series(ls_obj["transaction_cost"], dtype=float).abs().fillna(0.0).values
    for sample_id in range(cfg.mc_friction_samples):
        mult = float(np.clip(rng.normal(1.0, 0.30), 0.50, 2.25))
        sample = gross - tc * mult
        eq = cfg.capital_initial * (1.0 + pd.Series(sample)).cumprod()
        obj = {"returns": pd.Series(sample), "equity": eq, "exposure": pd.Series(1.0, index=eq.index), "turnover": pd.Series(0.0, index=eq.index)}
        summary = summarize_object(obj, cfg, f"friction_{sample_id}")
        samples.append({"Method": "friction_multiplier_mc", "SampleId": sample_id, "CAGR": summary["CAGR"], "Sharpe": summary["Sharpe"], "Sortino": summary["Sortino"], "MaxDD": summary["MaxDD"]})

    sample_id = 0
    fold_meta = [(p["fold"], p["test_start"], p["test_end"]) for p in wf["ls_fold_payloads"]]
    for short_cap_mult in cfg.mc_short_cap_multipliers:
        for long_mult in cfg.mc_long_multipliers:
            for speed_mult in cfg.mc_speed_multipliers:
                for hedge_mult in cfg.mc_hedge_ratio_multipliers:
                    fold_objs = []
                    for payload in wf["ls_fold_payloads"]:
                        rebuilt = rebuild_ls_fold(
                            payload,
                            cfg,
                            costs,
                            override={
                                "short_cap_mult": short_cap_mult,
                                "long_multiplier_mult": long_mult,
                                "up_speed_mult": speed_mult,
                                "down_speed_mult": speed_mult,
                                "hedge_ratio_mult": hedge_mult,
                            },
                        )
                        fold_objs.append(rebuilt["ls_obj"])
                    stitched = stitch_objects(fold_objs, fold_meta, cfg, f"local_{sample_id}")
                    summary = summarize_object(stitched, cfg, f"local_{sample_id}")
                    samples.append({"Method": "local_param_neighborhood", "SampleId": sample_id, "CAGR": summary["CAGR"], "Sharpe": summary["Sharpe"], "Sortino": summary["Sortino"], "MaxDD": summary["MaxDD"]})
                    sample_id += 1

    samples_df = pd.DataFrame(samples)
    rows = []
    for method, sub in samples_df.groupby("Method", sort=False):
        rows.append(
            {
                "Method": method,
                "Samples": int(len(sub)),
                "MeanCAGR%": round(float(sub["CAGR"].mean()) * 100.0, 2),
                "P5_CAGR%": round(float(sub["CAGR"].quantile(0.05)) * 100.0, 2),
                "P50_CAGR%": round(float(sub["CAGR"].quantile(0.50)) * 100.0, 2),
                "P95_CAGR%": round(float(sub["CAGR"].quantile(0.95)) * 100.0, 2),
                "MeanSharpe": round(float(sub["Sharpe"].mean()), 4),
                "P5_Sharpe": round(float(sub["Sharpe"].quantile(0.05)), 4),
                "P50_Sharpe": round(float(sub["Sharpe"].quantile(0.50)), 4),
                "P95_Sharpe": round(float(sub["Sharpe"].quantile(0.95)), 4),
                "MeanSortino": round(float(sub["Sortino"].mean()), 4),
                "P5_Sortino": round(float(sub["Sortino"].quantile(0.05)), 4),
                "P50_Sortino": round(float(sub["Sortino"].quantile(0.50)), 4),
                "P95_Sortino": round(float(sub["Sortino"].quantile(0.95)), 4),
                "MeanMaxDD%": round(float(sub["MaxDD"].mean()) * 100.0, 2),
                "P5_MaxDD%": round(float(sub["MaxDD"].quantile(0.05)) * 100.0, 2),
                "P50_MaxDD%": round(float(sub["MaxDD"].quantile(0.50)) * 100.0, 2),
                "P95_MaxDD%": round(float(sub["MaxDD"].quantile(0.95)) * 100.0, 2),
                "Prob_Sharpe_lt_Baseline": round(float((sub["Sharpe"] < baseline["Sharpe"]).mean()), 4),
                "Prob_MaxDD_worse_Baseline": round(float((sub["MaxDD"] < baseline["MaxDD"]).mean()), 4),
                "Prob_CAGR_materially_worse_Baseline": round(float((sub["CAGR"] < baseline["CAGR"] - cfg.mc_material_cagr_gap).mean()), 4),
            }
        )
    return pd.DataFrame(rows), samples_df


def build_hedge_effectiveness_fast(
    comparison_df: pd.DataFrame,
    pnl_attr_df: pd.DataFrame,
    crisis_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    cfg: Mahoraga15AConfig,
) -> pd.DataFrame:
    comp = comparison_df.set_index("Variant")
    ls = comp.loc[cfg.ls_label]
    long_only = comp.loc[cfg.official_long_label]
    control = comp.loc[cfg.delevered_label]
    stitched_pnl = pnl_attr_df[(pnl_attr_df["SegmentType"] == "STITCHED") & (pnl_attr_df["Segment"] == "FULL_OOS")].iloc[0]
    crisis_avg_short = float(crisis_df["LS_GrossShort"].mean()) if len(crisis_df) else 0.0
    crisis_delta_control = float(crisis_df["DeltaReturn_vs_Delevered%"].mean()) if len(crisis_df) else 0.0
    crisis_delta_long = float(crisis_df["DeltaReturn_vs_LongOnly%"].mean()) if len(crisis_df) else 0.0
    reaction_sub = stress_df[stress_df["Scenario"].isin(["REACTION_SLOWER", "REACTION_FASTER"])]
    hedge_ratio_sub = stress_df[stress_df["Scenario"].isin(["HEDGE_RATIO_UNDERESTIMATED", "HEDGE_RATIO_OVERESTIMATED"])]
    reaction_move = 0.0
    if len(reaction_sub):
        reaction_move = max(
            float(reaction_sub["DeltaSharpe_vs_LSBase"].abs().max()),
            float(reaction_sub["DeltaCAGR_vs_LSBase%"].abs().max()) / 100.0,
            float((reaction_sub["GrossShort"] - float(ls["GrossShort"])).abs().max()),
        )
    hedge_ratio_move = 0.0
    if len(hedge_ratio_sub):
        hedge_ratio_move = max(
            float(hedge_ratio_sub["DeltaSharpe_vs_LSBase"].abs().max()),
            float(hedge_ratio_sub["DeltaCAGR_vs_LSBase%"].abs().max()) / 100.0,
            float((hedge_ratio_sub["GrossShort"] - float(ls["GrossShort"])).abs().max()),
        )
    baseline_timing = timing_df[timing_df["Scenario"] == "BASELINE_15A2"]
    delay_sub = timing_df[timing_df["Scenario"].isin(["EXECUTION_DELAY_1_REBALANCE", "EXECUTION_DELAY_2_REBALANCES"])]
    lead_lag_sub = timing_df[timing_df["Scenario"].isin(["HEDGE_LEAD_1_REBALANCE", "HEDGE_LAG_1_REBALANCE", "HEDGE_LAG_2_REBALANCES"])]
    delay_best_sharpe = float(delay_sub["DeltaSharpe_vs_LSBase"].max()) if len(delay_sub) else 0.0
    delay_best_cagr = float(delay_sub["DeltaCAGR_vs_LSBase%"].max()) if len(delay_sub) else 0.0
    lead_lag_best_sharpe = float(lead_lag_sub["DeltaSharpe_vs_LSBase"].max()) if len(lead_lag_sub) else 0.0
    lead_lag_best_cagr = float(lead_lag_sub["DeltaCAGR_vs_LSBase%"].max()) if len(lead_lag_sub) else 0.0
    lead_lag_move = 0.0
    if len(lead_lag_sub):
        lead_lag_move = max(
            float(lead_lag_sub["DeltaSharpe_vs_LSBase"].abs().max()),
            float(lead_lag_sub["DeltaCAGR_vs_LSBase%"].abs().max()) / 100.0,
            float(lead_lag_sub["DeltaGrossShort_vs_LSBase"].abs().max()) if "DeltaGrossShort_vs_LSBase" in lead_lag_sub else 0.0,
        )
    ls_vs_control_similar = (
        abs(float(ls["Sharpe"] - control["Sharpe"])) <= cfg.fast_fail_similarity_sharpe_tol
        and abs(float(ls["CAGR%"] - control["CAGR%"])) <= cfg.fast_fail_similarity_cagr_tol_pct
        and abs(float(ls["BetaQQQ"] - control["BetaQQQ"])) <= cfg.fast_fail_similarity_beta_tol
    )
    microscopic_sharpe_with_cagr_drop = (
        float(ls["Sharpe"] - long_only["Sharpe"]) < cfg.success_visible_sharpe_delta
        and float(ls["CAGR%"] - long_only["CAGR%"]) < -cfg.success_cagr_drop_max_pct
    )
    rows = [
        {"Section": "FAST_FAIL", "Metric": "GrossShortStitched", "Value": float(ls["GrossShort"]), "Threshold": cfg.success_gross_short_min, "Passed": bool(float(ls["GrossShort"]) >= cfg.success_gross_short_min), "Detail": "GrossShort stitched must be real, not decorative"},
        {"Section": "FAST_FAIL", "Metric": "GrossShortCrisisAvg", "Value": crisis_avg_short, "Threshold": cfg.success_crisis_gross_short_min, "Passed": bool(crisis_avg_short >= cfg.success_crisis_gross_short_min), "Detail": "crisis windows should show clearly higher short activity"},
        {"Section": "FAST_FAIL", "Metric": "ReactionStressMoves", "Value": reaction_move, "Threshold": cfg.fast_fail_sensitivity_tol, "Passed": bool(reaction_move > cfg.fast_fail_sensitivity_tol), "Detail": "reaction slower/faster must move the system"},
        {"Section": "FAST_FAIL", "Metric": "HedgeRatioStressMoves", "Value": hedge_ratio_move, "Threshold": cfg.fast_fail_sensitivity_tol, "Passed": bool(hedge_ratio_move > cfg.fast_fail_sensitivity_tol), "Detail": "hedge-ratio stress must move the system"},
        {"Section": "FAST_FAIL", "Metric": "TimingDelaySharpeImprovement", "Value": delay_best_sharpe, "Threshold": cfg.fast_fail_timing_sharpe_tol, "Passed": bool(delay_best_sharpe <= cfg.fast_fail_timing_sharpe_tol), "Detail": "delay +1/+2 should not improve Sharpe materially over baseline timing"},
        {"Section": "FAST_FAIL", "Metric": "TimingDelayCAGRImprovementPct", "Value": delay_best_cagr, "Threshold": cfg.fast_fail_timing_cagr_tol_pct, "Passed": bool(delay_best_cagr <= cfg.fast_fail_timing_cagr_tol_pct), "Detail": "delay +1/+2 should not improve CAGR materially over baseline timing"},
        {"Section": "FAST_FAIL", "Metric": "TimingLeadLagSensitivityMoves", "Value": lead_lag_move, "Threshold": cfg.fast_fail_sensitivity_tol, "Passed": bool(lead_lag_move > cfg.fast_fail_sensitivity_tol), "Detail": "hedge lead/lag sensitivity should move if the crisis sleeve is real"},
        {"Section": "FAST_FAIL", "Metric": "TimingLeadLagSharpeImprovement", "Value": lead_lag_best_sharpe, "Threshold": cfg.fast_fail_timing_sharpe_tol, "Passed": bool(lead_lag_best_sharpe <= cfg.fast_fail_timing_sharpe_tol), "Detail": "lead/lag should not dominate baseline timing by accident"},
        {"Section": "FAST_FAIL", "Metric": "TimingLeadLagCAGRImprovementPct", "Value": lead_lag_best_cagr, "Threshold": cfg.fast_fail_timing_cagr_tol_pct, "Passed": bool(lead_lag_best_cagr <= cfg.fast_fail_timing_cagr_tol_pct), "Detail": "lead/lag should not dominate baseline timing on CAGR"},
        {"Section": "FAST_FAIL", "Metric": "LSNotSameAsDeleveredControl", "Value": int(not ls_vs_control_similar), "Threshold": 1.0, "Passed": bool(not ls_vs_control_similar), "Detail": "LS should not be almost identical to delevered control"},
        {"Section": "FAST_FAIL", "Metric": "HedgePnLRelevant", "Value": float(abs(stitched_pnl["PnLSystematicHedgePctInit"])), "Threshold": 0.0025, "Passed": bool(abs(float(stitched_pnl["PnLSystematicHedgePctInit"])) >= 0.0025), "Detail": "systematic hedge PnL should be non-trivial"},
        {"Section": "FAST_FAIL", "Metric": "MicroscopicSharpeWithCAGRDrop", "Value": int(not microscopic_sharpe_with_cagr_drop), "Threshold": 1.0, "Passed": bool(not microscopic_sharpe_with_cagr_drop), "Detail": "microscopic Sharpe gain cannot justify a large CAGR drop"},
        {"Section": "FAST_FAIL", "Metric": "CrisisSeparationVsControl", "Value": crisis_delta_control, "Threshold": 0.0, "Passed": bool(crisis_delta_control > 0.0 or crisis_delta_long > 0.0), "Detail": "crisis windows should separate LS from long-only and/or delevered control"},
        {"Section": "SUCCESS_CHECK", "Metric": "SharpeVisibleDelta_vs_Long", "Value": float(ls["Sharpe"] - long_only["Sharpe"]), "Threshold": cfg.success_visible_sharpe_delta, "Passed": bool(float(ls["Sharpe"] - long_only["Sharpe"]) >= cfg.success_visible_sharpe_delta), "Detail": "Sharpe stitched should improve visibly"},
        {"Section": "SUCCESS_CHECK", "Metric": "SortinoVisibleDelta_vs_Long", "Value": float(ls["Sortino"] - long_only["Sortino"]), "Threshold": cfg.success_visible_sortino_delta, "Passed": bool(float(ls["Sortino"] - long_only["Sortino"]) >= cfg.success_visible_sortino_delta), "Detail": "Sortino stitched should improve visibly"},
        {"Section": "SUCCESS_CHECK", "Metric": "BetaQQQReduction_vs_Long", "Value": float(long_only["BetaQQQ"] - ls["BetaQQQ"]), "Threshold": cfg.success_beta_reduction_min, "Passed": bool(float(long_only["BetaQQQ"] - ls["BetaQQQ"]) >= cfg.success_beta_reduction_min), "Detail": "beta vs QQQ should fall clearly"},
        {"Section": "SUCCESS_CHECK", "Metric": "BetaSPYReduction_vs_Long", "Value": float(long_only["BetaSPY"] - ls["BetaSPY"]), "Threshold": cfg.success_beta_reduction_min, "Passed": bool(float(long_only["BetaSPY"] - ls["BetaSPY"]) >= cfg.success_beta_reduction_min), "Detail": "beta vs SPY should fall clearly"},
        {"Section": "SUCCESS_CHECK", "Metric": "SharpeDelta_vs_Delevered", "Value": float(ls["Sharpe"] - control["Sharpe"]), "Threshold": 0.0, "Passed": bool(float(ls["Sharpe"] - control["Sharpe"]) > 0.0), "Detail": "real hedge should beat equivalent delevered control"},
        {"Section": "SUCCESS_CHECK", "Metric": "SortinoDelta_vs_Delevered", "Value": float(ls["Sortino"] - control["Sortino"]), "Threshold": 0.0, "Passed": bool(float(ls["Sortino"] - control["Sortino"]) > 0.0), "Detail": "real hedge should improve downside behavior vs delevered control"},
        {"Section": "SUCCESS_CHECK", "Metric": "CAGRDelta_vs_Long%", "Value": float(ls["CAGR%"] - long_only["CAGR%"]), "Threshold": -cfg.success_cagr_drop_max_pct, "Passed": bool(float(ls["CAGR%"] - long_only["CAGR%"]) >= -cfg.success_cagr_drop_max_pct), "Detail": "CAGR should not be materially destroyed"},
        {"Section": "SUCCESS_CHECK", "Metric": "AlphaNW_QQQ_Delta_vs_Long", "Value": float(ls["AlphaNW_QQQ"] - long_only["AlphaNW_QQQ"]), "Threshold": -0.02, "Passed": bool(float(ls["AlphaNW_QQQ"] - long_only["AlphaNW_QQQ"]) >= -0.02), "Detail": "benchmark-adjusted alpha vs QQQ should not worsen materially"},
        {"Section": "SUCCESS_CHECK", "Metric": "AlphaNW_SPY_Delta_vs_Long", "Value": float(ls["AlphaNW_SPY"] - long_only["AlphaNW_SPY"]), "Threshold": -0.02, "Passed": bool(float(ls["AlphaNW_SPY"] - long_only["AlphaNW_SPY"]) >= -0.02), "Detail": "benchmark-adjusted alpha vs SPY should not worsen materially"},
    ]
    return pd.DataFrame(rows)


def build_candidate_audit_fast(wf: Dict[str, Any], cfg: Mahoraga15AConfig, pq_df: pd.DataFrame, hedge_effectiveness_df: pd.DataFrame) -> pd.DataFrame:
    rows = wf["freeze_validation_df"].copy()
    rows["Variant"] = cfg.official_long_label
    rows["Reference"] = ""
    rows["Metric"] = rows["Item"]
    rows["Threshold"] = cfg.freeze_rebuild_tol
    rows["PValue"] = np.nan
    rows["QValue"] = np.nan
    rows = rows[["Section", "Variant", "Reference", "Metric", "Value", "Threshold", "Passed", "Detail", "PValue", "QValue"]]

    pair_rows = []
    for rec in pq_df.to_dict("records"):
        pair_rows.append(
            {
                "Section": "PAIRWISE_PQ",
                "Variant": rec["Target"],
                "Reference": rec["Reference"],
                "Metric": "paired_ttest_greater",
                "Value": np.nan,
                "Threshold": np.nan,
                "Passed": bool(rec["p_value"] <= 0.05),
                "Detail": rec["Comparison"],
                "PValue": rec["p_value"],
                "QValue": rec["q_value"],
            }
        )

    hedge_rows = hedge_effectiveness_df.copy()
    hedge_rows["Variant"] = cfg.ls_label
    hedge_rows["Reference"] = cfg.delevered_label
    hedge_rows["PValue"] = np.nan
    hedge_rows["QValue"] = np.nan
    hedge_rows = hedge_rows[["Section", "Variant", "Reference", "Metric", "Value", "Threshold", "Passed", "Detail", "PValue", "QValue"]]

    phase15b = []
    placeholder = build_sparse_idio_short_interface(cfg)
    for signal in placeholder["candidate_signals"]:
        phase15b.append({"Section": "PHASE_15B_INTERFACE", "Variant": placeholder["label"][0], "Reference": "allocator", "Metric": "candidate_signal", "Value": np.nan, "Threshold": np.nan, "Passed": True, "Detail": signal, "PValue": np.nan, "QValue": np.nan})
    for flt in placeholder["mandatory_filters"]:
        phase15b.append({"Section": "PHASE_15B_INTERFACE", "Variant": placeholder["label"][0], "Reference": "risk_filters", "Metric": "mandatory_filter", "Value": np.nan, "Threshold": np.nan, "Passed": True, "Detail": flt, "PValue": np.nan, "QValue": np.nan})
    return pd.concat([rows, pd.DataFrame(pair_rows), hedge_rows, pd.DataFrame(phase15b)], ignore_index=True)


def _figure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#444444",
            "grid.color": "#D6D6D6",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "font.size": 10,
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_figures(wf: Dict[str, Any], cfg: Mahoraga15AConfig, beta_df: pd.DataFrame, exposure_df: pd.DataFrame, stress_df: pd.DataFrame) -> Dict[str, str]:
    figures_dir = Path(cfg.outputs_dir) / "figures"
    ensure_dir(str(figures_dir))
    _figure_style()
    paths: Dict[str, str] = {}
    long_obj = wf["frozen_long"]
    control_obj = wf["stitched_delevered_control"]
    ls_obj = wf["stitched_ls"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for obj, label, color, lw in [
        (long_obj, cfg.official_long_label, "#4C78A8", 2.0),
        (control_obj, cfg.delevered_label, "#F58518", 1.9),
        (ls_obj, cfg.ls_label, "#E45756", 2.2),
    ]:
        ax.plot(obj["equity"].index, obj["equity"].values, label=label, color=color, lw=lw)
    ax.set_title("Equity Curve: Long vs Delevered Control vs LS")
    ax.legend(frameon=False, ncol=3)
    path = figures_dir / "equity_curve_long_control_ls.png"
    _save_figure(fig, path)
    paths["equity_curve_long_control_ls"] = str(path)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(beta_df["Date"], beta_df["observed_beta_qqq_long"], color="#4C78A8", lw=1.6, label="Long beta QQQ")
    ax.plot(beta_df["Date"], beta_df["observed_beta_qqq_delevered"], color="#F58518", lw=1.6, label="Delevered beta QQQ")
    ax.plot(beta_df["Date"], beta_df["observed_beta_qqq_ls"], color="#E45756", lw=1.8, label="LS beta QQQ")
    ax.plot(beta_df["Date"], beta_df["target_beta_qqq"], color="#222222", lw=1.2, ls="--", label="Target beta QQQ")
    ax.set_title("QQQ Beta Through Time")
    ax.legend(frameon=False, ncol=4)
    path = figures_dir / "beta_qqq_through_time.png"
    _save_figure(fig, path)
    paths["beta_qqq_through_time"] = str(path)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(exposure_df["Date"], exposure_df["LongOnlyGrossLong"], color="#4C78A8", lw=1.6, label="Long-only gross long")
    axes[0].plot(exposure_df["Date"], exposure_df["DeleveredGrossLong"], color="#F58518", lw=1.6, label="Delevered gross long")
    axes[0].plot(exposure_df["Date"], exposure_df["LS_GrossLong"], color="#E45756", lw=1.8, label="LS gross long")
    axes[0].plot(exposure_df["Date"], exposure_df["LS_GrossShort"], color="#2A9D8F", lw=1.8, label="LS gross short")
    axes[0].set_title("Gross Exposure and Hedge Sleeve Activity")
    axes[0].legend(frameon=False, ncol=4)
    axes[1].plot(exposure_df["Date"], exposure_df["LS_NetExposure"], color="#E45756", lw=1.8, label="LS net exposure")
    axes[1].plot(exposure_df["Date"], exposure_df["CashBuffer"], color="#666666", lw=1.4, label="Cash buffer")
    axes[1].legend(frameon=False, ncol=2)
    path = figures_dir / "gross_exposure_and_cash.png"
    _save_figure(fig, path)
    paths["gross_exposure_and_cash"] = str(path)

    stress_plot = stress_df[(stress_df["Variant"] == cfg.ls_label) & (~stress_df["Scenario"].isin(["BASELINE_15A2"]))].copy()
    if len(stress_plot):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        axes[0].barh(stress_plot["Scenario"], stress_plot["DeltaSharpe_vs_LSBase"], color="#E45756")
        axes[0].set_title("Stress Delta Sharpe")
        axes[1].barh(stress_plot["Scenario"], stress_plot["GrossShort"], color="#2A9D8F")
        axes[1].set_title("Stress Gross Short")
        path = figures_dir / "stress_response_ls.png"
        _save_figure(fig, path)
        paths["stress_response_ls"] = str(path)
    return paths


def build_fast_report_text(comparison_df: pd.DataFrame, delevered_df: pd.DataFrame, pq_df: pd.DataFrame, pnl_attr_df: pd.DataFrame, crisis_df: pd.DataFrame, timing_df: pd.DataFrame, hedge_effectiveness_df: pd.DataFrame, stress_df: pd.DataFrame, mc_df: pd.DataFrame, audit_df: pd.DataFrame, cfg: Mahoraga15AConfig) -> str:
    failed_fast = audit_df[(audit_df["Section"] == "FAST_FAIL") & (~audit_df["Passed"])]
    success_rows = audit_df[audit_df["Section"] == "SUCCESS_CHECK"]
    if len(failed_fast):
        thesis_status = "THESIS FAILED FAST"
    elif len(success_rows) and bool(success_rows["Passed"].all()):
        thesis_status = "PROMISING"
    else:
        thesis_status = "INCONCLUSIVE"

    lines = [
        "# Mahoraga15A2 FAST",
        "",
        f"## Thesis status: {thesis_status}",
        "",
        "## 1. Stitched comparison",
        comparison_df.to_string(index=False),
        "",
        "## 2. Delevered control check",
        delevered_df.to_string(index=False),
        "",
        "## 3. Pairwise p-values / q-values",
        pq_df.to_string(index=False),
        "",
        "## 4. PnL attribution",
        pnl_attr_df.to_string(index=False),
        "",
        "## 5. Crisis / stress window scorecard",
        crisis_df.to_string(index=False),
        "",
        "## 6. Hedge effectiveness / fail-fast",
        hedge_effectiveness_df.to_string(index=False),
        "",
        "## 7. Timing sensitivity",
        timing_df.to_string(index=False),
        "",
        "## 8. Stress suite",
        stress_df.to_string(index=False),
        "",
        "## 9. Monte Carlo / bootstrap",
        mc_df.to_string(index=False),
    ]
    if len(failed_fast):
        lines.extend(["", "## 10. Explicit failure flags", failed_fast[["Metric", "Value", "Threshold", "Detail"]].to_string(index=False)])
    return "\n".join(lines)


def save_fast_outputs(wf: Dict[str, Any], cfg: Mahoraga15AConfig, costs) -> Dict[str, pd.DataFrame]:
    ensure_dir(cfg.outputs_dir)
    comparison_df = build_stitched_comparison_fast(wf, cfg)
    delevered_df = build_delevered_control_fast(comparison_df, cfg)
    pq_df = build_pairwise_pq_fast(wf, cfg)
    beta_df = build_beta_decomposition_fast(wf, cfg)
    exposure_df = build_exposure_trace_fast(wf, cfg)
    allocator_df = build_allocator_response_fast(wf, cfg)
    pnl_attr_df = build_pnl_attribution_fast(wf, cfg)
    crisis_df = build_crisis_window_scorecard_fast(wf, cfg, pnl_attr_df)
    short_activity_df = build_short_activity_summary_fast(wf, cfg, pnl_attr_df)
    stress_df = build_stress_suite_fast(wf, cfg, costs)
    timing_df = build_timing_sensitivity_fast(wf, cfg, stress_df)
    mc_summary_df, mc_samples_df = build_montecarlo_summary_fast(wf, cfg, costs)
    hedge_effectiveness_df = build_hedge_effectiveness_fast(comparison_df, pnl_attr_df, crisis_df, stress_df, timing_df, cfg)
    audit_df = build_candidate_audit_fast(wf, cfg, pq_df, hedge_effectiveness_df)
    figure_paths = generate_figures(wf, cfg, beta_df, exposure_df, stress_df)
    report_text = build_fast_report_text(comparison_df, delevered_df, pq_df, pnl_attr_df, crisis_df, timing_df, hedge_effectiveness_df, stress_df, mc_summary_df, audit_df, cfg)

    comparison_df.to_csv(Path(cfg.outputs_dir) / "ls_stitched_comparison_fast.csv", index=False)
    delevered_df.to_csv(Path(cfg.outputs_dir) / "ls_delevered_control_fast.csv", index=False)
    pq_df.to_csv(Path(cfg.outputs_dir) / "ls_pairwise_pq_fast.csv", index=False)
    beta_df.to_csv(Path(cfg.outputs_dir) / "ls_beta_decomposition_fast.csv", index=False)
    exposure_df.to_csv(Path(cfg.outputs_dir) / "ls_gross_net_exposure_fast.csv", index=False)
    allocator_df.to_csv(Path(cfg.outputs_dir) / "ls_allocator_response_fast.csv", index=False)
    pnl_attr_df.to_csv(Path(cfg.outputs_dir) / "ls_pnl_attribution_fast.csv", index=False)
    crisis_df.to_csv(Path(cfg.outputs_dir) / "ls_crisis_window_scorecard_fast.csv", index=False)
    short_activity_df.to_csv(Path(cfg.outputs_dir) / "ls_short_activity_summary_fast.csv", index=False)
    hedge_effectiveness_df.to_csv(Path(cfg.outputs_dir) / "ls_hedge_effectiveness_fast.csv", index=False)
    timing_df.to_csv(Path(cfg.outputs_dir) / "ls_timing_sensitivity_fast.csv", index=False)
    stress_df.to_csv(Path(cfg.outputs_dir) / "ls_stress_suite_fast.csv", index=False)
    mc_summary_df.to_csv(Path(cfg.outputs_dir) / "ls_montecarlo_summary_fast.csv", index=False)
    audit_df.to_csv(Path(cfg.outputs_dir) / "ls_candidate_audit_fast.csv", index=False)
    with open(Path(cfg.outputs_dir) / cfg.report_name, "w", encoding="utf-8") as f:
        f.write(report_text)
    return {
        "comparison": comparison_df,
        "delevered_control": delevered_df,
        "pairwise_pq": pq_df,
        "beta_trace": beta_df,
        "exposure_trace": exposure_df,
        "allocator_trace": allocator_df,
        "pnl_attribution": pnl_attr_df,
        "crisis_window_scorecard": crisis_df,
        "short_activity_summary": short_activity_df,
        "hedge_effectiveness": hedge_effectiveness_df,
        "timing_sensitivity": timing_df,
        "stress_suite": stress_df,
        "montecarlo_summary": mc_summary_df,
        "candidate_audit": audit_df,
        "figures": pd.DataFrame({"Figure": list(figure_paths.keys()), "Path": list(figure_paths.values())}),
        "montecarlo_samples": mc_samples_df,
    }
