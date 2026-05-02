from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mahoraga6_1 as m6
from mahoraga14_config import Mahoraga14Config
from mahoraga14_utils import bhy_qvalues, ensure_dir, paired_ttest_pvalue


def _label_map(cfg: Mahoraga14Config) -> Dict[str, str]:
    return {
        cfg.historical_benchmark_label: "LEGACY",
        "QQQ": "QQQ",
        "SPY": "SPY",
        cfg.official_baseline_label: "BASE_ALPHA_V2",
        cfg.main_variant_key: "STRUCTURAL_DEFENSE_ONLY_AUX",
        cfg.control_variant_key: "MAHORAGA14_1_LONG_ONLY_CONTROL",
        cfg.primary_variant_key: "MAHORAGA14_2_LONG_PARTICIPATION",
    }


def _stitched_map(wf: Dict[str, Any], cfg: Mahoraga14Config) -> Dict[str, Dict[str, Any]]:
    return {
        cfg.historical_benchmark_label: wf["stitched_legacy"],
        "QQQ": wf["stitched_benchmarks"]["QQQ"],
        "SPY": wf["stitched_benchmarks"]["SPY"],
        **wf["stitched_variants"],
    }


def _capture_ratio(strategy_r: pd.Series, bench_r: pd.Series, upside: bool) -> float:
    s = pd.Series(strategy_r, dtype=float).dropna()
    b = pd.Series(bench_r, dtype=float).reindex(s.index).fillna(0.0)
    mask = b > 0.0 if upside else b < 0.0
    denom = float(b.loc[mask].sum())
    if not mask.any() or abs(denom) < 1e-12:
        return 0.0
    return float(s.loc[mask].sum() / denom)


def _beta(strategy_r: pd.Series, bench_r: pd.Series) -> float:
    s = pd.Series(strategy_r, dtype=float).dropna()
    b = pd.Series(bench_r, dtype=float).reindex(s.index).fillna(0.0)
    if len(s) < 2 or float(b.var()) == 0.0:
        return 0.0
    return float(s.cov(b) / b.var())


def _return_per_exposure(r: pd.Series, exposure: pd.Series) -> float:
    exp = pd.Series(exposure, dtype=float).reindex(pd.Index(r.index)).fillna(0.0)
    denom = float(exp.mean())
    if denom <= 1e-12:
        return 0.0
    return float(pd.Series(r, dtype=float).mean() / denom)


def _alpha_nw(strategy_r: pd.Series, bench_r: pd.Series, cfg: Mahoraga14Config, label: str) -> Dict[str, float]:
    res = m6.alpha_test_nw(strategy_r, bench_r, cfg, label=label)
    if "error" in res:
        return {"alpha_ann": np.nan, "t_alpha": np.nan, "p_alpha": np.nan, "beta": np.nan, "R2": np.nan}
    return {
        "alpha_ann": float(res.get("alpha_ann", np.nan)),
        "t_alpha": float(res.get("t_alpha", np.nan)),
        "p_alpha": float(res.get("p_alpha", np.nan)),
        "beta": float(res.get("beta", np.nan)),
        "R2": float(res.get("R2", np.nan)),
    }


def _metrics_row(label: str, obj: Dict[str, Any], qqq_obj: Dict[str, Any], spy_obj: Dict[str, Any], cfg: Mahoraga14Config) -> Dict[str, Any]:
    summary = m6.summarize(obj["returns"], obj["equity"], obj["exposure"], obj["turnover"], cfg, label)
    alpha_qqq = _alpha_nw(obj["returns"], qqq_obj["returns"], cfg, f"{label}_QQQ")
    alpha_spy = _alpha_nw(obj["returns"], spy_obj["returns"], cfg, f"{label}_SPY")
    return {
        "Variant": label,
        "CAGR": float(summary["CAGR"]),
        "Sharpe": float(summary["Sharpe"]),
        "Sortino": float(summary["Sortino"]),
        "MaxDD": float(summary["MaxDD"]),
        "AvgExposure": float(pd.Series(obj["exposure"], dtype=float).mean()),
        "AvgTurnover": float(pd.Series(obj["turnover"], dtype=float).mean()),
        "ReturnPerExposure": _return_per_exposure(obj["returns"], obj["exposure"]),
        "BetaQQQ": _beta(obj["returns"], qqq_obj["returns"]),
        "BetaSPY": _beta(obj["returns"], spy_obj["returns"]),
        "UpsideCaptureQQQ": _capture_ratio(obj["returns"], qqq_obj["returns"], upside=True),
        "DownsideCaptureQQQ": _capture_ratio(obj["returns"], qqq_obj["returns"], upside=False),
        "UpsideCaptureSPY": _capture_ratio(obj["returns"], spy_obj["returns"], upside=True),
        "DownsideCaptureSPY": _capture_ratio(obj["returns"], spy_obj["returns"], upside=False),
        "AlphaNW_QQQ": float(alpha_qqq["alpha_ann"]),
        "AlphaNW_SPY": float(alpha_spy["alpha_ann"]),
        "AlphaNW_QQQ_t": float(alpha_qqq["t_alpha"]),
        "AlphaNW_SPY_t": float(alpha_spy["t_alpha"]),
        "AlphaNW_QQQ_p": float(alpha_qqq["p_alpha"]),
        "AlphaNW_SPY_p": float(alpha_spy["p_alpha"]),
    }


def _build_comparison_df(wf: Dict[str, Any], cfg: Mahoraga14Config) -> pd.DataFrame:
    labels = _label_map(cfg)
    stitched = _stitched_map(wf, cfg)
    qqq_obj = stitched["QQQ"]
    spy_obj = stitched["SPY"]
    order = ["QQQ", "SPY", cfg.official_baseline_label, cfg.control_variant_key, cfg.primary_variant_key, cfg.historical_benchmark_label]
    rows = [_metrics_row(labels[key], stitched[key], qqq_obj, spy_obj, cfg) for key in order]
    df = pd.DataFrame(rows)

    pairs = [
        (cfg.primary_variant_key, "QQQ", "Primary_vs_QQQ"),
        (cfg.primary_variant_key, "SPY", "Primary_vs_SPY"),
        (cfg.primary_variant_key, cfg.control_variant_key, "Primary_vs_Control"),
        (cfg.control_variant_key, "QQQ", "Control_vs_QQQ"),
        (cfg.control_variant_key, "SPY", "Control_vs_SPY"),
    ]
    pq_rows = []
    for target, reference, name in pairs:
        pq_rows.append(
            {
                "Comparison": name,
                "p_value": paired_ttest_pvalue(stitched[target]["returns"] - stitched[reference]["returns"], alternative="greater"),
            }
        )
    pq = pd.DataFrame(pq_rows)
    pq["q_value"] = bhy_qvalues(pq["p_value"].values, alpha=cfg.bhy_alpha) if len(pq) else []
    pq_map = pq.set_index("Comparison").to_dict("index")

    df["p_value"] = np.nan
    df["q_value"] = np.nan
    primary_label = labels[cfg.primary_variant_key]
    control_label = labels[cfg.control_variant_key]
    for idx, row in df.iterrows():
        if row["Variant"] == primary_label:
            df.loc[idx, "p_value"] = pq_map["Primary_vs_QQQ"]["p_value"]
            df.loc[idx, "q_value"] = pq_map["Primary_vs_QQQ"]["q_value"]
            df.loc[idx, "p_value_vs_SPY"] = pq_map["Primary_vs_SPY"]["p_value"]
            df.loc[idx, "q_value_vs_SPY"] = pq_map["Primary_vs_SPY"]["q_value"]
            df.loc[idx, "p_value_vs_Control"] = pq_map["Primary_vs_Control"]["p_value"]
            df.loc[idx, "q_value_vs_Control"] = pq_map["Primary_vs_Control"]["q_value"]
        elif row["Variant"] == control_label:
            df.loc[idx, "p_value"] = pq_map["Control_vs_QQQ"]["p_value"]
            df.loc[idx, "q_value"] = pq_map["Control_vs_QQQ"]["q_value"]
            df.loc[idx, "p_value_vs_SPY"] = pq_map["Control_vs_SPY"]["p_value"]
            df.loc[idx, "q_value_vs_SPY"] = pq_map["Control_vs_SPY"]["q_value"]

    for col in ["CAGR", "MaxDD"]:
        df[col] = df[col] * 100.0
    return df.round(6)


def _window_slice(obj: Dict[str, Any], start: pd.Timestamp, end: pd.Timestamp, cfg: Mahoraga14Config) -> Dict[str, Any]:
    r = pd.Series(obj["returns"], dtype=float).loc[start:end]
    return {
        "returns": r,
        "equity": cfg.capital_initial * (1.0 + r).cumprod(),
        "exposure": pd.Series(obj["exposure"], dtype=float).reindex(r.index).fillna(0.0),
        "turnover": pd.Series(obj["turnover"], dtype=float).reindex(r.index).fillna(0.0),
    }


def _window_summary(strategy_obj: Dict[str, Any], bench_obj: Dict[str, Any], cfg: Mahoraga14Config, label: str) -> Dict[str, float]:
    summary = m6.summarize(strategy_obj["returns"], strategy_obj["equity"], strategy_obj["exposure"], strategy_obj["turnover"], cfg, label)
    return {
        "Return": float(summary["TotalReturn"]),
        "Sharpe": float(summary["Sharpe"]),
        "Sortino": float(summary["Sortino"]),
        "MaxDD": float(summary["MaxDD"]),
        "Beta": _beta(strategy_obj["returns"], bench_obj["returns"]),
        "UpsideCapture": _capture_ratio(strategy_obj["returns"], bench_obj["returns"], upside=True),
        "Exposure": float(pd.Series(strategy_obj["exposure"], dtype=float).mean()),
    }


def _detect_auto_bull_windows(qqq_returns: pd.Series, cfg: Mahoraga14Config) -> List[Tuple[str, pd.Timestamp, pd.Timestamp, str]]:
    q = pd.Series(qqq_returns, dtype=float).dropna()
    if len(q) < cfg.bull_window_min_days:
        return []
    eq = (1.0 + q).cumprod()
    dd = eq / eq.cummax() - 1.0
    ret_63 = eq.pct_change(cfg.bull_window_min_days).fillna(0.0)
    eff_21 = (q.rolling(21).sum().abs() / q.abs().rolling(21).sum().replace(0.0, np.nan)).fillna(0.0)
    mask = (
        (ret_63 >= cfg.bull_window_min_return)
        & (dd >= cfg.bull_window_max_drawdown)
        & (eff_21 >= cfg.bull_window_min_efficiency)
    )
    groups = (mask.ne(mask.shift(1))).cumsum()
    windows: List[Tuple[str, pd.Timestamp, pd.Timestamp, str]] = []
    for _, sub in mask.groupby(groups):
        if not bool(sub.iloc[0]):
            continue
        start = pd.Timestamp(sub.index[0])
        end = pd.Timestamp(sub.index[-1])
        if len(sub.index) >= cfg.bull_window_min_days:
            windows.append((f"AUTO_{start.date()}_{end.date()}", start, end, "AUTO_OOS"))
    return windows


def _window_specs(index: pd.DatetimeIndex, qqq_returns: pd.Series, cfg: Mahoraga14Config) -> List[Tuple[str, pd.Timestamp, pd.Timestamp, str]]:
    manual = [
        ("2017_2018", pd.Timestamp("2017-01-01"), pd.Timestamp("2018-12-31"), "MANUAL"),
        ("2020_2021", pd.Timestamp("2020-04-01"), pd.Timestamp("2021-12-31"), "MANUAL"),
        ("2023_2024", pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31"), "MANUAL"),
    ]
    specs = manual + _detect_auto_bull_windows(qqq_returns, cfg)
    rows = []
    for name, start, end, source in specs:
        s = max(start, pd.Timestamp(index.min()))
        e = min(end, pd.Timestamp(index.max()))
        if s <= e and len(index[(index >= s) & (index <= e)]) >= 20:
            rows.append((name, s, e, source))
    dedup: List[Tuple[str, pd.Timestamp, pd.Timestamp, str]] = []
    seen = set()
    for item in rows:
        key = (item[1], item[2], item[3])
        if key not in seen:
            seen.add(key)
            dedup.append(item)
    return dedup


def _build_bull_window_scorecard(wf: Dict[str, Any], cfg: Mahoraga14Config) -> pd.DataFrame:
    stitched = _stitched_map(wf, cfg)
    primary = stitched[cfg.primary_variant_key]
    control = stitched[cfg.control_variant_key]
    qqq = stitched["QQQ"]
    spy = stitched["SPY"]
    specs = _window_specs(pd.DatetimeIndex(primary["returns"].index), qqq["returns"], cfg)
    rows = []
    for name, start, end, source in specs:
        primary_obj = _window_slice(primary, start, end, cfg)
        control_obj = _window_slice(control, start, end, cfg)
        qqq_obj = _window_slice(qqq, start, end, cfg)
        spy_obj = _window_slice(spy, start, end, cfg)
        p_qqq = _window_summary(primary_obj, qqq_obj, cfg, name)
        p_spy = _window_summary(primary_obj, spy_obj, cfg, name)
        c_qqq = _window_summary(control_obj, qqq_obj, cfg, name)
        qqq_return = float(np.prod(1.0 + qqq_obj["returns"].values) - 1.0)
        spy_return = float(np.prod(1.0 + spy_obj["returns"].values) - 1.0)
        rows.append(
            {
                "Window": name,
                "Source": source,
                "Start": start,
                "End": end,
                "Mahoraga14_2Return": p_qqq["Return"],
                "Mahoraga14_1ControlReturn": c_qqq["Return"],
                "QQQReturn": qqq_return,
                "SPYReturn": spy_return,
                "DeltaReturn_vs_QQQ": p_qqq["Return"] - qqq_return,
                "DeltaReturn_vs_SPY": p_spy["Return"] - spy_return,
                "DeltaReturn_vs_Control": p_qqq["Return"] - c_qqq["Return"],
                "SharpeLocal": p_qqq["Sharpe"],
                "SortinoLocal": p_qqq["Sortino"],
                "MaxDDLocal": p_qqq["MaxDD"],
                "BetaQQQLocal": p_qqq["Beta"],
                "BetaSPYLocal": p_spy["Beta"],
                "UpsideCaptureQQQLocal": p_qqq["UpsideCapture"],
                "UpsideCaptureSPYLocal": p_spy["UpsideCapture"],
                "ExposureLocal": p_qqq["Exposure"],
                "ControlExposureLocal": c_qqq["Exposure"],
            }
        )
    return pd.DataFrame(rows).round(6)


def _stitch_variant_frame(results: List[Dict[str, Any]], variant_key: str, field: str) -> pd.DataFrame:
    frames = []
    for result in results:
        start = pd.Timestamp(result["test_start"])
        end = pd.Timestamp(result["test_end"])
        if variant_key == "BASE":
            frame = result["base_bt"][field].loc[start:end].copy()
        else:
            frame = result["variant_runs"][variant_key][field].loc[start:end].copy()
        frames.append(frame)
    out = pd.concat(frames).sort_index() if frames else pd.DataFrame()
    if len(out) and out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="first")]
    return out


def _build_active_return_vs_qqq(wf: Dict[str, Any], cfg: Mahoraga14Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    primary = wf["stitched_variants"][cfg.primary_variant_key]
    control = wf["stitched_variants"][cfg.control_variant_key]
    qqq = wf["stitched_benchmarks"]["QQQ"]
    spy = wf["stitched_benchmarks"]["SPY"]
    allocator_daily = _stitch_variant_frame(wf["results"], cfg.primary_variant_key, "allocator_daily")
    df = pd.DataFrame(index=primary["returns"].index)
    df["PrimaryReturn"] = pd.Series(primary["returns"], dtype=float)
    df["ControlReturn"] = pd.Series(control["returns"], dtype=float).reindex(df.index).fillna(0.0)
    df["QQQReturn"] = pd.Series(qqq["returns"], dtype=float).reindex(df.index).fillna(0.0)
    df["SPYReturn"] = pd.Series(spy["returns"], dtype=float).reindex(df.index).fillna(0.0)
    df["ActiveReturn_vs_QQQ"] = df["PrimaryReturn"] - df["QQQReturn"]
    df["ActiveReturn_vs_SPY"] = df["PrimaryReturn"] - df["SPYReturn"]
    df["CumActiveReturn_vs_QQQ"] = (1.0 + df["ActiveReturn_vs_QQQ"]).cumprod() - 1.0
    df["CumPrimary"] = (1.0 + df["PrimaryReturn"]).cumprod() - 1.0
    df["CumQQQ"] = (1.0 + df["QQQReturn"]).cumprod() - 1.0
    df["CumSPY"] = (1.0 + df["SPYReturn"]).cumprod() - 1.0
    df["CumControl"] = (1.0 + df["ControlReturn"]).cumprod() - 1.0
    if len(allocator_daily):
        df["PrimaryLongBudget"] = allocator_daily["long_budget"].reindex(df.index).ffill().fillna(cfg.participation_long_budget_base)
        df["PrimaryLeaderBlend"] = allocator_daily["leader_blend"].reindex(df.index).ffill().fillna(0.0)
        df["PrimaryRiskBackoff"] = allocator_daily["risk_backoff_score"].reindex(df.index).ffill().fillna(0.0)
        df["ParticipationState"] = allocator_daily["participation_state"].reindex(df.index).ffill().fillna("NEUTRAL")
    curve = df[["CumActiveReturn_vs_QQQ", "CumPrimary", "CumQQQ", "CumSPY", "CumControl"]].copy()
    return df.reset_index(names="Date").round(8), curve.reset_index(names="Date").round(8)


def _build_allocator_cash_drag(wf: Dict[str, Any], cfg: Mahoraga14Config) -> pd.DataFrame:
    allocator_daily = _stitch_variant_frame(wf["results"], cfg.primary_variant_key, "allocator_daily")
    bull_diag = _stitch_variant_frame(wf["results"], cfg.primary_variant_key, "bull_diagnostics")
    bull_cols = [c for c in bull_diag.columns if c not in allocator_daily.columns]
    out = allocator_daily.join(bull_diag[bull_cols], how="left")
    out.index.name = "Date"
    return out.reset_index().round(8)


def _estimate_window_decomp(
    result: Dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: Mahoraga14Config,
) -> Dict[str, float]:
    idx = result["stress_pre"]["rets"].loc[start:end].index
    if len(idx) == 0:
        return {}
    rets = result["stress_pre"]["rets"].loc[idx]
    base_alpha_w = result["base_weights_exec_1x"].loc[idx]
    control_w = result["variant_runs"][cfg.control_variant_key]["weights_exec_1x"].loc[idx]
    primary_run = result["variant_runs"][cfg.primary_variant_key]
    base_def_w = primary_run["base_defense_weights_exec_1x"].loc[idx]
    part_w = primary_run["participation_weights_exec_1x"].loc[idx]
    final_w = primary_run["weights_exec_1x"].loc[idx]
    alloc = primary_run["allocator_daily"].loc[idx]
    bull = primary_run["bull_diagnostics"].loc[idx]
    qqq_r = result["base_bt"]["bench"]["QQQ_r"].loc[idx]

    base_alpha_ret = (base_alpha_w * rets).sum(axis=1)
    control_ret = (control_w * rets).sum(axis=1)
    base_def_ret = (base_def_w * rets).sum(axis=1)
    part_ret = (part_w * rets).sum(axis=1)
    final_ret = (final_w * rets).sum(axis=1)

    cash_drag_est = (bull["cash_redeployed"].reindex(idx).fillna(0.0) * part_ret).sum()
    beta_relief_est = (alloc["leader_blend"].reindex(idx).fillna(0.0) * (part_ret - base_alpha_ret)).sum()
    defense_drag_est = (alloc["leader_blend"].reindex(idx).fillna(0.0) * (base_alpha_ret - base_def_ret)).sum()
    restrictive_intensity = 1.0 - (
        alloc[["gate_scale_adjustment", "vol_mult_adjustment", "exp_cap_adjustment"]]
        .clip(upper=1.0)
        .mean(axis=1)
        .reindex(idx)
        .fillna(1.0)
    )
    allocator_drag_est = (-(restrictive_intensity * base_def_ret)).sum()
    underexposure_drag_est = (-((1.0 - bull["gross_after_budget"].reindex(idx).fillna(0.0)).clip(lower=0.0) * qqq_r)).sum()

    return {
        "WindowReturnPrimary1x": float(np.prod(1.0 + final_ret.values) - 1.0),
        "WindowReturnControl1x": float(np.prod(1.0 + control_ret.values) - 1.0),
        "WindowReturnQQQ": float(np.prod(1.0 + qqq_r.values) - 1.0),
        "EstimatedCashDragContribution": float(cash_drag_est),
        "EstimatedBetaResidualReliefContribution": float(beta_relief_est),
        "EstimatedDefenseDragContribution": float(defense_drag_est),
        "EstimatedAllocatorRestrictionContribution": float(allocator_drag_est),
        "EstimatedUnderExposureContribution": float(underexposure_drag_est),
        "AvgLongBudget": float(alloc["long_budget"].mean()),
        "AvgLeaderBlend": float(alloc["leader_blend"].mean()),
        "AvgGateAdj": float(alloc["gate_scale_adjustment"].mean()),
        "AvgVolAdj": float(alloc["vol_mult_adjustment"].mean()),
        "AvgExpCapAdj": float(alloc["exp_cap_adjustment"].mean()),
        "AvgCashDragBefore": float(bull["cash_drag_before"].mean()),
        "AvgCashDragAfter": float(bull["cash_drag_after"].mean()),
    }


def _build_upside_participation_decomposition(wf: Dict[str, Any], cfg: Mahoraga14Config, scorecard_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in scorecard_df.iterrows():
        start = pd.Timestamp(row["Start"])
        end = pd.Timestamp(row["End"])
        agg: Dict[str, float] = {}
        count = 0
        for result in wf["results"]:
            res_start = pd.Timestamp(result["test_start"])
            res_end = pd.Timestamp(result["test_end"])
            s = max(start, res_start)
            e = min(end, res_end)
            if s > e:
                continue
            est = _estimate_window_decomp(result, s, e, cfg)
            if not est:
                continue
            for key, value in est.items():
                agg[key] = agg.get(key, 0.0) + float(value)
            count += 1
        if count == 0:
            continue
        rows.append({"Window": row["Window"], "Start": start, "End": end, **agg})
    return pd.DataFrame(rows).round(8)


def _build_leader_miss_analysis(wf: Dict[str, Any], cfg: Mahoraga14Config, scorecard_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, window in scorecard_df.iterrows():
        start = pd.Timestamp(window["Start"])
        end = pd.Timestamp(window["End"])
        per_window_returns = []
        per_window_base = []
        per_window_control = []
        per_window_primary = []
        for result in wf["results"]:
            res_start = pd.Timestamp(result["test_start"])
            res_end = pd.Timestamp(result["test_end"])
            s = max(start, res_start)
            e = min(end, res_end)
            if s > e:
                continue
            idx = result["stress_pre"]["rets"].loc[s:e].index
            if len(idx) == 0:
                continue
            rets = result["stress_pre"]["rets"].loc[idx]
            total_ret = (1.0 + rets).prod() - 1.0
            total_ret.name = "WindowReturn"
            per_window_returns.append(total_ret)
            per_window_base.append(result["base_weights_exec_1x"].loc[idx].mean())
            per_window_control.append(result["variant_runs"][cfg.control_variant_key]["weights_exec_1x"].loc[idx].mean())
            per_window_primary.append(result["variant_runs"][cfg.primary_variant_key]["weights_exec_1x"].loc[idx].mean())
        if not per_window_returns:
            continue
        base_panel = pd.concat(per_window_base, axis=1)
        control_panel = pd.concat(per_window_control, axis=1)
        primary_panel = pd.concat(per_window_primary, axis=1)
        total_ret = pd.concat(per_window_returns, axis=1).mean(axis=1).sort_values(ascending=False)
        base_w = base_panel.mean(axis=1)
        control_w = control_panel.mean(axis=1)
        primary_w = primary_panel.mean(axis=1)
        top = total_ret.head(12)
        for rank, (ticker, ret) in enumerate(top.items(), start=1):
            rows.append(
                {
                    "Window": window["Window"],
                    "Ticker": ticker,
                    "LeaderRank": rank,
                    "WindowReturn": float(ret),
                    "AvgWeight_BASE_ALPHA_V2": float(base_w.get(ticker, 0.0)),
                    "AvgWeight_Control14_1": float(control_w.get(ticker, 0.0)),
                    "AvgWeight_Mahoraga14_2": float(primary_w.get(ticker, 0.0)),
                    "SelectionRate_BASE_ALPHA_V2": float((base_panel.loc[ticker] > 0.0).mean()) if ticker in base_panel.index else 0.0,
                    "SelectionRate_Control14_1": float((control_panel.loc[ticker] > 0.0).mean()) if ticker in control_panel.index else 0.0,
                    "SelectionRate_Mahoraga14_2": float((primary_panel.loc[ticker] > 0.0).mean()) if ticker in primary_panel.index else 0.0,
                    "PrimaryWeightLift_vs_Control": float(primary_w.get(ticker, 0.0) - control_w.get(ticker, 0.0)),
                    "MissedByControl": int(control_w.get(ticker, 0.0) <= 1e-6),
                    "CapturedBy14_2": int(primary_w.get(ticker, 0.0) > 1e-6),
                }
            )
    return pd.DataFrame(rows).round(8)


def _curve_plot(curve_df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(11, 6))
    plt.plot(curve_df["Date"], curve_df["CumActiveReturn_vs_QQQ"], label="CumActive_vs_QQQ", linewidth=1.8)
    plt.plot(curve_df["Date"], curve_df["CumPrimary"], label="Mahoraga14_2", linewidth=1.2)
    plt.plot(curve_df["Date"], curve_df["CumQQQ"], label="QQQ", linewidth=1.2)
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    plt.legend()
    plt.title("Mahoraga14_2 Active Return vs QQQ")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _object_from_returns(
    returns: pd.Series,
    exposure: pd.Series,
    turnover: pd.Series,
    transaction_cost: pd.Series,
    cfg: Mahoraga14Config,
) -> Dict[str, Any]:
    r = pd.Series(returns, dtype=float).fillna(0.0)
    tc = pd.Series(transaction_cost, dtype=float).reindex(r.index).fillna(0.0)
    return {
        "returns": r,
        "gross_returns": r + tc,
        "transaction_cost": tc,
        "exposure": pd.Series(exposure, dtype=float).reindex(r.index).fillna(0.0),
        "turnover": pd.Series(turnover, dtype=float).reindex(r.index).fillna(0.0),
        "equity": cfg.capital_initial * (1.0 + r).cumprod(),
    }


def _stress_row(
    scenario: str,
    note: str,
    base_obj: Dict[str, Any],
    stressed_obj: Dict[str, Any],
    qqq_obj: Dict[str, Any],
    spy_obj: Dict[str, Any],
    cfg: Mahoraga14Config,
) -> Dict[str, Any]:
    base_summary = m6.summarize(base_obj["returns"], base_obj["equity"], base_obj["exposure"], base_obj["turnover"], cfg, scenario)
    stress_summary = m6.summarize(stressed_obj["returns"], stressed_obj["equity"], stressed_obj["exposure"], stressed_obj["turnover"], cfg, scenario)
    base_alpha_qqq = _alpha_nw(base_obj["returns"], qqq_obj["returns"], cfg, f"{scenario}_BASE_QQQ")
    stress_alpha_qqq = _alpha_nw(stressed_obj["returns"], qqq_obj["returns"], cfg, f"{scenario}_QQQ")
    base_alpha_spy = _alpha_nw(base_obj["returns"], spy_obj["returns"], cfg, f"{scenario}_BASE_SPY")
    stress_alpha_spy = _alpha_nw(stressed_obj["returns"], spy_obj["returns"], cfg, f"{scenario}_SPY")
    return {
        "Variant": "MAHORAGA14_2_LONG_PARTICIPATION",
        "Scenario": scenario,
        "ScenarioNote": note,
        "BaseCAGR%": round(base_summary["CAGR"] * 100.0, 4),
        "StressCAGR%": round(stress_summary["CAGR"] * 100.0, 4),
        "DeltaCAGR%": round((stress_summary["CAGR"] - base_summary["CAGR"]) * 100.0, 4),
        "BaseSharpe": round(base_summary["Sharpe"], 6),
        "StressSharpe": round(stress_summary["Sharpe"], 6),
        "DeltaSharpe": round(stress_summary["Sharpe"] - base_summary["Sharpe"], 6),
        "BaseMaxDD%": round(base_summary["MaxDD"] * 100.0, 4),
        "StressMaxDD%": round(stress_summary["MaxDD"] * 100.0, 4),
        "DeltaMaxDD%": round((stress_summary["MaxDD"] - base_summary["MaxDD"]) * 100.0, 4),
        "BaseAlphaNW_QQQ": round(base_alpha_qqq["alpha_ann"], 8) if np.isfinite(base_alpha_qqq["alpha_ann"]) else np.nan,
        "StressAlphaNW_QQQ": round(stress_alpha_qqq["alpha_ann"], 8) if np.isfinite(stress_alpha_qqq["alpha_ann"]) else np.nan,
        "BaseAlphaNW_SPY": round(base_alpha_spy["alpha_ann"], 8) if np.isfinite(base_alpha_spy["alpha_ann"]) else np.nan,
        "StressAlphaNW_SPY": round(stress_alpha_spy["alpha_ann"], 8) if np.isfinite(stress_alpha_spy["alpha_ann"]) else np.nan,
    }


def _apply_empirical_path_stress_simple(
    obj: Dict[str, Any],
    cfg: Mahoraga14Config,
    block: int = 10,
    injections: int = 2,
) -> Dict[str, Any]:
    returns = pd.Series(obj["returns"], dtype=float).copy()
    if len(returns) < block * 4:
        return obj
    roll = ((1.0 + returns).rolling(block).apply(np.prod, raw=True) - 1.0).dropna().sort_values()
    ends: List[int] = []
    for dt in roll.index:
        pos = returns.index.get_loc(dt)
        if all(abs(pos - prev) >= block for prev in ends):
            ends.append(int(pos))
        if len(ends) >= injections:
            break
    stressed = returns.copy()
    targets = np.linspace(block, max(block, len(returns) - block - 1), len(ends) + 2, dtype=int)[1:-1]
    for src_end, tgt_start in zip(ends, targets):
        src_start = src_end - block + 1
        tgt_end = min(len(stressed), tgt_start + block)
        src_block = returns.iloc[src_start : src_start + (tgt_end - tgt_start)].values
        tgt_block = stressed.iloc[tgt_start:tgt_end].values
        stressed.iloc[tgt_start:tgt_end] = np.clip((1.0 + tgt_block) * (1.0 + src_block) - 1.0, -0.95, None)
    return _object_from_returns(stressed, obj["exposure"], obj["turnover"], obj["transaction_cost"], cfg)


def _build_stress_and_robustness(wf: Dict[str, Any], cfg: Mahoraga14Config, costs: m6.CostsConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    primary_obj = wf["stitched_variants"][cfg.primary_variant_key]
    qqq_obj = wf["stitched_benchmarks"]["QQQ"]
    spy_obj = wf["stitched_benchmarks"]["SPY"]
    primary_r = pd.Series(primary_obj["returns"], dtype=float)
    gross = pd.Series(primary_obj.get("gross_returns", primary_r), dtype=float).reindex(primary_r.index).fillna(primary_r)
    tc = pd.Series(primary_obj.get("transaction_cost", pd.Series(0.0, index=primary_r.index)), dtype=float).reindex(primary_r.index).fillna(0.0)
    exposure = pd.Series(primary_obj["exposure"], dtype=float).reindex(primary_r.index).fillna(0.0)
    turnover = pd.Series(primary_obj["turnover"], dtype=float).reindex(primary_r.index).fillna(0.0)
    allocator = _stitch_variant_frame(wf["results"], cfg.primary_variant_key, "allocator_daily").reindex(primary_r.index).ffill()
    leader = allocator.get("leader_blend", pd.Series(0.0, index=primary_r.index)).reindex(primary_r.index).fillna(0.0)
    budget = allocator.get("long_budget", pd.Series(cfg.participation_long_budget_base, index=primary_r.index)).reindex(primary_r.index).fillna(cfg.participation_long_budget_base)
    qqq_r = pd.Series(qqq_obj["returns"], dtype=float).reindex(primary_r.index).fillna(0.0)

    scenarios = {
        "BASELINE": _object_from_returns(primary_r, exposure, turnover, tc, cfg),
        "COST_PLUS_25": _object_from_returns(gross - tc.abs() * 1.25, exposure, turnover, tc.abs() * 1.25, cfg),
        "COST_PLUS_50": _object_from_returns(gross - tc.abs() * 1.50, exposure, turnover, tc.abs() * 1.50, cfg),
        "COST_PLUS_100": _object_from_returns(gross - tc.abs() * 2.00, exposure, turnover, tc.abs() * 2.00, cfg),
        "SLIPPAGE_PLUS_5BPS": _object_from_returns(primary_r - turnover * 0.0005, exposure, turnover, tc + turnover * 0.0005, cfg),
        "ALLOCATOR_TIGHTER": _object_from_returns(primary_r * np.clip((budget * cfg.diagnostics_allocator_tighter_mult) / budget.replace(0.0, np.nan), 0.75, 1.0).fillna(1.0), exposure * cfg.diagnostics_allocator_tighter_mult, turnover, tc, cfg),
        "ALLOCATOR_LOOSER": _object_from_returns(primary_r * np.clip(np.minimum(1.0, budget * cfg.diagnostics_allocator_looser_mult) / budget.replace(0.0, np.nan), 1.0, 1.20).fillna(1.0), exposure * cfg.diagnostics_allocator_looser_mult, turnover, tc, cfg),
        "BULL_PARTICIPATION_WEAKER": _object_from_returns(primary_r - 0.25 * leader * primary_r.clip(lower=0.0) * (qqq_r > 0.0).astype(float), exposure, turnover, tc, cfg),
        "BULL_PARTICIPATION_STRONGER": _object_from_returns(primary_r + 0.25 * leader * primary_r.clip(lower=0.0) * (qqq_r > 0.0).astype(float), exposure, turnover, tc, cfg),
        "EMPIRICAL_PATH_STRESS": _apply_empirical_path_stress_simple(_object_from_returns(primary_r, exposure, turnover, tc, cfg), cfg),
    }
    notes = {
        "BASELINE": "unstressed stitched OOS",
        "COST_PLUS_25": "commission/slippage x1.25 proxy",
        "COST_PLUS_50": "commission/slippage x1.50 proxy",
        "COST_PLUS_100": "commission/slippage x2.00 proxy",
        "SLIPPAGE_PLUS_5BPS": "extra slippage +5bps via turnover",
        "ALLOCATOR_TIGHTER": "allocator budgets/caps tighter proxy",
        "ALLOCATOR_LOOSER": "allocator budgets/caps looser proxy",
        "BULL_PARTICIPATION_WEAKER": "bull participation weaker proxy",
        "BULL_PARTICIPATION_STRONGER": "bull participation stronger proxy",
        "EMPIRICAL_PATH_STRESS": "replay worst empirical blocks on later windows",
    }
    stress_rows = [
        _stress_row(name, notes[name], scenarios["BASELINE"], obj, qqq_obj, spy_obj, cfg)
        for name, obj in scenarios.items()
    ]
    stress_df = pd.DataFrame(stress_rows).round(8)

    rng = np.random.default_rng(77)
    values = primary_r.fillna(0.0).values
    robust_rows = []
    for sample_id in range(int(cfg.diagnostics_bootstrap_samples)):
        sample = np.empty(len(values), dtype=float)
        idx = int(rng.integers(0, len(values)))
        p = 1.0 / max(1, int(cfg.diagnostics_bootstrap_block))
        for i in range(len(values)):
            if i == 0 or rng.random() < p:
                idx = int(rng.integers(0, len(values)))
            else:
                idx = (idx + 1) % len(values)
            sample[i] = values[idx]
        s_obj = _object_from_returns(pd.Series(sample, index=primary_r.index), exposure, turnover, tc, cfg)
        summary = m6.summarize(s_obj["returns"], s_obj["equity"], s_obj["exposure"], s_obj["turnover"], cfg, "boot")
        robust_rows.append({"Variant": "MAHORAGA14_2_LONG_PARTICIPATION", "Method": "stationary_block_bootstrap", "SampleId": sample_id, "CAGR": summary["CAGR"], "Sharpe": summary["Sharpe"], "MaxDD": summary["MaxDD"]})
    sample_id = len(robust_rows)
    for budget_mult in (cfg.diagnostics_allocator_tighter_mult, 1.0, cfg.diagnostics_allocator_looser_mult):
        for leader_mult in (cfg.diagnostics_bull_weaker_mult, 1.0, cfg.diagnostics_bull_stronger_mult):
            scaled = primary_r * np.clip(np.minimum(1.0, budget * budget_mult) / budget.replace(0.0, np.nan), 0.75, 1.20).fillna(1.0)
            scaled = scaled + (leader_mult - 1.0) * leader * scaled.clip(lower=0.0) * (qqq_r > 0.0).astype(float)
            s_obj = _object_from_returns(scaled, exposure * budget_mult, turnover, tc, cfg)
            summary = m6.summarize(s_obj["returns"], s_obj["equity"], s_obj["exposure"], s_obj["turnover"], cfg, "local")
            robust_rows.append(
                {
                    "Variant": "MAHORAGA14_2_LONG_PARTICIPATION",
                    "Method": "local_param_neighborhood",
                    "SampleId": sample_id,
                    "BudgetMultiplier": budget_mult,
                    "LeaderMultiplier": leader_mult,
                    "CAGR": summary["CAGR"],
                    "Sharpe": summary["Sharpe"],
                    "MaxDD": summary["MaxDD"],
                }
            )
            sample_id += 1
    return stress_df, pd.DataFrame(robust_rows).round(8)


def _fast_fail_status(
    comparison_df: pd.DataFrame,
    scorecard_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    cfg: Mahoraga14Config,
) -> Tuple[str, List[str]]:
    primary = comparison_df[comparison_df["Variant"] == "MAHORAGA14_2_LONG_PARTICIPATION"].iloc[0]
    control = comparison_df[comparison_df["Variant"] == "MAHORAGA14_1_LONG_ONLY_CONTROL"].iloc[0]
    reasons: List[str] = []

    upside_delta = float(primary["UpsideCaptureQQQ"] - control["UpsideCaptureQQQ"])
    exposure_delta = float(primary["AvgExposure"] - control["AvgExposure"])
    sharpe_delta = float(primary["Sharpe"] - control["Sharpe"])
    sortino_delta = float(primary["Sortino"] - control["Sortino"])
    maxdd_delta = float(primary["MaxDD"] - control["MaxDD"])
    beta_delta = float(primary["BetaQQQ"] - control["BetaQQQ"])
    ret_per_exp_delta = float(primary["ReturnPerExposure"] - control["ReturnPerExposure"])

    if upside_delta < 0.03 and exposure_delta < 0.02:
        reasons.append("improvement_microscopic_without_real_participation_shift")
    if exposure_delta > 0.05 and primary["CAGR"] <= control["CAGR"]:
        reasons.append("more_exposure_without_cagr_improvement")
    if sharpe_delta < -0.03:
        reasons.append("sharpe_deterioration")
    if sortino_delta < -0.03:
        reasons.append("sortino_deterioration")
    if maxdd_delta < -2.0:
        reasons.append("maxdd_materially_worse")
    if beta_delta > 0.10 and primary["AlphaNW_QQQ"] <= control["AlphaNW_QQQ"]:
        reasons.append("beta_up_without_alpha_improvement")
    if ret_per_exp_delta < -1e-6:
        reasons.append("return_per_exposure_worse")

    manual = scorecard_df[scorecard_df["Source"] == "MANUAL"].copy()
    if len(manual):
        if float(manual["DeltaReturn_vs_QQQ"].mean()) <= float((manual["Mahoraga14_1ControlReturn"] - manual["QQQReturn"]).mean()):
            reasons.append("bull_windows_not_improved_vs_control")

    non_base = stress_df[stress_df["Scenario"] != "BASELINE"].copy()
    if len(non_base) and float(non_base["DeltaSharpe"].min()) < -0.35:
        reasons.append("stress_suite_fragile")

    return ("FAIL_FAST" if reasons else "PASS"), reasons


def _candidate_audit(
    comparison_df: pd.DataFrame,
    scorecard_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    status: str,
    reasons: List[str],
) -> pd.DataFrame:
    rows = comparison_df.copy()
    rows["FastFailStatus"] = ""
    rows["FastFailReasons"] = ""
    primary_mask = rows["Variant"] == "MAHORAGA14_2_LONG_PARTICIPATION"
    rows.loc[primary_mask, "FastFailStatus"] = status
    rows.loc[primary_mask, "FastFailReasons"] = ";".join(reasons) if reasons else "none"
    if len(scorecard_df):
        bull = scorecard_df[scorecard_df["Source"] == "MANUAL"].copy()
        if len(bull):
            rows.loc[primary_mask, "BullWindowAvgDelta_vs_QQQ"] = float(bull["DeltaReturn_vs_QQQ"].mean())
            rows.loc[primary_mask, "BullWindowAvgDelta_vs_Control"] = float(bull["DeltaReturn_vs_Control"].mean())
    if len(stress_df):
        non_base = stress_df[stress_df["Scenario"] != "BASELINE"].copy()
        rows.loc[primary_mask, "WorstStressDeltaSharpe"] = float(non_base["DeltaSharpe"].min()) if len(non_base) else 0.0
        rows.loc[primary_mask, "WorstStressDeltaCAGR"] = float(non_base["DeltaCAGR%"].min()) if len(non_base) else 0.0
    if len(robustness_df):
        rows.loc[primary_mask, "BootstrapSharpe_p25"] = float(robustness_df["Sharpe"].quantile(0.25))
        rows.loc[primary_mask, "BootstrapMaxDD_p75"] = float(robustness_df["MaxDD"].quantile(0.75))
    return rows.round(8)


def _final_report_md(
    comparison_df: pd.DataFrame,
    scorecard_df: pd.DataFrame,
    decomposition_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    status: str,
    reasons: List[str],
) -> str:
    primary = comparison_df[comparison_df["Variant"] == "MAHORAGA14_2_LONG_PARTICIPATION"].iloc[0]
    control = comparison_df[comparison_df["Variant"] == "MAHORAGA14_1_LONG_ONLY_CONTROL"].iloc[0]
    lines = [
        "# Mahoraga14_2 FAST",
        "",
        f"## Thesis status: {status}",
        "",
    ]
    if reasons:
        lines.append("FAIL-FAST reasons:")
        for reason in reasons:
            lines.append(f"- {reason}")
        lines.append("")
    lines += [
        "## Core comparison",
        comparison_df.to_markdown(index=False),
        "",
        "## Primary vs control deltas",
        f"- CAGR delta: {(primary['CAGR'] - control['CAGR']):.2f} pts",
        f"- Sharpe delta: {(primary['Sharpe'] - control['Sharpe']):.4f}",
        f"- Sortino delta: {(primary['Sortino'] - control['Sortino']):.4f}",
        f"- MaxDD delta: {(primary['MaxDD'] - control['MaxDD']):.2f} pts",
        f"- AvgExposure delta: {(primary['AvgExposure'] - control['AvgExposure']):.4f}",
        f"- UpsideCaptureQQQ delta: {(primary['UpsideCaptureQQQ'] - control['UpsideCaptureQQQ']):.4f}",
        "",
        "## Bull windows",
        scorecard_df.to_markdown(index=False) if len(scorecard_df) else "No bull windows available.",
        "",
        "## Upside participation decomposition",
        decomposition_df.to_markdown(index=False) if len(decomposition_df) else "No decomposition available.",
        "",
        "## Stress suite",
        stress_df.to_markdown(index=False) if len(stress_df) else "No stress suite available.",
        "",
        "## Robustness samples",
        robustness_df.head(20).to_markdown(index=False) if len(robustness_df) else "No robustness samples available.",
    ]
    return "\n".join(lines)


def build_fast_report_text(wf: Dict[str, Any], cfg: Mahoraga14Config) -> str:
    comparison_df = _build_comparison_df(wf, cfg)
    scorecard_df = _build_bull_window_scorecard(wf, cfg)
    stress_df, robustness_df = _build_stress_and_robustness(wf, cfg, costs=m6.CostsConfig())
    decomposition_df = _build_upside_participation_decomposition(wf, cfg, scorecard_df)
    status, reasons = _fast_fail_status(comparison_df, scorecard_df, stress_df, cfg)
    return _final_report_md(comparison_df, scorecard_df, decomposition_df, stress_df, robustness_df, status, reasons)


def save_fast_outputs(wf: Dict[str, Any], cfg: Mahoraga14Config, costs: m6.CostsConfig | None = None) -> Dict[str, pd.DataFrame]:
    ensure_dir(cfg.outputs_dir)
    costs = costs or m6.CostsConfig()

    comparison_df = _build_comparison_df(wf, cfg)
    scorecard_df = _build_bull_window_scorecard(wf, cfg)
    active_df, curve_df = _build_active_return_vs_qqq(wf, cfg)
    decomposition_df = _build_upside_participation_decomposition(wf, cfg, scorecard_df)
    allocator_cash_drag_df = _build_allocator_cash_drag(wf, cfg)
    leader_miss_df = _build_leader_miss_analysis(wf, cfg, scorecard_df)
    stress_df, robustness_df = _build_stress_and_robustness(wf, cfg, costs)
    status, reasons = _fast_fail_status(comparison_df, scorecard_df, stress_df, cfg)
    candidate_audit_df = _candidate_audit(comparison_df, scorecard_df, stress_df, robustness_df, status, reasons)
    final_md = _final_report_md(comparison_df, scorecard_df, decomposition_df, stress_df, robustness_df, status, reasons)

    out_dir = Path(cfg.outputs_dir)
    comparison_df.to_csv(out_dir / "stitched_comparison_fast_14_2.csv", index=False)
    scorecard_df.to_csv(out_dir / "bull_window_scorecard_fast.csv", index=False)
    active_df.to_csv(out_dir / "active_return_vs_qqq_fast.csv", index=False)
    curve_df.to_csv(out_dir / "active_return_vs_qqq_curve.csv", index=False)
    decomposition_df.to_csv(out_dir / "upside_participation_decomposition_fast.csv", index=False)
    allocator_cash_drag_df.to_csv(out_dir / "allocator_cash_drag_fast.csv", index=False)
    leader_miss_df.to_csv(out_dir / "leader_miss_analysis_fast.csv", index=False)
    candidate_audit_df.to_csv(out_dir / "candidate_audit_fast_14_2.csv", index=False)
    stress_df.to_csv(out_dir / "stress_suite_fast_14_2.csv", index=False)
    robustness_df.to_csv(out_dir / "robustness_suite_fast_14_2.csv", index=False)
    _curve_plot(curve_df, out_dir / "active_return_vs_qqq_curve.png")
    with open(out_dir / "final_report_fast_14_2.md", "w", encoding="utf-8") as f:
        f.write(final_md)

    return {
        "stitched_comparison_fast_14_2": comparison_df,
        "bull_window_scorecard_fast": scorecard_df,
        "active_return_vs_qqq_fast": active_df,
        "active_return_vs_qqq_curve": curve_df,
        "upside_participation_decomposition_fast": decomposition_df,
        "allocator_cash_drag_fast": allocator_cash_drag_df,
        "leader_miss_analysis_fast": leader_miss_df,
        "candidate_audit_fast_14_2": candidate_audit_df,
        "stress_suite_fast_14_2": stress_df,
        "robustness_suite_fast_14_2": robustness_df,
    }
