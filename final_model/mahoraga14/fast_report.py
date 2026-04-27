from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

import mahoraga6_1 as m6
from mahoraga14_config import Mahoraga14Config
from mahoraga14_utils import bhy_qvalues, ensure_dir, paired_ttest_pvalue


def _variant_keys(cfg: Mahoraga14Config) -> List[str]:
    return [
        cfg.official_baseline_label,
        cfg.main_variant_key,
        cfg.continuation_variant_key,
        cfg.combo_variant_key,
    ]


def _variant_label_map(cfg: Mahoraga14Config) -> Dict[str, str]:
    return {
        cfg.historical_benchmark_label: "LEGACY",
        "QQQ": "QQQ",
        "SPY": "SPY",
        cfg.official_baseline_label: "BASE_ALPHA_V2",
        cfg.main_variant_key: "BASE_ALPHA_V2 + STRUCTURAL_DEFENSE_ONLY",
        cfg.continuation_variant_key: "BASE_ALPHA_V2 + CONTINUATION_PRESSURE_V2_ONLY",
        cfg.combo_variant_key: "BASE_ALPHA_V2 + STRUCTURAL_DEFENSE_ONLY + CONTINUATION_PRESSURE_V2",
    }


def _generic_summary(obj: Dict[str, Any], cfg: Mahoraga14Config, label: str) -> Dict[str, float]:
    return m6.summarize(obj["returns"], obj["equity"], obj["exposure"], obj["turnover"], cfg, label)


def _capture_ratio(strategy_r: pd.Series, bench_r: pd.Series, upside: bool) -> float:
    s = pd.Series(strategy_r).dropna()
    b = pd.Series(bench_r).reindex(s.index).fillna(0.0)
    mask = b > 0.0 if upside else b < 0.0
    denom = float(b.loc[mask].sum())
    if not mask.any() or abs(denom) < 1e-12:
        return 0.0
    return float(s.loc[mask].sum() / denom)


def _beta(strategy_r: pd.Series, bench_r: pd.Series) -> float:
    s = pd.Series(strategy_r).dropna()
    b = pd.Series(bench_r).reindex(s.index).fillna(0.0)
    if len(s) < 2 or float(b.var()) == 0.0:
        return 0.0
    return float(s.cov(b) / b.var())


def _return_per_exposure(r: pd.Series, exposure: pd.Series) -> float:
    exp = pd.Series(exposure).reindex(pd.Index(r.index)).fillna(0.0)
    denom = float(exp.mean())
    if denom <= 1e-12:
        return 0.0
    return float(pd.Series(r).mean() / denom)


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


def _stitched_map(wf: Dict[str, Any], cfg: Mahoraga14Config) -> Dict[str, Dict[str, Any]]:
    return {
        cfg.historical_benchmark_label: wf["stitched_legacy"],
        "QQQ": wf["stitched_benchmarks"]["QQQ"],
        "SPY": wf["stitched_benchmarks"]["SPY"],
        **wf["stitched_variants"],
    }


def _fold_map(result: Dict[str, Any], cfg: Mahoraga14Config) -> Dict[str, Dict[str, Any]]:
    start = pd.Timestamp(result["test_start"])
    end = pd.Timestamp(result["test_end"])
    base_bt = result["base_bt"]

    def _bench_obj(key: str) -> Dict[str, Any]:
        r = base_bt["bench"][f"{key}_r"].loc[start:end]
        return {
            "returns": r,
            "equity": cfg.capital_initial * (1.0 + r).cumprod(),
            "exposure": pd.Series(1.0, index=r.index),
            "turnover": pd.Series(0.0, index=r.index),
        }

    def _bt_obj(bt: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "returns": bt["returns_net"].loc[start:end],
            "equity": cfg.capital_initial * (1.0 + bt["returns_net"].loc[start:end]).cumprod(),
            "exposure": bt["exposure"].loc[start:end],
            "turnover": bt["turnover"].loc[start:end],
        }

    return {
        cfg.historical_benchmark_label: _bt_obj(result["legacy_bt"]),
        "QQQ": _bench_obj("QQQ"),
        "SPY": _bench_obj("SPY"),
        cfg.official_baseline_label: _bt_obj(result["variant_bts"][cfg.official_baseline_label]),
        cfg.main_variant_key: _bt_obj(result["variant_bts"][cfg.main_variant_key]),
        cfg.continuation_variant_key: _bt_obj(result["variant_bts"][cfg.continuation_variant_key]),
        cfg.combo_variant_key: _bt_obj(result["variant_bts"][cfg.combo_variant_key]),
    }


def _metrics_row(label: str, obj: Dict[str, Any], qqq_obj: Dict[str, Any], spy_obj: Dict[str, Any], cfg: Mahoraga14Config) -> Dict[str, Any]:
    summary = _generic_summary(obj, cfg, label)
    alpha_qqq = _alpha_nw(obj["returns"], qqq_obj["returns"], cfg, f"{label}_QQQ")
    alpha_spy = _alpha_nw(obj["returns"], spy_obj["returns"], cfg, f"{label}_SPY")
    return {
        "Variant": label,
        "CAGR%": round(summary["CAGR"] * 100.0, 2),
        "Sharpe": round(summary["Sharpe"], 4),
        "MaxDD%": round(summary["MaxDD"] * 100.0, 2),
        "AvgExposure": round(float(obj["exposure"].mean()), 4),
        "AvgTurnover": round(float(obj["turnover"].mean()), 4),
        "ReturnPerExposure": round(_return_per_exposure(obj["returns"], obj["exposure"]), 6),
        "BetaQQQ": round(_beta(obj["returns"], qqq_obj["returns"]), 4),
        "BetaSPY": round(_beta(obj["returns"], spy_obj["returns"]), 4),
        "UpsideCaptureQQQ": round(_capture_ratio(obj["returns"], qqq_obj["returns"], upside=True), 4),
        "DownsideCaptureQQQ": round(_capture_ratio(obj["returns"], qqq_obj["returns"], upside=False), 4),
        "AlphaNW_QQQ": round(alpha_qqq["alpha_ann"], 6) if np.isfinite(alpha_qqq["alpha_ann"]) else np.nan,
        "AlphaNW_QQQ_t": round(alpha_qqq["t_alpha"], 3) if np.isfinite(alpha_qqq["t_alpha"]) else np.nan,
        "AlphaNW_SPY": round(alpha_spy["alpha_ann"], 6) if np.isfinite(alpha_spy["alpha_ann"]) else np.nan,
        "AlphaNW_SPY_t": round(alpha_spy["t_alpha"], 3) if np.isfinite(alpha_spy["t_alpha"]) else np.nan,
    }


def _build_stitched_comparison_fast(wf: Dict[str, Any], cfg: Mahoraga14Config) -> pd.DataFrame:
    labels = _variant_label_map(cfg)
    objects = _stitched_map(wf, cfg)
    order = [
        cfg.historical_benchmark_label,
        "QQQ",
        "SPY",
        cfg.official_baseline_label,
        cfg.main_variant_key,
        cfg.continuation_variant_key,
        cfg.combo_variant_key,
    ]
    qqq_obj = objects["QQQ"]
    spy_obj = objects["SPY"]
    rows = [_metrics_row(labels[key], objects[key], qqq_obj, spy_obj, cfg) for key in order]
    return pd.DataFrame(rows)


def _build_fold_summary_fast(wf: Dict[str, Any]) -> pd.DataFrame:
    return wf["fold_df"].copy().sort_values("fold")


def _build_ablation_fast_df(wf: Dict[str, Any], cfg: Mahoraga14Config) -> pd.DataFrame:
    labels = _variant_label_map(cfg)
    rows: List[Dict[str, Any]] = []
    order = _variant_keys(cfg)
    for result in wf["results"]:
        objects = _fold_map(result, cfg)
        base_sharpe = _generic_summary(objects[cfg.official_baseline_label], cfg, "BASE")["Sharpe"]
        legacy_sharpe = _generic_summary(objects[cfg.historical_benchmark_label], cfg, "LEGACY")["Sharpe"]
        qqq_sharpe = _generic_summary(objects["QQQ"], cfg, "QQQ")["Sharpe"]
        spy_sharpe = _generic_summary(objects["SPY"], cfg, "SPY")["Sharpe"]
        qqq_obj = objects["QQQ"]
        spy_obj = objects["SPY"]
        for variant in order:
            row = _metrics_row(labels[variant], objects[variant], qqq_obj, spy_obj, cfg)
            row.update(
                {
                    "Segment": f"FOLD_{int(result['fold'])}",
                    "Fold": int(result["fold"]),
                    "SharpeDeltaVsBASE_ALPHA": round(row["Sharpe"] - base_sharpe, 4),
                    "SharpeDeltaVsLEGACY": round(row["Sharpe"] - legacy_sharpe, 4),
                    "SharpeDeltaVsQQQ": round(row["Sharpe"] - qqq_sharpe, 4),
                    "SharpeDeltaVsSPY": round(row["Sharpe"] - spy_sharpe, 4),
                }
            )
            rows.append(row)
    stitched = _build_stitched_comparison_fast(wf, cfg).set_index("Variant")
    base_sharpe = float(stitched.loc[labels[cfg.official_baseline_label], "Sharpe"])
    legacy_sharpe = float(stitched.loc[labels[cfg.historical_benchmark_label], "Sharpe"])
    qqq_sharpe = float(stitched.loc[labels["QQQ"], "Sharpe"])
    spy_sharpe = float(stitched.loc[labels["SPY"], "Sharpe"])
    for variant in order:
        row = stitched.loc[labels[variant]].to_dict()
        row.update(
            {
                "Segment": "STITCHED",
                "Fold": 0,
                "Variant": labels[variant],
                "SharpeDeltaVsBASE_ALPHA": round(float(row["Sharpe"]) - base_sharpe, 4),
                "SharpeDeltaVsLEGACY": round(float(row["Sharpe"]) - legacy_sharpe, 4),
                "SharpeDeltaVsQQQ": round(float(row["Sharpe"]) - qqq_sharpe, 4),
                "SharpeDeltaVsSPY": round(float(row["Sharpe"]) - spy_sharpe, 4),
            }
        )
        rows.append(row)
    cols = [
        "Segment",
        "Fold",
        "Variant",
        "CAGR%",
        "Sharpe",
        "MaxDD%",
        "AvgExposure",
        "AvgTurnover",
        "ReturnPerExposure",
        "SharpeDeltaVsBASE_ALPHA",
        "SharpeDeltaVsLEGACY",
        "SharpeDeltaVsQQQ",
        "SharpeDeltaVsSPY",
    ]
    return pd.DataFrame(rows)[cols]


def _stitched_override_weekly(wf: Dict[str, Any], variant: str) -> pd.DataFrame:
    frames = []
    for result in wf["results"]:
        frame = result["variant_runs"][variant]["override_weekly"].copy()
        frame["fold"] = int(result["fold"])
        frames.append(frame)
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()


def _build_pvalue_qvalue_fast(wf: Dict[str, Any], cfg: Mahoraga14Config) -> pd.DataFrame:
    labels = _variant_label_map(cfg)
    rows: List[Dict[str, Any]] = []

    def add_segment(segment: str, fold: int, objects: Dict[str, Dict[str, Any]]) -> None:
        pairs = [
            (cfg.official_baseline_label, cfg.historical_benchmark_label),
            (cfg.official_baseline_label, "QQQ"),
            (cfg.official_baseline_label, "SPY"),
            (cfg.main_variant_key, cfg.official_baseline_label),
            (cfg.main_variant_key, cfg.historical_benchmark_label),
            (cfg.main_variant_key, "QQQ"),
            (cfg.main_variant_key, "SPY"),
            (cfg.continuation_variant_key, cfg.official_baseline_label),
            (cfg.continuation_variant_key, cfg.historical_benchmark_label),
            (cfg.continuation_variant_key, "QQQ"),
            (cfg.continuation_variant_key, "SPY"),
            (cfg.combo_variant_key, cfg.official_baseline_label),
            (cfg.combo_variant_key, cfg.historical_benchmark_label),
            (cfg.combo_variant_key, "QQQ"),
            (cfg.combo_variant_key, "SPY"),
        ]
        for target, reference in pairs:
            rows.append(
                {
                    "Segment": segment,
                    "Fold": fold,
                    "Target": labels[target],
                    "Reference": labels[reference],
                    "Comparison": f"{labels[target]}_vs_{labels[reference]}",
                    "p_value": paired_ttest_pvalue(objects[target]["returns"] - objects[reference]["returns"], alternative="greater"),
                }
            )

    for result in wf["results"]:
        add_segment(f"FOLD_{int(result['fold'])}", int(result["fold"]), _fold_map(result, cfg))
    add_segment("STITCHED", 0, _stitched_map(wf, cfg))
    df = pd.DataFrame(rows)
    df["q_value"] = bhy_qvalues(df["p_value"].values, alpha=cfg.bhy_alpha) if len(df) else []
    df["p_value"] = df["p_value"].round(6)
    df["q_value"] = df["q_value"].round(6)
    return df


def _build_override_usage_fast(wf: Dict[str, Any], cfg: Mahoraga14Config) -> pd.DataFrame:
    labels = _variant_label_map(cfg)
    rows: List[Dict[str, Any]] = []

    def summarize(frame: pd.DataFrame, weekly_frame: pd.DataFrame, segment: str, fold: int, variant: str) -> Dict[str, Any]:
        if len(frame) == 0:
            return {
                "Segment": segment,
                "Fold": fold,
                "Variant": labels[variant],
                "OverrideRate": 0.0,
                "StructuralRate": 0.0,
                "ContinuationLiftRate": 0.0,
                "ContinuationActivationRate": 0.0,
                "MeanDefenseBlend": 0.0,
                "MeanGate": 1.0,
                "MeanVolMult": 1.0,
                "MeanExpCap": 1.0,
            }
        return {
            "Segment": segment,
            "Fold": fold,
            "Variant": labels[variant],
            "OverrideRate": round(float(frame["is_override"].mean()), 4),
            "StructuralRate": round(float(frame["is_structural_override"].mean()), 4),
            "ContinuationLiftRate": round(float(frame.get("is_continuation_lift", pd.Series(0.0, index=frame.index)).mean()), 4),
            "ContinuationActivationRate": round(float(weekly_frame.get("is_continuation_activation", pd.Series(0.0, index=weekly_frame.index)).mean()) if len(weekly_frame) else 0.0, 4),
            "MeanDefenseBlend": round(float(frame["defense_blend"].mean()), 4),
            "MeanGate": round(float(frame["gate_scale"].mean()), 4),
            "MeanVolMult": round(float(frame["vol_mult"].mean()), 4),
            "MeanExpCap": round(float(frame["exp_cap"].mean()), 4),
        }

    for result in wf["results"]:
        fold = int(result["fold"])
        rows.append(summarize(pd.DataFrame(), pd.DataFrame(), f"FOLD_{fold}", fold, cfg.official_baseline_label))
        for variant in [cfg.main_variant_key, cfg.continuation_variant_key, cfg.combo_variant_key]:
            frame = result["variant_runs"][variant]["override_daily"].loc[result["test_start"] : result["test_end"]]
            weekly_frame = result["variant_runs"][variant]["override_weekly"]
            rows.append(summarize(frame, weekly_frame, f"FOLD_{fold}", fold, variant))

    rows.append(summarize(pd.DataFrame(), pd.DataFrame(), "STITCHED", 0, cfg.official_baseline_label))
    for variant in [cfg.main_variant_key, cfg.continuation_variant_key, cfg.combo_variant_key]:
        rows.append(summarize(wf["stitched_override_daily"][variant], _stitched_override_weekly(wf, variant), "STITCHED", 0, variant))
    return pd.DataFrame(rows)


def _build_continuation_usage_fast(wf: Dict[str, Any], cfg: Mahoraga14Config) -> pd.DataFrame:
    labels = _variant_label_map(cfg)
    rows: List[Dict[str, Any]] = []

    def summarize(frame: pd.DataFrame, weekly_frame: pd.DataFrame, segment: str, fold: int, variant: str) -> Dict[str, Any]:
        if len(frame) == 0:
            return {
                "Segment": segment,
                "Fold": fold,
                "Variant": labels[variant],
                "ContinuationLiftRate": 0.0,
                "ContinuationActivationRate": 0.0,
                "MeanTriggerP": 0.0,
                "MeanPressureP": 0.0,
                "MeanBreakRiskP": 0.0,
                "MeanTriggerScore": 0.0,
                "MeanPressureScore": 0.0,
                "MeanContinuationPressure": 0.0,
                "MeanBenchmarkScore": 0.0,
                "MeanStructuralHeadroom": 0.0,
                "MeanBreakHeadroom": 0.0,
            }
        return {
            "Segment": segment,
            "Fold": fold,
            "Variant": labels[variant],
            "ContinuationLiftRate": round(float(frame.get("is_continuation_lift", pd.Series(0.0, index=frame.index)).mean()), 4),
            "ContinuationActivationRate": round(float(weekly_frame.get("is_continuation_activation", pd.Series(0.0, index=weekly_frame.index)).mean()) if len(weekly_frame) else 0.0, 4),
            "MeanTriggerP": round(float(frame.get("continuation_trigger_p", pd.Series(0.0, index=frame.index)).mean()), 4),
            "MeanPressureP": round(float(frame.get("continuation_pressure_p", pd.Series(0.0, index=frame.index)).mean()), 4),
            "MeanBreakRiskP": round(float(frame.get("continuation_break_risk_p", pd.Series(0.0, index=frame.index)).mean()), 4),
            "MeanTriggerScore": round(float(frame.get("continuation_trigger_score", pd.Series(0.0, index=frame.index)).mean()), 4),
            "MeanPressureScore": round(float(frame.get("continuation_pressure_score", pd.Series(0.0, index=frame.index)).mean()), 4),
            "MeanContinuationPressure": round(float(frame.get("continuation_pressure", pd.Series(0.0, index=frame.index)).mean()), 4),
            "MeanBenchmarkScore": round(float(frame.get("continuation_benchmark_score", pd.Series(0.0, index=frame.index)).mean()), 4),
            "MeanStructuralHeadroom": round(float(frame.get("continuation_structural_headroom", pd.Series(0.0, index=frame.index)).mean()), 4),
            "MeanBreakHeadroom": round(float(frame.get("continuation_break_headroom", pd.Series(0.0, index=frame.index)).mean()), 4),
        }

    for result in wf["results"]:
        fold = int(result["fold"])
        rows.append(summarize(pd.DataFrame(), pd.DataFrame(), f"FOLD_{fold}", fold, cfg.official_baseline_label))
        for variant in [cfg.main_variant_key, cfg.continuation_variant_key, cfg.combo_variant_key]:
            frame = result["variant_runs"][variant]["override_daily"].loc[result["test_start"] : result["test_end"]]
            weekly_frame = result["variant_runs"][variant]["override_weekly"]
            rows.append(summarize(frame, weekly_frame, f"FOLD_{fold}", fold, variant))

    rows.append(summarize(pd.DataFrame(), pd.DataFrame(), "STITCHED", 0, cfg.official_baseline_label))
    for variant in [cfg.main_variant_key, cfg.continuation_variant_key, cfg.combo_variant_key]:
        rows.append(summarize(wf["stitched_override_daily"][variant], _stitched_override_weekly(wf, variant), "STITCHED", 0, variant))
    return pd.DataFrame(rows)


def _event_window_returns(r: pd.Series, dt: pd.Timestamp, horizon: int) -> float:
    s = pd.Series(r).sort_index()
    if dt not in s.index:
        return np.nan
    loc = s.index.get_loc(dt)
    if isinstance(loc, slice):
        loc = loc.start
    start = int(loc) + 1
    end = min(len(s), start + horizon)
    if start >= end:
        return np.nan
    return float(np.prod(1.0 + s.iloc[start:end].values) - 1.0)


def _event_window_drawdown(eq: pd.Series, dt: pd.Timestamp, horizon: int) -> float:
    s = pd.Series(eq).sort_index().ffill()
    if dt not in s.index:
        return np.nan
    loc = s.index.get_loc(dt)
    if isinstance(loc, slice):
        loc = loc.start
    start = int(loc)
    end = min(len(s), start + horizon + 1)
    if start + 1 >= end:
        return np.nan
    dd = s / s.cummax() - 1.0
    return float(dd.iloc[start + 1 : end].min() - dd.iloc[start])


def _continuation_event_rows(result: Dict[str, Any], cfg: Mahoraga14Config) -> List[Dict[str, Any]]:
    labels = _variant_label_map(cfg)
    rows: List[Dict[str, Any]] = []
    for variant in [cfg.continuation_variant_key, cfg.combo_variant_key]:
        weekly = result["variant_runs"][variant]["override_weekly"].copy()
        daily_bt = result["variant_bts"][variant]
        events = weekly.index[weekly["is_continuation_activation"] > 0.0]
        controls = weekly.index[weekly["is_continuation_activation"] <= 0.0]
        event_metrics = []
        control_metrics = []
        for dt in events:
            event_metrics.append(
                {
                    "r1": _event_window_returns(daily_bt["returns_net"], dt, 5),
                    "r2": _event_window_returns(daily_bt["returns_net"], dt, 10),
                    "r4": _event_window_returns(daily_bt["returns_net"], dt, 20),
                    "dd4": _event_window_drawdown(daily_bt["equity"], dt, 20),
                }
            )
        for dt in controls:
            control_metrics.append(
                {
                    "r1": _event_window_returns(daily_bt["returns_net"], dt, 5),
                    "r2": _event_window_returns(daily_bt["returns_net"], dt, 10),
                    "r4": _event_window_returns(daily_bt["returns_net"], dt, 20),
                    "dd4": _event_window_drawdown(daily_bt["equity"], dt, 20),
                }
            )
        e = pd.DataFrame(event_metrics)
        c = pd.DataFrame(control_metrics)
        rows.append(
            {
                "Segment": f"FOLD_{int(result['fold'])}",
                "Fold": int(result["fold"]),
                "Variant": labels[variant],
                "Activations": int(len(events)),
                "HitRate": round(float((e["r4"] > 0.0).mean()) if len(e) else 0.0, 4),
                "MeanRet1W": round(float(e["r1"].mean()) if len(e) else 0.0, 6),
                "MeanRet2W": round(float(e["r2"].mean()) if len(e) else 0.0, 6),
                "MeanRet4W": round(float(e["r4"].mean()) if len(e) else 0.0, 6),
                "MeanPostDD4W": round(float(e["dd4"].mean()) if len(e) else 0.0, 6),
                "NoActivationCount": int(len(controls)),
                "NoActHitRate": round(float((c["r4"] > 0.0).mean()) if len(c) else 0.0, 4),
                "NoActMeanRet1W": round(float(c["r1"].mean()) if len(c) else 0.0, 6),
                "NoActMeanRet2W": round(float(c["r2"].mean()) if len(c) else 0.0, 6),
                "NoActMeanRet4W": round(float(c["r4"].mean()) if len(c) else 0.0, 6),
                "NoActMeanPostDD4W": round(float(c["dd4"].mean()) if len(c) else 0.0, 6),
            }
        )
    return rows


def _build_continuation_event_study_fast(wf: Dict[str, Any], cfg: Mahoraga14Config) -> pd.DataFrame:
    rows = []
    for result in wf["results"]:
        rows.extend(_continuation_event_rows(result, cfg))
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    stitched_rows = []
    for variant in df["Variant"].unique():
        sub = df[df["Variant"] == variant].copy()
        weights = sub["Activations"].replace(0, np.nan)
        stitched_rows.append(
            {
                "Segment": "STITCHED",
                "Fold": 0,
                "Variant": variant,
                "Activations": int(sub["Activations"].sum()),
                "HitRate": round(float(np.average(sub["HitRate"], weights=sub["Activations"].clip(lower=1))), 4),
                "MeanRet1W": round(float(np.average(sub["MeanRet1W"], weights=sub["Activations"].clip(lower=1))), 6),
                "MeanRet2W": round(float(np.average(sub["MeanRet2W"], weights=sub["Activations"].clip(lower=1))), 6),
                "MeanRet4W": round(float(np.average(sub["MeanRet4W"], weights=sub["Activations"].clip(lower=1))), 6),
                "MeanPostDD4W": round(float(np.average(sub["MeanPostDD4W"], weights=sub["Activations"].clip(lower=1))), 6),
                "NoActivationCount": int(sub["NoActivationCount"].sum()),
                "NoActHitRate": round(float(np.average(sub["NoActHitRate"], weights=sub["NoActivationCount"].clip(lower=1))), 4),
                "NoActMeanRet1W": round(float(np.average(sub["NoActMeanRet1W"], weights=sub["NoActivationCount"].clip(lower=1))), 6),
                "NoActMeanRet2W": round(float(np.average(sub["NoActMeanRet2W"], weights=sub["NoActivationCount"].clip(lower=1))), 6),
                "NoActMeanRet4W": round(float(np.average(sub["NoActMeanRet4W"], weights=sub["NoActivationCount"].clip(lower=1))), 6),
                "NoActMeanPostDD4W": round(float(np.average(sub["NoActMeanPostDD4W"], weights=sub["NoActivationCount"].clip(lower=1))), 6),
            }
        )
    return pd.concat([df, pd.DataFrame(stitched_rows)], ignore_index=True)


def _build_alpha_nw_fast(wf: Dict[str, Any], cfg: Mahoraga14Config) -> pd.DataFrame:
    labels = _variant_label_map(cfg)
    rows: List[Dict[str, Any]] = []
    for result in wf["results"]:
        objects = _fold_map(result, cfg)
        for variant in [cfg.historical_benchmark_label, cfg.official_baseline_label, cfg.main_variant_key, cfg.continuation_variant_key, cfg.combo_variant_key]:
            for bench in ["QQQ", "SPY"]:
                alpha = _alpha_nw(objects[variant]["returns"], objects[bench]["returns"], cfg, f"{variant}_{bench}")
                rows.append({"Segment": f"FOLD_{int(result['fold'])}", "Fold": int(result["fold"]), "Variant": labels[variant], "Benchmark": labels[bench], **alpha})
    stitched = _stitched_map(wf, cfg)
    for variant in [cfg.historical_benchmark_label, cfg.official_baseline_label, cfg.main_variant_key, cfg.continuation_variant_key, cfg.combo_variant_key]:
        for bench in ["QQQ", "SPY"]:
            alpha = _alpha_nw(stitched[variant]["returns"], stitched[bench]["returns"], cfg, f"{variant}_{bench}")
            rows.append({"Segment": "STITCHED", "Fold": 0, "Variant": labels[variant], "Benchmark": labels[bench], **alpha})
    return pd.DataFrame(rows)


def _variant_hard_selection_fast(wf: Dict[str, Any], cfg: Mahoraga14Config) -> Dict[str, Dict[str, Any]]:
    fold_df = wf["fold_df"].copy().sort_values("fold")
    labels = _variant_label_map(cfg)
    stitched = _build_stitched_comparison_fast(wf, cfg).set_index("Variant")
    overrides = _build_override_usage_fast(wf, cfg)
    stitched_overrides = overrides[overrides["Segment"] == "STITCHED"].set_index("Variant")
    base_label = labels[cfg.official_baseline_label]
    main_label = labels[cfg.main_variant_key]
    statuses = {
        cfg.historical_benchmark_label: {"status": "BENCHMARK_ONLY", "notes": "historical_only"},
        "QQQ": {"status": "BENCHMARK_ONLY", "notes": "market_reference"},
        "SPY": {"status": "BENCHMARK_ONLY", "notes": "market_reference"},
        cfg.official_baseline_label: {"status": "OFFICIAL_BASELINE", "notes": "official_baseline"},
    }

    base_stitched = float(stitched.loc[base_label, "Sharpe"])
    base_fold5 = float(fold_df.loc[fold_df["fold"] == 5, "BASE_Sharpe"].iloc[0])
    main_fold3 = float(fold_df.loc[fold_df["fold"] == 3, "MAIN_Sharpe"].iloc[0])
    base_ceiling_mean = float(fold_df.loc[fold_df["fold"].isin(cfg.ceiling_folds), "BASE_Sharpe"].mean())
    main_override_rate = float(stitched_overrides.loc[main_label, "OverrideRate"]) if main_label in stitched_overrides.index else 0.0
    override_cap = max(float(cfg.hard_override_rate_abs_cap), main_override_rate + float(cfg.hard_override_rate_buffer))

    for variant, col in {
        cfg.main_variant_key: "MAIN_Sharpe",
        cfg.continuation_variant_key: "CONT_Sharpe",
        cfg.combo_variant_key: "COMBO_Sharpe",
    }.items():
        label = labels[variant]
        reasons = []
        fold4_delta = float(fold_df.loc[fold_df["fold"] == 4, col].iloc[0] - fold_df.loc[fold_df["fold"] == 4, "BASE_Sharpe"].iloc[0])
        ceiling_delta = float(fold_df.loc[fold_df["fold"].isin(cfg.ceiling_folds), col].mean() - base_ceiling_mean)
        fold3_value = float(fold_df.loc[fold_df["fold"] == 3, col].iloc[0])
        fold5_delta = float(fold_df.loc[fold_df["fold"] == 5, col].iloc[0] - base_fold5)
        override_rate = float(stitched_overrides.loc[label, "OverrideRate"]) if label in stitched_overrides.index else 0.0
        activation_rate = float(stitched_overrides.loc[label, "ContinuationActivationRate"]) if label in stitched_overrides.index else 0.0
        stitched_delta = float(stitched.loc[label, "Sharpe"] - base_stitched)

        if fold4_delta < float(cfg.hard_fold4_sharpe_tol):
            reasons.append("fold4_degradation")
        if ceiling_delta < float(cfg.hard_ceiling_mean_sharpe_tol):
            reasons.append("ceiling_mean_degradation")
        if variant == cfg.main_variant_key:
            if fold3_value < float(fold_df.loc[fold_df["fold"] == 3, "BASE_Sharpe"].iloc[0]):
                reasons.append("fold3_lost_structural_gain")
            if stitched_delta < 0.0:
                reasons.append("stitched_below_base")
            status = "MAIN_BRANCH_PASS" if not reasons else "MAIN_BRANCH_FAIL"
        else:
            if fold3_value < main_fold3 + float(cfg.hard_fold3_vs_main_tol):
                reasons.append("fold3_below_main")
            if fold5_delta <= float(cfg.hard_fold5_sharpe_min_delta):
                reasons.append("fold5_no_improvement")
            if activation_rate <= float(cfg.hard_reentry_rate_min):
                reasons.append("activation_near_zero")
            if override_rate > override_cap:
                reasons.append("override_rate_spike")
            if not reasons and stitched_delta >= float(cfg.promising_stitched_sharpe_delta):
                status = "PROMISING"
            elif not reasons:
                status = "PASS_NOT_PROMISING"
            else:
                status = "REJECT"
        statuses[variant] = {"status": status, "notes": "ok" if not reasons else ";".join(reasons)}
    return statuses


def _build_floor_ceiling_summary_fast(wf: Dict[str, Any], cfg: Mahoraga14Config) -> pd.DataFrame:
    labels = _variant_label_map(cfg)
    hard = _variant_hard_selection_fast(wf, cfg)
    rows = []
    stitched = _stitched_map(wf, cfg)
    base_stitched = _generic_summary(stitched[cfg.official_baseline_label], cfg, "BASE")["Sharpe"]
    base_ceiling = float(wf["fold_df"].loc[wf["fold_df"]["fold"].isin(cfg.ceiling_folds), "BASE_Sharpe"].mean())
    base_floor = float(wf["fold_df"].loc[wf["fold_df"]["fold"].isin(cfg.floor_folds), "BASE_Sharpe"].mean())
    base_fold5 = float(wf["fold_df"].loc[wf["fold_df"]["fold"] == 5, "BASE_Sharpe"].iloc[0])

    for key in [cfg.historical_benchmark_label, "QQQ", "SPY", cfg.official_baseline_label, cfg.main_variant_key, cfg.continuation_variant_key, cfg.combo_variant_key]:
        per_fold = []
        for result in wf["results"]:
            obj = _fold_map(result, cfg)[key]
            per_fold.append({"fold": int(result["fold"]), "Sharpe": _generic_summary(obj, cfg, key)["Sharpe"]})
        fold_df = pd.DataFrame(per_fold)
        ceiling = float(fold_df.loc[fold_df["fold"].isin(cfg.ceiling_folds), "Sharpe"].mean())
        floor = float(fold_df.loc[fold_df["fold"].isin(cfg.floor_folds), "Sharpe"].mean())
        fold5 = float(fold_df.loc[fold_df["fold"] == 5, "Sharpe"].iloc[0])
        stitched_sharpe = _generic_summary(stitched[key], cfg, key)["Sharpe"]
        rows.append(
            {
                "Variant": labels[key],
                "CeilingMeanSharpe": round(ceiling, 4),
                "CeilingDeltaVsBase": round(ceiling - base_ceiling, 4),
                "FloorMeanSharpe": round(floor, 4),
                "FloorDeltaVsBase": round(floor - base_floor, 4),
                "Fold5Sharpe": round(fold5, 4),
                "Fold5DeltaVsBase": round(fold5 - base_fold5, 4),
                "StitchedSharpe": round(stitched_sharpe, 4),
                "StitchedDeltaVsBase": round(stitched_sharpe - base_stitched, 4),
                "HardSelectionStatus": hard.get(key, {}).get("status", "BENCHMARK_ONLY"),
                "HardSelectionNotes": hard.get(key, {}).get("notes", "n/a"),
            }
        )
    return pd.DataFrame(rows)


def build_fast_report_text(wf: Dict[str, Any], cfg: Mahoraga14Config) -> str:
    comparison_df = _build_stitched_comparison_fast(wf, cfg)
    fold_df = _build_fold_summary_fast(wf)
    floor_ceiling_df = _build_floor_ceiling_summary_fast(wf, cfg)
    ablation_df = _build_ablation_fast_df(wf, cfg)
    override_df = _build_override_usage_fast(wf, cfg)
    continuation_df = _build_continuation_usage_fast(wf, cfg)
    event_df = _build_continuation_event_study_fast(wf, cfg)
    pq_df = _build_pvalue_qvalue_fast(wf, cfg)
    alpha_nw_df = _build_alpha_nw_fast(wf, cfg)
    lines = [
        "MAHORAGA 14 — FAST REPORT",
        "=" * 78,
        "",
        "STITCHED COMPARISON",
        comparison_df.to_string(index=False),
        "",
        "FLOOR / CEILING SUMMARY",
        floor_ceiling_df.to_string(index=False),
        "",
        "P-VALUE / Q-VALUE",
        pq_df.to_string(index=False),
        "",
        "OVERRIDE USAGE",
        override_df.to_string(index=False),
        "",
        "CONTINUATION USAGE",
        continuation_df.to_string(index=False),
        "",
        "CONTINUATION EVENT STUDY",
        event_df.to_string(index=False),
        "",
        "ALPHA NW",
        alpha_nw_df.to_string(index=False),
        "",
        "ABLATION",
        ablation_df.to_string(index=False),
        "",
        "FOLD SUMMARY",
        fold_df.to_string(index=False),
    ]
    return "\n".join(lines)


def save_fast_outputs(wf: Dict[str, Any], cfg: Mahoraga14Config) -> Dict[str, pd.DataFrame]:
    ensure_dir(cfg.outputs_dir)
    comparison_df = _build_stitched_comparison_fast(wf, cfg)
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

    comparison_df.to_csv(f"{cfg.outputs_dir}/stitched_comparison_fast.csv", index=False)
    fold_df.to_csv(f"{cfg.outputs_dir}/fold_summary_fast.csv", index=False)
    floor_ceiling_df.to_csv(f"{cfg.outputs_dir}/floor_ceiling_summary_fast.csv", index=False)
    ablation_df.to_csv(f"{cfg.outputs_dir}/ablation_fast.csv", index=False)
    override_df.to_csv(f"{cfg.outputs_dir}/override_usage_fast.csv", index=False)
    continuation_df.to_csv(f"{cfg.outputs_dir}/continuation_usage_fast.csv", index=False)
    event_df.to_csv(f"{cfg.outputs_dir}/continuation_event_study_fast.csv", index=False)
    selected_df.to_csv(f"{cfg.outputs_dir}/selected_candidates_fast.csv", index=False)
    support_df.to_csv(f"{cfg.outputs_dir}/selected_config_support_fast.csv", index=False)
    pq_df.to_csv(f"{cfg.outputs_dir}/pvalue_qvalue_fast.csv", index=False)
    alpha_nw_df.to_csv(f"{cfg.outputs_dir}/alpha_nw_fast.csv", index=False)
    continuation_calibration_df.to_csv(f"{cfg.outputs_dir}/continuation_calibration_fast.csv", index=False)

    with open(f"{cfg.outputs_dir}/final_report_fast.txt", "w", encoding="utf-8") as f:
        f.write(build_fast_report_text(wf, cfg))

    return {
        "stitched_comparison_fast": comparison_df,
        "fold_summary_fast": fold_df,
        "floor_ceiling_summary_fast": floor_ceiling_df,
        "ablation_fast": ablation_df,
        "override_usage_fast": override_df,
        "continuation_usage_fast": continuation_df,
        "continuation_event_study_fast": event_df,
        "selected_candidates_fast": selected_df,
        "selected_config_support_fast": support_df,
        "pvalue_qvalue_fast": pq_df,
        "alpha_nw_fast": alpha_nw_df,
        "continuation_calibration_fast": continuation_calibration_df,
    }
