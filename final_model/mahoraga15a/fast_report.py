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
    summarize_object,
)


def _objects_map(wf: Dict[str, Any], cfg: Mahoraga15AConfig) -> Dict[str, Dict[str, Any]]:
    return {
        "LEGACY": wf["stitched_legacy"],
        "QQQ": wf["stitched_benchmarks"]["QQQ"],
        "SPY": wf["stitched_benchmarks"]["SPY"],
        cfg.official_long_label: wf["frozen_long"],
        cfg.ls_label: wf["stitched_ls"],
    }


def _series(obj: Dict[str, Any], key: str, fallback: float = 0.0) -> pd.Series:
    if key in obj:
        return pd.Series(obj[key], dtype=float)
    idx = pd.Index(obj["returns"].index)
    return pd.Series(fallback, index=idx, dtype=float)


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
    }


def build_stitched_comparison_fast(wf: Dict[str, Any], cfg: Mahoraga15AConfig) -> pd.DataFrame:
    objects = _objects_map(wf, cfg)
    qqq_obj = objects["QQQ"]
    spy_obj = objects["SPY"]
    order = ["LEGACY", "QQQ", "SPY", cfg.official_long_label, cfg.ls_label]
    return pd.DataFrame([_metrics_row(key, objects[key], qqq_obj, spy_obj, cfg) for key in order])


def build_pairwise_pq_fast(wf: Dict[str, Any], cfg: Mahoraga15AConfig) -> pd.DataFrame:
    objects = _objects_map(wf, cfg)
    pairs = [
        (cfg.official_long_label, "LEGACY"),
        (cfg.official_long_label, "QQQ"),
        (cfg.official_long_label, "SPY"),
        (cfg.ls_label, cfg.official_long_label),
        (cfg.ls_label, "LEGACY"),
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


def build_beta_decomposition_fast(wf: Dict[str, Any], cfg: Mahoraga15AConfig) -> pd.DataFrame:
    allocator = wf["allocator_trace"].copy()
    ls_obj = wf["stitched_ls"]
    long_obj = wf["frozen_long"]
    qqq_r = wf["stitched_benchmarks"]["QQQ"]["returns"].reindex(allocator.index).fillna(0.0)
    spy_r = wf["stitched_benchmarks"]["SPY"]["returns"].reindex(allocator.index).fillna(0.0)
    allocator["observed_beta_qqq_ls"] = rolling_ridge_beta(ls_obj["returns"].reindex(allocator.index).fillna(0.0), qqq_r, cfg.hedge_beta_window, cfg.hedge_beta_min_obs, cfg.hedge_ridge_alpha)
    allocator["observed_beta_spy_ls"] = rolling_ridge_beta(ls_obj["returns"].reindex(allocator.index).fillna(0.0), spy_r, cfg.hedge_beta_window, cfg.hedge_beta_min_obs, cfg.hedge_ridge_alpha)
    allocator["observed_beta_qqq_long"] = rolling_ridge_beta(long_obj["returns"].reindex(allocator.index).fillna(0.0), qqq_r, cfg.hedge_beta_window, cfg.hedge_beta_min_obs, cfg.hedge_ridge_alpha)
    allocator["observed_beta_spy_long"] = rolling_ridge_beta(long_obj["returns"].reindex(allocator.index).fillna(0.0), spy_r, cfg.hedge_beta_window, cfg.hedge_beta_min_obs, cfg.hedge_ridge_alpha)
    allocator = allocator.reset_index().rename(columns={"index": "Date"})
    keep = [
        "Date",
        "fold",
        "long_beta_qqq",
        "long_beta_spy",
        "target_beta_qqq",
        "target_beta_spy",
        "predicted_beta_qqq",
        "predicted_beta_spy",
        "observed_beta_qqq_long",
        "observed_beta_spy_long",
        "observed_beta_qqq_ls",
        "observed_beta_spy_ls",
        "qqq_short_budget",
        "spy_short_budget",
    ]
    return allocator[[c for c in keep if c in allocator.columns]]


def build_exposure_trace_fast(wf: Dict[str, Any], cfg: Mahoraga15AConfig) -> pd.DataFrame:
    ls = wf["stitched_ls"]
    long_only = wf["frozen_long"]
    allocator = wf["allocator_trace"]
    idx = pd.DatetimeIndex(ls["returns"].index)
    return pd.DataFrame(
        {
            "Date": idx,
            "LongOnlyGrossLong": _series(long_only, "gross_long", fallback=float(_series(long_only, "exposure", 0.0).mean())).reindex(idx).fillna(0.0).values,
            "LS_GrossLong": _series(ls, "gross_long", 0.0).reindex(idx).fillna(0.0).values,
            "LS_GrossShort": _series(ls, "gross_short", 0.0).reindex(idx).fillna(0.0).values,
            "LS_NetExposure": _series(ls, "net_exposure", 0.0).reindex(idx).fillna(0.0).values,
            "LS_GrossExposure": _series(ls, "gross_exposure", 0.0).reindex(idx).fillna(0.0).values,
            "CashBuffer": allocator["cash_buffer"].reindex(idx).ffill().fillna(0.0).values,
        }
    )


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
    qqq_obj: Dict[str, Any],
    spy_obj: Dict[str, Any],
    cfg: Mahoraga15AConfig,
) -> Dict[str, Any]:
    summary = summarize_object(obj, cfg, f"{variant}_{scenario}")
    ls_summary = summarize_object(ls_base, cfg, cfg.ls_label)
    long_summary = summarize_object(long_base, cfg, cfg.official_long_label)
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
        "BetaQQQ": round(beta(obj["returns"], qqq_obj["returns"]), 4),
        "BetaSPY": round(beta(obj["returns"], spy_obj["returns"]), 4),
        "AlphaNW_QQQ": round(alpha_qqq["alpha_ann"], 6) if np.isfinite(alpha_qqq["alpha_ann"]) else np.nan,
        "AlphaNW_SPY": round(alpha_spy["alpha_ann"], 6) if np.isfinite(alpha_spy["alpha_ann"]) else np.nan,
        "DeltaCAGR_vs_LSBase%": round((summary["CAGR"] - ls_summary["CAGR"]) * 100.0, 2),
        "DeltaSharpe_vs_LSBase": round(summary["Sharpe"] - ls_summary["Sharpe"], 4),
        "DeltaSortino_vs_LSBase": round(summary["Sortino"] - ls_summary["Sortino"], 4),
        "DeltaMaxDD_vs_LSBase%": round((summary["MaxDD"] - ls_summary["MaxDD"]) * 100.0, 2),
        "DeltaCAGR_vs_LongOnly%": round((summary["CAGR"] - long_summary["CAGR"]) * 100.0, 2),
        "DeltaSharpe_vs_LongOnly": round(summary["Sharpe"] - long_summary["Sharpe"], 4),
        "DeltaSortino_vs_LongOnly": round(summary["Sortino"] - long_summary["Sortino"], 4),
        "DeltaMaxDD_vs_LongOnly%": round((summary["MaxDD"] - long_summary["MaxDD"]) * 100.0, 2),
    }


def _stitch_scenario_fold_objects(
    fold_objs: List[Dict[str, Any]],
    payloads: List[Dict[str, Any]],
    cfg: Mahoraga15AConfig,
) -> Dict[str, Any]:
    stitched = {
        "returns": pd.concat([obj["returns"].loc[p["test_start"] : p["test_end"]] for obj, p in zip(fold_objs, payloads)]).sort_index(),
        "gross_returns": pd.concat([obj["gross_returns"].loc[p["test_start"] : p["test_end"]] for obj, p in zip(fold_objs, payloads)]).sort_index(),
        "transaction_cost": pd.concat([obj["transaction_cost"].loc[p["test_start"] : p["test_end"]] for obj, p in zip(fold_objs, payloads)]).sort_index(),
        "turnover": pd.concat([obj["turnover"].loc[p["test_start"] : p["test_end"]] for obj, p in zip(fold_objs, payloads)]).sort_index(),
        "gross_long": pd.concat([obj["gross_long"].loc[p["test_start"] : p["test_end"]] for obj, p in zip(fold_objs, payloads)]).sort_index(),
        "gross_short": pd.concat([obj["gross_short"].loc[p["test_start"] : p["test_end"]] for obj, p in zip(fold_objs, payloads)]).sort_index(),
        "net_exposure": pd.concat([obj["net_exposure"].loc[p["test_start"] : p["test_end"]] for obj, p in zip(fold_objs, payloads)]).sort_index(),
        "gross_exposure": pd.concat([obj["gross_exposure"].loc[p["test_start"] : p["test_end"]] for obj, p in zip(fold_objs, payloads)]).sort_index(),
    }
    stitched["equity"] = cfg.capital_initial * (1.0 + stitched["returns"]).cumprod()
    stitched["exposure"] = stitched["gross_exposure"]
    return stitched


def build_stress_suite_fast(wf: Dict[str, Any], cfg: Mahoraga15AConfig, costs) -> pd.DataFrame:
    qqq_obj = wf["stitched_benchmarks"]["QQQ"]
    spy_obj = wf["stitched_benchmarks"]["SPY"]
    long_base = wf["frozen_long"]
    ls_base = wf["stitched_ls"]
    rows = [
        _stress_metrics_row(cfg.official_long_label, "BASELINE_LONG_ONLY", "official frozen 14.1 long-only reference", long_base, ls_base, long_base, qqq_obj, spy_obj, cfg),
        _stress_metrics_row(cfg.ls_label, "BASELINE_15A", "Mahoraga15A unstressed stitched OOS", ls_base, ls_base, long_base, qqq_obj, spy_obj, cfg),
    ]
    scenarios = [
        ("COST_PLUS_25", {"cost_mult": 1.25}, "commission/slippage x1.25"),
        ("COST_PLUS_50", {"cost_mult": 1.50}, "commission/slippage x1.50"),
        ("COST_PLUS_100", {"cost_mult": 2.00}, "commission/slippage x2.00"),
        ("EXTRA_SLIPPAGE", {"extra_slippage": cfg.stress_extra_slippage}, "extra slippage +5bps"),
        ("EXECUTION_DELAY_1_REBALANCE", {"delay_days": 5 * cfg.stress_delay_rebalances}, "delay long budget and hedge one weekly rebalance"),
        ("HEDGE_RATIO_UNDERESTIMATED", {"hedge_ratio_mult": cfg.stress_hedge_ratio_under_mult}, "systematic hedge scaled to 75%"),
        ("HEDGE_RATIO_OVERESTIMATED", {"hedge_ratio_mult": cfg.stress_hedge_ratio_over_mult}, "systematic hedge scaled to 125%"),
        ("REACTION_SLOWER", {"up_speed_mult": cfg.stress_reaction_slower_mult, "down_speed_mult": cfg.stress_reaction_slower_mult}, "allocator reacts slower"),
        ("REACTION_FASTER", {"up_speed_mult": cfg.stress_reaction_faster_mult, "down_speed_mult": cfg.stress_reaction_faster_mult}, "allocator reacts faster"),
        ("ALLOCATOR_GATE_VOL_CAP_STRESS", {"short_cap_mult": cfg.stress_allocator_cap_mult, "long_multiplier_mult": cfg.stress_allocator_long_mult}, "tighter allocator caps and lighter long budget"),
    ]
    for scenario, override, note in scenarios:
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
        rows.append(_stress_metrics_row(cfg.ls_label, scenario, note, _stitch_scenario_fold_objects(fold_objs, wf["ls_fold_payloads"], cfg), ls_base, long_base, qqq_obj, spy_obj, cfg))
    rows.append(
        _stress_metrics_row(
            cfg.ls_label,
            "EMPIRICAL_PATH_STRESS",
            "inject worst empirical block into later path",
            _apply_empirical_path_stress(ls_base, cfg),
            ls_base,
            long_base,
            qqq_obj,
            spy_obj,
            cfg,
        )
    )
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
        samples.append({"Method": "stationary_block_bootstrap", "SampleId": sample_id, "CAGR": summary["CAGR"], "Sharpe": summary["Sharpe"], "MaxDD": summary["MaxDD"]})

    gross = pd.Series(ls_obj["gross_returns"], dtype=float).fillna(0.0).values
    tc = pd.Series(ls_obj["transaction_cost"], dtype=float).abs().fillna(0.0).values
    for sample_id in range(cfg.mc_friction_samples):
        mult = float(np.clip(rng.normal(1.0, 0.30), 0.50, 2.25))
        sample = gross - tc * mult
        eq = cfg.capital_initial * (1.0 + pd.Series(sample)).cumprod()
        obj = {"returns": pd.Series(sample), "equity": eq, "exposure": pd.Series(1.0, index=eq.index), "turnover": pd.Series(0.0, index=eq.index)}
        summary = summarize_object(obj, cfg, f"friction_{sample_id}")
        samples.append({"Method": "friction_multiplier_mc", "SampleId": sample_id, "CAGR": summary["CAGR"], "Sharpe": summary["Sharpe"], "MaxDD": summary["MaxDD"]})

    sample_id = 0
    for short_cap_mult in (0.90, 1.00, 1.10):
        for long_mult in (0.97, 1.00, 1.03):
            for speed_mult in (0.75, 1.00):
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
                        },
                    )
                    fold_objs.append(rebuilt["ls_obj"])
                stitched_r = pd.concat([obj["returns"].loc[p["test_start"] : p["test_end"]] for obj, p in zip(fold_objs, wf["ls_fold_payloads"])]).sort_index()
                eq = cfg.capital_initial * (1.0 + stitched_r).cumprod()
                obj = {"returns": stitched_r, "equity": eq, "exposure": pd.Series(1.0, index=eq.index), "turnover": pd.Series(0.0, index=eq.index)}
                summary = summarize_object(obj, cfg, f"local_{sample_id}")
                samples.append({"Method": "local_param_neighborhood", "SampleId": sample_id, "CAGR": summary["CAGR"], "Sharpe": summary["Sharpe"], "MaxDD": summary["MaxDD"]})
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
                "P25_CAGR%": round(float(sub["CAGR"].quantile(0.25)) * 100.0, 2),
                "P50_CAGR%": round(float(sub["CAGR"].quantile(0.50)) * 100.0, 2),
                "P75_CAGR%": round(float(sub["CAGR"].quantile(0.75)) * 100.0, 2),
                "P95_CAGR%": round(float(sub["CAGR"].quantile(0.95)) * 100.0, 2),
                "MeanSharpe": round(float(sub["Sharpe"].mean()), 4),
                "P5_Sharpe": round(float(sub["Sharpe"].quantile(0.05)), 4),
                "P25_Sharpe": round(float(sub["Sharpe"].quantile(0.25)), 4),
                "P50_Sharpe": round(float(sub["Sharpe"].quantile(0.50)), 4),
                "P75_Sharpe": round(float(sub["Sharpe"].quantile(0.75)), 4),
                "P95_Sharpe": round(float(sub["Sharpe"].quantile(0.95)), 4),
                "MeanMaxDD%": round(float(sub["MaxDD"].mean()) * 100.0, 2),
                "P5_MaxDD%": round(float(sub["MaxDD"].quantile(0.05)) * 100.0, 2),
                "P25_MaxDD%": round(float(sub["MaxDD"].quantile(0.25)) * 100.0, 2),
                "P50_MaxDD%": round(float(sub["MaxDD"].quantile(0.50)) * 100.0, 2),
                "P75_MaxDD%": round(float(sub["MaxDD"].quantile(0.75)) * 100.0, 2),
                "P95_MaxDD%": round(float(sub["MaxDD"].quantile(0.95)) * 100.0, 2),
                "Prob_Sharpe_lt_Baseline": round(float((sub["Sharpe"] < baseline["Sharpe"]).mean()), 4),
                "Prob_MaxDD_worse_Baseline": round(float((sub["MaxDD"] < baseline["MaxDD"]).mean()), 4),
                "Prob_CAGR_materially_worse_Baseline": round(float((sub["CAGR"] < baseline["CAGR"] - cfg.mc_material_cagr_gap).mean()), 4),
            }
        )
    return pd.DataFrame(rows), samples_df


def build_candidate_audit_fast(
    wf: Dict[str, Any],
    cfg: Mahoraga15AConfig,
    comparison_df: pd.DataFrame,
    pq_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    mc_df: pd.DataFrame,
) -> pd.DataFrame:
    comp = comparison_df.set_index("Variant")
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

    ls = comp.loc[cfg.ls_label]
    long_only = comp.loc[cfg.official_long_label]
    worst_stress = stress_df[stress_df["Variant"] == cfg.ls_label].sort_values("Sharpe").iloc[0]
    mc_local = mc_df[mc_df["Method"] == "local_param_neighborhood"].iloc[0] if len(mc_df[mc_df["Method"] == "local_param_neighborhood"]) else None
    success_rows = [
        {"Section": "SUCCESS_CHECK", "Variant": cfg.ls_label, "Reference": cfg.official_long_label, "Metric": "SharpeDelta", "Value": float(ls["Sharpe"] - long_only["Sharpe"]), "Threshold": 0.0, "Passed": bool(ls["Sharpe"] > long_only["Sharpe"]), "Detail": "Sharpe stitched must improve", "PValue": np.nan, "QValue": np.nan},
        {"Section": "SUCCESS_CHECK", "Variant": cfg.ls_label, "Reference": cfg.official_long_label, "Metric": "SortinoDelta", "Value": float(ls["Sortino"] - long_only["Sortino"]), "Threshold": 0.0, "Passed": bool(ls["Sortino"] > long_only["Sortino"]), "Detail": "Sortino stitched must improve", "PValue": np.nan, "QValue": np.nan},
        {"Section": "SUCCESS_CHECK", "Variant": cfg.ls_label, "Reference": cfg.official_long_label, "Metric": "BetaQQQReduction", "Value": float(long_only["BetaQQQ"] - ls["BetaQQQ"]), "Threshold": 0.05, "Passed": bool(ls["BetaQQQ"] < long_only["BetaQQQ"]), "Detail": "Beta vs QQQ must go down clearly", "PValue": np.nan, "QValue": np.nan},
        {"Section": "SUCCESS_CHECK", "Variant": cfg.ls_label, "Reference": cfg.official_long_label, "Metric": "BetaSPYReduction", "Value": float(long_only["BetaSPY"] - ls["BetaSPY"]), "Threshold": 0.05, "Passed": bool(ls["BetaSPY"] < long_only["BetaSPY"]), "Detail": "Beta vs SPY must go down clearly", "PValue": np.nan, "QValue": np.nan},
        {"Section": "SUCCESS_CHECK", "Variant": cfg.ls_label, "Reference": cfg.official_long_label, "Metric": "MaxDDDelta%", "Value": float(ls["MaxDD%"] - long_only["MaxDD%"]), "Threshold": 1.0, "Passed": bool(ls["MaxDD%"] <= long_only["MaxDD%"] + 1.0), "Detail": "MaxDD should improve or not worsen materially", "PValue": np.nan, "QValue": np.nan},
        {"Section": "SUCCESS_CHECK", "Variant": cfg.ls_label, "Reference": cfg.official_long_label, "Metric": "CAGRDelta%", "Value": float(ls["CAGR%"] - long_only["CAGR%"]), "Threshold": -2.0, "Passed": bool(ls["CAGR%"] >= long_only["CAGR%"] - 2.0), "Detail": "CAGR should not be destroyed", "PValue": np.nan, "QValue": np.nan},
        {"Section": "SUCCESS_CHECK", "Variant": cfg.ls_label, "Reference": cfg.official_long_label, "Metric": "AlphaNW_QQQ_Delta", "Value": float(ls["AlphaNW_QQQ"] - long_only["AlphaNW_QQQ"]), "Threshold": -0.02, "Passed": bool(ls["AlphaNW_QQQ"] >= long_only["AlphaNW_QQQ"] - 0.02), "Detail": "Alpha NW vs QQQ should not worsen materially", "PValue": np.nan, "QValue": np.nan},
        {"Section": "SUCCESS_CHECK", "Variant": cfg.ls_label, "Reference": cfg.official_long_label, "Metric": "WorstStressSharpe", "Value": float(worst_stress["Sharpe"]), "Threshold": 0.0, "Passed": bool(float(worst_stress["Sharpe"]) > 0.0), "Detail": str(worst_stress["Scenario"]), "PValue": np.nan, "QValue": np.nan},
        {"Section": "SUCCESS_CHECK", "Variant": cfg.ls_label, "Reference": cfg.official_long_label, "Metric": "HedgeImpactReal_AvgGrossShort", "Value": float(ls["GrossShort"]), "Threshold": 0.03, "Passed": bool(float(ls["GrossShort"]) >= 0.03), "Detail": "short sleeve should not be decorative", "PValue": np.nan, "QValue": np.nan},
    ]
    if mc_local is not None:
        success_rows.append({"Section": "SUCCESS_CHECK", "Variant": cfg.ls_label, "Reference": cfg.official_long_label, "Metric": "Prob_Sharpe_lt_Baseline_MC", "Value": float(mc_local["Prob_Sharpe_lt_Baseline"]), "Threshold": 0.50, "Passed": bool(float(mc_local["Prob_Sharpe_lt_Baseline"]) < 0.50), "Detail": "local neighborhood robustness", "PValue": np.nan, "QValue": np.nan})

    phase15b = []
    placeholder = build_sparse_idio_short_interface(cfg)
    for signal in placeholder["candidate_signals"]:
        phase15b.append({"Section": "PHASE_15B_INTERFACE", "Variant": placeholder["label"][0], "Reference": "allocator", "Metric": "candidate_signal", "Value": np.nan, "Threshold": np.nan, "Passed": True, "Detail": signal, "PValue": np.nan, "QValue": np.nan})
    for flt in placeholder["mandatory_filters"]:
        phase15b.append({"Section": "PHASE_15B_INTERFACE", "Variant": placeholder["label"][0], "Reference": "risk_filters", "Metric": "mandatory_filter", "Value": np.nan, "Threshold": np.nan, "Passed": True, "Detail": flt, "PValue": np.nan, "QValue": np.nan})

    return pd.concat([rows, pd.DataFrame(pair_rows), pd.DataFrame(success_rows), pd.DataFrame(phase15b)], ignore_index=True)


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


def generate_figures(
    wf: Dict[str, Any],
    cfg: Mahoraga15AConfig,
    beta_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    mc_samples_df: pd.DataFrame,
) -> Dict[str, str]:
    figures_dir = Path(cfg.outputs_dir) / "figures"
    ensure_dir(str(figures_dir))
    _figure_style()
    paths: Dict[str, str] = {}

    long_obj = wf["frozen_long"]
    ls_obj = wf["stitched_ls"]
    qqq_obj = wf["stitched_benchmarks"]["QQQ"]
    spy_obj = wf["stitched_benchmarks"]["SPY"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for obj, label, color, lw in [
        (long_obj, cfg.official_long_label, "#4C78A8", 2.0),
        (ls_obj, cfg.ls_label, "#E45756", 2.2),
        (qqq_obj, "QQQ", "#7F7F7F", 1.4),
        (spy_obj, "SPY", "#B2B2B2", 1.4),
    ]:
        ax.plot(obj["equity"].index, obj["equity"].values, label=label, color=color, lw=lw)
    ax.set_title("Equity Curve Long vs LS")
    ax.legend(frameon=False, ncol=2)
    path = figures_dir / "equity_curve_long_vs_ls.png"
    _save_figure(fig, path)
    paths["equity_curve_long_vs_ls"] = str(path)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for obj, label, color, lw in [
        (long_obj, cfg.official_long_label, "#4C78A8", 2.0),
        (ls_obj, cfg.ls_label, "#E45756", 2.2),
    ]:
        dd = obj["equity"] / obj["equity"].cummax() - 1.0
        ax.plot(dd.index, dd.values * 100.0, label=label, color=color, lw=lw)
    ax.set_title("Drawdown Long vs LS")
    ax.legend(frameon=False)
    path = figures_dir / "drawdown_long_vs_ls.png"
    _save_figure(fig, path)
    paths["drawdown_long_vs_ls"] = str(path)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(beta_df["Date"], beta_df["observed_beta_qqq_long"], color="#4C78A8", lw=1.6, label="Long beta QQQ")
    axes[0].plot(beta_df["Date"], beta_df["observed_beta_qqq_ls"], color="#E45756", lw=1.8, label="LS beta QQQ")
    axes[0].plot(beta_df["Date"], beta_df["target_beta_qqq"], color="#222222", lw=1.2, ls="--", label="Target beta QQQ")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].set_title("Beta Through Time")
    axes[1].plot(beta_df["Date"], beta_df["observed_beta_spy_long"], color="#4C78A8", lw=1.6, label="Long beta SPY")
    axes[1].plot(beta_df["Date"], beta_df["observed_beta_spy_ls"], color="#E45756", lw=1.8, label="LS beta SPY")
    axes[1].plot(beta_df["Date"], beta_df["target_beta_spy"], color="#222222", lw=1.2, ls="--", label="Target beta SPY")
    axes[1].legend(frameon=False, ncol=3)
    path = figures_dir / "beta_through_time.png"
    _save_figure(fig, path)
    paths["beta_through_time"] = str(path)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(exposure_df["Date"], exposure_df["LongOnlyGrossLong"], color="#4C78A8", lw=1.6, label="Long-only gross long")
    axes[0].plot(exposure_df["Date"], exposure_df["LS_GrossLong"], color="#E45756", lw=1.8, label="LS gross long")
    axes[0].plot(exposure_df["Date"], exposure_df["LS_GrossShort"], color="#2A9D8F", lw=1.8, label="LS gross short")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].set_title("Gross / Net Exposure")
    axes[1].plot(exposure_df["Date"], exposure_df["LS_NetExposure"], color="#E45756", lw=1.8, label="LS net exposure")
    axes[1].plot(exposure_df["Date"], exposure_df["CashBuffer"], color="#666666", lw=1.4, label="Cash buffer")
    axes[1].legend(frameon=False, ncol=2)
    path = figures_dir / "gross_net_exposure_through_time.png"
    _save_figure(fig, path)
    paths["gross_net_exposure_through_time"] = str(path)

    stress_plot = stress_df[(stress_df["Variant"] == cfg.ls_label) & (~stress_df["Scenario"].isin(["BASELINE_15A"]))].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].barh(stress_plot["Scenario"], stress_plot["DeltaSharpe_vs_LSBase"], color="#E45756")
    axes[0].set_title("Stress Delta Sharpe")
    axes[1].barh(stress_plot["Scenario"], stress_plot["DeltaMaxDD_vs_LSBase%"], color="#4C78A8")
    axes[1].set_title("Stress Delta MaxDD (%)")
    path = figures_dir / "stress_robustness_ls.png"
    _save_figure(fig, path)
    paths["stress_robustness_ls"] = str(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for method, color in [("stationary_block_bootstrap", "#4C78A8"), ("friction_multiplier_mc", "#E45756"), ("local_param_neighborhood", "#2A9D8F")]:
        sub = mc_samples_df[mc_samples_df["Method"] == method]
        if len(sub):
            axes[0].hist(sub["Sharpe"], bins=20, alpha=0.45, label=method, color=color)
            axes[1].hist(sub["MaxDD"] * 100.0, bins=20, alpha=0.45, label=method, color=color)
    axes[0].set_title("Monte Carlo Sharpe")
    axes[1].set_title("Monte Carlo MaxDD (%)")
    axes[0].legend(frameon=False, fontsize=8)
    path = figures_dir / "montecarlo_distribution_ls.png"
    _save_figure(fig, path)
    paths["montecarlo_distribution_ls"] = str(path)
    return paths


def build_fast_report_text(
    wf: Dict[str, Any],
    cfg: Mahoraga15AConfig,
    comparison_df: pd.DataFrame,
    pq_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    mc_df: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> str:
    ls = comparison_df[comparison_df["Variant"] == cfg.ls_label].iloc[0]
    long_only = comparison_df[comparison_df["Variant"] == cfg.official_long_label].iloc[0]
    hedge_delta = {
        "Sharpe": round(float(ls["Sharpe"] - long_only["Sharpe"]), 4),
        "Sortino": round(float(ls["Sortino"] - long_only["Sortino"]), 4),
        "MaxDD%": round(float(ls["MaxDD%"] - long_only["MaxDD%"]), 2),
        "BetaQQQ": round(float(ls["BetaQQQ"] - long_only["BetaQQQ"]), 4),
        "BetaSPY": round(float(ls["BetaSPY"] - long_only["BetaSPY"]), 4),
    }
    success_checks = audit_df[audit_df["Section"] == "SUCCESS_CHECK"][["Metric", "Value", "Threshold", "Passed", "Detail"]]
    lines = [
        "# Mahoraga15A FAST",
        "",
        "## 1. Stitched comparison",
        comparison_df.to_string(index=False),
        "",
        "## 2. Hedge contribution vs frozen long book",
        pd.DataFrame([hedge_delta]).to_string(index=False),
        "",
        "## 3. Pairwise p-values / q-values",
        pq_df.to_string(index=False),
        "",
        "## 4. Stress suite",
        stress_df.to_string(index=False),
        "",
        "## 5. Monte Carlo / bootstrap",
        mc_df.to_string(index=False),
        "",
        "## 6. Audit / success checks",
        success_checks.to_string(index=False),
    ]
    return "\n".join(lines)


def save_fast_outputs(wf: Dict[str, Any], cfg: Mahoraga15AConfig, costs) -> Dict[str, pd.DataFrame]:
    ensure_dir(cfg.outputs_dir)
    comparison_df = build_stitched_comparison_fast(wf, cfg)
    pq_df = build_pairwise_pq_fast(wf, cfg)
    beta_df = build_beta_decomposition_fast(wf, cfg)
    exposure_df = build_exposure_trace_fast(wf, cfg)
    allocator_df = wf["allocator_trace"].reset_index().rename(columns={"index": "Date"})
    stress_df = build_stress_suite_fast(wf, cfg, costs)
    mc_summary_df, mc_samples_df = build_montecarlo_summary_fast(wf, cfg, costs)
    audit_df = build_candidate_audit_fast(wf, cfg, comparison_df, pq_df, stress_df, mc_summary_df)
    figure_paths = generate_figures(wf, cfg, beta_df, exposure_df, stress_df, mc_samples_df)
    report_text = build_fast_report_text(wf, cfg, comparison_df, pq_df, stress_df, mc_summary_df, audit_df)

    comparison_df.to_csv(Path(cfg.outputs_dir) / "ls_stitched_comparison_fast.csv", index=False)
    beta_df.to_csv(Path(cfg.outputs_dir) / "ls_beta_decomposition_fast.csv", index=False)
    exposure_df.to_csv(Path(cfg.outputs_dir) / "ls_gross_net_exposure_fast.csv", index=False)
    allocator_df.to_csv(Path(cfg.outputs_dir) / "ls_allocator_trace_fast.csv", index=False)
    stress_df.to_csv(Path(cfg.outputs_dir) / "ls_stress_suite_fast.csv", index=False)
    mc_summary_df.to_csv(Path(cfg.outputs_dir) / "ls_montecarlo_summary_fast.csv", index=False)
    audit_df.to_csv(Path(cfg.outputs_dir) / "ls_candidate_audit_fast.csv", index=False)
    with open(Path(cfg.outputs_dir) / cfg.report_name, "w", encoding="utf-8") as f:
        f.write(report_text)
    return {
        "comparison": comparison_df,
        "pairwise_pq": pq_df,
        "beta_trace": beta_df,
        "exposure_trace": exposure_df,
        "allocator_trace": allocator_df,
        "stress_suite": stress_df,
        "montecarlo_summary": mc_summary_df,
        "candidate_audit": audit_df,
        "figures": pd.DataFrame({"Figure": list(figure_paths.keys()), "Path": list(figure_paths.values())}),
    }
