from __future__ import annotations

import sys
from copy import deepcopy
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
_M14_DIR = _PARENT_DIR / "mahoraga14"
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_PARENT_DIR) not in sys.path:
    sys.path.append(str(_PARENT_DIR))
if str(_M14_DIR) not in sys.path:
    sys.path.append(str(_M14_DIR))

import mahoraga6_1 as m6

from dynamic_capital_allocator import build_allocator_controls, finalize_allocator_trace
from long_book_14_1 import build_frozen_long_book
from mahoraga15a_config import Mahoraga15AConfig
from mahoraga15a_utils import build_weight_backtest, stitch_objects
from portfolio_state import build_shared_state
from systematic_hedge_sleeve import build_raw_hedge_plan

_M14_BACKTEST_SPEC = importlib.util.spec_from_file_location("m14_backtest_executor", _M14_DIR / "backtest_executor.py")
if _M14_BACKTEST_SPEC is None or _M14_BACKTEST_SPEC.loader is None:
    raise ImportError("Unable to load Mahoraga14 backtest executor.")
_M14_BACKTEST = importlib.util.module_from_spec(_M14_BACKTEST_SPEC)
_M14_BACKTEST_SPEC.loader.exec_module(_M14_BACKTEST)
run_walk_forward_mahoraga14 = _M14_BACKTEST.run_walk_forward_mahoraga14


def _fold_meta(results: List[Dict[str, Any]]) -> List[Tuple[int, pd.Timestamp, pd.Timestamp]]:
    return [(int(r["fold"]), pd.Timestamp(r["test_start"]), pd.Timestamp(r["test_end"])) for r in results]


def _shift_series(series: pd.Series, periods: int) -> pd.Series:
    return pd.Series(series, dtype=float).shift(periods).fillna(0.0)


def _budget_multiplier(native_gross: pd.Series, target_budget: pd.Series) -> pd.Series:
    base = pd.Series(native_gross, dtype=float)
    target = pd.Series(target_budget, dtype=float).reindex(base.index).fillna(0.0)
    mult = target.divide(base.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return mult.clip(lower=0.0)


def _compose_scaled_long_object(
    long_book_fold: Dict[str, Any],
    budget_target: pd.Series,
    cfg: Mahoraga15AConfig,
    costs,
    label: str,
) -> Dict[str, Any]:
    idx = pd.DatetimeIndex(long_book_fold["weights"].index)
    asset_returns = long_book_fold["asset_returns"].reindex(idx).fillna(0.0)
    native_weights = long_book_fold["weights"].reindex(idx).fillna(0.0)
    native_gross = long_book_fold["gross_long"].reindex(idx).fillna(0.0)
    budget_multiplier = _budget_multiplier(native_gross, budget_target)
    scaled_weights = native_weights.mul(budget_multiplier, axis=0)
    obj = build_weight_backtest(scaled_weights, asset_returns[scaled_weights.columns], cfg, costs, label=label)
    obj["budget_multiplier"] = budget_multiplier
    obj["target_budget"] = pd.Series(budget_target, dtype=float).reindex(idx).fillna(0.0)
    obj["asset_returns"] = asset_returns
    return obj


def _recompute_allocator_from_sleeves(allocator_trace: pd.DataFrame) -> pd.DataFrame:
    out = allocator_trace.copy()
    out["crash_short_budget"] = out[["crash_qqq_short_budget", "crash_spy_short_budget"]].sum(axis=1)
    out["bear_short_budget"] = out[["bear_qqq_short_budget", "bear_spy_short_budget"]].sum(axis=1)
    out["qqq_short_budget"] = out["crash_qqq_short_budget"] + out["bear_qqq_short_budget"]
    out["spy_short_budget"] = out["crash_spy_short_budget"] + out["bear_spy_short_budget"]
    out["systematic_short_budget"] = out["crash_short_budget"] + out["bear_short_budget"]
    out["net_exposure"] = out["long_budget"] - out["systematic_short_budget"]
    out["gross_exposure"] = out["long_budget"] + out["systematic_short_budget"]
    out["cash_buffer"] = (1.0 - out["long_budget"] - out["systematic_short_budget"]).clip(lower=0.0)
    out["predicted_beta_qqq"] = out["long_beta_qqq"] * out["long_multiplier"] - (out["qqq_short_budget"] + out["spy_short_budget"] * out["spy_beta_qqq"])
    out["predicted_beta_spy"] = out["long_beta_spy"] * out["long_multiplier"] - (out["qqq_short_budget"] * out["qqq_beta_spy"] + out["spy_short_budget"])
    return out


def _build_ls_weights(
    long_book_fold: Dict[str, Any],
    allocator_trace: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = pd.DatetimeIndex(long_book_fold["weights"].index)
    long_budget = allocator_trace["long_budget"].reindex(idx).fillna(0.0)
    native_gross = long_book_fold["gross_long"].reindex(idx).fillna(0.0)
    long_multiplier = _budget_multiplier(native_gross, long_budget)
    long_weights = long_book_fold["weights"].reindex(idx).fillna(0.0).mul(long_multiplier, axis=0)

    base = pd.DataFrame(0.0, index=idx, columns=long_book_fold["asset_returns"].columns)
    base.loc[:, long_weights.columns] = long_weights.values

    crash_weights = pd.DataFrame(0.0, index=idx, columns=base.columns)
    crash_weights.loc[:, "QQQ"] = -allocator_trace["crash_qqq_short_budget"].reindex(idx).fillna(0.0).values
    crash_weights.loc[:, "SPY"] = -allocator_trace["crash_spy_short_budget"].reindex(idx).fillna(0.0).values

    bear_weights = pd.DataFrame(0.0, index=idx, columns=base.columns)
    bear_weights.loc[:, "QQQ"] = -allocator_trace["bear_qqq_short_budget"].reindex(idx).fillna(0.0).values
    bear_weights.loc[:, "SPY"] = -allocator_trace["bear_spy_short_budget"].reindex(idx).fillna(0.0).values

    total_weights = base.add(crash_weights, fill_value=0.0).add(bear_weights, fill_value=0.0)
    return long_weights, crash_weights, bear_weights, total_weights


def _compose_ls_object(
    long_book_fold: Dict[str, Any],
    allocator_trace: pd.DataFrame,
    cfg: Mahoraga15AConfig,
    costs,
    label: str,
) -> Dict[str, Any]:
    asset_returns = long_book_fold["asset_returns"].reindex(allocator_trace.index).fillna(0.0)
    long_weights, crash_weights, bear_weights, total_weights = _build_ls_weights(long_book_fold, allocator_trace)
    long_obj = build_weight_backtest(long_weights, asset_returns[long_weights.columns], cfg, costs, label=f"{label}_LONG")
    crash_obj = build_weight_backtest(crash_weights, asset_returns[crash_weights.columns], cfg, costs, label=f"{label}_CRASH")
    bear_obj = build_weight_backtest(bear_weights, asset_returns[bear_weights.columns], cfg, costs, label=f"{label}_BEAR")
    total_obj = build_weight_backtest(total_weights, asset_returns[total_weights.columns], cfg, costs, label=label)
    total_obj["long_gross_contribution"] = long_obj["gross_returns"]
    total_obj["crash_gross_contribution"] = crash_obj["gross_returns"]
    total_obj["bear_gross_contribution"] = bear_obj["gross_returns"]
    total_obj["short_gross_contribution"] = crash_obj["gross_returns"] + bear_obj["gross_returns"]
    total_obj["long_net_contribution"] = long_obj["returns"]
    total_obj["crash_net_contribution"] = crash_obj["returns"]
    total_obj["bear_net_contribution"] = bear_obj["returns"]
    total_obj["short_net_contribution"] = crash_obj["returns"] + bear_obj["returns"]
    total_obj["long_weights"] = long_weights
    total_obj["crash_short_weights"] = crash_weights
    total_obj["bear_short_weights"] = bear_weights
    total_obj["allocator_trace"] = allocator_trace
    total_obj["asset_returns"] = asset_returns
    for col in [
        "cash_buffer",
        "long_budget",
        "crash_short_budget",
        "bear_short_budget",
        "systematic_short_budget",
        "crash_qqq_short_budget",
        "crash_spy_short_budget",
        "bear_qqq_short_budget",
        "bear_spy_short_budget",
        "qqq_short_budget",
        "spy_short_budget",
    ]:
        total_obj[col] = allocator_trace[col].reindex(asset_returns.index).fillna(0.0)
    return total_obj


def _compose_delevered_control_object(
    long_book_fold: Dict[str, Any],
    allocator_trace: pd.DataFrame,
    cfg: Mahoraga15AConfig,
    costs,
) -> Dict[str, Any]:
    net_target = allocator_trace["net_exposure"].clip(lower=0.0)
    control = _compose_scaled_long_object(long_book_fold, net_target, cfg, costs, label=cfg.delevered_label)
    control["matched_net_exposure_target"] = net_target.reindex(control["returns"].index).fillna(0.0)
    return control


def _stitch_series_from_objects(
    fold_objects: List[Dict[str, Any]],
    payloads: List[Dict[str, Any]],
    key: str,
) -> pd.Series:
    parts = []
    for obj, payload in zip(fold_objects, payloads):
        if key not in obj:
            continue
        parts.append(pd.Series(obj[key], dtype=float).loc[payload["test_start"] : payload["test_end"]])
    return pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)


def rebuild_ls_fold(
    payload: Dict[str, Any],
    cfg: Mahoraga15AConfig,
    costs,
    override: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    override = override or {}
    controls = build_allocator_controls(payload["state_df"], cfg)
    if "short_cap_mult" in override:
        cap_mult = float(override["short_cap_mult"])
        controls["crash_cap_dynamic"] = controls["crash_cap_dynamic"].mul(cap_mult).clip(0.0, cfg.allocator_crash_budget_crisis_cap)
        controls["bear_cap_dynamic"] = controls["bear_cap_dynamic"].mul(cap_mult).clip(0.0, cfg.allocator_bear_budget_crisis_cap)
        controls["total_short_cap_dynamic"] = np.minimum(cfg.allocator_total_short_budget_cap, controls["crash_cap_dynamic"] + controls["bear_cap_dynamic"])
    if "long_multiplier_mult" in override:
        controls["long_multiplier_target"] = controls["long_multiplier_target"].mul(float(override["long_multiplier_mult"])).clip(
            cfg.allocator_long_multiplier_floor,
            cfg.allocator_long_multiplier_ceiling,
        )
    if "up_speed_mult" in override:
        speed_mult = float(override["up_speed_mult"])
        for col in ["crash_up_speed", "bear_up_speed"]:
            controls[col] = controls[col].mul(speed_mult).clip(0.03, 0.98)
    if "down_speed_mult" in override:
        speed_mult = float(override["down_speed_mult"])
        for col in ["crash_down_speed", "bear_down_speed"]:
            controls[col] = controls[col].mul(speed_mult).clip(0.02, 0.95)

    raw_hedge = build_raw_hedge_plan(payload["state_df"], controls, cfg)
    hedge_ratio_mult = float(override.get("hedge_ratio_mult", 1.0))
    if hedge_ratio_mult != 1.0:
        for col in [
            "raw_crash_short_qqq",
            "raw_crash_short_spy",
            "raw_bear_short_qqq",
            "raw_bear_short_spy",
            "raw_short_qqq",
            "raw_short_spy",
        ]:
            raw_hedge[col] = raw_hedge[col].mul(hedge_ratio_mult).clip(lower=0.0)
        raw_hedge["raw_crash_short_budget"] = raw_hedge["raw_crash_short_qqq"] + raw_hedge["raw_crash_short_spy"]
        raw_hedge["raw_bear_short_budget"] = raw_hedge["raw_bear_short_qqq"] + raw_hedge["raw_bear_short_spy"]
        raw_hedge["raw_short_budget"] = raw_hedge["raw_crash_short_budget"] + raw_hedge["raw_bear_short_budget"]

    allocator_trace = finalize_allocator_trace(payload["state_df"], raw_hedge, controls, cfg)

    delay_days = int(override.get("delay_days", 0))
    if delay_days != 0:
        for col in [
            "long_multiplier",
            "long_budget",
            "crash_short_budget",
            "bear_short_budget",
            "systematic_short_budget",
            "cash_buffer",
            "net_exposure",
            "gross_exposure",
            "crash_qqq_short_budget",
            "crash_spy_short_budget",
            "bear_qqq_short_budget",
            "bear_spy_short_budget",
            "qqq_short_budget",
            "spy_short_budget",
        ]:
            allocator_trace[col] = _shift_series(allocator_trace[col], delay_days)
        allocator_trace = _recompute_allocator_from_sleeves(allocator_trace)

    hedge_shift_days = int(override.get("hedge_shift_days", 0))
    crash_shift_days = int(override.get("crash_shift_days", hedge_shift_days))
    bear_shift_days = int(override.get("bear_shift_days", hedge_shift_days))
    if crash_shift_days != 0:
        for col in ["crash_qqq_short_budget", "crash_spy_short_budget"]:
            allocator_trace[col] = _shift_series(allocator_trace[col], crash_shift_days)
    if bear_shift_days != 0:
        for col in ["bear_qqq_short_budget", "bear_spy_short_budget"]:
            allocator_trace[col] = _shift_series(allocator_trace[col], bear_shift_days)
    if crash_shift_days != 0 or bear_shift_days != 0:
        allocator_trace = _recompute_allocator_from_sleeves(allocator_trace)

    ls_obj = _compose_ls_object(payload["long_book_fold"], allocator_trace, cfg, costs, payload["label"])
    delevered_obj = _compose_delevered_control_object(payload["long_book_fold"], allocator_trace, cfg, costs)
    return {
        **payload,
        "controls": controls,
        "raw_hedge": raw_hedge,
        "allocator_trace": allocator_trace,
        "ls_obj": ls_obj,
        "delevered_obj": delevered_obj,
    }


def _build_fold_payload(
    result: Dict[str, Any],
    long_book_fold: Dict[str, Any],
    cfg: Mahoraga15AConfig,
    costs,
) -> Dict[str, Any]:
    state_df = build_shared_state(result, long_book_fold, cfg)
    payload = {
        "fold": int(result["fold"]),
        "test_start": pd.Timestamp(result["test_start"]),
        "test_end": pd.Timestamp(result["test_end"]),
        "label": f"{cfg.ls_label}_{int(result['fold'])}",
        "result": result,
        "long_book_fold": long_book_fold,
        "state_df": state_df,
    }
    return rebuild_ls_fold(payload, cfg, costs)


def _concat_trace(payloads: List[Dict[str, Any]], key: str) -> pd.DataFrame:
    frames = []
    for payload in payloads:
        frame = payload[key].loc[payload["test_start"] : payload["test_end"]].copy()
        frame["fold"] = int(payload["fold"])
        frames.append(frame)
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()


def run_walk_forward_mahoraga15a(
    ohlcv: Dict[str, pd.DataFrame],
    cfg: Mahoraga15AConfig,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    wf14 = run_walk_forward_mahoraga14(ohlcv, cfg, costs, universe_schedule)
    stitched_long, long_fold_books, freeze_validation_df = build_frozen_long_book(wf14, cfg, costs)
    payloads = [
        _build_fold_payload(result, long_fold_book, cfg, costs)
        for result, long_fold_book in zip(wf14["results"], long_fold_books)
    ]
    fold_meta = _fold_meta(wf14["results"])
    ls_fold_objects = [payload["ls_obj"] for payload in payloads]
    delevered_fold_objects = [payload["delevered_obj"] for payload in payloads]

    stitched_ls = stitch_objects(ls_fold_objects, fold_meta, cfg, cfg.ls_label)
    for key in [
        "long_net_contribution",
        "crash_net_contribution",
        "bear_net_contribution",
        "short_net_contribution",
        "long_gross_contribution",
        "crash_gross_contribution",
        "bear_gross_contribution",
        "short_gross_contribution",
        "cash_buffer",
        "long_budget",
        "crash_short_budget",
        "bear_short_budget",
        "systematic_short_budget",
        "crash_qqq_short_budget",
        "crash_spy_short_budget",
        "bear_qqq_short_budget",
        "bear_spy_short_budget",
        "qqq_short_budget",
        "spy_short_budget",
    ]:
        stitched_ls[key] = _stitch_series_from_objects(ls_fold_objects, payloads, key)

    stitched_delevered = stitch_objects(delevered_fold_objects, fold_meta, cfg, cfg.delevered_label)
    stitched_delevered["matched_net_exposure_target"] = _stitch_series_from_objects(delevered_fold_objects, payloads, "matched_net_exposure_target")

    allocator_trace = _concat_trace(payloads, "allocator_trace")
    state_trace = _concat_trace(payloads, "state_df")
    raw_hedge_trace = _concat_trace(payloads, "raw_hedge")
    controls_trace = _concat_trace(payloads, "controls")

    return {
        "cfg": cfg,
        "costs": costs,
        "upstream_wf14": wf14,
        "frozen_long": stitched_long,
        "frozen_long_fold_books": long_fold_books,
        "freeze_validation_df": freeze_validation_df,
        "ls_fold_payloads": payloads,
        "ls_fold_objects": ls_fold_objects,
        "delevered_fold_objects": delevered_fold_objects,
        "stitched_delevered_control": stitched_delevered,
        "stitched_ls": stitched_ls,
        "stitched_legacy": wf14["stitched_legacy"],
        "stitched_benchmarks": wf14["stitched_benchmarks"],
        "allocator_trace": allocator_trace,
        "state_trace": state_trace,
        "raw_hedge_trace": raw_hedge_trace,
        "controls_trace": controls_trace,
        "ls_component_trace": pd.DataFrame(
            {
                "long_net_contribution": stitched_ls["long_net_contribution"],
                "crash_net_contribution": stitched_ls["crash_net_contribution"],
                "bear_net_contribution": stitched_ls["bear_net_contribution"],
                "short_net_contribution": stitched_ls["short_net_contribution"],
                "long_gross_contribution": stitched_ls["long_gross_contribution"],
                "crash_gross_contribution": stitched_ls["crash_gross_contribution"],
                "bear_gross_contribution": stitched_ls["bear_gross_contribution"],
                "short_gross_contribution": stitched_ls["short_gross_contribution"],
                "cash_buffer": stitched_ls["cash_buffer"],
                "long_budget": stitched_ls["long_budget"],
                "crash_short_budget": stitched_ls["crash_short_budget"],
                "bear_short_budget": stitched_ls["bear_short_budget"],
                "systematic_short_budget": stitched_ls["systematic_short_budget"],
            }
        ),
    }
