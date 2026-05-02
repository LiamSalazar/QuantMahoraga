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


def _delay_series(series: pd.Series, periods: int) -> pd.Series:
    return pd.Series(series, dtype=float).shift(periods).fillna(0.0)


def _build_ls_weights(long_book_fold: Dict[str, Any], allocator_trace: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = pd.DatetimeIndex(long_book_fold["weights"].index)
    long_weights = long_book_fold["weights"].reindex(idx).mul(allocator_trace["long_multiplier"].reindex(idx).fillna(0.0), axis=0)
    total = pd.DataFrame(0.0, index=idx, columns=long_book_fold["asset_returns"].columns)
    total.loc[:, long_weights.columns] = long_weights.values
    short_weights = pd.DataFrame(0.0, index=idx, columns=total.columns)
    short_weights.loc[:, "QQQ"] = -allocator_trace["qqq_short_budget"].reindex(idx).fillna(0.0).values
    short_weights.loc[:, "SPY"] = -allocator_trace["spy_short_budget"].reindex(idx).fillna(0.0).values
    total = total.add(short_weights, fill_value=0.0)
    return long_weights, short_weights, total


def _compose_ls_object(
    long_book_fold: Dict[str, Any],
    allocator_trace: pd.DataFrame,
    cfg: Mahoraga15AConfig,
    costs,
    label: str,
) -> Dict[str, Any]:
    asset_returns = long_book_fold["asset_returns"].reindex(allocator_trace.index).fillna(0.0)
    long_weights, short_weights, total_weights = _build_ls_weights(long_book_fold, allocator_trace)
    long_obj = build_weight_backtest(long_weights, asset_returns[long_weights.columns], cfg, costs, label=f"{label}_LONG")
    short_obj = build_weight_backtest(short_weights, asset_returns[short_weights.columns], cfg, costs, label=f"{label}_SHORT")
    total_obj = build_weight_backtest(total_weights, asset_returns[total_weights.columns], cfg, costs, label=label)
    total_obj["long_gross_contribution"] = long_obj["gross_returns"]
    total_obj["short_gross_contribution"] = short_obj["gross_returns"]
    total_obj["long_net_contribution"] = long_obj["returns"]
    total_obj["short_net_contribution"] = short_obj["returns"]
    total_obj["long_weights"] = long_weights
    total_obj["short_weights"] = short_weights
    total_obj["allocator_trace"] = allocator_trace
    total_obj["asset_returns"] = asset_returns
    return total_obj


def rebuild_ls_fold(
    payload: Dict[str, Any],
    cfg: Mahoraga15AConfig,
    costs,
    override: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    override = override or {}
    controls = build_allocator_controls(payload["state_df"], cfg)
    if "short_cap_mult" in override:
        controls["short_cap_dynamic"] = controls["short_cap_dynamic"].mul(float(override["short_cap_mult"])).clip(0.0, cfg.allocator_short_budget_crisis_cap)
    if "long_multiplier_mult" in override:
        controls["long_multiplier_target"] = controls["long_multiplier_target"].mul(float(override["long_multiplier_mult"])).clip(
            cfg.allocator_long_multiplier_floor,
            cfg.allocator_long_multiplier_ceiling,
        )
    if "up_speed_mult" in override:
        controls["short_up_speed"] = controls["short_up_speed"].mul(float(override["up_speed_mult"])).clip(0.03, 0.95)
    if "down_speed_mult" in override:
        controls["short_down_speed"] = controls["short_down_speed"].mul(float(override["down_speed_mult"])).clip(0.02, 0.80)

    raw_hedge = build_raw_hedge_plan(payload["state_df"], controls, cfg)
    hedge_ratio_mult = float(override.get("hedge_ratio_mult", 1.0))
    if hedge_ratio_mult != 1.0:
        raw_hedge["raw_short_qqq"] = raw_hedge["raw_short_qqq"].mul(hedge_ratio_mult).clip(lower=0.0)
        raw_hedge["raw_short_spy"] = raw_hedge["raw_short_spy"].mul(hedge_ratio_mult).clip(lower=0.0)
        raw_hedge["raw_short_budget"] = raw_hedge["raw_short_qqq"] + raw_hedge["raw_short_spy"]

    allocator_trace = finalize_allocator_trace(payload["state_df"], raw_hedge, controls, cfg)
    delay_days = int(override.get("delay_days", 0))
    if delay_days > 0:
        for col in ["long_multiplier", "long_budget", "systematic_short_budget", "cash_buffer", "qqq_short_budget", "spy_short_budget"]:
            allocator_trace[col] = _delay_series(allocator_trace[col], delay_days)
        allocator_trace["predicted_beta_qqq"] = _delay_series(allocator_trace["predicted_beta_qqq"], delay_days)
        allocator_trace["predicted_beta_spy"] = _delay_series(allocator_trace["predicted_beta_spy"], delay_days)

    ls_obj = _compose_ls_object(payload["long_book_fold"], allocator_trace, cfg, costs, payload["label"])
    payload_out = {
        **payload,
        "controls": controls,
        "raw_hedge": raw_hedge,
        "allocator_trace": allocator_trace,
        "ls_obj": ls_obj,
    }
    return payload_out


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
    ls_fold_objects = [payload["ls_obj"] for payload in payloads]
    stitched_ls = stitch_objects(ls_fold_objects, _fold_meta(wf14["results"]), cfg, cfg.ls_label)
    allocator_trace = _concat_trace(payloads, "allocator_trace")
    state_trace = _concat_trace(payloads, "state_df")
    raw_hedge_trace = _concat_trace(payloads, "raw_hedge")

    return {
        "cfg": cfg,
        "costs": costs,
        "upstream_wf14": wf14,
        "frozen_long": stitched_long,
        "frozen_long_fold_books": long_fold_books,
        "freeze_validation_df": freeze_validation_df,
        "ls_fold_payloads": payloads,
        "ls_fold_objects": ls_fold_objects,
        "stitched_ls": stitched_ls,
        "stitched_legacy": wf14["stitched_legacy"],
        "stitched_benchmarks": wf14["stitched_benchmarks"],
        "allocator_trace": allocator_trace,
        "state_trace": state_trace,
        "raw_hedge_trace": raw_hedge_trace,
    }
