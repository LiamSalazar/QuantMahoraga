from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from mahoraga15a_config import Mahoraga15AConfig
from mahoraga15a_utils import build_weight_backtest, stitch_objects


def build_asset_return_frame(result: Dict[str, Any], native_bt: Dict[str, Any]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(result["stress_pre"]["idx"])
    asset_rets = result["stress_pre"]["rets"].reindex(idx).fillna(0.0).copy()
    qqq_r = pd.Series(native_bt["bench"]["QQQ_r"], dtype=float).reindex(idx).fillna(0.0)
    spy_r = pd.Series(native_bt["bench"]["SPY_r"], dtype=float).reindex(idx).fillna(0.0)
    asset_rets["QQQ"] = qqq_r.values
    asset_rets["SPY"] = spy_r.values
    return asset_rets.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _official_variant_key(cfg: Mahoraga15AConfig, result: Dict[str, Any]) -> str:
    key = str(cfg.official_long_variant_key)
    if key not in result["variant_bts"]:
        raise KeyError(f"Official long variant '{key}' is not present in fold {result['fold']}.")
    return key


def validate_official_long_guardrail(wf14: Dict[str, Any], cfg: Mahoraga15AConfig) -> None:
    key = str(cfg.official_long_variant_key)
    if not cfg.require_official_long_guardrail:
        return
    for result in wf14["results"]:
        _official_variant_key(cfg, result)
    if key != str(wf14.get("continuation_variant_key", "")):
        raise ValueError(
            f"Mahoraga15A expects the frozen long book to be '{cfg.official_long_variant_key}', "
            f"but the upstream continuation key is '{wf14.get('continuation_variant_key')}'."
        )


def rebuild_fold_long_book(
    result: Dict[str, Any],
    cfg: Mahoraga15AConfig,
    costs,
) -> Dict[str, Any]:
    variant_key = _official_variant_key(cfg, result)
    bt = result["variant_bts"][variant_key]
    asset_rets = build_asset_return_frame(result, bt)
    weights = bt["weights_scaled"].reindex(asset_rets.index).fillna(0.0)
    rebuilt = build_weight_backtest(weights, asset_rets[weights.columns], cfg, costs, label=f"{cfg.official_long_label}_{int(result['fold'])}")
    rebuilt["native_bt"] = bt
    rebuilt["asset_returns"] = asset_rets
    rebuilt["test_start"] = pd.Timestamp(result["test_start"])
    rebuilt["test_end"] = pd.Timestamp(result["test_end"])
    rebuilt["fold"] = int(result["fold"])
    rebuilt["variant_key"] = variant_key
    return rebuilt


def build_frozen_long_book(
    wf14: Dict[str, Any],
    cfg: Mahoraga15AConfig,
    costs,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]:
    validate_official_long_guardrail(wf14, cfg)
    fold_books: List[Dict[str, Any]] = []
    validation_rows: List[Dict[str, Any]] = []
    fold_meta: List[Tuple[int, pd.Timestamp, pd.Timestamp]] = []

    for result in wf14["results"]:
        rebuilt = rebuild_fold_long_book(result, cfg, costs)
        native_bt = rebuilt["native_bt"]
        start = rebuilt["test_start"]
        end = rebuilt["test_end"]
        fold_meta.append((rebuilt["fold"], start, end))
        fold_books.append(rebuilt)

        native_slice = native_bt["returns_net"].loc[start:end]
        rebuilt_slice = rebuilt["returns"].loc[start:end]
        max_abs_diff = float((native_slice - rebuilt_slice).abs().max()) if len(native_slice) else 0.0
        validation_rows.append(
            {
                "Section": "FREEZE_VALIDATION",
                "Fold": rebuilt["fold"],
                "Item": "returns_rebuild_diff",
                "Value": max_abs_diff,
                "Passed": bool(max_abs_diff <= cfg.freeze_rebuild_tol),
                "Detail": f"variant={rebuilt['variant_key']}",
            }
        )

    stitched = stitch_objects(fold_books, fold_meta, cfg, cfg.official_long_label)
    upstream = wf14["stitched_variants"][cfg.official_long_variant_key]
    common_idx = stitched["returns"].index.intersection(upstream["returns"].index)
    stitched_diff = float((stitched["returns"].loc[common_idx] - upstream["returns"].loc[common_idx]).abs().max()) if len(common_idx) else 0.0
    validation_rows.append(
        {
            "Section": "FREEZE_VALIDATION",
            "Fold": 0,
            "Item": "stitched_returns_rebuild_diff",
            "Value": stitched_diff,
            "Passed": bool(stitched_diff <= cfg.freeze_rebuild_tol),
            "Detail": f"variant={cfg.official_long_variant_key}",
        }
    )
    return stitched, fold_books, pd.DataFrame(validation_rows)
