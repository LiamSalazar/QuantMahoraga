from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import mahoraga6_1 as m6
from base_alpha_engine import (
    backtest_from_1x_weights,
    blend_engine_paths,
    build_global_alpha_components,
    fit_base_alpha_model,
    precompute_engine_path,
    slice_alpha_components,
)
from bull_participation_layer import apply_bull_participation_layer, derive_participation_engine_params
from continuation_v2_model import apply_continuation_v2_model, fit_continuation_v2_model
from mahoraga14_config import Mahoraga14Config
from mahoraga14_universe import members_at_date, union_universe
from mahoraga14_utils import bhy_qvalues, iter_grid, paired_ttest_pvalue, time_split_index
from override_policy import build_override_weekly, iter_policy_candidates, weekly_to_daily_override
from path_structure_features import build_candidate_daily_context, build_market_path_context, build_weekly_path_dataset
from participation_allocator_v1 import apply_participation_allocator_v1, fit_participation_allocator_v1
from risk_backoff_layer import apply_risk_backoff_layer
from structural_defense_model import annotate_structural_labels, apply_structural_defense_model, fit_structural_defense_model
from transition_recovery_model import build_hawkes_transition_features


def _load_folds(cfg: Mahoraga14Config, ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    folds = pd.DataFrame(m6.build_contiguous_folds(cfg, pd.DatetimeIndex(ohlcv["close"].index)))
    return folds[folds["fold"].isin(cfg.mode_folds())].copy().sort_values("fold")


def _prepare_global_invariants(
    ohlcv: Dict[str, pd.DataFrame],
    cfg: Mahoraga14Config,
    universe_schedule: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    universe_master = union_universe(ohlcv, universe_schedule, list(cfg.universe_static))
    close = ohlcv["close"][universe_master].copy()
    high = ohlcv["high"][universe_master].copy()
    low = ohlcv["low"][universe_master].copy()
    rets = close.pct_change().fillna(0.0)
    idx = close.index
    qqq = m6.to_s(ohlcv["close"][cfg.bench_qqq].reindex(idx).ffill(), "QQQ")
    spy = m6.to_s(ohlcv["close"][cfg.bench_spy].reindex(idx).ffill(), "SPY")
    vix_series = None
    if cfg.bench_vix in ohlcv.get("close", pd.DataFrame()).columns:
        vix_series = m6.to_s(ohlcv["close"][cfg.bench_vix].reindex(idx).ffill(), "VIX")

    turb_scale = m6.compute_turbulence(close, ohlcv["volume"][universe_master], qqq, cfg)
    corr_rho, corr_scale_legacy, corr_state = m6.compute_corr_shield_series(
        rets,
        idx,
        cfg,
        universe_master,
        use_pit_universe=universe_schedule is not None and len(universe_schedule) > 0,
        universe_schedule=universe_schedule,
        vix=vix_series,
    )
    reb_dates = set(close.resample(cfg.rebalance_freq).last().index)
    return {
        "close": close,
        "high": high,
        "low": low,
        "rets": rets,
        "idx": idx,
        "qqq": qqq,
        "spy": spy,
        "turb_scale": turb_scale,
        "corr_rho": corr_rho,
        "corr_scale_legacy": corr_scale_legacy,
        "corr_state": corr_state,
        "reb_dates": reb_dates,
        "members_at": lambda dt: members_at_date(universe_schedule, dt, universe_master),
        "universe_master": universe_master,
    }


def _prepare_fold_cfg(
    cfg_base: Mahoraga14Config,
    global_pre: Dict[str, Any],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    universe_schedule: Optional[pd.DataFrame],
) -> Mahoraga14Config:
    cfg = deepcopy(cfg_base)
    dd_thr, vol_thr = m6.calibrate_crisis_thresholds(global_pre["qqq"], str(train_start.date()), str(train_end.date()), cfg)
    cfg.crisis_dd_thr = dd_thr
    cfg.crisis_vol_zscore_thr = vol_thr
    train_tickers = m6.get_training_universe(
        str(train_end.date()),
        universe_schedule,
        cfg.universe_static,
        list(global_pre["close"].columns),
    )
    close_univ = global_pre["close"][train_tickers]
    wt, wm, wr = m6.fit_ic_weights(close_univ, global_pre["qqq"].loc[train_start:train_end], cfg, str(train_start.date()), str(train_end.date()))
    cfg.w_trend, cfg.w_mom, cfg.w_rel = wt, wm, wr
    return cfg


def _build_fold_pre(global_pre: Dict[str, Any], cfg: Mahoraga14Config, end: pd.Timestamp) -> Dict[str, Any]:
    idx = global_pre["idx"][global_pre["idx"] <= end]
    qqq_slice = global_pre["qqq"].loc[idx]
    crisis_scale, crisis_state = m6.compute_crisis_gate(qqq_slice, cfg)
    return {
        "close": global_pre["close"].loc[idx],
        "high": global_pre["high"].loc[idx],
        "low": global_pre["low"].loc[idx],
        "rets": global_pre["rets"].loc[idx],
        "idx": idx,
        "qqq": qqq_slice,
        "spy": global_pre["spy"].loc[idx],
        "crisis_scale": crisis_scale.loc[idx],
        "crisis_state": crisis_state.loc[idx],
        "turb_scale": global_pre["turb_scale"].loc[idx],
        "corr_rho": global_pre["corr_rho"].loc[idx],
        "corr_scale_legacy": global_pre["corr_scale_legacy"].loc[idx],
        "corr_state": global_pre["corr_state"].loc[idx],
        "reb_dates": {dt for dt in global_pre["reb_dates"] if dt <= end},
        "members_at": global_pre["members_at"],
    }


def _build_legacy_baseline_bt(pre: Dict[str, Any], cfg: Mahoraga14Config, costs: m6.CostsConfig) -> Dict[str, Any]:
    score = m6.compute_scores(pre["close"], pre["qqq"], cfg)
    close = pre["close"]
    high = pre["high"]
    low = pre["low"]
    rets = pre["rets"]
    idx = pre["idx"]

    weights = pd.DataFrame(0.0, index=idx, columns=close.columns)
    last_w = pd.Series(0.0, index=close.columns)
    for dt in idx:
        if dt in pre["reb_dates"]:
            members = [m for m in pre["members_at"](dt) if m in close.columns]
            if members:
                row = score.loc[dt, members]
                chosen = [name for name in row.nlargest(cfg.top_k).index.tolist() if row.get(name, 0.0) > 0.0]
                if len(chosen) == 1:
                    last_w = pd.Series(0.0, index=close.columns)
                    last_w[chosen[0]] = 1.0
                elif len(chosen) >= 2:
                    hist = rets.loc[:dt, chosen].tail(cfg.hrp_window).dropna(how="any")
                    if len(hist) < 60:
                        hist = rets.loc[:dt, chosen].dropna(how="any")
                    if len(hist):
                        w_hrp = m6.hrp_weights(hist).reindex(chosen, fill_value=0.0)
                    else:
                        w_hrp = pd.Series(1.0 / len(chosen), index=chosen)
                    w_hrp = w_hrp.clip(upper=cfg.weight_cap)
                    w_hrp = w_hrp / w_hrp.sum() if w_hrp.sum() > 0 else pd.Series(1.0 / len(chosen), index=chosen)
                    last_w = pd.Series(0.0, index=close.columns)
                    last_w[chosen] = w_hrp.reindex(chosen).values
                else:
                    last_w = pd.Series(0.0, index=close.columns)
            else:
                last_w = pd.Series(0.0, index=close.columns)
        weights.loc[dt] = last_w.values

    weights_after_stops, _ = m6.apply_chandelier(weights, close, high, low, cfg)
    weights_exec_1x = weights_after_stops.shift(1).fillna(0.0)
    ones = pd.Series(1.0, index=idx)
    bt = backtest_from_1x_weights(pre, weights_exec_1x, ones, ones, ones, cfg, costs, label="LEGACY_BASELINE")
    bt["scores"] = score
    bt["weights_exec_1x"] = weights_exec_1x
    return bt


def _build_base_candidate_bt(
    pre: Dict[str, Any],
    engine_cache: Dict[str, Any],
    cfg: Mahoraga14Config,
    costs: m6.CostsConfig,
    label: str,
) -> Dict[str, Any]:
    ones = pd.Series(1.0, index=pre["idx"])
    bt = backtest_from_1x_weights(pre, engine_cache["weights_exec_1x"], ones, ones, ones, cfg, costs, label=label)
    bt["scores"] = engine_cache["scores"]
    bt["stop_active_share"] = engine_cache["stop_active_share"]
    bt["new_stop_share"] = engine_cache["new_stop_share"]
    bt["weights_exec_1x"] = engine_cache["weights_exec_1x"]
    return bt


def _summarize_window(bt: Dict[str, Any], start: pd.Timestamp, end: pd.Timestamp, cfg: Mahoraga14Config, label: str) -> Dict[str, float]:
    ret = bt["returns_net"].loc[start:end]
    eq = cfg.capital_initial * (1.0 + ret).cumprod()
    exp = bt["exposure"].loc[start:end]
    turnover = bt["turnover"].loc[start:end]
    return m6.summarize(ret, eq, exp, turnover, cfg, label)


def _candidate_metrics(
    main_bt: Dict[str, Any],
    base_bt: Dict[str, Any],
    legacy_bt: Dict[str, Any],
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    override_daily: pd.DataFrame,
    fold_n: int,
    cfg: Mahoraga14Config,
) -> Dict[str, float]:
    s_legacy = _summarize_window(legacy_bt, val_start, val_end, cfg, "VAL_LEGACY")
    s_base = _summarize_window(base_bt, val_start, val_end, cfg, "VAL_BASE")
    s_main = _summarize_window(main_bt, val_start, val_end, cfg, "VAL_MAIN")

    base_r = base_bt["returns_net"].loc[val_start:val_end]
    legacy_r = legacy_bt["returns_net"].loc[val_start:val_end]
    main_r = main_bt["returns_net"].loc[val_start:val_end]

    base_vs_legacy_sharpe = s_base["Sharpe"] - s_legacy["Sharpe"]
    base_vs_legacy_cagr = s_base["CAGR"] - s_legacy["CAGR"]
    base_vs_legacy_dd = abs(s_legacy["MaxDD"]) - abs(s_base["MaxDD"])
    main_vs_base_sharpe = s_main["Sharpe"] - s_base["Sharpe"]
    main_vs_base_cagr = s_main["CAGR"] - s_base["CAGR"]
    main_vs_base_dd = abs(s_base["MaxDD"]) - abs(s_main["MaxDD"])
    override_rate = float(override_daily["is_override"].loc[main_r.index].mean())
    structural_rate = float(override_daily["is_structural_override"].loc[main_r.index].mean())
    mean_defense_blend = float(override_daily["defense_blend"].loc[main_r.index].mean())

    if fold_n in cfg.ceiling_folds:
        utility = (
            1.80 * base_vs_legacy_sharpe
            + 0.40 * base_vs_legacy_cagr
            + 0.28 * base_vs_legacy_dd
            + 0.24 * main_vs_base_sharpe
            + 0.06 * main_vs_base_cagr
            + 0.08 * main_vs_base_dd
            - 4.00 * max(0.0, -base_vs_legacy_sharpe - 0.01)
            - 8.00 * max(0.0, -main_vs_base_sharpe)
            - 1.40 * max(0.0, override_rate - cfg.ceiling_override_rate_cap)
            - 0.45 * max(0.0, structural_rate - 0.14)
            - 0.20 * mean_defense_blend
        )
    else:
        utility = (
            0.70 * base_vs_legacy_sharpe
            + 0.25 * base_vs_legacy_cagr
            + 0.20 * base_vs_legacy_dd
            + 1.95 * main_vs_base_sharpe
            + 0.55 * main_vs_base_cagr
            + 0.35 * main_vs_base_dd
            - 1.50 * max(0.0, -main_vs_base_sharpe - 0.01)
            - 0.45 * max(0.0, override_rate - cfg.floor_override_rate_cap)
            - 0.30 * max(0.0, structural_rate - 0.28)
        )

    return {
        "utility": float(utility),
        "val_legacy_sharpe": float(s_legacy["Sharpe"]),
        "val_base_sharpe": float(s_base["Sharpe"]),
        "val_main_sharpe": float(s_main["Sharpe"]),
        "val_base_vs_legacy_sharpe": float(base_vs_legacy_sharpe),
        "val_main_vs_base_sharpe": float(main_vs_base_sharpe),
        "val_main_vs_legacy_sharpe": float(s_main["Sharpe"] - s_legacy["Sharpe"]),
        "val_base_cagr": float(s_base["CAGR"]),
        "val_main_cagr": float(s_main["CAGR"]),
        "val_base_vs_legacy_cagr": float(base_vs_legacy_cagr),
        "val_main_vs_base_cagr": float(main_vs_base_cagr),
        "val_base_vs_legacy_maxdd_improve": float(base_vs_legacy_dd),
        "val_main_vs_base_maxdd_improve": float(main_vs_base_dd),
        "base_vs_legacy_val_pvalue": float(paired_ttest_pvalue(base_r - legacy_r, alternative="greater")),
        "main_vs_base_val_pvalue": float(paired_ttest_pvalue(main_r - base_r, alternative="greater")),
        "main_vs_legacy_val_pvalue": float(paired_ttest_pvalue(main_r - legacy_r, alternative="greater")),
        "override_rate": override_rate,
        "structural_rate": structural_rate,
        "mean_defense_blend": mean_defense_blend,
    }


def _select_candidate(calib_df: pd.DataFrame, fold_n: int, cfg: Mahoraga14Config) -> Dict[str, Any]:
    if len(calib_df) == 0:
        return {}
    if fold_n in cfg.ceiling_folds:
        admissible = calib_df[
            (calib_df["val_base_vs_legacy_sharpe"] >= cfg.ceiling_base_sharpe_tol)
            & (calib_df["val_main_vs_base_sharpe"] >= cfg.ceiling_main_sharpe_tol)
            & (calib_df["override_rate"] <= cfg.ceiling_override_rate_cap)
        ]
        if len(admissible):
            return admissible.iloc[0].to_dict()
        admissible = calib_df[
            (calib_df["val_main_vs_base_sharpe"] >= -0.01)
            & (calib_df["override_rate"] <= cfg.ceiling_override_rate_cap * 1.15)
        ]
        if len(admissible):
            return admissible.iloc[0].to_dict()
    else:
        admissible = calib_df[
            (calib_df["val_main_vs_base_sharpe"] >= cfg.floor_main_sharpe_floor)
            & (calib_df["override_rate"] <= cfg.floor_override_rate_cap)
        ]
        if len(admissible):
            return admissible.iloc[0].to_dict()
        admissible = calib_df[calib_df["val_main_vs_base_sharpe"] >= -0.01]
        if len(admissible):
            return admissible.iloc[0].to_dict()
    return calib_df.iloc[0].to_dict()


def _candidate_support(calib_df: pd.DataFrame, selected_id: float, cfg: Mahoraga14Config) -> pd.DataFrame:
    if len(calib_df) == 0:
        return pd.DataFrame()
    support = calib_df.head(cfg.top_support_candidates).copy()
    if selected_id not in set(support["candidate_id"].tolist()):
        extra = calib_df.loc[calib_df["candidate_id"] == selected_id]
        support = pd.concat([support, extra], axis=0, ignore_index=True)
    support = support.drop_duplicates(subset=["candidate_id"]).copy()
    support["support_rank"] = np.arange(1, len(support) + 1)
    support["is_selected"] = (support["candidate_id"] == selected_id).astype(int)
    keep = [
        "candidate_id",
        "support_rank",
        "is_selected",
        "utility",
        "val_base_vs_legacy_sharpe",
        "val_main_vs_base_sharpe",
        "override_rate",
        "structural_rate",
        "base_vs_legacy_val_pvalue",
        "base_vs_legacy_val_qvalue",
        "main_vs_base_val_pvalue",
        "main_vs_base_val_qvalue",
        "base_mix",
        "defense_mix",
        "base_beta_penalty",
        "defense_beta_penalty",
        "raw_rel_boost",
        "structural_enter_thr",
        "hawkes_weight",
        "structural_blend",
        "structural_gate",
        "structural_exp_cap",
        "StructuralModel",
    ]
    return support[[c for c in keep if c in support.columns]]


def _slice_bt_to_test_window(bt: Dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, pd.Series]:
    returns = bt["returns_net"].loc[start:end].copy()
    if len(returns) == 0:
        raise ValueError(f"Empty test window slice for {bt.get('label', 'UNKNOWN')} between {start} and {end}.")
    gross_returns = bt.get("returns_gross", pd.Series(dtype=float)).reindex(returns.index).fillna(0.0).copy()
    transaction_cost = bt.get("transaction_cost", pd.Series(dtype=float)).reindex(returns.index).fillna(0.0).copy()
    exposure = bt["exposure"].reindex(returns.index).fillna(0.0).copy()
    turnover = bt["turnover"].reindex(returns.index).fillna(0.0).copy()
    equity = bt.get("equity", pd.Series(dtype=float)).reindex(returns.index)
    return {
        "returns": returns,
        "gross_returns": gross_returns,
        "transaction_cost": transaction_cost,
        "exposure": exposure,
        "turnover": turnover,
        "equity": equity if len(equity) else pd.Series(dtype=float),
    }


def _stitch_from_getter(results: List[Dict[str, Any]], getter, cfg: Mahoraga14Config, label: str) -> Dict[str, Any]:
    slices: List[Dict[str, pd.Series]] = []
    trace_rows: List[Dict[str, Any]] = []
    for result in results:
        start = pd.Timestamp(result["test_start"])
        end = pd.Timestamp(result["test_end"])
        bt = getter(result)
        window = _slice_bt_to_test_window(bt, start, end)
        slices.append(window)
        trace_rows.append(
            {
                "Label": label,
                "Fold": int(result["fold"]),
                "SourceBacktest": bt.get("label", label),
                "WindowType": "TEST_ONLY",
                "RequestedStart": start,
                "RequestedEnd": end,
                "SliceStart": window["returns"].index.min(),
                "SliceEnd": window["returns"].index.max(),
                "Rows": int(len(window["returns"])),
            }
        )
    returns = pd.concat([window["returns"] for window in slices]).sort_index()
    if returns.index.has_duplicates:
        dupes = returns.index[returns.index.duplicated()].unique()
        raise ValueError(f"Duplicate periods detected in stitched {label}: {list(dupes[:5])}")
    gross_returns = pd.concat([window["gross_returns"] for window in slices]).sort_index().reindex(returns.index).fillna(0.0)
    transaction_cost = pd.concat([window["transaction_cost"] for window in slices]).sort_index().reindex(returns.index).fillna(0.0)
    exposure = pd.concat([window["exposure"] for window in slices]).sort_index().reindex(returns.index).fillna(0.0)
    turnover = pd.concat([window["turnover"] for window in slices]).sort_index().reindex(returns.index).fillna(0.0)
    equity = cfg.capital_initial * (1.0 + returns).cumprod()
    trace = pd.DataFrame(trace_rows).sort_values(["Fold", "SliceStart"]).reset_index(drop=True)
    return {
        "label": label,
        "returns": returns,
        "gross_returns": gross_returns,
        "transaction_cost": transaction_cost,
        "exposure": exposure,
        "turnover": turnover,
        "equity": equity,
        "trace": trace,
    }


def _stitch_benchmark(results: List[Dict[str, Any]], bench_key: str, cfg: Mahoraga14Config, label: str) -> Dict[str, Any]:
    rows = []
    trace_rows = []
    for result in results:
        start = pd.Timestamp(result["test_start"])
        end = pd.Timestamp(result["test_end"])
        bt = result["base_bt"]
        window = bt["bench"][bench_key].loc[start:end]
        rows.append(window)
        trace_rows.append(
            {
                "Label": label,
                "Fold": int(result["fold"]),
                "SourceBacktest": bt.get("label", label),
                "WindowType": "TEST_ONLY",
                "RequestedStart": start,
                "RequestedEnd": end,
                "SliceStart": window.index.min() if len(window) else pd.NaT,
                "SliceEnd": window.index.max() if len(window) else pd.NaT,
                "Rows": int(len(window)),
            }
        )
    returns = pd.concat(rows).sort_index() if rows else pd.Series(dtype=float)
    if len(returns) and returns.index.has_duplicates:
        dupes = returns.index[returns.index.duplicated()].unique()
        raise ValueError(f"Duplicate benchmark periods detected in stitched {label}: {list(dupes[:5])}")
    equity = cfg.capital_initial * (1.0 + returns).cumprod() if len(returns) else pd.Series(dtype=float)
    exposure = pd.Series(1.0, index=returns.index, dtype=float)
    turnover = pd.Series(0.0, index=returns.index, dtype=float)
    trace = pd.DataFrame(trace_rows).sort_values(["Fold", "SliceStart"]).reset_index(drop=True)
    return {
        "label": label,
        "returns": returns,
        "gross_returns": returns.copy(),
        "transaction_cost": pd.Series(0.0, index=returns.index, dtype=float),
        "equity": equity,
        "exposure": exposure,
        "turnover": turnover,
        "trace": trace,
    }


def _stitch(results: List[Dict[str, Any]], key: str, cfg: Mahoraga14Config, label: str) -> Dict[str, Any]:
    return _stitch_from_getter(results, lambda result: result[key], cfg, label)


def _stitch_override_daily(results: List[Dict[str, Any]], variant_key: str) -> pd.DataFrame:
    windows: List[pd.DataFrame] = []
    for result in results:
        start = pd.Timestamp(result["test_start"])
        end = pd.Timestamp(result["test_end"])
        window = result["variant_runs"][variant_key]["override_daily"].loc[start:end].copy()
        window["fold"] = int(result["fold"])
        windows.append(window)
    out = pd.concat(windows).sort_index() if windows else pd.DataFrame()
    if len(out) and out.index.has_duplicates:
        dupes = out.index[out.index.duplicated()].unique()
        raise ValueError(f"Duplicate override windows detected for {variant_key}: {list(dupes[:5])}")
    return out


def _neutralize_pre_test_override(override_daily: pd.DataFrame, test_start: pd.Timestamp) -> pd.DataFrame:
    out = override_daily.copy()
    pre_test_mask = out.index < test_start
    if pre_test_mask.any():
        out.loc[pre_test_mask, "override_type"] = "BASELINE"
        out.loc[pre_test_mask, "override_detail"] = "BASELINE"
        out.loc[pre_test_mask, "defense_blend"] = 0.0
        out.loc[pre_test_mask, "gate_scale"] = 1.0
        out.loc[pre_test_mask, "vol_mult"] = 1.0
        out.loc[pre_test_mask, "exp_cap"] = 1.0
        out.loc[pre_test_mask, "is_override"] = 0.0
        out.loc[pre_test_mask, "is_structural_override"] = 0.0
        if "is_continuation_lift" in out.columns:
            out.loc[pre_test_mask, "is_continuation_lift"] = 0.0
        if "is_continuation_activation" in out.columns:
            out.loc[pre_test_mask, "is_continuation_activation"] = 0.0
    return out


def _weekly_to_daily_state(
    weekly_df: pd.DataFrame,
    idx: pd.DatetimeIndex,
    numeric_defaults: Optional[Dict[str, float]] = None,
    text_defaults: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    out = weekly_df.reindex(idx).ffill()
    for col, default in (numeric_defaults or {}).items():
        if col in out.columns:
            out[col] = pd.Series(out[col], index=idx, dtype=float).fillna(default)
    for col, default in (text_defaults or {}).items():
        if col in out.columns:
            out[col] = out[col].fillna(default)
    return out


def _run_variant_backtest(
    weekly_full: pd.DataFrame,
    variant_mode: str,
    variant_label: str,
    policy_params: Dict[str, float],
    continuation_info: Optional[Dict[str, Any]],
    pre: Dict[str, Any],
    base_cache_full: Dict[str, Any],
    defense_cache_full: Dict[str, Any],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    cfg_fold: Mahoraga14Config,
    costs: m6.CostsConfig,
    fold_n: int,
) -> Dict[str, Any]:
    override_weekly_full = build_override_weekly(
        weekly_full.loc[:test_end],
        policy_params,
        cfg_fold,
        variant=variant_mode,
        continuation_info=continuation_info,
    )
    override_weekly = override_weekly_full.loc[test_start:test_end].copy()
    override_daily = weekly_to_daily_override(override_weekly_full, pre["idx"], cfg_fold)
    override_daily = _neutralize_pre_test_override(override_daily, test_start)
    weights_exec_1x = blend_engine_paths(base_cache_full, defense_cache_full, override_daily["defense_blend"])
    bt = backtest_from_1x_weights(
        pre,
        weights_exec_1x,
        override_daily["gate_scale"],
        override_daily["vol_mult"],
        override_daily["exp_cap"],
        cfg_fold,
        costs,
        label=f"{variant_label}_{fold_n}",
    )
    return {
        "bt": bt,
        "override_weekly": override_weekly.assign(fold=fold_n),
        "override_daily": override_daily,
        "weights_exec_1x": weights_exec_1x,
    }


def _run_primary_participation_backtest(
    weekly_full: pd.DataFrame,
    allocator_fit: Dict[str, Any],
    continuation_info: Dict[str, Any],
    pre: Dict[str, Any],
    base_cache_full: Dict[str, Any],
    defense_cache_full: Dict[str, Any],
    participation_cache_full: Dict[str, Any],
    policy_params: Dict[str, float],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    cfg_fold: Mahoraga14Config,
    costs: m6.CostsConfig,
    fold_n: int,
    participation_engine_params: Dict[str, float],
) -> Dict[str, Any]:
    override_weekly_full = build_override_weekly(
        weekly_full.loc[:test_end],
        policy_params,
        cfg_fold,
        variant=cfg_fold.combo_variant_key,
        continuation_info=continuation_info,
    )
    override_weekly = override_weekly_full.loc[test_start:test_end].copy()
    override_daily = weekly_to_daily_override(override_weekly_full, pre["idx"], cfg_fold)
    override_daily = _neutralize_pre_test_override(override_daily, test_start)

    allocator_weekly_raw = apply_participation_allocator_v1(allocator_fit, weekly_full.loc[:test_end], cfg_fold)
    allocator_weekly = apply_risk_backoff_layer(allocator_weekly_raw, weekly_full.loc[:test_end], cfg_fold)
    allocator_weekly["fold"] = fold_n
    allocator_daily = _weekly_to_daily_state(
        allocator_weekly,
        pre["idx"],
        numeric_defaults={
            "long_budget_raw": float(cfg_fold.participation_long_budget_base),
            "long_budget": float(cfg_fold.participation_long_budget_base),
            "gate_scale_adjustment_raw": 1.0,
            "gate_scale_adjustment": 1.0,
            "vol_mult_adjustment_raw": 1.0,
            "vol_mult_adjustment": 1.0,
            "exp_cap_adjustment_raw": 1.0,
            "exp_cap_adjustment": 1.0,
            "leader_blend_raw": 0.0,
            "leader_blend": 0.0,
            "risk_backoff_score": 0.0,
            "risk_backoff_hard_guard": 0.0,
            "continuation_allocator_score": 0.0,
            "benchmark_strength_score": 0.0,
            "persistence_strength_score": 0.0,
            "breadth_health_score": 0.0,
            "fragility_score": 0.0,
            "benchmark_weakness_score": 0.0,
            "bull_score_raw": 0.0,
            "backoff_score_raw": 0.0,
            "underparticipation_score": 0.0,
        },
        text_defaults={
            "participation_state_raw": "NEUTRAL",
            "participation_state": "NEUTRAL",
        },
    )
    pre_test_mask = allocator_daily.index < test_start
    if pre_test_mask.any():
        allocator_daily.loc[pre_test_mask, "long_budget"] = float(cfg_fold.participation_long_budget_base)
        allocator_daily.loc[pre_test_mask, "gate_scale_adjustment"] = 1.0
        allocator_daily.loc[pre_test_mask, "vol_mult_adjustment"] = 1.0
        allocator_daily.loc[pre_test_mask, "exp_cap_adjustment"] = 1.0
        allocator_daily.loc[pre_test_mask, "leader_blend"] = 0.0
        allocator_daily.loc[pre_test_mask, "risk_backoff_score"] = 0.0
        allocator_daily.loc[pre_test_mask, "risk_backoff_hard_guard"] = 0.0
        allocator_daily.loc[pre_test_mask, "participation_state"] = "NEUTRAL"

    base_defense_weights = blend_engine_paths(base_cache_full, defense_cache_full, override_daily["defense_blend"])
    primary_weights_1x, bull_diag = apply_bull_participation_layer(
        base_defense_weights,
        participation_cache_full["weights_exec_1x"],
        allocator_daily,
        cfg_fold,
    )
    gate_scale = (override_daily["gate_scale"] * allocator_daily["gate_scale_adjustment"]).clip(0.0, cfg_fold.max_gate_scale)
    vol_mult = (override_daily["vol_mult"] * allocator_daily["vol_mult_adjustment"]).clip(lower=0.0)
    exp_cap = (override_daily["exp_cap"] * allocator_daily["exp_cap_adjustment"]).clip(0.0, cfg_fold.participation_exp_cap_max)
    bt = backtest_from_1x_weights(
        pre,
        primary_weights_1x,
        gate_scale,
        vol_mult,
        exp_cap,
        cfg_fold,
        costs,
        label=f"{cfg_fold.combo_variant_key}_{fold_n}",
    )
    return {
        "bt": bt,
        "override_weekly": override_weekly.assign(fold=fold_n),
        "override_daily": override_daily,
        "allocator_weekly": allocator_weekly,
        "allocator_daily": allocator_daily,
        "bull_diagnostics": bull_diag,
        "weights_exec_1x": primary_weights_1x,
        "base_defense_weights_exec_1x": base_defense_weights,
        "participation_weights_exec_1x": participation_cache_full["weights_exec_1x"],
        "participation_engine_params": participation_engine_params,
    }


def _prepare_weekly_candidate_frame(
    base_bt: Dict[str, Any],
    base_cache: Dict[str, Any],
    market_ctx: pd.DataFrame,
    cfg: Mahoraga14Config,
    train_end: pd.Timestamp,
    hawkes_thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    daily_ctx = build_candidate_daily_context(base_bt, base_cache, market_ctx)
    weekly = build_weekly_path_dataset(daily_ctx, cfg)
    hawkes, thresholds = build_hawkes_transition_features(weekly, cfg, thresholds=hawkes_thresholds)
    weekly = weekly.join(hawkes, how="left")
    weekly = annotate_structural_labels(weekly, train_end)
    return weekly, thresholds


def _run_single_fold(
    row: pd.Series,
    ohlcv: Dict[str, pd.DataFrame],
    global_pre: Dict[str, Any],
    cfg_base: Mahoraga14Config,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    fold_n = int(row["fold"])
    train_start = pd.Timestamp(row["train_start"])
    train_end = pd.Timestamp(row["train_end"])
    test_start = pd.Timestamp(row["test_start"])
    test_end = pd.Timestamp(row["test_end"])

    cfg_fold = _prepare_fold_cfg(cfg_base, global_pre, train_start, train_end, universe_schedule)
    pre = _build_fold_pre(global_pre, cfg_fold, test_end)
    pre_cal = _build_fold_pre(global_pre, cfg_fold, train_end)
    market_ctx_full = build_market_path_context(ohlcv, pre, cfg_fold)
    market_ctx_cal = build_market_path_context(ohlcv, pre_cal, cfg_fold)

    alpha_fit = fit_base_alpha_model(pre["close"], pre["qqq"], pre["spy"], cfg_fold, train_start, train_end)
    components_full = build_global_alpha_components(pre["close"], pre["qqq"], pre["spy"], alpha_fit["factor_spec"], cfg_fold)
    components_cal = slice_alpha_components(components_full, pre_cal["idx"])

    legacy_bt = _build_legacy_baseline_bt(pre, cfg_fold, costs)
    legacy_bt_cal = _build_legacy_baseline_bt(pre_cal, cfg_fold, costs)

    engine_candidates = list(iter_grid(cfg_fold.engine_grid()))
    engine_cache_full: Dict[Tuple[float, float, float], Dict[str, Any]] = {}
    engine_cache_cal: Dict[Tuple[float, float, float], Dict[str, Any]] = {}

    for eng in engine_candidates:
        for mix, beta_pen in [
            (eng["base_mix"], eng["base_beta_penalty"]),
            (eng["defense_mix"], eng["defense_beta_penalty"]),
        ]:
            key = (round(mix, 6), round(beta_pen, 6), round(eng["raw_rel_boost"], 6))
            if key not in engine_cache_full:
                engine_cache_full[key] = precompute_engine_path(
                    pre,
                    components_full,
                    alpha_fit,
                    mix,
                    beta_pen,
                    eng["raw_rel_boost"],
                    cfg_fold,
                )
            if key not in engine_cache_cal:
                engine_cache_cal[key] = precompute_engine_path(
                    pre_cal,
                    components_cal,
                    alpha_fit,
                    mix,
                    beta_pen,
                    eng["raw_rel_boost"],
                    cfg_fold,
                )

    policy_candidates = list(iter_policy_candidates(cfg_fold))
    leaderboard: List[Dict[str, Any]] = []

    for eng in engine_candidates:
        base_key = (round(eng["base_mix"], 6), round(eng["base_beta_penalty"], 6), round(eng["raw_rel_boost"], 6))
        defense_key = (round(eng["defense_mix"], 6), round(eng["defense_beta_penalty"], 6), round(eng["raw_rel_boost"], 6))
        base_cache_cal = engine_cache_cal[base_key]
        defense_cache_cal = engine_cache_cal[defense_key]

        base_bt_cal = _build_base_candidate_bt(pre_cal, base_cache_cal, cfg_fold, costs, label="BASE_ALPHA_V2_VAL")
        weekly_cal, _ = _prepare_weekly_candidate_frame(base_bt_cal, base_cache_cal, market_ctx_cal, cfg_fold, train_end)
        train_weekly = weekly_cal.loc[:train_end].copy()

        structural_fit = fit_structural_defense_model(train_weekly, cfg_fold, cfg_fold.outer_parallel)
        weekly_cal["structural_p"] = apply_structural_defense_model(structural_fit, weekly_cal)

        cut = time_split_index(train_weekly.index, cfg_fold.inner_val_frac, cfg_fold.min_train_weeks)
        val_start = train_weekly.index[cut]
        val_end = train_end

        for policy_params in policy_candidates:
            override_weekly = build_override_weekly(weekly_cal.loc[:train_end], policy_params, cfg_fold, variant=cfg_fold.main_variant_key)
            override_daily = weekly_to_daily_override(override_weekly, pre_cal["idx"], cfg_fold)
            weights_exec_1x = blend_engine_paths(base_cache_cal, defense_cache_cal, override_daily["defense_blend"])
            main_bt_cal = backtest_from_1x_weights(
                pre_cal,
                weights_exec_1x,
                override_daily["gate_scale"],
                override_daily["vol_mult"],
                override_daily["exp_cap"],
                cfg_fold,
                costs,
                label="M14_MAIN_VAL",
            )
            metrics = _candidate_metrics(main_bt_cal, base_bt_cal, legacy_bt_cal, val_start, val_end, override_daily, fold_n, cfg_fold)
            if fold_n in cfg_fold.ceiling_folds:
                metrics["utility"] -= 0.12 * max(0.0, float(eng["base_mix"]) - 0.18) / 0.12
                metrics["utility"] -= 0.03 * max(0.0, float(eng["raw_rel_boost"]) - 1.0)
            else:
                metrics["utility"] -= 0.04 * max(0.0, float(eng["base_mix"]) - 0.18) / 0.12
            leaderboard.append(
                {
                    **eng,
                    **policy_params,
                    **metrics,
                    "fold": fold_n,
                    "SelectionVariant": cfg_fold.main_variant_key,
                    "StructuralModel": structural_fit["name"],
                }
            )

    calib_df = pd.DataFrame(leaderboard).sort_values(
        ["utility", "val_main_vs_base_sharpe", "val_base_vs_legacy_sharpe"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    if len(calib_df):
        calib_df["candidate_id"] = np.arange(len(calib_df), dtype=int)
        calib_df["base_vs_legacy_val_qvalue"] = bhy_qvalues(calib_df["base_vs_legacy_val_pvalue"].values, alpha=cfg_fold.bhy_alpha)
        calib_df["main_vs_base_val_qvalue"] = bhy_qvalues(calib_df["main_vs_base_val_pvalue"].values, alpha=cfg_fold.bhy_alpha)
        calib_df["main_vs_legacy_val_qvalue"] = bhy_qvalues(calib_df["main_vs_legacy_val_pvalue"].values, alpha=cfg_fold.bhy_alpha)

    best = _select_candidate(calib_df, fold_n, cfg_fold) if len(calib_df) else {
        "candidate_id": -1,
        "base_mix": 0.28,
        "defense_mix": 0.45,
        "base_beta_penalty": 0.00,
        "defense_beta_penalty": 0.05,
        "raw_rel_boost": 1.00,
        "structural_enter_thr": 0.82,
        "hawkes_weight": 0.14,
        "structural_blend": 0.30,
        "structural_gate": 0.88,
        "structural_exp_cap": 0.80,
    }

    base_key = (round(float(best["base_mix"]), 6), round(float(best["base_beta_penalty"]), 6), round(float(best["raw_rel_boost"]), 6))
    defense_key = (round(float(best["defense_mix"]), 6), round(float(best["defense_beta_penalty"]), 6), round(float(best["raw_rel_boost"]), 6))
    base_cache_full = engine_cache_full[base_key]
    defense_cache_full = engine_cache_full[defense_key]
    base_cache_cal = engine_cache_cal[base_key]
    participation_engine_params = derive_participation_engine_params(best, cfg_fold)
    participation_key = (
        round(float(participation_engine_params["mix"]), 6),
        round(float(participation_engine_params["beta_penalty"]), 6),
        round(float(participation_engine_params["raw_rel_boost"]), 6),
    )
    if participation_key not in engine_cache_full:
        engine_cache_full[participation_key] = precompute_engine_path(
            pre,
            components_full,
            alpha_fit,
            participation_engine_params["mix"],
            participation_engine_params["beta_penalty"],
            participation_engine_params["raw_rel_boost"],
            cfg_fold,
        )
    participation_cache_full = engine_cache_full[participation_key]

    base_bt = _build_base_candidate_bt(pre, base_cache_full, cfg_fold, costs, label=cfg_fold.official_baseline_label)
    base_bt_cal = _build_base_candidate_bt(pre_cal, base_cache_cal, cfg_fold, costs, label=f"{cfg_fold.official_baseline_label}_CAL")

    weekly_cal, hawkes_thresholds = _prepare_weekly_candidate_frame(base_bt_cal, base_cache_cal, market_ctx_cal, cfg_fold, train_end)
    train_weekly = weekly_cal.loc[:train_end].copy()
    structural_fit = fit_structural_defense_model(train_weekly, cfg_fold, cfg_fold.outer_parallel)
    continuation_fit = fit_continuation_v2_model(train_weekly, cfg_fold, cfg_fold.outer_parallel)
    weekly_cal["structural_p"] = apply_structural_defense_model(structural_fit, weekly_cal)
    weekly_cal = weekly_cal.join(apply_continuation_v2_model(continuation_fit, weekly_cal), how="left")
    allocator_fit = fit_participation_allocator_v1(weekly_cal.loc[:train_end].copy(), cfg_fold)

    weekly_full, _ = _prepare_weekly_candidate_frame(base_bt, base_cache_full, market_ctx_full, cfg_fold, train_end, hawkes_thresholds=hawkes_thresholds)
    weekly_full["structural_p"] = apply_structural_defense_model(structural_fit, weekly_full)
    weekly_full = weekly_full.join(apply_continuation_v2_model(continuation_fit, weekly_full), how="left")

    policy_params = {k: float(best[k]) for k in cfg_fold.policy_grid().keys()}
    variant_runs = {
        cfg_fold.main_variant_key: _run_variant_backtest(
            weekly_full,
            variant_mode=cfg_fold.main_variant_key,
            variant_label=cfg_fold.main_variant_key,
            policy_params=policy_params,
            continuation_info=None,
            pre=pre,
            base_cache_full=base_cache_full,
            defense_cache_full=defense_cache_full,
            test_start=test_start,
            test_end=test_end,
            cfg_fold=cfg_fold,
            costs=costs,
            fold_n=fold_n,
        ),
        cfg_fold.continuation_variant_key: _run_variant_backtest(
            weekly_full,
            variant_mode=cfg_fold.continuation_variant_key,
            variant_label=cfg_fold.continuation_variant_key,
            policy_params=policy_params,
            continuation_info=continuation_fit,
            pre=pre,
            base_cache_full=base_cache_full,
            defense_cache_full=defense_cache_full,
            test_start=test_start,
            test_end=test_end,
            cfg_fold=cfg_fold,
            costs=costs,
            fold_n=fold_n,
        ),
        cfg_fold.combo_variant_key: _run_primary_participation_backtest(
            weekly_full,
            allocator_fit=allocator_fit,
            continuation_info=continuation_fit,
            pre=pre,
            base_cache_full=base_cache_full,
            defense_cache_full=defense_cache_full,
            participation_cache_full=participation_cache_full,
            policy_params=policy_params,
            test_start=test_start,
            test_end=test_end,
            cfg_fold=cfg_fold,
            costs=costs,
            fold_n=fold_n,
            participation_engine_params=participation_engine_params,
        ),
    }
    variant_bts = {
        cfg_fold.official_baseline_label: base_bt,
        cfg_fold.main_variant_key: variant_runs[cfg_fold.main_variant_key]["bt"],
        cfg_fold.continuation_variant_key: variant_runs[cfg_fold.continuation_variant_key]["bt"],
        cfg_fold.combo_variant_key: variant_runs[cfg_fold.combo_variant_key]["bt"],
    }

    s_legacy = _summarize_window(legacy_bt, test_start, test_end, cfg_fold, f"LEGACY_{fold_n}")
    s_base = _summarize_window(base_bt, test_start, test_end, cfg_fold, f"BASE_{fold_n}")
    s_main = _summarize_window(variant_bts[cfg_fold.main_variant_key], test_start, test_end, cfg_fold, f"MAIN_{fold_n}")
    s_cont = _summarize_window(variant_bts[cfg_fold.continuation_variant_key], test_start, test_end, cfg_fold, f"CONT_{fold_n}")
    s_combo = _summarize_window(variant_bts[cfg_fold.combo_variant_key], test_start, test_end, cfg_fold, f"COMBO_{fold_n}")

    legacy_r = legacy_bt["returns_net"].loc[test_start:test_end]
    base_r = base_bt["returns_net"].loc[test_start:test_end]
    main_r = variant_bts[cfg_fold.main_variant_key]["returns_net"].loc[test_start:test_end]
    cont_r = variant_bts[cfg_fold.continuation_variant_key]["returns_net"].loc[test_start:test_end]
    combo_r = variant_bts[cfg_fold.combo_variant_key]["returns_net"].loc[test_start:test_end]

    main_override = variant_runs[cfg_fold.main_variant_key]["override_daily"].loc[test_start:test_end]
    cont_override = variant_runs[cfg_fold.continuation_variant_key]["override_daily"].loc[test_start:test_end]
    combo_override = variant_runs[cfg_fold.combo_variant_key]["override_daily"].loc[test_start:test_end]
    cont_override_weekly = variant_runs[cfg_fold.continuation_variant_key]["override_weekly"]
    combo_override_weekly = variant_runs[cfg_fold.combo_variant_key]["override_weekly"]
    fold_role = "CEILING" if fold_n in cfg_fold.ceiling_folds else "FLOOR"

    fold_row = {
        "fold": fold_n,
        "fold_role": fold_role,
        "train": f"{train_start.date()}→{train_end.date()}",
        "test": f"{test_start.date()}→{test_end.date()}",
        "LEGACY_CAGR%": round(s_legacy["CAGR"] * 100, 2),
        "LEGACY_Sharpe": round(s_legacy["Sharpe"], 4),
        "LEGACY_MaxDD%": round(s_legacy["MaxDD"] * 100, 2),
        "BASE_CAGR%": round(s_base["CAGR"] * 100, 2),
        "BASE_Sharpe": round(s_base["Sharpe"], 4),
        "BASE_MaxDD%": round(s_base["MaxDD"] * 100, 2),
        "MAIN_CAGR%": round(s_main["CAGR"] * 100, 2),
        "MAIN_Sharpe": round(s_main["Sharpe"], 4),
        "MAIN_MaxDD%": round(s_main["MaxDD"] * 100, 2),
        "CONT_CAGR%": round(s_cont["CAGR"] * 100, 2),
        "CONT_Sharpe": round(s_cont["Sharpe"], 4),
        "CONT_MaxDD%": round(s_cont["MaxDD"] * 100, 2),
        "COMBO_CAGR%": round(s_combo["CAGR"] * 100, 2),
        "COMBO_Sharpe": round(s_combo["Sharpe"], 4),
        "COMBO_MaxDD%": round(s_combo["MaxDD"] * 100, 2),
        "PRIMARY_CAGR%": round(s_combo["CAGR"] * 100, 2),
        "PRIMARY_Sharpe": round(s_combo["Sharpe"], 4),
        "PRIMARY_Sortino": round(s_combo["Sortino"], 4),
        "PRIMARY_MaxDD%": round(s_combo["MaxDD"] * 100, 2),
        "StructuralModel": structural_fit["name"],
        "ContinuationModel": continuation_fit["name"],
        "AllocatorModel": allocator_fit["name"],
        "ContinuationTriggerAUC": round(float(continuation_fit.get("validation_auc_trigger", 0.5)), 4),
        "ContinuationPressureAUC": round(float(continuation_fit.get("validation_auc_pressure", 0.5)), 4),
        "ContinuationBreakRiskAUC": round(float(continuation_fit.get("validation_auc_break_risk", 0.5)), 4),
        "ContinuationTriggerEntry": round(float(continuation_fit.get("trigger_enter", np.nan)), 4),
        "ContinuationPressureEntry": round(float(continuation_fit.get("pressure_enter", np.nan)), 4),
        "ContinuationBreakRiskCap": round(float(continuation_fit.get("break_risk_cap", np.nan)), 4),
        "MainOverrideRate": round(float(main_override["is_override"].mean()), 4),
        "MainStructuralRate": round(float(main_override["is_structural_override"].mean()), 4),
        "ContinuationLiftRate": round(float(cont_override["is_continuation_lift"].mean()), 4),
        "ContinuationActivationRate": round(float(cont_override_weekly["is_continuation_activation"].mean()), 4),
        "ComboOverrideRate": round(float(combo_override["is_override"].mean()), 4),
        "ComboStructuralRate": round(float(combo_override["is_structural_override"].mean()), 4),
        "ComboContinuationLiftRate": round(float(combo_override["is_continuation_lift"].mean()), 4),
        "ComboContinuationActivationRate": round(float(combo_override_weekly["is_continuation_activation"].mean()), 4),
        "PrimaryAvgLongBudget": round(float(variant_runs[cfg_fold.combo_variant_key]["allocator_daily"]["long_budget"].loc[test_start:test_end].mean()), 4),
        "PrimaryAvgLeaderBlend": round(float(variant_runs[cfg_fold.combo_variant_key]["allocator_daily"]["leader_blend"].loc[test_start:test_end].mean()), 4),
        "PrimaryAvgRiskBackoff": round(float(variant_runs[cfg_fold.combo_variant_key]["allocator_daily"]["risk_backoff_score"].loc[test_start:test_end].mean()), 4),
        "PrimaryCashRedeployed": round(float(variant_runs[cfg_fold.combo_variant_key]["bull_diagnostics"]["cash_redeployed"].loc[test_start:test_end].mean()), 4),
        "Base_vs_Legacy_Val_pvalue": round(float(best.get("base_vs_legacy_val_pvalue", 1.0)), 6),
        "Base_vs_Legacy_Val_qvalue": round(float(best.get("base_vs_legacy_val_qvalue", 1.0)), 6),
        "Main_vs_Base_Val_pvalue": round(float(best.get("main_vs_base_val_pvalue", 1.0)), 6),
        "Main_vs_Base_Val_qvalue": round(float(best.get("main_vs_base_val_qvalue", 1.0)), 6),
        "Base_vs_Legacy_Test_pvalue": round(float(paired_ttest_pvalue(base_r - legacy_r, alternative="greater")), 6),
        "Main_vs_Base_Test_pvalue": round(float(paired_ttest_pvalue(main_r - base_r, alternative="greater")), 6),
        "Continuation_vs_Base_Test_pvalue": round(float(paired_ttest_pvalue(cont_r - base_r, alternative="greater")), 6),
        "Combo_vs_Base_Test_pvalue": round(float(paired_ttest_pvalue(combo_r - base_r, alternative="greater")), 6),
        "Base_vs_Legacy_Test_qvalue": 1.0,
        "Main_vs_Base_Test_qvalue": 1.0,
        "Continuation_vs_Base_Test_qvalue": 1.0,
        "Combo_vs_Base_Test_qvalue": 1.0,
        "SelectedCandidateId": int(best.get("candidate_id", -1)),
    }

    selected_candidate = {
        "fold": fold_n,
        "fold_role": fold_role,
        "SelectionVariant": cfg_fold.main_variant_key,
        "SelectedCandidateId": int(best.get("candidate_id", -1)),
        "StructuralModel": structural_fit["name"],
        "ContinuationModel": continuation_fit["name"],
        "ContinuationTriggerAUC": continuation_fit.get("validation_auc_trigger", np.nan),
        "ContinuationPressureAUC": continuation_fit.get("validation_auc_pressure", np.nan),
        "ContinuationBreakRiskAUC": continuation_fit.get("validation_auc_break_risk", np.nan),
        "ContinuationTriggerEntry": continuation_fit.get("trigger_enter", np.nan),
        "ContinuationPressureEntry": continuation_fit.get("pressure_enter", np.nan),
        "ContinuationBreakRiskCap": continuation_fit.get("break_risk_cap", np.nan),
        "AllocatorModel": allocator_fit["name"],
        "ParticipationMix": participation_engine_params["mix"],
        "ParticipationBetaPenalty": participation_engine_params["beta_penalty"],
        "ParticipationRawRelBoost": participation_engine_params["raw_rel_boost"],
        **{k: best.get(k) for k in list(cfg_fold.engine_grid().keys()) + list(cfg_fold.policy_grid().keys())},
        "utility": best.get("utility", np.nan),
        "base_vs_legacy_val_pvalue": best.get("base_vs_legacy_val_pvalue", np.nan),
        "base_vs_legacy_val_qvalue": best.get("base_vs_legacy_val_qvalue", np.nan),
        "main_vs_base_val_pvalue": best.get("main_vs_base_val_pvalue", np.nan),
        "main_vs_base_val_qvalue": best.get("main_vs_base_val_qvalue", np.nan),
        "main_vs_legacy_val_pvalue": best.get("main_vs_legacy_val_pvalue", np.nan),
        "main_vs_legacy_val_qvalue": best.get("main_vs_legacy_val_qvalue", np.nan),
    }

    continuation_calibration = continuation_fit.get("calibration", pd.DataFrame()).copy()
    if len(continuation_calibration):
        continuation_calibration["fold"] = fold_n

    stress_pre = {
        "idx": pre["idx"],
        "rets": pre["rets"],
        "qqq": pre["qqq"],
        "spy": pre["spy"],
        "crisis_scale": pre["crisis_scale"],
        "turb_scale": pre["turb_scale"],
        "corr_rho": pre["corr_rho"],
        "corr_state": pre["corr_state"],
    }

    return {
        "fold": fold_n,
        "test_start": test_start,
        "test_end": test_end,
        "legacy_bt": legacy_bt,
        "base_bt": base_bt,
        "main_bt": variant_bts[cfg_fold.main_variant_key],
        "variant_bts": variant_bts,
        "variant_runs": variant_runs,
        "weekly_full": weekly_full.loc[test_start:test_end].copy(),
        "weekly_full_all": weekly_full.copy(),
        "fold_row": fold_row,
        "calibration_df": calib_df,
        "selected_candidate": selected_candidate,
        "selection_support": _candidate_support(calib_df, float(best.get("candidate_id", -1)), cfg_fold).assign(fold=fold_n),
        "continuation_calibration": continuation_calibration,
        "cfg_fold": cfg_fold,
        "policy_params": policy_params,
        "continuation_fit": continuation_fit,
        "allocator_fit": allocator_fit,
        "participation_engine_params": participation_engine_params,
        "stress_pre": stress_pre,
        "base_weights_exec_1x": base_cache_full["weights_exec_1x"],
        "defense_weights_exec_1x": defense_cache_full["weights_exec_1x"],
        "participation_weights_exec_1x": participation_cache_full["weights_exec_1x"],
    }


def run_walk_forward_mahoraga14(
    ohlcv: Dict[str, pd.DataFrame],
    cfg: Mahoraga14Config,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    folds = _load_folds(cfg, ohlcv)
    global_pre = _prepare_global_invariants(ohlcv, cfg, universe_schedule)
    tasks = [row for _, row in folds.iterrows()]
    use_parallel = cfg.outer_parallel and len(tasks) > 1
    if use_parallel:
        results = Parallel(n_jobs=min(cfg.max_outer_jobs, len(tasks)), backend=cfg.outer_backend)(
            delayed(_run_single_fold)(row, ohlcv, global_pre, cfg, costs, universe_schedule) for row in tasks
        )
    else:
        results = [_run_single_fold(row, ohlcv, global_pre, cfg, costs, universe_schedule) for row in tasks]

    results = sorted(results, key=lambda x: x["fold"])
    fold_df = pd.DataFrame([r["fold_row"] for r in results]).sort_values("fold")
    if len(fold_df):
        fold_df["Base_vs_Legacy_Test_qvalue"] = bhy_qvalues(fold_df["Base_vs_Legacy_Test_pvalue"].values, alpha=cfg.bhy_alpha)
        fold_df["Main_vs_Base_Test_qvalue"] = bhy_qvalues(fold_df["Main_vs_Base_Test_pvalue"].values, alpha=cfg.bhy_alpha)
        fold_df["Continuation_vs_Base_Test_qvalue"] = bhy_qvalues(fold_df["Continuation_vs_Base_Test_pvalue"].values, alpha=cfg.bhy_alpha)
        fold_df["Combo_vs_Base_Test_qvalue"] = bhy_qvalues(fold_df["Combo_vs_Base_Test_pvalue"].values, alpha=cfg.bhy_alpha)

    selected_df = pd.DataFrame([r["selected_candidate"] for r in results]).sort_values("fold") if results else pd.DataFrame()
    support_df = pd.concat([r["selection_support"] for r in results], axis=0, ignore_index=True) if results else pd.DataFrame()
    continuation_calibration_df = pd.concat([r["continuation_calibration"] for r in results], axis=0, ignore_index=True) if results else pd.DataFrame()

    stitched_legacy = _stitch(results, "legacy_bt", cfg, cfg.historical_benchmark_label)
    stitched_base = _stitch(results, "base_bt", cfg, cfg.official_baseline_label)
    stitched_variants = {
        cfg.official_baseline_label: stitched_base,
        cfg.main_variant_key: _stitch_from_getter(results, lambda result: result["variant_bts"][cfg.main_variant_key], cfg, cfg.main_variant_key),
        cfg.continuation_variant_key: _stitch_from_getter(results, lambda result: result["variant_bts"][cfg.continuation_variant_key], cfg, cfg.continuation_variant_key),
        cfg.combo_variant_key: _stitch_from_getter(results, lambda result: result["variant_bts"][cfg.combo_variant_key], cfg, cfg.combo_variant_key),
    }
    stitched_override_daily = {
        cfg.main_variant_key: _stitch_override_daily(results, cfg.main_variant_key),
        cfg.continuation_variant_key: _stitch_override_daily(results, cfg.continuation_variant_key),
        cfg.combo_variant_key: _stitch_override_daily(results, cfg.combo_variant_key),
    }
    stitched_benchmarks = {
        "QQQ": _stitch_benchmark(results, "QQQ_r", cfg, "QQQ"),
        "SPY": _stitch_benchmark(results, "SPY_r", cfg, "SPY"),
    }
    stitched_traces = {
        cfg.historical_benchmark_label: stitched_legacy["trace"].copy(),
        "QQQ": stitched_benchmarks["QQQ"]["trace"].copy(),
        "SPY": stitched_benchmarks["SPY"]["trace"].copy(),
        cfg.official_baseline_label: stitched_base["trace"].copy(),
        cfg.main_variant_key: stitched_variants[cfg.main_variant_key]["trace"].copy(),
        cfg.continuation_variant_key: stitched_variants[cfg.continuation_variant_key]["trace"].copy(),
        cfg.combo_variant_key: stitched_variants[cfg.combo_variant_key]["trace"].copy(),
    }
    stitched_model = stitched_variants[cfg.main_variant_key]
    return {
        "calendar_index": global_pre["idx"].copy(),
        "folds": folds.copy(),
        "results": results,
        "fold_df": fold_df,
        "selected_df": selected_df,
        "support_df": support_df,
        "continuation_calibration_df": continuation_calibration_df,
        "official_baseline": cfg.official_baseline_label,
        "historical_benchmark": cfg.historical_benchmark_label,
        "main_variant_key": cfg.main_variant_key,
        "continuation_variant_key": cfg.continuation_variant_key,
        "combo_variant_key": cfg.combo_variant_key,
        "primary_variant_key": cfg.primary_variant_key,
        "control_variant_key": cfg.control_variant_key,
        "stitched_legacy": stitched_legacy,
        "stitched_base": stitched_base,
        "stitched_model": stitched_model,
        "stitched_variants": stitched_variants,
        "stitched_override_daily": stitched_override_daily,
        "stitched_benchmarks": stitched_benchmarks,
        "stitched_traces": stitched_traces,
        "stitched_test_trace": stitched_model["trace"].copy(),
    }
