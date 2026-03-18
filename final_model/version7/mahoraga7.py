from __future__ import annotations

"""
Mahoraga 7
===========
Hawkes-inspired diagnostics and overlay over frozen Mahoraga 6.1.

Important honesty note
----------------------
This implementation uses a *discrete-time Hawkes-inspired exponential intensity*
on regime events. It is designed to be fast, testable, and explainable on top of
Mahoraga 6.1. It is not a full continuous-time multivariate MLE Hawkes package.

Variants
--------
- 7A: diagnostics only (event log, stress/recovery intensities, alignment vs gate)
- 7B: Hawkes overlay with bounded interventions:
      BASELINE / REL_TILT / DEFENSIVE_LIGHT / RECOVERY_OVERRIDE

Design choices for speed
------------------------
- Frozen Mahoraga 6.1 baseline (reuses walk_forward_folds.csv; no 72-combo sweep)
- Weekly decision cadence
- Precompute context table once
- Optional parallel outer folds
- Compact calibration grid

This file must live next to `mahoraga6_1.py` and the Mahoraga 6.1 outputs.
"""

import os
import json
from dataclasses import dataclass
from copy import deepcopy
from itertools import product as iproduct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import mahoraga6_1 as m6

try:
    from joblib import Parallel, delayed
    _JOBLIB = True
except Exception:
    _JOBLIB = False

_SCRIPT_DIR = Path(__file__).resolve().parent

DISCLAIMER = r"""
═══════════════════════════════════════════════════════════════════════════════
  MAHORAGA 7 — HAWKES-INSPIRED REGIME LAYER OVER MAHORAGA 6.1
───────────────────────────────────────────────────────────────────────────────
  • Mahoraga 6.1 remains frozen as the execution / risk baseline.
  • 7A adds diagnostics only: regime events + Hawkes-like stress/recovery intensity.
  • 7B adds a bounded overlay on top of 6.1.
  • No price prediction. No replacing the baseline.
  • Goal: react faster to clustered stress and faster to clustered recovery,
    especially in folds where the baseline is fragile.
═══════════════════════════════════════════════════════════════════════════════
"""


@dataclass
class Mahoraga7Config(m6.Mahoraga6Config):
    variant: str = "7B"  # "7A" or "7B"
    outputs_dir: str = "mahoraga7_outputs"
    plots_dir: str = "mahoraga7_plots"
    label: str = "MAHORAGA_7"

    # Baseline 6.1 artifacts
    baseline_outputs_dir: str = "mahoraga6_1_outputs"
    baseline_folds_csv: str = ""

    # Runtime controls
    run_mode: str = "FULL"   # FULL or FAST
    make_plots_flag: bool = False
    decision_freq: str = "W-FRI"
    outer_parallel: bool = True
    outer_backend: str = "auto"  # auto / threading / loky
    max_outer_jobs: int = 5

    # FAST mode folds (focus on difficult folds)
    fast_folds: Tuple[int, ...] = (3, 5)

    # Event quantile grids (chosen by inner validation, not by hand picking one value)
    stress_q_grid: Tuple[float, ...] = (0.85, 0.90)
    recovery_q_grid: Tuple[float, ...] = (0.80, 0.85)

    # Hawkes-like intensity parameters
    hawkes_decay_grid: Tuple[float, ...] = (0.65, 0.80)
    stress_scale_grid: Tuple[float, ...] = (0.8, 1.0)
    recovery_scale_grid: Tuple[float, ...] = (0.8, 1.0)

    # Overlay parameters (7B)
    defensive_scale_grid: Tuple[float, ...] = (0.80, 0.90)
    recovery_floor_grid: Tuple[float, ...] = (0.35, 0.50)
    rel_tilt_grid: Tuple[float, ...] = (0.55, 0.60)
    stress_trigger_q_grid: Tuple[float, ...] = (0.75, 0.85)
    recovery_trigger_q_grid: Tuple[float, ...] = (0.70, 0.80)

    # Inner validation split within train
    inner_val_frac: float = 0.30
    min_train_weeks: int = 80

    # Overlay targets
    target_intervention_rate: float = 0.18
    target_recovery_rate: float = 0.05

    # Recovery override only matters when crisis gate is active
    require_crisis_for_recovery_override: bool = True


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _find_baseline_folds_csv(cfg: Mahoraga7Config) -> str:
    candidates = []
    if cfg.baseline_folds_csv:
        candidates.append(Path(cfg.baseline_folds_csv))
    candidates += [
        _SCRIPT_DIR / cfg.baseline_outputs_dir / "walk_forward_folds.csv",
        _SCRIPT_DIR / "walk_forward_folds.csv",
        Path.cwd() / cfg.baseline_outputs_dir / "walk_forward_folds.csv",
        Path.cwd() / "walk_forward_folds.csv",
    ]
    for p in candidates:
        if Path(p).exists():
            return str(Path(p))
    raise FileNotFoundError(
        "No baseline walk_forward_folds.csv found. Set `baseline_folds_csv` or place Mahoraga 6.1 outputs next to this file."
    )


def _load_baseline_folds(cfg: Mahoraga7Config) -> pd.DataFrame:
    p = _find_baseline_folds_csv(cfg)
    df = pd.read_csv(p)
    req = [
        "fold", "best_weight_cap", "best_k_atr", "best_turb_zscore_thr",
        "best_turb_scale_min", "best_vol_target_ann"
    ]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in baseline fold file: {miss}")
    return df


def _parse_range(s: str) -> Tuple[str, str]:
    a, b = s.split("→")
    return a, b


def _get_fold_cfg(
    ohlcv: Dict[str, pd.DataFrame],
    base_cfg: Mahoraga7Config,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
    fold_row: pd.Series,
) -> Mahoraga7Config:
    cfg = deepcopy(base_cfg)
    cfg.weight_cap = float(fold_row["best_weight_cap"])
    cfg.k_atr = float(fold_row["best_k_atr"])
    cfg.turb_zscore_thr = float(fold_row["best_turb_zscore_thr"])
    cfg.turb_scale_min = float(fold_row["best_turb_scale_min"])
    cfg.vol_target_ann = float(fold_row["best_vol_target_ann"])

    train_start, train_end = _parse_range(fold_row["train"])
    qqq = m6.to_s(ohlcv["close"][cfg.bench_qqq].ffill(), "QQQ")
    dd_thr, vol_thr = m6.calibrate_crisis_thresholds(qqq, train_start, train_end, cfg)
    cfg.crisis_dd_thr = dd_thr
    cfg.crisis_vol_zscore_thr = vol_thr

    train_tickers = m6.get_training_universe(train_end, universe_schedule, cfg.universe_static, list(ohlcv["close"].columns))
    close_univ = ohlcv["close"][train_tickers]
    wt, wm, wr = m6.fit_ic_weights(close_univ, qqq.loc[train_start:train_end], cfg, train_start, train_end)
    cfg.w_trend, cfg.w_mom, cfg.w_rel = wt, wm, wr
    return cfg


def _weekly_dates(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    dates = pd.Series(1, index=idx).resample(freq).last().dropna().index
    return pd.DatetimeIndex([d for d in dates if d in idx])


def _avg_offdiag_corr(df: pd.DataFrame) -> float:
    if df is None or df.empty or df.shape[1] <= 1:
        return np.nan
    corr = df.corr().replace([np.inf, -np.inf], np.nan)
    if corr.shape[0] <= 1:
        return np.nan
    arr = corr.values.astype(float)
    mask = np.triu(np.ones_like(arr, dtype=bool), 1)
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else np.nan


def _component_scores(close: pd.DataFrame, qqq: pd.Series, cfg: Mahoraga7Config) -> Dict[str, pd.DataFrame]:
    idx = close.index
    qqq_ = m6.to_s(qqq, "QQQ").reindex(idx).ffill()
    out = {
        "trend": pd.DataFrame(index=idx, columns=close.columns, dtype=float),
        "mom": pd.DataFrame(index=idx, columns=close.columns, dtype=float),
        "rel": pd.DataFrame(index=idx, columns=close.columns, dtype=float),
    }
    for t in close.columns:
        cl = close[t].reindex(idx).ffill()
        out["trend"][t] = m6._trend(cl, cfg).fillna(0.0)
        out["mom"][t] = m6._mom(cl, cfg).fillna(0.0)
        out["rel"][t] = m6._rel(cl, qqq_, cfg).fillna(0.0)
    return out


def _build_context_table(
    ohlcv: Dict[str, pd.DataFrame],
    cfg: Mahoraga7Config,
    universe_schedule: Optional[pd.DataFrame],
) -> pd.DataFrame:
    close = ohlcv["close"].copy()
    qqq = m6.to_s(ohlcv["close"][cfg.bench_qqq].reindex(close.index).ffill(), "QQQ")
    vix = m6.to_s(ohlcv["close"][cfg.bench_vix].reindex(close.index).ffill(), "VIX") if cfg.bench_vix in ohlcv["close"].columns else pd.Series(np.nan, index=close.index)
    qqq_r = qqq.pct_change().fillna(0.0)
    dec_idx = _weekly_dates(close.index, cfg.decision_freq)

    # Build component scores once on the superset universe
    tickers = [c for c in close.columns if c not in {cfg.bench_qqq, cfg.bench_spy, cfg.bench_vix}]
    comp = _component_scores(close[tickers], qqq, cfg)
    rows = []
    for dt in dec_idx:
        if universe_schedule is not None:
            members = [t for t in m6.get_universe_at_date(universe_schedule, dt) if t in tickers]
        else:
            members = [t for t in cfg.universe_static if t in tickers]
        if len(members) == 0:
            continue
        sub21 = close[members].loc[:dt].tail(21).pct_change().dropna(how="all")
        sub63 = close[members].loc[:dt].tail(63).pct_change().dropna(how="all")
        xs5 = close[members].pct_change(5).loc[dt]
        xs21 = close[members].pct_change(21).loc[dt]
        breadth63 = (close[members].loc[dt] > close[members].shift(63).loc[dt]).mean()
        tr_row = comp["trend"].loc[dt, members].astype(float)
        mo_row = comp["mom"].loc[dt, members].astype(float)
        re_row = comp["rel"].loc[dt, members].astype(float)
        rel_top = re_row.nlargest(min(len(re_row), cfg.top_k))
        row = {
            "avg_corr_21": _avg_offdiag_corr(sub21),
            "avg_corr_63": _avg_offdiag_corr(sub63),
            "xs_disp_5d": float(xs5.std()) if xs5.notna().sum() > 1 else np.nan,
            "xs_disp_21d": float(xs21.std()) if xs21.notna().sum() > 1 else np.nan,
            "breadth_63d": float(breadth63) if np.isfinite(breadth63) else np.nan,
            "qqq_ret_5d": float(qqq.pct_change(5).loc[dt]) if dt in qqq.index else np.nan,
            "qqq_ret_21d": float(qqq.pct_change(21).loc[dt]) if dt in qqq.index else np.nan,
            "qqq_drawdown": float((qqq.loc[:dt] / qqq.loc[:dt].cummax() - 1.0).iloc[-1]),
            "qqq_vol_21": float(qqq_r.loc[:dt].tail(21).std() * np.sqrt(cfg.trading_days)),
            "vix_level": float(vix.loc[dt]) if dt in vix.index and np.isfinite(vix.loc[dt]) else np.nan,
            "vix_z_63": float(m6.safe_z(vix.ffill().fillna(0.0), 63).loc[dt]) if dt in vix.index else np.nan,
            "qqq_above_ema20": float(qqq.loc[dt] > qqq.ewm(span=20, adjust=False).mean().loc[dt]),
            "trend_mean": float(tr_row.mean()),
            "mom_mean": float(mo_row.mean()),
            "rel_mean": float(re_row.mean()),
            "rel_top_mean": float(rel_top.mean()) if len(rel_top) else np.nan,
            "rel_minus_mom": float(re_row.mean() - mo_row.mean()),
            "rel_minus_trend": float(re_row.mean() - tr_row.mean()),
            "n_members": float(len(members)),
        }
        rows.append((dt, row))
    feat = pd.DataFrame({d: r for d, r in rows}).T.sort_index()
    feat.index.name = "date"
    return feat


def _thresholds_from_train(train_feat: pd.DataFrame, stress_q: float, recovery_q: float) -> Dict[str, float]:
    qhi = lambda s: float(np.nanquantile(train_feat[s].dropna(), stress_q)) if train_feat[s].dropna().size else np.nan
    qlo = lambda s: float(np.nanquantile(train_feat[s].dropna(), 1.0 - stress_q)) if train_feat[s].dropna().size else np.nan
    rhi = lambda s: float(np.nanquantile(train_feat[s].dropna(), recovery_q)) if train_feat[s].dropna().size else np.nan
    rlo = lambda s: float(np.nanquantile(train_feat[s].dropna(), 1.0 - recovery_q)) if train_feat[s].dropna().size else np.nan
    return {
        "vix_spike": qhi("vix_level"),
        "corr_jump": qhi("avg_corr_21"),
        "breadth_break": qlo("breadth_63d"),
        "dd_break": qlo("qqq_drawdown"),
        "vol_spike": qhi("qqq_vol_21"),
        "rel_break": qlo("rel_top_mean"),
        "breadth_recovery": rhi("breadth_63d"),
        "corr_normalization": rlo("avg_corr_21"),
        "reclaim_ema": rhi("qqq_above_ema20"),
        "rel_recovery": rhi("rel_top_mean"),
        "ret_recovery": rhi("qqq_ret_5d"),
    }


def _build_event_table(feat: pd.DataFrame, thr: Dict[str, float]) -> pd.DataFrame:
    e = pd.DataFrame(index=feat.index)
    e["vix_spike"] = (feat["vix_level"] >= thr["vix_spike"]).astype(float)
    e["corr_jump"] = (feat["avg_corr_21"] >= thr["corr_jump"]).astype(float)
    e["breadth_break"] = (feat["breadth_63d"] <= thr["breadth_break"]).astype(float)
    e["dd_break"] = (feat["qqq_drawdown"] <= thr["dd_break"]).astype(float)
    e["vol_spike"] = (feat["qqq_vol_21"] >= thr["vol_spike"]).astype(float)
    e["rel_break"] = (feat["rel_top_mean"] <= thr["rel_break"]).astype(float)

    e["breadth_recovery"] = (feat["breadth_63d"] >= thr["breadth_recovery"]).astype(float)
    e["corr_normalization"] = (feat["avg_corr_21"] <= thr["corr_normalization"]).astype(float)
    e["reclaim_ema"] = (feat["qqq_above_ema20"] >= thr["reclaim_ema"]).astype(float)
    e["rel_recovery"] = (feat["rel_top_mean"] >= thr["rel_recovery"]).astype(float)
    e["ret_recovery"] = (feat["qqq_ret_5d"] >= thr["ret_recovery"]).astype(float)

    e["stress_events"] = e[["vix_spike", "corr_jump", "breadth_break", "dd_break", "vol_spike", "rel_break"]].sum(axis=1)
    e["recovery_events"] = e[["breadth_recovery", "corr_normalization", "reclaim_ema", "rel_recovery", "ret_recovery"]].sum(axis=1)
    return e


def _hawkes_intensity(events: pd.Series, decay: float, scale: float) -> pd.Series:
    vals = pd.Series(0.0, index=events.index, dtype=float)
    prev = 0.0
    base = float(events.mean()) if len(events) else 0.0
    for dt in events.index:
        prev = base + decay * prev + scale * float(events.shift(1).fillna(0.0).loc[dt])
        vals.loc[dt] = max(0.0, prev)
    return vals


def _build_hawkes_signals(feat: pd.DataFrame, stress_q: float, recovery_q: float, decay: float, stress_scale: float, recovery_scale: float, train_slice: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    thr = _thresholds_from_train(train_slice, stress_q, recovery_q)
    ev = _build_event_table(feat, thr)
    stress_int = _hawkes_intensity(ev["stress_events"], decay=decay, scale=stress_scale)
    recovery_int = _hawkes_intensity(ev["recovery_events"], decay=decay, scale=recovery_scale)
    out = pd.concat([feat, ev], axis=1)
    out["stress_intensity"] = stress_int
    out["recovery_intensity"] = recovery_int
    out["intensity_spread"] = stress_int - recovery_int
    return out, thr


def _make_rel_tilt_cfg(base_cfg: Mahoraga7Config, rel_tilt: float) -> Mahoraga7Config:
    cfg = deepcopy(base_cfg)
    bw = np.array(m6._normalize_triplet(cfg.w_trend, cfg.w_mom, cfg.w_rel), dtype=float)
    rel_target = float(np.clip(rel_tilt, 0.34, 0.75))
    rem = max(0.0, 1.0 - rel_target)
    tm = bw[:2]
    tm = np.array([0.5, 0.5]) if tm.sum() <= 0 else tm / tm.sum()
    w = np.array([rem * tm[0], rem * tm[1], rel_target], dtype=float)
    w = w / w.sum()
    cfg.w_trend, cfg.w_mom, cfg.w_rel = map(float, w)
    return cfg


def _policy_from_intensities(
    weekly_df: pd.DataFrame,
    stress_trigger_q: float,
    recovery_trigger_q: float,
    defensive_scale: float,
    recovery_floor: float,
    rel_tilt: float,
    cfg: Mahoraga7Config,
    crisis_state_daily: pd.Series,
) -> pd.DataFrame:
    idx = weekly_df.index
    stress_thr = float(np.nanquantile(weekly_df["stress_intensity"].dropna(), stress_trigger_q)) if weekly_df["stress_intensity"].dropna().size else np.inf
    rec_thr = float(np.nanquantile(weekly_df["recovery_intensity"].dropna(), recovery_trigger_q)) if weekly_df["recovery_intensity"].dropna().size else np.inf
    rows = []
    for dt in idx:
        stress = float(weekly_df.at[dt, "stress_intensity"])
        rec = float(weekly_df.at[dt, "recovery_intensity"])
        crisis_on = bool(crisis_state_daily.reindex([dt]).fillna(0.0).iloc[0] >= 1.0)
        action = "BASELINE"
        ext_scale = 1.0
        recovery_override_scale = 0.0
        if stress >= stress_thr and stress > rec:
            action = "DEFENSIVE_LIGHT"
            ext_scale = defensive_scale
        elif rec >= rec_thr and rec > stress:
            if (not cfg.require_crisis_for_recovery_override) or crisis_on:
                action = "RECOVERY_OVERRIDE"
                recovery_override_scale = recovery_floor
            else:
                action = "REL_TILT"
        elif weekly_df.at[dt, "rel_minus_trend"] > 0 and weekly_df.at[dt, "rel_minus_mom"] > 0:
            action = "REL_TILT"
        rows.append({
            "date": dt,
            "action": action,
            "ext_scale": float(ext_scale),
            "recovery_override_scale": float(recovery_override_scale),
            "rel_tilt": float(rel_tilt if action == "REL_TILT" else np.nan),
            "stress_thr": float(stress_thr),
            "recovery_thr": float(rec_thr),
        })
    return pd.DataFrame(rows).set_index("date")


def _custom_backtest_with_overlay(
    ohlcv: Dict[str, pd.DataFrame],
    cfg: Mahoraga7Config,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
    daily_policy: pd.DataFrame,
    label: str,
) -> Dict[str, Any]:
    # Copy of active Mahoraga 6.1 logic with two additions:
    # 1) REL_TILT can change score selection on rebalance dates
    # 2) RECOVERY_OVERRIDE can lift crisis_scale partially during crisis
    np.random.seed(cfg.random_seed)

    if universe_schedule is not None:
        all_sched_tickers = set()
        for members_json in universe_schedule["members"]:
            all_sched_tickers |= set(json.loads(members_json))
        univ_master = sorted(all_sched_tickers & set(ohlcv["close"].columns))
        use_pit_universe = True
    else:
        univ_master = [t for t in cfg.universe_static if t in ohlcv["close"].columns]
        use_pit_universe = False
    if not univ_master:
        raise ValueError("[custom_backtest_with_overlay] No valid tickers in universe")

    close = ohlcv["close"][univ_master].copy()
    high = ohlcv["high"][univ_master].copy()
    low = ohlcv["low"][univ_master].copy()
    volume = ohlcv["volume"][univ_master].copy()
    idx = close.index
    qqq = m6.to_s(ohlcv["close"][cfg.bench_qqq].reindex(idx).ffill(), "QQQ")
    spy = m6.to_s(ohlcv["close"][cfg.bench_spy].reindex(idx).ffill(), "SPY")

    crisis_scale, crisis_state = m6.compute_crisis_gate(qqq, cfg)
    turb_scale = m6.compute_turbulence(close, volume, qqq, cfg)
    corr_scale, corr_rho, corr_state = m6.compute_corr_shield(close, cfg)

    score_base = m6.compute_scores(close, qqq, cfg)
    cfg_rel = _make_rel_tilt_cfg(cfg, float(daily_policy["rel_tilt"].dropna().iloc[0])) if daily_policy["rel_tilt"].notna().any() else _make_rel_tilt_cfg(cfg, 0.60)
    score_rel = m6.compute_scores(close, qqq, cfg_rel)
    rets = close.pct_change().fillna(0.0)

    action_daily = daily_policy["action"].reindex(idx).ffill().fillna("BASELINE")
    ext_scale = daily_policy["ext_scale"].reindex(idx).ffill().fillna(1.0).clip(0.0, 1.0)
    rec_override = daily_policy["recovery_override_scale"].reindex(idx).ffill().fillna(0.0).clip(0.0, 1.0)

    # Rebalance-aware name selection with REL_TILT only when action says so
    w = pd.DataFrame(0.0, index=idx, columns=univ_master, dtype=float)
    last_w = pd.Series(0.0, index=univ_master)
    rebal = close.resample(cfg.rebalance_freq).last().index.intersection(idx)
    for dt in idx:
        if dt in rebal:
            sc_use = score_rel if action_daily.loc[dt] == "REL_TILT" else score_base
            if use_pit_universe:
                pit_members = m6.get_universe_at_date(universe_schedule, dt)
                pit_members = [t for t in pit_members if t in univ_master]
                if not pit_members:
                    last_w = pd.Series(0.0, index=univ_master)
                    w.loc[dt] = last_w.values
                    continue
                pit_scores = sc_use.loc[dt, pit_members]
                sel_names = pit_scores.nlargest(cfg.top_k).index.tolist()
                names = [n for n in sel_names if pit_scores.get(n, 0) > 0]
            else:
                sc = sc_use.loc[dt]
                names = sc.nlargest(cfg.top_k)
                names = [n for n in names.index.tolist() if sc.get(n, 0) > 0]
            if not names:
                last_w = pd.Series(0.0, index=univ_master)
            elif len(names) == 1:
                last_w = pd.Series(0.0, index=univ_master)
                last_w[names[0]] = 1.0
            else:
                lb = rets.loc[:dt].tail(cfg.hrp_window)[names].dropna()
                if len(lb) < 60:
                    _ret_fallback = lb if len(lb) > len(names) else rets.loc[:dt][names].dropna()
                    ww = m6.hrp_weights(_ret_fallback).reindex(names, fill_value=0.0)
                else:
                    ww = m6.hrp_weights(lb).reindex(names, fill_value=0.0)
                if ww.sum() > 0:
                    ww = ww.clip(upper=cfg.weight_cap) / ww.clip(upper=cfg.weight_cap).sum()
                last_w = pd.Series(0.0, index=univ_master)
                last_w[names] = ww.reindex(names, fill_value=0.0).values
        w.loc[dt] = last_w.values

    w_stop, stop_hits = m6.apply_chandelier(w, close, high, low, cfg)
    w_exec_1x = w_stop.shift(1).fillna(0.0)
    gross_1x = (w_exec_1x * rets).sum(axis=1)
    vol_sc = m6.vol_target_scale(gross_1x, cfg)

    crisis_eff = np.maximum(crisis_scale.values, rec_override.values)
    crisis_eff = pd.Series(crisis_eff, index=idx).clip(0.0, 1.0)
    cap = (crisis_eff * turb_scale * corr_scale * ext_scale).clip(0.0, cfg.max_exposure)
    tgt_sc = pd.Series(np.minimum(vol_sc.values, cap.values), index=idx).clip(0.0, cfg.max_exposure)
    exec_sc = tgt_sc.shift(1).fillna(0.0)
    w_exec = w_exec_1x.mul(exec_sc, axis=0)
    to, tc = m6._costs(w_exec, costs)
    port_net = ((w_exec * rets).sum(axis=1) - tc).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    equity = cfg.capital_initial * (1.0 + port_net).cumprod()
    exposure = w_exec.abs().sum(axis=1).clip(0.0, cfg.max_exposure)
    qqq_r = qqq.pct_change().fillna(0.0) - costs.qqq_expense_ratio
    qqq_eq = cfg.capital_initial * (1.0 + qqq_r).cumprod()
    return {
        "label": label,
        "returns_net": port_net,
        "equity": equity,
        "exposure": exposure,
        "turnover": to,
        "weights_scaled": w_exec,
        "total_scale": exec_sc,
        "total_scale_target": tgt_sc,
        "cap": cap,
        "turb_scale": turb_scale,
        "crisis_scale": crisis_eff,
        "crisis_state": crisis_state,
        "vol_scale": vol_sc,
        "corr_scale": corr_scale,
        "corr_rho": corr_rho,
        "corr_state": corr_state,
        "external_scale": ext_scale,
        "recovery_override_scale": rec_override,
        "action_daily": action_daily,
        "stop_hits": stop_hits,
        "bench": {"QQQ_r": qqq_r, "QQQ_eq": qqq_eq},
    }


def _score_overlay(base_sum: Dict[str, float], ov_sum: Dict[str, float], intervention_rate: float, recovery_rate: float, cfg: Mahoraga7Config) -> float:
    delta_sh = ov_sum["Sharpe"] - base_sum["Sharpe"]
    delta_dd = abs(base_sum["MaxDD"]) - abs(ov_sum["MaxDD"])
    delta_cagr = ov_sum["CAGR"] - base_sum["CAGR"]
    pen_int = max(0.0, intervention_rate - cfg.target_intervention_rate)
    pen_rec = max(0.0, recovery_rate - cfg.target_recovery_rate)
    return float(0.55 * delta_sh + 0.30 * delta_dd + 0.15 * delta_cagr - 0.10 * pen_int - 0.05 * pen_rec)


def _diagnostic_alignment_score(test_df: pd.DataFrame, crisis_state_weekly: pd.Series) -> float:
    # Reward stress intensity leading crisis and recovery leading crisis-off
    if test_df.empty:
        return -np.inf
    x = test_df["stress_intensity"].astype(float).values
    y = crisis_state_weekly.reindex(test_df.index).fillna(0.0).astype(float).values
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        corr_now = 0.0
    else:
        corr_now = float(np.corrcoef(x, y)[0, 1])
    x2 = test_df["recovery_intensity"].astype(float).values
    y2 = (1.0 - crisis_state_weekly.reindex(test_df.index).fillna(0.0).astype(float).values)
    if np.nanstd(x2) == 0 or np.nanstd(y2) == 0:
        corr_rec = 0.0
    else:
        corr_rec = float(np.corrcoef(x2, y2)[0, 1])
    return corr_now + corr_rec


def _calibrate_hawkes_and_overlay(
    feat_full: pd.DataFrame,
    ohlcv: Dict[str, pd.DataFrame],
    cfg_fold: Mahoraga7Config,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
    train_start: str,
    train_end: str,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    feat_train = feat_full.loc[train_start:train_end].copy()
    if len(feat_train) < cfg_fold.min_train_weeks:
        return {
            "stress_q": cfg_fold.stress_q_grid[0],
            "recovery_q": cfg_fold.recovery_q_grid[0],
            "decay": cfg_fold.hawkes_decay_grid[0],
            "stress_scale": cfg_fold.stress_scale_grid[0],
            "recovery_scale": cfg_fold.recovery_scale_grid[0],
            "defensive_scale": cfg_fold.defensive_scale_grid[0],
            "recovery_floor": cfg_fold.recovery_floor_grid[0],
            "rel_tilt": cfg_fold.rel_tilt_grid[0],
            "stress_trigger_q": cfg_fold.stress_trigger_q_grid[0],
            "recovery_trigger_q": cfg_fold.recovery_trigger_q_grid[0],
        }, pd.DataFrame()

    split_n = max(int(len(feat_train) * (1.0 - cfg_fold.inner_val_frac)), cfg_fold.min_train_weeks)
    split_n = min(split_n, len(feat_train) - max(10, len(feat_train) // 5))
    inner_train_end = str(feat_train.index[split_n - 1].date())
    inner_val_start = str(feat_train.index[split_n].date())
    inner_val_end = train_end

    # Build a baseline on inner validation once
    base_bt = m6.backtest(ohlcv, cfg_fold, costs, label="BASE_INNER", universe_schedule=universe_schedule)
    base_r = base_bt["returns_net"].loc[inner_val_start:inner_val_end]
    base_eq = base_bt["equity"].loc[inner_val_start:inner_val_end]
    base_exp = base_bt["exposure"].loc[inner_val_start:inner_val_end]
    base_to = base_bt["turnover"].loc[inner_val_start:inner_val_end]
    base_sum = m6.summarize(base_r, base_eq, base_exp, base_to, cfg_fold, "BASE_INNER")

    crisis_state_weekly = base_bt["crisis_state"].resample(cfg_fold.decision_freq).last().reindex(feat_full.index).fillna(method="ffill").fillna(0.0)

    rows = []
    best_params = None
    best_score = -np.inf

    grid = iproduct(
        cfg_fold.stress_q_grid,
        cfg_fold.recovery_q_grid,
        cfg_fold.hawkes_decay_grid,
        cfg_fold.stress_scale_grid,
        cfg_fold.recovery_scale_grid,
        cfg_fold.defensive_scale_grid,
        cfg_fold.recovery_floor_grid,
        cfg_fold.rel_tilt_grid,
        cfg_fold.stress_trigger_q_grid,
        cfg_fold.recovery_trigger_q_grid,
    )

    for stress_q, recovery_q, decay, stress_scale, recovery_scale, defensive_scale, recovery_floor, rel_tilt, stress_trigger_q, recovery_trigger_q in grid:
        hawkes_df, _ = _build_hawkes_signals(feat_full, stress_q, recovery_q, decay, stress_scale, recovery_scale, feat_full.loc[train_start:inner_train_end])
        policy_weekly = _policy_from_intensities(
            hawkes_df.loc[inner_val_start:inner_val_end],
            stress_trigger_q=stress_trigger_q,
            recovery_trigger_q=recovery_trigger_q,
            defensive_scale=defensive_scale,
            recovery_floor=recovery_floor,
            rel_tilt=rel_tilt,
            cfg=cfg_fold,
            crisis_state_daily=base_bt["crisis_state"],
        )
        policy_daily = policy_weekly.reindex(base_bt["returns_net"].index).ffill().fillna({
            "action": "BASELINE",
            "ext_scale": 1.0,
            "recovery_override_scale": 0.0,
        })
        ov_bt = _custom_backtest_with_overlay(ohlcv, cfg_fold, costs, universe_schedule, policy_daily, label="OV_INNER")
        ov_r = ov_bt["returns_net"].loc[inner_val_start:inner_val_end]
        ov_eq = ov_bt["equity"].loc[inner_val_start:inner_val_end]
        ov_exp = ov_bt["exposure"].loc[inner_val_start:inner_val_end]
        ov_to = ov_bt["turnover"].loc[inner_val_start:inner_val_end]
        ov_sum = m6.summarize(ov_r, ov_eq, ov_exp, ov_to, cfg_fold, "OV_INNER")
        intervention_rate = float((policy_weekly["action"] != "BASELINE").mean())
        recovery_rate = float((policy_weekly["action"] == "RECOVERY_OVERRIDE").mean())
        score = _score_overlay(base_sum, ov_sum, intervention_rate, recovery_rate, cfg_fold)
        if cfg_fold.variant == "7A":
            score = _diagnostic_alignment_score(hawkes_df.loc[inner_val_start:inner_val_end], crisis_state_weekly)
        row = {
            "stress_q": stress_q,
            "recovery_q": recovery_q,
            "decay": decay,
            "stress_scale": stress_scale,
            "recovery_scale": recovery_scale,
            "defensive_scale": defensive_scale,
            "recovery_floor": recovery_floor,
            "rel_tilt": rel_tilt,
            "stress_trigger_q": stress_trigger_q,
            "recovery_trigger_q": recovery_trigger_q,
            "score": score,
            "base_sharpe": base_sum["Sharpe"],
            "ov_sharpe": ov_sum["Sharpe"],
            "ov_cagr": ov_sum["CAGR"],
            "ov_maxdd": ov_sum["MaxDD"],
            "intervention_rate": intervention_rate,
            "recovery_rate": recovery_rate,
        }
        rows.append(row)
        if score > best_score:
            best_score = score
            best_params = row.copy()

    return best_params, pd.DataFrame(rows).sort_values("score", ascending=False)


def _run_single_fold(
    fold: Dict[str, Any],
    baseline_row: pd.Series,
    ohlcv: Dict[str, pd.DataFrame],
    cfg_base: Mahoraga7Config,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
    feat_full: pd.DataFrame,
) -> Dict[str, Any]:
    fold_n = int(fold["fold"])
    train_start, train_end = fold["train_start"], fold["train_end"]
    test_start, test_end = fold["test_start"], fold["test_end"]
    val_start, val_end = fold["val_start"], fold["val_end"]

    print(f"\n  ── {cfg_base.variant} FOLD {fold_n}/{len(m6.build_contiguous_folds(cfg_base, pd.DatetimeIndex(ohlcv['close'].index)))} ──")
    cfg_fold = _get_fold_cfg(ohlcv, cfg_base, costs, universe_schedule, baseline_row)
    print(f"  [IC] trend={cfg_fold.w_trend:.3f} mom={cfg_fold.w_mom:.3f} rel={cfg_fold.w_rel:.3f}")
    print(f"  [fold {fold_n}] Calibrating {cfg_base.variant} on train via inner validation …")
    best_params, calib_df = _calibrate_hawkes_and_overlay(feat_full, ohlcv, cfg_fold, costs, universe_schedule, train_start, train_end)

    hawkes_df, thr = _build_hawkes_signals(
        feat_full,
        best_params["stress_q"],
        best_params["recovery_q"],
        best_params["decay"],
        best_params["stress_scale"],
        best_params["recovery_scale"],
        feat_full.loc[train_start:train_end],
    )

    # Frozen baseline on test
    base_bt = m6.backtest(ohlcv, cfg_fold, costs, label=f"BASE_{fold_n}", universe_schedule=universe_schedule)

    weekly_policy = _policy_from_intensities(
        hawkes_df.loc[test_start:test_end],
        stress_trigger_q=best_params["stress_trigger_q"],
        recovery_trigger_q=best_params["recovery_trigger_q"],
        defensive_scale=best_params["defensive_scale"],
        recovery_floor=best_params["recovery_floor"],
        rel_tilt=best_params["rel_tilt"],
        cfg=cfg_fold,
        crisis_state_daily=base_bt["crisis_state"],
    )
    policy_daily = weekly_policy.reindex(base_bt["returns_net"].index).ffill().fillna({
        "action": "BASELINE",
        "ext_scale": 1.0,
        "recovery_override_scale": 0.0,
    })

    if cfg_base.variant == "7A":
        ov_bt = base_bt
        label_ai = "HAWKES_DIAG"
    else:
        ov_bt = _custom_backtest_with_overlay(ohlcv, cfg_fold, costs, universe_schedule, policy_daily, label=f"H7B_{fold_n}")
        label_ai = "HAWKES_OVERLAY"

    rb = base_bt["returns_net"].loc[test_start:test_end]
    qb = base_bt["equity"].loc[test_start:test_end]
    eb = base_bt["exposure"].loc[test_start:test_end]
    tb = base_bt["turnover"].loc[test_start:test_end]
    sb = m6.summarize(rb, qb, eb, tb, cfg_fold, f"BASE_{fold_n}")

    ro = ov_bt["returns_net"].loc[test_start:test_end]
    qo = ov_bt["equity"].loc[test_start:test_end]
    eo = ov_bt["exposure"].loc[test_start:test_end]
    to = ov_bt["turnover"].loc[test_start:test_end]
    so = m6.summarize(ro, qo, eo, to, cfg_fold, f"{label_ai}_{fold_n}")

    # missed rebound analysis: QQQ up while baseline nearly flat exposure due to crisis
    qqq_r = base_bt["bench"]["QQQ_r"].loc[test_start:test_end]
    missed_rebound = float(qqq_r[(base_bt["crisis_state"].loc[test_start:test_end] >= 1.0) & (qqq_r > 0)].sum())

    print(f"  [fold {fold_n}] BASE Sharpe={sb['Sharpe']:.3f} | {cfg_base.variant} Sharpe={so['Sharpe']:.3f} | Δ={so['Sharpe'] - sb['Sharpe']:+.3f}")

    return {
        "fold": fold_n,
        "base_bt": base_bt,
        "ov_bt": ov_bt,
        "fold_row": {
            "fold": fold_n,
            "train": f"{train_start}→{train_end}",
            "val": f"{val_start}→{val_end}",
            "test": f"{test_start}→{test_end}",
            "BASE_CAGR%": round(sb["CAGR"] * 100, 2),
            "BASE_Sharpe": round(sb["Sharpe"], 4),
            "BASE_MaxDD%": round(sb["MaxDD"] * 100, 2),
            f"{cfg_base.variant}_CAGR%": round(so["CAGR"] * 100, 2),
            f"{cfg_base.variant}_Sharpe": round(so["Sharpe"], 4),
            f"{cfg_base.variant}_MaxDD%": round(so["MaxDD"] * 100, 2),
            "DeltaSharpe": round(so["Sharpe"] - sb["Sharpe"], 4),
            "InterventionRate": round(float((weekly_policy["action"] != "BASELINE").mean()), 4),
            "RecoveryRate": round(float((weekly_policy["action"] == "RECOVERY_OVERRIDE").mean()), 4),
            "MissedReboundQQQ": round(missed_rebound * 100, 2),
        },
        "weekly_policy": weekly_policy.assign(fold=fold_n),
        "hawkes_df": hawkes_df.loc[test_start:test_end].assign(fold=fold_n),
        "calib_df": calib_df.assign(fold=fold_n) if not calib_df.empty else calib_df,
        "best_params": best_params,
    }


def run_walk_forward_h7(
    ohlcv: Dict[str, pd.DataFrame],
    cfg: Mahoraga7Config,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    baseline_df = _load_baseline_folds(cfg)
    feat_full = _build_context_table(ohlcv, cfg, universe_schedule)

    folds = m6.build_contiguous_folds(cfg, pd.DatetimeIndex(ohlcv["close"].index))
    if cfg.run_mode.upper() == "FAST":
        folds = [f for f in folds if int(f["fold"]) in set(cfg.fast_folds)]
        baseline_df = baseline_df[baseline_df["fold"].isin(list(cfg.fast_folds))].copy()

    tasks = []
    for fold in folds:
        row = baseline_df[baseline_df["fold"] == int(fold["fold"])].iloc[0]
        tasks.append((fold, row))

    use_parallel = cfg.outer_parallel and _JOBLIB and len(tasks) > 1
    backend = cfg.outer_backend
    if backend == "auto":
        backend = "threading" if os.name == "nt" else "loky"

    if use_parallel:
        n_jobs = min(cfg.max_outer_jobs, len(tasks))
        results = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
            delayed(_run_single_fold)(f, r, ohlcv, cfg, costs, universe_schedule, feat_full)
            for f, r in tasks
        )
    else:
        results = [_run_single_fold(f, r, ohlcv, cfg, costs, universe_schedule, feat_full) for f, r in tasks]

    results = sorted(results, key=lambda x: x["fold"])
    base_r = pd.concat([x["base_bt"]["returns_net"] for x in results]).sort_index()
    base_eq = cfg.capital_initial * (1.0 + base_r).cumprod()
    base_exp = pd.concat([x["base_bt"]["exposure"] for x in results]).sort_index().reindex(base_r.index).fillna(0.0)
    base_to = pd.concat([x["base_bt"]["turnover"] for x in results]).sort_index().reindex(base_r.index).fillna(0.0)

    ov_r = pd.concat([x["ov_bt"]["returns_net"] for x in results]).sort_index()
    ov_eq = cfg.capital_initial * (1.0 + ov_r).cumprod()
    ov_exp = pd.concat([x["ov_bt"]["exposure"] for x in results]).sort_index().reindex(ov_r.index).fillna(0.0)
    ov_to = pd.concat([x["ov_bt"]["turnover"] for x in results]).sort_index().reindex(ov_r.index).fillna(0.0)

    return {
        "base_oos_r": base_r,
        "base_oos_eq": base_eq,
        "base_oos_exp": base_exp,
        "base_oos_to": base_to,
        "ov_oos_r": ov_r,
        "ov_oos_eq": ov_eq,
        "ov_oos_exp": ov_exp,
        "ov_oos_to": ov_to,
        "fold_results": pd.DataFrame([x["fold_row"] for x in results]),
        "policy_artifacts": pd.concat([x["weekly_policy"] for x in results], ignore_index=False),
        "hawkes_artifacts": pd.concat([x["hawkes_df"] for x in results], ignore_index=False),
        "calibration_grid": pd.concat([x["calib_df"] for x in results if isinstance(x["calib_df"], pd.DataFrame) and not x["calib_df"].empty], ignore_index=True) if results else pd.DataFrame(),
        "feat_full": feat_full,
    }


def _selection_audit(wf: Dict[str, Any], cfg: Mahoraga7Config) -> pd.DataFrame:
    fr = wf["fold_results"].copy()
    if fr.empty:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "Method": "BASELINE_6_1_FROZEN",
            "MeanSharpe": fr["BASE_Sharpe"].mean(),
            "MeanCAGR%": fr["BASE_CAGR%"].mean(),
            "MeanMaxDD%": fr["BASE_MaxDD%"].mean(),
            "MeanInterventionRate": 0.0,
        },
        {
            "Method": cfg.variant,
            "MeanSharpe": fr[f"{cfg.variant}_Sharpe"].mean(),
            "MeanCAGR%": fr[f"{cfg.variant}_CAGR%"].mean(),
            "MeanMaxDD%": fr[f"{cfg.variant}_MaxDD%"].mean(),
            "MeanInterventionRate": fr["InterventionRate"].mean(),
        },
    ])


def _regime_comparison(base_r: pd.Series, ov_r: pd.Series, cfg: Mahoraga7Config, ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    idx = base_r.index.intersection(ov_r.index)
    if cfg.bench_vix in ohlcv.get("close", pd.DataFrame()).columns:
        vix = m6.to_s(ohlcv["close"][cfg.bench_vix].reindex(idx).ffill(), "VIX")
    else:
        vix = pd.Series(np.nan, index=idx)
    regimes = {
        "NORMAL": (vix < 18) if vix.notna().any() else pd.Series(True, index=idx),
        "STRESS": ((vix >= 18) & (vix < 24)) if vix.notna().any() else pd.Series(False, index=idx),
        "PANIC": (vix >= 24) if vix.notna().any() else pd.Series(False, index=idx),
    }
    rows = []
    for rg, mask in regimes.items():
        mask = mask.reindex(idx).fillna(False)
        if mask.sum() < 5:
            continue
        rb = base_r.loc[idx[mask]]
        ro = ov_r.loc[idx[mask]]
        sb = m6.summarize(rb, cfg.capital_initial * (1.0 + rb).cumprod(), pd.Series(1.0, index=rb.index), pd.Series(0.0, index=rb.index), cfg, f"BASE_{rg}")
        so = m6.summarize(ro, cfg.capital_initial * (1.0 + ro).cumprod(), pd.Series(1.0, index=ro.index), pd.Series(0.0, index=ro.index), cfg, f"{cfg.variant}_{rg}")
        rows.append({
            "Regime": rg,
            "Days": int(mask.sum()),
            "BASE_CAGR%": round(sb["CAGR"] * 100, 2),
            "BASE_Sharpe": round(sb["Sharpe"], 4),
            "BASE_MaxDD%": round(sb["MaxDD"] * 100, 2),
            f"{cfg.variant}_CAGR%": round(so["CAGR"] * 100, 2),
            f"{cfg.variant}_Sharpe": round(so["Sharpe"], 4),
            f"{cfg.variant}_MaxDD%": round(so["MaxDD"] * 100, 2),
            "DeltaSharpe": round(so["Sharpe"] - sb["Sharpe"], 4),
        })
    return pd.DataFrame(rows)


def _final_report_text(cfg: Mahoraga7Config, wf: Dict[str, Any], ohlcv: Dict[str, pd.DataFrame]) -> str:
    base_sum = m6.summarize(wf["base_oos_r"], wf["base_oos_eq"], wf["base_oos_exp"], wf["base_oos_to"], cfg, "BASELINE_6_1")
    ov_sum = m6.summarize(wf["ov_oos_r"], wf["ov_oos_eq"], wf["ov_oos_exp"], wf["ov_oos_to"], cfg, cfg.variant)
    sel = _selection_audit(wf, cfg)
    reg = _regime_comparison(wf["base_oos_r"], wf["ov_oos_r"], cfg, ohlcv)
    lines = []
    lines.append(f"MAHORAGA {cfg.variant} — FINAL REPORT")
    lines.append("")
    lines.append(DISCLAIMER)
    lines.append("\nOOS SUMMARY")
    lines.append(f"BASELINE  CAGR={base_sum['CAGR']*100:.2f}%  Sharpe={base_sum['Sharpe']:.3f}  MaxDD={base_sum['MaxDD']*100:.2f}%")
    lines.append(f"{cfg.variant}  CAGR={ov_sum['CAGR']*100:.2f}%  Sharpe={ov_sum['Sharpe']:.3f}  MaxDD={ov_sum['MaxDD']*100:.2f}%")
    lines.append("\nSELECTION AUDIT")
    lines.append(sel.to_string(index=False) if not sel.empty else "[empty]")
    lines.append("\nFOLD RESULTS")
    lines.append(wf["fold_results"].to_string(index=False) if not wf["fold_results"].empty else "[empty]")
    lines.append("\nREGIME COMPARISON")
    lines.append(reg.to_string(index=False) if not reg.empty else "[empty]")
    return "\n".join(lines)


def save_outputs_h7(cfg: Mahoraga7Config, wf: Dict[str, Any], ohlcv: Dict[str, pd.DataFrame]):
    d = cfg.outputs_dir
    _ensure_dir(d)
    base_cmp = m6.summarize(wf["base_oos_r"], wf["base_oos_eq"], wf["base_oos_exp"], wf["base_oos_to"], cfg, "BASELINE_6_1")
    ov_cmp = m6.summarize(wf["ov_oos_r"], wf["ov_oos_eq"], wf["ov_oos_exp"], wf["ov_oos_to"], cfg, cfg.variant)
    pd.DataFrame([base_cmp, ov_cmp]).to_csv(os.path.join(d, "comparison_oos.csv"), index=False)
    # full sample approximated by stitched OOS vs QQQ baseline; keep naming stable
    pd.DataFrame(wf["fold_results"]).to_csv(os.path.join(d, f"walk_forward_folds_{cfg.variant.lower()}.csv"), index=False)
    wf["policy_artifacts"].to_csv(os.path.join(d, "dynamic_mode_controls.csv"), index=True)
    wf["hawkes_artifacts"].to_csv(os.path.join(d, "hawkes_events_intensities.csv"), index=True)
    wf["feat_full"].to_csv(os.path.join(d, "meta_features_snapshot.csv"), index=True)
    if isinstance(wf.get("calibration_grid"), pd.DataFrame):
        wf["calibration_grid"].to_csv(os.path.join(d, "walk_forward_sweeps.csv"), index=False)
    _selection_audit(wf, cfg).to_csv(os.path.join(d, "selection_audit.csv"), index=False)
    _regime_comparison(wf["base_oos_r"], wf["ov_oos_r"], cfg, ohlcv).to_csv(os.path.join(d, "regime_comparison.csv"), index=False)
    with open(os.path.join(d, "final_report.txt"), "w", encoding="utf-8") as f:
        f.write(_final_report_text(cfg, wf, ohlcv))


def run_mahoraga7(variant: str = "7B", make_plots_flag: bool = False, run_mode: str = "FULL") -> Dict[str, Any]:
    cfg = Mahoraga7Config(variant=variant, make_plots_flag=make_plots_flag, run_mode=run_mode)
    cfg.outputs_dir = f"mahoraga{variant}_outputs"
    cfg.plots_dir = f"mahoraga{variant}_plots"
    cfg.label = f"MAHORAGA_{variant}"
    costs = m6.CostsConfig()
    _ensure_dir(cfg.outputs_dir)
    _ensure_dir(cfg.plots_dir)

    print("=" * 80)
    print(f"  MAHORAGA {variant} — Hawkes-inspired regime layer over 6.1")
    print("=" * 80)
    print(DISCLAIMER)

    print("\n[1] Downloading data …")
    equity_tickers = sorted(set(list(cfg.universe_static)))
    bench_tickers = [cfg.bench_qqq, cfg.bench_spy, cfg.bench_vix]
    all_tickers = sorted(set(equity_tickers + bench_tickers))
    ohlcv = m6.download_ohlcv(all_tickers, cfg.data_start, cfg.data_end, cfg.cache_dir)

    print("\n[2] Fama-French factors …")
    _ = m6.load_ff_factors(cfg.cache_dir)

    print("\n[3] Canonical universe engine …")
    ucfg = m6.UniverseConfig()
    asset_registry = m6.build_asset_registry(equity_tickers, cfg, bench_tickers)
    dqr = m6.compute_data_quality_report(ohlcv, equity_tickers, cfg)
    clean_equity = m6.filter_equity_candidates([t for t in equity_tickers if t in ohlcv["close"].columns], asset_registry, dqr, cfg)
    universe_schedule, _snap = m6.build_canonical_universe_schedule(ohlcv["close"], ohlcv["volume"], ucfg, clean_equity, cfg.data_start, cfg.data_end, registry_df=asset_registry, quality_df=dqr)
    print(f"  [universe] {len(universe_schedule)} reconstitution dates built")

    print(f"\n[4] Walk-forward {variant} over frozen Mahoraga 6.1 base …")
    wf = run_walk_forward_h7(ohlcv, cfg, costs, universe_schedule)
    save_outputs_h7(cfg, wf, ohlcv)
    return {"cfg": cfg, "wf": wf, "ohlcv": ohlcv}


if __name__ == "__main__":
    # Change variant to "7A" for diagnostics-only run.
    results = run_mahoraga7(variant="7B", make_plots_flag=False, run_mode="FULL")
