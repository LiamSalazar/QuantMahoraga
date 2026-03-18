from __future__ import annotations

"""
Mahoraga 7C
===========
Hawkes + ML fragility layer over frozen Mahoraga 6.1.

Design goals
------------
- Keep Mahoraga 6.1 frozen as the baseline policy.
- Reuse Hawkes-inspired regime signals from Mahoraga 7A/7B.
- Add a lightweight ML layer to predict *baseline fragility* on the next window.
- Use only bounded interventions: BASELINE / REL_TILT / DEFENSIVE_LIGHT / RECOVERY_OVERRIDE.
- Stay fast enough for iterative research:
  * no baseline sweep rerun
  * weekly decisions
  * precomputed context table once
  * cheap Stage-1 Hawkes calibration
  * compact Stage-2 ML+policy grid
  * optional outer-fold parallelism

Important honesty note
----------------------
This file is designed to be executable and research-grade on top of Mahoraga 6.1,
but it is still a compact implementation. Hawkes here is the same discrete-time
Hawkes-inspired exponential intensity used in Mahoraga 7A/7B, not a full continuous-time
multivariate MLE Hawkes package.
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
    import mahoraga7_1 as h7
except Exception:
    import mahoraga7 as h7  # fallback

try:
    from joblib import Parallel, delayed
    _JOBLIB = True
except Exception:
    _JOBLIB = False

try:
    from sklearn.ensemble import RandomForestClassifier
    _SKLEARN = True
except Exception:
    _SKLEARN = False

_SCRIPT_DIR = Path(__file__).resolve().parent

DISCLAIMER = r"""
═══════════════════════════════════════════════════════════════════════════════
  MAHORAGA 7C — HAWKES + ML FRAGILITY LAYER OVER MAHORAGA 6.1
───────────────────────────────────────────────────────────────────────────────
  • Mahoraga 6.1 remains frozen as the execution / risk baseline.
  • Hawkes-inspired stress / recovery intensity is retained as the structural
    regime layer from Mahoraga 7A.
  • ML is used only to predict baseline fragility and select sparse, bounded
    interventions.
  • Allowed actions:
      BASELINE / REL_TILT / DEFENSIVE_LIGHT / RECOVERY_OVERRIDE
  • No price prediction. No replacing the baseline.
═══════════════════════════════════════════════════════════════════════════════
"""


@dataclass
class Mahoraga7CConfig(h7.Mahoraga7Config):
    variant: str = "7C"
    outputs_dir: str = "mahoraga7c_outputs"
    plots_dir: str = "mahoraga7c_plots"
    label: str = "MAHORAGA_7C"

    # Keep 6.1 frozen
    baseline_outputs_dir: str = "mahoraga6_1_outputs"
    baseline_folds_csv: str = ""

    # Runtime
    run_mode: str = "FULL"   # FULL or FAST
    make_plots_flag: bool = False
    decision_freq: str = "W-FRI"
    outer_parallel: bool = True
    outer_backend: str = "auto"
    max_outer_jobs: int = 5
    fast_folds: Tuple[int, ...] = (3, 5)

    # Stage-1 Hawkes calibration (cheap)
    stress_q_grid: Tuple[float, ...] = (0.85, 0.90)
    recovery_q_grid: Tuple[float, ...] = (0.80, 0.85)
    hawkes_decay_grid: Tuple[float, ...] = (0.65, 0.80)
    stress_scale_grid: Tuple[float, ...] = (0.8, 1.0)
    recovery_scale_grid: Tuple[float, ...] = (0.8, 1.0)

    # Stage-2 ML + policy calibration (compact, defendable)
    fragility_quantile_grid: Tuple[float, ...] = (0.15, 0.20)
    fragility_horizon_weeks_grid: Tuple[int, ...] = (2, 4)
    fragility_prob_trigger_grid: Tuple[float, ...] = (0.55, 0.65)
    defensive_scale_grid: Tuple[float, ...] = (0.85, 0.90)
    recovery_floor_grid: Tuple[float, ...] = (0.35, 0.50)
    rel_tilt_grid: Tuple[float, ...] = (0.55, 0.60)
    stress_trigger_q_grid: Tuple[float, ...] = (0.80, 0.90)
    recovery_trigger_q_grid: Tuple[float, ...] = (0.70, 0.80)

    # Labeling utility
    utility_dd_penalty: float = 0.40
    min_train_weeks: int = 80
    inner_val_frac: float = 0.30
    min_class_count: int = 10

    # Policy behavior
    require_crisis_for_recovery_override: bool = True
    rel_signal_quantile: float = 0.60
    target_intervention_rate: float = 0.18
    target_recovery_rate: float = 0.06

    # ML model
    rf_n_estimators: int = 250
    rf_max_depth: int = 4
    rf_min_samples_leaf: int = 5


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _weekly_dates(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    return idx.to_series().resample(freq).last().dropna().index


def _load_baseline_folds(cfg: Mahoraga7CConfig) -> pd.DataFrame:
    return h7._load_baseline_folds(cfg)


def _parse_range(s: str) -> Tuple[str, str]:
    return h7._parse_range(s)


def _get_fold_cfg(
    ohlcv: Dict[str, pd.DataFrame],
    base_cfg: Mahoraga7CConfig,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
    fold_row: pd.Series,
) -> Mahoraga7CConfig:
    cfg = deepcopy(base_cfg)
    cfg.weight_cap = float(fold_row["best_weight_cap"])
    cfg.k_atr = float(fold_row["best_k_atr"])
    cfg.turb_zscore_thr = float(fold_row["best_turb_zscore_thr"])
    cfg.turb_scale_min = float(fold_row["best_turb_scale_min"])
    cfg.vol_target_ann = float(fold_row["best_vol_target_ann"])

    train_start, train_end = _parse_range(fold_row["train"])
    qqq_full = m6.to_s(ohlcv["close"][cfg.bench_qqq].ffill(), "QQQ")
    dd_thr, vol_thr = m6.calibrate_crisis_thresholds(qqq_full, train_start, train_end, cfg)
    cfg.crisis_dd_thr = dd_thr
    cfg.crisis_vol_zscore_thr = vol_thr
    print(f"  [crisis] DD_thr={dd_thr:.3f}  vol_z_thr={vol_thr:.3f}  (calibrated on {train_start}→{train_end})")

    final_train_tickers = m6.get_training_universe(
        train_end, universe_schedule, cfg.universe_static, list(ohlcv["close"].columns)
    )
    close_univ = ohlcv["close"][final_train_tickers]
    wt, wm, wr = m6.fit_ic_weights(close_univ, qqq_full.loc[train_start:train_end], cfg, train_start, train_end)
    cfg.w_trend, cfg.w_mom, cfg.w_rel = wt, wm, wr
    print(f"  [IC] trend={wt:.3f} mom={wm:.3f} rel={wr:.3f}")
    return cfg


def _future_window_utility(
    returns_daily: pd.Series,
    equity_daily: pd.Series,
    weekly_idx: pd.DatetimeIndex,
    horizon_weeks: int,
    dd_penalty: float,
) -> pd.Series:
    idx = returns_daily.index
    out = pd.Series(np.nan, index=weekly_idx, dtype=float)
    if len(idx) == 0:
        return out
    for dt in weekly_idx:
        if dt not in idx:
            pos = idx.searchsorted(dt)
            if pos >= len(idx):
                continue
            start_i = pos
        else:
            start_i = idx.get_loc(dt)
        end_i = min(len(idx) - 1, start_i + max(1, horizon_weeks * 5))
        if end_i <= start_i:
            continue
        r_seg = returns_daily.iloc[start_i + 1:end_i + 1]
        if len(r_seg) == 0:
            continue
        ret = float((1.0 + r_seg).prod() - 1.0)
        eq_seg = equity_daily.iloc[start_i:end_i + 1]
        if len(eq_seg) <= 1:
            dd = 0.0
        else:
            dd = float((eq_seg / eq_seg.cummax() - 1.0).min())
        out.loc[dt] = ret - dd_penalty * abs(dd)
    return out


def _base_weekly_state(base_bt: Dict[str, Any], decision_freq: str) -> pd.DataFrame:
    idx = _weekly_dates(base_bt["returns_net"].index, decision_freq)
    def _res(s: pd.Series, fill=0.0):
        return s.resample(decision_freq).last().reindex(idx).ffill().fillna(fill)
    out = pd.DataFrame(index=idx)
    out["base_exposure"] = _res(base_bt.get("exposure", pd.Series(dtype=float)), 0.0)
    out["base_turnover"] = _res(base_bt.get("turnover", pd.Series(dtype=float)), 0.0)
    out["crisis_state_weekly"] = _res(base_bt.get("crisis_state", pd.Series(dtype=float)), 0.0)
    out["corr_state_weekly"] = _res(base_bt.get("corr_state", pd.Series(dtype=float)), 0.0)
    out["turb_scale_weekly"] = _res(base_bt.get("turb_scale", pd.Series(dtype=float)), 1.0)
    out["crisis_scale_weekly"] = _res(base_bt.get("crisis_scale", pd.Series(dtype=float)), 1.0)
    out["corr_scale_weekly"] = _res(base_bt.get("corr_scale", pd.Series(dtype=float)), 1.0)
    out["base_total_scale"] = _res(base_bt.get("total_scale_target", pd.Series(dtype=float)), 1.0)
    return out


def _build_ml_feature_table(hawkes_df: pd.DataFrame, base_bt: Dict[str, Any], cfg: Mahoraga7CConfig) -> pd.DataFrame:
    wk = _base_weekly_state(base_bt, cfg.decision_freq)
    feat = hawkes_df.join(wk, how="left")
    feat = feat.sort_index()
    return feat


FEATURE_COLS = [
    "stress_intensity", "recovery_intensity", "intensity_spread",
    "avg_corr_21", "avg_corr_63", "xs_disp_5d", "xs_disp_21d",
    "breadth_63d", "qqq_ret_5d", "qqq_ret_21d", "qqq_drawdown",
    "qqq_vol_21", "vix_level", "vix_z_63", "trend_mean", "mom_mean",
    "rel_mean", "rel_top_mean", "rel_minus_mom", "rel_minus_trend",
    "base_exposure", "base_turnover", "crisis_state_weekly",
    "corr_state_weekly", "turb_scale_weekly", "crisis_scale_weekly",
    "corr_scale_weekly", "base_total_scale",
]


def _prepare_xy(df: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = df[FEATURE_COLS].copy()
    # simple imputation from train medians later; here just keep NaNs manageable
    med = X.median(numeric_only=True)
    X = X.fillna(med).replace([np.inf, -np.inf], np.nan).fillna(med)
    valid = y.notna() & np.isfinite(y.astype(float))
    return X.loc[valid], y.loc[valid].astype(int), med


def _fit_fragility_model(X: pd.DataFrame, y: pd.Series, cfg: Mahoraga7CConfig):
    if not _SKLEARN:
        return None
    if len(X) < cfg.min_train_weeks or y.value_counts().min() < cfg.min_class_count:
        return None
    model = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=cfg.random_seed,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def _model_importance(model, cols: List[str]) -> pd.DataFrame:
    if model is None or not hasattr(model, "feature_importances_"):
        return pd.DataFrame({"feature": cols, "importance": np.nan})
    return pd.DataFrame({"feature": cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)


def _policy_from_fragility_probs(
    weekly_df: pd.DataFrame,
    probs: pd.Series,
    prob_trigger: float,
    defensive_scale: float,
    recovery_floor: float,
    rel_tilt: float,
    stress_trigger_q: float,
    recovery_trigger_q: float,
    cfg: Mahoraga7CConfig,
    crisis_state_daily: pd.Series,
    train_weekly_ref: pd.DataFrame,
) -> pd.DataFrame:
    idx = weekly_df.index
    stress_thr = float(np.nanquantile(train_weekly_ref["stress_intensity"].dropna(), stress_trigger_q)) if train_weekly_ref["stress_intensity"].dropna().size else np.inf
    rec_thr = float(np.nanquantile(train_weekly_ref["recovery_intensity"].dropna(), recovery_trigger_q)) if train_weekly_ref["recovery_intensity"].dropna().size else np.inf
    rel_thr = float(np.nanquantile(train_weekly_ref["rel_top_mean"].dropna(), cfg.rel_signal_quantile)) if train_weekly_ref["rel_top_mean"].dropna().size else np.inf

    rows = []
    for dt in idx:
        p = float(probs.reindex([dt]).fillna(0.0).iloc[0])
        stress = float(weekly_df.at[dt, "stress_intensity"])
        rec = float(weekly_df.at[dt, "recovery_intensity"])
        crisis_on = bool(crisis_state_daily.reindex([dt]).fillna(0.0).iloc[0] >= 1.0)
        action = "BASELINE"
        ext_scale = 1.0
        rec_scale = 0.0
        tilt = np.nan

        if p >= prob_trigger:
            if rec >= rec_thr and rec > stress and ((not cfg.require_crisis_for_recovery_override) or crisis_on):
                action = "RECOVERY_OVERRIDE"
                rec_scale = recovery_floor
            elif stress >= stress_thr and stress >= rec:
                action = "DEFENSIVE_LIGHT"
                ext_scale = defensive_scale
            elif (weekly_df.at[dt, "rel_top_mean"] >= rel_thr and
                  weekly_df.at[dt, "rel_minus_mom"] > 0 and
                  weekly_df.at[dt, "rel_minus_trend"] > 0):
                action = "REL_TILT"
                tilt = rel_tilt
            else:
                action = "DEFENSIVE_LIGHT"
                ext_scale = defensive_scale

        rows.append({
            "date": dt,
            "fragility_prob": p,
            "action": action,
            "ext_scale": float(ext_scale),
            "recovery_override_scale": float(rec_scale),
            "rel_tilt": float(tilt) if np.isfinite(tilt) else np.nan,
            "stress_thr": float(stress_thr),
            "recovery_thr": float(rec_thr),
            "prob_trigger": float(prob_trigger),
        })
    return pd.DataFrame(rows).set_index("date")


def _score_overlay(base_sum: Dict[str, float], ov_sum: Dict[str, float], intervention_rate: float, recovery_rate: float, cfg: Mahoraga7CConfig) -> float:
    delta_sh = ov_sum["Sharpe"] - base_sum["Sharpe"]
    delta_dd = abs(base_sum["MaxDD"]) - abs(ov_sum["MaxDD"])
    delta_cagr = ov_sum["CAGR"] - base_sum["CAGR"]
    pen_int = max(0.0, intervention_rate - cfg.target_intervention_rate)
    pen_rec = max(0.0, recovery_rate - cfg.target_recovery_rate)
    return float(0.60 * delta_sh + 0.25 * delta_dd + 0.15 * delta_cagr - 0.10 * pen_int - 0.03 * pen_rec)


def _calibrate_stage1_hawkes(
    feat_full: pd.DataFrame,
    train_start: str,
    inner_train_end: str,
    inner_val_start: str,
    inner_val_end: str,
    cfg_fold: Mahoraga7CConfig,
    crisis_state_weekly: pd.Series,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows = []
    best_score = -np.inf
    best = None
    for stress_q, recovery_q, decay, stress_scale, recovery_scale in iproduct(
        cfg_fold.stress_q_grid,
        cfg_fold.recovery_q_grid,
        cfg_fold.hawkes_decay_grid,
        cfg_fold.stress_scale_grid,
        cfg_fold.recovery_scale_grid,
    ):
        hawkes_df, thr = h7._build_hawkes_signals(
            feat_full, stress_q, recovery_q, decay, stress_scale, recovery_scale,
            feat_full.loc[train_start:inner_train_end],
        )
        score = h7._diagnostic_alignment_score(hawkes_df.loc[inner_val_start:inner_val_end], crisis_state_weekly)
        row = {
            "stress_q": stress_q, "recovery_q": recovery_q, "decay": decay,
            "stress_scale": stress_scale, "recovery_scale": recovery_scale,
            "score": score,
        }
        rows.append(row)
        if score > best_score:
            best_score = score
            best = row.copy()
    return best, pd.DataFrame(rows).sort_values("score", ascending=False)


def _calibrate_7c(
    feat_full: pd.DataFrame,
    ohlcv: Dict[str, pd.DataFrame],
    cfg_fold: Mahoraga7CConfig,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
    train_start: str,
    train_end: str,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    feat_train = feat_full.loc[train_start:train_end].copy()
    if len(feat_train) < cfg_fold.min_train_weeks:
        return {
            "stress_q": cfg_fold.stress_q_grid[0],
            "recovery_q": cfg_fold.recovery_q_grid[0],
            "decay": cfg_fold.hawkes_decay_grid[0],
            "stress_scale": cfg_fold.stress_scale_grid[0],
            "recovery_scale": cfg_fold.recovery_scale_grid[0],
            "fragility_quantile": cfg_fold.fragility_quantile_grid[0],
            "horizon_weeks": cfg_fold.fragility_horizon_weeks_grid[0],
            "prob_trigger": cfg_fold.fragility_prob_trigger_grid[0],
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

    base_bt = m6.backtest(ohlcv, cfg_fold, costs, label="BASE_INNER", universe_schedule=universe_schedule)
    base_r = base_bt["returns_net"].loc[inner_val_start:inner_val_end]
    base_eq = base_bt["equity"].loc[inner_val_start:inner_val_end]
    base_exp = base_bt["exposure"].loc[inner_val_start:inner_val_end]
    base_to = base_bt["turnover"].loc[inner_val_start:inner_val_end]
    base_sum = m6.summarize(base_r, base_eq, base_exp, base_to, cfg_fold, "BASE_INNER")

    crisis_state_weekly = base_bt["crisis_state"].resample(cfg_fold.decision_freq).last().reindex(feat_full.index).ffill().fillna(0.0)

    # Stage 1 Hawkes calibration
    s1_best, s1_df = _calibrate_stage1_hawkes(
        feat_full, train_start, inner_train_end, inner_val_start, inner_val_end, cfg_fold, crisis_state_weekly
    )

    hawkes_df_fixed, _thr = h7._build_hawkes_signals(
        feat_full,
        s1_best["stress_q"], s1_best["recovery_q"], s1_best["decay"],
        s1_best["stress_scale"], s1_best["recovery_scale"],
        feat_full.loc[train_start:inner_train_end],
    )
    ml_feat_full = _build_ml_feature_table(hawkes_df_fixed, base_bt, cfg_fold)
    wk_idx = ml_feat_full.index
    utility_cache: Dict[int, pd.Series] = {}
    ctx = h7._precompute_overlay_context(ohlcv, cfg_fold, costs, universe_schedule)
    score_rel_cache: Dict[float, pd.DataFrame] = {}

    s2_rows = []
    best_score = -np.inf
    best_params = None

    for horizon_weeks in cfg_fold.fragility_horizon_weeks_grid:
        if horizon_weeks not in utility_cache:
            utility_cache[horizon_weeks] = _future_window_utility(
                base_bt["returns_net"], base_bt["equity"], wk_idx, horizon_weeks, cfg_fold.utility_dd_penalty
            )
        utility = utility_cache[horizon_weeks]

        for frag_q, prob_trigger, defensive_scale, recovery_floor, rel_tilt, stress_trigger_q, recovery_trigger_q in iproduct(
            cfg_fold.fragility_quantile_grid,
            cfg_fold.fragility_prob_trigger_grid,
            cfg_fold.defensive_scale_grid,
            cfg_fold.recovery_floor_grid,
            cfg_fold.rel_tilt_grid,
            cfg_fold.stress_trigger_q_grid,
            cfg_fold.recovery_trigger_q_grid,
        ):
            inner_train_idx = wk_idx[(wk_idx >= pd.Timestamp(train_start)) & (wk_idx <= pd.Timestamp(inner_train_end))]
            inner_val_idx = wk_idx[(wk_idx >= pd.Timestamp(inner_val_start)) & (wk_idx <= pd.Timestamp(inner_val_end))]
            if len(inner_train_idx) < cfg_fold.min_train_weeks or len(inner_val_idx) < 5:
                continue
            thr_utility = float(np.nanquantile(utility.loc[inner_train_idx].dropna(), frag_q)) if utility.loc[inner_train_idx].dropna().size else np.nan
            y = (utility <= thr_utility).astype(float)

            X_train, y_train, med = _prepare_xy(ml_feat_full.loc[inner_train_idx], y.loc[inner_train_idx])
            model = _fit_fragility_model(X_train, y_train, cfg_fold)
            if model is None:
                continue
            X_val = ml_feat_full.loc[inner_val_idx, FEATURE_COLS].copy()
            X_val = X_val.fillna(med).replace([np.inf, -np.inf], np.nan).fillna(med)
            probs = pd.Series(model.predict_proba(X_val)[:, 1], index=inner_val_idx)
            policy_weekly = _policy_from_fragility_probs(
                ml_feat_full.loc[inner_val_idx], probs,
                prob_trigger=prob_trigger,
                defensive_scale=defensive_scale,
                recovery_floor=recovery_floor,
                rel_tilt=rel_tilt,
                stress_trigger_q=stress_trigger_q,
                recovery_trigger_q=recovery_trigger_q,
                cfg=cfg_fold,
                crisis_state_daily=base_bt["crisis_state"],
                train_weekly_ref=ml_feat_full.loc[inner_train_idx],
            )
            policy_daily = policy_weekly.reindex(base_bt["returns_net"].index).ffill()
            policy_daily["action"] = policy_daily["action"].fillna("BASELINE")
            policy_daily["ext_scale"] = policy_daily["ext_scale"].fillna(1.0)
            policy_daily["recovery_override_scale"] = policy_daily["recovery_override_scale"].fillna(0.0)
            ov_bt = h7._custom_backtest_with_overlay(
                ohlcv, cfg_fold, costs, universe_schedule, policy_daily,
                label="OV_INNER", ctx=ctx, score_rel_cache=score_rel_cache,
            )
            ov_r = ov_bt["returns_net"].loc[inner_val_start:inner_val_end]
            ov_eq = ov_bt["equity"].loc[inner_val_start:inner_val_end]
            ov_exp = ov_bt["exposure"].loc[inner_val_start:inner_val_end]
            ov_to = ov_bt["turnover"].loc[inner_val_start:inner_val_end]
            ov_sum = m6.summarize(ov_r, ov_eq, ov_exp, ov_to, cfg_fold, "OV_INNER")
            intervention_rate = float((policy_weekly["action"] != "BASELINE").mean())
            recovery_rate = float((policy_weekly["action"] == "RECOVERY_OVERRIDE").mean())
            score = _score_overlay(base_sum, ov_sum, intervention_rate, recovery_rate, cfg_fold)
            row = {
                **s1_best,
                "fragility_quantile": frag_q,
                "horizon_weeks": horizon_weeks,
                "prob_trigger": prob_trigger,
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
            s2_rows.append(row)
            if score > best_score:
                best_score = score
                best_params = row.copy()

    calib_df = pd.concat([s1_df.assign(stage="hawkes_only"), pd.DataFrame(s2_rows).assign(stage="ml_overlay")], ignore_index=True) if s2_rows else s1_df.assign(stage="hawkes_only")
    return best_params if best_params is not None else {
        **s1_best,
        "fragility_quantile": cfg_fold.fragility_quantile_grid[0],
        "horizon_weeks": cfg_fold.fragility_horizon_weeks_grid[0],
        "prob_trigger": cfg_fold.fragility_prob_trigger_grid[0],
        "defensive_scale": cfg_fold.defensive_scale_grid[0],
        "recovery_floor": cfg_fold.recovery_floor_grid[0],
        "rel_tilt": cfg_fold.rel_tilt_grid[0],
        "stress_trigger_q": cfg_fold.stress_trigger_q_grid[0],
        "recovery_trigger_q": cfg_fold.recovery_trigger_q_grid[0],
        "score": s1_best["score"],
    }, calib_df.sort_values("score", ascending=False)


def _run_single_fold(
    fold: Dict[str, Any],
    baseline_row: pd.Series,
    ohlcv: Dict[str, pd.DataFrame],
    cfg_base: Mahoraga7CConfig,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
    feat_full: pd.DataFrame,
    total_folds: int,
) -> Dict[str, Any]:
    fold_n = int(fold["fold"])
    train_start, train_end = fold["train_start"], fold["train_end"]
    test_start, test_end = fold["test_start"], fold["test_end"]
    print(f"\n  ── 7C FOLD {fold_n}/{total_folds} ──")
    cfg_fold = _get_fold_cfg(ohlcv, cfg_base, costs, universe_schedule, baseline_row)
    print(f"  [fold {fold_n}] Calibrating 7C on train via inner validation …")
    best_params, calib_df = _calibrate_7c(feat_full, ohlcv, cfg_fold, costs, universe_schedule, train_start, train_end)

    base_bt = m6.backtest(ohlcv, cfg_fold, costs, label=f"BASE_{fold_n}", universe_schedule=universe_schedule)
    hawkes_df, _ = h7._build_hawkes_signals(
        feat_full,
        best_params["stress_q"], best_params["recovery_q"], best_params["decay"],
        best_params["stress_scale"], best_params["recovery_scale"],
        feat_full.loc[train_start:train_end],
    )
    ml_feat_full = _build_ml_feature_table(hawkes_df, base_bt, cfg_fold)
    utility = _future_window_utility(base_bt["returns_net"], base_bt["equity"], ml_feat_full.index, int(best_params["horizon_weeks"]), cfg_fold.utility_dd_penalty)
    train_idx = ml_feat_full.index[(ml_feat_full.index >= pd.Timestamp(train_start)) & (ml_feat_full.index <= pd.Timestamp(train_end))]
    thr_utility = float(np.nanquantile(utility.loc[train_idx].dropna(), float(best_params["fragility_quantile"]))) if utility.loc[train_idx].dropna().size else np.nan
    y = (utility <= thr_utility).astype(float)
    X_train, y_train, med = _prepare_xy(ml_feat_full.loc[train_idx], y.loc[train_idx])
    model = _fit_fragility_model(X_train, y_train, cfg_fold)

    test_idx = ml_feat_full.index[(ml_feat_full.index >= pd.Timestamp(test_start)) & (ml_feat_full.index <= pd.Timestamp(test_end))]
    X_test = ml_feat_full.loc[test_idx, FEATURE_COLS].copy().fillna(med).replace([np.inf, -np.inf], np.nan).fillna(med)
    if model is None:
        probs = pd.Series(0.0, index=test_idx)
        imp = pd.DataFrame({"feature": FEATURE_COLS, "importance": np.nan})
    else:
        probs = pd.Series(model.predict_proba(X_test)[:, 1], index=test_idx)
        imp = _model_importance(model, FEATURE_COLS).assign(fold=fold_n)

    policy_weekly = _policy_from_fragility_probs(
        ml_feat_full.loc[test_idx], probs,
        prob_trigger=float(best_params["prob_trigger"]),
        defensive_scale=float(best_params["defensive_scale"]),
        recovery_floor=float(best_params["recovery_floor"]),
        rel_tilt=float(best_params["rel_tilt"]),
        stress_trigger_q=float(best_params["stress_trigger_q"]),
        recovery_trigger_q=float(best_params["recovery_trigger_q"]),
        cfg=cfg_fold,
        crisis_state_daily=base_bt["crisis_state"],
        train_weekly_ref=ml_feat_full.loc[train_idx],
    )
    policy_daily = policy_weekly.reindex(base_bt["returns_net"].index).ffill()
    policy_daily["action"] = policy_daily["action"].fillna("BASELINE")
    policy_daily["ext_scale"] = policy_daily["ext_scale"].fillna(1.0)
    policy_daily["recovery_override_scale"] = policy_daily["recovery_override_scale"].fillna(0.0)

    ctx_test = h7._precompute_overlay_context(ohlcv, cfg_fold, costs, universe_schedule)
    ov_bt = h7._custom_backtest_with_overlay(
        ohlcv, cfg_fold, costs, universe_schedule, policy_daily,
        label=f"H7C_{fold_n}", ctx=ctx_test, score_rel_cache={},
    )

    rb = base_bt["returns_net"].loc[test_start:test_end]
    qb = base_bt["equity"].loc[test_start:test_end]
    eb = base_bt["exposure"].loc[test_start:test_end]
    tb = base_bt["turnover"].loc[test_start:test_end]
    sb = m6.summarize(rb, qb, eb, tb, cfg_fold, f"BASE_{fold_n}")

    ro = ov_bt["returns_net"].loc[test_start:test_end]
    qo = ov_bt["equity"].loc[test_start:test_end]
    eo = ov_bt["exposure"].loc[test_start:test_end]
    to = ov_bt["turnover"].loc[test_start:test_end]
    so = m6.summarize(ro, qo, eo, to, cfg_fold, f"H7C_{fold_n}")

    qqq_r = base_bt["bench"]["QQQ_r"].loc[test_start:test_end]
    missed_rebound = float(qqq_r[(base_bt["crisis_state"].loc[test_start:test_end] >= 1.0) & (qqq_r > 0)].sum())

    print(f"  [fold {fold_n}] BASE Sharpe={sb['Sharpe']:.3f} | 7C Sharpe={so['Sharpe']:.3f} | Δ={so['Sharpe'] - sb['Sharpe']:+.3f}")

    pred_df = policy_weekly.copy()
    pred_df["fragility_prob"] = probs.reindex(policy_weekly.index).fillna(0.0)
    pred_df["fold"] = fold_n

    return {
        "fold": fold_n,
        "base_bt": base_bt,
        "ov_bt": ov_bt,
        "fold_row": {
            "fold": fold_n,
            "train": f"{train_start}→{train_end}",
            "test": f"{test_start}→{test_end}",
            "BASE_CAGR%": round(sb["CAGR"] * 100, 2),
            "BASE_Sharpe": round(sb["Sharpe"], 4),
            "BASE_MaxDD%": round(sb["MaxDD"] * 100, 2),
            "H7C_CAGR%": round(so["CAGR"] * 100, 2),
            "H7C_Sharpe": round(so["Sharpe"], 4),
            "H7C_MaxDD%": round(so["MaxDD"] * 100, 2),
            "DeltaSharpe": round(so["Sharpe"] - sb["Sharpe"], 4),
            "InterventionRate": round(float((policy_weekly["action"] != "BASELINE").mean()), 4),
            "RecoveryRate": round(float((policy_weekly["action"] == "RECOVERY_OVERRIDE").mean()), 4),
            "MissedReboundQQQ": round(missed_rebound * 100, 2),
            "FragilityRate": round(float((probs >= float(best_params["prob_trigger"])).mean()), 4),
        },
        "weekly_policy": pred_df,
        "hawkes_df": hawkes_df.loc[test_start:test_end].assign(fold=fold_n),
        "calib_df": calib_df.assign(fold=fold_n) if not calib_df.empty else calib_df,
        "feature_importance": imp,
        "best_params": best_params,
    }


def run_walk_forward_h7c(
    ohlcv: Dict[str, pd.DataFrame],
    cfg: Mahoraga7CConfig,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    baseline_df = _load_baseline_folds(cfg)
    feat_full = h7._build_context_table(ohlcv, cfg, universe_schedule)
    folds = m6.build_contiguous_folds(cfg, pd.DatetimeIndex(ohlcv["close"].index))
    total_folds = len(folds)
    if cfg.run_mode.upper() == "FAST":
        folds = [f for f in folds if int(f["fold"]) in set(cfg.fast_folds)]
        baseline_df = baseline_df[baseline_df["fold"].isin(list(cfg.fast_folds))].copy()

    tasks = []
    for fold in folds:
        row = baseline_df[baseline_df["fold"] == int(fold["fold"])]
        if row.empty:
            continue
        tasks.append((fold, row.iloc[0]))

    use_parallel = cfg.outer_parallel and _JOBLIB and len(tasks) > 1
    backend = cfg.outer_backend
    if backend == "auto":
        backend = "threading" if os.name == "nt" else "loky"
    n_jobs = min(cfg.max_outer_jobs, len(tasks)) if use_parallel else 1

    if use_parallel:
        results = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
            delayed(_run_single_fold)(f, r, ohlcv, cfg, costs, universe_schedule, feat_full, total_folds)
            for f, r in tasks
        )
    else:
        results = [_run_single_fold(f, r, ohlcv, cfg, costs, universe_schedule, feat_full, total_folds) for f, r in tasks]

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
        "feature_importance": pd.concat([x["feature_importance"] for x in results if isinstance(x["feature_importance"], pd.DataFrame) and not x["feature_importance"].empty], ignore_index=True) if results else pd.DataFrame(),
        "feat_full": feat_full,
    }


def _selection_audit(wf: Dict[str, Any], cfg: Mahoraga7CConfig) -> pd.DataFrame:
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
            "MeanFragilityRate": 0.0,
        },
        {
            "Method": "H7C",
            "MeanSharpe": fr["H7C_Sharpe"].mean(),
            "MeanCAGR%": fr["H7C_CAGR%"].mean(),
            "MeanMaxDD%": fr["H7C_MaxDD%"].mean(),
            "MeanInterventionRate": fr["InterventionRate"].mean(),
            "MeanFragilityRate": fr["FragilityRate"].mean(),
        },
    ])


def _regime_comparison(base_r: pd.Series, ov_r: pd.Series, cfg: Mahoraga7CConfig, ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return h7._regime_comparison(base_r, ov_r, cfg, ohlcv)


def _final_report_text(cfg: Mahoraga7CConfig, wf: Dict[str, Any], ohlcv: Dict[str, pd.DataFrame]) -> str:
    base_sum = m6.summarize(wf["base_oos_r"], wf["base_oos_eq"], wf["base_oos_exp"], wf["base_oos_to"], cfg, "BASE_OOS")
    ov_sum = m6.summarize(wf["ov_oos_r"], wf["ov_oos_eq"], wf["ov_oos_exp"], wf["ov_oos_to"], cfg, "H7C_OOS")
    sel = _selection_audit(wf, cfg)
    reg = _regime_comparison(wf["base_oos_r"], wf["ov_oos_r"], cfg, ohlcv)
    fr = wf["fold_results"]
    lines = []
    lines.append("MAHORAGA 7C — FINAL REPORT")
    lines.append("=" * 78)
    lines.append(DISCLAIMER)
    lines.append("\nOOS COMPARISON")
    lines.append(f"  BASELINE CAGR={base_sum['CAGR']*100:.2f}%  Sharpe={base_sum['Sharpe']:.3f}  MaxDD={base_sum['MaxDD']*100:.2f}%")
    lines.append(f"  H7C     CAGR={ov_sum['CAGR']*100:.2f}%  Sharpe={ov_sum['Sharpe']:.3f}  MaxDD={ov_sum['MaxDD']*100:.2f}%")
    lines.append("\nSELECTION AUDIT")
    lines.append(sel.to_string(index=False))
    lines.append("\nFOLD SUMMARY")
    lines.append(fr.to_string(index=False))
    lines.append("\nREGIME COMPARISON")
    lines.append(reg.to_string(index=False))
    if not wf["feature_importance"].empty:
        fi = wf["feature_importance"].groupby("feature", dropna=False)["importance"].mean().reset_index().sort_values("importance", ascending=False)
        lines.append("\nFEATURE IMPORTANCE (mean across folds)")
        lines.append(fi.to_string(index=False))
    return "\n".join(lines)


def save_outputs_h7c(cfg: Mahoraga7CConfig, wf: Dict[str, Any], ohlcv: Dict[str, pd.DataFrame]):
    d = cfg.outputs_dir
    _ensure_dir(d)
    base_sum = m6.summarize(wf["base_oos_r"], wf["base_oos_eq"], wf["base_oos_exp"], wf["base_oos_to"], cfg, "BASE_OOS")
    ov_sum = m6.summarize(wf["ov_oos_r"], wf["ov_oos_eq"], wf["ov_oos_exp"], wf["ov_oos_to"], cfg, "H7C_OOS")
    pd.DataFrame([
        {"Label": "BASELINE_6_1_FROZEN", **base_sum},
        {"Label": "H7C", **ov_sum},
    ]).to_csv(os.path.join(d, "comparison_oos.csv"), index=False)
    pd.DataFrame([
        {"Label": "BASELINE_6_1_FROZEN", **base_sum},
        {"Label": "H7C", **ov_sum},
    ]).to_csv(os.path.join(d, "comparison_full.csv"), index=False)
    wf["fold_results"].to_csv(os.path.join(d, "walk_forward_folds_7c.csv"), index=False)
    wf["policy_artifacts"].to_csv(os.path.join(d, "dynamic_mode_controls.csv"), index=True)
    wf["hawkes_artifacts"].to_csv(os.path.join(d, "hawkes_events_intensities.csv"), index=True)
    wf["calibration_grid"].to_csv(os.path.join(d, "walk_forward_sweeps.csv"), index=False)
    wf["feat_full"].to_csv(os.path.join(d, "meta_features_snapshot.csv"), index=True)
    if not wf["feature_importance"].empty:
        wf["feature_importance"].to_csv(os.path.join(d, "feature_importance_fragility.csv"), index=False)
    _selection_audit(wf, cfg).to_csv(os.path.join(d, "selection_audit.csv"), index=False)
    _regime_comparison(wf["base_oos_r"], wf["ov_oos_r"], cfg, ohlcv).to_csv(os.path.join(d, "regime_comparison.csv"), index=False)
    with open(os.path.join(d, "final_report.txt"), "w", encoding="utf-8") as f:
        f.write(_final_report_text(cfg, wf, ohlcv))
    print(f"\n  [outputs → ./{d}/]")
    print("    comparison_oos.csv, comparison_full.csv, walk_forward_folds_7c.csv")
    print("    dynamic_mode_controls.csv, hawkes_events_intensities.csv, walk_forward_sweeps.csv")
    print("    meta_features_snapshot.csv, feature_importance_fragility.csv, selection_audit.csv")
    print("    regime_comparison.csv, final_report.txt")


def run_mahoraga7c(make_plots_flag: bool = False, run_mode: str = "FULL") -> Dict[str, Any]:
    if not _SKLEARN:
        raise RuntimeError("scikit-learn is required for Mahoraga 7C")
    print("=" * 80)
    print("  MAHORAGA 7C — Hawkes + ML fragility layer over 6.1")
    print("=" * 80)
    print(DISCLAIMER)

    cfg = Mahoraga7CConfig()
    cfg.make_plots_flag = make_plots_flag
    cfg.run_mode = run_mode.upper()
    costs = m6.CostsConfig()
    ucfg = m6.UniverseConfig()
    _ensure_dir(cfg.cache_dir)
    _ensure_dir(cfg.outputs_dir)

    print("\n[1] Downloading data …")
    equity_tickers = sorted(set(list(cfg.universe_static)))
    bench_tickers = [cfg.bench_qqq, cfg.bench_spy, cfg.bench_vix]
    all_tickers = sorted(set(equity_tickers + bench_tickers))
    ohlcv = m6.download_ohlcv(all_tickers, cfg.data_start, cfg.data_end, cfg.cache_dir)

    print("\n[2] Canonical universe engine …")
    asset_registry = m6.build_asset_registry(equity_tickers, cfg, bench_tickers)
    data_quality_report = m6.compute_data_quality_report(ohlcv, equity_tickers, cfg)
    clean_equity = m6.filter_equity_candidates(
        [t for t in equity_tickers if t in ohlcv["close"].columns], asset_registry, data_quality_report, cfg
    )
    universe_schedule, universe_snapshots = m6.build_canonical_universe_schedule(
        ohlcv["close"], ohlcv["volume"], ucfg, clean_equity,
        cfg.data_start, cfg.data_end,
        registry_df=asset_registry, quality_df=data_quality_report,
    )
    print(f"  [universe] {len(universe_schedule)} reconstitution dates built")

    print(f"\n[3] Walk-forward {cfg.variant} over frozen Mahoraga 6.1 base …")
    wf = run_walk_forward_h7c(ohlcv, cfg, costs, universe_schedule)
    save_outputs_h7c(cfg, wf, ohlcv)
    return {"cfg": cfg, "wf": wf, "ohlcv": ohlcv}


if __name__ == "__main__":
    results = run_mahoraga7c(make_plots_flag=False, run_mode="FULL")
