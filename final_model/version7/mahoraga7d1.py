from __future__ import annotations

"""
Mahoraga 7D
===========
BOCPD-lite + conformal risk overlay over frozen Mahoraga 6.1.

Design goals
------------
- Keep Mahoraga 6.1 frozen as the baseline policy.
- Reuse Hawkes-inspired regime signals from Mahoraga 7A/7B.
- Reuse the dual-ML fragility/recovery layer from 7C.
- Add a fast online change-point detector (BOCPD-lite) to react to genuine
  regime shifts.
- Add a split-conformal risk wrapper to cap exposure when forecast tail-loss
  is elevated.
- Stay fast enough for iterative research:
  * no baseline sweep rerun
  * weekly decisions
  * precomputed context table once
  * cheap Stage-1 Hawkes calibration
  * compact Stage-2 ML+policy grid
  * optional outer-fold parallelism
  * fast inner scoring without full summarize() on every combo

Important honesty note
----------------------
This file is executable and research-grade, but two modules are intentionally
compact for speed and robustness:
- BOCPD-lite uses a scalar latent regime score with truncated run-length state.
- Conformal risk control uses a split-conformal upper tail-loss estimate based
  on a simple ridge-style linear predictor.
Both are designed as pragmatic research layers rather than fully general
production libraries.
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

try:
    from scipy.stats import t as student_t
    _SCIPY = True
except Exception:
    student_t = None
    _SCIPY = False

_SCRIPT_DIR = Path(__file__).resolve().parent

DISCLAIMER = r"""
═══════════════════════════════════════════════════════════════════════════════
  MAHORAGA 7D — HAWKES + DUAL-ML LAYER OVER MAHORAGA 6.1
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
class Mahoraga7DConfig(h7.Mahoraga7Config):
    variant: str = "7D"
    outputs_dir: str = "mahoraga7d_outputs"
    plots_dir: str = "mahoraga7d_plots"
    label: str = "MAHORAGA_7D"

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

    # Stage-2 ML + policy calibration
    fragility_quantile_grid: Tuple[float, ...] = (0.15, 0.20)
    fragility_horizon_weeks_grid: Tuple[int, ...] = (2, 4)
    fragility_prob_trigger_grid: Tuple[float, ...] = (0.55, 0.65)
    defensive_scale_grid: Tuple[float, ...] = (0.82, 0.88)
    recovery_floor_grid: Tuple[float, ...] = (0.40, 0.55)
    rel_tilt_grid: Tuple[float, ...] = (0.55, 0.60)
    stress_trigger_q_grid: Tuple[float, ...] = (0.80, 0.90)
    recovery_trigger_q_grid: Tuple[float, ...] = (0.65, 0.80)

    # Labeling utility
    utility_dd_penalty: float = 0.40
    min_train_weeks: int = 80
    inner_val_frac: float = 0.30
    min_class_count: int = 10

    # Policy behavior
    require_crisis_for_recovery_override: bool = False
    rel_signal_quantile: float = 0.58
    target_intervention_rate: float = 0.18
    target_recovery_rate: float = 0.10
    target_recovery_hit_rate: float = 0.40
    recovery_prob_trigger: float = 0.48
    recovery_return_quantile: float = 0.70
    recovery_low_scale_quantile: float = 0.60
    recovery_spread_quantile: float = 0.45
    recovery_breadth_quantile: float = 0.50
    recovery_floor_cap: float = 0.85
    recovery_prob_boost: float = 0.26
    recovery_spread_boost: float = 0.16
    recovery_scale_relief_boost: float = 0.18
    recovery_extra_prob_margin: float = 0.03
    recovery_min_conditions: int = 3
    recovery_relaxed_scale_quantile: float = 0.75
    recovery_relaxed_qqq_buffer_quantile: float = 0.55
    recovery_stress_ratio_gate: float = 0.95
    recovery_mode_full_prob_margin: float = 0.10
    recovery_mode_full_spread_mult: float = 1.10
    recovery_light_blend: float = 0.55
    recovery_full_rel_confirm: bool = False

    chop_abs_return_quantile: float = 0.40
    chop_abs_spread_quantile: float = 0.45
    chop_breadth_low: float = 0.35
    chop_breadth_high: float = 0.65

    # Tail / panic engineering
    panic_vix_quantile: float = 0.80
    panic_corr_quantile: float = 0.80
    panic_dd_quantile: float = 0.25
    panic_min_conditions: int = 2
    panic_defensive_scale: float = 0.35
    stress_defensive_scale_cap: float = 0.82
    cooldown_weeks: int = 2
    cooldown_ext_scale: float = 0.60
    cooldown_release_recovery_prob: float = 0.72

    # 7D — BOCPD-lite regime change layer
    cp_hazard: float = 1.0 / 52.0
    cp_max_run_length: int = 104
    cp_prior_kappa: float = 1.0
    cp_prior_alpha: float = 1.0
    cp_prior_beta: float = 1.0
    cp_prob_quantile: float = 0.90
    cp_severity_quantile: float = 0.80
    cp_stress_budget_mult: float = 0.70
    cp_recovery_budget_boost: float = 0.15

    # 7D — conformal risk wrapper
    conformal_horizon_weeks: int = 2
    conformal_alpha: float = 0.90
    conformal_l2: float = 1e-3
    conformal_budget_low_q: float = 0.55
    conformal_budget_high_q: float = 0.90
    conformal_min_exposure: float = 0.30

    # Overlay score weights
    score_w_sharpe: float = 0.42
    score_w_dd: float = 0.20
    score_w_cagr: float = 0.12
    score_w_recovery_hit: float = 0.14
    score_w_recovery_fire: float = 0.06
    score_w_recovery_cover: float = 0.10
    score_w_cvar: float = 0.14
    score_w_panic: float = 0.16
    score_pen_intervention: float = 0.10
    score_pen_recovery_excess: float = 0.01
    score_pen_recovery_miss: float = 0.12
    score_pen_missed_rebound: float = 0.18
    score_pen_tail: float = 0.12

    # ML model
    rf_n_estimators: int = 250
    rf_max_depth: int = 4
    rf_min_samples_leaf: int = 5
    rf_n_jobs: int = -1


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _weekly_dates(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    return idx.to_series().resample(freq).last().dropna().index


def _load_baseline_folds(cfg: Mahoraga7DConfig) -> pd.DataFrame:
    return h7._load_baseline_folds(cfg)


def _parse_range(s: str) -> Tuple[str, str]:
    return h7._parse_range(s)


def _get_fold_cfg(
    ohlcv: Dict[str, pd.DataFrame],
    base_cfg: Mahoraga7DConfig,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
    fold_row: pd.Series,
) -> Mahoraga7DConfig:
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
    step = max(1, horizon_weeks * 5)
    r_arr = returns_daily.to_numpy(dtype=float)
    eq_arr = equity_daily.to_numpy(dtype=float)
    n = len(idx)
    log1r = np.log1p(np.clip(r_arr, -0.9999, None))
    cumlog = np.concatenate([[0.0], np.cumsum(log1r)])
    pos_arr = idx.searchsorted(weekly_idx.values, side="left")
    vals = np.empty(len(weekly_idx), dtype=float)
    vals[:] = np.nan
    for i, pos in enumerate(pos_arr):
        if pos >= n:
            continue
        end_i = min(n - 1, pos + step)
        if end_i <= pos:
            continue
        ret = np.expm1(cumlog[end_i + 1] - cumlog[pos + 1])
        eq_seg = eq_arr[pos:end_i + 1]
        if len(eq_seg) <= 1:
            dd = 0.0
        else:
            cummax = np.maximum.accumulate(eq_seg)
            dd = float((eq_seg / np.maximum(cummax, 1e-12) - 1.0).min())
        vals[i] = ret - dd_penalty * abs(dd)
    return pd.Series(vals, index=weekly_idx, dtype=float)


def _base_weekly_state(base_bt: Dict[str, Any], decision_freq: str) -> pd.DataFrame:
    idx = _weekly_dates(base_bt["returns_net"].index, decision_freq)
    keys = [
        ("exposure",            "base_exposure",          0.0),
        ("turnover",            "base_turnover",          0.0),
        ("crisis_state",        "crisis_state_weekly",    0.0),
        ("corr_state",          "corr_state_weekly",      0.0),
        ("turb_scale",          "turb_scale_weekly",      1.0),
        ("crisis_scale",        "crisis_scale_weekly",    1.0),
        ("corr_scale",          "corr_scale_weekly",      1.0),
        ("total_scale_target",  "base_total_scale",       1.0),
    ]
    out = pd.DataFrame(index=idx)
    for src_key, col_name, fill in keys:
        s = base_bt.get(src_key, pd.Series(dtype=float))
        out[col_name] = s.resample(decision_freq).last().reindex(idx).ffill().fillna(fill)
    return out


def _build_ml_feature_table(hawkes_df: pd.DataFrame, base_bt: Dict[str, Any], cfg: Mahoraga7DConfig) -> pd.DataFrame:
    wk = _base_weekly_state(base_bt, cfg.decision_freq)
    feat = hawkes_df.join(wk, how="left").sort_index()
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
    "regime_score", "cp_prob", "cp_severity", "cp_direction",
    "conf_pred_loss", "conf_upper_loss", "risk_budget",
]


def _prepare_xy(df: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Safe feature extraction for 7D.

    We reindex instead of strict column selection so the pipeline remains robust
    even if a new feature was not materialized on some code path yet.
    """
    X = df.reindex(columns=FEATURE_COLS)
    X = X.replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med).fillna(0.0)
    valid = y.notna() & np.isfinite(y.astype(float))
    return X.loc[valid], y.loc[valid].astype(int), med


def _prepare_x_with_med(df: pd.DataFrame, med: pd.Series) -> pd.DataFrame:
    X = df.reindex(columns=FEATURE_COLS)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(med).fillna(0.0)
    return X


def _fit_binary_model(X: pd.DataFrame, y: pd.Series, cfg: Mahoraga7DConfig):
    if not _SKLEARN:
        return None
    vc = y.value_counts()
    if len(X) < cfg.min_train_weeks or len(vc) < 2 or vc.min() < cfg.min_class_count:
        return None
    model = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=cfg.random_seed,
        n_jobs=int(cfg.rf_n_jobs),
    )
    model.fit(X, y)
    return model


def _future_window_return(returns_daily: pd.Series, weekly_idx: pd.DatetimeIndex, horizon_weeks: int) -> pd.Series:
    idx = returns_daily.index
    out = pd.Series(np.nan, index=weekly_idx, dtype=float)
    if len(idx) == 0:
        return out
    step = max(1, horizon_weeks * 5)
    r_arr = returns_daily.to_numpy(dtype=float)
    idx_arr = idx
    pos_arr = idx_arr.searchsorted(weekly_idx.values, side="left")
    n = len(idx)
    log1r = np.log1p(np.clip(r_arr, -0.9999, None))
    cumlog = np.concatenate([[0.0], np.cumsum(log1r)])
    vals = np.empty(len(weekly_idx), dtype=float)
    vals[:] = np.nan
    for i, pos in enumerate(pos_arr):
        if pos >= n:
            continue
        end_i = min(n - 1, pos + step)
        if end_i <= pos:
            continue
        vals[i] = np.expm1(cumlog[end_i + 1] - cumlog[pos + 1])
    out = pd.Series(vals, index=weekly_idx, dtype=float)
    return out


def _build_recovery_labels(
    ml_feat_full: pd.DataFrame,
    qqq_future_ret: pd.Series,
    train_idx: pd.DatetimeIndex,
    cfg: Mahoraga7DConfig,
) -> pd.Series:
    train_ret = qqq_future_ret.loc[train_idx].dropna()
    if train_ret.empty:
        return pd.Series(0.0, index=ml_feat_full.index)
    ret_thr = float(np.nanquantile(train_ret, cfg.recovery_return_quantile))
    scale_train = ml_feat_full.loc[train_idx, "base_total_scale"].dropna()
    spread_train = ml_feat_full.loc[train_idx, "intensity_spread"].dropna()
    breadth_train = ml_feat_full.loc[train_idx, "breadth_63d"].dropna()
    qqq21_train = ml_feat_full.loc[train_idx, "qqq_ret_21d"].dropna()

    scale_thr = float(np.nanquantile(scale_train, cfg.recovery_low_scale_quantile)) if scale_train.size else 1.0
    spread_thr = float(np.nanquantile(spread_train, cfg.recovery_spread_quantile)) if spread_train.size else 0.0
    breadth_thr = float(np.nanquantile(breadth_train, cfg.recovery_breadth_quantile)) if breadth_train.size else 0.0
    qqq_buf = float(np.nanquantile(qqq21_train, cfg.recovery_relaxed_qqq_buffer_quantile)) if qqq21_train.size else 0.0

    rel_scale_thr = float(np.nanquantile(scale_train, cfg.recovery_relaxed_scale_quantile)) if scale_train.size else 1.0

    cond_ret = qqq_future_ret >= ret_thr
    cond_scale = (ml_feat_full["base_total_scale"] <= scale_thr) | (ml_feat_full["crisis_state_weekly"] >= 1.0) | (ml_feat_full["base_total_scale"] <= rel_scale_thr)
    cond_spread = (ml_feat_full["intensity_spread"] >= spread_thr) | (ml_feat_full["recovery_intensity"] > ml_feat_full["stress_intensity"])
    cond_breadth = (ml_feat_full["breadth_63d"] >= breadth_thr) | (ml_feat_full["qqq_ret_21d"] >= qqq_buf)

    cond_count = cond_scale.astype(int) + cond_spread.astype(int) + cond_breadth.astype(int)
    y = (cond_ret & (cond_count >= 2)).astype(float)
    return y


def _model_importance(model, cols: List[str]) -> pd.DataFrame:
    if model is None or not hasattr(model, "feature_importances_"):
        return pd.DataFrame({"feature": cols, "importance": np.nan})
    return pd.DataFrame({"feature": cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)


def _safe_z_on_train(s: pd.Series, train_idx: pd.DatetimeIndex) -> pd.Series:
    tr = s.loc[train_idx].dropna()
    mu = float(tr.mean()) if len(tr) else 0.0
    sd = float(tr.std(ddof=1)) if len(tr) > 1 else 1.0
    if not np.isfinite(sd) or sd <= 1e-8:
        sd = 1.0
    out = (s - mu) / sd
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _student_t_pdf_vec(x: float, mu: np.ndarray, kappa: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    if not _SCIPY:
        scale = np.sqrt(np.maximum(beta * (kappa + 1.0) / np.maximum(alpha * kappa, 1e-8), 1e-8))
        z = (x - mu) / np.maximum(scale, 1e-8)
        return np.exp(-0.5 * z * z) / np.maximum(scale, 1e-8)
    nu = np.maximum(2.0 * alpha, 1.0)
    scale = np.sqrt(np.maximum(beta * (kappa + 1.0) / np.maximum(alpha * kappa, 1e-8), 1e-8))
    return student_t.pdf(x, df=nu, loc=mu, scale=np.maximum(scale, 1e-8))


def _bocpd_lite_regime_features(weekly_df: pd.DataFrame, train_idx: pd.DatetimeIndex, cfg: Mahoraga7DConfig) -> pd.DataFrame:
    stress_z = _safe_z_on_train(weekly_df["stress_intensity"], train_idx)
    rec_z = _safe_z_on_train(weekly_df["recovery_intensity"], train_idx)
    corr_z = _safe_z_on_train(weekly_df["avg_corr_21"], train_idx)
    vix_z = _safe_z_on_train(weekly_df["vix_level"], train_idx)
    dd_z = _safe_z_on_train(-weekly_df["qqq_drawdown"], train_idx)
    breadth_stress = _safe_z_on_train(-weekly_df["breadth_63d"], train_idx)

    regime_score = (
        0.30 * stress_z +
        0.22 * corr_z +
        0.18 * vix_z +
        0.18 * dd_z +
        0.08 * breadth_stress -
        0.18 * rec_z
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    x = regime_score.to_numpy(dtype=float)
    n = len(x)
    max_r = max(8, int(cfg.cp_max_run_length))
    H = float(np.clip(cfg.cp_hazard, 1e-4, 0.5))
    train_x = regime_score.loc[train_idx].dropna()
    mu0 = float(train_x.mean()) if len(train_x) else 0.0
    var0 = float(train_x.var(ddof=1)) if len(train_x) > 1 else 1.0
    if not np.isfinite(var0) or var0 <= 1e-8:
        var0 = 1.0

    R = np.array([1.0], dtype=float)
    mu = np.array([mu0], dtype=float)
    kappa = np.array([max(cfg.cp_prior_kappa, 1e-3)], dtype=float)
    alpha = np.array([max(cfg.cp_prior_alpha, 1e-3)], dtype=float)
    beta = np.array([max(cfg.cp_prior_beta * var0, 1e-3)], dtype=float)

    cp_prob = np.zeros(n, dtype=float)
    cp_sev = np.zeros(n, dtype=float)
    pred_mean_prev = mu0
    pred_std0 = float(np.sqrt(var0))
    _prior_mu0 = mu0
    _prior_kappa0 = max(cfg.cp_prior_kappa, 1e-3)
    _prior_alpha0 = max(cfg.cp_prior_alpha, 1e-3)
    _prior_beta0 = max(cfg.cp_prior_beta * var0, 1e-3)

    for t, xt in enumerate(x):
        pred = _student_t_pdf_vec(float(xt), mu, kappa, alpha, beta)
        growth = pred * R * (1.0 - H)
        cp_mass = float((pred * R).sum() * H)
        new_len = min(len(growth) + 1, max_r + 1)
        R_new = np.empty(new_len, dtype=float)
        R_new[0] = cp_mass
        g_take = min(len(growth), max_r)
        R_new[1:g_take + 1] = growth[:g_take]
        z = R_new.sum()
        if not np.isfinite(z) or z <= 0:
            R_new[:] = 0.0
            R_new[0] = 1.0
        else:
            R_new /= z
        cp_prob[t] = float(R_new[0])
        cp_sev[t] = abs((xt - pred_mean_prev) / max(pred_std0, 1e-8))

        keep = min(len(mu), max_r)
        mu_new = np.empty(new_len, dtype=float)
        kappa_new = np.empty(new_len, dtype=float)
        alpha_new = np.empty(new_len, dtype=float)
        beta_new = np.empty(new_len, dtype=float)
        mu_new[0] = _prior_mu0
        kappa_new[0] = _prior_kappa0
        alpha_new[0] = _prior_alpha0
        beta_new[0] = _prior_beta0
        if keep > 0:
            kk = kappa[:keep]
            mm = mu[:keep]
            aa = alpha[:keep]
            bb = beta[:keep]
            kk1 = kk + 1.0
            diff = xt - mm
            mu_new[1:keep + 1] = (kk * mm + xt) / kk1
            kappa_new[1:keep + 1] = kk1
            alpha_new[1:keep + 1] = aa + 0.5
            beta_new[1:keep + 1] = bb + (kk * diff * diff) / (2.0 * kk1)

        pred_mean_prev = float(np.dot(R_new[:len(mu_new)], mu_new[:len(R_new)]))
        R, mu, kappa, alpha, beta = R_new, mu_new, kappa_new, alpha_new, beta_new

    delta = regime_score.diff().fillna(0.0)
    direction = np.where((delta <= 0) | (weekly_df["recovery_intensity"] > weekly_df["stress_intensity"]), 1.0, -1.0)
    return pd.DataFrame({
        "regime_score": regime_score,
        "cp_prob": cp_prob,
        "cp_severity": cp_sev,
        "cp_direction": direction,
    }, index=weekly_df.index)


def _ridge_split_conformal_risk(weekly_df: pd.DataFrame, returns_daily: pd.Series, train_idx: pd.DatetimeIndex, cfg: Mahoraga7DConfig) -> pd.DataFrame:
    feat_cols = [
        "regime_score", "cp_prob", "cp_severity", "stress_intensity",
        "recovery_intensity", "avg_corr_21", "vix_level", "qqq_drawdown",
        "qqq_vol_21", "base_total_scale", "breadth_63d",
    ]
    target = (-_future_window_return(returns_daily, weekly_df.index, cfg.conformal_horizon_weeks)).clip(lower=0.0)
    X = weekly_df[feat_cols].copy().replace([np.inf, -np.inf], np.nan)
    med = X.loc[train_idx].median(numeric_only=True)
    X = X.fillna(med).replace([np.inf, -np.inf], np.nan).fillna(med)
    mu = X.loc[train_idx].mean(numeric_only=True)
    sd = X.loc[train_idx].std(ddof=1, numeric_only=True).replace(0.0, 1.0).fillna(1.0)
    Xs = (X - mu) / sd
    mask = target.loc[train_idx].notna()
    if int(mask.sum()) < max(24, cfg.min_train_weeks // 2):
        upper = pd.Series(0.0, index=weekly_df.index)
        risk_budget = pd.Series(1.0, index=weekly_df.index)
        return pd.DataFrame({"conf_pred_loss": 0.0, "conf_upper_loss": upper, "risk_budget": risk_budget}, index=weekly_df.index)
    Xtr = Xs.loc[train_idx].loc[mask].to_numpy(dtype=float)
    ytr = target.loc[train_idx].loc[mask].to_numpy(dtype=float)
    Xtr_i = np.column_stack([np.ones(len(Xtr)), Xtr])
    ridge = np.eye(Xtr_i.shape[1]) * float(cfg.conformal_l2)
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(Xtr_i.T @ Xtr_i + ridge, Xtr_i.T @ ytr)
    Xall = np.column_stack([np.ones(len(Xs)), Xs.to_numpy(dtype=float)])
    pred = Xall @ beta
    resid = ytr - (Xtr_i @ beta)
    q = float(np.nanquantile(resid, cfg.conformal_alpha)) if len(resid) else 0.0
    q = max(0.0, q)
    upper = np.clip(pred + q, 0.0, None)
    upper_train = pd.Series(upper, index=weekly_df.index).loc[train_idx].dropna()
    lo = float(np.nanquantile(upper_train, cfg.conformal_budget_low_q)) if len(upper_train) else 0.0
    hi = float(np.nanquantile(upper_train, cfg.conformal_budget_high_q)) if len(upper_train) else max(lo + 1e-6, 1.0)
    if hi <= lo:
        hi = lo + 1e-6
    interp = 1.0 - (upper - lo) / (hi - lo)
    risk_budget = cfg.conformal_min_exposure + (1.0 - cfg.conformal_min_exposure) * np.clip(interp, 0.0, 1.0)
    risk_budget = np.clip(risk_budget, cfg.conformal_min_exposure, 1.0)
    return pd.DataFrame({
        "conf_pred_loss": pred,
        "conf_upper_loss": upper,
        "risk_budget": risk_budget,
    }, index=weekly_df.index)


def _precompute_fixed_thresholds(
    train_weekly_ref: pd.DataFrame,
    cfg: Mahoraga7DConfig,
) -> Dict[str, float]:
    """
    7D: reusable thresholds for policy evaluation inside the fold.
    Includes panic, change-point and conformal-risk diagnostics.
    """
    rel_train     = train_weekly_ref["rel_top_mean"].dropna()
    spread_train  = train_weekly_ref["intensity_spread"].dropna()
    scale_train   = train_weekly_ref["base_total_scale"].dropna()
    breadth_train = train_weekly_ref["breadth_63d"].dropna()
    qqq21_train   = train_weekly_ref["qqq_ret_21d"].dropna()
    vix_train     = train_weekly_ref["vix_level"].dropna()
    corr_train    = train_weekly_ref["avg_corr_21"].dropna()
    dd_train      = train_weekly_ref["qqq_drawdown"].dropna()
    cp_prob_train = train_weekly_ref.get("cp_prob", pd.Series(dtype=float)).dropna()
    cp_sev_train  = train_weekly_ref.get("cp_severity", pd.Series(dtype=float)).dropna()
    conf_train    = train_weekly_ref.get("conf_upper_loss", pd.Series(dtype=float)).dropna()
    return {
        "rel_thr": float(np.nanquantile(rel_train, cfg.rel_signal_quantile)) if rel_train.size else np.inf,
        "spread_pos_thr": float(np.nanquantile(spread_train, cfg.recovery_spread_quantile)) if spread_train.size else 0.0,
        "scale_low_thr": float(np.nanquantile(scale_train, cfg.recovery_low_scale_quantile)) if scale_train.size else 1.0,
        "scale_relaxed_thr": float(np.nanquantile(scale_train, cfg.recovery_relaxed_scale_quantile)) if scale_train.size else 1.0,
        "breadth_rec_thr": float(np.nanquantile(breadth_train, cfg.recovery_breadth_quantile)) if breadth_train.size else cfg.chop_breadth_high,
        "chop_ret_thr": float(np.nanquantile(qqq21_train.abs(), cfg.chop_abs_return_quantile)) if qqq21_train.size else 0.0,
        "chop_spread_thr": float(np.nanquantile(spread_train.abs(), cfg.chop_abs_spread_quantile)) if spread_train.size else 0.0,
        "qqq_rec_buf": float(np.nanquantile(qqq21_train, cfg.recovery_relaxed_qqq_buffer_quantile)) if qqq21_train.size else 0.0,
        "panic_vix_thr": float(np.nanquantile(vix_train, cfg.panic_vix_quantile)) if vix_train.size else 24.0,
        "panic_corr_thr": float(np.nanquantile(corr_train, cfg.panic_corr_quantile)) if corr_train.size else 0.75,
        "panic_dd_thr": float(np.nanquantile(dd_train, cfg.panic_dd_quantile)) if dd_train.size else -0.10,
        "cp_prob_thr": float(np.nanquantile(cp_prob_train, cfg.cp_prob_quantile)) if cp_prob_train.size else 1.0,
        "cp_sev_thr": float(np.nanquantile(cp_sev_train, cfg.cp_severity_quantile)) if cp_sev_train.size else np.inf,
        "conf_hi_thr": float(np.nanquantile(conf_train, cfg.conformal_budget_high_q)) if conf_train.size else np.inf,
    }


def _policy_from_dual_probs_vec(
    weekly_df: pd.DataFrame,
    probs_frag: pd.Series,
    probs_rec: pd.Series,
    prob_trigger: float,
    recovery_prob_trigger: float,
    defensive_scale: float,
    recovery_floor: float,
    rel_tilt: float,
    stress_thr: float,
    rec_thr: float,
    fixed_thr: Dict[str, float],
    cfg: Mahoraga7DConfig,
    crisis_state_daily: pd.Series,
) -> pd.DataFrame:
    idx = weekly_df.index

    pf       = probs_frag.reindex(idx).fillna(0.0).to_numpy(dtype=float)
    pr       = probs_rec.reindex(idx).fillna(0.0).to_numpy(dtype=float)
    stress   = weekly_df["stress_intensity"].to_numpy(dtype=float)
    rec      = weekly_df["recovery_intensity"].to_numpy(dtype=float)
    spread   = weekly_df["intensity_spread"].to_numpy(dtype=float)
    bscale   = weekly_df["base_total_scale"].to_numpy(dtype=float)
    qqq21    = weekly_df["qqq_ret_21d"].to_numpy(dtype=float)
    breadth  = weekly_df["breadth_63d"].to_numpy(dtype=float)
    rel_top  = weekly_df["rel_top_mean"].to_numpy(dtype=float)
    rm_mom   = weekly_df["rel_minus_mom"].to_numpy(dtype=float)
    rm_tr    = weekly_df["rel_minus_trend"].to_numpy(dtype=float)
    avg_corr = weekly_df["avg_corr_21"].to_numpy(dtype=float)
    vix      = weekly_df["vix_level"].to_numpy(dtype=float)
    qqq_dd   = weekly_df["qqq_drawdown"].to_numpy(dtype=float)
    cp_prob  = weekly_df.get("cp_prob", pd.Series(0.0, index=idx)).to_numpy(dtype=float)
    cp_sev   = weekly_df.get("cp_severity", pd.Series(0.0, index=idx)).to_numpy(dtype=float)
    cp_dir   = weekly_df.get("cp_direction", pd.Series(0.0, index=idx)).to_numpy(dtype=float)
    risk_budget_weekly = weekly_df.get("risk_budget", pd.Series(1.0, index=idx)).clip(cfg.conformal_min_exposure, 1.0).to_numpy(dtype=float)
    conf_upper = weekly_df.get("conf_upper_loss", pd.Series(0.0, index=idx)).to_numpy(dtype=float)
    crisis_on = crisis_state_daily.reindex(idx).fillna(0.0).to_numpy(dtype=float) >= 1.0

    rel_thr_f         = fixed_thr["rel_thr"]
    spread_pos_thr    = fixed_thr["spread_pos_thr"]
    scale_low_thr     = fixed_thr["scale_low_thr"]
    scale_relaxed_thr = fixed_thr["scale_relaxed_thr"]
    breadth_rec_thr   = fixed_thr["breadth_rec_thr"]
    chop_ret_thr      = fixed_thr["chop_ret_thr"]
    chop_spread_thr   = fixed_thr["chop_spread_thr"]
    qqq_rec_buf       = fixed_thr["qqq_rec_buf"]
    panic_vix_thr     = fixed_thr["panic_vix_thr"]
    panic_corr_thr    = fixed_thr["panic_corr_thr"]
    panic_dd_thr      = fixed_thr["panic_dd_thr"]
    cp_prob_thr       = fixed_thr["cp_prob_thr"]
    cp_sev_thr        = fixed_thr["cp_sev_thr"]
    conf_hi_thr       = fixed_thr["conf_hi_thr"]

    cp_stress = (cp_prob >= cp_prob_thr) & (cp_sev >= cp_sev_thr) & (cp_dir < 0)
    cp_recover = (cp_prob >= cp_prob_thr) & (cp_sev >= cp_sev_thr) & (cp_dir > 0)

    chop = (
        (np.abs(qqq21) <= chop_ret_thr)
        & (np.abs(spread) <= chop_spread_thr)
        & (breadth >= cfg.chop_breadth_low)
        & (breadth <= cfg.chop_breadth_high)
    )
    rec_prob_hi = pr >= (recovery_prob_trigger + cfg.recovery_extra_prob_margin)

    cond_rec = (rec >= rec_thr) | cp_recover
    cond_spread = (spread >= spread_pos_thr) | (rec > stress)
    cond_scale = (bscale <= scale_low_thr) | crisis_on | (bscale <= scale_relaxed_thr)
    cond_breadth = (breadth >= breadth_rec_thr) | (qqq21 >= qqq_rec_buf)
    cond_rel = (rel_top >= rel_thr_f) | (rm_mom > 0) | (rm_tr > 0)
    cond_count = cond_rec.astype(int) + cond_spread.astype(int) + cond_scale.astype(int) + cond_breadth.astype(int) + cond_rel.astype(int)

    recover_opp = (pr >= recovery_prob_trigger) & ~chop & (cond_count >= cfg.recovery_min_conditions)
    can_frag = ((pf >= prob_trigger) | cp_stress) & ~(chop & ~crisis_on)

    panic_count = (
        (vix >= panic_vix_thr).astype(int)
        + (avg_corr >= panic_corr_thr).astype(int)
        + (qqq_dd <= panic_dd_thr).astype(int)
        + (stress >= stress_thr).astype(int)
        + cp_stress.astype(int)
    )
    panic_mode = crisis_on & (panic_count >= cfg.panic_min_conditions)

    allow_recover = recover_opp & (
        crisis_on
        | rec_prob_hi
        | cp_recover
        | ((rec >= stress * cfg.recovery_stress_ratio_gate) & (spread >= 0.0))
        | ((bscale <= scale_low_thr) & (qqq21 >= qqq_rec_buf))
    ) & (~cp_stress) & (conf_upper < conf_hi_thr)

    strong_recover = allow_recover & (
        (pr >= recovery_prob_trigger + cfg.recovery_mode_full_prob_margin)
        & ((spread >= spread_pos_thr * cfg.recovery_mode_full_spread_mult) | (rec > stress) | cp_recover)
        & ((breadth >= breadth_rec_thr) | (qqq21 >= qqq_rec_buf) | (rel_top >= rel_thr_f))
    )
    if cfg.recovery_full_rel_confirm:
        strong_recover = strong_recover & ((rm_mom > 0) | (rm_tr > 0))

    rel_confirm = (rel_top >= rel_thr_f) & ((rm_mom > 0) | (rm_tr > 0)) & ~crisis_on & ~panic_mode & ~cp_stress
    stress_dominant = (stress >= stress_thr) & ((stress > rec) | (spread < spread_pos_thr))

    frag_action = np.where(rel_confirm & ~stress_dominant, "REL_TILT", "DEFENSIVE_LIGHT")
    action = np.where(
        allow_recover,
        "RECOVERY_OVERRIDE",
        np.where(can_frag | panic_mode | stress_dominant, frag_action, "BASELINE"),
    )

    cooldown_active = np.zeros(len(idx), dtype=bool)
    cooldown = 0
    for i in range(len(idx)):
        trigger = bool(panic_mode[i] or cp_stress[i]) and action[i] != "RECOVERY_OVERRIDE"
        release = bool(pr[i] >= cfg.cooldown_release_recovery_prob and strong_recover[i])
        if release:
            cooldown = 0
        if trigger:
            cooldown = max(cooldown, int(cfg.cooldown_weeks))
        elif cooldown > 0:
            cooldown_active[i] = True
            cooldown -= 1

    action = np.where(cooldown_active & (action == "BASELINE") & ~strong_recover, "DEFENSIVE_LIGHT", action)

    defensive_scale_eff = np.where(
        panic_mode | cp_stress,
        cfg.panic_defensive_scale,
        np.minimum(defensive_scale, cfg.stress_defensive_scale_cap),
    )
    ext_scale_arr = np.where(action == "DEFENSIVE_LIGHT", defensive_scale_eff, 1.0)
    ext_scale_arr = np.where(cooldown_active & (action != "RECOVERY_OVERRIDE"), np.minimum(ext_scale_arr, cfg.cooldown_ext_scale), ext_scale_arr)

    denom_prob = max(1e-6, 1.0 - recovery_prob_trigger)
    denom_spread = max(1e-6, max(abs(spread_pos_thr), 1.0))
    denom_scale = max(1e-6, max(scale_relaxed_thr, 1.0))
    denom_momentum = max(1e-6, max(abs(qqq_rec_buf), 1.0))

    prob_excess = np.clip(pr - recovery_prob_trigger, 0.0, None) / denom_prob
    spread_excess = np.clip(spread - spread_pos_thr, 0.0, None) / denom_spread
    scale_gap = np.clip(scale_relaxed_thr - bscale, 0.0, None) / denom_scale
    momentum_relief = np.clip(qqq21 - qqq_rec_buf, 0.0, None) / denom_momentum

    rec_scale_raw = (
        recovery_floor
        + cfg.recovery_prob_boost * np.minimum(1.0, prob_excess)
        + cfg.recovery_spread_boost * np.minimum(1.0, spread_excess)
        + cfg.recovery_scale_relief_boost * np.minimum(1.0, scale_gap)
        + 0.08 * np.minimum(1.0, momentum_relief)
    )
    rec_scale_raw = np.where(cp_recover, rec_scale_raw + cfg.cp_recovery_budget_boost, rec_scale_raw)
    rec_scale_raw = np.clip(rec_scale_raw, 0.0, cfg.recovery_floor_cap)
    rec_scale_arr = np.where(
        action == "RECOVERY_OVERRIDE",
        np.where(strong_recover, rec_scale_raw, recovery_floor + cfg.recovery_light_blend * (rec_scale_raw - recovery_floor)),
        0.0,
    )

    risk_budget = np.clip(risk_budget_weekly, cfg.conformal_min_exposure, 1.0)
    risk_budget = np.where(cp_stress & (conf_upper >= conf_hi_thr), np.maximum(cfg.conformal_min_exposure, risk_budget * cfg.cp_stress_budget_mult), risk_budget)
    risk_budget = np.where(cp_recover & strong_recover, np.minimum(1.0, np.maximum(risk_budget, recovery_floor + cfg.cp_recovery_budget_boost)), risk_budget)
    risk_budget = np.where(panic_mode, np.minimum(risk_budget, cfg.panic_defensive_scale), risk_budget)

    out = pd.DataFrame(index=idx)
    out["action"] = action
    out["ext_scale"] = ext_scale_arr
    out["recovery_override_scale"] = rec_scale_arr
    out["recover_opp"] = recover_opp.astype(float)
    out["recover_cond_count"] = cond_count.astype(float)
    out["panic_mode"] = panic_mode.astype(float)
    out["cooldown_active"] = cooldown_active.astype(float)
    out["cp_stress"] = cp_stress.astype(float)
    out["cp_recover"] = cp_recover.astype(float)
    out["cp_prob"] = cp_prob
    out["cp_severity"] = cp_sev
    out["cp_direction"] = cp_dir
    out["risk_budget"] = risk_budget
    out["conf_upper_loss"] = conf_upper
    return out


def _policy_from_dual_probs(
    weekly_df: pd.DataFrame,
    probs_frag: pd.Series,
    probs_rec: pd.Series,
    prob_trigger: float,
    recovery_prob_trigger: float,
    defensive_scale: float,
    recovery_floor: float,
    rel_tilt: float,
    stress_trigger_q: float,
    recovery_trigger_q: float,
    cfg: Mahoraga7DConfig,
    crisis_state_daily: pd.Series,
    train_weekly_ref: pd.DataFrame,
) -> pd.DataFrame:
    """
    Thin wrapper for _run_single_fold (test phase — called once per fold).
    Computes thresholds on the fly and delegates to the vectorized implementation.
    """
    stress_train = train_weekly_ref["stress_intensity"].dropna()
    rec_train = train_weekly_ref["recovery_intensity"].dropna()
    stress_thr = float(np.nanquantile(stress_train, stress_trigger_q)) if stress_train.size else np.inf
    rec_thr = float(np.nanquantile(rec_train, recovery_trigger_q)) if rec_train.size else np.inf
    fixed_thr = _precompute_fixed_thresholds(train_weekly_ref, cfg)
    return _policy_from_dual_probs_vec(
        weekly_df, probs_frag, probs_rec,
        prob_trigger, recovery_prob_trigger,
        defensive_scale, recovery_floor, rel_tilt,
        stress_thr, rec_thr, fixed_thr,
        cfg, crisis_state_daily,
    )


def _forward_positive_capture(
    policy_weekly: pd.DataFrame,
    qqq_future: pd.Series,
) -> Tuple[float, float, float]:
    pos = qqq_future.reindex(policy_weekly.index).fillna(0.0).clip(lower=0.0)
    opp = policy_weekly.get("recover_opp", pd.Series(0.0, index=policy_weekly.index)).astype(bool)
    fired = (policy_weekly["action"] == "RECOVERY_OVERRIDE")
    missed = float(pos[opp & ~fired].sum())
    captured = float(pos[opp & fired].sum())
    total = captured + missed
    capture_rate = float(captured / total) if total > 0 else 0.0
    return missed, captured, capture_rate


def _panic_summary(
    returns_daily: pd.Series,
    base_returns_daily: pd.Series,
    panic_mask_daily: pd.Series,
    cfg: Mahoraga7DConfig,
) -> Dict[str, float]:
    mask = panic_mask_daily.reindex(returns_daily.index).fillna(0.0) >= 1.0
    r_ov = returns_daily.loc[mask]
    r_bs = base_returns_daily.reindex(r_ov.index).fillna(0.0)
    if len(r_ov) < 5:
        return {"panic_days": int(mask.sum()), "panic_sharpe_delta": 0.0, "panic_cvar_delta": 0.0}
    sh_ov = m6.sharpe(r_ov, cfg.rf_annual, cfg.trading_days)
    sh_bs = m6.sharpe(r_bs, cfg.rf_annual, cfg.trading_days)
    cv_ov = m6.cvar95(r_ov)
    cv_bs = m6.cvar95(r_bs)
    return {
        "panic_days": int(mask.sum()),
        "panic_sharpe_delta": float(sh_ov - sh_bs),
        "panic_cvar_delta": float(abs(cv_bs) - abs(cv_ov)),
        "panic_sharpe": float(sh_ov),
        "panic_cvar": float(cv_ov),
    }


def _fast_metrics(returns_daily: pd.Series, equity_daily: pd.Series, cfg: Mahoraga7DConfig, label: str) -> Dict[str, float]:
    r = returns_daily.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    eq = equity_daily.reindex(r.index).ffill().dropna()
    if len(r) == 0:
        return {"Label": label, "CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0, "CVaR_5": 0.0}
    cagr = m6.annualize(r, cfg.trading_days)
    shp = m6.sharpe(r, cfg.rf_annual, cfg.trading_days)
    cvar = m6.cvar95(r)
    eqv = eq.to_numpy(dtype=float)
    runmax = np.maximum.accumulate(eqv)
    maxdd = float(np.min(eqv / np.maximum(runmax, 1e-12) - 1.0)) if len(eqv) else 0.0
    return {"Label": label, "CAGR": float(cagr), "Sharpe": float(shp), "MaxDD": maxdd, "CVaR_5": float(cvar)}


def _score_overlay(
    base_sum: Dict[str, float],
    ov_sum: Dict[str, float],
    intervention_rate: float,
    recovery_rate: float,
    recovery_opp_rate: float,
    recovery_hit_rate: float,
    recovery_capture_rate: float,
    missed_rebound: float,
    panic_sharpe_delta: float,
    panic_cvar_delta: float,
    cfg: Mahoraga7DConfig,
) -> float:
    delta_sh = ov_sum["Sharpe"] - base_sum["Sharpe"]
    delta_dd = abs(base_sum["MaxDD"]) - abs(ov_sum["MaxDD"])
    delta_cagr = ov_sum["CAGR"] - base_sum["CAGR"]
    delta_cvar = abs(base_sum["CVaR_5"]) - abs(ov_sum["CVaR_5"])
    pen_int = max(0.0, intervention_rate - cfg.target_intervention_rate)
    pen_rec = max(0.0, recovery_rate - cfg.target_recovery_rate)
    opp_gap = max(0.0, recovery_opp_rate - recovery_rate)
    tail_pen = max(0.0, abs(ov_sum["CVaR_5"]) - abs(base_sum["CVaR_5"]))
    hit_bonus = recovery_hit_rate if recovery_opp_rate > 0 else 0.0
    fire_bonus = min(recovery_rate, cfg.target_recovery_rate)
    return float(
        cfg.score_w_sharpe * delta_sh
        + cfg.score_w_dd * delta_dd
        + cfg.score_w_cagr * delta_cagr
        + cfg.score_w_cvar * delta_cvar
        + cfg.score_w_panic * (0.65 * panic_sharpe_delta + 0.35 * panic_cvar_delta)
        + cfg.score_w_recovery_hit * hit_bonus
        + cfg.score_w_recovery_fire * fire_bonus
        + cfg.score_w_recovery_cover * recovery_capture_rate
        - cfg.score_pen_intervention * pen_int
        - cfg.score_pen_recovery_excess * pen_rec
        - cfg.score_pen_recovery_miss * opp_gap
        - cfg.score_pen_missed_rebound * missed_rebound
        - cfg.score_pen_tail * tail_pen
    )


def _score_from_weights(
    w_exec_1x: pd.DataFrame,
    vol_sc: pd.Series,
    ctx: Dict[str, Any],
    policy_daily: pd.DataFrame,
    costs: m6.CostsConfig,
    cfg: Mahoraga7DConfig,
    inner_val_start: str,
    inner_val_end: str,
    base_sum: Dict[str, float],
    policy_weekly: pd.DataFrame,
    base_r_window: pd.Series,
    qqq_future_window: pd.Series,
) -> Tuple[float, Dict[str, float], float, float, float, float, float, float, float, float]:
    idx = ctx["idx"]
    rets = ctx["rets"]
    crisis_scale = ctx["crisis_scale"]
    turb_scale = ctx["turb_scale"]
    corr_scale = ctx["corr_scale"]

    ext_scale = policy_daily["ext_scale"].reindex(idx).ffill().fillna(1.0).clip(0.0, 1.0)
    rec_override = policy_daily["recovery_override_scale"].reindex(idx).ffill().fillna(0.0).clip(0.0, 1.0)
    risk_budget = policy_daily.get("risk_budget", pd.Series(1.0, index=policy_daily.index)).reindex(idx).ffill().fillna(1.0).clip(cfg.conformal_min_exposure, cfg.max_exposure)

    crisis_eff = pd.Series(np.maximum(crisis_scale.values, rec_override.values), index=idx).clip(0.0, 1.0)
    cap_raw = (crisis_eff * turb_scale * corr_scale * ext_scale).clip(0.0, cfg.max_exposure)
    cap = pd.Series(np.minimum(cap_raw.values, risk_budget.values), index=idx).clip(0.0, cfg.max_exposure)
    tgt_sc = pd.Series(np.minimum(vol_sc.values, cap.values), index=idx).clip(0.0, cfg.max_exposure)
    exec_sc = tgt_sc.shift(1).fillna(0.0)
    w_exec = w_exec_1x.mul(exec_sc, axis=0)
    _, tc = m6._costs(w_exec, costs)
    port_net = ((w_exec * rets).sum(axis=1) - tc).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    equity = cfg.capital_initial * (1.0 + port_net).cumprod()

    ov_r = port_net.loc[inner_val_start:inner_val_end]
    ov_eq = equity.loc[inner_val_start:inner_val_end]
    ov_sum = _fast_metrics(ov_r, ov_eq, cfg, "OV_INNER")

    intervention_rate = float((policy_weekly["action"] != "BASELINE").mean())
    recovery_rate = float((policy_weekly["action"] == "RECOVERY_OVERRIDE").mean())
    recovery_opp_rate = float(policy_weekly.get("recover_opp", pd.Series(0.0, index=policy_weekly.index)).mean())
    recovery_hit_rate = float(recovery_rate / recovery_opp_rate) if recovery_opp_rate > 0 else 0.0
    missed_rebound, _captured_rebound, recovery_capture_rate = _forward_positive_capture(policy_weekly, qqq_future_window)
    panic_metrics = _panic_summary(
        ov_r,
        base_r_window.reindex(ov_r.index).fillna(0.0),
        policy_daily.get("panic_mode", pd.Series(0.0, index=policy_daily.index)).loc[inner_val_start:inner_val_end],
        cfg,
    )
    score = _score_overlay(
        base_sum,
        ov_sum,
        intervention_rate,
        recovery_rate,
        recovery_opp_rate,
        recovery_hit_rate,
        recovery_capture_rate,
        missed_rebound,
        panic_metrics.get("panic_sharpe_delta", 0.0),
        panic_metrics.get("panic_cvar_delta", 0.0),
        cfg,
    )
    return (
        score,
        ov_sum,
        intervention_rate,
        recovery_rate,
        recovery_opp_rate,
        recovery_hit_rate,
        missed_rebound,
        recovery_capture_rate,
        panic_metrics.get("panic_sharpe", 0.0),
        panic_metrics.get("panic_days", 0.0),
    )




def _expand_weekly_policy_to_daily(policy_weekly: pd.DataFrame, daily_idx: pd.DatetimeIndex) -> pd.DataFrame:
    cols_num = {
        "ext_scale": 1.0,
        "recovery_override_scale": 0.0,
        "panic_mode": 0.0,
        "risk_budget": 1.0,
    }
    out = pd.DataFrame(index=daily_idx)
    if policy_weekly.empty:
        out["action"] = "BASELINE"
        for c, v in cols_num.items():
            out[c] = v
        return out
    widx = pd.DatetimeIndex(policy_weekly.index)
    didx = pd.DatetimeIndex(daily_idx)
    pos = np.searchsorted(widx.values, didx.values, side="right") - 1
    valid = pos >= 0
    action_src = policy_weekly.get("action", pd.Series("BASELINE", index=widx)).astype(object).to_numpy()
    action = np.full(len(didx), "BASELINE", dtype=object)
    action[valid] = action_src[pos[valid]]
    out["action"] = action
    for c, default in cols_num.items():
        src = policy_weekly.get(c, pd.Series(default, index=widx)).to_numpy(dtype=float)
        arr = np.full(len(didx), float(default), dtype=float)
        arr[valid] = src[pos[valid]]
        out[c] = arr
    return out


def _calibrate_stage1_hawkes(
    feat_full: pd.DataFrame,
    train_start: str,
    inner_train_end: str,
    inner_val_start: str,
    inner_val_end: str,
    cfg_fold: Mahoraga7DConfig,
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
    cfg_fold: Mahoraga7DConfig,
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
            "recovery_prob_trigger": cfg_fold.recovery_prob_trigger,
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

    s1_best, s1_df = _calibrate_stage1_hawkes(feat_full, train_start, inner_train_end, inner_val_start, inner_val_end, cfg_fold, crisis_state_weekly)

    hawkes_df_fixed, _thr = h7._build_hawkes_signals(feat_full, s1_best["stress_q"], s1_best["recovery_q"], s1_best["decay"], s1_best["stress_scale"], s1_best["recovery_scale"], feat_full.loc[train_start:inner_train_end])
    ml_feat_full = _build_ml_feature_table(hawkes_df_fixed, base_bt, cfg_fold)
    wk_idx = ml_feat_full.index

    inner_train_idx = wk_idx[(wk_idx >= pd.Timestamp(train_start)) & (wk_idx <= pd.Timestamp(inner_train_end))]
    inner_val_idx = wk_idx[(wk_idx >= pd.Timestamp(inner_val_start)) & (wk_idx <= pd.Timestamp(inner_val_end))]

    # 7D regime/risk features must exist during calibration too, not only in test.
    cp_df = _bocpd_lite_regime_features(ml_feat_full, inner_train_idx, cfg_fold)
    ml_feat_full = ml_feat_full.join(cp_df)
    conf_df = _ridge_split_conformal_risk(ml_feat_full, base_bt["returns_net"], inner_train_idx, cfg_fold)
    ml_feat_full = ml_feat_full.join(conf_df)

    utility_cache: Dict[int, pd.Series] = {}
    qqq_future_cache: Dict[int, pd.Series] = {}
    ctx = h7._precompute_overlay_context(ohlcv, cfg_fold, costs, universe_schedule)
    score_rel_cache: Dict[float, pd.DataFrame] = {}

    s2_rows = []
    best_score = -np.inf
    best_params = None

    for horizon_weeks, frag_q in iproduct(cfg_fold.fragility_horizon_weeks_grid, cfg_fold.fragility_quantile_grid):
        if len(inner_train_idx) < cfg_fold.min_train_weeks or len(inner_val_idx) < 5:
            continue
        if horizon_weeks not in utility_cache:
            utility_cache[horizon_weeks] = _future_window_utility(base_bt["returns_net"], base_bt["equity"], wk_idx, horizon_weeks, cfg_fold.utility_dd_penalty)
        utility = utility_cache[horizon_weeks]
        if horizon_weeks not in qqq_future_cache:
            qqq_future_cache[horizon_weeks] = _future_window_return(base_bt["bench"]["QQQ_r"], wk_idx, horizon_weeks)
        qqq_future = qqq_future_cache[horizon_weeks]

        thr_utility = float(np.nanquantile(utility.loc[inner_train_idx].dropna(), frag_q)) if utility.loc[inner_train_idx].dropna().size else np.nan
        y_frag = (utility <= thr_utility).astype(float)
        y_rec = _build_recovery_labels(ml_feat_full, qqq_future, inner_train_idx, cfg_fold)

        X_train_frag, y_train_frag, med_frag = _prepare_xy(ml_feat_full.loc[inner_train_idx], y_frag.loc[inner_train_idx])
        model_frag = _fit_binary_model(X_train_frag, y_train_frag, cfg_fold)
        if model_frag is None:
            continue
        X_val_frag = _prepare_x_with_med(ml_feat_full.loc[inner_val_idx], med_frag)
        probs_frag_val = pd.Series(model_frag.predict_proba(X_val_frag)[:, 1], index=inner_val_idx)

        X_train_rec, y_train_rec, med_rec = _prepare_xy(ml_feat_full.loc[inner_train_idx], y_rec.loc[inner_train_idx])
        model_rec = _fit_binary_model(X_train_rec, y_train_rec, cfg_fold)
        if model_rec is None:
            probs_rec_val = pd.Series(0.0, index=inner_val_idx)
        else:
            X_val_rec = _prepare_x_with_med(ml_feat_full.loc[inner_val_idx], med_rec)
            probs_rec_val = pd.Series(model_rec.predict_proba(X_val_rec)[:, 1], index=inner_val_idx)

        for prob_trigger, defensive_scale, recovery_floor, rel_tilt, stress_trigger_q, recovery_trigger_q in iproduct(cfg_fold.fragility_prob_trigger_grid, cfg_fold.defensive_scale_grid, cfg_fold.recovery_floor_grid, cfg_fold.rel_tilt_grid, cfg_fold.stress_trigger_q_grid, cfg_fold.recovery_trigger_q_grid):
            pass  # DEAD CODE — removed by PERF-7C2

        # ── PERF-7C2: precompute the 8 fixed thresholds once per outer pair ──
        fixed_thr = _precompute_fixed_thresholds(ml_feat_full.loc[inner_train_idx], cfg_fold)
        # Cache the 2 grid-varying thresholds for all 4 (stq, rtq) pairs
        stress_thr_cache: Dict[Tuple, Tuple[float, float]] = {}
        stress_s = ml_feat_full.loc[inner_train_idx, "stress_intensity"].dropna()
        rec_s    = ml_feat_full.loc[inner_train_idx, "recovery_intensity"].dropna()
        for stq, rtq in iproduct(cfg_fold.stress_trigger_q_grid, cfg_fold.recovery_trigger_q_grid):
            stress_thr_cache[(stq, rtq)] = (
                float(np.nanquantile(stress_s, stq)) if stress_s.size else np.inf,
                float(np.nanquantile(rec_s,    rtq)) if rec_s.size    else np.inf,
            )

        # ── Phase A: precompute weights+chandelier+vol_sc once per unique
        # (prob_trigger, stress_trigger_q, recovery_trigger_q, rel_tilt).
        # Uses vectorized policy — no Python date loop.
        weight_cache: Dict[Tuple, Tuple[pd.DataFrame, pd.Series]] = {}
        for pt, stq, rtq, rlt in iproduct(
            cfg_fold.fragility_prob_trigger_grid,
            cfg_fold.stress_trigger_q_grid,
            cfg_fold.recovery_trigger_q_grid,
            cfg_fold.rel_tilt_grid,
        ):
            if rlt not in score_rel_cache:
                cfg_rel = h7._make_rel_tilt_cfg(cfg_fold, rlt)
                score_rel_cache[rlt] = m6.compute_scores(ctx["close"], ctx["qqq"], cfg_rel)
            score_rel_wc = score_rel_cache[rlt]

            st, rt = stress_thr_cache[(stq, rtq)]
            # PERF-7C2: vectorized, uses precomputed fixed_thr
            pol_wc = _policy_from_dual_probs_vec(
                ml_feat_full.loc[inner_val_idx], probs_frag_val, probs_rec_val,
                pt, cfg_fold.recovery_prob_trigger,
                cfg_fold.defensive_scale_grid[0],  # dummy — doesn't affect action
                cfg_fold.recovery_floor_grid[0],   # dummy — doesn't affect action
                rlt, st, rt, fixed_thr,
                cfg_fold, base_bt["crisis_state"],
            )
            pol_daily_wc = _expand_weekly_policy_to_daily(pol_wc, base_bt["returns_net"].index)
            action_daily_wc = pol_daily_wc["action"]

            w_exec_1x_wc, vol_sc_wc = _compute_weights_and_vol(
                ctx, cfg_fold, universe_schedule, score_rel_wc, action_daily_wc
            )
            weight_cache[(pt, stq, rtq, rlt)] = (w_exec_1x_wc, vol_sc_wc)

        # ── Phase B: sweep all 64 combos, reuse cached weights.
        # Uses vectorized policy — no Python date loop per combo.
        for prob_trigger, defensive_scale, recovery_floor, rel_tilt, stress_trigger_q, recovery_trigger_q in iproduct(
            cfg_fold.fragility_prob_trigger_grid,
            cfg_fold.defensive_scale_grid,
            cfg_fold.recovery_floor_grid,
            cfg_fold.rel_tilt_grid,
            cfg_fold.stress_trigger_q_grid,
            cfg_fold.recovery_trigger_q_grid,
        ):
            st, rt = stress_thr_cache[(stress_trigger_q, recovery_trigger_q)]
            # PERF-7C2: vectorized, uses precomputed fixed_thr + cached stress/rec thresholds
            policy_weekly = _policy_from_dual_probs_vec(
                ml_feat_full.loc[inner_val_idx], probs_frag_val, probs_rec_val,
                prob_trigger, cfg_fold.recovery_prob_trigger,
                defensive_scale, recovery_floor, rel_tilt,
                st, rt, fixed_thr,
                cfg_fold, base_bt["crisis_state"],
            )
            policy_daily = _expand_weekly_policy_to_daily(policy_weekly, base_bt["returns_net"].index)

            w_exec_1x, vol_sc = weight_cache[(prob_trigger, stress_trigger_q, recovery_trigger_q, rel_tilt)]
            score, ov_sum, intervention_rate, recovery_rate, recovery_opp_rate, recovery_hit_rate, missed_rebound, recovery_capture_rate, panic_sharpe, panic_days = _score_from_weights(
                w_exec_1x, vol_sc, ctx, policy_daily, costs, cfg_fold,
                inner_val_start, inner_val_end, base_sum, policy_weekly,
                base_r, qqq_future.loc[inner_val_idx],
            )
            row = {**s1_best, "fragility_quantile": frag_q, "horizon_weeks": horizon_weeks, "prob_trigger": prob_trigger, "recovery_prob_trigger": cfg_fold.recovery_prob_trigger, "defensive_scale": defensive_scale, "recovery_floor": recovery_floor, "rel_tilt": rel_tilt, "stress_trigger_q": stress_trigger_q, "recovery_trigger_q": recovery_trigger_q, "score": score, "base_sharpe": base_sum["Sharpe"], "ov_sharpe": ov_sum["Sharpe"], "ov_cagr": ov_sum["CAGR"], "ov_maxdd": ov_sum["MaxDD"], "ov_cvar": ov_sum["CVaR_5"], "intervention_rate": intervention_rate, "recovery_rate": recovery_rate, "recovery_capture_rate": recovery_capture_rate, "missed_rebound": missed_rebound, "panic_sharpe": panic_sharpe, "panic_days": panic_days}
            s2_rows.append(row)
            if score > best_score:
                best_score = score
                best_params = row.copy()

    calib_df = pd.concat([s1_df.assign(stage="hawkes_only"), pd.DataFrame(s2_rows).assign(stage="dual_ml_overlay")], ignore_index=True) if s2_rows else s1_df.assign(stage="hawkes_only")
    fallback = {**s1_best, "fragility_quantile": cfg_fold.fragility_quantile_grid[0], "horizon_weeks": cfg_fold.fragility_horizon_weeks_grid[0], "prob_trigger": cfg_fold.fragility_prob_trigger_grid[0], "recovery_prob_trigger": cfg_fold.recovery_prob_trigger, "defensive_scale": cfg_fold.defensive_scale_grid[0], "recovery_floor": cfg_fold.recovery_floor_grid[0], "rel_tilt": cfg_fold.rel_tilt_grid[0], "stress_trigger_q": cfg_fold.stress_trigger_q_grid[0], "recovery_trigger_q": cfg_fold.recovery_trigger_q_grid[0], "score": s1_best["score"]}
    return (best_params if best_params is not None else fallback), calib_df.sort_values("score", ascending=False)


def _run_single_fold(
    fold: Dict[str, Any],
    baseline_row: pd.Series,
    ohlcv: Dict[str, pd.DataFrame],
    cfg_base: Mahoraga7DConfig,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
    feat_full: pd.DataFrame,
    total_folds: int,
) -> Dict[str, Any]:
    fold_n = int(fold["fold"])
    train_start, train_end = fold["train_start"], fold["train_end"]
    test_start, test_end = fold["test_start"], fold["test_end"]
    print(f"\n  ── 7D FOLD {fold_n}/{total_folds} ──")
    cfg_fold = _get_fold_cfg(ohlcv, cfg_base, costs, universe_schedule, baseline_row)
    print(f"  [fold {fold_n}] Calibrating 7D on train via inner validation …")
    best_params, calib_df = _calibrate_7d(feat_full, ohlcv, cfg_fold, costs, universe_schedule, train_start, train_end)

    base_bt = m6.backtest(ohlcv, cfg_fold, costs, label=f"BASE_{fold_n}", universe_schedule=universe_schedule)
    hawkes_df, _ = h7._build_hawkes_signals(feat_full, best_params["stress_q"], best_params["recovery_q"], best_params["decay"], best_params["stress_scale"], best_params["recovery_scale"], feat_full.loc[train_start:train_end])
    ml_feat_full = _build_ml_feature_table(hawkes_df, base_bt, cfg_fold)
    train_idx = ml_feat_full.index[(ml_feat_full.index >= pd.Timestamp(train_start)) & (ml_feat_full.index <= pd.Timestamp(train_end))]
    cp_df = _bocpd_lite_regime_features(ml_feat_full, train_idx, cfg_fold)
    ml_feat_full = ml_feat_full.join(cp_df)
    conf_df = _ridge_split_conformal_risk(ml_feat_full, base_bt["returns_net"], train_idx, cfg_fold)
    ml_feat_full = ml_feat_full.join(conf_df)

    utility = _future_window_utility(base_bt["returns_net"], base_bt["equity"], ml_feat_full.index, int(best_params["horizon_weeks"]), cfg_fold.utility_dd_penalty)
    qqq_future = _future_window_return(base_bt["bench"]["QQQ_r"], ml_feat_full.index, int(best_params["horizon_weeks"]))
    thr_utility = float(np.nanquantile(utility.loc[train_idx].dropna(), float(best_params["fragility_quantile"]))) if utility.loc[train_idx].dropna().size else np.nan
    y_frag = (utility <= thr_utility).astype(float)
    y_rec = _build_recovery_labels(ml_feat_full, qqq_future, train_idx, cfg_fold)

    X_train_frag, y_train_frag, med_frag = _prepare_xy(ml_feat_full.loc[train_idx], y_frag.loc[train_idx])
    model_frag = _fit_binary_model(X_train_frag, y_train_frag, cfg_fold)
    X_train_rec, y_train_rec, med_rec = _prepare_xy(ml_feat_full.loc[train_idx], y_rec.loc[train_idx])
    model_rec = _fit_binary_model(X_train_rec, y_train_rec, cfg_fold)

    test_idx = ml_feat_full.index[(ml_feat_full.index >= pd.Timestamp(test_start)) & (ml_feat_full.index <= pd.Timestamp(test_end))]
    if model_frag is None:
        probs_frag = pd.Series(0.0, index=test_idx)
        imp_frag = pd.DataFrame({"feature": FEATURE_COLS, "importance": np.nan}).assign(model="fragility", fold=fold_n)
    else:
        X_test_frag = _prepare_x_with_med(ml_feat_full.loc[test_idx], med_frag)
        probs_frag = pd.Series(model_frag.predict_proba(X_test_frag)[:, 1], index=test_idx)
        imp_frag = _model_importance(model_frag, FEATURE_COLS).assign(model="fragility", fold=fold_n)
    if model_rec is None:
        probs_rec = pd.Series(0.0, index=test_idx)
        imp_rec = pd.DataFrame({"feature": FEATURE_COLS, "importance": np.nan}).assign(model="recovery", fold=fold_n)
    else:
        X_test_rec = _prepare_x_with_med(ml_feat_full.loc[test_idx], med_rec)
        probs_rec = pd.Series(model_rec.predict_proba(X_test_rec)[:, 1], index=test_idx)
        imp_rec = _model_importance(model_rec, FEATURE_COLS).assign(model="recovery", fold=fold_n)

    policy_weekly = _policy_from_dual_probs(ml_feat_full.loc[test_idx], probs_frag, probs_rec, float(best_params["prob_trigger"]), float(best_params.get("recovery_prob_trigger", cfg_fold.recovery_prob_trigger)), float(best_params["defensive_scale"]), float(best_params["recovery_floor"]), float(best_params["rel_tilt"]), float(best_params["stress_trigger_q"]), float(best_params["recovery_trigger_q"]), cfg_fold, base_bt["crisis_state"], ml_feat_full.loc[train_idx])
    policy_daily = _expand_weekly_policy_to_daily(policy_weekly, base_bt["returns_net"].index)

    ctx_test = h7._precompute_overlay_context(ohlcv, cfg_fold, costs, universe_schedule)
    ov_bt = h7._custom_backtest_with_overlay(ohlcv, cfg_fold, costs, universe_schedule, policy_daily, label=f"H7D_{fold_n}", ctx=ctx_test, score_rel_cache={})

    rb = base_bt["returns_net"].loc[test_start:test_end]
    qb = base_bt["equity"].loc[test_start:test_end]
    eb = base_bt["exposure"].loc[test_start:test_end]
    tb = base_bt["turnover"].loc[test_start:test_end]
    sb = m6.summarize(rb, qb, eb, tb, cfg_fold, f"BASE_{fold_n}")
    ro = ov_bt["returns_net"].loc[test_start:test_end]
    qo = ov_bt["equity"].loc[test_start:test_end]
    eo = ov_bt["exposure"].loc[test_start:test_end]
    to = ov_bt["turnover"].loc[test_start:test_end]
    so = m6.summarize(ro, qo, eo, to, cfg_fold, f"H7D_{fold_n}")

    qqq_future_test = _future_window_return(base_bt["bench"]["QQQ_r"], policy_weekly.index, int(best_params["horizon_weeks"]))
    missed_rebound, captured_rebound, recovery_capture_rate = _forward_positive_capture(policy_weekly, qqq_future_test)
    panic_daily = policy_daily.get("panic_mode", pd.Series(0.0, index=policy_daily.index)).loc[test_start:test_end]
    panic_metrics = _panic_summary(ro, rb, panic_daily, cfg_fold)

    print(f"  [fold {fold_n}] BASE Sharpe={sb['Sharpe']:.3f} | 7D Sharpe={so['Sharpe']:.3f} | Δ={so['Sharpe'] - sb['Sharpe']:+.3f}")

    pred_df = policy_weekly.copy()
    pred_df["fragility_prob"] = probs_frag.reindex(policy_weekly.index).fillna(0.0)
    pred_df["recovery_prob"] = probs_rec.reindex(policy_weekly.index).fillna(0.0)
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
            "H7D_CAGR%": round(so["CAGR"] * 100, 2),
            "H7D_Sharpe": round(so["Sharpe"], 4),
            "H7D_MaxDD%": round(so["MaxDD"] * 100, 2),
            "H7D_CVaR5%": round(so["CVaR_5"] * 100, 2),
            "DeltaSharpe": round(so["Sharpe"] - sb["Sharpe"], 4),
            "InterventionRate": round(float((policy_weekly["action"] != "BASELINE").mean()), 4),
            "RecoveryRate": round(float((policy_weekly["action"] == "RECOVERY_OVERRIDE").mean()), 4),
            "MissedReboundQQQ": round(missed_rebound * 100, 2),
            "RecoveryCaptureRate": round(recovery_capture_rate, 4),
            "FragilityRate": round(float((probs_frag >= float(best_params["prob_trigger"])).mean()), 4),
            "RecoveryOppRate": round(float((probs_rec >= float(best_params.get("recovery_prob_trigger", cfg_fold.recovery_prob_trigger))).mean()), 4),
            "RecoveryHitRate": round(float(((policy_weekly["action"] == "RECOVERY_OVERRIDE").mean() / max((policy_weekly.get("recover_opp", pd.Series(0.0, index=policy_weekly.index)).mean()), 1e-9)) if policy_weekly.get("recover_opp", pd.Series(0.0, index=policy_weekly.index)).mean() > 0 else 0.0), 4),
            "PanicRate": round(float(policy_weekly.get("panic_mode", pd.Series(0.0, index=policy_weekly.index)).mean()), 4),
            "PanicSharpe": round(float(panic_metrics.get("panic_sharpe", 0.0)), 4),
            "CooldownRate": round(float(policy_weekly.get("cooldown_active", pd.Series(0.0, index=policy_weekly.index)).mean()), 4),
            "ChopRate": round(float((((policy_weekly.get("recover_opp", pd.Series(0.0, index=policy_weekly.index)) == 0.0) & (policy_weekly["action"] == "BASELINE")).mean())), 4),
            "CPRate": round(float((policy_weekly.get("cp_prob", pd.Series(0.0, index=policy_weekly.index)) > 0.5).mean()), 4),
            "MeanRiskBudget": round(float(policy_weekly.get("risk_budget", pd.Series(1.0, index=policy_weekly.index)).mean()), 4),
        },
        "weekly_policy": pred_df,
        "hawkes_df": hawkes_df.loc[test_idx],
        "calib_df": calib_df.assign(fold=fold_n),
        "feature_importance": pd.concat([imp_frag, imp_rec], ignore_index=True),
    }


def run_walk_forward_h7d(
    ohlcv: Dict[str, pd.DataFrame],
    cfg: Mahoraga7DConfig,
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

    # Prevent nested oversubscription:
    # when folds run in parallel, each RandomForest should use only a share of CPUs.
    n_cpus = os.cpu_count() or 1
    if use_parallel:
        if int(getattr(cfg, "rf_n_jobs", 1)) == -1:
            cfg.rf_n_jobs = max(1, n_cpus // max(1, n_jobs))
        else:
            cfg.rf_n_jobs = max(1, min(int(cfg.rf_n_jobs), n_cpus // max(1, n_jobs) or 1))
    else:
        if int(getattr(cfg, "rf_n_jobs", 1)) == -1:
            cfg.rf_n_jobs = n_cpus
        else:
            cfg.rf_n_jobs = max(1, min(int(cfg.rf_n_jobs), n_cpus))

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

    qqq_r = pd.concat([x["base_bt"]["bench"]["QQQ_r"] for x in results]).sort_index().reindex(base_r.index).fillna(0.0)

    return {
        "base_oos_r": base_r,
        "base_oos_eq": base_eq,
        "base_oos_exp": base_exp,
        "base_oos_to": base_to,
        "qqq_oos_r": qqq_r,
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


def _selection_audit(wf: Dict[str, Any], cfg: Mahoraga7DConfig) -> pd.DataFrame:
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
            "Method": "H7D",
            "MeanSharpe": fr["H7D_Sharpe"].mean(),
            "MeanCAGR%": fr["H7D_CAGR%"].mean(),
            "MeanMaxDD%": fr["H7D_MaxDD%"].mean(),
            "MeanInterventionRate": fr["InterventionRate"].mean(),
            "MeanFragilityRate": fr["FragilityRate"].mean(),
            "MeanRecoveryRate": fr.get("RecoveryRate", pd.Series(dtype=float)).mean() if "RecoveryRate" in fr else np.nan,
            "MeanRecoveryHitRate": fr.get("RecoveryHitRate", pd.Series(dtype=float)).mean() if "RecoveryHitRate" in fr else np.nan,
        },
    ])


def _regime_comparison(base_r: pd.Series, ov_r: pd.Series, cfg: Mahoraga7DConfig, ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return h7._regime_comparison(base_r, ov_r, cfg, ohlcv)


def _final_report_text(cfg: Mahoraga7DConfig, wf: Dict[str, Any], ohlcv: Dict[str, pd.DataFrame]) -> str:
    base_sum = m6.summarize(wf["base_oos_r"], wf["base_oos_eq"], wf["base_oos_exp"], wf["base_oos_to"], cfg, "BASE_OOS")
    ov_sum = m6.summarize(wf["ov_oos_r"], wf["ov_oos_eq"], wf["ov_oos_exp"], wf["ov_oos_to"], cfg, "H7D_OOS")
    sel = _selection_audit(wf, cfg)
    reg = _regime_comparison(wf["base_oos_r"], wf["ov_oos_r"], cfg, ohlcv)
    fr = wf["fold_results"]
    alpha_base = m6.alpha_test_nw(wf["base_oos_r"], wf["qqq_oos_r"], cfg, label="BASE_OOS")
    alpha_ov = m6.alpha_test_nw(wf["ov_oos_r"], wf["qqq_oos_r"], cfg, label="H7D_OOS")
    alpha_ov_cond = m6.alpha_test_nw(wf["ov_oos_r"], wf["qqq_oos_r"], cfg, label="H7D_OOS|exp>0", conditional=True, exposure=wf["ov_oos_exp"])
    lines = []
    lines.append("MAHORAGA 7D — FINAL REPORT")
    lines.append("=" * 78)
    lines.append(DISCLAIMER)
    lines.append("\nOOS COMPARISON")
    lines.append(f"  BASELINE CAGR={base_sum['CAGR']*100:.2f}%  Sharpe={base_sum['Sharpe']:.3f}  MaxDD={base_sum['MaxDD']*100:.2f}%  CVaR5={base_sum['CVaR_5']*100:.2f}%")
    lines.append(f"  H7D    CAGR={ov_sum['CAGR']*100:.2f}%  Sharpe={ov_sum['Sharpe']:.3f}  MaxDD={ov_sum['MaxDD']*100:.2f}%  CVaR5={ov_sum['CVaR_5']*100:.2f}%")
    lines.append("\nALPHA — Newey-West HAC vs QQQ")
    for row in [alpha_base, alpha_ov, alpha_ov_cond]:
        lines.append(str(row))
    lines.append("\nSELECTION AUDIT")
    lines.append(sel.to_string(index=False))
    lines.append("\nFOLD SUMMARY")
    lines.append(fr.to_string(index=False))
    lines.append("\nREGIME COMPARISON")
    lines.append(reg.to_string(index=False))
    if not wf["feature_importance"].empty:
        fi = wf["feature_importance"].groupby(["model", "feature"], dropna=False)["importance"].mean().reset_index().sort_values(["model", "importance"], ascending=[True, False])
        lines.append("\nFEATURE IMPORTANCE (mean across folds)")
        lines.append(fi.to_string(index=False))
    return "\n".join(lines)


def save_outputs_h7d(cfg: Mahoraga7DConfig, wf: Dict[str, Any], ohlcv: Dict[str, pd.DataFrame]):
    d = cfg.outputs_dir
    _ensure_dir(d)
    base_sum = m6.summarize(wf["base_oos_r"], wf["base_oos_eq"], wf["base_oos_exp"], wf["base_oos_to"], cfg, "BASE_OOS")
    ov_sum = m6.summarize(wf["ov_oos_r"], wf["ov_oos_eq"], wf["ov_oos_exp"], wf["ov_oos_to"], cfg, "H7D_OOS")
    comparison_rows = [
        {"Label": "BASELINE_6_1_FROZEN", **base_sum},
        {"Label": "H7D", **ov_sum},
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(os.path.join(d, "comparison_oos.csv"), index=False)
    comparison_df.to_csv(os.path.join(d, "comparison_full.csv"), index=False)
    pd.DataFrame([
        m6.alpha_test_nw(wf["base_oos_r"], wf["qqq_oos_r"], cfg, label="BASE_OOS"),
        m6.alpha_test_nw(wf["ov_oos_r"], wf["qqq_oos_r"], cfg, label="H7D_OOS"),
        m6.alpha_test_nw(wf["ov_oos_r"], wf["qqq_oos_r"], cfg, label="H7D_OOS|exp>0", conditional=True, exposure=wf["ov_oos_exp"]),
    ]).to_csv(os.path.join(d, "alpha_comparison.csv"), index=False)
    wf["fold_results"].to_csv(os.path.join(d, "walk_forward_folds_7d.csv"), index=False)
    wf["policy_artifacts"].to_csv(os.path.join(d, "dynamic_mode_controls.csv"), index=True)
    wf["hawkes_artifacts"].to_csv(os.path.join(d, "hawkes_events_intensities.csv"), index=True)
    wf["calibration_grid"].to_csv(os.path.join(d, "walk_forward_sweeps.csv"), index=False)
    wf["feat_full"].to_csv(os.path.join(d, "meta_features_snapshot.csv"), index=True)
    if not wf["feature_importance"].empty:
        wf["feature_importance"].to_csv(os.path.join(d, "feature_importance_dual.csv"), index=False)
    _selection_audit(wf, cfg).to_csv(os.path.join(d, "selection_audit.csv"), index=False)
    _regime_comparison(wf["base_oos_r"], wf["ov_oos_r"], cfg, ohlcv).to_csv(os.path.join(d, "regime_comparison.csv"), index=False)
    with open(os.path.join(d, "final_report.txt"), "w", encoding="utf-8") as f:
        f.write(_final_report_text(cfg, wf, ohlcv))
    print(f"\n  [outputs → ./{d}/]")
    print("    comparison_oos.csv, comparison_full.csv, alpha_comparison.csv, walk_forward_folds_7d.csv")
    print("    dynamic_mode_controls.csv, hawkes_events_intensities.csv, walk_forward_sweeps.csv")
    print("    meta_features_snapshot.csv, feature_importance_dual.csv, selection_audit.csv")
    print("    regime_comparison.csv, final_report.txt")


def run_mahoraga7d(make_plots_flag: bool = False, run_mode: str = "FULL") -> Dict[str, Any]:
    if not _SKLEARN:
        raise RuntimeError("scikit-learn is required for Mahoraga 7D")
    print("=" * 80)
    print("  MAHORAGA 7D — Hawkes + dual-ML layer over 6.1")
    print("=" * 80)
    print(DISCLAIMER)

    cfg = Mahoraga7DConfig()
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
    wf = run_walk_forward_h7d(ohlcv, cfg, costs, universe_schedule)
    save_outputs_h7d(cfg, wf, ohlcv)
    return {"cfg": cfg, "wf": wf, "ohlcv": ohlcv}


if __name__ == "__main__":
    results = run_mahoraga7d(make_plots_flag=False, run_mode="FULL")
