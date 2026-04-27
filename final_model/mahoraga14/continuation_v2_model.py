from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mahoraga14_config import Mahoraga14Config
from mahoraga14_utils import time_split_index


CONTINUATION_TRIGGER_FEATURES = [
    "transition_hawkes_recovery",
    "transition_hawkes_stress",
    "stop_release_2w",
    "stop_density_change_2w",
    "breadth_rebound_4w",
    "corr_release_4w",
    "compression_score_2w",
    "compression_score_4w",
]

CONTINUATION_PRESSURE_FEATURES = [
    "path_efficiency_2w",
    "path_efficiency_4w",
    "swing_amplitude_vs_net_2w",
    "swing_amplitude_vs_net_4w",
    "base_ret_1w",
    "base_ret_2w",
    "base_dd",
    "base_dd_change_2w",
    "qqq_ret_2w",
    "qqq_rebound_2w",
    "base_minus_qqq_2w",
]

BREAK_RISK_FEATURES = [
    "path_efficiency_2w",
    "path_efficiency_4w",
    "base_ret_1w",
    "base_ret_2w",
    "base_dd",
    "base_dd_change_2w",
    "qqq_ret_2w",
    "base_minus_qqq_2w",
    "transition_hawkes_stress",
    "transition_hawkes_recovery",
    "compression_score_2w",
    "corr_release_4w",
]


def _future_compound_return(r: pd.Series, horizon: int) -> pd.Series:
    out = pd.Series(np.nan, index=r.index, dtype=float)
    vals = r.fillna(0.0).values
    for i in range(len(r)):
        j = min(len(r), i + horizon + 1)
        if j <= i + 1:
            continue
        out.iloc[i] = float(np.prod(1.0 + vals[i + 1 : j]) - 1.0)
    return out


def _future_drawdown_change(eq: pd.Series, horizon: int) -> pd.Series:
    eq = eq.ffill()
    dd = eq / eq.cummax() - 1.0
    out = pd.Series(np.nan, index=eq.index, dtype=float)
    vals = dd.values
    for i in range(len(eq)):
        j = min(len(eq), i + horizon + 1)
        if j <= i + 1:
            continue
        out.iloc[i] = float(np.nanmin(vals[i + 1 : j]) - vals[i])
    return out


def _future_positive_share(r: pd.Series, horizon: int) -> pd.Series:
    out = pd.Series(np.nan, index=r.index, dtype=float)
    vals = r.fillna(0.0).values
    for i in range(len(r)):
        j = min(len(r), i + horizon + 1)
        if j <= i + 1:
            continue
        out.iloc[i] = float(np.mean(vals[i + 1 : j] > 0.0))
    return out


def _train_quantile(series: pd.Series, q: float, default: float) -> float:
    s = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return float(default)
    return float(s.quantile(q))


def _ordered_bounds(low: float, high: float, default_span: float = 0.05) -> tuple[float, float]:
    lo = float(low) if np.isfinite(low) else 0.0
    hi = float(high) if np.isfinite(high) else lo + default_span
    if hi <= lo:
        hi = lo + max(default_span, abs(lo) * 0.05 + 1e-4)
    return lo, hi


def _ramp(series: pd.Series, low: float, high: float) -> pd.Series:
    lo, hi = _ordered_bounds(low, high)
    span = max(hi - lo, np.finfo(float).eps)
    return pd.Series(((pd.Series(series, dtype=float) - lo) / span).clip(0.0, 1.0), index=series.index)


def _reverse_ramp(series: pd.Series, low: float, high: float) -> pd.Series:
    return 1.0 - _ramp(series, low, high)


def _band_score(series: pd.Series, low: float, high: float) -> pd.Series:
    lo, hi = _ordered_bounds(low, high, default_span=0.02)
    mid = 0.5 * (lo + hi)
    half_span = max((hi - lo) / 2.0, np.finfo(float).eps)
    score = 1.0 - (pd.Series(series, dtype=float) - mid).abs() / half_span
    return pd.Series(score.clip(0.0, 1.0), index=series.index)


def augment_continuation_v2_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    out = weekly_df.copy()
    defaults = {
        "path_efficiency_2w": out.get("base_eff_2w", pd.Series(0.0, index=out.index)),
        "path_efficiency_4w": out.get("base_eff_4w", pd.Series(0.0, index=out.index)),
        "compression_score_2w": pd.Series(0.0, index=out.index),
        "compression_score_4w": pd.Series(0.0, index=out.index),
        "swing_amplitude_vs_net_2w": pd.Series(0.0, index=out.index),
        "swing_amplitude_vs_net_4w": pd.Series(0.0, index=out.index),
        "base_minus_qqq_2w": out.get("base_ret_2w", pd.Series(0.0, index=out.index)) - out.get("qqq_ret_2w", pd.Series(0.0, index=out.index)),
        "stop_density_change_2w": out.get("stop_density_2w", pd.Series(0.0, index=out.index)).diff(2),
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    for col in set(CONTINUATION_TRIGGER_FEATURES + CONTINUATION_PRESSURE_FEATURES + BREAK_RISK_FEATURES):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.Series(out[col], index=out.index, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["path_efficiency_2w"] = out["path_efficiency_2w"].clip(0.0, 1.0)
    out["path_efficiency_4w"] = out["path_efficiency_4w"].clip(0.0, 1.0)
    out["swing_amplitude_vs_net_2w"] = out["swing_amplitude_vs_net_2w"].clip(lower=0.0)
    out["swing_amplitude_vs_net_4w"] = out["swing_amplitude_vs_net_4w"].clip(lower=0.0)
    out["compression_score_2w"] = out["compression_score_2w"].clip(lower=0.0)
    out["compression_score_4w"] = out["compression_score_4w"].clip(lower=0.0)
    return out


def _guard_thresholds(train_df: pd.DataFrame) -> Dict[str, float]:
    return {
        "compression_2w_min": _train_quantile(train_df["compression_score_2w"], 0.55, 0.25),
        "compression_2w_high": _train_quantile(train_df["compression_score_2w"], 0.82, 0.45),
        "compression_4w_min": _train_quantile(train_df["compression_score_4w"], 0.55, 0.30),
        "compression_4w_high": _train_quantile(train_df["compression_score_4w"], 0.82, 0.52),
        "swing_2w_min": _train_quantile(train_df["swing_amplitude_vs_net_2w"], 0.55, 1.10),
        "swing_2w_high": _train_quantile(train_df["swing_amplitude_vs_net_2w"], 0.82, 1.60),
        "swing_4w_min": _train_quantile(train_df["swing_amplitude_vs_net_4w"], 0.55, 1.20),
        "swing_4w_high": _train_quantile(train_df["swing_amplitude_vs_net_4w"], 0.82, 1.85),
        "path_eff_2w_low": _train_quantile(train_df["path_efficiency_2w"], 0.20, 0.25),
        "path_eff_2w_high": _train_quantile(train_df["path_efficiency_2w"], 0.65, 0.65),
        "path_eff_4w_low": _train_quantile(train_df["path_efficiency_4w"], 0.20, 0.30),
        "path_eff_4w_high": _train_quantile(train_df["path_efficiency_4w"], 0.65, 0.70),
        "base_dd_floor": _train_quantile(train_df["base_dd"], 0.20, -0.08),
        "base_dd_cap": _train_quantile(train_df["base_dd"], 0.75, -0.01),
        "base_dd_change_floor": _train_quantile(train_df["base_dd_change_2w"], 0.30, -0.03),
        "base_dd_change_cap": _train_quantile(train_df["base_dd_change_2w"], 0.75, 0.01),
        "stop_release_min": _train_quantile(train_df["stop_release_2w"], 0.55, 0.0),
        "stop_release_high": _train_quantile(train_df["stop_release_2w"], 0.82, 0.02),
        "stop_density_soft": _train_quantile(train_df["stop_density_2w"], 0.55, 0.015),
        "stop_density_cap": _train_quantile(train_df["stop_density_2w"], 0.82, 0.03),
        "stop_density_change_floor": _train_quantile(train_df["stop_density_change_2w"], 0.20, -0.01),
        "stop_density_change_high": _train_quantile(train_df["stop_density_change_2w"], 0.60, 0.0),
        "breadth_min": _train_quantile(train_df["breadth_rebound_4w"], 0.55, 0.0),
        "breadth_high": _train_quantile(train_df["breadth_rebound_4w"], 0.82, 0.02),
        "corr_min": _train_quantile(train_df["corr_release_4w"], 0.55, 0.0),
        "corr_high": _train_quantile(train_df["corr_release_4w"], 0.82, 0.02),
        "base_ret_1w_support_min": _train_quantile(train_df["base_ret_1w"], 0.55, 0.0),
        "base_ret_1w_support_high": _train_quantile(train_df["base_ret_1w"], 0.82, 0.02),
        "base_ret_2w_support_min": _train_quantile(train_df["base_ret_2w"], 0.55, 0.0),
        "base_ret_2w_support_high": _train_quantile(train_df["base_ret_2w"], 0.82, 0.03),
        "qqq_ret_floor": _train_quantile(train_df["qqq_ret_2w"], 0.35, -0.02),
        "qqq_ret_support": _train_quantile(train_df["qqq_ret_2w"], 0.75, 0.03),
        "qqq_rebound_floor": _train_quantile(train_df["qqq_rebound_2w"], 0.35, 0.0),
        "qqq_rebound_support": _train_quantile(train_df["qqq_rebound_2w"], 0.75, 0.025),
        "base_minus_qqq_floor": _train_quantile(train_df["base_minus_qqq_2w"], 0.35, -0.01),
        "base_minus_qqq_support": _train_quantile(train_df["base_minus_qqq_2w"], 0.75, 0.02),
        "hawkes_recovery_min": _train_quantile(train_df["transition_hawkes_recovery"], 0.55, 0.0),
        "hawkes_recovery_high": _train_quantile(train_df["transition_hawkes_recovery"], 0.82, 1.0),
        "hawkes_stress_soft": _train_quantile(train_df["transition_hawkes_stress"], 0.55, 0.8),
        "hawkes_stress_cap": _train_quantile(train_df["transition_hawkes_stress"], 0.82, 1.2),
        "loss_share_fragile": _train_quantile(train_df.get("loss_share_4w", pd.Series(0.0, index=train_df.index)), 0.80, 0.65),
        "corr_persist_fragile": _train_quantile(train_df.get("corr_persist_4w", pd.Series(0.0, index=train_df.index)), 0.80, 0.80),
    }


def _build_soft_context_scores(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["compression_score"] = pd.concat(
        [
            _ramp(df["compression_score_2w"], thresholds["compression_2w_min"], thresholds["compression_2w_high"]),
            _ramp(df["compression_score_4w"], thresholds["compression_4w_min"], thresholds["compression_4w_high"]),
            _ramp(df["swing_amplitude_vs_net_2w"], thresholds["swing_2w_min"], thresholds["swing_2w_high"]),
            _ramp(df["swing_amplitude_vs_net_4w"], thresholds["swing_4w_min"], thresholds["swing_4w_high"]),
            _reverse_ramp(df["path_efficiency_2w"], thresholds["path_eff_2w_low"], thresholds["path_eff_2w_high"]),
            _reverse_ramp(df["path_efficiency_4w"], thresholds["path_eff_4w_low"], thresholds["path_eff_4w_high"]),
        ],
        axis=1,
    ).mean(axis=1).clip(0.0, 1.0)

    out["pause_score"] = pd.concat(
        [
            _band_score(df["base_dd"], thresholds["base_dd_floor"], thresholds["base_dd_cap"]),
            _ramp(df["base_dd_change_2w"], thresholds["base_dd_change_floor"], thresholds["base_dd_change_cap"]),
            _reverse_ramp(df["stop_density_2w"], thresholds["stop_density_soft"], thresholds["stop_density_cap"]),
            _ramp(df["stop_release_2w"], thresholds["stop_release_min"], thresholds["stop_release_high"]),
        ],
        axis=1,
    ).mean(axis=1).clip(0.0, 1.0)

    out["benchmark_score"] = pd.concat(
        [
            _ramp(df["qqq_ret_2w"], thresholds["qqq_ret_floor"], thresholds["qqq_ret_support"]),
            _ramp(df["qqq_rebound_2w"], thresholds["qqq_rebound_floor"], thresholds["qqq_rebound_support"]),
            _ramp(df["base_minus_qqq_2w"], thresholds["base_minus_qqq_floor"], thresholds["base_minus_qqq_support"]),
        ],
        axis=1,
    ).mean(axis=1).clip(0.0, 1.0)

    out["support_score"] = pd.concat(
        [
            _ramp(df["base_ret_1w"], thresholds["base_ret_1w_support_min"], thresholds["base_ret_1w_support_high"]),
            _ramp(df["base_ret_2w"], thresholds["base_ret_2w_support_min"], thresholds["base_ret_2w_support_high"]),
            _ramp(df["breadth_rebound_4w"], thresholds["breadth_min"], thresholds["breadth_high"]),
            _ramp(df["corr_release_4w"], thresholds["corr_min"], thresholds["corr_high"]),
            _ramp(df["transition_hawkes_recovery"], thresholds["hawkes_recovery_min"], thresholds["hawkes_recovery_high"]),
            _reverse_ramp(df["transition_hawkes_stress"], thresholds["hawkes_stress_soft"], thresholds["hawkes_stress_cap"]),
        ],
        axis=1,
    ).mean(axis=1).clip(0.0, 1.0)

    structural_health = pd.concat(
        [
            _band_score(df["base_dd"], thresholds["base_dd_floor"], thresholds["base_dd_cap"]),
            _reverse_ramp(df["stop_density_4w"], thresholds["stop_density_soft"], thresholds["stop_density_cap"]),
            _reverse_ramp(df.get("loss_share_4w", pd.Series(0.0, index=df.index)), 0.45, thresholds["loss_share_fragile"]),
            _reverse_ramp(df.get("corr_persist_4w", pd.Series(0.0, index=df.index)), 0.50, thresholds["corr_persist_fragile"]),
            _ramp(df["path_efficiency_4w"], thresholds["path_eff_4w_low"], thresholds["path_eff_4w_high"]),
        ],
        axis=1,
    ).mean(axis=1).clip(0.0, 1.0)

    out["structural_health_score"] = structural_health
    out["trigger_context_score"] = (
        0.35 * out["compression_score"]
        + 0.25 * out["pause_score"]
        + 0.20 * out["support_score"]
        + 0.10 * out["benchmark_score"]
        + 0.10 * structural_health
    ).clip(0.0, 1.0)
    out["pressure_context_score"] = (
        0.30 * out["support_score"]
        + 0.25 * out["benchmark_score"]
        + 0.25 * structural_health
        + 0.20 * (1.0 - out["pause_score"].sub(0.5).abs().clip(0.0, 0.5) * 2.0)
    ).clip(0.0, 1.0)
    return out


def evaluate_continuation_v2_context(weekly_df: pd.DataFrame, guard_thresholds: Dict[str, float]) -> pd.DataFrame:
    df = augment_continuation_v2_features(weekly_df)
    out = _build_soft_context_scores(df, guard_thresholds)
    out["compression_valid"] = (
        (df["compression_score_2w"] >= guard_thresholds["compression_2w_min"])
        | (df["compression_score_4w"] >= guard_thresholds["compression_4w_min"])
    ).astype(int)
    out["pause_valid"] = (
        (df["base_dd"] >= guard_thresholds["base_dd_floor"])
        & (df["base_dd"] <= guard_thresholds["base_dd_cap"])
        & (df["base_dd_change_2w"] >= guard_thresholds["base_dd_change_floor"])
    ).astype(int)
    out["benchmark_valid"] = (
        (df["qqq_ret_2w"] >= guard_thresholds["qqq_ret_floor"])
        | (df["qqq_rebound_2w"] >= guard_thresholds["qqq_rebound_floor"])
        | (df["base_minus_qqq_2w"] >= guard_thresholds["base_minus_qqq_floor"])
    ).astype(int)
    out["trigger_valid"] = (
        (df["transition_hawkes_recovery"] >= guard_thresholds["hawkes_recovery_min"])
        | (df["stop_release_2w"] >= guard_thresholds["stop_release_min"])
        | (df["breadth_rebound_4w"] >= guard_thresholds["breadth_min"])
        | (df["corr_release_4w"] >= guard_thresholds["corr_min"])
        | (df["stop_density_change_2w"] >= guard_thresholds["stop_density_change_floor"])
    ).astype(int)
    structural_floor = float(guard_thresholds.get("structural_health_floor", 0.45))
    out["structural_low"] = (out["structural_health_score"] >= structural_floor).astype(int)
    return out


def annotate_continuation_v2_labels(weekly_df: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    out = augment_continuation_v2_features(weekly_df)
    train_df = out.loc[:train_end].copy()
    thresholds = _guard_thresholds(train_df)
    train_context_base = _build_soft_context_scores(train_df, thresholds)
    thresholds["structural_health_floor"] = _train_quantile(train_context_base["structural_health_score"], 0.45, 0.45)
    context = evaluate_continuation_v2_context(out, thresholds)

    fwd1 = _future_compound_return(out["base_r"], 1)
    fwd2 = _future_compound_return(out["base_r"], 2)
    fwd4 = _future_compound_return(out["base_r"], 4)
    dd1 = _future_drawdown_change(out["base_eq"], 1)
    dd2 = _future_drawdown_change(out["base_eq"], 2)
    dd4 = _future_drawdown_change(out["base_eq"], 4)
    pos4 = _future_positive_share(out["base_r"], 4)

    train_idx = train_df.index
    fwd2_min = _train_quantile(fwd2.loc[train_idx], 0.60, 0.01)
    fwd4_min = _train_quantile(fwd4.loc[train_idx], 0.65, 0.015)
    pos4_min = _train_quantile(pos4.loc[train_idx], 0.60, 0.50)
    dd2_floor = _train_quantile(dd2.loc[train_idx], 0.40, -0.02)
    dd4_floor = _train_quantile(dd4.loc[train_idx], 0.40, -0.03)
    break_fwd1_low = _train_quantile(fwd1.loc[train_idx], 0.35, -0.005)
    break_fwd2_low = _train_quantile(fwd2.loc[train_idx], 0.35, 0.0)
    break_dd1_low = _train_quantile(dd1.loc[train_idx], 0.35, -0.015)
    break_dd2_low = _train_quantile(dd2.loc[train_idx], 0.35, -0.025)
    trigger_floor = _train_quantile(context.loc[train_idx, "trigger_context_score"], 0.55, 0.45)
    pressure_floor = _train_quantile(context.loc[train_idx, "pressure_context_score"], 0.55, 0.45)

    future_ret_ok = (fwd2 >= fwd2_min) | (fwd4 >= fwd4_min)
    persistence_ok = (fwd4 >= fwd4_min) & (pos4 >= pos4_min)
    future_dd_ok = (dd2 >= dd2_floor) & (dd4 >= dd4_floor)
    non_fragile_now = context["structural_low"].astype(bool)

    out["continuation_y"] = (
        context["compression_valid"].astype(bool)
        & context["pause_valid"].astype(bool)
        & context["benchmark_valid"].astype(bool)
        & context["trigger_valid"].astype(bool)
        & non_fragile_now
        & (context["trigger_context_score"] >= trigger_floor)
        & (context["pressure_context_score"] >= pressure_floor)
        & future_ret_ok
        & persistence_ok
        & future_dd_ok
    ).astype(int)

    out["break_risk_y"] = (
        context["compression_valid"].astype(bool)
        & context["benchmark_valid"].astype(bool)
        & (
            (fwd1 <= break_fwd1_low)
            | (fwd2 <= break_fwd2_low)
            | (dd1 <= break_dd1_low)
            | (dd2 <= break_dd2_low)
        )
    ).astype(int)

    for col in context.columns:
        out[f"continuation_{col}"] = context[col]
    return out


def _fit_time_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    cfg: Mahoraga14Config,
    outer_parallel: bool,
    logit_name: str,
    rf_name: str,
) -> Dict[str, Any]:
    df = train_df[feature_cols + [label_col]].dropna().copy()
    if len(df) < max(cfg.min_train_weeks, 40) or df[label_col].nunique() < 2:
        base_prob = float(df[label_col].mean()) if len(df) else 0.0
        return {"name": "neutral", "model": None, "base_prob": base_prob, "score": 0.5}

    cut = time_split_index(df.index, cfg.inner_val_frac, cfg.min_train_weeks)
    tr = df.iloc[:cut].copy()
    va = df.iloc[cut:].copy()
    X_tr, y_tr = tr[feature_cols], tr[label_col]
    X_va, y_va = va[feature_cols], va[label_col]

    models = []

    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.7, max_iter=2000, class_weight="balanced"),
    )
    logit.fit(X_tr, y_tr)
    p_logit = pd.Series(logit.predict_proba(X_va)[:, 1], index=X_va.index)
    auc_logit = roc_auc_score(y_va, p_logit) if y_va.nunique() > 1 else 0.5
    models.append({"name": logit_name, "model": logit, "score": float(auc_logit), "val_prob": p_logit, "val_y": y_va})

    if cfg.enable_rf_challenger:
        rf = RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            min_samples_leaf=cfg.rf_min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=cfg.rf_random_state,
            n_jobs=1 if outer_parallel else -1,
        )
        rf.fit(X_tr, y_tr)
        p_rf = pd.Series(rf.predict_proba(X_va)[:, 1], index=X_va.index)
        auc_rf = roc_auc_score(y_va, p_rf) if y_va.nunique() > 1 else 0.5
        models.append({"name": rf_name, "model": rf, "score": float(auc_rf), "val_prob": p_rf, "val_y": y_va})

    return max(models, key=lambda x: x["score"])


def _calibration_table(prob: pd.Series, y: pd.Series, label: str) -> pd.DataFrame:
    df = pd.DataFrame({"prob": prob, "y": y}).dropna().copy()
    if len(df) == 0:
        return pd.DataFrame(columns=["bucket", "pred_mean", "realized_mean", "count", "model_label"])
    rank = df["prob"].rank(method="first")
    df["bucket"] = pd.qcut(rank, q=min(5, len(df)), labels=False, duplicates="drop")
    table = (
        df.groupby("bucket")
        .agg(pred_mean=("prob", "mean"), realized_mean=("y", "mean"), count=("y", "size"))
        .reset_index()
    )
    table["model_label"] = label
    return table


def _validation_entry_threshold(
    model_info: Dict[str, Any],
    train_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    floor_q: float,
    ceil_q: float,
    target_rate: float,
    rate_cap: float,
    precision_weight: float,
    recall_weight: float,
) -> float:
    df = train_df[feature_cols + [label_col]].dropna().copy()
    if len(df) == 0:
        return 0.5
    cut = time_split_index(df.index, 0.30, min(40, max(20, len(df) // 2)))
    va = df.iloc[cut:].copy()
    if len(va) == 0:
        return float(df[label_col].mean())

    if model_info.get("model") is None:
        return float(np.clip(model_info.get("base_prob", 0.0), 0.01, 0.99))

    probs = pd.Series(model_info["model"].predict_proba(va[feature_cols])[:, 1], index=va.index)
    y_va = pd.Series(va[label_col], index=va.index, dtype=int)
    floor = float(np.clip(probs.quantile(floor_q), 0.01, 0.99))
    ceil = float(np.clip(probs.quantile(ceil_q), floor, 0.99))
    candidate_values = sorted(set(float(x) for x in np.linspace(floor, ceil, num=7)))

    positive_count = int((y_va == 1).sum())
    best_thr = floor
    best_utility = float("-inf")
    for thr in candidate_values:
        chosen = probs >= thr
        if int(chosen.sum()) == 0:
            continue
        rate = float(chosen.mean())
        precision = float(y_va.loc[chosen].mean())
        recall = float(((chosen) & (y_va == 1)).sum() / max(1, positive_count))
        utility = precision_weight * precision + recall_weight * recall
        utility -= 0.60 * max(0.0, rate - rate_cap)
        utility -= 0.30 * max(0.0, target_rate - rate)
        if utility > best_utility:
            best_utility = utility
            best_thr = thr
    return float(best_thr)


def fit_continuation_v2_model(train_weekly: pd.DataFrame, cfg: Mahoraga14Config, outer_parallel: bool) -> Dict[str, Any]:
    train_df = augment_continuation_v2_features(train_weekly)
    if "continuation_y" not in train_df.columns or "break_risk_y" not in train_df.columns:
        train_df = annotate_continuation_v2_labels(train_df, pd.Timestamp(train_df.index.max()))

    guard_thresholds = _guard_thresholds(train_df)
    train_context_base = _build_soft_context_scores(train_df, guard_thresholds)
    guard_thresholds["structural_health_floor"] = _train_quantile(train_context_base["structural_health_score"], 0.45, 0.45)
    context = evaluate_continuation_v2_context(train_df, guard_thresholds)

    trigger_model = _fit_time_model(
        train_df,
        CONTINUATION_TRIGGER_FEATURES,
        "continuation_y",
        cfg,
        outer_parallel,
        "logit_continuation_trigger",
        "rf_continuation_trigger",
    )
    pressure_model = _fit_time_model(
        train_df,
        CONTINUATION_PRESSURE_FEATURES,
        "continuation_y",
        cfg,
        outer_parallel,
        "logit_continuation_pressure",
        "rf_continuation_pressure",
    )
    break_model = _fit_time_model(
        train_df,
        BREAK_RISK_FEATURES,
        "break_risk_y",
        cfg,
        outer_parallel,
        "logit_break_risk",
        "rf_break_risk",
    )

    trigger_p = apply_probability_model(trigger_model, train_df, CONTINUATION_TRIGGER_FEATURES, "continuation_trigger_p")
    pressure_p = apply_probability_model(pressure_model, train_df, CONTINUATION_PRESSURE_FEATURES, "continuation_pressure_p")
    break_p = apply_probability_model(break_model, train_df, BREAK_RISK_FEATURES, "continuation_break_risk_p")

    trigger_score = (0.65 * trigger_p + 0.35 * context["trigger_context_score"]).clip(0.0, 1.0)
    pressure_score = (0.60 * pressure_p + 0.40 * context["pressure_context_score"]).clip(0.0, 1.0)

    trigger_enter = _validation_entry_threshold(
        trigger_model,
        train_df,
        CONTINUATION_TRIGGER_FEATURES,
        "continuation_y",
        cfg.continuation_trigger_floor_quantile,
        cfg.continuation_trigger_ceiling_quantile,
        cfg.continuation_target_rate,
        cfg.continuation_rate_cap,
        precision_weight=0.8,
        recall_weight=1.2,
    )
    pressure_enter = _validation_entry_threshold(
        pressure_model,
        train_df,
        CONTINUATION_PRESSURE_FEATURES,
        "continuation_y",
        cfg.continuation_pressure_floor_quantile,
        cfg.continuation_pressure_ceiling_quantile,
        cfg.continuation_target_rate,
        cfg.continuation_rate_cap,
        precision_weight=1.1,
        recall_weight=0.8,
    )
    trigger_ceiling = float(np.clip(trigger_score.quantile(cfg.continuation_trigger_ceiling_quantile), trigger_enter, 0.99))
    pressure_ceiling = float(np.clip(pressure_score.quantile(cfg.continuation_pressure_ceiling_quantile), pressure_enter, 0.99))
    break_risk_cap = float(np.clip(break_p.quantile(cfg.continuation_break_risk_cap_quantile), 0.01, 0.99))
    break_risk_floor = float(np.clip(break_p.quantile(cfg.continuation_break_risk_floor_quantile), 0.0, break_risk_cap))

    calibration_df = pd.concat(
        [
            _calibration_table(trigger_model.get("val_prob", pd.Series(dtype=float)), trigger_model.get("val_y", pd.Series(dtype=float)), "trigger"),
            _calibration_table(pressure_model.get("val_prob", pd.Series(dtype=float)), pressure_model.get("val_y", pd.Series(dtype=float)), "pressure"),
            _calibration_table(break_model.get("val_prob", pd.Series(dtype=float)), break_model.get("val_y", pd.Series(dtype=float)), "break_risk"),
        ],
        axis=0,
        ignore_index=True,
    )

    return {
        "name": f"{trigger_model['name']}|{pressure_model['name']}|{break_model['name']}",
        "trigger_model": trigger_model,
        "pressure_model": pressure_model,
        "break_model": break_model,
        "guard_thresholds": guard_thresholds,
        "trigger_features": list(CONTINUATION_TRIGGER_FEATURES),
        "pressure_features": list(CONTINUATION_PRESSURE_FEATURES),
        "break_features": list(BREAK_RISK_FEATURES),
        "trigger_enter": float(trigger_enter),
        "trigger_ceiling": float(trigger_ceiling),
        "pressure_enter": float(pressure_enter),
        "pressure_ceiling": float(pressure_ceiling),
        "break_risk_floor": float(break_risk_floor),
        "break_risk_cap": float(break_risk_cap),
        "validation_auc_trigger": float(trigger_model.get("score", 0.5)),
        "validation_auc_pressure": float(pressure_model.get("score", 0.5)),
        "validation_auc_break_risk": float(break_model.get("score", 0.5)),
        "calibration": calibration_df,
    }


def apply_probability_model(
    model_info: Dict[str, Any],
    weekly_df: pd.DataFrame,
    feature_cols: list[str],
    output_name: str,
) -> pd.Series:
    df = augment_continuation_v2_features(weekly_df)
    if model_info.get("model") is None:
        return pd.Series(float(model_info.get("base_prob", 0.0)), index=df.index, name=output_name)
    p = model_info["model"].predict_proba(df[feature_cols].fillna(0.0))[:, 1]
    return pd.Series(p, index=df.index, name=output_name)


def apply_continuation_v2_model(model_info: Dict[str, Any], weekly_df: pd.DataFrame) -> pd.DataFrame:
    df = augment_continuation_v2_features(weekly_df)
    context = evaluate_continuation_v2_context(df, model_info.get("guard_thresholds", {}))
    trigger_p = apply_probability_model(model_info["trigger_model"], df, model_info["trigger_features"], "continuation_trigger_p")
    pressure_p = apply_probability_model(model_info["pressure_model"], df, model_info["pressure_features"], "continuation_pressure_p")
    break_p = apply_probability_model(model_info["break_model"], df, model_info["break_features"], "continuation_break_risk_p")
    out = pd.DataFrame(index=df.index)
    out["continuation_trigger_p"] = trigger_p
    out["continuation_pressure_p"] = pressure_p
    out["continuation_break_risk_p"] = break_p
    out["continuation_trigger_score"] = (0.65 * trigger_p + 0.35 * context["trigger_context_score"]).clip(0.0, 1.0)
    out["continuation_pressure_score"] = (0.60 * pressure_p + 0.40 * context["pressure_context_score"]).clip(0.0, 1.0)
    out["continuation_trigger_context"] = context["trigger_context_score"]
    out["continuation_pressure_context"] = context["pressure_context_score"]
    out["continuation_compression_score"] = context["compression_score"]
    out["continuation_pause_score"] = context["pause_score"]
    out["continuation_support_score"] = context["support_score"]
    out["continuation_benchmark_score"] = context["benchmark_score"]
    out["continuation_structural_health_score"] = context["structural_health_score"]
    out["continuation_compression_valid"] = context["compression_valid"]
    out["continuation_pause_valid"] = context["pause_valid"]
    out["continuation_benchmark_valid"] = context["benchmark_valid"]
    out["continuation_trigger_valid"] = context["trigger_valid"]
    out["continuation_structural_low"] = context["structural_low"]
    return out
