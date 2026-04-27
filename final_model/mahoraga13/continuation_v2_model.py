from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mahoraga13_config import Mahoraga13Config
from mahoraga13_utils import time_split_index


CONTINUATION_V2_FEATURES = [
    "path_efficiency_2w",
    "path_efficiency_4w",
    "swing_amplitude_vs_net_2w",
    "swing_amplitude_vs_net_4w",
    "compression_score_2w",
    "compression_score_4w",
    "base_ret_1w",
    "base_ret_2w",
    "base_dd",
    "base_dd_change_2w",
    "stop_density_2w",
    "stop_release_2w",
    "breadth_rebound_4w",
    "scale_recovery_2w",
    "corr_release_4w",
    "qqq_ret_2w",
    "qqq_rebound_2w",
    "transition_hawkes_recovery",
    "transition_hawkes_stress",
]


def _future_compound_return(r: pd.Series, horizon: int) -> pd.Series:
    out = pd.Series(np.nan, index=r.index, dtype=float)
    vals = r.fillna(0.0).values
    for i in range(len(r)):
        j = min(len(r), i + horizon + 1)
        if j <= i + 1:
            continue
        out.iloc[i] = float(np.prod(1.0 + vals[i + 1:j]) - 1.0)
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
        out.iloc[i] = float(np.nanmin(vals[i + 1:j]) - vals[i])
    return out


def _train_quantile(series: pd.Series, q: float, default: float) -> float:
    s = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return float(default)
    return float(s.quantile(q))


def augment_continuation_v2_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    out = weekly_df.copy()
    defaults = {
        "path_efficiency_2w": out.get("base_eff_2w", pd.Series(0.0, index=out.index)),
        "path_efficiency_4w": out.get("base_eff_4w", pd.Series(0.0, index=out.index)),
        "compression_score_2w": pd.Series(0.0, index=out.index),
        "compression_score_4w": pd.Series(0.0, index=out.index),
        "swing_amplitude_vs_net_2w": pd.Series(0.0, index=out.index),
        "swing_amplitude_vs_net_4w": pd.Series(0.0, index=out.index),
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    out["path_efficiency_2w"] = pd.Series(out["path_efficiency_2w"], index=out.index, dtype=float).clip(0.0, 1.0).fillna(0.0)
    out["path_efficiency_4w"] = pd.Series(out["path_efficiency_4w"], index=out.index, dtype=float).clip(0.0, 1.0).fillna(0.0)
    out["swing_amplitude_vs_net_2w"] = pd.Series(out["swing_amplitude_vs_net_2w"], index=out.index, dtype=float).clip(lower=0.0).fillna(0.0)
    out["swing_amplitude_vs_net_4w"] = pd.Series(out["swing_amplitude_vs_net_4w"], index=out.index, dtype=float).clip(lower=0.0).fillna(0.0)
    out["compression_score_2w"] = pd.Series(out["compression_score_2w"], index=out.index, dtype=float).clip(lower=0.0).fillna(0.0)
    out["compression_score_4w"] = pd.Series(out["compression_score_4w"], index=out.index, dtype=float).clip(lower=0.0).fillna(0.0)

    for col in CONTINUATION_V2_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.Series(out[col], index=out.index, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _guard_thresholds(train_df: pd.DataFrame) -> Dict[str, float]:
    return {
        "compression_2w_min": _train_quantile(train_df["compression_score_2w"], 0.65, 0.35),
        "compression_4w_min": _train_quantile(train_df["compression_score_4w"], 0.65, 0.45),
        "swing_2w_min": _train_quantile(train_df["swing_amplitude_vs_net_2w"], 0.60, 1.2),
        "swing_4w_min": _train_quantile(train_df["swing_amplitude_vs_net_4w"], 0.60, 1.4),
        "path_eff_2w_max": _train_quantile(train_df["path_efficiency_2w"], 0.45, 0.55),
        "path_eff_4w_max": _train_quantile(train_df["path_efficiency_4w"], 0.45, 0.60),
        "base_dd_floor": _train_quantile(train_df["base_dd"], 0.20, -0.08),
        "base_dd_cap": _train_quantile(train_df["base_dd"], 0.75, -0.01),
        "base_dd_change_floor": _train_quantile(train_df["base_dd_change_2w"], 0.35, -0.025),
        "stop_density_cap": _train_quantile(train_df["stop_density_2w"], 0.75, 0.03),
        "base_ret_1w_support_min": _train_quantile(train_df["base_ret_1w"], 0.55, 0.0),
        "base_ret_2w_support_min": _train_quantile(train_df["base_ret_2w"], 0.55, 0.0),
        "stop_release_min": _train_quantile(train_df["stop_release_2w"], 0.55, 0.0),
        "breadth_min": _train_quantile(train_df["breadth_rebound_4w"], 0.55, 0.0),
        "scale_min": _train_quantile(train_df["scale_recovery_2w"], 0.55, 0.0),
        "corr_min": _train_quantile(train_df["corr_release_4w"], 0.55, 0.0),
        "qqq_ret_floor": _train_quantile(train_df["qqq_ret_2w"], 0.35, -0.02),
        "qqq_rebound_floor": _train_quantile(train_df["qqq_rebound_2w"], 0.35, 0.0),
        "hawkes_recovery_min": _train_quantile(train_df["transition_hawkes_recovery"], 0.55, 0.0),
        "hawkes_stress_cap": _train_quantile(train_df["transition_hawkes_stress"], 0.75, 1.0),
    }


def _compression_recent(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.Series:
    short_ok = (
        (df["compression_score_2w"] >= thresholds["compression_2w_min"])
        & (df["swing_amplitude_vs_net_2w"] >= thresholds["swing_2w_min"])
        & (df["path_efficiency_2w"] <= thresholds["path_eff_2w_max"])
    )
    medium_ok = (
        (df["compression_score_4w"] >= thresholds["compression_4w_min"])
        & (df["swing_amplitude_vs_net_4w"] >= thresholds["swing_4w_min"])
        & (df["path_efficiency_4w"] <= thresholds["path_eff_4w_max"])
    )
    return short_ok | medium_ok


def _pause_recent(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.Series:
    return (
        (df["base_dd"] >= thresholds["base_dd_floor"])
        & (df["base_dd"] <= thresholds["base_dd_cap"])
        & (df["base_dd_change_2w"] >= thresholds["base_dd_change_floor"])
        & (df["stop_density_2w"] <= thresholds["stop_density_cap"])
    )


def _support_now(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.Series:
    return (
        (df["base_ret_1w"] >= thresholds["base_ret_1w_support_min"])
        | (df["base_ret_2w"] >= thresholds["base_ret_2w_support_min"])
        | (df["stop_release_2w"] >= thresholds["stop_release_min"])
        | (df["breadth_rebound_4w"] >= thresholds["breadth_min"])
        | (df["scale_recovery_2w"] >= thresholds["scale_min"])
        | (df["corr_release_4w"] >= thresholds["corr_min"])
        | (df["transition_hawkes_recovery"] >= thresholds["hawkes_recovery_min"])
    )


def _benchmark_ok(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.Series:
    return (
        (df["qqq_ret_2w"] >= thresholds["qqq_ret_floor"])
        | (df["qqq_rebound_2w"] >= thresholds["qqq_rebound_floor"])
    )


def evaluate_continuation_v2_context(weekly_df: pd.DataFrame, guard_thresholds: Dict[str, float]) -> pd.DataFrame:
    df = augment_continuation_v2_features(weekly_df)
    out = pd.DataFrame(index=df.index)
    out["compression_valid"] = _compression_recent(df, guard_thresholds).astype(int)
    out["pause_valid"] = _pause_recent(df, guard_thresholds).astype(int)
    out["support_valid"] = _support_now(df, guard_thresholds).astype(int)
    out["benchmark_valid"] = _benchmark_ok(df, guard_thresholds).astype(int)
    out["stress_ok"] = (
        df["transition_hawkes_stress"] <= float(guard_thresholds.get("hawkes_stress_cap", np.inf))
    ).astype(int)
    return out


def annotate_continuation_v2_labels(weekly_df: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    out = augment_continuation_v2_features(weekly_df)
    train_df = out.loc[:train_end].copy()
    thresholds = _guard_thresholds(train_df)

    fwd2 = _future_compound_return(out["base_r"], 2)
    fwd4 = _future_compound_return(out["base_r"], 4)
    dd2 = _future_drawdown_change(out["base_eq"], 2)
    dd4 = _future_drawdown_change(out["base_eq"], 4)

    train_idx = train_df.index
    fwd2_min = _train_quantile(fwd2.loc[train_idx], 0.60, 0.01)
    fwd4_min = _train_quantile(fwd4.loc[train_idx], 0.65, 0.015)
    dd2_floor = _train_quantile(dd2.loc[train_idx], 0.40, -0.02)
    dd4_floor = _train_quantile(dd4.loc[train_idx], 0.40, -0.03)

    context = evaluate_continuation_v2_context(out, thresholds)
    compression_recent = context["compression_valid"].astype(bool)
    pause_recent = context["pause_valid"].astype(bool)
    support_now = context["support_valid"].astype(bool)
    benchmark_ok = context["benchmark_valid"].astype(bool)
    stress_ok = context["stress_ok"].astype(bool)
    structural_ok = ~pd.Series(out.get("structural_y", 0.0), index=out.index, dtype=float).fillna(0.0).astype(bool)

    future_ret_ok = (fwd2 >= fwd2_min) | (fwd4 >= fwd4_min)
    future_dd_ok = (dd2 >= dd2_floor) & (dd4 >= dd4_floor)

    out["continuation_compression_valid"] = context["compression_valid"]
    out["continuation_pause_valid"] = context["pause_valid"]
    out["continuation_support_valid"] = context["support_valid"]
    out["continuation_benchmark_valid"] = context["benchmark_valid"]

    out["continuation_y"] = (
        compression_recent
        & pause_recent
        & support_now
        & benchmark_ok
        & stress_ok
        & structural_ok
        & future_ret_ok
        & future_dd_ok
    ).astype(int)
    out["continuation_v2_y"] = out["continuation_y"]
    return out


def _fit_time_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    cfg: Mahoraga13Config,
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
    models.append({"name": logit_name, "model": logit, "score": float(auc_logit)})

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
        models.append({"name": rf_name, "model": rf, "score": float(auc_rf)})

    return max(models, key=lambda x: x["score"])


def _validation_entry_threshold(model_info: Dict[str, Any], train_df: pd.DataFrame, cfg: Mahoraga13Config) -> float:
    floor = float(cfg.continuation_v2_entry_floor)
    cap = float(cfg.continuation_v2_entry_cap)
    if model_info.get("model") is None:
        return float(np.clip(max(floor, float(model_info.get("base_prob", 0.0))), floor, cap))

    df = train_df[CONTINUATION_V2_FEATURES + ["continuation_y"]].dropna().copy()
    if len(df) == 0:
        return floor

    cut = time_split_index(df.index, cfg.inner_val_frac, cfg.min_train_weeks)
    va = df.iloc[cut:].copy()
    if len(va) == 0:
        return floor

    probs = pd.Series(model_info["model"].predict_proba(va[CONTINUATION_V2_FEATURES])[:, 1], index=va.index)
    y_va = pd.Series(va["continuation_y"], index=va.index, dtype=int)
    pos_probs = probs.loc[y_va == 1]
    neg_probs = probs.loc[y_va == 0]

    if len(pos_probs) == 0:
        return float(np.clip(probs.quantile(0.85), floor, cap))

    candidate_values = [
        floor,
        cap,
        float(pos_probs.quantile(0.40)),
        float(pos_probs.median()),
        float(probs.quantile(0.70)),
        float(probs.quantile(0.75)),
        float(probs.quantile(0.80)),
        float(probs.quantile(0.85)),
    ]
    if len(neg_probs):
        candidate_values.extend([float(neg_probs.quantile(0.80)), float(neg_probs.quantile(0.85))])

    best_thr = floor
    best_utility = float("-inf")
    positive_count = int((y_va == 1).sum())
    for raw_thr in candidate_values:
        thr = float(np.clip(raw_thr, floor, cap))
        chosen = probs >= thr
        rate = float(chosen.mean())
        if int(chosen.sum()) == 0:
            continue

        precision = float(y_va.loc[chosen].mean())
        recall = float(((chosen) & (y_va == 1)).sum() / max(1, positive_count))
        utility = precision + 0.25 * recall
        utility -= 0.60 * max(0.0, rate - float(cfg.continuation_v2_rate_cap))
        utility -= 0.18 * max(0.0, float(cfg.continuation_v2_target_rate) - rate)
        utility -= 0.10 * max(0.0, float(cfg.continuation_v2_min_rate) - rate)

        if utility > best_utility:
            best_utility = utility
            best_thr = thr

    return float(np.clip(best_thr, floor, cap))


def fit_continuation_v2_model(train_weekly: pd.DataFrame, cfg: Mahoraga13Config, outer_parallel: bool) -> Dict[str, Any]:
    train_df = augment_continuation_v2_features(train_weekly)
    if "continuation_y" not in train_df.columns:
        train_df = annotate_continuation_v2_labels(train_df, pd.Timestamp(train_df.index.max()))

    model_info = _fit_time_model(
        train_df,
        CONTINUATION_V2_FEATURES,
        "continuation_y",
        cfg,
        outer_parallel,
        "logit_continuation_v2",
        "rf_continuation_v2",
    )
    model_info["entry_threshold"] = _validation_entry_threshold(model_info, train_df, cfg)
    model_info["guard_thresholds"] = _guard_thresholds(train_df)
    model_info["feature_cols"] = list(CONTINUATION_V2_FEATURES)
    return model_info


def apply_continuation_v2_model(model_info: Dict[str, Any], weekly_df: pd.DataFrame) -> pd.Series:
    df = augment_continuation_v2_features(weekly_df)
    if model_info.get("model") is None:
        return pd.Series(float(model_info.get("base_prob", 0.0)), index=df.index, name="continuation_v2_p")
    p = model_info["model"].predict_proba(df[CONTINUATION_V2_FEATURES].fillna(0.0))[:, 1]
    return pd.Series(p, index=df.index, name="continuation_v2_p")
