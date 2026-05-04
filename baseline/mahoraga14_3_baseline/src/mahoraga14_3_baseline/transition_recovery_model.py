from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mahoraga14_config import Mahoraga14Config
from mahoraga14_utils import time_split_index


TRANSITION_FEATURES = [
    "base_ret_1w",
    "base_ret_2w",
    "base_dd",
    "base_dd_change_2w",
    "base_dd_velocity_2w",
    "base_rebound_2w",
    "loss_share_2w",
    "stop_density_2w",
    "base_eff_2w",
    "base_scale_gap",
    "avg_corr_21",
    "corr_persist_4w",
    "qqq_ret_2w",
    "qqq_drawdown",
    "vix_z_63",
    "transition_hawkes_stress",
]


RECOVERY_FEATURES = [
    "base_ret_1w",
    "base_dd",
    "base_dd_change_2w",
    "base_rebound_2w",
    "base_rebound_4w",
    "base_rebound_asym_4w",
    "base_eff_2w",
    "loss_share_2w",
    "stop_release_2w",
    "breadth_rebound_4w",
    "scale_recovery_2w",
    "qqq_ret_2w",
    "qqq_rebound_2w",
    "corr_release_4w",
    "corr_persist_4w",
    "transition_hawkes_recovery",
]


CONTINUATION_FEATURES = [
    "base_ret_1w",
    "base_ret_2w",
    "base_ret_4w",
    "base_dd",
    "base_dd_duration_4w",
    "base_dd_velocity_2w",
    "base_eff_2w",
    "base_eff_4w",
    "down_up_ratio_4w",
    "loss_share_4w",
    "xs_disp_21",
    "stop_release_2w",
    "breadth_rebound_4w",
    "scale_recovery_2w",
    "corr_release_4w",
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


def _future_positive_share(r: pd.Series, horizon: int) -> pd.Series:
    out = pd.Series(np.nan, index=r.index, dtype=float)
    vals = r.fillna(0.0).values
    for i in range(len(r)):
        j = min(len(r), i + horizon + 1)
        if j <= i + 1:
            continue
        out.iloc[i] = float(np.mean(vals[i + 1:j] > 0.0))
    return out


def build_hawkes_transition_features(
    weekly_df: pd.DataFrame,
    cfg: Mahoraga14Config,
    thresholds: Dict[str, float] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    out = pd.DataFrame(index=weekly_df.index)
    if thresholds is None:
        thresholds = {
            "qqq_low": float(weekly_df["qqq_ret_2w"].quantile(cfg.hawkes_event_q_low)) if len(weekly_df) else -0.02,
            "qqq_high": float(weekly_df["qqq_ret_2w"].quantile(cfg.hawkes_event_q_high)) if len(weekly_df) else 0.02,
            "base_ret_low": float(weekly_df["base_ret_1w"].quantile(0.25)) if len(weekly_df) else -0.02,
            "base_ddchg_low": float(weekly_df["base_dd_change_2w"].quantile(0.25)) if len(weekly_df) else -0.02,
            "breadth_low": float(weekly_df["breadth_63"].quantile(0.25)) if len(weekly_df) else 0.35,
            "breadth_high": float(weekly_df["breadth_63"].quantile(0.55)) if len(weekly_df) else 0.55,
            "rebound_high": float(weekly_df["base_rebound_2w"].quantile(0.60)) if len(weekly_df) else 0.02,
            "stop_release_high": float(weekly_df["stop_release_2w"].quantile(0.60)) if len(weekly_df) else 0.0,
        }

    stress_event = (
        (weekly_df["base_ret_1w"] <= thresholds["base_ret_low"])
        | (weekly_df["base_dd_change_2w"] <= thresholds["base_ddchg_low"])
        | (weekly_df["qqq_ret_2w"] <= thresholds["qqq_low"])
        | (weekly_df["vix_z_63"] >= 1.0)
        | (weekly_df["breadth_63"] <= thresholds["breadth_low"])
    ).astype(float)
    recovery_event = (
        (weekly_df["base_rebound_2w"] >= thresholds["rebound_high"])
        & (weekly_df["qqq_ret_2w"] >= thresholds["qqq_high"])
        & (weekly_df["breadth_63"] >= thresholds["breadth_high"])
        & (weekly_df["stop_release_2w"] >= thresholds["stop_release_high"])
    ).astype(float)

    stress_intensity = np.zeros(len(weekly_df), dtype=float)
    recovery_intensity = np.zeros(len(weekly_df), dtype=float)
    decay = float(cfg.hawkes_decay)
    for i in range(1, len(weekly_df)):
        stress_intensity[i] = decay * stress_intensity[i - 1] + float(stress_event.iloc[i - 1])
        recovery_intensity[i] = decay * recovery_intensity[i - 1] + float(recovery_event.iloc[i - 1])

    out["stress_event"] = stress_event
    out["recovery_event"] = recovery_event
    out["transition_hawkes_stress"] = pd.Series(stress_intensity, index=weekly_df.index).clip(0.0, 5.0)
    out["transition_hawkes_recovery"] = pd.Series(recovery_intensity, index=weekly_df.index).clip(0.0, 5.0)
    return out, thresholds


def annotate_transition_recovery_labels(weekly_df: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    out = weekly_df.copy()
    fwd2 = _future_compound_return(out["base_r"], 2)
    dd2 = _future_drawdown_change(out["base_eq"], 2)
    qqq_fwd2 = out["qqq_ret_2w"].shift(-2)

    train_idx = out.loc[:train_end].index
    q2_low = float(fwd2.loc[train_idx].dropna().quantile(0.35)) if len(train_idx) else -0.02
    dd2_low = float(dd2.loc[train_idx].dropna().quantile(0.35)) if len(train_idx) else -0.03
    q2_hi = float(fwd2.loc[train_idx].dropna().quantile(0.70)) if len(train_idx) else 0.02
    stop_release_hi = float(out.loc[train_idx, "stop_release_2w"].dropna().quantile(0.55)) if len(train_idx) else 0.0
    breadth_rebound_hi = float(out.loc[train_idx, "breadth_rebound_4w"].dropna().quantile(0.55)) if len(train_idx) else 0.0

    stressed_now = (
        (out["base_dd"] <= out["base_dd"].loc[train_idx].quantile(0.50))
        | (out["transition_hawkes_stress"] >= out["transition_hawkes_stress"].loc[train_idx].quantile(0.60))
    )
    release_now = (
        (out["stop_release_2w"] >= stop_release_hi)
        | (out["breadth_rebound_4w"] >= breadth_rebound_hi)
        | (out["corr_release_4w"] > 0.0)
    )

    out["transition_y"] = ((fwd2 <= q2_low) | (dd2 <= dd2_low)).astype(int)
    out["recovery_y"] = (
        stressed_now
        & release_now
        & (fwd2 >= q2_hi)
        & (qqq_fwd2 > 0.0)
        & (dd2 > -0.02)
    ).astype(int)
    return out


def annotate_continuation_labels(weekly_df: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    out = weekly_df.copy()
    train_idx = out.loc[:train_end].index

    fwd3 = _future_compound_return(out["base_r"], 3)
    dd3 = _future_drawdown_change(out["base_eq"], 3)
    pos3 = _future_positive_share(out["base_r"], 3)

    eff_low = float(out.loc[train_idx, "base_eff_4w"].dropna().quantile(0.40)) if len(train_idx) else 0.35
    chop_high = float(out.loc[train_idx, "down_up_ratio_4w"].dropna().quantile(0.60)) if len(train_idx) else 1.15
    dd_floor = float(out.loc[train_idx, "base_dd"].dropna().quantile(0.20)) if len(train_idx) else -0.08
    dd_cap = float(out.loc[train_idx, "base_dd"].dropna().quantile(0.70)) if len(train_idx) else -0.01
    loss_cap = float(out.loc[train_idx, "loss_share_4w"].dropna().quantile(0.70)) if len(train_idx) else 0.65
    release_high = float(out.loc[train_idx, "stop_release_2w"].dropna().quantile(0.55)) if len(train_idx) else 0.0
    breadth_high = float(out.loc[train_idx, "breadth_rebound_4w"].dropna().quantile(0.55)) if len(train_idx) else 0.0
    scale_high = float(out.loc[train_idx, "scale_recovery_2w"].dropna().quantile(0.55)) if len(train_idx) else 0.0
    corr_high = float(out.loc[train_idx, "corr_release_4w"].dropna().quantile(0.55)) if len(train_idx) else 0.0
    qqq_high = float(out.loc[train_idx, "qqq_rebound_2w"].dropna().quantile(0.55)) if len(train_idx) else 0.0
    ret_high = float(out.loc[train_idx, "base_ret_1w"].dropna().quantile(0.60)) if len(train_idx) else 0.01
    fwd_high = float(fwd3.loc[train_idx].dropna().quantile(0.70)) if len(train_idx) else 0.02
    dd_future_floor = float(dd3.loc[train_idx].dropna().quantile(0.45)) if len(train_idx) else -0.02
    pos_future_high = float(pos3.loc[train_idx].dropna().quantile(0.60)) if len(train_idx) else 2.0 / 3.0

    compression_now = (
        (out["base_eff_4w"] <= eff_low)
        & (out["down_up_ratio_4w"] >= chop_high)
    )
    pause_now = (
        (out["base_dd"] >= dd_floor)
        & (out["base_dd"] <= dd_cap)
        & (out["loss_share_4w"] <= loss_cap)
    )
    resume_now = (
        (out["base_ret_1w"] >= ret_high)
        | (out["stop_release_2w"] >= release_high)
        | (out["breadth_rebound_4w"] >= breadth_high)
        | (out["scale_recovery_2w"] >= scale_high)
        | (out["corr_release_4w"] >= corr_high)
        | (out["qqq_rebound_2w"] >= qqq_high)
    )
    persist_future = (
        (fwd3 >= fwd_high)
        & (dd3 >= dd_future_floor)
        & (pos3 >= pos_future_high)
    )

    out["continuation_y"] = (compression_now & pause_now & resume_now & persist_future).astype(int)
    return out


def _fit_time_model(
    train_weekly: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    cfg: Mahoraga14Config,
    outer_parallel: bool,
    logit_name: str,
    rf_name: str,
) -> Dict[str, Any]:
    df = train_weekly[feature_cols + [label_col]].dropna().copy()
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


def fit_transition_model(train_weekly: pd.DataFrame, cfg: Mahoraga14Config, outer_parallel: bool) -> Dict[str, Any]:
    return _fit_time_model(
        train_weekly,
        TRANSITION_FEATURES,
        "transition_y",
        cfg,
        outer_parallel,
        "logit_transition",
        "rf_transition",
    )


def fit_recovery_model(train_weekly: pd.DataFrame, cfg: Mahoraga14Config, outer_parallel: bool) -> Dict[str, Any]:
    return _fit_time_model(
        train_weekly,
        RECOVERY_FEATURES,
        "recovery_y",
        cfg,
        outer_parallel,
        "logit_recovery",
        "rf_recovery",
    )


def fit_continuation_model(train_weekly: pd.DataFrame, cfg: Mahoraga14Config, outer_parallel: bool) -> Dict[str, Any]:
    return _fit_time_model(
        train_weekly,
        CONTINUATION_FEATURES,
        "continuation_y",
        cfg,
        outer_parallel,
        "logit_continuation",
        "rf_continuation",
    )


def apply_transition_model(model_info: Dict[str, Any], weekly_df: pd.DataFrame) -> pd.Series:
    if model_info.get("model") is None:
        return pd.Series(float(model_info.get("base_prob", 0.0)), index=weekly_df.index, name="transition_p")
    p = model_info["model"].predict_proba(weekly_df[TRANSITION_FEATURES].fillna(0.0))[:, 1]
    return pd.Series(p, index=weekly_df.index, name="transition_p")


def apply_recovery_model(model_info: Dict[str, Any], weekly_df: pd.DataFrame) -> pd.Series:
    if model_info.get("model") is None:
        return pd.Series(float(model_info.get("base_prob", 0.0)), index=weekly_df.index, name="recovery_p")
    p = model_info["model"].predict_proba(weekly_df[RECOVERY_FEATURES].fillna(0.0))[:, 1]
    return pd.Series(p, index=weekly_df.index, name="recovery_p")


def apply_continuation_model(model_info: Dict[str, Any], weekly_df: pd.DataFrame) -> pd.Series:
    if model_info.get("model") is None:
        return pd.Series(float(model_info.get("base_prob", 0.0)), index=weekly_df.index, name="continuation_p")
    p = model_info["model"].predict_proba(weekly_df[CONTINUATION_FEATURES].fillna(0.0))[:, 1]
    return pd.Series(p, index=weekly_df.index, name="continuation_p")

