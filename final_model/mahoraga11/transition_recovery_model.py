from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mahoraga11_config import Mahoraga11Config
from mahoraga11_utils import time_split_index


TRANSITION_FEATURES = [
    "base_ret_1w",
    "base_ret_2w",
    "base_dd",
    "base_dd_change_2w",
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
    "base_eff_2w",
    "loss_share_2w",
    "breadth_63",
    "qqq_ret_2w",
    "qqq_rebound_2w",
    "corr_persist_4w",
    "transition_hawkes_recovery",
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


def build_hawkes_transition_features(
    weekly_df: pd.DataFrame,
    cfg: Mahoraga11Config,
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

    stressed_now = (
        (out["base_dd"] <= out["base_dd"].loc[train_idx].quantile(0.50))
        | (out["transition_hawkes_stress"] >= out["transition_hawkes_stress"].loc[train_idx].quantile(0.60))
    )
    out["transition_y"] = ((fwd2 <= q2_low) | (dd2 <= dd2_low)).astype(int)
    out["recovery_y"] = (
        stressed_now
        & (fwd2 >= q2_hi)
        & (qqq_fwd2 > 0.0)
        & (dd2 > -0.02)
    ).astype(int)
    return out


def _fit_time_model(
    train_weekly: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    cfg: Mahoraga11Config,
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


def fit_transition_model(train_weekly: pd.DataFrame, cfg: Mahoraga11Config, outer_parallel: bool) -> Dict[str, Any]:
    return _fit_time_model(
        train_weekly,
        TRANSITION_FEATURES,
        "transition_y",
        cfg,
        outer_parallel,
        "logit_transition",
        "rf_transition",
    )


def fit_recovery_model(train_weekly: pd.DataFrame, cfg: Mahoraga11Config, outer_parallel: bool) -> Dict[str, Any]:
    return _fit_time_model(
        train_weekly,
        RECOVERY_FEATURES,
        "recovery_y",
        cfg,
        outer_parallel,
        "logit_recovery",
        "rf_recovery",
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
