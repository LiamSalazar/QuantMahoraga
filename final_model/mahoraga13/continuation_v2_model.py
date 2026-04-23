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
    "swing_amplitude_vs_net_disp",
    "path_efficiency",
    "stop_pressure_release",
    "stop_density_decay",
    "corr_release_4w",
    "breadth_rebound_4w",
    "scale_recovery_2w",
    "qqq_continuation",
    "qqq_rebound_2w",
    "local_compression",
    "local_breakout_efficiency",
    "transition_hawkes_stress",
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


def _future_positive_share(r: pd.Series, horizon: int) -> pd.Series:
    out = pd.Series(np.nan, index=r.index, dtype=float)
    vals = r.fillna(0.0).values
    for i in range(len(r)):
        j = min(len(r), i + horizon + 1)
        if j <= i + 1:
            continue
        out.iloc[i] = float(np.mean(vals[i + 1:j] > 0.0))
    return out


def _train_quantile(series: pd.Series, q: float, default: float) -> float:
    s = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return float(default)
    return float(s.quantile(q))


def augment_continuation_v2_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    out = weekly_df.copy()
    net_disp = out["base_ret_4w"].abs().clip(lower=0.01)
    swing_amp = out["xs_disp_21"].abs() + out["down_up_ratio_4w"].clip(lower=0.0)
    path_eff = out["base_eff_4w"].clip(lower=0.0, upper=1.0)

    out["swing_amplitude_vs_net_disp"] = (swing_amp / net_disp).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["path_efficiency"] = path_eff.fillna(0.0)
    out["stop_pressure_release"] = out["stop_release_2w"].fillna(0.0)
    out["stop_density_decay"] = (-out["stop_density_4w"].diff()).fillna(0.0)
    out["qqq_continuation"] = (0.5 * out["qqq_ret_2w"] + 0.5 * out["qqq_rebound_2w"]).fillna(0.0)
    out["local_compression"] = (out["swing_amplitude_vs_net_disp"] * (1.0 - out["path_efficiency"])).fillna(0.0)
    out["local_breakout_efficiency"] = (
        out["base_ret_1w"].clip(lower=-0.15, upper=0.15)
        * (0.5 + 0.5 * out["base_eff_2w"].clip(lower=0.0, upper=1.0))
    ).fillna(0.0)
    return out


def _guard_thresholds(train_df: pd.DataFrame) -> Dict[str, float]:
    return {
        "compression_min": _train_quantile(train_df["local_compression"], 0.60, 0.8),
        "swing_ratio_min": _train_quantile(train_df["swing_amplitude_vs_net_disp"], 0.60, 2.0),
        "breakout_min": _train_quantile(train_df["local_breakout_efficiency"], 0.60, 0.005),
        "release_min": _train_quantile(train_df["stop_pressure_release"], 0.55, 0.0),
        "decay_min": _train_quantile(train_df["stop_density_decay"], 0.55, 0.0),
        "corr_min": _train_quantile(train_df["corr_release_4w"], 0.55, 0.0),
        "breadth_min": _train_quantile(train_df["breadth_rebound_4w"], 0.55, 0.0),
        "scale_min": _train_quantile(train_df["scale_recovery_2w"], 0.55, 0.0),
        "qqq_min": _train_quantile(train_df["qqq_continuation"], 0.55, 0.0),
        "base_dd_floor": _train_quantile(train_df["base_dd"], 0.20, -0.08),
        "base_dd_cap": _train_quantile(train_df["base_dd"], 0.75, -0.01),
        "loss_share_cap": _train_quantile(train_df["loss_share_4w"], 0.75, 0.65),
    }


def annotate_continuation_v2_labels(weekly_df: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    out = augment_continuation_v2_features(weekly_df)
    train_df = out.loc[:train_end].copy()
    thresholds = _guard_thresholds(train_df)

    fwd4 = _future_compound_return(out["base_r"], 4)
    dd4 = _future_drawdown_change(out["base_eq"], 4)
    pos4 = _future_positive_share(out["base_r"], 4)

    future_ret_min = _train_quantile(fwd4.loc[train_df.index], 0.65, 0.02)
    future_dd_floor = _train_quantile(dd4.loc[train_df.index], 0.45, -0.02)
    future_pos_min = _train_quantile(pos4.loc[train_df.index], 0.60, 0.60)

    compression_now = (
        (out["local_compression"] >= thresholds["compression_min"])
        & (out["swing_amplitude_vs_net_disp"] >= thresholds["swing_ratio_min"])
    )
    pause_now = (
        (out["base_dd"] >= thresholds["base_dd_floor"])
        & (out["base_dd"] <= thresholds["base_dd_cap"])
        & (out["loss_share_4w"] <= thresholds["loss_share_cap"])
    )
    release_now = (
        (out["local_breakout_efficiency"] >= thresholds["breakout_min"])
        & (
            (out["stop_pressure_release"] >= thresholds["release_min"])
            | (out["stop_density_decay"] >= thresholds["decay_min"])
            | (out["corr_release_4w"] >= thresholds["corr_min"])
            | (out["breadth_rebound_4w"] >= thresholds["breadth_min"])
            | (out["scale_recovery_2w"] >= thresholds["scale_min"])
            | (out["qqq_continuation"] >= thresholds["qqq_min"])
        )
    )
    persist_future = (
        (fwd4 >= future_ret_min)
        & (dd4 >= future_dd_floor)
        & (pos4 >= future_pos_min)
    )

    out["continuation_v2_y"] = (compression_now & pause_now & release_now & persist_future).astype(int)
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

    df = train_df[CONTINUATION_V2_FEATURES + ["continuation_v2_y"]].dropna().copy()
    if len(df) == 0:
        return floor

    cut = time_split_index(df.index, cfg.inner_val_frac, cfg.min_train_weeks)
    va = df.iloc[cut:].copy()
    if len(va) == 0:
        return floor

    probs = pd.Series(model_info["model"].predict_proba(va[CONTINUATION_V2_FEATURES])[:, 1], index=va.index)
    y_va = va["continuation_v2_y"]
    pos_probs = probs.loc[y_va == 1]
    neg_probs = probs.loc[y_va == 0]

    if len(pos_probs):
        thr = float(pos_probs.median())
        if len(neg_probs):
            thr = max(thr, float(neg_probs.quantile(0.75)))
    else:
        thr = float(probs.quantile(0.75))
    return float(np.clip(thr, floor, cap))


def fit_continuation_v2_model(train_weekly: pd.DataFrame, cfg: Mahoraga13Config, outer_parallel: bool) -> Dict[str, Any]:
    train_df = augment_continuation_v2_features(train_weekly)
    if "continuation_v2_y" not in train_df.columns:
        train_df = annotate_continuation_v2_labels(train_df, pd.Timestamp(train_df.index.max()))

    model_info = _fit_time_model(
        train_df,
        CONTINUATION_V2_FEATURES,
        "continuation_v2_y",
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
