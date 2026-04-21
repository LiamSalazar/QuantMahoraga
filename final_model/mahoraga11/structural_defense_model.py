from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mahoraga11_config import Mahoraga11Config
from mahoraga11_utils import time_split_index


STRUCTURAL_FEATURES = [
    "base_dd",
    "base_dd_change_5d",
    "base_ret_4w",
    "base_rebound_4w",
    "base_eff_4w",
    "loss_share_4w",
    "down_up_ratio_4w",
    "stop_density_4w",
    "turnover_2w",
    "base_minus_qqq_4w",
    "avg_corr_63",
    "corr_persist_21",
    "breadth_63",
    "xs_disp_21",
    "qqq_drawdown",
    "vix_z_63",
    "crisis_scale",
    "turb_scale",
    "corr_rho",
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


def annotate_structural_labels(weekly_df: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    out = weekly_df.copy()
    fwd4 = _future_compound_return(out["base_r"], 4)
    fwd6 = _future_compound_return(out["base_r"], 6)
    dd4 = _future_drawdown_change(out["base_eq"], 4)
    rel4 = _future_compound_return(out["base_r"] - out["qqq_ret_2w"].div(2.0), 4)

    train_idx = out.loc[:train_end].index
    q4_low = float(fwd4.loc[train_idx].dropna().quantile(0.35)) if len(train_idx) else -0.03
    q6_low = float(fwd6.loc[train_idx].dropna().quantile(0.35)) if len(train_idx) else -0.04
    dd4_low = float(dd4.loc[train_idx].dropna().quantile(0.35)) if len(train_idx) else -0.04
    rel4_low = float(rel4.loc[train_idx].dropna().quantile(0.35)) if len(train_idx) else -0.02

    out["structural_y"] = (
        (fwd4 <= q4_low)
        | (fwd6 <= q6_low)
        | (dd4 <= dd4_low)
        | (rel4 <= rel4_low)
    ).astype(int)
    return out


def fit_structural_defense_model(
    train_weekly: pd.DataFrame,
    cfg: Mahoraga11Config,
    outer_parallel: bool,
) -> Dict[str, Any]:
    df = train_weekly[STRUCTURAL_FEATURES + ["structural_y"]].dropna().copy()
    if len(df) < max(cfg.min_train_weeks, 40) or df["structural_y"].nunique() < 2:
        base_prob = float(df["structural_y"].mean()) if len(df) else 0.0
        return {"name": "neutral", "model": None, "base_prob": base_prob, "score": 0.5}

    cut = time_split_index(df.index, cfg.inner_val_frac, cfg.min_train_weeks)
    tr = df.iloc[:cut].copy()
    va = df.iloc[cut:].copy()
    X_tr, y_tr = tr[STRUCTURAL_FEATURES], tr["structural_y"]
    X_va, y_va = va[STRUCTURAL_FEATURES], va["structural_y"]

    models = []

    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.7, max_iter=2000, class_weight="balanced"),
    )
    logit.fit(X_tr, y_tr)
    p_logit = pd.Series(logit.predict_proba(X_va)[:, 1], index=X_va.index)
    auc_logit = roc_auc_score(y_va, p_logit) if y_va.nunique() > 1 else 0.5
    models.append({"name": "logit_struct", "model": logit, "score": float(auc_logit)})

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
        models.append({"name": "rf_struct", "model": rf, "score": float(auc_rf)})

    return max(models, key=lambda x: x["score"])


def apply_structural_defense_model(model_info: Dict[str, Any], weekly_df: pd.DataFrame) -> pd.Series:
    if model_info.get("model") is None:
        return pd.Series(float(model_info.get("base_prob", 0.0)), index=weekly_df.index, name="structural_p")
    p = model_info["model"].predict_proba(weekly_df[STRUCTURAL_FEATURES].fillna(0.0))[:, 1]
    return pd.Series(p, index=weekly_df.index, name="structural_p")
