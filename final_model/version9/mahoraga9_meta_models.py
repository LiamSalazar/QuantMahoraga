from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mahoraga9_config import Mahoraga9Config


FEATURE_COLS = [
    "avg_corr_21", "avg_corr_63", "breadth_63", "xs_disp_5", "xs_disp_21",
    "qqq_ret_5", "qqq_ret_21", "qqq_drawdown", "qqq_vol_21", "vix_level",
    "vix_z_63", "beta_63", "base_exposure", "base_turnover", "crisis_scale",
    "turb_scale", "corr_scale", "stress_intensity", "recovery_intensity",
]


@dataclass
class ModelFit:
    name: str
    estimator: object
    auc_val: float
    brier_val: float
    prob_col: str


def _fit_logit(X_tr, y_tr, c: float):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=c, max_iter=2000, class_weight="balanced")),
    ]).fit(X_tr, y_tr)


def _fit_rf(X_tr, y_tr, cfg: Mahoraga9Config):
    return RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=cfg.random_seed,
    ).fit(X_tr, y_tr)


def _score_model(model, X_val, y_val):
    p = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, p) if len(np.unique(y_val)) > 1 else 0.5
    brier = brier_score_loss(y_val, p)
    return float(auc), float(brier), p


def fit_best_model(df: pd.DataFrame, y_col: str, cfg: Mahoraga9Config) -> Dict:
    work = df.dropna(subset=FEATURE_COLS + [y_col]).copy()
    if len(work) < max(60, cfg.min_train_weeks):
        return {"name": "neutral", "estimator": None, "auc_val": np.nan, "brier_val": np.nan}

    cut = int(len(work) * (1.0 - cfg.inner_val_frac))
    cut = max(cut, min(len(work) - 20, cfg.min_train_weeks))
    train_df = work.iloc[:cut].copy()
    val_df = work.iloc[cut:].copy()
    X_tr, y_tr = train_df[FEATURE_COLS], train_df[y_col].astype(int)
    X_val, y_val = val_df[FEATURE_COLS], val_df[y_col].astype(int)

    fits: List[Dict] = []
    if "logit" in cfg.meta_model_candidates:
        for c in cfg.logit_c_grid:
            est = _fit_logit(X_tr, y_tr, c)
            auc, brier, _ = _score_model(est, X_val, y_val)
            fits.append({"name": f"logit_C{c}", "estimator": est, "auc_val": auc, "brier_val": brier})
    if "rf" in cfg.meta_model_candidates:
        est = _fit_rf(X_tr, y_tr, cfg)
        auc, brier, _ = _score_model(est, X_val, y_val)
        fits.append({"name": "rf", "estimator": est, "auc_val": auc, "brier_val": brier})

    fits = sorted(fits, key=lambda d: (-d["auc_val"], d["brier_val"]))
    return fits[0] if fits else {"name": "neutral", "estimator": None, "auc_val": np.nan, "brier_val": np.nan}


def apply_model(model_fit: Dict, df: pd.DataFrame, out_col: str) -> pd.Series:
    if model_fit.get("estimator") is None:
        return pd.Series(0.5, index=df.index, name=out_col, dtype=float)
    p = model_fit["estimator"].predict_proba(df[FEATURE_COLS])[:, 1]
    return pd.Series(p, index=df.index, name=out_col, dtype=float)
