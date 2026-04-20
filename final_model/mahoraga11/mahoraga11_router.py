from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mahoraga11_config import Mahoraga11Config
from mahoraga11_hawkes import build_hawkes_features
from mahoraga11_utils import iter_grid, sigmoid, time_split_index


FEATURE_COLS = [
    "avg_corr_21", "avg_corr_63", "breadth_63", "xs_disp_21", "qqq_ret_2w",
    "qqq_drawdown", "qqq_vol_21", "vix_z_63", "crisis_scale", "turb_scale",
    "corr_rho", "base_exposure", "base_turnover",
]


LABEL_COLS = ["structural_y", "fast_fragility_y", "recovery_y"]


def _avg_pairwise_corr(rets: pd.DataFrame, window: int, idx: pd.DatetimeIndex) -> pd.Series:
    cols = list(rets.columns)
    if len(cols) <= 1:
        return pd.Series(0.0, index=idx)
    rolling_corr = rets.rolling(window).corr()
    eye = pd.DataFrame(np.eye(len(cols), dtype=bool), index=cols, columns=cols)
    vals = []
    dates = []
    for dt, mat in rolling_corr.groupby(level=0):
        block = mat.droplevel(0)
        block = block.reindex(index=cols, columns=cols)
        s = block.mask(eye).stack(dropna=True)
        vals.append(float(s.mean()) if len(s) else np.nan)
        dates.append(dt)
    return pd.Series(vals, index=pd.DatetimeIndex(dates)).reindex(idx).ffill().fillna(0.0)


def build_daily_context(base_bt: Dict[str, Any], ohlcv: Dict[str, pd.DataFrame], cfg: Mahoraga11Config, pre: Dict[str, Any]) -> pd.DataFrame:
    idx = pre["idx"]
    qqq = pre["qqq"].reindex(idx).ffill()
    vix = ohlcv["close"][cfg.bench_vix].reindex(idx).ffill() if cfg.bench_vix in ohlcv["close"].columns else pd.Series(np.nan, index=idx)
    rets = pre["close"].pct_change().replace([np.inf, -np.inf], np.nan)
    qqq_r = qqq.pct_change().fillna(0.0)
    qqq_eq = (1.0 + qqq_r).cumprod()
    qqq_dd = qqq_eq / qqq_eq.cummax() - 1.0

    out = pd.DataFrame(index=idx)
    out["base_r"] = base_bt["returns_net"].reindex(idx).fillna(0.0)
    out["avg_corr_21"] = _avg_pairwise_corr(rets, 21, idx)
    out["avg_corr_63"] = _avg_pairwise_corr(rets, 63, idx)
    out["breadth_63"] = (rets.rolling(63).mean() > 0).mean(axis=1).fillna(0.0)
    out["xs_disp_21"] = rets.rolling(21).std(ddof=1).mean(axis=1).fillna(0.0)
    out["qqq_ret_2w"] = qqq.pct_change(10).fillna(0.0)
    out["qqq_drawdown"] = qqq_dd.fillna(0.0)
    out["qqq_vol_21"] = (qqq_r.rolling(21).std(ddof=1) * np.sqrt(cfg.trading_days)).ffill().fillna(0.0)
    out["vix_z_63"] = ((vix - vix.rolling(63).mean()) / vix.rolling(63).std(ddof=1).replace(0.0, np.nan)).fillna(0.0)
    out["crisis_scale"] = base_bt.get("crisis_scale", pd.Series(1.0, index=idx)).reindex(idx).fillna(1.0)
    out["turb_scale"] = base_bt.get("turb_scale", pd.Series(1.0, index=idx)).reindex(idx).fillna(1.0)
    out["corr_rho"] = pre["corr_rho"].reindex(idx).fillna(0.0)
    out["base_exposure"] = base_bt.get("exposure", pd.Series(0.0, index=idx)).reindex(idx).fillna(0.0)
    out["base_turnover"] = base_bt.get("turnover", pd.Series(0.0, index=idx)).reindex(idx).fillna(0.0)
    return out


def _future_compound_return(r: pd.Series, horizon: int) -> pd.Series:
    out = pd.Series(np.nan, index=r.index, dtype=float)
    vals = r.fillna(0.0).values
    for i in range(len(r)):
        j = min(len(r), i + horizon + 1)
        if j <= i + 1:
            continue
        seg = vals[i + 1:j]
        out.iloc[i] = float(np.prod(1.0 + seg) - 1.0)
    return out


def _future_drawdown_change(eq: pd.Series, horizon: int) -> pd.Series:
    eq = eq.fillna(method="ffill")
    dd = eq / eq.cummax() - 1.0
    out = pd.Series(np.nan, index=eq.index, dtype=float)
    vals = dd.values
    for i in range(len(eq)):
        j = min(len(eq), i + horizon + 1)
        if j <= i + 1:
            continue
        out.iloc[i] = float(np.nanmin(vals[i + 1:j]) - vals[i])
    return out


def build_weekly_dataset(daily_ctx: pd.DataFrame, cfg: Mahoraga11Config, train_end: pd.Timestamp) -> pd.DataFrame:
    weekly = pd.DataFrame(index=daily_ctx.resample(cfg.decision_freq).last().dropna().index)
    for c in daily_ctx.columns:
        if c == "base_r":
            weekly[c] = (1.0 + daily_ctx[c]).resample(cfg.decision_freq).prod().reindex(weekly.index).fillna(1.0) - 1.0
        else:
            weekly[c] = daily_ctx[c].resample(cfg.decision_freq).last().reindex(weekly.index).ffill().fillna(0.0)

    weekly["base_eq"] = (1.0 + weekly["base_r"].fillna(0.0)).cumprod()
    fwd2 = _future_compound_return(weekly["base_r"], 2)
    fwd4 = _future_compound_return(weekly["base_r"], 4)
    dd2 = _future_drawdown_change(weekly["base_eq"], 2)
    dd4 = _future_drawdown_change(weekly["base_eq"], 4)
    qqq_fwd2 = weekly["qqq_ret_2w"].shift(-2)

    train_idx = weekly.loc[:train_end].index
    train_fwd2 = fwd2.loc[train_idx].dropna()
    train_fwd4 = fwd4.loc[train_idx].dropna()
    train_dd2 = dd2.loc[train_idx].dropna()
    train_dd4 = dd4.loc[train_idx].dropna()

    q2_low = float(train_fwd2.quantile(0.35)) if len(train_fwd2) else -0.02
    q4_low = float(train_fwd4.quantile(0.35)) if len(train_fwd4) else -0.03
    dd2_low = float(train_dd2.quantile(0.35)) if len(train_dd2) else -0.03
    dd4_low = float(train_dd4.quantile(0.35)) if len(train_dd4) else -0.05
    q2_hi = float(train_fwd2.quantile(0.70)) if len(train_fwd2) else 0.02

    weekly["structural_y"] = ((fwd4 <= q4_low) | (dd4 <= dd4_low)).astype(int)
    weekly["fast_fragility_y"] = ((fwd2 <= q2_low) | (dd2 <= dd2_low)).astype(int)
    weekly["recovery_y"] = ((fwd2 >= q2_hi) & (qqq_fwd2 > 0.0) & (dd2 > -0.02)).astype(int)
    weekly[LABEL_COLS] = weekly[LABEL_COLS].fillna(0).astype(int)
    return weekly


def fit_best_classifier(train_weekly: pd.DataFrame, label_col: str, cfg: Mahoraga11Config, outer_parallel: bool) -> Dict[str, Any]:
    df = train_weekly[FEATURE_COLS + [label_col]].dropna().copy()
    if len(df) < max(cfg.min_train_weeks, 40) or df[label_col].nunique() < 2:
        p = float(df[label_col].mean()) if len(df) else 0.0
        return {"name": "neutral", "model": None, "base_prob": p, "score": 0.5}

    cut = time_split_index(df.index, cfg.inner_val_frac, cfg.min_train_weeks)
    tr = df.iloc[:cut].copy()
    va = df.iloc[cut:].copy()
    X_tr, y_tr = tr[FEATURE_COLS], tr[label_col]
    X_va, y_va = va[FEATURE_COLS], va[label_col]

    candidates: List[Dict[str, Any]] = []

    logit = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    logit.fit(X_tr, y_tr)
    p_log = pd.Series(logit.predict_proba(X_va)[:, 1], index=X_va.index)
    auc_log = roc_auc_score(y_va, p_log) if y_va.nunique() > 1 else 0.5
    candidates.append({"name": "logit_C1.0", "model": logit, "score": float(auc_log)})

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
        candidates.append({"name": "rf", "model": rf, "score": float(auc_rf)})

    best = max(candidates, key=lambda x: x["score"])
    return best


def apply_classifier(model_info: Dict[str, Any], weekly_df: pd.DataFrame, label_name: str) -> pd.Series:
    if model_info.get("model") is None:
        return pd.Series(float(model_info.get("base_prob", 0.0)), index=weekly_df.index, name=label_name)
    p = model_info["model"].predict_proba(weekly_df[FEATURE_COLS].fillna(0.0))[:, 1]
    return pd.Series(p, index=weekly_df.index, name=label_name)


def iter_router_candidates(cfg: Mahoraga11Config):
    yield from iter_grid(cfg.router_grid())


def build_router_weekly(
    weekly_df: pd.DataFrame,
    router_params: Dict[str, float],
    cfg: Mahoraga11Config,
) -> pd.DataFrame:
    decay = 0.70
    hawkes = build_hawkes_features(weekly_df, decay, cfg) if cfg.use_hawkes else pd.DataFrame(index=weekly_df.index, data={"stress_intensity": 0.0, "recovery_intensity": 0.0})
    out = weekly_df.join(hawkes, how="left").copy()

    hawkes_weight = float(router_params["hawkes_weight"])
    stress_int = out.get("stress_intensity", pd.Series(0.0, index=out.index)).fillna(0.0)
    recv_int = out.get("recovery_intensity", pd.Series(0.0, index=out.index)).fillna(0.0)
    stress_norm = pd.Series(sigmoid(stress_int - recv_int), index=out.index)
    recv_norm = pd.Series(sigmoid(recv_int - stress_int), index=out.index)

    structural_score = out["structural_p"].fillna(0.0) + 0.5 * hawkes_weight * stress_norm
    fast_score = out["fast_fragility_p"].fillna(0.0) + hawkes_weight * stress_norm
    recovery_score = out["recovery_p"].fillna(0.0) + hawkes_weight * recv_norm

    structural_mask = structural_score >= float(router_params["structural_prob_thr"])
    fast_mask = fast_score >= float(router_params["fast_prob_thr"])
    recovery_mask = (~fast_mask) & (recovery_score >= float(router_params["recovery_prob_thr"]))

    # Require some persistence before committing hard to the floor engine.
    structural_persist = structural_mask.rolling(2, min_periods=1).mean() >= 0.5

    out["floor_blend"] = 0.0
    out.loc[structural_persist, "floor_blend"] = float(router_params["floor_blend_max"])
    out.loc[recovery_mask, "floor_blend"] = out.loc[recovery_mask, "floor_blend"] * 0.25

    out["gate_scale"] = 1.0
    out["vol_mult"] = 1.0
    out["exp_cap"] = 1.0
    out["mode"] = "CEILING"
    out["is_intervening"] = 0.0

    out.loc[structural_persist, "mode"] = "FLOOR"
    out.loc[structural_persist, "is_intervening"] = 1.0

    out.loc[fast_mask, "gate_scale"] = float(router_params["gate_floor"])
    out.loc[fast_mask, "vol_mult"] = float(router_params["vol_mult_stress"])
    out.loc[fast_mask, "exp_cap"] = float(router_params["max_exp_stress"])
    out.loc[fast_mask, "mode"] = np.where(structural_persist.loc[fast_mask], "FLOOR", "DEFENSIVE_LIGHT")
    out.loc[fast_mask, "is_intervening"] = 1.0

    out.loc[recovery_mask, "vol_mult"] = float(router_params["vol_mult_recovery"])
    out.loc[recovery_mask, "exp_cap"] = float(router_params["max_exp_recovery"])
    out.loc[recovery_mask, "mode"] = "TRANSITION"
    out.loc[recovery_mask, "is_intervening"] = 1.0

    return out


def weekly_to_daily_router(router_weekly: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    cols = ["gate_scale", "vol_mult", "exp_cap", "floor_blend", "is_intervening", "mode"]
    out = router_weekly[cols].reindex(idx).ffill().bfill()
    out["floor_blend"] = out["floor_blend"].fillna(0.0).clip(0.0, 1.0)
    out["gate_scale"] = out["gate_scale"].fillna(1.0).clip(0.0, 1.0)
    out["vol_mult"] = out["vol_mult"].fillna(1.0)
    out["exp_cap"] = out["exp_cap"].fillna(1.0)
    out["is_intervening"] = out["is_intervening"].fillna(0.0)
    out["mode"] = out["mode"].fillna("CEILING")
    return out
