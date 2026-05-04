from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mahoraga10_config import Mahoraga10Config
from mahoraga10_hawkes import build_hawkes_features
from mahoraga10_utils import iter_grid, time_split_index


FEATURE_COLS = [
    "avg_corr_21", "avg_corr_63", "breadth_63", "xs_disp_21", "qqq_ret_2w",
    "qqq_drawdown", "qqq_vol_21", "vix_z_63", "crisis_scale", "turb_scale",
    "corr_rho", "base_exposure", "base_turnover",
]


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


def build_daily_context(base_bt: Dict[str, Any], ohlcv: Dict[str, pd.DataFrame], cfg: Mahoraga10Config, pre: Dict[str, Any]) -> pd.DataFrame:
    idx = pre["idx"]
    qqq = pre["qqq"].reindex(idx).ffill()
    vix = ohlcv["close"][cfg.bench_vix].reindex(idx).ffill() if cfg.bench_vix in ohlcv["close"].columns else pd.Series(np.nan, index=idx)
    rets = pre["close"].pct_change().replace([np.inf, -np.inf], np.nan)
    qqq_r = qqq.pct_change().fillna(0.0)
    qqq_eq = (1.0 + qqq_r).cumprod()
    qqq_dd = qqq_eq / qqq_eq.cummax() - 1.0

    out = pd.DataFrame(index=idx)
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


def build_weekly_dataset(daily_ctx: pd.DataFrame, cfg: Mahoraga10Config, train_end: pd.Timestamp) -> pd.DataFrame:
    weekly = pd.DataFrame(index=daily_ctx.resample(cfg.decision_freq).last().dropna().index)
    for c in daily_ctx.columns:
        weekly[c] = daily_ctx[c].resample(cfg.decision_freq).last().reindex(weekly.index).ffill().fillna(0.0)

    train_idx = weekly.loc[:train_end].index
    fwd_ret = weekly["qqq_ret_2w"].shift(-cfg.weekly_horizon_weeks)
    train_fwd = fwd_ret.loc[train_idx].dropna()
    if len(train_fwd) == 0:
        q_low, q_high = -0.02, 0.02
    else:
        q_low = float(train_fwd.quantile(0.30))
        q_high = float(train_fwd.quantile(0.70))

    dd_change = weekly["qqq_drawdown"].shift(-cfg.weekly_horizon_weeks) - weekly["qqq_drawdown"]
    weekly["fragility_y"] = ((fwd_ret <= q_low) | (dd_change < -0.03)).astype(int)
    weekly["recovery_y"] = ((fwd_ret >= q_high) & (dd_change > 0.02)).astype(int)
    weekly[["fragility_y", "recovery_y"]] = weekly[["fragility_y", "recovery_y"]].fillna(0).astype(int)
    return weekly


def fit_best_classifier(train_weekly: pd.DataFrame, label_col: str, cfg: Mahoraga10Config, outer_parallel: bool) -> Dict[str, Any]:
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


def iter_policy_candidates(cfg: Mahoraga10Config):
    yield from iter_grid(cfg.policy_grid())


def build_policy_weekly(weekly_df: pd.DataFrame, alpha_mix_base: float, policy_params: Dict[str, float], cfg: Mahoraga10Config) -> pd.DataFrame:
    decay = float(0.70)
    if "hawkes_decay" in policy_params:
        decay = float(policy_params["hawkes_decay"])
    hawkes = build_hawkes_features(weekly_df, decay, cfg) if cfg.use_hawkes else pd.DataFrame(index=weekly_df.index, data={"stress_intensity": 0.0, "recovery_intensity": 0.0})

    out = weekly_df.join(hawkes, how="left").copy()
    frag_score = out["fragility_p"].fillna(0.0) + float(policy_params["hawkes_weight"]) * out.get("stress_intensity", pd.Series(0.0, index=out.index)).fillna(0.0)
    recv_score = out["recovery_p"].fillna(0.0) + float(policy_params["hawkes_weight"]) * out.get("recovery_intensity", pd.Series(0.0, index=out.index)).fillna(0.0)

    out["gate_scale"] = 1.0
    out["vol_mult"] = 1.0
    out["exp_cap"] = 1.0
    out["alpha_mix"] = float(alpha_mix_base)
    out["is_intervening"] = 0.0

    frag_mask = frag_score >= float(policy_params["fragility_prob_thr"])
    recv_mask = (~frag_mask) & (recv_score >= float(policy_params["recovery_prob_thr"]))

    out.loc[frag_mask, "gate_scale"] = float(policy_params["gate_floor"])
    out.loc[frag_mask, "vol_mult"] = float(policy_params["vol_mult_stress"])
    out.loc[frag_mask, "exp_cap"] = float(policy_params["max_exp_stress"])
    out.loc[frag_mask, "alpha_mix"] = np.clip(float(alpha_mix_base) + float(policy_params["alpha_tilt"]), 0.0, 1.0)
    out.loc[frag_mask, "is_intervening"] = 1.0

    out.loc[recv_mask, "vol_mult"] = float(policy_params["vol_mult_recovery"])
    out.loc[recv_mask, "exp_cap"] = float(policy_params["max_exp_recovery"])
    out.loc[recv_mask, "alpha_mix"] = np.clip(float(alpha_mix_base) - float(policy_params["alpha_tilt"]), 0.0, 1.0)
    out.loc[recv_mask, "is_intervening"] = 1.0
    return out


def weekly_to_daily_policy(policy_weekly: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    cols = ["gate_scale", "vol_mult", "exp_cap", "alpha_mix", "is_intervening"]
    return policy_weekly[cols].reindex(idx).ffill().bfill()
