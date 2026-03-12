from __future__ import annotations

"""
MahoragaAI1.0
==============
Mode-operating and regime-severity AI layer over Mahoraga 6.1.

Design principles
-----------------
- Keep Mahoraga 6.1 as the execution/risk baseline.
- Do NOT predict prices directly.
- Use tabular ML to classify operating mode and estimate hostile-regime severity.
- Preserve walk-forward purity: models are trained per fold using only history available
  before the fold's test window.
- Keep the risk engine sovereign: AI only modulates mode/exposure.

Usage
-----
Place this file in the same directory as `mahoraga6_1.py` and run:
    python mahoragaAI1_0.py
"""

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

import mahoraga6_1 as m6


AI1_DISCLAIMER = """
═══════════════════════════════════════════════════════════════════════════════
  MAHORAGA AI1.0 — METHODOLOGY DISCLAIMER
───────────────────────────────────────────────────────────────────────────────
  This version adds a tabular machine-learning layer that selects an operating
  mode and estimates regime severity. It does NOT predict asset prices.
  The AI layer is evaluated in a walk-forward protocol and remains subordinate
  to Mahoraga 6.1's execution/risk engine. Results remain conditional on the
  same ex-post universe design limitations disclosed by Mahoraga 6.1.
═══════════════════════════════════════════════════════════════════════════════
"""


@dataclass
class MahoragaAI1Config(m6.Mahoraga6Config):
    plots_dir: str = "mahoragaAI1_0_plots"
    outputs_dir: str = "mahoragaAI1_0_outputs"
    label: str = "MAHORAGA_AI1_0"

    # Runtime
    parallel_sweep: bool = True

    # AI settings
    ai_enabled: bool = True
    ai_modes: Tuple[str, ...] = (
        "BALANCED",
        "REL_HEAVY",
        "MOM_HEAVY",
        "DEFENSIVE",
        "CASH_BIAS",
    )
    ai_base_mode: str = "BALANCED"
    ai_min_train_samples: int = 80
    ai_min_class_count: int = 8
    ai_min_mode_confidence: float = 0.35
    ai_recent_sample_bias: float = 1.50
    ai_label_dd_penalty: float = 0.40
    ai_severity_ret_thr: float = -0.020
    ai_severity_dd_thr: float = -0.040
    ai_panic_vix_label_thr: float = 24.0
    ai_severity_scale_strength: float = 0.65
    ai_severity_floor: float = 0.10
    ai_severity_high_prob: float = 0.70
    ai_mode_hard_defensive_prob: float = 0.80


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _mode_library(cfg: MahoragaAI1Config) -> Dict[str, Dict[str, float]]:
    """Operating modes chosen to create materially different portfolio/risk states."""
    return {
        "BALANCED":  {"w_trend": cfg.w_trend, "w_mom": cfg.w_mom, "w_rel": cfg.w_rel,
                       "exposure": 1.00, "top_k": float(cfg.top_k), "weight_cap": cfg.weight_cap},
        "REL_HEAVY": {"w_trend": 0.20, "w_mom": 0.20, "w_rel": 0.60,
                       "exposure": 1.00, "top_k": float(cfg.top_k), "weight_cap": cfg.weight_cap},
        "MOM_HEAVY": {"w_trend": 0.20, "w_mom": 0.60, "w_rel": 0.20,
                       "exposure": 1.00, "top_k": float(cfg.top_k), "weight_cap": cfg.weight_cap},
        "DEFENSIVE": {"w_trend": 0.25, "w_mom": 0.20, "w_rel": 0.55,
                       "exposure": 0.60, "top_k": float(max(2, cfg.top_k - 1)), "weight_cap": min(cfg.weight_cap, 0.45)},
        "CASH_BIAS": {"w_trend": 0.20, "w_mom": 0.15, "w_rel": 0.65,
                       "exposure": 0.20, "top_k": float(max(1, cfg.top_k - 2)), "weight_cap": min(cfg.weight_cap, 0.35)},
    }


def _component_scores(close: pd.DataFrame, qqq: pd.Series, cfg: MahoragaAI1Config) -> Dict[str, pd.DataFrame]:
    idx = close.index
    qqq_ = m6.to_s(qqq, "QQQ").reindex(idx).ffill()
    out = {"trend": pd.DataFrame(index=idx, columns=close.columns, dtype=float),
           "mom": pd.DataFrame(index=idx, columns=close.columns, dtype=float),
           "rel": pd.DataFrame(index=idx, columns=close.columns, dtype=float)}
    for t in close.columns:
        p = close[t].reindex(idx).ffill()
        tr = m6._trend(p, cfg)
        mo = m6._mom(p, cfg)
        re = m6._rel(p, qqq_, cfg)
        tr.iloc[:cfg.burn_in] = 0.0
        mo.iloc[:cfg.burn_in] = 0.0
        re.iloc[:cfg.burn_in] = 0.0
        out["trend"][t] = tr.fillna(0.0)
        out["mom"][t] = mo.fillna(0.0)
        out["rel"][t] = re.fillna(0.0)
    return out


def _avg_offdiag_corr(sub: pd.DataFrame) -> float:
    if sub.shape[1] < 2 or len(sub) < 5:
        return np.nan
    c = sub.corr().values
    n = c.shape[0]
    if n <= 1:
        return np.nan
    return float((c.sum() - n) / (n * (n - 1)))


def _active_members(dt: pd.Timestamp, universe_schedule: Optional[pd.DataFrame],
                    universe_static: Tuple[str, ...], available: List[str]) -> List[str]:
    if universe_schedule is not None and not universe_schedule.empty:
        members = m6.get_universe_at_date(universe_schedule, dt)
    else:
        members = list(universe_static)
    return [t for t in members if t in available]


def _build_meta_feature_table(
    ohlcv: Dict[str, pd.DataFrame],
    cfg: MahoragaAI1Config,
    universe_schedule: Optional[pd.DataFrame],
    comp: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    close = ohlcv["close"][comp["trend"].columns]
    volume = ohlcv["volume"][comp["trend"].columns].reindex(close.index)
    idx = close.index
    rets = close.pct_change().fillna(0.0)
    qqq = m6.to_s(ohlcv["close"][cfg.bench_qqq].reindex(idx).ffill(), "QQQ")
    qqq_r = qqq.pct_change().fillna(0.0)
    vix = None
    if cfg.bench_vix in ohlcv.get("close", pd.DataFrame()).columns:
        vix = m6.to_s(ohlcv["close"][cfg.bench_vix].reindex(idx).ffill(), "VIX")
    else:
        vix = pd.Series(np.nan, index=idx, name="VIX")

    crisis_scale, crisis_state = m6.compute_crisis_gate(qqq, cfg)
    turb_scale = m6.compute_turbulence(close, volume, qqq, cfg)
    corr_rho, corr_scale, corr_state = m6.compute_corr_shield_series(
        rets, idx, cfg, list(close.columns), True, universe_schedule=universe_schedule, vix=vix
    )

    reb_idx = close.resample(cfg.rebalance_freq).last().index
    rows = []
    for dt in reb_idx:
        if dt not in idx:
            continue
        members = _active_members(dt, universe_schedule, cfg.universe_static, list(close.columns))
        if not members:
            continue
        sub21 = rets[members].loc[:dt].tail(21).dropna(how="all")
        sub63 = rets[members].loc[:dt].tail(63).dropna(how="all")
        xs5 = close[members].pct_change(5).loc[dt]
        xs21 = close[members].pct_change(21).loc[dt]
        breadth63 = (close[members].loc[dt] > close[members].shift(63).loc[dt]).mean()
        tr_row = comp["trend"].loc[dt, members].astype(float)
        mo_row = comp["mom"].loc[dt, members].astype(float)
        re_row = comp["rel"].loc[dt, members].astype(float)
        row = {
            "avg_corr_21": _avg_offdiag_corr(sub21),
            "avg_corr_63": _avg_offdiag_corr(sub63),
            "xs_disp_5d": float(xs5.std()) if xs5.notna().sum() > 1 else np.nan,
            "xs_disp_21d": float(xs21.std()) if xs21.notna().sum() > 1 else np.nan,
            "breadth_63d": float(breadth63) if np.isfinite(breadth63) else np.nan,
            "qqq_vol_21": float(qqq_r.loc[:dt].tail(21).std() * np.sqrt(cfg.trading_days)),
            "qqq_vol_63": float(qqq_r.loc[:dt].tail(63).std() * np.sqrt(cfg.trading_days)),
            "qqq_ret_5d": float(qqq.pct_change(5).loc[dt]) if dt in qqq.index else np.nan,
            "qqq_ret_21d": float(qqq.pct_change(21).loc[dt]) if dt in qqq.index else np.nan,
            "qqq_drawdown": float((qqq.loc[:dt] / qqq.loc[:dt].cummax() - 1.0).iloc[-1]),
            "vix_level": float(vix.loc[dt]) if dt in vix.index and np.isfinite(vix.loc[dt]) else np.nan,
            "vix_chg_5d": float(vix.pct_change(5).loc[dt]) if dt in vix.index else np.nan,
            "vix_z_63": float(m6.safe_z(vix.fillna(method="ffill").fillna(0.0), 63).loc[dt]) if dt in vix.index else np.nan,
            "crisis_state": float(crisis_state.loc[dt]),
            "turb_scale": float(turb_scale.loc[dt]),
            "corr_rho": float(corr_rho.loc[dt]) if np.isfinite(corr_rho.loc[dt]) else np.nan,
            "corr_scale": float(corr_scale.loc[dt]),
            "trend_mean": float(tr_row.mean()),
            "mom_mean": float(mo_row.mean()),
            "rel_mean": float(re_row.mean()),
            "trend_top_mean": float(tr_row.nlargest(min(len(tr_row), cfg.top_k)).mean()) if len(tr_row) else np.nan,
            "mom_top_mean": float(mo_row.nlargest(min(len(mo_row), cfg.top_k)).mean()) if len(mo_row) else np.nan,
            "rel_top_mean": float(re_row.nlargest(min(len(re_row), cfg.top_k)).mean()) if len(re_row) else np.nan,
            "rel_minus_mom": float(re_row.mean() - mo_row.mean()),
            "rel_minus_trend": float(re_row.mean() - tr_row.mean()),
            "n_members": float(len(members)),
        }
        rows.append((dt, row))
    feat = pd.DataFrame({dt: row for dt, row in rows}).T.sort_index()
    feat.index.name = "date"
    return feat


def _select_mode_names(
    dt: pd.Timestamp,
    members: List[str],
    comp: Dict[str, pd.DataFrame],
    mode_def: Dict[str, float],
    top_k: int,
) -> List[str]:
    if not members:
        return []
    score = (
        mode_def["w_trend"] * comp["trend"].loc[dt, members].astype(float)
        + mode_def["w_mom"] * comp["mom"].loc[dt, members].astype(float)
        + mode_def["w_rel"] * comp["rel"].loc[dt, members].astype(float)
    )
    names = score.nlargest(top_k).index.tolist()
    return [n for n in names if score.get(n, 0.0) > 0.0]


def _period_mode_utility(
    dt: pd.Timestamp,
    next_dt: pd.Timestamp,
    names: List[str],
    rets: pd.DataFrame,
    cfg: MahoragaAI1Config,
    mode_def: Dict[str, float],
) -> float:
    if not names:
        return -1e9
    lb = rets.loc[:dt, names].tail(cfg.hrp_window).dropna()
    if len(lb) < 20:
        return -1e9
    ww = m6.hrp_weights(lb).reindex(names, fill_value=0.0)
    if ww.sum() <= 0:
        return -1e9
    cap_w = ww.clip(upper=float(mode_def["weight_cap"]))
    if cap_w.sum() > 0:
        ww = cap_w / cap_w.sum()
    period = rets.loc[(rets.index > dt) & (rets.index <= next_dt), names]
    if period.empty:
        return -1e9
    port_r = float(mode_def["exposure"]) * period.dot(ww).fillna(0.0)
    eq = (1.0 + port_r).cumprod()
    util = m6.total_ret(port_r) - float(cfg.ai_label_dd_penalty) * abs(m6.max_dd(eq))
    return float(util)


def _build_training_labels(
    ohlcv: Dict[str, pd.DataFrame],
    cfg: MahoragaAI1Config,
    universe_schedule: Optional[pd.DataFrame],
    comp: Dict[str, pd.DataFrame],
    feature_table: pd.DataFrame,
    train_start: str,
    train_end: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    close = ohlcv["close"][comp["trend"].columns]
    idx = close.index
    rets = close.pct_change().fillna(0.0)
    qqq = m6.to_s(ohlcv["close"][cfg.bench_qqq].reindex(idx).ffill(), "QQQ")
    qqq_r = qqq.pct_change().fillna(0.0)
    vix = None
    if cfg.bench_vix in ohlcv.get("close", pd.DataFrame()).columns:
        vix = m6.to_s(ohlcv["close"][cfg.bench_vix].reindex(idx).ffill(), "VIX")
    else:
        vix = pd.Series(np.nan, index=idx, name="VIX")
    reb_dates = [d for d in feature_table.index if pd.Timestamp(train_start) <= d <= pd.Timestamp(train_end)]
    mode_lib = _mode_library(cfg)
    X_rows, y_mode, y_sev = [], [], []
    for i, dt in enumerate(reb_dates[:-1]):
        next_dt = reb_dates[i + 1]
        if next_dt > pd.Timestamp(train_end):
            break
        members = _active_members(dt, universe_schedule, cfg.universe_static, list(close.columns))
        if len(members) < 2:
            continue
        mode_scores = {}
        for mode_name, mode_def in mode_lib.items():
            names = _select_mode_names(dt, members, comp, mode_def, int(mode_def["top_k"]))
            mode_scores[mode_name] = _period_mode_utility(dt, next_dt, names, rets, cfg, mode_def)
        best_mode = max(mode_scores.items(), key=lambda kv: kv[1])[0]

        qqq_period = qqq_r.loc[(qqq_r.index > dt) & (qqq_r.index <= next_dt)].fillna(0.0)
        qqq_eq = (1.0 + qqq_period).cumprod()
        qqq_tot = m6.total_ret(qqq_period)
        qqq_dd = m6.max_dd(qqq_eq) if len(qqq_eq) else 0.0
        vix_future = float(vix.loc[(vix.index > dt) & (vix.index <= next_dt)].mean()) if vix is not None else np.nan
        hostile = int(
            (qqq_tot <= cfg.ai_severity_ret_thr) or
            (qqq_dd <= cfg.ai_severity_dd_thr) or
            (np.isfinite(vix_future) and vix_future >= cfg.ai_panic_vix_label_thr)
        )

        X_rows.append(feature_table.loc[dt].to_dict())
        y_mode.append(best_mode)
        y_sev.append(hostile)

    if not X_rows:
        return pd.DataFrame(), pd.Series(dtype=object), pd.Series(dtype=int)
    X = pd.DataFrame(X_rows, index=pd.DatetimeIndex(reb_dates[:len(X_rows)]))
    y_mode_s = pd.Series(y_mode, index=X.index, name="mode_label")
    y_sev_s = pd.Series(y_sev, index=X.index, name="severity_label")
    return X, y_mode_s, y_sev_s


class _TabularModel:
    def __init__(self, classifier: HistGradientBoostingClassifier, label_encoder: Optional[LabelEncoder] = None):
        self.classifier = classifier
        self.label_encoder = label_encoder
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", self.classifier),
        ])
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None):
        self.feature_names = list(X.columns)
        y_fit = y
        if self.label_encoder is not None:
            y_fit = self.label_encoder.fit_transform(y.astype(str))
        self.pipeline.fit(X, y_fit, model__sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame):
        pred = self.pipeline.predict(X[self.feature_names])
        if self.label_encoder is not None:
            pred = self.label_encoder.inverse_transform(pred.astype(int))
        return pred

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.pipeline.named_steps["model"], "predict_proba"):
            # Approximate from decision function not available for HGB; fallback to one-hot on predict
            pred = self.predict(X)
            if self.label_encoder is None:
                uniq = np.sort(pd.Series(pred).unique())
                probs = np.zeros((len(pred), len(uniq)))
                mapping = {k: i for i, k in enumerate(uniq)}
                for i, p in enumerate(pred):
                    probs[i, mapping[p]] = 1.0
                return probs
            classes = self.label_encoder.classes_
            probs = np.zeros((len(pred), len(classes)))
            mapping = {k: i for i, k in enumerate(classes)}
            for i, p in enumerate(pred):
                probs[i, mapping[p]] = 1.0
            return probs
        return self.pipeline.predict_proba(X[self.feature_names])

    def feature_importance(self) -> pd.Series:
        mdl = self.pipeline.named_steps["model"]
        if hasattr(mdl, "feature_importances_"):
            vals = mdl.feature_importances_
        else:
            vals = np.full(len(self.feature_names), np.nan)
        return pd.Series(vals, index=self.feature_names, name="importance")


def _fit_ai_models(
    X: pd.DataFrame,
    y_mode: pd.Series,
    y_sev: pd.Series,
    cfg: MahoragaAI1Config,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "mode_model": None,
        "severity_model": None,
        "mode_encoder": None,
        "train_X": X.copy(),
        "train_mode": y_mode.copy(),
        "train_sev": y_sev.copy(),
    }
    if X.empty or len(X) < cfg.ai_min_train_samples:
        return out

    sample_weight = np.linspace(1.0, cfg.ai_recent_sample_bias, len(X))

    mode_counts = y_mode.value_counts()
    keep_idx = y_mode.index[y_mode.map(mode_counts) >= cfg.ai_min_class_count]
    if len(keep_idx) >= max(cfg.ai_min_train_samples // 2, 40) and y_mode.loc[keep_idx].nunique() >= 2:
        enc = LabelEncoder()
        mode_clf = HistGradientBoostingClassifier(
            random_state=cfg.random_seed,
            learning_rate=0.05,
            max_depth=3,
            max_iter=200,
            min_samples_leaf=15,
        )
        mode_model = _TabularModel(mode_clf, enc).fit(X.loc[keep_idx], y_mode.loc[keep_idx], sample_weight=sample_weight[:len(keep_idx)])
        out["mode_model"] = mode_model
        out["mode_encoder"] = enc

    if y_sev.nunique() >= 2:
        sev_clf = HistGradientBoostingClassifier(
            random_state=cfg.random_seed + 7,
            learning_rate=0.05,
            max_depth=3,
            max_iter=200,
            min_samples_leaf=15,
        )
        sev_model = _TabularModel(sev_clf, None).fit(X, y_sev.astype(int), sample_weight=sample_weight)
        out["severity_model"] = sev_model

    return out


def _predict_ai_controls(
    feature_table: pd.DataFrame,
    ai_models: Dict[str, Any],
    cfg: MahoragaAI1Config,
) -> pd.DataFrame:
    out = pd.DataFrame(index=feature_table.index)
    out["mode"] = cfg.ai_base_mode
    out["mode_conf"] = 0.0
    out["severity_prob"] = 0.0

    mode_model = ai_models.get("mode_model")
    if mode_model is not None and getattr(mode_model, "is_fitted", False):
        probs = mode_model.predict_proba(feature_table)
        pred = mode_model.predict(feature_table)
        if mode_model.label_encoder is not None:
            classes = list(mode_model.label_encoder.classes_)
        else:
            classes = sorted(pd.Series(pred).unique().tolist())
        maxp = probs.max(axis=1)
        chosen = []
        for i, p in enumerate(pred):
            if maxp[i] < cfg.ai_min_mode_confidence:
                chosen.append(cfg.ai_base_mode)
            else:
                chosen.append(str(p))
        out["mode"] = chosen
        out["mode_conf"] = maxp
        for j, cls in enumerate(classes):
            out[f"mode_prob_{cls}"] = probs[:, j]

    sev_model = ai_models.get("severity_model")
    if sev_model is not None and getattr(sev_model, "is_fitted", False):
        probs = sev_model.predict_proba(feature_table)
        if probs.shape[1] == 2:
            out["severity_prob"] = probs[:, 1]
        else:
            out["severity_prob"] = probs.max(axis=1)

    sev_scale = 1.0 - cfg.ai_severity_scale_strength * out["severity_prob"].astype(float)
    sev_scale = sev_scale.clip(lower=cfg.ai_severity_floor, upper=1.0)
    out["severity_scale"] = sev_scale
    return out


def backtest_ai(
    ohlcv: Dict[str, pd.DataFrame],
    cfg: MahoragaAI1Config,
    costs: m6.CostsConfig,
    ai_models: Dict[str, Any],
    label: str = "MAHORAGA_AI1_0",
    universe_schedule: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    np.random.seed(cfg.random_seed)

    all_sched_tickers = set()
    if universe_schedule is not None and not universe_schedule.empty:
        for members_json in universe_schedule["members"]:
            all_sched_tickers |= set(json.loads(members_json))
        univ_master = sorted(all_sched_tickers & set(ohlcv["close"].columns))
        use_pit_universe = True
    else:
        univ_master = [t for t in cfg.universe_static if t in ohlcv["close"].columns]
        use_pit_universe = False
    if not univ_master:
        raise ValueError("[backtest_ai] No valid tickers in universe")

    close = ohlcv["close"][univ_master].copy()
    high = ohlcv["high"][univ_master].copy()
    low = ohlcv["low"][univ_master].copy()
    volume = ohlcv["volume"][univ_master].copy()
    idx = close.index
    qqq = m6.to_s(ohlcv["close"][cfg.bench_qqq].reindex(idx).ffill(), "QQQ")
    spy = m6.to_s(ohlcv["close"][cfg.bench_spy].reindex(idx).ffill(), "SPY")
    rets = close.pct_change().fillna(0.0)

    crisis_scale, crisis_state = m6.compute_crisis_gate(qqq, cfg)
    turb_scale = m6.compute_turbulence(close, volume, qqq, cfg)
    vix_series = None
    if cfg.bench_vix in ohlcv.get("close", pd.DataFrame()).columns:
        vix_series = m6.to_s(ohlcv["close"][cfg.bench_vix].reindex(idx).ffill(), "VIX")
    corr_rho, corr_scale, corr_state = m6.compute_corr_shield_series(
        rets, idx, cfg, univ_master, use_pit_universe, universe_schedule=universe_schedule, vix=vix_series
    )

    comp = _component_scores(close, qqq, cfg)
    feature_table = _build_meta_feature_table(ohlcv, cfg, universe_schedule, comp)
    controls = _predict_ai_controls(feature_table, ai_models, cfg)
    mode_lib = _mode_library(cfg)

    reb_dates = set(close.resample(cfg.rebalance_freq).last().index)
    mode_daily = pd.Series(cfg.ai_base_mode, index=idx, name="ai_mode")
    mode_conf_daily = pd.Series(0.0, index=idx, name="ai_mode_conf")
    sev_prob_daily = pd.Series(0.0, index=idx, name="ai_severity_prob")
    ext_scale_daily = pd.Series(1.0, index=idx, name="ai_external_scale")

    w = pd.DataFrame(0.0, index=idx, columns=univ_master)
    last_w = pd.Series(0.0, index=univ_master)
    current_mode = cfg.ai_base_mode
    current_mode_conf = 0.0
    current_sev_prob = 0.0
    current_ext_scale = 1.0

    for dt in idx:
        if dt in reb_dates:
            if dt in controls.index:
                current_mode = str(controls.loc[dt, "mode"])
                current_mode_conf = float(controls.loc[dt, "mode_conf"])
                current_sev_prob = float(controls.loc[dt, "severity_prob"])
                current_ext_scale = float(controls.loc[dt, "severity_scale"])
            mode_def = mode_lib.get(current_mode, mode_lib[cfg.ai_base_mode])

            if current_sev_prob >= cfg.ai_mode_hard_defensive_prob:
                current_mode = "CASH_BIAS"
                mode_def = mode_lib[current_mode]
            elif current_sev_prob >= cfg.ai_severity_high_prob and current_mode not in ("DEFENSIVE", "CASH_BIAS"):
                current_mode = "DEFENSIVE"
                mode_def = mode_lib[current_mode]

            members = _active_members(dt, universe_schedule, cfg.universe_static, univ_master) if use_pit_universe else univ_master
            members = [t for t in members if t in univ_master]
            names = _select_mode_names(dt, members, comp, mode_def, int(mode_def["top_k"]))
            if not names:
                last_w = pd.Series(0.0, index=univ_master)
            elif len(names) == 1:
                last_w = pd.Series(0.0, index=univ_master)
                last_w[names[0]] = 1.0
            else:
                lb = rets.loc[:dt].tail(cfg.hrp_window)[names].dropna()
                if len(lb) < 60:
                    _ret_fallback = lb if len(lb) > len(names) else rets.loc[:dt][names].dropna()
                    ww = m6.hrp_weights(_ret_fallback).reindex(names, fill_value=0.0)
                else:
                    ww = m6.hrp_weights(lb).reindex(names, fill_value=0.0)
                if ww.sum() > 0:
                    ww = ww.clip(upper=float(mode_def["weight_cap"]))
                    ww = ww / ww.sum()
                last_w = pd.Series(0.0, index=univ_master)
                last_w[names] = ww.values
            current_ext_scale = float(np.clip(current_ext_scale * float(mode_def["exposure"]), cfg.ai_severity_floor, 1.0))

        w.loc[dt] = last_w.values
        mode_daily.loc[dt] = current_mode
        mode_conf_daily.loc[dt] = current_mode_conf
        sev_prob_daily.loc[dt] = current_sev_prob
        ext_scale_daily.loc[dt] = current_ext_scale

    w_stop, stop_hits = m6.apply_chandelier(w, close, high, low, cfg)
    w_exec_1x = w_stop.shift(1).fillna(0.0)
    gross_1x = (w_exec_1x * rets).sum(axis=1)
    vol_sc = m6.vol_target_scale(gross_1x, cfg)
    cap = (crisis_scale * turb_scale * corr_scale * ext_scale_daily).clip(0.0, cfg.max_exposure)
    tgt_sc = pd.Series(np.minimum(vol_sc.values, cap.values), index=idx).clip(0.0, cfg.max_exposure)
    exec_sc = tgt_sc.shift(1).fillna(0.0)
    w_exec = w_exec_1x.mul(exec_sc, axis=0)
    to, tc = m6._costs(w_exec, costs)
    port_net = ((w_exec * rets).sum(axis=1) - tc).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    equity = cfg.capital_initial * (1.0 + port_net).cumprod()
    exposure = w_exec.abs().sum(axis=1).clip(0.0, cfg.max_exposure)
    qqq_r = qqq.pct_change().fillna(0.0) - costs.qqq_expense_ratio
    spy_r = spy.pct_change().fillna(0.0)
    qqq_eq = cfg.capital_initial * (1.0 + qqq_r).cumprod()
    spy_eq = cfg.capital_initial * (1.0 + spy_r).cumprod()

    return {
        "label": label,
        "returns_net": port_net,
        "equity": equity,
        "exposure": exposure,
        "turnover": to,
        "weights_scaled": w_exec,
        "total_scale": exec_sc,
        "total_scale_target": tgt_sc,
        "cap": cap,
        "turb_scale": turb_scale,
        "crisis_scale": crisis_scale,
        "crisis_state": crisis_state,
        "vol_scale": vol_sc,
        "external_scale": ext_scale_daily,
        "corr_scale": corr_scale,
        "corr_rho": corr_rho,
        "corr_state": corr_state,
        "stop_hits": stop_hits,
        "scores": None,
        "bench": {"QQQ_r": qqq_r, "QQQ_eq": qqq_eq, "SPY_r": spy_r, "SPY_eq": spy_eq},
        "ai_controls": pd.DataFrame({
            "mode": mode_daily,
            "mode_conf": mode_conf_daily,
            "severity_prob": sev_prob_daily,
            "external_scale": ext_scale_daily,
            "corr_rho": corr_rho.reindex(idx),
            "corr_scale": corr_scale.reindex(idx),
        }),
        "meta_features": feature_table.copy(),
        "mode_probabilities": controls.copy(),
    }


def run_walk_forward_ai(
    ohlcv: Dict[str, pd.DataFrame],
    cfg_base: MahoragaAI1Config,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    trading_idx = pd.DatetimeIndex(ohlcv["close"].index)
    folds = m6.build_contiguous_folds(cfg_base, trading_idx)
    ai_oos_r, ai_oos_exp, ai_oos_to = [], [], []
    base_oos_r, base_oos_exp, base_oos_to = [], [], []
    fold_results, all_sweeps = [], []
    fold_artifacts: Dict[int, Dict[str, Any]] = {}
    last_best = None

    for fold in folds:
        fold_n = fold["fold"]
        train_start, train_end = fold["train_start"], fold["train_end"]
        val_start, val_end = fold["val_start"], fold["val_end"]
        test_start, test_end = fold["test_start"], fold["test_end"]
        print(f"\n  ── AI FOLD {fold_n}/{len(folds)} ──")

        cfg_f = deepcopy(cfg_base)
        qqq_full = m6.to_s(ohlcv["close"][cfg_base.bench_qqq].ffill())
        dd_thr, vol_thr = m6.calibrate_crisis_thresholds(qqq_full, train_start, train_end, cfg_f)
        cfg_f.crisis_dd_thr = dd_thr
        cfg_f.crisis_vol_zscore_thr = vol_thr

        print(f"  [fold {fold_n}] Fitting IC weights on train …")
        train_tickers = m6.get_training_universe(train_end, universe_schedule,
                                                 cfg_base.universe_static, list(ohlcv["close"].columns))
        close_univ = ohlcv["close"][train_tickers]
        ic_weights = m6.fit_ic_weights(close_univ, qqq_full.loc[train_start:train_end], cfg_f, train_start, train_end)

        print(f"  [fold {fold_n}] Sweeping base combo on val {val_start}→{val_end} …")
        sweep_df, best = m6.run_fold_sweep(
            ohlcv, cfg_f, costs, ic_weights, val_start, val_end, fold_n,
            universe_schedule=universe_schedule
        )
        sweep_df["fold"] = fold_n
        all_sweeps.append(sweep_df)
        last_best = best
        best_cfg = deepcopy(best["cfg"])

        print(f"  [fold {fold_n}] Training AI meta-models on {train_start}→{train_end} …")
        # conservative: train only on train window
        all_sched_tickers = set()
        if universe_schedule is not None and not universe_schedule.empty:
            for members_json in universe_schedule["members"]:
                all_sched_tickers |= set(json.loads(members_json))
            ai_cols = sorted(all_sched_tickers & set(ohlcv["close"].columns))
        else:
            ai_cols = [t for t in best_cfg.universe_static if t in ohlcv["close"].columns]
        close_master = ohlcv["close"][ai_cols]
        comp = _component_scores(close_master, qqq_full, best_cfg)
        feat = _build_meta_feature_table(ohlcv, best_cfg, universe_schedule, comp)
        X_train, y_mode, y_sev = _build_training_labels(ohlcv, best_cfg, universe_schedule, comp, feat, train_start, train_end)
        ai_models = _fit_ai_models(X_train, y_mode, y_sev, best_cfg)

        print(f"  [fold {fold_n}] Testing baseline and AI on {test_start}→{test_end} …")
        res_base = m6.backtest(ohlcv, best_cfg, costs, label=f"FOLD{fold_n}_BASE", universe_schedule=universe_schedule)
        res_ai = backtest_ai(ohlcv, best_cfg, costs, ai_models, label=f"FOLD{fold_n}_AI", universe_schedule=universe_schedule)

        r_base = res_base["returns_net"].loc[test_start:test_end]
        eq_base = best_cfg.capital_initial * (1.0 + r_base).cumprod()
        exp_base = res_base["exposure"].loc[r_base.index]
        to_base = res_base["turnover"].loc[r_base.index]
        s_base = m6.summarize(r_base, eq_base, exp_base, to_base, best_cfg, f"FOLD{fold_n}_BASE")

        r_ai = res_ai["returns_net"].loc[test_start:test_end]
        eq_ai = best_cfg.capital_initial * (1.0 + r_ai).cumprod()
        exp_ai = res_ai["exposure"].loc[r_ai.index]
        to_ai = res_ai["turnover"].loc[r_ai.index]
        s_ai = m6.summarize(r_ai, eq_ai, exp_ai, to_ai, best_cfg, f"FOLD{fold_n}_AI")

        mode_slice = res_ai["ai_controls"].loc[r_ai.index, "mode"] if not r_ai.empty else pd.Series(dtype=object)
        sev_slice = res_ai["ai_controls"].loc[r_ai.index, "severity_prob"] if not r_ai.empty else pd.Series(dtype=float)

        fold_results.append({
            "fold": fold_n,
            "train": f"{train_start}→{train_end}",
            "val": f"{val_start}→{val_end}",
            "test": f"{test_start}→{test_end}",
            **{f"best_{k}": v for k, v in best["combo_params"].items()},
            "val_score": round(best["score"], 4),
            "val_sharpe": round(best["s_val"]["Sharpe"], 4),
            "val_q_value": round(best["q_value"], 6),
            "base_Sharpe": round(s_base["Sharpe"], 4),
            "base_CAGR%": round(s_base["CAGR"] * 100, 2),
            "base_MaxDD%": round(s_base["MaxDD"] * 100, 2),
            "ai_Sharpe": round(s_ai["Sharpe"], 4),
            "ai_CAGR%": round(s_ai["CAGR"] * 100, 2),
            "ai_MaxDD%": round(s_ai["MaxDD"] * 100, 2),
            "delta_Sharpe": round(s_ai["Sharpe"] - s_base["Sharpe"], 4),
            "delta_CAGR%": round((s_ai["CAGR"] - s_base["CAGR"]) * 100, 2),
            "delta_MaxDD%": round((s_ai["MaxDD"] - s_base["MaxDD"]) * 100, 2),
            "avg_severity_prob": round(float(sev_slice.mean()) if len(sev_slice) else 0.0, 4),
            "mode_BALANCED_pct": round(float((mode_slice == "BALANCED").mean() * 100) if len(mode_slice) else 0.0, 2),
            "mode_REL_HEAVY_pct": round(float((mode_slice == "REL_HEAVY").mean() * 100) if len(mode_slice) else 0.0, 2),
            "mode_MOM_HEAVY_pct": round(float((mode_slice == "MOM_HEAVY").mean() * 100) if len(mode_slice) else 0.0, 2),
            "mode_DEFENSIVE_pct": round(float((mode_slice == "DEFENSIVE").mean() * 100) if len(mode_slice) else 0.0, 2),
            "mode_CASH_BIAS_pct": round(float((mode_slice == "CASH_BIAS").mean() * 100) if len(mode_slice) else 0.0, 2),
        })

        ai_oos_r.append(r_ai)
        ai_oos_exp.append(exp_ai)
        ai_oos_to.append(to_ai)
        base_oos_r.append(r_base)
        base_oos_exp.append(exp_base)
        base_oos_to.append(to_base)

        fold_artifacts[fold_n] = {
            "best_cfg": deepcopy(best_cfg),
            "ai_models": ai_models,
            "res_ai": res_ai,
            "res_base": res_base,
            "features": feat,
            "train_X": X_train,
            "train_y_mode": y_mode,
            "train_y_sev": y_sev,
        }

        print(f"  [fold {fold_n}] BASE Sharpe={s_base['Sharpe']:.3f} | AI Sharpe={s_ai['Sharpe']:.3f} | Δ={s_ai['Sharpe'] - s_base['Sharpe']:+.3f}")

    def _stitch(parts: List[pd.Series], fill_zero: bool = False) -> pd.Series:
        s = pd.concat(parts).sort_index()
        s = s[~s.index.duplicated(keep="first")]
        return s.fillna(0.0) if fill_zero else s

    ai_oos_r_s = _stitch(ai_oos_r)
    ai_oos_exp_s = _stitch(ai_oos_exp)
    ai_oos_to_s = _stitch(ai_oos_to, fill_zero=True)
    base_oos_r_s = _stitch(base_oos_r)
    base_oos_exp_s = _stitch(base_oos_exp)
    base_oos_to_s = _stitch(base_oos_to, fill_zero=True)
    _, oos_label = m6.validate_oos_continuity(folds, trading_idx)

    ai_oos_eq = cfg_base.capital_initial * (1.0 + ai_oos_r_s).cumprod()
    base_oos_eq = cfg_base.capital_initial * (1.0 + base_oos_r_s).cumprod()

    all_sweeps_df = pd.concat(all_sweeps, ignore_index=True) if all_sweeps else pd.DataFrame()
    selection_audit_df, info_by_source = m6._selection_audit_from_sweeps(all_sweeps_df, last_best)
    chosen_source = str(getattr(cfg_base, "final_selection_method", "last_fold_winner"))
    if chosen_source not in info_by_source:
        chosen_source = "last_fold_winner" if "last_fold_winner" in info_by_source else next(iter(info_by_source.keys()))
    selected_config_info = deepcopy(info_by_source[chosen_source])
    selected_config_info["selection_audit_df"] = selection_audit_df.copy()
    selected_config_info["fold_artifacts"] = fold_artifacts
    selected_config_info["selection_alternatives"] = {k: v for k, v in info_by_source.items() if k != chosen_source}

    return {
        "ai_oos_r": ai_oos_r_s,
        "ai_oos_eq": ai_oos_eq,
        "ai_oos_exp": ai_oos_exp_s,
        "ai_oos_to": ai_oos_to_s,
        "base_oos_r": base_oos_r_s,
        "base_oos_eq": base_oos_eq,
        "base_oos_exp": base_oos_exp_s,
        "base_oos_to": base_oos_to_s,
        "oos_label": oos_label,
        "fold_results": fold_results,
        "all_sweeps_df": all_sweeps_df,
        "selected_config_info": selected_config_info,
        "fold_artifacts": fold_artifacts,
    }


def _compare_summary_rows(cfg: MahoragaAI1Config, label: str, r: pd.Series, eq: pd.Series,
                          exp: pd.Series, to: pd.Series) -> Dict[str, Any]:
    return m6.summarize(r, eq, exp, to, cfg, label)


def _build_final_ai_report(
    cfg: MahoragaAI1Config,
    selected_config_info: Dict[str, Any],
    comp_oos_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    full_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    mode_dist_df: pd.DataFrame,
) -> str:
    chosen = selected_config_info
    lines = []
    lines.append("MAHORAGA AI1.0 — FINAL REPORT")
    lines.append("=" * 92)
    lines.append("")
    lines.append("AI layer: mode selection + regime severity (no direct price prediction).")
    lines.append(f"Model label: {cfg.label}")
    lines.append(f"Selection method (base combo): {chosen.get('source')}")
    lines.append(f"Selected combo params: {chosen.get('combo_params')}")
    lines.append(
        f"Statistical support of base combo: p={chosen.get('val_p_value', np.nan):.6f}  "
        f"q={chosen.get('val_q_value', np.nan):.6f}  label={chosen.get('val_stat_label')}"
    )
    lines.append("")
    lines.append("OOS COMPARISON")
    lines.append("-" * 92)
    lines.append(comp_oos_df.to_string(index=False))
    lines.append("")
    lines.append("FULL PERIOD COMPARISON")
    lines.append("-" * 92)
    lines.append(full_df.to_string(index=False))
    lines.append("")
    lines.append("FOLD SUMMARY")
    lines.append("-" * 92)
    lines.append(fold_df.to_string(index=False))
    if not regime_df.empty:
        lines.append("")
        lines.append("REGIME OOS")
        lines.append("-" * 92)
        lines.append(regime_df.to_string(index=False))
    if not mode_dist_df.empty:
        lines.append("")
        lines.append("MODE DISTRIBUTION (OOS)")
        lines.append("-" * 92)
        lines.append(mode_dist_df.to_string(index=False))
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- AI1.0 aims to improve stability and hostile-regime handling more than raw CAGR.")
    lines.append("- Improvements must be interpreted against the same universe and multiple-testing caveats as Mahoraga 6.1.")
    return "\n".join(lines)


def save_outputs_ai(
    cfg: MahoragaAI1Config,
    results: Dict[str, Any],
    full_base: Dict[str, Any],
    full_ai: Dict[str, Any],
    universe_schedule: Optional[pd.DataFrame],
):
    d = cfg.outputs_dir
    _ensure_dir(d)

    fold_df = pd.DataFrame(results["fold_results"])
    fold_df.to_csv(f"{d}/walk_forward_folds_ai.csv", index=False)
    results["all_sweeps_df"].to_csv(f"{d}/walk_forward_sweeps.csv", index=False)

    comp_oos = pd.DataFrame([
        _compare_summary_rows(cfg, "MAHORAGA_AI1_0_OOS", results["ai_oos_r"], results["ai_oos_eq"], results["ai_oos_exp"], results["ai_oos_to"]),
        _compare_summary_rows(cfg, "MAHORAGA_6_1_BASE_OOS", results["base_oos_r"], results["base_oos_eq"], results["base_oos_exp"], results["base_oos_to"]),
        _compare_summary_rows(cfg, "QQQ_OOS", full_base["bench"]["QQQ_r"].loc[results["ai_oos_r"].index], full_base["bench"]["QQQ_eq"].loc[results["ai_oos_r"].index], None, None),
    ])
    comp_oos.to_csv(f"{d}/comparison_oos.csv", index=False)

    comp_full = pd.DataFrame([
        _compare_summary_rows(cfg, "MAHORAGA_AI1_0_FULL", full_ai["returns_net"], full_ai["equity"], full_ai["exposure"], full_ai["turnover"]),
        _compare_summary_rows(cfg, "MAHORAGA_6_1_BASE_FULL", full_base["returns_net"], full_base["equity"], full_base["exposure"], full_base["turnover"]),
        _compare_summary_rows(cfg, "QQQ_FULL", full_base["bench"]["QQQ_r"], full_base["bench"]["QQQ_eq"], None, None),
    ])
    comp_full.to_csv(f"{d}/comparison_full.csv", index=False)

    # OOS regime comparison using AI controls and VIX buckets
    oos_idx = results["ai_oos_r"].index
    ai_ctrl_oos = full_ai["ai_controls"].reindex(oos_idx).ffill()
    if cfg.bench_vix in full_ai.get("ai_controls", pd.DataFrame()).columns:
        pass
    vix_series = None
    if cfg.bench_vix in full_base.get("ai_controls", pd.DataFrame()).columns:
        vix_series = None
    # use raw VIX from full-ai context table if available via mode_probabilities merge fallback
    regime_rows = []
    close = None
    for name, mask in {
        "ALL_OOS": pd.Series(True, index=oos_idx),
        "HIGH_SEVERITY": ai_ctrl_oos["severity_prob"] >= cfg.ai_severity_high_prob,
        "DEFENSIVE_OR_CASH": ai_ctrl_oos["mode"].isin(["DEFENSIVE", "CASH_BIAS"]),
    }.items():
        rr = results["ai_oos_r"].loc[mask]
        if len(rr) == 0:
            continue
        eq = cfg.capital_initial * (1.0 + rr).cumprod()
        regime_rows.append(_compare_summary_rows(cfg, name, rr, eq, results["ai_oos_exp"].loc[rr.index], results["ai_oos_to"].loc[rr.index]))
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(f"{d}/regime_oos_ai.csv", index=False)

    controls = full_ai["ai_controls"].copy()
    controls.to_csv(f"{d}/dynamic_mode_controls.csv", index=True)
    full_ai["meta_features"].to_csv(f"{d}/meta_features_snapshot.csv", index=True)
    full_ai["mode_probabilities"].to_csv(f"{d}/meta_predictions.csv", index=True)
    controls[["severity_prob"]].rename(columns={"severity_prob": "regime_severity_score"}).to_csv(f"{d}/regime_severity_score.csv", index=True)

    # Selection audit if available from base combo selection
    sel_df = results["selected_config_info"].get("selection_audit_df", pd.DataFrame())
    if not sel_df.empty:
        sel_df.to_csv(f"{d}/selection_audit.csv", index=False)

    # Feature importances
    fold_artifacts = results.get("fold_artifacts", {})
    mode_imp_rows = []
    sev_imp_rows = []
    for fold_n, art in fold_artifacts.items():
        mm = art["ai_models"].get("mode_model")
        sm = art["ai_models"].get("severity_model")
        if mm is not None and getattr(mm, "is_fitted", False):
            s = mm.feature_importance().rename("importance").reset_index().rename(columns={"index": "feature"})
            s["fold"] = fold_n
            mode_imp_rows.append(s)
        if sm is not None and getattr(sm, "is_fitted", False):
            s = sm.feature_importance().rename("importance").reset_index().rename(columns={"index": "feature"})
            s["fold"] = fold_n
            sev_imp_rows.append(s)
    if mode_imp_rows:
        pd.concat(mode_imp_rows, ignore_index=True).to_csv(f"{d}/feature_importance_mode.csv", index=False)
    if sev_imp_rows:
        pd.concat(sev_imp_rows, ignore_index=True).to_csv(f"{d}/feature_importance_severity.csv", index=False)

    mode_dist = controls["mode"].value_counts(normalize=True).rename("pct").mul(100).round(2).reset_index().rename(columns={"index": "mode"})
    mode_dist.to_csv(f"{d}/mode_distribution.csv", index=False)

    report_text = _build_final_ai_report(cfg, results["selected_config_info"], comp_oos, fold_df, comp_full, regime_df, mode_dist)
    with open(f"{d}/final_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    if universe_schedule is not None:
        universe_schedule.to_csv(f"{d}/universe_schedule.csv", index=False)

    print(f"\n  [outputs → ./{d}/]")
    print("    comparison_full.csv, comparison_oos.csv, walk_forward_folds_ai.csv, walk_forward_sweeps.csv")
    print("    dynamic_mode_controls.csv, meta_features_snapshot.csv, meta_predictions.csv, regime_severity_score.csv")
    print("    feature_importance_mode.csv, feature_importance_severity.csv, selection_audit.csv, final_report.txt")


def make_plots_ai(cfg: MahoragaAI1Config, results: Dict[str, Any], full_ai: Dict[str, Any], full_base: Dict[str, Any]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(cfg.plots_dir)
    oos_idx = results["ai_oos_eq"].index
    plt.figure(figsize=(12, 6))
    plt.plot(results["ai_oos_eq"].index, results["ai_oos_eq"], label="AI1.0 OOS")
    plt.plot(results["base_oos_eq"].index, results["base_oos_eq"], label="Base 6.1 OOS")
    plt.plot(full_base["bench"]["QQQ_eq"].loc[oos_idx].index, full_base["bench"]["QQQ_eq"].loc[oos_idx], label="QQQ OOS", linestyle="--")
    plt.yscale("log")
    plt.title("MahoragaAI1.0 vs Mahoraga6.1 vs QQQ (OOS)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.plots_dir}/01_equity_oos_ai.png", dpi=150)
    plt.close()

    ctrl = full_ai["ai_controls"]
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(ctrl.index, ctrl["severity_prob"], label="Severity prob")
    ax1.plot(ctrl.index, ctrl["external_scale"], label="AI external scale")
    ax1.set_title("AI Severity & External Scale")
    ax1.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.plots_dir}/02_severity_scale.png", dpi=150)
    plt.close()


def run_mahoraga_ai1_0(make_plots_flag: bool = True) -> Dict[str, Any]:
    print("=" * 80)
    print("  MAHORAGA AI1.0 — Mode AI + Regime Severity")
    print("=" * 80)
    print(m6.UNIVERSE_BIAS_DISCLAIMER)
    print(AI1_DISCLAIMER)

    cfg = MahoragaAI1Config()
    costs = m6.CostsConfig()
    ucfg = m6.UniverseConfig()
    _ensure_dir(cfg.outputs_dir)
    _ensure_dir(cfg.plots_dir)
    _ensure_dir(cfg.cache_dir)

    print("\n[1] Downloading data …")
    equity_tickers = sorted(set(list(cfg.universe_static) + [t for u in m6.ALTERNATE_UNIVERSES.values() for t in u]))
    bench_tickers = [cfg.bench_qqq, cfg.bench_spy, cfg.bench_vix]
    all_tickers = sorted(set(equity_tickers + bench_tickers))
    ohlcv = m6.download_ohlcv(all_tickers, cfg.data_start, cfg.data_end, cfg.cache_dir)

    print("\n[2] Canonical universe …")
    asset_registry = m6.build_asset_registry(equity_tickers, cfg, bench_tickers)
    data_quality_report = m6.compute_data_quality_report(ohlcv, equity_tickers, cfg)
    clean_equity = m6.filter_equity_candidates(
        [t for t in equity_tickers if t in ohlcv["close"].columns],
        asset_registry, data_quality_report, cfg,
    )
    universe_schedule = None
    if cfg.use_canonical_universe:
        universe_schedule, _ = m6.build_canonical_universe_schedule(
            ohlcv["close"], ohlcv["volume"], ucfg, clean_equity,
            cfg.data_start, cfg.data_end,
            registry_df=asset_registry, quality_df=data_quality_report,
        )

    print("\n[3] Walk-forward AI over Mahoraga 6.1 base …")
    wf = run_walk_forward_ai(ohlcv, cfg, costs, universe_schedule=universe_schedule)

    print("\n[4] Final selected config + full-period evaluation …")
    cfg_final = deepcopy(cfg)
    selected = wf["selected_config_info"]
    qqq_full = m6.to_s(ohlcv["close"][cfg.bench_qqq].ffill())
    last_train_end = cfg.wf_folds[-1][0]
    dd_thr, vol_thr = m6.calibrate_crisis_thresholds(qqq_full, cfg.wf_train_start, last_train_end, cfg_final)
    cfg_final.crisis_dd_thr = dd_thr
    cfg_final.crisis_vol_zscore_thr = vol_thr
    for k, v in selected["combo_params"].items():
        setattr(cfg_final, k, v)

    final_train_tickers = m6.get_training_universe(last_train_end, universe_schedule,
                                                   cfg.universe_static, list(ohlcv["close"].columns))
    close_univ = ohlcv["close"][final_train_tickers]
    wt, wm, wr = m6.fit_ic_weights(close_univ, qqq_full.loc[cfg.wf_train_start:last_train_end], cfg_final,
                                   cfg.wf_train_start, last_train_end)
    cfg_final.w_trend = wt
    cfg_final.w_mom = wm
    cfg_final.w_rel = wr

    # Train final AI models conservatively on train window only
    comp_final = _component_scores(ohlcv["close"][final_train_tickers], qqq_full, cfg_final)
    feat_final = _build_meta_feature_table(ohlcv, cfg_final, universe_schedule, comp_final)
    X_train, y_mode, y_sev = _build_training_labels(ohlcv, cfg_final, universe_schedule, comp_final, feat_final,
                                                    cfg.wf_train_start, last_train_end)
    ai_models_final = _fit_ai_models(X_train, y_mode, y_sev, cfg_final)

    full_base = m6.backtest(ohlcv, cfg_final, costs, label="MAHORAGA_6_1_BASE_FINAL", universe_schedule=universe_schedule)
    full_ai = backtest_ai(ohlcv, cfg_final, costs, ai_models_final, label=cfg_final.label, universe_schedule=universe_schedule)

    print("\n[5] Saving outputs …")
    save_outputs_ai(cfg_final, wf, full_base, full_ai, universe_schedule)

    if make_plots_flag:
        print("\n[6] Generating plots …")
        make_plots_ai(cfg_final, wf, full_ai, full_base)

    return {
        "cfg": cfg_final,
        "wf": wf,
        "full_base": full_base,
        "full_ai": full_ai,
        "ai_models_final": ai_models_final,
    }


if __name__ == "__main__":
    results = run_mahoraga_ai1_0(make_plots_flag=True)
