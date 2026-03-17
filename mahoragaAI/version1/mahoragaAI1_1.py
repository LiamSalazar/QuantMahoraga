from __future__ import annotations

"""
MahoragaAI1.1
==============
Fragility-detection + bounded-intervention AI layer over Mahoraga 6.1.

Design summary
--------------
- Keep Mahoraga 6.1 as the execution/risk baseline.
- Do NOT predict prices directly.
- Learn when the baseline is likely to be fragile in the next rebalance window.
- When fragility is high, choose among a *small* set of bounded interventions:
    BASELINE / REL_TILT / DEFENSIVE_LIGHT
- Eliminate CASH_BIAS and broad de-risking heuristics from AI1.0.
- Determine AI parameters through inner walk-forward calibration on the train slice.

This file must live in the same directory as `mahoraga6_1.py`.
"""

import os
import json
from dataclasses import dataclass
from copy import deepcopy
from itertools import product as iproduct
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

import mahoraga6_1 as m6

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

AI11_DISCLAIMER = r"""
═══════════════════════════════════════════════════════════════════════════════
  MAHORAGA AI1.1 — METHODOLOGY DISCLAIMER
───────────────────────────────────────────────────────────────────────────────
  AI1.1 does not predict asset prices. It predicts:
    (1) the probability that Mahoraga 6.1 is fragile in the next rebalance
        interval, and
    (2) which bounded intervention is preferable when fragility is high.

  AI1.1 is calibrated via inner walk-forward on the train slice. The AI layer
  remains subordinate to Mahoraga 6.1's execution/risk engine.
═══════════════════════════════════════════════════════════════════════════════
"""


@dataclass
class MahoragaAI11Config(m6.Mahoraga6Config):
    plots_dir: str = "mahoragaAI1_1_plots"
    outputs_dir: str = "mahoragaAI1_1_outputs"
    label: str = "MAHORAGA_AI1_1"

    parallel_sweep: bool = True

    # AI1.1: sparse intervention over the 6.1 baseline
    ai_base_action: str = "BASELINE"
    ai_min_train_samples: int = 80
    ai_min_class_count: int = 10
    ai_recent_sample_bias: float = 1.25
    ai_inner_val_frac: float = 0.30
    ai_target_intervention_rate: float = 0.20

    # Candidate grids calibrated on inner WFO
    ai_fragility_q_grid: Tuple[float, ...] = (0.15, 0.20, 0.25)
    ai_intervention_prob_grid: Tuple[float, ...] = (0.65, 0.75)
    ai_action_conf_grid: Tuple[float, ...] = (0.50, 0.60)
    ai_rel_tilt_grid: Tuple[float, ...] = (0.55, 0.65)
    ai_defensive_exposure_grid: Tuple[float, ...] = (0.80, 0.90)
    ai_defensive_cap_grid: Tuple[float, ...] = (0.45, 0.55)
    ai_action_gain_grid: Tuple[float, ...] = (0.0, 0.0025)

    # Selected/calibrated values (populated per fold)
    ai_fragility_q: float = 0.20
    ai_intervention_prob: float = 0.75
    ai_action_conf: float = 0.55
    ai_rel_tilt: float = 0.60
    ai_defensive_exposure: float = 0.85
    ai_defensive_cap: float = 0.50
    ai_action_gain: float = 0.0


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _normalize_triplet(a: float, b: float, c: float) -> Tuple[float, float, float]:
    arr = np.array([a, b, c], dtype=float)
    s = arr.sum()
    if not np.isfinite(s) or s <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    arr = arr / s
    return tuple(float(x) for x in arr)


def _mode_library(cfg: MahoragaAI11Config, params: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Small bounded action set: default baseline + two sparse interventions."""
    bw = np.array(_normalize_triplet(cfg.w_trend, cfg.w_mom, cfg.w_rel), dtype=float)
    rel_target = float(params["rel_tilt"])
    rem = max(0.0, 1.0 - rel_target)
    trend_mom = bw[:2]
    tm_sum = trend_mom.sum()
    if tm_sum <= 0:
        tm = np.array([0.5, 0.5])
    else:
        tm = trend_mom / tm_sum
    rel_w = np.array([rem * tm[0], rem * tm[1], rel_target], dtype=float)
    rel_w = rel_w / rel_w.sum()

    return {
        "BASELINE": {
            "w_trend": float(bw[0]),
            "w_mom": float(bw[1]),
            "w_rel": float(bw[2]),
            "exposure": 1.00,
            "top_k": float(cfg.top_k),
            "weight_cap": float(cfg.weight_cap),
        },
        "REL_TILT": {
            "w_trend": float(rel_w[0]),
            "w_mom": float(rel_w[1]),
            "w_rel": float(rel_w[2]),
            "exposure": 1.00,
            "top_k": float(cfg.top_k),
            "weight_cap": float(cfg.weight_cap),
        },
        "DEFENSIVE_LIGHT": {
            "w_trend": float(rel_w[0]),
            "w_mom": float(rel_w[1]),
            "w_rel": float(rel_w[2]),
            "exposure": float(params["defensive_exposure"]),
            "top_k": float(max(2, cfg.top_k - 1)),
            "weight_cap": float(min(cfg.weight_cap, params["defensive_cap"])),
        },
    }


def _component_scores(close: pd.DataFrame, qqq: pd.Series, cfg: MahoragaAI11Config) -> Dict[str, pd.DataFrame]:
    idx = close.index
    qqq_ = m6.to_s(qqq, "QQQ").reindex(idx).ffill()

    out = {
        "trend": pd.DataFrame(index=idx, columns=close.columns, dtype=float),
        "mom":   pd.DataFrame(index=idx, columns=close.columns, dtype=float),
        "rel":   pd.DataFrame(index=idx, columns=close.columns, dtype=float),
    }

    for t in close.columns:
        cl = close[t].reindex(idx).ffill()

        # Usar exactamente la misma definición de señal que Mahoraga 6.1
        tr = m6._trend(cl, cfg)
        mo = m6._mom(cl, cfg)
        re = m6._rel(cl, qqq_, cfg)

        tr = tr.fillna(0.0)
        mo = mo.fillna(0.0)
        re = re.fillna(0.0)

        tr.iloc[:cfg.burn_in] = 0.0
        mo.iloc[:cfg.burn_in] = 0.0
        re.iloc[:cfg.burn_in] = 0.0

        out["trend"][t] = tr
        out["mom"][t] = mo
        out["rel"][t] = re

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
    cfg: MahoragaAI11Config,
    universe_schedule: Optional[pd.DataFrame],
    comp: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    all_sched_tickers = set()
    if universe_schedule is not None and not universe_schedule.empty:
        for members_json in universe_schedule["members"]:
            all_sched_tickers |= set(json.loads(members_json))
        universe = sorted(all_sched_tickers & set(ohlcv["close"].columns))
    else:
        universe = [t for t in cfg.universe_static if t in ohlcv["close"].columns]
    close = ohlcv["close"][universe].copy()
    volume = ohlcv["volume"][universe].copy()
    idx = close.index
    qqq = m6.to_s(ohlcv["close"][cfg.bench_qqq].reindex(idx).ffill(), "QQQ")
    qqq_r = qqq.pct_change().fillna(0.0)
    rets = close.pct_change().fillna(0.0)
    if comp is None:
        comp = _component_scores(close, qqq, cfg)

    ic_df = m6.rolling_ic_multi_horizon(close, qqq, cfg, window=63)
    # Flatten multi-horizon IC snapshots by nearest date lookup.
    ic_cols = []
    if isinstance(ic_df, pd.DataFrame) and len(ic_df) > 0:
        ic_cols = list(ic_df.columns)
        ic_df = ic_df.reindex(idx).ffill()
    else:
        ic_df = pd.DataFrame(index=idx)

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
        if len(members) < 2:
            continue
        sub21 = rets[members].loc[:dt].tail(21).dropna(how="all")
        sub63 = rets[members].loc[:dt].tail(63).dropna(how="all")
        xs5 = close[members].pct_change(5).loc[dt]
        xs21 = close[members].pct_change(21).loc[dt]
        breadth63 = (close[members].loc[dt] > close[members].shift(63).loc[dt]).mean()
        row = {
            "dt": dt,
            "n_members": len(members),
            "avg_corr_21": _avg_offdiag_corr(sub21),
            "avg_corr_63": _avg_offdiag_corr(sub63),
            "xs_disp_5": float(xs5.std(skipna=True)),
            "xs_disp_21": float(xs21.std(skipna=True)),
            "breadth63": float(breadth63),
            "qqq_ret_5": float(qqq.pct_change(5).loc[dt]),
            "qqq_ret_21": float(qqq.pct_change(21).loc[dt]),
            "qqq_dd_63": float((qqq.loc[:dt] / qqq.loc[:dt].cummax() - 1.0).tail(63).min()),
            "qqq_rv_21": float(qqq_r.loc[:dt].tail(21).std() * np.sqrt(252)),
            "qqq_rv_63": float(qqq_r.loc[:dt].tail(63).std() * np.sqrt(252)),
            "vix": float(vix.loc[dt]) if pd.notna(vix.loc[dt]) else np.nan,
            "vix_chg_5": float(vix.pct_change(5).loc[dt]) if pd.notna(vix.loc[dt]) else np.nan,
            "crisis_scale": float(crisis_scale.loc[dt]),
            "crisis_state": float(crisis_state.loc[dt]),
            "turb_scale": float(turb_scale.loc[dt]),
            "corr_rho": float(corr_rho.loc[dt]) if pd.notna(corr_rho.loc[dt]) else np.nan,
            "corr_scale": float(corr_scale.loc[dt]) if pd.notna(corr_scale.loc[dt]) else np.nan,
            "corr_state": float(corr_state.loc[dt]) if pd.notna(corr_state.loc[dt]) else np.nan,
            "trend_mean": float(comp["trend"].loc[dt, members].mean()),
            "mom_mean": float(comp["mom"].loc[dt, members].mean()),
            "rel_mean": float(comp["rel"].loc[dt, members].mean()),
            "trend_std": float(comp["trend"].loc[dt, members].std(skipna=True)),
            "mom_std": float(comp["mom"].loc[dt, members].std(skipna=True)),
            "rel_std": float(comp["rel"].loc[dt, members].std(skipna=True)),
            "rel_minus_trend": float(comp["rel"].loc[dt, members].mean() - comp["trend"].loc[dt, members].mean()),
            "rel_minus_mom": float(comp["rel"].loc[dt, members].mean() - comp["mom"].loc[dt, members].mean()),
        }
        for c in ic_cols:
            row[f"ic_{c}"] = float(ic_df.loc[dt, c]) if c in ic_df.columns and pd.notna(ic_df.loc[dt, c]) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows).set_index("dt").sort_index()
    return df


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
    mode_def: Dict[str, float],
    dd_penalty: float = 0.35,
    turnover_penalty: float = 0.02,
) -> float:
    if len(names) == 0:
        return 0.0
    rr = rets.loc[(rets.index > dt) & (rets.index <= next_dt), names].fillna(0.0)
    if rr.empty:
        return 0.0
    port = rr.mean(axis=1) * float(mode_def["exposure"])
    eq = (1.0 + port).cumprod()
    total_ret = float(eq.iloc[-1] - 1.0)
    maxdd = float((eq / eq.cummax() - 1.0).min())
    util = total_ret - dd_penalty * abs(maxdd) - turnover_penalty * max(0, len(names) - 1) / max(1, len(names))
    return util


def _build_training_labels(
    feature_table: pd.DataFrame,
    ohlcv: Dict[str, pd.DataFrame],
    cfg: MahoragaAI11Config,
    train_start: str,
    train_end: str,
    universe_schedule: Optional[pd.DataFrame],
    params: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    all_sched_tickers = set()
    if universe_schedule is not None and not universe_schedule.empty:
        for members_json in universe_schedule["members"]:
            all_sched_tickers |= set(json.loads(members_json))
        universe = sorted(all_sched_tickers & set(ohlcv["close"].columns))
    else:
        universe = [t for t in cfg.universe_static if t in ohlcv["close"].columns]
    close = ohlcv["close"][universe].copy()
    qqq = m6.to_s(ohlcv["close"][cfg.bench_qqq].reindex(close.index).ffill(), "QQQ")
    rets = close.pct_change().fillna(0.0)
    comp = _component_scores(close, qqq, cfg)
    reb_dates = [d for d in feature_table.index if pd.Timestamp(train_start) <= d <= pd.Timestamp(train_end)]
    mode_lib = _mode_library(cfg, params)

    rows = []
    for i, dt in enumerate(reb_dates[:-1]):
        next_dt = reb_dates[i + 1]
        if next_dt > pd.Timestamp(train_end):
            break
        members = _active_members(dt, universe_schedule, cfg.universe_static, list(close.columns))
        if len(members) < 2:
            continue

        action_scores = {}
        for action_name, mode_def in mode_lib.items():
            names = _select_mode_names(dt, members, comp, mode_def, int(mode_def["top_k"]))
            action_scores[action_name] = _period_mode_utility(dt, next_dt, names, rets, mode_def)

        base_util = float(action_scores["BASELINE"])
        candidate_actions = {k: v for k, v in action_scores.items() if k != "BASELINE"}
        best_action, best_util = max(candidate_actions.items(), key=lambda kv: kv[1])
        gain_vs_base = float(best_util - base_util)
        chosen_action = best_action if gain_vs_base > float(params["action_gain"]) else "BASELINE"

        row = feature_table.loc[dt].copy()
        row["baseline_util"] = base_util
        row["best_action"] = chosen_action
        row["best_action_gain"] = gain_vs_base
        for k, v in action_scores.items():
            row[f"util_{k}"] = float(v)
        rows.append(row)

    lab = pd.DataFrame(rows)
    if lab.empty:
        return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=object), pd.DataFrame()

    frag_cut = float(lab["baseline_util"].quantile(float(params["fragility_q"])))
    lab["fragile"] = (lab["baseline_util"] <= frag_cut).astype(int)
    # Outside fragile set, keep baseline as label.
    lab.loc[lab["fragile"] == 0, "best_action"] = "BASELINE"

    X = lab[feature_table.columns].copy()
    y_frag = lab["fragile"].astype(int)
    y_action = lab["best_action"].astype(str)
    meta = lab[["baseline_util", "best_action_gain"] + [c for c in lab.columns if c.startswith("util_")]].copy()
    meta["frag_cut"] = frag_cut
    return X, y_frag, y_action, meta


class _TabularModel:
    def __init__(self, model, label_encoder: Optional[LabelEncoder] = None):
        self.label_encoder = label_encoder
        self.pipeline = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("model", model),
        ])
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None):
        self.feature_names = list(X.columns)
        yy = y.copy()
        if self.label_encoder is not None:
            yy = pd.Series(self.label_encoder.fit_transform(yy.astype(str)), index=y.index)
        self.pipeline.fit(X[self.feature_names], yy.values, model__sample_weight=sample_weight)
        return self

    def predict(self, X: pd.DataFrame):
        pred = self.pipeline.predict(X[self.feature_names])
        if self.label_encoder is not None:
            pred = self.label_encoder.inverse_transform(pred.astype(int))
        return pred

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        model = self.pipeline.named_steps["model"]
        if hasattr(model, "predict_proba"):
            return model.predict_proba(self.pipeline.named_steps["imp"].transform(X[self.feature_names]))
        pred = self.predict(X)
        classes = getattr(self.label_encoder, "classes_", np.sort(pd.Series(pred).unique()))
        probs = np.zeros((len(pred), len(classes)))
        mapping = {k: i for i, k in enumerate(classes)}
        for i, p in enumerate(pred):
            probs[i, mapping[p]] = 1.0
        return probs


def _sample_weights(n: int, bias: float) -> np.ndarray:
    if n <= 0:
        return np.array([])
    if not np.isfinite(bias) or bias <= 1.0:
        return np.ones(n, dtype=float)
    return np.linspace(1.0, bias, n, dtype=float)


def _fit_ai11_models(
    feature_table: pd.DataFrame,
    ohlcv: Dict[str, pd.DataFrame],
    cfg: MahoragaAI11Config,
    train_start: str,
    train_end: str,
    universe_schedule: Optional[pd.DataFrame],
    params: Dict[str, float],
) -> Dict[str, Any]:
    X, y_frag, y_action, meta = _build_training_labels(feature_table, ohlcv, cfg, train_start, train_end, universe_schedule, params)
    out: Dict[str, Any] = {"X": X, "y_frag": y_frag, "y_action": y_action, "meta": meta, "params": deepcopy(params)}
    if len(X) < cfg.ai_min_train_samples:
        return out

    sample_weight = _sample_weights(len(X), cfg.ai_recent_sample_bias)

    # Fragility model
    if y_frag.nunique() >= 2 and y_frag.value_counts().min() >= cfg.ai_min_class_count:
        frag_clf = HistGradientBoostingClassifier(
            random_state=cfg.random_seed + 101,
            learning_rate=0.05,
            max_depth=3,
            max_iter=250,
            min_samples_leaf=12,
        )
        out["fragility_model"] = _TabularModel(frag_clf, None).fit(X, y_frag.astype(int), sample_weight=sample_weight)

    # Action model trained on fragile samples only
    frag_idx = y_frag[y_frag == 1].index
    if len(frag_idx) >= cfg.ai_min_train_samples // 2:
        y_act = y_action.loc[frag_idx]
        if y_act.nunique() >= 2 and y_act.value_counts().min() >= cfg.ai_min_class_count:
            enc = LabelEncoder()
            act_clf = HistGradientBoostingClassifier(
                random_state=cfg.random_seed + 202,
                learning_rate=0.05,
                max_depth=3,
                max_iter=250,
                min_samples_leaf=10,
            )
            # re-map weights to fragile subset
            sw_frag = _sample_weights(len(frag_idx), cfg.ai_recent_sample_bias)
            out["action_model"] = _TabularModel(act_clf, enc).fit(X.loc[frag_idx], y_act.astype(str), sample_weight=sw_frag)
    return out


def _predict_controls(
    feature_table: pd.DataFrame,
    ai_models: Dict[str, Any],
    cfg: MahoragaAI11Config,
    params: Dict[str, float],
) -> pd.DataFrame:
    out = pd.DataFrame(index=feature_table.index)
    out["fragility_prob"] = 0.0
    out["action"] = cfg.ai_base_action
    out["action_conf"] = 0.0
    out["intervene"] = 0

    frag_model = ai_models.get("fragility_model")
    if frag_model is not None and len(feature_table) > 0:
        probs = frag_model.predict_proba(feature_table)
        if probs.shape[1] >= 2:
            out["fragility_prob"] = probs[:, 1]
        else:
            out["fragility_prob"] = probs.max(axis=1)

    act_model = ai_models.get("action_model")
    if act_model is not None and len(feature_table) > 0:
        act_pred = act_model.predict(feature_table)
        act_probs = act_model.predict_proba(feature_table)
        out["action"] = act_pred
        out["action_conf"] = act_probs.max(axis=1)

    intervene = (
        (out["fragility_prob"].astype(float) >= float(params["intervention_prob"]))
        & (out["action"].astype(str) != "BASELINE")
        & (out["action_conf"].astype(float) >= float(params["action_conf"]))
    )
    out.loc[~intervene, "action"] = "BASELINE"
    out.loc[~intervene, "action_conf"] = 0.0
    out["intervene"] = intervene.astype(int)
    return out


def backtest_ai11(
    ohlcv: Dict[str, pd.DataFrame],
    cfg: MahoragaAI11Config,
    costs: m6.CostsConfig,
    ai_models: Dict[str, Any],
    params: Dict[str, float],
    label: str = "MAHORAGA_AI1_1",
    universe_schedule: Optional[pd.DataFrame] = None,
    precomputed_controls: Optional[pd.DataFrame] = None,
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
        raise ValueError("[backtest_ai11] No valid tickers in universe")

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
    controls = precomputed_controls.copy() if precomputed_controls is not None else _predict_controls(feature_table, ai_models, cfg, params)
    mode_lib = _mode_library(cfg, params)

    reb_dates = set(close.resample(cfg.rebalance_freq).last().index)
    action_daily = pd.Series(cfg.ai_base_action, index=idx, name="ai_action")
    action_conf_daily = pd.Series(0.0, index=idx, name="ai_action_conf")
    frag_prob_daily = pd.Series(0.0, index=idx, name="ai_fragility_prob")
    intervene_daily = pd.Series(0, index=idx, name="ai_intervene")

    w = pd.DataFrame(0.0, index=idx, columns=univ_master)
    last_w = pd.Series(0.0, index=univ_master)
    current_action = cfg.ai_base_action
    current_action_conf = 0.0
    current_frag_prob = 0.0

    for dt in idx:
        if dt in reb_dates:
            if dt in controls.index:
                current_action = str(controls.loc[dt, "action"])
                current_action_conf = float(controls.loc[dt, "action_conf"])
                current_frag_prob = float(controls.loc[dt, "fragility_prob"])
            mode_def = mode_lib.get(current_action, mode_lib[cfg.ai_base_action])
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
                last_w[names] = ww.values * float(mode_def["exposure"])

        w.loc[dt] = last_w.values
        action_daily.loc[dt] = current_action
        action_conf_daily.loc[dt] = current_action_conf
        frag_prob_daily.loc[dt] = current_frag_prob
        intervene_daily.loc[dt] = int(current_action != "BASELINE")

    w_stop, stop_hits = m6.apply_chandelier(w, close, high, low, cfg)
    w_exec_1x = w_stop.shift(1).fillna(0.0)
    gross_1x = (w_exec_1x * rets).sum(axis=1)
    vol_sc = m6.vol_target_scale(gross_1x, cfg)
    cap = (crisis_scale * turb_scale * corr_scale).clip(0.0, cfg.max_exposure)
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

    controls_daily = pd.concat([action_daily, action_conf_daily, frag_prob_daily, intervene_daily], axis=1)
    controls_reb = controls.copy()
    return {
        "label": label,
        "returns_net": port_net,
        "equity": equity,
        "exposure": exposure,
        "turnover": to,
        "weights_scaled": w_exec,
        "total_scale_target": tgt_sc,
        "crisis_scale": crisis_scale,
        "turb_scale": turb_scale,
        "corr_scale": corr_scale,
        "qqq_eq": qqq_eq,
        "spy_eq": spy_eq,
        "bench": {"QQQ_eq": qqq_eq, "SPY_eq": spy_eq},
        "ai_controls": controls_daily,
        "ai_controls_reb": controls_reb,
        "meta_features": feature_table,
        "stop_hits": stop_hits,
    }


def _score_inner_val(base_s: Dict[str, float], ai_s: Dict[str, float], intervention_rate: float, target_rate: float) -> float:
    delta_sharpe = float(ai_s["Sharpe"] - base_s["Sharpe"])
    dd_improve = float(abs(base_s["MaxDD"]) - abs(ai_s["MaxDD"]))
    cagr_loss = float(max(0.0, base_s["CAGR"] - ai_s["CAGR"]))
    interv_pen = float(max(0.0, intervention_rate - target_rate))
    return delta_sharpe + 0.50 * dd_improve - 0.75 * cagr_loss - 0.15 * interv_pen


def calibrate_ai11_params(
    ohlcv: Dict[str, pd.DataFrame],
    cfg: MahoragaAI11Config,
    costs: m6.CostsConfig,
    train_start: str,
    train_end: str,
    universe_schedule: Optional[pd.DataFrame],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    feature_table = _build_meta_feature_table(ohlcv, cfg, universe_schedule)
    dates = [d for d in feature_table.index if pd.Timestamp(train_start) <= d <= pd.Timestamp(train_end)]
    if len(dates) < max(cfg.ai_min_train_samples, 25):
        default = {
            "fragility_q": cfg.ai_fragility_q,
            "intervention_prob": cfg.ai_intervention_prob,
            "action_conf": cfg.ai_action_conf,
            "rel_tilt": cfg.ai_rel_tilt,
            "defensive_exposure": cfg.ai_defensive_exposure,
            "defensive_cap": cfg.ai_defensive_cap,
            "action_gain": cfg.ai_action_gain,
        }
        return default, pd.DataFrame([default])

    cut = int(len(dates) * (1.0 - cfg.ai_inner_val_frac))
    cut = max(cut, cfg.ai_min_train_samples)
    cut = min(cut, len(dates) - 5)
    inner_train_start, inner_train_end = dates[0], dates[cut - 1]
    inner_val_start, inner_val_end = dates[cut], dates[-1]

    base_res = m6.backtest(ohlcv, cfg, costs, label="AI11_INNER_BASE", universe_schedule=universe_schedule)
    r_base = base_res["returns_net"].loc[inner_val_start:inner_val_end]
    eq_base = cfg.capital_initial * (1.0 + r_base).cumprod()
    exp_base = base_res["exposure"].loc[r_base.index]
    to_base = base_res["turnover"].loc[r_base.index]
    s_base = m6.summarize(r_base, eq_base, exp_base, to_base, cfg, "AI11_INNER_BASE")

    grid = list(iproduct(
        cfg.ai_fragility_q_grid,
        cfg.ai_intervention_prob_grid,
        cfg.ai_action_conf_grid,
        cfg.ai_rel_tilt_grid,
        cfg.ai_defensive_exposure_grid,
        cfg.ai_defensive_cap_grid,
        cfg.ai_action_gain_grid,
    ))

    rows = []
    best_score = -1e18
    best_params: Optional[Dict[str, float]] = None
    for frag_q, p_thr, a_thr, rel_tilt, def_exp, def_cap, gain in grid:
        params = {
            "fragility_q": float(frag_q),
            "intervention_prob": float(p_thr),
            "action_conf": float(a_thr),
            "rel_tilt": float(rel_tilt),
            "defensive_exposure": float(def_exp),
            "defensive_cap": float(def_cap),
            "action_gain": float(gain),
        }
        try:
            models = _fit_ai11_models(feature_table, ohlcv, cfg, str(inner_train_start.date()), str(inner_train_end.date()), universe_schedule, params)
            val_features = feature_table.loc[(feature_table.index >= pd.Timestamp(inner_val_start)) & (feature_table.index <= pd.Timestamp(inner_val_end))]
            controls_val = _predict_controls(val_features, models, cfg, params)
            ai_res = backtest_ai11(ohlcv, cfg, costs, models, params, label="AI11_INNER", universe_schedule=universe_schedule, precomputed_controls=controls_val)
            r_ai = ai_res["returns_net"].loc[inner_val_start:inner_val_end]
            eq_ai = cfg.capital_initial * (1.0 + r_ai).cumprod()
            exp_ai = ai_res["exposure"].loc[r_ai.index]
            to_ai = ai_res["turnover"].loc[r_ai.index]
            s_ai = m6.summarize(r_ai, eq_ai, exp_ai, to_ai, cfg, "AI11_INNER")
            intervention_rate = float(controls_val["intervene"].mean()) if len(controls_val) else 0.0
            score = _score_inner_val(s_base, s_ai, intervention_rate, cfg.ai_target_intervention_rate)
            row = {**params,
                   "inner_base_sharpe": s_base["Sharpe"], "inner_ai_sharpe": s_ai["Sharpe"],
                   "inner_base_cagr": s_base["CAGR"], "inner_ai_cagr": s_ai["CAGR"],
                   "inner_base_maxdd": s_base["MaxDD"], "inner_ai_maxdd": s_ai["MaxDD"],
                   "intervention_rate": intervention_rate, "score": score}
            rows.append(row)
            if score > best_score:
                best_score = score
                best_params = params
        except Exception as e:
            rows.append({**params, "score": -1e18, "error": str(e)})

    calib_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    if best_params is None:
        best_params = {
            "fragility_q": cfg.ai_fragility_q,
            "intervention_prob": cfg.ai_intervention_prob,
            "action_conf": cfg.ai_action_conf,
            "rel_tilt": cfg.ai_rel_tilt,
            "defensive_exposure": cfg.ai_defensive_exposure,
            "defensive_cap": cfg.ai_defensive_cap,
            "action_gain": cfg.ai_action_gain,
        }
    return best_params, calib_df


def run_walk_forward_ai11(
    ohlcv: Dict[str, pd.DataFrame],
    cfg_base: MahoragaAI11Config,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    folds = cfg_base.wf_folds
    ai_oos_r_all, ai_oos_exp_all, ai_oos_to_all = [], [], []
    base_oos_r_all, base_oos_exp_all, base_oos_to_all = [], [], []
    all_sweeps, fold_results, fold_artifacts = [], [], []
    last_best = None

    for fold_n, (train_end, val_start, val_end, test_start, test_end) in enumerate(folds, start=1):
        print(f"\n  ── AI1.1 FOLD {fold_n}/{len(folds)} ──")
        cfg_f = deepcopy(cfg_base)
        qqq_full = m6.to_s(ohlcv["close"][cfg_base.bench_qqq].ffill())
        dd_thr, vol_thr = m6.calibrate_crisis_thresholds(qqq_full, cfg_base.wf_train_start, train_end, cfg_f)
        cfg_f.crisis_dd_thr = dd_thr
        cfg_f.crisis_vol_zscore_thr = vol_thr

        print(f"  [fold {fold_n}] Fitting IC weights on train …")
        train_tickers = m6.get_training_universe(train_end, universe_schedule,
                                                 cfg_base.universe_static, list(ohlcv["close"].columns))
        close_univ = ohlcv["close"][train_tickers]
        ic_weights = m6.fit_ic_weights(close_univ, qqq_full.loc[cfg_base.wf_train_start:train_end], cfg_f, cfg_base.wf_train_start, train_end)

        print(f"  [fold {fold_n}] Sweeping base combo on val {val_start}→{val_end} …")
        sweep_df, best = m6.run_fold_sweep(
            ohlcv, cfg_f, costs, ic_weights, val_start, val_end, fold_n,
            universe_schedule=universe_schedule
        )
        sweep_df["fold"] = fold_n
        all_sweeps.append(sweep_df)
        last_best = best

        best_cfg = deepcopy(cfg_f)
        for k, v in best["combo_params"].items():
            setattr(best_cfg, k, v)
        best_cfg.w_trend, best_cfg.w_mom, best_cfg.w_rel = ic_weights

        print(f"  [fold {fold_n}] Calibrating AI1.1 on train via inner WFO …")
        best_params, calib_df = calibrate_ai11_params(
            ohlcv, best_cfg, costs, cfg_base.wf_train_start, train_end, universe_schedule
        )
        for k, v in best_params.items():
            setattr(best_cfg, f"ai_{k}" if not k.startswith("ai_") else k, v)

        feature_table = _build_meta_feature_table(ohlcv, best_cfg, universe_schedule)
        ai_models = _fit_ai11_models(feature_table, ohlcv, best_cfg, cfg_base.wf_train_start, train_end, universe_schedule, best_params)

        print(f"  [fold {fold_n}] Testing baseline and AI1.1 on {test_start}→{test_end} …")
        res_base = m6.backtest(ohlcv, best_cfg, costs, label=f"FOLD{fold_n}_BASE", universe_schedule=universe_schedule)
        test_features = feature_table.loc[(feature_table.index >= pd.Timestamp(test_start)) & (feature_table.index <= pd.Timestamp(test_end))]
        controls_test = _predict_controls(test_features, ai_models, best_cfg, best_params)
        res_ai = backtest_ai11(ohlcv, best_cfg, costs, ai_models, best_params, label=f"FOLD{fold_n}_AI11", universe_schedule=universe_schedule, precomputed_controls=controls_test)

        r_base = res_base["returns_net"].loc[test_start:test_end]
        eq_base = best_cfg.capital_initial * (1.0 + r_base).cumprod()
        exp_base = res_base["exposure"].loc[r_base.index]
        to_base = res_base["turnover"].loc[r_base.index]
        s_base = m6.summarize(r_base, eq_base, exp_base, to_base, best_cfg, f"FOLD{fold_n}_BASE")

        r_ai = res_ai["returns_net"].loc[test_start:test_end]
        eq_ai = best_cfg.capital_initial * (1.0 + r_ai).cumprod()
        exp_ai = res_ai["exposure"].loc[r_ai.index]
        to_ai = res_ai["turnover"].loc[r_ai.index]
        s_ai = m6.summarize(r_ai, eq_ai, exp_ai, to_ai, best_cfg, f"FOLD{fold_n}_AI11")

        print(f"  [fold {fold_n}] BASE Sharpe={s_base['Sharpe']:.3f} | AI1.1 Sharpe={s_ai['Sharpe']:.3f} | Δ={s_ai['Sharpe']-s_base['Sharpe']:+.3f}")

        ai_oos_r_all.append(r_ai)
        ai_oos_exp_all.append(exp_ai)
        ai_oos_to_all.append(to_ai)
        base_oos_r_all.append(r_base)
        base_oos_exp_all.append(exp_base)
        base_oos_to_all.append(to_base)

        ctrl_slice = controls_test.copy()
        fold_results.append({
            "fold": fold_n,
            "train": f"{cfg_base.wf_train_start}→{train_end}",
            "val": f"{val_start}→{val_end}",
            "test": f"{test_start}→{test_end}",
            **{f"best_{k}": v for k, v in best['combo_params'].items()},
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
            "fragility_prob_mean": round(float(ctrl_slice["fragility_prob"].mean()) if len(ctrl_slice) else 0.0, 4),
            "intervention_rate%": round(float(ctrl_slice["intervene"].mean() * 100) if len(ctrl_slice) else 0.0, 2),
            "mode_BASELINE_pct": round(float((ctrl_slice["action"] == "BASELINE").mean() * 100) if len(ctrl_slice) else 0.0, 2),
            "mode_REL_TILT_pct": round(float((ctrl_slice["action"] == "REL_TILT").mean() * 100) if len(ctrl_slice) else 0.0, 2),
            "mode_DEFENSIVE_LIGHT_pct": round(float((ctrl_slice["action"] == "DEFENSIVE_LIGHT").mean() * 100) if len(ctrl_slice) else 0.0, 2),
            **{f"ai11_{k}": v for k, v in best_params.items()},
        })
        fold_artifacts.append({
            "controls_test": controls_test,
            "calibration": calib_df,
            "feature_table": feature_table,
            "ai_models": ai_models,
            "best_params": best_params,
        })

    ai_oos_r_s = pd.concat(ai_oos_r_all).sort_index()
    ai_oos_exp_s = pd.concat(ai_oos_exp_all).sort_index().reindex(ai_oos_r_s.index)
    ai_oos_to_s = pd.concat(ai_oos_to_all).sort_index().reindex(ai_oos_r_s.index)
    ai_oos_eq = cfg_base.capital_initial * (1.0 + ai_oos_r_s).cumprod()

    base_oos_r_s = pd.concat(base_oos_r_all).sort_index()
    base_oos_exp_s = pd.concat(base_oos_exp_all).sort_index().reindex(base_oos_r_s.index)
    base_oos_to_s = pd.concat(base_oos_to_all).sort_index().reindex(base_oos_r_s.index)
    base_oos_eq = cfg_base.capital_initial * (1.0 + base_oos_r_s).cumprod()

    oos_label = "stitched_fold_tests"
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
        "fold_results": pd.DataFrame(fold_results),
        "all_sweeps": all_sweeps_df,
        "selected_config_info": selected_config_info,
    }


def _compare_summary_rows(cfg, label, r, eq, exp, to):
    s = m6.summarize(r, eq, exp, to, cfg, label)
    return {"Label": label, **s}


def _build_final_ai11_report(cfg: MahoragaAI11Config, comparison_full: pd.DataFrame, comparison_oos: pd.DataFrame,
                             fold_df: pd.DataFrame, mode_dist_df: pd.DataFrame, regime_df: pd.DataFrame,
                             selected_config_info: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 92)
    lines.append("  MAHORAGA AI1.1 — FINAL REPORT")
    lines.append("=" * 92)
    lines.append(AI11_DISCLAIMER.strip())
    lines.append("")
    lines.append(f"Model label: {cfg.label}")
    lines.append(f"Base selection method: {selected_config_info.get('source', 'unknown')}")
    lines.append("")
    lines.append("FULL SAMPLE COMPARISON")
    lines.append("-" * 92)
    lines.append(comparison_full.to_string(index=False))
    lines.append("")
    lines.append("OOS COMPARISON")
    lines.append("-" * 92)
    lines.append(comparison_oos.to_string(index=False))
    lines.append("")
    lines.append("FOLD SUMMARY")
    lines.append("-" * 92)
    lines.append(fold_df.to_string(index=False))
    if not mode_dist_df.empty:
        lines.append("")
        lines.append("MODE DISTRIBUTION (OOS)")
        lines.append("-" * 92)
        lines.append(mode_dist_df.to_string(index=False))
    if not regime_df.empty:
        lines.append("")
        lines.append("AI1.1 REGIME SLICES")
        lines.append("-" * 92)
        lines.append(regime_df.to_string(index=False))
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- AI1.1 aims to preserve Mahoraga 6.1 as the default policy and intervene sparsely.")
    lines.append("- Interventions are selected by inner-WFO calibration, not by fixed hard-coded intuition.")
    lines.append("- Improvements should be judged primarily on weak folds, hostile slices and preservation of CAGR.")
    return "\n".join(lines)


def _perm_importance_df(model_wrap: Optional[_TabularModel], X: pd.DataFrame, y: pd.Series, random_state: int = 0) -> pd.DataFrame:
    if model_wrap is None or X is None or len(X) < 20 or y is None or len(y) != len(X):
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    try:
        X_imp = model_wrap.pipeline.named_steps["imp"].transform(X[model_wrap.feature_names])
        model = model_wrap.pipeline.named_steps["model"]
        pi = permutation_importance(model, X_imp, y, n_repeats=8, random_state=random_state, scoring="accuracy")
        df = pd.DataFrame({
            "feature": model_wrap.feature_names,
            "importance_mean": pi.importances_mean,
            "importance_std": pi.importances_std,
        }).sort_values("importance_mean", ascending=False)
        return df
    except Exception:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])


def save_outputs_ai11(cfg: MahoragaAI11Config, wf: Dict[str, Any], full_ai: Dict[str, Any], full_base: Dict[str, Any]):
    d = cfg.outputs_dir
    _ensure_dir(d)
    oos_idx = wf["ai_oos_eq"].index

    comparison_full = pd.DataFrame([
        _compare_summary_rows(cfg, "MAHORAGA_AI1_1_FULL", full_ai["returns_net"], full_ai["equity"], full_ai["exposure"], full_ai["turnover"]),
        _compare_summary_rows(cfg, "MAHORAGA_6_1_FULL", full_base["returns_net"], full_base["equity"], full_base["exposure"], full_base["turnover"]),
        _compare_summary_rows(cfg, "QQQ_FULL", (full_base["bench"]["QQQ_eq"].pct_change().fillna(0.0)), full_base["bench"]["QQQ_eq"], pd.Series(1.0, index=full_base["bench"]["QQQ_eq"].index), pd.Series(0.0, index=full_base["bench"]["QQQ_eq"].index)),
    ])
    comparison_oos = pd.DataFrame([
        _compare_summary_rows(cfg, "MAHORAGA_AI1_1_OOS", wf["ai_oos_r"], wf["ai_oos_eq"], wf["ai_oos_exp"], wf["ai_oos_to"]),
        _compare_summary_rows(cfg, "MAHORAGA_6_1_OOS", wf["base_oos_r"], wf["base_oos_eq"], wf["base_oos_exp"], wf["base_oos_to"]),
        _compare_summary_rows(cfg, "QQQ_OOS", full_base["bench"]["QQQ_eq"].pct_change().fillna(0.0).loc[oos_idx], full_base["bench"]["QQQ_eq"].loc[oos_idx], pd.Series(1.0, index=oos_idx), pd.Series(0.0, index=oos_idx)),
    ])
    comparison_full.to_csv(f"{d}/comparison_full.csv", index=False)
    comparison_oos.to_csv(f"{d}/comparison_oos.csv", index=False)

    fold_df = wf["fold_results"].copy()
    fold_df.to_csv(f"{d}/walk_forward_folds_ai.csv", index=False)
    wf["all_sweeps"].to_csv(f"{d}/walk_forward_sweeps.csv", index=False)
    wf["selected_config_info"]["selection_audit_df"].to_csv(f"{d}/selection_audit.csv", index=False)

    controls_reb = full_ai["ai_controls_reb"].copy()
    controls_reb.to_csv(f"{d}/dynamic_mode_controls.csv", index=True)
    full_ai["meta_features"].to_csv(f"{d}/meta_features_snapshot.csv", index=True)

    pred_df = controls_reb.copy()
    pred_df.rename(columns={"action": "predicted_action", "fragility_prob": "fragility_prob", "action_conf": "action_conf", "intervene": "intervene"}, inplace=True)
    pred_df.to_csv(f"{d}/meta_predictions.csv", index=True)
    pred_df[["fragility_prob"]].rename(columns={"fragility_prob": "regime_fragility_score"}).to_csv(f"{d}/regime_fragility_score.csv", index=True)

    # Regime slices for AI OOS
    ai_ctrl_oos = full_ai["ai_controls"].reindex(oos_idx).ffill().fillna({"ai_action": "BASELINE", "ai_action_conf": 0.0, "ai_fragility_prob": 0.0, "ai_intervene": 0})
    regime_rows = []
    masks = {
        "ALL_OOS": pd.Series(True, index=oos_idx),
        "HIGH_FRAGILITY": ai_ctrl_oos["ai_fragility_prob"] >= cfg.ai_intervention_prob,
        "INTERVENED": ai_ctrl_oos["ai_intervene"] >= 1,
    }
    for name, mask in masks.items():
        rr = wf["ai_oos_r"].loc[mask]
        if len(rr) == 0:
            continue
        eq = cfg.capital_initial * (1.0 + rr).cumprod()
        regime_rows.append(_compare_summary_rows(cfg, name, rr, eq, wf["ai_oos_exp"].loc[rr.index], wf["ai_oos_to"].loc[rr.index]))
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(f"{d}/regime_oos_ai11.csv", index=False)

    mode_dist = pd.DataFrame({
        "Action": controls_reb["action"].value_counts(normalize=True).index,
        "Pct": (controls_reb["action"].value_counts(normalize=True).values * 100).round(2),
    })
    mode_dist.to_csv(f"{d}/mode_distribution_oos.csv", index=False)

    # Feature importances from final models
    final_art = wf["selected_config_info"]["fold_artifacts"][-1] if wf["selected_config_info"]["fold_artifacts"] else None
    if final_art is not None:
        models = final_art["ai_models"]
        X = models.get("X")
        y_frag = models.get("y_frag")
        y_action = models.get("y_action")
        frag_imp = _perm_importance_df(models.get("fragility_model"), X, y_frag.astype(int) if y_frag is not None and len(y_frag) == len(X) else None, random_state=cfg.random_seed + 1)
        act_idx = y_frag[y_frag == 1].index if y_frag is not None and len(y_frag) else pd.Index([])
        X_act = X.loc[act_idx] if X is not None and len(act_idx) else pd.DataFrame()
        y_act = y_action.loc[act_idx] if y_action is not None and len(act_idx) else pd.Series(dtype=object)
        act_model = models.get("action_model")
        act_imp = _perm_importance_df(act_model, X_act, y_act.astype(str) if len(y_act) == len(X_act) else None, random_state=cfg.random_seed + 2)
    else:
        frag_imp = pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
        act_imp = pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    frag_imp.to_csv(f"{d}/feature_importance_fragility.csv", index=False)
    act_imp.to_csv(f"{d}/feature_importance_action.csv", index=False)

    report = _build_final_ai11_report(cfg, comparison_full, comparison_oos, fold_df, mode_dist, regime_df, wf["selected_config_info"])
    with open(f"{d}/final_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n  [outputs → ./{d}/]")
    print("    comparison_full.csv, comparison_oos.csv, walk_forward_folds_ai.csv, walk_forward_sweeps.csv")
    print("    dynamic_mode_controls.csv, meta_features_snapshot.csv, meta_predictions.csv, regime_fragility_score.csv")
    print("    feature_importance_fragility.csv, feature_importance_action.csv, selection_audit.csv, final_report.txt")


def make_plots_ai11(cfg: MahoragaAI11Config, wf: Dict[str, Any], full_ai: Dict[str, Any], full_base: Dict[str, Any]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(cfg.plots_dir)
    oos_idx = wf["ai_oos_eq"].index
    plt.figure(figsize=(12, 6))
    plt.plot(wf["ai_oos_eq"].index, wf["ai_oos_eq"], label="AI1.1 OOS")
    plt.plot(wf["base_oos_eq"].index, wf["base_oos_eq"], label="Base 6.1 OOS")
    plt.plot(full_base["bench"]["QQQ_eq"].loc[oos_idx].index, full_base["bench"]["QQQ_eq"].loc[oos_idx], label="QQQ OOS", linestyle="--")
    plt.yscale("log")
    plt.title("MahoragaAI1.1 vs Mahoraga6.1 vs QQQ (OOS)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.plots_dir}/01_equity_oos_ai11.png", dpi=150)
    plt.close()

    ctrl = full_ai["ai_controls"].reindex(oos_idx).ffill()
    plt.figure(figsize=(12, 4))
    plt.plot(ctrl.index, ctrl["ai_fragility_prob"], label="Fragility probability")
    plt.axhline(cfg.ai_intervention_prob, linestyle="--", label="Intervention threshold")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{cfg.plots_dir}/02_fragility_prob_oos.png", dpi=150)
    plt.close()


def run_mahoraga_ai1_1(make_plots_flag: bool = True) -> Dict[str, Any]:
    print("=" * 80)
    print("  MAHORAGA AI1.1 — Fragility Detection + Bounded Intervention")
    print("=" * 80)
    print(AI11_DISCLAIMER)

    cfg = MahoragaAI11Config()
    costs = m6.CostsConfig()
    ucfg = m6.UniverseConfig()
    _ensure_dir(cfg.cache_dir)
    _ensure_dir(cfg.plots_dir)
    _ensure_dir(cfg.outputs_dir)

    print("\n[1] Downloading data …")
    equity_tickers = sorted(set(list(cfg.universe_static) + [t for u in m6.ALTERNATE_UNIVERSES.values() for t in u]))
    bench_tickers = [cfg.bench_qqq, cfg.bench_spy, cfg.bench_vix]
    all_tickers = sorted(set(equity_tickers + bench_tickers))
    ohlcv = m6.download_ohlcv(all_tickers, cfg.data_start, cfg.data_end, cfg.cache_dir)

    print("\n[1b] Canonical universe prep …")
    asset_registry = m6.build_asset_registry(equity_tickers, cfg, bench_tickers)
    data_quality_report = m6.compute_data_quality_report(ohlcv, equity_tickers, cfg)
    clean_equity = m6.filter_equity_candidates(
        [t for t in equity_tickers if t in ohlcv["close"].columns],
        asset_registry, data_quality_report, cfg
    )
    if len(clean_equity) < cfg.min_universe_names:
        clean_equity = [t for t in equity_tickers if t in ohlcv["close"].columns]

    universe_schedule = None
    if cfg.use_canonical_universe:
        universe_schedule, _ = m6.build_canonical_universe_schedule(
            ohlcv["close"], ohlcv["volume"], ucfg, clean_equity,
            cfg.data_start, cfg.data_end,
            registry_df=asset_registry, quality_df=data_quality_report,
        )

    print("\n[2] Walk-forward AI1.1 over Mahoraga 6.1 base …")
    wf = run_walk_forward_ai11(ohlcv, cfg, costs, universe_schedule=universe_schedule)

    print("\n[3] Final selected config + full-period evaluation …")
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
    wt, wm, wr = m6.fit_ic_weights(close_univ, qqq_full.loc[cfg.wf_train_start:last_train_end], cfg_final, cfg.wf_train_start, last_train_end)
    cfg_final.w_trend, cfg_final.w_mom, cfg_final.w_rel = wt, wm, wr

    # Calibrate/train final AI1.1 on the full training span available to 6.1.
    final_params, _ = calibrate_ai11_params(ohlcv, cfg_final, costs, cfg.wf_train_start, last_train_end, universe_schedule)
    feature_table = _build_meta_feature_table(ohlcv, cfg_final, universe_schedule)
    ai_models = _fit_ai11_models(feature_table, ohlcv, cfg_final, cfg.wf_train_start, last_train_end, universe_schedule, final_params)
    full_ai = backtest_ai11(ohlcv, cfg_final, costs, ai_models, final_params, label="MAHORAGA_AI1_1_FULL", universe_schedule=universe_schedule)
    full_base = m6.backtest(ohlcv, cfg_final, costs, label="MAHORAGA_6_1_FULL", universe_schedule=universe_schedule)

    save_outputs_ai11(cfg_final, wf, full_ai, full_base)
    if make_plots_flag:
        make_plots_ai11(cfg_final, wf, full_ai, full_base)

    return {"cfg": cfg_final, "wf": wf, "full_ai": full_ai, "full_base": full_base, "final_ai_params": final_params}


if __name__ == "__main__":
    results = run_mahoraga_ai1_1(make_plots_flag=True)
