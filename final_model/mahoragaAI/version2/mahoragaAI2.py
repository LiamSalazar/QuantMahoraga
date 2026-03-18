from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
"""
MahoragaAI2
===========
Sparse fragility-aware AI layer over Mahoraga 6.1.

Core design
-----------
- Freeze Mahoraga 6.1 as the execution/risk baseline.
- Reuse per-fold winning baseline parameters from Mahoraga 6.1 outputs.
- Predict *baseline fragility* (not generic market panic) on weekly decision dates.
- Allow only bounded interventions:
    BASELINE / REL_TILT / DEFENSIVE_LIGHT
- Calibrate the AI layer only on the train slice of each outer fold.
- Keep runtime practical by:
    * avoiding the 72-combo base sweep inside AI2,
    * precomputing features once,
    * deciding weekly (not daily),
    * using a compact inner grid.

This file must live next to `mahoraga6_1.py`.
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

AI2_DISCLAIMER = r"""
═══════════════════════════════════════════════════════════════════════════════
  MAHORAGA AI2 — METHODOLOGY DISCLAIMER
───────────────────────────────────────────────────────────────────────────────
  AI2 does NOT predict prices. It predicts whether the frozen Mahoraga 6.1
  baseline is likely to be fragile on the next decision window, and if so,
  whether a small bounded intervention is preferable.

  Allowed actions:
    • BASELINE         → keep Mahoraga 6.1 unchanged
    • REL_TILT         → modest tilt toward Relative Strength
    • DEFENSIVE_LIGHT  → modest risk reduction (not cash, not hard-off)

  AI2 is trained only on the train slice of each outer fold using inner
  validation. The baseline is frozen from Mahoraga 6.1 fold outputs.
═══════════════════════════════════════════════════════════════════════════════
"""


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


@dataclass
class MahoragaAI2Config(m6.Mahoraga6Config):
    outputs_dir: str = "mahoragaAI2_outputs"
    plots_dir: str = "mahoragaAI2_plots"
    label: str = "MAHORAGA_AI2"

    # Baseline 6.1 artifacts
    baseline_outputs_dir: str = "mahoraga6_1_outputs"
    baseline_folds_csv = r"C:\Users\AMD RYZEN 7\OneDrive\Documentos\QuantMahoraga\mahoragaAI\version2\mahoraga6_1_outputs\walk_forward_folds.csv"

    # Runtime / decision frequency
    ai_decision_freq: str = "W-FRI"
    ai_parallel_backtests: bool = False

    # Minimum data requirements
    ai_min_train_samples: int = 60
    ai_min_fragile_samples: int = 12
    ai_min_action_samples: int = 8
    ai_recent_sample_bias: float = 1.25
    ai_inner_val_frac: float = 0.30
    ai_target_intervention_rate: float = 0.20

    # Candidate horizon / label grids (compact by design)
    ai_horizon_grid: Tuple[int, ...] = (10, 20)
    ai_fragility_q_grid: Tuple[float, ...] = (0.15, 0.20)
    ai_intervention_prob_grid: Tuple[float, ...] = (0.60, 0.70)
    ai_action_conf_grid: Tuple[float, ...] = (0.50, 0.60)
    ai_min_gain_grid: Tuple[float, ...] = (0.000, 0.003)

    # Action libraries to calibrate (small bounded set)
    ai_rel_tilt_grid: Tuple[float, ...] = (0.55, 0.60)
    ai_defensive_vol_mult_grid: Tuple[float, ...] = (0.80, 0.90)
    ai_defensive_cap_mult_grid: Tuple[float, ...] = (0.80, 0.90)

    # Selected per fold (populated during calibration)
    ai_horizon: int = 20
    ai_fragility_q: float = 0.20
    ai_intervention_prob: float = 0.65
    ai_action_conf: float = 0.55
    ai_min_gain: float = 0.0
    ai_rel_tilt: float = 0.60
    ai_defensive_vol_mult: float = 0.90
    ai_defensive_cap_mult: float = 0.90


def _find_baseline_folds_csv(cfg):
    candidates = []

    if getattr(cfg, "baseline_folds_csv", None):
        candidates.append(Path(cfg.baseline_folds_csv))

    candidates.extend([
        _SCRIPT_DIR / "mahoraga6_1_outputs" / "walk_forward_folds.csv",
        _SCRIPT_DIR / "walk_forward_folds.csv",
        Path.cwd() / "mahoraga6_1_outputs" / "walk_forward_folds.csv",
        Path.cwd() / "walk_forward_folds.csv",
    ])

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "No baseline walk_forward_folds.csv found. "
        "Set `baseline_folds_csv` or place Mahoraga 6.1 outputs next to this file."
    )


def _load_frozen_baseline_folds(cfg: MahoragaAI2Config) -> pd.DataFrame:
    p = _find_baseline_folds_csv(cfg)
    df = pd.read_csv(p)
    req = {
        "fold", "best_weight_cap", "best_k_atr", "best_turb_zscore_thr",
        "best_turb_scale_min", "best_vol_target_ann"
    }
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Baseline fold CSV missing required columns: {sorted(miss)}")
    return df


def _apply_frozen_fold_params(cfg_fold: MahoragaAI2Config, baseline_df: pd.DataFrame, fold_n: int) -> MahoragaAI2Config:
    row = baseline_df.loc[baseline_df["fold"] == fold_n]
    if row.empty:
        raise ValueError(f"No frozen baseline row for fold={fold_n}")
    row = row.iloc[0]
    cfg_fold.weight_cap = float(row["best_weight_cap"])
    cfg_fold.k_atr = float(row["best_k_atr"])
    cfg_fold.turb_zscore_thr = float(row["best_turb_zscore_thr"])
    cfg_fold.turb_scale_min = float(row["best_turb_scale_min"])
    cfg_fold.vol_target_ann = float(row["best_vol_target_ann"])
    return cfg_fold


def _normalize_triplet(a: float, b: float, c: float) -> Tuple[float, float, float]:
    arr = np.array([a, b, c], dtype=float)
    s = arr.sum()
    if not np.isfinite(s) or s <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    arr = arr / s
    return tuple(float(x) for x in arr)


def _component_scores(close: pd.DataFrame, qqq: pd.Series, cfg: MahoragaAI2Config) -> Dict[str, pd.DataFrame]:
    idx = close.index
    qqq_ = m6.to_s(qqq, "QQQ").reindex(idx).ffill()
    out = {
        "trend": pd.DataFrame(index=idx, columns=close.columns, dtype=float),
        "mom":   pd.DataFrame(index=idx, columns=close.columns, dtype=float),
        "rel":   pd.DataFrame(index=idx, columns=close.columns, dtype=float),
    }
    for t in close.columns:
        cl = close[t].reindex(idx).ffill()
        out["trend"][t] = m6._trend(cl, cfg).fillna(0.0)
        out["mom"][t]   = m6._mom(cl, cfg).fillna(0.0)
        out["rel"][t]   = m6._rel(cl, qqq_, cfg).fillna(0.0)
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
    cfg: MahoragaAI2Config,
    universe_schedule: Optional[pd.DataFrame],
    comp: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if universe_schedule is not None and not universe_schedule.empty:
        all_sched = set()
        for members_json in universe_schedule["members"]:
            all_sched |= set(json.loads(members_json))
        universe = sorted(all_sched & set(ohlcv["close"].columns))
    else:
        universe = [t for t in cfg.universe_static if t in ohlcv["close"].columns]

    close = ohlcv["close"][universe].copy()
    volume = ohlcv["volume"][universe].copy()
    idx = close.index
    qqq = m6.to_s(ohlcv["close"][cfg.bench_qqq].reindex(idx).ffill(), "QQQ")
    qqq_r = qqq.pct_change().fillna(0.0)
    rets = close.pct_change().fillna(0.0)
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

    # Weekly decision dates (or nearest existing index)
    decision_idx = close.resample(cfg.ai_decision_freq).last().index
    rows = []
    for dt in decision_idx:
        if dt not in idx:
            prev = idx[idx <= dt]
            if len(prev) == 0:
                continue
            dt = prev[-1]
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
            "vix_z_63": float(m6.safe_z(vix.ffill().fillna(0.0), 63).loc[dt]) if dt in vix.index else np.nan,
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
            "trend_minus_mom": float(tr_row.mean() - mo_row.mean()),
            "n_members": float(len(members)),
        }
        rows.append((dt, row))

    feat = pd.DataFrame({dt: row for dt, row in rows}).T.sort_index()
    feat.index.name = "date"
    return feat


def _make_mode_weights(cfg: MahoragaAI2Config, rel_tilt: float) -> Tuple[float, float, float]:
    bw = np.array(_normalize_triplet(cfg.w_trend, cfg.w_mom, cfg.w_rel), dtype=float)
    rel_target = float(np.clip(rel_tilt, 0.34, 0.75))
    rem = max(0.0, 1.0 - rel_target)
    tm = bw[:2]
    tm_sum = tm.sum()
    tm = np.array([0.5, 0.5]) if tm_sum <= 0 else tm / tm_sum
    rel_w = np.array([rem * tm[0], rem * tm[1], rel_target], dtype=float)
    rel_w = rel_w / rel_w.sum()
    return tuple(float(x) for x in rel_w)


def _make_action_cfgs(base_cfg: MahoragaAI2Config, rel_tilt: float, def_vol_mult: float, def_cap_mult: float) -> Dict[str, MahoragaAI2Config]:
    cfg_base = deepcopy(base_cfg)

    cfg_rel = deepcopy(base_cfg)
    wr = _make_mode_weights(cfg_rel, rel_tilt)
    cfg_rel.w_trend, cfg_rel.w_mom, cfg_rel.w_rel = wr

    cfg_def = deepcopy(base_cfg)
    cfg_def.vol_target_ann = float(np.clip(base_cfg.vol_target_ann * def_vol_mult, 0.10, base_cfg.vol_target_ann))
    cfg_def.weight_cap = float(np.clip(base_cfg.weight_cap * def_cap_mult, 0.25, base_cfg.weight_cap))
    cfg_def.max_exposure = float(min(base_cfg.max_exposure, 0.95))

    return {
        "BASELINE": cfg_base,
        "REL_TILT": cfg_rel,
        "DEFENSIVE_LIGHT": cfg_def,
    }


def _run_action_backtests(
    ohlcv: Dict[str, pd.DataFrame],
    cfg: MahoragaAI2Config,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame],
    rel_tilt: float,
    def_vol_mult: float,
    def_cap_mult: float,
) -> Dict[str, Dict[str, Any]]:
    cfgs = _make_action_cfgs(cfg, rel_tilt, def_vol_mult, def_cap_mult)
    out = {}
    for action, c in cfgs.items():
        res = m6.backtest(ohlcv, c, costs, label=f"{cfg.label}_{action}", universe_schedule=universe_schedule)
        out[action] = res
    return out


def _forward_utility_from_series(r: pd.Series, start_dt: pd.Timestamp, horizon: int, cfg: MahoragaAI2Config, qqq_r: pd.Series) -> Tuple[float, Dict[str, float]]:
    idx = r.index
    if start_dt not in idx:
        nxt = idx[idx > start_dt]
        if len(nxt) == 0:
            return np.nan, {}
        start_dt = nxt[0]
    pos = idx.get_loc(start_dt)
    end_pos = min(len(idx), pos + horizon)
    r_slice = r.iloc[pos:end_pos]
    if len(r_slice) < max(5, horizon // 2):
        return np.nan, {}
    eq = cfg.capital_initial * (1.0 + r_slice).cumprod()
    q_slice = qqq_r.reindex(r_slice.index).fillna(0.0)
    qeq = cfg.capital_initial * (1.0 + q_slice).cumprod()
    s = m6.summarize(r_slice, eq, pd.Series(1.0, index=r_slice.index), pd.Series(0.0, index=r_slice.index), cfg, "tmp")
    sq = m6.summarize(q_slice, qeq, pd.Series(1.0, index=q_slice.index), pd.Series(0.0, index=q_slice.index), cfg, "QQQ")
    u = m6.objective(s, sq, cfg)
    metrics = {"CAGR": s["CAGR"], "Sharpe": s["Sharpe"], "MaxDD": s["MaxDD"], "Objective": u}
    return float(u), metrics


def _build_training_targets(
    feat: pd.DataFrame,
    action_results: Dict[str, Dict[str, Any]],
    cfg: MahoragaAI2Config,
    qqq_r: pd.Series,
    train_start: str,
    train_end: str,
    horizon: int,
    fragility_q: float,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    feat_train = feat.loc[train_start:train_end].copy()
    rows = []
    for dt in feat_train.index:
        u = {}
        for action, res in action_results.items():
            val, metrics = _forward_utility_from_series(res["returns_net"], dt, horizon, cfg, qqq_r)
            u[action] = val
        if any(not np.isfinite(v) for v in u.values()):
            continue
        best_action = max(u, key=u.get)
        regret = float(u[best_action] - u["BASELINE"])
        rows.append((dt, {
            "u_baseline": u["BASELINE"],
            "u_rel": u["REL_TILT"],
            "u_def": u["DEFENSIVE_LIGHT"],
            "best_action": best_action,
            "regret": regret,
        }))
    if not rows:
        return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=object), pd.DataFrame()

    ydf = pd.DataFrame({dt: row for dt, row in rows}).T.sort_index()
    pos_regret = ydf["regret"].clip(lower=0.0)
    thr = float(pos_regret.quantile(fragility_q)) if (pos_regret > 0).sum() >= 5 else float(pos_regret.max())
    if not np.isfinite(thr):
        thr = 0.0
    fragile = ((ydf["regret"] >= thr) & (ydf["best_action"] != "BASELINE")).astype(int)
    X = feat.reindex(ydf.index).copy()
    action_y = ydf["best_action"].copy()
    action_y = action_y.where(fragile.astype(bool), "BASELINE")
    return X, fragile, action_y, ydf


def _sample_weights(index: pd.Index, cfg: MahoragaAI2Config) -> np.ndarray:
    n = len(index)
    if n == 0:
        return np.array([], dtype=float)
    x = np.linspace(0.0, 1.0, n)
    return 1.0 + (cfg.ai_recent_sample_bias - 1.0) * x


def _fit_fragility_model(X: pd.DataFrame, y_fragile: pd.Series, cfg: MahoragaAI2Config):
    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_depth=3,
            max_iter=200,
            min_samples_leaf=20,
            random_state=cfg.random_seed,
        )),
    ])
    model.fit(X, y_fragile, clf__sample_weight=_sample_weights(X.index, cfg))
    return model


def _fit_action_model(X: pd.DataFrame, y_action: pd.Series, cfg: MahoragaAI2Config):
    le = LabelEncoder()
    y_enc = le.fit_transform(y_action.astype(str))
    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_depth=3,
            max_iter=200,
            min_samples_leaf=20,
            random_state=cfg.random_seed,
        )),
    ])
    model.fit(X, y_enc, clf__sample_weight=_sample_weights(X.index, cfg))
    return {"model": model, "label_encoder": le}


def _predict_policy(
    feat_slice: pd.DataFrame,
    frag_model,
    action_model,
    cfg: MahoragaAI2Config,
) -> pd.DataFrame:
    idx = feat_slice.index
    frag_prob = pd.Series(0.0, index=idx, name="fragility_prob")
    action = pd.Series("BASELINE", index=idx, name="action")
    action_conf = pd.Series(0.0, index=idx, name="action_conf")

    if frag_model is None or len(feat_slice) == 0:
        return pd.concat([frag_prob, action, action_conf], axis=1)

    pp = frag_model.predict_proba(feat_slice)
    # binary classes [0,1]
    if pp.shape[1] == 2:
        frag_prob[:] = pp[:, 1]
    else:
        frag_prob[:] = 0.0

    if action_model is not None:
        pr = action_model["model"].predict_proba(feat_slice)
        pred_idx = np.argmax(pr, axis=1)
        pred_label = action_model["label_encoder"].inverse_transform(pred_idx)
        pred_conf = pr.max(axis=1)
        action[:] = pred_label
        action_conf[:] = pred_conf

    use_intervention = (frag_prob >= cfg.ai_intervention_prob) & (action_conf >= cfg.ai_action_conf) & (action != "BASELINE")
    action = action.where(use_intervention, "BASELINE")
    action_conf = action_conf.where(use_intervention, 0.0)
    return pd.concat([frag_prob, action, action_conf], axis=1)


def _stitch_action_series(
    action_results: Dict[str, Dict[str, Any]],
    policy_decisions: pd.DataFrame,
    start: str,
    end: str,
    cfg: MahoragaAI2Config,
) -> Dict[str, Any]:
    baseline_idx = action_results["BASELINE"]["returns_net"].index
    daily_idx = baseline_idx[(baseline_idx >= pd.Timestamp(start)) & (baseline_idx <= pd.Timestamp(end))]
    if len(daily_idx) == 0:
        return {
            "returns_net": pd.Series(dtype=float),
            "equity": pd.Series(dtype=float),
            "exposure": pd.Series(dtype=float),
            "turnover": pd.Series(dtype=float),
            "policy_daily": pd.DataFrame(index=daily_idx),
        }

    # Forward-fill weekly decisions to daily index
    decisions = policy_decisions.copy()
    decisions = decisions.reindex(decisions.index.union(daily_idx)).sort_index().ffill()
    decisions = decisions.reindex(daily_idx).ffill()
    actions_daily = decisions["action"].fillna("BASELINE")
    frag_prob_daily = decisions["fragility_prob"].fillna(0.0)
    action_conf_daily = decisions["action_conf"].fillna(0.0)

    r = pd.Series(0.0, index=daily_idx, name="returns_net")
    exp = pd.Series(0.0, index=daily_idx, name="exposure")
    to = pd.Series(0.0, index=daily_idx, name="turnover")
    chosen = pd.Series("BASELINE", index=daily_idx, name="action")

    for action_name, res in action_results.items():
        mask = actions_daily == action_name
        if mask.any():
            r.loc[mask] = res["returns_net"].reindex(daily_idx).fillna(0.0).loc[mask]
            exp.loc[mask] = res["exposure"].reindex(daily_idx).fillna(0.0).loc[mask]
            to.loc[mask] = res["turnover"].reindex(daily_idx).fillna(0.0).loc[mask]
            chosen.loc[mask] = action_name

    eq = cfg.capital_initial * (1.0 + r).cumprod()
    policy_daily = pd.DataFrame({
        "action": chosen,
        "fragility_prob": frag_prob_daily,
        "action_conf": action_conf_daily,
    }, index=daily_idx)
    return {
        "returns_net": r,
        "equity": eq,
        "exposure": exp,
        "turnover": to,
        "policy_daily": policy_daily,
    }


def _calibration_score(base_sum: Dict[str, float], ai_sum: Dict[str, float], intervention_rate: float, cfg: MahoragaAI2Config) -> float:
    delta_sh = ai_sum["Sharpe"] - base_sum["Sharpe"]
    delta_cagr = ai_sum["CAGR"] - base_sum["CAGR"]
    delta_dd = abs(base_sum["MaxDD"]) - abs(ai_sum["MaxDD"])
    penalty_intervention = max(0.0, intervention_rate - cfg.ai_target_intervention_rate)
    return float(0.60 * delta_sh + 0.30 * delta_dd + 0.10 * delta_cagr - 0.10 * penalty_intervention)


def _collect_feature_importance(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 8) -> pd.DataFrame:
    if model is None or len(X) < 20:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    try:
        pi = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=123, n_jobs=1)
        out = pd.DataFrame({
            "feature": list(X.columns),
            "importance_mean": pi.importances_mean,
            "importance_std": pi.importances_std,
        }).sort_values("importance_mean", ascending=False)
        return out
    except Exception:
        return pd.DataFrame({"feature": list(X.columns), "importance_mean": np.nan, "importance_std": np.nan})


def calibrate_ai2_params(
    feat: pd.DataFrame,
    action_results_factory,
    cfg_fold: MahoragaAI2Config,
    qqq_r: pd.Series,
    train_start: str,
    train_end: str,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    feat_train = feat.loc[train_start:train_end].copy()
    if len(feat_train) < cfg_fold.ai_min_train_samples:
        return {
            "horizon": cfg_fold.ai_horizon_grid[0],
            "fragility_q": cfg_fold.ai_fragility_q_grid[0],
            "intervention_prob": cfg_fold.ai_intervention_prob_grid[0],
            "action_conf": cfg_fold.ai_action_conf_grid[0],
            "min_gain": cfg_fold.ai_min_gain_grid[0],
            "rel_tilt": cfg_fold.ai_rel_tilt_grid[0],
            "def_vol_mult": cfg_fold.ai_defensive_vol_mult_grid[0],
            "def_cap_mult": cfg_fold.ai_defensive_cap_mult_grid[0],
        }, pd.DataFrame()

    split_n = max(int(len(feat_train) * (1.0 - cfg_fold.ai_inner_val_frac)), cfg_fold.ai_min_train_samples)
    split_n = min(split_n, len(feat_train) - max(10, len(feat_train) // 5))
    train_sub_end = feat_train.index[split_n - 1]
    val_sub_start = feat_train.index[split_n]

    rows = []
    best_score = -np.inf
    best_params = None

    grid = list(iproduct(
        cfg_fold.ai_horizon_grid,
        cfg_fold.ai_fragility_q_grid,
        cfg_fold.ai_intervention_prob_grid,
        cfg_fold.ai_action_conf_grid,
        cfg_fold.ai_min_gain_grid,
        cfg_fold.ai_rel_tilt_grid,
        cfg_fold.ai_defensive_vol_mult_grid,
        cfg_fold.ai_defensive_cap_mult_grid,
    ))

    for (horizon, fq, iprob, aconf, min_gain, rel_tilt, dvm, dcm) in grid:
        action_results = action_results_factory(rel_tilt, dvm, dcm)
        X_tr, y_frag, y_act, ydf = _build_training_targets(
            feat, action_results, cfg_fold, qqq_r, train_start, str(train_sub_end.date()), horizon, fq
        )
        if len(X_tr) < cfg_fold.ai_min_train_samples:
            continue
        frag_count = int(y_frag.sum())
        if frag_count < cfg_fold.ai_min_fragile_samples:
            continue

        y_act_frag = y_act.loc[y_frag.index][y_frag.astype(bool)]
        counts = y_act_frag.value_counts()
        if len(counts) == 0:
            continue
        if counts.min() < cfg_fold.ai_min_action_samples:
            # keep BASELINE-only action model
            action_model = None
        else:
            action_model = _fit_action_model(X_tr.loc[y_frag.astype(bool)], y_act_frag, cfg_fold)

        frag_model = _fit_fragility_model(X_tr, y_frag, cfg_fold)

        X_val = feat.loc[val_sub_start:train_end].copy()
        if len(X_val) < 5:
            continue
        policy_decisions = _predict_policy(X_val, frag_model, action_model, cfg_fold)

        # require estimated gain over baseline on predicted intervention dates
        if min_gain > 0 and len(policy_decisions) > 0:
            # Approximate gain by realized next-window regret from train label table where available
            common = policy_decisions.index.intersection(ydf.index)
            if len(common) > 0:
                low_gain = ydf.reindex(common)["regret"].fillna(0.0) < float(min_gain)
                mask = (policy_decisions.loc[common, "action"] != "BASELINE") & low_gain
                policy_decisions.loc[common[mask], "action"] = "BASELINE"
                policy_decisions.loc[common[mask], "action_conf"] = 0.0

        stitched_ai = _stitch_action_series(action_results, policy_decisions, val_sub_start, train_end, cfg_fold)
        stitched_base = _stitch_action_series(action_results, policy_decisions.assign(action="BASELINE", action_conf=0.0), val_sub_start, train_end, cfg_fold)
        if len(stitched_ai["returns_net"]) < 20:
            continue
        s_ai = m6.summarize(stitched_ai["returns_net"], stitched_ai["equity"], stitched_ai["exposure"], stitched_ai["turnover"], cfg_fold, "AI2_VAL")
        s_base = m6.summarize(stitched_base["returns_net"], stitched_base["equity"], stitched_base["exposure"], stitched_base["turnover"], cfg_fold, "BASE_VAL")
        intervention_rate = float((policy_decisions["action"] != "BASELINE").mean()) if len(policy_decisions) else 0.0
        score = _calibration_score(s_base, s_ai, intervention_rate, cfg_fold)
        rows.append({
            "horizon": horizon,
            "fragility_q": fq,
            "intervention_prob": iprob,
            "action_conf": aconf,
            "min_gain": min_gain,
            "rel_tilt": rel_tilt,
            "def_vol_mult": dvm,
            "def_cap_mult": dcm,
            "score": score,
            "val_sharpe_base": s_base["Sharpe"],
            "val_sharpe_ai": s_ai["Sharpe"],
            "val_cagr_base": s_base["CAGR"],
            "val_cagr_ai": s_ai["CAGR"],
            "val_maxdd_base": s_base["MaxDD"],
            "val_maxdd_ai": s_ai["MaxDD"],
            "intervention_rate": intervention_rate,
        })
        if score > best_score:
            best_score = score
            best_params = {
                "horizon": horizon,
                "fragility_q": fq,
                "intervention_prob": iprob,
                "action_conf": aconf,
                "min_gain": min_gain,
                "rel_tilt": rel_tilt,
                "def_vol_mult": dvm,
                "def_cap_mult": dcm,
            }

    if best_params is None:
        best_params = {
            "horizon": cfg_fold.ai_horizon_grid[0],
            "fragility_q": cfg_fold.ai_fragility_q_grid[0],
            "intervention_prob": cfg_fold.ai_intervention_prob_grid[-1],
            "action_conf": cfg_fold.ai_action_conf_grid[-1],
            "min_gain": cfg_fold.ai_min_gain_grid[0],
            "rel_tilt": cfg_fold.ai_rel_tilt_grid[0],
            "def_vol_mult": cfg_fold.ai_defensive_vol_mult_grid[-1],
            "def_cap_mult": cfg_fold.ai_defensive_cap_mult_grid[-1],
        }

    return best_params, pd.DataFrame(rows)


def run_walk_forward_ai2(
    ohlcv: Dict[str, pd.DataFrame],
    cfg: MahoragaAI2Config,
    costs: m6.CostsConfig,
    universe_schedule: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    trading_idx = pd.DatetimeIndex(ohlcv["close"].index)
    folds = m6.build_contiguous_folds(cfg, trading_idx)
    baseline_df = _load_frozen_baseline_folds(cfg)

    # Build features once for the whole run using a neutral copy of cfg.
    if universe_schedule is not None and not universe_schedule.empty:
        all_sched = set()
        for members_json in universe_schedule["members"]:
            all_sched |= set(json.loads(members_json))
        ai_cols = sorted(all_sched & set(ohlcv["close"].columns))
    else:
        ai_cols = [t for t in cfg.universe_static if t in ohlcv["close"].columns]
    close_master = ohlcv["close"][ai_cols].copy()
    qqq_full = m6.to_s(ohlcv["close"][cfg.bench_qqq].reindex(close_master.index).ffill(), "QQQ")
    comp_full = _component_scores(close_master, qqq_full, cfg)
    feat_full = _build_meta_feature_table(ohlcv, cfg, universe_schedule, comp_full)

    fold_rows = []
    all_sweeps = []
    policy_artifacts = {}
    ai_oos_r = []
    ai_oos_exp = []
    ai_oos_to = []
    base_oos_r = []
    base_oos_exp = []
    base_oos_to = []
    feature_imp_frag = []
    feature_imp_action = []

    for fold in folds:
        fold_n = fold["fold"]
        train_start, train_end = fold["train_start"], fold["train_end"]
        test_start, test_end = fold["test_start"], fold["test_end"]
        print(f"\n  ── AI2 FOLD {fold_n}/{len(folds)} ──")

        cfg_f = deepcopy(cfg)
        cfg_f = _apply_frozen_fold_params(cfg_f, baseline_df, fold_n)
        dd_thr, vol_thr = m6.calibrate_crisis_thresholds(qqq_full, train_start, train_end, cfg_f)
        cfg_f.crisis_dd_thr = dd_thr
        cfg_f.crisis_vol_zscore_thr = vol_thr

        print(f"  [fold {fold_n}] Fitting IC weights on train …")
        train_tickers = m6.get_training_universe(train_end, universe_schedule, cfg_f.universe_static, list(ohlcv["close"].columns))
        close_univ = ohlcv["close"][train_tickers]
        wt, wm, wr = m6.fit_ic_weights(close_univ, qqq_full.loc[train_start:train_end], cfg_f, train_start, train_end)
        cfg_f.w_trend, cfg_f.w_mom, cfg_f.w_rel = wt, wm, wr
        print(f"    [IC] trend={wt:.3f} mom={wm:.3f} rel={wr:.3f}")

        cache = {}
        def action_results_factory(rel_tilt, dvm, dcm):
            key = (round(float(rel_tilt), 4), round(float(dvm), 4), round(float(dcm), 4))
            if key not in cache:
                cache[key] = _run_action_backtests(ohlcv, cfg_f, costs, universe_schedule, rel_tilt, dvm, dcm)
            return cache[key]

        print(f"  [fold {fold_n}] Calibrating AI2 on train via inner validation …")
        best_params, calib_df = calibrate_ai2_params(feat_full, action_results_factory, cfg_f, qqq_full.pct_change().fillna(0.0), train_start, train_end)
        all_sweeps.append(calib_df.assign(fold=fold_n))
        cfg_f.ai_horizon = int(best_params["horizon"])
        cfg_f.ai_fragility_q = float(best_params["fragility_q"])
        cfg_f.ai_intervention_prob = float(best_params["intervention_prob"])
        cfg_f.ai_action_conf = float(best_params["action_conf"])
        cfg_f.ai_min_gain = float(best_params["min_gain"])
        cfg_f.ai_rel_tilt = float(best_params["rel_tilt"])
        cfg_f.ai_defensive_vol_mult = float(best_params["def_vol_mult"])
        cfg_f.ai_defensive_cap_mult = float(best_params["def_cap_mult"])

        action_results = action_results_factory(cfg_f.ai_rel_tilt, cfg_f.ai_defensive_vol_mult, cfg_f.ai_defensive_cap_mult)
        X_train, y_frag, y_action, ydf = _build_training_targets(
            feat_full, action_results, cfg_f, qqq_full.pct_change().fillna(0.0), train_start, train_end, cfg_f.ai_horizon, cfg_f.ai_fragility_q
        )

        frag_model = None
        action_model = None
        if len(X_train) >= cfg_f.ai_min_train_samples and int(y_frag.sum()) >= cfg_f.ai_min_fragile_samples:
            frag_model = _fit_fragility_model(X_train, y_frag, cfg_f)
            X_frag = X_train.loc[y_frag.astype(bool)]
            y_frag_action = y_action.loc[X_frag.index]
            vc = y_frag_action.value_counts()
            if len(vc) > 0 and vc.min() >= cfg_f.ai_min_action_samples and y_frag_action.nunique() >= 2:
                action_model = _fit_action_model(X_frag, y_frag_action, cfg_f)

        feat_test = feat_full.loc[test_start:test_end].copy()
        print(f"  [fold {fold_n}] Testing baseline and AI2 on {test_start}→{test_end} …")
        policy_decisions = _predict_policy(feat_test, frag_model, action_model, cfg_f)

        # Apply minimum expected gain gate if train labels support it.
        if cfg_f.ai_min_gain > 0 and not ydf.empty:
            common = policy_decisions.index.intersection(ydf.index)
            if len(common) > 0:
                low_gain = ydf.reindex(common)["regret"].fillna(0.0) < cfg_f.ai_min_gain
                mask = (policy_decisions.loc[common, "action"] != "BASELINE") & low_gain
                policy_decisions.loc[common[mask], "action"] = "BASELINE"
                policy_decisions.loc[common[mask], "action_conf"] = 0.0

        ai_stitched = _stitch_action_series(action_results, policy_decisions, test_start, test_end, cfg_f)
        base_policy = policy_decisions.copy()
        base_policy["action"] = "BASELINE"
        base_policy["action_conf"] = 0.0
        base_stitched = _stitch_action_series(action_results, base_policy, test_start, test_end, cfg_f)

        s_base = m6.summarize(base_stitched["returns_net"], base_stitched["equity"], base_stitched["exposure"], base_stitched["turnover"], cfg_f, f"FOLD{fold_n}_BASE")
        s_ai = m6.summarize(ai_stitched["returns_net"], ai_stitched["equity"], ai_stitched["exposure"], ai_stitched["turnover"], cfg_f, f"FOLD{fold_n}_AI2")
        delta_sh = s_ai["Sharpe"] - s_base["Sharpe"]
        print(f"  [fold {fold_n}] BASE Sharpe={s_base['Sharpe']:.3f} | AI2 Sharpe={s_ai['Sharpe']:.3f} | Δ={delta_sh:+.3f}")

        base_oos_r.append(base_stitched["returns_net"])
        base_oos_exp.append(base_stitched["exposure"])
        base_oos_to.append(base_stitched["turnover"])
        ai_oos_r.append(ai_stitched["returns_net"])
        ai_oos_exp.append(ai_stitched["exposure"])
        ai_oos_to.append(ai_stitched["turnover"])

        intervention_rate = float((policy_decisions["action"] != "BASELINE").mean()) if len(policy_decisions) else 0.0
        mode_mix = policy_decisions["action"].value_counts(normalize=True).to_dict() if len(policy_decisions) else {}
        fold_rows.append({
            "fold": fold_n,
            "train": f"{train_start}→{train_end}",
            "test": f"{test_start}→{test_end}",
            "BASE_CAGR%": round(s_base["CAGR"] * 100, 2),
            "BASE_Sharpe": round(s_base["Sharpe"], 4),
            "BASE_MaxDD%": round(s_base["MaxDD"] * 100, 2),
            "AI_CAGR%": round(s_ai["CAGR"] * 100, 2),
            "AI_Sharpe": round(s_ai["Sharpe"], 4),
            "AI_MaxDD%": round(s_ai["MaxDD"] * 100, 2),
            "DeltaSharpe": round(delta_sh, 4),
            "InterventionRate": round(intervention_rate, 4),
            "Mode_BASELINE": round(mode_mix.get("BASELINE", 0.0), 4),
            "Mode_REL_TILT": round(mode_mix.get("REL_TILT", 0.0), 4),
            "Mode_DEFENSIVE_LIGHT": round(mode_mix.get("DEFENSIVE_LIGHT", 0.0), 4),
            "horizon": cfg_f.ai_horizon,
            "fragility_q": cfg_f.ai_fragility_q,
            "intervention_prob": cfg_f.ai_intervention_prob,
            "action_conf": cfg_f.ai_action_conf,
            "min_gain": cfg_f.ai_min_gain,
            "rel_tilt": cfg_f.ai_rel_tilt,
            "def_vol_mult": cfg_f.ai_defensive_vol_mult,
            "def_cap_mult": cfg_f.ai_defensive_cap_mult,
        })

        policy_artifacts[fold_n] = {
            "policy_decisions": policy_decisions,
            "policy_daily": ai_stitched["policy_daily"],
            "calibration": calib_df,
        }

        if frag_model is not None:
            fi_frag = _collect_feature_importance(frag_model, X_train, y_frag)
            if not fi_frag.empty:
                fi_frag["fold"] = fold_n
                feature_imp_frag.append(fi_frag)
        if action_model is not None:
            X_frag = X_train.loc[y_frag.astype(bool)]
            y_frag_action = y_action.loc[X_frag.index]
            if len(X_frag) >= 10:
                y_enc = action_model["label_encoder"].transform(y_frag_action.astype(str))
                fi_act = _collect_feature_importance(action_model["model"], X_frag, pd.Series(y_enc, index=X_frag.index))
                if not fi_act.empty:
                    fi_act["fold"] = fold_n
                    feature_imp_action.append(fi_act)

    base_r = pd.concat(base_oos_r).sort_index()
    base_exp = pd.concat(base_oos_exp).sort_index().reindex(base_r.index).fillna(0.0)
    base_to = pd.concat(base_oos_to).sort_index().reindex(base_r.index).fillna(0.0)
    base_eq = cfg.capital_initial * (1.0 + base_r).cumprod()

    ai_r = pd.concat(ai_oos_r).sort_index()
    ai_exp = pd.concat(ai_oos_exp).sort_index().reindex(ai_r.index).fillna(0.0)
    ai_to = pd.concat(ai_oos_to).sort_index().reindex(ai_r.index).fillna(0.0)
    ai_eq = cfg.capital_initial * (1.0 + ai_r).cumprod()

    return {
        "base_oos_r": base_r,
        "base_oos_eq": base_eq,
        "base_oos_exp": base_exp,
        "base_oos_to": base_to,
        "ai_oos_r": ai_r,
        "ai_oos_eq": ai_eq,
        "ai_oos_exp": ai_exp,
        "ai_oos_to": ai_to,
        "fold_results": pd.DataFrame(fold_rows),
        "policy_artifacts": policy_artifacts,
        "calibration_grid": pd.concat([x for x in all_sweeps if len(x) > 0], ignore_index=True) if all_sweeps else pd.DataFrame(),
        "feature_importance_fragility": pd.concat(feature_imp_frag, ignore_index=True) if feature_imp_frag else pd.DataFrame(),
        "feature_importance_action": pd.concat(feature_imp_action, ignore_index=True) if feature_imp_action else pd.DataFrame(),
        "feat_full": feat_full,
    }


def _selection_audit_baseline_vs_ai(wf_ai: Dict[str, Any], cfg: MahoragaAI2Config) -> pd.DataFrame:
    fr = wf_ai["fold_results"].copy()
    if fr.empty:
        return pd.DataFrame()
    out = pd.DataFrame([
        {
            "Method": "BASELINE_6_1_FROZEN",
            "MeanSharpe": fr["BASE_Sharpe"].mean(),
            "MeanCAGR%": fr["BASE_CAGR%"].mean(),
            "MeanMaxDD%": fr["BASE_MaxDD%"].mean(),
            "MeanInterventionRate": 0.0,
        },
        {
            "Method": "AI2_POLICY",
            "MeanSharpe": fr["AI_Sharpe"].mean(),
            "MeanCAGR%": fr["AI_CAGR%"].mean(),
            "MeanMaxDD%": fr["AI_MaxDD%"].mean(),
            "MeanInterventionRate": fr["InterventionRate"].mean(),
        },
    ])
    return out


def _regime_comparison(base_r: pd.Series, ai_r: pd.Series, cfg: MahoragaAI2Config, ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    idx = base_r.index.intersection(ai_r.index)
    if cfg.bench_vix in ohlcv.get("close", pd.DataFrame()).columns:
        vix = m6.to_s(ohlcv["close"][cfg.bench_vix].reindex(idx).ffill(), "VIX")
    else:
        vix = pd.Series(np.nan, index=idx)

    regimes = {
        "NORMAL": (vix < 18) if vix.notna().any() else pd.Series(True, index=idx),
        "STRESS": (vix >= 18) & (vix < 24) if vix.notna().any() else pd.Series(False, index=idx),
        "PANIC": (vix >= 24) if vix.notna().any() else pd.Series(False, index=idx),
    }
    rows = []
    for rg, mask in regimes.items():
        mask = mask.reindex(idx).fillna(False)
        if mask.sum() < 5:
            continue
        rb = base_r.loc[idx[mask]]
        ra = ai_r.loc[idx[mask]]
        sb = m6.summarize(rb, cfg.capital_initial * (1.0 + rb).cumprod(), pd.Series(1.0, index=rb.index), pd.Series(0.0, index=rb.index), cfg, f"BASE_{rg}")
        sa = m6.summarize(ra, cfg.capital_initial * (1.0 + ra).cumprod(), pd.Series(1.0, index=ra.index), pd.Series(0.0, index=ra.index), cfg, f"AI2_{rg}")
        rows.append({
            "Regime": rg,
            "Days": int(mask.sum()),
            "BASE_CAGR%": round(sb["CAGR"] * 100, 2),
            "BASE_Sharpe": round(sb["Sharpe"], 4),
            "BASE_MaxDD%": round(sb["MaxDD"] * 100, 2),
            "AI_CAGR%": round(sa["CAGR"] * 100, 2),
            "AI_Sharpe": round(sa["Sharpe"], 4),
            "AI_MaxDD%": round(sa["MaxDD"] * 100, 2),
            "DeltaSharpe": round(sa["Sharpe"] - sb["Sharpe"], 4),
        })
    return pd.DataFrame(rows)


def _final_report_text(cfg: MahoragaAI2Config, wf_ai: Dict[str, Any], ohlcv: Dict[str, pd.DataFrame]) -> str:
    base_sum = m6.summarize(wf_ai["base_oos_r"], wf_ai["base_oos_eq"], wf_ai["base_oos_exp"], wf_ai["base_oos_to"], cfg, "BASELINE_6_1")
    ai_sum = m6.summarize(wf_ai["ai_oos_r"], wf_ai["ai_oos_eq"], wf_ai["ai_oos_exp"], wf_ai["ai_oos_to"], cfg, "MAHORAGA_AI2")
    fr = wf_ai["fold_results"]
    selection = _selection_audit_baseline_vs_ai(wf_ai, cfg)
    regime = _regime_comparison(wf_ai["base_oos_r"], wf_ai["ai_oos_r"], cfg, ohlcv)

    lines = []
    lines.append("MAHORAGA AI2 — FINAL REPORT")
    lines.append("=" * 78)
    lines.append(AI2_DISCLAIMER.strip())
    lines.append("")
    lines.append("[OOS SUMMARY]")
    lines.append(f"  BASE  CAGR={base_sum['CAGR']*100:.2f}%  Sharpe={base_sum['Sharpe']:.3f}  MaxDD={base_sum['MaxDD']*100:.2f}%")
    lines.append(f"  AI2   CAGR={ai_sum['CAGR']*100:.2f}%  Sharpe={ai_sum['Sharpe']:.3f}  MaxDD={ai_sum['MaxDD']*100:.2f}%")
    lines.append(f"  Delta Sharpe={ai_sum['Sharpe'] - base_sum['Sharpe']:+.3f}  Delta MaxDD={(abs(base_sum['MaxDD']) - abs(ai_sum['MaxDD']))*100:+.2f} pp")
    lines.append("")
    lines.append("[FOLD SUMMARY]")
    if not fr.empty:
        for _, row in fr.iterrows():
            lines.append(
                f"  Fold {int(row['fold'])}: BASE Sharpe={row['BASE_Sharpe']:.3f} | AI2 Sharpe={row['AI_Sharpe']:.3f} | "
                f"Δ={row['DeltaSharpe']:+.3f} | InterventionRate={row['InterventionRate']:.2%}"
            )
    lines.append("")
    lines.append("[SELECTION AUDIT]")
    if not selection.empty:
        for _, row in selection.iterrows():
            lines.append(
                f"  {row['Method']}: MeanSharpe={row['MeanSharpe']:.3f}  MeanCAGR={row['MeanCAGR%']:.2f}%  "
                f"MeanMaxDD={row['MeanMaxDD%']:.2f}%  MeanIntervention={row['MeanInterventionRate']:.2%}"
            )
    lines.append("")
    lines.append("[REGIME COMPARISON]")
    if not regime.empty:
        for _, row in regime.iterrows():
            lines.append(
                f"  {row['Regime']}: BASE Sharpe={row['BASE_Sharpe']:.3f} | AI2 Sharpe={row['AI_Sharpe']:.3f} | Δ={row['DeltaSharpe']:+.3f}"
            )
    return "\n".join(lines) + "\n"


def save_outputs_ai2(cfg: MahoragaAI2Config, wf_ai: Dict[str, Any], ohlcv: Dict[str, pd.DataFrame]):
    d = cfg.outputs_dir
    _ensure_dir(d)

    base_sum = m6.summarize(wf_ai["base_oos_r"], wf_ai["base_oos_eq"], wf_ai["base_oos_exp"], wf_ai["base_oos_to"], cfg, "BASELINE_6_1")
    ai_sum = m6.summarize(wf_ai["ai_oos_r"], wf_ai["ai_oos_eq"], wf_ai["ai_oos_exp"], wf_ai["ai_oos_to"], cfg, "MAHORAGA_AI2")
    pd.DataFrame([base_sum, ai_sum]).to_csv(os.path.join(d, "comparison_oos.csv"), index=False)

    full_base = m6.final_evaluation(ohlcv, cfg, m6.CostsConfig(), None, wf_ai["base_oos_r"], wf_ai["base_oos_eq"], wf_ai["base_oos_exp"], wf_ai["base_oos_to"], "BASELINE_6_1")
    full_ai = m6.final_evaluation(ohlcv, cfg, m6.CostsConfig(), None, wf_ai["ai_oos_r"], wf_ai["ai_oos_eq"], wf_ai["ai_oos_exp"], wf_ai["ai_oos_to"], "MAHORAGA_AI2")
    # Only save summary tables, not whole dict.
    pd.DataFrame(full_base["comparison_full"] + full_ai["comparison_full"]).to_csv(os.path.join(d, "comparison_full.csv"), index=False)

    wf_ai["fold_results"].to_csv(os.path.join(d, "walk_forward_folds_ai.csv"), index=False)
    if len(wf_ai["calibration_grid"]) > 0:
        wf_ai["calibration_grid"].to_csv(os.path.join(d, "walk_forward_sweeps.csv"), index=False)
    if len(wf_ai["feature_importance_fragility"]) > 0:
        wf_ai["feature_importance_fragility"].to_csv(os.path.join(d, "feature_importance_fragility.csv"), index=False)
    if len(wf_ai["feature_importance_action"]) > 0:
        wf_ai["feature_importance_action"].to_csv(os.path.join(d, "feature_importance_action.csv"), index=False)
    wf_ai["feat_full"].to_csv(os.path.join(d, "meta_features_snapshot.csv"), index=True)

    # Policy artifacts
    all_daily = []
    all_decisions = []
    for fold_n, art in wf_ai["policy_artifacts"].items():
        dec = art["policy_decisions"].copy()
        dec["fold"] = fold_n
        all_decisions.append(dec.reset_index().rename(columns={"index": "date"}))
        pol = art["policy_daily"].copy()
        pol["fold"] = fold_n
        all_daily.append(pol.reset_index().rename(columns={"index": "date"}))
    if all_decisions:
        pd.concat(all_decisions, ignore_index=True).to_csv(os.path.join(d, "meta_predictions.csv"), index=False)
    if all_daily:
        dynamic = pd.concat(all_daily, ignore_index=True)
        dynamic.to_csv(os.path.join(d, "dynamic_mode_controls.csv"), index=False)
        dynamic[["date", "fragility_prob", "fold"]].rename(columns={"fragility_prob": "fragility_score"}).to_csv(os.path.join(d, "regime_fragility_score.csv"), index=False)

    _selection_audit_baseline_vs_ai(wf_ai, cfg).to_csv(os.path.join(d, "selection_audit.csv"), index=False)
    _regime_comparison(wf_ai["base_oos_r"], wf_ai["ai_oos_r"], cfg, ohlcv).to_csv(os.path.join(d, "regime_comparison.csv"), index=False)

    with open(os.path.join(d, "final_report.txt"), "w", encoding="utf-8") as f:
        f.write(_final_report_text(cfg, wf_ai, ohlcv))

    print(f"\n  [outputs → ./{d}/]")
    print("    comparison_full.csv, comparison_oos.csv, walk_forward_folds_ai.csv")
    print("    walk_forward_sweeps.csv, meta_features_snapshot.csv, meta_predictions.csv")
    print("    dynamic_mode_controls.csv, regime_fragility_score.csv, selection_audit.csv")
    print("    feature_importance_fragility.csv, feature_importance_action.csv, final_report.txt")


def run_mahoraga_ai2(make_plots_flag: bool = False) -> Dict[str, Any]:
    print("=" * 80)
    print("  MAHORAGA AI2 — Fragility-aware sparse intervention over 6.1")
    print("=" * 80)
    print(AI2_DISCLAIMER)

    cfg = MahoragaAI2Config()
    costs = m6.CostsConfig()
    _ensure_dir(cfg.outputs_dir)
    _ensure_dir(cfg.plots_dir)

    print("\n[1] Downloading data …")
    equity_tickers = sorted(set(list(cfg.universe_static) + [cfg.bench_qqq, cfg.bench_spy]))
    if cfg.bench_vix:
        equity_tickers.append(cfg.bench_vix)
    all_tickers = sorted(set(equity_tickers))
    ohlcv = m6.download_ohlcv(all_tickers, cfg.data_start, cfg.data_end, cfg.cache_dir)

    print("\n[2] Fama-French factors …")
    ff = m6.load_ff_factors(cfg.cache_dir)

    print("\n[3] Canonical universe engine …")
    asset_registry = m6.build_asset_registry([t for t in cfg.universe_static if t in ohlcv['close'].columns], cfg, [cfg.bench_qqq, cfg.bench_spy, cfg.bench_vix])
    data_quality_report = m6.compute_data_quality_report(ohlcv, [t for t in cfg.universe_static if t in ohlcv['close'].columns], cfg)
    clean_equity = m6.filter_equity_candidates(
        [t for t in cfg.universe_static if t in ohlcv['close'].columns],
        asset_registry, data_quality_report, cfg,
    )
    ucfg = m6.UniverseConfig()
    universe_schedule, universe_snapshots = m6.build_canonical_universe_schedule(
        ohlcv['close'], ohlcv['volume'], ucfg, clean_equity,
        cfg.data_start, cfg.data_end,
        registry_df=asset_registry, quality_df=data_quality_report,
    )
    print(f"  [universe] {len(universe_schedule)} reconstitution dates built")

    print("\n[4] Walk-forward AI2 over Mahoraga 6.1 base …")
    wf_ai = run_walk_forward_ai2(ohlcv, cfg, costs, universe_schedule=universe_schedule)

    save_outputs_ai2(cfg, wf_ai, ohlcv)
    return {"cfg": cfg, "wf_ai": wf_ai}


if __name__ == "__main__":
    results = run_mahoraga_ai2(make_plots_flag=False)
