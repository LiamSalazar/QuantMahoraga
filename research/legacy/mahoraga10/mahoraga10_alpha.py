from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import mahoraga6_1 as m6
from mahoraga10_config import Mahoraga10Config
from mahoraga10_utils import cross_sectional_z


def _trend(price: pd.Series, short_spans: tuple[int, ...], long_spans: tuple[int, ...]) -> pd.Series:
    votes = []
    for sp in short_spans:
        for lp in long_spans:
            if sp >= lp:
                continue
            es = price.ewm(span=sp, adjust=False).mean().shift(1)
            el = price.ewm(span=lp, adjust=False).mean().shift(1)
            votes.append(((es > el) & price.notna()).astype(float))
    return sum(votes) / len(votes) if votes else pd.Series(0.0, index=price.index)


def _momentum(price: pd.Series, windows: tuple[int, ...]) -> pd.Series:
    raw = sum((price / price.shift(w) - 1.0).shift(1) for w in windows) / float(len(windows))
    return raw.clip(-1.0, 1.0)


def _rolling_beta(asset_r: pd.Series, bench_r: pd.Series, window: int) -> pd.Series:
    cov = asset_r.rolling(window).cov(bench_r)
    var = bench_r.rolling(window).var().replace(0.0, np.nan)
    beta = cov / var
    return beta.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)


def fit_alpha_model(close: pd.DataFrame, qqq: pd.Series, cfg: Mahoraga10Config, train_start: pd.Timestamp, train_end: pd.Timestamp) -> Dict[str, Any]:
    qqq_train = qqq.loc[train_start:train_end].ffill()
    close_train = close.loc[train_start:train_end]

    raw_w = np.array(m6.fit_ic_weights(close_train, qqq_train, cfg, str(train_start.date()), str(train_end.date())), dtype=float)
    raw_prior = np.array([1/3, 1/3, 1/3], dtype=float)
    raw_w = cfg.raw_weight_shrink * raw_w + (1.0 - cfg.raw_weight_shrink) * raw_prior
    raw_w = np.clip(raw_w, 1e-6, None)
    raw_w = raw_w / raw_w.sum()

    qqq_r = qqq_train.pct_change().fillna(0.0)
    resid_close = pd.DataFrame(index=close_train.index, columns=close_train.columns, dtype=float)
    for t in close_train.columns:
        px = close_train[t].ffill()
        r = px.pct_change().fillna(0.0)
        beta = _rolling_beta(r, qqq_r, cfg.residual_beta_window)
        resid_r = (r - beta * qqq_r).clip(-0.25, 0.25)
        resid_close[t] = (1.0 + resid_r).cumprod()

    resid_tr = pd.DataFrame(index=resid_close.index, columns=resid_close.columns, dtype=float)
    resid_mo = pd.DataFrame(index=resid_close.index, columns=resid_close.columns, dtype=float)
    for t in resid_close.columns:
        px = resid_close[t].ffill()
        resid_tr[t] = _trend(px, cfg.residual_spans_short, cfg.residual_spans_long)
        resid_mo[t] = _momentum(px, cfg.residual_mom_windows)
    future = resid_close.pct_change().shift(-5)
    ic_vals = []
    for sig in [resid_tr, resid_mo]:
        cs = []
        for dt in sig.index:
            s = sig.loc[dt]
            f = future.loc[dt]
            ok = s.notna() & f.notna()
            if ok.sum() >= 5:
                cs.append(float(s[ok].corr(f[ok], method="spearman")))
        ic_vals.append(np.nanmean(cs) if len(cs) else 0.0)
    resid_w = np.array(ic_vals, dtype=float)
    resid_prior = np.array([0.5, 0.5], dtype=float)
    resid_w = cfg.resid_weight_shrink * resid_w + (1.0 - cfg.resid_weight_shrink) * resid_prior
    resid_w = np.clip(np.abs(resid_w), 1e-6, None)
    resid_w = resid_w / resid_w.sum()
    return {
        "raw_weights": raw_w,
        "resid_weights": resid_w,
    }


def build_alpha_components(close: pd.DataFrame, qqq: pd.Series, cfg: Mahoraga10Config) -> Dict[str, pd.DataFrame]:
    idx = close.index
    qqq_ = qqq.reindex(idx).ffill()
    qqq_r = qqq_.pct_change().fillna(0.0)

    raw_tr = pd.DataFrame(index=idx, columns=close.columns, dtype=float)
    raw_mo = pd.DataFrame(index=idx, columns=close.columns, dtype=float)
    raw_re = pd.DataFrame(index=idx, columns=close.columns, dtype=float)
    resid_tr = pd.DataFrame(index=idx, columns=close.columns, dtype=float)
    resid_mo = pd.DataFrame(index=idx, columns=close.columns, dtype=float)
    beta_df = pd.DataFrame(index=idx, columns=close.columns, dtype=float)

    for t in close.columns:
        px = close[t].reindex(idx).ffill()
        raw_tr[t] = _trend(px, cfg.spans_short, cfg.spans_long)
        raw_mo[t] = _momentum(px, cfg.mom_windows)
        raw_re[t] = _momentum(px, cfg.rel_windows) - _momentum(qqq_, cfg.rel_windows)

        r = px.pct_change().fillna(0.0)
        beta = _rolling_beta(r, qqq_r, cfg.residual_beta_window)
        beta_df[t] = beta
        resid_r = (r - beta * qqq_r).clip(-0.25, 0.25)
        resid_px = (1.0 + resid_r).cumprod()
        resid_tr[t] = _trend(resid_px, cfg.residual_spans_short, cfg.residual_spans_long)
        resid_mo[t] = _momentum(resid_px, cfg.residual_mom_windows)

    out = {
        "raw_tr": cross_sectional_z(raw_tr),
        "raw_mo": cross_sectional_z(raw_mo),
        "raw_re": cross_sectional_z(raw_re),
        "resid_tr": cross_sectional_z(resid_tr),
        "resid_mo": cross_sectional_z(resid_mo),
        "beta_df": beta_df.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0),
    }
    return out


def build_alpha_scores(components: Dict[str, pd.DataFrame], alpha_fit: Dict[str, Any], alpha_params: Dict[str, float], cfg: Mahoraga10Config) -> pd.DataFrame:
    rw = np.array(alpha_fit["raw_weights"], dtype=float)
    rw = np.array([rw[0], rw[1], rw[2] * float(alpha_params["raw_rel_boost"])], dtype=float)
    rw = np.clip(rw, 1e-6, None)
    rw = rw / rw.sum()
    uw = np.array(alpha_fit["resid_weights"], dtype=float)
    uw = np.clip(uw, 1e-6, None)
    uw = uw / uw.sum()

    raw = rw[0] * components["raw_tr"] + rw[1] * components["raw_mo"] + rw[2] * components["raw_re"]
    resid = uw[0] * components["resid_tr"] + uw[1] * components["resid_mo"]
    beta_pen = (components["beta_df"] - 1.0).clip(lower=0.0) * float(alpha_params["beta_penalty"])
    mix = float(alpha_params["alpha_mix_base"])
    score = (1.0 - mix) * raw + mix * resid - beta_pen
    score = cross_sectional_z(score).fillna(0.0)
    score.iloc[:max(cfg.burn_in, cfg.residual_burn_in)] = 0.0
    return score


def precompute_alpha_path(pre: Dict[str, Any], components: Dict[str, pd.DataFrame], alpha_fit: Dict[str, Any], alpha_params: Dict[str, float], cfg: Mahoraga10Config) -> Dict[str, Any]:
    idx = pre["idx"]
    close = pre["close"]
    high = pre["high"]
    low = pre["low"]
    rets = pre["rets"]
    score = build_alpha_scores(components, alpha_fit, alpha_params, cfg)

    w = pd.DataFrame(0.0, index=idx, columns=close.columns)
    last_w = pd.Series(0.0, index=close.columns)
    for dt in idx:
        if dt in pre["reb_dates"]:
            members = pre["members_at"](dt)
            members = [m for m in members if m in close.columns]
            if members:
                row = score.loc[dt, members]
                chosen = [n for n in row.nlargest(cfg.top_k).index.tolist() if row.get(n, 0.0) > 0.0]
                if len(chosen) == 1:
                    last_w = pd.Series(0.0, index=close.columns)
                    last_w[chosen[0]] = 1.0
                elif len(chosen) >= 2:
                    hist = rets.loc[:dt, chosen].tail(cfg.hrp_window).dropna(how="any")
                    if len(hist) < 60:
                        hist = rets.loc[:dt, chosen].dropna(how="any")
                    ww = m6.hrp_weights(hist).reindex(chosen, fill_value=0.0) if len(hist) else pd.Series(1.0 / len(chosen), index=chosen)
                    ww = ww.clip(upper=cfg.weight_cap)
                    ww = ww / ww.sum() if ww.sum() > 0 else pd.Series(1.0 / len(chosen), index=chosen)
                    last_w = pd.Series(0.0, index=close.columns)
                    last_w[chosen] = ww.reindex(chosen).values
                else:
                    last_w = pd.Series(0.0, index=close.columns)
            else:
                last_w = pd.Series(0.0, index=close.columns)
        w.loc[dt] = last_w.values

    w_stop, stop_hits = m6.apply_chandelier(w, close, high, low, cfg)
    w_exec_1x = w_stop.shift(1).fillna(0.0)
    gross_1x = (w_exec_1x * rets).sum(axis=1)
    turnover_1x = w_exec_1x.diff().abs().sum(axis=1).fillna(0.0)
    return {
        "alpha_params": alpha_params,
        "scores": score,
        "weights_target": w,
        "weights_after_stops": w_stop,
        "weights_exec_1x": w_exec_1x,
        "gross_1x": gross_1x,
        "turnover_1x": turnover_1x,
        "stop_hits": stop_hits,
    }
