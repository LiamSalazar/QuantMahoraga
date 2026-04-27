from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

import mahoraga6_1 as m6
from mahoraga14_config import Mahoraga14Config
from mahoraga14_utils import cross_sectional_z


def _trend(price: pd.Series, short_spans: tuple[int, ...], long_spans: tuple[int, ...]) -> pd.Series:
    votes = []
    for sp in short_spans:
        for lp in long_spans:
            if sp >= lp:
                continue
            short_ma = price.ewm(span=sp, adjust=False).mean().shift(1)
            long_ma = price.ewm(span=lp, adjust=False).mean().shift(1)
            votes.append(((short_ma > long_ma) & price.notna()).astype(float))
    return sum(votes) / len(votes) if votes else pd.Series(0.0, index=price.index)


def _momentum(price: pd.Series, windows: tuple[int, ...]) -> pd.Series:
    raw = sum((price / price.shift(w) - 1.0).shift(1) for w in windows) / float(len(windows))
    return raw.clip(-1.0, 1.0)


def _rolling_beta(asset_r: pd.Series, bench_r: pd.Series, window: int) -> pd.Series:
    cov = asset_r.rolling(window).cov(bench_r)
    var = bench_r.rolling(window).var().replace(0.0, np.nan)
    beta = cov / var
    return beta.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)


def _pc1_weights(train_rets: pd.DataFrame) -> pd.Series:
    cols = list(train_rets.columns)
    if not cols:
        return pd.Series(dtype=float)
    valid = train_rets.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    keep = valid.std(ddof=1).replace(0.0, np.nan).dropna().index.tolist()
    if len(keep) < 2:
        return pd.Series(1.0 / max(1, len(cols)), index=cols, dtype=float)

    z = valid[keep].copy()
    z = z.sub(z.mean(axis=0), axis=1)
    z = z.div(z.std(axis=0, ddof=1).replace(0.0, np.nan), axis=1).fillna(0.0)
    cov = np.cov(z.values, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    vec = evecs[:, int(np.argmax(evals))]
    raw = pd.Series(vec, index=keep, dtype=float)
    market_proxy = valid[keep].mean(axis=1)
    if raw.dot(valid[keep].mean(axis=0)) < 0.0 or valid[keep].mul(raw, axis=1).sum(axis=1).corr(market_proxy) < 0.0:
        raw = -raw
    denom = float(raw.abs().sum())
    if denom <= 0.0 or not np.isfinite(denom):
        raw = pd.Series(1.0 / len(keep), index=keep, dtype=float)
    else:
        raw = raw / denom
    out = pd.Series(0.0, index=cols, dtype=float)
    out.loc[keep] = raw
    return out


def _build_factor_frame(
    close: pd.DataFrame,
    qqq: pd.Series,
    spy: pd.Series,
    factor_spec: Dict[str, Any],
) -> pd.DataFrame:
    idx = close.index
    rets = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    qqq_r = qqq.reindex(idx).ffill().pct_change().fillna(0.0)
    spy_r = spy.reindex(idx).ffill().pct_change().fillna(0.0)
    tech_r = rets.mean(axis=1).fillna(0.0)
    pc1_w = pd.Series(factor_spec.get("pc1_weights", pd.Series(0.0, index=close.columns)), dtype=float).reindex(close.columns).fillna(0.0)
    pc1_r = rets.mul(pc1_w, axis=1).sum(axis=1).fillna(0.0)
    return pd.DataFrame(
        {
            "QQQ": qqq_r,
            "SPY": spy_r,
            "TECH": tech_r,
            "PC1": pc1_r,
        },
        index=idx,
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _fit_ridge(x_hist: np.ndarray, y_hist: np.ndarray, ridge_alpha: float) -> tuple[float, np.ndarray]:
    x_mean = np.nanmean(x_hist, axis=0)
    y_mean = float(np.nanmean(y_hist))
    x_c = np.nan_to_num(x_hist - x_mean, nan=0.0)
    y_c = np.nan_to_num(y_hist - y_mean, nan=0.0)
    gram = x_c.T @ x_c + ridge_alpha * np.eye(x_c.shape[1], dtype=float)
    rhs = x_c.T @ y_c
    try:
        beta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(gram) @ rhs
    intercept = float(y_mean - x_mean @ beta)
    return intercept, beta


def _rolling_multifactor_residual(
    asset_r: pd.Series,
    factor_df: pd.DataFrame,
    cfg: Mahoraga14Config,
) -> pd.Series:
    idx = asset_r.index
    y = pd.Series(asset_r, index=idx, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = factor_df.reindex(idx).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = pd.Series(np.nan, index=idx, dtype=float)
    window = int(cfg.residual_beta_window)
    min_obs = int(min(window, max(12, cfg.residual_beta_min_obs)))
    x_vals = x.to_numpy(dtype=float)
    y_vals = y.to_numpy(dtype=float)

    for i in range(len(idx)):
        start = max(0, i - window)
        hist_len = i - start
        if hist_len < min_obs:
            continue
        intercept, beta = _fit_ridge(x_vals[start:i], y_vals[start:i], float(cfg.residual_ridge_alpha))
        pred = intercept + float(x_vals[i] @ beta)
        out.iloc[i] = float(np.clip(y_vals[i] - pred, -cfg.residual_return_clip, cfg.residual_return_clip))

    fallback = y.clip(-cfg.residual_return_clip, cfg.residual_return_clip)
    return out.fillna(fallback).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def fit_base_alpha_model(
    close: pd.DataFrame,
    qqq: pd.Series,
    spy: pd.Series,
    cfg: Mahoraga14Config,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> Dict[str, Any]:
    qqq_train = qqq.loc[train_start:train_end].ffill()
    spy_train = spy.loc[train_start:train_end].ffill()
    close_train = close.loc[train_start:train_end]

    raw_w = np.array(
        m6.fit_ic_weights(close_train, qqq_train, cfg, str(train_start.date()), str(train_end.date())),
        dtype=float,
    )
    raw_prior = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    raw_w = cfg.raw_weight_shrink * raw_w + (1.0 - cfg.raw_weight_shrink) * raw_prior
    raw_w = np.clip(raw_w, 1e-6, None)
    raw_w = raw_w / raw_w.sum()

    train_rets = close_train.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    factor_spec = {
        "pc1_weights": _pc1_weights(train_rets),
        "factor_names": ("QQQ", "SPY", "TECH", "PC1"),
    }
    factor_df = _build_factor_frame(close_train, qqq_train, spy_train, factor_spec)

    resid_close = pd.DataFrame(index=close_train.index, columns=close_train.columns, dtype=float)
    for ticker in close_train.columns:
        px = close_train[ticker].ffill()
        asset_r = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        resid_r = _rolling_multifactor_residual(asset_r, factor_df, cfg)
        resid_close[ticker] = (1.0 + resid_r).cumprod()

    resid_tr = pd.DataFrame(index=resid_close.index, columns=resid_close.columns, dtype=float)
    resid_mo = pd.DataFrame(index=resid_close.index, columns=resid_close.columns, dtype=float)
    for ticker in resid_close.columns:
        px = resid_close[ticker].ffill()
        resid_tr[ticker] = _trend(px, cfg.residual_spans_short, cfg.residual_spans_long)
        resid_mo[ticker] = _momentum(px, cfg.residual_mom_windows)

    future = resid_close.pct_change().shift(-5)
    ic_vals = []
    for sig in [resid_tr, resid_mo]:
        cs_vals = []
        for dt in sig.index:
            sig_row = sig.loc[dt]
            fut_row = future.loc[dt]
            ok = sig_row.notna() & fut_row.notna()
            if ok.sum() >= 5:
                cs_vals.append(float(sig_row[ok].corr(fut_row[ok], method="spearman")))
        ic_vals.append(np.nanmean(cs_vals) if cs_vals else 0.0)

    resid_w = np.array(ic_vals, dtype=float)
    resid_prior = np.array([0.5, 0.5], dtype=float)
    resid_w = cfg.resid_weight_shrink * resid_w + (1.0 - cfg.resid_weight_shrink) * resid_prior
    resid_w = np.clip(np.abs(resid_w), 1e-6, None)
    resid_w = resid_w / resid_w.sum()
    return {
        "raw_weights": raw_w,
        "resid_weights": resid_w,
        "factor_spec": factor_spec,
    }


def build_global_alpha_components(
    close: pd.DataFrame,
    qqq: pd.Series,
    spy: pd.Series,
    factor_spec: Dict[str, Any],
    cfg: Mahoraga14Config,
) -> Dict[str, pd.DataFrame]:
    idx = close.index
    qqq_ = qqq.reindex(idx).ffill()
    qqq_r = qqq_.pct_change().fillna(0.0)
    factor_df = _build_factor_frame(close, qqq, spy, factor_spec)

    raw_tr = pd.DataFrame(index=idx, columns=close.columns, dtype=float)
    raw_mo = pd.DataFrame(index=idx, columns=close.columns, dtype=float)
    raw_re = pd.DataFrame(index=idx, columns=close.columns, dtype=float)
    resid_tr = pd.DataFrame(index=idx, columns=close.columns, dtype=float)
    resid_mo = pd.DataFrame(index=idx, columns=close.columns, dtype=float)
    beta_df = pd.DataFrame(index=idx, columns=close.columns, dtype=float)

    for ticker in close.columns:
        px = close[ticker].reindex(idx).ffill()
        raw_tr[ticker] = _trend(px, cfg.spans_short, cfg.spans_long)
        raw_mo[ticker] = _momentum(px, cfg.mom_windows)
        raw_re[ticker] = _momentum(px, cfg.rel_windows) - _momentum(qqq_, cfg.rel_windows)

        asset_r = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        beta_df[ticker] = _rolling_beta(asset_r, qqq_r, cfg.residual_beta_window)
        resid_r = _rolling_multifactor_residual(asset_r, factor_df, cfg)
        resid_px = (1.0 + resid_r).cumprod()
        resid_tr[ticker] = _trend(resid_px, cfg.residual_spans_short, cfg.residual_spans_long)
        resid_mo[ticker] = _momentum(resid_px, cfg.residual_mom_windows)

    return {
        "legacy_tr": raw_tr.fillna(0.0),
        "legacy_mo": raw_mo.fillna(0.0),
        "legacy_re": raw_re.fillna(0.0),
        "raw_tr": cross_sectional_z(raw_tr),
        "raw_mo": cross_sectional_z(raw_mo),
        "raw_re": cross_sectional_z(raw_re),
        "resid_tr": cross_sectional_z(resid_tr),
        "resid_mo": cross_sectional_z(resid_mo),
        "beta_df": beta_df.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0),
        "factor_frame": factor_df,
    }


def slice_alpha_components(components: Dict[str, pd.DataFrame], idx: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
    return {key: value.loc[idx] for key, value in components.items()}


def build_engine_score(
    components: Dict[str, pd.DataFrame],
    alpha_fit: Dict[str, Any],
    mix: float,
    beta_penalty: float,
    raw_rel_boost: float,
    cfg: Mahoraga14Config,
) -> pd.DataFrame:
    legacy_core = cross_sectional_z(
        cfg.w_trend * components["legacy_tr"] + cfg.w_mom * components["legacy_mo"] + cfg.w_rel * components["legacy_re"]
    )

    raw_w = np.array(alpha_fit["raw_weights"], dtype=float)
    raw_w = np.array([raw_w[0], raw_w[1], raw_w[2] * raw_rel_boost], dtype=float)
    raw_w = np.clip(raw_w, 1e-6, None)
    raw_w = raw_w / raw_w.sum()

    resid_w = np.array(alpha_fit["resid_weights"], dtype=float)
    resid_w = np.clip(resid_w, 1e-6, None)
    resid_w = resid_w / resid_w.sum()

    raw_score = raw_w[0] * components["raw_tr"] + raw_w[1] * components["raw_mo"] + raw_w[2] * components["raw_re"]
    resid_score = resid_w[0] * components["resid_tr"] + resid_w[1] * components["resid_mo"]
    beta_drag = (components["beta_df"] - 1.0).clip(lower=0.0) * beta_penalty
    adaptive_score = 0.62 * raw_score + 0.38 * resid_score - beta_drag
    score = (1.0 - mix) * legacy_core + mix * adaptive_score
    score = cross_sectional_z(score).fillna(0.0)
    score.iloc[: max(cfg.burn_in, cfg.residual_burn_in)] = 0.0
    return score


def _build_target_weights(score: pd.DataFrame, pre: Dict[str, Any], cfg: Mahoraga14Config) -> pd.DataFrame:
    idx = pre["idx"]
    close = pre["close"]
    rets = pre["rets"]

    weights = pd.DataFrame(0.0, index=idx, columns=close.columns)
    last_w = pd.Series(0.0, index=close.columns)
    for dt in idx:
        if dt in pre["reb_dates"]:
            members = [m for m in pre["members_at"](dt) if m in close.columns]
            if members:
                row = score.loc[dt, members]
                chosen = [name for name in row.nlargest(cfg.top_k).index.tolist() if row.get(name, 0.0) > 0.0]
                if len(chosen) == 1:
                    last_w = pd.Series(0.0, index=close.columns)
                    last_w[chosen[0]] = 1.0
                elif len(chosen) >= 2:
                    hist = rets.loc[:dt, chosen].tail(cfg.hrp_window).dropna(how="any")
                    if len(hist) < 60:
                        hist = rets.loc[:dt, chosen].dropna(how="any")
                    if len(hist):
                        w_hrp = m6.hrp_weights(hist).reindex(chosen, fill_value=0.0)
                    else:
                        w_hrp = pd.Series(1.0 / len(chosen), index=chosen)
                    w_hrp = w_hrp.clip(upper=cfg.weight_cap)
                    w_hrp = w_hrp / w_hrp.sum() if w_hrp.sum() > 0 else pd.Series(1.0 / len(chosen), index=chosen)
                    last_w = pd.Series(0.0, index=close.columns)
                    last_w[chosen] = w_hrp.reindex(chosen).values
                else:
                    last_w = pd.Series(0.0, index=close.columns)
            else:
                last_w = pd.Series(0.0, index=close.columns)
        weights.loc[dt] = last_w.values
    return weights


def precompute_engine_path(
    pre: Dict[str, Any],
    components: Dict[str, pd.DataFrame],
    alpha_fit: Dict[str, Any],
    mix: float,
    beta_penalty: float,
    raw_rel_boost: float,
    cfg: Mahoraga14Config,
) -> Dict[str, Any]:
    close = pre["close"]
    high = pre["high"]
    low = pre["low"]
    rets = pre["rets"]

    score = build_engine_score(components, alpha_fit, mix, beta_penalty, raw_rel_boost, cfg)
    target_weights = _build_target_weights(score, pre, cfg)
    weights_after_stops, _ = m6.apply_chandelier(target_weights, close, high, low, cfg)
    weights_exec_1x = weights_after_stops.shift(1).fillna(0.0)
    gross_1x = (weights_exec_1x * rets).sum(axis=1)
    turnover_1x = weights_exec_1x.diff().abs().sum(axis=1).fillna(0.0)
    stop_active = ((target_weights > 0.0) & (weights_after_stops == 0.0)).astype(float)
    stop_new = stop_active.diff().clip(lower=0.0).fillna(0.0)
    return {
        "scores": score,
        "weights_target": target_weights,
        "weights_after_stops": weights_after_stops,
        "weights_exec_1x": weights_exec_1x,
        "gross_1x": gross_1x,
        "turnover_1x": turnover_1x,
        "stop_active_share": stop_active.mean(axis=1),
        "new_stop_share": stop_new.mean(axis=1),
    }


def blend_engine_paths(
    primary_cache: Dict[str, Any],
    defense_cache: Dict[str, Any],
    defense_blend: pd.Series,
) -> pd.DataFrame:
    blend = defense_blend.reindex(primary_cache["weights_exec_1x"].index).fillna(0.0).clip(0.0, 1.0)
    return primary_cache["weights_exec_1x"].mul(1.0 - blend, axis=0) + defense_cache["weights_exec_1x"].mul(blend, axis=0)


def backtest_from_1x_weights(
    pre: Dict[str, Any],
    weights_exec_1x: pd.DataFrame,
    gate_scale: pd.Series,
    vol_mult: pd.Series,
    exp_cap: pd.Series,
    cfg: Mahoraga14Config,
    costs: m6.CostsConfig,
    label: str,
) -> Dict[str, Any]:
    idx = pre["idx"]
    rets = pre["rets"]
    weights_exec_1x = weights_exec_1x.reindex(idx).fillna(0.0)

    gross_1x = (weights_exec_1x * rets).sum(axis=1)
    realized = gross_1x.rolling(cfg.port_vol_window).std(ddof=1) * np.sqrt(cfg.trading_days)
    target_vol = cfg.vol_target_ann * vol_mult.reindex(idx).ffill().bfill().fillna(1.0)
    vol_scale = (target_vol / realized.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_scale = vol_scale.clip(cfg.min_exposure, cfg.max_exposure)

    corr_secondary = pd.Series(1.0, index=idx, dtype=float)
    if cfg.use_corr_as_secondary_veto:
        corr_hit = pre["corr_rho"].reindex(idx).fillna(0.0) >= cfg.corr_secondary_rho
        corr_secondary.loc[corr_hit] = cfg.corr_secondary_scale

    cap = (
        pre["crisis_scale"].reindex(idx).fillna(1.0)
        * pre["turb_scale"].reindex(idx).fillna(1.0)
        * corr_secondary
        * gate_scale.reindex(idx).ffill().bfill().fillna(1.0)
        * exp_cap.reindex(idx).ffill().bfill().fillna(1.0)
    ).clip(0.0, cfg.max_exposure)
    total_scale_target = pd.Series(np.minimum(vol_scale.values, cap.values), index=idx).clip(0.0, cfg.max_exposure)
    total_scale = total_scale_target.shift(1).fillna(0.0)
    weights_scaled = weights_exec_1x.mul(total_scale, axis=0)

    turnover, tc = m6._costs(weights_scaled, costs)
    port_net = ((weights_scaled * rets).sum(axis=1) - tc).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    equity = cfg.capital_initial * (1.0 + port_net).cumprod()
    exposure = weights_scaled.abs().sum(axis=1).clip(0.0, cfg.max_exposure)

    qqq_r = pre["qqq"].pct_change().fillna(0.0) - costs.qqq_expense_ratio
    spy_r = pre["spy"].pct_change().fillna(0.0)
    qqq_eq = cfg.capital_initial * (1.0 + qqq_r).cumprod()
    spy_eq = cfg.capital_initial * (1.0 + spy_r).cumprod()

    return {
        "label": label,
        "returns_net": port_net,
        "equity": equity,
        "exposure": exposure,
        "turnover": turnover,
        "weights_scaled": weights_scaled,
        "total_scale": total_scale,
        "total_scale_target": total_scale_target,
        "cap": cap,
        "vol_scale": vol_scale,
        "crisis_scale": pre["crisis_scale"],
        "turb_scale": pre["turb_scale"],
        "corr_rho": pre["corr_rho"],
        "corr_state": pre["corr_state"],
        "bench": {
            "QQQ_r": qqq_r,
            "QQQ_eq": qqq_eq,
            "SPY_r": spy_r,
            "SPY_eq": spy_eq,
        },
    }
