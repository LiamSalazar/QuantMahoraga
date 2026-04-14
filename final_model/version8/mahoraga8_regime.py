from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from mahoraga8_config import Mahoraga8Config

FEATURE_COLS = [
    'stress_intensity', 'recovery_intensity', 'intensity_spread',
    'avg_corr_21', 'avg_corr_63', 'xs_disp_5d', 'xs_disp_21d',
    'breadth_63d', 'qqq_ret_5d', 'qqq_ret_21d', 'qqq_drawdown',
    'qqq_vol_21', 'vix_level', 'vix_z_63', 'trend_mean', 'mom_mean',
    'rel_mean', 'rel_top_mean', 'rel_minus_mom', 'rel_minus_trend',
    'base_exposure', 'base_turnover', 'crisis_state_weekly',
    'corr_state_weekly', 'turb_scale_weekly', 'crisis_scale_weekly',
    'corr_scale_weekly', 'base_total_scale',
]


def _build_base_weekly_state(base_bt: Dict[str, Any], decision_freq: str) -> pd.DataFrame:
    idx = base_bt['returns_net'].index.to_series().resample(decision_freq).last().dropna().index
    keys = [
        ('exposure', 'base_exposure', 0.0), ('turnover', 'base_turnover', 0.0),
        ('crisis_state', 'crisis_state_weekly', 0.0), ('corr_state', 'corr_state_weekly', 0.0),
        ('turb_scale', 'turb_scale_weekly', 1.0), ('crisis_scale', 'crisis_scale_weekly', 1.0),
        ('corr_scale', 'corr_scale_weekly', 1.0), ('total_scale_target', 'base_total_scale', 1.0),
    ]
    out = pd.DataFrame(index=idx)
    for src_key, col_name, fill in keys:
        s = base_bt.get(src_key, pd.Series(dtype=float))
        out[col_name] = s.resample(decision_freq).last().reindex(idx).ffill().fillna(fill)
    return out


def _safe_z_on_train(s: pd.Series, train_idx: pd.DatetimeIndex) -> pd.Series:
    tr = s.loc[train_idx].dropna()
    mu = float(tr.mean()) if len(tr) else 0.0
    sd = float(tr.std(ddof=1)) if len(tr) > 1 else 1.0
    if not np.isfinite(sd) or sd <= 1e-8:
        sd = 1.0
    return ((s - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _student_pdf_fallback(x: float, mu: np.ndarray, kappa: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.maximum(beta * (kappa + 1.0) / np.maximum(alpha * kappa, 1e-8), 1e-8))
    z = (x - mu) / np.maximum(scale, 1e-8)
    return np.exp(-0.5 * z * z) / np.maximum(scale, 1e-8)


def _build_regime_score(weekly_df: pd.DataFrame, train_idx: pd.DatetimeIndex) -> pd.Series:
    stress_z = _safe_z_on_train(weekly_df['stress_intensity'], train_idx)
    rec_z = _safe_z_on_train(weekly_df['recovery_intensity'], train_idx)
    corr_z = _safe_z_on_train(weekly_df['avg_corr_21'], train_idx)
    vix_z = _safe_z_on_train(weekly_df['vix_level'], train_idx)
    dd_z = _safe_z_on_train(-weekly_df['qqq_drawdown'], train_idx)
    breadth_stress = _safe_z_on_train(-weekly_df['breadth_63d'], train_idx)
    score = (0.32 * stress_z + 0.24 * corr_z + 0.18 * vix_z + 0.16 * dd_z + 0.08 * breadth_stress - 0.14 * rec_z)
    return score.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _bocpd_lite_regime_features(weekly_df: pd.DataFrame, train_idx: pd.DatetimeIndex, cfg: Mahoraga8Config) -> pd.DataFrame:
    regime_score = _build_regime_score(weekly_df, train_idx)
    x = regime_score.to_numpy(dtype=float)
    max_r = max(8, int(cfg.cp_max_run_length))
    H = float(np.clip(cfg.cp_hazard, 1e-4, 0.5))
    train_x = regime_score.loc[train_idx].dropna()
    mu0 = float(train_x.mean()) if len(train_x) else 0.0
    var0 = float(train_x.var(ddof=1)) if len(train_x) > 1 else 1.0
    if not np.isfinite(var0) or var0 <= 1e-8:
        var0 = 1.0
    R = np.array([1.0], dtype=float)
    mu = np.array([mu0], dtype=float)
    kappa = np.array([max(cfg.cp_prior_kappa, 1e-3)], dtype=float)
    alpha = np.array([max(cfg.cp_prior_alpha, 1e-3)], dtype=float)
    beta = np.array([max(cfg.cp_prior_beta * var0, 1e-3)], dtype=float)
    cp_prob = np.zeros(len(x), dtype=float)
    cp_sev = np.zeros(len(x), dtype=float)
    pred_mean_prev = mu0
    pred_std0 = float(np.sqrt(var0))
    for t, xt in enumerate(x):
        pred = _student_pdf_fallback(float(xt), mu, kappa, alpha, beta)
        growth = pred * R * (1.0 - H)
        cp_mass = float((pred * R).sum() * H)
        new_len = min(len(growth) + 1, max_r + 1)
        R_new = np.empty(new_len, dtype=float)
        R_new[0] = cp_mass
        g_take = min(len(growth), max_r)
        R_new[1:g_take + 1] = growth[:g_take]
        z = R_new.sum()
        if not np.isfinite(z) or z <= 0:
            R_new[:] = 0.0
            R_new[0] = 1.0
        else:
            R_new /= z
        cp_prob[t] = float(R_new[0])
        cp_sev[t] = abs((xt - pred_mean_prev) / max(pred_std0, 1e-8))
        keep = min(len(mu), max_r)
        mu_new = np.empty(new_len, dtype=float)
        kappa_new = np.empty(new_len, dtype=float)
        alpha_new = np.empty(new_len, dtype=float)
        beta_new = np.empty(new_len, dtype=float)
        mu_new[0] = mu0
        kappa_new[0] = max(cfg.cp_prior_kappa, 1e-3)
        alpha_new[0] = max(cfg.cp_prior_alpha, 1e-3)
        beta_new[0] = max(cfg.cp_prior_beta * var0, 1e-3)
        if keep > 0:
            kk = kappa[:keep]
            mm = mu[:keep]
            aa = alpha[:keep]
            bb = beta[:keep]
            kk1 = kk + 1.0
            diff = xt - mm
            mu_new[1:keep + 1] = (kk * mm + xt) / kk1
            kappa_new[1:keep + 1] = kk1
            alpha_new[1:keep + 1] = aa + 0.5
            beta_new[1:keep + 1] = bb + (kk * diff * diff) / (2.0 * kk1)
        pred_mean_prev = float(np.dot(R_new[:len(mu_new)], mu_new[:len(R_new)]))
        R, mu, kappa, alpha, beta = R_new, mu_new, kappa_new, alpha_new, beta_new
    delta = regime_score.diff().fillna(0.0)
    direction = np.where((delta <= 0) | (weekly_df['recovery_intensity'] > weekly_df['stress_intensity']), 1.0, -1.0)
    return pd.DataFrame({'regime_score': regime_score, 'cp_prob': cp_prob, 'cp_severity': cp_sev, 'cp_direction': direction}, index=weekly_df.index)


def _future_window_return(returns_daily: pd.Series, weekly_idx: pd.DatetimeIndex, horizon_weeks: int) -> pd.Series:
    idx = returns_daily.index
    out = pd.Series(np.nan, index=weekly_idx, dtype=float)
    if len(idx) == 0:
        return out
    step = max(1, horizon_weeks * 5)
    r_arr = returns_daily.to_numpy(dtype=float)
    pos_arr = idx.searchsorted(weekly_idx.values, side='left')
    n = len(idx)
    log1r = np.log1p(np.clip(r_arr, -0.9999, None))
    cumlog = np.concatenate([[0.0], np.cumsum(log1r)])
    vals = np.empty(len(weekly_idx), dtype=float)
    vals[:] = np.nan
    for i, pos in enumerate(pos_arr):
        if pos >= n:
            continue
        end_i = min(n - 1, pos + step)
        if end_i <= pos:
            continue
        vals[i] = np.expm1(cumlog[end_i + 1] - cumlog[pos + 1])
    return pd.Series(vals, index=weekly_idx, dtype=float)


def _ridge_split_conformal_risk(weekly_df: pd.DataFrame, returns_daily: pd.Series, train_idx: pd.DatetimeIndex, cfg: Mahoraga8Config) -> pd.DataFrame:
    feat_cols = ['regime_score', 'cp_prob', 'cp_severity', 'stress_intensity', 'recovery_intensity', 'avg_corr_21', 'vix_level', 'qqq_drawdown', 'qqq_vol_21', 'base_total_scale', 'breadth_63d']
    target = (-_future_window_return(returns_daily, weekly_df.index, cfg.conformal_horizon_weeks)).clip(lower=0.0)
    X = weekly_df.reindex(columns=feat_cols).copy().replace([np.inf, -np.inf], np.nan)
    med = X.loc[train_idx].median(numeric_only=True)
    X = X.fillna(med).fillna(0.0)
    mu = X.loc[train_idx].mean(numeric_only=True)
    sd = X.loc[train_idx].std(ddof=1, numeric_only=True).replace(0.0, 1.0).fillna(1.0)
    Xs = (X - mu) / sd
    mask = target.loc[train_idx].notna()
    if int(mask.sum()) < max(24, cfg.min_train_weeks // 2):
        upper = pd.Series(0.0, index=weekly_df.index)
        risk_budget = pd.Series(1.0, index=weekly_df.index)
        return pd.DataFrame({'conf_pred_loss': 0.0, 'conf_upper_loss': upper, 'risk_budget': risk_budget}, index=weekly_df.index)
    Xtr = Xs.loc[train_idx].loc[mask].to_numpy(dtype=float)
    ytr = target.loc[train_idx].loc[mask].to_numpy(dtype=float)
    Xtr_i = np.column_stack([np.ones(len(Xtr)), Xtr])
    ridge = np.eye(Xtr_i.shape[1]) * float(cfg.conformal_l2)
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(Xtr_i.T @ Xtr_i + ridge, Xtr_i.T @ ytr)
    Xall = np.column_stack([np.ones(len(Xs)), Xs.to_numpy(dtype=float)])
    pred = Xall @ beta
    resid = ytr - (Xtr_i @ beta)
    q = float(np.nanquantile(resid, cfg.conformal_alpha)) if len(resid) else 0.0
    q = max(0.0, q)
    upper = np.clip(pred + q, 0.0, None)
    upper_train = pd.Series(upper, index=weekly_df.index).loc[train_idx].dropna()
    lo = float(np.nanquantile(upper_train, cfg.conformal_budget_low_q)) if len(upper_train) else 0.0
    hi = float(np.nanquantile(upper_train, cfg.conformal_budget_high_q)) if len(upper_train) else max(lo + 1e-6, 1.0)
    if hi <= lo:
        hi = lo + 1e-6
    interp = 1.0 - (upper - lo) / (hi - lo)
    risk_budget = cfg.conformal_min_exposure + (1.0 - cfg.conformal_min_exposure) * np.clip(interp, 0.0, 1.0)
    risk_budget = np.clip(risk_budget, cfg.conformal_min_exposure, 1.0)
    return pd.DataFrame({'conf_pred_loss': pred, 'conf_upper_loss': upper, 'risk_budget': risk_budget}, index=weekly_df.index)


def _enforce_min_persistence(raw_state: pd.Series, min_counts: Dict[str, int]) -> pd.Series:
    vals = raw_state.astype(str).tolist()
    if not vals:
        return raw_state
    out = vals[:]
    run_start = 0
    while run_start < len(vals):
        run_end = run_start
        while run_end + 1 < len(vals) and vals[run_end + 1] == vals[run_start]:
            run_end += 1
        state = vals[run_start]
        run_len = run_end - run_start + 1
        need = int(min_counts.get(state, 1))
        if run_len < need and run_start > 0:
            prev_state = out[run_start - 1]
            for i in range(run_start, run_end + 1):
                out[i] = prev_state
        run_start = run_end + 1
    return pd.Series(out, index=raw_state.index)


def _classify_regime_state(regime_df: pd.DataFrame, train_idx: pd.DatetimeIndex, cfg: Mahoraga8Config) -> pd.DataFrame:
    train = regime_df.loc[train_idx].copy()
    cp_prob_thr = float(np.nanquantile(train['cp_prob'], cfg.cp_prob_quantile)) if len(train) else 1.0
    cp_sev_thr = float(np.nanquantile(train['cp_severity'], cfg.cp_severity_quantile)) if len(train) else np.inf
    stress_in = float(np.nanquantile(train['regime_score'], cfg.stress_entry_quantile)) if len(train) else 0.5
    stress_out = float(np.nanquantile(train['regime_score'], cfg.stress_exit_quantile)) if len(train) else 0.2
    panic_in = float(np.nanquantile(train['regime_score'], cfg.panic_entry_quantile)) if len(train) else 1.0
    panic_out = float(np.nanquantile(train['regime_score'], cfg.panic_exit_quantile)) if len(train) else 0.7
    rec_q = float(np.nanquantile(train['regime_score'], cfg.recovery_entry_quantile)) if len(train) else 0.0

    rs = regime_df['regime_score'].fillna(0.0)
    cp = regime_df['cp_prob'].fillna(0.0)
    sev = regime_df['cp_severity'].fillna(0.0)
    direction = regime_df['cp_direction'].fillna(0.0)
    rb = regime_df['risk_budget'].fillna(1.0)

    raw = []
    prev = 'NORMAL'
    for i in range(len(regime_df)):
        cur = prev
        rsi = float(rs.iloc[i]); cpi = float(cp.iloc[i]); sevi = float(sev.iloc[i]); di = float(direction.iloc[i]); rbi = float(rb.iloc[i])
        if prev == 'PANIC':
            if (rsi >= panic_out) or ((cpi >= cp_prob_thr) and (di < 0)):
                cur = 'PANIC'
            elif (rsi >= stress_out):
                cur = 'STRESS'
            else:
                cur = 'NORMAL'
        elif prev == 'STRESS':
            if (rsi >= panic_in) or ((cpi >= cp_prob_thr) and (sevi >= cp_sev_thr) and (di < 0) and (rbi <= 0.45)):
                cur = 'PANIC'
            elif (rsi >= stress_out) or ((di < 0) and (cpi >= cp_prob_thr)):
                cur = 'STRESS'
            elif (di > 0) and (rsi <= rec_q) and (rbi >= 0.55):
                cur = 'RECOVERY'
            else:
                cur = 'NORMAL'
        elif prev == 'RECOVERY':
            if (rsi >= panic_in) and (di < 0):
                cur = 'PANIC'
            elif (rsi >= stress_in) and (di <= 0):
                cur = 'STRESS'
            elif (di > 0) or (rsi <= stress_out):
                cur = 'RECOVERY'
            else:
                cur = 'NORMAL'
        else:
            if (rsi >= panic_in) and ((cpi >= cp_prob_thr) or (rbi <= 0.40)):
                cur = 'PANIC'
            elif (rsi >= stress_in) or ((di < 0) and (cpi >= cp_prob_thr)):
                cur = 'STRESS'
            elif (di > 0) and (rsi <= rec_q) and (rbi >= 0.60):
                cur = 'RECOVERY'
            else:
                cur = 'NORMAL'
        raw.append(cur)
        prev = cur

    raw_s = pd.Series(raw, index=regime_df.index)
    raw_s = _enforce_min_persistence(raw_s, {
        'STRESS': cfg.stress_min_persistence,
        'PANIC': cfg.panic_min_persistence,
        'RECOVERY': cfg.recovery_min_persistence,
        'NORMAL': 1,
    })

    conf = np.where(raw_s.values == 'PANIC', np.maximum(cp.to_numpy(), sev.to_numpy() / max(cp_sev_thr, 1e-6)),
                    np.where(raw_s.values == 'STRESS', np.maximum(cp.to_numpy(), np.clip(rs.to_numpy() / max(stress_in, 1e-6), 0.0, 1.5)),
                             np.where(raw_s.values == 'RECOVERY', np.maximum(cp.to_numpy(), np.clip(1.0 - rb.to_numpy(), 0.0, 1.0)), 1.0 - np.clip(cp.to_numpy(), 0.0, 1.0))))
    out = regime_df.copy()
    out['regime_state'] = raw_s.ffill().bfill()
    out['regime_confidence'] = pd.Series(conf, index=regime_df.index).clip(0.0, 2.0).fillna(0.0)
    return out


def build_regime_table(hawkes_df: pd.DataFrame, base_bt: Dict[str, Any], returns_daily: pd.Series, train_idx: pd.DatetimeIndex, cfg: Mahoraga8Config) -> pd.DataFrame:
    wk = _build_base_weekly_state(base_bt, cfg.decision_freq)
    feat = hawkes_df.join(wk, how='left').sort_index()
    feat = feat.reindex(columns=list(dict.fromkeys(FEATURE_COLS))).copy()
    for c in feat.columns:
        if feat[c].dtype.kind in 'biufc':
            feat[c] = feat[c].replace([np.inf, -np.inf], np.nan)
    regime_df = _bocpd_lite_regime_features(feat, train_idx, cfg)
    feat = feat.join(regime_df, how='left')
    conf = _ridge_split_conformal_risk(feat, returns_daily, train_idx, cfg)
    feat = feat.join(conf, how='left')
    feat = _classify_regime_state(feat, train_idx, cfg)
    return feat
