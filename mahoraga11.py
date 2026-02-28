"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        MAHORAGA  1.1                                         ║
║   Long-Only Weekly-Rebalanced Tech Rotation — Research Edition               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Mahoraga 1.1 implements a long-only, weekly-rebalanced tech rotation        ║
║  strategy with no-look-ahead execution, shrinkage-based HRP allocation,      ║
║  ATR exits, a crisis overlay, turbulence filtering, and validation through   ║
║  expanding-window walk-forward splits, cost/gap stress tests, local          ║
║  parameter sensitivity maps, alternate-universe evaluation, and              ║
║  multiple-testing-aware reporting.                                            ║
║                                                                              ║
║  Changes from 1.0 → 1.1                                                      ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  [INTEGRITY]                                                                  ║
║  · Honest header — only claims what it delivers                              ║
║  · Crisis gate thresholds calibrated on TRAIN data only (percentile-based)  ║
║  · VIX downloaded directly (^VIX); proxy fallback clearly labelled          ║
║  · Sharpe CI renamed "Asymptotic Sharpe CI" (removes false JK claim)        ║
║  · IC evaluated at 1-day, 5-day AND 21-day horizons                         ║
║                                                                              ║
║  [VALIDATION]                                                                 ║
║  · Expanding-window walk-forward: 5 folds, step=1y, embargo=252bd           ║
║  · Stitched OOS equity curve from 5 independent test years                  ║
║  · Sweep uses small fixed grid (72 combos) on EACH fold's val period        ║
║  · Best config selected per fold; final config = modal choice                ║
║  · BHY/FDR q-values reported; configs labelled when statistically weak      ║
║                                                                              ║
║  [ROBUSTNESS — Sec 3]                                                         ║
║  · Cost/gap stress: base / ×2 / ×3 costs / light gap / heavy gap           ║
║  · Alternate universes: full-mega / semis / platforms / AI-core             ║
║  · Local parameter sensitivity: Sharpe surface (vol_target × weight_cap)   ║
║  · stop_keep_cash ablation: research mode vs aggressive mode                ║
║  · IC weight ablation vs equal-weight signal mix                             ║
║                                                                              ║
║  [ATTRIBUTION]                                                                ║
║  · Alpha conditional on exposure>0 (corrects cash-drag bias)                ║
║  · FF5+UMD with graceful fallback                                            ║
║  · Calmar added to objective function (penalises DD during selection)        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product as iproduct
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
import statsmodels.api as sm


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CostsConfig:
    """
    Transaction cost model.
    commission  : one-way brokerage (10 bp = IB Pro tier)
    slippage    : blended bid-ask + market impact (3 bp for large-cap tech)
    gap_factor  : additional overnight gap slippage multiplier (stress test)
    qqq_expense : daily fraction of QQQ 20 bp p.a. expense ratio
    """
    commission:       float = 0.0010
    slippage:         float = 0.0003
    apply_slippage:   bool  = True
    gap_factor:       float = 1.0          # 1.0=none, 1.5=light gap, 2.0=heavy gap
    qqq_expense_ratio:float = 0.0020 / 252


@dataclass
class Mahoraga11Config:
    # ── Universe & benchmarks ─────────────────────────────────────────────────
    universe: Tuple[str, ...] = (
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
        "META", "AVGO", "ASML", "TSM",  "ADBE", "NFLX", "AMD"
    )
    bench_qqq:  str = "QQQ"
    bench_spy:  str = "SPY"
    bench_vix:  str = "^VIX"   # downloaded; proxy used if unavailable

    # ── Capital & dates ───────────────────────────────────────────────────────
    capital_initial: float = 100_000.0
    data_start: str = "2005-01-01"   # extra year for burn-in before train
    data_end:   str = "2026-02-20"
    trading_days: int = 252
    rf_annual:    float = 0.0

    # ── Walk-forward (expanding-window, 5 folds) ──────────────────────────────
    # Each fold: train expands, val=2y fixed, test=1y fixed, embargo=252bd
    # Fold 1: train 2006-2013  val 2014-2015*  test 2016
    # Fold 2: train 2006-2015  val 2016-2017*  test 2018
    # Fold 3: train 2006-2017  val 2018-2019*  test 2020
    # Fold 4: train 2006-2019  val 2020-2021*  test 2022
    # Fold 5: train 2006-2021  val 2022-2023*  test 2024
    # (*) val window starts after embargo from train_end
    # Final evaluation uses full train (2006-2021) + val (2022-2023) + test (2024-2026)
    wf_train_start: str = "2006-01-01"
    wf_folds: Tuple[Tuple[str,str,str,str], ...] = (
        # (train_end, val_start, val_end, test_end)
        ("2013-12-31", "2015-01-01", "2016-12-31", "2017-12-31"),
        ("2015-12-31", "2017-01-01", "2018-12-31", "2019-12-31"),
        ("2017-12-31", "2019-01-01", "2020-12-31", "2021-12-31"),
        ("2019-12-31", "2021-01-01", "2022-12-31", "2023-12-31"),
        ("2021-12-31", "2023-01-01", "2024-06-30", "2026-02-20"),
    )
    embargo_days: int = 252

    # ── Rebalance & selection ─────────────────────────────────────────────────
    rebalance_freq: str = "W-FRI"
    top_k: int = 3   # STRUCTURAL: concentration limit

    # ── Signal look-backs (STRUCTURAL, not swept) ─────────────────────────────
    spans_short:  Tuple[int, ...] = (42, 84)
    spans_long:   Tuple[int, ...] = (126, 252)
    mom_windows:  Tuple[int, ...] = (63, 126, 252)
    rel_windows:  Tuple[int, ...] = (63, 126)
    burn_in: int = 252

    # ── IC weights (fitted on each fold's train; defaults = equal prior) ──────
    w_trend: float = 0.333
    w_mom:   float = 0.333
    w_rel:   float = 0.334

    # ── HRP ───────────────────────────────────────────────────────────────────
    hrp_window:  int   = 252
    weight_cap:  float = 0.60   # FREE PARAM #1

    # ── Chandelier Stop ───────────────────────────────────────────────────────
    atr_window:         int   = 14    # Wilder (1978)
    k_atr:              float = 2.5   # FREE PARAM #2
    stop_on:            bool  = True
    allow_reentry:      bool  = True
    reentry_atr_buffer: float = 0.25
    stop_keep_cash:     bool  = True  # DEFAULT: research mode (cash on stop)

    # ── Crisis Gate (thresholds calibrated on TRAIN data, percentile-based) ───
    crisis_gate_use:     bool  = True
    crisis_dd_pct:       float = 0.05   # train drawdown percentile for DD trigger
    crisis_vol_pct:      float = 0.90   # train vol-zscore percentile for vol trigger
    crisis_min_days_on:  int   = 5
    crisis_min_days_off: int   = 10
    crisis_scale:        float = 0.0
    # Calibrated values (set by calibrate_crisis_thresholds(), not manually)
    crisis_dd_thr:       float = 0.20   # overwritten at runtime
    crisis_vol_zscore_thr: float = 1.5  # overwritten at runtime

    # ── Turbulence Filter ─────────────────────────────────────────────────────
    turb_window:     int   = 63
    illiq_window:    int   = 21
    turb_zscore_thr: float = 1.2   # FREE PARAM #3
    turb_scale_min:  float = 0.30  # FREE PARAM #4
    turb_eval_on_rebalance_only: bool = True

    # ── Vol Targeting ─────────────────────────────────────────────────────────
    vol_target_on:   bool  = True
    vol_target_ann:  float = 0.30   # FREE PARAM #5
    port_vol_window: int   = 63
    max_exposure:    float = 1.0
    min_exposure:    float = 0.0

    # ── Objective (val scoring) ───────────────────────────────────────────────
    target_maxdd:        float = 0.28
    dd_penalty_strength: float = 4.0   # includes Calmar term now
    calmar_weight:       float = 0.15  # explicit Calmar term in objective
    turnover_soft_cap:   float = 12.0
    turnover_penalty:    float = 0.02

    # ── Engineering ───────────────────────────────────────────────────────────
    cache_dir:   str = "data_cache"
    random_seed: int = 42
    plots_dir:   str = "mahoraga11_plots"
    outputs_dir: str = "mahoraga11_outputs"
    label:       str = "MAHORAGA_1.1"


# ── Alternate universes for robustness ────────────────────────────────────────
ALTERNATE_UNIVERSES: Dict[str, Tuple[str, ...]] = {
    "FULL_MEGA":   ("AAPL","MSFT","NVDA","GOOGL","AMZN","META","AVGO","ASML","TSM","ADBE","NFLX","AMD"),
    "SEMIS":       ("NVDA","AVGO","ASML","TSM","AMD","INTC","QCOM","MU","AMAT","LRCX"),
    "PLATFORMS":   ("AAPL","MSFT","GOOGL","AMZN","META","NFLX","ADBE","CRM","NOW","SHOP"),
    "AI_CORE":     ("NVDA","MSFT","GOOGL","AMZN","META","AVGO","AMD","ARM","PLTR","SMCI"),
}

# ── Stress episodes ───────────────────────────────────────────────────────────
STRESS_EPISODES: Dict[str, Tuple[str, str]] = {
    "GFC_2008":       ("2008-01-01", "2009-06-30"),
    "EURO_DEBT_2011": ("2011-07-01", "2011-12-31"),
    "Q4_2018":        ("2018-10-01", "2018-12-31"),
    "COVID_2020":     ("2020-02-15", "2020-06-30"),
    "RATES_2022":     ("2022-01-01", "2022-12-31"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def _hash(obj: dict) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def to_s(x, name: str = "x") -> pd.Series:
    if x is None: return pd.Series(dtype=float, name=name)
    if isinstance(x, pd.Series): return x.rename(name)
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0].rename(name) if x.shape[1] else pd.Series(dtype=float, name=name)
    return pd.Series(np.asarray(x, dtype=float).ravel(), name=name)

def safe_z(s: pd.Series, w: int) -> pd.Series:
    m  = s.rolling(w).mean()
    sd = s.rolling(w).std().replace(0, np.nan)
    return ((s - m) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)

def annualize(r: pd.Series, td: int = 252) -> float:
    r = to_s(r).dropna().replace([np.inf, -np.inf], np.nan).dropna()
    if not len(r): return 0.0
    return float((1.0 + r).prod() ** (td / len(r)) - 1.0)

def ann_vol(r: pd.Series, td: int = 252) -> float:
    r = to_s(r).dropna()
    return float(r.std(ddof=1) * np.sqrt(td)) if len(r) > 1 else np.nan

def sharpe(r: pd.Series, rf: float = 0.0, td: int = 252) -> float:
    r = to_s(r).dropna()
    ex = r - (1.0 + rf) ** (1.0 / td) + 1.0
    sd = ex.std(ddof=1)
    return float(np.sqrt(td) * ex.mean() / sd) if sd and np.isfinite(sd) else 0.0

def sortino(r: pd.Series, rf: float = 0.0, td: int = 252) -> float:
    r = to_s(r).dropna()
    ex = r - (1.0 + rf) ** (1.0 / td) + 1.0
    sd = ex.clip(upper=0.0).std(ddof=1)
    return float(np.sqrt(td) * ex.mean() / sd) if sd and np.isfinite(sd) else 0.0

def max_dd(eq: pd.Series) -> float:
    eq = to_s(eq).dropna()
    return float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0

def calmar(r: pd.Series, eq: pd.Series, td: int = 252) -> float:
    a = annualize(r, td); d = max_dd(eq)
    return float(a / abs(d)) if d != 0 else np.inf

def cvar95(r: pd.Series) -> float:
    x = to_s(r).dropna().values
    if not len(x): return np.nan
    q = np.quantile(x, 0.05)
    t = x[x <= q]
    return float(t.mean()) if len(t) else float(q)

def total_ret(r: pd.Series) -> float:
    r = to_s(r).dropna()
    return float((1.0 + r).prod() - 1.0) if len(r) else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA
# ═══════════════════════════════════════════════════════════════════════════════

def download_ohlcv(tickers: List[str], start: str, end: str, cache_dir: str) -> Dict[str, pd.DataFrame]:
    _ensure_dir(cache_dir)
    key  = _hash({"tickers": sorted(tickers), "start": start, "end": end, "v": 7})
    path = os.path.join(cache_dir, f"ohlcv_{key}.pkl")
    if os.path.exists(path):
        print(f"  [cache] {path}")
        return pd.read_pickle(path)

    print(f"  [download] {len(tickers)} tickers …")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      group_by="ticker", progress=False)
    idx = pd.DatetimeIndex(raw.index)
    if getattr(idx, "tz", None) is not None: idx = idx.tz_localize(None)
    idx = idx[~idx.duplicated()]

    close = pd.DataFrame(index=idx)
    high  = pd.DataFrame(index=idx)
    low   = pd.DataFrame(index=idx)
    vol   = pd.DataFrame(index=idx)

    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                g = lambda f: raw[(t,f)] if (t,f) in raw.columns else None
            else:
                g = lambda f: raw[f] if f in raw.columns else None
            c=g("Close"); h=g("High"); l=g("Low"); v=g("Volume")
            if c is None: continue
            ri = lambda s: pd.Series(s, index=raw.index).reindex(idx).ffill(limit=5)
            close[t]=ri(c); high[t]=ri(h) if h is not None else ri(c)
            low[t]=ri(l) if l is not None else ri(c)
            vol[t]=ri(v) if v is not None else np.nan
        except Exception as e:
            print(f"  [warn] {t}: {e}")

    out = {k: v.dropna(how="all") for k,v in
           [("close",close),("high",high),("low",low),("volume",vol)]}
    pd.to_pickle(out, path)
    return out


def load_ff_factors(cache_dir: str) -> Optional[pd.DataFrame]:
    _ensure_dir(cache_dir)
    path = os.path.join(cache_dir, "ff5_umd.pkl")
    if os.path.exists(path): return pd.read_pickle(path)
    try:
        import pandas_datareader.data as pdr
        ff5 = pdr.get_data_famafrench("F-F_Research_Data_5_Factors_2x3_daily", start="2005-01-01")[0]
        umd = pdr.get_data_famafrench("F-F_Momentum_Factor_daily", start="2005-01-01")[0]
        ff5.index = pd.to_datetime(ff5.index, format="%Y%m%d")
        umd.index = pd.to_datetime(umd.index, format="%Y%m%d")
        ff = ff5.join(umd[["Mom"]], how="left").rename(columns={"Mom":"UMD"}) / 100.0
        pd.to_pickle(ff, path)
        return ff
    except Exception as e:
        print(f"  [warn] FF factors unavailable ({e}). Attribution skipped.")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ASSERTIONS & SPLIT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_fold(train_start: str, train_end: str, val_start: str,
                  val_end: str, test_end: str, embargo_days: int,
                  fold_n: int = 0):
    te = pd.Timestamp(train_end)
    vs = pd.Timestamp(val_start)
    ve = pd.Timestamp(val_end)
    ts_end = pd.Timestamp(test_end)
    embargo_end = te + pd.offsets.BDay(embargo_days)
    assert vs >= embargo_end, (
        f"[fold {fold_n}] val_start={val_start} violates embargo "
        f"(train_end={train_end} + {embargo_days}bd → {embargo_end.date()})"
    )
    assert ve < ts_end, f"[fold {fold_n}] val_end must be before test_end"
    test_start = ve + pd.offsets.BDay(1)
    assert test_start <= pd.Timestamp(test_end)
    print(f"  [fold {fold_n}] OK  train:{train_start}→{train_end}  "
          f"val:{val_start}→{val_end}  test:{test_start.date()}→{test_end}")
    return str(test_start.date())

def validate_no_lookahead(res: Dict, label: str = ""):
    exec_sc  = to_s(res["total_scale"]).fillna(0.0)
    tgt_sc   = to_s(res["total_scale_target"]).fillna(0.0)
    diff = float((exec_sc - tgt_sc.shift(1).fillna(0.0)).abs().max())
    assert diff < 1e-10, f"[{label}] look-ahead in total_scale (max_diff={diff:.2e})"
    w    = res["weights_scaled"]
    to_r = 0.5 * w.diff().abs().fillna(0.0).sum(axis=1)
    to_s_= to_s(res["turnover"]).fillna(0.0)
    diff2= float((to_r - to_s_).abs().max())
    assert diff2 < 1e-8, f"[{label}] turnover mismatch (max_diff={diff2:.2e})"
    print(f"  [OK] no look-ahead [{label}]")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — COVARIANCE
# ═══════════════════════════════════════════════════════════════════════════════

def lw_cov(returns: pd.DataFrame) -> pd.DataFrame:
    lw  = LedoitWolf().fit(returns.values)
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)

def cov_kappa(cov: pd.DataFrame) -> float:
    ev = np.linalg.eigvalsh(cov.values)
    ev = ev[ev > 0]
    return float(np.log10(ev.max() / ev.min())) if len(ev) > 1 else np.inf


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — HRP
# ═══════════════════════════════════════════════════════════════════════════════

def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """HRP (Lopez de Prado 2016) with Ledoit-Wolf covariance."""
    if returns.shape[1] == 1:
        return pd.Series([1.0], index=returns.columns)
    cov  = lw_cov(returns)
    corr = returns.corr()
    dm   = np.sqrt(np.clip(0.5*(1.0 - corr.values), 0, 1))
    np.fill_diagonal(dm, 0.0)
    lnk  = linkage(squareform(dm, checks=False), method="single")

    def _qd(lm):
        lm = lm.astype(int)
        si = pd.Series([lm[-1,0], lm[-1,1]])
        n  = lm[-1,3]
        while si.max() >= n:
            si.index = range(0, len(si)*2, 2)
            df0 = si[si >= n]; i=df0.index; j=df0.values-n
            si[i] = lm[j,0]
            si = pd.concat([si, pd.Series(lm[j,1], index=i+1)]).sort_index()
            si.index = range(len(si))
        return si.tolist()

    ordered = corr.index[_qd(lnk)]
    cov_    = cov.loc[ordered, ordered]
    w       = pd.Series(1.0, index=ordered)

    def _cv(cm, items):
        s  = cm.loc[items,items].values
        iv = 1.0/np.diag(s); iv /= iv.sum()
        return float(iv @ s @ iv)

    clusters = [ordered.tolist()]
    while True:
        clusters = [c for c in clusters if len(c)>1]
        if not clusters: break
        nc=[]
        for c in clusters:
            s=len(c)//2; c1,c2=c[:s],c[s:]
            v1,v2=_cv(cov_,c1),_cv(cov_,c2)
            a=1.0-v1/(v1+v2) if (v1+v2) else 0.5
            w[c1]*=a; w[c2]*=(1.0-a); nc+=[c1,c2]
        clusters=nc

    return (w/w.sum()).astype(float)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════

def _trend(price: pd.Series, cfg: Mahoraga11Config) -> pd.Series:
    votes=[]
    for sp in cfg.spans_short:
        for lp in cfg.spans_long:
            if sp>=lp: continue
            es=price.ewm(span=sp,adjust=False).mean().shift(1)
            el=price.ewm(span=lp,adjust=False).mean().shift(1)
            votes.append(((es>el)&price.notna()).astype(float))
    return (sum(votes)/len(votes)) if votes else pd.Series(0.0, index=price.index)

def _mom(price: pd.Series, cfg: Mahoraga11Config) -> pd.Series:
    raw = sum((price/price.shift(w)-1.0).shift(1) for w in cfg.mom_windows)/len(cfg.mom_windows)
    return ((raw.clip(-1.0,1.0)+1.0)/2.0)

def _rel(price: pd.Series, bench: pd.Series, cfg: Mahoraga11Config) -> pd.Series:
    raw = sum(
        (price/price.shift(w)-1.0).shift(1) - (bench/bench.shift(w)-1.0).shift(1)
        for w in cfg.rel_windows
    ) / len(cfg.rel_windows)
    return ((raw.clip(-1.0,1.0)+1.0)/2.0)


def fit_ic_weights(
    close:      pd.DataFrame,
    qqq:        pd.Series,
    cfg:        Mahoraga11Config,
    train_start: str,
    train_end:   str,
    horizons:   Tuple[int,...] = (1, 5, 21),
) -> Tuple[float, float, float]:
    """
    IC-based signal weights, evaluated at multiple forward-return horizons
    (1d, 5d, 21d). Weights = softmax(mean_IC_across_horizons), clipped and
    smoothed toward equal-weight prior to avoid extreme concentration.
    Computed strictly on [train_start, train_end].
    """
    sub   = close.loc[train_start:train_end]
    qqq_  = to_s(qqq.loc[train_start:train_end].ffill())
    idx   = sub.index

    ic_by_horizon: Dict[int, List[float]] = {h: [] for h in horizons}

    for t in sub.columns:
        p  = sub[t].ffill()
        tr = _trend(p, cfg).reindex(idx)
        mo = _mom(p, cfg).reindex(idx)
        re = _rel(p, qqq_.reindex(idx).ffill(), cfg).reindex(idx)
        composite = cfg.w_trend*tr + cfg.w_mom*mo + cfg.w_rel*re

        for h in horizons:
            fwd = sub[t].pct_change().shift(-h).reindex(idx)
            ok  = composite.notna() & fwd.notna()
            if ok.sum() < 50: continue
            for sig, sig_ic in [(tr,"trend"),(mo,"mom"),(re,"rel")]:
                pass  # collected below

        for h in horizons:
            fwd = sub[t].pct_change().shift(-h).reindex(idx)
            ok  = fwd.notna()
            if ok.sum() < 50: continue
            ics=[]
            for sig in [tr, mo, re]:
                common = sig.notna() & ok
                if common.sum() < 30: continue
                r, _ = stats.spearmanr(sig[common], fwd[common])
                ics.append(float(r) if np.isfinite(r) else 0.0)
            if len(ics)==3:
                for idx_i, ic_val in enumerate(ics):
                    ic_by_horizon[h].append(ic_val)

    # Flatten: one IC triplet per (asset × horizon)
    trend_ics, mom_ics, rel_ics = [], [], []
    for h in horizons:
        vals = ic_by_horizon[h]
        if len(vals) == 0: continue
        n3 = (len(vals)//3)*3
        vals = vals[:n3]
        for i in range(0, n3, 3):
            trend_ics.append(vals[i])
            mom_ics.append(vals[i+1])
            rel_ics.append(vals[i+2])

    ic = np.array([
        max(np.nanmean(trend_ics) if trend_ics else 0.01, 0.005),
        max(np.nanmean(mom_ics)   if mom_ics   else 0.01, 0.005),
        max(np.nanmean(rel_ics)   if rel_ics   else 0.01, 0.005),
    ])
    # Softmax then blend toward equal weights (smoothing)
    w_raw = np.exp(ic*20.0) / np.exp(ic*20.0).sum()
    w_eq  = np.array([1/3, 1/3, 1/3])
    blend = 0.4   # 40% toward equal weight → avoids extreme weights from noise
    w     = blend*w_eq + (1.0-blend)*w_raw
    w     = np.clip(w, 0.10, 0.70)
    w    /= w.sum()
    print(f"    [IC] trend={w[0]:.3f} mom={w[1]:.3f} rel={w[2]:.3f}  "
          f"(raw IC @{horizons}: {ic[0]:.4f}/{ic[1]:.4f}/{ic[2]:.4f})")
    return float(w[0]), float(w[1]), float(w[2])


def compute_scores(close: pd.DataFrame, qqq: pd.Series, cfg: Mahoraga11Config) -> pd.DataFrame:
    idx  = close.index
    qqq_ = to_s(qqq,"QQQ").reindex(idx).ffill()
    sc   = pd.DataFrame(index=idx, columns=close.columns, dtype=float)
    for t in close.columns:
        p  = close[t].reindex(idx).ffill()
        s  = cfg.w_trend*_trend(p,cfg) + cfg.w_mom*_mom(p,cfg) + cfg.w_rel*_rel(p,qqq_,cfg)
        s.iloc[:cfg.burn_in] = 0.0
        sc[t] = s.fillna(0.0)
    return sc.fillna(0.0)


def rolling_ic_multi_horizon(
    close: pd.DataFrame,
    qqq:   pd.Series,
    cfg:   Mahoraga11Config,
    window: int = 63,
    horizons: Tuple[int,...] = (1, 5, 21),
) -> pd.DataFrame:
    """Rolling IC at 1d, 5d, 21d forward-return horizons."""
    idx  = close.index
    qqq_ = to_s(qqq,"QQQ").reindex(idx).ffill()
    ic_df= pd.DataFrame(index=idx, dtype=float)

    for h in horizons:
        fwd  = close.pct_change().shift(-h)
        comp = pd.DataFrame(index=idx, columns=close.columns, dtype=float)
        for t in close.columns:
            p     = close[t].reindex(idx).ffill()
            score = cfg.w_trend*_trend(p,cfg)+cfg.w_mom*_mom(p,cfg)+cfg.w_rel*_rel(p,qqq_,cfg)
            comp[t] = score

        ic_series = pd.Series(np.nan, index=idx)
        for i in range(len(idx)):
            s_ = comp.iloc[i].dropna()
            f_ = fwd.iloc[i][s_.index].dropna()
            c_ = s_.index.intersection(f_.index)
            if len(c_) >= 4:
                r, _ = stats.spearmanr(s_[c_], f_[c_])
                if np.isfinite(r): ic_series.iloc[i] = r
        ic_df[f"IC_composite_{h}d"] = ic_series.rolling(window).mean().fillna(0.0)

    return ic_df


def select_topk(scores: pd.DataFrame, k: int, freq: str) -> pd.DataFrame:
    idx  = scores.index
    reb  = set(scores.resample(freq).last().index)
    mask = pd.DataFrame(0.0, index=idx, columns=scores.columns)
    last = np.zeros(scores.shape[1])
    for dt in idx:
        if dt in reb:
            row   = scores.loc[dt].values
            order = np.argsort(-row)
            last  = np.zeros(len(order))
            last[order[:k]] = 1.0
        mask.loc[dt] = last
    return mask.fillna(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — RISK OVERLAYS
# ═══════════════════════════════════════════════════════════════════════════════

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev = close.shift(1)
    tr   = pd.concat([(high-low).abs(),(high-prev).abs(),(low-prev).abs()],axis=1).max(axis=1)
    return tr.ewm(span=2*window-1, adjust=False).mean()


def apply_chandelier(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    high:  pd.DataFrame,
    low:   pd.DataFrame,
    cfg:   Mahoraga11Config,
) -> Tuple[pd.DataFrame, int]:
    if not cfg.stop_on:
        return weights.copy(), 0

    out  = weights.copy()
    idx  = out.index
    reb  = set(out.resample(cfg.rebalance_freq).last().index)
    hits = 0

    for t in out.columns:
        p    = close[t].reindex(idx).ffill()
        atr_ = _atr(high[t].reindex(idx).ffill(),
                     low[t].reindex(idx).ffill(),
                     p, cfg.atr_window).bfill().fillna(0.0)
        wt       = out[t].values.copy()
        in_pos   = stopped = False
        maxp     = np.nan
        last_sl  = np.nan

        for i, dt in enumerate(idx):
            if dt in reb and stopped:
                rec_thr = (last_sl + cfg.reentry_atr_buffer * float(atr_.iloc[i])
                           if np.isfinite(last_sl) else -np.inf)
                if float(p.iloc[i]) > rec_thr:
                    stopped = False

            if wt[i] <= 0:
                in_pos = stopped = False; maxp = np.nan; continue
            if stopped:
                wt[i] = 0.0; continue
            if not in_pos:
                in_pos = True; maxp = float(p.iloc[i])
            else:
                maxp = max(maxp, float(p.iloc[i]))

            sl = maxp - cfg.k_atr * float(atr_.iloc[i])
            if float(p.iloc[i]) < sl:
                wt[i] = 0.0; stopped = True; in_pos = False
                last_sl = sl; maxp = np.nan; hits += 1

        out[t] = wt

    if not cfg.stop_keep_cash:
        d = out.sum(axis=1).replace(0, np.nan)
        out = out.div(d, axis=0).fillna(0.0)

    return out, hits


def calibrate_crisis_thresholds(
    qqq_close:   pd.Series,
    train_start: str,
    train_end:   str,
    cfg:         Mahoraga11Config,
) -> Tuple[float, float]:
    """
    Estimate crisis_dd_thr and crisis_vol_zscore_thr from TRAIN data only.
    DD threshold  = cfg.crisis_dd_pct-th percentile of drawdown series (e.g. 5th = most extreme 5%)
    Vol threshold = cfg.crisis_vol_pct-th percentile of rolling-vol z-score (e.g. 90th)
    """
    p_tr = to_s(qqq_close.loc[train_start:train_end]).ffill()
    r_tr = p_tr.pct_change().fillna(0.0)

    dd_tr = p_tr / p_tr.cummax() - 1.0
    dd_thr = float(abs(np.nanpercentile(dd_tr, cfg.crisis_dd_pct * 100)))
    dd_thr = max(dd_thr, 0.10)   # floor: at least 10% DD to trigger

    vol_tr = r_tr.rolling(63).std() * np.sqrt(cfg.trading_days)
    vol_z_tr = safe_z(vol_tr, 252)
    vol_thr = float(np.nanpercentile(vol_z_tr.dropna(), cfg.crisis_vol_pct * 100))
    vol_thr = max(vol_thr, 0.5)

    print(f"  [crisis] DD threshold={dd_thr:.3f}  vol-z threshold={vol_thr:.3f}  "
          f"(calibrated on {train_start}→{train_end})")
    return dd_thr, vol_thr


def compute_crisis_gate(
    qqq_close:   pd.Series,
    cfg:         Mahoraga11Config,
) -> Tuple[pd.Series, pd.Series]:
    p   = to_s(qqq_close,"QQQ").ffill()
    idx = p.index
    r   = p.pct_change().fillna(0.0)
    vol = r.rolling(cfg.port_vol_window).std() * np.sqrt(cfg.trading_days)
    vol_z = safe_z(vol, cfg.port_vol_window * 4)
    dd    = p / p.cummax() - 1.0

    cond     = ((dd <= -cfg.crisis_dd_thr) | (vol_z >= cfg.crisis_vol_zscore_thr)).astype(int)
    on_flag  = cond.rolling(cfg.crisis_min_days_on).mean().fillna(0.0) >= 0.8
    off_flag = (1-cond).rolling(cfg.crisis_min_days_off).mean().fillna(0.0) >= 0.8

    state  = pd.Series(0.0, index=idx)
    in_c   = False
    for dt in idx:
        if not in_c and bool(on_flag.loc[dt]):   in_c = True
        elif in_c and bool(off_flag.loc[dt]):    in_c = False
        state.loc[dt] = 1.0 if in_c else 0.0

    scale = pd.Series(1.0, index=idx, dtype=float)
    if cfg.crisis_gate_use:
        scale[state==1.0] = cfg.crisis_scale
    scale.iloc[:cfg.burn_in] = cfg.crisis_scale
    state.iloc[:cfg.burn_in] = 1.0
    return scale, state


def compute_turbulence(
    close:  pd.DataFrame,
    volume: pd.DataFrame,
    qqq:    pd.Series,
    cfg:    Mahoraga11Config,
) -> pd.Series:
    idx   = close.index
    qqq_r = to_s(qqq,"QQQ").reindex(idx).ffill().pct_change().fillna(0.0)
    rets  = close.pct_change().fillna(0.0)

    vol_q = qqq_r.rolling(cfg.turb_window).std() * np.sqrt(cfg.trading_days)
    vol_z = safe_z(vol_q, 252)

    avg_corr  = pd.Series(0.0, index=idx)
    reb_dates = set(close.resample(cfg.rebalance_freq).last().index)
    w = cfg.turb_window
    for dt in (reb_dates if cfg.turb_eval_on_rebalance_only else idx):
        if dt not in idx: continue
        loc = idx.get_loc(dt)
        if loc < w: continue
        sub = rets.iloc[loc-w+1:loc+1]
        c   = sub.corr().values; n=c.shape[0]
        avg_corr.loc[dt] = (c.sum()-n)/(n*(n-1)) if n>1 else 0.0
    avg_corr = avg_corr.replace(0,np.nan).ffill().fillna(0.0)
    corr_z   = safe_z(avg_corr, 252)

    dv = (close*volume).replace(0,np.nan)
    illiq = (rets.abs()/dv).replace([np.inf,-np.inf],np.nan)
    illiq_avg = np.log1p(illiq.rolling(cfg.illiq_window).mean().mean(axis=1).fillna(0.0))
    illiq_z   = safe_z(illiq_avg, 252)

    turb = (vol_z + corr_z + illiq_z).ewm(span=10, adjust=False).mean()
    a    = 1.2
    s    = pd.Series(1.0/(1.0+np.exp(a*(turb-cfg.turb_zscore_thr))), index=idx)
    s    = s.clip(lower=cfg.turb_scale_min, upper=1.0)
    s.iloc[:cfg.burn_in] = cfg.turb_scale_min
    return s


def vol_target_scale(port_r: pd.Series, cfg: Mahoraga11Config) -> pd.Series:
    if not cfg.vol_target_on:
        return pd.Series(1.0, index=port_r.index)
    rv = to_s(port_r).fillna(0.0).rolling(cfg.port_vol_window).std() * np.sqrt(cfg.trading_days)
    s  = (cfg.vol_target_ann / rv).replace([np.inf,-np.inf],np.nan).fillna(1.0)
    return s.clip(cfg.min_exposure, cfg.max_exposure)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — CORE BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

def _costs(w: pd.DataFrame, c: CostsConfig) -> Tuple[pd.Series, pd.Series]:
    dw = w.diff().abs().fillna(0.0)
    to = 0.5 * dw.sum(axis=1)
    slip = c.slippage * c.gap_factor if c.apply_slippage else 0.0
    tc = to * (c.commission + slip)
    return to, tc


def backtest(
    ohlcv:  Dict[str, pd.DataFrame],
    cfg:    Mahoraga11Config,
    costs:  CostsConfig,
    label:  str = "MAHORAGA_1.1",
) -> Dict:
    np.random.seed(cfg.random_seed)
    univ   = list(cfg.universe)
    close  = ohlcv["close"][univ].copy()
    high   = ohlcv["high"][univ].copy()
    low    = ohlcv["low"][univ].copy()
    volume = ohlcv["volume"][univ].copy()
    idx    = close.index

    qqq = to_s(ohlcv["close"][cfg.bench_qqq].reindex(idx).ffill(), "QQQ")
    spy = to_s(ohlcv["close"][cfg.bench_spy].reindex(idx).ffill(), "SPY")

    crisis_scale, crisis_state = compute_crisis_gate(qqq, cfg)
    turb_scale                  = compute_turbulence(close, volume, qqq, cfg)

    scores     = compute_scores(close, qqq, cfg)
    active     = select_topk(scores, cfg.top_k, cfg.rebalance_freq)
    rets       = close.pct_change().fillna(0.0)
    reb_dates  = set(close.resample(cfg.rebalance_freq).last().index)

    w = pd.DataFrame(0.0, index=idx, columns=univ)
    last_w = pd.Series(0.0, index=univ)

    for dt in idx:
        if dt in reb_dates:
            sel   = active.loc[dt]
            names = sel[sel>0].index.tolist()
            if not names:
                last_w = pd.Series(0.0, index=univ)
            elif len(names)==1:
                last_w = pd.Series(0.0, index=univ); last_w[names[0]]=1.0
            else:
                lb = rets.loc[:dt].tail(cfg.hrp_window)[names].dropna()
                if len(lb)<60:
                    cov  = lw_cov(lb if len(lb)>len(names) else rets.loc[:dt][names].dropna())
                    iv   = 1.0/np.diag(cov.values); iv/=iv.sum()
                    ww   = pd.Series(iv, index=names)
                else:
                    ww   = hrp_weights(lb)
                ww = (ww.clip(upper=cfg.weight_cap) / ww.clip(upper=cfg.weight_cap).sum())
                last_w = pd.Series(0.0, index=univ)
                last_w[names] = ww.values
        w.loc[dt] = last_w.values

    w_stop, stop_hits = apply_chandelier(w, close, high, low, cfg)
    w_exec_1x = w_stop.shift(1).fillna(0.0)
    gross_1x  = (w_exec_1x * rets).sum(axis=1)
    vol_sc    = vol_target_scale(gross_1x, cfg)
    cap       = (crisis_scale * turb_scale).clip(0.0, cfg.max_exposure)
    tgt_sc    = pd.Series(np.minimum(vol_sc.values, cap.values), index=idx).clip(0.0, cfg.max_exposure)
    exec_sc   = tgt_sc.shift(1).fillna(0.0)
    w_exec    = w_exec_1x.mul(exec_sc, axis=0)
    to, tc    = _costs(w_exec, costs)
    port_net  = ((w_exec*rets).sum(axis=1) - tc).replace([np.inf,-np.inf],0.0).fillna(0.0)
    equity    = cfg.capital_initial * (1.0+port_net).cumprod()
    exposure  = w_exec.abs().sum(axis=1).clip(0.0, cfg.max_exposure)

    qqq_r  = qqq.pct_change().fillna(0.0) - costs.qqq_expense_ratio
    spy_r  = spy.pct_change().fillna(0.0)
    qqq_eq = cfg.capital_initial * (1.0+qqq_r).cumprod()
    spy_eq = cfg.capital_initial * (1.0+spy_r).cumprod()

    return {
        "label": label, "returns_net": port_net, "equity": equity,
        "exposure": exposure, "turnover": to,
        "weights_scaled": w_exec, "total_scale": exec_sc, "total_scale_target": tgt_sc,
        "cap": cap, "turb_scale": turb_scale, "crisis_scale": crisis_scale,
        "crisis_state": crisis_state, "vol_scale": vol_sc,
        "stop_hits": stop_hits, "scores": scores,
        "bench": {"QQQ_r": qqq_r, "QQQ_eq": qqq_eq, "SPY_r": spy_r, "SPY_eq": spy_eq},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def summarize(r, eq, exp, to, cfg: Mahoraga11Config, label="") -> Dict:
    r   = to_s(r).replace([np.inf,-np.inf],np.nan).dropna()
    eq  = to_s(eq).dropna()
    exp_= to_s(exp).reindex(r.index).fillna(0.0) if exp is not None else pd.Series(np.nan, index=r.index)
    to_ = to_s(to).reindex(r.index).fillna(0.0) if to  is not None else pd.Series(0.0, index=r.index)
    T   = len(r)
    return {
        "Label": label,
        "FinalEquity":  float(eq.iloc[-1]) if len(eq) else np.nan,
        "TotalReturn":  total_ret(r),
        "CAGR":         annualize(r, cfg.trading_days),
        "AnnVol":       ann_vol(r, cfg.trading_days),
        "Sharpe":       sharpe(r, cfg.rf_annual, cfg.trading_days),
        "Sortino":      sortino(r, cfg.rf_annual, cfg.trading_days),
        "MaxDD":        max_dd(eq),
        "Calmar":       calmar(r, eq, cfg.trading_days),
        "CVaR_5":       cvar95(r),
        "AvgExposure":  float(exp_.mean()),
        "TimeInMkt":    float((exp_>0).mean()),
        "TurnoverAnn":  float(to_.sum()*cfg.trading_days/T) if T else 0.0,
        "Days": int(T),
    }


def asymptotic_sharpe_ci(r: pd.Series, cfg: Mahoraga11Config, alpha: float = 0.05) -> Dict:
    """
    Asymptotic Sharpe CI using the delta method under i.i.d. normality.
    SE(SR_daily) = sqrt((1 + SR_daily^2/2) / T).
    This is an approximation; serial correlation inflates true SE.
    """
    r  = to_s(r).dropna(); T = len(r)
    sr_d = sharpe(r, cfg.rf_annual, 1)
    se_d = np.sqrt((1.0 + sr_d**2/2.0)/T)
    z    = stats.norm.ppf(1.0 - alpha/2.0)
    td   = cfg.trading_days
    sr_a = sr_d*np.sqrt(td); se_a = se_d*np.sqrt(td)
    t_s  = sr_d/se_d if se_d>0 else 0.0
    return {
        "SR": round(sr_a,4), "CI_lo": round(sr_a-z*se_a,4),
        "CI_hi": round(sr_a+z*se_a,4), "SE": round(se_a,4),
        "t_stat": round(t_s,3), "p_val": round(2.0*(1.0-stats.norm.cdf(abs(t_s))),5),
        "note": "Asymptotic (delta-method); underestimates SE under autocorrelation",
    }


def alpha_test_nw(r_s, r_b, cfg: Mahoraga11Config, label: str = "",
                  conditional: bool = False, exposure: Optional[pd.Series] = None) -> Dict:
    """
    Newey-West HAC alpha test.
    If conditional=True, restricts sample to days with exposure>0,
    removing the cash-drag bias from alpha estimation.
    """
    r_s = to_s(r_s).dropna()
    r_b = to_s(r_b).reindex(r_s.index).fillna(0.0)
    if conditional and exposure is not None:
        exp_ = to_s(exposure).reindex(r_s.index).fillna(0.0)
        mask = exp_ > 0.01
        r_s  = r_s[mask]; r_b = r_b[mask]
    common = r_s.index.intersection(r_b.index)
    r_s = r_s[common]; r_b = r_b[common]
    if len(r_s) < 100:
        return {"Label": label, "error": f"Insufficient data ({len(r_s)} obs)"}
    X    = sm.add_constant(r_b.values)
    lags = int(4*(len(r_s)/100)**(2.0/9.0))
    try:
        ols  = sm.OLS(r_s.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        a_d  = float(ols.params[0])
        a_a  = float((1.0+a_d)**cfg.trading_days - 1.0)
        return {
            "Label": label, "conditional": conditional,
            "alpha_ann": round(a_a,6), "t_alpha": round(float(ols.tvalues[0]),3),
            "p_alpha": round(float(ols.pvalues[0]),5),
            "beta": round(float(ols.params[1]),4), "R2": round(float(ols.rsquared),4),
            "NW_lags": lags, "n_obs": int(len(r_s)),
            "sig_5pct": bool(ols.pvalues[0]<0.05), "sig_1pct": bool(ols.pvalues[0]<0.01),
        }
    except Exception as e:
        return {"Label": label, "error": str(e)}


def factor_attribution(r_s, ff, cfg: Mahoraga11Config, label="") -> Optional[Dict]:
    if ff is None: return None
    r   = to_s(r_s).dropna()
    ff_ = ff.reindex(r.index).dropna()
    c   = r.index.intersection(ff_.index)
    if len(c)<252: return {"Label":label,"error":"Insufficient FF data"}
    y    = r[c].values - ff_.loc[c,"RF"].values
    X    = sm.add_constant(ff_.loc[c,["Mkt-RF","SMB","HML","RMW","CMA","UMD"]].values)
    lags = int(4*(len(y)/100)**(2.0/9.0))
    try:
        ols = sm.OLS(y,X).fit(cov_type="HAC",cov_kwds={"maxlags":lags})
        a_a = float((1.0+float(ols.params[0]))**cfg.trading_days-1.0)
        p   = ols.params; tv=ols.tvalues; pv=ols.pvalues
        return {
            "Label":label,"alpha_ann":round(a_a,6),
            "t_alpha":round(float(tv[0]),3),"p_alpha":round(float(pv[0]),5),
            "beta_mkt":round(float(p[1]),4),"beta_smb":round(float(p[2]),4),
            "beta_hml":round(float(p[3]),4),"beta_rmw":round(float(p[4]),4),
            "beta_cma":round(float(p[5]),4),"beta_umd":round(float(p[6]),4),
            "R2_adj":round(float(ols.rsquared_adj),4),
        }
    except Exception as e:
        return {"Label":label,"error":str(e)}


def regime_analysis(r_s, r_b, ohlcv, cfg: Mahoraga11Config) -> pd.DataFrame:
    """VIX regime decomposition. Uses real ^VIX if downloaded, else SPY vol proxy."""
    idx = to_s(r_s).index
    vix_available = False
    if cfg.bench_vix in ohlcv.get("close",pd.DataFrame()).columns:
        vix = to_s(ohlcv["close"][cfg.bench_vix].reindex(idx).ffill())
        vix_available = True
    else:
        spy_r = to_s(r_b).fillna(0.0)
        vix   = (spy_r.rolling(21).std()*np.sqrt(252)*100).reindex(idx).ffill()
    vix_label = "VIX" if vix_available else "realized-vol proxy (VIX unavailable)"

    p20 = float(np.nanpercentile(vix,20))
    p80 = float(np.nanpercentile(vix,80))
    regimes = {
        f"CALM ({vix_label}<{p20:.0f})":  vix<p20,
        f"NORMAL ({p20:.0f}≤{vix_label}<{p80:.0f})": (vix>=p20)&(vix<p80),
        f"STRESS ({vix_label}≥{p80:.0f})": vix>=p80,
    }
    rows=[]
    for name, mask in regimes.items():
        mask = mask.reindex(to_s(r_s).index).fillna(False)
        rs   = to_s(r_s)[mask]; rb=to_s(r_b).reindex(rs.index).fillna(0.0)
        if len(rs)<20: continue
        eq = cfg.capital_initial*(1.0+rs).cumprod()
        rows.append({
            "Regime":name,"VIX_source":"real" if vix_available else "proxy",
            "Days":int(len(rs)),"Days%":round(100*len(rs)/len(to_s(r_s)),1),
            "CAGR%":round(annualize(rs,cfg.trading_days)*100,2),
            "CAGR_bench%":round(annualize(rb,cfg.trading_days)*100,2),
            "Excess_CAGR%":round((annualize(rs,cfg.trading_days)-annualize(rb,cfg.trading_days))*100,2),
            "Sharpe":round(sharpe(rs,cfg.rf_annual,cfg.trading_days),3),
            "MaxDD%":round(max_dd(eq)*100,2),
            "Hit_Rate%":round(100*(rs>0).mean(),1),
        })
    return pd.DataFrame(rows)


def stress_report(r, exp, episodes, cfg: Mahoraga11Config, r_bench=None) -> pd.DataFrame:
    rows=[]
    for name,(a,b) in episodes.items():
        sub=to_s(r).loc[a:b]
        if len(sub)<40: continue
        ee = cfg.capital_initial*(1.0+sub).cumprod()
        ss = summarize(sub,ee,to_s(exp).loc[sub.index],None,cfg)
        bt = None
        if r_bench is not None:
            rb=to_s(r_bench).loc[a:b]; bt=round((1.0+rb).prod()-1.0,4)
        rows.append({
            "Episode":name,"Days":int(len(sub)),
            "Total%":round((1.0+sub).prod()*100-100,2),
            "Bench_Total%":round(bt*100,2) if bt is not None else None,
            "Excess%":round(((1+sub).prod()-(1+to_s(r_bench).loc[a:b]).prod())*100,2) if r_bench is not None else None,
            "WorstDay%":round(sub.min()*100,2),
            "Sharpe":round(ss["Sharpe"],3),"MaxDD%":round(ss["MaxDD"]*100,2),
            "Calmar":round(ss["Calmar"],3),"AvgExp%":round(ss["AvgExposure"]*100,1),
        })
    return pd.DataFrame(rows)


def moving_block_bootstrap(r, block=20, n=1000, seed=42) -> Dict:
    rng = np.random.default_rng(seed)
    x   = to_s(r).replace([np.inf,-np.inf],np.nan).dropna().values
    T   = len(x)
    if T<block*5: return {"dd_p50":np.nan,"dd_p5_worst":np.nan,"ruin_prob_30dd":np.nan,"ruin_prob_50dd":np.nan}
    dds=[]
    for _ in range(n):
        st  = rng.integers(0,T-block,size=int(np.ceil(T/block)))
        smp = np.concatenate([x[s:s+block] for s in st])[:T]
        eq  = np.cumprod(1.0+smp)
        dds.append(float(np.min(eq/np.maximum.accumulate(eq)-1.0)))
    dds = np.array(dds)
    return {
        "dd_p50":         float(np.quantile(dds,0.50)),
        "dd_p5_worst":    float(np.quantile(dds,0.05)),
        "dd_p1_worst":    float(np.quantile(dds,0.01)),
        "ruin_prob_30dd": float(np.mean(dds<-0.30))*100.0,
        "ruin_prob_50dd": float(np.mean(dds<-0.50))*100.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — BASELINES
# ═══════════════════════════════════════════════════════════════════════════════

def baseline_eqw(ohlcv, cfg: Mahoraga11Config, costs: CostsConfig) -> Dict:
    close = ohlcv["close"][list(cfg.universe)].copy()
    n=len(cfg.universe); idx=close.index
    w    = pd.DataFrame(1.0/n, index=idx, columns=cfg.universe)
    rets = close.pct_change().fillna(0.0)
    we   = w.shift(1).fillna(0.0)
    r_n  = ((we*rets).sum(axis=1) - _costs(we,costs)[1]).fillna(0.0)
    eq   = cfg.capital_initial*(1.0+r_n).cumprod()
    return {"r":r_n,"eq":eq,"exp":we.abs().sum(axis=1),"to":_costs(we,costs)[0],"label":"EQW_Tech"}


def baseline_mom(ohlcv, cfg: Mahoraga11Config, costs: CostsConfig) -> Dict:
    close = ohlcv["close"][list(cfg.universe)].copy()
    mom   = (close.shift(21)/close.shift(273)-1.0).shift(1).fillna(0.0)
    sel   = select_topk(mom,cfg.top_k,cfg.rebalance_freq)
    d     = sel.sum(axis=1).replace(0,np.nan)
    w     = sel.div(d,axis=0).fillna(0.0)
    rets  = close.pct_change().fillna(0.0)
    we    = w.shift(1).fillna(0.0)
    to,tc = _costs(we,costs)
    r_n   = ((we*rets).sum(axis=1)-tc).fillna(0.0)
    eq    = cfg.capital_initial*(1.0+r_n).cumprod()
    return {"r":r_n,"eq":eq,"exp":we.abs().sum(axis=1),"to":to,"label":"MOM_12_1_TopK"}


def baseline_signal_decomp(ohlcv, cfg: Mahoraga11Config, costs: CostsConfig) -> Dict[str,Dict]:
    results={}
    for comp,(wt,wm,wr) in [
        ("TREND_ONLY",(1.0,0.0,0.0)),
        ("MOM_ONLY",  (0.0,1.0,0.0)),
        ("REL_ONLY",  (0.0,0.0,1.0)),
    ]:
        c2 = deepcopy(cfg); c2.w_trend=wt; c2.w_mom=wm; c2.w_rel=wr
        results[comp] = backtest(ohlcv,c2,costs,label=comp)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — SWEEP & OBJECTIVE
# ═══════════════════════════════════════════════════════════════════════════════

def objective(s_val: Dict, s_qqq: Dict, cfg: Mahoraga11Config) -> float:
    """
    Objective function includes explicit Calmar term to penalise deep drawdowns
    during validation selection — this directly addresses the MaxDD-in-NORMAL issue.
    """
    exc_cagr   = s_val["CAGR"]   - s_qqq["CAGR"]
    exc_sharpe = s_val["Sharpe"] - s_qqq["Sharpe"]
    exc_sort   = s_val["Sortino"]- s_qqq["Sortino"]
    exc_calmar = s_val["Calmar"] - s_qqq["Calmar"]
    dd_pen = cfg.dd_penalty_strength * max(0.0, abs(s_val["MaxDD"])-cfg.target_maxdd)
    to_pen = cfg.turnover_penalty * max(0.0, s_val["TurnoverAnn"]-cfg.turnover_soft_cap)
    return float(1.00*exc_cagr + 0.20*exc_sharpe + 0.20*exc_sort
                 + cfg.calmar_weight*exc_calmar - dd_pen - to_pen)


def bhy(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    n   = len(p_values)
    c_n = np.sum(1.0/np.arange(1,n+1))
    si  = np.argsort(p_values); sp=p_values[si]
    thr = (np.arange(1,n+1)/(n*c_n))*alpha
    sig = np.zeros(n,dtype=bool)
    for i in range(n-1,-1,-1):
        if sp[i]<=thr[i]:
            sig[si[:i+1]]=True; break
    return sig


SWEEP_GRID = {
    "weight_cap":      [0.55, 0.65],
    "k_atr":           [2.0,  2.5,  3.0],
    "turb_zscore_thr": [1.0,  1.5],
    "turb_scale_min":  [0.25, 0.40],
    "vol_target_ann":  [0.25, 0.30, 0.35],
}


def run_fold_sweep(
    ohlcv:      Dict,
    cfg_base:   Mahoraga11Config,
    costs:      CostsConfig,
    ic_weights: Tuple[float,float,float],
    val_start:  str,
    val_end:    str,
    fold_n:     int,
) -> Tuple[pd.DataFrame, Dict]:
    """Run SWEEP_GRID on a single fold's val period. Returns sweep DF + best config."""
    wt,wm,wr = ic_weights
    keys   = list(SWEEP_GRID.keys())
    combos = list(iproduct(*[SWEEP_GRID[k] for k in keys]))

    rows, best = [], None
    for combo in combos:
        kw  = dict(zip(keys, combo))
        cfg = deepcopy(cfg_base)
        for k,v in kw.items(): setattr(cfg,k,v)
        cfg.w_trend=wt; cfg.w_mom=wm; cfg.w_rel=wr

        res = backtest(ohlcv, cfg, costs, label=f"sweep_f{fold_n}")
        r_v = res["returns_net"].loc[val_start:val_end]
        if len(r_v)<50: continue
        eq_v  = cfg.capital_initial*(1.0+r_v).cumprod()
        exp_v = res["exposure"].loc[r_v.index]
        to_v  = res["turnover"].loc[r_v.index]
        s_v   = summarize(r_v,eq_v,exp_v,to_v,cfg)
        qr    = res["bench"]["QQQ_r"].loc[val_start:val_end]
        s_q   = summarize(qr,cfg.capital_initial*(1.0+qr).cumprod(),None,None,cfg)
        ci    = asymptotic_sharpe_ci(r_v,cfg)
        sc    = objective(s_v,s_q,cfg)

        row = {**kw,"fold":fold_n,"score_val":sc,
               "VAL_Sharpe":round(s_v["Sharpe"],4),
               "VAL_CAGR%": round(s_v["CAGR"]*100,3),
               "VAL_MaxDD%":round(s_v["MaxDD"]*100,3),
               "VAL_SR_tstat":round(ci["t_stat"],3),
               "VAL_SR_pval": round(ci["p_val"],5)}
        rows.append(row)
        if best is None or sc>best["score"]:
            best={"score":sc,"cfg":cfg,"s_val":s_v,"s_qqq":s_q}

    df = pd.DataFrame(rows).sort_values("score_val",ascending=False)
    pv = df["VAL_SR_pval"].values
    df["SR_sig_BHY"] = bhy(pv)
    c_n = np.sum(1.0/np.arange(1,len(combos)+1))
    t_min = stats.norm.ppf(1.0-(0.05/(len(combos)*c_n))/2.0)
    df["SR_above_HLZ_tmin"] = df["VAL_SR_tstat"] > t_min
    df["stat_label"] = df.apply(
        lambda x: "sig@5%" if x["SR_sig_BHY"] else "econ_strong_stat_weak", axis=1
    )
    return df, best


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — WALK-FORWARD ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_walk_forward(
    ohlcv:    Dict,
    cfg_base: Mahoraga11Config,
    costs:    CostsConfig,
) -> Tuple[pd.Series, pd.Series, List[Dict], pd.DataFrame]:
    """
    Expanding-window walk-forward with 5 folds.
    Returns:
      - oos_r:    stitched OOS daily returns (5 independent test years)
      - oos_eq:   cumulative equity of stitched OOS
      - fold_results: list of per-fold dicts with metrics
      - all_sweeps:   combined sweep DataFrame across all folds
    """
    train_start = cfg_base.wf_train_start
    all_oos_r   = []
    fold_results= []
    all_sweeps  = []
    final_cfg   = None

    for fold_n, (train_end, val_start, val_end, test_end) in enumerate(cfg_base.wf_folds):
        print(f"\n  ── FOLD {fold_n+1}/5 ──")
        test_start = validate_fold(train_start, train_end, val_start, val_end, test_end,
                                   cfg_base.embargo_days, fold_n+1)

        # Calibrate crisis thresholds on this fold's train data
        cfg_f = deepcopy(cfg_base)
        qqq_full = to_s(ohlcv["close"][cfg_base.bench_qqq].ffill())
        dd_thr, vol_thr = calibrate_crisis_thresholds(qqq_full, train_start, train_end, cfg_f)
        cfg_f.crisis_dd_thr = dd_thr
        cfg_f.crisis_vol_zscore_thr = vol_thr

        # Fit IC weights on this fold's train data
        print(f"  [fold {fold_n+1}] Fitting IC weights on train …")
        close_univ = ohlcv["close"][list(cfg_base.universe)]
        qqq_tr     = qqq_full.loc[train_start:train_end]
        ic_weights = fit_ic_weights(close_univ, qqq_tr, cfg_f, train_start, train_end)

        # Sweep on val
        print(f"  [fold {fold_n+1}] Sweeping on val {val_start}→{val_end} …")
        sweep_df, best = run_fold_sweep(ohlcv,cfg_f,costs,ic_weights,val_start,val_end,fold_n+1)
        best_cfg = best["cfg"]
        sweep_df["fold"] = fold_n+1
        all_sweeps.append(sweep_df)

        # Evaluate on test (OOS)
        res_test = backtest(ohlcv, best_cfg, costs, label=f"FOLD{fold_n+1}_TEST")
        r_test   = res_test["returns_net"].loc[test_start:test_end]
        eq_test  = best_cfg.capital_initial*(1.0+r_test).cumprod()
        exp_test = res_test["exposure"].loc[r_test.index]
        s_test   = summarize(r_test,eq_test,exp_test,res_test["turnover"].loc[r_test.index],best_cfg,f"FOLD{fold_n+1}_TEST")

        qqq_test = res_test["bench"]["QQQ_r"].loc[test_start:test_end]
        s_qqq    = summarize(qqq_test,best_cfg.capital_initial*(1.0+qqq_test).cumprod(),None,None,best_cfg,"QQQ")
        alpha_nw_fold = alpha_test_nw(r_test,qqq_test,best_cfg,f"fold{fold_n+1}_test")

        top_row = sweep_df.iloc[0]
        fold_results.append({
            "fold": fold_n+1,
            "train": f"{train_start}→{train_end}",
            "val":   f"{val_start}→{val_end}",
            "test":  f"{test_start}→{test_end}",
            "best_weight_cap":      best_cfg.weight_cap,
            "best_k_atr":           best_cfg.k_atr,
            "best_turb_thr":        best_cfg.turb_zscore_thr,
            "best_turb_min":        best_cfg.turb_scale_min,
            "best_vol_tgt":         best_cfg.vol_target_ann,
            "val_score":            round(best["score"],4),
            "val_sharpe":           round(best["s_val"]["Sharpe"],4),
            "val_stat_label":       str(top_row.get("stat_label","—")),
            "test_CAGR%":           round(s_test["CAGR"]*100,2),
            "test_Sharpe":          round(s_test["Sharpe"],4),
            "test_MaxDD%":          round(s_test["MaxDD"]*100,2),
            "test_Calmar":          round(s_test["Calmar"],4),
            "test_alpha_ann%":      round(alpha_nw_fold.get("alpha_ann",np.nan)*100,2),
            "test_t_alpha":         round(alpha_nw_fold.get("t_alpha",np.nan),3),
            "AvgExposure":          round(s_test["AvgExposure"]*100,1),
        })
        all_oos_r.append(r_test)
        final_cfg = best_cfg

        print(f"  [fold {fold_n+1}] test Sharpe={s_test['Sharpe']:.3f}  "
              f"CAGR={s_test['CAGR']*100:.1f}%  MaxDD={s_test['MaxDD']*100:.1f}%  "
              f"α_ann={alpha_nw_fold.get('alpha_ann',0)*100:.1f}% (t={alpha_nw_fold.get('t_alpha',0):.2f})")

    # Stitch OOS returns
    oos_r  = pd.concat(all_oos_r).sort_index()
    oos_r  = oos_r[~oos_r.index.duplicated()]
    oos_eq = cfg_base.capital_initial * (1.0+oos_r).cumprod()

    return oos_r, oos_eq, fold_results, pd.concat(all_sweeps,ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — ROBUSTNESS SUITE
# ═══════════════════════════════════════════════════════════════════════════════

def cost_gap_stress(ohlcv, cfg: Mahoraga11Config, base_costs: CostsConfig) -> pd.DataFrame:
    """
    5 cost scenarios × full backtest period.
    Scenarios: base / ×2 costs / ×3 costs / light-gap / heavy-gap.
    """
    scenarios = {
        "BASE":        CostsConfig(commission=base_costs.commission, slippage=base_costs.slippage),
        "COST×2":      CostsConfig(commission=base_costs.commission*2, slippage=base_costs.slippage*2),
        "COST×3":      CostsConfig(commission=base_costs.commission*3, slippage=base_costs.slippage*3),
        "GAP_LIGHT":   CostsConfig(commission=base_costs.commission, slippage=base_costs.slippage, gap_factor=1.5),
        "GAP_HEAVY":   CostsConfig(commission=base_costs.commission, slippage=base_costs.slippage, gap_factor=2.5),
    }
    rows=[]
    for name, c in scenarios.items():
        res = backtest(ohlcv, cfg, c, label=name)
        r   = res["returns_net"]
        eq  = res["equity"]
        s   = summarize(r, eq, res["exposure"], res["turnover"], cfg, name)
        rows.append({"Scenario":name,"CAGR%":round(s["CAGR"]*100,2),
                     "Sharpe":round(s["Sharpe"],3),"MaxDD%":round(s["MaxDD"]*100,2),
                     "Calmar":round(s["Calmar"],3),"TurnAnn":round(s["TurnoverAnn"],2),
                     "FinalEq":round(s["FinalEquity"],0)})
        print(f"  [cost_stress] {name:<12} Sharpe={s['Sharpe']:.3f}  CAGR={s['CAGR']*100:.1f}%  DD={s['MaxDD']*100:.1f}%")
    return pd.DataFrame(rows)


def alternate_universe_stress(
    ohlcv_full: Dict,
    cfg_base:   Mahoraga11Config,
    costs:      CostsConfig,
    universes:  Dict[str, Tuple[str,...]] = ALTERNATE_UNIVERSES,
) -> pd.DataFrame:
    """
    Run strategy on each alternate universe.
    Only uses tickers available in ohlcv_full; skips missing ones with a warning.
    """
    rows=[]
    for uname, tickers in universes.items():
        avail = [t for t in tickers if t in ohlcv_full["close"].columns]
        if len(avail) < 4:
            print(f"  [alt_univ] {uname}: only {len(avail)} tickers available, skipping")
            continue
        c2          = deepcopy(cfg_base)
        c2.universe = tuple(avail)
        print(f"  [alt_univ] {uname}: {len(avail)} tickers …")
        try:
            res = backtest(ohlcv_full, c2, costs, label=uname)
            r   = res["returns_net"]
            eq  = res["equity"]
            s   = summarize(r,eq,res["exposure"],res["turnover"],c2,uname)
            rows.append({"Universe":uname,"N_tickers":len(avail),
                         "CAGR%":round(s["CAGR"]*100,2),"Sharpe":round(s["Sharpe"],3),
                         "MaxDD%":round(s["MaxDD"]*100,2),"Calmar":round(s["Calmar"],3),
                         "FinalEq":round(s["FinalEquity"],0)})
        except Exception as e:
            print(f"  [alt_univ] {uname}: ERROR {e}")
    return pd.DataFrame(rows)


def local_sensitivity(
    ohlcv:   Dict,
    cfg_win: Mahoraga11Config,
    costs:   CostsConfig,
    param_a: str = "vol_target_ann",
    param_b: str = "weight_cap",
    grid_a:  Tuple = (0.20, 0.25, 0.30, 0.35, 0.40),
    grid_b:  Tuple = (0.45, 0.50, 0.55, 0.60, 0.65),
    period_start: str = "2020-01-01",
    period_end:   str = "2026-02-20",
) -> pd.DataFrame:
    """
    5×5 Sharpe surface around winning config for two most influential parameters.
    Evaluated on TEST period (same period used for final evaluation).
    """
    rows=[]
    for va in grid_a:
        for vb in grid_b:
            c2 = deepcopy(cfg_win)
            setattr(c2, param_a, va); setattr(c2, param_b, vb)
            res = backtest(ohlcv, c2, costs, label="sens")
            r   = res["returns_net"].loc[period_start:period_end]
            eq  = cfg_win.capital_initial*(1.0+r).cumprod()
            s   = summarize(r,eq,res["exposure"].loc[r.index],res["turnover"].loc[r.index],c2)
            rows.append({param_a:va, param_b:vb,
                         "Sharpe":round(s["Sharpe"],4),
                         "CAGR%":round(s["CAGR"]*100,2),
                         "MaxDD%":round(s["MaxDD"]*100,2)})
    return pd.DataFrame(rows)


def stop_keep_cash_ablation(
    ohlcv:   Dict,
    cfg_win: Mahoraga11Config,
    costs:   CostsConfig,
) -> pd.DataFrame:
    """Compare stop_keep_cash=True (research) vs False (aggressive)."""
    rows=[]
    for mode, skc in [("RESEARCH (keep_cash=True)", True), ("AGGRESSIVE (keep_cash=False)", False)]:
        c2 = deepcopy(cfg_win); c2.stop_keep_cash = skc
        res = backtest(ohlcv, c2, costs, label=mode)
        r   = res["returns_net"]; eq=res["equity"]
        s   = summarize(r,eq,res["exposure"],res["turnover"],c2,mode)
        rows.append({"Mode":mode,"CAGR%":round(s["CAGR"]*100,2),
                     "Sharpe":round(s["Sharpe"],3),"MaxDD%":round(s["MaxDD"]*100,2),
                     "Calmar":round(s["Calmar"],3),"AvgExp%":round(s["AvgExposure"]*100,1)})
        print(f"  [stop_ablation] {mode}: Sharpe={s['Sharpe']:.3f}  MaxDD={s['MaxDD']*100:.1f}%")
    return pd.DataFrame(rows)


def ic_weight_ablation(ohlcv, cfg_win: Mahoraga11Config, costs: CostsConfig) -> pd.DataFrame:
    """IC-derived weights vs fixed equal weights (1/3, 1/3, 1/3)."""
    rows=[]
    for mode, (wt,wm,wr) in [
        (f"IC_weights ({cfg_win.w_trend:.2f}/{cfg_win.w_mom:.2f}/{cfg_win.w_rel:.2f})",
         (cfg_win.w_trend, cfg_win.w_mom, cfg_win.w_rel)),
        ("EQUAL_WEIGHTS (0.33/0.33/0.33)", (1/3, 1/3, 1/3)),
    ]:
        c2 = deepcopy(cfg_win); c2.w_trend=wt; c2.w_mom=wm; c2.w_rel=wr
        res = backtest(ohlcv,c2,costs,label=mode)
        r=res["returns_net"]; eq=res["equity"]
        s=summarize(r,eq,res["exposure"],res["turnover"],c2,mode)
        rows.append({"Mode":mode,"CAGR%":round(s["CAGR"]*100,2),
                     "Sharpe":round(s["Sharpe"],3),"MaxDD%":round(s["MaxDD"]*100,2),
                     "Calmar":round(s["Calmar"],3)})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig, path):
    fig.tight_layout(); fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] {path}")

def _colors(): return ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]

def plot_equity(curves: Dict[str,pd.Series], title: str, path: str):
    fig,ax=plt.subplots(figsize=(14,6)); c=_colors()
    for i,(k,s) in enumerate(curves.items()):
        ax.plot(to_s(s).dropna().index, to_s(s).dropna().values,
                label=k, linewidth=2.0 if i==0 else 1.2,
                linestyle=["-","--","-.",":",(0,(3,1,1,1))][i%5], color=c[i%6])
    ax.set_yscale("log"); ax.set_title(title,fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Equity (log)"); ax.legend(); ax.grid(alpha=.3)
    _save(fig,path)

def plot_drawdown(curves: Dict[str,pd.Series], title: str, path: str):
    fig,ax=plt.subplots(figsize=(14,5)); c=_colors()
    for i,(k,eq) in enumerate(curves.items()):
        eq=to_s(eq).dropna(); dd=eq/eq.cummax()-1.0
        ax.fill_between(dd.index,dd.values,0,alpha=.25 if i==0 else .12,color=c[i%6])
        ax.plot(dd.index,dd.values,label=k,linewidth=1.5 if i==0 else .8,color=c[i%6])
    ax.axhline(-0.30,color="red",ls="--",lw=.8,label="-30% threshold")
    ax.set_title(title,fontweight="bold"); ax.set_xlabel("Date"); ax.set_ylabel("Drawdown")
    ax.legend(); ax.grid(alpha=.3); _save(fig,path)

def plot_wf_oos(oos_eq: pd.Series, qqq_eq: pd.Series,
                fold_results: List[Dict], title: str, path: str):
    fig, axes = plt.subplots(2,1,figsize=(14,10),sharex=True)
    ax=axes[0]
    ax.plot(to_s(oos_eq).index, to_s(oos_eq).values, label="OOS stitched", color="#1f77b4",lw=2)
    ax.plot(to_s(qqq_eq).index, to_s(qqq_eq).values, label="QQQ",           color="#ff7f0e",lw=1.2,ls="--")
    ax.set_yscale("log"); ax.set_title(title,fontweight="bold"); ax.legend(); ax.grid(alpha=.3)
    ax.set_ylabel("Equity (log)")
    # Mark fold boundaries
    for fd in fold_results:
        ts = fd["test"].split("→")[0]
        ax.axvline(pd.Timestamp(ts), color="gray", lw=0.5, ls=":")
    ax2=axes[1]
    fold_sharpes = [f["test_Sharpe"] for f in fold_results]
    fold_labels  = [f"F{f['fold']}" for f in fold_results]
    bars = ax2.bar(fold_labels, fold_sharpes, color=["#2ca02c" if s>0 else "#d62728" for s in fold_sharpes])
    ax2.axhline(0,color="black",lw=.5); ax2.axhline(1.0,color="green",lw=.8,ls="--",label="Sharpe=1.0")
    ax2.set_title("Test Sharpe by Fold",fontsize=11); ax2.set_ylabel("Sharpe"); ax2.legend(); ax2.grid(alpha=.3,axis="y")
    _save(fig,path)

def plot_risk_overlays(res: Dict, title: str, path: str):
    fig,axes=plt.subplots(4,1,figsize=(14,12),sharex=True)
    for ax,(s,c,lbl) in zip(axes,[
        (res["exposure"],    "#1f77b4","Scaled Exposure"),
        (res["vol_scale"],   "#ff7f0e","Vol-Target Scale"),
        (res["turb_scale"],  "#2ca02c","Turbulence Scale"),
        (res["crisis_scale"],"#d62728","Crisis Gate Scale"),
    ]):
        sv=to_s(s).fillna(0.0)
        ax.plot(sv.index,sv.values,color=c,lw=1.0); ax.fill_between(sv.index,sv.values,alpha=.2,color=c)
        ax.set_ylabel(lbl,fontsize=9); ax.set_ylim(-0.05,1.1); ax.grid(alpha=.3)
    axes[0].set_title(title,fontweight="bold"); _save(fig,path)

def plot_ic_multi_horizon(ic_df: pd.DataFrame, title: str, path: str):
    fig,ax=plt.subplots(figsize=(14,5))
    colors_h={"IC_composite_1d":"#d62728","IC_composite_5d":"#ff7f0e","IC_composite_21d":"#2ca02c"}
    for col in ic_df.columns:
        ax.plot(ic_df.index,ic_df[col],label=col,color=colors_h.get(col,"gray"),
                linewidth=1.8 if "21d" in col else 1.2)
    ax.axhline(0,color="black",lw=.5)
    ax.axhline(0.04,color="gray",ls="--",lw=.7,label="IC=0.04 (meaningful)")
    ax.set_title(title,fontweight="bold"); ax.set_xlabel("Date"); ax.set_ylabel("Rolling 63d IC")
    ax.legend(); ax.grid(alpha=.3); _save(fig,path)

def plot_sharpe_surface(sens_df: pd.DataFrame, param_a: str, param_b: str, title: str, path: str):
    piv = sens_df.pivot(index=param_a, columns=param_b, values="Sharpe")
    fig,ax=plt.subplots(figsize=(8,6))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn",
                   vmin=max(0, piv.values.min()-0.1), vmax=piv.values.max()+0.1)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{v:.2f}" for v in piv.columns])
    ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels([f"{v:.2f}" for v in piv.index])
    ax.set_xlabel(param_b); ax.set_ylabel(param_a)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            ax.text(j,i,f"{piv.values[i,j]:.2f}",ha="center",va="center",fontsize=9,
                    color="black" if 0.3<im.norm(piv.values[i,j])<0.7 else "white")
    plt.colorbar(im,ax=ax,label="Sharpe"); ax.set_title(title,fontweight="bold")
    _save(fig,path)

def plot_regime_bars(regime_df: pd.DataFrame, title: str, path: str):
    if regime_df.empty: return
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,m,c in zip(axes,["CAGR%","Sharpe","MaxDD%"],["#1f77b4","#ff7f0e","#d62728"]):
        vals=regime_df.set_index("Regime")[m]
        vals.plot(kind="bar",ax=ax,color=c,alpha=0.8)
        ax.set_title(f"{m} by Regime",fontsize=11)
        ax.set_xticklabels(vals.index,rotation=20,ha="right",fontsize=8)
        ax.axhline(0,color="black",lw=.5); ax.grid(alpha=.3,axis="y")
    fig.suptitle(title,fontsize=13,fontweight="bold"); _save(fig,path)

def plot_signal_decomp(decomp: Dict, bench_r: pd.Series, cfg: Mahoraga11Config,
                       title: str, path: str):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
    clr={"TREND_ONLY":"#1f77b4","MOM_ONLY":"#ff7f0e","REL_ONLY":"#2ca02c"}
    shs={}
    for comp,res in decomp.items():
        eq=to_s(res["equity"]).dropna()
        ax1.plot(eq.index,eq.values,label=comp,color=clr[comp],lw=1.2)
        shs[comp]=sharpe(res["returns_net"],cfg.rf_annual,cfg.trading_days)
    beq=cfg.capital_initial*(1.0+to_s(bench_r).fillna(0.0)).cumprod()
    ax1.plot(beq.index,beq.values,label="QQQ",color="gray",lw=.8,ls="--")
    ax1.set_yscale("log"); ax1.legend(); ax1.grid(alpha=.3); ax1.set_title("Equity (log)")
    ax2.bar(list(shs.keys()),list(shs.values()),color=[clr[k] for k in shs],alpha=.8)
    ax2.axhline(0,color="black",lw=.5); ax2.set_title("Sharpe by Component"); ax2.grid(alpha=.3,axis="y")
    fig.suptitle(title,fontweight="bold"); _save(fig,path)

def plot_weights_heatmap(weights: pd.DataFrame, title: str, path: str):
    wm=weights.resample("ME").mean()
    fig,ax=plt.subplots(figsize=(16,5))
    im=ax.imshow(wm.T.values,aspect="auto",cmap="YlOrRd",vmin=0,vmax=weights.values.max())
    ax.set_yticks(range(len(wm.columns))); ax.set_yticklabels(wm.columns,fontsize=9)
    xt=range(0,len(wm),max(1,len(wm)//20))
    ax.set_xticks(list(xt)); ax.set_xticklabels([wm.index[i].strftime("%Y-%m") for i in xt],
                                                 rotation=45,ha="right",fontsize=8)
    plt.colorbar(im,ax=ax,label="Weight"); ax.set_title(title,fontweight="bold"); _save(fig,path)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 16 — FINAL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def final_evaluation(
    ohlcv:   Dict,
    cfg_win: Mahoraga11Config,
    costs:   CostsConfig,
    ff:      Optional[pd.DataFrame],
    oos_r:   pd.Series,
    oos_eq:  pd.Series,
) -> Dict:
    """
    Full evaluation on complete date range + all analytics.
    OOS metrics come from walk-forward stitched curve (not single split).
    """
    res      = backtest(ohlcv, cfg_win, costs, label=cfg_win.label)
    validate_no_lookahead(res, cfg_win.label)

    qqq_r = res["bench"]["QQQ_r"]
    qqq_eq= res["bench"]["QQQ_eq"]
    eqw   = baseline_eqw(ohlcv,  cfg_win, costs)
    mom   = baseline_mom(ohlcv,  cfg_win, costs)

    def _bs(r,eq): return summarize(r,eq,pd.Series(1.0,index=r.index),None,cfg_win,"QQQ")

    s_full     = summarize(res["returns_net"],res["equity"],res["exposure"],res["turnover"],cfg_win,cfg_win.label)
    s_qqq_full = _bs(qqq_r,qqq_eq)
    s_eqw      = summarize(eqw["r"],eqw["eq"],eqw["exp"],eqw["to"],cfg_win,"EQW_Tech")
    s_mom      = summarize(mom["r"],mom["eq"],mom["exp"],mom["to"],cfg_win,"MOM_12_1_TopK")

    # OOS (stitched walk-forward)
    oos_eq_r   = oos_r
    s_oos      = summarize(oos_eq_r, oos_eq, None, None, cfg_win, "OOS_WalkForward")
    qqq_oos_r  = qqq_r.reindex(oos_r.index).fillna(0.0)
    s_qqq_oos  = _bs(qqq_oos_r, cfg_win.capital_initial*(1.0+qqq_oos_r).cumprod())

    # Statistical tests — full
    sr_ci_full = asymptotic_sharpe_ci(res["returns_net"], cfg_win)
    sr_ci_oos  = asymptotic_sharpe_ci(oos_r, cfg_win)
    alpha_full = alpha_test_nw(res["returns_net"],qqq_r,cfg_win,cfg_win.label)
    alpha_oos  = alpha_test_nw(oos_r,qqq_oos_r,cfg_win,"OOS_WalkForward")
    # Conditional alpha (exposure>0 only)
    alpha_cond = alpha_test_nw(res["returns_net"],qqq_r,cfg_win,
                               f"{cfg_win.label}_conditional",
                               conditional=True, exposure=res["exposure"])
    ff_full    = factor_attribution(res["returns_net"],ff,cfg_win,cfg_win.label)
    ff_oos     = factor_attribution(oos_r,ff,cfg_win,"OOS_WalkForward")

    # Regime & stress
    regime_full = regime_analysis(res["returns_net"],qqq_r,ohlcv,cfg_win)
    regime_oos  = regime_analysis(oos_r,qqq_oos_r,ohlcv,cfg_win)
    stress      = stress_report(res["returns_net"],res["exposure"],STRESS_EPISODES,cfg_win,qqq_r)
    stress_oos  = stress_report(oos_r,pd.Series(np.nan,index=oos_r.index),STRESS_EPISODES,cfg_win,qqq_oos_r)

    boot_full   = moving_block_bootstrap(res["returns_net"],seed=cfg_win.random_seed)
    boot_oos    = moving_block_bootstrap(oos_r,seed=cfg_win.random_seed)

    return {
        "cfg":cfg_win,"res":res,"eqw":eqw,"mom":mom,
        "full":s_full,"qqq_full":s_qqq_full,"eqw_full":s_eqw,"mom_full":s_mom,
        "oos":s_oos,"qqq_oos":s_qqq_oos,
        "sr_ci_full":sr_ci_full,"sr_ci_oos":sr_ci_oos,
        "alpha_full":alpha_full,"alpha_oos":alpha_oos,"alpha_cond":alpha_cond,
        "ff_full":ff_full,"ff_oos":ff_oos,
        "regime_full":regime_full,"regime_oos":regime_oos,
        "stress":stress,"stress_oos":stress_oos,
        "boot_full":boot_full,"boot_oos":boot_oos,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 17 — PRINT & SAVE
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(s):
    return {
        "Model":s["Label"],"FinalEq":f"{s['FinalEquity']:,.0f}",
        "CAGR%":f"{s['CAGR']*100:.2f}","Vol%":f"{s['AnnVol']*100:.2f}",
        "Sharpe":f"{s['Sharpe']:.3f}","Sortino":f"{s['Sortino']:.3f}",
        "MaxDD%":f"{s['MaxDD']*100:.2f}","Calmar":f"{s['Calmar']:.3f}",
        "AvgExp%":f"{s['AvgExposure']*100:.1f}","TurnAnn":f"{s['TurnoverAnn']:.2f}",
    }

def print_results(out, fold_results, ff):
    sep="="*110
    print(f"\n{sep}\n  MAHORAGA 1.1 — FULL RESULTS\n{sep}")

    print(f"\n{'─'*70}  FULL PERIOD (ALL AVAILABLE DATA)")
    print(pd.DataFrame([_fmt(out["full"]),_fmt(out["qqq_full"]),
                        _fmt(out["eqw_full"]),_fmt(out["mom_full"])]).to_string(index=False))

    print(f"\n{'─'*70}  OOS — STITCHED WALK-FORWARD (5 independent test years)")
    print(pd.DataFrame([_fmt(out["oos"]),_fmt(out["qqq_oos"])]).to_string(index=False))

    print(f"\n{'─'*70}  WALK-FORWARD FOLD SUMMARY")
    print(pd.DataFrame(fold_results).to_string(index=False))

    print(f"\n{'─'*70}  ASYMPTOTIC SHARPE CI (delta-method approximation)")
    for lbl,ci in [("Full",out["sr_ci_full"]),("OOS",out["sr_ci_oos"])]:
        print(f"  {lbl:6s}: SR={ci['SR']:.4f}  95%CI=[{ci['CI_lo']:.4f},{ci['CI_hi']:.4f}]  "
              f"t={ci['t_stat']:.3f}  p={ci['p_val']:.5f}  [{ci['note']}]")

    print(f"\n{'─'*70}  ALPHA — Newey-West HAC vs QQQ")
    for lbl,a in [("Full",out["alpha_full"]),("OOS",out["alpha_oos"]),("Full|exp>0",out["alpha_cond"])]:
        if "error" in a:
            print(f"  {lbl}: ERROR — {a['error']}"); continue
        sig = "***" if a["sig_1pct"] else ("**" if a["sig_5pct"] else "   ")
        cond_str = " [conditional on exposure>0]" if a.get("conditional") else ""
        print(f"  {lbl:18s}: α={a['alpha_ann']*100:.2f}%  t={a['t_alpha']:.3f}  "
              f"p={a['p_alpha']:.5f}  β={a['beta']:.4f}  R²={a['R2']:.4f}  "
              f"n={a.get('n_obs','—')}  {sig}{cond_str}")

    if out.get("ff_full") and "error" not in out["ff_full"]:
        print(f"\n{'─'*70}  FF5+UMD ATTRIBUTION")
        for lbl,fa in [("Full",out["ff_full"]),("OOS",out["ff_oos"])]:
            if fa and "error" not in fa:
                print(f"  {lbl:6s}: α={fa['alpha_ann']*100:.2f}%  t={fa['t_alpha']:.3f}  "
                      f"β_mkt={fa['beta_mkt']:.3f}  β_umd={fa['beta_umd']:.3f}  "
                      f"β_smb={fa['beta_smb']:.3f}  R²_adj={fa['R2_adj']:.3f}")

    print(f"\n{'─'*70}  REGIME ANALYSIS")
    print("  Full period:"); print(out["regime_full"].to_string(index=False))
    print("  OOS walk-forward:"); print(out["regime_oos"].to_string(index=False))

    print(f"\n{'─'*70}  STRESS EPISODES")
    print(out["stress"].to_string(index=False))

    print(f"\n{'─'*70}  BOOTSTRAP DD (moving block, 1000 samples)")
    for lbl,b in [("Full",out["boot_full"]),("OOS",out["boot_oos"])]:
        print(f"  {lbl:6s}: median_DD={b['dd_p50']*100:.1f}%  "
              f"p5_worst={b['dd_p5_worst']*100:.1f}%  "
              f"P(DD<-30%)={b['ruin_prob_30dd']:.1f}%  "
              f"P(DD<-50%)={b['ruin_prob_50dd']:.1f}%")


def save_csvs(out, fold_results, ic_df, rob, out_dir):
    _ensure_dir(out_dir)
    def _df(rows): return pd.DataFrame([{k:round(v,6) if isinstance(v,float) else v
                                          for k,v in r.items()} for r in rows])
    _df([out["full"],out["qqq_full"],out["eqw_full"],out["mom_full"]]).to_csv(
        f"{out_dir}/comparison_full.csv",index=False)
    _df([out["oos"],out["qqq_oos"]]).to_csv(f"{out_dir}/comparison_oos.csv",index=False)
    pd.DataFrame(fold_results).to_csv(f"{out_dir}/walk_forward_folds.csv",index=False)
    out["stress"].to_csv(f"{out_dir}/stress_full.csv",index=False)
    out["regime_full"].to_csv(f"{out_dir}/regime_full.csv",index=False)
    out["regime_oos"].to_csv(f"{out_dir}/regime_oos.csv",index=False)
    ic_df.to_csv(f"{out_dir}/rolling_ic_multi.csv")
    if rob.get("cost_stress") is not None: rob["cost_stress"].to_csv(f"{out_dir}/cost_stress.csv",index=False)
    if rob.get("alt_univ")   is not None: rob["alt_univ"].to_csv(f"{out_dir}/alt_universes.csv",index=False)
    if rob.get("sensitivity") is not None: rob["sensitivity"].to_csv(f"{out_dir}/local_sensitivity.csv",index=False)
    if rob.get("stop_ablation") is not None: rob["stop_ablation"].to_csv(f"{out_dir}/stop_ablation.csv",index=False)
    if rob.get("ic_ablation")   is not None: rob["ic_ablation"].to_csv(f"{out_dir}/ic_ablation.csv",index=False)
    alpha_rows=[out["alpha_full"],out["alpha_oos"],out["alpha_cond"]]
    pd.DataFrame(alpha_rows).to_csv(f"{out_dir}/alpha_nw.csv",index=False)
    sr_rows=[out["sr_ci_full"],out["sr_ci_oos"]]
    pd.DataFrame(sr_rows).to_csv(f"{out_dir}/sharpe_ci.csv",index=False)
    if out.get("ff_full"): pd.DataFrame([out["ff_full"],out.get("ff_oos",{})]).to_csv(f"{out_dir}/ff_attribution.csv",index=False)
    print(f"  [CSVs → ./{out_dir}/]")


def make_plots(out, oos_r, oos_eq, fold_results, ic_df, decomp, rob, cfg):
    p = cfg.plots_dir; _ensure_dir(p)
    res=out["res"]; eqw=out["eqw"]; mom=out["mom"]

    plot_equity({cfg.label:res["equity"],"QQQ":res["bench"]["QQQ_eq"],
                 "EQW":eqw["eq"],"MOM_12_1":mom["eq"]},
                "Full Period Equity — Mahoraga 1.1",f"{p}/01_equity_full.png")
    plot_equity({cfg.label:oos_eq,"QQQ (OOS)":res["bench"]["QQQ_eq"].reindex(oos_r.index)},
                "Walk-Forward OOS Equity — Mahoraga 1.1",f"{p}/02_equity_oos.png")
    plot_drawdown({cfg.label:res["equity"],"QQQ":res["bench"]["QQQ_eq"],"MOM_12_1":mom["eq"]},
                  "Drawdown",f"{p}/03_drawdown.png")
    plot_wf_oos(oos_eq, res["bench"]["QQQ_eq"].reindex(oos_r.index),
                fold_results,"Walk-Forward OOS by Fold",f"{p}/04_walkforward.png")
    plot_risk_overlays(res,"Risk Overlays",f"{p}/05_risk_overlays.png")
    plot_weights_heatmap(res["weights_scaled"],"Portfolio Weights (monthly avg)",f"{p}/06_weights.png")
    plot_ic_multi_horizon(ic_df,"Rolling IC — 1d / 5d / 21d horizons",f"{p}/07_ic_multi.png")
    plot_regime_bars(out["regime_full"],"Regime Analysis — Full Period",f"{p}/08_regime_full.png")
    plot_regime_bars(out["regime_oos"], "Regime Analysis — OOS",f"{p}/09_regime_oos.png")
    if decomp: plot_signal_decomp(decomp,res["bench"]["QQQ_r"],cfg,"Signal Decomposition",f"{p}/10_decomp.png")
    if rob.get("sensitivity") is not None:
        plot_sharpe_surface(rob["sensitivity"],"vol_target_ann","weight_cap",
                            "Sharpe Surface — vol_target × weight_cap",f"{p}/11_sensitivity.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 18 — MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_mahoraga11(make_plots_flag: bool = True, run_robustness: bool = True) -> Dict:
    print("="*80)
    print("  MAHORAGA 1.1 — Research Edition")
    print("="*80)

    cfg   = Mahoraga11Config()
    costs = CostsConfig()
    _ensure_dir(cfg.cache_dir)
    _ensure_dir(cfg.plots_dir)
    _ensure_dir(cfg.outputs_dir)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1] Downloading data …")
    all_tickers = (list(cfg.universe) + [cfg.bench_qqq, cfg.bench_spy, cfg.bench_vix]
                   + [t for u in ALTERNATE_UNIVERSES.values() for t in u])
    all_tickers = sorted(set(all_tickers))
    ohlcv = download_ohlcv(all_tickers, cfg.data_start, cfg.data_end, cfg.cache_dir)

    print("\n[1b] Fama-French factors …")
    ff = load_ff_factors(cfg.cache_dir)

    # ── Walk-forward ──────────────────────────────────────────────────────────
    print("\n[2] Expanding-window walk-forward (5 folds) …")
    oos_r, oos_eq, fold_results, all_sweeps = run_walk_forward(ohlcv, cfg, costs)
    oos_summary = summarize(oos_r,oos_eq,None,None,cfg,"OOS_WalkForward")
    print(f"\n  OOS Sharpe={oos_summary['Sharpe']:.3f}  "
          f"CAGR={oos_summary['CAGR']*100:.1f}%  MaxDD={oos_summary['MaxDD']*100:.1f}%")

    # Choose final config: use last fold's best (most data in train)
    # Recalibrate crisis on full train through fold-5 train_end
    last_train_end = cfg.wf_folds[-1][0]
    qqq_full = to_s(ohlcv["close"][cfg.bench_qqq].ffill())
    cfg_final = deepcopy(cfg)
    dd_thr, vol_thr = calibrate_crisis_thresholds(qqq_full, cfg.wf_train_start, last_train_end, cfg_final)
    cfg_final.crisis_dd_thr = dd_thr
    cfg_final.crisis_vol_zscore_thr = vol_thr

    # Re-fit IC weights on fold-5 train
    print("\n[3] Fitting IC weights on final train period …")
    close_univ = ohlcv["close"][list(cfg.universe)]
    qqq_tr_final = qqq_full.loc[cfg.wf_train_start:last_train_end]
    wt,wm,wr = fit_ic_weights(close_univ,qqq_tr_final,cfg_final,cfg.wf_train_start,last_train_end)
    cfg_final.w_trend=wt; cfg_final.w_mom=wm; cfg_final.w_rel=wr

    # Take modal hyperparams from fold sweeps (most frequent best config)
    for param, options in SWEEP_GRID.items():
        freq = {}
        for fd in fold_results:
            val = fd.get(f"best_{param.replace('_ann','_tgt').replace('turb_zscore_thr','turb_thr').replace('turb_scale_min','turb_min')}")
            if val is not None:
                freq[val] = freq.get(val,0)+1
        if freq:
            modal_val = max(freq, key=freq.get)
            if modal_val in options:
                setattr(cfg_final, param, modal_val)

    print(f"\n  Final config: weight_cap={cfg_final.weight_cap}  k_atr={cfg_final.k_atr}  "
          f"turb_thr={cfg_final.turb_zscore_thr}  turb_min={cfg_final.turb_scale_min}  "
          f"vol_tgt={cfg_final.vol_target_ann}")

    # ── Full evaluation ───────────────────────────────────────────────────────
    print("\n[4] Full evaluation …")
    out = final_evaluation(ohlcv, cfg_final, costs, ff, oos_r, oos_eq)

    # ── Rolling IC (multi-horizon) ────────────────────────────────────────────
    print("\n[5] Rolling IC (1d/5d/21d) …")
    ic_df = rolling_ic_multi_horizon(close_univ, qqq_full, cfg_final, window=63)

    # ── Signal decomposition ──────────────────────────────────────────────────
    print("\n[6] Signal decomposition …")
    decomp = baseline_signal_decomp(ohlcv, cfg_final, costs)

    # ── Robustness suite ──────────────────────────────────────────────────────
    rob={}
    if run_robustness:
        print("\n[7a] Cost/gap stress …")
        rob["cost_stress"] = cost_gap_stress(ohlcv, cfg_final, costs)

        print("\n[7b] Alternate universes …")
        rob["alt_univ"] = alternate_universe_stress(ohlcv, cfg_final, costs)

        print("\n[7c] Local parameter sensitivity …")
        rob["sensitivity"] = local_sensitivity(ohlcv, cfg_final, costs)

        print("\n[7d] Stop ablation (keep_cash) …")
        rob["stop_ablation"] = stop_keep_cash_ablation(ohlcv, cfg_final, costs)

        print("\n[7e] IC weight ablation …")
        rob["ic_ablation"] = ic_weight_ablation(ohlcv, cfg_final, costs)

        # Print robustness tables
        print("\n  COST/GAP STRESS:"); print(rob["cost_stress"].to_string(index=False))
        print("\n  ALTERNATE UNIVERSES:"); print(rob["alt_univ"].to_string(index=False))
        print("\n  STOP ABLATION:"); print(rob["stop_ablation"].to_string(index=False))
        print("\n  IC ABLATION:"); print(rob["ic_ablation"].to_string(index=False))

    # ── Print, save, plot ─────────────────────────────────────────────────────
    print_results(out, fold_results, ff)
    save_csvs(out, fold_results, ic_df, rob, cfg_final.outputs_dir)

    if make_plots_flag:
        print("\n[8] Generating plots …")
        make_plots(out, oos_r, oos_eq, fold_results, ic_df, decomp, rob, cfg_final)

    # ── Save sweeps ───────────────────────────────────────────────────────────
    all_sweeps.to_csv(f"{cfg_final.outputs_dir}/walk_forward_sweeps.csv",index=False)
    print(f"\n  All sweep results → {cfg_final.outputs_dir}/walk_forward_sweeps.csv")

    return {"cfg":cfg_final,"out":out,"oos_r":oos_r,"oos_eq":oos_eq,
            "fold_results":fold_results,"ic_df":ic_df,"decomp":decomp,"rob":rob}


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_mahoraga11(make_plots_flag=True, run_robustness=True)
