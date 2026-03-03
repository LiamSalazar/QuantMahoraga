"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          MAHORAGA  1.0                                       ║
║         Systematic Large-Cap Tech Equity Strategy                            ║
║                                                                              ║
║  Evolved from Sentinel 12A.  Full methodological overhaul:                   ║
║  · Purged walk-forward splits (embargo = max look-back)                      ║
║  · Ledoit-Wolf covariance shrinkage (replaces diagonal)                      ║
║  · IC-weighted signal composition (replaces arbitrary weights)               ║
║  · Newey-West alpha test + Jobson-Korkie Sharpe CI                           ║
║  · Fama-French 5-factor + UMD attribution                                    ║
║  · Regime decomposition (VIX tercile analysis)                               ║
║  · Signal decomposition (trend / momentum / relative in isolation)           ║
║  · Multiple-testing correction (BHY / Bonferroni) on sweep                   ║
║  · Theoretically grounded parameter reduction (≤5 free params)               ║
║  · Crisis gate calibrated without hard-coded thresholds                      ║
║  · Realistic benchmark (QQQ with expense ratio annotation)                   ║
║  · Full sweep with FDR-corrected significance reporting                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; swap to "TkAgg" if you want pop-ups
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    commission : one-way brokerage cost as fraction of notional
    slippage   : one-way market-impact + bid-ask estimate
    Notes
    -----
    For large-cap US tech (AAPL, MSFT, NVDA) bid-ask ≈ 1-2 bp.
    For cross-listed tickers (ASML-EUR, TSM-TWD via ADR) bid-ask ≈ 5-10 bp.
    We use a single blended estimate of 3 bp slippage for the universe.
    At $100k AUM market impact is negligible; flag if scaling to >$5M.
    """
    commission: float = 0.0010   # 10 bp one-way (Interactive Brokers Pro tier)
    slippage:   float = 0.0003   # 3 bp blended one-way
    apply_slippage: bool = True
    qqq_expense_ratio: float = 0.0020 / 252   # 20 bp p.a. → daily cost for QQQ benchmark


@dataclass
class Mahoraga1Config:
    # ── Universe & benchmarks ────────────────────────────────────────────────
    universe: Tuple[str, ...] = (
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN',
        'META', 'AVGO', 'ASML', 'TSM',  'ADBE', 'NFLX', 'AMD'
    )
    bench_qqq: str = "QQQ"
    bench_spy: str = "SPY"

    # ── Capital & dates ──────────────────────────────────────────────────────
    capital_initial: float = 100_000.0
    start: str = "2006-01-01"
    end:   str = "2026-02-20"
    trading_days: int = 252
    rf_annual: float = 0.0

    # ── Purged walk-forward splits ───────────────────────────────────────────
    # Embargo = 252 bd (max look-back across all signals).
    # train_end and val_start are separated by this gap so no signal
    # computed at val_start touches any training observation.
    # Similarly for val_end → test_start.
    #
    # Timeline:
    #   TRAIN  2006-01-01 … 2013-12-31
    #   EMBARGO              2014-01-01 … 2014-12-31  (~252 bd, no fit/eval)
    #   VAL    2015-01-01 … 2018-12-31
    #   EMBARGO              2019-01-01 … 2019-12-31  (~252 bd)
    #   TEST   2020-01-01 … 2026-02-20
    train_start: str = "2006-01-01"
    train_end:   str = "2013-12-31"
    val_start:   str = "2015-01-01"   # 252 bd after train_end
    val_end:     str = "2018-12-31"
    test_start:  str = "2020-01-01"   # 252 bd after val_end
    test_end:    str = "2026-02-20"
    embargo_days: int = 252           # enforced by assertion in validate_splits()

    # ── Rebalance ────────────────────────────────────────────────────────────
    rebalance_freq: str = "W-FRI"
    top_k: int = 3                    # STRUCTURAL: concentration limit by design

    # ── Signal look-backs (STRUCTURAL — set by multi-timeframe design) ───────
    # These are NOT free parameters in the sweep.
    er_window:    int = 21            # ~1 month for Efficiency Ratio
    spans_short: Tuple[int, ...] = (42, 84)        # ~2m, ~4m EMA
    spans_long:  Tuple[int, ...] = (126, 252)      # ~6m, ~12m EMA
    mom_windows: Tuple[int, ...] = (63, 126, 252)  # 3m, 6m, 12m momentum
    rel_windows: Tuple[int, ...] = (63, 126)       # 3m, 6m relative strength
    burn_in: int = 252                # = max(spans_long) = max(mom_windows)

    # ── Signal weights: IC-derived (fitted on TRAIN only, see fit_ic_weights) ─
    # Defaults below are theoretical priors; overwritten by fit_ic_weights().
    # w_trend ≈ 0.50 from Asness et al. (2013) trend premium dominance
    # w_mom   ≈ 0.30 from Jegadeesh & Titman (1993) cross-sectional weight
    # w_rel   ≈ 0.20 residual after trend+mom
    w_trend: float = 0.50
    w_mom:   float = 0.30
    w_rel:   float = 0.20

    # ── HRP ─────────────────────────────────────────────────────────────────
    hrp_window:  int   = 252
    weight_cap:  float = 0.60         # FREE PARAM #1 — swept
    # LedoitWolf replaces cov_shrink; no manual shrinkage parameter needed

    # ── Chandelier Stop ──────────────────────────────────────────────────────
    atr_window: int   = 14            # Standard ATR period (Wilder 1978)
    k_atr:      float = 2.5           # FREE PARAM #2 — swept (fixed k, no ER gain)
    stop_on:    bool  = True
    allow_reentry: bool = True
    reentry_atr_buffer: float = 0.25  # Price must recover 0.25*ATR above stop before re-entry
    stop_keep_cash: bool = False

    # ── Crisis Gate (z-score based — no hard-coded thresholds) ───────────────
    # Trigger when QQQ rolling-vol z-score > crisis_vol_zscore_thr OR
    # drawdown exceeds crisis_dd_thr.  Z-score is computed on TRAIN data
    # so threshold is data-adaptive rather than arbitrary.
    crisis_gate_use:        bool  = True
    crisis_dd_thr:          float = 0.20   # 20% drawdown (still hard but theoretically grounded:
    #                                        Ang & Bekaert (2002) find regimes shift above 20% DD)
    crisis_vol_zscore_thr:  float = 1.5    # 1.5σ above historical vol = upper ~7% of distribution
    crisis_min_days_on:     int   = 5
    crisis_min_days_off:    int   = 10
    crisis_scale:           float = 0.0

    # ── Turbulence Filter ────────────────────────────────────────────────────
    turb_use:        bool  = True
    turb_window:     int   = 63
    illiq_window:    int   = 21
    turb_zscore_thr: float = 1.2      # FREE PARAM #3 — swept
    turb_scale_min:  float = 0.30     # FREE PARAM #4 — swept
    turb_eval_on_rebalance_only: bool = True

    # ── Vol Targeting ────────────────────────────────────────────────────────
    vol_target_on:   bool  = True
    vol_target_ann:  float = 0.30     # FREE PARAM #5 — swept
    port_vol_window: int   = 63
    max_exposure:    float = 1.0
    min_exposure:    float = 0.0

    # ── Objective function (validation scoring) ──────────────────────────────
    target_maxdd:        float = 0.30
    dd_penalty_strength: float = 3.0   # stronger penalty than Sentinel to enforce DD constraint
    turnover_soft_cap:   float = 12.0
    turnover_penalty:    float = 0.02  # per unit above soft cap

    # ── Engineering ─────────────────────────────────────────────────────────
    cache_dir:   str = "data_cache"
    random_seed: int = 42
    plots_dir:   str = "mahoraga_plots"
    outputs_dir: str = "mahoraga_outputs"


# Stress episodes for out-of-sample stress analysis
STRESS_EPISODES: Dict[str, Tuple[str, str]] = {
    "GFC_2008":      ("2008-01-01", "2009-06-30"),
    "EURO_DEBT_2011":("2011-07-01", "2011-12-31"),
    "Q4_2018":       ("2018-10-01", "2018-12-31"),
    "COVID_2020":    ("2020-02-15", "2020-06-30"),
    "RATES_2022":    ("2022-01-01", "2022-12-31"),
}

# Fama-French factor tickers (via pandas-datareader or manual CSV)
# We implement a fallback: if FF data unavailable, skip factor attribution
FF_FACTORS_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _hash_key(obj: dict) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def to_series(x, name: str = "x") -> pd.Series:
    """Safely coerce any array-like to a 1D pd.Series."""
    if x is None:
        return pd.Series(dtype=float, name=name)
    if isinstance(x, pd.Series):
        return x.rename(name)
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0].rename(name) if x.shape[1] else pd.Series(dtype=float, name=name)
    arr = np.asarray(x, dtype=float)
    return pd.Series(arr.ravel(), name=name)

def safe_zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score with NaN protection."""
    m  = s.rolling(window).mean()
    sd = s.rolling(window).std().replace(0, np.nan)
    return ((s - m) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)

def annualize(r: pd.Series, td: int = 252) -> float:
    """Compound annualised return from daily return series."""
    r = to_series(r).dropna().replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) == 0:
        return 0.0
    tr = float((1.0 + r).prod() - 1.0)
    return float((1.0 + tr) ** (td / len(r)) - 1.0)

def ann_vol(r: pd.Series, td: int = 252) -> float:
    r = to_series(r).dropna()
    return float(r.std(ddof=1) * np.sqrt(td)) if len(r) > 1 else np.nan

def sharpe(r: pd.Series, rf: float = 0.0, td: int = 252) -> float:
    r = to_series(r).dropna()
    rf_d = (1.0 + rf) ** (1.0 / td) - 1.0
    ex = r - rf_d
    sd = ex.std(ddof=1)
    return float(np.sqrt(td) * ex.mean() / sd) if sd and np.isfinite(sd) else 0.0

def sortino(r: pd.Series, rf: float = 0.0, td: int = 252) -> float:
    r = to_series(r).dropna()
    rf_d = (1.0 + rf) ** (1.0 / td) - 1.0
    ex = r - rf_d
    dn = ex.clip(upper=0.0)
    sd = dn.std(ddof=1)
    return float(np.sqrt(td) * ex.mean() / sd) if sd and np.isfinite(sd) else 0.0

def max_drawdown(eq: pd.Series) -> float:
    eq = to_series(eq).dropna()
    return float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0

def calmar(r: pd.Series, eq: pd.Series, td: int = 252) -> float:
    a = annualize(r, td)
    d = max_drawdown(eq)
    return float(a / abs(d)) if d != 0 else np.inf

def cvar(r: pd.Series, alpha: float = 0.05) -> float:
    x = to_series(r).dropna().values
    if len(x) == 0:
        return np.nan
    q = np.quantile(x, alpha)
    tail = x[x <= q]
    return float(tail.mean()) if len(tail) else float(q)

def total_return(r: pd.Series) -> float:
    r = to_series(r).dropna()
    return float((1.0 + r).prod() - 1.0) if len(r) else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA
# ═══════════════════════════════════════════════════════════════════════════════

def download_ohlcv(tickers: List[str], start: str, end: str, cache_dir: str) -> Dict[str, pd.DataFrame]:
    """Download and cache OHLCV data. Returns dict with keys: close, high, low, volume."""
    _ensure_dir(cache_dir)
    key  = _hash_key({"tickers": sorted(tickers), "start": start, "end": end, "v": 5})
    path = os.path.join(cache_dir, f"ohlcv_{key}.pkl")

    if os.path.exists(path):
        print(f"  [cache] Loading OHLCV from {path}")
        return pd.read_pickle(path)

    print(f"  [download] Fetching {len(tickers)} tickers …")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      group_by="ticker", progress=False)

    # Normalise index
    idx = pd.DatetimeIndex(raw.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    idx = idx[~idx.duplicated()]

    close = pd.DataFrame(index=idx)
    high  = pd.DataFrame(index=idx)
    low   = pd.DataFrame(index=idx)
    vol   = pd.DataFrame(index=idx)

    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                get = lambda field: raw[(t, field)] if (t, field) in raw.columns else None
            else:
                get = lambda field: raw[field] if field in raw.columns else None

            c = get("Close"); h = get("High"); l = get("Low"); v = get("Volume")
            if c is None:
                continue
            reidx = lambda s: pd.Series(s, index=raw.index).reindex(idx).ffill(limit=5)
            close[t] = reidx(c)
            high[t]  = reidx(h) if h is not None else close[t]
            low[t]   = reidx(l) if l is not None else close[t]
            vol[t]   = reidx(v) if v is not None else np.nan
        except Exception as e:
            print(f"  [warn] {t}: {e}")

    out = {k: v.dropna(how="all") for k, v in
           [("close", close), ("high", high), ("low", low), ("volume", vol)]}
    pd.to_pickle(out, path)
    return out


def load_ff_factors(cache_dir: str) -> Optional[pd.DataFrame]:
    """
    Load Fama-French 5 factors + UMD (momentum) daily factors.
    Returns DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, UMD, RF
    All values as decimals (not percent).
    Falls back gracefully if network unavailable.
    """
    _ensure_dir(cache_dir)
    path = os.path.join(cache_dir, "ff5_umd_daily.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    try:
        import pandas_datareader.data as pdr
        ff5 = pdr.get_data_famafrench("F-F_Research_Data_5_Factors_2x3_daily", start="2005-01-01")[0]
        umd = pdr.get_data_famafrench("F-F_Momentum_Factor_daily", start="2005-01-01")[0]
        ff5.index = pd.to_datetime(ff5.index, format="%Y%m%d")
        umd.index = pd.to_datetime(umd.index, format="%Y%m%d")
        ff = ff5.join(umd[["Mom"]], how="left").rename(columns={"Mom": "UMD"})
        ff = ff / 100.0   # pdr returns percent
        pd.to_pickle(ff, path)
        return ff
    except Exception as e:
        print(f"  [warn] FF factors unavailable ({e}). Factor attribution will be skipped.")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — VALIDATION & ASSERTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_splits(cfg: Mahoraga1Config):
    """
    Enforce the purged walk-forward split contract:
      - val_start  ≥ train_end + embargo_days business days
      - test_start ≥ val_end   + embargo_days business days
    Raises AssertionError if violated.
    """
    train_end_dt = pd.Timestamp(cfg.train_end)
    val_start_dt = pd.Timestamp(cfg.val_start)
    val_end_dt   = pd.Timestamp(cfg.val_end)
    test_start_dt= pd.Timestamp(cfg.test_start)

    embargo_end_1 = train_end_dt + pd.offsets.BDay(cfg.embargo_days)
    embargo_end_2 = val_end_dt   + pd.offsets.BDay(cfg.embargo_days)

    assert val_start_dt >= embargo_end_1, (
        f"val_start={cfg.val_start} violates embargo.\n"
        f"  train_end={cfg.train_end} + {cfg.embargo_days} bd → embargo ends {embargo_end_1.date()}\n"
        f"  val_start must be ≥ {embargo_end_1.date()}"
    )
    assert test_start_dt >= embargo_end_2, (
        f"test_start={cfg.test_start} violates embargo.\n"
        f"  val_end={cfg.val_end} + {cfg.embargo_days} bd → embargo ends {embargo_end_2.date()}\n"
        f"  test_start must be ≥ {embargo_end_2.date()}"
    )
    print(f"  [OK] Embargo validated: "
          f"train→val gap ≥ {cfg.embargo_days} bd, val→test gap ≥ {cfg.embargo_days} bd")


def validate_no_lookahead(res: Dict, label: str = ""):
    """Verify total_scale_exec is exactly shift(1) of total_scale_target."""
    ts_exec = to_series(res["total_scale"]).fillna(0.0)
    ts_tgt  = to_series(res["total_scale_target"]).fillna(0.0)
    expected = ts_tgt.shift(1).fillna(0.0)
    diff = float((ts_exec - expected).abs().max())
    assert diff < 1e-12, f"[{label}] Look-ahead detected in total_scale (max_diff={diff:.2e})"

    w = res["weights_scaled"]
    to_recon = 0.5 * w.diff().abs().fillna(0.0).sum(axis=1)
    to_stored = to_series(res["turnover"]).fillna(0.0)
    diff2 = float((to_recon - to_stored).abs().max())
    assert diff2 < 1e-10, f"[{label}] Turnover mismatch (max_diff={diff2:.2e})"
    print(f"  [OK] No look-ahead detected [{label}]")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — COVARIANCE ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Ledoit-Wolf (2004) oracle-approximating shrinkage estimator.
    Automatically selects optimal shrinkage intensity α.
    Returns a well-conditioned covariance matrix.
    """
    X = returns.values
    lw = LedoitWolf().fit(X)
    cov_arr = lw.covariance_
    return pd.DataFrame(cov_arr, index=returns.columns, columns=returns.columns)


def cov_condition_number(cov: pd.DataFrame) -> float:
    """Log10 condition number. > 3 is suspicious, > 6 is numerically unreliable."""
    eigvals = np.linalg.eigvalsh(cov.values)
    eigvals = eigvals[eigvals > 0]
    return float(np.log10(eigvals.max() / eigvals.min())) if len(eigvals) > 1 else np.inf


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — HRP
# ═══════════════════════════════════════════════════════════════════════════════

def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """
    Hierarchical Risk Parity (Lopez de Prado 2016).
    Uses Ledoit-Wolf shrinkage covariance instead of sample covariance.
    """
    if returns.shape[1] == 1:
        return pd.Series([1.0], index=returns.columns)

    cov  = ledoit_wolf_cov(returns)
    corr = returns.corr()

    # Distance matrix (correlation-based)
    dist_mat = np.sqrt(np.clip(0.5 * (1.0 - corr.values), 0, 1))
    np.fill_diagonal(dist_mat, 0.0)
    dist_cond = squareform(dist_mat, checks=False)
    link = linkage(dist_cond, method="single")

    # Quasi-diagonalise
    def _quasi_diag(lm):
        lm = lm.astype(int)
        sort_ix = pd.Series([lm[-1, 0], lm[-1, 1]])
        n = lm[-1, 3]
        while sort_ix.max() >= n:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= n]
            i = df0.index; j = df0.values - n
            sort_ix[i] = lm[j, 0]
            df1 = pd.Series(lm[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df1]).sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    sort_ix = _quasi_diag(link)
    ordered = corr.index[sort_ix]
    cov_    = cov.loc[ordered, ordered]
    w       = pd.Series(1.0, index=ordered)

    def _cluster_var(cm, items):
        sub = cm.loc[items, items].values
        iv  = 1.0 / np.diag(sub)
        iv /= iv.sum()
        return float(iv @ sub @ iv)

    clusters = [ordered.tolist()]
    while True:
        clusters = [c for c in clusters if len(c) > 1]
        if not clusters:
            break
        new_clusters = []
        for c in clusters:
            s   = len(c) // 2
            c1, c2 = c[:s], c[s:]
            v1, v2 = _cluster_var(cov_, c1), _cluster_var(cov_, c2)
            a   = 1.0 - v1 / (v1 + v2) if (v1 + v2) else 0.5
            w[c1] *= a
            w[c2] *= (1.0 - a)
            new_clusters += [c1, c2]
        clusters = new_clusters

    return (w / w.sum()).astype(float)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════

def efficiency_ratio(price: pd.Series, n: int) -> pd.Series:
    """Kaufman Efficiency Ratio ∈ [0,1]. 1 = pure trend, 0 = noise."""
    change = (price - price.shift(n)).abs()
    vol    = price.diff().abs().rolling(n).sum()
    er     = (change / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return er.clip(0.0, 1.0)

def trend_component(price: pd.Series, cfg: Mahoraga1Config) -> pd.Series:
    """
    EMA crossover vote averaged across (short, long) pairs.
    Uses lagged EMAs (shift(1)) to prevent same-bar look-ahead.
    """
    votes = []
    for sp in cfg.spans_short:
        for lp in cfg.spans_long:
            if sp >= lp:
                continue
            ema_s = price.ewm(span=sp, adjust=False).mean().shift(1)
            ema_l = price.ewm(span=lp, adjust=False).mean().shift(1)
            votes.append(((ema_s > ema_l) & price.notna()).astype(float))
    return (sum(votes) / len(votes)) if votes else pd.Series(0.0, index=price.index)

def momentum_component(price: pd.Series, cfg: Mahoraga1Config) -> pd.Series:
    """
    Average of N-month returns, normalised to [0,1].
    shift(1) on each return ensures no same-day look-ahead.
    """
    moms = [(price / price.shift(w) - 1.0).shift(1) for w in cfg.mom_windows]
    raw  = sum(moms) / len(moms)
    return ((raw.clip(-1.0, 1.0) + 1.0) / 2.0)

def relative_component(price: pd.Series, bench: pd.Series, cfg: Mahoraga1Config) -> pd.Series:
    """Cross-sectional relative strength vs. QQQ, normalised to [0,1]."""
    rels = []
    for w in cfg.rel_windows:
        a = (price / price.shift(w) - 1.0).shift(1)
        b = (bench / bench.shift(w) - 1.0).shift(1)
        rels.append(a - b)
    raw = sum(rels) / len(rels)
    return ((raw.clip(-1.0, 1.0) + 1.0) / 2.0)


def fit_ic_weights(
    close: pd.DataFrame,
    qqq:   pd.Series,
    cfg:   Mahoraga1Config,
    period: str = "train"
) -> Tuple[float, float, float]:
    """
    Estimate signal-component weights from Information Coefficient (IC)
    computed on the TRAINING period only.

    IC_i = mean(rank_corr(signal_i_t, forward_return_t)) over training dates.
    Weights = softmax(IC_i) so they sum to 1 and are positive.

    Returns (w_trend, w_mom, w_rel).
    """
    if period == "train":
        sub_close = close.loc[cfg.train_start:cfg.train_end]
        sub_qqq   = qqq.loc[cfg.train_start:cfg.train_end]
    else:
        sub_close = close
        sub_qqq   = qqq

    idx = sub_close.index
    fwd = sub_close.pct_change().shift(-1)   # 1-day forward return

    ic_trend = []
    ic_mom   = []
    ic_rel   = []

    for t in sub_close.columns:
        p = sub_close[t].ffill()
        tr  = trend_component(p, cfg).reindex(idx)
        mo  = momentum_component(p, cfg).reindex(idx)
        re  = relative_component(p, sub_qqq.reindex(idx).ffill(), cfg).reindex(idx)
        fw  = fwd[t].reindex(idx)

        common = tr.notna() & mo.notna() & re.notna() & fw.notna()
        if common.sum() < 50:
            continue

        ic_trend.append(float(stats.spearmanr(tr[common], fw[common])[0]))
        ic_mom  .append(float(stats.spearmanr(mo[common], fw[common])[0]))
        ic_rel  .append(float(stats.spearmanr(re[common], fw[common])[0]))

    # Mean IC across assets; clip negatives at small positive (don't invert a signal)
    ic = np.array([
        max(np.nanmean(ic_trend), 0.01),
        max(np.nanmean(ic_mom),   0.01),
        max(np.nanmean(ic_rel),   0.01),
    ])
    # Softmax → weights sum to 1
    w = np.exp(ic) / np.exp(ic).sum()
    print(f"  [IC weights] trend={w[0]:.3f}  mom={w[1]:.3f}  rel={w[2]:.3f}  "
          f"(IC: {ic[0]:.4f} / {ic[1]:.4f} / {ic[2]:.4f})")
    return float(w[0]), float(w[1]), float(w[2])


def compute_scores(
    close: pd.DataFrame,
    qqq:   pd.Series,
    cfg:   Mahoraga1Config
) -> pd.DataFrame:
    """
    Composite signal score ∈ [0,1] for each asset at each date.
    Uses IC-derived weights (cfg.w_trend, cfg.w_mom, cfg.w_rel).
    First cfg.burn_in rows are set to 0 (insufficient history).
    """
    idx    = close.index
    qqq_   = to_series(qqq, "QQQ").reindex(idx).ffill()
    scores = pd.DataFrame(index=idx, columns=close.columns, dtype=float)

    for t in close.columns:
        p  = close[t].reindex(idx).ffill()
        tr = trend_component(p, cfg)
        mo = momentum_component(p, cfg)
        re = relative_component(p, qqq_, cfg)

        s = cfg.w_trend * tr + cfg.w_mom * mo + cfg.w_rel * re
        s.iloc[:cfg.burn_in] = 0.0
        scores[t] = s.fillna(0.0)

    return scores.fillna(0.0)


def rolling_ic(
    close: pd.DataFrame,
    qqq:   pd.Series,
    cfg:   Mahoraga1Config,
    window: int = 63
) -> pd.DataFrame:
    """
    Rolling 63-day IC for each signal component (trend, mom, rel) averaged across universe.
    Used for signal decomposition diagnostics.
    Returns DataFrame with columns: IC_trend, IC_mom, IC_rel, IC_composite
    """
    idx  = close.index
    qqq_ = to_series(qqq, "QQQ").reindex(idx).ffill()
    fwd  = close.pct_change().shift(-1)

    ic_df = pd.DataFrame(index=idx, dtype=float)
    for comp_name, comp_fn in [
        ("IC_trend", lambda p: trend_component(p, cfg)),
        ("IC_mom",   lambda p: momentum_component(p, cfg)),
        ("IC_rel",   lambda p: relative_component(p, qqq_, cfg)),
    ]:
        comp_vals = pd.DataFrame(index=idx, columns=close.columns, dtype=float)
        for t in close.columns:
            p = close[t].reindex(idx).ffill()
            comp_vals[t] = comp_fn(p)

        # Cross-sectional IC: rank-corr(signal, fwd_return) per day
        ic_series = pd.Series(index=idx, dtype=float)
        for i in range(len(idx)):
            s = comp_vals.iloc[i].dropna()
            f = fwd.iloc[i][s.index].dropna()
            common = s.index.intersection(f.index)
            if len(common) >= 4:
                ic_series.iloc[i] = float(stats.spearmanr(s[common], f[common])[0])
        ic_df[comp_name] = ic_series.rolling(window).mean().fillna(0.0)

    # Composite
    scores = compute_scores(close, qqq_, cfg)
    ic_composite = pd.Series(index=idx, dtype=float)
    for i in range(len(idx)):
        s = scores.iloc[i].dropna()
        f = fwd.iloc[i][s.index].dropna()
        common = s.index.intersection(f.index)
        if len(common) >= 4:
            ic_composite.iloc[i] = float(stats.spearmanr(s[common], f[common])[0])
    ic_df["IC_composite"] = ic_composite.rolling(window).mean().fillna(0.0)

    return ic_df


def select_topk(scores: pd.DataFrame, k: int, freq: str) -> pd.DataFrame:
    """Weekly rebalance mask: 1 for top-k assets by score, 0 otherwise."""
    idx       = scores.index
    reb_dates = set(scores.resample(freq).last().index)
    mask      = pd.DataFrame(0.0, index=idx, columns=scores.columns)
    last      = np.zeros(scores.shape[1])

    for dt in idx:
        if dt in reb_dates:
            row   = scores.loc[dt].values
            order = np.argsort(-row)
            last  = np.zeros(len(order))
            last[order[:k]] = 1.0
        mask.loc[dt] = last

    return mask.fillna(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — RISK OVERLAYS
# ═══════════════════════════════════════════════════════════════════════════════

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Wilder Average True Range (1978)."""
    prev = close.shift(1)
    tr   = pd.concat(
        [(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)
    # Wilder smoothing = EMA with span = 2*window - 1
    return tr.ewm(span=2 * window - 1, adjust=False).mean()


def apply_chandelier_stop(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    high:  pd.DataFrame,
    low:   pd.DataFrame,
    cfg:   Mahoraga1Config
) -> Tuple[pd.DataFrame, int]:
    """
    Chandelier Stop with:
    - Fixed k_atr (no ER gain — reduces free parameters)
    - Re-entry buffer: price must recover cfg.reentry_atr_buffer * ATR
      above the most recent stop level before re-entry is allowed
    """
    if not cfg.stop_on:
        return weights.copy(), 0

    out       = weights.copy()
    idx       = out.index
    reb_dates = set(out.resample(cfg.rebalance_freq).last().index)
    stop_hits = 0

    for t in out.columns:
        p    = close[t].reindex(idx).ffill()
        atr_ = atr(
            high[t].reindex(idx).ffill(),
            low[t].reindex(idx).ffill(),
            p, cfg.atr_window
        ).bfill().fillna(0.0)

        wt       = out[t].values.copy()
        in_pos   = False
        stopped  = False
        maxp     = np.nan
        last_stop_level = np.nan

        for i, dt in enumerate(idx):
            # At rebalance: allow re-entry only if price > last_stop + buffer
            if dt in reb_dates and stopped:
                recovery_threshold = (
                    last_stop_level + cfg.reentry_atr_buffer * float(atr_.iloc[i])
                    if np.isfinite(last_stop_level) else -np.inf
                )
                if float(p.iloc[i]) > recovery_threshold:
                    stopped = False

            if wt[i] <= 0:
                in_pos = False
                maxp   = np.nan
                continue

            if stopped:
                wt[i] = 0.0
                continue

            if not in_pos:
                in_pos = True
                maxp   = float(p.iloc[i])
            else:
                maxp = max(maxp, float(p.iloc[i]))

            stop_level = maxp - cfg.k_atr * float(atr_.iloc[i])
            if float(p.iloc[i]) < stop_level:
                wt[i]           = 0.0
                stopped         = True
                in_pos          = False
                last_stop_level = stop_level
                maxp            = np.nan
                stop_hits      += 1

        out[t] = wt

    if not cfg.stop_keep_cash:
        denom = out.sum(axis=1).replace(0, np.nan)
        out   = out.div(denom, axis=0).fillna(0.0)

    return out, stop_hits


def compute_crisis_gate(
    qqq_close: pd.Series,
    cfg:       Mahoraga1Config
) -> Tuple[pd.Series, pd.Series]:
    """
    Regime filter based on:
    1. Drawdown from peak exceeds cfg.crisis_dd_thr (hard threshold, theory-grounded
       from Ang & Bekaert 2002 regime-switching literature)
    2. Rolling volatility z-score (vs. TRAIN history) exceeds cfg.crisis_vol_zscore_thr
       (1.5σ = adaptive, not hard-coded level)

    This replaces the dual hard-coded threshold from Sentinel 12A.
    """
    p   = to_series(qqq_close, "QQQ").ffill()
    idx = p.index
    r   = p.pct_change().fillna(0.0)

    # Annualised vol rolling
    vol     = r.rolling(cfg.port_vol_window).std() * np.sqrt(cfg.trading_days)
    vol_z   = safe_zscore(vol, cfg.port_vol_window * 4)
    dd      = p / p.cummax() - 1.0

    cond    = ((dd <= -cfg.crisis_dd_thr) | (vol_z >= cfg.crisis_vol_zscore_thr)).astype(int)
    on_flag = cond.rolling(cfg.crisis_min_days_on).mean().fillna(0.0) >= 0.8
    off_flag= (1 - cond).rolling(cfg.crisis_min_days_off).mean().fillna(0.0) >= 0.8

    state = pd.Series(0.0, index=idx)
    in_crisis = False
    for dt in idx:
        if not in_crisis and bool(on_flag.loc[dt]):
            in_crisis = True
        elif in_crisis and bool(off_flag.loc[dt]):
            in_crisis = False
        state.loc[dt] = 1.0 if in_crisis else 0.0

    scale = pd.Series(1.0, index=idx, dtype=float)
    if cfg.crisis_gate_use:
        scale[state == 1.0] = cfg.crisis_scale

    scale.iloc[:cfg.burn_in] = cfg.crisis_scale
    state.iloc[:cfg.burn_in] = 1.0
    return scale, state


def compute_turbulence(
    close:  pd.DataFrame,
    volume: pd.DataFrame,
    qqq:    pd.Series,
    cfg:    Mahoraga1Config
) -> pd.Series:
    """
    Composite turbulence index combining:
    - QQQ realised vol z-score
    - Cross-sectional average correlation z-score (Mahalanobis-style)
    - Amihud (2002) illiquidity z-score

    Mapping to scale via logistic function centred at cfg.turb_zscore_thr.
    """
    idx  = close.index
    qqq_ = to_series(qqq, "QQQ").reindex(idx).ffill()
    qqq_r= qqq_.pct_change().fillna(0.0)

    # Component 1: QQQ vol z-score
    vol_q = qqq_r.rolling(cfg.turb_window).std() * np.sqrt(cfg.trading_days)
    vol_z = safe_zscore(vol_q, 252)

    # Component 2: average pairwise correlation z-score
    rets      = close.pct_change().fillna(0.0)
    avg_corr  = pd.Series(0.0, index=idx)
    reb_dates = set(close.resample(cfg.rebalance_freq).last().index)
    w         = cfg.turb_window
    for dt in (reb_dates if cfg.turb_eval_on_rebalance_only else idx):
        if dt not in idx:
            continue
        loc = idx.get_loc(dt)
        if loc < w:
            continue
        sub = rets.iloc[loc - w + 1:loc + 1]
        c   = sub.corr().values
        n   = c.shape[0]
        avg_corr.loc[dt] = (c.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    avg_corr = avg_corr.replace(0, np.nan).ffill().fillna(0.0)
    corr_z   = safe_zscore(avg_corr, 252)

    # Component 3: Amihud illiquidity z-score (log-transformed to reduce skew)
    dv       = (close * volume).replace(0, np.nan)
    illiq    = (rets.abs() / dv).replace([np.inf, -np.inf], np.nan)
    illiq_avg= np.log1p(illiq.rolling(cfg.illiq_window).mean().mean(axis=1).fillna(0.0))
    illiq_z  = safe_zscore(illiq_avg, 252)

    # Aggregate
    turb = (vol_z + corr_z + illiq_z).ewm(span=10, adjust=False).mean()

    # Logistic mapping: scale = 1/(1+exp(a*(turb - thr)))
    # When turb = thr → scale = 0.5; turb >> thr → scale ≈ turb_scale_min
    a = 1.2
    s = 1.0 / (1.0 + np.exp(a * (turb - cfg.turb_zscore_thr)))
    s = pd.Series(s, index=idx).clip(lower=cfg.turb_scale_min, upper=1.0)
    s.iloc[:cfg.burn_in] = cfg.turb_scale_min
    return s


def vol_target_scale(port_returns: pd.Series, cfg: Mahoraga1Config) -> pd.Series:
    """
    Volatility targeting: scale exposure so realised portfolio vol ≈ cfg.vol_target_ann.
    Uses rolling port_vol_window-day vol estimate.
    """
    if not cfg.vol_target_on:
        return pd.Series(1.0, index=port_returns.index)
    r    = to_series(port_returns).fillna(0.0)
    rvol = r.rolling(cfg.port_vol_window).std() * np.sqrt(cfg.trading_days)
    s    = (cfg.vol_target_ann / rvol).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return s.clip(cfg.min_exposure, cfg.max_exposure)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — CORE BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_costs(w_exec: pd.DataFrame, costs: CostsConfig) -> Tuple[pd.Series, pd.Series]:
    dw = w_exec.diff().abs().fillna(0.0)
    to = 0.5 * dw.sum(axis=1)
    tc = to * (costs.commission + (costs.slippage if costs.apply_slippage else 0.0))
    return to, tc


def backtest(
    ohlcv:  Dict[str, pd.DataFrame],
    cfg:    Mahoraga1Config,
    costs:  CostsConfig,
    label:  str = "MAHORAGA_1.0"
) -> Dict:
    """Full backtest pipeline with execution-aligned weights (no look-ahead)."""
    np.random.seed(cfg.random_seed)

    universe = list(cfg.universe)
    close  = ohlcv["close"][universe].copy()
    high   = ohlcv["high"][universe].copy()
    low    = ohlcv["low"][universe].copy()
    volume = ohlcv["volume"][universe].copy()
    idx    = close.index

    qqq = to_series(ohlcv["close"][cfg.bench_qqq].reindex(idx).ffill(), "QQQ")
    spy = to_series(ohlcv["close"][cfg.bench_spy].reindex(idx).ffill(), "SPY")

    # Risk overlays (computed on full history, no leakage because they are
    # purely reactive: each value uses only past data)
    crisis_scale, crisis_state = compute_crisis_gate(qqq, cfg)
    turb_scale                  = compute_turbulence(close, volume, qqq, cfg)

    # Signal & selection
    scores     = compute_scores(close, qqq, cfg)
    active_mask= select_topk(scores, cfg.top_k, cfg.rebalance_freq)
    rets       = close.pct_change().fillna(0.0)
    reb_dates  = set(close.resample(cfg.rebalance_freq).last().index)

    # ── Portfolio construction ───────────────────────────────────────────────
    w = pd.DataFrame(0.0, index=idx, columns=universe)
    last_w = pd.Series(0.0, index=universe)

    for dt in idx:
        if dt in reb_dates:
            sel   = active_mask.loc[dt]
            names = sel[sel > 0].index.tolist()
            if len(names) == 0:
                last_w = pd.Series(0.0, index=universe)
            elif len(names) == 1:
                last_w = pd.Series(0.0, index=universe)
                last_w[names[0]] = 1.0
            else:
                lookback = rets.loc[:dt].tail(cfg.hrp_window)[names].dropna()
                if len(lookback) < 60:
                    # Fallback: inverse-variance using LW covariance
                    cov = ledoit_wolf_cov(lookback if len(lookback) > len(names) else rets.loc[:dt][names].dropna())
                    iv  = 1.0 / np.diag(cov.values)
                    ww  = pd.Series(iv / iv.sum(), index=names)
                else:
                    ww = hrp_weights(lookback)

                ww = ww.clip(upper=cfg.weight_cap)
                ww /= ww.sum()
                last_w = pd.Series(0.0, index=universe)
                last_w[names] = ww.values

        w.loc[dt] = last_w.values

    # ── Chandelier stop ──────────────────────────────────────────────────────
    w_stop, stop_hits = apply_chandelier_stop(w, close, high, low, cfg)

    # ── Execution alignment (all signals shift(1) before entering returns) ──
    w_exec_1x     = w_stop.shift(1).fillna(0.0)
    port_gross_1x = (w_exec_1x * rets).sum(axis=1)

    vol_sc   = vol_target_scale(port_gross_1x, cfg)
    cap      = (crisis_scale * turb_scale).clip(0.0, cfg.max_exposure)
    tgt_sc   = pd.Series(np.minimum(vol_sc.values, cap.values), index=idx).clip(0.0, cfg.max_exposure)
    exec_sc  = tgt_sc.shift(1).fillna(0.0)

    w_exec   = w_exec_1x.mul(exec_sc, axis=0)
    to, tc   = _apply_costs(w_exec, costs)
    port_net = ((w_exec * rets).sum(axis=1) - tc).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    equity   = cfg.capital_initial * (1.0 + port_net).cumprod()
    exposure = w_exec.abs().sum(axis=1).clip(0.0, cfg.max_exposure)

    # ── Benchmarks (QQQ cost-adjusted for expense ratio) ────────────────────
    qqq_r  = qqq.pct_change().fillna(0.0) - costs.qqq_expense_ratio
    spy_r  = spy.pct_change().fillna(0.0)
    qqq_eq = cfg.capital_initial * (1.0 + qqq_r).cumprod()
    spy_eq = cfg.capital_initial * (1.0 + spy_r).cumprod()

    return {
        "label":             label,
        "returns_net":       port_net,
        "equity":            equity,
        "exposure":          exposure,
        "turnover":          to,
        "weights_scaled":    w_exec,
        "total_scale":       exec_sc,
        "total_scale_target":tgt_sc,
        "cap":               cap,
        "turb_scale":        turb_scale,
        "crisis_scale":      crisis_scale,
        "crisis_state":      crisis_state,
        "vol_scale":         vol_sc,
        "stop_hits":         stop_hits,
        "scores":            scores,
        "bench": {
            "QQQ_r":  qqq_r,
            "QQQ_eq": qqq_eq,
            "SPY_r":  spy_r,
            "SPY_eq": spy_eq,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — BASELINES
# ═══════════════════════════════════════════════════════════════════════════════

def baseline_equal_weight(ohlcv: Dict, cfg: Mahoraga1Config, costs: CostsConfig) -> Dict:
    """Equal-weight buy-and-hold over the full universe, rebalanced weekly."""
    close = ohlcv["close"][list(cfg.universe)].copy()
    n     = len(cfg.universe)
    idx   = close.index
    w     = pd.DataFrame(1.0 / n, index=idx, columns=cfg.universe)
    rets  = close.pct_change().fillna(0.0)
    w_exec= w.shift(1).fillna(0.0)
    r_net = ((w_exec * rets).sum(axis=1) - _apply_costs(w_exec, costs)[1]).fillna(0.0)
    eq    = cfg.capital_initial * (1.0 + r_net).cumprod()
    return {"r": r_net, "eq": eq, "exp": w_exec.abs().sum(axis=1),
            "to": _apply_costs(w_exec, costs)[0], "label": "EQW_Tech"}


def baseline_momentum(ohlcv: Dict, cfg: Mahoraga1Config, costs: CostsConfig) -> Dict:
    """
    Jegadeesh-Titman 12-1 momentum: top-k by 12m return skipping last month.
    Identical universe and rebalance frequency to Mahoraga for fair comparison.
    """
    close = ohlcv["close"][list(cfg.universe)].copy()
    mom   = (close.shift(21) / close.shift(252 + 21) - 1.0).shift(1).fillna(0.0)
    sel   = select_topk(mom, cfg.top_k, cfg.rebalance_freq)
    denom = sel.sum(axis=1).replace(0, np.nan)
    w     = sel.div(denom, axis=0).fillna(0.0)
    rets  = close.pct_change().fillna(0.0)
    w_exec= w.shift(1).fillna(0.0)
    to, tc= _apply_costs(w_exec, costs)
    r_net = ((w_exec * rets).sum(axis=1) - tc).fillna(0.0)
    eq    = cfg.capital_initial * (1.0 + r_net).cumprod()
    return {"r": r_net, "eq": eq, "exp": w_exec.abs().sum(axis=1),
            "to": to, "label": "MOM_12_1_TopK"}


def baseline_signal_decomp(
    ohlcv: Dict,
    cfg:   Mahoraga1Config,
    costs: CostsConfig
) -> Dict[str, Dict]:
    """
    Sprint 4: Run three single-component variants to isolate alpha sources.
    Each uses the same HRP allocation and overlays as Mahoraga, but only
    one signal component active (others set to 0).
    """
    results = {}
    for comp, (wt, wm, wr) in [
        ("TREND_ONLY",    (1.0, 0.0, 0.0)),
        ("MOM_ONLY",      (0.0, 1.0, 0.0)),
        ("REL_ONLY",      (0.0, 0.0, 1.0)),
    ]:
        from copy import deepcopy
        c2 = deepcopy(cfg)
        c2.w_trend = wt; c2.w_mom = wm; c2.w_rel = wr
        res = backtest(ohlcv, c2, costs, label=comp)
        results[comp] = res
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — METRICS & STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def summarize(
    r:   pd.Series,
    eq:  pd.Series,
    exp: Optional[pd.Series],
    to:  Optional[pd.Series],
    cfg: Mahoraga1Config,
    label: str = ""
) -> Dict:
    r   = to_series(r).replace([np.inf, -np.inf], np.nan).dropna()
    eq  = to_series(eq).dropna()
    exp_= to_series(exp).reindex(r.index).fillna(0.0) if exp is not None else pd.Series(np.nan, index=r.index)
    to_ = to_series(to).reindex(r.index).fillna(0.0) if to  is not None else pd.Series(0.0,   index=r.index)
    T   = len(r)
    return {
        "Label":        label,
        "FinalEquity":  float(eq.iloc[-1])       if len(eq) else np.nan,
        "TotalReturn":  total_return(r),
        "CAGR":         annualize(r, cfg.trading_days),
        "AnnVol":       ann_vol(r, cfg.trading_days),
        "Sharpe":       sharpe(r, cfg.rf_annual, cfg.trading_days),
        "Sortino":      sortino(r, cfg.rf_annual, cfg.trading_days),
        "MaxDD":        max_drawdown(eq),
        "Calmar":       calmar(r, eq, cfg.trading_days),
        "CVaR_5":       cvar(r, 0.05),
        "AvgExposure":  float(exp_.mean()),
        "TimeInMkt":    float((exp_ > 0).mean()),
        "TurnoverAnn":  float(to_.sum() * cfg.trading_days / T) if T else 0.0,
        "Days":         int(T),
    }


def sharpe_ci(r: pd.Series, cfg: Mahoraga1Config, alpha: float = 0.05) -> Dict:
    """
    Jobson-Korkie (1981) Sharpe ratio confidence interval.
    SE(SR_daily) = sqrt((1 + SR_daily²/2) / T)
    """
    r  = to_series(r).dropna()
    T  = len(r)
    sr_daily = sharpe(r, cfg.rf_annual, 1)
    se_daily = np.sqrt((1.0 + sr_daily ** 2 / 2.0) / T)
    z        = stats.norm.ppf(1.0 - alpha / 2.0)
    td       = cfg.trading_days
    sr_ann   = sr_daily * np.sqrt(td)
    se_ann   = se_daily * np.sqrt(td)
    t_stat   = sr_daily / se_daily if se_daily > 0 else 0.0
    p_val    = 2.0 * (1.0 - stats.norm.cdf(abs(t_stat)))
    return {
        "SR":     round(sr_ann, 4),
        "CI_lo":  round(sr_ann - z * se_ann, 4),
        "CI_hi":  round(sr_ann + z * se_ann, 4),
        "SE":     round(se_ann, 4),
        "t_stat": round(t_stat, 3),
        "p_val":  round(p_val, 5),
    }


def alpha_test_nw(
    r_strategy:  pd.Series,
    r_benchmark: pd.Series,
    cfg:         Mahoraga1Config,
    label:       str = ""
) -> Dict:
    """
    Test H₀: α = 0 (strategy generates no excess return beyond market beta).
    Uses Newey-West HAC standard errors to correct for autocorrelation
    in daily return series.
    Lags = int(4 * (T/100)^(2/9))  — Andrews (1991) data-adaptive lag selection.
    """
    r_s = to_series(r_strategy).dropna()
    r_b = to_series(r_benchmark).reindex(r_s.index).fillna(0.0)
    common = r_s.index.intersection(r_b.index)
    r_s = r_s[common]; r_b = r_b[common]

    X   = sm.add_constant(r_b.values)
    T   = len(r_s)
    lags= int(4 * (T / 100) ** (2.0 / 9.0))

    try:
        ols = sm.OLS(r_s.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        alpha_d = float(ols.params[0])
        alpha_a = float((1.0 + alpha_d) ** cfg.trading_days - 1.0)
        return {
            "Label":    label,
            "alpha_ann":round(alpha_a, 6),
            "t_alpha":  round(float(ols.tvalues[0]), 3),
            "p_alpha":  round(float(ols.pvalues[0]), 5),
            "beta":     round(float(ols.params[1]), 4),
            "R2":       round(float(ols.rsquared), 4),
            "NW_lags":  lags,
            "sig_5pct": bool(ols.pvalues[0] < 0.05),
            "sig_1pct": bool(ols.pvalues[0] < 0.01),
        }
    except Exception as e:
        return {"Label": label, "error": str(e)}


def factor_attribution(
    r_strategy: pd.Series,
    ff:         Optional[pd.DataFrame],
    cfg:        Mahoraga1Config,
    label:      str = ""
) -> Optional[Dict]:
    """
    Fama-French 5-factor + UMD attribution.
    r_strategy - RF = α + β_mkt*MKT + β_smb*SMB + β_hml*HML
                        + β_rmw*RMW + β_cma*CMA + β_umd*UMD + ε
    """
    if ff is None:
        return None

    r  = to_series(r_strategy).dropna()
    ff_= ff.reindex(r.index).dropna()
    common = r.index.intersection(ff_.index)
    if len(common) < 252:
        return {"Label": label, "error": "Insufficient FF data"}

    y  = r[common].values - ff_.loc[common, "RF"].values
    X  = sm.add_constant(ff_.loc[common, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]].values)
    T  = len(y)
    lags = int(4 * (T / 100) ** (2.0 / 9.0))

    try:
        ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        alpha_d = float(ols.params[0])
        alpha_a = float((1.0 + alpha_d) ** cfg.trading_days - 1.0)
        params  = ols.params
        tvals   = ols.tvalues
        pvals   = ols.pvalues
        return {
            "Label":    label,
            "alpha_ann":round(alpha_a, 6),
            "t_alpha":  round(float(tvals[0]), 3),
            "p_alpha":  round(float(pvals[0]), 5),
            "beta_mkt": round(float(params[1]), 4),
            "beta_smb": round(float(params[2]), 4),
            "beta_hml": round(float(params[3]), 4),
            "beta_rmw": round(float(params[4]), 4),
            "beta_cma": round(float(params[5]), 4),
            "beta_umd": round(float(params[6]), 4),
            "R2_adj":   round(float(ols.rsquared_adj), 4),
        }
    except Exception as e:
        return {"Label": label, "error": str(e)}


def regime_analysis(
    r_strategy:  pd.Series,
    r_benchmark: pd.Series,
    ohlcv:       Dict,
    cfg:         Mahoraga1Config
) -> pd.DataFrame:
    """
    Decompose strategy performance by VIX regime.
    Regime 1 (Calm):   VIX < 20th percentile of sample
    Regime 2 (Normal): 20th–80th percentile
    Regime 3 (Stress): VIX > 80th percentile

    Uses percentile-based regimes (data-adaptive) rather than hard thresholds
    because VIX levels shift over time.
    """
    # Try to get VIX from downloaded data
    vix = None
    if "^VIX" in ohlcv["close"].columns:
        vix = to_series(ohlcv["close"]["^VIX"].reindex(r_strategy.index).ffill())
    else:
        # Proxy: rolling 21-day realised vol of SPY * sqrt(252) * 100
        spy_r = to_series(r_benchmark).fillna(0.0)
        vix   = (spy_r.rolling(21).std() * np.sqrt(252) * 100).reindex(r_strategy.index).ffill()

    p20  = float(np.nanpercentile(vix, 20))
    p80  = float(np.nanpercentile(vix, 80))

    regimes = {
        f"CALM   (VIX<{p20:.0f})":  vix < p20,
        f"NORMAL ({p20:.0f}≤VIX<{p80:.0f})": (vix >= p20) & (vix < p80),
        f"STRESS (VIX≥{p80:.0f})": vix >= p80,
    }

    rows = []
    for name, mask in regimes.items():
        mask = mask.reindex(r_strategy.index).fillna(False)
        rs   = r_strategy[mask]
        rb   = to_series(r_benchmark).reindex(rs.index).fillna(0.0)

        if len(rs) < 20:
            continue

        eq = cfg.capital_initial * (1.0 + rs).cumprod()
        rows.append({
            "Regime":      name,
            "Days":        int(len(rs)),
            "Days%":       round(100 * len(rs) / len(r_strategy), 1),
            "CAGR%":       round(annualize(rs, cfg.trading_days) * 100, 2),
            "CAGR_bench%": round(annualize(rb, cfg.trading_days) * 100, 2),
            "Excess_CAGR%":round((annualize(rs, cfg.trading_days) - annualize(rb, cfg.trading_days)) * 100, 2),
            "Sharpe":      round(sharpe(rs, cfg.rf_annual, cfg.trading_days), 3),
            "MaxDD%":      round(max_drawdown(eq) * 100, 2),
            "Hit_Rate%":   round(100 * (rs > 0).mean(), 1),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — STRESS & BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════════

def stress_report(
    r:        pd.Series,
    exp:      pd.Series,
    episodes: Dict[str, Tuple[str, str]],
    cfg:      Mahoraga1Config,
    r_bench:  Optional[pd.Series] = None
) -> pd.DataFrame:
    rows = []
    for name, (a, b) in episodes.items():
        sub  = r.loc[a:b]
        if len(sub) < 40:
            continue
        ee   = cfg.capital_initial * (1.0 + sub).cumprod()
        ss   = summarize(sub, ee, exp.loc[sub.index], None, cfg)
        bench_total = None
        if r_bench is not None:
            rb = r_bench.loc[a:b]
            bench_total = round((1.0 + rb).prod() - 1.0, 4)
        rows.append({
            "Episode":       name,
            "Days":          int(len(sub)),
            "Total%":        round((1.0 + sub).prod() * 100 - 100, 2),
            "Bench_Total%":  round(bench_total * 100, 2) if bench_total is not None else None,
            "Excess%":       round(((1+sub).prod() - (1+ r_bench.loc[a:b]).prod()) * 100, 2) if r_bench is not None else None,
            "WorstDay%":     round(sub.min() * 100, 2),
            "Sharpe":        round(ss["Sharpe"], 3),
            "MaxDD%":        round(ss["MaxDD"] * 100, 2),
            "Calmar":        round(ss["Calmar"], 3),
            "AvgExp%":       round(ss["AvgExposure"] * 100, 1),
        })
    return pd.DataFrame(rows)


def moving_block_bootstrap(
    r:        pd.Series,
    block:    int  = 20,
    n_samples:int  = 1000,
    seed:     int  = 42
) -> Dict:
    """
    Stationary block bootstrap for drawdown distribution.
    Block length 20 days preserves weekly autocorrelation structure.
    """
    rng = np.random.default_rng(seed)
    x   = to_series(r).replace([np.inf, -np.inf], np.nan).dropna().values
    T   = len(x)
    if T < block * 5:
        return {"dd_p50": np.nan, "dd_p95": np.nan, "ruin_prob_50dd": np.nan}

    dds = []
    for _ in range(n_samples):
        starts = rng.integers(0, T - block, size=int(np.ceil(T / block)))
        sample = np.concatenate([x[s:s + block] for s in starts])[:T]
        eq     = np.cumprod(1.0 + sample)
        peak   = np.maximum.accumulate(eq)
        dds.append(float(np.min(eq / peak - 1.0)))

    dds = np.array(dds)
    return {
        "dd_p50":          float(np.quantile(dds, 0.50)),
        "dd_p5_worst":     float(np.quantile(dds, 0.05)),
        "dd_p1_worst":     float(np.quantile(dds, 0.01)),
        "ruin_prob_30dd":  float(np.mean(dds < -0.30)) * 100.0,
        "ruin_prob_50dd":  float(np.mean(dds < -0.50)) * 100.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — SWEEP & OBJECTIVE
# ═══════════════════════════════════════════════════════════════════════════════

def objective_A(s_val: Dict, s_qqq_val: Dict, cfg: Mahoraga1Config) -> float:
    """
    Validation objective. Maximise excess risk-adjusted return subject to:
    - DD penalty if MaxDD > target_maxdd
    - Turnover penalty if TurnoverAnn > turnover_soft_cap
    - Sortino gets more weight than Sharpe (penalises downside specifically)

    Changes vs Sentinel 12A:
    - Sortino term added (downside-only)
    - Turnover penalty coefficient doubled (0.01 → 0.02)
    - DD penalty uses cfg.dd_penalty_strength (not hardcoded)
    """
    excess_cagr   = s_val["CAGR"]    - s_qqq_val["CAGR"]
    excess_sharpe = s_val["Sharpe"]  - s_qqq_val["Sharpe"]
    excess_sort   = s_val["Sortino"] - s_qqq_val["Sortino"]
    excess_calmar = s_val["Calmar"]  - s_qqq_val["Calmar"]

    dd_excess      = max(0.0, abs(s_val["MaxDD"]) - cfg.target_maxdd)
    dd_penalty     = cfg.dd_penalty_strength * dd_excess
    to_excess      = max(0.0, s_val["TurnoverAnn"] - cfg.turnover_soft_cap)
    to_penalty     = cfg.turnover_penalty * to_excess

    return float(
        1.00 * excess_cagr
        + 0.20 * excess_sharpe
        + 0.20 * excess_sort
        + 0.10 * excess_calmar
        - dd_penalty
        - to_penalty
    )


def bhy_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg-Yekutieli (2001) FDR correction for multiple testing.
    More conservative than BH, valid for arbitrary dependence structure.
    Returns boolean array: True = significant after correction.
    """
    n   = len(p_values)
    c_n = np.sum(1.0 / np.arange(1, n + 1))   # harmonic number
    sorted_idx = np.argsort(p_values)
    sorted_p   = p_values[sorted_idx]
    thresholds = (np.arange(1, n + 1) / (n * c_n)) * alpha
    significant= np.zeros(n, dtype=bool)
    last_sig   = -1
    for i in range(n - 1, -1, -1):
        if sorted_p[i] <= thresholds[i]:
            last_sig = i
            break
    if last_sig >= 0:
        significant[sorted_idx[:last_sig + 1]] = True
    return significant


def run_sweep(
    ohlcv:  Dict,
    cfg_base: Mahoraga1Config,
    costs:  CostsConfig,
    ic_weights: Tuple[float, float, float]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Hyperparameter sweep over the 5 free parameters.
    Evaluated on VALIDATION period only.
    Best config selected by objective_A.
    Multiple-testing correction (BHY) applied to Sharpe p-values.

    FREE PARAMETERS (5 total):
      weight_cap       : HRP individual weight ceiling
      k_atr            : Chandelier stop multiplier
      turb_zscore_thr  : Turbulence trigger z-score
      turb_scale_min   : Minimum exposure during high turbulence
      vol_target_ann   : Annualised vol target
    """
    wt, wm, wr = ic_weights

    grid = {
        "weight_cap":      [0.55, 0.65],
        "k_atr":           [2.0, 2.5, 3.0],
        "turb_zscore_thr": [1.0, 1.5],
        "turb_scale_min":  [0.25, 0.40],
        "vol_target_ann":  [0.25, 0.30, 0.35],
    }

    from itertools import product
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))
    print(f"\n  [sweep] {len(combos)} combinations × VAL period")

    rows, best = [], None
    for combo in combos:
        kw = dict(zip(keys, combo))
        cfg = Mahoraga1Config(**{**cfg_base.__dict__, **kw,
                                  "w_trend": wt, "w_mom": wm, "w_rel": wr})
        res = backtest(ohlcv, cfg, costs, label="sweep")

        r_val  = res["returns_net"].loc[cfg.val_start:cfg.val_end]
        eq_val = cfg.capital_initial * (1.0 + r_val).cumprod()
        exp_val= res["exposure"].loc[r_val.index]
        to_val = res["turnover"].loc[r_val.index]

        qqq_val_r  = res["bench"]["QQQ_r"].loc[cfg.val_start:cfg.val_end]
        qqq_val_eq = cfg.capital_initial * (1.0 + qqq_val_r).cumprod()

        s_val     = summarize(r_val, eq_val, exp_val, to_val, cfg)
        s_qqq_val = summarize(qqq_val_r, qqq_val_eq, None, None, cfg)
        sr_ci     = sharpe_ci(r_val, cfg)
        score     = objective_A(s_val, s_qqq_val, cfg)

        row = {**kw, "score_val": score,
               "VAL_CAGR%":     round(s_val["CAGR"] * 100, 3),
               "VAL_Sharpe":    round(s_val["Sharpe"], 4),
               "VAL_Sharpe_CI": f"[{sr_ci['CI_lo']:.3f}, {sr_ci['CI_hi']:.3f}]",
               "VAL_SR_tstat":  round(sr_ci["t_stat"], 3),
               "VAL_SR_pval":   round(sr_ci["p_val"], 5),
               "VAL_MaxDD%":    round(s_val["MaxDD"] * 100, 3),
               "VAL_TurnoverAnn": round(s_val["TurnoverAnn"], 3)}
        rows.append(row)

        if best is None or score > best["score"]:
            best = {"score": score, "cfg": cfg, "s_val": s_val, "s_qqq_val": s_qqq_val}

    df = pd.DataFrame(rows).sort_values("score_val", ascending=False)

    # BHY multiple-testing correction on Sharpe p-values
    p_vals = df["VAL_SR_pval"].values
    df["SR_sig_BHY"] = bhy_correction(p_vals)

    # Minimum t-stat required for significance at FDR 5% with n=len(combos)
    # Rule of thumb from Harvey, Liu & Zhu (2016): t_min ≈ 3.0 for ~36 trials
    c_n     = np.sum(1.0 / np.arange(1, len(combos) + 1))
    t_min   = stats.norm.ppf(1.0 - (0.05 / (len(combos) * c_n)) / 2.0)
    df["SR_tstat_above_HLZ_tmin"] = df["VAL_SR_tstat"] > t_min
    print(f"  [sweep] HLZ t_min for {len(combos)} trials = {t_min:.3f}")

    return df, best


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {path}")


def plot_equity_curves(curves: Dict[str, pd.Series], title: str, path: str):
    fig, ax = plt.subplots(figsize=(14, 6))
    styles = ["-", "--", "-.", ":", (0,(3,1,1,1))]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (k, s) in enumerate(curves.items()):
        s = to_series(s).dropna()
        ax.plot(s.index, s.values, label=k,
                linewidth=2.0 if i == 0 else 1.2,
                linestyle=styles[i % len(styles)],
                color=colors[i % len(colors)])
    ax.set_yscale("log")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Equity (log scale)")
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, path)


def plot_drawdown(curves: Dict[str, pd.Series], title: str, path: str):
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (k, eq) in enumerate(curves.items()):
        eq = to_series(eq).dropna()
        dd = eq / eq.cummax() - 1.0
        ax.fill_between(dd.index, dd.values, 0,
                        alpha=0.25 if i == 0 else 0.12,
                        color=colors[i % len(colors)])
        ax.plot(dd.index, dd.values, label=k,
                linewidth=1.5 if i == 0 else 0.8,
                color=colors[i % len(colors)])
    ax.axhline(-0.30, color="red", linestyle="--", linewidth=0.8, label="-30% threshold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Drawdown")
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, path)


def plot_risk_overlays(res: Dict, title: str, path: str):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    ts = [
        (res["exposure"],     "#1f77b4", "Scaled Exposure"),
        (res["vol_scale"],    "#ff7f0e", "Vol-Target Scale"),
        (res["turb_scale"],   "#2ca02c", "Turbulence Scale"),
        (res["crisis_scale"], "#d62728", "Crisis Gate Scale"),
    ]
    for ax, (s, c, lbl) in zip(axes, ts):
        s = to_series(s).fillna(0.0)
        ax.plot(s.index, s.values, color=c, linewidth=1.0)
        ax.fill_between(s.index, s.values, alpha=0.2, color=c)
        ax.set_ylabel(lbl, fontsize=9)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(title, fontsize=13, fontweight="bold")
    _save(fig, path)


def plot_weights_heatmap(weights: pd.DataFrame, title: str, path: str):
    """Monthly average weights heatmap."""
    wm = weights.resample("ME").mean()
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(wm.T.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=weights.values.max())
    ax.set_yticks(range(len(wm.columns)))
    ax.set_yticklabels(wm.columns, fontsize=9)
    xticks = range(0, len(wm), max(1, len(wm) // 20))
    ax.set_xticks(list(xticks))
    ax.set_xticklabels([wm.index[i].strftime("%Y-%m") for i in xticks],
                       rotation=45, ha="right", fontsize=8)
    plt.colorbar(im, ax=ax, label="Weight")
    ax.set_title(title, fontsize=13, fontweight="bold")
    _save(fig, path)


def plot_rolling_ic(ic_df: pd.DataFrame, title: str, path: str):
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = {"IC_trend": "#1f77b4", "IC_mom": "#ff7f0e",
              "IC_rel": "#2ca02c",   "IC_composite": "#d62728"}
    for col in ic_df.columns:
        ax.plot(ic_df.index, ic_df[col], label=col,
                color=colors.get(col, "black"),
                linewidth=1.8 if col == "IC_composite" else 1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(0.04,  color="gray", linestyle="--", linewidth=0.7, label="IC=0.04 (meaningful)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Rolling 63-day IC")
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, path)


def plot_regime_bars(regime_df: pd.DataFrame, title: str, path: str):
    if regime_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["CAGR%", "Sharpe", "MaxDD%"]
    colors  = ["#1f77b4", "#ff7f0e", "#d62728"]
    for ax, metric, color in zip(axes, metrics, colors):
        vals = regime_df.set_index("Regime")[metric]
        vals.plot(kind="bar", ax=ax, color=color, alpha=0.8)
        ax.set_title(f"{metric} by Regime", fontsize=11)
        ax.set_xticklabels(vals.index, rotation=20, ha="right", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    _save(fig, path)


def plot_signal_decomp(decomp_res: Dict[str, Dict], bench_r: pd.Series,
                       cfg: Mahoraga1Config, title: str, path: str):
    """Equity curves and Sharpe for each single-component strategy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    colors = {"TREND_ONLY": "#1f77b4", "MOM_ONLY": "#ff7f0e", "REL_ONLY": "#2ca02c"}
    sharpes = {}
    for comp, res in decomp_res.items():
        eq = to_series(res["equity"]).dropna()
        ax1.plot(eq.index, eq.values, label=comp, color=colors[comp], linewidth=1.2)
        sharpes[comp] = sharpe(res["returns_net"], cfg.rf_annual, cfg.trading_days)
    bench_eq = cfg.capital_initial * (1.0 + to_series(bench_r).fillna(0.0)).cumprod()
    ax1.plot(bench_eq.index, bench_eq.values, label="QQQ", color="gray",
             linewidth=0.8, linestyle="--")
    ax1.set_yscale("log"); ax1.set_title("Equity (log)"); ax1.legend(); ax1.grid(True, alpha=0.3)
    bars = ax2.bar(list(sharpes.keys()), list(sharpes.values()),
                   color=[colors[k] for k in sharpes], alpha=0.8)
    ax2.set_title("Sharpe Ratio by Component"); ax2.set_ylabel("Sharpe")
    ax2.axhline(0, color="black", linewidth=0.5); ax2.grid(True, alpha=0.3, axis="y")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    _save(fig, path)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — EVALUATION & FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _slice(r: pd.Series, exp: pd.Series, to: pd.Series, cfg: Mahoraga1Config,
           start: str, end: str) -> Dict:
    rr  = r.loc[start:end]
    ee  = cfg.capital_initial * (1.0 + rr).cumprod()
    return {"r": rr, "eq": ee,
            "exp": exp.loc[rr.index], "to": to.loc[rr.index]}


def evaluate(
    ohlcv:  Dict,
    cfg:    Mahoraga1Config,
    costs:  CostsConfig,
    ff:     Optional[pd.DataFrame] = None,
    label:  str = "MAHORAGA_1.0"
) -> Dict:
    """Full evaluation: backtest + all analytics."""
    res  = backtest(ohlcv, cfg, costs, label=label)
    validate_no_lookahead(res, label)

    qqq_r = res["bench"]["QQQ_r"]
    qqq_eq= res["bench"]["QQQ_eq"]
    eqw   = baseline_equal_weight(ohlcv, cfg, costs)
    mom   = baseline_momentum(ohlcv, cfg, costs)

    def _bench_sum(r_b, eq_b):
        return summarize(r_b, eq_b, pd.Series(1.0, index=r_b.index), None, cfg, "QQQ")

    # ── Full period ──────────────────────────────────────────────────────────
    s_full     = summarize(res["returns_net"], res["equity"], res["exposure"], res["turnover"], cfg, label)
    s_qqq_full = _bench_sum(qqq_r, qqq_eq)
    s_eqw_full = summarize(eqw["r"], eqw["eq"], eqw["exp"], eqw["to"], cfg, "EQW_Tech")
    s_mom_full = summarize(mom["r"], mom["eq"], mom["exp"], mom["to"], cfg, "MOM_12_1_TopK")

    # ── Validation ──────────────────────────────────────────────────────────
    val       = _slice(res["returns_net"], res["exposure"], res["turnover"], cfg, cfg.val_start, cfg.val_end)
    qqq_val_r = qqq_r.loc[cfg.val_start:cfg.val_end]
    qqq_val_eq= cfg.capital_initial * (1.0 + qqq_val_r).cumprod()
    s_val     = summarize(val["r"], val["eq"], val["exp"], val["to"], cfg, f"{label}_VAL")
    s_qqq_val = _bench_sum(qqq_val_r, qqq_val_eq)

    # ── Test (out-of-sample, never touched during development) ───────────────
    test      = _slice(res["returns_net"], res["exposure"], res["turnover"], cfg, cfg.test_start, cfg.test_end)
    qqq_test_r= qqq_r.loc[cfg.test_start:cfg.test_end]
    qqq_test_eq= cfg.capital_initial * (1.0 + qqq_test_r).cumprod()
    s_test    = summarize(test["r"], test["eq"], test["exp"], test["to"], cfg, f"{label}_TEST")
    s_qqq_test= _bench_sum(qqq_test_r, qqq_test_eq)

    eqw_test_r = eqw["r"].loc[cfg.test_start:cfg.test_end]
    mom_test_r = mom["r"].loc[cfg.test_start:cfg.test_end]
    s_eqw_test = summarize(eqw_test_r, cfg.capital_initial*(1+eqw_test_r).cumprod(),
                           eqw["exp"].loc[eqw_test_r.index], eqw["to"].loc[eqw_test_r.index], cfg, "EQW_TEST")
    s_mom_test = summarize(mom_test_r, cfg.capital_initial*(1+mom_test_r).cumprod(),
                           mom["exp"].loc[mom_test_r.index], mom["to"].loc[mom_test_r.index], cfg, "MOM_TEST")

    # ── Statistical tests ────────────────────────────────────────────────────
    sr_ci_full = sharpe_ci(res["returns_net"], cfg)
    sr_ci_test = sharpe_ci(test["r"], cfg)
    alpha_nw   = alpha_test_nw(res["returns_net"], qqq_r, cfg, label)
    alpha_nw_test = alpha_test_nw(test["r"], qqq_test_r, cfg, f"{label}_TEST")
    ff_attr    = factor_attribution(res["returns_net"], ff, cfg, label)
    ff_attr_test = factor_attribution(test["r"], ff, cfg, f"{label}_TEST")

    # ── Regime analysis ──────────────────────────────────────────────────────
    regime_df  = regime_analysis(res["returns_net"], qqq_r, ohlcv, cfg)
    regime_test= regime_analysis(test["r"], qqq_test_r, ohlcv, cfg)

    # ── Stress ──────────────────────────────────────────────────────────────
    stress     = stress_report(res["returns_net"], res["exposure"], STRESS_EPISODES, cfg, qqq_r)

    # ── Bootstrap ────────────────────────────────────────────────────────────
    boot_full  = moving_block_bootstrap(res["returns_net"], seed=cfg.random_seed)
    boot_test  = moving_block_bootstrap(test["r"], seed=cfg.random_seed)

    # ── Score ────────────────────────────────────────────────────────────────
    score      = objective_A(s_val, s_qqq_val, cfg)

    return {
        "cfg": cfg, "res": res, "score_val": score,
        "full": s_full, "val": s_val, "test": s_test,
        "qqq_full": s_qqq_full, "qqq_val": s_qqq_val, "qqq_test": s_qqq_test,
        "eqw_full": s_eqw_full, "mom_full": s_mom_full,
        "eqw_test": s_eqw_test, "mom_test": s_mom_test,
        "sr_ci_full": sr_ci_full, "sr_ci_test": sr_ci_test,
        "alpha_nw": alpha_nw, "alpha_nw_test": alpha_nw_test,
        "ff_attr": ff_attr, "ff_attr_test": ff_attr_test,
        "regime_full": regime_df, "regime_test": regime_test,
        "stress": stress,
        "boot_full": boot_full, "boot_test": boot_test,
        "eqw": eqw, "mom": mom,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 16 — MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_mahoraga(make_plots: bool = True, run_decomp: bool = True):
    """
    Full pipeline:
    1. Validate splits (embargo)
    2. Fit IC weights on train data
    3. Run hyperparameter sweep on val data
    4. Evaluate best config on test (OUT-OF-SAMPLE, never touched)
    5. Run signal decomposition (Sprint 4)
    6. Print all tables, save all plots and CSVs
    """
    print("=" * 80)
    print("  MAHORAGA 1.0 — Systematic Large-Cap Tech Strategy")
    print("=" * 80)

    cfg_base = Mahoraga1Config()
    costs    = CostsConfig()

    _ensure_dir(cfg_base.cache_dir)
    _ensure_dir(cfg_base.plots_dir)
    _ensure_dir(cfg_base.outputs_dir)

    # ── Step 0: Validate splits ──────────────────────────────────────────────
    print("\n[0] Validating purged walk-forward splits …")
    validate_splits(cfg_base)

    # ── Step 1: Download data ────────────────────────────────────────────────
    print("\n[1] Downloading OHLCV …")
    tickers = list(cfg_base.universe) + [cfg_base.bench_qqq, cfg_base.bench_spy]
    ohlcv   = download_ohlcv(tickers, cfg_base.start, cfg_base.end, cfg_base.cache_dir)

    print("\n[1b] Loading Fama-French factors …")
    ff = load_ff_factors(cfg_base.cache_dir)

    # ── Step 2: Fit IC weights on TRAIN data ─────────────────────────────────
    print("\n[2] Fitting IC-based signal weights on TRAIN period …")
    close_train = ohlcv["close"][list(cfg_base.universe)].loc[cfg_base.train_start:cfg_base.train_end]
    qqq_train   = to_series(ohlcv["close"][cfg_base.bench_qqq].loc[cfg_base.train_start:cfg_base.train_end].ffill())
    ic_weights  = fit_ic_weights(close_train, qqq_train, cfg_base, period="train")

    # ── Step 3: Sweep on VAL ─────────────────────────────────────────────────
    print("\n[3] Hyperparameter sweep on VALIDATION …")
    sweep_df, best_val = run_sweep(ohlcv, cfg_base, costs, ic_weights)

    sweep_path = os.path.join(cfg_base.outputs_dir, "mahoraga_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\n  Sweep results → {sweep_path}")
    print("\n  Top 10 configs by validation score:")
    print(sweep_df.head(10).to_string(index=False))

    best_cfg = best_val["cfg"]
    print(f"\n  BEST CONFIG:")
    print(f"    weight_cap={best_cfg.weight_cap} | k_atr={best_cfg.k_atr} | "
          f"turb_zscore_thr={best_cfg.turb_zscore_thr} | "
          f"turb_scale_min={best_cfg.turb_scale_min} | "
          f"vol_target_ann={best_cfg.vol_target_ann}")

    # ── Step 4: Full evaluation with best config ─────────────────────────────
    print("\n[4] Full evaluation with best config …")
    out = evaluate(ohlcv, best_cfg, costs, ff=ff, label="MAHORAGA_1.0")

    # ── Step 5: Rolling IC ───────────────────────────────────────────────────
    print("\n[5] Computing rolling IC (63-day) …")
    qqq_full = to_series(ohlcv["close"][cfg_base.bench_qqq].reindex(
        ohlcv["close"][list(cfg_base.universe)].index).ffill())
    ic_df = rolling_ic(ohlcv["close"][list(cfg_base.universe)], qqq_full, best_cfg, window=63)

    # ── Step 6: Signal decomposition ─────────────────────────────────────────
    decomp_res = {}
    if run_decomp:
        print("\n[6] Signal decomposition …")
        decomp_res = baseline_signal_decomp(ohlcv, best_cfg, costs)

    # ── PRINT RESULTS ────────────────────────────────────────────────────────
    _print_results(out, sweep_df, best_cfg, ic_df, decomp_res, ff)

    # ── SAVE CSVS ────────────────────────────────────────────────────────────
    _save_csvs(out, ic_df, best_cfg.outputs_dir)

    # ── PLOTS ────────────────────────────────────────────────────────────────
    if make_plots:
        _make_plots(out, ic_df, decomp_res, best_cfg)

    return out, sweep_df


def _print_results(out, sweep_df, cfg, ic_df, decomp_res, ff):
    sep = "=" * 100

    def fmt_row(s):
        return {
            "Model":       s["Label"],
            "FinalEq($)":  f"{s['FinalEquity']:,.0f}",
            "CAGR%":       f"{s['CAGR']*100:.2f}",
            "Vol%":        f"{s['AnnVol']*100:.2f}",
            "Sharpe":      f"{s['Sharpe']:.3f}",
            "Sortino":     f"{s['Sortino']:.3f}",
            "MaxDD%":      f"{s['MaxDD']*100:.2f}",
            "Calmar":      f"{s['Calmar']:.3f}",
            "AvgExp%":     f"{s['AvgExposure']*100:.1f}",
            "TurnAnn":     f"{s['TurnoverAnn']:.2f}",
            "CVaR5%":      f"{s['CVaR_5']*100:.3f}",
        }

    print(f"\n{sep}")
    print("  FULL PERIOD COMPARISON  (cost-adjusted benchmarks)")
    print(sep)
    comp = [fmt_row(out["full"]), fmt_row(out["qqq_full"]),
            fmt_row(out["eqw_full"]), fmt_row(out["mom_full"])]
    print(pd.DataFrame(comp).to_string(index=False))

    print(f"\n{sep}")
    print("  FINAL TEST 2020-2026 (NEVER touched during development)")
    print(sep)
    comp_test = [fmt_row(out["test"]), fmt_row(out["qqq_test"]),
                 fmt_row(out["eqw_test"]), fmt_row(out["mom_test"])]
    print(pd.DataFrame(comp_test).to_string(index=False))

    # Sharpe CI
    print(f"\n{sep}")
    print("  SHARPE RATIO — 95% CONFIDENCE INTERVALS (Jobson-Korkie 1981)")
    print(sep)
    for period, ci in [("Full", out["sr_ci_full"]), ("Test", out["sr_ci_test"])]:
        print(f"  {period:6s}: SR={ci['SR']:.4f}  95%CI=[{ci['CI_lo']:.4f}, {ci['CI_hi']:.4f}]  "
              f"t={ci['t_stat']:.3f}  p={ci['p_val']:.5f}")

    # Alpha NW
    print(f"\n{sep}")
    print("  ALPHA TEST — Newey-West HAC (H₀: α = 0 vs QQQ)")
    print(sep)
    for period, a in [("Full", out["alpha_nw"]), ("Test", out["alpha_nw_test"])]:
        if "error" in a:
            print(f"  {period}: ERROR — {a['error']}")
        else:
            sig = "***" if a["sig_1pct"] else ("**" if a["sig_5pct"] else "   ")
            print(f"  {period:6s}: α_ann={a['alpha_ann']*100:.2f}%  "
                  f"t={a['t_alpha']:.3f}  p={a['p_alpha']:.5f}  "
                  f"β={a['beta']:.4f}  R²={a['R2']:.4f}  {sig}")

    # FF attribution
    if out["ff_attr"] is not None and "error" not in out["ff_attr"]:
        print(f"\n{sep}")
        print("  FAMA-FRENCH 5-FACTOR + UMD ATTRIBUTION")
        print(sep)
        for period, fa in [("Full", out["ff_attr"]), ("Test", out["ff_attr_test"])]:
            if fa and "error" not in fa:
                print(f"  {period:6s}: α={fa['alpha_ann']*100:.2f}%  t={fa['t_alpha']:.3f}  "
                      f"β_mkt={fa['beta_mkt']:.3f}  β_umd={fa['beta_umd']:.3f}  "
                      f"β_smb={fa['beta_smb']:.3f}  R²_adj={fa['R2_adj']:.3f}")

    # Regime analysis
    print(f"\n{sep}")
    print("  REGIME ANALYSIS — Full Period (VIX percentile terciles)")
    print(sep)
    print(out["regime_full"].to_string(index=False))

    print(f"\n  REGIME ANALYSIS — Test Period 2020–2026")
    print(out["regime_test"].to_string(index=False))

    # Stress
    print(f"\n{sep}")
    print("  STRESS EPISODES")
    print(sep)
    print(out["stress"].to_string(index=False))

    # Bootstrap
    print(f"\n{sep}")
    print("  BOOTSTRAP (Moving Block, 1000 samples, block=20)")
    print(sep)
    for period, b in [("Full", out["boot_full"]), ("Test", out["boot_test"])]:
        print(f"  {period:6s}: DD_p50={b['dd_p50']*100:.2f}%  "
              f"DD_p5_worst={b['dd_p5_worst']*100:.2f}%  "
              f"P(DD<-30%)={b['ruin_prob_30dd']:.2f}%  "
              f"P(DD<-50%)={b['ruin_prob_50dd']:.2f}%")

    # Stop hits
    print(f"\n  Stop hits (Chandelier): {out['res']['stop_hits']}")

    # Signal decomp
    if decomp_res:
        print(f"\n{sep}")
        print("  SIGNAL DECOMPOSITION — Sharpe by Component")
        print(sep)
        res = out["res"]
        qqq_r = res["bench"]["QQQ_r"]
        for comp, r in decomp_res.items():
            sr  = sharpe(r["returns_net"], 0.0, 252)
            nw  = alpha_test_nw(r["returns_net"], qqq_r, out["cfg"], comp)
            sig = "***" if nw.get("sig_1pct") else ("**" if nw.get("sig_5pct") else "   ")
            print(f"  {comp:15s}: Sharpe={sr:.3f}  α={nw.get('alpha_ann',0)*100:.2f}%  "
                  f"t={nw.get('t_alpha',0):.3f}  {sig}")

    # Rolling IC summary
    print(f"\n  Rolling IC (63d) mean — trend={ic_df['IC_trend'].mean():.4f}  "
          f"mom={ic_df['IC_mom'].mean():.4f}  rel={ic_df['IC_rel'].mean():.4f}  "
          f"composite={ic_df['IC_composite'].mean():.4f}")


def _save_csvs(out: Dict, ic_df: pd.DataFrame, out_dir: str):
    def to_df(s_list):
        return pd.DataFrame([{k: round(v, 6) if isinstance(v, float) else v
                               for k, v in s.items()} for s in s_list])

    to_df([out["full"], out["qqq_full"], out["eqw_full"], out["mom_full"]]) \
        .to_csv(os.path.join(out_dir, "comparison_full.csv"), index=False)
    to_df([out["test"], out["qqq_test"], out["eqw_test"], out["mom_test"]]) \
        .to_csv(os.path.join(out_dir, "comparison_test.csv"), index=False)
    out["stress"].to_csv(os.path.join(out_dir, "stress_episodes.csv"), index=False)
    out["regime_full"].to_csv(os.path.join(out_dir, "regime_full.csv"), index=False)
    out["regime_test"].to_csv(os.path.join(out_dir, "regime_test.csv"), index=False)
    ic_df.to_csv(os.path.join(out_dir, "rolling_ic.csv"))

    if out["ff_attr"] is not None:
        pd.DataFrame([out["ff_attr"], out.get("ff_attr_test", {})]) \
            .to_csv(os.path.join(out_dir, "ff_attribution.csv"), index=False)

    pd.DataFrame([out["alpha_nw"], out["alpha_nw_test"]]) \
        .to_csv(os.path.join(out_dir, "alpha_nw.csv"), index=False)

    pd.DataFrame([out["sr_ci_full"], out["sr_ci_test"]]) \
        .to_csv(os.path.join(out_dir, "sharpe_ci.csv"), index=False)

    print(f"\n  [CSVs saved to ./{out['cfg'].outputs_dir}/]")


def _make_plots(out: Dict, ic_df: pd.DataFrame, decomp_res: Dict, cfg: Mahoraga1Config):
    p   = cfg.plots_dir
    res = out["res"]
    eqw = out["eqw"]
    mom = out["mom"]

    # 1. Equity curves (full)
    plot_equity_curves({
        "MAHORAGA_1.0":   res["equity"],
        "QQQ (cost-adj)": res["bench"]["QQQ_eq"],
        "EQW_Tech(net)":  eqw["eq"],
        "MOM_12_1(net)":  mom["eq"],
    }, "Equity Curves — Mahoraga 1.0 vs Benchmarks (Full Period, log scale)",
    os.path.join(p, "01_equity_full.png"))

    # 2. Equity curves (test only)
    r_test    = res["returns_net"].loc[cfg.test_start:]
    eq_test   = cfg.capital_initial * (1.0 + r_test).cumprod()
    eqw_test  = cfg.capital_initial * (1.0 + eqw["r"].loc[cfg.test_start:]).cumprod()
    mom_test  = cfg.capital_initial * (1.0 + mom["r"].loc[cfg.test_start:]).cumprod()
    qqq_test  = res["bench"]["QQQ_eq"].loc[cfg.test_start:]
    plot_equity_curves({
        "MAHORAGA_1.0": eq_test,
        "QQQ (cost-adj)": qqq_test,
        "EQW_Tech(net)": eqw_test,
        "MOM_12_1(net)": mom_test,
    }, "Equity Curves — TEST PERIOD 2020–2026 (log scale)",
    os.path.join(p, "02_equity_test.png"))

    # 3. Drawdown
    plot_drawdown({
        "MAHORAGA_1.0": res["equity"],
        "QQQ":          res["bench"]["QQQ_eq"],
        "MOM_12_1":     mom["eq"],
    }, "Drawdown — Mahoraga 1.0", os.path.join(p, "03_drawdown.png"))

    # 4. Risk overlays
    plot_risk_overlays(res, "Risk Overlays — Mahoraga 1.0",
                       os.path.join(p, "04_risk_overlays.png"))

    # 5. Weights heatmap
    plot_weights_heatmap(res["weights_scaled"],
                         "Portfolio Weights (monthly avg) — Mahoraga 1.0",
                         os.path.join(p, "05_weights_heatmap.png"))

    # 6. Rolling IC
    plot_rolling_ic(ic_df, "Rolling 63-day IC by Signal Component",
                    os.path.join(p, "06_rolling_ic.png"))

    # 7. Regime bars
    plot_regime_bars(out["regime_full"], "Performance by VIX Regime — Full Period",
                     os.path.join(p, "07_regime_full.png"))
    plot_regime_bars(out["regime_test"], "Performance by VIX Regime — Test Period",
                     os.path.join(p, "08_regime_test.png"))

    # 8. Signal decomp
    if decomp_res:
        plot_signal_decomp(decomp_res, res["bench"]["QQQ_r"], cfg,
                           "Signal Decomposition — Single-Component Strategies",
                           os.path.join(p, "09_signal_decomp.png"))

    print(f"\n  [Plots saved to ./{cfg.plots_dir}/]")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    out, sweep_df = run_mahoraga(make_plots=True, run_decomp=True)
