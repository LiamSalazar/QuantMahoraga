"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        MAHORAGA  1.3                                         ║
║   Long-Only Weekly-Rebalanced Tech Rotation — Research Edition               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Changes from 1.2 → 1.3                                                      ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  [WALK-FORWARD — OOS purity guarantees (stable)]                              ║
║  · 5-tuple explicit fold boundaries: (train_end, val_start, val_end,         ║
║    test_start, test_end) — no heuristic inference of test_end                ║
║  · Purity asserted: val∩test=∅, test∩test=∅, contiguous, embargo ✓         ║
║  · Gap detection uses REAL trading calendar, not bdate_range                ║
║  · Final config = winning combo of last fold (no parameter mixing)           ║
║  · p/q/stat_label tied exactly to selected combo (full traceability)        ║
║  · Hyperparams stored as explicit combo_params dict; no string mapping       ║
║                                                                              ║
║  [STATISTICS — fix #6]                                                        ║
║  · Sharpe: excess = r - rf_daily (correct daily subtraction)                ║
║  · Sortino: downside deviation on excess returns clipped at 0               ║
║                                                                              ║
║  [HRP ROBUSTNESS — fix #7]                                                    ║
║  · Drop degenerate columns (< min_unique_returns) before clustering          ║
║  · Validate condensed distance matrix finiteness; fallback = inv-vol         ║
║                                                                              ║
║  [PARALLELISATION — fix #8]                                                   ║
║  · Parallelisation documented as operational optimisation only               ║
║  · No RAM/speed guarantees; backend configurable                             ║
║                                                                              ║
║  [OUTPUTS — fix #9]                                                           ║
║  · walk_forward_meta.csv                                                     ║
║  · selected_config_support.csv                                               ║
║  · final_report.txt                                                          ║
║  · universe_snapshots/ (per reconstitution date)                             ║
║  · universe_methodology.json                                                 ║
║                                                                              ║
║  [UNIVERSE ENGINE — governs simulation (stable)]                              ║
║  · Quarterly PIT reconstitution; schedule passed to backtest point-in-time  ║
║  · tickers_in_scope = equity candidates only (benchmarks/ETFs excluded)     ║
║  · Eligibility: seasoning ≥ 63d, ADDV ≥ $50M, volume-continuity proxy      ║
║  · Ranking: LiqSizeProxy = price × volume (NOT FFMC — honestly named)      ║
║    To use real FFMC/GICS-IT classification, provide CRSP/Compustat PIT data ║
║  · Buffer rule: top-8 auto, incumbents to rank-10, buffer to rank-13        ║
║  · Snapshots with entry/exit/reason saved; universe_methodology.json        ║
║  · Without CRSP/Compustat PIT, survivorship bias cannot be eliminated       ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║  MANDATORY METHODOLOGY DISCLAIMER                                            ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  The primary universe of this strategy consists of mega-cap technology       ║
║  stocks selected ex post based on their prominence as of the analysis date.  ║
║  The canonical universe engine (Section 3B) implements quarterly             ║
║  point-in-time reconstitution using LiqSizeProxy (price × volume) — a       ║
║  liquidity-weighted size proxy, NOT float-adjusted market cap (FFMC).        ║
║  Real FFMC/GICS-IT requires CRSP/Compustat PIT data, which is not used here.║
║  Without such data, survivorship bias cannot be fully eliminated. All        ║
║  results are conditional on this universe design and do not constitute       ║
║  evidence free of survivorship bias, nor proof of generalisable alpha.       ║
║  This is a conditional research exercise, not a trading recommendation.      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import hashlib
import textwrap
from copy import deepcopy
from dataclasses import dataclass, field, asdict
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

try:
    from joblib import Parallel, delayed as joblib_delayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False
    print("  [warn] joblib not found — parallel_sweep disabled automatically")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

UNIVERSE_BIAS_DISCLAIMER = """
═══════════════════════════════════════════════════════════════════════════════
  METHODOLOGY DISCLAIMER — UNIVERSE BIAS
───────────────────────────────────────────────────────────────────────────────
  The primary universe consists of mega-cap technology stocks selected ex post.
  The canonical universe engine uses LiqSizeProxy (price × volume), which is a
  liquidity-weighted size proxy — NOT float-adjusted market cap (FFMC) and NOT
  a GICS-IT classification. Real FFMC/GICS-IT requires CRSP/Compustat PIT data,
  which is not integrated here. Without such data, survivorship bias cannot be
  fully eliminated. Results are conditional on universe design. Not evidence of
  generalisable alpha. This is a conditional research exercise, not a
  trading recommendation.
═══════════════════════════════════════════════════════════════════════════════
"""

EXECUTION_STRESS_DISCLAIMER = """
  NOTE — Execution_Sensitivity_Stress evaluates performance degradation under
  severe execution friction (cost/slippage sensitivity, panic-execution
  penalties). It does NOT simulate realistic overnight price gaps or actual
  market microstructure events. It is a simplified operational robustness test.
"""


@dataclass
class CostsConfig:
    """
    gap_factor: execution friction multiplier in Execution_Sensitivity_Stress.
    This does NOT simulate real overnight price gaps — it models performance
    degradation under severe friction (operational robustness test only).
    """
    commission:        float = 0.0010
    slippage:          float = 0.0003
    apply_slippage:    bool  = True
    gap_factor:        float = 1.0
    qqq_expense_ratio: float = 0.0020 / 252


@dataclass
class UniverseConfig:
    """
    Canonical universe engine configuration (Section 3B).
    Implements quarterly point-in-time reconstitution using LiqSizeProxy
    (price × volume) ranking and incumbency buffers.

    NOTE: This is NOT GICS-IT classification and NOT float-adjusted market cap
    (FFMC). Real FFMC/GICS-IT requires CRSP/Compustat PIT data. The LiqSizeProxy
    is an investability proxy ranked by recent price × volume turnover.
    """
    target_size:        int   = 10    # exogenous design constraint, NOT optimised
    auto_entry_rank:    int   = 8     # top-8 enter automatically
    retention_rank:     int   = 10    # incumbents retained within top-10
    buffer_rank:        int   = 13    # additional incumbency buffer to rank-13
    min_seasoning_days: int   = 63    # ≈ 3 calendar months
    min_free_float:     float = 0.10  # 10% free float minimum
    min_addv_usd:       float = 50e6  # average daily dollar volume, 3-month
    recon_freq:         str   = "QS"  # quarterly reconstitution


@dataclass
class Mahoraga4Config:
    # ── Static fallback universe (used when canonical engine data unavailable) ─
    # Labelled explicitly as ex-post selected; disclaimer always printed
    universe_static: Tuple[str, ...] = (
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
        "META", "AVGO", "ASML", "TSM", "ADBE", "NFLX", "AMD"
    )
    use_canonical_universe: bool = True   # set True when CRSP/PIT data available

    bench_qqq: str = "QQQ"
    bench_spy: str = "SPY"
    bench_vix: str = "^VIX"

    capital_initial: float = 100_000.0
    data_start:      str   = "2005-01-01"
    data_end:        str   = "2026-02-20"
    trading_days:    int   = 252
    rf_annual:       float = 0.0

    # ── Walk-forward ──────────────────────────────────────────────────────────
    # wf_folds: 5-tuple (train_end, val_start, val_end, test_start, test_end)
    #
    # EXPANDING WFO SCHEDULE (fixes #1 + #5):
    #   - val and test are strictly disjoint WITHIN each fold
    #   - test windows are mutually exclusive across folds
    #   - test windows are temporally contiguous across folds
    #   - Boundaries are EXPLICIT — no heuristic inference of test_end
    #   - Later validation windows may reuse earlier realised history, including
    #     periods that were OOS in prior folds. This is standard in expanding
    #     WFO and is disclosed explicitly; it does not create day-level look-ahead
    #     inside any test block.
    #
    # Embargo between train_end and val_start >= 252 trading days.
    wf_train_start: str = "2006-01-01"
    wf_folds: Tuple[Tuple[str, str, str, str, str], ...] = (
        # (train_end,   val_start,    val_end,      test_start,   test_end    )
        ("2013-12-31", "2015-01-02", "2016-12-30", "2017-01-03", "2018-12-31"),
        ("2015-12-31", "2017-01-03", "2018-12-31", "2019-01-02", "2020-12-31"),
        ("2017-12-29", "2019-01-02", "2020-12-31", "2021-01-04", "2022-12-30"),
        # FIX 1.3.1: extend fold 4 test through 2024-06-28 so the stitched OOS
        # path remains contiguous into fold 5, whose explicit test starts on
        # 2024-07-01 after a longer validation block.
        ("2019-12-31", "2021-01-04", "2022-12-30", "2023-01-03", "2024-06-28"),
        ("2021-12-31", "2023-01-03", "2024-06-28", "2024-07-01", "2026-02-20"),
    )
    embargo_days: int = 252

    # ── Rebalance & selection ─────────────────────────────────────────────────
    rebalance_freq: str = "W-FRI"
    top_k:          int = 3

    # ── Signals ───────────────────────────────────────────────────────────────
    spans_short:  Tuple[int, ...] = (42, 84)
    spans_long:   Tuple[int, ...] = (126, 252)
    mom_windows:  Tuple[int, ...] = (63, 126, 252)
    rel_windows:  Tuple[int, ...] = (63, 126)
    burn_in:      int = 252

    w_trend: float = 0.333
    w_mom:   float = 0.333
    w_rel:   float = 0.334

    hrp_window:  int   = 252
    weight_cap:  float = 0.60

    atr_window:         int   = 14
    k_atr:              float = 2.5
    stop_on:            bool  = True
    allow_reentry:      bool  = True
    reentry_atr_buffer: float = 0.25
    stop_keep_cash:     bool  = True

    crisis_gate_use:        bool  = True
    crisis_dd_pct:          float = 0.05
    crisis_vol_pct:         float = 0.90
    crisis_min_days_on:     int   = 5
    crisis_min_days_off:    int   = 10
    crisis_scale:           float = 0.0
    crisis_dd_thr:          float = 0.20
    crisis_vol_zscore_thr:  float = 1.5

    turb_window:                  int   = 63
    illiq_window:                 int   = 21
    turb_zscore_thr:              float = 1.2
    turb_scale_min:               float = 0.30
    turb_eval_on_rebalance_only:  bool  = True

    vol_target_on:   bool  = True
    vol_target_ann:  float = 0.30
    port_vol_window: int   = 63
    max_exposure:    float = 1.0
    min_exposure:    float = 0.0

    target_maxdd:        float = 0.28
    dd_penalty_strength: float = 4.0
    calmar_weight:       float = 0.15
    turnover_soft_cap:   float = 12.0
    turnover_penalty:    float = 0.02

    cache_dir:      str  = "data_cache"
    random_seed:    int  = 42
    plots_dir:      str  = "mahoraga4_plots"
    outputs_dir:    str  = "mahoraga4_outputs"
    label:          str  = "MAHORAGA_4"
    parallel_sweep: bool = True   # operational optimisation only; no speed guarantee


ALTERNATE_UNIVERSES: Dict[str, Tuple[str, ...]] = {
    "FULL_MEGA":   ("AAPL","MSFT","NVDA","GOOGL","AMZN","META","AVGO","ASML","TSM","ADBE","NFLX","AMD"),
    "SEMIS":       ("NVDA","AVGO","ASML","TSM","AMD","INTC","QCOM","MU","AMAT","LRCX"),
    "PLATFORMS":   ("AAPL","MSFT","GOOGL","AMZN","META","NFLX","ADBE","CRM","NOW","SHOP"),
    "AI_CORE":     ("NVDA","MSFT","GOOGL","AMZN","META","AVGO","AMD","ARM","PLTR","SMCI"),
}

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
    """
    FIX (#6): excess return = r - rf_daily (correct daily subtraction).
    1.1/1.2 had a precedence bug: r - (1+rf)^(1/td) + 1 ≠ r - rf_daily.
    """
    r = to_s(r).dropna()
    rf_d = (1.0 + rf) ** (1.0 / td) - 1.0   # daily risk-free rate
    ex   = r - rf_d
    sd   = ex.std(ddof=1)
    return float(np.sqrt(td) * ex.mean() / sd) if (sd and np.isfinite(sd) and sd > 0) else 0.0

def sortino(r: pd.Series, rf: float = 0.0, td: int = 252) -> float:
    """
    FIX (#6): downside deviation computed on excess returns clipped at 0.
    """
    r    = to_s(r).dropna()
    rf_d = (1.0 + rf) ** (1.0 / td) - 1.0
    ex   = r - rf_d
    dd   = ex.clip(upper=0.0).std(ddof=1)
    return float(np.sqrt(td) * ex.mean() / dd) if (dd and np.isfinite(dd) and dd > 0) else 0.0

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
# SECTION 3A — DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def download_ohlcv(tickers: List[str], start: str, end: str,
                   cache_dir: str) -> Dict[str, pd.DataFrame]:
    _ensure_dir(cache_dir)
    key  = _hash({"tickers": sorted(tickers), "start": start, "end": end, "v": 8})
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
                g = lambda f, _t=t: raw[(_t, f)] if (_t, f) in raw.columns else None
            else:
                g = lambda f: raw[f] if f in raw.columns else None
            c = g("Close"); h = g("High"); l = g("Low"); v = g("Volume")
            if c is None: continue
            ri = lambda s: pd.Series(s, index=raw.index).reindex(idx).ffill(limit=5)
            close[t] = ri(c)
            high[t]  = ri(h) if h is not None else ri(c)
            low[t]   = ri(l) if l is not None else ri(c)
            vol[t]   = ri(v) if v is not None else np.nan
        except Exception as e:
            print(f"  [warn] {t}: {e}")

    out = {k: v.dropna(how="all") for k, v in
           [("close", close), ("high", high), ("low", low), ("volume", vol)]}
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
        ff = ff5.join(umd[["Mom"]], how="left").rename(columns={"Mom": "UMD"}) / 100.0
        pd.to_pickle(ff, path)
        return ff
    except Exception as e:
        print(f"  [warn] FF factors unavailable ({e}). Attribution skipped.")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3B — CANONICAL UNIVERSE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _build_universe_snapshot(
    date:             pd.Timestamp,
    close:            pd.DataFrame,
    volume:           pd.DataFrame,
    universe_cfg:     UniverseConfig,
    prior_members:    Optional[List[str]],
    first_date:       Optional[pd.Timestamp],
    tickers_in_scope: List[str],
) -> Tuple[List[str], pd.DataFrame]:
    """
    Point-in-time universe construction for one reconstitution date.

    RANKING VARIABLE — LiqSizeProxy (NOT FFMC):
      LiqSizeProxy = mean(price × volume) over last 30 trading days.
      This is a liquidity-weighted size proxy. It is NOT float-adjusted market
      cap (FFMC). FFMC = price × shares_outstanding × free_float requires
      CRSP/Compustat PIT data. Without that data, this proxy ranks by
      investability (price × turnover), not pure market cap. Named honestly.

    FREE FLOAT FILTER (fix #2):
      min_free_float is approximated as: if volume data is very sparse or zero
      for long stretches, the ticker fails. Without actual float data from a
      PIT database, we use volume continuity as a proxy for float adequacy.
      This is explicitly disclosed. Plug in real float data for institutional use.

    TICKERS IN SCOPE (fix #3):
      tickers_in_scope must contain ONLY equity candidates (no benchmarks,
      ETFs, or index tickers). Filtering is enforced by the caller.

    Returns (selected_members, snapshot_df_with_reasons).
    """
    idx  = close.index
    past = idx[idx <= date]
    if len(past) < universe_cfg.min_seasoning_days + 20:
        return list(tickers_in_scope)[:universe_cfg.target_size], pd.DataFrame()

    rows = []
    for t in tickers_in_scope:
        if t not in close.columns:
            continue
        p_series = close[t].reindex(past).dropna()
        v_series = (volume[t].reindex(past).dropna()
                    if t in volume.columns else pd.Series(dtype=float))

        # ── Seasoning filter ──────────────────────────────────────────────────
        if len(p_series) < universe_cfg.min_seasoning_days:
            rows.append({"ticker": t, "eligible": False, "reason": "seasoning_fail",
                         "liq_size_proxy": 0.0, "addv_proxy": 0.0})
            continue

        # ── ADDV filter (liquidity screen, hard cutoff) ───────────────────────
        if len(v_series) >= 20:
            last63_p = p_series.iloc[-63:] if len(p_series) >= 63 else p_series
            last63_v = v_series.reindex(last63_p.index).fillna(0)
            addv = float((last63_p * last63_v).mean())
        else:
            addv = 0.0

        if addv < universe_cfg.min_addv_usd:
            rows.append({"ticker": t, "eligible": False, "reason": "addv_fail",
                         "liq_size_proxy": 0.0, "addv_proxy": addv})
            continue

        # ── Free-float proxy filter (fix #2: actually applied) ────────────────
        # Proxy: fraction of trading days with non-zero volume in last 63 days.
        # A ticker with vol_coverage < min_free_float is likely thinly floated
        # or illiquid; exclude it. This is a proxy — not actual float data.
        if len(v_series) >= 20:
            last63_v2 = v_series.iloc[-63:] if len(v_series) >= 63 else v_series
            vol_coverage = float((last63_v2 > 0).mean())
        else:
            vol_coverage = 0.0

        if vol_coverage < universe_cfg.min_free_float:
            rows.append({"ticker": t, "eligible": False,
                         "reason": f"free_float_proxy_fail (coverage={vol_coverage:.2f})",
                         "liq_size_proxy": 0.0, "addv_proxy": addv})
            continue

        # ── LiqSizeProxy ranking variable (NOT FFMC — explicitly named) ───────
        last30_p = p_series.iloc[-30:] if len(p_series) >= 30 else p_series
        last30_v = (v_series.reindex(last30_p.index).fillna(0)
                    if len(v_series) >= 10 else pd.Series([1.0] * len(last30_p)))
        liq_size = float((last30_p * last30_v).mean())

        rows.append({
            "ticker":          t,
            "eligible":        True,
            "reason":          "eligible",
            "liq_size_proxy":  liq_size,
            "addv_proxy":      addv,
            "vol_coverage":    vol_coverage,
        })

    if not rows:
        return [], pd.DataFrame()

    snap = pd.DataFrame(rows).sort_values("liq_size_proxy", ascending=False)
    eligible = snap[snap["eligible"]].reset_index(drop=True)
    eligible["rank"] = eligible.index + 1

    prior    = set(prior_members) if prior_members else set()
    ucfg     = universe_cfg
    selected = []

    # Step 1: auto-entry — top-N by LiqSizeProxy rank
    top_auto = eligible[eligible["rank"] <= ucfg.auto_entry_rank]["ticker"].tolist()
    selected.extend(top_auto)

    # Step 2: retain incumbents within retention band
    for t in eligible[(eligible["rank"] > ucfg.auto_entry_rank) &
                       (eligible["rank"] <= ucfg.retention_rank)]["ticker"].tolist():
        if t in prior and t not in selected:
            selected.append(t)
            eligible.loc[eligible["ticker"] == t, "reason"] = "retained_incumbent_rank10"

    # Step 3: buffer incumbents to buffer band
    if len(selected) < ucfg.target_size:
        for t in eligible[(eligible["rank"] > ucfg.retention_rank) &
                           (eligible["rank"] <= ucfg.buffer_rank)]["ticker"].tolist():
            if t in prior and t not in selected and len(selected) < ucfg.target_size:
                selected.append(t)
                eligible.loc[eligible["ticker"] == t, "reason"] = "retained_incumbent_buffer"

    # Step 4: fill remaining by rank
    if len(selected) < ucfg.target_size:
        for t in eligible["ticker"].tolist():
            if t not in selected and len(selected) < ucfg.target_size:
                selected.append(t)

    # Annotate entry/exit reasons in snapshot
    snap["in_universe"]  = snap["ticker"].isin(selected)
    snap["recon_date"]   = date.date()
    snap["prior_member"] = snap["ticker"].isin(prior)
    snap["entered"]      = snap["in_universe"] & ~snap["prior_member"]
    snap["exited"]       = ~snap["in_universe"] & snap["prior_member"]

    return selected, snap


def build_canonical_universe_schedule(
    close:        pd.DataFrame,
    volume:       pd.DataFrame,
    universe_cfg: UniverseConfig,
    tickers_in_scope: List[str],
    start:        str,
    end:          str,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    Build quarterly universe schedule from start to end.
    Returns:
      schedule_df: date → list of selected tickers (as JSON string)
      snapshots:   list of snapshot DataFrames (one per reconstitution)
    """
    recon_dates = pd.date_range(start, end, freq=universe_cfg.recon_freq)
    recon_dates = pd.DatetimeIndex([d for d in recon_dates if d >= pd.Timestamp(start)])

    schedule_rows = []
    snapshots     = []
    prior_members = None
    first_date    = recon_dates[0] if len(recon_dates) else None

    for rd in recon_dates:
        members, snap = _build_universe_snapshot(
            rd, close, volume, universe_cfg, prior_members, first_date, tickers_in_scope
        )
        schedule_rows.append({"recon_date": rd, "members": json.dumps(members),
                               "n_members": len(members)})
        snapshots.append(snap)
        prior_members = members

    schedule_df = pd.DataFrame(schedule_rows)
    return schedule_df, snapshots


def get_universe_at_date(schedule_df: pd.DataFrame, date: pd.Timestamp) -> List[str]:
    """Return the most recent universe valid at the given date."""
    valid = schedule_df[schedule_df["recon_date"] <= date]
    if valid.empty:
        return []
    return json.loads(valid.iloc[-1]["members"])


def get_training_universe(
    train_end:         str,
    universe_schedule: Optional[pd.DataFrame],
    universe_static:   Tuple[str, ...],
    ohlcv_close_cols:  List[str],
) -> List[str]:
    """
    Return the set of tickers that should be used for SIGNAL TRAINING
    up to train_end.

    Logic:
      · PIT mode (universe_schedule is not None):
          Union of all members that appeared in ANY reconstitution date
          <= train_end.  This captures every ticker the strategy could have
          traded during training — no ex-post look-ahead, no static ex-post list.
      · Static fallback (no schedule):
          cfg.universe_static — labelled as ex-post, disclaimer persists.

    The resulting list is intersected with ohlcv_close_cols so that callers
    don't need a separate availability check.
    """
    if universe_schedule is not None and not universe_schedule.empty:
        te = pd.Timestamp(train_end)
        past_recons = universe_schedule[universe_schedule["recon_date"] <= te]
        if past_recons.empty:
            # No reconstitution happened before train_end — use all eligible
            all_members: set = set()
            for members_json in universe_schedule["members"]:
                all_members |= set(json.loads(members_json))
        else:
            all_members = set()
            for members_json in past_recons["members"]:
                all_members |= set(json.loads(members_json))
        tickers = sorted(all_members & set(ohlcv_close_cols))
        if not tickers:
            # Fallback to static if schedule yields nothing
            tickers = [t for t in universe_static if t in ohlcv_close_cols]
        return tickers
    else:
        return [t for t in universe_static if t in ohlcv_close_cols]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — WALK-FORWARD FOLD GENERATOR  (fix #1, #2)
# ═══════════════════════════════════════════════════════════════════════════════

def build_contiguous_folds(
    cfg:         Mahoraga4Config,
    trading_idx: pd.DatetimeIndex,
) -> List[Dict]:
    """
    Build walk-forward folds from explicit 5-tuple boundaries in cfg.wf_folds:
      (train_end, val_start, val_end, test_start, test_end)

    EXPANDING WALK-FORWARD INVARIANTS — asserted, not assumed:
      1. Within each fold, val_end < test_start  (fold-level purity)
      2. Test windows are mutually exclusive across folds
      3. Test windows are contiguous on the REAL trading calendar
      4. train_end + embargo_days <= val_start  (embargo respected)

    Important methodological note:
      In an expanding WFO, later validation windows may legitimately reuse
      earlier realised history, including periods that were test/OOS in prior
      folds. That does NOT create day-level look-ahead for any test block, but
      it does mean that the stitched OOS path should be interpreted as a
      sequentially revalidated trajectory rather than a globally untouched
      holdout after fold 1.

    Partial detection uses the REAL trading-day count from trading_idx.
    """
    assert len(cfg.wf_folds) > 0, "wf_folds is empty"

    for i, entry in enumerate(cfg.wf_folds):
        assert len(entry) == 5, (
            f"wf_folds[{i}] must be 5-tuple "
            f"(train_end, val_start, val_end, test_start, test_end), got len={len(entry)}"
        )

    parsed = [tuple(pd.Timestamp(s) for s in entry) for entry in cfg.wf_folds]

    folds = []
    prior_test_days: set = set()
    prior_test_windows = []

    for i, (te, vs, ve, ts, tend) in enumerate(parsed):
        fold_n = i + 1

        embargo_end = te + pd.offsets.BDay(cfg.embargo_days)
        assert vs >= embargo_end, (
            f"[fold {fold_n}] embargo violated: val_start={vs.date()} < "
            f"train_end+{cfg.embargo_days}bd={embargo_end.date()}"
        )

        assert ve < ts, (
            f"[fold {fold_n}] val_end={ve.date()} >= test_start={ts.date()}"
        )

        v_days = set(trading_idx[(trading_idx >= vs) & (trading_idx <= ve)].tolist())
        t_days = set(trading_idx[(trading_idx >= ts) & (trading_idx <= tend)].tolist())

        t_t_overlap = t_days & prior_test_days
        assert not t_t_overlap, (
            f"[fold {fold_n}] test {ts.date()}→{tend.date()} overlaps {len(t_t_overlap)} "
            f"days already in prior test windows. OOS exclusivity violated."
        )

        val_reuse_days = len(v_days & prior_test_days)
        if val_reuse_days > 0:
            print(
                f"  [fold {fold_n}] NOTE: validation reuses {val_reuse_days} days "
                f"that were OOS in prior folds (standard expanding WFO)."
            )

        avail_test = trading_idx[(trading_idx >= ts) & (trading_idx <= tend)]
        actual_td  = len(avail_test)
        is_partial = actual_td < int(cfg.trading_days * 0.90)

        fold = {
            "fold":                  fold_n,
            "train_start":           cfg.wf_train_start,
            "train_end":             str(te.date()),
            "val_start":             str(vs.date()),
            "val_end":               str(ve.date()),
            "test_start":            str(ts.date()),
            "test_end":              str(tend.date()),
            "is_partial":            is_partial,
            "actual_test_days":      actual_td,
            "val_reuses_prior_test": bool(val_reuse_days > 0),
            "val_reuse_days":        int(val_reuse_days),
        }
        folds.append(fold)
        prior_test_days |= t_days
        prior_test_windows.append((ts, tend))

        partial_str = " [PARTIAL]" if is_partial else ""
        print(f"  [fold {fold_n}] train:{cfg.wf_train_start}→{te.date()}  "
              f"val:{vs.date()}→{ve.date()}  "
              f"test:{ts.date()}→{tend.date()} ({actual_td}bd){partial_str}")

    for i in range(1, len(parsed)):
        prev_tend = parsed[i - 1][4]
        curr_ts   = parsed[i][3]
        nxt = trading_idx[trading_idx > prev_tend]
        assert len(nxt) > 0 and curr_ts == nxt[0], (
            f"Test windows not contiguous between fold {i} and fold {i+1}: "
            f"after {prev_tend.date()} expected {nxt[0].date() if len(nxt) else 'N/A'}, "
            f"got {curr_ts.date()}"
        )

    print("  [wf_purity] within-fold val<test ✓  test∩test=∅ ✓  contiguous OOS ✓  embargo ✓")
    return folds



def validate_oos_continuity(
    folds:        List[Dict],
    trading_idx:  pd.DatetimeIndex,
) -> Tuple[bool, str]:
    """
    FIX (#2): Validate OOS continuity against REAL trading calendar.
    Returns (is_continuous, label).
    """
    for i in range(1, len(folds)):
        prev_end  = pd.Timestamp(folds[i-1]["test_end"])
        curr_start = pd.Timestamp(folds[i]["test_start"])
        expected_next = trading_idx[trading_idx > prev_end]
        if len(expected_next) == 0:
            return False, "OOS_truncated"
        expected = expected_next[0]
        if curr_start != expected:
            return False, "OOS_stitched_with_gaps"
    return True, "OOS_continuous"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — COVARIANCE
# ═══════════════════════════════════════════════════════════════════════════════

def lw_cov(returns: pd.DataFrame) -> pd.DataFrame:
    lw = LedoitWolf().fit(returns.values)
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)

def cov_kappa(cov: pd.DataFrame) -> float:
    ev = np.linalg.eigvalsh(cov.values)
    ev = ev[ev > 0]
    return float(np.log10(ev.max() / ev.min())) if len(ev) > 1 else np.inf


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — HRP  (fix #7: robust to degenerate columns)
# ═══════════════════════════════════════════════════════════════════════════════

def _drop_degenerate_columns(
    returns: pd.DataFrame,
    min_unique: int = 10,
    min_std:    float = 1e-8,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    FIX (#7): Remove columns that are nearly constant or have insufficient
    unique values before HRP clustering — these produce non-finite distances.
    Returns (cleaned_df, list_of_dropped_tickers).
    """
    dropped = []
    for col in returns.columns:
        col_data = returns[col].dropna()
        if len(col_data.unique()) < min_unique or col_data.std() < min_std:
            dropped.append(col)
    clean = returns.drop(columns=dropped)
    if dropped:
        print(f"  [hrp] Dropped degenerate columns: {dropped}")
    return clean, dropped


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """
    HRP (Lopez de Prado 2016) with Ledoit-Wolf covariance.
    FIX (#7): Full robustness pipeline:
      1) Drop degenerate columns
      2) Clean correlation NaN/inf before distance matrix
      3) Validate condensed matrix is finite
      4) Fallback to inverse-vol if clustering fails
    """
    returns, dropped = _drop_degenerate_columns(returns)

    if returns.shape[1] == 0:
        return pd.Series(dtype=float)
    if returns.shape[1] == 1:
        return pd.Series([1.0], index=returns.columns)

    try:
        cov  = lw_cov(returns)
        corr = returns.corr()

        # Clean correlation matrix — .copy() ensures writable array
        corr_clean = corr.fillna(0.0).values.copy()
        np.fill_diagonal(corr_clean, 1.0)
        corr_clean = np.clip(corr_clean, -1.0, 1.0)
        corr_df    = pd.DataFrame(corr_clean, index=corr.index, columns=corr.columns)

        dm = np.sqrt(np.clip(0.5 * (1.0 - corr_clean), 0.0, 1.0)).copy()
        dm = np.nan_to_num(dm, nan=1.0, posinf=1.0, neginf=0.0)   # belt-and-suspenders
        dm = (dm + dm.T) / 2.0                                      # enforce symmetry
        np.fill_diagonal(dm, 0.0)

        dm_cond = squareform(dm, checks=False)

        # Validate finiteness — fallback if any non-finite remain
        if not np.all(np.isfinite(dm_cond)) or np.any(dm_cond < 0):
            raise ValueError("Non-finite or negative values in condensed distance matrix")

        lnk = linkage(dm_cond, method="single")

        def _qd(lm):
            lm = lm.astype(int)
            si = pd.Series([lm[-1, 0], lm[-1, 1]])
            n  = lm[-1, 3]
            while si.max() >= n:
                si.index = range(0, len(si) * 2, 2)
                df0 = si[si >= n]; ii = df0.index; jj = df0.values - n
                si[ii] = lm[jj, 0]
                si = pd.concat([si, pd.Series(lm[jj, 1], index=ii + 1)]).sort_index()
                si.index = range(len(si))
            return si.tolist()

        ordered = corr_df.index[_qd(lnk)]
        cov_ = cov.loc[ordered, ordered]
        w    = pd.Series(1.0, index=ordered)

        def _cv(cm, items):
            s  = cm.loc[items, items].values
            iv = 1.0 / np.diag(s); iv /= iv.sum()
            return float(iv @ s @ iv)

        clusters = [ordered.tolist()]
        while True:
            clusters = [c for c in clusters if len(c) > 1]
            if not clusters: break
            nc = []
            for c in clusters:
                s = len(c) // 2; c1, c2 = c[:s], c[s:]
                v1, v2 = _cv(cov_, c1), _cv(cov_, c2)
                a = 1.0 - v1 / (v1 + v2) if (v1 + v2) else 0.5
                w[c1] *= a; w[c2] *= (1.0 - a); nc += [c1, c2]
            clusters = nc

        w = (w / w.sum()).astype(float)

        # Re-attach dropped columns with zero weight
        if dropped:
            for d in dropped:
                w[d] = 0.0

        return w

    except Exception as e:
        print(f"  [hrp] Clustering failed ({e}), falling back to inverse-vol")
        # Formal fallback: inverse volatility weights
        vols = returns.std(ddof=1).replace(0, np.nan)
        inv  = (1.0 / vols).fillna(0.0)
        if inv.sum() == 0:
            inv = pd.Series(1.0 / len(returns.columns), index=returns.columns)
        else:
            inv = inv / inv.sum()
        if dropped:
            for d in dropped:
                inv[d] = 0.0
        return inv.astype(float)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════

def _trend(price: pd.Series, cfg: Mahoraga4Config) -> pd.Series:
    votes = []
    for sp in cfg.spans_short:
        for lp in cfg.spans_long:
            if sp >= lp: continue
            es = price.ewm(span=sp, adjust=False).mean().shift(1)
            el = price.ewm(span=lp, adjust=False).mean().shift(1)
            votes.append(((es > el) & price.notna()).astype(float))
    return (sum(votes) / len(votes)) if votes else pd.Series(0.0, index=price.index)

def _mom(price: pd.Series, cfg: Mahoraga4Config) -> pd.Series:
    raw = sum((price / price.shift(w) - 1.0).shift(1) for w in cfg.mom_windows) / len(cfg.mom_windows)
    return (raw.clip(-1.0, 1.0) + 1.0) / 2.0

def _rel(price: pd.Series, bench: pd.Series, cfg: Mahoraga4Config) -> pd.Series:
    raw = sum(
        (price / price.shift(w) - 1.0).shift(1) - (bench / bench.shift(w) - 1.0).shift(1)
        for w in cfg.rel_windows
    ) / len(cfg.rel_windows)
    return (raw.clip(-1.0, 1.0) + 1.0) / 2.0


def fit_ic_weights(
    close:       pd.DataFrame,
    qqq:         pd.Series,
    cfg:         Mahoraga4Config,
    train_start: str,
    train_end:   str,
    horizons:    Tuple[int, ...] = (1, 5, 21),
) -> Tuple[float, float, float]:
    sub  = close.loc[train_start:train_end]
    qqq_ = to_s(qqq.loc[train_start:train_end].ffill())
    idx  = sub.index
    ic_by_horizon: Dict[int, List[float]] = {h: [] for h in horizons}

    for t in sub.columns:
        p  = sub[t].ffill()
        tr = _trend(p, cfg).reindex(idx)
        mo = _mom(p, cfg).reindex(idx)
        re = _rel(p, qqq_.reindex(idx).ffill(), cfg).reindex(idx)
        for h in horizons:
            fwd = sub[t].pct_change().shift(-h).reindex(idx)
            ok  = fwd.notna()
            if ok.sum() < 50: continue
            ics = []
            for sig in [tr, mo, re]:
                common = sig.notna() & ok
                if common.sum() < 30: continue
                r, _ = stats.spearmanr(sig[common], fwd[common])
                ics.append(float(r) if np.isfinite(r) else 0.0)
            if len(ics) == 3:
                for ic_val in ics:
                    ic_by_horizon[h].append(ic_val)

    trend_ics, mom_ics, rel_ics = [], [], []
    for h in horizons:
        vals = ic_by_horizon[h]
        if not vals: continue
        n3 = (len(vals) // 3) * 3; vals = vals[:n3]
        for i in range(0, n3, 3):
            trend_ics.append(vals[i]); mom_ics.append(vals[i+1]); rel_ics.append(vals[i+2])

    ic = np.array([
        max(np.nanmean(trend_ics) if trend_ics else 0.01, 0.005),
        max(np.nanmean(mom_ics)   if mom_ics   else 0.01, 0.005),
        max(np.nanmean(rel_ics)   if rel_ics   else 0.01, 0.005),
    ])
    w_raw = np.exp(ic * 20.0) / np.exp(ic * 20.0).sum()
    w     = 0.4 * np.array([1/3, 1/3, 1/3]) + 0.6 * w_raw
    w     = np.clip(w, 0.10, 0.70); w /= w.sum()
    print(f"    [IC] trend={w[0]:.3f} mom={w[1]:.3f} rel={w[2]:.3f}  "
          f"(raw IC @{horizons}: {ic[0]:.4f}/{ic[1]:.4f}/{ic[2]:.4f})")
    return float(w[0]), float(w[1]), float(w[2])


def compute_scores(close: pd.DataFrame, qqq: pd.Series, cfg: Mahoraga4Config) -> pd.DataFrame:
    idx  = close.index
    qqq_ = to_s(qqq, "QQQ").reindex(idx).ffill()
    sc   = pd.DataFrame(index=idx, columns=close.columns, dtype=float)
    for t in close.columns:
        p = close[t].reindex(idx).ffill()
        s = cfg.w_trend * _trend(p, cfg) + cfg.w_mom * _mom(p, cfg) + cfg.w_rel * _rel(p, qqq_, cfg)
        s.iloc[:cfg.burn_in] = 0.0
        sc[t] = s.fillna(0.0)
    return sc.fillna(0.0)


def rolling_ic_multi_horizon(
    close:   pd.DataFrame,
    qqq:     pd.Series,
    cfg:     Mahoraga4Config,
    window:  int = 63,
    horizons: Tuple[int, ...] = (1, 5, 21),
) -> pd.DataFrame:
    idx  = close.index
    qqq_ = to_s(qqq, "QQQ").reindex(idx).ffill()
    ic_df = pd.DataFrame(index=idx, dtype=float)
    for h in horizons:
        fwd  = close.pct_change().shift(-h)
        comp = pd.DataFrame(index=idx, columns=close.columns, dtype=float)
        for t in close.columns:
            p = close[t].reindex(idx).ffill()
            comp[t] = cfg.w_trend*_trend(p,cfg)+cfg.w_mom*_mom(p,cfg)+cfg.w_rel*_rel(p,qqq_,cfg)
        ic_s = pd.Series(np.nan, index=idx)
        for i in range(len(idx)):
            s_ = comp.iloc[i].dropna(); f_ = fwd.iloc[i][s_.index].dropna()
            c_ = s_.index.intersection(f_.index)
            if len(c_) >= 4:
                r, _ = stats.spearmanr(s_[c_], f_[c_])
                if np.isfinite(r): ic_s.iloc[i] = r
        ic_df[f"IC_composite_{h}d"] = ic_s.rolling(window).mean().fillna(0.0)
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
            last  = np.zeros(len(order)); last[order[:k]] = 1.0
        mask.loc[dt] = last
    return mask.fillna(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — RISK OVERLAYS
# ═══════════════════════════════════════════════════════════════════════════════

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev = close.shift(1)
    tr   = pd.concat([(high-low).abs(), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(span=2*window-1, adjust=False).mean()


def apply_chandelier(
    weights: pd.DataFrame,
    close:   pd.DataFrame,
    high:    pd.DataFrame,
    low:     pd.DataFrame,
    cfg:     Mahoraga4Config,
) -> Tuple[pd.DataFrame, int]:
    if not cfg.stop_on:
        return weights.copy(), 0
    out  = weights.copy(); idx = out.index
    reb  = set(out.resample(cfg.rebalance_freq).last().index); hits = 0
    for t in out.columns:
        p    = close[t].reindex(idx).ffill()
        atr_ = _atr(high[t].reindex(idx).ffill(), low[t].reindex(idx).ffill(),
                    p, cfg.atr_window).bfill().fillna(0.0)
        wt = out[t].values.copy()
        in_pos = stopped = False; maxp = np.nan; last_sl = np.nan
        for i, dt in enumerate(idx):
            if dt in reb and stopped:
                rec_thr = (last_sl + cfg.reentry_atr_buffer * float(atr_.iloc[i])
                           if np.isfinite(last_sl) else -np.inf)
                if float(p.iloc[i]) > rec_thr: stopped = False
            if wt[i] <= 0:
                in_pos = stopped = False; maxp = np.nan; continue
            if stopped: wt[i] = 0.0; continue
            if not in_pos: in_pos = True; maxp = float(p.iloc[i])
            else: maxp = max(maxp, float(p.iloc[i]))
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
    cfg:         Mahoraga4Config,
) -> Tuple[float, float]:
    p_tr = to_s(qqq_close.loc[train_start:train_end]).ffill()
    r_tr = p_tr.pct_change().fillna(0.0)
    dd_tr  = p_tr / p_tr.cummax() - 1.0
    dd_thr = float(abs(np.nanpercentile(dd_tr, cfg.crisis_dd_pct * 100)))
    dd_thr = max(dd_thr, 0.10)
    vol_tr = r_tr.rolling(63).std() * np.sqrt(cfg.trading_days)
    vol_thr = float(np.nanpercentile(safe_z(vol_tr, 252).dropna(),
                                      cfg.crisis_vol_pct * 100))
    vol_thr = max(vol_thr, 0.5)
    print(f"  [crisis] DD_thr={dd_thr:.3f}  vol_z_thr={vol_thr:.3f}  "
          f"(calibrated on {train_start}→{train_end})")
    return dd_thr, vol_thr


def compute_crisis_gate(
    qqq_close: pd.Series,
    cfg:       Mahoraga4Config,
) -> Tuple[pd.Series, pd.Series]:
    p   = to_s(qqq_close, "QQQ").ffill(); idx = p.index
    r   = p.pct_change().fillna(0.0)
    vol = r.rolling(cfg.port_vol_window).std() * np.sqrt(cfg.trading_days)
    vol_z = safe_z(vol, cfg.port_vol_window * 4)
    dd    = p / p.cummax() - 1.0
    cond     = ((dd <= -cfg.crisis_dd_thr) | (vol_z >= cfg.crisis_vol_zscore_thr)).astype(int)
    on_flag  = cond.rolling(cfg.crisis_min_days_on).mean().fillna(0.0) >= 0.8
    off_flag = (1-cond).rolling(cfg.crisis_min_days_off).mean().fillna(0.0) >= 0.8
    state = pd.Series(0.0, index=idx); in_c = False
    for dt in idx:
        if not in_c and bool(on_flag.loc[dt]):   in_c = True
        elif in_c and bool(off_flag.loc[dt]):    in_c = False
        state.loc[dt] = 1.0 if in_c else 0.0
    scale = pd.Series(1.0, index=idx, dtype=float)
    if cfg.crisis_gate_use: scale[state == 1.0] = cfg.crisis_scale
    scale.iloc[:cfg.burn_in] = cfg.crisis_scale
    state.iloc[:cfg.burn_in] = 1.0
    return scale, state


def compute_turbulence(
    close:  pd.DataFrame,
    volume: pd.DataFrame,
    qqq:    pd.Series,
    cfg:    Mahoraga4Config,
) -> pd.Series:
    idx   = close.index
    qqq_r = to_s(qqq, "QQQ").reindex(idx).ffill().pct_change().fillna(0.0)
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
        c   = sub.corr().values; n = c.shape[0]
        avg_corr.loc[dt] = (c.sum()-n)/(n*(n-1)) if n > 1 else 0.0
    avg_corr = avg_corr.replace(0, np.nan).ffill().fillna(0.0)
    corr_z   = safe_z(avg_corr, 252)
    dv = (close * volume).replace(0, np.nan)
    illiq = (rets.abs() / dv).replace([np.inf, -np.inf], np.nan)
    illiq_avg = np.log1p(illiq.rolling(cfg.illiq_window).mean().mean(axis=1).fillna(0.0))
    illiq_z   = safe_z(illiq_avg, 252)
    turb = (vol_z + corr_z + illiq_z).ewm(span=10, adjust=False).mean()
    s    = pd.Series(1.0/(1.0+np.exp(1.2*(turb-cfg.turb_zscore_thr))), index=idx)
    s    = s.clip(lower=cfg.turb_scale_min, upper=1.0)
    s.iloc[:cfg.burn_in] = cfg.turb_scale_min
    return s


def vol_target_scale(port_r: pd.Series, cfg: Mahoraga4Config) -> pd.Series:
    if not cfg.vol_target_on:
        return pd.Series(1.0, index=port_r.index)
    rv = to_s(port_r).fillna(0.0).rolling(cfg.port_vol_window).std() * np.sqrt(cfg.trading_days)
    s  = (cfg.vol_target_ann / rv).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return s.clip(cfg.min_exposure, cfg.max_exposure)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — CORE BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

def _costs(w: pd.DataFrame, c: CostsConfig) -> Tuple[pd.Series, pd.Series]:
    dw  = w.diff().abs().fillna(0.0)
    to  = 0.5 * dw.sum(axis=1)
    slip = c.slippage * c.gap_factor if c.apply_slippage else 0.0
    tc  = to * (c.commission + slip)
    return to, tc


def backtest(
    ohlcv:             Dict[str, pd.DataFrame],
    cfg:               Mahoraga4Config,
    costs:             CostsConfig,
    label:             str = "MAHORAGA_4",
    universe:          Optional[List[str]] = None,
    universe_schedule: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Core backtest.

    Universe modes (fix #2 — universe governs simulation):
      1. universe_schedule is provided (canonical engine, point-in-time):
         At each rebalance date, the active universe is looked up from the
         schedule. Scores, HRP, and weights are computed only over tickers
         active at that date. This is the honest mode.
      2. universe list provided: fixed universe, used throughout.
      3. Neither provided: falls back to cfg.universe_static.

    The universe_schedule overrides universe and cfg.universe_static when set.
    """
    np.random.seed(cfg.random_seed)

    # Determine master ticker list (superset across all dates)
    if universe_schedule is not None:
        # Collect all tickers that ever appear in the schedule
        all_sched_tickers = set()
        for members_json in universe_schedule["members"]:
            all_sched_tickers |= set(json.loads(members_json))
        univ_master = sorted(all_sched_tickers & set(ohlcv["close"].columns))
        use_pit_universe = True
    elif universe is not None:
        univ_master = [t for t in universe if t in ohlcv["close"].columns]
        use_pit_universe = False
    else:
        univ_master = [t for t in cfg.universe_static if t in ohlcv["close"].columns]
        use_pit_universe = False

    if not univ_master:
        raise ValueError("[backtest] No valid tickers in universe")

    close  = ohlcv["close"][univ_master].copy()
    high   = ohlcv["high"][univ_master].copy()
    low    = ohlcv["low"][univ_master].copy()
    volume = ohlcv["volume"][univ_master].copy()
    idx    = close.index

    qqq = to_s(ohlcv["close"][cfg.bench_qqq].reindex(idx).ffill(), "QQQ")
    spy = to_s(ohlcv["close"][cfg.bench_spy].reindex(idx).ffill(), "SPY")

    crisis_scale, crisis_state = compute_crisis_gate(qqq, cfg)
    turb_scale                  = compute_turbulence(close, volume, qqq, cfg)
    # Compute scores over full master universe
    scores                      = compute_scores(close, qqq, cfg)
    # In static mode, active selection works over full master universe
    active                      = select_topk(scores, cfg.top_k, cfg.rebalance_freq)
    rets                        = close.pct_change().fillna(0.0)
    reb_dates                   = set(close.resample(cfg.rebalance_freq).last().index)

    w      = pd.DataFrame(0.0, index=idx, columns=univ_master)
    last_w = pd.Series(0.0, index=univ_master)

    for dt in idx:
        if dt in reb_dates:
            # ── Point-in-time universe (fix #2: schedule governs simulation) ──
            if use_pit_universe:
                pit_members = get_universe_at_date(universe_schedule, dt)
                pit_members = [t for t in pit_members if t in univ_master]
                if not pit_members:
                    last_w = pd.Series(0.0, index=univ_master)
                    continue
                # Restrict scores to PIT universe at this date
                pit_scores = scores.loc[dt, pit_members] if pit_members else pd.Series(dtype=float)
                sel_names  = pit_scores.nlargest(cfg.top_k).index.tolist()
                names      = [n for n in sel_names if pit_scores.get(n, 0) > 0]
            else:
                sel   = active.loc[dt]
                names = sel[sel > 0].index.tolist()
            if not names:
                last_w = pd.Series(0.0, index=univ_master)
            elif len(names) == 1:
                last_w = pd.Series(0.0, index=univ_master)
                last_w[names[0]] = 1.0
            else:
                lb = rets.loc[:dt].tail(cfg.hrp_window)[names].dropna()
                if len(lb) < 60:
                    # hrp_weights handles min_rows gracefully; keep path consistent
                    _ret_fallback = (lb if len(lb) > len(names)
                                     else rets.loc[:dt][names].dropna())
                    ww = hrp_weights(_ret_fallback)
                    ww = ww.reindex(names, fill_value=0.0)
                else:
                    ww = hrp_weights(lb)
                    ww = ww.reindex(names, fill_value=0.0)
                if ww.sum() > 0:
                    ww = ww.clip(upper=cfg.weight_cap) / ww.clip(upper=cfg.weight_cap).sum()
                last_w = pd.Series(0.0, index=univ_master)
                last_w[names] = ww.reindex(names, fill_value=0.0).values
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
    port_net  = ((w_exec * rets).sum(axis=1) - tc).replace([np.inf,-np.inf], 0.0).fillna(0.0)
    equity    = cfg.capital_initial * (1.0 + port_net).cumprod()
    exposure  = w_exec.abs().sum(axis=1).clip(0.0, cfg.max_exposure)

    qqq_r  = qqq.pct_change().fillna(0.0) - costs.qqq_expense_ratio
    spy_r  = spy.pct_change().fillna(0.0)
    qqq_eq = cfg.capital_initial * (1.0 + qqq_r).cumprod()
    spy_eq = cfg.capital_initial * (1.0 + spy_r).cumprod()

    return {
        "label": label, "returns_net": port_net, "equity": equity,
        "exposure": exposure, "turnover": to,
        "weights_scaled": w_exec, "total_scale": exec_sc, "total_scale_target": tgt_sc,
        "cap": cap, "turb_scale": turb_scale, "crisis_scale": crisis_scale,
        "crisis_state": crisis_state, "vol_scale": vol_sc,
        "stop_hits": stop_hits, "scores": scores,
        "bench": {"QQQ_r": qqq_r, "QQQ_eq": qqq_eq, "SPY_r": spy_r, "SPY_eq": spy_eq},
    }


def validate_no_lookahead(res: Dict, label: str = ""):
    exec_sc = to_s(res["total_scale"]).fillna(0.0)
    tgt_sc  = to_s(res["total_scale_target"]).fillna(0.0)
    diff    = float((exec_sc - tgt_sc.shift(1).fillna(0.0)).abs().max())
    assert diff < 1e-10, f"[{label}] look-ahead in total_scale (max_diff={diff:.2e})"
    w    = res["weights_scaled"]
    to_r = 0.5 * w.diff().abs().fillna(0.0).sum(axis=1)
    to_s_= to_s(res["turnover"]).fillna(0.0)
    diff2 = float((to_r - to_s_).abs().max())
    assert diff2 < 1e-8, f"[{label}] turnover mismatch (max_diff={diff2:.2e})"
    print(f"  [OK] no look-ahead [{label}]")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def summarize(r, eq, exp, to, cfg: Mahoraga4Config, label="") -> Dict:
    r    = to_s(r).replace([np.inf,-np.inf], np.nan).dropna()
    eq   = to_s(eq).dropna()
    exp_ = to_s(exp).reindex(r.index).fillna(0.0) if exp is not None else pd.Series(np.nan, index=r.index)
    to_  = to_s(to).reindex(r.index).fillna(0.0) if to  is not None else pd.Series(0.0, index=r.index)
    T    = len(r)
    return {
        "Label":       label,
        "FinalEquity": float(eq.iloc[-1]) if len(eq) else np.nan,
        "TotalReturn": total_ret(r),
        "CAGR":        annualize(r, cfg.trading_days),
        "AnnVol":      ann_vol(r, cfg.trading_days),
        "Sharpe":      sharpe(r, cfg.rf_annual, cfg.trading_days),
        "Sortino":     sortino(r, cfg.rf_annual, cfg.trading_days),
        "MaxDD":       max_dd(eq),
        "Calmar":      calmar(r, eq, cfg.trading_days),
        "CVaR_5":      cvar95(r),
        "AvgExposure": float(exp_.mean()),
        "TimeInMkt":   float((exp_ > 0).mean()),
        "TurnoverAnn": float(to_.sum() * cfg.trading_days / T) if T else 0.0,
        "Days":        int(T),
    }


def asymptotic_sharpe_ci(r: pd.Series, cfg: Mahoraga4Config, alpha: float = 0.05) -> Dict:
    r    = to_s(r).dropna(); T = len(r)
    rf_d = (1.0 + cfg.rf_annual) ** (1.0 / cfg.trading_days) - 1.0
    ex   = r - rf_d
    sr_d = ex.mean() / ex.std(ddof=1) if ex.std(ddof=1) > 0 else 0.0
    se_d = np.sqrt((1.0 + sr_d**2 / 2.0) / T)
    z    = stats.norm.ppf(1.0 - alpha/2.0)
    td   = cfg.trading_days
    sr_a = sr_d * np.sqrt(td); se_a = se_d * np.sqrt(td)
    t_s  = sr_d / se_d if se_d > 0 else 0.0
    return {
        "SR": round(sr_a, 4), "CI_lo": round(sr_a - z*se_a, 4),
        "CI_hi": round(sr_a + z*se_a, 4), "SE": round(se_a, 4),
        "t_stat": round(t_s, 3), "p_val": round(2.0*(1.0-stats.norm.cdf(abs(t_s))), 6),
        "note": "Asymptotic (delta-method); underestimates SE under autocorrelation",
    }


def alpha_test_nw(r_s, r_b, cfg: Mahoraga4Config, label: str = "",
                  conditional: bool = False, exposure: Optional[pd.Series] = None) -> Dict:
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
        ols = sm.OLS(r_s.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        a_d = float(ols.params[0])
        a_a = float((1.0 + a_d)**cfg.trading_days - 1.0)
        return {
            "Label": label, "conditional": conditional,
            "alpha_ann": round(a_a, 6), "t_alpha": round(float(ols.tvalues[0]), 3),
            "p_alpha": round(float(ols.pvalues[0]), 6),
            "beta": round(float(ols.params[1]), 4), "R2": round(float(ols.rsquared), 4),
            "NW_lags": lags, "n_obs": int(len(r_s)),
            "sig_5pct": bool(ols.pvalues[0] < 0.05), "sig_1pct": bool(ols.pvalues[0] < 0.01),
        }
    except Exception as e:
        return {"Label": label, "error": str(e)}


def factor_attribution(r_s, ff, cfg: Mahoraga4Config, label="") -> Optional[Dict]:
    if ff is None: return None
    r   = to_s(r_s).dropna()
    ff_ = ff.reindex(r.index).dropna()
    c   = r.index.intersection(ff_.index)
    if len(c) < 252: return {"Label": label, "error": "Insufficient FF data"}
    y    = r[c].values - ff_.loc[c, "RF"].values
    X    = sm.add_constant(ff_.loc[c, ["Mkt-RF","SMB","HML","RMW","CMA","UMD"]].values)
    lags = int(4*(len(y)/100)**(2.0/9.0))
    try:
        ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        a_a = float((1.0 + float(ols.params[0]))**cfg.trading_days - 1.0)
        p   = ols.params; tv = ols.tvalues; pv = ols.pvalues
        return {
            "Label": label, "alpha_ann": round(a_a, 6),
            "t_alpha": round(float(tv[0]), 3), "p_alpha": round(float(pv[0]), 6),
            "beta_mkt": round(float(p[1]), 4), "beta_smb": round(float(p[2]), 4),
            "beta_hml": round(float(p[3]), 4), "beta_rmw": round(float(p[4]), 4),
            "beta_cma": round(float(p[5]), 4), "beta_umd": round(float(p[6]), 4),
            "R2_adj": round(float(ols.rsquared_adj), 4),
        }
    except Exception as e:
        return {"Label": label, "error": str(e)}


def regime_analysis(r_s, r_b, ohlcv, cfg: Mahoraga4Config) -> pd.DataFrame:
    idx = to_s(r_s).index
    vix_available = cfg.bench_vix in ohlcv.get("close", pd.DataFrame()).columns
    if vix_available:
        vix = to_s(ohlcv["close"][cfg.bench_vix].reindex(idx).ffill())
    else:
        vix = (to_s(r_b).fillna(0.0).rolling(21).std()*np.sqrt(252)*100).reindex(idx).ffill()
    vix_label = "VIX" if vix_available else "realized-vol proxy"
    p20 = float(np.nanpercentile(vix, 20))
    p80 = float(np.nanpercentile(vix, 80))
    regimes = {
        f"CALM (<{p20:.0f})":  vix < p20,
        f"NORMAL ({p20:.0f}–{p80:.0f})": (vix >= p20) & (vix < p80),
        f"STRESS (≥{p80:.0f})": vix >= p80,
    }
    if vix_available:
        vix_panic_thr = float(getattr(cfg, "corr_vix_thr", 24.0))
        regimes = {f"PANIC (VIX≥{vix_panic_thr:.0f})": vix >= vix_panic_thr, **regimes}

    rows = []
    for name, mask in regimes.items():
        mask = mask.reindex(to_s(r_s).index).fillna(False)
        rs   = to_s(r_s)[mask]; rb = to_s(r_b).reindex(rs.index).fillna(0.0)
        if len(rs) < 20: continue
        eq = cfg.capital_initial * (1.0 + rs).cumprod()
        rows.append({
            "Regime": name, "VIX_source": "real" if vix_available else "proxy",
            "Days": int(len(rs)), "Days%": round(100*len(rs)/len(to_s(r_s)), 1),
            "CAGR%": round(annualize(rs, cfg.trading_days)*100, 2),
            "CAGR_bench%": round(annualize(rb, cfg.trading_days)*100, 2),
            "Excess_CAGR%": round((annualize(rs, cfg.trading_days)-annualize(rb, cfg.trading_days))*100, 2),
            "Sharpe": round(sharpe(rs, cfg.rf_annual, cfg.trading_days), 3),
            "MaxDD%": round(max_dd(eq)*100, 2),
            "Hit_Rate%": round(100*(rs > 0).mean(), 1),
        })
    return pd.DataFrame(rows)


def stress_report(r, exp, episodes, cfg: Mahoraga4Config, r_bench=None) -> pd.DataFrame:
    rows = []
    for name, (a, b) in episodes.items():
        sub = to_s(r).loc[a:b]
        if len(sub) < 40: continue
        ee  = cfg.capital_initial * (1.0 + sub).cumprod()
        ss  = summarize(sub, ee, to_s(exp).loc[sub.index], None, cfg)
        bt  = None
        if r_bench is not None:
            rb = to_s(r_bench).loc[a:b]; bt = round((1.0+rb).prod()-1.0, 4)
        rows.append({
            "Episode": name, "Days": int(len(sub)),
            "Total%": round((1.0+sub).prod()*100-100, 2),
            "Bench_Total%": round(bt*100, 2) if bt is not None else None,
            "Excess%": round(((1+sub).prod()-(1+to_s(r_bench).loc[a:b]).prod())*100, 2) if r_bench is not None else None,
            "WorstDay%": round(sub.min()*100, 2),
            "Sharpe": round(ss["Sharpe"], 3), "MaxDD%": round(ss["MaxDD"]*100, 2),
            "Calmar": round(ss["Calmar"], 3), "AvgExp%": round(ss["AvgExposure"]*100, 1),
        })
    return pd.DataFrame(rows)


def moving_block_bootstrap(r, block=20, n=1000, seed=42) -> Dict:
    rng = np.random.default_rng(seed)
    x   = to_s(r).replace([np.inf,-np.inf], np.nan).dropna().values
    T   = len(x)
    if T < block*5:
        return {"dd_p50": np.nan, "dd_p5_worst": np.nan, "ruin_prob_30dd": np.nan, "ruin_prob_50dd": np.nan}
    dds = []
    for _ in range(n):
        st  = rng.integers(0, T-block, size=int(np.ceil(T/block)))
        smp = np.concatenate([x[s:s+block] for s in st])[:T]
        eq  = np.cumprod(1.0 + smp)
        dds.append(float(np.min(eq/np.maximum.accumulate(eq)-1.0)))
    dds = np.array(dds)
    return {
        "dd_p50": float(np.quantile(dds, 0.50)),
        "dd_p5_worst": float(np.quantile(dds, 0.05)),
        "dd_p1_worst": float(np.quantile(dds, 0.01)),
        "ruin_prob_30dd": float(np.mean(dds < -0.30))*100.0,
        "ruin_prob_50dd": float(np.mean(dds < -0.50))*100.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — BASELINES
# ═══════════════════════════════════════════════════════════════════════════════

def baseline_eqw(
    ohlcv:             Dict,
    cfg:               Mahoraga4Config,
    costs:             CostsConfig,
    universe_schedule: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Equal-weight baseline.
    · PIT mode: weights constructed from pit_members at each rebalance date.
    · Static mode: equal-weight over cfg.universe_static throughout.
    Benchmark lives under the same universe regime as the strategy.
    """
    all_cols = (
        sorted(set().union(*[json.loads(r) for r in universe_schedule["members"]])
               & set(ohlcv["close"].columns))
        if universe_schedule is not None and not universe_schedule.empty
        else [t for t in cfg.universe_static if t in ohlcv["close"].columns]
    )
    close = ohlcv["close"][all_cols].copy()
    idx   = close.index
    rets  = close.pct_change().fillna(0.0)
    reb_dates = set(close.resample(cfg.rebalance_freq).last().index)

    if universe_schedule is not None and not universe_schedule.empty:
        w_df = pd.DataFrame(0.0, index=idx, columns=all_cols)
        last_w = pd.Series(0.0, index=all_cols)
        for dt in idx:
            if dt in reb_dates:
                pit = [t for t in get_universe_at_date(universe_schedule, dt)
                       if t in all_cols]
                if pit:
                    last_w = pd.Series(0.0, index=all_cols)
                    for t in pit: last_w[t] = 1.0 / len(pit)
            w_df.loc[dt] = last_w.values
    else:
        n = len(all_cols)
        w_df = pd.DataFrame(1.0 / n, index=idx, columns=all_cols)

    we   = w_df.shift(1).fillna(0.0)
    r_n  = ((we * rets).sum(axis=1) - _costs(we, costs)[1]).fillna(0.0)
    eq   = cfg.capital_initial * (1.0 + r_n).cumprod()
    return {"r": r_n, "eq": eq, "exp": we.abs().sum(axis=1),
            "to": _costs(we, costs)[0], "label": "EQW_Tech"}


def baseline_mom(
    ohlcv:             Dict,
    cfg:               Mahoraga4Config,
    costs:             CostsConfig,
    universe_schedule: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    12-1 momentum baseline.
    · PIT mode: top-k selected from pit_members at each rebalance date.
    · Static mode: top-k from cfg.universe_static throughout.
    Benchmark lives under the same universe regime as the strategy.
    """
    all_cols = (
        sorted(set().union(*[json.loads(r) for r in universe_schedule["members"]])
               & set(ohlcv["close"].columns))
        if universe_schedule is not None and not universe_schedule.empty
        else [t for t in cfg.universe_static if t in ohlcv["close"].columns]
    )
    close = ohlcv["close"][all_cols].copy()
    idx   = close.index
    rets  = close.pct_change().fillna(0.0)
    reb_dates = set(close.resample(cfg.rebalance_freq).last().index)

    # 12-1 momentum: skip last month to avoid reversal bias
    mom_raw = (close.shift(21) / close.shift(273) - 1.0)

    w_df = pd.DataFrame(0.0, index=idx, columns=all_cols)
    last_w = pd.Series(0.0, index=all_cols)
    for dt in idx:
        if dt in reb_dates:
            if universe_schedule is not None and not universe_schedule.empty:
                pool = [t for t in get_universe_at_date(universe_schedule, dt)
                        if t in all_cols]
            else:
                pool = all_cols
            if not pool:
                last_w = pd.Series(0.0, index=all_cols)
            else:
                m = mom_raw.loc[dt, pool].dropna()
                top = m.nlargest(min(cfg.top_k, len(m)))
                if top.empty:
                    last_w = pd.Series(0.0, index=all_cols)
                else:
                    last_w = pd.Series(0.0, index=all_cols)
                    for t in top.index: last_w[t] = 1.0 / len(top)
        w_df.loc[dt] = last_w.values

    we     = w_df.shift(1).fillna(0.0)
    to, tc = _costs(we, costs)
    r_n    = ((we * rets).sum(axis=1) - tc).fillna(0.0)
    eq     = cfg.capital_initial * (1.0 + r_n).cumprod()
    return {"r": r_n, "eq": eq, "exp": we.abs().sum(axis=1), "to": to, "label": "MOM_12_1_TopK"}


def baseline_signal_decomp(
    ohlcv:             Dict,
    cfg:               Mahoraga4Config,
    costs:             CostsConfig,
    universe_schedule: Optional[pd.DataFrame] = None,
) -> Dict[str, Dict]:
    results = {}
    for comp, (wt, wm, wr) in [
        ("TREND_ONLY", (1.0, 0.0, 0.0)),
        ("MOM_ONLY",   (0.0, 1.0, 0.0)),
        ("REL_ONLY",   (0.0, 0.0, 1.0)),
    ]:
        c2 = deepcopy(cfg); c2.w_trend=wt; c2.w_mom=wm; c2.w_rel=wr
        results[comp] = backtest(
            ohlcv, c2, costs, label=comp,
            universe_schedule=universe_schedule
        )
    return results



# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — SWEEP & OBJECTIVE  (fixes #3, #4, #5; BHY q-values; parallel)
# ═══════════════════════════════════════════════════════════════════════════════

def objective(s_val: Dict, s_qqq: Dict, cfg: Mahoraga4Config) -> float:
    exc_cagr   = s_val["CAGR"]    - s_qqq["CAGR"]
    exc_sharpe = s_val["Sharpe"]  - s_qqq["Sharpe"]
    exc_sort   = s_val["Sortino"] - s_qqq["Sortino"]
    exc_calmar = s_val["Calmar"]  - s_qqq["Calmar"]
    dd_pen = cfg.dd_penalty_strength * max(0.0, abs(s_val["MaxDD"]) - cfg.target_maxdd)
    to_pen = cfg.turnover_penalty   * max(0.0, s_val["TurnoverAnn"] - cfg.turnover_soft_cap)
    return float(1.00*exc_cagr + 0.20*exc_sharpe + 0.20*exc_sort
                 + cfg.calmar_weight*exc_calmar - dd_pen - to_pen)


def bhy_q_values(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg-Yekutieli (BHY) correction.
    Returns (q_values, significant_mask_at_alpha).
    A low q_value provides statistical support under FDR control — it does NOT
    imply certainty or absence of data-mining bias.
    """
    m    = len(p_values)
    c_m  = np.sum(1.0 / np.arange(1, m+1))
    order    = np.argsort(p_values)
    p_sorted = p_values[order]
    ranks    = np.arange(1, m+1)
    q_raw    = p_sorted * (m * c_m) / ranks
    q_mono   = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_clamp  = np.clip(q_mono, 0.0, 1.0)
    q_result = np.empty(m); q_result[order] = q_clamp
    return q_result, q_result <= alpha


SWEEP_GRID = {
    "weight_cap":      [0.55, 0.65],
    "k_atr":           [2.0, 2.5, 3.0],
    "turb_zscore_thr": [1.0, 1.5],
    "turb_scale_min":  [0.25, 0.40],
    "vol_target_ann":  [0.25, 0.30, 0.35],
}


def _evaluate_single_combo(
    combo:             Tuple,
    keys:              List[str],
    cfg_base:          Mahoraga4Config,
    ic_weights:        Tuple[float, float, float],
    ohlcv:             Dict,
    costs:             CostsConfig,
    val_start:         str,
    val_end:           str,
    fold_n:            int,
    base_seed:         int,
    universe_schedule: Optional[pd.DataFrame] = None,
) -> Optional[Dict]:
    """
    Worker for one sweep combination. Deterministic seed = base_seed XOR md5(combo).
    NOTE (fix #8): passed as argument to loky worker — on Windows/macOS ohlcv
    is fully pickled per process. Set parallel_sweep=False if RAM is constrained.
    universe_schedule is passed through to backtest for PIT universe mode.
    """
    kw  = dict(zip(keys, combo))
    cfg = deepcopy(cfg_base)
    for k, v in kw.items(): setattr(cfg, k, v)
    wt, wm, wr = ic_weights
    cfg.w_trend = wt; cfg.w_mom = wm; cfg.w_rel = wr
    combo_int   = int(hashlib.md5(json.dumps(kw, sort_keys=True).encode()).hexdigest(), 16) % (2**31)
    cfg.random_seed = (base_seed + combo_int) % (2**31)
    try:
        res = backtest(ohlcv, cfg, costs, label=f"sweep_f{fold_n}",
                       universe_schedule=universe_schedule)
        r_v = res["returns_net"].loc[val_start:val_end]
        if len(r_v) < 50: return None
        eq_v  = cfg.capital_initial * (1.0 + r_v).cumprod()
        exp_v = res["exposure"].loc[r_v.index]
        to_v  = res["turnover"].loc[r_v.index]
        s_v   = summarize(r_v, eq_v, exp_v, to_v, cfg)
        qr    = res["bench"]["QQQ_r"].loc[val_start:val_end]
        s_q   = summarize(qr, cfg.capital_initial*(1.0+qr).cumprod(), None, None, cfg)
        ci    = asymptotic_sharpe_ci(r_v, cfg)
        sc    = objective(s_v, s_q, cfg)
        # FIX (#5): store combo as an explicit dict — no string transform needed
        return {
            "combo_params": kw,           # exact combo dict for reproducibility
            "fold": fold_n, "score_val": sc,
            "VAL_Sharpe":   round(s_v["Sharpe"], 4),
            "VAL_CAGR%":    round(s_v["CAGR"]*100, 3),
            "VAL_MaxDD%":   round(s_v["MaxDD"]*100, 3),
            "VAL_Calmar":   round(s_v["Calmar"], 4),
            "VAL_SR_tstat": round(ci["t_stat"], 3),
            "p_value":      round(ci["p_val"], 6),
            "q_value":      np.nan,
            "significant_5pct":  False,
            "significant_10pct": False,
            "_cfg":   cfg,
            "_s_val": s_v,
            "_s_qqq": s_q,
        }
    except Exception as e:
        print(f"  [sweep] fold={fold_n} combo={kw} ERROR: {e}")
        return None


def run_fold_sweep(
    ohlcv:             Dict,
    cfg_base:          Mahoraga4Config,
    costs:             CostsConfig,
    ic_weights:        Tuple[float, float, float],
    val_start:         str,
    val_end:           str,
    fold_n:            int,
    universe_schedule: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    FIX (#8): Parallel sweep documented as operational optimisation only.
    No guaranteed RAM benefit — loky uses CoW on Linux, full pickle on Windows.
    """
    keys     = list(SWEEP_GRID.keys())
    combos   = list(iproduct(*[SWEEP_GRID[k] for k in keys]))
    base_seed = cfg_base.random_seed

    use_parallel = cfg_base.parallel_sweep and _JOBLIB_AVAILABLE and len(combos) > 4
    if use_parallel:
        print(f"  [sweep] parallel/loky, {len(combos)} combos "
              f"(RAM note: ohlcv pickled per process on non-Linux) …")
        raw = Parallel(n_jobs=-1, backend="loky", verbose=0)(
            joblib_delayed(_evaluate_single_combo)(
                c, keys, cfg_base, ic_weights, ohlcv, costs,
                val_start, val_end, fold_n, base_seed, universe_schedule
            ) for c in combos
        )
    else:
        if cfg_base.parallel_sweep and not _JOBLIB_AVAILABLE:
            print("  [sweep] joblib unavailable — serial mode")
        else:
            print(f"  [sweep] serial, {len(combos)} combos …")
        raw = [
            _evaluate_single_combo(c, keys, cfg_base, ic_weights, ohlcv, costs,
                                   val_start, val_end, fold_n, base_seed, universe_schedule)
            for c in combos
        ]

    rows = [r for r in raw if r is not None]
    if not rows:
        raise RuntimeError(f"[fold {fold_n}] All sweep combos failed.")

    # BHY q-values over the full family
    p_vals = np.array([r["p_value"] for r in rows])
    q_vals, sig5  = bhy_q_values(p_vals, alpha=0.05)
    _, sig10       = bhy_q_values(p_vals, alpha=0.10)
    c_n   = np.sum(1.0 / np.arange(1, len(combos)+1))
    t_min = stats.norm.ppf(1.0 - (0.05/(len(combos)*c_n))/2.0)

    final = []
    for i, row in enumerate(rows):
        row["q_value"]           = round(float(q_vals[i]), 6)
        row["significant_5pct"]  = bool(sig5[i])
        row["significant_10pct"] = bool(sig10[i])
        row["SR_above_HLZ_tmin"] = row["VAL_SR_tstat"] > t_min
        row["stat_label"] = (
            "sig@5%_FDR_support"  if row["significant_5pct"]  else
            "sig@10%_FDR_support" if row["significant_10pct"] else
            "econ_strong_stat_weak"
        )
        # Flatten combo_params for CSV convenience
        for k, v in row["combo_params"].items():
            row[k] = v
        final.append(row)

    df = pd.DataFrame(final).sort_values("score_val", ascending=False)
    df = df.drop(columns=["_cfg", "_s_val", "_s_qqq"], errors="ignore")

    best_row  = final[0]  # already sorted by score
    best_row_ = sorted(final, key=lambda x: -x["score_val"])[0]
    best = {
        "score":       best_row_["score_val"],
        "cfg":         best_row_["_cfg"],
        "s_val":       best_row_["_s_val"],
        "s_qqq":       best_row_["_s_qqq"],
        "combo_params": best_row_["combo_params"],
        "p_value":     best_row_["p_value"],
        "q_value":     best_row_["q_value"],
        "sig_5pct":    best_row_["significant_5pct"],
        "stat_label":  best_row_["stat_label"],
    }

    # Clean _* columns from df now
    for col in ["_cfg", "_s_val", "_s_qqq"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df, best


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — WALK-FORWARD ENGINE  (fixes #1–#5)
# ═══════════════════════════════════════════════════════════════════════════════

def run_walk_forward(
    ohlcv:             Dict,
    cfg_base:          Mahoraga4Config,
    costs:             CostsConfig,
    universe_schedule: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, pd.Series, List[Dict], pd.DataFrame, str, Dict]:
    """
    Walk-forward engine with full OOS purity guarantees.

    Fold boundaries: explicit 5-tuples (train_end, val_start, val_end,
    test_start, test_end). Purity assertions run in build_contiguous_folds().

    universe_schedule: if provided, each fold's backtest consumes the
    point-in-time universe at each rebalance date (fix #2). If None,
    the static universe from cfg_base is used.

    Returns:
      oos_r, oos_eq, fold_results, all_sweeps_df, oos_label, selected_config_info
    """
    trading_idx = pd.DatetimeIndex(ohlcv["close"].index)
    folds       = build_contiguous_folds(cfg_base, trading_idx)

    all_oos_r    = []
    fold_results = []
    all_sweeps   = []
    last_best    = None

    for fold in folds:
        fold_n     = fold["fold"]
        train_start = fold["train_start"]
        train_end   = fold["train_end"]
        val_start   = fold["val_start"]
        val_end     = fold["val_end"]
        test_start  = fold["test_start"]
        test_end    = fold["test_end"]
        is_partial  = fold["is_partial"]

        print(f"\n  ── FOLD {fold_n}/{len(folds)} ──")

        cfg_f    = deepcopy(cfg_base)
        qqq_full = to_s(ohlcv["close"][cfg_base.bench_qqq].ffill())
        dd_thr, vol_thr = calibrate_crisis_thresholds(qqq_full, train_start, train_end, cfg_f)
        cfg_f.crisis_dd_thr         = dd_thr
        cfg_f.crisis_vol_zscore_thr = vol_thr

        print(f"  [fold {fold_n}] Fitting IC weights on train …")
        train_tickers = get_training_universe(
            train_end, universe_schedule,
            cfg_base.universe_static, list(ohlcv["close"].columns)
        )
        close_univ = ohlcv["close"][train_tickers]
        ic_weights = fit_ic_weights(close_univ, qqq_full.loc[train_start:train_end],
                                    cfg_f, train_start, train_end)

        print(f"  [fold {fold_n}] Sweeping val {val_start}→{val_end} …")
        sweep_df, best = run_fold_sweep(ohlcv, cfg_f, costs, ic_weights,
                                        val_start, val_end, fold_n,
                                        universe_schedule=universe_schedule)
        sweep_df["fold"] = fold_n
        all_sweeps.append(sweep_df)
        last_best = best   # FIX (#3): retain last fold's winner

        best_cfg = best["cfg"]
        res_test = backtest(ohlcv, best_cfg, costs, label=f"FOLD{fold_n}_TEST",
                             universe_schedule=universe_schedule)
        r_test   = res_test["returns_net"].loc[test_start:test_end]
        eq_test  = best_cfg.capital_initial * (1.0 + r_test).cumprod()
        exp_test = res_test["exposure"].loc[r_test.index]
        s_test   = summarize(r_test, eq_test, exp_test,
                             res_test["turnover"].loc[r_test.index],
                             best_cfg, f"FOLD{fold_n}_TEST")
        qqq_test     = res_test["bench"]["QQQ_r"].loc[test_start:test_end]
        alpha_nw     = alpha_test_nw(r_test, qqq_test, best_cfg, f"fold{fold_n}_test")

        fold_results.append({
            "fold":            fold_n,
            "train":           f"{train_start}→{train_end}",
            "val":             f"{val_start}→{val_end}",
            "test":            f"{test_start}→{test_end}",
            "is_partial":      is_partial,
            "actual_test_days": fold["actual_test_days"],
            # FIX (#5): explicit combo params — no string transformation
            **{f"best_{k}": v for k, v in best["combo_params"].items()},
            "val_score":       round(best["score"], 4),
            "val_sharpe":      round(best["s_val"]["Sharpe"], 4),
            # FIX (#4): p/q tied to exact selected combo
            "val_p_value":     round(best["p_value"], 6),
            "val_q_value":     round(best["q_value"], 6),
            "val_sig_5pct":    best["sig_5pct"],
            "val_stat_label":  best["stat_label"],
            "test_CAGR%":      round(s_test["CAGR"]*100, 2),
            "test_Sharpe":     round(s_test["Sharpe"], 4),
            "test_MaxDD%":     round(s_test["MaxDD"]*100, 2),
            "test_Calmar":     round(s_test["Calmar"], 4),
            "test_alpha_ann%": round(alpha_nw.get("alpha_ann", np.nan)*100, 2),
            "test_t_alpha":    round(alpha_nw.get("t_alpha", np.nan), 3),
            "AvgExposure":     round(s_test["AvgExposure"]*100, 1),
        })
        all_oos_r.append(r_test)

        partial_str = " [PARTIAL FOLD]" if is_partial else ""
        print(f"  [fold {fold_n}] Sharpe={s_test['Sharpe']:.3f}  "
              f"CAGR={s_test['CAGR']*100:.1f}%  DD={s_test['MaxDD']*100:.1f}%  "
              f"q={best['q_value']:.4f}  {best['stat_label']}{partial_str}")

    # Stitch OOS
    oos_r = pd.concat(all_oos_r).sort_index()
    oos_r = oos_r[~oos_r.index.duplicated()]

    # FIX (#2): validate continuity against real trading calendar
    is_cont, oos_label = validate_oos_continuity(folds, trading_idx)
    print(f"\n  [wf] OOS type: {oos_label}")

    oos_eq = cfg_base.capital_initial * (1.0 + oos_r).cumprod()

    # FIX (#3): final config is the real winning combo from last fold
    # FIX (#4): statistical support tied exactly to that combo
    selected_config_info = {
        "source":          "last_fold_winner",
        "fold":            last_best and len(fold_results),
        "combo_params":    last_best["combo_params"] if last_best else {},
        "val_score":       last_best["score"] if last_best else np.nan,
        "val_p_value":     last_best["p_value"] if last_best else np.nan,
        "val_q_value":     last_best["q_value"] if last_best else np.nan,
        "val_sig_5pct":    last_best["sig_5pct"] if last_best else False,
        "val_stat_label":  last_best["stat_label"] if last_best else "unknown",
        "val_sharpe":      last_best["s_val"]["Sharpe"] if last_best else np.nan,
    }

    return oos_r, oos_eq, fold_results, pd.concat(all_sweeps, ignore_index=True), \
           oos_label, selected_config_info


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — ROBUSTNESS SUITE
# ═══════════════════════════════════════════════════════════════════════════════

def execution_sensitivity_stress(ohlcv, cfg: Mahoraga4Config, base_costs: CostsConfig,
                                  universe_schedule: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Execution_Sensitivity_Stress: evaluates performance degradation under severe
    execution friction. NOT a simulation of realistic overnight price gaps.
    """
    scenarios = {
        "BASE":                  CostsConfig(commission=base_costs.commission, slippage=base_costs.slippage),
        "COST_SENSITIVITY_×2":   CostsConfig(commission=base_costs.commission*2, slippage=base_costs.slippage*2),
        "COST_SENSITIVITY_×3":   CostsConfig(commission=base_costs.commission*3, slippage=base_costs.slippage*3),
        "SLIPPAGE_SENSITIVITY":  CostsConfig(commission=base_costs.commission, slippage=base_costs.slippage, gap_factor=1.5),
        "PANIC_EXECUTION_HEAVY": CostsConfig(commission=base_costs.commission, slippage=base_costs.slippage, gap_factor=2.5),
    }
    rows = []
    for name, c in scenarios.items():
        res = backtest(ohlcv, cfg, c, label=name, universe_schedule=universe_schedule)
        r   = res["returns_net"]; eq = res["equity"]
        s   = summarize(r, eq, res["exposure"], res["turnover"], cfg, name)
        stress_type = ("cost_sensitivity" if "COST" in name else
                       "slippage_sensitivity" if "SLIPPAGE" in name else
                       "panic_execution" if "PANIC" in name else "baseline")
        rows.append({"Scenario": name, "Stress_Type": stress_type,
                     "CAGR%": round(s["CAGR"]*100,2), "Sharpe": round(s["Sharpe"],3),
                     "MaxDD%": round(s["MaxDD"]*100,2), "Calmar": round(s["Calmar"],3),
                     "TurnAnn": round(s["TurnoverAnn"],2), "FinalEq": round(s["FinalEquity"],0)})
        print(f"  [exec_stress] {name:<28} Sharpe={s['Sharpe']:.3f}  DD={s['MaxDD']*100:.1f}%")
    return pd.DataFrame(rows)


def alternate_universe_stress(
    ohlcv_full: Dict,
    cfg_base:   Mahoraga4Config,
    costs:      CostsConfig,
    universes:  Dict[str, Tuple[str, ...]] = ALTERNATE_UNIVERSES,
) -> pd.DataFrame:
    rows = []
    for uname, tickers in universes.items():
        avail = [t for t in tickers if t in ohlcv_full["close"].columns]
        if len(avail) < 4:
            print(f"  [alt_univ] {uname}: only {len(avail)} tickers, skipping"); continue
        c2 = deepcopy(cfg_base); c2.universe_static = tuple(avail)
        print(f"  [alt_univ] {uname}: {len(avail)} tickers …")
        try:
            res = backtest(ohlcv_full, c2, costs, label=uname,
                           universe=list(avail))
            r   = res["returns_net"]; eq = res["equity"]
            s   = summarize(r, eq, res["exposure"], res["turnover"], c2, uname)
            rows.append({"Universe": uname, "N_tickers": len(avail),
                         "CAGR%": round(s["CAGR"]*100,2), "Sharpe": round(s["Sharpe"],3),
                         "MaxDD%": round(s["MaxDD"]*100,2), "Calmar": round(s["Calmar"],3),
                         "FinalEq": round(s["FinalEquity"],0)})
        except Exception as e:
            print(f"  [alt_univ] {uname}: ERROR {e}")
    return pd.DataFrame(rows)


def local_sensitivity(
    ohlcv:        Dict,
    cfg_win:      Mahoraga4Config,
    costs:        CostsConfig,
    universe_schedule: Optional[pd.DataFrame] = None,
    param_a:      str   = "vol_target_ann",
    param_b:      str   = "weight_cap",
    grid_a:       Tuple = (0.20, 0.25, 0.30, 0.35, 0.40),
    grid_b:       Tuple = (0.45, 0.50, 0.55, 0.60, 0.65),
    period_start: str   = "2020-01-01",
    period_end:   str   = "2026-02-20",
) -> pd.DataFrame:
    rows = []
    for va in grid_a:
        for vb in grid_b:
            c2 = deepcopy(cfg_win)
            setattr(c2, param_a, va); setattr(c2, param_b, vb)
            res = backtest(ohlcv, c2, costs, label="sens", universe_schedule=universe_schedule)
            r   = res["returns_net"].loc[period_start:period_end]
            eq  = cfg_win.capital_initial * (1.0 + r).cumprod()
            s   = summarize(r, eq, res["exposure"].loc[r.index], res["turnover"].loc[r.index], c2)
            rows.append({param_a: va, param_b: vb,
                         "Sharpe": round(s["Sharpe"],4), "CAGR%": round(s["CAGR"]*100,2),
                         "MaxDD%": round(s["MaxDD"]*100,2)})
    return pd.DataFrame(rows)


def stop_keep_cash_ablation(ohlcv, cfg_win: Mahoraga4Config, costs: CostsConfig,
                             universe_schedule: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    rows = []
    for mode, skc in [("RESEARCH (keep_cash=True)", True), ("AGGRESSIVE (keep_cash=False)", False)]:
        c2 = deepcopy(cfg_win); c2.stop_keep_cash = skc
        res = backtest(ohlcv, c2, costs, label=mode, universe_schedule=universe_schedule)
        r = res["returns_net"]; eq = res["equity"]
        s = summarize(r, eq, res["exposure"], res["turnover"], c2, mode)
        rows.append({"Mode": mode, "CAGR%": round(s["CAGR"]*100,2),
                     "Sharpe": round(s["Sharpe"],3), "MaxDD%": round(s["MaxDD"]*100,2),
                     "Calmar": round(s["Calmar"],3), "AvgExp%": round(s["AvgExposure"]*100,1)})
        print(f"  [stop_ablation] {mode}: Sharpe={s['Sharpe']:.3f}  DD={s['MaxDD']*100:.1f}%")
    return pd.DataFrame(rows)


def ic_weight_ablation(ohlcv, cfg_win: Mahoraga4Config, costs: CostsConfig,
                        universe_schedule: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    rows = []
    for mode, (wt, wm, wr) in [
        (f"IC_weights ({cfg_win.w_trend:.2f}/{cfg_win.w_mom:.2f}/{cfg_win.w_rel:.2f})",
         (cfg_win.w_trend, cfg_win.w_mom, cfg_win.w_rel)),
        ("EQUAL_WEIGHTS (0.33/0.33/0.33)", (1/3, 1/3, 1/3)),
    ]:
        c2 = deepcopy(cfg_win); c2.w_trend=wt; c2.w_mom=wm; c2.w_rel=wr
        res = backtest(ohlcv, c2, costs, label=mode, universe_schedule=universe_schedule)
        r = res["returns_net"]; eq = res["equity"]
        s = summarize(r, eq, res["exposure"], res["turnover"], c2, mode)
        rows.append({"Mode": mode, "CAGR%": round(s["CAGR"]*100,2),
                     "Sharpe": round(s["Sharpe"],3), "MaxDD%": round(s["MaxDD"]*100,2),
                     "Calmar": round(s["Calmar"],3)})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — PLOTS
# ═══════════════════════════════════════════════════════════════════════════════


def corr_shield_ablation(
    ohlcv: Dict,
    cfg_win: Mahoraga4Config,
    costs: CostsConfig,
    universe_schedule: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Ablation: Corr Shield ON vs OFF + performance inside VIX panic (if VIX available)."""
    rows = []
    vix_available = cfg_win.bench_vix in ohlcv.get("close", pd.DataFrame()).columns
    vix_thr = float(getattr(cfg_win, "corr_vix_thr", 24.0))

    for mode, flag in [("CORR_SHIELD_ON", True), ("CORR_SHIELD_OFF", False)]:
        c2 = deepcopy(cfg_win)
        if hasattr(c2, "enable_corr_shield"):
            c2.enable_corr_shield = bool(flag)

        res = backtest(ohlcv, c2, costs, label=mode, universe_schedule=universe_schedule)
        r = res["returns_net"]
        s = summarize(r, res["equity"], res["exposure"], res["turnover"], c2, mode)

        out = {
            "Mode": mode,
            "CAGR%": round(s["CAGR"] * 100, 3),
            "Vol%": round(s["AnnVol"] * 100, 3),
            "Sharpe": round(s["Sharpe"], 4),
            "MaxDD%": round(s["MaxDD"] * 100, 3),
            "AvgExp%": round(s["AvgExposure"] * 100, 2),
            "TurnAnn": round(s["TurnoverAnn"], 3),
        }

        if vix_available:
            vix = to_s(ohlcv["close"][cfg_win.bench_vix].reindex(r.index).ffill(), "VIX")
            mask = (vix >= vix_thr).reindex(r.index).fillna(False)
            r_p = to_s(r)[mask]
            if len(r_p) >= 25:
                eq_p = c2.capital_initial * (1.0 + r_p).cumprod()
                s_p = summarize(r_p, eq_p,
                                res["exposure"].reindex(r_p.index).fillna(0.0),
                                res["turnover"].reindex(r_p.index).fillna(0.0),
                                c2, mode + "_PANIC")
                out.update({
                    "PanicDays": int(s_p["Days"]),
                    "PanicCAGR%": round(s_p["CAGR"] * 100, 3),
                    "PanicSharpe": round(s_p["Sharpe"], 4),
                    "PanicMaxDD%": round(s_p["MaxDD"] * 100, 3),
                    "PanicAvgExp%": round(s_p["AvgExposure"] * 100, 2),
                })
            else:
                out.update({"PanicDays": int(len(r_p))})

        rows.append(out)

    return pd.DataFrame(rows)




def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] {path}")

def _colors(): return ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]

def plot_equity(curves: Dict[str, pd.Series], title: str, path: str):
    fig, ax = plt.subplots(figsize=(14,6)); c = _colors()
    for i, (k, s) in enumerate(curves.items()):
        ax.plot(to_s(s).dropna().index, to_s(s).dropna().values, label=k,
                linewidth=2.0 if i==0 else 1.2,
                linestyle=["-","--","-.",":",(0,(3,1,1,1))][i%5], color=c[i%6])
    ax.set_yscale("log"); ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Equity (log)"); ax.legend(); ax.grid(alpha=.3)
    _save(fig, path)

def plot_drawdown(curves: Dict[str, pd.Series], title: str, path: str):
    fig, ax = plt.subplots(figsize=(14,5)); c = _colors()
    for i, (k, eq) in enumerate(curves.items()):
        eq = to_s(eq).dropna(); dd = eq/eq.cummax()-1.0
        ax.fill_between(dd.index, dd.values, 0, alpha=.25 if i==0 else .12, color=c[i%6])
        ax.plot(dd.index, dd.values, label=k, linewidth=1.5 if i==0 else .8, color=c[i%6])
    ax.axhline(-0.30, color="red", ls="--", lw=.8, label="-30% threshold")
    ax.set_title(title, fontweight="bold"); ax.legend(); ax.grid(alpha=.3); _save(fig, path)

def plot_wf_oos(oos_eq: pd.Series, qqq_eq: pd.Series,
                fold_results: List[Dict], title: str, path: str,
                oos_label: str = "OOS_continuous"):
    """BUG FIX from 1.1: fold_labels always plain strings (no Timestamps)."""
    fig, axes = plt.subplots(2, 1, figsize=(14,10), sharex=False)
    ax = axes[0]
    ax.plot(to_s(oos_eq).index, to_s(oos_eq).values,
            label=f"OOS ({oos_label})", color="#1f77b4", lw=2)
    ax.plot(to_s(qqq_eq).index, to_s(qqq_eq).values,
            label="QQQ", color="#ff7f0e", lw=1.2, ls="--")
    ax.set_yscale("log"); ax.set_title(title, fontweight="bold")
    ax.legend(); ax.grid(alpha=.3); ax.set_ylabel("Equity (log)")
    for fd in fold_results:
        ts = fd["test"].split("→")[0]
        try: ax.axvline(pd.Timestamp(ts), color="gray", lw=0.5, ls=":")
        except Exception: pass

    ax2 = axes[1]
    fold_sharpes = [float(f["test_Sharpe"]) for f in fold_results]
    fold_labels  = [f"F{int(f['fold'])}" for f in fold_results]   # plain strings
    bar_colors   = ["#2ca02c" if s > 0 else "#d62728" for s in fold_sharpes]
    ax2.bar(fold_labels, fold_sharpes, color=bar_colors)
    ax2.axhline(0, color="black", lw=.5)
    ax2.axhline(1.0, color="green", lw=.8, ls="--", label="Sharpe=1.0")
    for i, fd in enumerate(fold_results):
        if fd.get("is_partial"):
            ax2.text(i, fold_sharpes[i]+0.05, "PARTIAL", ha="center", fontsize=7, color="gray")
    ax2.set_title("Test Sharpe by Fold", fontsize=11)
    ax2.set_ylabel("Sharpe"); ax2.legend(); ax2.grid(alpha=.3, axis="y")
    _save(fig, path)

def plot_risk_overlays(res: Dict, title: str, path: str):
    specs = [
        ("exposure",     "#1f77b4", "Scaled Exposure", (-0.05, 1.10)),
        ("vol_scale",    "#ff7f0e", "Vol-Target Scale", (-0.05, 1.10)),
        ("turb_scale",   "#2ca02c", "Turbulence Scale", (-0.05, 1.10)),
        ("crisis_scale", "#d62728", "Crisis Gate Scale", (-0.05, 1.10)),
        ("corr_scale",   "#9467bd", "Corr Shield Scale", (-0.05, 1.10)),
        ("corr_rho",     "#8c564b", "Avg Pairwise Corr (rho)", (-0.05, 1.05)),
    ]
    fig, axes = plt.subplots(len(specs), 1, figsize=(14, 3.0 * len(specs)), sharex=True)
    if len(specs) == 1:
        axes = [axes]

    for ax, (key, c, lbl, ylim) in zip(axes, specs):
        s = res.get(key, pd.Series(np.nan, index=res.get("equity", pd.Series(dtype=float)).index))
        sv = to_s(s).fillna(method="ffill").fillna(0.0)
        ax.plot(sv.index, sv.values, color=c, lw=1.0)
        ax.fill_between(sv.index, sv.values, alpha=.18, color=c)
        ax.set_ylabel(lbl, fontsize=9)
        ax.set_ylim(*ylim)
        ax.grid(alpha=.3)

    axes[0].set_title(title, fontweight="bold")
    _save(fig, path)

def plot_ic_multi_horizon(ic_df: pd.DataFrame, title: str, path: str):
    fig, ax = plt.subplots(figsize=(14,5))
    ch = {"IC_composite_1d":"#d62728","IC_composite_5d":"#ff7f0e","IC_composite_21d":"#2ca02c"}
    for col in ic_df.columns:
        ax.plot(ic_df.index, ic_df[col], label=col, color=ch.get(col,"gray"),
                linewidth=1.8 if "21d" in col else 1.2)
    ax.axhline(0, color="black", lw=.5)
    ax.axhline(0.04, color="gray", ls="--", lw=.7, label="IC=0.04 (meaningful)")
    ax.set_title(title, fontweight="bold"); ax.legend(); ax.grid(alpha=.3); _save(fig, path)

def plot_sharpe_surface(sens_df: pd.DataFrame, param_a: str, param_b: str, title: str, path: str):
    piv = sens_df.pivot(index=param_a, columns=param_b, values="Sharpe")
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn",
                   vmin=max(0, piv.values.min()-0.1), vmax=piv.values.max()+0.1)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{v:.2f}" for v in piv.columns])
    ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels([f"{v:.2f}" for v in piv.index])
    ax.set_xlabel(param_b); ax.set_ylabel(param_a)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            ax.text(j, i, f"{piv.values[i,j]:.2f}", ha="center", va="center", fontsize=9,
                    color="black" if 0.3 < im.norm(piv.values[i,j]) < 0.7 else "white")
    plt.colorbar(im, ax=ax, label="Sharpe"); ax.set_title(title, fontweight="bold"); _save(fig, path)

def plot_regime_bars(regime_df: pd.DataFrame, title: str, path: str):
    if regime_df.empty: return
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    for ax, m, c in zip(axes, ["CAGR%","Sharpe","MaxDD%"], ["#1f77b4","#ff7f0e","#d62728"]):
        vals = regime_df.set_index("Regime")[m]
        vals.plot(kind="bar", ax=ax, color=c, alpha=0.8)
        ax.set_title(f"{m} by Regime", fontsize=11)
        ax.set_xticklabels(vals.index, rotation=20, ha="right", fontsize=8)
        ax.axhline(0, color="black", lw=.5); ax.grid(alpha=.3, axis="y")
    fig.suptitle(title, fontsize=13, fontweight="bold"); _save(fig, path)

def plot_signal_decomp(decomp: Dict, bench_r: pd.Series, cfg: Mahoraga4Config, title: str, path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    clr = {"TREND_ONLY":"#1f77b4","MOM_ONLY":"#ff7f0e","REL_ONLY":"#2ca02c"}
    shs = {}
    for comp, res in decomp.items():
        eq = to_s(res["equity"]).dropna()
        ax1.plot(eq.index, eq.values, label=comp, color=clr[comp], lw=1.2)
        shs[comp] = sharpe(res["returns_net"], cfg.rf_annual, cfg.trading_days)
    beq = cfg.capital_initial * (1.0 + to_s(bench_r).fillna(0.0)).cumprod()
    ax1.plot(beq.index, beq.values, label="QQQ", color="gray", lw=.8, ls="--")
    ax1.set_yscale("log"); ax1.legend(); ax1.grid(alpha=.3); ax1.set_title("Equity (log)")
    ax2.bar(list(shs.keys()), list(shs.values()), color=[clr[k] for k in shs], alpha=.8)
    ax2.axhline(0, color="black", lw=.5); ax2.set_title("Sharpe by Component"); ax2.grid(alpha=.3, axis="y")
    fig.suptitle(title, fontweight="bold"); _save(fig, path)

def plot_weights_heatmap(weights: pd.DataFrame, title: str, path: str):
    wm = weights.resample("ME").mean()
    fig, ax = plt.subplots(figsize=(16,5))
    im = ax.imshow(wm.T.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=weights.values.max())
    ax.set_yticks(range(len(wm.columns))); ax.set_yticklabels(wm.columns, fontsize=9)
    xt = range(0, len(wm), max(1, len(wm)//20))
    ax.set_xticks(list(xt))
    ax.set_xticklabels([wm.index[i].strftime("%Y-%m") for i in xt], rotation=45, ha="right", fontsize=8)
    plt.colorbar(im, ax=ax, label="Weight"); ax.set_title(title, fontweight="bold"); _save(fig, path)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 16 — FINAL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def final_evaluation(
    ohlcv:             Dict,
    cfg_win:           Mahoraga4Config,
    costs:             CostsConfig,
    ff:                Optional[pd.DataFrame],
    oos_r:             pd.Series,
    oos_eq:            pd.Series,
    oos_exp:           pd.Series,
    oos_to:            pd.Series,
    oos_label:         str = "OOS_continuous",
    universe_schedule: Optional[pd.DataFrame] = None,
) -> Dict:
    res = backtest(ohlcv, cfg_win, costs, label=cfg_win.label,
                   universe_schedule=universe_schedule)
    validate_no_lookahead(res, cfg_win.label)
    qqq_r  = res["bench"]["QQQ_r"]; qqq_eq = res["bench"]["QQQ_eq"]
    eqw    = baseline_eqw(ohlcv, cfg_win, costs, universe_schedule=universe_schedule)
    mom    = baseline_mom(ohlcv, cfg_win, costs, universe_schedule=universe_schedule)

    def _bs(r, eq): return summarize(r, eq, pd.Series(1.0, index=r.index), None, cfg_win, "QQQ")

    s_full     = summarize(res["returns_net"], res["equity"], res["exposure"], res["turnover"], cfg_win, cfg_win.label)
    s_qqq_full = _bs(qqq_r, qqq_eq)
    s_eqw      = summarize(eqw["r"], eqw["eq"], eqw["exp"], eqw["to"], cfg_win, "EQW_Tech")
    s_mom      = summarize(mom["r"], mom["eq"], mom["exp"], mom["to"], cfg_win, "MOM_12_1_TopK")

    # Baselines restricted to the stitched OOS calendar (for apples-to-apples comparison)
    eqw_oos_r  = eqw["r"].reindex(oos_r.index).fillna(0.0)
    eqw_oos_eq = cfg_win.capital_initial * (1.0 + eqw_oos_r).cumprod()
    s_eqw_oos  = summarize(eqw_oos_r, eqw_oos_eq,
                           eqw["exp"].reindex(oos_r.index).fillna(0.0),
                           eqw["to"].reindex(oos_r.index).fillna(0.0),
                           cfg_win, "EQW_Tech_OOS")

    mom_oos_r  = mom["r"].reindex(oos_r.index).fillna(0.0)
    mom_oos_eq = cfg_win.capital_initial * (1.0 + mom_oos_r).cumprod()
    s_mom_oos  = summarize(mom_oos_r, mom_oos_eq,
                           mom["exp"].reindex(oos_r.index).fillna(0.0),
                           mom["to"].reindex(oos_r.index).fillna(0.0),
                           cfg_win, "MOM_12_1_TopK_OOS")
    s_oos      = summarize(oos_r, oos_eq, oos_exp, oos_to, cfg_win, oos_label)
    qqq_oos_r  = qqq_r.reindex(oos_r.index).fillna(0.0)
    s_qqq_oos  = _bs(qqq_oos_r, cfg_win.capital_initial*(1.0+qqq_oos_r).cumprod())

    sr_ci_full = asymptotic_sharpe_ci(res["returns_net"], cfg_win)
    sr_ci_oos  = asymptotic_sharpe_ci(oos_r, cfg_win)
    alpha_full = alpha_test_nw(res["returns_net"], qqq_r, cfg_win, cfg_win.label)
    alpha_oos  = alpha_test_nw(oos_r, qqq_oos_r, cfg_win, oos_label)
    alpha_cond = alpha_test_nw(res["returns_net"], qqq_r, cfg_win,
                               f"{cfg_win.label}_conditional",
                               conditional=True, exposure=res["exposure"])
    ff_full = factor_attribution(res["returns_net"], ff, cfg_win, cfg_win.label)
    ff_oos  = factor_attribution(oos_r, ff, cfg_win, oos_label)
    regime_full = regime_analysis(res["returns_net"], qqq_r, ohlcv, cfg_win)
    regime_oos  = regime_analysis(oos_r, qqq_oos_r, ohlcv, cfg_win)
    stress      = stress_report(res["returns_net"], res["exposure"], STRESS_EPISODES, cfg_win, qqq_r)
    stress_oos  = stress_report(oos_r, oos_exp.reindex(oos_r.index), STRESS_EPISODES, cfg_win, qqq_oos_r)
    boot_full   = moving_block_bootstrap(res["returns_net"], seed=cfg_win.random_seed)
    boot_oos    = moving_block_bootstrap(oos_r, seed=cfg_win.random_seed)

    return {
        "cfg": cfg_win, "res": res, "eqw": eqw, "mom": mom, "oos_label": oos_label,
        "full": s_full, "qqq_full": s_qqq_full, "eqw_full": s_eqw, "mom_full": s_mom,
        "oos": s_oos, "qqq_oos": s_qqq_oos, "eqw_oos": s_eqw_oos, "mom_oos": s_mom_oos,
        "sr_ci_full": sr_ci_full, "sr_ci_oos": sr_ci_oos,
        "alpha_full": alpha_full, "alpha_oos": alpha_oos, "alpha_cond": alpha_cond,
        "ff_full": ff_full, "ff_oos": ff_oos,
        "regime_full": regime_full, "regime_oos": regime_oos,
        "stress": stress, "stress_oos": stress_oos,
        "boot_full": boot_full, "boot_oos": boot_oos,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 17 — PRINT & SAVE  (fix #9: full audit trail)
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(s: Dict) -> Dict:
    return {
        "Model": s["Label"], "FinalEq": f"{s['FinalEquity']:,.0f}",
        "CAGR%": f"{s['CAGR']*100:.2f}", "Vol%": f"{s['AnnVol']*100:.2f}",
        "Sharpe": f"{s['Sharpe']:.3f}", "Sortino": f"{s['Sortino']:.3f}",
        "MaxDD%": f"{s['MaxDD']*100:.2f}", "Calmar": f"{s['Calmar']:.3f}",
        "AvgExp%": f"{s['AvgExposure']*100:.1f}", "TurnAnn": f"{s['TurnoverAnn']:.2f}",
    }


def print_results(out: Dict, fold_results: List[Dict], ff, selected_config_info: Dict):
    sep = "="*110
    print(UNIVERSE_BIAS_DISCLAIMER)
    print(f"\n{sep}\n  MAHORAGA 4 — FULL RESULTS\n{sep}")
    oos_label = out.get("oos_label", "OOS")

    print(f"\n{'─'*70}  FULL PERIOD")
    print(pd.DataFrame([_fmt(out["full"]), _fmt(out["qqq_full"]),
                        _fmt(out["eqw_full"]), _fmt(out["mom_full"])]).to_string(index=False))

    print(f"\n{'─'*70}  OOS — {oos_label}")
    print(pd.DataFrame([_fmt(out["oos"]), _fmt(out["qqq_oos"])]).to_string(index=False))

    print(f"\n{'─'*70}  WALK-FORWARD FOLD SUMMARY")
    fold_df = pd.DataFrame(fold_results).copy()
    for i, row in fold_df.iterrows():
        if row.get("is_partial"):
            fold_df.at[i, "fold"] = f"{row['fold']} (PARTIAL)"
    print(fold_df.to_string(index=False))

    # Selected config summary (fix #4: tied to exact combo)
    print(f"\n{'─'*70}  SELECTED CONFIGURATION (source: {selected_config_info['source']})")
    print(f"  combo:       {selected_config_info['combo_params']}")
    print(f"  val_score:   {selected_config_info['val_score']:.4f}")
    print(f"  val_sharpe:  {selected_config_info.get('val_sharpe', '—'):.4f}")
    print(f"  p_value:     {selected_config_info['val_p_value']:.6f}")
    print(f"  q_value:     {selected_config_info['val_q_value']:.6f}")
    print(f"  sig@5%FDR:   {selected_config_info['val_sig_5pct']}")
    print(f"  stat_label:  {selected_config_info['val_stat_label']}")
    if not selected_config_info.get("val_sig_5pct"):
        print("  [NOTE] Config does NOT pass BHY@5% — economically motivated but "
              "statistical support weak under FDR control. Interpret with caution.")

    print(f"\n{'─'*70}  ASYMPTOTIC SHARPE CI")
    for lbl, ci in [("Full", out["sr_ci_full"]), ("OOS", out["sr_ci_oos"])]:
        print(f"  {lbl:6s}: SR={ci['SR']:.4f}  95%CI=[{ci['CI_lo']:.4f},{ci['CI_hi']:.4f}]  "
              f"t={ci['t_stat']:.3f}  p={ci['p_val']:.6f}")

    print(f"\n{'─'*70}  ALPHA — Newey-West HAC vs QQQ")
    for lbl, a in [("Full", out["alpha_full"]), ("OOS", out["alpha_oos"]), ("Full|exp>0", out["alpha_cond"])]:
        if "error" in a:
            print(f"  {lbl}: ERROR — {a['error']}"); continue
        sig = "***" if a["sig_1pct"] else ("**" if a["sig_5pct"] else "   ")
        cond_str = " [conditional on exposure>0]" if a.get("conditional") else ""
        print(f"  {lbl:18s}: α={a['alpha_ann']*100:.2f}%  t={a['t_alpha']:.3f}  "
              f"p={a['p_alpha']:.6f}  β={a['beta']:.4f}  R²={a['R2']:.4f}  "
              f"n={a.get('n_obs','—')}  {sig}{cond_str}")

    if out.get("ff_full") and "error" not in (out["ff_full"] or {}):
        print(f"\n{'─'*70}  FF5+UMD ATTRIBUTION")
        for lbl, fa in [("Full", out["ff_full"]), ("OOS", out["ff_oos"])]:
            if fa and "error" not in fa:
                print(f"  {lbl}: α={fa['alpha_ann']*100:.2f}%  t={fa['t_alpha']:.3f}  "
                      f"β_mkt={fa['beta_mkt']:.3f}  β_umd={fa['beta_umd']:.3f}  R²_adj={fa['R2_adj']:.3f}")

    print(f"\n{'─'*70}  REGIME ANALYSIS")
    print("  Full period:"); print(out["regime_full"].to_string(index=False))
    print("  OOS:"); print(out["regime_oos"].to_string(index=False))

    print(f"\n{'─'*70}  STRESS EPISODES")
    print(out["stress"].to_string(index=False))

    print(f"\n{'─'*70}  BOOTSTRAP DD (moving block, 1000 samples)")
    for lbl, b in [("Full", out["boot_full"]), ("OOS", out["boot_oos"])]:
        print(f"  {lbl}: median_DD={b['dd_p50']*100:.1f}%  "
              f"p5_worst={b['dd_p5_worst']*100:.1f}%  "
              f"P(DD<-30%)={b['ruin_prob_30dd']:.1f}%  P(DD<-50%)={b['ruin_prob_50dd']:.1f}%")

    print(EXECUTION_STRESS_DISCLAIMER)
    print(UNIVERSE_BIAS_DISCLAIMER)


def _build_final_report_text(out: Dict, fold_results: List[Dict],
                              selected_config_info: Dict, oos_label: str) -> str:
    lines = [
        "MAHORAGA 4 — FINAL REPORT",
        "="*80,
        UNIVERSE_BIAS_DISCLAIMER,
        "",
        f"OOS type: {oos_label}",
        "",
        "FULL PERIOD SUMMARY",
        "-"*40,
    ]
    for k, v in _fmt(out["full"]).items():
        lines.append(f"  {k}: {v}")
    lines += ["", "OOS SUMMARY", "-"*40]
    for k, v in _fmt(out["oos"]).items():
        lines.append(f"  {k}: {v}")
    lines += ["", "SELECTED CONFIGURATION", "-"*40,
              f"  Source: {selected_config_info['source']}",
              f"  combo_params: {selected_config_info['combo_params']}",
              f"  val_score:    {selected_config_info['val_score']:.4f}",
              f"  p_value:      {selected_config_info['val_p_value']:.6f}",
              f"  q_value:      {selected_config_info['val_q_value']:.6f}",
              f"  sig@5%FDR:    {selected_config_info['val_sig_5pct']}",
              f"  stat_label:   {selected_config_info['val_stat_label']}",
              ]
    if not selected_config_info.get("val_sig_5pct"):
        lines.append("  [NOTE] Does not pass BHY@5% — interpret with caution.")
    lines += ["", EXECUTION_STRESS_DISCLAIMER, "", UNIVERSE_BIAS_DISCLAIMER]
    return "\n".join(lines)


def save_outputs(
    out:                 Dict,
    fold_results:        List[Dict],
    ic_df:               pd.DataFrame,
    rob:                 Dict,
    all_sweeps:          pd.DataFrame,
    selected_config_info: Dict,
    oos_label:           str,
    cfg:                 Mahoraga4Config,
    universe_schedule:   Optional[pd.DataFrame] = None,
    universe_snapshots:  Optional[List[pd.DataFrame]] = None,
):
    """
    FIX (#9): Complete audit trail.
    Saves:
      - comparison_full.csv, comparison_oos.csv
      - walk_forward_folds.csv
      - walk_forward_meta.csv          ← NEW
      - selected_config_support.csv   ← NEW
      - final_report.txt              ← NEW
      - walk_forward_sweeps.csv (includes q_value)
      - universe_snapshots/ (per recon date) ← NEW
      - universe_methodology.json     ← NEW
      - stress, regime, alpha, ff, sharpe_ci CSVs
    """
    d = cfg.outputs_dir; _ensure_dir(d)

    def _df(rows):
        return pd.DataFrame([{k: round(v,6) if isinstance(v,float) else v
                               for k,v in r.items()} for r in rows])

    _df([out["full"],out["qqq_full"],out["eqw_full"],out["mom_full"]]).to_csv(
        f"{d}/comparison_full.csv", index=False)
    _df([out["oos"],out["qqq_oos"]]).to_csv(f"{d}/comparison_oos.csv", index=False)
    pd.DataFrame(fold_results).to_csv(f"{d}/walk_forward_folds.csv", index=False)

    # walk_forward_meta.csv
    meta = {
        "oos_label":          oos_label,
        "n_folds":            len(fold_results),
        "oos_start":          fold_results[0]["test"].split("→")[0] if fold_results else "—",
        "oos_end":            fold_results[-1]["test"].split("→")[1] if fold_results else "—",
        "any_partial_fold":   any(f.get("is_partial") for f in fold_results),
        "total_oos_days":     sum(f.get("actual_test_days",0) for f in fold_results),
        "sweep_grid_combos":  len(list(iproduct(*[SWEEP_GRID[k] for k in SWEEP_GRID]))),
        "parallel_sweep":     cfg.parallel_sweep,
        "universe_mode":      "canonical" if cfg.use_canonical_universe else "static_expost",
    }
    pd.DataFrame([meta]).to_csv(f"{d}/walk_forward_meta.csv", index=False)

    # selected_config_support.csv (fix #4)
    support = {
        **selected_config_info.get("combo_params", {}),
        "source":         selected_config_info["source"],
        "val_score":      selected_config_info["val_score"],
        "val_p_value":    selected_config_info["val_p_value"],
        "val_q_value":    selected_config_info["val_q_value"],
        "val_sig_5pct":   selected_config_info["val_sig_5pct"],
        "val_stat_label": selected_config_info["val_stat_label"],
        "val_sharpe":     selected_config_info.get("val_sharpe", np.nan),
    }
    pd.DataFrame([support]).to_csv(f"{d}/selected_config_support.csv", index=False)

    # final_report.txt
    report_text = _build_final_report_text(out, fold_results, selected_config_info, oos_label)
    with open(f"{d}/final_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    # Sweep CSV (includes p_value, q_value, significant_5pct, significant_10pct)
    all_sweeps.to_csv(f"{d}/walk_forward_sweeps.csv", index=False)

    out["stress"].to_csv(f"{d}/stress_full.csv", index=False)
    out["regime_full"].to_csv(f"{d}/regime_full.csv", index=False)
    out["regime_oos"].to_csv(f"{d}/regime_oos.csv", index=False)
    ic_df.to_csv(f"{d}/rolling_ic_multi.csv")
    if rob.get("exec_stress") is not None:
        rob["exec_stress"].to_csv(f"{d}/execution_sensitivity_stress.csv", index=False)
    if rob.get("alt_univ") is not None:
        rob["alt_univ"].to_csv(f"{d}/alt_universes.csv", index=False)
    if rob.get("sensitivity") is not None:
        rob["sensitivity"].to_csv(f"{d}/local_sensitivity.csv", index=False)
    if rob.get("stop_ablation") is not None:
        rob["stop_ablation"].to_csv(f"{d}/stop_ablation.csv", index=False)
    if rob.get("ic_ablation") is not None:
        rob["ic_ablation"].to_csv(f"{d}/ic_ablation.csv", index=False)
    alpha_rows = [out["alpha_full"], out["alpha_oos"], out["alpha_cond"]]
    pd.DataFrame(alpha_rows).to_csv(f"{d}/alpha_nw.csv", index=False)
    pd.DataFrame([out["sr_ci_full"], out["sr_ci_oos"]]).to_csv(f"{d}/sharpe_ci.csv", index=False)
    if out.get("ff_full"):
        pd.DataFrame([out["ff_full"], out.get("ff_oos", {})]).to_csv(f"{d}/ff_attribution.csv", index=False)

    # Universe snapshots (fix #9)
    if universe_schedule is not None:
        universe_schedule.to_csv(f"{d}/universe_schedule.csv", index=False)
    if universe_snapshots:
        snap_dir = os.path.join(d, "universe_snapshots")
        _ensure_dir(snap_dir)
        for snap in universe_snapshots:
            if snap.empty: continue
            rd = str(snap["recon_date"].iloc[0]).replace("-","")
            snap.to_csv(f"{snap_dir}/snapshot_{rd}.csv", index=False)

    # universe_methodology.json (fix #9)
    universe_cfg = UniverseConfig()
    methodology = {
        "version":            "1.3",
        "universe_mode":      "canonical_engine" if cfg.use_canonical_universe else "static_expost",
        "static_universe":    list(cfg.universe_static),
        "universe_note":      "static universe is ex-post selected mega-cap tech; not survivor-bias-free",
        "canonical_ranking":  "LiqSizeProxy = mean(price × volume, 30d) — NOT float-adjusted market cap (FFMC)",
        "canonical_ranking_note": "FFMC/GICS-IT require CRSP/Compustat PIT data, which is not integrated",
        "canonical_recon":    universe_cfg.recon_freq,
        "canonical_target_n": universe_cfg.target_size,
        "auto_entry_rank":    universe_cfg.auto_entry_rank,
        "retention_rank":     universe_cfg.retention_rank,
        "buffer_rank":        universe_cfg.buffer_rank,
        "min_seasoning_days": universe_cfg.min_seasoning_days,
        "min_free_float_proxy": "volume_continuity (fraction of days with non-zero vol) — NOT actual float data",
        "min_addv_usd":       universe_cfg.min_addv_usd,
        "data_source":        "yfinance (simulation proxy — not CRSP/Compustat PIT)",
        "survivorship_bias":  "PRESENT — cannot be eliminated without CRSP PIT",
        "disclaimer":         UNIVERSE_BIAS_DISCLAIMER.strip(),
    }
    with open(f"{d}/universe_methodology.json", "w", encoding="utf-8") as f:
        json.dump(methodology, f, indent=2)

    print(f"\n  [outputs → ./{d}/]")
    print(f"    walk_forward_meta.csv, selected_config_support.csv, final_report.txt")
    print(f"    universe_methodology.json, universe_snapshots/ (if canonical engine used)")
    print(f"    walk_forward_sweeps.csv (columns: p_value, q_value, significant_5pct, significant_10pct)")


def make_plots(out, oos_r, oos_eq, fold_results, ic_df, decomp, rob, cfg,
               oos_label: str = "OOS_continuous"):
    p = cfg.plots_dir; _ensure_dir(p)
    res = out["res"]; eqw = out["eqw"]; mom = out["mom"]
    plot_equity({cfg.label: res["equity"], "QQQ": res["bench"]["QQQ_eq"],
                 "EQW": eqw["eq"], "MOM_12_1": mom["eq"]},
                "Full Period Equity — Mahoraga 4", f"{p}/01_equity_full.png")
    plot_equity({cfg.label: oos_eq, "QQQ (OOS)": res["bench"]["QQQ_eq"].reindex(oos_r.index)},
                f"Walk-Forward {oos_label} — Mahoraga 4", f"{p}/02_equity_oos.png")
    plot_drawdown({cfg.label: res["equity"], "QQQ": res["bench"]["QQQ_eq"], "MOM_12_1": mom["eq"]},
                  "Drawdown", f"{p}/03_drawdown.png")
    plot_wf_oos(oos_eq, res["bench"]["QQQ_eq"].reindex(oos_r.index),
                fold_results, f"Walk-Forward OOS — {oos_label}", f"{p}/04_walkforward.png", oos_label)
    plot_risk_overlays(res, "Risk Overlays", f"{p}/05_risk_overlays.png")
    plot_weights_heatmap(res["weights_scaled"], "Portfolio Weights (monthly avg)", f"{p}/06_weights.png")
    plot_ic_multi_horizon(ic_df, "Rolling IC — 1d/5d/21d", f"{p}/07_ic_multi.png")
    plot_regime_bars(out["regime_full"], "Regime Analysis — Full", f"{p}/08_regime_full.png")
    plot_regime_bars(out["regime_oos"],  f"Regime Analysis — {oos_label}", f"{p}/09_regime_oos.png")
    if decomp:
        plot_signal_decomp(decomp, res["bench"]["QQQ_r"], cfg, "Signal Decomposition", f"{p}/10_decomp.png")
    if rob.get("sensitivity") is not None:
        plot_sharpe_surface(rob["sensitivity"], "vol_target_ann", "weight_cap",
                            "Sharpe Surface — vol_target × weight_cap", f"{p}/11_sensitivity.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 18 — MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_mahoraga4(make_plots_flag: bool = True, run_robustness: bool = True) -> Dict:
    print("="*80)
    print("  MAHORAGA 4 — Research Edition")
    print("="*80)
    print(UNIVERSE_BIAS_DISCLAIMER)

    cfg   = Mahoraga4Config()
    costs = CostsConfig()
    ucfg  = UniverseConfig()
    _ensure_dir(cfg.cache_dir)
    _ensure_dir(cfg.plots_dir)
    _ensure_dir(cfg.outputs_dir)

    # ── [1] Data ──────────────────────────────────────────────────────────────
    print("\n[1] Downloading data …")

    # Fix #3: equity candidates and benchmarks are strictly separated.
    # benchmarks/indices are NEVER passed to tickers_in_scope in the universe engine.
    equity_tickers = sorted(set(
        list(cfg.universe_static)
        + [t for u in ALTERNATE_UNIVERSES.values() for t in u]
    ))
    bench_tickers = [cfg.bench_qqq, cfg.bench_spy, cfg.bench_vix]
    all_tickers   = sorted(set(equity_tickers + bench_tickers))
    ohlcv = download_ohlcv(all_tickers, cfg.data_start, cfg.data_end, cfg.cache_dir)

    print("\n[1b] Fama-French factors …")
    ff = load_ff_factors(cfg.cache_dir)

    # ── [1c] Universe engine — governs simulation point-in-time (fix #2, #3) ──
    universe_schedule  = None
    universe_snapshots = None
    if cfg.use_canonical_universe:
        print("\n[1c] Canonical universe engine (quarterly LiqSizeProxy reconstitution) …")
        # Fix #3: only equity tickers in scope — no QQQ/SPY/^VIX
        valid_equity = [t for t in equity_tickers if t in ohlcv["close"].columns]
        print(f"  [universe] tickers_in_scope = {len(valid_equity)} equity candidates (benchmarks excluded)")
        universe_schedule, universe_snapshots = build_canonical_universe_schedule(
            ohlcv["close"], ohlcv["volume"], ucfg, valid_equity,
            cfg.data_start, cfg.data_end
        )
        n_recon = len(universe_schedule)
        print(f"  [universe] {n_recon} reconstitution dates built")
        if n_recon > 0:
            first_u = json.loads(universe_schedule.iloc[0]["members"])
            last_u  = json.loads(universe_schedule.iloc[-1]["members"])
            print(f"  [universe] first recon ({universe_schedule.iloc[0]['recon_date']}): {first_u}")
            print(f"  [universe] last  recon ({universe_schedule.iloc[-1]['recon_date']}): {last_u}")
        # Fix #2: schedule is passed to walk-forward and backtest directly.
        # It is NOT collapsed to latest_members — it governs the simulation PIT.
        print("  [universe] Schedule governs backtest point-in-time at each rebalance.")
    else:
        print("\n[1c] *** WARNING: use_canonical_universe=False ***")
        print("     Running on static ex-post universe. Survivorship bias is PRESENT.")
        print("     Results with this setting cannot be presented as bias-corrected.")
        print(f"     Static universe: {list(cfg.universe_static)}")
        print("     To use the canonical LiqSizeProxy engine, set use_canonical_universe=True.")

    # ── [2] Walk-forward (explicit boundaries, OOS purity asserted) ───────────
    print("\n[2] Walk-forward …")
    oos_r, oos_eq, fold_results, all_sweeps, oos_label, selected_config_info = \
        run_walk_forward(ohlcv, cfg, costs, universe_schedule=universe_schedule)


    oos_summary = summarize(oos_r, oos_eq, None, None, cfg, oos_label)
    print(f"\n  OOS type:   {oos_label}")
    print(f"  OOS Sharpe={oos_summary['Sharpe']:.3f}  "
          f"CAGR={oos_summary['CAGR']*100:.1f}%  MaxDD={oos_summary['MaxDD']*100:.1f}%")

    # ── [3] Final config (FIX #3): real last-fold winner ─────────────────────
    print("\n[3] Applying last-fold winning combo as final config …")
    last_train_end = cfg.wf_folds[-1][0]
    cfg_final      = deepcopy(cfg)
    qqq_full       = to_s(ohlcv["close"][cfg.bench_qqq].ffill())

    dd_thr, vol_thr = calibrate_crisis_thresholds(qqq_full, cfg.wf_train_start, last_train_end, cfg_final)
    cfg_final.crisis_dd_thr         = dd_thr
    cfg_final.crisis_vol_zscore_thr = vol_thr

    # FIX (#5): apply combo params directly — no string key transformation
    for k, v in selected_config_info["combo_params"].items():
        setattr(cfg_final, k, v)

    # Re-fit IC on full final train — use PIT-aware ticker set
    final_train_tickers = get_training_universe(
        last_train_end, universe_schedule,
        cfg.universe_static, list(ohlcv["close"].columns)
    )
    close_univ = ohlcv["close"][final_train_tickers]
    wt, wm, wr = fit_ic_weights(close_univ, qqq_full.loc[cfg.wf_train_start:last_train_end],
                                 cfg_final, cfg.wf_train_start, last_train_end)
    cfg_final.w_trend = wt; cfg_final.w_mom = wm; cfg_final.w_rel = wr

    # FIX (#4): print exact statistical support for selected combo
    print(f"\n  Final config: {selected_config_info['combo_params']}")
    print(f"  Statistical support (last fold):")
    print(f"    p_value    = {selected_config_info['val_p_value']:.6f}")
    print(f"    q_value    = {selected_config_info['val_q_value']:.6f}")
    print(f"    sig@5%FDR  = {selected_config_info['val_sig_5pct']}")
    print(f"    stat_label = {selected_config_info['val_stat_label']}")
    if not selected_config_info["val_sig_5pct"]:
        print("  [NOTE] Config economically strong but not statistically supported "
              "at BHY@5% FDR threshold. A low q_value provides FDR support — "
              "it does NOT imply certainty or absence of data-mining bias.")

    # ── [4] Full evaluation ───────────────────────────────────────────────────
    print("\n[4] Full evaluation …")
    out = final_evaluation(ohlcv, cfg_final, costs, ff, oos_r, oos_eq, oos_label,
                           universe_schedule=universe_schedule)

    # ── [5] Rolling IC ────────────────────────────────────────────────────────
    print("\n[5] Rolling IC (1d/5d/21d) …")
    ic_df = rolling_ic_multi_horizon(close_univ, qqq_full, cfg_final, window=63)

    # ── [6] Signal decomposition ──────────────────────────────────────────────
    print("\n[6] Signal decomposition …")
    decomp = baseline_signal_decomp(ohlcv, cfg_final, costs,
                                    universe_schedule=universe_schedule)

    # ── [7] Robustness ────────────────────────────────────────────────────────
    rob = {}
    if run_robustness:
        print("\n[7a] Execution_Sensitivity_Stress …")
        print(EXECUTION_STRESS_DISCLAIMER)
        rob["exec_stress"] = execution_sensitivity_stress(ohlcv, cfg_final, costs,
                                                          universe_schedule=universe_schedule)

        print("\n[7b] Alternate universes …")
        rob["alt_univ"] = alternate_universe_stress(ohlcv, cfg_final, costs)

        print("\n[7c] Local parameter sensitivity …")
        rob["sensitivity"] = local_sensitivity(ohlcv, cfg_final, costs,
                                              universe_schedule=universe_schedule)

        print("\n[7d] Stop ablation …")
        rob["stop_ablation"] = stop_keep_cash_ablation(ohlcv, cfg_final, costs,
                                                      universe_schedule=universe_schedule)

        print("\n[7e] IC weight ablation …")
        rob["ic_ablation"] = ic_weight_ablation(ohlcv, cfg_final, costs,
                                              universe_schedule=universe_schedule)

        print("\n  EXECUTION_SENSITIVITY_STRESS:")
        print(rob["exec_stress"].to_string(index=False))
        print("\n  ALTERNATE UNIVERSES:")
        print(rob["alt_univ"].to_string(index=False))
        print("\n  STOP ABLATION:")
        print(rob["stop_ablation"].to_string(index=False))

    # ── [8] Print, save, plot ─────────────────────────────────────────────────
    print_results(out, fold_results, ff, selected_config_info)
    save_outputs(out, fold_results, ic_df, rob, all_sweeps, selected_config_info,
                 oos_label, cfg_final, universe_schedule, universe_snapshots)

    if make_plots_flag:
        print("\n[8] Generating plots …")
        make_plots(out, oos_r, oos_eq, fold_results, ic_df, decomp, rob, cfg_final, oos_label)

    return {
        "cfg":                  cfg_final,
        "out":                  out,
        "oos_r":                oos_r,
        "oos_eq":               oos_eq,
        "oos_label":            oos_label,
        "fold_results":         fold_results,
        "ic_df":                ic_df,
        "decomp":               decomp,
        "rob":                  rob,
        "selected_config_info": selected_config_info,
        "universe_schedule":    universe_schedule,
    }




# ═══════════════════════════════════════════════════════════════════════════════
# MAHORAGA 5 EXTENSION LAYER
#   - Data hygiene & universe hardening
#   - Fold diagnostics
#   - ML regime gate (research module)
#   - Plot/output/reporting upgrades
# ═══════════════════════════════════════════════════════════════════════════════

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss


KNOWN_TICKER_START_OVERRIDES: Dict[str, str] = {
    # Manual override to prevent obvious ticker-history contamination under free data.
    # Extend cautiously when you have verified listing / ticker-reuse details.
    "ARM": "2023-09-14",
}


@dataclass
class Mahoraga5Config(Mahoraga4Config):
    plots_dir:      str  = "mahoraga5_plots"
    outputs_dir:    str  = "mahoraga5_outputs"
    label:          str  = "MAHORAGA_5"

    # Data hygiene / universe hardening
    enable_data_quality_layer: bool = True
    asset_registry_path: Optional[str] = None
    dq_min_history_days: int = 252
    dq_max_missing_close_pct: float = 0.10
    dq_max_zero_volume_pct: float = 0.35
    dq_max_stale_price_streak: int = 10
    min_universe_names: int = 6

    # ML regime gate (research module; does not replace core alpha engine)
    enable_ml_regime_gate: bool = True
    ml_horizon_days: int = 5
    ml_min_train_samples: int = 80
    ml_policy_threshold_grid: Tuple[float, ...] = (0.45, 0.50, 0.55, 0.60)
    ml_policy_low_scale_grid: Tuple[float, ...] = (0.25, 0.50, 0.75)
    ml_model_choices: Tuple[str, ...] = ("logit", "gb")


def _longest_true_streak(mask: pd.Series) -> int:
    arr = np.asarray(mask.fillna(False).astype(bool).values, dtype=bool)
    best = cur = 0
    for x in arr:
        if x:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def build_asset_registry(
    tickers: List[str],
    cfg: Mahoraga5Config,
    bench_tickers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Lightweight asset registry for free-data environments.
    This does NOT replace CRSP/Compustat PIT. It exists to:
      - separate equity candidates from benchmarks/indices,
      - enforce manual start-date overrides for obviously contaminated symbols,
      - provide an audit trail for candidate eligibility.
    """
    bench_set = set(bench_tickers or [cfg.bench_qqq, cfg.bench_spy, cfg.bench_vix])
    rows = []
    for t in sorted(set(tickers)):
        live_from = KNOWN_TICKER_START_OVERRIDES.get(t)
        rows.append({
            "ticker": t,
            "asset_type": "benchmark" if t in bench_set or t.startswith("^") else "equity",
            "allowed_in_universe": (t not in bench_set and not t.startswith("^")),
            "live_from_override": live_from,
            "notes": "manual_override" if live_from else "auto",
        })

    reg = pd.DataFrame(rows)
    path = cfg.asset_registry_path
    if path and os.path.exists(path):
        try:
            user_reg = pd.read_csv(path)
            req = {"ticker"}
            if req.issubset(user_reg.columns):
                reg = reg.drop(columns=[c for c in reg.columns if c in user_reg.columns and c != "ticker"]).merge(
                    user_reg, on="ticker", how="left", suffixes=("", "_user")
                )
                reg["allowed_in_universe"] = reg["allowed_in_universe"].fillna(True)
        except Exception as e:
            print(f"  [registry] WARNING: failed to load asset_registry_path ({e}) — using default registry")
    return reg.sort_values("ticker").reset_index(drop=True)


def compute_data_quality_report(
    ohlcv: Dict[str, pd.DataFrame],
    tickers: List[str],
    cfg: Mahoraga5Config,
) -> pd.DataFrame:
    close = ohlcv["close"]
    volume = ohlcv["volume"]
    rows = []
    for t in sorted(set(tickers)):
        if t not in close.columns:
            continue
        p = close[t]
        v = volume[t] if t in volume.columns else pd.Series(index=close.index, dtype=float)
        first_valid = p.first_valid_index()
        if first_valid is None:
            rows.append({
                "ticker": t,
                "first_valid_date": None,
                "history_days": 0,
                "missing_close_pct": 1.0,
                "zero_volume_pct": 1.0,
                "stale_price_streak": np.nan,
                "eligible_data_quality": False,
                "dq_reason": "no_price_history",
            })
            continue
        p_live = p.loc[first_valid:]
        v_live = v.reindex(p_live.index)
        history_days = int(p_live.notna().sum())
        missing_close_pct = float(p_live.isna().mean())
        zero_volume_pct = float((v_live.fillna(0.0) <= 0).mean())
        stale_mask = p_live.ffill().diff().abs().fillna(1.0) < 1e-12
        stale_streak = _longest_true_streak(stale_mask)

        reasons = []
        if history_days < cfg.dq_min_history_days:
            reasons.append("short_history")
        if missing_close_pct > cfg.dq_max_missing_close_pct:
            reasons.append("missing_close")
        if zero_volume_pct > cfg.dq_max_zero_volume_pct:
            reasons.append("zero_volume")
        if stale_streak > cfg.dq_max_stale_price_streak:
            reasons.append("stale_price")
        eligible = len(reasons) == 0

        rows.append({
            "ticker": t,
            "first_valid_date": str(pd.Timestamp(first_valid).date()),
            "history_days": history_days,
            "missing_close_pct": round(missing_close_pct, 6),
            "zero_volume_pct": round(zero_volume_pct, 6),
            "stale_price_streak": stale_streak,
            "eligible_data_quality": bool(eligible),
            "dq_reason": "ok" if eligible else ";".join(reasons),
        })
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def filter_equity_candidates(
    tickers: List[str],
    registry_df: pd.DataFrame,
    dq_df: pd.DataFrame,
    cfg: Mahoraga5Config,
) -> List[str]:
    reg = registry_df.set_index("ticker") if registry_df is not None and not registry_df.empty else pd.DataFrame()
    dq  = dq_df.set_index("ticker") if dq_df is not None and not dq_df.empty else pd.DataFrame()
    clean = []
    for t in tickers:
        allowed = True
        if not reg.empty and t in reg.index:
            allowed = bool(reg.at[t, "allowed_in_universe"])
        dq_ok = True
        if cfg.enable_data_quality_layer and not dq.empty and t in dq.index:
            dq_ok = bool(dq.at[t, "eligible_data_quality"])
        if allowed and dq_ok:
            clean.append(t)
    return sorted(set(clean))


def build_universe_diagnostics(schedule_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame()
    rows = []
    prev = set()
    for _, row in schedule_df.sort_values("recon_date").iterrows():
        members = set(json.loads(row["members"]))
        entered = sorted(members - prev)
        exited = sorted(prev - members)
        turnover = (len(entered) + len(exited)) / max(len(members | prev), 1)
        rows.append({
            "recon_date": row["recon_date"],
            "n_members": len(members),
            "n_entered": len(entered),
            "n_exited": len(exited),
            "turnover_ratio": round(float(turnover), 6),
            "entered": json.dumps(entered),
            "exited": json.dumps(exited),
        })
        prev = members
    return pd.DataFrame(rows)


def _build_universe_snapshot(
    date:             pd.Timestamp,
    close:            pd.DataFrame,
    volume:           pd.DataFrame,
    universe_cfg:     UniverseConfig,
    prior_members:    Optional[List[str]],
    first_date:       Optional[pd.Timestamp],
    tickers_in_scope: List[str],
    registry_df:      Optional[pd.DataFrame] = None,
    quality_df:       Optional[pd.DataFrame] = None,
) -> Tuple[List[str], pd.DataFrame]:
    idx  = close.index
    past = idx[idx <= date]
    if len(past) < universe_cfg.min_seasoning_days + 20:
        return list(tickers_in_scope)[:universe_cfg.target_size], pd.DataFrame()

    reg_map = registry_df.set_index("ticker") if registry_df is not None and not registry_df.empty else pd.DataFrame()
    dq_map  = quality_df.set_index("ticker") if quality_df is not None and not quality_df.empty else pd.DataFrame()

    rows = []
    for t in tickers_in_scope:
        if t not in close.columns:
            continue
        if not reg_map.empty and t in reg_map.index:
            if not bool(reg_map.at[t, "allowed_in_universe"]):
                rows.append({"ticker": t, "eligible": False, "reason": "registry_excluded",
                             "liq_size_proxy": 0.0, "addv_proxy": 0.0})
                continue
            live_from = reg_map.at[t, "live_from_override"]
            if pd.notna(live_from) and str(live_from) not in ("", "None"):
                if pd.Timestamp(date) < pd.Timestamp(live_from):
                    rows.append({"ticker": t, "eligible": False, "reason": f"not_live_yet<{live_from}>",
                                 "liq_size_proxy": 0.0, "addv_proxy": 0.0})
                    continue
        if not dq_map.empty and t in dq_map.index and not bool(dq_map.at[t, "eligible_data_quality"]):
            rows.append({"ticker": t, "eligible": False,
                         "reason": f"dq_fail:{dq_map.at[t, 'dq_reason']}",
                         "liq_size_proxy": 0.0, "addv_proxy": 0.0})
            continue

        p_series = close[t].reindex(past).dropna()
        v_series = (volume[t].reindex(past).dropna()
                    if t in volume.columns else pd.Series(dtype=float))

        if len(p_series) < universe_cfg.min_seasoning_days:
            rows.append({"ticker": t, "eligible": False, "reason": "seasoning_fail",
                         "liq_size_proxy": 0.0, "addv_proxy": 0.0})
            continue

        if len(v_series) >= 20:
            last63_p = p_series.iloc[-63:] if len(p_series) >= 63 else p_series
            last63_v = v_series.reindex(last63_p.index).fillna(0)
            addv = float((last63_p * last63_v).mean())
        else:
            addv = 0.0
        if addv < universe_cfg.min_addv_usd:
            rows.append({"ticker": t, "eligible": False, "reason": "addv_fail",
                         "liq_size_proxy": 0.0, "addv_proxy": addv})
            continue

        if len(v_series) >= 20:
            last63_v2 = v_series.iloc[-63:] if len(v_series) >= 63 else v_series
            vol_coverage = float((last63_v2 > 0).mean())
        else:
            vol_coverage = 0.0
        if vol_coverage < universe_cfg.min_free_float:
            rows.append({"ticker": t, "eligible": False,
                         "reason": f"free_float_proxy_fail (coverage={vol_coverage:.2f})",
                         "liq_size_proxy": 0.0, "addv_proxy": addv})
            continue

        last30_p = p_series.iloc[-30:] if len(p_series) >= 30 else p_series
        last30_v = (v_series.reindex(last30_p.index).fillna(0)
                    if len(v_series) >= 10 else pd.Series([1.0] * len(last30_p)))
        liq_size = float((last30_p * last30_v).mean())
        rows.append({
            "ticker": t,
            "eligible": True,
            "reason": "eligible",
            "liq_size_proxy": liq_size,
            "addv_proxy": addv,
            "vol_coverage": vol_coverage,
        })

    if not rows:
        return [], pd.DataFrame()

    snap = pd.DataFrame(rows).sort_values("liq_size_proxy", ascending=False)
    eligible = snap[snap["eligible"]].reset_index(drop=True)
    if eligible.empty:
        return [], snap
    eligible["rank"] = eligible.index + 1

    prior    = set(prior_members) if prior_members else set()
    ucfg     = universe_cfg
    selected = []

    top_auto = eligible[eligible["rank"] <= ucfg.auto_entry_rank]["ticker"].tolist()
    selected.extend(top_auto)
    for t in eligible[(eligible["rank"] > ucfg.auto_entry_rank) &
                      (eligible["rank"] <= ucfg.retention_rank)]["ticker"].tolist():
        if t in prior and t not in selected:
            selected.append(t)
            eligible.loc[eligible["ticker"] == t, "reason"] = "retained_incumbent_rank10"
    if len(selected) < ucfg.target_size:
        for t in eligible[(eligible["rank"] > ucfg.retention_rank) &
                          (eligible["rank"] <= ucfg.buffer_rank)]["ticker"].tolist():
            if t in prior and t not in selected and len(selected) < ucfg.target_size:
                selected.append(t)
                eligible.loc[eligible["ticker"] == t, "reason"] = "retained_incumbent_buffer"
    if len(selected) < ucfg.target_size:
        for t in eligible["ticker"].tolist():
            if t not in selected and len(selected) < ucfg.target_size:
                selected.append(t)

    snap["in_universe"]  = snap["ticker"].isin(selected)
    snap["recon_date"]   = date.date()
    snap["prior_member"] = snap["ticker"].isin(prior)
    snap["entered"]      = snap["in_universe"] & ~snap["prior_member"]
    snap["exited"]       = ~snap["in_universe"] & snap["prior_member"]
    return selected, snap


def build_canonical_universe_schedule(
    close:        pd.DataFrame,
    volume:       pd.DataFrame,
    universe_cfg: UniverseConfig,
    tickers_in_scope: List[str],
    start:        str,
    end:          str,
    registry_df:  Optional[pd.DataFrame] = None,
    quality_df:   Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    recon_dates = pd.date_range(start, end, freq=universe_cfg.recon_freq)
    recon_dates = pd.DatetimeIndex([d for d in recon_dates if d >= pd.Timestamp(start)])

    schedule_rows = []
    snapshots     = []
    prior_members = None
    first_date    = recon_dates[0] if len(recon_dates) else None
    for rd in recon_dates:
        members, snap = _build_universe_snapshot(
            rd, close, volume, universe_cfg, prior_members, first_date, tickers_in_scope,
            registry_df=registry_df, quality_df=quality_df
        )
        schedule_rows.append({"recon_date": rd, "members": json.dumps(members), "n_members": len(members)})
        snapshots.append(snap)
        prior_members = members
    return pd.DataFrame(schedule_rows), snapshots



# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8B — CORRELATION SHIELD (Mahoraga 6)
# ═══════════════════════════════════════════════════════════════════════════════

def _pairwise_corr_avg(corr: pd.DataFrame) -> float:
    """Average off-diagonal pairwise correlation. Returns NaN if ill-defined."""
    if corr is None or corr.empty:
        return np.nan
    c = corr.values.astype(float)
    n = c.shape[0]
    if n < 2:
        return np.nan
    # Exclude diagonal
    denom = n * (n - 1)
    return float((np.nansum(c) - np.nansum(np.diag(c))) / denom) if denom > 0 else np.nan


def compute_corr_shield_series(
    rets: pd.DataFrame,
    idx: pd.DatetimeIndex,
    cfg: Mahoraga4Config,
    univ_master: List[str],
    use_pit_universe: bool,
    universe_schedule: Optional[pd.DataFrame] = None,
    vix: Optional[pd.Series] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Correlation Shield = exposure-scale circuit breaker driven by avg pairwise correlation.

    Ex-ante guarantee:
      For day t we only use returns in [t-window, t-1]. The scale is then shifted
      by 1 day in backtest execution (same as other overlays).
    """
    rho = pd.Series(np.nan, index=idx, name="corr_rho_avg")
    sc  = pd.Series(1.0,  index=idx, name="corr_scale")
    st  = pd.Series(0,    index=idx, name="corr_state")

    if not bool(getattr(cfg, "enable_corr_shield", False)):
        return rho.fillna(0.0), sc, st

    w        = int(getattr(cfg, "corr_window", 21))
    rho_in   = float(getattr(cfg, "corr_rho_in", 0.75))
    rho_out  = float(getattr(cfg, "corr_rho_out", 0.65))
    gamma    = float(getattr(cfg, "corr_gamma", 2.0))
    floor    = float(getattr(cfg, "corr_scale_floor", 0.0))
    hard_k   = bool(getattr(cfg, "corr_hard_kill", True))
    hard_rho = float(getattr(cfg, "corr_hard_rho", 0.90))
    vix_conf = bool(getattr(cfg, "corr_use_vix_confirm", False))
    vix_thr  = float(getattr(cfg, "corr_vix_thr", 24.0))

    _st = 0
    for i, dt in enumerate(idx):
        members = None
        if use_pit_universe and universe_schedule is not None and not universe_schedule.empty:
            members = get_universe_at_date(universe_schedule, dt)
        else:
            members = univ_master

        members = [t for t in members if t in rets.columns]
        if len(members) < 2 or i < w:
            st.iloc[i] = _st
            continue

        win = rets[members].iloc[i - w:i]  # up to dt-1
        # Corr needs enough data; relax min periods to avoid dropping too much.
        corr = win.corr(min_periods=max(3, w // 3)).replace([np.inf, -np.inf], np.nan)
        rbar = _pairwise_corr_avg(corr)
        rho.iloc[i] = rbar

        # Hysteresis state machine
        if _st == 0 and np.isfinite(rbar) and rbar >= rho_in:
            _st = 1
        elif _st == 1 and (not np.isfinite(rbar) or rbar <= rho_out):
            _st = 0
        st.iloc[i] = _st

        if _st == 0:
            sc.iloc[i] = 1.0
            continue

        base = rho_out  # once in, we keep shielding until rho <= rho_out
        denom = max(1e-9, 1.0 - base)
        lam = 0.0 if not np.isfinite(rbar) else max(0.0, min(1.0, (rbar - base) / denom))
        scale = 1.0 - (lam ** gamma)
        scale = max(floor, min(1.0, scale))

        # Optional hard-kill (force cash)
        if hard_k and np.isfinite(rbar) and rbar >= hard_rho:
            if not vix_conf:
                scale = 0.0
            else:
                vx = np.nan
                if vix is not None and dt in vix.index:
                    vx = float(vix.loc[dt])
                if np.isfinite(vx) and vx >= vix_thr:
                    scale = 0.0

        sc.iloc[i] = scale

    rho = rho.ffill().fillna(0.0)
    sc  = sc.ffill().fillna(1.0).clip(0.0, 1.0)
    st  = st.ffill().fillna(0).astype(int)
    return rho, sc, st

def backtest(
    ohlcv:             Dict[str, pd.DataFrame],
    cfg:               Mahoraga4Config,
    costs:             CostsConfig,
    label:             str = "MAHORAGA_5",
    universe:          Optional[List[str]] = None,
    universe_schedule: Optional[pd.DataFrame] = None,
    external_scale:    Optional[pd.Series] = None,
) -> Dict:
    np.random.seed(cfg.random_seed)
    if universe_schedule is not None:
        all_sched_tickers = set()
        for members_json in universe_schedule["members"]:
            all_sched_tickers |= set(json.loads(members_json))
        univ_master = sorted(all_sched_tickers & set(ohlcv["close"].columns))
        use_pit_universe = True
    elif universe is not None:
        univ_master = [t for t in universe if t in ohlcv["close"].columns]
        use_pit_universe = False
    else:
        univ_master = [t for t in cfg.universe_static if t in ohlcv["close"].columns]
        use_pit_universe = False
    if not univ_master:
        raise ValueError("[backtest] No valid tickers in universe")

    close  = ohlcv["close"][univ_master].copy()
    high   = ohlcv["high"][univ_master].copy()
    low    = ohlcv["low"][univ_master].copy()
    volume = ohlcv["volume"][univ_master].copy()
    idx    = close.index

    qqq = to_s(ohlcv["close"][cfg.bench_qqq].reindex(idx).ffill(), "QQQ")
    spy = to_s(ohlcv["close"][cfg.bench_spy].reindex(idx).ffill(), "SPY")

    crisis_scale, crisis_state = compute_crisis_gate(qqq, cfg)
    turb_scale = compute_turbulence(close, volume, qqq, cfg)
    scores = compute_scores(close, qqq, cfg)
    active = select_topk(scores, cfg.top_k, cfg.rebalance_freq)
    rets = close.pct_change().fillna(0.0)

    # ── Correlation Shield (Mahoraga 6) ───────────────────────────────────────
    vix_series = None
    if cfg.bench_vix in ohlcv.get("close", pd.DataFrame()).columns:
        vix_series = to_s(ohlcv["close"][cfg.bench_vix].reindex(idx).ffill(), "VIX")
    corr_rho, corr_scale, corr_state = compute_corr_shield_series(
        rets, idx, cfg, univ_master, use_pit_universe,
        universe_schedule=universe_schedule, vix=vix_series
    )
    reb_dates = set(close.resample(cfg.rebalance_freq).last().index)

    w = pd.DataFrame(0.0, index=idx, columns=univ_master)
    last_w = pd.Series(0.0, index=univ_master)
    for dt in idx:
        if dt in reb_dates:
            if use_pit_universe:
                pit_members = get_universe_at_date(universe_schedule, dt)
                pit_members = [t for t in pit_members if t in univ_master]
                if not pit_members:
                    last_w = pd.Series(0.0, index=univ_master)
                    continue
                pit_scores = scores.loc[dt, pit_members] if pit_members else pd.Series(dtype=float)
                sel_names = pit_scores.nlargest(cfg.top_k).index.tolist()
                names = [n for n in sel_names if pit_scores.get(n, 0) > 0]
            else:
                sel = active.loc[dt]
                names = sel[sel > 0].index.tolist()
            if not names:
                last_w = pd.Series(0.0, index=univ_master)
            elif len(names) == 1:
                last_w = pd.Series(0.0, index=univ_master)
                last_w[names[0]] = 1.0
            else:
                lb = rets.loc[:dt].tail(cfg.hrp_window)[names].dropna()
                if len(lb) < 60:
                    _ret_fallback = (lb if len(lb) > len(names) else rets.loc[:dt][names].dropna())
                    ww = hrp_weights(_ret_fallback).reindex(names, fill_value=0.0)
                else:
                    ww = hrp_weights(lb).reindex(names, fill_value=0.0)
                if ww.sum() > 0:
                    ww = ww.clip(upper=cfg.weight_cap) / ww.clip(upper=cfg.weight_cap).sum()
                last_w = pd.Series(0.0, index=univ_master)
                last_w[names] = ww.reindex(names, fill_value=0.0).values
        w.loc[dt] = last_w.values

    w_stop, stop_hits = apply_chandelier(w, close, high, low, cfg)
    w_exec_1x = w_stop.shift(1).fillna(0.0)
    gross_1x  = (w_exec_1x * rets).sum(axis=1)
    vol_sc    = vol_target_scale(gross_1x, cfg)
    ext_scale = to_s(external_scale, "external_scale").reindex(idx).ffill().fillna(1.0).clip(0.0, 1.0) \
        if external_scale is not None else pd.Series(1.0, index=idx, name="external_scale")
    cap = (crisis_scale * turb_scale * corr_scale * ext_scale).clip(0.0, cfg.max_exposure)
    tgt_sc = pd.Series(np.minimum(vol_sc.values, cap.values), index=idx).clip(0.0, cfg.max_exposure)
    exec_sc = tgt_sc.shift(1).fillna(0.0)
    w_exec = w_exec_1x.mul(exec_sc, axis=0)
    to, tc = _costs(w_exec, costs)
    port_net = ((w_exec * rets).sum(axis=1) - tc).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    equity = cfg.capital_initial * (1.0 + port_net).cumprod()
    exposure = w_exec.abs().sum(axis=1).clip(0.0, cfg.max_exposure)
    qqq_r  = qqq.pct_change().fillna(0.0) - costs.qqq_expense_ratio
    spy_r  = spy.pct_change().fillna(0.0)
    qqq_eq = cfg.capital_initial * (1.0 + qqq_r).cumprod()
    spy_eq = cfg.capital_initial * (1.0 + spy_r).cumprod()
    return {
        "label": label, "returns_net": port_net, "equity": equity,
        "exposure": exposure, "turnover": to,
        "weights_scaled": w_exec, "total_scale": exec_sc, "total_scale_target": tgt_sc,
        "cap": cap, "turb_scale": turb_scale, "crisis_scale": crisis_scale,
        "crisis_state": crisis_state, "vol_scale": vol_sc,
        "external_scale": ext_scale,
        "corr_scale": corr_scale, "corr_rho": corr_rho, "corr_state": corr_state,
        "stop_hits": stop_hits, "scores": scores,
        "bench": {"QQQ_r": qqq_r, "QQQ_eq": qqq_eq, "SPY_r": spy_r, "SPY_eq": spy_eq},
    }


def run_walk_forward(
    ohlcv:             Dict,
    cfg_base:          Mahoraga4Config,
    costs:             CostsConfig,
    universe_schedule: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, List[Dict], pd.DataFrame, str, Dict]:
    trading_idx = pd.DatetimeIndex(ohlcv["close"].index)
    folds = build_contiguous_folds(cfg_base, trading_idx)
    all_oos_r, all_oos_exp, all_oos_to, fold_results, all_sweeps = [], [], [], [], []
    last_best = None
    fold_runtime = []

    for fold in folds:
        fold_n = fold["fold"]
        train_start, train_end = fold["train_start"], fold["train_end"]
        val_start, val_end = fold["val_start"], fold["val_end"]
        test_start, test_end = fold["test_start"], fold["test_end"]
        is_partial = fold["is_partial"]
        print(f"\n  ── FOLD {fold_n}/{len(folds)} ──")

        cfg_f = deepcopy(cfg_base)
        qqq_full = to_s(ohlcv["close"][cfg_base.bench_qqq].ffill())
        dd_thr, vol_thr = calibrate_crisis_thresholds(qqq_full, train_start, train_end, cfg_f)
        cfg_f.crisis_dd_thr = dd_thr
        cfg_f.crisis_vol_zscore_thr = vol_thr

        print(f"  [fold {fold_n}] Fitting IC weights on train …")
        train_tickers = get_training_universe(train_end, universe_schedule,
                                              cfg_base.universe_static, list(ohlcv["close"].columns))
        close_univ = ohlcv["close"][train_tickers]
        ic_weights = fit_ic_weights(close_univ, qqq_full.loc[train_start:train_end], cfg_f, train_start, train_end)

        print(f"  [fold {fold_n}] Sweeping val {val_start}→{val_end} …")
        sweep_df, best = run_fold_sweep(ohlcv, cfg_f, costs, ic_weights,
                                        val_start, val_end, fold_n,
                                        universe_schedule=universe_schedule)
        sweep_df["fold"] = fold_n
        all_sweeps.append(sweep_df)
        last_best = best

        best_cfg = deepcopy(best["cfg"])
        res_test = backtest(ohlcv, best_cfg, costs, label=f"FOLD{fold_n}_TEST",
                            universe_schedule=universe_schedule)
        r_test = res_test["returns_net"].loc[test_start:test_end]
        eq_test = best_cfg.capital_initial * (1.0 + r_test).cumprod()
        exp_test = res_test["exposure"].loc[r_test.index]
        to_test = res_test["turnover"].loc[r_test.index]
        s_test = summarize(r_test, eq_test, exp_test, to_test, best_cfg, f"FOLD{fold_n}_TEST")
        qqq_test = res_test["bench"]["QQQ_r"].loc[test_start:test_end]
        alpha_nw = alpha_test_nw(r_test, qqq_test, best_cfg, f"fold{fold_n}_test")

        score_disp = float(res_test["scores"].loc[r_test.index].std(axis=1).mean()) if len(r_test) else np.nan
        fold_results.append({
            "fold": fold_n,
            "train": f"{train_start}→{train_end}",
            "val": f"{val_start}→{val_end}",
            "test": f"{test_start}→{test_end}",
            "is_partial": is_partial,
            "actual_test_days": fold["actual_test_days"],
            **{f"best_{k}": v for k, v in best["combo_params"].items()},
            "val_score": round(best["score"], 4),
            "val_sharpe": round(best["s_val"]["Sharpe"], 4),
            "val_p_value": round(best["p_value"], 6),
            "val_q_value": round(best["q_value"], 6),
            "val_sig_5pct": best["sig_5pct"],
            "val_stat_label": best["stat_label"],
            "test_CAGR%": round(s_test["CAGR"] * 100, 2),
            "test_Sharpe": round(s_test["Sharpe"], 4),
            "test_MaxDD%": round(s_test["MaxDD"] * 100, 2),
            "test_Calmar": round(s_test["Calmar"], 4),
            "test_alpha_ann%": round(alpha_nw.get("alpha_ann", np.nan) * 100, 2),
            "test_t_alpha": round(alpha_nw.get("t_alpha", np.nan), 3),
            "AvgExposure": round(s_test["AvgExposure"] * 100, 1),
            "AvgTurnover%": round(float(to_test.mean() * 100), 4),
            "AvgTurbScale": round(float(res_test["turb_scale"].loc[r_test.index].mean()), 4),
            "AvgCrisisScale": round(float(res_test["crisis_scale"].loc[r_test.index].mean()), 4),
            "AvgVolScale": round(float(res_test["vol_scale"].loc[r_test.index].mean()), 4),
            "AvgCash%": round(float((1.0 - exp_test).clip(lower=0.0).mean() * 100), 2),
            "ScoreDispersion": round(score_disp, 6) if np.isfinite(score_disp) else np.nan,
        })
        fold_runtime.append({
            "fold": fold_n,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
            "best_cfg": deepcopy(best_cfg),
        })
        all_oos_r.append(r_test)
        all_oos_exp.append(exp_test)
        all_oos_to.append(to_test)

        partial_str = " [PARTIAL FOLD]" if is_partial else ""
        print(f"  [fold {fold_n}] Sharpe={s_test['Sharpe']:.3f}  CAGR={s_test['CAGR']*100:.1f}%  "
              f"DD={s_test['MaxDD']*100:.1f}%  q={best['q_value']:.4f}  {best['stat_label']}{partial_str}")

    oos_r = pd.concat(all_oos_r).sort_index()
    oos_exp = pd.concat(all_oos_exp).sort_index().reindex(oos_r.index)
    oos_to  = pd.concat(all_oos_to).sort_index().reindex(oos_r.index).fillna(0.0)
    dup = oos_r.index.duplicated()
    if dup.any():
        oos_r   = oos_r[~dup]
        oos_exp = oos_exp[~dup]
        oos_to  = oos_to[~dup]
    _, oos_label = validate_oos_continuity(folds, trading_idx)
    print(f"\n  [wf] OOS type: {oos_label}")
    oos_eq = cfg_base.capital_initial * (1.0 + oos_r).cumprod()
    selected_config_info = {
        "source": "last_fold_winner",
        "fold": last_best and len(fold_results),
        "combo_params": last_best["combo_params"] if last_best else {},
        "val_score": last_best["score"] if last_best else np.nan,
        "val_p_value": last_best["p_value"] if last_best else np.nan,
        "val_q_value": last_best["q_value"] if last_best else np.nan,
        "val_sig_5pct": last_best["sig_5pct"] if last_best else False,
        "val_stat_label": last_best["stat_label"] if last_best else "unknown",
        "val_sharpe": last_best["s_val"]["Sharpe"] if last_best else np.nan,
        "fold_runtime": fold_runtime,
    }
    return oos_r, oos_eq, oos_exp, oos_to, fold_results, pd.concat(all_sweeps, ignore_index=True), oos_label, selected_config_info


def _future_compound_return(r: pd.Series, horizon: int) -> pd.Series:
    s = to_s(r).fillna(0.0)
    vals = s.values
    out = np.full(len(vals), np.nan)
    for i in range(len(vals) - horizon):
        out[i] = float(np.prod(1.0 + vals[i+1:i+1+horizon]) - 1.0)
    return pd.Series(out, index=s.index, name=f"fwd_{horizon}")


def build_regime_feature_frame(
    ohlcv: Dict[str, pd.DataFrame],
    base_res: Dict,
    cfg: Mahoraga4Config,
) -> pd.DataFrame:
    scores = base_res["scores"].copy()
    idx = scores.index
    univ_cols = [c for c in scores.columns if c in ohlcv["close"].columns]
    close = ohlcv["close"][univ_cols].reindex(idx).ffill()
    rets = close.pct_change().fillna(0.0)
    qqq_close = to_s(ohlcv["close"][cfg.bench_qqq].reindex(idx).ffill(), "QQQ")
    qqq_r = qqq_close.pct_change().fillna(0.0)
    qqq_dd = qqq_close / qqq_close.cummax() - 1.0

    avg_corr = pd.Series(np.nan, index=idx)
    win = 21
    if len(univ_cols) >= 2:
        for i in range(win, len(idx)):
            sub = rets.iloc[i-win+1:i+1]
            c = sub.corr().values
            n = c.shape[0]
            avg_corr.iloc[i] = (c.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    avg_corr = avg_corr.ffill().fillna(0.0)

    sma63 = close.rolling(63).mean()
    breadth_63dma = (close > sma63).mean(axis=1)
    breadth_pos21 = (close.pct_change(21) > 0).mean(axis=1)
    dispersion_21 = rets.rolling(21).std().mean(axis=1)
    score_disp = scores.std(axis=1)

    def _top_gap(row):
        vals = np.sort(row.dropna().values)
        if len(vals) == 0:
            return np.nan
        top = vals[-1]
        k = min(3, len(vals))
        peer = vals[-k:].mean()
        return float(top - peer)

    top_gap = scores.apply(_top_gap, axis=1)
    feat = pd.DataFrame({
        "qqq_ret_5": qqq_close.pct_change(5),
        "qqq_ret_21": qqq_close.pct_change(21),
        "qqq_ret_63": qqq_close.pct_change(63),
        "qqq_vol_21": qqq_r.rolling(21).std() * np.sqrt(cfg.trading_days),
        "qqq_vol_63": qqq_r.rolling(63).std() * np.sqrt(cfg.trading_days),
        "qqq_dd": qqq_dd,
        "qqq_z_21": safe_z(qqq_r.rolling(21).std().fillna(0.0), 126),
        "avg_corr_21": avg_corr,
        "dispersion_21": dispersion_21,
        "breadth_63dma": breadth_63dma,
        "breadth_pos21": breadth_pos21,
        "score_dispersion": score_disp,
        "score_top_gap": top_gap,
        "exposure": base_res["exposure"].reindex(idx).fillna(0.0),
        "turnover_21": base_res["turnover"].reindex(idx).fillna(0.0).rolling(21).mean(),
        "turb_scale": base_res["turb_scale"].reindex(idx).fillna(1.0),
        "crisis_scale": base_res["crisis_scale"].reindex(idx).fillna(1.0),
        "vol_scale": base_res["vol_scale"].reindex(idx).fillna(1.0),
    }, index=idx)
    feat = feat.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return feat


def build_regime_dataset(
    feature_df: pd.DataFrame,
    strategy_r: pd.Series,
    bench_r: pd.Series,
    rebalance_freq: str,
    horizon_days: int,
) -> pd.DataFrame:
    reb_dates = feature_df.resample(rebalance_freq).last().index
    reb_dates = reb_dates.intersection(feature_df.index)
    ds = feature_df.loc[reb_dates].copy()
    fwd_s = _future_compound_return(strategy_r, horizon_days).reindex(ds.index)
    fwd_b = _future_compound_return(bench_r, horizon_days).reindex(ds.index)
    ds["fwd_strategy"] = fwd_s
    ds["fwd_bench"] = fwd_b
    ds["fwd_excess"] = fwd_s - fwd_b
    ds["y"] = (ds["fwd_excess"] > 0).astype(float)
    ds = ds.dropna(subset=["fwd_strategy", "fwd_bench", "fwd_excess"])
    return ds


def _ml_candidate_models(cfg: Mahoraga5Config) -> Dict[str, Any]:
    models = {}
    if "logit" in cfg.ml_model_choices:
        models["logit"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg.random_seed)),
        ])
    if "gb" in cfg.ml_model_choices:
        models["gb"] = GradientBoostingClassifier(
            random_state=cfg.random_seed, n_estimators=150, learning_rate=0.05,
            max_depth=2, subsample=0.85
        )
    return models


def _gate_policy_objective(gate: np.ndarray, strat_fwd: np.ndarray, bench_fwd: np.ndarray) -> float:
    gated = gate * strat_fwd
    excess = gated - bench_fwd
    mu = np.nanmean(excess)
    sd = np.nanstd(excess, ddof=1)
    sharpe_like = mu / sd if np.isfinite(sd) and sd > 1e-12 else -999.0
    participation = np.nanmean(gate)
    return float(sharpe_like + 0.25 * mu - 0.05 * abs(participation - 0.75))


def _fit_select_ml_gate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    cfg: Mahoraga5Config,
) -> Dict[str, Any]:
    result = {
        "status": "neutral",
        "model_name": "neutral",
        "threshold": np.nan,
        "low_scale": 1.0,
        "auc_val": np.nan,
        "brier_val": np.nan,
        "feature_importance": pd.DataFrame(),
    }
    if len(train_df) < cfg.ml_min_train_samples or len(val_df) < 10:
        return result
    y_train = train_df["y"].astype(int)
    if y_train.nunique() < 2:
        return result

    best = None
    for model_name, model in _ml_candidate_models(cfg).items():
        try:
            model.fit(train_df[feature_cols], y_train)
            proba_val = model.predict_proba(val_df[feature_cols])[:, 1]
            auc_val = roc_auc_score(val_df["y"].astype(int), proba_val) if val_df["y"].nunique() > 1 else np.nan
            brier_val = brier_score_loss(val_df["y"].astype(int), proba_val)
            for thr in cfg.ml_policy_threshold_grid:
                for low_scale in cfg.ml_policy_low_scale_grid:
                    gate = np.where(proba_val >= thr, 1.0, low_scale)
                    obj = _gate_policy_objective(gate, val_df["fwd_strategy"].values, val_df["fwd_bench"].values)
                    cand = {
                        "model_name": model_name,
                        "model": model,
                        "threshold": float(thr),
                        "low_scale": float(low_scale),
                        "objective": float(obj),
                        "auc_val": float(auc_val) if np.isfinite(auc_val) else np.nan,
                        "brier_val": float(brier_val),
                        "proba_val": proba_val,
                    }
                    if best is None or cand["objective"] > best["objective"]:
                        best = cand
        except Exception as e:
            print(f"  [ml_gate] WARNING: model {model_name} failed on validation ({e})")

    if best is None:
        return result

    # Refit winner on train+val for test-time prediction.
    X_tv = pd.concat([train_df[feature_cols], val_df[feature_cols]], axis=0)
    y_tv = pd.concat([train_df["y"].astype(int), val_df["y"].astype(int)], axis=0)
    winner = _ml_candidate_models(cfg)[best["model_name"]]
    winner.fit(X_tv, y_tv)

    # Feature importance extraction
    fi_rows = []
    if best["model_name"] == "logit":
        clf = winner.named_steps["clf"]
        coefs = clf.coef_[0]
        for f, v in zip(feature_cols, coefs):
            fi_rows.append({"feature": f, "importance": float(abs(v)), "signed_weight": float(v)})
    elif hasattr(winner, "feature_importances_"):
        for f, v in zip(feature_cols, winner.feature_importances_):
            fi_rows.append({"feature": f, "importance": float(v), "signed_weight": np.nan})

    return {
        "status": "trained",
        "model_name": best["model_name"],
        "model": winner,
        "threshold": best["threshold"],
        "low_scale": best["low_scale"],
        "auc_val": best["auc_val"],
        "brier_val": best["brier_val"],
        "feature_importance": pd.DataFrame(fi_rows).sort_values("importance", ascending=False) if fi_rows else pd.DataFrame(),
    }


def _daily_gate_from_rebalance_predictions(
    pred_df: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    default_scale: float = 1.0,
) -> pd.Series:
    if pred_df is None or pred_df.empty:
        return pd.Series(default_scale, index=daily_index)
    pred_df = pred_df.sort_values("date")
    out = pd.Series(np.nan, index=daily_index, dtype=float)
    dates = list(pd.DatetimeIndex(pred_df["date"]))
    scales = list(pred_df["gate_scale"].astype(float))
    for i, dt in enumerate(dates):
        next_dt = dates[i+1] if i+1 < len(dates) else (daily_index[-1] + pd.Timedelta(days=1))
        mask = (daily_index >= dt) & (daily_index < next_dt)
        out.loc[mask] = scales[i]
    return out.ffill().fillna(default_scale).clip(0.0, 1.0)


def run_ml_regime_gate(
    ohlcv: Dict[str, pd.DataFrame],
    costs: CostsConfig,
    cfg: Mahoraga5Config,
    selected_config_info: Dict,
    universe_schedule: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    fold_runtime = selected_config_info.get("fold_runtime", [])
    if not fold_runtime:
        return {"status": "disabled", "reason": "no_fold_runtime"}

    all_oos_r_ml = []
    fold_rows = []
    pred_rows = []
    calib_rows = []
    fi_rows = []
    horizon = cfg.ml_horizon_days

    for fd in fold_runtime:
        fold_n = fd["fold"]
        best_cfg = deepcopy(fd["best_cfg"])
        base_res = backtest(ohlcv, best_cfg, costs, label=f"ML_BASE_F{fold_n}", universe_schedule=universe_schedule)
        feat = build_regime_feature_frame(ohlcv, base_res, best_cfg)
        ds = build_regime_dataset(feat, base_res["returns_net"], base_res["bench"]["QQQ_r"], best_cfg.rebalance_freq, horizon)
        feature_cols = [c for c in ds.columns if c not in {"fwd_strategy", "fwd_bench", "fwd_excess", "y"}]

        train_df = ds.loc[fd["train_start"]:fd["train_end"]].copy()
        val_df   = ds.loc[fd["val_start"]:fd["val_end"]].copy()
        test_df  = ds.loc[fd["test_start"]:fd["test_end"]].copy()

        ml_fit = _fit_select_ml_gate(train_df, val_df, feature_cols, cfg)
        if ml_fit["status"] != "trained" or test_df.empty:
            gate_daily = pd.Series(1.0, index=base_res["returns_net"].index)
            pred_test = pd.DataFrame({
                "fold": fold_n,
                "date": pd.to_datetime(test_df.index),
                "model": "neutral",
                "probability": 1.0,
                "gate_scale": 1.0,
                "y_true": test_df["y"].values if not test_df.empty else [],
                "fwd_excess": test_df["fwd_excess"].values if not test_df.empty else [],
            }) if not test_df.empty else pd.DataFrame()
            auc_test = np.nan
            brier_test = np.nan
        else:
            proba_test = ml_fit["model"].predict_proba(test_df[feature_cols])[:, 1]
            pred_test = pd.DataFrame({
                "fold": fold_n,
                "date": pd.to_datetime(test_df.index),
                "model": ml_fit["model_name"],
                "probability": proba_test,
                "gate_scale": np.where(proba_test >= ml_fit["threshold"], 1.0, ml_fit["low_scale"]),
                "y_true": test_df["y"].astype(int).values,
                "fwd_excess": test_df["fwd_excess"].values,
            })
            gate_daily = pd.Series(1.0, index=base_res["returns_net"].index)
            gate_daily.update(_daily_gate_from_rebalance_predictions(pred_test[["date", "gate_scale"]], base_res["returns_net"].index))
            auc_test = roc_auc_score(test_df["y"].astype(int), proba_test) if test_df["y"].nunique() > 1 else np.nan
            brier_test = brier_score_loss(test_df["y"].astype(int), proba_test)
            if not ml_fit["feature_importance"].empty:
                fi = ml_fit["feature_importance"].copy()
                fi["fold"] = fold_n
                fi["model"] = ml_fit["model_name"]
                fi_rows.append(fi)

        gated_res = backtest(ohlcv, best_cfg, costs, label=f"FOLD{fold_n}_ML_GATE",
                             universe_schedule=universe_schedule, external_scale=gate_daily)
        r_ml = gated_res["returns_net"].loc[fd["test_start"]:fd["test_end"]]
        eq_ml = best_cfg.capital_initial * (1.0 + r_ml).cumprod()
        s_ml = summarize(r_ml, eq_ml, gated_res["exposure"].loc[r_ml.index], gated_res["turnover"].loc[r_ml.index], best_cfg,
                         f"FOLD{fold_n}_ML_GATE")
        base_r = base_res["returns_net"].loc[fd["test_start"]:fd["test_end"]]
        base_eq = best_cfg.capital_initial * (1.0 + base_r).cumprod()
        s_base = summarize(base_r, base_eq, base_res["exposure"].loc[base_r.index], base_res["turnover"].loc[base_r.index], best_cfg,
                           f"FOLD{fold_n}_BASE")
        qqq_test = base_res["bench"]["QQQ_r"].loc[fd["test_start"]:fd["test_end"]]
        qqq_eq = best_cfg.capital_initial * (1.0 + qqq_test).cumprod()
        s_qqq = summarize(qqq_test, qqq_eq, None, None, best_cfg, "QQQ")

        fold_rows.append({
            "fold": fold_n,
            "model": ml_fit.get("model_name", "neutral"),
            "threshold": ml_fit.get("threshold", np.nan),
            "low_scale": ml_fit.get("low_scale", 1.0),
            "auc_val": ml_fit.get("auc_val", np.nan),
            "brier_val": ml_fit.get("brier_val", np.nan),
            "auc_test": auc_test,
            "brier_test": brier_test,
            "base_sharpe": round(s_base["Sharpe"], 4),
            "ml_sharpe": round(s_ml["Sharpe"], 4),
            "base_cagr": round(s_base["CAGR"], 6),
            "ml_cagr": round(s_ml["CAGR"], 6),
            "base_maxdd": round(s_base["MaxDD"], 6),
            "ml_maxdd": round(s_ml["MaxDD"], 6),
            "qqq_cagr": round(s_qqq["CAGR"], 6),
            "test_days": int(len(r_ml)),
        })
        all_oos_r_ml.append(r_ml)
        if not pred_test.empty:
            pred_rows.append(pred_test)
        calib_rows.append({
            "fold": fold_n,
            "model": ml_fit.get("model_name", "neutral"),
            "auc_val": ml_fit.get("auc_val", np.nan),
            "brier_val": ml_fit.get("brier_val", np.nan),
            "auc_test": auc_test,
            "brier_test": brier_test,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
        })

    if not all_oos_r_ml:
        return {"status": "disabled", "reason": "no_ml_runs"}

    oos_r_ml = pd.concat(all_oos_r_ml).sort_index()
    oos_r_ml = oos_r_ml[~oos_r_ml.index.duplicated()]
    oos_eq_ml = cfg.capital_initial * (1.0 + oos_r_ml).cumprod()
    s_ml_oos = summarize(oos_r_ml, oos_eq_ml, None, None, cfg, "OOS_ML_RegimeGate")
    pred_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    fi_df = pd.concat(fi_rows, ignore_index=True) if fi_rows else pd.DataFrame(columns=["feature", "importance", "signed_weight", "fold", "model"])
    calib_df = pd.DataFrame(calib_rows)
    fold_df = pd.DataFrame(fold_rows)
    return {
        "status": "ok",
        "oos_r_ml": oos_r_ml,
        "oos_eq_ml": oos_eq_ml,
        "oos_summary_ml": s_ml_oos,
        "ml_fold_summary": fold_df,
        "ml_predictions": pred_df,
        "feature_importance": fi_df,
        "calibration_report": calib_df,
    }


def print_results(out: Dict, fold_results: List[Dict], ff, selected_config_info: Dict,
                  ml_artifacts: Optional[Dict[str, Any]] = None):
    sep = "=" * 110
    print(UNIVERSE_BIAS_DISCLAIMER)
    print(f"\n{sep}\n  MAHORAGA 5 — FULL RESULTS\n{sep}")
    oos_label = out.get("oos_label", "OOS")
    full_rows = [_fmt(out["full"]), _fmt(out["qqq_full"]), _fmt(out["eqw_full"]), _fmt(out["mom_full"])]
    print(f"\n{'─'*70}  FULL PERIOD")
    print(pd.DataFrame(full_rows).to_string(index=False))
    oos_rows = [_fmt(out["oos"]), _fmt(out["qqq_oos"])]
    if ml_artifacts and ml_artifacts.get("status") == "ok":
        oos_rows.append(_fmt(ml_artifacts["oos_summary_ml"]))
    print(f"\n{'─'*70}  OOS — {oos_label}")
    print(pd.DataFrame(oos_rows).to_string(index=False))

    print(f"\n{'─'*70}  WALK-FORWARD FOLD SUMMARY")
    fold_df = pd.DataFrame(fold_results).copy()
    for i, row in fold_df.iterrows():
        if row.get("is_partial"):
            fold_df.at[i, "fold"] = f"{row['fold']} (PARTIAL)"
    print(fold_df.to_string(index=False))

    print(f"\n{'─'*70}  SELECTED CONFIGURATION (source: {selected_config_info['source']})")
    print(f"  combo:       {selected_config_info['combo_params']}")
    print(f"  val_score:   {selected_config_info['val_score']:.4f}")
    print(f"  val_sharpe:  {selected_config_info.get('val_sharpe', np.nan):.4f}")
    print(f"  p_value:     {selected_config_info['val_p_value']:.6f}")
    print(f"  q_value:     {selected_config_info['val_q_value']:.6f}")
    print(f"  sig@5%FDR:   {selected_config_info['val_sig_5pct']}")
    print(f"  stat_label:  {selected_config_info['val_stat_label']}")
    if not selected_config_info.get("val_sig_5pct"):
        print("  [NOTE] Config does NOT pass BHY@5% — interpret with caution.")

    if ml_artifacts and ml_artifacts.get("status") == "ok":
        print(f"\n{'─'*70}  ML REGIME GATE — OOS MODULE")
        ml_s = ml_artifacts["oos_summary_ml"]
        print(f"  OOS_ML_RegimeGate: Sharpe={ml_s['Sharpe']:.4f}  CAGR={ml_s['CAGR']*100:.2f}%  MaxDD={ml_s['MaxDD']*100:.2f}%")
        if not ml_artifacts["ml_fold_summary"].empty:
            print(ml_artifacts["ml_fold_summary"].to_string(index=False))

    print(f"\n{'─'*70}  ASYMPTOTIC SHARPE CI")
    for lbl, ci in [("Full", out["sr_ci_full"]), ("OOS", out["sr_ci_oos"])]:
        print(f"  {lbl:6s}: SR={ci['SR']:.4f}  95%CI=[{ci['CI_lo']:.4f},{ci['CI_hi']:.4f}]  t={ci['t_stat']:.3f}  p={ci['p_val']:.6f}")

    print(f"\n{'─'*70}  ALPHA — Newey-West HAC vs QQQ")
    for lbl, a in [("Full", out["alpha_full"]), ("OOS", out["alpha_oos"]), ("Full|exp>0", out["alpha_cond"])]:
        if "error" in a:
            print(f"  {lbl}: ERROR — {a['error']}")
            continue
        sig = "***" if a["sig_1pct"] else ("**" if a["sig_5pct"] else "   ")
        cond_str = " [conditional on exposure>0]" if a.get("conditional") else ""
        print(f"  {lbl:18s}: α={a['alpha_ann']*100:.2f}%  t={a['t_alpha']:.3f}  p={a['p_alpha']:.6f}  β={a['beta']:.4f}  R²={a['R2']:.4f}  n={a.get('n_obs','—')}  {sig}{cond_str}")
    print(EXECUTION_STRESS_DISCLAIMER)
    print(UNIVERSE_BIAS_DISCLAIMER)


def _build_final_report_text(out: Dict, fold_results: List[Dict], selected_config_info: Dict,
                             oos_label: str, ml_artifacts: Optional[Dict[str, Any]] = None) -> str:
    lines = [
        "MAHORAGA 5 — FINAL REPORT",
        "=" * 80,
        UNIVERSE_BIAS_DISCLAIMER,
        "",
        f"OOS type: {oos_label}",
        "",
        "FULL PERIOD SUMMARY",
        "-" * 40,
    ]
    for k, v in _fmt(out["full"]).items():
        lines.append(f"  {k}: {v}")
    lines += ["", "OOS SUMMARY", "-" * 40]
    for k, v in _fmt(out["oos"]).items():
        lines.append(f"  {k}: {v}")
    if ml_artifacts and ml_artifacts.get("status") == "ok":
        lines += ["", "OOS ML REGIME GATE SUMMARY", "-" * 40]
        for k, v in _fmt(ml_artifacts["oos_summary_ml"]).items():
            lines.append(f"  {k}: {v}")
    lines += ["", "SELECTED CONFIGURATION", "-" * 40,
              f"  Source:       {selected_config_info['source']}",
              f"  combo_params: {selected_config_info['combo_params']}",
              f"  val_score:    {selected_config_info['val_score']:.4f}",
              f"  p_value:      {selected_config_info['val_p_value']:.6f}",
              f"  q_value:      {selected_config_info['val_q_value']:.6f}",
              f"  sig@5%FDR:    {selected_config_info['val_sig_5pct']}",
              f"  stat_label:   {selected_config_info['val_stat_label']}"]
    if not selected_config_info.get("val_sig_5pct"):
        lines.append("  [NOTE] Does not pass BHY@5% — interpret with caution.")
    if ml_artifacts and ml_artifacts.get("status") == "ok" and not ml_artifacts["ml_fold_summary"].empty:
        lines += ["", "ML GATE FOLD SUMMARY", "-" * 40, ml_artifacts["ml_fold_summary"].to_string(index=False)]
    lines += ["", EXECUTION_STRESS_DISCLAIMER, "", UNIVERSE_BIAS_DISCLAIMER]
    return "\n".join(lines)


def save_outputs(
    out: Dict,
    fold_results: List[Dict],
    ic_df: pd.DataFrame,
    rob: Dict,
    all_sweeps: pd.DataFrame,
    selected_config_info: Dict,
    oos_label: str,
    cfg: Mahoraga5Config,
    universe_schedule: Optional[pd.DataFrame] = None,
    universe_snapshots: Optional[List[pd.DataFrame]] = None,
    universe_diagnostics: Optional[pd.DataFrame] = None,
    data_quality_report: Optional[pd.DataFrame] = None,
    asset_registry: Optional[pd.DataFrame] = None,
    ml_artifacts: Optional[Dict[str, Any]] = None,
):
    d = cfg.outputs_dir
    _ensure_dir(d)
    def _df(rows):
        return pd.DataFrame([{k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()} for r in rows])

    full_rows = [out["full"], out["qqq_full"], out["eqw_full"], out["mom_full"]]
    _df(full_rows).to_csv(f"{d}/comparison_full.csv", index=False)
    oos_rows = [out["oos"], out["qqq_oos"]]
    if ml_artifacts and ml_artifacts.get("status") == "ok":
        oos_rows.append(ml_artifacts["oos_summary_ml"])
    _df(oos_rows).to_csv(f"{d}/comparison_oos.csv", index=False)
    pd.DataFrame(fold_results).to_csv(f"{d}/walk_forward_folds.csv", index=False)
    pd.DataFrame(fold_results).to_csv(f"{d}/fold_diagnostics.csv", index=False)

    meta = {
        "oos_label": oos_label,
        "n_folds": len(fold_results),
        "oos_start": fold_results[0]["test"].split("→")[0] if fold_results else "—",
        "oos_end": fold_results[-1]["test"].split("→")[1] if fold_results else "—",
        "any_partial_fold": any(f.get("is_partial") for f in fold_results),
        "total_oos_days": sum(f.get("actual_test_days", 0) for f in fold_results),
        "sweep_grid_combos": len(list(iproduct(*[SWEEP_GRID[k] for k in SWEEP_GRID]))),
        "parallel_sweep": cfg.parallel_sweep,
        "universe_mode": "canonical" if cfg.use_canonical_universe else "static_expost",
        "ml_regime_gate": cfg.enable_ml_regime_gate,
    }
    pd.DataFrame([meta]).to_csv(f"{d}/walk_forward_meta.csv", index=False)

    support = {
        **selected_config_info.get("combo_params", {}),
        "source": selected_config_info["source"],
        "val_score": selected_config_info["val_score"],
        "val_p_value": selected_config_info["val_p_value"],
        "val_q_value": selected_config_info["val_q_value"],
        "val_sig_5pct": selected_config_info["val_sig_5pct"],
        "val_stat_label": selected_config_info["val_stat_label"],
        "val_sharpe": selected_config_info.get("val_sharpe", np.nan),
    }
    pd.DataFrame([support]).to_csv(f"{d}/selected_config_support.csv", index=False)

    report_text = _build_final_report_text(out, fold_results, selected_config_info, oos_label, ml_artifacts=ml_artifacts)
    with open(f"{d}/final_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    all_sweeps.to_csv(f"{d}/walk_forward_sweeps.csv", index=False)
    out["stress"].to_csv(f"{d}/stress_full.csv", index=False)
    out["regime_full"].to_csv(f"{d}/regime_full.csv", index=False)
    out["regime_oos"].to_csv(f"{d}/regime_oos.csv", index=False)
    ic_df.to_csv(f"{d}/rolling_ic_multi.csv")
    if rob.get("exec_stress") is not None:
        rob["exec_stress"].to_csv(f"{d}/execution_sensitivity_stress.csv", index=False)
    if rob.get("alt_univ") is not None:
        rob["alt_univ"].to_csv(f"{d}/alt_universes.csv", index=False)
    if rob.get("sensitivity") is not None:
        rob["sensitivity"].to_csv(f"{d}/local_sensitivity.csv", index=False)
    if rob.get("stop_ablation") is not None:
        rob["stop_ablation"].to_csv(f"{d}/stop_ablation.csv", index=False)
    if rob.get("ic_ablation") is not None:
        rob["ic_ablation"].to_csv(f"{d}/ic_ablation.csv", index=False)
    pd.DataFrame([out["alpha_full"], out["alpha_oos"], out["alpha_cond"]]).to_csv(f"{d}/alpha_nw.csv", index=False)
    pd.DataFrame([out["sr_ci_full"], out["sr_ci_oos"]]).to_csv(f"{d}/sharpe_ci.csv", index=False)
    if out.get("ff_full"):
        pd.DataFrame([out["ff_full"], out.get("ff_oos", {})]).to_csv(f"{d}/ff_attribution.csv", index=False)

    if universe_schedule is not None:
        universe_schedule.to_csv(f"{d}/universe_schedule.csv", index=False)
    if universe_snapshots:
        snap_dir = os.path.join(d, "universe_snapshots")
        _ensure_dir(snap_dir)
        for snap in universe_snapshots:
            if snap is None or snap.empty:
                continue
            rd = str(snap["recon_date"].iloc[0]).replace("-", "")
            snap.to_csv(f"{snap_dir}/snapshot_{rd}.csv", index=False)
    if universe_diagnostics is not None and not universe_diagnostics.empty:
        universe_diagnostics.to_csv(f"{d}/universe_diagnostics.csv", index=False)
    if data_quality_report is not None and not data_quality_report.empty:
        data_quality_report.to_csv(f"{d}/data_quality_report.csv", index=False)
    if asset_registry is not None and not asset_registry.empty:
        asset_registry.to_csv(f"{d}/asset_registry.csv", index=False)

    universe_cfg = UniverseConfig()
    methodology = {
        "version": "5.0",
        "universe_mode": "canonical_engine" if cfg.use_canonical_universe else "static_expost",
        "static_universe": list(cfg.universe_static),
        "universe_note": "static universe is ex-post selected mega-cap tech; not survivor-bias-free",
        "canonical_ranking": "LiqSizeProxy = mean(price × volume, 30d) — NOT float-adjusted market cap (FFMC)",
        "canonical_ranking_note": "FFMC/GICS-IT require CRSP/Compustat PIT data, which is not integrated",
        "canonical_recon": universe_cfg.recon_freq,
        "canonical_target_n": universe_cfg.target_size,
        "auto_entry_rank": universe_cfg.auto_entry_rank,
        "retention_rank": universe_cfg.retention_rank,
        "buffer_rank": universe_cfg.buffer_rank,
        "min_seasoning_days": universe_cfg.min_seasoning_days,
        "min_free_float_proxy": "volume_continuity (fraction of days with non-zero vol) — NOT actual float data",
        "min_addv_usd": universe_cfg.min_addv_usd,
        "data_source": "yfinance (simulation proxy — not CRSP/Compustat PIT)",
        "survivorship_bias": "PRESENT — cannot be eliminated without CRSP PIT",
        "data_quality_layer": cfg.enable_data_quality_layer,
        "ml_regime_gate": cfg.enable_ml_regime_gate,
        "disclaimer": UNIVERSE_BIAS_DISCLAIMER.strip(),
    }
    with open(f"{d}/universe_methodology.json", "w", encoding="utf-8") as f:
        json.dump(methodology, f, indent=2)

    if ml_artifacts and ml_artifacts.get("status") == "ok":
        ml_artifacts["ml_fold_summary"].to_csv(f"{d}/ml_ablation_comparison.csv", index=False)
        ml_artifacts["ml_predictions"].to_csv(f"{d}/ml_regime_predictions.csv", index=False)
        ml_artifacts["feature_importance"].to_csv(f"{d}/feature_importance.csv", index=False)
        ml_artifacts["calibration_report"].to_csv(f"{d}/calibration_report.csv", index=False)

    print(f"\n  [outputs → ./{d}/]")
    print("    comparison_full.csv, comparison_oos.csv, walk_forward_folds.csv, fold_diagnostics.csv")
    print("    walk_forward_meta.csv, selected_config_support.csv, final_report.txt, walk_forward_sweeps.csv")
    print("    universe_schedule.csv, universe_diagnostics.csv, data_quality_report.csv, asset_registry.csv")
    if ml_artifacts and ml_artifacts.get("status") == "ok":
        print("    ml_ablation_comparison.csv, ml_regime_predictions.csv, feature_importance.csv, calibration_report.csv")


def make_plots(out, oos_r, oos_eq, fold_results, ic_df, decomp, rob, cfg,
               oos_label: str = "OOS_continuous", ml_artifacts: Optional[Dict[str, Any]] = None):
    p = cfg.plots_dir
    _ensure_dir(p)
    res = out["res"]
    eqw = out["eqw"]
    mom = out["mom"]
    plot_equity({cfg.label: res["equity"], "QQQ": res["bench"]["QQQ_eq"], "EQW": eqw["eq"], "MOM_12_1": mom["eq"]},
                "Full Period Equity — Mahoraga 5", f"{p}/01_equity_full.png")
    qqq_oos_r = res["bench"]["QQQ_r"].reindex(oos_r.index).fillna(0.0)
    qqq_oos_eq = cfg.capital_initial * (1.0 + qqq_oos_r).cumprod()
    eq_curves = {cfg.label: oos_eq, "QQQ (OOS)": qqq_oos_eq}
    if ml_artifacts and ml_artifacts.get("status") == "ok":
        eq_curves["ML_RegimeGate (OOS)"] = ml_artifacts["oos_eq_ml"]
    plot_equity(eq_curves, f"Walk-Forward {oos_label} — Mahoraga 5", f"{p}/02_equity_oos.png")
    plot_drawdown({cfg.label: res["equity"], "QQQ": res["bench"]["QQQ_eq"], "MOM_12_1": mom["eq"]},
                  "Drawdown", f"{p}/03_drawdown.png")
    plot_wf_oos(oos_eq, qqq_oos_eq, fold_results,
                f"Walk-Forward OOS — {oos_label}", f"{p}/04_walkforward.png", oos_label)
    plot_risk_overlays(res, "Risk Overlays", f"{p}/05_risk_overlays.png")
    plot_weights_heatmap(res["weights_scaled"], "Portfolio Weights (monthly avg)", f"{p}/06_weights.png")
    plot_ic_multi_horizon(ic_df, "Rolling IC — 1d/5d/21d", f"{p}/07_ic_multi.png")
    plot_regime_bars(out["regime_full"], "Regime Analysis — Full", f"{p}/08_regime_full.png")
    plot_regime_bars(out["regime_oos"], f"Regime Analysis — {oos_label}", f"{p}/09_regime_oos.png")
    if decomp:
        plot_signal_decomp(decomp, res["bench"]["QQQ_r"], cfg, "Signal Decomposition", f"{p}/10_decomp.png")
    if rob.get("sensitivity") is not None:
        plot_sharpe_surface(rob["sensitivity"], "vol_target_ann", "weight_cap",
                            "Sharpe Surface — vol_target × weight_cap", f"{p}/11_sensitivity.png")


def run_mahoraga6(make_plots_flag: bool = True, run_robustness: bool = True) -> Dict:
    print("=" * 80)
    print("  MAHORAGA 5 — Research Edition")
    print("=" * 80)
    print(UNIVERSE_BIAS_DISCLAIMER)

    cfg = Mahoraga5Config()
    costs = CostsConfig()
    ucfg = UniverseConfig()
    _ensure_dir(cfg.cache_dir)
    _ensure_dir(cfg.plots_dir)
    _ensure_dir(cfg.outputs_dir)

    print("\n[1] Downloading data …")
    equity_tickers = sorted(set(list(cfg.universe_static) + [t for u in ALTERNATE_UNIVERSES.values() for t in u]))
    bench_tickers = [cfg.bench_qqq, cfg.bench_spy, cfg.bench_vix]
    all_tickers = sorted(set(equity_tickers + bench_tickers))
    ohlcv = download_ohlcv(all_tickers, cfg.data_start, cfg.data_end, cfg.cache_dir)

    print("\n[1b] Fama-French factors …")
    ff = load_ff_factors(cfg.cache_dir)

    print("\n[1c] Data hygiene & asset registry …")
    asset_registry = build_asset_registry(equity_tickers, cfg, bench_tickers)
    data_quality_report = compute_data_quality_report(ohlcv, equity_tickers, cfg)
    clean_equity = filter_equity_candidates([t for t in equity_tickers if t in ohlcv["close"].columns],
                                            asset_registry, data_quality_report, cfg)
    print(f"  [dq] raw equity candidates:   {len([t for t in equity_tickers if t in ohlcv['close'].columns])}")
    print(f"  [dq] clean equity candidates: {len(clean_equity)}")
    if len(clean_equity) < cfg.min_universe_names:
        print("  [dq] WARNING: too few clean names after quality filters; reverting to raw equity candidates")
        clean_equity = [t for t in equity_tickers if t in ohlcv["close"].columns]

    universe_schedule = None
    universe_snapshots = None
    universe_diagnostics = pd.DataFrame()
    if cfg.use_canonical_universe:
        print("\n[1d] Canonical universe engine (quarterly LiqSizeProxy reconstitution) …")
        print(f"  [universe] tickers_in_scope = {len(clean_equity)} clean equity candidates (benchmarks excluded)")
        universe_schedule, universe_snapshots = build_canonical_universe_schedule(
            ohlcv["close"], ohlcv["volume"], ucfg, clean_equity,
            cfg.data_start, cfg.data_end,
            registry_df=asset_registry, quality_df=data_quality_report,
        )
        universe_diagnostics = build_universe_diagnostics(universe_schedule)
        n_recon = len(universe_schedule)
        print(f"  [universe] {n_recon} reconstitution dates built")
        if n_recon > 0:
            first_u = json.loads(universe_schedule.iloc[0]["members"])
            last_u = json.loads(universe_schedule.iloc[-1]["members"])
            print(f"  [universe] first recon ({universe_schedule.iloc[0]['recon_date']}): {first_u}")
            print(f"  [universe] last  recon ({universe_schedule.iloc[-1]['recon_date']}): {last_u}")
        print("  [universe] Schedule governs backtest point-in-time at each rebalance.")
    else:
        print("\n[1d] *** WARNING: use_canonical_universe=False ***")
        print("     Running on static ex-post universe. Survivorship bias is PRESENT.")
        print(f"     Static universe: {list(cfg.universe_static)}")

    print("\n[2] Walk-forward …")
    oos_r, oos_eq, fold_results, all_sweeps, oos_label, selected_config_info = \
        run_walk_forward(ohlcv, cfg, costs, universe_schedule=universe_schedule)
    oos_summary = summarize(oos_r, oos_eq, None, None, cfg, oos_label)
    print(f"\n  OOS type:   {oos_label}")
    print(f"  OOS Sharpe={oos_summary['Sharpe']:.3f}  CAGR={oos_summary['CAGR']*100:.1f}%  MaxDD={oos_summary['MaxDD']*100:.1f}%")

    print("\n[3] Applying last-fold winning combo as final config …")
    last_train_end = cfg.wf_folds[-1][0]
    cfg_final = deepcopy(cfg)
    qqq_full = to_s(ohlcv["close"][cfg.bench_qqq].ffill())
    dd_thr, vol_thr = calibrate_crisis_thresholds(qqq_full, cfg.wf_train_start, last_train_end, cfg_final)
    cfg_final.crisis_dd_thr = dd_thr
    cfg_final.crisis_vol_zscore_thr = vol_thr
    for k, v in selected_config_info["combo_params"].items():
        setattr(cfg_final, k, v)
    final_train_tickers = get_training_universe(last_train_end, universe_schedule,
                                                cfg.universe_static, list(ohlcv["close"].columns))
    close_univ = ohlcv["close"][final_train_tickers]
    wt, wm, wr = fit_ic_weights(close_univ, qqq_full.loc[cfg.wf_train_start:last_train_end], cfg_final,
                                 cfg.wf_train_start, last_train_end)
    cfg_final.w_trend = wt
    cfg_final.w_mom = wm
    cfg_final.w_rel = wr

    print(f"\n  Final config: {selected_config_info['combo_params']}")
    print(f"  Statistical support (last fold): p={selected_config_info['val_p_value']:.6f}  q={selected_config_info['val_q_value']:.6f}  sig@5%={selected_config_info['val_sig_5pct']}  label={selected_config_info['val_stat_label']}")

    print("\n[4] Full evaluation …")
    out = final_evaluation(ohlcv, cfg_final, costs, ff, oos_r, oos_eq, oos_label, universe_schedule=universe_schedule)

    print("\n[5] Rolling IC (1d/5d/21d) …")
    ic_df = rolling_ic_multi_horizon(close_univ, qqq_full, cfg_final, window=63)

    print("\n[6] Signal decomposition …")
    decomp = baseline_signal_decomp(ohlcv, cfg_final, costs, universe_schedule=universe_schedule)

    ml_artifacts = {"status": "disabled", "reason": "config_off"}
    if cfg.enable_ml_regime_gate:
        print("\n[6b] ML RegimeGate (nested research module) …")
        ml_artifacts = run_ml_regime_gate(ohlcv, costs, cfg_final, selected_config_info, universe_schedule=universe_schedule)
        if ml_artifacts.get("status") == "ok":
            ml_s = ml_artifacts["oos_summary_ml"]
            print(f"  [ml_gate] OOS Sharpe={ml_s['Sharpe']:.3f}  CAGR={ml_s['CAGR']*100:.1f}%  MaxDD={ml_s['MaxDD']*100:.1f}%")
        else:
            print(f"  [ml_gate] skipped — {ml_artifacts.get('reason', 'unknown')}")

    rob = {}
    if run_robustness:
        print("\n[7a] Execution_Sensitivity_Stress …")
        print(EXECUTION_STRESS_DISCLAIMER)
        rob["exec_stress"] = execution_sensitivity_stress(ohlcv, cfg_final, costs, universe_schedule=universe_schedule)
        print("\n[7b] Alternate universes …")
        rob["alt_univ"] = alternate_universe_stress(ohlcv, cfg_final, costs)
        print("\n[7c] Local parameter sensitivity …")
        rob["sensitivity"] = local_sensitivity(ohlcv, cfg_final, costs, universe_schedule=universe_schedule)
        print("\n[7d] Stop ablation …")
        rob["stop_ablation"] = stop_keep_cash_ablation(ohlcv, cfg_final, costs, universe_schedule=universe_schedule)
        print("\n[7e] IC weight ablation …")
        rob["ic_ablation"] = ic_weight_ablation(ohlcv, cfg_final, costs, universe_schedule=universe_schedule)

    print_results(out, fold_results, ff, selected_config_info, ml_artifacts=ml_artifacts)
    save_outputs(out, fold_results, ic_df, rob, all_sweeps, selected_config_info, oos_label, cfg_final,
                 universe_schedule=universe_schedule, universe_snapshots=universe_snapshots,
                 universe_diagnostics=universe_diagnostics, data_quality_report=data_quality_report,
                 asset_registry=asset_registry, ml_artifacts=ml_artifacts)

    if make_plots_flag:
        print("\n[8] Generating plots …")
        make_plots(out, oos_r, oos_eq, fold_results, ic_df, decomp, rob, cfg_final, oos_label, ml_artifacts=ml_artifacts)

    return {
        "cfg": cfg_final,
        "out": out,
        "oos_r": oos_r,
        "oos_eq": oos_eq,
        "oos_label": oos_label,
        "fold_results": fold_results,
        "ic_df": ic_df,
        "decomp": decomp,
        "rob": rob,
        "selected_config_info": selected_config_info,
        "universe_schedule": universe_schedule,
        "universe_diagnostics": universe_diagnostics,
        "data_quality_report": data_quality_report,
        "asset_registry": asset_registry,
        "ml_artifacts": ml_artifacts,
    }



@dataclass
class Mahoraga6Config(Mahoraga5Config):
    """
    MAHORAGA 6 — research hardening release.

    Goals:
      1) Fix audit/reporting bugs (OOS exposure/turnover were NaN/0 in v5).
      2) Add a correlation-based risk 'circuit breaker' to address systemic
         correlation blow-ups (esp. VIX panic regimes).
      3) Expand audit trail + ablations so claims match implementation.

    Notes:
      - No ML/IA is used in Mahoraga 6. The ML gate remains an optional,
        explicitly-separated research module (Mahoraga 7+).
    """

    plots_dir:      str = "mahoraga6_plots"
    outputs_dir:    str = "mahoraga6_outputs"
    label:          str = "MAHORAGA_6"

    # Disable ML/IA modules in v6 (planned for v7+)
    enable_ml_regime_gate: bool = False

    # ── Correlation Shield (new in v6) ─────────────────────────────────────────
    enable_corr_shield: bool = True

    # Rolling window (trading days) for pairwise-correlation estimate.
    corr_window: int = 21

    # Hysteresis: enter shielding above rho_in; exit below rho_out.
    corr_rho_in:  float = 0.75
    corr_rho_out: float = 0.65

    # Scaling shape: VT_t = VT0 * (1 - Lambda^gamma), Lambda in [0,1].
    corr_gamma: float = 2.0

    # Floor for the correlation scale (soft mode). Use 0 to allow full cash.
    corr_scale_floor: float = 0.00

    # Optional hard-kill: if avg correlation >= hard_rho then force cash.
    corr_hard_kill: bool = True
    corr_hard_rho:  float = 0.90

    # Optional VIX confirm for hard-kill (reduces false positives).
    corr_use_vix_confirm: bool = False
    corr_vix_thr: float = 24.0

    # Selection protocol (kept default = last fold winner for continuity).
    final_selection_method: str = "last_fold_winner"  # {"last_fold_winner","avg_val_across_folds"}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 19 — MAHORAGA 5 CONSOLIDATED  (NO ML / NO IA)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Mahoraga5Config(Mahoraga4Config):
    """
    Mahoraga 5 consolidates the stable Mahoraga 4 research stack and keeps:
      - data hygiene / asset registry,
      - canonical PIT universe schedule,
      - walk-forward purity / audit trail,
      - robustness suite,
      - corrected OOS benchmark rebasing in plots.

    The ML regime gate explored previously is intentionally REMOVED from the
    execution path in this version. Mahoraga 6 is the designated version for
    any future machine-learning module.
    """
    use_canonical_universe: bool = True
    plots_dir:   str = "mahoraga5_canonical_plots"
    outputs_dir: str = "mahoraga5_canonical_outputs"
    label:       str = "MAHORAGA_5"

    # Data hygiene / universe hardening
    enable_data_quality_layer: bool = True
    asset_registry_path: Optional[str] = None
    dq_min_history_days: int = 252
    dq_max_missing_close_pct: float = 0.10
    dq_max_zero_volume_pct: float = 0.35
    dq_max_stale_price_streak: int = 10
    min_universe_names: int = 6


def print_results(out: Dict, fold_results: List[Dict], ff, selected_config_info: Dict):
    sep = "=" * 110
    print(UNIVERSE_BIAS_DISCLAIMER)
    print(f"\n{sep}\n  MAHORAGA 5 — FULL RESULTS\n{sep}")
    oos_label = out.get("oos_label", "OOS")

    print(f"\n{'─'*70}  FULL PERIOD")
    print(pd.DataFrame([_fmt(out["full"]), _fmt(out["qqq_full"]),
                        _fmt(out["eqw_full"]), _fmt(out["mom_full"])]).to_string(index=False))

    print(f"\n{'─'*70}  OOS — {oos_label}")
    print(pd.DataFrame([_fmt(out["oos"]), _fmt(out["qqq_oos"])]).to_string(index=False))

    print(f"\n{'─'*70}  WALK-FORWARD FOLD SUMMARY")
    fold_df = pd.DataFrame(fold_results).copy()
    for i, row in fold_df.iterrows():
        if row.get("is_partial"):
            fold_df.at[i, "fold"] = f"{row['fold']} (PARTIAL)"
    print(fold_df.to_string(index=False))

    print(f"\n{'─'*70}  SELECTED CONFIGURATION (source: {selected_config_info['source']})")
    print(f"  combo:       {selected_config_info['combo_params']}")
    print(f"  val_score:   {selected_config_info['val_score']:.4f}")
    print(f"  val_sharpe:  {selected_config_info.get('val_sharpe', np.nan):.4f}")
    print(f"  p_value:     {selected_config_info['val_p_value']:.6f}")
    print(f"  q_value:     {selected_config_info['val_q_value']:.6f}")
    print(f"  sig@5%FDR:   {selected_config_info['val_sig_5pct']}")
    print(f"  stat_label:  {selected_config_info['val_stat_label']}")
    if not selected_config_info.get("val_sig_5pct"):
        print("  [NOTE] Config does NOT pass BHY@5% — interpret with caution.")

    print(f"\n{'─'*70}  ASYMPTOTIC SHARPE CI")
    for lbl, ci in [("Full", out["sr_ci_full"]), ("OOS", out["sr_ci_oos"] )]:
        print(f"  {lbl:6s}: SR={ci['SR']:.4f}  95%CI=[{ci['CI_lo']:.4f},{ci['CI_hi']:.4f}]  "
              f"t={ci['t_stat']:.3f}  p={ci['p_val']:.6f}")

    print(f"\n{'─'*70}  ALPHA — Newey-West HAC vs QQQ")
    for lbl, a in [("Full", out["alpha_full"]), ("OOS", out["alpha_oos"]), ("Full|exp>0", out["alpha_cond"] )]:
        if "error" in a:
            print(f"  {lbl}: ERROR — {a['error']}")
            continue
        sig = "***" if a["sig_1pct"] else ("**" if a["sig_5pct"] else "   ")
        cond_str = " [conditional on exposure>0]" if a.get("conditional") else ""
        print(f"  {lbl:18s}: α={a['alpha_ann']*100:.2f}%  t={a['t_alpha']:.3f}  "
              f"p={a['p_alpha']:.6f}  β={a['beta']:.4f}  R²={a['R2']:.4f}  "
              f"n={a.get('n_obs','—')}  {sig}{cond_str}")

    if out.get("ff_full") and "error" not in (out["ff_full"] or {}):
        print(f"\n{'─'*70}  FF5+UMD ATTRIBUTION")
        for lbl, fa in [("Full", out["ff_full"]), ("OOS", out["ff_oos"] )]:
            if fa and "error" not in fa:
                print(f"  {lbl}: α={fa['alpha_ann']*100:.2f}%  t={fa['t_alpha']:.3f}  "
                      f"β_mkt={fa['beta_mkt']:.3f}  β_umd={fa['beta_umd']:.3f}  R²_adj={fa['R2_adj']:.3f}")

    print(f"\n{'─'*70}  REGIME ANALYSIS")
    print("  Full period:"); print(out["regime_full"].to_string(index=False))
    print("  OOS:"); print(out["regime_oos"].to_string(index=False))

    print(f"\n{'─'*70}  STRESS EPISODES")
    print(out["stress"].to_string(index=False))

    print(f"\n{'─'*70}  BOOTSTRAP DD (moving block, 1000 samples)")
    for lbl, b in [("Full", out["boot_full"]), ("OOS", out["boot_oos"] )]:
        print(f"  {lbl}: median_DD={b['dd_p50']*100:.1f}%  "
              f"p5_worst={b['dd_p5_worst']*100:.1f}%  "
              f"P(DD<-30%)={b['ruin_prob_30dd']:.1f}%  P(DD<-50%)={b['ruin_prob_50dd']:.1f}%")

    print(EXECUTION_STRESS_DISCLAIMER)
    print(UNIVERSE_BIAS_DISCLAIMER)


def _build_final_report_text(out: Dict, fold_results: List[Dict],
                             selected_config_info: Dict, oos_label: str) -> str:
    cfg = out.get("cfg")
    lines = [
        "MAHORAGA 6 — FINAL REPORT",
        "=" * 80,
        UNIVERSE_BIAS_DISCLAIMER,
        "",
        f"OOS type: {oos_label}",
        "",
        "FULL PERIOD SUMMARY",
        "-" * 40,
    ]
    for k, v in _fmt(out["full"]).items():
        lines.append(f"  {k}: {v}")

    lines += ["", "OOS SUMMARY", "-" * 40]
    for k, v in _fmt(out["oos"]).items():
        lines.append(f"  {k}: {v}")

    # Apples-to-apples OOS baselines
    lines += ["", "OOS BASELINES", "-" * 40]
    for key in ["qqq_oos", "eqw_oos", "mom_oos"]:
        if key not in out or not isinstance(out[key], dict):
            continue
        lines.append(f"  [{key}]")
        for k, v in _fmt(out[key]).items():
            lines.append(f"    {k}: {v}")

    # Correlation Shield summary
    if cfg is not None:
        lines += ["", "CORRELATION SHIELD (v6)", "-" * 40]
        lines.append(f"  enabled: {bool(getattr(cfg, 'enable_corr_shield', False))}")
        lines.append(f"  corr_window: {getattr(cfg, 'corr_window', '—')}")
        lines.append(f"  rho_in / rho_out: {getattr(cfg, 'corr_rho_in', '—')} / {getattr(cfg, 'corr_rho_out', '—')}")
        lines.append(f"  gamma: {getattr(cfg, 'corr_gamma', '—')}")
        lines.append(f"  hard_kill: {bool(getattr(cfg, 'corr_hard_kill', False))}  hard_rho: {getattr(cfg, 'corr_hard_rho', '—')}")
        lines.append(f"  vix_confirm: {bool(getattr(cfg, 'corr_use_vix_confirm', False))}  vix_thr: {getattr(cfg, 'corr_vix_thr', '—')}")

    lines += ["", "SELECTED CONFIGURATION", "-" * 40,
              f"  Source:       {selected_config_info['source']}",
              f"  combo_params: {selected_config_info['combo_params']}",
              f"  val_score:    {selected_config_info['val_score']}",
              f"  p_value:      {selected_config_info['val_p_value']}",
              f"  q_value:      {selected_config_info['val_q_value']}",
              f"  sig@5%FDR:    {selected_config_info['val_sig_5pct']}",
              f"  stat_label:   {selected_config_info['val_stat_label']}"]

    if selected_config_info.get("val_sig_5pct") is False:
        lines.append("  [NOTE] Does not pass BHY@5% — interpret with caution.")

    lines += ["", EXECUTION_STRESS_DISCLAIMER, "", UNIVERSE_BIAS_DISCLAIMER]
    return "\n".join([str(x) for x in lines])




def save_outputs(
    out: Dict,
    fold_results: List[Dict],
    ic_df: pd.DataFrame,
    rob: Dict,
    all_sweeps: pd.DataFrame,
    selected_config_info: Dict,
    oos_label: str,
    cfg: Mahoraga5Config,
    universe_schedule: Optional[pd.DataFrame] = None,
    universe_snapshots: Optional[List[pd.DataFrame]] = None,
    universe_diagnostics: Optional[pd.DataFrame] = None,
    data_quality_report: Optional[pd.DataFrame] = None,
    asset_registry: Optional[pd.DataFrame] = None,
):
    d = cfg.outputs_dir
    _ensure_dir(d)
    def _df(rows):
        return pd.DataFrame([{k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()} for r in rows])

    _df([out["full"], out["qqq_full"], out["eqw_full"], out["mom_full"]]).to_csv(f"{d}/comparison_full.csv", index=False)
    _df([out["oos"], out["qqq_oos"], out.get("eqw_oos", {}), out.get("mom_oos", {})]).to_csv(f"{d}/comparison_oos.csv", index=False)
    pd.DataFrame(fold_results).to_csv(f"{d}/walk_forward_folds.csv", index=False)
    pd.DataFrame(fold_results).to_csv(f"{d}/fold_diagnostics.csv", index=False)

    meta = {
        "oos_label": oos_label,
        "n_folds": len(fold_results),
        "oos_start": fold_results[0]["test"].split("→")[0] if fold_results else "—",
        "oos_end": fold_results[-1]["test"].split("→")[1] if fold_results else "—",
        "any_partial_fold": any(f.get("is_partial") for f in fold_results),
        "total_oos_days": sum(f.get("actual_test_days", 0) for f in fold_results),
        "sweep_grid_combos": len(list(iproduct(*[SWEEP_GRID[k] for k in SWEEP_GRID]))),
        "parallel_sweep": cfg.parallel_sweep,
        "universe_mode": "canonical" if cfg.use_canonical_universe else "static_expost",
        "data_quality_layer": cfg.enable_data_quality_layer,
        "ml_regime_gate": bool(getattr(cfg, "enable_ml_regime_gate", False)),
    }
    pd.DataFrame([meta]).to_csv(f"{d}/walk_forward_meta.csv", index=False)

    support = {
        **selected_config_info.get("combo_params", {}),
        "source": selected_config_info["source"],
        "val_score": selected_config_info["val_score"],
        "val_p_value": selected_config_info["val_p_value"],
        "val_q_value": selected_config_info["val_q_value"],
        "val_sig_5pct": selected_config_info["val_sig_5pct"],
        "val_stat_label": selected_config_info["val_stat_label"],
        "val_sharpe": selected_config_info.get("val_sharpe", np.nan),
    }
    pd.DataFrame([support]).to_csv(f"{d}/selected_config_support.csv", index=False)

    with open(f"{d}/final_report.txt", "w", encoding="utf-8") as f:
        f.write(_build_final_report_text(out, fold_results, selected_config_info, oos_label))

    all_sweeps.to_csv(f"{d}/walk_forward_sweeps.csv", index=False)
    out["stress"].to_csv(f"{d}/stress_full.csv", index=False)
    out["regime_full"].to_csv(f"{d}/regime_full.csv", index=False)
    out["regime_oos"].to_csv(f"{d}/regime_oos.csv", index=False)
    ic_df.to_csv(f"{d}/rolling_ic_multi.csv")

    # Correlation Shield diagnostics (Mahoraga 6)
    if out.get("res") is not None and "corr_scale" in out["res"]:
        df_corr = pd.DataFrame({
            "corr_rho":  to_s(out["res"].get("corr_rho")).reindex(out["res"]["equity"].index),
            "corr_scale": to_s(out["res"].get("corr_scale")).reindex(out["res"]["equity"].index),
            "corr_state": to_s(out["res"].get("corr_state")).reindex(out["res"]["equity"].index),
        })
        df_corr.to_csv(f"{d}/corr_shield_diagnostics.csv, corr_shield_ablation.csv")
    if rob.get("exec_stress") is not None:
        rob["exec_stress"].to_csv(f"{d}/execution_sensitivity_stress.csv", index=False)
    if rob.get("alt_univ") is not None:
        rob["alt_univ"].to_csv(f"{d}/alt_universes.csv", index=False)
    if rob.get("sensitivity") is not None:
        rob["sensitivity"].to_csv(f"{d}/local_sensitivity.csv", index=False)
    if rob.get("stop_ablation") is not None:
        rob["stop_ablation"].to_csv(f"{d}/stop_ablation.csv", index=False)
    if rob.get("ic_ablation") is not None:
        rob["ic_ablation"].to_csv(f"{d}/ic_ablation.csv, corr_shield_diagnostics.csv, corr_shield_ablation.csv", index=False)
    if rob.get("corr_ablation") is not None:
        rob["corr_ablation"].to_csv(f"{d}/corr_shield_ablation.csv", index=False)
    pd.DataFrame([out["alpha_full"], out["alpha_oos"], out["alpha_cond"]]).to_csv(f"{d}/alpha_nw.csv", index=False)
    pd.DataFrame([out["sr_ci_full"], out["sr_ci_oos"]]).to_csv(f"{d}/sharpe_ci.csv", index=False)
    if out.get("ff_full"):
        pd.DataFrame([out["ff_full"], out.get("ff_oos", {})]).to_csv(f"{d}/ff_attribution.csv", index=False)

    if universe_schedule is not None:
        universe_schedule.to_csv(f"{d}/universe_schedule.csv", index=False)
    if universe_snapshots:
        snap_dir = os.path.join(d, "universe_snapshots")
        _ensure_dir(snap_dir)
        for snap in universe_snapshots:
            if snap is None or snap.empty:
                continue
            rd = str(snap["recon_date"].iloc[0]).replace("-", "")
            snap.to_csv(f"{snap_dir}/snapshot_{rd}.csv", index=False)
    if universe_diagnostics is not None and not universe_diagnostics.empty:
        universe_diagnostics.to_csv(f"{d}/universe_diagnostics.csv", index=False)
    if data_quality_report is not None and not data_quality_report.empty:
        data_quality_report.to_csv(f"{d}/data_quality_report.csv", index=False)
    if asset_registry is not None and not asset_registry.empty:
        asset_registry.to_csv(f"{d}/asset_registry.csv", index=False)

    universe_cfg = UniverseConfig()
    methodology = {
        "version": "6.0",
        "universe_mode": "canonical_engine" if cfg.use_canonical_universe else "static_expost",
        "static_universe": list(cfg.universe_static),
        "universe_note": "static universe is ex-post selected mega-cap tech; not survivor-bias-free",
        "canonical_ranking": "LiqSizeProxy = mean(price × volume, 30d) — NOT float-adjusted market cap (FFMC)",
        "canonical_ranking_note": "FFMC/GICS-IT require CRSP/Compustat PIT data, which is not integrated",
        "canonical_recon": universe_cfg.recon_freq,
        "canonical_target_n": universe_cfg.target_size,
        "auto_entry_rank": universe_cfg.auto_entry_rank,
        "retention_rank": universe_cfg.retention_rank,
        "buffer_rank": universe_cfg.buffer_rank,
        "min_seasoning_days": universe_cfg.min_seasoning_days,
        "min_free_float_proxy": "volume_continuity (fraction of days with non-zero vol) — NOT actual float data",
        "min_addv_usd": universe_cfg.min_addv_usd,
        "data_source": "yfinance (simulation proxy — not CRSP/Compustat PIT)",
        "survivorship_bias": "PRESENT — cannot be eliminated without CRSP PIT",
        "data_quality_layer": cfg.enable_data_quality_layer,
        "ml_regime_gate": bool(getattr(cfg, "enable_ml_regime_gate", False)),
        "disclaimer": UNIVERSE_BIAS_DISCLAIMER.strip(),
    }
    with open(f"{d}/universe_methodology.json", "w", encoding="utf-8") as f:
        json.dump(methodology, f, indent=2)

    print(f"\n  [outputs → ./{d}/]")
    print("    comparison_full.csv, comparison_oos.csv, walk_forward_folds.csv, fold_diagnostics.csv")
    print("    walk_forward_meta.csv, selected_config_support.csv, final_report.txt, walk_forward_sweeps.csv")
    print("    universe_schedule.csv, universe_diagnostics.csv, data_quality_report.csv, asset_registry.csv")


def make_plots(out, oos_r, oos_eq, fold_results, ic_df, decomp, rob, cfg,
               oos_label: str = "OOS_continuous"):
    p = cfg.plots_dir
    _ensure_dir(p)
    res = out["res"]
    eqw = out["eqw"]
    mom = out["mom"]

    plot_equity({cfg.label: res["equity"], "QQQ": res["bench"]["QQQ_eq"], "EQW": eqw["eq"], "MOM_12_1": mom["eq"]},
                "Full Period Equity — Mahoraga 6", f"{p}/01_equity_full.png")

    qqq_oos_r = res["bench"]["QQQ_r"].reindex(oos_r.index).fillna(0.0)
    qqq_oos_eq = cfg.capital_initial * (1.0 + qqq_oos_r).cumprod()
    plot_equity({cfg.label: oos_eq, "QQQ (OOS)": qqq_oos_eq},
                f"Walk-Forward {oos_label} — Mahoraga 6", f"{p}/02_equity_oos.png")

    plot_drawdown({cfg.label: res["equity"], "QQQ": res["bench"]["QQQ_eq"], "MOM_12_1": mom["eq"]},
                  "Drawdown", f"{p}/03_drawdown.png")
    plot_wf_oos(oos_eq, qqq_oos_eq, fold_results,
                f"Walk-Forward OOS — {oos_label}", f"{p}/04_walkforward.png", oos_label)

    # OOS equity with baselines on the same stitched calendar
    eqw_oos_r  = eqw["r"].reindex(oos_r.index).fillna(0.0)
    mom_oos_r  = mom["r"].reindex(oos_r.index).fillna(0.0)
    eqw_oos_eq = cfg.capital_initial * (1.0 + eqw_oos_r).cumprod()
    mom_oos_eq = cfg.capital_initial * (1.0 + mom_oos_r).cumprod()
    plot_equity({cfg.label: oos_eq, "QQQ": qqq_oos_eq, "EQW": eqw_oos_eq, "MOM_12_1": mom_oos_eq},
                f"OOS Equity vs Baselines — {oos_label}", f"{p}/04b_oos_vs_baselines.png")
    plot_risk_overlays(res, "Risk Overlays", f"{p}/05_risk_overlays.png")
    plot_weights_heatmap(res["weights_scaled"], "Portfolio Weights (monthly avg)", f"{p}/06_weights.png")
    plot_ic_multi_horizon(ic_df, "Rolling IC — 1d/5d/21d", f"{p}/07_ic_multi.png")
    plot_regime_bars(out["regime_full"], "Regime Analysis — Full", f"{p}/08_regime_full.png")
    plot_regime_bars(out["regime_oos"], f"Regime Analysis — {oos_label}", f"{p}/09_regime_oos.png")
    if decomp:
        plot_signal_decomp(decomp, res["bench"]["QQQ_r"], cfg, "Signal Decomposition", f"{p}/10_decomp.png")
    if rob.get("sensitivity") is not None:
        plot_sharpe_surface(rob["sensitivity"], "vol_target_ann", "weight_cap",
                            "Sharpe Surface — vol_target × weight_cap", f"{p}/11_sensitivity.png")


def run_mahoraga6(make_plots_flag: bool = True, run_robustness: bool = True) -> Dict:
    print("=" * 80)
    print("  MAHORAGA 6 — Research Edition")
    print("=" * 80)
    print(UNIVERSE_BIAS_DISCLAIMER)

    cfg = Mahoraga6Config()
    costs = CostsConfig()
    ucfg = UniverseConfig()
    _ensure_dir(cfg.cache_dir)
    _ensure_dir(cfg.plots_dir)
    _ensure_dir(cfg.outputs_dir)

    print("\n[1] Downloading data …")
    equity_tickers = sorted(set(list(cfg.universe_static) + [t for u in ALTERNATE_UNIVERSES.values() for t in u]))
    bench_tickers = [cfg.bench_qqq, cfg.bench_spy, cfg.bench_vix]
    all_tickers = sorted(set(equity_tickers + bench_tickers))
    ohlcv = download_ohlcv(all_tickers, cfg.data_start, cfg.data_end, cfg.cache_dir)

    print("\n[1b] Fama-French factors …")
    ff = load_ff_factors(cfg.cache_dir)

    print("\n[1c] Data hygiene & asset registry …")
    asset_registry = build_asset_registry(equity_tickers, cfg, bench_tickers)
    data_quality_report = compute_data_quality_report(ohlcv, equity_tickers, cfg)
    clean_equity = filter_equity_candidates([t for t in equity_tickers if t in ohlcv["close"].columns],
                                            asset_registry, data_quality_report, cfg)
    print(f"  [dq] raw equity candidates:   {len([t for t in equity_tickers if t in ohlcv['close'].columns])}")
    print(f"  [dq] clean equity candidates: {len(clean_equity)}")
    if len(clean_equity) < cfg.min_universe_names:
        print("  [dq] WARNING: too few clean names after quality filters; reverting to raw equity candidates")
        clean_equity = [t for t in equity_tickers if t in ohlcv["close"].columns]

    universe_schedule = None
    universe_snapshots = None
    universe_diagnostics = pd.DataFrame()
    if cfg.use_canonical_universe:
        print("\n[1d] Canonical universe engine (quarterly LiqSizeProxy reconstitution) …")
        print(f"  [universe] tickers_in_scope = {len(clean_equity)} clean equity candidates (benchmarks excluded)")
        universe_schedule, universe_snapshots = build_canonical_universe_schedule(
            ohlcv["close"], ohlcv["volume"], ucfg, clean_equity,
            cfg.data_start, cfg.data_end,
            registry_df=asset_registry, quality_df=data_quality_report,
        )
        universe_diagnostics = build_universe_diagnostics(universe_schedule)
        n_recon = len(universe_schedule)
        print(f"  [universe] {n_recon} reconstitution dates built")
        if n_recon > 0:
            first_u = json.loads(universe_schedule.iloc[0]["members"])
            last_u = json.loads(universe_schedule.iloc[-1]["members"])
            print(f"  [universe] first recon ({universe_schedule.iloc[0]['recon_date']}): {first_u}")
            print(f"  [universe] last  recon ({universe_schedule.iloc[-1]['recon_date']}): {last_u}")
        print("  [universe] Schedule governs backtest point-in-time at each rebalance.")
    else:
        print("\n[1d] *** WARNING: use_canonical_universe=False ***")
        print("     Running on static ex-post universe. Survivorship bias is PRESENT.")
        print(f"     Static universe: {list(cfg.universe_static)}")

    print("\n[2] Walk-forward …")
    oos_r, oos_eq, oos_exp, oos_to, fold_results, all_sweeps, oos_label, selected_config_info = \
        run_walk_forward(ohlcv, cfg, costs, universe_schedule=universe_schedule)

    oos_summary = summarize(oos_r, oos_eq, oos_exp, oos_to, cfg, oos_label)
    print(f"\n  OOS type:   {oos_label}")
    print(f"  OOS Sharpe={oos_summary['Sharpe']:.3f}  CAGR={oos_summary['CAGR']*100:.1f}%  MaxDD={oos_summary['MaxDD']*100:.1f}%")

    print("\n[3] Applying last-fold winning combo as final config …")
    last_train_end = cfg.wf_folds[-1][0]
    cfg_final = deepcopy(cfg)
    qqq_full = to_s(ohlcv["close"][cfg.bench_qqq].ffill())
    dd_thr, vol_thr = calibrate_crisis_thresholds(qqq_full, cfg.wf_train_start, last_train_end, cfg_final)
    cfg_final.crisis_dd_thr = dd_thr
    cfg_final.crisis_vol_zscore_thr = vol_thr
    for k, v in selected_config_info["combo_params"].items():
        setattr(cfg_final, k, v)

    final_train_tickers = get_training_universe(last_train_end, universe_schedule,
                                                cfg.universe_static, list(ohlcv["close"].columns))
    close_univ = ohlcv["close"][final_train_tickers]
    wt, wm, wr = fit_ic_weights(close_univ, qqq_full.loc[cfg.wf_train_start:last_train_end], cfg_final,
                                cfg.wf_train_start, last_train_end)
    cfg_final.w_trend = wt
    cfg_final.w_mom = wm
    cfg_final.w_rel = wr

    print(f"\n  Final config: {selected_config_info['combo_params']}")
    print(f"  Statistical support (last fold): p={selected_config_info['val_p_value']:.6f}  q={selected_config_info['val_q_value']:.6f}  sig@5%={selected_config_info['val_sig_5pct']}  label={selected_config_info['val_stat_label']}")

    print("\n[4] Full evaluation …")
    out = final_evaluation(ohlcv, cfg_final, costs, ff, oos_r, oos_eq, oos_label,
                           universe_schedule=universe_schedule)

    print("\n[5] Rolling IC (1d/5d/21d) …")
    ic_df = rolling_ic_multi_horizon(close_univ, qqq_full, cfg_final, window=63)

    print("\n[6] Signal decomposition …")
    decomp = baseline_signal_decomp(ohlcv, cfg_final, costs,
                                    universe_schedule=universe_schedule)

    rob = {}
    if run_robustness:
        print("\n[7a] Execution_Sensitivity_Stress …")
        print(EXECUTION_STRESS_DISCLAIMER)
        rob["exec_stress"] = execution_sensitivity_stress(ohlcv, cfg_final, costs,
                                                          universe_schedule=universe_schedule)
        print("\n[7b] Alternate universes …")
        rob["alt_univ"] = alternate_universe_stress(ohlcv, cfg_final, costs)
        print("\n[7c] Local parameter sensitivity …")
        rob["sensitivity"] = local_sensitivity(ohlcv, cfg_final, costs,
                                               universe_schedule=universe_schedule)
        print("\n[7d] Stop ablation …")
        rob["stop_ablation"] = stop_keep_cash_ablation(ohlcv, cfg_final, costs,
                                                        universe_schedule=universe_schedule)
        print("\n[7e] IC weight ablation …")
        rob["ic_ablation"] = ic_weight_ablation(ohlcv, cfg_final, costs,
                                                 universe_schedule=universe_schedule)

    print("\n[7f] Correlation Shield ablation …")
    rob["corr_ablation"] = corr_shield_ablation(ohlcv, cfg_final, costs,
                                                   universe_schedule=universe_schedule)

    print_results(out, fold_results, ff, selected_config_info)
    save_outputs(out, fold_results, ic_df, rob, all_sweeps, selected_config_info,
                 oos_label, cfg_final, universe_schedule=universe_schedule,
                 universe_snapshots=universe_snapshots,
                 universe_diagnostics=universe_diagnostics,
                 data_quality_report=data_quality_report,
                 asset_registry=asset_registry)

    if make_plots_flag:
        print("\n[8] Generating plots …")
        make_plots(out, oos_r, oos_eq, fold_results, ic_df, decomp, rob, cfg_final, oos_label)

    return {
        "cfg": cfg_final,
        "out": out,
        "oos_r": oos_r,
        "oos_eq": oos_eq,
        "oos_exp": oos_exp,
        "oos_to": oos_to,
        "oos_label": oos_label,
        "fold_results": fold_results,
        "ic_df": ic_df,
        "decomp": decomp,
        "rob": rob,
        "selected_config_info": selected_config_info,
        "universe_schedule": universe_schedule,
        "universe_diagnostics": universe_diagnostics,
        "data_quality_report": data_quality_report,
        "asset_registry": asset_registry,
    }


# ═══════════════════════════════════════════════════════════════════════════════

# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_mahoraga6(make_plots_flag=True, run_robustness=True)
