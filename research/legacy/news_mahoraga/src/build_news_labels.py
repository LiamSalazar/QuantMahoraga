
from __future__ import annotations

"""
build_news_labels.py
====================

Genera etiquetas semanales para la rama de noticias de Mahoraga.

Idea de V1:
- Trabaja sobre weekly_news_rollup.parquet.
- Construye un contexto semanal de mercado con QQQ, ^VIX y el universo tech.
- Define panic_target y opportunity_target con lógica alineada a Mahoraga 7C.3,
  pero en modo proxy de mercado para no depender todavía del motor quant completo.
- También genera dos auxiliares para Keras:
    clarity_target
    breadth_target

Importante:
- Si se pasa --quant-ref-path, el script intenta usar columnas del sistema quant
  para sobreescribir proxies cuando existan (base_total_scale, stress_intensity,
  recovery_intensity, etc.).
- Si no se pasa, usa contexto proxy de mercado.
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "AVGO", "ASML", "TSM", "ADBE", "NFLX", "AMD"
]


@dataclass
class LabelsManifest:
    weekly_rows_in: int
    weekly_rows_out: int
    min_week_end: Optional[str]
    max_week_end: Optional[str]
    horizon_weeks: int
    dd_penalty: float
    fragility_quantile: float
    recovery_return_quantile: float
    min_history_weeks: int
    quant_ref_used: bool
    panic_positive_rate: Optional[float]
    opportunity_positive_rate: Optional[float]
    clarity_positive_rate: Optional[float]
    breadth_positive_rate: Optional[float]
    output_labels_path: str


def normalize_key(s: str) -> str:
    return "".join(ch for ch in str(s).lower().strip() if ch.isalnum())


def choose_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    normalized = {normalize_key(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_key(cand)
        if key in normalized:
            return normalized[key]
    return None


def read_any_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return pd.DataFrame(obj if isinstance(obj, list) else [obj])
    raise ValueError(f"Formato no soportado: {path}")


def to_weekly_index(series: pd.Series, weekly_idx: pd.DatetimeIndex) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    weekly_idx = pd.DatetimeIndex(weekly_idx).tz_localize(None)
    return s.reindex(weekly_idx).ffill()


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=max(5, window // 2)).mean()
    sd = s.rolling(window, min_periods=max(5, window // 2)).std(ddof=0).replace(0, np.nan)
    return ((s - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def expanding_past_quantile(s: pd.Series, q: float, min_history: int) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=vals.index, dtype=float)
    arr = vals.to_numpy(dtype=float)
    for i in range(len(arr)):
        hist = arr[:i]
        hist = hist[np.isfinite(hist)]
        if hist.size >= min_history:
            out.iloc[i] = float(np.nanquantile(hist, q))
    return out


def future_window_return(
    returns_daily: pd.Series,
    weekly_idx: pd.DatetimeIndex,
    horizon_weeks: int,
) -> pd.Series:
    idx = returns_daily.index
    out = pd.Series(np.nan, index=weekly_idx, dtype=float)
    step = max(1, horizon_weeks * 5)
    if len(idx) == 0:
        return out
    for dt in weekly_idx:
        pos = idx.searchsorted(dt)
        if pos >= len(idx):
            continue
        end_i = min(len(idx) - 1, pos + step)
        if end_i <= pos:
            continue
        r_seg = returns_daily.iloc[pos + 1:end_i + 1]
        if len(r_seg) == 0:
            continue
        out.loc[dt] = float((1.0 + r_seg).prod() - 1.0)
    return out


def future_window_utility(
    returns_daily: pd.Series,
    equity_daily: pd.Series,
    weekly_idx: pd.DatetimeIndex,
    horizon_weeks: int,
    dd_penalty: float,
) -> pd.Series:
    idx = returns_daily.index
    out = pd.Series(np.nan, index=weekly_idx, dtype=float)
    step = max(1, horizon_weeks * 5)
    if len(idx) == 0:
        return out
    for dt in weekly_idx:
        pos = idx.searchsorted(dt)
        if pos >= len(idx):
            continue
        end_i = min(len(idx) - 1, pos + step)
        if end_i <= pos:
            continue
        r_seg = returns_daily.iloc[pos + 1:end_i + 1]
        eq_seg = equity_daily.iloc[pos:end_i + 1]
        if len(r_seg) == 0:
            continue
        ret = float((1.0 + r_seg).prod() - 1.0)
        if len(eq_seg) <= 1:
            dd = 0.0
        else:
            dd = float((eq_seg / eq_seg.cummax() - 1.0).min())
        out.loc[dt] = ret - dd_penalty * abs(dd)
    return out


def _extract_close_from_yf_download(obj: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if obj.empty:
        return pd.DataFrame()
    if isinstance(obj.columns, pd.MultiIndex):
        if "Close" in obj.columns.get_level_values(0):
            close = obj["Close"].copy()
        elif "Adj Close" in obj.columns.get_level_values(0):
            close = obj["Adj Close"].copy()
        else:
            raise ValueError("No se encontró Close/Adj Close en descarga de yfinance.")
    else:
        # caso de un solo ticker
        if "Close" in obj.columns:
            close = obj[["Close"]].copy()
        elif "Adj Close" in obj.columns:
            close = obj[["Adj Close"]].copy()
        else:
            raise ValueError("No se encontró Close/Adj Close en descarga de yfinance.")
        if len(tickers) == 1:
            close.columns = tickers
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def build_market_proxy_context(
    weekly_df: pd.DataFrame,
    universe: list[str],
    history_pad_days: int = 180,
    qqq_ticker: str = "QQQ",
    vix_ticker: str = "^VIX",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if yf is None:
        raise ImportError("No se encontró yfinance. Instale con: pip install yfinance")

    wk = weekly_df.copy()
    wk["week_end"] = pd.to_datetime(wk["week_end"]).dt.tz_localize(None)
    weekly_idx = pd.DatetimeIndex(wk["week_end"]).sort_values()

    start = (weekly_idx.min() - pd.Timedelta(days=history_pad_days)).date().isoformat()
    end = (weekly_idx.max() + pd.Timedelta(days=35)).date().isoformat()

    tickers = [qqq_ticker, vix_ticker] + list(universe)
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    close = _extract_close_from_yf_download(raw, tickers=tickers).sort_index()

    missing = [tk for tk in [qqq_ticker, vix_ticker] if tk not in close.columns]
    if missing:
        raise ValueError(f"Faltan benchmarks en descarga: {missing}")

    qqq_close = close[qqq_ticker].dropna()
    vix_close = close[vix_ticker].dropna()
    universe_cols = [c for c in universe if c in close.columns]
    if not universe_cols:
        raise ValueError("No se descargó ninguna serie del universo tech.")

    qqq_r = qqq_close.pct_change().fillna(0.0)
    qqq_eq = (1.0 + qqq_r).cumprod()
    qqq_ret_5d = qqq_close.pct_change(5)
    qqq_ret_21d = qqq_close.pct_change(21)
    qqq_vol_21 = qqq_r.rolling(21, min_periods=10).std(ddof=0) * np.sqrt(252)
    qqq_drawdown = (qqq_eq / qqq_eq.cummax() - 1.0).fillna(0.0)

    universe_close = close[universe_cols].copy().ffill()
    breadth_63d = universe_close.pct_change(63).gt(0).mean(axis=1)

    vix_z_63 = rolling_zscore(vix_close, 63)
    qqq_ret21_z = rolling_zscore(qqq_ret_21d.fillna(0.0), 63)
    breadth_z_63 = rolling_zscore(breadth_63d.fillna(0.0), 63)
    drawdown_stress = rolling_zscore((-qqq_drawdown).fillna(0.0), 63)

    stress_intensity = (
        np.maximum(vix_z_63, 0.0)
        + np.maximum(-breadth_z_63, 0.0)
        + np.maximum(drawdown_stress, 0.0)
    ) / 3.0

    recovery_intensity = (
        np.maximum(qqq_ret21_z, 0.0)
        + np.maximum(breadth_z_63, 0.0)
        + np.maximum(-vix_z_63, 0.0)
    ) / 3.0

    intensity_spread = (recovery_intensity - stress_intensity).astype(float)
    base_total_scale = (1.0 - np.clip(stress_intensity / 3.0, 0.0, 1.0)).astype(float)
    crisis_state_weekly = ((vix_close >= 24.0) | (qqq_drawdown <= -0.10)).astype(float)

    out = pd.DataFrame(index=weekly_idx)
    out["qqq_ret_5d"] = to_weekly_index(qqq_ret_5d, weekly_idx)
    out["qqq_ret_21d"] = to_weekly_index(qqq_ret_21d, weekly_idx)
    out["qqq_vol_21"] = to_weekly_index(qqq_vol_21, weekly_idx)
    out["qqq_drawdown"] = to_weekly_index(qqq_drawdown, weekly_idx)
    out["vix_level"] = to_weekly_index(vix_close, weekly_idx)
    out["vix_z_63"] = to_weekly_index(vix_z_63, weekly_idx)
    out["breadth_63d"] = to_weekly_index(breadth_63d, weekly_idx)
    out["stress_intensity"] = to_weekly_index(stress_intensity, weekly_idx)
    out["recovery_intensity"] = to_weekly_index(recovery_intensity, weekly_idx)
    out["intensity_spread"] = to_weekly_index(intensity_spread, weekly_idx)
    out["base_total_scale"] = to_weekly_index(base_total_scale, weekly_idx)
    out["crisis_state_weekly"] = to_weekly_index(crisis_state_weekly, weekly_idx)

    return out.reset_index(names="week_end"), qqq_r, qqq_eq


def merge_optional_quant_ref(
    weekly_df: pd.DataFrame,
    quant_ref_path: Optional[Path],
) -> tuple[pd.DataFrame, bool]:
    if quant_ref_path is None:
        return weekly_df, False

    ref = read_any_table(quant_ref_path)
    week_col = choose_existing_col(ref, ["week_end", "date", "dt", "decision_date"])
    if week_col is None:
        raise ValueError("No encontré columna de fecha semanal en quant_ref.")
    ref = ref.copy()
    ref["week_end"] = pd.to_datetime(ref[week_col]).dt.tz_localize(None)

    candidates = {
        "stress_intensity": ["stress_intensity"],
        "recovery_intensity": ["recovery_intensity"],
        "intensity_spread": ["intensity_spread"],
        "breadth_63d": ["breadth_63d"],
        "qqq_ret_21d": ["qqq_ret_21d"],
        "base_total_scale": ["base_total_scale", "total_scale_target", "base_scale"],
        "crisis_state_weekly": ["crisis_state_weekly", "crisis_state"],
    }

    keep = ["week_end"]
    rename = {}
    for target, names in candidates.items():
        col = choose_existing_col(ref, names)
        if col is not None:
            keep.append(col)
            rename[col] = target

    ref = ref[keep].rename(columns=rename).drop_duplicates(subset=["week_end"], keep="last")
    merged = weekly_df.merge(ref, on="week_end", how="left", suffixes=("", "_quant"))

    # si viene de quant, sobreescribe proxy
    for c in candidates.keys():
        qcol = f"{c}_quant"
        if qcol in merged.columns:
            merged[c] = merged[qcol].combine_first(merged[c])
            merged = merged.drop(columns=[qcol])

    return merged, True


def build_auxiliary_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    topic_active_count = (
        (pd.to_numeric(out.get("macro_topic_count", 0), errors="coerce").fillna(0) > 0).astype(int)
        + (pd.to_numeric(out.get("tech_topic_count", 0), errors="coerce").fillna(0) > 0).astype(int)
        + (pd.to_numeric(out.get("ticker_topic_count", 0), errors="coerce").fillna(0) > 0).astype(int)
        + (pd.to_numeric(out.get("panic_like_count", 0), errors="coerce").fillna(0) > 0).astype(int)
        + (pd.to_numeric(out.get("opportunity_like_count", 0), errors="coerce").fillna(0) > 0).astype(int)
    )

    panic_share = pd.to_numeric(out.get("panic_share", 0.0), errors="coerce").fillna(0.0)
    opp_share = pd.to_numeric(out.get("opportunity_share", 0.0), errors="coerce").fillna(0.0)
    dominance = (panic_share - opp_share).abs()
    contradiction = ((panic_share > 0.015) & (opp_share > 0.015)).astype(float)

    clarity_proxy = pd.to_numeric(out.get("avg_clarity_proxy", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    source_weight = pd.to_numeric(out.get("avg_source_weight", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    n_sources = pd.to_numeric(out.get("n_sources", 0.0), errors="coerce").fillna(0.0)
    source_breadth = np.clip(n_sources / 10.0, 0.0, 1.0)

    clarity_score = (
        0.45 * clarity_proxy
        + 0.25 * source_weight
        + 0.20 * np.clip(dominance * 10.0, 0.0, 1.0)
        + 0.10 * source_breadth
        - 0.20 * contradiction
    ).clip(0.0, 1.0)

    volume_z = pd.to_numeric(out.get("news_volume_z_8w", 0.0), errors="coerce").fillna(0.0)
    volume_boost = np.clip((volume_z + 1.0) / 3.0, 0.0, 1.0)
    ticker_cnt = pd.to_numeric(out.get("ticker_topic_count", 0.0), errors="coerce").fillna(0.0)
    breadth_score = (
        0.40 * volume_boost
        + 0.25 * np.clip(n_sources / 12.0, 0.0, 1.0)
        + 0.20 * np.clip(topic_active_count / 4.0, 0.0, 1.0)
        + 0.15 * np.clip(ticker_cnt / 8.0, 0.0, 1.0)
    ).clip(0.0, 1.0)

    out["clarity_score"] = clarity_score
    out["breadth_score"] = breadth_score
    out["clarity_target"] = (clarity_score >= 0.55).astype(float)
    out["breadth_target"] = (breadth_score >= 0.55).astype(float)
    return out


def build_primary_targets(
    df: pd.DataFrame,
    qqq_r_daily: pd.Series,
    qqq_eq_daily: pd.Series,
    horizon_weeks: int,
    dd_penalty: float,
    fragility_quantile: float,
    recovery_return_quantile: float,
    recovery_low_scale_quantile: float,
    recovery_spread_quantile: float,
    recovery_breadth_quantile: float,
    recovery_relaxed_scale_quantile: float,
    recovery_relaxed_qqq_buffer_quantile: float,
    min_history_weeks: int,
) -> pd.DataFrame:
    out = df.copy().sort_values("week_end").reset_index(drop=True)
    out["week_end"] = pd.to_datetime(out["week_end"]).dt.tz_localize(None)
    wk_idx = pd.DatetimeIndex(out["week_end"])

    qqq_r_daily = pd.to_numeric(qqq_r_daily, errors="coerce").fillna(0.0).copy()
    qqq_r_daily.index = pd.to_datetime(qqq_r_daily.index).tz_localize(None)
    qqq_eq_daily = pd.to_numeric(qqq_eq_daily, errors="coerce").copy()
    qqq_eq_daily.index = pd.to_datetime(qqq_eq_daily.index).tz_localize(None)

    qqq_future_ret = future_window_return(qqq_r_daily, wk_idx, horizon_weeks)
    future_utility = future_window_utility(qqq_r_daily, qqq_eq_daily, wk_idx, horizon_weeks, dd_penalty)

    out["qqq_future_ret"] = qqq_future_ret.to_numpy()
    out["future_utility"] = future_utility.to_numpy()

    utility_thr = expanding_past_quantile(future_utility, fragility_quantile, min_history_weeks)
    out["fragility_utility_threshold"] = utility_thr.to_numpy()

    # A partir de aquí trabajamos con las columnas ya insertadas en out para que
    # todas las operaciones compartan el mismo RangeIndex y no haya alineaciones
    # accidentales entre índices temporales y enteros.
    future_utility_s = pd.to_numeric(out["future_utility"], errors="coerce")
    utility_thr_s = pd.to_numeric(out["fragility_utility_threshold"], errors="coerce")

    panic_mask = (future_utility_s.notna() & utility_thr_s.notna()).to_numpy()
    out["panic_target"] = np.nan
    panic_values = (future_utility_s <= utility_thr_s).astype(float).to_numpy()
    out.loc[panic_mask, "panic_target"] = panic_values[panic_mask]

    base_scale = pd.to_numeric(out["base_total_scale"], errors="coerce")
    spread = pd.to_numeric(out["intensity_spread"], errors="coerce")
    breadth = pd.to_numeric(out["breadth_63d"], errors="coerce")
    qqq21 = pd.to_numeric(out["qqq_ret_21d"], errors="coerce")
    crisis = pd.to_numeric(out["crisis_state_weekly"], errors="coerce").fillna(0.0)
    stress_i = pd.to_numeric(out["stress_intensity"], errors="coerce")
    recovery_i = pd.to_numeric(out["recovery_intensity"], errors="coerce")

    ret_thr = expanding_past_quantile(qqq_future_ret, recovery_return_quantile, min_history_weeks)
    scale_thr = expanding_past_quantile(base_scale, recovery_low_scale_quantile, min_history_weeks)
    spread_thr = expanding_past_quantile(spread, recovery_spread_quantile, min_history_weeks)
    breadth_thr = expanding_past_quantile(breadth, recovery_breadth_quantile, min_history_weeks)
    qqq_buf = expanding_past_quantile(qqq21, recovery_relaxed_qqq_buffer_quantile, min_history_weeks)
    rel_scale_thr = expanding_past_quantile(base_scale, recovery_relaxed_scale_quantile, min_history_weeks)

    out["recovery_ret_threshold"] = ret_thr.to_numpy()
    out["recovery_scale_threshold"] = scale_thr.to_numpy()
    out["recovery_spread_threshold"] = spread_thr.to_numpy()
    out["recovery_breadth_threshold"] = breadth_thr.to_numpy()
    out["recovery_qqq_buffer_threshold"] = qqq_buf.to_numpy()
    out["recovery_relaxed_scale_threshold"] = rel_scale_thr.to_numpy()

    qqq_future_ret_s = pd.to_numeric(out["qqq_future_ret"], errors="coerce")
    ret_thr_s = pd.to_numeric(out["recovery_ret_threshold"], errors="coerce")
    scale_thr_s = pd.to_numeric(out["recovery_scale_threshold"], errors="coerce")
    spread_thr_s = pd.to_numeric(out["recovery_spread_threshold"], errors="coerce")
    breadth_thr_s = pd.to_numeric(out["recovery_breadth_threshold"], errors="coerce")
    qqq_buf_s = pd.to_numeric(out["recovery_qqq_buffer_threshold"], errors="coerce")
    rel_scale_thr_s = pd.to_numeric(out["recovery_relaxed_scale_threshold"], errors="coerce")

    cond_ret = qqq_future_ret_s >= ret_thr_s
    cond_scale = (base_scale <= scale_thr_s) | (crisis >= 1.0) | (base_scale <= rel_scale_thr_s)
    cond_spread = (spread >= spread_thr_s) | (recovery_i > stress_i)
    cond_breadth = (breadth >= breadth_thr_s) | (qqq21 >= qqq_buf_s)

    cond_count = cond_scale.astype(float).fillna(0.0) + cond_spread.astype(float).fillna(0.0) + cond_breadth.astype(float).fillna(0.0)
    opp_mask = (
        qqq_future_ret_s.notna()
        & ret_thr_s.notna()
        & scale_thr_s.notna()
        & spread_thr_s.notna()
        & breadth_thr_s.notna()
    ).to_numpy()
    out["opportunity_target"] = np.nan
    opp_values = (cond_ret & (cond_count >= 2)).astype(float).to_numpy()
    out.loc[opp_mask, "opportunity_target"] = opp_values[opp_mask]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye weekly_targets.parquet para Mahoraga news.")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Ruta base de news_mahoraga. Si se omite, se infiere desde src.")
    parser.add_argument("--weekly-path", type=str, default=None,
                        help="Ruta directa a weekly_news_rollup.parquet/csv.")
    parser.add_argument("--quant-ref-path", type=str, default=None,
                        help="Ruta opcional a features semanales del quant para sobreescribir proxies.")
    parser.add_argument("--horizon-weeks", type=int, default=2,
                        help="Horizonte futuro en semanas para labels. V1 default=2.")
    parser.add_argument("--fragility-quantile", type=float, default=0.20,
                        help="Cuantil para panic_target. Alineado a 7C.3 (0.15-0.20).")
    parser.add_argument("--dd-penalty", type=float, default=0.40,
                        help="Penalización de drawdown dentro de utility. Alineado a 7C.3.")
    parser.add_argument("--recovery-return-quantile", type=float, default=0.70,
                        help="Cuantil de retorno futuro para opportunity_target.")
    parser.add_argument("--recovery-low-scale-quantile", type=float, default=0.60)
    parser.add_argument("--recovery-spread-quantile", type=float, default=0.45)
    parser.add_argument("--recovery-breadth-quantile", type=float, default=0.50)
    parser.add_argument("--recovery-relaxed-scale-quantile", type=float, default=0.75)
    parser.add_argument("--recovery-relaxed-qqq-buffer-quantile", type=float, default=0.55)
    parser.add_argument("--min-history-weeks", type=int, default=8,
                        help="Semanas mínimas de historia para calcular umbrales expanding sin leakage.")
    parser.add_argument("--write-csv-too", action="store_true",
                        help="Si se activa, además de parquet exporta csv.")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    inferred_base = script_path.parent.parent
    base_dir = Path(args.base_dir).resolve() if args.base_dir else inferred_base

    weekly_dir = base_dir / "news_data" / "weekly"
    labels_dir = base_dir / "news_data" / "labels"
    manifests_dir = base_dir / "news_data" / "manifests"

    labels_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    weekly_path = Path(args.weekly_path).resolve() if args.weekly_path else (weekly_dir / "weekly_news_rollup.parquet")
    quant_ref_path = Path(args.quant_ref_path).resolve() if args.quant_ref_path else None

    print("=" * 80)
    print(" BUILD NEWS LABELS — MAHORAGA")
    print("=" * 80)
    print(f"[base]   {base_dir}")
    print(f"[weekly] {weekly_path}")
    if quant_ref_path:
        print(f"[quant]  {quant_ref_path}")
    print(f"[horizon_weeks] {args.horizon_weeks}")

    weekly_df = read_any_table(weekly_path)
    if weekly_df.empty:
        raise ValueError("weekly_news_rollup está vacío.")

    week_col = choose_existing_col(weekly_df, ["week_end"])
    if week_col is None:
        raise ValueError("No encontré week_end en weekly_news_rollup.")
    weekly_df = weekly_df.copy()
    weekly_df["week_end"] = pd.to_datetime(weekly_df[week_col]).dt.tz_localize(None)
    if week_col != "week_end":
        weekly_df = weekly_df.drop(columns=[week_col])

    rows_in = len(weekly_df)
    print(f"[load] semanas input: {rows_in:,}")

    market_ctx_df, qqq_r_daily, qqq_eq_daily = build_market_proxy_context(
        weekly_df=weekly_df,
        universe=DEFAULT_UNIVERSE,
    )
    merged = weekly_df.merge(market_ctx_df, on="week_end", how="left")
    merged, quant_used = merge_optional_quant_ref(merged, quant_ref_path)

    merged = build_primary_targets(
        df=merged,
        qqq_r_daily=qqq_r_daily,
        qqq_eq_daily=qqq_eq_daily,
        horizon_weeks=args.horizon_weeks,
        dd_penalty=args.dd_penalty,
        fragility_quantile=args.fragility_quantile,
        recovery_return_quantile=args.recovery_return_quantile,
        recovery_low_scale_quantile=args.recovery_low_scale_quantile,
        recovery_spread_quantile=args.recovery_spread_quantile,
        recovery_breadth_quantile=args.recovery_breadth_quantile,
        recovery_relaxed_scale_quantile=args.recovery_relaxed_scale_quantile,
        recovery_relaxed_qqq_buffer_quantile=args.recovery_relaxed_qqq_buffer_quantile,
        min_history_weeks=args.min_history_weeks,
    )
    merged = build_auxiliary_targets(merged)

    merged["label_ready_primary"] = merged["panic_target"].notna() & merged["opportunity_target"].notna()
    merged["label_ready_all"] = merged["label_ready_primary"] & merged["clarity_target"].notna() & merged["breadth_target"].notna()

    out_path = labels_dir / "weekly_targets.parquet"
    merged.to_parquet(out_path, index=False)
    if args.write_csv_too:
        merged.to_csv(labels_dir / "weekly_targets.csv", index=False, encoding="utf-8-sig")

    manifest = LabelsManifest(
        weekly_rows_in=rows_in,
        weekly_rows_out=len(merged),
        min_week_end=str(merged["week_end"].min()) if not merged.empty else None,
        max_week_end=str(merged["week_end"].max()) if not merged.empty else None,
        horizon_weeks=args.horizon_weeks,
        dd_penalty=args.dd_penalty,
        fragility_quantile=args.fragility_quantile,
        recovery_return_quantile=args.recovery_return_quantile,
        min_history_weeks=args.min_history_weeks,
        quant_ref_used=quant_used,
        panic_positive_rate=float(merged["panic_target"].dropna().mean()) if merged["panic_target"].notna().any() else None,
        opportunity_positive_rate=float(merged["opportunity_target"].dropna().mean()) if merged["opportunity_target"].notna().any() else None,
        clarity_positive_rate=float(merged["clarity_target"].dropna().mean()) if merged["clarity_target"].notna().any() else None,
        breadth_positive_rate=float(merged["breadth_target"].dropna().mean()) if merged["breadth_target"].notna().any() else None,
        output_labels_path=str(out_path),
    )

    manifest_path = manifests_dir / "labels_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, ensure_ascii=False, indent=2)

    print(f"[done] weekly_targets exportado: {out_path}")
    print(f"[done] manifest exportado:      {manifest_path}")


if __name__ == "__main__":
    main()
