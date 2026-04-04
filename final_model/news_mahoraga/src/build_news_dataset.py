from __future__ import annotations

"""
build_news_dataset.py
=====================

Construye el dataset base de noticias para la futura capa Keras de Mahoraga.

Qué hace:
1) Lee noticias crudas desde news_data/raw
2) Estandariza columnas
3) Limpia timestamps y texto
4) Elimina duplicados
5) Detecta relevancia básica (macro / tech / ticker / low)
6) Agrega por semana (W-FRI)
7) Exporta:
   - news_data/processed/news_clean.parquet
   - news_data/weekly/weekly_news_rollup.parquet
   - news_data/manifests/build_manifest.json

No crea labels todavía. Ese paso conviene hacerlo en un segundo script separado,
usando la lógica de Mahoraga 6.1 / 7C.3.
"""

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ============================================================================
# CONFIG
# ============================================================================

DEFAULT_TECH_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "AVGO", "ASML", "TSM", "ADBE", "NFLX", "AMD"
]

DEFAULT_SOURCE_WEIGHTS = {
    "reuters": 1.00,
    "bloomberg": 1.00,
    "financial times": 0.98,
    "wall street journal": 0.98,
    "wsj": 0.98,
    "cnbc": 0.92,
    "marketwatch": 0.88,
    "yahoo finance": 0.82,
    "seeking alpha": 0.76,
    "benzinga": 0.72,
    "investing.com": 0.74,
    "the information": 0.94,
    "techcrunch": 0.85,
    "the verge": 0.78,
    "unknown": 0.60,
}

MACRO_KEYWORDS = [
    "fed", "fomc", "interest rate", "rates", "inflation", "cpi", "ppi",
    "unemployment", "jobs report", "payrolls", "recession", "gdp", "treasury",
    "yield", "credit crisis", "banking crisis", "liquidity", "default",
    "tariff", "tariffs", "trade war", "sanctions", "war", "geopolitical",
    "oil shock", "debt ceiling", "fiscal", "monetary policy"
]

TECH_KEYWORDS = [
    "artificial intelligence", "ai", "semiconductor", "chip", "chips", "gpu",
    "cloud", "data center", "datacenter", "foundry", "server", "compute",
    "software", "saas", "enterprise software", "antitrust", "regulation",
    "export controls", "capacity expansion", "capex", "guidance", "earnings"
]

PANIC_KEYWORDS = [
    "collapse", "crash", "panic", "meltdown", "bankruptcy", "default",
    "deep recession", "mass layoffs", "layoffs", "contagion", "war",
    "invasion", "sanctions", "tariffs", "credit crunch", "bank run",
    "bubble burst", "burst", "crisis"
]

OPPORTUNITY_KEYWORDS = [
    "record demand", "strong guidance", "guidance raised", "beat expectations",
    "beats expectations", "surge", "breakthrough", "expansion", "approval",
    "rate cuts", "soft landing", "stimulus", "major contract", "capacity ramp",
    "demand acceleration", "orders jump", "revenue outlook raised"
]


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class BuildManifest:
    raw_files_found: int
    rows_loaded_raw: int
    rows_after_timestamp_filter: int
    rows_after_dedup: int
    rows_final_clean: int
    weeks_final: int
    min_timestamp_utc: Optional[str]
    max_timestamp_utc: Optional[str]
    output_clean_path: str
    output_weekly_path: str
    universe_tickers: list[str]


# ============================================================================
# HELPERS
# ============================================================================

def normalize_text(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower().strip())


def safe_to_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(v).strip().upper() for v in value if str(v).strip()]
    if isinstance(value, tuple):
        return [str(v).strip().upper() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return []
    # soporta strings tipo "AAPL, MSFT" o '["AAPL","MSFT"]'
    s = s.strip("[]")
    parts = re.split(r"[,\|;/]+", s)
    return [p.strip().strip('"').strip("'").upper() for p in parts if p.strip()]


def choose_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    normalized = {normalize_key(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_key(cand)
        if key in normalized:
            return normalized[key]
    return None


def read_any_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
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
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            # intenta varias estructuras comunes
            for key in ("data", "items", "articles", "results", "news"):
                if key in obj and isinstance(obj[key], list):
                    return pd.DataFrame(obj[key])
            return pd.DataFrame([obj])
    raise ValueError(f"Formato no soportado: {path.name}")


def source_weight(source: str) -> float:
    s = normalize_text(source).lower()
    if not s:
        return DEFAULT_SOURCE_WEIGHTS["unknown"]
    for k, v in DEFAULT_SOURCE_WEIGHTS.items():
        if k in s:
            return v
    return DEFAULT_SOURCE_WEIGHTS["unknown"]


def detect_topic(text: str) -> str:
    t = text.lower()
    macro_hits = sum(1 for k in MACRO_KEYWORDS if k in t)
    tech_hits = sum(1 for k in TECH_KEYWORDS if k in t)
    panic_hits = sum(1 for k in PANIC_KEYWORDS if k in t)
    opp_hits = sum(1 for k in OPPORTUNITY_KEYWORDS if k in t)

    scores = {
        "macro": macro_hits,
        "tech": tech_hits,
        "panic": panic_hits,
        "opportunity": opp_hits,
    }
    best_topic = max(scores, key=scores.get)
    return best_topic if scores[best_topic] > 0 else "general"


def infer_tickers(text: str, universe: list[str]) -> list[str]:
    found = []
    upper = f" {text.upper()} "
    for tk in universe:
        if re.search(rf"(?<![A-Z]){re.escape(tk)}(?![A-Z])", upper):
            found.append(tk)
    return sorted(set(found))


def compute_relevance_kind(topic: str, tickers: list[str], text: str) -> str:
    lower = text.lower()
    if topic in {"panic", "macro"}:
        return "macro"
    if tickers:
        return "ticker"
    if any(k in lower for k in TECH_KEYWORDS):
        return "tech"
    return "low"


def compute_event_strength(topic: str, relevance_kind: str, clarity_proxy: float) -> float:
    base = 0.0
    if topic == "panic":
        base += 1.0
    elif topic == "opportunity":
        base += 0.85
    elif topic == "macro":
        base += 0.65
    elif topic == "tech":
        base += 0.55
    else:
        base += 0.30

    if relevance_kind == "macro":
        base += 0.25
    elif relevance_kind == "ticker":
        base += 0.15
    elif relevance_kind == "tech":
        base += 0.10

    base += 0.20 * clarity_proxy
    return float(min(base, 1.50))


def hash_event_key(timestamp_utc: pd.Timestamp, source: str, headline: str) -> str:
    base = f"{timestamp_utc.date().isoformat()}|{normalize_text(source).lower()}|{normalize_text(headline).lower()}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def build_text_block(group: pd.DataFrame, max_articles: int = 30, max_chars: int = 20000) -> str:
    if group.empty:
        return ""
    g = group.sort_values(
        by=["event_strength", "source_weight", "timestamp_utc"],
        ascending=[False, False, True]
    ).head(max_articles)

    chunks = []
    total = 0
    for _, row in g.iterrows():
        part = " [SEP] ".join(
            p for p in [
                normalize_text(row.get("headline", "")),
                normalize_text(row.get("snippet", "")),
            ] if p
        )
        if not part:
            continue
        if total + len(part) > max_chars:
            break
        chunks.append(part)
        total += len(part)

    return " || ".join(chunks)


# ============================================================================
# STANDARDIZATION
# ============================================================================

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "news_id", "timestamp_utc", "source", "headline", "snippet", "body",
            "language", "ticker_tags", "topic", "url"
        ])

    colmap = {
        "news_id": choose_existing_col(df, ["news_id", "id", "guid", "uuid", "article_id"]),
        "timestamp_utc": choose_existing_col(df, ["timestamp_utc", "timestamp", "published_at", "published", "date", "datetime", "time", "created_at"]),
        "source": choose_existing_col(df, ["source", "publisher", "provider", "outlet"]),
        "headline": choose_existing_col(df, ["headline", "title", "subject"]),
        "snippet": choose_existing_col(df, ["snippet", "summary", "description", "abstract", "teaser", "subheadline"]),
        "body": choose_existing_col(df, ["body", "content", "text", "article", "full_text"]),
        "language": choose_existing_col(df, ["language", "lang"]),
        "ticker_tags": choose_existing_col(df, ["ticker_tags", "tickers", "symbols", "symbol", "mentioned_tickers"]),
        "topic": choose_existing_col(df, ["topic", "topics", "category", "categories", "tag", "tags"]),
        "url": choose_existing_col(df, ["url", "link", "article_url"]),
    }

    out = pd.DataFrame()
    for std, src in colmap.items():
        if src is not None:
            out[std] = df[src]
        else:
            out[std] = None

    return out


# ============================================================================
# CORE BUILD
# ============================================================================

def load_raw_files(raw_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    files = []
    for ext in ("*.csv", "*.parquet", "*.json", "*.jsonl", "*.ndjson"):
        files.extend(sorted(raw_dir.glob(ext)))

    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos crudos en {raw_dir}. "
            f"Coloca ahí uno o más .csv / .parquet / .json / .jsonl"
        )

    frames = []
    for fp in files:
        try:
            tmp = read_any_table(fp)
            tmp = standardize_columns(tmp)
            tmp["_source_file"] = fp.name
            frames.append(tmp)
        except Exception as e:
            print(f"[warn] No se pudo leer {fp.name}: {e}")

    if not frames:
        raise RuntimeError("No se pudo leer ningún archivo crudo válido.")

    return pd.concat(frames, ignore_index=True), files


def clean_news(df: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    # texto
    for c in ["source", "headline", "snippet", "body", "language", "topic", "url"]:
        out[c] = out[c].map(normalize_text)

    # timestamps
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=True)
    out = out[out["timestamp_utc"].notna()].copy()

    # si no hay id, generar uno
    if out["news_id"].isna().all():
        out["news_id"] = None
    out["news_id"] = out["news_id"].astype("string")
    missing_id = out["news_id"].isna() | (out["news_id"].str.strip() == "")
    out.loc[missing_id, "news_id"] = [
        hashlib.sha1(f"{ts}|{h}|{s}".encode("utf-8")).hexdigest()
        for ts, h, s in zip(
            out.loc[missing_id, "timestamp_utc"].astype(str),
            out.loc[missing_id, "headline"].astype(str),
            out.loc[missing_id, "source"].astype(str),
        )
    ]

    # ticker tags
    out["ticker_tags"] = out["ticker_tags"].map(safe_to_list)
    inferred_tickers = []
    for _, row in out.iterrows():
        text = " ".join([
            row.get("headline", ""),
            row.get("snippet", ""),
            row.get("body", ""),
            " ".join(row.get("ticker_tags", [])),
        ])
        inferred = infer_tickers(text, universe)
        combined = sorted(set(row.get("ticker_tags", []) + inferred))
        inferred_tickers.append(combined)
    out["ticker_tags"] = inferred_tickers

    # topic inferido
    inferred_topics = []
    for _, row in out.iterrows():
        text = " ".join([row.get("headline", ""), row.get("snippet", ""), row.get("body", "")])
        base_topic = normalize_text(row.get("topic", "")).lower()
        inferred_topic = detect_topic(text)
        if base_topic:
            inferred_topics.append(base_topic)
        else:
            inferred_topics.append(inferred_topic)
    out["topic"] = inferred_topics

    # relevance
    relevance_kind = []
    clarity_proxy = []
    event_strength = []
    source_w = []

    for _, row in out.iterrows():
        text = " ".join([row.get("headline", ""), row.get("snippet", ""), row.get("body", "")])
        rel = compute_relevance_kind(row["topic"], row["ticker_tags"], text)
        panic_hits = sum(1 for k in PANIC_KEYWORDS if k in text.lower())
        opp_hits = sum(1 for k in OPPORTUNITY_KEYWORDS if k in text.lower())
        # claridad simple: más alta si una dirección domina
        denom = panic_hits + opp_hits
        cp = abs(panic_hits - opp_hits) / denom if denom > 0 else 0.35
        sw = source_weight(row.get("source", "unknown"))
        es = compute_event_strength(row["topic"], rel, cp) * sw

        relevance_kind.append(rel)
        clarity_proxy.append(cp)
        event_strength.append(es)
        source_w.append(sw)

    out["relevance_kind"] = relevance_kind
    out["clarity_proxy"] = clarity_proxy
    out["source_weight"] = source_w
    out["event_strength"] = event_strength

    # filtros mínimos de calidad
    out = out[
        (out["headline"].str.len() > 5) |
        (out["snippet"].str.len() > 20) |
        (out["body"].str.len() > 40)
    ].copy()

    # deduplicación fuerte 1: por fileds exactos relevantes
    out["headline_norm"] = out["headline"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    out["snippet_norm"] = out["snippet"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.drop_duplicates(
        subset=["timestamp_utc", "source", "headline_norm", "snippet_norm"],
        keep="first"
    ).copy()

    # deduplicación fuerte 2: evento por fecha + source + headline
    out["event_key"] = [
        hash_event_key(ts, src, head)
        for ts, src, head in zip(out["timestamp_utc"], out["source"], out["headline"])
    ]
    out = out.drop_duplicates(subset=["event_key"], keep="first").copy()

    # semana alineada a viernes
    ts_naive = out["timestamp_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    out["week_end"] = ts_naive.dt.to_period("W-FRI").dt.end_time.dt.normalize()

    # columnas finales ordenadas
    cols = [
        "news_id", "timestamp_utc", "week_end", "source", "source_weight",
        "headline", "snippet", "body", "language", "ticker_tags", "topic",
        "relevance_kind", "clarity_proxy", "event_strength", "url",
        "_source_file", "headline_norm", "snippet_norm", "event_key"
    ]
    out = out[cols].sort_values(["timestamp_utc", "source"]).reset_index(drop=True)
    return out


def build_weekly_rollup(clean_df: pd.DataFrame, universe: list[str], max_articles_per_week: int) -> pd.DataFrame:
    if clean_df.empty:
        return pd.DataFrame(columns=[
            "week_end", "n_articles", "n_sources", "n_macro", "n_tech", "n_ticker",
            "n_low", "avg_source_weight", "avg_event_strength", "avg_clarity_proxy",
            "top_topics", "top_sources", "top_tickers", "headline_concat",
            "snippet_concat", "text_block"
        ])

    rows = []
    for week_end, g in clean_df.groupby("week_end", sort=True):
        topics = g["topic"].value_counts().head(5).index.tolist()
        sources = g["source"].value_counts().head(5).index.tolist()
        ticker_list = []
        for vals in g["ticker_tags"]:
            ticker_list.extend(vals)
        top_tickers = pd.Series(ticker_list).value_counts().head(8).index.tolist() if ticker_list else []

        sorted_g = g.sort_values(
            by=["event_strength", "source_weight", "timestamp_utc"],
            ascending=[False, False, True]
        ).head(max_articles_per_week)

        headline_concat = " || ".join([h for h in sorted_g["headline"].tolist() if h])
        snippet_concat = " || ".join([s for s in sorted_g["snippet"].tolist() if s])
        text_block = build_text_block(sorted_g, max_articles=max_articles_per_week)

        rows.append({
            "week_end": week_end,
            "n_articles": int(len(g)),
            "n_sources": int(g["source"].nunique()),
            "n_macro": int((g["relevance_kind"] == "macro").sum()),
            "n_tech": int((g["relevance_kind"] == "tech").sum()),
            "n_ticker": int((g["relevance_kind"] == "ticker").sum()),
            "n_low": int((g["relevance_kind"] == "low").sum()),
            "avg_source_weight": float(g["source_weight"].mean()),
            "avg_event_strength": float(g["event_strength"].mean()),
            "avg_clarity_proxy": float(g["clarity_proxy"].mean()),
            "panic_like_count": int((g["topic"] == "panic").sum()),
            "opportunity_like_count": int((g["topic"] == "opportunity").sum()),
            "macro_topic_count": int((g["topic"] == "macro").sum()),
            "tech_topic_count": int((g["topic"] == "tech").sum()),
            "top_topics": json.dumps(topics, ensure_ascii=False),
            "top_sources": json.dumps(sources, ensure_ascii=False),
            "top_tickers": json.dumps(top_tickers, ensure_ascii=False),
            "headline_concat": headline_concat,
            "snippet_concat": snippet_concat,
            "text_block": text_block,
        })

    weekly = pd.DataFrame(rows).sort_values("week_end").reset_index(drop=True)

    # algunas features semanales útiles desde ya
    if not weekly.empty:
        weekly["news_volume_z_8w"] = (
            (weekly["n_articles"] - weekly["n_articles"].rolling(8, min_periods=4).mean()) /
            weekly["n_articles"].rolling(8, min_periods=4).std(ddof=0).replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        weekly["panic_share"] = np.where(
            weekly["n_articles"] > 0,
            weekly["panic_like_count"] / weekly["n_articles"],
            0.0
        )
        weekly["opportunity_share"] = np.where(
            weekly["n_articles"] > 0,
            weekly["opportunity_like_count"] / weekly["n_articles"],
            0.0
        )
        weekly["macro_share"] = np.where(
            weekly["n_articles"] > 0,
            weekly["n_macro"] / weekly["n_articles"],
            0.0
        )
        weekly["ticker_share"] = np.where(
            weekly["n_articles"] > 0,
            weekly["n_ticker"] / weekly["n_articles"],
            0.0
        )

    return weekly


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Construye dataset base semanal de noticias para Mahoraga.")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Ruta base de news_mahoraga. Si se omite, se infiere como carpeta padre de src.")
    parser.add_argument("--max-articles-per-week", type=int, default=30,
                        help="Máximo de noticias que se usarán para construir el bloque textual semanal.")
    parser.add_argument("--write-csv-too", action="store_true",
                        help="Si se activa, además de parquet exporta csv.")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    inferred_base = script_path.parent.parent
    base_dir = Path(args.base_dir).resolve() if args.base_dir else inferred_base

    raw_dir = base_dir / "news_data" / "raw"
    processed_dir = base_dir / "news_data" / "processed"
    weekly_dir = base_dir / "news_data" / "weekly"
    manifests_dir = base_dir / "news_data" / "manifests"

    processed_dir.mkdir(parents=True, exist_ok=True)
    weekly_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    universe = DEFAULT_TECH_UNIVERSE

    print("=" * 80)
    print(" BUILD NEWS DATASET — MAHORAGA")
    print("=" * 80)
    print(f"[base] {base_dir}")
    print(f"[raw]  {raw_dir}")

    raw_df, files = load_raw_files(raw_dir)
    rows_loaded_raw = len(raw_df)

    print(f"[load] archivos encontrados: {len(files)}")
    print(f"[load] filas crudas cargadas: {rows_loaded_raw:,}")

    clean_df = clean_news(raw_df, universe=universe)
    rows_after_timestamp_filter = int(raw_df["timestamp_utc"].notna().sum()) if "timestamp_utc" in raw_df.columns else 0
    rows_after_dedup = len(clean_df)  # aproximación útil para manifest simple

    clean_path = processed_dir / "news_clean.parquet"
    clean_df.to_parquet(clean_path, index=False)

    if args.write_csv_too:
        clean_df.to_csv(processed_dir / "news_clean.csv", index=False, encoding="utf-8-sig")

    print(f"[clean] filas limpias finales: {len(clean_df):,}")
    print(f"[clean] exportado: {clean_path}")

    weekly_df = build_weekly_rollup(
        clean_df=clean_df,
        universe=universe,
        max_articles_per_week=args.max_articles_per_week,
    )

    weekly_path = weekly_dir / "weekly_news_rollup.parquet"
    weekly_df.to_parquet(weekly_path, index=False)

    if args.write_csv_too:
        weekly_df.to_csv(weekly_dir / "weekly_news_rollup.csv", index=False, encoding="utf-8-sig")

    print(f"[weekly] semanas generadas: {len(weekly_df):,}")
    print(f"[weekly] exportado: {weekly_path}")

    manifest = BuildManifest(
        raw_files_found=len(files),
        rows_loaded_raw=rows_loaded_raw,
        rows_after_timestamp_filter=rows_after_timestamp_filter,
        rows_after_dedup=rows_after_dedup,
        rows_final_clean=len(clean_df),
        weeks_final=len(weekly_df),
        min_timestamp_utc=str(clean_df["timestamp_utc"].min()) if not clean_df.empty else None,
        max_timestamp_utc=str(clean_df["timestamp_utc"].max()) if not clean_df.empty else None,
        output_clean_path=str(clean_path),
        output_weekly_path=str(weekly_path),
        universe_tickers=universe,
    )

    manifest_path = manifests_dir / "build_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, ensure_ascii=False, indent=2)

    print(f"[manifest] exportado: {manifest_path}")
    print("[done] dataset base construido correctamente.")


if __name__ == "__main__":
    main()