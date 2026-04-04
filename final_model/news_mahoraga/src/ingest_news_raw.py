from __future__ import annotations

"""
ingest_news_raw.py
==================

Construye la fuente cruda de noticias para Mahoraga usando:
- GDELT DOC API para descubrimiento de URLs/noticias
- Trafilatura para extracción de texto principal y metadatos

Salidas:
- news_data/raw/news_raw_gdelt.parquet
- news_data/raw/news_raw_gdelt.jsonl

Después de correr este script, ejecuta:
    build_news_dataset.py

Notas:
- Este script NO crea labels todavía.
- Algunas URLs no bajarán por bloqueo, JS pesado, paywalls o link rot.
- Para historia muy profunda, probablemente necesitarás complementar con archivos,
  archivos web o una fuente comercial.
"""

import argparse
import json
import time
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
import threading

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import trafilatura


# ============================================================================
# CONFIG
# ============================================================================

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "AVGO", "ASML", "TSM", "ADBE", "NFLX", "AMD"
]

DEFAULT_QUERY_PACKS = {
    "macro_risk": r'("federal reserve" OR fomc OR inflation OR unemployment OR recession OR "credit crisis" OR "banking crisis" OR tariffs OR sanctions OR war OR geopolitical)',
    "tech_ai": r'("artificial intelligence" OR AI OR semiconductor OR chip OR gpu OR datacenter OR cloud OR "export controls" OR antitrust OR "AI bubble")',
    "universe": r'(AAPL OR Apple OR MSFT OR Microsoft OR NVDA OR Nvidia OR GOOGL OR Google OR AMZN OR Amazon OR META OR Meta OR AVGO OR Broadcom OR ASML OR TSM OR ADBE OR Adobe OR NFLX OR Netflix OR AMD)'
}

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MahoragaNewsIngest/1.0; +https://localhost)"
}

REQUEST_TIMEOUT = 20
HTML_TIMEOUT = 10          # reducido de 25s → 10s: fail-fast en URLs lentas/caídas
SLEEP_BETWEEN_API_CALLS = 1.2
SLEEP_BETWEEN_HTML_CALLS = 0.0   # eliminado: el throttling lo maneja el pool

# Concurrencia para fetch de HTML — I/O-bound, 20 workers es seguro y rápido.
# Reducir si el servidor de destino bloquea por rate-limit agresivo.
HTML_FETCH_WORKERS = 20

# Checkpoint: guarda progreso cada N artículos procesados.
# Permite reanudar si el script se interrumpe.
CHECKPOINT_EVERY = 500


# ============================================================================
# MODELS
# ============================================================================

@dataclass
class IngestManifest:
    start_date: str
    end_date: str
    query_packs: dict[str, str]
    rows_discovered: int
    rows_after_url_dedup: int
    rows_fetched_ok: int
    rows_extracted_ok: int
    output_parquet: str
    output_jsonl: str


# ============================================================================
# HELPERS
# ============================================================================

def make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(DEFAULT_HEADERS)
    return session


# Thread-local storage: cada worker thread tiene su propia Session,
# evitando contención y errores de concurrencia.
_thread_local = threading.local()


def get_thread_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = make_session()
    return _thread_local.session


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False) if obj is not None else "[]"


def chunk_days(start_date: datetime, end_date: datetime) -> Iterable[tuple[datetime, datetime]]:
    current = start_date
    while current <= end_date:
        nxt = min(current + timedelta(days=1), end_date + timedelta(days=1))
        yield current, nxt
        current = nxt


def to_gdelt_dt(dt: datetime) -> str:
    # GDELT suele esperar YYYYMMDDHHMMSS en UTC
    return dt.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def infer_tickers(text: str, universe: list[str]) -> list[str]:
    found = []
    upper = f" {text.upper()} "
    for tk in universe:
        if re.search(rf"(?<![A-Z]){re.escape(tk)}(?![A-Z])", upper):
            found.append(tk)
    return sorted(set(found))


# ============================================================================
# GDELT DISCOVERY
# ============================================================================

def gdelt_artlist_query(
    session: requests.Session,
    query: str,
    dt_start: datetime,
    dt_end: datetime,
    max_records: int = 250,
) -> list[dict[str, Any]]:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "sort": "DateDesc",
        "maxrecords": str(max_records),
        "startdatetime": to_gdelt_dt(dt_start),
        "enddatetime": to_gdelt_dt(dt_end),
    }

    resp = session.get(GDELT_DOC_API, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    data = resp.json()
    articles = data.get("articles", [])
    if not isinstance(articles, list):
        return []
    return articles


def discover_news(
    session: requests.Session,
    start_date: datetime,
    end_date: datetime,
    query_packs: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for bucket_start, bucket_end in chunk_days(start_date, end_date):
        for pack_name, query in query_packs.items():
            try:
                articles = gdelt_artlist_query(
                    session=session,
                    query=query,
                    dt_start=bucket_start,
                    dt_end=bucket_end,
                    max_records=250,
                )
            except Exception as e:
                print(f"[warn] GDELT falló en {pack_name} {bucket_start.date()} → {bucket_end.date()}: {e}")
                time.sleep(SLEEP_BETWEEN_API_CALLS)
                continue

            for art in articles:
                rows.append({
                    "gdelt_pack": pack_name,
                    "gdelt_query": query,
                    "url": art.get("url"),
                    "domain": art.get("domain"),
                    "title": art.get("title") or art.get("seendate") or "",
                    "seendate": art.get("seendate"),
                    "socialimage": art.get("socialimage"),
                    "language": art.get("language"),
                    "sourcecountry": art.get("sourcecountry"),
                })

            print(f"[gdelt] {bucket_start.date()} {pack_name}: {len(articles)} artículos")
            time.sleep(SLEEP_BETWEEN_API_CALLS)

    if not rows:
        return pd.DataFrame(columns=[
            "gdelt_pack", "gdelt_query", "url", "domain", "title", "seendate",
            "socialimage", "language", "sourcecountry"
        ])

    df = pd.DataFrame(rows)
    df["url"] = df["url"].map(normalize_text)
    df = df[df["url"].str.startswith("http")].copy()
    df["title"] = df["title"].map(normalize_text)
    df["domain"] = df["domain"].map(normalize_text)
    df["language"] = df["language"].map(normalize_text)
    df["sourcecountry"] = df["sourcecountry"].map(normalize_text)
    df["seendate"] = pd.to_datetime(df["seendate"], errors="coerce", utc=True)

    # dedupe por URL
    df = df.sort_values(["url", "seendate"]).drop_duplicates(subset=["url"], keep="last").reset_index(drop=True)
    return df


# ============================================================================
# HTML + EXTRACTION
# ============================================================================

def fetch_html(session: requests.Session, url: str) -> str | None:
    """Descarga HTML. En el pool paralelo usa get_thread_session() en vez de session."""
    try:
        resp = session.get(url, timeout=HTML_TIMEOUT)
        resp.raise_for_status()
        if "text/html" not in resp.headers.get("Content-Type", "") and "<html" not in resp.text.lower():
            return None
        return resp.text
    except Exception:
        return None


def fetch_html_thread(url: str) -> tuple[str, str | None]:
    """Versión para ThreadPoolExecutor: usa session thread-local, devuelve (url, html)."""
    session = get_thread_session()
    return url, fetch_html(session, url)


def extract_article(url: str, html: str) -> dict[str, Any]:
    # trafilatura puede devolver JSON con metadata
    result_json = trafilatura.extract(
        html,
        url=url,
        output_format="json",
        with_metadata=True,
        favor_precision=True,
    )

    if not result_json:
        # fallback rápido
        text = trafilatura.extract(
            html,
            url=url,
            output_format="txt",
            with_metadata=False,
            favor_precision=True,
        )
        return {
            "headline_extracted": "",
            "text_extracted": normalize_text(text),
            "author": "",
            "date_extracted": "",
            "sitename": "",
            "excerpt": "",
            "extract_ok": bool(text and len(text.strip()) > 100),
        }

    try:
        obj = json.loads(result_json)
    except Exception:
        return {
            "headline_extracted": "",
            "text_extracted": "",
            "author": "",
            "date_extracted": "",
            "sitename": "",
            "excerpt": "",
            "extract_ok": False,
        }

    text_extracted = normalize_text(obj.get("text", ""))
    excerpt = normalize_text(obj.get("excerpt", ""))
    return {
        "headline_extracted": normalize_text(obj.get("title", "")),
        "text_extracted": text_extracted,
        "author": normalize_text(obj.get("author", "")),
        "date_extracted": normalize_text(obj.get("date", "")),
        "sitename": normalize_text(obj.get("sitename", "")),
        "excerpt": excerpt,
        "extract_ok": bool(text_extracted and len(text_extracted) > 100),
    }


def _process_one_row(row_tuple: tuple[int, Any], html_map: dict[str, str | None], universe: list[str]) -> dict[str, Any]:
    """Procesa una fila usando el HTML ya descargado. CPU-bound (trafilatura)."""
    i, row = row_tuple
    url = row["url"]
    html = html_map.get(url)

    fetched_ok = html is not None
    if fetched_ok:
        extracted = extract_article(url, html)
    else:
        extracted = {
            "headline_extracted": "",
            "text_extracted": "",
            "author": "",
            "date_extracted": "",
            "sitename": "",
            "excerpt": "",
            "extract_ok": False,
        }

    source_name = normalize_text(extracted["sitename"] or row.get("domain") or "unknown")
    headline    = normalize_text(extracted["headline_extracted"] or row.get("title") or "")
    snippet     = normalize_text(extracted["excerpt"])
    body        = normalize_text(extracted["text_extracted"])
    language    = normalize_text(row.get("language") or "")
    ts          = row.get("seendate")

    full_text = " ".join([headline, snippet, body])
    tickers   = infer_tickers(full_text, universe)
    news_id   = sha1(url)

    return {
        "news_id":       news_id,
        "timestamp":     ts.isoformat() if pd.notna(ts) else "",
        "source":        source_name,
        "headline":      headline,
        "snippet":       snippet,
        "body":          body,
        "language":      language if language else "en",
        "ticker_tags":   tickers,
        "topic":         row.get("gdelt_pack", "general"),
        "url":           url,
        "domain":        normalize_text(row.get("domain", "")),
        "sourcecountry": normalize_text(row.get("sourcecountry", "")),
        "gdelt_query":   normalize_text(row.get("gdelt_query", "")),
        "fetched_ok":    fetched_ok,
        "extract_ok":    bool(extracted["extract_ok"]),
    }


def enrich_with_html_and_text(
    session: requests.Session,
    discovered_df: pd.DataFrame,
    universe: list[str],
    checkpoint_path: Path | None = None,
    n_workers: int = HTML_FETCH_WORKERS,
) -> pd.DataFrame:
    """
    Descarga HTML en paralelo (ThreadPoolExecutor) y extrae texto.

    Estrategia:
    1. Cargar checkpoint si existe (permite reanudar runs interrumpidos).
    2. Filtrar URLs ya procesadas.
    3. Fetch paralelo de HTML con HTML_FETCH_WORKERS workers (I/O-bound).
    4. Extracción serial con trafilatura (CPU-bound, pero ~10ms/art — no dominante).
    5. Guardar checkpoint cada CHECKPOINT_EVERY artículos.
    """
    total = len(discovered_df)

    # ── Cargar checkpoint previo ──────────────────────────────────────────────
    done_rows: list[dict] = []
    done_urls: set[str]   = set()
    if checkpoint_path and checkpoint_path.exists():
        try:
            ck = pd.read_parquet(checkpoint_path)
            done_rows = ck.to_dict("records")
            done_urls = set(ck["url"].tolist())
            print(f"[checkpoint] retomando: {len(done_urls):,} URLs ya procesadas")
        except Exception as e:
            print(f"[checkpoint] no se pudo leer ({e}), comenzando desde cero")

    # ── Filtrar pendientes ────────────────────────────────────────────────────
    pending_df = discovered_df[~discovered_df["url"].isin(done_urls)].reset_index(drop=True)
    n_pending  = len(pending_df)
    print(f"[fetch] {n_pending:,} artículos pendientes de {total:,} descubiertos")

    if n_pending == 0:
        return pd.DataFrame(done_rows)

    # ── Fetch paralelo de HTML ────────────────────────────────────────────────
    urls        = pending_df["url"].tolist()
    html_map: dict[str, str | None] = {}
    fetched_ok  = 0

    print(f"[fetch] iniciando pool de {n_workers} workers …")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(fetch_html_thread, url): url for url in urls}
        for idx, fut in enumerate(as_completed(futures), 1):
            url, html = fut.result()
            html_map[url] = html
            if html is not None:
                fetched_ok += 1
            if idx % 250 == 0 or idx == n_pending:
                pct = idx / n_pending * 100
                print(f"[fetch] {idx:,}/{n_pending:,} ({pct:.0f}%)  ok={fetched_ok:,}")

    print(f"[fetch] completado: {fetched_ok:,}/{n_pending:,} URLs exitosas")

    # ── Extracción + checkpoint ───────────────────────────────────────────────
    new_rows: list[dict] = []
    for i, row in pending_df.iterrows():
        result = _process_one_row((i, row), html_map, universe)
        new_rows.append(result)

        if len(new_rows) % CHECKPOINT_EVERY == 0:
            all_so_far = done_rows + new_rows
            if checkpoint_path:
                try:
                    pd.DataFrame(all_so_far).to_parquet(checkpoint_path, index=False)
                except Exception:
                    pass
            n_total_done = len(all_so_far)
            print(f"[extract] {n_total_done:,}/{total:,} procesados (checkpoint guardado)")

    # ── Checkpoint final + merge ──────────────────────────────────────────────
    all_rows = done_rows + new_rows
    if checkpoint_path:
        try:
            pd.DataFrame(all_rows).to_parquet(checkpoint_path, index=False)
            print(f"[checkpoint] guardado: {len(all_rows):,} artículos")
        except Exception:
            pass

    return pd.DataFrame(all_rows)


# ============================================================================
# SAVE
# ============================================================================

def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Descubre y descarga noticias crudas para Mahoraga.")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Ruta base de news_mahoraga. Si se omite, se infiere desde src.")
    parser.add_argument("--start", type=str, required=True, help="Fecha inicio YYYY-MM-DD")
    parser.add_argument("--end",   type=str, required=True, help="Fecha fin YYYY-MM-DD")
    parser.add_argument("--no-tech-pack",     action="store_true", help="Desactiva query pack tech/AI")
    parser.add_argument("--no-universe-pack", action="store_true", help="Desactiva query pack del universo")
    parser.add_argument("--no-macro-pack",    action="store_true", help="Desactiva query pack macro/risk")
    parser.add_argument("--workers", type=int, default=HTML_FETCH_WORKERS,
                        help=f"Workers paralelos para fetch de HTML (default: {HTML_FETCH_WORKERS}). "
                             "Reducir si hay bloqueos por rate-limit.")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Desactiva checkpoint. No reanudar runs anteriores.")
    args = parser.parse_args()

    script_path   = Path(__file__).resolve()
    inferred_base = script_path.parent.parent
    base_dir      = Path(args.base_dir).resolve() if args.base_dir else inferred_base

    raw_dir       = base_dir / "news_data" / "raw"
    manifests_dir = base_dir / "news_data" / "manifests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = None if args.no_checkpoint else (raw_dir / "news_raw_checkpoint.parquet")

    start_date = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_date   = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    query_packs: dict[str, str] = {}
    if not args.no_tech_pack:
        query_packs["tech_ai"] = DEFAULT_QUERY_PACKS["tech_ai"]
    if not args.no_universe_pack:
        query_packs["universe"] = DEFAULT_QUERY_PACKS["universe"]
    if not args.no_macro_pack:
        query_packs["macro_risk"] = DEFAULT_QUERY_PACKS["macro_risk"]

    if not query_packs:
        raise ValueError("Debe quedar al menos un query pack activo.")

    n_workers = args.workers
    session = make_session()

    print("=" * 80)
    print(" INGEST NEWS RAW — MAHORAGA")
    print("=" * 80)
    print(f"[base]     {base_dir}")
    print(f"[range]    {args.start} -> {args.end}")
    print(f"[packs]    {list(query_packs.keys())}")
    print(f"[workers]  {n_workers} threads para fetch HTML")
    print(f"[timeout]  HTML_TIMEOUT={HTML_TIMEOUT}s  (fail-fast)")
    if checkpoint_path:
        print(f"[checkpoint] {checkpoint_path}")

    discovered_df = discover_news(
        session=session,
        start_date=start_date,
        end_date=end_date,
        query_packs=query_packs,
    )

    if discovered_df.empty:
        print("[done] No se descubrieron artículos.")
        return

    print(f"[gdelt] descubiertos: {len(discovered_df):,} artículos únicos por URL")

    raw_df = enrich_with_html_and_text(
        session=session,
        discovered_df=discovered_df,
        universe=DEFAULT_UNIVERSE,
        checkpoint_path=checkpoint_path,
        n_workers=n_workers,
    )

    # dedupe final por URL / news_id
    raw_df = raw_df.sort_values(["news_id", "extract_ok", "fetched_ok"]).drop_duplicates(
        subset=["news_id"], keep="last"
    ).reset_index(drop=True)

    parquet_path = raw_dir / "news_raw_gdelt.parquet"
    jsonl_path   = raw_dir / "news_raw_gdelt.jsonl"

    raw_df.to_parquet(parquet_path, index=False)
    save_jsonl(raw_df, jsonl_path)

    # Limpiar checkpoint si todo fue bien
    if checkpoint_path and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            print("[checkpoint] eliminado (run completado)")
        except Exception:
            pass

    manifest = IngestManifest(
        start_date=args.start,
        end_date=args.end,
        query_packs=query_packs,
        rows_discovered=int(len(discovered_df)),
        rows_after_url_dedup=int(len(discovered_df)),
        rows_fetched_ok=int(raw_df["fetched_ok"].fillna(False).sum()),
        rows_extracted_ok=int(raw_df["extract_ok"].fillna(False).sum()),
        output_parquet=str(parquet_path),
        output_jsonl=str(jsonl_path),
    )

    manifest_path = manifests_dir / "ingest_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, ensure_ascii=False, indent=2)

    print(f"[save] parquet:  {parquet_path}")
    print(f"[save] jsonl:    {jsonl_path}")
    print(f"[save] manifest: {manifest_path}")
    print("[done] Fuente cruda construida.")


if __name__ == "__main__":
    main()