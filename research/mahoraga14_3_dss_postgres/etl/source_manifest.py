from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from .control_plane import upsert_source_manifest, write_control_json
from .discover_artifacts import EXPECTED_ARTIFACTS
from .lfs_guard import assert_not_lfs_pointer
from .paths import DssPaths, ensure_output_dirs, get_paths

SMALL_FILE_HASH_BYTES = 64 * 1024 * 1024
SAMPLE_BYTES = 1024 * 1024
DATE_COLUMNS = ["date", "date_value", "decision_date", "timestamp", "created_at"]
DIMENSION_COLUMNS = ["candidate_id", "universe_id", "fold", "horizon", "module_name"]


@dataclass(frozen=True)
class SourceManifestEntry:
    source_name: str
    source_path: str
    source_type: str
    source_hash: str
    row_count: int | None = None
    min_date: str | None = None
    max_date: str | None = None
    candidate_id: str | None = None
    universe_id: str | None = None
    fold: int | None = None
    horizon: int | None = None
    module_name: str | None = None
    size_bytes: int | None = None
    modified_at_ns: int | None = None
    hash_strategy: str = "full"
    exists: bool = True


@dataclass(frozen=True)
class SourceDiff:
    new_sources: list[SourceManifestEntry]
    modified_sources: list[SourceManifestEntry]
    unchanged_sources: list[SourceManifestEntry]
    missing_sources: list[str]
    changed_sources: list[SourceManifestEntry]
    changed_row_count: int
    total_row_count: int

    @property
    def changed_sources_count(self) -> int:
        return len(self.changed_sources)

    @property
    def changed_ratio(self) -> float:
        if self.total_row_count <= 0:
            return 1.0 if self.changed_sources else 0.0
        return min(1.0, self.changed_row_count / self.total_row_count)


def _artifact_paths(paths: DssPaths) -> list[tuple[str, Path]]:
    return [(artifact.role, paths.repo_root / artifact.relative_path) for artifact in EXPECTED_ARTIFACTS]


def compute_file_hash(path: Path, small_file_threshold: int = SMALL_FILE_HASH_BYTES) -> tuple[str, str]:
    assert_not_lfs_pointer(path)
    stat = path.stat()
    hasher = hashlib.sha256()
    if stat.st_size <= small_file_threshold:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest(), "full"

    hasher.update(str(stat.st_size).encode())
    hasher.update(str(stat.st_mtime_ns).encode())
    with path.open("rb") as handle:
        hasher.update(handle.read(SAMPLE_BYTES))
        handle.seek(max(0, stat.st_size - SAMPLE_BYTES))
        hasher.update(handle.read(SAMPLE_BYTES))
    return hasher.hexdigest(), "metadata_sample"


def _scan_frame(path: Path) -> pl.LazyFrame | None:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.scan_parquet(path)
    if suffix == ".csv":
        return pl.scan_csv(path, infer_schema_length=1000, ignore_errors=True)
    return None


def _sample(path: Path, limit: int = 2048) -> pl.DataFrame:
    lf = _scan_frame(path)
    if lf is None:
        return pl.DataFrame()
    try:
        return lf.head(limit).collect()
    except Exception:
        return pl.DataFrame()


def _single_value(sample: pl.DataFrame, column: str) -> Any:
    if sample.is_empty() or column not in sample.columns:
        return None
    values = sample.get_column(column).drop_nulls().unique()
    if len(values) == 1:
        return values[0]
    return None


def _date_bounds(path: Path, sample: pl.DataFrame) -> tuple[str | None, str | None]:
    date_col = next((column for column in DATE_COLUMNS if column in sample.columns), None)
    if not date_col:
        return None, None
    lf = _scan_frame(path)
    if lf is None:
        return None, None
    try:
        result = lf.select(
            pl.col(date_col).cast(pl.Date, strict=False).min().alias("min_date"),
            pl.col(date_col).cast(pl.Date, strict=False).max().alias("max_date"),
        ).collect()
        min_date = result.item(0, "min_date")
        max_date = result.item(0, "max_date")
        return _date_to_str(min_date), _date_to_str(max_date)
    except Exception:
        try:
            parsed = sample.select(pl.col(date_col).cast(pl.Date, strict=False).min().alias("min_date"), pl.col(date_col).cast(pl.Date, strict=False).max().alias("max_date"))
            return _date_to_str(parsed.item(0, "min_date")), _date_to_str(parsed.item(0, "max_date"))
        except Exception:
            return None, None


def _date_to_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def _row_count(path: Path) -> int | None:
    lf = _scan_frame(path)
    if lf is None:
        return None
    try:
        return int(lf.select(pl.len()).collect().item())
    except Exception:
        return None


def infer_source_metadata(path: Path, dataframe_sample: pl.DataFrame) -> dict[str, Any]:
    min_date, max_date = _date_bounds(path, dataframe_sample)
    metadata = {
        "row_count": _row_count(path),
        "min_date": min_date,
        "max_date": max_date,
        "candidate_id": _single_value(dataframe_sample, "candidate_id"),
        "universe_id": _single_value(dataframe_sample, "universe_id"),
        "fold": _single_value(dataframe_sample, "fold"),
        "horizon": _single_value(dataframe_sample, "horizon"),
        "module_name": _single_value(dataframe_sample, "module_name"),
    }
    for key in ["fold", "horizon"]:
        if metadata[key] is not None:
            try:
                metadata[key] = int(metadata[key])
            except Exception:
                metadata[key] = None
    for key in ["candidate_id", "universe_id", "module_name"]:
        if metadata[key] is not None:
            metadata[key] = str(metadata[key])
    return metadata


def scan_sources(paths: Iterable[Path] | None = None, config: Any | None = None) -> list[SourceManifestEntry]:
    dss_paths = ensure_output_dirs(get_paths())
    items: list[tuple[str, Path]]
    if paths is None:
        items = _artifact_paths(dss_paths)
    else:
        items = [("explicit_source", Path(path)) for path in paths]

    entries: list[SourceManifestEntry] = []
    for role, path in items:
        source_name = path.stem
        source_path = str(path.resolve())
        source_type = path.suffix.lower().lstrip(".") or "directory"
        if not path.exists() or not path.is_file():
            entries.append(
                SourceManifestEntry(
                    source_name=source_name,
                    source_path=source_path,
                    source_type=source_type,
                    source_hash="missing",
                    exists=False,
                )
            )
            continue
        file_hash, strategy = compute_file_hash(path)
        stat = path.stat()
        sample = _sample(path)
        metadata = infer_source_metadata(path, sample)
        entries.append(
            SourceManifestEntry(
                source_name=source_name,
                source_path=source_path,
                source_type=source_type,
                source_hash=file_hash,
                size_bytes=stat.st_size,
                modified_at_ns=stat.st_mtime_ns,
                hash_strategy=strategy,
                **metadata,
            )
        )
    return entries


def load_previous_manifest(database_url: str | None = None) -> dict[str, SourceManifestEntry]:
    if not database_url:
        paths = get_paths()
        candidates = sorted(paths.control_root.glob("source_manifest_*.json")) if paths.control_root.exists() else []
        if not candidates:
            return {}
        try:
            payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
            return {
                f"{row['source_name']}::{row['source_path']}": SourceManifestEntry(**{key: row.get(key) for key in SourceManifestEntry.__dataclass_fields__})
                for row in payload.get("sources", [])
            }
        except Exception:
            return {}
    try:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(database_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT source_name, source_path, source_type, source_hash, row_count,
                           min_date::text AS min_date, max_date::text AS max_date,
                           candidate_id, universe_id, fold, horizon, module_name
                    FROM oltp.source_manifest
                    """
                )
                rows = cur.fetchall()
        entries = []
        for row in rows:
            entries.append(SourceManifestEntry(**{field: row.get(field) for field in SourceManifestEntry.__dataclass_fields__ if field in row}))
        return {f"{entry.source_name}::{entry.source_path}": entry for entry in entries}
    except Exception:
        return {}


def diff_manifests(current: list[SourceManifestEntry], previous: dict[str, SourceManifestEntry] | list[SourceManifestEntry]) -> SourceDiff:
    previous_map = previous if isinstance(previous, dict) else {f"{entry.source_name}::{entry.source_path}": entry for entry in previous}
    current_map = {f"{entry.source_name}::{entry.source_path}": entry for entry in current}
    new_sources: list[SourceManifestEntry] = []
    modified_sources: list[SourceManifestEntry] = []
    unchanged_sources: list[SourceManifestEntry] = []

    for key, entry in current_map.items():
        old = previous_map.get(key)
        if old is None:
            new_sources.append(entry)
        elif old.source_hash != entry.source_hash or old.row_count != entry.row_count:
            modified_sources.append(entry)
        else:
            unchanged_sources.append(entry)

    missing_sources = sorted(set(previous_map) - set(current_map))
    changed_sources = [*new_sources, *modified_sources]
    changed_row_count = sum(int(entry.row_count or 0) for entry in changed_sources)
    total_row_count = sum(int(entry.row_count or 0) for entry in current)
    return SourceDiff(
        new_sources=new_sources,
        modified_sources=modified_sources,
        unchanged_sources=unchanged_sources,
        missing_sources=missing_sources,
        changed_sources=changed_sources,
        changed_row_count=changed_row_count,
        total_row_count=total_row_count,
    )


def write_manifest_report(entries: list[SourceManifestEntry], diff: SourceDiff, run_id: str, paths: DssPaths | None = None) -> Path:
    paths = ensure_output_dirs(paths or get_paths())
    payload = {
        "run_id": run_id,
        "summary": {
            "sources": len(entries),
            "new_sources": len(diff.new_sources),
            "modified_sources": len(diff.modified_sources),
            "unchanged_sources": len(diff.unchanged_sources),
            "missing_sources": len(diff.missing_sources),
            "changed_row_count": diff.changed_row_count,
            "total_row_count": diff.total_row_count,
            "changed_ratio": diff.changed_ratio,
        },
        "sources": [asdict(entry) for entry in entries],
        "missing_sources": diff.missing_sources,
    }
    return write_control_json(paths, f"source_manifest_{run_id}.json", payload)


def persist_manifest(database_url: str | None, entries: list[SourceManifestEntry], run_id: str) -> None:
    upsert_source_manifest(database_url, entries, run_id)
