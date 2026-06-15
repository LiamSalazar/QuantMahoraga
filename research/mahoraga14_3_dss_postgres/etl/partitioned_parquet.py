from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from .control_plane import log_partition_manifest, write_control_json
from .paths import DssPaths, ensure_output_dirs, get_paths

PARTITION_COLUMNS: dict[str, list[str]] = {
    "fact_position_daily": ["year", "fold"],
    "fact_signal_daily": ["year", "fold"],
    "fact_market_bar": ["year"],
    "fact_outcome": ["horizon", "fold"],
    "fact_module_trace": ["module_name", "fold"],
    "fact_whatif": ["horizon", "fold", "demo_mode"],
    "fact_path_recursive": ["year", "fold"],
}


@dataclass(frozen=True)
class PartitionManifestEntry:
    table_name: str
    partition_key: str
    partition_type: str
    row_count: int
    min_date: str | None
    max_date: str | None
    source_hash: str | None = None
    status: str = "written"


def _with_partition_helpers(df: pl.DataFrame) -> pl.DataFrame:
    if "year" not in df.columns:
        if "date_value" in df.columns:
            return df.with_columns(pl.col("date_value").dt.year().alias("year"))
        if "decision_date" in df.columns:
            return df.with_columns(pl.col("decision_date").dt.year().alias("year"))
    return df


def write_partitioned_table(df: pl.DataFrame, table_name: str, partition_cols: list[str] | None = None, root: Path | None = None) -> list[PartitionManifestEntry]:
    if df.is_empty():
        return []
    root = root or get_paths().outputs_root / "parquet_partitioned"
    partition_cols = partition_cols or PARTITION_COLUMNS.get(table_name)
    if not partition_cols:
        return []
    df = _with_partition_helpers(df)
    missing = [column for column in partition_cols if column not in df.columns]
    if missing:
        return []
    table_root = root / table_name
    table_root.mkdir(parents=True, exist_ok=True)
    manifests: list[PartitionManifestEntry] = []
    for values, part in df.partition_by(partition_cols, as_dict=True, maintain_order=True).items():
        if not isinstance(values, tuple):
            values = (values,)
        partition_path = table_root
        key_parts = []
        for column, value in zip(partition_cols, values, strict=False):
            safe_value = "__null__" if value is None else str(value).replace("/", "_")
            partition_path = partition_path / f"{column}={safe_value}"
            key_parts.append(f"{column}={safe_value}")
        partition_path.mkdir(parents=True, exist_ok=True)
        target = partition_path / "part-000.parquet"
        part.write_parquet(target)
        min_date, max_date = _bounds(part)
        manifests.append(
            PartitionManifestEntry(
                table_name=table_name,
                partition_key="/".join(key_parts),
                partition_type="logical_parquet",
                row_count=part.height,
                min_date=min_date,
                max_date=max_date,
            )
        )
    return manifests


def _bounds(df: pl.DataFrame) -> tuple[str | None, str | None]:
    column = "date_value" if "date_value" in df.columns else "decision_date" if "decision_date" in df.columns else None
    if not column:
        return None, None
    result = df.select(pl.col(column).min().alias("min_date"), pl.col(column).max().alias("max_date"))
    return str(result.item(0, "min_date"))[:10], str(result.item(0, "max_date"))[:10]


def read_partitioned_table(table_name: str, filters: list[tuple[str, str, Any]] | None = None, root: Path | None = None) -> pl.DataFrame:
    root = root or get_paths().outputs_root / "parquet_partitioned"
    table_root = root / table_name
    if not table_root.exists():
        return pl.DataFrame()
    lf = pl.scan_parquet(str(table_root / "**" / "*.parquet"), hive_partitioning=True)
    if filters:
        for column, operator, value in filters:
            if operator == "=":
                lf = lf.filter(pl.col(column) == value)
            elif operator == "in":
                lf = lf.filter(pl.col(column).is_in(value))
    return lf.collect()


def write_partition_manifest(
    run_id: str,
    entries: list[PartitionManifestEntry],
    paths: DssPaths | None = None,
    database_url: str | None = None,
) -> None:
    if not entries:
        return
    paths = ensure_output_dirs(paths or get_paths())
    payload = {"run_id": run_id, "partitions": [asdict(entry) for entry in entries]}
    write_control_json(paths, f"partition_manifest_{run_id}.json", payload)
    log_partition_manifest(database_url, run_id, [asdict(entry) for entry in entries])
