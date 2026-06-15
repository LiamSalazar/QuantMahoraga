from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from .control_plane import log_partition_manifest
from .load_postgres import _parquet_for
from .partition_rules import PARTITION_SPECS, TABLE_DATE_COLUMN, parse_partition_label
from .partitioned_parquet import read_partitioned_table
from .paths import DssPaths, get_paths
from .source_manifest import SourceDiff


@dataclass(frozen=True)
class PartitionKey:
    table_name: str
    key: dict[str, Any]

    def label(self) -> str:
        return "/".join(f"{name}={value}" for name, value in self.key.items())


def _qualified(table_name: str) -> str:
    return table_name if "." in table_name else f"dw.{table_name}"


def _with_year(df: pl.DataFrame, table_name: str) -> pl.DataFrame:
    if "year" in df.columns:
        return df
    date_column = TABLE_DATE_COLUMN.get(table_name)
    if date_column and date_column in df.columns:
        return df.with_columns(pl.col(date_column).dt.year().alias("year"))
    return df


def derive_affected_partitions(source_diff: SourceDiff, table_name: str) -> list[PartitionKey]:
    # Source-level metadata can infer coarse partition keys only for horizons/modules.
    keys: list[PartitionKey] = []
    for source in source_diff.changed_sources:
        if table_name == "fact_outcome" and source.horizon is not None:
            keys.append(PartitionKey(table_name, {"horizon": source.horizon}))
        elif table_name == "fact_module_trace" and source.module_name:
            keys.append(PartitionKey(table_name, {"module_name": source.module_name}))
        elif table_name in {"fact_position_daily", "fact_signal_daily", "fact_market_bar", "fact_path_recursive"} and source.min_date:
            keys.append(PartitionKey(table_name, {"year": int(source.min_date[:4])}))
    seen = set()
    deduped = []
    for key in keys:
        label = key.label()
        if label not in seen:
            deduped.append(key)
            seen.add(label)
    return deduped


def partitions_from_frame(table_name: str, df: pl.DataFrame) -> list[PartitionKey]:
    spec = PARTITION_SPECS.get(table_name)
    if not spec or df.is_empty():
        return []
    df = _with_year(df, table_name)
    if not set(spec).issubset(df.columns):
        return []
    rows = df.select(spec).unique().to_dicts()
    return [PartitionKey(table_name, row) for row in rows]


def _filter_frame(df: pl.DataFrame, key: PartitionKey) -> pl.DataFrame:
    if key.key.get("ALL"):
        return df
    df = _with_year(df, key.table_name)
    expr = None
    for column, value in key.key.items():
        condition = pl.col(column).is_null() if value is None else pl.col(column) == value
        expr = condition if expr is None else expr & condition
    return df.filter(expr) if expr is not None else df


def _delete_sql(key: PartitionKey) -> tuple[str, dict[str, Any]]:
    if key.key.get("ALL"):
        return f"DELETE FROM {_qualified(key.table_name)}", {}
    clauses = []
    params: dict[str, Any] = {}
    for idx, (column, value) in enumerate(key.key.items()):
        param = f"p{idx}"
        if column == "year":
            date_column = TABLE_DATE_COLUMN.get(key.table_name, "date_value")
            clauses.append(f"{date_column} >= %({param})s::date AND {date_column} < %({param}_end)s::date")
            params[param] = f"{int(value)}-01-01"
            params[f"{param}_end"] = f"{int(value) + 1}-01-01"
        elif value is None:
            clauses.append(f"{column} IS NULL")
        else:
            clauses.append(f"{column} = %({param})s")
            params[param] = value
    return f"DELETE FROM {_qualified(key.table_name)} WHERE {' AND '.join(clauses)}", params


def _copy_frame(database_url: str, table_name: str, partition_filter: PartitionKey, load_df: pl.DataFrame, original_columns: list[str]) -> int:
    import psycopg

    drop_cols = [column for column in ["year"] if column in load_df.columns and column not in original_columns]
    if drop_cols:
        load_df = load_df.drop(drop_cols)
    if load_df.is_empty():
        return 0
    delete_sql, params = _delete_sql(partition_filter)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", newline="", encoding="utf-8", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        load_df.write_csv(tmp_path)
        columns = ", ".join(f'"{column}"' for column in load_df.columns)
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(delete_sql, params)
                with tmp_path.open("r", encoding="utf-8", newline="") as handle:
                    with cur.copy(f"COPY {_qualified(table_name)} ({columns}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)") as copy:
                        while chunk := handle.read(1024 * 1024):
                            copy.write(chunk)
                cur.execute(f"ANALYZE {_qualified(table_name)}")
            conn.commit()
    finally:
        tmp_path.unlink(missing_ok=True)
    return load_df.height


def replace_logical_partition(database_url: str, table_name: str, partition_filter: PartitionKey, parquet_path: Path) -> int:
    df = pl.read_parquet(parquet_path)
    if df.is_empty():
        return 0
    return _copy_frame(database_url, table_name, partition_filter, _filter_frame(df, partition_filter), df.columns)


def _partition_filters(key: PartitionKey) -> list[tuple[str, str, Any]]:
    if key.key.get("ALL"):
        return []
    return [(column, "=", value) for column, value in key.key.items()]


def _read_partition_frame(paths: DssPaths, table_name: str, key: PartitionKey, parquet_path: Path) -> tuple[pl.DataFrame, list[str]]:
    partitioned_root = paths.outputs_root / "parquet_partitioned" / table_name
    if partitioned_root.exists() and not key.key.get("ALL"):
        frame = read_partitioned_table(table_name, _partition_filters(key), root=paths.outputs_root / "parquet_partitioned")
        if not frame.is_empty():
            original_columns = [column for column in frame.columns if column != "year"]
            return _filter_frame(frame, key), original_columns
    if not parquet_path.exists():
        return pl.DataFrame(), []
    lf = pl.scan_parquet(parquet_path)
    df = lf.collect() if key.key.get("ALL") else _filter_frame(lf.collect(), key)
    return df, pl.read_parquet(parquet_path).columns


def _keys_from_plan(table_name: str, plan: Any) -> list[PartitionKey]:
    labels = (getattr(plan, "affected_partitions", {}) or {}).get(table_name) or []
    return [PartitionKey(table_name, parse_partition_label(table_name, label)) for label in labels]


def load_incremental_partitions(database_url: str, plan: Any, paths: DssPaths | None = None, run_id: str | None = None) -> dict[str, int]:
    paths = paths or get_paths()
    loaded: dict[str, int] = {}
    for table_name in plan.affected_tables:
        if table_name not in PARTITION_SPECS:
            continue
        parquet_path = _parquet_for(paths, f"dw.{table_name}")
        keys = _keys_from_plan(table_name, plan)
        if not keys and parquet_path.exists():
            keys = partitions_from_frame(table_name, pl.read_parquet(parquet_path))
        if not keys:
            continue
        manifest_rows = []
        for key in keys:
            frame, original_columns = _read_partition_frame(paths, table_name, key, parquet_path)
            if frame.is_empty():
                loaded[f"{table_name}:{key.label()}"] = 0
                continue
            rows = _copy_frame(database_url, table_name, key, frame, original_columns or frame.columns)
            loaded[f"{table_name}:{key.label()}"] = rows
            manifest_rows.append(
                {
                    "table_name": table_name,
                    "partition_key": key.label(),
                    "partition_type": "logical_postgres",
                    "row_count": rows,
                    "min_date": _min_bound(frame, table_name),
                    "max_date": _max_bound(frame, table_name),
                    "source_hash": None,
                    "status": "loaded",
                }
            )
        if run_id:
            log_partition_manifest(database_url, run_id, manifest_rows)
    return loaded


def _min_bound(df: pl.DataFrame, table_name: str) -> str | None:
    column = TABLE_DATE_COLUMN.get(table_name)
    if not column or column not in df.columns or df.is_empty():
        return None
    value = df.select(pl.col(column).min()).item()
    return str(value)[:10] if value is not None else None


def _max_bound(df: pl.DataFrame, table_name: str) -> str | None:
    column = TABLE_DATE_COLUMN.get(table_name)
    if not column or column not in df.columns or df.is_empty():
        return None
    value = df.select(pl.col(column).max()).item()
    return str(value)[:10] if value is not None else None


def analyze_affected_tables(database_url: str, plan: ExecutionPlan) -> None:
    if not database_url or not plan.affected_tables:
        return
    try:
        import psycopg

        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                for table in plan.affected_tables:
                    if table in PARTITION_SPECS:
                        cur.execute(f"ANALYZE {_qualified(table)}")
            conn.commit()
    except Exception:
        return
