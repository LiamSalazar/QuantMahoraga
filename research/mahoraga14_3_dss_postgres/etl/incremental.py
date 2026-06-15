from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from .adaptive_planner import ExecutionPlan
from .load_postgres import _parquet_for
from .paths import DssPaths, get_paths
from .source_manifest import SourceDiff


@dataclass(frozen=True)
class PartitionKey:
    table_name: str
    key: dict[str, Any]

    def label(self) -> str:
        return "/".join(f"{name}={value}" for name, value in self.key.items())


PARTITION_SPECS: dict[str, list[str]] = {
    "fact_position_daily": ["year", "fold", "candidate_id", "universe_id"],
    "fact_signal_daily": ["year", "fold", "candidate_id", "universe_id"],
    "fact_market_bar": ["year"],
    "fact_outcome": ["horizon", "fold", "candidate_id", "universe_id"],
    "fact_module_trace": ["module_name", "fold", "candidate_id", "universe_id"],
    "fact_whatif": ["scenario_id", "fold", "horizon", "demo_mode"],
    "fact_path_recursive": ["year", "candidate_id", "fold"],
}

TABLE_DATE_COLUMN = {
    "fact_position_daily": "date_value",
    "fact_signal_daily": "date_value",
    "fact_market_bar": "date_value",
    "fact_path_recursive": "date_value",
    "fact_outcome": "decision_date",
    "fact_module_trace": "date_value",
}


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
    df = _with_year(df, key.table_name)
    expr = None
    for column, value in key.key.items():
        condition = pl.col(column).is_null() if value is None else pl.col(column) == value
        expr = condition if expr is None else expr & condition
    return df.filter(expr) if expr is not None else df


def _delete_sql(key: PartitionKey) -> tuple[str, dict[str, Any]]:
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


def replace_logical_partition(database_url: str, table_name: str, partition_filter: PartitionKey, parquet_path: Path) -> int:
    import psycopg

    df = pl.read_parquet(parquet_path)
    if df.is_empty():
        return 0
    load_df = _filter_frame(df, partition_filter)
    drop_cols = [column for column in ["year"] if column in load_df.columns and column not in pl.read_parquet(parquet_path, n_rows=0).columns]
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


def build_incremental_tables(config: Any, paths: DssPaths, plan: ExecutionPlan) -> dict[str, pl.DataFrame]:
    from .run_pipeline import build_all

    tables, _, _ = build_all(config, paths)
    wanted = set(plan.affected_tables)
    return {name: frame for name, frame in tables.items() if name in wanted}


def load_incremental_partitions(database_url: str, plan: ExecutionPlan, paths: DssPaths | None = None) -> dict[str, int]:
    paths = paths or get_paths()
    loaded: dict[str, int] = {}
    for table_name in plan.affected_tables:
        if table_name not in PARTITION_SPECS:
            continue
        parquet_path = _parquet_for(paths, f"dw.{table_name}")
        if not parquet_path.exists():
            continue
        df = pl.read_parquet(parquet_path)
        for key in partitions_from_frame(table_name, df):
            loaded[f"{table_name}:{key.label()}"] = replace_logical_partition(database_url, table_name, key, parquet_path)
    return loaded


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
