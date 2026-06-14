from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import polars as pl

from .config import RuntimeConfig, make_config
from .paths import DssPaths, get_paths

LOAD_ORDER = [
    "oltp.research_run",
    "oltp.data_snapshot",
    "oltp.artifact_inventory",
    "oltp.candidate_grid",
    "dw.dim_date",
    "dw.dim_asset",
    "dw.dim_candidate",
    "dw.dim_universe",
    "dw.dim_fold",
    "dw.dim_module",
    "dw.dim_regime",
    "dw.dim_horizon",
    "dw.dim_metric",
    "dw.dim_scenario",
    "dw.fact_market_bar",
    "dw.fact_signal_daily",
    "dw.fact_decision_state",
    "dw.fact_position_daily",
    "dw.fact_module_trace",
    "dw.fact_outcome",
    "dw.fact_candidate_metric",
    "dw.fact_robustness_surface",
    "dw.fact_cost_sensitivity",
    "dw.fact_universe_sensitivity",
    "dw.fact_whatif",
    "dw.fact_path_recursive",
    "dw.fact_data_quality",
]


def _parquet_for(paths: DssPaths, qualified_table: str) -> Path:
    schema, table = qualified_table.split(".", 1)
    if schema == "oltp":
        family = "oltp"
    elif schema == "dw" and table.startswith("dim_"):
        family = "dimensions"
    elif schema == "dw" and table.startswith("fact_"):
        family = "facts"
    else:
        family = schema
    return paths.parquet_root / family / f"{table}.parquet"


def execute_sql_file(database_url: str, sql_path: Path) -> None:
    import psycopg

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql_path.read_text(encoding="utf-8"))
        conn.commit()


def bootstrap_schema(database_url: str, paths: DssPaths) -> None:
    for name in [
        "001_create_schemas.sql",
        "002_create_oltp.sql",
        "003_create_dimensions.sql",
        "004_create_facts.sql",
        "005_create_partitions.sql",
        "006_create_indexes.sql",
        "007_create_materialized_views.sql",
    ]:
        execute_sql_file(database_url, paths.sql_root / name)


def truncate_loaded_tables(database_url: str) -> None:
    import psycopg

    table_list = ", ".join(LOAD_ORDER)
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {table_list} RESTART IDENTITY CASCADE")
        conn.commit()


def _copy_table(database_url: str, qualified_table: str, parquet_path: Path) -> int:
    import psycopg

    if not parquet_path.exists():
        return 0
    df = pl.read_parquet(parquet_path)
    if df.is_empty():
        return 0
    with tempfile.NamedTemporaryFile("w", suffix=".csv", newline="", encoding="utf-8", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.write_csv(tmp_path)
        columns = ", ".join(f'"{column}"' for column in df.columns)
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                with tmp_path.open("r", encoding="utf-8", newline="") as handle:
                    with cur.copy(f"COPY {qualified_table} ({columns}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)") as copy:
                        while chunk := handle.read(1024 * 1024):
                            copy.write(chunk)
            conn.commit()
    finally:
        tmp_path.unlink(missing_ok=True)
    return df.height


def load_all(config: RuntimeConfig, paths: DssPaths | None = None, bootstrap: bool = True, truncate: bool = False) -> dict[str, int]:
    paths = paths or get_paths()
    if not config.database_url:
        raise RuntimeError("DATABASE_URL is required for Postgres mode")
    if bootstrap:
        bootstrap_schema(config.database_url, paths)
    if truncate:
        truncate_loaded_tables(config.database_url)
    counts: dict[str, int] = {}
    for qualified_table in LOAD_ORDER:
        counts[qualified_table] = _copy_table(config.database_url, qualified_table, _parquet_for(paths, qualified_table))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Load DSS parquet outputs into Postgres.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--truncate", action="store_true")
    args = parser.parse_args()
    config = make_config(mode="postgres", database_url=args.database_url)
    counts = load_all(config, bootstrap=not args.no_bootstrap, truncate=args.truncate)
    for table, rows in counts.items():
        print(f"{table}: {rows}")


if __name__ == "__main__":
    main()
