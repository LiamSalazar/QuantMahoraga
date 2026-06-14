from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from .load_postgres import LOAD_ORDER, _parquet_for
from .paths import get_paths

REQUIRED_MATERIALIZED_VIEWS = [
    "mart.mv_scorecard_candidate",
    "mart.mv_decision_replay",
    "mart.mv_module_effectiveness",
    "mart.mv_ticker_contribution",
    "mart.mv_regime_behavior",
    "mart.mv_whatif_grid",
    "mart.mv_query_performance",
]


def _parquet_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def validate(database_url: str | None = None) -> dict[str, Any]:
    import psycopg

    from psycopg.rows import dict_row

    database_url = database_url or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for Postgres validation")

    paths = get_paths()
    failures: list[str] = []
    table_counts: dict[str, int] = {}
    parquet_counts: dict[str, int | None] = {}
    view_counts: dict[str, int] = {}

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            for table in LOAD_ORDER:
                cur.execute(f"SELECT COUNT(*) AS count FROM {table}")
                actual = int(cur.fetchone()["count"])
                table_counts[table] = actual
                parquet_count = _parquet_rows(_parquet_for(paths, table))
                parquet_counts[table] = parquet_count
                if parquet_count is not None and actual != parquet_count:
                    failures.append(f"{table}: Postgres count {actual} != Parquet count {parquet_count}")
                if table in {
                    "oltp.research_run",
                    "oltp.data_snapshot",
                    "oltp.artifact_inventory",
                    "oltp.candidate_grid",
                    "dw.dim_candidate",
                    "dw.dim_universe",
                    "dw.fact_decision_state",
                    "dw.fact_position_daily",
                    "dw.fact_module_trace",
                    "dw.fact_outcome",
                    "dw.fact_whatif",
                } and actual <= 0:
                    failures.append(f"{table}: expected > 0 rows")

            for view in REQUIRED_MATERIALIZED_VIEWS:
                cur.execute(f"SELECT COUNT(*) AS count FROM {view}")
                count = int(cur.fetchone()["count"])
                view_counts[view] = count
                if view != "mart.mv_query_performance" and count <= 0:
                    failures.append(f"{view}: expected > 0 rows")

            cur.execute(
                """
                SELECT endpoint, backend, count(*) AS query_count, max(created_at) AS last_seen_at
                FROM oltp.dss_query_log
                GROUP BY endpoint, backend
                ORDER BY last_seen_at DESC NULLS LAST
                LIMIT 20
                """
            )
            query_performance_rows = cur.fetchall()

    return {
        "passed": not failures,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checked_tables": LOAD_ORDER,
        "table_counts": table_counts,
        "parquet_counts": parquet_counts,
        "checked_views": REQUIRED_MATERIALIZED_VIEWS,
        "view_counts": view_counts,
        "query_performance": {"count": len(query_performance_rows), "rows": query_performance_rows},
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DSS Postgres load against generated Parquet outputs.")
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()
    report = validate(args.database_url)
    print(json.dumps(report, indent=2, default=str))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
