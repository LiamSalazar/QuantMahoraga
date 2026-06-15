from __future__ import annotations

import argparse
import time

from .config import make_config
from .control_plane import _safe_execute
from .mart_dependencies import ALL_MARTS, marts_for_tables


def _has_concurrent_index(cur, mart_name: str) -> bool:
    schema, relation = mart_name.split(".", 1)
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_index i ON i.indrelid = c.oid
            WHERE n.nspname = %(schema)s
              AND c.relname = %(relation)s
              AND i.indisunique
              AND i.indisvalid
              AND i.indpred IS NULL
              AND i.indexprs IS NULL
        ) AS ok
        """,
        {"schema": schema, "relation": relation},
    )
    return bool(cur.fetchone()["ok"])


def _refresh_one(database_url: str, mart_name: str, run_id: str | None, strategy: str) -> None:
    import psycopg
    from psycopg.rows import dict_row

    started = time.perf_counter()
    status = "COMPLETED"
    error = None
    rows_after = None
    refresh_strategy = strategy
    try:
        with psycopg.connect(database_url, row_factory=dict_row, autocommit=True) as conn:
            with conn.cursor() as cur:
                can_concurrent = _has_concurrent_index(cur, mart_name)
                if can_concurrent:
                    try:
                        cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {mart_name}")
                        refresh_strategy = "concurrent" if strategy == "full" else f"{strategy}:concurrent"
                    except Exception as exc:
                        error = f"concurrent refresh failed, fallback normal: {exc}"
                        cur.execute(f"REFRESH MATERIALIZED VIEW {mart_name}")
                        refresh_strategy = "fallback"
                else:
                    cur.execute(f"REFRESH MATERIALIZED VIEW {mart_name}")
                    refresh_strategy = "normal" if strategy == "full" else f"{strategy}:normal"
                cur.execute(f"SELECT COUNT(*) AS count FROM {mart_name}")
                rows_after = int(cur.fetchone()["count"])
    except Exception as exc:
        status = "FAILED"
        error = str(exc)
        raise
    finally:
        if run_id:
            _safe_execute(
                database_url,
                """
                INSERT INTO oltp.mart_refresh_log
                    (run_id, mart_name, refresh_strategy, status, started_at, finished_at,
                     duration_ms, rows_after_refresh, error_message)
                VALUES
                    (%(run_id)s, %(mart_name)s, %(strategy)s, %(status)s,
                     now() - (%(duration_ms)s::text || ' milliseconds')::interval, now(),
                     %(duration_ms)s, %(rows_after)s, %(error)s)
                """,
                {
                    "run_id": run_id,
                    "mart_name": mart_name,
                    "strategy": refresh_strategy,
                    "status": status,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    "rows_after": rows_after,
                    "error": error,
                },
            )


def refresh(
    database_url: str | None = None,
    *,
    strategy: str = "full",
    changed_tables: list[str] | None = None,
    run_id: str | None = None,
) -> list[str]:
    config = make_config(mode="postgres", database_url=database_url)
    if not config.database_url:
        raise RuntimeError("DATABASE_URL is required to refresh materialized views")
    if strategy == "dependency" and changed_tables:
        marts = marts_for_tables(changed_tables)
    elif strategy == "fast":
        marts = [
            "mart.mv_scorecard_candidate",
            "mart.mv_whatif_grid",
            "mart.mv_query_performance",
        ]
    else:
        marts = ALL_MARTS
    for mart_name in marts:
        _refresh_one(config.database_url, mart_name, run_id, strategy)
    return marts


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Mahoraga DSS materialized views.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--strategy", choices=["full", "dependency", "fast"], default="full")
    parser.add_argument("--changed-tables", default="", help="Comma-separated fact names for dependency refresh.")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    changed_tables = [item.strip() for item in args.changed_tables.split(",") if item.strip()]
    marts = refresh(args.database_url, strategy=args.strategy, changed_tables=changed_tables, run_id=args.run_id)
    print(f"refreshed materialized views: {', '.join(marts)}")


if __name__ == "__main__":
    main()
