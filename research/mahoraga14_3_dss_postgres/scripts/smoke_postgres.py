from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any

SCHEMAS = ["oltp", "dw", "mart", "audit"]

REQUIRED_TABLES = [
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
]

REQUIRED_VIEWS = [
    "mart.mv_scorecard_candidate",
    "mart.mv_decision_replay",
    "mart.mv_module_effectiveness",
    "mart.mv_ticker_contribution",
    "mart.mv_regime_behavior",
    "mart.mv_whatif_grid",
]


def smoke(database_url: str | None = None) -> dict[str, Any]:
    import psycopg

    from psycopg.rows import dict_row

    database_url = database_url or os.getenv("DATABASE_URL")
    if not database_url:
        return {
            "passed": False,
            "checked_tables": REQUIRED_TABLES,
            "row_counts": {},
            "checked_views": REQUIRED_VIEWS,
            "failures": ["DATABASE_URL is required"],
        }

    failures: list[str] = []
    row_counts: dict[str, int] = {}
    view_counts: dict[str, int] = {}

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.execute(
                """
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name = ANY(%(schemas)s)
                """,
                {"schemas": SCHEMAS},
            )
            found_schemas = {row["schema_name"] for row in cur.fetchall()}
            for schema in SCHEMAS:
                if schema not in found_schemas:
                    failures.append(f"Missing schema: {schema}")

            for table in REQUIRED_TABLES:
                cur.execute(f"SELECT COUNT(*) AS count FROM {table}")
                count = int(cur.fetchone()["count"])
                row_counts[table] = count
                if count <= 0:
                    failures.append(f"{table}: expected > 0 rows")

            for view in REQUIRED_VIEWS:
                cur.execute(f"SELECT COUNT(*) AS count FROM {view}")
                count = int(cur.fetchone()["count"])
                view_counts[view] = count
                if count <= 0:
                    failures.append(f"{view}: expected > 0 rows")

    return {
        "passed": not failures,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checked_tables": REQUIRED_TABLES,
        "row_counts": row_counts,
        "checked_views": REQUIRED_VIEWS,
        "view_counts": view_counts,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test a Mahoraga DSS Postgres database.")
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()
    result = smoke(args.database_url)
    print(json.dumps(result, indent=2, default=str))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
