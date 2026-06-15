from __future__ import annotations

import argparse
import json
import os
from typing import Any

from etl.paths import ensure_output_dirs

QUERIES = {
    "fact_position_daily_date_range": """
        SELECT count(*)
        FROM dw.fact_position_daily
        WHERE date_value >= DATE '2020-01-01' AND date_value < DATE '2021-01-01'
    """,
    "fact_outcome_horizon": "SELECT count(*) FROM dw.fact_outcome WHERE horizon = 20",
    "fact_module_trace_module": "SELECT count(*) FROM dw.fact_module_trace WHERE module_name = 'risk_backoff_layer_v2'",
}


def _collect_nodes(node: dict[str, Any], nodes: list[dict[str, Any]]) -> None:
    nodes.append(node)
    for child in node.get("Plans", []) or []:
        _collect_nodes(child, nodes)


def _evidence(plan_json: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    _collect_nodes(plan_json[0]["Plan"], nodes)
    relations = sorted({node.get("Relation Name") for node in nodes if node.get("Relation Name")})
    return {
        "relations": relations,
        "partition_like_relations": [rel for rel in relations if rel and ("fact_position_daily_" in rel or "fact_outcome_h" in rel or "fact_module_trace_" in rel)],
        "scan_types": sorted({node.get("Node Type") for node in nodes if node.get("Node Type") and "Scan" in node.get("Node Type")}),
        "execution_time_ms": plan_json[0].get("Execution Time"),
    }


def run_demo(database_url: str) -> list[dict[str, Any]]:
    import psycopg
    from psycopg.rows import dict_row

    paths = ensure_output_dirs()
    out = []
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            for name, sql in QUERIES.items():
                cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}")
                plan_json = cur.fetchone()["QUERY PLAN"]
                evidence = _evidence(plan_json)
                out.append({"query": name, **evidence})
    lines = ["# Partition Pruning Demo", ""]
    for row in out:
        lines.append(f"## {row['query']}")
        lines.append(f"- execution_time_ms: `{row['execution_time_ms']}`")
        lines.append(f"- scan_types: `{', '.join(row['scan_types'])}`")
        lines.append(f"- relations: `{', '.join(row['relations'])}`")
        if row["partition_like_relations"]:
            lines.append(f"- partition evidence: `{', '.join(row['partition_like_relations'])}`")
        else:
            lines.append("- partition evidence: plan did not expose specific partition relations clearly.")
        lines.append("")
    (paths.benchmarks_root / "partition_pruning_report.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain partition pruning evidence for partitioned facts.")
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()
    database_url = args.database_url or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    print(json.dumps(run_demo(database_url), indent=2, default=str))


if __name__ == "__main__":
    main()
