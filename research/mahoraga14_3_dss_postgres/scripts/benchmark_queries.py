from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from etl.paths import ensure_output_dirs

QUERIES = {
    "command_center_scorecard": "SELECT * FROM mart.mv_scorecard_candidate ORDER BY sharpe DESC NULLS LAST LIMIT 50",
    "decision_replay": """
        SELECT d.*, o.horizon, o.realized_return
        FROM dw.fact_decision_state d
        LEFT JOIN dw.fact_outcome o
          ON o.candidate_id=d.candidate_id AND o.universe_id=d.universe_id
         AND o.fold=d.fold AND o.decision_date=d.date_value AND o.horizon=20
        WHERE d.candidate_id='B1.05_C1.10_L1.10_R1.05'
        ORDER BY d.date_value DESC
        LIMIT 100
    """,
    "whatif_grid": "SELECT * FROM mart.mv_whatif_grid WHERE universe_id='base_universe_12' AND horizon=20 ORDER BY robust_score DESC NULLS LAST LIMIT 500",
    "ticker_contribution": "SELECT * FROM mart.mv_ticker_contribution WHERE candidate_id='B1.05_C1.10_L1.10_R1.05' ORDER BY total_pnl_contribution DESC NULLS LAST LIMIT 100",
    "regime_behavior": "SELECT * FROM mart.mv_regime_behavior WHERE candidate_id='B1.05_C1.10_L1.10_R1.05' ORDER BY observations DESC LIMIT 100",
    "module_effectiveness": "SELECT * FROM mart.mv_module_effectiveness WHERE candidate_id='B1.05_C1.10_L1.10_R1.05' ORDER BY module_name, horizon LIMIT 200",
    "olap_preset": """
        SELECT fold, avg(avg_realized_return) AS avg_realized_return, avg(avg_alpha_vs_qqq) AS avg_alpha_vs_qqq
        FROM mart.mv_performance_by_fold
        WHERE candidate_id='B1.05_C1.10_L1.10_R1.05'
        GROUP BY fold
        ORDER BY avg_alpha_vs_qqq DESC NULLS LAST
    """,
    "execution_evidence": """
        SELECT endpoint, source_relation, count(*) AS query_count, avg(elapsed_ms) AS avg_elapsed_ms
        FROM oltp.dss_query_log
        GROUP BY endpoint, source_relation
        ORDER BY query_count DESC NULLS LAST
        LIMIT 100
    """,
}


def _walk_plan(node: dict[str, Any], relations: set[str], scan_types: set[str]) -> None:
    if relation := node.get("Relation Name"):
        relations.add(relation)
    if node_type := node.get("Node Type"):
        if "Scan" in node_type:
            scan_types.add(node_type)
    for child in node.get("Plans", []) or []:
        _walk_plan(child, relations, scan_types)


def _summarize_plan(plan_json: list[dict[str, Any]]) -> dict[str, Any]:
    root = plan_json[0]
    plan = root["Plan"]
    relations: set[str] = set()
    scan_types: set[str] = set()
    _walk_plan(plan, relations, scan_types)
    return {
        "planning_time_ms": root.get("Planning Time"),
        "execution_time_ms": root.get("Execution Time"),
        "rows_returned_estimate": plan.get("Actual Rows"),
        "shared_hit_blocks": plan.get("Shared Hit Blocks", 0),
        "shared_read_blocks": plan.get("Shared Read Blocks", 0),
        "relations_scanned": ",".join(sorted(relations)),
        "scan_types": ",".join(sorted(scan_types)),
        "uses_index_scan": any("Index" in scan for scan in scan_types),
        "uses_seq_scan": any(scan == "Seq Scan" for scan in scan_types),
    }


def run_benchmarks(database_url: str, smoke: bool = False) -> list[dict[str, Any]]:
    import psycopg
    from psycopg.rows import dict_row

    paths = ensure_output_dirs()
    bench_root = paths.benchmarks_root
    plans_root = bench_root / "query_plans"
    plans_root.mkdir(parents=True, exist_ok=True)
    selected = dict(list(QUERIES.items())[:3]) if smoke else QUERIES
    rows = []
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            for name, sql in selected.items():
                cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}")
                plan_json = cur.fetchone()["QUERY PLAN"]
                (plans_root / f"{name}.json").write_text(json.dumps(plan_json, indent=2, default=str), encoding="utf-8")
                rows.append({"query_name": name, **_summarize_plan(plan_json)})
    summary_path = bench_root / "query_benchmark_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["query_name"])
        writer.writeheader()
        writer.writerows(rows)
    report = bench_root / "query_benchmark_report.md"
    lines = ["# Query Benchmark Report", "", "| Query | Planning ms | Execution ms | Scans | Relations |", "| --- | ---: | ---: | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['query_name']} | {row['planning_time_ms']} | {row['execution_time_ms']} | {row['scan_types']} | {row['relations_scanned']} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark representative DSS Postgres queries.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    database_url = args.database_url or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    rows = run_benchmarks(database_url, smoke=args.smoke)
    print(json.dumps({"queries": len(rows), "summary": str(ensure_output_dirs().benchmarks_root / "query_benchmark_summary.csv")}, indent=2))


if __name__ == "__main__":
    main()
