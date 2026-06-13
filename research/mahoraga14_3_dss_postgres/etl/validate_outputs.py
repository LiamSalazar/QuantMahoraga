from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from .paths import DssPaths, get_paths

REQUIRED_TABLES = [
    "dim_date",
    "dim_candidate",
    "dim_universe",
    "dim_fold",
    "fact_decision_state",
    "fact_position_daily",
    "fact_module_trace",
    "fact_outcome",
    "fact_candidate_metric",
    "fact_whatif",
]


def _table_path(paths: DssPaths, table: str) -> Path:
    family = "dimensions" if table.startswith("dim_") else "facts"
    return paths.parquet_root / family / f"{table}.parquet"


def validate(paths: DssPaths | None = None) -> dict[str, Any]:
    paths = paths or get_paths()
    checks: list[dict[str, Any]] = []
    for table in REQUIRED_TABLES:
        path = _table_path(paths, table)
        exists = path.exists()
        rows = int(pl.scan_parquet(path).select(pl.len()).collect().item()) if exists else 0
        checks.append(
            {
                "check_name": f"{table}_exists_and_has_rows",
                "severity": "error",
                "passed": exists and rows > 0,
                "observed_value": rows,
                "expected_value": "> 0",
                "table_name": table,
            }
        )
    duplicate_specs = {
        "fact_position_daily": ["date_value", "candidate_id", "fold", "universe_id", "ticker", "run_id"],
        "fact_outcome": ["decision_date", "candidate_id", "fold", "universe_id", "horizon", "run_id"],
        "fact_decision_state": ["date_value", "candidate_id", "fold", "universe_id", "run_id"],
    }
    for table, key in duplicate_specs.items():
        path = _table_path(paths, table)
        if not path.exists():
            continue
        df = pl.read_parquet(path)
        dupes = df.group_by(key).len().filter(pl.col("len") > 1).height if set(key).issubset(df.columns) else -1
        checks.append(
            {
                "check_name": f"{table}_duplicate_key_check",
                "severity": "error",
                "passed": dupes == 0,
                "observed_value": dupes,
                "expected_value": "0",
                "table_name": table,
            }
        )
    whatif_path = _table_path(paths, "fact_whatif")
    if whatif_path.exists():
        whatif = pl.read_parquet(whatif_path)
        synthetic = whatif.filter(pl.col("source_artifact") == "demo_synthetic_whatif_grid")
        unflagged = synthetic.filter(~pl.col("demo_mode")).height if not synthetic.is_empty() else 0
        checks.append(
            {
                "check_name": "demo_whatif_rows_flagged",
                "severity": "error",
                "passed": unflagged == 0,
                "observed_value": unflagged,
                "expected_value": "0",
                "table_name": "fact_whatif",
            }
        )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": all(item["passed"] for item in checks if item["severity"] == "error"),
        "checks": checks,
    }
    paths.reports_root.mkdir(parents=True, exist_ok=True)
    (paths.reports_root / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pl.DataFrame(checks).write_csv(paths.reports_root / "validation_report.csv")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DSS parquet outputs.")
    parser.parse_args()
    report = validate()
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"])}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
