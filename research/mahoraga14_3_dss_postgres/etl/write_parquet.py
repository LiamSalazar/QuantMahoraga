from __future__ import annotations

import json
from datetime import datetime, timezone

import polars as pl

from .paths import DssPaths, ensure_output_dirs


def write_tables(tables: dict[str, pl.DataFrame], paths: DssPaths, run_id: str) -> dict[str, int]:
    ensure_output_dirs(paths)
    row_counts: dict[str, int] = {}
    for name, frame in tables.items():
        target_dir = paths.parquet_root / ("dimensions" if name.startswith("dim_") else "facts" if name.startswith("fact_") else "oltp")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{name}.parquet"
        frame.write_parquet(target)
        row_counts[name] = frame.height
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_counts": row_counts,
        "total_rows": sum(row_counts.values()),
    }
    (paths.reports_root / "parquet_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return row_counts
