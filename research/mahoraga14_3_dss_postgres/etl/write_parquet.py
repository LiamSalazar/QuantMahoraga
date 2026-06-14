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
    manifest_path = paths.reports_root / "parquet_manifest.json"
    manifest_counts: dict[str, int] = {}
    if manifest_path.exists():
        try:
            manifest_counts = {str(name): int(count) for name, count in json.loads(manifest_path.read_text(encoding="utf-8")).get("row_counts", {}).items()}
        except Exception:
            manifest_counts = {}
    manifest_counts.update(row_counts)
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_counts": manifest_counts,
        "total_rows": sum(manifest_counts.values()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return row_counts
