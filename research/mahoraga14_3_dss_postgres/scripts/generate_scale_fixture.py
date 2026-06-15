from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from etl.paths import ensure_output_dirs


def generate_fixture(target_rows: int) -> Path:
    paths = ensure_output_dirs()
    label = f"scale_{target_rows // 1_000_000}m" if target_rows >= 1_000_000 else f"scale_{target_rows}"
    root = paths.outputs_root / "scale_fixtures" / label
    root.mkdir(parents=True, exist_ok=True)
    batch_size = min(1_000_000, max(100_000, target_rows))
    written = 0
    part = 0
    while written < target_rows:
        n = min(batch_size, target_rows - written)
        df = pl.DataFrame(
            {
                "benchmark_mode": [True] * n,
                "row_id": range(written, written + n),
                "candidate_id": [f"BENCH_C{idx % 64:02d}" for idx in range(written, written + n)],
                "universe_id": [f"bench_universe_{idx % 8}" for idx in range(written, written + n)],
                "fold": [(idx % 5) + 1 for idx in range(written, written + n)],
                "horizon": [[1, 5, 20, 60][idx % 4] for idx in range(written, written + n)],
                "metric_value": [(idx % 1000) / 1000 for idx in range(written, written + n)],
            }
        )
        df.write_parquet(root / f"part-{part:05d}.parquet")
        written += n
        part += 1
    manifest = {
        "benchmark_mode": True,
        "target_rows": target_rows,
        "rows_written": written,
        "parts": part,
        "not_research_evidence": True,
        "usage": "Engineering throughput/query benchmark only. Do not load into the standard DSS pipeline.",
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark-only scale fixture Parquet data.")
    parser.add_argument("--target-rows", type=int, required=True)
    args = parser.parse_args()
    if args.target_rows <= 0:
        raise SystemExit("--target-rows must be positive")
    root = generate_fixture(args.target_rows)
    print(f"wrote benchmark-only scale fixture: {root}")


if __name__ == "__main__":
    main()
