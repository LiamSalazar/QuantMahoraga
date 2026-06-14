from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from .config import PHASE
from .lfs_guard import assert_not_lfs_pointer
from .paths import DssPaths, ensure_output_dirs, get_paths


@dataclass(frozen=True)
class ExpectedArtifact:
    role: str
    relative_path: str
    required: bool


EXPECTED_ARTIFACTS = [
    ExpectedArtifact("baseline_output", "baseline/mahoraga14_3_baseline/outputs/active_return_vs_qqq_official.csv", True),
    ExpectedArtifact("baseline_output", "baseline/mahoraga14_3_baseline/outputs/fold_summary_official.csv", True),
    ExpectedArtifact("baseline_output", "baseline/mahoraga14_3_baseline/outputs/stitched_comparison_official.csv", True),
    ExpectedArtifact("baseline_audit", "baseline/mahoraga14_3_baseline/audit/allocator_cash_drag_official.csv", True),
    ExpectedArtifact("baseline_config", "baseline/mahoraga14_3_baseline/config/OFFICIAL_FREEZE.json", True),
    ExpectedArtifact("extended_config", "research/mahoraga14_3_extended_analysis/configs/analysis_config.json", True),
    ExpectedArtifact("extended_summary", "research/mahoraga14_3_extended_analysis/outputs/extended_multiplier_robustness/extended_multiplier_summary.csv", True),
    ExpectedArtifact("extended_summary", "research/mahoraga14_3_extended_analysis/outputs/extended_multiplier_robustness/extended_multiplier_fold_summary.csv", True),
    ExpectedArtifact("extended_summary", "research/mahoraga14_3_extended_analysis/outputs/universe_robustness/universe_robustness_summary.csv", True),
    ExpectedArtifact("audit_cube", "research/mahoraga14_3_extended_analysis/outputs/audit_cube/decision_date_cube.parquet", True),
    ExpectedArtifact("audit_cube", "research/mahoraga14_3_extended_analysis/outputs/audit_cube/position_cube.parquet", True),
    ExpectedArtifact("audit_cube", "research/mahoraga14_3_extended_analysis/outputs/audit_cube/module_trace_cube.parquet", True),
    ExpectedArtifact("audit_cube", "research/mahoraga14_3_extended_analysis/outputs/audit_cube/outcome_cube.parquet", True),
    ExpectedArtifact("audit_cube", "research/mahoraga14_3_extended_analysis/outputs/audit_cube/market_context_cube.parquet", True),
    ExpectedArtifact("audit_dictionary", "research/mahoraga14_3_extended_analysis/outputs/audit_cube/cube_dictionary.md", False),
    ExpectedArtifact("extended_manifest", "research/mahoraga14_3_extended_analysis/outputs/manifests/file_manifest.csv", False),
]


def _schema(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.suffix.lower() not in {".csv", ".parquet"}:
        return []
    try:
        assert_not_lfs_pointer(path)
        if path.suffix.lower() == ".parquet":
            schema = pl.scan_parquet(path).collect_schema()
        else:
            schema = pl.scan_csv(path, infer_schema_length=200).collect_schema()
        return [{"name": name, "dtype": str(dtype)} for name, dtype in schema.items()]
    except Exception as exc:  # pragma: no cover - defensive inventory
        return [{"name": "__schema_error__", "dtype": str(exc)}]


def _row_count(path: Path) -> int | None:
    if not path.exists() or path.suffix.lower() not in {".csv", ".parquet"}:
        return None
    try:
        assert_not_lfs_pointer(path)
        if path.suffix.lower() == ".parquet":
            return int(pl.scan_parquet(path).select(pl.len()).collect().item())
        return int(pl.scan_csv(path, infer_schema_length=200).select(pl.len()).collect().item())
    except Exception:
        return None


def discover(paths: DssPaths | None = None, run_id: str = "manual") -> pl.DataFrame:
    paths = paths or get_paths()
    rows: list[dict[str, Any]] = []
    for artifact in EXPECTED_ARTIFACTS:
        path = paths.repo_root / artifact.relative_path
        schema = _schema(path)
        rows.append(
            {
                "run_id": run_id,
                "artifact_role": artifact.role,
                "relative_path": artifact.relative_path,
                "storage_format": path.suffix.lower().lstrip(".") or "directory",
                "exists_flag": path.exists(),
                "row_count": _row_count(path),
                "column_count": len(schema),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
                "required_flag": artifact.required,
                "demo_mode": False,
                "schema_json": json.dumps(schema),
                "phase": PHASE,
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover Mahoraga artifacts for the DSS layer.")
    parser.add_argument("--run-id", default="manual")
    args = parser.parse_args()
    paths = ensure_output_dirs()
    inventory = discover(paths, run_id=args.run_id)
    out = paths.reports_root / "artifact_inventory.csv"
    inventory.write_csv(out)
    print(f"wrote {out} ({inventory.height} artifacts)")


if __name__ == "__main__":
    main()
