from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DssPaths:
    repo_root: Path
    phase_root: Path
    baseline_root: Path
    extended_root: Path
    outputs_root: Path
    parquet_root: Path
    reports_root: Path
    logs_root: Path
    demo_root: Path
    sql_root: Path


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        if (path / ".git").exists() and (path / "baseline").exists() and (path / "research").exists():
            return path
    return Path(__file__).resolve().parents[3]


def get_paths() -> DssPaths:
    phase_root = Path(__file__).resolve().parents[1]
    repo_root = find_repo_root(phase_root)
    outputs_root = phase_root / "outputs"
    return DssPaths(
        repo_root=repo_root,
        phase_root=phase_root,
        baseline_root=repo_root / "baseline" / "mahoraga14_3_baseline",
        extended_root=repo_root / "research" / "mahoraga14_3_extended_analysis",
        outputs_root=outputs_root,
        parquet_root=outputs_root / "parquet",
        reports_root=outputs_root / "reports",
        logs_root=outputs_root / "logs",
        demo_root=outputs_root / "demo_data",
        sql_root=phase_root / "sql",
    )


def ensure_output_dirs(paths: DssPaths | None = None) -> DssPaths:
    paths = paths or get_paths()
    for directory in [paths.outputs_root, paths.parquet_root, paths.reports_root, paths.logs_root, paths.demo_root]:
        directory.mkdir(parents=True, exist_ok=True)
    return paths
