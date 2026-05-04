from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoLayout:
    repo_root: Path
    baseline_root: Path
    research_root: Path
    shared_root: Path
    docs_root: Path
    data_cache_root: Path

    def baseline_package_root(self, name: str) -> Path:
        return self.baseline_root / name

    def baseline_src_root(self, name: str) -> Path:
        return self.baseline_package_root(name) / "src"

    def baseline_outputs_root(self, name: str) -> Path:
        return self.baseline_package_root(name) / "outputs"

    def baseline_audit_root(self, name: str) -> Path:
        return self.baseline_package_root(name) / "audit"

    def baseline_docs_root(self, name: str) -> Path:
        return self.baseline_package_root(name) / "docs"

    def baseline_manifests_root(self, name: str) -> Path:
        return self.baseline_package_root(name) / "manifests"

    def baseline_paper_pack_root(self, name: str) -> Path:
        return self.baseline_package_root(name) / "paper_pack"

    def baseline_tests_root(self, name: str) -> Path:
        return self.baseline_package_root(name) / "tests"


def discover_repo_layout(start: str | Path | None = None) -> RepoLayout:
    cursor = Path(start or __file__).resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for path in (cursor, *cursor.parents):
        has_repo_markers = (
            (path / ".git").exists()
            or (
                (path / "baseline").exists()
                and (path / "research").exists()
                and (path / "shared").exists()
            )
        )
        if has_repo_markers:
            return RepoLayout(
                repo_root=path,
                baseline_root=path / "baseline",
                research_root=path / "research",
                shared_root=path / "shared",
                docs_root=path / "docs",
                data_cache_root=path / "data_cache",
            )
    raise RuntimeError(f"Could not discover repository root from {cursor}")
