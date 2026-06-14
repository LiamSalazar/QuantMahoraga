from __future__ import annotations

from pathlib import Path

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def assert_not_lfs_pointer(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    with path.open("rb") as handle:
        prefix = handle.read(len(LFS_POINTER_PREFIX))
    if prefix == LFS_POINTER_PREFIX:
        raise RuntimeError(
            f"Data artifact is a Git LFS pointer, not real data: {path}. "
            "Run: git lfs install && git lfs pull."
        )
