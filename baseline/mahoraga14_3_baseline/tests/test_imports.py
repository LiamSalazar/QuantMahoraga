from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap() -> None:
    test_file = Path(__file__).resolve()
    repo_root = test_file.parents[3]
    src_root = test_file.parents[1] / "src"
    for path in (repo_root, src_root, src_root / "mahoraga14_3_baseline"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def test_official_runner_imports() -> None:
    _bootstrap()
    from mahoraga14_3_baseline.official_baseline_runner import run_official_baseline

    assert callable(run_official_baseline)
