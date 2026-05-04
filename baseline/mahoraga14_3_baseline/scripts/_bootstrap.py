from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_paths() -> Path:
    here = Path(__file__).resolve()
    baseline_root = here.parents[1]
    repo_root = here.parents[3]
    src_root = baseline_root / "src"
    for path in (repo_root, src_root, src_root / "mahoraga14_3_baseline"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return repo_root
