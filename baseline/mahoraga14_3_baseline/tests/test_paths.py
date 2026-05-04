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


def test_repo_layout_and_baseline_dirs() -> None:
    _bootstrap()
    from mahoraga14_3_baseline.mahoraga14_config import Mahoraga14Config
    from shared.pathing import discover_repo_layout

    layout = discover_repo_layout(__file__)
    cfg = Mahoraga14Config()

    assert layout.repo_root.name == "QuantMahoraga"
    assert Path(cfg.outputs_dir).name == "outputs"
    assert Path(cfg.audit_dir).name == "audit"
    assert Path(cfg.paper_pack_dir).name == "paper_pack"
