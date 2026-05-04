from __future__ import annotations

import json
from pathlib import Path


def test_official_freeze_file() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "config" / "OFFICIAL_FREEZE.json").read_text(encoding="utf-8"))

    assert payload["official_candidate_id"] == "B1.05_C1.10_L1.10_R1.05"
    assert payload["official_knobs"]["budget_multiplier"] == 1.05
    assert payload["status"] == "OFFICIAL_LONG_ONLY_BASELINE"
