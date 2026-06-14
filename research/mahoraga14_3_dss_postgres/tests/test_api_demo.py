from __future__ import annotations

import pytest
from fastapi import HTTPException

from api import main


def test_health_and_options() -> None:
    payload = main.health()
    assert payload["ok"] is True
    assert payload["backend"] == "parquet"

    options = main.metadata_options()
    assert "B1.05_C1.10_L1.10_R1.05" in options["candidates"]
    assert "base_universe_12" in options["universes"]
    assert options["horizons"]


def test_core_endpoints_return_structured_payloads() -> None:
    calls = [
        lambda: main.overview(),
        lambda: main.scorecard(limit=200),
        lambda: main.robustness_surface(limit=5000),
        lambda: main.whatif_grid(limit=5000),
        lambda: main.decision_replay(),
        lambda: main.slice_query(dimensions=["candidate_id", "fold"], limit=500),
        lambda: main.module_effectiveness(),
        lambda: main.ticker_contribution(limit=200),
        lambda: main.regime_behavior(),
        lambda: main.fold_performance(),
        lambda: main.candidate_compare(),
        lambda: main.query_performance(),
    ]
    for call in calls:
        payload = call()
        assert isinstance(payload, dict)


def test_invalid_filter_is_rejected() -> None:
    with pytest.raises(HTTPException) as exc:
        main.scorecard(candidate_id="NOT_A_CANDIDATE")
    assert exc.value.status_code == 422
