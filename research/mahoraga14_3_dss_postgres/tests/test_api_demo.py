from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_and_options() -> None:
    health = client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["ok"] is True
    assert payload["backend"] == "parquet"

    options = client.get("/metadata/options").json()
    assert "B1.05_C1.10_L1.10_R1.05" in options["candidates"]
    assert "base_universe_12" in options["universes"]
    assert options["horizons"]


def test_core_endpoints_return_rows() -> None:
    for path in [
        "/overview",
        "/scorecard",
        "/robustness/surface",
        "/whatif/grid",
        "/decision/replay",
        "/slice",
        "/module/effectiveness",
        "/ticker/contribution",
        "/regime/behavior",
        "/fold/performance",
        "/candidate/compare",
    ]:
        response = client.get(path)
        assert response.status_code == 200, path
        assert isinstance(response.json(), dict)


def test_invalid_filter_is_rejected() -> None:
    response = client.get("/scorecard?candidate_id=NOT_A_CANDIDATE")
    assert response.status_code == 422
