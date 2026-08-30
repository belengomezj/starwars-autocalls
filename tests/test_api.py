from __future__ import annotations

from fastapi.testclient import TestClient

import starwars_autocalls.api.main as api_main
from starwars_autocalls.api.main import create_app
from starwars_autocalls.config import Settings


def _valid_prediction_payload() -> dict[str, object]:
    return {
        "product_type": "P1",
        "underlyings": "AAA",
        "basket_type": "single",
        "autocall_barrier_pct": 1.0,
        "protection_barrier_pct": 0.6,
        "no_call_period_months": 6,
        "observation_frequency": "1M",
        "quoted_implied_vol": 0.25,
        "notional_credits": 100_000,
        "requested_date": "2024-06-01",
        "start_date": "2024-06-02",
        "end_date": "2027-06-02",
    }


def test_readiness_requires_a_model_artifact(tmp_path) -> None:
    client = TestClient(create_app(Settings(project_root=tmp_path)))

    response = client.get("/ready")

    assert response.status_code == 503
    assert "Run training first" in response.json()["detail"]


def test_predict_batch_returns_predictions_in_request_order(monkeypatch, tmp_path) -> None:
    def fake_predict_batch(payloads, settings):
        assert len(payloads) == 2
        assert settings.project_root == tmp_path
        assert all("avg_duration_months" not in payload.model_dump() for payload in payloads)
        return [
            {
                "predicted_avg_duration_months": 10.0 + index,
                "model_version": "test",
            }
            for index, _ in enumerate(payloads)
        ]

    monkeypatch.setattr(api_main, "predict_batch", fake_predict_batch)
    client = TestClient(create_app(Settings(project_root=tmp_path)))

    historical_payload = {
        **_valid_prediction_payload(),
        "rfq_id": "RFQ-NEW",
        "executed": False,
        "avg_duration_months": None,
    }
    response = client.post(
        "/predict-batch",
        json={"requests": [_valid_prediction_payload(), historical_payload]},
    )

    assert response.status_code == 200
    assert [item["predicted_avg_duration_months"] for item in response.json()["predictions"]] == [
        10.0,
        11.0,
    ]


def test_predict_batch_rejects_an_empty_batch(tmp_path) -> None:
    client = TestClient(create_app(Settings(project_root=tmp_path)))

    response = client.post("/predict-batch", json={"requests": []})

    assert response.status_code == 422
