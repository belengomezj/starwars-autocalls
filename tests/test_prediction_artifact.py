from __future__ import annotations

import pytest

from starwars_autocalls.config import Settings
from starwars_autocalls.data.schemas import PredictionRequest
from starwars_autocalls.serving.prediction import predict_batch, predict_one


def test_prediction_with_sample_payload_when_artifact_exists() -> None:
    settings = Settings()
    if not settings.model_path.exists():
        pytest.skip("model artifact has not been trained yet")
    payload = PredictionRequest.model_validate_json(
        settings.project_root.joinpath("sample_payload.json").read_text(encoding="utf-8")
    )
    result = predict_one(payload, settings)
    assert result["model_version"] == "0.1.0"
    assert result["predicted_avg_duration_months"] > 0

    batch_results = predict_batch([payload, payload], settings)
    assert len(batch_results) == 2
    assert [item["predicted_avg_duration_months"] for item in batch_results] == [
        result["predicted_avg_duration_months"],
        result["predicted_avg_duration_months"],
    ]
