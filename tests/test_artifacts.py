from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from starwars_autocalls.config import Settings
from starwars_autocalls.modeling.artifacts import (
    MODEL_ARTIFACT_FORMAT_VERSION,
    InvalidModelArtifactError,
    load_model_artifact,
    save_model_bundle,
)


def build_model_artifact(pipeline: Any, metadata: dict[str, object]) -> dict[str, Any]:
    return {
        "pipeline": pipeline,
        "metadata": metadata,
        "reference": pd.DataFrame({"underlying": ["AAA"]}),
        "volatility": pd.DataFrame({"underlying": ["AAA"]}),
    }


def test_save_and_load_model_bundle(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    pipeline = DummyRegressor().fit([[0.0]], [3.0])
    metadata = {"model_name": "dummy", "test_metrics": {"mae": 1.0}}

    save_model_bundle(build_model_artifact(pipeline, metadata), metadata, settings)

    loaded = load_model_artifact(settings.model_path)
    readable_metadata = json.loads(settings.model_metadata_path.read_text(encoding="utf-8"))
    assert loaded["artifact_format_version"] == MODEL_ARTIFACT_FORMAT_VERSION
    assert loaded["metadata"]["model_name"] == "dummy"
    assert readable_metadata["artifact_format_version"] == MODEL_ARTIFACT_FORMAT_VERSION
    assert settings.model_checksum_path.exists()


def test_artifact_checksum_mismatch_is_rejected(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    pipeline = DummyRegressor().fit([[0.0]], [3.0])
    metadata = {"model_name": "dummy"}
    save_model_bundle(build_model_artifact(pipeline, metadata), metadata, settings)
    settings.model_checksum_path.write_text("0" * 64, encoding="utf-8")

    with pytest.raises(InvalidModelArtifactError, match="checksum mismatch"):
        load_model_artifact(settings.model_path)


def test_invalid_model_bundle_is_rejected(tmp_path) -> None:
    path = tmp_path / "invalid.joblib"
    joblib.dump({}, path)

    with pytest.raises(InvalidModelArtifactError, match="missing keys"):
        load_model_artifact(path)
