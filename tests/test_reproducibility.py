from __future__ import annotations

import json

import pytest

from starwars_autocalls.config import Settings
from starwars_autocalls.modeling.training import _load_final_model_spec
from starwars_autocalls.reproducibility import file_sha256


def test_final_model_config_reconstructs_selected_model() -> None:
    spec, config = _load_final_model_spec(Settings())
    assert spec.name == config["model_name"]
    assert config["serving_strategy"] == "global"


def test_model_metadata_uv_lock_hash_matches_repo() -> None:
    """Guards against artifacts/model_metadata.json drifting from the real uv.lock.

    A mismatch means the dependency environment recorded for the served model
    cannot be reconstructed from the uv.lock currently in the repo (e.g. the
    lockfile was regenerated after training without retraining/re-exporting
    the metadata).
    """
    settings = Settings()
    if not settings.model_metadata_path.exists():
        pytest.skip("model metadata has not been generated yet")
    metadata = json.loads(settings.model_metadata_path.read_text(encoding="utf-8"))
    recorded_hash = metadata.get("uv_lock_sha256")
    actual_hash = file_sha256(settings.project_root / "uv.lock")
    assert recorded_hash == actual_hash, (
        "artifacts/model_metadata.json uv_lock_sha256 "
        f"({recorded_hash}) does not match the repo's current uv.lock ({actual_hash}). "
        "The served model's dependency environment is not verifiable against the current "
        "lockfile. Regenerate the metadata by retraining (`uv run starwars-autocalls train`) "
        "after resolving the uv.lock discrepancy."
    )
