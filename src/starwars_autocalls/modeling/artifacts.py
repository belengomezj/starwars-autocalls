"""Artifacts module."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib

from starwars_autocalls import __version__
from starwars_autocalls.config import Settings
from starwars_autocalls.observability import get_logger

MODEL_ARTIFACT_FORMAT_VERSION = 1
REQUIRED_ARTIFACT_KEYS = {"metadata", "reference", "volatility"}

logger = get_logger(__name__)


def _sha256(path: Path) -> str:
    """Handle sha256."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class InvalidModelArtifactError(ValueError):
    """Raised when a persisted artifact does not satisfy the serving contract."""


def save_model_bundle(
    artifact: dict[str, Any], metadata: dict[str, Any], settings: Settings
) -> None:
    """Persist the serving artifact and its human-readable metadata atomically."""
    settings.ensure_output_dirs()
    payload = {**artifact, "artifact_format_version": MODEL_ARTIFACT_FORMAT_VERSION}
    metadata_payload = {**metadata, "artifact_format_version": MODEL_ARTIFACT_FORMAT_VERSION}

    artifact_tmp = settings.model_path.with_suffix(f"{settings.model_path.suffix}.tmp")
    metadata_tmp = settings.model_metadata_path.with_suffix(
        f"{settings.model_metadata_path.suffix}.tmp"
    )
    joblib.dump(payload, artifact_tmp)
    metadata_tmp.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    artifact_tmp.replace(settings.model_path)
    metadata_tmp.replace(settings.model_metadata_path)
    settings.model_checksum_path.write_text(f"{_sha256(settings.model_path)}\n", encoding="utf-8")
    logger.info(
        "model_artifact_saved",
        artifact_path=str(settings.model_path),
        metadata_path=str(settings.model_metadata_path),
        model_name=metadata_payload.get("model_name"),
    )


def load_model_artifact(path: Path) -> dict[str, Any]:
    """Return load model artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}. Run training first.")
    checksum_path = path.with_name(f"{path.name}.sha256")
    if checksum_path.exists():
        expected = checksum_path.read_text(encoding="utf-8").strip().split()[0]
        actual = _sha256(path)
        if actual != expected:
            raise InvalidModelArtifactError(
                f"Invalid model artifact at {path}: SHA-256 checksum mismatch."
            )
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise InvalidModelArtifactError(f"Invalid model artifact at {path}: expected a mapping.")
    missing = REQUIRED_ARTIFACT_KEYS - set(artifact)
    if missing:
        raise InvalidModelArtifactError(
            f"Invalid model artifact at {path}: missing keys {sorted(missing)}."
        )
    if "pipeline" not in artifact and "pipelines" not in artifact:
        raise InvalidModelArtifactError(
            f"Invalid model artifact at {path}: no prediction pipeline found."
        )
    version = artifact.get("artifact_format_version")
    if version != MODEL_ARTIFACT_FORMAT_VERSION:
        raise InvalidModelArtifactError(
            f"Invalid model artifact at {path}: unsupported format version {version!r}; "
            f"expected {MODEL_ARTIFACT_FORMAT_VERSION}."
        )
    metadata = artifact.get("metadata", {})
    package_version = metadata.get("package_version")
    if package_version and package_version != __version__:
        raise InvalidModelArtifactError(
            f"Invalid model artifact at {path}: package version {package_version!r} "
            f"does not match runtime version {__version__!r}."
        )
    if "feature_manifest" not in metadata:
        logger.warning("artifact_feature_manifest_missing", artifact_path=str(path))
    return artifact
