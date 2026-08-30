"""Reproducibility module."""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
from typing import Any

from starwars_autocalls.config import Settings


def file_sha256(path: Path) -> str | None:
    """Return a file's SHA-256 hash, or None if it does not exist."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_versions() -> dict[str, str | None]:
    """Return the installed version of each project dependency."""
    packages = [
        "catboost",
        "fastapi",
        "lightgbm",
        "mlflow",
        "numpy",
        "optuna",
        "pandas",
        "pandera",
        "scikit-learn",
        "structlog",
        "uvicorn",
        "xgboost",
    ]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def data_hashes(settings: Settings) -> dict[str, str | None]:
    """Return SHA-256 hashes for the input datasets."""
    return {
        "rfqs_csv_sha256": file_sha256(settings.rfqs_path),
        "daily_volatility_csv_sha256": file_sha256(settings.volatility_path),
        "underlyings_reference_csv_sha256": file_sha256(settings.underlyings_path),
    }


def reproducibility_manifest(settings: Settings) -> dict[str, Any]:
    """Build metadata needed to reproduce a model run."""
    return {
        "data_hashes": data_hashes(settings),
        "uv_lock_sha256": file_sha256(settings.project_root / "uv.lock"),
        "dependency_versions": dependency_versions(),
    }
