"""Config module."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MLFLOW_EXPERIMENT = "starwars-autocalls"
RANDOM_SEED = 31415


class Settings(BaseSettings):
    """Runtime paths and constants used by the CLI, API, and model workflows."""

    model_config = SettingsConfigDict(env_prefix="STARWARS_AUTOCALLS_", extra="ignore")

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    raw_data_dir: Path = Path("data/raw")
    artifact_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    model_filename: str = "model.joblib"
    model_metadata_filename: str = "model_metadata.json"
    model_checksum_filename: str = "model.joblib.sha256"
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = MLFLOW_EXPERIMENT

    @property
    def raw_dir(self) -> Path:
        """Return the configured raw-data directory when it exists."""
        candidate = self.project_root / self.raw_data_dir
        return candidate if candidate.exists() else self.project_root

    @property
    def rfqs_path(self) -> Path:
        """Return the path to the RFQs dataset."""
        return self._first_existing("rfqs.csv")

    @property
    def volatility_path(self) -> Path:
        """Return the path to the daily-volatility dataset."""
        return self._first_existing("daily_volatility.csv")

    @property
    def underlyings_path(self) -> Path:
        """Return the path to the underlyings reference dataset."""
        return self._first_existing("underlyings_reference.csv")

    @property
    def model_path(self) -> Path:
        """Return the path to the serialized model artifact."""
        return self.project_root / self.artifact_dir / self.model_filename

    @property
    def model_metadata_path(self) -> Path:
        """Return the path to the model metadata artifact."""
        return self.project_root / self.artifact_dir / self.model_metadata_filename

    @property
    def model_checksum_path(self) -> Path:
        """Return the path to the model checksum file."""
        return self.project_root / self.artifact_dir / self.model_checksum_filename

    @property
    def final_model_config_path(self) -> Path:
        """Return the frozen final-model configuration path."""
        return self.project_root / "config" / "final_model.json"

    @property
    def default_mlflow_tracking_uri(self) -> str:
        """Return the default local MLflow tracking URI."""
        return f"sqlite:///{(self.project_root / 'mlflow.db').resolve()}"

    @property
    def metrics_dir(self) -> Path:
        """Return the directory for model evaluation reports."""
        return self.project_root / self.reports_dir / "model_evaluation"

    @property
    def explainability_dir(self) -> Path:
        """Return the directory for explainability reports."""
        return self.project_root / self.reports_dir / "explainability"

    @property
    def explainability_figures_dir(self) -> Path:
        """Return the directory for explainability figures."""
        return self.explainability_dir / "figures"

    @property
    def eda_dir(self) -> Path:
        """Return the directory for exploratory data analysis reports."""
        return self.project_root / self.reports_dir / "data_analysis"

    @property
    def eda_overview_dir(self) -> Path:
        """Return the directory for EDA overview reports."""
        return self.eda_dir / "eda"

    @property
    def split_audit_dir(self) -> Path:
        """Return the directory for data-split audit reports."""
        return self.eda_dir / "split_audit"

    @property
    def feature_audit_dir(self) -> Path:
        """Return the directory for feature audit reports."""
        return self.project_root / self.reports_dir / "feature_audit"

    def ensure_output_dirs(self) -> None:
        """Create all artifact and report directories if needed."""
        for path in [
            self.project_root / self.artifact_dir,
            self.project_root / self.reports_dir,
            self.metrics_dir,
            self.explainability_dir,
            self.explainability_figures_dir,
            self.eda_dir,
            self.eda_overview_dir,
            self.split_audit_dir,
            self.feature_audit_dir,
            self.eda_overview_dir / "supplemental",
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _first_existing(self, filename: str) -> Path:
        """Find a dataset in the configured and project-root locations."""
        candidates = [
            self.project_root / self.raw_data_dir / filename,
            self.project_root / filename,
            self.raw_dir / filename,
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"Could not find {filename} in project root or data/raw.")
