from __future__ import annotations

from starwars_autocalls.config import Settings
from starwars_autocalls.observability.mlflow import configure_mlflow_tracking


class FakeMlflow:
    tracking_uri: str | None = None
    experiment: str | None = None

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    def set_experiment(self, experiment: str) -> None:
        self.experiment = experiment


def test_mlflow_defaults_to_project_local_sqlite(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    mlflow = FakeMlflow()

    configure_mlflow_tracking(mlflow, Settings(project_root=tmp_path))

    assert mlflow.tracking_uri == f"sqlite:///{tmp_path / 'mlflow.db'}"
    assert mlflow.experiment == "starwars-autocalls"


def test_standard_mlflow_tracking_uri_takes_precedence(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    mlflow = FakeMlflow()

    configure_mlflow_tracking(mlflow, Settings(project_root=tmp_path))

    assert mlflow.tracking_uri == "https://mlflow.example.test"
