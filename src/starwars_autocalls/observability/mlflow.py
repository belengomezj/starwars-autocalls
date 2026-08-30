"""Mlflow module."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator

from starwars_autocalls.config import Settings
from starwars_autocalls.observability.logging import get_logger

logger = get_logger(__name__)


def configure_mlflow_tracking(mlflow: Any, settings: Settings | None = None) -> None:
    """Perform configure mlflow tracking."""
    settings = settings or Settings()
    tracking_uri = (
        os.environ.get("MLFLOW_TRACKING_URI")
        or settings.mlflow_tracking_uri
        or settings.default_mlflow_tracking_uri
    )
    # MLflow emits database/bootstrap details at INFO level for every first
    # connection. They are implementation details for this CLI, not benchmark
    # progress or actionable diagnostics.
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    logger.debug(
        "mlflow_configured",
        tracking_uri=tracking_uri,
        experiment=settings.mlflow_experiment,
    )


def mlflow_dataset_name(feature_block: str, split_name: str, fold_name: str | None = None) -> str:
    """Handle mlflow dataset name."""
    parts = [feature_block, split_name]
    if fold_name:
        parts.append(fold_name)
    return "__".join(parts)


def metric_dataset_from_features(
    X: pd.DataFrame,
    y: pd.Series,
    index: pd.Index,
    feature_columns: list[str],
    target_name: str = "avg_duration_months",
) -> pd.DataFrame:
    """Handle metric dataset from features."""
    dataset = X.loc[index, feature_columns].copy()
    dataset[target_name] = y.loc[index].to_numpy()
    integer_columns = dataset.select_dtypes(include="integer").columns
    if len(integer_columns):
        # MLflow warns that integer columns cannot represent missing values at
        # inference time. The metric dataset is a logging artifact, so use the
        # nullable-safe representation without changing the model features.
        dataset = dataset.astype({column: "float64" for column in integer_columns})
    return dataset


def short_data_hash_tags(data_hashes: dict[str, str | None]) -> dict[str, str]:
    """Handle short data hash tags."""
    key_map = {
        "rfqs_csv_sha256": "data_rfqs_sha256",
        "daily_volatility_csv_sha256": "data_daily_volatility_sha256",
        "underlyings_reference_csv_sha256": "data_underlyings_sha256",
    }
    tags: dict[str, str] = {}
    for source_key, tag_key in key_map.items():
        value = data_hashes.get(source_key)
        if value:
            tags[tag_key] = value[:12]
    return tags


@contextmanager
def mlflow_parent_run(
    run_name: str,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
):
    """Handle mlflow parent run."""
    try:
        import mlflow

        from starwars_autocalls.modeling.specs import model_family_from_name

        configure_mlflow_tracking(mlflow)
        run_context = mlflow.start_run(run_name=run_name)
    except Exception as exc:
        logger.warning(
            "mlflow_parent_run_unavailable",
            run_name=run_name,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        yield None
        return

    run_params = dict(params or {})
    run_tags = dict(tags or {})
    model_name = str(run_tags.get("model_name") or run_params.get("model_name") or run_name)
    model_family = str(run_tags.get("model_family") or model_family_from_name(model_name))
    run_tags.setdefault("model_family", model_family)
    run_params.setdefault("model_family", model_family)
    with run_context as run:
        try:
            if run_tags:
                mlflow.set_tags(run_tags)
            if run_params:
                mlflow.log_params(run_params)
        except Exception as exc:
            logger.warning(
                "mlflow_parent_metadata_failed",
                run_name=run_name,
                error_type=type(exc).__name__,
                error=str(exc),
            )
        yield run.info.run_id


def log_mlflow_run(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: dict[str, pd.DataFrame | dict[str, Any] | Path] | None = None,
    tags: dict[str, str] | None = None,
    model: BaseEstimator | None = None,
    input_example: pd.DataFrame | None = None,
    metric_dataset: pd.DataFrame | None = None,
    dataset_name: str = "validation",
    nested: bool = False,
    registered_model_name: str | None = None,
    model_artifact_name: str | None = None,
) -> str | None:
    """Perform log mlflow run."""
    try:
        import mlflow
        import mlflow.sklearn

        from starwars_autocalls.modeling.specs import model_family_from_name

        configure_mlflow_tracking(mlflow)
        run_params = dict(params)
        run_tags = dict(tags or {})
        model_name = str(run_tags.get("model_name") or run_params.get("model_name") or run_name)
        model_family = str(run_tags.get("model_family") or model_family_from_name(model_name))
        run_tags.setdefault("model_family", model_family)
        run_params.setdefault("model_family", model_family)
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            if run_tags:
                mlflow.set_tags(run_tags)
            mlflow.log_params(run_params)
            logged_model_id = None
            if model is not None:
                try:
                    safe_input_example = input_example
                    if input_example is not None:
                        safe_input_example = input_example.copy()
                        integer_columns = safe_input_example.select_dtypes(
                            include=["integer"]
                        ).columns
                        safe_input_example[integer_columns] = safe_input_example[
                            integer_columns
                        ].astype(float)
                    model_info = mlflow.sklearn.log_model(
                        sk_model=model,
                        name=model_artifact_name or model_name,
                        params=run_params,
                        input_example=safe_input_example,
                        registered_model_name=registered_model_name,
                        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                    )
                    logged_model_id = model_info.model_id
                    mlflow.set_tag("model_logged", "true")
                    mlflow.set_tag("logged_model_id", logged_model_id)
                except Exception as exc:
                    mlflow.set_tag("model_logged", "false")
                    mlflow.set_tag("model_log_error", f"{type(exc).__name__}: {exc}"[:500])
                    logger.warning(
                        "mlflow_model_log_failed",
                        run_name=run_name,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )

            if metric_dataset is not None:
                try:
                    dataset = mlflow.data.from_pandas(metric_dataset, name=dataset_name)
                    if logged_model_id:
                        mlflow.log_metrics(metrics, model_id=logged_model_id, dataset=dataset)
                    else:
                        mlflow.log_metrics(metrics, dataset=dataset)
                except Exception as exc:
                    logger.warning(
                        "mlflow_dataset_log_failed",
                        run_name=run_name,
                        dataset_name=dataset_name,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    mlflow.log_metrics(metrics)
            else:
                mlflow.log_metrics(metrics)
            if artifacts:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir)
                    for name, artifact in artifacts.items():
                        if isinstance(artifact, pd.DataFrame):
                            path = tmp / f"{name}.csv"
                            artifact.to_csv(path, index=False)
                            mlflow.log_artifact(str(path))
                        elif isinstance(artifact, dict):
                            path = tmp / f"{name}.json"
                            payload = artifact
                            if payload.get("mlflow_run_id") is None:
                                payload = {**payload, "mlflow_run_id": run.info.run_id}
                            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                            mlflow.log_artifact(str(path))
                        else:
                            mlflow.log_artifact(str(artifact))
            return run.info.run_id
    except Exception as exc:
        logger.warning(
            "mlflow_run_failed",
            run_name=run_name,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return None
