"""Model Commands module."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from starwars_autocalls.cli.common import (
    command_status,
    print_json_panel,
    print_success,
    validated_data,
)
from starwars_autocalls.config import Settings
from starwars_autocalls.data.loading import trainable_rfqs
from starwars_autocalls.data.schemas import PredictionRequest
from starwars_autocalls.features import FeatureBuilder
from starwars_autocalls.modeling.evaluation import regression_metrics, temporal_split
from starwars_autocalls.modeling.training import train_final_model
from starwars_autocalls.serving.prediction import (
    load_model_artifact,
    load_model_metadata,
    predict_one,
)


def train(
    model_name: Annotated[
        str | None,
        typer.Option(
            "--model-name",
            help="Especificación exacta, por ejemplo hist_gradient_boosting__all_without_noise.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Familia de estimador usada con --feature-block."),
    ] = None,
    feature_block: Annotated[
        str | None,
        typer.Option("--feature-block", help="Bloque de features usado con --model."),
    ] = None,
    use_tuned_best: Annotated[
        bool,
        typer.Option(
            "--use-tuned-best",
            help="Usa el mejor resultado de tuning si mejora el benchmark actual.",
        ),
    ] = False,
    use_strategy_selection: Annotated[
        bool,
        typer.Option(
            "--use-strategy-selection",
            help="Usa la estrategia elegida por compare-serving-strategies.",
        ),
    ] = False,
) -> None:
    """Entrena y guarda el artefacto final que utilizará la API."""
    settings = Settings()
    try:
        with command_status("Entrenando y guardando el modelo final"):
            metadata = train_final_model(
                settings,
                model_name=model_name,
                model=model,
                feature_block=feature_block,
                use_tuned_best=use_tuned_best,
                use_strategy_selection=use_strategy_selection,
            )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_json_panel(
        "Modelo final entrenado",
        {
            "model_name": metadata["model_name"],
            "serving_strategy": metadata.get("serving_strategy", "global"),
            "test_metrics": metadata["test_metrics"],
            "artifact_path": settings.model_path,
            "metadata_path": settings.model_metadata_path,
            "mlflow_run_id": metadata.get("mlflow_run_id"),
        },
    )


def serve(
    host: str = typer.Option("127.0.0.1", help="Interfaz en la que escucha la API."),
    port: int = typer.Option(8000, min=1, max=65535, help="Puerto de la API."),
    reload: bool = typer.Option(False, help="Recarga al cambiar archivos Python."),
) -> None:
    """Levanta la API local de inferencia con FastAPI y Uvicorn."""
    print_success(f"API disponible en http://{host}:{port} (Ctrl+C para detenerla)")
    uvicorn.run(
        "starwars_autocalls.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


def evaluate() -> None:
    """Evalúa sobre test el artefacto de modelo guardado actualmente."""
    settings = Settings()
    with command_status("Evaluando el artefacto guardado sobre test"):
        rfqs, volatility, reference = validated_data(settings)
        trainable = trainable_rfqs(rfqs)
        metadata = load_model_metadata(settings)
        artifact = load_model_artifact(settings.model_path)
        split = temporal_split(trainable)
        feature_set = FeatureBuilder().build(trainable, volatility, reference, include_target=True)
        if feature_set.target is None:
            raise RuntimeError("Evaluation requires target values.")
        strategy = artifact.get("strategy", metadata.get("serving_strategy", "global"))
        if strategy == "segmented_by_basket_type":
            predictions = feature_set.target.loc[split.test_index].astype(float).copy()
            predictions.loc[:] = float("nan")
            segment_masks = {
                "single": feature_set.frame["is_single_underlying"].eq(1),
                "worst_of": feature_set.frame["is_worst_of"].eq(1),
            }
            for segment, pipeline in artifact["pipelines"].items():
                segment_index = split.test_index.intersection(
                    feature_set.frame.index[segment_masks[segment]]
                )
                predictions.loc[segment_index] = pipeline.predict(
                    feature_set.frame.loc[segment_index]
                )
            if predictions.isna().any():
                raise RuntimeError("Segmented artifact did not score every test row.")
        else:
            predictions = artifact["pipeline"].predict(feature_set.frame.loc[split.test_index])
        metrics = regression_metrics(feature_set.target.loc[split.test_index], predictions)
    print_json_panel(
        "Evaluación final",
        {
            "metadata_model": metadata.get("model_name"),
            "serving_strategy": strategy,
            "test_metrics": metrics,
        },
    )


def predict(
    input: Annotated[
        Path, typer.Option("--input", exists=True, help="Payload JSON de predicción.")
    ],
) -> None:
    """Realiza una predicción local directa, sin utilizar HTTP."""
    with command_status("Cargando el artefacto y calculando la predicción"):
        payload = PredictionRequest.model_validate_json(input.read_text(encoding="utf-8"))
        prediction = predict_one(payload, Settings())
    print_json_panel("Predicción local", prediction)


def register(app: typer.Typer) -> None:
    """Perform register."""
    panel = "Modelo e inferencia"
    app.command(rich_help_panel=panel)(train)
    app.command(rich_help_panel=panel)(serve)
    app.command(rich_help_panel=panel)(evaluate)
    app.command(rich_help_panel=panel)(predict)
