"""Main module."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from starwars_autocalls import __version__
from starwars_autocalls.config import Settings
from starwars_autocalls.data.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
)
from starwars_autocalls.modeling.artifacts import InvalidModelArtifactError
from starwars_autocalls.observability import configure_logging, get_logger
from starwars_autocalls.serving.prediction import load_model_metadata, predict_batch, predict_one

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Perform create app."""
    settings = settings or Settings()
    configure_logging(settings.log_level, settings.log_format)
    application = FastAPI(title="Star Wars Autocalls API", version=__version__)

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Handle health."""
        return HealthResponse(status="ok", version=__version__)

    @application.get("/ready", response_model=ReadinessResponse)
    def ready() -> ReadinessResponse:
        """Return ready."""
        try:
            metadata = load_model_metadata(settings)
        except (FileNotFoundError, InvalidModelArtifactError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return ReadinessResponse(status="ready", model_name=metadata.get("model_name"))

    @application.get("/model-info")
    def model_info() -> dict:
        """Handle model info."""
        try:
            return load_model_metadata(settings)
        except (FileNotFoundError, InvalidModelArtifactError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest) -> PredictionResponse:
        """Generate predictions."""
        try:
            result = predict_one(payload, settings)
        except (FileNotFoundError, InvalidModelArtifactError) as exc:
            logger.warning("prediction_service_unavailable", error=str(exc))
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            logger.info("prediction_rejected", reason=str(exc))
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictionResponse(**result)

    @application.post("/predict-batch", response_model=BatchPredictionResponse)
    def predict_batch_endpoint(payload: BatchPredictionRequest) -> BatchPredictionResponse:
        """Generate predictions for a batch of RFQs in request order."""
        try:
            results = predict_batch(payload.requests, settings)
        except (FileNotFoundError, InvalidModelArtifactError) as exc:
            logger.warning("batch_prediction_service_unavailable", error=str(exc))
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            logger.info("batch_prediction_rejected", reason=str(exc))
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**result) for result in results]
        )

    return application


app = create_app()
