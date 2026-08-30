"""Prediction module."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from starwars_autocalls import __version__
from starwars_autocalls.config import Settings
from starwars_autocalls.data.schemas import PredictionRequest
from starwars_autocalls.features import FeatureBuilder
from starwars_autocalls.features.builders import normalize_observation_frequency, parse_underlyings
from starwars_autocalls.modeling.artifacts import load_model_artifact as read_model_artifact
from starwars_autocalls.observability import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def _load_cached_artifact(path: str, modified_at_ns: int) -> dict[str, Any]:
    """Handle load cached artifact."""
    del modified_at_ns
    return read_model_artifact(Path(path))


def load_model_artifact(path: Path) -> dict[str, Any]:
    """Return load model artifact."""
    if not path.exists():
        return read_model_artifact(path)
    return _load_cached_artifact(str(path.resolve()), path.stat().st_mtime_ns)


def prediction_request_to_frame(payload: PredictionRequest) -> pd.DataFrame:
    """Handle prediction request to frame."""
    row = payload.model_dump()
    row["rfq_id"] = "PREDICTION"
    row["executed"] = False
    row["avg_duration_months"] = None
    return pd.DataFrame([row])


def _validate_serving_coverage(
    payload: PredictionRequest,
    artifact: dict[str, Any],
) -> tuple[str, int, list[str], bool]:
    """Handle validate serving coverage."""
    underlyings = parse_underlyings(payload.underlyings)
    accepted = set(artifact["reference"]["underlying"].astype(str))
    unknown = sorted(set(underlyings) - accepted)
    if unknown:
        raise ValueError(f"Unknown underlyings: {unknown}. Accepted values: {sorted(accepted)}")

    metadata = artifact.get("metadata", {})
    accepted_products = set(metadata.get("accepted_product_types", []))
    if accepted_products and payload.product_type not in accepted_products:
        raise ValueError(
            f"Unknown product_type={payload.product_type!r}. "
            f"Accepted values: {sorted(accepted_products)}"
        )

    volatility = artifact["volatility"].copy()
    volatility["date"] = pd.to_datetime(volatility["date"])
    request_date = pd.Timestamp(payload.requested_date)
    latest_dates: list[pd.Timestamp] = []
    for underlying in underlyings:
        history = volatility.loc[
            (volatility["underlying"] == underlying) & (volatility["date"] <= request_date),
            "date",
        ]
        if history.empty:
            raise ValueError(
                f"No market data is available for {underlying} on or before {payload.requested_date}."
            )
        latest_dates.append(pd.Timestamp(history.max()))

    ages = [(request_date - date).days for date in latest_dates]
    max_age = max(ages)
    max_staleness = int(metadata.get("max_market_data_staleness_days", 10))
    if max_age > max_staleness:
        raise ValueError(
            f"Market data is stale by {max_age} days; maximum allowed is {max_staleness} days."
        )

    warnings: list[str] = []
    out_of_distribution = False
    ranges = metadata.get("training_request_ranges", {})
    for field in [
        "autocall_barrier_pct",
        "protection_barrier_pct",
        "no_call_period_months",
        "quoted_implied_vol",
        "notional_credits",
    ]:
        bounds = ranges.get(field)
        value = float(getattr(payload, field))
        if bounds and not float(bounds["min"]) <= value <= float(bounds["max"]):
            out_of_distribution = True
            warnings.append(
                f"{field}={value:g} is outside the training range "
                f"[{float(bounds['min']):g}, {float(bounds['max']):g}]"
            )
    return min(latest_dates).date().isoformat(), max_age, warnings, out_of_distribution


def _contractual_bounds(payload: PredictionRequest) -> tuple[float, float]:
    """Handle contractual bounds."""
    maturity = (payload.end_date - payload.start_date).days / 30.4375
    first_call = max(
        float(payload.no_call_period_months),
        normalize_observation_frequency(payload.observation_frequency),
    )
    return first_call, maturity


def _predict_with_artifact(
    payload: PredictionRequest,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    """Generate one prediction using an already loaded artifact."""
    market_data_as_of, max_age, warnings, out_of_distribution = _validate_serving_coverage(
        payload, artifact
    )
    row = prediction_request_to_frame(payload)
    feature_set = FeatureBuilder().build(
        row,
        artifact["volatility"],
        artifact["reference"],
        include_target=False,
    )
    metadata = artifact.get("metadata", {})
    strategy = artifact.get("strategy", metadata.get("serving_strategy", "global"))
    if strategy == "segmented_by_basket_type":
        pipelines = artifact.get("pipelines", {})
        try:
            pipeline = pipelines[payload.basket_type]
        except KeyError as exc:
            raise ValueError(
                f"No segmented model found for basket_type={payload.basket_type!r}."
            ) from exc
    else:
        pipeline = artifact["pipeline"]
    raw_prediction = float(pipeline.predict(feature_set.frame)[0])
    lower_bound, upper_bound = _contractual_bounds(payload)
    clipping = metadata.get("contractual_clipping", {})
    prediction = (
        min(max(raw_prediction, lower_bound), upper_bound)
        if clipping.get("enabled", False)
        else raw_prediction
    )
    calibration = metadata.get("conformal_calibration", {})
    interval_radius = calibration.get("absolute_residual_quantile")
    interval_lower: float | None = None
    interval_upper: float | None = None
    if interval_radius is not None:
        interval_lower = max(prediction - float(interval_radius), 0.0)
        interval_upper = prediction + float(interval_radius)
        if clipping.get("enabled", False):
            interval_lower = max(interval_lower, lower_bound)
            interval_upper = min(interval_upper, upper_bound)
    logger.info(
        "prediction_completed",
        model_name=metadata.get("model_name", "unknown"),
        serving_strategy=strategy,
        basket_type=payload.basket_type,
    )
    return {
        "predicted_avg_duration_months": round(prediction, 4),
        "model_version": metadata.get("package_version", __version__),
        "model_name": metadata.get("model_name", "unknown"),
        "serving_strategy": strategy,
        "prediction_interval_lower_months": (
            round(interval_lower, 4) if interval_lower is not None else None
        ),
        "prediction_interval_upper_months": (
            round(interval_upper, 4) if interval_upper is not None else None
        ),
        "interval_nominal_coverage": calibration.get("nominal_coverage"),
        "out_of_distribution": out_of_distribution,
        "warnings": warnings,
        "market_data_as_of": market_data_as_of,
        "max_market_data_age_days": max_age,
    }


def predict_one(payload: PredictionRequest, settings: Settings | None = None) -> dict[str, Any]:
    """Generate a prediction for one RFQ."""
    settings = settings or Settings()
    artifact = load_model_artifact(settings.model_path)
    return _predict_with_artifact(payload, artifact)


def predict_batch(
    payloads: list[PredictionRequest],
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Generate predictions for multiple RFQs while loading the artifact once."""
    settings = settings or Settings()
    artifact = load_model_artifact(settings.model_path)
    return [_predict_with_artifact(payload, artifact) for payload in payloads]


def load_model_metadata(settings: Settings | None = None) -> dict[str, Any]:
    """Return load model metadata."""
    settings = settings or Settings()
    if settings.model_metadata_path.exists():
        return json.loads(settings.model_metadata_path.read_text(encoding="utf-8"))
    artifact = load_model_artifact(settings.model_path)
    return artifact.get("metadata", {})
