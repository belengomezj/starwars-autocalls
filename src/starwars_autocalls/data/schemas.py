"""Schemas module."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from starwars_autocalls.features.builders import normalize_observation_frequency, parse_underlyings

HISTORICAL_RFQ_ONLY_FIELDS = frozenset({"rfq_id", "executed", "avg_duration_months"})

RFQ_SCHEMA = DataFrameSchema(
    {
        "rfq_id": Column(str, nullable=False),
        "product_type": Column(str, nullable=False),
        "underlyings": Column(str, nullable=False),
        "basket_type": Column(str, Check.isin(["single", "worst_of"]), nullable=False),
        "autocall_barrier_pct": Column(float, Check.in_range(0.5, 2.0), nullable=False),
        "protection_barrier_pct": Column(float, Check.in_range(0.0, 1.5), nullable=False),
        "no_call_period_months": Column(int, Check.ge(0), nullable=False),
        "observation_frequency": Column(str, nullable=False),
        "quoted_implied_vol": Column(float, Check.in_range(0.0, 2.0), nullable=False),
        "notional_credits": Column(float, Check.gt(0), nullable=False),
        "counterparty": Column(str, nullable=False),
        "trader_id": Column(str, nullable=False),
        "requested_date": Column(pa.DateTime, nullable=False),
        "executed": Column(bool, nullable=False),
        "start_date": Column(pa.DateTime, nullable=False),
        "end_date": Column(pa.DateTime, nullable=False),
        "avg_duration_months": Column(float, Check.ge(0), nullable=True),
    },
    coerce=True,
)

DAILY_VOL_SCHEMA = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "underlying": Column(str, nullable=False),
        "realized_vol_63d": Column(float, Check.in_range(0.0, 2.0), nullable=False),
    },
    coerce=True,
)

UNDERLYING_REFERENCE_SCHEMA = DataFrameSchema(
    {
        "underlying": Column(str, nullable=False, unique=True),
        "sector": Column(str, nullable=False),
        "structural_base_vol": Column(float, Check.in_range(0.0, 2.0), nullable=False),
    },
    coerce=True,
)


class PredictionRequest(BaseModel):
    """Represent PredictionRequest."""

    model_config = ConfigDict(extra="forbid")

    product_type: str
    underlyings: str
    basket_type: str = Field(pattern="^(single|worst_of)$")
    autocall_barrier_pct: float = Field(ge=0.5, le=2.0)
    protection_barrier_pct: float = Field(gt=0, le=1.5)
    no_call_period_months: int = Field(ge=0)
    observation_frequency: str
    quoted_implied_vol: float = Field(ge=0, le=2.0)
    notional_credits: float = Field(gt=0)
    counterparty: str = "UNKNOWN"
    trader_id: str = "UNKNOWN"
    requested_date: date
    start_date: date
    end_date: date

    @model_validator(mode="before")
    @classmethod
    def discard_historical_only_fields(cls, value: object) -> object:
        """Accept historical RFQ rows without exposing post-request fields to inference."""
        if not isinstance(value, Mapping):
            return value
        return {key: item for key, item in value.items() if key not in HISTORICAL_RFQ_ONLY_FIELDS}

    @field_validator("underlyings")
    @classmethod
    def normalize_underlyings(cls, value: str) -> str:
        """Return normalize underlyings."""
        underlyings = parse_underlyings(value)
        if not underlyings:
            raise ValueError("underlyings must contain at least one ticker")
        normalized = [underlying.upper() for underlying in underlyings]
        if len(normalized) != len(set(normalized)):
            raise ValueError("underlyings must not contain duplicate tickers")
        return "|".join(normalized)

    @field_validator("observation_frequency")
    @classmethod
    def validate_observation_frequency(cls, value: str) -> str:
        """Perform validate observation frequency."""
        normalize_observation_frequency(value)
        return value.strip()

    @model_validator(mode="after")
    def validate_contract(self) -> PredictionRequest:
        """Perform validate contract."""
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        if self.requested_date > self.start_date:
            raise ValueError("requested_date must be on or before contractual start_date")
        maturity_months = (self.end_date - self.start_date).days / 30.4375
        if self.no_call_period_months > maturity_months:
            raise ValueError("no_call_period_months must not exceed contractual maturity")
        basket_size = len(parse_underlyings(self.underlyings))
        if self.basket_type == "single" and basket_size != 1:
            raise ValueError("basket_type='single' requires exactly one underlying")
        if self.basket_type == "worst_of" and basket_size < 2:
            raise ValueError("basket_type='worst_of' requires at least two underlyings")
        return self


class PredictionResponse(BaseModel):
    """Represent PredictionResponse."""

    predicted_avg_duration_months: float
    model_version: str
    model_name: str | None = None
    serving_strategy: str | None = None
    prediction_interval_lower_months: float | None = None
    prediction_interval_upper_months: float | None = None
    interval_nominal_coverage: float | None = None
    out_of_distribution: bool = False
    warnings: list[str] = Field(default_factory=list)
    market_data_as_of: date | None = None
    max_market_data_age_days: int | None = None


class BatchPredictionRequest(BaseModel):
    """Represent a bounded collection of prediction requests."""

    requests: list[PredictionRequest] = Field(min_length=1, max_length=1_000)


class BatchPredictionResponse(BaseModel):
    """Represent predictions returned in request order."""

    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Represent HealthResponse."""

    status: str
    version: str


class ReadinessResponse(BaseModel):
    """Represent ReadinessResponse."""

    status: str
    model_name: str | None = None
