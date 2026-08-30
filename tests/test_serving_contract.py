from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from starwars_autocalls.data.loading import parse_boolean_series
from starwars_autocalls.data.schemas import BatchPredictionRequest, PredictionRequest
from starwars_autocalls.serving.prediction import _validate_serving_coverage


def build_valid_payload(**overrides: object) -> dict[str, object]:
    payload = {
        "product_type": "P1",
        "underlyings": " AAA ",
        "basket_type": "single",
        "autocall_barrier_pct": 1.0,
        "protection_barrier_pct": 0.6,
        "no_call_period_months": 6,
        "observation_frequency": "1M",
        "quoted_implied_vol": 0.25,
        "notional_credits": 100_000,
        "requested_date": "2024-06-01",
        "start_date": "2024-06-02",
        "end_date": "2027-06-02",
    }
    payload.update(overrides)
    return payload


def test_prediction_contract_normalizes_tickers() -> None:
    request = PredictionRequest.model_validate(build_valid_payload())
    assert request.underlyings == "AAA"


def test_prediction_contract_accepts_and_discards_historical_only_fields() -> None:
    request = PredictionRequest.model_validate(
        build_valid_payload(
            rfq_id="RFQ-NEW",
            executed="False",
            avg_duration_months="",
        )
    )

    assert set(request.model_dump()).isdisjoint({"rfq_id", "executed", "avg_duration_months"})


def test_prediction_contract_still_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(build_valid_payload(unexpected_column="value"))


def test_batch_prediction_contract_requires_between_one_and_one_thousand_requests() -> None:
    with pytest.raises(ValidationError):
        BatchPredictionRequest.model_validate({"requests": []})

    with pytest.raises(ValidationError):
        BatchPredictionRequest.model_validate({"requests": [build_valid_payload()] * 1_001})


@pytest.mark.parametrize(
    "overrides",
    [
        {"end_date": "2024-06-02"},
        {"requested_date": "2024-06-03"},
        {"no_call_period_months": 60},
        {"underlyings": "AAA|AAA"},
        {"basket_type": "worst_of", "underlyings": "AAA"},
        {"basket_type": "single", "underlyings": "AAA|BBB"},
        {"observation_frequency": "every blue moon"},
    ],
)
def test_prediction_contract_rejects_business_inconsistencies(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(build_valid_payload(**overrides))


def test_unknown_underlying_is_rejected_against_artifact_reference() -> None:
    request = PredictionRequest.model_validate(build_valid_payload(underlyings="ZZZ"))
    artifact = {
        "reference": pd.DataFrame({"underlying": ["AAA"]}),
        "volatility": pd.DataFrame({"underlying": ["AAA"], "date": pd.to_datetime(["2024-06-01"])}),
        "metadata": {},
    }
    with pytest.raises(ValueError, match="Unknown underlyings"):
        _validate_serving_coverage(request, artifact)


def test_stale_market_data_is_rejected() -> None:
    request = PredictionRequest.model_validate(build_valid_payload())
    artifact = {
        "reference": pd.DataFrame({"underlying": ["AAA"]}),
        "volatility": pd.DataFrame({"underlying": ["AAA"], "date": pd.to_datetime(["2024-05-01"])}),
        "metadata": {"max_market_data_staleness_days": 10},
    }
    with pytest.raises(ValueError, match="stale"):
        _validate_serving_coverage(request, artifact)


def test_boolean_parser_does_not_treat_false_string_as_true() -> None:
    parsed = parse_boolean_series(pd.Series(["True", "false", "1", "0"]), "executed")
    assert parsed.tolist() == [True, False, True, False]


def test_boolean_parser_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unsupported boolean"):
        parse_boolean_series(pd.Series(["maybe"]), "executed")
