"""Validation module."""

from __future__ import annotations

import pandas as pd

from starwars_autocalls.data.schemas import (
    DAILY_VOL_SCHEMA,
    RFQ_SCHEMA,
    UNDERLYING_REFERENCE_SCHEMA,
)
from starwars_autocalls.features import parse_underlyings
from starwars_autocalls.features.builders import normalize_observation_frequency


def validate_rfqs(rfqs: pd.DataFrame) -> pd.DataFrame:
    """Perform validate rfqs."""
    validated = RFQ_SCHEMA.validate(rfqs)
    if validated["rfq_id"].duplicated().any():
        duplicates = validated.loc[validated["rfq_id"].duplicated(), "rfq_id"].head(10).tolist()
        raise ValueError(f"rfq_id must be unique; duplicates include: {duplicates}")
    invalid_dates = validated["end_date"].le(validated["start_date"])
    if invalid_dates.any():
        raise ValueError("end_date must be after start_date for every RFQ")
    starts_before_request = validated["start_date"].lt(validated["requested_date"])
    if starts_before_request.any():
        raise ValueError("start_date must be on or after requested_date for every RFQ")
    maturity_months = (validated["end_date"] - validated["start_date"]).dt.days / 30.4375
    if validated["no_call_period_months"].gt(maturity_months).any():
        raise ValueError("no_call_period_months must not exceed contractual maturity")
    target_known = validated["avg_duration_months"].notna()
    if not target_known.equals(validated["executed"]):
        raise ValueError("avg_duration_months must be present exactly when executed=True")
    unsupported_frequencies: list[str] = []
    for value in validated["observation_frequency"].dropna().unique():
        try:
            normalize_observation_frequency(value)
        except ValueError:
            unsupported_frequencies.append(str(value))
    if unsupported_frequencies:
        raise ValueError(f"Unsupported observation frequencies: {unsupported_frequencies}")
    return validated


def validate_daily_volatility(volatility: pd.DataFrame) -> pd.DataFrame:
    """Perform validate daily volatility."""
    validated = DAILY_VOL_SCHEMA.validate(volatility)
    duplicate_mask = validated.duplicated(["underlying", "date"])
    if duplicate_mask.any():
        sample = validated.loc[duplicate_mask, ["underlying", "date"]].head(10).to_dict("records")
        raise ValueError(f"Daily volatility contains duplicate underlying/date rows: {sample}")
    max_gap = (
        validated.sort_values(["underlying", "date"])
        .groupby("underlying")["date"]
        .diff()
        .dt.days.max()
    )
    if pd.notna(max_gap) and float(max_gap) > 14:
        raise ValueError(
            f"Daily volatility contains an extreme calendar gap of {int(max_gap)} days"
        )
    return validated


def validate_underlyings_reference(reference: pd.DataFrame) -> pd.DataFrame:
    """Perform validate underlyings reference."""
    return UNDERLYING_REFERENCE_SCHEMA.validate(reference)


def validate_underlying_membership(rfqs: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Perform validate underlying membership."""
    known = set(reference["underlying"])
    observed = {u for basket in rfqs["underlyings"] for u in parse_underlyings(basket)}
    missing = sorted(observed - known)
    if missing:
        raise ValueError(f"RFQs contain underlyings absent from reference table: {missing}")


def validate_market_history_coverage(rfqs: pd.DataFrame, volatility: pd.DataFrame) -> None:
    """Perform validate market history coverage."""
    first_market_date = volatility.groupby("underlying")["date"].min().to_dict()
    uncovered: list[str] = []
    for row in rfqs[["rfq_id", "requested_date", "underlyings"]].itertuples(index=False):
        for underlying in parse_underlyings(row.underlyings):
            if (
                underlying not in first_market_date
                or first_market_date[underlying] > row.requested_date
            ):
                uncovered.append(str(row.rfq_id))
                break
    if uncovered:
        raise ValueError(
            f"RFQs lack point-in-time market history on or before requested_date: {uncovered[:10]}"
        )


def validate_basket_structure(rfqs: pd.DataFrame) -> None:
    """Perform validate basket structure."""
    invalid_rows: list[str] = []
    duplicate_rows: list[str] = []
    for idx, row in rfqs[["rfq_id", "basket_type", "underlyings"]].iterrows():
        underlyings = parse_underlyings(row["underlyings"])
        if len(underlyings) != len(set(underlyings)):
            duplicate_rows.append(str(row.get("rfq_id", idx)))
        basket_type = row["basket_type"]
        if basket_type == "single" and len(underlyings) != 1:
            invalid_rows.append(str(row.get("rfq_id", idx)))
        if basket_type == "worst_of" and len(underlyings) < 2:
            invalid_rows.append(str(row.get("rfq_id", idx)))
    if duplicate_rows:
        sample = duplicate_rows[:10]
        raise ValueError(f"RFQs contain duplicate underlyings inside a basket: {sample}")
    if invalid_rows:
        sample = invalid_rows[:10]
        raise ValueError(f"RFQs contain basket_type values inconsistent with basket size: {sample}")


def validate_all(
    rfqs: pd.DataFrame, volatility: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform validate all."""
    rfqs = validate_rfqs(rfqs)
    volatility = validate_daily_volatility(volatility)
    reference = validate_underlyings_reference(reference)
    validate_underlying_membership(rfqs, reference)
    validate_basket_structure(rfqs)
    validate_market_history_coverage(rfqs, volatility)
    return rfqs, volatility, reference
