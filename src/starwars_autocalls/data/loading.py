"""Loading module."""

from __future__ import annotations

import pandas as pd

from starwars_autocalls.config import Settings

DATE_COLUMNS = ["requested_date", "start_date", "end_date"]
TRUE_VALUES = {"true", "1", "yes", "y", "si", "sí"}
FALSE_VALUES = {"false", "0", "no", "n"}


def parse_boolean_series(values: pd.Series, column_name: str) -> pd.Series:
    """Return parse boolean series."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    unknown = sorted(set(normalized.dropna()) - TRUE_VALUES - FALSE_VALUES)
    if unknown:
        raise ValueError(f"Unsupported boolean values in {column_name}: {unknown[:10]}")
    mapped = normalized.map(
        lambda value: True if value in TRUE_VALUES else False if value in FALSE_VALUES else pd.NA
    )
    if mapped.isna().any():
        raise ValueError(f"Missing boolean values in {column_name}")
    return mapped.astype(bool)


def load_rfqs(settings: Settings) -> pd.DataFrame:
    """Return load rfqs."""
    df = pd.read_csv(settings.rfqs_path)
    for column in DATE_COLUMNS:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    df["executed"] = parse_boolean_series(df["executed"], "executed")
    return df


def load_daily_volatility(settings: Settings) -> pd.DataFrame:
    """Return load daily volatility."""
    df = pd.read_csv(settings.volatility_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_underlyings_reference(settings: Settings) -> pd.DataFrame:
    """Return load underlyings reference."""
    return pd.read_csv(settings.underlyings_path)


def load_all(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return load all."""
    return (
        load_rfqs(settings),
        load_daily_volatility(settings),
        load_underlyings_reference(settings),
    )


def trainable_rfqs(rfqs: pd.DataFrame) -> pd.DataFrame:
    """Perform trainable rfqs."""
    mask = rfqs["executed"] & rfqs["avg_duration_months"].notna()
    return rfqs.loc[mask].copy().reset_index(drop=True)
