"""Evaluation module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

TRAIN_END_YEAR = 2021
VALIDATION_YEAR = 2022
TEST_START_YEAR = 2023


@dataclass(frozen=True)
class TemporalSplit:
    """Represent TemporalSplit."""

    train_index: pd.Index
    validation_index: pd.Index
    test_index: pd.Index
    description: str


@dataclass(frozen=True)
class RollingTemporalFold:
    """Represent RollingTemporalFold."""

    train_index: pd.Index
    validation_index: pd.Index
    train_end_year: int
    validation_year: int
    description: str


def temporal_split(frame: pd.DataFrame, date_column: str = "requested_date") -> TemporalSplit:
    """Handle temporal split."""
    dates = pd.to_datetime(frame[date_column])
    train_index = frame.index[dates.dt.year <= TRAIN_END_YEAR]
    validation_index = frame.index[dates.dt.year == VALIDATION_YEAR]
    test_index = frame.index[dates.dt.year >= TEST_START_YEAR]

    if len(train_index) and len(validation_index) and len(test_index):
        return TemporalSplit(
            train_index=train_index,
            validation_index=validation_index,
            test_index=test_index,
            description=(
                f"train<={TRAIN_END_YEAR}, validation={VALIDATION_YEAR}, test>={TEST_START_YEAR}"
            ),
        )

    ordered = frame.sort_values(date_column)
    n = len(ordered)
    train_cut = int(n * 0.7)
    validation_cut = int(n * 0.85)
    return TemporalSplit(
        train_index=ordered.index[:train_cut],
        validation_index=ordered.index[train_cut:validation_cut],
        test_index=ordered.index[validation_cut:],
        description="70/15/15 chronological fallback split",
    )


def rolling_temporal_folds(
    frame: pd.DataFrame,
    date_column: str = "requested_date",
    min_train_years: int = 2,
    max_validation_year: int | None = VALIDATION_YEAR,
) -> list[RollingTemporalFold]:
    """Handle rolling temporal folds."""
    dates = pd.to_datetime(frame[date_column])
    years = sorted(dates.dt.year.dropna().unique())
    folds: list[RollingTemporalFold] = []
    for position, validation_year in enumerate(years):
        if max_validation_year is not None and validation_year > max_validation_year:
            continue
        train_years = years[:position]
        if len(train_years) < min_train_years:
            continue
        train_index = frame.index[dates.dt.year < validation_year]
        validation_index = frame.index[dates.dt.year == validation_year]
        if len(train_index) == 0 or len(validation_index) == 0:
            continue
        folds.append(
            RollingTemporalFold(
                train_index=train_index,
                validation_index=validation_index,
                train_end_year=int(max(train_years)),
                validation_year=int(validation_year),
                description=f"train<={int(max(train_years))}, validation={int(validation_year)}",
            )
        )
    return folds


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Handle regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
    }


def segment_mae(
    rows: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    column: str,
) -> pd.DataFrame:
    """Handle segment mae."""
    data = rows[[column]].copy()
    data["actual"] = y_true.to_numpy()
    data["prediction"] = y_pred
    data["absolute_error"] = (data["actual"] - data["prediction"]).abs()
    return (
        data.groupby(column, dropna=False)
        .agg(mae=("absolute_error", "mean"), count=("absolute_error", "size"))
        .reset_index()
        .sort_values(["mae", "count"], ascending=[False, False])
    )


def duration_bucket_metrics(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Handle duration bucket metrics."""
    data = pd.DataFrame({"actual": y_true.to_numpy(), "prediction": y_pred})
    data["duration_bucket"] = pd.cut(
        data["actual"],
        bins=[-0.01, 12, 24, 36, 60, np.inf],
        labels=["0-12", "12-24", "24-36", "36-60", "60+"],
    )
    data["absolute_error"] = (data["actual"] - data["prediction"]).abs()
    return (
        data.groupby("duration_bucket", observed=False)
        .agg(mae=("absolute_error", "mean"), count=("absolute_error", "size"))
        .reset_index()
    )
