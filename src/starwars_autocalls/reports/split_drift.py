"""Split Drift module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance


def write_split_drift_tables(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    split_column: str = "temporal_split",
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    """Compare each evaluation split with train using effect sizes and coverage."""
    compared_splits = [value for value in frame[split_column].unique() if value != "train"]
    numeric = _numeric_drift(frame, compared_splits, split_column)
    categorical = _categorical_drift(frame, compared_splits, split_column)
    missingness = _missingness_drift(frame, compared_splits, split_column)
    tables = {
        "numeric_drift": numeric,
        "categorical_drift": categorical,
        "missingness_drift": missingness,
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return tables, paths


def _numeric_drift(
    frame: pd.DataFrame, compared_splits: list[str], split_column: str
) -> pd.DataFrame:
    """Handle numeric drift."""
    excluded = {"row_id", "requested_year"}
    numeric_columns = [
        column for column in frame.select_dtypes(include="number").columns if column not in excluded
    ]
    train = frame.loc[frame[split_column] == "train"]
    rows = []
    for split_name in compared_splits:
        compared = frame.loc[frame[split_column] == split_name]
        for column in numeric_columns:
            baseline = pd.to_numeric(train[column], errors="coerce").dropna()
            current = pd.to_numeric(compared[column], errors="coerce").dropna()
            if baseline.empty or current.empty:
                continue
            train_std = float(baseline.std(ddof=0))
            standardized_mean_shift = (
                float((current.mean() - baseline.mean()) / train_std) if train_std else np.nan
            )
            ks = ks_2samp(baseline, current)
            status = _drift_status(
                warning=ks.statistic >= 0.10 or abs(standardized_mean_shift) >= 0.25,
                failure=ks.statistic >= 0.20 or abs(standardized_mean_shift) >= 0.50,
            )
            rows.append(
                {
                    "split": split_name,
                    "feature": column,
                    "train_rows": len(baseline),
                    "compared_rows": len(current),
                    "train_mean": float(baseline.mean()),
                    "compared_mean": float(current.mean()),
                    "standardized_mean_shift": standardized_mean_shift,
                    "ks_statistic": float(ks.statistic),
                    "ks_p_value": float(ks.pvalue),
                    "wasserstein_distance": float(wasserstein_distance(baseline, current)),
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def _categorical_drift(
    frame: pd.DataFrame, compared_splits: list[str], split_column: str
) -> pd.DataFrame:
    """Handle categorical drift."""
    excluded = {
        split_column,
        "rfq_id",
        "requested_date",
        "start_date",
        "end_date",
        "underlying_list",
        "underlyings",
        "requested_quarter",
        "requested_month",
    }
    categorical_columns = [
        column
        for column in frame.columns
        if column not in excluded
        and (
            frame[column].dtype == "object" or isinstance(frame[column].dtype, pd.CategoricalDtype)
        )
    ]
    train = frame.loc[frame[split_column] == "train"]
    rows = []
    for split_name in compared_splits:
        compared = frame.loc[frame[split_column] == split_name]
        for column in categorical_columns:
            train_values = train[column].fillna("__missing__").astype(str)
            compared_values = compared[column].fillna("__missing__").astype(str)
            categories = sorted(set(train_values).union(compared_values))
            train_share = train_values.value_counts(normalize=True).reindex(
                categories, fill_value=0.0
            )
            compared_share = compared_values.value_counts(normalize=True).reindex(
                categories, fill_value=0.0
            )
            js_divergence = float(jensenshannon(train_share, compared_share, base=2) ** 2)
            max_share_delta = float((compared_share - train_share).abs().max())
            status = _drift_status(
                warning=js_divergence >= 0.05 or max_share_delta >= 0.10,
                failure=js_divergence >= 0.10 or max_share_delta >= 0.20,
            )
            rows.append(
                {
                    "split": split_name,
                    "feature": column,
                    "train_categories": train_values.nunique(),
                    "compared_categories": compared_values.nunique(),
                    "jensen_shannon_divergence": js_divergence,
                    "max_category_share_delta": max_share_delta,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def _missingness_drift(
    frame: pd.DataFrame, compared_splits: list[str], split_column: str
) -> pd.DataFrame:
    """Handle missingness drift."""
    train = frame.loc[frame[split_column] == "train"]
    rows = []
    for split_name in compared_splits:
        compared = frame.loc[frame[split_column] == split_name]
        for column in frame.columns:
            if column == split_column:
                continue
            train_rate = float(train[column].isna().mean())
            compared_rate = float(compared[column].isna().mean())
            delta = compared_rate - train_rate
            rows.append(
                {
                    "split": split_name,
                    "feature": column,
                    "train_missing_rate": train_rate,
                    "compared_missing_rate": compared_rate,
                    "missing_rate_delta": delta,
                    "status": _drift_status(abs(delta) >= 0.05, abs(delta) >= 0.10),
                }
            )
    return pd.DataFrame(rows)


def _drift_status(warning: bool, failure: bool) -> str:
    """Handle drift status."""
    if failure:
        return "fail"
    if warning:
        return "warning"
    return "pass"
