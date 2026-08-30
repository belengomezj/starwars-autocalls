"""Categorical Analysis module."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from starwars_autocalls.features import clean_observation_frequency, parse_underlyings
from starwars_autocalls.modeling.evaluation import temporal_split

RFQ_CATEGORICALS = [
    "product_type",
    "basket_type",
    "observation_frequency_clean",
    "counterparty",
    "trader_id",
    "underlyings",
]
UNDERLYING_CATEGORICALS = ["underlying", "sector"]
BASE_CATEGORICALS = [*RFQ_CATEGORICALS, *UNDERLYING_CATEGORICALS]


def _prepare_categorical_frames(
    rfqs: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Handle prepare categorical frames."""
    df = rfqs.reset_index(drop=True).copy()
    df["row_id"] = df.index
    df["observation_frequency_clean"] = df["observation_frequency"].map(clean_observation_frequency)
    df["underlying_list"] = df["underlyings"].map(parse_underlyings)
    exploded = df.explode("underlying_list").rename(columns={"underlying_list": "underlying"})
    return df, exploded.merge(reference[["underlying", "sector"]], on="underlying", how="left")


def _frame_for_feature(
    feature: str, rfq_frame: pd.DataFrame, underlying_frame: pd.DataFrame
) -> pd.DataFrame:
    """Handle frame for feature."""
    return underlying_frame if feature in UNDERLYING_CATEGORICALS else rfq_frame


def _add_split_labels(frame: pd.DataFrame, split) -> pd.DataFrame:
    """Handle add split labels."""
    labeled = frame.copy()
    labeled["split"] = "unused"
    labeled.loc[labeled["row_id"].isin(split.train_index), "split"] = "train"
    labeled.loc[labeled["row_id"].isin(split.validation_index), "split"] = "validation"
    labeled.loc[labeled["row_id"].isin(split.test_index), "split"] = "test"
    return labeled


def categorical_summary(rfqs: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Handle categorical summary."""
    rfq_frame, underlying_frame = _prepare_categorical_frames(rfqs, reference)
    rows = []
    for column in BASE_CATEGORICALS:
        frame = _frame_for_feature(column, rfq_frame, underlying_frame)
        counts = frame[column].value_counts(dropna=False)
        rare_count = int((counts < 20).sum())
        stats = (
            frame.groupby(column, dropna=False)["avg_duration_months"]
            .agg(["count", "mean", "median", "std"])
            .reset_index()
        )
        rows.append(
            {
                "feature": column,
                "observation_level": "underlying" if column in UNDERLYING_CATEGORICALS else "rfq",
                "cardinality": int(counts.shape[0]),
                "rare_categories_lt_20": rare_count,
                "top_category": str(counts.index[0]),
                "top_category_count": int(counts.iloc[0]),
                "target_mean_range": float(stats["mean"].max() - stats["mean"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values("cardinality", ascending=False)


def temporal_category_counts(rfqs: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Handle temporal category counts."""
    rfq_frame, underlying_frame = _prepare_categorical_frames(rfqs, reference)
    split = temporal_split(rfqs.reset_index(drop=True))
    rfq_frame = _add_split_labels(rfq_frame, split)
    underlying_frame = _add_split_labels(underlying_frame, split)
    parts = []
    for column in BASE_CATEGORICALS:
        frame = _frame_for_feature(column, rfq_frame, underlying_frame)
        counts = (
            frame.groupby(["split", column], dropna=False)
            .size()
            .reset_index(name="count")
            .assign(feature=column)
            .rename(columns={column: "category"})
        )
        counts["observation_level"] = "underlying" if column in UNDERLYING_CATEGORICALS else "rfq"
        parts.append(counts)
    return pd.concat(parts, ignore_index=True)


def unseen_categories_report(
    rfqs: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    include_test: bool = False,
) -> pd.DataFrame:
    """Handle unseen categories report."""
    counts = temporal_category_counts(rfqs, reference)
    rows = []
    for feature, group in counts.groupby("feature"):
        normalized = group.assign(category=group["category"].fillna("__missing__").astype(str))
        train_categories = set(normalized.loc[normalized["split"] == "train", "category"])
        compared_splits = ["validation", "test"] if include_test else ["validation"]
        for split_name in compared_splits:
            split_rows = normalized.loc[normalized["split"] == split_name]
            split_categories = set(split_rows["category"])
            unseen_values = split_categories - train_categories
            affected_rows = int(
                split_rows.loc[split_rows["category"].isin(unseen_values), "count"].sum()
            )
            total_rows = int(split_rows["count"].sum())
            rows.append(
                {
                    "feature": feature,
                    "split": split_name,
                    "observation_level": str(group["observation_level"].iloc[0]),
                    "unseen_categories": len(unseen_values),
                    "affected_rows": affected_rows,
                    "affected_row_share": affected_rows / total_rows if total_rows else 0.0,
                    "unseen_category_values": "|".join(sorted(unseen_values)[:25]),
                }
            )
    return pd.DataFrame(rows)


def write_categorical_analysis(
    rfqs: pd.DataFrame,
    reference: pd.DataFrame,
    output_dir: Path,
    *,
    include_test: bool = False,
) -> dict[str, Path]:
    """Perform write categorical analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = rfqs.reset_index(drop=True)
    split = temporal_split(normalized)
    analysis_index = split.train_index.union(split.validation_index)
    if include_test:
        analysis_index = analysis_index.union(split.test_index)
    analysis_rfqs = normalized.loc[analysis_index]
    summary = categorical_summary(analysis_rfqs, reference)
    temporal_counts = temporal_category_counts(rfqs, reference)
    if not include_test:
        temporal_counts = temporal_counts.loc[temporal_counts["split"] != "test"].copy()
    unseen = unseen_categories_report(rfqs, reference, include_test=include_test)
    target_stats = (
        analysis_rfqs.groupby(["product_type", "basket_type"], dropna=False)["avg_duration_months"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )

    paths = {
        "summary": output_dir / "categorical_analysis_summary.csv",
        "temporal_counts": output_dir / "categorical_temporal_counts.csv",
        "unseen": output_dir / "categorical_unseen_categories.csv",
        "target_stats": output_dir / "categorical_target_stats.csv",
        "notes": output_dir / "categorical_analysis_notes.json",
    }
    summary.to_csv(paths["summary"], index=False)
    temporal_counts.to_csv(paths["temporal_counts"], index=False)
    unseen.to_csv(paths["unseen"], index=False)
    target_stats.to_csv(paths["target_stats"], index=False)
    notes = {
        "interpretation": [
            "Counterparty and trader_id are included in the study but excluded from the default final feature block to reduce memorization risk.",
            "Underlyings are decomposed into basket-size, sector, structural volatility, and as-of realized volatility features instead of using raw basket strings as the main representation.",
            "One-hot and ordinal encoding are compared in benchmark results; optional native categorical libraries are used only if installed.",
        ]
    }
    paths["notes"].write_text(json.dumps(notes, indent=2), encoding="utf-8")
    return paths
