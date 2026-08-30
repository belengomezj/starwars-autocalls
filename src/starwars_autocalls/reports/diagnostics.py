"""Diagnostics module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from starwars_autocalls.config import Settings
from starwars_autocalls.data.loading import load_all, trainable_rfqs
from starwars_autocalls.data.validation import validate_all
from starwars_autocalls.features import (
    CATEGORICAL_FEATURE_GROUPS,
    FEATURE_BLOCKS,
    NUMERIC_FEATURE_GROUPS,
    FeatureBuilder,
    select_feature_block,
)
from starwars_autocalls.modeling.evaluation import temporal_split

SPLIT_ORDER = ["train", "validation", "test"]
DECISION_SPLITS = ["train", "validation"]
SEGMENT_ORDER = ["single", "worst_of"]
BLOCK_COMPARISONS = [
    ("all_without_commercial", "all_without_noise"),
    ("all_without_noise", "global_stable"),
    ("global_stable", "global_stable_tail"),
    ("global_stable", "global_stable_no_sector"),
    ("global_stable", "global_risk_underlying"),
    ("global_risk_underlying", "global_all_underlying"),
    ("global_stable_tail", "global_tail_underlying"),
    ("single_core", "single_without_noise"),
    ("single_without_noise", "single_stable"),
    ("single_stable", "single_underlying"),
    ("single_underlying", "single_underlying_no_sector"),
    ("worst_of_core", "worst_of_without_noise"),
    ("worst_of_without_noise", "worst_of_stable"),
    ("worst_of_stable", "worst_of_tail_focus"),
    ("worst_of_stable", "worst_of_risk_underlying"),
    ("worst_of_tail_focus", "worst_of_tail_underlying"),
]


def run_feature_audit(settings: Settings) -> dict[str, Any]:
    """Audit engineered features on train and validation without consulting test."""
    settings.ensure_output_dirs()
    output_dir = settings.feature_audit_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rfqs, volatility, reference = load_all(settings)
    rfqs, volatility, reference = validate_all(rfqs, volatility, reference)
    trainable = trainable_rfqs(rfqs).reset_index(drop=True)
    split = temporal_split(trainable)
    feature_set = FeatureBuilder().build(trainable, volatility, reference, include_target=True)
    X = feature_set.frame
    y = feature_set.target
    if y is None:
        raise ValueError("Diagnostics require target values.")

    modeling_frame = _modeling_frame(trainable, X, y, split)
    decision_frame = _decision_modeling_frame(modeling_frame)
    tables = {
        "target_distribution_by_segment_split": _target_distribution(
            decision_frame,
            ["segment", "split"],
        ),
        "target_distribution_by_product_segment_split": _target_distribution(
            decision_frame,
            ["segment", "split", "product_type"],
        ),
        "target_outliers_by_segment": _target_outlier_summary(decision_frame),
        "target_outlier_rows": _target_outlier_rows(decision_frame),
        "categorical_mix_by_segment_split": _categorical_mix(decision_frame),
        "feature_drift_by_segment": _feature_drift(X, y, decision_frame),
        "feature_correlation_stability": _feature_correlation_stability(
            X,
            y,
            decision_frame,
        ),
        "feature_block_diff_summary": _feature_block_diff_summary(),
        "feature_block_inventory": _feature_block_inventory(),
        "feature_signal_comparison_best_blocks": _feature_signal_comparison_best_blocks(
            X,
            y,
            decision_frame,
        ),
    }

    paths: dict[str, str] = {}
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = str(path)

    summary = _feature_audit_summary(tables, split.description, paths)
    summary_path = output_dir / "feature_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["feature_audit_summary"] = str(summary_path)
    return {"summary": summary, "paths": paths}


def _decision_modeling_frame(modeling_frame: pd.DataFrame) -> pd.DataFrame:
    """Keep feature diagnostics scoped to train/validation selection data."""
    return modeling_frame.loc[modeling_frame["split"].isin(DECISION_SPLITS)].copy()


def _modeling_frame(
    trainable: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    split,
) -> pd.DataFrame:
    """Handle modeling frame."""
    frame = trainable[
        ["requested_date", "product_type", "basket_type", "observation_frequency", "underlyings"]
    ].copy()
    frame["target"] = y.to_numpy()
    frame["split"] = "unassigned"
    frame.loc[split.train_index, "split"] = "train"
    frame.loc[split.validation_index, "split"] = "validation"
    frame.loc[split.test_index, "split"] = "test"
    frame["segment"] = np.select(
        [X["is_single_underlying"].eq(1), X["is_worst_of"].eq(1)],
        ["single", "worst_of"],
        default="other",
    )
    frame["requested_year"] = pd.to_datetime(frame["requested_date"]).dt.year
    frame["duration_bucket"] = pd.cut(
        frame["target"],
        bins=[-np.inf, 12, 24, 36, 60, np.inf],
        labels=["<=12", "12-24", "24-36", "36-60", "60+"],
        right=True,
    )
    frame["duration_60_plus"] = frame["target"].ge(60)
    return frame


def _target_distribution(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Handle target distribution."""

    def quantile(level: float):
        """Handle quantile."""
        return lambda series: series.quantile(level)

    summary = (
        frame.groupby(group_cols, dropna=False, observed=False)["target"]
        .agg(
            rows="size",
            mean="mean",
            median="median",
            std="std",
            min="min",
            p10=quantile(0.10),
            p25=quantile(0.25),
            p75=quantile(0.75),
            p90=quantile(0.90),
            p95=quantile(0.95),
            max="max",
            duration_60_plus_rate=lambda series: series.ge(60).mean(),
        )
        .reset_index()
    )
    return summary.sort_values(group_cols).reset_index(drop=True)


def _target_outlier_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Handle target outlier summary."""
    rows: list[dict[str, Any]] = []
    for segment, group in frame.groupby("segment", dropna=False):
        baseline = group.loc[group["split"] == "train", "target"]
        q1 = float(baseline.quantile(0.25))
        q3 = float(baseline.quantile(0.75))
        iqr = q3 - q1
        high_threshold = q3 + 1.5 * iqr
        low_threshold = q1 - 1.5 * iqr
        for split_name, split_group in group.groupby("split", dropna=False):
            is_outlier = split_group["target"].lt(low_threshold) | split_group["target"].gt(
                high_threshold
            )
            rows.append(
                {
                    "segment": segment,
                    "split": split_name,
                    "rows": len(split_group),
                    "low_threshold": low_threshold,
                    "high_threshold": high_threshold,
                    "outlier_rows": int(is_outlier.sum()),
                    "outlier_rate": float(is_outlier.mean()) if len(split_group) else 0.0,
                    "duration_60_plus_rows": int(split_group["target"].ge(60).sum()),
                    "duration_60_plus_rate": float(split_group["target"].ge(60).mean())
                    if len(split_group)
                    else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["segment", "split"]).reset_index(drop=True)


def _target_outlier_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Handle target outlier rows."""
    rows = []
    for _segment, group in frame.groupby("segment", dropna=False):
        baseline = group.loc[group["split"] == "train", "target"]
        q1 = float(baseline.quantile(0.25))
        q3 = float(baseline.quantile(0.75))
        high_threshold = q3 + 1.5 * (q3 - q1)
        segment_outliers = group.loc[group["target"].gt(high_threshold)].copy()
        segment_outliers["segment_high_outlier_threshold"] = high_threshold
        rows.append(segment_outliers)
    if not rows:
        return pd.DataFrame()
    columns = [
        "segment",
        "split",
        "requested_date",
        "product_type",
        "basket_type",
        "observation_frequency",
        "underlyings",
        "target",
        "segment_high_outlier_threshold",
    ]
    return pd.concat(rows, ignore_index=True)[columns].sort_values(
        ["segment", "target"],
        ascending=[True, False],
    )


def _categorical_mix(frame: pd.DataFrame) -> pd.DataFrame:
    """Handle categorical mix."""
    rows = []
    for column in ["product_type", "basket_type", "observation_frequency", "duration_bucket"]:
        counts = (
            frame.groupby(["segment", "split", column], dropna=False, observed=False)
            .size()
            .reset_index(name="rows")
        )
        totals = counts.groupby(["segment", "split"], dropna=False)["rows"].transform("sum")
        counts["share"] = counts["rows"] / totals
        counts["column"] = column
        counts = counts.rename(columns={column: "value"})
        rows.append(counts[["column", "segment", "split", "value", "rows", "share"]])
    return pd.concat(rows, ignore_index=True).sort_values(["column", "segment", "split", "value"])


def _feature_drift(X: pd.DataFrame, y: pd.Series, frame: pd.DataFrame) -> pd.DataFrame:
    """Handle feature drift."""
    numeric_features = [column for column in X.columns if pd.api.types.is_numeric_dtype(X[column])]
    rows: list[dict[str, Any]] = []
    for segment in SEGMENT_ORDER:
        segment_index = frame.index[frame["segment"].eq(segment)]
        train_index = segment_index.intersection(frame.index[frame["split"].eq("train")])
        validation_index = segment_index.intersection(frame.index[frame["split"].eq("validation")])
        for feature in numeric_features:
            train_values = pd.to_numeric(X.loc[train_index, feature], errors="coerce")
            validation_values = pd.to_numeric(X.loc[validation_index, feature], errors="coerce")
            train_valid = train_values.dropna()
            validation_valid = validation_values.dropna()
            train_std = float(train_values.std(ddof=0))
            if train_valid.empty or validation_valid.empty:
                ks_statistic = np.nan
                ks_p_value = np.nan
                wasserstein = np.nan
            else:
                ks_result = ks_2samp(train_valid, validation_valid)
                ks_statistic = float(ks_result.statistic)
                ks_p_value = float(ks_result.pvalue)
                wasserstein = float(wasserstein_distance(train_valid, validation_valid))
            rows.append(
                {
                    "segment": segment,
                    "feature": feature,
                    "train_mean": float(train_values.mean()),
                    "validation_mean": float(validation_values.mean()),
                    "train_p95": float(train_values.quantile(0.95)),
                    "validation_p95": float(validation_values.quantile(0.95)),
                    "train_missing_rate": float(train_values.isna().mean()),
                    "validation_missing_rate": float(validation_values.isna().mean()),
                    "missing_rate_delta": float(
                        validation_values.isna().mean() - train_values.isna().mean()
                    ),
                    "ks_statistic": ks_statistic,
                    "ks_p_value": ks_p_value,
                    "wasserstein_distance": wasserstein,
                    "validation_shift_in_train_sd": _standardized_shift(
                        validation_values.mean(),
                        train_values.mean(),
                        train_std,
                    ),
                    "train_target_corr": _safe_corr(train_values, y.loc[train_index]),
                    "validation_target_corr": _safe_corr(
                        validation_values,
                        y.loc[validation_index],
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["abs_validation_shift_in_train_sd"] = result["validation_shift_in_train_sd"].abs()
    return result.sort_values(
        ["segment", "abs_validation_shift_in_train_sd"],
        ascending=[True, False],
    ).reset_index(drop=True)


def _feature_correlation_stability(
    X: pd.DataFrame,
    y: pd.Series,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Handle feature correlation stability."""
    drift = _feature_drift(X, y, frame)
    rows = []
    for row in drift.itertuples(index=False):
        correlations = [
            row.train_target_corr,
            row.validation_target_corr,
        ]
        finite = [value for value in correlations if pd.notna(value)]
        rows.append(
            {
                "segment": row.segment,
                "feature": row.feature,
                "train_target_corr": row.train_target_corr,
                "validation_target_corr": row.validation_target_corr,
                "corr_range": max(finite) - min(finite) if finite else np.nan,
                "validation_sign_flip_vs_train": _sign_flip(
                    row.train_target_corr,
                    row.validation_target_corr,
                ),
                "abs_validation_shift_in_train_sd": row.abs_validation_shift_in_train_sd,
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "segment",
            "validation_sign_flip_vs_train",
            "corr_range",
            "abs_validation_shift_in_train_sd",
        ],
        ascending=[True, False, False, False],
    )


def _feature_block_diff_summary() -> pd.DataFrame:
    """Handle feature block diff summary."""
    rows = []
    for from_block, to_block in BLOCK_COMPARISONS:
        from_features = FEATURE_BLOCKS[from_block]
        to_features = FEATURE_BLOCKS[to_block]
        removed = [feature for feature in from_features if feature not in to_features]
        added = [feature for feature in to_features if feature not in from_features]
        rows.append(
            {
                "from_block": from_block,
                "to_block": to_block,
                "from_feature_count": len(from_features),
                "to_feature_count": len(to_features),
                "removed_count": len(removed),
                "added_count": len(added),
                "removed_features": "|".join(removed),
                "added_features": "|".join(added),
            }
        )
    return pd.DataFrame(rows)


def _feature_block_inventory() -> pd.DataFrame:
    """Handle feature block inventory."""
    rows = []
    for block, features in FEATURE_BLOCKS.items():
        for feature in features:
            rows.append({"feature_block": block, "feature": feature})
    return pd.DataFrame(rows).sort_values(["feature_block", "feature"])


def _feature_signal_comparison_best_blocks(
    X: pd.DataFrame,
    y: pd.Series,
    modeling_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Handle feature signal comparison best blocks."""
    compared_blocks = {
        "global_52_all_without_noise": "all_without_noise",
        "single_27_compact_core": "compact_core",
        "worst_of_46_core": "worst_of_core",
    }
    block_features = {label: set(FEATURE_BLOCKS[block]) for label, block in compared_blocks.items()}
    all_compared_features = sorted(set().union(*block_features.values()))
    numeric_features, _ = select_feature_block("all_features")
    numeric_set = set(numeric_features)
    train_mask = modeling_frame["split"].eq("train")
    validation_mask = modeling_frame["split"].eq("validation")
    train_masks = {
        "global": X.index.isin(modeling_frame.index[train_mask]),
        "single": X.index.isin(modeling_frame.index[train_mask]) & X["is_single_underlying"].eq(1),
        "worst_of": X.index.isin(modeling_frame.index[train_mask]) & X["is_worst_of"].eq(1),
    }
    validation_masks = {
        "global": X.index.isin(modeling_frame.index[validation_mask]),
        "single": X.index.isin(modeling_frame.index[validation_mask])
        & X["is_single_underlying"].eq(1),
        "worst_of": X.index.isin(modeling_frame.index[validation_mask]) & X["is_worst_of"].eq(1),
    }

    rows: list[dict[str, Any]] = []
    for feature in all_compared_features:
        feature_type = "numeric" if feature in numeric_set else "categorical"
        membership = {label: feature in features for label, features in block_features.items()}
        included_labels = [label for label, included in membership.items() if included]
        signals = {
            f"{scope}_train_signal": _feature_signal(
                X,
                y,
                feature,
                train_masks[scope],
                feature_type,
            )
            for scope in train_masks
        }
        validation_signals = {
            f"{scope}_validation_signal": _feature_signal(
                X,
                y,
                feature,
                validation_masks[scope],
                feature_type,
            )
            for scope in validation_masks
        }
        train_signal_values = {scope: signals[f"{scope}_train_signal"] for scope in train_masks}
        highest_scope = _highest_signal_scope(train_signal_values)
        rows.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "feature_group": _feature_group(feature),
                "business_signal": _business_signal_label(feature),
                "in_global_52_all_without_noise": membership["global_52_all_without_noise"],
                "in_single_27_compact_core": membership["single_27_compact_core"],
                "in_worst_of_46_core": membership["worst_of_46_core"],
                "presence_pattern": "+".join(included_labels),
                "global_train_rows": int(train_masks["global"].sum()),
                "single_train_rows": int(train_masks["single"].sum()),
                "worst_of_train_rows": int(train_masks["worst_of"].sum()),
                "highest_train_signal_scope": highest_scope,
                "signal_metric": "abs_spearman"
                if feature_type == "numeric"
                else "category_median_spread_over_target_std",
                **signals,
                **validation_signals,
            }
        )
    result = pd.DataFrame(rows)
    signal_sort = result[
        ["global_train_signal", "single_train_signal", "worst_of_train_signal"]
    ].max(axis=1)
    result["_sort_signal"] = signal_sort
    return (
        result.sort_values(
            [
                "presence_pattern",
                "_sort_signal",
                "feature_group",
                "feature",
            ],
            ascending=[True, False, True, True],
        )
        .drop(columns="_sort_signal")
        .reset_index(drop=True)
    )


def _feature_signal(
    X: pd.DataFrame,
    y: pd.Series,
    feature: str,
    mask: pd.Series | np.ndarray,
    feature_type: str,
) -> float:
    """Handle feature signal."""
    mask_series = pd.Series(mask, index=X.index)
    values = X.loc[mask_series, feature]
    target = y.loc[mask_series]
    valid = values.notna() & target.notna()
    values = values.loc[valid]
    target = target.loc[valid]
    if len(values) < 30 or target.nunique(dropna=True) < 2:
        return np.nan
    if feature_type == "numeric":
        if values.nunique(dropna=True) < 2:
            return np.nan
        return abs(_safe_corr(values.rank(method="average"), target.rank(method="average")))
    grouped = (
        pd.DataFrame({"feature": values.astype("string"), "target": target})
        .groupby("feature", dropna=False)["target"]
        .agg(["median", "size"])
        .query("size >= 20")
    )
    if len(grouped) < 2:
        return np.nan
    target_std = float(target.std())
    if not np.isfinite(target_std) or target_std == 0:
        return np.nan
    return float((grouped["median"].max() - grouped["median"].min()) / target_std)


def _feature_group(feature: str) -> str:
    """Handle feature group."""
    for group, features in NUMERIC_FEATURE_GROUPS.items():
        if feature in features:
            return group
    for group, features in CATEGORICAL_FEATURE_GROUPS.items():
        if feature in features:
            return group
    return "unknown"


def _business_signal_label(feature: str) -> str:
    """Handle business signal label."""
    group = _feature_group(feature)
    labels = {
        "contractual": "estructura contractual y calendario de autocall",
        "product": "tipo de producto/madurez nominal como regimen",
        "basket": "composicion, complejidad y riesgo estructural de cesta",
        "market": "volatilidad implicita/realizada y spreads de mercado",
        "date": "regimen temporal del RFQ",
        "commercial": "señal comercial potencialmente inestable",
    }
    if feature in {"is_single_underlying", "is_worst_of"}:
        return "separacion explicita de regimen single/worst_of"
    return labels.get(group, "sin clasificar")


def _highest_signal_scope(signals: dict[str, float]) -> str:
    """Handle highest signal scope."""
    valid = {key: value for key, value in signals.items() if np.isfinite(value)}
    if not valid:
        return ""
    return max(valid, key=valid.get)


def _model_performance_summary(metrics_dir: Path) -> pd.DataFrame:
    """Handle model performance summary."""
    rows: list[dict[str, Any]] = []
    rows.extend(_benchmark_rows(metrics_dir / "benchmark_comparison.csv", "global_benchmark"))
    rows.extend(
        _benchmark_rows(
            metrics_dir / "global_stable_benchmark.csv",
            "global_stable_benchmark",
        )
    )
    rows.extend(_tuning_rows(metrics_dir / "tuning_comparison.csv", "global_tuning"))
    rows.extend(
        _tuning_rows(
            metrics_dir / "global_stable_tuning_comparison.csv",
            "global_stable_tuning",
        )
    )
    rows.extend(
        _benchmark_rows(
            metrics_dir / "segmented_benchmark_single_worstof.csv",
            "segmented_benchmark",
        )
    )
    rows.extend(_segmented_tuning_rows(metrics_dir / "segmented_tuning_comparison.csv"))
    final_path = metrics_dir / "final_test_metrics.csv"
    metadata_path = metrics_dir.parent.parent / "artifacts" / "model_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validation_metrics = (metadata.get("tuning_selection") or {}).get(
            "refit_validation_metrics",
            {},
        )
        test_metrics = metadata.get("test_metrics", {})
        rows.append(
            {
                "source": "final_model",
                "model_name": metadata.get("model_name", "final_model"),
                "segment": "global",
                "feature_block": metadata.get("feature_block", ""),
                "encoding_strategy": metadata.get("encoding_strategy", ""),
                "validation_mae": validation_metrics.get(
                    "mae",
                    metadata.get("selected_validation_mae", np.nan),
                ),
                "validation_rmse": validation_metrics.get("rmse", np.nan),
                "validation_r2": validation_metrics.get("r2", np.nan),
                "validation_median_absolute_error": validation_metrics.get(
                    "median_absolute_error",
                    np.nan,
                ),
                "test_mae": test_metrics.get("mae", np.nan),
                "test_rmse": test_metrics.get("rmse", np.nan),
                "test_r2": test_metrics.get("r2", np.nan),
                "test_median_absolute_error": test_metrics.get(
                    "median_absolute_error",
                    np.nan,
                ),
                "train_rows": metadata.get("train_rows", np.nan),
                "validation_rows": metadata.get("validation_rows", np.nan),
                "test_rows": metadata.get("test_rows", np.nan),
                "fit_seconds": np.nan,
                "validation_predict_seconds": np.nan,
                "final_fit_seconds": np.nan,
                "test_predict_seconds": np.nan,
            }
        )
    elif final_path.exists():
        final = pd.read_csv(final_path)
        if not final.empty:
            rows.append(
                {
                    "source": "final_model",
                    "model_name": "final_model",
                    "segment": "global",
                    "feature_block": "",
                    "encoding_strategy": "",
                    "validation_mae": np.nan,
                    "validation_rmse": np.nan,
                    "validation_r2": np.nan,
                    "validation_median_absolute_error": np.nan,
                    "test_mae": final.loc[0].get("mae", np.nan),
                    "test_rmse": final.loc[0].get("rmse", np.nan),
                    "test_r2": final.loc[0].get("r2", np.nan),
                    "test_median_absolute_error": final.loc[0].get(
                        "median_absolute_error",
                        np.nan,
                    ),
                    "train_rows": np.nan,
                    "validation_rows": np.nan,
                    "test_rows": np.nan,
                    "fit_seconds": np.nan,
                    "validation_predict_seconds": np.nan,
                    "final_fit_seconds": np.nan,
                    "test_predict_seconds": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _model_experiment_feature_comparison(metrics_dir: Path) -> pd.DataFrame:
    """Handle model experiment feature comparison."""
    performance = _model_performance_summary(metrics_dir)
    rows: list[dict[str, Any]] = []
    for row in performance.to_dict(orient="records"):
        rows.append(
            {
                **row,
                "evaluation_protocol": "temporal_holdout",
                **_feature_block_details(str(row.get("feature_block", ""))),
                "rolling_mae_mean": np.nan,
                "rolling_mae_std": np.nan,
                "rolling_mae_min": np.nan,
                "rolling_mae_max": np.nan,
                "rolling_r2_mean": np.nan,
                "rolling_n_folds": np.nan,
                "rolling_min_validation_rows": np.nan,
                "rolling_max_validation_rows": np.nan,
            }
        )

    rolling_path = metrics_dir / "segmented_rolling_benchmark_summary.csv"
    if rolling_path.exists():
        rolling = pd.read_csv(rolling_path)
        for row in rolling.to_dict(orient="records"):
            rows.append(
                {
                    "source": "segmented_rolling",
                    "model_name": row.get("model_name", ""),
                    "base_model_name": "",
                    "segment": row.get("segment", ""),
                    "feature_block": row.get("feature_block", ""),
                    "encoding_strategy": row.get("encoding_strategy", ""),
                    "evaluation_protocol": "rolling_yearly",
                    "validation_mae": np.nan,
                    "validation_rmse": np.nan,
                    "validation_r2": np.nan,
                    "validation_median_absolute_error": np.nan,
                    "test_mae": np.nan,
                    "test_rmse": np.nan,
                    "test_r2": np.nan,
                    "test_median_absolute_error": np.nan,
                    "train_rows": np.nan,
                    "validation_rows": np.nan,
                    "test_rows": np.nan,
                    "fit_seconds": row.get("mean_fit_seconds", np.nan),
                    "validation_predict_seconds": row.get(
                        "mean_validation_predict_seconds",
                        np.nan,
                    ),
                    "final_fit_seconds": np.nan,
                    "test_predict_seconds": np.nan,
                    **_feature_block_details(str(row.get("feature_block", ""))),
                    "rolling_mae_mean": row.get("rolling_mae_mean", np.nan),
                    "rolling_mae_std": row.get("rolling_mae_std", np.nan),
                    "rolling_mae_min": row.get("rolling_mae_min", np.nan),
                    "rolling_mae_max": row.get("rolling_mae_max", np.nan),
                    "rolling_r2_mean": row.get("rolling_r2_mean", np.nan),
                    "rolling_n_folds": row.get("n_folds", np.nan),
                    "rolling_min_validation_rows": row.get(
                        "min_validation_rows",
                        np.nan,
                    ),
                    "rolling_max_validation_rows": row.get(
                        "max_validation_rows",
                        np.nan,
                    ),
                }
            )

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison
    ordered_columns = [
        "source",
        "evaluation_protocol",
        "segment",
        "model_name",
        "base_model_name",
        "feature_block",
        "encoding_strategy",
        "feature_count",
        "numeric_feature_count",
        "categorical_feature_count",
        "validation_mae",
        "validation_rmse",
        "validation_r2",
        "validation_median_absolute_error",
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_median_absolute_error",
        "rolling_mae_mean",
        "rolling_mae_std",
        "rolling_mae_min",
        "rolling_mae_max",
        "rolling_r2_mean",
        "rolling_n_folds",
        "train_rows",
        "validation_rows",
        "test_rows",
        "rolling_min_validation_rows",
        "rolling_max_validation_rows",
        "fit_seconds",
        "validation_predict_seconds",
        "final_fit_seconds",
        "test_predict_seconds",
        "numeric_features",
        "categorical_features",
        "feature_names",
    ]
    existing_columns = [column for column in ordered_columns if column in comparison.columns]
    remaining_columns = [column for column in comparison.columns if column not in existing_columns]
    return comparison[existing_columns + remaining_columns].sort_values(
        ["segment", "source", "validation_mae"],
        na_position="last",
    )


def _feature_block_details(feature_block: str) -> dict[str, Any]:
    """Handle feature block details."""
    if feature_block not in FEATURE_BLOCKS:
        return {
            "feature_count": np.nan,
            "numeric_feature_count": np.nan,
            "categorical_feature_count": np.nan,
            "numeric_features": "",
            "categorical_features": "",
            "feature_names": "",
        }
    numeric_features, categorical_features = select_feature_block(feature_block)
    feature_names = numeric_features + categorical_features
    return {
        "feature_count": len(feature_names),
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "numeric_features": "|".join(numeric_features),
        "categorical_features": "|".join(categorical_features),
        "feature_names": "|".join(feature_names),
    }


def _benchmark_rows(path: Path, source: str) -> list[dict[str, Any]]:
    """Handle benchmark rows."""
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    rows = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            {
                "source": source,
                "model_name": row.get("model_name", ""),
                "segment": row.get("segment", "global"),
                "feature_block": row.get("feature_block", ""),
                "encoding_strategy": row.get("encoding_strategy", ""),
                "validation_mae": row.get("validation_mae", np.nan),
                "validation_rmse": row.get("validation_rmse", np.nan),
                "validation_r2": row.get("validation_r2", np.nan),
                "validation_median_absolute_error": row.get(
                    "validation_median_absolute_error",
                    np.nan,
                ),
                "test_mae": row.get("test_mae", np.nan),
                "test_rmse": row.get("test_rmse", np.nan),
                "test_r2": row.get("test_r2", np.nan),
                "test_median_absolute_error": row.get(
                    "test_median_absolute_error",
                    np.nan,
                ),
                "train_rows": row.get("train_rows", np.nan),
                "validation_rows": row.get("validation_rows", np.nan),
                "test_rows": row.get("test_rows", np.nan),
                "fit_seconds": row.get("fit_seconds", np.nan),
                "validation_predict_seconds": row.get("validation_predict_seconds", np.nan),
                "final_fit_seconds": row.get("final_fit_seconds", np.nan),
                "test_predict_seconds": row.get("test_predict_seconds", np.nan),
            }
        )
    return rows


def _tuning_rows(path: Path, source: str) -> list[dict[str, Any]]:
    """Handle tuning rows."""
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    rows = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            {
                "source": source,
                "model_name": row.get("tuned_model_name", row.get("base_model_name", "")),
                "base_model_name": row.get("base_model_name", ""),
                "segment": "global",
                "feature_block": row.get(
                    "feature_block",
                    _block_from_model_name(str(row.get("tuned_model_name", ""))),
                ),
                "encoding_strategy": row.get("encoding_strategy", ""),
                "validation_mae": row.get("refit_validation_mae", row.get("best_validation_mae")),
                "validation_rmse": row.get("validation_rmse", np.nan),
                "validation_r2": row.get("validation_r2", np.nan),
                "validation_median_absolute_error": row.get(
                    "validation_median_absolute_error",
                    np.nan,
                ),
                "test_mae": row.get("test_mae", np.nan),
                "test_rmse": row.get("test_rmse", np.nan),
                "test_r2": row.get("test_r2", np.nan),
                "test_median_absolute_error": row.get(
                    "test_median_absolute_error",
                    np.nan,
                ),
                "train_rows": row.get("train_rows", np.nan),
                "validation_rows": row.get("validation_rows", np.nan),
                "test_rows": row.get("test_rows", np.nan),
                "fit_seconds": row.get("fit_seconds", np.nan),
                "validation_predict_seconds": row.get("validation_predict_seconds", np.nan),
                "final_fit_seconds": row.get("final_fit_seconds", np.nan),
                "test_predict_seconds": row.get("test_predict_seconds", np.nan),
            }
        )
    return rows


def _segmented_tuning_rows(path: Path) -> list[dict[str, Any]]:
    """Handle segmented tuning rows."""
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    rows = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            {
                "source": "segmented_tuning",
                "model_name": row.get("tuned_model_name", ""),
                "base_model_name": row.get("base_model_name", ""),
                "segment": row.get("segment", ""),
                "feature_block": row.get("feature_block", ""),
                "encoding_strategy": row.get("encoding_strategy", ""),
                "validation_mae": row.get("refit_validation_mae", row.get("best_validation_mae")),
                "validation_rmse": np.nan,
                "validation_r2": np.nan,
                "validation_median_absolute_error": np.nan,
                "test_mae": row.get("test_mae", np.nan),
                "test_rmse": row.get("test_rmse", np.nan),
                "test_r2": row.get("test_r2", np.nan),
                "test_median_absolute_error": row.get(
                    "test_median_absolute_error",
                    np.nan,
                ),
                "train_rows": row.get("train_rows", np.nan),
                "validation_rows": row.get("validation_rows", np.nan),
                "test_rows": row.get("test_rows", np.nan),
                "fit_seconds": row.get("fit_seconds", np.nan),
                "validation_predict_seconds": row.get("validation_predict_seconds", np.nan),
                "final_fit_seconds": row.get("final_fit_seconds", np.nan),
                "test_predict_seconds": row.get("test_predict_seconds", np.nan),
            }
        )
    return rows


def _feature_audit_summary(
    tables: dict[str, pd.DataFrame],
    split_description: str,
    paths: dict[str, str],
) -> dict[str, Any]:
    """Handle feature audit summary."""
    target = tables["target_distribution_by_segment_split"]
    drift = tables["feature_drift_by_segment"]
    return {
        "split": split_description,
        "audit_policy": (
            "Feature diagnostics use train as baseline and validation as comparison. "
            "Test is excluded from every feature-treatment decision."
        ),
        "audit_splits": DECISION_SPLITS,
        "target_rows_by_segment_split": target[
            ["segment", "split", "rows", "mean", "median", "p95", "max", "duration_60_plus_rate"]
        ].to_dict(orient="records"),
        "largest_feature_drifts": drift.head(25).to_dict(orient="records"),
        "artifacts": paths,
    }


def _standardized_shift(value: float, baseline: float, baseline_std: float) -> float:
    """Handle standardized shift."""
    if not np.isfinite(baseline_std) or baseline_std == 0:
        return np.nan
    return float((value - baseline) / baseline_std)


def _safe_corr(values: pd.Series, target: pd.Series) -> float:
    """Handle safe corr."""
    if values.nunique(dropna=True) < 2 or target.nunique(dropna=True) < 2:
        return np.nan
    return float(values.corr(target))


def _sign_flip(left: float, right: float) -> bool:
    """Handle sign flip."""
    if pd.isna(left) or pd.isna(right) or left == 0 or right == 0:
        return False
    return bool(np.sign(left) != np.sign(right))


def _block_from_model_name(model_name: str) -> str:
    """Handle block from model name."""
    for block in sorted(FEATURE_BLOCKS, key=len, reverse=True):
        if block in model_name:
            return block
    return ""
