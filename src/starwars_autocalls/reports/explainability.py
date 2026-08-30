"""Explainability module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from starwars_autocalls.config import RANDOM_SEED, Settings
from starwars_autocalls.data.loading import load_all, trainable_rfqs
from starwars_autocalls.data.validation import validate_all
from starwars_autocalls.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FeatureBuilder
from starwars_autocalls.modeling.evaluation import temporal_split
from starwars_autocalls.modeling.specs import (
    ModelSpec,
    build_pipeline,
    default_model_specs,
    spec_name,
)
from starwars_autocalls.modeling.tuning import load_tuned_specs
from starwars_autocalls.observability.progress import report_progress

TREE_EXPLAINABILITY_CANDIDATES = [
    spec_name("xgboost_ordinal", "all_without_commercial"),
    spec_name("catboost_native", "all_without_commercial"),
]


def _direction(value: float | None, threshold: float = 0.05) -> str:
    """Handle direction."""
    if value is None or not np.isfinite(value):
        return "not_applicable"
    if value > threshold:
        return "positive"
    if value < -threshold:
        return "negative"
    return "mixed_or_flat"


def _spearman(left: pd.Series, right: pd.Series) -> float | None:
    """Handle spearman."""
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return None
    value = frame["left"].corr(frame["right"], method="spearman")
    return None if pd.isna(value) else float(value)


def feature_signal_report(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    """Handle feature signal report."""
    numeric = numeric_features or NUMERIC_FEATURES
    categorical = categorical_features or CATEGORICAL_FEATURES
    rows: list[dict[str, object]] = []
    for feature in numeric:
        if feature not in X:
            continue
        corr = _spearman(pd.to_numeric(X[feature], errors="coerce"), y)
        rows.append(
            {
                "feature": feature,
                "feature_type": "numeric",
                "train_target_spearman": corr,
                "target_correlation_direction": _direction(corr),
                "n_unique": int(X[feature].nunique(dropna=True)),
                "missing_rate": float(X[feature].isna().mean()),
            }
        )
    for feature in categorical:
        if feature not in X:
            continue
        rows.append(
            {
                "feature": feature,
                "feature_type": "categorical",
                "train_target_spearman": None,
                "target_correlation_direction": "categorical_not_ordered",
                "n_unique": int(X[feature].nunique(dropna=True)),
                "missing_rate": float(X[feature].isna().mean()),
            }
        )
    return pd.DataFrame(rows)


def _transformed_frame(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Handle transformed frame."""
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        return X.copy()
    transformed = preprocessor.transform(X)
    if isinstance(transformed, pd.DataFrame):
        return transformed
    try:
        columns = preprocessor.get_feature_names_out()
    except Exception:
        columns = [f"feature_{idx}" for idx in range(transformed.shape[1])]
    return pd.DataFrame(transformed, columns=columns, index=X.index)


def _xgboost_shap_values(pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Handle xgboost shap values."""
    import xgboost as xgb

    transformed = _transformed_frame(pipeline, X)
    dmatrix = xgb.DMatrix(transformed, feature_names=list(transformed.columns))
    shap_values = pipeline.named_steps["model"].get_booster().predict(dmatrix, pred_contribs=True)
    return transformed, shap_values[:, :-1]


def _catboost_shap_values(pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Handle catboost shap values."""
    from catboost import Pool

    transformed = _transformed_frame(pipeline, X)
    cat_features = [feature for feature in transformed.columns if feature in CATEGORICAL_FEATURES]
    pool = Pool(transformed, cat_features=cat_features)
    shap_values = pipeline.named_steps["model"].get_feature_importance(pool, type="ShapValues")
    return transformed, shap_values[:, :-1]


def _shap_values(model_name: str, pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Handle shap values."""
    if model_name.startswith("xgboost"):
        return _xgboost_shap_values(pipeline, X)
    if model_name.startswith("catboost"):
        return _catboost_shap_values(pipeline, X)
    raise ValueError(f"SHAP explainability is not implemented for {model_name!r}.")


def _summarize_shap(
    model_name: str,
    transformed: pd.DataFrame,
    shap_values: np.ndarray,
    signal: pd.DataFrame,
) -> pd.DataFrame:
    """Handle summarize shap."""
    signal_by_feature = signal.set_index("feature").to_dict(orient="index")
    rows: list[dict[str, object]] = []
    for idx, feature in enumerate(transformed.columns):
        values = shap_values[:, idx]
        feature_type = "categorical" if feature in CATEGORICAL_FEATURES else "numeric"
        value_shap_corr = None
        shap_direction = "categorical_not_ordered"
        if feature_type == "numeric":
            value_shap_corr = _spearman(
                pd.to_numeric(transformed[feature], errors="coerce"),
                pd.Series(values, index=transformed.index),
            )
            shap_direction = _direction(value_shap_corr)
        signal_info = signal_by_feature.get(feature, {})
        rows.append(
            {
                "model_name": model_name,
                "feature": feature,
                "feature_type": feature_type,
                "mean_abs_shap": float(np.mean(np.abs(values))),
                "mean_shap": float(np.mean(values)),
                "feature_value_shap_spearman": value_shap_corr,
                "shap_direction": shap_direction,
                "train_target_spearman": signal_info.get("train_target_spearman"),
                "target_correlation_direction": signal_info.get("target_correlation_direction"),
                "missing_rate": signal_info.get("missing_rate"),
                "n_unique": signal_info.get("n_unique"),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("mean_abs_shap", ascending=False)
        .assign(shap_rank=lambda frame: np.arange(1, len(frame) + 1))
    )


def _categorical_effects(
    model_name: str,
    raw_X: pd.DataFrame,
    transformed: pd.DataFrame,
    shap_values: np.ndarray,
    max_categories: int = 25,
) -> pd.DataFrame:
    """Handle categorical effects."""
    rows: list[pd.DataFrame] = []
    for idx, feature in enumerate(transformed.columns):
        if feature not in CATEGORICAL_FEATURES or feature not in raw_X:
            continue
        values = pd.Series(shap_values[:, idx], index=raw_X.index, name="shap_value")
        frame = pd.DataFrame({"category": raw_X[feature].astype("string"), "shap_value": values})
        summary = (
            frame.groupby("category", dropna=False)["shap_value"]
            .agg(["mean", "count"])
            .sort_values("count", ascending=False)
            .head(max_categories)
            .reset_index()
        )
        summary.insert(0, "model_name", model_name)
        summary.insert(1, "feature", feature)
        rows.append(summary)
    if not rows:
        return pd.DataFrame(columns=["model_name", "feature", "category", "mean", "count"])
    return pd.concat(rows, ignore_index=True)


def _pruning_candidates(combined: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    """Handle pruning candidates."""
    ranked = combined.copy()
    ranked["relative_mean_abs_shap"] = ranked["mean_abs_shap"] / ranked.groupby("model_name")[
        "mean_abs_shap"
    ].transform("sum")
    consensus = (
        ranked.groupby("feature")
        .agg(
            model_count=("model_name", "nunique"),
            avg_rank=("shap_rank", "mean"),
            worst_rank=("shap_rank", "max"),
            avg_relative_mean_abs_shap=("relative_mean_abs_shap", "mean"),
            max_relative_mean_abs_shap=("relative_mean_abs_shap", "max"),
        )
        .reset_index()
    )
    consensus = consensus.merge(
        signal[
            [
                "feature",
                "feature_type",
                "train_target_spearman",
                "target_correlation_direction",
            ]
        ],
        on="feature",
        how="left",
    )
    consensus["abs_train_target_spearman"] = consensus["train_target_spearman"].abs()
    consensus["pruning_candidate"] = (
        (consensus["model_count"] >= 2)
        & (consensus["avg_rank"] >= 32)
        & (consensus["max_relative_mean_abs_shap"] < 0.01)
        & (consensus["abs_train_target_spearman"].fillna(0) < 0.08)
    )
    return consensus.sort_values(
        ["pruning_candidate", "avg_relative_mean_abs_shap"],
        ascending=[False, True],
    )


def _write_shap_bar_plot(summary: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    """Handle write shap bar plot."""
    import matplotlib.pyplot as plt

    top = summary.sort_values("mean_abs_shap", ascending=True).tail(top_n)
    colors = top["shap_direction"].map(
        {
            "positive": "#2b8cbe",
            "negative": "#d7301f",
            "mixed_or_flat": "#7f7f7f",
            "categorical_not_ordered": "#756bb1",
        }
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["mean_abs_shap"], color=colors.fillna("#7f7f7f"))
    ax.set_xlabel("mean absolute SHAP value")
    ax.set_ylabel("")
    ax.set_title(f"Top SHAP features - {summary['model_name'].iloc[0]}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _specs_for_explainability(
    settings: Settings,
    model_names: list[str],
) -> dict[str, tuple[ModelSpec, dict[str, Any]]]:
    """Handle specs for explainability."""
    tuned_specs = load_tuned_specs(settings, model_names)
    default_specs = {
        spec.name: (spec, {"source": "benchmark_default"}) for spec in default_model_specs()
    }
    specs: dict[str, tuple[ModelSpec, dict[str, Any]]] = {}
    for model_name in model_names:
        if model_name in tuned_specs:
            specs[model_name] = tuned_specs[model_name]
        elif model_name in default_specs:
            specs[model_name] = default_specs[model_name]
    return specs


def write_tree_shap_reports(
    settings: Settings,
    model_names: list[str] | None = None,
    max_rows: int = 1000,
) -> dict[str, str]:
    """Perform write tree shap reports."""
    settings.ensure_output_dirs()
    rfqs, volatility, reference = load_all(settings)
    rfqs, volatility, reference = validate_all(rfqs, volatility, reference)
    trainable = trainable_rfqs(rfqs)
    split = temporal_split(trainable)
    feature_set = FeatureBuilder().build(trainable, volatility, reference, include_target=True)
    X = feature_set.frame
    y = feature_set.target
    if y is None:
        raise ValueError("Explainability requires target values.")

    train_X = X.loc[split.train_index]
    train_y = y.loc[split.train_index]
    validation_X = X.loc[split.validation_index]
    if len(validation_X) > max_rows:
        validation_X = validation_X.sample(n=max_rows, random_state=RANDOM_SEED).sort_index()
    signal = feature_signal_report(
        train_X,
        train_y,
        feature_set.numeric_features,
        feature_set.categorical_features,
    )

    selected_model_names = model_names or TREE_EXPLAINABILITY_CANDIDATES
    specs = _specs_for_explainability(settings, selected_model_names)
    if not specs:
        raise ValueError("No explainability candidates are available.")

    outputs: dict[str, str] = {}
    signal_path = settings.explainability_dir / "feature_signal_analysis.csv"
    signal.to_csv(signal_path, index=False)
    outputs["feature_signal_analysis"] = str(signal_path)

    combined_rows: list[pd.DataFrame] = []
    categorical_rows: list[pd.DataFrame] = []
    for base_model_name, (spec, _) in specs.items():
        report_progress(f"Entrenando modelo para explicabilidad: {spec.name}")
        pipeline = build_pipeline(spec)
        pipeline.fit(train_X, train_y)
        report_progress(f"Calculando SHAP para {spec.name} sobre {len(validation_X):,} filas")
        transformed, shap_values = _shap_values(base_model_name, pipeline, validation_X)
        summary = _summarize_shap(spec.name, transformed, shap_values, signal)
        effects = _categorical_effects(spec.name, validation_X, transformed, shap_values)
        summary_path = settings.explainability_dir / f"shap_{spec.name}.csv"
        effects_path = settings.explainability_dir / f"shap_{spec.name}_categorical_effects.csv"
        figure_path = settings.explainability_figures_dir / f"shap_{spec.name}.png"
        summary.to_csv(summary_path, index=False)
        effects.to_csv(effects_path, index=False)
        _write_shap_bar_plot(summary, figure_path)
        outputs[f"shap_{spec.name}"] = str(summary_path)
        outputs[f"shap_{spec.name}_categorical_effects"] = str(effects_path)
        outputs[f"shap_{spec.name}_figure"] = str(figure_path)
        combined_rows.append(summary)
        categorical_rows.append(effects)

    combined = pd.concat(combined_rows, ignore_index=True)
    combined_path = settings.explainability_dir / "shap_tree_models_summary.csv"
    combined.to_csv(combined_path, index=False)
    outputs["shap_tree_models_summary"] = str(combined_path)
    pruning = _pruning_candidates(combined, signal)
    pruning_path = settings.explainability_dir / "feature_pruning_candidates.csv"
    pruning.to_csv(pruning_path, index=False)
    outputs["feature_pruning_candidates"] = str(pruning_path)
    if categorical_rows:
        categorical = pd.concat(categorical_rows, ignore_index=True)
        categorical_path = settings.explainability_dir / "shap_categorical_effects_summary.csv"
        categorical.to_csv(categorical_path, index=False)
        outputs["shap_categorical_effects_summary"] = str(categorical_path)
    return outputs
