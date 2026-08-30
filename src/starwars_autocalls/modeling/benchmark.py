"""Benchmark module."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from starwars_autocalls.config import Settings
from starwars_autocalls.features import FeatureBuilder, parse_underlyings, select_feature_block
from starwars_autocalls.modeling.evaluation import (
    duration_bucket_metrics,
    regression_metrics,
    rolling_temporal_folds,
    segment_mae,
    temporal_split,
)
from starwars_autocalls.modeling.specs import (
    ModelSpec,
    ablation_specs,
    build_pipeline,
    default_model_specs,
    global_stable_specs,
    segmented_model_specs,
    split_spec_name,
)
from starwars_autocalls.observability.mlflow import (
    log_mlflow_run,
    metric_dataset_from_features,
    mlflow_dataset_name,
    short_data_hash_tags,
)
from starwars_autocalls.observability.progress import report_progress
from starwars_autocalls.reproducibility import data_hashes

BENCHMARK_PREDICTIONS_FILENAME = "benchmark_validation_predictions.csv"


def run_type_for_spec(spec: ModelSpec) -> str:
    """Perform run type for spec."""
    if spec.name.startswith(("global_", "median_by_")):
        return "baseline"
    if spec.name in {candidate.name for candidate in global_stable_specs()}:
        return "global_stable_benchmark"
    parsed = split_spec_name(spec.name)
    if parsed and parsed[0] == "hist_gradient_boosting_ablation":
        return "feature_block_ablation"
    return "model_benchmark"


def mlflow_run_name(spec: ModelSpec) -> str:
    """Handle mlflow run name."""
    return spec.name


def candidate_specs(
    model_names: list[str] | None = None,
    include_ablations: bool = True,
) -> list[ModelSpec]:
    """Handle candidate specs."""
    specs = default_model_specs()
    if include_ablations:
        specs.extend(ablation_specs())
    specs.extend(global_stable_specs())
    if not model_names:
        return specs
    return _filter_specs(specs, model_names, "benchmark")


def _filter_specs(
    specs: list[ModelSpec],
    model_names: list[str],
    command_context: str,
) -> list[ModelSpec]:
    """Handle filter specs."""
    wanted = set(model_names)
    available = {spec.name for spec in specs}
    unknown = sorted(wanted - available)
    if unknown:
        raise ValueError(
            f"Modelos desconocidos para {command_context}: {unknown}. "
            "Consulta los nombres disponibles en la documentación o en el benchmark."
        )
    return [spec for spec in specs if spec.name in wanted]


def segmented_candidate_specs(
    model_names: list[str] | None = None,
) -> dict[str, list[ModelSpec]]:
    """Handle segmented candidate specs."""
    specs_by_segment = segmented_model_specs()
    if not model_names:
        return specs_by_segment
    all_specs = [spec for specs in specs_by_segment.values() for spec in specs]
    selected = _filter_specs(all_specs, model_names, "benchmark segmentado")
    wanted = {spec.name for spec in selected}
    return {
        segment: [spec for spec in specs if spec.name in wanted]
        for segment, specs in specs_by_segment.items()
    }


def global_stable_candidate_specs(model_names: list[str] | None = None) -> list[ModelSpec]:
    """Handle global stable candidate specs."""
    specs = global_stable_specs()
    if not model_names:
        return specs
    return _filter_specs(specs, model_names, "experimento global estable")


def _rows_per_second(row_count: int, seconds: float) -> float:
    """Handle rows per second."""
    if seconds <= 0:
        return 0.0
    return float(row_count / seconds)


def evaluate_spec(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    train_index: pd.Index,
    validation_index: pd.Index,
) -> dict[str, object]:
    """Perform evaluate spec."""
    report_progress(
        f"Entrenando {spec.name} ({len(train_index):,} train / "
        f"{len(validation_index):,} validation)"
    )
    pipeline = build_pipeline(spec)
    fit_start = time.perf_counter()
    pipeline.fit(X.loc[train_index], y.loc[train_index])
    fit_seconds = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    predictions = pipeline.predict(X.loc[validation_index])
    predict_seconds = time.perf_counter() - predict_start
    metrics = regression_metrics(y.loc[validation_index], predictions)
    report_progress(f"Completado {spec.name}: MAE={metrics['mae']:.4f}, fit={fit_seconds:.2f}s")
    return {
        "model_name": spec.name,
        "estimator_class": spec.estimator.__class__.__name__,
        "feature_block": spec.feature_block,
        "encoding_strategy": spec.encoding_strategy,
        **{f"validation_{key}": value for key, value in metrics.items()},
        "fit_seconds": fit_seconds,
        "validation_predict_seconds": predict_seconds,
        "validation_rows_per_second_predict": _rows_per_second(
            len(validation_index), predict_seconds
        ),
        "pipeline": pipeline,
        "validation_predictions": pd.Series(predictions, index=validation_index),
    }


def run_benchmark(
    rfqs: pd.DataFrame,
    volatility: pd.DataFrame,
    reference: pd.DataFrame,
    settings: Settings,
    include_ablations: bool = True,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Perform run benchmark."""
    settings.ensure_output_dirs()
    rfqs = rfqs.reset_index(drop=True)
    split = temporal_split(rfqs)
    feature_set = FeatureBuilder().build(rfqs, volatility, reference, include_target=True)
    X = feature_set.frame
    y = feature_set.target
    if y is None:
        raise ValueError("Benchmark requires target values.")

    specs = candidate_specs(
        include_ablations=include_ablations,
        model_names=model_names,
    )
    hash_tags = short_data_hash_tags(data_hashes(settings))

    rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for spec_index, spec in enumerate(specs, start=1):
        report_progress(f"Benchmark global {spec_index}/{len(specs)}: {spec.name}")
        numeric_features, categorical_features = select_feature_block(spec.feature_block)
        feature_columns = [*numeric_features, *categorical_features]
        run_type = run_type_for_spec(spec)
        result = evaluate_spec(spec, X, y, split.train_index, split.validation_index)
        result.pop("pipeline")
        predictions = result.pop("validation_predictions")
        prediction_rows.append(
            pd.DataFrame(
                {
                    "model_name": spec.name,
                    "row_index": split.validation_index,
                    "rfq_id": rfqs.loc[split.validation_index, "rfq_id"].to_numpy(),
                    "actual": y.loc[split.validation_index].to_numpy(),
                    "prediction": predictions.to_numpy(),
                }
            )
        )
        result["n_numeric_features"] = len(numeric_features)
        result["n_categorical_features"] = len(categorical_features)
        result["n_total_features"] = len(numeric_features) + len(categorical_features)
        rows.append(result.copy())
        validation_dataset = metric_dataset_from_features(
            X,
            y,
            split.validation_index,
            feature_columns,
        )
        log_mlflow_run(
            run_name=mlflow_run_name(spec),
            params={
                "model_name": result["model_name"],
                "run_type": run_type,
                "estimator_class": result["estimator_class"],
                "feature_block": result["feature_block"],
                "encoding_strategy": result["encoding_strategy"],
                "n_numeric_features": len(numeric_features),
                "n_categorical_features": len(categorical_features),
                "n_total_features": len(numeric_features) + len(categorical_features),
                "split": split.description,
            },
            metrics={
                key: float(value)
                for key, value in result.items()
                if (
                    key.startswith("validation_")
                    or key in {"fit_seconds", "validation_predict_seconds"}
                )
                and isinstance(value, float)
            },
            tags={
                "run_type": run_type,
                "model_name": str(result["model_name"]),
                "feature_block": str(result["feature_block"]),
                "encoding_strategy": str(result["encoding_strategy"]),
                "estimator_class": str(result["estimator_class"]),
                **hash_tags,
            },
            metric_dataset=validation_dataset,
            dataset_name=mlflow_dataset_name(spec.feature_block, "validation"),
        )

    comparison = pd.DataFrame(rows).sort_values("validation_mae")
    output = settings.metrics_dir / "benchmark_comparison.csv"
    comparison.to_csv(output, index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        settings.metrics_dir / BENCHMARK_PREDICTIONS_FILENAME,
        index=False,
    )
    return comparison


def run_global_stable_experiment(
    rfqs: pd.DataFrame,
    volatility: pd.DataFrame,
    reference: pd.DataFrame,
    settings: Settings,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate global models with diagnostics-driven feature treatments."""
    settings.ensure_output_dirs()
    rfqs = rfqs.reset_index(drop=True)
    split = temporal_split(rfqs)
    feature_set = FeatureBuilder().build(rfqs, volatility, reference, include_target=True)
    X = feature_set.frame
    y = feature_set.target
    if y is None:
        raise ValueError("Global stable experiment requires target values.")

    specs = global_stable_candidate_specs(model_names=model_names)
    hash_tags = short_data_hash_tags(data_hashes(settings))
    rows: list[dict[str, object]] = []
    for spec_index, spec in enumerate(specs, start=1):
        report_progress(f"Benchmark estable {spec_index}/{len(specs)}: {spec.name}")
        numeric_features, categorical_features = select_feature_block(spec.feature_block)
        feature_columns = [*numeric_features, *categorical_features]
        validation_result = evaluate_spec(spec, X, y, split.train_index, split.validation_index)
        validation_result.pop("pipeline")
        row = {
            **validation_result,
            "segment_strategy": "global",
            "segment": "all",
            "n_numeric_features": len(numeric_features),
            "n_categorical_features": len(categorical_features),
            "n_total_features": len(numeric_features) + len(categorical_features),
            "train_rows": len(split.train_index),
            "validation_rows": len(split.validation_index),
            "split": split.description,
        }
        rows.append(row)
        validation_dataset = metric_dataset_from_features(
            X,
            y,
            split.validation_index,
            feature_columns,
        )
        log_mlflow_run(
            run_name=spec.name,
            params={
                "model_name": spec.name,
                "run_type": "global_stable_benchmark",
                "estimator_class": spec.estimator.__class__.__name__,
                "feature_block": spec.feature_block,
                "encoding_strategy": spec.encoding_strategy,
                "split": split.description,
                "train_rows": len(split.train_index),
                "validation_rows": len(split.validation_index),
                "n_numeric_features": len(numeric_features),
                "n_categorical_features": len(categorical_features),
                "n_total_features": len(numeric_features) + len(categorical_features),
            },
            metrics={
                **{
                    key: float(value)
                    for key, value in validation_result.items()
                    if (
                        key.startswith("validation_")
                        or key in {"fit_seconds", "validation_predict_seconds"}
                    )
                    and isinstance(value, float)
                },
            },
            tags={
                "run_type": "global_stable_benchmark",
                "model_name": spec.name,
                "feature_block": spec.feature_block,
                "encoding_strategy": spec.encoding_strategy,
                "estimator_class": spec.estimator.__class__.__name__,
                **hash_tags,
            },
            metric_dataset=validation_dataset,
            dataset_name=mlflow_dataset_name(spec.feature_block, "validation"),
        )

    detail = pd.DataFrame(rows)
    if not detail.empty:
        detail = detail.sort_values("validation_mae")
    detail.to_csv(settings.metrics_dir / "global_stable_benchmark.csv", index=False)
    return detail


def run_rolling_benchmark(
    rfqs: pd.DataFrame,
    volatility: pd.DataFrame,
    reference: pd.DataFrame,
    settings: Settings,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Perform run rolling benchmark."""
    settings.ensure_output_dirs()
    rfqs = rfqs.reset_index(drop=True)
    feature_set = FeatureBuilder().build(rfqs, volatility, reference, include_target=True)
    X = feature_set.frame
    y = feature_set.target
    if y is None:
        raise ValueError("Rolling benchmark requires target values.")

    specs = candidate_specs(model_names=model_names)
    folds = rolling_temporal_folds(rfqs)
    hash_tags = short_data_hash_tags(data_hashes(settings))
    rows: list[dict[str, object]] = []
    for spec_index, spec in enumerate(specs, start=1):
        for fold_index, fold in enumerate(folds, start=1):
            report_progress(
                f"Rolling modelo {spec_index}/{len(specs)}, fold {fold_index}/{len(folds)} "
                f"({fold.description}): {spec.name}"
            )
            result = evaluate_spec(spec, X, y, fold.train_index, fold.validation_index)
            result.pop("pipeline")
            rows.append(
                {
                    **result,
                    "train_end_year": fold.train_end_year,
                    "validation_year": fold.validation_year,
                    "split": fold.description,
                }
            )
    detail = pd.DataFrame(rows)
    detail_path = settings.metrics_dir / "rolling_benchmark_by_year.csv"
    detail.to_csv(detail_path, index=False)
    if detail.empty:
        return detail
    summary = (
        detail.groupby(["model_name", "feature_block", "encoding_strategy"], dropna=False)
        .agg(
            rolling_mae_mean=("validation_mae", "mean"),
            rolling_mae_std=("validation_mae", "std"),
            rolling_mae_max=("validation_mae", "max"),
            n_folds=("validation_mae", "size"),
        )
        .reset_index()
        .sort_values(["rolling_mae_mean", "rolling_mae_std"])
    )
    summary.to_csv(settings.metrics_dir / "rolling_benchmark_summary.csv", index=False)
    for row in summary.to_dict(orient="records"):
        model_name = str(row["model_name"])
        detail_subset = detail[detail["model_name"].astype(str).eq(model_name)].copy()
        log_mlflow_run(
            run_name=f"{model_name}__rolling_summary",
            params={
                "model_name": model_name,
                "run_type": "rolling_benchmark",
                "evaluation_protocol": "rolling_yearly",
                "split_role": "rolling_validation",
                "feature_block": str(row["feature_block"]),
                "encoding_strategy": str(row["encoding_strategy"]),
                "n_folds": int(row["n_folds"]),
            },
            metrics={
                "rolling_mae_mean": float(row["rolling_mae_mean"]),
                "rolling_mae_std": float(row["rolling_mae_std"])
                if pd.notna(row["rolling_mae_std"])
                else 0.0,
                "rolling_mae_max": float(row["rolling_mae_max"]),
                "n_folds": float(row["n_folds"]),
            },
            artifacts={"rolling_fold_detail": detail_subset},
            tags={
                "run_type": "rolling_benchmark",
                "evaluation_protocol": "rolling_yearly",
                "split_role": "rolling_validation",
                "model_name": model_name,
                "feature_block": str(row["feature_block"]),
                "encoding_strategy": str(row["encoding_strategy"]),
                **hash_tags,
            },
        )
    return summary


def run_segmented_rolling_benchmark(
    rfqs: pd.DataFrame,
    volatility: pd.DataFrame,
    reference: pd.DataFrame,
    settings: Settings,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Rolling temporal validation for independently trained segment models."""
    settings.ensure_output_dirs()
    rfqs = rfqs.reset_index(drop=True)
    feature_set = FeatureBuilder().build(rfqs, volatility, reference, include_target=True)
    X = feature_set.frame
    y = feature_set.target
    if y is None:
        raise ValueError("Segmented rolling benchmark requires target values.")

    specs_by_segment = segmented_candidate_specs(model_names=model_names)
    folds = rolling_temporal_folds(rfqs)
    hash_tags = short_data_hash_tags(data_hashes(settings))
    segment_masks = {
        "single": X["is_single_underlying"].eq(1),
        "worst_of": X["is_worst_of"].eq(1),
    }
    rows: list[dict[str, object]] = []
    for segment_name, specs in specs_by_segment.items():
        segment_index = X.index[segment_masks[segment_name]]
        for fold_index, fold in enumerate(folds, start=1):
            train_index = fold.train_index.intersection(segment_index)
            validation_index = fold.validation_index.intersection(segment_index)
            if len(train_index) == 0 or len(validation_index) == 0:
                continue
            for spec_index, spec in enumerate(specs, start=1):
                report_progress(
                    f"Rolling segmentado {segment_name}: modelo {spec_index}/{len(specs)}, "
                    f"fold {fold_index}/{len(folds)} ({fold.description}): {spec.name}"
                )
                numeric_features, categorical_features = select_feature_block(spec.feature_block)
                result = evaluate_spec(spec, X, y, train_index, validation_index)
                result.pop("pipeline")
                rows.append(
                    {
                        **result,
                        "segment_strategy": "basket_type_segmented",
                        "segment": segment_name,
                        "train_end_year": fold.train_end_year,
                        "validation_year": fold.validation_year,
                        "train_rows": len(train_index),
                        "validation_rows": len(validation_index),
                        "n_numeric_features": len(numeric_features),
                        "n_categorical_features": len(categorical_features),
                        "n_total_features": len(numeric_features) + len(categorical_features),
                        "split": fold.description,
                    }
                )

    detail = pd.DataFrame(rows)
    detail_path = settings.metrics_dir / "segmented_rolling_benchmark_by_year.csv"
    detail.to_csv(detail_path, index=False)
    if detail.empty:
        detail.to_csv(settings.metrics_dir / "segmented_rolling_benchmark_summary.csv", index=False)
        return detail

    summary = (
        detail.groupby(
            ["segment", "model_name", "feature_block", "encoding_strategy"], dropna=False
        )
        .agg(
            rolling_mae_mean=("validation_mae", "mean"),
            rolling_mae_std=("validation_mae", "std"),
            rolling_mae_min=("validation_mae", "min"),
            rolling_mae_max=("validation_mae", "max"),
            rolling_rmse_mean=("validation_rmse", "mean"),
            rolling_r2_mean=("validation_r2", "mean"),
            mean_fit_seconds=("fit_seconds", "mean"),
            mean_validation_predict_seconds=("validation_predict_seconds", "mean"),
            min_validation_rows=("validation_rows", "min"),
            max_validation_rows=("validation_rows", "max"),
            n_folds=("validation_mae", "size"),
            n_total_features=("n_total_features", "first"),
        )
        .reset_index()
        .sort_values(["segment", "rolling_mae_mean", "rolling_mae_max"])
    )
    summary.to_csv(settings.metrics_dir / "segmented_rolling_benchmark_summary.csv", index=False)
    for row in summary.to_dict(orient="records"):
        model_name = str(row["model_name"])
        segment_name = str(row["segment"])
        detail_subset = detail[
            detail["model_name"].astype(str).eq(model_name)
            & detail["segment"].astype(str).eq(segment_name)
        ].copy()
        log_mlflow_run(
            run_name=f"{model_name}__rolling_summary",
            params={
                "segment": segment_name,
                "model_name": model_name,
                "run_type": "segmented_rolling_benchmark",
                "evaluation_protocol": "rolling_yearly",
                "split_role": "rolling_validation",
                "feature_block": str(row["feature_block"]),
                "encoding_strategy": str(row["encoding_strategy"]),
                "n_folds": int(row["n_folds"]),
            },
            metrics={
                "rolling_mae_mean": float(row["rolling_mae_mean"]),
                "rolling_mae_std": float(row["rolling_mae_std"])
                if pd.notna(row["rolling_mae_std"])
                else 0.0,
                "rolling_mae_min": float(row["rolling_mae_min"]),
                "rolling_mae_max": float(row["rolling_mae_max"]),
                "rolling_rmse_mean": float(row["rolling_rmse_mean"]),
                "rolling_r2_mean": float(row["rolling_r2_mean"]),
                "n_folds": float(row["n_folds"]),
            },
            artifacts={"rolling_fold_detail": detail_subset},
            tags={
                "run_type": "segmented_rolling_benchmark",
                "evaluation_protocol": "rolling_yearly",
                "split_role": "rolling_validation",
                "segment": segment_name,
                "model_name": model_name,
                "feature_block": str(row["feature_block"]),
                "encoding_strategy": str(row["encoding_strategy"]),
                **hash_tags,
            },
        )

    weighted = _segmented_rolling_weighted_summary(detail)
    if not weighted.empty:
        weighted.to_csv(
            settings.metrics_dir / "segmented_rolling_benchmark_weighted_by_year.csv",
            index=False,
        )
    return summary


def _segmented_rolling_weighted_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Handle segmented rolling weighted summary."""
    rows: list[dict[str, object]] = []
    for validation_year, group in detail.groupby("validation_year", dropna=False):
        model_count_by_segment = group.groupby("segment", dropna=False)["model_name"].nunique()
        if model_count_by_segment.max() != 1:
            continue
        total_rows = int(group["validation_rows"].sum())
        if total_rows == 0:
            continue
        rows.append(
            {
                "validation_year": validation_year,
                "weighted_validation_mae": float(
                    (group["validation_mae"] * group["validation_rows"]).sum() / total_rows
                ),
                "validation_rows": total_rows,
                "segments": ",".join(sorted(group["segment"].astype(str).unique())),
                "portfolio_model_names": " + ".join(sorted(group["model_name"].astype(str))),
            }
        )
    weighted = pd.DataFrame(rows)
    if weighted.empty:
        return weighted
    return weighted.sort_values("validation_year")


def run_segmented_benchmark(
    rfqs: pd.DataFrame,
    volatility: pd.DataFrame,
    reference: pd.DataFrame,
    settings: Settings,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate models trained separately for single and worst_of products."""
    settings.ensure_output_dirs()
    rfqs = rfqs.reset_index(drop=True)
    split = temporal_split(rfqs)
    feature_set = FeatureBuilder().build(rfqs, volatility, reference, include_target=True)
    X = feature_set.frame
    y = feature_set.target
    if y is None:
        raise ValueError("Segmented benchmark requires target values.")

    specs_by_segment = segmented_candidate_specs(model_names=model_names)
    hash_tags = short_data_hash_tags(data_hashes(settings))
    segment_masks = {
        "single": X["is_single_underlying"].eq(1),
        "worst_of": X["is_worst_of"].eq(1),
    }
    rows: list[dict[str, object]] = []
    for segment_name, specs in specs_by_segment.items():
        segment_mask = segment_masks[segment_name]
        train_index = split.train_index.intersection(X.index[segment_mask])
        validation_index = split.validation_index.intersection(X.index[segment_mask])
        if len(train_index) == 0 or len(validation_index) == 0:
            continue
        for spec_index, spec in enumerate(specs, start=1):
            report_progress(
                f"Benchmark segmentado {segment_name} {spec_index}/{len(specs)}: {spec.name}"
            )
            numeric_features, categorical_features = select_feature_block(spec.feature_block)
            feature_columns = [*numeric_features, *categorical_features]
            validation_result = evaluate_spec(spec, X, y, train_index, validation_index)
            validation_result.pop("pipeline")
            row = {
                **validation_result,
                "segment_strategy": "basket_type_segmented",
                "segment": segment_name,
                "n_numeric_features": len(numeric_features),
                "n_categorical_features": len(categorical_features),
                "n_total_features": len(numeric_features) + len(categorical_features),
                "train_rows": len(train_index),
                "validation_rows": len(validation_index),
                "split": split.description,
            }
            rows.append(row)
            metric_dataset = metric_dataset_from_features(
                X,
                y,
                validation_index,
                feature_columns,
            )
            log_mlflow_run(
                run_name=spec.name,
                params={
                    "model_name": spec.name,
                    "run_type": "segmented_benchmark",
                    "segment": segment_name,
                    "estimator_class": spec.estimator.__class__.__name__,
                    "feature_block": spec.feature_block,
                    "encoding_strategy": spec.encoding_strategy,
                    "split": split.description,
                    "train_rows": len(train_index),
                    "validation_rows": len(validation_index),
                    "n_numeric_features": len(numeric_features),
                    "n_categorical_features": len(categorical_features),
                    "n_total_features": len(numeric_features) + len(categorical_features),
                },
                metrics={
                    **{
                        key: float(value)
                        for key, value in validation_result.items()
                        if (
                            key.startswith("validation_")
                            or key in {"fit_seconds", "validation_predict_seconds"}
                        )
                        and isinstance(value, float)
                    },
                },
                tags={
                    "run_type": "segmented_benchmark",
                    "segment": segment_name,
                    "model_name": spec.name,
                    "feature_block": spec.feature_block,
                    "encoding_strategy": spec.encoding_strategy,
                    "estimator_class": spec.estimator.__class__.__name__,
                    **hash_tags,
                },
                metric_dataset=metric_dataset,
                dataset_name=mlflow_dataset_name(spec.feature_block, "validation"),
            )
    detail = pd.DataFrame(rows)
    if not detail.empty:
        detail = detail.sort_values(["segment", "validation_mae"])
    detail.to_csv(settings.metrics_dir / "segmented_benchmark_single_worstof.csv", index=False)
    _write_segmented_summary(detail, settings)
    return detail


def _write_segmented_summary(detail: pd.DataFrame, settings: Settings) -> None:
    """Handle write segmented summary."""
    if detail.empty:
        detail.to_csv(settings.metrics_dir / "segmented_benchmark_summary.csv", index=False)
        return
    summary = (
        detail.groupby(
            ["segment", "model_name", "feature_block", "encoding_strategy"], dropna=False
        )
        .agg(
            validation_mae=("validation_mae", "mean"),
            n_total_features=("n_total_features", "first"),
            train_rows=("train_rows", "first"),
            validation_rows=("validation_rows", "first"),
        )
        .reset_index()
        .sort_values(["segment", "validation_mae"])
    )
    summary.to_csv(settings.metrics_dir / "segmented_benchmark_summary.csv", index=False)


def run_robustness_report(
    rfqs: pd.DataFrame,
    volatility: pd.DataFrame,
    reference: pd.DataFrame,
    settings: Settings,
    model_names: list[str] | None = None,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    """Perform run robustness report."""
    settings.ensure_output_dirs()
    rfqs = rfqs.reset_index(drop=True)
    split = temporal_split(rfqs)
    y = pd.to_numeric(rfqs["avg_duration_months"], errors="coerce")
    if y.isna().any():
        raise ValueError("Robustness report requires target values.")

    comparison_path = settings.metrics_dir / "benchmark_comparison.csv"
    predictions_path = settings.metrics_dir / BENCHMARK_PREDICTIONS_FILENAME
    if not comparison_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(
            "Error analysis consumes benchmark outputs and never retrains models. "
            "Run `starwars-autocalls benchmark` first to create the ranking and "
            f"{BENCHMARK_PREDICTIONS_FILENAME}."
        )
    comparison = load_benchmark_table(comparison_path)
    names = model_names or comparison.head(top_n)["model_name"].astype(str).tolist()
    specs = candidate_specs(model_names=names)
    cached_predictions = pd.read_csv(predictions_path)
    missing_models = sorted(set(names) - set(cached_predictions["model_name"].astype(str)))
    if missing_models:
        raise ValueError(
            "No cached validation predictions were found for: "
            f"{missing_models}. Re-run benchmark with those models first."
        )

    summary_rows: list[dict[str, object]] = []
    segment_rows: list[pd.DataFrame] = []
    duration_rows: list[pd.DataFrame] = []
    underlying_rows: list[pd.DataFrame] = []

    validation_rows = rfqs.loc[split.validation_index].copy()
    validation_rows["requested_year"] = pd.to_datetime(validation_rows["requested_date"]).dt.year
    for spec_index, spec in enumerate(specs, start=1):
        report_progress(
            f"Análisis de errores {spec_index}/{len(specs)} desde predicciones cacheadas: "
            f"{spec.name}"
        )
        model_predictions = cached_predictions.loc[
            cached_predictions["model_name"] == spec.name
        ].set_index("row_index")
        expected_index = split.validation_index.astype(int)
        if set(model_predictions.index.astype(int)) != set(expected_index):
            raise ValueError(
                f"Cached validation rows for {spec.name} do not match the current temporal split. "
                "Re-run benchmark before error-analysis."
            )
        model_predictions = model_predictions.loc[expected_index]
        expected_rfq_ids = rfqs.loc[split.validation_index, "rfq_id"].astype(str).tolist()
        if model_predictions["rfq_id"].astype(str).tolist() != expected_rfq_ids:
            raise ValueError(
                f"Cached RFQ ids for {spec.name} do not match the current data. "
                "Re-run benchmark before error-analysis."
            )
        predictions = model_predictions["prediction"].to_numpy()
        metrics = regression_metrics(y.loc[split.validation_index], predictions)
        summary_rows.append(
            {
                "model_name": spec.name,
                "feature_block": spec.feature_block,
                "encoding_strategy": spec.encoding_strategy,
                "estimator_class": spec.estimator.__class__.__name__,
                "split": split.description,
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )

        for column in ["requested_year", "basket_type", "product_type"]:
            frame = segment_mae(
                validation_rows,
                y.loc[split.validation_index],
                predictions,
                column,
            )
            frame["model_name"] = spec.name
            frame["segment_column"] = column
            frame = frame.rename(columns={column: "segment_value"})
            segment_rows.append(frame)

        duration = duration_bucket_metrics(y.loc[split.validation_index], predictions)
        duration["model_name"] = spec.name
        duration_rows.append(duration)

        underlying = _underlying_error_report(
            validation_rows,
            y.loc[split.validation_index],
            predictions,
        )
        underlying["model_name"] = spec.name
        underlying_rows.append(underlying)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["model_name", "feature_block", "encoding_strategy"])

    outputs = {
        "summary": summary,
        "segments": pd.concat(segment_rows, ignore_index=True) if segment_rows else pd.DataFrame(),
        "duration_buckets": pd.concat(duration_rows, ignore_index=True)
        if duration_rows
        else pd.DataFrame(),
        "underlyings": pd.concat(underlying_rows, ignore_index=True)
        if underlying_rows
        else pd.DataFrame(),
    }
    for name, frame in outputs.items():
        frame.to_csv(settings.metrics_dir / f"robustness_{name}.csv", index=False)
    return outputs


def _underlying_error_report(
    test_rows: pd.DataFrame,
    y_true: pd.Series,
    predictions: pd.Series,
) -> pd.DataFrame:
    """Handle underlying error report."""
    data = test_rows[["underlyings"]].copy()
    data["actual"] = y_true.to_numpy()
    data["prediction"] = predictions
    data["absolute_error"] = (data["actual"] - data["prediction"]).abs()
    data["underlying"] = data["underlyings"].map(parse_underlyings)
    exploded = data.explode("underlying").dropna(subset=["underlying"])
    return (
        exploded.groupby("underlying", dropna=False)
        .agg(mae=("absolute_error", "mean"), count=("absolute_error", "size"))
        .reset_index()
        .sort_values(["mae", "count"], ascending=[False, False])
    )


def load_benchmark_table(path: Path) -> pd.DataFrame:
    """Return load benchmark table."""
    return pd.read_csv(path).sort_values("validation_mae")
