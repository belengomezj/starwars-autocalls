"""Training module."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from starwars_autocalls import __version__
from starwars_autocalls.config import RANDOM_SEED, Settings
from starwars_autocalls.data.loading import load_all, trainable_rfqs
from starwars_autocalls.data.validation import validate_all
from starwars_autocalls.features import FeatureBuilder, select_feature_block
from starwars_autocalls.modeling.artifacts import save_model_bundle
from starwars_autocalls.modeling.benchmark import (
    load_benchmark_table,
    run_benchmark,
    run_segmented_benchmark,
)
from starwars_autocalls.modeling.evaluation import (
    duration_bucket_metrics,
    regression_metrics,
    segment_mae,
    temporal_split,
)
from starwars_autocalls.modeling.specs import (
    ModelSpec,
    ablation_specs,
    build_pipeline,
    default_model_specs,
    global_stable_specs,
    model_family_from_name,
)
from starwars_autocalls.modeling.tuning import (
    load_best_tuned_spec,
    tune_segmented_models,
    tuned_spec_from_params,
)
from starwars_autocalls.observability import get_logger
from starwars_autocalls.observability.mlflow import (
    log_mlflow_run,
    metric_dataset_from_features,
    mlflow_dataset_name,
    short_data_hash_tags,
)
from starwars_autocalls.observability.progress import report_progress
from starwars_autocalls.reports.explainability import feature_signal_report
from starwars_autocalls.reproducibility import reproducibility_manifest

logger = get_logger(__name__)


def _request_contract_metadata(
    trainable: pd.DataFrame,
    train_validation_index: pd.Index,
    volatility: pd.DataFrame,
) -> dict[str, Any]:
    """Handle request contract metadata."""
    development = trainable.loc[train_validation_index]
    range_columns = [
        "autocall_barrier_pct",
        "protection_barrier_pct",
        "no_call_period_months",
        "quoted_implied_vol",
        "notional_credits",
    ]
    return {
        "accepted_product_types": sorted(development["product_type"].astype(str).unique()),
        "accepted_observation_frequencies": sorted(
            development["observation_frequency"].astype(str).unique()
        ),
        "training_request_ranges": {
            column: {
                "min": float(development[column].min()),
                "max": float(development[column].max()),
            }
            for column in range_columns
        },
        "market_data_as_of": pd.to_datetime(volatility["date"]).max().date().isoformat(),
        "max_market_data_staleness_days": 10,
    }


def _calibration_and_clipping_metadata(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    trainable: pd.DataFrame,
    train_index: pd.Index,
    validation_index: pd.Index,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Handle calibration and clipping metadata."""
    report_progress(f"Calibrando incertidumbre y límites contractuales: {spec.name}")
    calibration_pipeline = build_pipeline(spec)
    calibration_pipeline.fit(X.loc[train_index], y.loc[train_index])
    predictions = np.asarray(calibration_pipeline.predict(X.loc[validation_index]), dtype=float)
    actual = y.loc[validation_index].to_numpy(dtype=float)
    absolute_residuals = np.abs(actual - predictions)
    nominal_coverage = 0.90
    quantile = float(np.quantile(absolute_residuals, nominal_coverage, method="higher"))
    empirical_coverage = float(np.mean(absolute_residuals <= quantile))

    validation_features = X.loc[validation_index]
    lower = np.maximum(
        trainable.loc[validation_index, "no_call_period_months"].to_numpy(dtype=float),
        validation_features["observation_interval_months"].to_numpy(dtype=float),
    )
    upper = validation_features["nominal_maturity_months"].to_numpy(dtype=float)
    clipped_predictions = np.minimum(np.maximum(predictions, lower), upper)
    raw_metrics = regression_metrics(actual, predictions)
    clipped_metrics = regression_metrics(actual, clipped_predictions)
    clipping_enabled = clipped_metrics["mae"] <= raw_metrics["mae"]
    return (
        {
            "method": "symmetric_absolute_residual_quantile",
            "calibration_split": "validation=2022",
            "nominal_coverage": nominal_coverage,
            "empirical_coverage": empirical_coverage,
            "absolute_residual_quantile": quantile,
            "calibration_rows": len(validation_index),
            "selection_bias_warning": (
                "The same development validation year informed model comparison; "
                "coverage must be monitored on future cohorts."
            ),
        },
        {
            "enabled": clipping_enabled,
            "rule": "clip to [max(no_call_period, observation_interval), nominal_maturity]",
            "validation_raw_metrics": raw_metrics,
            "validation_clipped_metrics": clipped_metrics,
        },
    )


def _best_spec_from_name(model_name: str) -> ModelSpec:
    """Handle best spec from name."""
    for spec in [*default_model_specs(), *ablation_specs(), *global_stable_specs()]:
        if spec.name == model_name:
            return spec
    raise ValueError(f"Could not reconstruct model spec {model_name!r}.")


def _resolve_explicit_spec(
    model_name: str | None,
    model: str | None,
    feature_block: str | None,
) -> ModelSpec | None:
    """Handle resolve explicit spec."""
    if model_name and (model or feature_block):
        raise ValueError("model_name cannot be combined with model or feature_block.")
    if bool(model) != bool(feature_block):
        raise ValueError("model and feature_block must be provided together.")
    if model_name:
        return _best_spec_from_name(model_name)
    if model and feature_block:
        return _best_spec_from_name(f"{model}__{feature_block}")
    return None


def _load_selection_summary(settings: Settings) -> dict[str, Any] | None:
    """Handle load selection summary."""
    path = settings.metrics_dir / "model_selection_protocol_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_final_model_spec(settings: Settings) -> tuple[ModelSpec, dict[str, Any]]:
    """Load the frozen model selected by experimentation."""
    path = settings.final_model_config_path
    if not path.exists():
        raise FileNotFoundError(f"No existe la configuración del modelo final: {path}.")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("serving_strategy") != "global":
        raise ValueError("La configuración final sólo admite la estrategia global.")
    spec = tuned_spec_from_params(
        str(config["base_model_name"]),
        dict(config["hyperparameters"]),
    )
    if spec.name != config.get("model_name"):
        raise ValueError("El nombre del modelo final no coincide con su configuración.")
    return spec, config


def _load_selected_segmented_specs(
    settings: Settings,
) -> dict[str, tuple[ModelSpec, dict[str, Any]]]:
    """Handle load selected segmented specs."""
    summary_path = settings.metrics_dir / "optuna_segmented_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            "Segmented serving was selected, but optuna_segmented_summary.json is missing."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_by_segment: dict[str, dict[str, Any]] = {}
    for study in summary.get("studies", []):
        segment = str(study["segment"])
        current = best_by_segment.get(segment)
        if current is None or float(study["best_validation_mae"]) < float(
            current["best_validation_mae"]
        ):
            best_by_segment[segment] = study
    missing = {"single", "worst_of"} - set(best_by_segment)
    if missing:
        raise ValueError(f"Missing tuned segmented studies for: {sorted(missing)}")
    return {
        segment: (
            tuned_spec_from_params(str(study["base_model_name"]), dict(study["best_params"])),
            study,
        )
        for segment, study in best_by_segment.items()
    }


def _feature_importance_artifact(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Handle feature importance artifact."""
    try:
        result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=5,
            random_state=RANDOM_SEED,
            scoring="neg_mean_absolute_error",
        )
        return (
            pd.DataFrame(
                {
                    "feature": X_test.columns,
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
    except Exception as exc:
        logger.warning(
            "permutation_importance_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])


def train_final_model(
    settings: Settings,
    model_name: str | None = None,
    model: str | None = None,
    feature_block: str | None = None,
    use_tuned_best: bool = False,
    use_strategy_selection: bool = False,
) -> dict[str, Any]:
    """Perform train final model."""
    started_at = time.perf_counter()
    report_progress("Validando datos y preparando el entrenamiento final")
    logger.info("training_started", project_root=str(settings.project_root.resolve()))
    settings.ensure_output_dirs()
    rfqs, volatility, reference = load_all(settings)
    rfqs, volatility, reference = validate_all(rfqs, volatility, reference)
    trainable = trainable_rfqs(rfqs)
    logger.info(
        "training_data_ready",
        rfq_rows=len(rfqs),
        trainable_rows=len(trainable),
        volatility_rows=len(volatility),
        reference_rows=len(reference),
    )
    explicit_spec = _resolve_explicit_spec(model_name, model, feature_block)
    if explicit_spec is not None and (use_tuned_best or use_strategy_selection):
        raise ValueError(
            "La selección explícita de modelo no se puede combinar con "
            "use_tuned_best o use_strategy_selection."
        )
    if use_tuned_best and use_strategy_selection:
        raise ValueError("Usa sólo una fuente de selección adicional para training.")
    final_model_config: dict[str, Any] | None = None
    if explicit_spec is not None:
        comparison = pd.DataFrame()
        benchmark_best_name = explicit_spec.name
        benchmark_best_mae = None
        best_spec = explicit_spec
        selected_validation_mae = None
        selection_source = "explicit"
        tuning_selection: dict[str, Any] | None = None
    elif not use_tuned_best and not use_strategy_selection:
        best_spec, final_model_config = _load_final_model_spec(settings)
        comparison = pd.DataFrame()
        benchmark_best_name = str(final_model_config["model_name"])
        benchmark_best_mae = float(final_model_config["validation_mae"])
        selected_validation_mae = benchmark_best_mae
        selection_source = "final_config"
        tuning_selection = None
    else:
        benchmark_path = settings.metrics_dir / "benchmark_comparison.csv"
        comparison = (
            load_benchmark_table(benchmark_path)
            if benchmark_path.exists()
            else run_benchmark(trainable, volatility, reference, settings, include_ablations=True)
        )
        benchmark_best_name = str(comparison.iloc[0]["model_name"])
        benchmark_best_mae = float(comparison.iloc[0]["validation_mae"])
        best_spec = _best_spec_from_name(benchmark_best_name)
        selected_validation_mae = benchmark_best_mae
        selection_source = "benchmark"
        tuning_selection = None
    selection_summary = _load_selection_summary(settings) if use_strategy_selection else None
    if use_strategy_selection and selection_summary is None:
        raise FileNotFoundError(
            "No existe model_selection_protocol_summary.json. "
            "Ejecuta compare-serving-strategies primero."
        )
    if (
        explicit_spec is None
        and selection_summary
        and selection_summary.get("selected_strategy") == "segmented_by_basket_type"
    ):
        return _train_final_segmented_model(
            settings,
            trainable,
            volatility,
            reference,
            selection_summary,
        )

    if explicit_spec is None and selection_summary:
        selected_model_name = str(selection_summary.get("selected_model_name", ""))
        tuned = load_best_tuned_spec(settings)
        if tuned is not None and tuned[0].name == selected_model_name:
            best_spec = tuned[0]
        else:
            best_spec = _best_spec_from_name(selected_model_name)
        selected_validation_mae = float(
            next(
                candidate["validation_mae"]
                for candidate in selection_summary.get("candidates", [])
                if candidate.get("strategy") == "global"
            )
        )
        selection_source = "strategy_selection"

    if explicit_spec is None and use_tuned_best:
        tuned = load_best_tuned_spec(settings)
        if tuned is None:
            raise FileNotFoundError(
                "No existe un resultado de tuning compatible. Ejecuta tune primero."
            )
        tuned_spec, tuned_info = tuned
        tuned_mae = float(tuned_info["best_validation_mae"])
        if tuned_mae <= benchmark_best_mae:
            best_spec = tuned_spec
            selected_validation_mae = tuned_mae
            selection_source = "optuna"
            tuning_selection = tuned_info

    split = temporal_split(trainable)
    feature_set = FeatureBuilder().build(trainable, volatility, reference, include_target=True)
    X = feature_set.frame
    y = feature_set.target
    if y is None:
        raise ValueError("Training requires target values.")

    train_validation_index = split.train_index.union(split.validation_index)
    conformal_calibration, contractual_clipping = _calibration_and_clipping_metadata(
        best_spec,
        X,
        y,
        trainable,
        split.train_index,
        split.validation_index,
    )
    serving_contract = _request_contract_metadata(
        trainable,
        train_validation_index,
        volatility,
    )
    report_progress(
        f"Entrenando modelo final {best_spec.name} con {len(train_validation_index):,} filas"
    )
    pipeline = build_pipeline(best_spec)
    final_fit_started = time.perf_counter()
    pipeline.fit(X.loc[train_validation_index], y.loc[train_validation_index])
    final_fit_seconds = time.perf_counter() - final_fit_started
    report_progress(f"Modelo final {best_spec.name} entrenado; evaluando test")
    test_predict_started = time.perf_counter()
    test_predictions = pipeline.predict(X.loc[split.test_index])
    test_predict_seconds = time.perf_counter() - test_predict_started
    test_metrics = regression_metrics(y.loc[split.test_index], test_predictions)
    numeric_features, categorical_features = select_feature_block(best_spec.feature_block)
    feature_columns = [*numeric_features, *categorical_features]

    test_rows = trainable.loc[split.test_index].copy()
    segment_artifacts = {
        "segment_mae_by_year": segment_mae(
            test_rows.assign(requested_year=test_rows["requested_date"].dt.year),
            y.loc[split.test_index],
            test_predictions,
            "requested_year",
        ),
        "segment_mae_by_product_type": segment_mae(
            test_rows, y.loc[split.test_index], test_predictions, "product_type"
        ),
        "segment_mae_by_basket_type": segment_mae(
            test_rows, y.loc[split.test_index], test_predictions, "basket_type"
        ),
        "segment_mae_by_duration_bucket": duration_bucket_metrics(
            y.loc[split.test_index], test_predictions
        ),
    }
    importance = _feature_importance_artifact(
        pipeline, X.loc[split.test_index], y.loc[split.test_index]
    )
    signal_analysis = feature_signal_report(
        X.loc[train_validation_index],
        y.loc[train_validation_index],
        numeric_features,
        categorical_features,
    )
    top_signal = signal_analysis.copy()
    top_signal["abs_train_target_spearman"] = top_signal["train_target_spearman"].abs()
    top_feature_signal = json.loads(
        top_signal.sort_values("abs_train_target_spearman", ascending=False)
        .head(15)
        .to_json(orient="records")
    )

    metadata = {
        "package_version": __version__,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "model_name": best_spec.name,
        "model_family": model_family_from_name(best_spec.name),
        "selection_source": selection_source,
        "feature_block": best_spec.feature_block,
        "encoding_strategy": best_spec.encoding_strategy,
        "split": split.description,
        "benchmark_best_model_name": benchmark_best_name,
        "benchmark_best_validation_mae": benchmark_best_mae,
        "selected_validation_mae": selected_validation_mae,
        "tuning_selection": tuning_selection,
        "final_model_config": final_model_config,
        "train_rows": len(split.train_index),
        "validation_rows": len(split.validation_index),
        "test_rows": len(split.test_index),
        "test_opened_at_utc": datetime.now(UTC).isoformat(),
        "test_open_reason": "final_model_evaluation_after_selection_was_frozen",
        "test_metrics": test_metrics,
        "final_fit_seconds": final_fit_seconds,
        "test_predict_seconds": test_predict_seconds,
        "test_rows_per_second": len(split.test_index) / max(test_predict_seconds, 1e-12),
        "conformal_calibration": conformal_calibration,
        "contractual_clipping": contractual_clipping,
        "validation_leaderboard_top5": comparison.head(5).to_dict(orient="records"),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_manifest": {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "numeric_feature_count": len(numeric_features),
            "categorical_feature_count": len(categorical_features),
            "selected_feature_block": best_spec.feature_block,
        },
        "mlflow_run_id": None,
        **reproducibility_manifest(settings),
        "top_feature_signal": top_feature_signal,
        "modeling_assumptions": [
            "Only executed RFQs with a known avg_duration_months target are used for training.",
            "Realized volatility is joined as of requested_date, never after requested_date.",
            "Nominal maturity uses start_date and end_date because those dates are known at RFQ time.",
            "1D observation frequency is approximated as one trading day in a 21-day month.",
            "The model is a tabular surrogate for the supplied simulated target, not a pricing engine.",
            "Predictions are conditional on product execution because avg_duration_months is available only for executed RFQs.",
        ],
        **serving_contract,
    }

    if not comparison.empty:
        comparison.to_csv(settings.metrics_dir / "benchmark_comparison.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(
        settings.metrics_dir / "final_test_metrics.csv", index=False
    )
    importance.to_csv(settings.explainability_dir / "permutation_importance.csv", index=False)
    signal_analysis.to_csv(settings.explainability_dir / "feature_signal_analysis.csv", index=False)
    for name, frame in segment_artifacts.items():
        frame.to_csv(settings.metrics_dir / f"{name}.csv", index=False)

    mlflow_run_id = log_mlflow_run(
        run_name=best_spec.name,
        params={
            "model_name": best_spec.name,
            "model_family": model_family_from_name(best_spec.name),
            "run_type": "final_model",
            "selection_source": selection_source,
            "estimator_class": best_spec.estimator.__class__.__name__,
            "feature_block": best_spec.feature_block,
            "encoding_strategy": best_spec.encoding_strategy,
            "split": split.description,
        },
        metrics={f"test_{key}": value for key, value in test_metrics.items()},
        artifacts={
            "model_metadata": metadata,
            "permutation_importance": importance,
            "feature_signal_analysis": signal_analysis,
            **segment_artifacts,
        },
        tags={
            "run_type": "final_model",
            "model_name": best_spec.name,
            "model_family": model_family_from_name(best_spec.name),
            "selection_source": selection_source,
            "feature_block": best_spec.feature_block,
            "encoding_strategy": best_spec.encoding_strategy,
            "estimator_class": best_spec.estimator.__class__.__name__,
            **short_data_hash_tags(metadata["data_hashes"]),
        },
        model=pipeline,
        input_example=X.loc[train_validation_index, feature_columns].head(5),
        metric_dataset=metric_dataset_from_features(
            X,
            y,
            split.test_index,
            feature_columns,
        ),
        dataset_name=mlflow_dataset_name(best_spec.feature_block, "test"),
        registered_model_name=model_family_from_name(best_spec.name),
    )
    metadata["mlflow_run_id"] = mlflow_run_id
    artifact = {
        "pipeline": pipeline,
        "metadata": metadata,
        "reference": reference,
        "volatility": volatility,
    }
    save_model_bundle(artifact, metadata, settings)
    operational_metrics = {
        "model_name": best_spec.name,
        "artifact_bytes": settings.model_path.stat().st_size,
        "final_fit_seconds": final_fit_seconds,
        "test_predict_seconds": test_predict_seconds,
        "test_rows": len(split.test_index),
        "test_rows_per_second": len(split.test_index) / max(test_predict_seconds, 1e-12),
        "total_training_command_seconds": time.perf_counter() - started_at,
    }
    (settings.metrics_dir / "operational_metrics.json").write_text(
        json.dumps(operational_metrics, indent=2), encoding="utf-8"
    )
    logger.info(
        "training_completed",
        model_name=best_spec.name,
        serving_strategy="global",
        test_mae=test_metrics["mae"],
        elapsed_seconds=round(time.perf_counter() - started_at, 3),
    )
    return metadata


def _train_final_segmented_model(
    settings: Settings,
    trainable: pd.DataFrame,
    volatility: pd.DataFrame,
    reference: pd.DataFrame,
    selection_summary: dict[str, Any],
) -> dict[str, Any]:
    """Handle train final segmented model."""
    started_at = time.perf_counter()
    logger.info("segmented_training_started", trainable_rows=len(trainable))
    split = temporal_split(trainable)
    feature_set = FeatureBuilder().build(trainable, volatility, reference, include_target=True)
    X = feature_set.frame
    y = feature_set.target
    if y is None:
        raise ValueError("Training requires target values.")

    selected_specs = _load_selected_segmented_specs(settings)
    train_validation_index = split.train_index.union(split.validation_index)
    segment_masks = {
        "single": X["is_single_underlying"].eq(1),
        "worst_of": X["is_worst_of"].eq(1),
    }
    pipelines: dict[str, Any] = {}
    segment_rows: list[dict[str, Any]] = []
    segment_feature_manifests: dict[str, Any] = {}
    test_predictions = pd.Series(index=split.test_index, dtype=float)
    validation_predictions = pd.Series(index=split.validation_index, dtype=float)
    importance_frames: list[pd.DataFrame] = []

    for segment, (spec, study) in sorted(selected_specs.items()):
        segment_index = X.index[segment_masks[segment]]
        segment_train_validation_index = train_validation_index.intersection(segment_index)
        segment_train_index = split.train_index.intersection(segment_index)
        segment_validation_index = split.validation_index.intersection(segment_index)
        segment_test_index = split.test_index.intersection(segment_index)
        report_progress(f"Calibrando segmento {segment}: {spec.name}")
        calibration_pipeline = build_pipeline(spec)
        calibration_pipeline.fit(X.loc[segment_train_index], y.loc[segment_train_index])
        validation_predictions.loc[segment_validation_index] = calibration_pipeline.predict(
            X.loc[segment_validation_index]
        )
        report_progress(
            f"Entrenando segmento final {segment}: {spec.name} "
            f"({len(segment_train_validation_index):,} filas)"
        )
        pipeline = build_pipeline(spec)
        pipeline.fit(X.loc[segment_train_validation_index], y.loc[segment_train_validation_index])
        report_progress(f"Segmento {segment} entrenado; evaluando test")
        predictions = pipeline.predict(X.loc[segment_test_index])
        test_predictions.loc[segment_test_index] = predictions
        pipelines[segment] = pipeline
        test_metrics = regression_metrics(y.loc[segment_test_index], predictions)
        numeric_features, categorical_features = select_feature_block(spec.feature_block)
        segment_feature_manifests[segment] = {
            "model_name": spec.name,
            "feature_block": spec.feature_block,
            "encoding_strategy": spec.encoding_strategy,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "numeric_feature_count": len(numeric_features),
            "categorical_feature_count": len(categorical_features),
            "total_feature_count": len(numeric_features) + len(categorical_features),
        }
        segment_rows.append(
            {
                "segment": segment,
                "model_name": spec.name,
                "base_model_name": study["base_model_name"],
                "feature_block": spec.feature_block,
                "encoding_strategy": spec.encoding_strategy,
                "selection_validation_mae": float(study["best_validation_mae"]),
                "test_mae": float(test_metrics["mae"]),
                "test_rmse": float(test_metrics["rmse"]),
                "test_r2": float(test_metrics["r2"]),
                "test_median_absolute_error": float(test_metrics["median_absolute_error"]),
                "train_validation_rows": len(segment_train_validation_index),
                "test_rows": len(segment_test_index),
                "n_numeric_features": len(numeric_features),
                "n_categorical_features": len(categorical_features),
                "n_total_features": len(numeric_features) + len(categorical_features),
            }
        )
        segment_importance = _feature_importance_artifact(
            pipeline,
            X.loc[segment_test_index],
            y.loc[segment_test_index],
        )
        if not segment_importance.empty:
            segment_importance.insert(0, "segment", segment)
            importance_frames.append(segment_importance)

    if test_predictions.isna().any():
        missing = test_predictions[test_predictions.isna()].index.tolist()[:10]
        raise ValueError(f"Segmented artifact did not score all test rows: {missing}")
    if validation_predictions.isna().any():
        missing = validation_predictions[validation_predictions.isna()].index.tolist()[:10]
        raise ValueError(f"Segmented calibration did not score all validation rows: {missing}")

    validation_actual = y.loc[split.validation_index].to_numpy(dtype=float)
    validation_pred = validation_predictions.loc[split.validation_index].to_numpy(dtype=float)
    absolute_residuals = np.abs(validation_actual - validation_pred)
    residual_quantile = float(np.quantile(absolute_residuals, 0.90, method="higher"))
    lower = np.maximum(
        trainable.loc[split.validation_index, "no_call_period_months"].to_numpy(dtype=float),
        X.loc[split.validation_index, "observation_interval_months"].to_numpy(dtype=float),
    )
    upper = X.loc[split.validation_index, "nominal_maturity_months"].to_numpy(dtype=float)
    clipped_validation = np.minimum(np.maximum(validation_pred, lower), upper)
    raw_validation_metrics = regression_metrics(validation_actual, validation_pred)
    clipped_validation_metrics = regression_metrics(validation_actual, clipped_validation)
    conformal_calibration = {
        "method": "symmetric_absolute_residual_quantile",
        "calibration_split": "validation=2022",
        "nominal_coverage": 0.90,
        "empirical_coverage": float(np.mean(absolute_residuals <= residual_quantile)),
        "absolute_residual_quantile": residual_quantile,
        "calibration_rows": len(split.validation_index),
    }
    contractual_clipping = {
        "enabled": clipped_validation_metrics["mae"] <= raw_validation_metrics["mae"],
        "rule": "clip to [max(no_call_period, observation_interval), nominal_maturity]",
        "validation_raw_metrics": raw_validation_metrics,
        "validation_clipped_metrics": clipped_validation_metrics,
    }

    test_metrics = regression_metrics(
        y.loc[split.test_index], test_predictions.loc[split.test_index]
    )
    test_rows = trainable.loc[split.test_index].copy()
    segment_artifacts = {
        "segment_mae_by_year": segment_mae(
            test_rows.assign(requested_year=test_rows["requested_date"].dt.year),
            y.loc[split.test_index],
            test_predictions.loc[split.test_index],
            "requested_year",
        ),
        "segment_mae_by_product_type": segment_mae(
            test_rows,
            y.loc[split.test_index],
            test_predictions.loc[split.test_index],
            "product_type",
        ),
        "segment_mae_by_basket_type": segment_mae(
            test_rows,
            y.loc[split.test_index],
            test_predictions.loc[split.test_index],
            "basket_type",
        ),
        "segment_mae_by_duration_bucket": duration_bucket_metrics(
            y.loc[split.test_index], test_predictions.loc[split.test_index]
        ),
    }
    importance = (
        pd.concat(importance_frames, ignore_index=True)
        if importance_frames
        else pd.DataFrame(columns=["segment", "feature", "importance_mean", "importance_std"])
    )
    selected_numeric_features = sorted(
        {
            feature
            for manifest in segment_feature_manifests.values()
            for feature in manifest["numeric_features"]
        }
    )
    selected_categorical_features = sorted(
        {
            feature
            for manifest in segment_feature_manifests.values()
            for feature in manifest["categorical_features"]
        }
    )
    signal_analysis = feature_signal_report(
        X.loc[train_validation_index],
        y.loc[train_validation_index],
        selected_numeric_features,
        selected_categorical_features,
    )
    top_signal = signal_analysis.copy()
    top_signal["abs_train_target_spearman"] = top_signal["train_target_spearman"].abs()
    top_feature_signal = json.loads(
        top_signal.sort_values("abs_train_target_spearman", ascending=False)
        .head(15)
        .to_json(orient="records")
    )

    candidates = selection_summary.get("candidates", [])
    selected_candidate = next(
        (
            candidate
            for candidate in candidates
            if candidate.get("strategy") == "segmented_by_basket_type"
        ),
        {},
    )
    metadata = {
        "package_version": __version__,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "serving_strategy": "segmented_by_basket_type",
        "model_name": selected_candidate.get(
            "model_name",
            "single + worst_of segmented models",
        ),
        "model_family": "segmented_router",
        "selection_source": "model_selection_protocol",
        "selection_rule": selection_summary.get("selection_rule"),
        "selection_metric": "validation_mae",
        "test_usage": "audit_only_after_model_selection",
        "selected_validation_mae": selected_candidate.get("validation_mae"),
        "selected_rolling_mae_mean": selected_candidate.get("rolling_mae_mean"),
        "selected_rolling_mae_max": selected_candidate.get("rolling_mae_max"),
        "split": split.description,
        "train_rows": len(split.train_index),
        "validation_rows": len(split.validation_index),
        "test_rows": len(split.test_index),
        "test_opened_at_utc": datetime.now(UTC).isoformat(),
        "test_open_reason": "final_model_evaluation_after_selection_was_frozen",
        "test_metrics": test_metrics,
        "conformal_calibration": conformal_calibration,
        "contractual_clipping": contractual_clipping,
        "segments": segment_rows,
        "selection_protocol_summary_path": str(
            settings.metrics_dir / "model_selection_protocol_summary.json"
        ),
        "numeric_features": selected_numeric_features,
        "categorical_features": selected_categorical_features,
        "feature_manifest": {
            "numeric_features": selected_numeric_features,
            "categorical_features": selected_categorical_features,
            "numeric_feature_count": len(selected_numeric_features),
            "categorical_feature_count": len(selected_categorical_features),
            "selected_feature_block": selected_candidate.get("feature_block"),
            "segments": segment_feature_manifests,
        },
        "mlflow_run_id": None,
        **reproducibility_manifest(settings),
        "top_feature_signal": top_feature_signal,
        "modeling_assumptions": [
            "Only executed RFQs with a known avg_duration_months target are used for training.",
            "Realized volatility is joined as of requested_date, never after requested_date.",
            "Nominal maturity uses start_date and end_date because those dates are known at RFQ time.",
            "1D observation frequency is approximated as one trading day in a 21-day month.",
            "The model is a tabular surrogate for the supplied simulated target, not a pricing engine.",
            "Predictions are conditional on product execution because avg_duration_months is available only for executed RFQs.",
            "The production artifact routes by basket_type into separate single and worst_of models.",
        ],
        **_request_contract_metadata(trainable, train_validation_index, volatility),
    }

    pd.DataFrame([test_metrics]).to_csv(
        settings.metrics_dir / "final_test_metrics.csv", index=False
    )
    pd.DataFrame(segment_rows).to_csv(
        settings.metrics_dir / "final_segmented_model_metrics.csv", index=False
    )
    importance.to_csv(settings.explainability_dir / "permutation_importance.csv", index=False)
    signal_analysis.to_csv(settings.explainability_dir / "feature_signal_analysis.csv", index=False)
    for name, frame in segment_artifacts.items():
        frame.to_csv(settings.metrics_dir / f"{name}.csv", index=False)

    serving_feature_columns = sorted(
        {
            feature
            for manifest in segment_feature_manifests.values()
            for feature in [*manifest["numeric_features"], *manifest["categorical_features"]]
        }
    )
    mlflow_run_id = log_mlflow_run(
        run_name="final_segmented_by_basket_type",
        params={
            "model_name": metadata["model_name"],
            "model_family": metadata["model_family"],
            "run_type": "final_model",
            "selection_source": "model_selection_protocol",
            "serving_strategy": "segmented_by_basket_type",
            "split": split.description,
        },
        metrics={
            **{f"test_{key}": float(value) for key, value in test_metrics.items()},
            "selection_validation_mae": float(metadata["selected_validation_mae"]),
            "selection_rolling_mae_mean": float(metadata["selected_rolling_mae_mean"]),
            "selection_rolling_mae_max": float(metadata["selected_rolling_mae_max"]),
        },
        artifacts={
            "model_metadata": metadata,
            "final_segmented_model_metrics": pd.DataFrame(segment_rows),
            "permutation_importance": importance,
            "feature_signal_analysis": signal_analysis,
            **segment_artifacts,
        },
        tags={
            "run_type": "final_model",
            "model_name": metadata["model_name"],
            "model_family": metadata["model_family"],
            "serving_strategy": "segmented_by_basket_type",
            "selection_source": "model_selection_protocol",
            **short_data_hash_tags(metadata["data_hashes"]),
        },
        metric_dataset=metric_dataset_from_features(
            X,
            y,
            split.test_index,
            serving_feature_columns,
        ),
        dataset_name=mlflow_dataset_name("segmented_by_basket_type", "test"),
    )
    metadata["mlflow_run_id"] = mlflow_run_id
    artifact = {
        "strategy": "segmented_by_basket_type",
        "pipelines": pipelines,
        "metadata": metadata,
        "reference": reference,
        "volatility": volatility,
    }
    save_model_bundle(artifact, metadata, settings)
    logger.info(
        "training_completed",
        model_name=metadata["model_name"],
        serving_strategy="segmented_by_basket_type",
        test_mae=test_metrics["mae"],
        elapsed_seconds=round(time.perf_counter() - started_at, 3),
    )
    return metadata


def _top_segmented_candidate_names(
    detail: pd.DataFrame,
    top_n_per_segment: int,
    ranking_metric: str,
) -> list[str]:
    """Handle top segmented candidate names."""
    allowed_ranking_metrics = {
        "validation_mae",
        "validation_rmse",
        "validation_median_absolute_error",
    }
    if ranking_metric not in allowed_ranking_metrics:
        raise ValueError(
            "Segmented tuning candidates can only be ranked by validation metrics "
            f"where lower is better: {sorted(allowed_ranking_metrics)}"
        )
    if ranking_metric not in detail.columns:
        raise ValueError(f"Unknown segmented ranking metric: {ranking_metric}")
    names: list[str] = []
    ranked = detail.sort_values(["segment", ranking_metric, "validation_mae"])
    for _, segment_rows in ranked.groupby("segment", sort=True):
        for name in segment_rows.head(top_n_per_segment)["model_name"].astype(str):
            if name not in names:
                names.append(name)
    return names


def _best_rows_by_segment(detail: pd.DataFrame, metric: str) -> list[dict[str, Any]]:
    """Handle best rows by segment."""
    if detail.empty or metric not in detail.columns:
        return []
    ranked = detail.sort_values(["segment", metric, "validation_mae"])
    return [row.to_dict() for _, row in ranked.groupby("segment", sort=True).head(1).iterrows()]


def train_segmented_with_optuna(
    settings: Settings,
    n_trials: int = 20,
    top_n_per_segment: int = 2,
    ranking_metric: str = "validation_mae",
    model_names: list[str] | None = None,
) -> dict[str, Any]:
    """Run segmented benchmark, tune top candidates, and save a workflow summary."""
    settings.ensure_output_dirs()
    rfqs, volatility, reference = load_all(settings)
    rfqs, volatility, reference = validate_all(rfqs, volatility, reference)
    trainable = trainable_rfqs(rfqs)

    benchmark_path = settings.metrics_dir / "segmented_benchmark_single_worstof.csv"
    benchmark_detail = (
        pd.read_csv(benchmark_path)
        if benchmark_path.exists()
        else run_segmented_benchmark(trainable, volatility, reference, settings)
    )
    if benchmark_detail.empty:
        raise ValueError("Segmented benchmark did not produce any candidate rows.")

    candidate_names = model_names or _top_segmented_candidate_names(
        benchmark_detail,
        top_n_per_segment=top_n_per_segment,
        ranking_metric=ranking_metric,
    )
    tuning_summary = tune_segmented_models(
        settings,
        n_trials=n_trials,
        model_names=candidate_names,
    )
    tuning_path = settings.metrics_dir / "segmented_tuning_comparison.csv"
    tuning_detail = pd.read_csv(tuning_path) if tuning_path.exists() else pd.DataFrame()

    summary = {
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "workflow": "segmented_benchmark_plus_optuna",
        "candidate_ranking_metric": ranking_metric,
        "top_n_per_segment": top_n_per_segment,
        "n_trials_per_model": n_trials,
        "candidate_names": candidate_names,
        "explicit_model_names": model_names is not None,
        "benchmark_best_by_validation": _best_rows_by_segment(
            benchmark_detail,
            "validation_mae",
        ),
        "tuned_results": tuning_detail.to_dict(orient="records"),
        "benchmark_detail_path": str(
            settings.metrics_dir / "segmented_benchmark_single_worstof.csv"
        ),
        "tuning_comparison_path": str(tuning_path),
        "tuning_summary_path": tuning_summary["summary_path"],
    }
    output_path = settings.metrics_dir / "segmented_optuna_training_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["summary_path"] = str(output_path)
    return summary
