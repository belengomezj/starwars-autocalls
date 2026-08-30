"""Tuning module."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from starwars_autocalls.config import RANDOM_SEED, Settings
from starwars_autocalls.data.loading import load_all, trainable_rfqs
from starwars_autocalls.data.validation import validate_all
from starwars_autocalls.features import FeatureBuilder, select_feature_block
from starwars_autocalls.modeling.benchmark import load_benchmark_table, run_global_stable_experiment
from starwars_autocalls.modeling.evaluation import regression_metrics, temporal_split
from starwars_autocalls.modeling.specs import (
    ModelSpec,
    build_pipeline,
    global_stable_specs,
    model_family_from_name,
    segmented_model_specs,
    spec_name,
    split_spec_name,
)
from starwars_autocalls.observability.mlflow import (
    log_mlflow_run,
    metric_dataset_from_features,
    mlflow_dataset_name,
    mlflow_parent_run,
    short_data_hash_tags,
)
from starwars_autocalls.observability.progress import report_progress
from starwars_autocalls.reproducibility import data_hashes, reproducibility_manifest

DEFAULT_TUNING_CANDIDATES = [
    spec_name("catboost_native", "all_without_commercial"),
    spec_name("lightgbm_native", "all_without_commercial"),
    spec_name("hist_gradient_boosting", "all_without_commercial"),
    spec_name("xgboost_ordinal", "all_without_commercial"),
]
TUNABLE_FEATURE_BLOCKS = {
    "all_without_commercial",
    "all_without_noise",
    "compact_core",
    "single_core",
    "single_without_noise",
    "single_stable",
    "single_underlying",
    "single_underlying_no_sector",
    "worst_of_core",
    "worst_of_without_noise",
    "worst_of_stable",
    "worst_of_tail_focus",
    "worst_of_risk_underlying",
    "worst_of_tail_underlying",
    "global_stable",
    "global_stable_tail",
    "global_stable_no_sector",
    "global_risk_underlying",
    "global_all_underlying",
    "global_tail_underlying",
}


def _candidate_model_names(
    settings: Settings,
    top_n: int,
    explicit_model_names: list[str] | None = None,
) -> list[str]:
    """Handle candidate model names."""
    if explicit_model_names:
        names = explicit_model_names
    else:
        benchmark_path = settings.metrics_dir / "benchmark_comparison.csv"
        if benchmark_path.exists():
            names = load_benchmark_table(benchmark_path)["model_name"].astype(str).tolist()
        else:
            names = DEFAULT_TUNING_CANDIDATES

    selected: list[str] = []
    for name in names:
        if name not in selected and _supports_tuning(name):
            selected.append(name)
        if len(selected) == top_n:
            break
    return selected


def _supports_tuning(model_name: str) -> bool:
    """Handle supports tuning."""
    return _tuning_family_and_block(model_name) is not None


def _tuning_family_and_block(model_name: str) -> tuple[str, str] | None:
    """Handle tuning family and block."""
    parsed = split_spec_name(model_name)
    if parsed is not None:
        estimator_key, feature_block = parsed
        family_by_key = {
            "catboost_native": "catboost_native",
            "lightgbm_native": "lightgbm_native",
            "hist_gradient_boosting": "hist_gradient_boosting",
            "hist_gradient_boosting_ablation": "hist_gradient_boosting",
            "xgboost_ordinal": "xgboost_ordinal",
        }
        family = family_by_key.get(estimator_key)
        if family and feature_block in TUNABLE_FEATURE_BLOCKS:
            return family, feature_block

    for spec in global_stable_specs():
        if model_name != spec.name:
            continue
        parsed_spec = split_spec_name(spec.name)
        estimator_key = parsed_spec[0] if parsed_spec else spec.name
        if estimator_key.startswith("hist_gradient_boosting"):
            return "hist_gradient_boosting", spec.feature_block
        if estimator_key.startswith("catboost_native"):
            return "catboost_native", spec.feature_block
        if estimator_key.startswith("lightgbm_native"):
            return "lightgbm_native", spec.feature_block
        if estimator_key.startswith("xgboost_ordinal"):
            return "xgboost_ordinal", spec.feature_block

    for specs in segmented_model_specs().values():
        for spec in specs:
            if model_name != spec.name:
                continue
            parsed_spec = split_spec_name(spec.name)
            estimator_key = parsed_spec[0] if parsed_spec else spec.name
            if estimator_key.startswith("hist_gradient_boosting"):
                return "hist_gradient_boosting", spec.feature_block
            if estimator_key.startswith("catboost_native"):
                return "catboost_native", spec.feature_block
            if estimator_key.startswith("lightgbm_native"):
                return "lightgbm_native", spec.feature_block
            if estimator_key.startswith("xgboost_ordinal"):
                return "xgboost_ordinal", spec.feature_block

    exact = {
        "catboost_native": ("catboost_native", "all_without_commercial"),
        "lightgbm_native": ("lightgbm_native", "all_without_commercial"),
        "hist_gradient_boosting": (
            "hist_gradient_boosting",
            "all_without_commercial",
        ),
        "hist_gradient_boosting_ordinal": (
            "hist_gradient_boosting",
            "all_without_commercial",
        ),
        "xgboost_ordinal": ("xgboost_ordinal", "all_without_commercial"),
        "hist_gradient_boosting_without_noise": (
            "hist_gradient_boosting",
            "all_without_noise",
        ),
        "hist_gradient_boosting_compact_core": (
            "hist_gradient_boosting",
            "compact_core",
        ),
    }
    if model_name in exact:
        return exact[model_name]
    for family in ["catboost_native", "lightgbm_native", "xgboost_ordinal"]:
        prefix = f"{family}_"
        if model_name.startswith(prefix):
            block = model_name.removeprefix(prefix)
            if block in TUNABLE_FEATURE_BLOCKS:
                return family, block
    return None


def _suggest_spec(model_name: str, trial: optuna.Trial) -> ModelSpec:
    """Handle suggest spec."""
    family_and_block = _tuning_family_and_block(model_name)
    if family_and_block is None:
        raise ValueError(f"Unsupported tuning candidate: {model_name}")
    family, feature_block = family_and_block
    if family == "hist_gradient_boosting":
        return ModelSpec(
            spec_name("hist_gradient_boosting_tuned", feature_block),
            HistGradientBoostingRegressor(
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                max_iter=trial.suggest_int("max_iter", 150, 800),
                max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 120),
                l2_regularization=trial.suggest_float("l2_regularization", 1e-5, 10.0, log=True),
                max_bins=trial.suggest_int("max_bins", 64, 255),
                random_state=RANDOM_SEED,
            ),
            "ordinal",
            feature_block,
        )
    if family == "catboost_native":
        from catboost import CatBoostRegressor

        return ModelSpec(
            spec_name("catboost_tuned", feature_block),
            CatBoostRegressor(
                loss_function="MAE",
                iterations=trial.suggest_int("iterations", 250, 900),
                depth=trial.suggest_int("depth", 4, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.1, 30.0, log=True),
                random_strength=trial.suggest_float("random_strength", 0.0, 3.0),
                bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
                random_seed=RANDOM_SEED,
                verbose=False,
                allow_writing_files=False,
            ),
            "native",
            feature_block,
            "object",
        )
    if family == "lightgbm_native":
        from lightgbm import LGBMRegressor

        return ModelSpec(
            spec_name("lightgbm_tuned", feature_block),
            LGBMRegressor(
                objective="regression_l1",
                n_estimators=trial.suggest_int("n_estimators", 250, 900),
                learning_rate=trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
                num_leaves=trial.suggest_int("num_leaves", 15, 127),
                min_child_samples=trial.suggest_int("min_child_samples", 10, 120),
                subsample=trial.suggest_float("subsample", 0.65, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.65, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbosity=-1,
            ),
            "native",
            feature_block,
            "category",
        )
    if family == "xgboost_ordinal":
        from xgboost import XGBRegressor

        return ModelSpec(
            spec_name("xgboost_tuned", feature_block),
            XGBRegressor(
                objective="reg:absoluteerror",
                tree_method="hist",
                n_estimators=trial.suggest_int("n_estimators", 250, 900),
                learning_rate=trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0),
                subsample=trial.suggest_float("subsample", 0.65, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.65, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            "ordinal",
            feature_block,
        )
    raise ValueError(f"Unsupported tuning candidate: {model_name}")


def _evaluate_spec(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    train_index: pd.Index,
    validation_index: pd.Index,
) -> tuple[float, dict[str, float]]:
    """Handle evaluate spec."""
    report_progress(
        f"Optuna entrena {spec.name} ({len(train_index):,} train / "
        f"{len(validation_index):,} validation)"
    )
    pipeline = build_pipeline(spec)
    pipeline.fit(X.loc[train_index], y.loc[train_index])
    preds = pipeline.predict(X.loc[validation_index])
    metrics = regression_metrics(y.loc[validation_index], preds)
    report_progress(f"Trial completado {spec.name}: MAE={metrics['mae']:.4f}")
    return metrics["mae"], metrics


def _fit_and_log_best_model(
    model_name: str,
    best_spec: ModelSpec,
    best_params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    split_description: str,
    train_index: pd.Index,
    validation_index: pd.Index,
    hash_tags: dict[str, str] | None = None,
    nested: bool = False,
    run_name: str | None = None,
) -> dict[str, float]:
    """Handle fit and log best model."""
    report_progress(f"Reentrenando mejor configuración de {model_name}: {best_spec.name}")
    pipeline = build_pipeline(best_spec)
    pipeline.fit(X.loc[train_index], y.loc[train_index])
    predictions = pipeline.predict(X.loc[validation_index])
    metrics = regression_metrics(y.loc[validation_index], predictions)
    numeric_features, categorical_features = select_feature_block(best_spec.feature_block)
    feature_columns = [*numeric_features, *categorical_features]
    validation_dataset = metric_dataset_from_features(
        X,
        y,
        validation_index,
        feature_columns,
    )
    run_type = str(best_params.get("run_type", "tuning_best_model"))
    segment = best_params.get("segment")
    logged_params = {
        key: value for key, value in best_params.items() if key not in {"run_type", "segment"}
    }
    if segment is not None:
        logged_params["segment"] = segment

    log_mlflow_run(
        run_name=run_name or "best",
        params={
            **logged_params,
            "model_name": best_spec.name,
            "base_model_name": model_name,
            "run_type": run_type,
            "feature_block": best_spec.feature_block,
            "encoding_strategy": best_spec.encoding_strategy,
            "model_family": model_family_from_name(best_spec.name),
            "split": split_description,
            "dataset_name": mlflow_dataset_name(best_spec.feature_block, "validation"),
        },
        metrics={f"validation_{key}": float(value) for key, value in metrics.items()},
        tags={
            "run_type": run_type,
            "model_name": best_spec.name,
            "base_model_name": model_name,
            "model_family": model_family_from_name(best_spec.name),
            "feature_block": best_spec.feature_block,
            "encoding_strategy": best_spec.encoding_strategy,
            "dataset_name": mlflow_dataset_name(best_spec.feature_block, "validation"),
            **({"segment": str(segment)} if segment is not None else {}),
            **(hash_tags or {}),
        },
        model=pipeline,
        input_example=X.loc[train_index, feature_columns].head(5),
        metric_dataset=validation_dataset,
        dataset_name=mlflow_dataset_name(best_spec.feature_block, "validation"),
        registered_model_name=model_family_from_name(best_spec.name),
        nested=nested,
    )
    return metrics


def tune_top_models(
    settings: Settings,
    n_trials: int = 25,
    top_n: int = 4,
    model_names: list[str] | None = None,
    timeout_seconds_per_model: int | None = 300,
) -> dict[str, object]:
    """Perform tune top models."""
    settings.ensure_output_dirs()
    rfqs, volatility, reference = load_all(settings)
    rfqs, volatility, reference = validate_all(rfqs, volatility, reference)
    trainable = trainable_rfqs(rfqs)
    split = temporal_split(trainable)
    feature_set = FeatureBuilder().build(trainable, volatility, reference, include_target=True)
    X = feature_set.frame
    y = feature_set.target
    if y is None:
        raise ValueError("Tuning requires target values.")

    candidates = _candidate_model_names(settings, top_n=top_n, explicit_model_names=model_names)
    if not candidates:
        raise ValueError("No supported tuning candidates were found.")

    hash_tags = short_data_hash_tags(data_hashes(settings))
    rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {
        "split": split.description,
        "n_trials_per_model": n_trials,
        "candidates": candidates,
        **reproducibility_manifest(settings),
        "studies": [],
    }

    for model_name in candidates:
        study = optuna.create_study(
            direction="minimize",
            study_name=f"{model_name}_validation_mae",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
        family_and_block = _tuning_family_and_block(model_name)
        parent_feature_block = family_and_block[1] if family_and_block else ""
        numeric_features, categorical_features = select_feature_block(parent_feature_block)
        validation_dataset = metric_dataset_from_features(
            X,
            y,
            split.validation_index,
            [*numeric_features, *categorical_features],
        )

        def objective(
            trial: optuna.Trial,
            candidate_name: str = model_name,
            objective_validation_dataset: pd.DataFrame = validation_dataset,
        ) -> float:
            """Handle objective."""
            spec = _suggest_spec(candidate_name, trial)
            mae, metrics = _evaluate_spec(
                spec,
                X,
                y,
                split.train_index,
                split.validation_index,
            )
            log_mlflow_run(
                run_name=f"global__{candidate_name}__optuna_trial_{trial.number:03d}",
                params={
                    **trial.params,
                    "trial_number": trial.number,
                    "model_name": spec.name,
                    "base_model_name": candidate_name,
                    "model_family": model_family_from_name(spec.name),
                    "run_type": "optuna_trial",
                    "feature_block": spec.feature_block,
                    "encoding_strategy": spec.encoding_strategy,
                    "scope": "global",
                    "segment": "all",
                    "dataset_name": mlflow_dataset_name(spec.feature_block, "validation"),
                    "split": split.description,
                },
                metrics={f"validation_{key}": float(value) for key, value in metrics.items()},
                metric_dataset=objective_validation_dataset,
                dataset_name=mlflow_dataset_name(spec.feature_block, "validation"),
                tags={
                    "run_type": "optuna_trial",
                    "model_name": spec.name,
                    "base_model_name": candidate_name,
                    "model_family": model_family_from_name(spec.name),
                    "feature_block": spec.feature_block,
                    "encoding_strategy": spec.encoding_strategy,
                    "scope": "global",
                    "segment": "all",
                    "dataset_name": mlflow_dataset_name(spec.feature_block, "validation"),
                    "trial_number": str(trial.number),
                    **hash_tags,
                },
                nested=True,
            )
            return mae

        with mlflow_parent_run(
            run_name=f"global__{model_name}__optuna_study",
            params={
                "model_name": model_name,
                "run_type": "optuna_study",
                "feature_block": parent_feature_block,
                "scope": "global",
                "segment": "all",
                "split": split.description,
                "n_trials": n_trials,
            },
            tags={
                "run_type": "optuna_study",
                "model_name": model_name,
                "feature_block": parent_feature_block,
                **hash_tags,
            },
        ):
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout_seconds_per_model,
                gc_after_trial=True,
            )
        best_spec = _suggest_spec(
            model_name,
            optuna.trial.FixedTrial(study.best_params),
        )
        best_metrics = _fit_and_log_best_model(
            model_name=model_name,
            best_spec=best_spec,
            best_params=study.best_params,
            X=X,
            y=y,
            split_description=split.description,
            train_index=split.train_index,
            validation_index=split.validation_index,
            hash_tags=hash_tags,
            run_name=f"{model_name}__best",
        )
        row = {
            "base_model_name": model_name,
            "tuned_model_name": best_spec.name,
            "best_validation_mae": float(study.best_value),
            "refit_validation_mae": float(best_metrics["mae"]),
            "n_trials": len(study.trials),
            "best_params": json.dumps(study.best_params, sort_keys=True),
        }
        rows.append(row)
        summaries["studies"].append(
            {
                "base_model_name": model_name,
                "tuned_model_name": best_spec.name,
                "best_validation_mae": float(study.best_value),
                "refit_validation_metrics": best_metrics,
                "n_trials": len(study.trials),
                "best_params": study.best_params,
            }
        )

    comparison = pd.DataFrame(rows).sort_values("best_validation_mae")
    comparison_path = settings.metrics_dir / "tuning_comparison.csv"
    summary_path = Path(settings.metrics_dir) / "optuna_top_models_summary.json"
    comparison.to_csv(comparison_path, index=False)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    summaries["comparison_path"] = str(comparison_path)
    summaries["summary_path"] = str(summary_path)
    return summaries


def _global_stable_candidate_names(
    settings: Settings,
    top_n: int,
    explicit_model_names: list[str] | None = None,
) -> list[str]:
    """Handle global stable candidate names."""
    if explicit_model_names:
        names = explicit_model_names
    else:
        benchmark_path = settings.metrics_dir / "global_stable_benchmark.csv"
        if benchmark_path.exists():
            table = pd.read_csv(benchmark_path).sort_values("validation_mae")
            names = table["model_name"].astype(str).tolist()
        else:
            names = [spec.name for spec in global_stable_specs()]

    selected: list[str] = []
    for name in names:
        if name not in selected and _supports_tuning(name):
            selected.append(name)
        if len(selected) == top_n:
            break
    return selected


def tune_global_stable_models(
    settings: Settings,
    n_trials: int = 20,
    top_n: int = 2,
    model_names: list[str] | None = None,
    timeout_seconds_per_model: int | None = 300,
) -> dict[str, object]:
    """Perform tune global stable models."""
    settings.ensure_output_dirs()
    rfqs, volatility, reference = load_all(settings)
    rfqs, volatility, reference = validate_all(rfqs, volatility, reference)
    trainable = trainable_rfqs(rfqs)
    benchmark_path = settings.metrics_dir / "global_stable_benchmark.csv"
    if not benchmark_path.exists() and model_names is None:
        run_global_stable_experiment(trainable, volatility, reference, settings)

    split = temporal_split(trainable)
    feature_set = FeatureBuilder().build(trainable, volatility, reference, include_target=True)
    X = feature_set.frame
    y = feature_set.target
    if y is None:
        raise ValueError("Global stable tuning requires target values.")

    candidates = _global_stable_candidate_names(
        settings,
        top_n=top_n,
        explicit_model_names=model_names,
    )
    if not candidates:
        raise ValueError("No supported global stable tuning candidates were found.")

    hash_tags = short_data_hash_tags(data_hashes(settings))
    rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {
        "split": split.description,
        "n_trials_per_model": n_trials,
        "candidates": candidates,
        **reproducibility_manifest(settings),
        "studies": [],
    }

    for model_name in candidates:
        study = optuna.create_study(
            direction="minimize",
            study_name=f"{model_name}_global_stable_validation_mae",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
        family_and_block = _tuning_family_and_block(model_name)
        parent_feature_block = family_and_block[1] if family_and_block else ""
        numeric_features, categorical_features = select_feature_block(parent_feature_block)
        validation_dataset = metric_dataset_from_features(
            X,
            y,
            split.validation_index,
            [*numeric_features, *categorical_features],
        )

        def objective(
            trial: optuna.Trial,
            candidate_name: str = model_name,
            objective_validation_dataset: pd.DataFrame = validation_dataset,
        ) -> float:
            """Handle objective."""
            spec = _suggest_spec(candidate_name, trial)
            mae, metrics = _evaluate_spec(
                spec,
                X,
                y,
                split.train_index,
                split.validation_index,
            )
            log_mlflow_run(
                run_name=f"global_stable__{candidate_name}__optuna_trial_{trial.number:03d}",
                params={
                    **trial.params,
                    "trial_number": trial.number,
                    "model_name": spec.name,
                    "base_model_name": candidate_name,
                    "model_family": model_family_from_name(spec.name),
                    "run_type": "global_stable_optuna_trial",
                    "feature_block": spec.feature_block,
                    "encoding_strategy": spec.encoding_strategy,
                    "scope": "global",
                    "segment": "all",
                    "dataset_name": mlflow_dataset_name(spec.feature_block, "validation"),
                    "split": split.description,
                },
                metrics={f"validation_{key}": float(value) for key, value in metrics.items()},
                metric_dataset=objective_validation_dataset,
                dataset_name=mlflow_dataset_name(spec.feature_block, "validation"),
                tags={
                    "run_type": "global_stable_optuna_trial",
                    "model_name": spec.name,
                    "base_model_name": candidate_name,
                    "model_family": model_family_from_name(spec.name),
                    "feature_block": spec.feature_block,
                    "encoding_strategy": spec.encoding_strategy,
                    "scope": "global",
                    "segment": "all",
                    "dataset_name": mlflow_dataset_name(spec.feature_block, "validation"),
                    "trial_number": str(trial.number),
                    **hash_tags,
                },
                nested=True,
            )
            return mae

        with mlflow_parent_run(
            run_name=f"global_stable__{model_name}__optuna_study",
            params={
                "model_name": model_name,
                "run_type": "global_stable_optuna_study",
                "feature_block": parent_feature_block,
                "scope": "global",
                "segment": "all",
                "split": split.description,
                "n_trials": n_trials,
            },
            tags={
                "run_type": "global_stable_optuna_study",
                "model_name": model_name,
                "feature_block": parent_feature_block,
                **hash_tags,
            },
        ):
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout_seconds_per_model,
                gc_after_trial=True,
            )
        best_spec = _suggest_spec(model_name, optuna.trial.FixedTrial(study.best_params))
        numeric_features, categorical_features = select_feature_block(best_spec.feature_block)
        feature_columns = [*numeric_features, *categorical_features]

        validation_pipeline = build_pipeline(best_spec)
        validation_fit_start = time.perf_counter()
        validation_pipeline.fit(X.loc[split.train_index], y.loc[split.train_index])
        validation_fit_seconds = time.perf_counter() - validation_fit_start
        validation_predict_start = time.perf_counter()
        validation_predictions = validation_pipeline.predict(X.loc[split.validation_index])
        validation_predict_seconds = time.perf_counter() - validation_predict_start
        validation_metrics = regression_metrics(
            y.loc[split.validation_index],
            validation_predictions,
        )

        validation_dataset = metric_dataset_from_features(
            X,
            y,
            split.validation_index,
            feature_columns,
        )

        log_mlflow_run(
            run_name=f"{model_name}__best_validation",
            params={
                **study.best_params,
                "model_name": best_spec.name,
                "base_model_name": model_name,
                "run_type": "global_stable_tuning_best_model",
                "feature_block": best_spec.feature_block,
                "encoding_strategy": best_spec.encoding_strategy,
                "model_family": model_family_from_name(best_spec.name),
                "split": split.description,
                "dataset_name": mlflow_dataset_name(best_spec.feature_block, "validation"),
                "train_rows": len(split.train_index),
                "validation_rows": len(split.validation_index),
                "n_numeric_features": len(numeric_features),
                "n_categorical_features": len(categorical_features),
                "n_total_features": len(numeric_features) + len(categorical_features),
            },
            metrics={
                **{f"validation_{key}": float(value) for key, value in validation_metrics.items()},
                "fit_seconds": float(validation_fit_seconds),
                "validation_predict_seconds": float(validation_predict_seconds),
            },
            tags={
                "run_type": "global_stable_tuning_best_model",
                "model_name": best_spec.name,
                "base_model_name": model_name,
                "model_family": model_family_from_name(best_spec.name),
                "feature_block": best_spec.feature_block,
                "encoding_strategy": best_spec.encoding_strategy,
                "scope": "global",
                "segment": "all",
                "dataset_name": mlflow_dataset_name(best_spec.feature_block, "validation"),
                **hash_tags,
            },
            model=validation_pipeline,
            input_example=X.loc[split.train_index, feature_columns].head(5),
            metric_dataset=validation_dataset,
            dataset_name=mlflow_dataset_name(best_spec.feature_block, "validation"),
            registered_model_name=model_family_from_name(best_spec.name),
        )

        row = {
            "base_model_name": model_name,
            "tuned_model_name": best_spec.name,
            "feature_block": best_spec.feature_block,
            "encoding_strategy": best_spec.encoding_strategy,
            "best_validation_mae": float(study.best_value),
            "refit_validation_mae": float(validation_metrics["mae"]),
            "validation_rmse": float(validation_metrics["rmse"]),
            "validation_r2": float(validation_metrics["r2"]),
            "validation_median_absolute_error": float(validation_metrics["median_absolute_error"]),
            "train_rows": len(split.train_index),
            "validation_rows": len(split.validation_index),
            "n_numeric_features": len(numeric_features),
            "n_categorical_features": len(categorical_features),
            "n_total_features": len(numeric_features) + len(categorical_features),
            "fit_seconds": validation_fit_seconds,
            "validation_predict_seconds": validation_predict_seconds,
            "n_trials": len(study.trials),
            "best_params": json.dumps(study.best_params, sort_keys=True),
        }
        rows.append(row)
        summaries["studies"].append(
            {
                "base_model_name": model_name,
                "tuned_model_name": best_spec.name,
                "best_validation_mae": float(study.best_value),
                "refit_validation_metrics": validation_metrics,
                "n_trials": len(study.trials),
                "best_params": study.best_params,
            }
        )

    comparison = pd.DataFrame(rows).sort_values("best_validation_mae")
    comparison_path = settings.metrics_dir / "global_stable_tuning_comparison.csv"
    summary_path = Path(settings.metrics_dir) / "optuna_global_stable_summary.json"
    comparison.to_csv(comparison_path, index=False)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    summaries["comparison_path"] = str(comparison_path)
    summaries["summary_path"] = str(summary_path)
    return summaries


def tune_segmented_models(
    settings: Settings,
    n_trials: int = 20,
    model_names: list[str] | None = None,
    timeout_seconds_per_model: int | None = 300,
) -> dict[str, object]:
    """Perform tune segmented models."""
    settings.ensure_output_dirs()
    rfqs, volatility, reference = load_all(settings)
    rfqs, volatility, reference = validate_all(rfqs, volatility, reference)
    trainable = trainable_rfqs(rfqs)
    split = temporal_split(trainable)
    feature_set = FeatureBuilder().build(trainable, volatility, reference, include_target=True)
    X = feature_set.frame
    y = feature_set.target
    if y is None:
        raise ValueError("Segmented tuning requires target values.")

    specs_by_segment = segmented_model_specs()
    wanted = set(model_names) if model_names else None
    hash_tags = short_data_hash_tags(data_hashes(settings))
    rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {
        "split": split.description,
        "n_trials_per_model": n_trials,
        **reproducibility_manifest(settings),
        "studies": [],
    }
    segment_masks = {
        "single": X["is_single_underlying"].eq(1),
        "worst_of": X["is_worst_of"].eq(1),
    }

    for segment, specs in specs_by_segment.items():
        segment_specs = [spec for spec in specs if wanted is None or spec.name in wanted]
        if not segment_specs:
            continue
        segment_index = X.index[segment_masks[segment]]
        train_index = split.train_index.intersection(segment_index)
        validation_index = split.validation_index.intersection(segment_index)
        if len(train_index) == 0 or len(validation_index) == 0:
            continue

        for base_spec in segment_specs:
            study = optuna.create_study(
                direction="minimize",
                study_name=f"{base_spec.name}_validation_mae",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            )
            numeric_features, categorical_features = select_feature_block(base_spec.feature_block)
            validation_dataset = metric_dataset_from_features(
                X,
                y,
                validation_index,
                [*numeric_features, *categorical_features],
            )

            def objective(
                trial: optuna.Trial,
                candidate_name: str = base_spec.name,
                segment_name: str = segment,
                objective_train_index: pd.Index = train_index,
                objective_validation_index: pd.Index = validation_index,
                objective_validation_dataset: pd.DataFrame = validation_dataset,
            ) -> float:
                """Handle objective."""
                spec = _suggest_spec(candidate_name, trial)
                mae, metrics = _evaluate_spec(
                    spec,
                    X,
                    y,
                    objective_train_index,
                    objective_validation_index,
                )
                log_mlflow_run(
                    run_name=(
                        f"segmented__{segment_name}__{candidate_name}"
                        f"__optuna_trial_{trial.number:03d}"
                    ),
                    params={
                        **trial.params,
                        "trial_number": trial.number,
                        "segment": segment_name,
                        "model_name": spec.name,
                        "base_model_name": candidate_name,
                        "model_family": model_family_from_name(spec.name),
                        "run_type": "segmented_optuna_trial",
                        "feature_block": spec.feature_block,
                        "encoding_strategy": spec.encoding_strategy,
                        "scope": "segmented",
                        "dataset_name": mlflow_dataset_name(spec.feature_block, "validation"),
                        "split": split.description,
                    },
                    metrics={f"validation_{key}": float(value) for key, value in metrics.items()},
                    metric_dataset=objective_validation_dataset,
                    dataset_name=mlflow_dataset_name(spec.feature_block, "validation"),
                    tags={
                        "run_type": "segmented_optuna_trial",
                        "segment": segment_name,
                        "model_name": spec.name,
                        "base_model_name": candidate_name,
                        "model_family": model_family_from_name(spec.name),
                        "feature_block": spec.feature_block,
                        "encoding_strategy": spec.encoding_strategy,
                        "scope": "segmented",
                        "dataset_name": mlflow_dataset_name(spec.feature_block, "validation"),
                        "trial_number": str(trial.number),
                        **hash_tags,
                    },
                    nested=True,
                )
                return mae

            with mlflow_parent_run(
                run_name=f"segmented__{segment}__{base_spec.name}__optuna_study",
                params={
                    "segment": segment,
                    "model_name": base_spec.name,
                    "run_type": "segmented_optuna_study",
                    "feature_block": base_spec.feature_block,
                    "scope": "segmented",
                    "split": split.description,
                    "n_trials": n_trials,
                },
                tags={
                    "run_type": "segmented_optuna_study",
                    "segment": segment,
                    "model_name": base_spec.name,
                    "feature_block": base_spec.feature_block,
                    **hash_tags,
                },
            ):
                study.optimize(
                    objective,
                    n_trials=n_trials,
                    timeout=timeout_seconds_per_model,
                    gc_after_trial=True,
                )
            best_spec = _suggest_spec(base_spec.name, optuna.trial.FixedTrial(study.best_params))
            validation_metrics = _fit_and_log_best_model(
                model_name=base_spec.name,
                best_spec=best_spec,
                best_params={
                    **study.best_params,
                    "segment": segment,
                    "run_type": "segmented_tuning_best_model",
                },
                X=X,
                y=y,
                split_description=split.description,
                train_index=train_index,
                validation_index=validation_index,
                hash_tags=hash_tags,
                run_name=f"{base_spec.name}__best",
            )
            row = {
                "segment": segment,
                "base_model_name": base_spec.name,
                "tuned_model_name": best_spec.name,
                "feature_block": best_spec.feature_block,
                "encoding_strategy": best_spec.encoding_strategy,
                "best_validation_mae": float(study.best_value),
                "refit_validation_mae": float(validation_metrics["mae"]),
                "train_rows": len(train_index),
                "validation_rows": len(validation_index),
                "n_trials": len(study.trials),
                "best_params": json.dumps(study.best_params, sort_keys=True),
            }
            rows.append(row)
            summaries["studies"].append(
                {
                    "segment": segment,
                    "base_model_name": base_spec.name,
                    "tuned_model_name": best_spec.name,
                    "best_validation_mae": float(study.best_value),
                    "refit_validation_metrics": validation_metrics,
                    "n_trials": len(study.trials),
                    "best_params": study.best_params,
                }
            )

    if not rows:
        raise ValueError("No supported segmented tuning candidates were found.")

    comparison = pd.DataFrame(rows).sort_values(["segment", "best_validation_mae"])
    comparison_path = settings.metrics_dir / "segmented_tuning_comparison.csv"
    summary_path = Path(settings.metrics_dir) / "optuna_segmented_summary.json"
    comparison.to_csv(comparison_path, index=False)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    summaries["comparison_path"] = str(comparison_path)
    summaries["summary_path"] = str(summary_path)
    return summaries


def tuned_spec_from_params(base_model_name: str, best_params: dict[str, Any]) -> ModelSpec:
    """Perform tuned spec from params."""
    return _suggest_spec(base_model_name, optuna.trial.FixedTrial(best_params))


def load_best_tuned_spec(settings: Settings) -> tuple[ModelSpec, dict[str, Any]] | None:
    """Return load best tuned spec."""
    summary_path = settings.metrics_dir / "optuna_top_models_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    studies = summary.get("studies", [])
    if not studies:
        return None
    best = min(studies, key=lambda item: float(item["best_validation_mae"]))
    spec = tuned_spec_from_params(
        str(best["base_model_name"]),
        dict(best["best_params"]),
    )
    return spec, best


def load_tuned_specs(
    settings: Settings,
    base_model_names: list[str] | None = None,
) -> dict[str, tuple[ModelSpec, dict[str, Any]]]:
    """Return load tuned specs."""
    summary_path = settings.metrics_dir / "optuna_top_models_summary.json"
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    wanted = set(base_model_names) if base_model_names else None
    specs: dict[str, tuple[ModelSpec, dict[str, Any]]] = {}
    for study in summary.get("studies", []):
        base_model_name = str(study["base_model_name"])
        if wanted is not None and base_model_name not in wanted:
            continue
        specs[base_model_name] = (
            tuned_spec_from_params(base_model_name, dict(study["best_params"])),
            study,
        )
    return specs
