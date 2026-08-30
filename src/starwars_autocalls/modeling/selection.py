"""Selection module."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from starwars_autocalls.config import RANDOM_SEED, Settings
from starwars_autocalls.data.loading import load_all, trainable_rfqs
from starwars_autocalls.data.validation import validate_all
from starwars_autocalls.features import FeatureBuilder
from starwars_autocalls.modeling.benchmark import (
    load_benchmark_table,
    run_benchmark,
    run_segmented_benchmark,
)
from starwars_autocalls.modeling.evaluation import (
    regression_metrics,
    rolling_temporal_folds,
    temporal_split,
)
from starwars_autocalls.modeling.specs import (
    ModelSpec,
    ablation_specs,
    build_pipeline,
    default_model_specs,
    global_stable_specs,
    segmented_model_specs,
)
from starwars_autocalls.modeling.tuning import load_best_tuned_spec, tuned_spec_from_params
from starwars_autocalls.observability.progress import report_progress
from starwars_autocalls.reproducibility import reproducibility_manifest

MIN_PRACTICAL_MAE_IMPROVEMENT_MONTHS = 0.25


@dataclass(frozen=True)
class EvaluatedCandidate:
    """Represent EvaluatedCandidate."""

    strategy: str
    model_name: str
    feature_block: str
    encoding_strategy: str
    validation_mae: float
    rolling_mae_mean: float | None
    rolling_mae_max: float | None
    train_rows: int
    validation_rows: int
    details: dict[str, Any]
    validation_abs_errors: pd.Series
    validation_blocks: pd.Series
    rolling_fold_maes: np.ndarray


def _spec_by_name(model_name: str) -> ModelSpec:
    """Handle spec by name."""
    specs = [*default_model_specs(), *ablation_specs(), *global_stable_specs()]
    for segment_specs in segmented_model_specs().values():
        specs.extend(segment_specs)
    for spec in specs:
        if spec.name == model_name:
            return spec
    raise ValueError(f"Could not reconstruct model spec {model_name!r}.")


def _best_global_spec(
    trainable: pd.DataFrame,
    volatility: pd.DataFrame,
    reference: pd.DataFrame,
    settings: Settings,
    allow_tuned: bool = True,
) -> ModelSpec:
    """Handle best global spec."""
    tuned = load_best_tuned_spec(settings) if allow_tuned else None
    if tuned is not None:
        return tuned[0]
    benchmark_path = settings.metrics_dir / "benchmark_comparison.csv"
    comparison = (
        load_benchmark_table(benchmark_path)
        if benchmark_path.exists()
        else run_benchmark(trainable, volatility, reference, settings, include_ablations=True)
    )
    return _spec_by_name(str(comparison.iloc[0]["model_name"]))


def _best_segmented_specs(
    trainable: pd.DataFrame,
    volatility: pd.DataFrame,
    reference: pd.DataFrame,
    settings: Settings,
    allow_tuned: bool = True,
) -> dict[str, ModelSpec]:
    """Handle best segmented specs."""
    summary_path = settings.metrics_dir / "optuna_segmented_summary.json"
    if allow_tuned and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        best_by_segment: dict[str, dict[str, Any]] = {}
        for study in summary.get("studies", []):
            segment = str(study["segment"])
            current = best_by_segment.get(segment)
            if current is None or float(study["best_validation_mae"]) < float(
                current["best_validation_mae"]
            ):
                best_by_segment[segment] = study
        if {"single", "worst_of"}.issubset(best_by_segment):
            return {
                segment: tuned_spec_from_params(
                    str(study["base_model_name"]),
                    dict(study["best_params"]),
                )
                for segment, study in best_by_segment.items()
            }

    benchmark_path = settings.metrics_dir / "segmented_benchmark_single_worstof.csv"
    detail = (
        pd.read_csv(benchmark_path)
        if benchmark_path.exists()
        else run_segmented_benchmark(trainable, volatility, reference, settings)
    )
    if detail.empty:
        raise ValueError("Segmented benchmark did not produce candidates.")
    specs: dict[str, ModelSpec] = {}
    for segment, rows in detail.sort_values(["segment", "validation_mae"]).groupby("segment"):
        specs[str(segment)] = _spec_by_name(str(rows.iloc[0]["model_name"]))
    if {"single", "worst_of"} - set(specs):
        raise ValueError("Segmented candidates must include both single and worst_of.")
    return specs


def _rolling_metrics_for_spec(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    rfqs: pd.DataFrame,
) -> tuple[float | None, float | None, np.ndarray]:
    """Handle rolling metrics for spec."""
    maes: list[float] = []
    for fold in rolling_temporal_folds(rfqs):
        report_progress(f"Comparación de estrategias, rolling {fold.description}: {spec.name}")
        pipeline = build_pipeline(spec)
        pipeline.fit(X.loc[fold.train_index], y.loc[fold.train_index])
        predictions = pipeline.predict(X.loc[fold.validation_index])
        maes.append(regression_metrics(y.loc[fold.validation_index], predictions)["mae"])
    if not maes:
        return None, None, np.array([])
    return float(np.mean(maes)), float(np.max(maes)), np.array(maes)


def _evaluate_global(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    rfqs: pd.DataFrame,
) -> EvaluatedCandidate:
    """Handle evaluate global."""
    split = temporal_split(rfqs)
    report_progress(f"Comparación global en validation: {spec.name}")
    validation_pipeline = build_pipeline(spec)
    validation_pipeline.fit(X.loc[split.train_index], y.loc[split.train_index])
    validation_predictions = validation_pipeline.predict(X.loc[split.validation_index])
    validation_metrics = regression_metrics(y.loc[split.validation_index], validation_predictions)
    validation_abs_errors = (y.loc[split.validation_index] - validation_predictions).abs()

    rolling_mean, rolling_max, rolling_fold_maes = _rolling_metrics_for_spec(spec, X, y, rfqs)

    return EvaluatedCandidate(
        strategy="global",
        model_name=spec.name,
        feature_block=spec.feature_block,
        encoding_strategy=spec.encoding_strategy,
        validation_mae=float(validation_metrics["mae"]),
        rolling_mae_mean=rolling_mean,
        rolling_mae_max=rolling_max,
        train_rows=len(split.train_index),
        validation_rows=len(split.validation_index),
        details={
            "validation_metrics": validation_metrics,
            "selection_metric": "validation_mae",
            "test_usage": "forbidden_during_selection",
        },
        validation_abs_errors=validation_abs_errors,
        validation_blocks=pd.to_datetime(rfqs.loc[split.validation_index, "requested_date"])
        .dt.to_period("Q")
        .astype(str),
        rolling_fold_maes=rolling_fold_maes,
    )


def _weighted_metric(rows: list[dict[str, Any]], metric: str, weight: str) -> float:
    """Handle weighted metric."""
    total = sum(int(row[weight]) for row in rows)
    return float(sum(float(row[metric]) * int(row[weight]) for row in rows) / total)


def _evaluate_segmented(
    specs: dict[str, ModelSpec],
    X: pd.DataFrame,
    y: pd.Series,
    rfqs: pd.DataFrame,
) -> EvaluatedCandidate:
    """Handle evaluate segmented."""
    split = temporal_split(rfqs)
    segment_masks = {
        "single": X["is_single_underlying"].eq(1),
        "worst_of": X["is_worst_of"].eq(1),
    }
    segment_rows: list[dict[str, Any]] = []
    validation_abs_errors_parts: list[pd.Series] = []
    for segment, spec in specs.items():
        segment_index = X.index[segment_masks[segment]]
        train_index = split.train_index.intersection(segment_index)
        validation_index = split.validation_index.intersection(segment_index)

        report_progress(f"Comparación segmentada en validation, {segment}: {spec.name}")
        validation_pipeline = build_pipeline(spec)
        validation_pipeline.fit(X.loc[train_index], y.loc[train_index])
        validation_predictions = validation_pipeline.predict(X.loc[validation_index])
        validation_metrics = regression_metrics(y.loc[validation_index], validation_predictions)
        validation_abs_errors_parts.append((y.loc[validation_index] - validation_predictions).abs())

        segment_rows.append(
            {
                "segment": segment,
                "model_name": spec.name,
                "feature_block": spec.feature_block,
                "encoding_strategy": spec.encoding_strategy,
                "validation_mae": float(validation_metrics["mae"]),
                "train_rows": len(train_index),
                "validation_rows": len(validation_index),
                "validation_metrics": validation_metrics,
            }
        )

    rolling_fold_maes: list[float] = []
    for fold in rolling_temporal_folds(rfqs):
        fold_rows: list[dict[str, float | int]] = []
        for segment, spec in specs.items():
            segment_index = X.index[segment_masks[segment]]
            train_index = fold.train_index.intersection(segment_index)
            validation_index = fold.validation_index.intersection(segment_index)
            if len(train_index) == 0 or len(validation_index) == 0:
                continue
            report_progress(
                f"Comparación segmentada rolling {fold.description}, {segment}: {spec.name}"
            )
            pipeline = build_pipeline(spec)
            pipeline.fit(X.loc[train_index], y.loc[train_index])
            predictions = pipeline.predict(X.loc[validation_index])
            metrics = regression_metrics(y.loc[validation_index], predictions)
            fold_rows.append(
                {
                    "validation_mae": float(metrics["mae"]),
                    "validation_rows": len(validation_index),
                }
            )
        if fold_rows:
            rolling_fold_maes.append(
                _weighted_metric(fold_rows, "validation_mae", "validation_rows")
            )

    model_name = " + ".join(f"{segment}:{spec.name}" for segment, spec in sorted(specs.items()))
    feature_block = " + ".join(
        f"{segment}:{spec.feature_block}" for segment, spec in sorted(specs.items())
    )
    encoding_strategy = " + ".join(
        f"{segment}:{spec.encoding_strategy}" for segment, spec in sorted(specs.items())
    )
    return EvaluatedCandidate(
        strategy="segmented_by_basket_type",
        model_name=model_name,
        feature_block=feature_block,
        encoding_strategy=encoding_strategy,
        validation_mae=_weighted_metric(segment_rows, "validation_mae", "validation_rows"),
        rolling_mae_mean=float(np.mean(rolling_fold_maes)) if rolling_fold_maes else None,
        rolling_mae_max=float(np.max(rolling_fold_maes)) if rolling_fold_maes else None,
        train_rows=sum(int(row["train_rows"]) for row in segment_rows),
        validation_rows=sum(int(row["validation_rows"]) for row in segment_rows),
        details={
            "segments": segment_rows,
            "selection_metric": "validation_mae",
            "test_usage": "forbidden_during_selection",
        },
        validation_abs_errors=pd.concat(validation_abs_errors_parts).sort_index(),
        validation_blocks=pd.to_datetime(rfqs.loc[split.validation_index, "requested_date"])
        .dt.to_period("Q")
        .astype(str),
        rolling_fold_maes=np.array(rolling_fold_maes),
    )


def _bootstrap_mae_diff(
    values_global: np.ndarray,
    values_segmented: np.ndarray,
    seed: int = RANDOM_SEED,
    n_bootstrap: int = 5000,
) -> dict[str, Any]:
    """Percentile bootstrap CI for (mean(values_global) - mean(values_segmented)).

    Positive point estimate means the global candidate has higher mean error
    (segmented candidate wins); negative means the opposite. A 95% CI that
    excludes zero is treated as a statistically distinguishable difference.
    """
    n = min(len(values_global), len(values_segmented))
    values_global = values_global[:n]
    values_segmented = values_segmented[:n]
    if n == 0:
        return {
            "point_estimate_global_minus_segmented_mae": None,
            "ci_95_low": None,
            "ci_95_high": None,
            "n_bootstrap": n_bootstrap,
            "n_observations": 0,
            "significant_at_5pct": False,
        }
    rng = np.random.default_rng(seed)
    resample_idx = rng.integers(0, n, size=(n_bootstrap, n))
    diffs = values_global[resample_idx].mean(axis=1) - values_segmented[resample_idx].mean(axis=1)
    ci_low, ci_high = (float(value) for value in np.percentile(diffs, [2.5, 97.5]))
    return {
        "point_estimate_global_minus_segmented_mae": float(
            values_global.mean() - values_segmented.mean()
        ),
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "n_bootstrap": n_bootstrap,
        "n_observations": int(n),
        "significant_at_5pct": bool(ci_low > 0 or ci_high < 0),
    }


def _block_bootstrap_mae_diff(
    values_global: pd.Series,
    values_segmented: pd.Series,
    blocks: pd.Series,
    seed: int = RANDOM_SEED,
    n_bootstrap: int = 5000,
) -> dict[str, Any]:
    """Handle block bootstrap mae diff."""
    common_index = values_global.index.intersection(values_segmented.index).intersection(
        blocks.index
    )
    data = pd.DataFrame(
        {
            "global": values_global.loc[common_index],
            "segmented": values_segmented.loc[common_index],
            "block": blocks.loc[common_index].astype(str),
        }
    )
    unique_blocks = data["block"].unique()
    if len(unique_blocks) == 0:
        return _bootstrap_mae_diff(np.array([]), np.array([]), seed, n_bootstrap)
    rows_by_block = {block: rows for block, rows in data.groupby("block")}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        sample = pd.concat([rows_by_block[block] for block in sampled_blocks], ignore_index=True)
        diffs[iteration] = sample["global"].mean() - sample["segmented"].mean()
    ci_low, ci_high = (float(value) for value in np.percentile(diffs, [2.5, 97.5]))
    return {
        "point_estimate_global_minus_segmented_mae": float(
            data["global"].mean() - data["segmented"].mean()
        ),
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "n_bootstrap": n_bootstrap,
        "n_observations": len(data),
        "n_blocks": len(unique_blocks),
        "block_unit": "requested_quarter",
        "significant_at_5pct": bool(ci_low > 0 or ci_high < 0),
    }


def _statistical_comparison(
    global_candidate: EvaluatedCandidate,
    segmented_candidate: EvaluatedCandidate,
) -> dict[str, Any]:
    """Handle statistical comparison."""
    common_index = global_candidate.validation_abs_errors.index.intersection(
        segmented_candidate.validation_abs_errors.index
    )
    validation_row_bootstrap = _bootstrap_mae_diff(
        global_candidate.validation_abs_errors.loc[common_index].to_numpy(),
        segmented_candidate.validation_abs_errors.loc[common_index].to_numpy(),
    )
    validation_block_bootstrap = _block_bootstrap_mae_diff(
        global_candidate.validation_abs_errors,
        segmented_candidate.validation_abs_errors,
        global_candidate.validation_blocks,
    )
    rolling_bootstrap = _bootstrap_mae_diff(
        global_candidate.rolling_fold_maes,
        segmented_candidate.rolling_fold_maes,
    )
    validation_significant = validation_block_bootstrap["significant_at_5pct"]
    validation_point = validation_block_bootstrap["point_estimate_global_minus_segmented_mae"]
    practically_material = bool(
        validation_point is not None
        and abs(validation_point) >= MIN_PRACTICAL_MAE_IMPROVEMENT_MONTHS
    )
    if validation_significant and practically_material and validation_point is not None:
        preferred_strategy = "segmented_by_basket_type" if validation_point > 0 else "global"
        decision_basis = "validation_quarter_block_bootstrap_significant"
    else:
        preferred_strategy = "global"
        decision_basis = "not_significant_or_not_material_default_to_simpler_model"
    return {
        "method": (
            "Quarter-block percentile bootstrap (5000 resamples, seed="
            f"{RANDOM_SEED}) on validation |error| (primary, n="
            f"{validation_block_bootstrap['n_observations']}) and on per-fold rolling MAE "
            f"(secondary, n={rolling_bootstrap['n_observations']} folds). "
            "Positive point estimate favors segmented; CI excluding zero is treated "
            "as a statistically distinguishable difference at the 5% level."
        ),
        "validation_block_bootstrap": validation_block_bootstrap,
        "validation_row_bootstrap_diagnostic": validation_row_bootstrap,
        "rolling_fold_bootstrap": rolling_bootstrap,
        "minimum_practical_mae_improvement_months": MIN_PRACTICAL_MAE_IMPROVEMENT_MONTHS,
        "practically_material": practically_material,
        "decision_basis": decision_basis,
        "preferred_strategy_by_significance": preferred_strategy,
    }


def _candidate_to_row(candidate: EvaluatedCandidate, selected_strategy: str) -> dict[str, Any]:
    """Handle candidate to row."""
    return {
        "strategy": candidate.strategy,
        "selected_for_serving": candidate.strategy == selected_strategy,
        "model_name": candidate.model_name,
        "feature_block": candidate.feature_block,
        "encoding_strategy": candidate.encoding_strategy,
        "selection_metric": "validation_mae",
        "validation_mae": candidate.validation_mae,
        "rolling_mae_mean": candidate.rolling_mae_mean,
        "rolling_mae_max": candidate.rolling_mae_max,
        "train_rows": candidate.train_rows,
        "validation_rows": candidate.validation_rows,
    }


def write_model_selection_report(settings: Settings) -> dict[str, Any]:
    """Perform write model selection report."""
    settings.ensure_output_dirs()
    rfqs, volatility, reference = load_all(settings)
    rfqs, volatility, reference = validate_all(rfqs, volatility, reference)
    trainable = trainable_rfqs(rfqs).reset_index(drop=True)
    feature_set = FeatureBuilder().build(trainable, volatility, reference, include_target=True)
    if feature_set.target is None:
        raise ValueError("Model selection requires target values.")

    has_global_tuning = (settings.metrics_dir / "optuna_top_models_summary.json").exists() or (
        settings.metrics_dir / "optuna_global_stable_summary.json"
    ).exists()
    has_segmented_tuning = (settings.metrics_dir / "optuna_segmented_summary.json").exists()
    use_tuned_candidates = has_global_tuning and has_segmented_tuning
    global_spec = _best_global_spec(
        trainable,
        volatility,
        reference,
        settings,
        allow_tuned=use_tuned_candidates,
    )
    segmented_specs = _best_segmented_specs(
        trainable,
        volatility,
        reference,
        settings,
        allow_tuned=use_tuned_candidates,
    )
    global_candidate = _evaluate_global(
        global_spec, feature_set.frame, feature_set.target, trainable
    )
    segmented_candidate = _evaluate_segmented(
        segmented_specs, feature_set.frame, feature_set.target, trainable
    )
    candidates = [global_candidate, segmented_candidate]

    statistical_comparison = _statistical_comparison(global_candidate, segmented_candidate)
    selected_strategy = statistical_comparison["preferred_strategy_by_significance"]
    selected = next(
        candidate for candidate in candidates if candidate.strategy == selected_strategy
    )

    rows = [_candidate_to_row(candidate, selected.strategy) for candidate in candidates]
    comparison = pd.DataFrame(rows).sort_values(
        ["selected_for_serving", "validation_mae"],
        ascending=[False, True],
    )
    comparison_path = settings.metrics_dir / "model_selection_protocol.csv"
    comparison.to_csv(comparison_path, index=False)

    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_rule": (
            "primary: statistical significance of the validation-residual bootstrap CI "
            "for (global_mae - segmented_mae); ties/non-significant differences default "
            "to the simpler 'global' strategy; rolling-fold bootstrap is secondary. "
            "The final test holdout is inaccessible to this command."
        ),
        "selected_strategy": selected.strategy,
        "selected_model_name": selected.model_name,
        "candidate_source": "tuned" if use_tuned_candidates else "untuned_benchmark",
        "statistical_comparison": statistical_comparison,
        "comparison_path": str(comparison_path),
        "candidates": rows,
        "candidate_details": {candidate.strategy: candidate.details for candidate in candidates},
        **reproducibility_manifest(settings),
    }
    summary_path = settings.metrics_dir / "model_selection_protocol_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
