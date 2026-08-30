"""Experiment Commands module."""

from __future__ import annotations

import pandas as pd
import typer

from starwars_autocalls.cli.common import (
    command_status,
    parse_model_names,
    print_dataframe,
    print_json_panel,
    validated_data,
)
from starwars_autocalls.config import Settings
from starwars_autocalls.data.loading import trainable_rfqs
from starwars_autocalls.modeling.benchmark import (
    run_benchmark,
    run_global_stable_experiment,
    run_robustness_report,
    run_rolling_benchmark,
    run_segmented_benchmark,
    run_segmented_rolling_benchmark,
)
from starwars_autocalls.modeling.selection import write_model_selection_report
from starwars_autocalls.modeling.training import train_segmented_with_optuna
from starwars_autocalls.modeling.tuning import (
    tune_global_stable_models,
    tune_segmented_models,
    tune_top_models,
)


def benchmark(
    models: str | None = typer.Option(
        None,
        help="Modelos separados por comas. Sin opción evalúa todos los candidatos.",
    ),
) -> None:
    """Compara candidatos globales en el split temporal de validation."""
    settings = Settings()
    with command_status("Ejecutando benchmark global"):
        rfqs, volatility, reference = validated_data(settings)
        comparison = run_benchmark(
            trainable_rfqs(rfqs),
            volatility,
            reference,
            settings,
            model_names=parse_model_names(models),
        )
    print_dataframe(
        comparison,
        title="Benchmark global",
        columns=[
            "model_name",
            "feature_block",
            "encoding_strategy",
            "validation_mae",
            "validation_rmse",
            "validation_r2",
            "fit_seconds",
        ],
        max_rows=10,
    )


def rolling_benchmark(
    models: str | None = typer.Option(
        None,
        help="Modelos separados por comas. Sin opción usa los candidatos del benchmark.",
    ),
) -> None:
    """Comprueba la estabilidad de candidatos en ventanas temporales sucesivas."""
    settings = Settings()
    with command_status("Ejecutando benchmark temporal rolling"):
        rfqs, volatility, reference = validated_data(settings)
        summary = run_rolling_benchmark(
            trainable_rfqs(rfqs),
            volatility,
            reference,
            settings,
            model_names=parse_model_names(models),
        )
    print_dataframe(
        summary,
        title="Benchmark rolling",
        columns=[
            "model_name",
            "feature_block",
            "rolling_mae_mean",
            "rolling_mae_std",
            "rolling_mae_max",
            "n_folds",
        ],
        max_rows=10,
    )


def segmented_benchmark(
    models: str | None = typer.Option(
        None,
        help="Modelos segmentados separados por comas.",
    ),
) -> None:
    """Compara candidatos separados para productos single y worst_of."""
    settings = Settings()
    with command_status("Ejecutando benchmark segmentado"):
        rfqs, volatility, reference = validated_data(settings)
        detail = run_segmented_benchmark(
            trainable_rfqs(rfqs),
            volatility,
            reference,
            settings,
            model_names=parse_model_names(models),
        )
    print_dataframe(
        detail,
        title="Benchmark segmentado",
        columns=[
            "segment",
            "model_name",
            "feature_block",
            "validation_mae",
            "validation_rmse",
            "validation_r2",
            "fit_seconds",
        ],
        max_rows=20,
    )


def segmented_rolling_benchmark(
    models: str | None = typer.Option(
        None,
        help="Modelos segmentados separados por comas.",
    ),
) -> None:
    """Combina validación temporal rolling y segmentación por tipo de cesta."""
    settings = Settings()
    with command_status("Ejecutando benchmark segmentado rolling"):
        rfqs, volatility, reference = validated_data(settings)
        summary = run_segmented_rolling_benchmark(
            trainable_rfqs(rfqs),
            volatility,
            reference,
            settings,
            model_names=parse_model_names(models),
        )
    print_dataframe(
        summary,
        title="Benchmark segmentado rolling",
        columns=[
            "segment",
            "model_name",
            "feature_block",
            "rolling_mae_mean",
            "rolling_mae_std",
            "rolling_mae_max",
            "n_folds",
        ],
        max_rows=20,
    )


def global_stable_experiment(
    n_trials: int = typer.Option(20, help="Trials de Optuna por modelo global."),
    top_n: int = typer.Option(2, help="Candidatos globales que se ajustarán."),
    timeout_seconds_per_model: int = typer.Option(
        300, help="Presupuesto máximo de Optuna por modelo, en segundos."
    ),
    models: str | None = typer.Option(
        None,
        help="Modelos globales estables separados por comas.",
    ),
) -> None:
    """Compara y ajusta con Optuna los candidatos globales más estables."""
    settings = Settings()
    with command_status("Ejecutando experimento global estable y tuning"):
        rfqs, volatility, reference = validated_data(settings)
        model_names = parse_model_names(models)
        benchmark_path = settings.metrics_dir / "global_stable_benchmark.csv"
        benchmark_detail = (
            pd.read_csv(benchmark_path).sort_values("validation_mae")
            if benchmark_path.exists() and model_names is None
            else run_global_stable_experiment(
                trainable_rfqs(rfqs),
                volatility,
                reference,
                settings,
                model_names=model_names,
            )
        )
        tuning_summary = tune_global_stable_models(
            settings,
            n_trials=n_trials,
            top_n=top_n,
            model_names=model_names,
            timeout_seconds_per_model=timeout_seconds_per_model,
        )
    print_dataframe(benchmark_detail, title="Candidatos globales estables", max_rows=5)
    print_json_panel(
        "Salidas del experimento",
        {
            "benchmark_path": settings.metrics_dir / "global_stable_benchmark.csv",
            "tuning_path": tuning_summary["comparison_path"],
        },
    )


def error_analysis(
    top_n: int = typer.Option(10, help="Candidatos mejor clasificados que se analizarán."),
    models: str | None = typer.Option(
        None,
        help="Modelos separados por comas en lugar de usar el ranking de validation.",
    ),
) -> None:
    """Desglosa los errores de candidatos por año, producto, cesta y subyacente."""
    settings = Settings()
    with command_status("Analizando errores por slices"):
        rfqs, volatility, reference = validated_data(settings)
        outputs = run_robustness_report(
            trainable_rfqs(rfqs),
            volatility,
            reference,
            settings,
            model_names=parse_model_names(models),
            top_n=top_n,
        )
    print_dataframe(outputs["summary"], title="Análisis de errores", max_rows=20)


def tune(
    n_trials: int = typer.Option(25, help="Trials de Optuna por modelo."),
    top_n: int = typer.Option(4, help="Candidatos del ranking que se ajustarán."),
    timeout_seconds_per_model: int = typer.Option(
        300, help="Presupuesto máximo de Optuna por modelo, en segundos."
    ),
    models: str | None = typer.Option(
        None,
        help="Modelos separados por comas en lugar de usar el ranking del benchmark.",
    ),
) -> None:
    """Ajusta con Optuna candidatos globales seleccionados por validation."""
    with command_status("Ajustando candidatos globales con Optuna"):
        summary = tune_top_models(
            Settings(),
            n_trials=n_trials,
            top_n=top_n,
            model_names=parse_model_names(models),
            timeout_seconds_per_model=timeout_seconds_per_model,
        )
    print_json_panel("Resumen del tuning global", summary)


def tune_segmented(
    n_trials: int = typer.Option(20, help="Trials de Optuna por modelo segmentado."),
    timeout_seconds_per_model: int = typer.Option(
        300, help="Presupuesto máximo de Optuna por modelo, en segundos."
    ),
    models: str | None = typer.Option(
        None,
        help="Modelos segmentados separados por comas.",
    ),
) -> None:
    """Ajusta con Optuna modelos definidos para cada segmento."""
    with command_status("Ajustando candidatos segmentados con Optuna"):
        summary = tune_segmented_models(
            Settings(),
            n_trials=n_trials,
            model_names=parse_model_names(models),
            timeout_seconds_per_model=timeout_seconds_per_model,
        )
    print_json_panel("Resumen del tuning segmentado", summary)


def prepare_segmented_selection(
    n_trials: int = typer.Option(20, help="Trials de Optuna por modelo segmentado seleccionado."),
    top_n_per_segment: int = typer.Option(
        2,
        help="Candidatos del benchmark que se ajustarán por segmento.",
    ),
    ranking_metric: str = typer.Option(
        "validation_mae",
        help="Métrica del benchmark usada para seleccionar candidatos.",
    ),
    models: str | None = typer.Option(
        None,
        help="Modelos segmentados separados por comas en lugar del ranking.",
    ),
) -> None:
    """Selecciona y ajusta los mejores candidatos de cada segmento."""
    with command_status("Seleccionando y ajustando modelos por segmento"):
        summary = train_segmented_with_optuna(
            Settings(),
            n_trials=n_trials,
            top_n_per_segment=top_n_per_segment,
            ranking_metric=ranking_metric,
            model_names=parse_model_names(models),
        )
    print_json_panel("Selección de modelos segmentados", summary)


def compare_serving_strategies() -> None:
    """Compara estadísticamente una estrategia global con una segmentada."""
    with command_status("Comparando estrategias de serving"):
        summary = write_model_selection_report(Settings())
    print_json_panel("Comparación de estrategias", summary)


def register(app: typer.Typer) -> None:
    """Perform register."""
    panel = "Experimentos"
    app.command(rich_help_panel=panel)(benchmark)
    app.command("rolling-benchmark", rich_help_panel=panel)(rolling_benchmark)
    app.command("segmented-benchmark", rich_help_panel=panel)(segmented_benchmark)
    app.command("segmented-rolling-benchmark", rich_help_panel=panel)(segmented_rolling_benchmark)
    app.command("global-stable-experiment", rich_help_panel=panel)(global_stable_experiment)
    app.command("error-analysis", rich_help_panel=panel)(error_analysis)
    app.command(rich_help_panel=panel)(tune)
    app.command("tune-segmented", rich_help_panel=panel)(tune_segmented)
    app.command("select-segmented-models", rich_help_panel=panel)(prepare_segmented_selection)
    app.command("compare-serving-strategies", rich_help_panel=panel)(compare_serving_strategies)
