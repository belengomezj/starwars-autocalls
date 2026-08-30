"""Report Commands module."""

from __future__ import annotations

import json

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from starwars_autocalls.cli.common import (
    command_status,
    parse_model_names,
    print_dataframe,
    print_paths,
    validated_data,
)
from starwars_autocalls.config import Settings
from starwars_autocalls.reports.diagnostics import run_feature_audit
from starwars_autocalls.reports.eda import run_eda
from starwars_autocalls.reports.experiment_summary import write_experiment_summary
from starwars_autocalls.reports.explainability import write_tree_shap_reports
from starwars_autocalls.reports.temporal_audit import write_split_audit


def eda(
    library_reports: bool = typer.Option(
        False,
        "--library-reports/--no-library-reports",
        help="Genera informes Sweetviz/Skrub adicionales.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Muestra el resumen final como JSON para automatizaciones.",
    ),
) -> None:
    """Genera el análisis exploratorio general de los datos."""
    settings = Settings()
    console = Console(stderr=True)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Validando datos de entrada", total=4)
        rfqs, volatility, reference = validated_data(settings)
        progress.advance(task_id)

        def update_progress(message: str) -> None:
            """Handle update progress."""
            progress.update(task_id, description=message)
            progress.advance(task_id)

        result = run_eda(
            rfqs,
            volatility,
            reference,
            settings.eda_overview_dir,
            raw_data_dir=settings.raw_dir,
            write_library_reports=library_reports,
            progress=update_progress,
        )
        progress.update(task_id, description="Análisis EDA completado", completed=4)

    console.print(f"[bold green]✓ Informe principal:[/bold green] {result['main_report']}")
    if result["library_reports"]:
        console.print("[bold green]✓ Reportes complementarios:[/bold green]")
        for path in result["library_reports"]:
            console.print(f"  {path}")
    else:
        console.print("[yellow]No se generaron reportes complementarios.[/yellow]")
    if json_output:
        typer.echo(json.dumps(result["summary"], indent=2))
        return

    summary = result["summary"]
    summary_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    summary_table.add_column("Métrica", style="cyan")
    summary_table.add_column("Valor", justify="right")
    summary_table.add_row("Filas RFQ", f"{summary['rfq_rows']:,}")
    summary_table.add_row("Filas entrenables", f"{summary['trainable_rows']:,}")
    summary_table.add_row("Filas de desarrollo", f"{summary['development_rows']:,}")
    summary_table.add_row("Periodo", f"{summary['date_min']} → {summary['date_max']}")
    summary_table.add_row("Duración media", f"{float(summary['target_mean']):.2f} meses")
    summary_table.add_row("Mediana de duración", f"{float(summary['target_median']):.2f} meses")
    summary_table.add_row("Target ausente", f"{summary['target_missing_rows']:,} filas")
    Console().print(Panel(summary_table, title="Resumen EDA", border_style="green"))


def split_audit(
    include_test: bool = typer.Option(
        False,
        "--include-test",
        help="Incluye el holdout final sólo para una auditoría posterior a la selección.",
    ),
) -> None:
    """Audita cobertura temporal y categórica de los splits de desarrollo."""
    settings = Settings()
    with command_status("Validando datos y auditando splits"):
        rfqs, _, reference = validated_data(settings)
        paths = write_split_audit(
            rfqs,
            reference,
            settings.split_audit_dir,
            include_test=include_test,
        )
    print_paths("Salidas de la auditoría de splits", paths)


def feature_audit() -> None:
    """Audita estabilidad y señal de las features entre train y validation."""
    with command_status("Construyendo y auditando features"):
        result = run_feature_audit(Settings())
    print_paths("Salidas de la auditoría de features", result["paths"])


def experiment_summary() -> None:
    """Consolida los experimentos disponibles sin volver a entrenar modelos."""
    try:
        with command_status("Consolidando resultados de experimentos"):
            result = write_experiment_summary(Settings())
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_dataframe(
        result["best"],
        title="Mejor resultado por protocolo y segmento",
        columns=[
            "evaluation_protocol",
            "segment",
            "model_name",
            "feature_block",
            "selection_metric",
            "selection_mae",
            "test_mae",
            "fit_seconds",
        ],
        max_rows=20,
    )
    print_paths("Resumen de experimentos", result["summary"]["artifacts"])


def explain(
    max_rows: int = typer.Option(1000, help="Máximo de filas de validation usadas por SHAP."),
    model_name: str | None = typer.Option(
        None,
        "--model-name",
        help="Especificación exacta de un modelo que explicar.",
    ),
    models: str | None = typer.Option(
        None,
        "--models",
        help="Modelos separados por comas para comparar explicaciones.",
    ),
) -> None:
    """Genera explicaciones SHAP para modelos de árboles seleccionados."""
    if model_name and models:
        raise typer.BadParameter("Usa --model-name o --models, pero no ambos.")
    if not model_name and not models:
        raise typer.BadParameter("Indica el modelo con --model-name o --models.")
    with command_status("Calculando explicaciones SHAP"):
        paths = write_tree_shap_reports(
            Settings(),
            model_names=[model_name] if model_name else parse_model_names(models),
            max_rows=max_rows,
        )
    print_paths("Salidas SHAP", paths)


def register(app: typer.Typer) -> None:
    """Perform register."""
    panel = "Datos y análisis"
    app.command(rich_help_panel=panel)(eda)
    app.command("split-audit", rich_help_panel=panel)(split_audit)
    app.command("feature-audit", rich_help_panel=panel)(feature_audit)
    app.command("experiment-summary", rich_help_panel="Experimentos")(experiment_summary)
    app.command(rich_help_panel="Modelo e inferencia")(explain)
