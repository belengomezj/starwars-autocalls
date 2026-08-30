"""App module."""

from __future__ import annotations

from typing import Annotated

import typer

from starwars_autocalls.cli.data_commands import register as register_data_commands
from starwars_autocalls.cli.experiment_commands import register as register_experiment_commands
from starwars_autocalls.cli.model_commands import register as register_model_commands
from starwars_autocalls.cli.report_commands import register as register_report_commands
from starwars_autocalls.cli.workflow_commands import register as register_workflow_commands
from starwars_autocalls.config import Settings
from starwars_autocalls.observability import configure_logging

app = typer.Typer(help="CLI de entrenamiento, experimentación e inferencia de Star Wars Autocalls.")


@app.callback()
def configure_runtime(
    log_level: Annotated[
        str | None,
        typer.Option(help="Nivel de log. Usa STARWARS_AUTOCALLS_LOG_LEVEL o INFO por defecto."),
    ] = None,
    log_format: Annotated[
        str | None,
        typer.Option(help="Formato de log: console o json."),
    ] = None,
) -> None:
    """Perform configure runtime."""
    settings = Settings()
    selected_format = log_format or settings.log_format
    if selected_format not in {"console", "json"}:
        raise typer.BadParameter("log-format debe ser 'console' o 'json'.")
    configure_logging(log_level or settings.log_level, selected_format)


register_data_commands(app)
register_report_commands(app)
register_experiment_commands(app)
register_model_commands(app)
register_workflow_commands(app)
