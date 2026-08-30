"""Data Commands module."""

from __future__ import annotations

import typer
from rich import box
from rich.table import Table

from starwars_autocalls.cli.common import command_status, console, validated_data
from starwars_autocalls.config import Settings
from starwars_autocalls.data.loading import trainable_rfqs


def validate_data() -> None:
    """Valida los CSV de entrada y muestra el número de filas utilizables."""
    settings = Settings()
    with command_status("Validando los CSV de entrada"):
        rfqs, volatility, reference = validated_data(settings)
    table = Table(title="Datos validados", box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("Dataset")
    table.add_column("Filas", justify="right")
    table.add_row("RFQs", f"{len(rfqs):,}")
    table.add_row("RFQs entrenables", f"{len(trainable_rfqs(rfqs)):,}")
    table.add_row("Volatilidad", f"{len(volatility):,}")
    table.add_row("Referencia", f"{len(reference):,}")
    console.print(table)


def register(app: typer.Typer) -> None:
    """Perform register."""
    app.command("validate-data", rich_help_panel="Datos y análisis")(validate_data)
