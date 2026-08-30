"""Common module."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text

from starwars_autocalls.config import Settings
from starwars_autocalls.data.loading import load_all
from starwars_autocalls.data.validation import validate_all
from starwars_autocalls.observability.progress import progress_reporting


def validated_data(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform validated data."""
    rfqs, volatility, reference = load_all(settings)
    return validate_all(rfqs, volatility, reference)


def parse_model_names(models: str | None) -> list[str] | None:
    """Return parse model names."""
    if not models:
        return None
    names = [name.strip() for name in models.split(",") if name.strip()]
    return names or None


console = Console()


@contextmanager
def command_status(message: str) -> Iterator[Status]:
    """Handle command status."""

    def render_progress(progress_message: str) -> None:
        """Handle render progress."""
        console.print(Text(f"  → {progress_message}", style="cyan"))

    with progress_reporting(render_progress):
        with console.status(message, spinner="dots") as status:
            yield status


def print_json_panel(title: str, payload: Any) -> None:
    """Handle print json panel."""
    rendered = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    console.print(Panel(rendered, title=title, border_style="green"))


def print_paths(title: str, paths: dict[str, str | Path]) -> None:
    """Handle print paths."""
    table = Table(title=title, box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("Salida")
    table.add_column("Ruta", overflow="fold")
    for name, path in paths.items():
        table.add_row(name, str(path))
    console.print(table)


def print_dataframe(
    frame: pd.DataFrame,
    *,
    title: str,
    columns: list[str] | None = None,
    max_rows: int = 10,
) -> None:
    """Handle print dataframe."""
    if frame.empty:
        console.print(Panel("No hay resultados disponibles.", title=title, border_style="yellow"))
        return
    selected = frame.loc[:, [column for column in (columns or list(frame)) if column in frame]]
    selected = selected.head(max_rows)
    table = Table(title=title, box=box.SIMPLE_HEAVY, header_style="bold cyan")
    for column in selected.columns:
        table.add_column(str(column), overflow="fold")
    for row in selected.itertuples(index=False, name=None):
        table.add_row(*[_format_value(value) for value in row])
    console.print(table)


def print_success(message: str) -> None:
    """Handle print success."""
    console.print(Text(f"OK  {message}", style="bold green"))


def _format_value(value: Any) -> str:
    """Handle format value."""
    if pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
