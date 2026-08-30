"""End-to-end workflow commands."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass

import typer
from rich.console import Console

from starwars_autocalls.cli.common import print_success
from starwars_autocalls.config import Settings
from starwars_autocalls.modeling.benchmark import candidate_specs, run_type_for_spec
from starwars_autocalls.modeling.specs import global_stable_specs


@dataclass(frozen=True)
class SuiteStep:
    """One command in the complete reproducibility suite."""

    name: str
    arguments: tuple[str, ...]


def _full_suite_steps(settings: Settings) -> list[SuiteStep]:
    benchmark_models = ",".join(
        spec.name for spec in candidate_specs() if run_type_for_spec(spec) != "baseline"
    )
    stable_models = ",".join(spec.name for spec in global_stable_specs())
    final_config = json.loads(settings.final_model_config_path.read_text(encoding="utf-8"))
    explain_model = str(final_config["base_model_name"])
    return [
        SuiteStep("Validación de datos", ("validate-data",)),
        SuiteStep("EDA", ("eda",)),
        SuiteStep("Auditoría temporal", ("split-audit",)),
        SuiteStep("Auditoría de features", ("feature-audit",)),
        SuiteStep("Benchmark global", ("benchmark", "--models", benchmark_models)),
        SuiteStep("Benchmark rolling", ("rolling-benchmark",)),
        SuiteStep("Benchmark segmentado", ("segmented-benchmark",)),
        SuiteStep("Benchmark segmentado rolling", ("segmented-rolling-benchmark",)),
        SuiteStep(
            "Experimento global estable",
            ("global-stable-experiment", "--models", stable_models),
        ),
        SuiteStep("Tuning global", ("tune",)),
        SuiteStep("Tuning segmentado", ("tune-segmented",)),
        SuiteStep("Selección segmentada", ("select-segmented-models",)),
        SuiteStep("Comparación de estrategias", ("compare-serving-strategies",)),
        SuiteStep("Análisis de errores", ("error-analysis",)),
        SuiteStep("Resumen experimental", ("experiment-summary",)),
        SuiteStep("Interpretabilidad", ("explain", "--model-name", explain_model)),
        SuiteStep("Entrenamiento final", ("train",)),
        SuiteStep("Evaluación final", ("evaluate",)),
    ]


def full_suite(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Muestra el recorrido completo sin ejecutarlo.",
    ),
) -> None:
    """Ejecuta toda la suite reproducible excepto los modelos baseline."""
    settings = Settings()
    steps = _full_suite_steps(settings)
    console = Console()
    for index, step in enumerate(steps, start=1):
        command = [sys.executable, "-m", "starwars_autocalls.cli", *step.arguments]
        console.rule(f"{index}/{len(steps)} · {step.name}")
        console.print(" ".join(command))
        if dry_run:
            continue
        result = subprocess.run(command, cwd=settings.project_root, check=False)
        if result.returncode:
            raise typer.Exit(result.returncode)
    print_success("Suite completa finalizada" if not dry_run else "Suite completa validada")


def register(app: typer.Typer) -> None:
    """Register workflow commands."""
    app.command("full-suite", rich_help_panel="Experimentos")(full_suite)
