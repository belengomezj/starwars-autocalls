from __future__ import annotations

import re

import pandas as pd
import pytest
from typer.testing import CliRunner

from starwars_autocalls.cli import report_commands
from starwars_autocalls.cli.app import app
from starwars_autocalls.modeling.benchmark import candidate_specs

runner = CliRunner()


def test_cli_help_is_grouped_and_describes_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Datos y análisis" in result.stdout
    assert "Experimentos" in result.stdout
    assert "Modelo e inferencia" in result.stdout
    assert "error-analysis" in result.stdout
    assert "compare-serving-strategies" in result.stdout
    assert "split-audit" in result.stdout
    assert "feature-audit" in result.stdout
    assert "experiment-summary" in result.stdout
    assert "categorical-analysis" not in result.stdout
    assert "diagnostics" not in result.stdout
    assert "temporal-distribution-audit" not in result.stdout
    assert "robustness-report" not in result.stdout
    assert "selection-report" not in result.stdout


def test_explain_requires_an_explicit_model_selection() -> None:
    result = runner.invoke(app, ["explain"], color=False)

    assert result.exit_code != 0
    stderr = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.stderr)
    assert "--model-name" in stderr
    assert "--models" in stderr


def test_benchmark_rejects_unknown_model_names() -> None:
    with pytest.raises(ValueError, match="Modelos desconocidos"):
        candidate_specs(model_names=["modelo_que_no_existe"])


def test_experiment_summary_renders_best_result_with_rich(monkeypatch) -> None:
    best = pd.DataFrame(
        [
            {
                "evaluation_protocol": "temporal_holdout",
                "segment": "global",
                "model_name": "model_a",
                "feature_block": "all_features",
                "selection_metric": "validation_mae",
                "selection_mae": 3.0,
                "fit_seconds": 1.0,
            }
        ]
    )
    monkeypatch.setattr(
        report_commands,
        "write_experiment_summary",
        lambda _settings: {
            "best": best,
            "summary": {
                "artifacts": {
                    "comparison": "experiment_summary.csv",
                    "best": "experiment_best_by_protocol.csv",
                    "summary": "experiment_summary.json",
                }
            },
        },
    )

    result = runner.invoke(app, ["experiment-summary"])

    assert result.exit_code == 0
    assert "Mejor resultado por protocolo y segmento" in result.stdout
    assert "model_a" in result.stdout
