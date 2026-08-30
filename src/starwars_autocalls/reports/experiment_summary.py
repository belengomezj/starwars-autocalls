"""Experiment Summary module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from starwars_autocalls.config import Settings
from starwars_autocalls.reports.diagnostics import _model_experiment_feature_comparison


def write_experiment_summary(settings: Settings) -> dict[str, Any]:
    """Consolidate available experiment outputs without rerunning any model."""
    settings.ensure_output_dirs()
    comparison = _model_experiment_feature_comparison(settings.metrics_dir)
    comparison = _append_global_rolling(comparison, settings.metrics_dir)
    if comparison.empty:
        raise FileNotFoundError(
            "No hay resultados de experimentos. Ejecuta al menos benchmark o rolling-benchmark."
        )

    comparison = _with_selection_metric(comparison)
    comparison = comparison.sort_values(
        ["evaluation_protocol", "segment", "selection_mae", "model_name"],
        na_position="last",
    ).reset_index(drop=True)
    eligible = comparison.dropna(subset=["selection_mae"])
    best_indices = eligible.groupby(["evaluation_protocol", "segment"], dropna=False)[
        "selection_mae"
    ].idxmin()
    best = (
        eligible.loc[best_indices]
        .sort_values(["evaluation_protocol", "segment"])
        .reset_index(drop=True)
    )

    comparison_path = settings.metrics_dir / "experiment_summary.csv"
    best_path = settings.metrics_dir / "experiment_best_by_protocol.csv"
    summary_path = settings.metrics_dir / "experiment_summary.json"
    comparison.to_csv(comparison_path, index=False)
    best.to_csv(best_path, index=False)
    payload = {
        "selection_policy": (
            "Los holdouts se ordenan por validation_mae y los protocolos rolling por "
            "rolling_mae_mean. Test nunca se usa para seleccionar."
        ),
        "available_sources": sorted(comparison["source"].dropna().astype(str).unique()),
        "experiment_rows": len(comparison),
        "best_by_protocol_and_segment": _records_without_nan(best),
        "artifacts": {
            "comparison": str(comparison_path),
            "best": str(best_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return {"comparison": comparison, "best": best, "summary": payload}


def _append_global_rolling(comparison: pd.DataFrame, metrics_dir: Path) -> pd.DataFrame:
    """Handle append global rolling."""
    path = metrics_dir / "rolling_benchmark_summary.csv"
    if not path.exists():
        return comparison
    rolling = pd.read_csv(path)
    rows = []
    for row in rolling.to_dict(orient="records"):
        rows.append(
            {
                "source": "global_rolling",
                "evaluation_protocol": "rolling_yearly",
                "segment": "global",
                "model_name": row.get("model_name", ""),
                "feature_block": row.get("feature_block", ""),
                "encoding_strategy": row.get("encoding_strategy", ""),
                "validation_mae": np.nan,
                "rolling_mae_mean": row.get("rolling_mae_mean", np.nan),
                "rolling_mae_std": row.get("rolling_mae_std", np.nan),
                "rolling_mae_max": row.get("rolling_mae_max", np.nan),
                "rolling_n_folds": row.get("n_folds", np.nan),
            }
        )
    rolling_frame = pd.DataFrame(rows)
    return pd.concat([comparison, rolling_frame], ignore_index=True, sort=False)


def _with_selection_metric(comparison: pd.DataFrame) -> pd.DataFrame:
    """Handle with selection metric."""
    result = comparison.copy()
    result["segment"] = result["segment"].fillna("global").replace({"": "global", "all": "global"})
    holdout = result["evaluation_protocol"].eq("temporal_holdout")
    result["selection_metric"] = np.where(
        holdout,
        "validation_mae",
        "rolling_mae_mean",
    )
    result["selection_mae"] = result["rolling_mae_mean"]
    result.loc[holdout, "selection_mae"] = result.loc[holdout, "validation_mae"]
    return result


def _json_default(value: Any) -> Any:
    """Handle json default."""
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return str(value)


def _records_without_nan(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Handle records without nan."""
    cleaned = frame.astype(object).where(pd.notna(frame), None)
    return cleaned.to_dict(orient="records")
