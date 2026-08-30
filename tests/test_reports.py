from __future__ import annotations

import json

import pandas as pd
import pytest

from starwars_autocalls.config import Settings
from starwars_autocalls.modeling.benchmark import run_robustness_report
from starwars_autocalls.reports.categorical_analysis import (
    categorical_summary,
    unseen_categories_report,
)
from starwars_autocalls.reports.eda import _executed_rfqs_for_sweetviz
from starwars_autocalls.reports.experiment_summary import write_experiment_summary


def test_sweetviz_rfqs_use_all_and_only_explicitly_executed_rows() -> None:
    rfqs = pd.DataFrame(
        {
            "rfq_id": [
                "train",
                "validation",
                "test",
                "executed-without-target",
                "not-executed",
                "missing-executed",
            ],
            "executed": pd.Series([True, True, True, True, False, pd.NA], dtype="boolean"),
            "avg_duration_months": [10.0, 20.0, 30.0, pd.NA, pd.NA, pd.NA],
        }
    )

    result = _executed_rfqs_for_sweetviz(rfqs)

    assert result["rfq_id"].tolist() == [
        "train",
        "validation",
        "test",
        "executed-without-target",
    ]


def test_error_analysis_reuses_cached_benchmark_predictions(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    settings.metrics_dir.mkdir(parents=True)
    rfqs = pd.DataFrame(
        {
            "rfq_id": ["A", "B", "C", "D", "E", "F"],
            "requested_date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2021-01-01",
                    "2022-01-01",
                    "2022-06-01",
                    "2023-01-01",
                    "2024-01-01",
                ]
            ),
            "avg_duration_months": [10.0, 12.0, 20.0, 30.0, 40.0, 50.0],
            "basket_type": ["single"] * 6,
            "product_type": ["P1"] * 6,
            "underlyings": ["AAA"] * 6,
        }
    )
    pd.DataFrame([{"model_name": "global_mean", "validation_mae": 1.5}]).to_csv(
        settings.metrics_dir / "benchmark_comparison.csv", index=False
    )
    pd.DataFrame(
        {
            "model_name": ["global_mean", "global_mean"],
            "row_index": [2, 3],
            "rfq_id": ["C", "D"],
            "actual": [20.0, 30.0],
            "prediction": [22.0, 29.0],
        }
    ).to_csv(settings.metrics_dir / "benchmark_validation_predictions.csv", index=False)

    outputs = run_robustness_report(
        rfqs,
        pd.DataFrame(),
        pd.DataFrame(),
        settings,
        model_names=["global_mean"],
    )

    assert outputs["summary"].iloc[0]["validation_mae"] == pytest.approx(1.5)
    assert (settings.metrics_dir / "robustness_segments.csv").exists()


def test_categorical_summary_does_not_weight_rfqs_by_basket_size() -> None:
    rfqs = pd.DataFrame(
        {
            "product_type": ["single_product", "basket_product"],
            "basket_type": ["single", "worst_of"],
            "observation_frequency": ["1M", "1M"],
            "counterparty": ["A", "B"],
            "trader_id": ["T1", "T2"],
            "underlyings": ["AAA", "AAA|BBB|CCC"],
            "avg_duration_months": [12.0, 36.0],
            "requested_date": pd.to_datetime(["2021-01-01", "2022-01-01"]),
        }
    )
    reference = pd.DataFrame(
        {
            "underlying": ["AAA", "BBB", "CCC"],
            "sector": ["Tech", "Energy", "Health"],
        }
    )

    summary = categorical_summary(rfqs, reference).set_index("feature")

    assert summary.loc["product_type", "top_category_count"] == 1
    assert summary.loc["product_type", "observation_level"] == "rfq"
    assert summary.loc["underlying", "top_category_count"] == 2
    assert summary.loc["underlying", "observation_level"] == "underlying"


def test_unseen_categories_reports_affected_share_without_test() -> None:
    rfqs = pd.DataFrame(
        {
            "product_type": ["P1", "P1", "P2", "P2"],
            "basket_type": ["single"] * 4,
            "observation_frequency": ["1M"] * 4,
            "counterparty": ["A", "A", "B", "B"],
            "trader_id": ["T1", "T1", "T2", "T2"],
            "underlyings": ["AAA", "AAA", "BBB", "BBB"],
            "avg_duration_months": [12.0, 13.0, 14.0, 15.0],
            "requested_date": pd.to_datetime(
                ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"]
            ),
        }
    )
    reference = pd.DataFrame({"underlying": ["AAA", "BBB"], "sector": ["Tech", "Energy"]})

    report = unseen_categories_report(rfqs, reference)
    product = report.loc[report["feature"] == "product_type"].iloc[0]

    assert set(report["split"]) == {"validation"}
    assert product["unseen_categories"] == 1
    assert product["affected_row_share"] == pytest.approx(1.0)


def test_experiment_summary_selects_by_validation_mae(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    settings.metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "model_name": "model_b",
                "feature_block": "all_features",
                "encoding_strategy": "ordinal",
                "validation_mae": 4.0,
                "validation_rmse": 5.0,
                "validation_r2": 0.8,
                "fit_seconds": 2.0,
            },
            {
                "model_name": "model_a",
                "feature_block": "all_without_noise",
                "encoding_strategy": "ordinal",
                "validation_mae": 3.0,
                "validation_rmse": 4.0,
                "validation_r2": 0.9,
                "fit_seconds": 1.0,
            },
        ]
    ).to_csv(settings.metrics_dir / "benchmark_comparison.csv", index=False)
    settings.model_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_metadata_path.write_text(
        json.dumps(
            {
                "model_name": "model_a",
                "feature_block": "all_without_noise",
                "encoding_strategy": "ordinal",
                "selected_validation_mae": 3.0,
                "tuning_selection": None,
                "test_metrics": {"mae": 3.5},
            }
        ),
        encoding="utf-8",
    )

    result = write_experiment_summary(settings)

    assert result["best"].iloc[0]["model_name"] == "model_a"
    assert result["best"].iloc[0]["selection_metric"] == "validation_mae"
    assert (settings.metrics_dir / "experiment_summary.csv").exists()
    assert (settings.metrics_dir / "experiment_best_by_protocol.csv").exists()
