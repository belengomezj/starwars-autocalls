from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from starwars_autocalls import __version__
from starwars_autocalls.api.main import app
from starwars_autocalls.data.validation import (
    validate_basket_structure,
    validate_underlying_membership,
)
from starwars_autocalls.features import (
    FeatureBuilder,
    canonical_basket_signature,
    clean_observation_frequency,
    normalize_observation_frequency,
    parse_underlyings,
    select_feature_block,
)
from starwars_autocalls.modeling.evaluation import (
    regression_metrics,
    rolling_temporal_folds,
    temporal_split,
)
from starwars_autocalls.modeling.specs import (
    default_model_specs,
    global_stable_specs,
    segmented_model_specs,
)
from starwars_autocalls.modeling.training import _top_segmented_candidate_names
from starwars_autocalls.reports.diagnostics import (
    _decision_modeling_frame,
    _feature_block_diff_summary,
    _feature_correlation_stability,
    _feature_drift,
    _target_distribution,
)
from starwars_autocalls.reports.eda import discover_raw_csv_tables
from starwars_autocalls.reports.temporal_audit import write_split_audit


def _assert_feature_block_contents(
    block_name: str,
    *,
    numeric_features: Iterable[str] = (),
    missing_numeric_features: Iterable[str] = (),
    categorical_features: Iterable[str] = (),
    missing_categorical_features: Iterable[str] = (),
) -> None:
    numeric, categorical = select_feature_block(block_name)

    for feature in numeric_features:
        assert feature in numeric, f"{feature!r} should be numeric in {block_name!r}"
    for feature in missing_numeric_features:
        assert feature not in numeric, f"{feature!r} should not be numeric in {block_name!r}"
    for feature in categorical_features:
        assert feature in categorical, f"{feature!r} should be categorical in {block_name!r}"
    for feature in missing_categorical_features:
        assert feature not in categorical, (
            f"{feature!r} should not be categorical in {block_name!r}"
        )


def test_version() -> None:
    assert __version__ == "0.1.0"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1M", 1.0),
        ("Monthly", 1.0),
        ("mensual", 1.0),
        ("M", 1.0),
        ("1 month", 1.0),
        ("2M", 2.0),
        ("3M", 3.0),
        ("Quarterly", 3.0),
        ("trimestral", 3.0),
        ("Q", 3.0),
        ("6M", 6.0),
        ("1Y", 12.0),
        ("12M", 12.0),
        ("Annual", 12.0),
        ("anual", 12.0),
    ],
)
def test_observation_frequency_normalization(value: str, expected: float) -> None:
    assert normalize_observation_frequency(value) == expected


def test_daily_frequency_is_explicit() -> None:
    assert normalize_observation_frequency("1D") == pytest.approx(1 / 21)
    assert clean_observation_frequency("1D") == "daily"


def test_parse_underlyings() -> None:
    assert parse_underlyings("KYBR| CORL |") == ["KYBR", "CORL"]


def test_canonical_basket_signature() -> None:
    assert canonical_basket_signature(["CORL", "KYBR"]) == "CORL|KYBR"
    assert canonical_basket_signature(["KYBR", "CORL"]) == "CORL|KYBR"


def test_underlying_membership_validation() -> None:
    rfqs = pd.DataFrame({"underlyings": ["AAA|BBB"]})
    reference = pd.DataFrame({"underlying": ["AAA"]})
    with pytest.raises(ValueError):
        validate_underlying_membership(rfqs, reference)


def test_basket_structure_validation() -> None:
    valid = pd.DataFrame(
        {
            "rfq_id": ["s", "w"],
            "basket_type": ["single", "worst_of"],
            "underlyings": ["AAA", "AAA|BBB"],
        }
    )
    validate_basket_structure(valid)

    invalid = pd.DataFrame(
        {
            "rfq_id": ["bad"],
            "basket_type": ["single"],
            "underlyings": ["AAA|BBB"],
        }
    )
    with pytest.raises(ValueError, match="inconsistent"):
        validate_basket_structure(invalid)

    duplicate = pd.DataFrame(
        {
            "rfq_id": ["dup"],
            "basket_type": ["worst_of"],
            "underlyings": ["AAA|AAA"],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_basket_structure(duplicate)


def test_as_of_volatility_join_uses_latest_non_future_observation() -> None:
    rfqs = pd.DataFrame(
        {
            "product_type": ["P"],
            "underlyings": ["AAA"],
            "basket_type": ["single"],
            "autocall_barrier_pct": [1.0],
            "protection_barrier_pct": [0.6],
            "no_call_period_months": [3],
            "observation_frequency": ["1M"],
            "quoted_implied_vol": [0.3],
            "notional_credits": [100000],
            "counterparty": ["C"],
            "trader_id": ["T"],
            "requested_date": [pd.Timestamp("2024-01-15")],
            "start_date": [pd.Timestamp("2024-01-15")],
            "end_date": [pd.Timestamp("2029-01-15")],
            "avg_duration_months": [24.0],
        }
    )
    volatility = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-20"]),
            "underlying": ["AAA", "AAA"],
            "realized_vol_63d": [0.2, 0.9],
        }
    )
    reference = pd.DataFrame(
        {"underlying": ["AAA"], "sector": ["Sector"], "structural_base_vol": [0.25]}
    )
    features = FeatureBuilder().build(rfqs, volatility, reference)
    row = features.frame.loc[0]
    assert row["realized_vol_63d_mean"] == 0.2
    assert row["basket_realized_vol_21d_mean"] == pytest.approx(0.2)
    assert row["basket_realized_vol_126d_mean"] == pytest.approx(0.2)
    assert row["basket_realized_vol_trend_mean"] == pytest.approx(0.0)
    assert row["barrier_gap_pct"] == pytest.approx(0.4)
    assert row["autocall_barrier_above_par_pct"] == pytest.approx(0.0)
    assert row["protection_buffer_pct"] == pytest.approx(0.4)
    assert row["autocall_to_protection_ratio"] == pytest.approx(1.0 / 0.6)
    assert row["callable_maturity_months"] < row["nominal_maturity_months"]
    assert row["no_call_observation_count"] == pytest.approx(3.0)
    assert row["is_single_underlying"] == 1
    assert row["is_worst_of"] == 0
    assert row["basket_complexity_score"] == 0
    assert row["worst_of_pressure"] == 0
    assert row["worst_of_vol_pressure"] == 0
    assert row["sector_concentration"] == pytest.approx(1.0)
    assert row["has_multiple_sectors"] == 0
    assert row["primary_underlying"] == "AAA"
    assert row["basket_signature"] == "AAA"
    assert row["highest_vol_sector"] == "Sector"
    assert row["highest_structural_vol_underlying"] == "AAA"
    assert row["second_highest_structural_vol_underlying"] == "UNKNOWN"
    assert row["lowest_structural_vol_underlying"] == "AAA"
    assert row["highest_realized_vol_underlying"] == "AAA"
    assert row["second_highest_realized_vol_underlying"] == "UNKNOWN"
    assert row["lowest_realized_vol_underlying"] == "AAA"
    assert row["log_notional_credits"] > 0
    assert row["quoted_minus_structural_vol_mean"] == pytest.approx(0.05)
    assert row["realized_minus_structural_vol_mean"] == pytest.approx(-0.05)
    assert row["realized_to_structural_vol_ratio"] == pytest.approx(0.8)
    assert row["quoted_to_realized_vol_ratio"] == pytest.approx(1.5)
    assert row["quoted_to_structural_vol_ratio"] == pytest.approx(1.2)
    assert row["realized_vol_dispersion_ratio"] == pytest.approx(0.0)
    assert row["structural_vol_dispersion_ratio"] == pytest.approx(0.0)
    assert row["realized_vol_63d_top2_spread"] == pytest.approx(0.0)
    assert row["structural_base_vol_top2_spread"] == pytest.approx(0.0)
    assert row["estimated_observation_count"] > row["post_no_call_observation_count"]
    assert row["is_daily_observation"] == 0
    assert "executed" not in features.frame.columns
    assert "avg_duration_months" not in features.frame.columns
    assert "start_date" not in features.frame.columns
    assert "end_date" not in features.frame.columns
    assert "requested_date" not in features.frame.columns
    assert "nominal_maturity_months" in features.frame.columns


def test_temporal_split() -> None:
    frame = pd.DataFrame(
        {"requested_date": pd.to_datetime(["2021-01-01", "2022-01-01", "2023-01-01"])}
    )
    split = temporal_split(frame)
    assert list(split.train_index) == [0]
    assert list(split.validation_index) == [1]
    assert list(split.test_index) == [2]


def test_feature_block_selection() -> None:
    _assert_feature_block_contents(
        "all_without_commercial",
        numeric_features=[
            "quoted_to_realized_vol_ratio",
            "barrier_gap_pct",
            "protection_buffer_pct",
            "structural_vol_dispersion_ratio",
            "underlying_KYBR",
            "pair_CORL_KYBR",
        ],
        missing_numeric_features=["avg_duration_months"],
        categorical_features=["basket_signature"],
        missing_categorical_features=["counterparty"],
    )
    _assert_feature_block_contents(
        "all_without_noise",
        missing_numeric_features=["requested_month", "notional_credits"],
        missing_categorical_features=["counterparty"],
    )


def test_segmented_candidate_ranking_rejects_test_metrics() -> None:
    detail = pd.DataFrame(
        {
            "segment": ["single"],
            "model_name": ["candidate"],
            "validation_mae": [1.0],
            "test_mae": [0.5],
        }
    )
    with pytest.raises(ValueError, match="validation metrics"):
        _top_segmented_candidate_names(detail, top_n_per_segment=1, ranking_metric="test_mae")


def test_feature_block_groups() -> None:
    _assert_feature_block_contents(
        "contractual",
        numeric_features=["estimated_observation_count", "callable_maturity_months"],
    )
    _assert_feature_block_contents(
        "basket",
        numeric_features=["basket_complexity_score", "worst_of_vol_pressure"],
        categorical_features=[
            "dominant_sector",
            "highest_vol_sector",
            "primary_underlying",
            "highest_structural_vol_underlying",
            "highest_realized_vol_underlying",
        ],
    )
    _assert_feature_block_contents(
        "market",
        numeric_features=[
            "realized_to_structural_vol_ratio",
            "quoted_minus_structural_vol_mean",
            "basket_realized_vol_zscore_mean",
        ],
    )
    _assert_feature_block_contents(
        "single_without_noise",
        numeric_features=["structural_base_vol_mean"],
        missing_numeric_features=["basket_size"],
        categorical_features=["dominant_sector"],
    )
    _assert_feature_block_contents(
        "single_stable",
        missing_numeric_features=["basket_realized_vol_trend_mean"],
    )
    _assert_feature_block_contents(
        "single_underlying",
        categorical_features=["primary_underlying"],
    )
    _assert_feature_block_contents(
        "worst_of_without_noise",
        numeric_features=["basket_size", "structural_base_vol_range"],
        categorical_features=["highest_vol_sector"],
    )
    _assert_feature_block_contents(
        "worst_of_stable",
        missing_numeric_features=["realized_to_structural_vol_ratio"],
    )
    _assert_feature_block_contents(
        "worst_of_tail_focus",
        numeric_features=["basket_realized_vol_21d_max"],
        missing_numeric_features=["basket_realized_vol_21d_mean"],
    )
    _assert_feature_block_contents(
        "worst_of_risk_underlying",
        categorical_features=[
            "highest_structural_vol_underlying",
            "highest_realized_vol_underlying",
        ],
    )


def test_business_baseline_v2_specs() -> None:
    names = {spec.name for spec in default_model_specs()}

    assert "median_by_basket_type" in names
    assert "median_by_product_frequency" in names
    assert "median_by_product_maturity_bucket" in names
    assert "median_by_product_frequency_maturity" in names
    assert "median_by_product_basket_size" in names
    assert "median_by_single_worstof" in names
    assert "median_by_product_single_worstof" in names


def test_segmented_model_specs_are_segment_specific() -> None:
    specs = segmented_model_specs()

    assert set(specs) == {"single", "worst_of"}
    assert all("single_" in spec.name for spec in specs["single"])
    assert all(spec.feature_block.startswith("single_") for spec in specs["single"])
    assert all("worst_of_" in spec.name for spec in specs["worst_of"])
    assert all(spec.feature_block.startswith("worst_of_") for spec in specs["worst_of"])


def test_global_stable_feature_blocks_are_diagnostics_driven() -> None:
    _assert_feature_block_contents(
        "global_stable",
        numeric_features=["is_worst_of"],
        missing_numeric_features=[
            "requested_year",
            "basket_realized_vol_trend_mean",
            "realized_to_structural_vol_ratio",
        ],
        categorical_features=["product_type"],
    )
    _assert_feature_block_contents(
        "global_stable_tail",
        numeric_features=["realized_vol_63d_max"],
        missing_numeric_features=["realized_vol_63d_mean"],
    )
    _assert_feature_block_contents(
        "global_stable_no_sector",
        missing_categorical_features=["dominant_sector"],
    )
    _assert_feature_block_contents(
        "global_risk_underlying",
        categorical_features=[
            "highest_structural_vol_underlying",
            "highest_realized_vol_underlying",
        ],
    )


def test_global_stable_model_specs_are_global_candidates() -> None:
    specs = global_stable_specs()

    assert specs
    assert all("_global_" in spec.name for spec in specs)
    assert {spec.feature_block for spec in specs} == {
        "global_stable",
        "global_stable_tail",
        "global_stable_no_sector",
        "global_risk_underlying",
        "global_all_underlying",
        "global_tail_underlying",
    }


def test_feature_diagnostics_block_diff_summary() -> None:
    summary = _feature_block_diff_summary()

    single_diff = summary.loc[
        (summary["from_block"] == "single_core") & (summary["to_block"] == "single_without_noise")
    ].iloc[0]
    assert "requested_year" in single_diff["removed_features"]
    assert single_diff["added_count"] == 0
    stable_diff = summary.loc[
        (summary["from_block"] == "worst_of_without_noise")
        & (summary["to_block"] == "worst_of_stable")
    ].iloc[0]
    assert "realized_to_structural_vol_ratio" in stable_diff["removed_features"]
    global_diff = summary.loc[
        (summary["from_block"] == "all_without_noise") & (summary["to_block"] == "global_stable")
    ].iloc[0]
    assert "requested_year" in global_diff["removed_features"]


def test_feature_diagnostics_target_distribution() -> None:
    frame = pd.DataFrame(
        {
            "segment": ["single", "single", "worst_of"],
            "split": ["train", "test", "test"],
            "target": [12.0, 72.0, 84.0],
        }
    )

    distribution = _target_distribution(frame, ["segment", "split"])

    single_test = distribution.loc[
        (distribution["segment"] == "single") & (distribution["split"] == "test")
    ].iloc[0]
    assert single_test["rows"] == 1
    assert single_test["duration_60_plus_rate"] == pytest.approx(1.0)


def test_feature_decision_diagnostics_exclude_test_split() -> None:
    frame = pd.DataFrame(
        {
            "segment": ["single", "single", "single", "worst_of"],
            "split": ["train", "validation", "test", "test"],
            "target": [10.0, 12.0, 80.0, 90.0],
        }
    )
    X = pd.DataFrame(
        {
            "signal": [1.0, 2.0, 100.0, 200.0],
            "is_single_underlying": [1, 1, 1, 0],
            "is_worst_of": [0, 0, 0, 1],
        }
    )
    y = frame["target"]

    decision_frame = _decision_modeling_frame(frame)
    drift = _feature_drift(X, y, decision_frame)
    stability = _feature_correlation_stability(X, y, decision_frame)

    assert set(decision_frame["split"]) == {"train", "validation"}
    assert "test_mean" not in drift.columns
    assert "test_target_corr" not in drift.columns
    assert "test_sign_flip_vs_train" not in stability.columns


def test_top_segmented_candidate_names_selects_per_segment() -> None:
    detail = pd.DataFrame(
        {
            "segment": ["single", "single", "worst_of", "worst_of"],
            "model_name": ["s_a", "s_b", "w_a", "w_b"],
            "validation_mae": [2.0, 1.0, 4.0, 3.0],
        }
    )

    assert _top_segmented_candidate_names(detail, 1, "validation_mae") == ["s_b", "w_b"]


def test_rolling_temporal_folds() -> None:
    frame = pd.DataFrame(
        {"requested_date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"])}
    )
    folds = rolling_temporal_folds(frame, min_train_years=2)
    assert len(folds) == 1
    assert folds[0].description == "train<=2021, validation=2022"

    all_folds = rolling_temporal_folds(frame, min_train_years=2, max_validation_year=None)
    assert len(all_folds) == 2


def test_regression_metrics() -> None:
    metrics = regression_metrics(pd.Series([1.0, 2.0]), pd.Series([1.5, 1.5]))
    assert metrics["mae"] == pytest.approx(0.5)


def test_health_endpoint() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_discover_raw_csv_tables(tmp_path) -> None:
    tmp_path.joinpath("usable.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    tmp_path.joinpath("one_column.csv").write_text("a\n1\n", encoding="utf-8")
    tmp_path.joinpath("notes.txt").write_text("ignore", encoding="utf-8")

    tables = discover_raw_csv_tables(tmp_path)

    assert [table.name for table in tables] == ["usable"]


def test_split_audit_outputs_exclude_test_by_default(tmp_path) -> None:
    rfqs = pd.DataFrame(
        {
            "rfq_id": ["r1", "r2", "r3", "r4", "r5"],
            "product_type": ["P1", "P2", "P1", "P2", "P1"],
            "underlyings": ["AAA", "AAA|BBB", "BBB", "AAA|BBB", "AAA"],
            "basket_type": ["single", "worst_of", "single", "worst_of", "single"],
            "autocall_barrier_pct": [1.0, 1.1, 1.0, 1.1, 1.0],
            "protection_barrier_pct": [0.6, 0.6, 0.6, 0.6, 0.6],
            "no_call_period_months": [3, 6, 3, 6, 3],
            "observation_frequency": ["1M", "3M", "1M", "3M", "1M"],
            "quoted_implied_vol": [0.25, 0.3, 0.25, 0.3, 0.25],
            "notional_credits": [1000.0, 2000.0, 1000.0, 2000.0, 1000.0],
            "counterparty": ["C"] * 5,
            "trader_id": ["T"] * 5,
            "requested_date": pd.to_datetime(
                ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"]
            ),
            "executed": [True, True, True, True, False],
            "start_date": pd.to_datetime(
                ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"]
            ),
            "end_date": pd.to_datetime(
                ["2025-01-01", "2026-01-01", "2027-01-01", "2028-01-01", "2029-01-01"]
            ),
            "avg_duration_months": [24.0, 36.0, 30.0, 42.0, None],
        }
    )
    reference = pd.DataFrame(
        {
            "underlying": ["AAA", "BBB"],
            "sector": ["Tech", "Energy"],
            "structural_base_vol": [0.2, 0.3],
        }
    )

    paths = write_split_audit(rfqs, reference, tmp_path)

    assert paths["html"].exists()
    assert paths["summary"].exists()
    assert paths["numeric_drift"].exists()
    assert paths["categorical_drift"].exists()
    html = paths["html"].read_text(encoding="utf-8")
    summary = paths["summary"].read_text(encoding="utf-8")
    assert "Auditoría De Splits" in html
    assert "Split del pipeline" in html
    assert "Auditoría De Rolling Temporal Validation" in html
    assert "El foco principal son RFQs entrenables" not in html
    assert "Criterios mínimos: 1.000 filas de train" not in html
    assert "10. Conclusiones Automáticas" not in html
    assert "Split detectado desde el pipeline" not in html
    assert "<h3>Missingness</h3>" not in html
    assert "Esta sección revisa volumen" not in html
    assert "Comparativa directa entre single-underlying products" not in html
    assert "Esta sección responde si el dataset" not in html
    assert "La rolling temporal validation con ventana expansiva" not in html
    assert "La columna <code>underlyings</code> se explota" not in html
    assert "Distribución global del target: avg_duration_months" not in html
    assert '"supervised_rows": 3' in summary
    assert '"includes_test": false' in summary
    categorical_counts = pd.read_csv(paths["categorical_temporal_counts"])
    assert set(categorical_counts["split"]) == {"train", "validation"}
