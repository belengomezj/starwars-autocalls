"""Temporal Audit module."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

from starwars_autocalls.data.loading import trainable_rfqs
from starwars_autocalls.features import parse_underlyings
from starwars_autocalls.modeling.evaluation import rolling_temporal_folds, temporal_split
from starwars_autocalls.reports.categorical_analysis import write_categorical_analysis
from starwars_autocalls.reports.eda_render import fig_html, html_document, metric, table_html
from starwars_autocalls.reports.split_drift import write_split_drift_tables

VIABILITY_RULES = {
    "min_train_rows": 1000,
    "min_validation_rows": 200,
    "min_years": 3,
    "min_products": 1,
    "min_underlyings": 1,
}

# Thresholds used by `_automatic_conclusions` to flag concentration risk in the
# supervised dataset. Kept alongside VIABILITY_RULES rather than as literals
# inline in the conclusion logic.
CONCENTRATION_THRESHOLDS = {
    "max_top_product_share": 0.35,
    "max_top_year_share": 0.25,
}


def write_split_audit(
    rfqs: pd.DataFrame,
    reference: pd.DataFrame,
    output_dir: Path,
    *,
    include_test: bool = False,
) -> dict[str, Path]:
    """Perform write split audit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    supervised = _prepare_supervised_frame(trainable_rfqs(rfqs))
    supervised = _assign_temporal_split(supervised)
    if not include_test:
        supervised = supervised.loc[supervised["temporal_split"] != "test"].copy()
    exploded = _explode_underlyings(supervised, reference)
    rolling = _rolling_audit(supervised, exploded)
    segment_viability = _segment_viability(supervised, exploded)
    product_model_viability = _product_model_viability(supervised)
    drift_tables, drift_paths = write_split_drift_tables(supervised, output_dir)
    conclusions = _automatic_conclusions(
        supervised,
        exploded,
        rolling,
        segment_viability,
        product_model_viability,
    )
    summary = _summary_payload(
        rfqs,
        supervised,
        exploded,
        rolling,
        segment_viability,
        product_model_viability,
        conclusions,
    )

    summary["includes_test"] = include_test
    summary["drift_status_counts"] = {
        name: table["status"].value_counts().to_dict()
        for name, table in drift_tables.items()
        if not table.empty
    }
    html_path = output_dir / "split_audit.html"
    json_path = output_dir / "split_audit_summary.json"
    html_path.write_text(
        _render_html_report(
            rfqs,
            supervised,
            exploded,
            reference,
            rolling,
            segment_viability,
            product_model_viability,
            drift_tables,
        ),
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    categorical_paths = write_categorical_analysis(
        trainable_rfqs(rfqs),
        reference,
        output_dir,
        include_test=include_test,
    )
    categorical_paths = {f"categorical_{name}": path for name, path in categorical_paths.items()}
    return {
        "html": html_path,
        "summary": json_path,
        **categorical_paths,
        **drift_paths,
    }


def _prepare_supervised_frame(trainable: pd.DataFrame) -> pd.DataFrame:
    """Handle prepare supervised frame."""
    frame = trainable.reset_index(drop=True).copy()
    frame["row_id"] = frame.index
    frame["requested_date"] = pd.to_datetime(frame["requested_date"])
    frame["requested_year"] = frame["requested_date"].dt.year
    frame["requested_quarter"] = frame["requested_date"].dt.to_period("Q").astype(str)
    frame["requested_month"] = frame["requested_date"].dt.to_period("M").astype(str)
    frame["underlying_list"] = frame["underlyings"].map(parse_underlyings)
    frame["basket_size"] = frame["underlying_list"].map(len)
    return frame


def _assign_temporal_split(supervised: pd.DataFrame) -> pd.DataFrame:
    """Handle assign temporal split."""
    frame = supervised.copy()
    split = temporal_split(frame)
    frame["temporal_split"] = "unassigned"
    frame.loc[split.train_index, "temporal_split"] = "train"
    frame.loc[split.validation_index, "temporal_split"] = "validation"
    frame.loc[split.test_index, "temporal_split"] = "test"
    frame.attrs["split_description"] = split.description
    return frame


def _explode_underlyings(supervised: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Handle explode underlyings."""
    exploded = supervised[
        [
            "row_id",
            "product_type",
            "basket_type",
            "requested_year",
            "temporal_split",
            "underlying_list",
        ]
    ].explode("underlying_list")
    exploded = exploded.rename(columns={"underlying_list": "underlying"})
    return exploded.merge(reference, on="underlying", how="left")


def _rolling_audit(supervised: pd.DataFrame, exploded: pd.DataFrame) -> pd.DataFrame:
    """Handle rolling audit."""
    rows: list[dict[str, Any]] = []
    folds = rolling_temporal_folds(supervised)
    all_products = set(supervised["product_type"].dropna())
    all_underlyings = set(exploded["underlying"].dropna())
    for fold in folds:
        train_rows = supervised.loc[fold.train_index]
        validation_rows = supervised.loc[fold.validation_index]
        validation_underlyings = set(
            exploded.loc[exploded["row_id"].isin(validation_rows["row_id"]), "underlying"].dropna()
        )
        train_underlyings = set(
            exploded.loc[exploded["row_id"].isin(train_rows["row_id"]), "underlying"].dropna()
        )
        validation_products = set(validation_rows["product_type"].dropna())
        train_products = set(train_rows["product_type"].dropna())
        rows.append(
            {
                "fold": fold.description,
                "train_years": _compact_year_range(train_rows["requested_year"]),
                "validation_year": fold.validation_year,
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "validation_product_distribution": _count_share_text(
                    validation_rows["product_type"]
                ),
                "validation_basket_distribution": _count_share_text(validation_rows["basket_type"]),
                "validation_underlyings": len(validation_underlyings),
                "unseen_products_in_validation": _join_values(validation_products - train_products),
                "unseen_underlyings_in_validation": _join_values(
                    validation_underlyings - train_underlyings
                ),
                "products_absent_from_validation": _join_values(all_products - validation_products),
                "underlyings_absent_from_validation": _join_values(
                    all_underlyings - validation_underlyings
                ),
            }
        )
    return pd.DataFrame(rows)


def _segment_viability(supervised: pd.DataFrame, exploded: pd.DataFrame) -> pd.DataFrame:
    """Handle segment viability."""
    rows: list[dict[str, Any]] = []
    for basket_type, group in supervised.groupby("basket_type", dropna=False):
        split_counts = group["temporal_split"].value_counts()
        group_underlyings = set(
            exploded.loc[exploded["row_id"].isin(group["row_id"]), "underlying"].dropna()
        )
        train_rows = group.loc[group["temporal_split"] == "train"]
        train_underlyings = set(
            exploded.loc[exploded["row_id"].isin(train_rows["row_id"]), "underlying"].dropna()
        )
        validation_rows = group.loc[group["temporal_split"] == "validation"]
        validation_underlyings = set(
            exploded.loc[exploded["row_id"].isin(validation_rows["row_id"]), "underlying"].dropna()
        )
        checks = {
            "train_rows_ok": int(split_counts.get("train", 0)) >= VIABILITY_RULES["min_train_rows"],
            "validation_rows_ok": int(split_counts.get("validation", 0))
            >= VIABILITY_RULES["min_validation_rows"],
            "multi_year_ok": group["requested_year"].nunique() >= VIABILITY_RULES["min_years"],
            "products_ok": group["product_type"].nunique() >= VIABILITY_RULES["min_products"],
            "underlyings_ok": len(group_underlyings) >= VIABILITY_RULES["min_underlyings"],
            "validation_underlyings_seen_in_train": validation_underlyings.issubset(
                train_underlyings
            ),
        }
        viable = all(checks.values())
        rows.append(
            {
                "segment": str(basket_type),
                "total_rows": len(group),
                "train_rows": int(split_counts.get("train", 0)),
                "validation_rows": int(split_counts.get("validation", 0)),
                "years_present": _compact_year_range(group["requested_year"]),
                "products_present": _join_values(group["product_type"].dropna().unique()),
                "underlyings_present": len(group_underlyings),
                "target_mean": float(group["avg_duration_months"].mean()),
                "target_median": float(group["avg_duration_months"].median()),
                "target_std": float(group["avg_duration_months"].std(ddof=1)),
                "missing_validation_underlyings_in_train": _join_values(
                    validation_underlyings - train_underlyings
                ),
                "viability_checks_passed": int(sum(checks.values())),
                "viability_checks_total": len(checks),
                "is_viable_for_segment_model": bool(viable),
                "recommendation": "viable_to_test"
                if viable
                else "keep_global_or_compare_carefully",
            }
        )
    return pd.DataFrame(rows).sort_values("segment")


def _product_model_viability(supervised: pd.DataFrame) -> pd.DataFrame:
    """Handle product model viability."""
    rows: list[dict[str, Any]] = []
    for product_type, group in supervised.groupby("product_type", dropna=False):
        split_counts = group["temporal_split"].value_counts()
        viable = (
            int(split_counts.get("train", 0)) >= VIABILITY_RULES["min_train_rows"]
            and int(split_counts.get("validation", 0)) >= VIABILITY_RULES["min_validation_rows"]
            and group["requested_year"].nunique() >= VIABILITY_RULES["min_years"]
        )
        rows.append(
            {
                "product_type": str(product_type),
                "total_rows": len(group),
                "train_rows": int(split_counts.get("train", 0)),
                "validation_rows": int(split_counts.get("validation", 0)),
                "years_present": int(group["requested_year"].nunique()),
                "basket_types": _join_values(group["basket_type"].dropna().unique()),
                "is_viable_for_product_model": bool(viable),
            }
        )
    return pd.DataFrame(rows).sort_values("total_rows", ascending=False)


def _summary_payload(
    rfqs: pd.DataFrame,
    supervised: pd.DataFrame,
    exploded: pd.DataFrame,
    rolling: pd.DataFrame,
    segment_viability: pd.DataFrame,
    product_model_viability: pd.DataFrame,
    conclusions: list[str],
) -> dict[str, Any]:
    """Handle summary payload."""
    split_summary = _split_summary(supervised).set_index("split").to_dict(orient="index")
    return {
        "supervised_rows": len(supervised),
        "full_dataset_rows": len(rfqs),
        "requested_date_min": _date_text(supervised["requested_date"].min()),
        "requested_date_max": _date_text(supervised["requested_date"].max()),
        "split_description": str(supervised.attrs.get("split_description", "")),
        "split_rows": split_summary,
        "product_counts": _value_counts_payload(supervised["product_type"]),
        "basket_type_counts": _value_counts_payload(supervised["basket_type"]),
        "underlying_counts": _value_counts_payload(exploded["underlying"]),
        "rolling_folds": len(rolling),
        "segment_viability": _records_for_json(segment_viability),
        "product_model_viability": _records_for_json(product_model_viability),
        "conclusions": conclusions,
    }


def _render_html_report(
    rfqs: pd.DataFrame,
    supervised: pd.DataFrame,
    exploded: pd.DataFrame,
    reference: pd.DataFrame,
    rolling: pd.DataFrame,
    segment_viability: pd.DataFrame,
    product_model_viability: pd.DataFrame,
    drift_tables: dict[str, pd.DataFrame],
) -> str:
    """Handle render html report."""
    tables = _report_tables(rfqs, supervised, exploded, reference)
    figures = _report_figures(rfqs, supervised, exploded)
    metrics = _global_metrics(rfqs, supervised, exploded)
    split_description = escape(str(supervised.attrs.get("split_description", "")))

    body = f"""
  <section>
    <h2>1. Resumen Global Del Dataset Supervisado</h2>
    <div class="metric-grid">{"".join(metrics)}</div>
    {figures["executed_distribution"]}
  </section>

  <section>
    <h2>2. Distribución Por Producto</h2>
    {table_html(tables["product_summary"])}
    {figures["product_volume"]}
    {figures["target_by_product"]}
  </section>

  <section>
    <h2>3. Distribución Por Tipo De Cesta</h2>
    {table_html(tables["basket_summary"])}
    {figures["basket_volume_year"]}
    {figures["target_by_basket"]}
  </section>

  <section>
    <h2>4. Distribución Temporal Global</h2>
    {table_html(tables["year_summary"])}
    {figures["year_volume"]}
    {figures["quarter_volume"]}
    {figures["target_by_year"]}
    {figures["product_by_year"]}
    {figures["basket_by_year"]}
  </section>

  <section>
    <h2>5. Auditoría Del Split Temporal De Desarrollo</h2>
    <div class="callout">Split del pipeline: <strong>{split_description}</strong>.</div>
    {table_html(tables["split_summary"])}
    <h3>split x product_type</h3>
    {table_html(tables["split_product"])}
    <h3>split x basket_type</h3>
    {table_html(tables["split_basket"])}
    <h3>split x requested_year</h3>
    {table_html(tables["split_year"])}
    <h3>split x underlying</h3>
    {table_html(tables["split_underlying"])}
  </section>

  <section>
    <h2>6. Drift Entre Train Y Splits De Comparación</h2>
    <p>Los estados se basan en tamaños de efecto y cambios de cobertura; los p-values se conservan como contexto, no como criterio único.</p>
    <h3>Variables numéricas</h3>
    {table_html(drift_tables["numeric_drift"], max_rows=30)}
    <h3>Variables categóricas</h3>
    {table_html(drift_tables["categorical_drift"], max_rows=30)}
  </section>

  <section>
    <h2>7. Auditoría De Rolling Temporal Validation</h2>
    {table_html(rolling)}
  </section>

  <section>
    <h2>8. Representación De Los 14 Underlyings</h2>
    {table_html(tables["underlying_summary"])}
    {figures["underlying_volume"]}
    {figures["underlying_year_heatmap"]}
    {figures["underlying_split_heatmap"]}
    {figures["underlying_product_heatmap"]}
  </section>

  <section>
    <h2>9. Auditoría Para Posible Modelo Segmentado</h2>
    {table_html(segment_viability)}
    <h3>Viabilidad De Modelo Por Product Type</h3>
    {table_html(product_model_viability)}
  </section>
"""
    return html_document(
        title="Auditoría De Splits — starwars_autocalls",
        header_title="Auditoría De Splits",
        header_subtitle=(
            "Cobertura temporal y categórica del conjunto de desarrollo, rolling validation "
            "con ventana expansiva y viabilidad de modelos segmentados."
        ),
        body=body,
    )


def _report_tables(
    rfqs: pd.DataFrame,
    supervised: pd.DataFrame,
    exploded: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Handle report tables."""
    total = len(supervised)
    product_summary = (
        supervised.groupby("product_type", dropna=False)
        .agg(
            trainable_rows=("rfq_id", "size"),
            basket_type_associated=("basket_type", lambda s: _join_values(s.dropna().unique())),
            mean_underlyings=("basket_size", "mean"),
            min_requested_date=("requested_date", "min"),
            max_requested_date=("requested_date", "max"),
            target_mean=("avg_duration_months", "mean"),
            target_median=("avg_duration_months", "median"),
            target_std=("avg_duration_months", "std"),
        )
        .reset_index()
    )
    product_summary["pct_total"] = _pct(product_summary["trainable_rows"], total)
    product_summary = product_summary[
        [
            "product_type",
            "trainable_rows",
            "pct_total",
            "basket_type_associated",
            "mean_underlyings",
            "min_requested_date",
            "max_requested_date",
            "target_mean",
            "target_median",
            "target_std",
        ]
    ].sort_values("trainable_rows", ascending=False)

    basket_summary = (
        supervised.groupby("basket_type", dropna=False)
        .agg(
            trainable_rows=("rfq_id", "size"),
            products=("product_type", "nunique"),
            target_mean=("avg_duration_months", "mean"),
            target_median=("avg_duration_months", "median"),
            target_std=("avg_duration_months", "std"),
        )
        .reset_index()
        .sort_values("trainable_rows", ascending=False)
    )
    basket_summary["pct_total"] = _pct(basket_summary["trainable_rows"], total)

    year_summary = (
        supervised.groupby("requested_year", dropna=False)
        .agg(
            trainable_rows=("rfq_id", "size"),
            product_types=("product_type", "nunique"),
            basket_types=("basket_type", "nunique"),
            target_mean=("avg_duration_months", "mean"),
            target_median=("avg_duration_months", "median"),
            target_std=("avg_duration_months", "std"),
        )
        .reset_index()
    )
    year_summary["pct_total"] = _pct(year_summary["trainable_rows"], total)

    underlying_mentions = (
        exploded.groupby("underlying", dropna=False)
        .agg(
            total_appearances=("row_id", "size"),
            years_present=("requested_year", "nunique"),
            splits_present=("temporal_split", "nunique"),
            basket_types=("basket_type", lambda s: _join_values(s.dropna().unique())),
            products=("product_type", lambda s: _join_values(s.dropna().unique())),
        )
        .reset_index()
    )
    underlying_summary = (
        reference.merge(underlying_mentions, on="underlying", how="left")
        .fillna(
            {
                "total_appearances": 0,
                "years_present": 0,
                "splits_present": 0,
                "basket_types": "",
                "products": "",
            }
        )
        .sort_values("total_appearances", ascending=False)
    )

    return {
        "product_summary": _format_table(product_summary),
        "basket_summary": _format_table(basket_summary),
        "year_summary": _format_table(year_summary),
        "split_summary": _format_table(_split_summary(supervised)),
        "split_product": _format_table(_crosstab(supervised, "temporal_split", "product_type")),
        "split_basket": _format_table(_crosstab(supervised, "temporal_split", "basket_type")),
        "split_year": _format_table(_crosstab(supervised, "temporal_split", "requested_year")),
        "split_underlying": _format_table(_crosstab(exploded, "temporal_split", "underlying")),
        "underlying_summary": _format_table(underlying_summary),
        "executed_distribution": _format_table(
            rfqs["executed"]
            .value_counts(dropna=False)
            .rename_axis("executed")
            .reset_index(name="rows")
        ),
    }


def _report_figures(
    rfqs: pd.DataFrame,
    supervised: pd.DataFrame,
    exploded: pd.DataFrame,
) -> dict[str, str]:
    """Handle report figures."""
    include_js = True

    def _fig(fig: Any) -> str:
        """Handle fig."""
        nonlocal include_js
        html = fig_html(fig, include_plotlyjs=include_js)
        include_js = False
        return html

    product_counts = (
        supervised["product_type"]
        .value_counts()
        .rename_axis("product_type")
        .reset_index(name="rows")
    )
    basket_year = (
        supervised.groupby(["requested_year", "basket_type"]).size().reset_index(name="rows")
    )
    year_counts = supervised.groupby("requested_year").size().reset_index(name="rows")
    quarter_counts = supervised.groupby("requested_quarter").size().reset_index(name="rows")
    target_year = (
        supervised.groupby("requested_year")["avg_duration_months"]
        .agg(target_mean="mean", target_median="median")
        .reset_index()
    )
    product_year = (
        supervised.groupby(["requested_year", "product_type"]).size().reset_index(name="rows")
    )
    underlying_counts = (
        exploded["underlying"].value_counts().rename_axis("underlying").reset_index(name="rows")
    )
    underlying_year = _pivot_count(exploded, "underlying", "requested_year")
    underlying_split = _pivot_count(exploded, "underlying", "temporal_split")
    underlying_product = _pivot_count(exploded, "underlying", "product_type")

    return {
        "executed_distribution": _fig(
            px.bar(
                rfqs["executed"]
                .value_counts(dropna=False)
                .rename_axis("executed")
                .reset_index(name="rows"),
                x="executed",
                y="rows",
                title="Distribución de executed antes del filtro entrenable",
            )
        ),
        "product_volume": _fig(
            px.bar(
                product_counts,
                x="product_type",
                y="rows",
                title="RFQs entrenables por producto",
            )
        ),
        "target_by_product": _fig(
            px.box(
                supervised,
                x="product_type",
                y="avg_duration_months",
                color="basket_type",
                title="Distribución del target por product_type",
            )
        ),
        "basket_volume_year": _fig(
            px.bar(
                basket_year,
                x="requested_year",
                y="rows",
                color="basket_type",
                barmode="group",
                title="RFQs entrenables por basket_type y año",
            )
        ),
        "target_by_basket": _fig(
            px.violin(
                supervised,
                x="basket_type",
                y="avg_duration_months",
                box=True,
                points=False,
                title="Distribución del target por basket_type",
            )
        ),
        "year_volume": _fig(
            px.bar(year_counts, x="requested_year", y="rows", title="RFQs entrenables por año")
        ),
        "quarter_volume": _fig(
            px.bar(
                quarter_counts,
                x="requested_quarter",
                y="rows",
                title="RFQs entrenables por trimestre",
            )
        ),
        "target_by_year": _fig(
            px.line(
                target_year,
                x="requested_year",
                y=["target_mean", "target_median"],
                markers=True,
                title="Target medio y mediano por año",
            )
        ),
        "product_by_year": _fig(
            px.bar(
                product_year,
                x="requested_year",
                y="rows",
                color="product_type",
                title="Composición de productos por año",
            )
        ),
        "basket_by_year": _fig(
            px.bar(
                basket_year,
                x="requested_year",
                y="rows",
                color="basket_type",
                title="Composición de cestas por año",
            )
        ),
        "underlying_volume": _fig(
            px.bar(
                underlying_counts,
                x="underlying",
                y="rows",
                title="Apariciones de underlyings en RFQs entrenables",
            )
        ),
        "underlying_year_heatmap": _fig(
            px.imshow(
                underlying_year,
                text_auto=True,
                aspect="auto",
                title="Apariciones de underlying x año",
            )
        ),
        "underlying_split_heatmap": _fig(
            px.imshow(
                underlying_split,
                text_auto=True,
                aspect="auto",
                title="Apariciones de underlying x split temporal",
            )
        ),
        "underlying_product_heatmap": _fig(
            px.imshow(
                underlying_product,
                text_auto=True,
                aspect="auto",
                title="Apariciones de underlying x product_type",
            )
        ),
    }


def _global_metrics(
    rfqs: pd.DataFrame,
    supervised: pd.DataFrame,
    exploded: pd.DataFrame,
) -> list[str]:
    """Handle global metrics."""
    target = supervised["avg_duration_months"]
    return [
        metric("RFQs entrenables", len(supervised), "executed=True y target no nulo"),
        metric(
            "Rango requested_date",
            f"{_date_text(supervised['requested_date'].min())} - {_date_text(supervised['requested_date'].max())}",
        ),
        metric("Años cubiertos", supervised["requested_year"].nunique()),
        metric("Tipos de producto", supervised["product_type"].nunique()),
        metric("Tipos de cesta", supervised["basket_type"].nunique()),
        metric("Underlyings únicos", exploded["underlying"].nunique()),
        metric("Target medio", f"{target.mean():.2f} meses"),
        metric("Target mediano", f"{target.median():.2f} meses"),
        metric("Target std", f"{target.std(ddof=1):.2f} meses"),
        metric("RFQs dataset completo", len(rfqs), "antes del filtro supervisado"),
    ]


def _split_summary(supervised: pd.DataFrame) -> pd.DataFrame:
    """Handle split summary."""
    total = len(supervised)
    rows: list[dict[str, Any]] = []
    for split_name in sorted(supervised["temporal_split"].unique()):
        group = supervised.loc[supervised["temporal_split"] == split_name]
        rows.append(
            {
                "split": split_name,
                "rows": len(group),
                "pct_total": _pct_value(len(group), total),
                "min_requested_date": _date_text(group["requested_date"].min()),
                "max_requested_date": _date_text(group["requested_date"].max()),
                "product_types": int(group["product_type"].nunique()),
                "basket_types": int(group["basket_type"].nunique()),
                "target_mean": float(group["avg_duration_months"].mean()),
                "target_median": float(group["avg_duration_months"].median()),
                "target_std": float(group["avg_duration_months"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def _automatic_conclusions(
    supervised: pd.DataFrame,
    exploded: pd.DataFrame,
    rolling: pd.DataFrame,
    segment_viability: pd.DataFrame,
    product_model_viability: pd.DataFrame,
) -> list[str]:
    """Handle automatic conclusions."""
    conclusions: list[str] = []
    total = len(supervised)
    product_counts = supervised["product_type"].value_counts()
    year_counts = supervised["requested_year"].value_counts()
    basket_counts = supervised["basket_type"].value_counts()
    top_product_share = float(product_counts.iloc[0] / total) if total else 0.0
    top_year_share = float(year_counts.iloc[0] / total) if total else 0.0
    all_underlyings = set(exploded["underlying"].dropna())
    split_underlyings = {
        split_name: set(group["underlying"].dropna())
        for split_name, group in exploded.groupby("temporal_split")
    }
    missing_by_split = {
        split_name: sorted(all_underlyings - values)
        for split_name, values in split_underlyings.items()
    }
    if top_product_share <= CONCENTRATION_THRESHOLDS["max_top_product_share"]:
        conclusions.append(
            "El dataset supervisado está razonablemente balanceado por producto; ningún product_type domina de forma extrema."
        )
    else:
        conclusions.append(
            f"El dataset supervisado tiene concentración por producto: {product_counts.index[0]} representa {top_product_share:.1%} de las filas."
        )
    if top_year_share <= CONCENTRATION_THRESHOLDS["max_top_year_share"]:
        conclusions.append(
            "La distribución anual es razonablemente repartida para una validación temporal."
        )
    else:
        conclusions.append(
            f"La distribución anual está concentrada: {year_counts.index[0]} aporta {top_year_share:.1%} de las filas entrenables."
        )
    conclusions.append(
        f"La composición single vs worst_of es {basket_counts.to_dict()}, por lo que el split debe preservar ambos grupos para comparar modelos de forma justa."
    )
    if all(not missing for missing in missing_by_split.values()):
        conclusions.append(
            "Los splits analizados preservan todos los underlyings observados en train."
        )
    else:
        missing_text = "; ".join(
            f"{split}: {', '.join(values) if values else 'none'}"
            for split, values in sorted(missing_by_split.items())
        )
        conclusions.append(
            f"Hay underlyings ausentes en algún split temporal ({missing_text}); debe considerarse al interpretar validation."
        )
    conclusions.append(
        "La validación temporal con holdout está justificada porque respeta el orden de requested_date y mantiene el test más reciente fuera de tuning."
    )
    if rolling.empty:
        conclusions.append(
            "La rolling temporal validation con ventana expansiva no tiene suficientes folds con los años disponibles."
        )
    else:
        conclusions.append(
            f"La rolling temporal validation está justificada como prueba de estabilidad: se han auditado {len(rolling)} folds anuales expansivos."
        )
    for _, row in segment_viability.iterrows():
        segment = row["segment"]
        if bool(row["is_viable_for_segment_model"]):
            conclusions.append(
                f"Es viable probar un modelo separado para {segment}, pero debe compararse contra el modelo global con el mismo protocolo temporal."
            )
        else:
            conclusions.append(
                f"No conviene aislar directamente un modelo para {segment} sin cautela: no supera todos los criterios mínimos de volumen/representación."
            )
    viable_products = product_model_viability["is_viable_for_product_model"].sum()
    if viable_products == len(product_model_viability):
        conclusions.append(
            "Un modelo por product_type parece viable por volumen mínimo, aunque aumenta complejidad operativa."
        )
    else:
        conclusions.append(
            "Un modelo por product_type no parece igual de viable para todos los productos; el riesgo principal es comparar segmentos con poco volumen temporal."
        )
    conclusions.append(
        "La alternativa más prudente antes de segmentar es mantener un modelo global con variables de interacción estructural como basket_type, basket_size y worst_of_pressure."
    )
    return conclusions


def _crosstab(frame: pd.DataFrame, index: str, columns: str) -> pd.DataFrame:
    """Handle crosstab."""
    table = pd.crosstab(frame[index], frame[columns], dropna=False)
    return table.reset_index()


def _pivot_count(frame: pd.DataFrame, index: str, columns: str) -> pd.DataFrame:
    """Handle pivot count."""
    table = pd.crosstab(frame[index], frame[columns], dropna=False)
    return table.sort_index()


def _format_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Handle format table."""
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_datetime64_any_dtype(formatted[column]):
            formatted[column] = formatted[column].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].round(3)
    return formatted


def _pct(series: pd.Series, denominator: int) -> pd.Series:
    """Handle pct."""
    if denominator == 0:
        return pd.Series(np.nan, index=series.index)
    return (series / denominator * 100).round(2)


def _pct_value(value: int, denominator: int) -> float:
    """Handle pct value."""
    return round(value / denominator * 100, 2) if denominator else float("nan")


def _date_text(value: Any) -> str:
    """Handle date text."""
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _compact_year_range(years: pd.Series) -> str:
    """Handle compact year range."""
    clean_years = sorted(pd.Series(years).dropna().astype(int).unique())
    if not clean_years:
        return ""
    if len(clean_years) == 1:
        return str(clean_years[0])
    return f"{clean_years[0]}-{clean_years[-1]}"


def _join_values(values: Any, max_items: int = 20) -> str:
    """Handle join values."""
    clean = sorted({str(value) for value in values if pd.notna(value)})
    if not clean:
        return ""
    if len(clean) > max_items:
        return ", ".join(clean[:max_items]) + f" ... (+{len(clean) - max_items})"
    return ", ".join(clean)


def _count_share_text(series: pd.Series) -> str:
    """Handle count share text."""
    counts = series.value_counts(dropna=False)
    total = int(counts.sum())
    parts = []
    for value, count in counts.items():
        parts.append(f"{value}: {count} ({_pct_value(int(count), total):.1f}%)")
    return "; ".join(parts)


def _value_counts_payload(series: pd.Series) -> dict[str, int]:
    """Handle value counts payload."""
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _records_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Handle records for json."""
    records = frame.replace({np.nan: None}).to_dict(orient="records")
    return [{str(key): _json_safe(value) for key, value in row.items()} for row in records]


def _json_safe(value: Any) -> Any:
    """Handle json safe."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return _date_text(value)
    return value
