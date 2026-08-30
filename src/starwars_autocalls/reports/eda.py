"""Eda module."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
from scipy.stats import chi2_contingency, ks_2samp

from starwars_autocalls.features import (
    NUMERIC_FEATURES,
    FeatureBuilder,
    parse_underlyings,
)
from starwars_autocalls.modeling.evaluation import temporal_split
from starwars_autocalls.observability import get_logger
from starwars_autocalls.reports.eda_render import fig_html, html_document, table_html

logger = get_logger(__name__)

# Selection-bias study: does executed=False look statistically different from
# executed=True on the raw RFQ terms? A real difference would mean the model
# (trained only on executed=True rows) is scored at inference time on a
# population it never saw in training (MNAR risk).
SELECTION_BIAS_CONTINUOUS_FEATURES = [
    "autocall_barrier_pct",
    "protection_barrier_pct",
    "quoted_implied_vol",
    "notional_credits",
    "no_call_period_months",
]
SELECTION_BIAS_CATEGORICAL_FEATURES = [
    "basket_type",
    "product_type",
    "observation_frequency",
    "trader_id",
    "counterparty",
]
SELECTION_BIAS_ALPHA = 0.05
SELECTION_BIAS_TEST_COUNT = len(SELECTION_BIAS_CONTINUOUS_FEATURES) + len(
    SELECTION_BIAS_CATEGORICAL_FEATURES
)
SELECTION_BIAS_BONFERRONI_ALPHA = SELECTION_BIAS_ALPHA / SELECTION_BIAS_TEST_COUNT


@dataclass(frozen=True)
class RawTable:
    """Represent RawTable."""

    name: str
    frame: pd.DataFrame


def discover_raw_csv_tables(raw_data_dir: Path) -> list[RawTable]:
    """Return discover raw csv tables."""
    tables: list[RawTable] = []
    for path in sorted(raw_data_dir.glob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            logger.warning("raw_csv_read_failed", path=str(path), exc_info=True)
            continue
        if not frame.empty and len(frame.columns) > 1:
            tables.append(RawTable(name=path.stem, frame=frame))
    return tables


def _reset_supplemental_dir(supplemental_dir: Path) -> None:
    """Handle reset supplemental dir."""
    supplemental_dir.mkdir(parents=True, exist_ok=True)
    for path in supplemental_dir.glob("*"):
        if path.is_file():
            path.unlink()


def _target_column_for_sweetviz(frame: pd.DataFrame) -> str | None:
    """Handle target column for sweetviz."""
    if "avg_duration_months" not in frame:
        return None
    if frame["avg_duration_months"].isna().any():
        return None
    return "avg_duration_months"


def _executed_rfqs_for_sweetviz(rfqs: pd.DataFrame) -> pd.DataFrame:
    """Return every RFQ explicitly marked as executed for the Sweetviz report."""
    executed_mask = rfqs["executed"].eq(True).fillna(False)
    return rfqs.loc[executed_mask].copy().reset_index(drop=True)


@contextmanager
def _quiet_library_report_warnings() -> Iterator[None]:
    """Hide known, non-actionable plotting warnings from optional report libraries.

    Sweetviz and Skrub use Matplotlib internally. Their reports are valid, but
    older versions can emit repeated font fallback messages and warnings for
    string categories that are numeric-looking. Keep those implementation
    details out of the CLI while preserving unrelated warnings and errors.
    """
    matplotlib_logger = logging.getLogger("matplotlib")
    previous_level = matplotlib_logger.level
    matplotlib_logger.setLevel(logging.ERROR)
    try:
        # Sweetviz/Skrub expose progress bars and a few plotting diagnostics on
        # the process streams. They are implementation noise for this CLI;
        # real failures still propagate to the exception handler below.
        with warnings.catch_warnings(), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"Using categorical units to plot a list of strings that are all parsable "
                    r"as floats or dates\. If these strings should be plotted as numbers, cast "
                    r"to the appropriate data type before plotting\."
                ),
                category=UserWarning,
            )
            yield
    finally:
        matplotlib_logger.setLevel(previous_level)


def _write_library_reports(
    raw_tables: list[RawTable],
    output_dir: Path,
    executed_rfqs: pd.DataFrame | None = None,
) -> list[str]:
    """Generate optional Sweetviz and Skrub reports under `output_dir / 'supplemental'`.

    Skrub profiles every raw table. Sweetviz is limited to the RFQs because its target-aware
    analysis is the only part that complements Skrub: it uses all rows explicitly marked
    executed=True, independently of the temporal modeling split.
    """
    supplemental_dir = output_dir / "supplemental"
    _reset_supplemental_dir(supplemental_dir)
    generated: list[str] = []

    try:
        import sweetviz as sv
    except Exception:
        sv = None
        logger.warning("optional_dependency_missing", dependency="sweetviz")
    if sv is not None:
        for table in (table for table in raw_tables if table.name == "rfqs"):
            sweetviz_frame = executed_rfqs if executed_rfqs is not None else table.frame
            target = _target_column_for_sweetviz(sweetviz_frame)
            try:
                with _quiet_library_report_warnings():
                    report = sv.analyze(sweetviz_frame, target_feat=target)
                    path = supplemental_dir / f"sweetviz_{table.name}_report.html"
                    report.show_html(str(path), open_browser=False)
                generated.append(str(path))
            except Exception:
                logger.warning(
                    "sweetviz_report_failed",
                    table_name=table.name,
                    exc_info=True,
                )
                continue

    try:
        from skrub import TableReport
    except Exception:
        TableReport = None
        logger.warning("optional_dependency_missing", dependency="skrub")
    if TableReport is not None:
        for table in raw_tables:
            path = supplemental_dir / f"skrub_{table.name}_table_report.html"
            try:
                with _quiet_library_report_warnings():
                    path.write_text(TableReport(table.frame).html(), encoding="utf-8")
                generated.append(str(path))
            except Exception:
                logger.warning(
                    "skrub_report_failed",
                    table_name=table.name,
                    exc_info=True,
                )
                continue
    return generated


def _executed_selection_bias_report(rfqs: pd.DataFrame) -> pd.DataFrame:
    """Compare executed=True vs executed=False on raw RFQ terms.

    Continuous features: two-sample Kolmogorov-Smirnov test (distribution shape,
    not just mean shift). Categorical features: chi-squared test of independence
    on the executed x category contingency table. p-values are reported both
    against alpha=0.05 and against a Bonferroni-corrected alpha for the
    SELECTION_BIAS_TEST_COUNT tests run here.
    """
    executed_true = rfqs.loc[rfqs["executed"] == True]  # noqa: E712
    executed_false = rfqs.loc[rfqs["executed"] == False]  # noqa: E712
    rows: list[dict[str, object]] = []

    for column in SELECTION_BIAS_CONTINUOUS_FEATURES:
        sample_true = executed_true[column].dropna()
        sample_false = executed_false[column].dropna()
        statistic, p_value = ks_2samp(sample_true, sample_false)
        rows.append(
            {
                "variable": column,
                "test": "ks_2samp",
                "n_executed_true": int(sample_true.size),
                "n_executed_false": int(sample_false.size),
                "mean_executed_true": float(sample_true.mean()),
                "mean_executed_false": float(sample_false.mean()),
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        )

    for column in SELECTION_BIAS_CATEGORICAL_FEATURES:
        contingency = pd.crosstab(rfqs["executed"], rfqs[column].fillna("__missing__"))
        statistic, p_value, _dof, _expected = chi2_contingency(contingency)
        rows.append(
            {
                "variable": column,
                "test": "chi2_contingency",
                "n_executed_true": int(executed_true[column].notna().sum()),
                "n_executed_false": int(executed_false[column].notna().sum()),
                "mean_executed_true": None,
                "mean_executed_false": None,
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        )

    report = pd.DataFrame(rows)
    report["significant_alpha_0.05"] = report["p_value"] < SELECTION_BIAS_ALPHA
    report["significant_bonferroni_alpha"] = report["p_value"] < SELECTION_BIAS_BONFERRONI_ALPHA
    return report


def _build_combined_frame(
    trainable: pd.DataFrame,
    volatility: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Handle build combined frame."""
    feature_set = FeatureBuilder().build(trainable, volatility, reference, include_target=True)
    combined = feature_set.frame.copy()
    combined["avg_duration_months"] = feature_set.target.to_numpy()
    combined["product_type"] = trainable["product_type"].to_numpy()
    combined["basket_type"] = trainable["basket_type"].to_numpy()
    combined["underlyings"] = trainable["underlyings"].to_numpy()
    combined["requested_date"] = trainable["requested_date"].to_numpy()
    return combined


def _reference_usage(trainable: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Handle reference usage."""
    exploded = trainable[["underlyings", "avg_duration_months"]].copy()
    exploded["underlying"] = exploded["underlyings"].map(parse_underlyings)
    exploded = exploded.explode("underlying")
    usage = (
        exploded.groupby("underlying", dropna=False)
        .agg(rfq_mentions=("underlying", "size"), target_mean=("avg_duration_months", "mean"))
        .reset_index()
    )
    return (
        reference.merge(usage, on="underlying", how="left")
        .fillna({"rfq_mentions": 0})
        .sort_values("rfq_mentions", ascending=False)
    )


def _maturity_consistency(rfqs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the supervised maturity comparison and its excess-threshold summary."""
    supervised = rfqs.loc[rfqs["avg_duration_months"].notna()].copy()
    supervised["nominal_maturity_months"] = (
        supervised["end_date"] - supervised["start_date"]
    ).dt.days / 30.4375
    supervised["excess_over_maturity_months"] = (
        supervised["avg_duration_months"] - supervised["nominal_maturity_months"]
    )
    thresholds = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 6.0, 12.0]
    rows = []
    for threshold in thresholds:
        count = int(supervised["excess_over_maturity_months"].gt(threshold).sum())
        rows.append(
            {
                "Exceso sobre madurez": f"> {threshold:g} meses",
                "RFQs": count,
                "Porcentaje": f"{count / len(supervised) * 100:.2f} %",
            }
        )
    return supervised, pd.DataFrame(rows)


def _curated_report_data(
    rfqs: pd.DataFrame,
    trainable: pd.DataFrame,
    volatility: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict[str, Any]:
    """Pure computation layer: aggregates, joins, and the selection-bias study.

    No HTML or figure rendering happens here, so this can be tested or reused
    independently of the report template.
    """
    combined = _build_combined_frame(trainable, volatility, reference)
    reference_usage = _reference_usage(trainable, reference)
    maturity_comparison, maturity_thresholds = _maturity_consistency(rfqs)
    corr_features = ["avg_duration_months"] + [
        feature for feature in NUMERIC_FEATURES if feature in combined.columns
    ]
    corr = combined[corr_features].corr(numeric_only=True)
    target_corr = (
        corr["avg_duration_months"]
        .drop("avg_duration_months")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .reset_index()
    )
    target_corr.columns = ["feature", "correlation_with_target"]
    selection_bias = _executed_selection_bias_report(rfqs)

    monthly_duration = (
        trainable.assign(requested_month=trainable["requested_date"].dt.to_period("M").astype(str))
        .groupby("requested_month")["avg_duration_months"]
        .mean()
        .reset_index()
    )
    monthly_vol = (
        volatility.groupby([pd.Grouper(key="date", freq="MS"), "underlying"])["realized_vol_63d"]
        .mean()
        .reset_index()
    )

    selection_bias_display = selection_bias[["variable", "test", "p_value"]].copy()
    selection_bias_display["test"] = selection_bias_display["test"].replace(
        {"ks_2samp": "KS (numérica)", "chi2_contingency": "Chi-cuadrado (categórica)"}
    )
    selection_bias_display["p_value_ajustado"] = (
        selection_bias_display.pop("p_value").mul(SELECTION_BIAS_TEST_COUNT).clip(upper=1.0)
    ).map(lambda value: f"{value:.4f}")
    selection_bias_display = selection_bias_display.rename(
        columns={
            "variable": "Variable",
            "test": "Contraste",
            "p_value_ajustado": "p-value ajustado",
        }
    )

    return {
        "combined": combined,
        "reference_usage": reference_usage,
        "maturity_comparison": maturity_comparison,
        "maturity_thresholds": maturity_thresholds,
        "target_corr": target_corr,
        "selection_bias_display": selection_bias_display,
        "monthly_duration": monthly_duration,
        "monthly_vol": monthly_vol,
    }


def _curated_report_figures(data: dict[str, Any]) -> list[str]:
    """Handle curated report figures."""
    maturity_comparison = data["maturity_comparison"]
    maturity_limit = float(
        maturity_comparison[["nominal_maturity_months", "avg_duration_months"]].max().max()
    )
    maturity_figure = px.scatter(
        maturity_comparison,
        x="nominal_maturity_months",
        y="avg_duration_months",
        color="product_type",
        opacity=0.45,
        title="Target frente a madurez nominal",
    )
    maturity_figure.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=maturity_limit,
        y1=maturity_limit,
        line={"color": "black", "dash": "dash"},
    )
    return [
        fig_html(
            px.bar(
                data["reference_usage"],
                x="underlying",
                y="rfq_mentions",
                color="structural_base_vol",
                hover_data=["sector", "target_mean"],
                title="Tabla de referencia unida al uso en cestas de RFQs",
            ),
            include_plotlyjs=True,
        ),
        fig_html(
            px.line(
                data["monthly_duration"],
                x="requested_month",
                y="avg_duration_months",
                title="Duración media a lo largo del tiempo",
            )
        ),
        fig_html(
            px.line(
                data["monthly_vol"],
                x="date",
                y="realized_vol_63d",
                color="underlying",
                title="Volatilidad realizada por underlying",
            )
        ),
        fig_html(
            px.scatter(
                data["combined"],
                x="realized_vol_63d_mean",
                y="quoted_implied_vol",
                color="product_type",
                title="Volatilidad implícita cotizada vs volatilidad realizada as-of",
            )
        ),
        fig_html(maturity_figure),
    ]


def _render_curated_report_html(
    data: dict[str, Any],
    figures: list[str],
) -> str:
    """Handle render curated report html."""
    body = f"""
    <section>
      <h2>Relaciones Temporales Y De Mercado</h2>
      {figures[0]}
      {figures[1]}
      {figures[2]}
    </section>

    <section>
      <h2>Features Unidas Point-In-Time</h2>
      <p>Esta sección usa la misma construcción de features leakage-safe que el modelado: la volatilidad realizada se une con la última observación disponible con date <= requested_date.</p>
      <h3>Correlaciones Numéricas Más Fuertes Con El Target</h3>
      {table_html(data["target_corr"], max_rows=20)}
      {figures[3]}
    </section>

    <section>
      <h2>Coherencia Con La Madurez Nominal</h2>
      <p>El gráfico compara la madurez nominal contractual (`nominal_maturity_months`) con el target (`avg_duration_months`). La tabla resume las observaciones que superan la madurez nominal por distintos umbrales, con su número de RFQs y porcentaje sobre la muestra supervisada.</p>
      {figures[4]}
      <h3>Observaciones por meses de exceso</h3>
      {table_html(data["maturity_thresholds"], max_rows=20)}
    </section>

    <section>
      <h2>Sesgo De Selección Por Executed</h2>
      {table_html(data["selection_bias_display"], max_rows=20)}
    </section>

    """
    return html_document(
        title="EDA. Complementario a los automáticos",
        header_title="EDA. Complementario a los automáticos",
        header_subtitle="",
        body=body,
    )


def _write_curated_report(
    rfqs: pd.DataFrame,
    trainable: pd.DataFrame,
    volatility: pd.DataFrame,
    reference: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Handle write curated report."""
    data = _curated_report_data(rfqs, trainable, volatility, reference)
    figures = _curated_report_figures(data)
    html = _render_curated_report_html(data, figures)
    path = output_dir / "eda_report.html"
    path.write_text(html, encoding="utf-8")
    return path


def run_eda(
    rfqs: pd.DataFrame,
    volatility: pd.DataFrame,
    reference: pd.DataFrame,
    output_dir: Path,
    raw_data_dir: Path | None = None,
    write_library_reports: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Perform run eda."""

    def report_progress(message: str) -> None:
        """Perform report progress."""
        if progress is not None:
            progress(message)

    output_dir.mkdir(parents=True, exist_ok=True)
    trainable = rfqs.loc[(rfqs["executed"] == True) & rfqs["avg_duration_months"].notna()].copy()  # noqa: E712
    trainable = trainable.reset_index(drop=True)
    executed_rfqs = _executed_rfqs_for_sweetviz(rfqs)
    split = temporal_split(trainable)
    development_index = split.train_index.union(split.validation_index)
    development = trainable.loc[development_index].copy()
    volatility = volatility.copy()
    volatility["date"] = pd.to_datetime(volatility["date"])

    summary = {
        "rfq_rows": len(rfqs),
        "trainable_rows": len(trainable),
        "development_rows": len(development),
        "split": split.description,
        "date_min": str(rfqs["requested_date"].min().date()),
        "date_max": str(rfqs["requested_date"].max().date()),
        "target_mean": float(development["avg_duration_months"].mean()),
        "target_median": float(development["avg_duration_months"].median()),
        "target_missing_rows": int(rfqs["avg_duration_months"].isna().sum()),
    }
    report_progress("Generando el informe EDA principal")
    report_path = _write_curated_report(rfqs, development, volatility, reference, output_dir)
    raw_tables = (
        discover_raw_csv_tables(raw_data_dir) if raw_data_dir and write_library_reports else []
    )
    report_progress(
        "Generando reportes complementarios Sweetviz/Skrub"
        if raw_tables
        else "Reportes complementarios desactivados"
    )
    library_reports = (
        _write_library_reports(raw_tables, output_dir, executed_rfqs=executed_rfqs)
        if raw_tables
        else []
    )
    report_progress("Análisis EDA completado")
    return {
        "summary": summary,
        "main_report": str(report_path),
        "library_reports": library_reports,
    }
