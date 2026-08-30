"""Genera figuras y tablas reproducibles para la documentación MkDocs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "model_evaluation"
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets" / "figures"
INCLUDES = DOCS / "includes"

COLORS = {
    "blue": "#2563eb",
    "orange": "#ea580c",
    "green": "#16a34a",
    "purple": "#7c3aed",
    "red": "#dc2626",
    "gray": "#64748b",
}


def _setup() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    INCLUDES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(ASSETS / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _format(value: object, decimals: int = 4) -> str:
    if value is None or pd.isna(value):
        return "No registrado"
    if isinstance(value, float | np.floating):
        return f"{float(value):.{decimals}f}"
    return str(value)


def _write_markdown_table(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        clean = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(clean) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_eda_figures() -> None:
    rfqs = pd.read_csv(
        ROOT / "data" / "raw" / "rfqs.csv",
        parse_dates=["requested_date", "start_date", "end_date"],
    )
    supervised = rfqs.loc[rfqs["executed"] & rfqs["avg_duration_months"].notna()].copy()
    supervised["nominal_maturity_months"] = (
        supervised["end_date"] - supervised["start_date"]
    ).dt.days / 30.4375

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 130, 35)
    axes[0].hist(
        supervised["avg_duration_months"],
        bins=bins,
        color=COLORS["blue"],
        alpha=0.82,
        label="Duración media simulada",
    )
    axes[0].hist(
        supervised["nominal_maturity_months"],
        bins=bins,
        color=COLORS["orange"],
        alpha=0.55,
        label="Madurez nominal",
    )
    axes[0].set(title="Distribución de duración y madurez", xlabel="Meses", ylabel="RFQs")
    axes[0].legend(frameon=False)

    groups = [
        supervised.loc[supervised["basket_type"].eq(segment), "avg_duration_months"]
        for segment in ["single", "worst_of"]
    ]
    axes[1].boxplot(groups, tick_labels=["single", "worst_of"], showfliers=False)
    axes[1].set(title="Duración por tipo de cesta", ylabel="Meses")
    _save(fig, "eda_target_distribution.png")

    yearly = (
        rfqs.assign(year=rfqs["requested_date"].dt.year)
        .groupby("year")
        .agg(rfqs=("rfq_id", "size"), executed=("executed", "sum"))
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(yearly.index, yearly["rfqs"], color=COLORS["gray"], label="RFQs totales")
    ax.bar(yearly.index, yearly["executed"], color=COLORS["blue"], label="RFQs ejecutadas")
    ax.set(title="Volumen anual y muestra supervisada", xlabel="Año", ylabel="RFQs")
    ax.legend(frameon=False, ncol=2)
    _save(fig, "eda_volume_by_year.png")

    products = sorted(supervised["product_type"].unique())
    product_values = [
        supervised.loc[supervised["product_type"].eq(product), "avg_duration_months"]
        for product in products
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(product_values, orientation="horizontal", tick_labels=products, showfliers=False)
    ax.set(title="Duración simulada por producto", xlabel="Meses")
    _save(fig, "eda_target_by_product.png")

    fig, ax = plt.subplots(figsize=(6, 5.5))
    delta = supervised["avg_duration_months"] - supervised["nominal_maturity_months"]
    regular = delta <= 0.1
    ax.scatter(
        supervised.loc[regular, "nominal_maturity_months"],
        supervised.loc[regular, "avg_duration_months"],
        s=7,
        alpha=0.18,
        color=COLORS["blue"],
        label="Dentro de madurez",
    )
    ax.scatter(
        supervised.loc[~regular, "nominal_maturity_months"],
        supervised.loc[~regular, "avg_duration_months"],
        s=12,
        alpha=0.7,
        color=COLORS["red"],
        label="Exceso > 0,1 meses",
    )
    limit = 132
    ax.plot([0, limit], [0, limit], linestyle="--", color="black", linewidth=1, label="y = x")
    ax.set(
        title="Target frente a madurez contractual",
        xlabel="Madurez nominal (meses)",
        ylabel="Duración media simulada (meses)",
        xlim=(0, limit),
        ylim=(0, limit),
    )
    ax.legend(frameon=False)
    _save(fig, "target_vs_maturity.png")


def generate_experiment_figures() -> None:
    benchmark = pd.read_csv(REPORTS / "benchmark_comparison.csv")
    baseline_mask = benchmark["estimator_class"].isin(["ConstantRegressor", "GroupMedianRegressor"])
    baselines = benchmark.loc[baseline_mask].sort_values("validation_mae")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        baselines["model_name"],
        baselines["validation_mae"],
        color=[COLORS["green"]] + [COLORS["gray"]] * (len(baselines) - 1),
    )
    ax.invert_yaxis()
    ax.set(title="Baselines de negocio", xlabel="MAE de validación (meses)")
    _save(fig, "baseline_comparison.png")

    models = benchmark.loc[~baseline_mask].copy()
    families = sorted(models["estimator_class"].unique())
    palette = list(COLORS.values())
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for index, family in enumerate(families):
        data = models.loc[models["estimator_class"].eq(family)]
        ax.scatter(
            data["fit_seconds"],
            data["validation_mae"],
            alpha=0.75,
            s=32,
            label=family.replace("Regressor", ""),
            color=palette[index % len(palette)],
        )
    best = models.nsmallest(1, "validation_mae").iloc[0]
    ax.scatter(
        [best["fit_seconds"]],
        [best["validation_mae"]],
        marker="*",
        s=150,
        color="black",
        label="Mejor MAE",
        zorder=5,
    )
    ax.set(
        title="Calidad frente a coste de entrenamiento",
        xlabel="Tiempo de ajuste (s)",
        ylabel="MAE de validación (meses)",
    )
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _save(fig, "model_comparison.png")

    rolling = pd.read_csv(REPORTS / "rolling_benchmark_by_year.csv")
    selected = [
        "catboost_native__all_without_noise",
        "lightgbm_native__all_without_noise",
        "hist_gradient_boosting__all_without_noise",
        "median_by_product_frequency_maturity",
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for index, model in enumerate(selected):
        data = rolling.loc[rolling["model_name"].eq(model)].sort_values("validation_year")
        if data.empty:
            continue
        ax.plot(
            data["validation_year"],
            data["validation_mae"],
            marker="o",
            linewidth=1.8,
            color=palette[index],
            label=model.replace("__", " · "),
        )
    ax.set(title="Estabilidad temporal de candidatos", xlabel="Año validado", ylabel="MAE (meses)")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, "rolling_stability.png")

    tuning = pd.read_csv(REPORTS / "tuning_comparison.csv")
    base_mae = benchmark.set_index("model_name")["validation_mae"]
    labels = [name.split("__")[0] for name in tuning["base_model_name"]]
    x = np.arange(len(tuning))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(
        x - width / 2,
        [base_mae.get(name, np.nan) for name in tuning["base_model_name"]],
        width,
        label="Sin tuning",
        color=COLORS["gray"],
    )
    ax.bar(
        x + width / 2,
        tuning["best_validation_mae"],
        width,
        label="Con tuning",
        color=COLORS["blue"],
    )
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set(title="Efecto del ajuste de hiperparámetros", ylabel="MAE de validación (meses)")
    ax.legend(frameon=False)
    _save(fig, "tuning_improvement.png")

    strategy = pd.read_csv(REPORTS / "model_selection_protocol.csv")
    x = np.arange(len(strategy))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - width / 2, strategy["validation_mae"], width, label="Holdout 2022")
    ax.bar(x + width / 2, strategy["rolling_mae_mean"], width, label="Media rolling")
    ax.set_xticks(x, ["Global", "Segmentada"])
    ax.set(title="Comparación de estrategias de serving", ylabel="MAE (meses)")
    ax.legend(frameon=False)
    _save(fig, "strategy_comparison.png")


def generate_result_figures() -> None:
    from starwars_autocalls.config import Settings
    from starwars_autocalls.data.loading import load_all, trainable_rfqs
    from starwars_autocalls.features import FeatureBuilder
    from starwars_autocalls.modeling.artifacts import load_model_artifact
    from starwars_autocalls.modeling.evaluation import temporal_split

    settings = Settings()
    rfqs, volatility, reference = load_all(settings)
    trainable = trainable_rfqs(rfqs)
    split = temporal_split(trainable)
    feature_set = FeatureBuilder().build(
        trainable,
        volatility,
        reference,
        include_target=True,
    )
    artifact = load_model_artifact(settings.model_path)
    actual = feature_set.target.loc[split.test_index].astype(float)
    predictions = artifact["pipeline"].predict(feature_set.frame.loc[split.test_index])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].scatter(actual, predictions, s=10, alpha=0.25, color=COLORS["blue"])
    limit = max(float(actual.max()), float(np.max(predictions))) + 2
    axes[0].plot([0, limit], [0, limit], linestyle="--", color="black", linewidth=1)
    axes[0].set(
        title="Predicción frente a target",
        xlabel="Target simulado (meses)",
        ylabel="Predicción (meses)",
        xlim=(0, limit),
        ylim=(0, limit),
    )
    residuals = actual.to_numpy() - predictions
    axes[1].hist(residuals, bins=35, color=COLORS["orange"], alpha=0.82)
    axes[1].axvline(0, linestyle="--", color="black", linewidth=1)
    axes[1].set(
        title="Distribución de residuos",
        xlabel="Target menos predicción (meses)",
        ylabel="RFQs",
    )
    _save(fig, "final_predictions.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    configurations = [
        ("segment_mae_by_year.csv", "requested_year", "Error por año"),
        ("segment_mae_by_basket_type.csv", "basket_type", "Error por tipo de cesta"),
        ("segment_mae_by_duration_bucket.csv", "duration_bucket", "Error por duración"),
        ("segment_mae_by_product_type.csv", "product_type", "Error por producto"),
    ]
    for ax, (filename, category, title) in zip(axes.flat, configurations, strict=True):
        data = pd.read_csv(REPORTS / filename).sort_values("mae")
        ax.barh(data[category].astype(str), data["mae"], color=COLORS["blue"])
        ax.set(title=title, xlabel="MAE de test (meses)")
        for position, (_, row) in enumerate(data.iterrows()):
            ax.text(row["mae"] + 0.04, position, f"n={int(row['count'])}", va="center", fontsize=7)
    _save(fig, "final_error_analysis.png")

    shap_path = ROOT / "reports" / "explainability" / "shap_catboost_tuned__all_without_noise.csv"
    shap = pd.read_csv(shap_path).nsmallest(15, "shap_rank").sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(shap["feature"], shap["mean_abs_shap"], color=COLORS["purple"])
    ax.set(title="Importancia SHAP del candidato diagnóstico", xlabel="Media de |SHAP| (meses)")
    _save(fig, "interpretability_shap_importance.png")

    source_figure = (
        ROOT
        / "reports"
        / "explainability"
        / "figures"
        / "shap_catboost_tuned__all_without_noise.png"
    )
    if source_figure.exists():
        shutil.copy2(source_figure, ASSETS / "interpretability_shap_summary.png")


def generate_tables() -> None:
    from starwars_autocalls.features.builders import (
        CATEGORICAL_FEATURE_GROUPS,
        FEATURE_BLOCKS,
        NUMERIC_FEATURE_GROUPS,
    )

    feature_rows: list[list[object]] = []
    for feature_type, groups in [
        ("Numérica", NUMERIC_FEATURE_GROUPS),
        ("Categórica", CATEGORICAL_FEATURE_GROUPS),
    ]:
        for group, features in groups.items():
            for feature in features:
                if feature.startswith(("underlying_", "pair_")):
                    continue
                feature_rows.append([group, feature_type, f"`{feature}`"])
    feature_rows.extend(
        [
            ["basket", "Numérica expandida", "`underlying_<ID>`: 14 indicadores multi-hot"],
            ["basket", "Numérica expandida", "`pair_<ID1>_<ID2>`: 91 indicadores de parejas"],
        ]
    )
    _write_markdown_table(
        INCLUDES / "feature-catalog.md",
        ["Grupo", "Tipo", "Variable"],
        feature_rows,
    )
    _write_markdown_table(
        INCLUDES / "feature-blocks.md",
        ["Bloque", "Nº features"],
        [[f"`{name}`", len(features)] for name, features in FEATURE_BLOCKS.items()],
    )

    benchmark = pd.read_csv(REPORTS / "benchmark_comparison.csv")
    baselines = benchmark.loc[
        benchmark["estimator_class"].isin(["ConstantRegressor", "GroupMedianRegressor"])
    ].sort_values("validation_mae")
    _write_markdown_table(
        INCLUDES / "baseline-table.md",
        ["Modelo", "Agrupación", "MAE", "RMSE", "R²", "MedAE", "Ajuste (s)", "Predicción (s)"],
        [
            [
                f"`{row.model_name}`",
                f"`{row.feature_block}`",
                _format(row.validation_mae),
                _format(row.validation_rmse),
                _format(row.validation_r2),
                _format(row.validation_median_absolute_error),
                _format(row.fit_seconds, 3),
                _format(row.validation_predict_seconds, 3),
            ]
            for row in baselines.itertuples()
        ],
    )

    _write_markdown_table(
        INCLUDES / "benchmark-table.md",
        [
            "#",
            "Modelo",
            "Familia",
            "Features",
            "Encoding",
            "MAE",
            "RMSE",
            "R²",
            "MedAE",
            "Ajuste (s)",
            "Predicción (s)",
            "Nº features",
        ],
        [
            [
                index,
                f"`{row.model_name}`",
                row.estimator_class,
                f"`{row.feature_block}`",
                row.encoding_strategy,
                _format(row.validation_mae),
                _format(row.validation_rmse),
                _format(row.validation_r2),
                _format(row.validation_median_absolute_error),
                _format(row.fit_seconds, 3),
                _format(row.validation_predict_seconds, 3),
                int(row.n_total_features),
            ]
            for index, row in enumerate(benchmark.itertuples(), start=1)
        ],
    )

    rolling = pd.read_csv(REPORTS / "rolling_benchmark_summary.csv").sort_values("rolling_mae_mean")
    _write_markdown_table(
        INCLUDES / "rolling-table.md",
        ["#", "Modelo", "Features", "Encoding", "MAE medio", "Desv. MAE", "MAE máximo", "Folds"],
        [
            [
                index,
                f"`{row.model_name}`",
                f"`{row.feature_block}`",
                row.encoding_strategy,
                _format(row.rolling_mae_mean),
                _format(row.rolling_mae_std),
                _format(row.rolling_mae_max),
                int(row.n_folds),
            ]
            for index, row in enumerate(rolling.itertuples(), start=1)
        ],
    )

    tuning = pd.read_csv(REPORTS / "tuning_comparison.csv")
    base_mae = benchmark.set_index("model_name")["validation_mae"]
    tuning_rows: list[list[object]] = []
    for row in tuning.itertuples():
        before = base_mae.get(row.base_model_name, np.nan)
        tuning_rows.append(
            [
                "Global",
                f"`{row.tuned_model_name}`",
                f"`{row.base_model_name.split('__')[-1]}`",
                _format(before),
                _format(row.best_validation_mae),
                _format(before - row.best_validation_mae),
                int(row.n_trials),
                "No registrado",
                f"`{row.best_params}`",
            ]
        )
    stable = pd.read_csv(REPORTS / "global_stable_tuning_comparison.csv")
    stable_benchmark = pd.read_csv(REPORTS / "global_stable_benchmark.csv").set_index("model_name")
    for row in stable.itertuples():
        before = stable_benchmark.loc[row.base_model_name, "validation_mae"]
        tuning_rows.append(
            [
                "Global estable",
                f"`{row.tuned_model_name}`",
                f"`{row.feature_block}`",
                _format(before),
                _format(row.best_validation_mae),
                _format(before - row.best_validation_mae),
                int(row.n_trials),
                _format(row.fit_seconds, 3),
                f"`{row.best_params}`",
            ]
        )
    segmented = pd.read_csv(REPORTS / "segmented_tuning_comparison.csv")
    segmented_base = pd.read_csv(REPORTS / "segmented_benchmark_single_worstof.csv").set_index(
        ["segment", "model_name"]
    )
    for row in segmented.itertuples():
        before = segmented_base.loc[(row.segment, row.base_model_name), "validation_mae"]
        tuning_rows.append(
            [
                row.segment,
                f"`{row.tuned_model_name}`",
                f"`{row.feature_block}`",
                _format(before),
                _format(row.best_validation_mae),
                _format(before - row.best_validation_mae),
                int(row.n_trials),
                "No registrado",
                f"`{row.best_params}`",
            ]
        )
    _write_markdown_table(
        INCLUDES / "tuning-table.md",
        [
            "Ámbito",
            "Modelo",
            "Features",
            "MAE base",
            "MAE ajustado",
            "Mejora",
            "Trials completados",
            "Ajuste final (s)",
            "Mejores hiperparámetros",
        ],
        tuning_rows,
    )

    selection = pd.read_csv(REPORTS / "model_selection_protocol.csv")
    _write_markdown_table(
        INCLUDES / "strategy-table.md",
        [
            "Estrategia",
            "Modelo",
            "Features",
            "MAE validación",
            "MAE rolling",
            "Máximo rolling",
            "Servida",
        ],
        [
            [
                row.strategy,
                f"`{row.model_name}`",
                f"`{row.feature_block}`",
                _format(row.validation_mae),
                _format(row.rolling_mae_mean),
                _format(row.rolling_mae_max),
                "Sí" if row.selected_for_serving else "No",
            ]
            for row in selection.itertuples()
        ],
    )

    metadata = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text(encoding="utf-8"))
    metrics = metadata["test_metrics"]
    _write_markdown_table(
        INCLUDES / "final-model-table.md",
        [
            "Modelo",
            "Estrategia",
            "Features",
            "Filas ajuste final",
            "MAE",
            "RMSE",
            "R²",
            "MedAE",
            "Ajuste (s)",
            "Predicción (s)",
        ],
        [
            [
                f"`{metadata['model_name']}`",
                "Global",
                f"`{metadata['feature_block']}`",
                metadata["train_rows"] + metadata["validation_rows"],
                _format(metrics["mae"]),
                _format(metrics["rmse"]),
                _format(metrics["r2"]),
                _format(metrics["median_absolute_error"]),
                _format(metadata["final_fit_seconds"], 3),
                _format(metadata["test_predict_seconds"], 3),
            ]
        ],
    )


def main() -> None:
    _setup()
    generate_eda_figures()
    generate_experiment_figures()
    generate_result_figures()
    generate_tables()
    print(f"Figuras: {ASSETS}")
    print(f"Tablas: {INCLUDES}")


if __name__ == "__main__":
    main()
