"""Specs module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from starwars_autocalls.config import RANDOM_SEED
from starwars_autocalls.features import select_feature_block


class ConstantRegressor(BaseEstimator, RegressorMixin):
    """Represent ConstantRegressor."""

    def __init__(self, strategy: str = "mean") -> None:
        """Handle init."""
        self.strategy = strategy

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ConstantRegressor:
        """Fit the estimator."""
        self.value_ = float(np.median(y) if self.strategy == "median" else np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        return np.full(shape=len(X), fill_value=self.value_, dtype=float)


class GroupMedianRegressor(BaseEstimator, RegressorMixin):
    """Represent GroupMedianRegressor."""

    def __init__(self, group_columns: tuple[str, ...]) -> None:
        """Handle init."""
        self.group_columns = group_columns

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GroupMedianRegressor:
        """Fit the estimator."""
        frame = X[list(self.group_columns)].copy()
        frame["target"] = y.to_numpy()
        self.global_median_ = float(np.median(y))
        self.group_medians_ = frame.groupby(list(self.group_columns), dropna=False)[
            "target"
        ].median()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        predictions = []
        for _, row in X[list(self.group_columns)].iterrows():
            key = tuple(row[column] for column in self.group_columns)
            if len(self.group_columns) == 1:
                key = key[0]
            predictions.append(float(self.group_medians_.get(key, self.global_median_)))
        return np.asarray(predictions, dtype=float)


class NativeCategoricalPreprocessor(BaseEstimator, TransformerMixin):
    """Select features and preserve categoricals for native boosting engines."""

    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
        categorical_dtype: str = "category",
    ) -> None:
        """Handle init."""
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.categorical_dtype = categorical_dtype

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> NativeCategoricalPreprocessor:
        """Fit the estimator."""
        numeric = X[self.numeric_features].apply(pd.to_numeric, errors="coerce")
        self.numeric_medians_ = numeric.median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the estimator."""
        numeric = X[self.numeric_features].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(self.numeric_medians_)
        categorical = X[self.categorical_features].copy()
        for column in self.categorical_features:
            categorical[column] = categorical[column].astype("string").fillna("UNKNOWN")
            if self.categorical_dtype == "category":
                categorical[column] = categorical[column].astype("category")
        return pd.concat([numeric, categorical], axis=1)


@dataclass(frozen=True)
class ModelSpec:
    """Represent ModelSpec."""

    name: str
    estimator: BaseEstimator
    encoding_strategy: str
    feature_block: str = "all_without_commercial"
    native_categorical_dtype: str = "category"


def spec_name(estimator_key: str, feature_block: str) -> str:
    """Handle spec name."""
    return f"{estimator_key}__{feature_block}"


def split_spec_name(model_name: str) -> tuple[str, str] | None:
    """Return split spec name."""
    if "__" not in model_name:
        return None
    estimator_key, feature_block = model_name.split("__", 1)
    if not estimator_key or not feature_block:
        return None
    return estimator_key, feature_block


def model_family_from_name(model_name: str) -> str:
    """Handle model family from name."""
    parsed = split_spec_name(model_name)
    estimator_key = parsed[0] if parsed else model_name
    if estimator_key.startswith("catboost"):
        return "catboost"
    if estimator_key.startswith("lightgbm"):
        return "lightgbm"
    if estimator_key.startswith("xgboost"):
        return "xgboost"
    if estimator_key.startswith("hist_gradient_boosting"):
        return "hist_gradient_boosting"
    if estimator_key.startswith("ridge"):
        return "ridge"
    if estimator_key.startswith("extra_trees"):
        return "extra_trees"
    if model_name.startswith(("global_mean", "global_median")):
        return "constant_baseline"
    if model_name.startswith("median_by_"):
        return "group_median_baseline"
    return estimator_key


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    encoding_strategy: str = "onehot",
) -> ColumnTransformer:
    """Perform build preprocessor."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    if encoding_strategy == "ordinal":
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]
        )
    elif encoding_strategy == "onehot":
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=10,
                        sparse_output=False,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported encoding_strategy: {encoding_strategy}")

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(spec: ModelSpec) -> Pipeline:
    """Perform build pipeline."""
    if isinstance(spec.estimator, ConstantRegressor | GroupMedianRegressor):
        return Pipeline(steps=[("model", clone(spec.estimator))])

    numeric, categorical = select_feature_block(spec.feature_block)
    if spec.encoding_strategy == "native":
        preprocessor = NativeCategoricalPreprocessor(
            numeric,
            categorical,
            categorical_dtype=spec.native_categorical_dtype,
        )
        estimator = clone(spec.estimator)
        if estimator.__class__.__module__.startswith("catboost"):
            estimator.set_params(cat_features=categorical)
        return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
    preprocessor = build_preprocessor(numeric, categorical, spec.encoding_strategy)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", spec.estimator)])


def default_model_specs() -> list[ModelSpec]:
    """Handle default model specs."""
    specs = [
        ModelSpec("global_mean", ConstantRegressor("mean"), "onehot", "all_without_commercial"),
        ModelSpec("global_median", ConstantRegressor("median"), "onehot", "all_without_commercial"),
        ModelSpec(
            "median_by_product_type",
            GroupMedianRegressor(("product_type",)),
            "onehot",
            "product",
        ),
        ModelSpec(
            "median_by_product_basket",
            GroupMedianRegressor(("product_type", "basket_type")),
            "onehot",
            "product",
        ),
        ModelSpec(
            "median_by_basket_type",
            GroupMedianRegressor(("basket_type",)),
            "onehot",
            "product",
        ),
        ModelSpec(
            "median_by_product_frequency",
            GroupMedianRegressor(("product_type", "observation_frequency_clean")),
            "onehot",
            "all_without_commercial",
        ),
        ModelSpec(
            "median_by_product_maturity_bucket",
            GroupMedianRegressor(("product_type", "nominal_maturity_bucket")),
            "onehot",
            "all_without_commercial",
        ),
        ModelSpec(
            "median_by_product_frequency_maturity",
            GroupMedianRegressor(
                ("product_type", "observation_frequency_clean", "nominal_maturity_bucket")
            ),
            "onehot",
            "all_without_commercial",
        ),
        ModelSpec(
            "median_by_product_basket_size",
            GroupMedianRegressor(("product_type", "basket_size")),
            "onehot",
            "all_without_commercial",
        ),
        ModelSpec(
            "median_by_single_worstof",
            GroupMedianRegressor(("is_worst_of",)),
            "onehot",
            "all_without_commercial",
        ),
        ModelSpec(
            "median_by_product_single_worstof",
            GroupMedianRegressor(("product_type", "is_worst_of")),
            "onehot",
            "all_without_commercial",
        ),
        ModelSpec(
            "ridge_onehot",
            Ridge(alpha=5.0, random_state=RANDOM_SEED),
            "onehot",
            "all_without_commercial",
        ),
        ModelSpec(
            "ridge_ordinal",
            Ridge(alpha=5.0, random_state=RANDOM_SEED),
            "ordinal",
            "all_without_commercial",
        ),
        ModelSpec(
            "extra_trees_onehot",
            ExtraTreesRegressor(
                n_estimators=200,
                min_samples_leaf=8,
                max_features=0.8,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            "onehot",
            "all_without_commercial",
        ),
        ModelSpec(
            spec_name("hist_gradient_boosting", "all_without_commercial"),
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=250,
                l2_regularization=0.05,
                random_state=RANDOM_SEED,
            ),
            "ordinal",
            "all_without_commercial",
        ),
    ]
    specs.extend(optional_boosting_specs())
    return specs


def optional_boosting_specs() -> list[ModelSpec]:
    """Handle optional boosting specs."""
    specs: list[ModelSpec] = []
    try:
        from catboost import CatBoostRegressor

        specs.append(
            ModelSpec(
                spec_name("catboost_native", "all_without_commercial"),
                CatBoostRegressor(
                    loss_function="MAE",
                    iterations=350,
                    depth=6,
                    learning_rate=0.05,
                    random_seed=RANDOM_SEED,
                    verbose=False,
                ),
                "native",
                "all_without_commercial",
                "object",
            )
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor

        specs.append(
            ModelSpec(
                spec_name("lightgbm_native", "all_without_commercial"),
                LGBMRegressor(
                    n_estimators=350,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=RANDOM_SEED,
                    objective="regression_l1",
                    verbosity=-1,
                ),
                "native",
                "all_without_commercial",
                "category",
            )
        )
    except Exception:
        pass
    try:
        from xgboost import XGBRegressor

        specs.append(
            ModelSpec(
                spec_name("xgboost_ordinal", "all_without_commercial"),
                XGBRegressor(
                    n_estimators=350,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=RANDOM_SEED,
                    objective="reg:absoluteerror",
                ),
                "ordinal",
                "all_without_commercial",
            )
        )
    except Exception:
        pass
    return specs


def ablation_specs() -> list[ModelSpec]:
    """Handle ablation specs."""
    specs = [
        ModelSpec(
            spec_name("hist_gradient_boosting_ablation", block),
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=200,
                l2_regularization=0.05,
                random_state=RANDOM_SEED,
            ),
            "ordinal",
            block,
        )
        for block in [
            "contractual",
            "product",
            "basket",
            "market",
            "commercial",
            "all_without_commercial",
            "all_features",
            "all_without_noise",
            "compact_core",
            "single_core",
            "single_without_noise",
            "worst_of_core",
            "worst_of_without_noise",
        ]
    ]
    specs.extend(feature_selection_specs())
    return specs


def segmented_model_specs() -> dict[str, list[ModelSpec]]:
    """Candidate families for models trained separately by basket segment."""
    return {
        "single": _segment_specs(
            "single",
            [
                "single_core",
                "single_without_noise",
                "single_stable",
                "single_underlying",
                "single_underlying_no_sector",
            ],
        ),
        "worst_of": _segment_specs(
            "worst_of",
            [
                "worst_of_core",
                "worst_of_without_noise",
                "worst_of_stable",
                "worst_of_tail_focus",
                "worst_of_risk_underlying",
                "worst_of_tail_underlying",
            ],
        ),
    }


def global_stable_specs() -> list[ModelSpec]:
    """Global candidates using feature treatments derived from diagnostics."""
    return _segment_specs(
        "global",
        [
            "global_stable",
            "global_stable_tail",
            "global_stable_no_sector",
            "global_risk_underlying",
            "global_all_underlying",
            "global_tail_underlying",
        ],
    )


def _segment_specs(segment: str, feature_blocks: list[str]) -> list[ModelSpec]:
    """Handle segment specs."""
    specs: list[ModelSpec] = []
    for block in feature_blocks:
        specs.append(
            ModelSpec(
                spec_name("hist_gradient_boosting", block),
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=250,
                    l2_regularization=0.05,
                    random_state=RANDOM_SEED,
                ),
                "ordinal",
                block,
            )
        )
    try:
        from lightgbm import LGBMRegressor

        for block in feature_blocks:
            specs.append(
                ModelSpec(
                    spec_name("lightgbm_native", block),
                    LGBMRegressor(
                        n_estimators=350,
                        learning_rate=0.05,
                        num_leaves=31,
                        random_state=RANDOM_SEED,
                        objective="regression_l1",
                        verbosity=-1,
                    ),
                    "native",
                    block,
                    "category",
                )
            )
    except Exception:
        pass
    try:
        from xgboost import XGBRegressor

        for block in feature_blocks:
            specs.append(
                ModelSpec(
                    spec_name("xgboost_ordinal", block),
                    XGBRegressor(
                        n_estimators=350,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=RANDOM_SEED,
                        objective="reg:absoluteerror",
                    ),
                    "ordinal",
                    block,
                )
            )
    except Exception:
        pass
    try:
        from catboost import CatBoostRegressor

        for block in feature_blocks:
            specs.append(
                ModelSpec(
                    spec_name("catboost_native", block),
                    CatBoostRegressor(
                        loss_function="MAE",
                        iterations=350,
                        depth=6,
                        learning_rate=0.05,
                        random_seed=RANDOM_SEED,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                    "native",
                    block,
                    "object",
                )
            )
    except Exception:
        pass
    return specs


def feature_selection_specs() -> list[ModelSpec]:
    """Handle feature selection specs."""
    specs = [
        ModelSpec(
            spec_name("hist_gradient_boosting", "all_without_noise"),
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=250,
                l2_regularization=0.05,
                random_state=RANDOM_SEED,
            ),
            "ordinal",
            "all_without_noise",
        ),
        ModelSpec(
            spec_name("hist_gradient_boosting", "compact_core"),
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=250,
                l2_regularization=0.05,
                random_state=RANDOM_SEED,
            ),
            "ordinal",
            "compact_core",
        ),
    ]
    try:
        from catboost import CatBoostRegressor

        for block in ["all_without_noise", "compact_core"]:
            specs.append(
                ModelSpec(
                    spec_name("catboost_native", block),
                    CatBoostRegressor(
                        loss_function="MAE",
                        iterations=350,
                        depth=6,
                        learning_rate=0.05,
                        random_seed=RANDOM_SEED,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                    "native",
                    block,
                    "object",
                )
            )
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor

        for block in ["all_without_noise", "compact_core"]:
            specs.append(
                ModelSpec(
                    spec_name("lightgbm_native", block),
                    LGBMRegressor(
                        n_estimators=350,
                        learning_rate=0.05,
                        num_leaves=31,
                        random_state=RANDOM_SEED,
                        objective="regression_l1",
                        verbosity=-1,
                    ),
                    "native",
                    block,
                    "category",
                )
            )
    except Exception:
        pass
    try:
        from xgboost import XGBRegressor

        for block in ["all_without_noise", "compact_core"]:
            specs.append(
                ModelSpec(
                    spec_name("xgboost_ordinal", block),
                    XGBRegressor(
                        n_estimators=350,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=RANDOM_SEED,
                        objective="reg:absoluteerror",
                    ),
                    "ordinal",
                    block,
                )
            )
    except Exception:
        pass
    return specs
