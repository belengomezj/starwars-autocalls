"""Builders module."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

TARGET = "avg_duration_months"
KNOWN_UNDERLYINGS = [
    "BSKR",
    "CLNE",
    "CORL",
    "DRC",
    "HTTX",
    "JEDI",
    "KYBR",
    "MNDO",
    "NABO",
    "POBK",
    "REBL",
    "SITH",
    "TECH",
    "WOOK",
]
UNDERLYING_MULTI_HOT_FEATURES = [f"underlying_{underlying}" for underlying in KNOWN_UNDERLYINGS]
UNDERLYING_PAIR_FEATURES = [
    f"pair_{left}_{right}" for left, right in combinations(KNOWN_UNDERLYINGS, 2)
]

NUMERIC_FEATURE_GROUPS = {
    "contractual": [
        "autocall_barrier_pct",
        "protection_barrier_pct",
        "barrier_gap_pct",
        "autocall_barrier_above_par_pct",
        "protection_buffer_pct",
        "autocall_to_protection_ratio",
        "no_call_period_months",
        "no_call_fraction_of_maturity",
        "callable_maturity_months",
        "observation_interval_months",
        "is_daily_observation",
        "no_call_observation_count",
        "estimated_observation_count",
        "post_no_call_observation_count",
        "notional_credits",
        "log_notional_credits",
        "nominal_maturity_months",
    ],
    "date": [
        "requested_year",
        "requested_month",
        "requested_quarter",
    ],
    "basket": [
        "basket_size",
        "is_worst_of",
        "is_single_underlying",
        *UNDERLYING_MULTI_HOT_FEATURES,
        *UNDERLYING_PAIR_FEATURES,
        "basket_complexity_score",
        "worst_of_pressure",
        "worst_of_vol_pressure",
        "structural_base_vol_mean",
        "structural_base_vol_min",
        "structural_base_vol_max",
        "structural_base_vol_second_highest",
        "structural_base_vol_range",
        "structural_base_vol_top2_spread",
        "structural_vol_dispersion_ratio",
        "sector_count",
        "sector_concentration",
        "has_multiple_sectors",
        "realized_vol_63d_second_highest",
        "realized_vol_63d_top2_spread",
        "worst_of_structural_base_vol_range",
        "worst_of_structural_vol_top2_spread",
        "worst_of_realized_vol_63d_range",
        "worst_of_realized_vol_top2_spread",
    ],
    "market": [
        "quoted_implied_vol",
        "realized_vol_63d_mean",
        "realized_vol_63d_min",
        "realized_vol_63d_max",
        "realized_vol_63d_range",
        "basket_realized_vol_21d_mean",
        "basket_realized_vol_21d_max",
        "basket_realized_vol_126d_mean",
        "basket_realized_vol_126d_max",
        "basket_realized_vol_trend_mean",
        "basket_realized_vol_trend_max",
        "basket_realized_vol_zscore_mean",
        "basket_realized_vol_zscore_max",
        "basket_realized_vol_change_21d_mean",
        "basket_realized_vol_change_21d_max",
        "implied_minus_realized_vol_mean",
        "quoted_minus_structural_vol_mean",
        "realized_minus_structural_vol_mean",
        "realized_to_structural_vol_ratio",
        "quoted_to_realized_vol_ratio",
        "quoted_to_structural_vol_ratio",
        "realized_vol_dispersion_ratio",
    ],
}

CATEGORICAL_FEATURE_GROUPS = {
    "contractual": ["observation_frequency_clean"],
    "product": ["product_type", "basket_type", "nominal_maturity_bucket"],
    "basket": [
        "basket_signature",
        "primary_underlying",
        "dominant_sector",
        "highest_vol_sector",
        "highest_structural_vol_underlying",
        "second_highest_structural_vol_underlying",
        "lowest_structural_vol_underlying",
        "highest_realized_vol_underlying",
        "second_highest_realized_vol_underlying",
        "lowest_realized_vol_underlying",
    ],
    "commercial": ["counterparty", "trader_id"],
}


def _unique(items: list[str]) -> list[str]:
    """Handle unique."""
    return list(dict.fromkeys(items))


def _features_for_group(group: str) -> list[str]:
    """Handle features for group."""
    return NUMERIC_FEATURE_GROUPS.get(group, []) + CATEGORICAL_FEATURE_GROUPS.get(group, [])


NUMERIC_FEATURES = _unique(
    [feature for features in NUMERIC_FEATURE_GROUPS.values() for feature in features]
)
CATEGORICAL_FEATURES = _unique(
    [feature for features in CATEGORICAL_FEATURE_GROUPS.values() for feature in features]
)

FEATURE_BLOCKS = {
    "contractual": _features_for_group("contractual"),
    "product": _features_for_group("product"),
    "basket": _features_for_group("basket"),
    "market": _features_for_group("market"),
    "date": _features_for_group("date"),
    "commercial": _features_for_group("commercial"),
}
FEATURE_BLOCKS["all_features"] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
FEATURE_BLOCKS["all_without_commercial"] = _unique(
    [
        feature
        for feature in FEATURE_BLOCKS["all_features"]
        if feature not in FEATURE_BLOCKS["commercial"]
    ]
)
NOISE_CANDIDATE_FEATURES = [
    "requested_month",
    "requested_quarter",
    "notional_credits",
    "log_notional_credits",
    "quoted_to_structural_vol_ratio",
    "quoted_minus_structural_vol_mean",
    "realized_minus_structural_vol_mean",
    "quoted_to_realized_vol_ratio",
    "implied_minus_realized_vol_mean",
    "no_call_observation_count",
]
FEATURE_BLOCKS["all_without_noise"] = _unique(
    [
        feature
        for feature in FEATURE_BLOCKS["all_without_commercial"]
        if feature not in NOISE_CANDIDATE_FEATURES
    ]
)
FEATURE_BLOCKS["compact_core"] = [
    "nominal_maturity_months",
    "callable_maturity_months",
    "autocall_barrier_pct",
    "protection_barrier_pct",
    "barrier_gap_pct",
    "no_call_period_months",
    "observation_interval_months",
    "basket_size",
    "is_worst_of",
    "basket_complexity_score",
    "worst_of_vol_pressure",
    "structural_base_vol_min",
    "structural_base_vol_max",
    "structural_base_vol_range",
    "structural_vol_dispersion_ratio",
    "sector_count",
    "sector_concentration",
    "realized_vol_63d_min",
    "realized_vol_63d_max",
    "basket_realized_vol_21d_mean",
    "basket_realized_vol_126d_mean",
    "basket_realized_vol_trend_mean",
    "basket_realized_vol_zscore_mean",
    "product_type",
    "observation_frequency_clean",
    "dominant_sector",
    "highest_vol_sector",
]
FEATURE_BLOCKS["single_core"] = [
    "nominal_maturity_months",
    "callable_maturity_months",
    "autocall_barrier_pct",
    "protection_barrier_pct",
    "barrier_gap_pct",
    "no_call_period_months",
    "no_call_fraction_of_maturity",
    "observation_interval_months",
    "is_daily_observation",
    "estimated_observation_count",
    "post_no_call_observation_count",
    "quoted_implied_vol",
    "structural_base_vol_mean",
    "realized_vol_63d_mean",
    "basket_realized_vol_21d_mean",
    "basket_realized_vol_126d_mean",
    "basket_realized_vol_trend_mean",
    "basket_realized_vol_zscore_mean",
    "basket_realized_vol_change_21d_mean",
    "requested_year",
    "product_type",
    "observation_frequency_clean",
    "nominal_maturity_bucket",
    "dominant_sector",
]
FEATURE_BLOCKS["single_without_noise"] = [
    feature
    for feature in FEATURE_BLOCKS["single_core"]
    if feature
    not in {
        "requested_year",
        "post_no_call_observation_count",
        "basket_realized_vol_zscore_mean",
        "basket_realized_vol_change_21d_mean",
    }
]
FEATURE_BLOCKS["single_stable"] = [
    feature
    for feature in FEATURE_BLOCKS["single_without_noise"]
    if feature
    not in {
        "basket_realized_vol_trend_mean",
    }
]
FEATURE_BLOCKS["single_underlying"] = _unique(
    [*FEATURE_BLOCKS["single_stable"], "primary_underlying"]
)
FEATURE_BLOCKS["single_underlying_no_sector"] = [
    feature for feature in FEATURE_BLOCKS["single_underlying"] if feature != "dominant_sector"
]
FEATURE_BLOCKS["worst_of_core"] = [
    "nominal_maturity_months",
    "callable_maturity_months",
    "autocall_barrier_pct",
    "protection_barrier_pct",
    "barrier_gap_pct",
    "no_call_period_months",
    "no_call_fraction_of_maturity",
    "observation_interval_months",
    "is_daily_observation",
    "estimated_observation_count",
    "post_no_call_observation_count",
    "basket_size",
    *UNDERLYING_MULTI_HOT_FEATURES,
    *UNDERLYING_PAIR_FEATURES,
    "basket_complexity_score",
    "worst_of_pressure",
    "worst_of_vol_pressure",
    "structural_base_vol_mean",
    "structural_base_vol_min",
    "structural_base_vol_max",
    "structural_base_vol_second_highest",
    "structural_base_vol_range",
    "structural_base_vol_top2_spread",
    "sector_count",
    "sector_concentration",
    "has_multiple_sectors",
    "quoted_implied_vol",
    "realized_vol_63d_mean",
    "realized_vol_63d_min",
    "realized_vol_63d_max",
    "realized_vol_63d_second_highest",
    "realized_vol_63d_range",
    "realized_vol_63d_top2_spread",
    "worst_of_structural_base_vol_range",
    "worst_of_structural_vol_top2_spread",
    "worst_of_realized_vol_63d_range",
    "worst_of_realized_vol_top2_spread",
    "basket_realized_vol_21d_mean",
    "basket_realized_vol_21d_max",
    "basket_realized_vol_126d_mean",
    "basket_realized_vol_126d_max",
    "basket_realized_vol_trend_mean",
    "basket_realized_vol_trend_max",
    "basket_realized_vol_zscore_mean",
    "basket_realized_vol_zscore_max",
    "implied_minus_realized_vol_mean",
    "realized_to_structural_vol_ratio",
    "structural_vol_dispersion_ratio",
    "realized_vol_dispersion_ratio",
    "requested_year",
    "product_type",
    "basket_type",
    "basket_signature",
    "observation_frequency_clean",
    "nominal_maturity_bucket",
    "dominant_sector",
    "highest_vol_sector",
    "highest_structural_vol_underlying",
    "second_highest_structural_vol_underlying",
    "lowest_structural_vol_underlying",
    "highest_realized_vol_underlying",
    "second_highest_realized_vol_underlying",
    "lowest_realized_vol_underlying",
]
FEATURE_BLOCKS["worst_of_without_noise"] = [
    feature
    for feature in FEATURE_BLOCKS["worst_of_core"]
    if feature
    not in {
        "requested_year",
        "post_no_call_observation_count",
        "basket_realized_vol_zscore_mean",
        "basket_realized_vol_zscore_max",
        "implied_minus_realized_vol_mean",
        "realized_vol_dispersion_ratio",
    }
]
FEATURE_BLOCKS["worst_of_stable"] = [
    feature
    for feature in FEATURE_BLOCKS["worst_of_without_noise"]
    if feature
    not in {
        "basket_realized_vol_trend_mean",
        "basket_realized_vol_trend_max",
        "realized_to_structural_vol_ratio",
    }
]
FEATURE_BLOCKS["worst_of_tail_focus"] = [
    feature
    for feature in FEATURE_BLOCKS["worst_of_stable"]
    if feature
    not in {
        "structural_base_vol_mean",
        "realized_vol_63d_mean",
        "basket_realized_vol_21d_mean",
        "basket_realized_vol_126d_mean",
    }
]
FEATURE_BLOCKS["worst_of_risk_underlying"] = _unique(
    [
        *FEATURE_BLOCKS["worst_of_stable"],
        "highest_structural_vol_underlying",
        "second_highest_structural_vol_underlying",
        "lowest_structural_vol_underlying",
        "highest_realized_vol_underlying",
        "second_highest_realized_vol_underlying",
        "lowest_realized_vol_underlying",
    ]
)
FEATURE_BLOCKS["worst_of_tail_underlying"] = _unique(
    [
        *FEATURE_BLOCKS["worst_of_tail_focus"],
        "highest_structural_vol_underlying",
        "second_highest_structural_vol_underlying",
        "lowest_structural_vol_underlying",
        "highest_realized_vol_underlying",
        "second_highest_realized_vol_underlying",
        "lowest_realized_vol_underlying",
    ]
)
FEATURE_BLOCKS["global_stable"] = [
    feature
    for feature in FEATURE_BLOCKS["all_without_noise"]
    if feature
    not in {
        "requested_year",
        "post_no_call_observation_count",
        "basket_realized_vol_trend_mean",
        "basket_realized_vol_trend_max",
        "basket_realized_vol_zscore_mean",
        "basket_realized_vol_zscore_max",
        "basket_realized_vol_change_21d_mean",
        "basket_realized_vol_change_21d_max",
        "realized_to_structural_vol_ratio",
        "realized_vol_dispersion_ratio",
    }
]
FEATURE_BLOCKS["global_stable_tail"] = [
    feature
    for feature in FEATURE_BLOCKS["global_stable"]
    if feature
    not in {
        "structural_base_vol_mean",
        "realized_vol_63d_mean",
        "basket_realized_vol_21d_mean",
        "basket_realized_vol_126d_mean",
    }
]
FEATURE_BLOCKS["global_stable_no_sector"] = [
    feature
    for feature in FEATURE_BLOCKS["global_stable"]
    if feature not in {"dominant_sector", "highest_vol_sector"}
]
FEATURE_BLOCKS["global_risk_underlying"] = _unique(
    [
        *FEATURE_BLOCKS["global_stable"],
        "highest_structural_vol_underlying",
        "highest_realized_vol_underlying",
    ]
)
FEATURE_BLOCKS["global_all_underlying"] = _unique(
    [
        *FEATURE_BLOCKS["global_risk_underlying"],
        "primary_underlying",
    ]
)
FEATURE_BLOCKS["global_tail_underlying"] = _unique(
    [
        *FEATURE_BLOCKS["global_stable_tail"],
        "highest_structural_vol_underlying",
        "highest_realized_vol_underlying",
    ]
)


FREQUENCY_MONTHS = {
    "1m": 1.0,
    "monthly": 1.0,
    "mensual": 1.0,
    "m": 1.0,
    "1 month": 1.0,
    "2m": 2.0,
    "2 months": 2.0,
    "3m": 3.0,
    "quarterly": 3.0,
    "trimestral": 3.0,
    "q": 3.0,
    "3 months": 3.0,
    "6m": 6.0,
    "6 months": 6.0,
    "1y": 12.0,
    "12m": 12.0,
    "annual": 12.0,
    "anual": 12.0,
    "y": 12.0,
    # Daily observations are modeled as roughly one trading day out of a 21-day month.
    "1d": 1.0 / 21.0,
}


def normalize_observation_frequency(value: object) -> float:
    """Return normalize observation frequency."""
    key = str(value).strip().lower()
    if key not in FREQUENCY_MONTHS:
        raise ValueError(f"Unsupported observation_frequency: {value!r}")
    return FREQUENCY_MONTHS[key]


def clean_observation_frequency(value: object) -> str:
    """Return clean observation frequency."""
    months = normalize_observation_frequency(value)
    if np.isclose(months, 1.0 / 21.0):
        return "daily"
    return f"{int(months)}m"


def parse_underlyings(value: str) -> list[str]:
    """Return parse underlyings."""
    return [part.strip() for part in str(value).split("|") if part.strip()]


def canonical_basket_signature(underlyings: list[str]) -> str:
    """Handle canonical basket signature."""
    return "|".join(sorted(underlyings))


def month_diff(start: pd.Series, end: pd.Series) -> pd.Series:
    """Handle month diff."""
    days = (end - start).dt.days
    return days / 30.4375


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Handle safe divide."""
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class FeatureSet:
    """Represent FeatureSet."""

    frame: pd.DataFrame
    numeric_features: list[str]
    categorical_features: list[str]
    target: pd.Series | None


class FeatureBuilder:
    """Build deterministic, leakage-safe model features."""

    def build(
        self,
        rfqs: pd.DataFrame,
        volatility: pd.DataFrame,
        reference: pd.DataFrame,
        include_target: bool = True,
    ) -> FeatureSet:
        """Perform build."""
        base = rfqs.reset_index(drop=True).copy()
        base["row_id"] = np.arange(len(base))
        base["requested_date"] = pd.to_datetime(base["requested_date"])
        base["start_date"] = pd.to_datetime(base["start_date"])
        base["end_date"] = pd.to_datetime(base["end_date"])
        base["observation_interval_months"] = base["observation_frequency"].map(
            normalize_observation_frequency
        )
        base["observation_frequency_clean"] = base["observation_frequency"].map(
            clean_observation_frequency
        )
        base["is_daily_observation"] = base["observation_frequency_clean"].eq("daily").astype(int)
        base["underlying_list"] = base["underlyings"].map(parse_underlyings)
        base["basket_size"] = base["underlying_list"].map(len)
        base["basket_signature"] = base["underlying_list"].map(canonical_basket_signature)
        underlying_sets = base["underlying_list"].map(set)
        basket_indicators = {
            f"underlying_{underlying}": underlying_sets.map(
                lambda underlyings, item=underlying: int(item in underlyings)
            )
            for underlying in KNOWN_UNDERLYINGS
        }
        basket_indicators.update(
            {
                f"pair_{left}_{right}": underlying_sets.map(
                    lambda underlyings, a=left, b=right: int(a in underlyings and b in underlyings)
                )
                for left, right in combinations(KNOWN_UNDERLYINGS, 2)
            }
        )
        base = pd.concat([base, pd.DataFrame(basket_indicators, index=base.index)], axis=1)
        base["primary_underlying"] = base["underlying_list"].map(
            lambda underlyings: underlyings[0] if len(underlyings) == 1 else "MULTI"
        )
        base["nominal_maturity_months"] = month_diff(base["start_date"], base["end_date"])
        base["requested_year"] = base["requested_date"].dt.year
        base["requested_month"] = base["requested_date"].dt.month
        base["requested_quarter"] = base["requested_date"].dt.quarter
        base["nominal_maturity_bucket"] = pd.cut(
            base["nominal_maturity_months"],
            bins=[-0.01, 24, 36, 60, 84, np.inf],
            labels=["0-24m", "24-36m", "36-60m", "60-84m", "84m+"],
        ).astype("string")

        ref_features = self._reference_aggregates(base, reference)
        vol_features = self._volatility_aggregates(base, volatility)
        features = base.join(ref_features, on="row_id").join(vol_features, on="row_id")
        features["implied_minus_realized_vol_mean"] = (
            features["quoted_implied_vol"] - features["realized_vol_63d_mean"]
        )
        for column in [
            "dominant_sector",
            "highest_vol_sector",
            "highest_structural_vol_underlying",
            "second_highest_structural_vol_underlying",
            "lowest_structural_vol_underlying",
            "highest_realized_vol_underlying",
            "second_highest_realized_vol_underlying",
            "lowest_realized_vol_underlying",
            "primary_underlying",
            "basket_signature",
            "nominal_maturity_bucket",
        ]:
            features[column] = features[column].fillna("UNKNOWN")
        features = self._add_derived_features(features)

        selected = features[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
        target = features[TARGET].astype(float) if include_target and TARGET in features else None
        return FeatureSet(
            frame=selected,
            numeric_features=NUMERIC_FEATURES.copy(),
            categorical_features=CATEGORICAL_FEATURES.copy(),
            target=target,
        )

    def _explode_underlyings(self, base: pd.DataFrame) -> pd.DataFrame:
        """Handle explode underlyings."""
        exploded = base[["row_id", "requested_date", "underlying_list"]].explode("underlying_list")
        exploded = exploded.rename(columns={"underlying_list": "underlying"})
        return exploded.dropna(subset=["underlying"])

    def _add_derived_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle add derived features."""
        features = features.copy()
        callable_months = (
            features["nominal_maturity_months"] - features["no_call_period_months"]
        ).clip(lower=0)

        features["barrier_gap_pct"] = (
            features["autocall_barrier_pct"] - features["protection_barrier_pct"]
        )
        features["autocall_barrier_above_par_pct"] = features["autocall_barrier_pct"] - 1.0
        features["protection_buffer_pct"] = 1.0 - features["protection_barrier_pct"]
        features["autocall_to_protection_ratio"] = safe_divide(
            features["autocall_barrier_pct"],
            features["protection_barrier_pct"],
        )
        features["no_call_fraction_of_maturity"] = safe_divide(
            features["no_call_period_months"],
            features["nominal_maturity_months"],
        )
        features["callable_maturity_months"] = callable_months
        features["no_call_observation_count"] = safe_divide(
            features["no_call_period_months"],
            features["observation_interval_months"],
        )
        features["estimated_observation_count"] = safe_divide(
            features["nominal_maturity_months"],
            features["observation_interval_months"],
        )
        features["post_no_call_observation_count"] = safe_divide(
            callable_months,
            features["observation_interval_months"],
        )
        features["is_worst_of"] = features["basket_type"].eq("worst_of").astype(int)
        features["is_single_underlying"] = features["basket_size"].eq(1).astype(int)
        features["basket_complexity_score"] = features["basket_size"] * features["is_worst_of"]
        features["worst_of_pressure"] = (
            features["basket_complexity_score"] * features["autocall_barrier_pct"]
        )
        features["worst_of_vol_pressure"] = (
            features["basket_complexity_score"] * features["structural_base_vol_max"]
        )
        features["structural_base_vol_top2_spread"] = (
            features["structural_base_vol_max"] - features["structural_base_vol_second_highest"]
        ).fillna(0.0)
        features["realized_vol_63d_top2_spread"] = (
            features["realized_vol_63d_max"] - features["realized_vol_63d_second_highest"]
        ).fillna(0.0)
        features["worst_of_structural_base_vol_range"] = (
            features["is_worst_of"] * features["structural_base_vol_range"]
        )
        features["worst_of_structural_vol_top2_spread"] = (
            features["is_worst_of"] * features["structural_base_vol_top2_spread"]
        )
        features["worst_of_realized_vol_63d_range"] = (
            features["is_worst_of"] * features["realized_vol_63d_range"]
        )
        features["worst_of_realized_vol_top2_spread"] = (
            features["is_worst_of"] * features["realized_vol_63d_top2_spread"]
        )
        features["sector_concentration"] = safe_divide(
            features["dominant_sector_count"],
            features["basket_size"],
        )
        features["has_multiple_sectors"] = features["sector_count"].gt(1).astype(int)
        features["log_notional_credits"] = np.log1p(features["notional_credits"])
        features["quoted_minus_structural_vol_mean"] = (
            features["quoted_implied_vol"] - features["structural_base_vol_mean"]
        )
        features["realized_minus_structural_vol_mean"] = (
            features["realized_vol_63d_mean"] - features["structural_base_vol_mean"]
        )
        features["realized_to_structural_vol_ratio"] = safe_divide(
            features["realized_vol_63d_mean"],
            features["structural_base_vol_mean"],
        )
        features["quoted_to_realized_vol_ratio"] = safe_divide(
            features["quoted_implied_vol"],
            features["realized_vol_63d_mean"],
        )
        features["quoted_to_structural_vol_ratio"] = safe_divide(
            features["quoted_implied_vol"],
            features["structural_base_vol_mean"],
        )
        features["realized_vol_dispersion_ratio"] = safe_divide(
            features["realized_vol_63d_range"],
            features["realized_vol_63d_mean"],
        )
        features["structural_vol_dispersion_ratio"] = safe_divide(
            features["structural_base_vol_range"],
            features["structural_base_vol_mean"],
        )
        return features.replace([np.inf, -np.inf], np.nan)

    def _reference_aggregates(self, base: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
        """Handle reference aggregates."""
        exploded = self._explode_underlyings(base).merge(reference, on="underlying", how="left")
        grouped = exploded.groupby("row_id", sort=False)
        vol = grouped["structural_base_vol"].agg(["mean", "min", "max"])
        vol.columns = [f"structural_base_vol_{column}" for column in vol.columns]
        vol["structural_base_vol_second_highest"] = grouped["structural_base_vol"].apply(
            self._second_highest
        )
        vol["structural_base_vol_range"] = (
            vol["structural_base_vol_max"] - vol["structural_base_vol_min"]
        )
        sector_stats = grouped["sector"].agg(
            sector_count=lambda s: s.dropna().nunique(),
            dominant_sector_count=lambda s: (
                s.dropna().value_counts().iloc[0] if not s.dropna().empty else 0
            ),
            dominant_sector=lambda s: (
                s.dropna().mode().iloc[0] if not s.dropna().mode().empty else "UNKNOWN"
            ),
        )
        structural_risk = self._structural_vol_rankings_by_row(exploded, vol.index)
        return vol.join(sector_stats).join(structural_risk)

    @staticmethod
    def _second_highest(values: pd.Series) -> float:
        """Handle second highest."""
        ordered = values.dropna().sort_values(ascending=False)
        if len(ordered) < 2:
            return np.nan
        return float(ordered.iloc[1])

    def _structural_vol_rankings_by_row(
        self,
        exploded: pd.DataFrame,
        row_index: pd.Index,
    ) -> pd.DataFrame:
        """Handle structural vol rankings by row."""
        valid = exploded.dropna(subset=["structural_base_vol", "sector"])
        if valid.empty:
            return pd.DataFrame(
                {
                    "highest_vol_sector": "UNKNOWN",
                    "highest_structural_vol_underlying": "UNKNOWN",
                    "second_highest_structural_vol_underlying": "UNKNOWN",
                    "lowest_structural_vol_underlying": "UNKNOWN",
                },
                index=row_index,
            )
        ranked = valid.sort_values(["row_id", "structural_base_vol"], ascending=[True, False])
        highest = ranked.drop_duplicates("row_id").set_index("row_id").reindex(row_index)
        second_underlying = (
            ranked.groupby("row_id", sort=False)["underlying"]
            .apply(lambda values: values.iloc[1] if len(values) > 1 else "UNKNOWN")
            .reindex(row_index)
        )
        lowest = (
            valid.sort_values(["row_id", "structural_base_vol"], ascending=[True, True])
            .drop_duplicates("row_id")
            .set_index("row_id")
            .reindex(row_index)
        )
        return pd.DataFrame(
            {
                "highest_vol_sector": highest["sector"].fillna("UNKNOWN"),
                "highest_structural_vol_underlying": highest["underlying"].fillna("UNKNOWN"),
                "second_highest_structural_vol_underlying": second_underlying.fillna("UNKNOWN"),
                "lowest_structural_vol_underlying": lowest["underlying"].fillna("UNKNOWN"),
            },
            index=row_index,
        )

    def _volatility_aggregates(
        self,
        base: pd.DataFrame,
        volatility: pd.DataFrame,
    ) -> pd.DataFrame:
        """Handle volatility aggregates."""
        exploded = self._explode_underlyings(base)
        pieces: list[pd.DataFrame] = []
        vol = self._volatility_history_features(volatility)
        asof_columns = [
            "date",
            "realized_vol_63d",
            "realized_vol_63d_21d_mean",
            "realized_vol_63d_126d_mean",
            "realized_vol_63d_trend_21d_126d",
            "realized_vol_63d_zscore_126d",
            "realized_vol_63d_change_21d",
        ]
        for underlying, left in exploded.groupby("underlying", sort=False):
            right = vol.loc[vol["underlying"] == underlying, asof_columns]
            left_sorted = left.sort_values("requested_date")
            right_sorted = right.sort_values("date")
            if right_sorted.empty:
                matched = left_sorted.assign(
                    **{column: np.nan for column in asof_columns if column != "date"}
                )
            else:
                matched = pd.merge_asof(
                    left_sorted,
                    right_sorted,
                    left_on="requested_date",
                    right_on="date",
                    direction="backward",
                )
            pieces.append(
                matched[
                    [
                        "row_id",
                        "underlying",
                        *[column for column in asof_columns if column != "date"],
                    ]
                ]
            )
        if not pieces:
            return pd.DataFrame(index=base["row_id"])
        matched_all = pd.concat(pieces, ignore_index=True)
        grouped = matched_all.groupby("row_id", sort=False)["realized_vol_63d"]
        out = grouped.agg(["mean", "min", "max"])
        out.columns = [f"realized_vol_63d_{column}" for column in out.columns]
        out["realized_vol_63d_second_highest"] = grouped.apply(self._second_highest)
        out["realized_vol_63d_range"] = out["realized_vol_63d_max"] - out["realized_vol_63d_min"]
        out = out.join(self._realized_vol_rankings_by_row(matched_all))
        out["basket_realized_vol_21d_mean"] = matched_all.groupby("row_id", sort=False)[
            "realized_vol_63d_21d_mean"
        ].mean()
        out["basket_realized_vol_21d_max"] = matched_all.groupby("row_id", sort=False)[
            "realized_vol_63d_21d_mean"
        ].max()
        out["basket_realized_vol_126d_mean"] = matched_all.groupby("row_id", sort=False)[
            "realized_vol_63d_126d_mean"
        ].mean()
        out["basket_realized_vol_126d_max"] = matched_all.groupby("row_id", sort=False)[
            "realized_vol_63d_126d_mean"
        ].max()
        out["basket_realized_vol_trend_mean"] = matched_all.groupby("row_id", sort=False)[
            "realized_vol_63d_trend_21d_126d"
        ].mean()
        out["basket_realized_vol_trend_max"] = matched_all.groupby("row_id", sort=False)[
            "realized_vol_63d_trend_21d_126d"
        ].max()
        out["basket_realized_vol_zscore_mean"] = matched_all.groupby("row_id", sort=False)[
            "realized_vol_63d_zscore_126d"
        ].mean()
        out["basket_realized_vol_zscore_max"] = matched_all.groupby("row_id", sort=False)[
            "realized_vol_63d_zscore_126d"
        ].max()
        out["basket_realized_vol_change_21d_mean"] = matched_all.groupby("row_id", sort=False)[
            "realized_vol_63d_change_21d"
        ].mean()
        out["basket_realized_vol_change_21d_max"] = matched_all.groupby("row_id", sort=False)[
            "realized_vol_63d_change_21d"
        ].max()
        return out

    def _realized_vol_rankings_by_row(self, matched_all: pd.DataFrame) -> pd.DataFrame:
        """Handle realized vol rankings by row."""
        valid = matched_all.dropna(subset=["realized_vol_63d", "underlying"])
        row_index = matched_all["row_id"].drop_duplicates()
        if valid.empty:
            return pd.DataFrame(
                {
                    "highest_realized_vol_underlying": "UNKNOWN",
                    "second_highest_realized_vol_underlying": "UNKNOWN",
                    "lowest_realized_vol_underlying": "UNKNOWN",
                },
                index=row_index,
            )
        ranked = valid.sort_values(["row_id", "realized_vol_63d"], ascending=[True, False])
        highest = ranked.drop_duplicates("row_id").set_index("row_id").reindex(row_index)
        second_underlying = (
            ranked.groupby("row_id", sort=False)["underlying"]
            .apply(lambda values: values.iloc[1] if len(values) > 1 else "UNKNOWN")
            .reindex(row_index)
        )
        lowest = (
            valid.sort_values(["row_id", "realized_vol_63d"], ascending=[True, True])
            .drop_duplicates("row_id")
            .set_index("row_id")
            .reindex(row_index)
        )
        return pd.DataFrame(
            {
                "highest_realized_vol_underlying": highest["underlying"].fillna("UNKNOWN"),
                "second_highest_realized_vol_underlying": second_underlying.fillna("UNKNOWN"),
                "lowest_realized_vol_underlying": lowest["underlying"].fillna("UNKNOWN"),
            },
            index=row_index,
        )

    def _volatility_history_features(self, volatility: pd.DataFrame) -> pd.DataFrame:
        """Handle volatility history features."""
        vol = volatility.copy()
        vol["date"] = pd.to_datetime(vol["date"])
        vol = vol.sort_values(["underlying", "date"])
        grouped = vol.groupby("underlying", sort=False)["realized_vol_63d"]
        vol["realized_vol_63d_21d_mean"] = grouped.transform(
            lambda s: s.rolling(window=21, min_periods=1).mean()
        )
        vol["realized_vol_63d_126d_mean"] = grouped.transform(
            lambda s: s.rolling(window=126, min_periods=1).mean()
        )
        vol["realized_vol_63d_126d_std"] = grouped.transform(
            lambda s: s.rolling(window=126, min_periods=2).std()
        )
        vol["realized_vol_63d_trend_21d_126d"] = (
            vol["realized_vol_63d_21d_mean"] - vol["realized_vol_63d_126d_mean"]
        )
        vol["realized_vol_63d_zscore_126d"] = safe_divide(
            vol["realized_vol_63d"] - vol["realized_vol_63d_126d_mean"],
            vol["realized_vol_63d_126d_std"],
        )
        vol["realized_vol_63d_change_21d"] = vol["realized_vol_63d"] - grouped.shift(21)
        return vol


def select_feature_block(block_name: str) -> tuple[list[str], list[str]]:
    """Return select feature block."""
    if block_name not in FEATURE_BLOCKS:
        raise ValueError(
            f"Unknown feature block {block_name!r}. Valid blocks: {sorted(FEATURE_BLOCKS)}"
        )
    features = FEATURE_BLOCKS[block_name]
    numeric = [feature for feature in NUMERIC_FEATURES if feature in features]
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in features]
    return numeric, categorical
