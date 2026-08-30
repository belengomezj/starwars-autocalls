from pathlib import Path

from starwars_autocalls.features import (
    CATEGORICAL_FEATURES,
    FEATURE_BLOCKS,
    NUMERIC_FEATURES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_feature_catalog_covers_all_blocks_and_named_features() -> None:
    catalog = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            PROJECT_ROOT / "docs" / "data" / "preprocessing.md",
            PROJECT_ROOT / "docs" / "includes" / "feature-catalog.md",
            PROJECT_ROOT / "docs" / "includes" / "feature-blocks.md",
        ]
    )

    undocumented_blocks = [block for block in FEATURE_BLOCKS if block not in catalog]
    named_features = [
        feature
        for feature in [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
        if not feature.startswith(("underlying_", "pair_"))
    ]
    undocumented_features = [feature for feature in named_features if feature not in catalog]

    assert undocumented_blocks == []
    assert undocumented_features == []
    assert "underlying_<ID>" in catalog
    assert "pair_<ID1>_<ID2>" in catalog
