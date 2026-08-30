from __future__ import annotations

import pytest

from starwars_autocalls.modeling.training import _resolve_explicit_spec


def test_explicit_model_name_resolves_to_a_model_spec() -> None:
    spec = _resolve_explicit_spec("hist_gradient_boosting__all_without_noise", None, None)

    assert spec is not None
    assert spec.name == "hist_gradient_boosting__all_without_noise"
    assert spec.feature_block == "all_without_noise"


def test_model_and_feature_block_resolve_to_a_model_spec() -> None:
    spec = _resolve_explicit_spec(None, "hist_gradient_boosting", "compact_core")

    assert spec is not None
    assert spec.name == "hist_gradient_boosting__compact_core"


@pytest.mark.parametrize(
    ("model_name", "model", "feature_block"),
    [
        ("hist_gradient_boosting__all_without_noise", "ridge", "product"),
        (None, "ridge", None),
        (None, None, "product"),
    ],
)
def test_explicit_model_options_must_be_unambiguous(
    model_name: str | None,
    model: str | None,
    feature_block: str | None,
) -> None:
    with pytest.raises(ValueError):
        _resolve_explicit_spec(model_name, model, feature_block)
