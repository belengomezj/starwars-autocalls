from starwars_autocalls.cli.workflow_commands import _full_suite_steps
from starwars_autocalls.config import Settings


def test_full_suite_is_complete_and_excludes_baselines() -> None:
    steps = _full_suite_steps(Settings())
    commands = [step.arguments[0] for step in steps]
    assert commands[-2:] == ["train", "evaluate"]
    assert "benchmark" in commands
    assert "tune" in commands
    assert "tune-segmented" in commands
    benchmark = next(step for step in steps if step.arguments[0] == "benchmark")
    model_names = benchmark.arguments[2].split(",")
    assert "global_mean" not in model_names
    assert not any(name.startswith("median_by_") for name in model_names)
