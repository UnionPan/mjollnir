"""Execute every example as a test so the gallery cannot rot."""

import pathlib
import runpy

import pytest

EXAMPLES = sorted(
    (pathlib.Path(__file__).parent.parent / "examples").glob("*.py"))


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(script):
    runpy.run_path(str(script), run_name="__main__")
