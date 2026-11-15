"""adamacs.responsiveness.dff0 re-exports element_calcium_imaging.dff.

The dF/F0 tests themselves live with the implementation in the SFB1089 element fork
(tests/test_dff.py). Here only the re-export is checked; CI without the element skips.
"""
import pytest

pytest.importorskip("numpy", reason="needs numpy")
pytest.importorskip("scipy", reason="needs scipy")
ecd = pytest.importorskip("element_calcium_imaging.dff", reason="needs element-calcium-imaging >= 0.1.0b1")

from adamacs.responsiveness import dff0  # noqa: E402


def test_reexport_is_the_element_implementation():
    assert dff0.compute_dff0 is ecd.compute_dff0
    assert dff0.KNOWN_RECIPES is ecd.KNOWN_RECIPES
    assert dff0.CASCADE_METHOD_RECIPES is ecd.CASCADE_METHOD_RECIPES
