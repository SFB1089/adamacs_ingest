import sys
import types

import pytest


if "datajoint" not in sys.modules:
    dj_stub = types.ModuleType("datajoint")
    dj_stub.config = {}
    sys.modules["datajoint"] = dj_stub

from adamacs.schemas import analysis
import adamacs.helpers.trace_helpers as trace_helpers


def test_analysis_schema_shim_is_importable():
    assert hasattr(analysis, "ANALYSIS_EXTENSION_AVAILABLE")


def test_trace_helper_shim_behavior():
    if trace_helpers.ANALYSIS_HELPERS_AVAILABLE:
        assert hasattr(trace_helpers, "FilterEvents")
    else:
        with pytest.raises(ModuleNotFoundError, match="adamacs_analysis"):
            _ = trace_helpers.FilterEvents
