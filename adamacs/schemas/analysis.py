"""Compatibility shim for analysis schema moved to `adamacs_analysis`."""

from importlib import import_module

ANALYSIS_EXTENSION_AVAILABLE = False

try:
    _analysis_module = import_module("adamacs_analysis.schemas.analysis")
except ModuleNotFoundError:
    _analysis_module = None
else:
    ANALYSIS_EXTENSION_AVAILABLE = True
    for _name in dir(_analysis_module):
        if not _name.startswith("_"):
            globals()[_name] = getattr(_analysis_module, _name)
