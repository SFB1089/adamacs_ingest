"""Compatibility shim for `adamacs_analysis.helpers.arena_render`."""

from importlib import import_module

ANALYSIS_HELPERS_AVAILABLE = False

try:
    _helpers_module = import_module("adamacs_analysis.helpers.arena_render")
except ModuleNotFoundError as _missing:
    _helpers_module = None
    _missing_module = _missing
else:
    ANALYSIS_HELPERS_AVAILABLE = True
    for _name in dir(_helpers_module):
        if not _name.startswith("_"):
            globals()[_name] = getattr(_helpers_module, _name)


def __getattr__(name):
    if _helpers_module is None:
        raise ModuleNotFoundError(
            "Arena rendering helpers moved to `adamacs_analysis`. "
            "Install `adamacs_analysis` to use the Blender rendering utilities."
        ) from _missing_module
    return getattr(_helpers_module, name)
