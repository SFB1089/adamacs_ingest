"""Session-wide fixtures for the ingest tests.

Several test modules guard their imports with

    if "datajoint" not in sys.modules:
        sys.modules["datajoint"] = <stub with a config dict>

so that they can run without DataJoint installed. Whichever of them pytest collects
first wins, and since collection is alphabetical the stub then leaks into every later
module -- including the ones that need the real package, which fail with
``AttributeError: module 'datajoint' has no attribute 'schema'`` during collection
rather than skipping.

Importing the real package here, before any test module is loaded, makes those guards
no-ops and gives every module the same DataJoint. Importing it does not open a
connection; only schema activation does.
"""

import sys
import types


def _ensure_datajoint():
    if "datajoint" in sys.modules:
        return
    try:
        import datajoint  # noqa: F401
    except Exception:
        stub = types.ModuleType("datajoint")
        stub.config = {}
        sys.modules["datajoint"] = stub


_ensure_datajoint()
