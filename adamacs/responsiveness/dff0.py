"""dF/F0 recipes: re-export of ``element_calcium_imaging.dff`` (the single implementation).

The code that lived here (adamacs_ingest PR #25) moved into the SFB1089 element fork, because
``imaging.Activity.make`` (CASCADE) now reads ``imaging.Dff0`` and the element must not import
``adamacs``. This module keeps ``from adamacs.responsiveness import dff0`` working.
"""
from element_calcium_imaging.dff import *  # noqa: F401,F403
from element_calcium_imaging.dff import (  # noqa: F401  (names used across adamacs)
    CASCADE_CODE_CHANGE,
    CASCADE_METHOD_RECIPES,
    CASCADE_RECIPES,
    KNOWN_RECIPES,
    RECIPE_COLUMNS,
    REGISTRY,
    compute_dff0,
    element_version,
    normalise_recipe,
    recipe_hash,
    resolve_dark,
    shutter_dark_frames,
)
