"""Responsiveness schema: provenance of the legacy CASCADE activities (and, later, responses).

Design: ``docs/RESPONSIVENESS_SCHEMA_PROPOSAL.md``. dF/F0 itself now lives in ``imaging``
(SFB1089 element fork >= 0.1.0b1: ``imaging.DffParamSet``, ``imaging.Dff0``,
``imaging.CascadeRecipe``, ``imaging.Activity.CascadeInput``; see
``docs/DFF0_IN_ACTIVITY_MAKE_PROPOSAL.md``), because CASCADE in ``imaging.Activity.make``
reads it. This schema keeps what ``imaging`` cannot know about rows written before that
change:

    CascadeInput     (Computed) each legacy CASCADE activity -> the dF/F0 recipe of the code that wrote it,
                                dated from the external-store timestamp (BUGPU code edited in place 2026-09-03)
    CascadeRecompute (Manual)   CASCADE re-run on imaging.Dff0 vs the stored activity (verification)

Activities computed by the new code carry exact provenance in ``imaging.Activity.CascadeInput``
and are excluded here.

Nothing is declared on import
-----------------------------
As in ``cell_registration.py``: call :func:`activate` explicitly. The upstream modules are
passed in (``adamacs.pipeline`` modules or DataJoint virtual modules), so the schema can be
used without importing ``adamacs.pipeline`` (which writes on import).
"""
from __future__ import annotations

import datetime as _dt

import datajoint as dj

from adamacs.responsiveness import dff0 as _dff0

schema = dj.Schema()

#: Upstream modules, set by activate(): {'imaging', 'scan', 'event'}
_UP = {}


def _up(name):
    if name not in _UP:
        raise dj.DataJointError(
            "responsiveness.activate() has not been called, so the upstream modules are "
            "unknown. Call activate(schema_name, imaging=..., scan=..., event=...) first.")
    return _UP[name]


@schema
class CascadeInput(dj.Computed):
    definition = """
    # The dF/F0 recipe a legacy CASCADE activity was computed from, dated from its first stored trace
    -> imaging.Activity
    ---
    -> imaging.DffParamSet                # the recipe of the code version that wrote the row
    written_at           : datetime       # external-store timestamp of the earliest trace blob of this activity
    code_era             : enum('before_2026_09_03','since_2026_09_03')  # BUGPU element imaging.py edited in place 2026-09-03 15:22
    """

    @property
    def key_source(self):
        imaging = _up("imaging")
        legacy = imaging.Activity & [{"extraction_method": m} for m in _dff0.CASCADE_METHOD_RECIPES]
        return legacy - imaging.Activity.CascadeInput     # new-code rows have exact provenance

    def make(self, key):
        imaging = _up("imaging")
        ext = dj.Schema(imaging.schema.database, create_schema=False,
                        create_tables=False).external["external-raw"]
        q = (imaging.Activity.Trace & key).proj("activity_trace") * ext.proj(
            activity_trace="hash", written="timestamp")
        written = dj.U().aggr(q, first="min(written)").fetch1("first")
        before, since = _dff0.CASCADE_METHOD_RECIPES[key["extraction_method"]]
        era_before = written < _dt.datetime.fromisoformat(_dff0.CASCADE_CODE_CHANGE)
        idx = before if era_before else since
        if idx is None:
            raise dj.DataJointError(
                f"no recipe registered for {key['extraction_method']} written {written}")
        self.insert1({**key, "dff_paramset_idx": idx, "written_at": written,
                      "code_era": "before_2026_09_03" if era_before else "since_2026_09_03"})


@schema
class CascadeRecompute(dj.Manual):
    definition = """
    # CASCADE re-run on imaging.Dff0 of the recipe, compared with the stored activity (inserted by the verification script)
    -> CascadeInput
    ---
    -> imaging.Dff0                       # the dF/F0 the re-run used (same curation, recipe of CascadeInput)
    model_name           : varchar(64)    # CASCADE model, from imaging.ActivityCascadeTask
    cascade_commit       : varchar(64)    # git commit of the Cascade checkout that ran it
    run_host             : varchar(32)
    n_masks_checked      : int
    max_abs_diff         : float          # [spikes/frame] max |stored - recomputed| over checked masks and frames
    median_abs_diff      : float          # [spikes/frame]
    min_corr             : float          # smallest per-mask Pearson r, stored vs recomputed
    checked_at           : datetime
    """


def activate(schema_name: str, *, imaging, scan, event,
             create_schema: bool = True, create_tables: bool = True):
    """Activate the schema. Nothing is declared before this call.

    ``imaging``, ``scan``, ``event``: the upstream modules (``adamacs.pipeline`` modules or
    ``dj.create_virtual_module`` objects). ``imaging`` must contain ``DffParamSet`` and ``Dff0``
    (element fork >= 0.1.0b1).
    """
    for name in ("DffParamSet", "Dff0"):
        if not hasattr(imaging, name):
            raise dj.DataJointError(
                f"imaging.{name} is missing: the imaging schema predates element-calcium-imaging 0.1.0b1")
    _UP.update(imaging=imaging, scan=scan, event=event)
    schema.activate(schema_name, create_schema=create_schema, create_tables=create_tables,
                    add_objects={"imaging": imaging, "scan": scan, "event": event})
    return schema
