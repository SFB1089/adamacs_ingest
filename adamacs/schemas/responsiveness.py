"""Responsiveness schema, phase 1: dF/F0 as a stored, parameterised table.

Design: ``docs/RESPONSIVENESS_SCHEMA_PROPOSAL.md``. This module holds the first block of
that design -- the traces -- and the link from the stored CASCADE activities back to the
dF/F0 they were computed from:

    DffParamSet (Lookup)  one row per dF/F0 recipe, one typed column per step
    Dff0Task    (Manual)  which curation x recipe to compute
    Dff0        (Computed) + Dff0.Trace   the dF/F0 of every mask, stored once
    CascadeInput(Computed) each stored CASCADE activity -> the recipe it was computed from (dated)
    CascadeRecompute (Manual) CASCADE re-run on Dff0 vs the stored activity (verification)

Why dF/F0 is a table
--------------------
Until now the dF/F0 was computed inside ``imaging.Activity.make`` for CASCADE and thrown
away; its parameters were rules on the extraction-method *name*; the dark-signal code
changed in place on BUGPU on 2026-09-03 without any record, so one extraction method holds
rows from two code versions. Here every choice is a column, the dark window actually used
is stored per scan, and the computation is one pure function
(:func:`adamacs.responsiveness.dff0.compute_dff0`).

Nothing is declared on import
-----------------------------
As in ``cell_registration.py``: call :func:`activate` explicitly. The upstream modules are
passed in (real ``adamacs.pipeline`` modules or DataJoint virtual modules), so the schema
can be tested under a personal prefix without importing ``adamacs.pipeline`` (which writes
on import). The external store for the traces is also an argument, so a test schema can
write to a scratch store instead of the production ``external-raw``.
"""
from __future__ import annotations

import datetime as _dt
import os as _os
import subprocess as _subprocess

import datajoint as dj
import numpy as _np

from adamacs.responsiveness import dff0 as _dff0

schema = dj.Schema()

#: Upstream modules, set by activate(): {'imaging', 'scan', 'event'}
_UP = {}

#: Store name substituted into the blob attributes at activation.
_STORE = {"name": "external-raw"}


def _code_version() -> str:
    """Git commit of this checkout (+ '-dirty'), or 'unknown'."""
    here = _os.path.dirname(_os.path.abspath(__file__))
    try:
        sha = _subprocess.check_output(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                                       stderr=_subprocess.DEVNULL).decode().strip()
        dirty = _subprocess.call(["git", "-C", here, "diff", "--quiet", "--", "."],
                                 stderr=_subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _up(name):
    if name not in _UP:
        raise dj.DataJointError(
            "responsiveness.activate() has not been called, so the upstream modules are "
            "unknown. Call activate(schema_name, imaging=..., scan=..., event=...) first.")
    return _UP[name]


# ======================================================================================
# Block A: traces
# ======================================================================================

@schema
class DffParamSet(dj.Lookup):
    definition = """
    # One dF/F0 recipe. Steps run in the order of the column groups. Same recipe -> same hash. Never edited.
    dff_paramset_idx       : smallint
    ---
    dff_desc               : varchar(255)   # plain language
    dff_hash               : uuid           # hash of all recipe columns (normalised)
    unique index (dff_hash)
    # 1 dark signal
    dark_method            : varchar(16)    # 'none' | 'frames' (fixed frame range) | 'shutter' (shutter-closed window)
    dark_frame_start=null  : int            # 'frames': first frame; 'shutter': first fallback frame
    dark_frame_stop=null   : int            # last frame (exclusive)
    dark_margin_s=null     : float          # [s] 'shutter': margin inside the shutter-closed interval
    # 2 neuropil
    neuropil_factor        : float          # a in F - a*Fneu (0 = no neuropil subtraction)
    neuropil_offset        : varchar(8)     # added back: 'none' | 'median' (+a*median(Fneu)) | 'mean' (+a*mean(Fneu))
    # 3 detrending before the baseline
    detrend_method         : varchar(16)    # 'none' | 'linear' | 'highpass'
    detrend_window_s=null  : float          # [s] 'highpass': moving-average window removed
    # 4 baseline F0
    baseline_method        : varchar(24)    # 'running_percentile' | 'maximin' | 'global_percentile'
    baseline_percentile=null : float        # percentile for running/global percentile
    baseline_window_s=null : float          # [s] running window
    baseline_smooth_s=null : float          # [s] gaussian pre-smoothing sigma ('maximin')
    baseline_step=1        : smallint       # running filter on every k-th frame, interpolated (1 = exact)
    # 5 F0 floor
    f0_floor_method        : varchar(24)    # 'none' | 'flag' | 'nan' | 'fraction_of_median_F' | 'quantile_of_F0'
    f0_floor_value=null    : float          # fraction or quantile for the floor methods
    # anything a future method needs (part of the hash)
    extra=null             : longblob
    """

    @classmethod
    def insert_new_params(cls, dff_paramset_idx: int, dff_desc: str, **params):
        """Validate, normalise and hash a recipe, then insert it.

        Refuses a recipe that already exists under another index (same hash), and an index
        that is taken by a different recipe. Re-inserting an identical row is a no-op.
        """
        rec = _dff0.normalise_recipe(params)
        h = _dff0.recipe_hash(rec)
        same_hash = (cls & {"dff_hash": h}).fetch("dff_paramset_idx")
        if len(same_hash) and int(same_hash[0]) != int(dff_paramset_idx):
            raise dj.DataJointError(
                f"this recipe already exists as dff_paramset_idx={int(same_hash[0])}")
        taken = cls & {"dff_paramset_idx": dff_paramset_idx}
        if taken and taken.fetch1("dff_hash") != h:
            raise dj.DataJointError(f"dff_paramset_idx={dff_paramset_idx} holds a different recipe")
        cls.insert1({"dff_paramset_idx": dff_paramset_idx, "dff_desc": dff_desc, "dff_hash": h, **rec},
                    skip_duplicates=True)

    @classmethod
    def insert_known_recipes(cls):
        """Insert the recipes found in the lab's code (proposal §4.2, rows 1-10)."""
        for idx, (desc, rec) in _dff0.KNOWN_RECIPES.items():
            cls.insert_new_params(idx, desc, **rec)

    @classmethod
    def recipe(cls, key) -> dict:
        """The recipe columns of one row, as the dict compute_dff0 takes."""
        row = (cls & key).fetch1()
        return {k: row[k] for k in _dff0.RECIPE_COLUMNS}


@schema
class Dff0Task(dj.Manual):
    definition = """
    # Compute dF/F0 for this curation with this recipe. Inserted by a person or helper; populated by a worker.
    -> imaging.Fluorescence
    -> DffParamSet
    ---
    task_owner           : varchar(32)
    task_note=''         : varchar(255)
    """


@schema
class Dff0(dj.Computed):
    definition = """
    # dF/F0 of every mask of one curation under one recipe, computed once and stored
    -> Dff0Task
    ---
    fps                  : float          # [Hz] scan.ScanInfo.fps used for the windows
    n_masks              : int
    n_frames             : int
    dark_signal          : float          # [a.u.] subtracted from F and Fneu
    dark_frame_start=null: int            # first frame of the dark window actually used
    dark_frame_stop=null : int            # last frame (exclusive)
    dark_source          : varchar(16)    # 'shutter' | 'frames' | 'fallback_frames' (no usable shutter events) | 'none'
    n_f0_nonpositive     : int            # masks whose F0 is <= 0 somewhere
    code_version         : varchar(64)    # git commit of adamacs_ingest
    compute_seconds      : float
    """

    class Trace(dj.Part):
        definition = """
        # dF/F0 of one mask
        -> master
        -> imaging.Fluorescence.Trace
        ---
        dff                  : blob@{store}   # float32 dF/F0 per frame
        f0_median            : float          # [a.u.] median baseline
        f0_min               : float          # [a.u.] minimum baseline
        f0_nonpositive       : bool           # F0 <= 0 somewhere: dF/F0 is sign-inverted there
        """

    def make(self, key):
        imaging, scan, event = _up("imaging"), _up("scan"), _up("event")
        t0 = _dt.datetime.now()
        scan_key = {"session_id": key["session_id"], "scan_id": key["scan_id"]}
        rows = (imaging.Fluorescence.Trace & key).fetch(
            "KEY", "fluorescence", "neuropil_fluorescence", order_by="mask, fluo_channel")
        keys, F, Fneu = rows[0], _np.vstack(rows[1]), _np.vstack(rows[2])
        fps = float((scan.ScanInfo & scan_key).fetch1("fps"))
        rec = DffParamSet.recipe(key)
        frame_times = starts = ends = None
        if rec["dark_method"] == "shutter":
            frame_times = (event.Event & scan_key & 'event_type LIKE "%%2p_frames%%"').fetch(
                "event_start_time", order_by="event_start_time").astype(float)
            sh = event.Event & scan_key & "event_type='shutter'"
            starts, ends = sh.fetch("event_start_time", "event_end_time", order_by="event_start_time")
            starts = _np.asarray(starts, dtype=float)
            ends = _np.asarray([_np.nan if e is None else float(e) for e in ends])
        out = _dff0.compute_dff0(F, Fneu, fps, rec, frame_times, starts, ends, n_jobs=32)
        self.insert1({
            **key, "fps": fps, "n_masks": int(F.shape[0]), "n_frames": int(F.shape[1]),
            "dark_signal": out["dark_signal"], "dark_frame_start": out["dark_frame_start"],
            "dark_frame_stop": out["dark_frame_stop"], "dark_source": out["dark_source"],
            "n_f0_nonpositive": int(out["f0_nonpositive"].sum()), "code_version": _code_version(),
            "compute_seconds": (_dt.datetime.now() - t0).total_seconds()})
        self.Trace.insert(
            {**key, **k, "dff": out["dff"][i], "f0_median": float(out["f0_median"][i]),
             "f0_min": float(out["f0_min"][i]), "f0_nonpositive": bool(out["f0_nonpositive"][i])}
            for i, k in enumerate(keys))


# ======================================================================================
# The link from the stored CASCADE activities to their input
# ======================================================================================

@schema
class CascadeInput(dj.Computed):
    definition = """
    # The dF/F0 recipe a stored CASCADE activity was computed from, dated from its first stored trace
    -> imaging.Activity
    ---
    -> DffParamSet                        # the recipe of the code version that wrote the row
    written_at           : datetime       # external-store timestamp of the earliest trace blob of this activity
    code_era             : enum('before_2026_09_03','since_2026_09_03')  # BUGPU element imaging.py edited in place 2026-09-03 15:22
    """

    @property
    def key_source(self):
        imaging = _up("imaging")
        methods = list(_dff0.CASCADE_METHOD_RECIPES)
        return imaging.Activity & [{"extraction_method": m} for m in methods]

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
    # CASCADE re-run on the stored dF/F0 of the recipe, compared with the stored activity (inserted by the verification script)
    -> CascadeInput
    ---
    -> Dff0                               # the dF/F0 the re-run used (same curation, recipe of CascadeInput)
    model_name           : varchar(64)    # CASCADE model, from imaging.ActivityCascadeTask
    cascade_commit       : varchar(64)    # git commit of the Cascade checkout that ran it
    run_host             : varchar(32)
    n_masks_checked      : int
    max_abs_diff         : float          # [spikes/frame] max |stored - recomputed| over checked masks and frames
    median_abs_diff      : float          # [spikes/frame]
    min_corr             : float          # smallest per-mask Pearson r, stored vs recomputed
    checked_at           : datetime
    """


# ======================================================================================

def activate(schema_name: str, *, imaging, scan, event, store: str = "external-raw",
             create_schema: bool = True, create_tables: bool = True):
    """Activate the schema. Nothing is declared before this call.

    ``imaging``, ``scan``, ``event``: the upstream modules (``adamacs.pipeline`` modules or
    ``dj.create_virtual_module`` objects). ``store``: the external store for the traces.
    """
    _UP.update(imaging=imaging, scan=scan, event=event)
    _STORE["name"] = store
    Dff0.Trace.definition = Dff0.Trace.definition.replace("{store}", store)
    schema.activate(schema_name, create_schema=create_schema, create_tables=create_tables,
                    add_objects={"imaging": imaging, "scan": scan, "event": event})
    return schema
