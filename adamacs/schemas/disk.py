"""
DISK (Deep Imputation of SKeleton data) integration schema.

Fills in DeepLabCut pupil markers that fell below the production likelihood threshold,
so that frames which currently yield no eye angle yield one. Reference:
https://github.com/bozeklab/DISK. The evaluation that justifies it is at
https://github.com/SFB1089/disk-pupil-imputation, and the design review that produced
the shapes below is ``DISK_SCHEMA_DESIGN_REVIEW.md``, beside this file.

Nothing is declared on import
-----------------------------
This module uses **deferred activation**, the same pattern as the ``element_*`` schemas
in ``adamacs/pipeline.py``. Importing it creates no tables; someone must call
:func:`activate` explicitly. That matters here for a specific reason: the previous version
declared its schema at import time, and because ``adamacs/pipeline.py`` imports this
module, the tables were created the first time anybody imported the pipeline. Nobody ran
a migration -- the schema simply appeared. Deferred activation makes that a decision
instead of a side effect.

Stale tables still on the server -- TO BE DELETED
-------------------------------------------------
``roselab_disk`` currently holds tables that this module no longer defines. They are all
**empty**, nothing reads them, and they are harmless where they sit -- but none of them is
the schema below, and they should not be mistaken for it.

**Delete when convenient (no hurry, they hold nothing):**

    mocap_imputation_task
    __mocap_imputation
    __mocap_imputation__tracking_position_d_i_s_k
    __mocap_imputation__rigid_body_position_d_i_s_k

The motion-capture branch was removed on 2026-09-13 because its shape was wrong rather than
merely premature. OptiTrack rigid bodies are *solved from* marker sets, so imputing
``qw/qx/qy/qz`` independently inverts the dependency and does not preserve unit norm -- a
re-solve from imputed markers belongs downstream in ``mocap``, not in an imputation output
table. And OptiTrack's dominant failure is marker identity swaps rather than dropout, which
imputation smooths over instead of fixing. Mocap imputation should return with its own
evaluation and its own table design; DISK's published benchmarks are motion capture, so the
ambition is sound even though these four tables were not.

**Superseded, pending migration:**

    #d_i_s_k_model            (old shape: n_keypoints, no Keypoint/TrainingAnimal parts)
    d_l_c_imputation_task     (old key: no parameter_id, no imputation_paramset_id)
    __d_l_c_imputation        (old counters)
    __d_l_c_imputation__body_part_position_d_i_s_k
                              (old: boolean imputed_mask, original_likelihood,
                               per-frame uncertainty, all in-MySQL blobs)

These four are replaced by the definitions below the first time someone runs
:func:`activate` against a schema they have been dropped from. Until then the code and the
server disagree, which is safe only because every one of them is empty.

**Why they keep coming back.** They have been dropped twice and re-declared twice, both
times by a machine importing ``adamacs.pipeline`` from a checkout that still had the old
module-level ``@schema``. ``roselab_disk.~log`` records it: once from ``ibehavegpu1`` at
20:55 local on 2026-09-13, and again from ``tatchu3`` at 21:45, fifteen minutes after the
deferred-activation fix was merged and deployed *on ibehavegpu1 only*. Deferred activation
stops a machine re-declaring these tables; it stops it only on a checkout that has pulled
the change. **So dropping them sticks only once every machine that imports this pipeline is
up to date** -- otherwise they reappear on the next import, which for the population cron is
within five minutes.

Table hierarchy
---------------
Declared into ``<prefix>disk``::

    DISKModel (Lookup)                 the trained artefact, identified rather than described
      |- .Keypoint (Part)              ordered marker contract  -- which body part is column i
      \\- .TrainingAnimal (Part)        which animals the model saw -- makes "held out" checkable

    ImputationParamSet (Lookup)        refusal policy and gap ceiling, hashed

    DLCImputationTask (Manual) -> DLCImputation (Computed)
      |- .BodyPart (Part)              per-marker arrays, on the external store
      \\- .Gap (Part)                   one row per imputed segment -- scalar, stays in MySQL

Declared into ``<prefix>pupil_tracking``, because they are statements about the eye data
rather than about DISK::

    PupilFrameQuality (Computed)       classify before imputing: which frames may be filled
    PupilEllipseFittingImputed (Computed)   the A1 sibling of PupilEllipseFittingFreeMoving

Which schema a table lands in is decided by ``activate()``, not by which file the class
body sits in, so moving these two class definitions into ``pupil_tracking.py`` later is a
pure code move with no schema consequence.

Design decisions worth not re-litigating
----------------------------------------
Each of these was a finding in the design review; the short form is here so the reasoning
travels with the code.

**The parameter set is in the primary key.** ``llh_thres_pupil`` decides which samples DISK
is asked to fill, so it changes the output completely. The live database has three
parameter sets in active use (0.9, 0.9, 0.5). Without ``parameter_id`` in the key, two runs
at different thresholds collide on the same row. Keying on it also lines the imputation up
with the downstream ellipse fit, so the join is trivial.

**There is an effective likelihood, not just the original one.** The production fit masks
with ``likelihood < llh_thres_pupil``. Every imputed sample is by construction one whose
*original* likelihood was below that threshold, so a table storing only
``original_likelihood`` would have its imputations masked straight back out and change
nothing downstream. ``original_likelihood`` is not stored at all -- it is a verbatim copy
of a column in the foreign-keyed parent, recoverable by join.

**Sample provenance is a code, not a boolean.** A sample is one of five things after
imputation, and the evaluation found all five worth distinguishing -- notably "refused"
against "unfilled", which are different failures with different remedies. See
:data:`SAMPLE_SOURCE`. Note that the *justification* originally given for the
``dlc_low_kept`` state has since been withdrawn; the state is still needed, but keeping a
sub-threshold point is a per-sample judgement, not a free win. See the note on
``dlc_low_kept`` in :data:`SAMPLE_SOURCE`.

**Uncertainty lives on gaps, not on frames.** DISK collapses its per-sample sigma to one
scalar per imputed segment before exposing it. A per-frame array would be a claim the data
does not support, so it is stored in ``DLCImputation.Gap``. Keeping that table scalar also
makes it a queryable index into data that otherwise lives on a file server.

**Heavy arrays go to the external store.** ``blob@external-raw`` is already configured and
already used by ``mocap`` and ``wfield``. Roughly 7 GB per variant stays out of a MySQL
instance that already carries 40 GB of pose blobs.

**The marker order is a stored contract.** Three orders are in play and no two are
interchangeable: alphabetical (what DISK returns, and what this database happens to
return), the production fit's column order, and the anatomical ring order. The first two
differ only in the ``IR``/``nose_corner`` slots; the ring order differs from both in all
eight pupil positions. ``DISKModel.Keypoint`` records which body part is column *i* so that
nothing has to infer it.
"""

from __future__ import annotations

import datetime
import importlib
import inspect
from pathlib import Path

import datajoint as dj
import numpy as np

# The eye-trace resampling and the gaze geometry, shared verbatim with the production chain
# in pupil_tracking.py so the imputed tables cannot drift from their twins. That module
# declares nothing, opens no connection and needs only numpy/scipy/tqdm, so importing it
# here does not cost this module its independence from the element_* tree (below) -- which
# importing pupil_tracking.py itself very much would, along with declaring its tables.
from adamacs.helpers import eye_geometry as eg

# element_interface is imported lazily inside the two classmethods that hash, so that the
# constants and definitions in this module can be read from environments that carry
# DataJoint but not the element_* tree -- the DISK evaluation environment, for one.

schema = dj.Schema()
pupil_schema = dj.Schema()

_linking_module = None
_param_table = None
_headfixed_param_table = None

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

#: Per-frame, per-marker provenance stored in ``DLCImputation.BodyPart.sample_source``.
#: A uint8 array, one value per frame. The old boolean ``imputed_mask`` is recoverable
#: as ``sample_source == SAMPLE_SOURCE["disk"]``.
SAMPLE_SOURCE = {
    "observed": 0,      # DeepLabCut, above threshold. Untouched.
    "dlc_low_kept": 1,  # DeepLabCut's own low-confidence prediction, kept rather than
                        # discarded.
                        #
                        # WITHDRAWN (2026-09-13): an earlier version of this comment said
                        # these points beat every interpolation at every gap length, and
                        # that thresholding them out cost accuracy. Both are false. That
                        # claim came off a ruler -- the radial residual to the ellipse
                        # through the surviving markers -- which is blind to a marker
                        # sliding ALONG the pupil margin. Measured with a leave-one-out
                        # ruler that predicts a point, and that was validated by
                        # recovering three known errors to within 1%, a rejected point is
                        # 10.0 px rms at a 1-2 frame gap against linear interpolation's
                        # 3.65, and 33.2 against 14.2 at 31-46 frames. It beats
                        # interpolation at NO gap length.
                        #
                        # The state is still needed. The MEDIAN rejected point at a short
                        # gap is about 0.9 px off -- as good as DISK -- and it carries
                        # image evidence DISK does not have. The rms is 10 px because a
                        # minority are catastrophic (3.6% grossly wrong at short gaps,
                        # 36% beyond 46 frames). So keeping one is defensible per sample,
                        # gated on its likelihood and on its agreement with the other
                        # markers' geometry, and never as a blanket policy.
    "disk": 2,          # Filled by the network.
    "refused": 3,       # Classified no_reflection or tracking_lost: no pupil to rebuild.
    "unfilled": 4,      # Gap longer than the model's window. 44% of the cohort's missing
                        # frames are in gaps this long.
    "not_modelled": 5,  # The marker is not in this model's keypoint contract, so it was
                        # never offered to the network and is stored exactly as DeepLabCut
                        # produced it. Distinct from "unfilled", which means the model was
                        # asked and the gap was too long -- conflating the two would hide
                        # the difference between "we could not" and "we did not try".
                        #
                        # Only arises where the contract is a strict subset of the
                        # recording's markers, which is the head-fixed cohort: its model
                        # covers IR and the eight pupil dots. The two corner markers are
                        # tracked inconsistently there -- nose_corner above threshold in
                        # 24% of frames on average, lateral_corner in 44%, both ranging
                        # 0-98.6% by recording -- and since DISK trains only on windows
                        # where every channel is clean throughout, including them cuts the
                        # usable cohort from 46 recordings in 65 to 2. They also contribute
                        # nothing downstream: the head-fixed fit thresholds them and then
                        # computes the angle from IR and the pupil ring alone.
    "interpolated": 6,  # Filled by LINEAR INTERPOLATION, on purpose, because the gap was
                        # at or below the measured crossover for the quantity being read
                        # -- see ``ImputationParamSet.Crossover``.
                        #
                        # This is not a failure state. Below the crossover interpolation
                        # is genuinely the better estimator: at a one- to two-frame gap on
                        # the topcam it puts the body centroid 0.09 px from the truth
                        # where DISK puts it 0.32 px away, because over two frames the
                        # animal has barely moved and a straight line between two observed
                        # positions is very nearly exact.
                        #
                        # Distinct from "unfilled" (the model was asked and declined, or
                        # the gap was too long) and from "not_modelled" (never offered).
                        # Conflating it with either would hide the fact that a value IS
                        # present and IS trustworthy -- just not from the network.
                        #
                        # Only the topcam chain produces it. The two eye chains have no
                        # measured crossover, so they never interpolate deliberately.
}

#: Per-frame classification stored in ``PupilFrameQuality.frame_class``. Named for what is
#: observed, not for what might be inferred -- see the class docstring.
FRAME_CLASS = {"ok": 0, "partial": 1, "no_reflection": 2, "tracking_lost": 3}

#: The classes that must not be imputed.
REFUSE_CLASSES = (FRAME_CLASS["no_reflection"], FRAME_CLASS["tracking_lost"])


def activate(
    disk_schema_name: str,
    pupil_tracking_schema_name: str | None = None,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: str | object | None = None,
):
    """Declare this module's tables. Nothing exists until this is called.

    Parameters
    ----------
    disk_schema_name
        Schema for the DISK tables, e.g. ``roselab_disk``.
    pupil_tracking_schema_name
        Schema for :class:`PupilFrameQuality` and :class:`PupilEllipseFittingImputed`.
        Defaults to ``disk_schema_name`` with a ``pupil_tracking`` suffix derived from the
        configured prefix. Pass it explicitly in anything that matters.
    create_tables
        Pass ``False`` to bind to tables that already exist without declaring anything.
    linking_module
        Module (or its name) providing the upstream tables the definitions refer to:
        ``model``, ``pupil_tracking``, ``subject``. ``adamacs.pipeline`` is the usual
        answer.

    Notes
    -----
    Declaring tables is a schema write. Do not call this to "see if it works" -- the
    declarations can be checked without touching the server; see
    ``scripts/validate_disk_schema.py``.
    """
    global _linking_module

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(linking_module), (
        "activate() needs a linking_module supplying model / pupil_tracking / subject"
    )
    _linking_module = linking_module

    if pupil_tracking_schema_name is None:
        prefix = dj.config["custom"]["database.prefix"]
        pupil_tracking_schema_name = prefix + "pupil_tracking"

    # The parameter table this chain keys to is `#pupil_ellipse_parameter_free_moving`
    # -- SINGULAR. That is the one PupilEllipseFittingFreeMoving foreign-keys to, and
    # therefore the one every row of real gaze data in this pipeline hangs off: 396
    # production fits, 353 rotation rows, 378 reconstructed-gaze rows, all of it.
    #
    # A second table, `#pupil_ellipse_parameters_free_moving` (plural), exists because the
    # class was renamed 22 minutes after the fit table had already bound its foreign key,
    # and DataJoint makes a renamed class a new table. Only `pupil_tracking.py` defines it,
    # and until 2026-09-14 this chain keyed to it -- which meant the imputed fit masked its
    # markers by one table's thresholds while inheriting a calibration computed under the
    # other's. They agreed, so nothing broke; nothing enforced that they would.
    #
    # It is imported rather than reflected. Until 2026-09-14 `pupil_tracking.py` defined
    # no class for this table -- only for the plural one -- so this module reached it
    # through a virtual module. That class now exists, so the table comes from the
    # linking module like every other upstream table here.
    global _param_table, _headfixed_param_table
    _param_table = _linking_module.pupil_tracking.PupilEllipseParameterFreeMoving

    # The head-fixed chain keys to `#pupil_ellipse_parameter`, a different table with a
    # different shape: ONE likelihood threshold for every marker rather than separate
    # pupil/IR/corner thresholds, and no pupil_min_dots column at all. Both tables call
    # their key `parameter_id`, which is exactly why the two chains cannot share a table
    # anywhere -- a set 4 in one means nothing like a set 4 in the other.
    _headfixed_param_table = _linking_module.pupil_tracking.PupilEllipseParameter

    # The imputed rotation and gaze tables foreign-key upstream tables that live in
    # pupil_tracking but are not attributes of the linking module itself, so they are named
    # here the same way the two parameter tables are. Without them a `-> EyeCamTimeSource`
    # in a definition has nothing to resolve against.
    _objects = {**_linking_module.__dict__,
                "PupilEllipseParameterFreeMoving": _param_table,
                "PupilEllipseParameter": _headfixed_param_table,
                **{n: upstream(n) for n in _UPSTREAM_DECLARED}}
    schema.activate(
        disk_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_objects,
    )
    pupil_schema.activate(
        pupil_tracking_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects={**_objects, "DLCImputation": DLCImputation,
                     "HeadfixedImputation": HeadfixedImputation,
                     "PupilEllipseFittingImputed": PupilEllipseFittingImputed,
                     "PupilRotationOptiTrackImputed": PupilRotationOptiTrackImputed},
    )


# --------------------------------------------------------------------------------------
# Helpers used by DLCImputation.make()
# --------------------------------------------------------------------------------------

#: Interpreter that can import DISK. The two environments on this host are mutually
#: exclusive -- ``datajoint`` carries adamacs and the element_* tree but no DISK, and
#: ``disk`` carries DISK and a CUDA 12 torch but cannot import ``adamacs.pipeline``. So the
#: DISK half of make() runs as a subprocess. Override in dj.config["custom"] if the
#: environment moves.
DISK_PYTHON = "/home/backup_user/miniconda3/envs/disk/bin/python"

#: Largest movement tolerated on a sample DISK was not entitled to change. See the
#: alignment gate in DLCImputation.make() for why this is a tolerance and not zero.
ALIGNMENT_TOLERANCE_PX = 1e-6

#: Largest spread tolerated when recovering the production calibration's constant
#: centres from the (T, 2) arrays the production table stores. Not slack: the value
#: must be constant by construction, so anything above float noise means the row
#: being inherited from was computed against a different reflection array.
CALIBRATION_TOLERANCE_PX = 1e-6

#: Largest disagreement tolerated between the reused production fit and the angles
#: production already stored, measured on production's own input. A radian here is a
#: whole eye rotation, so 1e-6 rad is about 0.2 arcseconds -- far below anything the
#: estimator resolves, and far above float round-trip noise.
PRODUCTION_AGREEMENT_TOLERANCE_RAD = 1e-6

#: Where job directories are written. One per imputation, kept so a run can be inspected.
DISK_WORK_ROOT = "/mnt/data/backup_user/disk_pupil_imputation/deployed"

#: Which production parameter set each imputation set inherits its calibration from.
#:
#: Sets 5 and 6 were created as exact copies of production sets 1 and 2 -- identical
#: thresholds, differing only in ``description`` -- so that the imputed chain could live
#: under its own ids and be removed in one cascading delete. ``populate_natasha.py`` mirrors
#: the same way round when it assigns work: a recording production-fitted at set 1 is
#: imputed at set 5, one fitted at set 2 is imputed at set 6.
#:
#: The calibration is inherited rather than re-estimated (see PupilEllipseFittingImputed),
#: and it is inherited FROM THE PRODUCTION FIT AT THE MIRRORED ID. The alternative -- also
#: fitting the production estimator at sets 5 and 6 so the twin sits at the same id -- was
#: rejected deliberately: it would add rows to `PupilEllipseFittingFreeMoving`, and through
#: it to `PupilRotationOptiTrack` and `GazeReconstruction3D`, which several analysis
#: callers query without pinning `parameter_id`. `multi_trial_lib.get_scan_pose_data()`
#: takes `fetch("KEY")[0]`, so a second row there is not an error anyone would see; it is a
#: different recording's worth of gaze, silently substituted. Reading across the mirror
#: instead leaves every production table untouched and every existing query returning
#: exactly what it returns today.
#:
#: Because the two sets carry identical thresholds, the inherited calibration is the same
#: number either way -- the mirror costs nothing numerically. :func:`assert_mirror_is_a_copy`
#: proves that rather than assuming it, and `_run_imputed_fit`'s reproduction gate would
#: fail loudly if it ever stopped being true.
PRODUCTION_MIRROR = {5: 1, 6: 2}

#: Parameter-table columns that define the estimator. `description` is prose and
#: `parameter_id` is the identity being mapped, so neither takes part in the comparison.
_MIRROR_COMPARED_FIELDS = ("llh_thres_pupil", "llh_thres_ir", "llh_thres_corner",
                           "pupil_min_dots", "exclude_ir_std", "ellipticity_thres")


#: Which module of the linking package each bare upstream name can be found on, in order of
#: preference. The imputed rotation and gaze tables reference four tables this module does
#: not define and the linking module does not expose as attributes of itself.
#:
#: ``RigidMouseTracking`` is the awkward one and is worth spelling out, because getting it
#: wrong fails only for some callers. ``pupil_tracking.py`` *imports* it from
#: ``virtual_markers_optitrack``, so it is an attribute of the real module -- but
#: ``dj.create_virtual_module("pupil_tracking", ...)`` reflects one schema and exposes only
#: the tables declared *in* it, so the same lookup returns nothing there. Since
#: ``populate_natasha.py``, ``run_chain.sh`` and ``validate_disk_schema.py`` all build their
#: linking module out of virtual modules, resolving it through ``pupil_tracking`` alone
#: would work under ``adamacs.pipeline`` and fail under every script.
_UPSTREAM_SOURCES = {
    "EyeCamTimeSource": ("pupil_tracking",),
    "EyeModel": ("pupil_tracking",),
    "TorsionCalibManual": ("pupil_tracking",),
    "RigidMouseTracking": ("virtual_markers_optitrack", "pupil_tracking"),
    "EyeSideNote": ("pupil_tracking",),
    "Event": ("event",),
}

#: The subset that must resolve at DECLARATION time, i.e. the names appearing in a
#: `-> Parent` line. The rest are only needed when make() runs, so a linking module missing
#: them can still declare.
_UPSTREAM_DECLARED = ("EyeCamTimeSource", "EyeModel", "TorsionCalibManual",
                      "RigidMouseTracking")


def upstream(name: str):
    """An upstream table by bare name, wherever this linking module happens to keep it."""
    if _linking_module is None:
        raise RuntimeError("activate() has not run; upstream tables are not bound")
    if name not in _UPSTREAM_SOURCES:
        raise KeyError(f"{name} is not one of the upstream tables this module resolves: "
                       f"{sorted(_UPSTREAM_SOURCES)}")
    for mod_name in _UPSTREAM_SOURCES[name]:
        mod = getattr(_linking_module, mod_name, None)
        if mod is not None and hasattr(mod, name):
            return getattr(mod, name)
    if hasattr(_linking_module, name):
        return getattr(_linking_module, name)
    raise AttributeError(
        f"cannot find {name} on the linking module; looked at "
        f"{list(_UPSTREAM_SOURCES[name])} and at the module itself. A linking module built "
        f"from dj.create_virtual_module must include every schema these tables are DECLARED "
        f"in -- notably virtual_markers_optitrack, which is where RigidMouseTracking lives; "
        f"pupil_tracking.py only imports it.")


def production_parameter_id(parameter_id: int) -> int:
    """The production set whose calibration an imputation set inherits."""
    try:
        return PRODUCTION_MIRROR[int(parameter_id)]
    except KeyError:
        raise KeyError(
            f"parameter_id {parameter_id} is not an imputation set; PRODUCTION_MIRROR "
            f"maps {sorted(PRODUCTION_MIRROR)}. An imputation at a set with no production "
            f"mirror has no calibration to inherit and must not be fitted.") from None


def production_key_for(key: dict) -> dict:
    """``key`` with ``parameter_id`` moved to its mirrored production set.

    Only the attributes a production fit is keyed on are kept, so the result can restrict
    `PupilEllipseFittingFreeMoving` directly.
    """
    out = {k: v for k, v in key.items()
           if k in ("session_id", "scan_id", "recording_id", "model_name")}
    out["parameter_id"] = production_parameter_id(key["parameter_id"])
    return out


def assert_mirror_is_a_copy() -> dict:
    """Fail if an imputation set and the production set it mirrors ever diverge.

    The mirror is only legitimate while the two sets describe the same estimator. If
    somebody edits set 5's thresholds, its imputations would be masked one way and
    calibrated from a fit made another way -- the exact confusion that the singular/plural
    parameter-table mix-up produced before, which "agreed, so nothing broke; nothing
    enforced that they would".
    """
    if _param_table is None:
        raise RuntimeError("activate() has not run; the parameter table is not bound")
    rows = {r["parameter_id"]: r for r in _param_table.fetch(as_dict=True)}
    checked = {}
    for imp_id, prod_id in PRODUCTION_MIRROR.items():
        if imp_id not in rows or prod_id not in rows:
            raise RuntimeError(
                f"PRODUCTION_MIRROR maps {imp_id} -> {prod_id} but "
                f"{'both are' if imp_id not in rows and prod_id not in rows else 'one is'} "
                f"missing from {_param_table.table_name}")
        differ = {f: (rows[imp_id][f], rows[prod_id][f]) for f in _MIRROR_COMPARED_FIELDS
                  if rows[imp_id][f] != rows[prod_id][f]}
        if differ:
            raise RuntimeError(
                f"imputation set {imp_id} is no longer a copy of production set {prod_id}: "
                f"{differ}. The inherited calibration would come from a different estimator "
                f"than the one that masked the markers.")
        checked[imp_id] = prod_id
    return checked


def production_fit_hash() -> str:
    """SHA-256 of the production ellipse fit, read from the adamacs source.

    Stamped into every :class:`DLCImputation` row so that a change to the production
    estimator makes these rows visibly stale rather than quietly wrong. The source text is
    hashed; nothing is executed.
    """
    import ast
    import hashlib

    src_path = Path(__file__).with_name("pupil_tracking.py")
    text = src_path.read_text()
    tree = ast.parse(text)
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and cls.name == "PupilEllipseFittingFreeMoving":
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "fit_ellipse":
                    seg = ast.get_source_segment(text, fn)
                    return hashlib.sha256(seg.encode()).hexdigest()
    raise RuntimeError(
        "PupilEllipseFittingFreeMoving.fit_ellipse not found in pupil_tracking.py; the "
        "production code has been restructured and this stamp needs updating")


def _runs(mask: np.ndarray):
    """Yield ``(start, length)`` for each contiguous True run in a 1-D boolean array."""
    if not mask.any():
        return
    padded = np.r_[False, mask, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for s, e in zip(edges[::2], edges[1::2]):
        yield int(s), int(e - s)


def _yields_angle(likelihood, markers, params) -> np.ndarray:
    """Frames the production fit would turn into an eye angle.

    Exact rather than approximate, and it needs no conic fit: the production estimator
    expresses pupil coordinates relative to the corneal reflection, so a frame yields an
    angle exactly when the IR marker survives its threshold and at least
    ``pupil_min_dots`` pupil markers survive theirs.
    """
    idx = {m: i for i, m in enumerate(markers)}
    pupil = [i for m, i in idx.items() if m.startswith("pupil_")]
    n_pupil = (likelihood[:, pupil] >= params["llh_thres_pupil"]).sum(axis=1)
    ir_ok = likelihood[:, idx["IR"]] >= params["llh_thres_ir"]
    return (n_pupil >= params["pupil_min_dots"]) & ir_ok


def _write_dlc_csv(path: Path, markers, xy, likelihood) -> None:
    """Write the DeepLabCut CSV layout DISK's reader expects.

    Three details are load-bearing and easy to get wrong (``DISK/create_dataset.py``,
    ``dlc_csv`` branch): the header is parsed with ``header=[1, 2]`` so row 0 is scorer,
    row 1 body parts and row 2 coordinates; the first column must be labelled
    ``bodyparts``/``coords`` exactly as DeepLabCut writes it, because the reader filters
    that name out to find the keypoints; and DISK sorts keypoint names alphabetically
    internally, so the columns are written in the caller's contract order and mapped back
    by name on the way out, never by position.
    """
    import csv

    n, k = xy.shape[0], len(markers)
    body = np.empty((n, 3 * k), dtype=np.float64)
    body[:, 0::3] = xy[:, :, 0]
    body[:, 1::3] = xy[:, :, 1]
    body[:, 2::3] = likelihood
    table = np.column_stack([np.arange(n, dtype=np.float64), body])

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scorer"] + ["adamacs"] * (3 * k))
        w.writerow(["bodyparts"] + [m for m in markers for _ in range(3)])
        w.writerow(["coords"] + ["x", "y", "likelihood"] * k)
        for row in table:
            # %.17g, not %g: %g keeps only 6 significant figures, which alters
            # every float64 on the round trip and makes an exact alignment gate
            # impossible -- it reported the whole recording as shifted.
            w.writerow(["%.17g" % v if np.isfinite(v) else "" for v in row])


def _read_dlc_csv(path: Path, markers):
    """Read a DeepLabCut CSV back into ``(T, K, 2)`` in ``markers`` order, by NAME."""
    import pandas as pd

    df = pd.read_csv(path, header=[1, 2])
    cols = {}
    for c in df.columns:
        bp, coord = c[0], c[1]
        if bp in ("bodyparts", "coords") or not isinstance(coord, str):
            continue
        cols.setdefault(bp, {})[coord] = c
    missing = [m for m in markers if m not in cols]
    if missing:
        raise ValueError(f"{path} is missing markers {missing}")
    n = len(df)
    xy = np.full((n, len(markers), 2), np.nan)
    for i, m in enumerate(markers):
        xy[:, i, 0] = pd.to_numeric(df[cols[m]["x"]], errors="coerce").to_numpy()
        xy[:, i, 1] = pd.to_numeric(df[cols[m]["y"]], errors="coerce").to_numpy()
    return xy


# ======================================================================================
# The trained model
# ======================================================================================


@schema
class DISKModel(dj.Lookup):
    """Registry of trained DISK checkpoints.

    This table identifies a model rather than describing one. Every field below changes
    what the network does or what its numbers mean, and each was previously carried -- if
    at all -- in a free-text description.

    ``checkpoint_epoch`` earns its place: an earlier version of the evaluation quoted an
    epoch-15 checkpoint and understated the result roughly threefold, and one qualitative
    conclusion changed when the converged model replaced it. "Which checkpoint" is not a
    detail.

    ``training_fps`` earns its place too. DISK takes a single ``original_freq`` per
    dataset, and this cohort holds 472 recordings at 50 Hz, 360 at 60 Hz and 30 at 25 Hz.
    The 60 Hz group is a transfer test that has never been scored. ``make()`` compares this
    against the recording's own rate rather than letting a 50 Hz model be applied silently.

    ``model_status`` exists so ordinary analysis has one obvious variant to pin. With three
    optional key dimensions downstream, a query that forgets ``disk_model_name`` silently
    pools two networks -- the same failure the pipeline guidance warns about for
    ``paramset_idx``. Keep at most one row at ``'current'`` per ``data_type``.
    """

    definition = """
    disk_model_name          : varchar(64)    # unique model identifier
    ---
    checkpoint_path          : varchar(512)   # directory holding model_epoch*; .hydra config must be present
    checkpoint_epoch         : smallint unsigned  # WHICH checkpoint -- not a detail, see docstring
    val_rmse                 : float          # best validation RMSE at that epoch
    network_type             : enum('transformer','GRU','BiGRU','ST_GCN','TCN')
    seq_length               : smallint unsigned  # frames; hard ceiling on fillable gap length
    training_fps             : decimal(7,3)   # the dataset's original_freq
    fill_gap                 : smallint unsigned  # pre-dataset interpolation; 0 means none
    mu_sigma                 : tinyint unsigned   # 1 if trained to emit per-sample sigma
    data_type                : enum('dlc_2d','dlc_3d','mocap_3d')
    likelihood_threshold     : float          # dlc_likelihood_threshold used to build the dataset
    disk_version             : varchar(32)    # DISK-impute package version
    trained_on               : datetime
    model_status='candidate' : enum('candidate','current','superseded')
    keypoint_hash            : uuid           # dict_to_uuid of the ordered Keypoint list
    model_description=''     : varchar(512)
    """

    class Keypoint(dj.Part):
        """The marker contract: which body part is column *i* of the model's input.

        A count is not an identity. Storing only ``n_keypoints`` lets any 11-keypoint model
        be applied to any 11-keypoint recording with every marker silently permuted, and
        nothing about the result would look wrong -- the arrays have the right shape, the
        ellipse fit succeeds, the eye angles are smooth.

        DISK sorts keypoint names alphabetically internally, so for a model trained through
        ``dpi.export`` these rows are the alphabetical order. Recording it makes that a
        fact rather than an assumption.
        """

        definition = """
        -> master
        -> model.BodyPart
        ---
        keypoint_index : tinyint unsigned   # 0-based column index in the model's input
        unique index (disk_model_name, keypoint_index)
        """

    class TrainingAnimal(dj.Part):
        """Animals the model was fitted or validated on.

        The split is by animal, never by sequence: consecutive frames are nearly the same
        picture, so a sequence-level split reports a number that has nothing to do with
        performance on a new animal.

        Recording it makes "was this animal held out?" a restriction rather than a memory.
        Once imputed data is in the database, nothing else stops someone reporting an
        imputation-improved result on a training animal as though it were held out.
        Animals absent from this part table were not seen by the model.
        """

        definition = """
        -> master
        -> subject.Subject
        ---
        split : enum('train','val')   # test animals are ABSENT, not listed
        """

    @classmethod
    def keypoint_order(cls, disk_model_name: str) -> list[str]:
        """Ordered body-part names for a model, as its input columns."""
        rows = (cls.Keypoint & {"disk_model_name": disk_model_name}).fetch(
            "keypoint_index", "body_part", order_by="keypoint_index"
        )
        idx, names = rows
        if list(idx) != list(range(len(idx))):
            raise ValueError(
                f"{disk_model_name}: keypoint_index is not a contiguous 0..n-1 range "
                f"({list(idx)}). The marker contract is broken; refusing to guess."
            )
        return list(names)

    @classmethod
    def hash_keypoints(cls, ordered_body_parts: list[str]) -> str:
        """Hash an ordered keypoint list, for ``keypoint_hash``."""
        from element_interface.utils import dict_to_uuid

        return dict_to_uuid({str(i): bp for i, bp in enumerate(ordered_body_parts)})


@schema
class ImputationParamSet(dj.Lookup):
    """Policy applied around the model: what to refuse, and what not to attempt.

    Separate from :class:`DISKModel` on purpose. Policy is not the model, and sweeping it
    without retraining is a real use case -- the evaluation's own robustness check varies
    the refusal thresholds and shows the decision moves only between 3.73% and 4.89% of
    frames across the whole plausible range.

    ``max_gap_frames`` should normally equal the model's ``seq_length``: DISK cannot fill a
    gap longer than its window, and 44% of the cohort's missing frames sit in gaps longer
    than 60 frames. Those become ``sample_source == 'unfilled'`` and are counted, not
    quietly dropped.

    ``split_gaps_at_refused`` changes the GRANULARITY of refusal, not its threshold. At 0,
    the original behaviour, one refused frame anywhere in a run of missing samples withholds
    the entire run; at 1 the refused frames stay refused and the stretches between them are
    judged on their own length. The distinction matters because refusal is decided per
    frame and was applied per gap: measured over 40 recordings, 76.3% of all withheld
    frames were withheld by contamination rather than by length, in runs of median length 8.

    **Leave ``keep_dlc_low_conf`` at 0 unless you have a per-sample rule.** It only takes
    effect where DISK declines -- that is, in gaps longer than the window -- and that is
    precisely the regime where a rejected DeepLabCut point is worst. Measured by likelihood
    band in gaps beyond 60 frames, the *best* band (0.3-0.5) is still 11.2 px rms, roughly a
    fifth of a pupil diameter, and the worst (below 0.01) is 56.0 px with 62% of samples
    grossly wrong. No band stays usable, so a blanket 1 fills the hardest frames in the
    cohort with the least reliable data available. A defensible version of this flag is not
    a boolean but a likelihood floor combined with an agreement test against the other
    markers' geometry; that rule has been designed but not yet validated, so the boolean
    stays and its default stays 0.
    """

    definition = """
    imputation_paramset_id      : smallint unsigned
    ---
    refuse_no_reflection=1      : tinyint unsigned  # refuse frames with no corneal reflection
    refuse_tracking_lost=1      : tinyint unsigned  # refuse frames with nothing tracked
    keep_dlc_low_conf=0         : tinyint unsigned  # keep DLC's sub-threshold guess where DISK declines; see docstring before setting 1
    split_gaps_at_refused=0     : tinyint unsigned  # 0: one refused frame withholds the whole gap. 1: split the gap around it and judge each stretch on its own length
    max_gap_frames              : smallint unsigned # longer gaps are left unfilled, not filled badly
    paramset_hash               : uuid
    paramset_description=''     : varchar(255)
    unique index (paramset_hash)
    """

    @classmethod
    def hash_params(cls, params: dict) -> str:
        from element_interface.utils import dict_to_uuid

        return dict_to_uuid(
            {k: params[k] for k in sorted(params) if k not in ("imputation_paramset_id",
                                                               "paramset_hash",
                                                               "paramset_description")}
        )

    class Crossover(dj.Part):
        """Per-quantity gap length below which the model is NOT called.

        THE KEY CARRIES THE MODEL AS WELL AS THE QUANTITY
        -------------------------------------------------
        A crossover is a property of the CHECKPOINT paired with the quantity, not of the
        parameter set alone. Two checkpoints can legitimately share one parameter set --
        the held-out-animal topcam model shares set 4 with the within-animal one by
        design -- and there is no reason their crossovers should agree: they were fitted on
        different data and one of them has never seen the animal being imputed.

        Keyed on the parameter set alone, a lookup for the second checkpoint would silently
        return the first checkpoint's threshold, or its ``use_model = 0``. Both are wrong
        answers that look like right ones, and the failure is invisible because the row
        that comes back is well-formed. So ``disk_model_name`` is in the key and in every
        lookup.

        WHY THIS IS A PART TABLE AND NOT A COLUMN
        -----------------------------------------
        Two reasons, and the first is decisive.

        **One column cannot hold the finding.** The crossover is not a property of the
        model; it is a property of the model *paired with the quantity being consumed*, and
        the measured values differ. On the topcam benchmark the optimum is 20 frames for the
        body centroid and for ear heading, 45 for IR heading, and for centroid speed there is
        no useful crossover at all. A scalar attribute would have to pick one of those and be
        wrong for the other three.

        **A new master column would invalidate every stored hash.**
        :meth:`ImputationParamSet.hash_params` digests every attribute except the id, the
        hash and the description, so adding one would change the hash of parameter sets that
        already exist; a later insert of "the same" policy would then compute a different
        uuid and create a duplicate instead of matching. A part table leaves the master row,
        and therefore its hash, exactly as it was.

        WHY THIS EXISTS AT ALL
        ----------------------
        Because a deployment that ignores it does measurable harm. Measured on recordings the
        topcam checkpoint never trained on (``bench_topcam.py``, PR #106), calling DISK on
        every gap inside its window makes centroid speed **43% worse** than plain linear
        interpolation, while calling it only above the crossover cuts ear heading by 54%, the
        body centroid by 48% and IR heading by 40%. The gain and the harm come from the same
        model and are separated only by this number.

        The mechanism is not mysterious. Below the crossover the animal has barely moved, so
        a straight line between the two observed positions is very nearly exact and the model
        is reconstructing from dynamics it does not need. Above it there is no straight line
        to draw. Speed is the exception in kind rather than in degree: it is a central
        difference, so it is sensitive to how *smooth* a fill is rather than to where it
        sits, and DISK's fills are accurate but rougher than a straight line.

        HOW TO READ IT
        --------------
        ``use_model = 0`` means never call the model for this quantity, whatever the gap
        length, and ``crossover_frames`` is then advisory only. Otherwise: fill gaps of
        ``crossover_frames`` or shorter by interpolation and call the model above that, up to
        the master's ``max_gap_frames``.

        Absence of a row is NOT "no restriction". A consumer that finds no row for its
        quantity has no measured crossover for it, and the honest default is then to refuse
        the model rather than to assume it is safe -- assuming it is safe is precisely the
        configuration this table exists to prevent.
        """

        definition = """
        -> master
        -> DISKModel                            # the checkpoint the crossover was measured on
        consumer_quantity       : varchar(40)   # the quantity this rule governs
        ---
        crossover_frames        : smallint unsigned # interpolate gaps <= this; model above
        use_model=1             : tinyint unsigned  # 0 -> never call the model for this quantity
        evidence=''             : varchar(255)      # what measured it
        """



# ======================================================================================
# The production ellipse fit, reused rather than reimplemented
# ======================================================================================

#: Column order the production ``make()`` builds its arrays in: the two eye corners, the
#: corneal reflection, then the eight pupil-margin markers.
#:
#: This is a third ordering, and no two of the three are interchangeable. It is not the
#: model's keypoint contract (alphabetical, so ``IR`` comes first) and it is not the
#: anatomical ring order. Everything below reindexes by NAME onto this order, never by
#: position -- pairing one ordering's coordinates with another's mask produces plausible
#: numbers that are wrong, and that has already happened once on this project.
PRODUCTION_ORDER = [
    "nose_corner", "lateral_corner", "IR",
    "pupil_left", "pupil_left_lower", "pupil_left_up", "pupil_lower",
    "pupil_right", "pupil_right_lower", "pupil_right_up", "pupil_upper",
]


def _quiet_print(*args, **kwargs):
    """The production fit prints a line per malformed frame; silence it in bulk runs."""


def _production_fit_ellipse():
    """Extract, compile and return the production ``fit_ellipse``.

    The numerically delicate part is **not copied**. It is lifted out of
    ``pupil_tracking.py`` with :mod:`ast` at run time and compiled, exactly as
    ``dpi/ellipse.py`` does in the evaluation repository, so that this table measures the
    same estimator production measures. Writing a second conic fit would mean any
    apparent improvement could be an artefact of the rewrite rather than of imputation.

    The extracted function is a plain numerical routine -- it touches no DataJoint object
    -- so it runs unchanged with ``self`` unused. Its SHA-256 is what
    :func:`production_fit_hash` stamps into every :class:`DLCImputation` row.
    """
    import ast

    if getattr(_production_fit_ellipse, "_cached", None) is not None:
        return _production_fit_ellipse._cached

    src_path = Path(__file__).with_name("pupil_tracking.py")
    text = src_path.read_text()
    tree = ast.parse(text)

    node = None
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and cls.name == "PupilEllipseFittingFreeMoving":
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "fit_ellipse":
                    node = fn
    if node is None:
        raise RuntimeError(
            "PupilEllipseFittingFreeMoving.fit_ellipse not found in pupil_tracking.py; "
            "the production code has been restructured and this wrapper needs updating")

    namespace: dict = {"np": np, "print": _quiet_print}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(src_path), "exec"), namespace)
    _production_fit_ellipse._cached = namespace["fit_ellipse"]
    return _production_fit_ellipse._cached


def _mask_and_center(xy, likelihood, markers, params):
    """Apply the production likelihood thresholds and centre the pupil on the reflection.

    Mirrors the production ``make()`` step for step, including the IR outlier rejection:
    the corneal reflection is near-stationary in camera coordinates, so an "IR" far from
    the recording's own IR cloud is a misdetection rather than a moved reflection.

    Returns ``(pc_x, pc_y, ir_xg, ir_yg)`` -- IR-centred pupil coordinates ``(T, 8)`` and
    the cleaned reflection position ``(T,)``. Pupil coordinates are expressed relative to
    the reflection, so a frame whose IR is missing or rejected yields no eye angle however
    good its pupil markers are.
    """
    idx = {m: i for i, m in enumerate(markers)}
    missing = [m for m in PRODUCTION_ORDER if m not in idx]
    if missing:
        raise ValueError(f"markers {missing} absent; this is not the production eye model")
    order = [idx[m] for m in PRODUCTION_ORDER]

    x_all = xy[:, order, 0]
    y_all = xy[:, order, 1]
    llh_all = likelihood[:, order]

    mask_ir = llh_all[:, 2] < params["llh_thres_ir"]
    mask_pupil = llh_all[:, 3:] < params["llh_thres_pupil"]

    ir_x, ir_y = np.copy(x_all[:, 2]), np.copy(y_all[:, 2])
    ir_x[mask_ir] = np.nan
    ir_y[mask_ir] = np.nan

    p_x, p_y = np.copy(x_all[:, 3:]), np.copy(y_all[:, 3:])
    p_x[mask_pupil] = np.nan
    p_y[mask_pupil] = np.nan

    ir_std = np.linalg.norm([np.nanstd(ir_x), np.nanstd(ir_y)])
    ir_bad = ((ir_x - np.nanmean(ir_x)) ** 2 + (ir_y - np.nanmean(ir_y)) ** 2
              > (params["exclude_ir_std"] * ir_std) ** 2)
    ir_xg, ir_yg = np.copy(ir_x), np.copy(ir_y)
    ir_xg[ir_bad] = np.nan
    ir_yg[ir_bad] = np.nan

    return p_x - ir_xg[:, None], p_y - ir_yg[:, None], ir_xg, ir_yg


def _fit_frames(pc_x, pc_y, params):
    """Run the production conic fit on every frame clearing the dot minimum.

    Returns five ``(T,)`` arrays -- ``X0_in``, ``Y0_in``, ``long_axis``, ``short_axis``,
    ``angle_from_x`` -- NaN where no fit was produced. ``long_axis`` is the *semi*-major
    axis despite the name: the production fit stores ``long_axis/2`` under that key, and
    the production table's ``diameter`` column is that same half-value. Preserved rather
    than corrected, because the point is fidelity to production.
    """
    fit_ellipse = _production_fit_ellipse()
    n = pc_x.shape[0]
    keys = ("X0_in", "Y0_in", "long_axis", "short_axis", "angle_from_x")
    out = {k: np.full(n, np.nan) for k in keys}

    enough = (~np.isnan(pc_x)).sum(axis=1) >= params["pupil_min_dots"]
    for i in np.flatnonzero(enough):
        d = fit_ellipse(None, i, pc_x[i], pc_y[i])
        if not d:          # None on LinAlgError, or an all-NaN dict on a non-ellipse
            continue
        for k in keys:
            out[k][i] = d[k]
    return tuple(out[k] for k in keys)


def _nice_mask(pc_x, long_axis, short_axis, params):
    """Frames the production calibration is estimated from.

    At least seven pupil dots and a sufficiently elliptical fit. Both conditions matter:
    the camera centre is recovered from where the ellipses' major axes intersect, and a
    near-circular pupil carries no directional information to contribute.
    """
    with np.errstate(invalid="ignore"):
        return np.flatnonzero(
            ((~np.isnan(pc_x)).sum(axis=1) >= 7)
            & (short_axis / long_axis < params["ellipticity_thres"]))


def fetch_params(key) -> dict:
    """The ellipse parameters for ``key``, spelled as the table spells them.

    Reads `#pupil_ellipse_parameter_free_moving`, the table the production fit keys to.
    This used to rename the columns onto a second spelling, because the only class the
    source defined described the other, orphaned table. `pupil_tracking.py` defines this
    one now, so there is a single spelling and nothing left to translate.
    """
    if _param_table is None:
        raise RuntimeError("activate() has not run; the parameter table is not bound")
    return (_param_table & {"parameter_id": key["parameter_id"]}).fetch1()


#: The head-fixed production fit hard-codes its dot minimum. `PupilEllipseFitting.make()`
#: reads three values from its parameter table and then writes ``nd = 6`` inline, with the
#: comment "can be put in parameter table!". Six is therefore part of that estimator, not a
#: choice this module gets to make, and it is spelled out here rather than buried.
HEADFIXED_PUPIL_MIN_DOTS = 6


def fetch_headfixed_params(key) -> dict:
    """Head-fixed ellipse parameters, expressed in the shape the shared helpers expect.

    `#pupil_ellipse_parameter` carries a single ``likelihood_thres`` applied to every
    marker; the free-moving table splits it into ``llh_thres_pupil``, ``llh_thres_ir`` and
    ``llh_thres_corner``. The numerical helpers (:func:`_mask_and_center`,
    :func:`_fit_frames`, :func:`_nice_mask`, :func:`_yields_angle`) all read the
    free-moving spelling, so this translates rather than duplicating them.

    The translation is exact, not approximate: `PupilEllipseFitting.make()` applies its one
    threshold to the whole likelihood array in a single masked assignment, which is the
    same thing as applying it separately to pupil, IR and corner markers.
    """
    if _headfixed_param_table is None:
        raise RuntimeError("activate() has not run; the head-fixed parameter table is not bound")
    p = (_headfixed_param_table & {"parameter_id": key["parameter_id"]}).fetch1()
    thres = float(p["likelihood_thres"])
    return {
        "parameter_id": p["parameter_id"],
        "llh_thres_pupil": thres,
        "llh_thres_ir": thres,
        "llh_thres_corner": thres,
        "pupil_min_dots": HEADFIXED_PUPIL_MIN_DOTS,
        "exclude_ir_std": float(p["exclude_ir_std"]),
        "ellipticity_thres": float(p["ellipticity_thres"]),
        "description": p.get("description", ""),
    }


def assert_fit_ellipse_shared() -> str:
    """Fail loudly if the two production ``fit_ellipse`` implementations have diverged.

    ``PupilEllipseFitting.fit_ellipse`` and ``PupilEllipseFittingFreeMoving.fit_ellipse``
    are, as of 2026-09-15, byte-identical -- the same 109 lines with the same SHA-256. That
    is why :func:`_production_fit_ellipse` can serve both chains, and why
    :func:`production_fit_hash` means the same thing for either.

    It is also a fact about the current source, not a guarantee. If someone edits one and
    not the other, the head-fixed chain would silently keep running the free-moving
    estimator and stamping the free-moving hash, and every head-fixed imputed angle would
    be measuring an estimator production no longer uses. So it is asserted, once, at the
    top of each head-fixed make(). Returns the shared hash.
    """
    import ast
    import hashlib

    src_path = Path(__file__).with_name("pupil_tracking.py")
    text = src_path.read_text()
    tree = ast.parse(text)
    got = {}
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and cls.name in (
                "PupilEllipseFitting", "PupilEllipseFittingFreeMoving"):
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "fit_ellipse":
                    got[cls.name] = ast.get_source_segment(text, fn)
    missing = {"PupilEllipseFitting", "PupilEllipseFittingFreeMoving"} - set(got)
    if missing:
        raise RuntimeError(
            f"could not find fit_ellipse on {sorted(missing)} in pupil_tracking.py; the "
            "production code has been restructured and the head-fixed chain needs review")
    h = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in got.items()}
    if h["PupilEllipseFitting"] != h["PupilEllipseFittingFreeMoving"]:
        raise RuntimeError(
            "PupilEllipseFitting.fit_ellipse and PupilEllipseFittingFreeMoving.fit_ellipse "
            "have DIVERGED (%s vs %s). The head-fixed chain reuses the free-moving "
            "extraction and would now be running the wrong estimator. Give the head-fixed "
            "chain its own extraction before populating anything further."
            % (h["PupilEllipseFitting"][:12], h["PupilEllipseFittingFreeMoving"][:12]))
    return h["PupilEllipseFitting"]


def _recover_calibration(cam_center, eye_center, ir_xg, ir_yg):
    """Recover the production calibration's two scalar centres from what it stored.

    The production table stores ``cam_center`` and ``eye_center`` as ``(T, 2)`` arrays
    already un-centred back into original pixel coordinates: each row is the constant plus
    that frame's reflection position. The constants themselves are never stored, so they
    are recovered by subtracting the same reflection array production subtracted. That is
    why this must be handed the **original** markers' ``ir_xg``/``ir_yg`` and never the
    imputed ones -- an imputed reflection is present on frames where production's was NaN,
    and the difference would silently become part of the "constant".

    The subtraction must return a constant, and that is asserted rather than assumed. If
    it does not, the row being inherited from was computed against a different reflection
    array, and its camera centre is not the one that produced its angles.
    """
    out = []
    for arr, what in ((cam_center, "cam_center"), (eye_center, "eye_center")):
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"production {what} has shape {arr.shape}, expected (T, 2)")
        comps = (arr[:, 0] - ir_xg, arr[:, 1] - ir_yg)
        const = []
        for comp, axis in zip(comps, "xy"):
            finite = comp[np.isfinite(comp)]
            if finite.size == 0:
                raise ValueError(
                    f"production {what} minus the original reflection is empty on every "
                    f"frame; the inherited calibration cannot be recovered")
            spread = float(finite.max() - finite.min())
            if spread > CALIBRATION_TOLERANCE_PX:
                raise ValueError(
                    f"production {what}.{axis} minus the original reflection varies by "
                    f"{spread:g} px across frames, but it must be constant. The stored "
                    f"row was computed against a different reflection array, so its "
                    f"calibration is not the one that produced its angles.")
            const.append(float(finite.mean()))
        out.append(np.array(const))
    return out[0], out[1]


def _to_angles(x0_in, y0_in, cam_cent, scale, horizontal_offset, vertical_offset):
    """Convert IR-centred pupil centres to eye rotation angles, in radians.

    The production conversion, unchanged: azimuth from the horizontal displacement of the
    pupil centre from the camera centre, elevation from the vertical displacement
    corrected for azimuth, both then referred to the eye centre.
    """
    with np.errstate(invalid="ignore"):
        theta_cam = np.arcsin((x0_in - cam_cent[0]) / scale)
        phi_cam = np.arcsin((y0_in - cam_cent[1]) / np.cos(theta_cam) / scale)
    return theta_cam - horizontal_offset, phi_cam - vertical_offset


def _check_reproduces_production(theta_ours, theta_prod, key) -> None:
    """Assert that the reused fit returns what production stored, on production's input.

    This is the gate that makes the sibling table's comparison meaningful. Run over the
    *original* markers under the *inherited* calibration, the extracted fit is doing
    exactly what production did, so it must return exactly what production stored. Any
    disagreement means the masking, the marker ordering or the conic fit has drifted from
    production, and the difference between the two tables would then measure that drift
    rather than imputation.

    Both the values and the *set of frames that produced a value* are checked: a fit that
    agreed everywhere it produced a number, but produced numbers on a different set of
    frames, would still be a different estimator.
    """
    if theta_ours.shape != theta_prod.shape:
        raise RuntimeError(
            f"reused fit returned {theta_ours.shape} angles but production stored "
            f"{theta_prod.shape} for {key}")

    ours_ok, prod_ok = np.isfinite(theta_ours), np.isfinite(theta_prod)
    if int((ours_ok != prod_ok).sum()):
        n = int((ours_ok != prod_ok).sum())
        raise RuntimeError(
            f"reused fit yielded an angle on {n} frames where production did not, or vice "
            f"versa ({int(ours_ok.sum())} against {int(prod_ok.sum())} frames) for {key}. "
            f"The estimator has drifted from production.")

    both = ours_ok & prod_ok
    if not both.any():
        raise RuntimeError(f"production stored no finite angle at all for {key}")
    worst = float(np.max(np.abs(theta_ours[both] - theta_prod[both])))
    if worst > PRODUCTION_AGREEMENT_TOLERANCE_RAD:
        raise RuntimeError(
            f"reused fit disagrees with production by up to {worst:g} rad "
            f"({np.degrees(worst):g} deg) on the original markers for {key}, tolerance "
            f"{PRODUCTION_AGREEMENT_TOLERANCE_RAD:g}. This table would be measuring that "
            f"disagreement rather than imputation.")


# ======================================================================================
# Frame quality -- classify before imputing
# ======================================================================================


@pupil_schema
class PupilFrameQuality(dj.Computed):
    """Per-frame classification of what the eye camera actually shows.

    **This runs before imputation, and it is the safety property of the whole pipeline.**
    Filling a gap where there is no pupil returns a smooth, confident, entirely invented
    gaze direction, and nothing downstream can tell that sample from a real one.

    Classes are named for what is *observed*, not for what might be inferred:

    ``ok``
        Six or more pupil markers survive. The production fit works.
    ``partial``
        Pupil markers short, but the corneal reflection is present. The cornea is exposed,
        so there is a pupil to reconstruct. **Impute.**
    ``no_reflection``
        Pupil markers short and the reflection gone, with an eye corner still tracked.
        Lid, extreme gaze angle, or motion -- the cause is not identified. **Refuse.**
    ``tracking_lost``
        Pupil, reflection and corners all gone. **Refuse.**

    Three rules carried from the evaluation:

    1. **Report the refusal fraction as data, not as a silent gate.** It ranges 12-48%
       across animals and rises steeply with gaze eccentricity -- the corneal reflection is
       specular and vanishes when the eye rotates off-axis while wide open (0.08% of frames
       at centre gaze against 6.6% in the top eccentricity decile). A fixed gate would bias
       the dataset toward centred gaze without saying so.
    2. **Classify each eye independently.** Mouse eye closures are frequently unilateral or
       asynchronous. Requiring the fellow camera to corroborate would discard real events,
       and the two cameras have separate clocks in any case.
    3. **Do not export a blink rate.** The blink interpretation of ``no_reflection`` was
       tested and withdrawn: median episode is 2 frames against a rodent blink of 100-200 ms,
       and about a quarter of the class has the specular cause above. The refusal
       *decision* is sound; the behavioural label was not.

    Keyed on the parameter set because the thresholds *are* the parameter set -- classify at
    0.9 rather than 0.5 and the answer differs.
    """

    definition = """
    -> PupilEllipseParameterFreeMoving
    -> model.PoseEstimationNew
    ---
    frame_class          : blob@external-raw   # uint8 per frame, see FRAME_CLASS
    n_frames             : int unsigned
    frac_ok              : float
    frac_partial         : float
    frac_no_reflection   : float
    frac_tracking_lost   : float
    """

    @property
    def key_source(self):
        """Eye recordings only, crossed with the parameter sets.

        Without this, the default key source is every ``PoseEstimationNew`` row crossed
        with every parameter set: 4,779 x 4 = 19,116 jobs spanning 27 DeepLabCut models.
        Only 9 of those models are eye models, so **11,184 of those jobs would raise** --
        :meth:`classify` requires ``IR``, ``nose_corner`` and ``lateral_corner``, and a
        topcam or body model has none of them. The rest would quietly classify eight other
        eye models at all four parameter sets, writing thousands of rows nobody asked for.

        So the key source is restricted to models that actually carry the markers this
        table reads: the corneal reflection, both eye corners, and at least one pupil
        marker. Anything narrower than that -- a particular cohort, a particular parameter
        set -- belongs in the ``populate()`` restriction, not here.
        """
        model = _linking_module.model
        pupil_tracking = _linking_module.pupil_tracking

        bp = model.Model.BodyPart
        eye_models = (dj.U("model_name") & (bp & 'body_part = "IR"')) \
            & (dj.U("model_name") & (bp & 'body_part = "nose_corner"')) \
            & (dj.U("model_name") & (bp & 'body_part = "lateral_corner"')) \
            & (dj.U("model_name") & (bp & 'body_part like "pupil%"'))

        return (model.PoseEstimationNew & eye_models) * _param_table

    @staticmethod
    def classify(markers: list[str], likelihood: np.ndarray, params: dict) -> np.ndarray:
        """Classify frames from a ``(T, K)`` likelihood array. Pure; no database access.

        ``markers`` names the columns of ``likelihood``, in its own order -- the mapping is
        taken from the names, never assumed.
        """
        idx = {m: i for i, m in enumerate(markers)}
        missing = {"IR", "nose_corner", "lateral_corner"} - set(idx)
        if missing:
            raise ValueError(f"reference markers absent from this model: {sorted(missing)}")

        pupil_cols = [i for m, i in idx.items() if m.startswith("pupil_")]
        if not pupil_cols:
            raise ValueError("no pupil_* markers; this is not an eye model")

        n_pupil = (likelihood[:, pupil_cols] >= params["llh_thres_pupil"]).sum(axis=1)
        ir_ok = likelihood[:, idx["IR"]] >= params["llh_thres_ir"]
        corner_ok = (
            likelihood[:, [idx["nose_corner"], idx["lateral_corner"]]]
            >= params["llh_thres_corner"]
        ).any(axis=1)

        return np.where(
            n_pupil >= params["pupil_min_dots"], FRAME_CLASS["ok"],
            np.where(ir_ok, FRAME_CLASS["partial"],
                     np.where(corner_ok, FRAME_CLASS["no_reflection"],
                              FRAME_CLASS["tracking_lost"])),
        ).astype(np.uint8)

    def make(self, key):
        model = _linking_module.model
        pupil_tracking = _linking_module.pupil_tracking

        params = fetch_params(key)
        pose_key = {k: key[k] for k in model.PoseEstimationNew.primary_key}

        # order_by is not cosmetic: DataJoint's fetch order is not contractual, and
        # pairing an array with the wrong marker name is the most damaging silent error
        # available here.
        rows = (model.PoseEstimationNew.BodyPartPosition & pose_key).fetch(
            "body_part", "likelihood", as_dict=True, order_by="body_part"
        )
        markers = [r["body_part"] for r in rows]
        likelihood = np.stack([np.asarray(r["likelihood"], float) for r in rows], axis=1)

        frame_class = self.classify(markers, likelihood, params)
        n = int(frame_class.size)
        frac = {name: float((frame_class == code).mean()) for name, code in FRAME_CLASS.items()}

        self.insert1({
            **key,
            "frame_class": frame_class,
            "n_frames": n,
            "frac_ok": frac["ok"],
            "frac_partial": frac["partial"],
            "frac_no_reflection": frac["no_reflection"],
            "frac_tracking_lost": frac["tracking_lost"],
        })


# ======================================================================================
# Imputation
# ======================================================================================


def _run_imputation(table, key, *, task_table, quality_table, fit_params):
    """The whole imputation, shared by the free-moving and head-fixed chains.

    Lifted verbatim out of ``DLCImputation.make()`` when the head-fixed sibling was added.
    Only four things differ between the two chains -- which task table to read, which
    frame-quality table, how to fetch the ellipse parameters, and which table to insert
    into -- so those are arguments and everything else is identical by construction. A
    second copy of these 250 lines would drift, and the parts most likely to drift are the
    alignment gate and the gap bookkeeping, which are exactly the parts that must not.

    ``table`` is the populating table instance; its ``BodyPart`` and ``Gap`` parts receive
    the per-marker arrays and the per-gap records.
    """
    """Impute one recording, then record what was filled, refused and left alone.

    The DISK half runs as a subprocess in the ``disk`` environment -- see
    :data:`DISK_PYTHON` for why. Everything before and after it is DataJoint work and
    happens here.
    """
    import json
    import shutil
    import subprocess
    import tempfile
    import time

    model = _linking_module.model
    pupil_tracking = _linking_module.pupil_tracking

    task = (task_table & key).fetch1()
    mdl = (DISKModel & key).fetch1()
    pset = (ImputationParamSet & key).fetch1()
    fitp = fit_params(key)

    if task["task_mode"] != "trigger":
        raise NotImplementedError("only task_mode='trigger' is implemented")
    if not mdl["mu_sigma"]:
        raise ValueError(
            f"{mdl['disk_model_name']} was trained without mu_sigma, so it reports no "
            "uncertainty and Gap.uncertainty could not be filled honestly")

    pose_key = {k: key[k] for k in model.PoseEstimationNew.primary_key}

    # -- 1. markers, reindexed onto the model's own contract ----------------------
    # Never onto fetch order: DataJoint's ordering is not contractual, and pairing an
    # array with the wrong marker name is the most damaging silent error available.
    contract = DISKModel.keypoint_order(key["disk_model_name"])
    rows = (model.PoseEstimationNew.BodyPartPosition & pose_key).fetch(
        "body_part", "x_pos", "y_pos", "likelihood", as_dict=True,
        order_by="body_part")
    by_name = {r["body_part"]: r for r in rows}
    # The contract must be a SUBSET of what the recording carries, not equal to it. A
    # missing contract marker is still fatal -- the model cannot run without its inputs --
    # but extra markers are fine and are stored untouched, because the downstream ellipse
    # fit loads every marker the production fit loads, contract or not.
    absent = sorted(set(contract) - set(by_name))
    if absent:
        raise ValueError(
            f"recording is missing {absent}, which {mdl['disk_model_name']}'s keypoint "
            f"contract requires. It has {sorted(set(by_name))}.")
    extra = [m for m in sorted(set(by_name) - set(contract))]
    xy = np.stack([np.c_[np.asarray(by_name[m]["x_pos"], float),
                         np.asarray(by_name[m]["y_pos"], float)] for m in contract],
                  axis=1)
    lik = np.stack([np.asarray(by_name[m]["likelihood"], float) for m in contract],
                   axis=1)
    n_frames, n_kp = lik.shape

    # -- 2. the rate gate ---------------------------------------------------------
    # Read, never inferred from frame indices. A 50 Hz model applied to a 60 Hz
    # recording is a transfer test the evaluation never scored.
    fps = float((model.RecordingInfoNew & pose_key).fetch1("fps"))
    if abs(fps - float(mdl["training_fps"])) > 0.5:
        raise ValueError(
            f"recording is {fps} Hz but {mdl['disk_model_name']} was trained at "
            f"{mdl['training_fps']} Hz. Applying it across rates is untested; register "
            "a model trained at this rate instead.")

    # -- 3. classification: refuse before imputing --------------------------------
    frame_class = (quality_table & key).fetch1("frame_class")
    refuse_codes = []
    if pset["refuse_no_reflection"]:
        refuse_codes.append(FRAME_CLASS["no_reflection"])
    if pset["refuse_tracking_lost"]:
        refuse_codes.append(FRAME_CLASS["tracking_lost"])
    refused_frame = np.isin(frame_class, refuse_codes)

    # -- 4. what counts as missing ------------------------------------------------
    # DISK masks with <= , not < . A sample at exactly the threshold is missing, and
    # using < here would score a different set of gaps than DISK fills.
    thres = float(mdl["likelihood_threshold"])
    missing = lik <= thres
    src = np.full((n_frames, n_kp), SAMPLE_SOURCE["observed"], dtype=np.uint8)

    # Refusal is a per-FRAME judgement -- PupilFrameQuality classifies each frame -- and
    # the original rule applied it per GAP: one refused frame anywhere in a run withheld
    # the whole run from the network. Measured over 40 recordings that rule accounts for
    # 76.3% of all withheld frames, and the runs it withholds have a median length of 8
    # frames, comfortably inside the model's window. They were discarded for being
    # contaminated, not for being hard.
    #
    # `split_gaps_at_refused` keeps the refused frames refused and judges the stretches
    # between them on their own length. Gap rows still partition the missing frames
    # exactly, which is what makes DLCImputation.Gap usable as the provenance index.
    #
    # Measured headroom before doing it, on the same 40 recordings: 19.9% of missing
    # samples sit in a refused gap without being refused themselves, and 14.8% of missing
    # samples would come back inside the window -- 28.5% filled today against a 43.2%
    # ceiling, so about 1.52x as many samples offered to the network. Those sub-runs have
    # a median length of 2 frames, which is the regime where linear interpolation is
    # already close, so more filled markers does not imply a different answer downstream.
    split = bool(pset.get("split_gaps_at_refused", 0))
    max_gap = int(pset["max_gap_frames"])

    gaps = []           # (marker_index, start, length, status)
    submit = missing.copy()
    for j in range(n_kp):
        for start, length in _runs(missing[:, j]):
            sl = slice(start, start + length)
            if not refused_frame[sl].any():
                status = "too_long" if length > max_gap else "filled"
                if status != "filled":
                    submit[sl, j] = False
                gaps.append([j, start, length, status])
                continue

            if not split:
                submit[sl, j] = False
                gaps.append([j, start, length, "refused"])
                continue

            # Split the run into maximal stretches of equal refusal status, and judge each
            # on its own. A refused stretch stays refused; a clean one is offered if it
            # fits the window.
            local = refused_frame[start:start + length]
            i = 0
            while i < length:
                k = i
                while k < length and bool(local[k]) == bool(local[i]):
                    k += 1
                sub_len, s0 = k - i, start + i
                ssl = slice(s0, s0 + sub_len)
                if bool(local[i]):
                    submit[ssl, j] = False
                    gaps.append([j, s0, sub_len, "refused"])
                elif sub_len > max_gap:
                    submit[ssl, j] = False
                    gaps.append([j, s0, sub_len, "too_long"])
                else:
                    gaps.append([j, s0, sub_len, "filled"])
                i = k

    # -- 5. hand the masked recording to DISK -------------------------------------
    xy_in = xy.copy()
    xy_in[submit] = np.nan
    lik_in = lik.copy()
    lik_in[submit] = 0.0

    root = Path(dj.config.get("custom", {}).get("disk_work_root", DISK_WORK_ROOT))
    jobdir = Path(tempfile.mkdtemp(prefix="imp_", dir=str(root / "jobs")))
    try:
        _write_dlc_csv(jobdir / "input.csv", contract, xy_in, lik_in)
        (jobdir / "job.json").write_text(json.dumps({
            "length": int(mdl["seq_length"]),
            "stride": int(mdl["seq_length"]) // 2,
            "fps": fps,
            "likelihood_threshold": thres,
            "checkpoint": mdl["checkpoint_path"],
            "model_name": mdl["disk_model_name"],
            "batch_size": 64,
        }))
        worker = Path(__file__).with_name("disk_worker.py")
        python = dj.config.get("custom", {}).get("disk_python", DISK_PYTHON)
        t0 = time.time()
        proc = subprocess.run([python, str(worker), str(jobdir)],
                              capture_output=True, text=True)
        runtime = time.time() - t0
        if proc.returncode != 0:
            raise RuntimeError(
                f"disk_worker failed ({proc.returncode}).\n"
                f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}\n"
                f"job kept at {jobdir}")
        xy_out = _read_dlc_csv(jobdir / "imputed.csv", contract)
        seg = json.loads((jobdir / "gaps.json").read_text())
    finally:
        keep = str(jobdir)

    if xy_out.shape != xy.shape:
        raise RuntimeError(f"DISK changed the shape: {xy.shape} -> {xy_out.shape}; "
                           f"job kept at {keep}")

    # -- 6. the gate: nothing we did not ask about may have moved -----------------
    # A column shift is invisible -- plausible coordinates, a successful ellipse fit,
    # smooth angles, all belonging to the wrong markers. So it is asserted, not
    # trusted.
    # DISK is entitled to change exactly two kinds of sample: the ones we
    # blanked, and the recording's own sub-threshold samples, which it masks
    # internally at the same threshold and imputes as well. Everything else --
    # every coordinate DeepLabCut was confident about -- must come back
    # bit-identical, or a marker column has shifted.
    protected = ~missing & np.isfinite(xy).all(axis=2)
    # The tolerance is not slack. Exact equality is unattainable: DISK normalises
    # the whole array before inference and de-normalises afterwards, so even a
    # sample it never touched comes back through x/s*s, which is not the identity in
    # floating point. Measured on this cohort that noise is ~1e-13 px, roughly
    # machine epsilon against a 500-px coordinate.
    #
    # A real column shift is not subtle. When this gate first fired on a genuine
    # fault it reported 545 px. So 1e-6 px sits about seven orders of magnitude
    # below anything meaningful and seven above the noise, and the gate keeps all
    # of its power.
    if protected.any():
        delta = np.abs(xy[protected] - xy_out[protected])
        worst = float(np.nanmax(delta)) if delta.size else 0.0
        if worst > ALIGNMENT_TOLERANCE_PX:
            n_bad = int((delta > ALIGNMENT_TOLERANCE_PX).any(axis=1).sum())
            raise RuntimeError(
                f"alignment gate failed: {n_bad} protected samples moved by more "
                f"than {ALIGNMENT_TOLERANCE_PX:g} px, worst {worst:g} px. A marker "
                f"column has shifted and every downstream number would be "
                f"meaningless. Job kept at {keep}")

    # -- 7. assemble --------------------------------------------------------------
    filled = submit & np.isfinite(xy_out).all(axis=2)
    xy_eff = np.where(filled[:, :, None], xy_out, xy)
    lik_eff = lik.copy()
    lik_eff[filled] = 1.0          # effective: what downstream thresholds on

    src[missing] = SAMPLE_SOURCE["unfilled"]
    src[refused_frame[:, None] & missing] = SAMPLE_SOURCE["refused"]
    src[filled] = SAMPLE_SOURCE["disk"]
    if pset["keep_dlc_low_conf"]:
        # Deliberately narrow, and off by default: see the ImputationParamSet
        # docstring. It only fires where DISK declined, which is the regime where a
        # rejected DeepLabCut point is least reliable.
        keepable = missing & ~filled & ~refused_frame[:, None] & np.isfinite(xy).all(axis=2)
        xy_eff = np.where(keepable[:, :, None], xy, xy_eff)
        lik_eff[keepable] = lik[keepable]
        src[keepable] = SAMPLE_SOURCE["dlc_low_kept"]

    # A submitted gap can come back PARTIALLY filled -- `filled` is computed per frame
    # -- so one status per original gap cannot describe it. The first version stamped
    # the whole segment 'unfilled' whenever any frame was missed, which silently
    # contradicted sample_source: a gap of ten frames with eight filled appeared as
    # ten unfilled frames in Gap and eight 'disk' frames in BodyPart. Anything reading
    # Gap as the SQL-queryable index of provenance -- which is what it is for -- would
    # then under-report what was imputed.
    #
    # So a submitted gap is split into contiguous runs of equal outcome and one row is
    # emitted per run. Gap rows then partition the missing frames exactly, and the
    # union of 'filled' rows equals sample_source == disk by construction rather than
    # by coincidence. Refused and too-long gaps are uniform and stay whole.
    def _uncertainty(start, length):
        """Worst uncertainty among the DISK segments overlapping this run.

        Per DISK segment rather than per marker-gap: one badly guessed marker is what
        damages the ellipse fit, so the run takes the worst it overlaps.
        """
        over = [s["uncertainty"] for s in seg
                if s["uncertainty"] is not None
                and s["start"] < start + length and s["start"] + s["length"] > start]
        return max(over) if over else None

    resolved = []
    for j, start, length, status in gaps:
        if status != "filled":                      # refused / too_long: uniform
            resolved.append([j, start, length, status, _uncertainty(start, length)])
            continue
        col = filled[start:start + length, j]
        run_start = 0
        while run_start < length:
            val = col[run_start]
            run_end = run_start
            while run_end < length and col[run_end] == val:
                run_end += 1
            s0, ln = start + run_start, run_end - run_start
            resolved.append([j, s0, ln, "filled" if val else "unfilled",
                             _uncertainty(s0, ln) if val else None])
            run_start = run_end
    gaps = resolved

    before = _yields_angle(lik, contract, fitp)
    after = _yields_angle(lik_eff, contract, fitp)
    filled_lengths = [g[2] for g in gaps if g[3] == "filled"]

    table.insert1({
        **key,
        "imputation_time": datetime.datetime.now(),
        "recording_fps": fps,
        "production_fit_hash": production_fit_hash(),
        # Counted over the CONTRACT markers only. With a subset contract, folding in a
        # marker the model never touched would distort every counter here: in the
        # head-fixed cohort nose_corner is above threshold in only 24% of frames on
        # average, so including it would drive n_frames_observed toward zero while saying
        # nothing whatever about how well the imputation worked.
        "n_frames_total": n_frames,
        "n_frames_observed": int((src == SAMPLE_SOURCE["observed"]).all(axis=1).sum()),
        "n_frames_imputed": int((src == SAMPLE_SOURCE["disk"]).any(axis=1).sum()),
        "n_frames_refused": int((src == SAMPLE_SOURCE["refused"]).any(axis=1).sum()),
        "n_frames_unfilled": int((src == SAMPLE_SOURCE["unfilled"]).any(axis=1).sum()),
        "n_gaps_attempted": int(sum(1 for g in gaps if g[3] in ("filled", "unfilled"))),
        "n_gaps_too_long": int(sum(1 for g in gaps if g[3] == "too_long")),
        "max_gap_filled": int(max(filled_lengths)) if filled_lengths else 0,
        "n_angle_before": int(before.sum()),
        "n_angle_after": int(after.sum()),
        "disk_runtime_s": round(runtime, 1),
    })
    body_rows = [
        {**key, "body_part": m,
         "x_pos": xy_eff[:, i, 0], "y_pos": xy_eff[:, i, 1],
         "likelihood": lik_eff[:, i], "sample_source": src[:, i]}
        for i, m in enumerate(contract)]
    # Markers outside the contract are stored exactly as DeepLabCut produced them -- same
    # coordinates, same likelihood, no imputation -- flagged not_modelled wherever they
    # fall below the threshold so that nothing downstream can mistake them for something
    # the model considered and declined.
    for m in extra:
        r = by_name[m]
        lk = np.asarray(r["likelihood"], float)
        s = np.where(lk <= thres, SAMPLE_SOURCE["not_modelled"],
                     SAMPLE_SOURCE["observed"]).astype(np.uint8)
        body_rows.append({**key, "body_part": m,
                          "x_pos": np.asarray(r["x_pos"], float),
                          "y_pos": np.asarray(r["y_pos"], float),
                          "likelihood": lk, "sample_source": s})
    table.BodyPart.insert(body_rows)
    table.Gap.insert([
        {**key, "body_part": contract[j], "gap_start": start,
         "gap_length": length, "uncertainty": unc, "gap_status": status}
        for j, start, length, status, unc in gaps])

    shutil.rmtree(keep, ignore_errors=True)


def _run_imputed_fit(table, key, *, imputation_table, production_table, fit_params,
                     production_key=lambda k: k):
    """The imputed ellipse fit, shared by the free-moving and head-fixed chains.

    Lifted verbatim out of ``PupilEllipseFittingImputed.make()``. Three things differ --
    which imputation table supplies the markers, which production table supplies the
    calibration to inherit, and how the ellipse parameters are fetched. The estimator
    itself, the inherited calibration, and both gates are identical for the two chains,
    and must stay that way: the whole point of the sibling is that it is comparable to its
    production twin.
    """
    """Refit the ellipse on imputed markers, holding the production calibration fixed.

    The conic fit is the production one, extracted from ``pupil_tracking.py`` at run
    time rather than reimplemented, so this table and its production twin measure the
    same estimator. Two gates protect the comparison; both raise rather than warn,
    because a silent failure here produces smooth, plausible angles that are wrong.
    """
    model = _linking_module.model
    pupil_tracking = _linking_module.pupil_tracking

    params = fit_params(key)

    def _stack(rows, what):
        by = {r["body_part"]: r for r in rows}
        miss = [m for m in PRODUCTION_ORDER if m not in by]
        if miss:
            raise ValueError(f"{what} is missing markers {miss} for {key}")
        names = list(by)
        xy = np.stack([np.c_[np.asarray(by[m]["x_pos"], float),
                             np.asarray(by[m]["y_pos"], float)] for m in names], axis=1)
        lik = np.stack([np.asarray(by[m]["likelihood"], float) for m in names], axis=1)
        return names, xy, lik

    # -- 1. imputed markers, and the originals production saw ----------------------
    # Both are needed. The imputed ones are what this table fits; the original ones
    # are what the inherited calibration was estimated from, and the only way to
    # recover it from what production stored.
    names_i, xy_i, lik_i = _stack(
        (imputation_table.BodyPart & key).fetch(
            "body_part", "x_pos", "y_pos", "likelihood", as_dict=True,
            order_by="body_part"), "%s.BodyPart" % imputation_table.__name__)

    pose_key = {k: key[k] for k in model.PoseEstimationNew.primary_key}
    names_o, xy_o, lik_o = _stack(
        (model.PoseEstimationNew.BodyPartPosition & pose_key).fetch(
            "body_part", "x_pos", "y_pos", "likelihood", as_dict=True,
            order_by="body_part"), "PoseEstimationNew.BodyPartPosition")

    pc_x, pc_y, ir_xg, ir_yg = _mask_and_center(xy_i, lik_i, names_i, params)
    pc_x_o, pc_y_o, ir_xg_o, ir_yg_o = _mask_and_center(xy_o, lik_o, names_o, params)

    # -- 2. inherit the calibration -----------------------------------------------
    # From the MIRRORED production set, not from this key's own parameter_id -- see
    # PRODUCTION_MIRROR. `prod_key` is this same recording; only the set id differs, and
    # the two sets carry identical thresholds.
    prod_key = production_key(key)
    cam_center_prod, eye_center_prod, scale = (
        production_table & prod_key).fetch1(
            "cam_center", "eye_center", "scale")
    scale = float(scale)
    cam_cent, eye_cent = _recover_calibration(
        np.asarray(cam_center_prod, float), np.asarray(eye_center_prod, float),
        ir_xg_o, ir_yg_o)

    horizontal_offset = float(np.arcsin((eye_cent[0] - cam_cent[0]) / scale))
    vertical_offset = float(np.arcsin(
        (eye_cent[1] - cam_cent[1]) / np.cos(horizontal_offset) / scale))

    # -- 3. gate: does the reused fit reproduce production's own angles? -----------
    # Run the extracted fit over the ORIGINAL markers under the inherited
    # calibration. That is precisely what production did, so it must return
    # precisely what production stored. If it does not, the masking, the marker
    # ordering or the fit differs from production somewhere, and every comparison
    # this table exists to support would be measuring that difference instead of
    # imputation. This pass also supplies n_nice_frames honestly -- with an
    # inherited calibration, the frames that contributed to it are production's,
    # not this row's.
    X0_o, Y0_o, la_o, sa_o, _ = _fit_frames(pc_x_o, pc_y_o, params)
    theta_o, _ = _to_angles(X0_o, Y0_o, cam_cent, scale,
                            horizontal_offset, vertical_offset)
    theta_prod = np.asarray(
        (production_table & prod_key).fetch1("theta"), float)
    _check_reproduces_production(theta_o, theta_prod, key)
    n_nice_frames = int(_nice_mask(pc_x_o, la_o, sa_o, params).size)

    # -- 4. the fit this table is for ---------------------------------------------
    X0_in, Y0_in, long_axis, short_axis, angle_from_x = _fit_frames(pc_x, pc_y, params)
    theta, phi = _to_angles(X0_in, Y0_in, cam_cent, scale,
                            horizontal_offset, vertical_offset)

    # How much of each frame's evidence is invented. Without this the table cannot
    # distinguish an angle fitted through eight real markers from one fitted through
    # eight synthetic ones -- measured on Natasha's cohort, 2.1% of frames that yield
    # an angle would be fitted entirely from DISK output, and nothing downstream could
    # tell. Counted over the markers that actually entered the fit, so a marker DISK
    # filled but that still failed the threshold is not counted.
    # Counted over the eight PUPIL markers, in PRODUCTION_ORDER, which are the ones
    # the conic is fitted through -- and only those that actually entered the fit, so
    # a marker DISK filled that still failed the threshold is not counted. The
    # corneal reflection is deliberately excluded: it is the origin the pupil
    # coordinates are expressed relative to, not a point on the ellipse. An imputed
    # IR shifts every coordinate in the frame rather than adding one of eight, so
    # folding it into the same count would understate it; read it from the imputation
    # table's BodyPart part when that matters.
    src = {r["body_part"]: np.asarray(r["sample_source"]) for r in
           (imputation_table.BodyPart & key).fetch(
               "body_part", "sample_source", as_dict=True)}
    synth = np.stack([src[m] == SAMPLE_SOURCE["disk"] for m in PRODUCTION_ORDER[3:]],
                     axis=1)
    used = ~np.isnan(pc_x)                        # (T, 8): entered the conic fit
    n_imputed = (used & synth).sum(axis=1).astype(np.uint8)

    # Un-centred back into original pixel coordinates with the IMPUTED reflection,
    # which is the same convention production uses with its own -- and is part of the
    # point, since a recovered IR is one of the ways a frame is won back.
    table.insert1({
        **key,
        "theta": theta,
        "phi": phi,
        "diameter": long_axis,
        "pupil_center": np.column_stack((X0_in + ir_xg, Y0_in + ir_yg)),
        "x0_in": X0_in,
        "y0_in": Y0_in,
        "long_axis": long_axis,
        "short_axis": short_axis,
        "angle_from_x": angle_from_x,
        "cam_center": np.column_stack((cam_cent[0] + ir_xg, cam_cent[1] + ir_yg)),
        "eye_center": np.column_stack((eye_cent[0] + ir_xg, eye_cent[1] + ir_yg)),
        "scale": scale,
        "horizontal_offset": horizontal_offset,
        "vertical_offset": vertical_offset,
        "n_nice_frames": n_nice_frames,
        "n_imputed_markers": n_imputed,
        "n_angle": int(np.isfinite(theta).sum()),
        "calibration_source": "inherited",
    })


@schema
class DLCImputationTask(dj.Manual):
    """One imputation to perform: a recording, a threshold, a model, a policy.

    All four are in the primary key, which is what makes multiple imputations of the same
    recording possible without collision -- a transformer result beside a GRU result, or
    the same model at two thresholds, or under two refusal policies. Because
    :class:`DLCImputation` is Computed from this Manual table, nothing multiplies on its
    own: a recording gets exactly the variants someone inserts task rows for.
    """

    definition = """
    -> model.PoseEstimationNew
    -> PupilEllipseParameterFreeMoving                    # what counts as missing
    -> DISKModel                                          # which trained network
    -> ImputationParamSet                                 # refusal policy, gap ceiling
    ---
    task_mode='trigger'  : enum('load','trigger')  # 'load': read prepared output; 'trigger': run inference
    output_dir=''        : varchar(512)
    task_description=''  : varchar(255)
    """


@schema
class DLCImputation(dj.Computed):
    """Imputed DeepLabCut markers, with per-sample provenance and per-gap uncertainty.

    The counters are chosen to answer the question the project was actually about.
    ``n_angle_before`` and ``n_angle_after`` are the yield -- how many frames produced an
    eye angle from the original markers, and how many do after imputation. The previous
    draft recorded ``n_frames_imputed`` and ``n_values_imputed``, neither of which anyone
    wants to know.

    ``production_fit_hash`` is the SHA-256 of the production ellipse fit as extracted at
    run time, so a change to that estimator makes these rows visibly stale rather than
    quietly wrong.
    """

    definition = """
    -> DLCImputationTask
    ---
    imputation_time      : datetime
    recording_fps        : decimal(7,3)   # validated against DISKModel.training_fps
    production_fit_hash  : char(64)       # SHA-256 of the extracted production ellipse fit
    n_frames_total       : int unsigned
    n_frames_observed    : int unsigned   # every marker above threshold
    n_frames_imputed     : int unsigned   # at least one marker filled by DISK
    n_frames_refused     : int unsigned   # at least one marker refused
    n_frames_unfilled    : int unsigned   # at least one marker left for gap length
    n_gaps_attempted     : int unsigned
    n_gaps_too_long      : int unsigned   # exceeded seq_length: the untested regime
    max_gap_filled       : smallint unsigned
    n_angle_before       : int unsigned   # frames yielding an eye angle from the originals
    n_angle_after        : int unsigned   # ... and after imputation. The yield.
    disk_runtime_s=null  : float
    """

    class BodyPart(dj.Part):
        """Per-marker arrays. The heavy table; every array is on the external store.

        ``original_likelihood`` is deliberately absent: it is a verbatim copy of a column
        in the foreign-keyed parent row, recoverable by join, and at roughly 0.23 MB per
        marker per recording it is pure duplication. ``frame_index`` is absent for the same
        reason.
        """

        definition = """
        -> master
        -> model.Model.BodyPart
        ---
        x_pos         : blob@external-raw   # effective coordinates: observed or filled
        y_pos         : blob@external-raw
        likelihood    : blob@external-raw   # EFFECTIVE likelihood -- what downstream thresholds on
        sample_source : blob@external-raw   # uint8 per frame, see SAMPLE_SOURCE
        """

    class Gap(dj.Part):
        """One row per contiguous missing segment, per marker.

        This is where DISK's uncertainty honestly belongs. The network is trained with
        ``mu_sigma`` and produces a sigma per sample, but DISK collapses it to one scalar
        per imputed segment before exposing it -- so a per-frame array would be a claim the
        data does not support.

        Deliberately all-scalar, and therefore in MySQL rather than on the store. That
        makes it a queryable index into data that now lives on a file server: "every gap
        longer than 30 frames whose uncertainty exceeds x" is answered entirely in SQL,
        without opening a single blob.

        Use the uncertainty to flag, not to discard. It predicts the model's own error at
        Spearman 0.59, but keeping only the 80% most confident gaps buys a 10% error
        reduction for a fifth of the data.

        ``gap_status`` has four values, parallel to :data:`SAMPLE_SOURCE`:

        ``filled``
            DISK returned a value for every frame of the gap.
        ``unfilled``
            Submitted and inside the model's window, but DISK declined it.
        ``too_long``
            Longer than ``max_gap_frames``; never submitted. 44% of the cohort's missing
            frames are in gaps this long, so this is not a rare case.
        ``refused``
            Overlapped a frame classified ``no_reflection`` or ``tracking_lost``; never
            submitted, because there is no pupil to reconstruct.
        """

        definition = """
        -> master
        -> model.Model.BodyPart
        gap_start        : int unsigned      # first missing frame of the segment
        ---
        gap_length       : smallint unsigned
        uncertainty=null : float             # DISK's per-segment sigma; null if not filled
        gap_status       : enum('filled','unfilled','too_long','refused')
                                             # 'unfilled': submitted and within the
                                             # window, but DISK declined it
        """

    @property
    def key_source(self):
        """Only recordings whose frames have been classified.

        Refusal comes before imputation, so an unclassified recording is not ready.
        """
        return DLCImputationTask & PupilFrameQuality.proj()

    def make(self, key):
        """Impute one recording. See :func:`_run_imputation`, which carries the body."""
        _run_imputation(self, key, task_table=DLCImputationTask,
                        quality_table=PupilFrameQuality, fit_params=fetch_params)


# ======================================================================================
# Downstream -- the A1 sibling of the production ellipse fit
# ======================================================================================


@pupil_schema
class PupilEllipseFittingImputed(dj.Computed):
    """Eye angles from imputed markers, running beside the production fit.

    Decision A1: a sibling chain rather than a pose-source dimension on
    ``PupilEllipseFittingFreeMoving``. Adding a key attribute to that table is not possible
    in place -- DataJoint's ``alter()`` changes secondary attributes only -- so it would
    have meant dropping and repopulating roughly 9.9 GB of production gaze data. A sibling
    also keeps both fits queryable side by side, which is what reproducing the evaluation's
    comparative claim (0.32 degrees against 1.77) actually requires.

    ``parameter_id`` arrives through the ``DLCImputation`` key, so this table joins to its
    production twin on ``(parameter_id, session_id, scan_id, recording_id, model_name)``.

    **Inherit the calibration; do not re-estimate it.** The production fit estimates the
    recording's camera centre and pixel-to-angle scale from whichever frames survive
    thresholding and carry at least 7 dots. Winning frames back changes which frames
    qualify, which moves the calibration, which shifts *every* angle in the recording --
    including frames DISK never touched. A naive refit therefore produces angles that
    differ from production even on untouched frames, and the difference is a recalibration
    artefact rather than an improvement. ``calibration_source`` records which was done;
    ``'inherited'`` is correct for anything compared against production.

    ``ellipse_dict`` is deliberately not stored. The production table keeps a Python list
    holding one 16-key dictionary per frame -- roughly two-thirds of its 9.3 MB per row --
    and downstream reads five of those sixteen fields. Those five are stored as arrays
    instead, at roughly 1.6 MB per row. The production table is not touched.
    """

    definition = """
    -> DLCImputation
    ---
    theta               : blob@external-raw   # azimuth, radians, relative to eye centre
    phi                 : blob@external-raw   # elevation, radians
    diameter            : blob@external-raw   # ellipse semi-major axis, px (named as in production)
    pupil_center        : blob@external-raw   # (T, 2) in original pixel coordinates
    x0_in               : blob@external-raw   # the five fit outputs downstream actually reads,
    y0_in               : blob@external-raw   # stored as arrays instead of a list of dicts.
    long_axis           : blob@external-raw   # NB lower case: DataJoint attribute names must
    short_axis          : blob@external-raw   # be lower case, so the production spellings
    angle_from_x        : blob@external-raw   # X0_in / Y0_in cannot be used as columns --
                                              # they are dict keys there, not attributes.
    cam_center          : blob@external-raw
    eye_center          : blob@external-raw
    scale               : float
    horizontal_offset   : float
    vertical_offset     : float
    n_nice_frames       : int unsigned        # frames that contributed to the calibration
    n_angle             : int unsigned        # frames yielding a finite theta
    n_imputed_markers   : blob@external-raw   # per frame: how many of the markers this
                                              # frame was fitted through came from DISK
                                              # rather than from DeepLabCut
    calibration_source  : enum('inherited','reestimated')
    """

    @property
    def key_source(self):
        """Only imputations whose MIRRORED production twin exists, because calibration is inherited.

        A recording with no :class:`PupilEllipseFittingFreeMoving` row at the production
        set this imputation set mirrors (:data:`PRODUCTION_MIRROR`) has no camera centre
        or scale to inherit, and re-estimating one is not an option -- see the class
        docstring. Such a recording is therefore not *ready*, in exactly the sense that an
        unclassified recording is not ready for :class:`DLCImputation`, so it is skipped
        rather than failed.

        This reads ACROSS the mirror -- set 5 looks for a set-1 fit, set 6 for a set-2 one
        -- rather than requiring a twin at its own id. The version before 2026-09-16
        required the same id, which would have meant first populating the production
        estimator at sets 5 and 6; see :data:`PRODUCTION_MIRROR` for why that was rejected.
        Nothing here writes to, or requires a write to, any production table.
        """
        pupil_tracking = _linking_module.pupil_tracking
        P = pupil_tracking.PupilEllipseFittingFreeMoving

        # One OR-branch per mirror entry: "this imputation set, AND a production fit of the
        # same recording at the set it mirrors". `parameter_id` is renamed on the
        # production side so it does not join against the imputation's own -- what remains
        # common is (session_id, scan_id, recording_id, model_name), which is the recording.
        branches = []
        for imp_id, prod_id in PRODUCTION_MIRROR.items():
            twin = (P & {"parameter_id": prod_id}).proj(production_parameter_id="parameter_id")
            branches.append(dj.AndList([{"parameter_id": imp_id}, twin]))
        return DLCImputation & branches

    def make(self, key):
        """Fit the imputed markers. See :func:`_run_imputed_fit`, which carries the body."""
        # The mirror is checked HERE, not only in the utility scripts that set it up. This
        # table masks its markers with set 5's thresholds and inherits its calibration from
        # set 1, and that is only legitimate while the two are the same estimator. Without
        # this call a populate() run after somebody edited either lookup row would produce
        # rows that look like valid comparisons and are not. `_check_reproduces_production`
        # further down would eventually catch a real divergence, but it would report it as
        # estimator drift, which is the wrong diagnosis and the wrong place to start looking.
        assert_mirror_is_a_copy()
        _run_imputed_fit(
            self, key, imputation_table=DLCImputation,
            production_table=_linking_module.pupil_tracking.PupilEllipseFittingFreeMoving,
            fit_params=fetch_params, production_key=production_key_for)


def _eye_and_optitrack_clocks(base_key, eyecam_time_source):
    """The eye-camera and OptiTrack frame timestamps, in session time.

    The production twin of this is ``pupil_tracking.fetch_eye_and_optitrack_clocks``. It is
    not called directly because this module reaches pupil_tracking as a virtual module under
    every script that runs it, and a virtual module carries tables but no functions. The
    part that is actually logic -- which event_type a recording's frames are under -- is
    shared through ``eye_geometry.eye_camera_event_type``; only the two fetches are here.
    """
    Event = upstream("Event")
    suffix = (upstream("EyeCamTimeSource")
              & {"eyecam_time_source": eyecam_time_source}).fetch1("event_suffix")
    event_type = eg.eye_camera_event_type(base_key["recording_id"], suffix)

    event_key = {k: base_key[k] for k in ("session_id", "scan_id") if k in base_key}
    t_eyecam = (Event & event_key
                & f'event_type="{event_type}"').fetch("event_start_time").astype(float)
    if len(t_eyecam) == 0:
        raise ValueError(f"No eye camera events found for {event_type} "
                         f"({base_key['recording_id']})")
    t_optitrack = (Event & event_key
                   & 'event_type="optitrack_frames"').fetch("event_start_time").astype(float)
    if len(t_optitrack) == 0:
        raise ValueError("No optitrack_frames events found for this session/scan")
    return t_eyecam, t_optitrack


def _eye_side_sign(recording_id, subject_name):
    """+1 left, -1 right, with the EyeSideNote correction applied exactly as production does.

    Mirrors ``pupil_tracking.eye_side_sign``. The label read is shared
    (``eye_geometry.side_sign_from_recording_id``); the note lookup is a table read and so
    lives with each chain. Both must stay in step: if the recordings are ever renamed and
    the note updated, a chain still applying the old correction flips the sign of every
    gaze vector it produces, and nothing downstream would raise.
    """
    s = eg.side_sign_from_recording_id(recording_id)
    notes = (upstream("EyeSideNote") & {"subject": subject_name}).fetch(
        as_dict=True, order_by="note_date DESC", limit=1)
    if notes and notes[0]["sides_swapped"]:
        s = -s
    return s


def _torsion(key):
    """Torsion from head pitch through the subject's manual linear calibration.

    Despite being stored per frame this is not a per-frame measurement: it is a linear
    function of pitch with per-animal coefficients, and carries nothing the pitch does not.
    """
    param_coeff = (upstream("TorsionCalibManual") & key).fetch1("param_coeff")
    param_intercept = (upstream("TorsionCalibManual") & key).fetch1("param_intercept")
    pitch = (upstream("RigidMouseTracking") & key).fetch1("pitch")
    return param_coeff * pitch + param_intercept


@pupil_schema
class PupilRotationOptiTrackImputed(dj.Computed):
    """The imputed eye trace on the OptiTrack clock -- sibling of ``PupilRotationOptiTrack``.

    Second of the three tables decision A1 requires. It exists for one reason: gaze in
    space is computed frame by frame against the head quaternion, which is sampled by
    OptiTrack, while the eye is sampled by its own camera at roughly 50 Hz. Something has
    to put the two on one clock, and that something must be the same something production
    uses or the imputed and production gaze would differ by their resampling as well as by
    imputation.

    So it is: ``make()`` calls ``eye_geometry.upsample_eye_trace``, which is the body of
    ``PupilRotationOptiTrack.make()`` moved into a database-free module for exactly this
    purpose, and which the production table now calls too. There is no second
    implementation to keep in step.

    Note what this does NOT do: gap-filling here runs over the imputed theta/phi, so frames
    DISK won back are real samples the interpolator now works from rather than across. That
    is the whole point, and it is also why the two chains cannot share rows -- the same
    interpolator over different inputs.
    """

    definition = """
    -> PupilEllipseFittingImputed
    -> EyeCamTimeSource
    ---
    theta_opt       : longblob   # interpolated upscaled horizontal orientation in radians
    phi_opt         : longblob   # interpolated upscaled vertical orientation in radians
    diameter_opt    : longblob   # interpolated upscaled pupil diameter in pixels
    """

    @property
    def key_source(self):
        """Only the (recording, time source) pairs production actually has, read across the mirror.

        ``PupilRotationOptiTrack``'s own key source is ``PupilEllipseFittingFreeMoving *
        EyeCamTimeSource``, the full cross product with all three time sources, and it
        relies on ``make()`` raising for the two that have no events. That leaves two failed
        jobs in ``~jobs`` for every one that works. Restricting to the pairs production
        already resolved gives the same rows without the noise, and guarantees every imputed
        row has a production row to be compared against.
        """
        pupil_tracking = _linking_module.pupil_tracking
        branches = []
        for imp_id, prod_id in PRODUCTION_MIRROR.items():
            prod = (pupil_tracking.PupilRotationOptiTrack
                    & {"parameter_id": prod_id}).proj(production_parameter_id="parameter_id")
            branches.append(dj.AndList([{"parameter_id": imp_id}, prod]))
        return (PupilEllipseFittingImputed * upstream("EyeCamTimeSource")) & branches

    def make(self, key):
        base_key = {k: v for k, v in key.items() if k != "eyecam_time_source"}
        theta, phi, diameter = (PupilEllipseFittingImputed & base_key).fetch1(
            "theta", "phi", "diameter")

        t_eyecam, t_optitrack = _eye_and_optitrack_clocks(
            base_key, key["eyecam_time_source"])

        theta_opt, phi_opt, diameter_opt = eg.upsample_eye_trace(
            theta, phi, diameter, t_eyecam, t_optitrack)

        self.insert1({**key, "theta_opt": theta_opt, "phi_opt": phi_opt,
                      "diameter_opt": diameter_opt})


@pupil_schema
class GazeReconstruction3DImputed(dj.Computed):
    """Gaze in head and in space from imputed markers -- sibling of ``GazeReconstruction3D``.

    Third and last of the tables decision A1 requires, and the one the whole exercise is
    for: this is where DISK's recovered eye angles become a gaze direction in the arena,
    which is what every downstream analysis -- retinal position, gaze-to-target error, the
    PSTHs -- actually reads.

    ``make()`` is its production twin's, with the geometry taken from
    ``eye_geometry.compute_gaze_vectors``, which both tables now call. The eye-side sign,
    including the ``EyeSideNote`` correction for subjects whose eye videos were misnamed at
    acquisition, is composed the same way in both chains: a second copy of that correction
    is a second place for it to be applied twice, or not at all.

    Everything except the eye trace is shared with production by construction -- the same
    head quaternions from the same ``RigidMouseTracking`` row, the same ``EyeModel``, the
    same per-animal torsion calibration. The difference between a row here and its
    production twin is therefore the imputation and nothing else, which is what makes the
    comparison worth drawing.
    """

    definition = """
    -> PupilRotationOptiTrackImputed
    -> RigidMouseTracking
    -> EyeModel
    -> TorsionCalibManual
    ---
    gaze_in_head    : longblob
    gaze_in_space   : longblob
    torsion         : longblob   # torsion angle of the eye
    """

    @property
    def key_source(self):
        """The mirrored production reconstructions, so the two sets are row-for-row comparable.

        Without this the join would also multiply by every ``EyeModel`` row -- there are
        two, one from Sakatani & Isa and one from Oommen & Stahl -- and reconstruct gaze
        under an eye model the production row for the same recording never used.
        """
        pupil_tracking = _linking_module.pupil_tracking
        base = (PupilRotationOptiTrackImputed
                * upstream("RigidMouseTracking")
                * upstream("EyeModel")
                * upstream("TorsionCalibManual"))
        branches = []
        for imp_id, prod_id in PRODUCTION_MIRROR.items():
            prod = (pupil_tracking.GazeReconstruction3D
                    & {"parameter_id": prod_id}).proj(production_parameter_id="parameter_id")
            branches.append(dj.AndList([{"parameter_id": imp_id}, prod]))
        return base & branches

    def make(self, key):
        eye_model_azimuth, eye_model_elevation = (upstream("EyeModel") & key).fetch1(
            "eye_model_azimuth", "eye_model_elevation")

        rec_id = (PupilRotationOptiTrackImputed & key).fetch1("recording_id")
        s = _eye_side_sign(rec_id, key["subject"])

        theta, phi = (PupilRotationOptiTrackImputed & key).fetch1("theta_opt", "phi_opt")

        qx, qy, qz, qw = (upstream("RigidMouseTracking") & key).fetch1(
            "q_x", "q_y", "q_z", "q_w")
        q_head_world_xyzw = np.column_stack([qx, qy, qz, qw])

        gaze_in_head, gaze_in_space = eg.compute_gaze_vectors(
            theta, phi, q_head_world_xyzw, eye_model_azimuth, eye_model_elevation, s)

        torsion = _torsion(key)

        self.insert1({**key, "gaze_in_head": gaze_in_head,
                      "gaze_in_space": gaze_in_space, "torsion": torsion})


# ======================================================================================
# The head-fixed sibling chain
# ======================================================================================
#
# Leon Kremers' bench2p cohort is recorded head-fixed and fitted by
# `pupil_tracking.PupilEllipseFitting`, not `PupilEllipseFittingFreeMoving`. That is a
# different chain from top to bottom -- a different parameter table with a different shape,
# a different production fit table to inherit calibration from -- and the two cannot be
# merged, because both parameter tables call their key `parameter_id` and a given id means
# something unrelated in each.
#
# What they CAN share is everything numerical, and they do: `_run_imputation` and
# `_run_imputed_fit` carry the whole body for both chains, and the helpers beneath them
# (`_mask_and_center`, `_fit_frames`, `_nice_mask`, `_recover_calibration`, `_to_angles`,
# `_check_reproduces_production`) are called unchanged. The head-fixed side supplies only
# a parameter adapter. The two production `fit_ellipse` implementations are byte-identical,
# which is what makes that legitimate, and `assert_fit_ellipse_shared()` fails loudly if
# that ever stops being true.


@schema
class HeadfixedImputationTask(dj.Manual):
    """One imputation to run: a head-fixed recording, a model, and a policy.

    The free-moving twin of this table is :class:`DLCImputationTask`. They are separate
    rather than one table with a discriminator column because the foreign key differs:
    this one points at `#pupil_ellipse_parameter`, that one at
    `#pupil_ellipse_parameter_free_moving`, and DataJoint has no way to express "one or the
    other" in a primary key.
    """

    definition = """
    -> PupilEllipseParameter                              # what counts as missing
    -> model.PoseEstimationNew
    -> DISKModel
    -> ImputationParamSet
    ---
    task_mode='trigger'  : enum('trigger', 'load')
    output_dir=''        : varchar(255)
    task_description=''  : varchar(255)
    """


@pupil_schema
class HeadfixedFrameQuality(dj.Computed):
    """Frame classification for the head-fixed chain.

    Identical logic to :class:`PupilFrameQuality` -- it calls that class's ``classify``
    directly -- but keyed to the head-fixed parameter table. A shared table is not possible:
    ``parameter_id`` is the primary key of both parameter tables and means different things
    in each, so one classification row could not say which it belonged to.

    Note that ``pupil_min_dots`` reaches into this classification. ``classify`` branches on
    it, and the head-fixed parameter table has no such column, so the adapter supplies the
    six that `PupilEllipseFitting.make()` hard-codes. A frame with exactly five good pupil
    dots is therefore ``partial`` here, which is what production would do with it.
    """

    definition = """
    -> PupilEllipseParameter
    -> model.PoseEstimationNew
    ---
    frame_class          : blob@external-raw   # uint8 per frame, see FRAME_CLASS
    n_frames             : int unsigned
    frac_ok              : float
    frac_partial         : float
    frac_no_reflection   : float
    frac_tracking_lost   : float
    """

    @property
    def key_source(self):
        """Eye recordings only, crossed with the head-fixed parameter sets.

        Same restriction as :attr:`PupilFrameQuality.key_source` and for the same reason:
        without it the default source is every pose estimation crossed with every parameter
        set, most of which are not eye models at all and would raise in ``classify``.
        """
        model = _linking_module.model
        bp = model.Model.BodyPart
        eye_models = (dj.U("model_name") & (bp & 'body_part = "IR"')) \
            & (dj.U("model_name") & (bp & 'body_part = "nose_corner"')) \
            & (dj.U("model_name") & (bp & 'body_part = "lateral_corner"')) \
            & (dj.U("model_name") & (bp & 'body_part like "pupil%"'))
        return (model.PoseEstimationNew & eye_models) * _headfixed_param_table

    def make(self, key):
        """Classify, using the same pure function the free-moving chain uses."""
        model = _linking_module.model

        params = fetch_headfixed_params(key)
        pose_key = {k: key[k] for k in model.PoseEstimationNew.primary_key}
        rows = (model.PoseEstimationNew.BodyPartPosition & pose_key).fetch(
            "body_part", "likelihood", as_dict=True, order_by="body_part")
        markers = [r["body_part"] for r in rows]
        likelihood = np.stack([np.asarray(r["likelihood"], float) for r in rows], axis=1)

        frame_class = PupilFrameQuality.classify(markers, likelihood, params)
        n = int(frame_class.size)
        frac = {name: float((frame_class == code).mean()) for name, code in FRAME_CLASS.items()}
        self.insert1({
            **key,
            "frame_class": frame_class,
            "n_frames": n,
            "frac_ok": frac["ok"],
            "frac_partial": frac["partial"],
            "frac_no_reflection": frac["no_reflection"],
            "frac_tracking_lost": frac["tracking_lost"],
        })


@schema
class HeadfixedImputation(dj.Computed):
    """Imputed markers for a head-fixed recording. The twin of :class:`DLCImputation`.

    The entire body is :func:`_run_imputation`, the same function the free-moving chain
    calls. Only the three tables it reads differ.
    """

    definition = """
    -> HeadfixedImputationTask
    ---
    imputation_time      : datetime
    recording_fps        : decimal(7,3)   # validated against DISKModel.training_fps
    production_fit_hash  : char(64)       # SHA-256 of the extracted production ellipse fit
    n_frames_total       : int unsigned
    n_frames_observed    : int unsigned   # every marker above threshold
    n_frames_imputed     : int unsigned   # at least one marker filled by DISK
    n_frames_refused     : int unsigned   # at least one marker refused
    n_frames_unfilled    : int unsigned   # at least one marker left for gap length
    n_gaps_attempted     : int unsigned
    n_gaps_too_long      : int unsigned   # exceeded seq_length: the untested regime
    max_gap_filled       : smallint unsigned
    n_angle_before       : int unsigned   # frames yielding an eye angle from the originals
    n_angle_after        : int unsigned   # ... and after imputation. The yield.
    disk_runtime_s=null  : float
    """

    class BodyPart(dj.Part):
        """Per-marker arrays. See :class:`DLCImputation.BodyPart`, which this mirrors."""

        definition = """
        -> master
        -> model.Model.BodyPart
        ---
        x_pos         : blob@external-raw   # effective coordinates: observed or filled
        y_pos         : blob@external-raw
        likelihood    : blob@external-raw   # EFFECTIVE likelihood -- what downstream thresholds on
        sample_source : blob@external-raw   # uint8 per frame, see SAMPLE_SOURCE
        """

    class Gap(dj.Part):
        """One row per contiguous missing segment. See :class:`DLCImputation.Gap`."""

        definition = """
        -> master
        -> model.Model.BodyPart
        gap_start     : int unsigned
        ---
        gap_length    : int unsigned
        uncertainty=null : float          # DISK's per-segment sigma; null where it declined
        gap_status    : enum('filled', 'refused', 'too_long', 'unfilled')
        """

    @property
    def key_source(self):
        """Only recordings whose frames have been classified."""
        return HeadfixedImputationTask & HeadfixedFrameQuality.proj()

    def make(self, key):
        """Impute one head-fixed recording. Body in :func:`_run_imputation`."""
        assert_fit_ellipse_shared()
        _run_imputation(self, key, task_table=HeadfixedImputationTask,
                        quality_table=HeadfixedFrameQuality,
                        fit_params=fetch_headfixed_params)


@pupil_schema
class PupilEllipseFittingHeadfixedImputed(dj.Computed):
    """Eye angles from imputed markers, beside the head-fixed production fit.

    The twin of :class:`PupilEllipseFittingImputed`, and the same decision behind it: a
    sibling table rather than a pose-source dimension on `PupilEllipseFitting`, because
    DataJoint cannot add a primary-key attribute in place.

    Its ``key_source`` requires a production fit at the SAME ``parameter_id``, exactly as
    the free-moving twin does -- the calibration is inherited from that fit, not
    re-estimated, so without it there is nothing to inherit.
    """

    definition = """
    -> HeadfixedImputation
    ---
    theta               : blob@external-raw   # azimuth, radians, relative to eye centre
    phi                 : blob@external-raw   # elevation, radians
    diameter            : blob@external-raw   # ellipse semi-major axis, px
    pupil_center        : blob@external-raw   # (T, 2) in original pixel coordinates
    x0_in               : blob@external-raw
    y0_in               : blob@external-raw
    long_axis           : blob@external-raw
    short_axis          : blob@external-raw
    angle_from_x        : blob@external-raw
    cam_center          : blob@external-raw
    eye_center          : blob@external-raw
    scale               : float
    horizontal_offset   : float
    vertical_offset     : float
    n_nice_frames       : int unsigned
    n_imputed_markers   : blob@external-raw
    n_angle             : int unsigned
    calibration_source  : enum('inherited', 'refit')
    """

    @property
    def key_source(self):
        """Imputations whose head-fixed production twin exists at the same parameter set."""
        pupil_tracking = _linking_module.pupil_tracking
        return HeadfixedImputation & pupil_tracking.PupilEllipseFitting.proj()

    def make(self, key):
        """Fit the imputed markers. Body in :func:`_run_imputed_fit`."""
        assert_fit_ellipse_shared()
        _run_imputed_fit(
            self, key, imputation_table=HeadfixedImputation,
            production_table=_linking_module.pupil_tracking.PupilEllipseFitting,
            fit_params=fetch_headfixed_params)


# ======================================================================================
# Read-only helpers
# ======================================================================================


# ======================================================================================
# The topcam body-tracking chain -- the third sibling
# ======================================================================================
#
# WHY A THIRD CHAIN AND NOT A COLUMN ON THE FIRST
# -----------------------------------------------
# `DLCImputation`'s primary key is
#
#     session_id, scan_id, recording_id, model_name,   <- model.PoseEstimationNew
#     parameter_id,                                    <- PupilEllipseParameterFreeMoving
#     disk_model_name, imputation_paramset_id
#
# and the fifth attribute is the problem. `parameter_id` there is the *pupil* parameter
# set: it says what counts as a missing pupil marker, and it is the key of
# `#pupil_ellipse_parameter_free_moving`. A topcam row would have to supply one, and there
# is no body-camera parameter table to supply it from -- so the row could only exist by
# naming a pupil parameter set that has nothing to do with it. That is not a tight fit
# made loose; it is a foreign key pointing at the wrong thing.
#
# `HeadfixedImputation` already established the precedent and the reason: the two eye
# chains are separate tables rather than one table with a discriminator column because
# their fifth key attribute resolves to different parameter tables, and DataJoint cannot
# express "one or the other" in a primary key. The topcam is the same argument with the
# parameter table removed entirely.
#
# WHAT IS DIFFERENT HERE, BEYOND THE KEY
# --------------------------------------
# Three things, and each one removes something rather than adding it.
#
# 1. **There is no frame-quality gate.** `PupilFrameQuality` classifies frames as
#    `no_reflection` or `tracking_lost`; both are corneal-reflection concepts and neither
#    exists on a body camera. So this chain has no quality table, `key_source` is the task
#    table alone, and `refused` is not a reachable state. The two `refuse_*` flags on
#    `ImputationParamSet` are inert here, which is what set 4's description already says.
#
# 2. **There is no production ellipse fit**, so `production_fit_hash` has nothing to hash.
#    Its job -- make a row visibly stale when the thing it depends on changes -- is done
#    here by `crossover_hash`, the digest of the `ImputationParamSet.Crossover` rows in
#    force when the row was computed. Change the policy and every row that predates the
#    change is visibly, not quietly, out of date.
#
# 3. **The yield counters are consumer quantities, not eye angles.** `n_angle_before` and
#    `n_angle_after` answer "how many frames produce an eye angle". The topcam's equivalent
#    question is asked four times over, because it has four consumers -- and the headroom
#    measurement (`bench_topcam_headroom.py`) framed the whole project in exactly these
#    terms: 15.6% of frames have no ear heading today, 6.1% no body centroid.
#
# THE ONE REAL DESIGN PROBLEM, AND THE CHOICE MADE
# ------------------------------------------------
# The crossover is per *quantity* (20 frames for the centroid and ear heading, 45 for IR
# heading, never for speed), but `BodyPart.x_pos` is per *marker*. One stored marker array
# therefore cannot honour four different crossovers at once. Two ways out:
#
#   (a) Store the array at NO crossover -- DISK wherever it answered -- and let each
#       consumer apply its own crossover at read time. `sample_source` plus `Gap.gap_length`
#       carry everything needed to do that, and `ImputationParamSet.Crossover` states the
#       rule.
#   (b) Store one array per consumer quantity, keyed by `consumer_quantity`, honouring the
#       rule at write time.
#
# **(a) is chosen**, for one reason that outweighs the rest: (b) bakes today's measured
# policy into stored data, so re-measuring the crossover -- which the held-out-animal model
# may well force -- would invalidate every stored array rather than one lookup table. (b)
# also duplicates the heavy blobs up to four times.
#
# The cost of (a) is real and must be named: **the stored array is not safe to read
# directly.** Read naively it is the all-DISK configuration, which is the one measured to
# make centroid speed 43% worse than plain interpolation. That is why
# :meth:`TopcamImputation.markers_for` exists and why it raises rather than returning
# anything when no crossover row is on file. The correct read is one method call; the
# harmful read is the one that takes more work.
#
# NOTHING BELOW IS ACTIVATED
# --------------------------
# These are declarations. `schema.activate(..., create_tables=True)` will create them on
# any server missing them, which is exactly how the `MocapImputation*` tables came back
# twice from stale checkouts. Create them the way `add_crossover_policy.py` did it: name
# the table explicitly, diff the schema's table list before and after, and refuse if the
# difference is anything other than what was asked for.


#: The four quantities the topcam benchmark scored, and the only strings
#: :meth:`TopcamImputation.markers_for` will accept. They are the
#: ``consumer_quantity`` values in :class:`ImputationParamSet.Crossover`.
#:
#: ``centroid_speed`` is in the list because it must be *refusable* by name. Its measured
#: policy is ``use_model = 0``, and a consumer that asks for it should get a clear refusal
#: rather than silently fall through to a default.
TOPCAM_CONSUMER_QUANTITIES = (
    "body_centroid",   # mean of back, neck, tail_base
    "centroid_speed",  # its central difference -- DO NOT use the model
    "heading_ir",      # heading from the IR pair
    "heading_ear",     # heading from the ear pair
)

#: Which markers each consumer quantity is built from. Used to decide which gaps a given
#: quantity's crossover actually governs -- a crossover on ``heading_ear`` has no business
#: filtering the tail base.
TOPCAM_QUANTITY_MARKERS = {
    "body_centroid":  ("back", "neck", "tail_base"),
    "centroid_speed": ("back", "neck", "tail_base"),
    "heading_ir":     ("IR_left", "IR_right"),
    "heading_ear":    ("ear_left", "ear_right"),
}


def topcam_crossover(imputation_paramset_id: int, disk_model_name: str,
                     consumer_quantity: str) -> dict:
    """The measured crossover rule for one model and one quantity. Fails closed.

    Returns a dict with ``crossover_frames`` and ``use_model``. Raises if no row exists,
    because absence of a row means "nobody has measured this" and the honest response to
    that is a refusal, not a default -- see :class:`ImputationParamSet.Crossover`.

    ``disk_model_name`` is required, not optional. A crossover measured on one checkpoint
    says nothing about another, and two checkpoints can share a parameter set -- the
    held-out-animal topcam model shares set 4 with the within-animal one. Defaulting it
    would hand the caller the wrong checkpoint's threshold in exactly the case the
    distinction exists for.

    Read-only: one restriction and one ``fetch1``.
    """
    if consumer_quantity not in TOPCAM_CONSUMER_QUANTITIES:
        raise ValueError(
            "unknown consumer_quantity %r; expected one of %r"
            % (consumer_quantity, TOPCAM_CONSUMER_QUANTITIES)
        )
    rows = ImputationParamSet.Crossover & {
        "imputation_paramset_id": imputation_paramset_id,
        "disk_model_name": disk_model_name,
        "consumer_quantity": consumer_quantity,
    }
    if not rows:
        raise RuntimeError(
            "No ImputationParamSet.Crossover row for paramset %d, model %r, quantity %r. "
            "That means no crossover has been MEASURED for this CHECKPOINT and quantity, "
            "not that there is no restriction -- calling the model without one is the "
            "configuration measured to do harm (centroid speed 43%% worse than linear "
            "interpolation). A crossover measured on a different checkpoint does not "
            "transfer. Measure it with bench_topcam.py before reading this quantity."
            % (imputation_paramset_id, disk_model_name, consumer_quantity)
        )
    row = rows.fetch1()
    return {
        "crossover_frames": int(row["crossover_frames"]),
        "use_model": bool(row["use_model"]),
        "evidence": row["evidence"],
    }


@schema
class TopcamImputationTask(dj.Manual):
    """One topcam imputation to perform: a recording, a model, a policy.

    Three foreign keys where the two eye chains have four. There is no "what counts as
    missing" parameter table for a body camera: the threshold that defines a missing
    marker is ``DISKModel.likelihood_threshold``, which the model row already carries, so
    the model IS the missing-ness definition here.
    """

    definition = """
    -> model.PoseEstimationNew                            # the DLC run being repaired
    -> DISKModel                                          # which trained network
    -> ImputationParamSet                                 # gap ceiling, and the crossovers
    ---
    task_mode='trigger'  : enum('load','trigger')  # 'load': read prepared output; 'trigger': run inference
    output_dir=''        : varchar(512)
    task_description=''  : varchar(255)
    """


@schema
class TopcamImputation(dj.Computed):
    """Imputed topcam body markers, with per-sample provenance and per-gap uncertainty.

    The twin of :class:`DLCImputation` in shape and its opposite in what it is scored on.
    Counters here are the four consumer quantities the benchmark measured, before and
    after, because "how many frames yield a usable body heading" is the question this
    project exists to improve and "how many markers were filled" is not.

    ``crossover_hash`` is the SHA-256 of the whole effective policy when this row was
    computed: the :class:`ImputationParamSet` master attributes AND its
    :class:`~ImputationParamSet.Crossover` rows for this model, ordered by
    ``consumer_quantity``. It takes the job ``production_fit_hash`` does in the eye chain --
    a policy change makes existing rows visibly stale rather than quietly wrong.

    The master is in the digest because it can be edited in place. Changing
    ``max_gap_frames`` alters which gaps were submitted at all, so rows computed before and
    after are not comparable; a hash over the child rows alone would leave them looking
    current.

    **The stored arrays are the un-crossed-over DISK output.** Read them through
    :meth:`markers_for`, never directly -- see the section comment above this class.
    """

    definition = """
    -> TopcamImputationTask
    ---
    imputation_time        : datetime
    recording_fps          : decimal(7,3)   # validated against DISKModel.training_fps
    crossover_hash         : char(64)       # SHA-256 of the policy: master row + Crossover rows
    n_frames_total         : int unsigned
    n_frames_observed      : int unsigned   # every modelled marker above threshold
    n_frames_imputed       : int unsigned   # at least one marker filled by DISK
    n_frames_unfilled      : int unsigned   # at least one marker left: declined or too long
    n_gaps_attempted       : int unsigned
    n_gaps_too_long        : int unsigned   # exceeded seq_length: the untested regime
    n_gaps_declined        : int unsigned   # inside the window, DISK declined anyway (~1 in 6)
    max_gap_filled         : smallint unsigned
    n_centroid_before      : int unsigned   # frames yielding a body centroid from the originals
    n_centroid_after       : int unsigned   # ... and after imputation. The yield.
    n_heading_ir_before    : int unsigned
    n_heading_ir_after     : int unsigned
    n_heading_ear_before   : int unsigned   # the one with the most room: 15.6% missing today
    n_heading_ear_after    : int unsigned
    disk_runtime_s=null    : float
    """

    class BodyPart(dj.Part):
        """Per-marker arrays, un-crossed-over. See :class:`DLCImputation.BodyPart`.

        ``likelihood`` is the EFFECTIVE likelihood, which is what a downstream threshold
        sees. ``sample_source`` distinguishes ``observed``, ``disk``, ``interpolated`` and
        ``unfilled``; ``refused`` and ``dlc_low_kept`` are unreachable on a body camera.
        """

        definition = """
        -> master
        -> model.Model.BodyPart
        ---
        x_pos         : blob@external-raw   # effective coordinates: observed or filled
        y_pos         : blob@external-raw
        likelihood    : blob@external-raw   # EFFECTIVE likelihood -- what downstream thresholds on
        sample_source : blob@external-raw   # uint8 per frame, see SAMPLE_SOURCE
        """

    class Gap(dj.Part):
        """One row per contiguous missing segment, per marker.

        All-scalar and therefore in MySQL rather than on the store, so "every gap longer
        than the crossover whose uncertainty is in the top quintile" is answered in SQL
        without opening a blob. That query is the deployment gate the benchmark
        recommended, and this is the table it runs against.

        ``gap_status`` drops ``refused`` -- unreachable here -- and the three that remain
        mean what they do in :class:`DLCImputation.Gap`. Note that a gap at or below a
        quantity's crossover is NOT a separate status: whether it should have been
        interpolated instead depends on which quantity is asking, and the status is a
        property of the gap, not of the reader. ``gap_length`` plus
        :func:`topcam_crossover` answer that at read time.

        ``uncertainty`` predicts the model's own error at Spearman 0.67 over 5,398
        segments on the topcam cohort, with median error rising 0.90 -> 4.01 px across
        sigma quintiles. Use it to gate the top quintile back to interpolation; do not
        use it to discard data.
        """

        definition = """
        -> master
        -> model.Model.BodyPart
        gap_start        : int unsigned      # first missing frame of the segment
        ---
        gap_length       : int unsigned
        uncertainty=null : float             # DISK's per-segment sigma; null if not filled
        gap_status       : enum('filled','unfilled','too_long')
                                             # 'unfilled': submitted and within the
                                             # window, but DISK declined it
        """

    @property
    def key_source(self):
        """The task table, unrestricted.

        No frame-quality gate stands in front of this chain, because the two classes
        `PupilFrameQuality` exists to catch -- `no_reflection` and `tracking_lost` -- are
        corneal-reflection concepts that a body camera cannot produce.
        """
        return TopcamImputationTask()

    def markers_for(self, consumer_quantity: str):
        """The marker arrays for one quantity, with that quantity's crossover applied.

        THIS IS THE ONLY SAFE READ. The stored arrays are the all-DISK configuration;
        applying the crossover is what makes them better than plain interpolation instead
        of worse. Reading `BodyPart.x_pos` directly gives the configuration measured to
        make centroid speed 43% worse than doing nothing clever at all.

        Raises if no crossover has been measured for ``consumer_quantity``, and raises if
        the measured policy is ``use_model = 0`` -- as it is for ``centroid_speed``. A
        refusal is the finding, not a failure.
        """
        raise NotImplementedError(
            "TopcamImputation.markers_for() is declared but not implemented. It must: "
            "(1) call topcam_crossover() with THIS row's disk_model_name and refuse on "
            "use_model = 0; "
            "(2) fetch BodyPart rows for TOPCAM_QUANTITY_MARKERS[consumer_quantity]; "
            "(3) for every Gap row whose gap_length <= crossover_frames, overwrite that "
            "segment with linear interpolation between the bracketing observed samples "
            "and set sample_source to SAMPLE_SOURCE['interpolated']; "
            "(4) return the arrays plus the sample_source, never the arrays alone."
        )

    def make(self, key):
        """Not implemented. Declaration and review only -- see PR for disk/topcam-table.

        What it must do, in order, and each step has a reason recorded in the benchmark:

        1. Read the recording's markers from ``model.PoseEstimationNew.BodyPartPosition``
           and threshold at ``DISKModel.likelihood_threshold`` to define missing-ness.
        2. Assert the recording's marker set is a superset of ``DISKModel.Keypoint``, and
           mark any marker outside the contract ``not_modelled`` rather than imputing it.
        3. Validate ``recording_fps`` against ``DISKModel.training_fps``. The topcam runs
           at 120 fps and a mismatch silently changes what a gap length means.
        4. Run DISK through ``disk_worker``, gaps longer than ``max_gap_frames`` never
           submitted and counted as ``too_long``.
        5. **Verify the alignment gate before inserting anything.** Every sample DISK was
           not entitled to touch must come back bit-identical. On the benchmark this
           checked 3,810,229 untouched samples at a 1e-6 px tolerance and the worst drift
           was 2.27e-13 px. A marker-column shift produces plausible coordinates, a
           plausible centroid and a plausible heading, so nothing downstream would catch
           it.
        6. Compute the four before/after consumer counters from the un-crossed-over array
           for ``*_before`` and from ``markers_for()`` for ``*_after``.
        7. Hash the Crossover rows into ``crossover_hash`` and insert master, then parts,
           in one transaction.
        """
        raise NotImplementedError(
            "TopcamImputation.make() is declared but not implemented; see the docstring "
            "for the seven steps it owes. Populating this table is a database write and "
            "needs explicit authorisation."
        )


def missing_summary(pose_key: dict, likelihood_threshold: float = 0.5):
    """Missing-data summary for one pose estimation. Read-only; inserts nothing.

    ``pct_missing`` counts each sample once. The previous version of this helper summed
    NaN and sub-threshold counts separately, double-counting any sample that was both and
    allowing the percentage to exceed 100.
    """
    import pandas as pd

    model = _linking_module.model
    rows = (model.PoseEstimationNew.BodyPartPosition & pose_key).fetch(
        as_dict=True, order_by="body_part"
    )
    if not rows:
        return None

    out = []
    for r in rows:
        x = np.asarray(r["x_pos"], float)
        y = np.asarray(r["y_pos"], float)
        lik = np.asarray(r["likelihood"], float)
        nan = np.isnan(x) | np.isnan(y)
        low = lik < likelihood_threshold
        out.append({
            "body_part": r["body_part"],
            "n_frames": x.size,
            "n_nan": int(nan.sum()),
            "n_low_likelihood": int(low.sum()),
            "n_missing": int((nan | low).sum()),
            "pct_missing": float(100.0 * (nan | low).mean()) if x.size else 0.0,
        })
    return pd.DataFrame(out)
