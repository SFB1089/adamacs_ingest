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

# element_interface is imported lazily inside the two classmethods that hash, so that the
# constants and definitions in this module can be read from environments that carry
# DataJoint but not the element_* tree -- the DISK evaluation environment, for one.

schema = dj.Schema()
pupil_schema = dj.Schema()

_linking_module = None
_param_table = None

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
    global _param_table
    _param_table = _linking_module.pupil_tracking.PupilEllipseParameterFreeMoving

    _objects = {**_linking_module.__dict__,
                "PupilEllipseParameterFreeMoving": _param_table}
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
        add_objects={**_objects, "DLCImputation": DLCImputation},
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

        task = (DLCImputationTask & key).fetch1()
        mdl = (DISKModel & key).fetch1()
        pset = (ImputationParamSet & key).fetch1()
        fitp = fetch_params(key)

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
        if set(by_name) != set(contract):
            raise ValueError(
                f"marker set does not match {mdl['disk_model_name']}'s keypoint contract. "
                f"recording has {sorted(set(by_name) - set(contract))} extra, "
                f"{sorted(set(contract) - set(by_name))} missing")
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
        frame_class = (PupilFrameQuality & key).fetch1("frame_class")
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

        gaps = []           # (marker_index, start, length, status)
        submit = missing.copy()
        for j in range(n_kp):
            for start, length in _runs(missing[:, j]):
                sl = slice(start, start + length)
                if refused_frame[sl].any():
                    status = "refused"
                elif length > int(pset["max_gap_frames"]):
                    status = "too_long"
                else:
                    status = "filled"       # provisional; readback decides
                if status != "filled":
                    submit[sl, j] = False
                gaps.append([j, start, length, status])

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

        # Uncertainty is per DISK segment, not per marker-gap, so a gap takes the worst
        # uncertainty of the segments it overlaps -- one badly guessed marker is what
        # damages the ellipse fit.
        for g in gaps:
            j, start, length, status = g
            if status == "filled" and not filled[start:start + length, j].all():
                g[3] = "unfilled"
            overlap = [s["uncertainty"] for s in seg
                       if s["uncertainty"] is not None
                       and s["start"] < start + length and s["start"] + s["length"] > start]
            g.append(max(overlap) if overlap else None)

        before = _yields_angle(lik, contract, fitp)
        after = _yields_angle(lik_eff, contract, fitp)
        filled_lengths = [g[2] for g in gaps if g[3] == "filled"]

        self.insert1({
            **key,
            "imputation_time": datetime.datetime.now(),
            "recording_fps": fps,
            "production_fit_hash": production_fit_hash(),
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
        self.BodyPart.insert([
            {**key, "body_part": m,
             "x_pos": xy_eff[:, i, 0], "y_pos": xy_eff[:, i, 1],
             "likelihood": lik_eff[:, i], "sample_source": src[:, i]}
            for i, m in enumerate(contract)])
        self.Gap.insert([
            {**key, "body_part": contract[j], "gap_start": start,
             "gap_length": length, "uncertainty": unc, "gap_status": status}
            for j, start, length, status, unc in gaps])

        shutil.rmtree(keep, ignore_errors=True)


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
        """Only imputations whose production twin exists, because the calibration is inherited.

        A recording with no :class:`PupilEllipseFittingFreeMoving` row at this parameter
        set has no camera centre or scale to inherit, and re-estimating one is not an
        option -- see the class docstring. Such a recording is therefore not *ready*, in
        exactly the sense that an unclassified recording is not ready for
        :class:`DLCImputation`, so it is skipped rather than failed. Populating the
        production table for it is a separate and deliberate write to production data.

        On this cohort that restriction is severe rather than incidental: every
        parameter-set-4 production fit in the database belongs to ROS-2196.
        """
        pupil_tracking = _linking_module.pupil_tracking
        return DLCImputation & pupil_tracking.PupilEllipseFittingFreeMoving.proj()

    def make(self, key):
        """Refit the ellipse on imputed markers, holding the production calibration fixed.

        The conic fit is the production one, extracted from ``pupil_tracking.py`` at run
        time rather than reimplemented, so this table and its production twin measure the
        same estimator. Two gates protect the comparison; both raise rather than warn,
        because a silent failure here produces smooth, plausible angles that are wrong.
        """
        model = _linking_module.model
        pupil_tracking = _linking_module.pupil_tracking

        params = fetch_params(key)

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
            (DLCImputation.BodyPart & key).fetch(
                "body_part", "x_pos", "y_pos", "likelihood", as_dict=True,
                order_by="body_part"), "DLCImputation.BodyPart")

        pose_key = {k: key[k] for k in model.PoseEstimationNew.primary_key}
        names_o, xy_o, lik_o = _stack(
            (model.PoseEstimationNew.BodyPartPosition & pose_key).fetch(
                "body_part", "x_pos", "y_pos", "likelihood", as_dict=True,
                order_by="body_part"), "PoseEstimationNew.BodyPartPosition")

        pc_x, pc_y, ir_xg, ir_yg = _mask_and_center(xy_i, lik_i, names_i, params)
        pc_x_o, pc_y_o, ir_xg_o, ir_yg_o = _mask_and_center(xy_o, lik_o, names_o, params)

        # -- 2. inherit the calibration -----------------------------------------------
        cam_center_prod, eye_center_prod, scale = (
            pupil_tracking.PupilEllipseFittingFreeMoving & key).fetch1(
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
            (pupil_tracking.PupilEllipseFittingFreeMoving & key).fetch1("theta"), float)
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
        # folding it into the same count would understate it; read it from
        # DLCImputation.BodyPart when that matters.
        src = {r["body_part"]: np.asarray(r["sample_source"]) for r in
               (DLCImputation.BodyPart & key).fetch(
                   "body_part", "sample_source", as_dict=True)}
        synth = np.stack([src[m] == SAMPLE_SOURCE["disk"] for m in PRODUCTION_ORDER[3:]],
                         axis=1)
        used = ~np.isnan(pc_x)                        # (T, 8): entered the conic fit
        n_imputed = (used & synth).sum(axis=1).astype(np.uint8)

        # Un-centred back into original pixel coordinates with the IMPUTED reflection,
        # which is the same convention production uses with its own -- and is part of the
        # point, since a recovered IR is one of the ways a frame is won back.
        self.insert1({
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

# ======================================================================================
# Read-only helpers
# ======================================================================================


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
