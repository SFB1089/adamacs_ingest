"""Cell-registration schema: versioned cross-session cell identity with match quality.

Longitudinal imaging rests on the claim that *this cell on day 20 is that cell on day 0*.
Today that claim is carried by ``roselab_imaging.Segmentation.Mask.roi_group`` -- a
``smallint`` that is **not part of the primary key** and is overwritten in place by whichever
CellReg ingest ran last. Nothing records which run produced it, with what parameters, how
good the matches were, who checked them, or what the column held before. Wiki issue D1 is
the consequence: counts recorded by an analysis (111, 146) disagree with the live database
(116, 120), and T1 notes the delta can no longer be reconstructed.

This schema makes cross-session identity a first-class, versioned, queryable entity. The
design review beside this file (``CELL_REGISTRATION_SCHEMA_DESIGN_REVIEW.md``) carries the
evidence, the rejected alternatives and the open questions; read it before changing a key.

Design decisions that a reader must know
----------------------------------------
* **Strictly additive.** ``roi_group`` is not written, altered or deprecated here. The
  importer deliberately does *not* update it (design review ADR-R3), because that mutation
  is what produced D1. :class:`RoiGroupSnapshot` preserves its state instead.
* **Per-session local ``mask`` ids are the durable anchor.** A CellReg re-ingest rewrites
  ``roi_group`` but never the mask numbering, so ``(session_id, scan_id, paramset_idx,
  curation_id, mask)`` is stable and identity can be rebuilt on top of it.
* **Runs are content-addressed.** 100 distinct sha256 across 108 files on this server: the
  same run is reachable by several paths because scan folders hold byte-identical backups.
  Keyed on path, one registration would enter as up to three runs.
* **Quality is rows with columns, never blob arrays.** A value inside a ``longblob`` cannot
  be restricted, joined or selected on, which is precisely what cell-level QC requires.
* **Thresholds live downstream.** This schema stores continuous match quality only.
  Eligibility is a QC and cohort-selection act with its own version history.
* **The matching method is a lookup**, so a second method is a row rather than a migration.

Nothing is declared on import
-----------------------------
Deferred activation, as in ``disk.py`` and the ``element_*`` schemas. Importing this module
creates no tables; someone must call :func:`activate` explicitly. ``disk.py`` records why
this matters: its predecessor declared at import time, so the tables appeared the first time
anyone imported the pipeline, with no migration and no decision.

Parsing notes for ``CellRegResult.make()``
------------------------------------------
* CellReg writes 1-based mask indices and uses ``0`` for *not present*. The conversion is
  ``map - 1``, which sends absent to ``-1``; absence is then stored as **no Member row**,
  never as a sentinel. (Leon Kremers, 2024-05-24, ts=1716542082.875469.)
* ``mat73`` is the lab's sanctioned loader for these files; v7.3/HDF5 is 107 of 108 files on
  this server, the remaining one is an older MAT that ``scipy.io`` must handle.
* The score fields exist only when ``registration_approach = 'Probabilistic'`` and
  ``alignment_rotations`` only under ``'Translations and Rotations'``. Absent fields are
  NULL with a stated reason, never zero.
* **Parameters and run accuracy are not in the .mat at all** -- they live in the sibling
  ``logFile_*.txt``, whose timestamp trails the result file (76 s in a verified pair).
"""
from __future__ import annotations

import datetime as _dt
import pathlib as _pathlib
import hashlib
import importlib
import json

import datajoint as dj
import numpy as _np

schema = dj.Schema()

# Values CellReg can write for these two settings, normalised to lowercase snake case.
ALIGNMENT_MODES = ("translations", "translations_and_rotations", "non_rigid", "unknown")
REGISTRATION_APPROACHES = ("probabilistic", "simple_threshold", "unknown")

#: Bumped whenever ``CellRegResult.make`` changes how it reads a file, so that rows can be
#: told apart by the code that produced them rather than by their ingest date.
PARSER_VERSION = "1.0.0"

#: Sentinel parameter set for runs whose ``logFile_*.txt`` / ``settings.mat`` is missing.
#: A run with no recoverable parameters gets this rather than a plausible default.
UNKNOWN_PARAMSET = "unknown"


#: Set by :func:`activate`. ``schema.activate(add_objects=...)`` puts names into the
#: *declaration* context only (datajoint 0.14.3 schemas.py:152-153) -- it does NOT bind
#: them in this module's namespace -- so a bare ``imaging.Curation`` inside a method body
#: is a NameError. Method bodies go through :func:`_imaging` instead.
_linking_module = None


def activate(schema_name: str, *, create_schema: bool = True, create_tables: bool = True,
             linking_module: str = "adamacs.pipeline"):
    """Activate the cell-registration schema.

    Nothing is declared until this is called. ``linking_module`` supplies ``imaging`` (for
    the nullable ``Curation`` dependency) the same way the ``element_*`` schemas receive
    their dependencies from ``adamacs.pipeline``.
    """
    global _linking_module
    module = importlib.import_module(linking_module)
    _linking_module = module
    schema.activate(schema_name, create_schema=create_schema, create_tables=create_tables,
                    add_objects={"imaging": module.imaging, **globals()})
    return schema


def _imaging():
    """The upstream imaging module, for use inside method bodies.

    See the note on :data:`_linking_module`: ``add_objects`` does not make ``imaging`` a
    module-level name, so every runtime reference resolves through here and fails with a
    sentence that says what to do rather than with a bare NameError.
    """
    if _linking_module is None:
        raise dj.DataJointError(
            "cell_registration.activate() has not been called, so the upstream imaging "
            "module is unknown. Call activate(schema_name, linking_module=...) first.")
    return _linking_module.imaging


def canonical_json(params: dict) -> str:
    """Stable JSON for hashing: sorted keys, no insignificant whitespace."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(params: dict) -> str:
    """sha1 of the canonical JSON -- the identity of a parameter set's *content*."""
    return hashlib.sha1(canonical_json(params).encode("utf-8")).hexdigest()


class _HashedConfig:
    """Mixin making a parameter table immutable by content.

    Re-registering a slug with different parameters raises rather than updating, so a
    changed parameter is a new slug and every downstream key names the parameters that
    produced it. Same pattern as ``roselab_leondrift`` (its ADR-005).
    """

    @classmethod
    def register(cls, slug: str, params: dict, label: str = "", **extra) -> str:
        digest = config_hash(params)
        existing = (cls & {cls.primary_key[0]: slug}).fetch(as_dict=True)
        if existing:
            if existing[0]["params_hash"] != digest:
                raise ValueError(
                    f"{cls.__name__} slug {slug!r} already exists with different parameters "
                    f"({existing[0]['params_hash'][:8]} != {digest[:8]}); use a new slug")
            return slug
        cls.insert1({cls.primary_key[0]: slug, "params": params, "params_hash": digest,
                     "params_json": canonical_json(params), "label": label,
                     "created": _dt.datetime.now(), **extra})
        return slug

    @classmethod
    def params_of(cls, slug: str) -> dict:
        return (cls & {cls.primary_key[0]: slug}).fetch1("params")


# --------------------------------------------------------------------------- reference
@schema
class MatchingMethod(dj.Lookup):
    definition = """
    # How cells were matched across sessions. A second method is a row, not a migration.
    matching_method     : varchar(32)
    ---
    software            : varchar(64)      # e.g. 'CellReg'
    software_reference  : varchar(255)     # DOI or repository URL
    method_description  : varchar(512)
    """

    contents = [
        ("cellreg_probabilistic", "CellReg",
         "https://doi.org/10.1016/j.celrep.2017.10.013",
         "CellReg probabilistic modelling: p_same from a centroid-distance or "
         "spatial-correlation model, clustered at a p_same threshold."),
        ("cellreg_simple_threshold", "CellReg",
         "https://doi.org/10.1016/j.celrep.2017.10.013",
         "CellReg simple threshold: a fixed correlation or distance cut, no probability "
         "model. Produces no cell scores and no p_same."),
        ("cellreg_unknown", "CellReg",
         "https://doi.org/10.1016/j.celrep.2017.10.013",
         "CellReg, method not recoverable: the run has no logFile_*.txt, so the "
         "registration approach was never recorded. NOT a claim that a simple "
         "threshold was used -- routing unknown runs to the simple-threshold row "
         "would give method-based cohort queries false provenance."),
    ]


@schema
class CellRegParamSet(dj.Lookup, _HashedConfig):
    definition = """
    # Immutable, content-hashed registration parameters. Register via .register().
    cellreg_paramset_id : varchar(40)      # human-readable slug
    ---
    -> MatchingMethod
    params              : longblob         # the parameter dict as recovered
    params_hash         : char(40)         # sha1 of canonical JSON
    params_json         : varchar(4000)
    label               : varchar(128)
    created             : datetime
    """


# --------------------------------------------------------------------------- the run
@schema
class CellRegRun(dj.Manual):
    definition = """
    # One CellReg execution, identified by the content of its result file.
    cellreg_run_id          : char(40)     # first 40 hex chars of result_file_sha256
    ---
    -> CellRegParamSet
    subject                 : varchar(16)
    result_file             : varchar(512) # canonical path; copies live in .Copy
    result_file_sha256      : char(64)
    result_file_bytes       : bigint
    result_file_mtime       : datetime
    log_file=''             : varchar(512) # '' when no logFile_*.txt was found
    log_file_sha256=''      : char(64)
    mat_format              : enum('v7.3_hdf5','v5_v7')
    registration_approach   : enum('probabilistic','simple_threshold','unknown')
    alignment_mode          : enum('translations','translations_and_rotations','non_rigid','unknown')
    reference_session_index=null : smallint  # 1-based, as CellReg reports it
    n_sessions              : smallint
    n_registered_cells      : int
    ingested_at             : datetime
    ingest_host             : varchar(64)
    adamacs_sha=''          : varchar(40)
    run_note=''             : varchar(512)
    """

    class Copy(dj.Part):
        definition = """
        # Other paths holding a byte-identical copy of this run's result file.
        -> master
        copy_path           : varchar(512)
        ---
        copy_mtime          : datetime
        """

    class Quality(dj.Part):
        definition = """
        # Run-level accuracy. Present ONLY in logFile_*.txt, in no field of the .mat.
        -> master
        ---
        frac_false_positives=null    : float
        frac_false_negatives=null    : float
        frac_uncertain_pairs=null    : float
        avg_register_score=null      : float
        avg_true_positive_score=null : float
        avg_true_negative_score=null : float
        avg_exclusivity_score=null   : float
        model_discrepancy=null       : float
        avg_fraction_active=null     : float
        initial_threshold=null       : float
        final_threshold=null         : float
        maximal_distance_um=null     : float
        n_bins=null                  : smallint
        microns_per_pixel=null       : float
        # Frame geometry, recorded verbatim rather than derived. The log prints
        # "Image size - 513X513 [pixels]" and "FOV size - 615.6X615.6 [microns]" for a
        # 512-pixel scan, because the driver pads to ops.Lx+1 (cellreg_run.m:130-131) and
        # computes the FOV from the padded size. So microns_per_pixel is 615.6/513 = 1.2,
        # NOT 615.6/512 = 1.2024 -- and that factor scales every centroid distance through
        # normalized_maximal_distance. Storing both ends the temptation to recompute it.
        image_width_px=null          : smallint
        image_height_px=null         : smallint
        fov_width_um=null            : float
        fov_height_um=null           : float
        quality_source               : enum('log_file','absent')
        """

    class Session(dj.Part):
        definition = """
        # One RESOLVED column: this curation participated in this run, as this column.
        # The curation is in the primary key, so a row carries session_id, scan_id,
        # paramset_idx and curation_id and joins straight to Segmentation.Mask,
        # Fluorescence and Activity without a projection. Same shape as
        # roselab_leondrift.RunSet.Scan, which also keys on the upstream identity.
        # Columns that could NOT be resolved live in .UnresolvedColumn, so that this
        # table's key never has to be nullable.
        #
        # WHICH CURATION A COLUMN CAME FROM IS NOT RECOVERABLE FROM THE RUN.
        # The export step copies <ScanPath>/suite2p/plane0/Fall.mat -- a per-scan
        # FILESYSTEM path with no curation in the lookup -- so the curation was never an
        # input, and the log records only the scan id (converted_Fall_01_<scan_id>.mat).
        # Nor can it be recovered from the data: of 743 (session, scan, paramset) groups
        # holding more than one curation, 696 (94%) have BYTE-IDENTICAL mask sets, so
        # mask-set overlap scores 1.0 against every candidate and cannot discriminate.
        # Jaccard therefore resolves WHICH SCAN, never WHICH CURATION.
        # 'mask_set_identical' is the narrow tie-break Tobias authorised on
        # 2026-10-20: several curations matched the file's per-column centroid
        # count AND were byte-identical in mask set, so the choice cannot change
        # which masks, centroids or traces the column resolves to -- only which
        # curation_id is recorded. The pick is the latest such curation predating
        # the run, since one created afterwards cannot have been its input. Where
        # the candidates genuinely DIFFER the column stays unresolved: that would
        # be choosing between different answers. 537 columns; 25 left unresolved.
        # 'roi_group_match' identifies the curation BY CONTENT: the roi_group a column
        # implies matched one candidate curation's stored roi_group on every mask, and
        # no other candidate. That is evidence, not similarity -- but it only works
        # while the run's output is still the one in the column, so it recovers 307 of
        # the 1,266 historical columns and nothing for runs already overwritten.
        # 'manifest' is the only method that makes the curation a recorded fact: the
        # export step writes (column_index, session_id, scan_id, paramset_idx,
        # curation_id) beside the converted footprints. Absent a manifest the operator
        # declares it ('manual'); the importer must not guess.
        -> master
        -> imaging.Curation
        ---
        column_index                : smallint   # 0-based column of cell_to_index_map
        source_file=''              : varchar(512)  # the converted_*.mat named in the log
        resolution_method           : enum('manifest','log_file','mask_jaccard','roi_group_match','mask_set_identical','manual')
                                               # manifest = declared by the exporter
        mask_set_jaccard=null       : float      # stored ALWAYS: also a staleness measure
        n_masks_cellreg             : int
        n_masks_db=null             : int
        alignment_x_translation=null : float
        alignment_y_translation=null : float
        alignment_rotation=null     : float      # NULL unless translations_and_rotations
        zero_padding_x=null         : float
        zero_padding_y=null         : float
        unique index (cellreg_run_id, column_index)
        """

        def aligned_footprints(self, *, masks=None, dense=False):
            """This column's spatial footprints, aligned, generated on demand.

            CellReg's ``spatial_footprints_corrected`` is the one output this schema does
            not store: 190 MB in the eleven-session ROS-1881 result file, ~2.9 GB in
            memory as doubles, and redundant -- it is ``imaging.Segmentation.Mask`` put
            through a transform whose every parameter lives in this very row
            (``alignment_x_translation``, ``alignment_y_translation``,
            ``alignment_rotation``, ``zero_padding_x``, ``zero_padding_y``). This method
            is that transform, ported function-for-function from the pinned toolbox; the
            conventions it has to respect -- translations stored in **microns**, a
            **backward** map, and a 25 um radius crop that *deletes* distant pixels -- are
            documented in :mod:`adamacs.ingest.cellreg_footprints`.

            Needs only this row and the column's masks -- no result file, and no
            ``MaskCoordinates``: the centroid the crop is measured from is the *unaligned*
            footprint's, and that is the database mask, so it is recomputed rather than
            read. Verified against the stored footprints of the January 2026 ROS-1881 run: all
            3,494 footprints of three columns -- the reference, a translating and a
            rotating one -- reproduce at correlation 1.000000 with a maximum pixel
            difference of 2.4e-15. See ``scripts/verify_footprints.py``.

            Parameters
            ----------
            masks
                Restrict to these ``mask`` ids; ``None`` (default) does the whole column.
                Pass a list when you want three cells, not thirteen hundred.
            dense
                ``False`` (default) returns ``(ypix, xpix, weights)`` per mask -- an
                aligned footprint is ~0.1 % non-zero, and returning dense 513x513 frames
                by default is how the 2.9 GB array gets rebuilt one cell at a time.
                ``True`` returns the full frame, for plotting.

            Returns
            -------
            dict
                ``{mask: (ypix, xpix, weights)}`` or ``{mask: ndarray}``.

            Raises
            ------
            DataJointError
                If ``CellRegRun.Quality.microns_per_pixel`` is unknown. The stored
                translations are in microns and the crop radius is in microns, so without
                that factor there is no transform -- and a plausible default is exactly the
                wrong thing to supply (1.2 against 1.2024 shifts every centroid distance).
            """
            return importlib.import_module(
                "adamacs.ingest.cellreg").aligned_footprints_for_column(
                    self, masks=masks, dense=dense)

    class UnresolvedColumn(dj.Part):
        definition = """
        # A column of cell_to_index_map that could not be tied to exactly one curation.
        # Kept as evidence rather than guessed: the lab's realign_global_ids helper
        # majority-votes in this situation and returns -1, which is how wrong joins happen.
        -> master
        column_index            : smallint
        ---
        n_masks_cellreg         : int
        best_jaccard=null       : float
        ambiguous               : bool       # >1 candidate scan cleared the threshold
        candidates=null         : longblob   # [(scan key, jaccard), ...] as evidence
        reason                  : varchar(255)
        """


# --------------------------------------------------------------------------- the result
@schema
class CellRegResult(dj.Imported):
    definition = """
    # Parsed contents of one run's result file. All .mat reading happens in make().
    -> CellRegRun
    ---
    parse_ok                : bool
    parse_error=''          : varchar(512)
    parser_version          : varchar(16)
    source_sha256           : char(64)     # re-hashed at parse time; must match the run
    n_cells                 : int
    n_sessions              : smallint
    has_score_fields        : bool
    missing_fields=''       : varchar(255)
    p_same_consistent       : bool         # p_same matrix side == n_sessions
    # Pair-quality completeness, recorded rather than left to be recomputed. CellReg
    # emits NaN for some cell-pairs the index map calls the same cell, so PairQuality is
    # legitimately shorter than the index map implies and the difference must not look
    # like a parser bug: on the 11-session ROS-1881 run, 38,921 implied pairs = 38,767
    # stored + 154 non-finite.
    n_pairs_expected=null   : int          # sum over cells of C(sessions present, 2)
    n_pairs_nonfinite=null  : int          # of those, p_same was NaN and was NOT stored
    footprints_storage      : enum('in_file','external_path','absent')
    footprints_path=''      : varchar(512)
    parsed_at               : datetime
    compute_seconds         : float
    """

    class Cell(dj.Part):
        definition = """
        # One registered cell. Scores are NULL under the simple-threshold approach.
        -> master
        cell_index              : int          # 0-based row of cell_to_index_map
        ---
        cell_score=null             : float
        true_positive_score=null    : float
        true_negative_score=null    : float
        exclusivity_score=null      : float
        in_overlapping_fov=null     : bool
        centroid_x_registered=null  : float
        centroid_y_registered=null  : float
        n_sessions_present          : smallint
        min_p_same=null             : float    # weakest present pair -- the selector column
        median_p_same=null          : float
        """

        def aligned_footprints(self):
            """One registered cell's aligned footprint in every session it appears in.

            The query a longitudinal figure actually wants: give it one
            ``CellRegResult.Cell`` key and it returns that cell's footprint, in the common
            aligned frame, for each session where CellReg found it -- so the panels of a
            tracked cell are directly comparable without any further registration step.

            Returns
            -------
            dict
                ``{column_index: {"key": <curation key>, "mask": int,
                "footprint": (ypix, xpix, weights)}}``, ordered by column.
            """
            return importlib.import_module(
                "adamacs.ingest.cellreg").aligned_footprints_for_cell(self)

    class Member(dj.Part):
        definition = """
        # The join bridge: this cell is this mask in this scan. Absent = no row.
        # Inheriting CellRegRun.Session's key means a Member row carries session_id,
        # scan_id, paramset_idx and curation_id, so
        #   (CellRegResult.Member * imaging.Segmentation.Mask)
        # joins on the full curation key plus mask with no renaming. This is the table
        # that replaces roi_group as a join key.
        -> CellRegResult.Cell
        -> CellRegRun.Session
        ---
        mask                    : smallint     # local mask id, 0-based after conversion
        """

    class PairQuality(dj.Part):
        definition = """
        # p_same for one cell between two sessions it is present in. ~10.7 M rows lab-wide.
        # Keyed on the two COLUMN indices rather than two curation keys: p_same_registered_
        # pairs is indexed by column, and two renamed 5-attribute foreign keys would put ten
        # key columns on a ten-million-row table to say what two smallints already say.
        # Resolve a column to its curation through CellRegRun.Session.column_index.
        -> CellRegResult.Cell
        column_a                : smallint
        column_b                : smallint     # column_a < column_b
        ---
        p_same                  : float
        """

    def make(self, key):
        importlib.import_module("adamacs.ingest.cellreg").make_cellreg_result(self, key)


@schema
class MaskCoordinates(dj.Imported):
    definition = """
    # Mask centroids in the scan's own frame and after this run's alignment.
    # Keyed through CellRegRun.Session, so the curation key is present and the native
    # centroid can be checked against Segmentation.Mask.mask_center_x/y by direct join.
    -> CellRegRun.Session
    mask                        : smallint
    ---
    centroid_x_native           : float    # copy of Segmentation.Mask.mask_center_x
    centroid_y_native           : float    # copied AS OF this run, so re-segmentation shows
    centroid_x_registered=null  : float
    centroid_y_registered=null  : float
    """

    def make(self, key):
        importlib.import_module("adamacs.ingest.cellreg").make_mask_coordinates(self, key)


# --------------------------------------------------------------------- stable identity
@schema
class RegistrationSite(dj.Manual):
    definition = """
    # A field of view followed over days. same_site_id is the site's FIRST session id
    # ('sess9FOTVMAL'), matching the cellreg folder convention, upstream
    # roselab_session.session_same_site, and roselab_leondrift.RunSet.Site.
    #
    # varchar(12) deliberately mirrors roselab_session.session_same_site.same_site_id
    # exactly, so that if the upstream grouping is ever repaired this table can take a
    # real foreign key to it without a type migration. (leondrift chose varchar(16); the
    # live values are 12 characters.)
    #
    # Why a local table rather than an FK today: upstream is populated for all 3,460
    # sessions but DEFAULTS TO SELF-REFERENCE -- 2,561 of them have
    # same_site_id = session_id -- so "never assigned to a site" and "a site of exactly
    # one session" are indistinguishable there. 41 sites do carry genuine multi-session
    # groupings (up to 66 sessions). site_id_source records which case a row came from.
    subject                 : varchar(16)
    same_site_id            : varchar(12)
    ---
    site_description=''     : varchar(255)
    site_id_source          : enum('cellreg_folder','manual','same_site_id')
    created                 : datetime
    """


@schema
class StableCell(dj.Manual):
    definition = """
    # An identity that outlives any single registration run.
    -> RegistrationSite
    stable_cell_id          : int
    ---
    minted_at               : datetime
    minted_from_run         : char(40)     # the run that first produced this cell
    """


@schema
class StableCellMap(dj.Manual):
    definition = """
    # Which cell_index of which run this stable identity corresponds to.
    # Replaces recomputing realign_global_ids_to_current_roi_group() in every analysis.
    -> StableCell
    -> CellRegResult.Cell
    ---
    mapping_method          : enum('mask_set_exact','mask_set_majority','manual')
    support_n_masks         : smallint     # masks agreeing between the two runs
    conflict=0              : bool
    mapped_at               : datetime
    """


# --------------------------------------------------------------------------- curation
@schema
class CellRegTask(dj.Manual):
    definition = """
    # One PLANNED CellReg run over an explicit, ORDERED set of curations.
    #
    # This table is the fix for design review 3.3b. The existing export copies
    # <ScanPath>/suite2p/plane0/Fall.mat -- a per-scan FILESYSTEM path with no curation in
    # the lookup -- so the curation a column came from was never an input and cannot be
    # recovered afterwards (696 of 743 multi-curation groups have byte-identical mask sets,
    # so mask overlap cannot discriminate). Here the operator DECLARES the curation for each
    # column before anything runs, and .Session is that manifest.
    #
    # task_mode follows the house convention (imaging.ProcessingTask, wfield.py:372,
    # disk.py:986, denoising.py:95):
    #   'trigger' -> build the CellReg input files from the database for the declared
    #                curations, then run the MATLAB driver over the directory by hand
    #   'load'    -> a run already exists in cellreg_output_dir; record what is there
    -> RegistrationSite
    cellreg_task_id         : smallint
    ---
    -> CellRegParamSet
    task_mode='load'        : enum('load','trigger')  # 'load': record existing input; 'trigger': write it
    cellreg_output_dir      : varchar(512)   # FLAT folder the driver consumes; it creates Results/ inside
    task_label=''           : varchar(64)    # 'Gratings', 'Natural Images' -- one run per stimulus per site
    task_created            : datetime
    task_created_by=''      : varchar(64)
    task_note=''            : varchar(512)
    """

    class Session(dj.Part):
        definition = """
        # THE MANIFEST. One row per column, naming the curation that column is built from.
        # column_index is in the primary key because the ORDER is what the operator declares
        # and what the driver consumes; the curation is the declared value. The unique index
        # stops one curation being used for two columns of the same task.
        -> master
        column_index            : smallint   # 0-based; becomes CellRegRun.Session.column_index
        ---
        -> imaging.Curation
        unique index (subject, same_site_id, cellreg_task_id, session_id, scan_id, paramset_idx, curation_id)
        """

    @classmethod
    def auto_generate_entries(cls, site_key, curations, *, paramset, task_mode="trigger",
                              output_dir=None, label="", note="", created_by=""):
        """Plan a run from an ORDERED list of ``imaging.Curation`` keys.

        The order of ``curations`` is the column order of the run; it is recorded, never
        inferred. Nothing is written to disk here -- this only plans the task. Mirrors
        ``ProcessingTask.auto_generate_entries`` and
        ``PoseEstimationTask.insert_estimation_task``.
        """
        from adamacs.ingest.cellreg_export import export_dir_for

        if len(curations) < 2:
            raise ValueError(
                f"a CellReg run needs at least 2 sessions; got {len(curations)}. "
                f"Registering one session against itself is not a registration.")
        seen = [tuple(sorted(c.items())) for c in curations]
        if len(set(seen)) != len(seen):
            raise ValueError("the same curation appears twice in the declared column order")
        imaging = _imaging()
        missing = [c for c in curations if not (imaging.Curation & c)]
        if missing:
            raise ValueError(f"these curations do not exist: {missing}")

        existing = (cls & site_key).fetch("cellreg_task_id")
        task_id = (max(existing) + 1) if len(existing) else 0
        out = output_dir or export_dir_for(site_key, label=label, task_id=task_id)
        cls.insert1({**site_key, "cellreg_task_id": task_id,
                     "cellreg_paramset_id": paramset, "task_mode": task_mode,
                     "cellreg_output_dir": str(out), "task_label": label,
                     "task_created": _dt.datetime.now(), "task_created_by": created_by,
                     "task_note": note})
        cls.Session.insert({**site_key, "cellreg_task_id": task_id, "column_index": i, **c}
                           for i, c in enumerate(curations))
        return {**site_key, "cellreg_task_id": task_id}


@schema
class CellRegExport(dj.Imported):
    definition = """
    # What a 'trigger' task WROTE, or what a 'load' task FOUND. One row per task.
    #
    # Files are built from imaging.Segmentation.Mask for the DECLARED curation, so every
    # column's provenance is a recorded fact rather than an inference. Verified against the
    # existing on-disk Fall.mat for all 11 gratings scans of site sess9FPXNI8U (13,532
    # ROIs): identical pixel sets and identical lam values throughout.
    -> CellRegTask
    ---
    export_mode             : enum('load','trigger')
    export_dir              : varchar(512)
    settings_file=''        : varchar(512)   # settings.mat, written from the paramset
    settings_sha256=''      : char(64)
    n_files                 : smallint
    n_masks_total           : int
    exported_at             : datetime
    exported_by=''          : varchar(64)
    exporter_version        : varchar(16)
    adamacs_sha=''          : varchar(40)
    export_note=''          : varchar(512)
    """

    class File(dj.Part):
        definition = """
        # One input file per column, with the curation it was built from and its checksum.
        -> master
        column_index            : smallint
        ---
        -> imaging.Curation
        file_path               : varchar(512)
        file_name               : varchar(255)   # Fall_<NN>_<scan_id>.mat -- the name the log echoes
        file_sha256             : char(64)
        file_bytes              : bigint
        n_masks                 : int
        fov_height              : smallint
        fov_width               : smallint
        file_source='database'  : enum('database','suite2p_copy')
        """

    def make(self, key):
        """Write the declared input files ('trigger'), or record an existing set ('load').

        WRITES FILES in 'trigger' mode: one ``Fall_<NN>_<scan_id>.mat`` per declared column
        plus ``settings.mat``, into a NEW directory. It never writes inside a suite2p tree
        and never overwrites an existing file -- ``assert_safe_target`` enforces both, so a
        curated suite2p output cannot be damaged by a mistaken task.
        """
        from adamacs.ingest import cellreg_export as cx

        task = (CellRegTask & key).fetch1()
        mode = task["task_mode"]
        out = _pathlib.Path(task["cellreg_output_dir"])
        columns = (CellRegTask.Session & key).fetch(order_by="column_index", as_dict=True)
        if not columns:
            raise ValueError(f"CellRegTask {key} declares no columns; nothing to export")
        if [int(c["column_index"]) for c in columns] != list(range(len(columns))):
            raise ValueError(
                f"column_index must be contiguous from 0; got "
                f"{[int(c['column_index']) for c in columns]}. A gap would silently shift "
                f"every later column against cell_to_index_map.")

        common = {"session_id", "scan_id", "paramset_idx", "curation_id"}
        if mode == "trigger":
            files, note = [], ""
            for col in columns:
                ckey = {k: col[k] for k in common}
                masks = (_imaging().Segmentation.Mask & ckey).fetch(
                    "mask", "mask_xpix", "mask_ypix", "mask_weights",
                    order_by="mask", as_dict=True)
                summary = (_imaging().MotionCorrection.Summary & ckey).fetch1(
                    "average_image")
                height, width = _np.asarray(summary).shape[:2]
                dest = out / cx.fall_name(int(col["column_index"]), col["scan_id"])
                written = cx.write_fall_mat(dest, masks, height, width, mean_img=summary)
                files.append({"column_index": int(col["column_index"]), **ckey, **written})
            params = (CellRegParamSet & task).fetch1("params")
            settings = cx.write_settings_mat(out / "settings.mat", params)
            record = {"files": files, "note": note, **settings}

        elif mode == "load":
            record = cx.inspect_existing(out, columns)

        else:
            raise ValueError(f"Unknown task mode: {mode}")

        self.insert1({**key, "export_mode": mode, "export_dir": str(out),
                      "settings_file": record["settings_file"],
                      "settings_sha256": record["settings_sha256"],
                      "n_files": len(record["files"]),
                      "n_masks_total": sum(f["n_masks"] for f in record["files"]),
                      "exported_at": _dt.datetime.now(),
                      "exported_by": cx.current_user(),
                      "exporter_version": cx.EXPORTER_VERSION,
                      "adamacs_sha": cx.adamacs_sha(),
                      "export_note": record.get("note", "")})
        self.File.insert({**key, **f} for f in record["files"])


@schema
class RegistrationCuration(dj.Manual):
    definition = """
    # A curation pass over one run. Append-only: disagreeing means a new curation_id.
    -> CellRegRun
    curation_id             : int
    ---
    curator                 : varchar(64)
    curated_at              : datetime
    curation_note=''        : varchar(512)
    """

    class CellDecision(dj.Part):
        definition = """
        # One judgement about one registered cell. Never updated; superseded by a new pass.
        -> master
        -> CellRegResult.Cell
        ---
        decision            : enum('accept','reject','split','merge','uncertain')
        reason=''           : varchar(255)
        """


# ------------------------------------------------------- roi_group as a derived cache
@schema
class RoiGroupProjection(dj.Manual):
    """What ``roi_group`` currently reflects, and where it came from.

    Design decision ADR-R6 (Tobias, 2026-10-15): ``roi_group`` is not an independent fact
    but a **derived projection** -- the cell identity of the latest curated run for a site,
    materialised onto ``imaging.Segmentation.Mask`` because that is the query path the lab
    already uses. Overwriting a cache whose source is recorded is not data loss, which is
    what made the original D1 overwrite harmful.

    What is projected is ``cell_index``, not ``stable_cell_id``: upstream ``roi_group`` is
    ``smallint`` (max 32,767) and the largest single run already holds 25,724 registered
    cells, so an identifier that accumulates across re-registrations would eventually
    overflow a production column. Widening it to ``int`` is the follow-on migration, not
    part of this pass.

    This table only RECORDS a projection. Performing one writes to a production table and
    requires explicit authorisation in that conversation turn; see the dry-run helper in
    ``adamacs.ingest.cellreg``.
    """

    definition = """
    # One projection of cell identity onto imaging.Segmentation.Mask.roi_group.
    projection_id           : char(40)     # sha1 over the projected (mask -> value) rows
    ---
    -> CellRegRun
    registration_curation_id=null : int    # the RegistrationCuration pass, if curated
    projected_value         : enum('cell_index','stable_cell_id')
    unmatched_value         : smallint     # convention: -1
    projected_at            : datetime
    projected_by            : varchar(64)
    n_masks_written         : int
    n_unmatched             : int
    pre_snapshot_id=''      : char(40)     # RoiGroupSnapshot taken immediately before
    dry_run                 : bool         # 1 = computed and compared, nothing written
    note=''                 : varchar(512)
    """

    class Scan(dj.Part):
        definition = """
        # Per-scan outcome of the projection.
        -> master
        -> imaging.Curation
        ---
        n_masks             : int
        n_unmatched         : int
        n_changed=null      : int          # vs the roi_group present before the projection
        """


# ------------------------------------------------------------------ legacy preservation
@schema
class RoiGroupSnapshot(dj.Manual):
    definition = """
    # What Segmentation.Mask.roi_group held at one moment. Reads production, writes here.
    snapshot_id             : char(40)     # sha1 over the part rows
    ---
    taken_at                : datetime
    taken_by                : varchar(64)
    note=''                 : varchar(512)
    n_rows                  : int
    fingerprint             : char(40)
    """

    class Mask(dj.Part):
        definition = """
        -> master
        session_id          : varchar(12)
        scan_id             : varchar(12)
        paramset_idx        : smallint
        curation_id         : int
        mask                : smallint
        ---
        roi_group           : smallint
        """

# --------------------------------------------------------------------- deletion
def runs_touching(curation_key) -> "dj.expression.QueryExpression":
    """Every :class:`CellRegRun` that used this curation as one of its columns.

    A plain semijoin: ``cellreg_run_id`` is the only attribute the two share, so the
    restriction matches on it alone. Projecting first would drag the curation attributes
    into the restriction, which ``CellRegRun`` does not have.
    """
    return CellRegRun & (CellRegRun.Session & curation_key)


def delete_runs_for(curation_key, *, dry_run: bool = True,
                    allow_curated: bool = False) -> dict:
    """Delete whole registration runs that used ``curation_key``, never part of one.

    WHY THE UNIT IS THE RUN
    ----------------------
    A registration over eleven sessions minus one session is not a valid registration: its
    ``n_sessions`` is wrong, every cell's ``n_sessions_present`` overcounts, and
    ``CellRegResult.PairQuality`` -- keyed on two plain smallints rather than two foreign
    keys, deliberately, to keep ten key columns off a ten-million-row table -- is left
    pointing at a column that no longer exists. So a curation can only be removed together
    with the runs that used it.

    WHAT DATAJOINT ALREADY DOES
    ---------------------------
    Nothing in this schema depends on production except through ``imaging.Curation``, and
    every table that does is a **part** table (``CellRegRun.Session``,
    ``CellRegTask.Session``, ``CellRegExport.File``, ``RoiGroupProjection.Scan``). Deleting
    a session or a curation therefore makes DataJoint's cascade reach a part without its
    master, and ``Table.delete`` refuses with "Attempt to delete part table X before
    deleting from its master Y first" (datajoint 0.14.3 table.py:147-156). **An upstream
    delete is blocked by default, and that is correct.** Two flags change it:

    * ``force_masters=True`` pulls the master into the cascade (table.py:92-109) -- the
      whole run goes, which is what this function does explicitly and visibly instead.
    * ``force_parts=True`` performs the half-delete described above. **Do not use it on
      this schema.**

    This function deletes from :class:`CellRegRun` itself, so the cascade runs downward
    from a master and no flag is needed at all.

    WHAT CANNOT BE RECOVERED
    ------------------------
    A run is identified by the sha256 of its result file and that file is immutable, so
    almost everything here can be rebuilt by re-ingesting it. Two things cannot, because
    they are not in the file:

    * :class:`RegistrationCuration` and its ``.CellDecision`` rows -- human decisions;
    * :class:`StableCellMap` -- the cross-run stable identities.

    So a run carrying curation decisions is refused unless ``allow_curated=True``.
    :class:`RoiGroupSnapshot` is untouched by design: it has no foreign key precisely so
    that it outlives the production rows it describes, and it is the only record of what
    ``roi_group`` held.

    Parameters
    ----------
    curation_key
        Any restriction on ``imaging.Curation`` -- a full curation key, a scan, a session.
    dry_run
        ``True`` (default) reports and deletes nothing. Deletion is never the default.
    allow_curated
        Permit deleting runs that carry :class:`RegistrationCuration` rows.

    Returns
    -------
    dict
        ``run_ids``, per-table row counts, ``curated_runs``, and ``deleted``.
    """
    runs = runs_touching(curation_key)
    run_ids = sorted(runs.fetch("cellreg_run_id"))
    restriction = [{"cellreg_run_id": r} for r in run_ids] or [{"cellreg_run_id": ""}]

    curated = sorted((RegistrationCuration & restriction).fetch("cellreg_run_id")) \
        if run_ids else []
    report = {
        "run_ids": run_ids,
        "curated_runs": sorted(set(curated)),
        "counts": {
            "CellRegRun": len(run_ids),
            "CellRegRun.Session": len(CellRegRun.Session & restriction) if run_ids else 0,
            "CellRegResult.Cell": len(CellRegResult.Cell & restriction) if run_ids else 0,
            "CellRegResult.Member": len(CellRegResult.Member & restriction) if run_ids else 0,
            "CellRegResult.PairQuality": (len(CellRegResult.PairQuality & restriction)
                                          if run_ids else 0),
            "MaskCoordinates": len(MaskCoordinates & restriction) if run_ids else 0,
            "RegistrationCuration": len(curated),
            "StableCellMap": (len(StableCellMap & restriction) if run_ids else 0),
        },
        "deleted": False,
    }
    if not run_ids:
        return report
    if report["curated_runs"] and not allow_curated:
        raise dj.DataJointError(
            f"runs {report['curated_runs']} carry RegistrationCuration rows. Those "
            "decisions are not in the result file and re-ingesting cannot restore them. "
            "Pass allow_curated=True only if you intend to lose them.")
    if dry_run:
        return report
    (CellRegRun & restriction).delete()
    report["deleted"] = True
    return report
