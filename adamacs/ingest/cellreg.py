"""Reading CellReg output into the cell-registration schema.

Every function that touches a ``.mat`` or a ``logFile_*.txt`` lives here; the schema module
only declares tables. The pure parsers below take paths or strings and return plain Python,
so they are unit-testable without a database connection -- which matters because the
production database is read-only for this work.

What the files actually look like (measured on 108 files, 2026-10-15)
--------------------------------------------------------------------
* 107 are MATLAB v7.3 (HDF5), 1 is an older v5/v7. The v7.3 files do **not** begin with the
  HDF5 signature: they carry a 116-byte ASCII userblock, so the signature sits at offset
  512 and the first bytes read ``MATLAB 7.3 MAT-file``. Detect by content, not by a guess.
* ``cell_to_index_map`` is 1-based with ``0`` meaning *not present*.
* ``cell_scores`` / ``true_positive_scores`` / ``true_negative_scores`` /
  ``exclusivity_scores`` / ``p_same_registered_pairs`` exist only when the run used the
  probabilistic approach; ``alignment_rotations`` only under translations-and-rotations
  (86 of 107 files).
* ``p_same_registered_pairs`` holds one ``n_sessions x n_sessions`` matrix per registered
  cell (verified square and equal to the session count in 106 of 107 files).
* ``spatial_footprints_corrected`` is either an in-file 3-D stack (up to 513x513x3363,
  ~7.1 GB as float64) or a path string. Never load it into the database.
* The result file contains **no parameters**. Those and the run-level accuracy live in the
  sibling ``logFile_*.txt``, whose timestamp trails the result file's.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
import re
import socket
import time

import datajoint as dj
import numpy as np

#: Fields present only under the probabilistic registration approach.
SCORE_FIELDS = ("cell_scores", "true_positive_scores", "true_negative_scores",
                "exclusivity_scores", "p_same_registered_pairs")

#: Minimum mask-set overlap for Jaccard-based column resolution to be accepted at all.
#: Matches the threshold the lab's own realign_global_ids_to_current_roi_group() uses.
MIN_JACCARD = 0.5

#: If a second candidate scan is within this of the best one, the column is ambiguous
#: rather than resolved -- we do not pick a winner by a hair.
AMBIGUITY_MARGIN = 0.1


# --------------------------------------------------------------------------- hashing
def file_sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def run_id_from_sha(sha256_hex: str) -> str:
    """The run's primary key: the first 40 hex chars of its content hash."""
    if len(sha256_hex) != 64:
        raise ValueError(f"expected a 64-char sha256, got {len(sha256_hex)} chars")
    return sha256_hex[:40]


# --------------------------------------------------------------------- format detection
def detect_mat_format(path) -> str:
    """``'v7.3_hdf5'``, ``'v5_v7'``, or ``'unknown:<header>'``.

    MATLAB v7.3 files are HDF5 behind a 116-byte ASCII userblock, so the signature is at
    offset 512 and byte 0 is text. Checking only offset 0 for the HDF5 magic misclassifies
    every v7.3 file -- which is exactly what happened on the first inventory pass.
    """
    with open(path, "rb") as fh:
        head = fh.read(140)
        fh.seek(512)
        sig = fh.read(8)
    text = head[:20].decode("latin-1", "replace")
    if sig == b"\x89HDF\r\n\x1a\n" or text.startswith("MATLAB 7.3"):
        return "v7.3_hdf5"
    if text.startswith("MATLAB 5.0"):
        return "v5_v7"
    return "unknown:" + text.strip()


# --------------------------------------------------------------------------- loading
#: The only fields the parser reads. Everything else is deliberately left on disk.
#:
#: ``spatial_footprints_corrected`` is excluded ON PURPOSE and this is load-bearing: 93 of
#: 107 files embed the stack, the largest is 513x513x3363 (~7.1 GB as float64), and nothing
#: in this schema stores it. An unrestricted ``mat73.loadmat()`` materialises it anyway and
#: will exhaust memory on the big runs for a field that is then discarded. Storage type is
#: determined from HDF5 *metadata* instead -- see :func:`probe_footprints_storage`.
#:
#: ``centroid_locations_corrected`` is also excluded for now: only MaskCoordinates needs it
#: and that make() is deliberately unimplemented pending the ordering check.
NEEDED_FIELDS = (
    "cell_to_index_map",
    "cell_scores", "true_positive_scores", "true_negative_scores", "exclusivity_scores",
    "p_same_registered_pairs",
    "is_cell_in_overlapping_FOV",
    "registered_cells_centroids",
    "alignment_x_translations", "alignment_y_translations", "alignment_rotations",
    "adjustment_x_zero_padding", "adjustment_y_zero_padding",
)

STRUCT_NAME = "cell_registered_struct"


def list_fields(path) -> list:
    """Field names present in the struct, read from HDF5 metadata without loading values."""
    if detect_mat_format(path) != "v7.3_hdf5":
        import scipy.io
        raw = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
        return list(getattr(raw[STRUCT_NAME], "_fieldnames", []))
    import h5py
    with h5py.File(path, "r") as f:
        grp = f.get(STRUCT_NAME)
        return list(grp.keys()) if grp is not None else []


def probe_footprints_storage(path) -> tuple:
    """``(storage, path)`` for the footprints, from metadata only.

    Opens the dataset but reads at most one small char array, so the multi-gigabyte stack is
    never materialised. Returns ``('in_file', '')``, ``('external_path', <path>)`` or
    ``('absent', '')``.
    """
    if detect_mat_format(path) != "v7.3_hdf5":
        return "absent", ""
    import h5py
    with h5py.File(path, "r") as f:
        ds = f.get(f"{STRUCT_NAME}/spatial_footprints_corrected")
        if ds is None:
            return "absent", ""
        if ds.dtype == object and ds.size:
            ref = ds[0, 0] if ds.ndim == 2 else ds[0]
            sub = f[ref]
            # A path is stored as a short char array; a stack is 3-D float. Only the
            # former is ever read.
            if sub.dtype.kind in "iu" and sub.ndim <= 2 and max(sub.shape or (0,)) < 4096:
                text = "".join(chr(int(c)) for c in np.asarray(sub).ravel())
                if "/" in text or "\\" in text:
                    return "external_path", text
            return "in_file", ""
        return "in_file", ""


def _load_v73_h5py(path, names) -> dict:
    """Read named struct fields with plain h5py, dereferencing cell arrays.

    Only the fields in ``names`` are touched, so the footprint stack is never read. Arrays
    come back in HDF5 (transposed) order; that is handled by :func:`orient_index_map` for
    the index map and by :func:`_centroid` for the centroids, and is irrelevant for vectors
    and for the square p_same matrices.
    """
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        grp = f[STRUCT_NAME]
        for name in names:
            ds = grp[name]
            if ds.dtype == object:                      # MATLAB cell array
                items = []
                refs = np.asarray(ds).ravel()
                for ref in refs:
                    try:
                        items.append(np.asarray(f[ref], dtype=float))
                    except (KeyError, ValueError, TypeError):
                        items.append(None)
                out[name] = items
            else:
                out[name] = np.asarray(ds)
    return out


def load_cellreg_struct(path, fields=NEEDED_FIELDS) -> dict:
    """Load the needed fields of ``cell_registered_struct`` as numpy arrays.

    ``mat73`` is the loader the lab documents for these files (Leon Kremers, 2024-05-24,
    ts=1716542082.875469) and it accepts ``only_include`` as a list of HDF5 paths, so the
    read is restricted to ``fields`` -- which is what keeps the footprint stack out of
    memory. Paths are intersected with what the file actually contains first, because 21 of
    107 files have no ``alignment_rotations`` and the score fields are absent from any
    simple-threshold run.
    """
    fmt = detect_mat_format(path)
    if fmt == "v7.3_hdf5":
        present = set(list_fields(path))
        names = [name for name in fields if name in present]
        try:
            import mat73
        except ImportError:
            # mat73 is absent on ibehavegpu1's shared environments and those are not ours
            # to mutate (leondrift ADR-008). For the handful of fields read here -- numeric
            # vectors, one 2-D map, and a cell array of square matrices -- plain h5py is
            # equivalent, because orientation is either irrelevant (vectors, square
            # matrices) or normalised downstream (orient_index_map, _centroid).
            return _load_v73_h5py(path, names)
        wanted = [f"{STRUCT_NAME}/{name}" for name in names]
        raw = mat73.loadmat(path, only_include=wanted, verbose=False)
        struct = raw.get(STRUCT_NAME, {})
        return {k: v for k, v in struct.items() if v is not None}
    if fmt == "v5_v7":
        # scipy cannot restrict to a struct field, so the whole struct is read. Acceptable
        # only because exactly one file on this server is in this format; if more appear,
        # this path needs its own memory story.
        import scipy.io
        raw = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
        struct = raw[STRUCT_NAME]
        return {name: getattr(struct, name) for name in struct._fieldnames
                if name in set(fields)}
    raise ValueError(f"unrecognised MAT format for {path}: {fmt}")


def orient_index_map(arr, n_sessions_hint=None) -> np.ndarray:
    """Return ``cell_to_index_map`` as ``[n_cells, n_sessions]``.

    Loaders disagree about orientation: raw ``h5py`` yields ``[n_sessions, n_cells]`` while
    ``mat73`` transposes back to MATLAB's ``[n_cells, n_sessions]``. Rather than trusting
    either, orient explicitly -- and refuse when the answer is genuinely ambiguous instead
    of assuming "sessions are the shorter axis", which is false for a 2-cell test file.
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"cell_to_index_map must be 2-D, got shape {a.shape}")
    if n_sessions_hint is not None:
        if a.shape[1] == n_sessions_hint and a.shape[0] != n_sessions_hint:
            return a
        if a.shape[0] == n_sessions_hint and a.shape[1] != n_sessions_hint:
            return a.T
        if a.shape == (n_sessions_hint, n_sessions_hint):
            return a  # square: orientation is unknowable and also irrelevant
        raise ValueError(
            f"cell_to_index_map shape {a.shape} matches neither axis to the "
            f"{n_sessions_hint} sessions the log file reports")
    if a.shape[0] == a.shape[1]:
        raise ValueError(
            f"cell_to_index_map is square {a.shape} and no session count was supplied; "
            "orientation cannot be determined -- pass n_sessions_hint from the log file")
    return a if a.shape[0] > a.shape[1] else a.T


# --------------------------------------------------------------------------- log file
_LOG_PATTERNS = {
    "n_sessions": r"Number of sessions\s*-\s*(\d+)",
    "reference_session_index": r"Reference session\s*-\s*(\d+)",
    "alignment_type": r"Alignment Type\s*-\s*(.+)",
    "maximal_distance_um": r"Model maximal distance\s*-\s*([\d.]+)",
    "n_bins": r"Number of bins\s*-\s*(\d+)",
    "initial_registration_type": r"Initial registration type\s*-\s*(.+)",
    "initial_threshold": r"Initial threshold\s*-\s*([\d.]+)",
    "registration_approach": r"Registration approach\s*-\s*(.+)",
    "model_type": r"Final registration type\s*-\s*(.+)",
    "final_threshold": r"P_same threshold\s*-\s*([\d.]+)",
    "n_cells_final": r"Final number of cells\s*-\s*(\d+)",
    "avg_fraction_active": r"Average fraction of active cells\s*-\s*([\d.]+)",
    "frac_false_positives": r"Fraction of false positives\s*-\s*([\d.]+)",
    "frac_false_negatives": r"Fraction of false negatives\s*-\s*([\d.]+)",
    "frac_uncertain_pairs": r"Fraction of uncertain cell-pairs\s*-\s*([\d.]+)",
    "avg_register_score": r"Average register score\s*-\s*([\d.]+)",
    "avg_true_positive_score": r"Average true positive score\s*-\s*([\d.]+)",
    "avg_true_negative_score": r"Average true negative score\s*-\s*([\d.]+)",
    "avg_exclusivity_score": r"Average exclusivity score\s*-\s*([\d.]+)",
    "model_discrepancy": r"Discrepancy of the model\s*-\s*([\d.]+)",
}
_FLOAT_KEYS = {"maximal_distance_um", "initial_threshold", "final_threshold",
               "avg_fraction_active", "frac_false_positives", "frac_false_negatives",
               "frac_uncertain_pairs", "avg_register_score", "avg_true_positive_score",
               "avg_true_negative_score", "avg_exclusivity_score", "model_discrepancy"}
_INT_KEYS = {"n_sessions", "reference_session_index", "n_bins", "n_cells_final"}


def parse_log_file(text: str) -> dict:
    """Parse ``logFile_*.txt`` into parameters, accuracy and the session list.

    Returns a dict with the scalar fields above plus ``session_files``: the per-column
    source paths, in column order, as listed under "Sessions list".
    """
    out = {}
    for key, pattern in _LOG_PATTERNS.items():
        m = re.search(pattern, text)
        if not m:
            continue
        value = m.group(1).strip()
        if key in _FLOAT_KEYS:
            out[key] = float(value)
        elif key in _INT_KEYS:
            out[key] = int(value)
        else:
            out[key] = value
    sessions = re.findall(r"^Session\s+(\d+)\s*-\s*(.+?)\s*$", text, re.M)
    out["session_files"] = [path for _, path in sorted(sessions, key=lambda kv: int(kv[0]))]
    return out


def normalise_alignment_mode(alignment_type) -> str:
    text = (alignment_type or "").strip().lower()
    if text.startswith("translations and rotations"):
        return "translations_and_rotations"
    if text.startswith("non-rigid") or text.startswith("non rigid"):
        return "non_rigid"
    if text.startswith("translations"):
        return "translations"
    return "unknown"


def normalise_registration_approach(approach) -> str:
    text = (approach or "").strip().lower()
    if text.startswith("probabilistic"):
        return "probabilistic"
    if text.startswith("simple"):
        return "simple_threshold"
    return "unknown"


def find_log_file(result_path) -> str | None:
    """The ``logFile_*.txt`` belonging to a result file.

    The log is written *after* the ``.mat`` (76 s later in a verified pair), so the
    timestamps do not match and an exact-name pairing finds nothing. Rule: within the same
    directory, the nearest log at or after the result file's own timestamp; failing that,
    the nearest earlier one. Returns ``None`` when the directory has no log at all.
    """
    directory = os.path.dirname(result_path)
    stamp = re.search(r"(\d{8}_\d{6})", os.path.basename(result_path))
    try:
        logs = [f for f in os.listdir(directory)
                if f.lower().startswith("logfile") and f.lower().endswith(".txt")]
    except OSError:
        return None
    if not logs:
        return None
    if stamp is None or len(logs) == 1:
        return os.path.join(directory, sorted(logs)[0])
    target = stamp.group(1)
    stamped = []
    for name in logs:
        m = re.search(r"(\d{8}_\d{6})", name)
        if m:
            stamped.append((m.group(1), name))
    if not stamped:
        return os.path.join(directory, sorted(logs)[0])
    later = sorted(s for s in stamped if s[0] >= target)
    if later:
        return os.path.join(directory, later[0][1])
    return os.path.join(directory, sorted(stamped)[-1][1])


# ------------------------------------------------------------------- column resolution
def scan_id_from_source_file(name) -> str | None:
    """Pull the scan id out of a converted session filename.

    ``converted_Fall_01_scan9FOW97IH.mat`` -> ``scan9FOW97IH``. This is what makes the log
    file authoritative for the column-to-scan mapping rather than merely suggestive.
    """
    m = re.search(r"(scan[0-9A-Za-z]{8})", os.path.basename(str(name)))
    return m.group(1) if m else None


def jaccard(a, b) -> float:
    """Overlap of two mask-id sets; 0.0 when both are empty."""
    sa, sb = set(a), set(b)
    union = sa | sb
    return (len(sa & sb) / len(union)) if union else 0.0


def resolve_column(column_masks, candidates, *, log_scan_id=None,
                   min_jaccard=MIN_JACCARD, margin=AMBIGUITY_MARGIN) -> dict:
    """Decide which scan a CellReg column belongs to.

    ``candidates`` maps a scan key (any hashable, typically a dict made hashable by the
    caller) to that scan's mask-id set. Jaccard is computed against **every** candidate and
    reported whatever the outcome, because a low overlap on a column the log resolved is
    evidence of a stale ``Fall.mat`` rather than a resolution failure (Leon Kremers /
    Tobias Rose, 2024-05-24, ts=1716546557.549569).

    Returns ``resolution_method``, ``scan_key`` (None when unresolved), ``mask_set_jaccard``
    and ``ambiguous``.
    """
    scores = {key: jaccard(column_masks, masks) for key, masks in candidates.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_key, best_score = (ranked[0] if ranked else (None, 0.0))
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

    if log_scan_id is not None:
        match = [key for key in candidates if log_scan_id in str(key)]
        if len(match) == 1:
            return {"resolution_method": "log_file", "scan_key": match[0],
                    "mask_set_jaccard": scores[match[0]], "ambiguous": False}

    if best_key is not None and best_score >= min_jaccard:
        if best_score - runner_up < margin:
            return {"resolution_method": "unresolved", "scan_key": None,
                    "mask_set_jaccard": best_score, "ambiguous": True}
        return {"resolution_method": "mask_jaccard", "scan_key": best_key,
                "mask_set_jaccard": best_score, "ambiguous": False}

    return {"resolution_method": "unresolved", "scan_key": None,
            "mask_set_jaccard": best_score if ranked else None, "ambiguous": False}


# ------------------------------------------------------------------ map interpretation
def members_from_index_map(index_map) -> dict:
    """``{cell_index: [(column_index, mask_0based), ...]}`` from a 1-based index map.

    CellReg writes a 1-based mask id and ``0`` for *not present*; subtracting one sends
    absent to ``-1``. Absence is represented by **omitting the entry**, never by storing a
    sentinel -- otherwise every consumer inherits the obligation to filter ``< 0`` forever,
    which is what the lab's realign helper has to do today.
    """
    a = np.asarray(index_map)
    out = {}
    for cell_index in range(a.shape[0]):
        row = a[cell_index]
        present = [(int(col), int(row[col]) - 1)
                   for col in range(a.shape[1]) if int(row[col]) > 0]
        out[cell_index] = present
    return out


def pair_quality_rows(p_same_matrix, present_columns) -> list[tuple[int, int, float]]:
    """Upper-triangle ``(column_a, column_b, p_same)`` for one cell's present columns.

    ``p_same_registered_pairs`` is one ``n_sessions x n_sessions`` matrix per registered
    cell. Only pairs where the cell is present in *both* sessions are emitted -- that
    restriction is what keeps this table at ~10.7 M rows lab-wide instead of
    cells x all-pairs.
    """
    if p_same_matrix is None:
        return []
    m = np.asarray(p_same_matrix, dtype=float)
    # mat73 hands each cell-array entry back with a leading singleton axis -- a real file
    # yields (1, n_sessions, n_sessions), not (n_sessions, n_sessions). Strip leading
    # 1-axes only, so a genuinely wrong shape still raises.
    while m.ndim > 2 and m.shape[0] == 1:
        m = m[0]
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"p_same matrix must be square, got shape {m.shape}")
    cols = sorted(int(c) for c in present_columns)
    rows = []
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1:]:
            if col_a >= m.shape[0] or col_b >= m.shape[0]:
                continue
            value = m[col_a, col_b]
            if np.isfinite(value):
                rows.append((col_a, col_b, float(value)))
    return rows


def _fields_except_p_same(path):
    # NEEDED_FIELDS minus p_same_registered_pairs, which is read separately.
    #
    # Derived from NEEDED_FIELDS, NOT from the file: enumerating the file pulls in
    # spatial_footprints_corrected and centroid_locations_corrected, which is exactly
    # what the NEEDED_FIELDS default exists to prevent (~2.9 GB for 11 sessions).
    if detect_mat_format(path) != "v7.3_hdf5":
        return NEEDED_FIELDS
    return tuple(f for f in NEEDED_FIELDS if f != "p_same_registered_pairs")

def read_p_same_fast(path):
    """``p_same_registered_pairs`` by raw HDF5 dereference, or None for a non-v7.3 file.

    mat73 takes 94.9 s to hand back this one field for a 4,107-cell run; dereferencing the
    same 4,107 object references with h5py takes 0.68 s -- 140x -- and yields identical
    values (verified element-wise on sampled cells). The field is a cell array with one
    n_sessions x n_sessions matrix PER REGISTERED CELL, so the cost is per-cell overhead in
    mat73's generic machinery, not in HDF5: at 597,212 cells lab-wide the difference is the
    whole ingest.

    Entries are returned squeezed to 2-D. ``pair_quality_rows`` and
    ``p_same_is_consistent`` both tolerate a leading singleton axis anyway, so callers do
    not need to care which loader produced the list.
    """
    if detect_mat_format(path) != "v7.3_hdf5":
        return None
    import h5py   # optional dependency: imported locally, as elsewhere
    with h5py.File(path, "r") as handle:
        names = []
        handle.visititems(
            lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
        hits = [n for n in names if n.rsplit("/", 1)[-1] == "p_same_registered_pairs"]
        if not hits:
            return None
        refs = _np_ravel(handle[hits[0]])
        out = []
        for ref in refs:
            arr = np.asarray(handle[ref], dtype=float)
            while arr.ndim > 2 and arr.shape[0] == 1:
                arr = arr[0]
            out.append(arr)
    return out


def _np_ravel(dataset):
    """Flatten an HDF5 reference dataset to a 1-D sequence of references."""
    return np.array(dataset).ravel()


def p_same_is_consistent(p_same_all, n_sessions) -> bool:
    """Is ``p_same_registered_pairs`` shaped ``n_sessions x n_sessions`` per cell?

    Each entry arrives with a LEADING SINGLETON AXIS: a real file yields
    ``(1, n_sessions, n_sessions)`` per cell, not ``(n_sessions, n_sessions)``.
    ``pair_quality_rows`` has always stripped that; the inline check in
    ``make_cellreg_result`` did not, so every probabilistic run was declared
    inconsistent and the ``if consistent`` guard then suppressed EVERY PairQuality row
    and left min/median p_same NULL on every cell. Extracted here so the quirk is
    asserted by a test rather than rediscovered by populating 101 runs.
    """
    if p_same_all is None or not len(p_same_all):
        return True
    first = np.asarray(p_same_all[0])
    while first.ndim > 2 and first.shape[0] == 1:
        first = first[0]
    return bool(first.ndim == 2 and first.shape[0] == first.shape[1]
                and first.shape[0] == n_sessions)


def summarise_p_same(pair_rows) -> tuple[float | None, float | None]:
    """``(min, median)`` over a cell's present pairs -- the queryable selector columns."""
    if not pair_rows:
        return None, None
    values = np.array([p for _, _, p in pair_rows], dtype=float)
    return float(values.min()), float(np.median(values))


def footprints_storage(struct) -> tuple[str, str]:
    """Same answer as :func:`probe_footprints_storage`, but for an ALREADY-LOADED struct.

    Not the production path -- ``make()`` uses the metadata probe so that the stack is never
    read. This variant exists for callers that already hold a struct in memory (and for the
    tests), and it is only safe when that struct was loaded with ``spatial_footprints_
    corrected`` restricted out or known small.

    Where the stack is external the path is recorded verbatim and **not** dereferenced --
    one such path is a Windows drive letter that cannot resolve on the Linux server.
    """
    value = struct.get("spatial_footprints_corrected")
    if value is None:
        return "absent", ""
    first = value[0] if isinstance(value, (list, tuple)) and value else value
    if isinstance(first, (str, bytes)):
        text = first.decode() if isinstance(first, bytes) else first
        return "external_path", str(text)
    if isinstance(first, np.ndarray) and first.dtype.kind in "iu" and first.ndim <= 2:
        try:
            text = "".join(chr(int(c)) for c in np.asarray(first).ravel())
            if "/" in text or "\\" in text:
                return "external_path", text
        except (TypeError, ValueError):
            pass
    return "in_file", ""


# --------------------------------------------------------------------------- make()
def make_cellreg_result(table, key):
    """Populate :class:`CellRegResult` and its parts for one run.

    Failure is recorded, never swallowed: an unreadable or internally inconsistent file
    yields a master row with ``parse_ok = 0`` and a stated error, and no child rows.
    """
    # Imported inside the function: the schema module imports this one lazily from make(),
    # so a module-level import here would be circular on first load.
    from adamacs.schemas import cell_registration as schema_module

    started = time.time()
    run_row = (schema_module.CellRegRun & key).fetch1()
    path = run_row["result_file"]
    base = dict(key, parser_version=schema_module.PARSER_VERSION,
                parsed_at=_dt.datetime.now())

    try:
        sha = file_sha256(path)
        if sha != run_row["result_file_sha256"]:
            raise ValueError(
                f"result file changed since registration: sha256 {sha[:12]} != "
                f"{run_row['result_file_sha256'][:12]}")
        struct = load_cellreg_struct(path, fields=_fields_except_p_same(path))
        fast = read_p_same_fast(path)
        if fast is not None:
            struct["p_same_registered_pairs"] = fast
        elif "p_same_registered_pairs" not in struct:
            struct.update(load_cellreg_struct(
                path, fields=("p_same_registered_pairs",)))
    except Exception as exc:  # noqa: BLE001 -- the failure IS the record
        table.insert1(dict(base, parse_ok=0, parse_error=f"{type(exc).__name__}: {exc}"[:512],
                           source_sha256=run_row["result_file_sha256"], n_cells=0,
                           n_sessions=run_row["n_sessions"], has_score_fields=0,
                           missing_fields="", p_same_consistent=0,
                           footprints_storage="absent", footprints_path="",
                           compute_seconds=time.time() - started))
        return

    missing = [f for f in SCORE_FIELDS if f not in struct]
    index_map = orient_index_map(struct["cell_to_index_map"],
                                 n_sessions_hint=run_row["n_sessions"])
    n_cells, n_sessions = index_map.shape

    p_same_all = struct.get("p_same_registered_pairs")
    consistent = p_same_is_consistent(p_same_all, n_sessions)

    # From metadata, not from the loaded struct: the footprint stack is never read.
    storage, fp_path = probe_footprints_storage(path)
    table.insert1(dict(base, parse_ok=1, parse_error="", source_sha256=sha,
                       n_cells=int(n_cells), n_sessions=int(n_sessions),
                       has_score_fields=int(not missing), missing_fields=";".join(missing)[:255],
                       p_same_consistent=int(consistent), footprints_storage=storage,
                       footprints_path=fp_path[:512], compute_seconds=time.time() - started))

    # Pair-quality completeness, counted BEFORE the per-cell loop so the stored row count
    # can be compared against what the index map implies (see CellRegResult.n_pairs_*).
    present_counts = (index_map > 0).sum(axis=1)
    n_pairs_expected = int((present_counts * (present_counts - 1) // 2).sum())

    members = members_from_index_map(index_map)
    # column_index -> the curation key that column was resolved to. Unresolved columns are
    # simply absent, which is what makes the Member loop below able to skip them.
    curation_attrs = ("session_id", "scan_id", "paramset_idx", "curation_id")
    column_to_curation = {
        int(row["column_index"]): {a: row[a] for a in curation_attrs}
        for row in (schema_module.CellRegRun.Session & key).fetch(as_dict=True)}
    cell_rows, member_rows, pair_rows = [], [], []
    n_pairs_emitted = 0
    for cell_index, present in members.items():
        columns = [col for col, _ in present]
        pairs = []
        if consistent and p_same_all is not None and cell_index < len(p_same_all):
            pairs = pair_quality_rows(p_same_all[cell_index], columns)
        n_pairs_emitted += len(pairs)
        lo, mid = summarise_p_same(pairs)
        cell_rows.append(dict(
            key, cell_index=int(cell_index),
            cell_score=_scalar(struct.get("cell_scores"), cell_index),
            true_positive_score=_scalar(struct.get("true_positive_scores"), cell_index),
            true_negative_score=_scalar(struct.get("true_negative_scores"), cell_index),
            exclusivity_score=_scalar(struct.get("exclusivity_scores"), cell_index),
            in_overlapping_fov=_bool(struct.get("is_cell_in_overlapping_FOV"), cell_index),
            centroid_x_registered=_centroid(struct.get("registered_cells_centroids"), cell_index, 0),
            centroid_y_registered=_centroid(struct.get("registered_cells_centroids"), cell_index, 1),
            n_sessions_present=len(present), min_p_same=lo, median_p_same=mid))
        for col, mask in present:
            # Member inherits CellRegRun.Session, whose key is the CURATION
            # (session_id, scan_id, paramset_idx, curation_id) -- column_index is a
            # secondary attribute there and is not an attribute of Member at all. Writing
            # column_index here raised on every insert; it was never caught because the
            # database is read-only in this project and no test inserts.
            curation = column_to_curation.get(int(col))
            if curation is None:
                # The column could not be resolved to a curation, so it lives in
                # .UnresolvedColumn and HAS no curation key. The cell still counts this
                # session in n_sessions_present -- the membership is real, only its
                # database identity is unknown -- but there is no row to join through.
                continue
            member_rows.append(dict(key, cell_index=int(cell_index), **curation,
                                    mask=int(mask)))
        for col_a, col_b, value in pairs:
            pair_rows.append(dict(key, cell_index=int(cell_index), column_a=col_a,
                                  column_b=col_b, p_same=value))

    # Completeness recorded on the master now that the loop has counted. update1 is
    # the dj 0.14 API for amending a secondary attribute; _update does not exist on a
    # restricted table and raised AttributeError inside make().
    # n_pairs_nonfinite means ONE thing: a p_same value was read and found non-finite.
    # When the field is absent (simple-threshold runs) or inconsistent, no value was
    # ever read, so the difference is unavailability -- recording it as non-finite
    # mislabelled 5,970 pairs of one run and inflated the lab-wide total. NULL there,
    # with p_same_consistent carrying the explanation.
    n_pairs_nonfinite = (max(0, n_pairs_expected - n_pairs_emitted)
                         if (consistent and p_same_all is not None) else None)
    table.update1({**key, "n_pairs_expected": n_pairs_expected,
                  "n_pairs_nonfinite": n_pairs_nonfinite})

    table.Cell.insert(cell_rows)
    table.Member.insert(member_rows)
    if pair_rows:
        table.PairQuality.insert(pair_rows)


def _scalar(array, index):
    if array is None:
        return None
    flat = np.asarray(array).ravel()
    if index >= flat.size:
        return None
    value = float(flat[index])
    return None if not np.isfinite(value) else value


def _bool(array, index):
    value = _scalar(array, index)
    return None if value is None else int(bool(value))


def _centroid(array, index, axis):
    if array is None:
        return None
    a = np.asarray(array, dtype=float)
    if a.ndim != 2:
        return None
    if a.shape[0] == 2 and a.shape[1] != 2:
        a = a.T
    if index >= a.shape[0] or axis >= a.shape[1]:
        return None
    value = a[index, axis]
    return None if not np.isfinite(value) else float(value)


def make_mask_coordinates(table, key):
    """Populate :class:`MaskCoordinates` for one resolved column.

    The ordering question that blocked this (known issue L8) is settled by the file
    itself: ``centroid_locations_corrected`` is a per-session cell array whose entry for
    session 1 of the ROS-1881 run is a ``(1131, 2)`` MATLAB array -- exactly that scan's
    mask count -- so row ``k`` is mask ``k`` in the session's own order, the same order
    the converted input file was written in.

    Measured by ``scripts/verify_footprints.py``, which prints the array's shape beside the
    curation's mask count for whichever column it is given; columns 0, 4 and 7 of
    ``cellRegistered_20260129_075504.mat`` reported ``(2, 1131)``, ``(2, 1111)`` and
    ``(2, 1252)`` (HDF5 reverses MATLAB's dimension order) against 1131, 1111 and 1252
    masks in ``Segmentation.Mask``. Re-run it for any run whose ordering you doubt.

    Centroids are in **pixels**: ``compute_centroid_locations.m`` returns
    ``max_ind + gaussfit_offset`` with no micron scaling, unlike the translations, which
    are scaled on the way out. They are MATLAB 1-based and are stored 0-based here, to
    match ``Segmentation.Mask.mask_center_x/y``.
    """
    from adamacs.schemas import cell_registration as schema_module

    run_key = {"cellreg_run_id": key["cellreg_run_id"]}
    run_row = (schema_module.CellRegRun & run_key).fetch1()
    column_index = int((schema_module.CellRegRun.Session & key).fetch1("column_index"))

    struct = load_cellreg_struct(run_row["result_file"],
                                 fields=("centroid_locations_corrected",))
    per_session = struct.get("centroid_locations_corrected")
    if per_session is None:
        raise ValueError(
            f"{run_row['result_file']} has no centroid_locations_corrected; "
            "MaskCoordinates cannot be populated for this run")
    centroids = np.asarray(per_session[column_index], dtype=float)
    centroids = np.squeeze(centroids)
    if centroids.ndim == 2 and centroids.shape[0] == 2 and centroids.shape[1] != 2:
        centroids = centroids.T              # (2, n) -> (n, 2); square n == 2 is absurd

    native = (_imaging_module().Segmentation.Mask & key).fetch(
        "mask", "mask_center_x", "mask_center_y", order_by="mask", as_dict=True)
    if len(native) != centroids.shape[0]:
        raise ValueError(
            f"column {column_index} has {centroids.shape[0]} centroids but the curation "
            f"has {len(native)} masks -- the input file is not this curation")

    table.insert([
        dict(key, mask=int(row["mask"]),
             centroid_x_native=float(row["mask_center_x"]),
             centroid_y_native=float(row["mask_center_y"]),
             centroid_x_registered=float(centroids[k, 0]) - 1.0,
             centroid_y_registered=float(centroids[k, 1]) - 1.0)
        for k, row in enumerate(native)])


def _imaging_module():
    """The upstream imaging module, via the schema module's linking module."""
    from adamacs.schemas import cell_registration as schema_module
    return schema_module._imaging()


def aligned_footprints_for_column(session_table, *, masks=None, dense=False) -> dict:
    """Implementation of :meth:`CellRegRun.Session.aligned_footprints`.

    See :mod:`adamacs.ingest.cellreg_footprints` for the five conventions this has to
    respect. There is no approximation: the pre-alignment centroid the radius crop is
    measured from is recomputed by ``centroid_location()`` from the mask itself, so this
    needs neither the result file nor ``MaskCoordinates``.
    """
    from adamacs.ingest import cellreg_footprints as fpmod
    from adamacs.schemas import cell_registration as schema_module

    key = session_table.fetch1("KEY")
    row = session_table.fetch1()
    run_key = {"cellreg_run_id": key["cellreg_run_id"]}
    run_row = (schema_module.CellRegRun & run_key).fetch1()

    quality = (schema_module.CellRegRun.Quality & run_key)
    mpp = quality.fetch1("microns_per_pixel") if quality else None
    if mpp is None:
        raise dj.DataJointError(
            "microns_per_pixel is unknown for this run, and the aligned footprint cannot "
            "be built without it: the stored translations are in microns and the radius "
            "crop is in microns. Populate CellRegRun.Quality from the run's logFile.")

    mask_query = _imaging_module().Segmentation.Mask & key
    if masks is not None:
        mask_query = mask_query & [{"mask": int(m)} for m in np.atleast_1d(masks)]
    mask_rows = mask_query.fetch("mask", "mask_xpix", "mask_ypix", "mask_weights",
                                 order_by="mask", as_dict=True)

    # Frame geometry: the log's padded size if it was parsed, else the motion-corrected
    # summary image plus the driver's own +1 (cellreg_run.m:130-131).
    height = width = None
    if quality:
        height = quality.fetch1("image_height_px")
        width = quality.fetch1("image_width_px")
    if not height or not width:
        summary = (_imaging_module().MotionCorrection.Summary & key).fetch1("average_image")
        height, width = np.asarray(summary).shape[0] + 1, np.asarray(summary).shape[1] + 1

    pad_h = int(height) + int(row["zero_padding_y"] or 0)
    pad_w = int(width) + int(row["zero_padding_x"] or 0)
    is_reference = (run_row["reference_session_index"] is not None
                    and int(row["column_index"]) + 1 == int(run_row["reference_session_index"]))

    out = {}
    for m in mask_rows:
        mask_id = int(m["mask"])
        img = fpmod.aligned_footprint(
            m["mask_xpix"], m["mask_ypix"], m["mask_weights"],
            height=int(height), width=int(width),
            dx_um=row["alignment_x_translation"] or 0.0,
            dy_um=row["alignment_y_translation"] or 0.0,
            rotation_deg=row["alignment_rotation"],
            microns_per_pixel=float(mpp), is_reference=is_reference,
            pad_height=pad_h, pad_width=pad_w)
        out[mask_id] = img if dense else fpmod.to_sparse(img)
    return out


def aligned_footprints_for_cell(cell_table) -> dict:
    """Implementation of :meth:`CellRegResult.Cell.aligned_footprints`."""
    from adamacs.schemas import cell_registration as schema_module

    key = cell_table.fetch1("KEY")
    members = (schema_module.CellRegResult.Member & key).fetch(as_dict=True)
    curation_attrs = ("session_id", "scan_id", "paramset_idx", "curation_id")
    out = {}
    for member in members:
        ckey = {"cellreg_run_id": key["cellreg_run_id"],
                **{a: member[a] for a in curation_attrs}}
        session_row = schema_module.CellRegRun.Session & ckey
        column_index = int(session_row.fetch1("column_index"))
        footprints = aligned_footprints_for_column(session_row,
                                                   masks=[int(member["mask"])])
        out[column_index] = {"key": ckey, "mask": int(member["mask"]),
                             "footprint": footprints[int(member["mask"])]}
    return dict(sorted(out.items()))


def ingest_host() -> str:
    return socket.gethostname()
