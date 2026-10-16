"""Build CellReg input files from the database, instead of copying ``Fall.mat``.

Why this module exists
---------------------
The existing export copies ``<ScanPath>/suite2p/plane0/Fall.mat`` -- a per-scan
**filesystem** path with no curation in the lookup. The curation a CellReg column came from
was therefore never an input, and it cannot be recovered afterwards: of 743
``(session, scan, paramset)`` groups holding more than one curation, 696 (94%) have
byte-identical mask sets, so mask-set overlap scores 1.0 against every candidate (design
review 3.3b). Copying the file also inherits the staleness problem -- ``Fall.mat`` does not
update when suite2p curation does (2024-05-24, ts=1716546557.549569) and on-disk folders
keep only the last extraction (ts=1722492807.667179) -- so the input can correspond to *no*
curation row in the database.

Building the file from ``imaging.Segmentation.Mask`` for an explicit
``(paramset_idx, curation_id)`` removes both problems at once: the curation is an input by
construction, and the masks are the curated set by definition.

What the MATLAB driver actually needs
-------------------------------------
``Leon_CellReg.m`` with ``input_format = 'Suite2p'`` reads **exactly** (lines 82-92):

* ``ops.Lx``, ``ops.Ly`` -- the FOV, used as ``Lx+1`` x ``Ly+1``
* ``size(stat, 2)`` -- the ROI count, so ``stat`` must be a 1xN cell array
* ``stat{k}.ypix``, ``stat{k}.xpix``, ``stat{k}.lam`` -- the footprint, 0-based pixels

Nothing else. No ``F``, no ``Fneu``, no ``spks``, **not even** ``iscell``: the driver
registers every entry of ``stat``, so the file must contain only the ROIs that belong to the
curation -- which is what the database holds. ``iscell`` and ``ops.meanImg`` are written
anyway so the file opens in other tools, but the driver ignores them.

Verified: for all 11 gratings scans of site ``sess9FPXNI8U`` (13,532 ROIs) the generated
file reproduces the on-disk ``Fall.mat`` exactly -- identical pixel sets and identical
``lam`` values, in the same order, for every ROI. Because the conversion is a deterministic
scatter of ``lam`` into ``(ypix+1, xpix+1)``, identical pixels and weights means identical
converted footprints by construction.

A note on what is and is not a parameter
---------------------------------------
``settings.mat`` carries exactly the 16 fields the driver reads as ``settings.*``. Two
quantities that appear in ``logFile_*.txt`` and look like parameters are **not**:
``number_of_bins`` is computed by ``estimate_number_of_bins`` (line 245) and
``initial_threshold`` is the fitted model's p_same=0.5 crossing, or a hardcoded 0.65 fallback
(lines 353-355). They are *outputs*. They belong in ``CellRegRun.Quality``, never in
``CellRegParamSet``.
"""
from __future__ import annotations

import datetime as _dt
import getpass
import hashlib
import os
import pathlib
import subprocess

import numpy as np
import scipy.io

EXPORTER_VERSION = "1.0.0"

#: The complete set of fields ``Leon_CellReg.m`` reads as ``settings.<field>``.
SETTINGS_FIELDS = (
    "input_format",
    "alignment_type",
    "initial_registration_type",
    "registration_approach",
    "model_type",
    "maximal_distance",
    "maximal_rotation",
    "transformation_smoothness",
    "microns_per_pixel",
    "reference_session_index",
    "p_same_threshold",
    "p_same_certainty_threshold",
    "sufficient_correlation_centroids",
    "sufficient_correlation_footprints",
    "memory_efficient_run",
    "use_parallel_processing",
)

#: Defaults matching the lab's standard probabilistic run.
SETTINGS_DEFAULTS = {
    "input_format": "Suite2p",
    "alignment_type": "Translations and Rotations",
    "initial_registration_type": "Spatial correlation",
    "registration_approach": "Probabilistic modeling",
    "model_type": "Spatial correlations",
    "maximal_distance": 12.0,
    "maximal_rotation": 30.0,
    "transformation_smoothness": 2.0,
    "microns_per_pixel": 1.2024,
    "reference_session_index": 1,
    "p_same_threshold": 0.5,
    "p_same_certainty_threshold": 0.95,
    "sufficient_correlation_centroids": 0.1,
    "sufficient_correlation_footprints": 0.1,
    "memory_efficient_run": True,
    "use_parallel_processing": True,
}

DERIVED_NOT_PARAMETERS = ("number_of_bins", "n_bins", "initial_threshold")


def current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER", "")


def adamacs_sha() -> str:
    """Short git sha of the checkout this is running from, '' if not a repo."""
    try:
        here = pathlib.Path(__file__).resolve().parent
        out = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip()[:40] if out.returncode == 0 else ""
    except Exception:
        return ""


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_safe_target(path, *, allow_overwrite: bool = False) -> pathlib.Path:
    """Refuse to write anywhere that could damage a suite2p output.

    Two hard rules, because this module writes into the shared data folder:

    1. never inside a ``suite2p`` directory, and never a file called ``Fall.mat`` --
       the curated suite2p output is not ours to touch;
    2. never over an existing file unless explicitly allowed.
    """
    p = pathlib.Path(path)
    parts = {q.lower() for q in p.parts}
    if "suite2p" in parts or any(q.lower().startswith("plane") for q in p.parts):
        raise ValueError(
            f"refusing to write inside a suite2p output tree: {p}. CellReg inputs belong in "
            f"a separate export directory; the suite2p Fall.mat must not be overwritten.")
    if p.name.lower() == "fall.mat":
        raise ValueError(
            f"refusing to write a file named Fall.mat: {p}. Use the Fall_<NN>_<scan_id>.mat "
            f"form so an export can never be mistaken for a suite2p output.")
    if p.exists() and not allow_overwrite:
        raise FileExistsError(f"{p} exists; pass allow_overwrite=True to replace it")
    return p


def export_dir_for(site_key: dict, *, label: str = "", task_id: int = 0,
                   root: str = "/datajoint-data/data") -> pathlib.Path:
    """Where a task's inputs go: a NEW subfolder, never a suite2p or scan directory.

    ``<root>/<subject>/cellreg_export/<same_site_id>[_<label>]_task<NN>``. The MATLAB driver
    consumes this folder flat and creates ``Results/`` inside it.
    """
    subject = site_key["subject"]
    site = site_key["same_site_id"]
    stem = f"{site}_{label}" if label else site
    stem = stem.replace(" ", "_").replace("/", "_")
    return pathlib.Path(root) / subject / "cellreg_export" / f"{stem}_task{task_id:02d}"


def fall_payload(masks, fov_height: int, fov_width: int, mean_img=None) -> dict:
    """The ``Fall.mat`` contents the driver needs, from database mask rows.

    ``masks`` is an iterable of dicts carrying ``mask_xpix``, ``mask_ypix`` and
    ``mask_weights``, **already ordered by** ``mask`` -- the order becomes ``stat``'s order
    and therefore the 1-based index CellReg writes into ``cell_to_index_map``.
    """
    masks = list(masks)
    if not masks:
        raise ValueError("no masks: refusing to write an empty Fall file")
    stat = np.empty((1, len(masks)), dtype=object)
    for i, r in enumerate(masks):
        xp = np.asarray(r["mask_xpix"], dtype=np.int32).ravel()
        yp = np.asarray(r["mask_ypix"], dtype=np.int32).ravel()
        lam = np.asarray(r["mask_weights"], dtype=np.float64).ravel()
        if not (xp.size == yp.size == lam.size):
            raise ValueError(
                f"mask {r.get('mask')} has inconsistent pixel arrays: "
                f"xpix={xp.size} ypix={yp.size} lam={lam.size}")
        if xp.max(initial=-1) >= fov_width or yp.max(initial=-1) >= fov_height:
            raise ValueError(
                f"mask {r.get('mask')} has pixels outside the {fov_height}x{fov_width} FOV")
        stat[0, i] = {"xpix": xp[None, :], "ypix": yp[None, :], "lam": lam[None, :]}
    ops = {"Ly": np.int32(fov_height), "Lx": np.int32(fov_width)}
    if mean_img is not None:
        ops["meanImg"] = np.asarray(mean_img, dtype=np.float32)
    # iscell is written for other tools only -- the driver registers every stat entry, so
    # the file must already contain exactly the curation's ROIs.
    return {"stat": stat, "ops": ops, "iscell": np.ones((len(masks), 2), dtype=np.float64)}


def write_fall_mat(path, masks, fov_height, fov_width, mean_img=None,
                   *, allow_overwrite: bool = False) -> dict:
    """Write one column's Fall file. Returns its path, checksum, size and ROI count."""
    p = assert_safe_target(path, allow_overwrite=allow_overwrite)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = fall_payload(masks, fov_height, fov_width, mean_img)
    scipy.io.savemat(str(p), payload, do_compression=True)
    return {"file_path": str(p), "file_name": p.name, "file_sha256": file_sha256(p),
            "file_bytes": p.stat().st_size, "n_masks": int(np.size(payload["stat"])),
            "fov_height": int(fov_height), "fov_width": int(fov_width),
            "file_source": "database"}


def settings_payload(params: dict) -> dict:
    """Build ``settings.mat`` from a CellRegParamSet's parameters.

    Only the 16 fields the driver reads are written. A derived quantity passed in by mistake
    raises, because recording an output as a parameter is how a paramset stops describing
    the run (see the module docstring).
    """
    bad = [k for k in DERIVED_NOT_PARAMETERS if k in params]
    if bad:
        raise ValueError(
            f"{bad} are DERIVED by CellReg, not settings: number_of_bins comes from "
            f"estimate_number_of_bins() and initial_threshold from the fitted model "
            f"(or a 0.65 fallback). Record them in CellRegRun.Quality, not the paramset.")
    unknown = sorted(set(params) - set(SETTINGS_FIELDS))
    if unknown:
        raise ValueError(f"not settings the driver reads: {unknown}; "
                         f"allowed fields are {list(SETTINGS_FIELDS)}")
    out = {**SETTINGS_DEFAULTS, **params}
    return {k: out[k] for k in SETTINGS_FIELDS}


def write_settings_mat(path, params: dict, *, allow_overwrite: bool = False) -> dict:
    p = pathlib.Path(path)
    if p.exists() and not allow_overwrite:
        raise FileExistsError(f"{p} exists; pass allow_overwrite=True to replace it")
    p.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(str(p), settings_payload(params))
    return {"settings_file": str(p), "settings_sha256": file_sha256(p)}


def fall_name(column_index: int, scan_id: str) -> str:
    """``Fall_<NN>_<scan_id>.mat`` -- 1-based NN, matching the names the log echoes."""
    return f"Fall_{column_index + 1:02d}_{scan_id}.mat"


def inspect_existing(export_dir, columns) -> dict:
    """'load' mode: record what is already in the directory, without writing anything."""
    d = pathlib.Path(export_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"task_mode='load' but {d} is not a directory")
    files = []
    for col in columns:
        name = fall_name(int(col["column_index"]), col["scan_id"])
        p = d / name
        if not p.exists():
            raise FileNotFoundError(
                f"task_mode='load' expects {name} in {d} for column "
                f"{col['column_index']} ({col['scan_id']}); it is absent. The declared "
                f"manifest and the directory contents disagree -- fix one of them rather "
                f"than letting the column be resolved by guesswork later.")
        files.append({"column_index": int(col["column_index"]),
                      "session_id": col["session_id"], "scan_id": col["scan_id"],
                      "paramset_idx": col["paramset_idx"], "curation_id": col["curation_id"],
                      "file_path": str(p), "file_name": name,
                      "file_sha256": file_sha256(p), "file_bytes": p.stat().st_size,
                      "n_masks": _stat_count(p), "fov_height": 0, "fov_width": 0,
                      "file_source": "suite2p_copy"})
    settings = d / "settings.mat"
    s = ({"settings_file": str(settings), "settings_sha256": file_sha256(settings)}
         if settings.exists() else {"settings_file": "", "settings_sha256": ""})
    return {"files": files, "note": "loaded from an existing export directory", **s}


def _stat_count(path) -> int:
    """ROI count of an existing Fall file, without loading the footprints densely."""
    try:
        d = scipy.io.loadmat(str(path), variable_names=["stat"], struct_as_record=False,
                             squeeze_me=True)
        return int(np.size(d["stat"]))
    except Exception:
        return 0
