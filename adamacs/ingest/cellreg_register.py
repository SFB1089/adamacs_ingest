"""Register EXISTING CellReg result files into the cell_registration schema.

This is run 1 of the ingest: parse what is already on disk and record it, in the
chronological order the registrations were produced. It writes ONLY to the
cell_registration schema. It never writes ``roselab_imaging`` -- in particular it never
writes ``Segmentation.Mask.roi_group`` -- and it never populates ``RoiGroupProjection``
or ``RoiGroupSnapshot``.

Curation attachment
-------------------
``CellRegRun.Session`` is keyed on ``imaging.Curation``, so every resolved column needs a
curation. Leon's exports cannot supply one: they copied
``<ScanPath>/suite2p/plane0/Fall.mat``, a per-scan filesystem path with no curation in the
lookup, and the log records only the scan id. Nor does mask-set overlap discriminate --
of 743 (session, scan, paramset) groups holding more than one curation, 696 (94%) have
byte-identical mask sets, so Jaccard scores 1.0 against every candidate.

What *does* discriminate is the output itself. A column of ``cell_to_index_map`` implies a
roi_group vector for its session's masks (ADR-R6), and that vector can be compared
element-for-element against what a candidate curation actually holds. An exact match over
every mask identifies, by content, the curation this run's output was written into. That
is a recorded fact, not an inference, and it is how the ROS-1985 cross-check matched 28 of
28 populated columns. It is stored as ``resolution_method='roi_group_match'``.

Anything short of an exact and UNIQUE match goes to ``CellRegRun.UnresolvedColumn`` with
its candidates attached as evidence. Two curations that are byte-identical AND carry the
same roi_group both match exactly; that is genuine indistinguishability, so the column is
left unresolved rather than attached to an arbitrary one of them.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import socket
from collections import defaultdict

import numpy as np

from . import cellreg as CR

PARSER_VERSION = "1"

# Geometry lines the main parser does not cover. Recorded verbatim because
# microns_per_pixel is 615.6/513 = 1.2, not 615.6/512: the driver pads to ops.Lx+1
# and computes the FOV from the padded size (cellreg_run.m:130-131).
_GEOMETRY = {
    "image_size": r"Image size\s*-\s*(\d+)\s*[Xx]\s*(\d+)",
    "fov_size": r"FOV size\s*-\s*([\d.]+)\s*[Xx]\s*([\d.]+)",
}


def parse_log_geometry(text: str) -> dict:
    """Image and FOV size from a logFile, plus the microns_per_pixel they imply."""
    out = {}
    m = re.search(_GEOMETRY["image_size"], text)
    if m:
        out["image_width_px"], out["image_height_px"] = int(m.group(1)), int(m.group(2))
    m = re.search(_GEOMETRY["fov_size"], text)
    if m:
        out["fov_width_um"], out["fov_height_um"] = float(m.group(1)), float(m.group(2))
    if "fov_width_um" in out and out.get("image_width_px"):
        out["microns_per_pixel"] = out["fov_width_um"] / out["image_width_px"]
    return out


# ------------------------------------------------------------------ curation candidates
def curation_census(conn, imaging_schema="roselab_imaging") -> dict:
    """``{scan_id: [(session_id, scan_id, paramset_idx, curation_id, curation_time)]}``."""
    census = defaultdict(list)
    for row in conn.query(
            f"SELECT session_id, scan_id, paramset_idx, curation_id, curation_time "
            f"FROM `{imaging_schema}`.curation ORDER BY curation_time").fetchall():
        census[row[1]].append((row[0], row[1], int(row[2]), int(row[3]), row[4]))
    return dict(census)


def production_roi_group(conn, key, imaging_schema="roselab_imaging"):
    """The roi_group vector a curation actually holds, indexed by ``mask``.

    Unset positions carry -2 so that a mask absent from the curation can never compare
    equal to a mask a run left unclaimed (-1).
    """
    rows = conn.query(
        f"SELECT mask, roi_group FROM `{imaging_schema}`.`__segmentation__mask` "
        f"WHERE session_id=%s AND scan_id=%s AND paramset_idx=%s AND curation_id=%s "
        f"ORDER BY mask", (key[0], key[1], key[2], key[3])).fetchall()
    if not rows:
        return None
    vec = np.full(max(int(m) for m, _ in rows) + 1, -2, dtype=np.int64)
    for mask, rg in rows:
        vec[int(mask)] = int(rg)
    return vec


def implied_roi_group(index_map, column: int):
    """The roi_group vector one column of ``cell_to_index_map`` implies.

    ``index_map[cell, column]`` is a 1-based mask id, 0 when the cell is absent from that
    session. The implied value at that mask is the cell's 0-based row index; masks no cell
    claims are -1. This is the convention verified against production in the ROS-1985
    cross-check (28 of 28 populated columns exact).
    """
    col = np.asarray(index_map)[:, column]
    present = np.nonzero(col > 0)[0]
    if present.size == 0:
        return None
    masks = col[present].astype(np.int64) - 1
    vec = np.full(int(masks.max()) + 1, -1, dtype=np.int64)
    vec[masks] = present
    return vec


def centroid_counts(path):
    """Per-column centroid counts from ``centroid_locations_corrected``.

    ``centroid_locations_corrected`` is a per-session cell array whose entry for a column
    has one row per mask in that session's own order. Its length must equal the
    curation's mask count; ``make_mask_coordinates`` refuses the column otherwise.
    Checking it here as well means a contradicted attachment is never written at all.

    Returns a list with one count per column, or None for a column whose entry cannot be
    interpreted -- callers skip the check for those rather than reject the column on the
    strength of a read failure. An earlier version wrapped the whole function in
    ``except Exception: return None``, which silently disabled the check for every
    column of every run: the entries are per-session and one unreadable entry must not
    void the other ten.

    This check catches 8 of the 573 columns registered in run 1, every one attached by
    'log_file' -- the FORCED case where the scan has exactly one curation, so no content
    evidence was required. Two shapes of fault: off-by-one (1034 vs 1035, 1280 vs 1279,
    1115 vs 1116), where the curation gained or lost an ROI after the run; and wholesale
    disagreement (316 vs 910, 370 vs 1088, 336 vs 1064), where the input file is a
    different segmentation entirely. One candidate is not evidence that it is the right
    one.
    """
    struct = CR.load_cellreg_struct(path, fields=("centroid_locations_corrected",))
    per_session = struct.get("centroid_locations_corrected")
    if per_session is None:
        return None
    counts = []
    for entry in per_session:
        if entry is None:
            counts.append(None)
            continue
        arr = np.squeeze(np.asarray(entry, dtype=float))
        if arr.ndim != 2:
            counts.append(None)
            continue
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            arr = arr.T          # (2, n) -> (n, 2); a square n == 2 is absurd
        counts.append(int(arr.shape[0]))
    return counts


def resolve_column_curation(conn, implied, candidates, *, imaging_schema="roselab_imaging"):
    """Compare ``implied`` against every candidate curation's stored roi_group.

    Returns ``(resolved_key, method, evidence)``. ``resolved_key`` is None unless exactly
    one candidate matches on every element.
    """
    evidence = []
    exact = []
    for key in candidates:
        prod = production_roi_group(conn, key, imaging_schema)
        if prod is None:
            evidence.append({"key": key[:4], "agree": None, "exact": False,
                             "reason": "curation has no masks"})
            continue
        if len(prod) != len(implied):
            n = min(len(prod), len(implied))
            evidence.append({"key": key[:4], "exact": False,
                             "agree": float((prod[:n] == implied[:n]).mean()) if n else 0.0,
                             "len_prod": len(prod), "len_run": len(implied)})
            continue
        is_exact = bool((prod == implied).all())
        evidence.append({"key": key[:4], "exact": is_exact,
                         "agree": float((prod == implied).mean()),
                         "len_prod": len(prod), "len_run": len(implied)})
        if is_exact:
            exact.append(key)
    evidence.sort(key=lambda e: (-bool(e.get("exact")), -(e.get("agree") or 0.0)))
    if len(exact) == 1:
        return exact[0], "roi_group_match", evidence
    return None, ("ambiguous_roi_group_match" if len(exact) > 1 else "no_match"), evidence


# ------------------------------------------------------------------- parameter identity
def paramset_for(parsed: dict) -> tuple[str, str, dict]:
    """``(slug, matching_method, params)`` for one run's log-derived parameters.

    Parameters only -- accuracy statistics belong to CellRegRun.Quality, not to the
    identity of a parameter set. The slug is readable; content equality is enforced by
    _HashedConfig, so two runs with the same parameters share a slug and a run with
    different parameters under the same slug raises rather than silently updating.
    """
    approach = CR.normalise_registration_approach(parsed.get("registration_approach"))
    alignment = CR.normalise_alignment_mode(parsed.get("alignment_type"))
    method = ("cellreg_probabilistic" if approach == "probabilistic"
              else "cellreg_simple_threshold")
    params = {
        "registration_approach": approach,
        "alignment_mode": alignment,
        "model_type": parsed.get("model_type"),
        "initial_registration_type": parsed.get("initial_registration_type"),
        "initial_threshold": parsed.get("initial_threshold"),
        "final_threshold": parsed.get("final_threshold"),
        "maximal_distance_um": parsed.get("maximal_distance_um"),
        "n_bins": parsed.get("n_bins"),
    }
    # The slug must be unique BY CONTENT. A readable stem alone is not: 86 of 101 existing
    # runs share approach, alignment and p_same threshold while differing in
    # maximal_distance_um, n_bins, initial_threshold or model_type, so a stem-only slug
    # makes _HashedConfig refuse the second run -- correctly, since re-registering a slug
    # with different parameters must never silently update. Appending the first 8 hex of
    # the content hash keeps the slug readable AND one-to-one with the parameters.
    thr = params["final_threshold"]
    stem = ("prob" if approach == "probabilistic" else
            "simple" if approach == "simple_threshold" else "unk")
    align = {"translations_and_rotations": "transrot", "translations": "trans",
             "non_rigid": "nonrigid"}.get(alignment, "unk")
    stem = f"{stem}_{align}" + (f"_p{thr:g}" if thr is not None else "")
    slug = f"{stem}_{_config_hash(params)[:8]}"[:40]
    return slug, method, params


def _config_hash(params: dict) -> str:
    """The schema's own content hash, imported lazily to avoid an import cycle."""
    from ..schemas.cell_registration import config_hash
    return config_hash(params)


# --------------------------------------------------------------------------- the driver
def register_run(mod, record: dict, conn, *, dry_run=True, imaging_schema="roselab_imaging",
                 census=None, adamacs_sha="", note=""):
    """Register ONE existing result file. Returns the plan (dry_run) or the inserted keys.

    Writes only to ``mod``'s schema. Reads ``imaging_schema`` to resolve curations.
    """
    path = record["canonical"]
    census = census if census is not None else curation_census(conn, imaging_schema)
    log_path = record.get("log_file") or ""
    parsed, geometry = {}, {}
    if log_path and os.path.exists(log_path):
        with open(log_path, errors="replace") as fh:
            text = fh.read()
        parsed = CR.parse_log_file(text)
        geometry = parse_log_geometry(text)

    raw_map = CR.load_cellreg_struct(path, fields=("cell_to_index_map",))["cell_to_index_map"]
    try:
        index_map = CR.orient_index_map(raw_map, n_sessions_hint=parsed.get("n_sessions"))
    except ValueError as exc:
        # The log's "Number of sessions" can disagree with the map: a run launched over N
        # sessions whose map has a different column count (see wiki issue D3 -- columns
        # that contributed nothing are still counted in the header). orient_index_map is
        # deliberately strict, so fall back to the shape heuristic and record that the
        # hint was unusable rather than refusing the whole run.
        index_map = CR.orient_index_map(raw_map)
        note = ((note + " | ") if note else "") + (
            f"log n_sessions={parsed.get('n_sessions')} matches neither axis of "
            f"{tuple(np.shape(raw_map))}; orientation taken from shape ({exc})")[:400]
    n_cells, n_cols = index_map.shape

    scans = [CR.scan_id_from_source_file(os.path.basename(p))
             for p in parsed.get("session_files", [])]

    centroids = centroid_counts(path)
    sessions, unresolved = [], []
    for col in range(n_cols):
        scan = scans[col] if col < len(scans) else None
        implied = implied_roi_group(index_map, col)
        n_masks_col = int((index_map[:, col] > 0).sum())
        if implied is None:
            unresolved.append({"column_index": col, "n_masks_cellreg": 0,
                               "best_jaccard": None, "ambiguous": False, "candidates": None,
                               "reason": "column contains no cells"})
            continue
        candidates = census.get(scan, []) if scan else []
        if not scan:
            unresolved.append({"column_index": col, "n_masks_cellreg": n_masks_col,
                               "best_jaccard": None, "ambiguous": False, "candidates": None,
                               "reason": "no scan id recoverable from the log file"})
            continue
        if not candidates:
            unresolved.append({"column_index": col, "n_masks_cellreg": n_masks_col,
                               "best_jaccard": None, "ambiguous": False,
                               "candidates": [scan],
                               "reason": f"scan {scan} has no curation in the database"})
            continue
        if len(candidates) == 1:
            key, method, evidence = candidates[0], "log_file", []
        else:
            key, method, evidence = resolve_column_curation(
                conn, implied, candidates, imaging_schema=imaging_schema)
        if key is None:
            best = evidence[0] if evidence else {}
            unresolved.append({
                "column_index": col, "n_masks_cellreg": n_masks_col,
                "best_jaccard": best.get("agree"),
                "ambiguous": method == "ambiguous_roi_group_match",
                "candidates": evidence[:6],
                "reason": (f"{len([e for e in evidence if e.get('exact')])} curations of "
                           f"{scan} match roi_group exactly" if method ==
                           "ambiguous_roi_group_match" else
                           f"no curation of {scan} matches roi_group "
                           f"(best {best.get('agree')})")})
            continue
        if (centroids is not None and col < len(centroids) and centroids[col] is not None):
            n_cent = centroids[col]
            n_masks = conn.query(
                f"SELECT COUNT(*) FROM `{imaging_schema}`.`__segmentation__mask` "
                f"WHERE session_id=%s AND scan_id=%s AND paramset_idx=%s "
                f"AND curation_id=%s", key[:4]).fetchone()[0]
            if n_cent != int(n_masks):
                unresolved.append({
                    "column_index": col, "n_masks_cellreg": n_masks_col,
                    "best_jaccard": None, "ambiguous": False,
                    "candidates": [{"scan_id": key[1],
                                    "curation_id": key[3],
                                    "n_centroids_file": n_cent,
                                    "n_masks_db": int(n_masks)}],
                    "reason": (f"file has {n_cent} centroids for this column but "
                               f"curation {key[3]} of {key[1]} has {int(n_masks)} "
                               f"masks -- the input file is not this curation")[:255]})
                continue
        n_db = conn.query(
            f"SELECT COUNT(*) FROM `{imaging_schema}`.`__segmentation__mask` WHERE "
            f"session_id=%s AND scan_id=%s AND paramset_idx=%s AND curation_id=%s",
            key[:4]).fetchone()[0]
        agree = next((e.get("agree") for e in evidence if tuple(e["key"]) == tuple(key[:4])),
                     None)
        sessions.append({"session_id": key[0], "scan_id": key[1], "paramset_idx": key[2],
                         "curation_id": key[3], "column_index": col,
                         "source_file": (os.path.basename(parsed["session_files"][col])
                                         if col < len(parsed.get("session_files", []))
                                         else ""),
                         "resolution_method": method, "mask_set_jaccard": agree,
                         "n_masks_cellreg": n_masks_col, "n_masks_db": int(n_db)})

    slug, method_name, params = paramset_for(parsed)
    run_key = {"cellreg_run_id": record["run_id"]}
    master = dict(run_key,
                  cellreg_paramset_id=slug,
                  subject=(record.get("subject", "") or _subject_of(conn, sessions)
                           or _subject_from_folder(record.get("folder", ""))),
                  result_file=path, result_file_sha256=record["sha256"],
                  result_file_bytes=int(record["bytes"]),
                  result_file_mtime=record["mtime"].replace("T", " ")[:19],
                  log_file=log_path, log_file_sha256=CR.file_sha256(log_path)
                  if log_path and os.path.exists(log_path) else "",
                  mat_format=record["mat_format"],
                  registration_approach=CR.normalise_registration_approach(
                      parsed.get("registration_approach")),
                  alignment_mode=CR.normalise_alignment_mode(parsed.get("alignment_type")),
                  reference_session_index=parsed.get("reference_session_index"),
                  n_sessions=int(n_cols), n_registered_cells=int(n_cells),
                  ingested_at=_dt.datetime.now(), ingest_host=socket.gethostname(),
                  adamacs_sha=adamacs_sha, run_note=note)
    quality = dict(run_key, quality_source="log_file" if parsed else "absent",
                   **{k: parsed.get(k) for k in
                      ("frac_false_positives", "frac_false_negatives",
                       "frac_uncertain_pairs", "avg_register_score",
                       "avg_true_positive_score", "avg_true_negative_score",
                       "avg_exclusivity_score", "model_discrepancy",
                       "avg_fraction_active", "initial_threshold", "final_threshold",
                       "maximal_distance_um", "n_bins")},
                   **geometry)
    copies = [dict(run_key, copy_path=p,
                   copy_mtime=_dt.datetime.fromtimestamp(os.stat(p).st_mtime)
                   .strftime("%Y-%m-%d %H:%M:%S"))
              for p in record.get("copies", []) if os.path.exists(p)]

    plan = {"run_id": record["run_id"], "paramset": (slug, method_name, params),
            "master": master, "quality": quality, "copies": copies,
            "sessions": sessions, "unresolved": unresolved,
            "n_columns": int(n_cols), "n_resolved": len(sessions),
            "n_unresolved": len(unresolved)}
    if dry_run:
        return plan

    mod.MatchingMethod()  # touch: the lookup must exist before a paramset references it
    mod.CellRegParamSet.register(slug, params, label=f"recovered from {os.path.basename(log_path) or 'result file'}",
                                 matching_method=method_name)
    with conn.transaction:
        mod.CellRegRun.insert1(master, skip_duplicates=True)
        mod.CellRegRun.Quality.insert1(quality, skip_duplicates=True)
        if copies:
            mod.CellRegRun.Copy.insert(copies, skip_duplicates=True)
        if sessions:
            mod.CellRegRun.Session.insert([dict(run_key, **s) for s in sessions],
                                          skip_duplicates=True)
        if unresolved:
            mod.CellRegRun.UnresolvedColumn.insert(
                [dict(run_key, **u) for u in unresolved], skip_duplicates=True)
    return plan


def _subject_of(conn, sessions, session_schema="roselab_session"):
    """Subject from the first resolved session; '' when nothing resolved."""
    if not sessions:
        return ""
    row = conn.query(f"SELECT subject FROM `{session_schema}`.session WHERE session_id=%s",
                     (sessions[0]["session_id"],)).fetchone()
    return row[0] if row else ""


def _subject_from_folder(folder: str) -> str:
    """Last-resort subject: the ROS-#### in a cellreg folder name, else ''.

    Used only when no column resolved, so roselab_session cannot supply one. The folder
    name is a human convention, not provenance -- which is why it is the last resort.
    """
    m = re.search(r"(ROS-\d+)", folder or "")
    return m.group(1) if m else ""
