"""Facemap schema: face-camera motion and movie SVDs, versioned per facemap run, aligned to the DAQ clock.

Design: docs/FACEMAP_SCHEMA_PROPOSAL.md. Nothing is declared on import; call activate().

What it holds
-------------
Facemap (Syeda et al. 2024, Nat Neurosci 27:187) reduces a face video to per-frame motion energy and to
the temporal components of two SVDs: of the frame-to-frame motion (motSVD) and of the movie itself
(movSVD). The spatial components (masks) form a BASIS; temporal components are only comparable between
videos decomposed in the SAME run. Leon Kremers' site runs therefore concatenate the registered face
videos of one imaging site and stimulus type, run facemap once, and split the temporal components back
per scan; his earlier runs decomposed each video on its own.

Relation to DataJoint's element-facemap (github.com/datajoint/element-facemap, 0.3.0)
---------------------------------------------------------------------------------------
Names and concepts follow it where the meaning is the same: FacemapTask -> FacemapProcessing ->
FacialSignal with the parts Region (region 0 = the full frame, others = facemap ROIs) and Summary. It is
not used as is: its FacemapTask belongs to ONE VideoRecording of one session (no basis shared across
sessions); it declares its own session-keyed VideoRecording (here: the lab's scan-keyed
model.VideoRecordingNew); it stores one row per component with mask and trace inline (here: arrays in the
external store, the first N_STORED components).

Design decisions (as in cell_registration.py)
---------------------------------------------
* **A run is the unit, not a video.** Every run (site run, per-video run, future runs) is its own
  FacemapTask / FacemapProcessing row and is addressable on its own: by facemap_task_id, by its paramset
  (video_assembly) or by a video it contains (runs_of). Nothing is keyed by the video alone.
* **Runs are content-addressed**: the run file's sha256 is recorded and re-checked (size, mtime) on reuse.
  Leon's per-scan split files are verified slices of the site run (FacemapProcessing.Segment).
* **Deterministic ids**: facemap_task_id = hash of the paramset and the ordered manifest, so parallel
  planners converge. Imported / Computed tables use the three-part make (make_fetch / make_compute /
  make_insert): the 5-11 GB reads happen outside the insert transaction.
* **Clock alignment is explicit and versioned** (FrameSync x SyncMethod); the choice is downstream.
* **Nothing is declared on import.**
"""
from __future__ import annotations

import datetime as _dt
import glob as _glob
import hashlib
import importlib
import json
import os
import re
import socket
import subprocess

import datajoint as dj
import numpy as np

schema = dj.Schema()
_linking_module = None

PARSER_VERSION = "1.0.0"
N_STORED = 100              # temporal and spatial components kept in the database per SVD (decision 2026-11-26)
UNSAVED_TRIGGERS = 121      # camera triggers after main_track_gate opens before Bonsai saves (v1-drift-audit t43/t44)
N_RANDOM_SPLITS = 20        # random half-splits for the within-segment reference (as roselab_leondrift.SessionFacemap)
LEON_SITE_ROOT = "/datajoint-data/data/leonk/analysis/facemap_registered_videos"
LEON_DATA_ROOT = "/datajoint-data/data/leonk"
FACE_CAMERA = "bench2p_face"


def activate(schema_name: str, *, create_schema: bool = True, create_tables: bool = True,
             linking_module: str = "adamacs.pipeline"):
    """Activate the facemap schema. ``linking_module`` supplies ``model`` (VideoRecordingNew,
    RecordingInfoNew), ``event`` (Event: camera triggers, gate, per-frame camera events), ``scan`` and
    ``session`` (for planning Leon's site runs)."""
    global _linking_module
    module = importlib.import_module(linking_module)
    _linking_module = module
    schema.activate(schema_name, create_schema=create_schema, create_tables=create_tables,
                    add_objects={"model": module.model, "event": module.event, **globals()})
    return schema


def _upstream(name):
    if _linking_module is None:
        raise dj.DataJointError("facemap.activate() has not been called; call activate(schema_name) first.")
    return getattr(_linking_module, name)


# ------------------------------------------------------------------------------------------------------
# Pure helpers (no database; unit-tested in tests/test_facemap.py)
# ------------------------------------------------------------------------------------------------------

def canonical_json(params: dict) -> str:
    """Stable JSON for hashing: sorted keys, no insignificant whitespace."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


def params_hash(params: dict) -> str:
    return hashlib.sha1(canonical_json(params).encode()).hexdigest()


def task_id(paramset: str, recordings, run_identity: str) -> str:
    """Deterministic task id: sha1 over the paramset, the ORDERED recording keys and the run's identity, first 12
    hex digits. Planning the same run twice gives the same id; a re-run gets a new one because its identity differs:
    for 'load' tasks the result file (path, size, mtime: file_identity), for 'trigger' tasks an explicit
    run instance ('instance:<n>')."""
    payload = canonical_json({"paramset": paramset, "run": run_identity,
                              "videos": [[r["session_id"], r["scan_id"], r["recording_id"]] for r in recordings]})
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def file_identity(path: str) -> str:
    """Cheap identity of a result file for planning: path, size and modification time (the sha256 follows in
    FacemapProcessing, which also re-checks size and mtime)."""
    st = os.stat(path)
    return f"file:{path}|{st.st_size}|{int(st.st_mtime)}"


def sha256_file(path: str, chunk: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def segment_offsets(n_frames) -> np.ndarray:
    """First row of each segment in the concatenation (Leon: np.cumsum of the per-video frame counts)."""
    return np.concatenate([[0], np.cumsum(np.asarray(n_frames, int))[:-1]]).astype(int)


def classify_split_motion(n_motion: int, n_segment: int, n_total: int) -> str:
    """'split' (motion holds this segment only), 'full_concat' (May-2026 files: the whole run) or 'absent'."""
    if n_motion == n_segment:
        return "split"
    if n_motion == n_total:
        return "full_concat"
    return "absent"


def dist_refs(A: np.ndarray, seed: int, n_splits: int = N_RANDOM_SPLITS) -> tuple:
    """Within-segment references for a [frames, k] time course (roselab_leondrift.SessionFacemap): Euclidean
    distance between the medians of the two temporal halves, and the mean over random half-splits."""
    n = A.shape[0]
    h = n // 2
    temporal = float(np.linalg.norm(np.median(A[:h], 0) - np.median(A[h:2 * h], 0)))
    rng = np.random.default_rng(seed)
    acc = []
    for _ in range(n_splits):
        p = rng.permutation(n)
        acc.append(np.linalg.norm(np.median(A[p[:h]], 0) - np.median(A[p[h:2 * h]], 0)))
    return temporal, float(np.mean(acc))


def motion_refs(motion: np.ndarray, seed: int, n_splits: int = N_RANDOM_SPLITS) -> tuple:
    """(median, temporal-half reference, random-half reference) of a motion-energy trace."""
    m = np.asarray(motion, float)
    h = len(m) // 2
    rng = np.random.default_rng(seed)
    rnd = [abs(np.median(m[p[:h]]) - np.median(m[p[h:2 * h]])) for p in (rng.permutation(len(m)) for _ in range(n_splits))]
    return float(np.median(m)), float(abs(np.median(m[:h]) - np.median(m[h:2 * h]))), float(np.mean(rnd))


def gate_rule_times(aux: np.ndarray, gate: tuple, n_video: int):
    """Gate rule: video frame i = the last n_video aux_cam triggers strictly inside main_track_gate.
    Returns (times[n_video] or None, n_in_gate, n_unsaved). None when the gate has fewer triggers than frames."""
    in_gate = aux[(aux > gate[0]) & (aux < gate[1])]
    n_unsaved = int(in_gate.size - n_video)
    if n_unsaved < 0:
        return None, int(in_gate.size), n_unsaved
    return in_gate[in_gate.size - n_video:], int(in_gate.size), n_unsaved


def start_anchored_times(aux: np.ndarray, gate_start: float, n_video: int, unsaved: int = UNSAVED_TRIGGERS):
    """Start-anchored rule: video frame i = trigger unsaved + i after the gate opens (may run past the gate end).
    Frames past the last trigger get NaN. Returns (times[n_video], n_triggers_after_gate_start)."""
    after = aux[aux > gate_start]
    t = np.full(n_video, np.nan)
    avail = after[unsaved:unsaved + n_video]
    t[:avail.size] = avail
    return t, int(after.size)


def rows_to_times(video_times: np.ndarray, n_rows: int, video_frame_offset: int = 0) -> np.ndarray:
    """DAQ time of each facemap row: row r = video frame r + video_frame_offset; NaN outside the video."""
    out = np.full(n_rows, np.nan)
    idx = np.arange(n_rows) + int(video_frame_offset)
    ok = (idx >= 0) & (idx < len(video_times))
    out[ok] = np.asarray(video_times, float)[idx[ok]]
    return out


def count_status(n_events: int, n_needed: int) -> str:
    if n_events == 0:
        return "no_events"
    if n_events == n_needed:
        return "exact"
    return "events_longer" if n_events > n_needed else "events_shorter"


def _first(x):
    """Facemap stores per-video entries as one-element lists; return the element."""
    return x[0] if isinstance(x, (list, tuple)) else x


def _masks(proc: dict, kind: str, k: int) -> np.ndarray:
    """[ly, lx, k] float32 spatial components of 'mot' or 'mov' (the *_reshape key, else the flat key reshaped)."""
    if f"{kind}Mask_reshape" in proc:
        m = _first(proc[f"{kind}Mask_reshape"])
    else:
        ly, lx = int(np.ravel(proc["Lybin"])[0]), int(np.ravel(proc["Lxbin"])[0])
        m = _first(proc[f"{kind}Mask"]).reshape(ly, lx, -1)
    return np.ascontiguousarray(m[..., :k]).astype(np.float32)


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), capture_output=True,
                              text=True, timeout=10).stdout.strip()[:40]
    except Exception:
        return ""


# ------------------------------------------------------------------------------------------------------
# Lookups
# ------------------------------------------------------------------------------------------------------

_LEON_FACEMAP = {"package": "github.com/LeonKremers/facemap (fork; version not recorded)", "sbin": 1,
                 "motSVD": True, "movSVD": True, "n_components": 500, "rois": [], "fullSVD": True,
                 "exclusion_mask": {"kind": "eye_ellipse", "source": "DLC eye corners + PupilEllipseFitting",
                                    "long_scale": 5, "short_scale": 5, "applied": "blacked out in the video "
                                    "before facemap; not stored in the proc file"}}
_PARAMSETS = {
    "leon_per_video_2026": ("per_video",
                            {**_LEON_FACEMAP, "assembly": "per_video"},
                            "Leon Kremers, Feb-Apr 2026: facemap on each face video alone (own basis per video); "
                            "<scan>_bonsai_face_video_<ts>_proc.npy, 231 files"),
    "leon_site_concat_2026": ("site_concat_registered",
                              {**_LEON_FACEMAP, "assembly": "site_concat_registered",
                               "registration": "LUT histogram match to video 1; ECC translation on 100-frame "
                                               "averages; crop to the shared region (transforms not saved)",
                               "scan_selection": "define_scans(same_site_id, stim_type, paramset=10, "
                                                 "behavior_stage='baseline'), primary-key order"},
                              "Leon Kremers, May-Nov 2026: registered face videos of one site and stimulus type "
                              "concatenated, one facemap run, split per scan; 32 site runs"),
}


@schema
class FacemapParamSet(dj.Lookup):
    definition = """
    # How a facemap run was (or will be) made. Content-addressed: same params -> same hash. Never edited.
    facemap_paramset        : varchar(32)                  # slug, e.g. 'leon_site_concat_2026'
    ---
    video_assembly          : enum('per_video','site_concat_registered')   # one video per run | registered + concatenated videos
    params                  : longblob                     # the parameter dict
    params_hash             : char(40)                     # sha1 of canonical JSON
    params_json             : varchar(4000)
    paramset_desc=''        : varchar(512)
    unique index (params_hash)
    """
    contents = [(slug, assembly, p, params_hash(p), canonical_json(p), desc)
                for slug, (assembly, p, desc) in _PARAMSETS.items()]


@schema
class SyncMethod(dj.Lookup):
    definition = """
    # How camera frames get DAQ times (frame i of a video of n frames).
    sync_method             : varchar(32)
    ---
    sync_description        : varchar(1024)
    """
    contents = [
        ("camera_frames", "event.Event '<camera>_frames' (e.g. bench2p_face_frames), one row per saved frame; frame i = "
                          "event i. Stored for recordings that passed the camera-timing QC (v1-drift-audit t43/t44). "
                          "Times only when the event count equals the video frames"),
        ("gate_rule", "the last n aux_cam triggers strictly inside main_track_gate. Right when about 121 in-gate "
                      "triggers have no frame (230 of 234 audited face videos)"),
        ("start_anchored", "frame i = aux_cam trigger 121 + i after main_track_gate opens (may run past the gate end); "
                           "frames without a trigger get NaN. For recordings whose unsaved count is not ~121 "
                           "(audit: scan9FPZHKKB, scan9FQ1UYJD, scan9FQACAG8, scan9G2HLHF1; rule found post hoc)"),
        ("aux_cam_first_n", "Leon Kremers' original extraction.facemotion_analysis: facemap row r = aux_cam trigger r. "
                            "aux_cam starts before the camera saves, so 2.9-6.0 s early. For reproducing old results only"),
    ]


# ------------------------------------------------------------------------------------------------------
# Planning
# ------------------------------------------------------------------------------------------------------

@schema
class FacemapTask(dj.Manual):
    definition = """
    # One PLANNED facemap run over an explicit, ORDERED set of face videos (the manifest is .Video).
    # task_mode: 'load' = the run exists on disk, record it; 'trigger' = a worker runs facemap, then loads it.
    facemap_task_id         : char(12)                     # task_id(paramset, ordered manifest, run_identity)
    ---
    -> FacemapParamSet
    task_mode='load'        : enum('load','trigger')
    run_identity            : varchar(640)                 # load: 'file:<path>|<bytes>|<mtime>'; trigger: 'instance:<n>'
    run_dir                 : varchar(512)                 # where the run's files are (trigger: <dir>/facemap/<task_id>/)
    run_file=''             : varchar(512)                 # load: the run's proc.npy (site-level file for site runs)
    same_site_id=''         : varchar(12)                  # site of the videos; '' for per-video runs
    task_label=''           : varchar(64)                  # e.g. 'gratings', 'images' (Leon: one site run per stimulus type)
    task_created            : datetime
    task_created_by=''      : varchar(64)
    task_note=''            : varchar(512)
    """

    class Video(dj.Part):
        definition = """
        # THE MANIFEST: segment k of the run is this face video, in this order.
        -> master
        segment_index           : smallint                 # 0-based position in the concatenation (0 for per-video runs)
        ---
        -> model.VideoRecordingNew
        unique index (facemap_task_id, session_id, scan_id, recording_id)
        """

    @classmethod
    def insert_planned(cls, paramset: str, videos: list, *, run_dir: str, run_file: str = "", task_mode: str = "load",
                       instance: int = 0, same_site_id: str = "", label: str = "", note: str = "",
                       created_by: str = "") -> str:
        """Insert one task with its ordered manifest. Idempotent: the id is the hash of paramset, manifest and run
        identity (the result file for 'load', the instance number for 'trigger'; a deliberate re-run = instance + 1)."""
        ident = file_identity(run_file) if task_mode == "load" else f"instance:{int(instance)}"
        tid = task_id(paramset, videos, ident)
        cls.insert1(dict(facemap_task_id=tid, facemap_paramset=paramset, task_mode=task_mode, run_identity=ident,
                         run_dir=run_dir,
                         run_file=run_file, same_site_id=same_site_id, task_label=label,
                         task_created=_dt.datetime.now().replace(microsecond=0), task_created_by=created_by,
                         task_note=note), skip_duplicates=True)
        cls.Video.insert([dict(facemap_task_id=tid, segment_index=i, session_id=v["session_id"], scan_id=v["scan_id"],
                               recording_id=v["recording_id"]) for i, v in enumerate(videos)], skip_duplicates=True)
        return tid


def _face_recording(scan_key):
    """The face-camera VideoRecordingNew key of one scan, or None."""
    model = _upstream("model")
    rows = (model.VideoRecordingNew & scan_key & {"camera": FACE_CAMERA}).fetch("KEY")
    return rows[0] if len(rows) == 1 else None


def leon_site_scans(same_site_id: str, stim: str):
    """Leon's define_scans(same_site_id, stim, paramset, behavior_stage='baseline'): scan keys in the order his
    fetch returned them (primary-key order). Verified against the run's content at ingest (Segment.split_*)."""
    session, scan, event = _upstream("session"), _upstream("scan"), _upstream("event")
    sess = (session.Session * session.SessionSameSite & {"same_site_id": same_site_id}).fetch("KEY")
    keys = (scan.Scan * event.BehaviorRecording & sess & f'scan_notes LIKE "%%{stim}%%"'
            & 'scan_notes NOT LIKE "%%Do not use!%%"' & 'recording_notes LIKE "%%baseline%%"').fetch(
        "KEY", order_by="session_id, scan_id")
    return [{"session_id": k["session_id"], "scan_id": k["scan_id"]} for k in keys]


def plan_leon_site_runs(insert: bool = False, created_by: str = "facemap ingest"):
    """One 'load' task per site run folder <same_site_id>_<stim> under LEON_SITE_ROOT. Returns the plan rows."""
    plan = []
    for d in sorted(_glob.glob(f"{LEON_SITE_ROOT}/*_*")):
        name = os.path.basename(d)
        m = re.fullmatch(r"(sess\w+)_(gratings|images)", name)
        run_file = f"{d}/unified_registered_video_proc.npy"
        if not m or not os.path.exists(run_file):
            plan.append(dict(folder=name, status="skipped: no site run file"))
            continue
        site, stim = m.groups()
        scans = leon_site_scans(site, stim)
        videos, missing = [], []
        for s in scans:
            r = _face_recording(s)
            (videos.append(r) if r else missing.append(s["scan_id"]))
        row = dict(folder=name, same_site_id=site, stim=stim, n_scans=len(scans), missing_face_video=missing,
                   facemap_task_id=task_id("leon_site_concat_2026", videos, file_identity(run_file)) if not missing else "")
        if missing:
            row["status"] = "skipped: scans without a face recording"
        elif insert:
            FacemapTask.insert_planned("leon_site_concat_2026", videos, run_dir=d, run_file=run_file,
                                       same_site_id=site, label=stim, created_by=created_by,
                                       note="Leon Kremers' site run (notebook 54); manifest = define_scans order")
            row["status"] = "planned"
        else:
            row["status"] = "dry run"
        plan.append(row)
    return plan


def plan_leon_per_video_runs(insert: bool = False, created_by: str = "facemap ingest"):
    """One 'load' task per <scan>_bonsai_face_video_<ts>_proc.npy under LEON_DATA_ROOT/*/facemap/."""
    model = _upstream("model")
    plan = []
    for f in sorted(_glob.glob(f"{LEON_DATA_ROOT}/*/facemap/*_bonsai_face_video_*_proc.npy")):
        scan_id = os.path.basename(f).split("_bonsai_face_video_")[0]
        video_name = os.path.basename(f)[:-len("_proc.npy")] + ".mp4"
        rec = (model.VideoRecordingNew & {"scan_id": scan_id} & {"camera": FACE_CAMERA}).fetch("KEY")
        row = dict(file=f, scan_id=scan_id)
        if len(rec) != 1:
            row["status"] = f"skipped: {len(rec)} face recordings"
            plan.append(row)
            continue
        paths = (model.VideoRecordingNew.File & rec[0]).fetch("file_path")
        if not any(os.path.basename(p) == video_name for p in paths):
            row["status"] = f"skipped: face recording file is not {video_name}"
            plan.append(row)
            continue
        row["facemap_task_id"] = task_id("leon_per_video_2026", rec, file_identity(f))
        if insert:
            FacemapTask.insert_planned("leon_per_video_2026", rec, run_dir=os.path.dirname(f), run_file=f,
                                       created_by=created_by, note="Leon Kremers' per-video run (Feb-Apr 2026)")
            row["status"] = "planned"
        else:
            row["status"] = "dry run"
        plan.append(row)
    return plan


# ------------------------------------------------------------------------------------------------------
# The run
# ------------------------------------------------------------------------------------------------------

@schema
class FacemapProcessing(dj.Imported):
    definition = """
    # One facemap run as found on disk (task_mode 'load') or as written by the worker ('trigger').
    -> FacemapTask
    ---
    run_file                : varchar(512)                 # canonical path; other paths in .Copy
    run_file_sha256         : char(64)
    run_file_bytes          : bigint
    run_file_mtime          : datetime
    source_video=''         : varchar(512)                 # the video facemap read ('filenames' in proc), e.g. unified_registered_video.mp4
    n_frames_total          : int                          # rows of motSVD over all segments
    n_components            : smallint                     # facemap components per SVD (500 in Leon's runs)
    proc_keys               : varchar(1024)                # keys found in the proc dict, comma separated
    parser_version          : varchar(16)
    ingested_at             : datetime
    ingest_host             : varchar(64)
    adamacs_sha=''          : varchar(40)
    """

    class Copy(dj.Part):
        definition = """
        # Another path holding a byte-identical run file.
        -> master
        copy_path               : varchar(512)
        ---
        copy_mtime              : datetime
        """

    class Segment(dj.Part):
        definition = """
        # One video's frames inside the run (Leon: np.cumsum of the per-video frame counts).
        -> master
        -> FacemapTask.Video
        ---
        frame_offset            : int                      # first row of this segment in the run's motSVD
        n_frames                : int                      # facemap rows of this segment
        n_frames_video          : int                      # frames of the video (model.RecordingInfoNew.nframes)
        frames_match            : bool                     # n_frames == n_frames_video
        video_frame_offset=0    : smallint                 # facemap row r = video frame r + this (11 scans of ROS-1912/2356: ~4)
        offset_source='assumed' : enum('assumed','motion_vs_dlc')   # how video_frame_offset was set
        split_file=''           : varchar(512)             # Leon's per-scan copy (<scan>_face_cam_concatenate_proc.npy); '' if none
        split_file_sha256=''    : char(64)
        split_motion='absent'   : enum('split','full_concat','absent')   # 'full_concat': May-2026 files kept the whole motion trace
        split_matches_run=null  : bool                     # the split file's motSVD/movSVD/motion equal the run's slice
        """

    def make_fetch(self, key):
        model = _upstream("model")
        task = (FacemapTask * FacemapParamSet & key).fetch1("facemap_task_id", "task_mode", "run_file", "video_assembly")
        videos = []
        for v in (FacemapTask.Video & key).fetch(as_dict=True, order_by="segment_index"):
            rk = {k: v[k] for k in ("session_id", "scan_id", "recording_id")}
            path = (model.VideoRecordingNew.File & rk).fetch("file_path", order_by="file_id", limit=1)
            nvid = (model.RecordingInfoNew & rk).fetch("nframes")
            videos.append(dict(segment_index=int(v["segment_index"]), **rk, video_path=str(path[0]) if len(path) else "",
                               n_frames_video=int(nvid[0]) if len(nvid) else -1))
        return (tuple(task), videos)

    def make_compute(self, key, task, videos):
        tid, mode, run_file, assembly = task
        if mode != "load":
            raise NotImplementedError("task_mode 'trigger' (running facemap) is not implemented yet")
        if not run_file or not os.path.exists(run_file):
            raise FileNotFoundError(f"run file missing: {run_file!r}")
        st = os.stat(run_file)
        sha = sha256_file(run_file)
        proc = np.load(run_file, allow_pickle=True).item()
        mot, mov = _first(proc["motSVD"]), _first(proc["movSVD"])
        motion = np.asarray(_first(proc["motion"]))
        n_total, n_comp = int(mot.shape[0]), int(mot.shape[1])
        filenames = np.ravel(np.asarray(proc.get("filenames", []), dtype=object)).tolist()
        run_row = dict(key, run_file=run_file, run_file_sha256=sha, run_file_bytes=int(st.st_size),
                       run_file_mtime=_dt.datetime.fromtimestamp(st.st_mtime).replace(microsecond=0),
                       source_video=str(filenames[0]) if filenames else "", n_frames_total=n_total, n_components=n_comp,
                       proc_keys=",".join(sorted(proc))[:1024], parser_version=PARSER_VERSION,
                       ingested_at=_dt.datetime.now().replace(microsecond=0), ingest_host=socket.gethostname()[:64],
                       adamacs_sha=_git_sha())
        if assembly == "per_video" and len(videos) != 1:
            raise ValueError(f"per-video run with {len(videos)} videos")
        rows, off = [], 0
        for v in videos:   # in manifest order, so each segment's offset is known when its split file is read
            row = dict(key, segment_index=v["segment_index"], frame_offset=int(off), n_frames_video=int(v["n_frames_video"]),
                       split_file="", split_file_sha256="", split_motion="absent", split_matches_run=None)
            f = os.path.join(os.path.dirname(v["video_path"]), "facemap", f"{v['scan_id']}_face_cam_concatenate_proc.npy")
            if assembly == "per_video":
                n = n_total
            elif os.path.exists(f):            # Leon's split copy gives the segment length and is checked against the run
                sp = np.load(f, allow_pickle=True).item()
                smot, smov = _first(sp["motSVD"]), _first(sp["movSVD"])
                smotion = np.asarray(_first(sp["motion"]))
                n = int(smot.shape[0])
                kind = classify_split_motion(smotion.size, n, n_total)
                ok = bool(off + n <= n_total and np.array_equal(smot, mot[off:off + n])
                          and np.array_equal(smov, mov[off:off + n]))
                if kind == "split":
                    ok = ok and bool(np.array_equal(smotion, motion[off:off + n]))
                elif kind == "full_concat":
                    ok = ok and bool(np.array_equal(smotion, motion))
                row.update(split_file=f, split_file_sha256=sha256_file(f), split_motion=kind, split_matches_run=ok)
                del sp, smot, smov, smotion
            else:                              # no split copy: the video frame count (Leon split by cv2 frame counts)
                n = int(v["n_frames_video"])
            row.update(n_frames=int(n), frames_match=bool(n == v["n_frames_video"]))
            rows.append(row)
            off += n
        if off != n_total:
            raise ValueError(f"segments sum to {off} rows, the run has {n_total}: manifest or order does not match the run")
        return run_row, rows

    def make_insert(self, key, run_row, rows):
        self.insert1(run_row)
        self.Segment.insert(rows)


@schema
class FacialSignal(dj.Imported):
    definition = """
    # Facemap results of one run, per region (element-facemap's FacialSignal, stored as arrays).
    -> FacemapProcessing
    ---
    n_stored                : smallint                     # components stored per SVD (masks and traces)
    """

    class Summary(dj.Part):
        definition = """
        -> master
        ---
        sbin                    : tinyint                  # spatial bin size
        ly                      : smallint                 # binned frame height
        lx                      : smallint
        avg_frame               : blob@external-raw        # [ly, lx] float32
        avg_motion              : blob@external-raw        # [ly, lx] float32
        """

    class Region(dj.Part):
        definition = """
        # Region 0 = the full frame (Leon's runs have only this); 1.. = facemap ROIs.
        -> master
        roi_no                  : smallint
        ---
        roi_name=''             : varchar(32)
        roi_mask=null           : blob@external-raw        # [ly, lx] bool: pixels of the ROI; NULL = full frame
        exclusion_mask=null     : blob@external-raw        # [ly, lx] bool blacked out before facemap; NULL = not stored (Leon: in the video)
        """

    class Basis(dj.Part):
        definition = """
        # The spatial basis of one SVD of one region: temporal components are comparable only within it.
        -> master.Region
        svd                     : enum('motion','movie')
        ---
        singular_values         : blob@external-raw        # [n_components] float32 motSv / movSv (all components)
        n_stored                : smallint                 # spatial components stored (first k)
        masks                   : blob@external-raw        # [ly, lx, n_stored] float32 (DataJoint blobs have no float16)
        """

    class Trace(dj.Part):
        definition = """
        # Per-frame output of one segment and region, in facemap rows (not yet on the DAQ clock).
        -> master.Region
        -> FacemapProcessing.Segment
        ---
        motion_energy           : blob@external-raw        # [n_frames] float32, facemap 'motion'
        n_stored                : smallint                 # temporal components stored (first k)
        mot_svd                 : blob@external-raw        # [n_frames, n_stored] float32, motSVD (unscaled; Sv in Basis)
        mov_svd                 : blob@external-raw        # [n_frames, n_stored] float32, movSVD
        """

    class Position(dj.Part):
        definition = """
        # Where a segment sits in its run's basis, from ALL components (roselab_leondrift.SessionFacemap measures).
        # Comparable across segments of ONE run only.
        -> master.Region
        -> FacemapProcessing.Segment
        ---
        k_position              : smallint                 # components used (all of the run)
        mot_median              : blob@external-raw        # [k] float64 median over frames of motSVD
        mov_median              : blob@external-raw        # [k]
        mot_sd                  : blob@external-raw        # [k] SD over frames
        mov_sd                  : blob@external-raw
        mot_within_temporal     : float                    # distance between the medians of the two temporal halves
        mot_within_random       : float                    # mean over 20 random half-splits (seed = segment_index)
        mov_within_temporal     : float
        mov_within_random       : float
        motion_median           : float                    # median face motion energy
        motion_within_temporal  : float
        motion_within_random    : float
        """

    def make_fetch(self, key):
        run = (FacemapProcessing & key).fetch1("run_file", "run_file_bytes", "run_file_mtime", "n_frames_total")
        segs = (FacemapProcessing.Segment & key).fetch("segment_index", "frame_offset", "n_frames", as_dict=True,
                                                       order_by="segment_index")
        return (tuple(str(x) for x in run), [dict(s) for s in segs])

    def make_compute(self, key, run, segs):
        run_file, nbytes, mtime, n_total = run
        st = os.stat(run_file)
        if st.st_size != int(nbytes) or _dt.datetime.fromtimestamp(st.st_mtime).replace(microsecond=0) != \
                _dt.datetime.fromisoformat(mtime):
            raise ValueError(f"{run_file} changed since FacemapProcessing recorded it")
        proc = np.load(run_file, allow_pickle=True).item()
        mot, mov = _first(proc["motSVD"]), _first(proc["movSVD"])
        motion = np.asarray(_first(proc["motion"]), np.float32)
        k = min(N_STORED, mot.shape[1])
        avg_f = np.asarray(proc["avgframe_reshape"], np.float32)
        avg_m = np.asarray(proc["avgmotion_reshape"], np.float32)
        region = dict(key, roi_no=0)
        master = dict(key, n_stored=k)
        summary = dict(key, sbin=int(proc.get("sbin", 1)), ly=int(avg_f.shape[0]), lx=int(avg_f.shape[1]),
                       avg_frame=avg_f, avg_motion=avg_m)
        regions = [dict(region, roi_name="full_frame", roi_mask=None, exclusion_mask=None)]
        basis = [dict(region, svd="motion", singular_values=np.asarray(proc["motSv"], np.float32), n_stored=k,
                      masks=_masks(proc, "mot", k)),
                 dict(region, svd="movie", singular_values=np.asarray(proc["movSv"], np.float32), n_stored=k,
                      masks=_masks(proc, "mov", k))]
        traces, positions = [], []
        for s in segs:
            off, n = int(s["frame_offset"]), int(s["n_frames"])
            seg_key = dict(region, segment_index=int(s["segment_index"]))
            a, b = np.asarray(mot[off:off + n], np.float32), np.asarray(mov[off:off + n], np.float32)
            e = motion[off:off + n] if motion.size == int(n_total) else motion
            if e.size != n:
                raise ValueError(f"motion has {motion.size} rows; cannot take segment {s['segment_index']}")
            traces.append(dict(seg_key, motion_energy=e, n_stored=k, mot_svd=np.ascontiguousarray(a[:, :k]),
                               mov_svd=np.ascontiguousarray(b[:, :k])))
            i = int(s["segment_index"])
            af, bf = a.astype(float), b.astype(float)      # float64, as roselab_leondrift.SessionFacemap
            mt, mr = dist_refs(af, i)
            vt, vr = dist_refs(bf, i + 1)
            m_med, m_t, m_r = motion_refs(e, i + 2)
            positions.append(dict(seg_key, k_position=int(a.shape[1]), mot_median=np.median(af, 0), mov_median=np.median(bf, 0),
                                  mot_sd=af.std(0), mov_sd=bf.std(0), mot_within_temporal=mt, mot_within_random=mr,
                                  mov_within_temporal=vt, mov_within_random=vr, motion_median=m_med,
                                  motion_within_temporal=m_t, motion_within_random=m_r))
        return master, summary, regions, basis, traces, positions

    def make_insert(self, key, master, summary, regions, basis, traces, positions):
        self.insert1(master)
        self.Summary.insert1(summary)
        self.Region.insert(regions)
        self.Basis.insert(basis)
        self.Trace.insert(traces)
        self.Position.insert(positions)


# ------------------------------------------------------------------------------------------------------
# Clock
# ------------------------------------------------------------------------------------------------------

@schema
class FrameSync(dj.Computed):
    definition = """
    # DAQ time of every facemap row of one segment, by one SyncMethod.
    -> FacemapProcessing.Segment
    -> SyncMethod
    ---
    camera_event_type       : varchar(64)                  # e.g. 'bench2p_face_frames' or 'aux_cam'
    n_events                : int                          # events the method drew from (see sync_description)
    n_unsaved=null          : int                          # in-gate triggers minus video frames (~121 normally)
    count_status            : enum('exact','events_longer','events_shorter','no_events')
    n_frames_untimed        : int                          # rows without a time (NaN in frame_times)
    frame_times=null        : blob@external-raw            # [n_frames] float64 s, NaN where untimed; NULL if no times
    first_frame_s=null      : float
    """

    def make(self, key):
        event, model = _upstream("event"), _upstream("model")
        seg = (FacemapProcessing.Segment & key).fetch1()
        vid = dict(zip(("session_id", "scan_id", "recording_id"),
                       (FacemapTask.Video & key).fetch1("session_id", "scan_id", "recording_id")))
        sk = {"session_id": vid["session_id"], "scan_id": vid["scan_id"]}      # session_id + scan_id: fast on event.Event
        camera = (model.VideoRecordingNew & vid).fetch1("camera")
        n_rows, n_video, off = int(seg["n_frames"]), int(seg["n_frames_video"]), int(seg["video_frame_offset"])
        method = key["sync_method"]

        def ev(t):
            return np.sort((event.Event & sk & {"event_type": t}).fetch("event_start_time").astype(float))

        def gate():
            g = (event.Event & sk & {"event_type": "main_track_gate"}).fetch(
                "event_start_time", "event_end_time", order_by="event_start_time")
            return (float(g[0][0]), float(g[1][0])) if len(g[0]) else None

        n_unsaved, times = None, None
        if method == "camera_frames":
            etype = f"{camera}_frames"
            e = ev(etype)
            status = count_status(e.size, n_video)
            if status == "exact":
                times = rows_to_times(e, n_rows, off)
            n_events = e.size
        else:
            etype = "aux_cam"
            aux = ev("aux_cam")
            g = gate()
            n_events = aux.size
            if method == "aux_cam_first_n":
                status = count_status(aux.size, n_rows)
                times = rows_to_times(aux[:n_rows], n_rows, 0) if aux.size else None
            elif g is None or aux.size == 0:
                status = "no_events"
            elif method == "gate_rule":
                vt, n_events, n_unsaved = gate_rule_times(aux, g, n_video)
                status = count_status(n_events, n_video)
                times = rows_to_times(vt, n_rows, off) if vt is not None else None
            elif method == "start_anchored":
                vt, n_events = start_anchored_times(aux, g[0], n_video)
                n_unsaved = int(aux[(aux > g[0]) & (aux < g[1])].size - n_video)
                status = count_status(n_events, n_video + UNSAVED_TRIGGERS)
                times = rows_to_times(vt, n_rows, off)
            else:
                raise ValueError(method)
        untimed = n_rows if times is None else int(np.isnan(times).sum())
        first = None if times is None or untimed == n_rows else float(np.nanmin(times))
        self.insert1(dict(key, camera_event_type=etype, n_events=int(n_events), n_unsaved=n_unsaved,
                          count_status=status, n_frames_untimed=untimed, frame_times=times, first_frame_s=first))


# ------------------------------------------------------------------------------------------------------
# Cross-session distances (within one run's basis)
# ------------------------------------------------------------------------------------------------------

@schema
class SegmentDistance(dj.Computed):
    definition = """
    # Pairwise distances between the segments of one run and region (Leon's svd_distances), from ALL components.
    -> FacialSignal.Region
    ---
    n_segments              : smallint
    """

    class Pair(dj.Part):
        definition = """
        -> master
        segment_a               : smallint
        segment_b               : smallint
        ---
        mot_distance            : float                    # Euclidean distance between mot_median of a and b
        mov_distance            : float
        """

    def make(self, key):
        idx, mot, mov = (FacialSignal.Position & key).fetch("segment_index", "mot_median", "mov_median",
                                                            order_by="segment_index")
        self.insert1(dict(key, n_segments=len(idx)))
        pairs = []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                pairs.append(dict(key, segment_a=int(idx[i]), segment_b=int(idx[j]),
                                  mot_distance=float(np.linalg.norm(np.asarray(mot[i]) - np.asarray(mot[j]))),
                                  mov_distance=float(np.linalg.norm(np.asarray(mov[i]) - np.asarray(mov[j])))))
        self.Pair.insert(pairs)


# ------------------------------------------------------------------------------------------------------
# Addressing runs
# ------------------------------------------------------------------------------------------------------

def runs_of(video_restriction, assembly=None):
    """All facemap runs that contain the given face video(s), optionally of one generation only
    ('per_video' or 'site_concat_registered'). Each run is a separate row; nothing is merged."""
    runs = FacemapTask * FacemapParamSet & (FacemapTask.Video & video_restriction)
    return runs & {"video_assembly": assembly} if assembly else runs
