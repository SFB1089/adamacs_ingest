"""Facemap schema: face-camera motion and movie SVDs, versioned per facemap run, aligned to the DAQ clock.

DRAFT for review (docs/FACEMAP_SCHEMA_PROPOSAL.md). Nothing is declared on import; the make() bodies
are written after the design is approved.

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
not used as is, for three reasons:
* its FacemapTask belongs to ONE VideoRecording of one session, so a basis shared across sessions (Leon's
  site runs) cannot be expressed; here a task is an ordered manifest of videos (FacemapTask.Video);
* it declares its own session-keyed VideoRecording; here the lab's scan-keyed model.VideoRecordingNew is
  referenced, so DLC, pupil and facemap stay on one video identity;
* it stores one row per component with mask and trace inline (500 x 2 per region, ~3.5 GB per run);
  here masks and traces are arrays in the external store, limited to the first N_STORED components.

Design decisions (as in cell_registration.py)
---------------------------------------------
* **A run is the unit, not a video.** Every run (site run, per-video run, future runs) is its own
  FacemapTask / FacemapProcessing row and is addressable on its own: by facemap_task_id, by its paramset
  (video_assembly 'site_concat_registered' or 'per_video') or by a video it contains. A video can be in
  any number of runs; nothing is keyed by the video alone, so nothing is overwritten.
* **Runs are content-addressed.** The run file is identified by its sha256; the same file under other
  paths is one run with .Copy rows. Leon's per-scan split files are verified slices of the site run
  (FacemapProcessing.Segment), not separate runs.
* **Deterministic ids**, so parallel planners and workers converge: facemap_task_id = hash of the paramset
  and the ordered manifest. Imported / Computed tables populate with reserve_jobs; the 5-11 GB reads happen
  outside the insert transaction (three-part make), so they never hold locks other workers wait on.
* **Clock alignment is explicit and versioned** (FrameSync, SyncMethod): stored per-frame camera events,
  the gate rule, the start-anchored rule for recordings with an unusual count of unsaved triggers, and
  Leon's original method for reproduction only.
* **Thresholds and cohort choices live downstream.** No 'good run' flag here.
* **Nothing is declared on import**: activate() must be called (as cell_registration, disk, element_*).
"""
from __future__ import annotations

import hashlib
import importlib
import json

import datajoint as dj

schema = dj.Schema()
_linking_module = None

PARSER_VERSION = "0.1.0-draft"
N_STORED = 100              # temporal and spatial components kept in the database per SVD (decision 2026-11-26)
UNSAVED_TRIGGERS = 121      # camera triggers after main_track_gate opens before Bonsai saves (v1-drift-audit t43/t44)


def activate(schema_name: str, *, create_schema: bool = True, create_tables: bool = True,
             linking_module: str = "adamacs.pipeline"):
    """Activate the facemap schema. ``linking_module`` supplies ``model`` (VideoRecordingNew,
    RecordingInfoNew) and ``event`` (Event: camera triggers, gate, per-frame camera events)."""
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


def canonical_json(params: dict) -> str:
    """Stable JSON for hashing: sorted keys, no insignificant whitespace."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


def params_hash(params: dict) -> str:
    return hashlib.sha1(canonical_json(params).encode()).hexdigest()


def task_id(paramset: str, recordings) -> str:
    """Deterministic task id: sha1 over the paramset and the ORDERED recording keys, first 12 hex digits.
    Two people (or two workers) planning the same run get the same id."""
    payload = canonical_json({"paramset": paramset,
                              "videos": [[r["session_id"], r["scan_id"], r["recording_id"]] for r in recordings]})
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------------
# Lookups
# ------------------------------------------------------------------------------------------------------

_LEON_FACEMAP = {"package": "github.com/LeonKremers/facemap (fork; version not recorded)", "sbin": 1,
                 "motSVD": True, "movSVD": True, "n_components": 500, "rois": [], "fullSVD": None,
                 "exclusion_mask": {"kind": "eye_ellipse", "source": "DLC eye corners + PupilEllipseFitting",
                                    "long_scale": 5, "short_scale": 5}}
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
                              "concatenated, one facemap run, split per scan; 34 site runs"),
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
                          "Requires event count == video frames"),
        ("gate_rule", "the last n aux_cam triggers strictly inside main_track_gate. Right when about 121 in-gate "
                      "triggers have no frame (230 of 234 audited face videos)"),
        ("start_anchored", "frame i = aux_cam trigger 121 + i after main_track_gate opens (may run past the gate end); "
                           "frames without a trigger get NaN. For recordings whose unsaved count is not ~121 "
                           "(audit: scan9FPZHKKB, scan9FQ1UYJD, scan9FQACAG8, scan9G2HLHF1; rule found post hoc)"),
        ("aux_cam_first_n", "Leon Kremers' original extraction.facemotion_analysis: the first n aux_cam triggers. aux_cam "
                            "starts before the camera saves, so 2.9-6.0 s early. For reproducing old results only"),
    ]


# ------------------------------------------------------------------------------------------------------
# Planning
# ------------------------------------------------------------------------------------------------------

@schema
class FacemapTask(dj.Manual):
    definition = """
    # One PLANNED facemap run over an explicit, ORDERED set of face videos (the manifest is .Video).
    # task_mode: 'load' = the run exists on disk, record it; 'trigger' = a worker runs facemap, then loads it.
    facemap_task_id         : char(12)                     # task_id(paramset, ordered manifest)
    ---
    -> FacemapParamSet
    task_mode='load'        : enum('load','trigger')
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


@schema
class FacialSignal(dj.Imported):
    definition = """
    # Facemap results of one run, per region (element-facemap's FacialSignal, stored as arrays).
    -> FacemapProcessing
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
        exclusion_mask=null     : blob@external-raw        # [ly, lx] bool: pixels blacked out before facemap (Leon: eye ellipse)
        """

    class Basis(dj.Part):
        definition = """
        # The spatial basis of one SVD of one region: temporal components are comparable only within it.
        -> master.Region
        svd                     : enum('motion','movie')
        ---
        singular_values         : longblob                 # [n_components] motSv / movSv (all components)
        n_stored                : smallint                 # spatial components stored (first k, N_STORED)
        masks                   : blob@external-raw        # [ly, lx, n_stored] float16
        """

    class Trace(dj.Part):
        definition = """
        # Per-frame output of one segment and region, in facemap rows (not yet on the DAQ clock).
        -> master.Region
        -> FacemapProcessing.Segment
        ---
        motion_energy           : blob@external-raw        # [n_frames] float32, facemap 'motion'
        n_stored                : smallint                 # temporal components stored (first k, N_STORED)
        mot_svd                 : blob@external-raw        # [n_frames, n_stored] float32, motSVD (unscaled; Sv in Basis)
        mov_svd                 : blob@external-raw        # [n_frames, n_stored] float32, movSVD
        """


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
    n_events                : int                          # events the method drew from (in-gate triggers for gate_rule)
    n_unsaved=null          : int                          # in-gate triggers minus video frames (~121 normally)
    count_status            : enum('exact','events_longer','events_shorter','no_events')
    n_frames_untimed        : int                          # rows without a time (NaN in frame_times)
    frame_times=null        : blob@external-raw            # [n_frames] float64 s, NaN where untimed; NULL if no_events
    first_frame_s=null      : float
    """


# ------------------------------------------------------------------------------------------------------
# Summaries (cross-session comparisons only within one run's basis)
# ------------------------------------------------------------------------------------------------------

@schema
class SegmentSummary(dj.Computed):
    definition = """
    # Where a segment sits in its run's basis (full-frame region): medians and spread of the first k components,
    # motion level, and within-segment references (Leon's temporal / random halves). Comparable within ONE run.
    -> FacialSignal.Trace
    ---
    k_summary               : smallint
    mot_median              : longblob                     # [k]
    mov_median              : longblob                     # [k]
    mot_sd                  : longblob                     # [k]
    mov_sd                  : longblob
    motion_median           : float
    mot_within_temporal     : float                        # distance between the medians of the two temporal halves
    mot_within_random       : float                        # mean over random half-splits
    mov_within_temporal     : float
    mov_within_random       : float
    """


@schema
class SegmentDistance(dj.Computed):
    definition = """
    # Pairwise distances between the segments of one run and region (Leon's svd_distances).
    -> FacialSignal.Region
    ---
    k_summary               : smallint
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


# ------------------------------------------------------------------------------------------------------
# Addressing runs
# ------------------------------------------------------------------------------------------------------

def runs_of(video_restriction, assembly=None):
    """All facemap runs that contain the given face video(s), optionally one generation only
    ('per_video' or 'site_concat_registered'). Each run is a separate row; nothing is merged."""
    runs = FacemapTask * FacemapParamSet & (FacemapTask.Video & video_restriction)
    return runs & {"video_assembly": assembly} if assembly else runs
