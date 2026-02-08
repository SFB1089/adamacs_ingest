"""
Simplified (non-Element) DataJoint schema for OptiTrack / Motive motion-capture data
==================================================================================
This module is **self-contained** (no Element dependencies) and mirrors the style
you use for *PupilEllipse* tables.

Highlights
----------
* Names are prefixed with your project-level ``db_prefix``.
* Handles Motive 1.2x CSV files that have a 5-row multi-index header **and** a
  metadata line in row 0.
* Robust quaternion → Euler conversion that **keeps NaNs** (no frame is dropped).
* Registers **all** tracking IDs (markers *and* rigid-body names) before inserting.
* Coordinates are from Motive's coordinate system 
  to a standard right-handed system.
"""

from __future__ import annotations

# ───────────────────────────── imports ──────────────────────────────
import datajoint as dj
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------
# Pipeline-level imports (adapt the path to your repo if necessary)
# ---------------------------------------------------------------------
from ..pipeline import session, scan, equipment, db_prefix

schema = dj.schema(db_prefix + "mocap")  # e.g. "roselab_mocap"

# ════════════════════════════════════════════════════════════════════
#                              Parsers
# ════════════════════════════════════════════════════════════════════
def _detect_motive_version(csv_file: str | Path) -> tuple[str, list[int]]:
    """
    Return (format_version, header_levels) for a Motive CSV based on row 0.
    Raises if version is not 1.24 or 1.25.
    """
    with open(csv_file, "r", encoding="utf-8", errors="ignore") as fh:
        first_line = fh.readline()
    tokens = [t.strip() for t in first_line.split(",")]
    try:
        version_idx = tokens.index("Format Version") + 1
        format_version = tokens[version_idx]
    except (ValueError, IndexError):
        raise ValueError("Cannot determine Motive Format Version from first line")

    if format_version == "1.24":
        header_levels = [0, 1, 2, 3, 4]
    elif format_version == "1.25":
        header_levels = [0, 1, 2, 3, 4, 5]
    else:
        raise ValueError(f"Unsupported Motive Format Version: {format_version}")
    return format_version, header_levels


def _parse_motive_csv(csv_file: str | Path):
    """
    Parse a Motive CSV.

    Supported exporter versions:
    * 1.24 → 5-row multi-index header
    * 1.25 → 6-row header (extra axis row dropped)

    Returns
    -------
    frame_idx      : np.ndarray  (N,) 
    timestamps     : np.ndarray  (N,)
    markers        : dict[id -> {'x','y','z'} → np.ndarray]
    marker_uids    : dict[name -> uid]
    rigid_bodies   : dict[rb -> {'pos','quat','euler'}]
    rb_uids        : dict[name -> uid]
        * pos  → {'x','y','z'} each (N,)
        * quat → {'w','x','y','z'} each (N,)
        * euler→ {'roll','pitch','yaw'} each (N,)
    """
    format_version, header_levels = _detect_motive_version(csv_file)

    df = pd.read_csv(
        csv_file,
        skiprows=2,
        header=header_levels,
        low_memory=False,
    )

    # Motive 1.25 adds an extra header row; drop it so we always work with 5 levels.
    if format_version == "1.25":
        df.columns = df.columns.droplevel(-3) # drop the extra axis level - 3rd before frame etc.

    # ---- helper to locate housekeeping columns across header levels
    def _first_col(label: str):
        return next(c for c in df.columns if any(str(l) == label for l in c))

    frame_idx  = df[_first_col("Frame")].to_numpy(int,  copy=True)
    timestamps = df[_first_col("Time (Seconds)")].to_numpy(float, copy=True)

    markers: dict[str, dict[str, np.ndarray]] = {}
    marker_uids: dict[str, str] = {}  # name -> uid
    rigid_bodies: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    rb_uids: dict[str, str] = {}  # name -> uid

    for col in df.columns:
        ctype, name, _id, meas, axis = col
        axis = axis.lower().strip()

        # skip housekeeping
        if any(str(l) in ("Frame", "Time (Seconds)") for l in col):
            continue

        # ---------- markers (incl. rigid-body markers & unlabeled) ----------
        if ctype.lower().startswith(("marker", "rigid body marker", "unlabeled")):
            if meas != "Position":
                continue
            markers.setdefault(name, {"x": None, "y": None, "z": None})
                        
            markers[name][axis] = df[col].to_numpy(float, copy=True) 
                
            # Store the marker's unique ID if not already stored
            if name not in marker_uids and _id:
                marker_uids[name] = str(_id)
            continue

        # ---------- rigid bodies -------------------------------------------
        if ctype == "Rigid Body":
            rb = name
            rigid_bodies.setdefault(
                rb,
                {
                    "pos":  {a: np.full(len(df), np.nan) for a in "xyz"},
                    "quat": {q: np.full(len(df), np.nan) for q in "wxyz"},
                    "euler": None,
                },
            )
            # Store the rigid body's unique ID if not already stored
            if rb not in rb_uids and _id:
                rb_uids[rb] = str(_id)
            if meas == "Position":
                rigid_bodies[rb]["pos"][axis] = df[col].to_numpy(float, copy=True)
            elif meas == "Rotation":
                rigid_bodies[rb]["quat"][axis] = df[col].to_numpy(float, copy=True)

    # ---------- quaternion → Euler (xyz), keep NaNs intact ---------------
    for rb, dat in rigid_bodies.items():
        q = np.column_stack([dat["quat"][k] for k in "xyzw"])  # x y z w order
        eul = np.full((len(q), 3), np.nan)

        complete = ~np.isnan(q).any(axis=1)        # rows with full quaternion
        if complete.any():
            eul[complete] = R.from_quat(q[complete]).as_euler("yxz", degrees=False)

        dat["euler"] = {
            "yaw":  eul[:, 0], # (rotation about Y)
            "pitch": eul[:, 1], # (rotation about X)
            "roll":   eul[:, 2], # (rotation about Z)
        }

    return frame_idx, timestamps, markers, marker_uids, rigid_bodies, rb_uids


def _read_motive_metadata(csv_file: str | Path) -> dict:
    """
    Row-0 of Motive CSV is a flat key/value list, e.g.:
    ``Format Version,1.24,Take Name,ROS-...,Capture Frame Rate,240,...``
    This turns successive pairs into a dict, promoting numbers where possible.
    """
    with open(csv_file, "r", encoding="utf-8", errors="ignore") as fh:
        first_line = fh.readline().strip()

    tokens = [t.strip() for t in first_line.split(",")]
    md = {}
    i = 0
    while i < len(tokens):
        key = tokens[i]
        val = tokens[i + 1] if i + 1 < len(tokens) else ""
        # numeric promotion
        try:
            num = float(val)
            val = int(num) if num.is_integer() else num
        except ValueError:
            pass
        md[key] = val
        i += 2
    return md

# ════════════════════════════════════════════════════════════════════
#                                 Tables
# ════════════════════════════════════════════════════════════════════
@schema
class MocapRecording(dj.Manual):
    definition = """
    -> scan.Scan
    ---
    -> equipment.Device
    """

    class File(dj.Part):
        definition = """
        -> master
        file_id   : int
        ---
        file_path : varchar(255)  # path to .tak (CSV is same stem)
        """


@schema
class MocapRecordingInfo(dj.Imported):
    definition = """
    -> MocapRecording
    ---
    nframes          : int
    sampling_rate_hz : float
    duration_s       : float
    metadata         : longblob
    """

    def make(self, key):
        csv_path = Path(
            (MocapRecording.File & key).fetch1("file_path")
        ).with_suffix(".csv")

        meta_dict = _read_motive_metadata(csv_path)

        # Fast path: only read frame/time columns (skip marker data) for speed.
        format_version, header_levels = _detect_motive_version(csv_path)
        data_start = 2 + len(header_levels)  # metadata rows + header rows
        df_ft = pd.read_csv(
            csv_path,
            skiprows=data_start,
            header=None,
            usecols=[0, 1],  # Frame, Time (Seconds)
            names=["frame", "time_seconds"],
            low_memory=False,
        )

        frame_idx = df_ft["frame"].to_numpy(int, copy=True)
        ts        = df_ft["time_seconds"].to_numpy(float, copy=True)
        dt        = np.nanmean(np.diff(ts)) if len(ts) > 1 else np.nan

        self.insert1(
            dict(
                **key,
                nframes=len(frame_idx),
                sampling_rate_hz=(1.0 / dt) if np.isfinite(dt) else np.nan,
                duration_s=(ts[-1] - ts[0]) if len(ts) > 1 else 0,
                metadata=meta_dict,
            )
        )


@schema
class TrackingId(dj.Lookup):
    definition = """
    tracking_id : varchar(64)     # name of marker or rigid body
    ---
    uid  =''       : varchar(64)     # unique ID hash
    description='' : varchar(255)
    """


@schema
class Mocap(dj.Manual):
    definition = """
    mocap_name : varchar(64)
    ---
    description='' : varchar(255)
    """

    class TrackingId(dj.Part):
        definition = """
        -> master
        -> TrackingId
        """


@schema
class MotionCaptureTask(dj.Manual):
    definition = """
    -> MocapRecording
    -> Mocap
    ---
    csv_path='' : varchar(255)  # override if CSV not alongside .tak
    """


@schema
class MotionCapture(dj.Computed):
    definition = """
    -> MotionCaptureTask
    ---
    import_time : datetime
    """

    class TrackingPosition(dj.Part):
        definition = """
        -> master
        -> Mocap.TrackingId
        ---
        frame_index : longblob
        timestamps  : longblob
        x_pos       : longblob
        y_pos       : longblob
        z_pos       : longblob
        """

    class RigidBodyPosition(dj.Part):
        definition = """
        -> master
        -> Mocap.TrackingId
        rigid_body  : varchar(64)
        ---
        frame_index : longblob
        timestamps  : longblob
        x_pos       : longblob
        y_pos       : longblob
        z_pos       : longblob
        qw          : longblob
        qx          : longblob
        qy          : longblob
        qz          : longblob
        roll        : longblob
        pitch       : longblob
        yaw         : longblob
        """

    # -----------------------------------------------------------------
    #                        populate logic
    # -----------------------------------------------------------------
    def make(self, key):
        rec_file = (MocapRecording.File & key).fetch1("file_path")
        rec_stem = Path(rec_file).with_suffix("")
        override = (MotionCaptureTask & key).fetch1("csv_path")
        csv_path = Path(override) if override else rec_stem.with_suffix(".csv")

        frame_idx, ts, markers, marker_uids, rigid_bodies, rb_uids = _parse_motive_csv(csv_path)

        # ---- ensure every ID (markers + rigid-bodies) exists in TrackingId
        all_ids = set(markers) | set(rigid_bodies)
        existing = set(TrackingId.fetch("tracking_id"))
        to_insert = all_ids - existing
        if to_insert:
            TrackingId.insert([{
                "tracking_id": tid,
                "uid": marker_uids.get(tid, rb_uids.get(tid, tid))  # Use tracking_id as fallback if no UID available
            } for tid in to_insert])

        # ---- link IDs to this Mocap (Mocap.TrackingId)
        missing_links = [
            {"mocap_name": key["mocap_name"], "tracking_id": tid}
            for tid in all_ids
            if not (Mocap.TrackingId & {"mocap_name": key["mocap_name"],
                                        "tracking_id": tid})
        ]
        if missing_links:
            Mocap.TrackingId.insert(missing_links)

        # ---- main row
        self.insert1({**key, "import_time": datetime.utcnow()})

        # ---- marker part-table rows
        marker_rows = [
            {
                **key,
                "tracking_id": tid,
                "frame_index": frame_idx,
                "timestamps":  ts,
                "x_pos": dat["x"],
                "y_pos": dat["y"],  
                "z_pos": dat["z"],  
            }
            for tid, dat in markers.items()
        ]
        self.TrackingPosition.insert(marker_rows)

        # ---- rigid-body part-table rows
        rb_rows = []
        for rb, dat in rigid_bodies.items():
            rb_rows.append(
                {
                    **key,
                    "tracking_id": rb,        # FK column
                    "rigid_body":  rb,
                    "frame_index": frame_idx,
                    "timestamps":  ts,
                    "x_pos": dat["pos"]["x"],
                    "y_pos": dat["pos"]["y"],  
                    "z_pos": dat["pos"]["z"], 
                    "qw": dat["quat"]["w"],
                    "qx": dat["quat"]["x"],
                    "qy": dat["quat"]["y"],    
                    "qz": dat["quat"]["z"],    
                    "roll":  dat["euler"]["roll"],
                    "pitch": dat["euler"]["pitch"], 
                    "yaw":   dat["euler"]["yaw"],    
                }
            )
        self.RigidBodyPosition.insert(rb_rows)

    # -----------------------------------------------------------------
    #                      convenience accessor
    # -----------------------------------------------------------------
    @classmethod
    def get_marker_xyz(cls, key: dict, marker: str):
        """
        Fetch timestamped XYZ for a single marker.
        Returns (timestamps, x, y, z) arrays.
        """
        return (
            cls.TrackingPosition
            & key
            & {"tracking_id": marker}
        ).fetch1("timestamps", "x_pos", "y_pos", "z_pos")
