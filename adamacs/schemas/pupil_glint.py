"""Corrected corneal glint (IR reflection) for the head-fixed bench2p face camera.

The DLC ``IR`` body part is offset from the glint in the image: on average 2.3 px above (range 1.4–5.0 px) and 1.2 px to
the left, drifting within a session (5-block range 0.1–1.4 px) and between sessions (0.2–3 px within an animal;
v1-drift-audit t31/t31b). The pupil–glint gaze vector inherits that bias. ``CorrectedGlint`` stores a per-frame glint
detected in the image around the DLC marker (``adamacs.helpers.glint_detect``), with the DLC marker minus its local
offset as a flagged fallback. Use it in place of the DLC ``IR`` marker when forming pupil − glint.

The tables live in the ``pupil_tracking`` database schema; this module only declares them (it changes nothing in
``pupil_tracking.py``).
"""
import datajoint as dj
import numpy as np
from element_deeplabcut.model import PoseEstimationNew
from ..pipeline import model, db_prefix
from adamacs.helpers import glint_detect as gd

schema = dj.schema(db_prefix + "pupil_tracking")


@schema
class GlintDetectionParameter(dj.Lookup):
    definition = """
    glint_param_id       : smallint unsigned
    ---
    tophat_kernel_px     : smallint unsigned   # white top-hat kernel diameter
    threshold_nmad       : double              # threshold = max(threshold_min, median + nmad * MAD) of the top-hat
    threshold_min        : double
    search_radius_px     : double              # candidates within this distance of the search centre
    area_min_px          : smallint unsigned
    area_max_px          : smallint unsigned
    fill_min             : double              # component area / bounding-box area
    window_half_px       : smallint unsigned   # search window = 2 * window_half_px + 1
    llh_center           : double              # DLC IR likelihood needed to centre the window on the marker
    anchor_tol_px        : double              # ... and the marker must lie this close to the anchor
    anchor_n             : smallint unsigned   # anchor = median of the last anchor_n accepted detections
    llh_anchor           : double              # initial anchor: median of DLC IR positions at this likelihood
    reject_px            : double              # temporal check: reject detections this far from the rolling median
    reject_win_frames    : smallint unsigned
    offset_win_frames    : int unsigned        # fallback: DLC − rolling median offset over +- this many frames
    llh_fallback         : double              # fallback only where the DLC IR likelihood reaches this
    max_offset_px        : double              # median |DLC − image| above this → not the DLC glint → make() raises
    rel_peak_min         : double              # reject detections below this fraction of the recording's median contrast
    dlc_consistency_px   : double              # reject detections this far from a confident DLC marker minus its local offset
    block_frames         : int unsigned        # block gate: block length (frames)
    block_dev_px         : double              # block median offset must lie within this of the recording median
    max_bad_block_frac   : double              # more frames than this in failing blocks → make() raises
    code_version         : varchar(16)
    param_description="" : varchar(255)
    gate_px=null         : double              # glint-1.3: drop samples this far from the rolling median of image detections, re-detect there; null = off
    """
    contents = [(1, 15, 6.0, 15.0, 14.0, 2, 150, 0.4, 20, 0.9, 15.0, 30, 0.95, 3.0, 31, 1800, 0.5, 8.0, 0.35, 3.0, 18000, 4.0, 0.3, "glint-1.1",
                 "top-hat glint around the DLC IR marker; validated in v1-drift-audit t31b/t31c", None),
                (2, 15, 6.0, 15.0, 14.0, 2, 150, 0.4, 20, 0.9, 15.0, 30, 0.95, 3.0, 31, 1800, 0.5, 8.0, 0.35, 3.0, 18000, 4.0, 0.3, "glint-1.2",
                 "as set 1; DLC-image offsets (block gate, recording offset, local fallback offset) from DLC-confident frames only", None),
                (3, 15, 6.0, 15.0, 14.0, 2, 150, 0.4, 20, 0.9, 15.0, 30, 0.95, 3.0, 31, 1800, 0.5, 8.0, 0.35, 3.0, 18000, 4.0, 0.3, "glint-1.3",
                 "as set 2; samples > gate_px from the rolling median of image detections dropped, re-detected at that reference (source 3)",
                 40.0)]

    @staticmethod
    def as_params(row):
        return dict(k=row["tophat_kernel_px"], nmad=row["threshold_nmad"], thr_min=row["threshold_min"],
                    radius=row["search_radius_px"], amin=row["area_min_px"], amax=row["area_max_px"], fill_min=row["fill_min"],
                    W=row["window_half_px"], llh_center=row["llh_center"], anchor_tol=row["anchor_tol_px"],
                    anchor_n=row["anchor_n"], llh_anchor=row["llh_anchor"], reject_px=row["reject_px"],
                    reject_win=row["reject_win_frames"], offset_win=row["offset_win_frames"], llh_fallback=row["llh_fallback"],
                    max_offset_px=row["max_offset_px"], rel_peak_min=row["rel_peak_min"],
                    dlc_consistency_px=row["dlc_consistency_px"],
                    block_frames=row["block_frames"], block_dev_px=row["block_dev_px"], max_bad_block_frac=row["max_bad_block_frac"],
                    gate_px=None if row.get("gate_px") is None else float(row["gate_px"]))


@schema
class CorrectedGlint(dj.Computed):
    definition = """
    -> GlintDetectionParameter
    -> PoseEstimationNew
    ---
    glint_x               : longblob   # float32 per frame (px, image coordinates); NaN where glint_source == 0
    glint_y               : longblob
    glint_source          : longblob   # uint8 per frame: 1 image detection, 2 DLC IR minus local offset, 3 image re-detection at the reference (glint-1.3), 0 none
    n_frames              : int unsigned   # frames stored = min(video frames read, DLC frames)
    n_dlc_frames          : int unsigned   # DLC frames (differs from n_frames for truncated videos)
    frac_image            : float      # sources 1 and 3
    frac_fallback         : float
    frac_none             : float
    n_rejected_temporal   : int unsigned
    n_rejected_contrast   : int unsigned
    n_rejected_dlc_inconsistent : int unsigned
    n_bad_blocks          : smallint unsigned
    frac_bad_block_frames : float
    dlc_offset_dx         : float      # median DLC IR − image glint over image frames (px)
    dlc_offset_dy         : float
    dlc_offset_block_range=null : float  # max over x/y of the range of the 5 per-block median offsets (px)
    code_version          : varchar(16)
    """

    @property
    def key_source(self):
        face = PoseEstimationNew & 'recording_id LIKE "%%bench2p_face"'
        # only parameter sets of the current code version: populate never recomputes older sets (kept for provenance);
        # make() can still reproduce them (gd.CODE_VERSIONS), which the golden tests use
        return (GlintDetectionParameter & {"code_version": gd.CODE_VERSION}) * (face & (PoseEstimationNew.BodyPartPosition & 'body_part="IR"').proj())

    def make(self, key):
        row = (GlintDetectionParameter & key).fetch1()
        # error messages must stay ASCII: the jobs table's error_message column is latin1, and a non-ASCII character makes
        # DataJoint's error bookkeeping itself fail, which aborts the whole populate (28.9)
        if row["code_version"] not in gd.CODE_VERSIONS:
            raise ValueError(f"parameter set {key['glint_param_id']} is for {row['code_version']}, code implements {gd.CODE_VERSIONS}")
        pose = {k: key[k] for k in PoseEstimationNew.primary_key}
        ir = PoseEstimationNew.BodyPartPosition & pose & 'body_part="IR"'
        if len(ir) != 1:
            raise ValueError(f"{pose}: no DLC IR body part")
        x, y, q = ir.fetch1("x_pos", "y_pos", "likelihood")
        vkey = {k: key[k] for k in ("session_id", "scan_id", "recording_id")}
        video = (model.VideoRecordingNew.File & vkey).fetch1("file_path")
        r = gd.track_video(video, x, y, q, params=GlintDetectionParameter.as_params(row), version=row["code_version"])
        d = r["diagnostics"]
        if d["n_read"] == 0:
            raise ValueError(f"{pose}: no frames read from {video}")
        if not d["offset_plausible"]:
            raise ValueError(f"{pose}: implausible glint: median DLC - image offset ({d['dlc_offset_dx']:.1f}, {d['dlc_offset_dy']:.1f}) px "
                             f"(limit {row['max_offset_px']}), frames in failing blocks {d['frac_bad_block_frames']:.2f} "
                             f"(limit {row['max_bad_block_frac']}); frac_image {d['frac_image']:.2f}")
        self.insert1(dict(key, glint_x=r["x"], glint_y=r["y"], glint_source=r["source"], n_frames=d["n_frames"], n_dlc_frames=d["n_dlc_frames"],
                          frac_image=d["frac_image"], frac_fallback=d["frac_fallback"], frac_none=d["frac_none"],
                          n_rejected_temporal=d["n_rejected_temporal"], n_rejected_contrast=d["n_rejected_contrast"],
                          n_rejected_dlc_inconsistent=d["n_rejected_dlc_inconsistent"],
                          n_bad_blocks=d["n_bad_blocks"], frac_bad_block_frames=d["frac_bad_block_frames"], dlc_offset_dx=d["dlc_offset_dx"],
                          dlc_offset_dy=d["dlc_offset_dy"],
                          dlc_offset_block_range=None if not np.isfinite(d["dlc_offset_block_range"]) else d["dlc_offset_block_range"],
                          code_version=row["code_version"]))
