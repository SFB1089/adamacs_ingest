"""Eye-trace resampling and gaze geometry, with no database in it.

This module holds the numerical core shared by the production gaze chain in
``adamacs/schemas/pupil_tracking.py`` and the DISK-imputed sibling chain in
``adamacs/schemas/disk.py``:

    PupilRotationOptiTrack          PupilRotationOptiTrackImputed
    GazeReconstruction3D            GazeReconstruction3DImputed

The two chains differ in exactly one thing -- whether the eye angles come from
``PupilEllipseFittingFreeMoving`` or from ``PupilEllipseFittingImputed``. Everything after
that is identical, and the whole value of the sibling chain is that a difference between a
row and its production twin is the imputation and nothing else. A second copy of the
interpolation or the rotation maths would put that at the mercy of someone remembering to
edit both.

**Why a separate module rather than functions on ``pupil_tracking``.** ``disk.py`` is
reached through ``dj.create_virtual_module`` by ``populate_natasha.py``,
``validate_disk_schema.py`` and ``run_chain.sh``, because importing ``adamacs.pipeline``
executes a module-level ``Lab.insert1`` and declares tables as a side effect of the import.
A virtual module reflects a schema, so it carries that schema's *tables* and nothing else:
``_linking_module.pupil_tracking.compute_gaze_vectors`` exists under ``adamacs.pipeline``
and does not exist under a virtual module. Putting the maths here means both chains reach
it the same way in both worlds.

Nothing in this file touches DataJoint, so it is importable without a database connection
and testable without one. The table lookups each chain needs -- event timestamps, the
eye-side note, the torsion calibration -- stay with their chain, since those genuinely
differ in where they read from.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# --------------------------------------------------------------------------------------
# Resampling the eye trace onto the OptiTrack clock
# --------------------------------------------------------------------------------------

def interpolate_fill_gaps(data, t_orig):
    """Fill internal NaN runs by linear interpolation, without extrapolating at the edges.

    A run of NaNs is filled only when it has a finite sample on both sides, so a recording
    that starts or ends mid-blink keeps its NaNs rather than growing an invented tail.
    """
    t = np.asarray(t_orig, float)
    y = np.asarray(data, float)

    good = np.isfinite(y)
    if good.sum() < 2:
        raise ValueError(
            f"interpolate_fill_gaps: need >=2 finite samples to interpolate; got {good.sum()}."
        )

    # linear interpolant on the good samples
    f = interp1d(t[good], y[good], kind="linear", bounds_error=False,
                 fill_value=np.nan, assume_sorted=True)

    y_filled = y.copy()
    bad_idx = np.flatnonzero(~good)
    if bad_idx.size == 0:
        return y_filled

    # fill only internal NaN runs (no edge extrapolation)
    cuts = np.where(np.diff(bad_idx) > 1)[0] + 1
    for run in np.split(bad_idx, cuts):
        i0, i1 = run[0], run[-1]
        left_ok = (i0 - 1) >= 0 and np.isfinite(y[i0 - 1])
        right_ok = (i1 + 1) < len(y) and np.isfinite(y[i1 + 1])
        if left_ok and right_ok:
            y_filled[i0:i1 + 1] = f(t[i0:i1 + 1])
    return y_filled


def resample_eyecam_to_optitrack(data, t_orig, t_resample):
    """Resample linearly onto another clock. No extrapolation, so NaN spans survive."""
    t = np.asarray(t_orig, float)
    y = np.asarray(data, float)
    t_new = np.asarray(t_resample, float)

    f = interp1d(t, y, kind="linear", bounds_error=False, fill_value=np.nan,
                 assume_sorted=True)
    return f(t_new)


def upsample_eye_trace(theta, phi, diameter, t_eyecam, t_optitrack):
    """Gap-fill on the eye-camera clock, then resample onto the OptiTrack clock.

    theta and phi are not treated as circular data because of their small range, so they
    share the linear interpolation with the diameter.

    Note what this is and is not: the result is sampled at the OptiTrack rate but carries
    no information above the eye camera's own ~50 Hz. It is an upsampling, not a
    measurement, and ``PupilRotationOptiTrack.theta_opt`` has been misread as a 240 Hz eye
    signal before.
    """
    theta_filled = interpolate_fill_gaps(theta, t_eyecam)
    phi_filled = interpolate_fill_gaps(phi, t_eyecam)
    diameter_filled = interpolate_fill_gaps(diameter, t_eyecam)

    return (resample_eyecam_to_optitrack(theta_filled, t_eyecam, t_optitrack),
            resample_eyecam_to_optitrack(phi_filled, t_eyecam, t_optitrack),
            resample_eyecam_to_optitrack(diameter_filled, t_eyecam, t_optitrack))


# --------------------------------------------------------------------------------------
# Rotations
# --------------------------------------------------------------------------------------

def rot_x(x):
    return np.array([[1, 0, 0],
                     [0, np.cos(x), -np.sin(x)],
                     [0, np.sin(x), np.cos(x)]])


def rot_y(y):
    return np.array([[np.cos(y), 0, np.sin(y)],
                     [0, 1, 0],
                     [-np.sin(y), 0, np.cos(y)]])


def rot_z(z):
    return np.array([[np.cos(z), -np.sin(z), 0],
                     [np.sin(z), np.cos(z), 0],
                     [0, 0, 1]])


def eye_camera_event_type(recording_id, event_suffix):
    """The ``event.Event`` type carrying this recording's eye-camera frame timestamps.

    ``event_suffix`` comes from the chain's own ``EyeCamTimeSource`` row, which is a table
    lookup and so stays with the caller.

    This reads the recording's LABEL and must NOT be corrected by ``EyeSideNote``. Even
    where that label names the wrong eye, the video, its DeepLabCut output and its frame
    events all come from the same physical camera, so they share one self-consistent time
    base. Only the eye-in-head azimuth sign is corrected; see
    :func:`side_sign_from_recording_id`.
    """
    if "eye_left" in recording_id:
        camera_type = "mini2p1_eye_left"
    elif "eye_right" in recording_id:
        camera_type = "mini2p1_eye_right"
    else:
        raise ValueError(f"Cannot resolve eye side from recording_id: {recording_id}")
    return f"{camera_type}_{event_suffix}"


def side_sign_from_recording_id(recording_id):
    """+1 for a recording labelled left, -1 for one labelled right.

    This reads the LABEL. For some subjects the label is wrong -- the eye videos were
    named incorrectly at acquisition -- and the correction lives in ``EyeSideNote``, which
    is a database lookup and so stays with each chain. Callers must apply it; see
    ``pupil_tracking.eye_side_sign``.
    """
    if "eye_right" in recording_id:
        return -1
    if "eye_left" in recording_id:
        return 1
    raise ValueError(f"Cannot resolve eye side from recording_id: {recording_id}")


# --------------------------------------------------------------------------------------
# Gaze
# --------------------------------------------------------------------------------------

def compute_gaze_vectors(theta, phi, q_head_world_xyzw, eye_model_azimuth_deg,
                         eye_model_elevation_deg, side_sign, chunk_size=100000,
                         progress=True, verbose=True):
    """Gaze direction in head and in world coordinates, per frame.

    Parameters
    ----------
    theta, phi
        Eye-in-orbit rotation, radians, already on the OptiTrack clock.
    q_head_world_xyzw
        (n, 4) head quaternion in XYZW order, on that same clock.
    eye_model_azimuth_deg, eye_model_elevation_deg
        The resting eye-in-head direction, in DEGREES as ``EyeModel`` stores it.
    side_sign
        +1 for the left eye, -1 for the right, after the ``EyeSideNote`` correction.

    Returns
    -------
    (gaze_in_head, gaze_in_space), both (n, 3) unit vectors, NaN on any frame with a
    non-finite quaternion or angle. Inputs are truncated to their common length.
    """
    # --- eye in head position ---
    eye_in_head_azimuth = np.radians(eye_model_azimuth_deg)      # pi/3 in literature
    eye_in_head_elevation = np.radians(eye_model_elevation_deg)  # pi/6 in literature

    R_eye_in_head = rot_z(side_sign * eye_in_head_azimuth) @ rot_x(eye_in_head_elevation)

    # turn the static eye-in-head rotation into a Rotation object once
    R_eye_in_head_static = R.from_matrix(R_eye_in_head)

    q_head_world_xyzw = np.asarray(q_head_world_xyzw, float)
    theta = np.asarray(theta, float)
    phi = np.asarray(phi, float)

    n = min(q_head_world_xyzw.shape[0], theta.shape[0])
    q_head_world_xyzw = q_head_world_xyzw[:n]
    theta = theta[:n]
    phi = phi[:n]

    # find invalid quaternions
    invalid_mask = np.any(~np.isfinite(q_head_world_xyzw), axis=1) | np.isclose(
        np.linalg.norm(q_head_world_xyzw, axis=1), 0)

    # also require finite theta/phi
    valid_mask = (~invalid_mask) & np.isfinite(theta) & np.isfinite(phi)

    if verbose:
        print("Invalid quaternion indices:", np.where(invalid_mask)[0])

    # allocate outputs instead of (n,3,3) rotation matrices
    eye_axis = np.array([0, 1, 0])
    gaze_in_head = np.full((n, 3), np.nan, float)
    gaze_in_space = np.full((n, 3), np.nan, float)

    total_chunks = (n + chunk_size - 1) // chunk_size
    steps = range(0, n, chunk_size)
    if progress:
        steps = tqdm(steps, total=total_chunks, desc="computing gaze by chunks")

    for start in steps:
        stop = min(start + chunk_size, n)

        # indices in the full array for the valid frames in this chunk
        idx = np.nonzero(valid_mask[start:stop])[0] + start
        if idx.size == 0:
            continue

        # dynamic eye rotation for this chunk.
        # originally theta: left = negative, but in the head system left is positive
        angles_chunk = np.column_stack((-theta[idx], phi[idx]))         # (k, 2)
        R_eye_in_orbit = R.from_euler("zx", angles_chunk)

        # Eye->head per frame. Rotation composition applies the rightmost first.
        R_eye_in_head_dyn = R_eye_in_orbit * R_eye_in_head_static
        gaze_in_head[idx] = R_eye_in_head_dyn.apply(eye_axis)

        # head->world from the quaternions, then eye->world
        R_head_in_space = R.from_quat(q_head_world_xyzw[idx])
        gaze_in_space[idx] = (R_head_in_space * R_eye_in_head_dyn).apply(eye_axis)

    # Rotation.apply already returns unit vectors, but this is cheap and safe
    norms = np.linalg.norm(gaze_in_head, axis=1, keepdims=True)
    gaze_in_head = np.divide(gaze_in_head, norms, out=gaze_in_head, where=norms != 0)

    norms_w = np.linalg.norm(gaze_in_space, axis=1, keepdims=True)
    gaze_in_space = np.divide(gaze_in_space, norms_w, out=gaze_in_space, where=norms_w != 0)

    if verbose:
        print("gaze_in_head:", gaze_in_head.shape)
        print("gaze_in_space:", gaze_in_space.shape)

    return gaze_in_head, gaze_in_space

def nearest_real_sample_gap(t_target, t_source, real_mask):
    """For each target time, seconds to the nearest SOURCE sample that was really measured.

    The gaze chain interpolates every interior dropout in the eye trace and caps nothing,
    so a frame whose value was drawn across a 30-second gap arrives downstream finite,
    unflagged, and indistinguishable from a measurement. Measured on 60 recordings, 10.3%
    of native eye samples are missing, the pipeline fills all but the edge frames, and 36.6%
    of what it fills sits in gaps longer than one second.

    This returns the one number that makes that visible per frame: how far, in seconds, the
    nearest genuinely observed eye sample is. Zero-ish means measured; large means the value
    is a straight line drawn between two observations that far apart. Downstream can then
    threshold it instead of trusting every finite value equally.

    Returns +inf everywhere if the source has no real samples at all.
    """
    ts = np.asarray(t_source, float)
    m = np.asarray(real_mask, bool)
    n = min(ts.size, m.size)
    ts = ts[:n][m[:n]]
    tt = np.asarray(t_target, float)
    if ts.size == 0:
        return np.full(tt.shape, np.inf)
    if ts.size == 1:
        return np.abs(tt - ts[0])

    idx = np.searchsorted(ts, tt)
    left = ts[np.clip(idx - 1, 0, ts.size - 1)]
    right = ts[np.clip(idx, 0, ts.size - 1)]
    return np.minimum(np.abs(tt - left), np.abs(right - tt))
