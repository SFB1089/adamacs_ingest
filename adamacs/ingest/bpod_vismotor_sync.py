"""DB-free part of the VisMotor Bpod ingest: file selection and AUX-pulse <-> trial matching.

Kept separate from bpod_vismotor.py so it can be unit-tested without a database connection.
"""
import pathlib

import numpy as np

VISMOTOR_PATTERN = "VisMotor"
AUX_TRIAL_CHANNEL = "reward"        # Auxfile.bpod_channels key for analog channel 2 (WaveSurfer 'BPOD 2')
SYNC_STATE = "DotIsOn"             # Bpod raises the BPOD 2 TTL when this state EXITS (entry into CueDelay)
MAX_RESIDUAL_S = 0.010              # pulse-to-trial-start residual tolerated after the fit


class VisMotorPulseMismatch(RuntimeError):
    """The AUX trial-pulse train does not belong to this Bpod file."""


def has_vismotor_files(scan_dir):
    """True if any file of a VisMotor protocol (.mat, _backup.m, ...) is in ``scan_dir``."""
    return any(VISMOTOR_PATTERN in p.name for p in pathlib.Path(scan_dir).iterdir())


def find_vismotor_file(scan_dir):
    """The VisMotor Bpod session file in ``scan_dir`` or None. Never the *_LM_coord.mat."""
    mats = sorted(p for p in pathlib.Path(scan_dir).glob("*.mat")
                  if VISMOTOR_PATTERN in p.name and not p.stem.endswith("_LM_coord"))
    return mats[0] if mats else None


def match_pulses_to_trials(pulses, starts, max_residual=MAX_RESIDUAL_S):
    """Return (bpod_idx, aux_idx) pairs aligning AUX pulses to Bpod trial starts.

    Tries every lag that pairs at least 80 % of the shorter train; among the lags whose interval
    sequences agree within ``max_residual`` (median), keeps the one with the most pairs, then the
    smallest error. Raises VisMotorPulseMismatch if no such lag exists.
    """
    pulses, starts = np.asarray(pulses, float), np.asarray(starts, float)
    if len(pulses) < 2 or len(starts) < 2:
        raise VisMotorPulseMismatch(
            f"{len(pulses)} AUX trial pulses for {len(starts)} Bpod trials - Bpod did not run "
            "during this scan (stale .mat in the folder?)")
    # Coverage first: only lags that pair at least 80 % of the shorter train are candidates. Ranking
    # short edge overlaps together with full alignments lets a 2-pair overlap with a slightly smaller
    # median error win (regular trial spacing makes every lag fit), and the session is then rejected.
    min_n = max(2, int(np.ceil(0.8 * min(len(starts), len(pulses)))))
    cands = []
    for lag in range(-(len(starts) - 2), len(pulses) - 1):
        b0, a0 = max(0, -lag), max(0, lag)
        n = min(len(starts) - b0, len(pulses) - a0)
        if n < min_n:
            continue
        err = np.abs(np.diff(pulses[a0:a0 + n]) - np.diff(starts[b0:b0 + n]))
        cands.append((float(np.median(err)), n, b0, a0))
    good = [c for c in cands if c[0] <= max_residual]
    if not good:
        best_err = min(c[0] for c in cands) if cands else float("nan")
        raise VisMotorPulseMismatch(
            f"AUX pulse intervals do not match Bpod trial intervals (best median error "
            f"{best_err*1000:.1f} ms with >= {min_n} pairs) - wrong .mat for this scan?")
    # within tolerance: most pairs first, then smallest interval error
    med_err, n, b0, a0 = min(good, key=lambda c: (-c[1], c[0]))
    return np.arange(b0, b0 + n), np.arange(a0, a0 + n)


def fit_clock(starts, pulses, bpod_idx, aux_idx, max_residual=MAX_RESIDUAL_S, min_inlier_fraction=0.8):
    """Robust linear clock model aux = a + b * bpod from paired trial starts and pulses.

    Fits once, drops pairs whose residual exceeds ``max_residual`` (e.g. an aborted first
    trial or a dropped pulse), refits on the inliers. Returns (coeffs, inlier_mask, residuals).
    """
    from numpy.polynomial import polynomial as P
    x = np.asarray(starts, float)[bpod_idx]
    y = np.asarray(pulses, float)[aux_idx]
    keep = np.ones(len(x), bool)
    for _ in range(5):
        coeffs = P.polyfit(x[keep], y[keep], deg=1)
        resid = y - P.polyval(x, coeffs)
        new = np.abs(resid - np.median(resid[keep])) < max_residual
        if new.sum() < 2 or (new == keep).all():
            break
        keep = new
    coeffs = P.polyfit(x[keep], y[keep], deg=1)
    resid = y - P.polyval(x, coeffs)
    if keep.mean() < min_inlier_fraction or np.max(np.abs(resid[keep])) > max_residual:
        raise VisMotorPulseMismatch(
            f"only {keep.sum()}/{len(keep)} trial/pulse pairs agree within {max_residual*1000:.0f} ms")
    return coeffs, keep, resid


def sync_reference_times(trial_starts, dot_exits):
    """Bpod-clock time of each trial's BPOD 2 TTL: trial start + exit of the DotIsOn state.

    The TTL is NOT at trial start. Verified 2026-11-18 on 9 sessions: in trials where the dot
    ended early the pulse arrives early by exactly (5 s - DotIsOn duration).
    """
    return np.asarray(trial_starts, float) + np.asarray(dot_exits, float)
