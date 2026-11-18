"""DB-free part of the VisMotor Bpod ingest: file selection and AUX-pulse <-> trial matching.

Kept separate from bpod_vismotor.py so it can be unit-tested without a database connection.
"""
import pathlib

import numpy as np

VISMOTOR_PATTERN = "VisMotor"
AUX_TRIAL_CHANNEL = "reward"        # Auxfile.bpod_channels key for analog channel 2 (WaveSurfer 'BPOD 2')
MAX_RESIDUAL_S = 0.010              # pulse-to-trial-start residual tolerated after the fit


class VisMotorPulseMismatch(RuntimeError):
    """The AUX trial-pulse train does not belong to this Bpod file."""


def find_vismotor_file(scan_dir):
    """The VisMotor Bpod session file in ``scan_dir`` or None. Never the *_LM_coord.mat."""
    mats = sorted(p for p in pathlib.Path(scan_dir).glob("*.mat")
                  if VISMOTOR_PATTERN in p.name and not p.stem.endswith("_LM_coord"))
    return mats[0] if mats else None


def match_pulses_to_trials(pulses, starts, max_residual=MAX_RESIDUAL_S):
    """Return (bpod_idx, aux_idx) pairs aligning AUX pulses to Bpod trial starts.

    Tries every lag between the two trains and keeps the one whose interval sequences agree
    best; raises VisMotorPulseMismatch if none agrees within ``max_residual``.
    """
    pulses, starts = np.asarray(pulses, float), np.asarray(starts, float)
    if len(pulses) < 2 or len(starts) < 2:
        raise VisMotorPulseMismatch(
            f"{len(pulses)} AUX trial pulses for {len(starts)} Bpod trials - Bpod did not run "
            "during this scan (stale .mat in the folder?)")
    best = None
    for lag in range(-(len(starts) - 2), len(pulses) - 1):
        b0, a0 = max(0, -lag), max(0, lag)
        n = min(len(starts) - b0, len(pulses) - a0)
        if n < 2:
            continue
        err = np.abs(np.diff(pulses[a0:a0 + n]) - np.diff(starts[b0:b0 + n]))
        score = (np.median(err), -n)
        if best is None or score < best[0]:
            best = (score, b0, a0, n)
    (med_err, _), b0, a0, n = best
    if med_err > max_residual or n < max(2, 0.8 * min(len(starts), len(pulses))):
        raise VisMotorPulseMismatch(
            f"AUX pulse intervals do not match Bpod trial intervals (median error "
            f"{med_err*1000:.1f} ms over {n} pairs) - wrong .mat for this scan?")
    return np.arange(b0, b0 + n), np.arange(a0, a0 + n)
