"""Physical stimulus timing from the photodiode in the bench2p WaveSurfer aux .h5 (read-only).

Algorithm of the v1-drift-audit (``src/v1audit/photodiode.py``, notebook 04), unchanged:

* channels: header/AIChannelNames; 'Visual Stim ID' is the Bonsai plateau DataJoint's onsets come from,
  'Photodiode' the diode on a screen-corner square. Same 20 kHz DAQ clock as the 2P frame clock, so no
  clock alignment is needed; event.Event times are seconds from the sweep start.
* the square is bright during the grey ITI and dark during the stimulus; the raw trace carries a refresh
  pulse every display frame (large while bright).
* refresh period T = median interval between bright refresh peaks; envelope = running max over T; bright /
  dark state by an Otsu threshold, states shorter than 3 frames removed.
* physical onset of trial k = first bright->dark flip in [on_k - 150 ms, on_k + 400 ms]; physical offset =
  the next dark->bright flip. Resolution about +-T/2 (+-8 ms at 60 Hz).

Measured on 33 sessions (ROS-2209/2210/2328, from Nov 2025): latency 80-120 ms, IQR ~20 ms. Older animals have
no 'Photodiode' channel.
"""
from __future__ import annotations

import numpy as np


def read_photodiode(path):
    """(trace float32 or None, sample rate, channel names)."""
    import h5py
    with h5py.File(path, "r") as h:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in h["header"]["AIChannelNames"][()]]
        sr = float(np.asarray(h["header"]["AcquisitionSampleRate"][()]).ravel()[0])
        if "Photodiode" not in names:
            return None, sr, names
        sw = [k for k in h.keys() if k.startswith("sweep")][0]
        return np.asarray(h[sw]["analogScans"][names.index("Photodiode"), :], dtype=np.float32), sr, names


def refresh_period(trace, sr):
    from scipy.signal import find_peaks
    med, hi = np.median(trace[::50]), np.percentile(trace[::50], 99)
    peaks, _ = find_peaks(trace, height=med + 0.5 * (hi - med), distance=int(0.6 * sr / 60))
    d = np.diff(peaks / sr)
    return float(np.median(d[d < 0.05]))


def bright_state(trace, sr, T, ds=20):
    """Binary bright / dark state at sr/ds Hz -> (bright, fs)."""
    from scipy.ndimage import binary_closing, binary_opening, maximum_filter1d
    from skimage.filters import threshold_otsu
    env = maximum_filter1d(trace, size=int(round(T * sr)))[::ds]
    bright = env > float(threshold_otsu(env[::10]))
    k = int(round(3 * T * sr / ds))
    return binary_closing(binary_opening(bright, np.ones(k)), np.ones(k)), sr / ds


def physical_times(bright, fs, on):
    """Per DataJoint onset: (physical onset, physical offset) [s], NaN where no flip is found."""
    d = np.diff(bright.astype(np.int8))
    down = (np.flatnonzero(d == -1) + 1) / fs
    up = (np.flatnonzero(d == 1) + 1) / fs
    p_on = np.full(len(on), np.nan); p_off = np.full(len(on), np.nan)
    for k, t0 in enumerate(np.asarray(on, float)):
        j = np.searchsorted(down, t0 - 0.150)
        if j < len(down) and down[j] <= t0 + 0.400:
            p_on[k] = down[j]
            u = np.searchsorted(up, down[j])
            if u < len(up):
                p_off[k] = up[u]
    return p_on, p_off


def screen_times(path, on):
    """-> dict(has_diode, refresh_hz, on, off) for the DataJoint onsets ``on`` of one recording."""
    tr, sr, _ = read_photodiode(path)
    if tr is None:
        return {"has_diode": False}
    T = refresh_period(tr, sr)
    b, fs = bright_state(tr, sr, T)
    p_on, p_off = physical_times(b, fs, on)
    return {"has_diode": True, "refresh_hz": 1.0 / T, "on": p_on, "off": p_off}
