"""ZETA per trace, exactly as the drift audit ran it (t73 / t140), plus per-condition and condition-vs-rest tests.

Audit definitions (v1-drift-audit scripts/t73_zeta_inclusion.py, t140_zeta_all.py):
* time-series ZETA, zetapy ``zetatstest``, 100 resamples;
* window = median onset-to-onset interval of the event set;
* onsets dropped when ``onset - 2*window <= first frame`` or ``onset + 3*window >= last frame``;
* frame times = 2P frame timestamps; the trace is cut to the number of frame times; NaN samples dropped.

Per condition: one-sample ZETA on that condition's onsets (same window): does the cell respond
to THIS stimulus.
Random seeds: numpy's global generator is seeded with crc32(scan_id) + mask before each call,
so a re-run reproduces the p-values.
"""
from __future__ import annotations

import zlib

import numpy as np


def window_and_onsets(onsets, frame_times):
    ev = np.sort(np.asarray(onsets, float))
    w = float(np.median(np.diff(ev)))
    keep = (ev - 2 * w > frame_times[0]) & (ev + 3 * w < frame_times[-1])
    return w, keep


def map_cells(func, jobs, procs, timeout_s):
    """``[func(j) for j in jobs]`` in a process pool of ``procs`` workers, with a time limit.

    The pool uses the 'forkserver' start method: its children are forked from a fresh, single-threaded server
    process. With 'fork' they were forked from the populate worker itself, which runs many threads (DataJoint,
    BLAS); a child could inherit a lock held by another thread and wait on it forever. That hung two production
    workers for 16 h (2026-11-25/26, 2 of ~1,100 Zeta jobs). ``timeout_s``: the whole map raises
    ``multiprocessing.TimeoutError`` instead of hanging, the pool's children are terminated when the ``with``
    block exits, and DataJoint records the job as an error.

    Like 'spawn', 'forkserver' children import the calling script as ``__mp_main__``: a script that populates Zeta
    must keep its work under ``if __name__ == "__main__":`` (the responsiveness worker does). Without the guard every
    child starts the work again (seen 2026-11-26 in a test script)."""
    import multiprocessing as mp
    if procs <= 1:
        return [func(j) for j in jobs]
    ctx = mp.get_context("forkserver")
    with ctx.Pool(procs) as pool:
        return pool.map_async(func, jobs, chunksize=max(1, len(jobs) // (4 * procs))).get(timeout=timeout_s)


def seed_for(scan_id: str, mask: int) -> int:
    return (zlib.crc32(scan_id.encode()) + int(mask)) % (2 ** 32)


def zeta_one(frame_times, trace, onsets, window, resampling, seed):
    from zetapy import zetatstest
    T = np.asarray(frame_times, float); x = np.asarray(trace, float)
    n = min(T.size, x.size); T, x = T[:n], x[:n]
    ok = np.isfinite(x)
    np.random.seed(seed)
    p, d = zetatstest(T[ok], x[ok], np.asarray(onsets, float), max_duration=window,
                      resampling_number=resampling, plot_enabled=False)
    return float(np.squeeze(p)), float(np.squeeze(d.get("zeta_score", np.nan))), float(np.squeeze(d.get("latency_zeta", np.nan)))


def zeta_cell(args):
    """Worker function (picklable): all ZETA results for one trace.

    args = (frame_times, trace, onsets, conditions, window, params, seed)
    Returns (pooled (p, z, latency), {condition: p}).
    """
    frame_times, trace, onsets, conditions, window, params, seed = args
    res = params.get("resampling", 100)
    try:
        pooled = zeta_one(frame_times, trace, onsets, window, res, seed)
    except Exception:
        pooled = (np.nan, np.nan, np.nan)
    per = {}
    if params.get("per_condition"):
        cond = np.asarray(conditions)
        for c in np.unique(cond):
            sel = cond == c
            try:
                p1 = zeta_one(frame_times, trace, onsets[sel], window, res, seed)[0]
            except Exception:
                p1 = np.nan
            per[str(c)] = p1
    return pooled, per
