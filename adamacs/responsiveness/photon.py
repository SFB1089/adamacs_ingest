"""Detector calibration from raw frames: photon transfer curve (free intercept) and dark-referenced gain.

Lees et al. 2025 (Nat Protoc, doi:10.1038/s41596-024-01120-w), Procedure 7: adjacent frame
pairs X, X' give per pixel M = round((X + X')/2) and D = (X - X')^2 / 2; binned by intensity,
V_i = mean D at M = I_i (mean, not median: D is chi-square distributed); robust (Huber) fit V = q (I - I0), weighted by pixel counts.

Dark-referenced variant (user request; cf. Heintzmann et al. 2016): offset and read-noise
variance measured in shutter-closed frames, then (V - dark_var) = q (I - dark_offset) with the
intercept fixed. Photons per pixel per frame = enf^2 * (X - offset) / q (enf: PMT excess noise
factor, kurtosis-gain-checker convention).
"""
from __future__ import annotations

import numpy as np


def ptc_bins(frames, margin=4, n_bins=40, max_pct=100.0):
    """frames: (T, Ly, Lx) raw. Returns (I, V, C): bin centre intensity, mean D, pixel counts.

    Bins are percentile-spaced in M (as the kurtosis-gain-checker does; Lees bins by unique value).
    """
    X = np.asarray(frames, float)[:, margin:-margin or None, margin:-margin or None]
    M = np.round(0.5 * (X[1:] + X[:-1])).ravel()
    D = (0.5 * (X[1:] - X[:-1]) ** 2).ravel()
    edges = np.unique(np.percentile(M, np.linspace(0, max_pct, n_bins + 1)))
    which = np.clip(np.searchsorted(edges, M, side="right") - 1, 0, len(edges) - 2)
    I = np.array([np.median(M[which == b]) if np.any(which == b) else np.nan for b in range(len(edges) - 1)])
    V = np.array([np.mean(D[which == b]) if np.any(which == b) else np.nan for b in range(len(edges) - 1)])
    C = np.bincount(which, minlength=len(edges) - 1).astype(float)
    ok = np.isfinite(I) & np.isfinite(V) & (C > 0)
    return I[ok], V[ok], C[ok]


def _r2(y, yhat, w):
    ybar = np.average(y, weights=w)
    ss_res = np.sum(w * (y - yhat) ** 2); ss_tot = np.sum(w * (y - ybar) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def fit_ptc(I, V, C, fit_pct=(0, 50)):
    """Free-intercept Huber fit V = q*(I - I0) over the given intensity-percentile range -> (q, I0, r2)."""
    from sklearn.linear_model import HuberRegressor
    lo, hi = np.percentile(I, fit_pct)
    sel = (I >= lo) & (I <= hi)
    if sel.sum() < 3:
        return np.nan, np.nan, np.nan
    h = HuberRegressor().fit(I[sel, None], V[sel], sample_weight=C[sel])
    q = float(h.coef_[0]); I0 = float(-h.intercept_ / q) if q != 0 else np.nan
    return q, I0, _r2(V[sel], h.predict(I[sel, None]), C[sel])


def fit_dark_referenced(I, V, C, dark_offset, dark_var, fit_pct=(0, 50)):
    """Fit (V - dark_var) = q*(I - dark_offset) with the intercept fixed -> (q, r2). Weighted least squares
    with Huber-type reweighting (3 iterations) on the given intensity-percentile range."""
    lo, hi = np.percentile(I, fit_pct)
    sel = (I >= lo) & (I <= hi) & (I > dark_offset)
    if sel.sum() < 3:
        return np.nan, np.nan
    x = I[sel] - dark_offset; y = V[sel] - dark_var; w = C[sel].copy()
    q = np.sum(w * x * y) / np.sum(w * x * x)
    for _ in range(3):
        res = y - q * x
        s = 1.4826 * np.median(np.abs(res)) or 1.0
        k = 1.345 * s
        hw = np.where(np.abs(res) <= k, 1.0, k / np.abs(res))
        ww = w * hw
        q = np.sum(ww * x * y) / np.sum(ww * x * x)
    return float(q), _r2(y, q * x, w)


def dark_stats(frames, margin=4):
    """Offset (mean) and temporal variance of dark (shutter-closed) frames, frame-pair based like the PTC."""
    X = np.asarray(frames, float)[:, margin:-margin or None, margin:-margin or None]
    offset = float(X.mean())
    var = float((0.5 * (X[1:] - X[:-1]) ** 2).mean()) if X.shape[0] > 1 else np.nan
    return offset, var


def photons_per_s(F0, offset, npix, q, enf, fps):
    """Detected photons per second of one ROI: enf^2 * (F0 - offset) * npix / q * fps."""
    return float(enf ** 2 * (F0 - offset) * npix / q * fps)
