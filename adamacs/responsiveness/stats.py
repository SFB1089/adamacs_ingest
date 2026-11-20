"""Pure statistics behind the responsiveness tables (numpy/scipy only; no DataJoint).

``loo_oracle`` and ``signal_noise_var`` reproduce ``v1audit.oracle.loo_oracle`` / ``snr_bc``
(repeats balanced to the minimum count, earliest presentations first).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _st


# --- windows ---------------------------------------------------------------------------

def window_indices(frame_times, onsets, start_s, stop_s):
    """For every onset, the frame indices with frame_time in [onset+start, onset+stop)."""
    ft = np.asarray(frame_times, float)
    lo = np.searchsorted(ft, np.asarray(onsets) + start_s, side="left")
    hi = np.searchsorted(ft, np.asarray(onsets) + stop_s, side="left")
    return lo, hi


def window_means(X, lo, hi):
    """X: (n_cells, n_frames). Mean over [lo_k, hi_k) per onset -> (n_cells, n_onsets); NaN if empty."""
    X = np.asarray(X, float)
    cs = np.concatenate([np.zeros((X.shape[0], 1)), np.cumsum(np.nan_to_num(X), axis=1)], axis=1)
    n = (hi - lo).astype(float)
    return (cs[:, hi] - cs[:, lo]) / np.where(n > 0, n, np.nan)


# --- responsiveness --------------------------------------------------------------------

def effect_d(resp, base):
    """Paired Cohen's d of resp - base (sign = excited / suppressed)."""
    d = np.asarray(resp, float) - np.asarray(base, float)
    sd = np.nanstd(d, ddof=1)
    return float(np.nanmean(d) / sd) if sd > 0 else np.nan


def wilcoxon_p(resp, base):
    d = np.asarray(resp, float) - np.asarray(base, float)
    d = d[np.isfinite(d)]
    if d.size < 3 or np.all(d == 0):
        return np.nan
    return float(_st.wilcoxon(d).pvalue)


def holm_min(pvals):
    """Smallest Holm-adjusted p of a family (= m * min p, capped at 1)."""
    p = np.asarray([v for v in pvals if v is not None and np.isfinite(v)], float)
    return float(min(p.size * p.min(), 1.0)) if p.size else np.nan


# --- response reliability (QC of the stimulus response; repeats needed) ------------------

def balanced_repeats(conditions, order, min_k=2):
    """(levels, idx[I, K]): onset indices per condition, K = min repeat count, earliest first (audit rule)."""
    cond = np.asarray(conditions); order = np.asarray(order)
    levels, counts = np.unique(cond, return_counts=True)
    keep = counts >= min_k
    if not keep.any():
        return levels[:0], np.zeros((0, 0), int)
    levels = levels[keep]; K = int(counts[keep].min())
    idx = np.empty((len(levels), K), int)
    for j, u in enumerate(levels):
        t = np.where(cond == u)[0]
        idx[j] = t[np.argsort(order[t])][:K]
    return levels, idx


def loo_oracle(r):
    """r: (I, K) -> leave-one-repeat-out oracle correlation (v1audit.oracle.loo_oracle)."""
    K = r.shape[-1]
    m = (r.sum(-1, keepdims=True) - r) / (K - 1)
    x = r - r.mean(-2, keepdims=True); y = m - m.mean(-2, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (x * y).sum(-2) / np.sqrt((x ** 2).sum(-2) * (y ** 2).sum(-2))
    return float(np.nanmean(rho))


def signal_noise_var(r):
    """r: (I, K) -> (signal_var, noise_var); their ratio is v1audit.oracle.snr_bc."""
    K = r.shape[-1]
    noise = float(r.var(-1, ddof=1).mean(-1))
    signal = float(r.mean(-1).var(-1, ddof=1) - noise / K)
    return signal, noise


# --- trace quality ---------------------------------------------------------------------

def robust_noise_sd(x):
    """1.4826 * MAD of first differences / sqrt(2): white-noise SD robust to transients."""
    d = np.diff(np.asarray(x, float)); d = d[np.isfinite(d)]
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) if d.size else np.nan


def snr_suite2p(x):
    """Suite2p 2026: 1 - var(diff x) / (2 var x)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    v = np.var(x)
    return float(1 - np.var(np.diff(x)) / (2 * v)) if v > 0 else np.nan


def noise_nu(dff, fps):
    """CASCADE noise level [%/sqrt(Hz)]: 100 * median|diff(dF/F)| / sqrt(fps) (Rupprecht et al. 2021;
    cascade2p.utils.calculate_noise_levels)."""
    d = np.abs(np.diff(np.asarray(dff, float)))
    return float(100 * np.nanmedian(d) / np.sqrt(fps))


def event_rate(dff, fps, k=3.0, min_frames=2):
    """[1/min] runs of >= min_frames frames above k * robust noise SD."""
    x = np.asarray(dff, float); sd = robust_noise_sd(x)
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    above = (np.nan_to_num(x) > k * sd).astype(int)
    starts = np.flatnonzero(np.diff(np.concatenate([[0], above])) == 1)
    ends = np.flatnonzero(np.diff(np.concatenate([above, [0]])) == -1)
    return int(np.sum((ends - starts + 1) >= min_frames)) / (x.size / fps / 60.0)


def kurtosis(F):
    """Pearson kurtosis of raw F (normal = 3)."""
    F = np.asarray(F, float); F = F[np.isfinite(F)]
    return float(_st.kurtosis(F, fisher=False)) if F.size > 3 else np.nan


def pearson_r(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def neuropil_sweep(F, Fneu, alphas=None, bin_frames=1):
    """Mean off-diagonal Pearson r between cells of F - alpha*Fneu for each alpha -> (alphas, mean_r)."""
    alphas = np.round(np.arange(0, 2.0001, 0.05), 2) if alphas is None else np.asarray(alphas)
    F = np.asarray(F, float); N = np.asarray(Fneu, float)
    if bin_frames > 1:
        T = (F.shape[1] // bin_frames) * bin_frames
        F = F[:, :T].reshape(F.shape[0], -1, bin_frames).mean(-1)
        N = N[:, :T].reshape(N.shape[0], -1, bin_frames).mean(-1)
    iu = np.triu_indices(F.shape[0], 1)
    mean_r = []
    for a in alphas:
        C = np.corrcoef(F - a * N)[iu]
        mean_r.append(np.nanmean(C))
    return alphas, np.array(mean_r)


# --- read-out per onset (ResponseParamSet.measure / correction / test) ---------------------

MEASURES = ("mean", "median", "integral", "peak", "trough", "peak_abs", "auc_pos", "peak_latency")


def _boxcar(X, k):
    if k <= 1:
        return X
    ker = np.ones(k) / k
    return np.apply_along_axis(lambda x: np.convolve(np.nan_to_num(x), ker, mode="same"), 1, X)


def measure(X, frame_times, starts, stops, method, smooth_frames=0):
    """Read-out of X (cells, frames) in [start_k, stop_k) per onset -> (cells, onsets); NaN for an empty window.

    mean          average level                      (robust default; = audit 'resp' window mean)
    median        median level                       (robust to single-frame artefacts)
    integral      mean * window length  [unit*s]     (area; grows with duration - compare equal windows only)
    peak          maximum (after boxcar smoothing)   (transient responses; biased up by noise -> smooth / z-score)
    trough        minimum                            (suppression)
    peak_abs      signed value of max |x|            (excitation or suppression, whichever is larger)
    auc_pos       area of the positive part [unit*s] (rectified area; for event-like traces)
    peak_latency  [s] time of the maximum after window start
    """
    if method not in MEASURES + ("sd",):
        raise ValueError(f"unknown measure {method!r}")
    ft = np.asarray(frame_times, float); X = np.asarray(X, float)
    lo = np.searchsorted(ft, starts); hi = np.searchsorted(ft, stops)
    dur = np.asarray(stops, float) - np.asarray(starts, float)
    if method in ("mean", "integral"):
        m = window_means(X, lo, hi)
        return m * dur if method == "integral" else m
    if method in ("peak", "trough", "peak_abs", "peak_latency"):
        X = _boxcar(X, smooth_frames)
    dt = float(np.median(np.diff(ft)))
    out = np.full((X.shape[0], len(lo)), np.nan)
    for j, (a, b) in enumerate(zip(lo, hi)):
        if b <= a:
            continue
        w = X[:, a:b]
        if method == "median":
            out[:, j] = np.nanmedian(w, 1)
        elif method == "peak":
            out[:, j] = np.nanmax(w, 1)
        elif method == "trough":
            out[:, j] = np.nanmin(w, 1)
        elif method == "peak_abs":
            out[:, j] = w[np.arange(w.shape[0]), np.nanargmax(np.abs(w), 1)]
        elif method == "sd":
            out[:, j] = np.nanstd(w, 1, ddof=1) if w.shape[1] > 1 else np.nan
        elif method == "auc_pos":
            out[:, j] = np.nansum(np.clip(w, 0, None), 1) * dt
        else:  # peak_latency
            out[:, j] = ft[a + np.nanargmax(w, 1)] - starts[j]
    return out


def correct(R, B, mode, B_sd=None):
    """Baseline correction of read-outs R (cells, onsets) by baseline means B.

    zscore: (R - B) / s, s = median over onsets of the frame SD inside the baseline window
    (``B_sd`` = measure(..., 'sd') of the baseline windows): response in units of pre-stimulus noise.
    """
    if mode == "subtract":
        return R - B
    if mode == "divide":
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(np.abs(B) > 1e-9, (R - B) / np.abs(B), np.nan)
    if mode == "zscore":
        sd = np.nanmedian(B_sd, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(sd > 0, (R - B) / sd, np.nan)
    if mode == "none":
        return R
    raise ValueError(f"unknown correction {mode!r}")


def paired_test(resp, base, test, n_perm=10000, seed=0):
    """p of response vs baseline per onset (pass zeros as base to test against 0)."""
    d = np.asarray(resp, float) - np.asarray(base, float)
    d = d[np.isfinite(d)]
    if d.size < 3 or np.all(d == 0):
        return np.nan
    if test == "wilcoxon":
        return float(_st.wilcoxon(d).pvalue)
    if test == "ttest":
        return float(_st.ttest_1samp(d, 0.0).pvalue)
    if test == "ttest_greater":                     # v1-drift-audit gratings p_resp: one-sided, response > baseline
        t, p = _st.ttest_1samp(d, 0.0)
        return float(p / 2 if t > 0 else 1 - p / 2)
    if test == "signflip":
        rng = np.random.default_rng(seed)
        null = np.abs((rng.choice([-1.0, 1.0], size=(n_perm, d.size)) * d).mean(1))
        return float((np.sum(null >= abs(d.mean())) + 1) / (n_perm + 1))
    raise ValueError(f"unknown test {test!r}")


# --- preferred condition --------------------------------------------------------------

def preferred(values, conditions):
    """(label, mean, p vs 0) of the condition with the largest mean of ``values`` (per-onset read-outs).

    Chosen and measured on the same onsets: the mean is biased upward; use preferred_cv for an unbiased value.
    """
    v = np.asarray(values, float); c = np.asarray(conditions)
    levels = np.unique(c)
    means = np.array([np.nanmean(v[c == u]) if np.isfinite(v[c == u]).any() else -np.inf for u in levels])
    if not np.isfinite(means).any():
        return None, np.nan, np.nan
    j = int(np.argmax(means))
    sel = v[c == levels[j]]
    return str(levels[j]), float(means[j]), paired_test(sel, np.zeros_like(sel), "wilcoxon")


def preferred_cv(values, conditions):
    """Cross-validated preferred response: choose the best condition on odd presentations of every
    condition, read it on even ones, and the reverse; average the two. (label chosen on all data, value)."""
    v = np.asarray(values, float); c = np.asarray(conditions)
    rank = np.zeros(v.size, int)
    for u in np.unique(c):
        idx = np.flatnonzero(c == u); rank[idx] = np.arange(idx.size)
    odd = rank % 2 == 1
    out = []
    for a, b in ((odd, ~odd), (~odd, odd)):
        lab, _, _ = preferred(v[a], c[a])
        if lab is not None and np.any(b & (c == lab)):
            out.append(np.nanmean(v[b & (c == lab)]))
    return float(np.mean(out)) if out else np.nan


def kruskal_p(values, groups):
    """Kruskal-Wallis p across groups (NaN dropped; v1-drift-audit kw_p without tie correction is within 1e-3 for
    continuous read-outs). NaN if fewer than two non-empty groups or no variance."""
    v = np.asarray(values, float); g = np.asarray(groups)
    parts = [v[(g == u) & np.isfinite(v)] for u in np.unique(g)]
    parts = [x for x in parts if x.size]
    if len(parts) < 2 or np.ptp(np.concatenate(parts)) == 0:
        return np.nan
    return float(_st.kruskal(*parts).pvalue)


# --- vectorised group tests across cells (Readout: responsiveness and selectivity families) ----------------

def _group_masks(labels):
    lab = np.asarray(labels)
    return [lab == u for u in np.unique(lab)]


def kw_rows(X, labels):
    """Kruskal-Wallis p per row of X (cells x observations) across the groups in labels, with the tie correction of
    scipy.stats.kruskal (CASCADE traces have many ties at 0). NaN observations are dropped per row. NaN where a row
    has fewer than two non-empty groups or no variance (all values tied)."""
    X = np.atleast_2d(np.asarray(X, float)); masks = _group_masks(labels)
    out = np.full(X.shape[0], np.nan)
    if len(masks) < 2:
        return out
    rows_nan = np.isnan(X).any(1)
    full = np.where(~rows_nan)[0]
    if full.size:
        Y = X[full]; N = Y.shape[1]
        R = _st.rankdata(Y, axis=1)
        H = np.zeros(full.size)
        for m in masks:
            n = m.sum()
            if n:
                H += n * (R[:, m].mean(1) - (N + 1) / 2.0) ** 2
        H *= 12.0 / (N * (N + 1))
        S = np.sort(Y, 1)
        ties = np.zeros(full.size)
        for i in range(full.size):                                   # tie correction: sum(t^3 - t) over tied runs
            _, c = np.unique(S[i], return_counts=True)
            ties[i] = np.sum(c.astype(float) ** 3 - c)
        corr = 1.0 - ties / (N ** 3 - N)
        with np.errstate(invalid="ignore", divide="ignore"):
            Hc = np.where(corr > 0, H / corr, np.nan)
        k = sum(1 for m in masks if m.any())
        out[full] = np.where(np.isfinite(Hc), _st.chi2.sf(Hc, k - 1), np.nan)
    for i in np.where(rows_nan)[0]:                                    # rows with NaN: drop them per row
        out[i] = kruskal_p(X[i], np.asarray(labels))
    return out


def anova_rows(X, labels):
    """One-way ANOVA p per row of X across the groups in labels (scipy.stats.f_oneway), NaN-aware. NaN where a
    row has fewer than two non-empty groups, no within-group degrees of freedom or no variance."""
    X = np.atleast_2d(np.asarray(X, float)); masks = _group_masks(labels)
    ok = np.isfinite(X)
    ns = np.stack([(ok & m).sum(1) for m in masks], 1).astype(float)                    # rows x groups
    sums = np.stack([np.where(ok & m, X, 0).sum(1) for m in masks], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / ns
        N = ns.sum(1); grand = sums.sum(1) / N
        ssb = np.nansum(ns * (means - grand[:, None]) ** 2, 1)
        ssw = np.zeros(X.shape[0])
        for j, m in enumerate(masks):
            d = np.where(ok & m, X - means[:, j:j + 1], 0.0)
            ssw += (d ** 2).sum(1)
        k = (ns > 0).sum(1)
        dfb, dfw = k - 1, N - k
        F = (ssb / dfb) / (ssw / dfw)
        p = _st.f.sf(F, dfb, dfw)
    bad = (k < 2) | (dfw < 1) | ~np.isfinite(F) | (ssw + ssb == 0)
    p[bad] = np.nan
    return p


# --- reliability, sparseness and effect size (literature-defined; see Readout.Cell for the column definitions) ----

def reliability_components(r):
    """r: (I conditions, K repeats), balanced. Variance components of the condition responses and the measures built on them.

    signal_var  = Var_I(mean over repeats) - noise_var / K, unclipped (negative = no stimulus-locked signal): the
                  signal power of Sahani & Linden 2003 (NIPS 15) for conditions instead of time bins; equal to the
                  cross-repeat covariance used by Stringer et al. 2019 (Nature 571:361, doi:10.1038/s41586-019-1346-5)
    noise_var   = mean over conditions of the across-repeat variance
    fev         = (Var_total - noise_var) / Var_total, Var_total over all single-trial responses: fraction of explainable
                  variance, Cadena et al. 2019 (PLoS Comput Biol 15:e1006897, doi:10.1371/journal.pcbi.1006897)
    cc_max      = 1 / sqrt(1 + NP / (K * SP)), SP = [Var(sum_n R_n) - sum_n Var(R_n)] / (K (K - 1)), NP = mean_n Var(R_n) - SP:
                  the largest correlation any model of the condition means can reach, Schoppe et al. 2016 (Front Comput
                  Neurosci 10:10, doi:10.3389/fncom.2016.00010; Hsu, Borst & Theunissen 2004); NaN when SP <= 0
    """
    r = np.asarray(r, float)
    I, K = r.shape
    if I < 2 or K < 2 or not np.all(np.isfinite(r)):
        return dict(signal_var=np.nan, noise_var=np.nan, fev=np.nan, cc_max=np.nan)
    noise = float(r.var(1, ddof=1).mean())
    signal = float(r.mean(1).var(ddof=1) - noise / K)
    tot = float(r.var(ddof=1))
    fev = (tot - noise) / tot if tot > 0 else np.nan
    v_sum = float(r.sum(1).var(ddof=1)); v_each = r.var(0, ddof=1)        # Var across conditions of each repeat
    sp = (v_sum - float(v_each.sum())) / (K * (K - 1))
    npow = float(v_each.mean()) - sp
    cc = 1.0 / np.sqrt(1.0 + npow / (K * sp)) if sp > 0 and npow >= 0 else np.nan
    return dict(signal_var=signal, noise_var=noise, fev=fev, cc_max=cc)


def lifetime_sparseness(means):
    """Vinje & Gallant 2000 (Science 287:1273, doi:10.1126/science.287.5456.1273), after Rolls & Tovee 1995 (J Neurophysiol
    73:713): S = [1 - (sum r/n)^2 / sum(r^2/n)] / (1 - 1/n) over the n condition means r >= 0 (negative means are
    clipped to 0). 0 = equal response to every stimulus, 1 = response to a single stimulus. Biased towards 1 for noisy,
    sparse cells: read it together with the reliability measures. NaN if fewer than 2 conditions or no positive mean."""
    r = np.clip(np.asarray(means, float), 0, None)
    r = r[np.isfinite(r)]; n = r.size
    if n < 2 or np.sum(r ** 2) <= 0:
        return np.nan
    return float((1.0 - (r.sum() / n) ** 2 / np.sum(r ** 2 / n)) / (1.0 - 1.0 / n))


def auroc_rows(A, B):
    """Area under the ROC curve between each row of A (responses) and B (baselines), unpaired, ties counted as 1/2:
    AUC = U / (n_A n_B), the Mann-Whitney U (Green & Swets 1966; Britten et al. 1992, J Neurosci 12:4745,
    doi:10.1523/JNEUROSCI.12-12-04745.1992). 0.5 = no difference, > 0.5 = responses larger. NaN-aware per row."""
    A = np.atleast_2d(np.asarray(A, float)); B = np.atleast_2d(np.asarray(B, float))
    out = np.full(A.shape[0], np.nan)
    for i in range(A.shape[0]):
        a, b = A[i][np.isfinite(A[i])], B[i][np.isfinite(B[i])]
        if a.size and b.size:
            r = _st.rankdata(np.r_[a, b])
            out[i] = (r[:a.size].sum() - a.size * (a.size + 1) / 2.0) / (a.size * b.size)
    return out
