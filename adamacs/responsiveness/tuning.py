"""Orientation and direction tuning of moving-grating (or moving-bar) responses.

Main reference: Mazurek, Kager & Van Hooser (2014) Robust quantification of orientation selectivity
and direction selectivity. Front Neural Circuits 8:92, doi:10.3389/fncir.2014.00092.
Their recommendations, followed here:

* selectivity as vector lengths: 1-CirVar = |sum R e^{2i theta}| / sum R (orientation) and
  1-DirCirVar = |sum R e^{i theta}| / sum R (direction; no factor 2, so maximal = 1). These are
  far less biased than OI / DI at low selectivity and need fewer cells for group comparisons;
* significance of orientation tuning: Hotelling's T^2 on per-repeat orientation vectors vs [0, 0];
* significance of direction tuning: per-repeat direction vectors projected on the preferred
  orientation axis ("direction dot products"), one-sample t-test;
* preferred angle and width from a double Gaussian
  R(theta) = C + Rp exp(-d(theta-theta_p)^2 / 2 s^2) + Rn exp(-d(theta-theta_p+180)^2 / 2 s^2),
  bounds s >= step/2, C in [-M, M], Rp, Rn in [0, 3M], start s in {step/2, step, 40, 60, 90}.
  Rp is the larger peak: a fit that converges with Rn > Rp describes the same curve with the peaks
  swapped, and is reported as Rp <-> Rn, theta_p + 180. Mazurek et al. report the fit only for
  significantly tuned cells; here it is stored for every cell whose curve is not flat, with its
  goodness of fit (in-sample R2, cross-validated R2, chi2 test, bound flag): filter on those and on
  hotelling_p. The in-sample R2 alone is not a criterion (median 0.41-0.44 on random onsets).

Classic point indices are kept for comparison with older work (OI, DI; OSI / DSI as in
Niell & Stryker 2008), computed from the measured condition means after the same rectification as
the vector measures (so OI, OSI, DI, DSI lie in [0, 1] when rectified).

All measures that use the geometry of the directions (vectors, Hotelling, dot products, cross-validated
selectivity, shuffles, bootstrap, fit, point indices) need every direction of the protocol present and
evenly spaced; when a direction has no finite trial they are NaN (only the Kruskal-Wallis tests remain).

Angles are in degrees (compass convention is irrelevant for the measures). Responses are per-onset
read-outs from responsiveness.Readout, as stored there: the production read-out 16 is the RAW response
mean (no baseline subtraction; the audit's convention), so selectivity is relative to the total
response including spontaneous activity, not to the evoked part.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _st


def _kw(v, g):
    parts = [v[(g == u) & np.isfinite(v)] for u in np.unique(g)]
    parts = [x for x in parts if x.size]
    if len(parts) < 2 or np.ptp(np.concatenate(parts)) == 0:
        return np.nan
    return float(_st.kruskal(*parts).pvalue)


def _angdiff(x, period):
    """Signed smallest difference on a circle of the given period."""
    return (np.asarray(x, float) + period / 2) % period - period / 2


def _wrap(x, period):
    """x mod period in [0, period): float rounding can make -1e-17 % 180 == 180.0."""
    y = float(x) % period
    return 0.0 if y >= period else y


def complete_circle(ang):
    """True when the sampled directions are evenly spaced around the full circle (the vector, Hotelling and
    dot-product measures assume sum cos(k theta) = sum sin(k theta) = 0 over the directions)."""
    a = np.sort(np.asarray(ang, float) % 360.0)
    if a.size < 4:
        return False
    d = np.diff(np.r_[a, a[0] + 360.0])
    return bool(np.allclose(d, 360.0 / a.size, atol=1e-6))


def condition_matrix(values, conditions, min_k=2):
    """(angles[I], R[I, K]) per-repeat responses, repeats balanced to the smallest count (earliest first)."""
    c = np.asarray(conditions); v = np.asarray(values, float)
    ang = np.array(sorted({float(x) for x in c}))
    groups = [v[np.array([float(x) == a for x in c])] for a in ang]
    groups = [g[np.isfinite(g)] for g in groups]
    K = min(g.size for g in groups) if groups else 0
    if K < min_k:
        return ang, np.zeros((ang.size, 0))
    return ang, np.vstack([g[:K] for g in groups])


def rectify(m, mode):
    if mode == "none":
        return m
    if mode == "clip":
        return np.clip(m, 0, None)
    raise ValueError(mode)


def vector_measures(angles, means):
    """(1-CirVar, 1-DirCirVar, preferred orientation [0,180), preferred direction [0,360))."""
    th = np.deg2rad(angles); m = np.asarray(means, float)
    tot = np.sum(m)
    if not np.isfinite(tot) or tot <= 0:
        return np.nan, np.nan, np.nan, np.nan
    zo = np.sum(m * np.exp(2j * th)); zd = np.sum(m * np.exp(1j * th))
    return (float(abs(zo) / tot), float(abs(zd) / tot),
            _wrap(np.rad2deg(np.angle(zo)) / 2, 180.0), _wrap(np.rad2deg(np.angle(zd)), 360.0))


def point_indices(angles, means):
    """Classic indices from the condition means (pass the rectified means): pref = direction with the largest mean.

    R_null = response at pref+180; R_orth = mean of pref+90 and pref-90 (nearest sampled angles).
    OI = (Rp - Ro)/Rp, DI = (Rp - Rn)/Rp, OSI = (Rp - Ro)/(Rp + Ro), DSI = (Rp - Rn)/(Rp + Rn).
    NaN when Rp <= 0 (no response to normalise by).
    """
    a = np.asarray(angles, float); m = np.asarray(means, float)
    keys = ("r_pref", "r_null", "r_orth", "oi", "di", "osi", "dsi")
    if not np.any(np.isfinite(m)):
        return dict.fromkeys(keys, np.nan)
    j = int(np.nanargmax(m)); p = a[j]

    def at(t):
        return m[int(np.argmin(np.abs(_angdiff(a - t, 360))))]
    rp, rn = m[j], at(p + 180)
    ro = 0.5 * (at(p + 90) + at(p - 90))
    out = dict(r_pref=float(rp), r_null=float(rn), r_orth=float(ro), oi=np.nan, di=np.nan, osi=np.nan, dsi=np.nan)
    if rp > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            out.update(oi=float((rp - ro) / rp), di=float((rp - rn) / rp),
                       osi=float((rp - ro) / (rp + ro)) if rp + ro != 0 else np.nan,
                       dsi=float((rp - rn) / (rp + rn)) if rp + rn != 0 else np.nan)
    return out


def hotelling_orientation_p(angles, R):
    """Hotelling's T^2: per-repeat orientation vectors sum_k R_jk e^{2i theta_k} vs [0, 0]."""
    K = R.shape[1]
    if K < 3:
        return np.nan
    th = np.deg2rad(angles)[:, None]
    X = np.column_stack([(R * np.cos(2 * th)).sum(0), (R * np.sin(2 * th)).sum(0)])   # (K, 2)
    xbar = X.mean(0); S = np.cov(X, rowvar=False)
    if np.linalg.matrix_rank(S) < 2:
        return np.nan
    t2 = K * xbar @ np.linalg.solve(S, xbar)
    F = (K - 2) / (2 * (K - 1)) * t2
    return float(_st.f.sf(F, 2, K - 2))


def direction_dot_test(angles, R, pref_ori):
    """Per-repeat direction vectors projected on the preferred orientation axis; one-sample t-test.

    Returns (p, signed mean): sign > 0 means the preferred direction is pref_ori, < 0 pref_ori + 180.
    """
    K = R.shape[1]
    if K < 2 or not np.isfinite(pref_ori):
        return np.nan, np.nan
    th = np.deg2rad(angles)[:, None]
    u = np.deg2rad(pref_ori)
    d = (R * np.cos(th)).sum(0) * np.cos(u) + (R * np.sin(th)).sum(0) * np.sin(u)
    if np.std(d) == 0:
        return np.nan, float(d.mean())
    return float(_st.ttest_1samp(d, 0.0).pvalue), float(d.mean())


def _double_gauss(p, th):
    c, rp, rn, tp, s = p
    return (c + rp * np.exp(-_angdiff(th - tp, 360) ** 2 / (2 * s ** 2))
            + rn * np.exp(-_angdiff(th - tp + 180, 360) ** 2 / (2 * s ** 2)))


FIT_KEYS = ("c", "rp", "rn", "pref_dir", "sigma", "hwhh", "r2", "osi_fit", "dsi_fit", "r2_cv", "chi2_red", "chi2_p", "at_bound")
N_FIT_PARAMS = 5


def _fit_curve(a, m):
    """Least-squares double Gaussian on one curve (Mazurek et al. 2014 bounds and starts). Returns
    (params with rp >= rn, least_squares result, step, M) or None for a flat or too short curve."""
    from scipy.optimize import least_squares
    if a.size < N_FIT_PARAMS or not np.all(np.isfinite(m)) or np.ptp(m) == 0:
        return None
    step = float(np.min(np.diff(np.sort(a))))
    M = float(np.max(np.abs(m)))
    tm = a[int(np.argmax(m))]
    lo = [-M, 0, 0, -np.inf, step / 2]; hi = [M, 3 * M, 3 * M, np.inf, 180.0]
    best = None
    for s0 in (step / 2, step, 40.0, 60.0, 90.0):
        s0 = min(max(s0, step / 2 + 1e-6), 179.9)
        try:
            r = least_squares(lambda p: _double_gauss(p, a) - m, [0.0, M, M, tm, s0], bounds=(lo, hi))
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    c, rp, rn, tp, s = best.x
    if rn > rp:           # the same curve with the peaks named the other way round: Rp is the larger peak
        rp, rn, tp = rn, rp, tp + 180.0
    return np.array([c, rp, rn, _wrap(tp, 360.0), s]), best, step, M


def fit_double_gaussian(angles, means, trials=None):
    """Mazurek et al. 2014 constrained fit of the condition means, with its goodness of fit.

    angles[D], means[D]; trials: optional list of D arrays of the per-trial values (finite) behind each mean, for
    the goodness of fit. Returns dict with
      c, rp, rn, pref_dir, sigma, hwhh   fit parameters; rp >= rn, pref_dir in [0, 360) is the larger peak
      r2        in-sample R2 of the means (5 parameters on D points: high on pure noise, not a criterion alone)
      osi_fit, dsi_fit   from the fitted curve at pref, pref+90, pref+180
      r2_cv     cross-validated R2: fit on the means of the odd trials, R2 of that curve on the means of the even
                trials, and the reverse; average. Near or below 0 = the fit does not predict new trials
      chi2_red  sum((mean - fit)^2 / sem^2) / (D - 5), sem_d = sqrt(pooled within-direction variance / n_d)
                (pooled: a direction without trial variance, e.g. CASCADE zeros, does not make it infinite)
      chi2_p    upper tail of chi2(D - 5); small = the double Gaussian does not describe the curve within noise
      at_bound  1 when the width sits at a bound (step/2 or 180 deg), an amplitude or the offset at a bound,
                or least_squares did not converge; else 0
    All NaN for a flat curve (nothing to fit); chi2 / r2_cv NaN without trials or with D <= 5."""
    a = np.asarray(angles, float); m = np.asarray(means, float)
    out = dict.fromkeys(FIT_KEYS, np.nan)
    f = _fit_curve(a, m)
    if f is None:
        return out
    p, res, step, M = f
    c, rp, rn, tp, s = p
    ss = np.sum((m - m.mean()) ** 2)
    fp = _double_gauss(p, np.array([tp, tp + 90, tp + 180]))
    tol = 1e-3
    at_bound = (res.status <= 0 or s <= step / 2 * (1 + tol) or s >= 180.0 * (1 - tol) or abs(c) >= M * (1 - tol)
                or max(rp, rn) >= 3 * M * (1 - tol))
    with np.errstate(divide="ignore", invalid="ignore"):
        out.update(c=float(c), rp=float(rp), rn=float(rn), pref_dir=float(tp), sigma=float(s),
                   hwhh=float(s * np.sqrt(2 * np.log(2))), r2=float(1 - 2 * res.cost / ss),
                   osi_fit=float((fp[0] - fp[1]) / (fp[0] + fp[1])) if fp[0] + fp[1] != 0 else np.nan,
                   dsi_fit=float((fp[0] - fp[2]) / (fp[0] + fp[2])) if fp[0] + fp[2] != 0 else np.nan,
                   at_bound=float(at_bound))
    if trials is None or len(trials) != a.size:
        return out
    tr = [np.asarray(t, float)[np.isfinite(t)] for t in trials]
    n = np.array([t.size for t in tr])
    dof = a.size - N_FIT_PARAMS
    if dof > 0 and np.all(n >= 2):
        pooled = np.sum([np.sum((t - t.mean()) ** 2) for t in tr]) / np.sum(n - 1)
        if pooled > 0:
            chi2 = float(np.sum((m - _double_gauss(p, a)) ** 2 / (pooled / n)))
            out.update(chi2_red=chi2 / dof, chi2_p=float(_st.chi2.sf(chi2, dof)))
    if np.all(n >= 2):
        halves = [np.array([t[h::2].mean() for t in tr]) for h in (0, 1)]
        r2 = []
        for h in (0, 1):
            fh, test = _fit_curve(a, halves[h]), halves[1 - h]
            sst = np.sum((test - test.mean()) ** 2)
            if fh is not None and sst > 0:
                r2.append(1 - np.sum((test - _double_gauss(fh[0], a)) ** 2) / sst)
        if len(r2) == 2:
            out["r2_cv"] = float(np.mean(r2))
    return out


def permutation_p(values, conditions, n_perm, mode, seed=0):
    """Label-shuffle p of 1-CirVar and 1-DirCirVar (calcium-imaging convention, e.g. Kondo & Ohki 2016)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float); c = np.asarray(conditions); ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    ang = np.array(sorted({float(x) for x in c})); cf = c.astype(float)

    def curve(lbl):
        return rectify(np.array([v[lbl == a].mean() for a in ang]), mode)
    lo, ld, _, _ = vector_measures(ang, curve(cf))
    if not np.isfinite(lo):
        return np.nan, np.nan
    no = nd = 0
    for _ in range(n_perm):
        o, d, _, _ = vector_measures(ang, curve(rng.permutation(cf)))
        no += o >= lo; nd += d >= ld
    return (no + 1) / (n_perm + 1), (nd + 1) / (n_perm + 1)


def bootstrap_pref_ori_ci(values, conditions, n_boot, mode, seed=0):
    """Width [deg] of the symmetric interval around the vector preferred orientation that holds 95 % of the bootstrap
    estimates (2 x the 95th percentile of |bootstrap - estimate|; trials resampled within condition). Not a percentile
    interval: in simulations it covers the true preferred orientation 91-95 % of the time (lower for weak tuning)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float); c = np.asarray(conditions).astype(float); ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    ang = np.array(sorted(set(c)))
    idx = [np.flatnonzero(c == a) for a in ang]
    _, _, p0, _ = vector_measures(ang, rectify(np.array([v[i].mean() for i in idx]), mode))
    if not np.isfinite(p0):
        return np.nan
    dev = []
    for _ in range(n_boot):
        m = rectify(np.array([v[rng.choice(i, i.size)].mean() for i in idx]), mode)
        _, _, p, _ = vector_measures(ang, m)
        if np.isfinite(p):
            dev.append(abs(_angdiff(p - p0, 180)))
    return float(2 * np.percentile(dev, 95)) if dev else np.nan


ANGLE_CONVENTIONS = {
    # stimulus label L (Bonsai 'Moving_Gratings:<L>') -> angle used for tuning [deg]
    "90_minus_label": lambda L: (90.0 - L) % 360.0,   # Leon export / v1-drift-audit (gr_common.load_animal: theta = (90 - L) mod 360)
    "minus_label": lambda L: (-L) % 360.0,            # motion angle measured in the Bonsai frame logs (GRATINGS_AUDIT, +X right / +Y up)
    "label": lambda L: L % 360.0,                     # the label itself
}


def to_angle(labels, convention):
    return ANGLE_CONVENTIONS[convention](np.asarray(labels, float))


def cv_selectivity(v, c, ang, rectify_mode):
    """Cross-validated (split-half) orientation and direction selectivity: the trials of each direction are split into
    odd and even presentations, the Fourier components F2 = sum R e^(2i theta), F1 = sum R e^(i theta) are computed on
    each half's (rectified) means, and osi_cv = sign(c2) sqrt(|c2|) / F0, c2 = Re(F2_odd conj(F2_even)); dsi_cv the
    same with F1; F0 = sum of the (rectified) means over all trials. The product of two independent halves has no
    noise bias in |F|^2, unlike |F2| / F0 (1 - CirVar), which is biased upward with few trials (Mazurek et al. 2014);
    after the square root it is slightly low for weak tuning (simulated: true 0.105 -> mean 0.078) and centred on 0
    without tuning. The
    estimator is the v1-drift-audit's (figures_fable/scripts/gratings/gr_common.py: osi_cv / dsi_cv), with one fixed
    odd/even split. It can be negative (no reliable tuning); it is not clipped."""
    th = np.deg2rad(ang)
    halves = []
    for h in (0, 1):
        mh = []
        for a in ang:
            x = v[c.astype(float) == a]
            mh.append(x[h::2].mean() if x[h::2].size else np.nan)
        halves.append(rectify(np.array(mh), rectify_mode))
    full = rectify(np.array([v[c.astype(float) == a].mean() for a in ang]), rectify_mode)
    F0 = np.nansum(full)
    if not np.all(np.isfinite(halves[0])) or not np.all(np.isfinite(halves[1])) or not F0 > 0:
        return dict(osi_cv=np.nan, dsi_cv=np.nan)
    f = lambda m, k: np.sum(m * np.exp(1j * k * th))
    c2 = np.real(f(halves[0], 2) * np.conj(f(halves[1], 2))); c1 = np.real(f(halves[0], 1) * np.conj(f(halves[1], 1)))
    return dict(osi_cv=float(np.sign(c2) * np.sqrt(abs(c2)) / F0), dsi_cv=float(np.sign(c1) * np.sqrt(abs(c1)) / F0))


def direction_tuning(values, conditions, rectify_mode="clip", fit=True, n_perm=1000, n_boot=200, seed=0,
                     angle_convention="90_minus_label"):
    """All measures for one cell. values: per-onset read-outs; conditions: direction labels [deg] as in the stimulus
    string; they are converted with ``angle_convention`` first, so every angle out is in that convention.

    The Kruskal-Wallis tests use whatever directions have finite trials. Every other measure needs the full,
    evenly spaced protocol: when a direction has no finite trial (or the protocol is not a full circle) they are NaN."""
    v = np.asarray(values, float); c = np.asarray(conditions)
    ang_all = np.unique(to_angle(c.astype(float), angle_convention)) if c.size else np.zeros(0)
    ok = np.isfinite(v); v, c = v[ok], c[ok]
    th = to_angle(c.astype(float), angle_convention)
    kw_dir = _kw(v, th); kw_ori = _kw(v, th % 180.0)         # audit p_dir (12 directions) / p_tuned (6 orientations)
    c = np.array([f"{x:g}" for x in th])
    ang = np.array(sorted({float(x) for x in c}))
    out = dict(n_directions=int(ang_all.size), n_repeats=0, dir_kw_p=kw_dir, ori_kw_p=kw_ori)
    if ang.size != ang_all.size or not complete_circle(ang):
        return out                                           # every other column NULL
    raw = np.array([v[c.astype(float) == a].mean() for a in ang])
    m = rectify(raw, rectify_mode)
    lo, ld, po, pd = vector_measures(ang, m)
    _, R = condition_matrix(v, c)
    p_ori = hotelling_orientation_p(ang, R) if R.size else np.nan
    p_dir, dsign = direction_dot_test(ang, R, po) if R.size else (np.nan, np.nan)
    out.update(n_repeats=int(R.shape[1]) if R.size else 0,
               ori_vec=lo, dir_vec=ld, pref_ori_vec=po, pref_dir_vec=pd,
               hotelling_p=p_ori, dir_dot_p=p_dir,
               pref_dir_dot=(po if dsign > 0 else _wrap(po + 180, 360.0)) if np.isfinite(dsign) else np.nan)
    out.update(point_indices(ang, m))
    out.update(cv_selectivity(v, c, ang, rectify_mode))
    pp = permutation_p(v, c, n_perm, rectify_mode, seed) if n_perm else (np.nan, np.nan)
    out.update(ori_vec_perm_p=pp[0], dir_vec_perm_p=pp[1],
               pref_ori_ci=bootstrap_pref_ori_ci(v, c, n_boot, rectify_mode, seed) if n_boot else np.nan)
    if fit:   # the fit takes the unrectified means: its offset C may be negative
        f = fit_double_gaussian(ang, raw, [v[c.astype(float) == a] for a in ang])
        out.update({f"fit_{k}": val for k, val in f.items()})
    return out
