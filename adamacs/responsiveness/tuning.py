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
  bounds s >= step/2, C in [-M, M], Rp, Rn in [0, 3M], start s in {step/2, step, 40, 60, 90};
  reported only when the orientation tuning is significant.

Classic point indices are kept for comparison with older work (OI, DI; OSI / DSI as in
Niell & Stryker 2008), computed from the measured condition means.

Angles are in degrees (compass convention is irrelevant for the measures). Responses are per-onset
read-outs from responsiveness.Readout (already baseline-corrected).
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
            float(np.rad2deg(np.angle(zo)) / 2 % 180), float(np.rad2deg(np.angle(zd)) % 360))


def point_indices(angles, means):
    """Classic indices from the measured curve: pref = direction with the largest mean.

    R_null = response at pref+180; R_orth = mean of pref+90 and pref-90 (nearest sampled angles).
    OI = (Rp - Ro)/Rp, DI = (Rp - Rn)/Rp, OSI = (Rp - Ro)/(Rp + Ro), DSI = (Rp - Rn)/(Rp + Rn).
    """
    a = np.asarray(angles, float); m = np.asarray(means, float)
    j = int(np.nanargmax(m)); p = a[j]

    def at(t):
        return m[int(np.argmin(np.abs(_angdiff(a - t, 360))))]
    rp, rn = m[j], at(p + 180)
    ro = 0.5 * (at(p + 90) + at(p - 90))
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(r_pref=float(rp), r_null=float(rn), r_orth=float(ro),
                    oi=float((rp - ro) / rp), di=float((rp - rn) / rp),
                    osi=float((rp - ro) / (rp + ro)), dsi=float((rp - rn) / (rp + rn)))


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


def fit_double_gaussian(angles, means):
    """Mazurek et al. 2014 constrained fit. Returns dict(c, rp, rn, pref_dir, sigma, hwhh, r2,
    osi_fit, dsi_fit) or NaNs. OSI/DSI from the fitted curve at pref, pref+90, pref+180."""
    from scipy.optimize import least_squares
    a = np.asarray(angles, float); m = np.asarray(means, float)
    nan = dict(c=np.nan, rp=np.nan, rn=np.nan, pref_dir=np.nan, sigma=np.nan, hwhh=np.nan, r2=np.nan,
               osi_fit=np.nan, dsi_fit=np.nan)
    if a.size < 5 or not np.all(np.isfinite(m)):
        return nan
    step = float(np.min(np.diff(np.sort(a)))) if a.size > 1 else 45.0
    M = float(np.max(np.abs(m))) or 1.0
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
        return nan
    c, rp, rn, tp, s = best.x
    ss = np.sum((m - m.mean()) ** 2)
    fp = _double_gauss(best.x, np.array([tp, tp + 90, tp + 180]))
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(c=float(c), rp=float(rp), rn=float(rn), pref_dir=float(tp % 360), sigma=float(s),
                    hwhh=float(s * np.sqrt(2 * np.log(2))), r2=float(1 - 2 * best.cost / ss) if ss > 0 else np.nan,
                    osi_fit=float((fp[0] - fp[1]) / (fp[0] + fp[1])), dsi_fit=float((fp[0] - fp[2]) / (fp[0] + fp[2])))


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
    """Width [deg] of the 95 % bootstrap interval of the vector preferred orientation (trials resampled within condition)."""
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


def direction_tuning(values, conditions, rectify_mode="clip", fit=True, n_perm=1000, n_boot=200, seed=0,
                     angle_convention="90_minus_label"):
    """All measures for one cell. values: per-onset read-outs; conditions: direction labels [deg] as in the stimulus
    string; they are converted with ``angle_convention`` first, so every angle out is in that convention."""
    v = np.asarray(values, float); c = np.asarray(conditions)
    ok = np.isfinite(v); v, c = v[ok], c[ok]
    th = to_angle(c.astype(float), angle_convention)
    kw_dir = _kw(v, th); kw_ori = _kw(v, th % 180.0)         # audit p_dir (12 directions) / p_tuned (6 orientations)
    c = np.array([f"{x:g}" for x in th])
    ang = np.array(sorted({float(x) for x in c}))
    raw = np.array([v[c.astype(float) == a].mean() for a in ang])
    m = rectify(raw, rectify_mode)
    lo, ld, po, pd = vector_measures(ang, m)
    _, R = condition_matrix(v, c)
    p_ori = hotelling_orientation_p(ang, R) if R.size else np.nan
    p_dir, dsign = direction_dot_test(ang, R, po) if R.size else (np.nan, np.nan)
    out = dict(n_directions=int(ang.size), n_repeats=int(R.shape[1]) if R.size else 0, dir_kw_p=kw_dir, ori_kw_p=kw_ori,
               ori_vec=lo, dir_vec=ld, pref_ori_vec=po, pref_dir_vec=pd,
               hotelling_p=p_ori, dir_dot_p=p_dir,
               pref_dir_dot=(po if dsign > 0 else (po + 180) % 360) if np.isfinite(dsign) else np.nan)
    out.update(point_indices(ang, raw))
    pp = permutation_p(v, c, n_perm, rectify_mode, seed) if n_perm else (np.nan, np.nan)
    out.update(ori_vec_perm_p=pp[0], dir_vec_perm_p=pp[1],
               pref_ori_ci=bootstrap_pref_ori_ci(v, c, n_boot, rectify_mode, seed) if n_boot else np.nan)
    f = fit_double_gaussian(ang, raw) if fit else {}
    out.update({f"fit_{k}": val for k, val in f.items()})
    return out
