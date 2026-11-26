"""Orientation and direction tuning of moving-grating (or moving-bar) responses.

One function per measure. Each docstring gives the formula and its source; the short source keys
([Mazurek 2014] etc.) are listed in full under References at the end of this docstring.

Names (agreed 2026-11-25)
-------------------------
gosi, gdsi        global (vector) OSI / DSI = 1-CirVar / 1-DirCirVar = |F_2| / F_0, |F_1| / F_0 (recommended)
gosi_cv, gdsi_cv  their split-half cross-validated versions
pref_ori, pref_dir  angles of F_2 / 2 and F_1
osi, dsi, oi, di  classic point indices from R_pref, R_orth, R_null
Leon's code and the v1-drift-audit call gosi / gdsi 'osi' / 'dsi' (and gosi_cv 'osi_cv'); bare osi / dsi here
are always the point indices.

Notation
--------
theta_k   stimulus direction k = 1..D in degrees, evenly spaced around the circle
R_k       condition mean: mean response over the trials of direction k (rectified, see ``rectify``)
R_jk      response on repeat j to direction k (repeats balanced to the same count K per direction)
F_n       n-th circular Fourier component of the tuning curve, F_n = sum_k R_k exp(i n theta_k);
          F_0 = sum_k R_k, F_1 is the direction vector, F_2 the orientation vector

Inputs
------
Per-onset read-outs from ``responsiveness.Readout`` as stored there. The production read-out 16 is the
RAW response mean (no baseline subtraction; the v1-drift-audit's convention), so selectivity is relative
to the total response including spontaneous activity, not to the evoked part.

All measures that use the geometry of the directions (vectors, Hotelling, dot products, cross-validated
selectivity, shuffles, bootstrap, fit, point indices) need every direction of the protocol present and
evenly spaced. When a direction has no finite trial they are NaN; only the Kruskal-Wallis tests remain.

Fits
----
[Mazurek 2014] report fit parameters only for cells with significant orientation tuning (Hotelling).
Here the fit is stored for every cell whose curve is not flat, with its goodness of fit (in-sample R2,
cross-validated R2, chi2 test, bound flag); filter on those and on ``hotelling_p``. The in-sample R2 alone
is not a criterion (median 0.41-0.44 on random onsets; design doc section 2.6).

References
----------
[Batschelet 1981]    Batschelet E (1981) Circular Statistics in Biology. Academic Press.
[Bevington 2003]     Bevington PR, Robinson DK (2003) Data Reduction and Error Analysis for the Physical
                     Sciences, 3rd ed. McGraw-Hill. (chi-square goodness of fit of a least-squares fit)
[Efron 1993]         Efron B, Tibshirani RJ (1993) An Introduction to the Bootstrap. Chapman & Hall.
[Ernst 2004]         Ernst MD (2004) Permutation methods: a basis for exact inference. Stat Sci 19:676-685.
[Hotelling 1931]     Hotelling H (1931) The generalization of Student's ratio. Ann Math Stat 2:360-378.
[Kruskal 1952]       Kruskal WH, Wallis WA (1952) Use of ranks in one-criterion variance analysis.
                     J Am Stat Assoc 47:583-621.
[Mazurek 2014]       Mazurek M, Kager M, Van Hooser SD (2014) Robust quantification of orientation
                     selectivity and direction selectivity. Front Neural Circuits 8:92.
                     doi:10.3389/fncir.2014.00092 (vector measures, Hotelling and dot-product tests,
                     constrained double-Gaussian fit, OI / DI / OSI / DSI)
[Niell 2008]         Niell CM, Stryker MP (2008) Highly selective receptive fields in mouse visual cortex.
                     J Neurosci 28:7520-7536. (OSI / DSI normalised form)
[Phipson 2010]       Phipson B, Smyth GK (2010) Permutation p-values should never be zero.
                     Stat Appl Genet Mol Biol 9:39. ((b + 1) / (n + 1))
[Ringach 2002]       Ringach DL, Shapley RM, Hawken MJ (2002) Orientation selectivity in macaque V1:
                     diversity and laminar dependence. J Neurosci 22:5639-5651. (circular variance)
[v1-drift-audit]     Rose lab, internal: figures_fable/scripts/gratings/gr_common.py (osi_cv / dsi_cv;
                     angle convention theta = (90 - label) mod 360). No published source.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _st

FULL_CIRCLE = 360.0
HALF_CIRCLE = 180.0
MIN_DIRECTIONS = 4           # fewer sampled directions cannot define an orientation vector
MIN_REPEATS_HOTELLING = 3    # T^2 with 2 dimensions needs K - 2 >= 1 denominator degrees of freedom
MIN_REPEATS_DOT = 2          # a one-sample t-test needs 2 values


# ----------------------------------------------------------------------------------------------------------
# Angles
# ----------------------------------------------------------------------------------------------------------

ANGLE_CONVENTIONS = {
    # stimulus label L (Bonsai 'Moving_Gratings:<L>') -> angle used for tuning [deg]
    "90_minus_label": lambda L: (90.0 - L) % 360.0,   # Leon export / [v1-drift-audit] gr_common.load_animal
    "minus_label": lambda L: (-L) % 360.0,            # motion angle measured in the Bonsai frame logs (+X right, +Y up)
    "label": lambda L: L % 360.0,                     # the label itself
}


def to_angle(labels, convention):
    """Stimulus labels -> direction angles [deg] with one of ``ANGLE_CONVENTIONS``."""
    return ANGLE_CONVENTIONS[convention](np.asarray(labels, float))


def _angdiff(x, period):
    """Signed smallest difference on a circle of the given period, in [-period/2, period/2)."""
    return (np.asarray(x, float) + period / 2) % period - period / 2


def _wrap(x, period):
    """x mod period in [0, period). Float rounding can make -1e-17 % 180 == 180.0; that is mapped to 0."""
    wrapped = float(x) % period
    return 0.0 if wrapped >= period else wrapped


ANGLE_PERIODS = {"pref_ori": 180.0, "pref_dir": 360.0, "pref_dir_dot": 360.0, "fit_pref_dir": 360.0}   # [deg] table columns


def stored_angle(x, period):
    """``x`` as the table stores it (MySQL float = float32), still in [0, period).

    A value just below the period, e.g. 179.999995, is fine in float64 but rounds to 180.0 in float32 (seen in
    production on 2026-11-26: 2 of 53,024 cells). Rounded up to the period, it is mapped to 0. None / NaN stay as they are."""
    if x is None or not np.isfinite(x):
        return x
    x32 = float(np.float32(_wrap(x, period)))
    return 0.0 if x32 >= period else x32


def complete_circle(angles):
    """True when the directions are evenly spaced around the full circle.

    The vector, Hotelling and dot-product measures assume sum_k cos(n theta_k) = sum_k sin(n theta_k) = 0
    for n = 1, 2, which holds only for such a protocol."""
    a = np.sort(np.asarray(angles, float) % FULL_CIRCLE)
    if a.size < MIN_DIRECTIONS:
        return False
    gaps = np.diff(np.r_[a, a[0] + FULL_CIRCLE])
    return bool(np.allclose(gaps, FULL_CIRCLE / a.size, atol=1e-6))


# ----------------------------------------------------------------------------------------------------------
# Condition means and repeats
# ----------------------------------------------------------------------------------------------------------

def rectify(means, mode):
    """'clip': negative condition means set to 0 (the vector measures need R_k >= 0); 'none': unchanged."""
    if mode == "none":
        return means
    if mode == "clip":
        return np.clip(means, 0, None)
    raise ValueError(mode)


def condition_matrix(values, conditions, min_k=2):
    """(angles[D], R[D, K]): per-repeat responses R_jk, repeats balanced to the smallest count K over the
    directions, earliest presentations first. K < min_k gives an empty R."""
    labels = np.asarray(conditions)
    values = np.asarray(values, float)
    angles = np.array(sorted({float(x) for x in labels}))
    groups = [values[np.array([float(x) == a for x in labels])] for a in angles]
    groups = [g[np.isfinite(g)] for g in groups]
    n_repeats = min(g.size for g in groups) if groups else 0
    if n_repeats < min_k:
        return angles, np.zeros((angles.size, 0))
    return angles, np.vstack([g[:n_repeats] for g in groups])


# ----------------------------------------------------------------------------------------------------------
# Kruskal-Wallis
# ----------------------------------------------------------------------------------------------------------

def _kw(values, groups):
    """Kruskal-Wallis H test across groups [Kruskal 1952]; NaN with < 2 groups or all values equal."""
    parts = [values[(groups == g) & np.isfinite(values)] for g in np.unique(groups)]
    parts = [x for x in parts if x.size]
    if len(parts) < 2 or np.ptp(np.concatenate(parts)) == 0:
        return np.nan
    return float(_st.kruskal(*parts).pvalue)


# ----------------------------------------------------------------------------------------------------------
# Vector measures [Mazurek 2014; circular variance: Batschelet 1981, Ringach 2002]
# ----------------------------------------------------------------------------------------------------------

def vector_measures(angles, means):
    """(1-CirVar, 1-DirCirVar, preferred orientation [0, 180), preferred direction [0, 360)).

        1-CirVar    = |F_2| / F_0 = |sum_k R_k exp(2i theta_k)| / sum_k R_k
        1-DirCirVar = |F_1| / F_0 = |sum_k R_k exp(i theta_k)|  / sum_k R_k
        pref_ori    = arg(F_2) / 2,   pref_dir = arg(F_1)

    [Mazurek 2014] L_ori, L_dir; 1-DirCirVar without the classic factor 2, so a maximally selective cell
    has 1. NaN when F_0 <= 0 (pass rectified means)."""
    theta = np.deg2rad(angles)
    means = np.asarray(means, float)
    f0 = np.sum(means)
    if not np.isfinite(f0) or f0 <= 0:
        return np.nan, np.nan, np.nan, np.nan
    f2 = np.sum(means * np.exp(2j * theta))
    f1 = np.sum(means * np.exp(1j * theta))
    return (float(abs(f2) / f0), float(abs(f1) / f0),
            _wrap(np.rad2deg(np.angle(f2)) / 2, HALF_CIRCLE), _wrap(np.rad2deg(np.angle(f1)), FULL_CIRCLE))


# ----------------------------------------------------------------------------------------------------------
# Point indices [Mazurek 2014; normalised form: Niell 2008]
# ----------------------------------------------------------------------------------------------------------

def point_indices(angles, means):
    """Classic indices from the condition means (pass the rectified means).

        R_pref = largest condition mean, at direction theta_p
        R_null = mean at theta_p + 180
        R_orth = (mean at theta_p + 90 + mean at theta_p - 90) / 2      (nearest sampled directions)
        OI  = (R_pref - R_orth) / R_pref       DI  = (R_pref - R_null) / R_pref
        OSI = (R_pref - R_orth) / (R_pref + R_orth)
        DSI = (R_pref - R_null) / (R_pref + R_null)

    OSI / DSI: the normalised form of [Niell 2008], here on the measured (not fitted) curve. DI and DSI
    are as in [Mazurek 2014]. OI here is the orientation-space form of [Mazurek 2014] applied with the
    preferred DIRECTION's response; their direction-space OI pools the null response,
    (R_pref + R_null - 2 R_orth) / (R_pref + R_null), and differs for direction-selective cells.
    The indices are NaN when R_pref <= 0."""
    angles = np.asarray(angles, float)
    means = np.asarray(means, float)
    if not np.any(np.isfinite(means)):
        return dict.fromkeys(("r_pref", "r_null", "r_orth", "oi", "di", "osi", "dsi"), np.nan)
    best = int(np.nanargmax(means))
    pref_dir = angles[best]

    def mean_at(direction):
        return means[int(np.argmin(np.abs(_angdiff(angles - direction, FULL_CIRCLE))))]

    r_pref = means[best]
    r_null = mean_at(pref_dir + 180)
    r_orth = 0.5 * (mean_at(pref_dir + 90) + mean_at(pref_dir - 90))
    out = dict(r_pref=float(r_pref), r_null=float(r_null), r_orth=float(r_orth),
               oi=np.nan, di=np.nan, osi=np.nan, dsi=np.nan)
    if r_pref > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["oi"] = float((r_pref - r_orth) / r_pref)
            out["di"] = float((r_pref - r_null) / r_pref)
            out["osi"] = float((r_pref - r_orth) / (r_pref + r_orth)) if r_pref + r_orth != 0 else np.nan
            out["dsi"] = float((r_pref - r_null) / (r_pref + r_null)) if r_pref + r_null != 0 else np.nan
    return out


# ----------------------------------------------------------------------------------------------------------
# Significance of tuning [Mazurek 2014; Hotelling 1931]
# ----------------------------------------------------------------------------------------------------------

def hotelling_orientation_p(angles, repeats):
    """Is the cell orientation tuned? Hotelling's T^2 of the per-repeat orientation vectors against [0, 0].

        x_j  = (sum_k R_jk cos 2 theta_k,  sum_k R_jk sin 2 theta_k)      one 2-D vector per repeat j = 1..K
        T^2  = K  xbar' S^-1 xbar                                        S: sample covariance of the x_j
        F    = (K - 2) / (2 (K - 1)) T^2  ~  F(2, K - 2)                 p = upper tail

    [Mazurek 2014] (test) and [Hotelling 1931] (statistic). ``repeats`` is R[D, K] from
    ``condition_matrix``. NaN with K < 3 or a singular S (e.g. all responses 0)."""
    n_repeats = repeats.shape[1]
    if n_repeats < MIN_REPEATS_HOTELLING:
        return np.nan
    theta = np.deg2rad(angles)[:, None]
    vectors = np.column_stack([(repeats * np.cos(2 * theta)).sum(0),
                               (repeats * np.sin(2 * theta)).sum(0)])          # (K, 2)
    mean_vector = vectors.mean(0)
    covariance = np.cov(vectors, rowvar=False)
    if np.linalg.matrix_rank(covariance) < 2:
        return np.nan
    t2 = n_repeats * mean_vector @ np.linalg.solve(covariance, mean_vector)
    f_stat = (n_repeats - 2) / (2 * (n_repeats - 1)) * t2
    return float(_st.f.sf(f_stat, 2, n_repeats - 2))


def direction_dot_test(angles, repeats, pref_ori):
    """Is the cell direction tuned? One-sample t-test of the 'direction dot products' against 0.

        d_j = <sum_k R_jk (cos theta_k, sin theta_k), (cos pref_ori, sin pref_ori)>
            = projection of repeat j's direction vector onto the preferred orientation axis

    [Mazurek 2014]. Returns (p, mean of d_j): mean > 0 means the preferred direction is pref_ori,
    < 0 means pref_ori + 180. NaN with K < 2, no preferred orientation, or all d_j equal."""
    n_repeats = repeats.shape[1]
    if n_repeats < MIN_REPEATS_DOT or not np.isfinite(pref_ori):
        return np.nan, np.nan
    theta = np.deg2rad(angles)[:, None]
    axis = np.deg2rad(pref_ori)
    dots = (repeats * np.cos(theta)).sum(0) * np.cos(axis) + (repeats * np.sin(theta)).sum(0) * np.sin(axis)
    if np.std(dots) == 0:
        return np.nan, float(dots.mean())
    return float(_st.ttest_1samp(dots, 0.0).pvalue), float(dots.mean())


# ----------------------------------------------------------------------------------------------------------
# Cross-validated selectivity [v1-drift-audit]
# ----------------------------------------------------------------------------------------------------------

def cv_selectivity(values, labels, angles, rectify_mode):
    """Split-half cross-validated orientation and direction selectivity.

        odd / even:  condition means of the odd and of the even presentations of each direction (rectified)
        c_n      = Re( F_n(odd) * conj(F_n(even)) )          n = 2 (orientation), n = 1 (direction)
        gosi_cv  = sign(c_2) sqrt(|c_2|) / F_0               F_0 = sum of the rectified means of all trials
        gdsi_cv  = sign(c_1) sqrt(|c_1|) / F_0

    Same scale as gOSI = 1-CirVar / gDSI = 1-DirCirVar. The halves are independent, so E[c_n] = |F_n|^2 without the
    noise bias of |F_n|^2 from one curve [Mazurek 2014 discuss that bias]. After the square root it is
    slightly low for weak tuning (simulated: true 0.105 -> mean 0.078); centred on 0 without tuning and not
    clipped (negative = no reliable tuning). Source: [v1-drift-audit] osi_cv / dsi_cv, here with one fixed
    odd/even split instead of the audit's random splits."""
    theta = np.deg2rad(angles)
    label_angles = labels.astype(float)
    half_means = []
    for half in (0, 1):
        means = []
        for a in angles:
            trials = values[label_angles == a][half::2]
            means.append(trials.mean() if trials.size else np.nan)
        half_means.append(rectify(np.array(means), rectify_mode))
    all_means = rectify(np.array([values[label_angles == a].mean() for a in angles]), rectify_mode)
    f0 = np.nansum(all_means)
    if not np.all(np.isfinite(half_means[0])) or not np.all(np.isfinite(half_means[1])) or not f0 > 0:
        return dict(gosi_cv=np.nan, gdsi_cv=np.nan)

    def fourier(means, n):
        return np.sum(means * np.exp(1j * n * theta))

    c2 = np.real(fourier(half_means[0], 2) * np.conj(fourier(half_means[1], 2)))
    c1 = np.real(fourier(half_means[0], 1) * np.conj(fourier(half_means[1], 1)))
    return dict(gosi_cv=float(np.sign(c2) * np.sqrt(abs(c2)) / f0),
                gdsi_cv=float(np.sign(c1) * np.sqrt(abs(c1)) / f0))


# ----------------------------------------------------------------------------------------------------------
# Resampling [Ernst 2004; Phipson 2010; Efron 1993]
# ----------------------------------------------------------------------------------------------------------

def permutation_p(values, conditions, n_perm, mode, seed=0):
    """Label-shuffle p of 1-CirVar and 1-DirCirVar.

        p = (b + 1) / (n_perm + 1),   b = number of shuffles with a vector length >= the observed one

    The direction labels are permuted across all trials; the condition means and vector lengths are
    recomputed per shuffle. Permutation test [Ernst 2004]; the +1 keeps p > 0 [Phipson 2010]."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    labels = np.asarray(conditions)
    finite = np.isfinite(values)
    values, labels = values[finite], labels[finite]
    angles = np.array(sorted({float(x) for x in labels}))
    label_angles = labels.astype(float)

    def curve(assignment):
        return rectify(np.array([values[assignment == a].mean() for a in angles]), mode)

    ori_obs, dir_obs, _, _ = vector_measures(angles, curve(label_angles))
    if not np.isfinite(ori_obs):
        return np.nan, np.nan
    n_ori = n_dir = 0
    for _ in range(n_perm):
        ori, dirn, _, _ = vector_measures(angles, curve(rng.permutation(label_angles)))
        n_ori += ori >= ori_obs
        n_dir += dirn >= dir_obs
    return (n_ori + 1) / (n_perm + 1), (n_dir + 1) / (n_perm + 1)


def bootstrap_pref_ori_ci(values, conditions, n_boot, mode, seed=0):
    """Width [deg] of the interval around the vector preferred orientation that holds 95 % of the bootstrap
    estimates.

        width = 2 * 95th percentile over bootstraps of |pref_ori(boot) - pref_ori|   (difference on the
                180-deg circle; trials resampled with replacement within each direction)

    Bootstrap [Efron 1993]. A symmetric interval, not a percentile interval: in simulations it covers the
    true preferred orientation 91-95 % of the time (lower for weak tuning)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    label_angles = np.asarray(conditions).astype(float)
    finite = np.isfinite(values)
    values, label_angles = values[finite], label_angles[finite]
    angles = np.array(sorted(set(label_angles)))
    trial_index = [np.flatnonzero(label_angles == a) for a in angles]
    observed_means = rectify(np.array([values[i].mean() for i in trial_index]), mode)
    _, _, pref_observed, _ = vector_measures(angles, observed_means)
    if not np.isfinite(pref_observed):
        return np.nan
    deviations = []
    for _ in range(n_boot):
        means = rectify(np.array([values[rng.choice(i, i.size)].mean() for i in trial_index]), mode)
        _, _, pref, _ = vector_measures(angles, means)
        if np.isfinite(pref):
            deviations.append(abs(_angdiff(pref - pref_observed, HALF_CIRCLE)))
    return float(2 * np.percentile(deviations, 95)) if deviations else np.nan


# ----------------------------------------------------------------------------------------------------------
# Double-Gaussian fit [Mazurek 2014] and its goodness of fit [Bevington 2003]
# ----------------------------------------------------------------------------------------------------------

FIT_KEYS = ("c", "rp", "rn", "pref_dir", "sigma", "hwhh", "r2", "osi_fit", "dsi_fit", "oi_fit", "di_fit", "r2_cv",
            "chi2_red", "chi2_p", "at_bound")
N_FIT_PARAMS = 5                   # c, rp, rn, pref_dir, sigma
AMPLITUDE_BOUND = 3.0              # rp, rn in [0, 3 M]           [Mazurek 2014]
SIGMA_MAX = 180.0                  # [deg] upper bound of sigma (not in [Mazurek 2014]; keeps the fit finite)
SIGMA_STARTS = (40.0, 60.0, 90.0)  # [deg] plus step/2 and step [Mazurek 2014]
BOUND_TOL = 1e-3                   # relative distance that counts as 'at the bound'
HWHH_PER_SIGMA = np.sqrt(2 * np.log(2))   # half-width at half-height = sqrt(2 ln 2) sigma [Mazurek 2014]


def _double_gauss(params, theta):
    """R(theta) = C + Rp exp(-d(theta - theta_p)^2 / (2 s^2)) + Rn exp(-d(theta - theta_p + 180)^2 / (2 s^2)),
    d = signed angle difference on the 360-deg circle [Mazurek 2014]. params = (C, Rp, Rn, theta_p, s)."""
    offset, r_pref, r_null, theta_pref, sigma = params
    return (offset + r_pref * np.exp(-_angdiff(theta - theta_pref, FULL_CIRCLE) ** 2 / (2 * sigma ** 2))
            + r_null * np.exp(-_angdiff(theta - theta_pref + 180, FULL_CIRCLE) ** 2 / (2 * sigma ** 2)))


def _fit_curve(angles, means):
    """Least-squares double Gaussian on one curve with the [Mazurek 2014] bounds and starts:

        sigma >= step / 2,  C in [-M, M],  Rp, Rn in [0, 3 M]     M = max |mean|, step = direction spacing
        starts: theta_p = direction of the largest mean, Rp = Rn = M, C = 0,
                sigma in {step/2, step, 40, 60, 90}; the start with the lowest squared error is kept

    A fit that converges with Rn > Rp describes the same curve with the peaks named the other way round;
    it is returned with Rp <-> Rn and theta_p + 180, so Rp is always the larger peak.
    Returns (params, least_squares result, step, M), or None for a flat or too short curve."""
    from scipy.optimize import least_squares
    if angles.size < N_FIT_PARAMS or not np.all(np.isfinite(means)) or np.ptp(means) == 0:
        return None
    step = float(np.min(np.diff(np.sort(angles))))
    max_abs = float(np.max(np.abs(means)))
    theta_start = angles[int(np.argmax(means))]
    lower = [-max_abs, 0, 0, -np.inf, step / 2]
    upper = [max_abs, AMPLITUDE_BOUND * max_abs, AMPLITUDE_BOUND * max_abs, np.inf, SIGMA_MAX]
    best = None
    for sigma_start in (step / 2, step) + SIGMA_STARTS:
        sigma_start = min(max(sigma_start, step / 2 + 1e-6), SIGMA_MAX - 0.1)   # strictly inside the bounds
        try:
            result = least_squares(lambda p: _double_gauss(p, angles) - means,
                                   [0.0, max_abs, max_abs, theta_start, sigma_start], bounds=(lower, upper))
        except ValueError:
            continue
        if best is None or result.cost < best.cost:
            best = result
    if best is None:
        return None
    offset, r_pref, r_null, theta_pref, sigma = best.x
    if r_null > r_pref:
        r_pref, r_null, theta_pref = r_null, r_pref, theta_pref + 180.0
    return np.array([offset, r_pref, r_null, _wrap(theta_pref, FULL_CIRCLE), sigma]), best, step, max_abs


def fit_double_gaussian(angles, means, trials=None):
    """[Mazurek 2014] constrained double-Gaussian fit of the condition means, with its goodness of fit.

    angles[D], means[D] (unrectified: the offset C may be negative); trials: optional list of D arrays of the
    per-trial values behind each mean, needed for chi2 and r2_cv. Returns a dict with

      c, rp, rn, pref_dir, sigma    fit parameters (``_double_gauss``); rp >= rn, pref_dir in [0, 360)
      hwhh      half-width at half-height = sqrt(2 ln 2) sigma
      r2        in-sample R2 = 1 - SSE / sum_k (R_k - mean R)^2 (5 parameters on D points: high on noise)
      osi_fit   (R(p) - R(p + 90)) / (R(p) + R(p + 90)) on the fitted curve R, p = pref_dir
      dsi_fit   (R(p) - R(p + 180)) / (R(p) + R(p + 180))
      oi_fit    (R(p) - R(p + 90)) / R(p)        (the fit-based "OSI" of the VH-lab wiki; Mazurek 2014 OI)
      di_fit    (R(p) - R(p + 180)) / R(p)       (Mazurek 2014 DI); both NaN when R(p) <= 0
      r2_cv     fit on the odd trials' means, R2 of that curve on the even trials' means, and the reverse;
                the mean of the two. Near or below 0 = the fit does not predict new trials
      chi2_red  chi2 / (D - 5),  chi2 = sum_k (R_k - R(theta_k))^2 / sem_k^2,
                sem_k^2 = pooled within-direction variance / n_k (pooled: a direction without trial
                variance, e.g. CASCADE zeros, does not make chi2 infinite)          [Bevington 2003]
      chi2_p    P(chi2(D - 5) >= chi2); small = the double Gaussian does not describe the curve within noise
      at_bound  1 when sigma is at step/2 or 180, C at +-M, an amplitude at 3 M, rp at 0 (no peak: only the
                offset), or least_squares did not converge; else 0. rn = 0 is not flagged: no response to the
                null direction is a valid, strongly direction-selective fit

    All NaN for a flat curve; chi2 / r2_cv NaN without trials, with fewer than 2 trials per direction, or
    with D <= 5."""
    angles = np.asarray(angles, float)
    means = np.asarray(means, float)
    out = dict.fromkeys(FIT_KEYS, np.nan)
    fitted = _fit_curve(angles, means)
    if fitted is None:
        return out
    params, result, step, max_abs = fitted
    offset, r_pref, r_null, theta_pref, sigma = params
    total_ss = np.sum((means - means.mean()) ** 2)
    at_pref, at_orth, at_null = _double_gauss(params, np.array([theta_pref, theta_pref + 90, theta_pref + 180]))
    at_bound = (result.status <= 0
                or sigma <= step / 2 * (1 + BOUND_TOL)
                or sigma >= SIGMA_MAX * (1 - BOUND_TOL)
                or abs(offset) >= max_abs * (1 - BOUND_TOL)
                or max(r_pref, r_null) >= AMPLITUDE_BOUND * max_abs * (1 - BOUND_TOL)
                or r_pref <= BOUND_TOL * max_abs)
    with np.errstate(divide="ignore", invalid="ignore"):
        out.update(c=float(offset), rp=float(r_pref), rn=float(r_null), pref_dir=float(theta_pref),
                   sigma=float(sigma), hwhh=float(sigma * HWHH_PER_SIGMA),
                   r2=float(1 - 2 * result.cost / total_ss),          # least_squares cost = SSE / 2
                   osi_fit=float((at_pref - at_orth) / (at_pref + at_orth)) if at_pref + at_orth != 0 else np.nan,
                   dsi_fit=float((at_pref - at_null) / (at_pref + at_null)) if at_pref + at_null != 0 else np.nan,
                   oi_fit=float((at_pref - at_orth) / at_pref) if at_pref > 0 else np.nan,
                   di_fit=float((at_pref - at_null) / at_pref) if at_pref > 0 else np.nan,
                   at_bound=float(at_bound))
    if trials is None or len(trials) != angles.size:
        return out
    trials = [np.asarray(t, float)[np.isfinite(t)] for t in trials]
    n_trials = np.array([t.size for t in trials])
    if not np.all(n_trials >= 2):
        return out

    dof = angles.size - N_FIT_PARAMS
    if dof > 0:
        pooled_var = np.sum([np.sum((t - t.mean()) ** 2) for t in trials]) / np.sum(n_trials - 1)
        if pooled_var > 0:
            chi2 = float(np.sum((means - _double_gauss(params, angles)) ** 2 / (pooled_var / n_trials)))
            out.update(chi2_red=chi2 / dof, chi2_p=float(_st.chi2.sf(chi2, dof)))

    half_means = [np.array([t[half::2].mean() for t in trials]) for half in (0, 1)]
    r2_halves = []
    for half in (0, 1):
        train, test = half_means[half], half_means[1 - half]
        fitted_half = _fit_curve(angles, train)
        test_ss = np.sum((test - test.mean()) ** 2)
        if fitted_half is not None and test_ss > 0:
            r2_halves.append(1 - np.sum((test - _double_gauss(fitted_half[0], angles)) ** 2) / test_ss)
    if len(r2_halves) == 2:
        out["r2_cv"] = float(np.mean(r2_halves))
    return out


# ----------------------------------------------------------------------------------------------------------
# All measures for one cell
# ----------------------------------------------------------------------------------------------------------

def direction_tuning(values, conditions, rectify_mode="clip", fit=True, n_perm=1000, n_boot=200, seed=0,
                     angle_convention="90_minus_label"):
    """All measures for one cell.

    values: per-onset read-outs (NaN = onset outside the recording); conditions: direction labels as in the
    stimulus string, converted with ``angle_convention`` first, so every angle out is in that convention.
    The Kruskal-Wallis tests use whatever directions have finite trials; every other measure needs the full,
    evenly spaced protocol and is missing from the result otherwise (NULL in the table)."""
    values = np.asarray(values, float)
    labels = np.asarray(conditions)
    protocol_angles = np.unique(to_angle(labels.astype(float), angle_convention)) if labels.size else np.zeros(0)
    finite = np.isfinite(values)
    values, labels = values[finite], labels[finite]
    theta = to_angle(labels.astype(float), angle_convention)
    out = dict(n_directions=int(protocol_angles.size), n_repeats=0,
               dir_kw_p=_kw(values, theta),                 # [v1-drift-audit] p_dir: across the D directions
               ori_kw_p=_kw(values, theta % HALF_CIRCLE))   # p_tuned: across the D/2 orientations
    labels = np.array([f"{x:g}" for x in theta])            # angle strings: exact grouping keys
    angles = np.array(sorted({float(x) for x in labels}))
    if angles.size != protocol_angles.size or not complete_circle(angles):
        return out

    label_angles = labels.astype(float)
    raw_means = np.array([values[label_angles == a].mean() for a in angles])
    means = rectify(raw_means, rectify_mode)
    gosi, gdsi, pref_ori, pref_dir = vector_measures(angles, means)
    _, repeats = condition_matrix(values, labels)
    hotelling_p = hotelling_orientation_p(angles, repeats) if repeats.size else np.nan
    dir_dot_p, dot_mean = direction_dot_test(angles, repeats, pref_ori) if repeats.size else (np.nan, np.nan)
    if np.isfinite(dot_mean):
        pref_dir_dot = pref_ori if dot_mean > 0 else _wrap(pref_ori + 180, FULL_CIRCLE)
    else:
        pref_dir_dot = np.nan
    out.update(n_repeats=int(repeats.shape[1]) if repeats.size else 0,
               gosi=gosi, gdsi=gdsi, pref_ori=pref_ori, pref_dir=pref_dir,
               hotelling_p=hotelling_p, dir_dot_p=dir_dot_p, pref_dir_dot=pref_dir_dot)
    out.update(point_indices(angles, means))
    out.update(cv_selectivity(values, labels, angles, rectify_mode))
    ori_perm_p, dir_perm_p = permutation_p(values, labels, n_perm, rectify_mode, seed) if n_perm else (np.nan, np.nan)
    out.update(gosi_perm_p=ori_perm_p, gdsi_perm_p=dir_perm_p,
               pref_ori_ci=bootstrap_pref_ori_ci(values, labels, n_boot, rectify_mode, seed) if n_boot else np.nan)
    if fit:
        trials = [values[label_angles == a] for a in angles]
        out.update({f"fit_{k}": v for k, v in fit_double_gaussian(angles, raw_means, trials).items()})
    return out
