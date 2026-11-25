"""Image-refined pupil fit for head-fixed (bench2p) face-cam recordings.

The bench2p face camera shows a BRIGHT pupil (IR illumination) against a dark iris, so the pupil rim is a strong
bright->dark edge. Instead of fitting an ellipse to the DLC pupil markers (production, PupilEllipseFitting set 4),
this module fits the ellipse to the rim in the video frame itself and scores every fit, so that frames on which the
markers are confidently wrong (dim or half-closed eye, blink, bright streak) are rejected instead of producing
outliers.

Per video frame (grayscale, Gaussian sigma 1.5 px):
  1. guide: center = median of the valid pupil markers (>= 3), else the last accepted center; radius R0 = session
     median marker radius on frames with >= 7 valid markers. Markers are DISK head-fixed markers (observed or
     imputed) when available, else the DLC markers above the likelihood threshold.
  2. Starburst-style rays: 48 directions, sampled from 0.35 to 1.8 x R0 in 0.5 px steps; per ray the edge is the
     strongest bright->dark step (most negative derivative).
  3. robust ellipse: RANSAC (60 draws of 6 edge points, cv2.fitEllipse; semi-axes within 0.3-3 x R0), inlier =
     normalised radial residual < 2 px; final cv2.fitEllipse on the inliers.
  4. quality: inlier fraction, and rim step = median gray level 3 px inside the fitted ellipse minus 3 px outside
     (bright pupil -> positive; a closed or dim eye gives a small step).
Accept (v2): inlier_frac >= 0.6, rim step >= 0.4 x session median, both semi-axes within 0.25-3 x session median.
Temporal consistency: accepted centers > 25 px from the 2 s rolling median of accepted centers are rejected when
the excursion lasts < 0.25 s (a locked-on wrong ellipse); sustained shifts (real gaze changes) are kept.

v4 (opt-in per parameter set, see V4_DEFAULTS): rays scaled by the per-frame guide radius when the pupil is dilated
(start of every recording), a long-ray retry for frames without an acceptable fit, and fits at the inlier floor
accepted only when confident fits nearby in time corroborate them.

Ported unchanged from the V1-drift audit (trose-neuro/v1-drift-audit: scripts/t8_pupil_image.py = fit, quality;
scripts/t8_chain.py = v2 accept, temporal consistency), where it was validated on 18 sessions (ROS-1985, ROS-2328)
and used for the audit's retrained models. The RANSAC random stream runs sequentially over the frames from one
generator (seed 0), so a full sequential pass reproduces the reference results exactly.

Pure numpy / OpenCV / pandas: no DataJoint, safe to import anywhere.
"""
import numpy as np
import pandas as pd
import cv2

CODE_VERSION = "image_rim_v3 (gate + animal fissure + multistart, 2026-09-27)"
# v3 changes the acceptance (spatial gate) and the eye model (animal-median fissure prior, multi-start fit) only; the
# per-frame rim fits are identical to v2, so v2 caches and stored v2 ellipses stay valid inputs.

# spatial gate (applied after the temporal check; reference: v1-drift-audit scripts/t8_chain.py, review 27.9 C4):
# accepted centres farther than max(gate_min_px, gate_r0_mult * R0) from the median of the temporally kept centres are
# rejected whatever their duration (sustained 0.3-0.75 s lock-ons onto a fixed bright object passed the temporal check)
GATE_DEFAULTS = dict(gate_min_px=100.0, gate_r0_mult=3.0)

DEFAULTS = dict(n_rays=48, r_min=0.35, r_max=1.8, r_step=0.5, blur_sigma=1.5, ransac_draws=60, inlier_tol=2.0,
                min_inlier_frac=0.6, min_rim_step_rel=0.4, axis_min=0.25, axis_max=3.0,
                tc_dist_px=25.0, tc_window_s=2.0, tc_max_brief_s=0.25, seed=0)

CODE_VERSION_V4 = "image_rim_v4 (guide-scaled rays + retry + weak-fit check, 2026-11-17)"
# v4 (both off when the parameter is None, so v3 parameter rows reproduce v3 exactly):
#   r_max_guide: rays end at max(r_max * R0, r_max_guide * rho_f), rho_f = rolling median (guide_r_window_s) of the
#     per-frame guide-marker radius. The pupil is 1.5-2x its session size in the first ~5-10 s of every recording, where
#     rays ending at r_max * R0 = 1.8 R0 stop at or inside the rim.
#   guide_r_engage: rays are guide-scaled only where the rolling guide radius exceeds guide_r_engage * R0 (None: R0).
#     All other frames keep the v3 rays exactly; the RANSAC stream draws the same numbers per frame whatever the ray
#     length, so their fits are bit-identical to v3 (unless the fallback centre differs: frames with < 3 markers use
#     the last accepted centre, which can differ after a frame that v4 fitted differently).
#   r_max_retry: a frame with no fit or inlier_frac < min_inlier_frac is fitted again with rays to r_max_retry x
#     max(R0, rolling guide radius), from its own random stream (seed + 1), and the retry is kept if its inlier fraction
#     is higher. For the first seconds of a recording, when the pupil is so large that the markers are lost.
#     The fallback centre for frames with < 3 guide markers (the last fit centre) moves with the first fit only, as
#     in v3: letting retry fits move it walked the centre off the eye for ~70 s in ROS-1881 A (no DLC after ~400 s),
#     and moving it only with confident fits (inlier_frac >= 0.8) got it stuck (acceptance 0.83 vs 0.99).
#   weak_inlier_frac: a fit with inlier_frac below it (but >= min_inlier_frac) is accepted only if confident fits
#     nearby in time corroborate it: the median of the v2-accepted fits with inlier_frac >= weak_ref_min_inlier within
#     +-weak_ref_window_s exists, its centre is within weak_ref_dist_max * R0 and its semi-major axis ratio within
#     [weak_ref_axis_min, weak_ref_axis_max] (a weak fit is never its own reference). Marker-independent (the markers are
#     unreliable exactly where the rim is weak: occlusion, blinks).
V4_DEFAULTS = dict(r_max_guide=None, guide_r_window_s=0.5, guide_r_engage=None, r_max_retry=None, weak_inlier_frac=None,
                   weak_ref_min_inlier=0.8, weak_ref_window_s=0.5, weak_ref_dist_max=0.3, weak_ref_axis_min=0.8,
                   weak_ref_axis_max=1.25)

CODE_VERSION_V41 = "image_rim_v4.1 (v4 + anchored fallback centre, 2026-11-24)"
# v4.1 = v4 plus an anchor for the fallback centre of frames without a guide (< 3 guide markers). Bug fixed: in v4 (and
# v3) such a frame starts at the previous frame's FIRST fit centre, accepted or not, so a run of guide-less frames is a
# free random walk; in ROS-1881 A scan9FPZHKKB (DISK refuses 92 % of frames, no DLC either, from ~400 s on) a blink at
# frame 116,021 produced rejected fits 180 px below the eye, the chain followed them and ended locked on an object
# ~1050 px off the eye (set 11: frac_accepted 0.575, every kept centre off the eye; set 10 0.989).
# The anchor is the coordinate-wise median of the centres of the last anchor_med_n (120 = 2 s at 60 fps, the temporal
# check's window) fits that pass an online accept test: finite, inlier_frac >= anchor_min_inlier (= v4's
# weak_inlier_frac), both semi-axes within [axis_min, axis_max] x R0, rim step >= anchor_rim_rel x the median rim step of
# the last anchor_rim_n passing fits with inlier_frac >= anchor_rim_min_inlier. A guide frame without a passing fit adds
# its guide centre. A guide-less frame whose v4 start (the previous FIRST fit) is farther than anchor_r0_mult x R0 from
# the anchor starts at the anchor instead.
# Why a median and not the last passing fit (tried first): in ROS-1881 A a blink at frame 184,267 dragged a bright,
# well-fitting blob 60 px down the lower lid in 7 frames (inlier 0.9, high rim step, each step < 2 R0); the anchor
# followed it and the pupil, back 60 px above, was out of reach of the rays for 60 s. The offline temporal check rejects
# such brief excursions; the 2-s median ignores them in the same way, and follows a sustained gaze shift within 1 s.
# Nothing else changes: the v4 start is used whenever it is within the radius. Frames whose start differs from v4's
# ("divergent": from a re-anchored frame to the next guide frame) are also fitted from v4's start (a shadow of the v4
# chain) so that both random streams (RANSAC seed, retry seed + 1) advance exactly as in v4; every other frame (all
# guide frames, and every frame of a recording that is never re-anchored) is therefore bit-identical to v4.
# Off unless the parameter row's code_version is CODE_VERSION_V41 (no new table columns), see anchor_params.
ANCHOR_DEFAULTS = dict(anchor_r0_mult=2.0, anchor_min_inlier=0.7, anchor_rim_rel=0.4, anchor_rim_n=600, anchor_rim_min_inlier=0.8,
                       anchor_med_n=120)


def anchor_params(cfg):
    """Fallback-anchor parameters for a PupilEllipseImageFitParameter row: ANCHOR_DEFAULTS when its code_version is
    CODE_VERSION_V41, else None (v2 / v3 / v4 rows reproduce their code exactly)."""
    return dict(ANCHOR_DEFAULTS) if cfg.get("code_version") == CODE_VERSION_V41 else None


# Which code a PupilEllipseImageFitParameter row runs, in one place. Stored sets: 10 = v3, 11 = v4, 12 = v4.1.
#   v3   : DEFAULTS + GATE_DEFAULTS + eye model; the v4 columns are NULL.
#   v4   : v3 + the v4 columns set (guide-scaled rays, retry, weak-fit check), switched on by their values.
#   v4.1 : v4 + anchored fallback centre, switched on by code_version == CODE_VERSION_V41.
# The v4 features follow the column values (not the code_version string) because that is how sets 11 and 12 were
# computed; changing the switch would not change any stored row, but it would make a v3 row with v4 values behave
# differently from the code that the rows were made with.
KNOWN_CODE_VERSIONS = (CODE_VERSION, CODE_VERSION_V4, CODE_VERSION_V41)


def fit_settings(cfg):
    """(params, v4, anchor) for a PupilEllipseImageFitParameter row:
    params: the per-frame fit / v2-accept parameters present in the row (DEFAULTS keys);
    v4: the v4 columns that are set (empty for a v3 row);
    anchor: the v4.1 anchor parameters, or None (see anchor_params).
    Raises for a code_version this module does not implement, so a parameter row made for other code fails loudly."""
    if cfg.get("code_version") not in KNOWN_CODE_VERSIONS:
        raise ValueError(f"code_version {cfg.get('code_version')!r} is not implemented here; known: {KNOWN_CODE_VERSIONS}")
    params = {k: cfg[k] for k in DEFAULTS if k in cfg}
    v4 = {k: cfg[k] for k in V4_DEFAULTS if cfg.get(k) is not None}
    return params, v4, anchor_params(cfg)


def guide_from_markers(X, Y, ok, min_markers=3, r0_min_markers=7):
    """Guide centers and radius from pupil markers.

    X, Y, ok: (n_markers, n_frames); ok marks valid markers.
    Returns gx, gy (per-frame median of the valid markers), gok (>= min_markers valid), R0 (session median of the
    per-frame median marker radius on frames with >= r0_min_markers valid markers).
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float); ok = np.asarray(ok, bool)
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sw:
        sw.filter(RuntimeWarning)
        Xv, Yv = np.where(ok, X, np.nan), np.where(ok, Y, np.nan)
        gx = np.nanmedian(Xv, 0); gy = np.nanmedian(Yv, 0)
        rad = np.nanmedian(np.hypot(Xv - gx, Yv - gy), 0)
    gok = ok.sum(0) >= min_markers
    R0 = float(np.nanmedian(rad[ok.sum(0) >= r0_min_markers]))
    return gx, gy, gok, R0


def guide_radius(X, Y, ok, gx, gy, min_markers=3):
    """Per-frame guide radius: median distance of the valid markers from the guide centre (NaN with < min_markers),
    the per-frame quantity whose session median (over frames with >= 7 markers) is R0."""
    X = np.asarray(X, float); Y = np.asarray(Y, float); ok = np.asarray(ok, bool)
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sw:
        sw.filter(RuntimeWarning)
        rad = np.nanmedian(np.hypot(np.where(ok, X, np.nan) - gx, np.where(ok, Y, np.nan) - gy), 0)
    rad[ok.sum(0) < min_markers] = np.nan
    return rad


def rolling_guide_radius(grad, fps, guide_r_window_s=0.5, **_):
    """Centred rolling median of the per-frame guide radius over guide_r_window_s (NaN where the window has fewer than
    a quarter of its frames with a guide radius)."""
    win = int(round(guide_r_window_s * fps)) | 1
    return pd.Series(np.asarray(grad, float)).rolling(win, center=True, min_periods=max(1, win // 4)).median().to_numpy()


def ray_end_px(grad, R0, fps, r_max=1.8, r_max_guide=None, guide_r_window_s=0.5, guide_r_engage=None, **_):
    """Per-frame ray end (px): r_max_guide * rho on frames whose rolling guide radius rho exceeds guide_r_engage * R0
    (None: exceeds R0) and where that is longer than r_max * R0; else r_max * R0. None when r_max_guide is None (v3:
    one ray length for the session). Frames without a rolling guide radius keep r_max * R0."""
    if r_max_guide is None or grad is None:
        return None
    rho = rolling_guide_radius(grad, fps, guide_r_window_s)
    eng = 1.0 if guide_r_engage is None else float(guide_r_engage)
    with np.errstate(invalid="ignore"):
        on = np.isfinite(rho) & (rho > eng * R0)
        return np.where(on, np.fmax(r_max * R0, r_max_guide * rho), r_max * R0)


class RimFitter:
    """Per-frame rim fit with the ray geometry fixed by the session radius R0 (reference: t8_pupil_image.fit_frame)."""

    def __init__(self, R0, n_rays=48, r_min=0.35, r_max=1.8, r_step=0.5, ransac_draws=60, inlier_tol=2.0, seed=0, **_):
        self.R0 = float(R0); self.n_rays = int(n_rays); self.draws = int(ransac_draws); self.tol = float(inlier_tol)
        self.ANG = np.linspace(0, 2 * np.pi, self.n_rays, endpoint=False)
        self.RS = np.arange(r_min * self.R0, r_max * self.R0, r_step)
        self.r_start, self.r_step, self.r_end0 = r_min * self.R0, float(r_step), r_max * self.R0
        self.rng = np.random.default_rng(seed)
        self.rng_retry = np.random.default_rng(int(seed) + 1)   # v4 retry: own stream, the main stream stays as in v3

    def fit(self, g, cx, cy, r_end=None, r_retry=None, min_inlier_frac=0.6):
        """Rim fit of one frame (fit_once), and with r_retry (v4) a second fit with rays to r_retry px when the first
        has no fit or inlier_frac < min_inlier_frac; the retry (own random stream) replaces the first fit if its
        inlier fraction is higher. Returns the fit_once tuple plus a flag: 1 = the retry was used."""
        r = self.fit_once(g, cx, cy, r_end=r_end)
        self.first = r                                          # the first fit (fit_video moves its fallback centre with it)
        if r_retry is not None and not (r[5] >= min_inlier_frac):
            r2 = self.fit_once(g, cx, cy, r_end=r_retry, rng=self.rng_retry)
            if np.isfinite(r2[0]) and not (r2[5] <= r[5]):
                return r2 + (1,)
        return r + (0,)

    def fit_once(self, g, cx, cy, r_end=None, rng=None):
        """g: blurred grayscale frame (float32). Returns (cx, cy, a, b, angle_deg, inlier_frac, rim_step); a, b = cv2
        semi-axes (width/2, height/2), angle = cv2's rotation of the first axis in degrees (image coordinates).
        r_end: optional ray end for this frame (px); rays longer than the session rays only when r_end > r_max * R0
        (v4 guide-scaled rays), else exactly the session rays. rng: random stream (default: the main one)."""
        R0, ANG, RS = self.R0, self.ANG, self.RS
        rng = self.rng if rng is None else rng
        if r_end is not None and r_end > self.r_end0:
            RS = np.arange(self.r_start, r_end, self.r_step)
        xs = (cx + np.cos(ANG)[:, None] * RS[None]).astype(np.float32); ys = (cy + np.sin(ANG)[:, None] * RS[None]).astype(np.float32)
        prof = cv2.remap(g, xs, ys, cv2.INTER_LINEAR)
        k = np.argmin(np.diff(prof, axis=1), 1)                                # strongest bright->dark step per ray
        r_edge = RS[k] + 0.25
        pts = np.stack([cx + np.cos(ANG) * r_edge, cy + np.sin(ANG) * r_edge], 1).astype(np.float32)
        best, bin_ = None, None
        for _ in range(self.draws):                                            # RANSAC
            s = rng.choice(self.n_rays, 6, replace=False)
            try:
                e = cv2.fitEllipse(pts[s])
            except cv2.error:
                continue
            (ex, ey), (A, B), th = e
            if not (0.3 * R0 < A / 2 < 3 * R0 and 0.3 * R0 < B / 2 < 3 * R0):
                continue
            c, sn = np.cos(np.radians(th)), np.sin(np.radians(th))
            u = (pts[:, 0] - ex) * c + (pts[:, 1] - ey) * sn; v = -(pts[:, 0] - ex) * sn + (pts[:, 1] - ey) * c
            inl = np.abs(np.sqrt((u / (A / 2)) ** 2 + (v / (B / 2)) ** 2) - 1) * (A + B) / 4 < self.tol
            if bin_ is None or inl.sum() > bin_.sum():
                best, bin_ = e, inl
        if best is None or bin_.sum() < 6:
            return (np.nan,) * 7
        try:
            (ex, ey), (A, B), th = cv2.fitEllipse(pts[bin_])
        except cv2.error:
            return (np.nan,) * 7
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False); c_, s_ = np.cos(np.radians(th)), np.sin(np.radians(th))

        def band(delta):
            a, b = max(A / 2 + delta, 1), max(B / 2 + delta, 1); u, v = a * np.cos(t), b * np.sin(t)
            return cv2.remap(g, (ex + u * c_ - v * s_).astype(np.float32)[None], (ey + u * s_ + v * c_).astype(np.float32)[None],
                             cv2.INTER_LINEAR)[0]
        return ex, ey, A / 2, B / 2, th, float(bin_.mean()), float(np.median(band(-3)) - np.median(band(+3)))


def fit_video(video_path, gx, gy, gok, R0, blur_sigma=1.5, progress_every=0, log=print, guide_r=None, anchor=None,
              **params):
    """Sequential pass over the video. Returns res (n_marker_frames, 7) float32 [cx, cy, a, b, angle_deg, inlier_frac,
    rim_step] (NaN where not fitted) and a dict with the frame counts, the video's fps and the number of frames fitted
    with guide-scaled (longer) rays. guide_r: per-frame guide radius (guide_radius), used only with r_max_guide (v4).
    anchor: v4.1 fallback-anchor parameters (anchor_params / ANCHOR_DEFAULTS); None = v4 behaviour."""
    T = len(gx)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {video_path}")
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = float(cap.get(cv2.CAP_PROP_FPS))
    fitter = RimFitter(R0, **params)
    rp = {k: params[k] for k in ("r_max", "r_max_guide", "guide_r_window_s", "guide_r_engage") if params.get(k) is not None}
    r_end = ray_end_px(guide_r, R0, fps, **rp)
    n_long = 0 if r_end is None else int((r_end[:min(T, n_vid)] > fitter.r_end0).sum())
    r_retry = None
    if params.get("r_max_retry") is not None:          # retry rays: r_max_retry x max(R0, rolling guide radius)
        rho = (rolling_guide_radius(guide_r, fps, params.get("guide_r_window_s") or 0.5) if guide_r is not None
               else np.full(T, np.nan))
        r_retry = float(params["r_max_retry"]) * np.where(np.isfinite(rho), np.fmax(R0, rho), R0)
    min_inl = float(params.get("min_inlier_frac", DEFAULTS["min_inlier_frac"]))
    retried = np.zeros(T, bool)
    reanchored = np.zeros(T, bool)
    divergent = np.zeros(T, bool)
    res = np.full((T, 7), np.nan, np.float32)
    last = (np.nanmedian(gx), np.nanmedian(gy)); n_read = 0
    if anchor is not None:                            # v4.1: anchored fallback centre (see CODE_VERSION_V41)
        from collections import deque
        an = {**ANCHOR_DEFAULTS, **anchor}
        a_rad = float(an["anchor_r0_mult"]) * float(R0)
        a_rim = deque(maxlen=int(an["anchor_rim_n"]))
        a_inl = max(min_inl, float(an["anchor_min_inlier"])); a_rim_inl = float(an["anchor_rim_min_inlier"])
        a_cx = deque(maxlen=int(an["anchor_med_n"])); a_cy = deque(maxlen=int(an["anchor_med_n"]))
        a_lo = float(params.get("axis_min", DEFAULTS["axis_min"])) * float(R0)
        a_hi = float(params.get("axis_max", DEFAULTS["axis_max"])) * float(R0)
        anc = last
        last4 = last                                  # v4's fallback centre (shadow chain), see the divergent frames
    for f in range(min(T, n_vid)):
        ok, img = cap.read()
        if not ok:
            break
        n_read += 1
        g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), blur_sigma)
        c = (gx[f], gy[f]) if gok[f] else last
        if anchor is not None and not gok[f] and np.hypot(last[0] - anc[0], last[1] - anc[1]) > a_rad:
            c = anc; reanchored[f] = True             # v4 start too far from the last passing fit: start at the anchor
        fk = dict(r_end=None if r_end is None else r_end[f], r_retry=None if r_retry is None else r_retry[f], min_inlier_frac=min_inl)
        if anchor is not None and not gok[f] and c != last4:
            # v4.1 divergent frame (its start differs from v4's): also run v4's fit from v4's start (shadow chain
            # last4) and leave both random streams where v4 leaves them, so that every frame whose start equals v4's
            # (all guide frames) draws the same RANSAC samples as in v4 and is bit-identical to it.
            divergent[f] = True
            st = (fitter.rng.bit_generator.state, fitter.rng_retry.bit_generator.state)
            fitter.fit(g, *last4, **fk)
            if np.isfinite(fitter.first[0]):
                last4 = (fitter.first[0], fitter.first[1])
            st4 = (fitter.rng.bit_generator.state, fitter.rng_retry.bit_generator.state)
            fitter.rng.bit_generator.state, fitter.rng_retry.bit_generator.state = st
            out = fitter.fit(g, *c, **fk)
            fitter.rng.bit_generator.state, fitter.rng_retry.bit_generator.state = st4
        else:
            out = fitter.fit(g, *c, **fk)
        res[f] = out[:7]; retried[f] = bool(out[7])
        mv = fitter.first if retried[f] else res[f]   # v4: the fallback centre moves with the FIRST fit, as in v3; a retry
        if np.isfinite(mv[0]):                        # fit moving it let the centre walk off the eye where markers are lost
            last = (mv[0], mv[1])
        if anchor is not None and not divergent[f]:
            last4 = last                              # same start, same random draws: v4's first fit is this one
        if anchor is not None:                        # v4.1: the anchor moves only with fits that pass the online test
            r = res[f]
            passes = bool(np.isfinite(r[0]) and r[5] >= a_inl and a_lo < r[2] < a_hi and a_lo < r[3] < a_hi
                          and r[6] >= (float(an["anchor_rim_rel"]) * float(np.median(a_rim)) if a_rim else 0.0))
            if passes or gok[f]:
                a_cx.append(float(r[0]) if passes else float(gx[f])); a_cy.append(float(r[1]) if passes else float(gy[f]))
                anc = (float(np.median(a_cx)), float(np.median(a_cy)))
                if passes and r[5] >= a_rim_inl:
                    a_rim.append(float(r[6]))
        if progress_every and f % progress_every == 0:
            log(f"image rim fit: frame {f} / {T}")
    cap.release()
    return res, dict(n_frames_markers=T, n_frames_video=n_vid, n_frames_read=n_read, fps=fps, n_frames_long_rays=n_long,
                n_frames_retry=int(retried.sum()), retried=retried, n_frames_reanchored=int(reanchored.sum()),
                reanchored=reanchored, n_frames_divergent=int(divergent.sum()), divergent=divergent)


def session_medians(res):
    fin = np.isfinite(res[:, 0])
    return float(np.nanmedian(res[fin, 6])), float(np.nanmedian(res[fin, 2]))     # rim step, semi-axis a


def accept_v2(res, rim_step_median=None, axis_median=None, min_inlier_frac=0.6, min_rim_step_rel=0.4, axis_min=0.25,
              axis_max=3.0, **_):
    """v2 accept rule (reference: t8_chain.py). Medians default to the session medians over the fitted frames."""
    con, ax = session_medians(res)
    con = con if rim_step_median is None else rim_step_median; ax = ax if axis_median is None else axis_median
    return (np.isfinite(res[:, 0]) & (res[:, 5] >= min_inlier_frac) & (res[:, 6] >= min_rim_step_rel * con)
            & (res[:, 2] > axis_min * ax) & (res[:, 2] < axis_max * ax) & (res[:, 3] > axis_min * ax) & (res[:, 3] < axis_max * ax))


def temporal_brief_jumps(cx, cy, fps, tc_dist_px=25.0, tc_window_s=2.0, tc_max_brief_s=0.25, **_):
    """Accepted centers (NaN = not accepted) farther than tc_dist_px from the centred rolling median over
    tc_window_s, in excursions shorter than tc_max_brief_s. At 60 fps: window 121 frames, brief < 15 frames
    (reference: t8_chain.py image_tc)."""
    win = int(round(tc_window_s * fps)) | 1; brief_n = int(round(tc_max_brief_s * fps))
    rx = pd.Series(cx).rolling(win, center=True, min_periods=20).median().to_numpy()
    ry = pd.Series(cy).rolling(win, center=True, min_periods=20).median().to_numpy()
    far = np.isfinite(cx) & (np.hypot(cx - rx, cy - ry) > tc_dist_px)
    runs = np.diff(np.r_[0, far.astype(int), 0]); st, en = np.where(runs == 1)[0], np.where(runs == -1)[0]
    brief = np.zeros(len(far), bool)
    for a, b in zip(st, en):
        if b - a < brief_n:
            brief[a:b] = True
    return brief


def long_axis_angle_rad(a, b, angle_deg):
    """Direction of the long axis in radians, image coordinates (the convention of PupilEllipseFitting.fit_ellipse's
    angle_from_x: atan2 of the long-axis eigenvector). cv2's angle is the rotation of its first axis (semi-axis a)."""
    th = np.radians(angle_deg)
    ang = np.where(a >= b, th, th + np.pi / 2)
    return np.arctan2(np.sin(ang), np.cos(ang))


def weak_uncorroborated(res, acc, R0, fps, weak_inlier_frac=None, weak_ref_min_inlier=0.8, weak_ref_window_s=0.5,
                        weak_ref_dist_max=0.3, weak_ref_axis_min=0.8, weak_ref_axis_max=1.25, **_):
    """v4: accepted fits (acc) with inlier_frac < weak_inlier_frac that confident fits nearby in time do not
    corroborate. Reference = centred rolling median over +-weak_ref_window_s of the accepted fits with inlier_frac >=
    weak_ref_min_inlier (a weak fit is never in its own reference). Uncorroborated: no confident fit in the window, or
    centre farther than weak_ref_dist_max * R0 from the reference centre, or semi-major / reference semi-major outside
    [weak_ref_axis_min, weak_ref_axis_max]. All False when weak_inlier_frac is None (v3).
    (A nearest-confident-neighbour reference, tried as an alternative to tolerate saccades, kept half of the visually
    wrong weak fits; the window median rejected all of them, at the cost of ~1 in 8 good weak fits.)"""
    T = len(res)
    if weak_inlier_frac is None:
        return np.zeros(T, bool)
    a = np.fmax(res[:, 2], res[:, 3]).astype(float)
    conf = acc & (res[:, 5] >= weak_ref_min_inlier)
    ref = pd.DataFrame({"x": np.where(conf, res[:, 0], np.nan), "y": np.where(conf, res[:, 1], np.nan), "a": np.where(conf, a, np.nan)})
    win = 2 * int(round(weak_ref_window_s * fps)) + 1
    rm = ref.rolling(win, center=True, min_periods=1).median().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.hypot(res[:, 0] - rm[:, 0], res[:, 1] - rm[:, 1]) / float(R0)
        ratio = a / rm[:, 2]
        ok = (dist <= weak_ref_dist_max) & (ratio >= weak_ref_axis_min) & (ratio <= weak_ref_axis_max)
        weak = acc & (res[:, 5] < weak_inlier_frac)
    return weak & ~ok


def accepted_centers(res, fps, params=None, R0=None):
    """res -> (accepted & temporally consistent mask, accept mask, brief-jump mask). accept = v2 accept rule, and with
    v4's weak_inlier_frac (needs R0) also not weak_uncorroborated (before the temporal check, so a rejected weak fit
    does not enter its rolling median)."""
    p = {**DEFAULTS, **V4_DEFAULTS, **{k: v for k, v in (params or {}).items() if v is not None}}
    acc = accept_v2(res, **p)
    if p.get("weak_inlier_frac") is not None:
        if R0 is None:
            raise ValueError("weak_inlier_frac needs R0")
        acc = acc & ~weak_uncorroborated(res, acc, R0, fps, **p)
    c = np.where(acc[:, None], res[:, :2], np.nan).astype(float)
    brief = temporal_brief_jumps(c[:, 0], c[:, 1], fps, **p)
    return acc & ~brief, acc, brief


def spatial_gate(cx, cy, keep, R0, gate_min_px=100.0, gate_r0_mult=3.0, **_):
    """Frames in `keep` whose centre is farther than max(gate_min_px, gate_r0_mult * R0) from the median of the kept
    centres -> (rejected mask, gate radius px). Single pass, as in t8_chain.py."""
    gate = max(float(gate_min_px), float(gate_r0_mult) * float(R0))
    k = keep & np.isfinite(cx) & np.isfinite(cy)
    if not k.any():
        return np.zeros(len(cx), bool), gate
    mx, my = np.median(cx[k]), np.median(cy[k])
    with np.errstate(invalid="ignore"):
        far = k & (np.hypot(cx - mx, cy - my) > gate)
    return far, gate


def accepted_centers_gated(res, fps, R0, params=None):
    """res -> (kept = accept & temporally consistent & inside the spatial gate, accept, brief-jump, gate-rejected,
    gate radius px). accept: see accepted_centers."""
    p = {**DEFAULTS, **GATE_DEFAULTS, **(params or {})}
    keep, acc, brief = accepted_centers(res, fps, p, R0=R0)
    far, gate = spatial_gate(res[:, 0].astype(float), res[:, 1].astype(float), keep, R0, **p)
    return keep & ~far, acc, brief, far, gate


# ======================================================================================
# Eye angles: pooled 3D eye model (reference: v1-drift-audit scripts/t13_eye_model.py)
# ======================================================================================
# Orthographic camera (a ~3 mm eye at ~10 cm). The eye's rotation centre projects to image point c; the pupil centre
# moves on a sphere of radius R (px). For gaze (theta, phi) relative to the camera axis
#     p = c + R (sin theta, cos theta sin phi)            image position of the pupil centre
#     b/a = cos(omega) = sqrt(1 - |p - c|^2 / R^2)        minor/major axis ratio of the pupil ellipse
#     minor-axis direction parallel to (p - c)            the pupil disc tilts towards the camera centre
# The production estimator (PupilEllipseFitting.make) uses only the third relation, on frames with b/a < 0.85: with
# small head-fixed eye movements and accurate (near-circular) ellipses few frames qualify and their axes are nearly
# parallel, so the camera centre and scale are ill-conditioned. Here c and R are fitted jointly from all accepted
# frames, with a weak anatomical prior R = k * palpebral fissure length (k = 0.42 +- 0.15: mouse effective pupil
# rotation radius ~1.25 mm, Stahl et al. 2000; fissure ~3 mm), robust (IRLS, Huber weights).

EYE_MODEL_DEFAULTS = dict(e3d_k0=1.25 / 3.0, e3d_k_sd=0.15, e3d_prior_weight=0.02, e3d_huber_k=0.05, e3d_irls_iter=4,
                          e3d_n_frames=30000, corner_llh=0.7, corner_min_frac=0.2, static_corner_llh=0.9, static_min_frames=50)
# v3 additions (read with defaults so that v2 parameter rows keep working):
#   fissure_prior 'animal_median': the prior length is the median over the animal's sessions of the per-session
#     fissure (valid range fissure_min_px..fissure_max_px); the face camera's zoom and distance are fixed, so the
#     per-session variation (0.8-3.7 % CV) is corner-tracking noise. 'session' = v2 behaviour.
#   e3d_ms_deg / e3d_ms_n: multi-start grid of the mean gaze offset (theta0, phi0) in [-deg, +deg], n x n starts; the
#     start with the lowest robust objective wins (single-start fits landed in local optima at +-13 / 22 / 28 deg).
EYE_MODEL_V3_DEFAULTS = dict(fissure_prior="animal_median", fissure_min_px=150.0, fissure_max_px=280.0,
                             e3d_ms_deg=30.0, e3d_ms_n=5)


def ellipse_shape(res):
    """res -> (q = minor/major axis ratio, minor-axis direction in radians, semi-major axis)."""
    A2, B2, th = res[:, 2], res[:, 3], np.radians(res[:, 4])
    a = np.maximum(A2, B2); b = np.minimum(A2, B2)
    minor = np.where(A2 <= B2, th, th + np.pi / 2)   # cv2 angle = first ("width") axis; minor axis if it is the shorter
    return b / a, minor, a


def fissure_length(nose_x, nose_y, nose_llh, lat_x, lat_y, lat_llh, corner_llh=0.7, corner_min_frac=0.2,
                   static_corner_llh=0.9, static_min_frames=50, **_):
    """Palpebral fissure length (px) for the eye-model prior, and its source.
    per-frame: median corner distance over frames with both corners above corner_llh, if >= corner_min_frac of frames;
    static: distance between the session medians of each corner over its own frames >= static_corner_llh (valid
    because the head is fixed; e.g. ROS-1985, whose nose corner is rarely co-tracked); else (nan, 'none')."""
    both = (nose_llh > corner_llh) & (lat_llh > corner_llh)
    if both.mean() >= corner_min_frac:
        return float(np.median(np.hypot(lat_x[both] - nose_x[both], lat_y[both] - nose_y[both]))), "per_frame"
    hn, hl = nose_llh >= static_corner_llh, lat_llh >= static_corner_llh
    if hn.sum() > static_min_frames and hl.sum() > static_min_frames:
        return float(np.hypot(np.median(lat_x[hl]) - np.median(nose_x[hn]), np.median(lat_y[hl]) - np.median(nose_y[hn]))), "static"
    return float("nan"), "none"


def fit_eye_model(px, py, q, beta, L, amed, e3d_k0=1.25 / 3.0, e3d_k_sd=0.15, e3d_prior_weight=0.02, e3d_huber_k=0.05,
                  e3d_irls_iter=4, x0=None, **_):
    """Pooled 3D eye-model fit -> (cx, cy, R). L: fissure length (prior on R) or nan/None (no prior; R bounded by the
    pupil size). Robust loss on the data residuals only (IRLS with Huber weights); the prior enters unweighted.
    x0: optional start (cx, cy, R); default (median pupil position, k0 * L)."""
    from scipy.optimize import least_squares
    # float64 throughout: with float32 inputs the finite-difference Jacobian step falls below float32 resolution and
    # the fit stays at its starting point (this silently pinned R to the prior in the audit's first T13 run)
    px, py, q, beta = (np.asarray(v, np.float64) for v in (px, py, q, beta))
    L = None if (L is None or not np.isfinite(L)) else float(L)
    lo, hi = (0.15 * L, 1.5 * L) if L else (1.5 * amed, 15 * amed)
    w = np.ones(2 * len(px))

    def resid(v):
        cx, cy, R = v; dx, dy = px - cx, py - cy; d2 = dx * dx + dy * dy
        pred = np.sqrt(np.clip(1 - d2 / (R * R), 0, 1)); gam = np.arctan2(dy, dx)
        r = np.r_[q - pred, (1 - q) * np.sin(2 * (beta - gam))] * np.sqrt(w)
        prior = np.array([(R / L - e3d_k0) / e3d_k_sd]) * np.sqrt(len(px)) * e3d_prior_weight if L else np.zeros(0)
        return np.r_[r, prior]
    x = np.array([np.median(px), np.median(py), np.clip(e3d_k0 * L if L else 3 * amed, lo * 1.01, hi * 0.99)])
    if x0 is not None:
        x = np.array([float(x0[0]), float(x0[1]), np.clip(float(x0[2]), lo * 1.01, hi * 0.99)])
    for _ in range(int(e3d_irls_iter)):
        x = least_squares(resid, x, loss="linear", bounds=([-np.inf, -np.inf, lo], [np.inf, np.inf, hi])).x
        raw = resid(x)[: 2 * len(px)] / np.sqrt(w)
        w = np.where(np.abs(raw) <= e3d_huber_k, 1.0, e3d_huber_k / np.abs(raw))
    return x


def eye_angles(px, py, cx, cy, R):
    """Gaze relative to the camera axis, radians (the production convention): theta = asin(dx/R), phi = asin(dy/(R cos theta))."""
    th = np.arcsin(np.clip((px - cx) / R, -1, 1)); ph = np.arcsin(np.clip((py - cy) / (R * np.cos(th)), -1, 1))
    return th, ph


def eye_model_objective(x, px, py, q, beta, L, e3d_k0=1.25 / 3.0, e3d_k_sd=0.15, e3d_prior_weight=0.02,
                        e3d_huber_k=0.05, **_):
    """Robust objective used to rank multi-start solutions: mean Huber loss of the data residuals (threshold
    e3d_huber_k, the IRLS threshold) plus the squared prior term / n. Lower is better."""
    px, py, q, beta = (np.asarray(v, np.float64) for v in (px, py, q, beta))
    cx, cy, R = x; dx, dy = px - cx, py - cy
    pred = np.sqrt(np.clip(1 - (dx * dx + dy * dy) / (R * R), 0, 1)); gam = np.arctan2(dy, dx)
    r = np.abs(np.r_[q - pred, (1 - q) * np.sin(2 * (beta - gam))]); k = float(e3d_huber_k)
    hub = np.where(r <= k, 0.5 * r * r, k * (r - 0.5 * k))
    L = None if (L is None or not np.isfinite(L)) else float(L)
    prior = ((R / L - e3d_k0) / e3d_k_sd * e3d_prior_weight) ** 2 if L else 0.0
    return float(hub.mean() + 0.5 * prior)


def fit_eye_model_multistart(px, py, q, beta, L, amed, e3d_ms_deg=30.0, e3d_ms_n=5, **p):
    """Multi-start pooled eye-model fit -> ((cx, cy, R), info). Starts: rotation centre placed so that the mean pupil
    position has gaze offset (theta0, phi0) on an e3d_ms_n x e3d_ms_n grid in [-e3d_ms_deg, +e3d_ms_deg] (includes 0,
    the v2 single start, when e3d_ms_n is odd); R starts at k0 * L. Each start is fitted with fit_eye_model; the lowest
    eye_model_objective wins. info: objective of the winner, of the v2 single start, and of the best solution whose
    offset differs from the winner's by > 5 deg (e3d_objective_far / e3d_objective near 1 = a near-equivalent distant
    optimum: the absolute offset is not identified)."""
    px, py, q, beta = (np.asarray(v, np.float64) for v in (px, py, q, beta))
    Lf = None if (L is None or not np.isfinite(L)) else float(L)
    k0 = p.get("e3d_k0", 1.25 / 3.0)
    R0 = k0 * Lf if Lf else 3 * amed
    mx, my = float(np.mean(px)), float(np.mean(py))
    grid = np.radians(np.linspace(-e3d_ms_deg, e3d_ms_deg, int(e3d_ms_n))) if int(e3d_ms_n) > 1 else np.zeros(1)
    sols = []
    x_single = fit_eye_model(px, py, q, beta, L, amed, **p)
    sols.append((x_single, eye_model_objective(x_single, px, py, q, beta, L, **p), "single"))
    for t0 in grid:
        for f0 in grid:
            c0 = (mx - R0 * np.sin(t0), my - R0 * np.cos(t0) * np.sin(f0), R0)
            x = fit_eye_model(px, py, q, beta, L, amed, x0=c0, **p)
            sols.append((x, eye_model_objective(x, px, py, q, beta, L, **p), "grid"))
    best = min(sols, key=lambda s_: s_[1])

    def offset(x):
        th, ph = eye_angles(np.array([mx]), np.array([my]), *x); return np.degrees(th[0]), np.degrees(ph[0])
    ob = offset(best[0])
    far = [s_[1] for s_ in sols if np.hypot(*np.subtract(offset(s_[0]), ob)) > 5.0]
    info = dict(e3d_objective=best[1], e3d_objective_single=sols[0][1],
                e3d_objective_far=min(far) if far else np.nan, e3d_n_starts=len(sols))
    return best[0], info


def animal_fissure(values, fissure_min_px=150.0, fissure_max_px=280.0, **_):
    """Median of the valid per-session fissure lengths of one animal (nan if none), and the number used."""
    v = np.asarray([x for x in values if x is not None], float)
    v = v[np.isfinite(v) & (v >= fissure_min_px) & (v <= fissure_max_px)]
    return (float(np.median(v)) if len(v) else float("nan")), int(len(v))


def eye_model_from_results(results, L, L_source, seed, cfg, L_session=None, L_session_source=None, n_sessions=None):
    """Pooled 3D eye model on the kept image-fit ellipses (results: the make() layout; NaN = not kept) -> (cam_center,
    eye_center, scale, theta, phi, info). Shared by PupilEllipseFitting.make() and the stored-ellipse update pass (recompute_from_stored)
    so both use identical code. Multi-start when cfg has e3d_ms_n (v3), else the v2 single start."""
    p = {k: cfg[k] for k in EYE_MODEL_DEFAULTS}
    px, py = np.asarray(results["X0+ir"], float), np.asarray(results["Y0+ir"], float)
    la, sa, ang = (np.asarray(results[k], float) for k in ("long_axis", "short_axis", "angle_from_x"))
    q, beta = sa / la, ang + np.pi / 2                     # minor/major ratio; minor-axis direction
    idx = np.where(np.isfinite(px) & np.isfinite(py) & np.isfinite(q) & np.isfinite(beta))[0]
    if len(idx) < 100:
        raise ValueError(f"only {len(idx)} accepted frames for the eye model")
    rng = np.random.default_rng(int(seed))
    sub = rng.choice(idx, min(int(p["e3d_n_frames"]), len(idx)), replace=False)
    amed = float(np.nanmedian(la[idx]))
    ms = {}
    if cfg.get("e3d_ms_n") is not None and cfg.get("e3d_ms_deg") is not None:
        (cx, cy, R), ms = fit_eye_model_multistart(px[sub], py[sub], q[sub], beta[sub], L, amed,
                                                       e3d_ms_deg=float(cfg["e3d_ms_deg"]), e3d_ms_n=int(cfg["e3d_ms_n"]), **p)
    else:
        cx, cy, R = fit_eye_model(px[sub], py[sub], q[sub], beta[sub], L, amed, **p)
    th_cam, ph_cam = eye_angles(px, py, cx, cy, R)
    th_cam[~np.isfinite(px)] = np.nan; ph_cam[~np.isfinite(px)] = np.nan
    ex, ey = float(np.nanmean(px[idx])), float(np.nanmean(py[idx]))
    th_off, ph_off = (float(v) for v in eye_angles(np.array([ex]), np.array([ey]), cx, cy, R))
    T = len(px)
    fin = lambda v: None if v is None or not np.isfinite(v) else float(v)   # noqa: E731
    info = dict(angle_estimator="eye_model_3d", e3d_cx=float(cx), e3d_cy=float(cy), e3d_r=float(R),
                fissure_px=fin(L), fissure_source=L_source,
                theta_offset=th_off, phi_offset=ph_off, n_frames_eye_model=int(len(sub)))
    if cfg.get("fissure_prior") is not None:
        info.update(fissure_session_px=fin(L_session), fissure_session_source=L_session_source, fissure_n_sessions=n_sessions)
    if ms:
        info.update(e3d_objective=fin(ms["e3d_objective"]), e3d_objective_single=fin(ms["e3d_objective_single"]),
                    e3d_objective_far=fin(ms["e3d_objective_far"]), e3d_n_starts=int(ms["e3d_n_starts"]))
    return (np.tile([cx, cy], (T, 1)), np.tile([ex, ey], (T, 1)), float(R), th_cam - th_off, ph_cam - ph_off, info)


GEOM_FIELDS = ("X0", "Y0", "a", "b", "long_axis", "short_axis", "angle_to_x", "angle_from_x", "X0_in", "Y0_in", "X0+ir", "Y0+ir")


def recompute_from_stored(ellipse_dict, r0, cfg, thres_ellipticity, L, L_source, n_sessions=None, L_session=None,
                          L_session_source=None):
    """v3 update of a stored v2 image-fit row, no video: spatial gate on the stored kept centres, then the eye model.
    ellipse_dict: the stored per-frame list (v2 layout: NaN geometry where not kept). Returns dict with
      far (gate-rejected frames), gate, ellipse_dict (new list, or None if the gate rejects nothing), diameter,
      cam_center, eye_center, scale, theta, phi, and imagefit (the ImageFit fields that change).
    The gate is a function of the kept centres only, so gate(stored v2 keep) == gate inside make() (same frames)."""
    T = len(ellipse_dict)
    col = lambda k: np.fromiter((d[k] for d in ellipse_dict), float, T)   # noqa: E731
    cx, cy = col("X0+ir"), col("Y0+ir")
    keep = np.fromiter((bool(d["accepted"]) for d in ellipse_dict), bool, T)
    if (np.isfinite(cx) != keep).any():
        raise ValueError("stored ellipses: finite centre != accepted flag (not a v2 image-fit row?)")
    gp = {k: cfg[k] for k in GATE_DEFAULTS}
    far, gate = spatial_gate(cx, cy, keep, r0, **gp)
    keep_new = keep & ~far
    res = {k: np.where(keep_new, col(k), np.nan) for k in ("X0+ir", "Y0+ir", "long_axis", "short_axis", "angle_from_x", "X0_in", "Y0_in")}
    cam_center, eye_center, scale, theta, phi, info = eye_model_from_results(
        res, L, L_source, int(cfg["seed"]), cfg, L_session=L_session, L_session_source=L_session_source, n_sessions=n_sessions)
    inl = col("inlier_frac")
    with np.errstate(invalid="ignore", divide="ignore"):
        nice = (keep_new & (inl >= cfg["nice_min_inlier_frac"]) & np.isfinite(res["X0_in"]) & np.isfinite(res["Y0_in"])
                & (res["short_axis"] / res["long_axis"] < thres_ellipticity))
    new_dict = None
    if far.any():
        nan = float("nan"); fi = set(np.where(far)[0].tolist())
        new_dict = [({**d, **{k: nan for k in GEOM_FIELDS}, "accepted": False, "rejected_spatial": True} if i in fi
                     else {**d, "rejected_spatial": False}) for i, d in enumerate(ellipse_dict)]
    imagefit = dict(info, frac_accepted=float(keep_new.mean()), n_frames_rejected_spatial=int(far.sum()), gate_px=float(gate),
                    n_nice_frames=int(nice.sum()), code_version=cfg["code_version"])
    return dict(far=far, gate=gate, ellipse_dict=new_dict, diameter=res["long_axis"], cam_center=cam_center,
                eye_center=eye_center, scale=scale, theta=theta, phi=phi, imagefit=imagefit)
