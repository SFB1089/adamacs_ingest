"""Image-based corneal-glint (IR reflection) tracking for the head-fixed face camera.

Why: the DLC ``IR`` marker sits on average 2.3 px above (range 1.4–5.0) and 1.2 px left of the glint in the image, and
that offset drifts within a session (5-block range 0.1–1.4 px) and between sessions (0.2–3 px within an animal). The
pupil–glint gaze vector then inherits the bias (v1-drift-audit t31/t31b).

Detector (``detect``): the glint is a small compact bright spot, sometimes saturated and up to ~12 px wide, often touching
the bright pupil. A white top-hat (elliptical kernel ``k`` px) removes the bright pupil and slow background; threshold =
max(``thr_min``, median + ``nmad``·MAD of the top-hat in the window). Among components within ``radius`` px of the window
centre, take the one with the highest top-hat peak (not the one nearest the DLC marker, which biases toward DLC and picks
rim flecks when DLC is off); require area ``amin``–``amax`` px², fill ratio area/bbox ≥ ``fill_min``, no contact with the
window edge. Centre = top-hat-weighted centroid. Robustness (t31b, 8 sessions): 97–100 % of confident frames detected;
kernel 11–19 px × threshold 4–8 MAD change the session median by ≤ 0.14 px; planted Gaussian glints recovered with
≤ 0.2 px bias, also within 4 px of the pupil.

Tracker (``track_video``), per frame:
  search centre   the DLC IR marker when its likelihood ≥ ``llh_center``; otherwise the anchor (``anchor_tol`` is unused
                  since glint-1.2). Anchor = the median of the last ``anchor_n`` accepted image detections, started
                  from the session median of confident DLC IR positions (the glint barely moves in a head-fixed session)
  contrast floor  detections whose top-hat peak is below ``rel_peak_min`` × the recording's median peak are rejected: when
                  the real glint merges with a whisker or lid highlight (fails the compactness test) the detector otherwise
                  falls back to a weak fleck 10–15 px away (0.1–3 % of single frames, v1-drift-audit F41)
  temporal check  after the pass, detections more than ``reject_px`` from a rolling median (``reject_win`` frames) of the
                  detections are rejected (glare, eyelid highlights, flecks)
  block gate      per block of ``block_frames`` (5 min at 60 Hz): the block's median DLC − image offset must lie within
                  ``block_dev_px`` of the recording median. A failing block means the detector or DLC sat on a wrong spot for
                  minutes, which the rolling checks follow; which one is wrong cannot be decided, so all its frames get
                  source 0 (none). If more than ``max_bad_block_frac`` of the frames are in failing blocks the recording is
                  implausible
  DLC consistency where the DLC marker is confident (likelihood ≥ ``llh_center``), a detection must agree with it within
                  ``dlc_consistency_px`` after removing DLC's slowly varying offset (rolling median of DLC − image over
                  ± ``offset_win`` frames, two passes). This vetoes long runs on a wrong spot (the glint merged with the bright
                  pupil or a whisker highlight for many frames, which the temporal check follows); DLC only vetoes, the
                  stored position is the image measurement, and the offset level is not truncated
  fallback        where no accepted detection: DLC IR − rolling median of (DLC − image) over ± ``offset_win`` frames, if the
                  DLC likelihood ≥ ``llh_fallback``; else NaN
  distance gate   (glint-1.3, ``gate_px``; None = off, glint-1.2 behaviour) DLC can be confidently (likelihood 1.0) on a
                  wrong spot -- the pupil rim, a whisker, a corner of the image up to ~1000 px away. The fallback followed it,
                  and occasionally so did an accepted detection (which passed the DLC-consistency check because both sat on
                  the same wrong spot; v1-drift-audit nb41/t160: up to 36 frames per recording > 60 px off).
                  Reference = centred rolling median of the accepted image detections over ± ``offset_win`` frames (at least
                  ``anchor_n`` detections in the window), else their recording median. Accepted detections farther than
                  ``gate_px`` from it are dropped, and the reference is recomputed once without them (so they cannot
                  contaminate it); the final decisions use that second reference. The DLC offsets and the fallback are then
                  computed from the remaining detections as before; fallbacks farther than ``gate_px`` are dropped.
                  Threshold (set 3: 40 px; 22 recordings, 17.11.2026): image detections confirmed by a confident DLC marker lie
                  within 35 px of the reference in 99.999 % of 6.0 M frames; by eye, fallbacks > 40 px away sit on the pupil
                  rim or off the eye while the glint is visible near the reference
  re-detection    (glint-1.3, with the gate) frames whose sample was dropped by the gate (image detection or fallback) are
                  read again and the detector is run on a window centred on the reference, with the existing criteria
                  (``detect``, contrast floor against the recording's median peak, temporal check against the rolling median of
                  accepted + re-detected positions). A passing re-detection replaces the sample (source 3); otherwise the
                  frame is none. Fallbacks within the gate stay fallbacks (source 2), as in glint-1.2. No DLC-consistency check:
                  DLC is what is wrong there.
                  Re-detections do not enter the DLC offsets
  source          1 = image detection, 2 = DLC − local offset, 3 = image re-detection at the reference (glint-1.3), 0 = none;
                  ``frac_image`` counts sources 1 and 3
  plausibility    if the median |DLC − image| offset exceeds ``max_offset_px`` the detections are not the DLC glint (e.g. a
                  truncated video or a different reflection): ``diagnostics['offset_plausible']`` is False and callers must
                  not store the result
"""
import numpy as np
import cv2

DEFAULTS = dict(W=20, k=15, nmad=6.0, thr_min=15.0, radius=14.0, amin=2, amax=150, fill_min=0.4,
                llh_center=0.9, anchor_tol=15.0, anchor_n=30, reject_px=3.0, reject_win=31,
                offset_win=1800, llh_fallback=0.5, llh_anchor=0.95, max_offset_px=8.0, rel_peak_min=0.35, dlc_consistency_px=3.0,
                block_frames=18000, block_dev_px=4.0, max_bad_block_frac=0.3, gate_px=None)
CODE_VERSION = "glint-1.3"
# Parameter sets written by earlier code versions, and how this code reproduces them (version dispatch, one place):
#   glint-1.1 (set 1): the search window follows DLC only within anchor_tol of the anchor, and the DLC-image offsets
#                      (block gate, recording offset, local fallback offset, block range) use every accepted frame;
#   glint-1.2 (set 2): window on DLC whenever DLC is confident; offsets from DLC-confident frames only;
#   glint-1.3 (set 3): glint-1.2 + distance gate and re-detection (gate_px; None = off = glint-1.2).
# track_video(version=...) runs the code of that version; glint-1.1 and glint-1.2 need gate_px None.
CODE_VERSIONS = ("glint-1.1", "glint-1.2", "glint-1.3")


def detect(win, k=15, nmad=6.0, thr_min=15.0, radius=14.0, amin=2, amax=150, fill_min=0.4, **_):
    """Glint in a square grey window (uint8). Returns (x, y, area, peak) in window coordinates, or None."""
    th = cv2.morphologyEx(win, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    med = float(np.median(th)); thr = max(thr_min, med + nmad * 1.4826 * float(np.median(np.abs(th - med))))
    n, lab, st, cen = cv2.connectedComponentsWithStats((th >= thr).astype(np.uint8), 8)
    c = (win.shape[1] // 2, win.shape[0] // 2); best = None
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]; bx, by, bw, bh = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
        if np.hypot(cen[i, 0] - c[0], cen[i, 1] - c[1]) > radius or not (amin <= a <= amax) or a / (bw * bh) < fill_min:
            continue
        if bx == 0 or by == 0 or bx + bw >= win.shape[1] or by + bh >= win.shape[0]:
            continue
        pk = float(th[lab == i].max())
        if best is None or pk > best[1]:
            best = (i, pk)
    if best is None:
        return None
    yy, xx = np.nonzero(lab == best[0]); wt = th[yy, xx]
    return float((xx * wt).sum() / wt.sum()), float((yy * wt).sum() / wt.sum()), int(len(xx)), best[1]


def _rolling_median(v, win):
    """NaN-aware centred rolling median (v: (T, 2))."""
    import pandas as pd
    return pd.DataFrame(v).rolling(win, center=True, min_periods=max(3, win // 4)).median().to_numpy()


def _reference(raw, acc, p):
    """Reference glint (T, 2): centred rolling median of accepted detections over +- offset_win frames (>= anchor_n of them),
    else their recording median; NaN if there is no accepted detection."""
    import pandas as pd
    X = np.where(acc[:, None], raw, np.nan).astype(float)
    ref = pd.DataFrame(X).rolling(2 * int(p["offset_win"]) + 1, center=True, min_periods=int(p["anchor_n"])).median().to_numpy()
    return np.where(np.isfinite(ref), ref, np.nanmedian(X[acc], 0) if acc.any() else np.nan)


SEEK_GAP = 600   # re-detection pass: seek when the next frame is this far ahead, else decode forward


def _read_frames(video, frames):
    """Yield (frame index, grey image) for sorted frame indices."""
    cap = cv2.VideoCapture(str(video)); pos = 0
    try:
        for f in frames:
            if f < pos or f - pos > SEEK_GAP:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); pos = int(f)
            while pos < f:
                cap.grab(); pos += 1
            ok, img = cap.read(); pos += 1
            if not ok:
                return
            yield f, (img[:, :, 0] if img.ndim == 3 else img)
    finally:
        cap.release()


def track_video(video, dlc_x, dlc_y, dlc_q, params=None, frames=None, progress=None, version=CODE_VERSION):
    """Per-frame glint over a video. dlc_*: per-frame DLC IR marker (x, y, likelihood). ``frames``: optional frame count
    limit. ``version``: the code version of the parameter set (see CODE_VERSIONS). Returns dict(x, y, source uint8,
    raw_x, raw_y, diagnostics dict)."""
    p = dict(DEFAULTS, **(params or {})); W = int(p["W"])
    if version not in CODE_VERSIONS:
        raise ValueError(f"glint code version {version!r} is not implemented here; known: {CODE_VERSIONS}")
    if version != "glint-1.3" and p.get("gate_px") is not None:
        raise ValueError(f"gate_px is a glint-1.3 parameter, not {version}")
    v11 = version == "glint-1.1"
    dlc_x, dlc_y, dlc_q = (np.asarray(v, float) for v in (dlc_x, dlc_y, dlc_q))
    conf = dlc_q >= p["llh_anchor"]
    if conf.sum() < 10:
        conf = dlc_q >= 0.5
    anchor = np.array([np.nanmedian(dlc_x[conf]), np.nanmedian(dlc_y[conf])]) if conf.any() else None
    cap = cv2.VideoCapture(str(video)); T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    T = min(T, len(dlc_x)) if frames is None else min(T, len(dlc_x), frames)
    raw = np.full((T, 2), np.nan, np.float32); pk = np.full(T, np.nan, np.float32); recent = []; n_read = 0
    for f in range(T):
        ok, img = cap.read()
        if not ok:
            break
        n_read += 1
        g = img[:, :, 0] if img.ndim == 3 else img
        if anchor is None:
            if dlc_q[f] < 0.5:
                continue
            anchor = np.array([dlc_x[f], dlc_y[f]])
        # centre on the DLC marker whenever DLC is confident; the anchor is only the fallback for unconfident stretches.
        # (glint-1.1 also required the marker to lie within anchor_tol of the anchor: once the anchor had drifted onto a
        # wrong spot > anchor_tol from DLC, the window never returned to DLC, and the tracker followed the wrong spot for
        # tens of minutes -- 40-47 % of several drift-study recordings, v1-drift-audit t31d/t31e, 28.9)
        use_dlc = dlc_q[f] >= p["llh_center"] and (not v11 or np.hypot(dlc_x[f] - anchor[0], dlc_y[f] - anchor[1]) <= p["anchor_tol"])
        cx, cy = (dlc_x[f], dlc_y[f]) if use_dlc else (anchor[0], anchor[1])
        x0, y0 = int(round(cx)) - W, int(round(cy)) - W
        if x0 < 0 or y0 < 0 or x0 + 2 * W + 1 > g.shape[1] or y0 + 2 * W + 1 > g.shape[0]:
            continue
        d = detect(g[y0:y0 + 2 * W + 1, x0:x0 + 2 * W + 1], **p)
        if d is not None:
            raw[f] = (x0 + d[0], y0 + d[1]); pk[f] = d[3]; recent.append(raw[f].astype(float))
            if len(recent) > p["anchor_n"]:
                recent.pop(0)
            anchor = np.median(np.array(recent), 0)
        if progress and f % 20000 == 0:
            progress(f, T)
    cap.release()
    det = np.isfinite(raw[:, 0])
    weak = det & (pk < p["rel_peak_min"] * np.nanmedian(pk)) if det.any() else det
    raw[weak] = np.nan; det = det & ~weak
    med = _rolling_median(raw.astype(float), int(p["reject_win"]))
    bad = det & (np.hypot(*(raw - med).T) > p["reject_px"])
    acc = det & ~bad
    DL = np.column_stack([dlc_x[:T], dlc_y[:T]]); qc = dlc_q[:T] >= p["llh_center"]
    offq = np.ones(T, bool) if v11 else qc     # frames whose DLC-image offset counts (glint-1.1: every accepted frame)
    badblk = np.zeros(T, bool); n_bad_blocks = 0
    if acc.any():
        # offsets only where DLC itself is confident: elsewhere its coordinates are not a glint position, and they dragged
        # block medians past the gate in recordings whose confident frames agree within ~1 px (glint-1.2, 28.9)
        off0 = DL - raw; gmed = np.nanmedian(off0[acc & offq], 0) if (acc & offq).any() else np.nanmedian(off0[acc], 0)
        B = int(p["block_frames"])
        for b0 in range(0, T, B):
            m = np.zeros(T, bool); m[b0:b0 + B] = True; mm = m & acc & offq
            if mm.sum() < 50:
                continue
            if np.hypot(*(np.nanmedian(off0[mm], 0) - gmed)) > p["block_dev_px"]:
                badblk |= m; n_bad_blocks += 1
        acc = acc & ~badblk
    pre = np.nanmedian((DL - raw)[acc & offq], 0) if (acc & offq).any() else np.array([np.nan, np.nan])   # offset before the DLC veto
    incons = np.zeros(T, bool)
    for _ in range(2):
        off = DL - np.where(acc[:, None], raw, np.nan)
        off_loc = _rolling_median(np.where((acc & offq)[:, None], off, np.nan), 2 * int(p["offset_win"]) + 1)
        r = acc & qc & np.isfinite(off_loc[:, 0]) & (np.hypot(*(off - off_loc).T) > p["dlc_consistency_px"])
        incons |= r; acc = acc & ~r
    gate = p.get("gate_px"); gated_img = np.zeros(T, bool); ref = None
    if gate is not None:
        ref = _reference(raw, acc, p)
        far = acc & ~(np.hypot(*(raw - ref).T) <= gate)
        if far.any():                                     # once more without them, so they cannot pull the reference
            ref = _reference(raw, acc & ~far, p)
            far = acc & ~(np.hypot(*(raw - ref).T) <= gate)
        gated_img = far; acc = acc & ~far
    X = np.where(acc[:, None], raw, np.nan).astype(float)
    off = DL - X
    off_loc = _rolling_median(np.where((acc & offq)[:, None], off, np.nan), 2 * int(p["offset_win"]) + 1)
    glob = np.nanmedian(off[acc & offq], 0) if (acc & offq).any() else np.array([np.nan, np.nan])
    off_loc = np.where(np.isfinite(off_loc), off_loc, glob)
    fb = ~acc & ~badblk & (dlc_q[:T] >= p["llh_fallback"]) & np.isfinite(off_loc[:, 0])
    out = X.copy(); out[fb] = np.column_stack([dlc_x[:T], dlc_y[:T]])[fb] - off_loc[fb]
    gated_fb = np.zeros(T, bool); redet = np.zeros(T, bool); n_redetect_tried = 0
    if gate is not None:
        dref = np.hypot(*(out - ref).T)
        gated_fb = fb & ~(dref <= gate)                   # no reference (no accepted detection at all) -> dropped too
        out[gated_fb] = np.nan; fb = fb & ~gated_fb
        cand = np.flatnonzero(~badblk & np.isfinite(ref[:, 0]) & (gated_img | gated_fb))   # gated samples only
        n_redetect_tried = len(cand)
        if len(cand):
            pk_med = np.nanmedian(pk)       # the contrast floor reference of the first pass (all its detections)
            R = np.full((T, 2), np.nan)
            for f, g in _read_frames(video, cand):
                x0, y0 = int(round(ref[f, 0])) - W, int(round(ref[f, 1])) - W
                if x0 < 0 or y0 < 0 or x0 + 2 * W + 1 > g.shape[1] or y0 + 2 * W + 1 > g.shape[0]:
                    continue
                d = detect(g[y0:y0 + 2 * W + 1, x0:x0 + 2 * W + 1], **p)
                if d is not None and d[3] >= p["rel_peak_min"] * pk_med:
                    R[f] = (x0 + d[0], y0 + d[1])
            # temporal check against accepted + re-detected positions, as for the first pass
            C = np.where(acc[:, None], raw, R).astype(float)
            medc = _rolling_median(C, int(p["reject_win"]))
            redet = np.isfinite(R[:, 0]) & ~(np.hypot(*(R - medc).T) > p["reject_px"]) & (np.hypot(*(R - ref).T) <= gate)
            out[redet] = R[redet]; fb = fb & ~redet
    src = np.zeros(T, np.uint8); src[acc] = 1; src[fb] = 2; src[redet] = 3
    blocks = [np.nanmedian(off[acc & offq & (np.arange(T) * 5 // max(T, 1) == b)], 0) for b in range(5)]
    blocks = np.array([b for b in blocks if np.all(np.isfinite(b))]) if acc.any() else np.zeros((0, 2))
    diag = dict(n_frames=int(T), n_read=int(n_read), frac_image=float((acc | redet).mean()) if T else 0.0,
                frac_fallback=float(fb.mean()) if T else 0.0, frac_none=float((src == 0).mean()) if T else 1.0,
                n_rejected_temporal=int(bad.sum()), n_rejected_contrast=int(weak.sum()), n_rejected_dlc_inconsistent=int(incons.sum()),
                n_image_gated=int(gated_img.sum()), n_fallback_gated=int(gated_fb.sum()), n_redetect_tried=int(n_redetect_tried),
                n_redetected=int(redet.sum()), frac_redetected=float(redet.mean()) if T else 0.0,
                n_bad_blocks=int(n_bad_blocks), frac_bad_block_frames=float(badblk.mean()) if T else 0.0,
                dlc_offset_dx=float(glob[0]), dlc_offset_dy=float(glob[1]),
                dlc_offset_block_range=float(np.ptp(blocks, 0).max()) if len(blocks) > 1 else np.nan,
                n_dlc_frames=int(len(dlc_x)), dlc_offset_preveto_dx=float(pre[0]), dlc_offset_preveto_dy=float(pre[1]),
                offset_plausible=bool(np.all(np.isfinite(glob)) and np.hypot(*glob) <= p["max_offset_px"]
                                      and (badblk.mean() if T else 1.0) <= p["max_bad_block_frac"]))
    return dict(x=out[:, 0].astype(np.float32), y=out[:, 1].astype(np.float32), source=src,
                raw_x=raw[:, 0], raw_y=raw[:, 1], diagnostics=diag)
