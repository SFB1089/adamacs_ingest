"""Reconstruct CellReg's aligned spatial footprints from the database, on demand.

WHY THIS EXISTS
---------------
``cell_registered_struct.spatial_footprints_corrected`` is the one CellReg output the
schema does not store. For the eleven-session ROS-1881 run it is 190 MB in the result
file and ~2.9 GB in memory as doubles (11 x 1131 x 513 x 513), and it is **entirely
redundant**: it is the segmentation masks this lab already keeps in
``imaging.Segmentation.Mask``, put through a transform whose every parameter the schema
*does* store -- ``CellRegRun.Session.alignment_x_translation``,
``alignment_y_translation``, ``alignment_rotation``, ``zero_padding_x`` and
``zero_padding_y``.

So the schema stores the transform and regenerates the pixels. This module is that
regeneration, ported function-for-function from the pinned toolbox
(``zivlab/CellReg`` @ 70668513c665e1cb0de218c33264b8736f2976eb).

THE PIPELINE, AS CellReg ACTUALLY RUNS IT
-----------------------------------------
Per session, in order (``cellreg_run.m`` lines 82-92, then 233-252):

1. **Suite2p conversion** -- ``footprints(k, ypix+1, xpix+1) = lam`` in a
   ``(Ly+1) x (Lx+1)`` frame. The ``+1`` on the FOV is the driver's, not a typo: the
   log's "Image size - 513X513" for a 512-pixel scan is this (``cellreg_run.m``:130-131).
2. **normalize_spatial_footprints.m** -- each footprint divided by its own sum. The
   ``pixel_threshold`` branch is dead (the constant is 0), so this is just ``w / sum(w)``.
3. **adjust_FOV_size.m** -- every session zero-padded to the largest FOV, placed at
   ``[1:sz_y, 1:sz_x]``, i.e. **padding is added at the bottom and right**.
   ``adjustment_zero_padding = [adjusted_x - sz_x ; adjusted_y - sz_y]``.
4. **align_images.m** -- per cell, ``translate_spatial_footprint`` or (if the rotation
   exceeds 0.3 degrees) ``rotate_spatial_footprint``.

FIVE CONVENTIONS THAT WILL BITE YOU
-----------------------------------
Each of these was read out of the toolbox source, not guessed.

1. **The stored translations are in MICRONS, not pixels.** ``align_images.m``:214 ends
   with ``best_translations = microns_per_pixel * [best_x_translations ;
   best_y_translations]``, and that product is what reaches
   ``cell_registered_struct.alignment_x/y_translations``. Divide by
   ``microns_per_pixel`` before using them as pixel offsets. The rotations are NOT
   scaled -- they stay in degrees.

2. **The transform is a backward map.** ``translate_spatial_footprint`` computes
   ``out(p,q) = interp(in, p + dx, q + dy)``, so the image moves by ``-(dx, dy)``.
   Flipping this sign produces a footprint that is wrong by twice the translation and
   still looks like a cell.

3. **Pixels far from the centroid are DELETED, not moved.** Both transforms zero any
   output pixel whose source coordinate lies more than ``maximal_cell_radius`` from the
   cell's centroid -- 25 um when translating, 30 um when rotating. An aligned footprint
   is therefore *not* a shifted copy of the mask: it is a shifted copy cropped to a disc.
   Any check that assumes mass is conserved will fail, correctly.

4. **Footprints are transposed before transforming and transposed back after**
   (``align_images.m``:140, 168). Inside the transform ``p`` indexes x and ``q`` indexes
   y, which is why ``translations`` arrives as ``[dy dx]`` and is unpacked as
   ``a = translations(2)``.

5. **The rotation centre is taken from the wrong axis.** ``align_images.m``:77-78 sets
   ``center_of_FOV(1) = size(2)/2`` (the y extent) and ``(2) = size(3)/2`` (the x
   extent), but the footprint handed to ``rotate_spatial_footprint`` is transposed, so
   ``center_of_FOV(1)`` is subtracted from an x index. For a square FOV -- every scan in
   this lab -- the two are equal and nothing happens. For a non-square FOV the toolbox's
   own rotation is off-centre, and this port reproduces that rather than silently
   "fixing" it, because the goal is to match the run that was actually performed.

Two further exactness notes:

* **The reference session is passed through untouched** (``align_images.m``:60-61): no
  translation, no rotation, no radius crop. Its aligned footprint is the normalised,
  padded mask.
* ``interpolate_pixel_value.m`` contains a clamp bug -- when the second coordinate
  exceeds ``M+1`` it clamps the *first* coordinate. It can only fire outside the image,
  where the value is zero anyway, so it is not reproduced here; this note exists so the
  next reader does not think it was missed.

NO APPROXIMATION
----------------
An earlier draft recovered the pre-alignment centroid the radius crop needs as
``corrected - (dx, dy)`` from the stored aligned centroid, since CellReg saves only
``centroid_locations_corrected``. Checked against the January 2026 ROS-1881 run that was
not good enough: column 0 reproduced to a median correlation of 1.000000 but only
1050 of 1131 cells cleared 0.9999, and 37 cells fell below 0.99 -- the ones whose pixels
sit near the ~21 px crop boundary, where a sub-pixel centroid error changes which pixels
survive.

So the centroid is computed instead, by :func:`centroid_location`, a port of
``compute_centroid_locations.m``. CellReg measures the crop from the centroid of the
*unaligned* footprint, and the unaligned footprint is the database mask -- nothing is
missing. This also removes any dependency on ``MaskCoordinates``: a column's aligned
footprints need only its masks and its one row of alignment parameters.

VERIFIED
--------
Against ``cellRegistered_20260129_075504.mat`` (ROS-1881, site sess9FPXNI8U, 11 gratings
sessions), every footprint of three columns -- 3,494 cells in total -- compared against the
run's own ``spatial_footprints_corrected``:

====== ======================== ============ ==========================
column branch                   correlation  max absolute pixel difference
====== ======================== ============ ==========================
4      reference (pass-through) 1.000000     0.0
0      translation              1.000000     2.8e-17
7      rotation (0.362 deg)     1.000000     2.4e-15
====== ======================== ============ ==========================

That is exact to floating-point noise, including the radius crop and the sub-pixel
interpolation. ``scripts/verify_footprints.py`` is the measurement; run it after any change
here rather than assuming the port still holds.

"""
from __future__ import annotations

import numpy as np

#: ``translate_spatial_footprint.m``:14 -- radius, in microns, outside which a translated
#: pixel is zeroed.
MAXIMAL_CELL_RADIUS_TRANSLATE_UM = 25.0

#: ``rotate_spatial_footprint.m``:17 -- the same radius for the rotating branch.
MAXIMAL_CELL_RADIUS_ROTATE_UM = 30.0

#: ``align_images.m``:44 -- below this rotation CellReg translates only, and the 30 um
#: radius never applies.
MINIMAL_ROTATION_DEG = 0.3


def footprint_image(xpix, ypix, weights, height, width) -> np.ndarray:
    """One mask as a dense ``(height, width)`` image, in the driver's own indexing.

    ``cellreg_run.m``:82-92 writes ``footprints(k, ypix+1, xpix+1) = lam`` into a
    ``(Ly+1, Lx+1)`` frame, so pass ``height = Ly + 1`` and ``width = Lx + 1``.
    ``mask_xpix`` / ``mask_ypix`` / ``mask_weights`` in ``imaging.Segmentation.Mask`` are
    exactly suite2p's ``xpix`` / ``ypix`` / ``lam``.
    """
    img = np.zeros((int(height), int(width)), dtype=float)
    img[np.asarray(ypix, dtype=int), np.asarray(xpix, dtype=int)] = np.asarray(
        weights, dtype=float)
    return img


def normalise_footprint(img) -> np.ndarray:
    """``w / sum(w)`` -- ``normalize_spatial_footprints.m``, ``pixel_threshold = 0`` branch.

    A footprint summing to zero yields NaN in MATLAB; that is reproduced rather than
    silently turned into zeros, because a NaN footprint is a data problem worth seeing.
    """
    total = np.nansum(img)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(img, dtype=float) / total


def pad_to(img, height, width) -> np.ndarray:
    """Zero-pad at the bottom and right -- ``adjust_FOV_size.m``.

    Note the asymmetry: the original occupies ``[0:sz_y, 0:sz_x]`` of the padded frame, so
    padding never shifts a pixel. A run whose sessions all share one FOV -- every run in
    this lab so far -- pads by zero and this is the identity.
    """
    out = np.zeros((int(height), int(width)), dtype=float)
    out[:img.shape[0], :img.shape[1]] = img
    return out


def _bilinear(img, yy, xx) -> np.ndarray:
    """``interpolate_pixel_value.m``, vectorised, on 0-based coordinates.

    MATLAB's 1-based ``(p, q)`` and numpy's 0-based ``(y, x)`` differ by exactly one on
    both axes, and the bilinear weights depend only on the fractional part, so the port is
    an index shift with identical weights. Out-of-range corners contribute zero, matching
    the four guarded branches in the original.
    """
    y0, x0 = np.floor(yy).astype(int), np.floor(xx).astype(int)
    y1, x1 = np.ceil(yy).astype(int), np.ceil(xx).astype(int)
    beta, alpha = yy - y0, xx - x0          # beta along dim 1, alpha along dim 2
    h, w = img.shape

    def at(ys, xs):
        ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        vals = np.zeros(ys.shape, dtype=float)
        vals[ok] = img[ys[ok], xs[ok]]
        return vals

    return ((1 - alpha) * (1 - beta) * at(y0, x0)
            + (1 - alpha) * beta * at(y1, x0)
            + alpha * (1 - beta) * at(y0, x1)
            + alpha * beta * at(y1, x1))


def align_footprint(img, *, dx_px, dy_px, rotation_deg, centroid_xy,
                    microns_per_pixel, center_of_fov=None) -> np.ndarray:
    """Apply one session's alignment to one already normalised, padded footprint.

    Parameters mirror the toolbox's, in the toolbox's units:

    ``dx_px`` / ``dy_px``
        Translations **in pixels** -- divide the stored micron values by
        ``microns_per_pixel`` first.
    ``rotation_deg``
        The stored ``alignment_rotation`` in degrees, or ``None``/0 for a translation-only
        run. ``align_images.m``:140 passes ``-best_rotation`` into the transform, and that
        negation happens here, not in the caller.
    ``centroid_xy``
        ``(x, y)`` of the cell's centroid **before** alignment, 0-based, used only for the
        radius crop.
    ``center_of_fov``
        Rotation centre; defaults to ``(height/2, width/2)`` reproducing the toolbox's
        axis mix-up described in the module docstring. Ignored without a rotation.
    """
    img = np.asarray(img, dtype=float)
    h, w = img.shape
    rotating = rotation_deg is not None and abs(float(rotation_deg)) > MINIMAL_ROTATION_DEG
    radius_um = (MAXIMAL_CELL_RADIUS_ROTATE_UM if rotating
                 else MAXIMAL_CELL_RADIUS_TRANSLATE_UM)
    radius_px = radius_um / float(microns_per_pixel)

    yy, xx = np.mgrid[0:h, 0:w].astype(float)       # output coordinates (0-based)

    if rotating:
        if center_of_fov is None:
            # align_images.m:77-78 sets center_of_FOV = size/2 on MATLAB's 1-based axis,
            # so for a 513 px frame the centre is the coordinate 256.5. A 0-based index i
            # is MATLAB's p - 1, hence the same point is i = 255.5 = size/2 - 1. Dropping
            # the -1 leaves the rotation centre one pixel off; see docstring note 5 for
            # the axis mix-up this deliberately preserves.
            center_of_fov = (h / 2.0 - 1.0, w / 2.0 - 1.0)
        c1, c2 = float(center_of_fov[0]), float(center_of_fov[1])
        theta = np.deg2rad(-float(rotation_deg))    # align_images.m:140 negates
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # transformation = [cos -sin; sin cos]' ; trans_inv = transformation^-1
        #               => trans_inv = [[cos, -sin], [sin, cos]]
        px, qy = xx - c1 + dx_px, yy - c2 + dy_px
        src_x = cos_t * px - sin_t * qy + c1
        src_y = sin_t * px + cos_t * qy + c2
    else:
        src_x, src_y = xx + dx_px, yy + dy_px

    # The radius test is on the SOURCE coordinate against the UNALIGNED centroid
    # (translate_spatial_footprint.m:24) -- not on the output coordinate.
    cx, cy = float(centroid_xy[0]), float(centroid_xy[1])
    inside = ((src_x - cx) ** 2 + (src_y - cy) ** 2) <= radius_px ** 2

    out = np.zeros((h, w), dtype=float)
    if inside.any():
        out[inside] = _bilinear(img, src_y[inside], src_x[inside])
    return out


def aligned_footprint(xpix, ypix, weights, *, height, width,
                      dx_um, dy_um, rotation_deg, microns_per_pixel,
                      is_reference=False, pad_height=None, pad_width=None,
                      centroid_xy=None) -> np.ndarray:
    """Mask row -> aligned footprint, the whole chain, in CellReg's units.

    ``dx_um`` / ``dy_um`` are the stored translations **in microns** (convention 1).
    ``is_reference=True`` short-circuits to the normalised, padded mask, which is exactly
    what the reference session's entry contains (``align_images.m``:60-61).

    The centroid the radius crop needs is **computed**, not approximated: CellReg measures
    the crop from the centroid of the unaligned footprint, and the unaligned footprint is
    the database mask, so :func:`centroid_location` recomputes it. Pass ``centroid_xy`` only
    to override that.
    """
    img = normalise_footprint(footprint_image(xpix, ypix, weights, height, width))
    img = pad_to(img, pad_height or height, pad_width or width)
    if is_reference:
        return img

    dx_px = float(dx_um) / float(microns_per_pixel)
    dy_px = float(dy_um) / float(microns_per_pixel)
    if centroid_xy is None:
        centroid_xy = centroid_location(img, microns_per_pixel)
    return align_footprint(img, dx_px=dx_px, dy_px=dy_px, rotation_deg=rotation_deg,
                           centroid_xy=centroid_xy,
                           microns_per_pixel=microns_per_pixel)


# --------------------------------------------------------------- centroids, exactly
#: ``compute_centroid_locations.m``:11 -- the cell size that sets the gaussfit window.
#: Note this is 12 um here and 10 um in ``align_images.m``:45; they are different
#: constants in the toolbox and are not to be unified.
TYPICAL_CELL_SIZE_UM = 12.0


def _gaussfit_mu(x, y, sigma0) -> float:
    """``gaussfit.m``, returning only ``mu`` -- 50 Gauss-Newton steps on a unit-area Gaussian.

    Two faithfulness notes:

    * **The ``sigma0`` argument is ignored.** ``gaussfit.m`` tests ``nargin == 3`` for
      sigma and ``nargin == 4`` for mu, so when all four arguments are supplied -- which
      is how ``compute_centroid_locations`` calls it -- only the ``mu`` branch fires and
      sigma keeps its data-derived estimate ``1/(sqrt(2*pi)*ymax)``. Honouring ``sigma0``
      here would make this port *disagree* with every run the lab has done. The parameter
      is accepted and dropped so the call site reads like the original.
    * The normalisation guard (divide ``y`` by its trapezoidal sum when that falls outside
      ``[0.5, 1.5]``) is reproduced, because a projection of a near-edge footprint can trip
      it.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]

    total = float(np.sum(np.diff(x) * 0.5 * (y[:-1] + y[1:])))
    if total > 1.5 or total < 0.5:
        if total != 0:
            y = y / total

    ymax = float(np.max(y))
    if not np.isfinite(ymax) or ymax <= 0:
        return 0.0
    sigma = 1.0 / (np.sqrt(2 * np.pi) * ymax)   # sigma0 deliberately unused -- see above
    mu = 0.0                                    # mu0, the fourth argument at every call

    for _ in range(50):
        if not np.isfinite(sigma) or sigma == 0:
            break
        gauss = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
        dfdsigma = -gauss / sigma + gauss * ((x - mu) ** 2 / sigma ** 3)
        dfdmu = gauss * (x - mu) / sigma ** 2
        F = np.column_stack([dfdsigma, dfdmu])
        try:
            delta, *_ = np.linalg.lstsq(F, y - gauss, rcond=None)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)):
            break
        sigma, mu = sigma + delta[0], mu + delta[1]
    return float(mu)


def _localised(projection, max_ind, radius) -> np.ndarray:
    """The zero-padded +/- ``radius`` window around ``max_ind`` -- the three branches of
    ``compute_centroid_locations.m``, on 0-based indices."""
    n = projection.size
    lo, hi = max_ind - radius, max_ind + radius
    if lo >= 0 and hi < n:
        return projection[lo:hi + 1]
    if lo < 0:
        return np.concatenate([np.zeros(-lo), projection[:hi + 1]])
    return np.concatenate([projection[lo:], np.zeros(hi - n + 1)])


def centroid_location(img, microns_per_pixel) -> tuple:
    """``(x, y)`` centroid of one footprint, 0-based -- ``compute_centroid_locations.m``.

    Not a centre of mass: CellReg takes the x and y projections, finds each maximum, and
    fits a Gaussian to a window around it. The returned value is
    ``argmax + gaussfit_mu``, in **pixels** (no micron scaling, unlike the translations).

    This is what makes the radius crop reproducible without the result file: CellReg
    measures the crop from the centroid of the *unaligned* footprint, and saves only the
    aligned one, but the unaligned footprint is the database mask -- so its centroid can
    simply be recomputed here.
    """
    img = np.asarray(img, dtype=float)
    normalized_typical = TYPICAL_CELL_SIZE_UM / float(microns_per_pixel)
    radius = int(round(2 * normalized_typical))
    sigma0 = 0.5 * normalized_typical

    out = []
    for projection in (np.nansum(img, axis=0), np.nansum(img, axis=1)):   # x then y
        max_ind = int(np.argmax(projection))
        window = _localised(projection, max_ind, radius)
        total = window.sum()
        if total <= 0:
            out.append(float(max_ind))
            continue
        mu = _gaussfit_mu(np.arange(-radius, radius + 1), window / total, sigma0)
        out.append(float(max_ind) + mu)
    return out[0], out[1]


def to_sparse(img, *, tol=0.0):
    """``(ypix, xpix, weights)`` of the non-zero pixels -- the form worth returning.

    An aligned footprint is ~0.1 % non-zero; returning the dense 513x513 frame by default
    is how a 2.9 GB array gets rebuilt one cell at a time.
    """
    ys, xs = np.nonzero(np.abs(img) > tol)
    return ys, xs, img[ys, xs]
