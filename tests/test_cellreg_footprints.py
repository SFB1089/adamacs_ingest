"""Tests for the on-the-fly aligned-footprint reconstruction.

These pin the conventions that were expensive to establish, so a later refactor cannot
quietly flip a sign. The ground-truth agreement against a real run is measured separately
by scripts/verify_footprints.py, which needs the server.
"""
import numpy as np
import pytest

from adamacs.ingest import cellreg_footprints as fp


def _blob(cx=20, cy=30, size=61, sigma=2.5):
    """A small Gaussian blob as (xpix, ypix, weights), the shape a mask row has."""
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
    img[img < 1e-3] = 0
    ys, xs = np.nonzero(img)
    return xs, ys, img[ys, xs]


def test_footprint_image_places_lam_at_ypix_xpix():
    img = fp.footprint_image([1, 3], [2, 4], [0.25, 0.75], 6, 6)
    assert img[2, 1] == 0.25 and img[4, 3] == 0.75
    assert img.sum() == pytest.approx(1.0)


def test_normalisation_is_by_sum_not_by_max():
    img = fp.footprint_image([0, 1], [0, 0], [1.0, 3.0], 4, 4)
    out = fp.normalise_footprint(img)
    assert out.sum() == pytest.approx(1.0)
    assert out.max() == pytest.approx(0.75)      # by max would give 1.0


def test_padding_goes_bottom_right_so_no_pixel_moves():
    img = np.zeros((4, 4)); img[0, 0] = 1.0
    out = fp.pad_to(img, 7, 9)
    assert out.shape == (7, 9)
    assert out[0, 0] == 1.0 and out.sum() == 1.0


def test_zero_translation_is_the_identity():
    xs, ys, w = _blob()
    img = fp.normalise_footprint(fp.footprint_image(xs, ys, w, 61, 61))
    out = fp.align_footprint(img, dx_px=0, dy_px=0, rotation_deg=0,
                             centroid_xy=(20, 30), microns_per_pixel=1.2)
    assert np.abs(out - img).max() < 1e-12


def test_translation_is_a_backward_map():
    """out(y, x) = in(y + dy, x + dx): a positive offset moves the image NEGATIVE."""
    img = np.zeros((41, 41)); img[20, 20] = 1.0
    out = fp.align_footprint(img, dx_px=3, dy_px=0, rotation_deg=0,
                             centroid_xy=(20, 20), microns_per_pixel=1.2)
    assert out[20, 17] == pytest.approx(1.0)
    assert out[20, 23] == 0.0


def test_pixels_beyond_the_crop_radius_are_deleted():
    """25 um at 1.2 um/px is ~20.8 px; mass is NOT conserved."""
    img = np.zeros((81, 81))
    img[40, 40] = 0.5          # at the centroid
    img[40, 75] = 0.5          # 35 px away -- outside the disc
    out = fp.align_footprint(img, dx_px=0, dy_px=0, rotation_deg=0,
                             centroid_xy=(40, 40), microns_per_pixel=1.2)
    assert out[40, 40] == pytest.approx(0.5)
    assert out[40, 75] == 0.0
    assert out.sum() == pytest.approx(0.5)


def test_crop_radius_differs_between_the_two_branches():
    assert fp.MAXIMAL_CELL_RADIUS_TRANSLATE_UM == 25.0
    assert fp.MAXIMAL_CELL_RADIUS_ROTATE_UM == 30.0
    assert fp.MINIMAL_ROTATION_DEG == 0.3


def test_small_rotations_do_not_take_the_rotating_branch():
    """align_images.m translates only when |rotation| <= 0.3 deg."""
    img = np.zeros((41, 41)); img[20, 20] = 1.0
    small = fp.align_footprint(img, dx_px=2, dy_px=0, rotation_deg=0.2,
                               centroid_xy=(20, 20), microns_per_pixel=1.2)
    none = fp.align_footprint(img, dx_px=2, dy_px=0, rotation_deg=None,
                              centroid_xy=(20, 20), microns_per_pixel=1.2)
    assert np.abs(small - none).max() == 0.0


def test_bilinear_interpolation_at_a_half_pixel():
    img = np.zeros((5, 5)); img[2, 2] = 1.0
    out = fp.align_footprint(img, dx_px=0.5, dy_px=0.0, rotation_deg=0,
                             centroid_xy=(2, 2), microns_per_pixel=1.2)
    assert out[2, 1] == pytest.approx(0.5)
    assert out[2, 2] == pytest.approx(0.5)


def test_centroid_location_recovers_a_known_centre():
    xs, ys, w = _blob(cx=22, cy=31, size=81)
    img = fp.normalise_footprint(fp.footprint_image(xs, ys, w, 81, 81))
    cx, cy = fp.centroid_location(img, 1.2)
    assert cx == pytest.approx(22, abs=0.5)
    assert cy == pytest.approx(31, abs=0.5)


def test_centroid_is_in_pixels_not_microns():
    """compute_centroid_locations.m applies no micron scaling, unlike the translations."""
    xs, ys, w = _blob(cx=22, cy=31, size=81)
    img = fp.normalise_footprint(fp.footprint_image(xs, ys, w, 81, 81))
    a = fp.centroid_location(img, 1.2)
    b = fp.centroid_location(img, 2.4)
    assert a[0] == pytest.approx(b[0], abs=1.0)   # differs only via the window size


def test_gaussfit_ignores_sigma0_as_the_original_does():
    """gaussfit.m tests nargin==3 for sigma, so a 4-argument call drops sigma0."""
    x = np.arange(-20, 21)
    y = np.exp(-(x - 2.0) ** 2 / (2 * 3.0 ** 2))
    y = y / y.sum()
    mu_a = fp._gaussfit_mu(x, y, 0.5)
    mu_b = fp._gaussfit_mu(x, y, 50.0)
    assert mu_a == pytest.approx(mu_b)
    assert mu_a == pytest.approx(2.0, abs=0.3)


def test_reference_session_is_passed_through_untouched():
    xs, ys, w = _blob()
    out = fp.aligned_footprint(xs, ys, w, height=61, width=61,
                               dx_um=12.0, dy_um=-7.0, rotation_deg=1.5,
                               microns_per_pixel=1.2, is_reference=True)
    expect = fp.normalise_footprint(fp.footprint_image(xs, ys, w, 61, 61))
    assert np.abs(out - expect).max() == 0.0


def test_translations_are_interpreted_as_microns():
    """Passing 12 um at 1.2 um/px must shift by 10 px, not 12."""
    img_xs, img_ys, w = [20], [20], [1.0]
    out = fp.aligned_footprint(img_xs, img_ys, w, height=41, width=41,
                               dx_um=12.0, dy_um=0.0, rotation_deg=None,
                               microns_per_pixel=1.2)
    assert out[20, 10] == pytest.approx(1.0)


def test_to_sparse_round_trips():
    img = np.zeros((9, 9)); img[3, 4] = 0.5; img[7, 1] = 0.25
    ys, xs, vals = fp.to_sparse(img)
    assert set(zip(ys.tolist(), xs.tolist())) == {(3, 4), (7, 1)}
    assert vals.sum() == pytest.approx(0.75)


def test_a_zero_footprint_yields_nan_as_matlab_does():
    out = fp.normalise_footprint(np.zeros((4, 4)))
    assert np.isnan(out).all()
