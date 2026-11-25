"""DirectionTuning measures (adversarial review 2026-11-25): fit naming, degenerate cells, rectification, missing
directions, angle wrap, goodness of fit."""
import numpy as np
import pytest

T = pytest.importorskip("adamacs.responsiveness.tuning")


def dg(ang, c, rp, rn, tp, s):
    return T._double_gauss([c, rp, rn, tp, s], np.asarray(ang, float))


def trials(ang, curve, k, sd, rng):
    """values, labels with k trials per direction; labels are the angles themselves ('label' convention)."""
    v = np.concatenate([curve[i] + sd * rng.standard_normal(k) for i in range(len(ang))])
    lab = np.repeat([f"{a:g}" for a in ang], k)
    order = rng.permutation(v.size)
    return v[order], lab[order]


def test_fit_reports_larger_peak():
    ang = np.arange(8) * 45.0
    m = dg(ang, 0.0, 5.0, 2.0, 182.0, 25.0)                 # the review's counterexample: fit converged with rn > rp
    f = T.fit_double_gaussian(ang, m)
    assert f["rp"] >= f["rn"]
    assert abs(T._angdiff(f["pref_dir"] - 182.0, 360)) < 1.0
    assert f["dsi_fit"] > 0


def test_fit_larger_peak_random_directions():
    rng = np.random.default_rng(0)
    ang = np.arange(12) * 30.0
    for _ in range(200):
        tp = rng.uniform(0, 360)
        m = dg(ang, 0.5, 4.0, 1.5, tp, 30.0) + 0.3 * rng.standard_normal(ang.size)
        f = T.fit_double_gaussian(ang, m)
        assert f["rp"] >= f["rn"] and 0 <= f["pref_dir"] < 360


def test_fit_flat_curve_all_null():
    ang = np.arange(12) * 30.0
    f = T.fit_double_gaussian(ang, np.zeros(12), [np.zeros(5)] * 12)
    assert all(np.isnan(f[k]) for k in T.FIT_KEYS)


def test_fit_goodness_of_fit():
    rng = np.random.default_rng(1)
    ang = np.arange(12) * 30.0
    curve = dg(ang, 1.0, 6.0, 2.0, 60.0, 25.0)
    v, lab = trials(ang, curve, 15, 1.0, rng)
    out = T.direction_tuning(v, lab, angle_convention="label", n_perm=0, n_boot=0)
    assert out["fit_r2_cv"] > 0.8 and out["fit_chi2_p"] > 0.001 and out["fit_at_bound"] == 0
    assert abs(T._angdiff(out["fit_pref_dir"] - 60.0, 360)) < 10
    # a curve the model cannot describe (three equal peaks) is rejected by chi2
    bad = 1.0 + 5.0 * (np.arange(12) % 4 == 0)
    v, lab = trials(ang, bad, 15, 0.5, rng)
    out = T.direction_tuning(v, lab, angle_convention="label", n_perm=0, n_boot=0)
    assert out["fit_chi2_p"] < 1e-3


def test_chi2_calibrated_when_the_model_is_true():
    rng = np.random.default_rng(2)
    ang = np.arange(12) * 30.0
    curve = dg(ang, 1.0, 6.0, 2.0, 100.0, 30.0)
    p = []
    for _ in range(300):
        v, lab = trials(ang, curve, 15, 1.0, rng)
        p.append(T.direction_tuning(v, lab, angle_convention="label", n_perm=0, n_boot=0)["fit_chi2_p"])
    assert 0.02 <= np.mean(np.array(p) < 0.05) <= 0.09


def test_r2_cv_low_on_noise():
    rng = np.random.default_rng(3)
    ang = np.arange(12) * 30.0
    r2, r2cv = [], []
    for _ in range(200):
        v, lab = trials(ang, np.full(12, 2.0), 15, 1.0, rng)
        out = T.direction_tuning(v, lab, angle_convention="label", n_perm=0, n_boot=0)
        r2.append(out["fit_r2"]); r2cv.append(out["fit_r2_cv"])
    assert np.nanmedian(r2) > 0.3          # in-sample R2 is high on pure noise ...
    assert np.nanmedian(r2cv) < 0          # ... the cross-validated one is not


def test_point_indices_rectified_in_unit_range():
    ang = np.arange(12) * 30.0
    m = np.array([3.0, 1, -2, -1, 0.5, -0.5, 2, 0.2, -1, -3, 0.1, 0.4])
    pi = T.point_indices(ang, T.rectify(m, "clip"))
    for k in ("oi", "di", "osi", "dsi"):
        assert 0 <= pi[k] <= 1
    assert np.isnan(T.point_indices(ang, T.rectify(-np.ones(12), "clip"))["osi"])


def test_missing_direction_nulls_geometry_measures():
    rng = np.random.default_rng(4)
    ang = np.arange(8) * 45.0
    v, lab = trials(ang, np.full(8, 3.0), 10, 1.0, rng)
    v[lab == "90"] = np.nan                          # one direction lost entirely
    out = T.direction_tuning(v, lab, angle_convention="label", n_perm=50, n_boot=20)
    assert out["n_directions"] == 8 and np.isfinite(out["dir_kw_p"])
    assert "hotelling_p" not in out and "ori_vec" not in out and "fit_pref_dir" not in out


def test_preferred_angles_in_range():
    assert T.vector_measures([0, 90, 180, 270], [5, 1, 3, 1])[2] < 180.0
    assert T._wrap(-1e-17, 180.0) == 0.0 and T._wrap(360.0, 360.0) == 0.0


def test_complete_circle():
    assert T.complete_circle(np.arange(12) * 30.0) and T.complete_circle(np.arange(24) * 15.0)
    assert not T.complete_circle(np.array([0, 45, 90, 180, 225, 270, 315.0]))


def test_at_bound_flags_missing_peak_not_missing_null():
    ang = np.arange(12) * 30.0
    # perfectly direction-selective: no null response (rn = 0) is a valid fit, not at a bound
    ds = T.fit_double_gaussian(ang, dg(ang, 0.5, 4.0, 0.0, 90.0, 30.0))
    assert ds["rn"] < 1e-3 and ds["at_bound"] == 0
    # no preferred peak (rp at 0, everything else inside its bounds) must be flagged
    class Res:
        status, cost = 1, 0.1
    m = np.linspace(1.0, 2.0, 12)
    orig = T._fit_curve
    try:
        T._fit_curve = lambda a, mm: (np.array([1.5, 0.0, 0.0, 90.0, 40.0]), Res(), 30.0, 2.0)
        assert T.fit_double_gaussian(ang, m)["at_bound"] == 1
        T._fit_curve = lambda a, mm: (np.array([1.5, 1.0, 0.0, 90.0, 40.0]), Res(), 30.0, 2.0)
        assert T.fit_double_gaussian(ang, m)["at_bound"] == 0
    finally:
        T._fit_curve = orig
