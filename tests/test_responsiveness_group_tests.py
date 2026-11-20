"""kw_rows / anova_rows agree with scipy per row (ties, NaN, degenerate rows)."""
import numpy as np
import pytest

st = pytest.importorskip("scipy.stats")
from adamacs.responsiveness import stats as S  # noqa: E402


def _ref_kw(x, lab):
    parts = [x[(lab == u) & np.isfinite(x)] for u in np.unique(lab)]
    parts = [p for p in parts if p.size]
    if len(parts) < 2 or np.ptp(np.concatenate(parts)) == 0:
        return np.nan
    return st.kruskal(*parts).pvalue


def _ref_anova(x, lab):
    parts = [x[(lab == u) & np.isfinite(x)] for u in np.unique(lab)]
    parts = [p for p in parts if p.size]
    if len(parts) < 2 or np.ptp(np.concatenate(parts)) == 0:
        return np.nan
    return st.f_oneway(*parts).pvalue


def test_group_tests_match_scipy():
    rng = np.random.default_rng(0)
    X = rng.gamma(0.3, 1.0, (40, 60)); X[X < 0.2] = 0.0             # many ties at 0, like CASCADE
    X[3] = 0.0                                                        # degenerate row
    X[5, [2, 7]] = np.nan                                             # NaN in one row
    lab = np.repeat(np.arange(6), 10)
    k, a = S.kw_rows(X, lab), S.anova_rows(X, lab)
    for i in range(X.shape[0]):
        np.testing.assert_allclose(k[i], _ref_kw(X[i], lab), rtol=1e-9, equal_nan=True)
        np.testing.assert_allclose(a[i], _ref_anova(X[i], lab), rtol=1e-9, equal_nan=True)


def test_two_groups_stim_vs_baseline():
    rng = np.random.default_rng(1)
    R, B = rng.normal(1, 1, (10, 50)), rng.normal(0, 1, (10, 50))
    X = np.hstack([R, B]); lab = np.r_[np.ones(50), np.zeros(50)]
    np.testing.assert_allclose(S.kw_rows(X, lab), [st.kruskal(R[i], B[i]).pvalue for i in range(10)], rtol=1e-9)
    np.testing.assert_allclose(S.anova_rows(X, lab), [st.ttest_ind(R[i], B[i]).pvalue for i in range(10)], rtol=1e-9)


# --- literature measures --------------------------------------------------------------------------------------

def test_reliability_components_definitions():
    rng = np.random.default_rng(2)
    I, K = 100, 10
    true = rng.normal(0, 2.0, I)                                     # signal variance 4
    r = true[:, None] + rng.normal(0, 1.0, (I, K))                    # noise variance 1
    out = S.reliability_components(r)
    sp = (r.sum(1).var(ddof=1) - r.var(0, ddof=1).sum()) / (K * (K - 1))
    # Schoppe's SP and the variance-component signal estimate are two unbiased estimators of the same signal power
    np.testing.assert_allclose(out["signal_var"], sp, rtol=0.1)
    assert abs(out["signal_var"] - 4) < 1.2 and abs(out["noise_var"] - 1) < 0.15
    assert 0.7 < out["fev"] < 0.9 and 0.9 < out["cc_max"] <= 1
    noise_only = S.reliability_components(rng.normal(0, 1, (100, 10)))
    assert abs(noise_only["signal_var"]) < 0.05


def test_lifetime_sparseness_bounds():
    assert S.lifetime_sparseness(np.ones(10)) == pytest.approx(0.0)
    assert S.lifetime_sparseness(np.r_[1.0, np.zeros(9)]) == pytest.approx(1.0)


def test_auroc_equals_mann_whitney():
    rng = np.random.default_rng(3)
    A, B = rng.poisson(2, (5, 30)).astype(float), rng.poisson(1, (5, 20)).astype(float)   # ties
    auc = S.auroc_rows(A, B)
    for i in range(5):
        u = st.mannwhitneyu(A[i], B[i], alternative="two-sided").statistic
        np.testing.assert_allclose(auc[i], u / (30 * 20), rtol=1e-9)


def test_cv_selectivity_unbiased_on_noise():
    from adamacs.responsiveness import tuning as T
    rng = np.random.default_rng(4)
    ang = np.arange(0, 360, 30.0); c = np.repeat(ang, 16)
    noise = [T.cv_selectivity(rng.normal(5, 1, c.size), np.array([f"{x:g}" for x in c]), ang, "clip")["osi_cv"] for _ in range(200)]
    assert abs(np.mean(noise)) < 0.01                                 # 1 - CirVar would be biased upward here
    tuned = 5 + 4 * np.cos(np.deg2rad(2 * c)) + rng.normal(0, 1, c.size)
    assert T.cv_selectivity(tuned, np.array([f"{x:g}" for x in c]), ang, "clip")["osi_cv"] > 0.3
