"""Regression tests for the p_same leading-singleton-axis quirk.

CellReg hands back one ``(1, n_sessions, n_sessions)`` matrix per registered cell. Code
that assumes ``(n_sessions, n_sessions)`` silently produces NO pair-quality rows at all,
which is what happened until the ROS-1881 populate produced 0 links where 38,767 were
independently known.
"""
import pytest

np = pytest.importorskip("numpy")
cellreg = pytest.importorskip("adamacs.ingest.cellreg")


def _p_same(n_cells, n_sessions, *, leading_axis):
    m = np.full((n_sessions, n_sessions), 0.95)
    np.fill_diagonal(m, np.nan)
    entry = m[None, :, :] if leading_axis else m
    return [entry for _ in range(n_cells)]


@pytest.mark.parametrize("leading_axis", [True, False])
def test_consistency_accepts_both_shapes(leading_axis):
    """A (1, k, k) entry is as consistent as a (k, k) one -- this is THE bug."""
    assert cellreg.p_same_is_consistent(_p_same(3, 11, leading_axis=leading_axis), 11)


def test_consistency_rejects_wrong_session_count():
    assert not cellreg.p_same_is_consistent(_p_same(3, 7, leading_axis=True), 11)


def test_consistency_tolerates_absent_field():
    assert cellreg.p_same_is_consistent(None, 11)
    assert cellreg.p_same_is_consistent([], 11)


@pytest.mark.parametrize("leading_axis", [True, False])
def test_pair_rows_emitted_for_both_shapes(leading_axis):
    """C(k, 2) finite pairs for a cell present in every session."""
    entries = _p_same(1, 5, leading_axis=leading_axis)
    rows = cellreg.pair_quality_rows(entries[0], present_columns=range(5))
    assert len(rows) == 10
    assert all(0.0 <= p <= 1.0 for _, _, p in rows)
    assert all(a < b for a, b, _ in rows)


def test_non_finite_pairs_are_skipped_not_stored():
    """NaN p_same is omitted, so stored rows are legitimately fewer than C(k, 2).

    On the 11-session ROS-1881 run: 38,921 implied pairs = 38,767 stored + 154 NaN.
    CellRegResult.n_pairs_expected / n_pairs_nonfinite record that difference so it can
    never be mistaken for a parser fault.
    """
    m = np.full((4, 4), 0.9)
    np.fill_diagonal(m, np.nan)
    m[0, 2] = m[2, 0] = np.nan
    rows = cellreg.pair_quality_rows(m[None, :, :], present_columns=range(4))
    assert len(rows) == 5          # C(4,2) = 6, minus the one NaN pair
    assert (0, 2) not in [(a, b) for a, b, _ in rows]
