"""volume_times: one time per trace sample on multi-plane scans (bug found 2026-11-24 on two-plane recordings)."""
import numpy as np
import pytest

rs = pytest.importorskip("adamacs.schemas.responsiveness")


def test_single_plane_unchanged():
    ft = np.arange(100) / 30.0
    assert np.array_equal(rs.volume_times(ft, 1, 100), ft)


def test_two_planes_one_time_per_volume():
    # 15 Hz volumes, two planes: triggers at 30 Hz
    ft = np.arange(2 * 600) / 30.0
    vt = rs.volume_times(ft, 2, 600)
    assert vt.size == 600
    assert np.allclose(np.diff(vt), 1 / 15.0)
    assert np.allclose(vt[0], (ft[0] + ft[1]) / 2)
    # the trace now spans the whole recording, not half of it
    assert vt[-1] == pytest.approx(ft[-1] - 1 / 60.0)


def test_truncated_triggers_keep_whole_volumes():
    ft = np.arange(2 * 500 + 1) / 30.0          # recording stopped mid-volume
    vt = rs.volume_times(ft, 2, 600)
    assert vt.size == 500


def test_triggers_per_volume_left_alone():
    ft = np.arange(600) / 15.0                    # one trigger per volume already
    assert np.array_equal(rs.volume_times(ft, 2, 600), ft)
