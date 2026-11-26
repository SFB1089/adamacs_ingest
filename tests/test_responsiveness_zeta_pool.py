"""Zeta process pool (2026-11-26): forkserver children and a time limit instead of a fork pool that could hang."""
import math
import multiprocessing as mp
import time

import pytest

zt = pytest.importorskip("adamacs.responsiveness.zeta_tools")


def test_map_cells_results_in_order():
    assert zt.map_cells(math.sqrt, [1.0, 4.0, 9.0, 16.0], 2, 60) == [1.0, 2.0, 3.0, 4.0]


def test_map_cells_single_process():
    assert zt.map_cells(math.sqrt, [4.0], 1, 60) == [2.0]


def test_map_cells_times_out_instead_of_hanging():
    t0 = time.time()
    with pytest.raises(mp.TimeoutError):
        zt.map_cells(time.sleep, [30, 30], 2, 2)
    assert time.time() - t0 < 20                     # the pool's children are terminated, not waited for

