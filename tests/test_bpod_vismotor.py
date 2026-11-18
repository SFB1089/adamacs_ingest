import sys
import types

import numpy as np
import pytest

if "datajoint" not in sys.modules:
    try:
        import datajoint  # noqa: F401
    except ImportError:  # CI installs only pytest + numpy
        stub = types.ModuleType("datajoint")
        stub.config = {}
        sys.modules["datajoint"] = stub

from adamacs.ingest import bpod_vismotor_sync as bv  # noqa: E402


def test_pairs_with_extra_trailing_pulse():
    starts = np.cumsum(np.r_[35.0, np.full(39, 15.034)])
    pulses = np.r_[starts - 28.2, starts[-1] - 28.2 + 15.03]
    bi, ai = bv.match_pulses_to_trials(pulses, starts)
    assert len(bi) == 40 and bi[0] == 0 and ai[0] == 0


def test_stale_file_without_pulses_is_refused():
    with pytest.raises(bv.VisMotorPulseMismatch):
        bv.match_pulses_to_trials([], np.arange(0, 400, 10.044))


def test_wrong_schedule_is_refused():
    starts = np.arange(0, 400, 10.044)       # 10 s ITI file
    pulses = np.arange(5, 600, 15.034)       # 15 s ITI session
    with pytest.raises(bv.VisMotorPulseMismatch):
        bv.match_pulses_to_trials(pulses, starts)


def test_fit_excludes_aborted_first_trial():
    starts = np.cumsum(np.r_[35.0, 0.5, np.full(40, 15.034)])   # trial 0 aborted after 0.5 s
    pulses = starts - 28.2
    pulses[0] -= 4.4                                            # its pulse is off by seconds
    bi, ai = bv.match_pulses_to_trials(pulses, starts)
    coeffs, keep, resid = bv.fit_clock(starts, pulses, bi, ai)
    assert not keep[0] and keep[1:].all()
    assert abs(coeffs[0] + 28.2) < 1e-6
