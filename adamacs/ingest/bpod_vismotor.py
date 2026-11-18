"""Bpod ingest for the VisMotor protocols (mini2p1, Anh Nguyen, 2026-11).

The generic openfield Bpod ingest (``Bpodfile.ingest``) syncs Bpod to the WaveSurfer AUX
clock with *non-timeout* trials and the ``bpod_at_target`` state of the landmark task,
taking the AUX trial TTL from analog channel 1. VisMotor protocols have none of these:
every trial is a fixed ``DotIsOn -> CueDelay -> ITI`` schedule, and Bpod emits one TTL per
trial on AUX analog channel 2 (WaveSurfer name ``BPOD 2``) at trial start.

This module subclasses ``Bpodfile`` and overrides only three things, so the database rows
are written by the same ``Bpodfile.ingest`` code every other user goes through:

1. ``subject_id`` - taken from ``session.Session``, not from the Bpod file. VisMotor files
   name the mouse after the Bpod protocol slot (e.g. ``ROS-9956``), which is not the
   PyRAT subject and would trigger a spurious PyRAT import.
2. ``_aux_timestamps`` - pairs each AUX ``BPOD 2`` pulse with a Bpod trial start by
   matching inter-trial intervals (tolerates one extra trailing pulse for an unsaved last
   trial), fits ``aux = a + b * bpod`` and refuses the file when the pulse train does not
   match its trials. That refusal is what keeps a stale ``.mat`` - copied into the folder
   of a session in which Bpod did not run - out of the database.
3. ``trial(idx)`` - VisMotor event names (``bpod_dot_on``, ``bpod_dot_off``,
   ``bpod_iti_start``, port-array inputs) instead of the landmark/lick names.

Entry point: ``ingest_bpod_vismotor(sessi, scansi, ...)``. ``behavior.ingest_bpod`` routes
here only when the scan folder contains a ``*VisMotor*.mat`` file; nothing else changes.
"""
import numpy as np
from numpy.polynomial import polynomial as P
from element_interface.utils import find_full_path

from ..pipeline import session, scan
from ..paths import get_experiment_root_data_dir
from .aux import Auxfile
from .bpod import Bpodfile, Trial
from .bpod_vismotor_sync import (VISMOTOR_PATTERN, AUX_TRIAL_CHANNEL, MAX_RESIDUAL_S,  # noqa: F401
                                  VisMotorPulseMismatch, find_vismotor_file, match_pulses_to_trials, fit_clock)

class VisMotorTrial(Trial):
    @property
    def attributes(self):
        if not self._attributes:
            self._attributes = {"protocol": VISMOTOR_PATTERN}
        return self._attributes

    @property
    def events(self):
        if not self._events:
            dot = self._states.get("DotIsOn", [None, None])
            iti = self._states.get("ITI", [None, None])
            self._events = {
                "bpod_dot_on": dot[0],
                "bpod_dot_off": dot[1],
                "bpod_iti_start": iti[0],
                **self._ports_in,        # bpod_PA1_Port<k>In, as in the generic ingest
            }
        return self._events


class VisMotorBpodfile(Bpodfile):
    def __init__(self, bpod_path, session_id):
        super().__init__(bpod_path)
        self.bpod_subject_name = self.subject_id
        self.subject_id = (session.Session & {"session_id": session_id}).fetch1("subject")

    def trial(self, idx):
        if idx not in self._trials:
            self._trials[idx] = VisMotorTrial(idx, self._bpod_path_full, self.session_data, self.trial_data)
        return self._trials[idx]

    def _aux_timestamps(self):
        aux_paths = list(self._bpod_path_full.parent.glob("*.h5"))
        assert len(aux_paths) == 1, f"Found more than one Aux h5 file\n{aux_paths}"
        aux = Auxfile(aux_paths[0])
        pulses = np.asarray(aux.bpod_channels[AUX_TRIAL_CHANNEL], float)
        starts = np.asarray(self.session_data["TrialStartTimestamp"], float)[: self.n_trials]
        bi, ai = match_pulses_to_trials(pulses, starts)
        coeffs, keep, resid = fit_clock(starts, pulses, bi, ai)
        if (~keep).any():
            print(f"VisMotor sync: excluded {(~keep).sum()} outlier pair(s) from the clock fit: "
                  f"bpod trials {bi[~keep].tolist()} (residuals {np.round(resid[~keep], 3).tolist()} s)")
        resid = resid[keep]
        drift = (coeffs[1] - 1.0) * (starts[-1] - starts[0])
        print(f"VisMotor sync: {int(keep.sum())}/{self.n_trials} trials paired with {len(pulses)} AUX pulses; "
              f"aux = {coeffs[0]:.4f} + {coeffs[1]:.8f}*bpod; drift {drift*1000:.2f} ms; "
              f"residual std {np.std(resid)*1000:.3f} ms")
        self.sync_report = {"n_pairs": int(keep.sum()), "n_excluded": int((~keep).sum()), "n_trials": int(self.n_trials), "n_pulses": int(len(pulses)),
                            "intercept": float(coeffs[0]), "slope": float(coeffs[1]),
                            "residual_std_ms": float(np.std(resid) * 1000), "drift_ms": float(drift * 1000)}
        return P.polyval(starts, coeffs), pulses


def ingest_bpod_vismotor(sessi, scansi, include_raw_bpod_events=True, include_raw_bpod_states=True, verbose=False):
    """Ingest the VisMotor Bpod file of one scan through the standard Bpodfile.ingest."""
    scan_key = (scan.Scan & {"session_id": sessi, "scan_id": scansi}).fetch1("KEY")
    scan_dir = find_full_path(get_experiment_root_data_dir(), (scan.ScanPath & scan_key).fetch1("path"))
    bpod_file = find_vismotor_file(scan_dir)
    if bpod_file is None:
        from ..ingest_log import StepSkipped
        raise StepSkipped(f"VisMotor Bpod: no *{VISMOTOR_PATTERN}*.mat session file in {scan_dir} (Bpod not run?)")
    bp = VisMotorBpodfile(bpod_file, sessi)
    try:
        bp._aux_timestamps()   # validate the pulse train before anything is written
    except VisMotorPulseMismatch as exc:
        if "did not run" in str(exc):
            from ..ingest_log import StepSkipped
            raise StepSkipped(f"VisMotor Bpod: {exc}") from exc
        raise
    bp.ingest(sessi, scansi, include_raw_bpod_events=include_raw_bpod_events,
              include_raw_bpod_states=include_raw_bpod_states)
    return getattr(bp, "sync_report", None)
