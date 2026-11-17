"""Responsiveness schema v6: imaging QC and stimulus responsiveness.

Design: ``docs/RESPONSIVENESS_SCHEMA_PROPOSAL.md`` (v6). Scope: data quality and "does this cell
respond (to this stimulus)?" No drift / cross-session measures.

KEYS. Every result inherits the curation key (imaging.Fluorescence) and, per cell, the trace key
(imaging.Fluorescence.Trace: + mask, fluo_channel). The trace type is ``trace_source`` - a
TraceSource row points at exactly one imaging.ActivityExtractionMethod (CASCADE 25 ms, OASIS =
suite2p_deconvolution, ...) or one imaging.DffParamSet - so dF/F0 and deconvolved traces share one
key structure. Event-dependent results add the EventSet and their own parameter set, so any number
of variants coexist and every row is reachable through foreign keys.

UNIVERSAL, event-independent
    TraceSource            Lookup    trace type -> extraction method or dF/F0 recipe
    Quality(.Cell)         Computed  per curation, raw F: detector gain from dark frames, neuropil sweep;
                                     per cell kurtosis, neuropil r, photons/s
    TraceQuality(.Cell)    Computed  per curation x trace source: SNR, noise, event rate, mean level

UNIVERSAL, event-dependent
    ReadoutParamSet        Lookup    PSTH read-out: window edges (onset/offset anchored), baseline, measure,
                                     correction, test
    ZetaParamSet           Lookup    ZETA anchor, window, resampling, per condition
    ResponseTask(.Readout, .Zeta)
                           Manual    curation x trace source x event selector, and which variants to compute
    Readout(.Cell, .Condition)
                           Computed  per-onset read-out (trial_resp), test, effect, oracle, response SNR,
                                     preferred condition (plain and cross-validated)
    Zeta(.Cell, .Condition)
                           Computed  ZETA p / z / latency, per-condition p
    CellRule               Lookup    named thresholds bound to trace source / read-out / ZETA variant; cells(rule)

STIMULUS-SPECIFIC follow-ups (read Readout.Cell.trial_resp; keyed on Readout x their own paramset)
    TuningParamSet         Lookup    rectification, fit, shuffles, bootstraps
    DirectionTuning(.Cell) Computed  moving gratings / bars: 1-CirVar, 1-DirCirVar, Hotelling, direction
                                     dot-product test, double-Gaussian fit, OI/DI/OSI/DSI

PARADIGM (the only place a stimulus or task is described; built on the trial schema)
    EventSelector          Lookup    which trial events, trial types, how to read the condition, random control
    EventSet(.Onset)       Computed  onsets of one scan; each Onset is a trial.TrialEvent (-> trial.Trial with its
                                     trial_type / attributes, -> event.Event with time and stimulus string)

A new paradigm = one EventSelector row (+ a parser if its labels are new); a new trace type = one
TraceSource row. Nothing is declared on import: call :func:`activate`.
Worker: ``adamacs/workers/responsiveness_worker.py`` (ibehaveGPU1, env datajoint_ingest).
"""
from __future__ import annotations

import hashlib as _hashlib
import json as _json
import os as _os
import subprocess as _subprocess
import uuid as _uuid
import zlib as _zlib

import datajoint as dj
import numpy as _np

from adamacs.responsiveness import parsers as _parsers
from adamacs.responsiveness import photodiode as _photodiode
from adamacs.responsiveness import photon as _photon
from adamacs.responsiveness import stats as _stats
from adamacs.responsiveness import tuning as _tuning
from adamacs.responsiveness import zeta_tools as _zt

schema = dj.Schema()
_UP = {}


def _up(name):
    if name not in _UP:
        raise dj.DataJointError("responsiveness.activate() has not been called.")
    return _UP[name]


def code_version() -> str:
    here = _os.path.dirname(_os.path.abspath(__file__))
    try:
        sha = _subprocess.check_output(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                                       stderr=_subprocess.DEVNULL).decode().strip()
        dirty = _subprocess.call(["git", "-C", here, "diff", "--quiet", "--", "."]) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _hash(d) -> _uuid.UUID:
    return _uuid.UUID(_hashlib.md5(_json.dumps(d, sort_keys=True, default=str).encode()).hexdigest())


def _isnull(v):
    """True for None and NaN: fetch1("attr") returns NaN for a NULL float, fetch1() returns None."""
    return v is None or (isinstance(v, float) and not _np.isfinite(v))


def _none(v):
    """float or None (NaN/inf -> NULL)."""
    return None if v is None or not _np.isfinite(v) else float(v)


def _scan_key(key):
    return {"session_id": key["session_id"], "scan_id": key["scan_id"]}


def _fluo_key(key):
    return {k: key[k] for k in _up("imaging").Fluorescence.primary_key}


def frame_times(key):
    return (_up("event").Event & _scan_key(key) & 'event_type LIKE "%%2p_frames%%"').fetch(
        "event_start_time", order_by="event_start_time").astype(float)


def fps(key):
    return float((_up("scan").ScanInfo & _scan_key(key)).fetch1("fps"))


# ======================================================================================
# Trace types
# ======================================================================================

@schema
class TraceSource(dj.Lookup):
    definition = """
    # A trace type: exactly one imaging activity extraction method or one dF/F0 recipe
    trace_source            : varchar(24)
    ---
    -> [nullable] imaging.ActivityExtractionMethod
    -> [nullable] imaging.DffParamSet
    trace_unit              : varchar(16)            # 'dF/F0' | 'spikes/s' | 'a.u.'
    """
    contents = [("dff0_r2", None, 2, "dF/F0"),
                ("cascade25", "cascade_inference_25ms_filter", None, "spikes/s"),
                ("oasis", "suite2p_deconvolution", None, "a.u.")]


def available():
    """imaging.Fluorescence x TraceSource restricted to the combinations whose traces exist."""
    imaging = _up("imaging")
    pairs = imaging.Fluorescence.proj() * TraceSource.proj("extraction_method", "dff_paramset_idx")
    return pairs & [imaging.Activity.proj(), imaging.Dff0.proj()]


_TRACE_CACHE = {}


def traces(key):
    """(trace_keys, X[n_traces, n_frames]) of the trace source in ``key``; ordered by mask, fluo_channel.
    trace_keys are imaging.Fluorescence.Trace keys. The last trace set is kept in memory: populate visits the
    read-outs / ZETA variants of one trace set consecutively, so each set is downloaded once."""
    fk = _fluo_key(key)
    ck = (tuple(sorted(fk.items())), key["trace_source"])
    if ck not in _TRACE_CACHE:
        _TRACE_CACHE.clear()
        _TRACE_CACHE[ck] = _traces(key)
    tk, X = _TRACE_CACHE[ck]
    return [dict(t) for t in tk], X


def _traces(key):
    imaging = _up("imaging")
    src = (TraceSource & key).fetch1()
    fk = _fluo_key(key)
    if src["extraction_method"] is not None:
        tk, X = (imaging.Activity.Trace & fk & {"extraction_method": src["extraction_method"]}).fetch(
            "KEY", "activity_trace", order_by="mask, fluo_channel")
    else:
        tk, X = (imaging.Dff0.Trace & fk & {"dff_paramset_idx": src["dff_paramset_idx"]}).fetch(
            "KEY", "dff", order_by="mask, fluo_channel")
    keep = imaging.Fluorescence.Trace.primary_key
    n = min(len(x) for x in X)
    return [{k: t[k] for k in keep} for t in tk], _np.vstack([_np.asarray(x, float)[:n] for x in X])


# ======================================================================================
# UNIVERSAL, event-independent
# ======================================================================================

@schema
class ScanQuality(dj.Computed):
    definition = """
    # Detector calibration of one scan from its raw TIFF (independent of processing and curation):
    # gain from the pixel-intensity distribution, referenced to shutter-closed dark frames (Lees et al. 2025)
    -> scan.Scan
    ---
    gain_method             : enum('dark','ptc')  # dark: offset + read noise from shutter-closed frames; ptc: free-intercept fit (no dark frames)
    gain=null               : double              # [grey levels / detected photon] (V - dark_var) = gain*(I - offset), frame-pair variance
    offset=null             : double              # [grey levels] zero-photon level (mean dark frame, or fitted intercept)
    gain_r2=null            : float               # fit quality over the lower half of the intensity range
    n_frames                : int                 # raw frames used for the fit (dark frames come in addition)
    n_dark_frames=0         : int                 # shutter-closed frames used (0: no dark frames, ptc fallback)
    """

    N_FRAMES = 500       # bright frames for the transfer curve

    @property
    def key_source(self):
        return _up("scan").Scan & ResponseTask

    def make(self, key):
        from element_calcium_imaging import dff as _dff
        import tifffile
        imaging, scan = _up("imaging"), _up("scan")
        info = (scan.ScanInfo & key).fetch1()
        files = (scan.ScanInfo.ScanFile & key).fetch("file_path", order_by="file_path")
        # Read only the frames needed, page by page (channel 0, depth 0, field 0). scanreader would load the
        # whole field first: 214 GB for a 147,300-frame scan in the 2026-11-15 pilot.
        step = int(info["nchannels"] or 1) * int(info["ndepths"] or 1)
        with tifffile.TiffFile(files[0]) as tf:
            def frames(f0, f1):
                return _np.stack([tf.pages[f * step].asarray() for f in range(f0, f1)]).astype(_np.float64)
            ft, starts, ends = imaging._shutter_inputs(key)
            dark = None
            if ft is not None:
                try:
                    a, b = _dff.shutter_dark_frames(ft, starts, ends, 0.09)
                    dark = (a, b) if b - a >= 3 else None
                except ValueError:
                    dark = None
            b0 = (dark[1] if dark else 0) + 10
            n = int(min(self.N_FRAMES, int(info["nframes"]) - b0))
            I, V, C = _photon.ptc_bins(frames(b0, b0 + n))
            g = _np.nan
            if dark:
                off, dvar = _photon.dark_stats(frames(dark[0], dark[1]))
                g, r2 = _photon.fit_dark_referenced(I, V, C, off, dvar)
                method = "dark"
            if not (_np.isfinite(g) and g > 0):
                g, off, r2 = _photon.fit_ptc(I, V, C)
                method = "ptc"
            if not (_np.isfinite(g) and g > 0):          # no usable transfer curve: store NULL, never a negative gain
                g = off = _np.nan
        self.insert1({**key, "gain_method": method, "gain": _none(g), "offset": _none(off), "gain_r2": _none(r2),
                      "n_frames": n, "n_dark_frames": int(dark[1] - dark[0]) if dark and method == "dark" else 0})


@schema
class Quality(dj.Computed):
    definition = """
    # Data fidelity of one curation from raw fluorescence (Labrigger 2026 catechism); detector gain from ScanQuality
    -> imaging.Fluorescence
    ---
    alpha_opt=null          : float               # neuropil factor that brings the mean pairwise cell correlation closest to 0
    mean_r_at_alpha07=null  : float               # mean pairwise correlation at the pipeline's 0.7
    n_sweep_cells=0         : int                 # soma masks (channel 0, raw kurtosis > 5) in the sweep
    """

    class Cell(dj.Part):
        definition = """
        -> master
        -> imaging.Fluorescence.Trace
        ---
        kurtosis=null           : float           # Pearson kurtosis of raw F (normal 3); low = noise or neuropil-dominated
        neuropil_r=null         : float           # r(F, Fneu); high = contaminated
        photons_per_s=null      : float           # enf^2*(F0 - offset)*npix/gain*fps, F0 = 30th pct of raw F, gain/offset from ScanQuality
        """

    ENF = 1.2            # PMT excess-noise factor (kurtosis-gain-checker convention)
    SWEEP_BIN_S = 1.0    # neuropil is slow: correlate 1 s bins

    @property
    def key_source(self):
        return (_up("imaging").Fluorescence & ResponseTask) & ScanQuality

    def make(self, key):
        imaging = _up("imaging")
        sq = (ScanQuality & _scan_key(key)).fetch1()
        g = sq["gain"] if not _isnull(sq["gain"]) else _np.nan
        off = sq["offset"] if not _isnull(sq["offset"]) else _np.nan
        f = fps(key)
        rows = (imaging.Fluorescence.Trace * imaging.Segmentation.Mask.proj("mask_npix") & key).fetch(
            "mask", "fluo_channel", "fluorescence", "neuropil_fluorescence", "mask_npix", order_by="mask, fluo_channel")
        cells = []
        for m, ch, F, Fn, npx in zip(*rows):
            pps = _photon.photons_per_s(float(_np.nanpercentile(F, 30)), off, int(npx), g, self.ENF, f) \
                if _np.isfinite(g) and g > 0 else _np.nan
            cells.append({**key, "mask": int(m), "fluo_channel": int(ch), "kurtosis": _none(_stats.kurtosis(F)),
                          "neuropil_r": _none(_stats.pearson_r(F, Fn)), "photons_per_s": _none(pps)})
        # neuropil sweep: soma masks of channel 0 with raw kurtosis > 5 (kurtosis-gain-checker)
        soma = set((imaging.MaskClassification.MaskType & key & 'mask_type="soma"').fetch("mask").tolist())
        use = [i for i, c in enumerate(cells)
               if c["fluo_channel"] == 0 and c["mask"] in soma and (c["kurtosis"] or 0) > 5]
        a_opt = r07 = None
        if len(use) >= 3:
            T = min(len(rows[2][i]) for i in use)
            a, mr = _stats.neuropil_sweep(_np.vstack([rows[2][i][:T] for i in use]),
                                          _np.vstack([rows[3][i][:T] for i in use]),
                                          bin_frames=max(1, int(round(self.SWEEP_BIN_S * f))))
            a_opt = float(a[_np.nanargmin(_np.abs(mr))]); r07 = _none(mr[_np.argmin(_np.abs(a - 0.7))])
        self.insert1({**key, "alpha_opt": a_opt, "mean_r_at_alpha07": r07, "n_sweep_cells": len(use)})
        self.Cell.insert(cells)


@schema
class TraceQuality(dj.Computed):
    definition = """
    # Event-independent activity statistics of one trace type of one curation (ACTIVE level)
    -> imaging.Fluorescence
    -> TraceSource
    """

    class Cell(dj.Part):
        definition = """
        -> master
        -> imaging.Fluorescence.Trace
        ---
        snr_s2p=null            : float           # Suite2p: 1 - var(diff x)/(2 var x)
        noise_nu=null           : float           # [percent/sqrt(Hz)] CASCADE noise level (meaningful for dF/F0 only)
        event_rate=null         : float           # [1/min] runs >= 2 frames above 3 robust SD
        mean_level=null         : float           # mean of the trace (for spike-rate traces: mean rate)
        max_value=null          : float           # maximum of the trace (audit rule: CASCADE cells > 50 are non-physical)
        """

    @property
    def key_source(self):
        return available() & ResponseTask

    def make(self, key):
        f = fps(key)
        tk, X = traces(key)
        self.insert1(key)
        self.Cell.insert({**key, **k, "snr_s2p": _none(_stats.snr_suite2p(x)), "noise_nu": _none(_stats.noise_nu(x, f)),
                          "event_rate": _none(_stats.event_rate(x, f)), "mean_level": _none(_np.nanmean(x)),
                          "max_value": _none(_np.nanmax(x))}
                         for k, x in zip(tk, X))


# ======================================================================================
# PARADIGM: EventSelector -> EventSet  (the only stimulus- or task-specific tables)
# ======================================================================================

_SEL_COLS = ("event_selector", "event_type_like", "trial_type_like", "condition_source", "condition_parser",
             "parser_args", "min_repeats", "random_seed", "circular_period")


@schema
class EventSelector(dj.Lookup):
    definition = """
    # How one paradigm's trial events become onsets with a condition label. Onsets are trial.TrialEvent rows:
    # bench2p trial = stimulus block (trial_type e.g. 'Moving_Gratings'), Bpod trial = task trial. Never edited.
    event_selector          : varchar(48)
    ---
    event_type_like         : varchar(128)           # onset events among trial.TrialEvent (LIKE on event_type)
    trial_type_like=''      : varchar(128)           # optional: only trials of this type (trial.Trial.trial_type)
    condition_source='event_type' : enum('event_type','trial_type','trial_attribute')
                                                     # stimulus string of the event | its trial's type | a trial.Trial.Attribute
    trial_attribute=''      : varchar(16)            # attribute_name when condition_source = 'trial_attribute'
    condition_parser='none' : varchar(32)            # adamacs.responsiveness.parsers; 'none' = one condition
    parser_args=''          : varchar(64)            # e.g. bonsai_fields '2,3' -> label 'x|y'
    min_repeats=1           : smallint               # drop conditions shown fewer times
    random_seed=null        : int                    # set = no-stimulus control: every onset shifted to a seeded random time
    circular_period=null    : float                  # [deg] 360: labels are directions (DirectionTuning applies); 180: orientations
    stim_duration_s=null    : float                  # [s] stimulus end = onset + this; NULL = event.Event.event_end_time
                                                     # (set it where event_end_time is a trigger, not the stimulus end)
    display_latency_s=0.1   : float                  # [s] image on screen after the DataJoint onset (photodiode, v1-drift-audit
                                                     # notebook 04: 90-110 ms, IQR ~20 ms; = Leon's +3 frames); used when time_ref='display'
    """
    # RF labels: Bonsai fields 2, 3 (, 4) = azimuth | elevation (| size); field map in adamacs.responsiveness.parsers
    contents = [dict(zip(_SEL_COLS, r)) for r in [
        ("gratings", "Moving_Gratings:%", "", "event_type", "bonsai_semicolon", "", 2, None, 360.0),
        ("gratings_orientation", "Moving_Gratings:%", "", "event_type", "bonsai_orientation", "", 2, None, 180.0),
        ("gratings_random", "Moving_Gratings:%", "", "event_type", "bonsai_semicolon", "", 2, 0, None),
        ("moving_bars", "Moving_Bars:%", "", "event_type", "bonsai_semicolon", "", 2, None, 360.0),
        ("nat_images", "Natural Image:%", "", "event_type", "image_number", "", 1, None, None),
        ("nat_images_repeated", "Natural Image:%", "", "event_type", "image_number", "", 5, None, None),
        ("nat_images_repeated_random", "Natural Image:%", "", "event_type", "image_number", "", 5, 0, None),
        ("rf_checkerboard", "Appearing_Checkerboard_Noise:%", "", "event_type", "bonsai_fields", "2,3", 2, None, None),
        ("rf_checkerboard_random", "Appearing_Checkerboard_Noise:%", "", "event_type", "bonsai_fields", "2,3", 2, 0, None),
        ("rf_classical", "Classical Receptive Field%", "", "event_type", "bonsai_fields", "2,3,4", 2, None, None),
    ]]


@schema
class EventSet(dj.Computed):
    definition = """
    # The onsets one EventSelector picks among the trial events of one scan (computed for tasked scans)
    -> scan.Scan
    -> EventSelector
    ---
    n_onsets                : int
    n_conditions            : smallint
    n_trials                : smallint               # trials (blocks / task trials) the onsets come from
    n_outside_trials        : int                    # matching event.Event rows in no trial (not used)
    median_duration_s=null  : float                  # median stimulus duration used (stim_duration_s, or event end - start)
    median_ioi_s=null       : float                  # median onset-to-onset interval
    """

    class Onset(dj.Part):
        definition = """
        # one onset = one trial event; time, stimulus string and trial come through the foreign key
        -> master
        -> trial.TrialEvent
        ---
        onset_idx               : smallint           # presentation order in this set = index into Readout.Cell.trial_resp
        condition               : varchar(64)        # parsed label ('all' if none)
        shift_s=0               : double             # [s] added to the event time: 0, or the seeded shift of a random-onset control
        unique index (session_id, scan_id, event_selector, onset_idx)
        """

    @property
    def key_source(self):
        return (_up("scan").Scan * EventSelector) & ResponseTask

    def make(self, key):
        event, trial = _up("event"), _up("trial")
        sel = (EventSelector & key).fetch1()
        sk = _scan_key(key)
        pat = sel["event_type_like"].replace("%", "%%")
        q = trial.TrialEvent * trial.Trial * event.Event & sk & f'event_type LIKE "{pat}"'
        if sel["trial_type_like"]:
            q = q & f'trial_type LIKE "{sel["trial_type_like"].replace("%", "%%")}"'
        rows = q.fetch("KEY", "event_start_time", "event_end_time", "event_type", "trial_id", "trial_type",
                       order_by="event_start_time", as_dict=False)
        ek, t0, t1, et, tid, ttype = rows
        t0 = _np.asarray(t0, float)
        t1 = _np.array([_np.nan if v is None else float(v) for v in t1])
        tek = trial.TrialEvent.primary_key
        ek = [{k: e[k] for k in tek} for e in ek]
        parse = _parsers.get(sel["condition_parser"])
        if sel["condition_source"] == "event_type":
            src = list(et)
        elif sel["condition_source"] == "trial_type":
            src = list(ttype)
        else:
            attrs = dict(zip(*(trial.Trial.Attribute & sk & {"attribute_name": sel["trial_attribute"]}).fetch(
                "trial_id", "attribute_value")))
            src = [attrs.get(t) for t in tid]
        ok = _np.array([x is not None for x in src], bool)
        cond = _np.array([parse(x, sel["parser_args"]) if x is not None else "" for x in src], dtype=object).astype(str)
        if sel["min_repeats"] > 1 and ok.any():
            u, c = _np.unique(cond[ok], return_counts=True)
            ok &= _np.isin(cond, u[c >= sel["min_repeats"]])
        ii = _np.flatnonzero(ok)
        t0, t1, cond = t0[ii], t1[ii], cond[ii]
        ek = [ek[i] for i in ii]; tid = _np.asarray(tid)[ii]
        if sel["stim_duration_s"] is not None:
            t1 = t0 + float(sel["stim_duration_s"])
        dur = _np.nanmedian(t1 - t0) if t0.size and _np.isfinite(t1).any() else _np.nan
        ioi = float(_np.median(_np.diff(t0))) if t0.size > 1 else _np.nan
        shift = _np.zeros(t0.size)
        if sel["random_seed"] is not None and t0.size:
            ft = frame_times(key)
            w = ioi if _np.isfinite(ioi) else 1.0
            rng = _np.random.default_rng((int(sel["random_seed"]) + _zlib.crc32(key["scan_id"].encode())) % 2 ** 32)
            new = _np.sort(rng.uniform(ft[0] + 2 * w + 1, ft[-1] - 3 * w - (dur if _np.isfinite(dur) else 0) - 1, t0.size))
            shift = new - t0
            cond = _np.array(["all"] * t0.size)
        n_all = len(event.Event & sk & f'event_type LIKE "{pat}"')
        self.insert1({**key, "n_onsets": int(t0.size), "n_conditions": int(len(_np.unique(cond))),
                      "n_trials": int(len(_np.unique(tid))), "n_outside_trials": int(n_all - len(q)),
                      "median_duration_s": _none(dur), "median_ioi_s": _none(ioi)})
        order = _np.argsort(t0 + shift, kind="stable")
        self.Onset.insert({**key, **ek[j], "onset_idx": i, "condition": str(cond[j]), "shift_s": float(shift[j])}
                          for i, j in enumerate(order))


@schema
class ScreenTiming(dj.Imported):
    definition = """
    # When the stimulus was physically on screen: photodiode flips in the scan's aux .h5 (diode from Nov 2025 on).
    # Algorithm of the v1-drift-audit (notebook 04), adamacs.responsiveness.photodiode.
    -> EventSet
    ---
    has_diode               : bool                # the aux file has a 'Photodiode' channel
    n_detected              : int                 # onsets with a physical (LED) onset found in [on - 150 ms, on + 400 ms]
    latency_s=null          : float               # [s] median (photodiode onset - DataJoint onset)
    latency_iqr_s=null      : float               # [s] interquartile range of the per-onset latency
    refresh_hz=null         : float               # display refresh rate seen by the diode
    aux_file=''             : varchar(255)        # event.BehaviorRecording.File read
    """

    class Onset(dj.Part):
        definition = """
        -> master
        -> EventSet.Onset
        ---
        diode_on=null           : double          # [s] physical onset (first bright->dark flip of the corner square)
        diode_off=null          : double          # [s] physical offset (next dark->bright flip)
        """

    @property
    def key_source(self):
        return EventSet & (EventSelector & "random_seed IS NULL")

    def make(self, key):
        event = _up("event")
        files = (event.BehaviorRecording.File & _scan_key(key)).fetch("filepath")
        q = EventSet.Onset & key
        ok, t0 = q.fetch("KEY", "event_start_time", order_by="onset_idx")
        t0 = _np.asarray(t0, float)
        res = _photodiode.screen_times(files[0], t0) if len(files) else {"has_diode": False}
        if not res["has_diode"] or not t0.size:
            self.insert1({**key, "has_diode": bool(res["has_diode"]), "n_detected": 0,
                          "aux_file": str(files[0]) if len(files) else ""})
            return
        lat = res["on"] - t0
        self.insert1({**key, "has_diode": True, "n_detected": int(_np.isfinite(lat).sum()),
                      "latency_s": _none(_np.nanmedian(lat)), "aux_file": str(files[0]),
                      "latency_iqr_s": _none(_np.subtract(*_np.nanpercentile(lat, [75, 25]))) if _np.isfinite(lat).any() else None,
                      "refresh_hz": _none(res["refresh_hz"])})
        self.Onset.insert({**k, "diode_on": _none(a), "diode_off": _none(b)} for k, a, b in zip(ok, res["on"], res["off"]))


# ======================================================================================
# UNIVERSAL, event-dependent: parameter sets
# ======================================================================================

LENGTH_FREE = ("mean",)       # measures comparable between a response and a baseline window of different length

_RCOLS = ("time_ref", "window_unit", "resp_start_ref", "resp_start", "resp_stop_ref", "resp_stop", "resp_length",
          "base_start_ref", "base_start", "base_stop_ref", "base_stop", "base_length", "measure", "smooth_s", "correction", "test")
_READOUTS = {
    1: ("stimulus mean: response [onset, offset), baseline [onset - 0.5 s, onset); screen time",
        "display", "s", "onset", 0.0, "offset", 0.0, "per_onset", "onset", -0.5, "onset", 0.0, "explicit",
        "mean", 0.0, "subtract", "ttest"),
    2: ("Leon export, gratings (v1-drift-audit 234/234): frames [on, on + n), n = off - on + 2 of the first trial; control the same "
        "length ending 1 frame before onset; DataJoint event time; raw mean; one-sided t (audit p_resp)",
        "event", "frames", "onset", 0, "offset", 2, "first_onset", "onset", None, "onset", -1, "match_response",
        "mean", 0.0, "none", "ttest_greater"),
    3: ("Leon export, natural images (v1-drift-audit 84/84): frames [on + 3, on + 3 + n), n = off - on + 2 of the first trial; "
        "DataJoint event time (+3 frames = display latency); no baseline; raw mean",
        "event", "frames", "onset", 3, "offset", 5, "first_onset", None, None, None, None, "explicit",
        "mean", 0.0, "none", "wilcoxon"),
    4: ("early: response [onset, onset + 0.5 s), baseline [onset - 0.5 s, onset); screen time",
        "display", "s", "onset", 0.0, "onset", 0.5, "per_onset", "onset", -0.5, "onset", 0.0, "explicit",
        "mean", 0.0, "subtract", "ttest"),
    5: ("OFF: response [offset, offset + 1.0 s), baseline [onset - 0.5 s, onset); screen time",
        "display", "s", "offset", 0.0, "offset", 1.0, "per_onset", "onset", -0.5, "onset", 0.0, "explicit",
        "mean", 0.0, "subtract", "ttest"),
    6: ("peak of a 200 ms boxcar in [onset, offset) minus the same peak in the equally long window ending at onset, "
        "in units of the baseline frame SD; screen time",
        "display", "s", "onset", 0.0, "offset", 0.0, "per_onset", None, None, "onset", 0.0, "match_response",
        "peak", 0.2, "zscore", "ttest"),
    7: ("Leon export, CASCADE 'opt' trees (v1-drift-audit opt_json): frames [on + 1, on + 15); DataJoint event time; no baseline",
        "event", "frames", "onset", 1, "onset", 15, "per_onset", None, None, None, None, "explicit",
        "mean", 0.0, "none", "wilcoxon"),
    8: ("baseline [onset - 1.0 s, onset - 0.1 s), response [onset + 0.1 s, onset + 2.0 s); mean; subtract; screen time",
        "display", "s", "onset", 0.1, "onset", 2.0, "per_onset", "onset", -1.0, "onset", -0.1, "explicit",
        "mean", 0.0, "subtract", "ttest"),
    9: ("peak of a 100 ms boxcar: response [onset + 0.1 s, onset + 2.0 s) minus the same peak in the equally long baseline "
        "[onset - 2.0 s, onset - 0.1 s); screen time",
        "display", "s", "onset", 0.1, "onset", 2.0, "per_onset", None, None, "onset", -0.1, "match_response",
        "peak", 0.1, "subtract", "ttest"),
    10: ("standard: response = stimulus on screen, frames [on, off); baseline = 30 frames (1 s at 30 Hz) ending 2 frames before "
         "onset, frames [on - 32, on - 2) (CASCADE's non-causal kernel leaks ~2 frames before a response); screen time; mean; t-test",
         "display", "frames", "onset", 0, "offset", 0, "per_onset", "onset", -32, "onset", -2, "explicit",
         "mean", 0.0, "subtract", "ttest"),
    11: ("CASCADE images (v1-drift-audit q1 consensus window, leave-one-animal-out on the oracle): frames [on - 1, on + 15); "
         "DataJoint event time; raw mean, no baseline (starts 1 frame early: CASCADE's non-causal kernel)",
         "event", "frames", "onset", -1, "onset", 15, "per_onset", None, None, None, None, "explicit",
         "mean", 0.0, "none", "ttest"),
    12: ("CASCADE images, common-default start (Slack 2026-09-24: start +1 GCaMP8 / -1 GCaMP6 -> 0 as common default), "
         "consensus length 16: frames [on, on + 16); DataJoint event time; raw mean, no baseline",
         "event", "frames", "onset", 0, "onset", 16, "per_onset", None, None, None, None, "explicit",
         "mean", 0.0, "none", "ttest"),
    13: ("CASCADE images, GCaMP8 start (Slack 2026-09-24: +1), consensus length 16: frames [on + 1, on + 17); "
         "DataJoint event time; raw mean, no baseline (the GCaMP6 start -1 is read-out 11)",
         "event", "frames", "onset", 1, "onset", 17, "per_onset", None, None, None, None, "explicit",
         "mean", 0.0, "none", "ttest"),
    14: ("Leon MEI restimulation (Slack 2026-11-12, dF/F0): response frames [on + 3, off + 3) minus the mean of the 3 frames "
         "[on, on + 3); DataJoint event time; t-test",
         "event", "frames", "onset", 3, "offset", 3, "per_onset", "onset", 0, "onset", 3, "explicit",
         "mean", 0.0, "subtract", "ttest"),
    15: ("as 10 (stimulus on screen [on, off) vs 30 frames ending 2 frames before onset), but the per-trial value is the RAW "
         "response mean (no subtraction; the baseline is used only by the test) - Leon's / the audit's convention for tuning curves",
         "display", "frames", "onset", 0, "offset", 0, "per_onset", "onset", -32, "onset", -2, "explicit",
         "mean", 0.0, "none", "ttest"),
    16: ("gratings standard: response = stimulus on screen, frames [on, off), RAW per-trial value; baseline (test only) = frames "
         "[on - 60, on - 10) = -2.0 to -0.33 s: after the previous stimulus' OFF response has decayed (< 10 percent from -63 frames on "
         "for dF/F0, the slowest trace) and before CASCADE's non-causal pre-onset rise (from -7 frames); screen time; t-test",
         "display", "frames", "onset", 0, "offset", 0, "per_onset", "onset", -60, "onset", -10, "explicit",
         "mean", 0.0, "none", "ttest"),
}


@schema
class ReadoutParamSet(dj.Lookup):
    definition = """
    # Peri-event (PSTH) read-out, independent of trace and paradigm. The response and the baseline window each have an
    # explicit start and end: edge = (stimulus onset or offset) + value. Never edited; same content -> same hash.
    readout_idx             : smallint
    ---
    readout_desc            : varchar(400)
    readout_hash            : uuid
    unique index (readout_hash)
    time_ref='display'      : enum('display','event')       # display: event time + EventSelector.display_latency_s (the image on
                                                            # screen; photodiode 90-110 ms); event: DataJoint event time as stored
    window_unit='s'         : enum('s','frames')            # s: seconds; frames: 2P frames from the frame NEAREST each reference time
    resp_start_ref          : enum('onset','offset')        # response window START = stimulus onset or offset ...
    resp_start              : float                         # ... + this (s or frames)
    resp_stop_ref           : enum('onset','offset')        # response window END (exclusive) = stimulus onset or offset ...
    resp_stop               : float                         # ... + this (e.g. 0 = at stimulus end; -0.2 = 200 ms before)
    resp_length='per_onset' : enum('per_onset','first_onset')   # first_onset: every response window as long as the first one (Leon export)
    base_start_ref=null     : enum('onset','offset')        # baseline window START reference (NULL: no baseline; tests against 0)
    base_start=null         : float                         # ... + this
    base_stop_ref=null      : enum('onset','offset')        # baseline window END (exclusive) reference
    base_stop=null          : float                         # ... + this
    base_length='explicit'  : enum('explicit','match_response')   # match_response: baseline START = baseline END - response length
    measure='mean'          : enum('mean','median','integral','peak','trough','peak_abs','auc_pos','peak_latency')
    smooth_s=0              : float                         # [s] boxcar before peak / trough / peak_abs / peak_latency
    correction='subtract'   : enum('subtract','divide','zscore','none')   # r-b | (r-b)/|b| | (r-b)/baseline frame SD | r
    test='wilcoxon'         : enum('wilcoxon','ttest','ttest_greater','signflip')   # paired response vs baseline; wilcoxon / signflip
                                                    # assume a symmetric difference: equal windows only (pilot: t-test)
    """
    contents = [{"readout_idx": i, "readout_desc": v[0], "readout_hash": _hash(dict(zip(_RCOLS, v[1:]))),
                 **dict(zip(_RCOLS, v[1:]))} for i, v in _READOUTS.items()]


_ZCOLS = ("time_ref", "zeta_anchor", "zeta_window_s", "zeta_resampling", "per_condition", "mask_type")
_ZETAS = {
    1: ("v1-drift-audit t73/t140: DataJoint event time, window = median onset-to-onset interval, 100 resamples, pooled, soma masks",
        "event", "onset", None, 100, 0, "soma"),
    2: ("as 1 + one-sample ZETA per condition, soma masks only", "event", "onset", None, 100, 1, "soma"),
    3: ("as 1, screen time (event + display latency), soma masks only", "display", "onset", None, 100, 0, "soma"),
}


@schema
class ZetaParamSet(dj.Lookup):
    definition = """
    # Time-series ZETA settings (zetapy). Response shape free; only the analysis window is chosen. Never edited.
    zeta_idx                : smallint
    ---
    zeta_desc               : varchar(255)
    zeta_hash               : uuid
    unique index (zeta_hash)
    time_ref='event'        : enum('display','event')       # as ReadoutParamSet.time_ref (shifts latencies, not p-values)
    zeta_anchor='onset'     : enum('onset','offset')
    zeta_window_s=null      : float                         # [s] analysis window; NULL = median onset-to-onset interval of the REAL
                                                            # events (EventSet.median_ioi_s; random-onset sets keep the real window, as t73)
    zeta_resampling=100     : smallint
    per_condition=0         : bool                          # also one-sample ZETA per condition
    mask_type=''            : varchar(16)                   # imaging.MaskClassification type to test ('soma'); '' = every mask
    """
    contents = [{"zeta_idx": i, "zeta_desc": v[0], "zeta_hash": _hash(dict(zip(_ZCOLS, v[1:]))),
                 **dict(zip(_ZCOLS, v[1:]))} for i, v in _ZETAS.items()]


_TCOLS = ("angle_convention", "rectify", "fit", "n_perm", "n_boot")
_TUNINGS = {
    1: ("v1-drift-audit convention theta = (90 - label) mod 360; Mazurek et al. 2014 measures on condition means clipped at 0; "
        "double-Gaussian fit; 1000 label shuffles; 200 bootstraps", "90_minus_label", "clip", 1, 1000, 200),
    2: ("as 1 without rectification (audit OSI = |F2|/F0, DSI = |F1|/F0 exactly)", "90_minus_label", "none", 1, 1000, 200),
    3: ("as 1 with the motion angle measured in the Bonsai logs, theta = -label", "minus_label", "clip", 1, 1000, 200),
}


@schema
class TuningParamSet(dj.Lookup):
    definition = """
    # Settings of the stimulus-specific tuning follow-ups (DirectionTuning). Never edited; same content -> same hash.
    tuning_idx              : smallint
    ---
    tuning_desc             : varchar(255)
    tuning_hash             : uuid
    unique index (tuning_hash)
    angle_convention='90_minus_label' : enum('90_minus_label','minus_label','label')   # stimulus label -> angle (all output angles)
    rectify='clip'          : enum('clip','none')   # condition means < 0 set to 0 before vector / point indices
    fit=1                   : bool                  # double-Gaussian fit (Mazurek et al. 2014 constraints)
    n_perm=1000             : smallint              # label shuffles for the vector-length p (0 = skip)
    n_boot=200              : smallint              # bootstraps for the preferred-orientation interval (0 = skip)
    """
    contents = [{"tuning_idx": i, "tuning_desc": v[0], "tuning_hash": _hash(dict(zip(_TCOLS, v[1:]))),
                 **dict(zip(_TCOLS, v[1:]))} for i, v in _TUNINGS.items()]


# ======================================================================================
# What to compute
# ======================================================================================

@schema
class ResponseTask(dj.Manual):
    definition = """
    # Quantify these onsets on this trace type of this curation (staged with plan_tasks())
    -> imaging.Fluorescence
    -> TraceSource
    -> EventSelector
    ---
    task_owner              : varchar(32)
    """

    class Readout(dj.Part):
        definition = """
        # read-outs to compute for this task
        -> master
        -> ReadoutParamSet
        """

    class Zeta(dj.Part):
        definition = """
        # ZETA variants to compute for this task
        -> master
        -> ZetaParamSet
        """

    class Tuning(dj.Part):
        definition = """
        # stimulus-specific tuning follow-ups: on this read-out, with these settings
        -> master.Readout
        -> TuningParamSet
        """


def plan_tasks(restriction, trace_sources, selectors, readouts, zetas, owner, tunings=()):
    """Stage work for the curations in ``imaging.Fluorescence & restriction``.

    One ResponseTask per curation x trace source x selector whose events exist among the scan's
    trial events (trial.TrialEvent), with
    the requested read-outs and ZETA variants; ``tunings`` = [(readout_idx, tuning_idx), ...] for the
    stimulus-specific follow-ups (only selectors that qualify, e.g. circular_period 360 for DirectionTuning). Deconvolved traces must already exist (no
    imaging.Activity -> skipped); dF/F0 sources get an imaging.Dff0Task so the worker computes them.
    Returns the number of new ResponseTasks.
    """
    imaging, trial = _up("imaging"), _up("trial")
    dff_tasks, tasks, rparts, zparts, tparts = [], [], [], [], []
    for fk in (imaging.Fluorescence & restriction).fetch("KEY"):
        for ts in trace_sources:
            src = (TraceSource & {"trace_source": ts}).fetch1()
            if src["extraction_method"] is not None:
                if not (imaging.Activity & fk & {"extraction_method": src["extraction_method"]}):
                    continue
            else:
                dff_tasks.append({**fk, "dff_paramset_idx": src["dff_paramset_idx"], "task_owner": owner,
                                  "task_note": "responsiveness"})
            for s in selectors:
                pat = (EventSelector & {"event_selector": s}).fetch1("event_type_like").replace("%", "%%")
                if not (trial.TrialEvent & _scan_key(fk) & f'event_type LIKE "{pat}"'):
                    continue
                tk = {**fk, "trace_source": ts, "event_selector": s}
                tasks.append({**tk, "task_owner": owner})
                rparts += [{**tk, "readout_idx": i} for i in readouts]
                zparts += [{**tk, "zeta_idx": i} for i in zetas]
                tparts += [{**tk, "readout_idx": r, "tuning_idx": t} for r, t in tunings if r in readouts]
    if dff_tasks:
        imaging.Dff0Task.insert(dff_tasks, skip_duplicates=True)
    n0 = len(ResponseTask())
    ResponseTask.insert(tasks, skip_duplicates=True)
    ResponseTask.Readout.insert(rparts, skip_duplicates=True)
    ResponseTask.Zeta.insert(zparts, skip_duplicates=True)
    ResponseTask.Tuning.insert(tparts, skip_duplicates=True)
    return len(ResponseTask()) - n0


# ======================================================================================
# Shared by Readout and Zeta
# ======================================================================================

def _inputs(key):
    """Traces and onsets of one task: (trace_keys, X, frame_times, onset, offset, condition, info).

    Times are DataJoint event times (DAQ clock, plus shift_s for random-onset controls); info holds the selector's
    display latency and the median onset interval of the REAL events (also for random-onset sets).
    """
    tk, X = traces(key)
    ft = frame_times(key)
    n = min(ft.size, X.shape[1])
    es = {k: key[k] for k in EventSet.primary_key}
    q = EventSet.Onset * _up("event").Event.proj("event_end_time") & es
    t0, t1, sh, cond = q.fetch("event_start_time", "event_end_time", "shift_s", "condition", order_by="onset_idx")
    sh = _np.asarray(sh, float)
    t0 = _np.asarray(t0, float) + sh
    sel = (EventSelector & es).fetch1()                     # whole-row fetch1: NULL -> None
    if not _isnull(sel["stim_duration_s"]):
        t1 = t0 + float(sel["stim_duration_s"])
    else:
        t1 = _np.array([_np.nan if v is None else float(v) for v in t1]) + sh
    # display latency: the scan's photodiode median where measured (real event sets only), else the selector default
    st = (ScreenTiming & es).fetch(as_dict=True)
    if st and st[0]["has_diode"] and not _isnull(st[0]["latency_s"]):
        lat, src = float(st[0]["latency_s"]), "photodiode"
    else:
        lat, src = float(sel["display_latency_s"] or 0.0), "default"
    info = {"latency_s": lat, "latency_source": src,
            "ioi_s": float((EventSet & es).fetch1()["median_ioi_s"] or _np.nan)}
    return tk, X[:, :n], ft[:n], t0, t1, _np.asarray(cond).astype(str), info


def fill_offsets(t0, t1):
    """Missing offsets take the median duration; with none at all an event lasts IOI/2."""
    if _np.isfinite(t1).any():
        return _np.where(_np.isfinite(t1), t1, t0 + _np.nanmedian(t1 - t0))
    ioi = float(_np.median(_np.diff(t0))) if t0.size > 1 else 0.0
    return t0 + ioi / 2


def nearest_frame(ft, t):
    """Index of the 2P frame nearest each time, ties to the earlier frame (Leon export / v1-drift-audit)."""
    i = _np.clip(_np.searchsorted(ft, t), 1, len(ft) - 1)
    return _np.where(_np.abs(ft[i] - t) < _np.abs(ft[i - 1] - t), i, i - 1)


def windows(t0, t1, ps, ft, latency_s=0.0):
    """Resolve a ReadoutParamSet for every onset -> (r0, r1), (b0, b1) or None, all in session time [s].

    Reference times are the DataJoint onset / offset, plus ``latency_s`` when ps['time_ref'] == 'display'.
    Each edge = reference (onset or offset) + value. window_unit 'frames': the reference is the 2P frame nearest
    the reference time and the value counts frames; the returned times are frame times, so [r0, r1) selects exactly
    those frames. Windows that leave the recording come back NaN (the caller drops those onsets).
    """
    t0 = _np.asarray(t0, float); t1 = fill_offsets(t0, _np.asarray(t1, float))
    if ps["time_ref"] == "display":
        t0, t1 = t0 + latency_s, t1 + latency_s
    ft = _np.asarray(ft, float)
    if ps["window_unit"] == "frames":
        ext = _np.append(ft, ft[-1] + _np.median(_np.diff(ft)))        # an exclusive end may be one past the last frame
        ref = {"onset": nearest_frame(ft, t0), "offset": nearest_frame(ft, t1)}

        def edge(which, value):
            return ref[which] + int(round(value))

        def to_t(i):
            i = _np.asarray(i)
            return _np.where((i >= 0) & (i < ext.size), ext[_np.clip(i, 0, ext.size - 1)], _np.nan)
    else:
        ref = {"onset": t0, "offset": t1}

        def edge(which, value):
            return ref[which] + float(value)

        def to_t(x):
            return x
    s = edge(ps["resp_start_ref"], ps["resp_start"]); e = edge(ps["resp_stop_ref"], ps["resp_stop"])
    if ps["resp_length"] == "first_onset" and s.size:
        e = s + (e[0] - s[0])
    resp = (to_t(s), to_t(e))
    if _isnull(ps["base_stop"]) or ps["base_stop_ref"] is None or ps["measure"] == "peak_latency":
        return resp, None
    be = edge(ps["base_stop_ref"], ps["base_stop"])
    if ps["base_length"] == "match_response":
        bs = be - (e - s)
    else:
        bs = edge(ps["base_start_ref"], ps["base_start"])
    return resp, (to_t(bs), to_t(be))


def _key_source(task_part):
    """Trace sets that exist x event sets (with ScreenTiming done for real sets) x the variants a task asks for."""
    ready = [ScreenTiming.proj(), (EventSelector & "random_seed IS NOT NULL").proj()]     # OR-restriction
    return (available().proj() * EventSet.proj() * task_part.proj()) & task_part & ready


# ======================================================================================
# Readout
# ======================================================================================

@schema
class Readout(dj.Computed):
    definition = """
    # Per-onset read-out of every trace of one trace type, for one event set and one read-out variant
    -> imaging.Fluorescence
    -> TraceSource
    -> EventSet
    -> ReadoutParamSet
    ---
    n_onsets                : int                 # onsets whose windows lie inside the recording
    n_repeats               : smallint            # K: repeats per condition after balancing (0 = no reliability)
    latency_s               : float               # [s] display latency added to the event times (0 for time_ref 'event')
    latency_source          : enum('photodiode','default','none')   # where latency_s came from (ScreenTiming or EventSelector)
    resp_start_s            : float               # [s] response window start, median over onsets, relative to the DataJoint onset
    resp_stop_s             : float               # [s] response window end (exclusive), same reference
    base_start_s=null       : float               # [s] baseline window start, same reference (NULL: no baseline)
    base_stop_s=null        : float               # [s] baseline window end (exclusive), same reference
    software                : varchar(64)         # adamacs_ingest commit
    """

    class Cell(dj.Part):
        definition = """
        -> master
        -> imaging.Fluorescence.Trace
        ---
        trial_resp              : blob@external-raw   # float32 (EventSet.n_onsets,) corrected read-out per onset, EventSet.Onset order; NaN = outside
        trial_base=null         : blob@external-raw   # float32 (EventSet.n_onsets,) baseline-window value per onset, same measure (NULL: no baseline)
        resp=null               : float           # mean of trial_resp over all onsets
        effect_d=null           : float           # mean / SD of trial_resp
        resp_p=null             : float           # RESPONSIVE  ReadoutParamSet.test: response vs baseline (NULL: no baseline)
        cond_kw_p=null          : float           # SELECTIVE  Kruskal-Wallis of trial_resp across conditions (audit p_dir)
        pref_condition=null     : varchar(64)     # condition with the largest mean read-out (preferred orientation, image, RF position)
        pref_resp=null          : float           # mean read-out at pref_condition (chosen on the same onsets: biased up)
        pref_resp_cv=null       : float           # cross-validated: chosen on odd presentations, read on even, and reverse
        pref_p=null             : float           # Wilcoxon of the read-out at pref_condition vs 0 (not corrected for the choice)
        oracle_r=null           : float           # RELIABLE  leave-one-repeat-out correlation of the condition curve
        response_snr=null       : float           # RELIABLE  signal_var / noise_var across condition means (snr_bc)
        """

    class Condition(dj.Part):
        definition = """
        -> master.Cell
        condition               : varchar(64)
        ---
        n_onsets                : smallint
        resp=null               : float           # mean read-out over this condition
        resp_sem=null           : float
        """

    @property
    def key_source(self):
        return _key_source(ResponseTask.Readout)

    def make(self, key):
        ps = (ReadoutParamSet & key).fetch1()
        tk, X, ft, t0, t1, cond, info = _inputs(key)
        lat = info["latency_s"] if ps["time_ref"] == "display" else 0.0
        f = 1.0 / float(_np.median(_np.diff(ft)))
        (r0, r1), base = windows(t0, t1, ps, ft, info["latency_s"])
        dt = 1.0 / f
        with _np.errstate(invalid="ignore"):
            inside = (r0 >= ft[0]) & (r1 <= ft[-1] + dt) & (r1 > r0)
            if base is not None:
                inside &= (base[0] >= ft[0]) & (base[1] > base[0]) & (base[1] <= ft[-1] + dt)
        ii = _np.flatnonzero(inside)
        sm = int(round(ps["smooth_s"] * f))
        R = _stats.measure(X, ft, r0[ii], r1[ii], ps["measure"], sm)
        if base is None:
            Bm, C = _np.zeros_like(R), R
        else:
            # the baseline is read with the SAME measure; only the mean is comparable across window lengths
            if ps["measure"] not in LENGTH_FREE and ii.size and _np.nanmax(_np.abs((r1 - r0)[ii] - (base[1] - base[0])[ii])) > 1.5 / f:
                raise ValueError(f"measure {ps['measure']!r} needs equal response and baseline windows (base_length='match_response')")
            Bm = _stats.measure(X, ft, base[0][ii], base[1][ii], ps["measure"], sm)
            Bsd = _stats.measure(X, ft, base[0][ii], base[1][ii], "sd") if ps["correction"] == "zscore" else None
            C = _stats.correct(R, Bm, ps["correction"], Bsd)
        T = _np.full((X.shape[0], t0.size), _np.nan); T[:, ii] = C
        TB = None
        if base is not None:
            TB = _np.full((X.shape[0], t0.size), _np.nan); TB[:, ii] = Bm
        cin = cond[ii]
        levels = _np.unique(cin)
        multi = len(levels) > 1
        _, idx = _stats.balanced_repeats(cin, _np.arange(ii.size), min_k=2)
        K = int(idx.shape[1]) if idx.size and multi else 0

        cells, conds = [], []
        for i, k in enumerate(tk):
            c = C[i]
            row = {**key, **k, "trial_resp": T[i].astype(_np.float32),
                   "trial_base": TB[i].astype(_np.float32) if TB is not None else None,
                   "resp": _none(_np.nanmean(c)) if c.size else None,
                   "effect_d": _none(_stats.effect_d(c, _np.zeros_like(c))),
                   "resp_p": _none(_stats.paired_test(R[i], Bm[i], ps["test"], seed=_zt.seed_for(key["scan_id"], k["mask"])))
                             if base is not None else None,          # no baseline: a level, not a test
                   "cond_kw_p": _none(_stats.kruskal_p(c, cin)) if multi else None,
                   "pref_condition": None, "pref_resp": None, "pref_resp_cv": None, "pref_p": None,
                   "oracle_r": None, "response_snr": None}
            if multi:
                lab, m, p = _stats.preferred(c, cin)
                row.update(pref_condition=lab, pref_resp=_none(m), pref_p=_none(p),
                           pref_resp_cv=_none(_stats.preferred_cv(c, cin)))
            if K >= 2 and _np.all(_np.isfinite(c[idx])):
                s_, nv = _stats.signal_noise_var(c[idx])
                row.update(oracle_r=_none(_stats.loo_oracle(c[idx])), response_snr=_none(s_ / nv) if nv > 0 else None)
            cells.append(row)
            if multi:
                for lv in levels:
                    v = c[cin == lv]
                    conds.append({**key, **k, "condition": str(lv), "n_onsets": int(v.size), "resp": _none(_np.nanmean(v)),
                                  "resp_sem": _none(_np.nanstd(v, ddof=1) / _np.sqrt(v.size)) if v.size > 1 else None})
        rel = lambda x: float(_np.median(x[ii] - t0[ii])) if ii.size else _np.nan     # edges relative to the DataJoint onset
        self.insert1({**key, "n_onsets": int(ii.size), "n_repeats": K, "latency_s": lat,
                      "latency_source": info["latency_source"] if ps["time_ref"] == "display" else "none",
                      "resp_start_s": rel(r0), "resp_stop_s": rel(r1),
                      "base_start_s": _none(rel(base[0])) if base is not None else None,
                      "base_stop_s": _none(rel(base[1])) if base is not None else None,
                      "software": code_version()})
        self.Cell.insert(cells)
        self.Condition.insert(conds)


# ======================================================================================
# Zeta
# ======================================================================================

def trace_hashes(restriction, src):
    """{(paramset_idx, curation_id, mask, fluo_channel): content hash} of one trace source, without downloading blobs.
    Identical traces share an external-store hash, across curations too."""
    imaging = _up("imaging")
    if src["extraction_method"] is not None:
        q = (imaging.Activity.Trace & restriction & {"extraction_method": src["extraction_method"]}).proj(hx="HEX(activity_trace)")
    else:
        q = (imaging.Dff0.Trace & restriction & {"dff_paramset_idx": src["dff_paramset_idx"]}).proj(hx="HEX(dff)")
    p, c, m, ch, h = q.fetch("paramset_idx", "curation_id", "mask", "fluo_channel", "hx")
    return {(int(a), int(b), int(d), int(e)): x for a, b, d, e, x in zip(p, c, m, ch, h)}

@schema
class Zeta(dj.Computed):
    definition = """
    # Time-series ZETA of every trace of one trace type, for one event set and one ZETA variant
    -> imaging.Fluorescence
    -> TraceSource
    -> EventSet
    -> ZetaParamSet
    ---
    n_onsets                : int                 # onsets whose jitter window (-2w, +3w) lies inside the recording
    n_reused=0              : int                 # cells copied from another curation of this scan with a byte-identical trace
    window_s                : float               # [s] analysis window actually used
    latency_s               : float               # [s] display latency added to the onsets (0 for time_ref 'event')
    latency_source          : enum('photodiode','default','none')
    software                : varchar(64)         # zetapy version | adamacs_ingest commit
    """

    class Cell(dj.Part):
        definition = """
        -> master
        -> imaging.Fluorescence.Trace
        ---
        zeta_p=null             : float           # RESPONSIVE  time-locked to the onsets, any response shape
        zeta_z=null             : float           # zetapy zeta_score
        latency=null            : float           # [s] ZETA peak latency, relative to the (latency-corrected) onset
        condition_p=null        : float           # SPECIFIC  Holm-adjusted min of the per-condition p (per_condition only)
        source_curation_id=null : smallint        # NULL: computed here; else copied from this curation (identical trace, same events,
                                                  # same seed -> the same result a recomputation gives)
        """

    class Condition(dj.Part):
        definition = """
        -> master.Cell
        condition               : varchar(64)
        ---
        zeta_p=null             : float           # responds to THIS condition
        """

    N_PROCS = int(_os.environ.get("RESPONSIVENESS_ZETA_PROCS", "24"))

    @property
    def key_source(self):
        return _key_source(ResponseTask.Zeta)

    def make(self, key):
        from importlib.metadata import version
        ps = (ZetaParamSet & key).fetch1()
        tk, X, ft, t0, t1, cond, info = _inputs(key)
        if ps["mask_type"]:
            keep = set((_up("imaging").MaskClassification.MaskType & _fluo_key(key) & {"mask_type": ps["mask_type"]}).fetch("mask").tolist())
            sel = [i for i, k in enumerate(tk) if k["mask"] in keep]
            tk, X = [tk[i] for i in sel], X[sel]
        lat = info["latency_s"] if ps["time_ref"] == "display" else 0.0
        t = (t0 if ps["zeta_anchor"] == "onset" else fill_offsets(t0, t1)) + lat
        # window: the real events' median onset interval (t73: also for the random-onset null), unless set explicitly
        w = float(ps["zeta_window_s"]) if not _isnull(ps["zeta_window_s"]) else info["ioi_s"]
        inside = (t - 2 * w > ft[0]) & (t + 3 * w < ft[-1])
        t, cond = t[inside], cond[inside]
        zp = {"resampling": int(ps["zeta_resampling"]),
              "per_condition": bool(ps["per_condition"]) and len(_np.unique(cond)) > 1}
        # reuse: cells of other curations of this scan (same paramset) with a byte-identical trace and the same task
        src = (TraceSource & key).fetch1()
        mine = trace_hashes(_fluo_key(key), src)
        same = {kk: key[kk] for kk in ("session_id", "scan_id", "paramset_idx", "trace_source", "event_selector", "zeta_idx")}
        done = (Zeta.Cell & same & f"curation_id != {int(key['curation_id'])}" & "source_curation_id IS NULL")
        known = {}
        if done:
            sib = trace_hashes((dj.U("session_id", "scan_id", "paramset_idx", "curation_id") & done), src)
            rows_ = done.fetch(as_dict=True)
            conds_ = {}
            for c_ in (Zeta.Condition & done.proj()).fetch(as_dict=True):
                conds_.setdefault((c_["curation_id"], c_["mask"], c_["fluo_channel"]), {})[c_["condition"]] = c_["zeta_p"]
            for r_ in rows_:
                h = sib.get((r_["paramset_idx"], r_["curation_id"], r_["mask"], r_["fluo_channel"]))
                if h is not None and h not in known:
                    known[h] = (r_, conds_.get((r_["curation_id"], r_["mask"], r_["fluo_channel"]), {}))
        hk = [mine.get((key["paramset_idx"], key["curation_id"], k["mask"], k["fluo_channel"])) for k in tk]
        todo = [i for i, h in enumerate(hk) if h not in known]
        jobs = [(ft, X[i], t, cond, w, zp, _zt.seed_for(key["scan_id"], tk[i]["mask"])) for i in todo]
        procs = max(1, min(self.N_PROCS, (_os.cpu_count() or 2) - 8, len(jobs)))
        if procs > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(procs) as pool:
                out = pool.map(_zt.zeta_cell, jobs, chunksize=max(1, len(jobs) // (4 * procs)))
        else:
            out = [_zt.zeta_cell(j) for j in jobs]
        computed = dict(zip(todo, out))
        res, srcc = [], []
        for i, h in enumerate(hk):
            if i in computed:
                res.append(computed[i]); srcc.append(None)
            else:
                r_, pc = known[h]
                res.append(((r_["zeta_p"], r_["zeta_z"], r_["latency"]), pc)); srcc.append(int(r_["curation_id"]))
        self.insert1({**key, "n_onsets": int(t.size), "n_reused": int(sum(x is not None for x in srcc)), "window_s": float(w), "latency_s": lat,
                      "latency_source": info["latency_source"] if ps["time_ref"] == "display" else "none",
                      "software": f"zetapy {version('zetapy')} | {code_version()}"})
        self.Cell.insert({**key, **k, "zeta_p": _none(p), "zeta_z": _none(z), "latency": _none(lat_),
                          "condition_p": _none(_stats.holm_min(list(pc.values()))) if pc else None, "source_curation_id": sc}
                         for k, ((p, z, lat_), pc), sc in zip(tk, res, srcc))
        self.Condition.insert({**key, **k, "condition": c, "zeta_p": _none(v)}
                              for k, (_, pc) in zip(tk, res) for c, v in pc.items())


# ======================================================================================
# STIMULUS-SPECIFIC follow-ups (read from Readout; one table per stimulus family)
# ======================================================================================

@schema
class DirectionTuning(dj.Computed):
    definition = """
    # Orientation / direction tuning from one Readout of a direction stimulus (moving gratings, moving bars).
    # Mazurek, Kager & Van Hooser 2014 (doi:10.3389/fncir.2014.00092); classic indices for comparison.
    -> Readout
    -> TuningParamSet
    ---
    n_directions            : smallint
    n_repeats               : smallint            # repeats per direction after balancing (Hotelling / dot-product tests)
    software                : varchar(64)
    """

    class Cell(dj.Part):
        definition = """
        -> master
        -> Readout.Cell
        ---
        ori_kw_p=null           : float           # TUNED  Kruskal-Wallis across orientations, opposite directions pooled (audit p_tuned)
        dir_kw_p=null           : float           # Kruskal-Wallis across directions (audit p_dir)
        ori_vec=null            : float           # 1-CirVar = |sum R e^(2i theta)| / sum R   (recommended orientation selectivity)
        dir_vec=null            : float           # 1-DirCirVar = |sum R e^(i theta)| / sum R (recommended direction selectivity)
        pref_ori_vec=null       : float           # [deg, 0-180) angle of the orientation vector (TuningParamSet.angle_convention)
        pref_dir_vec=null       : float           # [deg, 0-360) angle of the direction vector
        hotelling_p=null        : float           # orientation tuning significant? Hotelling T2, per-repeat orientation vectors vs 0
        dir_dot_p=null          : float           # direction tuning significant? t-test of per-repeat direction vectors on the orientation axis
        pref_dir_dot=null       : float           # [deg] preferred direction from the dot-product sign (pref_ori_vec or +180)
        ori_vec_perm_p=null     : float           # label-shuffle p of ori_vec
        dir_vec_perm_p=null     : float           # label-shuffle p of dir_vec
        pref_ori_ci=null        : float           # [deg] width of the 95-percent bootstrap interval of pref_ori_vec
        r_pref=null             : float           # measured mean at the best direction
        r_null=null             : float           # ... at best + 180
        r_orth=null             : float           # ... mean of best +/- 90
        oi=null                 : float           # (Rp - Ro)/Rp      (biased at low selectivity; for comparison)
        di=null                 : float           # (Rp - Rn)/Rp
        osi=null                : float           # (Rp - Ro)/(Rp + Ro)  (Niell & Stryker 2008 form, measured curve)
        dsi=null                : float           # (Rp - Rn)/(Rp + Rn)
        fit_pref_dir=null       : float           # [deg] double-Gaussian preferred direction (report only if hotelling_p small)
        fit_sigma=null          : float           # [deg]
        fit_hwhh=null           : float           # [deg] half-width at half-height = sigma*sqrt(2 ln 2)
        fit_rp=null             : float           # response above offset at preferred
        fit_rn=null             : float           # response above offset at null
        fit_c=null              : float           # offset
        fit_r2=null             : float
        fit_osi=null            : float           # OSI from the fitted curve
        fit_dsi=null            : float           # DSI from the fitted curve
        """

    @property
    def key_source(self):
        return ((Readout.proj() * TuningParamSet.proj()) & ResponseTask.Tuning
                & (EventSelector & "circular_period = 360"))

    def make(self, key):
        tp = (TuningParamSet & key).fetch1()
        cond = (EventSet.Onset & key).fetch("condition", order_by="onset_idx").astype(str)
        rows = (Readout.Cell & key).fetch("KEY", "trial_resp")
        out, nrep = [], 0
        for k, tr in zip(*rows):
            r = _tuning.direction_tuning(_np.asarray(tr, float), cond, tp["rectify"], bool(tp["fit"]),
                                         int(tp["n_perm"]), int(tp["n_boot"]), seed=_zt.seed_for(key["scan_id"], k["mask"]),
                                         angle_convention=tp["angle_convention"])
            nrep = max(nrep, r["n_repeats"])
            out.append({**key, **k, **{c: _none(r.get(src)) for c, src in _DT_COLS.items()}})
        self.insert1({**key, "n_directions": len(set(cond)), "n_repeats": nrep, "software": code_version()})
        self.Cell.insert(out)


_DT_COLS = {c: c for c in ("ori_kw_p", "dir_kw_p", "ori_vec", "dir_vec", "pref_ori_vec", "pref_dir_vec", "hotelling_p", "dir_dot_p",
                           "pref_dir_dot", "ori_vec_perm_p", "dir_vec_perm_p", "pref_ori_ci", "r_pref", "r_null",
                           "r_orth", "oi", "di", "osi", "dsi", "fit_pref_dir", "fit_sigma", "fit_hwhh", "fit_rp",
                           "fit_rn", "fit_c", "fit_r2")}
_DT_COLS.update(fit_osi="fit_osi_fit", fit_dsi="fit_dsi_fit")


# ======================================================================================
# Thresholds
# ======================================================================================

@schema
class CellRule(dj.Lookup):
    definition = """
    # Named thresholds, bound to the variants they apply to. NULL threshold = criterion not used. Never edited.
    cell_rule               : varchar(32)
    ---
    -> [nullable] TraceSource                     # trace type the ACTIVE / RESPONSIVE / RELIABLE criteria refer to (NULL: every one)
    -> [nullable] ReadoutParamSet                 # read-out for resp_p / oracle_r
    -> [nullable] ZetaParamSet                    # ZETA variant for zeta_p / condition_p
    mask_type=''            : varchar(16)         # QC          imaging.MaskClassification.MaskType (e.g. 'soma'); '' = any
    max_trace_value=null    : float               # QC          TraceQuality.Cell.max_value (audit: 50 for CASCADE)
    min_kurtosis=null       : float               # QC          Quality.Cell.kurtosis
    min_photons_per_s=null  : float               # QC          Quality.Cell.photons_per_s
    min_snr_s2p=null        : float               # ACTIVE      TraceQuality.Cell.snr_s2p
    max_zeta_p=null         : float               # RESPONSIVE  Zeta.Cell.zeta_p
    max_resp_p=null         : float               # RESPONSIVE  Readout.Cell.resp_p
    max_condition_p=null    : float               # SPECIFIC    Zeta.Cell.condition_p
    min_oracle_r=null       : float               # RELIABLE    Readout.Cell.oracle_r
    rule_note               : varchar(255)
    """
    contents = [{"cell_rule": "zeta05_dff0", "trace_source": "dff0_r2", "readout_idx": None, "zeta_idx": 2,
                 "max_zeta_p": 0.05, "max_condition_p": 0.05,
                 "rule_note": "dF/F0 recipe 2: ZETA p < 0.05 pooled; Holm per condition (ZETA variant 2); no QC cut"}]


def _flags(r, *crit):
    parts = [f"({col} {op} {r[name]})" for name, col, op in crit if r[name] is not None]
    return "(" + " AND ".join(parts) + ")" if parts else "NULL"


def cells(rule, restriction=None):
    """One CellRule applied on both levels, per trace x trace source x event set: is_qc, is_active,
    is_responsive, is_specific, is_reliable (NULL where the rule sets no threshold). Joined on the
    inherited keys to the rule's own variants. A query, nothing stored."""
    r = (CellRule & {"cell_rule": rule}).fetch1()
    src = {"trace_source": r["trace_source"]} if r["trace_source"] is not None else {}
    q = Quality.Cell.proj("kurtosis", "photons_per_s") * (TraceQuality.Cell & src).proj("snr_s2p", "max_value")
    if r["mask_type"]:
        q = q & (_up("imaging").MaskClassification.MaskType & {"mask_type": r["mask_type"]})
    resp, spec, rel = [], [], []
    if r["zeta_idx"] is not None:
        q = q * (Zeta.Cell & src & {"zeta_idx": r["zeta_idx"]}).proj("zeta_p", "condition_p")
        resp.append(("max_zeta_p", "zeta_p", "<")); spec.append(("max_condition_p", "condition_p", "<"))
    if r["readout_idx"] is not None:
        q = q * (Readout.Cell & src & {"readout_idx": r["readout_idx"]}).proj("resp_p", "oracle_r")
        resp.append(("max_resp_p", "resp_p", "<")); rel.append(("min_oracle_r", "oracle_r", ">="))
    q = q & restriction if restriction is not None else q
    return q.proj(is_qc=_flags(r, ("min_kurtosis", "kurtosis", ">="), ("min_photons_per_s", "photons_per_s", ">="),
                               ("max_trace_value", "max_value", "<=")),
                  is_active=_flags(r, ("min_snr_s2p", "snr_s2p", ">=")),
                  is_responsive=_flags(r, *resp), is_specific=_flags(r, *spec), is_reliable=_flags(r, *rel))


# ======================================================================================
# Derivable helpers (no tables)
# ======================================================================================

def legacy_cascade_recipe(activity_key):
    """dF/F0 recipe of a CASCADE activity written before imaging.Activity.CascadeInput existed.

    Dated by the external-store timestamp of its traces: before 2026-09-03 15:22 -> recipe 3 (dark frames
    12-24), after -> recipe 2 (shutter window). Returns (dff_paramset_idx, written_at). Replaces the
    pre-v6 CascadeInput table: the answer is derivable, so it is not stored.
    """
    import datetime as _dt
    from element_calcium_imaging import dff as _dff
    imaging = _up("imaging")
    ext = dj.Schema(imaging.schema.database, create_schema=False, create_tables=False).external["external-raw"]
    q = (imaging.Activity.Trace & activity_key).proj("activity_trace") * ext.proj(activity_trace="hash", w="timestamp")
    written = dj.U().aggr(q, first="min(w)").fetch1("first")
    before, since = _dff.CASCADE_METHOD_RECIPES[activity_key["extraction_method"]]
    idx = before if written < _dt.datetime.fromisoformat(_dff.CASCADE_CODE_CHANGE) else since
    return idx, written


# ======================================================================================

def activate(schema_name: str, *, imaging, scan, event, trial, create_schema=True, create_tables=True):
    """Activate. ``imaging`` must hold DffParamSet / Dff0 (element-calcium-imaging >= 0.1.0b1)."""
    for name in ("DffParamSet", "Dff0", "_shutter_inputs"):
        if not hasattr(imaging, name):
            raise dj.DataJointError(f"imaging.{name} missing: element-calcium-imaging < 0.1.0b1")
    _UP.update(imaging=imaging, scan=scan, event=event, trial=trial)
    schema.activate(schema_name, create_schema=create_schema, create_tables=create_tables,
                    add_objects={"imaging": imaging, "scan": scan, "event": event, "trial": trial})
    return schema
