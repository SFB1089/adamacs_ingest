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


def traces(key):
    """(trace_keys, X[n_traces, n_frames]) of the trace source in ``key``; ordered by mask, fluo_channel.
    trace_keys are imaging.Fluorescence.Trace keys."""
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
class Quality(dj.Computed):
    definition = """
    # Data fidelity of one curation from raw fluorescence (Labrigger 2026 catechism; Lees et al. 2025 photon transfer)
    -> imaging.Fluorescence
    ---
    gain_method             : enum('dark','ptc')  # dark: offset + read noise from shutter-closed frames; ptc: free-intercept fit (no dark frames)
    gain=null               : double              # [grey levels / detected photon] (V - dark_var) = gain*(I - offset), frame-pair variance
    offset=null             : double              # [grey levels] zero-photon level (mean dark frame, or fitted intercept)
    gain_r2=null            : float               # fit quality over the lower half of the intensity range
    n_frames                : int                 # raw frames used for the fit (dark frames come in addition)
    alpha_opt=null          : float               # neuropil factor that brings the mean pairwise cell correlation closest to 0
    mean_r_at_alpha07=null  : float               # mean pairwise correlation at the pipeline's 0.7
    """

    class Cell(dj.Part):
        definition = """
        -> master
        -> imaging.Fluorescence.Trace
        ---
        kurtosis=null           : float           # Pearson kurtosis of raw F (normal 3); low = noise or neuropil-dominated
        neuropil_r=null         : float           # r(F, Fneu); high = contaminated
        photons_per_s=null      : float           # enf^2*(F0 - offset)*npix/gain*fps, F0 = 30th pct of raw F (typical 1e3-1e4)
        """

    N_FRAMES = 500       # bright frames for the transfer curve
    ENF = 1.2            # PMT excess-noise factor (kurtosis-gain-checker convention)
    SWEEP_BIN_S = 1.0    # neuropil is slow: correlate 1 s bins

    @property
    def key_source(self):
        return _up("imaging").Fluorescence & ResponseTask

    def make(self, key):
        from element_calcium_imaging import dff as _dff
        import tifffile
        imaging, scan = _up("imaging"), _up("scan")
        sk = _scan_key(key)
        info = (scan.ScanInfo & sk).fetch1()
        files = (scan.ScanInfo.ScanFile & sk).fetch("file_path", order_by="file_path")
        # Read only the frames needed, page by page (channel 0, depth 0, field 0). scanreader would load the
        # whole field first: 214 GB for a 147,300-frame scan in the 2026-11-15 pilot.
        step = int(info["nchannels"] or 1) * int(info["ndepths"] or 1)
        tf = tifffile.TiffFile(files[0])

        def frames(f0, f1):
            return _np.stack([tf.pages[f * step].asarray() for f in range(f0, f1)]).astype(_np.float64)
        ft, starts, ends = imaging._shutter_inputs(sk)
        dark = None
        if ft is not None:
            try:
                a, b = _dff.shutter_dark_frames(ft, starts, ends, 0.09)
                dark = (a, b) if b - a >= 3 else None
            except ValueError:
                dark = None
        b0 = (dark[1] if dark else 0) + 10
        n = int(min(self.N_FRAMES, int(info["nframes"]) - b0))
        bright = frames(b0, b0 + n)
        I, V, C = _photon.ptc_bins(bright)
        g = _np.nan
        if dark:
            off, dvar = _photon.dark_stats(frames(dark[0], dark[1]))
            g, r2 = _photon.fit_dark_referenced(I, V, C, off, dvar)
            method = "dark"
        if not (_np.isfinite(g) and g > 0):
            g, off, r2 = _photon.fit_ptc(I, V, C)
            method = "ptc"
        tf.close()

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

        self.insert1({**key, "gain_method": method, "gain": _none(g), "offset": _none(off), "gain_r2": _none(r2),
                      "n_frames": n, "alpha_opt": a_opt, "mean_r_at_alpha07": r07})
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


# ======================================================================================
# UNIVERSAL, event-dependent: parameter sets
# ======================================================================================

_RCOLS = ("window_unit", "resp_start_ref", "resp_start", "resp_stop_ref", "resp_stop", "length_rule", "base_mode",
          "base_ref", "base_start", "base_stop", "measure", "smooth_s", "correction", "test")
_READOUTS = {
    1: ("[onset+0.1 s, offset) vs pre-onset gap (<= 3 s); mean; subtract",
        "s", "onset", 0.1, "offset", 0.0, "per_onset", "gap", "onset", -3.0, 0.0, "mean", 0.0, "subtract", "wilcoxon"),
    2: ("Leon export, gratings (reproduced 234/234 in v1-drift-audit): frames [on, on+n), n = off-on+2 of the first trial; "
        "control = same length ending 1 frame before onset; raw stimulus mean (no correction)",
        "frames", "onset", 0, "offset", 2, "first_onset", "matched", "onset", 0, -1, "mean", 0.0, "none", "wilcoxon"),
    3: ("Leon export, natural images (reproduced 84/84 in v1-drift-audit): frames [on+3, on+3+n), n = off-on+2 of the first trial; "
        "no baseline; raw mean",
        "frames", "onset", 3, "offset", 5, "first_onset", "none", "onset", None, 0, "mean", 0.0, "none", "wilcoxon"),
    4: ("peak of 200 ms boxcar in [onset, offset), z-scored to pre-onset gap noise",
        "s", "onset", 0.0, "offset", 0.0, "per_onset", "gap", "onset", -3.0, 0.0, "peak", 0.2, "zscore", "wilcoxon"),
    5: ("OFF: [offset+0.1, offset+1.0) vs [offset-0.5, offset); mean; subtract",
        "s", "offset", 0.1, "offset", 1.0, "per_onset", "fixed", "offset", -0.5, 0.0, "mean", 0.0, "subtract", "wilcoxon"),
    6: ("[onset, offset) vs [onset-0.5, onset); mean; subtract",
        "s", "onset", 0.0, "offset", 0.0, "per_onset", "fixed", "onset", -0.5, 0.0, "mean", 0.0, "subtract", "wilcoxon"),
    7: ("Leon export, CASCADE 'opt' trees (v1-drift-audit recompute, opt_json): frames [on+1, on+15); no baseline; raw mean",
        "frames", "onset", 1, "onset", 15, "per_onset", "none", "onset", None, 0, "mean", 0.0, "none", "wilcoxon"),
}


@schema
class ReadoutParamSet(dj.Lookup):
    definition = """
    # Peri-event (PSTH) read-out, independent of trace and paradigm. Every choice is a column.
    # Each window edge = reference edge of the stimulus + offset. Never edited; same content -> same hash.
    readout_idx             : smallint
    ---
    readout_desc            : varchar(255)
    readout_hash            : uuid
    unique index (readout_hash)
    window_unit='s'         : enum('s','frames')            # s: edges in seconds from the event times;
                                                            # frames: edges in 2P frames from the frame NEAREST each event time
                                                            # (the convention of the Leon export, reproduced by the v1-drift-audit)
    resp_start_ref='onset'  : enum('onset','offset')        # response window start: stimulus start or end ...
    resp_start              : float                         # ... + this (s or frames; e.g. 0.1 s, 3 frames)
    resp_stop_ref='offset'  : enum('onset','offset')        # response window end: stimulus start or end ...
    resp_stop=0             : float                         # ... + this (e.g. 0 = at stimulus end, -0.2 s = 200 ms before)
    length_rule='per_onset' : enum('per_onset','first_onset')   # first_onset: every window gets the first onset's length
                                                            # (Leon export; session-specific length - avoid for drift)
    base_mode='fixed'       : enum('fixed','gap','matched','none')
                                                            # fixed: [base_ref+base_start, base_ref+base_stop)
                                                            # gap: inter-stimulus gap before onset, at most |base_start| (s)
                                                            # matched: as long as the response window, ending at base_ref+base_stop
                                                            # none: no baseline (test against 0, e.g. spike rates)
    base_ref='onset'        : enum('onset','offset')
    base_start=null         : float                         # e.g. -0.5 s
    base_stop=0             : float                         # e.g. 0 s, or -1 frame
    measure='mean'          : enum('mean','median','integral','peak','trough','peak_abs','auc_pos','peak_latency')
    smooth_s=0              : float                         # [s] boxcar before peak / trough / peak_abs / peak_latency
    correction='subtract'   : enum('subtract','divide','zscore','none')   # r-b | (r-b)/|b| | (r-b)/baseline frame SD | r
    test='wilcoxon'         : enum('wilcoxon','ttest','signflip')         # paired response vs baseline (vs 0 if base 'none')
    """
    contents = [{"readout_idx": i, "readout_desc": v[0], "readout_hash": _hash(dict(zip(_RCOLS, v[1:]))),
                 **dict(zip(_RCOLS, v[1:]))} for i, v in _READOUTS.items()]


_ZCOLS = ("zeta_anchor", "zeta_window_s", "zeta_resampling", "per_condition")
_ZETAS = {
    1: ("audit t73/t140: onset, window = median onset-to-onset interval, 100 resamples, pooled", "onset", None, 100, 0),
    2: ("as 1 + one-sample ZETA per condition (cost x n_conditions)", "onset", None, 100, 1),
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
    zeta_anchor='onset'     : enum('onset','offset')
    zeta_window_s=null      : float                         # [s] analysis window after the anchor; NULL = median onset-to-onset interval
    zeta_resampling=100     : smallint
    per_condition=0         : bool                          # also one-sample ZETA per condition
    """
    contents = [{"zeta_idx": i, "zeta_desc": v[0], "zeta_hash": _hash(dict(zip(_ZCOLS, v[1:]))),
                 **dict(zip(_ZCOLS, v[1:]))} for i, v in _ZETAS.items()]


_TCOLS = ("rectify", "fit", "n_perm", "n_boot")
_TUNINGS = {
    1: ("Mazurek et al. 2014 measures on condition means clipped at 0; double-Gaussian fit; 1000 label shuffles; 200 bootstraps",
        "clip", 1, 1000, 200),
    2: ("as 1 without rectification (negative condition means kept; vector lengths can exceed 1)", "none", 1, 1000, 200),
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
    """(trace_keys, X, frame_times, onset, offset, condition), traces cut to the frame clock."""
    tk, X = traces(key)
    ft = frame_times(key)
    n = min(ft.size, X.shape[1])
    es = {k: key[k] for k in EventSet.primary_key}
    q = EventSet.Onset * _up("event").Event.proj("event_end_time") & es
    t0, t1, sh, cond = q.fetch("event_start_time", "event_end_time", "shift_s", "condition", order_by="onset_idx")
    sh = _np.asarray(sh, float)
    t0 = _np.asarray(t0, float) + sh
    dur = (EventSelector & es).fetch1("stim_duration_s")
    t1 = t0 + float(dur) if dur is not None else _np.array([_np.nan if v is None else float(v) for v in t1]) + sh
    return tk, X[:, :n], ft[:n], t0, t1, _np.asarray(cond).astype(str)


def fill_offsets(t0, t1):
    """Missing offsets take the median duration; with none at all an event lasts IOI/2."""
    if _np.isfinite(t1).any():
        return _np.where(_np.isfinite(t1), t1, t0 + _np.nanmedian(t1 - t0))
    ioi = float(_np.median(_np.diff(t0))) if t0.size > 1 else 0.0
    return t0 + ioi / 2


def nearest_frame(ft, t):
    """Index of the 2P frame nearest each time (Leon export / v1-drift-audit convention)."""
    i = _np.clip(_np.searchsorted(ft, t), 1, len(ft) - 1)
    return _np.where(_np.abs(ft[i] - t) < _np.abs(ft[i - 1] - t), i, i - 1)


def windows(t0, t1, ps, ft):
    """Per onset: response window [r0, r1) and baseline window [b0, b1) in session time (None if no baseline).

    window_unit 's': edge = (onset or offset) + seconds. 'frames': edge = frame nearest (onset or
    offset) + frames, returned as frame times so that [r0, r1) selects exactly those frames.
    Windows that leave the recording come back NaN. 'gap' baselines end at onset and last
    min(|base_start|, median offset-to-next-onset gap) seconds, so they never reach the previous event.
    """
    t0 = _np.asarray(t0, float); t1 = fill_offsets(t0, _np.asarray(t1, float))
    ft = _np.asarray(ft, float)
    if ps["window_unit"] == "frames":
        ext = _np.append(ft, ft[-1] + _np.median(_np.diff(ft)))          # exclusive end may be one past the last frame
        ref = {"onset": nearest_frame(ft, t0), "offset": nearest_frame(ft, t1)}

        def edge(name, off):
            return ref[name] + int(round(off))

        def to_t(i):
            i = _np.asarray(i)
            return _np.where((i >= 0) & (i < ext.size), ext[_np.clip(i, 0, ext.size - 1)], _np.nan)
    else:
        ref = {"onset": t0, "offset": t1}

        def edge(name, off):
            return ref[name] + off

        def to_t(x):
            return x
    s = edge(ps["resp_start_ref"], ps["resp_start"]); e = edge(ps["resp_stop_ref"], ps["resp_stop"])
    if ps["length_rule"] == "first_onset" and s.size:
        e = s + (e[0] - s[0])
    resp = (to_t(s), to_t(e))
    if ps["base_mode"] == "none" or ps["measure"] == "peak_latency":
        return resp, None
    if ps["base_mode"] == "gap":
        ioi = float(_np.median(_np.diff(t0))) if t0.size > 1 else _np.inf
        gap = float(_np.median(t0[1:] - t1[:-1])) if t0.size > 1 else _np.inf
        b = min(abs(ps["base_start"]), gap if gap > 0 else ioi / 2)
        return resp, (t0 - b, t0)
    if ps["base_mode"] == "matched":
        b1 = edge(ps["base_ref"], ps["base_stop"])
        return resp, (to_t(b1 - (e - s)), to_t(b1))
    return resp, (to_t(edge(ps["base_ref"], ps["base_start"])), to_t(edge(ps["base_ref"], ps["base_stop"])))


def _key_source(task_part):
    """Trace sets that exist x event sets x the variants a task asks for."""
    return (available().proj() * EventSet.proj() * task_part.proj()) & task_part


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
    resp_s                  : float               # [s] median response-window length actually used
    base_s=null             : float               # [s] median baseline length actually used (NULL: no baseline)
    software                : varchar(64)         # adamacs_ingest commit
    """

    class Cell(dj.Part):
        definition = """
        -> master
        -> imaging.Fluorescence.Trace
        ---
        trial_resp              : blob@external-raw   # float32 (EventSet.n_onsets,) corrected read-out per onset, EventSet.Onset order; NaN = outside
        resp=null               : float           # mean of trial_resp over all onsets
        effect_d=null           : float           # mean / SD of trial_resp
        resp_p=null             : float           # RESPONSIVE  ReadoutParamSet.test: response vs baseline (or vs 0)
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
        tk, X, ft, t0, t1, cond = _inputs(key)
        f = 1.0 / float(_np.median(_np.diff(ft)))
        (r0, r1), base = windows(t0, t1, ps, ft)
        dt = float(_np.median(_np.diff(ft)))
        with _np.errstate(invalid="ignore"):
            inside = (r0 >= ft[0]) & (r1 <= ft[-1] + dt) & (r1 > r0)
            if base is not None:
                inside &= (base[0] >= ft[0]) & (base[1] > base[0])
        ii = _np.flatnonzero(inside)
        R = _stats.measure(X, ft, r0[ii], r1[ii], ps["measure"], int(round(ps["smooth_s"] * f)))
        if base is None:
            Bm, C = _np.zeros_like(R), R
        else:
            Bm = _stats.measure(X, ft, base[0][ii], base[1][ii], "mean")
            Bsd = _stats.measure(X, ft, base[0][ii], base[1][ii], "sd") if ps["correction"] == "zscore" else None
            C = _stats.correct(R, Bm, ps["correction"], Bsd)
        T = _np.full((X.shape[0], t0.size), _np.nan); T[:, ii] = C
        cin = cond[ii]
        levels = _np.unique(cin)
        multi = len(levels) > 1
        _, idx = _stats.balanced_repeats(cin, _np.arange(ii.size), min_k=2)
        K = int(idx.shape[1]) if idx.size and multi else 0

        cells, conds = [], []
        for i, k in enumerate(tk):
            c = C[i]
            row = {**key, **k, "trial_resp": T[i].astype(_np.float32),
                   "resp": _none(_np.nanmean(c)) if c.size else None,
                   "effect_d": _none(_stats.effect_d(c, _np.zeros_like(c))),
                   "resp_p": _none(_stats.paired_test(R[i], Bm[i], ps["test"],
                                                      seed=_zt.seed_for(key["scan_id"], k["mask"]))),
                   "pref_condition": None, "pref_resp": None, "pref_resp_cv": None, "pref_p": None,
                   "oracle_r": None, "response_snr": None}
            if multi:
                lab, m, p = _stats.preferred(c, cin)
                row.update(pref_condition=lab, pref_resp=_none(m), pref_p=_none(p),
                           pref_resp_cv=_none(_stats.preferred_cv(c, cin)))
            if K >= 2 and _np.all(_np.isfinite(c[idx])):
                s, nv = _stats.signal_noise_var(c[idx])
                row.update(oracle_r=_none(_stats.loo_oracle(c[idx])), response_snr=_none(s / nv) if nv > 0 else None)
            cells.append(row)
            if multi:
                for lv in levels:
                    v = c[cin == lv]
                    conds.append({**key, **k, "condition": str(lv), "n_onsets": int(v.size), "resp": _none(_np.nanmean(v)),
                                  "resp_sem": _none(_np.nanstd(v, ddof=1) / _np.sqrt(v.size)) if v.size > 1 else None})
        self.insert1({**key, "n_onsets": int(ii.size), "n_repeats": K,
                      "resp_s": float(_np.median(r1[ii] - r0[ii])) if ii.size else 0.0,
                      "base_s": _none(_np.median(base[1][ii] - base[0][ii])) if base is not None and ii.size else None,
                      "software": code_version()})
        self.Cell.insert(cells)
        self.Condition.insert(conds)


# ======================================================================================
# Zeta
# ======================================================================================

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
    window_s                : float               # [s] analysis window actually used
    software                : varchar(64)         # zetapy version | adamacs_ingest commit
    """

    class Cell(dj.Part):
        definition = """
        -> master
        -> imaging.Fluorescence.Trace
        ---
        zeta_p=null             : float           # RESPONSIVE  time-locked to the onsets, any response shape
        zeta_z=null             : float           # zetapy zeta_score
        latency=null            : float           # [s] peak latency
        condition_p=null        : float           # SPECIFIC  Holm-adjusted min of the per-condition p (per_condition only)
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
        tk, X, ft, t0, t1, cond = _inputs(key)
        t = t0 if ps["zeta_anchor"] == "onset" else fill_offsets(t0, t1)
        w, inside = _zt.window_and_onsets(t, ft)
        if ps["zeta_window_s"] is not None:
            w = float(ps["zeta_window_s"])
            inside = (t - 2 * w > ft[0]) & (t + 3 * w < ft[-1])
        t, cond = t[inside], cond[inside]
        zp = {"resampling": int(ps["zeta_resampling"]),
              "per_condition": bool(ps["per_condition"]) and len(_np.unique(cond)) > 1}
        jobs = [(ft, X[i], t, cond, w, zp, _zt.seed_for(key["scan_id"], k["mask"])) for i, k in enumerate(tk)]
        procs = max(1, min(self.N_PROCS, (_os.cpu_count() or 2) - 8, len(jobs)))
        if procs > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(procs) as pool:
                res = pool.map(_zt.zeta_cell, jobs, chunksize=max(1, len(jobs) // (4 * procs)))
        else:
            res = [_zt.zeta_cell(j) for j in jobs]
        self.insert1({**key, "n_onsets": int(t.size), "window_s": float(w),
                      "software": f"zetapy {version('zetapy')} | {code_version()}"})
        self.Cell.insert({**key, **k, "zeta_p": _none(p), "zeta_z": _none(z), "latency": _none(lat),
                          "condition_p": _none(_stats.holm_min(list(pc.values()))) if pc else None}
                         for k, ((p, z, lat), pc) in zip(tk, res))
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
        ori_vec=null            : float           # 1-CirVar = |sum R e^(2i theta)| / sum R   (recommended orientation selectivity)
        dir_vec=null            : float           # 1-DirCirVar = |sum R e^(i theta)| / sum R (recommended direction selectivity)
        pref_ori_vec=null       : float           # [deg, 0-180) angle of the orientation vector
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
                                         int(tp["n_perm"]), int(tp["n_boot"]), seed=_zt.seed_for(key["scan_id"], k["mask"]))
            nrep = max(nrep, r["n_repeats"])
            out.append({**key, **k, **{c: _none(r.get(src)) for c, src in _DT_COLS.items()}})
        self.insert1({**key, "n_directions": len(set(cond)), "n_repeats": nrep, "software": code_version()})
        self.Cell.insert(out)


_DT_COLS = {c: c for c in ("ori_vec", "dir_vec", "pref_ori_vec", "pref_dir_vec", "hotelling_p", "dir_dot_p",
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

def activate(schema_name: str, *, imaging, scan, event, trial, create_schema=True, create_tables=True):
    """Activate. ``imaging`` must hold DffParamSet / Dff0 (element-calcium-imaging >= 0.1.0b1)."""
    for name in ("DffParamSet", "Dff0", "_shutter_inputs"):
        if not hasattr(imaging, name):
            raise dj.DataJointError(f"imaging.{name} missing: element-calcium-imaging < 0.1.0b1")
    _UP.update(imaging=imaging, scan=scan, event=event, trial=trial)
    schema.activate(schema_name, create_schema=create_schema, create_tables=create_tables,
                    add_objects={"imaging": imaging, "scan": scan, "event": event, "trial": trial})
    return schema
