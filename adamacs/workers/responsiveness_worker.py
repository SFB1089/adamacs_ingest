"""Worker for the responsiveness schema (adamacs/schemas/responsiveness.py).

Where: ibehaveGPU1 (``ssh BUGPU``), conda env ``datajoint_ingest`` (zetapy, scanreader, sklearn).
ZETA uses a process pool (RESPONSIVENESS_ZETA_PROCS, default 24); everything else is light.

Order of one pass (each step only touches keys that ResponseTask asks for):
    1. imaging.Dff0           dF/F0 for every Dff0Task a dF/F0 ResponseTask needs (any task_note)
    2. EventSet               onsets from trial.TrialEvent
    2b. ScreenTiming          photodiode (LED) onsets from the aux .h5 where the rig has a diode
    3. ScanQuality            raw-TIFF detector gain per scan (curation-independent; reads only the needed pages)
    3b. Quality               neuropil sweep and per-cell raw statistics per curation
    4. TraceQuality           per trace type
    5. Readout                PSTH read-outs (cheap)
    6. Zeta                   ZETA variants (expensive)
    7. DirectionTuning        stimulus-specific follow-ups on Readout

Usage:
    python -m adamacs.workers.responsiveness_worker --schema tobiasr_responsiveness_v7 --once
    python -m adamacs.workers.responsiveness_worker --schema tobiasr_responsiveness_v7 --poll 600

The datajoint_ingest env's OpenSSL fails the handshake with the server's TLS: pass --no-tls (LAN only).
Populate errors are recorded in the schema's jobs table (suppress_errors); nothing is deleted here.
"""
from __future__ import annotations

import argparse
import logging
import os
import time

# CPU-only worker (numpy, scipy, zetapy): hide every GPU so no library can allocate one (BUGPU: GPU 0 is never ours)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

log = logging.getLogger("responsiveness_worker")


def connect(conf_path: str | None, no_tls: bool):
    import datajoint as dj
    if conf_path:
        dj.config.load(conf_path)
    if no_tls:
        dj.config["database.use_tls"] = False
    dj.conn()
    return dj


def activate(schema_name: str):
    from adamacs.pipeline import event, imaging, scan, trial
    from adamacs.schemas import responsiveness as resp
    resp.activate(schema_name, imaging=imaging, scan=scan, event=event, trial=trial)
    return resp, imaging


def one_pass(resp, imaging, restriction=None, reserve_jobs=True):
    """Populate every step once; returns {table: (done_before, done_after)}.
    reserve_jobs=False: single-process runs (pilots) that must not depend on, or write to, the jobs table."""
    r = restriction if restriction is not None else {}
    # every Dff0Task a dF/F0 ResponseTask needs, whoever staged it (reserve_jobs avoids clashes with other workers)
    need = (resp.ResponseTask * resp.TraceSource & "dff_paramset_idx IS NOT NULL" & r).proj("dff_paramset_idx")
    dff_tasks = imaging.Dff0Task & need
    steps = [("imaging.Dff0", imaging.Dff0, dff_tasks),
             ("EventSet", resp.EventSet, r), ("ScreenTiming", resp.ScreenTiming, r), ("ScanQuality", resp.ScanQuality, r), ("Quality", resp.Quality, r), ("TraceQuality", resp.TraceQuality, r),
             ("Readout", resp.Readout, r), ("Zeta", resp.Zeta, r), ("DirectionTuning", resp.DirectionTuning, r)]
    out = {}
    if not reserve_jobs:
        steps = steps[1:]           # dF/F0 is shared with other workers: only compute it with job reservation
    for name, tbl, restr in steps:
        n0 = len(tbl & restr) if name != "imaging.Dff0" else len(imaging.Dff0 & dff_tasks)
        t0 = time.time()
        errs = tbl.populate(restr, reserve_jobs=reserve_jobs, suppress_errors=True, display_progress=False)
        for e in (errs or {}).get("error_list", []) if isinstance(errs, dict) else (errs or []):
            log.error("%s %s", name, str(e)[:500])
        n1 = len(tbl & restr) if name != "imaging.Dff0" else len(imaging.Dff0 & dff_tasks)
        log.info("%-16s %5d -> %5d  (%.0f s)", name, n0, n1, time.time() - t0)
        out[name] = (n0, n1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--schema", required=True, help="e.g. tobiasr_responsiveness_v7 (test) / roselab_responsiveness")
    ap.add_argument("--conf", default=os.path.expanduser("~/adamacs_ingest/dj_local_conf.json"))
    ap.add_argument("--no-tls", action="store_true")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--poll", type=int, default=600, help="seconds between passes")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    connect(a.conf, a.no_tls)
    resp, imaging = activate(a.schema)
    while True:
        one_pass(resp, imaging)
        if a.once:
            break
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
