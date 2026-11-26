"""Worker for the facemap schema (adamacs/schemas/facemap.py).

Where: TATCHU3 (the facemap files and the blob store are local disks there), conda env ``datajoint_analysis``
(numpy 2: Leon's proc files are numpy-2 pickles), PYTHONPATH = the adamacs_ingest checkout. CPU only.
Run it from a directory that holds dj_local_conf.json (as every lab worker): adamacs reads the schema prefix
('roselab_') from it at import, before --conf is loaded.
Memory: a site run file is ~11 GB in RAM, each per-scan split file ~5 GB, one at a time.

Usage:
    python -m adamacs.workers.facemap_worker --schema roselab_facemap --no-tls plan            # dry run of both plans
    python -m adamacs.workers.facemap_worker --schema roselab_facemap --no-tls plan --insert   # insert the tasks
    python -m adamacs.workers.facemap_worker --schema roselab_facemap --no-tls populate --once [--task <facemap_task_id>]

Populate order: FacemapProcessing -> FacialSignal -> FrameSync -> SegmentDistance, with reserve_jobs, so several
workers can run side by side. Errors land in the schema's jobs table (suppress_errors); nothing is deleted here.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

log = logging.getLogger("facemap_worker")


def connect(conf_path: str, no_tls: bool = False):
    import datajoint as dj
    dj.config.load(conf_path)
    if no_tls:       # datajoint_analysis's OpenSSL fails the server's TLS handshake (LAN only, as responsiveness_worker)
        dj.config["database.use_tls"] = False
    dj.conn()
    return dj


def activate(schema_name: str):
    from adamacs.schemas import facemap
    facemap.activate(schema_name, linking_module="adamacs.pipeline")
    return facemap


def plan(fm, insert: bool):
    out = {"site": fm.plan_leon_site_runs(insert=insert), "per_video": fm.plan_leon_per_video_runs(insert=insert)}
    for kind, rows in out.items():
        status = {}
        for r in rows:
            status[r["status"]] = status.get(r["status"], 0) + 1
        log.info("%s runs: %s", kind, status)
        for r in rows:
            if r["status"].startswith("skipped"):
                log.info("  %s", json.dumps(r))
    return out


def one_pass(fm, restriction=None):
    r = restriction or {}
    steps = [("FacemapProcessing", fm.FacemapProcessing), ("FacialSignal", fm.FacialSignal),
             ("FrameSync", fm.FrameSync), ("SegmentDistance", fm.SegmentDistance)]
    for name, tbl in steps:
        n0 = len(tbl & r)
        t0 = time.time()
        errs = tbl.populate(r, reserve_jobs=True, suppress_errors=True, display_progress=False)
        for e in (errs or {}).get("error_list", []) if isinstance(errs, dict) else (errs or []):
            log.error("%s %s", name, str(e)[:500])
        log.info("%-18s %5d -> %5d  (%.0f s)", name, n0, len(tbl & r), time.time() - t0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--schema", required=True)
    ap.add_argument("--conf", default=os.path.expanduser("~/adamacs/dj_local_conf.json"))
    ap.add_argument("--no-tls", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--insert", action="store_true")
    q = sub.add_parser("populate")
    q.add_argument("--once", action="store_true")
    q.add_argument("--poll", type=int, default=600)
    q.add_argument("--task", default=None, help="restrict to one facemap_task_id")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    connect(a.conf, a.no_tls)
    fm = activate(a.schema)
    if a.cmd == "plan":
        plan(fm, a.insert)
        return
    restriction = {"facemap_task_id": a.task} if a.task else None
    while True:
        one_pass(fm, restriction)
        if a.once:
            break
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
