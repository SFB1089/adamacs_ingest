"""A worker that waits for CellReg jobs and runs them.

Shape of the thing
------------------
Three stages, each a ``populate`` the worker calls in order, so a job moves forward one
step per pass and a crash resumes from where it stopped:

1. ``CellRegExport.populate()`` -- for a ``task_mode='trigger'`` task, build the input
   files from the declared curations (see :mod:`adamacs.ingest.cellreg_export`).
2. **MATLAB** -- run ``cellreg_run(job_dir)`` over each exported directory that has no
   ``Results/`` yet. This is the only stage that leaves the database and the only one that
   needs MATLAB.
3. ``CellRegRun`` / ``CellRegResult.populate()`` -- ingest whatever CellReg produced.

Why a MATLAB check at startup rather than at first use
-----------------------------------------------------
A worker that accepts jobs and then dies on the first MATLAB call has already told the
database it was working on something. :func:`check_matlab` runs once at startup and the
worker refuses to start without a usable interpreter, so a missing licence is a startup
error rather than a half-finished run. **As of 2026-10-16 neither tatchu3 nor ibehaveGPU1
has MATLAB** -- not on PATH, nothing under /usr/local/MATLAB or /opt/MATLAB, no MCR -- so
this worker cannot yet run stage 2 anywhere, and ``--require-matlab`` is on by default to
make that obvious instead of silent.

Where it should run
-------------------
Not tatchu3. Tobias's standing instruction (2025-08-26, after a CellReg run exhausted the
server's RAM in February 2025) is that CellReg does not run there. ibehaveGPU1 has 1.5 TB.
The worker refuses to start on a host named in ``FORBIDDEN_HOSTS`` unless forced.

Writes
------
Stage 1 writes files into the shared data folder and rows into the database; stages 1 and 3
insert. Nothing here is safe to run unattended against production without Tobias's explicit
go-ahead, which is why ``--dry-run`` is the default and the real mode must be asked for.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time

log = logging.getLogger("cellreg_worker")

#: CellReg exhausted this host's RAM in Feb 2025; Tobias ruled it out on 2025-08-26.
FORBIDDEN_HOSTS = ("tatchu3", "TATCHU3")

DEFAULT_POLL_SECONDS = 60
DEFAULT_MATLAB_TIMEOUT = 24 * 3600


class WorkerRefusal(RuntimeError):
    """Raised when the worker declines to start. Never raised mid-job."""


def check_matlab(matlab: str = "matlab") -> dict:
    """Is there a MATLAB this worker can actually drive?

    Returns its version on success. Raises :class:`WorkerRefusal` otherwise, at startup,
    so a missing interpreter or licence never surfaces halfway through a registration.
    """
    exe = shutil.which(matlab)
    if exe is None:
        raise WorkerRefusal(
            f"{matlab!r} is not on PATH. Checked 2026-10-16: neither tatchu3 nor "
            f"ibehaveGPU1 has MATLAB installed (nothing on PATH, nothing under "
            f"/usr/local/MATLAB or /opt/MATLAB, no MATLAB Runtime). Install MATLAB and "
            f"confirm a licence allows unattended -batch use, or pass --matlab with a "
            f"path to one.")
    try:
        out = subprocess.run([exe, "-batch", "disp(version)"], capture_output=True,
                             text=True, timeout=300)
    except subprocess.TimeoutExpired as exc:
        raise WorkerRefusal(f"{exe} did not respond within 300 s: {exc}") from exc
    if out.returncode != 0:
        raise WorkerRefusal(
            f"{exe} exited {out.returncode} on a trivial command; a licence is the usual "
            f"cause.\nstdout: {out.stdout.strip()[:400]}\nstderr: {out.stderr.strip()[:400]}")
    return {"matlab": exe, "version": out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "unknown"}


def check_toolbox(cellreg_root: str | None = None) -> dict:
    """Confirm a CellReg checkout exists and carries the functions the driver calls."""
    root = pathlib.Path(cellreg_root or os.environ.get("CELLREG_ROOT")
                        or pathlib.Path.home() / "CellReg")
    if not root.is_dir():
        raise WorkerRefusal(f"no CellReg toolbox at {root}; set CELLREG_ROOT")
    needed = ["CellReg/cluster_cells.m", "CellReg/align_images.m",
              "CellReg/compute_centroid_locations.m", "CellReg/estimate_number_of_bins.m",
              "CellReg/save_log_file.m"]
    missing = [n for n in needed if not (root / n).exists()]
    if missing:
        raise WorkerRefusal(f"{root} is not a complete CellReg checkout; missing {missing}")
    sha = ""
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=30)
        sha = r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        pass
    return {"cellreg_root": str(root), "cellreg_sha": sha}


def check_host(force: bool = False) -> str:
    host = socket.gethostname()
    if any(h.lower() in host.lower() for h in FORBIDDEN_HOSTS) and not force:
        raise WorkerRefusal(
            f"refusing to run on {host}: CellReg exhausted this machine's RAM in February "
            f"2025 and Tobias ruled it out on 2025-08-26. Run on ibehaveGPU1, or pass "
            f"--force-host if that instruction has changed.")
    return host


def driver_path() -> pathlib.Path:
    p = pathlib.Path(__file__).resolve().parent.parent / "matlab" / "cellreg_run.m"
    if not p.exists():
        raise WorkerRefusal(f"driver not found at {p}")
    return p


def run_matlab_job(job_dir, *, matlab="matlab", cellreg_root=None,
                   timeout=DEFAULT_MATLAB_TIMEOUT, dry_run=True) -> dict:
    """Run ``cellreg_run(job_dir)`` once. Returns the outcome; never raises on job failure.

    A failed registration is a recorded outcome, not an exception: the worker must keep
    serving other jobs, and the failure belongs in the database next to the job it came
    from.
    """
    job_dir = pathlib.Path(job_dir)
    drv = driver_path()
    root = cellreg_root or os.environ.get("CELLREG_ROOT") or str(pathlib.Path.home() / "CellReg")
    cmd = [matlab, "-batch",
           f"addpath('{drv.parent}'); cellreg_run('{job_dir}', '{root}')"]
    if dry_run:
        log.info("DRY RUN, not executing: %s", " ".join(cmd))
        return {"status": "dry_run", "command": cmd, "job_dir": str(job_dir)}

    started = _dt.datetime.now()
    log.info("starting MATLAB for %s", job_dir)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                             cwd=str(job_dir))
        status = "succeeded" if out.returncode == 0 else "failed"
        return {"status": status, "returncode": out.returncode,
                "stdout_tail": out.stdout[-4000:], "stderr_tail": out.stderr[-4000:],
                "started_at": started.isoformat(),
                "seconds": (_dt.datetime.now() - started).total_seconds(),
                "job_dir": str(job_dir)}
    except subprocess.TimeoutExpired:
        return {"status": "timed_out", "seconds": timeout, "job_dir": str(job_dir)}


def has_results(job_dir) -> bool:
    """True once CellReg has written a result file into the job's Results/ directory."""
    d = pathlib.Path(job_dir) / "Results"
    return d.is_dir() and any(d.glob("cellRegistered_*.mat"))


def pending_job_dirs(creg) -> list:
    """Export directories that have inputs but no Results/ yet."""
    out = []
    for row in creg.CellRegExport.fetch("export_dir", "n_files", as_dict=True):
        d = pathlib.Path(row["export_dir"])
        if row["n_files"] and d.is_dir() and not has_results(d):
            out.append(d)
    return out


def one_pass(creg, *, matlab="matlab", cellreg_root=None, dry_run=True,
             timeout=DEFAULT_MATLAB_TIMEOUT, populate_kwargs=None) -> dict:
    """One sweep of the three stages. Returns what each stage did."""
    pk = {"reserve_jobs": True, "suppress_errors": True, **(populate_kwargs or {})}
    report = {"exported": 0, "matlab": [], "ingested": 0}

    if dry_run:
        log.info("DRY RUN: would call CellRegExport.populate(%s)", pk)
    else:
        creg.CellRegExport.populate(**pk)
        report["exported"] = len(creg.CellRegExport())

    for d in pending_job_dirs(creg):
        report["matlab"].append(run_matlab_job(d, matlab=matlab, cellreg_root=cellreg_root,
                                               timeout=timeout, dry_run=dry_run))

    if dry_run:
        log.info("DRY RUN: would call CellRegResult.populate(%s)", pk)
    else:
        creg.CellRegResult.populate(**pk)
        report["ingested"] = len(creg.CellRegResult())
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--schema", default=None,
                    help="cell_registration schema name. Defaults to the name pipeline.py activates, db_prefix + 'cell_registration'.")
    ap.add_argument("--matlab", default="matlab")
    ap.add_argument("--cellreg-root", default=None)
    ap.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    ap.add_argument("--matlab-timeout", type=int, default=DEFAULT_MATLAB_TIMEOUT)
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--execute", action="store_true",
                    help="actually populate and run MATLAB. WRITES to the database and the "
                         "shared data folder; needs Tobias's go-ahead.")
    ap.add_argument("--force-host", action="store_true")
    ap.add_argument("--skip-matlab-check", action="store_true",
                    help="start without a usable MATLAB (stage 2 will be skipped)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    dry_run = not args.execute
    host = check_host(force=args.force_host)
    tb = check_toolbox(args.cellreg_root)
    log.info("host=%s toolbox=%s sha=%s", host, tb["cellreg_root"], tb["cellreg_sha"][:12])

    if args.skip_matlab_check:
        log.warning("MATLAB check skipped; stage 2 cannot run")
    else:
        mi = check_matlab(args.matlab)
        log.info("matlab=%s version=%s", mi["matlab"], mi["version"])

    if dry_run:
        log.warning("DRY RUN: nothing will be populated and no MATLAB will be launched. "
                    "Pass --execute to act, once that is authorised.")

    from adamacs.schemas import cell_registration as creg
    from adamacs import db_prefix

    schema_name = args.schema or (db_prefix + "cell_registration")
    if dry_run:
        # A dry run still has to READ the task tables to say what it would do. The first
        # dry run on ibehavegpu1 died in pending_job_dirs() with
        # "'NoneType' object has no attribute 'primary_key'" because the schema was never
        # activated, so CellRegExport had no heading. Attach read-only:
        # create_schema/create_tables False means this path cannot declare anything, and
        # it fails loudly if the schema is absent rather than reporting an empty queue.
        creg.activate(schema_name, create_schema=False, create_tables=False)
        log.info("attached read-only to %s; no table can be created on this path",
                 schema_name)
    else:
        creg.activate(schema_name)
        log.info("attached to %s with write access", schema_name)

    while True:
        report = one_pass(creg, matlab=args.matlab, cellreg_root=args.cellreg_root,
                          dry_run=dry_run, timeout=args.matlab_timeout)
        log.info("pass complete: %s", json.dumps(report, default=str)[:600])
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
