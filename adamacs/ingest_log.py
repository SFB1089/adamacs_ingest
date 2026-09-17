"""A durable record of what an ingest run did, and of everything it failed to do.

The GUI writes its progress into an ``ipywidgets.Output``. That is fine while you are
watching it and worthless afterwards: the widget's contents are not saved with the
notebook unless widget state happens to be stored, so a run that went wrong leaves
nothing behind to read. On 2026-09-16 an ingest of six sessions reported
"Workflow finished successfully for all 6 sessions" while every DLC step in it had
failed with ``PermissionError``; reconstructing that took a day of forensics on the
filesystem and the DataJoint ``~log`` tables, and the decisive fact -- that the user's
supplementary groups on that host did not include the one the data mount forces --
appeared in no log at all.

This module is the answer to that. ``RunLog.start()`` creates one directory per run:

    <log_dir>/<UTC timestamp>_<user>_<run_id>/
        env.json          host, uid/gid/groups, conda env, git SHA, DataJoint target,
                          and whether each configured data root is actually writable
        selections.json   exactly what the GUI was asked to do, verbatim
        console.log       every print from the run, including those from called
                          libraries, with timestamps
        steps.jsonl       one line per step: outcome, duration, error, traceback
        summary.json      counts and the list of failures

Nothing here raises into the caller. A logging facility that can break an ingest is
worse than no logging facility, so every method swallows its own errors and, at worst,
degrades to doing nothing.
"""

from __future__ import annotations

import getpass
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


__all__ = ["StepSkipped", "RunLog", "NullRunLog", "get_run_log"]


class StepSkipped(Exception):
    """Raised by an ingest step that had nothing to do, carrying the reason.

    This is deliberately not an error. It exists so that "there was no video matching
    this model's search string" stops being indistinguishable from success: before,
    such a step printed one line, returned ``None``, and was logged as ``DONE.``
    """


# --------------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe(fn, default=None):
    """Call fn(), returning default on any failure. Logging must never raise."""
    try:
        return fn()
    except Exception:
        return default


def _json_default(obj: Any) -> str:
    """Last-resort encoder so a stray numpy scalar cannot lose us the whole record."""
    for attr in ("tolist", "item"):
        if hasattr(obj, attr):
            try:
                return getattr(obj, attr)()
            except Exception:
                pass
    return repr(obj)


def _write_json(path: Path, payload: Any) -> None:
    def _dump():
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=_json_default)
    _safe(_dump)


class _Tee:
    """Write to the real stream and to a file at once, prefixing lines with a time.

    The GUI runs its body inside ``with output_widget:``, which swaps ``sys.stdout``
    for the widget. Teeing on top of that captures the 56 prints in the ingest module
    *and* everything printed by the ingest libraries it calls, without editing them.
    """

    def __init__(self, stream, fh, stamp: bool = True):
        self._stream = stream
        self._fh = fh
        self._stamp = stamp
        self._at_line_start = True

    def write(self, text: str) -> int:
        try:
            self._stream.write(text)
        except Exception:
            pass
        try:
            if self._stamp:
                self._write_stamped(text)
            else:
                self._fh.write(text)
            self._fh.flush()
        except Exception:
            pass
        return len(text)

    def _write_stamped(self, text: str) -> None:
        prefix = _utcnow().strftime("%H:%M:%S ")
        for chunk in text.splitlines(keepends=True):
            if self._at_line_start and chunk.strip():
                self._fh.write(prefix)
            self._fh.write(chunk)
            self._at_line_start = chunk.endswith("\n")

    def flush(self) -> None:
        for target in (self._stream, self._fh):
            _safe(target.flush)

    def isatty(self) -> bool:
        return bool(_safe(self._stream.isatty, False))

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _Step:
    """One unit of ingest work. Exactly one of ok/skipped/failed is recorded."""

    def __init__(self, run: "RunLog", description: str, key: Optional[Dict[str, Any]]):
        self._run = run
        self.description = description
        self.key = dict(key or {})
        self.started = time.time()
        self.outcome: Optional[str] = None
        self.detail: str = ""
        self.traceback: str = ""

    def ok(self, result: Any = None) -> None:
        if self.outcome is None:
            self.outcome = "ok"
            self.detail = "" if result is None or result is True else str(result)[:500]

    def skipped(self, reason: str) -> None:
        if self.outcome is None:
            self.outcome = "skipped"
            self.detail = str(reason)[:500]

    def failed(self, exc: BaseException) -> None:
        if self.outcome is None:
            self.outcome = "failed"
            self.detail = "%s: %s" % (type(exc).__name__, exc)
            self.traceback = _safe(
                lambda: "".join(traceback.format_exception(
                    type(exc), exc, exc.__traceback__)), "") or ""

    def as_record(self) -> Dict[str, Any]:
        return {
            "t": datetime.fromtimestamp(self.started, timezone.utc).isoformat(),
            "duration_s": round(time.time() - self.started, 3),
            "description": self.description,
            "outcome": self.outcome or "unknown",
            "detail": self.detail,
            "traceback": self.traceback,
            **self.key,
        }


# --------------------------------------------------------------------------------
# the real thing
# --------------------------------------------------------------------------------

class RunLog:
    """One ingest run, on disk. Create with :meth:`start`."""

    def __init__(self, directory: Path, run_id: str):
        self.dir = directory
        self.run_id = run_id
        self.started = time.time()
        self.steps: List[Dict[str, Any]] = []
        self._steps_fh = _safe(lambda: open(self.dir / "steps.jsonl", "a"))

    # -- construction ------------------------------------------------------------

    @classmethod
    def start(
        cls,
        repo_root=None,
        *,
        user: Optional[str] = None,
        log_dir=None,
        context: Optional[Dict[str, Any]] = None,
    ) -> "RunLog":
        """Create the run directory and capture the environment.

        Falls back to :class:`NullRunLog` if the directory cannot be created, so a
        read-only checkout degrades to today's behaviour instead of failing the run.
        """
        run_id = uuid.uuid4().hex[:8]
        stamp = _utcnow().strftime("%Y%m%dT%H%M%SZ")
        who = user or _safe(getpass.getuser, "unknown") or "unknown"

        base = _resolve_log_dir(repo_root, log_dir)
        directory = Path(base) / ("%s_%s_%s" % (stamp, who, run_id))
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # read-only checkout, full disk, ...
            print("-- ingest log unavailable (%s: %s); continuing without it"
                  % (type(exc).__name__, exc))
            return NullRunLog()

        run = cls(directory, run_id)
        env = collect_environment(repo_root)
        if context:
            env["context"] = context
        _write_json(directory / "env.json", env)
        return run

    # -- inputs ------------------------------------------------------------------

    def record_selections(self, selections: Any) -> None:
        """Persist exactly what the GUI was asked to do.

        Worth its own file: without it the only record of which DLC model a user
        picked is a screenshot they may or may not still have.
        """
        _write_json(self.dir / "selections.json", selections)

    def record(self, name: str, payload: Any) -> None:
        """Persist any other structured artefact of the run (e.g. a preflight report)."""
        _write_json(self.dir / ("%s.json" % name), payload)

    # -- the console -------------------------------------------------------------

    @contextmanager
    def tee(self):
        """Mirror stdout and stderr into ``console.log`` for the duration of the block."""
        fh = _safe(lambda: open(self.dir / "console.log", "a"))
        if fh is None:
            yield self
            return
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout = _Tee(old_out, fh)
            sys.stderr = _Tee(old_err, fh)
            yield self
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            _safe(fh.flush)
            _safe(fh.close)

    # -- the ledger --------------------------------------------------------------

    @contextmanager
    def step(self, description: str, key: Optional[Dict[str, Any]] = None):
        """Record one step. An outcome is always written, even if the body raises."""
        s = _Step(self, description, key)
        try:
            yield s
        finally:
            if s.outcome is None:
                s.outcome = "unknown"
            self._append(s.as_record())

    def _append(self, record: Dict[str, Any]) -> None:
        self.steps.append(record)
        if self._steps_fh is not None:
            def _w():
                self._steps_fh.write(json.dumps(record, default=_json_default) + "\n")
                self._steps_fh.flush()
            _safe(_w)

    # -- reading it back ---------------------------------------------------------

    def counts(self, **restriction) -> Dict[str, int]:
        out = {"ok": 0, "skipped": 0, "failed": 0, "unknown": 0}
        for rec in self._select(**restriction):
            out[rec.get("outcome", "unknown")] = out.get(rec.get("outcome", "unknown"), 0) + 1
        return out

    def failures(self, **restriction) -> List[Dict[str, Any]]:
        return [r for r in self._select(**restriction) if r.get("outcome") == "failed"]

    def skips(self, **restriction) -> List[Dict[str, Any]]:
        return [r for r in self._select(**restriction) if r.get("outcome") == "skipped"]

    def _select(self, **restriction) -> Iterable[Dict[str, Any]]:
        if not restriction:
            return list(self.steps)
        return [r for r in self.steps
                if all(r.get(k) == v for k, v in restriction.items())]

    def session_ok(self, session_id: str) -> bool:
        """True when nothing failed for this session. Used instead of counting only
        exceptions that escape ``_process_session``, which is how a run with twelve
        dead DLC steps came to be reported as fully successful."""
        return not self.failures(session_id=session_id)

    # -- finishing ---------------------------------------------------------------

    def finish(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        c = self.counts()
        summary = {
            "run_id": self.run_id,
            "directory": str(self.dir),
            "finished": _utcnow().isoformat(),
            "duration_s": round(time.time() - self.started, 1),
            "counts": c,
            "failures": [
                {k: v for k, v in f.items() if k != "traceback"} for f in self.failures()
            ],
            "skips": [
                {k: v for k, v in s.items() if k != "traceback"} for s in self.skips()
            ],
        }
        if extra:
            summary.update(extra)
        _write_json(self.dir / "summary.json", summary)
        if self._steps_fh is not None:
            _safe(self._steps_fh.close)
            self._steps_fh = None
        return summary

    def format_summary(self, max_listed: int = 10) -> str:
        """The block the GUI prints at the end. Says failed when things failed."""
        c = self.counts()
        total = sum(c.values())
        lines = [
            "=" * 60,
            "%d steps: %d ok, %d skipped, %d failed"
            % (total, c["ok"], c["skipped"], c["failed"]),
        ]
        for label, rows in (("FAILED", self.failures()), ("SKIPPED", self.skips())):
            for r in rows[:max_listed]:
                where = " ".join(
                    str(r[k]) for k in ("session_id", "scan_id", "recording_id") if r.get(k))
                lines.append("  %-7s %s %s" % (label, where, r.get("description", "")))
                if r.get("detail"):
                    lines.append("          %s" % r["detail"])
            if len(rows) > max_listed:
                lines.append("  ... and %d more %s" % (len(rows) - max_listed, label.lower()))
        lines.append("full log: %s" % self.dir)
        lines.append("=" * 60)
        return "\n".join(lines)


class NullRunLog(RunLog):
    """Stand-in used when no log directory could be created, and as the default
    argument of :func:`_run_ingestion_task` so existing call sites keep working."""

    def __init__(self):  # noqa: D107 - deliberately does not call super()
        self.dir = None
        self.run_id = "no-log"
        self.started = time.time()
        self.steps = []
        self._steps_fh = None

    def record_selections(self, selections): pass

    def record(self, name, payload): pass

    @contextmanager
    def tee(self):
        yield self

    def finish(self, extra=None):
        return {"run_id": self.run_id, "counts": self.counts()}

    def format_summary(self, max_listed: int = 10) -> str:
        c = self.counts()
        return "%d steps: %d ok, %d skipped, %d failed (no log directory)" % (
            sum(c.values()), c["ok"], c["skipped"], c["failed"])


def get_run_log(run) -> RunLog:
    """Normalise an optional run log argument."""
    return run if isinstance(run, RunLog) else NullRunLog()


# --------------------------------------------------------------------------------
# environment capture
# --------------------------------------------------------------------------------

def _resolve_log_dir(repo_root, log_dir) -> Path:
    """Where runs are written: explicit argument, then dj.config, then repo/logs."""
    if log_dir:
        return Path(log_dir)
    cfg = _safe(lambda: __import__("datajoint").config["custom"].get("ingest_log_dir"))
    if cfg:
        return Path(cfg)
    if repo_root:
        return Path(repo_root) / "logs" / "ingest"
    return Path.home() / ".adamacs_ingest_logs"


def _git_state(repo_root) -> Dict[str, Any]:
    if not repo_root:
        return {}

    def _run(*args):
        return subprocess.run(args, cwd=str(repo_root), capture_output=True,
                              text=True, timeout=10).stdout.strip()

    sha = _safe(lambda: _run("git", "rev-parse", "HEAD"))
    if not sha:
        return {}
    return {
        "sha": sha,
        "branch": _safe(lambda: _run("git", "rev-parse", "--abbrev-ref", "HEAD")),
        "dirty": bool(_safe(lambda: _run("git", "status", "--porcelain"))),
        "describe": _safe(lambda: _run("git", "describe", "--always", "--dirty")),
    }


def probe_writable(path) -> Dict[str, Any]:
    """Can this process actually create something here?

    ``os.access`` alone is not enough. The lab's data share is mounted over CIFS on
    the GPU hosts with ``forceuid,forcegid,dir_mode=0775``, so the permissions the
    client sees are synthetic and the only honest test is to create a directory and
    remove it again.
    """
    p = Path(path)
    out = {
        "path": str(p),
        "exists": bool(_safe(p.exists, False)),
        "access_w_ok": bool(_safe(lambda: os.access(str(p), os.W_OK), False)),
        "access_x_ok": bool(_safe(lambda: os.access(str(p), os.X_OK), False)),
        "mkdir_ok": None,
        "mkdir_error": None,
    }
    if not out["exists"]:
        return out
    probe = p / (".adamacs_write_probe_%d_%s" % (os.getpid(), uuid.uuid4().hex[:6]))
    try:
        probe.mkdir()
    except Exception as exc:
        out["mkdir_ok"] = False
        out["mkdir_error"] = "%s: %s" % (type(exc).__name__, exc)
    else:
        out["mkdir_ok"] = True
        _safe(probe.rmdir)
    return out


def collect_environment(repo_root=None) -> Dict[str, Any]:
    """Everything needed to explain, months later, why a run behaved as it did."""
    env: Dict[str, Any] = {
        "started": _utcnow().isoformat(),
        "host": _safe(socket.gethostname),
        "platform": _safe(platform.platform),
        "user": _safe(getpass.getuser),
        "uid": _safe(os.getuid),
        "gid": _safe(os.getgid),
        # The one that mattered: a user missing the group the data mount forces
        # cannot create an output directory, and nothing else in the log says so.
        "groups": _safe(lambda: sorted(os.getgroups()), []),
        "cwd": _safe(lambda: str(Path.cwd())),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "git": _git_state(repo_root),
    }

    def _dj():
        import datajoint as dj
        custom = dict(dj.config.get("custom", {}) or {})
        roots: List[Dict[str, Any]] = []
        for name in ("exp_root_data_dir", "imaging_root_data_dir",
                     "dlc_root_data_dir", "dlc_processed_data_dir"):
            value = custom.get(name)
            if not value:
                continue
            for entry in (value if isinstance(value, (list, tuple)) else [value]):
                roots.append(dict(probe_writable(entry), config_key=name))
        return {
            "version": dj.__version__,
            "host": dj.config.get("database.host"),
            "user": dj.config.get("database.user"),
            "prefix": custom.get("database.prefix"),
            "roots": roots,
        }

    env["datajoint"] = _safe(_dj, {})
    return env
