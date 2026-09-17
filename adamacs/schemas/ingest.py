"""A record in the database of every ingest run, and of what it failed to do.

``adamacs.ingest_log`` writes a run's full evidence to disk. That is the right place
for console output and tracebacks, but it answers only the question you already know
to ask. This schema answers the other direction: given a session that looks wrong, or
one whose rows are missing, which ingest run touched it, who ran it, from which
commit, on which host, and what did that run report at the time.

Two deliberate design choices:

* ``IngestRun`` has **no foreign keys**. The runs worth investigating are exactly the
  ones whose sessions were never created, or were later deleted; a foreign key would
  make those unrecordable in the first case and would cascade the evidence away in the
  second. ``session_id`` is therefore a plain attribute in the part table, not a
  reference to ``session.Session``.
* Nothing here is required for an ingest to work. :func:`record_run` swallows its own
  errors, so a database that is unreachable, or a schema that has not been created,
  costs you the row and nothing else.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import datajoint as dj

from ..pipeline import db_prefix

schema = dj.schema(db_prefix + "ingest")  # e.g. "roselab_ingest"


@schema
class IngestRun(dj.Manual):
    """One press of the Commit button in the ingest GUI."""

    definition = """
    # One ingest run: who, where, from which commit, and how it ended
    run_id                  : char(8)        # short uuid, also names the log directory
    ---
    run_start               : datetime       # UTC
    run_end=null            : datetime       # UTC; null if the run never finished
    user_initials=null      : varchar(8)     # as derived from the session folder names
    os_user=null            : varchar(64)    # account the process ran under
    host=null               : varchar(64)    # machine the ingest ran on
    git_sha=null            : varchar(40)    # adamacs_ingest commit
    git_dirty=0             : tinyint        # 1 if the checkout had local changes
    log_dir=null            : varchar(512)   # full evidence: console, steps, env
    n_sessions=0            : int            # sessions selected in the GUI
    n_sessions_failed=0     : int
    n_steps_ok=0            : int
    n_steps_skipped=0       : int
    n_steps_failed=0        : int
    environment=null        : longblob       # env.json, including uid/gid/groups
    selections=null         : longblob       # exactly what the GUI was asked to do
    summary=null            : longblob       # summary.json
    """

    class SessionOutcome(dj.Part):
        """Per-session result. `session_id` is intentionally not a foreign key."""

        definition = """
        # What this run did to one session
        -> master
        session_id              : varchar(32)   # NOT a foreign key -- see module docstring
        ---
        outcome                 : enum('ok','failed')
        session_steps_ok=0      : int           # named apart from the master's counts:
        session_steps_skipped=0 : int           # DataJoint refuses to join two tables
        session_steps_failed=0  : int           # that share a secondary attribute name
        first_error=''          : varchar(1000) # the first failure, for triage at a glance
        """

    class Failure(dj.Part):
        """Every failed or skipped step, so the table alone can answer "why"."""

        definition = """
        -> master
        failure_id          : smallint
        ---
        session_id=''       : varchar(32)
        scan_id=''          : varchar(32)
        camera=''           : varchar(64)
        model_name=''       : varchar(255)
        outcome             : enum('failed','skipped')
        description=''      : varchar(255)
        detail=''           : varchar(1000)
        """


# --------------------------------------------------------------------------------
# writing a run into the schema
# --------------------------------------------------------------------------------

def _truncate(value: Any, length: int) -> str:
    return ("" if value is None else str(value))[:length]


def _started_at(run) -> str:
    """UTC start time of a run that has no env.json to read it from."""
    from datetime import datetime, timezone
    epoch = getattr(run, "started", None) or datetime.now(timezone.utc).timestamp()
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def record_run(run, summary: Optional[Dict[str, Any]] = None, verbose: bool = True) -> bool:
    """Insert one :class:`adamacs.ingest_log.RunLog` into :class:`IngestRun`.

    Returns True when the row was written. Never raises: an ingest must not fail
    because its bookkeeping could not be stored.
    """
    try:
        if getattr(run, "dir", None) is None and not getattr(run, "steps", None):
            return False  # nothing at all to record

        summary = summary or run.finish()
        env = {}
        if run.dir:
            try:
                env = json.loads((run.dir / "env.json").read_text())
            except Exception:
                env = {}
        selections = []
        if run.dir:
            try:
                selections = json.loads((run.dir / "selections.json").read_text())
            except Exception:
                selections = []

        counts = summary.get("counts", {})
        git = env.get("git", {}) or {}
        # run_start is NOT NULL. A NullRunLog has no env.json, and that is exactly the
        # case where the database row is the only record left, so fall back to the
        # run's in-memory start time rather than losing the row to a rejected insert.
        started = (env.get("started") or "").replace("T", " ")[:19] or _started_at(run)
        row = {
            "run_id": run.run_id,
            "run_start": started,
            "run_end": summary.get("finished", "").replace("T", " ")[:19] or None,
            "user_initials": _truncate(env.get("context", {}).get("user_initials"), 8) or None,
            "os_user": _truncate(env.get("user"), 64) or None,
            "host": _truncate(env.get("host"), 64) or None,
            "git_sha": _truncate(git.get("sha"), 40) or None,
            "git_dirty": 1 if git.get("dirty") else 0,
            "log_dir": _truncate(run.dir, 512) or None,
            "n_sessions": int(summary.get("sessions_selected", 0) or 0),
            "n_sessions_failed": int(summary.get("sessions_failed", 0) or 0),
            "n_steps_ok": int(counts.get("ok", 0)),
            "n_steps_skipped": int(counts.get("skipped", 0)),
            "n_steps_failed": int(counts.get("failed", 0)),
            "environment": env,
            "selections": selections,
            "summary": summary,
        }
        IngestRun.insert1(row, skip_duplicates=True)

        failed_sessions = set(summary.get("sessions_with_failures", []) or [])
        session_ids = {
            s.get("session_id") for s in (selections or []) if s.get("session_id")
        } | failed_sessions
        outcomes = []
        for sid in sorted(session_ids):
            c = run.counts(session_id=sid)
            first = run.failures(session_id=sid)
            outcomes.append({
                "run_id": run.run_id,
                "session_id": _truncate(sid, 32),
                "outcome": "failed" if sid in failed_sessions or c["failed"] else "ok",
                "session_steps_ok": c["ok"],
                "session_steps_skipped": c["skipped"],
                "session_steps_failed": c["failed"],
                "first_error": _truncate(first[0]["detail"] if first else "", 1000),
            })
        if outcomes:
            IngestRun.SessionOutcome.insert(outcomes, skip_duplicates=True)

        problems = [dict(p, outcome="failed") for p in run.failures()] + \
                   [dict(p, outcome="skipped") for p in run.skips()]
        if problems:
            IngestRun.Failure.insert([
                {
                    "run_id": run.run_id,
                    "failure_id": i,
                    "session_id": _truncate(p.get("session_id"), 32),
                    "scan_id": _truncate(p.get("scan_id"), 32),
                    "camera": _truncate(p.get("camera"), 64),
                    "model_name": _truncate(p.get("model_name"), 255),
                    "outcome": p["outcome"],
                    "description": _truncate(p.get("description"), 255),
                    "detail": _truncate(p.get("detail"), 1000),
                }
                for i, p in enumerate(problems)
            ], skip_duplicates=True)

        if verbose:
            print(f'-- ingest run recorded in {schema.database}.IngestRun '
                  f'(run_id="{run.run_id}")')
        return True
    except Exception as exc:  # bookkeeping must never break an ingest
        if verbose:
            print(f'-- could not record the ingest run in the database '
                  f'({type(exc).__name__}: {exc}); the log on disk is unaffected')
        return False


def runs_touching(session_id: str):
    """Every recorded run that processed this session, newest first.

    The query to reach for when a session looks wrong and you want to know which
    ingest produced it.
    """
    return (IngestRun * IngestRun.SessionOutcome
            & f'session_id = "{session_id}"')
