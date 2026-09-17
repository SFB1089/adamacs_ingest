"""Check what an ingest is about to need, before it writes anything.

The run log says what went wrong. This says what is *going* to go wrong, while it is
still one line in a widget rather than a day of forensics.

It exists because of 2026-09-16, when six sessions were re-ingested and every DLC
step failed with ``PermissionError``: the account was missing the group that the CIFS
data mount forces, so the output directory could not be created. Nothing in the run
could have known that in advance, because nothing looked. One ``mkdir`` probe would
have said so at second zero:

    PREFLIGHT  ERROR  sess9FUDDCP7 mini2p1_eye_left
               cannot create the DLC output directory in
               /datajoint-data/data/nataliak/NK_ROS-2075_...
               PermissionError: [Errno 13] Permission denied

The checks are read-only apart from the write probes, which create a uniquely named
directory and remove it again. Nothing here raises: a check that cannot run reports
itself as ``unknown`` rather than taking the ingest down with it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from adamacs.ingest_log import probe_writable


__all__ = ["Finding", "PreflightReport", "run_preflight"]

ERROR, WARNING, OK, UNKNOWN = "error", "warning", "ok", "unknown"


class Finding(dict):
    """One check: its level, what it was about, and what it found."""

    def __init__(self, level: str, check: str, detail: str = "", **scope):
        super().__init__(level=level, check=check, detail=detail, **scope)

    @property
    def level(self) -> str:
        return self["level"]

    def where(self) -> str:
        return " ".join(
            str(self[k]) for k in ("session", "camera", "model_name") if self.get(k))


class PreflightReport(list):
    """The findings, plus the two questions anyone actually asks of them."""

    @property
    def counts(self) -> Dict[str, int]:
        out = {ERROR: 0, WARNING: 0, OK: 0, UNKNOWN: 0}
        for f in self:
            out[f["level"]] = out.get(f["level"], 0) + 1
        return out

    @property
    def ok(self) -> bool:
        return self.counts[ERROR] == 0

    def of(self, level: str) -> List[Finding]:
        return [f for f in self if f["level"] == level]

    def as_json(self) -> Dict[str, Any]:
        return {"ok": self.ok, "counts": self.counts, "findings": list(self)}

    def format(self, max_listed: int = 20) -> str:
        c = self.counts
        lines = ["-" * 60,
                 "preflight: %d checks, %d error, %d warning, %d could not be checked"
                 % (sum(c.values()), c[ERROR], c[WARNING], c[UNKNOWN])]
        for level in (ERROR, WARNING, UNKNOWN):
            rows = self.of(level)
            for f in rows[:max_listed]:
                lines.append("  %-8s %s %s" % (level.upper(), f.where(), f["check"]))
                if f.get("detail"):
                    lines.append("           %s" % f["detail"])
            if len(rows) > max_listed:
                lines.append("  ... and %d more %s" % (len(rows) - max_listed, level))
        if c[ERROR] == 0 and c[WARNING] == 0:
            lines.append("  nothing to report")
        lines.append("-" * 60)
        return "\n".join(lines)


# --------------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------------

def _check_session_directory(report, session_label, directory):
    """Exists, and can this account create something in it?

    ``os.access`` is not enough on the lab's CIFS mounts, where ``forceuid``,
    ``forcegid`` and ``dir_mode=0775`` make the permissions the client sees
    synthetic; only an actual mkdir tells the truth.
    """
    probe = probe_writable(directory)
    if not probe["exists"]:
        report.append(Finding(
            ERROR, "session directory is missing", str(directory), session=session_label))
        return False
    if probe["mkdir_ok"] is False:
        report.append(Finding(
            ERROR, "cannot create files in the session directory",
            "%s -- %s" % (directory, probe["mkdir_error"]), session=session_label))
        return False
    report.append(Finding(OK, "session directory is writable", str(directory),
                          session=session_label))
    return True


def _check_model_registered(report, model_table, model_name):
    try:
        known = bool(model_table & {"model_name": model_name})
    except Exception as exc:
        report.append(Finding(UNKNOWN, "could not check the model table",
                              "%s: %s" % (type(exc).__name__, exc),
                              model_name=model_name))
        return
    if known:
        report.append(Finding(OK, "model is registered", model_name=model_name))
    else:
        report.append(Finding(
            ERROR, "model is not in model.Model",
            "it cannot be selected or triggered until it is registered",
            model_name=model_name))


def _check_video_resolves(report, session_label, directory, camera, model_name,
                          video_idx, candidates_fn, search_name_fn):
    """Does this model's name actually resolve to a video in this session folder?

    The ingest derives the filename pattern from the third ';'-separated field of the
    model name and silently skips when nothing matches, which is how a model named
    '...; Topcam' -- capital T, no '_video' -- came to be unusable for every mini2p1
    dataset in the lab without anyone noticing.
    """
    search_name = search_name_fn(model_name, video_idx)
    try:
        search_str, hits = candidates_fn(directory, search_name)
    except Exception as exc:
        report.append(Finding(UNKNOWN, "could not search for a video",
                              "%s: %s" % (type(exc).__name__, exc),
                              session=session_label, camera=camera,
                              model_name=model_name))
        return
    scope = dict(session=session_label, camera=camera, model_name=model_name)
    if not hits:
        report.append(Finding(
            ERROR, "no video matches this model",
            'searched for "*%s*.mp4*" in %s -- the pattern is the 3rd ";" field of '
            '"%s"' % (search_str, directory, search_name), **scope))
    elif len(hits) > 1:
        report.append(Finding(
            WARNING, "several videos match this model",
            'pattern "*%s*.mp4*" matches %d files and the ingest takes whichever the '
            "filesystem lists first: %s"
            % (search_str, len(hits), ", ".join(p.name for p in hits)), **scope))
    else:
        report.append(Finding(OK, "video resolves", hits[0].name, **scope))


def _check_processed_dir(report, get_processed_dir):
    try:
        target = get_processed_dir()
    except Exception as exc:
        report.append(Finding(UNKNOWN, "could not read dlc_processed_data_dir",
                              "%s: %s" % (type(exc).__name__, exc)))
        return
    if not target:
        report.append(Finding(
            OK, "dlc_processed_data_dir is unset",
            "DLC output will be written next to each video"))
        return
    probe = probe_writable(target)
    if probe["mkdir_ok"] is False:
        report.append(Finding(ERROR, "dlc_processed_data_dir is not writable",
                              "%s -- %s" % (target, probe["mkdir_error"])))
    else:
        report.append(Finding(OK, "dlc_processed_data_dir is writable", str(target)))


# --------------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------------

def run_preflight(
    selections: Sequence[Dict[str, Any]],
    data_root,
    *,
    cameras: Optional[Sequence[str]] = None,
    model_table=None,
    get_processed_dir=None,
    candidates_fn=None,
    search_name_fn=None,
) -> PreflightReport:
    """Check everything the ingest is about to need. Writes nothing but probes.

    Args:
        selections: the GUI's selections, as returned by ``_get_widget_values``.
        data_root: directory the session folders live in.
        cameras: camera names per selection, in GUI order (three per session).
        model_table: ``model.Model``, for the registration check.
        get_processed_dir: ``get_dlc_processed_data_dir``.
        candidates_fn / search_name_fn: injected from adamacs_ingest_v2 so that this
            uses the same video-resolution rule as the ingest.
    """
    report = PreflightReport()
    if candidates_fn is None or search_name_fn is None:
        from adamacs.helpers.adamacs_ingest_v2 import (  # noqa: WPS433 - avoid a cycle
            dlc_search_name, dlc_video_candidates)
        candidates_fn = candidates_fn or dlc_video_candidates
        search_name_fn = search_name_fn or dlc_search_name

    report.append(Finding(OK, "process identity",
                          "uid=%s gid=%s groups=%s" % (
                              os.getuid(), os.getgid(), sorted(os.getgroups()))))

    if get_processed_dir is not None:
        _check_processed_dir(report, get_processed_dir)

    checked_models = set()
    for i, sel in enumerate(selections):
        label = sel.get("session_id") or sel.get("path") or "selection %d" % i
        directory = Path(data_root) / sel["path"]
        usable = _check_session_directory(report, label, directory)

        cams = (cameras[i] if cameras and i < len(cameras) else None) or []
        for video_idx, models in enumerate(sel.get("dlc_models") or []):
            camera = cams[video_idx] if video_idx < len(cams) else "camera_%d" % video_idx
            for model_name in models or []:
                if model_name == "dummy":
                    continue
                if model_table is not None and model_name not in checked_models:
                    checked_models.add(model_name)
                    _check_model_registered(report, model_table, model_name)
                if usable:
                    _check_video_resolves(report, label, directory, camera, model_name,
                                          video_idx, candidates_fn, search_name_fn)

    any_model = any(
        name != "dummy"
        for sel in selections
        for per_camera in (sel.get("dlc_models") or [])
        for name in (per_camera or [])
    )
    if not any_model:
        report.append(Finding(
            WARNING, "no DLC model is selected for any camera",
            "the ingest will not create any pose estimation task"))

    return report
