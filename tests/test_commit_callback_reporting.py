"""End-to-end: drive the real Commit callback and check what it leaves behind.

This is the regression test for 2026-09-16, when an ingest of six sessions printed
"Workflow finished successfully for all 6 sessions" while every DLC step in it had
died with PermissionError, and the evidence lived only in an unsaved widget.

The DataJoint layer is stubbed -- nothing is written to the database -- but the
control flow through _commit_button_callback, _process_session, _populate_dlc and
_run_ingestion_task is the real one.
"""

import json

import pytest

try:
    import adamacs.helpers.adamacs_ingest_v2 as ai
except Exception as exc:  # pragma: no cover - environment-dependent
    ai = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

pytestmark = pytest.mark.skipif(
    ai is None,
    reason="requires an importable DataJoint pipeline (%s)" % _IMPORT_ERROR,
)

from adamacs.ingest_log import StepSkipped  # noqa: E402


# --------------------------------------------------------------------------------
# the smallest widgets that _get_widget_values and the callback will accept
# --------------------------------------------------------------------------------

class _Value:
    def __init__(self, value):
        self.value = value


class _Layout:
    def __init__(self):
        self.visibility = "hidden"


class _Container:
    def __init__(self):
        self.layout = _Layout()


class _Output:
    """Stands in for ipywidgets.Output: a context manager that swallows nothing."""

    def __init__(self):
        self.cleared = 0

    def clear_output(self):
        self.cleared += 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _ProgressBar:
    def __init__(self):
        self.value = 0
        self.max = 0
        self.bar_style = ""


def _widgets_for(session_dir, dlc2, dlc3):
    return {
        "run_checkbox": _Value(True),
        "s2p_dropdown": _Value("dummy"),
        "project_dropdown": _Value("sc-lgn-actvis"),
        "location_dropdown": _Value("dummy"),
        "equipment_dropdown": _Value("dummy"),
        "same_site_dropdown": _Value(session_dir),
        "dlc1_multi": _Value(()),
        "dlc2_multi": _Value(tuple(dlc2)),
        "dlc3_multi": _Value(tuple(dlc3)),
        "note_textbox": _Value(""),
        "recording_notes_dropdown": _Value("no comment"),
        "dlc_cropping_checkbox": _Value(False),
        "curated_checkbox": _Value(False),
    }


# --------------------------------------------------------------------------------
# a DataJoint-shaped stub that writes nothing
# --------------------------------------------------------------------------------

class _Restricted:
    def __init__(self, rows):
        self._rows = rows

    def __bool__(self):
        return bool(self._rows)

    def __len__(self):
        return len(self._rows)

    def fetch(self, *fields, **kw):
        import numpy as np
        return np.array([r[fields[0]] for r in self._rows])

    def fetch1(self, field=None):
        row = self._rows[0]
        return row if field in (None, "KEY") else row[field]


class _Table:
    def __init__(self, rows):
        self._rows = rows

    def __call__(self):
        return self

    def __and__(self, _restriction):
        return _Restricted(self._rows)


EYE_MODEL = "NK; General_eye_fullsize-NK-2025-08-25; eye1_video"

SESSIONS = {
    # directory name                               -> how its DLC steps behave
    "NK_ROS-9001_2099-01-01_scanAAA0001_sessAAA0001": "ok",
    "NK_ROS-9002_2099-01-01_scanBBB0002_sessBBB0002": "permission",
    "NK_ROS-9003_2099-01-01_scanCCC0003_sessCCC0003": "no-video",
}


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Stub the database, keep the ingest control flow real."""
    import datajoint as dj

    custom = dict(dj.config.get("custom", {}) or {})
    custom["ingest_log_dir"] = str(tmp_path / "runs")
    monkeypatch.setitem(dj.config, "custom", custom)

    scan_rows = [{"session_id": "sess", "scan_id": "scanX"}]
    monkeypatch.setattr(ai.session, "Session", _Table([]), raising=False)
    monkeypatch.setattr(ai.scan, "Scan", _Table(scan_rows), raising=False)
    monkeypatch.setattr(
        ai.scan, "ScanInfo",
        _Table([{"userfunction_info": "mini2p1_openfield"}]), raising=False)
    monkeypatch.setattr(ai.isess, "ingest_session_scan", lambda *a, **k: None)

    # _process_scan is exercised separately; here it stands in for "the scan-level
    # work succeeded", so that anything the summary reports comes from DLC.
    monkeypatch.setattr(
        ai, "_process_scan",
        lambda sessi, scansi, *a, **k: ai._run_ingestion_task(
            lambda: True, "scan-level work"))

    def fake_ingest_dlc_model(scan_key, model_name, camera, aux_setup_typestr,
                              search_model_name=None, use_cropping=True):
        behaviour = fake_ingest_dlc_model.behaviour
        if behaviour == "permission":
            raise PermissionError(
                13, "Permission denied",
                "/datajoint-data/data/nataliak/…/device_%s_…_model_%s" % (camera, model_name))
        if behaviour == "no-video":
            raise StepSkipped('no video matching "face_video" in /datajoint-data/…')
        return "video_%s.mp4" % camera

    fake_ingest_dlc_model.behaviour = "ok"
    monkeypatch.setattr(ai, "_ingest_dlc_model", fake_ingest_dlc_model)

    # The database mirror is covered by tests/test_ingest_run_table.py, which runs its
    # inserts inside a transaction. Here it is replaced by a spy: these tests are about
    # the log on disk and the banner, and they must not write rows into a shared
    # database as a side effect.
    recorded = []
    monkeypatch.setattr(
        ai, "_record_run_in_database",
        lambda run, summary: recorded.append((run, summary)) or True)
    fake_ingest_dlc_model.recorded = recorded
    return fake_ingest_dlc_model


def _run_commit(session_dirs, all_widgets, do_population=False):
    output = _Output()
    bar = _ProgressBar()
    status, timer = _Value(""), _Value("")
    ai._commit_button_callback(
        None, session_dirs, all_widgets,
        ([0], ["dummy"]),                 # s2pparm_options
        output, _Container(), _Container(), bar, status, timer, _Container(),
        do_population, False, "trigger", True,
    )
    return bar, status


def _latest_run(tmp_path):
    runs = sorted((tmp_path / "runs").iterdir())
    assert runs, "no run directory was created"
    return runs[-1]


# --------------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------------

def test_a_clean_run_reports_success_and_leaves_a_full_record(harness, tmp_path, capsys):
    harness.behaviour = "ok"
    d = list(SESSIONS)[0]
    bar, status = _run_commit([d], [_widgets_for(d, [EYE_MODEL], [EYE_MODEL])])

    assert bar.bar_style == "success"
    run_dir = _latest_run(tmp_path)
    for name in ("env.json", "selections.json", "console.log", "steps.jsonl", "summary.json"):
        assert (run_dir / name).exists(), name

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["counts"]["failed"] == 0
    assert summary["sessions_successful"] == 1


def test_the_2026_09_16_failure_is_now_visible(harness, tmp_path, capsys):
    """Every DLC step fails with PermissionError. The run must not claim success."""
    harness.behaviour = "permission"
    d = list(SESSIONS)[1]
    bar, status = _run_commit([d], [_widgets_for(d, [EYE_MODEL], [EYE_MODEL])])

    printed = capsys.readouterr().out
    assert "finished successfully" not in printed
    assert bar.bar_style in ("warning", "danger")
    assert "failed" in status.value

    summary = json.loads((_latest_run(tmp_path) / "summary.json").read_text())
    assert summary["counts"]["failed"] == 2          # one per eye camera
    assert summary["sessions_failed"] == 1
    assert summary["sessions_with_failures"] == ["sessBBB0002"]
    assert any("PermissionError" in f["detail"] for f in summary["failures"])
    assert any(f.get("camera") == "mini2p1_eye_left" for f in summary["failures"])


def test_the_permission_error_text_survives_in_the_log(harness, tmp_path):
    harness.behaviour = "permission"
    d = list(SESSIONS)[1]
    _run_commit([d], [_widgets_for(d, [EYE_MODEL], [EYE_MODEL])])

    run_dir = _latest_run(tmp_path)
    console = (run_dir / "console.log").read_text()
    assert "PermissionError" in console
    assert "FAILED" in console

    steps = [json.loads(l) for l in (run_dir / "steps.jsonl").read_text().splitlines()]
    failed = [s for s in steps if s["outcome"] == "failed"]
    assert failed and "Traceback" in failed[0]["traceback"]
    assert failed[0]["model_name"] == EYE_MODEL


def test_a_model_that_matches_no_video_is_a_skip_not_a_success(harness, tmp_path):
    harness.behaviour = "no-video"
    d = list(SESSIONS)[2]
    bar, _ = _run_commit([d], [_widgets_for(d, [EYE_MODEL], [EYE_MODEL])])

    summary = json.loads((_latest_run(tmp_path) / "summary.json").read_text())
    assert summary["counts"]["skipped"] >= 2
    assert summary["counts"]["ok"] > 0        # the scan-level work still succeeded
    assert summary["counts"]["failed"] == 0
    # a skip is not a failure, so the run is still a success overall
    assert bar.bar_style == "success"
    assert any("face_video" in s["detail"] for s in summary["skips"])


def test_selecting_no_dlc_model_is_recorded(harness, tmp_path):
    harness.behaviour = "ok"
    d = list(SESSIONS)[0]
    _run_commit([d], [_widgets_for(d, [], [])])

    summary = json.loads((_latest_run(tmp_path) / "summary.json").read_text())
    assert any("no DLC models selected" in s["detail"] for s in summary["skips"])


def test_the_gui_selections_are_recoverable_afterwards(harness, tmp_path):
    """So that nobody has to send a screenshot of the GUI again."""
    harness.behaviour = "ok"
    d = list(SESSIONS)[0]
    _run_commit([d], [_widgets_for(d, [EYE_MODEL], [EYE_MODEL])])

    selections = json.loads((_latest_run(tmp_path) / "selections.json").read_text())
    assert selections[0]["project"] == "sc-lgn-actvis"
    assert selections[0]["dlc_models"] == [[], [EYE_MODEL], [EYE_MODEL]]
    assert selections[0]["use_dlc_cropping"] is False


def test_the_environment_that_explains_the_failure_is_recorded(harness, tmp_path):
    harness.behaviour = "permission"
    d = list(SESSIONS)[1]
    _run_commit([d], [_widgets_for(d, [EYE_MODEL], [EYE_MODEL])])

    env = json.loads((_latest_run(tmp_path) / "env.json").read_text())
    assert isinstance(env["groups"], list) and env["groups"]
    assert env["host"] and env["user"]
    assert env["context"]["n_sessions_selected"] == 1
    assert "sha" in env.get("git", {})


def test_a_mixed_run_attributes_failures_to_the_right_sessions(harness, tmp_path, monkeypatch):
    dirs = list(SESSIONS)
    behaviours = iter(["ok", "ok", "permission", "permission", "no-video", "no-video"])

    real = harness

    def dispatch(*args, **kwargs):
        real.behaviour = next(behaviours)
        return real(*args, **kwargs)

    monkeypatch.setattr(ai, "_ingest_dlc_model", dispatch)
    widgets = [_widgets_for(d, [EYE_MODEL], [EYE_MODEL]) for d in dirs]
    bar, status = _run_commit(dirs, widgets)

    summary = json.loads((_latest_run(tmp_path) / "summary.json").read_text())
    assert summary["sessions_selected"] == 3
    assert summary["sessions_with_failures"] == ["sessBBB0002"]
    assert summary["sessions_successful"] == 2
    assert summary["counts"]["failed"] == 2
    assert summary["counts"]["skipped"] >= 2
    assert bar.bar_style == "warning"


def test_nothing_selected_still_leaves_a_run_record(harness, tmp_path):
    d = list(SESSIONS)[0]
    widgets = _widgets_for(d, [EYE_MODEL], [EYE_MODEL])
    widgets["run_checkbox"] = _Value(False)
    _run_commit([d], [widgets])

    summary = json.loads((_latest_run(tmp_path) / "summary.json").read_text())
    assert summary["result"] == "nothing selected"


def test_a_session_that_cannot_be_ingested_says_dlc_was_skipped(harness, tmp_path, monkeypatch, capsys):
    def explode(*a, **k):
        raise RuntimeError("pyrat is down")

    monkeypatch.setattr(ai.isess, "ingest_session_scan", explode)
    d = list(SESSIONS)[0]
    _run_commit([d], [_widgets_for(d, [EYE_MODEL], [EYE_MODEL])])

    printed = capsys.readouterr().out
    assert "DLC and all later steps are skipped" in printed
    summary = json.loads((_latest_run(tmp_path) / "summary.json").read_text())
    assert any("pyrat is down" in f["detail"] for f in summary["failures"])
    assert summary["sessions_failed"] == 1


def test_the_run_is_offered_to_the_database_mirror(harness, tmp_path):
    """The callback hands the finished run to IngestRun exactly once, with the same
    summary it printed. The insert itself is tested against the real schema in
    tests/test_ingest_run_table.py."""
    harness.behaviour = "permission"
    d = list(SESSIONS)[1]
    _run_commit([d], [_widgets_for(d, [EYE_MODEL], [EYE_MODEL])])

    assert len(harness.recorded) == 1
    run, summary = harness.recorded[0]
    on_disk = json.loads((_latest_run(tmp_path) / "summary.json").read_text())
    assert summary == on_disk
    assert summary["counts"]["failed"] == 2


# --------------------------------------------------------------------------------
# review findings, 2026-09-17
# --------------------------------------------------------------------------------

def test_a_session_that_raises_outside_a_step_is_still_attributed(harness, tmp_path,
                                                                  monkeypatch):
    """Codex P1: an exception escaping _process_session used to bump a counter only.

    Its session id never reached `sessions_with_failures`, so IngestRun filed that
    session as 'ok', and combining the two counters with max() under-reported a run
    that had both kinds of failure.
    """
    dirs = list(SESSIONS)[:2]
    escaping, logged = dirs[0], dirs[1]

    real_process = ai._process_session

    def dispatch(sessi, *a, **k):
        if sessi == "sessAAA0001":
            raise RuntimeError("pyrat lookup exploded outside any step")
        return real_process(sessi, *a, **k)

    monkeypatch.setattr(ai, "_process_session", dispatch)
    harness.behaviour = "permission"          # the other session fails inside a step

    bar, status = _run_commit(dirs, [_widgets_for(d, [EYE_MODEL], [EYE_MODEL])
                                     for d in dirs])

    summary = json.loads((_latest_run(tmp_path) / "summary.json").read_text())
    assert sorted(summary["sessions_with_failures"]) == ["sessAAA0001", "sessBBB0002"]
    assert summary["sessions_failed"] == 2
    assert summary["sessions_successful"] == 0
    assert bar.bar_style == "danger"
    assert any("pyrat lookup exploded" in f["detail"] for f in summary["failures"])


def test_an_aborted_run_still_reaches_the_database_mirror(harness, tmp_path,
                                                          monkeypatch):
    """Codex P1: with suppress_errors=False the exception re-raises into the outer
    handler, which used to write only the disk summary -- losing the row for exactly
    the runs someone would later go looking for."""
    def explode(*a, **k):
        raise RuntimeError("fatal, not suppressed")

    monkeypatch.setattr(ai, "_process_session", explode)
    d = list(SESSIONS)[0]
    output, bar = _Output(), _ProgressBar()
    ai._commit_button_callback(
        None, [d], [_widgets_for(d, [EYE_MODEL], [EYE_MODEL])],
        ([0], ["dummy"]), output, _Container(), _Container(), bar,
        _Value(""), _Value(""), _Container(),
        False, False, "trigger", False,          # suppress_errors=False
    )

    assert bar.bar_style == "danger"
    assert len(harness.recorded) == 1, "the aborted run was never offered to IngestRun"
    _run, summary = harness.recorded[0]
    assert summary["result"] == "aborted"
    assert any("fatal, not suppressed" in f["detail"] for f in summary["failures"])
