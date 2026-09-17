"""Tests for adamacs.ingest_log.

These are deliberately about the failure modes that made a real ingest unreadable:
an error that was printed and then lost, a skip that was indistinguishable from a
success, and a run that reported "finished successfully" while every DLC step in it
had failed.
"""

import json
import os
import sys

import pytest

# No datajoint stub here: conftest.py settles which package the whole session gets,
# and ingest_log imports datajoint lazily, so these tests never open a connection.

from adamacs.ingest_log import (
    NullRunLog,
    RunLog,
    StepSkipped,
    collect_environment,
    get_run_log,
    probe_writable,
)


# --------------------------------------------------------------------------------
# the run directory and what goes in it
# --------------------------------------------------------------------------------

def test_start_creates_run_directory_with_environment(tmp_path):
    run = RunLog.start(log_dir=tmp_path, user="ZZ")
    assert run.dir.is_dir()
    assert run.dir.parent == tmp_path
    assert "ZZ" in run.dir.name and run.run_id in run.dir.name

    env = json.loads((run.dir / "env.json").read_text())
    for field in ("host", "user", "uid", "gid", "groups", "python", "cwd"):
        assert field in env, field


def test_environment_records_supplementary_groups():
    """The fact that explained a whole day of failed ingests.

    The data share is mounted with a forced gid; a user missing that group cannot
    create an output directory, and nothing else in any log says so.
    """
    env = collect_environment()
    assert isinstance(env["groups"], list)
    assert sorted(os.getgroups()) == env["groups"]


def test_start_falls_back_to_null_log_when_directory_cannot_be_made(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        run = RunLog.start(log_dir=blocked / "runs", user="ZZ")
        assert isinstance(run, NullRunLog)
        assert run.dir is None
        # and it still behaves like a run log
        with run.step("something") as s:
            s.ok()
        assert run.counts()["ok"] == 1
    finally:
        blocked.chmod(0o700)


def test_selections_are_persisted_verbatim(tmp_path):
    """Without this, the only record of which DLC model was chosen is a screenshot."""
    run = RunLog.start(log_dir=tmp_path, user="ZZ")
    selections = [{
        "session_id": "sess9FUDDCP7",
        "project": "sc-lgn-actvis",
        "dlc_models": [[], ["NK; General_eye_fullsize-NK-2025-08-25; eye1_video"],
                       ["NK; General_eye_fullsize-NK-2025-08-25; eye1_video"]],
        "use_dlc_cropping": False,
    }]
    run.record_selections(selections)
    written = json.loads((run.dir / "selections.json").read_text())
    assert written == selections


def test_record_writes_arbitrary_artefacts(tmp_path):
    run = RunLog.start(log_dir=tmp_path, user="ZZ")
    run.record("preflight", {"mini2p1_eye_left": {"mkdir_ok": False}})
    assert json.loads((run.dir / "preflight.json").read_text())["mini2p1_eye_left"]["mkdir_ok"] is False


def test_record_survives_unserialisable_values(tmp_path):
    class Opaque:
        def __repr__(self):
            return "<opaque>"

    run = RunLog.start(log_dir=tmp_path, user="ZZ")
    run.record("weird", {"value": Opaque()})
    assert "opaque" in (run.dir / "weird.json").read_text()


# --------------------------------------------------------------------------------
# the console
# --------------------------------------------------------------------------------

def test_tee_captures_stdout_and_stderr(tmp_path):
    run = RunLog.start(log_dir=tmp_path, user="ZZ")
    with run.tee():
        print("a line that must survive the notebook")
        print("to stderr", file=sys.stderr)
    text = (run.dir / "console.log").read_text()
    assert "a line that must survive the notebook" in text
    assert "to stderr" in text


def test_tee_restores_streams_even_on_exception(tmp_path):
    run = RunLog.start(log_dir=tmp_path, user="ZZ")
    before_out, before_err = sys.stdout, sys.stderr
    with pytest.raises(RuntimeError):
        with run.tee():
            print("printed before the failure")
            raise RuntimeError("boom")
    assert sys.stdout is before_out and sys.stderr is before_err
    assert "printed before the failure" in (run.dir / "console.log").read_text()


def test_tee_does_not_swallow_output_from_the_real_stream(tmp_path, capsys):
    run = RunLog.start(log_dir=tmp_path, user="ZZ")
    with run.tee():
        print("still visible in the widget")
    assert "still visible in the widget" in capsys.readouterr().out


# --------------------------------------------------------------------------------
# the ledger
# --------------------------------------------------------------------------------

def test_three_outcomes_are_distinguished(tmp_path):
    run = RunLog.start(log_dir=tmp_path, user="ZZ")
    with run.step("worked") as s:
        s.ok("video.mp4")
    with run.step("nothing to do") as s:
        s.skipped('no video matching "face_video"')
    with run.step("broke") as s:
        s.failed(PermissionError(13, "Permission denied"))

    counts = run.counts()
    assert counts == {"ok": 1, "skipped": 1, "failed": 1, "unknown": 0}
    assert run.failures()[0]["detail"].startswith("PermissionError")
    assert 'face_video' in run.skips()[0]["detail"]


def test_failed_step_keeps_the_full_traceback(tmp_path):
    run = RunLog.start(log_dir=tmp_path, user="ZZ")
    try:
        raise ValueError("the interesting part")
    except ValueError as exc:
        with run.step("broke") as s:
            s.failed(exc)
    tb = run.failures()[0]["traceback"]
    assert "ValueError: the interesting part" in tb
    assert "test_ingest_log.py" in tb


def test_step_without_an_outcome_is_recorded_as_unknown(tmp_path):
    run = RunLog.start(log_dir=tmp_path, user="ZZ")
    with run.step("forgot to say"):
        pass
    assert run.counts()["unknown"] == 1


def test_steps_are_written_as_they_happen(tmp_path):
    """The ledger has to survive a kernel that dies mid-run."""
    run = RunLog.start(log_dir=tmp_path, user="ZZ")
    with run.step("first", key={"session_id": "sessA"}) as s:
        s.ok()
    lines = (run.dir / "steps.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["session_id"] == "sessA"


def test_keys_tag_steps_and_drive_per_session_status(tmp_path):
    run = RunLog.start(log_dir=tmp_path, user="ZZ")
    with run.step("ok step", key={"session_id": "sessA"}) as s:
        s.ok()
    with run.step("dlc", key={"session_id": "sessB", "camera": "mini2p1_eye_left"}) as s:
        s.failed(PermissionError(13, "Permission denied"))

    assert run.session_ok("sessA") is True
    assert run.session_ok("sessB") is False
    assert run.counts(session_id="sessB") == {"ok": 0, "skipped": 0, "failed": 1, "unknown": 0}
    assert run.failures(session_id="sessB")[0]["camera"] == "mini2p1_eye_left"


# --------------------------------------------------------------------------------
# what the user is told at the end
# --------------------------------------------------------------------------------

def test_summary_names_the_failures(tmp_path):
    run = RunLog.start(log_dir=tmp_path, user="ZZ")
    with run.step("DLC model Video 2 (NK; General_eye_fullsize…)",
                  key={"session_id": "sess9FUDDCP7"}) as s:
        s.failed(PermissionError(13, "Permission denied"))

    text = run.format_summary()
    assert "1 failed" in text
    assert "sess9FUDDCP7" in text
    assert "PermissionError" in text
    assert str(run.dir) in text


def test_summary_truncates_long_failure_lists(tmp_path):
    run = RunLog.start(log_dir=tmp_path, user="ZZ")
    for i in range(15):
        with run.step("step %d" % i, key={"session_id": "sessA"}) as s:
            s.failed(RuntimeError("no"))
    text = run.format_summary(max_listed=10)
    assert "and 5 more failed" in text


def test_finish_writes_summary_json(tmp_path):
    run = RunLog.start(log_dir=tmp_path, user="ZZ")
    with run.step("a") as s:
        s.ok()
    with run.step("b") as s:
        s.failed(RuntimeError("no"))
    summary = run.finish({"sessions_selected": 6})

    on_disk = json.loads((run.dir / "summary.json").read_text())
    assert on_disk == summary
    assert summary["counts"]["failed"] == 1
    assert summary["sessions_selected"] == 6
    # the traceback stays in steps.jsonl; the summary stays readable
    assert "traceback" not in summary["failures"][0]


# --------------------------------------------------------------------------------
# the null implementation
# --------------------------------------------------------------------------------

def test_null_run_log_supports_the_whole_api():
    run = NullRunLog()
    assert run.dir is None
    run.record_selections([{"a": 1}])
    run.record("x", {"y": 2})
    with run.tee():
        with run.step("something", key={"session_id": "sessA"}) as s:
            s.failed(RuntimeError("no"))
    assert run.counts()["failed"] == 1
    assert run.session_ok("sessA") is False
    assert "1 failed" in run.format_summary()
    assert run.finish()["run_id"] == "no-log"


def test_get_run_log_normalises_none():
    assert isinstance(get_run_log(None), NullRunLog)
    assert isinstance(get_run_log("not a run log"), NullRunLog)
    real = NullRunLog()
    assert get_run_log(real) is real


# --------------------------------------------------------------------------------
# writability probing
# --------------------------------------------------------------------------------

def test_probe_writable_on_a_writable_directory(tmp_path):
    result = probe_writable(tmp_path)
    assert result["exists"] and result["mkdir_ok"] is True
    assert list(tmp_path.iterdir()) == []  # the probe cleans up after itself


def test_probe_writable_detects_a_directory_it_cannot_write(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        result = probe_writable(ro)
        assert result["exists"] is True
        assert result["mkdir_ok"] is False
        assert "PermissionError" in result["mkdir_error"]
    finally:
        ro.chmod(0o700)


def test_probe_writable_on_a_missing_path(tmp_path):
    result = probe_writable(tmp_path / "nope")
    assert result["exists"] is False
    assert result["mkdir_ok"] is None


def test_probe_is_used_for_every_configured_root(monkeypatch, tmp_path):
    import datajoint as dj
    monkeypatch.setattr(dj, "config", {
        "database.host": "db", "database.user": "u",
        "custom": {"database.prefix": "roselab_", "exp_root_data_dir": [str(tmp_path)]},
    }, raising=False)
    env = collect_environment()
    roots = env["datajoint"]["roots"]
    assert roots and roots[0]["config_key"] == "exp_root_data_dir"
    assert roots[0]["mkdir_ok"] is True


def test_step_skipped_is_not_an_error_type():
    assert issubclass(StepSkipped, Exception)
    assert not issubclass(StepSkipped, (OSError, ValueError))
