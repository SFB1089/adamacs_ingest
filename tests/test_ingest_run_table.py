"""Tests for adamacs.schemas.ingest.IngestRun.

The inserts run inside a transaction that is always rolled back, so this exercises
the real schema against the real database and leaves nothing behind.
"""

import json

import pytest

try:
    import datajoint as dj
    from adamacs.schemas import ingest as ing
    from adamacs.ingest_log import RunLog, StepSkipped
except Exception as exc:  # pragma: no cover - environment-dependent
    ing = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

pytestmark = pytest.mark.skipif(
    ing is None,
    reason="requires a reachable DataJoint database (%s)" % _IMPORT_ERROR,
)


@pytest.fixture
def rollback():
    """Everything written inside the test is discarded.

    The rollback is unconditional. Checking ``conn.in_transaction`` first looks
    tidier but leaves rows behind whenever a test fails in a way that clears the
    flag, which is exactly when you least want test data in a shared database.
    """
    conn = dj.conn()
    conn.start_transaction()
    try:
        yield conn
    finally:
        try:
            conn.cancel_transaction()
        except Exception:
            conn.query("ROLLBACK")


@pytest.fixture
def failed_run(tmp_path):
    """A run shaped like 2026-09-16: two sessions, one of them entirely broken."""
    run = RunLog.start(log_dir=tmp_path, user="NK")
    run.record_selections([
        {"session_id": "sessTEST001", "project": "sc-lgn-actvis",
         "dlc_models": [[], ["NK; General_eye_fullsize-NK-2025-08-25; eye1_video"], []]},
        {"session_id": "sessTEST002", "project": "sc-lgn-actvis",
         "dlc_models": [[], ["NK; General_eye_fullsize-NK-2025-08-25; eye1_video"], []]},
    ])
    with run.step("metadata", key={"session_id": "sessTEST001"}) as s:
        s.ok()
    for camera in ("mini2p1_eye_left", "mini2p1_eye_right"):
        with run.step("DLC model Video 2", key={
                "session_id": "sessTEST002", "scan_id": "scanTEST002",
                "camera": camera,
                "model_name": "NK; General_eye_fullsize-NK-2025-08-25; eye1_video"}) as s:
            s.failed(PermissionError(13, "Permission denied"))
    with run.step("DLC model Video 3", key={"session_id": "sessTEST002"}) as s:
        s.skipped('no video matching "face_video"')

    summary = run.finish({
        "sessions_selected": 2,
        "sessions_successful": 1,
        "sessions_failed": 1,
        "sessions_with_failures": ["sessTEST002"],
    })
    return run, summary


def test_schema_exists_with_its_parts():
    assert ing.schema.database.endswith("ingest")
    assert "IngestRun" in dir(ing)
    assert hasattr(ing.IngestRun, "SessionOutcome")
    assert hasattr(ing.IngestRun, "Failure")


def test_ingest_run_has_no_foreign_keys():
    """Deliberate: the runs worth investigating are the ones whose sessions were
    never created or were later deleted. A foreign key would lose exactly those."""
    assert ing.IngestRun.parents() == []
    assert ing.IngestRun.SessionOutcome.parents() == [ing.IngestRun.full_table_name]


def test_record_run_writes_the_master_row(rollback, failed_run):
    run, summary = failed_run
    assert ing.record_run(run, summary, verbose=False) is True

    row = (ing.IngestRun & f'run_id = "{run.run_id}"').fetch1()
    assert row["n_steps_failed"] == 2
    assert row["n_steps_skipped"] == 1
    assert row["n_sessions"] == 2
    assert row["n_sessions_failed"] == 1
    assert row["log_dir"] == str(run.dir)
    assert row["host"] and row["os_user"]


def test_recorded_environment_keeps_the_group_membership(rollback, failed_run):
    """The field that would have named the real-world failure in one query."""
    run, summary = failed_run
    ing.record_run(run, summary, verbose=False)
    env = (ing.IngestRun & f'run_id = "{run.run_id}"').fetch1("environment")
    assert isinstance(env, dict)
    assert isinstance(env["groups"], list) and env["groups"]


def test_recorded_selections_survive_the_round_trip(rollback, failed_run):
    run, summary = failed_run
    ing.record_run(run, summary, verbose=False)
    selections = (ing.IngestRun & f'run_id = "{run.run_id}"').fetch1("selections")
    assert [s["session_id"] for s in selections] == ["sessTEST001", "sessTEST002"]
    assert selections[1]["dlc_models"][1] == [
        "NK; General_eye_fullsize-NK-2025-08-25; eye1_video"]


def test_per_session_outcomes(rollback, failed_run):
    run, summary = failed_run
    ing.record_run(run, summary, verbose=False)
    rows = {r["session_id"]: r for r in
            (ing.IngestRun.SessionOutcome & f'run_id = "{run.run_id}"').fetch(as_dict=True)}
    assert rows["sessTEST001"]["outcome"] == "ok"
    assert rows["sessTEST002"]["outcome"] == "failed"
    assert rows["sessTEST002"]["session_steps_failed"] == 2
    assert rows["sessTEST002"]["session_steps_skipped"] == 1
    assert "PermissionError" in rows["sessTEST002"]["first_error"]


def test_failures_are_itemised_with_their_camera_and_model(rollback, failed_run):
    run, summary = failed_run
    ing.record_run(run, summary, verbose=False)
    rows = (ing.IngestRun.Failure & f'run_id = "{run.run_id}"').fetch(as_dict=True)
    failed = [r for r in rows if r["outcome"] == "failed"]
    skipped = [r for r in rows if r["outcome"] == "skipped"]
    assert {r["camera"] for r in failed} == {"mini2p1_eye_left", "mini2p1_eye_right"}
    assert all("General_eye_fullsize" in r["model_name"] for r in failed)
    assert len(skipped) == 1 and "face_video" in skipped[0]["detail"]


def test_the_two_tables_can_actually_be_joined():
    """Regression: the part table first shared n_steps_* with its master, and
    DataJoint refuses to join query expressions on a shared dependent attribute."""
    shared = (set(ing.IngestRun.heading.secondary_attributes)
              & set(ing.IngestRun.SessionOutcome.heading.secondary_attributes))
    assert shared == set(), shared


def test_runs_touching_finds_the_run_for_a_session(rollback, failed_run):
    run, summary = failed_run
    ing.record_run(run, summary, verbose=False)
    found = ing.runs_touching("sessTEST002").fetch(as_dict=True)
    assert [r["run_id"] for r in found] == [run.run_id]
    assert found[0]["outcome"] == "failed"


def test_record_run_is_idempotent(rollback, failed_run):
    run, summary = failed_run
    assert ing.record_run(run, summary, verbose=False) is True
    assert ing.record_run(run, summary, verbose=False) is True
    assert len(ing.IngestRun & f'run_id = "{run.run_id}"') == 1


def test_record_run_never_raises(monkeypatch, failed_run):
    run, summary = failed_run

    def explode(*a, **k):
        raise RuntimeError("database on fire")

    monkeypatch.setattr(ing.IngestRun, "insert1", explode)
    assert ing.record_run(run, summary, verbose=False) is False


def test_record_run_ignores_an_empty_null_log():
    from adamacs.ingest_log import NullRunLog
    assert ing.record_run(NullRunLog(), {}, verbose=False) is False


def test_long_values_are_truncated_not_rejected(rollback, tmp_path):
    run = RunLog.start(log_dir=tmp_path, user="NK")
    with run.step("x" * 500, key={"session_id": "s" * 100}) as s:
        s.failed(RuntimeError("y" * 5000))
    summary = run.finish({"sessions_selected": 1, "sessions_failed": 1,
                          "sessions_with_failures": ["s" * 100]})
    assert ing.record_run(run, summary, verbose=False) is True
    row = (ing.IngestRun.Failure & f'run_id = "{run.run_id}"').fetch1()
    assert len(row["description"]) <= 255 and len(row["detail"]) <= 1000
