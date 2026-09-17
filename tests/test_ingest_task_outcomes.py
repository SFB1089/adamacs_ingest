"""How _run_ingestion_task classifies what happened, and that the classification
reaches the run log.

Skipped when the GUI module cannot be imported (it pulls in the whole DataJoint
pipeline), so this file is safe to collect on a machine without a database.
"""

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

from adamacs.ingest_log import RunLog, StepSkipped  # noqa: E402


@pytest.fixture
def run(tmp_path):
    r = RunLog.start(log_dir=tmp_path, user="ZZ")
    with ai.ingest_run(r):
        yield r


def test_success_is_recorded_as_ok(run):
    assert ai._run_ingestion_task(lambda: "video.mp4", "a step") is True
    assert run.counts() == {"ok": 1, "skipped": 0, "failed": 0, "unknown": 0}
    assert run.steps[0]["detail"] == "video.mp4"


def test_skip_is_recorded_as_skip_not_as_success(run):
    """The regression this exists for.

    _ingest_dlc_model used to print one line and return None when the model name did
    not resolve to a video; _run_ingestion_task then logged 'DONE.' A skip and a
    success were indistinguishable in the only record that existed.
    """
    def nothing_to_do():
        raise StepSkipped('no video matching "face_video"')

    assert ai._run_ingestion_task(nothing_to_do, "DLC model Video 2") is False
    assert run.counts()["skipped"] == 1
    assert run.counts()["ok"] == 0
    assert "face_video" in run.skips()[0]["detail"]


def test_failure_is_recorded_with_its_traceback(run):
    def boom():
        raise PermissionError(13, "Permission denied")

    assert ai._run_ingestion_task(boom, "DLC model Video 2") is False
    failure = run.failures()[0]
    assert failure["detail"].startswith("PermissionError")
    assert "PermissionError" in failure["traceback"]
    assert failure["description"] == "DLC model Video 2"


def test_kwargs_are_forwarded(run):
    seen = {}

    def step(**kwargs):
        seen.update(kwargs)
        return True

    ai._run_ingestion_task(step, "a step", alpha=1, beta="two")
    assert seen == {"alpha": 1, "beta": "two"}


def test_steps_are_tagged_with_the_surrounding_context(run):
    with ai.ingest_context(session_id="sessA"):
        ai._run_ingestion_task(lambda: True, "session level")
        with ai.ingest_context(scan_id="scanA", camera="mini2p1_eye_left"):
            ai._run_ingestion_task(lambda: True, "camera level")

    outer, inner = run.steps
    assert outer["session_id"] == "sessA" and "scan_id" not in outer
    assert inner["session_id"] == "sessA"
    assert inner["scan_id"] == "scanA"
    assert inner["camera"] == "mini2p1_eye_left"


def test_context_is_restored_after_the_block(run):
    with ai.ingest_context(session_id="sessA"):
        pass
    ai._run_ingestion_task(lambda: True, "outside")
    assert "session_id" not in run.steps[0]


def test_session_status_comes_from_the_steps(run):
    """A session whose DLC steps all failed must not count as a success.

    Before, success was 'no exception escaped _process_session' -- but every step runs
    under _run_ingestion_task, which catches, so nothing ever escaped and a run with
    twelve dead DLC steps reported 'finished successfully for all 6 sessions'.
    """
    def boom():
        raise PermissionError(13, "Permission denied")

    with ai.ingest_context(session_id="sessGood"):
        ai._run_ingestion_task(lambda: True, "metadata")
    with ai.ingest_context(session_id="sessBad"):
        ai._run_ingestion_task(lambda: True, "metadata")
        for camera in ("mini2p1_eye_left", "mini2p1_eye_right"):
            with ai.ingest_context(camera=camera):
                ai._run_ingestion_task(boom, "DLC model")

    assert run.session_ok("sessGood") is True
    assert run.session_ok("sessBad") is False
    assert run.counts()["failed"] == 2
    assert "sessBad" in run.format_summary()


def test_run_log_is_optional(tmp_path):
    """Called outside a run (other notebooks do this), nothing breaks."""
    assert ai._run_ingestion_task(lambda: True, "no active run") is True
    assert isinstance(ai.current_run_log().dir, type(None))


def test_ingest_dlc_model_skips_when_no_video_matches(monkeypatch, tmp_path):
    """The real function, on a real directory with no matching video."""
    class _Restricted:
        def fetch1(self, _field):
            return str(tmp_path)

    class _ScanPath:
        def __call__(self):
            return self

        def __and__(self, _key):
            return _Restricted()

    monkeypatch.setattr(ai.scan, "ScanPath", _ScanPath(), raising=False)
    (tmp_path / "scanX_mini2p1_top_video_2025.mp4").write_text("")  # a top video only

    with pytest.raises(StepSkipped) as excinfo:
        ai._ingest_dlc_model(
            scan_key={"session_id": "sessX", "scan_id": "scanX"},
            model_name="YH; TrainingBox2_face-YH-2025-06-10; face_video",
            camera="mini2p1_eye_left",
            aux_setup_typestr="mini2p1_openfield",
        )
    message = str(excinfo.value)
    assert "face_video" in message
    assert str(tmp_path) in message
