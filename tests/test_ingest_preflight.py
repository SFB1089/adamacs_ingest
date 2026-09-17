"""Tests for adamacs.ingest_preflight.

Each check here corresponds to a failure that actually happened and was only
diagnosed afterwards.
"""

import os

import pytest

from adamacs.ingest_preflight import (
    ERROR,
    OK,
    UNKNOWN,
    WARNING,
    PreflightReport,
    run_preflight,
)


EYE_MODEL = "NK; General_eye_fullsize-NK-2025-08-25; eye1_video"
TOPCAM_MODEL = "JJ; Topcam_mini2p2_Effnet-JJ-2026-02-23; Topcam"
CAMERAS = ["mini2p1_top", "mini2p1_eye_left", "mini2p1_eye_right"]
SESSION_DIR = "NK_ROS-9001_2099-01-01_scanAAA0001_sessAAA0001"


def _search_name(model_name, video_idx):
    if video_idx == 2 and "eye1" in model_name:
        return model_name.replace("eye1", "eye2")
    return model_name


def _candidates(directory, search_name):
    import pathlib
    try:
        search_str = search_name.split(";")[2].replace(" ", "")
    except IndexError:
        search_str = "top"
    return search_str, list(pathlib.Path(directory).glob("*%s*.mp4*" % search_str))


@pytest.fixture
def data_root(tmp_path):
    """A session folder with the videos a mini2p1 recording really has."""
    d = tmp_path / SESSION_DIR
    d.mkdir()
    for name in ("scanAAA0001_mini2p1_top_video_2099.mp4",
                 "scanAAA0001_headcam_mini2p1_left_eye1_video_2099.mp4",
                 "scanAAA0001_headcam_mini2p1_right_eye2_video_2099.mp4"):
        (d / name).write_text("")
    return tmp_path


def _selection(dlc_models):
    return [{"session_id": "sessAAA0001", "path": SESSION_DIR, "dlc_models": dlc_models}]


def _run(selections, data_root, **kw):
    kw.setdefault("cameras", [CAMERAS])
    kw.setdefault("candidates_fn", _candidates)
    kw.setdefault("search_name_fn", _search_name)
    return run_preflight(selections, data_root, **kw)


# --------------------------------------------------------------------------------

def test_a_healthy_selection_passes(data_root):
    report = _run(_selection([[], [EYE_MODEL], [EYE_MODEL]]), data_root)
    assert report.ok
    assert report.counts[ERROR] == 0
    assert "nothing to report" in report.format()


def test_both_eye_cameras_resolve_their_own_video(data_root):
    report = _run(_selection([[], [EYE_MODEL], [EYE_MODEL]]), data_root)
    resolved = {f["camera"]: f["detail"] for f in report.of(OK)
                if f["check"] == "video resolves"}
    assert "eye1_video" in resolved["mini2p1_eye_left"]
    assert "eye2_video" in resolved["mini2p1_eye_right"]


def test_a_model_whose_name_matches_no_video_is_an_error(data_root):
    """The `; Topcam` case: capital T, no `_video`, so the glob matches nothing in a
    mini2p1 folder. Before, this was a silent skip discovered days later."""
    report = _run(_selection([[TOPCAM_MODEL], [], []]), data_root)
    assert not report.ok
    bad = report.of(ERROR)[0]
    assert bad["check"] == "no video matches this model"
    assert "*Topcam*.mp4*" in bad["detail"]
    assert bad["camera"] == "mini2p1_top"
    assert "Topcam" in report.format()


def test_an_unwritable_session_directory_is_an_error(data_root):
    """The 2026-09-16 cause. os.access is not enough on the CIFS mounts, so the
    check does a real mkdir."""
    (data_root / SESSION_DIR).chmod(0o500)
    try:
        report = _run(_selection([[], [EYE_MODEL], []]), data_root)
        assert not report.ok
        errors = [f for f in report.of(ERROR)
                  if f["check"] == "cannot create files in the session directory"]
        assert errors and "PermissionError" in errors[0]["detail"]
        # and it does not then produce noise about videos it could not reach
        assert not [f for f in report if f["check"] == "no video matches this model"]
    finally:
        (data_root / SESSION_DIR).chmod(0o700)


def test_a_missing_session_directory_is_an_error(tmp_path):
    report = _run(_selection([[], [EYE_MODEL], []]), tmp_path)
    assert not report.ok
    assert report.of(ERROR)[0]["check"] == "session directory is missing"


def test_several_matching_videos_is_a_warning_not_an_error(data_root):
    """Deinterlaced files sit next to their originals, and the ingest takes whichever
    the filesystem lists first."""
    (data_root / SESSION_DIR /
     "scanAAA0001_headcam_mini2p1_left_eye1_video_2099_deinterlaced.mp4").write_text("")
    report = _run(_selection([[], [EYE_MODEL], []]), data_root)
    assert report.ok                      # ambiguity does not block the run
    warn = [f for f in report.of(WARNING)
            if f["check"] == "several videos match this model"]
    assert warn and "filesystem lists first" in warn[0]["detail"]


def test_an_unregistered_model_is_an_error(data_root):
    class _Empty:
        def __and__(self, _restriction):
            return []

    report = _run(_selection([[], [EYE_MODEL], []]), data_root, model_table=_Empty())
    assert not report.ok
    assert any(f["check"] == "model is not in model.Model" for f in report.of(ERROR))


def test_a_registered_model_passes(data_root):
    class _Present:
        def __and__(self, _restriction):
            return [{"model_name": EYE_MODEL}]

    report = _run(_selection([[], [EYE_MODEL], []]), data_root, model_table=_Present())
    assert report.ok
    assert any(f["check"] == "model is registered" for f in report.of(OK))


def test_a_model_table_that_raises_is_unknown_not_fatal(data_root):
    class _Broken:
        def __and__(self, _restriction):
            raise RuntimeError("connection lost")

    report = _run(_selection([[], [EYE_MODEL], []]), data_root, model_table=_Broken())
    assert report.ok                      # unknown is not an error
    assert report.counts[UNKNOWN] == 1


def test_selecting_no_model_at_all_is_a_warning(data_root):
    report = _run(_selection([[], [], []]), data_root)
    assert report.ok
    assert any(f["check"] == "no DLC model is selected for any camera"
               for f in report.of(WARNING))


def test_an_unwritable_processed_dir_is_an_error(data_root, tmp_path):
    blocked = tmp_path / "processed"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        report = _run(_selection([[], [EYE_MODEL], []]), data_root,
                      get_processed_dir=lambda: str(blocked))
        assert not report.ok
        assert any(f["check"] == "dlc_processed_data_dir is not writable"
                   for f in report.of(ERROR))
    finally:
        blocked.chmod(0o700)


def test_an_unset_processed_dir_is_fine(data_root):
    report = _run(_selection([[], [EYE_MODEL], []]), data_root,
                  get_processed_dir=lambda: None)
    assert report.ok
    assert any(f["check"] == "dlc_processed_data_dir is unset" for f in report.of(OK))


def test_the_process_identity_is_recorded(data_root):
    """Because the answer, last time, was a missing supplementary group."""
    report = _run(_selection([[], [EYE_MODEL], []]), data_root)
    identity = [f for f in report if f["check"] == "process identity"]
    assert identity and str(sorted(os.getgroups())) in identity[0]["detail"]


def test_each_model_is_checked_against_the_table_only_once(data_root):
    calls = []

    class _Counting:
        def __and__(self, restriction):
            calls.append(restriction)
            return [{"model_name": EYE_MODEL}]

    _run(_selection([[], [EYE_MODEL], [EYE_MODEL]]), data_root, model_table=_Counting())
    assert len(calls) == 1


def test_the_report_serialises(data_root):
    report = _run(_selection([[TOPCAM_MODEL], [], []]), data_root)
    payload = report.as_json()
    assert payload["ok"] is False
    assert payload["counts"][ERROR] == 1
    assert isinstance(payload["findings"], list)
    import json
    json.loads(json.dumps(payload))      # must survive the run log's writer


def test_report_truncates_long_lists(data_root):
    report = PreflightReport()
    from adamacs.ingest_preflight import Finding
    for i in range(30):
        report.append(Finding(ERROR, "check %d" % i, session="sessA"))
    assert "and 10 more error" in report.format(max_listed=20)


def test_the_default_video_resolution_comes_from_the_ingest_module():
    """The preflight must use the ingest's own rule, or the two drift apart."""
    try:
        # Not importorskip: without datajoint installed the conftest stub makes this
        # raise AttributeError rather than ImportError, which importorskip re-raises.
        import adamacs.helpers.adamacs_ingest_v2 as ai
    except Exception as exc:
        pytest.skip("requires an importable DataJoint pipeline (%s)" % exc)
    assert ai.dlc_search_string("A; B; eye1_video") == "eye1_video"
    assert ai.dlc_search_string("no_semicolons_here") == "top"
    assert ai.dlc_search_name("A; B; eye1_video", 2) == "A; B; eye2_video"
    assert ai.dlc_search_name("A; B; eye1_video", 1) == "A; B; eye1_video"
