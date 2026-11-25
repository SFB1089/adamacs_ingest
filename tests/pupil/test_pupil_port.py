"""Tests for the pupil image rim fit, glint and DISK port.

Three layers:
  * pure (always run): which code a parameter row selects (pupil_image_fit.fit_settings), and the glint code version;
  * database, read-only (run when the adamacs DataJoint config exists): every ported table exists with the declared
    heading, and two cheap make() calls (DISK frame quality) and the eye model recomputed from stored ellipses
    reproduce the stored rows bitwise;
  * video (ADAMACS_GOLDEN_VIDEO=1): full make() of rim-fit sets 10-12, CorrectedGlint sets 2-3 and a DISK
    head-fixed imputation, minutes each. Use tests/pupil/golden.py directly to run those in parallel.
Nothing writes: see tests/pupil/readonly_db.py.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))


def _helper(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", ROOT / "adamacs" / "helpers" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cv2 = pytest.importorskip("cv2")
pif = _helper("pupil_image_fit")


def _row(code_version, **v4):
    row = dict(pif.DEFAULTS, **pif.GATE_DEFAULTS, code_version=code_version)
    row.update({k: None for k in pif.V4_DEFAULTS})     # the v4 columns are NULL in a v3 row
    row.update(v4)
    return row


V4_COLS = dict(r_max_guide=1.8, guide_r_window_s=0.5, guide_r_engage=1.15, r_max_retry=3.0, weak_inlier_frac=0.7,
               weak_ref_min_inlier=0.8, weak_ref_window_s=0.5, weak_ref_dist_max=0.3, weak_ref_axis_min=0.8,
               weak_ref_axis_max=1.25)


def test_v3_row_runs_v3():
    params, v4, anchor = pif.fit_settings(_row(pif.CODE_VERSION))
    assert params == pif.DEFAULTS and v4 == {} and anchor is None


def test_v4_row_runs_v4_without_anchor():
    params, v4, anchor = pif.fit_settings(_row(pif.CODE_VERSION_V4, **V4_COLS))
    assert v4 == V4_COLS and anchor is None


def test_v41_row_adds_the_anchor():
    _, v4, anchor = pif.fit_settings(_row(pif.CODE_VERSION_V41, **V4_COLS))
    assert v4 == V4_COLS and anchor == pif.ANCHOR_DEFAULTS


def test_unknown_code_version_fails():
    with pytest.raises(ValueError):
        pif.fit_settings(_row("image_rim_v9"))


def test_glint_code_version():
    assert _helper("glint_detect").CODE_VERSION == "glint-1.3"


# ----------------------------------------------------------------------------- database, read-only
import readonly_db  # noqa: E402

needs_db = pytest.mark.skipif(not readonly_db.available(), reason="no adamacs DataJoint config")


def _script(*args, timeout=3600):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)


@needs_db
def test_tables_exist():
    r = _script(str(HERE / "check_tables.py"))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


@needs_db
def test_headings_match_except_known():
    """Only TorsionCalibManual differs: its definition says float, the live table has longblob (unchanged by the port;
    adamacs main and adamacs_ingest main carry the same definition)."""
    r = _script(str(HERE / "check_headings.py"))
    diffs = [ln for ln in r.stdout.splitlines() if ln.startswith(("DIFF", "PARSE-FAIL"))]
    assert [d.split("\t")[2] for d in diffs] == ["TorsionCalibManual"], "\n".join(diffs) + r.stderr[-3000:]


def _golden(case, tmp_path, timeout=3600):
    for code in ("port",):
        r = _script(str(HERE / "golden.py"), "run", "--code", code, "--case", case, "--out", str(tmp_path), timeout=timeout)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    tsv = tmp_path / "golden.tsv"
    r = _script(str(HERE / "golden.py"), "compare", "--out", str(tmp_path), "--tsv", str(tsv))
    assert r.returncode == 0, r.stderr[-3000:]
    bad = [ln for ln in tsv.read_text().splitlines()[1:] if ln.split("\t")[1] == "port_vs_stored" and ln.split("\t")[4] != "True"]
    return bad


@needs_db
@pytest.mark.parametrize("case", ["hffq", "pfq"])
def test_frame_quality_reproduces_stored(case, tmp_path):
    assert _golden(case, tmp_path) == []


@needs_db
def test_eye_model_from_stored_ellipses(tmp_path):
    """Bitwise only in the environment that wrote the rows (the fissure passes ran in `v1audit`, numpy 1.24.4)."""
    tsv = tmp_path / "eye_model.tsv"
    r = _script(str(HERE / "golden_eye_model.py"), "--tsv", str(tsv))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    bad = [ln for ln in tsv.read_text().splitlines()[1:] if ln.split("\t")[4] != "True"]
    assert bad == []


video = pytest.mark.skipif(os.environ.get("ADAMACS_GOLDEN_VIDEO") != "1", reason="set ADAMACS_GOLDEN_VIDEO=1 (minutes per case)")


@needs_db
@video
@pytest.mark.parametrize("case", ["rim12_9G2N0274", "rim12_9FXEU7TJ", "rim11_9FXEU7TJ", "rim10_9FXEU7TJ", "glint3_9G2N0274",
                                  "glint1_9FSEII8M", "glint2_9FSEII8M", "glint3_9FSEII8M"])
def test_make_reproduces_stored_per_frame_results(case, tmp_path):
    """Per-frame results (ellipse_dict, glint arrays, ImageFit fit fields) must be bitwise equal to the stored row.
    Run in the environment the rows were made in (`Cascade`: numpy 1.23.3, OpenCV 4.11.0).
    The eye-model fields of rim-fit rows were last written by a fissure second pass (recompute_from_stored with the
    final per-mouse prior), not by make(); they are covered by test_eye_model_from_stored_ellipses."""
    eye = {"theta", "phi", "scale", "cam_center", "eye_center", "e3d_cx", "e3d_cy", "e3d_r", "theta_offset", "phi_offset",
           "e3d_objective", "e3d_objective_single", "e3d_objective_far", "e3d_n_starts", "fissure_px", "fissure_source",
           "fissure_n_sessions", "n_frames_eye_model"}
    bad = [b for b in _golden(case, tmp_path, timeout=7200) if b.split("\t")[3] not in eye]
    assert bad == []
