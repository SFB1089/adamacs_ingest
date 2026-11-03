"""Tests for the samesite re-key (ADR-R8).

Only the parts that can run without a live schema are unit-tested here. The membership
rule itself needs ``session.SessionSameSite``, so it is exercised against the live database
and the evidence recorded in the PR; see DESIGN_NOTE_samesite_key.md section 8.
"""
import pytest

cellreg_export = pytest.importorskip("adamacs.ingest.cellreg_export")


def test_export_dir_keeps_its_layout_when_subject_is_supplied():
    """The path must not change: run 2's directory is on disk and recorded in the export."""
    out = cellreg_export.export_dir_for(
        {"same_site_id": "sess9FUCURTR", "subject": "ROS-2101"},
        label="FOV466um", task_id=0)
    assert str(out) == ("/datajoint-data/data/ROS-2101/cellreg_export/"
                        "sess9FUCURTR_FOV466um_task00")


def test_export_dir_without_subject_says_what_to_do():
    """subject left the site key (ADR-R8); the error must name the fix, not say KeyError."""
    with pytest.raises(KeyError) as e:
        cellreg_export.export_dir_for({"same_site_id": "sess9FUCURTR"}, task_id=0)
    msg = str(e.value)
    assert "ADR-R8" in msg and "session.Session" in msg


def test_export_dir_is_never_inside_a_suite2p_tree():
    out = cellreg_export.export_dir_for(
        {"same_site_id": "sess9FUCURTR", "subject": "ROS-2101"}, task_id=3)
    assert "suite2p" not in str(out) and str(out).endswith("_task03")


cellreg_register = pytest.importorskip("adamacs.ingest.cellreg_register")


def test_manifest_note_does_not_need_subject():
    """ADR-R8 regression: the note used to index task_row['subject'], which raised KeyError
    on every task-backed registration -- the declared path. Caught in review, not by a test,
    which is why this test exists."""
    note = cellreg_register.manifest_note(
        {"cellreg_task_id": 0, "same_site_id": "sess9FUCURTR"})
    assert note == "columns declared by CellRegTask 0 of site sess9FUCURTR"


def test_manifest_note_fits_the_column():
    note = cellreg_register.manifest_note(
        {"cellreg_task_id": 99, "same_site_id": "s" * 12})
    assert len(note) <= 400
