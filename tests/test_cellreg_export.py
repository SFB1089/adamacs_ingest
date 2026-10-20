"""Tests for building CellReg inputs from the database.

These exercise the writer without a database: a mask row is a plain dict, so every guard
and every byte of the output can be checked offline. The one thing they cannot check is
that the real curation's masks round-trip -- that was verified separately against the
on-disk Fall.mat for all 11 gratings scans of site sess9FPXNI8U (13,532 ROIs, identical
pixel sets and lam values throughout).
"""
from __future__ import annotations

import pytest

# CI installs pytest only, so this module is skipped there rather than failing
# collection and taking the whole run down with it.
np = pytest.importorskip("numpy", reason="needs numpy; CI installs pytest only")
scipy_io = pytest.importorskip("scipy.io", reason="needs scipy; CI installs pytest only")
import scipy.io  # noqa: E402  -- guarded above

from adamacs.ingest import cellreg_export as cx


def _masks(n=3, fov=32, seed=0):
    """n synthetic mask rows shaped like imaging.Segmentation.Mask fetch output."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        k = int(rng.integers(5, 12))
        ys = rng.integers(0, fov, k).astype(np.int64)
        xs = rng.integers(0, fov, k).astype(np.int64)
        out.append({"mask": i, "mask_ypix": ys, "mask_xpix": xs,
                    "mask_weights": rng.random(k)})
    return out


# ---------------------------------------------------------------- the payload

def test_stat_is_a_1xn_cell_array_of_structs():
    """The driver does size(stat, 2) and stat{k}.ypix, so shape and fields are load-bearing."""
    p = cx.fall_payload(_masks(4), 32, 32)
    assert p["stat"].shape == (1, 4)
    for i in range(4):
        assert set(p["stat"][0, i]) == {"xpix", "ypix", "lam"}


def test_pixels_survive_a_savemat_roundtrip(tmp_path):
    masks = _masks(5)
    dest = tmp_path / "Fall_01_scanTEST.mat"
    cx.write_fall_mat(dest, masks, 32, 32)
    back = scipy.io.loadmat(str(dest), struct_as_record=False, squeeze_me=True)
    assert np.size(back["stat"]) == 5
    for i, m in enumerate(masks):
        s = back["stat"][i]
        assert set(zip(np.asarray(s.ypix).ravel().tolist(),
                       np.asarray(s.xpix).ravel().tolist())) == \
               set(zip(m["mask_ypix"].tolist(), m["mask_xpix"].tolist()))
        assert np.allclose(np.sort(np.asarray(s.lam).ravel()),
                           np.sort(m["mask_weights"]))


def test_ops_carries_the_fov_the_driver_reads(tmp_path):
    dest = tmp_path / "Fall_01_scanTEST.mat"
    cx.write_fall_mat(dest, _masks(2), 64, 48)
    back = scipy.io.loadmat(str(dest), struct_as_record=False, squeeze_me=True)
    assert int(back["ops"].Ly) == 64 and int(back["ops"].Lx) == 48


def test_mask_order_is_preserved_because_it_becomes_the_cellreg_index(tmp_path):
    """stat's order is the 1-based index CellReg writes into cell_to_index_map."""
    masks = _masks(6)
    dest = tmp_path / "Fall_01_scanTEST.mat"
    cx.write_fall_mat(dest, masks, 32, 32)
    back = scipy.io.loadmat(str(dest), struct_as_record=False, squeeze_me=True)
    first = [int(np.asarray(back["stat"][i].ypix).ravel()[0]) for i in range(6)]
    assert first == [int(m["mask_ypix"][0]) for m in masks]


# ---------------------------------------------------------------- the guards

def test_refuses_to_write_inside_a_suite2p_tree(tmp_path):
    bad = tmp_path / "scanX" / "suite2p" / "plane0" / "Fall_01_scanX.mat"
    with pytest.raises(ValueError, match="suite2p"):
        cx.write_fall_mat(bad, _masks(2), 32, 32)


def test_refuses_to_write_a_file_called_fall_mat(tmp_path):
    with pytest.raises(ValueError, match="Fall.mat"):
        cx.write_fall_mat(tmp_path / "Fall.mat", _masks(2), 32, 32)


def test_refuses_to_overwrite_without_being_told(tmp_path):
    dest = tmp_path / "Fall_01_scanTEST.mat"
    cx.write_fall_mat(dest, _masks(2), 32, 32)
    with pytest.raises(FileExistsError):
        cx.write_fall_mat(dest, _masks(2), 32, 32)
    cx.write_fall_mat(dest, _masks(2), 32, 32, allow_overwrite=True)


def test_rejects_pixels_outside_the_fov(tmp_path):
    m = [{"mask": 0, "mask_ypix": np.array([5, 99]), "mask_xpix": np.array([5, 6]),
          "mask_weights": np.array([1.0, 1.0])}]
    with pytest.raises(ValueError, match="outside"):
        cx.write_fall_mat(tmp_path / "Fall_01_scanTEST.mat", m, 32, 32)


def test_rejects_inconsistent_pixel_arrays(tmp_path):
    m = [{"mask": 7, "mask_ypix": np.array([1, 2]), "mask_xpix": np.array([1]),
          "mask_weights": np.array([1.0, 1.0])}]
    with pytest.raises(ValueError, match="inconsistent"):
        cx.write_fall_mat(tmp_path / "Fall_01_scanTEST.mat", m, 32, 32)


def test_refuses_an_empty_mask_set():
    with pytest.raises(ValueError, match="empty"):
        cx.fall_payload([], 32, 32)


# ---------------------------------------------------------------- settings.mat

def test_settings_has_exactly_the_fields_the_driver_reads():
    s = cx.settings_payload({})
    assert set(s) == set(cx.SETTINGS_FIELDS)
    assert len(cx.SETTINGS_FIELDS) == 16


def test_settings_rejects_derived_quantities_posing_as_parameters():
    """number_of_bins and initial_threshold are OUTPUTS of CellReg, not settings."""
    for bad in ("number_of_bins", "n_bins", "initial_threshold"):
        with pytest.raises(ValueError, match="DERIVED"):
            cx.settings_payload({bad: 100})


def test_settings_rejects_fields_the_driver_never_reads():
    with pytest.raises(ValueError, match="not settings"):
        cx.settings_payload({"nonsense_parameter": 1})


def test_settings_overrides_defaults(tmp_path):
    out = cx.write_settings_mat(tmp_path / "settings.mat", {"maximal_distance": 10.0})
    back = scipy.io.loadmat(out["settings_file"], struct_as_record=False, squeeze_me=True)
    assert float(back["maximal_distance"]) == 10.0
    assert str(back["input_format"]) == "Suite2p"


# ---------------------------------------------------------------- naming & load

def test_fall_name_is_one_based_to_match_the_log():
    assert cx.fall_name(0, "scan9FOW97IH") == "Fall_01_scan9FOW97IH.mat"
    assert cx.fall_name(10, "scanX") == "Fall_11_scanX.mat"


def test_export_dir_is_never_inside_a_suite2p_tree():
    d = cx.export_dir_for({"subject": "ROS-1881", "same_site_id": "sess9FPXNI8U"},
                          label="Gratings", task_id=3)
    assert "suite2p" not in {p.lower() for p in d.parts}
    assert d.name == "sess9FPXNI8U_Gratings_task03"


def test_load_mode_refuses_when_the_manifest_and_directory_disagree(tmp_path):
    cols = [{"column_index": 0, "session_id": "s1", "scan_id": "scanA",
             "paramset_idx": 10, "curation_id": 2},
            {"column_index": 1, "session_id": "s2", "scan_id": "scanB",
             "paramset_idx": 10, "curation_id": 2}]
    cx.write_fall_mat(tmp_path / "Fall_01_scanA.mat", _masks(2), 32, 32)
    with pytest.raises(FileNotFoundError, match="Fall_02_scanB.mat"):
        cx.inspect_existing(tmp_path, cols)


def test_load_mode_records_checksums_of_what_it_found(tmp_path):
    cols = [{"column_index": 0, "session_id": "s1", "scan_id": "scanA",
             "paramset_idx": 10, "curation_id": 2}]
    cx.write_fall_mat(tmp_path / "Fall_01_scanA.mat", _masks(3), 32, 32)
    got = cx.inspect_existing(tmp_path, cols)
    assert len(got["files"]) == 1
    assert got["files"][0]["n_masks"] == 3
    assert len(got["files"][0]["file_sha256"]) == 64


def test_load_mode_requires_the_directory_to_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a directory"):
        cx.inspect_existing(tmp_path / "nope", [])


# ------------------------------------------------- the enumerated-value guard
# These exist because the first real run silently fell through Stage 5: the driver compares
# these fields with strcmp, so a wrong literal raises nothing in MATLAB -- the branch never
# fires, 1985 cells are found, and then it dies on an unassigned variable with no result
# file and a one-line error log.

def test_defaults_are_values_the_driver_actually_accepts():
    s = cx.settings_payload({})
    for field, allowed in cx.SETTINGS_CHOICES.items():
        assert s[field] in allowed, f"{field}={s[field]!r} not in {allowed}"


def test_the_log_file_prose_is_rejected_outright():
    """logFile_*.txt says 'Probabilistic modeling' and 'Spatial correlations' (plural)."""
    with pytest.raises(ValueError, match="silently ignore"):
        cx.settings_payload({"registration_approach": "Probabilistic modeling"})
    with pytest.raises(ValueError, match="silently ignore"):
        cx.settings_payload({"model_type": "Spatial correlations"})


def test_from_log_value_maps_prose_onto_driver_literals():
    assert cx.from_log_value("registration_approach", "Probabilistic modeling") == "Probabilistic"
    assert cx.from_log_value("model_type", "Spatial correlations") == "Spatial correlation"
    assert cx.from_log_value("model_type", "Centroid distances") == "Centroid distance"
    # already-correct values pass straight through
    assert cx.from_log_value("model_type", "Spatial correlation") == "Spatial correlation"


def test_from_log_value_refuses_what_it_cannot_map():
    with pytest.raises(ValueError, match="cannot map"):
        cx.from_log_value("model_type", "Probabilistic")


def test_a_parsed_log_can_be_turned_into_valid_settings():
    """The round trip an importer will actually perform."""
    parsed = {"registration_approach": "Probabilistic modeling",
              "model_type": "Spatial correlations",
              "alignment_type": "Translations and Rotations"}
    fixed = {k: cx.from_log_value(k, v) for k, v in parsed.items()}
    s = cx.settings_payload(fixed)
    assert s["registration_approach"] == "Probabilistic"
    assert s["model_type"] == "Spatial correlation"


def test_every_choice_field_is_also_a_settings_field():
    assert set(cx.SETTINGS_CHOICES) <= set(cx.SETTINGS_FIELDS)
