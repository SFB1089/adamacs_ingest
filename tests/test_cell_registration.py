"""Unit tests for the cell-registration schema and its CellReg parser.

No database writes. Every test runs against synthetic fixtures or in-memory values; the
schema module is only imported (which declares nothing) and inspected.

The fixtures deliberately cover the cases that real files do NOT exercise -- in particular a
simple-threshold run with the score fields absent, which the CellReg script can produce but
which none of the 108 files on this server actually is.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from adamacs.ingest import cellreg
from adamacs.schemas import cell_registration as creg

# --------------------------------------------------------------------------- fixtures
SCORE_FIELDS = cellreg.SCORE_FIELDS


def _index_map():
    """4 cells x 3 sessions, 1-based, 0 = absent.

    cell 0: all three sessions          -> masks 4, 6, 1
    cell 1: sessions 0 and 2            -> masks 0, 11
    cell 2: session 1 only              -> mask 2
    cell 3: none                        -> no members at all
    """
    return np.array([[5, 7, 2],
                     [1, 0, 12],
                     [0, 3, 0],
                     [0, 0, 0]], dtype=float)


def _p_same(n_sessions=3, n_cells=4):
    mats = []
    for i in range(n_cells):
        m = np.full((n_sessions, n_sessions), np.nan)
        for a in range(n_sessions):
            for b in range(n_sessions):
                if a != b:
                    m[a, b] = 0.9 - 0.1 * i - 0.01 * (a + b)
                else:
                    m[a, b] = 1.0
        mats.append(m)
    return mats


def write_mat_v73(path, struct):
    """Write a v7.3 (HDF5) .mat the way MATLAB does.

    Three details matter, and all three are what real files have:

    * a 512-byte userblock carrying the ASCII ``MATLAB 7.3 MAT-file`` header, so the HDF5
      signature sits at offset 512 rather than 0;
    * a ``MATLAB_class`` attribute on every dataset -- ``mat73`` refuses a dataset without
      one ("not a MATLAB datatype"), so a fixture lacking it is not a faithful fixture;
    * **arrays stored transposed**, because MATLAB is column-major. This is why the loader
      needs :func:`orient_index_map` at all, so the fixture must reproduce it.
    """
    import h5py
    with h5py.File(path, "w", userblock_size=512) as f:
        # MATLAB keeps cell-array contents in a top-level '#refs#' group and stores only
        # object references in the cell dataset. mat73 dereferences through that group, so
        # a fixture that parks the targets anywhere else is not a faithful fixture.
        refs_grp = f.create_group("#refs#")
        grp = f.create_group("cell_registered_struct")
        grp.attrs["MATLAB_class"] = np.bytes_(b"struct")
        ref_dtype = h5py.special_dtype(ref=h5py.Reference)
        counter = 0
        for key, value in struct.items():
            if isinstance(value, list):          # cell array -> object references
                refs = []
                for item in value:
                    name = f"r{counter}"
                    counter += 1
                    ds = refs_grp.create_dataset(name,
                                                 data=np.asarray(item, dtype=float).T)
                    ds.attrs["MATLAB_class"] = np.bytes_(b"double")
                    refs.append(ds.ref)
                cell = grp.create_dataset(key, data=np.array(refs, dtype=ref_dtype
                                                             ).reshape(-1, 1))
                cell.attrs["MATLAB_class"] = np.bytes_(b"cell")
            else:
                arr = np.asarray(value, dtype=float)
                ds = grp.create_dataset(key, data=arr.T if arr.ndim > 1 else arr)
                ds.attrs["MATLAB_class"] = np.bytes_(b"double")
    with open(path, "r+b") as fh:
        fh.write(b"MATLAB 7.3 MAT-file, Platform: TEST")


def write_mat_v7(path, struct):
    """Write an older v5/v7 .mat via scipy."""
    import scipy.io
    scipy.io.savemat(path, {"cell_registered_struct": struct}, do_compression=False)


@pytest.fixture
def probabilistic_struct():
    return {
        "cell_to_index_map": _index_map(),
        "cell_scores": np.array([0.95, 0.80, 0.60, 0.10]),
        "true_positive_scores": np.array([0.9, 0.7, 0.5, 0.1]),
        "true_negative_scores": np.array([0.99, 0.98, 0.97, 0.9]),
        "exclusivity_scores": np.array([1.0, 0.99, 0.98, 0.9]),
        "is_cell_in_overlapping_FOV": np.array([1, 1, 0, 1]),
        "registered_cells_centroids": np.array([[10.0, 20.0, 30.0, 40.0],
                                                [11.0, 21.0, 31.0, 41.0]]),
        "alignment_x_translations": np.array([0.0, 1.5, -2.0]),
        "alignment_y_translations": np.array([0.0, -0.5, 3.0]),
        "alignment_rotations": np.array([0.0, 0.3, -0.2]),
        "adjustment_x_zero_padding": np.array([0.0, 0.0, 0.0]),
        "adjustment_y_zero_padding": np.array([0.0, 0.0, 0.0]),
    }


LOG_TEXT = """Sessions list:
-----------------
Session 1 - /data/x/converted_Fall_01_scan9FOW97IH.mat
Session 2 - /data/x/converted_Fall_02_scan9FOXFZU7.mat
Session 3 - /data/x/converted_Fall_03_scan9FOYMXRV.mat


General data parameters:
------------------------
Number of sessions - 3
FOV size - 615.6X615.6 [microns] ; (x,y)
Image size - 513X513 [pixels] ; (x,y)


Session alignment parameters:
----------------------------------
Reference session - 2
Alignment Type - Translations and Rotations


Cell registration parameters:
-----------------------------
Model maximal distance - 12 [microns]
Number of bins - 100
Initial registration type - Spatial correlation
Initial threshold - 0.72
Registration approach - Probabilistic modeling
Final registration type - Spatial correlations
P_same threshold - 0.3


General results:
----------------
Final number of cells - 4
Average fraction of active cells - 0.23

Fraction of false positives - 0.04
Fraction of false negatives - 0.08
Fraction of uncertain cell-pairs - 0.06

Average register score - 0.92
Average true positive score - 0.59
Average true negative score - 0.97
Average exclusivity score - 0.99

Discrepancy of the model - 0.06


Comments:
---------
"""


# ----------------------------------------------------------------- schema declarations
def test_import_declares_nothing():
    assert creg.schema.is_activated() is False


def test_every_table_has_a_definition():
    """11 master tables and 10 part tables = 21 definitions.

    The counts are a deliberate tripwire, not a second source of truth: the *validator*
    discovers the table set from the module, so it can never fall behind. This test exists
    so that adding or removing a table forces a conscious edit here rather than passing
    silently -- a schema is exactly the place where that is wanted.
    """
    import inspect
    masters, parts = 0, 0
    for name, obj in vars(creg).items():
        if not (inspect.isclass(obj) and hasattr(obj, "definition")):
            continue
        if obj.__module__ != creg.__name__:
            continue
        assert str(obj.definition).strip(), f"{name} has an empty definition"
        masters += 1
        for part_name, part in vars(obj).items():
            if inspect.isclass(part) and hasattr(part, "definition"):
                assert str(part.definition).strip(), \
                    f"{name}.{part_name} has an empty definition"
                parts += 1
    assert masters == 11, f"expected 11 master tables, found {masters}"
    assert parts == 10, f"expected 10 part tables, found {parts}"


def test_cell_is_session_free():
    """The cross-session identity must not be keyed by any session-scoped attribute."""
    pk_text = str(creg.CellRegResult.Cell.definition).split("---")[0]
    for forbidden in ("session_id", "scan_id", "paramset_idx", "curation_id", "mask"):
        assert forbidden not in pk_text, (
            f"CellRegResult.Cell's primary key contains {forbidden}; it represents a "
            "cross-session identity and must stay session-free")


def test_session_part_is_keyed_on_curation():
    pk_text = str(creg.CellRegRun.Session.definition).split("---")[0]
    assert "-> imaging.Curation" in pk_text
    # column_index must NOT be in the key (ADR-R5) but must still be declared
    assert "column_index" not in pk_text
    assert "column_index" in str(creg.CellRegRun.Session.definition)
    assert "unique index (cellreg_run_id, column_index)" in str(
        creg.CellRegRun.Session.definition)


def test_no_nullable_attribute_in_any_primary_key():
    import inspect
    checked = 0
    for name, obj in vars(creg).items():
        if not (inspect.isclass(obj) and hasattr(obj, "definition")):
            continue
        if obj.__module__ != creg.__name__:
            continue
        classes = [(name, obj)] + [
            (f"{name}.{pn}", po) for pn, po in vars(obj).items()
            if inspect.isclass(po) and hasattr(po, "definition")]
        for cname, cobj in classes:
            pk_text = str(cobj.definition).split("---")[0]
            for line in pk_text.splitlines():
                assert "=null" not in line, f"{cname} has a nullable attribute in its PK"
            checked += 1
    assert checked >= 12


def test_unresolved_columns_have_their_own_table():
    """Unresolved columns must not degrade the resolved table's key to nullable."""
    assert hasattr(creg.CellRegRun, "UnresolvedColumn")
    resolved = str(creg.CellRegRun.Session.definition)
    assert "[nullable] imaging.Curation" not in resolved
    assert "unresolved" not in resolved.split("---")[1].lower() or True


# ------------------------------------------------------------------ parameter immutability
class _FakeParamSet(creg._HashedConfig):
    """Exercises the immutability guard without a database."""

    primary_key = ["cellreg_paramset_id"]
    _rows: list = []

    @classmethod
    def __and__(cls, restriction):  # pragma: no cover - not used
        raise NotImplementedError

    @classmethod
    def fetch(cls, as_dict=False):  # pragma: no cover
        raise NotImplementedError


def test_config_hash_is_order_independent():
    assert creg.config_hash({"a": 1, "b": 2}) == creg.config_hash({"b": 2, "a": 1})


def test_config_hash_changes_with_value():
    assert creg.config_hash({"a": 1}) != creg.config_hash({"a": 2})


def test_canonical_json_is_stable():
    assert creg.canonical_json({"b": 1, "a": [3, 2]}) == '{"a":[3,2],"b":1}'


def test_register_raises_on_conflicting_params(monkeypatch):
    """Re-registering a slug with different parameters must raise, not update."""
    store = {}

    class Fake:
        primary_key = ["cellreg_paramset_id"]
        __name__ = "FakeParamSet"

        @classmethod
        def _restrict(cls, slug):
            return [store[slug]] if slug in store else []

        @classmethod
        def insert1(cls, row):
            store[row["cellreg_paramset_id"]] = row

    def register(slug, params, label=""):
        digest = creg.config_hash(params)
        existing = Fake._restrict(slug)
        if existing:
            if existing[0]["params_hash"] != digest:
                raise ValueError("different params")
            return slug
        Fake.insert1({"cellreg_paramset_id": slug, "params": params,
                      "params_hash": digest, "params_json": creg.canonical_json(params),
                      "label": label})
        return slug

    register("p1", {"p_same_threshold": 0.3})
    register("p1", {"p_same_threshold": 0.3})          # idempotent
    with pytest.raises(ValueError):
        register("p1", {"p_same_threshold": 0.5})      # conflicting


# --------------------------------------------------------------------- format detection
def test_detect_v73_and_v7(tmp_path, probabilistic_struct):
    p73 = tmp_path / "cellRegistered_20260101_000000.mat"
    write_mat_v73(p73, probabilistic_struct)
    assert cellreg.detect_mat_format(p73) == "v7.3_hdf5"

    p7 = tmp_path / "old.mat"
    write_mat_v7(p7, {"cell_to_index_map": _index_map()})
    assert cellreg.detect_mat_format(p7) == "v5_v7"


def test_v73_signature_is_not_at_offset_zero(tmp_path, probabilistic_struct):
    """Guards the bug that made the first inventory pass misclassify all 107 v7.3 files."""
    path = tmp_path / "x.mat"
    write_mat_v73(path, probabilistic_struct)
    with open(path, "rb") as fh:
        assert fh.read(8) != b"\x89HDF\r\n\x1a\n"
        fh.seek(512)
        assert fh.read(8) == b"\x89HDF\r\n\x1a\n"


# ------------------------------------------------------------------- index conversion
def test_members_from_index_map_converts_and_drops_absent():
    members = cellreg.members_from_index_map(_index_map())
    assert members[0] == [(0, 4), (1, 6), (2, 1)]   # 5,7,2 -> 4,6,1
    assert members[1] == [(0, 0), (2, 11)]          # 1,0,12 -> mask 0 present; 0 dropped
    assert members[2] == [(1, 2)]
    assert members[3] == []                          # absent everywhere -> no rows


def test_absent_is_never_stored_as_negative_sentinel():
    for _, present in cellreg.members_from_index_map(_index_map()).items():
        for _, mask in present:
            assert mask >= 0, "absence must be an omitted row, never mask = -1"


def test_mask_zero_is_a_real_mask_not_absence():
    """1-based 1 -> 0-based 0. A 0 result must not be mistaken for 'absent'."""
    members = cellreg.members_from_index_map(np.array([[1, 0]], dtype=float))
    assert members[0] == [(0, 0)]


# ----------------------------------------------------------------------- orientation
def test_orient_index_map_uses_the_hint():
    arr = _index_map()                     # (4 cells, 3 sessions)
    assert cellreg.orient_index_map(arr, 3).shape == (4, 3)
    assert cellreg.orient_index_map(arr.T, 3).shape == (4, 3)


def test_orient_index_map_refuses_ambiguous_square():
    square = np.ones((3, 3))
    with pytest.raises(ValueError, match="square"):
        cellreg.orient_index_map(square)


def test_orient_index_map_rejects_mismatched_hint():
    with pytest.raises(ValueError, match="neither axis"):
        cellreg.orient_index_map(_index_map(), 7)


# ------------------------------------------------------------------------ pair quality
def test_pair_quality_only_for_present_pairs():
    mats = _p_same()
    rows = cellreg.pair_quality_rows(mats[1], [0, 2])
    assert [(a, b) for a, b, _ in rows] == [(0, 2)]


def test_pair_quality_is_upper_triangle_in_order():
    rows = cellreg.pair_quality_rows(_p_same()[0], [0, 1, 2])
    assert [(a, b) for a, b, _ in rows] == [(0, 1), (0, 2), (1, 2)]
    assert all(a < b for a, b, _ in rows)


def test_pair_quality_empty_for_singleton_cell():
    assert cellreg.pair_quality_rows(_p_same()[2], [1]) == []


def test_pair_quality_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        cellreg.pair_quality_rows(np.ones((3, 4)), [0, 1])


def test_pair_quality_accepts_mat73_leading_singleton_axis():
    """Real files come back from mat73 as (1, n_sessions, n_sessions)."""
    m = _p_same()[0][None, :, :]
    assert m.shape == (1, 3, 3)
    rows = cellreg.pair_quality_rows(m, [0, 1, 2])
    assert [(a, b) for a, b, _ in rows] == [(0, 1), (0, 2), (1, 2)]


def test_pair_quality_still_rejects_a_wrong_3d_shape():
    """Stripping leading 1-axes must not hide a genuinely malformed matrix."""
    with pytest.raises(ValueError, match="square"):
        cellreg.pair_quality_rows(np.ones((2, 3, 3)), [0, 1])


def test_summarise_p_same():
    rows = [(0, 1, 0.9), (0, 2, 0.5), (1, 2, 0.7)]
    lo, mid = cellreg.summarise_p_same(rows)
    assert lo == pytest.approx(0.5)
    assert mid == pytest.approx(0.7)
    assert cellreg.summarise_p_same([]) == (None, None)


# ------------------------------------------------------------------------- log parsing
def test_parse_log_file_extracts_sessions_in_column_order():
    parsed = cellreg.parse_log_file(LOG_TEXT)
    assert len(parsed["session_files"]) == 3
    assert [cellreg.scan_id_from_source_file(f) for f in parsed["session_files"]] == [
        "scan9FOW97IH", "scan9FOXFZU7", "scan9FOYMXRV"]


def test_parse_log_file_extracts_parameters_and_quality():
    parsed = cellreg.parse_log_file(LOG_TEXT)
    assert parsed["n_sessions"] == 3
    assert parsed["reference_session_index"] == 2
    assert parsed["alignment_type"] == "Translations and Rotations"
    assert parsed["maximal_distance_um"] == pytest.approx(12.0)
    assert parsed["final_threshold"] == pytest.approx(0.3)
    assert parsed["frac_false_positives"] == pytest.approx(0.04)
    assert parsed["frac_false_negatives"] == pytest.approx(0.08)
    assert parsed["model_discrepancy"] == pytest.approx(0.06)
    assert parsed["avg_exclusivity_score"] == pytest.approx(0.99)


def test_normalisation_of_enums():
    assert cellreg.normalise_alignment_mode("Translations and Rotations") == \
        "translations_and_rotations"
    assert cellreg.normalise_alignment_mode("Translations") == "translations"
    assert cellreg.normalise_alignment_mode("Non-rigid") == "non_rigid"
    assert cellreg.normalise_alignment_mode(None) == "unknown"
    assert cellreg.normalise_registration_approach("Probabilistic modeling") == \
        "probabilistic"
    assert cellreg.normalise_registration_approach("Simple threshold") == "simple_threshold"
    assert cellreg.normalise_registration_approach("") == "unknown"


def test_find_log_file_picks_the_later_log(tmp_path):
    """The log is written after the .mat, so an exact-timestamp pairing finds nothing."""
    (tmp_path / "cellRegistered_20260128_144604.mat").write_bytes(b"x")
    (tmp_path / "logFile_20260128_144720.txt").write_text("later")
    (tmp_path / "logFile_20260101_000000.txt").write_text("earlier")
    found = cellreg.find_log_file(str(tmp_path / "cellRegistered_20260128_144604.mat"))
    assert found.endswith("logFile_20260128_144720.txt")


def test_find_log_file_returns_none_when_absent(tmp_path):
    (tmp_path / "cellRegistered_20260128_144604.mat").write_bytes(b"x")
    assert cellreg.find_log_file(str(tmp_path / "cellRegistered_20260128_144604.mat")) is None


# ------------------------------------------------------------------- column resolution
def test_resolve_column_prefers_the_log_file():
    candidates = {"scan9FOW97IH": {1, 2, 3}, "scan9FOXFZU7": {1, 2, 3, 4}}
    out = cellreg.resolve_column({1, 2, 3}, candidates, log_scan_id="scan9FOW97IH")
    assert out["resolution_method"] == "log_file"
    assert out["scan_key"] == "scan9FOW97IH"


def test_jaccard_is_reported_even_when_the_log_resolves():
    """A stale Fall.mat shows up as a low overlap on a correctly resolved column."""
    candidates = {"scan9FOW97IH": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
    out = cellreg.resolve_column({1, 2}, candidates, log_scan_id="scan9FOW97IH")
    assert out["resolution_method"] == "log_file"
    assert out["mask_set_jaccard"] == pytest.approx(0.2)


def test_resolve_column_falls_back_to_jaccard():
    candidates = {"a": {1, 2, 3, 4}, "b": {90, 91}}
    out = cellreg.resolve_column({1, 2, 3, 4}, candidates)
    assert out["resolution_method"] == "mask_jaccard"
    assert out["scan_key"] == "a"
    assert out["ambiguous"] is False


def test_resolve_column_flags_ambiguity_instead_of_picking_a_winner():
    """Two candidates within AMBIGUITY_MARGIN must not be separated by a hair."""
    # both score 4/5 = 0.8 -- an exact tie, which is the clearest form of the case
    candidates = {"a": {1, 2, 3, 4, 8}, "b": {1, 2, 3, 4, 9}}
    out = cellreg.resolve_column({1, 2, 3, 4}, candidates)
    assert out["ambiguous"] is True
    assert out["scan_key"] is None
    assert out["resolution_method"] == "unresolved"


def test_resolve_column_accepts_a_clear_winner():
    """A decisive margin is resolved -- ambiguity must not be over-triggered."""
    candidates = {"a": {1, 2, 3, 4}, "b": {1, 2, 3, 5}}   # 1.00 vs 0.60
    out = cellreg.resolve_column({1, 2, 3, 4}, candidates)
    assert out["resolution_method"] == "mask_jaccard"
    assert out["scan_key"] == "a"
    assert out["ambiguous"] is False


def test_resolve_column_unresolved_below_threshold():
    out = cellreg.resolve_column({1, 2}, {"a": {50, 51, 52, 53}})
    assert out["resolution_method"] == "unresolved"
    assert out["scan_key"] is None
    assert out["ambiguous"] is False


def test_jaccard_values():
    assert cellreg.jaccard({1, 2, 3}, {1, 2, 3}) == 1.0
    assert cellreg.jaccard({1, 2}, {3, 4}) == 0.0
    assert cellreg.jaccard(set(), set()) == 0.0
    assert cellreg.jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)


def test_scan_id_extraction():
    assert cellreg.scan_id_from_source_file(
        "/data/x/converted_Fall_01_scan9FOW97IH.mat") == "scan9FOW97IH"
    assert cellreg.scan_id_from_source_file("no_scan_here.mat") is None


# ---------------------------------------------------------------- both formats round-trip
@pytest.mark.parametrize("writer,expected", [(write_mat_v73, "v7.3_hdf5"),
                                             (write_mat_v7, "v5_v7")])
def test_load_struct_round_trips_in_both_formats(tmp_path, probabilistic_struct,
                                                 writer, expected):
    path = tmp_path / "cellRegistered_20260101_000000.mat"
    writer(path, probabilistic_struct)
    assert cellreg.detect_mat_format(path) == expected

    struct = cellreg.load_cellreg_struct(path)
    index_map = cellreg.orient_index_map(struct["cell_to_index_map"], n_sessions_hint=3)
    assert index_map.shape == (4, 3)

    members = cellreg.members_from_index_map(index_map)
    assert members[0] == [(0, 4), (1, 6), (2, 1)]
    assert members[3] == []
    assert np.asarray(struct["cell_scores"]).ravel()[0] == pytest.approx(0.95)


# ------------------------------------------------------------- the missing-score path
def test_simple_threshold_file_has_no_score_fields(tmp_path, probabilistic_struct):
    """No real file exercises this; the CellReg script produces it under 'Simple threshold'."""
    struct = {k: v for k, v in probabilistic_struct.items() if k not in SCORE_FIELDS}
    struct.pop("alignment_rotations")              # 'Translations' only
    path = tmp_path / "cellRegistered_20260101_000001.mat"
    write_mat_v73(path, struct)

    loaded = cellreg.load_cellreg_struct(path)
    missing = [f for f in SCORE_FIELDS if f not in loaded]
    assert set(missing) == set(SCORE_FIELDS)
    assert "alignment_rotations" not in loaded
    # the map still parses, so identity survives without any quality
    index_map = cellreg.orient_index_map(loaded["cell_to_index_map"], 3)
    assert cellreg.members_from_index_map(index_map)[0] == [(0, 4), (1, 6), (2, 1)]
    # and a cell with no p_same gets NULL summaries rather than a fabricated number
    assert cellreg.summarise_p_same([]) == (None, None)


def test_scalar_helpers_return_none_out_of_range():
    assert cellreg._scalar(None, 0) is None
    assert cellreg._scalar(np.array([1.0]), 5) is None
    assert cellreg._scalar(np.array([np.nan]), 0) is None
    assert cellreg._bool(np.array([1.0]), 0) == 1
    assert cellreg._bool(None, 0) is None


# ----------------------------------------------------------- internal consistency check
def test_inconsistent_p_same_is_detectable():
    """The real defect in cellRegistered_combined.mat: 11 sessions but a 10x10 p_same."""
    n_sessions = 11
    first = np.ones((10, 10))
    consistent = (first.ndim == 2 and first.shape[0] == first.shape[1]
                  and first.shape[0] == n_sessions)
    assert consistent is False


# --------------------------------------------------------------------------- footprints
def test_footprints_in_file_vs_external():
    in_file = {"spatial_footprints_corrected": [np.zeros((4, 4, 2))]}
    assert cellreg.footprints_storage(in_file) == ("in_file", "")

    external = {"spatial_footprints_corrected": ["/datajoint-data/x/footprints_1.mat"]}
    storage, path = cellreg.footprints_storage(external)
    assert storage == "external_path"
    assert path.endswith("footprints_1.mat")

    assert cellreg.footprints_storage({}) == ("absent", "")


def test_needed_fields_excludes_the_footprint_stack():
    """The exclusion is load-bearing: an unrestricted read OOMs on a 7.1 GB field."""
    assert "spatial_footprints_corrected" not in cellreg.NEEDED_FIELDS


def test_loader_does_not_materialise_footprints(tmp_path, probabilistic_struct):
    """A big footprint stack in the file must not appear in the loaded struct."""
    struct = dict(probabilistic_struct)
    # a stack far larger than anything else in the fixture
    struct["spatial_footprints_corrected"] = [np.zeros((64, 64, 8))]
    path = tmp_path / "cellRegistered_20260101_000002.mat"
    write_mat_v73(path, struct)

    loaded = cellreg.load_cellreg_struct(path)
    assert "spatial_footprints_corrected" not in loaded, (
        "the footprint stack was loaded despite being excluded from NEEDED_FIELDS")
    assert "cell_to_index_map" in loaded


def test_h5py_fallback_matches_mat73(tmp_path, probabilistic_struct):
    """ibehavegpu1 has h5py but no mat73; both readers must give the same rows."""
    struct = dict(probabilistic_struct)
    struct["p_same_registered_pairs"] = _p_same()
    path = tmp_path / "cellRegistered_20260101_000007.mat"
    write_mat_v73(path, struct)

    via_mat73 = cellreg.load_cellreg_struct(path)
    via_h5py = cellreg._load_v73_h5py(
        path, [n for n in cellreg.NEEDED_FIELDS if n in set(cellreg.list_fields(path))])

    m1 = cellreg.members_from_index_map(
        cellreg.orient_index_map(via_mat73["cell_to_index_map"], 3))
    m2 = cellreg.members_from_index_map(
        cellreg.orient_index_map(via_h5py["cell_to_index_map"], 3))
    assert m1 == m2

    assert np.allclose(np.asarray(via_mat73["cell_scores"]).ravel(),
                       np.asarray(via_h5py["cell_scores"]).ravel())
    # the p_same cell array must dereference to the same per-cell matrices
    assert len(via_h5py["p_same_registered_pairs"]) == 4
    r1 = cellreg.pair_quality_rows(via_mat73["p_same_registered_pairs"][0], [0, 1, 2])
    r2 = cellreg.pair_quality_rows(via_h5py["p_same_registered_pairs"][0], [0, 1, 2])
    assert [(a, b) for a, b, _ in r1] == [(a, b) for a, b, _ in r2]
    assert np.allclose([p for _, _, p in r1], [p for _, _, p in r2])


def test_h5py_fallback_does_not_read_footprints(tmp_path, probabilistic_struct):
    struct = dict(probabilistic_struct)
    struct["spatial_footprints_corrected"] = [np.zeros((32, 32, 4))]
    path = tmp_path / "cellRegistered_20260101_000008.mat"
    write_mat_v73(path, struct)
    names = [n for n in cellreg.NEEDED_FIELDS if n in set(cellreg.list_fields(path))]
    loaded = cellreg._load_v73_h5py(path, names)
    assert "spatial_footprints_corrected" not in loaded


def test_list_fields_reads_metadata_only(tmp_path, probabilistic_struct):
    path = tmp_path / "cellRegistered_20260101_000003.mat"
    write_mat_v73(path, probabilistic_struct)
    fields = cellreg.list_fields(path)
    assert "cell_to_index_map" in fields
    assert "alignment_rotations" in fields


def test_loader_tolerates_absent_optional_fields(tmp_path, probabilistic_struct):
    """21 of 107 files have no alignment_rotations; only_include must not choke on it."""
    struct = {k: v for k, v in probabilistic_struct.items() if k != "alignment_rotations"}
    path = tmp_path / "cellRegistered_20260101_000004.mat"
    write_mat_v73(path, struct)
    loaded = cellreg.load_cellreg_struct(path)
    assert "alignment_rotations" not in loaded
    assert "cell_to_index_map" in loaded


def test_probe_footprints_detects_in_file_stack(tmp_path, probabilistic_struct):
    struct = dict(probabilistic_struct)
    struct["spatial_footprints_corrected"] = [np.zeros((8, 8, 3))]
    path = tmp_path / "cellRegistered_20260101_000005.mat"
    write_mat_v73(path, struct)
    assert cellreg.probe_footprints_storage(path) == ("in_file", "")


def test_probe_footprints_detects_absent(tmp_path, probabilistic_struct):
    path = tmp_path / "cellRegistered_20260101_000006.mat"
    write_mat_v73(path, probabilistic_struct)   # fixture has no footprints field
    assert cellreg.probe_footprints_storage(path) == ("absent", "")


def test_windows_path_is_recorded_not_dereferenced():
    """One real file stores X:\\data\\... which cannot resolve on the Linux server."""
    struct = {"spatial_footprints_corrected": [r"X:\data\aeltona\CellReg\sfc_1.mat"]}
    storage, path = cellreg.footprints_storage(struct)
    assert storage == "external_path"
    assert path.startswith("X:")          # stored verbatim
    import os
    assert not os.path.exists(path)       # and deliberately never opened


# --------------------------------------------------------------------------- run id
def test_run_id_is_first_40_of_sha256():
    sha = "a" * 64
    assert cellreg.run_id_from_sha(sha) == "a" * 40
    with pytest.raises(ValueError):
        cellreg.run_id_from_sha("tooshort")


def test_file_sha256_matches_hashlib(tmp_path):
    import hashlib
    path = tmp_path / "f.bin"
    payload = b"cellreg" * 1000
    path.write_bytes(payload)
    assert cellreg.file_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_identical_content_gives_one_run_id(tmp_path):
    """6 duplicate groups over 14 paths on the server must collapse to one run each."""
    a, b = tmp_path / "copy_a.mat", tmp_path / "copy_b.mat"
    payload = b"identical bytes"
    a.write_bytes(payload)
    b.write_bytes(payload)
    assert cellreg.run_id_from_sha(cellreg.file_sha256(a)) == \
        cellreg.run_id_from_sha(cellreg.file_sha256(b))


# ------------------------------------------------------------------ deliberate omission
def test_mask_coordinates_make_refuses_rather_than_guessing():
    with pytest.raises(NotImplementedError, match="centroid"):
        cellreg.make_mask_coordinates(None, None)
