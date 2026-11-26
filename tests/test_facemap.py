"""Facemap ingest helpers: task identity, segment offsets, frame-timing rules, within-segment references."""
import numpy as np
import pytest

F = pytest.importorskip("adamacs.schemas.facemap")

V = [{"session_id": "sessA", "scan_id": "scanA", "recording_id": "scanA_bench2p_face"},
     {"session_id": "sessB", "scan_id": "scanB", "recording_id": "scanB_bench2p_face"}]


def test_task_id_deterministic_and_rerun_distinct():
    a = F.task_id("leon_site_concat_2026", V, "file:/x|10|1")
    assert a == F.task_id("leon_site_concat_2026", V, "file:/x|10|1")          # same plan twice: same id
    assert a != F.task_id("leon_site_concat_2026", V, "file:/x|11|2")          # regenerated file: new run
    assert a != F.task_id("leon_site_concat_2026", V[::-1], "file:/x|10|1")    # order is part of the manifest
    assert F.task_id("p", V, "instance:0") != F.task_id("p", V, "instance:1")  # deliberate trigger re-run
    assert len(a) == 12


def test_file_identity(tmp_path):
    f = tmp_path / "run_proc.npy"
    f.write_bytes(b"abc")
    assert F.file_identity(str(f)).startswith(f"file:{f}|3|")


def test_segment_offsets_and_split_motion():
    assert F.segment_offsets([5, 3, 4]).tolist() == [0, 5, 8]
    assert F.classify_split_motion(3, 3, 12) == "split"
    assert F.classify_split_motion(12, 3, 12) == "full_concat"
    assert F.classify_split_motion(7, 3, 12) == "absent"


def _aux(n_before, n_gate, n_after, dt=1 / 60):
    """Trigger train: n_before before the gate, n_gate inside it, n_after after; gate edges between pulses."""
    t = np.arange(n_before + n_gate + n_after) * dt
    gate = (t[n_before] - dt / 2, t[n_before + n_gate - 1] + dt / 2)
    return t, gate


def test_gate_rule_takes_last_n_in_gate():
    aux, gate = _aux(100, 1121, 20)
    times, n_in, n_unsaved = F.gate_rule_times(aux, gate, 1000)
    assert n_in == 1121 and n_unsaved == 121
    assert np.array_equal(times, aux[100 + 121:100 + 1121])


def test_gate_rule_too_few_triggers():
    aux, gate = _aux(10, 50, 10)
    times, n_in, n_unsaved = F.gate_rule_times(aux, gate, 60)
    assert times is None and n_unsaved == -10


def test_start_anchored_counts_from_gate_opening_and_leaves_tail_untimed():
    aux, gate = _aux(100, 1078, 0)              # 78 unsaved: the video runs past the last trigger
    t, n_after = F.start_anchored_times(aux, gate[0], 1000)
    assert n_after == 1078
    assert np.array_equal(t[:957], aux[100 + 121:100 + 1078])
    assert np.isnan(t[957:]).all() and np.isnan(t).sum() == 43


def test_start_anchored_equals_gate_rule_when_unsaved_is_121():
    aux, gate = _aux(100, 1121, 30)
    g, _, _ = F.gate_rule_times(aux, gate, 1000)
    s, _ = F.start_anchored_times(aux, gate[0], 1000)
    assert np.array_equal(g, s)


def test_rows_to_times_with_offset():
    vt = np.arange(10.0)
    assert np.array_equal(F.rows_to_times(vt, 10, 0), vt)
    t = F.rows_to_times(vt, 8, 4)                # facemap row r = video frame r + 4
    assert np.array_equal(t[:6], vt[4:]) and np.isnan(t[6:]).all()


def test_count_status():
    assert F.count_status(0, 5) == "no_events"
    assert F.count_status(5, 5) == "exact"
    assert F.count_status(6, 5) == "events_longer"
    assert F.count_status(4, 5) == "events_shorter"


def test_dist_refs_match_leondrift_definition():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(101, 4))
    temporal, rnd = F.dist_refs(A, seed=3)
    h = 50
    assert temporal == pytest.approx(np.linalg.norm(np.median(A[:h], 0) - np.median(A[h:2 * h], 0)))
    r = np.random.default_rng(3)
    ref = np.mean([np.linalg.norm(np.median(A[p[:h]], 0) - np.median(A[p[h:2 * h]], 0))
                   for p in (r.permutation(101) for _ in range(F.N_RANDOM_SPLITS))])
    assert rnd == pytest.approx(ref)


def test_masks_first_k_float32():
    proc = {"motMask_reshape": [np.ones((4, 5, 10), np.float32)]}
    m = F._masks(proc, "mot", 3)
    assert m.shape == (4, 5, 3) and m.dtype == np.float32
    flat = {"motMask": [np.arange(4 * 5 * 10, dtype=np.float32).reshape(20, 10)], "Lybin": np.array([4]), "Lxbin": np.array([5])}
    assert F._masks(flat, "mot", 2).shape == (4, 5, 2)


def test_definitions_parse():
    from datajoint.declare import compile_attribute
    n = 0
    for name, c in vars(F).items():
        if not (isinstance(c, type) and "definition" in vars(c) and c.__module__ == F.__name__):
            continue
        for cls in [c] + [p for p in vars(c).values() if isinstance(p, type) and "definition" in vars(p)]:
            for line in vars(cls)["definition"].splitlines():
                s = line.strip()
                if s and not s.startswith(("#", "->", "---", "unique index")):
                    compile_attribute(s, in_key=False, foreign_key_sql=[], context={})
                    n += 1
    assert n > 80


def _fake_proc(path, n=40, k=6, ly=5, lx=4, movie=True):
    rng = np.random.default_rng(0)
    proc = {"motSVD": [rng.normal(size=(n, k)).astype(np.float32)],
            "movSVD": [rng.normal(size=(n, k)).astype(np.float32)] if movie else [],
            "motion": [rng.random(n).astype(np.float32)],
            "motSv": np.arange(k, dtype=np.float32), "movSv": np.arange(k, dtype=np.float32),
            "motMask_reshape": [np.ones((ly, lx, k), np.float32)],
            "movMask_reshape": [np.ones((ly, lx, k), np.float32)] if movie else [],
            "avgframe_reshape": np.zeros((ly, lx), np.float32), "avgmotion_reshape": np.zeros((ly, lx), np.float32),
            "sbin": 1, "filenames": [["video.mp4"]]}
    np.save(path, proc, allow_pickle=True)
    return proc


@pytest.mark.parametrize("movie", [True, False])
def test_compute_with_and_without_movie_svd(tmp_path, movie):
    import datetime as dt, os
    f = str(tmp_path / "run_proc.npy")
    _fake_proc(f, movie=movie)
    key = {"facemap_task_id": "t" * 12}
    videos = [dict(segment_index=0, session_id="s", scan_id="c", recording_id="r", video_path=str(tmp_path / "v.mp4"),
                   n_frames_video=40)]
    run_row, seg_rows = F.FacemapProcessing.make_compute(None, key, ("t" * 12, "load", f, "per_video", F.file_identity(f)), videos)
    assert run_row["n_frames_total"] == 40 and seg_rows[0]["n_frames"] == 40 and seg_rows[0]["frames_match"]
    st = os.stat(f)
    run = (f, str(st.st_size), str(dt.datetime.fromtimestamp(st.st_mtime).replace(microsecond=0)), "40")
    out = F.FacialSignal.make_compute(None, key, run, [dict(segment_index=0, frame_offset=0, n_frames=40)])
    master, summary, regions, basis, traces, positions = out
    assert [b["svd"] for b in basis] == (["motion", "movie"] if movie else ["motion"])
    assert traces[0]["mot_svd"].shape == (40, 6)
    assert (traces[0]["mov_svd"] is None) == (not movie)
    assert (positions[0]["mov_median"] is None) == (not movie) and positions[0]["mot_median"].shape == (6,)


def test_replaced_file_is_refused(tmp_path):
    f = str(tmp_path / "run_proc.npy")
    _fake_proc(f)
    stale = "file:" + f + "|1|1"
    videos = [dict(segment_index=0, session_id="s", scan_id="c", recording_id="r", video_path="", n_frames_video=40)]
    with pytest.raises(ValueError, match="not the planned file"):
        F.FacemapProcessing.make_compute(None, {}, ("t" * 12, "load", f, "per_video", stale), videos)
