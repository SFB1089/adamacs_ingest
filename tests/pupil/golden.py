"""Golden tests for the pupil / glint / DISK port: run make() read-only and compare with the stored rows.

Two kinds of comparison, both bitwise (np.array_equal with NaN == NaN; floats with ==):
  * port vs stored: make() of the ported code (this repository) against the row in the live database;
  * port vs source: the same make() run from the SFB1089/adamacs code the rows were made with
    (fix/rimfit-early-radius 2bac5ac, fix/glint-fallback-gate e6d15c5, main 61f1bab for DISK).
Nothing is written: inserts are captured in memory (readonly_db.capture_inserts) and every other write path is blocked.

  python tests/pupil/golden.py run --code port|source --case NAME --out DIR     # one make(), result pickled to DIR
  python tests/pupil/golden.py compare --out DIR --tsv FILE                     # port vs stored, port vs source

Video fits take minutes per recording (CPU); run cases in parallel processes. CUDA_VISIBLE_DEVICES is forced empty.
"""
import argparse
import datetime
import decimal
import os
import pickle
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import readonly_db  # noqa: E402

PORT = Path(__file__).resolve().parents[2]
HOME = Path.home()
SOURCE = {"rim": HOME / "adamacs_worktrees" / "fix-rimfit", "glint": HOME / "adamacs_worktrees" / "fix-glint-fallback",
          "disk": HOME / "adamacs", "glint12": HOME / "adamacs",
          # glint-1.1 code: `git -C ~/adamacs archive 10455f5 | tar -x -C ~/scratch_ingest_port/src_10455f5`
          "glint11": HOME / "scratch_ingest_port" / "src_10455f5"}
SOURCE_COMMIT = {"rim": "2bac5ac", "glint": "e6d15c5", "disk": "61f1bab", "glint12": "61f1bab", "glint11": "10455f5"}
PFX = "roselab_"
FIT_PK = ("session_id", "scan_id", "recording_id", "model_name")
# Fields whose value is a time stamp or a run time: not results.
VOLATILE = {"imputation_time", "disk_runtime_s"}

# name: (family, module, table class path, restriction of the stored row[, source code key if not the family's])
CASES = {
    "rim12_9G2N0274": ("rim", "pupil_tracking", "PupilEllipseFitting", dict(scan_id="scan9G2N0274", parameter_id=12)),
    "rim11_9G2N0274": ("rim", "pupil_tracking", "PupilEllipseFitting", dict(scan_id="scan9G2N0274", parameter_id=11)),
    "rim12_9FXEU7TJ": ("rim", "pupil_tracking", "PupilEllipseFitting", dict(scan_id="scan9FXEU7TJ", parameter_id=12)),
    "rim11_9FXEU7TJ": ("rim", "pupil_tracking", "PupilEllipseFitting", dict(scan_id="scan9FXEU7TJ", parameter_id=11)),
    "rim10_9FXEU7TJ": ("rim", "pupil_tracking", "PupilEllipseFitting", dict(scan_id="scan9FXEU7TJ", parameter_id=10)),
    "glint3_9G2N0274": ("glint", "pupil_glint", "CorrectedGlint", dict(scan_id="scan9G2N0274", glint_param_id=3)),
    "glint2_9G2N0274": ("glint", "pupil_glint", "CorrectedGlint", dict(scan_id="scan9G2N0274", glint_param_id=2), "glint12"),
    # a short recording with all three glint sets (ROS-1985, 9,139 frames)
    "glint1_9FSEII8M": ("glint", "pupil_glint", "CorrectedGlint", dict(scan_id="scan9FSEII8M", glint_param_id=1), "glint11"),
    "glint2_9FSEII8M": ("glint", "pupil_glint", "CorrectedGlint", dict(scan_id="scan9FSEII8M", glint_param_id=2), "glint12"),
    "glint3_9FSEII8M": ("glint", "pupil_glint", "CorrectedGlint", dict(scan_id="scan9FSEII8M", glint_param_id=3)),
}
# DISK cases are filled in by disk_cases() from the database (a short recording of the 11-mouse model).
DISK_MODEL = "transformer_leon_samesite11_llh09_len60"


def source_key(case):
    c = CASES.get(case)
    return (c[4] if c and len(c) > 4 else c[0]) if c else "disk"


def load(code, family, src=None):
    root = PORT if code == "port" else SOURCE[src or family]
    P = readonly_db.load_pipeline(root)
    from adamacs.schemas import disk
    if getattr(disk, "_linking_module", None) is None:            # the source pipeline does not bind disk
        import types
        link = types.ModuleType("golden_disk_link")
        for n in ("model", "pupil_tracking", "subject", "session", "event", "virtual_markers_optitrack"):
            setattr(link, n, getattr(P, n))
        disk.activate(PFX + "disk", PFX + "pupil_tracking", create_schema=False, create_tables=False, linking_module=link)
    return P, disk


def table(P, disk, module, path):
    obj = disk if module == "disk" else getattr(P, module)
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def disk_cases(disk):
    """A short head-fixed recording with a stored 11-mouse imputation, its frame quality, and one free-moving
    PupilFrameQuality row."""
    imp = disk.HeadfixedImputation & {"disk_model_name": DISK_MODEL}
    k = (imp.proj("n_frames_total") & "n_frames_total < 60000").fetch("KEY", order_by="n_frames_total", limit=1)[0]
    pose = {f: k[f] for f in FIT_PK}
    fq = (disk.HeadfixedFrameQuality & pose & {"parameter_id": k["parameter_id"]}).fetch("KEY", limit=1)[0]
    pfq = (disk.PupilFrameQuality.proj("n_frames") & "n_frames < 60000").fetch("KEY", order_by="n_frames", limit=1)[0]
    return {"hfimp": ("disk", "disk", "HeadfixedImputation", k),
            "hffq": ("disk", "disk", "HeadfixedFrameQuality", fq),
            "pfq": ("disk", "disk", "PupilFrameQuality", pfq)}


def run(code, case, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    family = CASES[case][0] if case in CASES else "disk"
    P, disk = load(code, family, source_key(case))
    import datajoint as dj
    cases = dict(CASES)
    if case not in cases:
        cases.update(disk_cases(disk))
    family, module, path, restr = cases[case][:4]
    if family == "disk":
        work = out / "disk_work" / f"{case}_{code}"
        (work / "jobs").mkdir(parents=True, exist_ok=True)
        dj.config["custom"]["disk_work_root"] = str(work)
    dj.config["custom"]["pupil_image_fit_cache"] = ""
    T = table(P, disk, module, path)
    key = (T & restr).fetch1("KEY")
    t0 = time.time()
    with readonly_db.capture_inserts() as cap:
        T().make(dict(key))
    res = dict(case=case, code=code, family=family, module=module, path=path, key=key, rows=cap.rows,
               seconds=time.time() - t0, root=str(PORT if code == "port" else SOURCE[source_key(case)]),
               numpy=np.__version__, python=sys.version.split()[0])
    try:
        import cv2
        res["cv2"] = cv2.__version__
    except ImportError:
        pass
    with open(out / f"{case}__{code}.pkl", "wb") as f:
        pickle.dump(res, f)
    print(f"{case} {code}: {len(cap.rows)} rows in {res['seconds']:.0f}s", flush=True)


def _num(x):
    if isinstance(x, decimal.Decimal):
        return float(x)
    return x


def same(a, b):
    """(bitwise equal, max abs difference or None)."""
    a, b = _num(a), _num(b)
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False, None
        r = [same(a[k], b[k]) for k in a]
        return all(x for x, _ in r), max((d for _, d in r if d is not None), default=None)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and (len(a) == 0 or isinstance(a[0], dict)):
        if len(a) != len(b):
            return False, None
        r = [same(x, y) for x, y in zip(a, b)]
        return all(x for x, _ in r), max((d for _, d in r if d is not None), default=None)
    if isinstance(a, (str, bytes, datetime.datetime, datetime.date)) or a is None or b is None:
        return a == b, None
    try:
        A, B = np.asarray(a), np.asarray(b)
    except Exception:  # noqa: BLE001
        return a == b, None
    if A.dtype.kind in "OUS" or B.dtype.kind in "OUS":
        return bool(A.shape == B.shape and all(same(x, y)[0] for x, y in zip(A.ravel(), B.ravel()))), None
    if A.shape != B.shape:
        return False, None
    if A.dtype.kind == "f" or B.dtype.kind == "f":
        eq = bool(np.array_equal(A.astype(float), B.astype(float), equal_nan=True))
    else:
        eq = bool(np.array_equal(A, B))
    with np.errstate(invalid="ignore"):
        d = np.abs(A.astype(float) - B.astype(float))
    d = float(np.nanmax(d)) if d.size and np.isfinite(d).any() else 0.0
    return eq, d


def tables_of(T):
    """{table_name: class} for a master table and its parts."""
    import datajoint as dj
    out = {T.table_name: T}
    for n in dir(T):
        p = getattr(T, n, None)
        if isinstance(p, type) and issubclass(p, dj.Part) and p is not dj.Part and n[0].isupper():
            out[p().table_name] = p
    return out


def stored_rows(P, disk, res):
    T = table(P, disk, res["module"], res["path"])
    return {tn: (t & res["key"]).fetch(as_dict=True) for tn, t in tables_of(T).items()}


def as_stored(value, sql_type):
    """What the database gives back for a value: a MySQL `float` column is single precision and is returned with 6
    significant digits; `decimal(p,s)` has s digits. Applied to the freshly computed value before comparing it with
    the stored one; nothing else is rounded."""
    t = (sql_type or "").lower()
    if value is None:
        return value
    if t.startswith("float"):
        return float("%.6g" % np.float32(value)) if np.isfinite(value) else float(value)
    if t.startswith("decimal"):
        s = int(t.split(",")[1].rstrip(")")) if "," in t else 0
        return float(round(decimal.Decimal(str(float(value))), s))
    return value


def compare_rows(captured, stored, pk_hint, coltype=None):
    """Match captured to stored rows by primary key; one result row per (table, field). coltype(table, field) gives
    the column's SQL type when the reference is the database (then the computed value is stored-precision first)."""
    res = []
    by_table = {}
    for tn, r in captured:
        by_table.setdefault(tn, []).append(r)
    for tn in sorted(set(by_table) | set(stored)):
        cap, sto = by_table.get(tn, []), stored.get(tn, [])
        if len(cap) != len(sto):
            res.append((tn, "<row count>", False, None, f"{len(cap)} vs {len(sto)}"))
            continue
        if not cap:
            continue
        pk = [f for f in pk_hint(tn) if f in cap[0]]
        sto_by = {tuple(str(s[f]) for f in pk): s for s in sto}
        fields = sorted(set(cap[0]) - set(pk))
        for f in fields:
            if f in VOLATILE:
                continue
            ok, dmax, n_bad = True, None, 0
            for c in cap:
                s = sto_by.get(tuple(str(c[x]) for x in pk))
                if s is None:
                    ok = False; n_bad += 1; continue
                v = as_stored(c[f], coltype(tn, f)) if coltype else c[f]
                e, d = same(v, s.get(f))
                ok &= e; n_bad += not e
                if d is not None:
                    dmax = d if dmax is None else max(dmax, d)
            res.append((tn, f, ok, dmax, f"{len(cap)} rows" + ("" if ok else f", {n_bad} differ")))
    return res


def compare(out, tsv):
    out = Path(out)
    lines = ["case\tcomparison\ttable\tfield\tbitwise_equal\tmax_abs_diff\tnote"]
    P = disk = None
    for f in sorted(out.glob("*__port.pkl")):
        port = pickle.load(open(f, "rb"))
        if P is None:
            P, disk = load("port", port["family"])
        T = table(P, disk, port["module"], port["path"])

        tabs = tables_of(T)

        def pk_hint(tn, tabs=tabs):
            return list(tabs[tn].primary_key) if tn in tabs else []

        def coltype(tn, f, tabs=tabs):
            a = tabs[tn].heading.attributes.get(f) if tn in tabs else None
            return a.type if a is not None else None
        sto = stored_rows(P, disk, port)
        for tn, fld, ok, d, note in compare_rows(port["rows"], sto, pk_hint, coltype):
            lines.append(f"{port['case']}\tport_vs_stored\t{tn}\t{fld}\t{ok}\t{'' if d is None else repr(d)}\t{note}")
        src_f = out / f.name.replace("__port.pkl", "__source.pkl")
        if src_f.exists():
            src = pickle.load(open(src_f, "rb"))
            src_rows = {}
            for tn, r in src["rows"]:
                src_rows.setdefault(tn, []).append(r)
            for tn, fld, ok, d, note in compare_rows(port["rows"], src_rows, pk_hint):
                lines.append(f"{port['case']}\tport_vs_source@{SOURCE_COMMIT[source_key(port['case'])]}\t{tn}\t{fld}\t{ok}\t"
                             f"{'' if d is None else repr(d)}\t{note}")
        lines.append(f"{port['case']}\truntime\t\t\t\t\tport {port['seconds']:.0f}s; numpy {port['numpy']}, "
                     f"cv2 {port.get('cv2')}, python {port['python']}")
    Path(tsv).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "compare", "list"])
    ap.add_argument("--code", choices=["port", "source"], default="port")
    ap.add_argument("--case")
    ap.add_argument("--out", default=str(HOME / "scratch_ingest_port" / "golden"))
    ap.add_argument("--tsv")
    a = ap.parse_args()
    if a.what == "run":
        run(a.code, a.case, a.out)
    elif a.what == "list":
        print(" ".join(CASES), "hfimp hffq pfq")
    else:
        compare(a.out, a.tsv or str(Path(a.out) / "golden.tsv"))
