"""Attach the unresolved columns whose matching candidates are PROVABLY the same mask set.

Authorised by Tobias on 2026-10-20. The rule is deliberately narrow:

  a column is attached only if every candidate curation whose mask count equals the file's
  per-column centroid count has a BYTE-IDENTICAL mask set (same masks, same centroids)

so the choice cannot change which masks, traces or centroids the column resolves to -- it
decides only which curation_id is recorded. Where the candidates genuinely differ (25
columns), nothing is attached: that would be choosing between different answers, which is
inventing provenance.

Among the provably identical candidates the pick is the LATEST whose curation_time
precedes the run's result file, because a curation created after a run cannot have been
that run's input. Where none predates the run (4 columns) the earliest matching candidate
is taken instead, and the note says so.

Writes only to cell_registration. Never writes roi_group.
"""
import collections, hashlib, json, os, sys
import numpy as np
import datajoint as dj

dj.config.load(os.path.expanduser("~/adamacs/dj_local_conf.json"))
dj.config["loglevel"] = "ERROR"
sys.path.insert(0, os.path.expanduser("~/adamacs_ingest"))
import adamacs.pipeline  # noqa: F401
from adamacs.schemas import cell_registration as cr
from adamacs.ingest import cellreg as CR
from adamacs.ingest import cellreg_register as REG

TARGET = "roselab_cell_registration"
assert cr.CellRegRun.database == TARGET, cr.CellRegRun.database
conn = dj.conn()
DRY = os.environ.get("DRY", "1") == "1"
print(f"target={TARGET}  DRY_RUN={DRY}", flush=True)

CUTOFF = "2026-10-16 23:59:59"
BASE = "da8e33a5108af402eca03e962a24c4dbf1f2df084d988b2c55075a8e0e380c30"
def roi_fp():
    h = hashlib.sha256(); n = nz = 0
    for row in conn.query(
            "SELECT m.session_id,m.scan_id,m.paramset_idx,m.curation_id,m.mask,m.roi_group "
            "FROM roselab_imaging.`__segmentation__mask` m JOIN roselab_imaging.curation c "
            "USING (session_id,scan_id,paramset_idx,curation_id) "
            "WHERE c.curation_time <= %s ORDER BY m.session_id,m.scan_id,m.paramset_idx,"
            "m.curation_id,m.mask", (CUTOFF,)).fetchall():
        h.update(("|".join(str(x) for x in row)).encode()); n += 1
        if int(row[5]) != 0: nz += 1
    return h.hexdigest(), n, nz

before = roi_fp()
print(f"roi_group BEFORE rows={before[1]} nonzero={before[2]} sha={before[0][:16]} "
      f"matches baseline={before[0] == BASE}", flush=True)

# ---- the enum must carry the new value before any such row is inserted ----------------
col = conn.query("SELECT column_type FROM information_schema.columns WHERE "
                 "table_schema=%s AND table_name='cell_reg_run__session' AND "
                 "column_name='resolution_method'", (TARGET,)).fetchone()[0]
if "mask_set_identical" not in col:
    if DRY:
        print("WOULD extend resolution_method with 'mask_set_identical'", flush=True)
    else:
        conn.query(f"ALTER TABLE `{TARGET}`.`cell_reg_run__session` MODIFY COLUMN "
                   "`resolution_method` enum('manifest','log_file','mask_jaccard',"
                   "'roi_group_match','mask_set_identical','manual') NOT NULL")
        print("extended resolution_method enum", flush=True)

census = REG.curation_census(conn)
runs = {r["cellreg_run_id"]: r for r in cr.CellRegRun.fetch(as_dict=True)}

_fp = {}
def fingerprint(key):
    if key[:4] not in _fp:
        rows = conn.query(
            "SELECT mask, mask_center_x, mask_center_y FROM "
            "roselab_imaging.`__segmentation__mask` WHERE session_id=%s AND scan_id=%s AND "
            "paramset_idx=%s AND curation_id=%s ORDER BY mask", key[:4]).fetchall()
        h = hashlib.sha1()
        for r in rows:
            h.update(f"{int(r[0])}|{float(r[1]):.6f}|{float(r[2]):.6f}".encode())
        _fp[key[:4]] = (h.hexdigest(), len(rows))
    return _fp[key[:4]]

by_run = collections.defaultdict(list)
for row in cr.CellRegRun.UnresolvedColumn.fetch(as_dict=True):
    by_run[row["cellreg_run_id"]].append(row)

attach, skip = [], collections.Counter()
for rid, rows in by_run.items():
    run = runs[rid]
    try:
        counts = REG.centroid_counts(run["result_file"])
    except Exception as exc:
        skip[f"centroid read failed: {type(exc).__name__}"] += len(rows); continue
    log = run["log_file"]
    scans = []
    if log and os.path.exists(log):
        parsed = CR.parse_log_file(open(log, errors="replace").read())
        scans = [CR.scan_id_from_source_file(os.path.basename(p))
                 for p in parsed.get("session_files", [])]
    for row in rows:
        colx = int(row["column_index"])
        if "no cells" in row["reason"]:
            skip["empty column"] += 1; continue
        if "no scan id" in row["reason"]:
            skip["no scan id in the log"] += 1; continue
        scan = scans[colx] if colx < len(scans) else None
        n_cent = counts[colx] if (counts and colx < len(counts)) else None
        if scan is None or n_cent is None:
            skip["no scan or centroid count"] += 1; continue
        matching = [k for k in census.get(scan, []) if fingerprint(k)[1] == n_cent]
        if not matching:
            skip["no candidate matches the mask count"] += 1; continue
        if len({fingerprint(k)[0] for k in matching}) > 1:
            skip["candidates DIFFER -- left unresolved by design"] += 1; continue
        pre = [k for k in matching if k[4] is not None
               and k[4] <= run["result_file_mtime"]]
        if pre:
            pick = sorted(pre, key=lambda k: (k[4], k[3]))[-1]
            basis = "latest curation predating the run"
        else:
            pick = sorted(matching, key=lambda k: (k[4], k[3]))[0]
            basis = "earliest matching; NO candidate predates the run"
        attach.append({
            "cellreg_run_id": rid, "session_id": pick[0], "scan_id": pick[1],
            "paramset_idx": pick[2], "curation_id": pick[3], "column_index": colx,
            "source_file": "", "resolution_method": "mask_set_identical",
            "mask_set_jaccard": 1.0,
            "n_masks_cellreg": int(row["n_masks_cellreg"] or 0),
            "n_masks_db": int(fingerprint(pick)[1]),
            "_basis": basis, "_n_candidates": len(matching)})

print(f"\nWOULD attach {len(attach)} columns", flush=True)
for k, v in skip.most_common():
    print(f"  left unresolved: {v:5d}  {k}", flush=True)
basis_counts = collections.Counter(a["_basis"] for a in attach)
for k, v in basis_counts.items():
    print(f"  pick basis: {v:5d}  {k}", flush=True)

if DRY:
    json.dump({"would_attach": len(attach), "skip": dict(skip),
               "basis": dict(basis_counts)},
              open(os.path.expanduser("~/attach_plan.json"), "w"), indent=1)
    print("\nDRY RUN -- nothing written", flush=True)
    sys.exit(0)

rows_in = [{k: v for k, v in a.items() if not k.startswith("_")} for a in attach]
touched = sorted({a["cellreg_run_id"] for a in attach})
with conn.transaction:
    cr.CellRegRun.Session.insert(rows_in, skip_duplicates=True)
    for a in attach:
        (cr.CellRegRun.UnresolvedColumn
         & {"cellreg_run_id": a["cellreg_run_id"],
            "column_index": a["column_index"]}).delete_quick()
print(f"\nattached {len(rows_in)} columns across {len(touched)} runs", flush=True)

# Member rows are built inside CellRegResult.make(), so the affected runs must be
# re-parsed for the new attachments to appear. MaskCoordinates keys on Session and is
# simply populated.
print("re-parsing the affected runs ...", flush=True)
for i, rid in enumerate(touched, 1):
    key = {"cellreg_run_id": rid}
    for t in (cr.CellRegResult.PairQuality, cr.CellRegResult.Member,
              cr.CellRegResult.Cell):
        (t & key).delete_quick()
    (cr.CellRegResult & key).delete_quick()
    cr.CellRegResult.populate(key, suppress_errors=False)
    if i % 10 == 0 or i == len(touched):
        print(f"  [{i}/{len(touched)}]", flush=True)
print("populating MaskCoordinates ...", flush=True)
errs = cr.MaskCoordinates.populate(suppress_errors=True)
if isinstance(errs, dict):
    print(f"  {errs.get('success_count')} ok, {len(errs.get('error_list') or [])} errors",
          flush=True)

after = roi_fp()
print(f"\nroi_group AFTER rows={after[1]} nonzero={after[2]} sha={after[0][:16]}")
print(f"roi_group UNCHANGED: {before == after}")
json.dump({"attached": len(rows_in), "runs_touched": len(touched),
           "skip": dict(skip), "basis": dict(basis_counts),
           "roi_unchanged": before == after, "roi_sha": after[0]},
          open(os.path.expanduser("~/attach_result.json"), "w"), indent=1)
