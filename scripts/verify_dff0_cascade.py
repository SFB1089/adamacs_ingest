"""Verify DffParamSet recipes against the stored CASCADE activities.

Three stages, because CASCADE must run with the worker's own Cascade checkout on BUGPU:

  export   (TATCHU3, env datajoint)  for every responsiveness.CascadeInput row (legacy CASCADE activity)
           whose recipe has an imaging.Dff0, sample masks and write <out>/<activity>.npz with that dF/F0,
           the stored CASCADE trace and the model name
  predict  (BUGPU, env Cascade, CPU only)  run cascade.predict on each npz -> <name>.pred.npz
           (this stage is scripts/cascade_predict_npz.py, copied to BUGPU)
  compare  (TATCHU3)  stored vs recomputed per mask; insert CascadeRecompute rows

    python verify_dff0_cascade.py export  --out ~/resp_verify
    python verify_dff0_cascade.py compare --out ~/resp_verify --cascade-commit <sha> --host ibehavegpu1
"""
import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import datajoint as dj  # noqa: E402
import numpy as np  # noqa: E402

from adamacs.schemas import responsiveness as R  # noqa: E402


def _activate(schema):
    vm = {n: dj.create_virtual_module(n, "roselab_" + n, create_schema=False, create_tables=False)
          for n in ("imaging", "scan", "event")}
    R.activate(schema, imaging=vm["imaging"], scan=vm["scan"], event=vm["event"],
               create_schema=False, create_tables=False)
    return vm


def _name(k):
    """File stem carrying the full activity identity (scan, paramset, curation, method)."""
    return f"{k['scan_id']}_p{k['paramset_idx']}_c{k['curation_id']}_{k['extraction_method']}"


def _fingerprint(masks, channels, dff):
    """sha1 over the exported inputs, copied into the prediction so stale predictions are detected."""
    h = hashlib.sha1()
    for arr in (np.asarray(masks, np.int64), np.asarray(channels, np.int64), np.ascontiguousarray(dff, np.float32)):
        h.update(arr.tobytes())
    return h.hexdigest()


def export(a, vm):
    os.makedirs(a.out, exist_ok=True)
    im = vm["imaging"]
    rng = np.random.default_rng(0)
    for ci in (R.CascadeInput & (im.Dff0 * R.CascadeInput).proj()).fetch(as_dict=True):
        act = {k: ci[k] for k in ("session_id", "scan_id", "paramset_idx", "curation_id", "extraction_method")}
        dkey = {**{k: ci[k] for k in ("session_id", "scan_id", "paramset_idx", "curation_id")},
                "dff_paramset_idx": ci["dff_paramset_idx"]}
        masks = (im.Dff0.Trace & dkey).fetch("mask", order_by="mask")
        masks = np.unique(masks)
        pick = np.sort(rng.choice(masks, size=min(a.n_masks, len(masks)), replace=False))
        restr = [{"mask": int(m)} for m in pick]
        m1, c1, dff = (im.Dff0.Trace & dkey & restr).fetch("mask", "fluo_channel", "dff",
                                                          order_by="mask, fluo_channel")
        m2, c2, spk = (im.Activity.Trace & act & restr).fetch("mask", "fluo_channel", "activity_trace",
                                                              order_by="mask, fluo_channel")
        assert list(zip(m1, c1)) == list(zip(m2, c2)), (act, m1, c1, m2, c2)
        model = (im.ActivityCascadeTask & act).fetch1("model_name")
        dff = np.vstack(dff)
        stem = os.path.join(a.out, _name(act))
        if os.path.exists(stem + ".pred.npz"):      # a rewritten export invalidates its old prediction
            os.remove(stem + ".pred.npz")
        np.savez(stem + ".npz", masks=np.asarray(m1), channels=np.asarray(c1), dff=dff,
                 stored=np.vstack(spk), model_name=model, dff_paramset_idx=ci["dff_paramset_idx"],
                 key=json.dumps({k: (int(v) if isinstance(v, (int, np.integer)) else str(v)) for k, v in act.items()}),
                 fingerprint=_fingerprint(m1, c1, dff))
        print("exported", _name(act), "recipe", ci["dff_paramset_idx"], model, len(m1), "masks")


def compare(a, vm):
    rows = []
    for f in sorted(glob.glob(os.path.join(a.out, "*.npz"))):
        if f.endswith(".pred.npz"):
            continue
        p = f[:-4] + ".pred.npz"
        if not os.path.exists(p):
            print("no prediction for", f)
            continue
        z, q = np.load(f, allow_pickle=True), np.load(p, allow_pickle=True)
        if "fingerprint" not in q.files or str(q["fingerprint"]) != str(z["fingerprint"]):
            print("stale or unverifiable prediction, skipped:", os.path.basename(p))
            continue
        S, P = z["stored"].astype(float), q["pred"].astype(float)
        if S.shape != P.shape:
            print("shape mismatch, skipped:", os.path.basename(p), S.shape, P.shape)
            continue
        ok = np.isfinite(S) & np.isfinite(P)
        d = np.abs(S - P)[ok]
        corr = [np.corrcoef(s[o], r[o])[0, 1] for s, r, o in zip(S, P, ok) if o.sum() > 10 and np.std(s[o]) > 0]
        nan_agree = bool(np.array_equal(np.isnan(S), np.isnan(P)))
        key = json.loads(str(z["key"]))
        row = {**key, "dff_paramset_idx": int(z["dff_paramset_idx"]), "model_name": str(z["model_name"]),
               "cascade_commit": a.cascade_commit, "run_host": a.host, "n_masks_checked": int(S.shape[0]),
               "max_abs_diff": float(d.max()), "median_abs_diff": float(np.median(d)),
               "min_corr": float(np.min(corr)) if corr else float("nan"), "checked_at": dt.datetime.now()}
        rows.append(row)
        print(f"{os.path.basename(f)[:-4]:60s} recipe {row['dff_paramset_idx']}  max|d| {row['max_abs_diff']:.3g}"
              f"  median|d| {row['median_abs_diff']:.3g}  min r {row['min_corr']:.6f}  NaN pattern equal {nan_agree}")
    if a.insert:
        R.CascadeRecompute.insert(rows, skip_duplicates=True)
        print("inserted", len(rows), "CascadeRecompute rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["export", "compare"])
    ap.add_argument("--out", default=os.path.expanduser("~/resp_verify"))
    ap.add_argument("--schema", default="roselab_responsiveness")
    ap.add_argument("--n-masks", type=int, default=40)
    ap.add_argument("--cascade-commit", default="")
    ap.add_argument("--host", default="ibehavegpu1")
    ap.add_argument("--insert", action="store_true")
    a = ap.parse_args()
    vm = _activate(a.schema)
    export(a, vm) if a.stage == "export" else compare(a, vm)


if __name__ == "__main__":
    main()
