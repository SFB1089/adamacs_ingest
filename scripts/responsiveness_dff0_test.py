"""Phase-1 test of the responsiveness schema (dF/F0) under a personal prefix.

Declares ``tobiasr_responsiveness`` (or --schema), inserts the known dF/F0 recipes, stages
Dff0 tasks for a few curations, populates Dff0 and CascadeInput, and prints a summary.

Writes ONLY to the test schema and to a scratch external store (--store-dir). The
production schemas are read through virtual modules; ``adamacs.pipeline`` is never imported.

Run from a directory with dj_local_conf.json (e.g. ~/adamacs), env ``datajoint``:

    python ~/adamacs_ingest/scripts/responsiveness_dff0_test.py
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # adamacs_ingest checkout first

import datajoint as dj  # noqa: E402

from adamacs.responsiveness import dff0 as D  # noqa: E402
from adamacs.schemas import responsiveness as R  # noqa: E402

#: (session suffix, curation_id, [dff_paramset_idx, ...]) chosen to cover both code eras:
#: 9FUMXRWL: 25ms written 2026-09-24 (recipe 2), _11 written 2025-12-14 (recipe 5), + lab standard (1)
#: 9G1PMH2S: cascade_inference written 2026-08-27 (3), 25ms written 2026-09-03 23:53 (2)
#: 9FXY0HRX: everything written 2026-08-24..30, before the change (3, 5, 7)
TEST_SET = [("9FUMXRWL", 1, [1, 2, 5]), ("9G1PMH2S", 2, [2, 3]), ("9FXY0HRX", 5, [3, 5, 7])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="tobiasr_responsiveness")
    ap.add_argument("--store", default="resp-test")
    ap.add_argument("--store-dir", default=os.path.expanduser("~/dj_stores/responsiveness_test"))
    a = ap.parse_args()
    assert a.schema.startswith("tobiasr"), "test runs only under a personal prefix"
    assert not a.store_dir.startswith("/datajoint-db"), "never write the test store under /datajoint-db"

    os.makedirs(a.store_dir, exist_ok=True)
    dj.config.setdefault("stores", {})[a.store] = {"protocol": "file", "location": a.store_dir}

    vm = {n: dj.create_virtual_module(n, "roselab_" + n, create_schema=False, create_tables=False)
          for n in ("imaging", "scan", "event")}
    R.activate(a.schema, imaging=vm["imaging"], scan=vm["scan"], event=vm["event"], store=a.store)

    R.DffParamSet.insert_known_recipes()
    print("recipes:", len(R.DffParamSet()))

    activity_keys = []
    for sfx, cur, recipes in TEST_SET:
        k = {"session_id": "sess" + sfx, "scan_id": "scan" + sfx, "paramset_idx": 10, "curation_id": cur}
        fk = (vm["imaging"].Fluorescence & k).fetch1("KEY")
        R.Dff0Task.insert([{**fk, "dff_paramset_idx": i, "task_owner": "TR",
                            "task_note": "phase-1 test vs stored CASCADE"} for i in recipes],
                          skip_duplicates=True)
        activity_keys.append(k)

    R.Dff0.populate(display_progress=True, suppress_errors=False)
    R.CascadeInput.populate(activity_keys, display_progress=True, suppress_errors=False)

    print(R.Dff0.proj("n_masks", "n_frames", "dark_signal", "dark_frame_start", "dark_frame_stop",
                      "dark_source", "n_f0_nonpositive", "compute_seconds"))
    print(R.CascadeInput())


if __name__ == "__main__":
    main()
