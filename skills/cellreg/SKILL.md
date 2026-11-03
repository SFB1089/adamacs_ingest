<!-- Published copy of the `cellreg` Claude Science skill (skill_01RxNc1svG1CKoQFLfu4aMpM).
     Kept here so the workflow is readable by anyone, with or without the skill installed.
     The skill itself is the live copy; if the two disagree, the skill wins and this file is
     stale -- regenerate it rather than editing it here. -->

---
name: cellreg
description: "Register cells across sessions with CellReg in the Rose Lab ADAMACS pipeline, and query the result. Use for any task about cross-session cell identity, roi_group, cell_registration, CellRegTask/CellRegRun/CellRegResult, longitudinal tracking of the same neuron over days, registering a site's gratings or natural-image recordings, or running the CellReg worker and MATLAB on ibehavegpu1. Covers planning which scans and curations become columns, declaring a task, exporting inputs from the database, running the MATLAB driver, ingesting, verifying, and the traps that silently produce wrong answers."
---

# CellReg — cross-session cell identity in ADAMACS

The `roselab_cell_registration` schema gives every CellReg run a queryable, provenance-
bearing home: which curation each column came from, what parameters were used, how good the
matches were, and which masks belong to which cross-session cell. Live since 2026-10-20.

Read [[adamacs-connect]] first for hosts, the SSH route and the standing database rules.

## Hard rules — do not negotiate these away

1. **Never write `roi_group`.** `roselab_imaging.__segmentation__mask.roi_group` is a legacy
   cache that the old pipeline overwrites in place; this schema never touches it.
   `RoiGroupProjection` is the only table that would, and nothing calls it. If a task seems
   to need `roi_group` written, stop and ask.
2. **Any database write needs Tobias's explicit authorisation in that conversation turn.**
   Declaring a task, exporting, registering and ingesting are all writes. Planning and
   querying are not.
3. **CellReg must not run on tatchu3** — a run exhausted its RAM and took the database
   server down. MATLAB lives on ibehavegpu1 under `backup_user`.
4. **The worker's dry run is the default.** `--execute` is what writes. There is **no
   `--dry-run` flag**; passing one is an argument error.

## Where things live

| | |
|---|---|
| Code | `adamacs_ingest` on `main` — `adamacs/schemas/cell_registration.py`, `adamacs/ingest/cellreg*.py`, `adamacs/workers/cellreg_worker.py`, `adamacs/matlab/cellreg_run.m` |
| Notebooks | `notebooks/17_cell_registration_ingest.ipynb` (parse + query + worked example), `notebooks/18_cell_registration_trigger.ipynb` (edit one cell, run the rest) |
| MATLAB | `$HOME/MATLAB/bin/matlab` on ibehavegpu1, R2023a Update 8 |
| Toolbox | `$HOME/CellReg` on ibehavegpu1 and tatchu3, pinned to zivlab/CellReg `70668513c665` |
| Slack | `notify_cellreg.py` in **troselab/SOPs** branch `ibehavegpu1`, cron `5,35`, posts site/scans/curation ids every 30 min |
| Wiki | `Database/Schemas/roselab_cell_registration.md` |

## Connecting — two traps before any query

```python
import os, sys, pathlib
import datajoint as dj

# 1. PIN THE PACKAGE ROOT. `adamacs` is dev-installed in several envs pointing at
#    ~/adamacs, which does NOT contain the cell_registration code; the resulting
#    "cannot import name 'cellreg_plan' from 'adamacs.ingest'" names that other checkout
#    and gives no hint that the wrong repo is on sys.path.
ROOT = pathlib.Path(os.path.expanduser("~/adamacs_ingest"))
sys.path.insert(0, str(ROOT))

# 2. LOAD THE CONFIG FIRST. adamacs reads custom.database.prefix at IMPORT, and without
#    credentials dj.conn() prompts for a username on stdin (under cron: EOFError).
#    tatchu3 has it only at ~/adamacs/dj_local_conf.json; ibehavegpu1 also has one in the
#    repo root.
dj.config.load(os.path.expanduser("~/adamacs/dj_local_conf.json"))

import adamacs
assert adamacs.__file__.startswith(str(ROOT)), adamacs.__file__   # else RESTART the kernel
```

**Do not set `transaction_read_only` and then import `adamacs.pipeline`.** The import
performs an INSERT (`adamacs/schemas/subject.py` inserts a Lab row), fails part way on a
read-only connection, and leaves `adamacs.schemas` without its submodule attributes —
surfacing much later as a baffling `ImportError`. Plan with SQL via `dj.conn()`; import the
pipeline only when you intend to write.

## 1. Plan — which scans and curations become columns

`adamacs/ingest/cellreg_plan.py` is read-only and does the judgement that used to be manual:

```python
from adamacs.ingest import cellreg_plan as P
conn = dj.conn()

plan = P.plan_site_task(conn, subject="ROS-1881", site="sess9FPXNI8U",
                        paramset_idx=10, curation="max",
                        stimulus="Moving_Gratings")   # or scans=[...] / stimulus=None
P.print_plan(plan)          # columns kept, and every exclusion WITH ITS REASON
keys = P.curation_keys(plan)  # ordered imaging.Curation keys for the task
```

- `site` is the `same_site_id` — a site's **first** session id, matching the cellreg folder
  convention (`LE_ROS-1881_sess9FPXNI8U_Gratings_CR`) and `roselab_session.session_same_site`.
- `curation="max"` takes the **highest `curation_id`** per scan (the most recently curated);
  `"min"` or an integer also work. Whatever is chosen is RECORDED, never inferred later.
- **Always read the excluded list.** That is where the surprises are.

### Finding cohorts across the lab

"All of Leon's latest gratings recordings" is a query, not a guess:

```sql
-- sites, their size, and whether anything is registered yet
SELECT ss.same_site_id, MIN(s.session_datetime), COUNT(DISTINCT c.scan_id) AS n_scans,
       COUNT(DISTINCT rs.scan_id) AS n_registered
FROM roselab_session.session_same_site ss
JOIN roselab_session.session s ON s.session_id = ss.session_id
JOIN roselab_imaging.curation c ON c.session_id = ss.session_id
LEFT JOIN roselab_cell_registration.cell_reg_run__session rs
       ON rs.session_id = c.session_id AND rs.scan_id = c.scan_id
GROUP BY ss.same_site_id HAVING n_scans >= 5 ORDER BY 2 DESC
```

Then loop `plan_site_task` over the sites and report the plans **before** declaring
anything. Registering many sites is many tasks, one per site per stimulus — never one task
spanning sites.

## 2. Declare the task  — WRITES

```python
from adamacs.pipeline import cell_registration as cr
from adamacs.ingest import cellreg_export as EX

settings = {"input_format": "Suite2p", "alignment_type": "Translations and Rotations",
            "initial_registration_type": "Spatial correlation",
            "registration_approach": "Probabilistic",
            "model_type": "Spatial correlation",      # SINGULAR -- driver uses strcmp
            "maximal_distance": 12.0, "p_same_threshold": 0.3,
            "microns_per_pixel": plan["microns_per_pixel"],
            "reference_session_index": max(1, len(plan["columns"]) // 2)}   # 1-based

pset = cr.CellRegParamSet.register(f"prob_tr_p0.3_{cr.config_hash(settings)[:8]}",
                                   settings, matching_method="cellreg_probabilistic")
site_key = {"same_site_id": plan["same_site_id"]}   # ADR-R8: the site IS the lab's
                                                   # samesite id; subject is NOT in the key
out = EX.export_dir_for({**site_key, "subject": plan["subject"]},   # subject is in the PATH
                        label="Gratings", task_id=0)
task = cr.CellRegTask.auto_generate_entries(site_key, keys, paramset=pset,
                                            task_mode="trigger", output_dir=str(out),
                                            label="Gratings")
cr.CellRegExport.populate(task, suppress_errors=False)
```

`cellreg_paramset_id` is `varchar(40)` — keep slugs short. `assert_safe_target` refuses any

**`auto_generate_entries` will refuse the task** if any declared session does not belong to
`same_site_id` according to `session_same_site` — a registration spanning two sites is not a
registration. Do not work around it by picking a different site id: regroup the sessions
upstream in the ingest GUI, which is the only place that judgement belongs. Note the
foreign key onto `session.SessionSite` does *not* give you this: every session is inserted
there as its own site, so the FK catches a typo and nothing more.

To find which site a historical run covered, read `CellRegRun.Site` (derived from the run's
resolved columns; no row means its columns did not agree on one site, or some column has no
`SessionSameSite` row at all — never read absence as "unknown").

path inside a suite2p tree, any file named `Fall.mat`, and any overwrite.

## 3. Run CellReg, then ingest — on ibehavegpu1

```bash
ssh ibehavegpu1 && sudo -u backup_user -i
cd ~/adamacs_ingest
./scripts/cellreg_worker_ibehavegpu.sh --once             # dry: reports, writes nothing
./scripts/cellreg_worker_ibehavegpu.sh --execute --once    # runs MATLAB, registers, ingests
```

Four stages per pass: export → MATLAB → **`register_results`** → `CellRegResult.populate` +
`MaskCoordinates.populate`. Stage 3 is not optional: `CellRegResult` is keyed on
`CellRegRun`, so without it the worker runs a registration and ingests none of it while
reporting success.

## 4. Verify — the declaration is the test

```python
cols = (cr.CellRegRun.Session & {"cellreg_run_id": rid}).fetch(as_dict=True)
# every column should read resolution_method == 'manifest'
# Member/MaskCoordinates per scan should equal that curation's own mask count
```

`resolution_method` tells you what KIND of claim an attachment is:

| value | meaning |
|---|---|
| `manifest` | **declared** before the run — the only one that is provenance |
| `roi_group_match` | evidence: unique exact content match against a curation's `roi_group` |
| `mask_set_identical` | convention: candidates were byte-identical in mask set, so only the recorded `curation_id` was chosen |
| `log_file` | weak: the scan had exactly one curation, so the column was forced |

## 5. Querying cross-session cells

`CellRegResult.Cell` is the identity (no session in its key). `.Member` bridges cell ×
curation → `mask`, lining up attribute-for-attribute with `Segmentation.Mask`, so joins into
`Fluorescence.Trace` / `Activity.Trace` need no renaming. Absence of a `Member` row means
the cell was not found in that session — never a sentinel. Partial tracking is the normal
case, not the exception.

Aligned footprints are regenerated on demand by `adamacs/ingest/cellreg_footprints.py`
(`CellRegRun.Session.aligned_footprints()`), verified bit-exact; `spatial_footprints_corrected`
is deliberately not stored (~2.9 GB per run as doubles).

## Traps that silently produce wrong answers

- **Mixed fields of view.** CellReg takes ONE `microns_per_pixel` per run and only
  translates and rotates — it never rescales. A site can hold scans at different zooms
  (ROS-2101 has four at 466.13 µm and two at 932.26 µm over the same 512 px). `plan_site_task`
  groups by FOV and keeps the largest group; do not override that without a reason.
- **Two `microns_per_pixel` conventions, ~1 % apart.** `ScanInfo` gives 621.51/512 = 1.2139
  for ROS-1881; the January 2026 log records a 615.6 µm FOV, i.e. 1.2. It scales every
  centroid distance. UNRESOLVED — check the log of any earlier run on the same site before
  comparing results across vintages.
- **`scan_notes` is not the stimulus.** It is free text, often literally `'none'`. Stimulus
  comes from `event.Event`: gratings are `Moving_Gratings:<direction>`. And `event_type` is
  not a tidy vocabulary — Bonsai VR experiments write the texture path and timestamps into
  the string, so "grating" appearing there does NOT make it a gratings scan.
- **DataJoint table names are not derivable from class names.** Tier prefixes mean
  `event.Event` is `roselab_event._event` and `ScanInfo.Field` is `_scan_info__field`.
  Resolve from `information_schema`; a wrong guess is an empty result at worst.
- **A log-recovered paramset cannot drive a new run.** It carries CellReg's DERIVED `n_bins`
  and `initial_threshold` plus the schema's vocabulary; `settings_payload` refuses it. A
  trigger task needs a second paramset in the driver's vocabulary, so a task's paramset and
  its run's paramset legitimately differ.
- **The driver's settings are `strcmp`-compared.** The log prints "Probabilistic modeling"
  and "Spatial correlations" (plural) while the driver tests "Probabilistic" and "Spatial
  correlation" (singular). A wrong literal does not raise — the branch never fires and the
  run dies after Stage 5 with no result file. Use `cellreg_export.from_log_value()`.
- **`p_same_registered_pairs` has a leading singleton axis** — `(1, n, n)` per cell. Missing
  that once suppressed every `PairQuality` row and left `min_p_same` NULL everywhere, with
  no error. Guarded by `cellreg.p_same_is_consistent()` and seven regression tests.
- **mat73 is ~140× slower than raw h5py** for per-cell cell arrays (94.9 s vs 0.68 s for one
  run's `p_same`). Use `cellreg.read_p_same_fast()`. Never override `NEEDED_FIELDS` by
  enumerating a file's fields.
- **`n_pairs_nonfinite` counts only values that were READ.** Where `p_same` is absent or
  shape-inconsistent it is NULL, not a count. Query `IS NULL` as well or the unreadable run
  drops out of the reconciliation.
- **A run is identified by its file CONTENT**, not its path: scan folders hold byte-identical
  backups, so registration is keyed on sha256.
- **Deleting one scan's curation is refused by default** — the part-table design makes the
  cascade raise. `force_masters=True` deletes the whole run across all sessions (the
  intended semantics). **Never `force_parts=True`**: it half-deletes and orphans
  `PairQuality`.

## State as of 2026-10-23

102 runs · 1,106 resolved columns (317 `roi_group_match`, 537 `mask_set_identical`, 248
`log_file`, 4 `manifest`) + 164 unresolved = 1,270 = `SUM(n_sessions)` · 599,479 cells ·
1,461,170 memberships · 4,643,625 `p_same` rows · `RoiGroupProjection` **0 rows**.
