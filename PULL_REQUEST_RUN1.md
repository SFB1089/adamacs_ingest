# Run 1: register and parse the 101 existing CellReg runs

**Branch:** `feat/cell-registration-schema` → `main` (base `d42a129`, the merge of #16)

Follows **#16**, which declared `cell_registration` and left it deliberately empty. This PR
is the first ingest: it parses every CellReg result file already on the server and records
it, in the chronological order the registrations were produced. Tobias authorised the write
on 2026-10-20.

> **`roi_group` is not written by any of this.** Verified by fingerprint either side of
> every write: 3,276,142 rows, 1,041,040 non-zero, sha256
> `da8e33a5108af402eca03e962a24c4dbf1f2df084d988b2c55075a8e0e380c30`, identical before and
> after. `RoiGroupProjection` and `RoiGroupSnapshot` are **0 rows**. The registrar touches
> `roselab_imaging` with exactly four statements, all `SELECT`.

---

## 1. What was ingested

114 result files on disk collapse to **102 distinct runs by sha256** — the duplicates are
byte-identical backups scattered across scan folders, which is why a run is keyed by the
*content* of its result file and the other paths live in `CellRegRun.Copy`. One file is
unreadable (`OSError: could not read bytes`), leaving **101 runs** spanning 2023-07-06 to
2026-10-18 across ~16 subjects.

| Table | Rows |
|---|---:|
| `CellRegRun` | 101 |
| `CellRegRun.Session` (resolved columns) | 565 |
| `CellRegRun.UnresolvedColumn` | 701 |
| `CellRegRun.Quality` | 101 |
| `CellRegRun.Copy` | 12 |
| `CellRegParamSet` | 68 |
| `CellRegResult` | 101 |
| `CellRegResult.Cell` | 597,212 |
| `CellRegResult.Member` | 717,053 |
| `CellRegResult.PairQuality` | 4,641,778 |
| `MaskCoordinates` | 717,053 |
| `RoiGroupProjection`, `RoiGroupSnapshot` | **0, 0** |

Reconciliations, all checked against the database rather than asserted:

- 565 + 701 = 1,266 = `SUM(n_sessions)` over the 101 runs
- every run has a `CellRegResult`; every resolved column has `MaskCoordinates`
- 4,807,688 expected pairs − 165,910 non-finite = **4,641,778 stored**, matching the count
  predicted from the index maps *before* any ingest

## 2. How a column found its curation — and the 535 that could not

`CellRegRun.Session` is keyed on `imaging.Curation`, so every resolved column needs one.
The historical exports cannot supply it: they copied `<ScanPath>/suite2p/plane0/Fall.mat`,
a per-scan filesystem path with no curation in the lookup, and 696 of 743 multi-curation
groups have byte-identical mask sets, so Jaccard scores 1.0 against every candidate.

What *does* discriminate is the output. A column of `cell_to_index_map` implies a
`roi_group` vector for its session's masks, and that vector can be compared
element-for-element against what a candidate curation actually holds. A **unique exact**
match identifies the curation by content. New `resolution_method` value `roi_group_match`.

| Columns | Outcome |
|---:|---|
| 317 | `roi_group_match` — unique exact content match |
| 248 | `log_file` — the scan has exactly one curation, so forced |
| 535 | **no curation matches at all** |
| 69 | tie — two byte-identical curations both match exactly |
| 65 | empty column |
| 24 | no scan id recoverable from the log |
| 8 | centroid count contradicts the curation (demoted, §4) |

**The 535 are the result of this PR, not a defect in it** — but they are narrower than
"lost". `roi_group` is a `smallint` outside the primary key that each run overwrote in
place, so for any curation only the *last* run to target it survives, and a column of any
earlier run matches nothing there. What that costs, measured:

- **The clusters themselves survive for every run.** 597,212 registered cells with their
  scores and 4,641,778 `p_same` rows are in the schema, parsed from the files; the result
  files were never the thing at risk.
- **What is thin is the join down to masks.** 717,053 of the 1,540,370 memberships implied
  by the index maps (46.6 %) have a curation key and so reach `Segmentation.Mask`,
  `Fluorescence` and `Activity`. 254,150 cells (42.6 %) have every session attached.
- **Most of the shortfall is label ambiguity, not incompatibility.** Of the 701 unresolved
  columns, **562 (80 %) match the mask count of *several* candidate curations** — the file's
  segmentation is consistent with the scan, only the curation cannot be named, and since 696
  of 743 multi-curation groups are byte-identical the masks are the same whichever is named.
  **50 columns (7 %) match no curation's mask count at all** — those are genuine
  incompatibility, the segmentation changed. 65 are empty and 24 have no scan id.

So a tie-break convention would recover most of the 53 %. It is deliberately not applied
here, because choosing among provably equivalent candidates is a convention and choosing
among different ones would be inventing provenance — see §8.1.

The 69 ties are the honest limit of the method: when two curations are byte-identical in
masks **and** `roi_group`, no analysis separates them. They stay unresolved with their
candidates attached as evidence rather than being attached to an arbitrary one.

## 3. `log_file` resolution is not evidence

This is the finding most likely to bite a reader of the table.

`resolution_method='log_file'` means only one candidate curation existed for that scan. It
does **not** mean that curation is the right one. `MaskCoordinates.make()` refuses a column
when `centroid_locations_corrected` has a different number of rows than the curation has
masks — *"the input file is not this curation"* — and it rejected 8 of the 573 columns run 1
first wrote. Every one was `log_file`:

| Fault | Columns | Reading |
|---|---|---|
| off-by-one | 1034 vs 1035, 1280 vs 1279, 1115 vs 1116 | the curation gained or lost an ROI after the run |
| wholesale | 316 vs 910, 370 vs 1088, 336 vs 1064 | the input file is a different segmentation entirely |

They were demoted to `UnresolvedColumn` with both counts recorded (573 → 565 resolved,
693 → 701 unresolved), and `cellreg_register.centroid_counts()` now performs the same
comparison **before** a `Session` row is written, so the contradiction can never be stored.

## 4. Bugs found, all by testing against numbers known beforehand

The acceptance target is the 11-session ROS-1881 gratings run, reproduced independently
*before* this code existed: 4,107 cells, 38,921 pairwise links, 306 cells in all 11 sessions.

**4.1 `p_same` leading singleton axis — silent, and it would have voided match quality.**
`p_same_registered_pairs` arrives as `(1, n, n)` **per cell**. `pair_quality_rows` always
stripped that axis; the inline consistency check did not, so it set `p_same_consistent =
False`, and the `if consistent` guard then suppressed **every** `PairQuality` row and left
`min_p_same` / `median_p_same` NULL on every cell — with no error anywhere. The documented
selector `min_p_same > 0.9` would have returned nothing across the whole table. Extracted as
`p_same_is_consistent()` with 7 regression tests
(`tests/test_cellreg_pair_quality.py`).

**4.2 Parameter-set slugs collided.** 86 of 101 runs share approach, alignment and `p_same`
threshold while differing in `maximal_distance_um`, `n_bins`, `initial_threshold` or
`model_type`. `_HashedConfig` refused them — correctly, since re-registering a slug with
different parameters must never silently update. Slugs now carry 8 hex of the content hash
(`prob_transrot_p0.5_fc5bbba5`).

**4.3 mat73 is 140× slower than h5py for per-cell cell arrays.** Reading
`p_same_registered_pairs` for one 4,107-cell run: **94.9 s** via mat73 against **0.68 s**
dereferencing the same 4,107 object references with h5py, values identical. Across 597,212
cells this was the whole ingest — the projected runtime fell from ~14 hours to **20.7
minutes**, and the acceptance run from ~15 minutes to 16.6 s. `read_p_same_fast()` does the
raw read for v7.3 files and returns `None` otherwise, so the single v5/v7 file and any
environment without h5py fall back to the existing loader.

**4.4 Missing-data semantics for pair quality.** CellReg emits `NaN` `p_same` for pairs the
index map calls the same cell, so `PairQuality` is legitimately shorter than `C(k, 2)`. New
`CellRegResult.n_pairs_expected` and `n_pairs_nonfinite` record the difference, so it can
never be read as a parser fault: on ROS-1881, 38,921 implied = 38,767 stored + 154 `NaN`.

## 5. Acceptance

Eleven checks on the ROS-1881 run, all exact:

| | expected | got |
|---|---:|---:|
| `parse_ok` / `p_same_consistent` | True | True |
| cells | 4,107 | 4,107 |
| memberships | 13,532 | 13,532 |
| mask coordinates | 13,532 | 13,532 |
| pair-quality rows | 38,767 | 38,767 |
| `n_pairs_expected` | 38,921 | 38,921 |
| `n_pairs_nonfinite` | 154 | 154 |
| stored + non-finite | 38,921 | 38,921 |
| cells in all 11 sessions | 306 | 306 |
| `min_p_same` populated | 4,107 | 4,107 |

Rehearsed end to end in the isolated schema `tobiasr_cellreg_test` before production: 101/101
runs, 0 failures, `roi_group` digest identical either side. The rehearsal is what caught 4.2.

Suite: **160 passed, 43 skipped**. CI's lint selection
(`ruff check --select E9,F63,F7,F82 adamacs tests examples`) passes.

## 6. Schema changes (additive, applied to both schemas)

- `CellRegRun.Session.resolution_method` gains `roi_group_match`
- `CellRegResult` gains `n_pairs_expected`, `n_pairs_nonfinite` (both nullable)

Both applied by `ALTER TABLE` while the tables were empty. No table was dropped, no column
removed, no key changed; nothing outside `roselab_cell_registration` and
`tobiasr_cellreg_test` was altered.

## 7. What is NOT claimed

- **No projection onto `roi_group` has been performed.** `RoiGroupProjection` and
  `RoiGroupSnapshot` are empty by design, and writing them remains a separate, separately
  authorised step.
- **The 535 unmatched columns cannot be tied to a curation by content**, because
  `roi_group` no longer holds their output. Their clusters and `p_same` ARE in the
  schema; what is missing is the mask-level join. 562 of the 701 unresolved columns
  are mask-count-compatible with several curations, so a declared tie-break would
  recover most of them; 50 are incompatible with any and are genuinely beyond reach.
- **The 69 ties are not resolved**, and resolving them needs a convention Tobias declares,
  not an analysis.
- **The worker has still never processed a real task.** Run 2 — a task-driven run through
  `CellRegTask` → export → MATLAB → ingest — has not happened.
- **Deletion semantics are still read from the FK graph**, not from a live delete.
- **Two runs' logs contradict their index maps** (shapes `(1017, 2)` and `(6352, 11)` against
  a stated 10 sessions). The orientation falls back to the shape heuristic with the
  discrepancy recorded in `run_note`; why the logs disagree is unexplained.

## 8. Unresolved decisions

1. **A tie-break convention for the 69 tied columns**, or leaving them unresolved forever.
2. **Whether `log_file` resolution should exist at all** now that 8 of 256 proved wrong —
   the alternative is to treat a single candidate as unresolved unless the centroid count
   confirms it, which is what the new guard effectively does.
3. **The `ImagesAndGratings` run** (`cellRegistered_20260811_130701.mat`) is the newest file
   of all, the only one of its type, not superseded, all 14 columns populated — and matches
   nothing. Its identities were never written to `roi_group`.
4. **`frac_uncertain_pairs` differs between vintages** (0.19 vs 0.08 for the same site)
   before `CellRegRun.Quality` can be compared across runs.
5. Items 1–7 of #16 §9 that remain open: ADR-004 departure, `min_p_same` as a stored summary,
   transaction size, `same_site_id` upstream, who may insert a `RegistrationCuration`,
   licence hygiene, pinning a CellReg release matching Leon's.

## 9. Documentation

- **`notebooks/17_cell_registration_ingest.ipynb`** — run 1 end to end. Its census cell
  **re-derives** every number from the live schema and checks the total against
  `SUM(n_sessions)`, rather than restating counts; the notebook published stale counts twice
  during this work and re-deriving removes the failure mode.
- **`notebooks/18_cell_registration_trigger.ipynb`** — triggering a *new* run: plan a
  `CellRegTask`, declare the curation manifest, export inputs from the database, run the
  worker on ibehavegpu1, ingest the result. Every write is behind `RUN_WRITES = False`. New
  §6b carries the run-1 measurements as the argument for the manifest.
- **Wiki** — `trose-neuro/roselab-wiki`, branch `docs/cell-registration-schema`, head
  `8589b98`, pushed. (#16 §10 said it was unpushed; that was true when written and is not
  true now.)
