# The cell-registration schema — design review

**Status.** Declaration only. Nothing has been declared on the server and nothing has been
inserted. `roselab_registration` does not exist; the verification is in §7.

**Scope of this pass.** Cell registration only, strictly additive. `roi_group` is not
touched, not altered, and not deprecated by this change.

**Evidence.** Everything numeric below was measured on 2026-10-15, read-only, and is
reproducible from two artefacts: the per-file inventory (`cellreg_mat_inventory.csv`, 108
files) and its raw field dump (`cellreg_mat_inventory_raw.json`). Live-database facts came
from an `information_schema` probe under `SET SESSION transaction_read_only = 1`.

---

## 1. Scientific purpose

Longitudinal imaging claims rest on one proposition: *this cell on day 20 is that cell on
day 0.* In ADAMACS that proposition is currently carried by a single column,
`roselab_imaging.__segmentation__mask.roi_group` — `smallint(6)`, and **not part of the
primary key** (`column_key = ''`, verified). It is overwritten in place by whichever
CellReg ingest ran last. There is no record of which CellReg run produced the current
values, what parameters it used, how good the matches were, who checked them, or what the
column held before.

This is not hypothetical damage. The wiki records it as issue **D1**: counts recorded by
an analysis (111 and 146 cells at two sites) disagree with the live database (116 and 120)
because the `roi_group` state postdates that analysis, and **T1** notes the delta can no
longer be reconstructed. The lab has already paid for this twice over:

- `adamacs_leon/helpers/pipeline_blocks.py` contains
  `realign_global_ids_to_current_roi_group()`, a function whose entire purpose is to
  repair identities broken by a re-ingest. Its own docstring states the problem: *"After
  re-ingesting roi_group from a newer CellReg run, those global ids no longer match
  DataJoint's roi_group."*
- To repair them it must **guess** which CellReg column corresponds to which scan, by
  Jaccard overlap of mask-id sets with a `min_jaccard=0.5` acceptance, then **majority-vote**
  the current `roi_group` across sessions, returning `-1` where the vote fails.

A schema whose consumers must guess at their join keys is the defect. The purpose of this
schema is to make cross-session cell identity a first-class, versioned, queryable entity
with its own provenance and its own quality measurements, so that no analysis ever has to
guess again, and so a re-registration is a *new row* rather than a silent mutation.

The one genuinely reassuring fact, and the foundation of the whole design: the docstring
above also establishes that **per-session local `mask` ids are stable across re-ingests**
("re-ingest only rewrites roi_group, never the mask numbering"). So
`(session_id, scan_id, paramset_idx, curation_id, mask)` is a durable anchor, and a new
identity layer can be built on it additively.

---

## 2. Scope

**Global, not project-scoped.** Any subject, any paradigm, any number of runs per site.
This is forced by the data, not by ambition — the inventory covers **18 distinct subjects**
and **5 data owners** (leonk 99 files, aeltona 6, kevinh/rukhun/shanm 1 each — 108 in
total; of the 107 that parse, leonk accounts for 98), with
registration runs spanning **2 to 45 sessions** (modal 11, n = 49) and **126 to 25,724**
registered cells. Any design that assumes one project's cohort shape, a fixed session
count, or a single owner's directory convention is already wrong on the existing data.

**The matching method is a lookup, not an assumption.** CellReg is what the lab uses today;
it is not the only thing it could use (suite2p-native matching, manual matching, and
footprint-correlation variants all exist). `MatchingMethod` is a `Lookup` so that a second
method becomes a row rather than a schema change, and so that a query can ask "which
identities came from probabilistic CellReg" without string-matching a path.

**Thresholds are deliberately out of this schema.** Registration stores *continuous match
quality* — `cell_score`, `p_same`, exclusivity, false-positive fraction. Deciding which
cells are good enough is a QC and cohort-selection act, it is paradigm-dependent, and it
gets re-decided far more often than a registration gets re-run. Putting an eligibility
boolean here would bake one project's cut into shared data and force a re-populate every
time somebody disagreed with it. The threshold tables belong with QC and cohorts, keyed to
their own versioned parameter sets, consuming the columns below.

---

## 3. Table grain and primary keys

**Ten master tables and nine part tables — 19 definitions**, counted by the validator from
the module rather than from this sentence. Schema `db_prefix + 'cell_registration'`,
deferred activation (nothing declared on import), following `disk.py`.

### 3.0 Where "cross-session" lives — read this before the keys

A registered cell spans sessions, so no table that represents one may be keyed by a single
session. The design separates the three grains explicitly:

| Entity | Grain | Key | Spans sessions? |
|---|---|---|---|
| `CellRegRun` | one registration execution | `cellreg_run_id` | — |
| `CellRegRun.Session` | **input manifest**: one row per participating curation | run + `imaging.Curation` | one row *per* session |
| `CellRegRun.UnresolvedColumn` | a column that could not be tied to a curation | run + `column_index` | — |
| **`CellRegResult.Cell`** | **the cross-session identity** | run + `cell_index` — **no session, no curation** | **yes, by construction** |
| `CellRegResult.Member` | the many-to-many bridge | `Cell` × `Session` | the relation itself |
| `CellRegResult.PairQuality` | match quality for one cell between two sessions | `Cell` + two column indices | pairwise |
| `StableCell` | identity across *runs* as well as sessions | site + `stable_cell_id` | yes |

So "is this cell in session X" is a `Member` row, "which sessions is this cell in" is
`Member & cell_key`, and `n_sessions_present` on `Cell` is the denormalised count of exactly
that. Putting `imaging.Curation` in `Session`'s key (ADR-R5) does **not** make the
registration per-curation — `Session` enumerates the run's inputs the way
`roselab_leondrift.RunSet.Scan` enumerates a cohort's scans, while the cell identity sits in
`Cell`, one level up and session-free.

The rule the schema enforces: **`Cell` never gains a session-scoped attribute.** If a
quantity needs one, it belongs on `Member` (per cell per session) or `PairQuality` (per cell
per session pair), never on `Cell`.

### 3.0b How this links to scan, session and the suite2p curation

There is exactly one attachment point to the existing pipeline: `CellRegRun.Session` takes
`-> imaging.Curation`, whose primary key is
`(session_id, scan_id, paramset_idx, curation_id)` — verified on the live server. That one
foreign key carries the whole upstream identity, including **which suite2p run**
(`paramset_idx`, the `imaging.ProcessingParamSet` lookup) and **which curation of it**
(`curation_id`). Nothing else in this schema references upstream at all.

```
subject.Subject
 └─ session.Session (session_id)
     └─ scan.Scan (scan_id)
         └─ imaging.ProcessingTask → Processing → Curation (paramset_idx, curation_id)
              ├─ imaging.Segmentation.Mask        (+ mask)
              ├─ imaging.Fluorescence.Trace       (+ mask, fluo_channel)
              ├─ imaging.Activity.Trace           (+ extraction_method, mask, fluo_channel)
              └─ cell_registration.CellRegRun.Session   ← the only new edge
                    └─ CellRegResult.Member        (+ cell_index → mask)
```

Because `Member` inherits that key and adds `mask`, it lines up attribute-for-attribute with
`Segmentation.Mask`'s primary key, so the joins need no renaming:

```python
from adamacs.pipeline import imaging
from adamacs.schemas import cell_registration as creg

run = creg.CellRegRun & {'cellreg_run_id': '<40-hex>'}

# every (scan, mask) this registered cell is, across all its sessions
cell = creg.CellRegResult.Cell & run & 'cell_index = 42'
members = creg.CellRegResult.Member & cell          # one row per session it appears in

# ... and the suite2p masks those rows point at
masks = imaging.Segmentation.Mask & members.proj()

# ... and their traces, with the curation pinned implicitly by the join
traces = imaging.Fluorescence.Trace & members.proj() & 'fluo_channel = 0'

# cells tracked in at least 5 sessions whose weakest pairwise match is still good
good = (creg.CellRegResult.Cell & run
        & 'n_sessions_present >= 5' & 'min_p_same > 0.8')
```

Two consequences worth stating, because they are the point of the exercise:

- **`paramset_idx` and `curation_id` are pinned by construction.** The documented
  anti-pattern of mixing curation ids implicitly cannot happen through this path: a `Member`
  row names its curation, so a join through it is automatically restricted. The current
  `roi_group` path has no such property — `roi_group` is a bare column on
  `Segmentation.Mask` and says nothing about which run assigned it.
- **A registration is tied to the suite2p run it was computed from.** If a scan is
  re-processed under a new `paramset_idx`, or re-curated under a new `curation_id`, the
  existing registration still points at the old one — correctly, because that is what it was
  computed on. A registration for the new curation is a new `CellRegRun`, which is the
  versioning behaviour the brief asks for and the thing `roi_group` cannot express at all.

### 3.1 Reference and parameters

```
MatchingMethod (Lookup)
    matching_method     : varchar(32)      # 'cellreg_probabilistic', 'cellreg_simple_threshold', ...
    ---
    software            : varchar(64)      # 'CellReg'
    software_reference  : varchar(255)     # DOI or repository URL
    method_description  : varchar(512)
```

```
CellRegParamSet (Lookup)                   # immutable, hash-keyed
    cellreg_paramset_id : varchar(40)      # human slug
    ---
    -> MatchingMethod
    params              : longblob         # the parameter dict
    params_hash         : char(40)         # sha1 of canonical JSON
    params_json         : varchar(4000)
    label               : varchar(128)
    created             : datetime
```

Immutability is enforced the way `leondrift` enforces it (ADR-005): a `register()`
classmethod hashes the canonical JSON, and re-registering an existing slug with different
parameters **raises** rather than updating. A changed parameter is a new slug, so every
downstream row's key names the parameters that produced it.

### 3.2 The run

```
CellRegRun (Manual)
    cellreg_run_id      : char(40)         # first 40 hex chars of the result file's sha256
    ---
    -> CellRegParamSet
    subject             : varchar(16)
    result_file         : varchar(512)     # canonical path, one of possibly several copies
    result_file_sha256  : char(64)
    result_file_bytes   : bigint
    result_file_mtime   : datetime
    log_file            : varchar(512)     # '' when absent
    log_file_sha256     : char(64)         # '' when absent
    mat_format          : enum('v7.3_hdf5','v5_v7')
    registration_approach : enum('probabilistic','simple_threshold','unknown')
    alignment_mode      : enum('translations','translations_and_rotations','non_rigid','unknown')
    reference_session_index : smallint     # nullable; 1-based as CellReg reports it
    n_sessions          : smallint
    n_registered_cells  : int
    ingested_at         : datetime
    ingest_host         : varchar(64)
    adamacs_sha         : varchar(40)
    run_note            : varchar(512)
```

**Why the primary key is a content hash.** The inventory found **100 distinct sha256 over
108 paths** — 6 duplicate groups, because scan-folder "Data Backup" copies hold byte-identical
copies of the same run. Keyed on path, one registration would enter the database as up to
three distinct runs; keyed on content, the copies collapse and the extra paths are
provenance. The cost is a key no human can read, which is why `subject`, `result_file` and
`run_note` are there, and why the inventory CSV is the lookup table from path to run.

*Rejected:* `(subject, site, run_datetime)` parsed from the `cellRegistered_<timestamp>.mat`
filename. The duplicate copies share the timestamp too, so it fixes nothing; the timestamp
is a local MATLAB wall clock with no timezone; and `cellRegistered_combined.mat` has no
timestamp at all.

```
CellRegRun.Quality (Part)                  # run-level accuracy, parsed from the log file
    -> master
    ---
    frac_false_positives    : float        # nullable
    frac_false_negatives    : float
    frac_uncertain_pairs    : float
    avg_register_score      : float
    avg_true_positive_score : float
    avg_true_negative_score : float
    avg_exclusivity_score   : float
    model_discrepancy       : float
    avg_fraction_active     : float
    initial_threshold       : float
    final_threshold         : float
    maximal_distance_um     : float
    n_bins                  : smallint
    microns_per_pixel       : float
    quality_source          : enum('log_file','absent')
```

These numbers exist **only** in `logFile_*.txt` — they are in no field of the result
`.mat`. A verified example reads: false positives 0.04, false negatives 0.08, uncertain
cell-pairs 0.06, register score 0.92, true-positive 0.59, true-negative 0.97, exclusivity
0.99, model discrepancy 0.06. This is the run-level match quality the brief asks for, and
it would be lost entirely by an importer that read only the `.mat`.

### 3.3 The scan ↔ column mapping — the table that removes the guessing

```
CellRegRun.Session (Part)
    -> master
    column_index        : smallint         # 0-based column of cell_to_index_map
    ---
    -> [nullable] imaging.Curation         # NULL when the column could not be resolved
    source_file         : varchar(512)     # the converted_*.mat named in the log
    resolution_method   : enum('log_file','mask_jaccard','manual','unresolved')
    mask_set_jaccard    : float            # nullable; the measured overlap
    n_masks_cellreg     : int
    n_masks_db          : int              # nullable
    ambiguous           : bool             # >1 candidate scan above threshold
    alignment_x_translation : float        # nullable
    alignment_y_translation : float
    alignment_rotation      : float        # NULL unless alignment_mode = translations_and_rotations
    zero_padding_x          : float
    zero_padding_y          : float
```

**Superseded by ADR-R5 — see §12.** The shape above keyed on `column_index` with a
*nullable* dependency on `imaging.Curation`. It has been replaced: `CellRegRun.Session` now
takes `-> imaging.Curation` **in its primary key**, with `column_index` as a secondary
attribute under a unique index, and unresolved columns move to a separate
`CellRegRun.UnresolvedColumn` part table. The declarations in the module are the amended
ones; this subsection is kept so the earlier reasoning and its defect stay on the record.

**Resolution is evidence-ranked, not guessed.** The log file lists the sessions in column
order and the converted filenames carry the scan id verbatim —
`converted_Fall_01_scan9FOW97IH.mat`. That is authoritative and is tried first
(`resolution_method = 'log_file'`). Mask-set Jaccard overlap against
`imaging.Segmentation.Mask` is computed **in addition**, always, and stored — so the log
and the independent geometric evidence can be compared, and a disagreement surfaces as a
row rather than as a wrong join. Only where no log exists does Jaccard become the
resolution method, and where more than one scan clears the threshold the column is marked
`ambiguous` and left unresolved. This is strictly better than the current
`realign_global_ids_*` behaviour, which has only the Jaccard evidence and no place to
record it.

### 3.4 The result, its cells, and their quality

```
CellRegResult (Imported)
    -> CellRegRun
    ---
    parse_ok            : bool
    parse_error         : varchar(512)     # '' when ok
    parser_version      : varchar(16)
    source_sha256       : char(64)         # re-hashed at parse time, must equal the run's
    n_cells             : int
    n_sessions          : smallint
    has_score_fields    : bool
    missing_fields      : varchar(255)
    p_same_consistent   : bool             # p_same matrix side == n_sessions
    footprints_storage  : enum('in_file','external_path','absent')
    footprints_path     : varchar(512)     # '' unless external
    parsed_at           : datetime
    compute_seconds     : float
```

All `.mat` reading happens in this table's `make()`: both MAT formats, the MATLAB 1-based →
Python 0-based conversion, `0` interpreted as absent, missing fields written as NULL with a
stated reason, and the source checksum plus parser version recorded on every row.

```
CellRegResult.Cell (Part)
    -> master
    cell_index          : int              # 0-based row of cell_to_index_map
    ---
    cell_score              : float        # nullable
    true_positive_score     : float        # nullable
    true_negative_score     : float        # nullable
    exclusivity_score       : float        # nullable
    in_overlapping_fov      : bool
    centroid_x_registered   : float
    centroid_y_registered   : float
    n_sessions_present      : smallint
    min_p_same              : float        # nullable; weakest present pair — the selector column
    median_p_same           : float        # nullable
```

Per-cell quality is **rows with columns**, not a blob. This is the one change that the
brief's "transparent cell-level QC, not one opaque score" actually requires: in
`roselab_leondrift` the equivalent quantities live inside `longblob` arrays
(`__trial_response.roi_groups`, `Responsiveness.p_resp`, `Tuning.rho` — verified column
types), and a value inside a blob cannot be restricted, joined or selected on. `min_p_same`
is carried as a column precisely because "the weakest link in this cell's chain" is the
selector a longitudinal analysis actually wants.

```
CellRegResult.Member (Part)                # the join bridge
    -> CellRegResult.Cell
    -> CellRegRun.Session
    ---
    mask                : smallint         # local mask id in that scan
```

Grain: **one row per (run, cell, column) where the cell is present.** Absence is encoded as
the absence of a row — which is why `CellRegRun.Session` must exist independently, so
"scan was not in this run" and "cell was not matched in this scan" are distinguishable.
This table is the thing every longitudinal query joins through, and it is what finally
replaces `roi_group` as a join key.

```
CellRegResult.PairQuality (Part)
    -> CellRegResult.Cell
    column_a            : smallint
    column_b            : smallint         # column_a < column_b
    ---
    p_same              : float
```

Grain settled by measurement, not by reading the toolbox (which is not readable from this
account): `p_same_registered_pairs` has a square first entry whose side equals the session
count in **106 of 107** files ([11,11], [15,15], [22,22], …), so it is one
*n_sessions × n_sessions* matrix **per registered cell**.

**Rows were sized before being chosen.** Restricting to pairs where the cell is present in
*both* sessions, a 6-run sample spanning 2–45 sessions gave 602,078 pair rows from 49,397
cells (12.2 per cell), extrapolating to **≈ 10.7 M rows** across all 107 runs. Against
`event.Event` at 1.05 billion and `disk.HeadfixedImputation.gap` at 13.7 M, that is an
ordinary table. The blob alternative was therefore rejected: it saves nothing that matters
and costs queryability. Worth knowing for whoever writes the selectors: most registered
"cells" are singletons — median sessions-per-cell is 1 in four of the six sampled runs, and
the fraction with ≥ 2 sessions ranges from 7.4 % to 95.8 %.

### 3.5 Geometry

```
MaskCoordinates (Imported)
    -> CellRegRun.Session
    mask                : smallint
    ---
    centroid_x_native       : float        # the scan's own pixel frame
    centroid_y_native       : float
    centroid_x_registered   : float        # after this run's alignment to the reference
    centroid_y_registered   : float
```

Both frames, because they answer different questions and are produced by different things.
The registered centroid depends on the run's alignment, so it cannot live on the mask and
must be keyed by run. This is the table that makes an alignment failure visible: a run whose
registered centroids scatter where the native ones do not has a bad transform, and that is
checkable by query instead of by reading a figure.

**Amended after grilling — the native columns are a deliberate copy, and that needs saying.**
`roselab_imaging.__segmentation__mask` already carries `mask_center_x`, `mask_center_y` and
`mask_center_z`, all `int(11)` (verified). So `centroid_*_native` here is a **denormalised
copy of an upstream value**, which is the second-source-of-truth defect this review
criticises elsewhere. It is kept anyway, for one reason that does not apply to the general
case: the upstream mask table is itself re-ingestible. Known issue **T7** records that an
in-place re-ingest under the same `curation_id` is invisible, and the Slack record notes
that on-disk suite2p folders retain only the most recent extraction (Tobias Rose,
2024-08-01, ts=1722492807.667179). Copying the native centroid **as of the registration run**
therefore turns a silent upstream change into a detectable disagreement — the same reasoning
behind `RunSet.Scan.mask_fingerprint`. Note the type change: upstream is integer pixels,
stored here as float, because the registered coordinate is sub-pixel after a rotation.

If the reviewer rejects that argument, the fix is to drop the two native columns and join to
`Segmentation.Mask` — but then nothing detects a re-segmentation, and the `.Member` rows
would silently refer to masks whose geometry has moved.

### 3.6 Stable identity across runs

```
RegistrationSite (Manual)
    subject             : varchar(16)
    same_site_id        : varchar(16)      # the site's FIRST session id, e.g. 'sess9FOTVMAL'
    ---
    site_description    : varchar(255)
    site_id_source      : enum('cellreg_folder','manual','same_site_id')
    created             : datetime
```

**Correction to the wiki, measured 2026-10-15.** The working rules say `same_site_id` is
*"unpopulated — it self-references for every session"*. That is wrong in a way that matters.
Measured on the live server:

| | |
|---|---|
| `roselab_session.session_same_site` rows | **3,460** — one per session, i.e. fully populated |
| self-referencing (`same_site_id = session_id`) | **2,561 of 3,460 (74 %)** — not all |
| distinct `same_site_id` values | 2,562 |
| sites with ≥ 2 sessions | **41**, with up to **66** sessions (`sess9FU52PMO`, `sess9FTUT8MJ` 66; `sess9FVIF9IF` 63; `sess9FS1DQX9` 60; `sess9FXCJN0G` 50) |
| `roselab_session.session_site` rows | 3,503 — i.e. ~941 site rows no session points at |
| session ids lacking the `sess` prefix | 12 of 3,460 (legacy) |

So the column is populated and *works* for the 41 sites somebody filled in. The real defect
is narrower and more interesting: **it defaults to self-reference, so "never assigned to a
site" is indistinguishable from "a site of exactly one session."** That is the same
missing-data confusion this whole review is about, one table upstream.

A local site entity is therefore still right — but for a different reason than first stated.
Not "the column doesn't work", but "the column cannot say *unknown*", and 74 % of its rows
are in that ambiguous state. `site_id_source` records which case a row came from, and
`same_site_id` is declared `varchar(12)` to match upstream exactly so that a repaired
upstream column can be adopted as a real foreign key without a type migration.

**Amended after grilling — the label is not invented.** The lab already has a site
identifier and it is the site's **first session id**. The CellReg output convention is
`datajoint-data/leonk/analysis/cellreg/<samesiteID>` (Leon Kremers / Tobias Rose,
2024-05-24, ts=1716542821.819479), and the real directories follow it:
`LE_ROS-1841_sess9FOTVMAL_Gratings_CR` carries `sess9FOTVMAL`. `roselab_leondrift` uses the
same convention — `RunSet.Site.same_site_id : varchar(16)` holding values such as
`sess9G1EB7EE`. So the attribute is named `same_site_id`, carries the same values the rest
of the lab already uses, and `site_id_source` records whether it was read from the folder
name, set by hand, or (when the broken column is eventually populated) taken from upstream.
This keeps the new schema speaking the lab's existing vocabulary instead of minting a
parallel one, and it means that fixing `same_site_id` upstream later becomes a
reconciliation rather than a migration.

```
StableCell (Manual)
    -> RegistrationSite
    stable_cell_id      : int
    ---
    minted_at           : datetime
    minted_from_run     : char(40)

StableCellMap (Manual)
    -> StableCell
    -> CellRegResult.Cell
    ---
    mapping_method      : enum('mask_set_exact','mask_set_majority','manual')
    support_n_masks     : smallint
    conflict            : bool
```

This is `realign_global_ids_to_current_roi_group()` turned into data. Instead of
recomputing a majority vote inside every analysis, the mapping from a durable identity to
each run's `cell_index` is computed once, stored with the evidence that produced it, and
curatable when it conflicts. A re-registration adds rows; it never changes old ones.

### 3.7 Curation — append-only

```
RegistrationCuration (Manual)
    -> CellRegRun
    curation_id         : int
    ---
    curator             : varchar(64)
    curated_at          : datetime
    curation_note       : varchar(512)

RegistrationCuration.CellDecision (Part)
    -> master
    -> CellRegResult.Cell
    ---
    decision            : enum('accept','reject','split','merge','uncertain')
    reason              : varchar(255)
```

A curation is a version, not an edit. Disagreeing with an earlier judgement means a new
`curation_id`; the old decisions stay readable, which is what "preserve previous results
rather than silently overwriting them" means in practice. Note the deliberate omission:
there is no `curated_roi_group` output column. A curated cell *set* is a restriction over
these decisions, computed when asked, not a second mutable identity column — the mistake
this whole schema exists to undo.

### 3.8 Preserving the legacy column

```
RoiGroupSnapshot (Manual)
    snapshot_id         : char(40)
    ---
    taken_at            : datetime
    taken_by            : varchar(64)
    note                : varchar(512)
    n_rows              : int
    fingerprint         : char(40)

RoiGroupSnapshot.Mask (Part)
    -> master
    session_id          : varchar(12)
    scan_id             : varchar(12)
    paramset_idx        : smallint
    curation_id         : int
    mask                : smallint
    ---
    roi_group           : smallint
```

The mapping table the brief requires for an identifier change, and the only way to stop
losing history that is currently being lost on every re-ingest. It is additive, it reads
production without writing it, and it makes D1 answerable going forward even though it
cannot answer it retrospectively.

---

## 4. Dependencies and provenance

**Upstream.** `imaging.Curation` (hence `scan.Scan`, `session.Session`, `subject.Subject`)
through the nullable secondary dependency on `CellRegRun.Session`, and
`imaging.Segmentation.Mask` read — not referenced — for mask sets and native centroids.

**Provenance recorded on every run:** result-file path, sha256, byte size and mtime; log-file
path and sha256; MAT format; parser version; parameter-set slug and its parameter hash;
ingest timestamp, ingest host and `adamacs` commit. The checksum is re-computed at parse
time and compared with the run's, so a file edited between registration and parsing is
caught rather than trusted.

**The parameter-provenance gap, stated plainly.** The result `.mat` contains **no
parameters at all** — 14, 15 or 16 data fields and nothing else (21 files have 14, 85 have
15, exactly 1 has 16; the extra fields are `alignment_rotations` and, in that single file,
`file_names`. None carries a setting). Parameters live in `logFile_*.txt` (112 found
under the data tree) and `settings.mat` (136 found), which are *separate files*, and their
timestamps do not match the result file's: a verified pair is
`cellRegistered_20260128_144604.mat` beside `logFile_20260128_144720.txt`, **76 seconds
apart**, because the log is written after the save. Pairing must therefore be "nearest log
at or after the result file within the same `Results/` directory", and must record which
rule matched. Where no log exists the run gets the sentinel paramset `unknown` and
`quality_source = 'absent'` — it is not given a plausible default.

---

## 5. Failure modes and missing-data semantics

| # | Failure | Evidence | Semantics |
|---|---|---|---|
| 1 | Result file unreadable | 1 of 108 (`cellRegistered_combined.mat`, `OSError: could not read bytes`) | `CellRegResult.parse_ok = 0`, `parse_error` set, **no Cell/Member rows**. The run is recorded as having failed, not omitted. |
| 2 | Internally inconsistent file | 1 of 108: `cell_to_index_map` has 11 sessions, `p_same` is 10 × 10 | `p_same_consistent = 0`; PairQuality is **not** populated. Refuse rather than silently mis-index. |
| 3 | Score fields absent | 0 of 107 real files, but the script omits them whenever `registration_approach = 'Simple threshold'` | Score columns NULL, `has_score_fields = 0`, `missing_fields` names them. **Required by the code path, unexercised by the data — so only a synthetic fixture can test it.** |
| 4 | No rotation | 21 of 107 | `alignment_rotation` NULL, `alignment_mode` says why. NULL means "not applicable", never 0. |
| 5 | Column unresolvable to a scan | the condition `realign_global_ids_*` currently papers over | `imaging.Curation` NULL, `resolution_method = 'unresolved'`. No row in Member can then exist for it, so no analysis silently joins to the wrong scan. |
| 6 | Column ambiguous | >1 scan above the Jaccard threshold | `ambiguous = 1`, left unresolved. Never majority-voted into a guess. |
| 7 | Footprints point nowhere | 14 of 107 store an external path; 1 is `X:\data\aeltona\...`, a Windows drive that cannot resolve on the Linux server | `footprints_storage = 'external_path'`, path stored verbatim and **not** dereferenced at ingest. A broken path is a recorded fact. |
| 8 | Cell absent from a session | the common case — median sessions-per-cell is 1 in most runs | No `Member` row. Distinguished from "scan not in run" by `CellRegRun.Session`. |
| 9 | Duplicate copies of one run | 6 groups over 14 paths | Collapse by content hash; extra paths are provenance, not new runs. |
| 10 | Upstream curation deleted | cascade through the nullable FK | See §9 — this is the live risk of the ADR-004 departure. |
| 11 | **CellReg input stale relative to the curation** | Slack: *"Fall.mat used by CellReg not auto-updating after manual suite2p curation — curation updates iscell.npy but not Fall.mat"* (Leon Kremers / Tobias Rose / Setup User, 2024-05-24, ts=1716546557.549569) | The run's mask set can predate the curation it nominally belongs to, so `mask_set_jaccard` is legitimately < 1.0 even when the log file resolves the column correctly. **This is why Jaccard is stored even when the log resolves the column**: it is the only quantitative evidence of input staleness. A low overlap on a log-resolved column is a flag, not a resolution failure. |
| 12 | On-disk footprints correspond to a different paramset | *"Re-running suite2p with different parameters keeps all extractions in the database, but data folders on disk only retain the last extraction's data"* (Tobias Rose, 2024-08-01, ts=1722492807.667179) | An external `footprints_path` may point at a stack belonging to a later suite2p run than the registration's own. Path and checksum are recorded; the file is not dereferenced or trusted at ingest. |

The `0`-means-absent convention deserves its own line because it is the classic way this
goes wrong: CellReg writes a 1-based mask index and uses `0` for "not present". The
importer converts with `map - 1`, which sends absent to `-1`, and **absence must then be
represented as a missing row, never as mask `-1` stored in the table**. `realign_global_ids_*`
does exactly this conversion (`orig_map.astype(int) - 1  # matlab -> python numbering; 0 -> -1`)
and then has to filter `m < 0` at every use. A schema that stores absence as a sentinel
inherits that filtering obligation forever; one that stores it as no row does not.

---

## 6. Migration and backward compatibility

**Nothing migrates, and that is the design.** No production table is altered, no column is
dropped, no primary key is extended. `roi_group` keeps its current meaning and its current
values. Three consequences:

1. **Two identity systems coexist.** Existing code that reads `roi_group` keeps working
   unchanged and keeps its current ambiguity. New code joins through
   `CellRegResult.Member`. This is a real cost and should be named as such: until consumers
   are repointed, the defect persists for them.
2. **The bridge is `RoiGroupSnapshot`.** It records what `roi_group` held at a given moment
   so that a result computed against an earlier state remains interpretable. Taking the
   first snapshot is the one action worth doing soon, because every re-ingest between now
   and then is history that cannot be recovered.
3. **The real fix is deferred, not forgotten.** Promoting cell identity into
   `Segmentation.Mask`'s primary key is the correct end state and is **breaking**: DataJoint
   cannot extend a primary key in place, so it needs a drop-and-recreate of a 3.2-million-row
   production part table with re-insertion. That is a destructive operation on live data,
   it is out of scope here by explicit decision, and it should get its own review and its
   own authorisation.

Forward compatibility: adding a second `MatchingMethod`, a new `CellRegParamSet`, or a new
`RegistrationCuration` are all row inserts. Adding a quality measure is a new nullable
column on `Cell`. None of these is breaking.

---

## 7. Validation tests

Declaration-level, on fixtures and mocks, with no database writes:

1. **Every definition compiles.** A validator that discovers the table set *from the module*
   in source order and walks each master's namespace for parts — not from a hand-maintained
   list. The topcam review records why: its validator's hand-kept dict had silently fallen
   one table behind and printed *"All 18 definitions check out"*, a statement true about the
   list and silent about the module.
2. **Primary-key composition** of all ten tables, including that `column_index` is in
   `CellRegRun.Session`'s key and the curation FK is not, and that no nullable attribute
   appears in any primary key.
3. **Parameter-set immutability**: re-registering a slug with different parameters raises;
   re-registering identical parameters is idempotent.
4. **Index conversion**: a synthetic `cell_to_index_map` with known 1-based values and
   explicit zeros round-trips to the right 0-based masks, and every zero becomes a *missing
   Member row* rather than a `-1`.
5. **Both MAT formats**: `make()` runs against synthetic `.mat` files written as v7.3/HDF5
   and as v5/v7, and produces identical rows.
6. **Missing-score path**: a synthetic Simple-threshold file with the score fields absent
   yields NULLs, `has_score_fields = 0`, and a populated `missing_fields` — the only way to
   test failure mode 3, since no real file exercises it.
7. **Inconsistent-file refusal**: a synthetic file with a `p_same` side ≠ session count sets
   `p_same_consistent = 0` and populates no PairQuality rows.
8. **Jaccard resolution**, including the ambiguous case: two candidate scans above threshold
   produce `ambiguous = 1` and an unresolved column, not a winner.
9. **Stable-identifier mapping**: exact mask-set agreement maps cleanly; a deliberately
   conflicting case sets `conflict = 1`.

Plus a **read-only acceptance pass on real files** that parses and inserts nothing: on the
eight drift animals, reproduce the current `roi_group` assignment and the `artefacts/t78`
match scores from the v1-drift-audit; then parse all 108 inventoried files and report the
failures rather than suppressing them. The expected result is 107 parsed and 1 failed — if
that count changes, something regressed.

**Server state, verified 2026-10-15:** a name scan across every `roselab_*` schema for
`cellreg|cell_reg|regist|zeta|qc|quality|export|manifest|release|oracle|reliab` returns
three tables — `pupil_tracking.__pupil_frame_quality`,
`pupil_tracking.__headfixed_frame_quality` and `wfield.registration` (image registration,
not cells). No cell-registration table exists, and this review declares none.

---

## 8. What belongs in DataJoint versus downstream

**In the database:** identity (which masks are one cell), the mapping to scans and its
resolution evidence, per-cell and per-pair match quality, run-level accuracy, parameters by
reference, curation decisions, and centroids. These are small, relational, shared, and
queried by key.

**On disk, referenced by path and checksum:** the spatial footprints. 93 of 107 files embed
a 3-D stack and the largest is `513 × 513 × 3363`, which is **≈ 7.1 GB as float64** in a
single field; 14 files already keep them external. Storing those in MySQL — or in the blob
store — would dwarf the entire 1.12-billion-row database for no query that anyone performs.
The footprint array is read by a human looking at a figure, not by a join.

**Downstream, not here:** eligibility thresholds and cohort membership (§2); anything that
computes a scientific result from these identities; and the figures. The line is that this
schema answers *"is this the same cell, how well do we know that, and who said so"* and
stops. Whether that cell is good enough for a given analysis is a different question with a
different version history.

---

## 9. Departure from `leondrift` ADR-004, and the argument against it

ADR-004 says: **no foreign keys from our schema into adamacs tables.** This design declares
one — a nullable secondary dependency on `imaging.Curation`. The departure is deliberate and
the reasoning is contextual, not a disagreement on the merits.

ADR-004's three reasons were: cross-schema FKs need `REFERENCES` privileges; they block
upstream deletions; and they let somebody else's cascade silently remove our rows. Those
are the right concerns **for a project schema owned by one analysis**, which is what
`roselab_leondrift` is. This schema is different in kind: it is production, in
`adamacs_ingest`, in the same `roselab_` privilege domain, and the rest of the repository
already does exactly this — `disk.py` carries foreign keys into `model.PoseEstimationNew`
and `pupil_tracking`. A registration identity that does not reference the curation it was
computed from is not describing anything checkable.

**The honest counter-argument, which a reviewer should weigh rather than accept:** a cascade
from an upstream curation deletion would remove registration rows that cost a MATLAB run to
produce, and it would do so without announcement. Three things make that acceptable here
and should be stated rather than buried: the `.mat` file on disk remains the source of
truth, so a cascaded row is *recoverable* rather than lost; the dependency is nullable, so
an unresolved column survives independently of any curation; and deleting a curation whose
registrations are still in use is a thing the lab wants to notice, which a cascade causes
and a plain column does not. If the reviewer thinks the cascade risk outweighs referential
integrity, the alternative is plain `session_id`/`scan_id`/`paramset_idx`/`curation_id`
columns plus the `mask_fingerprint` approach `RunSet.Scan` already uses — and that choice
should be made now, because it is expensive to reverse once rows exist.

---

## 10. What a reviewer should attack

1. **The content-hash primary key.** `cellreg_run_id` is unreadable, and every error message,
   log line and notebook variable will carry a 40-character digest. Is collapsing 6 duplicate
   groups worth that, or should duplicates be a `.Copy` part table under a human-readable
   run key?
2. **The ADR-004 departure (§9).** This is the decision that is expensive to reverse.
3. **`min_p_same` as a stored column.** It is a summary of PairQuality and therefore a second
   source of truth — the exact defect the topcam review calls out about hand-maintained
   check lists. Defence: it is the selector column that makes cell-level QC queryable without
   a 10.7-million-row join, and it is derived in one place at populate time. Attack: it can
   drift from its own part table, and nothing in the schema prevents that.
4. **Two identity systems (§6.1).** Is a coexistence period acceptable, or does a half-migrated
   identity invite exactly the confusion this is meant to remove?
5. ~~**`RegistrationSite` invents a grouping**~~ — **resolved by ADR-R1**: it does not invent one,
   it reuses the lab's existing first-session-id convention. The residual question is whether
   `same_site_id` should be fixed upstream first (Q1), which would make this table a
   reconciliation rather than a source.
6. **PairQuality grain rests on `106 of 107`.** The shape was inferred from the data because
   the CellReg toolbox is not readable from this account. One file disagrees. If the toolbox
   semantics differ from the inference, this table is wrong in a way the tests cannot catch —
   someone with read access to `/home/leonk/CellReg/CellReg/cluster_cells.m` should confirm it.
7. **No part table for the p_same model curves.** `p_same_given_centroid_distance` and its
   spatial-correlation twin are what make the probabilities interpretable, they are small, and
   they are not stored. Deliberate omission or a gap?
8. **`StableCellMap` is Manual.** It encodes a computation (mask-set agreement across runs).
   Should it be Computed with its method as a config, so the mapping is reproducible rather
   than asserted?
9. **The schema name.** `registration` already means image registration in this database
   (`roselab_wfield.registration`). Naming a schema `roselab_registration` for *cell*
   registration invites exactly one wrong assumption, forever. See Q6 — the recommendation is
   `cell_registration`.
10. **ADR-R3 reverses a stated plan of Tobias's.** The argument is that the 2024 plan's own
    consequence is the defect being fixed. If that reading of D1/T1 is wrong, the central
    premise of this schema is wrong with it.

---

## 11. Open questions for Tobias

- **Q1.** Is the site grouping `(subject, site_id)` with a hand-assigned label acceptable, or
  should fixing `same_site_id` come first?
- **Q2.** Should the first `RoiGroupSnapshot` be taken now? It is a read of production and an
  insert into a new schema, so it needs an explicit write authorisation — and every
  re-ingest before it is history that cannot be recovered.
- **Q3.** Who is allowed to insert a `RegistrationCuration`, and should a decision carry the
  curator's own identity rather than a free-text name?
- **Q4.** The two `cellRegistered_combined.mat` files are hand-made and both are defective
  (one unreadable, one internally inconsistent). Should they be registered as failed runs for
  the record, or excluded from the inventory as not-a-run?
- **Q5.** `/home/leonk/CellReg` is unreadable from `backup_user`. Confirming attack #6 needs
  either read access or someone else's reading of `cluster_cells.m`.
- **Q6.** Schema name. `roselab_wfield` already contains a table called `registration`, with
  a `registration_method` column — but that is *image* registration, a different concept. No
  schema matching `%regist%` exists, so `roselab_registration` is free, but it is ambiguous
  against existing lab vocabulary. `roselab_cell_registration` or `roselab_cellreg` would
  not be. **Recommendation: `cell_registration`.** Naming is cheap now and expensive after
  the first row.
- **Q7.** ADR-R3 below reverses a plan you stated in 2024. Please confirm, since it is the
  one place this design knowingly contradicts a recorded intent of yours.

---

## 12. Grilling record — contradictions found and how they were resolved

The design above was challenged against the repository, the live database and the wiki's
Slack corpus. `grill-with-docs` is a stub that delegates to two skills (`grilling`,
`domain-modeling`) which are not present in this catalogue, so the challenge was run
directly against source evidence. Nine findings; the four that changed the design are
recorded as ADRs.

### ADR-R1 · Sites are identified by their first session id, not a new label — accepted 2026-10-15
*Supersedes the original §3.6.* The design initially minted a fresh `site_id` on the grounds
that `same_site_id` is broken. Source evidence shows the lab already has a working
convention: the CellReg folder layout is keyed on samesiteID (ts=1716542821.819479), the real
directories embed a session id, and `roselab_leondrift.RunSet.Site` uses
`same_site_id : varchar(16)` with values like `sess9G1EB7EE`. Minting a parallel identifier
would have created exactly the divergence this schema exists to remove. Resolved in favour of
the existing vocabulary, with `site_id_source` recording provenance.

### ADR-R2 · Jaccard overlap is stored even when the log file resolves the column — accepted 2026-10-15
*Strengthens §3.3.* After finding the log file, the obvious simplification was to resolve from
it and skip the geometric check. Source evidence forbids that: `Fall.mat` does not update when
suite2p curation does (ts=1716546557.549569), so a correctly-resolved column can still have a
mask set that disagrees with the curation's. Jaccard is therefore not a fallback — it is an
independent staleness measurement, stored unconditionally. New failure mode 11.

### ADR-R3 · The importer does NOT write `roi_group` — accepted 2026-10-15, contradicts a 2024 plan
**This is the one place the design knowingly departs from a recorded intent of Tobias's.** On
2024-05-24 the stated plan was *"a computed table with a make function incorporating the
CellReg pipeline, triggered via `populate_cellreg` call, reading cellreg folder data and
**updating the cellgroup index in the curation table**"* (ts=1716543113.442619).

This design adopts the first half verbatim — `CellRegResult` is a table whose `make()` reads
the cellreg folder — and **rejects the second half**. The resolution is by evidence, not
preference: writing identity back into `Segmentation.Mask.roi_group` is precisely the mutation
that produced known issue **D1** (recorded counts 111/146 versus live 116/120), left **T1**
unreconstructable, and forced the existence of
`realign_global_ids_to_current_roi_group()`. The 2024 plan's own consequence is the defect
this schema was commissioned to fix. Registration identity is therefore written only to the
new schema; `roi_group` is left untouched and snapshotted (§3.8). Flagged as **Q7** because a
reversal of a stated intent should be confirmed rather than assumed.

### ADR-R4 · `MaskCoordinates` keeps denormalised native centroids, with the reason stated — accepted 2026-10-15
*Amends §3.5.* `mask_center_x/y/z` already exist upstream as `int(11)`, making the native
columns a second source of truth. Kept deliberately, because the upstream table is
re-ingestible without trace (**T7**; ts=1722492807.667179) and a run-time copy is what makes
that detectable — the same argument as `mask_fingerprint`. Recorded rather than left implicit,
because it is a real cost.

### ADR-R5 · `CellRegRun.Session` is keyed on the curation, not on the column index — accepted 2026-10-15
*Supersedes §3.3.* Raised by Tobias during implementation: should the session entry not carry
the full upstream key — session, scan, paramset, curation?

It should. The original shape keyed on `(cellreg_run_id, column_index)` and carried
`-> [nullable] imaging.Curation` as a secondary attribute, justified by the unresolvable-column
case. That was the tail wagging the dog: a column that cannot be resolved has no usable
`Member` rows anyway, so letting it dictate a nullable key degraded the **common** case to
protect the rare one.

Amended shape:

```
CellRegRun.Session        PK: cellreg_run_id, session_id, scan_id, paramset_idx, curation_id
                          + unique index (cellreg_run_id, column_index)
CellRegRun.UnresolvedColumn  PK: cellreg_run_id, column_index
                          -- n_masks, best_jaccard, ambiguous, candidates, reason
```

Three things improve at once:

1. **Joins become direct.** A `Member` row now inherits the whole curation key, so
   `CellRegResult.Member * imaging.Segmentation.Mask` matches on
   `(session_id, scan_id, paramset_idx, curation_id, mask)` with no projection or renaming.
   That is the entire purpose of the table — replacing `roi_group` as a join key — and the
   nullable version made every such query go through a projection first.
2. **Referential integrity becomes real.** A non-nullable foreign key means the database,
   not a convention, guarantees that a registered session is a curation that exists. The
   cascade consequence is the one already accepted in §9.
3. **It matches the lab's own precedent.** `roselab_leondrift.RunSet.Scan` keys on
   `(run_set_id, session_id, scan_id)` — the upstream identity in the key, not beside it.

`column_index` survives as a secondary attribute with a unique index because
`p_same_registered_pairs` is indexed by column and the parser needs the mapping both ways.
`PairQuality` therefore keys on two plain `smallint` column indices: two renamed five-attribute
foreign keys would add ten key columns to a ten-million-row table to express what two
smallints already express, and the column→curation lookup is one join through `Session`.

The nullable-FK finding from the first pass is not wasted — it stays true and is still used
nowhere, which is the honest outcome: `-> [nullable]` is supported and has repository
precedent (`subject.py:30`), it simply is not the right tool here.

### ADR-R6 · `roi_group` is a derived projection of the latest curated run, carrying `cell_index` — accepted 2026-10-15, supersedes ADR-R3
Proposed by Tobias: *"roi_group is the latest cellreg curation of a run."* Accepted, and it
dissolves the contradiction ADR-R3 left open rather than choosing a side.

`roi_group` stops being an independent fact and becomes a **cache**: the cell identity of the
latest curated run for that site, materialised onto `Segmentation.Mask` because that is the
query path the lab already uses (ts=1772204733.321419). This satisfies all three positions at
once — the 2024 plan to write it, the 2026 convention to query it, and ADR-R3's objection —
because what made D1 harmful was that the previous value was **unrecoverable**. When every
write is derived from a recorded `(cellreg_run_id, curation_id)` and the sources are
append-only, overwriting a regenerable cache is not data loss.

**In CellReg terms `roi_group` is the row index of `cell_to_index_map`**, 0-based. Three
independent confirmations: the `realign_global_ids_*` docstring calls the global ids *"row
indices into the cell_to_index_map"*; `leondrift.TrialResponse.roi_groups` is commented
*"CellReg id (-1 = unmatched)"*; and `artefacts/cohort/global_roi_groups.json` begins
`[0, 1, 2, 3, 4, 8, 9, …]`. So it is exactly this schema's `cell_index`, and the projection is
one inversion of `Member`:

```
roi_group(session_id, scan_id, paramset_idx, curation_id, mask)
    = cell_index of the Member row carrying that mask, for the chosen run + curation
    = -1 where no Member row exists (mask matched to no registered cell)
```

**What is projected is `cell_index`, not `stable_cell_id`, and the reason is column width.**
Upstream `roi_group` is `smallint` (max 32,767) and the largest single run already holds
**25,724** registered cells — 78 % of the ceiling for one run. A run-scoped `cell_index`
always fits because it resets per run; a `stable_cell_id` accumulating across
re-registrations does not, and would eventually overflow a production column that
`leondrift.RunSet.Cell` also keys on. Projecting stable ids therefore requires widening
`roi_group` to `int` on a 3.2 M-row part table — the follow-on migration, written up rather
than smuggled in. `stable_cell_id` remains available in this schema for anyone wanting
re-run-stable identity.

Recorded by `RoiGroupProjection` (+ `.Scan`), append-only, carrying the source run, the
curation, the value kind, the pre-projection `RoiGroupSnapshot`, and a `dry_run` flag.
**Performing a projection writes to production and needs explicit authorisation in that
turn; nothing here does it.** A partially tracked cell has no mask in the sessions it is
absent from, so there is nothing to project there — `-1` means "mask belonging to no
registered cell", never "absent session".

### Findings that confirmed the design rather than changing it

- **`-> [nullable]` has production precedent**: `adamacs/schemas/subject.py:30` declares
  `-> [nullable] Lab`. The nullable secondary dependency in `CellRegRun.Session` is an
  existing repository pattern, not an invention.
- **Cross-schema FKs have precedent**: `disk.py` declares `-> model.PoseEstimationNew`
  (lines 866, 981), `-> subject.Subject` (496) and `-> model.Model.BodyPart`. §9's claim is
  verified, which is what makes the ADR-004 departure defensible.
- **The 1-based / `0`-means-absent convention is independently documented** by the lab:
  *"indexing is MATLAB-based (starts at 1, 0 = not detected) … convert to 0-based:
  `[int(x-1) for x in cell_to_index_map[index]]`"* (Leon Kremers, 2024-05-24,
  ts=1716542082.875469). The same message establishes **`mat73`** as the lab's sanctioned
  loader, so the importer should use it rather than raw `h5py`.
- **Column types match**: `session_id`/`scan_id` `varchar(12)`, `paramset_idx` `smallint(6)`,
  `curation_id` `int(11)`, `mask` `smallint(6)` — as declared above.

### Finding with an operational consequence, not a schema one

CellReg is not to be run on tatchu3 — *"use Loki or ibehaveGPU for analysis work and cellreg;
reserve tatchu3 for database hosting and pipeline jobs"* (Tobias Rose, 2025-08-26,
ts=1756201852.926469), after a CellReg run exhausted its RAM and took the server down
(2025-02-20, ts=1740048401.751339). Parsing 18.8 GB of `.mat` files is the same class of
work. `CellRegResult.make()` should therefore run on a worker, not on the database host —
a deployment note for the PR, not a table change.

### The contradiction that remains open

The current documented query path **is** `roi_group`: *"ROIgroups should be
entered/updated for cellreg/longitudinal cell matching … querying all longitudinal cells is
done by querying by roigroup"* (Tobias Rose, 2026-02-27, ts=1772204733.321419) — and at
2026-02 that is more recent than the 2024 plan. This design does not break that path, but it
does introduce a second one, and §6.1's "two identity systems" caveat is therefore a
documented divergence from current lab instruction rather than a hypothetical cost. It cannot
be resolved by this schema alone: it needs a decision about when consumers are repointed from
`roi_group` to `CellRegResult.Member`, which is Q2's companion and belongs with the QC and
cohort work rather than here.
