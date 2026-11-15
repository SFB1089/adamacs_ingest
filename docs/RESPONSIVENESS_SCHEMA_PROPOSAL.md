# Proposal: one lab-wide schema for dF/F0, cell activity, responsiveness and tuning

*Draft v3 for Tobias Rose, 15.11.2026. **Status: proposal only.** Nothing has been declared, activated or inserted.
All database inspection was read-only: server-side `SET SESSION transaction_read_only = 1`, virtual modules,
DataJoint calls. This version is designed from scratch, not from leondrift.*

---

## 1. The idea

**Four questions about every imaged cell**, each stricter than the one before:

| Level | Question | Example evidence |
|---|---|---|
| **Active** | Is this a real cell with real activity, regardless of any stimulus? | soma classifier, SNR, event rate |
| **Responsive** | Does it respond to the events of an experiment at all? | ZETA on all onsets; response vs baseline window |
| **Selective** | Does it respond *differently* to different conditions (tuned)? | Kruskal across conditions, orientation permutation test, reliability |
| **Specific** | *Which* conditions does it respond to? | ZETA or window test per condition, Holm-corrected |

**Four rules shape the design:**

1. **Measurements and decisions are separate.** Measurement tables store numbers only: p-values, effect sizes,
   reliabilities, one row per cell. A decision is a named rule ("ZETA p < 0.05 and orientation-tuned") applied to
   those numbers. A new threshold is a new rule, with no recomputation.
2. **Nothing names an experiment.** Gratings, natural images, receptive fields, Bpod tasks and oddball corridors enter
   only as rows of one lookup table (`EventClass`).
3. **dF/F0 becomes a stored, parameterised table.** Today it is computed and thrown away inside the CASCADE populate
   (§2.3). It is the common input of CASCADE, of trace quality and of any dF/F-based response analysis.
4. **Inputs are linked by foreign keys.** Every per-cell row points at `imaging.Fluorescence.Trace`. Every onset points
   at `event.Event` and/or `trial.Trial`.

---

## 2. What exists today

### 2.1 Production pipeline (`adamacs_ingest`, prefix `roselab_`)

There is **no** dF/F, responsiveness or tuning table in production. The inputs exist:

| Input | Where | What matters |
|---|---|---|
| Raw traces | `imaging.Fluorescence.Trace`: `fluorescence`, `neuropil_fluorescence` (`blob@external-raw`), key `session_id, scan_id, paramset_idx, curation_id, mask, fluo_channel` (3.6 M rows) | The common parent of every derived trace |
| Derived traces | `imaging.Activity.Trace` (5.3 M rows) with `extraction_method` | Values: `suite2p_deconvolution` (3,182 activities), CASCADE variants (`cascade_inference_25ms_filter` 860, `_11` 657, `cascade_inference` 83, others < 15). **No model name** in `Activity`: only via `ActivityCascadeTask`. |
| Curation | `imaging.Curation` (3,182 rows) | Always explicit. "Latest" changes over time. |
| Cell type | `imaging.MaskClassification.MaskType` | soma / dendrite / axon / neuropil / artefact / unknown, plus a confidence value |
| Events | `event.Event` (≈ 1 billion rows), key `session_id, scan_id, event_type, event_start_time decimal(11,5)` | The stimulus identity is written into the `event_type` text, e.g. `Moving_Gratings:210;Moving_Gratings;0;10;50:50;210;0.1;3;` |
| Frame clock, shutter | `event.Event` of type `bench2p_frames`, `shutter` | Frame timestamps on the DAQ clock. The shutter-closed interval is the dark-signal window. |
| Trials | `trial.Trial`, `.Attribute`, `TrialEvent` (§2.2) | Repeat blocks for visual stimulation; task trials for Bpod experiments |
| Cross-session identity | `cell_registration.CellRegResult.Member` | For "responsive in both sessions". Never use `Segmentation.Mask.roi_group`: CellReg overwrites it. |

`cell_registration` is the house template for new schemas:
* nothing is declared on import;
* quality is stored as rows, not blobs;
* thresholds live downstream of the measurements.

### 2.2 The `trial` schema: "trial" means three different things

| Table | Content | Rows |
|---|---|---|
| `TrialType` (Lookup) | `trial_type varchar(512)` | 80 |
| `Trial` (Imported) | `session_id, scan_id, trial_id` → `trial_type`, `trial_start_time`, `trial_stop_time` (float, s, DAQ clock) | 223,369 |
| `Trial.Attribute` | `attribute_name varchar(16)` → `attribute_value varchar(2000)` (all text) | 447,970 |
| `TrialEvent` | `Trial` ↔ `event.Event` | 47.4 M |
| `Block`, `BlockTrial` | declared, empty | 0 |

| Ingest path | One `Trial` row is | The condition is in | Example |
|---|---|---|---|
| **bench2p visual stimulation** (`behavior.get_and_ingest_trial_times`) | one **repeat block** (every stimulus once). The block length is inferred as *number of stimulus events / smallest per-stimulus count*. | the `event_type` of each presentation; `TrialEvent` links presentations to blocks | drift gratings: 15 trials × 12 presentations |
| **Bpod tasks** (`bpod.py`) | one **task trial** | `trial_type` (`BPOD: SubCricket; 45`, `DF_wood`, ...) and `Trial.Attribute` (`bpod_outcome`, `contingency`, `reward_prob`, ...). `TrialEvent` holds the Bpod states (`bpod_cue`, `bpod_at_target`, port events, reward, ...). | dynamic foraging: 158 trials |
| **Oddball / VR corridor** | one corridor run | `trial_type` (`1`, `2 : Oddball`, ...) plus texture events | 41 trials |

**Weak spots to keep in mind:**
* **Natural-image sessions have one single `Trial`** covering the whole recording. The single images (count 1) collapse
  the block heuristic, so for images repeats must be counted per image ID.
* **Block membership is inferred from counts**, which is right only for balanced designs.
* **Attribute values are text** (names ≤ 16 characters).
* **Trial times (float) and event times (decimal) must be joined through `TrialEvent`**, never by matching times.

### 2.3 dF/F0 today: computed inside the CASCADE populate and never stored

**Where it runs.** CASCADE runs on **BUGPU** (ibehaveGPU1, 172.25.70.3).
* Worker: `~/SOPs/Serverscripts/populate_tables_cascade.py`, in the `Cascade` env, log
  `~/logs/cascade_worker.log`.
* What it does: populates `imaging.Activity` from `imaging.ActivityCascadeTask` rows through the SFB1089 fork of
  element-calcium-imaging. The installed copy is pip `@main` commit `e75c3fd`, **edited in place on 2026-09-03 15:22**,
  so it is not the commit as published.
* Current log: `~/SOPs/Serverscripts/populate_tables_cascade.log`. The older `~/logs/cascade_worker.log` ends on
  2026-09-08.

The installed `Activity.make` on BUGPU (read 15.11.2026) does this for every `extraction_method` that contains
`cascade_inference`:

```
F, Fneu  = Fluorescence.Trace (all masks of the curation)
dark     = calculate_mean_darksignal(F, key)          # mean of the all-cell average trace in the shutter-closed window
                                                      # (falls back to frames 12–24 if the shutter events are missing)
F, Fneu  = F - dark, Fneu - dark
neu      = 0.0 if "no_neuropil" in extraction_method else 0.7      # rules on the method NAME
pct      = 11  if extraction_method.endswith("_11") else 8
           → cascade_inference, _25ms_filter, _global_causal: 0.7 / 8 · _11: 0.7 / 11 · _no_neuropil: 0 / 8 · _no_neuropil_11: 0 / 11
           ("25ms_filter" and "global_causal" differ only in the CASCADE model, not in the dF/F0)
dF       = (F - neu·Fneu) + neu·median(Fneu)           # neuropil subtraction, re-centred
F0       = running percentile(dF, pct, window = 60 s · fps)
dFF      = (dF - F0) / F0
spikes   = cascade.predict(model_name, dFF)            # model from ActivityCascadeTask
Activity.Trace ← spikes                                # dFF is discarded
```

The lab helper `adamacs/helpers/stack_helpers.calculate_dFF` implements the same steps for notebooks. leondrift calls
it to rebuild dF/F0 into disk caches.

**Problems with this arrangement** (verified in the code, and in leondrift's `known_issues` D7/D10, which executed both
functions on real scans):

1. **dF/F0 is not in the database.** Every analysis that needs it recomputes it with its own copy of the code, and
   nobody can check that two analyses used the same dF/F0.
2. **The parameters are hidden in the method name.** Since 2026-09-03 they follow string rules on the method name
   (above); before that, explicit `if` branches.
   * `cascade_inference_25ms_filter` (the drift primary signal) therefore uses **neuropil 0.7 + median, percentile 8,
     60 s**: the same dF/F0 as `cascade_inference`. This settles leondrift's open question Q10: its assumption of 8
     was right.
   * The rules are fragile in both directions. A future method called `cascade_inference_fast_11` would silently get
     percentile 11. The older branch code had no 25 ms branch, so it would have failed on that method with an
     undefined `neu`. The TATCHU3 copies (`datajoint` and `datajoint_analysis` envs) still have that older code.
3. **The dark signal depends on which code version ran, and nothing records which one did.**
   * **Old copy** (still installed on TATCHU3; used by BUGPU before 2026-09-03): it adds a float to a `Decimal` event
     time and swaps the arguments of `get_closest_timestamps`. The `try` block always fails, so it **always used the
     hard-coded frames 12–24**.
   * **BUGPU copy since 2026-09-03, and the lab helper:** cast correctly, pass the arguments in the right order, and
     use the shutter-closed window (≈ frames 8–29). The fallback to frames 12–24 remains when the shutter events are
     missing.
     * The old worker log shows 4 such fallbacks.
     * Two of them subtracted **negative** "dark" values (−13.6), i.e. frames 12–24 were not dark in those scans.
     * The current log does not record the dark window at all.
   * **Dating the stored CASCADE rows.** The timestamps in DataJoint's external-store table (`~external_raw`,
     joined on the `activity_trace` hash of mask 0 per activity) show when each row was written:

     | extraction_method | activities dated | written before the 2026-09-03 fix | so the dark window was |
     |---|---|---|---|
     | `cascade_inference_25ms_filter` | 889 | 6 | shutter (883), frames 12–24 or unknown (6) |
     | `cascade_inference_11` | 685 | 681 | frames 12–24 (681), shutter (4) |
     | `cascade_inference` | 98 | 83 | frames 12–24 (83), shutter (15) |
     | `cascade_inference_global_causal` | 7 | 5 | mixed |
     | `cascade_inference_no_neuropil` / `_11` | 6 / 12 | 6 / 12 | frames 12–24 |

     **So one extraction method can hold rows from two code versions.** The input of a stored CASCADE trace can only
     be named per activity, not per method. A blob shared between rows keeps its first timestamp, so these counts are
     approximate.
   * In clean recordings the two windows give nearly the same value (scan9FOW97IH: 0.85 vs 0.59 units). It is still a
     silent, unrecorded choice, and it is wrong where frames 12–24 are not dark.
4. **There is no floor on F0.** Masks whose neuropil-corrected fluorescence is ≤ 0 (empty ROIs) get F0 ≤ 0 and a
   finite, **sign-inverted** dF/F0. leondrift flags them (`f0_nonpositive`); the pipeline does not.
5. **The helper has two hygiene problems:**
   * Its docstring defaults (0.7 / 60 s / 15) differ from its signature (0.3 / 120 s / 15).
   * It imports `adamacs.pipeline` at module level, which activates schemas, i.e. it writes on import.

### 2.4 Responsiveness and tuning today (outside production)

**Drift audit, `t73`/`t140`: ZETA as the inclusion test** (TSV files):
* Test: zetapy 4.1 `zetatstest`, 100 resamples.
* Inputs: the deconvolved traces (suite2p, CASCADE-25) on the real frame times.
* Events and window:
  * events: all 180 grating onsets, or the 1,000 repeated-image onsets;
  * window = the median onset-to-onset interval;
  * edge events are dropped.
* Decision: p < 0.05, uncorrected.
* Calibration with random event times: 5–6 % false positives.
* Result: 91 % of grating cells and 68 % of image cells are responsive (OASIS).
* nb37 uses it as an inclusion filter only: responsive in both or all sessions; for orientation drift, also
  orientation-tuned (permutation test).

**`roselab_leondrift`** (drift project):
* tests: Kruskal, rank-sum, permutation, ZETA, Bauer CI width, OSI;
* reliability measures for images (split-half, r²_ER, SNR).

Its results are stored as one blob per scan holding all cells, keyed on a frozen cohort, with the yes/no stored at
compute time.

**Its ZETA differs from the audit's:**

| | Audit (t73/t140) | leondrift |
|---|---|---|
| time base | frame timestamps | `frame_index / fps` |
| window | median onset-to-onset interval | 3 s |
| missing samples | removed | filled with the median |
| edge events | dropped | kept |

The new schema implements the audit version.

---

## 3. What the literature recommends

| Question | Method | In use? |
|---|---|---|
| Active cell | classifier; CASCADE noise level ν; SNR; skewness; event rate (Rupprecht et al. 2021, 2026) | classifier only |
| Responsive | time-series ZETA: no bins or windows to choose, robust to calcium autocorrelation, calibrated (Montijn et al. 2021; Montijn/Heimel et al. 2023) | yes (audit) |
| | response vs baseline window (Wilcoxon, Kruskal) | yes (leondrift) |
| | ≥ 25 % of preferred-condition presentations significant (Allen Brain Observatory, de Vries et al. 2020) | no |
| Specific | ZETA or window test on one condition's onsets, Holm-corrected across conditions | no |
| Selective | Kruskal across conditions | yes |
| | vector OSI/DSI with label permutation, or Hotelling T² on single-trial vectors (Mazurek et al. 2014). The bare OSI index is biased. | permutation: yes |
| | two-sample ZETA, condition vs rest: also sees differences in response shape (`zetatstest2`, zetapy 4.1) | no |
| Reliable | split-half (Spearman–Brown), noise-corrected r² (Pospisil & Bair 2021), explainable variance | yes (images) |
| Behaviour dependence | running vs quiet presentations (Niell & Stryker 2010) | no |

**Consequences for the design:**
1. **ZETA alone does not show tuning.** It admits 91 % of grating cells, hence the ladder.
2. **Store effect sizes and reliabilities next to every p.** A p-value mixes effect size with repeat count (10 for
   images, 15 for gratings).
3. **Correct per-condition tests within each cell (Holm).** With 12 directions and no correction, about 46 % of cells
   would look "specific" by chance.
4. **Allow selecting on one half of the data and analysing the other** (odd/even repeat blocks). Same-data selection
   biases the estimate.
5. **Keep the sign of the response**, so excited and suppressed cells can be told apart.

---

## 4. Design

### 4.1 Overview (13 tables in four blocks)

```
 A. TRACES                                   B. EVENTS
 DffParamSet (L) ─┐                          EventClass (L) ──► EventSet (C)
 imaging.Fluorescence ─► Dff0 (C) + .Trace                       + .Event → event.Event
                          │                                       + .Trial → trial.Trial
                          ├─► TraceQuality (C) + .Mask   ACTIVE
                          └─► [later] SpikeInference (C) + .Trace   (CASCADE reading Dff0, see §4.4)
 Signal (L): "which trace": dff0 <idx> | imaging.Activity <extraction_method> | spikes <...>
 CascadeInput (C): each stored CASCADE activity → the DffParamSet it was computed from (dated, then recomputed)

 C. RESPONSES
 ResponseParamSet (L) ─┐
 Signal, EventClass ───┼─► ResponseTask (M) ─► ZetaTest (C) + .Mask + .Condition     RESPONSIVE / SPECIFIC / SELECTIVE
 imaging.Fluorescence ─┘                    └► WindowStats (C) + .Mask + .Condition  RESPONSIVE / SELECTIVE / SPECIFIC / TUNING

 D. DECISIONS
 CellRule (L) + .Criterion ─► CellCategory (C) + .Mask ──join──► cell_registration
```

**Conventions:**
* Element style throughout, as in `ProcessingParamSet → ProcessingTask → Processing` and
  `ActivityCascadeModel → ActivityCascadeTask → Activity`: parameter-set lookups with a unique hash, a manual task
  table, computed tables populated by workers.
* One part row per cell for anything filterable. Blobs only for traces and per-onset vectors (`blob@external-raw`,
  like `Fluorescence`).
* Lookup rows are never edited; a change is a new row.
* Nothing is declared on import: `activate()` is called explicitly.
* Parameter keys are named `dff_paramset_idx` and `resp_paramset_idx`, never `paramset_idx`, so they cannot collide
  with imaging's in a join.
* Every attribute has a `# comment` with units.

### 4.2 Block A: traces

#### `DffParamSet` (Lookup): one row per dF/F0 recipe

**Why a parameter set with explicit columns.** The recipes in use differ in every step:
* dark subtraction;
* neuropil factor;
* whether and how a neuropil offset is added back (median vs mean);
* detrending;
* baseline method and its parameters;
* how a non-positive F0 is handled.

Today these choices sit in `if` branches (CASCADE) or in free dicts. leondrift's `dff` config even carries a key
(`f0_floor_quantile`) that its code never reads: the code applies a 5 %-of-median-F floor instead. So the floor in
force is not the one named.

Here every step is a typed column. That makes it:
* queryable (`DffParamSet & 'neuropil_factor = 0.7'`);
* checked when a row is inserted;
* covered by a hash over the full, normalised row, so the same recipe can never enter twice under two numbers.

The dF/F0 is computed in this fixed order, one column group per step:

```
1 dark        F, Fneu ← F − dark, Fneu − dark
2 neuropil    Fc ← F − a·Fneu + a·offset(Fneu)                  offset: none | median | mean
3 detrend     Fc ← Fc − trend(Fc)                               optional, before the baseline
4 baseline    F0 ← baseline(Fc)                                 running percentile | maximin | global percentile
5 floor       F0 ≤ 0 handled                                    flag | NaN | floor at a fraction of median F | ...
6 dF/F0       (Fc − F0) / F0
```

```python
@schema
class DffParamSet(dj.Lookup):
    definition = """
    # One dF/F0 recipe. Steps run in the order of the column groups. Same recipe -> same hash. Never edited.
    dff_paramset_idx       : smallint
    ---
    dff_desc               : varchar(255)   # plain language, e.g. 'lab standard: shutter dark, neuropil 0.7 + median, running p8 / 60 s'
    dff_hash               : uuid           # hash of all columns below (normalised)
    unique index (dff_hash)
    # 1 dark signal
    dark_method            : varchar(16)    # 'none' | 'frames' (fixed frame range) | 'shutter' (shutter-closed window from event.Event)
    dark_frame_start=null  : int            # 'frames': first frame
    dark_frame_stop=null   : int            # 'frames': last frame (exclusive)
    dark_margin_s=null     : float          # [s] 'shutter': margin inside the shutter-closed interval (lab code 0.09)
    # 2 neuropil
    neuropil_factor        : float          # a in F - a*Fneu (0 = no neuropil subtraction)
    neuropil_offset        : varchar(8)     # added back: 'none' | 'median' (+a*median(Fneu), lab/CASCADE) | 'mean' (+a*mean(Fneu), Bauer 2024)
    # 3 detrending (before the baseline; usually 'none' because a running baseline detrends)
    detrend_method         : varchar(16)    # 'none' | 'linear' | 'highpass'
    detrend_window_s=null  : float          # [s] 'highpass': cut-off period
    # 4 baseline F0
    baseline_method        : varchar(24)    # 'running_percentile' | 'maximin' (suite2p: smooth, running min, running max) | 'global_percentile'
    baseline_percentile=null : float        # running/global percentile
    baseline_window_s=null : float          # [s] running window
    baseline_smooth_s=null : float          # [s] gaussian pre-smoothing sigma ('maximin')
    baseline_step=1        : smallint       # compute the running filter on every k-th frame and interpolate (1 = exact)
    # 5 F0 floor
    f0_floor_method        : varchar(24)    # 'flag' (keep, flag F0<=0) | 'nan' (dF/F0 = NaN where F0<=0) | 'fraction_of_median_F' | 'quantile_of_F0'
    f0_floor_value=null    : float          # fraction or quantile for the floor methods
    # anything a future method needs (part of the hash)
    extra=null             : longblob
    """

    @classmethod
    def insert_new_params(cls, dff_paramset_idx, dff_desc, **params):
        """Validate against the method registry (required columns per method, no unknown keys, no parameters
        that the chosen method ignores), normalise (e.g. neuropil_offset='none' when neuropil_factor=0),
        hash, and refuse an existing recipe under a new index."""
```

**Methods are short strings checked by a Python registry**, not SQL `enum`s. A new detrending or baseline method is
then a code change plus a row, never an `ALTER TABLE` on a populated table. The registry is the same module that
computes the dF/F0 (`adamacs/analysis/responsiveness/dff0.py`), so a method cannot exist in the table without an
implementation.

**First rows: every recipe found in the lab's code.**

| idx | recipe | dark | neuropil (a, offset) | detrend | baseline | floor | status |
|---|---|---|---|---|---|---|---|
| 1 | **lab standard** (proposed) | shutter, 0.09 s (fallback frames 12–24, recorded) | 0.7, median | none | running p8, 60 s | flag | to confirm (§6) |
| 2 | CASCADE p8, **since 2026-09-03** (`cascade_inference`, `_25ms_filter`, `_global_causal`) = **lab helper `calculate_dFF`** with CASCADE settings (leondrift `dff0`) | shutter, 0.09 s (fallback frames 12–24) | 0.7, median | none | running p8, 60 s | none* | read on BUGPU; the helper is the same recipe (same hash) |
| 3 | CASCADE p8, **before 2026-09-03** | frames 12–24 | 0.7, median | none | running p8, 60 s | none* | old code on TATCHU3 |
| 4 | CASCADE p11 since 2026-09-03 (`_11`) | shutter, 0.09 s (fallback 12–24) | 0.7, median | none | running p11, 60 s | none* | read on BUGPU |
| 5 | CASCADE p11 before 2026-09-03 (`_11`) | frames 12–24 | 0.7, median | none | running p11, 60 s | none* | old code |
| 6 | CASCADE no neuropil, p8, before 2026-09-03 | frames 12–24 | 0, none | none | running p8, 60 s | none* | old code |
| 7 | CASCADE no neuropil, p11, before 2026-09-03 | frames 12–24 | 0, none | none | running p11, 60 s | none* | old code |
| 8 | leondrift `dff` | none | 0.7, none | none | running p10, 60 s, step = window/60 | fraction_of_median_F 0.05 (the code's actual floor) | comparison |
| 9 | leondrift `dff_np` (Bauer 2024 style) | none | 0.7, mean | none | running p10, 60 s, step = window/60 | fraction_of_median_F 0.05 | comparison |
| 10 | suite2p convention | none | 0.7, none | none | maximin, σ 10 s, 60 s | flag | comparison |

\* `none` = no floor and no flag: what that code does. F0 ≤ 0 gives a sign-inverted dF/F0. `Dff0.Trace.f0_nonpositive`
is still recorded for every recipe, so the defect is visible even where the recipe does not handle it.

**Dark-window fallback.** With `dark_method = 'shutter'`, the `dark_frame_start/stop` columns name the fallback frames
used when a scan has no shutter events. `Dff0.dark_source` records per scan which window was actually used
(`shutter` or `fallback_frames`), together with the frames and the value. The silent fallback becomes a visible fact.

**Rows 2–7 are reconstructions from code.** They are confirmed per scan by recomputing CASCADE on the stored dF/F0
and comparing with `imaging.Activity` (§5 step 1).

**Using different recipes side by side.** Each recipe is just another `dff_paramset_idx`. A sensitivity analysis
("does responsiveness depend on the neuropil factor?") is a set of `Signal` rows pointing at different recipes, then
one restriction on the results. No new tables are needed.

#### `Dff0` (Computed): the dF/F0 traces

```python
@schema
class Dff0(dj.Computed):
    definition = """
    # dF/F0 of every mask of one curation, computed once and stored
    -> imaging.Fluorescence
    -> DffParamSet
    ---
    fps                  : float          # [Hz] from scan.ScanInfo
    dark_signal          : float          # [a.u.] subtracted from F and Fneu
    dark_frame_start=null: int            # first frame of the dark window actually used
    dark_frame_stop=null : int            # last frame (exclusive)
    dark_source          : varchar(32)    # 'shutter', 'frames', 'fallback_frames' (shutter events missing), 'none'
    n_f0_nonpositive     : int            # masks with F0 <= 0 somewhere
    code_version         : varchar(64)
    """
    class Trace(dj.Part):
        definition = """
        -> master
        -> imaging.Fluorescence.Trace
        ---
        dff                  : blob@external-raw   # dF/F0 per frame
        f0_median            : float      # [a.u.] median baseline
        f0_min               : float      # [a.u.] minimum baseline
        f0_nonpositive       : bool       # F0 <= 0 somewhere: dF/F0 sign-inverted there
        """
    # key_source: (Fluorescence * DffParamSet) restricted to what ResponseTask, TraceQuality or SpikeInference need,
    # so dF/F0 is computed only on demand (all curations x one parameter set would be of the order of the Fluorescence
    # store itself, ~TB).
```

The function is one pure function, `dff0(F, Fneu, frame_times, shutter_events, params)`. The helper and the CASCADE
populate can later call the same function, so the two copies of the dark-signal code disappear.

#### `TraceQuality` (Computed): ACTIVE

It is computed for every `Dff0` (the definitions are fixed and versioned by `code_version`). The cell type is not
copied; rules join `imaging.MaskClassification` directly.

```python
@schema
class TraceQuality(dj.Computed):
    definition = """
    # Event-independent quality and activity of every dF/F0 trace
    -> Dff0
    ---
    code_version         : varchar(64)
    """
    class Mask(dj.Part):
        definition = """
        -> master
        -> Dff0.Trace
        ---
        noise_nu             : float      # [%/sqrt(Hz)] CASCADE noise level: median|diff(dF/F)| * sqrt(fps) * 100
        snr                  : float      # 99th percentile of dF/F0 / robust noise SD
        skewness             : float      # of dF/F0; high for sparse transients
        event_rate           : float      # [1/min] transients > 3 x robust noise SD
        f0_drift             : float      # [%/min] slope of F0 over the session (bleaching, z-drift)
        neuropil_ratio       : float      # median Fneu / median F (after dark subtraction)
        """
```

#### `Signal` (Lookup): which trace an analysis uses

**The problem it solves.** Response analyses may run on dF/F0 (`Dff0`), on stored activity (`imaging.Activity`:
suite2p deconvolution, CASCADE), or later on `SpikeInference`. A `Signal` row names the trace. Every result keeps the
common parent `imaging.Fluorescence.Trace` as its foreign key.

```python
@schema
class Signal(dj.Lookup):
    definition = """
    # A named trace type that response analyses can use
    signal               : varchar(32)    # e.g. 'dff0', 'cascade25', 's2p_deconv'
    ---
    source               : enum('dff0','activity','spikes')
    -> [nullable] DffParamSet             # source = dff0
    -> [nullable] imaging.ActivityExtractionMethod   # source = activity
    description          : varchar(255)
    """
```

| signal | source | dff_paramset_idx | extraction_method |
|---|---|---|---|
| `dff0` | dff0 | 1 | – |
| `s2p_deconv` | activity | – | suite2p_deconvolution |
| `cascade25` | activity | – | cascade_inference_25ms_filter |
| `cascade11` | activity | – | cascade_inference_11 |

**`CascadeInput` (Computed, one row per existing CASCADE activity).** It answers "which dF/F0 recipe was this stored
CASCADE trace computed from?". This cannot be a per-method lookup, because one extraction method holds rows written
by two code versions (§2.3).

```python
@schema
class CascadeInput(dj.Computed):
    definition = """
    # The dF/F0 recipe a stored CASCADE activity was computed from, and how we know
    -> imaging.Activity                   # cascade_inference* rows only
    ---
    -> DffParamSet                        # rows 2-7
    written_at           : datetime       # external-store timestamp of the first trace blob
    evidence             : enum('dated','recomputed','recompute_mismatch')  # dated = from written_at vs the 2026-09-03 code change
    max_abs_diff=null    : float          # 'recomputed': max |stored - recomputed CASCADE| over masks
    """
```

* It starts as `dated`: before 2026-09-03 15:22, the old recipe; after it, the new one. The percentile and neuropil
  factor come from the method-name rules.
* Recomputing CASCADE on the `Dff0` of that recipe for a sample of scans turns rows into `recomputed` or
  `recompute_mismatch`. A mismatch means an undocumented code state, and the row says so.

### 4.2b Phase-1 test result (15.11.2026, test schema `tobiasr_responsiveness`)

**What was built.** `DffParamSet` (10 recipes), `Dff0Task`, `Dff0` + `.Trace`, `CascadeInput` and `CascadeRecompute`
were declared under the personal prefix. The traces went to a scratch store (`~/dj_stores/responsiveness_test`, not
`/datajoint-db`). Code: `adamacs/schemas/responsiveness.py`, `adamacs/responsiveness/dff0.py` (16 offline tests),
`scripts/populate_dff0_test_schema.py`, `scripts/verify_dff0_cascade.py`, `scripts/cascade_predict_npz.py`.

**What was checked.** 3 curations × 8 recipe rows were populated. CASCADE was re-run on the stored dF/F0 on BUGPU
(the worker's checkout, Cascade `4c113e4`, CPU only) for 40–80 masks per activity, and compared with the stored
`imaging.Activity` traces:

| activity | written | recipe | max \|Δ\| | median \|Δ\| | result |
|---|---|---|---|---|---|
| 9G1PMH2S `cascade_inference` | 2026-08-27 | 3 (frames 12–24) | 2.7e-4 | 7e-6 | **reproduced** |
| 9G1PMH2S `_25ms_filter` | 2026-09-03 23:53 | 2 (shutter) | 3.4e-4 | 4e-6 | **reproduced** |
| 9FXY0HRX `_25ms_filter` | 2026-08-30 | 3 | 3.0e-4 | 4e-6 | **reproduced** (the pre-fix code did handle 25 ms) |
| 9FXY0HRX `_global_causal` | 2026-08-30 | 3 | 8.5e-3 | 1e-5 | **reproduced** |
| 9FXY0HRX `cascade_inference` | 2026-08-25 | 3 | 452 | 5e-6 | reproduced except one mask with F0 ≤ 0 (min F0 −463) |
| 9FUMXRWL `_25ms_filter` | 2026-09-24 | 2 | 0.87 | 1e-5 | reproduced except one mask with F0 ≤ 0 |
| 9FUMXRWL / 9FXY0HRX `_11`, 9FXY0HRX `no_neuropil_11` | 2025-12 / 2026-08 | 5, 7 | 22–2,980 | 4e-3–5e-2 | **not reproduced** |

**Conclusions:**
1. **The era split is confirmed.** Before 2026-09-03 the input was dark frames 12–24; since then the shutter window.
   With the recipe matching each row's date, stored and recomputed CASCADE agree to float precision.
2. **The only deviations in reproduced activities are masks with F0 ≤ 0.** There the dF/F0 is ill-conditioned (and
   sign-inverted). `Dff0.Trace.f0_nonpositive` flags exactly those masks.
3. **The stored `cascade_inference_11` traces do not belong to their own masks.**
   * No dF/F0 variant reproduces them (dark window, neuropil offset, percentile 8/11, 60/120 s, both candidate models:
     every variant gives median r < 0).
   * They do not correlate with the `_25ms_filter` trace of the same mask. On **20 of 20** randomly sampled curations,
     the median same-mask r is between −0.12 and +0.03. Two correct methods reach r ≈ 0.97 on the same masks.
   * Up to 37 % of `_11` rows in a curation are flat.
   * The fluorescence predates these rows, so this is not a re-ingestion.

   The rows were written 2025-12 (681 of 685). **They should be treated as invalid until the cause is found.** This
   affects every analysis that used `cascade11` or leondrift's `cascade_any` fallback. The drift primary signal
   (`_25ms_filter`) is not affected.

   **Deleted 2026-11-15 13:08 on Tobias's instruction:** all `cascade_inference_11` and
   `cascade_inference_no_neuropil_11` rows (669 `imaging.Activity`, 811,014 `Activity.Trace`, 669
   `ActivityCascadeTask`).
   * Manifest: `~/adamacs/out/cascade11_deleted_2026-11-15.csv`.
   * The external blob files under `/datajoint-db` were left in place.
   * No production table depended on these rows.
   * leondrift's `cascade11` / `cascade_any` signals now have no source rows for those scans.

### 4.3 Block B: events

#### `EventClass` (Lookup): which onsets, which conditions

There are two modes, because "trial" means different things (§2.2):
* **`unit='event'`**: presentations from `event.Event`. The condition is parsed from `event_type`. The repeat block
  (`trial_id`) comes through `trial.TrialEvent` where trials exist.
* **`unit='trial'`**: `trial.Trial` rows aligned to an event inside each trial (e.g. `bpod_cue`), or to
  `trial_start_time`. The condition comes from `trial_type` or a `Trial.Attribute`; attributes can filter.

```python
@schema
class EventClass(dj.Lookup):
    definition = """
    # A rule that picks response onsets in a scan and labels their conditions. Never edited; a new rule gets a new name.
    event_class          : varchar(64)
    ---
    unit                 : enum('event','trial')
    event_type_like=''   : varchar(128)   # unit=event: the presentations; unit=trial: the alignment event ('' = trial start)
    trial_type_like=''   : varchar(128)   # unit=trial: which trials ('' = all)
    condition_from='none': enum('none','event_type','trial_type','trial_attribute')
    condition_axis=''    : varchar(32)    # name of the label ('direction', 'image_id', ...) or the attribute name
    parser='none'        : varchar(48)    # registered function turning the source text into the label
    circular_period=null : float          # [deg] if the axis is circular (360 for direction): enables tuning columns
    kind='real'          : enum('real','random')  # 'random' = same number of onsets at random times (no-stimulus control)
    selector=null        : longblob       # optional: {'min_repeats': 6}, {'block_parity': 'odd'},
                                          # {'attribute': {'bpod_outcome': ['Hit']}}, {'seed': 0}
    description          : varchar(255)
    """
```

| event_class | unit | event_type_like | trial_type_like | condition_from | axis | circ. | kind | selector |
|---|---|---|---|---|---|---|---|---|
| `gratings` | event | `Moving_Gratings:%` | – | event_type | direction | 360 | real | – |
| `gratings_random` | event | `Moving_Gratings:%` | – | none | – | – | random | `{'seed': 0}` |
| `gratings_odd_blocks` | event | `Moving_Gratings:%` | – | event_type | direction | 360 | real | `{'block_parity': 'odd'}` |
| `nat_images_repeated` | event | `Natural Image:%` | – | event_type | image_id | – | real | `{'min_repeats': 6}` |
| `cricket_cue` | trial | `bpod_cue` | `BPOD: SubCricket%` | trial_type | target_angle | 360 | real | – |
| `foraging_hits` | trial | `raw_bpod_ResponseWindow_entry` | `DF_%` | trial_type | cue | – | real | `{'attribute': {'bpod_outcome': ['Hit']}}` |
| `reward` | event | `aux_bpod_reward` | – | none | – | – | real | – |

**"Without stimulus"** means a `random` class. Its false-positive rate must come out ≈ 5 % (the audit measured 5–6 %).
Real spontaneous blocks (`Spontaneous_Activity` events, where recorded) are an ordinary class.

#### `EventSet` (Computed): the onsets actually used in one scan

```python
@schema
class EventSet(dj.Computed):
    definition = """
    # The onsets an EventClass selects in one scan, with condition labels
    -> scan.Scan
    -> EventClass
    ---
    n_onsets             : int
    n_conditions         : smallint       # 1 if no condition axis
    onsets               : longblob       # [s, DAQ clock] sorted (also holds random onsets)
    conditions           : longblob       # label per onset ('all' if none)
    trial_ids=null       : longblob       # trial.Trial trial_id per onset (repeat block in visual stimulation)
    median_ioi           : float          # [s] median onset-to-onset interval (the ZETA window)
    """
    class Event(dj.Part):                 # the event each onset is (unit=event) or is aligned to (unit=trial)
        definition = """
        -> master
        -> event.Event
        ---
        onset_index          : int
        """
    class Trial(dj.Part):                 # unit=trial: the trial each onset belongs to
        definition = """
        -> master
        -> trial.Trial
        ---
        onset_index          : int
        """
    # key_source = (scan.Scan * EventClass) & ResponseTask
```

### 4.4 CASCADE on `Dff0`: three options (needs your decision)

| Option | What changes | Imaging schema change? |
|---|---|---|
| **A. Link only** (recommended now) | Existing `imaging.Activity` stays as is. `CascadeInput` says, per stored activity, which `DffParamSet` recipe was its input (dated from the external-store timestamp, then confirmed by recomputing a sample). | no |
| **B. New spike table** (recommended for new runs) | `SpikeInference(-> Dff0, -> imaging.ActivityCascadeModel)` + `.Trace`; the BUGPU worker populates it from stored dF/F0. The model is in the key, so "which model?" is a plain restriction, and dF/F0 is computed once. | no (new table in this schema; the BUGPU worker script changes) |
| **C. Change `imaging.Activity`** | `ActivityCascadeTask` gains `-> Dff0`; `Activity.make` reads dF/F0 instead of computing it | **yes**: the element fork and the live `imaging` schema, used by several envs. Only with your explicit OK. |

### 4.5 Block C: responses

#### `ResponseParamSet` (Lookup) and `ResponseTask` (Manual)

```python
@schema
class ResponseParamSet(dj.Lookup):
    definition = """
    # Settings of one response method. Same settings -> same row. Never edited.
    resp_paramset_idx    : smallint
    ---
    method               : enum('zeta','window_stats')
    resp_desc            : varchar(255)
    resp_hash            : uuid
    unique index (resp_hash)
    params               : longblob
    """
    # zeta:         {'window': 'median_ioi', 'resampling': 100, 'jitter': 2, 'edge': 'drop',
    #                'per_condition': True, 'vs_rest': True}                  (per_condition False = audit t73)
    # window_stats: {'resp_window_s': [0.1, 0.6], 'base_window_s': [-0.3, 0], 'n_perm': 1000, 'n_boot': 1000}

@schema
class ResponseTask(dj.Manual):
    definition = """
    # Compute this response method on this trace type of this curation for these onsets. Workers populate.
    -> imaging.Fluorescence               # session, scan, paramset_idx, curation_id (explicit curation)
    -> Signal
    -> EventClass
    -> ResponseParamSet
    ---
    task_owner           : varchar(32)
    task_note=''         : varchar(255)
    """
```

A helper `plan_tasks(restriction, signals, event_classes, paramsets)` inserts the tasks for a cohort. It refuses
combinations whose trace does not exist, e.g. a CASCADE signal without the matching `imaging.Activity` row.

#### `ZetaTest` (Computed): RESPONSIVE, SPECIFIC, SELECTIVE with ZETA

| Variant | Column | zetapy call | Question |
|---|---|---|---|
| pooled (stimulus-agnostic) | `.Mask.zeta_p` | `zetatstest(t, x, all onsets, max_duration=median_ioi)` | Locked to the events at all? (= audit t73) |
| per condition (stimulus-specific) | `.Condition.zeta_p` | the same on one condition's onsets | Responds to **this** condition? |
| condition vs rest (stimulus-selective) | `.Condition.zeta2_p` | `zetatstest2(t, x, onsets_c, t, x, onsets_rest)` | Is the response to this condition **different** from the rest? |

```python
@schema
class ZetaTest(dj.Computed):
    definition = """
    # Time-series ZETA for every trace of one task
    -> ResponseTask                       # method = 'zeta'
    -> EventSet
    ---
    window_s             : float          # [s] resolved window
    n_onsets_used        : int            # after dropping edge onsets
    seed                 : int unsigned   # crc32(scan_id), reproducible
    zetapy_version       : varchar(16)
    code_version         : varchar(64)
    """
    class Mask(dj.Part):
        definition = """
        -> master
        -> imaging.Fluorescence.Trace
        ---
        zeta_p               : float      # time-locked to the onsets, any shape?
        zeta_z               : float      # zetapy 'zeta_score'
        latency=null         : float      # [s] time of the largest deviation
        meanrate_p=null      : float      # zetapy 'ttest_p_value'
        """
    class Condition(dj.Part):             # if params.per_condition
        definition = """
        -> master.Mask
        condition            : varchar(32)
        ---
        n_onsets             : smallint
        zeta_p               : float      # responds to THIS condition?
        zeta2_p=null         : float      # differs from the rest? (params.vs_rest)
        """
```

**Notes on ZETA:**
* "Each condition vs rest, Holm-corrected in the rule" avoids the circularity of picking a preferred condition and
  testing it on the same data.
* Cost scales with cells × conditions × resamples. Default: per-condition for ≤ 12 conditions; image sets (100) only
  on request.

#### `WindowStats` (Computed): window means, classical tests, reliability, tuning

```python
@schema
class WindowStats(dj.Computed):
    definition = """
    # Response/baseline window means per onset and their statistics, for every trace of one task
    -> ResponseTask                       # method = 'window_stats'
    -> EventSet
    ---
    n_onsets_used        : int
    n_repeats_min        : smallint
    code_version         : varchar(64)
    """
    class Mask(dj.Part):
        definition = """
        -> master
        -> imaging.Fluorescence.Trace
        ---
        resp                 : longblob   # (n_onsets,) response-window mean, EventSet order
        base                 : longblob   # (n_onsets,) baseline-window mean
        amp_mean             : float      # [signal unit] mean(resp - base); sign kept
        effect_d             : float      # paired Cohen's d
        resp_p               : float      # Wilcoxon resp vs base, all onsets
        frac_pref_sig=null   : float      # fraction of preferred-condition onsets > base + 2 SD (Allen-style)
        kw_cond_p=null       : float      # Kruskal across conditions
        splithalf_r=null     : float      # Spearman-Brown split-half reliability of the condition curve
        r2er=null            : float      # noise-corrected r2 between halves (Pospisil & Bair 2021)
        snr=null             : float      # variance across conditions / trial-noise variance
        pref_deg=null        : float      # [deg] preferred direction (circular axes only)
        pref_axis_deg=null   : float      # [deg] preferred orientation (doubled angle)
        dsi_vec=null         : float      # first-harmonic vector length
        osi_vec=null         : float      # second-harmonic vector length (1 - circular variance)
        osi_perm_p=null      : float      # label-permutation p of osi_vec (nb37 tuning test)
        dsi_perm_p=null      : float
        pref_ci_width=null   : float      # [deg] bootstrap 95 % CI of the preferred orientation
        """
    class Condition(dj.Part):
        definition = """
        -> master.Mask
        condition            : varchar(32)
        ---
        n_repeats            : smallint
        mean_resp            : float      # [signal unit] mean(resp - base)
        sem_resp             : float
        cond_p               : float      # Wilcoxon, this condition vs its baselines
        """
```

### 4.6 Block D: decisions

```python
@schema
class CellRule(dj.Lookup):
    definition = """
    # A named set of thresholds. Never edited; a changed threshold is a new rule.
    cell_rule            : varchar(64)
    ---
    decided_by           : varchar(64)
    decided_on           : date
    description          : varchar(512)
    """
    class Criterion(dj.Part):
        definition = """
        -> master
        criterion            : varchar(48)
        ---
        level                : enum('active','responsive','selective','specific')
        source               : varchar(64)    # 'TraceQuality.Mask', 'Dff0.Trace', 'ZetaTest.Mask', 'ZetaTest.Condition',
                                              # 'WindowStats.Mask', 'WindowStats.Condition', 'imaging.MaskClassification.MaskType'
        attribute            : varchar(64)
        op                   : enum('<','<=','>','>=','==')
        threshold            : varchar(32)    # number, or a label for '=='
        holm=0               : bool           # Holm across conditions within the cell
        -> [nullable] Signal
        -> [nullable] EventClass
        -> [nullable] ResponseParamSet
        """

@schema
class CellCategory(dj.Computed):
    definition = """
    # One rule applied to every mask of one curation
    -> CellRule
    -> imaging.Curation
    ---
    n_cells              : int
    n_active             : int
    n_responsive         : int
    n_selective          : int
    n_specific           : int
    """
    class Mask(dj.Part):
        definition = """
        -> master
        -> imaging.Segmentation.Mask
        ---
        is_active=null       : bool       # NULL if the rule defines no criterion at this level
        is_responsive=null   : bool       # requires is_active (if defined)
        is_selective=null    : bool       # requires is_responsive
        is_specific=null     : bool       # requires is_responsive
        specific_to=''       : varchar(255)  # conditions that passed, e.g. '30,210'
        """
```

All criteria of a level must pass, and each level requires the level below it. `CellCategory` is stored, not
evaluated on the fly:
* `CellCategory.Mask & 'is_responsive'` becomes a normal restriction that joins to `cell_registration`;
* it is deleted together with its inputs;
* it is cheap to populate.

**The nb37 rule as a `CellRule`:**

| criterion | level | source | attribute | op | threshold | holm | signal | event_class |
|---|---|---|---|---|---|---|---|---|
| soma | active | imaging.MaskClassification.MaskType | mask_type | == | soma | – | – | – |
| f0_ok | active | Dff0.Trace | f0_nonpositive | == | 0 | – | dff0 | – |
| zeta | responsive | ZetaTest.Mask | zeta_p | < | 0.05 | – | s2p_deconv | gratings |
| tuned | selective | WindowStats.Mask | osi_perm_p | < | 0.05 | – | s2p_deconv | gratings |
| which | specific | ZetaTest.Condition | zeta_p | < | 0.05 | yes | s2p_deconv | gratings |

### 4.7 Using it

```python
# declare
Dff0.populate(); TraceQuality.populate()                              # on demand (key_source restricted)
plan_tasks(imaging.Fluorescence & cohort, signals=['s2p_deconv', 'cascade25', 'dff0'],
           event_classes=['gratings', 'gratings_random'], paramsets=[ZETA, WINDOW_GRATINGS])
EventSet.populate(); ZetaTest.populate(reserve_jobs=True); WindowStats.populate(reserve_jobs=True)
CellCategory.populate()

# query
resp = CellCategory.Mask & {'cell_rule': 'drift2026_v1'} & 'is_responsive'
tuned_po = (WindowStats.Mask & {'signal': 's2p_deconv', 'event_class': 'gratings'}) * resp.proj()
dff = Dff0.Trace & {'dff_paramset_idx': 1} & resp.proj()              # the dF/F0 of responsive cells
null = ZetaTest.Mask & {'event_class': 'gratings_random'}
fp = len(null & 'zeta_p < 0.05') / len(null)                          # must be ≈ 0.05
```

Cross-session rules ("responsive in both sessions") are joins of `CellCategory.Mask` with
`cell_registration.CellRegResult.Member`, wrapped in one helper.

---

## 5. Build order and checks (test schema first, e.g. `tobiasr_responsiveness`)

1. **`DffParamSet` rows 1–10 + `Dff0`** on a few drift scans (BUGPU code read 15.11.2026, §2.3).
   * Row 1 must equal `calculate_dFF` (helper, shutter dark) to float precision.
   * Rows 2–7: CASCADE recomputed on them (on BUGPU, same model) must equal the stored `imaging.Activity` traces;
     this fills `CascadeInput.evidence`. Include one scan per code era and one fallback scan (negative dark value).
2. **`TraceQuality`.** Check the distributions per scan and the `f0_nonpositive` counts.
3. **`EventClass` + `EventSet`** for the 11 drift mice:
   * 180 grating onsets in 15 blocks;
   * repeated-image counts as expected;
   * onset times identical to t73's.

   Also one Bpod experiment (`cricket_cue`), as a test of the trial mode.
4. **`ZetaTest` pooled.** p-values equal `artefacts/t140/zeta_all.tsv`; random classes give 5–6 % false positives.
5. **`WindowStats`.** Values equal leondrift's Kruskal, `sub_vec` tuning and image reliabilities for the same windows.
6. **`ZetaTest` per condition** (gratings): null check and cost per scan.
7. **`CellRule 'drift2026_v1'` + `CellCategory`.** The cohort sizes reproduce nb37.
8. **Production `roselab_responsiveness`** only with your OK.
   * Code: `adamacs/schemas/responsiveness.py` (deferred activate) and pure functions in
     `adamacs/analysis/responsiveness/` (`dff0`, `zeta`, `window_stats`, `parsers`), with offline tests like
     `tests/test_cellreg_*`.

---

## 6. Decisions and open points

1. **CASCADE and `Dff0`:** option A now plus B for new runs (no imaging change), or C (change `imaging.Activity`; needs
   your explicit OK)?
2. **The CASCADE worker on BUGPU** (outside this schema, for you or the pipeline owners):
   * **Record the dF/F0 it used.** Option B makes this automatic; until then, at least log the dark window and value
     per scan.
   * **Replace the method-name rules with explicit parameters.**
   * **Update the TATCHU3 element copies**, which still carry the old dark-signal bug.
   * **Move the Slack webhook URLs out of the script.** They are written into
     `populate_tables_cascade.py` in clear text, so whoever can read the SOPs repository can post to the lab Slack.
3. **The lab standard dF/F0 (row 1):** shutter dark, neuropil 0.7, percentile 8, 60 s, F0 ≤ 0 flagged. Confirm, or
   name different defaults.
4. **Store `CellCategory`** (recommended) or evaluate on the fly?
5. **Per-condition ZETA:** default up to 12 conditions, images only on request?
6. **"Active" thresholds:** set now, or after looking at the `TraceQuality` distributions?
7. **Running vs quiet** event classes in a second phase?
8. **First non-drift test** of the trial mode: which Bpod or oddball experiment?
9. **Name and home:** `roselab_responsiveness` in `adamacs_ingest`?

## References

* Montijn JS et al. (2021) A parameter-free statistical test for neuronal responsiveness. *eLife* 10:e71969.
* Montijn JS, Heimel JA et al. (2023) A new family of statistical tests for neuronal spiking and autocorrelated
  timeseries data. *bioRxiv*.
* de Vries SEJ et al. (2020) A large-scale standardized physiological survey reveals functional organization of the
  mouse visual cortex. *Nat Neurosci* 23:138.
* Mazurek M, Kager M, Van Hooser SD (2014) Robust quantification of orientation selectivity and direction
  selectivity. *Front Neural Circuits* 8:92.
* Pospisil DA, Bair W (2021) Accounting for bias in the estimation of r² between two sets of noisy neural responses.
  *J Neurosci*.
* Stevenson IH et al. (2011) Statistical assessment of the stability of neural movement representations.
  *J Neurophysiol* 106:764.
* Rupprecht P et al. (2021) A database and deep learning toolbox for noise-optimized, generalized spike inference from
  calcium imaging. *Nat Neurosci* 24:1324. Rupprecht P et al. (2026) Spike inference from calcium imaging data
  acquired with GCaMP8 indicators. *Nat Methods*.
* Niell CM, Stryker MP (2010) Modulation of visual responses by behavioral state in mouse visual cortex. *Neuron*
  65:472.
