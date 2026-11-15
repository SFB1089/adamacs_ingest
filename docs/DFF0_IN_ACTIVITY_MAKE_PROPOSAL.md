# Proposal: CASCADE in `imaging.Activity.make` reads a stored, parameterised dF/F0

*Draft for Tobias Rose, 15.11.2026. It builds on PR #25 (merged): `adamacs/responsiveness/dff0.py`,
`adamacs/schemas/responsiveness.py`, `docs/RESPONSIVENESS_SCHEMA_PROPOSAL.md` §2.3 and §4.2b.*

> **Decision (15.11.2026): tables in `imaging`.** Implemented for review in SFB1089/element-calcium-imaging branch
> `dff0-tables` (version 0.1.0b1) and adamacs_ingest branch `dff0-activity-make-proposal`. Nothing is declared in
> `roselab_imaging` or deployed yet: rollout §8, steps 3–4 need your OK.

---

## 1. Goal

Today the CASCADE branch of `imaging.Activity.make` (SFB1089 fork of element-calcium-imaging) computes dF/F0 itself
and throws it away. That has three consequences:
* its parameters come from string rules on the extraction-method name;
* the code that ran is unknown per row (on BUGPU it was edited in place on 2026-09-03, so one method holds rows from
  two code versions);
* nobody can check the input of a stored spike trace.

**After this change:**
1. **dF/F0 is a table in `imaging`** (`DffParamSet`, `Dff0` + `.Trace`), computed by one pure function.
2. **The recipe of each CASCADE method is a row** (`CascadeRecipe`: extraction method → dF/F0 recipe), not an `if` on
   the name.
3. **`Activity.make` reads `Dff0.Trace`** and records which `Dff0`, model and code commits produced each new activity
   (`Activity.CascadeInput`, a part table with a foreign key to `Dff0`).
4. **Existing tables keep their definitions.** No `ALTER`; every schema change is a new table.

---

## 2. Where the code lives

| Piece | Home | Why |
|---|---|---|
| pure dF/F0 (`compute_dff0`, recipe validation and hash, shutter dark window) | **element fork**: `element_calcium_imaging/dff.py` (moved from `adamacs/responsiveness/dff0.py`) | `imaging` is upstream of everything; the element must not import `adamacs` |
| `adamacs.responsiveness.dff0` | thin re-export of `element_calcium_imaging.dff` (kept for imports) | one implementation |
| `adamacs.helpers.stack_helpers.calculate_dFF` | wrapper around `element_calcium_imaging.dff.compute_dff0` with a `DeprecationWarning` | removes the second, diverging dark-signal copy |
| `element_calcium_imaging.imaging.calculate_mean_darksignal` | deleted; replaced by `dff.shutter_dark_frames` + `dff.dark_signal` | removes the Decimal/float bug for good |
| tables `DffParamSet`, `Dff0`, `CascadeRecipe`, `Activity.CascadeInput` | **element fork `imaging.py`** → `roselab_imaging` | CASCADE (in `imaging`) depends on them, so they must live in `imaging` or upstream of it |
| `responsiveness` schema | keeps `Signal`, `CascadeInput` (dated, for legacy rows), `CascadeRecompute`, and later the events and response tables; references `imaging.Dff0` | its own `DffParamSet` / `Dff0` are dropped from the module before production |

---

## 3. Schema changes (all additive, in `roselab_imaging`)

```python
@schema
class DffParamSet(dj.Lookup):          # identical to the tested adamacs version (PR #25)
    definition = """
    # One dF/F0 recipe. Steps run in the order of the column groups. Same recipe -> same hash. Never edited.
    dff_paramset_idx       : smallint
    ---
    dff_desc               : varchar(255)
    dff_hash               : uuid
    unique index (dff_hash)
    dark_method            : varchar(16)    # 'none' | 'frames' | 'shutter'
    dark_frame_start=null  : int
    dark_frame_stop=null   : int
    dark_margin_s=null     : float
    neuropil_factor        : float
    neuropil_offset        : varchar(8)     # 'none' | 'median' | 'mean'
    detrend_method         : varchar(16)    # 'none' | 'linear' | 'highpass'
    detrend_window_s=null  : float
    baseline_method        : varchar(24)    # 'running_percentile' | 'maximin' | 'global_percentile'
    baseline_percentile=null : float
    baseline_window_s=null : float
    baseline_smooth_s=null : float
    baseline_step=1        : smallint
    f0_floor_method        : varchar(24)    # 'none' | 'flag' | 'nan' | 'fraction_of_median_F' | 'quantile_of_F0'
    f0_floor_value=null    : float
    extra=null             : longblob
    """

@schema
class Dff0Task(dj.Manual):             # dF/F0 asked for outside CASCADE (responsiveness, notebooks)
    definition = """
    -> Fluorescence
    -> DffParamSet
    ---
    task_owner : varchar(32)
    task_note='' : varchar(255)
    """

@schema
class Dff0(dj.Computed):
    definition = """
    # dF/F0 of every mask of one curation under one recipe
    -> Fluorescence
    -> DffParamSet
    ---
    fps                  : float
    n_masks              : int
    n_frames             : int
    dark_signal          : float
    dark_frame_start=null: int
    dark_frame_stop=null : int
    dark_source          : varchar(16)    # 'shutter' | 'frames' | 'fallback_frames' | 'none'
    n_f0_nonpositive     : int
    element_commit       : varchar(40)    # commit of the installed element fork (see §6)
    compute_seconds      : float
    """
    class Trace(dj.Part):
        definition = """
        -> master
        -> Fluorescence.Trace
        ---
        dff            : blob@external-raw   # float32
        f0_median      : float
        f0_min         : float
        f0_nonpositive : bool
        """

    @property
    def key_source(self):
        # only what someone needs: CASCADE tasks (through their recipe) and explicit Dff0Task rows
        for_cascade = (Fluorescence * ActivityCascadeTask * CascadeRecipe).proj("dff_paramset_idx")
        return (Fluorescence * DffParamSet).proj() & (for_cascade.proj() + Dff0Task.proj())

@schema
class CascadeRecipe(dj.Lookup):
    definition = """
    # The dF/F0 recipe a CASCADE extraction method uses (replaces the name rules in Activity.make)
    -> ActivityExtractionMethod
    ---
    -> DffParamSet
    """
```

**A new part table on the existing master** (DataJoint declares it as a new table; legacy activities simply have no
part rows):

```python
class Activity(dj.Computed):
    ...
    class CascadeInput(dj.Part):
        definition = """
        # What produced this CASCADE activity (rows exist for activities computed after this change)
        -> master
        ---
        -> Dff0                           # the exact dF/F0 used (FK: deleting that Dff0 removes this activity)
        -> ActivityCascadeModel           # model at compute time (the task row can change later)
        cascade_commit   : varchar(40)    # git commit of the Cascade checkout
        element_commit   : varchar(40)    # git commit of the installed element fork
        computed_at      : datetime
        """
```

**`CascadeRecipe` rows** (recipe numbers as in PR #25):

| extraction_method | dff_paramset_idx | note |
|---|---|---|
| `cascade_inference` | 2 | = what the code has done since 2026-09-03 |
| `cascade_inference_25ms_filter` | 2 | idem; "25ms" is the model, not the dF/F0 |
| `cascade_inference_global_causal` | 2 | idem |
| `cascade_inference_no_neuropil` | 11 (new: shutter dark, neuropil 0, p8) | the post-fix recipe was never registered |
| new `cascade_dff_lab` | 1 (lab standard, F0 ≤ 0 flagged) | if you want CASCADE on the corrected standard; needs a new `ActivityExtractionMethod` row |

The `_11` methods get no rows: their activities were deleted, and their name rule disappears with this change.

---

## 4. `Activity.make` and `key_source` (sketch of the diff)

```python
@property
def key_source(self):
    suite2p_key_source = ...                                   # unchanged
    caiman_key_source = ...                                    # unchanged
    cascade_key_source = (Fluorescence * ProcessingParamSet.proj("processing_method")
                          * ActivityCascadeTask * CascadeRecipe) & Dff0     # dF/F0 must exist first
    return suite2p_key_source.proj() + caiman_key_source.proj() + cascade_key_source.proj()

def make(self, key):
    if (CascadeRecipe & key):                                  # membership, not the method name
        return self._make_cascade(key)
    method, imaging_dataset = get_loader_result(key, Curation)   # moved: CASCADE no longer reads suite2p files
    ...                                                        # suite2p / caiman branches unchanged

def _make_cascade(self, key):
    from cascade2p import cascade                              # path setup as today
    fkey = {k: key[k] for k in Fluorescence.primary_key}
    dff_idx = (CascadeRecipe & key).fetch1("dff_paramset_idx")
    dkey = {**fkey, "dff_paramset_idx": dff_idx}
    tkeys, dff = (Dff0.Trace & dkey).fetch("KEY", "dff", order_by="mask, fluo_channel")
    model_name = (ActivityCascadeTask & key).fetch1("model_name")
    spikes = cascade.predict(model_name, np.vstack(dff), verbosity=0)
    self.insert1(key)
    self.Trace.insert({**key, "mask": t["mask"], "fluo_channel": t["fluo_channel"], "activity_trace": spikes[i]}
                      for i, t in enumerate(tkeys))
    self.CascadeInput.insert1({**key, "dff_paramset_idx": dff_idx, "model_name": model_name,
                               "cascade_commit": _git_commit(CASCADE_DIR), "element_commit": ELEMENT_COMMIT,
                               "computed_at": datetime.now()})
```

**What this changes in behaviour:**
* **Same numbers.** For the existing methods the spikes are identical to today's: recipe 2 = the post-2026-09-03 code,
  verified to float precision in PR #25.
* **The CASCADE populate no longer computes dF/F0.** It reads it.
  * `Dff0` (CPU only: percentile filter) can be populated by a CPU worker on TATCHU3. The BUGPU worker then only
    runs `cascade.predict`.
  * GPU time per scan drops by the percentile-filter time: for the 171k-frame scan, about 2 of roughly 5 minutes.
* **A changed recipe can never silently reuse old spikes.** A new recipe is a new `dff_paramset_idx`, which means a
  new method row (or a new `CascadeRecipe` row before any activity exists for that method).

---

## 5. Legacy rows (956 existing CASCADE activities)

* **They stay as they are.** There is no `Activity.CascadeInput` row for them, because inserting parts after the fact
  would claim knowledge we only inferred.
* **Their provenance stays in `responsiveness.CascadeInput`.** It is computed from the external-store timestamp
  (before or after 2026-09-03). Where verified, it is backed by `CascadeRecompute`.
* **Optional: recompute everything once under the new code** (897 curations, roughly 1 GPU-minute each plus dF/F0
  on CPU). Every CASCADE row would then have exact provenance and the pre-2026-09-03 dark-frame rows would disappear.
  This needs deleting and recomputing those `Activity` rows: your call, and a separate step.

---

## 6. Deployment hygiene: code = commit

The 2026-09-03 in-place edit on BUGPU is why one method holds two code versions. So:
* **Install from a tagged commit** of the fork in every env that imports it.
  * Environments: TATCHU3 `datajoint`, `datajoint_analysis`, `datajoint_server`, `leondrift`, `cellreg`; BUGPU
    `Cascade`.
  * Pin with `pip install git+https://github.com/SFB1089/element-calcium-imaging@<tag>`.
* **`ELEMENT_COMMIT` is read at import** from the installed package's `direct_url.json` (pip writes it). It is
  stored on every `Dff0` and `Activity.CascadeInput` row.
* **The worker refuses to start if the installed files differ from that commit.** It does a hash check of
  `imaging.py` and `dff.py` against the commit, so an in-place edit is caught.

---

## 7. Cost

* **Storage.**
  * One recipe for all 897 curations with CASCADE tasks: 1.07 M traces, about **0.38 TB** float32 in `external-raw`.
  * For all 3,182 curations: about 1.2 TB.
  * `/datajoint-db` has 4.2 TB free.
* **Compute.** dF/F0 is the percentile filter already run today; it moves, it is not added.

---

## 8. Rollout (each step names what needs your OK)

1. **Element fork PR:** `dff.py` (moved from adamacs, with its tests), the new tables, the `make` and `key_source`
   change, `ELEMENT_COMMIT`, deletion of `calculate_mean_darksignal`. Plus a matching `adamacs_ingest` PR:
   * the `dff0` re-export;
   * the `calculate_dFF` wrapper;
   * `responsiveness` switched to `imaging.DffParamSet` / `imaging.Dff0`.

   *Code review only.*
2. **Dry run with no writes:** for a sample of activities, run `_make_cascade`'s computation on BUGPU (CPU) and
   compare with the stored activity, as in PR #25. *No OK needed (read-only).*
3. **Declare the 4 new tables in `roselab_imaging`** by activating the fork (additive, no `ALTER`). Insert
   `DffParamSet` rows 1–11 and the `CascadeRecipe` rows. **Needs your OK.**
4. **Pin the fork tag in all envs on TATCHU3 and BUGPU, and restart the CASCADE worker** with the populate order
   `Dff0` → `Activity`. **Needs your OK** (it touches the running worker).
5. **Watch the first new tasks.** Check that `Activity.CascadeInput` is filled and that the spikes equal a recompute.
6. **Optional:** drop the test schema `tobiasr_responsiveness` and `~/dj_stores/responsiveness_test`. *Needs your OK
   (delete).*
7. **Optional:** recompute the legacy CASCADE rows (§5). **Needs your OK** (delete plus recompute).

---

## 9. Alternatives considered

| Alternative | Why not (or not now) |
|---|---|
| **`ALTER` `ActivityCascadeTask` to add `-> [nullable] DffParamSet`** | DataJoint 0.14 `alter()` cannot add foreign keys, and it would change a populated production table. `CascadeRecipe` gives the same link additively. |
| **New `SpikeInference` table in `responsiveness`, legacy `Activity` frozen** | No change to the element, but CASCADE output would then live in two tables, and analysts would have to know which one holds which rows. A fallback if you don't want `imaging` to grow. |
| **Keep computing dF/F0 inside `make`, only log the parameters** | Cheapest, but dF/F0 stays unstored: responsiveness, trace quality and notebooks would keep recomputing it. |

## 10. Decisions for you

1. **Tables in `imaging` (this proposal) or `SpikeInference` in `responsiveness`** (§9, row 2)?
2. **Register the lab-standard CASCADE method `cascade_dff_lab`** (recipe 1), or keep only the existing methods on
   recipe 2?
3. **Recompute the 956 legacy CASCADE activities** under the new code (§5), or leave them with dated provenance?
4. **Who reviews the element-fork PR, and when may the BUGPU worker be restarted?**
