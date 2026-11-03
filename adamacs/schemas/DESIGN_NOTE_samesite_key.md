# Design note — re-key RegistrationSite on the lab's `same_site_id`

**Status:** proposed, authorised by Tobias 2026-11-03 ("All three, including a site for the
legacy runs"). Adversarial by instruction: every section below states what could go wrong
with the proposal, not why it is good.

**Prompted by** Tobias asking "why don't you use the samesite_id as RegistrationSite???" —
a question that turned out to have three different answers depending on which part of the
claim you check.

## 1. Scientific purpose

A CellReg registration is only meaningful within one chronically imaged field of view. The
lab already names that object: `roselab_session.session_site.same_site_id`, grouped through
`session_same_site`. `RegistrationSite` must *be* that object, not a parallel one, so that
a registration, the drift analyses in `roselab_leondrift`, and any future per-site table all
agree on what "the same site" means without a translation step.

## 2. What is actually true today (measured, 2026-11-03)

| claim | status |
|---|---|
| `registration_site` is keyed on `same_site_id` | TRUE — PK is `(subject, same_site_id)`, `varchar(12)` |
| its two values are canonical site ids | TRUE — both present in `session_site`; group 26 and 11 sessions |
| anything enforces that | **FALSE — 0 referential constraints on the table** |
| `subject` is needed in the key | **FALSE — `session_site`'s PK is `same_site_id` alone** |
| the historical runs carry a site | **FALSE — all 102 have none** |
| the schema's own comment is still accurate | PARTLY — it says "all 3,460 sessions" and "2,561 of them have same_site_id = session_id"; today `session_site` holds 3,530 rows, 2,561 is the count of *distinct* samesites, and the singletons are 2,520. The 41 multi-session sites still holds. The comment needs these numbers refreshed or dropped — live counts do not belong frozen in a docstring. |
| a site can be derived for them | TRUE — 91 of 102 runs have resolved columns, and each maps to **exactly one** `same_site_id`; 0 runs span two; 0 of 1,106 columns lack a samesite row |

## 3. The foreign key is weaker than it sounds — and this was already on the record

**Correction to my own framing.** I first presented this as a newly discovered defect. It is
not. `RegistrationSite`'s definition in `adamacs/schemas/cell_registration.py` (lines 502-517)
carries a section headed *"Why a local table rather than an FK today"* stating that upstream
"DEFAULTS TO SELF-REFERENCE" and that therefore "never assigned to a site" and "a site of
exactly one session" are indistinguishable there. The absence of the FK was a deliberate,
documented decision, not an oversight. What follows is a re-measurement of it, not a
discovery, and the genuinely new part is the membership rule at the end of this section.

`adamacs/ingest/session.py` inserts **every** session into `SessionSite` with
`same_site_id = session_id`, and `SessionSameSite` maps each session to itself by default
("Update manually for chronic recordings!"). Measured consequence:

- `session_site`: **3,530** rows (the FK target)
- distinct `same_site_id` in `session_same_site`: **2,561**
- of those, grouping >1 session (real chronic sites): **41**
- singletons (a site that is just its own session): **2,520**

So `-> session.SessionSite` guarantees only "this string is a known site row" — one of 3,530,
of which 41 are chronic sites. **It catches a typo and nothing else.** A valid-but-wrong
session id passes silently, which is precisely the failure that produced the original
`roi_group` defect: a plausible identifier accepted without evidence.

The check with teeth is therefore NOT the FK but a membership rule:

> every session declared in `CellRegTask.Session` must map, through
> `session_same_site`, to the task's own `same_site_id`.

That is the constraint that makes a task's columns provably one field of view. It holds for
run 2's four columns today. The FK is still worth declaring — it makes the dependency visible
in the ERD and blocks typos — but it must not be mistaken for the guarantee.

## 4. Table grain and primary keys

- `RegistrationSite` — one row per chronically imaged field of view.
  PK becomes `same_site_id` alone (`-> session.SessionSite`); `subject` leaves the key.
  Secondary `site_description`, `site_id_source`, `created` are unchanged.
  `subject` is recoverable by joining `session_same_site * session.Session`.
- `CellRegTask`, `.Session`, `CellRegExport`, `.File`, `StableCell`, `StableCellMap` —
  grain unchanged; they simply lose the inherited `subject` column.
- `CellRegRun.Site` (new part) — one row per run, carrying the `same_site_id` derived from
  the run's resolved columns, plus `n_sessions_in_site` and how it was derived. A run whose
  columns span more than one samesite gets **no row**, not a guessed one.

## 5. Dependencies and provenance

Upstream: `session.SessionSite` (new FK), `imaging.Curation` (unchanged).
Nothing outside `roselab_cell_registration` references it — measured: **0 outside-schema
foreign keys into our schema**, so the drop is contained.
`site_id_source` keeps recording how a site id was obtained; ROS-1881's current value
`cellreg_folder` is corrected to `same_site_id` only because it was verified against
`session_site`, and that verification is the evidence for the change.

## 6. Failure modes and missing-data semantics

- A site id that is a singleton session rather than a chronic site: **passes the FK**, caught
  only by the §3 membership rule when a task is declared. Documented, not prevented.
- A run spanning two samesites: `CellRegRun.Site` stays empty for that run. Absence means
  "not coherent", never "unknown" — the same convention as `UnresolvedColumn`.
- The 11 runs with no resolved columns get no site, because they have no sessions to derive
  it from. Absence here means "nothing was resolvable", which is already recorded.
- `session_same_site` is mutable (`update1` is used in the ingest GUI). A site regrouped
  after a registration would leave `CellRegRun.Site` stale. Mitigation: `derived_on`
  timestamp on the part row, and the validation query re-runnable; NOT auto-corrected.

## 7. Migration and backward compatibility

**Breaking.** `subject` leaves the key of six tables. DataJoint cannot alter a key or add an
FK in place, so the seven site-branch tables are dropped and redeclared.

Scope, exhaustively: `registration_site` (2 rows), `cell_reg_task` (1),
`cell_reg_task__session` (4), `_cell_reg_export` (1), `_cell_reg_export__file` (4),
`stable_cell` (0), `stable_cell_map` (0) — **12 rows**, all reconstructible: the sites and
task from the values recorded in this note, the export by re-running the exporter over files
already on disk at `/datajoint-data/data/ROS-2101/cellreg_export/sess9FUCURTR_FOV466um_task00`.
No MATLAB re-run, no new CellReg execution.

**Not touched:** `cell_reg_run` and its subtree (8.17M rows across run, result, cell, member,
pair quality, mask coordinates — not downstream of the site), `roselab_session` (read-only FK
target), `roselab_imaging`, `roselab_subject`, `roselab_leondrift`, and
`__segmentation__mask.roi_group`.

Call sites to update in the same PR: `cellreg_plan.plan_site_task`,
`CellRegTask.auto_generate_entries`, `cellreg_register.task_for_result`,
`cellreg_export.export_dir_for`, notebook 18's config cell, notebook 19's census, and the
`cellreg` skill's recipes.

Mapping for anyone holding old keys: `(subject, same_site_id) -> (same_site_id)` is a
projection; the reverse is `session_same_site * session.Session`.

## 8. Validation tests

1. Part-table mechanism — **already verified** on `tobiasr_parttest`: adding part `B` under
   an already-declared master created `m__b` on activation and preserved the master row and
   part `A`'s row. This is what makes `CellRegRun.Site` safe to add.
2. The §3 membership rule as a unit test over run 2's four columns, and as an assertion
   inside `auto_generate_entries`.
3. No run spans two samesites — the query in §2, re-runnable, currently 91/91.
4. Post-migration: the 12 rows come back byte-identical except for the dropped `subject`.
5. `roi_group` fingerprint unchanged across the migration (75,135 rows, 0 non-zero).

## 9. What belongs in DataJoint vs downstream

In: the site identity, the task declaration, the derived per-run site, the membership
validation's *result*. Out: deciding which sessions should form a chronic site — that is a
curation judgement made in the ingest GUI and recorded in `session_same_site`; this schema
consumes it and must never write it.

## 10. Open

- `tobiasr_parttest` was created for test 8.1 and still exists. It is my scratch schema, not
  lab data; awaiting Tobias's word before dropping it, per the standing instruction.
- Run 2's link to its task is a prose `run_note`, not a foreign key. Out of scope here.
