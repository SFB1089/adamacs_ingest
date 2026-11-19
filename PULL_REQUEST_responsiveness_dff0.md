## What

Phase 1 of a lab-wide responsiveness schema: **dF/F0 as a stored, parameterised table**, plus the link from the stored CASCADE activities back to the dF/F0 they were computed from. The full design is in `docs/RESPONSIVENESS_SCHEMA_PROPOSAL.md`. It covers dF/F0, events and trials, ZETA and window statistics, rules and cell categories.

- `adamacs/responsiveness/dff0.py`: one pure implementation of dF/F0 (numpy/scipy only, no `adamacs.pipeline`).
  - One recipe per parameter set, with one column group per step: dark signal, neuropil factor and offset, detrend, baseline, F0 floor.
  - Recipes are validated on insert: unknown keys, missing parameters and parameters the chosen method ignores are refused. They are also hashed.
  - The 10 recipes found in lab code are included.
- `adamacs/schemas/responsiveness.py`: nothing is declared until `activate(schema, imaging=, scan=, event=, store=)` is called. Tables:
  - `DffParamSet`, `Dff0Task`, `Dff0` (+ `.Trace`);
  - `CascadeInput`: each stored CASCADE activity mapped to its recipe, dated from the external-store timestamp;
  - `CascadeRecompute`: the verification results.
- `scripts/populate_dff0_test_schema.py`: populates the test schema.
- `scripts/verify_dff0_cascade.py` + `scripts/cascade_predict_npz.py`: re-run CASCADE on BUGPU (CPU only) and compare with `imaging.Activity`.
- `tests/test_responsiveness_dff0.py`: 16 offline tests. They include line-by-line equivalence with the CASCADE branch of `Activity.make`.

## Why

- Until now, dF/F0 was computed inside `imaging.Activity.make` for CASCADE and then thrown away.
- Its parameters came from string rules on the extraction-method name.
- The dark-signal code was edited in place on BUGPU on 2026-09-03:
  - before that date, the dark window was always frames 12–24, because of a Decimal/float error and swapped arguments;
  - after it, the shutter window.
- Nothing recorded which code version wrote which row, so one extraction method holds rows from both versions.

## Verified (test schema `tobiasr_responsiveness`, 15.11.2026)

- **Coverage:** 3 curations × 8 recipe rows. CASCADE was re-run with the worker's checkout (Cascade `4c113e4`).
- **Reproduced to float precision:** the stored `cascade_inference`, `_25ms_filter` and `_global_causal` traces (max |Δ| ≈ 3e-4) when the recipe matches the write date. The only exceptions are masks with F0 ≤ 0, which `Dff0.Trace.f0_nonpositive` flags.
- **Not reproduced:** the stored `cascade_inference_11` rows did not correspond to their own masks.
  - Their same-mask r against `_25ms_filter` was ≈ 0 on 20/20 curations; the control gives 0.97.
  - They were deleted from `imaging` on 2026-11-15 on Tobias's instruction. Manifest: `~/adamacs/out/cascade11_deleted_2026-11-15.csv` on TATCHU3.

## Notes for reviewers

- **No production schema is declared or changed by this PR.** Production (`roselab_responsiveness`) needs a separate decision; open points are in §6 of the proposal.
- **Upstream modules are passed in.** This lets the schema be tested with virtual modules, without importing `adamacs.pipeline`, which writes on import.
- **The blob store name is an activation argument.** The test run used a scratch store outside `/datajoint-db`.
- **Out of scope here, but in the proposal:**
  - the CASCADE worker on BUGPU should record its dF/F0 parameters;
  - the TATCHU3 element copies still carry the old dark-signal bug;
  - `populate_tables_cascade.py` contains Slack webhook URLs in clear text.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
