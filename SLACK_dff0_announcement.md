*dF/F0 is now a table in the pipeline, and CASCADE reads it* :tada:

Since 15 Nov, dF/F0 is no longer computed and thrown away inside the CASCADE populate. It is stored in `roselab_imaging`, with explicit parameters.

*What's new*
• `imaging.Dff0.Trace`: dF/F0 per mask, stored once per curation and recipe (`dff`, `f0_median`, `f0_min`, `f0_nonpositive`)
• `imaging.DffParamSet`: one row per dF/F0 recipe (dark signal, neuropil factor/offset, detrending, baseline, F0 floor)
• `imaging.CascadeRecipe`: which recipe each CASCADE method uses
• `imaging.Activity.CascadeInput`: for every new CASCADE trace, the exact dF/F0, model and code version it came from

*What this means for you*
• CASCADE results don't change. `cascade_inference`, `_25ms_filter` and `_global_causal` use the same dF/F0 as since September (recipe 2). Recomputing the stored traces confirmed this to float precision.
• Please stop computing dF/F0 yourself. Use `imaging.Dff0.Trace & key & {"dff_paramset_idx": 2}`. If your curation has no row yet, insert an `imaging.Dff0Task` row (or ping me).
• If you need a different dF/F0 (other neuropil factor, baseline, …), add a new recipe with `imaging.DffParamSet.insert_new_params(...)`. Never edit an existing one.
• `stack_helpers.calculate_dFF` still works, but it is deprecated.
• Queuing CASCADE is unchanged: insert `ActivityCascadeTask` rows.

*Two exclusions for every query* (update 19 Nov)
• ROIs without fluorescence: exclude `imaging.RoiQC.Rejected` (see the ROI QC post). From 19 Nov, their dF/F0 and CASCADE are stored as NaN; older rows still hold neuropil.
• Negative baselines: `f0_nonpositive = 1` marks traces whose F0 is ≤ 0 somewhere (about 43k of 1.03 M recipe-2 traces). There dF/F0 changes sign. Drop them unless you know why you need them.
```
dff = (imaging.Dff0.Trace & key & {"dff_paramset_idx": 2} & "f0_nonpositive = 0") - imaging.RoiQC.Rejected.proj()
```

*Housekeeping*
• element-calcium-imaging `roselab-0.1.0b2` (19 Nov) is installed in all lab envs on tatchu3 and ibehavegpu1, and the workers were restarted. If you have a long-running Python session with the pipeline loaded, restart it.
• The old `cascade_inference_11` traces were removed. They did not match their own masks, and the drift analyses don't use them.
• Two things we learned the hard way:
  ◦ please don't edit installed element files in place: change the fork, tag a release, reinstall;
  ◦ a pip upgrade does not reach a worker that is already running, so restart it.

Details: README section "Stored dF/F0 and CASCADE" and `docs/DFF0_IN_ACTIVITY_MAKE_PROPOSAL.md` in adamacs_ingest; diagrams in `notebooks/21_schema_dependency_diagrams.ipynb`.
