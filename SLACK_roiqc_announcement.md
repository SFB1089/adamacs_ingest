*New in `imaging`: ROI QC — please exclude ROIs without fluorescence* :mag:

Some suite2p ROIs have *no fluorescence of their own*. Suite2p leaves out of F every pixel an ROI shares with another ROI. If all of an ROI's pixels are shared, its F is 0 in every frame. Its dF/F0 and CASCADE trace are then pure *neuropil*, and they look like a nicely responsive cell.

*How big*
• 61,511 of 3.65 M ROIs (1.7 %), 17,931 of them classified as soma
• 1,615 of 3,250 curations have at least one; 174 have 50 or more (mostly ROS-1985, ROS-2097, ROS-2101)
• 26 curations have more than half of their ROIs affected

*New tables* (`roselab_imaging`, element `roselab-0.1.0b2`)
• `imaging.RoiQC`: per curation, how many ROIs were checked and rejected
• `imaging.RoiQC.Rejected`: one row per rejected ROI
• `imaging.RoiQCCriterion`: the reasons. The first is `empty_f`; more QC reasons can be added later as rows

*What to do*
```
rejected = imaging.RoiQC.Rejected.proj()
dff    = (imaging.Dff0.Trace & key & {"dff_paramset_idx": 2}) - rejected
spikes = (imaging.Activity.Trace & key & {"extraction_method": "cascade_inference_25ms_filter"}) - rejected
traces = (imaging.Fluorescence.Trace & key) - rejected
```
• New dF/F0 and CASCADE (from today) are NaN for these ROIs.
• *Everything stored before today is not*: please exclude `RoiQC.Rejected` in your queries, including suite2p traces.
• Separately, `Dff0.Trace.f0_nonpositive` marks dF/F0 traces whose baseline is ≤ 0 somewhere (36k more traces). Drop those too unless you know why you need them.

Details: `docs/ROI_QC.md` in adamacs_ingest, and the diagrams notebook `notebooks/21_schema_dependency_diagrams.ipynb`. Questions → me.
