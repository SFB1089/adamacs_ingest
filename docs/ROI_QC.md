# ROI QC in `imaging`: `RoiQCCriterion` and `RoiQC`

Element-calcium-imaging `roselab-0.1.0b2` (SFB1089 fork, PR #7), live since 2026-11-19 in `roselab_imaging`.

## Why

Suite2p leaves out of F every pixel that an ROI shares with another ROI. When **all** of an ROI's pixels are shared,
its fluorescence is 0 in every frame (already in suite2p's `F.npy`), and every trace derived from it is neuropil:

- dF/F0 recipe 2 becomes −0.7·(Fneu − median) on a negative F0. This is inverted twice, so it follows the neuropil and
  looks like a responsive cell.
- CASCADE then infers spikes from that neuropil trace.
- suite2p's own deconvolution (`suite2p_deconvolution`) is neuropil as well.

## The tables

| Table | Tier | What it holds |
|---|---|---|
| `RoiQCCriterion` | Lookup | one row per reason to reject an ROI; `empty_f` = fluorescence 0 (or non-finite) in every frame |
| `RoiQC` | Computed | per curation × criterion: `n_rois` checked, `n_rejected` |
| `RoiQC.Rejected` | Part | one row per rejected ROI (`mask`, `fluo_channel`) |

The TATCHU3 worker (`populate_tables.py`) fills `RoiQC` for new curations, at most 25 per loop.

## State on 2026-11-19 (criterion `empty_f`)

- **Coverage:** all 3,250 curations checked.
- **Rejected:** 61,511 of 3.65 M ROIs (1.7 %), of which 17,931 are classified soma.
- **Where:**
  - 1,615 curations have at least one;
  - 174 have 50 or more, most in ROS-1985 (57 curations), ROS-2097 and ROS-2101;
  - 26 have more than half of their ROIs rejected.
- **Stored traces on these ROIs:** 6,966 dF/F0 recipe-2 traces and 6,966 CASCADE 25 ms traces.
- **Not the only cause of negative baselines:** of the 42,923 dF/F0 traces with F0 ≤ 0, only 6,945 are empty ROIs. The
  other 36,074 have another cause and are not flagged by `empty_f`; `Dff0.Trace.f0_nonpositive` marks them.

## How to use it

```python
from adamacs.pipeline import imaging
rejected = imaging.RoiQC.Rejected.proj()            # any criterion; or & {"roi_qc_criterion": "empty_f"}

traces = (imaging.Fluorescence.Trace & curation_key) - rejected
dff    = (imaging.Dff0.Trace & curation_key & {"dff_paramset_idx": 2}) - rejected
spikes = (imaging.Activity.Trace & curation_key & {"extraction_method": "cascade_inference_25ms_filter"}) - rejected

(imaging.RoiQC & curation_key).fetch(as_dict=True)  # n_rois / n_rejected per criterion
dff    = dff & "f0_nonpositive = 0"                  # also drop traces whose dF/F0 baseline is <= 0 somewhere
```

## What changes, what does not

- **From 2026-11-19 on,** `Dff0` and CASCADE store NaN for `empty_f` ROIs.
- **Rows stored before then are not blanked:** exclude `RoiQC.Rejected` in every query. The same holds for suite2p's
  traces, which come straight from suite2p.
- **The responsiveness schema** (`cells()`) leaves out `RoiQC.Rejected` automatically.
- **A new QC reason** is a new `RoiQCCriterion` row plus a branch in `element_calcium_imaging.dff.roi_qc_reject`,
  shipped as an element release. `RoiQC` computes it without touching the existing criteria.

## See also

- Diagrams and live counts: `notebooks/21_schema_dependency_diagrams.ipynb`.
- Examples: section 12.3 of `notebooks/responsiveness_v8_tour.ipynb`.
- dF/F0 tables: README, "Stored dF/F0 and CASCADE".
