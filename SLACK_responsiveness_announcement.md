*New schema: `roselab_responsiveness`, for QC, responsiveness, reliability and tuning per cell* :chart_with_upwards_trend:

One place for the questions every stimulus analysis asks: is the cell OK, does it respond, is the response reliable, what is it tuned to. Everything is keyed on the curation and the trace type (CASCADE 25 ms, OASIS, dF/F0), so it joins straight to `imaging`.

*What's computed*: the latest curation of every gratings, natural-image and checkerboard recording. The worker is filling it now; complete in about 1.5 days.

*Main tables*
• `ScanQuality`, `Quality`, `TraceQuality`: detector gain, neuropil, kurtosis, photons/s, SNR
• `EventSet`, `ScreenTiming`: stimulus onsets from the trial schema, and the photodiode time the stimulus really was on screen
• `Readout`: per-trial response in an explicit window (`ReadoutParamSet`), test vs baseline (`resp_p`), oracle reliability (`oracle_r`), direction (`resp_delta` > 0 = activated)
• `Zeta`: ZETA on the epoch of a read-out, so ZETA results compare across window settings
• `DirectionTuning`: orientation / direction tuning (Mazurek et al. 2014)
• `CellRule` + `cells()`: named thresholds; ROIs without fluorescence (`imaging.RoiQC`) are left out

*Standard windows*
• gratings: read-out 16 (stimulus on screen; baseline 2.0–0.33 s before)
• images: read-out 11 (CASCADE) / 3 (OASIS), with ZETA
• checkerboard: read-out 1
• ready-made rules: `gratings_cascade25`, `images_oasis`, …; add `_pos` for activated cells only

*How to use*
```
from adamacs.workers import responsiveness_worker as w
resp, imaging = w.activate("roselab_responsiveness")
good = resp.cells("gratings_cascade25", {"event_selector": "gratings"}) & "is_responsive AND is_reliable"
pos  = resp.cells("gratings_cascade25_pos", {"event_selector": "gratings"})        # activated only
tr   = (resp.Readout.Cell & cell & {"readout_idx": 16}).fetch1("trial_resp")        # per-trial responses
```
Want more recordings or another window? `resp.plan_tasks(...)` stages them, and the worker picks them up.

*Good to know*
• dF/F0 is for display and tuning only; make responsiveness calls on CASCADE / OASIS (or ZETA).
• Many grating cells respond to the *offset* of the grating. Read-out 16 doesn't see that, ZETA does.

*Notebooks*
• Guided tour with example PSTHs, all tested windows and the review findings: https://github.com/SFB1089/adamacs_ingest/blob/main/notebooks/responsiveness_v8_tour.ipynb
• Schema diagrams and descriptions: https://github.com/SFB1089/adamacs_ingest/blob/main/notebooks/21_schema_dependency_diagrams.ipynb
• Design and decisions: https://github.com/SFB1089/adamacs_ingest/blob/main/docs/RESPONSIVENESS_SCHEMA_PROPOSAL.md
A sanity-check notebook of the production run follows when the run is complete. Questions → me.
