# Facemap schema: proposal

**Status:** draft for review, 2026-11-26; decisions of 2026-11-26 included (section 7). Nothing is declared. The table definitions are in
`adamacs/schemas/facemap.py`; the ingest code (`make()`) follows after the design is approved.

**Goal:**
- Now: ingest the facemap output that exists on disk, Leon Kremers' alignment to the DAQ clock, and his
  cross-session treatment (one basis per site and stimulus type).
- Later: the same tables are filled automatically for new recordings, by a worker that runs facemap.

## 1. What exists today (verified 2026-11-26, read-only)

**In the database**
- `roselab_leondrift.SessionFacemap`: 168 scans of 8 animals, per-scan summaries only (median SVD position,
  within-session references, motion median, checks against Leon's numbers). Written by
  `LeonDriftAnalysis/src/leondrift/covariates/facemap.py`.
- No facemap table in any production schema. The per-frame traces exist only in files.
- The face videos are already registered: `model.VideoRecordingNew` (camera `bench2p_face`, 1,007 recordings)
  with `RecordingInfoNew.nframes` and fps.
- Each camera has its own frame events in `event.Event` (`bench2p_face_frames`, `bench2p_body_frames`,
  `bench2p_back_frames`), next to the shared trigger `aux_cam`.

**On disk:** facemap folders for 376 of Leon's scans and 1 of aeltona's. Two generations sit side by side:

| generation | files | dates | basis | size |
|---|---|---|---|---|
| per video | 231 `<scan>_bonsai_face_video_<ts>_proc.npy` | 2025-10 to 2026-04 | one per video | 10.8 GB |
| site concatenation | 294 `<scan>_face_cam_concatenate_proc.npy` | 2026-05 to 2026-11 | shared per site and stimulus type | 5.0 GB |
| site run itself | 34 `analysis/facemap_registered_videos/<sess>_<stim>/unified_registered_video_proc.npy` | | the basis | 11 GB |

**How Leon's site runs were made** (`LeonKremers-adamacs/notebooks/54_Leon_Behavior_runAnalysis.ipynb`)
1. Scans of a site and stimulus type are listed with `define_scans` (primary-key order).
2. The face videos are histogram-matched to video 1, registered by translation (ECC on 100-frame averages),
   cropped to the shared region and written as one `unified_registered_video.mp4`.
3. Facemap runs on it (Leon's fork; `sbin=1`, motion and movie SVD, 500 components, no ROIs, an
   eye-exclusion mask from DLC eye corners and the pupil ellipse).
4. The temporal components are split back per scan with `np.cumsum(frame_counts)`. Each per-scan file is a
   copy of the site result with that scan's rows; the offsets are not stored.
5. Files written before a later fix (May 2026) still hold the whole concatenated `motion` trace.

The registration transforms and the histogram mapping are not saved anywhere.

## 2. Design basis

**From the CellReg ingest**
- **The run is the entity.** A re-run is a new row, never an overwrite. Which run an analysis uses is chosen
  downstream.
- **Runs are content-addressed** by the sha256 of their file. A file reached by several paths is one run with
  `.Copy` rows.
- **The manifest is declared, not inferred.** `FacemapTask.Video` names each video and its position.
- **Parameters are a content-hashed lookup**; a new facemap setting is a row, not a migration.
- **Quality and alignment are rows with columns**, not values hidden in blobs.
- **Nothing is declared on import.**

**From DataJoint's element-facemap** (github.com/datajoint/element-facemap, 0.3.0, updated 2026-11-10)
- Names and concepts are taken over where the meaning is the same: `FacemapTask` → `FacemapProcessing` →
  `FacialSignal` with the parts `Region` (region 0 = the full frame, others = facemap ROIs) and `Summary`
  (sbin, average frame, average motion).
- The element is not used as is, for three reasons:
  1. Its `FacemapTask` belongs to one `VideoRecording` of one session, so a basis shared across sessions, as
     in Leon's site runs, cannot be expressed.
  2. It declares its own session-keyed `VideoRecording`. Here the lab's scan-keyed `model.VideoRecordingNew`
     is referenced, so DLC, pupil and facemap share one video identity.
  3. It stores one row per component, with mask and trace inline: 500 × 2 rows per region, about 3.5 GB per
     run. Here masks and traces are arrays in the external store, limited to the first 100 components.

## 3. Tables

```
FacemapParamSet (Lookup)        how a run is made; video_assembly 'per_video' | 'site_concat_registered'
                                seed rows: leon_per_video_2026, leon_site_concat_2026
SyncMethod (Lookup)             camera_frames | gate_rule | start_anchored | aux_cam_first_n (Leon's original)
FacemapTask (Manual)            one planned run; id = hash(paramset, ordered videos)
  .Video                        the manifest: segment_index -> model.VideoRecordingNew
FacemapProcessing (Imported)    the run file: path, sha256, size, frame totals
  .Copy                         other paths with the same file
  .Segment                      per video: frame offset, frame counts, facemap-vs-video offset,
                                Leon's split file and its check
FacialSignal (Imported)         the results, per region
  .Summary                      sbin, ly, lx, average frame and average motion
  .Region                       region 0 = full frame; ROI and exclusion masks
  .Basis                        per region and SVD: all singular values, first 100 spatial masks
  .Trace                        per region and segment: motion energy, first 100 motSVD and movSVD components
FrameSync (Computed)            DAQ time of every facemap row, per SyncMethod, with count checks
SegmentSummary (Computed)       a segment's position in its run's basis (Leon's SessionFacemap measures)
SegmentDistance (Computed)      pairwise segment distances within one run and region (Leon's svd_distances)
```

**Every run is addressable on its own.**
- Each of the 34 site runs and the 231 per-video runs is a separate `FacemapTask` / `FacemapProcessing` row.
- A scan that is in both generations has two runs. They differ in `facemap_task_id` and in the paramset's
  `video_assembly`, and they are never merged.
- `runs_of(video, assembly=...)` lists the runs containing a video, optionally of one generation.

**Cross-session comparisons stay within one run.** Temporal components are comparable only between segments
of the same run, because only they share a basis. `SegmentSummary` and `SegmentDistance` are keyed by the run,
so a query cannot mix bases by accident. Per-video runs have one segment each, so they give no cross-session
distances. Leon's `SessionFacemap` can be rebuilt from the site runs and checked against his values.

**Frame timing per recording** (`FrameSync`, one row per method and segment):

| method | rule | use for |
|---|---|---|
| `camera_frames` | stored `bench2p_face_frames` events, frame i = event i | recordings that passed the audit's timing QC |
| `gate_rule` | the last n `aux_cam` triggers inside `main_track_gate` | recordings without stored events and ~121 unsaved triggers |
| `start_anchored` | frame i = trigger 121 + i after the gate opens; frames past the last trigger get NaN | the 4 audited recordings with another unsaved count |
| `aux_cam_first_n` | the first n `aux_cam` triggers | reproducing Leon's old results only |

Which method an analysis uses is a downstream choice. Each row records the event count, the unsaved count and
how many rows got no time, so the choice can be made from the data.

## 4. Clock alignment

**History**
- Leon's original code timed camera frames with the first n `aux_cam` triggers. `aux_cam` starts before Bonsai
  saves, so every camera value was 2.9–6.0 s early (median 3.7 s).
- The v1-drift-audit found this on 2026-09-29/30. Since then the per-frame events `bench2p_face_frames` /
  `_body_` / `_back_` are in `event.Event` for 1,066 recordings that passed its timing QC. The rule: video
  frame i = the last n `aux_cam` triggers inside `main_track_gate` ("gate rule").
- Leon confirmed it on 2026-09-30 and uses the gate rule in `get_cam_frame_times` since then.
- On 2026-11-17 the audit found 4 recordings where the gate rule is off by 0.1–0.7 s because their count of
  unsaved triggers is not ~121. A start-anchored rule (frame i = trigger 121 + i after the gate opens) fits
  them; it was found post hoc. Posted to the lab on 2026-11-26 (thread of 2026-11-25 in
  #vc-lgn-representational-stability).

**Checked for this proposal on Leon's 168 facemap scans (2026-11-26)**

| | scans |
|---|---|
| `bench2p_face_frames` are an exact contiguous run of `aux_cam` triggers | 163 |
| facemap rows == face-camera events | 162 |
| face camera starts after the first `aux_cam` trigger by | 2.9–6.0 s (median 3.67 s) |
| no face-camera events: the 5 recordings that failed the audit's QC | scan9FOYO7M7, scan9FPZHKKB, scan9FQ1UYJD, scan9FQ7T8SV, scan9FQACAG8 |
| 4 more events than facemap rows | scan9FQIOMIX (excluded from the audit's facemap export) |

In two of the start-anchored recordings (scan9FPZHKKB, scan9FQ1UYJD) the last 20 and 22 video frames have no
trigger at all, so they get no time (NaN).

**Consequence for the schema:** `FrameSync` stores all applicable methods side by side (section 3), with the
counts that decide between them. Nothing re-derives times from frame rates.

## 5. Parallel runs and parallel ingests

- **Several runs per video:** the per-video run, the site run, and future runs with other parameters are
  separate `FacemapRun` rows. Nothing is keyed by the video alone, so nothing is overwritten.
- **Several people planning:** the task id is a hash of the paramset and the ordered manifest. Planning the
  same run twice gives the same id, and the second insert is skipped.
- **Several workers:** the Imported and Computed tables populate with `reserve_jobs`. The 5-11 GB reads and
  the sha256 run outside the insert transaction (DataJoint's three-part make), so a long read never holds
  locks that other workers wait on. That is the lock-timeout seen in the responsiveness run on 2026-11-24.
- **Duplicate files:** content addressing turns byte-identical copies into one run with `.Copy` rows.
- **New runs (trigger mode):** each task writes into its own directory, `<scan or site dir>/facemap/<task_id>/`.
  Today a re-run overwrites the previous file of the same name.

## 6. Ingest plan for the existing data

1. Seed the two paramsets.
2. **34 site runs**: one `load` task each, with the site-level
   `analysis/facemap_registered_videos/<sess>_<stim>/unified_registered_video_proc.npy` as the run file. The
   manifest comes from `define_scans` (primary-key order) and is checked against
   `roselab_leondrift.SessionFacemap.concat_index` where it exists (84 scans).
3. **Leon's 294 per-scan split files** become `Segment.split_file`. The ingest checks each against the run's
   slice, and slices the May-2026 files that still hold the whole `motion` trace.
4. **231 per-video runs**: one `load` task each, with the `leon_per_video_2026` paramset and the per-scan
   `<scan>_bonsai_face_video_<ts>_proc.npy` as the run file.
5. `FrameSync` for every applicable method; `SegmentSummary` and `SegmentDistance`; compare with
   `roselab_leondrift.SessionFacemap` (all 168 rows should match).
6. Set `Segment.video_frame_offset` from motion energy against face-marker speed, where facemap and video
   frame counts differ (11 scans of ROS-1912 and ROS-2356, about 4 frames).

**Storage** with 100 components (295,000 frames per scan): about 0.24 GB of traces per segment, so about
70 GB for the 294 site-run segments and about 55 GB for the 231 per-video runs. Masks: about 0.35 GB per run
and SVD at float16. The full files stay on disk, referenced with their sha256.

## 7. Decisions (2026-11-26)

1. **100 components** are stored per SVD (temporal and spatial). All singular values are stored.
2. **Both generations are ingested**: the 34 site runs and the 231 per-video runs, each addressable on its own.
3. **Schema name `roselab_facemap`.** element-facemap would be activated under a different name, so nothing
   clashes if it is ever used for other rigs.
4. **The start-anchored rule** is a `SyncMethod` row next to the stored events and the gate rule.

Still open:
- A table of traces binned to 2P frames (Leon's mean per 2P frame with forward fill): proposed later, as a
  Computed table with a method lookup.
- Leon has been told about the timing rule on Slack (thread of 2026-11-25 in #vc-lgn-representational-stability).

## 8. Found on the way

- `model.VideoRecordingNew.File` for `scan9FOWAIBH_bench2p_back` points to the **face** video file. The back
  camera's recording has the wrong file at least for this scan; to be checked across recordings.
