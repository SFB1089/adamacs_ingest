# Facemap schema: proposal

**Status:** implemented 2026-11-26 (`adamacs/schemas/facemap.py`, worker `adamacs/workers/facemap_worker.py`, tests `tests/test_facemap.py`); decisions in section 7; section 9 has the implementation notes. The table definitions are in
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
| site run itself | 32 (16 sites × 2 stimulus types) `analysis/facemap_registered_videos/<sess>_<stim>/unified_registered_video_proc.npy` | | the basis | 11 GB |

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
  .Position                     a segment's position in its run's basis, from ALL components (Leon's SessionFacemap measures)
SegmentDistance (Computed)      pairwise segment distances within one run and region (Leon's svd_distances)
```

**Every run is addressable on its own.**
- Each of the 32 site runs and the 231 per-video runs is a separate `FacemapTask` / `FacemapProcessing` row.
- A scan that is in both generations has two runs. They differ in `facemap_task_id` and in the paramset's
  `video_assembly`, and they are never merged.
- `runs_of(video, assembly=...)` lists the runs containing a video, optionally of one generation.

**Cross-session comparisons stay within one run.** Temporal components are comparable only between segments
of the same run, because only they share a basis. `FacialSignal.Position` and `SegmentDistance` are keyed by the run,
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
2. **32 site runs**: one `load` task each, with the site-level
   `analysis/facemap_registered_videos/<sess>_<stim>/unified_registered_video_proc.npy` as the run file. The
   manifest comes from `define_scans` (primary-key order) and is checked against
   `roselab_leondrift.SessionFacemap.concat_index` where it exists (84 scans).
3. **Leon's 294 per-scan split files** become `Segment.split_file`. The ingest checks each against the run's
   slice, and slices the May-2026 files that still hold the whole `motion` trace.
4. **231 per-video runs**: one `load` task each, with the `leon_per_video_2026` paramset and the per-scan
   `<scan>_bonsai_face_video_<ts>_proc.npy` as the run file.
5. `FrameSync` for every applicable method; `FacialSignal.Position` and `SegmentDistance`; compare with
   `roselab_leondrift.SessionFacemap` (all 168 rows should match).
6. Set `Segment.video_frame_offset` from motion energy against face-marker speed (done 2026-11-27, section 10).

**Storage** with 100 components: traces 2 × 4 bytes × 100 per frame (about 0.24 GB per 295,000-frame scan);
masks about 0.4 GB per run and SVD (904 × 1148 × 100 float32; DataJoint blobs have no float16), so about 0.8 GB
per run. In total roughly 300 GB for the 32 site runs and 231 per-video runs; the blob store has 3.7 TB free.
The full files stay on disk, referenced with their sha256.

## 7. Decisions (2026-11-26)

1. **100 components** are stored per SVD (temporal and spatial). All singular values are stored.
2. **Both generations are ingested**: the 32 site runs and the 231 per-video runs, each addressable on its own.
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

## 9. Implementation notes (2026-11-26)

- **Run identity in the task id** (Codex review of PR #43): `facemap_task_id` hashes the paramset, the ordered
  manifest and `run_identity`. For `load` tasks that is the result file (`file:<path>|<bytes>|<mtime>`), so a
  regenerated file is a new run; for `trigger` tasks an explicit `instance:<n>`, so a deliberate re-run is a new
  row. Planning the same run twice still gives the same id.
- **Every array is in the external store** (`blob@external-raw`): masks, singular values, traces, positions, frame
  times, average frames. Only the small parameter dictionary of `FacemapParamSet` is inline.
- **Positions use all components**, computed during the import from the full arrays, so Leon's 500-component
  medians and distances can be reproduced; only the first 100 components of traces and masks are stored.
- **Three-part make** (`make_fetch` / `make_compute` / `make_insert`) for `FacemapProcessing` and `FacialSignal`:
  file reads and hashing run outside the insert transaction.
- **Leon's split files** are checked one at a time in manifest order (each segment's offset is known when its
  file is read), so memory stays at one run plus one split file.
- **Where it runs**: TATCHU3, env `datajoint_analysis` (numpy 2: Leon's files are numpy-2 pickles), from a
  directory holding `dj_local_conf.json`, with `--no-tls` (that env's OpenSSL fails the server's TLS handshake).
  ```
  cd <dir with dj_local_conf.json>
  PYTHONPATH=<adamacs_ingest checkout> python -m adamacs.workers.facemap_worker --schema roselab_facemap --no-tls plan --insert
  PYTHONPATH=<adamacs_ingest checkout> python -m adamacs.workers.facemap_worker --schema roselab_facemap --no-tls populate --once
  ```

## 10. Facemap rows vs video frames: 15 segments shifted by 4 frames (2026-11-27)

**Ingest status.** All 263 runs are in `roselab_facemap` (2026-11-27): 32 site runs, 231 per-video runs, 2,100
`FrameSync` rows, no errors. Against `roselab_leondrift.SessionFacemap` (168 scans): medians within 0.001, temporal
references and motion medians identical. All 294 per-scan split files equal their slice of the site run.

**Method.** For every site segment, the lag L that maximises the correlation between log(1 + facemap motion energy)
at row r and log(1 + DLC face-marker speed) at video frame r + L, for L in −15…+15. Marker speed is the median over
the face markers of the frame-to-frame displacement, frames with likelihood < 0.9 excluded. Both signals come from
the same face video, so the lag measures the facemap split, not camera timing. When a scan has several face DLC
models, the one with the best median likelihood is used (some scans also carry an untrained model with likelihood ≈ 0,
which first produced spurious lags of −2 and +2).

**Result** (294 site segments): 278 at lag 0; 15 at +4, each improving the correlation by 0.04–0.08; one at +1 with
no improvement (noise, left at 0). Only lags that improve the correlation by ≥ 0.02 are stored.

| site run | segments at +4 | last segment (4 rows short) |
|---|---|---|
| ROS-1912 | scan9FQBJFU7 … scan9FQIOMIX (7) | scan9FQIOMIX |
| ROS-2356 | scan9G2ISHXX … scan9G2OQSVH (6) | scan9G2OQSVH |
| run with scan9FVRV8VM | scan9FVMJBHH, scan9FVRV8VM (2) | scan9FVRV8VM |

**Explanation.** In each run the segments are at lag 0 up to one boundary and at +4 from there on, and only the last
segment has 4 rows fewer than its video. Leon's concatenated (registered) video lost 4 frames at that boundary, and
the split by the original per-video frame counts carried the shift into every later segment. In ROS-1912 and
ROS-2356 the boundary follows scan9FQACAG8 and scan9G2HLHF1, two of the four recordings with an unusual count of
unsaved camera triggers. Leon's per-video runs of the same scans are not shifted.

**Stored.** `FacemapProcessing.Segment.video_frame_offset = 4`, `offset_source = 'motion_vs_dlc'` for the 15
segments; their `FrameSync` rows were recomputed. In segments shifted mid-run, the last 4 rows belong to the next
video and get no time; in the last segment of a run, row 0 gets the time of video frame 4. `FacialSignal.Position`
of these segments is unchanged: it is computed over Leon's split, which includes 4 rows of the neighbouring video
(as his own `SessionFacemap` values), a negligible effect on medians.

**Consequence.** Leon's split traces of these 15 scans are 4 frames (67 ms) off their videos: negligible per trial,
not frame by frame. Use `FrameSync`, which applies the offset.

## 11. Choosing the frame-timing method (2026-11-27)

`FrameSync` stores four methods per segment; `choose_sync_method` / `frame_times(segment_key)` in
`adamacs/schemas/facemap.py` pick one:

1. `camera_frames` when the stored per-frame events match the video (440 of 525 segments);
2. `start_anchored` when the count of unsaved in-gate triggers is outside 121 ± 5;
3. `gate_rule` otherwise.

**Why the count decides.** The gate rule assumes the video ends at the gate end. Where the unsaved count is off, it
does not, and the gate rule is off by (121 − unsaved) / 60 s. 16 recordings in this schema have such a count (82–165,
and three with thousands of unsaved triggers); none has stored events. They were checked with body-camera paw speed
(DLC) against treadmill speed, which is independent of the face camera: the start anchor puts the peak at −0.03 to
+0.04 s in all 15 testable recordings (scan9FVK3VDN has no body paw model); the gate rule is up to +0.73 s off and finds
no match at all for scan9FW07FNQ, scan9FXTKOX2 and scan9FXTNTHG. The full table is in v1-drift-audit
`docs/camera_timing_offcount_sessions.md`; notebook `notebooks/22_facemap_tour.ipynb`, section 8b, recomputes it.
