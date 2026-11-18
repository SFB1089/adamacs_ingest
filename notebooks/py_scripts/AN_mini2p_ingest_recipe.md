# Anh: ingesting mini2p1 sessions into ADAMACS (TATCHU3)

Written 2026-11-18 after the first 14 sessions (ROS-2356 / ROS-2359, 11–16 Nov 2026).
Everything below runs on TATCHU3 with DataJoint calls only. Code lives on branch
`feat/vismotor-bpod-ingest` of `adamacs_ingest` until it is merged.

## 0. At the rig: things that broke the first ingest
| Check | Why |
|---|---|
| Bpod subject = the DB subject, e.g. `ROS-2356` (not `ROS-9956`) | Bpod file and folder names carry the subject. The November typo had to be renamed on disk (55 files). |
| Exactly one Bpod `.mat` from *this* session in the scan folder | A stale `.mat` from another day is refused by the VisMotor sync (no AUX pulses for its trials). `_LM_coord.mat` is fine. |
| Motive: export the Take as CSV into the scan folder | `scan9G38Z5FT` and `scan9G390LYA` have only the `.tak`. Without the CSV there is no frame-count check and no mocap trajectories. |
| Right eye camera: check it delivers an image | In all November sessions it recorded a uniform grey frame with the counter stuck at 0. It cannot be synchronised and is skipped. |
| Keep WaveSurfer running until Motive stops | In `scan9G38ZLH5` WaveSurfer stopped about 8 s early (217,920 of 219,823 OptiTrack frames recorded). |

## 1. GUI ingest (standard route)
1. Copy the session folders to `/datajoint-data/data/anhn/` with the usual name
   `AN_<subject>_<YYYY-MM-DD>_scan<ID>_sess<ID>`.
2. Open `notebooks/00_ingest_gui_workflow_adamacs_ingest_v2.ipynb` and set your initials to `AN`.
   `user_configs/AN.ini` (git-ignored; copy it once from
   `/home/backup_user/adamacs_ingest.worktrees/anh-mini2p-ingest/user_configs/AN.ini` into your clone) pre-fills project `vis2action`, location V1, scanner `mini2p_07_fast`,
   aux setup `mini2p1_openfield`, suite2p paramset **203**, DLC models (top: Topcam_Bodyparts NK 2025-08-01;
   eyes: General_eye_fullsize NK 2025-08-25), cameras top/eye-left/eye-right, and the recording notes
   *plain background* / *zebra background*.
3. Same site: pick the first session of that mouse and FOV (so far `sess9G3608N1` for ROS-2356 and
   `sess9G361E8C` for ROS-2359).
4. Commit. The preflight must show 0 errors. Expect the Bpod step to fail in the GUI; it is done in step 3.

## 2. SUPPORT denoising + denoised suite2p (paramset 1203)
1. Notebook `05_imaging_denoising_pipeline.ipynb`: insert a `DenoisingTask` with `denoise_paramset_idx = 2`
   (model_200.pth, Natasha's model, same as Shan's).
2. Wait until `denoising.Denoising` is filled by the worker, then insert `ProcessingTask` with `paramset_idx = 1203`.
   Any paramset ≥ 1000 makes suite2p read `<scan>/support/`. 1203 = 203 with suite2p `denoise = False`.
3. **Known element bug:** the 1203 run writes into the same `<scan>/suite2p/` folder as 203 and overwrites
   the raw registration on disk. The DB rows of 203 survive if 203 finished first. If you need raw
   registered movies, make them **before** inserting 1203, or copy `suite2p/` **outside** the scan folder.
   Do not keep a copy inside the scan folder: the loader reads every `ops.npy` below it.

## 3. Behaviour and sync (driver script, same DataJoint calls as notebooks 07/09)
Run from the worktree, as yourself (`DJ_PASS` = your DB password in the environment):
```bash
# once: git clone https://github.com/SFB1089/adamacs_ingest ~/adamacs_ingest && cd ~/adamacs_ingest && git checkout feat/vismotor-bpod-ingest
# (use main once the branch is merged; your own clone is needed because the run logs are written into the repo)
cd ~/adamacs_ingest
conda activate datajoint
export PYTHONPATH=$PWD
export DJ_HOST=<lab DB host, see lab wiki>   # the script never reads another user's dj_local_conf.json
read -s DJ_PASS && export DJ_PASS            # your own DB password
export AN_SCANS=scan9G4XXXXX,scan9G4YYYYY        # optional: only the new scans
STAGE=optitrack python notebooks/py_scripts/an_mini2p_ingest.py   # 1. OptiTrack clock (must run first)
STAGE=bpod      python notebooks/py_scripts/an_mini2p_ingest.py   # 2. VisMotor Bpod trials + events
STAGE=events    python notebooks/py_scripts/an_mini2p_ingest.py   # 3. .tak, left-eye OCR, CameraTimestamps, topcam CSV
```
Your DB account can insert but not delete in `trial`/`event`. If something has to be redone, ask Tobias to delete the rows first.
(The script builds its session list from the folders in `/datajoint-data/data/anhn` dated ≥ 2026-11-11;
`AN_SCANS` restricts it.)

* `optitrack`: stores `optitrack_frames` from the WaveSurfer DI line *OPTI gated frames*. The shared
  `ingest_aux` maps 7-DI recordings positionally and would store the 2p volume clock there; this is an
  Anh-only override until the lab fixes the shared mapping.
* `bpod`: VisMotor sync. The AUX `BPOD 2` pulse marks the **end of `DotIsOn`**, not trial start; trials
  are fitted on TrialStart + DotIsOn exit.
* `events`: left eye only (`AN_EYES` overrides). The topdigit OCR fix is on the branch.

## 4. Checks after each ingest
| What | Expected |
|---|---|
| Bpod log line `VisMotor sync: n/n trials paired … residual std` | all trials paired, residual < 0.1 ms (November: 0.014–0.015 ms) |
| `optitrack` stage | `optitrack_frames` = Motive "Total Frames in Take" (240 Hz, 4.15 ms spacing) |
| `event.Event` `mini2p1_eye_left_frames` vs `CameraTimestamps.n_frames` | ≥ 98 % synchronised (November: 98.6–100 %) |
| `imaging.MaskClassification` 203 vs 1203 | both present; 1203 usually has more, larger ROIs |
| `trial.Trial` | one row per Bpod trial; none for sessions without Bpod |

## 5. Known limitations (November data)
* No HARP/IMU: the GUI looks for `*IMU_harp_weardata_20*`, and only `…IMU_harp_weardata_2PFrames…` exists. This is lab-wide (Shan has none since Jan 2026).
* The general eye DLC model tracks the pupil in about 1 % of frames. Retrain on your eye videos before using pupil data.
* SUPPORT-denoised data show stripe artifacts at the left image edge and larger, blob-like ROIs. Compare against 203 before choosing.
* Stimulus-aligned population signals: check for stimulus light reaching the PMT before interpreting them.
