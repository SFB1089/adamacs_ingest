# Batch Ingest Examples

Templates live in `examples/batch_ingest/`.

These scripts are intentionally minimal. Copy one into your own local automation script, edit keys/paths, and run from the repository root.

## Quick usage pattern
1. Copy a template:
```bash
cp examples/batch_ingest/01_dlc_pose_batch_ingest.py local_batch_dlc.py
```

2. Edit key lists/parameters in your copy.

3. Run it from the repo root:
```bash
python local_batch_dlc.py
```

4. Verify expected rows/tasks in DataJoint before scaling up.

## Template map
| Script | Primary operation | Main tables/functions touched |
| --- | --- | --- |
| `01_dlc_pose_batch_ingest.py` | Queue and populate DLC pose tasks | `model.PoseEstimationTaskNew`, `model.PoseEstimationNew.populate()` |
| `02_optitrack_batch_ingest.py` | Populate OptiTrack and rigid mouse tracking | `mocap.MotionCapture.populate()`, `virtual_markers_optitrack.RigidMouseTracking.populate()` |
| `03_eye_tracking_gaze_batch_ingest.py` | Populate eye tracking and gaze reconstruction | `pupil_tracking.PupilEllipseFittingFreeMoving.populate()`, `pupil_tracking.GazeReconstruction3D.populate()` |
| `04_ca_imaging_batch_ingest.py` | Queue and populate imaging processing | `imaging.ProcessingTask.insert1()`, `imaging.Processing.populate()` |
| `05_bpod_sync_batch_ingest.py` | Batch-ingest Bpod files | `adamacs.ingest.bpod.Bpodfile(...).ingest()` |
| `06_aux_stream_batch_ingest.py` | Batch-ingest AUX/HARP streams | `adamacs.ingest.behavior.ingest_aux()` |
| `template_batch_pipeline.py` | Generic starter for custom batch pipelines | Replace placeholder task/populate calls |

## Safety notes
- Most templates perform inserts/populates; test with a small key subset first.
- Keep database credentials in local `dj_local_conf.json` only.
- Do not commit machine-local batch scripts containing sensitive paths.
