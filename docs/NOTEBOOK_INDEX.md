# Ingest Notebook Index (Renumbered)

All notebooks below are ingest-facing examples. Run them only against a database where inserts/populates are intended.

| Notebook | Purpose | Main schema/table focus |
| --- | --- | --- |
| `notebooks/00_ingest_gui_workflow_adamacs_ingest_v2.ipynb` | Center-stage ingest GUI workflow | GUI-driven session ingest |
| `notebooks/01_pipeline_activation.ipynb` | Activate/import the ADAMACS DataJoint pipeline | `adamacs.pipeline` activation |
| `notebooks/02_deeplabcut_ingest.ipynb` | DLC ingest and processing setup | `model` pose-estimation tasks |
| `notebooks/03_imaging_processing_ingest.ipynb` | Ca imaging ingest and processing | `imaging.ProcessingTask`, `imaging.Processing` |
| `notebooks/04_pupil_tracking_ingestion.ipynb` | Pupil-tracking ingest flow | `pupil_tracking` task and output tables |
| `notebooks/05_imaging_denoising_pipeline.ipynb` | Imaging denoising workflow | `denoising` schema |
| `notebooks/06_imaging_cascade_pipeline.ipynb` | Imaging cascade workflow | imaging/model cascade tables |
| `notebooks/07_optitrack_insert.ipynb` | OptiTrack data insertion and population | `mocap` and related tables |
| `notebooks/08_disk_insert.ipynb` | DISK insert workflow | `disk` schema |
| `notebooks/09_eyecam_ingest.ipynb` | Eye camera ingest and linkage | eye cam + session-linked tables |
| `notebooks/10_batch_dlc_eye_ingestion.ipynb` | Batch DLC + eye ingest orchestration | DLC + eye tracking/gaze tables |
| `notebooks/11_optitrack_gaze_repopulation.ipynb` | OptiTrack + gaze repopulation | `virtual_markers_optitrack`, `pupil_tracking` |
| `notebooks/12_slam_worldcam_minimal.ipynb` | Minimal SLAM/worldcam ingest path | worldcam/gaze-aligned ingest tables |
| `notebooks/13_batch_eye_optitrack_ingest_template.ipynb` | Batch eye + optitrack ingest planning (adapted from tobiasr) | `behavior`, `mocap`, `virtual_markers_optitrack`, `pupil_tracking` |
| `notebooks/14_camera_sync_ingest_qc_template.ipynb` | Camera sync ingest QC before population (adapted from tobiasr) | `event`, `behavior.CamSyncRecording`, `model.VideoRecordingNew` |
| `notebooks/15_optitrack_redo_repopulation_template.ipynb` | OptiTrack re-do/repopulation triage (adapted from tobiasr) | `mocap.MotionCapture`, `virtual_markers_optitrack.RigidMouseTracking` |
| `notebooks/16_timestamp_alignment_batch_qc_template.ipynb` | Batch timestamp/alignment QC (adapted from Natasha sync workflows) | `trial.TrialEvent`, `event.Event`, `behavior.CamSyncRecording` |
| `notebooks/20_routine_quick_job_cleanup_template.ipynb` | Routine job cleanup triage template (derived from original notebook 20) | `schema.jobs`, `imaging/model/denoising` task triage |
| `notebooks/21_schema_dependency_diagrams.ipynb` | Render and export `dj.Diagram` dependency maps schema-by-schema | All core schemas with parent-key context |

Script exports generated from notebooks:
- `notebooks/py_scripts/*.py`
