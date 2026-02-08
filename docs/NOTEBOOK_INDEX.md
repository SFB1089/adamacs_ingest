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

Script exports generated from notebooks:
- `notebooks/py_scripts/*.py`
