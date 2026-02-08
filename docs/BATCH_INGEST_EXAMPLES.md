# Batch Ingest Examples

Examples live in `examples/batch_ingest/`.

- `01_dlc_pose_batch_ingest.py`: queue and populate DLC tasks
- `02_optitrack_batch_ingest.py`: populate OptiTrack motion capture and rigid mouse tracking
- `03_eye_tracking_gaze_batch_ingest.py`: populate pupil and gaze reconstruction tables
- `04_ca_imaging_batch_ingest.py`: queue Ca2+ processing tasks and populate imaging outputs
- `05_bpod_sync_batch_ingest.py`: ingest Bpod `.mat` files in batch
- `06_aux_stream_batch_ingest.py`: ingest AUX/HARP streams and summarize behavior metadata
- `template_batch_pipeline.py`: starter template for lab-specific batch ingest repositories
