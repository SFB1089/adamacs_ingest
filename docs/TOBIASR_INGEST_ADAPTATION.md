# Tobiasr Ingest Notebook Adaptation Map

This document summarizes how `adamacs/notebooks/tobiasr/*` ingest workflows were reviewed and adapted into reusable ingest templates in `adamacs_ingest`.

## Source families reviewed
- `notebooks/tobiasr/data ingest updates/*`
- `notebooks/tobiasr/Natasha/*` (ingest/sync/optitrack-related notebooks)
- `notebooks/20_manual_step_by_step_ingest_bpod_trial.ipynb` (job-cleanup scratchpad)

## Main source intents identified
- batch ingest for eye + optitrack modalities
- camera sync and timestamp alignment troubleshooting
- optitrack reconstruction re-runs/repopulation
- fast triage and cleanup of job table failures and stranded tasks

## Adapted notebooks in `adamacs_ingest/notebooks`
| New notebook | Primary source notebooks | Purpose |
| --- | --- | --- |
| `13_batch_eye_optitrack_ingest_template.ipynb` | `data ingest updates/Natasha Batch Ingest Eye Optitrack.ipynb` plus eye/optitrack Natasha notebooks | Standardized batch ingest planning for eye/gaze + optitrack |
| `14_camera_sync_ingest_qc_template.ipynb` | `data ingest updates/Complete_Camera_Sync.ipynb`, `data ingest updates/Top_Camera_Sync_Clean.ipynb` | Camera sync QC and missing-artifact detection before population |
| `15_optitrack_redo_repopulation_template.ipynb` | `data ingest updates/RedoOptitrack.ipynb`, `Natasha/optitrack_*` | Identify and plan optitrack re-runs/repopulation |
| `16_timestamp_alignment_batch_qc_template.ipynb` | `Natasha/dot_sync.ipynb`, `Natasha/headcams_sync*.ipynb`, `Natasha/trials_timing.ipynb` | Batch timestamp/event alignment diagnostics |
| `20_routine_quick_job_cleanup_template.ipynb` | `20_manual_step_by_step_ingest_bpod_trial.ipynb` | Safe, structured routine job-cleanup template |

## Safety policy used for adaptations
- default mode is read-only (no insert/delete/populate execution)
- optional write paths are explicit and gated (`ALLOW_DB_WRITES=False`)
- no database modifications are performed by default notebook flow

## Complete source listing
- Full reviewed source inventory: `docs/TOBIASR_NOTEBOOK_CATALOG.md`
