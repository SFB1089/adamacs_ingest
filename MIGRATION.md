# Migration Guide (`adamacs` -> split repos)

This guide maps the old monorepo layout into the two split repositories.

## Repository boundary
- `adamacs_ingest`: ingest/pipeline runtime, DataJoint schemas, GUI, ingest notebooks, batch ingest templates.
- `adamacs_analysis`: analysis-only helpers, analysis schema utilities, and personal analysis notebooks.

## Analysis code moved out of ingest package
| Old path | New path |
| --- | --- |
| `adamacs/helpers/trace_helpers.py` | `adamacs_analysis/adamacs_analysis/helpers/trace_helpers.py` |
| `adamacs/helpers/arena_helpers.py` | `adamacs_analysis/adamacs_analysis/helpers/arena_helpers.py` |
| `adamacs/helpers/arena_render.py` | `adamacs_analysis/adamacs_analysis/helpers/arena_render.py` |
| `adamacs/schemas/analysis.py` | `adamacs_analysis/adamacs_analysis/schemas/analysis.py` |

## Ingest notebook renumbering
| Old notebook | New notebook |
| --- | --- |
| `notebooks/01_pipeline.ipynb` | `notebooks/01_pipeline_activation.ipynb` |
| `notebooks/05_DeepLabCut.ipynb` | `notebooks/02_deeplabcut_ingest.ipynb` |
| `notebooks/07_imaging_processing.ipynb` | `notebooks/03_imaging_processing_ingest.ipynb` |
| `notebooks/14_pupil_tracking_ingestion.ipynb` | `notebooks/04_pupil_tracking_ingestion.ipynb` |
| `notebooks/15_imaging denoising pipeline.ipynb` | `notebooks/05_imaging_denoising_pipeline.ipynb` |
| `notebooks/16_imaging cascade pipeline.ipynb` | `notebooks/06_imaging_cascade_pipeline.ipynb` |
| `notebooks/17_optitrack_insert.ipynb` | `notebooks/07_optitrack_insert.ipynb` |
| `notebooks/18_DISK_insert.ipynb` | `notebooks/08_disk_insert.ipynb` |
| `notebooks/19_eyecam_ingest.ipynb` | `notebooks/09_eyecam_ingest.ipynb` |
| `notebooks/23_batch_dlc_eye_ingestion.ipynb` | `notebooks/10_batch_dlc_eye_ingestion.ipynb` |
| `notebooks/25_Optitrack_RigidMouse_and_Gaze_Repopulation.ipynb` | `notebooks/11_optitrack_gaze_repopulation.ipynb` |
| `notebooks/27_slam_worldcam_minimal.ipynb` | `notebooks/12_slam_worldcam_minimal.ipynb` |

## Compatibility behavior
- `adamacs_ingest` keeps compatibility shims at legacy helper/schema import paths.
- Analysis-level behavior at those paths requires `adamacs_analysis` to be installed.
