# Split Audit

## Classification rules
- `ingest`: required for schema activation, ingest/population, ingest GUI workflow, and ingest-first operational notebooks
- `analysis`: personal analysis notebooks, analysis helpers, and analysis-only schema utilities
- `shared`: minimal compatibility boundary between repos

## File/domain classification
- `adamacs/ingest/*`: ingest
- `adamacs/schemas/{subject,surgery,equipment,behavior,mocap,pupil_tracking,virtual_markers_optitrack,denoising,disk}.py`: ingest
- `adamacs/pipeline.py`, `adamacs/paths.py`, `adamacs/utility.py`: ingest
- `adamacs/helpers/{adamacs_ingest.py,adamacs_ingest_v2.py,stack_helpers.py,s2p_helpers.py,dj_helpers.py,user_defaults_manager.py}`: ingest
- `adamacs/helpers/{trace_helpers.py,arena_helpers.py,arena_render.py}`: analysis (moved, shim left here)
- `adamacs/schemas/analysis.py`: analysis (moved, shim left here)
- `notebooks/01-12_*`: ingest notebooks (renumbered)
- `adamacs_analysis/notebooks/04-09_*`: personal analysis notebooks from original root

## Shared layer (explicit + minimal)
- `adamacs_analysis` depends on `adamacs_ingest` for schema/pipeline imports.
- `adamacs_ingest` exposes compatibility shims so legacy imports resolve when `adamacs_analysis` is installed.
