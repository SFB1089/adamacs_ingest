# Split Audit

This document records the minimal boundary used for the `adamacs` split.

## Classification rules
- `ingest`: required for schema activation, ingestion, population, and GUI-first operations.
- `analysis`: plotting/query helpers, analysis-only schemas, and personal analysis notebooks.
- `shared`: explicit compatibility edges only; no duplicate pipeline implementations.

## Kept in `adamacs_ingest`
- `adamacs/ingest/*`
- `adamacs/pipeline.py`, `adamacs/paths.py`, `adamacs/utility.py`
- Ingest-facing schemas in `adamacs/schemas/*` (`subject`, `surgery`, `equipment`, `behavior`, `mocap`, `pupil_tracking`, `virtual_markers_optitrack`, `disk`, `denoising`)
- Ingest helpers: `adamacs/helpers/adamacs_ingest.py`, `adamacs/helpers/adamacs_ingest_v2.py`, `adamacs/helpers/stack_helpers.py`, `adamacs/helpers/s2p_helpers.py`, `adamacs/helpers/dj_helpers.py`, `adamacs/helpers/user_defaults_manager.py`
- Renumbered ingest notebooks `notebooks/00-12_*`
- Batch ingest templates in `examples/batch_ingest/`

## Moved to `adamacs_analysis`
- `adamacs/helpers/trace_helpers.py`
- `adamacs/helpers/arena_helpers.py`
- `adamacs/helpers/arena_render.py`
- `adamacs/schemas/analysis.py`
- Personal notebooks from original root, renumbered to `notebooks/04-09_*`

## Explicit shared boundary
- `adamacs_analysis` imports ingest/pipeline truth from `adamacs_ingest`.
- Compatibility shims remain in `adamacs_ingest` so legacy import paths still resolve when `adamacs_analysis` is installed.
