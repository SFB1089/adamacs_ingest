# adamacs_ingest

`adamacs_ingest` is the ingest-first ADAMACS repository.

It keeps DataJoint schemas, ingest/population logic, and the ingest GUI in one place so students can run routine ingest without forking this repo.

## What lives here
- Core `adamacs` package: schemas, pipeline activation, ingest modules
- Ingest GUI entrypoint: `adamacs.gui.select_sessions`
- Renumbered ingest notebooks in `notebooks/`
- Scripted notebook exports in `notebooks/py_scripts/`
- Batch ingest examples for DLC, OptiTrack, eye tracking/gaze, Ca2+ imaging, Bpod sync, and auxiliary streams
- User configuration templates (`user_configs/`, `user_data/`)

## Installation
1. Create and activate your Python environment.
2. Install dependencies and package:

```bash
pip install -r requirements.txt
pip install -e .
```

Alternative `datajoint_new` environment dependencies (includes SFB1089 element forks):

```bash
pip install -r requirements_datajoint_new.txt
```

3. Create `dj_local_conf.json` from `Example_dj_local_conf.json` and set your credentials/paths.

## Quick start (GUI)

```python
from adamacs.gui import select_sessions

select_sessions(["TR_WEZ-8701_2025-01-20_sessABC_scanXYZ"])
```

Center-stage ingest workflow notebook:
- `notebooks/00_ingest_gui_workflow_adamacs_ingest_v2.ipynb`

## Ingest notebook index
See `docs/NOTEBOOK_INDEX.md`.

## Batch ingest examples
See `examples/batch_ingest/` and `docs/BATCH_INGEST_EXAMPLES.md`.

## Student workflow (no fork required)
1. Clone `adamacs_ingest` and keep local branch up to date with `main`.
2. Use GUI, ingest notebooks, or batch templates for daily ingest work.
3. If you improve ingest behavior, open a feature branch in your clone and submit a PR to upstream `main`.

Detailed flow: `docs/STUDENT_WORKFLOW.md`.

## Split and migration docs
- Split audit: `docs/SPLIT_AUDIT.md`
- Path migration: `MIGRATION.md`

## Contributing
- Keep ingest behavior backward compatible unless change is documented.
- Add tests for new utility/refactor logic.
- Keep analysis-only code in `adamacs_analysis`.
