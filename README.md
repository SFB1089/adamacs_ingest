# adamacs_ingest

`adamacs_ingest` is the ingest-first ADAMACS repository.

Use this repo directly for day-to-day ingest and pipeline population. The ingest GUI remains the central user entrypoint, and ingest logic stays upstream in this repository (no routine fork required for students).

## Scope
- DataJoint pipeline activation and schemas (`adamacs/pipeline.py`, `adamacs/schemas/*`)
- Ingest and population logic (`adamacs/ingest/*`)
- Ingest GUI (`adamacs.gui.select_sessions`)
- Ingest notebooks and batch templates
- Shared lab defaults (`user_configs/`, `user_data/`)

## Repository map
- `adamacs/`: ingest package code
- `notebooks/`: renumbered ingest notebooks, including center-stage GUI workflow notebook
- `examples/batch_ingest/`: batch ingest templates by modality
- `docs/`: workflow, notebook index, split audit, batch example reference

## Installation

### One-stop install (recommended)
Run from repository root:

```bash
./scripts/install_datajoint_ingest.sh
```

Optional custom environment name:

```bash
./scripts/install_datajoint_ingest.sh my_ingest_env
```

The installer creates/updates a Python 3.11 conda env, installs Graphviz, installs the pinned DataJoint pre-2.0 stack (`datajoint==0.14.8`), installs this package in editable mode, and applies compatibility handling for `pywavesurfer` and `scanimage-tiff-reader`.

### Manual install (same stack as script)
```bash
conda create -n datajoint_ingest python=3.11 -y
conda install -n datajoint_ingest -y graphviz
conda run -n datajoint_ingest python -m pip install --upgrade pip wheel "setuptools<81"
conda run -n datajoint_ingest python -m pip install -r requirements_datajoint_new.txt
conda run -n datajoint_ingest python -m pip install -e .
conda run -n datajoint_ingest python -m pip install --no-deps "pywavesurfer @ git+https://github.com/SFB1089/PyWaveSurfer.git"
conda run -n datajoint_ingest python -m pip install scanimage-tiff-reader==1.4.1.4
```

If `scanimage-tiff-reader` wheel build fails:
```bash
conda install -n datajoint_ingest -y -c conda-forge scanimage-tiff-reader
```

## DataJoint local configuration

1. Create a local config file from the template:
```bash
cp Example_dj_local_conf.json dj_local_conf.json
```

2. Edit `dj_local_conf.json` with your local database credentials and paths.

3. Confirm the file is not tracked:
```bash
git ls-files | grep -Ei '(^|/)dj_local_conf.*\.(json|joson)$|(^|/)dj_local_.*\.(json|joson)$'
```

This repository ignores `dj_local_conf.json` and typo variants like `dj_local_conf.joson` by default.

## Run ingest workflows

### GUI entrypoint (primary workflow)
```python
from adamacs.gui import select_sessions
select_sessions(["TR_WEZ-8701_2025-01-20_sessABC_scanXYZ"])
```

Center-stage notebook:
- `notebooks/00_ingest_gui_workflow_adamacs_ingest_v2.ipynb`

### Ingest notebooks
- Notebook index: `docs/NOTEBOOK_INDEX.md`
- These notebooks are ingest-facing and can write/populate tables.
- Includes adapted templates from original `tobiasr` ingest workflows and original notebook 20 cleanup intent.

### Batch ingest templates
- Templates: `examples/batch_ingest/`
- Usage guide: `docs/BATCH_INGEST_EXAMPLES.md`

## Student-first workflow
- Use upstream `adamacs_ingest` directly for routine ingest.
- Keep your local branch synced with upstream `main`.
- Open focused PRs upstream when ingest logic needs improvement.

Detailed steps: `docs/STUDENT_WORKFLOW.md`.

## `adamacs_analysis` class generation utility
If you also install `adamacs_analysis` in the same environment, you can scaffold missing DataJoint classes:

```bash
python -m pip install -e ../adamacs_analysis
adamacs-analysis-generate \
  --schema-name adamacs_analysis \
  --output generated/my_analysis_schema.py \
  --class TrialLevelMetrics:Computed \
  --class SubjectSummary:Manual
```

## Local checks and CI parity
```bash
python -m pip install pytest ruff
python -m ruff check --select E9,F63,F7,F82 adamacs tests examples
python -m pytest -q
```

CI runs lint and pytest on Python 3.10 and 3.11.

## Additional docs
- Split audit: `docs/SPLIT_AUDIT.md`
- Path migration: `MIGRATION.md`
- Tobiasr ingest adaptation map: `docs/TOBIASR_INGEST_ADAPTATION.md`
- Tobiasr full notebook catalog: `docs/TOBIASR_NOTEBOOK_CATALOG.md`
- Diagram setup and troubleshooting: `docs/DIAGRAM_SETUP.md`
