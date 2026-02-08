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

### Recommended full install (`datajoint_ingest`)
This is the path for real ingest and notebook work.

1. Clone the repository:
```bash
git clone https://github.com/SFB1089/adamacs_ingest.git
cd adamacs_ingest
```

2. Create a fresh environment with a recent Python:
```bash
conda create -n datajoint_ingest python=3.10 -y
conda activate datajoint_ingest
python -m pip install --upgrade pip setuptools wheel
```

3. Install DataJoint pre-2.0 (pinned):
```bash
python -m pip install "datajoint==0.14.8" "PyMySQL>=1.1.0"
```

4. Install the ingest dependency stack (includes SFB1089 element forks):
```bash
python -m pip install -r requirements_datajoint_new.txt
```

5. Install `adamacs_ingest` in editable mode:
```bash
python -m pip install -e .
```

6. Sanity-check core imports:
```bash
python - <<'PY'
import datajoint as dj
import adamacs
print("DataJoint:", dj.__version__)
print("adamacs import: OK")
PY
```

### Lightweight install (code-only development)
Use this only when you are not running full DataJoint ingest workflows.

```bash
conda create -n adamacs_ingest python=3.10 -y
conda activate adamacs_ingest
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Environment-file install (alternative)
If you prefer conda YAML bootstrapping:

```bash
mamba env create -f environment_datajoint_new.yml
conda activate datajoint_new
python -m pip install -e .
python -m pip install "datajoint==0.14.8"
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
