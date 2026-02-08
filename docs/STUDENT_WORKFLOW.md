# Student Workflow (Ingest First)

This workflow keeps ingest operationally centralized while still allowing students to contribute upstream.

## 1) Daily ingest: use upstream `adamacs_ingest` directly
1. Clone upstream once:
```bash
git clone https://github.com/SFB1089/adamacs_ingest.git
cd adamacs_ingest
```

2. Configure your local database credentials in `dj_local_conf.json` (never commit this file).

3. Keep your local checkout current:
```bash
git checkout main
git pull --ff-only origin main
```

4. Run routine ingest from:
- GUI entrypoint: `adamacs.gui.select_sessions`
- Center notebook: `notebooks/00_ingest_gui_workflow_adamacs_ingest_v2.ipynb`
- Batch templates: `examples/batch_ingest/`

## 2) When ingest behavior needs a change: contribute via PR
1. Branch from updated `main`:
```bash
git checkout main
git pull --ff-only origin main
git checkout -b <your-feature-branch>
```

2. Make a focused change and add/update tests.

3. Run local checks:
```bash
python -m pytest -q
python -m ruff check --select E9,F63,F7,F82 adamacs tests examples
```

4. Push branch and open PR to `SFB1089/adamacs_ingest:main`.

## 3) For personal or project-specific analysis: fork `adamacs_analysis`
1. Fork `SFB1089/adamacs_analysis` into your namespace.
2. Keep custom analysis notebooks and exploratory code in that fork.
3. Submit reusable analysis improvements upstream to `adamacs_analysis`.

## 4) Scope boundaries
- Ingest and pipeline operations belong in `adamacs_ingest`.
- Student analysis notebooks and analysis-only helpers belong in `adamacs_analysis`.
- If a change touches both, submit separate PRs per repository.
