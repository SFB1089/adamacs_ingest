# Student Workflow

## 1) Use `adamacs_ingest` directly (no fork)
- Clone upstream `adamacs_ingest`.
- Configure `dj_local_conf.json` locally.
- Run ingest from GUI (`adamacs.gui.select_sessions`), renumbered ingest notebooks, or batch templates.
- Keep your local clone synchronized with upstream `main`.

## 2) Submit improvements upstream
- Create a local feature branch.
- Add targeted changes with tests.
- Open a PR against upstream `adamacs_ingest` `main`.

## 3) Fork `adamacs_analysis` for custom projects
- Fork `adamacs_analysis` into your own namespace.
- Keep your project-specific and personal analysis notebooks in your fork.
- Track upstream changes selectively.

## Contribution expectations
- Small PRs with clear scope.
- Keep ingest-facing code in `adamacs_ingest`.
- Keep exploratory and personal analysis notebooks in `adamacs_analysis`.
