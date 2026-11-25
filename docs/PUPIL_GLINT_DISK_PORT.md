# Pupil image rim fit, corneal glint and DISK, ported from adamacs

These modules came over from SFB1089/adamacs, where every stored row of these tables was computed:

| ingest file | from adamacs | what it is |
|---|---|---|
| `adamacs/helpers/pupil_image_fit.py` | `fix/rimfit-early-radius` 2bac5ac | per-frame rim fit, acceptance, 3D eye model |
| `adamacs/schemas/pupil_tracking.py` | `fix/rimfit-early-radius` 2bac5ac | `PupilEllipseFitting` (+ `ImageFit`), image-fit parameters |
| `adamacs/helpers/glint_detect.py` | `fix/glint-fallback-gate` e6d15c5 | glint tracker |
| `adamacs/schemas/pupil_glint.py` | `fix/glint-fallback-gate` e6d15c5 | `GlintDetectionParameter`, `CorrectedGlint` |
| `adamacs/schemas/disk.py`, `disk_worker.py`, `helpers/eye_geometry.py` | main 61f1bab | DISK imputation: free-moving, head-fixed and topcam chains |

adamacs_ingest's own `pupil_tracking.py` and `disk.py` were the adamacs files of 14.09.2026 (33c01a3); every change
ingest made to them since (EyeSideNote, stored_eye_prefix, free-moving parameter keying) is also in adamacs, so nothing
ingest-only was lost.

## Changes against the adamacs code (no numerical change)

- `adamacs/pipeline.py` imports `pupil_glint` and activates `disk` (roselab_disk + the pupil-side tables in
  roselab_pupil_tracking), with the pipeline as linking module. In adamacs, `disk` was bound lazily by
  `PupilEllipseFitting` or by each script.
- Rim fit: which code a parameter row runs is decided in one place, `pupil_image_fit.fit_settings(cfg)`:
  `image_rim_v3` (set 10), `image_rim_v4` (set 11), `image_rim_v4.1` (set 12). An unknown `code_version` raises.
- Glint: `glint_detect.track_video(version=...)` runs glint-1.1 (set 1), glint-1.2 (set 2) or glint-1.3 (set 3);
  the adamacs code only ran its own version, so sets 1 and 2 could no longer be reproduced. `CorrectedGlint.make`
  stores the parameter set's code version. `populate` still computes only the current version (key_source).

## Tests

`tests/pupil/` (read-only; `readonly_db.py` blocks every write path and runs the session READ ONLY):

- `check_tables.py`: every class maps to an existing table (before anything is imported).
- `check_headings.py`: definitions vs live headings. Only `TorsionCalibManual` differs (definition `float`, table
  `longblob`), unchanged from adamacs and from ingest main.
- `golden.py`: `make()` of the port and of the adamacs source, compared with the stored rows and with each other.
- `golden_eye_model.py`: the eye model recomputed from stored ellipses.
- `test_pupil_port.py`: pytest wrapper (pure tests always; database tests when the DataJoint config exists; video
  cases with `ADAMACS_GOLDEN_VIDEO=1`).

Bitwise equality with stored rows needs the environment that wrote them: the rim-fit and glint populates ran in the
`Cascade` env (numpy 1.23.3, OpenCV 4.11.0), the fissure second passes in `v1audit` (numpy 1.24.4). Least-squares
results (eye model) differ in the last bits (≤ 1e-5 px) between numpy builds; per-frame fits are exact.
