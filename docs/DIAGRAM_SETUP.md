# DataJoint Diagram Setup

This project uses `dj.Diagram` for schema dependency rendering.

## Required components
- DataJoint: `0.14.8`
- `setuptools<81` (keeps `pkg_resources` available for DataJoint 0.14.x)
- Python packages: `networkx`, `pydot`, `graphviz` (Python wrapper)
- System/conda binary: Graphviz `dot` executable on `PATH`

## Install in `datajoint_ingest`
```bash
./scripts/install_datajoint_ingest.sh
```

## Smoke test
```bash
python - <<'PY'
import shutil
import datajoint as dj
from adamacs.notebook_runtime import bootstrap_ingest_notebook
import adamacs.schemas.subject as subject_schema

ctx = bootstrap_ingest_notebook(verbose=False)
print("DataJoint:", dj.__version__)
print("dot path:", shutil.which("dot"))
diagram = dj.Diagram(subject_schema.schema)
_ = diagram.make_dot()
print("Diagram dot generation: OK")
PY
```

## TLS handshake troubleshooting
- Symptom: `SSLV3_ALERT_HANDSHAKE_FAILURE` during `dj.conn()` or `adamacs.pipeline` import.
- Cause: in DataJoint `0.14.x`, `database.use_tls: null` can trigger a TLS attempt.
- Fix: set `"database.use_tls": false` in your local `dj_local_conf.json` for non-TLS DB servers.
- Reminder: never commit `dj_local_conf.json`.

## Known issue classes and current status
- Missing `dot` binary (`FileNotFoundError: "dot" not found in path`):
  resolved by installing Graphviz in the environment.
- Historical `pydot/networkx` compatibility issues in older DataJoint versions:
  covered by current stack (`datajoint==0.14.8`, `pydot>=4`, `networkx>=3.4`).

## Upstream references
- DataJoint release `v0.14.8`:
  https://github.com/datajoint/datajoint-python/releases/tag/v0.14.8
- Diagram compatibility issue (closed): https://github.com/datajoint/datajoint-python/issues/1175
- `pydot==3` related issue (closed): https://github.com/datajoint/datajoint-python/issues/1169
- Missing `dot` executable issue (closed): https://github.com/datajoint/datajoint-python/issues/924
