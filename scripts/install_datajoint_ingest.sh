#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-datajoint_ingest}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found on PATH."
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Using existing conda environment: $ENV_NAME"
else
  echo "Creating conda environment: $ENV_NAME"
  conda create -n "$ENV_NAME" -y python=3.11 pip
fi

conda run -n "$ENV_NAME" python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"Environment Python must be 3.11, found {sys.version.split()[0]}. "
        "Recreate the conda environment with python=3.11."
    )
PY

conda install -n "$ENV_NAME" -y graphviz
conda run -n "$ENV_NAME" python -m pip install --upgrade pip wheel "setuptools<81"
conda run -n "$ENV_NAME" python -m pip uninstall -y adamacs-ingest >/dev/null 2>&1 || true
conda run -n "$ENV_NAME" python -m pip install -r "$REPO_ROOT/requirements_datajoint_new.txt"
conda run -n "$ENV_NAME" python -m pip install -e "$REPO_ROOT"

# Avoid forcing pywavesurfer's legacy transitive pins into the environment.
conda run -n "$ENV_NAME" python -m pip install --no-deps \
  "pywavesurfer @ git+https://github.com/SFB1089/PyWaveSurfer.git"

if ! conda run -n "$ENV_NAME" python -m pip install "scanimage-tiff-reader==1.4.1.4"; then
  echo "Falling back to conda-forge for scanimage-tiff-reader"
  conda install -n "$ENV_NAME" -y -c conda-forge scanimage-tiff-reader
fi

conda run -n "$ENV_NAME" python - <<'PY'
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
import datajoint as dj
import adamacs
import sys
print("DataJoint:", dj.__version__)
print("Python:", sys.version.split()[0])
print("adamacs import: OK")
PY

conda run -n "$ENV_NAME" python - <<'PY'
import subprocess
import sys

proc = subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    capture_output=True,
    text=True,
)

if proc.returncode == 0:
    print("pip check: OK")
    raise SystemExit(0)

lines = [line.strip() for line in (proc.stdout + "\n" + proc.stderr).splitlines() if line.strip()]
unexpected = [
    line
    for line in lines
    if not line.startswith("pywavesurfer 0.0.8 has requirement ")
]

if unexpected:
    print("Unexpected dependency conflicts detected:")
    for line in unexpected:
        print(line)
    raise SystemExit(1)

print("pip check: only expected pywavesurfer metadata conflicts detected (installed with --no-deps).")
PY

echo "Installation complete. Activate with: conda activate $ENV_NAME"
