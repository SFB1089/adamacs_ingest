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
print("DataJoint:", dj.__version__)
print("adamacs import: OK")
PY

echo "Installation complete. Activate with: conda activate $ENV_NAME"
