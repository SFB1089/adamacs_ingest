#!/usr/bin/env bash
# Run the CellReg worker on ibehavegpu1, under the service account.
#
# MATLAB lives only here: /home/backup_user/MATLAB (R2023a Update 8, licence 40976315),
# with the pinned toolbox at /home/backup_user/CellReg
# (zivlab/CellReg @ 70668513c665e1cb0de218c33264b8736f2976eb). It must not run on tatchu3 --
# that host is the database server and the lab has twice exhausted its RAM with MATLAB
# (2025-02-20, 2025-08-26). The worker refuses tatchu3 by hostname; this script does not
# pass --force-host.
#
# DEFAULT IS A DRY RUN. It attaches to the schema read-only (create_schema=False,
# create_tables=False), reports what it would do, and writes nothing. Pass --execute to act;
# that writes to the database and to the shared data folder and needs Tobias's go-ahead.
#
#   ./scripts/cellreg_worker_ibehavegpu.sh            # dry run, loops
#   ./scripts/cellreg_worker_ibehavegpu.sh --once     # dry run, one pass
#   ./scripts/cellreg_worker_ibehavegpu.sh --execute  # writes -- authorised only
#
# The worker never touches Segmentation.Mask.roi_group. Its three stages are
# CellRegExport.populate, the MATLAB driver, and CellRegResult.populate; writing roi_group
# is RoiGroupProjection's job and nothing here calls it.
set -euo pipefail

REPO="${CELLREG_REPO:-$HOME/adamacs_ingest}"
MATLAB="${CELLREG_MATLAB:-$HOME/MATLAB/bin/matlab}"
CELLREG_ROOT="${CELLREG_ROOT:-$HOME/CellReg}"
CONDA_ENV="${CELLREG_CONDA_ENV:-cellreg}"

[ -x "$MATLAB" ] || { echo "no MATLAB at $MATLAB" >&2; exit 1; }
[ -d "$CELLREG_ROOT" ] || { echo "no CellReg toolbox at $CELLREG_ROOT" >&2; exit 1; }
[ -f "$REPO/dj_local_conf.json" ] || {
  echo "no dj_local_conf.json in $REPO -- datajoint would prompt for credentials" >&2
  exit 1; }

# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$REPO"   # datajoint reads dj_local_conf.json from the working directory
exec python -u -m adamacs.workers.cellreg_worker \
  --matlab "$MATLAB" --cellreg-root "$CELLREG_ROOT" "$@"
