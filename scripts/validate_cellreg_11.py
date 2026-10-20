"""Validation run: reproduce Leon's January 2026 registration from the database alone.

Target: LE_ROS-1881_sess9FPXNI8U_Gratings_CR / cellRegistered_20260129_075504.mat, 4107 cells
over the 11 gratings scans of site sess9FPXNI8U.

Every input comes from roselab_imaging.Segmentation.Mask -- no suite2p file is read. The
settings are transcribed from that run's own logFile_20260129_075612.txt, with the log's prose
mapped onto the literals the driver's strcmp actually accepts.

One input is NOT recorded anywhere in Leon's run: which curation each column came from. The
export step copied a filesystem path, and the log carries scan ids only. This run declares
curation 2 -- the only curation at this site with a populated roi_group, and the one whose
per-scan mask counts match the run's cell_to_index_map maxima. If the registration reproduces,
that is evidence the choice was right; if it does not, the curation is the first suspect.

Read-only on the database. Writes only under the service account's home.
"""
import datetime, json, os, pathlib, shutil, subprocess, sys

import numpy as np
import datajoint as dj

sys.path.insert(0, os.path.expanduser("~/adamacs_ingest"))
from adamacs.ingest import cellreg_export as cx

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/cellreg_validation"))

# the 11 gratings scans, in the order Leon's log lists them
SCANS = ["scan9FPZGABI", "scan9FQ0MLVT", "scan9FQ1TN3R", "scan9FQ30DXR", "scan9FQ47DOQ",
         "scan9FQ5EL8J", "scan9FQ6L935", "scan9FQ7RT6T", "scan9FQ8YMPQ", "scan9FQA5Q0T",
         "scan9FQBCJ7R"]
PIDX, CID = 10, 2

# transcribed from logFile_20260129_075612.txt. microns_per_pixel is not printed directly:
# the log gives FOV 615.6 um over 513 px, so 615.6/513 = 1.2 exactly (not the 1.2024 that
# 615.6/512 would give -- the driver pads to Lx+1 before computing the FOV).
SETTINGS = {
    "registration_approach": cx.from_log_value("registration_approach", "Probabilistic modeling"),
    "model_type": cx.from_log_value("model_type", "Spatial correlations"),
    "initial_registration_type": "Spatial correlation",
    "alignment_type": "Translations and Rotations",
    "maximal_distance": 12.0,
    "p_same_threshold": 0.3,
    "reference_session_index": 5,
    "microns_per_pixel": 1.2,
    "use_parallel_processing": True,
    "memory_efficient_run": True,
}

dj.config["loglevel"] = "ERROR"
conn = dj.conn()
conn.query("SET SESSION transaction_read_only = 1")
assert conn.query("SELECT @@transaction_read_only").fetchone()[0] == 1
imaging = dj.create_virtual_module("imaging", "roselab_imaging",
                                   create_schema=False, create_tables=False)

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
print(f"scratch: {OUT}", flush=True)

manifest = []
for i, scan in enumerate(SCANS):
    sess = conn.query("SELECT session_id FROM roselab_scan.scan WHERE scan_id=%s",
                      (scan,)).fetchone()[0]
    key = {"session_id": sess, "scan_id": scan, "paramset_idx": PIDX, "curation_id": CID}
    masks = (imaging.Segmentation.Mask & key).fetch(
        "mask", "mask_xpix", "mask_ypix", "mask_weights", order_by="mask", as_dict=True)
    avg = np.asarray((imaging.MotionCorrection.Summary & key).fetch1("average_image"))
    h, w = avg.shape
    info = cx.write_fall_mat(OUT / cx.fall_name(i, scan), masks, h, w, mean_img=avg)
    info.update(column_index=i, scan_id=scan, session_id=sess,
                paramset_idx=PIDX, curation_id=CID)
    manifest.append(info)
    print(f"  col {i:2d}: {scan} -> {info['file_name']} "
          f"({info['n_masks']} masks, {info['file_bytes']/1e6:.1f} MB)", flush=True)

print(f"  total ROIs exported: {sum(m['n_masks'] for m in manifest)}", flush=True)

settings = cx.write_settings_mat(OUT / "settings.mat", SETTINGS)
print(f"  settings.mat ({settings['settings_sha256'][:12]})", flush=True)
(OUT / "manifest.json").write_text(json.dumps(
    {"scans": manifest, "settings": {k: (v.item() if hasattr(v, "item") else v)
                                     for k, v in settings.items()}},
    indent=2, default=str))

matlab = os.path.expanduser("~/MATLAB/bin/matlab")
drv = os.path.expanduser("~/adamacs_ingest/adamacs/matlab")
root = os.path.expanduser("~/CellReg")
cmd = [matlab, "-batch", f"addpath('{drv}'); cellreg_run('{OUT}', '{root}')"]
print(f"\nrunning: {' '.join(cmd)}", flush=True)
t0 = datetime.datetime.now()
r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(OUT))
dt = (datetime.datetime.now() - t0).total_seconds()
print(f"exit={r.returncode} after {dt/60:.1f} min", flush=True)
tail = "\n".join(l for l in r.stdout.splitlines() if "%[" not in l)[-4000:]
print("--- stdout tail ---"); print(tail)
if r.returncode != 0:
    print("--- stderr tail ---"); print(r.stderr[-4000:])

res = OUT / "Results"
for m in sorted(res.glob("cellRegistered_*.mat")) if res.is_dir() else []:
    print(f"  {m.name}  {m.stat().st_size/1e6:.1f} MB")
for l in sorted(res.glob("logFile_*.txt")) if res.is_dir() else []:
    print(f"\n=== {l.name} ===\n{open(l).read()}")
if res.is_dir() and (res / "log.txt").exists():
    print(f"\n=== log.txt ===\n{(res / 'log.txt').read_text()}")
