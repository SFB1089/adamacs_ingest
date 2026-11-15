"""Run CASCADE on exported dF/F0 (the 'predict' stage of verify_dff0_cascade.py).

Runs on BUGPU in the worker's env and checkout, on CPU only so the GPUs used by the
production worker are not touched:

    CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/Cascade/bin/python cascade_predict_npz.py <dir>

For every <name>.npz without a <name>.pred.npz (export deletes a stale one), calls cascade.predict(model_name, dff,
verbosity=0) exactly as imaging.Activity.make does (default threshold and padding).
"""
import glob
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
CASCADE = os.path.expanduser("~/Cascade")
d = os.path.abspath(sys.argv[1])
sys.path.append(CASCADE)
os.chdir(CASCADE)

import numpy as np  # noqa: E402
from cascade2p import cascade  # noqa: E402

for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
    if f.endswith(".pred.npz") or os.path.exists(f[:-4] + ".pred.npz"):
        continue
    z = np.load(f, allow_pickle=True)
    pred = cascade.predict(str(z["model_name"]), z["dff"], verbosity=0)
    # carry the input fingerprint so compare can reject a prediction made for a different export
    np.savez(f[:-4] + ".pred.npz", pred=pred,
             fingerprint=str(z["fingerprint"]) if "fingerprint" in z.files else "")
    print("predicted", os.path.basename(f), pred.shape, flush=True)
