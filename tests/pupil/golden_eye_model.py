"""Golden test (3): the 3D eye model recomputed from stored ellipses, port vs stored and port vs source.

For rows of sets 10, 11 and 12 across mice: pupil_image_fit.recompute_from_stored on the stored kept ellipses with
the stored prior (fissure_px / fissure_source / fissure_n_sessions, fissure_session_*) and the spatial gate off
(the stored ellipses are already gated), as the fissure second passes did (v1-drift-audit t165 / t166 / t30).
Compares theta, phi, scale, cam_center, eye_center and the ImageFit eye-model fields with the stored row, bitwise
(MySQL `float` columns at the precision the database returns);
and the port's result with the same call into the source helper (SFB1089/adamacs fix/rimfit-early-radius).

  python tests/pupil/golden_eye_model.py --tsv FILE

The last bits of a least-squares fit depend on the numpy / scipy / BLAS build: run it in the environment that wrote the
rows (the v1-drift-audit fissure passes ran in the `v1audit` env, numpy 1.24.4) to expect bitwise equality.
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import readonly_db  # noqa: E402
from golden import PORT, SOURCE, as_stored, same  # noqa: E402

ROWS = [  # (scan_id, parameter_id): mice ROS-1841, 1881, 1907, 1912, 1985, 2209, 2210, 2328, 2356, 2359; sets 10-12
    ("scan9FXEU7TJ", 10), ("scan9FXEU7TJ", 11), ("scan9FXEU7TJ", 12), ("scan9FPZHKKB", 12), ("scan9FXG08PJ", 12),
    ("scan9G2N0274", 12), ("scan9G2N0274", 11), ("scan9FQ6MJO2", 12), ("scan9G2L66VU", 12), ("scan9FXG08PJ", 10),
]
EYE_FIELDS = ("e3d_cx", "e3d_cy", "e3d_r", "theta_offset", "phi_offset", "n_frames_eye_model", "fissure_px",
              "fissure_source", "fissure_n_sessions", "e3d_objective", "e3d_objective_single", "e3d_objective_far",
              "e3d_n_starts", "frac_accepted", "n_nice_frames")


def helper(root):
    spec = importlib.util.spec_from_file_location(f"pif_{abs(hash(str(root)))}", Path(root) / "adamacs/helpers/pupil_image_fit.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main(tsv):
    P = readonly_db.load_pipeline(PORT)
    pt = P.pupil_tracking
    pif_port, pif_src = helper(PORT), helper(SOURCE["rim"])
    lines = ["scan_id\tparameter_id\tcomparison\tfield\tbitwise_equal\tmax_abs_diff"]
    for scan, pid in ROWS:
        k = (pt.PupilEllipseFitting.ImageFit & {"scan_id": scan, "parameter_id": pid}).fetch1("KEY")
        m = (pt.PupilEllipseFitting & k).fetch1()
        r = (pt.PupilEllipseFitting.ImageFit & k).fetch1()
        cfg = (pt.PupilEllipseImageFitParameter & {"parameter_id": pid}).fetch1()
        thr = float((pt.PupilEllipseParameter & {"parameter_id": pid}).fetch1("ellipticity_thres"))
        cfg_ng = {**cfg, "gate_min_px": float("inf")}
        L_s = r["fissure_session_px"] if r["fissure_session_source"] in ("per_frame", "static") else None
        args = (m["ellipse_dict"], float(r["r0"]), cfg_ng, thr, r["fissure_px"], r["fissure_source"])
        kw = dict(n_sessions=r["fissure_n_sessions"], L_session=L_s, L_session_source=r["fissure_session_source"])
        a, b = pif_port.recompute_from_stored(*args, **kw), pif_src.recompute_from_stored(*args, **kw)
        for f in ("theta", "phi", "scale", "cam_center", "eye_center", "diameter"):
            for comp, ref in (("port_vs_stored", m[f]), ("port_vs_source", b[f])):
                eq, d = same(a[f], ref)
                lines.append(f"{scan}\t{pid}\t{comp}\t{f}\t{eq}\t{'' if d is None else repr(d)}")
        types = {n: at.type for n, at in pt.PupilEllipseFitting.ImageFit.heading.attributes.items()}
        for f in EYE_FIELDS:
            for comp, ref in (("port_vs_stored", r[f]), ("port_vs_source", b["imagefit"].get(f))):
                v = a["imagefit"].get(f)
                eq, d = same(as_stored(v, types.get(f)) if comp == "port_vs_stored" else v, ref)
                lines.append(f"{scan}\t{pid}\t{comp}\t{f}\t{eq}\t{'' if d is None else repr(d)}")
        print(scan, pid, "done", flush=True)
    Path(tsv).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=str(Path.home() / "scratch_ingest_port" / "golden" / "eye_model.tsv"))
    main(ap.parse_args().tsv)
