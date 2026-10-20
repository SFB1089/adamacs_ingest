"""Verify the on-the-fly footprint reconstruction against a stored CellReg run.

Reconstructs spatial_footprints_corrected for one session from imaging.Segmentation.Mask
plus the alignment parameters in the result file, and compares against the footprints the
run actually wrote. Reads one session's footprint stack through h5py rather than mat73, so
the other ten (another ~24 GB as doubles) are never touched.
"""
import os, sys, json
import numpy as np, h5py, datajoint as dj

sys.path.insert(0, os.path.expanduser("~/adamacs_ingest"))
from adamacs.ingest import cellreg_footprints as fp

RESULT = ("/datajoint-data/data/leonk/analysis/cellreg/"
          "LE_ROS-1881_sess9FPXNI8U_Gratings_CR/Results/cellRegistered_20260129_075504.mat")
SCANS = ["scan9FPZGABI", "scan9FQ0MLVT", "scan9FQ1TN3R", "scan9FQ30DXR", "scan9FQ47DOQ",
         "scan9FQ5EL8J", "scan9FQ6L935", "scan9FQ7RT6T", "scan9FQ8YMPQ", "scan9FQA5Q0T",
         "scan9FQBCJ7R"]
PIDX, CID = 10, 2
MPP = 1.2                       # logFile: FOV 615.6 um / 513 px
REF_SESSION = 5                 # 1-based, from the log
COLUMN = int(sys.argv[1]) if len(sys.argv) > 1 else 0      # 0-based column to check
NCELLS = int(sys.argv[2]) if len(sys.argv) > 2 else 60     # cells to compare

S = "cell_registered_struct"
with h5py.File(RESULT, "r") as f:
    g = f[S]
    print("fields:", sorted(g.keys()), flush=True)
    dx_um = np.asarray(g["alignment_x_translations"]).ravel()
    dy_um = np.asarray(g["alignment_y_translations"]).ravel()
    rot = (np.asarray(g["alignment_rotations"]).ravel()
           if "alignment_rotations" in g else np.zeros_like(dx_um))
    padx = np.asarray(g["adjustment_x_zero_padding"]).ravel()
    pady = np.asarray(g["adjustment_y_zero_padding"]).ravel()
    print(f"translations x (um): {np.round(dx_um, 3).tolist()}")
    print(f"translations y (um): {np.round(dy_um, 3).tolist()}")
    print(f"rotations (deg):     {np.round(rot, 4).tolist()}")
    print(f"zero padding x/y:    {padx.tolist()} / {pady.tolist()}", flush=True)

    cen_ref = g["centroid_locations_corrected"][()]
    cen = np.asarray(f[cen_ref.ravel()[COLUMN]])
    print(f"centroid_locations_corrected[{COLUMN}] raw shape {cen.shape}", flush=True)
    cen = cen.T if cen.shape[0] == 2 else cen            # -> (n_cells, 2)

    fpr = g["spatial_footprints_corrected"][()]
    stack = f[fpr.ravel()[COLUMN]]
    print(f"spatial_footprints_corrected[{COLUMN}] shape {stack.shape} {stack.dtype}",
          flush=True)

    # MATLAB (n, y, x) is stored transposed by HDF5 -> (x, y, n)
    n_cells = max(stack.shape)
    ax_n = int(np.argmax(stack.shape))
    print(f"  interpreting axis {ax_n} as cells (n={n_cells})", flush=True)

    dj.config["loglevel"] = "ERROR"
    conn = dj.conn(); conn.query("SET SESSION transaction_read_only = 1")
    imaging = dj.create_virtual_module("imaging", "roselab_imaging",
                                       create_schema=False, create_tables=False)
    scan = SCANS[COLUMN]
    sess = conn.query("SELECT session_id FROM roselab_scan.scan WHERE scan_id=%s",
                      (scan,)).fetchone()[0]
    key = {"session_id": sess, "scan_id": scan, "paramset_idx": PIDX, "curation_id": CID}
    masks = (imaging.Segmentation.Mask & key).fetch(
        "mask", "mask_xpix", "mask_ypix", "mask_weights", order_by="mask", as_dict=True)
    avg = np.asarray((imaging.MotionCorrection.Summary & key).fetch1("average_image"))
    H, W = avg.shape[0] + 1, avg.shape[1] + 1
    print(f"{scan}: {len(masks)} masks from DB, frame {H}x{W}", flush=True)
    assert len(masks) == n_cells, f"mask count {len(masks)} != footprints {n_cells}"

    is_ref = (COLUMN + 1) == REF_SESSION
    rows, worst = [], None
    step = max(1, len(masks) // NCELLS)
    for k in range(0, len(masks), step):
        m = masks[k]
        mine = fp.aligned_footprint(
            m["mask_xpix"], m["mask_ypix"], m["mask_weights"], height=H, width=W,
            dx_um=dx_um[COLUMN], dy_um=dy_um[COLUMN],
            rotation_deg=rot[COLUMN] if rot.size > COLUMN else None,
            microns_per_pixel=MPP, is_reference=is_ref)
        sl = [slice(None)] * 3
        sl[ax_n] = k
        # HDF5 reverses MATLAB's dimension order, so a MATLAB (n, y, x) array arrives as
        # (x, y, n) and one cell's slice is indexed [x, y]. Transpose ALWAYS: for a square
        # FOV the shapes match either way, so a shape test silently passes the wrong
        # orientation (first run: mass 1.0 on both sides, correlation 0.000).
        theirs = np.asarray(stack[tuple(sl)], dtype=float).T
        num = float((mine * theirs).sum())
        den = float(np.sqrt((mine ** 2).sum() * (theirs ** 2).sum()))
        corr = num / den if den else float("nan")
        md = float(np.abs(mine - theirs).max())
        rows.append((k, corr, md, float(mine.sum()), float(theirs.sum())))
        if worst is None or corr < worst[1]:
            worst = (k, corr, md)

    cor = np.array([r[1] for r in rows])
    print(f"\ncompared {len(rows)} cells of {len(masks)} (every {step})")
    print(f"  correlation: min {np.nanmin(cor):.6f}  median {np.nanmedian(cor):.6f}  "
          f"mean {np.nanmean(cor):.6f}")
    print(f"  >0.9999: {int((cor > 0.9999).sum())}/{len(cor)}   "
          f">0.999: {int((cor > 0.999).sum())}/{len(cor)}   "
          f">0.99: {int((cor > 0.99).sum())}/{len(cor)}")
    print(f"  max abs pixel diff over all compared: {max(r[2] for r in rows):.3e}")
    print(f"  worst cell: index {worst[0]} corr {worst[1]:.6f} maxdiff {worst[2]:.3e}")
    print("  mass (mine vs theirs) for 5 cells:",
          [(round(r[3], 4), round(r[4], 4)) for r in rows[:5]])
    json.dump({"column": COLUMN, "scan": scan, "n_compared": len(rows),
               "corr_min": float(np.nanmin(cor)), "corr_median": float(np.nanmedian(cor)),
               "n_above_9999": int((cor > 0.9999).sum()),
               "max_abs_diff": max(r[2] for r in rows), "is_reference": bool(is_ref)},
              open(os.path.expanduser(f"~/fpverify_col{COLUMN}.json"), "w"), indent=2)
