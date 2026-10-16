"""Compare a database-driven CellReg run against the lab's existing result for the same site.

Both runs index cells per session by position in the same Fall ordering, so a row of
cell_to_index_map is directly comparable between runs: a cluster is the tuple of per-session
indices (0 = absent). This script reports

  * per-session cell counts (must match exactly -- different counts mean different inputs)
  * clusters reproduced verbatim (identical tuple over all 11 columns)
  * pairwise agreement, the measure that survives a single column differing: for each session
    pair, of the cell-pairs Leon's run links, what fraction does this run link too
  * the size distribution of clusters, which is what drives downstream n

CellReg fits a probabilistic model, so exact equality is not the expectation -- the number to
read is pairwise agreement.
"""
import json, pathlib, sys

import numpy as np
import mat73


def index_map(path):
    st = mat73.loadmat(str(path), only_include=["cell_registered_struct/cell_to_index_map"])
    return np.asarray(st["cell_registered_struct"]["cell_to_index_map"]).astype(np.int64)


def pair_sets(m):
    """Every (session_i, session_j, cell_i, cell_j) link the map asserts, as a set per pair."""
    out = {}
    n = m.shape[1]
    for i in range(n):
        for j in range(i + 1, n):
            both = (m[:, i] > 0) & (m[:, j] > 0)
            out[(i, j)] = set(zip(m[both, i].tolist(), m[both, j].tolist()))
    return out


def main(mine_path, theirs_path, out_json):
    mine, theirs = index_map(mine_path), index_map(theirs_path)
    rep = {"mine": str(mine_path), "theirs": str(theirs_path),
           "shape_mine": list(mine.shape), "shape_theirs": list(theirs.shape)}

    assert mine.shape[1] == theirs.shape[1], "different number of sessions"
    rep["n_cells_mine"] = int(mine.shape[0])
    rep["n_cells_theirs"] = int(theirs.shape[0])

    cm, ct = (mine > 0).sum(0), (theirs > 0).sum(0)
    rep["per_session_counts_mine"] = cm.tolist()
    rep["per_session_counts_theirs"] = ct.tolist()
    rep["per_session_counts_match"] = bool((cm == ct).all())

    # verbatim clusters
    sm = {tuple(r) for r in mine.tolist()}
    st = {tuple(r) for r in theirs.tolist()}
    rep["clusters_identical"] = len(sm & st)
    rep["frac_theirs_reproduced_verbatim"] = len(sm & st) / len(st) if st else None
    rep["frac_mine_also_in_theirs"] = len(sm & st) / len(sm) if sm else None

    # pairwise agreement
    pm, pt = pair_sets(mine), pair_sets(theirs)
    inter = sum(len(pm[k] & pt[k]) for k in pt)
    tot_t = sum(len(pt[k]) for k in pt)
    tot_m = sum(len(pm[k]) for k in pm)
    rep["links_theirs"] = tot_t
    rep["links_mine"] = tot_m
    rep["links_shared"] = inter
    rep["pairwise_recall"] = inter / tot_t if tot_t else None      # of Leon's links, how many I make
    rep["pairwise_precision"] = inter / tot_m if tot_m else None   # of my links, how many Leon makes
    rep["pairwise_jaccard"] = inter / (tot_m + tot_t - inter) if (tot_m + tot_t - inter) else None

    per_pair = {f"{i+1}-{j+1}": (len(pm[(i, j)] & pt[(i, j)]) / len(pt[(i, j)])
                                 if pt[(i, j)] else None) for (i, j) in sorted(pt)}
    rep["per_pair_recall"] = per_pair
    vals = [v for v in per_pair.values() if v is not None]
    rep["per_pair_recall_min"] = min(vals) if vals else None
    rep["per_pair_recall_median"] = float(np.median(vals)) if vals else None

    # cluster sizes
    for nm, m in (("mine", mine), ("theirs", theirs)):
        sizes = (m > 0).sum(1)
        rep[f"size_hist_{nm}"] = {str(k): int((sizes == k).sum())
                                  for k in range(1, m.shape[1] + 1)}
        rep[f"n_in_all_sessions_{nm}"] = int((sizes == m.shape[1]).sum())

    pathlib.Path(out_json).write_text(json.dumps(rep, indent=2))
    for k, v in rep.items():
        if not k.startswith(("per_pair_recall",)) or k != "per_pair_recall":
            print(f"{k}: {v}")
    return rep


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "cellreg_comparison.json")
