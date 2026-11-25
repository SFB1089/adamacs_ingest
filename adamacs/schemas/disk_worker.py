"""DISK-side worker for :meth:`adamacs.schemas.disk.DLCImputation.make`.

Why this file exists as a separate process
------------------------------------------
The two conda environments on this host are mutually exclusive:

* ``datajoint`` has adamacs and the ``element_*`` tree, but no ``DISK``.
* ``disk`` has ``DISK`` and a CUDA 12 torch, but cannot import ``adamacs.pipeline``
  (no ``element_session``).

Neither can be made to do both without disturbing an environment the rest of the lab
depends on. So ``make()`` does all the DataJoint work in the ``datajoint`` environment,
writes a job directory, and invokes this script in the ``disk`` environment. This file
imports **no adamacs code and touches no database** -- it reads a CSV and a JSON spec,
runs DISK, and writes a CSV and a JSON back.

Contract
--------
Input, in ``<jobdir>``::

    input.csv   DeepLabCut-format CSV, already masked: every sample the caller wants
                filled is blank, everything else carries its observed value
    job.json    {length, stride, fps, likelihood_threshold, checkpoint, model_name,
                 batch_size}

Output, in ``<jobdir>``::

    imputed.csv DISK's output, same layout and column order as input.csv
    gaps.json   [{start, length, uncertainty}, ...] -- one entry per segment DISK
                attempted, in full-recording frame numbers
    worker.log  DISK's own logging

Usage::

    python disk_worker.py <jobdir>

The per-gap uncertainty is the reason for the wrapper below. DISK computes a sigma for
every filled coordinate (the network is trained with ``mu_sigma``), then collapses it to
one scalar per imputed segment and uses it only for plotting -- it is never written
alongside the positions it belongs to. ``ImputeDataset.update_dataset`` is wrapped to
record it as imputation proceeds. The wrapper only observes; it delegates to the original
method unchanged.
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("disk_worker")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    return logger


class UncertaintyRecorder:
    """Capture DISK's per-segment uncertainty alongside the position it belongs to.

    Positions are read exactly the way DISK reads them, including its own
    scalar-to-nested normalisation: ``possible_indices`` is not a flat table of three
    columns -- each entry can itself be a sequence -- so mirroring the library rather than
    guessing at the layout is what keeps this correct.

    ``i_pos`` IS NOT AN ABSOLUTE FRAME NUMBER
    -----------------------------------------
    This is the bug this class carried until 2026-09-22, and it reached the database.

    ``i_pos`` is a position inside ``dataset_self.X[i_file]``, and during imputation there
    are THREE such datasets: ``DISK/impute_dataset.py`` loops over test, val and train, and
    ``create_dataset(sequential=True)`` splits a single recording BY TIME at
    ``p_train_val_test = [0, 0.7, 0.85, 1]``. Each partition's ``X`` therefore starts at a
    different absolute frame, and every record from the val and test slices was being
    attributed to the first 15% of the recording.

    Measured on one 215,439-frame topcam recording: segment windows whose start lay above
    the val-slice length landed on missing data 100.0% of the time (n = 1,169), and those
    below it 45.3% of the time (n = 698) -- chance. No record existed past 0.70 x T at all,
    though 30% of the recording lies there. In ``roselab_disk`` the share of *filled* gaps
    with a null uncertainty jumped from 0.0% across the first seven deciles of every
    recording to 84% (``DLCImputation.Gap``) and 98% (``HeadfixedImputation.Gap``) across
    the last three. Because ``_run_imputation`` attaches uncertainty by taking ``max()``
    over the segments overlapping a gap, the mislabelled records also INFLATED the first
    15%: mean stored uncertainty 3.17 there against 2.454 in the clean middle.

    THE FIX
    -------
    The dataset carries its own absolute time base. ``dataset_self.time[i_file]`` is
    seconds from the start of the RECORDING, not of the partition, at ``1 / self.freq``
    spacing, so the partition's frame offset is ``round(time[i_file][0] * freq)`` -- which
    on the recording above gives 0 for train, 150,807 for val and 183,123 for test, and
    those tile the recording exactly (150,807 + 32,316 + 32,316 = 215,439).

    The offset is computed once per ``(id(dataset), i_file)`` and added to ``i_pos``. If it
    cannot be computed, or is not an integer number of frames within a tenth of a frame,
    the record is emitted with ``start = -1`` -- which ``main()`` already filters out -- so
    the failure mode is a MISSING uncertainty rather than one attached to the wrong frames.
    Failing closed matters more than coverage here: a null tells a consumer it knows
    nothing, while a wrong number tells it something false.
    """

    def __init__(self):
        self.rows: list[dict] = []
        self._original = None
        self._offsets: dict = {}
        self.n_unmappable = 0

    def _offset(self, dataset_self, i_file):
        """Absolute frame of ``dataset_self.X[i_file][0]``, or None if not determinable."""
        cache_key = (id(dataset_self), int(i_file))
        if cache_key in self._offsets:
            return self._offsets[cache_key]
        off = None
        try:
            freq = float(dataset_self.freq)
            t0 = float(np.ravel(dataset_self.time[int(i_file)])[0])
            exact = t0 * freq
            if np.isfinite(exact) and abs(exact - round(exact)) < 0.1:
                off = int(round(exact))
        except Exception:
            off = None
        self._offsets[cache_key] = off
        return off

    def __enter__(self):
        from DISK.utils.dataset_utils import ImputeDataset

        self._original = ImputeDataset.update_dataset
        recorder = self

        def wrapped(dataset_self, index, new_x, uncertainty=None, threshold=0):
            unc = recorder._original(dataset_self, index, new_x, uncertainty, threshold)
            positions = []
            try:
                idx = np.asarray(index)
                pi = dataset_self.possible_indices
                i_file, i_pos, len_ = pi[idx, 0], pi[idx, 1], pi[idx, 2]
                try:
                    len(i_file)
                except Exception:
                    i_file, i_pos, len_ = [[i_file]], [[i_pos]], [[len_]]
                for ii in range(len(i_file)):
                    f = int(np.ravel(i_file[ii])[0])
                    pos = int(np.ravel(i_pos[ii])[0])
                    off = recorder._offset(dataset_self, f)
                    # start is ABSOLUTE; pos_in_partition is kept so a reader can audit
                    # the mapping instead of having to trust it.
                    positions.append((-1 if off is None else off + pos,
                                      int(np.ravel(len_[ii])[0]), f, pos, off))
            except Exception:
                positions = []

            values = list(unc) if isinstance(unc, (list, tuple)) else []
            for j, v in enumerate(values):
                if j < len(positions):
                    start, length, f, pos, off = positions[j]
                else:
                    start, length, f, pos, off = -1, -1, -1, -1, None
                if start < 0:
                    recorder.n_unmappable += 1
                recorder.rows.append({
                    "start": start,
                    "length": length,
                    # None means the network is not probabilistic, i.e. mu_sigma was off
                    # and the refusal rule would have nothing to work with.
                    "uncertainty": float(v) if v is not None else None,
                    # Provenance of the mapping, so a wrong offset is auditable after the
                    # fact rather than invisible.
                    "i_file": f,
                    "pos_in_partition": pos,
                    "partition_offset": off,
                })
            return unc

        ImputeDataset.update_dataset = wrapped
        return self

    def __exit__(self, *exc):
        from DISK.utils.dataset_utils import ImputeDataset

        ImputeDataset.update_dataset = self._original
        return False


def main(jobdir: Path) -> int:
    spec = json.loads((jobdir / "job.json").read_text())
    logger = setup_logger(jobdir / "worker.log")
    logger.info(f"job {jobdir}")

    from DISK.create_dataset import create_dataset
    from DISK.impute_dataset import impute

    project = jobdir / "project"
    dataset_path = project / "DISK_data" / f"len{spec['length']}_stride{spec['stride']}"
    for sub in ("DISK_data", "DISK_train", "DISK_impute"):
        (project / sub).mkdir(parents=True, exist_ok=True)
    dataset_path.mkdir(parents=True, exist_ok=True)

    files = [str(jobdir / "input.csv")]
    (project / "config_project.yaml").write_text(yaml.dump({
        "project_path": str(project), "data_files": files, "file_format": "dlc_csv"}))

    if not (dataset_path / "constants.py").exists():
        # DISK insists on carving a train/val/test split even for a job that has no use
        # for one, and aborts the whole build if any partition comes up empty.
        #
        # With sequential=True it splits ONE recording BY TIME -- p_train_val_test =
        # [0, 0.7, 0.85, 1] in create_dataset -- and raises "no data for <partition> with
        # 0 NaNs" if any slice holds no complete window. On a recording whose clean
        # stretches cluster in time that fails even with hundreds of usable windows
        # elsewhere: scan9FU63S5H_mini2p1_eye_right has 604 and was rejected. It cost 28
        # of Natasha's 268 recordings on 2026-09-14.
        #
        # With sequential=False DISK partitions by POSITION in the file list instead, and
        # demands at least three files. Handing it the same recording three times gives
        # every partition the whole recording, so none can be empty. Imputation reads the
        # entire dataset regardless of partition, so this changes nothing about the
        # result -- only whether the build is allowed to proceed.
        #
        # Tried in that order rather than switched outright: the time split is what the
        # 240 successful recordings were built with, and there is no reason to perturb a
        # path that works. The fallback fires only where the build would otherwise fail.
        def _build(file_list, sequential):
            create_dataset(
                str(dataset_path), file_list, "dlc_csv",
                spec["length"], spec["stride"],
                0,                  # fill_gap: no interpolation, as in training
                sequential,
                spec["fps"], spec["fps"],
                spec["likelihood_threshold"],
                0, 0, None, logger,
            )

        try:
            _build(files, True)
        except ValueError as exc:
            if "no data for" not in str(exc):
                raise
            logger.info("time-split build failed (%s); retrying with the recording "
                        "presented three times and sequential=False" % exc)
            for stale in dataset_path.glob("*"):
                if stale.is_file():
                    stale.unlink()
            _build(files * 3, False)
            logger.info("fallback build succeeded")

    impute_dir = project / "DISK_impute" / spec["model_name"]
    plot_dir = impute_dir / "plots"
    impute_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    with UncertaintyRecorder() as rec:
        ok = impute(
            project_dir=str(project),
            impute_dir=str(impute_dir),
            plot_dir=str(plot_dir),
            file_format="dlc_csv",
            dataset_path=str(dataset_path),
            skeleton_graph=None,
            checkpoint=spec["checkpoint"],
            batch_size=spec.get("batch_size", 64),
            threshold_error_score=np.inf,   # keep every fill; gating happens upstream
            total_n_plots=0,
            plot_only_holes=True,
            missing_pad=[1, 1],
            verbose=0,
            logger=logger,
        )

    out = list(impute_dir.glob("*.csv"))
    if not out:
        logger.error("DISK produced no output CSV")
        (jobdir / "gaps.json").write_text(json.dumps([]))
        print(json.dumps({"ok": False, "reason": "no output csv", "n_gaps": 0}))
        return 1

    shutil.copy(out[0], jobdir / "imputed.csv")
    gaps = [g for g in rec.rows if g["start"] >= 0]
    (jobdir / "gaps.json").write_text(json.dumps(gaps))

    # Say how many records could not be placed. Silence here is what let the frame-mapping
    # bug live in the database: the file simply had fewer rows than it should have, and
    # nothing compared the two numbers.
    if rec.n_unmappable:
        logger.warning("%d of %d uncertainty records could not be mapped to an absolute "
                       "frame and were dropped rather than guessed"
                       % (rec.n_unmappable, len(rec.rows)))
    offsets = sorted({g["partition_offset"] for g in gaps
                      if g.get("partition_offset") is not None})
    logger.info("partition frame offsets seen: %s" % (offsets,))

    logger.info(f"imputed -> {jobdir / 'imputed.csv'}; {len(gaps)} gaps recorded")
    print(json.dumps({"ok": bool(ok), "n_gaps": len(gaps),
                      "imputed_csv": str(jobdir / "imputed.csv")}))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: disk_worker.py <jobdir>")
    raise SystemExit(main(Path(sys.argv[1])))
