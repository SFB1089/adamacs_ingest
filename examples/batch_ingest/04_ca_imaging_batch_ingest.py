"""Batch template for Ca2+ imaging processing."""

from adamacs.pipeline import imaging


def queue_suite2p(scan_keys, paramset_idx, task_mode="trigger"):
    for scan_key in scan_keys:
        # Extend this dict with any lab-specific required columns from ProcessingTask.
        imaging.ProcessingTask.insert1(
            {
                **scan_key,
                "paramset_idx": paramset_idx,
                "task_mode": task_mode,
            },
            skip_duplicates=True,
        )


def populate_imaging(display_progress=True):
    imaging.Processing.populate(display_progress=display_progress)


if __name__ == "__main__":
    print("Edit scan keys and paramset before calling queue_suite2p().")
