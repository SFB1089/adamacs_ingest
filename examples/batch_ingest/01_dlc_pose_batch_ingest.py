"""Batch template for DLC pose ingestion and population."""

from adamacs.pipeline import model


def queue_pose_estimation(scan_keys, model_name):
    for scan_key in scan_keys:
        task_key = {
            **scan_key,
            "model_name": model_name,
        }
        model.PoseEstimationTaskNew.insert1(task_key, skip_duplicates=True)


def populate_pose_estimation(display_progress=True):
    model.PoseEstimationNew.populate(display_progress=display_progress)


if __name__ == "__main__":
    print("Edit scan keys and model name, then call queue_pose_estimation().")
