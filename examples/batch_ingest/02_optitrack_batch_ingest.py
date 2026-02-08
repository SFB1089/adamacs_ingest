"""Batch template for OptiTrack ingestion/population."""

from adamacs.schemas import mocap, virtual_markers_optitrack


def populate_optitrack_pipeline(display_progress=True):
    mocap.MotionCapture.populate(display_progress=display_progress)
    virtual_markers_optitrack.RigidMouseTracking.populate(
        display_progress=display_progress
    )


if __name__ == "__main__":
    print("Run populate_optitrack_pipeline() after inserting MotionCaptureTask rows.")
