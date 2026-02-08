"""Batch template for eye tracking and gaze reconstruction."""

from adamacs.schemas import pupil_tracking, virtual_markers_optitrack


def populate_eye_and_gaze(display_progress=True):
    virtual_markers_optitrack.RigidMouseTracking.populate(
        display_progress=display_progress
    )
    pupil_tracking.PupilEllipseFittingFreeMoving.populate(
        display_progress=display_progress
    )
    pupil_tracking.GazeReconstruction3D.populate(display_progress=display_progress)


if __name__ == "__main__":
    print("Run populate_eye_and_gaze() after required upstream tasks are inserted.")
