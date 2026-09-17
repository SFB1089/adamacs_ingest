"""Do the eye-camera timestamp CSVs match the raw frame count or the deinterlaced one?

I just doubled nframes for six recordings by repointing them at the deinterlaced
videos. If the timestamp files have one row per RAW frame, that is a problem for
anything aligning frames to time. Read-only.
"""
import pathlib

from adamacs.pipeline import model

FOLDERS = {
    "sess9FUDDCP7": "/datajoint-data/data/nataliak/NK_ROS-2075_2025-05-28_scan9FUDDCP7_sess9FUDDCP7",
    "sess9FUDDIBK": "/datajoint-data/data/nataliak/NK_ROS-2075_2025-05-28_scan9FUDDIBK_sess9FUDDIBK",
    "sess9FUDEO0G": "/datajoint-data/data/nataliak/NK_ROS-2081_2025-05-28_scan9FUDEO0G_sess9FUDEO0G",
    "sess9FUDF29V": "/datajoint-data/data/nataliak/NK_ROS-2082_2025-05-28_scan9FUDF29V_sess9FUDF29V",
}
CAMERAS = [("mini2p1_eye_left", "left_eye1"), ("mini2p1_eye_right", "right_eye2")]

print("%-14s %-18s %10s %10s %10s  %s"
      % ("session", "camera", "ts rows", "nframes", "ratio", "verdict"))
print("-" * 92)
for sess, folder in FOLDERS.items():
    scan = sess.replace("sess", "scan")
    for camera, token in CAMERAS:
        ts = sorted(pathlib.Path(folder).glob("*%s_video_timestamps_*.csv" % token))
        if not ts:
            print("%-14s %-18s %10s" % (sess, camera, "no csv"))
            continue
        with open(ts[0]) as fh:
            rows = sum(1 for _ in fh)
        rec = {"session_id": sess, "recording_id": "%s_%s" % (scan, camera)}
        try:
            n = (model.RecordingInfoNew & rec).fetch1("nframes")
        except Exception:
            n = 0
        ratio = (n / rows) if rows else 0
        if abs(ratio - 1) < 0.02:
            verdict = "matches the ingested video"
        elif abs(ratio - 2) < 0.05:
            verdict = "video has 2x the timestamps  <-- MISMATCH"
        elif abs(ratio - 0.5) < 0.02:
            verdict = "timestamps have 2x the video <-- MISMATCH"
        else:
            verdict = "ratio %.3f" % ratio
        print("%-14s %-18s %10d %10d %10.3f  %s"
              % (sess, camera, rows, n, ratio, verdict))
