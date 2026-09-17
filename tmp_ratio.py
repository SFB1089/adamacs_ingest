"""Is nframes = 2 x timestamp rows the normal state, or did I create it?

Samples eye recordings across the pipeline that were never touched today, and
compares nframes against the number of eye-camera frame events already ingested.
Read-only.
"""
import collections

from adamacs.pipeline import event, model

TOUCHED = {"sess9FUDDIBK", "sess9FUDDWXU", "sess9FUDEGBI", "sess9FUDF29V"}

rows = (model.RecordingInfoNew & 'recording_id LIKE "%mini2p1_eye%"').fetch(
    "session_id", "recording_id", "fps", "nframes", as_dict=True)
print("eye recordings with RecordingInfoNew:", len(rows))

fps_hist = collections.Counter(r["fps"] for r in rows)
print("fps distribution across the pipeline:", dict(sorted(fps_hist.items(),
                                                           key=lambda kv: -kv[1])))
print()

# Where eye-camera events exist, compare their count with nframes.
print("%-14s %-34s %5s %8s %9s %7s" % ("session", "recording", "fps", "nframes",
                                        "events", "ratio"))
print("-" * 86)
checked = 0
for r in rows:
    if r["session_id"] in TOUCHED:
        continue
    side = "left" if r["recording_id"].endswith("left") else "right"
    etype = "eye_%s_frame" % side
    n_ev = len(event.Event & {"session_id": r["session_id"]} & f'event_type LIKE "%{side}%"')
    if not n_ev:
        continue
    ratio = r["nframes"] / n_ev if n_ev else 0
    print("%-14s %-34s %5s %8d %9d %7.3f"
          % (r["session_id"], r["recording_id"], r["fps"], r["nframes"], n_ev, ratio))
    checked += 1
    if checked >= 12:
        break
if not checked:
    print("(no eye recording outside the touched sessions has ingested eye events)")
