# Ingest run logging

Every press of **Commit** in the ingest GUI now writes a durable record of what it
did and of everything it failed to do.

## Why

On 2026-09-16 six sessions were deleted deliberately so they could be re-ingested
from scratch. The re-ingest reported

```
Workflow finished successfully for all 6 sessions.
```

while every DLC step in it had died with `PermissionError` — the user's account was
missing the group that the CIFS data mount forces, so the DLC output directory could
not be created. Reconstructing that took a day of forensics across the filesystem and
the DataJoint `~log` tables, because:

* the error scrolled past inside an `ipywidgets.Output`, which is not saved with the
  notebook, so by the next morning the log did not exist;
* session success was counted as "no exception escaped `_process_session`", and every
  step runs under `_run_ingestion_task`, which catches — so nothing ever escaped;
* a step that had nothing to do returned `None` and was logged as `DONE.`, so a skip
  and a success were indistinguishable;
* nothing recorded which DLC model had been selected (that came from a screenshot),
  and nothing recorded the group membership that actually caused the failure.

## What you get

### On disk

One directory per run, printed at the start and at the end of the run:

```
<log_dir>/<UTC timestamp>_<initials>_<run_id>/
    env.json          host, uid/gid/groups, conda env, git SHA + dirty flag,
                      DataJoint target, and a real mkdir probe of every data root
    selections.json   exactly what the GUI was asked to do, verbatim
    console.log       every print from the run, timestamped, including prints
                      from the ingest libraries it calls
    steps.jsonl       one line per step, written as it happens
    summary.json      counts, the failure list and the skip list
```

`log_dir` is `dj.config["custom"]["ingest_log_dir"]` when set, otherwise
`<repo>/logs/ingest`.

### In the database

`roselab_ingest.IngestRun` mirrors the same run, with part tables
`SessionOutcome` (per session) and `Failure` (every failed or skipped step, with its
camera and model name). To find out which run produced a session:

```python
from adamacs.schemas import ingest
ingest.runs_touching("sess9FUDDCP7")
```

`IngestRun` deliberately has **no foreign keys**. The runs worth investigating are
the ones whose sessions were never created, or were later deleted; a foreign key
would make the first unrecordable and cascade the second away.

### In the GUI

The final banner is computed from the step ledger, so a run in which every DLC step
failed can no longer report success:

```
==============================================================
12 steps: 8 ok, 2 skipped, 2 failed
  FAILED  sess9FUDDCP7 scan9FUDDCP7 DLC model Video 2 (NK; General_eye_fullsize…)
          PermissionError: [Errno 13] Permission denied: '…/device_mini2p1_eye_left…'
  SKIPPED sess9FUDDCP7 DLC model Video 3 (YH; TrainingBox2_face…)
          no video matching "face_video" in /datajoint-data/…
full log: /home/…/logs/ingest/20260917T081500Z_NK_4f3a9c21
==============================================================
```

## Three outcomes, not two

`adamacs.ingest_log.StepSkipped` is raised by a step that had nothing to do, and
carries the reason. It is not an error: it is recorded as `skipped`, and the run can
still be a success. `_ingest_dlc_model` raises it when the model's name does not
resolve to a video file in the scan directory.

That case is common and used to be invisible. The video is found by globbing for the
**third `;`-separated field of the model name**, so a model registered as
`JJ; Topcam_mini2p2_Effnet-JJ-2026-02-23; Topcam` searches for `*Topcam*.mp4*` and
matches nothing in a mini2p1 folder, where the file is `…_mini2p1_top_video_….mp4`.

## Failure policy

Nothing in the logging path can fail an ingest. If the log directory cannot be
created the run falls back to a `NullRunLog` and continues; if the database cannot be
reached, the row is skipped and the directory on disk is unaffected.

## Using it outside the GUI

```python
from adamacs.ingest_log import RunLog
from adamacs.helpers.adamacs_ingest_v2 import ingest_run, ingest_context

run = RunLog.start(repo_root=".", user="NK")
with run.tee(), ingest_run(run):
    with ingest_context(session_id="sess…"):
        ...                       # _run_ingestion_task calls in here are recorded
print(run.format_summary())
```

---

# Preflight

Before the Commit button writes anything, the run checks what it is about to need
and reports it. The report is printed, written to `preflight.json` in the run
directory, and recorded as a step.

```
------------------------------------------------------------
preflight: 28 checks, 6 error, 10 warning, 0 could not be checked
  ERROR    sess9FUDDCP7 mini2p1_top JJ; Topcam_mini2p2_Effnet-JJ-2026-02-23; Topcam
           no video matches this model
           searched for "*Topcam*.mp4*" in /datajoint-data/data/nataliak/NK_ROS-2075_…
           -- the pattern is the 3rd ";" field of the model name
  WARNING  sess9FUDDIBK mini2p1_eye_left  several videos match this model
           pattern "*eye1_video*.mp4*" matches 2 files and the ingest takes whichever
           the filesystem lists first: …T15_50_54.mp4, …T15_50_54_deinterlaced.mp4
------------------------------------------------------------
```

## What it checks

| check | why |
|---|---|
| session directory exists | a typo in the filter costs a whole run otherwise |
| **an actual `mkdir` in it** | the 2026-09-16 cause. `os.access` lies on the CIFS mounts, where `forceuid`/`forcegid`/`dir_mode=0775` make the client-side permissions synthetic |
| each selected model is in `model.Model` | an unregistered model is not selectable, and the run then silently does no DLC |
| each model's name resolves to a video | the pattern is the third `;` field of the model name, enforced by nothing |
| more than one video matches | the ingest takes whichever the filesystem lists first |
| `dlc_processed_data_dir` is writable | when set, all DLC output goes there |
| uid / gid / supplementary groups | recorded, because that was the answer last time |

The video-resolution rule is imported from `adamacs_ingest_v2` (`dlc_search_name`,
`dlc_video_candidates`) rather than reimplemented, so the check and the ingest cannot
drift apart.

## Warn or refuse

Warn-and-continue is the default: a preflight that refuses on a check it got wrong is
worse than one nobody reads. To make errors stop the run instead:

```python
ai.select_sessions(sorted_dirs_root, ..., preflight_strict=True)
```

The run then ends before `ingest_session_scan`, with `result: "preflight failed"` in
the summary and a row in `IngestRun`.

## Which video wins

A deinterlaced copy always wins when one exists.

The pattern is a substring and the deinterlaced file is a superstring of the
original, so `*eye1_video*.mp4*` matches both
`..._eye1_video_2025-05-28T15_50_54.mp4` and
`..._eye1_video_2025-05-28T15_50_54_deinterlaced.mp4`; `glob` is unsorted, so which
one the ingest took used to be directory iteration order. Because the deinterlaced
copies were made per file rather than per folder, that was not even consistent within
one session: of the six sessions re-ingested on 2026-09-16, `sess9FUDDIBK` took the
raw file for the left eye and the deinterlaced one for the right, and
`RecordingInfoNew` recorded **25 fps / 5,648 frames** for one camera and
**50 fps / 11,288 frames** for the other — same animal, same 226 s recording.

Ten of the twelve eye recordings in those six sessions had both copies on disk, and
six of them had ingested the raw one. With the rule in place the preflight reports
`video resolves (deinterlaced copy preferred)` and names both files. A real
ambiguity that the rule cannot settle — two deinterlaced copies — is still a
warning.
