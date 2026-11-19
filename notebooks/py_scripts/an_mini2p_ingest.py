
"""Headless replica of the adamacs_ingest_v2 GUI commit path for Anh Nguyen's mini2p1 sessions.
STAGE=dry      : connect as anhn, build selections, run the GUI preflight (no writes)
STAGE=lookups  : session Project 'vis2action' + suite2p paramsets 203 (raw) / 1203 (SUPPORT input), cloned from Shan's 2048
STAGE=ingest   : the GUI commit path (_process_session per session inside RunLog), do_population=False, ingest_opt='trigger'
STAGE=denoise  : notebook-05 DenoisingTask (denoise_paramset_idx=2) for every scan
STAGE=s2pden   : ProcessingTask paramset 1203 for scans whose Denoising is complete
STAGE=optitrack: optitrack_frames from the 'OPTI gated frames' DI line (Anh-only override of the positional AUX mapping)
STAGE=bpod     : VisMotor Bpod trials/events (DotIsOn-exit sync)
STAGE=events   : OptiTrack .tak (notebook 07), left-eye OCR + CameraTimestamps + topcam CSV (notebook 09)
STAGE=bpodcheck: read-only comparison of the Bpod clock fit against trial.Trial
Env: DJ_PASS (Anh's DB password), AN_SCANS=scanA,scanB to restrict, AN_EYES (default mini2p1_eye_left)
"""
import os, sys, json, contextlib, traceback, configparser
import numpy as np
STAGE = os.environ["STAGE"]
# repo root = two levels above notebooks/py_scripts/ (override with ADAMACS_INGEST_ROOT)
WT = os.environ.get("ADAMACS_INGEST_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
ROOT = "/datajoint-data/data/anhn"
sys.path.insert(0, WT)
import datajoint as dj
if os.environ.get("DJ_CONF"):                                     # optional: your own dj_local_conf.json
    dj.config.load(os.environ["DJ_CONF"])
else:                                                              # minimal config; never load another user's conf file
    dj.config["database.host"] = os.environ["DJ_HOST"]            # DB host of the lab server (see lab wiki)
    dj.config["database.port"] = int(os.environ.get("DJ_PORT", "3306"))
    dj.config["enable_python_native_blobs"] = True
    dj.config["fetch_format"] = "array"
    dj.config["custom"] = {"database.prefix": "roselab_",
                           "suite2p_fast_tmp": ["/datajoint-data-2/tmp/"], "suite2p_tif_tmp": ["/datajoint-data-2/tmp/s2p_tiff/"]}
    dj.config["stores"] = {"external-raw": {"protocol": "file", "location": "/datajoint-db/blobs/roselab"}}   # host/stores/prefix; credentials overridden below
dj.config["database.user"] = "anhn"
dj.config["database.password"] = os.environ.pop("DJ_PASS")
dj.config["custom"]["exp_root_data_dir"] = [ROOT]
dj.config["custom"]["imaging_root_data_dir"] = [ROOT]
conn = dj.conn(reset=True)
print("connected as", conn.conn_info["user"], "prefix", dj.config["custom"]["database.prefix"])

SESSIONS = sorted(d for d in os.listdir(ROOT) if d.startswith("AN_ROS-235") and d.split("_")[2] >= "2026-11-11")
print(len(SESSIONS), "session folders")
SAME_SITE = {"ROS-2356": "sess9G3608N1", "ROS-2359": "sess9G361E8C"}   # first Anh session per mouse
TOP = "NK; Topcam_Bodyparts-NK-2025-08-01; top_video"
EYE = "NK; General_eye_fullsize-NK-2025-08-25; eye1_video"        # camera 3 search swaps eye1->eye2 (dlc_search_name)
RAW_IDX, DEN_IDX, SRC_IDX, DENOISE_IDX = 203, 1203, 2048, 2

if STAGE == "lookups":
    from adamacs.pipeline import subject, imaging
    subject.Project.insert1({"project": "vis2action",
        "project_description": "AN: vision-to-action, mini2p1 freely moving, plain vs zebra background"}, skip_duplicates=True)
    src = (imaging.ProcessingParamSet & {"paramset_idx": SRC_IDX}).fetch1()
    raw = dict(src["params"]); raw["fast_disk"] = "/datajoint-data-2/tmp/suite2p/"
    den = dict(raw); den["denoise"] = False
    for idx, p, desc in [(RAW_IDX, raw, "AN: mini2p1 raw, = SM 2048 (fs 37.1, diam10, tau0.7, thr0.8), fast_disk /datajoint-data-2/tmp/suite2p/"),
                         (DEN_IDX, den, "AN: mini2p1 SUPPORT-denoised input, = 203 with s2p denoise=False")]:
        imaging.ProcessingParamSet.insert_new_params(processing_method="suite2p", concatenation_method=src["concatenation_method"],
                                                     paramset_idx=idx, paramset_desc=desc, params=p)
    print((imaging.ProcessingParamSet & "paramset_idx in (203,1203,2048)").fetch("paramset_idx","paramset_desc","concatenation_method", as_dict=True))
    sys.exit(0)

import adamacs.helpers.adamacs_ingest_v2 as ai
from adamacs.pipeline import subject, session, scan, imaging, model, equipment, surgery
from adamacs.ingest_log import RunLog

# --- AN user config, same format as user_configs/SM.ini (indices into the GUI lookup lists) ---
projects = list(subject.Project.fetch("project")); equip = list(equipment.Equipment().fetch("scanner"))
locs = list(surgery.AnatomicalLocation().fetch("anatomical_location"))
dlcs = list(np.insert(model.Model.fetch("model_name"), 0, "dummy"))
s2p_idx = list(imaging.ProcessingParamSet.fetch("paramset_idx"))
need = {"project": "vis2action", "equipment": "mini2p_07_fast", "location": "V1"}
assert need["project"] in projects, "run STAGE=lookups first"
assert RAW_IDX in s2p_idx, "run STAGE=lookups first"
ini = configparser.ConfigParser()
ini["defaults"] = dict(project_idx=projects.index("vis2action"), location_idx=locs.index("V1"), equipment_idx=equip.index("mini2p_07_fast"),
                       s2p_param_idx=RAW_IDX, dlc1_indices=dlcs.index(TOP), dlc2_indices=dlcs.index(EYE), dlc3_indices=dlcs.index(EYE),
                       use_dlc_cropping=False)
ini["cameras"] = dict(camera1="mini2p1_top", camera2="mini2p1_eye_left", camera3="mini2p1_eye_right")
ini["setup"] = dict(aux_setup_type="mini2p1_openfield")
ini["recording_notes"] = {f"note{i}": v for i, v in enumerate(["", "no comment", "plain background", "zebra background", "habituation", "test session"])}
with open(os.path.join(WT, "user_configs", "AN.ini"), "w") as f: ini.write(f)
print("cam defaults", ai.get_user_cam_defaults(SESSIONS)[0], "aux", ai.get_user_aux_defaults(SESSIONS)[0])

selections = []
for d in SESSIONS:
    sid = ai.get_session_key_from_dir([d])[0]
    selections.append({"path": d, "session_id": sid, "project": "vis2action", "location": "V1", "equipment": "mini2p_07_fast",
        "s2p_param_idx": RAW_IDX, "same_site": SAME_SITE[d.split("_")[1]], "dlc_models": [[TOP], [EYE], [EYE]],
        "note": "", "recording_notes": "", "use_dlc_cropping": False, "is_curated": False})

if STAGE in ("dry", "ingest"):
    run = RunLog.start(repo_root=__import__("pathlib").Path(WT), user="AN",
        context={"user_initials": "AN", "n_sessions_offered": len(SESSIONS), "n_sessions_selected": len(selections),
                 "do_population": False, "rspace_upload": False, "ingest_opt": "trigger", "suppress_errors": True,
                 "preflight_strict": False, "driver": "headless replica of GUI commit callback", "stage": STAGE})
    with contextlib.ExitStack() as st:
        st.enter_context(run.tee()); st.enter_context(ai.ingest_run(run))
        print("-- run log:", run.dir)
        run.record_selections(selections)
        cams = ai.get_user_cam_defaults([s["path"] for s in selections]); auxs = ai.get_user_aux_defaults([s["path"] for s in selections])
        ok = ai._run_preflight(run, selections, cams, strict=False)
        if STAGE == "dry":
            run.finish({"result": "dry run (preflight only)"}); print("DRY DONE preflight_ok=", ok); sys.exit(0)
        populate_settings = {"display_progress": False, "suppress_errors": True}
        for i, si in enumerate(selections):
            print(f"\n{'='*50}\nStarting ingestion for session {si['session_id']} ({i+1}/{len(selections)})\n{'='*50}")
            try:
                ai._process_session(si["session_id"], si, cams[i], auxs[i], populate_settings, "trigger", False, False, True)
            except Exception as e:
                with run.step("session processing", key={"session_id": si["session_id"]}) as s: s.failed(e)
                traceback.print_exc(limit=2)
        bad = [s["session_id"] for s in selections if not run.session_ok(s["session_id"])]
        print(run.format_summary())
        summary = run.finish({"sessions_selected": len(selections), "sessions_successful": len(selections)-len(bad),
                              "sessions_failed": len(bad), "sessions_with_failures": bad})
        ai._record_run_in_database(run, summary)
        print("FAILED SESSIONS:", bad)

elif STAGE == "denoise":
    from adamacs.pipeline import denoising
    keys = (scan.Scan & [{"session_id": s["session_id"]} for s in selections]).fetch("KEY")
    for k in keys:
        denoising.DenoisingTask.insert1({**k, "denoise_paramset_idx": DENOISE_IDX,
            "processing_output_dir": (scan.ScanPath() & k).fetch1("path"), "task_mode": "trigger"}, skip_duplicates=True)
    print("DenoisingTask rows:", len(denoising.DenoisingTask & keys & {"denoise_paramset_idx": DENOISE_IDX}), "/", len(keys))

elif STAGE == "s2pden":
    from adamacs.pipeline import denoising
    keys = (scan.Scan & [{"session_id": s["session_id"]} for s in selections]).fetch("KEY")
    done = (denoising.Denoising & keys & {"denoise_paramset_idx": DENOISE_IDX}).fetch("KEY")
    for k in done:
        kk = {"session_id": k["session_id"], "scan_id": k["scan_id"]}
        imaging.ProcessingTask.insert1((kk["session_id"], kk["scan_id"], DEN_IDX, (scan.ScanPath() & kk).fetch1("path"), "trigger"), skip_duplicates=True)
    print("denoised scans:", len(done), "/", len(keys), "  ProcessingTask 1203:", len(imaging.ProcessingTask & keys & {"paramset_idx": DEN_IDX}))

elif STAGE == "optitrack":
    # Anh-only override (decided 2026-11-18): optitrack_frames from the WaveSurfer DI line named 'OPTI gated frames'.
    # ingest_aux maps 7-DI mini2p1_openfield recordings positionally and stores the 2p volume clock as optitrack_frames
    # and the OptiTrack train as HARP_gate. Run AFTER the GUI ingest and BEFORE the events stage (egocam OCR syncs to it).
    import glob, h5py
    from adamacs.pipeline import event
    from adamacs.ingest import behavior as ibe
    keys = (scan.Scan & [{"session_id": s["session_id"]} for s in selections]).fetch("KEY")
    if os.environ.get("AN_SCANS"): keys = [k for k in keys if k["scan_id"] in os.environ["AN_SCANS"].split(",")]
    BRF = [a for a in event.BehaviorRecording.File.heading.names if "file" in a][0]
    for k in keys:
        old = event.Event & k & 'event_type in ("optitrack_frames","HARP_gate")'
        files = [f for f in (event.BehaviorRecording.File & k).fetch(BRF) if str(f).endswith(".h5")]
        if not files: print(k["scan_id"], "no WaveSurfer h5 - skipped"); continue
        f = h5py.File(files[0], "r"); sw = [x for x in f if "sweep" in x][0]
        names = [x.decode() if isinstance(x, bytes) else x for x in np.ravel(f["header"]["DIChannelNames"][()])]
        if "OPTI gated frames" not in names: print(k["scan_id"], "no 'OPTI gated frames' DI line - skipped", names); continue
        sr = float(np.ravel(f["header"]["AcquisitionSampleRate"][()])[0])
        raw = f[sw]["digitalScans"][0].astype(np.int64)
        chan = ((raw >> names.index("OPTI gated frames")) & 1).astype(bool); chan[-1] = False
        ts = np.ravel(ibe.get_timestamps_robust_optitrack(chan, sr))
        n_old = len(event.Event & k & 'event_type="optitrack_frames"')
        if n_old == len(ts):
            print(k["scan_id"], "optitrack_frames already correct", n_old); continue
        if n_old:
            print(k["scan_id"], f"replacing {n_old} positional optitrack_frames/HARP_gate events (needs DELETE privilege)")
            old.delete_quick()
        event.Event.insert([{**k, "event_type": "optitrack_frames", "event_start_time": float(t)} for t in ts],
                           skip_duplicates=True, allow_direct_insert=True)
        tk = glob.glob((scan.ScanPath & k).fetch1("path") + "/*Take*.csv")
        hdr = open(tk[0], errors="ignore").readline().split(",") if tk else []
        motive = hdr[hdr.index("Total Frames in Take") + 1] if "Total Frames in Take" in hdr else "no CSV export"
        print(k["scan_id"], "optitrack_frames", len(ts), "| Motive total frames", motive)

elif STAGE in ("bpod", "events"):
    import pathlib, collections
    from adamacs.schemas import mocap
    from adamacs.ingest import behavior as ibe
    from adamacs.pipeline import event, trial
    keys = (scan.Scan & [{"session_id": s["session_id"]} for s in selections]).fetch("KEY")
    run = RunLog.start(repo_root=pathlib.Path(WT), user="AN", context={"user_initials": "AN", "stage": STAGE,
                       "driver": "headless; same calls as GUI (_ingest_behavioral_data) / notebooks 07+09"})
    with contextlib.ExitStack() as st:
        st.enter_context(run.tee()); st.enter_context(ai.ingest_run(run))
        print("-- run log:", run.dir)
        if STAGE == "events":
            mocap.Mocap.insert1({"mocap_name": "motive_raw_data", "description": "Unprocessed Motive export"}, skip_duplicates=True)
        # Anh, Nov 2026: right eye camera delivered no signal (uniform gray, counter stuck at 0) -> left eye only
        EYES = os.environ.get("AN_EYES", "mini2p1_eye_left").split(",")
        if os.environ.get("AN_SCANS"): keys = [k for k in keys if k["scan_id"] in os.environ["AN_SCANS"].split(",")]
        for k in keys:
            print(f"\n### {k['scan_id']}", flush=True)
            with ai.ingest_context(session_id=k["session_id"], scan_id=k["scan_id"]):
                if STAGE == "bpod":
                    aux = (scan.ScanInfo & k).fetch1("userfunction_info")
                    ai._run_ingestion_task(ibe.ingest_bpod, "BPOD data (VisMotor)", sessi=k["session_id"], scansi=k["scan_id"],
                                           verbose=True, aux_setup_type=aux, include_raw_bpod_events=True, include_raw_bpod_states=True)
                    continue
                def _mocap():
                    # notebook 07 with search_str='Take' (the GUI's '*ROS*.tak*' glob misses these names)
                    p = pathlib.Path((scan.ScanPath() & k).fetch1("path"))
                    taks = sorted(p.glob("*Take*.tak*"))
                    if not taks:
                        from adamacs.ingest_log import StepSkipped
                        raise StepSkipped("no Motive .tak in scan folder")
                    key = {**k, "camera": "mocap"}
                    mocap.MocapRecording.insert1(key, skip_duplicates=True, ignore_extra_fields=True)
                    mocap.MocapRecording.File.insert1({**key, "file_path": str(taks[0]), "file_id": 0}, ignore_extra_fields=True, skip_duplicates=True)
                    mocap.MocapRecordingInfo.populate(k, display_progress=False, suppress_errors=True)
                    mocap.MotionCaptureTask.insert1({**k, "mocap_name": "motive_raw_data"}, ignore_extra_fields=True, skip_duplicates=True)
                    return f"{taks[0].name} (csv export present: {taks[0].with_suffix('.csv').exists()})"
                ai._run_ingestion_task(_mocap, "OptiTrack (notebook 07)")
                ai._run_ingestion_task(lambda: ibe.ingest_egocams_ocr(k["session_id"], k["scan_id"], eye_cameras=EYES, verbose=False, write_log=True), "Egocam OCR frames (notebook 09)")
                ai._run_ingestion_task(lambda: ibe.ingest_camera_timestamps(k["session_id"], k["scan_id"], eye_cameras=EYES + ["mini2p1_worldcam"], verbose=False), "CameraTimestamps (notebook 09)")
                ai._run_ingestion_task(lambda: ibe.ingest_topcam_csv_timestamps(k["session_id"], k["scan_id"], camera_type="mini2p1_top", verbose=False), "Topcam CSV timestamps (notebook 09)")
        print(run.format_summary())
        summary = run.finish({"stage": STAGE})
        ai._record_run_in_database(run, summary)
    print("\ntrials per scan:", collections.Counter((trial.Trial & keys).fetch("scan_id")))
    print("event types:", sorted(collections.Counter((event.Event & keys & 'event_type like "mini2p1_%" or event_type like "bpod_%"').fetch("event_type")).items()))
    print("MocapRecording", len(mocap.MocapRecording & keys), "MotionCaptureTask", len(mocap.MotionCaptureTask & keys))

elif STAGE == "bpodcheck":
    from adamacs.pipeline import trial
    from adamacs.ingest.bpod_vismotor import VisMotorBpodfile, find_vismotor_file, VisMotorPulseMismatch
    from element_interface.utils import find_full_path
    from adamacs.paths import get_experiment_root_data_dir
    keys = (scan.Scan & [{"session_id": s["session_id"]} for s in selections]).fetch("KEY")
    for k in keys:
        d = find_full_path(get_experiment_root_data_dir(), (scan.ScanPath & k).fetch1("path"))
        f = find_vismotor_file(d)
        if f is None:
            print(k["scan_id"], "no VisMotor .mat"); continue
        bp = VisMotorBpodfile(f, k["session_id"])
        try:
            cs, _ = bp._aux_timestamps()
        except VisMotorPulseMismatch as e:
            print(k["scan_id"], "REFUSED:", e); continue
        db = (trial.Trial & k).fetch("trial_start_time", order_by="trial_id")
        shift = (np.asarray(db, float) - cs[: len(db)]) if len(db) else []
        print(k["scan_id"], bp.sync_report, "| DB trials:", len(db), "| DB-minus-corrected start:", (np.round(np.median(shift), 3) if len(db) else "n/a"),
              "| dot_on durations (s): min", round(float(min(bp.trial(n)._states["DotIsOn"][1] for n in range(bp.n_trials))), 3), flush=True)
