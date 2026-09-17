# Tobias Rose 2023: Routine ingest helpers
# Refactored for clarity, modularity, and maintainability.

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, HTML
from natsort import natsorted
import re
import pathlib
from datetime import datetime, date
import traceback
import os
import fnmatch
import tifffile
from scipy.ndimage import mean 
import datajoint as dj
import time
import threading
import contextvars
import contextlib
from contextlib import contextmanager

# =============================================================================
# DEBUG CONFIGURATION - Set to True to enable detailed debug output
# =============================================================================
DEBUG_DLC_PRESELECTION = False  # Enable detailed debug output for DLC model selection

# Import adamacs and element-calcium pipeline components
from adamacs.pipeline import subject, session, scan, equipment, surgery, event, trial, imaging, behavior, model
from adamacs.ingest import session as isess
from adamacs.ingest import behavior as ibe
from adamacs.schemas import mocap
from adamacs.helpers import stack_helpers as sh
from adamacs.helpers.user_defaults_manager import UserDefaultsManager
from adamacs.ingest_log import RunLog, NullRunLog, StepSkipped, get_run_log
from element_interface.utils import find_full_path
from adamacs.paths import get_dlc_root_data_dir

# Define DataJoint objects for convenience
sub, lab, protocol, line, mutation, user, project, subject_genotype, subject_death = (
    subject.Subject(), subject.Lab(), subject.Protocol(), subject.Line(), 
    subject.Mutation(), subject.User(), subject.Project(), subject.SubjectGenotype(), 
    subject.SubjectDeath()
)

# Initialize RSpace API client
try:
    import rspace_client
    # This assumes the API key is configured elsewhere (e.g., environment variables)
    api = rspace_client.Client()
except (ImportError, Exception) as e:
    api = None
    print(f"RSpace client not found or failed to initialize: {e}. RSpace functionality will be disabled.")

# =============================================================================
# Initialize user defaults manager for INI-based configuration
# No hardcoded path: UserDefaultsManager() resolves user_configs/ relative to this
# checkout. The previous value pointed into one lab member's home directory, which is
# not readable by anyone else, so importing this module raised PermissionError for
# every other user before they saw a single line of the GUI.
_user_defaults_manager = UserDefaultsManager()

# ------------------------- USER DEFAULTS (INI-BASED) ------------------------
# =============================================================================

def get_user_defaults(directory):
    """Get user default settings from INI files with fallback to hardcoded values."""
    user_initials = get_user_initials_from_dir(directory)
    
    new_array = []
    for initial in user_initials:
        try:
            # Try to get from INI file first
            user_array = _user_defaults_manager.get_user_array_format(initial)
            new_array.append(user_array)
        except Exception as e:
            print(f"⚠️ INI fallback for {initial}: using hardcoded defaults ({str(e)})")
            # Fallback to hardcoded defaults
            new_array.append(_get_hardcoded_user_defaults(initial))
    
    return new_array

def get_user_cam_defaults(directory):
    """Get user camera settings from INI files with fallback to hardcoded values.""" 
    user_initials = get_user_initials_from_dir(directory)
    
    new_array = []
    for initial in user_initials:
        try:
            # Try to get from INI file first
            cameras = _user_defaults_manager.get_user_cameras(initial)
            new_array.append(cameras)
        except Exception as e:
            print(f"⚠️ INI fallback for {initial}: using hardcoded cameras ({str(e)})")
            # Fallback to hardcoded defaults
            new_array.append(_get_hardcoded_cam_defaults(initial))
    
    return new_array

def get_user_aux_defaults(directory):
    """Get user AUX setup from INI files with fallback to hardcoded values."""
    user_initials = get_user_initials_from_dir(directory)
    
    new_array = []
    for initial in user_initials:
        try:
            # Try to get from INI file first
            aux_setup = _user_defaults_manager.get_user_aux_setup(initial)
            new_array.append(aux_setup)
        except Exception as e:
            print(f"⚠️ INI fallback for {initial}: using hardcoded AUX setup ({str(e)})")
            # Fallback to hardcoded defaults
            new_array.append(_get_hardcoded_aux_defaults(initial))
    
    return new_array

def get_user_recording_notes_defaults(directory):
    """Get user recording notes from INI files with fallback to hardcoded values."""
    user_initials = get_user_initials_from_dir(directory)
    
    new_array = []
    for initial in user_initials:
        try:
            # Try to get from INI file first
            recording_notes = _user_defaults_manager.get_user_recording_notes(initial)
            new_array.append(recording_notes)
        except Exception as e:
            print(f"⚠️ INI fallback for {initial}: using hardcoded recording notes ({str(e)})")
            # Fallback to hardcoded defaults
            new_array.append(_get_hardcoded_recording_notes_defaults(initial))
    
    return new_array

def get_user_dlc_cropping_defaults(directory):
    """Get DLC cropping settings from user INI files or fallback to hardcoded defaults."""
    user_initials = get_user_initials_from_dir(directory)
    
    new_array = []
    for initial in user_initials:
        try:
            # Try to get from INI file first
            use_cropping = _user_defaults_manager.get_user_dlc_cropping(initial)
            new_array.append(use_cropping)
        except Exception as e:
            print(f"⚠️ INI fallback for {initial}: using default DLC cropping (True) ({str(e)})")
            # Fallback to True (enable cropping by default)
            new_array.append(True)
    
    return new_array

# ------------------------- LEGACY HARDCODED DEFAULTS (FALLBACK) --------------
# =============================================================================

def _get_hardcoded_user_defaults(initial):
    """Legacy hardcoded user defaults for fallback."""
    hardcoded_defaults = {
        'JJ': [4, 7, 7, 0, 2, 8, 8],
        'RN': [2, 2, 0, 0, 7, 8, 8],
        'TR': [2, 2, 0, 0, 7, 8, 8],
        'LK': [9, 9, 0, 10, 10, 8, 8],
        'DB': [4, 0, 1, 0, 8, 8, 8],
        'NK': [7, 5, 1, 66, [12, 14, 15], 13, 13],
        'LE': [9, 9, 0, 10, 10, 8, 8],
        'AM': [8, 8, 0, 7, 7, 8, 8],
        'AA': [6, 5, 1, 66, 14, 0, 0],
        'SM': [5, 7, 10, 16, 2, 0, 0],
        'YH': [8, 5, 1, 66, 14, 0, 0],
        'KH': [9, 9, 0, 10, 10, 0, 0]
    }
    return hardcoded_defaults.get(initial, [0, 0, 0, 0, 0, 0, 0])

def _get_hardcoded_cam_defaults(initial):
    """Legacy hardcoded camera defaults for fallback."""
    hardcoded_cameras = {
        'JJ': ['mini2p1_top', 'mini2p1_eye_left', 'mini2p1_eye_right'],
        'RN': ['bench2p_face', 'bench2p_body', 'bench2p_back'],
        'TR': ['mini2p1_top', 'mini2p1_eye_left', 'mini2p1_eye_right'],
        'LK': ['bench2p_face', 'bench2p_body', 'bench2p_back'],
        'DB': ['bench2p_face', 'bench2p_body', 'bench2p_back'],
        'NK': ['mini2p1_top', 'mini2p1_eye_left', 'mini2p1_eye_right'],
        'LE': ['bench2p_face', 'bench2p_body', 'bench2p_back'],
        'AM': ['bench2p_face', 'bench2p_body', 'bench2p_back'],
        'AA': ['bench2p_face', 'bench2p_body', 'bench2p_back'],
        'SM': ['mini2p1_top', 'mini2p1_eye_left', 'mini2p1_eye_right'],
        'YH': ['bench2p_face', 'bench2p_body', 'bench2p_back'],
        'KH': ['bench2p_face', 'bench2p_body', 'bench2p_back']
    }
    return hardcoded_cameras.get(initial, ['camera1', 'camera2', 'camera3'])

def _get_hardcoded_aux_defaults(initial):
    """Legacy hardcoded AUX defaults for fallback."""
    hardcoded_aux = {
        'JJ': 'mini2p1_openfield',
        'RN': 'bench2p',
        'TR': 'mini2p1_openfield',
        'LK': 'bench2p',
        'DB': 'bench2p',
        'NK': 'mini2p1_openfield',
        'LE': 'bench2p',
        'AM': 'bench2p',
        'AA': 'behavior_box',
        'SM': 'mini2p1_openfield',
        'YH': 'behavior_box',
        'KH': 'bench2p'
    }
    return hardcoded_aux.get(initial, 'mini2p1_openfield')

def _get_hardcoded_recording_notes_defaults(initial):
    """Legacy hardcoded recording notes for fallback."""
    hardcoded_notes = {
        'JJ': ['', 'no comment', 'shaping', 'probe', 'recall', 'extinction', 'reversal', 'baseline', 'training', 'test session', 'habituation'],
        'RN': ['', 'no comment', 'baseline', 'probe trial', 'recall test', 'training session', 'behavioral shaping', 'performance test', 'motor learning'],
        'TR': ['', 'no comment', 'shaping', 'probe', 'recall', 'extinction', 'reversal', 'baseline', 'training', 'test session', 'habituation'],
        'LK': ['', 'no comment', 'baseline', 'probe trial', 'recall test', 'training session', 'behavioral shaping', 'performance test', 'motor learning'],
        'DB': ['', 'no comment', 'baseline', 'probe trial', 'recall test', 'training session', 'behavioral shaping', 'performance test', 'motor learning'],
        'NK': ['', 'no comment', 'shaping', 'probe', 'recall', 'extinction', 'reversal', 'baseline', 'training', 'test session', 'habituation'],
        'LE': ['', 'no comment', 'baseline', 'probe trial', 'recall test', 'training session', 'behavioral shaping', 'performance test', 'motor learning'],
        'AM': ['', 'no comment', 'baseline', 'probe trial', 'recall test', 'training session', 'behavioral shaping', 'performance test', 'motor learning'],
        'AA': ['', 'no comment', 'operant conditioning', 'free choice', 'forced choice', 'go/no-go', 'discrimination', 'reward learning', 'punishment'],
        'SM': ['', 'no comment', 'shaping', 'probe', 'recall', 'extinction', 'reversal', 'baseline', 'training', 'test session', 'habituation'],
        'YH': ['', 'no comment', 'operant conditioning', 'free choice', 'forced choice', 'go/no-go', 'discrimination', 'reward learning', 'punishment'],
        'KH': ['', 'no comment', 'baseline', 'probe trial', 'recall test', 'training session', 'behavioral shaping', 'performance test', 'motor learning']
    }
    return hardcoded_notes.get(initial, ['', 'no comment', 'baseline', 'training', 'test session'])

# ------------------------- HELPER FUNCTIONS ----------------------------------
# =============================================================================


def get_session_dir_key_from_dir(directory):
    return [path.split('/')[-1] for path in directory]
     
def get_scan_dir_key_from_dir(directory):
    return [path.split('/')[-1] for path in directory]

def get_session_key_from_dir(string):
    result = []
    for item in string:
        match = re.search(r'sess\S+', item)
        if match:
            result.append(match.group(0))
    return result

def get_user_initials_from_dir(string):
    result = [name[:2] for name in string]
    return result

def get_subject_key_from_dir(string):
    result = [item.split("_")[1] for item in string]
    return result

def get_date_key_from_dir(directory):
    return directory.split("_")[-3]


def parse_date_token(token, *, allow_month=False):
    token = str(token).strip()
    # tolerate wildcard/comparator-friendly forms like >2025-05-01*
    token = token.rstrip("*").strip()
    token = token.rstrip("-_/").strip()

    date_pattern = r"(20\d{2}[-_]\d{2}[-_]\d{2}|20\d{6})"
    if allow_month:
        date_pattern = r"(20\d{2}[-_]\d{2}[-_]\d{2}|20\d{6}|20\d{2}[-_]\d{2})"

    date_match = re.search(date_pattern, token)
    if date_match:
        token = date_match.group(1)

    formats = ["%Y-%m-%d", "%Y_%m_%d", "%Y%m%d"]
    if allow_month:
        formats.extend(["%Y-%m", "%Y_%m"])

    for fmt in formats:
        try:
            return datetime.strptime(token, fmt).date()
        except ValueError:
            pass
    return None


def extract_session_date(session_name):
    patterns = [
        r"(?<!\d)(20\d{2})[-_](\d{2})[-_](\d{2})(?!\d)",
        r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)",
    ]
    for pattern in patterns:
        match = re.search(pattern, str(session_name))
        if not match:
            continue
        year, month, day = map(int, match.groups())
        try:
            return date(year, month, day)
        except ValueError:
            continue
    return None


def parse_date_filter(expr):
    expr = (expr or "*").strip()
    if expr in {"", "*"}:
        return {"mode": "all"}

    if ":" in expr and not expr.startswith((">", "<")):
        lo_raw, hi_raw = expr.split(":", 1)
        lo = parse_date_token(lo_raw, allow_month=True) if lo_raw.strip() else None
        hi = parse_date_token(hi_raw, allow_month=True) if hi_raw.strip() else None
        if lo is None and hi is None:
            raise ValueError(
                f"Invalid ADAMACS_DATE_FILTER range: {expr!r}. "
                "Expected YYYY-MM-DD:YYYY-MM-DD (either side can be omitted)."
            )
        return {"mode": "range", "lo": lo, "hi": hi}

    for op in (">=", "<=", ">", "<"):
        if expr.startswith(op):
            rhs = parse_date_token(expr[len(op) :].strip(), allow_month=True)
            if rhs is None:
                raise ValueError(
                    f"Invalid ADAMACS_DATE_FILTER comparator: {expr!r}. "
                    "Expected forms like >=2025-01-01 or <2025-02-01."
                )
            return {"mode": "cmp", "op": op, "rhs": rhs}

    exact = parse_date_token(expr, allow_month=False)
    if exact is not None:
        return {"mode": "exact", "rhs": exact}

    return {"mode": "substring", "expr": expr}


def date_filter_matches(session_name, parsed_filter):
    mode = parsed_filter["mode"]
    if mode == "all":
        return True
    if mode == "substring":
        expr = parsed_filter["expr"]
        if fnmatch.fnmatch(session_name, f"*{expr}*"):
            return True
        session_date = extract_session_date(session_name)
        return session_date is not None and expr in session_date.isoformat()

    session_date = extract_session_date(session_name)
    if session_date is None:
        return False

    if mode == "exact":
        return session_date == parsed_filter["rhs"]
    if mode == "cmp":
        rhs = parsed_filter["rhs"]
        op = parsed_filter["op"]
        if op == ">=":
            return session_date >= rhs
        if op == "<=":
            return session_date <= rhs
        if op == ">":
            return session_date > rhs
        if op == "<":
            return session_date < rhs
        return False
    if mode == "range":
        lo = parsed_filter["lo"]
        hi = parsed_filter["hi"]
        if lo is not None and session_date < lo:
            return False
        if hi is not None and session_date > hi:
            return False
        return True
    return False


def filter_session_dirs(session_dirs, session_filter="*", date_filter="*"):
    parsed_date_filter = parse_date_filter(date_filter)
    session_matched_dirs = [d for d in session_dirs if fnmatch.fnmatch(d, session_filter)]
    filtered_dirs = [
        d for d in session_matched_dirs if date_filter_matches(d, parsed_date_filter)
    ]
    unmatched_date_dirs = [
        d for d in session_matched_dirs if extract_session_date(d) is None
    ]
    return filtered_dirs, parsed_date_filter, unmatched_date_dirs

def get_scan_key_from_dir(string):
    result = []
    for item in string:
        match = re.search(r'scan\S+_', item)
        if match:
            result.append(match.group(0)[:-1])
    return result

def unique_directory_strings(dirs1, dirs2):
    set1 = set(dirs1)
    set2 = set(dirs2)
    common_dirs = list(set1.intersection(set2))
    unique_dirs = list(set(set1.union(set2)) - set(common_dirs))
    return unique_dirs

# =============================================================================
# ------------------------- RUN LOGGING CONTEXT -------------------------------
# =============================================================================
# The run log is carried in context variables rather than threaded through every
# signature, so that _process_session, _process_scan, _populate_dlc and the rest
# keep the signatures other notebooks already call them with.

_RUN = contextvars.ContextVar('adamacs_ingest_run', default=None)
_KEY = contextvars.ContextVar('adamacs_ingest_key', default=None)


def current_run_log():
    """The RunLog for the ingest in progress, or a NullRunLog outside one."""
    return get_run_log(_RUN.get())


@contextmanager
def ingest_run(run):
    """Make `run` the active run log for the duration of the block."""
    token = _RUN.set(run)
    try:
        yield run
    finally:
        _RUN.reset(token)


@contextmanager
def ingest_context(**key):
    """Tag every step recorded inside the block with session_id / scan_id / ..."""
    merged = dict(_KEY.get() or {})
    merged.update({k: v for k, v in key.items() if v is not None})
    token = _KEY.set(merged)
    try:
        yield merged
    finally:
        _KEY.reset(token)


def _run_preflight(run, selections, user_cam_defaults, strict=False):
    """Check what the ingest is about to need. Returns False only to stop the run.

    The report is printed and written to the run directory either way. `strict`
    decides whether an error stops the run or is only reported: warn-and-continue is
    the default, because a preflight that refuses on a check it got wrong is worse
    than one nobody reads.
    """
    try:
        from adamacs.ingest_preflight import run_preflight
        from adamacs.paths import get_dlc_processed_data_dir
    except Exception as exc:
        # Fail closed: a caller who asked the preflight to guard every write must not
        # get an unguarded run because the check itself could not be imported.
        print(f'-- preflight unavailable ({type(exc).__name__}: {exc}); '
              f'{"refusing to continue (strict)" if strict else "continuing"}')
        return not strict

    try:
        data_root = dj.config.get('custom', {}).get('exp_root_data_dir', [None])
        data_root = data_root[0] if isinstance(data_root, (list, tuple)) else data_root
        report = run_preflight(
            selections, data_root,
            cameras=user_cam_defaults,
            model_table=model.Model,
            get_processed_dir=get_dlc_processed_data_dir,
        )
    except Exception as exc:
        print(f'-- preflight could not run ({type(exc).__name__}: {exc}); '
              f'{"refusing to continue (strict)" if strict else "continuing"}')
        traceback.print_exc(limit=2)
        return not strict

    print(report.format())
    run.record('preflight', report.as_json())
    with run.step('preflight', key=None) as s:
        counts = report.counts
        if report.ok:
            s.ok('%d checks, %d warnings' % (sum(counts.values()), counts['warning']))
        else:
            s.failed(RuntimeError(
                '%d preflight error(s): %s' % (
                    counts['error'],
                    '; '.join('%s %s' % (f.where(), f['check'])
                              for f in report.of('error')[:5]))))

    if report.ok:
        return True
    if strict:
        print('\nPreflight found errors and strict mode is on. Nothing was written.')
        return False
    print('\nPreflight found errors. Continuing anyway (pass preflight_strict=True '
          'to select_sessions to stop instead).')
    return True


def _record_run_in_database(run, summary):
    """Mirror the run into roselab_ingest.IngestRun, so that "which run produced this
    session, and what did it report" stays answerable after the log directory is gone.

    Imported lazily and guarded: the schema is optional, and an ingest must not fail
    because its bookkeeping could not be stored.
    """
    try:
        from adamacs.schemas import ingest as ingest_schema
    except Exception as exc:
        print(f'-- ingest run not recorded in the database '
              f'({type(exc).__name__}: {exc}); the log on disk is unaffected')
        return False
    return ingest_schema.record_run(run, summary)


def _run_ingestion_task(task_func, description, **kwargs):
    """Run one ingestion step, printing its progress and recording its outcome.

    Three outcomes, not two. A step that had nothing to do raises StepSkipped and is
    recorded as `skipped` with its reason, instead of returning None and being logged
    as DONE -- which is how "no video matched this DLC model" used to be
    indistinguishable from success.

    The return value says whether the step succeeded. Callers may ignore it; the
    ledger in the run log does not, and it is what the final summary is built from.
    """
    run = current_run_log()
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] -- {description}...', end='')

    with run.step(description, key=_KEY.get()) as step:
        try:
            result = task_func(**kwargs)
        except StepSkipped as skip:
            step.skipped(str(skip))
            ts_skip = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'[{ts_skip}] SKIPPED: {skip}')
            return False
        except Exception as e:
            step.failed(e)
            ts_err = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'[{ts_err}] FAILED: {description}. Error: {e}')
            # Two frames on screen; the full traceback goes to the run log.
            traceback.print_exc(limit=2)
            return False

        step.ok(result)
        ts_done = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # If DLC model ingestion returned a video filename, include it in the log
        if result and task_func.__name__ == '_ingest_dlc_model':
            print(f'[{ts_done}] DONE. (Video: {result})')
        else:
            print(f'[{ts_done}] DONE.')
        return True

# =============================================================================
# ---------------------- INGESTION SUB-ROUTINES -------------------------------
# =============================================================================

def _update_all_metadata(sessi, scansi, session_info, ingest_opt):
    """
    Inserts or updates all metadata from the UI for a given session and scan.
    This avoids using DELETE or `replace=True` as per user constraints.
    """
    session_key = {'session_id': sessi}
    scan_key = {'session_id': sessi, 'scan_id': scansi}

    # ---- Session Level Metadata ----
    # Project (manual table, relation between session and project)
    # NOTE: Without DELETE, we cannot change the project once set. We only insert if not present.
    session.ProjectSession.insert1({**session_key, 'project': session_info['project']}, skip_duplicates=True)
    
    # Session Note (part table)
    # This is a part table, so we can update the note attribute.
    # First, insert with skip_duplicates=True to ensure the session entry exists.
    session.SessionNote.insert1({**session_key, 'session_note': session_info['note']}, skip_duplicates=True)
    # Then, update the value. This ensures it works for both new and existing entries.
    session.SessionNote.update1({**session_key, 'session_note': session_info['note']})
    
    # Recording Notes in event.BehaviorRecording (if recording_notes is provided)
    if 'recording_notes' in session_info and session_info['recording_notes']:
        # Update recording notes in existing BehaviorRecording entries for this session
        behavior_recordings = event.BehaviorRecording & f'session_id = "{sessi}"'
        if behavior_recordings:
            for recording_key in behavior_recordings.fetch('KEY'):
                event.BehaviorRecording.update1({**recording_key, 'recording_notes': session_info['recording_notes']})
    
    # Same Site ID (part table)
    # This is a part table, so we can update the same_site_id attribute.
    session.SessionSameSite.insert1({**session_key, 'same_site_id': session_info['same_site']}, skip_duplicates=True)
    session.SessionSameSite.update1({**session_key, 'same_site_id': session_info['same_site']})

    # ---- Scan Level Metadata ----
    # Equipment (attribute in master table scan.Scan) & Scan Notes
    scan.Scan.update1({**scan_key, 'scanner': session_info['equipment'], 'scan_notes': session_info['note']})
    
    # Anatomical Location (part table)
    # This is a part table, so we can update the anatomical_location attribute.
    scan.ScanLocation.insert1({**scan_key, 'anatomical_location': session_info['location']}, skip_duplicates=True)
    scan.ScanLocation.update1({**scan_key, 'anatomical_location': session_info['location']})

    # ---- Suite2p Processing Task ----
    # The PK of ProcessingTask is (session_id, scan_id, paramset_idx).
    # Changing the paramset_idx in the UI creates a new PK.
    # NOTE: Without DELETE, we cannot remove previous tasks. New tasks will be added.
    # (imaging.ProcessingTask & scan_key).delete()
    
    if session_info['s2p_param_idx'] != 'dummy':
        dir_proc = (scan.ScanPath() & scan_key).fetch1('path')
        task_tuple = (
            sessi, 
            scansi, 
            session_info['s2p_param_idx'], 
            dir_proc, 
            ingest_opt
        )
        imaging.ProcessingTask.insert1(task_tuple, skip_duplicates=True)


def _ingest_behavioral_data(sessi, scansi, aux_setup_typestr):
    """Ingest all behavior-related data: AUX, STIM (Trial Times), and BPOD."""
    # Skip AUX data ingestion for behavior_box setups
    if "behavior_box" not in aux_setup_typestr:
        _run_ingestion_task(ibe.ingest_aux, 'AUX data', session_key=sessi, scan_key=scansi, verbose=False, aux_setup_type=aux_setup_typestr)
    else:
        print('-- Skipping AUX data ingestion for behavior_box setup.')
    
    # Check if the experiment type requires trial time ingestion
    behavior_setups = ["bench2p", "openfield", "mini2p1_openfield", "behavior_box"]
    if any(s in aux_setup_typestr for s in behavior_setups):
        _run_ingestion_task(ibe.get_and_ingest_trial_times, 'Trial Times', scan_key=scansi, aux_setup_type=aux_setup_typestr)

    # Always ingest BPOD data regardless of setup type
    _run_ingestion_task(
        ibe.ingest_bpod,
        'BPOD data',
        sessi=sessi,
        scansi=scansi,
        verbose=True,
        aux_setup_type=aux_setup_typestr,
        include_raw_bpod_events=True,
        include_raw_bpod_states=True,
    )

def _populate_behavior_tasks(scansi, aux_setup_typestr, populate_settings):
    """Populate computed behavior tables like Treadmill and HARP IMU."""
    ingest_query = (event.BehaviorRecording & f'scan_id = "{scansi}"')
    if "bench2p" in aux_setup_typestr:
        _run_ingestion_task(lambda **kwargs: behavior.TreadmillRecording.populate(ingest_query, **kwargs), 'Treadmill data', **populate_settings)
    if "mini2p1" in aux_setup_typestr:
        _run_ingestion_task(lambda **kwargs: behavior.HarpRecording.populate(ingest_query, **kwargs), 'HARP IMU data', **populate_settings)
        _run_ingestion_task(lambda **kwargs: behavior.CamSyncRecording.populate(ingest_query, **kwargs), 'CamSync data', **populate_settings)

def _populate_optitrack(scansi, aux_setup_typestr, populate_settings):
    """Find and populate OptiTrack/Motive motion capture data."""
    if "openfield" not in aux_setup_typestr and "mini2p1" not in aux_setup_typestr:
        return # Skip if not an openfield-type experiment

    mocap.Mocap.insert1({"mocap_name": "motive_raw_data", "description": "Unprocessed Motive export"}, skip_duplicates=True)
    
    scan_key = (scan.Scan & f'scan_id = "{scansi}"').fetch1('KEY')
    scan_path = (scan.ScanPath() & scan_key).fetch1("path")
    
    # Search for only .tak files
    # mocap_files = list(pathlib.Path(scan_path).glob("*ROS*.csv")) + list(pathlib.Path(scan_path).glob("*ROS*.tak*"))
    mocap_files = list(pathlib.Path(scan_path).glob("*ROS*.tak*"))
    
    if not mocap_files:
        return # No files found

    # Ingest the first file found
    key = scan_key.copy()
    key['camera'] = 'mocap'
    mocap.MocapRecording.insert1(key, skip_duplicates=True)
    
    key['file_path'] = str(mocap_files[0])
    key['file_id'] = 0
    mocap.MocapRecording.File.insert1(key, ignore_extra_fields=True, skip_duplicates=True)
    mocap.MocapRecordingInfo.populate(key, **populate_settings)
    
    # Create and populate the motion capture task
    task_key = scan_key.copy()
    task_key['mocap_name'] = 'motive_raw_data'
    mocap.MotionCaptureTask.insert1(task_key, ignore_extra_fields=True, skip_duplicates=True)
    # mocap.MotionCapture.populate(task_key, **populate_settings)

def _populate_cascade(scansi, paramset_idx, populate_settings):
    """Populate CASCADE activity extraction data based on the calcium indicator."""
    indicator_query = (subject.Subject * session.Session() * subject.Line() * scan.Scan() & f'scan_id = "{scansi}"')
    
    if not indicator_query:
        print(f"\n-- Could not find subject/line info for scan {scansi} to determine CASCADE model. Skipping.")
        return
        
    indicator = indicator_query.fetch1('line_name')
    
    if 'GCaMP8s' in indicator:
        model_name = 'GC8s_EXC_30Hz_smoothing25ms_high_noise'
    elif 'GCaMP6s' in indicator:
        model_name = 'Global_EXC_30Hz_smoothing50ms_high_noise'
    else:
        return # No model defined for this indicator

    try:
        # Find the corresponding fluorescence trace to create a cascade task
        insert_key = (imaging.Fluorescence * imaging.ProcessingParamSet.proj('processing_method') * imaging.ActivityExtractionMethod
                      & f'scan_id = "{scansi}"'
                      & f'paramset_idx = {paramset_idx}'
                      & 'curation_id = 1'
                      & 'extraction_method = "cascade_inference"').fetch1()
        insert_key['model_name'] = model_name
        imaging.ActivityCascadeTask.insert1(insert_key, ignore_extra_fields=True, skip_duplicates=True)
    except Exception:
        # This can fail if upstream tables aren't populated (e.g., do_population=False), which is expected.
        pass

def dlc_search_name(model_name, video_idx):
    """The model name used for video lookup on camera `video_idx`.

    The third camera is the right eye, and the eye models are all named for eye1, so
    the lookup name swaps in eye2. Factored out so that the preflight check and the
    ingest itself cannot drift apart.
    """
    if video_idx == 2 and "eye1" in model_name:
        return model_name.replace("eye1", "eye2")
    return model_name


def dlc_search_string(search_name):
    """The filename pattern a model name resolves to: its third ';'-separated field.

    This is a convention, not something the Model table enforces, so a model
    registered as 'JJ; Topcam_mini2p2_Effnet-JJ-2026-02-23; Topcam' searches for
    '*Topcam*.mp4*' and matches nothing in a mini2p1 folder. The preflight check
    reports that before anything is written.
    """
    try:
        return search_name.split(';')[2].replace(" ", "")
    except IndexError:
        return "top"  # Default search string if not specified in model name


def is_deinterlaced(path):
    """Whether a video file is a deinterlaced copy.

    Substring and case-insensitive, matching `_prefer_deinterlaced_video_files` in
    adamacs.ingest.behavior, so that the DLC lookup and the eye-camera ingest agree on
    what counts. Requiring the exact "_deinterlaced" would miss a separator variant
    such as "eye1_video-deinterlaced.mp4" and hand the choice back to the filesystem.
    """
    return "deinterlaced" in str(getattr(path, "name", path)).lower()


def dlc_video_candidates(scan_path, search_name):
    """(search_str, [paths]) that `search_name` resolves to in `scan_path`, best first.

    A deinterlaced copy always wins when one exists. The pattern is a substring and
    the deinterlaced file is a superstring of the original -- `*eye1_video*.mp4*`
    matches both `..._eye1_video_2025-05-28T15_50_54.mp4` and
    `..._eye1_video_2025-05-28T15_50_54_deinterlaced.mp4` -- and `glob` is unsorted,
    so which one the ingest took used to be directory iteration order. Because the
    deinterlaced copies were made per file rather than per folder, that was not even
    consistent inside one session: of the six sessions re-ingested on 2026-09-16,
    sess9FUDDIBK took the raw file for the left eye and the deinterlaced one for the
    right, and RecordingInfoNew recorded 25 fps for one camera and 50 for the other.

    Remaining ties are still the filesystem's order, and the preflight says so.
    """
    search_str = dlc_search_string(search_name)
    hits = list(pathlib.Path(scan_path).glob(f"*{search_str}*.mp4*"))
    deinterlaced = [p for p in hits if is_deinterlaced(p)]
    if deinterlaced:
        return search_str, deinterlaced + [p for p in hits if p not in deinterlaced]
    return search_str, hits


def _ingest_dlc_model(scan_key, model_name, camera, aux_setup_typestr, search_model_name=None, use_cropping=True):
    """Ingest a single DLC model, including video file and pose estimation task."""
    # Use search_model_name if provided (for eye camera handling), otherwise model_name
    search_name = search_model_name if search_model_name is not None else model_name
    search_str = dlc_search_string(search_name)

    scan_path = (scan.ScanPath() & scan_key).fetch1("path")
    search_str, movie_paths = dlc_video_candidates(scan_path, search_name)

    if not movie_paths:
        # Not an error, but not success either: the model's name did not resolve to a
        # video in this scan's directory. Raised rather than printed so that it is
        # recorded as a skip with its reason and appears in the run summary.
        raise StepSkipped(
            f'no video matching "{search_str}" in {scan_path} '
            f'(search string comes from the 3rd ";" field of "{search_name}")'
        )

    # Get just the filename (not full path) for cleaner logging
    video_filename = movie_paths[0].name

    rec_id = f"{scan_key['scan_id']}_{camera}"
    key = {**scan_key, 'recording_id': rec_id, 'camera': camera}

    # Insert video recording and file info
    model.VideoRecordingNew.insert1(key, skip_duplicates=True)
    
    # Create a separate key for file insertion to avoid modifying the base key
    file_key = {**key, 'file_path': str(movie_paths[0]), 'file_id': 0}
    model.VideoRecordingNew.File.insert1(file_key, ignore_extra_fields=True, skip_duplicates=True)

    # Create pose estimation task with optional dynamic cropping
    task_key = (model.VideoRecordingNew & f'recording_id="{rec_id}"').fetch1('KEY')
    task_key.update({'model_name': model_name, 'task_mode': 'trigger'})
    
    # Set analyze_videos_params based on cropping preference
    if use_cropping:
        analyze_videos_params = {'save_as_csv': True, 'dynamic': (True, .5, 60)}
        if DEBUG_DLC_PRESELECTION:
            print(f'-- Using DLC dynamic cropping for {video_filename}')
    else:
        analyze_videos_params = {'save_as_csv': True}
        if DEBUG_DLC_PRESELECTION:
            print(f'-- DLC dynamic cropping disabled for {video_filename}')
    
    model.PoseEstimationTaskNew.insert_estimation_task(task_key, task_key["model_name"], analyze_videos_params=analyze_videos_params)
    
    return video_filename
    
    # Handle DLClive for specific setups
    if "mini2p1_openfield" in aux_setup_typestr:
        model.DLCliveRecording.insert1(key, skip_duplicates=True) # Use original key
        model.DLCliveRecording.File.insert1(file_key, ignore_extra_fields=True, skip_duplicates=True) # Use file key
        task_key.update({'task_mode': 'load'})
        model.DLCLivePoseEstimationTask.insert1(task_key, ignore_extra_fields=True, skip_duplicates=True)

def _populate_dlc(scans_to_process, session_info, usercam_defaults_i, useraux_default_i, use_dlc_cropping=True):
    """Populate all selected DLC models for a session."""
    dlc_models = session_info['dlc_models']  # Now a list of 3 lists: [models_for_video1, models_for_video2, models_for_video3]
    
    # NOTE: As per user constraints, existing DLC tasks cannot be deleted.
    # If you change a model in the UI, a new task will be created, but the old one will remain.
    # Deselecting a model ('dummy') will not remove any existing tasks for that scan.

    run = current_run_log()

    # Check if any models are selected across all three video files
    if all(len(models) == 0 or all(m == 'dummy' for m in models) for models in dlc_models):
        # Previously a bare return: an ingest that did no DLC at all looked identical
        # to one that did. Record it, with what the cameras were offered.
        print('-- No DLC models selected; skipping DLC for this session.')
        with run.step('DLC', key=_KEY.get()) as s:
            s.skipped('no DLC models selected in the GUI for any of the three cameras')
        return # Skip if no DLC models are selected.

    for scansi in scans_to_process:
        scan_key = (scan.Scan & f'scan_id = "{scansi}"').fetch1('KEY')
        try:
            aux_setup_typestr = (scan.ScanInfo() & scan_key).fetch1("userfunction_info")
        except Exception as e:
            print(f'-- Could not fetch aux_setup_typestr for DLC processing of scan {scansi}, using user default "{useraux_default_i}". Error: {e}')
            # print(f'-- Error traceback:\n{traceback.format_exc()}')
            aux_setup_typestr = useraux_default_i  # Use user-specific default for aux setup type
        
        if "openfield" not in aux_setup_typestr and "bench2p" not in aux_setup_typestr and "behavior_box" not in aux_setup_typestr:
            with ingest_context(scan_id=scansi):
                with run.step('DLC', key=_KEY.get()) as s:
                    s.skipped(f'setup type "{aux_setup_typestr}" is not openfield / '
                              f'bench2p / behavior_box')
            continue # Skip if not a relevant experiment type

        print(f'-- Processing DLC for scan: {scansi}')
        
        # Process each of the three video files
        for video_idx, models_for_video in enumerate(dlc_models):
            camera = usercam_defaults_i[video_idx] if video_idx < len(usercam_defaults_i) else "camera_0"
            
            for model_name in models_for_video:
                if model_name != 'dummy':
                    # Special handling for eye cameras: if model contains "eye1" and this is the third video (video_idx == 2),
                    # replace "eye1" with "eye2" in the search string for video file detection
                    search_model_name = dlc_search_name(model_name, video_idx)

                    with ingest_context(scan_id=scansi, camera=camera,
                                        model_name=model_name):
                        _run_ingestion_task(
                            _ingest_dlc_model, f'DLC model Video {video_idx+1} ({model_name})',
                            scan_key=scan_key, model_name=model_name, camera=camera,
                            aux_setup_typestr=aux_setup_typestr, search_model_name=search_model_name, use_cropping=use_dlc_cropping
                        )

# =============================================================================
# ------------------------- CORE PROCESSING LOGIC -----------------------------
# =============================================================================

def _process_scan(sessi, scansi, session_info, populate_settings, ingest_opt, is_update=False, suppress_errors=True, user_aux_default="behavior_box"):
    """Run all ingestion and population tasks for a single scan."""
    print(f'Processing scan: {scansi}')
    run = current_run_log()

    with ingest_context(session_id=sessi, scan_id=scansi):
        # 1. Insert/Update all metadata from UI. This is always run to capture UI changes.
        _run_ingestion_task(_update_all_metadata, 'Session/Scan metadata', sessi=sessi, scansi=scansi, session_info=session_info, ingest_opt=ingest_opt)

        # 2. Populate ScanInfo to get metadata required for subsequent steps
        # The restriction must be passed into populate, not applied before it.
        aux_setup_source = 'scan.ScanInfo'
        try:
            scan.ScanInfo.populate(f'scan_id = "{scansi}"' , **populate_settings)
            aux_setup_typestr = (scan.ScanInfo() & f'scan_id = "{scansi}"').fetch1("userfunction_info")
            if 'dummy' in (scan.ScanInfo.ScanFile &  f'scan_id = "{scansi}"').fetch1('file_path'):
                print(f'-- Found dummy file for scan {scansi}, using user default "{user_aux_default}".')
                aux_setup_typestr = user_aux_default
                aux_setup_source = 'user default (dummy scan file)'
        except Exception as e:
            print(f'-- Could not fetch aux_setup_typestr for scan {scansi}, using user default "{user_aux_default}". Error: {e}')
            # traceback.print_exc(limit=2)  # Always print traceback to log
            aux_setup_typestr = user_aux_default  # Use user-specific default for aux setup type
            aux_setup_source = f'user default (fetch failed: {type(e).__name__}: {e})'
            if not suppress_errors:
                raise

        # Record the value the run actually used, and where it came from. The two
        # fallbacks above silently change what every later step does.
        with run.step('effective aux_setup_type', key=_KEY.get()) as s:
            s.ok(f'{aux_setup_typestr} (from {aux_setup_source})')

        # 3. Ingest behavioral and physiology data only for new sessions
        if not is_update:
            _ingest_behavioral_data(sessi, scansi, aux_setup_typestr)
        else:
            print('-- Skipping behavioral data ingestion for existing session.')
            with run.step('Behavioral data ingestion', key=_KEY.get()) as s:
                s.skipped('session already ingested (is_update=True)')

        # 4. Populate computed tables
        _run_ingestion_task(_populate_behavior_tasks, 'Behavior tasks (Treadmill/HARP)', scansi=scansi, aux_setup_typestr=aux_setup_typestr, populate_settings=populate_settings)
        _run_ingestion_task(_populate_optitrack, 'OptiTrack data', scansi=scansi, aux_setup_typestr=aux_setup_typestr, populate_settings=populate_settings)
        _run_ingestion_task(_populate_cascade, 'CASCADE data', scansi=scansi, paramset_idx=session_info['s2p_param_idx'], populate_settings=populate_settings)

def _process_session(sessi, session_info, usercam_defaults_i, useraux_default_i, populate_settings, ingest_opt, do_population, rspace_upload, suppress_errors=True):
    """Run all processing for a single session, including all its scans.

    Thin wrapper so that every step recorded below is tagged with this session_id
    without the body having to pass it anywhere.
    """
    with ingest_context(session_id=sessi):
        return _process_session_body(
            sessi, session_info, usercam_defaults_i, useraux_default_i,
            populate_settings, ingest_opt, do_population, rspace_upload,
            suppress_errors=suppress_errors,
        )


def _process_session_body(sessi, session_info, usercam_defaults_i, useraux_default_i, populate_settings, ingest_opt, do_population, rspace_upload, suppress_errors=True):
    print(f'\n{"="*30}\nProcessing session: {sessi}\n{"="*30}')
    run = current_run_log()

    # Check if the session is already ingested to determine if this is an update.
    is_update = bool(session.Session & f'session_id = "{sessi}"')
    with run.step('session state', key=_KEY.get()) as s:
        s.ok('update' if is_update else 'new session')

    # 1. Ingest session and discover scans from the directory.
    # This populates session.Session, scan.Scan, and related tables.
    print('-- Ingesting session and discovering scans...', end='')
    with run.step('Session/scan discovery', key=_KEY.get()) as s:
        try:
            # This function call is crucial and was previously misplaced inside the scan loop.
            isess.ingest_session_scan(
                sessi,
                verbose=False,
                project_key=session_info['project'],
                equipment_key=session_info['equipment'],
                location_key=session_info['location'],
                software_key='ScanImage'
            )
            print('Done.')
            s.ok()
        except Exception as e:
            s.failed(e)
            print(f'Failed during initial session ingestion. Error: {e}')
            traceback.print_exc(limit=2)  # Always print traceback to log
            if not suppress_errors:
                raise
            else:
                # Everything below, DLC included, is skipped for this session.
                # Say so, rather than leaving a silent return.
                print(f'-- Continuing with next session due to suppress_errors=True')
                print('-- NOTE: DLC and all later steps are skipped for this session.')
                return # Stop processing this session but continue with others

    # 2. Now that scans are in the database, fetch them for further processing.
    scans_to_process = (scan.Scan & f'session_id = "{sessi}"').fetch("scan_id")
    if not scans_to_process.size:
        print(f'-- No scans found for session {sessi} after ingestion. Check directory structure and logs.')
    
    for scansi in scans_to_process:
        _process_scan(sessi, scansi, session_info, populate_settings, ingest_opt, is_update=is_update, suppress_errors=suppress_errors, user_aux_default=useraux_default_i)

    # DLC processing is done at the session level after all individual scans are handled
    use_dlc_cropping = session_info.get('use_dlc_cropping', True)  # Default to True if not specified
    _populate_dlc(scans_to_process, session_info, usercam_defaults_i, useraux_default_i, use_dlc_cropping)

    # Run main population tasks for all tables if requested
    if do_population:
        print('\n---- Populating main imaging and DLC tables ----')
        try:
            imaging.Processing.populate(**populate_settings)

            try:
                for scansi in scans_to_process:
                    scan_key = (scan.Scan() & f'scan_id = "{scansi}"').fetch1('KEY')
                    # Note: Original curation logic was complex and depended on UI state.
                    # This is a placeholder to be reviewed.
                    manual_curation = session_info['is_curated']
                    imaging.Curation().create1_from_processing_task(scan_key, is_curated=manual_curation)
            except Exception as e:
                print(f"-- Could not create curation task for {sessi}. Error: {e}")
                traceback.print_exc(limit=2)  # Always print traceback to log
                if not suppress_errors:
                    raise

            _run_ingestion_task(imaging.MotionCorrection.populate, 'Motion Correction', **populate_settings)
            _run_ingestion_task(imaging.Segmentation.populate, 'Segmentation', **populate_settings)
            _run_ingestion_task(imaging.MaskClassification.populate, 'Mask Classification', **populate_settings)
            _run_ingestion_task(imaging.Fluorescence.populate, 'Fluorescence', **populate_settings)
            _run_ingestion_task(imaging.Activity.populate, 'Activity', **populate_settings)
            
            query_all = session.Session() * scan.Scan() & f'session_id = "{sessi}"'
            _run_ingestion_task(lambda **kwargs: model.RecordingInfoNew.populate(query_all, **kwargs), 'DLC RecordingInfo', **populate_settings)
            _run_ingestion_task(lambda **kwargs: model.PoseEstimationNew.populate(query_all, **kwargs), 'DLC PoseEstimation', **populate_settings)

        except Exception as e:
            print(f'Main population tasks failed for {sessi}. Error: {e}')
            traceback.print_exc(limit=2)  # Always print traceback to log
            if not suppress_errors:
                raise

    # Upload to RSpace if requested
    if rspace_upload:
        print('\n---- Uploading to RSpace ----')
        try:
            query = session.Session() * subject.User() & f'session_id = "{sessi}"'
            animalID = query.fetch1("subject")
            date = query.fetch1("session_datetime").strftime("%Y-%m-%d")
            userID = query.fetch1("initials")
            make_rspace_session_document(userID, animalID, date, sessi)
        except Exception as e:
            print(f'-- Failed to upload Rspace data for session = "{sessi}". Error: {e}')
            traceback.print_exc(limit=2)  # Always print traceback to log
            if not suppress_errors:
                raise

def _get_widget_values(available_sessions, all_widgets, s2pparm_options):
    """Extracts all selected values from the UI widgets into a structured dictionary."""
    selections = []
    for i, session_path in enumerate(available_sessions):
        if all_widgets[i]['run_checkbox'].value:
            s2p_param_text = all_widgets[i]['s2p_dropdown'].value
            # Find the index for the chosen s2p parameter description
            s2p_param_idx = s2pparm_options[0][list(s2pparm_options[1]).index(s2p_param_text)]

            # Get project name directly (no ID format needed)
            project_name = all_widgets[i]['project_dropdown'].value

            selections.append({
                'path': session_path,
                'session_id': get_session_key_from_dir([session_path])[0],
                'project': project_name,  # Use extracted project name
                'location': all_widgets[i]['location_dropdown'].value,
                'equipment': all_widgets[i]['equipment_dropdown'].value,
                's2p_param_idx': s2p_param_idx,
                'same_site': all_widgets[i]['same_site_dropdown'].value,
                'dlc_models': [
                    # Video 1: multi-select widget
                    list(all_widgets[i]['dlc1_multi'].value),
                    
                    # Video 2: multi-select widget
                    list(all_widgets[i]['dlc2_multi'].value),
                    
                    # Video 3: multi-select widget
                    list(all_widgets[i]['dlc3_multi'].value)
                ],
                'note': all_widgets[i]['note_textbox'].value,
                'recording_notes': all_widgets[i]['recording_notes_dropdown'].value,
                'use_dlc_cropping': all_widgets[i]['dlc_cropping_checkbox'].value,
                'is_curated': all_widgets[i]['curated_checkbox'].value
            })
    return selections

def _commit_button_callback(b, available_sessions, all_widgets, s2pparm_options, output_widget, progress_container, spinner_container, progress_bar, progress_status_label, timer_label, success_container, do_population, rspace_upload, ingest_opt, suppress_errors=True, preflight_strict=False):
    """Callback function for the 'Commit' button. Gathers UI data and starts processing."""
    output_widget.clear_output()
    success_container.layout.visibility = 'hidden'
    progress_container.layout.visibility = 'visible'
    spinner_container.layout.visibility = 'visible'
    progress_bar.value = 0
    progress_bar.bar_style = 'info'
    progress_status_label.value = "Preparing to process..."
    start_time = time.time()
    # Start background thread to update elapsed time every second
    stop_event = threading.Event()
    def _update_timer():
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            mins, secs = divmod(elapsed, 60)
            timer_label.value = f"Elapsed time: {int(mins):02d}:{int(secs):02d}"
            time.sleep(1)
    timer_thread = threading.Thread(target=_update_timer, daemon=True)
    timer_thread.start()
    
    # One directory per run, holding the environment, the GUI's selections, the full
    # console output and a per-step ledger. Created before anything is touched so that
    # a run which dies early still leaves a record.
    selections_preview = _get_widget_values(available_sessions, all_widgets, s2pparm_options)
    _initials = (get_user_initials_from_dir([s['path'] for s in selections_preview])[0]
                 if selections_preview else None)
    run = RunLog.start(
        repo_root=pathlib.Path(__file__).resolve().parents[2],
        user=_initials,
        context={
            'user_initials': _initials,
            'n_sessions_offered': len(available_sessions),
            'n_sessions_selected': len(selections_preview),
            'do_population': do_population,
            'rspace_upload': rspace_upload,
            'ingest_opt': ingest_opt,
            'suppress_errors': suppress_errors,
            'preflight_strict': preflight_strict,
        },
    )

    # Entered through an ExitStack rather than a `with` block so that the body below
    # keeps its indentation and the diff stays reviewable; closed in the finally.
    _log_ctx = contextlib.ExitStack()
    _log_ctx.enter_context(run.tee())
    _log_ctx.enter_context(ingest_run(run))

    try:
        with output_widget:
            if run.dir:
                print(f'-- run log: {run.dir}')
            selections = selections_preview
            run.record_selections(selections)
            if not selections:
                progress_status_label.value = "No sessions selected. Nothing to do."
                spinner_container.layout.visibility = 'hidden'
                print('No sessions selected (tick the run checkbox on a row). Nothing to do.')
                run.finish({'result': 'nothing selected'})
                return

            user_cam_defaults = get_user_cam_defaults([s['path'] for s in selections])
            user_aux_defaults = get_user_aux_defaults([s['path'] for s in selections])
            populate_settings = {'display_progress': False, 'suppress_errors': True}

            # Everything this run is about to need, checked before anything is
            # written: can we create files where the output goes, is each selected
            # model registered, and does its name actually resolve to a video?
            if not _run_preflight(run, selections, user_cam_defaults,
                                  strict=preflight_strict):
                progress_status_label.value = "Preflight failed. Nothing was written."
                progress_bar.bar_style = 'danger'
                spinner_container.layout.visibility = 'hidden'
                _record_run_in_database(run, run.finish({'result': 'preflight failed'}))
                return

            num_sessions = len(selections)
            progress_bar.max = num_sessions
            progress_status_label.value = f"Processing {num_sessions} sessions..."

            successful_sessions = 0
            failed_sessions = 0

            for i, session_info in enumerate(selections):
                # Log clear separator and verbose header for each session
                ts_session = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n{'='*50}\n[{ts_session}] Starting ingestion for session {session_info['session_id']} ({i+1}/{num_sessions})\n{'='*50}")
                progress_status_label.value = f"({i+1}/{num_sessions}) {session_info['session_id']}"
                
                try:
                    _process_session(
                        session_info['session_id'],
                        session_info,
                        user_cam_defaults[i],
                        user_aux_defaults[i],
                        populate_settings,
                        ingest_opt,
                        do_population,
                        rspace_upload,
                        suppress_errors
                    )
                    successful_sessions += 1
                except Exception as e:
                    # Record it as a step so the ledger stays the single source of
                    # truth. Counting it only here would leave this session absent
                    # from sessions_with_failures, and IngestRun would then file it
                    # as 'ok'.
                    with run.step('session processing',
                                  key={'session_id': session_info['session_id']}) as s:
                        s.failed(e)
                    print(f"\nERROR processing session {session_info['session_id']}: {e}")
                    traceback.print_exc(limit=2)  # Always print traceback to log
                    if suppress_errors:
                        print("-- Continuing with next session due to suppress_errors=True")
                    else:
                        # Re-raise the exception to break the loop
                        raise
                
                progress_bar.value = i + 1
                elapsed_seconds = time.time() - start_time
                mins, secs = divmod(elapsed_seconds, 60)
                timer_label.value = f"Elapsed time: {int(mins):02d}:{int(secs):02d}"
            
            # Update final status message.
            #
            # Session success is taken from the step ledger, not from "no exception
            # escaped _process_session". Those are not the same thing: every step runs
            # under _run_ingestion_task, which catches, so a session in which all DLC
            # steps failed used to be counted as a success and the run reported as
            # "finished successfully".
            step_counts = run.counts()
            sessions_with_failures = [
                s['session_id'] for s in selections
                if not run.session_ok(s['session_id'])
            ]
            # Derived from the ledger alone. Exceptions that escape _process_session
            # are recorded as steps above, so they are already in here; taking a max
            # against a separate counter would under-report when both kinds occur.
            failed_sessions = len(sessions_with_failures)
            successful_sessions = num_sessions - failed_sessions

            if failed_sessions == 0 and step_counts['failed'] == 0:
                finished_msg = f'\n\n{"="*30}\nWorkflow finished successfully for all {successful_sessions} sessions.\n{"="*30}'
                progress_status_label.value = f"Successfully processed {successful_sessions} sessions."
                progress_bar.bar_style = 'success'
                success_container.layout.visibility = 'visible'
            else:
                finished_msg = f'\n\n{"="*30}\nWorkflow finished: {successful_sessions} successful, {failed_sessions} failed.\n{"="*30}'
                progress_status_label.value = (
                    f"{successful_sessions} ok, {failed_sessions} failed, "
                    f"{step_counts['failed']} failed steps.")
                progress_bar.bar_style = 'warning' if successful_sessions > 0 else 'danger'

            print(finished_msg)
            print(run.format_summary())
            summary = run.finish({
                'sessions_selected': num_sessions,
                'sessions_successful': successful_sessions,
                'sessions_failed': failed_sessions,
                'sessions_with_failures': sessions_with_failures,
            })
            _record_run_in_database(run, summary)

    except Exception as e:
        error_msg = f"An error occurred during processing: {e}"
        progress_status_label.value = "Error! Check log output for details."
        progress_bar.bar_style = 'danger'
        with output_widget:
            print(error_msg)
            traceback.print_exc()  # Always print full traceback for top-level errors
        with run.step('commit callback', key=None) as s:
            s.failed(e)
        # An aborted run is the one most likely to be looked up later, so it gets the
        # same database record as a completed one.
        aborted_summary = run.finish({'result': 'aborted'})
        _record_run_in_database(run, aborted_summary)
    finally:
        # Stop timer thread and hide progress elements
        stop_event.set()
        progress_container.layout.visibility = 'hidden'
        spinner_container.layout.visibility = 'hidden'
        _log_ctx.close()
        if run.dir:
            with output_widget:
                print(f'-- run log written to: {run.dir}')

# =============================================================================
# ------------------------- UI & MAIN FUNCTION --------------------------------
# =============================================================================

def _create_session_widgets(db_info, defaults):
    """Creates all ipywidgets for a single session row in the UI."""
    
    # Determine widget values: use DB info if available, otherwise use user defaults
    project_name = db_info.get('project', defaults['project'])
    
    # Find the display format for project (name with ID)
    project_val = project_name  # Default fallback
    if 'project_names_lookup' in defaults:
        try:
            project_idx = defaults['project_names_lookup'].index(project_name)
            project_val = defaults['project_options'][project_idx]
        except (ValueError, IndexError):
            # If project not found, use the first option as fallback
            project_val = defaults['project_options'][0] if defaults['project_options'] else project_name
    
    location_val = db_info.get('location', defaults['location'])
    equipment_val = db_info.get('equipment', defaults['equipment'])
    
    # Find s2p description from index for dropdown, with robust fallback
    s2p_idx = db_info.get('s2p_param_idx', defaults['s2p_param_idx'])
    s2p_indices = list(defaults['s2p_options'][0])
    s2p_descs = defaults['s2p_options'][1]
    try:
        list_idx = s2p_indices.index(s2p_idx)
        s2p_val = s2p_descs[list_idx]
    except ValueError:
        # Fallback to default if the saved index is somehow invalid
        default_idx = s2p_indices.index(defaults['s2p_param_idx'])
        s2p_val = s2p_descs[default_idx]

    same_site_val = db_info.get('same_site', defaults['same_site'])
    
    # Correctly handle DLC models, falling back to defaults if the list is empty or missing
    dlc_vals = db_info.get('dlc_models')
    
    # Check if dlc_vals is None or contains only empty lists
    if not dlc_vals or (isinstance(dlc_vals, list) and all(len(lst) == 0 for lst in dlc_vals)):
        dlc_vals = defaults['dlc_models']
        
    # Ensure dlc_vals has proper structure and valid values
    if not isinstance(dlc_vals, list) or len(dlc_vals) < 3:
        dlc_vals = [[], [], []]
        
    # Filter dlc_vals to only include valid options that exist in dlc_options
    valid_options = [opt for opt in defaults['dlc_options'] if opt != 'dummy']
    dlc_vals = [
        [val for val in dlc_list if val in valid_options] if dlc_list else []
        for dlc_list in dlc_vals[:3]  # Take only first 3 lists
    ]
        
    note_val = db_info.get('note', 'no comment')
    recording_notes_val = db_info.get('recording_notes', '')
    
    # Handle recording notes: if database value is not in user options, add it to the options
    recording_notes_options = defaults['recording_notes_options'].copy()
    if recording_notes_val and recording_notes_val not in recording_notes_options:
        recording_notes_options.append(recording_notes_val)
    # If no value from database, use first option as default
    if not recording_notes_val:
        recording_notes_val = recording_notes_options[0]
    
    # Define consistent layout for widgets to ensure alignment in the GUI
    w_layout = widgets.Layout(width='150px')
    s2p_layout = widgets.Layout(width='200px')
    dlc_layout = widgets.Layout(width='200px')
    note_layout = widgets.Layout(width='200px')
    check_layout = widgets.Layout(width='70px', display='flex', justify_content='center')


    widgets_dict = {
        'project_dropdown': widgets.Dropdown(options=defaults['project_options'], value=project_val, layout=w_layout),
        'location_dropdown': widgets.Dropdown(options=defaults['location_options'], value=location_val, layout=w_layout),
        'equipment_dropdown': widgets.Dropdown(options=defaults['equipment_options'], value=equipment_val, layout=w_layout),
        's2p_dropdown': widgets.Dropdown(options=defaults['s2p_options'][1], value=s2p_val, layout=s2p_layout),
        'same_site_dropdown': widgets.Dropdown(options=db_info.get('same_site_options', [defaults['same_site']]), value=same_site_val, layout=w_layout),
        'dlc1_multi': widgets.SelectMultiple(
            options=sorted([opt for opt in defaults['dlc_options'] if opt != 'dummy']),
            value=tuple(dlc_vals[0]),
            description='',
            layout=widgets.Layout(
                width='200px', 
                height='80px',  # Increased default height
                resize='both',  # Allow resizing in both directions
                overflow='auto'  # Add scrollbars when needed
            )
        ),
        'dlc2_multi': widgets.SelectMultiple(
            options=sorted([opt for opt in defaults['dlc_options'] if opt != 'dummy']),
            value=tuple(dlc_vals[1]),
            description='',
            layout=widgets.Layout(
                width='200px', 
                height='80px',  # Increased default height
                resize='both',  # Allow resizing in both directions
                overflow='auto'  # Add scrollbars when needed
            )
        ),
        'dlc3_multi': widgets.SelectMultiple(
            options=sorted([opt for opt in defaults['dlc_options'] if opt != 'dummy']),
            value=tuple(dlc_vals[2]),
            description='',
            layout=widgets.Layout(
                width='200px', 
                height='80px',  # Increased default height
                resize='both',  # Allow resizing in both directions
                overflow='auto'  # Add scrollbars when needed
            )
        ),
        'note_textbox': widgets.Textarea(value=note_val, layout=note_layout, rows=1),
        'recording_notes_dropdown': widgets.Dropdown(options=recording_notes_options, value=recording_notes_val, layout=note_layout),
        'dlc_cropping_checkbox': widgets.Checkbox(value=db_info.get('use_dlc_cropping', defaults.get('use_dlc_cropping', True)), layout=check_layout, indent=False, description='Crop'),
        'run_checkbox': widgets.Checkbox(value=not db_info.get('is_ingested', False), layout=check_layout, indent=False),
        'curated_checkbox': widgets.Checkbox(value=db_info.get('is_curated', False), layout=check_layout, indent=False)
    }
    
    # Auto-scroll DLC multi-select widgets to first preselected item
    _auto_scroll_to_preselected(widgets_dict)
    
    return widgets_dict

def _auto_scroll_to_preselected(widgets_dict):
    """Improve visibility of preselected items by organizing options."""
    # For SelectMultiple widgets, we can't easily control scroll position,
    # but sorting already helps with organization.
    # The preselected values will be highlighted in blue automatically.
    # Future enhancement: Could add JavaScript-based scrolling if needed.
    pass

def _get_db_info(session_id, subject_id, session_path, subject_session_ids=None, user_cam_defaults=None):
    """Fetch existing information from the database for a given session to pre-populate the UI."""
    db_info = {'is_ingested': False, 'dlc_models': None} # Initialize with None to allow defaults fallback
    
    # ALWAYS populate same_site_options from the complete subject session list (database + current batch)
    if subject_session_ids:
        db_info['same_site_options'] = sorted(subject_session_ids)
        if DEBUG_DLC_PRESELECTION:
            print(f"   🎯 Setting same_site_options for {session_id}: {db_info['same_site_options']}")
    else:
        db_info['same_site_options'] = [session_id]
        if DEBUG_DLC_PRESELECTION:
            print(f"   ⚠️ No subject sessions found for {session_id}, using just current session")
    
    if not (session.Session & f'session_id = "{session_id}"'):
        # For new sessions, we've already set same_site_options above
        return db_info

    db_info['is_ingested'] = True
    try:
        # Safely fetch attributes, taking the first result if multiple exist (for scan-level data)
        # and handling cases where no data is found to prevent crashes.
        
        project_q = (session.ProjectSession & f'session_id = "{session_id}"').fetch("project")
        if project_q.size > 0: db_info['project'] = project_q[0]

        location_q = (scan.ScanLocation & f'session_id = "{session_id}"').fetch("anatomical_location")
        if location_q.size > 0: db_info['location'] = location_q[0]

        equipment_q = (scan.Scan & f'session_id = "{session_id}"').fetch("scanner")
        if equipment_q.size > 0: db_info['equipment'] = equipment_q[0]

        note_q = (session.SessionNote & f'session_id = "{session_id}"').fetch("session_note")
        if note_q.size > 0: db_info['note'] = note_q[0]

        # Fetch recording notes from event.BehaviorRecording
        recording_notes_q = (event.BehaviorRecording & f'session_id = "{session_id}"').fetch("recording_notes")
        if recording_notes_q.size > 0: db_info['recording_notes'] = recording_notes_q[0]

        same_site_q = (session.SessionSameSite & f'session_id = "{session_id}"').fetch("same_site_id")
        if same_site_q.size > 0: db_info['same_site'] = same_site_q[0]

        # Prioritize the latest processed task for the s2p parameter set default.
        # Fetches the paramset_idx from the most recent 'imaging.Processing' entry.
        latest_processed_q = (imaging.Processing & f'session_id="{session_id}"').fetch(
            'paramset_idx', order_by='processing_time DESC', limit=1
        )
        
        if latest_processed_q.size > 0:
            db_info['s2p_param_idx'] = latest_processed_q[0]
        else:
            # Fallback: If no processing has occurred, use the highest paramset_idx from the task table.
            s2p_task_q = (imaging.ProcessingTask & f'session_id="{session_id}"').fetch("paramset_idx")
            if s2p_task_q.size > 0:
                db_info['s2p_param_idx'] = np.max(s2p_task_q)

        curated_q = (imaging.Curation & f'session_id="{session_id}"').fetch("manual_curation")
        if curated_q.size > 0: db_info['is_curated'] = bool(curated_q[0])
        
        # Fetch all DLC models for this session - simplified query without problematic join
        try:
            # Query PoseEstimationTaskNew directly - this is more reliable than the join
            pose_query = model.PoseEstimationTaskNew & f'session_id="{session_id}"'
            if pose_query:
                # Fetch model names and recording_ids for this session
                pose_data = pose_query.fetch('model_name', 'recording_id')
                if DEBUG_DLC_PRESELECTION:
                    print(f"   🔍 Found {len(pose_data[0])} PoseEstimationTaskNew entries for {session_id}")
                
                # If we have DLC models, organize them by camera
                if len(pose_data[0]) > 0:
                    models_by_camera = {}
                    
                    # For each model, determine camera from recording_id
                    for model_name, recording_id in zip(pose_data[0], pose_data[1]):
                        if model_name != 'dummy':
                            # Extract camera name from recording_id (e.g., 'scan123_mini2p1_top' -> 'mini2p1_top')
                            camera = recording_id.split('_')[-1] if '_' in recording_id else 'unknown'
                            
                            if camera not in models_by_camera:
                                models_by_camera[camera] = []
                            if model_name not in models_by_camera[camera]:
                                models_by_camera[camera].append(model_name)
                    
                    if DEBUG_DLC_PRESELECTION:
                        print(f"   📊 Organized models by camera: {list(models_by_camera.keys())}")
                        for cam, models in models_by_camera.items():
                            print(f"     {cam}: {models}")
                    
                    # Convert to the expected format: three lists for three cameras
                    # Use actual camera names from user defaults if available
                    if user_cam_defaults and len(user_cam_defaults) >= 3:
                        camera_names = user_cam_defaults[:3]  # First three camera names
                        
                        # Create mapping from database camera names to user camera names
                        # Database cameras: 'top', 'left', 'right'
                        # User cameras: 'mini2p1_top', 'mini2p1_eye_left', 'mini2p1_eye_right'
                        camera_mapping = {}
                        for user_cam in camera_names:
                            if 'top' in user_cam.lower():
                                camera_mapping[user_cam] = models_by_camera.get('top', [])
                            elif 'left' in user_cam.lower():
                                camera_mapping[user_cam] = models_by_camera.get('left', [])
                            elif 'right' in user_cam.lower():
                                camera_mapping[user_cam] = models_by_camera.get('right', [])
                            else:
                                # Fallback: try exact match first, then partial matches
                                if user_cam in models_by_camera:
                                    camera_mapping[user_cam] = models_by_camera[user_cam]
                                else:
                                    # Find best partial match
                                    best_match = []
                                    for db_cam, models in models_by_camera.items():
                                        if db_cam.lower() in user_cam.lower() or user_cam.lower() in db_cam.lower():
                                            best_match = models
                                            break
                                    camera_mapping[user_cam] = best_match
                        
                        camera_models = [camera_mapping.get(cam, []) for cam in camera_names]
                        if DEBUG_DLC_PRESELECTION:
                            print(f"   ✅ Mapped database cameras to user order:")
                            for i, (user_cam, db_models) in enumerate(zip(camera_names, camera_models)):
                                print(f"     {user_cam}: {len(db_models)} models → {db_models[:2]}{'...' if len(db_models) > 2 else ''}")
                    else:
                        # Fallback to common camera patterns if user defaults not available
                        camera_patterns = ['top', 'eye_left', 'eye_right']
                        camera_models = []
                        for pattern in camera_patterns:
                            # Find camera that matches pattern
                            matching_cameras = [cam for cam in models_by_camera.keys() if pattern in cam]
                            if matching_cameras:
                                camera_models.append(models_by_camera[matching_cameras[0]])
                            else:
                                camera_models.append([])
                    
                    # Only use database DLC models if at least one camera has models
                    if any(len(models) > 0 for models in camera_models):
                        db_info['dlc_models'] = camera_models
                    else:
                        db_info['dlc_models'] = None  # Will trigger fallback to user defaults
                else:
                    db_info['dlc_models'] = None  # No models found, use defaults
            else:
                db_info['dlc_models'] = None  # No query results, use defaults
                
        except Exception as dlc_error:
            print(f"Warning: Could not fetch DLC models for {session_id}. Using defaults. Error: {dlc_error}")
            db_info['dlc_models'] = None  # Will trigger fallback to user defaults
        
        # Get all sessions for this subject to populate the 'same_site' dropdown
        same_site_query = session.SessionSameSite.proj("session_id") * session.Session.proj("subject") & f'subject = "{subject_id}"'
        existing_sessions = same_site_query.fetch("session_id")
        
        # Enhance same_site_options: combine database sessions with current batch sessions
        same_site_options = list(existing_sessions) if len(existing_sessions) > 0 else []
        
        # Add sessions from current batch for the same subject
        if subject_session_ids:
            for sess_id in subject_session_ids:
                if sess_id not in same_site_options:
                    same_site_options.append(sess_id)
        
        # Always include the current session_id in the options if not already present
        if session_id not in same_site_options:
            same_site_options.append(session_id)
        
        # Sort the options for better user experience
        same_site_options.sort()
        db_info['same_site_options'] = same_site_options

    except Exception as e:
        print(f"Warning: Could not fetch all DB info for {session_id}. Some fields may use defaults. Error: {e}")
        # Add a fallback for dlc_models in case of error
        if 'dlc_models' not in db_info:
            db_info['dlc_models'] = None  # Will trigger fallback to user defaults
        
    return db_info

def select_sessions(
    AvailableSessionDirB, 
    do_population=False, 
    rspace_upload=False, 
    ingest_opt='trigger',
    suppress_errors=True,
    preflight_strict=False
):
    """
    Main function to display a UI for selecting and configuring sessions for ingestion and processing.
    
    Parameters:
    - AvailableSessionDirB: List of session directory paths available for processing
    - do_population: Boolean flag for population processing options
    - rspace_upload: Boolean flag to enable RSpace document uploads
    - ingest_opt: String specifying ingestion optimization mode ('trigger', etc.)
    - suppress_errors: If True, continue processing even when errors occur in individual sessions.
                      If False, stop processing on the first error encountered.
    
    Returns:
    - get_selected_data: A callable function that when invoked returns a dictionary containing 
                        the currently selected session configurations from the UI, including 
                        DataJoint scan keys. Call this function to retrieve user selections 
                        after they have made their choices.
                        
    Example:
        # Create the UI and get the selection function
        get_selections = select_sessions(session_dirs)
        
        # After user makes selections in the UI, get the selected data
        selected_data = get_selections()
        
        # selected_data will be a dict with session_id keys and configuration values
        # Each entry includes 'scan_key' with DataJoint format: {'session_id': ..., 'scan_id': ...}
    """
    # --- 1. Create persistent UI elements that are not redrawn on refresh ---
    all_widgets = []  # This list will be populated by the refresh function
    
    output_widget = widgets.Output(layout=widgets.Layout(
        max_height='500px', 
        overflow='auto', 
        border='1px solid #ccc', 
        padding='10px'
    ))

    # --- Progress Bar & Status (above log window) ---
    progress_bar = widgets.FloatProgress(
        value=0.0, min=0.0, max=1.0, 
        bar_style='info', 
        orientation='horizontal',
        layout=widgets.Layout(width='99%', height='25px')
    )
    progress_status_label = widgets.Label(value="Waiting to start...")
    timer_label = widgets.HTML(value="Elapsed time: 00:00", layout=widgets.Layout(margin='0 0 0 20px'))
    status_box = widgets.HBox([progress_status_label, timer_label], layout=widgets.Layout(
        justify_content='space-between',
        align_items='center',
        width='100%'
    ))
    
    # Progress container (above log window)
    progress_container = widgets.VBox(
        [status_box, progress_bar], 
        layout=widgets.Layout(
            visibility='hidden', # Initially hidden
            margin='15px 0 5px 0',
            padding='10px',
            border='1px solid #e0e0e0',
            border_radius='5px',
            background_color='#f9f9f9'
        )
    )

    # --- Spinner (below log window) ---
    spinner_html = widgets.HTML('''
        <div style="display: flex; flex-direction: column; align-items: center;">
            <style>
                @keyframes spin {
                  0% { transform: rotate(0deg); }
                  100% { transform: rotate(360deg); }
                }
                .loader {
                  border: 8px solid #f3f3f3; /* Light grey */
                  border-top: 8px solid #28a745; /* Green to match button */
                  border-radius: 50%;
                  width: 40px;
                  height: 40px;
                  animation: spin 1s linear infinite;
                }
            </style>
            <div class="loader"></div>
        </div>
    ''')
    spinner_container = widgets.VBox(
        [spinner_html], 
        layout=widgets.Layout(
            visibility='hidden', # Initially hidden
            margin='5px 0 15px 0',
            padding='10px',
            border='1px solid #e0e0e0',
            border_radius='5px',
            background_color='#f9f9f9'
        )
    )

    # --- Success indicator ---
    success_html = widgets.HTML('''
        <div style="transform: scale(0.7); margin-top: -20px; margin-bottom: -20px;">
            <style>
                .success-checkmark { width: 80px; height: 115px; margin: 0 auto; }
                .success-checkmark .check-icon { width: 80px; height: 80px; position: relative; border-radius: 50%; box-sizing: content-box; border: 4px solid #4CAF50; }
                .success-checkmark .check-icon::before { top: 3px; left: -2px; width: 30px; height: 10px; border-radius: 50%; position: absolute; background-color: #ffffff; transform: rotate(45deg); content: ''; }
                .success-checkmark .check-icon::after { top: 23px; left: 30px; width: 15px; height: 10px; border-radius: 50%; position: absolute; background-color: #ffffff; transform: rotate(-45deg); content: ''; }
                .success-checkmark .check-icon .icon-line { height: 5px; background-color: #4CAF50; display: block; border-radius: 2px; position: absolute; z-index: 10; }
                .success-checkmark .check-icon .icon-line.line-tip { top: 46px; left: 14px; width: 25px; transform: rotate(45deg); animation: icon-line-tip 0.75s; }
                .success-checkmark .check-icon .icon-line.line-long { top: 38px; right: 8px; width: 47px; transform: rotate(-45deg); animation: icon-line-long 0.75s; }
                .success-checkmark .check-icon .icon-circle { top: -4px; left: -4px; z-index: 10; width: 80px; height: 80px; border-radius: 50%; position: absolute; box-sizing: content-box; border: 4px solid rgba(76, 175, 80, .5); }
                .success-checkmark .check-icon .icon-fix { top: 8px; width: 5px; left: 26px; z-index: 1; height: 85px; position: absolute; transform: rotate(-45deg); background-color: #ffffff; }
                @keyframes icon-line-tip { 0% { width: 0; left: 1px; top: 19px; } 54% { width: 0; left: 1px; top: 19px; } 70% { width: 50px; left: -8px; top: 37px; } 84% { width: 17px; left: 21px; top: 48px; } 100% { width: 25px; left: 14px; top: 45px; } }
                @keyframes icon-line-long { 0% { width: 0; right: 46px; top: 54px; } 65% { width: 0; right: 46px; top: 54px; } 84% { width: 55px; right: 0px; top: 35px; } 100% { width: 47px; right: 8px; top: 38px; } }
            </style>
            <div class="success-checkmark">
                <div class="check-icon">
                    <span class="icon-line line-tip"></span>
                    <span class="icon-line line-long"></span>
                    <div class="icon-circle"></div>
                    <div class="icon-fix"></div>
                </div>
            </div>
            <p style="text-align: center; font-family: sans-serif; margin-top: 10px; color: #28a745; font-size: 18px;">*GREAT SUCCESS*</p>
        </div>
    ''')
    success_container = widgets.VBox([success_html], layout=widgets.Layout(
        align_items='center',
        margin='15px 0 0 0',
        visibility='hidden'  # Initially hidden
    ))

    # --- Buttons ---
    commit_button = widgets.Button(
        description='Commit Selections', 
        icon='check-square-o', 
        button_style='success', 
        layout=widgets.Layout(width='auto', margin='25px 0 10px 0')
    )
    refresh_button = widgets.Button(
        description='Refresh GUI',
        icon='refresh',
        button_style='info',
        layout=widgets.Layout(width='auto', margin='25px 10px 10px 0')
    )
    select_all_button = widgets.Button(
        description='All',
        icon='check-square',
        button_style='warning',
        layout=widgets.Layout(width='50px', height='25px', margin='2px 0 0 0')
    )
    
    dlc_crop_select_all_button = widgets.Button(
        description='All',
        icon='check-square',
        button_style='info',
        layout=widgets.Layout(width='50px', height='25px', margin='2px 0 0 0')
    )

    # --- Custom CSS for buttons and row styling ---
    button_style_html = widgets.HTML("""
        <style>
            .jupyter-button.btn-success {
                font-size: 16px !important;
                padding: 10px 15px !important;
                border-radius: 5px !important;
            }
            .jupyter-button.btn-info {
                font-size: 16px !important;
                padding: 10px 15px !important;
                border-radius: 5px !important;
            }
            .jupyter-button.btn-warning {
                font-size: 12px !important;
                padding: 4px 8px !important;
                border-radius: 3px !important;
            }
            /* Force alternating row colors */
            .widget-hbox:nth-child(even) {
                background-color: #fafafa !important;
            }
            .widget-hbox:nth-child(odd) {
                background-color: #ffffff !important;
            }
            /* Override any theme styling */
            .widget-hbox {
                background-color: inherit !important;
                border: none !important;
            }
            /* Ensure consistent text colors */
            .widget-label, .widget-text, .widget-dropdown select, .widget-html {
                color: #333333 !important;
                background-color: inherit !important;
            }
            /* Style dropdown elements */
            .widget-dropdown select {
                background-color: #ffffff !important;
                border: 1px solid #d0d0d0 !important;
                color: #333333 !important;
            }
            /* Checkbox styling */
            .widget-checkbox input[type="checkbox"] {
                background-color: #ffffff !important;
            }
            /* Text input styling */
            .widget-text input {
                background-color: #ffffff !important;
                border: 1px solid #d0d0d0 !important;
                color: #333333 !important;
            }
        </style>
    """)
    
    # --- Placeholder to hold the UI, allowing it to be completely replaced ---
    ui_placeholder = widgets.VBox()

    # This will hold the latest lookup data for the commit callback
    latest_lookup = {}

    def refresh_ui(b=None):
        """Function to clear and redraw the entire UI."""
        # --- 1. Fetch data for UI options from the database ---
        
        # Fetch projects (note: Project table only has 'project' column, no project_id)
        project_data = project.fetch('project', as_dict=True)
        project_options = [p['project'] for p in project_data]  # Just use project names
        project_names_only = [p['project'] for p in project_data]  # Keep original names for backend compatibility
        
        lookup = {
            'projects': project_names_only,  # Backend compatibility
            'project_options': project_options,  # Display options (same as projects since no IDs)
            'equipments': equipment.Equipment().fetch('scanner'),
            'locations': surgery.AnatomicalLocation().fetch('anatomical_location'),
            's2p_parms': imaging.ProcessingParamSet.fetch("paramset_idx", "paramset_desc"),
            'dlc_models': np.insert(model.Model.fetch("model_name"), 0, 'dummy'), # Add 'dummy' option
            'ingested_sessions': get_session_dir_key_from_dir(session.SessionDirectory.fetch('session_dir'))
        }
        latest_lookup.update(lookup) # Update the shared lookup data
        user_defaults = get_user_defaults(AvailableSessionDirB)
        user_recording_notes_defaults = get_user_recording_notes_defaults(AvailableSessionDirB)
        user_dlc_cropping_defaults = get_user_dlc_cropping_defaults(AvailableSessionDirB)
        user_cam_defaults = get_user_cam_defaults(AvailableSessionDirB)  # Get camera defaults for headers
        
        # Get camera names for headers from the first session's user
        # This ensures headers show the actual camera names for consistency
        first_session_cameras = ['Multi-Select', 'Multi-Select', 'Multi-Select']
        if len(user_cam_defaults) > 0:
            first_session_cameras = user_cam_defaults[0]
        if len(user_cam_defaults) > 0:
            first_session_cameras = user_cam_defaults[0]
        
        # Pre-collect all session information to group by subject for same_site options
        session_subject_map = {}
        for session_path in AvailableSessionDirB:
            session_id = get_session_key_from_dir([session_path])[0]
            subject_id = get_subject_key_from_dir([session_path])[0]
            if subject_id not in session_subject_map:
                session_subject_map[subject_id] = []
            session_subject_map[subject_id].append(session_id)
        
        # Also add existing sessions from database for each subject
        for subject_id in session_subject_map.keys():
            try:
                # Fetch all existing sessions for this subject from the database
                existing_sessions = (session.Session & f'subject = "{subject_id}"').fetch('session_id')
                if DEBUG_DLC_PRESELECTION:
                    print(f"   🔍 Found {len(existing_sessions)} existing database sessions for subject {subject_id}")
                for existing_session in existing_sessions:
                    if existing_session not in session_subject_map[subject_id]:
                        session_subject_map[subject_id].append(existing_session)
                
                # Sort the session list for better organization
                session_subject_map[subject_id] = sorted(session_subject_map[subject_id])
                if DEBUG_DLC_PRESELECTION:
                    print(f"   📋 Complete session list for {subject_id}: {session_subject_map[subject_id]}")
            except Exception as e:
                # If database query fails, just use the current batch sessions
                print(f"⚠️ Could not fetch existing sessions for {subject_id}: {e}")
                pass
        
        # --- 2. Create widgets for each session ---
        all_widgets.clear()
        widget_rows = []
        for i, session_path in enumerate(AvailableSessionDirB):
            session_id = get_session_key_from_dir([session_path])[0]
            subject_id = get_subject_key_from_dir([session_path])[0]
            
            db_info = _get_db_info(session_id, subject_id, session_path, session_subject_map.get(subject_id, []), 
                                 user_cam_defaults[0] if len(user_cam_defaults) > 0 else None)
            
            # Build DLC defaults for each widget - handle both single indices and lists for multiple selection
            def build_dlc_models(user_default_idx, dlc_options):
                """Build DLC model list from user defaults, handling both single indices and lists"""
                if isinstance(user_default_idx, list):
                    # Multiple selection default - build list from multiple indices
                    models = []
                    for idx in user_default_idx:
                        if idx < len(dlc_options) and dlc_options[idx] != 'dummy':
                            models.append(dlc_options[idx])
                    return models
                else:
                    # Single selection default - wrap in list for consistency
                    if user_default_idx < len(dlc_options) and dlc_options[user_default_idx] != 'dummy':
                        return [dlc_options[user_default_idx]]
                    return []
            
            # PRIORITIZE PoseEstimationTaskNew data over user defaults for already ingested sessions
            if db_info.get('dlc_models') is not None:
                # Use existing database models for each widget (preselection from PoseEstimationTaskNew)
                dlc_model1 = db_info['dlc_models'][0] if len(db_info['dlc_models']) > 0 else []
                dlc_model2 = db_info['dlc_models'][1] if len(db_info['dlc_models']) > 1 else []
                dlc_model3 = db_info['dlc_models'][2] if len(db_info['dlc_models']) > 2 else []
                if DEBUG_DLC_PRESELECTION:
                    print(f"   📊 Using PoseEstimationTaskNew models for {session_id}: {[len(dlc_model1), len(dlc_model2), len(dlc_model3)]} models per camera")
            else:
                # Use user defaults for new sessions only (when no PoseEstimationTaskNew data exists)
                dlc_model1 = build_dlc_models(user_defaults[i][4], lookup['dlc_models'])
                dlc_model2 = build_dlc_models(user_defaults[i][5], lookup['dlc_models'])
                dlc_model3 = build_dlc_models(user_defaults[i][6], lookup['dlc_models'])
                if DEBUG_DLC_PRESELECTION:
                    print(f"   🎯 Using user defaults for {session_id}: no PoseEstimationTaskNew data found")
            
            defaults = {
                'project': lookup['projects'][user_defaults[i][0]],
                'location': lookup['locations'][user_defaults[i][1]],
                'equipment': lookup['equipments'][user_defaults[i][2]],
                's2p_param_idx': user_defaults[i][3],
                's2p_options': lookup['s2p_parms'],
                'same_site': session_id,
                'dlc_models': [dlc_model1, dlc_model2, dlc_model3],
                'dlc_options': lookup['dlc_models'],
                'project_options': lookup['project_options'],  # Use project names
                'project_names_lookup': lookup['projects'],  # Keep for value mapping
                'location_options': lookup['locations'],
                'equipment_options': lookup['equipments'],
                'recording_notes_options': user_recording_notes_defaults[i],
                'use_dlc_cropping': user_dlc_cropping_defaults[i],
            }
            
            widgets_for_row = _create_session_widgets(db_info, defaults)
            all_widgets.append(widgets_for_row)
            
            label_text = ('<b style="color: #007A00; font-size: 14px;">* </b>' if not db_info['is_ingested'] else '') + f'<span style="font-family: monospace;">{session_path}</span>'
            
            # Create styled row with explicit background color
            row_bg_color = '#fafafa' if i % 2 == 0 else '#ffffff'
            
            # Wrap the row in an HTML container with background color
            styled_row = widgets.HTML(
                value=f'<div style="background-color: {row_bg_color}; padding: 10px 8px; border-bottom: 1px solid #e0e0e0; margin: 0;"></div>',
                layout=widgets.Layout(width='100%', height='0px')
            )
            
            row = widgets.HBox([
                widgets.HTML(value=label_text, layout=widgets.Layout(width='400px')),
                widgets_for_row['project_dropdown'],
                widgets_for_row['location_dropdown'],
                widgets_for_row['equipment_dropdown'],
                widgets_for_row['s2p_dropdown'],
                widgets_for_row['same_site_dropdown'],
                widgets_for_row['dlc1_multi'],
                widgets_for_row['dlc2_multi'],
                widgets_for_row['dlc3_multi'],
                widgets_for_row['note_textbox'],
                widgets_for_row['recording_notes_dropdown'],
                widgets_for_row['dlc_cropping_checkbox'],
                widgets_for_row['run_checkbox'],
                widgets_for_row['curated_checkbox']
            ], layout=widgets.Layout(
                background_color=row_bg_color,
                border='bottom solid 1px #e0e0e0', 
                padding='10px 8px', 
                align_items='center'
            ))
            
            # Create a VBox to combine the background div and the actual row
            combined_row = widgets.VBox([
                widgets.HTML(
                    value=f'<div style="background-color: {row_bg_color}; width: 100%; height: 100%; position: absolute; top: 0; left: 0; z-index: -1;"></div>',
                    layout=widgets.Layout(width='100%', height='0px')
                ),
                row
            ], layout=widgets.Layout(
                background_color=row_bg_color,
                position='relative',
                margin='0',
                padding='0'
            ))
            
            widget_rows.append(row)

        # --- 3. Assemble UI ---
        header_image_widget = None
        header_image_path = 'images/header_image.jpg'
        if os.path.exists(header_image_path):
            with open(header_image_path, "rb") as f:
                image_data = f.read()
            header_image_widget = widgets.Image(value=image_data, format='jpg', width='50%')

        title = widgets.HTML("<h1>ADAMACS Ingestion Console v2.0</h1>", layout=widgets.Layout(margin='0 0 15px 0'))

        header_style = {'font_weight': 'bold', 'color': '#2c3e50'}
        header_style = {'font_weight': 'bold'}
        headers = [
            widgets.HTML(value='<b style="color: #2c3e50;">Scan Header</b>', layout=widgets.Layout(width='400px')),
            widgets.HTML(value='<b style="color: #2c3e50;">Project</b>', layout=widgets.Layout(width='150px')),
            widgets.HTML(value='<b style="color: #2c3e50;">Recording Location</b>', layout=widgets.Layout(width='150px')),
            widgets.HTML(value='<b style="color: #2c3e50;">Recording Setup</b>', layout=widgets.Layout(width='150px')),
            widgets.HTML(value='<b style="color: #2c3e50;">Suite2p Params</b>', layout=widgets.Layout(width='200px')),
            widgets.HTML(value='<b style="color: #2c3e50;">Same Site ID</b>', layout=widgets.Layout(width='150px')),
            widgets.HTML(value=f'<b style="color: #2c3e50;">DLC1 ({first_session_cameras[0]})</b>', layout=widgets.Layout(width='200px')),
            widgets.HTML(value=f'<b style="color: #2c3e50;">DLC2 ({first_session_cameras[1]})</b>', layout=widgets.Layout(width='200px')),
            widgets.HTML(value=f'<b style="color: #2c3e50;">DLC3 ({first_session_cameras[2]})</b>', layout=widgets.Layout(width='200px')),
            widgets.HTML(value='<b style="color: #2c3e50;">Session Comment</b>', layout=widgets.Layout(width='200px')),
            widgets.HTML(value='<b style="color: #2c3e50;">Behavior Stage</b>', layout=widgets.Layout(width='200px')),
            widgets.VBox([
                widgets.HTML(value='<b style="color: #2c3e50;">DLC Crop</b>', layout=widgets.Layout(width='70px', justify_content='center')),
                dlc_crop_select_all_button
            ], layout=widgets.Layout(width='80px', align_items='center')),
            widgets.VBox([
                widgets.HTML(value='<b style="color: #2c3e50;">Commit?</b>', layout=widgets.Layout(width='70px', justify_content='center')),
                select_all_button
            ], layout=widgets.Layout(width='80px', align_items='center')),
            widgets.HTML(value='<b style="color: #2c3e50;">Curated?</b>', layout=widgets.Layout(width='70px', justify_content='center'))
        ]
        header_box = widgets.HBox(headers, layout=widgets.Layout(
            border='bottom solid 2px #c0c0c0', 
            padding='12px 8px', 
            margin='0 0 8px 0',
            background_color='#e8f0f8'
        ))

        # Apply direct styling to each row with forced background colors
        for i, row in enumerate(widget_rows):
            row_bg_color = '#fafafa' if i % 2 == 0 else '#ffffff'
            row.layout = widgets.Layout(
                background_color=row_bg_color,
                border='bottom solid 1px #e0e0e0', 
                padding='10px 8px', 
                align_items='center'
            )
        
        buttons_box = widgets.HBox([commit_button, refresh_button])

        ui_elements = []
        if header_image_widget:
            ui_elements.append(widgets.HBox([header_image_widget], layout=widgets.Layout(justify_content='flex-start')))
        
        # Assemble UI: progress bar above log, spinner below log
        ui_elements.extend([title, header_box, *widget_rows, buttons_box, progress_container, output_widget, spinner_container, success_container, button_style_html])
        
        ui_container = widgets.VBox(
            ui_elements,
            layout=widgets.Layout(
                padding='20px', border='1px solid #e0e0e0', border_radius='8px', background_color='#ffffff'
            )
        )
        ui_placeholder.children = [ui_container]

    def toggle_all_checkboxes(b):
        """Toggle all commit checkboxes between selected and deselected."""
        if not all_widgets:
            return
        
        # Check if all are currently selected
        all_selected = all(widget['run_checkbox'].value for widget in all_widgets)
        
        # If all are selected, deselect all. Otherwise, select all.
        new_value = not all_selected
        
        for widget in all_widgets:
            widget['run_checkbox'].value = new_value
        
        # Update button text based on new state
        select_all_button.description = 'None' if new_value else 'All'
        select_all_button.icon = 'square' if new_value else 'check-square'

    def toggle_all_dlc_crop_checkboxes(button):
        """Toggle all DLC crop checkboxes between checked and unchecked."""
        all_selected = all(widget['dlc_cropping_checkbox'].value for widget in all_widgets)
        new_value = not all_selected
        
        for widget in all_widgets:
            widget['dlc_cropping_checkbox'].value = new_value
        
        # Update button text based on new state
        dlc_crop_select_all_button.description = 'None' if new_value else 'All'
        dlc_crop_select_all_button.icon = 'square' if new_value else 'check-square'

    # --- 4. Final setup ---
    # Define the commit callback here, outside refresh_ui, so it's only created once.
    def on_commit(b):
        # It uses `latest_lookup` which is updated by `refresh_ui`.
        _commit_button_callback(
            b, AvailableSessionDirB, all_widgets, latest_lookup['s2p_parms'], 
            output_widget, progress_container, spinner_container, progress_bar, progress_status_label, timer_label, success_container, 
            do_population, rspace_upload, ingest_opt, suppress_errors, preflight_strict
        )
        # The UI is NOT refreshed automatically. The user must click the "Refresh GUI" button.

    # Attach the callbacks ONCE.
    refresh_button.on_click(refresh_ui)
    commit_button.on_click(on_commit)
    select_all_button.on_click(toggle_all_checkboxes)
    dlc_crop_select_all_button.on_click(toggle_all_dlc_crop_checkboxes)
    
    # Initial drawing of the UI
    refresh_ui()

    # Display the placeholder that holds the UI
    display(ui_placeholder)
    
    # Return a function that can be called to get the current selections
    def get_selected_data():
        """
        Extract current selections from the UI widgets.
        
        Returns:
            dict: Dictionary containing selected session configurations with DataJoint scan keys
                  and DLC model indices
        """
        if not all_widgets:
            return {}
            
        selected_data = {}
        for i, (session_path, widgets_dict) in enumerate(zip(AvailableSessionDirB, all_widgets)):
            session_id = get_session_key_from_dir([session_path])[0]
            subject_id = get_subject_key_from_dir([session_path])[0]
            scan_id = get_scan_key_from_dir([session_path])[0]
            
            # Only include sessions that are selected for processing
            if widgets_dict['run_checkbox'].value:
                # Create DataJoint scan key (combination of session_id and scan_id)
                scan_key = {'session_id': session_id, 'scan_id': scan_id}
                
                # Get selected DLC models and their indices
                dlc1_models = list(widgets_dict['dlc1_multi'].value)
                dlc2_models = list(widgets_dict['dlc2_multi'].value)
                dlc3_models = list(widgets_dict['dlc3_multi'].value)
                
                # Calculate indices for selected models
                dlc1_indices = [latest_lookup['dlc_models'].tolist().index(model) for model in dlc1_models if model in latest_lookup['dlc_models']]
                dlc2_indices = [latest_lookup['dlc_models'].tolist().index(model) for model in dlc2_models if model in latest_lookup['dlc_models']]
                dlc3_indices = [latest_lookup['dlc_models'].tolist().index(model) for model in dlc3_models if model in latest_lookup['dlc_models']]
                
                # Extract project name from display format "Project Name (ID: X)"
                # Get project name directly (no ID format needed)
                project_name = widgets_dict['project_dropdown'].value
                
                selected_data[session_id] = {
                    'session_path': session_path,
                    'subject_id': subject_id,
                    'scan_key': scan_key,
                    'scan_id': scan_id,
                    'project': project_name,  # Use extracted project name
                    'location': widgets_dict['location_dropdown'].value,
                    'equipment': widgets_dict['equipment_dropdown'].value,
                    's2p_params': widgets_dict['s2p_dropdown'].value,
                    'same_site': widgets_dict['same_site_dropdown'].value,
                    'dlc_models': [dlc1_models, dlc2_models, dlc3_models],
                    'dlc_model_indices': [dlc1_indices, dlc2_indices, dlc3_indices],
                    'session_note': widgets_dict['note_textbox'].value,
                    'recording_notes': widgets_dict['recording_notes_dropdown'].value,
                    'is_curated': widgets_dict['curated_checkbox'].value,
                    'is_selected': widgets_dict['run_checkbox'].value
                }
        
        return selected_data
    
    def get_available_dlc_models():
        """
        Get all available DLC models with their indices.
        
        Returns:
            dict: Dictionary containing model information with indices
        """
        if 'dlc_models' not in latest_lookup:
            return {'error': 'DLC models not loaded. Please run the GUI first.'}
        
        models_info = {
            'models_list': latest_lookup['dlc_models'].tolist(),
            'models_with_indices': [(i, model) for i, model in enumerate(latest_lookup['dlc_models'])],
            'total_count': len(latest_lookup['dlc_models'])
        }
        
        return models_info
    
    # Return both functions so users can call them
    return get_selected_data, get_available_dlc_models

# =============================================================================
# ------------------------- RSPACE & PLOTTING ---------------------------------
# =============================================================================
# Note: These functions depend on a globally available 'api' object for RSpace,
# which is not defined in this script. They are kept for functional completeness
# but will not execute without proper RSpace API initialization.

def make_rspace_session_document(userID, animalID, date, sessionID):
    """
    Function to create an RSpace document for the current session
    """
    if not api:
        print("RSpace integration is disabled. Cannot create session document.")
        return None, None

    # find 'Experiments' folder and get ID
    folders = api.list_folder_tree()
    experiments_ids = []
    for record in folders['records']:
        if record['name'] == 'Experiments':
            experiments_ids.append(record['id'])

    # find folders under 'Experiments' folder
    names_and_ids = []
    subfolders = api.list_folder_tree(experiments_ids[0])
    names_and_ids.append([(record['name'], record['id']) for record in subfolders['records']])

    # find matches with animal IDs
    match_found = any(animalID in item[0] for sublist in names_and_ids for item in sublist)

    # if found, find folder id matching that animal
    if match_found:
        save_folder_id = [x[1] for sublist in names_and_ids for x in sublist if animalID in x[0]][0]
    else:
        # if not generate the directory
        new_folder = api.create_folder(name=f"{userID}_{animalID}", parent_folder_id=experiments_ids[0], notebook=True)
        save_folder_id = new_folder['id']


    # find matches with document IDs

    # find all documents in the animal folder
    all_animal_documents = api.list_folder_tree(save_folder_id)
    doc_names_and_ids = []
    doc_names_and_ids.append([(record['name'], record['id']) for record in all_animal_documents['records']])
    # find matches with sessionID
    match_found = any(sessionID in item[0] for sublist in doc_names_and_ids for item in sublist)

    if match_found:
        ids = [id for string, id in doc_names_and_ids[0] if sessionID in string]
        new_doc = {"id": ids[0]}
    else:
        new_doc = api.create_document(name=f"{date}_{sessionID}", parent_folder_id=save_folder_id)


    # create RSpace document
    fetchtable = session.Session() * scan.ScanPath() * session.SessionUser() * project.Project * session.SessionNote() * session.SessionSameSite() * scan.Scan() *scan.ScanInfo() & f'session_id = "{sessionID}"'
    df = fetchtable.fetch(format='frame')
    html_table = df.to_html()
    content = html_table
    api.append_content(new_doc['id'], content)

    # ingest document in rspace table
    session.SessionRspace.insert1({'session_id': sessionID, 'rspace_id': new_doc['id']}, skip_duplicates=True)
    # ingest animal folder id in subject Rspace table
    subject.SubjectRspace.insert1({'subject': animalID, 'rspace_subject_id': save_folder_id}, skip_duplicates=True)

def make_overview_figures_rspace(curation_key, rspace_id):
    # Figure Style settings for notebook.
    mpl.rcParams.update({
        'axes.spines.left': False,
        'axes.spines.bottom': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'figure.subplot.wspace': .01,
        'figure.subplot.hspace': .01,
        'figure.figsize': (18, 13),
        'ytick.major.left': False,
        'xtick.major.bottom': False
    })
    # Create colormap for traces
    jet = cm.get_cmap('jet')
    jet.set_bad(color='k')

    ref_image = (imaging.MotionCorrection.Summary & curation_key & 'field_idx=0').fetch1('ref_image')
    average_image = (imaging.MotionCorrection.Summary & curation_key & 'field_idx=0').fetch1('average_image')
    correlation_image = (imaging.MotionCorrection.Summary & curation_key & 'field_idx=0').fetch1('correlation_image')
    max_proj_image = (imaging.MotionCorrection.Summary & curation_key & 'field_idx=0').fetch1('max_proj_image')
    
    plt.ioff()
    plt.subplot(1, 4, 1)
    plt.imshow(ref_image, cmap='gray', )
    plt.title("Reference Image for Registration");

    plt.subplot(1, 4, 2)
    plt.imshow(average_image, cmap='gray')
    plt.title("Registered Image, Mean Projection");

    plt.subplot(1, 4, 3)
    plt.imshow(max_proj_image, cmap='gray')
    plt.title("Registered Image, Max Projection")

    plt.subplot(1, 4, 4)
    plt.imshow(correlation_image, cmap='gray')
    plt.title("Registered Image, Correlation Map")

    tmpdir = dj.config['custom'].get('suite2p_fast_tmp')[0]
    session_id = curation_key['session_id']
    scan_id = curation_key['scan_id']
    save_path = os.path.join(tmpdir, f'{session_id}_{scan_id}_templates.png')
    plt.savefig(save_path)
    plt.clf()

    print('bypassing upload')
    # append_rspace_session_image(rspace_id, save_path, captionimage='Suite2p Templates ' + session_id)   
    plt.ion()

def make_overview_movies_rspace(curation_key, rspace_id):
    path = (scan.ScanPath & curation_key).fetch1("path") + ("/suite2p/plane0/reg_tif")

    # Get a list of all tiff files in the folder
    tiff_files = [os.path.join(path, f) for f in natsorted(os.listdir(path)) if f.endswith('.tif')]

    # Load each tiff stack into a list of numpy arrays
    stacks = []
    for f in tiff_files:
        with tifffile.TiffFile(f) as tif:
            num_pages = len(tif.pages)
            stack = np.zeros((num_pages,) + tif.pages[0].shape, dtype=tif.pages[0].dtype)
            for i, page in enumerate(tif.pages):
                stack[i] = page.asarray()
        stacks.append(stack)

    volume = np.concatenate(stacks, axis=0)

    # delete registration tiff
    for f in tiff_files:
        os.remove(f) 
    
    ### moving average filter
    runav = 20
    running_z_projection = sh.rolling_average_filter(volume, runav)

    session_id = curation_key['session_id']
    scan_id = curation_key['scan_id']

    filename = os.path.join(path, 'registered_movie_' + session_id + '_' + scan_id + '_' + str(runav) + '_frame_runningaverage2' + '.mp4')

    fps = 120
    p1 = 2
    p2 = 99.998

    rescaled_image_8bit = sh.make_stack_movie(running_z_projection, filename, fps, p1, p2)

    tmpdir = dj.config['custom'].get('suite2p_fast_tmp')[0]

    # append_rspace_session_image(rspace_id, filename, captionimage='registered movie ' + scan_id)   

def append_rspace_session_image(rspace_id, fig, captionimage):
    """
    Function to append an image to the RSpace document
    """
    if not api:
        print("RSpace integration is disabled. Cannot append image.")
        return

    with open(fig, 'rb') as f:
        uploaded_image = api.upload_file(f, caption=captionimage)
        
        content = f"""
        <p>{captionimage}</p>
        <fileId={uploaded_image['id']}>
        """
        api.append_content(rspace_id, content)
        
        print(f"uploaded image id = {uploaded_image['id']}")
