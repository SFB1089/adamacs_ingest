import numpy as np
import datajoint as dj
import scipy.io as spio
from pathlib import Path
from bisect import bisect
from dateutil import parser
from element_interface.utils import find_full_path, find_root_directory
from ..pipeline import subject, session, trial, event, scan
from ..paths import get_experiment_root_data_dir
from .aux import Auxfile
from adamacs.ingest import aux

RAW_BPod_EVENT_PREFIX = "raw_bpod_"


def _coerce_time(value):
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iter_event_times(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        values = value.flatten().tolist()
    elif isinstance(value, (list, tuple)):
        values = []
        for item in value:
            if isinstance(item, (list, tuple, np.ndarray)):
                values.extend(_iter_event_times(item))
            else:
                values.append(item)
    else:
        values = [value]
    times = []
    for item in values:
        time_value = _coerce_time(item)
        if time_value is not None:
            times.append(time_value)
    return times


def _collect_state_pairs(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if value and all(not isinstance(item, (list, tuple)) for item in value):
            if len(value) >= 2:
                if len(value) % 2 == 0:
                    return [value[i:i + 2] for i in range(0, len(value), 2)]
                return [value[:2]]
            return []
        pairs = []
        for item in value:
            pairs.extend(_collect_state_pairs(item))
        return pairs
    return []


def _iter_state_intervals(value):
    pairs = _collect_state_pairs(value)
    intervals = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        entry = _coerce_time(pair[0])
        exit_time = _coerce_time(pair[1])
        intervals.append((entry, exit_time))
    return intervals


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.flatten().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


_BPOD_SETUP_ALIASES = {
    "openfield": {"openfield", "mini2p1_openfield"},
    "bench2p_oddball": {"bench2p_oddball"},
    "bench2p_Oddball_V2": {"bench2p_oddball_v2", "bench2p_Oddball_v2", "bench2p_Oddball_V2"},
    "bench2p_lineartrack": {"bench2p_lineartrack"},
    "behavior_box": {"behavior_box"},
}

_BPOD_SETUP_METHODS = {
    "openfield": "ingest",
    "bench2p_oddball": "ingest_oddball",
    "bench2p_Oddball_V2": "ingest_oddball_v2",
    "bench2p_lineartrack": "ingest_linear_track",
    "behavior_box": "ingest_behavior_box",
}


def _normalize_bpod_setup_type(aux_setup_type):
    if not aux_setup_type:
        return "openfield"
    normalized = str(aux_setup_type).strip()
    lower = normalized.lower()
    if "openfield" in lower:
        return "openfield"
    for canonical, aliases in _BPOD_SETUP_ALIASES.items():
        if lower in {alias.lower() for alias in aliases}:
            return canonical
    return normalized


def _attr_present(value):
    return value is not None and (
        not hasattr(value, '__len__') or len(value) > 0
    )


def _attr_truthy(value):
    return bool(value)


class Bpodfile(object):
    def __init__(self, bpod_path):
        self.bpod_path = Path(bpod_path)
        if not self.bpod_path.exists():
            self._bpod_path_full = find_full_path(
                get_experiment_root_data_dir(), bpod_path
            )
        else:
            self._bpod_path_full = Path(bpod_path)
        self._bpod_path_relative = self._bpod_path_full.relative_to(
            find_root_directory(get_experiment_root_data_dir(), self._bpod_path_full)
        )

        # NOTE: Daniel made a comment that np.squeeze didn't work on singleton
        #       dimensions, returning empty array. Chris couldn't find the same issue
        self._raw_data = spio.loadmat(self._bpod_path_full, simplify_cells=True)

        self._trials = {}  # dict to be loaded with all trials accessed
        self.session_data = self._raw_data["SessionData"]
        self.trial_data = self.session_data["RawEvents"]["Trial"]
        self.n_trials = self.session_data["nTrials"]
        try:
            self.subject_id = self.session_data["CurrentSubjectName"]
        except KeyError:
            self.subject_id = "Testmouse-1" #TR24: in case somebody did not include the Subject in the bpod script
        self.trial_starts = self.session_data["TrialStartTimestamp"] #TR23
        self.start_time = parser.parse(
            self.session_data["Info"]["SessionDate"]  # format: 18-Mar-2022
            + " "
            + self.session_data["Info"]["SessionStartTime_UTC"]  # format: 16:55:28
        )

    @property
    def trials(self):
        """Loads all trials for this file into memory"""
        [self.trial(idx) for idx in range(self.n_trials)]
        return self._trials

    def trial(self, idx):
        """Loads a specific trial into memory using the trial index"""
        if idx not in self._trials:
            self._trials[idx] = Trial(
                idx, self._bpod_path_full, self.session_data, self.trial_data
            )
        return self._trials[idx]
    
    def _get_trials_with_timeout(self):
        # Loop over the range of self.n_trials and collect trialnum where _raw_events.get("SoftCode10") does not retrun a value
        trials_with_timeout = [
            trialnum for trialnum in range(self.n_trials)
            if self.trial(trialnum)._raw_events.get("SoftCode10") is None and self.trial(trialnum)._raw_events.get("SoftCode15") is None
        ]
        return trials_with_timeout

    def ingest_new(self, session_id, scan_id, aux_setup_type="openfield", **kwargs):
        """Dispatch BPOD ingestion based on normalized setup type."""
        normalized = _normalize_bpod_setup_type(aux_setup_type)
        method_name = _BPOD_SETUP_METHODS.get(normalized)
        if not method_name:
            print(f"Unsupported aux_setup_type for BPOD ingest: {aux_setup_type}")
            return None
        return getattr(self, method_name)(session_id, scan_id, **kwargs)

    def _aux_timestamps(self):
        aux_paths = list(self._bpod_path_full.parent.glob("*.h5"))
        assert len(aux_paths) == 1, f"Found more than one Aux h5 file\n{aux_paths}"
        print(aux_paths[0])
        aux = Auxfile(aux_paths[0]) #TR23: The fact that we read in the aux file again is very redundant. We should read it in once and pass it to the Bpodfile class
        # aux_onset = aux.main_track_gate  # master trigger
        aux_trials = aux.bpod_channels["trial"]  # - aux_onset  # trial times wrt trigger - sweep]["analogScans"][1]
                                                 # corresponds to  bpod_trial_vis_chan = curr_aux[sweep]['analogScans'][1]
        aux_rewards = aux.bpod_channels["reward"] # - aux_onset  # rewards wrt trigger - ["analogScans"][2], self._sample_rate

        #TR23: BPod cam start earlier and end later than actual recording. We need to find the first and last BPOD trial that has a valid timestamp
        #Problem: BNClow does not seem to be in all recordings.
        # trials = self.trial_data
        # BNC1Low_events = [(i, trial['Events'].get('BNC1Low')) for i, trial in enumerate(trials) if 'BNC1Low' in trial['Events']] # get all trials that have a BNC1Low event. Returns a list of tuples (trial number, event time)

        # bpod_aux_starttrial = BNC1Low_events[0][0] + 1 # +1 because we want the trial that FOLLOWS the darkframe onset because the darkframe precedes the first recorded AUX trial

        timeouttrials = self._get_trials_with_timeout()
        if timeouttrials:
            print(f"Timeout trials detected [trial]: {timeouttrials} \n Adjusting timestamps by inserting fake aux triggers")
            bpod_to_aux_starttime_offset = self.session_data['TrialStartTimestamp'][0] - (aux_trials[0] - self.trial(0)._states.get("WaitForPosTriggerSoftCode", [None])[1])
            timeout_duration = self.trial(timeouttrials[0])._states.get("WaitForPosTriggerSoftCode", [None])[1]

            # TR26: Fix for IndexError when timeout trial indices exceed aux_trials length.
            # Process insertions in REVERSE order so each insert doesn't shift the indices
            # of remaining (lower) insertions. Clip index to len(aux_trials) for out-of-bounds.
            for trial_idx in sorted(timeouttrials, reverse=True):
                fake_trigger = self.session_data['TrialStartTimestamp'][trial_idx] + \
                               timeout_duration - bpod_to_aux_starttime_offset
                insert_idx = min(trial_idx, len(aux_trials))  # Clip to valid range
                aux_trials = np.insert(aux_trials, insert_idx, fake_trigger)

        self.n_trials = min(self.n_trials, len(aux_trials)) #TR23: Set the number of trials to the minimum of the number of AUX trials and the number of BPOD trials
        # self.n_trials = len(aux_trials) #TR23: Set the number of trials to the number of AUX trials

        # assert len(aux_trials) == self.n_trials, (
        #     "Number of trials do not match: "
        #     + f"BPod {self.n_trials} vs. Aux {len(aux_trials)}"
        # )
        return aux_trials, aux_rewards
    

    def _aux_oddball_timestamps(self):
        aux_paths = list(self._bpod_path_full.parent.glob("*.h5"))
        assert len(aux_paths) == 1, f"Found more than one Aux h5 file\n{aux_paths}"
        print(aux_paths[0])
        aux = Auxfile(aux_paths[0]) #TR23: The fact that we read in the aux file again is very redundant. We should read it in once and pass it to the Bpodfile class
        aux_gate = aux.main_track_gate  # get aux gate t
        aux_bpod_visstim = aux.bpod_channels["oddball_visstim"] # - aux_onset  # rewards wrt trigger - ["analogScans"][2], self._sample_rate

        # self.n_trials = min(self.n_trials, len(aux_trials)) #TR23: Set the number of trials to the minimum of the number of AUX trials and the number of BPOD trials
        # self.n_trials = len(aux_trials) #TR23: Set the number of trials to the number of AUX trials

        # assert len(aux_trials) == self.n_trials, (
        #     "Number of trials do not match: "
        #     + f"BPod {self.n_trials} vs. Aux {len(aux_trials)}"
        # )
        return aux_gate, aux_bpod_visstim

    def _aux_lineartrack_timestamps(self):
        aux_paths = list(self._bpod_path_full.parent.glob("*.h5"))
        assert len(aux_paths) == 1, f"Found more than one Aux h5 file\n{aux_paths}"
        print(aux_paths[0])
        aux = Auxfile(aux_paths[0]) #TR23: The fact that we read in the aux file again is very redundant. We should read it in once and pass it to the Bpodfile class
        aux_gate = aux.main_track_gate  # get aux gate t
        aux_bpod_trials = aux.bpod_channels["lineartrack_trial"] # - aux_onset  # rewards wrt trigger - ["analogScans"][2], self._sample_rate
        # aux_bpod_licks = aux.bpod_channels["lineartrack_licks"] # - aux_onset  # rewards wrt trigger - ["analogScans"][2], self._sample_rate
        # aux_bpod_rewardzone = aux.bpod_channels["lineartrack_rewardzone"] # - aux_onset  # rewards wrt trigger - ["analogScans"][2], self._sample_rate

        # self.n_trials = min(self.n_trials, len(aux_trials)) #TR23: Set the number of trials to the minimum of the number of AUX trials and the number of BPOD trials
        # self.n_trials = len(aux_trials) #TR23: Set the number of trials to the number of AUX trials

        # assert len(aux_trials) == self.n_trials, (
        #     "Number of trials do not match: "
        #     + f"BPod {self.n_trials} vs. Aux {len(aux_trials)}"
        # )
        return aux_gate, aux_bpod_trials

    def _raw_trial_payload(self, trial_idx):
        try:
            trial_entry = self.trial_data[trial_idx]
        except Exception:
            return {}, {}
        if isinstance(trial_entry, dict):
            raw_events = trial_entry.get("Events") or {}
            raw_states = trial_entry.get("States") or {}
            return raw_events, raw_states
        return {}, {}

    def _raw_event_keys_for_trial(self, session_id, scan_id, trial_idx, base_time,
                                  include_events, include_states):
        if base_time is None:
            return [], set()
        raw_events, raw_states = self._raw_trial_payload(trial_idx)
        event_keys = []
        event_types = set()

        if include_events:
            for name, times in raw_events.items():
                for event_time in _iter_event_times(times):
                    event_type = f"{RAW_BPod_EVENT_PREFIX}{name}"
                    event_types.add(event_type)
                    event_keys.append({
                        "session_id": session_id,
                        "scan_id": scan_id,
                        "trial_id": trial_idx,
                        "event_type": event_type,
                        "event_start_time": base_time + event_time,
                    })

        if include_states:
            for name, times in raw_states.items():
                for entry_time, exit_time in _iter_state_intervals(times):
                    if entry_time is not None:
                        event_type = f"{RAW_BPod_EVENT_PREFIX}{name}_entry"
                        event_types.add(event_type)
                        event_keys.append({
                            "session_id": session_id,
                            "scan_id": scan_id,
                            "trial_id": trial_idx,
                            "event_type": event_type,
                            "event_start_time": base_time + entry_time,
                        })
                    if exit_time is not None:
                        event_type = f"{RAW_BPod_EVENT_PREFIX}{name}_exit"
                        event_types.add(event_type)
                        event_keys.append({
                            "session_id": session_id,
                            "scan_id": scan_id,
                            "trial_id": trial_idx,
                            "event_type": event_type,
                            "event_start_time": base_time + exit_time,
                        })

        return event_keys, event_types

    def _collect_raw_event_state_names(self):
        event_names = set()
        state_names = set()
        max_trials = self.n_trials
        if hasattr(self.trial_data, "__len__"):
            max_trials = min(self.n_trials, len(self.trial_data))
        for trial_idx in range(max_trials):
            try:
                trial_entry = self.trial_data[trial_idx]
            except Exception:
                continue
            if not isinstance(trial_entry, dict):
                continue
            events = trial_entry.get("Events") or {}
            states = trial_entry.get("States") or {}
            if isinstance(events, dict):
                event_names.update(str(name) for name in events.keys())
            if isinstance(states, dict):
                state_names.update(str(name) for name in states.keys())
        event_names_list = sorted(event_names)
        state_names_list = sorted(state_names)
        softcodes = [name for name in event_names_list if name.startswith("SoftCode")]
        return event_names_list, state_names_list, softcodes

    def _relative_to_root(self, path):
        try:
            root = find_root_directory(get_experiment_root_data_dir(), path)
            return path.relative_to(root)
        except Exception:
            return path

    def _select_protocol_file(self, protocol_name=None, protocol_path=None):
        candidates = sorted(self._bpod_path_full.parent.glob("*.m"))
        if not candidates:
            return None, None

        def match_by_name(name):
            if not name:
                return None
            stem = Path(str(name)).stem.lower()
            for candidate in candidates:
                cand_stem = candidate.stem.lower()
                if stem == cand_stem or stem in cand_stem or cand_stem in stem:
                    return candidate
            return None

        selected = None
        if protocol_path:
            path_candidate = Path(str(protocol_path))
            if path_candidate.suffix == "":
                path_candidate = path_candidate.with_suffix(".m")
            if not path_candidate.is_absolute():
                local = self._bpod_path_full.parent / path_candidate.name
                if local.exists():
                    selected = local
            if selected is None and path_candidate.exists():
                selected = path_candidate
            if selected is None:
                selected = match_by_name(path_candidate.stem)
        if selected is None and protocol_name:
            selected = match_by_name(protocol_name)
        if selected is None and len(candidates) == 1:
            selected = candidates[0]
        if selected is None:
            selected = candidates[0]
        if selected is None:
            return None, None
        return selected, selected.stem

    def summarize_metadata(self, session_id=None, scan_id=None, max_values=6, print_output=True):
        """
        Summarize Bpod TrialSettings metadata and interpret the experiment.
        Returns a dict with summary text and structured metadata.
        """
        from collections import Counter

        def normalize_value(value):
            if value is None:
                return None
            if isinstance(value, np.ndarray):
                if value.shape == ():
                    value = value.item()
                else:
                    value = value.tolist()
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (bool, int, float, str)):
                try:
                    if isinstance(value, float) and np.isnan(value):
                        return None
                except TypeError:
                    pass
                return value
            return str(value)

        def format_values(values):
            if not values:
                return "n/a"
            if len(values) == 1:
                return str(next(iter(values)))
            if len(values) <= max_values:
                return "varies: " + ", ".join(str(v) for v in sorted(values, key=str))
            return f"varies ({len(values)} values)"

        def extract_gui_settings(trial_settings):
            gui_list = []
            for entry in _as_list(trial_settings):
                if isinstance(entry, dict):
                    gui = entry.get("GUI")
                    if gui is None and any(k in entry for k in ("RewardAmount", "VisualStim", "TrialIniTime")):
                        gui = entry
                    if isinstance(gui, dict):
                        gui_list.append(gui)
            return gui_list

        def extract_visualstim_menu(trial_settings):
            for entry in _as_list(trial_settings):
                if not isinstance(entry, dict):
                    continue
                meta = entry.get("GUIMeta") or {}
                if not isinstance(meta, dict):
                    continue
                visual = meta.get("VisualStim")
                if isinstance(visual, dict) and "String" in visual:
                    return [str(v) for v in _as_list(visual["String"])]
            return []

        session_data = self.session_data if isinstance(self.session_data, dict) else {}
        info = session_data.get("Info", {}) if isinstance(session_data, dict) else {}

        n_trials = session_data.get("nTrials")
        protocol_name = info.get("ProtocolName") or info.get("Protocol") or info.get("ProtocolPath")
        subject = session_data.get("CurrentSubjectName")

        trial_settings = session_data.get("TrialSettings")
        gui_list = extract_gui_settings(trial_settings)
        gui_keys = set().union(*(gui.keys() for gui in gui_list)) if gui_list else set()

        trial_type_names = _as_list(session_data.get("TrialTypeNames"))
        trial_types = _as_list(session_data.get("TrialTypes"))

        lines = []
        lines.append("Bpod metadata summary")
        lines.append("=" * 80)
        lines.append(f"File: {self._bpod_path_full}")
        if subject:
            lines.append(f"Subject: {subject}")
        if session_id or scan_id:
            lines.append(f"Session/Scan: {session_id or 'n/a'} / {scan_id or 'n/a'}")
        if protocol_name:
            lines.append(f"Protocol: {protocol_name}")
        if n_trials is not None:
            lines.append(f"Trials: {n_trials}")
        if gui_list:
            lines.append(f"TrialSettings GUI entries: {len(gui_list)}")
        lines.append("-" * 80)

        if trial_type_names:
            lines.append("Trial types (names):")
            if trial_types and len(trial_types) == len(gui_list):
                counts = Counter(trial_types)
                for idx, count in counts.most_common():
                    name = str(idx)
                    try:
                        idx_value = int(idx) - 1
                        if 0 <= idx_value < len(trial_type_names):
                            name = str(trial_type_names[idx_value])
                    except (TypeError, ValueError):
                        pass
                    lines.append(f"  {name}: {count}")
            else:
                lines.append("  " + ", ".join(str(n) for n in trial_type_names))
            lines.append("-" * 80)

        visualstim_menu = extract_visualstim_menu(trial_settings)
        key_groups = {
            "Timing": ["CueDelay", "TrialIniTime", "ResponseTime", "RewardDelay", "PunishDelay", "DrinkingTime"],
            "Reward": ["RewardAmount", "Rebait"],
            "Visual": ["VisualStim", "VisualStimSizemm", "VisualStimRandLoc", "VisualStimRandLocOffset",
                       "VisualStimRandLocMinDist", "VisualStimColorR", "VisualStimColorG", "VisualStimColorB"],
            "VirtualCricket": ["VirtualCricket", "VirtualCricketJumps", "VirtualCricketDistance", "VirtualCricketAngle",
                               "VirtualCricketRandom", "PortCricket", "VirtualCricketStickAngle",
                               "CricketOnStickTime", "ChaseDotTime"],
            "FadingBeacon": ["FadingBeacon", "FadingBeaconIndex", "FadingTime", "FadingContacts", "deadzone",
                             "VisualStimFixLocDeadDist"],
            "Tracking": ["bonsai_track", "hwtrigger"],
            "Sound": ["PlaySound", "PunishSound", "WithdrawalSound", "SoundDuration", "SinWaveFreqTrialIni", "SinWaveFreqReward"],
        }

        gui_summary = {}
        for key in sorted(gui_keys):
            values = set()
            for gui in gui_list:
                if key not in gui:
                    continue
                normalized = normalize_value(gui[key])
                if normalized is not None:
                    values.add(normalized)
            if values:
                gui_summary[key] = values

        for group_name, keys in key_groups.items():
            group_values = {k: gui_summary.get(k) for k in keys if k in gui_summary}
            if not group_values:
                continue
            lines.append(f"{group_name}:")
            for key in keys:
                values = group_values.get(key)
                if not values:
                    continue
                if key == "VisualStim" and visualstim_menu:
                    mapped = set()
                    for value in values:
                        try:
                            idx_value = int(float(value)) - 1
                            if 0 <= idx_value < len(visualstim_menu):
                                mapped.add(visualstim_menu[idx_value])
                        except (TypeError, ValueError):
                            pass
                    if mapped:
                        lines.append(f"  {key}={format_values(values)} -> {format_values(mapped)}")
                        continue
                lines.append(f"  {key}={format_values(values)}")
            lines.append("-" * 80)

        interpretation = []
        if gui_summary.get("VirtualCricket") and 1 in gui_summary["VirtualCricket"]:
            interpretation.append("Virtual cricket hunt is enabled (closed-loop stimulus).")
            if gui_summary.get("PortCricket") and 1 in gui_summary["PortCricket"]:
                interpretation.append("PortCricket is enabled (goal: drive cricket to wall).")
            if gui_summary.get("VirtualCricketDistance"):
                interpretation.append(f"Cricket distance: {format_values(gui_summary['VirtualCricketDistance'])} mm.")
            if gui_summary.get("VirtualCricketAngle"):
                interpretation.append(f"Cricket jump angle: {format_values(gui_summary['VirtualCricketAngle'])}.")
            if gui_summary.get("VirtualCricketRandom") and 1 in gui_summary["VirtualCricketRandom"]:
                interpretation.append("Cricket jump orientation is randomized.")
        if gui_summary.get("FadingBeacon") and 1 in gui_summary["FadingBeacon"]:
            interpretation.append("Fading beacon trials are enabled.")
        if gui_summary.get("VisualStim"):
            if visualstim_menu:
                mapped = []
                for value in gui_summary["VisualStim"]:
                    try:
                        idx_value = int(float(value)) - 1
                        if 0 <= idx_value < len(visualstim_menu):
                            mapped.append(visualstim_menu[idx_value])
                    except (TypeError, ValueError):
                        continue
                if mapped:
                    interpretation.append(f"Visual stimulus types: {', '.join(sorted(set(mapped)))}.")
            else:
                interpretation.append(f"VisualStim index: {format_values(gui_summary['VisualStim'])}.")
        if gui_summary.get("bonsai_track") and 1 in gui_summary["bonsai_track"]:
            interpretation.append("Tracking uses Bonsai (video-based).")
        if gui_summary.get("hwtrigger") and 1 in gui_summary["hwtrigger"]:
            interpretation.append("Hardware trigger enabled.")

        if interpretation:
            lines.append("Interpretation:")
            for line in interpretation:
                lines.append(f"  - {line}")
            lines.append("-" * 80)

        summary_text = "\n".join(lines)
        if print_output:
            print(summary_text)

        return {
            "bpod_path": str(self._bpod_path_full),
            "session_id": session_id,
            "scan_id": scan_id,
            "n_trials": n_trials,
            "trial_type_names": trial_type_names,
            "trial_types": trial_types,
            "protocol_name": protocol_name,
            "gui_summary": {k: sorted(v, key=str) for k, v in gui_summary.items()},
            "interpretation": interpretation,
            "summary_text": summary_text,
        }

    def _build_bpod_recording_key(self, session_id, scan_id, max_values=6):
        session_data = self.session_data if isinstance(self.session_data, dict) else {}
        info = session_data.get("Info", {}) if isinstance(session_data, dict) else {}
        protocol_name = info.get("ProtocolName") or info.get("Protocol") or info.get("ProtocolPath") or ""
        if protocol_name:
            protocol_name = str(protocol_name)
        protocol_path = info.get("ProtocolPath")
        protocol_path = str(protocol_path) if protocol_path else ""

        def normalize_list(values):
            normalized = []
            for value in values:
                if isinstance(value, np.ndarray):
                    if value.shape == ():
                        value = value.item()
                    else:
                        value = value.tolist()
                if isinstance(value, np.generic):
                    value = value.item()
                normalized.append(value)
            return normalized

        trial_type_names = normalize_list(_as_list(session_data.get("TrialTypeNames")))
        trial_types = normalize_list(_as_list(session_data.get("TrialTypes")))
        event_names, state_names, softcode_names = self._collect_raw_event_state_names()

        duration = None
        try:
            duration = float(np.nansum(
                session_data["TrialEndTimestamp"] - session_data["TrialStartTimestamp"]
            ))
        except Exception:
            duration = None

        summary = self.summarize_metadata(
            session_id=session_id,
            scan_id=scan_id,
            max_values=max_values,
            print_output=False,
        )
        summary_text = summary.get("summary_text")
        if not protocol_name:
            protocol_name = summary.get("protocol_name") or ""
        protocol_file, protocol_file_name = self._select_protocol_file(protocol_name, protocol_path)
        if protocol_file:
            protocol_name = protocol_file_name or protocol_name
            protocol_path = str(self._relative_to_root(protocol_file))

        session_data_blob = self.session_data if isinstance(self.session_data, dict) else None

        return {
            "session_id": session_id,
            "scan_id": scan_id,
            "bpod_recording_start_time": self.start_time,
            "bpod_recording_duration": duration,
            "bpod_protocol_name": protocol_name,
            "bpod_n_trials": int(self.n_trials) if self.n_trials is not None else None,
            "bpod_trial_types": trial_types or None,
            "bpod_trial_type_names": trial_type_names or None,
            "bpod_event_names": event_names or None,
            "bpod_state_names": state_names or None,
            "bpod_softcode_names": softcode_names or None,
            "bpod_info": info or None,
            "session_data": session_data_blob,
            "bpod_metadata": summary_text,
            "bpod_mat_filepath": str(self._bpod_path_relative),
            "bpod_protocol_filepath": protocol_path or None,
        }

    def _build_session_key(self, session_id):
        return {
            "session_id": session_id,
            "subject": self.subject_id,
            "session_datetime": self.start_time,
        }

    def _build_behavior_recording_key(self, session_id, scan_id, bpod_version, notes_suffix=""):
        notes = f"BPod version: {bpod_version}"
        if notes_suffix:
            notes = f"{notes}, {notes_suffix}"
        return {
            "session_id": session_id,
            "scan_id": scan_id,
            "recording_start_time": self.start_time,
            "recording_duration": sum(
                self.session_data["TrialEndTimestamp"]
                - self.session_data["TrialStartTimestamp"]
            ),
            "recording_notes": notes,
        }

    def _build_behavior_recording_fp_key(self, session_id, scan_id):
        return {
            "session_id": session_id,
            "scan_id": scan_id,
            "filepath": self._bpod_path_relative,
        }

    def _build_trial_type_keys(self, trial_types):
        return [{"trial_type": trial_type} for trial_type in np.unique(trial_types).tolist()]

    def _build_trial_attributes_keys(self, session_id, scan_id, trial_indices, predicate):
        keys = []
        for n in trial_indices:
            for attrib in self.trial(n).attributes:
                value = self.trial(n).attributes[attrib]
                if not predicate(value):
                    continue
                keys.append({
                    "session_id": session_id,
                    "scan_id": scan_id,
                    "trial_id": n,
                    "attribute_name": attrib,
                    "attribute_value": value,
                })
        return keys

    def _print_ingest_summary(self, event_keys):
        print(
            "\n\t".join(
                [
                    "BPod items to be inserted:",
                    f"Subject : {self.subject_id}",
                    f"Time    : {self.start_time}",
                    f"N Trials: {self.n_trials}",
                    f"N Events: {len(event_keys)}",
                ]
            )
        )

    def _confirm_prompt(self, prompt):
        if prompt and dj.utils.user_choice("Proceed with new subject(s) insert?") != "yes":
            print("Canceled insert.")
            return False
        return True

    def _insert_ingest_payload(self, session_key, behavior_recording_key, behavior_recording_fp_key,
                               bpod_recording_key, trial_type_keys, trial_keys,
                               trial_attributes_keys, event_type_keys, event_keys):
        with session.Session.connection.transaction:
            session.Session.insert1(session_key, skip_duplicates=True)
            event.BehaviorRecording.insert1(behavior_recording_key, skip_duplicates=True)
            event.BehaviorRecording.File.insert1(behavior_recording_fp_key, skip_duplicates=True)
            event.BpodRecording.insert1(bpod_recording_key, skip_duplicates=True)
            trial.TrialType.insert(trial_type_keys, skip_duplicates=True)
            trial.Trial.insert(trial_keys, allow_direct_insert=True, skip_duplicates=True)
            trial.Trial.Attribute.insert(trial_attributes_keys, allow_direct_insert=True)
            event.EventType.insert(event_type_keys, skip_duplicates=True)
            event_keys = Trial.split_event_times_list(event_keys)
            event.Event.insert(
                event_keys, allow_direct_insert=True, ignore_extra_fields=True, skip_duplicates=True
            )
            trial.TrialEvent.insert(event_keys, allow_direct_insert=True)

        
    def ingest(self, session_id, scan_id, prompt=False,
               include_raw_bpod_events=False, include_raw_bpod_states=False):
        """Ingest BPod data to session, event, and trial tables.

        :param prompt (bool): Optional, default True. Prompt with metadata before entry.
        :param include_raw_bpod_events (bool): Optional, default False. Store raw Bpod events.
        :param include_raw_bpod_states (bool): Optional, default False. Store raw Bpod states.
        """
        # -------------------------- Check if already exists --------------------------
        if event.BehaviorRecording.File & f"filepath='{self._bpod_path_relative}'":
            print("Session already exists, skipping...")  # check this bpod file path
            return
        if not subject.Subject & f'subject="{self.subject_id}"':  # check this subject
            from .pyrat import PyratIngestion

            print(
                f"Subject does not yet exist."
                + f"Attempting pyrat import: {self.subject_id}"
            )
            PyratIngestion().ingest_animal(self.subject_id, prompt=False)

        # ------------------------------- Some constants -------------------------------
        
        bpod_version = self.session_data["Info"]["StateMachineVersion"].split(" ")[-1]
        aux_trials, aux_rewards = self._aux_timestamps()
        

        # ------------------------------- Keys to insert -------------------------------
        session_key = self._build_session_key(session_id)
        behavior_recording_key = self._build_behavior_recording_key(
            session_id, scan_id, bpod_version
        )
        behavior_recording_fp_key = self._build_behavior_recording_fp_key(
            session_id, scan_id
        )
        bpod_recording_key = self._build_bpod_recording_key(session_id, scan_id)
        trial_type_keys = self._build_trial_type_keys(
            self.session_data["TrialTypeNames"]
        )
        trial_keys = [
            {
                "session_id": session_id,
                "scan_id": scan_id,
                "trial_id": n,
                "trial_type": self.trial(n).type,
                "trial_start_time": aux_trials[n] - self.trial(n).events['bpod_at_target'],     #TR23: aux start time is equivalent to bpod_at_target. This means: The signal of the visual target being triggered on the aux channel is subtracted by the BPOD reference for the same event. 
                "trial_stop_time": aux_trials[n] - self.trial(n).events['bpod_at_target'] + self.trial(n).duration, #TR23: removed - self.trial(n).events['bpod_cue'] because it was already subtracted above
            }
            for n in range(self.n_trials)
        ]
        trial_attributes_keys = self._build_trial_attributes_keys(
            session_id, scan_id, range(self.n_trials), _attr_present
        )
        raw_event_keys = []
        raw_event_types = set()
        if include_raw_bpod_events or include_raw_bpod_states:
            for n in range(self.n_trials):
                anchor_time = self.trial(n).events.get("bpod_at_target")
                base_time = aux_trials[n] - anchor_time if anchor_time is not None else None
                trial_raw_keys, trial_raw_types = self._raw_event_keys_for_trial(
                    session_id=session_id,
                    scan_id=scan_id,
                    trial_idx=n,
                    base_time=base_time,
                    include_events=include_raw_bpod_events,
                    include_states=include_raw_bpod_states,
                )
                raw_event_keys.extend(trial_raw_keys)
                raw_event_types.update(trial_raw_types)
        event_type_keys = [
            {"event_type": event_type}
            for event_type in set(
                event_type
                for n in range(self.n_trials)
                for event_type in self.trial(n).events
            ).union(raw_event_types)
        ]
        # event_type_keys.extend([
        #     {"event_type": "aux_reward"}
        # ])
        event_keys = [
            {
                "session_id": session_id,
                "scan_id": scan_id,
                "trial_id": n,
                "event_type": event,
                "event_start_time": aux_trials[n] - self.trial(n).events['bpod_at_target'] + event_start, #TR23: IMPORTANT SYNC LINE! subtracting bpod_at_target first since aux start time (visual stim location triggered) is equivalent to bpod_at_target
            }
            for n in range(self.n_trials)
            for event, event_start in self.trial(n).events.items()
            if event_start is not None
        ]
        if raw_event_keys:
            event_keys.extend(raw_event_keys)
        # event_keys.extend(
        #     [  # add reward times from aux
        #         {
        #             "session_id": session_id,
        #             "scan_id": scan_id,
        #             "trial_id": bisect(aux_trials, reward) - 1,  # finds trial ID
        #             "event_type": "aux_reward",
        #             "event_start_time": reward,
        #         }
        #         for reward in aux_rewards
        #     ]
        # )

        # ---------------------------------- Prompt ----------------------------------
        self._print_ingest_summary(event_keys)
        if not self._confirm_prompt(prompt):
            return

        self._insert_ingest_payload(
            session_key=session_key,
            behavior_recording_key=behavior_recording_key,
            behavior_recording_fp_key=behavior_recording_fp_key,
            bpod_recording_key=bpod_recording_key,
            trial_type_keys=trial_type_keys,
            trial_keys=trial_keys,
            trial_attributes_keys=trial_attributes_keys,
            event_type_keys=event_type_keys,
            event_keys=event_keys,
        )

    def ingest_oddball(self, session_id, scan_id, prompt=False, bood_trial_offset = 0,
                       include_raw_bpod_events=False, include_raw_bpod_states=False): # start bpod aux sync at this trial
        """Ingest Oddball BPod data to session, event, and trial tables.

        :param prompt (bool): Optional, default True. Prompt with metadata before entry.
        :param bood_trial_offset (int): Optional, default 0, # start bpod aux sync at this trial
        :param include_raw_bpod_events (bool): Optional, default False. Store raw Bpod events.
        :param include_raw_bpod_states (bool): Optional, default False. Store raw Bpod states.
        """
        # -------------------------- Check if already exists --------------------------
        self.subject_id = (session.Session & f'session_id="{session_id}"').fetch1('subject')

        if event.BehaviorRecording.File & f"filepath='{self._bpod_path_relative}'":
            print("Session already exists, skipping...")  # check this bpod file path
            return

        # ------------------------------- Some constants -------------------------------
        
        bpod_version = self.session_data["Info"]["StateMachineVersion"].split(" ")[-1]
        aux_gate, aux_bpod_visstim = self._aux_oddball_timestamps()
        aux_bpod_visstim = aux_bpod_visstim[aux_bpod_visstim > aux_gate + 1]

        # ------------------------------- Keys to insert -------------------------------
        
        bood_trial_offset = self.n_trials - len(aux_bpod_visstim) # start bpod aux sync at this trial
        print('!! Assuming bpod_trial_offset:', bood_trial_offset)
        
        session_key = self._build_session_key(session_id)
        behavior_recording_key = self._build_behavior_recording_key(
            session_id, scan_id, bpod_version
        )
        behavior_recording_fp_key = self._build_behavior_recording_fp_key(
            session_id, scan_id
        )
        bpod_recording_key = self._build_bpod_recording_key(session_id, scan_id)
        trial_type_keys = self._build_trial_type_keys(
            self.session_data["TrialTypes"]
        )
        trial_keys = [
            {
                "session_id": session_id,
                "scan_id": scan_id,
                "trial_id": n,
                "trial_type": self.trial(n).type,
                "trial_start_time": aux_bpod_visstim[n-bood_trial_offset] - self.trial(n).events['bpod_firststim_oddball'],     
                "trial_stop_time": aux_bpod_visstim[n-bood_trial_offset] - self.trial(n).events['bpod_firststim_oddball'] + self.trial(n).duration, 
            }
            for n in range(bood_trial_offset, self.n_trials) # start at second trial
        ]
        trial_attributes_keys = self._build_trial_attributes_keys(
            session_id, scan_id, range(bood_trial_offset, self.n_trials), _attr_truthy
        )
        raw_event_keys = []
        raw_event_types = set()
        if include_raw_bpod_events or include_raw_bpod_states:
            for n in range(bood_trial_offset, self.n_trials):
                anchor_time = self.trial(n).events.get("bpod_firststim_oddball")
                base_time = aux_bpod_visstim[n - bood_trial_offset] - anchor_time if anchor_time is not None else None
                trial_raw_keys, trial_raw_types = self._raw_event_keys_for_trial(
                    session_id=session_id,
                    scan_id=scan_id,
                    trial_idx=n,
                    base_time=base_time,
                    include_events=include_raw_bpod_events,
                    include_states=include_raw_bpod_states,
                )
                raw_event_keys.extend(trial_raw_keys)
                raw_event_types.update(trial_raw_types)
        event_type_keys = [
            {"event_type": event_type}
            for event_type in set(
                event_type
                for n in range(bood_trial_offset, self.n_trials) # start at second trial
                for event_type in self.trial(n).events
            ).union(raw_event_types)
        ]
        # event_type_keys.extend([
        #     {"event_type": "aux_reward"}
        # ])
        event_keys = [
            {
                "session_id": session_id,
                "scan_id": scan_id,
                "trial_id": n,
                "event_type": event,
                "event_start_time": aux_bpod_visstim[n-bood_trial_offset] - self.trial(n).events['bpod_firststim_oddball'] + event_start, #TR23: IMPORTANT SYNC LINE! subtracting bpod_at_target first since aux start time (visual stim location triggered) is equivalent to bpod_at_target
            }
            for n in range(bood_trial_offset, self.n_trials) # start at second trial
            for event, event_start in self.trial(n).events.items()
            if event_start is not None
        ]
        if raw_event_keys:
            event_keys.extend(raw_event_keys)
        # event_keys.extend(
        #     [  # add reward times from aux
        #         {
        #             "session_id": session_id,
        #             "scan_id": scan_id,
        #             "trial_id": bisect(aux_trials, reward) - 1,  # finds trial ID
        #             "event_type": "aux_reward",
        #             "event_start_time": reward,
        #         }
        #         for reward in aux_rewards
        #     ]
        # )

        # ---------------------------------- Prompt ----------------------------------
        self._print_ingest_summary(event_keys)
        if not self._confirm_prompt(prompt):
            return

        self._insert_ingest_payload(
            session_key=session_key,
            behavior_recording_key=behavior_recording_key,
            behavior_recording_fp_key=behavior_recording_fp_key,
            bpod_recording_key=bpod_recording_key,
            trial_type_keys=trial_type_keys,
            trial_keys=trial_keys,
            trial_attributes_keys=trial_attributes_keys,
            event_type_keys=event_type_keys,
            event_keys=event_keys,
        )
            
    def ingest_oddball_v2(self, session_id, scan_id, prompt=False, bood_trial_offset = 0,
                          include_raw_bpod_events=False, include_raw_bpod_states=False): # start bpod aux sync at this trial
        """Ingest Oddball BPod data to session, event, and trial tables.

        :param prompt (bool): Optional, default True. Prompt with metadata before entry.
        :param bood_trial_offset (int): Optional, default 0, # start bpod aux sync at this trial
        :param include_raw_bpod_events (bool): Optional, default False. Store raw Bpod events.
        :param include_raw_bpod_states (bool): Optional, default False. Store raw Bpod states.
        """
        # -------------------------- Check if already exists --------------------------
        scan_key = (scan.Scan & f'scan_id = "{scan_id}"').fetch1('KEY')
        self.subject_id = (session.Session & scan_key).fetch1('subject')

        if event.BehaviorRecording.File & f"filepath='{self._bpod_path_relative}'":
            print("Session already exists, skipping...")  # check this bpod file path
            return

        # ------------------------------- Some constants -------------------------------
        
        bpod_version = self.session_data["Info"]["StateMachineVersion"].split(" ")[-1]
        
        aux_gate = (event.Event & scan_key & 'event_type LIKE "%gate%"').fetch1('event_start_time')
        aux_bpod_trialstart = (event.Event & scan_key & 'event_type LIKE "%trial%"').fetch('event_start_time')
        aux_bpod_trialstart = aux_bpod_trialstart[aux_bpod_trialstart > aux_gate]
        aux_bpod_trialstart = aux_bpod_trialstart[:-1]  # drop last trial since trial ttl goes high after last trial

        # ------------------------------- Keys to insert -------------------------------
        
        bood_trial_offset = self.n_trials - len(aux_bpod_trialstart) # start bpod aux sync at this trial
        print('!! Assuming bpod_trial_offset:', bood_trial_offset)
        
        session_key = self._build_session_key(session_id)
        behavior_recording_key = self._build_behavior_recording_key(
            session_id, scan_id, bpod_version
        )
        behavior_recording_fp_key = self._build_behavior_recording_fp_key(
            session_id, scan_id
        )
        bpod_recording_key = self._build_bpod_recording_key(session_id, scan_id)
        trial_type_keys = self._build_trial_type_keys(
            self.session_data["TrialTypeNames"]
        )
        trial_keys = [
            {
                "session_id": session_id,
                "scan_id": scan_id,
                "trial_id": n,
                "trial_type": self.trial(n).type,
                "trial_start_time": aux_bpod_trialstart[n-bood_trial_offset] - self.trial(n).events['bpod_firststim_oddball'],     
                "trial_stop_time": aux_bpod_trialstart[n-bood_trial_offset] - self.trial(n).events['bpod_firststim_oddball'] + self.trial(n).duration, 
            }
            for n in range(bood_trial_offset, self.n_trials) # start at second trial
        ]
        trial_attributes_keys = self._build_trial_attributes_keys(
            session_id, scan_id, range(bood_trial_offset, self.n_trials), _attr_present
        )
        raw_event_keys = []
        raw_event_types = set()
        if include_raw_bpod_events or include_raw_bpod_states:
            for n in range(bood_trial_offset, self.n_trials):
                anchor_time = self.trial(n).events.get("bpod_firststim_oddball")
                base_time = aux_bpod_trialstart[n - bood_trial_offset] - anchor_time if anchor_time is not None else None
                trial_raw_keys, trial_raw_types = self._raw_event_keys_for_trial(
                    session_id=session_id,
                    scan_id=scan_id,
                    trial_idx=n,
                    base_time=base_time,
                    include_events=include_raw_bpod_events,
                    include_states=include_raw_bpod_states,
                )
                raw_event_keys.extend(trial_raw_keys)
                raw_event_types.update(trial_raw_types)
        event_type_keys = [
            {"event_type": event_type}
            for event_type in set(
                event_type
                for n in range(bood_trial_offset, self.n_trials) # start at second trial
                for event_type in self.trial(n).events
            ).union(raw_event_types)
        ]
        # event_type_keys.extend([
        #     {"event_type": "aux_reward"}
        # ])
        event_keys = [
            {
                "session_id": session_id,
                "scan_id": scan_id,
                "trial_id": n,
                "event_type": event,
                "event_start_time": aux_bpod_trialstart[n-bood_trial_offset] - self.trial(n).events['bpod_firststim_oddball'] + event_start, #TR23: IMPORTANT SYNC LINE! subtracting bpod_at_target first since aux start time (visual stim location triggered) is equivalent to bpod_at_target
            }
            for n in range(bood_trial_offset, self.n_trials) # start at second trial
            for event, event_start in self.trial(n).events.items()
            if event_start is not None
        ]
        if raw_event_keys:
            event_keys.extend(raw_event_keys)
        # event_keys.extend(
        #     [  # add reward times from aux
        #         {
        #             "session_id": session_id,
        #             "scan_id": scan_id,
        #             "trial_id": bisect(aux_trials, reward) - 1,  # finds trial ID
        #             "event_type": "aux_reward",
        #             "event_start_time": reward,
        #         }
        #         for reward in aux_rewards
        #     ]
        # )

        # ---------------------------------- Prompt ----------------------------------
        self._print_ingest_summary(event_keys)
        if not self._confirm_prompt(prompt):
            return

        self._insert_ingest_payload(
            session_key=session_key,
            behavior_recording_key=behavior_recording_key,
            behavior_recording_fp_key=behavior_recording_fp_key,
            bpod_recording_key=bpod_recording_key,
            trial_type_keys=trial_type_keys,
            trial_keys=trial_keys,
            trial_attributes_keys=trial_attributes_keys,
            event_type_keys=event_type_keys,
            event_keys=event_keys,
        )

    def ingest_behavior_box(self, session_id, scan_id, prompt=False, bood_trial_offset = 0,
                            include_raw_bpod_events=False, include_raw_bpod_states=False): # start bpod aux sync at this trial
        """Ingest Behavior Box BPod data to session, event, and trial tables.
        
        Note: For behavior_box setup, only BPod data is present (no auxdata synchronization needed).
        Uses TrialStartTimestamp directly for trial timing.

        :param prompt (bool): Optional, default True. Prompt with metadata before entry.
        :param bood_trial_offset (int): Optional, default 0, # start bpod trial sync at this trial
        :param include_raw_bpod_events (bool): Optional, default False. Store raw Bpod events.
        :param include_raw_bpod_states (bool): Optional, default False. Store raw Bpod states.
        """
        # -------------------------- Check if already exists --------------------------
        if event.BehaviorRecording.File & f"filepath='{self._bpod_path_relative}'":
            print("Session already exists, skipping...")  # check this bpod file path
            return
        if not subject.Subject & f'subject="{self.subject_id}"':  # check this subject
            from .pyrat import PyratIngestion

            print(
                f"Subject does not yet exist."
                + f"Attempting pyrat import: {self.subject_id}"
            )
            PyratIngestion().ingest_animal(self.subject_id, prompt=False)

        # ------------------------------- Some constants -------------------------------
        
        bpod_version = self.session_data["Info"]["StateMachineVersion"].split(" ")[-1]
        
        # For behavior_box: Use BPod timestamps directly (no auxdata sync needed)
        bpod_trial_timestamps = self.session_data["TrialStartTimestamp"]

        # ------------------------------- Keys to insert -------------------------------
        
        session_key = {
            "session_id": session_id,
            "subject": self.subject_id,
            "session_datetime": self.start_time,
        }
        session_key = self._build_session_key(session_id)
        behavior_recording_key = self._build_behavior_recording_key(
            session_id, scan_id, bpod_version, notes_suffix="Behavior Box setup"
        )
        behavior_recording_fp_key = self._build_behavior_recording_fp_key(
            session_id, scan_id
        )
        bpod_recording_key = self._build_bpod_recording_key(session_id, scan_id)
        
        # Handle both TrialTypeNames and TrialTypes
        try:
            trial_type_data = np.unique(self.session_data["TrialTypeNames"]).tolist()
        except:
            trial_type_data = np.unique(self.session_data["TrialTypes"]).tolist()
            
        trial_type_keys = self._build_trial_type_keys(trial_type_data)
        
        trial_keys = [
            {
                "session_id": session_id,
                "scan_id": scan_id,
                "trial_id": n,
                "trial_type": self.trial(n).type,
                "trial_start_time": bpod_trial_timestamps[n],  # Use BPod timestamp directly     
                "trial_stop_time": bpod_trial_timestamps[n] + self.trial(n).duration, 
            }
            for n in range(bood_trial_offset, self.n_trials)
        ]
        trial_attributes_keys = self._build_trial_attributes_keys(
            session_id, scan_id, range(bood_trial_offset, self.n_trials), _attr_present
        )
        raw_event_keys = []
        raw_event_types = set()
        if include_raw_bpod_events or include_raw_bpod_states:
            for n in range(bood_trial_offset, self.n_trials):
                base_time = bpod_trial_timestamps[n]
                trial_raw_keys, trial_raw_types = self._raw_event_keys_for_trial(
                    session_id=session_id,
                    scan_id=scan_id,
                    trial_idx=n,
                    base_time=base_time,
                    include_events=include_raw_bpod_events,
                    include_states=include_raw_bpod_states,
                )
                raw_event_keys.extend(trial_raw_keys)
                raw_event_types.update(trial_raw_types)
        event_type_keys = [
            {"event_type": event_type}
            for event_type in set(
                event_type
                for n in range(bood_trial_offset, self.n_trials)
                for event_type in self.trial(n).events
            ).union(raw_event_types)
        ]
        event_keys = [
            {
                "session_id": session_id,
                "scan_id": scan_id,
                "trial_id": n,
                "event_type": event,
                "event_start_time": bpod_trial_timestamps[n] + event_start,  # BPod timestamp + event offset
            }
            for n in range(bood_trial_offset, self.n_trials)
            for event, event_start in self.trial(n).events.items()
            if event_start is not None
        ]
        if raw_event_keys:
            event_keys.extend(raw_event_keys)

        # ---------------------------------- Prompt ----------------------------------
        print(
            "\n\t".join(
                [
                    "BPod Behavior Box items to be inserted:",
                    f"Subject : {self.subject_id}",
                    f"Time    : {self.start_time}",
                    f"N Trials: {self.n_trials}",
                    f"N Events: {len(event_keys)}",
                    f"Setup   : Behavior Box (BPod only, no auxdata sync)",
                ]
            )
        )
        if not self._confirm_prompt(prompt):
            return

        self._insert_ingest_payload(
            session_key=session_key,
            behavior_recording_key=behavior_recording_key,
            behavior_recording_fp_key=behavior_recording_fp_key,
            bpod_recording_key=bpod_recording_key,
            trial_type_keys=trial_type_keys,
            trial_keys=trial_keys,
            trial_attributes_keys=trial_attributes_keys,
            event_type_keys=event_type_keys,
            event_keys=event_keys,
        )

    def ingest_linear_track(self, session_id, scan_id, prompt=False, bood_trial_offset=0,
                            include_raw_bpod_events=False, include_raw_bpod_states=False):  # start bpod aux sync at this trial
        """Ingest Oddball BPod data to session, event, and trial tables.

        :param prompt (bool): Optional, default True. Prompt with metadata before entry.
        :param bood_trial_offset (int): Optional, default 0, # start bpod aux sync at this trial
        :param include_raw_bpod_events (bool): Optional, default False. Store raw Bpod events.
        :param include_raw_bpod_states (bool): Optional, default False. Store raw Bpod states.
        """
        # -------------------------- Check if already exists --------------------------
        if event.BehaviorRecording.File & f"filepath='{self._bpod_path_relative}'":
            print("Session already exists, skipping...")  # check this bpod file path
            return
        if not subject.Subject & f'subject="{self.subject_id}"':  # check this subject
            from .pyrat import PyratIngestion

            print(
                f"Subject does not yet exist."
                + f"Attempting pyrat import: {self.subject_id}"
            )
            PyratIngestion().ingest_animal(self.subject_id, prompt=False)

        
        bpod_version = self.session_data["Info"]["StateMachineVersion"].split(" ")[-1]
        aux_gate, aux_bpod_trials = self._aux_lineartrack_timestamps()
        aux_bpod_trials = aux_bpod_trials[aux_bpod_trials > aux_gate]

        # ------------------------------- Keys to insert -------------------------------
        
        # bood_trial_offset = 1 # start bpod aux sync at this trial
        
        session_key = self._build_session_key(session_id)
        behavior_recording_key = self._build_behavior_recording_key(
            session_id, scan_id, bpod_version
        )
        behavior_recording_fp_key = self._build_behavior_recording_fp_key(
            session_id, scan_id
        )
        bpod_recording_key = self._build_bpod_recording_key(session_id, scan_id)
        trial_type_keys = self._build_trial_type_keys(
            self.session_data["TrialTypes"]
        )
        trial_keys = [
            {
                "session_id": session_id,
                "scan_id": scan_id,
                "trial_id": n,
                "trial_type": self.trial(n).type,
                "trial_start_time": aux_bpod_trials[n-bood_trial_offset] - self.trial(n).events['bpod_bonsaitrial_lineartrack'],     
                "trial_stop_time": aux_bpod_trials[n-bood_trial_offset] - self.trial(n).events['bpod_bonsaitrial_lineartrack'] + self.trial(n).duration, 
            }
            for n in range(bood_trial_offset, self.n_trials) # start at second trial
        ]
        trial_attributes_keys = self._build_trial_attributes_keys(
            session_id, scan_id, range(bood_trial_offset, self.n_trials), _attr_truthy
        )
        raw_event_keys = []
        raw_event_types = set()
        if include_raw_bpod_events or include_raw_bpod_states:
            for n in range(bood_trial_offset, self.n_trials):
                anchor_time = self.trial(n).events.get("bpod_bonsaitrial_lineartrack")
                base_time = aux_bpod_trials[n - bood_trial_offset] - anchor_time if anchor_time is not None else None
                trial_raw_keys, trial_raw_types = self._raw_event_keys_for_trial(
                    session_id=session_id,
                    scan_id=scan_id,
                    trial_idx=n,
                    base_time=base_time,
                    include_events=include_raw_bpod_events,
                    include_states=include_raw_bpod_states,
                )
                raw_event_keys.extend(trial_raw_keys)
                raw_event_types.update(trial_raw_types)
        event_type_keys = [
            {"event_type": event_type}
            for event_type in set(
                event_type
                for n in range(bood_trial_offset, self.n_trials) # start at second trial
                for event_type in self.trial(n).events
            ).union(raw_event_types)
        ]
        # event_type_keys.extend([
        #     {"event_type": "aux_reward"}
        # ])
        event_keys = [
            {
                "session_id": session_id,
                "scan_id": scan_id,
                "trial_id": n,
                "event_type": event,
                "event_start_time": aux_bpod_trials[n-bood_trial_offset] - self.trial(n).events['bpod_bonsaitrial_lineartrack'] + event_start, #TR23: IMPORTANT SYNC LINE! subtracting bpod_at_target first since aux start time (visual stim location triggered) is equivalent to bpod_at_target
            }
            for n in range(bood_trial_offset, self.n_trials) # start at second trial
            for event, event_start in self.trial(n).events.items()
            if event_start is not None
        ]
        if raw_event_keys:
            event_keys.extend(raw_event_keys)
        # event_keys.extend(
        #     [  # add reward times from aux
        #         {
        #             "session_id": session_id,
        #             "scan_id": scan_id,
        #             "trial_id": bisect(aux_trials, reward) - 1,  # finds trial ID
        #             "event_type": "aux_reward",
        #             "event_start_time": reward,
        #         }
        #         for reward in aux_rewards
        #     ]
        # )

        # ---------------------------------- Prompt ----------------------------------
        print(
            "\n\t".join(
                [
                    "BPod items to be inserted:",
                    f"Subject : {self.subject_id}",
                    f"Time    : {self.start_time}",
                    f"N Trials: {self.n_trials}",
                    f"N Events: {len(event_keys)}",
                ]
            )
        )
        if not self._confirm_prompt(prompt):
            return

        self._insert_ingest_payload(
            session_key=session_key,
            behavior_recording_key=behavior_recording_key,
            behavior_recording_fp_key=behavior_recording_fp_key,
            bpod_recording_key=bpod_recording_key,
            trial_type_keys=trial_type_keys,
            trial_keys=trial_keys,
            trial_attributes_keys=trial_attributes_keys,
            event_type_keys=event_type_keys,
            event_keys=event_keys,
        )

class Trial(object):
    def __init__(self, idx, bpod_path_full, session_data=None, trial_data=None):
        if not session_data:
            session_data = Bpodfile(bpod_path_full).session_data
        if not trial_data:
            trial_data = Bpodfile(bpod_path_full).trial_data

        # -- properties --
        # for args above
        self._idx = idx
        self._bpod_path_full = bpod_path_full
        self._session_data = session_data
        self._trial_data = trial_data
        # for general use
        try:
            self.type = self._session_data["TrialTypeNames"][self._idx]
        except:
            self.type = self._session_data["TrialTypes"][self._idx]
        self.duration = (  # endtime - start time. Will use Aux start time above
            self._session_data["TrialEndTimestamp"][self._idx]
            - self._session_data["TrialStartTimestamp"][self._idx]
        )
        # lazy loading
        self._attributes = {}
        self._events = {}

        # clean up bpod states
        self._states = {
            k: v
            for k, v in self._trial_data[self._idx]["States"].items()
            # Filter out states with nan values:
            if not np.any(np.isnan(v))
        }  # [None] below makes it easier to differentiate values from returned None
        self._resp_delay = self._states.get("WaitForResponse", [None])
        if self._resp_delay[0] and np.isnan(self._resp_delay[0]):
            self._resp_delay = [None]  # sometimes _resp_delay is [nan, nan]
        self._time_to_port = (
            self._resp_delay[1] - self._resp_delay[0] if self._resp_delay[0] else None #TR24: this was a misunderstanding from Chris'side. I will correct this to absolute time to port later
        )

        self._reward_port_entry = next((value for key, value in self._states.items() if "Port" in key and "RewardDelay" in key and not np.isnan(value).all()), None) #TR24 now extracting timestamps for rewarded ports only

        # clean up bpod port events
        self._raw_events = self._trial_data[self._idx]["Events"]
        self._ports_in = {
            # f"bpod_{port}": self._raw_events[port]  - self._states.get("WaitForPosTriggerSoftCode", [None])[1] # e.g., in_port_2: 8.3 
            f"bpod_{port}": self._raw_events[port]  # TR23: removed subtraction of WaitForPosTriggerSoftCode from all events
            for port in self._raw_events
            if "In" in port
        }

        self._lick = {}
        
        # Extract lick onsets only from BNC1High events
        if "BNC1High" in self._raw_events:
            self._lick["bpod_lick_onset"] = self._raw_events["BNC1High"]
    @property
    def attributes(self):
        """Returns all attributes for trial.Trial.Attributes as a dict"""
        if not self._attributes:
            self._attributes = {
                "error": True if "Punish" in self._states else False,
                "timeout": (
                    True
                    if self._time_to_port
                    and self._time_to_port
                    >= self._session_data["TrialSettings"][self._idx]["GUI"][
                        "ResponseTime"
                    ]
                    else False
                ),
            }
        return self._attributes

    @property
    def events(self):
        """Returns trial events as a dict {event_type: event_time} WRT trial start"""
        if not self._events:
            self._events = {
                # TR23: Removed subtraction of WaitForPosTriggerSoftCode from all events

                # "bpod_cue": self._states.get("WaitForPosTriggerSoftCode", [None])[0] - self._states.get("WaitForPosTriggerSoftCode", [None])[1],
                # "bpod_at_target": self._states.get("WaitForPosTriggerSoftCode", [None])[1] - self._states.get("WaitForPosTriggerSoftCode", [None])[1],
                # "bpod_at_port": self._time_to_port if self._states.get("Drinking", [None])[0] else None,
                "bpod_cue": self._states.get("WaitForPosTriggerSoftCode", [None])[0],
                "bpod_at_target": self._states.get("WaitForPosTriggerSoftCode", [None])[1] if self._states.get("WaitForPosTriggerSoftCode", [None])[0] is not None else None,
                "bpod_at_subtarget": self._states.get("CueDelayTrialCricket", [None])[0],
                # "bpod_at_correct_port": self._time_to_port + self._resp_delay[0] if self._states.get("Drinking", [None])[0] else None,                # TR24: added self._resp_delay[0]  to convert to absolute port time. Also changed name of state to corect port. 
                "bpod_at_correct_port": self._reward_port_entry[0]  if self._states.get("Drinking", [None])[0] is not None else None, #TR24: now correctly retruning entry into reward ports only
                "bpod_reward": self._reward_port_entry[1] if self._states.get("Drinking", [None])[0]is not None else None, #TR24: now correctly retruning entry into reward ports only
                "bpod_reward_track": self._states.get("Reward", [None])[0],
                "bpod_firststim_oddball": self._states.get("WaitForStimulus", [None])[1] if self._states.get("WaitForStimulus", [None])[0] is not None else None,
                "bpod_bonsaitrial_lineartrack": self._states.get("WaitForTriggerStartSoftCode", [None])[1] if self._states.get("WaitForTriggerStartSoftCode", [None])[0] is not None else None, #Bonsai trial start
                "bpod_correctlick_lineartrack": self._states.get("WaitForLick", [None])[1] if self._states.get("WaitForLick", [None])[0] is not None else None, #Correct lick (leading to reward)
                "bpod_reward_lineartrack": self._states.get("RewardNoLick", [None])[0], # Reward presented
                "bpod_rewardzone_standard_lineartrack": self._states.get("WaitRewardLick", [None])[1] if self._states.get("WaitRewardLick", [None])[0] is not None else None, # Reward zone entered in standard trials
                "bpod_rewardzone_crutch_lineartrack": self._states.get("WaitRewardNoLick", [None])[1] if self._states.get("WaitRewardNoLick", [None])[0] is not None else None, # Reward zone entered in cruitch trials

                

                # "bpod_reward": (
                #     # NOTE: Now taking reward events from Aux - TR23: uncommented block to prevent foreign key constraint error
                #     self._time_to_port + self._resp_delay[0]  # TR24: added self._resp_delay[0]  to convert to absolute port time. This is misleading, I know. 
                #     + self._session_data["TrialSettings"][0]["GUI"]["RewardDelay"]
                #     if self._states.get("Drinking", [None])[0]
                #     else None
                # ),
                # **self._ports_in,
                # "bpod_drinking": self._states.get("Drinking", [None])[0]- self._states.get("WaitForPosTriggerSoftCode", [None])[1] if self._states.get("Drinking", [None])[0] else None,
                **self._ports_in,
                **self._lick,
                "bpod_drinking": self._states.get("Drinking", [None])[0] if self._states.get("Drinking", [None])[0] is not None else None,
            }
            
            # # Add all remaining states that aren't explicitly defined above
            # explicitly_defined_states = {
            #     "WaitForPosTriggerSoftCode", "CueDelayTrialCricket", "Drinking", "Reward", 
            #     "WaitForStimulus", "WaitForTriggerStartSoftCode", "WaitForLick", "RewardNoLick", 
            #     "WaitRewardLick", "WaitRewardNoLick"
            # }
            
            # for state_name, state_value in self._states.items():
            #     if state_name not in explicitly_defined_states:
            #         # Add entry time (first element) if available
            #         if state_value and len(state_value) > 0 and state_value[0] is not None:
            #             self._events[f"bpod_{state_name.lower()}_entry"] = state_value[0]
            #         # Add exit time (second element) if available
            #         if state_value and len(state_value) > 1 and state_value[1] is not None:
            #             self._events[f"bpod_{state_name.lower()}_exit"] = state_value[1]
        return self._events
    def split_event_times_list(event_keys_list):
        new_event_keys_list = []

        for entry in event_keys_list:
            event_start_time = entry['event_start_time']
            if isinstance(event_start_time, np.ndarray):
                # Split the entry for each value in the NumPy array
                for time in event_start_time:
                    new_entry = entry.copy()
                    new_entry['event_start_time'] = time
                    new_event_keys_list.append(new_entry)
            else:
                # Keep the entry as is
                new_event_keys_list.append(entry)

        return new_event_keys_list


# Public aliases for compatibility with _new call sites
Bpodfile_new = Bpodfile
BpodfileNew = Bpodfile
Trial_new = Trial
# --------------------- HELPER LOADER FUNCTIONS -----------------

# matlab script exact translation
# Depreciated by Trial attribute/event split above: port_num vs at_port time
def PortInEvents(bpod_session, idx):
    """Replicate MATLAB func: return list of tuples for input ports: #, events, name"""
    events = bpod_session["RawEvents"]["Trial"][idx]["Events"]
    in_ports = [f for f in events.keys() if "In" in f]
    in_port_raw_events = []
    for port in in_ports:
        # (port#, event times, portname)
        in_port_raw_events.append((port[4:-2], events[port], port))
    return in_port_raw_events
