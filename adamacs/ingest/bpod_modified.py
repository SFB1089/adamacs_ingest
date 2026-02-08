import numpy as np
import datajoint as dj
import scipy.io as spio
from pathlib import Path
from bisect import bisect
from dateutil import parser
from element_interface.utils import find_full_path, find_root_directory
from ..pipeline import subject, session, trial, event
from ..paths import get_experiment_root_data_dir
from .aux import Auxfile


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

    def _aux_timestamps(self):
        aux_paths = list(self._bpod_path_full.parent.glob("*.h5"))
        assert len(aux_paths) == 1, f"Found more than one Aux h5 file\n{aux_paths}"
        print(aux_paths[0])
        aux = Auxfile(aux_paths[0]) #TR23: The fact that we read in the aux file again is very redundant. We should read it in once and pass it to the Bpodfile class
        # aux_onset = aux.main_track_gate  # master trigger
        aux_trials = aux.bpod_channels()["trial"]  # - aux_onset  # trial times wrt trigger - sweep]["analogScans"][1]
                                                 # corresponds to  bpod_trial_vis_chan = curr_aux[sweep]['analogScans'][1]
        aux_rewards = aux.bpod_channels()["reward"] # - aux_onset  # rewards wrt trigger - ["analogScans"][2], self._sample_rate

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

            aux_fake_triggers = [
                self.session_data['TrialStartTimestamp'][trial] + timeout_duration - bpod_to_aux_starttime_offset
                for trial in timeouttrials
                ]
            aux_trials = np.insert(aux_trials, timeouttrials, aux_fake_triggers)

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
        aux_gate = aux.main_track_gate(gatechannel = 5, channels = 6)  # get aux gate t
        aux_bpod_visstim = aux.bpod_channels()["oddball_visstim"]  # Call as a method

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
        aux_bpod_trials = aux.bpod_channels()["lineartrack_trial"] # - aux_onset  # rewards wrt trigger - ["analogScans"][2], self._sample_rate
        # aux_bpod_licks = aux.bpod_channels["lineartrack_licks"] # - aux_onset  # rewards wrt trigger - ["analogScans"][2], self._sample_rate
        # aux_bpod_rewardzone = aux.bpod_channels["lineartrack_rewardzone"] # - aux_onset  # rewards wrt trigger - ["analogScans"][2], self._sample_rate

        # self.n_trials = min(self.n_trials, len(aux_trials)) #TR23: Set the number of trials to the minimum of the number of AUX trials and the number of BPOD trials
        # self.n_trials = len(aux_trials) #TR23: Set the number of trials to the number of AUX trials

        # assert len(aux_trials) == self.n_trials, (
        #     "Number of trials do not match: "
        #     + f"BPod {self.n_trials} vs. Aux {len(aux_trials)}"
        # )
        return aux_gate, aux_bpod_trials


        
    def ingest(self, session_id, scan_id, prompt=False):
        """Ingest BPod data to session, event, and trial tables.

        :param prompt (bool): Optional, default True. Prompt with metadata before entry.
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
        session_key = {
            "session_id": session_id,
            "subject": self.subject_id,
            "session_datetime": self.start_time,
        }
        behavior_recording_key = {
            "session_id": session_id,
            "scan_id": scan_id,
            "recording_start_time": self.start_time,
            "recording_duration": sum(
                # removes time between trials, following example matlab code
                self.session_data["TrialEndTimestamp"]
                - self.session_data["TrialStartTimestamp"]
            ),
            "recording_notes": f"BPod version: {bpod_version}",
        }
        behavior_recording_fp_key = {
            "session_id": session_id,
            "scan_id": scan_id,
            "filepath": self._bpod_path_relative,
        }
        trial_type_keys = [
            {
                "trial_type": trial_type
            }
            for trial_type in np.unique(self.session_data["TrialTypeNames"]).tolist()
        ]
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
        trial_attributes_keys = [
            {
                "session_id": session_id,
                "scan_id": scan_id,
                "trial_id": n,
                "attribute_name": attrib,
                "attribute_value": self.trial(n).attributes[attrib],
            }
            for n in range(self.n_trials)
            for attrib in self.trial(n).attributes
            if self.trial(n).attributes[attrib]
        ]
        event_type_keys = [
            {"event_type": event_type}
            for event_type in set(
                event_type
                for n in range(self.n_trials)
                for event_type in self.trial(n).events
            )
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
        if (
            prompt
            and dj.utils.user_choice("Proceed with new subject(s) insert?") != "yes"
        ):
            print("Canceled insert.")
            return

        # ----------------------------- Insert to schemas -----------------------------
        with session.Session.connection.transaction:
            session.Session.insert1(session_key, skip_duplicates=True)  # remove skip
            event.BehaviorRecording.insert1(behavior_recording_key, skip_duplicates=True)
            event.BehaviorRecording.File.insert1(behavior_recording_fp_key, skip_duplicates=True)
            trial.TrialType.insert(trial_type_keys, skip_duplicates=True)
            trial.Trial.insert(trial_keys, allow_direct_insert=True, skip_duplicates=True)
            trial.Trial.Attribute.insert(
                trial_attributes_keys, allow_direct_insert=True
            )
            event.EventType.insert(event_type_keys, skip_duplicates=True)
            event_keys = Trial.split_event_times_list(event_keys) #TR23: Split event times list if there are multiple values in a 'event_start_time' field
            event.Event.insert(
                event_keys, allow_direct_insert=True, ignore_extra_fields=True, skip_duplicates=True
            )  # ignore extra trial_id
            trial.TrialEvent.insert(event_keys, allow_direct_insert=True)

    def ingest_oddball(self, session_id, scan_id, prompt=False, bood_trial_offset = 0): # start bpod aux sync at this trial
        """Ingest Oddball BPod data to session, event, and trial tables.

        :param prompt (bool): Optional, default True. Prompt with metadata before entry.
        :param bood_trial_offset (int): Optional, default 0, # start bpod aux sync at this trial
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
        
        session_key = {
            "session_id": session_id,
            "subject": self.subject_id,
            "session_datetime": self.start_time,
        }
        behavior_recording_key = {
            "session_id": session_id,
            "scan_id": scan_id,
            "recording_start_time": self.start_time,
            "recording_duration": sum(
                # removes time between trials, following example matlab code
                self.session_data["TrialEndTimestamp"]
                - self.session_data["TrialStartTimestamp"]
            ),
            "recording_notes": f"BPod version: {bpod_version}",
        }
        behavior_recording_fp_key = {
            "session_id": session_id,
            "scan_id": scan_id,
            "filepath": self._bpod_path_relative,
        }
        trial_type_keys = [
            {
                "trial_type": trial_type
            }
            for trial_type in np.unique(self.session_data["TrialTypes"]).tolist()
        ]
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
        trial_attributes_keys = [
            {
                "session_id": session_id,
                "scan_id": scan_id,
                "trial_id": n,
                "attribute_name": attrib,
                "attribute_value": self.trial(n).attributes[attrib],
            }
            for n in range(bood_trial_offset, self.n_trials)
            for attrib in self.trial(n).attributes
            if self.trial(n).attributes[attrib]
        ]
        event_type_keys = [
            {"event_type": event_type}
            for event_type in set(
                event_type
                for n in range(bood_trial_offset, self.n_trials) # start at second trial
                for event_type in self.trial(n).events
            )
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
        if (
            prompt
            and dj.utils.user_choice("Proceed with new subject(s) insert?") != "yes"
        ):
            print("Canceled insert.")
            return

        # ----------------------------- Insert to schemas -----------------------------
        with session.Session.connection.transaction:
            session.Session.insert1(session_key, skip_duplicates=True)  # remove skip
            event.BehaviorRecording.insert1(behavior_recording_key, skip_duplicates=True)
            event.BehaviorRecording.File.insert1(behavior_recording_fp_key, skip_duplicates=True)
            trial.TrialType.insert(trial_type_keys, skip_duplicates=True)
            trial.Trial.insert(trial_keys, allow_direct_insert=True, skip_duplicates=True)
            trial.Trial.Attribute.insert(
                trial_attributes_keys, allow_direct_insert=True
            )
            event.EventType.insert(event_type_keys, skip_duplicates=True)
            event_keys = Trial.split_event_times_list(event_keys) #TR23: Split event times list if there are multiple values in a 'event_start_time' field
            event.Event.insert(
                event_keys, allow_direct_insert=True, ignore_extra_fields=True, skip_duplicates=True
            )  # ignore extra trial_id
            trial.TrialEvent.insert(event_keys, allow_direct_insert=True)
    def ingest_linear_track(self, session_id, scan_id, prompt=False, bood_trial_offset = 0): # start bpod aux sync at this trial
        """Ingest Oddball BPod data to session, event, and trial tables.

        :param prompt (bool): Optional, default True. Prompt with metadata before entry.
        :param bood_trial_offset (int): Optional, default 0, # start bpod aux sync at this trial
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
        
        session_key = {
            "session_id": session_id,
            "subject": self.subject_id,
            "session_datetime": self.start_time,
        }
        behavior_recording_key = {
            "session_id": session_id,
            "scan_id": scan_id,
            "recording_start_time": self.start_time,
            "recording_duration": sum(
                # removes time between trials, following example matlab code
                self.session_data["TrialEndTimestamp"]
                - self.session_data["TrialStartTimestamp"]
            ),
            "recording_notes": f"BPod version: {bpod_version}",
        }
        behavior_recording_fp_key = {
            "session_id": session_id,
            "scan_id": scan_id,
            "filepath": self._bpod_path_relative,
        }
        trial_type_keys = [
            {
                "trial_type": trial_type
            }
            for trial_type in np.unique(self.session_data["TrialTypes"]).tolist()
        ]
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
        trial_attributes_keys = [
            {
                "session_id": session_id,
                "scan_id": scan_id,
                "trial_id": n,
                "attribute_name": attrib,
                "attribute_value": self.trial(n).attributes[attrib],
            }
            for n in range(bood_trial_offset, self.n_trials)
            for attrib in self.trial(n).attributes
            if self.trial(n).attributes[attrib]
        ]
        event_type_keys = [
            {"event_type": event_type}
            for event_type in set(
                event_type
                for n in range(bood_trial_offset, self.n_trials) # start at second trial
                for event_type in self.trial(n).events
            )
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
        if (
            prompt
            and dj.utils.user_choice("Proceed with new subject(s) insert?") != "yes"
        ):
            print("Canceled insert.")
            return

        # ----------------------------- Insert to schemas -----------------------------
        with session.Session.connection.transaction:
            session.Session.insert1(session_key, skip_duplicates=True)  # remove skip
            event.BehaviorRecording.insert1(behavior_recording_key, skip_duplicates=True)
            event.BehaviorRecording.File.insert1(behavior_recording_fp_key, skip_duplicates=True)
            trial.TrialType.insert(trial_type_keys, skip_duplicates=True)
            trial.Trial.insert(trial_keys, allow_direct_insert=True, skip_duplicates=True)
            trial.Trial.Attribute.insert(
                trial_attributes_keys, allow_direct_insert=True
            )
            event.EventType.insert(event_type_keys, skip_duplicates=True)
            event_keys = Trial.split_event_times_list(event_keys) #TR23: Split event times list if there are multiple values in a 'event_start_time' field
            event.Event.insert(
                event_keys, allow_direct_insert=True, ignore_extra_fields=True, skip_duplicates=True
            )  # ignore extra trial_id
            trial.TrialEvent.insert(event_keys, allow_direct_insert=True)

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
                "bpod_reward_oddball": self._states.get("Reward", [None])[0],
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
                "bpod_drinking": self._states.get("Drinking", [None])[0] if self._states.get("Drinking", [None])[0] is not None else None,
            }
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
