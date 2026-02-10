"""Ingest behavioral events from aux and bpod files.
The aux file is the only .h5 file in a scan directory.
The bpod file contains StimArenaMaster and is a .mat file."""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pywavesurfer import ws
from ..paths import get_imaging_root_data_dir, get_experiment_root_data_dir
from element_interface.utils import find_full_path
from adamacs.pipeline import event, trial, scan, model
from adamacs.ingest.bpod import Bpodfile
import warnings
import pathlib
import re
import pdb
import pandas as pd
import cv2
import subprocess
from pathlib import Path
try:
    import skvideo.io
    SKVIDEO_AVAILABLE = True
except ImportError:
    SKVIDEO_AVAILABLE = False

def demultiplex(auxdata, channels=5):
    """Demultiplex the digital data"""
    auxdata = auxdata.flatten()
    binary = [[int(x) for x in f'{x:0{channels}b}'] for x in auxdata]
    return np.array(binary, dtype=bool).T

def demultiplex_fast(digital_data, n_channels):
    """
    Fast demultiplexing of digital signals using vectorized numpy operations.
    143x faster than the original ibe.demultiplex function.
    
    Parameters:
    -----------
    digital_data : array-like
        Raw digital data to demultiplex
    n_channels : int
        Number of digital channels to extract
        
    Returns:
    --------
    numpy.ndarray
        2D array with shape (n_channels, n_samples) containing demultiplexed data
    """
    import numpy as np
    
    digital_data = np.asarray(digital_data, dtype=np.uint16)
    
    # Create bit positions for extraction (MSB-first order)
    bit_positions = np.arange(n_channels)[:, np.newaxis]
    
    # Extract all bits at once using broadcasting
    demultiplexed = (digital_data >> bit_positions) & 1
    
    return demultiplexed.astype(np.uint8)

def get_timestamps(data, sr, thr=1):
    """"""
    if data.dtype == 'bool':
        data = data > 0.5
    else:
        data = data - np.min(data) #TR23: 0base data
        data = np.abs(data) > thr #TR23: np.abs to cope with negative voltages
    
    diff = np.diff(data)
    idc = np.argwhere(diff != 0)[:, 0]
    timestamps = idc / sr
    return timestamps

def get_timestamps_auxcam(data, sr, thr=1):
    """
    Extract aux camera trigger timestamps using rising edges only.

    Unlike ``get_timestamps``, this avoids counting both on/off transitions.
    """
    data = np.asarray(data)
    if data.dtype == 'bool':
        diff = np.diff(data.astype(np.int8))
        idc = np.argwhere(diff > 0)[:, 0]
    else:
        diff = np.diff(data.astype(float))
        idc = np.argwhere(diff > thr)[:, 0]
    return idc / sr

def get_timestamps_robust_optitrack(data, sr, expected_frequency=241.0, thr=1):
    """
    Robust OptiTrack timestamp extraction with clock recovery.
    
    This function replaces get_timestamps() specifically for OptiTrack channels
    to handle undersampling issues that cause missed frame events.
    
    Key difference: Detects OptiTrack FRAMES (rising edges only) at ~241 Hz,
    not transitions (rising+falling) at ~482 Hz.
    
    Parameters:
    -----------
    data : array-like
        Digital signal data from OptiTrack channel
    sr : float
        Sampling rate in Hz
    expected_frequency : float, optional
        Expected OptiTrack frame rate in Hz (default: 241.0)
    thr : float, optional
        Threshold for digital signal detection (default: 1)
        
    Returns:
    --------
    numpy.ndarray
        Array of frame timestamps in seconds with missing frames reconstructed
    """
    
    # Step 1: Get rising edges only (actual OptiTrack frames)
    if data.dtype == 'bool':
        data_bool = data > 0.5
    else:
        data_processed = data - np.min(data)  # 0-base data
        data_bool = np.abs(data_processed) > thr
    
    diff = np.diff(data_bool.astype(int))
    rising_edge_indices = np.argwhere(diff == 1)[:, 0]  # Only rising edges (0->1)
    initial_timestamps = rising_edge_indices / sr
    
    if len(initial_timestamps) < 5:
        return initial_timestamps  # Not enough data for reconstruction
    
    # Step 2: Clock recovery and reconstruction     
    intervals = np.diff(initial_timestamps)
    expected_interval = 1.0 / expected_frequency  # ~4.15ms for 241 Hz
    
    # Estimate fundamental period from normal intervals (90th percentile)
    # Use 90% since we expect more regular timing with frame detection
    normal_intervals = intervals[intervals <= np.percentile(intervals, 90)]
    if len(normal_intervals) < 3:
        return initial_timestamps  # Can't estimate period reliably
    
    fundamental_period = np.median(normal_intervals) 
    
    # Gap threshold: intervals > 150% of expected frame period indicate missing frames
    gap_threshold = fundamental_period * 1.5  
    
    # Step 3: Reconstruct missing frames
    reconstructed = [initial_timestamps[0]]
    total_inserted = 0
    
    for i, interval in enumerate(intervals):
        current_time = initial_timestamps[i]
        
        if interval > gap_threshold:
            # Calculate number of missing frames
            n_expected = int(np.round(interval / fundamental_period))
            n_missing = max(0, n_expected - 1)
            
            # Insert missing frame timestamps
            for j in range(1, n_expected):
                interpolated_time = current_time + j * fundamental_period
                reconstructed.append(interpolated_time)
                total_inserted += 1
        
        # Add next observed frame
        reconstructed.append(initial_timestamps[i + 1])
    
    reconstructed_array = np.array(sorted(reconstructed))
    
    return reconstructed_array

def prepare_timestamps_optitrack(ts, session_key, scan_key, event_type):
    """
    Prepares OptiTrack timestamps as instantaneous events.
    Creates artificial end times 1ms after each start time.
    OptiTrack frames are individual events, not start/stop pairs like other channels.
    """
    # OptiTrack frames are instantaneous - create 1ms duration events
    ts_start = ts
    ts_end = ts + 0.001  # 1ms duration
    
    to_insert = [list(ts_start), list(ts_end)]  
    to_insert = [[session_key, scan_key, event_type, *i] for i in zip(*to_insert)]  # transposes the list to get rows/cols right
    
    # No need for the missing event check since OptiTrack frames are complete events
    
    return to_insert

def get_timestamps_from_plateaus(data, sr, threshold=0.2, min_duration=1000):
    """
    Find plateaus in a signal, round their average values to 0.1, and exclude plateaus around zero.

    Parameters:
    - data: The input signal (numpy array).
    - threshold: The maximum allowed deviation within a plateau and the range around zero to exclude.
    - min_duration: The minimum duration (in samples) for a segment to be considered a plateau.

    Returns:
    - A list of tuples, each tuple containing the start, end indices, and rounded average value of a plateau.
    """
    plateaus = []
    start_idx = None
    for i in range(1, len(data)):
        if start_idx is None and abs(data[i] - data[i - 1]) <= threshold:
            start_idx = i - 1  # potential start of a plateau
        elif start_idx is not None and abs(data[i] - data[i - 1]) > threshold:
            if i - start_idx >= min_duration:
                average_value = np.mean(data[start_idx:i])
                # Exclude plateaus around zero and round the average value
                if abs(average_value) > threshold:
                    rounded_value = round(average_value, 1)
                    plateaus.append((start_idx, i - 1, rounded_value))  # end of a plateau
            start_idx = None  # reset for next potential plateau

    # Check if the last segment is a plateau
    if start_idx is not None and len(data) - start_idx >= min_duration:
        average_value = np.mean(data[start_idx:])
        # Exclude plateaus around zero and round the average value
        if abs(average_value) > threshold:
            rounded_value = round(average_value, 1)
            plateaus.append((start_idx, len(data) - 1, rounded_value))

    idc = np.array([[t[0], t[1]] for t in plateaus]).flatten()
    timestamps = idc / sr
    return timestamps, plateaus


def extract_burst_onsets_offsets(diode_signal, sr, threshold=None, min_burst_duration=0.02, max_inter_pulse_interval=0.02):
    """
    Extract onsets and offsets of bursting light stimulus from a diode channel.
    Detects burst episodes composed of high-frequency short pulses.
    
    Parameters:
    -----------
    diode_signal : np.ndarray
        The analog signal from the diode channel
    sr : float
        Sampling rate in Hz
    threshold : float or None
        Threshold to detect individual pulses. If None, will use 20% between min and max.
    min_burst_duration : float
        Minimum duration (in seconds) for a burst episode to be considered valid
    max_inter_pulse_interval : float
        Maximum gap between pulses (in seconds) to be considered part of the same burst
    
    Returns:
    --------
    onsets : np.ndarray
        Timestamps (in seconds) of burst episode onsets 
    offsets : np.ndarray
        Timestamps (in seconds) of burst episode offsets
    """
    # 1. Threshold the signal to get individual pulses
    if threshold is None:
        threshold = np.min(diode_signal) + 0.2 * (np.max(diode_signal) - np.min(diode_signal))
    burst = diode_signal > threshold

    # 2. Find individual pulse transitions
    diff = np.diff(burst.astype(int))
    pulse_onset_idxs = np.where(diff == 1)[0] + 1  # 0->1: pulse starts
    pulse_offset_idxs = np.where(diff == -1)[0] + 1  # 1->0: pulse ends

    # 3. Handle edge cases (pulse at start/end)
    if burst[0]:
        pulse_onset_idxs = np.insert(pulse_onset_idxs, 0, 0)
    if burst[-1]:
        pulse_offset_idxs = np.append(pulse_offset_idxs, len(burst)-1)
    
    if len(pulse_onset_idxs) == 0 or len(pulse_offset_idxs) == 0:
        return np.array([]), np.array([])

    # 4. Group pulses into burst episodes based on inter-pulse intervals
    pulse_onsets_sec = pulse_onset_idxs / sr
    pulse_offsets_sec = pulse_offset_idxs / sr
    
    burst_episodes = []
    current_burst_start = pulse_onsets_sec[0]
    last_pulse_end = pulse_offsets_sec[0]
    
    for i in range(1, len(pulse_onsets_sec)):
        inter_pulse_gap = pulse_onsets_sec[i] - last_pulse_end
        
        if inter_pulse_gap > max_inter_pulse_interval:
            # Gap too large, end current burst and start new one
            current_burst_end = last_pulse_end
            burst_duration = current_burst_end - current_burst_start
            
            if burst_duration >= min_burst_duration:
                burst_episodes.append((current_burst_start, current_burst_end))
            
            # Start new burst
            current_burst_start = pulse_onsets_sec[i]
        
        last_pulse_end = pulse_offsets_sec[i]
    
    # Don't forget the last burst
    current_burst_end = last_pulse_end
    burst_duration = current_burst_end - current_burst_start
    if burst_duration >= min_burst_duration:
        burst_episodes.append((current_burst_start, current_burst_end))
    
    # 5. Extract onsets and offsets
    if len(burst_episodes) == 0:
        return np.array([]), np.array([])
    
    onsets = np.array([episode[0] for episode in burst_episodes])
    offsets = np.array([episode[1] for episode in burst_episodes])
    
    return onsets, offsets


def prepare_timestamps(ts, session_key, scan_key, event_type):
    """Prepares timestamps for insert with datajoint"""
    ts_chan_start = ts[0::2]
    ts_chan_stop = ts[1::2]
    
    to_insert = [list(ts_chan_start), list(ts_chan_stop)]  
    to_insert = [[session_key, scan_key, event_type, *i] for i in zip(*to_insert)]  # transposes the list to get rows/cols right
    if len(to_insert) != len(ts_chan_start):
        to_insert.append([session_key, scan_key, event_type, ts_chan_start[-1],ts_chan_start[-1]+1])
        print("WARNING: Last event in aux file is not closed. Adding a dummy event to close it.")

    return to_insert

def ingest_bpod(sessi, scansi, root_paths=get_imaging_root_data_dir(), aux_setup_type = "openfield",
                        verbose=False, include_raw_bpod_events=False, include_raw_bpod_states=False): #TR23: included scan key, included setupID
     
    if not verbose:
        warnings.filterwarnings('ignore')
    scan_key = (scan.Scan & f'scan_id = "{scansi}"').fetch('KEY')[0]
    bpod_path_relative = (scan.ScanPath & scan_key).fetch("path")[0]
    bpod_path_full = list(find_full_path(
        get_experiment_root_data_dir(), bpod_path_relative
        ).glob("*mat"))[0]
    bpod_object = Bpodfile(bpod_path_full)
    if "openfield" in aux_setup_type:
        bpod_object.ingest(
            sessi, scansi,
            include_raw_bpod_events=include_raw_bpod_events,
            include_raw_bpod_states=include_raw_bpod_states,
        )
    elif aux_setup_type == "bench2p_oddball":
        bpod_object.ingest_oddball(
            sessi, scansi,
            include_raw_bpod_events=include_raw_bpod_events,
            include_raw_bpod_states=include_raw_bpod_states,
        )
    elif aux_setup_type == "bench2p_Oddball_v2":
        bpod_object.ingest_oddball_v2(
            sessi, scansi,
            include_raw_bpod_events=include_raw_bpod_events,
            include_raw_bpod_states=include_raw_bpod_states,
        )
    elif aux_setup_type == "bench2p_Oddball_v3":
        bpod_object.ingest_oddball_v3(
            sessi, scansi,
            include_raw_bpod_events=include_raw_bpod_events,
            include_raw_bpod_states=include_raw_bpod_states,
        )
    elif aux_setup_type == "bench2p_lineartrack":
        bpod_object.ingest_linear_track(
            sessi, scansi,
            include_raw_bpod_events=include_raw_bpod_events,
            include_raw_bpod_states=include_raw_bpod_states,
        )
    elif aux_setup_type == "behavior_box":
        bpod_object.ingest_behavior_box(
            sessi, scansi,
            include_raw_bpod_events=include_raw_bpod_events,
            include_raw_bpod_states=include_raw_bpod_states,
        )


def summarize_raw_bpod_events(session_id=None, scan_id=None, key=None, limit=50):
    """
    Summarize raw Bpod events stored in event.Event with the raw_bpod_ prefix.
    Returns a dict with counts and time range.
    """
    from collections import Counter

    if key is None:
        key = {}
    if session_id:
        key = {**key, "session_id": session_id}
    if scan_id:
        key = {**key, "scan_id": scan_id}
    if not key:
        raise ValueError("Provide session_id/scan_id or a key dict")

    query = event.Event & key & 'event_type LIKE "raw_bpod_%"'
    event_types = list(query.fetch("event_type"))
    if not event_types:
        print("No raw_bpod_* events found for", key)
        return {"key": key, "counts": {}, "n_events": 0, "time_range": None}

    counts = Counter(event_types)
    times = query.fetch("event_start_time")
    time_range = (float(min(times)), float(max(times))) if len(times) else None

    print(f"Raw Bpod events for {key}: {len(event_types):,} events, {len(counts)} types")
    for event_type, count in counts.most_common(limit):
        print(f"  {event_type}: {count:,}")
    if time_range:
        print(f"Time range: {time_range[0]:.3f} -> {time_range[1]:.3f} sec")

    return {"key": key, "counts": dict(counts), "n_events": len(event_types), "time_range": time_range}


def summarize_bpod_metadata(session_id=None, scan_id=None, bpod_path=None, max_values=6):
    """
    Summarize Bpod TrialSettings metadata and interpret the experiment.
    This reads the .mat file and does not modify the database.
    """
    if bpod_path is None:
        if not scan_id:
            raise ValueError("Provide scan_id or bpod_path")
        scan_key = (scan.Scan & f'scan_id = "{scan_id}"').fetch('KEY')[0]
        bpod_path_relative = (scan.ScanPath & scan_key).fetch("path")[0]
        bpod_path = list(find_full_path(
            get_experiment_root_data_dir(), bpod_path_relative
        ).glob("*mat"))[0]
    bpod_object = Bpodfile(bpod_path)
    return bpod_object.summarize_metadata(
        session_id=session_id,
        scan_id=scan_id,
        max_values=max_values,
        print_output=True,
    )

        
def ingest_aux(session_key, scan_key, root_paths=get_imaging_root_data_dir(), aux_setup_type = "openfield",
                        verbose=False): #TR23: included scan key, included setupID
     
    if not verbose:
        warnings.filterwarnings('ignore')

    paths = [pathlib.Path(path) for path in root_paths]
    valid_paths = [p for p in paths if p.is_dir()]
    match_paths = []
    for p in valid_paths:
        # match_paths.extend(list(p.rglob(f'*{session_key}*')))
        match_paths.extend([d for d in p.glob(f'*{scan_key}*{session_key}*') if d.is_dir()]) #TR23: limit to dirs only #TR24: dont use recursive search
    
    n_aux = len(match_paths)
    if verbose:
        print(f'Number of aux-files found: {n_aux}')
        print(match_paths)

    scan_pattern = "scan.{8}"
    basenames = [x.name for x in match_paths]
    scan_keys = [re.findall(scan_pattern, x) for x in basenames]
    scan_basenames = [x for x in basenames if bool(re.search(scan_pattern, x))]
    
    # For now this is only supposed to work for 1 scan per session
    if len(scan_basenames) != 1:
        raise ValueError(f"Found more or less than 1 AUX file in {session_key}")
    
    aux_files = []
    for k in scan_basenames:
        curr_path = find_full_path(root_paths, k)
        aux_file_paths = [fp.as_posix() for fp in curr_path.glob('*.h5')]
        if len(aux_file_paths) != 1:
            raise ValueError(f"More or less than 1 aux_files found in {k}")
        curr_file = ws.loadDataFile(filename=aux_file_paths[0], format_string='double' )
        aux_files.append(curr_file)
        
    start_datetime = aux_files[0]['header']['ClockAtRunStart'][:, 0]
    start_datetime = [x for x in start_datetime]
    for idx, x in enumerate(start_datetime):
        if idx != 5:
            start_datetime[idx] = str(int(start_datetime[idx]))
        else:
            start_datetime[idx] = str(float(start_datetime[idx]))
    start_datetime = '-'.join(start_datetime)
    sweep_duration = aux_files[0]['header']['SweepDuration'][0][0] 

    recording_notes = ''

    event.BehaviorRecording.insert1({'session_id': session_key, 'scan_id': scan_key, 'recording_start_time': start_datetime, 'recording_duration': sweep_duration, 'recording_notes': recording_notes}, skip_duplicates=True)
    
    for p in aux_file_paths:
        event.BehaviorRecording.File.insert1([session_key, scan_key, p], skip_duplicates=True)
    for curr_aux in aux_files:
        sweep = [x for x in curr_aux.keys() if 'sweep' in x][0]

        sr = curr_aux['header']['AcquisitionSampleRate'][0][0]
        numberDI = len(curr_aux['header']['DIChannelNames'])
        timebase = np.arange(curr_aux[sweep]['analogScans'].shape[1]) / sr

        if aux_setup_type == "mini2p1_openfield" or aux_setup_type == "openfield":
            # DIGITAL SIGNALS
            digital_channels = demultiplex(curr_aux[sweep]['digitalScans'][0], numberDI)
            
            if numberDI == 6:           
                main_track_gate_chan = digital_channels[5]
                shutter_chan = digital_channels[4]
                mini2p_frame_chan = digital_channels[3] #TR23: switched Vol and Frame Chan!!
                mini2p_line_chan = digital_channels[2]
                optitrack_frame_chan = digital_channels[1]
                mini2p_HARP_gate = digital_channels[0]

            elif numberDI == 7:
                main_track_gate_chan = digital_channels[6]
                shutter_chan = digital_channels[5]
                mini2p_frame_chan = digital_channels[4] #TR23: switched Vol and Frame Chan!!
                mini2p_line_chan = digital_channels[3]
                optitrack_frame_chan = digital_channels[2]
                mini2p_HARP_gate = digital_channels[1]
                landmark_LED_chan = digital_channels[0] ##

                landmark_LED_chan[-1] = 0 ##
                
                
            main_track_gate_chan[-1] = 0  
            shutter_chan[-1] = 0
            mini2p_frame_chan[-1] = 0
            mini2p_line_chan[-1] = 0
            optitrack_frame_chan[-1] = 0
            mini2p_HARP_gate[-1] = 0



            """Calculate timestamps"""
            ts_main_track_gate_chan = get_timestamps(main_track_gate_chan, sr)
            ts_shutter_chan = get_timestamps(shutter_chan, sr)
            ts_mini2p_frame_chan = get_timestamps(mini2p_frame_chan, sr)
            # ts_mini2p_line_chan = get_timestamps(mini2p_line_chan, sr)
            ts_optitrack_frame_chan = get_timestamps_robust_optitrack(optitrack_frame_chan, sr)
            ts_mini2p_HARP_gate = get_timestamps(mini2p_HARP_gate, sr)
            if numberDI == 7:
                ts_landmark_LED_chan = get_timestamps(landmark_LED_chan, sr)
                


            """Analog signals"""
            cam_trigger = curr_aux[sweep]['analogScans'][0]
            bpod_trial_vis_chan = curr_aux[sweep]['analogScans'][1]
            bpod_reward1_chan = curr_aux[sweep]['analogScans'][2]
            bpod_tone_chan = curr_aux[sweep]['analogScans'][3]
            light_flash_chan = curr_aux[sweep]['analogScans'][4]
            
            # Check if diode channel exists (channel 5)
            stim_diode_chan = None
            if curr_aux[sweep]['analogScans'].shape[0] > 5:
                stim_diode_chan = curr_aux[sweep]['analogScans'][5]
                stim_diode_chan[-1] = 0
            
            cam_trigger[-1] = 0
            bpod_trial_vis_chan[-1] = 0
            bpod_reward1_chan[-1] = 0
            bpod_tone_chan[-1] = 0
            light_flash_chan[-1] = 0

            ts_cam_trigger = get_timestamps(cam_trigger, sr)
            ts_bpod_visual = get_timestamps(bpod_trial_vis_chan, sr)
            ts_bpod_reward = get_timestamps(bpod_reward1_chan, sr)
            ts_bpod_tone = get_timestamps(bpod_tone_chan, sr)
            ts_light_flash =  get_timestamps(light_flash_chan, sr)
            
            # Extract diode burst episodes if channel exists
            if stim_diode_chan is not None:
                ts_stim_diode_onsets, ts_stim_diode_offsets = extract_burst_onsets_offsets(stim_diode_chan, sr)
                # Combine onsets and offsets for standard event format
                if len(ts_stim_diode_onsets) > 0 and len(ts_stim_diode_offsets) > 0:
                    ts_stim_diode = np.column_stack([ts_stim_diode_onsets, ts_stim_diode_offsets]).flatten()
                else:
                    ts_stim_diode = np.array([])
            else:
                ts_stim_diode = np.array([])
            
            # Insert timestamps into tables 

            # event_types = ['main_track_gate', 'HARP_gate', 'shutter',  'mini2p_frames', 'mini2p_lines', 'mini2p_volumes', 'aux_cam', 'arena_LED',
            #             'aux_bpod_visual', 'aux_bpod_reward', 'aux_bpod_tone']
            
            
            event_types = {
                'main_track_gate': ts_main_track_gate_chan,
                'HARP_gate': ts_mini2p_HARP_gate,
                'arena_LED': ts_light_flash,
                'shutter': ts_shutter_chan,
                'mini2p_frames': ts_mini2p_frame_chan,
                # 'mini2p_lines': ts_mini2p_line_chan,
                'optitrack_frames': ts_optitrack_frame_chan,
                'aux_cam': ts_cam_trigger,
                'aux_bpod_visual': ts_bpod_visual,
                'aux_bpod_reward': ts_bpod_reward,
                'aux_bpod_tone': ts_bpod_tone
            }
            
            # Add stim_diode events if they exist
            if len(ts_stim_diode) > 0:
                event_types['stim_diode'] = ts_stim_diode
            if numberDI == 7:
                event_types['landmark_LED'] = ts_landmark_LED_chan
                # event_types['aux_bonsai_vis'] = ts_bonsai_vis
                        
        elif aux_setup_type == "mini2p2_openfield":
            # DIGITAL SIGNALS (DI0..DI4: MainTrigger, Shutter, FrameClock, LineClock, VolumeClock)
            digital_channels = demultiplex(curr_aux[sweep]['digitalScans'][0], numberDI)
            
            if numberDI == 5:
                main_track_gate_chan = digital_channels[4]
                shutter_chan = digital_channels[3]
                mini2p_frame_chan = digital_channels[2]
                mini2p_line_chan = digital_channels[1]
                mini2p_vol_chan = digital_channels[0]
                
            main_track_gate_chan[-1] = 0  
            shutter_chan[-1] = 0
            mini2p_frame_chan[-1] = 0
            mini2p_line_chan[-1] = 0
            mini2p_vol_chan[-1] = 0

            """Calculate timestamps"""
            ts_main_track_gate_chan = get_timestamps(main_track_gate_chan, sr)
            ts_shutter_chan = get_timestamps(shutter_chan, sr)
            ts_mini2p_frame_chan = get_timestamps(mini2p_frame_chan, sr)
            # ts_mini2p_line_chan = get_timestamps(mini2p_line_chan, sr)
            # ts_mini2p_vol_chan = get_timestamps(mini2p_vol_chan, sr)

            """Analog signals (AI0..AI3: CameraTriggerIn, Visual, Reward, Tone)"""
            cam_trigger = curr_aux[sweep]['analogScans'][0]
            bpod_trial_vis_chan = curr_aux[sweep]['analogScans'][1]
            bpod_reward1_chan = curr_aux[sweep]['analogScans'][2]
            bpod_tone_chan = curr_aux[sweep]['analogScans'][3]
            
            cam_trigger[-1] = 0
            bpod_trial_vis_chan[-1] = 0
            bpod_reward1_chan[-1] = 0
            bpod_tone_chan[-1] = 0

            ts_cam_trigger = get_timestamps(cam_trigger, sr)
            ts_bpod_visual = get_timestamps(bpod_trial_vis_chan, sr)
            ts_bpod_reward = get_timestamps(bpod_reward1_chan, sr)
            ts_bpod_tone = get_timestamps(bpod_tone_chan, sr)
            
            # Insert timestamps into tables 
            event_types = {
                'main_track_gate': ts_main_track_gate_chan,
                'shutter': ts_shutter_chan,
                'mini2p_frames': ts_mini2p_frame_chan,
                # 'mini2p_lines': ts_mini2p_line_chan,
                # 'mini2p_volumes': ts_mini2p_vol_chan,
                'aux_cam': ts_cam_trigger,
                'aux_bpod_visual': ts_bpod_visual,
                'aux_bpod_reward': ts_bpod_reward,
                'aux_bpod_tone': ts_bpod_tone
            }
                        
        elif aux_setup_type == "headfixed": #TR23 - HEADFIXED MINI2p - #mini2p01 - needs to be set in scan schema! Taken from userfunction_consolidate_files argument
            print("not done")
        elif aux_setup_type == "bench2p" or aux_setup_type == "bench2p_kine": #TR23 - HEADFIXED Bench2p - #bench2p - needs to be set in scan schema! Taken from userfunction_consolidate_files argument
            
            # LOAD STIMINFO
            for k in scan_basenames:
                stim_file_paths = [fp.as_posix() for fp in curr_path.glob('*bonsai_stimulus_events*.csv')]
                if len(stim_file_paths) != 1:
                    # raise ValueError(f"More or less than 1 stim_files found in {k}")
                    print(f"More or less than 1 stim_files found in {k} - not extracting stim IDs")
                    vis_stim_event_list = []
                else:
                    # Load the csv file
                    df = pd.read_csv(stim_file_paths[0])
                    # Extract the third column
                    vis_stim_event_list = df['Value']
            
            # DIGITAL SIGNALS
            digital_channels = demultiplex(curr_aux[sweep]['digitalScans'][0], numberDI)
            main_track_gate_chan = digital_channels[4]
            shutter_chan = digital_channels[3]
            bench2p_frame_chan = digital_channels[2]
            bench2p_line_chan = digital_channels[1]
            bench2p_vol_chan = digital_channels[0]

            main_track_gate_chan[-1] = 0  # TR23 - to partially save truncated recordings, set all last samples to zero
            shutter_chan[-1] = 0
            bench2p_line_chan[-1] = 0
            bench2p_frame_chan[-1] = 0
            bench2p_vol_chan[-1] = 0


            """Calculate timestamps"""
            ts_main_track_gate_chan = get_timestamps(main_track_gate_chan, sr)
            ts_shutter_chan = get_timestamps(shutter_chan, sr)
            ts_bench2p_frame_chan = get_timestamps(bench2p_frame_chan, sr)
            # ts_bench2p_line_chan = get_timestamps(bench2p_line_chan, sr)
            ts_bench2p_vol_chan = get_timestamps(bench2p_vol_chan, sr)


            """Analog signals"""
            cam_trigger = curr_aux[sweep]['analogScans'][2]
            bonsai_vis_chan = curr_aux[sweep]['analogScans'][0]
            # bpod_speed_chan = curr_aux[sweep]['analogScans'][1]
            
            cam_trigger[-1] = np.min(cam_trigger) 
            bonsai_vis_chan[-1] = np.min(bonsai_vis_chan)
            # bpod_speed_chan[-1] = 0

            ts_cam_trigger = get_timestamps(cam_trigger, sr)
            ts_bonsai_vis, ts_bonsai_vis_plateau_values = get_timestamps_from_plateaus(bonsai_vis_chan, sr) #TR23: changed to get_timestamps_from_plateaus to cope with non-zero ITI aux files
            # ts_bpod_speed = get_timestamps(bpod_speed_chan, sr) #TR23: Data channel! Not Event channel!
            
            # Insert timestamps into tables             
            
            event_types = {
                'main_track_gate': ts_main_track_gate_chan,
                'shutter': ts_shutter_chan,
                'bench2p_frames': ts_bench2p_frame_chan,
                # 'bench2p_lines': ts_bench2p_line_chan,
                'bench2p_volumes': ts_bench2p_vol_chan,
                'aux_cam': ts_cam_trigger,
                # 'aux_bonsai_vis': ts_bonsai_vis,
            }
            
            j = 0
            for stim_event in vis_stim_event_list:  
                event_types[stim_event] = []
            for i, stim_event in enumerate(vis_stim_event_list):  
                event_types[stim_event] = np.append(event_types[stim_event], ts_bonsai_vis[j:j+2])
                j += 2
            
            if len(vis_stim_event_list) != ts_bonsai_vis.size / 2:
                print('Aux-File und StimLog have not the same number of stimulus onsets! CHECK THAT!')
            #     print('Attempting repair - THIS IS A HACK! ARTIFICIALLY INTRODUCING STIMULUS ENDINGS IN FILE! MAKE SURE TO CRRECT THAT DURING ACQ!')
                
            #     ITI = np.sort(np.unique(np.round(np.diff(ts_bonsai_vis))))[1] #find stim duration
            #     DUR = np.sort(np.unique(np.round(np.diff(ts_bonsai_vis))))[0] #find ITI duration
                
            #     block_edges = np.where(np.diff(ts_bonsai_vis)>ITI + 1) #find index of stim block edges
            #     ts_bonsai_vis[block_edges]+ITI #add ITI duration
                
            #     on_off_values_insert = np.array(list(zip(ts_bonsai_vis[block_edges[0]]+DUR, ts_bonsai_vis[block_edges[0]]+DUR + ITI)))
                
            #     ts_bonsai_vis_corrected = ts_bonsai_vis
                
            #     index = block_edges[0] + 1
            #     for value in on_off_values_insert:
            #             ts_bonsai_vis_corrected = np.insert(ts_bonsai_vis_corrected, index, value)
            #             index += len(value)

                 
                # raise ValueError(f"Aux-File und StimLog have not the same number of stimulus onsets!{k}")
                   
        elif aux_setup_type == "bench2p_lineartrack": #TR23 - HEADFIXED Bench2p - #bench2p - needs to be set in scan schema! Taken from userfunction_consolidate_files argument
                    
                    # LOAD STIMINFO
                    for k in scan_basenames:
                        stim_file_paths = [fp.as_posix() for fp in curr_path.glob('*bonsai_stimulus_log*.csv')]
                        if len(stim_file_paths) != 1:
                            # raise ValueError(f"More or less than 1 stim_files found in {k}")
                            print(f"More or less than 1 stim_files found in {k} - not extracting stim IDs")
                            vis_stim_event_list = []
                        else:
                            # Load the csv file AND CLEAN UP

                            # Prepare to collect the extracted strings with modifications
                            extracted_data_modified = []

                            # Regular expression to match strings enclosed by "((" and "))"
                            pattern = re.compile(r'\(\((.*?)\)\)')

                            # Flag to check if the previous line was "HARP_acquiring:ON"
                            acquiring_on = False

                            # Read the file and extract the strings
                            with open(stim_file_paths[0], 'r') as file:
                                # Skip the header line
                                next(file)
                                for line in file: #to dump all csv data that occured before the master trigger came
                                    if "Master_Trigger:ON" in line:
                                        acquiring_on = True
                                        continue  # Skip to the next line
                                    if "Master_Trigger:OFF" in line:
                                        acquiring_on = False
                                        continue  # Skip to the next line

                                    if acquiring_on:
                                        # Find all matches in the current line
                                        matches = pattern.findall(line)
                                        # For each match found, replace commas with semicolons and remove brackets
                                        for match in matches:
                                            modified_string = match.replace(',', ';').replace('(', '').replace(')', '')
                                            extracted_data_modified.append(modified_string)

                            # Convert the list of modified extracted strings into a DataFrame
                            df = pd.DataFrame(extracted_data_modified, columns=['Value'])

                            # Extract the event column
                            vis_stim_event_list = df['Value']
                    
                    # DIGITAL SIGNALS
                    digital_channels = demultiplex(curr_aux[sweep]['digitalScans'][0], numberDI)
                    main_track_gate_chan = digital_channels[4]
                    shutter_chan = digital_channels[3]
                    bench2p_frame_chan = digital_channels[2]
                    bench2p_line_chan = digital_channels[1]
                    bench2p_vol_chan = digital_channels[0]

                    main_track_gate_chan[-1] = 0  # TR23 - to partially save truncated recordings, set all last samples to zero
                    shutter_chan[-1] = 0
                    bench2p_line_chan[-1] = 0
                    bench2p_frame_chan[-1] = 0
                    bench2p_vol_chan[-1] = 0


                    """Calculate timestamps"""
                    ts_main_track_gate_chan = get_timestamps(main_track_gate_chan, sr)
                    ts_shutter_chan = get_timestamps(shutter_chan, sr)
                    ts_bench2p_frame_chan = get_timestamps(bench2p_frame_chan, sr)
                    # ts_bench2p_line_chan = get_timestamps(bench2p_line_chan, sr)
                    ts_bench2p_vol_chan = get_timestamps(bench2p_vol_chan, sr)


                    """Analog signals"""
                    cam_trigger = curr_aux[sweep]['analogScans'][2]
                    lick_chan = curr_aux[sweep]['analogScans'][3]
                    trial_chan = curr_aux[sweep]['analogScans'][4]

                    # bpod_speed_chan = curr_aux[sweep]['analogScans'][1]
                    
                    cam_trigger[-1] = np.min(cam_trigger) 
                    lick_chan[-1] = np.min(lick_chan) 
                    trial_chan[-1] = np.min(trial_chan)
                    # bpod_speed_chan[-1] = 0

                    ts_cam_trigger = get_timestamps(cam_trigger, sr)
                    ts_lick = get_timestamps(lick_chan, sr) 
                    ts_trial = get_timestamps(trial_chan, sr) 

                    # ts_bpod_speed = get_timestamps(bpod_speed_chan, sr) #TR23: Data channel! Not Event channel!
                    
                    # Insert timestamps into tables             
                    
                    event_types = {
                        'main_track_gate': ts_main_track_gate_chan,
                        'shutter': ts_shutter_chan,
                        'bench2p_frames': ts_bench2p_frame_chan,
                        # 'bench2p_lines': ts_bench2p_line_chan,
                        'bench2p_volumes': ts_bench2p_vol_chan,
                        'aux_cam': ts_cam_trigger,
                        'aux_lick': ts_lick,
                        'aux_trial': ts_trial,
                        # 'aux_bonsai_vis': ts_bonsai_vis,
                    }
                    
                    j = 0
                    for stim_event in vis_stim_event_list:  
                        event_types[stim_event] = []
                    for i, stim_event in enumerate(vis_stim_event_list):  
                        event_types[stim_event] = np.append(event_types[stim_event], ts_trial[j:j+2])
                        j += 2
                    
                    if len(vis_stim_event_list) != ts_trial.size / 2:
                        print('Aux-File und StimLog have not the same number of stimulus onsets! CHECK THAT!')  
                   
        elif aux_setup_type == "bench2p_oddball" or aux_setup_type == "bench2p_Oddball_V2": #TR23 - HEADFIXED Bench2p - #bench2p - needs to be set in scan schema! Taken from userfunction_consolidate_files argument
                    
                    # LOAD STIMINFO
                    for k in scan_basenames:
                        stim_file_paths = [fp.as_posix() for fp in curr_path.glob('*bonsai_stimulus_log*.csv')]
                        if len(stim_file_paths) != 1:
                            # raise ValueError(f"More or less than 1 stim_files found in {k}")
                            print(f"More or less than 1 stim_files found in {k} - not extracting stim IDs")
                            vis_stim_event_list = []
                        else:
                            # Load the csv file AND CLEAN UP

                            # Prepare to collect the extracted strings with modifications
                            extracted_data_modified = []

                            # Regular expression to match strings enclosed by "((" and "))"
                            pattern = re.compile(r'\(\((.*?)\)\)')

                            # Flag to check if the previous line was "HARP_acquiring:ON"
                            acquiring_on = False

                            # Read the file and extract the strings
                            with open(stim_file_paths[0], 'r') as file:
                                # Skip the header line
                                next(file)
                                for line in file: #to dump all csv data that occured before the master trigger came
                                    if aux_setup_type == "bench2p_oddball":
                                        if "HARP_acquiring:ON" in line:
                                            acquiring_on = True
                                            continue  # Skip to the next line
                                    else:
                                        acquiring_on = True

                                    if acquiring_on:
                                        # Find all matches in the current line
                                        matches = pattern.findall(line)
                                        # For each match found, replace commas with semicolons and remove brackets
                                        for match in matches:
                                            modified_string = match.replace(',', ';').replace('(', '').replace(')', '')
                                            extracted_data_modified.append(modified_string)

                            # Convert the list of modified extracted strings into a DataFrame
                            df = pd.DataFrame(extracted_data_modified, columns=['Value'])

                            # Extract the event column
                            vis_stim_event_list = df['Value']
                    
                    # DIGITAL SIGNALS
                    if aux_setup_type == "bench2p_oddball":
                        digital_channels = demultiplex(curr_aux[sweep]['digitalScans'][0], numberDI)
                        main_track_gate_chan = digital_channels[4]
                        shutter_chan = digital_channels[3]
                        bench2p_frame_chan = digital_channels[2]
                        bench2p_line_chan = digital_channels[1]
                        bench2p_vol_chan = digital_channels[0]
                    elif aux_setup_type == "bench2p_Oddball_V2":
                        digital_channels = demultiplex(curr_aux[sweep]['digitalScans'][0], numberDI)
                        main_track_gate_chan = digital_channels[5]
                        shutter_chan = digital_channels[4]
                        bench2p_frame_chan = digital_channels[3]
                        bench2p_line_chan = digital_channels[2]
                        bench2p_vol_chan = digital_channels[1]
                        stim_vis2_chan = digital_channels[0] #TR23: not used in V2
                        stim_vis2_chan[-1] = 0

                    main_track_gate_chan[-1] = 0  # TR23 - to partially save truncated recordings, set all last samples to zero
                    shutter_chan[-1] = 0
                    bench2p_line_chan[-1] = 0
                    bench2p_frame_chan[-1] = 0
                    bench2p_vol_chan[-1] = 0
                    

                    """Calculate timestamps"""
                    ts_main_track_gate_chan = get_timestamps(main_track_gate_chan, sr)
                    ts_shutter_chan = get_timestamps(shutter_chan, sr)
                    ts_bench2p_frame_chan = get_timestamps(bench2p_frame_chan, sr)
                    # ts_bench2p_line_chan = get_timestamps(bench2p_line_chan, sr)
                    ts_bench2p_vol_chan = get_timestamps(bench2p_vol_chan, sr)
                    if aux_setup_type == "bench2p_Oddball_V2":
                        ts_stim_vis2_chan = get_timestamps(stim_vis2_chan, sr)
                    


                    """Analog signals"""
                    if aux_setup_type == "bench2p_oddball":
                        cam_trigger = curr_aux[sweep]['analogScans'][2]
                        stim_vis_chan = curr_aux[sweep]['analogScans'][3]
                    elif aux_setup_type == "bench2p_Oddball_V2":
                        cam_trigger = curr_aux[sweep]['analogScans'][2]
                        stim_vis_chan = curr_aux[sweep]['analogScans'][0]
                    # bpod_speed_chan = curr_aux[sweep]['analogScans'][1]                    cam_trigger = curr_aux[sweep]['analogScans'][2]
                        lick_chan = curr_aux[sweep]['analogScans'][3]
                        trial_chan = curr_aux[sweep]['analogScans'][4]
                        reward_chan = curr_aux[sweep]['analogScans'][5]

                    # bpod_speed_chan = curr_aux[sweep]['analogScans'][1]
                    
                    cam_trigger[-1] = np.min(cam_trigger)
                    stim_vis_chan[-1] = np.min(stim_vis_chan) 

                    
                    if aux_setup_type == "bench2p_Oddball_V2":
                        lick_chan[-1] = np.min(lick_chan) 
                        trial_chan[-1] = np.min(trial_chan)
                        reward_chan[-1] = np.min(reward_chan)
                        
                        
                    ts_cam_trigger = get_timestamps(cam_trigger, sr)
                    
                    if aux_setup_type == "bench2p_Oddball_V2":
                        ts_reward = get_timestamps(reward_chan, sr)
                        ts_lick = get_timestamps(lick_chan, sr) 
                        ts_trial = get_timestamps(trial_chan, sr)
                        
                    # zero stim channel before main track gate -TR24
                    deadtime = 1 # time in s blocked for event detection after main track gate

                    if aux_setup_type == "bench2p_Oddball_V2":
                        deadtime = 0 # time in s blocked for event detection after main track gate

                    stim_vis_chan[:int(ts_main_track_gate_chan[0]*sr + deadtime * sr)] = 0

                    # bpod_speed_chan = curr_aux[sweep]['analogScans'][1]
                    
    
                    ts_stim_vis, ts_stim_vis_plateau_values = get_timestamps_from_plateaus(stim_vis_chan, sr) #TR23: changed to get_timestamps_from_plateaus to cope with non-zero ITI aux files
                    # ts_bpod_speed = get_timestamps(bpod_speed_chan, sr) #TR23: Data channel! Not Event channel!
                    
                    # Insert timestamps into tables             
                    
                    if aux_setup_type == "bench2p_oddball":
                        event_types = {
                            'main_track_gate': ts_main_track_gate_chan,
                            'shutter': ts_shutter_chan,
                            'bench2p_frames': ts_bench2p_frame_chan,
                            # 'bench2p_lines': ts_bench2p_line_chan,
                            'bench2p_volumes': ts_bench2p_vol_chan,
                            'aux_cam': ts_cam_trigger,
                            'aux_bonsai_vis': ts_stim_vis,
                        }
                    elif aux_setup_type == "bench2p_Oddball_V2":
                        event_types = {
                            'main_track_gate': ts_main_track_gate_chan,
                            'shutter': ts_shutter_chan,
                            'bench2p_frames': ts_bench2p_frame_chan,
                            # 'bench2p_lines': ts_bench2p_line_chan,
                            'bench2p_volumes': ts_bench2p_vol_chan,
                            'aux_cam': ts_cam_trigger,
                            'aux_bonsai_vis': ts_stim_vis,
                            'aux_bonsai_vis2': ts_stim_vis2_chan,
                            'aux_lick': ts_lick,
                            'aux_trial': ts_trial,
                            'aux_reward': ts_reward                            
                        }
                    
                    j = 0
                    for stim_event in vis_stim_event_list:  
                        event_types[stim_event] = []
                    for i, stim_event in enumerate(vis_stim_event_list):  
                        event_types[stim_event] = np.append(event_types[stim_event], ts_stim_vis[j:j+2])
                        j += 2
                    
                    if len(vis_stim_event_list) != ts_stim_vis.size / 2:
                        print('Aux-File und StimLog have not the same number of stimulus onsets! CHECK THAT!')  

        elif aux_setup_type == "bench2p_Oddball_v3":
                    event_types = ingest_bench2p_oddball_v3(scan_basenames, curr_path, curr_aux, sweep, sr)

        elif aux_setup_type == "bench2p_SP": #TR24 - HEADFIXED Bench2p - #bench2p - needs to be set in scan schema! Taken from userfunction_consolidate_files argument
                    
                    # LOAD STIMINFO
                    for k in scan_basenames:
                        stim_file_paths = [fp.as_posix() for fp in curr_path.glob('*bonsai_stimulus_log*.csv')]
                        if len(stim_file_paths) != 1:
                            # raise ValueError(f"More or less than 1 stim_files found in {k}")
                            print(f"More or less than 1 stim_files found in {k} - not extracting stim IDs")
                            vis_stim_event_list = []
                        else:
                            with open(stim_file_paths[0], 'r') as file:
                                raw_lines = file.readlines()

                            converted_lines = []
                            for line in raw_lines[1:]:  # Skip the header line
                                parts = line.strip().split(',')
                                # Parts[0]: Frame, Parts[1]: Timestamp, Parts[2:]: all remaining should be converted to semicolon-separated
                                new_line = parts[:2] + [';'.join(parts[2:])]
                                converted_lines.append(new_line)

                            # Create a new DataFrame from the converted lines
                            df = pd.DataFrame(converted_lines, columns=['Frame', 'Timestamp', 'Value'])
                            vis_stim_event_list = df['Value']

                    
                    # DIGITAL SIGNALS
                    digital_channels = demultiplex(curr_aux[sweep]['digitalScans'][0], numberDI)
                    main_track_gate_chan = digital_channels[5]
                    shutter_chan = digital_channels[4]
                    bench2p_frame_chan = digital_channels[3]
                    bench2p_line_chan = digital_channels[2]
                    bench2p_vol_chan = digital_channels[1]
                    stim_vis_chan = digital_channels[0]

                    main_track_gate_chan[-1] = 0  # TR23 - to partially save truncated recordings, set all last samples to zero
                    shutter_chan[-1] = 0
                    bench2p_line_chan[-1] = 0
                    bench2p_frame_chan[-1] = 0
                    bench2p_vol_chan[-1] = 0
                    stim_vis_chan[-1] = 0

                    """Calculate timestamps"""
                    ts_main_track_gate_chan = get_timestamps(main_track_gate_chan, sr)
                    ts_shutter_chan = get_timestamps(shutter_chan, sr)
                    ts_bench2p_frame_chan = get_timestamps(bench2p_frame_chan, sr)
                    # ts_bench2p_line_chan = get_timestamps(bench2p_line_chan, sr)
                    ts_bench2p_vol_chan = get_timestamps(bench2p_vol_chan, sr)


                    """Analog signals"""
                    trial_type = curr_aux[sweep]['analogScans'][0]
                    cam_trigger = curr_aux[sweep]['analogScans'][2]
                    fake_reward = curr_aux[sweep]['analogScans'][3]
                    reward = curr_aux[sweep]['analogScans'][4]
                    licks = curr_aux[sweep]['analogScans'][5]
                    hifi = curr_aux[sweep]['analogScans'][6]

                    

                    # zero stim channel before main track gate -TR24
                    deadtime = 1 # time in s blocked for event detection after main track gate

                    stim_vis_chan[:int(ts_main_track_gate_chan[0]*sr + deadtime * sr)] = 0

                    # bpod_speed_chan = curr_aux[sweep]['analogScans'][1]
                    
                    cam_trigger[-1] = np.min(cam_trigger) 
                    stim_vis_chan[-1] = np.min(stim_vis_chan)
                    # bpod_speed_chan[-1] = 0

                    ts_cam_trigger = get_timestamps(cam_trigger, sr)
                    ts_fake_reward = get_timestamps(fake_reward, sr)
                    ts_reward = get_timestamps(reward, sr)
                    ts_licks = get_timestamps(licks, sr)
                    ts_hifi = get_timestamps(hifi, sr)

                    ts_stim_vis  = get_timestamps(stim_vis_chan, sr)
                    ts_trial_type, trial_type_val = get_timestamps_from_plateaus(trial_type, sr, min_duration=10) #TR23: changed to get_timestamps_from_plateaus to cope with non-zero ITI aux files
                    
                    
                    # ts_bpod_speed = get_timestamps(bpod_speed_chan, sr) #TR23: Data channel! Not Event channel!
                    
                    # Insert timestamps into tables             
                    
                    event_types = {
                        'main_track_gate': ts_main_track_gate_chan,
                        'shutter': ts_shutter_chan,
                        'bench2p_frames': ts_bench2p_frame_chan,
                        # 'bench2p_lines': ts_bench2p_line_chan,
                        'bench2p_volumes': ts_bench2p_vol_chan,
                        'aux_cam': ts_cam_trigger,
                        'aux_fake_reward': ts_fake_reward,
                        'aux_reward': ts_reward,
                        'aux_licks': ts_licks,
                        'aux_hifi': ts_hifi,
                        'aux_stim_vis': ts_stim_vis,
                        'aux_trial_type': ts_trial_type,
                        
                        # 'aux_bonsai_vis': ts_bonsai_vis,
                    }
                    
                    j = 0
                    for stim_event in vis_stim_event_list:  
                        event_types[stim_event] = []
                    for i, stim_event in enumerate(vis_stim_event_list):  
                        event_types[stim_event] = np.append(event_types[stim_event], ts_stim_vis[j:j+2])
                        j += 2
                    
                    if len(vis_stim_event_list) != ts_stim_vis.size / 2:
                        print('Aux-File und StimLog have not the same number of stimulus onsets! CHECK THAT!')  


        elif aux_setup_type == "macroscope": #TR23 -  HEADFIXED Macroscope - #macroscope - needs to be set in scan schema! Connot be taken from userfunction_consolidate_files
            print("not done")
        
        
        
        # Insert into tables
        for e in event_types:
            event.EventType.insert1({'event_type': e, 'event_type_description': ''}, skip_duplicates=True)
            
        for event_type, timestamps in event_types.items():
            # Handle OptiTrack frames as instantaneous events
            if event_type == 'optitrack_frames' or (aux_setup_type == "bench2p_Oddball_v3" and event_type == 'aux_cam'):
                to_insert = prepare_timestamps_optitrack(timestamps, session_key, scan_key, event_type)
            else:
                to_insert = prepare_timestamps(timestamps, session_key, scan_key, event_type)
            event.Event.insert(to_insert, skip_duplicates=True, allow_direct_insert=True)
        
 
        
def get_and_ingest_trial_times(scan_key, aux_setup_type):                   
    session_key = (scan.Scan & f'scan_id = "{scan_key}"').fetch('session_id')[0]
    scan_key_key = (scan.Scan & f'scan_id = "{scan_key}"').fetch('KEY')[0]         
    # print('TODO: Update aux_setup_strings in get_and_ingest_trial_times !')

    if  aux_setup_type == "bench2p" or aux_setup_type == "bench2p_kine":
        
        # Extract Trials

        # stims_per_trial = len(set((event.Event & scan_key_key & 'event_type LIKE "%;%"').fetch("event_type")))
        all_stims = (event.Event & scan_key_key & 'event_type LIKE "%;%"').fetch("event_type")
        # trial_stims = {x: list(all_stims).count(x) for x in all_stims}
        trials = set([list(all_stims).count(x) for x in all_stims])
        stims_per_trial = int(len(all_stims) / min(trials))

        
        trial_start_edges = (event.Event & scan_key_key & 'event_type LIKE "%;%"').fetch("event_start_time",order_by = "event_start_time")[::stims_per_trial] 
        trial_end_edges = (event.Event & scan_key_key & 'event_type LIKE "%;%"').fetch("event_end_time",order_by = "event_end_time")[stims_per_trial-1::stims_per_trial]     
            
        trial_event_name = (event.Event & scan_key_key & 'event_type LIKE "%;%"').fetch("event_type")[0].split(':')[0]
        trial.TrialType().insert1({'trial_type': trial_event_name, 'trial_type_description': 'Stimulus nomenclature: Type; Class; Azimuth; Elevation; Size; Orientation; Spatial Frequency; Temporal Frequency'}, skip_duplicates=True)
                
        for trialnum in enumerate(trial_start_edges):
            trial.Trial.insert1({'session_id': session_key, 'scan_id': scan_key, 'trial_id': trialnum[0]+1, 'trial_type': trial_event_name, 'trial_start_time': trial_start_edges[trialnum[0]], 'trial_stop_time': trial_end_edges[trialnum[0]]},  allow_direct_insert=True, skip_duplicates=True)
            
            # generate query object from joint Trial Event table
            TrialEvent_query_keys = (event.Event * trial.Trial & scan_key_key & f'event_type LIKE "%;%"' & f'event_start_time <= "{trial_end_edges[trialnum[0]]}"' & f'event_end_time >= "{trial_start_edges[trialnum[0]]}"' & f'trial_id= "{trialnum[0] + 1}"') #.fetch(format = "frame", order_by = "event_start_time")
            
            # do server-side insert - fetch does not work. The number key seems to be rounded.
            trial.TrialEvent.insert(TrialEvent_query_keys,  allow_direct_insert=True, skip_duplicates=True, ignore_extra_fields=True)

    if  aux_setup_type == "bench2p_oddball":

        # Extract Trials

        # stims_per_trial = len(set((event.Event & scan_key_key & 'event_type LIKE "%;%"').fetch("event_type")))
        all_stims = (event.Event & scan_key_key & 'event_type LIKE "%;%"').fetch("event_type")
        all_stims = ['; '.join(item.split('; ')[:1]) for item in all_stims]
        # trial_stims = {x: list(all_stims).count(x) for x in all_stims}
        trials = set([list(all_stims).count(x) for x in all_stims])
        stims_per_trial = int(len(all_stims) / min(trials))

        
        trial_start_edges = (event.Event & scan_key_key & 'event_type LIKE "%;%"').fetch("event_start_time",order_by = "event_start_time")[::stims_per_trial] 
        trial_end_edges = (event.Event & scan_key_key & 'event_type LIKE "%;%"').fetch("event_end_time",order_by = "event_end_time")[stims_per_trial-1::stims_per_trial]     
            
        # trial_event_name = (event.Event & scan_key_key & 'event_type LIKE "%;%"').fetch("event_type")[0].split(':')[0]
        trial_event_name = 'Oddball'
        trial.TrialType().insert1({'trial_type': trial_event_name, 'trial_type_description': 'Stimulus nomenclature: Type; Class; Azimuth; Elevation; Size; Orientation; Spatial Frequency; Temporal Frequency'}, skip_duplicates=True)
                
        for trialnum in enumerate(range(min(trials))): #enumerate(trial_start_edges):
            trial.Trial.insert1({'session_id': session_key, 'scan_id': scan_key, 'trial_id': trialnum[0]+1, 'trial_type': trial_event_name, 'trial_start_time': trial_start_edges[trialnum[0]], 'trial_stop_time': trial_end_edges[trialnum[0]]},  allow_direct_insert=True, skip_duplicates=True)
            
            # generate query object from joint Trial Event table
            TrialEvent_query_keys = (event.Event * trial.Trial & scan_key_key & f'event_type LIKE "%;%"' & f'event_start_time <= "{trial_end_edges[trialnum[0]]}"' & f'event_end_time >= "{trial_start_edges[trialnum[0]]}"' & f'trial_id= "{trialnum[0] + 1}"') #.fetch(format = "frame", order_by = "event_start_time")
            
            # do server-side insert - fetch does not work. The number key seems to be rounded.
            trial.TrialEvent.insert(TrialEvent_query_keys,  allow_direct_insert=True, skip_duplicates=True, ignore_extra_fields=True)
    if aux_setup_type == "bench2p_Oddball_V2":
        # Extract Trials

        # stims_per_trial = len(set((event.Event & scan_key_key & 'event_type LIKE "%;%"').fetch("event_type")))
        all_stims = (event.Event & scan_key_key & 'event_type LIKE "%;%"').fetch("event_type")
        all_stims = ['; '.join(item.split('; ')[:1]) for item in all_stims]
        # trial_stims = {x: list(all_stims).count(x) for x in all_stims}
        trials = set([list(all_stims).count(x) for x in all_stims])
        stims_per_trial = int(len(all_stims) / min(trials))

            
            # trial_start_edges = (event.Event & scan_key_key & 'event_type LIKE "%;%"').fetch("event_start_time",order_by = "event_start_time")[::stims_per_trial] 
            # trial_end_edges = (event.Event & scan_key_key & 'event_type LIKE "%;%"').fetch("event_end_time",order_by = "event_end_time")[stims_per_trial-1::stims_per_trial]     
            
        trial_start_edges = (event.Event & scan_key_key & 'event_type LIKE "%trial%"').fetch('event_start_time',order_by = "event_start_time")
        trial_end_edges = (event.Event & scan_key_key & 'event_type LIKE "%trial%"').fetch('event_end_time', order_by = "event_end_time")
        
        # trial_event_name = (event.Event & scan_key_key & 'event_type LIKE "%;%"').fetch("event_type")[0].split(':')[0]
        trial_event_name = 'Oddball'
        trial.TrialType().insert1({'trial_type': trial_event_name, 'trial_type_description': 'Stimulus nomenclature: Type; Class; Azimuth; Elevation; Size; Orientation; Spatial Frequency; Temporal Frequency'}, skip_duplicates=True)
                
        for trialnum in enumerate(range(min(trials))): #enumerate(trial_start_edges):
            trial.Trial.insert1({'session_id': session_key, 'scan_id': scan_key, 'trial_id': trialnum[0]+1, 'trial_type': trial_event_name, 'trial_start_time': trial_start_edges[trialnum[0]], 'trial_stop_time': trial_end_edges[trialnum[0]]},  allow_direct_insert=True, skip_duplicates=True)
            
            # generate query object from joint Trial Event table
            TrialEvent_query_keys = (event.Event * trial.Trial & scan_key_key & f'event_type LIKE "%;%"' & f'event_start_time <= "{trial_end_edges[trialnum[0]+1]}"' & f'event_end_time >= "{trial_start_edges[trialnum[0]]}"' & f'trial_id= "{trialnum[0] + 1}"') #.fetch(format = "frame", order_by = "event_start_time")
            
            # do server-side insert - fetch does not work. The number key seems to be rounded.
            trial.TrialEvent.insert(TrialEvent_query_keys,  allow_direct_insert=True, skip_duplicates=True, ignore_extra_fields=True)

    if aux_setup_type == "bench2p_Oddball_v3":
        # Oddball v3 encodes trial boundaries on an explicit trial channel.
        all_stims = (event.Event & scan_key_key & 'event_type LIKE "%;%"').fetch("event_type")
        all_stims = ['; '.join(item.split('; ')[:1]) for item in all_stims]
        trial_start_edges = (event.Event & scan_key_key & 'event_type LIKE "%trial%"').fetch('event_start_time', order_by="event_start_time")
        trial_end_edges = (event.Event & scan_key_key & 'event_type LIKE "%trial%"').fetch('event_end_time', order_by="event_end_time")
        if len(trial_start_edges) == 0 or len(trial_end_edges) == 0:
            print("No aux_trial events found for bench2p_Oddball_v3, skipping Trial/TrialEvent insert.")
            return

        trial_event_name = 'Oddball'
        trial.TrialType().insert1({'trial_type': trial_event_name, 'trial_type_description': 'Stimulus nomenclature: Type; Class; Azimuth; Elevation; Size; Orientation; Spatial Frequency; Temporal Frequency'}, skip_duplicates=True)

        if len(all_stims) > 0:
            trial_repeats = set([list(all_stims).count(x) for x in all_stims])
            n_trials = min(min(trial_repeats), len(trial_start_edges), len(trial_end_edges))
        else:
            n_trials = min(len(trial_start_edges), len(trial_end_edges))

        for trialnum in enumerate(range(n_trials)):
            trial.Trial.insert1({'session_id': session_key, 'scan_id': scan_key, 'trial_id': trialnum[0] + 1, 'trial_type': trial_event_name, 'trial_start_time': trial_start_edges[trialnum[0]], 'trial_stop_time': trial_end_edges[trialnum[0]]}, allow_direct_insert=True, skip_duplicates=True)
            trial_end_idx = min(trialnum[0] + 1, len(trial_end_edges) - 1)
            TrialEvent_query_keys = (event.Event * trial.Trial & scan_key_key & f'event_type LIKE "%;%"' & f'event_start_time <= "{trial_end_edges[trial_end_idx]}"' & f'event_end_time >= "{trial_start_edges[trialnum[0]]}"' & f'trial_id= "{trialnum[0] + 1}"')
            trial.TrialEvent.insert(TrialEvent_query_keys, allow_direct_insert=True, skip_duplicates=True, ignore_extra_fields=True)


def ingest_bench2p_oddball_v3(scan_basenames, curr_path, curr_aux, sweep, sr):
    """Extract oddball v3 aux channels without affecting legacy oddball/v2 logic."""
    stim_log = pd.DataFrame()
    for _k in scan_basenames:
        stim_file_paths = [fp.as_posix() for fp in curr_path.glob('*bonsai_stimulus_log*.csv')]
        if len(stim_file_paths) != 1:
            print(f"More or less than 1 stim_files found in {_k} - not extracting stim IDs")
        else:
            stim_log = pd.read_csv(stim_file_paths[0])

    ai_channel_names = [name.decode('utf-8') if isinstance(name, (bytes, bytearray)) else str(name) for name in curr_aux['header']['AIChannelNames']]
    di_channel_names = [name.decode('utf-8') if isinstance(name, (bytes, bytearray)) else str(name) for name in curr_aux['header']['DIChannelNames']]
    analog_data = curr_aux[sweep]['analogScans']
    digital_data = curr_aux[sweep]['digitalScans']

    df = pd.DataFrame(analog_data.T, columns=ai_channel_names)
    digital_scans = demultiplex(digital_data[0], len(di_channel_names))
    digital_scans_df = pd.DataFrame(digital_scans.T, columns=di_channel_names[::-1])
    df = pd.concat([df, digital_scans_df], axis=1)

    def _pick_channel(name_candidates, required=True):
        for name in name_candidates:
            if name in df.columns:
                return df[name].values
        if required:
            raise ValueError(f"bench2p_Oddball_v3: missing required channel, expected one of {name_candidates}")
        return None

    main_track_gate_chan = _pick_channel(["Main Trigger", "MainTrigger", "main_track_gate"])
    shutter_chan = _pick_channel(["Bench2p 920 shutter", "Shutter", "shutter"])
    bench2p_frame_chan = _pick_channel(["Frame clock", "FrameClock", "bench2p_frames"])
    bench2p_vol_chan = _pick_channel(["Volume clock", "VolumeClock", "bench2p_volumes"])
    cam_trigger = _pick_channel(["Camera Trigger", "CameraTriggerIn", "aux_cam"])
    trial_chan = _pick_channel(["Trial_Onset", "TrialOnset", "aux_trial"])
    hifi_chan = _pick_channel(["HIFI", "HiFi", "aux_HIFI"], required=False)

    main_track_gate_chan[-1] = 0
    shutter_chan[-1] = 0
    bench2p_frame_chan[-1] = 0
    bench2p_vol_chan[-1] = 0
    cam_trigger[-1] = 0
    trial_chan[-1] = 0
    if hifi_chan is not None:
        hifi_chan[-1] = 0

    ts_main_track_gate_chan = get_timestamps(main_track_gate_chan, sr)
    ts_shutter_chan = get_timestamps(shutter_chan, sr)
    ts_bench2p_frame_chan = get_timestamps(bench2p_frame_chan, sr)
    ts_bench2p_vol_chan = get_timestamps(bench2p_vol_chan, sr)
    ts_cam_trigger = get_timestamps_auxcam(cam_trigger, sr)
    ts_trial = get_timestamps(trial_chan, sr)
    ts_hifi = get_timestamps(hifi_chan, sr) if hifi_chan is not None else np.array([])

    ts_texts_dict = get_oddball_stim_timestamps(df, sr, stim_log)

    event_types = {
        'main_track_gate': ts_main_track_gate_chan,
        'shutter': ts_shutter_chan,
        'bench2p_frames': ts_bench2p_frame_chan,
        'bench2p_volumes': ts_bench2p_vol_chan,
        'aux_cam': ts_cam_trigger,
        'aux_trial': ts_trial,
    }
    if ts_hifi.size:
        event_types['aux_HIFI'] = ts_hifi
    for key, values in ts_texts_dict.items():
        event_types[key] = np.asarray(values)
    return event_types


def butter_filter(sr, cutoff, order=4, btype='low'):
    from scipy.signal import butter
    """Create a Butterworth filter in SOS format."""
    nyquist = 0.5 * sr
    if cutoff >= nyquist:
        raise ValueError("Cutoff frequency must be less than the Nyquist frequency (sr / 2).")
    normal_cutoff = cutoff / nyquist
    return butter(order, normal_cutoff, btype=btype, analog=False, output='sos')


def get_pdiode_timestamps(curr_aux, sr, stim_log, thr=1.6):
    """Extract photodiode on/off timestamps for oddball v3 stimulus alignment."""
    if 'Photodiode' not in curr_aux.columns or 'Main Trigger' not in curr_aux.columns:
        return []

    pdiode = curr_aux['Photodiode'].values
    master_trigger = curr_aux['Main Trigger'].values.astype(int)

    try:
        from scipy.signal import sosfiltfilt
        sos_filter = butter_filter(sr, cutoff=29.98)
        pd_filtered = sosfiltfilt(sos_filter, pdiode)
    except Exception:
        pd_filtered = pdiode.astype(float)

    master_trigger_diff = np.diff(master_trigger)
    master_on_idx = np.where(master_trigger_diff > 0.5)[0]
    master_off_idx = np.where(master_trigger_diff < 0)[0]
    if len(master_on_idx) == 0 or len(master_off_idx) == 0:
        return []

    master_on = int(master_on_idx[0]) + 1
    master_off = int(master_off_idx[0]) + 1
    in_rec = pd_filtered[master_on:master_off]
    if in_rec.size == 0:
        return []

    pd_filtered[master_off:] = np.min(in_rec)
    pd_filtered[:master_on] = np.min(in_rec)
    pd_zero = pd_filtered - np.min(in_rec)
    pd_diff = np.diff(pd_zero)
    if master_off - master_on <= 2:
        return []

    thr_max = np.max(pd_diff[master_on:master_off - 1])
    pd_diff = np.where(pd_diff > thr_max, 0, pd_diff)
    pd_diff_max = pd_diff.max()
    if pd_diff_max == 0:
        return []
    pd_ons = np.where(pd_diff > (pd_diff_max / thr), 1, 0)
    pd_offs = np.where(pd_diff < -(pd_diff_max / thr), 1, 0)
    pd_ons, pd_offs = np.r_[0, pd_ons], np.r_[0, pd_offs]
    ons_matched = np.r_[0, np.diff(pd_ons)]
    offs_matched = np.r_[0, np.diff(pd_offs)]
    ons_idx = np.where(ons_matched > 0.5)[0]
    offs_idx = np.where(offs_matched > 0.5)[0]
    if len(ons_idx) == 0 or len(offs_idx) == 0:
        return []

    ons_ts = ons_idx / sr
    offs_ts = (offs_idx / sr)[:-1]
    expected = len(stim_log) if hasattr(stim_log, "__len__") else 0
    if expected and len(ons_ts) != expected:
        print('Aux-File und StimLog have not the same number of stimulus onsets! CHECK THAT!')

    all_ts = []
    for on, off in zip(ons_ts, offs_ts):
        all_ts.extend([float(on), float(off)])
    return all_ts


def get_oddball_stim_timestamps(data, sr, stim_log):
    """Build oddball v3 per-text timestamp events from stimulus ID and photodiode channels."""
    from collections import defaultdict

    if 'Visual Stim ID' not in data.columns:
        return {}

    pd_ts = get_pdiode_timestamps(data, sr, stim_log)
    _, id_plateaus = get_timestamps_from_plateaus(data['Visual Stim ID'].values, sr)

    texts_ts = defaultdict(list)
    i = 0
    for idc_on, idc_off, text_id in id_plateaus:
        text_label = f'{text_id:.0f}'
        texts_ts[f'bonsai_text_{text_label}'].append(float(idc_on / sr))
        texts_ts[f'bonsai_text_{text_label}'].append(float(idc_off / sr))
        if i + 1 < len(pd_ts):
            texts_ts[f'pdiode_text_{text_label}'].append(float(pd_ts[i]))
            texts_ts[f'pdiode_text_{text_label}'].append(float(pd_ts[i + 1]))
        i += 2
    return texts_ts

def compute_angular_velocity(time, angle, window):
    # Convert the angles to radians
    angle = np.radians(angle)
    
    # Unwrap the angles to handle wrap-around
    unwrapped_angle = np.unwrap(angle)
    
    # Convert back to degrees
    unwrapped_angle = np.abs(np.degrees(unwrapped_angle))
    
    # Calculate rolling mean with the defined window size
    unwrapped_angle_smoothed = np.convolve(unwrapped_angle, np.ones(window), 'valid') / window
    
    
    # Calculate the difference in angles and time
    angle_diff = np.diff(unwrapped_angle_smoothed)
    time_diff = np.diff(time)
    
    # Calculate angular velocity
    angular_velocity_smoothed = angle_diff / time_diff[:-window+1]
    
    return angular_velocity_smoothed, unwrapped_angle_smoothed

import numpy as np

import numpy as np

def _fix_frame_indices_new(frame_indices, verbose=True, initial_corruption_threshold=140000, initial_check_frames=120):
    """
    Fix temporal inconsistencies in binary-decoded frame indices.
    
    Three-step correction:
    1. FIRST: Check for corrupted initial frames with unrealistic high values
    2. SECOND: Set max bit values (524287 = 2^19-1) to 0
    3. THIRD: Replace both backward and forward noise with predicted increments
    
    Parameters:
    -----------
    frame_indices : list or array
        Raw frame indices from binary decoding
    verbose : bool
        Print correction details
    initial_corruption_threshold : int, optional
        Threshold for detecting corrupted initial readout (default: 140000)
        Lowered from 150000 to catch subtle corruption patterns
    initial_check_frames : int, optional
        Number of initial frames to check for corruption (default: 120)
        Increased from 100 to detect late-starting corruption
        
    Returns:
    --------
    list : Corrected frame indices with a monotonic sequence
    """
    corrected_indices = np.array(frame_indices, dtype=float)
    total_frames = len(corrected_indices)
    if total_frames <= 1:
        return frame_indices

    corrections_made = 0

    # STEP 0: Check for corrupted initial frames and find valid start point  
    initial_check_frames = min(initial_check_frames, total_frames)  # Use configurable parameter
    corruption_threshold = initial_corruption_threshold  # Use configurable parameter
    initial_sum = np.sum(corrected_indices[:initial_check_frames])
    if initial_sum > corruption_threshold:
        # Find the start of monotonically increasing frames WITHIN the corrupted region
        valid_start_idx = 0
        search_end = min(initial_check_frames + 30, total_frames - 10)  # Extended search for late corruption
        for start_idx in range(search_end):
            if start_idx + 10 > total_frames:
                break
            window = corrected_indices[start_idx:start_idx + 10]
            if len(window) < 10 or np.any(~np.isfinite(window)):
                continue
            diffs = np.diff(window)
            # Look for reasonable increments, allowing small backward jumps (OptiTrack jitter)
            # Most diffs should be positive, and overall trend should be increasing
            positive_diffs = np.sum(diffs > 0)
            small_negative_diffs = np.sum((diffs < 0) & (diffs >= -3))  # Allow small backward steps
            large_jumps = np.sum(np.abs(diffs) > 50)
            overall_trend = window[-1] - window[0]  # Should be positive overall
            
            if (positive_diffs >= 7 and large_jumps == 0 and overall_trend > 0 and 
                small_negative_diffs <= 2 and np.mean(diffs) > -1):
                valid_start_idx = start_idx
                break
        if valid_start_idx > 0:
            corrected_indices[:valid_start_idx] = 0
            corrections_made += valid_start_idx
            if verbose:
                print(f"   🔧 Detected corrupted initial frames: sum={initial_sum:,.0f} > {corruption_threshold:,}")
                print(f"   🔧 Found valid monotonic start at frame {valid_start_idx}, set first {valid_start_idx} frames to 0")
        elif verbose:
            print(f"   ⚠️ High initial sum detected ({initial_sum:,.0f}) but no valid monotonic start found")

    # STEP 1: Set max bit values to 0 (19-bit max = 524287)
    max_bit_value = 2**19 - 1  # 524287
    max_bit_mask = corrected_indices >= max_bit_value
    max_bit_count = int(np.sum(max_bit_mask))
    if max_bit_count > 0:
        corrected_indices[max_bit_mask] = 0
        corrections_made += max_bit_count
        if verbose:
            print(f"   🔧 Set {max_bit_count} max bit values ({max_bit_value:,}) to 0")

    # STEP 2: Robust step estimation (mode over small positive diffs, 1..50)
    vals = corrected_indices.copy()
    valid_mask_vals = np.isfinite(vals) & (vals > 0)
    pair_mask = valid_mask_vals[:-1] & valid_mask_vals[1:]
    left = vals[:-1][pair_mask]
    right = vals[1:][pair_mask]
    diffs = right - left
    pos_small = diffs[(diffs > 0) & (diffs <= 15)]  # focus on plausible per-frame steps
    if pos_small.size:
        # Histogram with 1-count bins from 0.5..50.5, pick most frequent step
        bins = np.arange(0.5, 50.5 + 1e-9, 1.0)
        hist, edges = np.histogram(pos_small, bins=bins)
        mode_bin = int(np.argmax(hist))
        step_mode = (edges[mode_bin] + edges[mode_bin + 1]) / 2.0
        # Refine: median of diffs within ±1 around the mode
        refine_mask = (pos_small >= (step_mode - 1.0)) & (pos_small <= (step_mode + 1.0))
        step_est = float(np.median(pos_small[refine_mask])) if np.any(refine_mask) else float(step_mode)
    else:
        step_est = 5.0  # fallback - optimal target increment based on analysis
    # Keep sane: clamp to [3, 15] based on analysis of both cameras
    step_est = float(np.clip(step_est, 3.0, 15.0))

    # Predictive thresholds (optimized for both simple and complex corruption)
    resync_tol = 2.5 * step_est   # slightly relaxed tolerance for natural variation
    # classification thresholds (for reporting only)
    low_read_slack = 2.0 * step_est  # increased tolerance for backward jumps
    forward_excess = 3.0 * step_est  # increased tolerance for forward jumps

    if verbose:
        print(f"   🔧 Estimated typical increment: {step_est:.2f} "
              f"(resync_tol={resync_tol:.2f}, low_slack={low_read_slack:.2f}, fwd_excess={forward_excess:.2f})")

    # STEP 3: Predictive reconstruction (both backward/forward noise -> prediction)
    backwards_fixes = 0
    forward_fixes = 0
    resync_hits = 0
    invalid_skips = 0
    predictions_used = 0

    for i in range(1, total_frames):
        prev_idx = corrected_indices[i-1]
        curr_obs = corrected_indices[i]

        # If previous is invalid/non-positive, we can't predict reliably; initialize from obs if possible
        if not np.isfinite(prev_idx) or prev_idx <= 0:
            corrected_indices[i] = max(curr_obs if np.isfinite(curr_obs) else 0.0, 0.0)
            continue

        predicted = prev_idx + step_est

        # Treat non-finite or non-positive observations as missing → use prediction
        if (not np.isfinite(curr_obs)) or (curr_obs <= 0):
            corrected_indices[i] = predicted
            corrections_made += 1
            invalid_skips += 1
            predictions_used += 1
            continue

        # If observation close to prediction, accept it (re-sync) while enforcing non-decrease
        if abs(curr_obs - predicted) <= resync_tol:
            corrected_indices[i] = max(curr_obs, prev_idx)
            resync_hits += 1
            continue

        # Otherwise treat as noise and use prediction
        delta = curr_obs - prev_idx
        if delta <= 0 or curr_obs < (prev_idx - low_read_slack):
            backwards_fixes += 1
        elif curr_obs > (prev_idx + forward_excess):
            forward_fixes += 1
        else:
            # mild mismatch but outside resync window
            pass
        corrected_indices[i] = predicted
        corrections_made += 1
        predictions_used += 1

    # Enforce monotonicity and integerize
    corrected_indices = np.maximum.accumulate(corrected_indices)
    corrected_indices = np.rint(corrected_indices).astype(int)
    corrected_indices = np.maximum.accumulate(corrected_indices)

    if verbose and corrections_made > 0:
        print(f"   🔧 Predictive fixes — backward: {backwards_fixes}, forward: {forward_fixes}, "
              f"resyncs: {resync_hits}, invalid: {invalid_skips}, predictions used: {predictions_used}, "
              f"total corrections: {corrections_made}")

    # Final verification
    final_valid_mask = corrected_indices > 0
    final_valid_indices = corrected_indices[final_valid_mask]
    if len(final_valid_indices) > 1:
        final_diffs = np.diff(final_valid_indices)
        remaining_backwards = np.sum(final_diffs < 0)
        if verbose and remaining_backwards == 0 and corrections_made > 0:
            print("   ✅ Sequence is monotonic (predictive mode)")

    return corrected_indices.tolist()

def _fix_frame_indices(frame_indices, verbose=True):
    """
    Fix temporal inconsistencies in binary-decoded frame indices.
    
    Three-step correction:
    1. FIRST: Check for corrupted initial frames with unrealistic high values
    2. SECOND: Set max bit values (524287 = 2^19-1) to 0
    3. THIRD: Set backwards jump frames equal to previous frame's index
    
    Parameters:
    -----------
    frame_indices : list or array
        Raw frame indices from binary decoding
    verbose : bool
        Print correction details
        
    Returns:
    --------
    list : Corrected frame indices with backwards jumps eliminated
    """
    corrected_indices = np.array(frame_indices, dtype=float)
    total_frames = len(corrected_indices)
    
    if total_frames <= 1:
        return frame_indices
    
    corrections_made = 0
    
    # STEP 0: Check for corrupted initial frames and find valid start point
    initial_check_frames = min(100, total_frames)
    corruption_threshold = 150000  # Threshold for detecting corrupted initial readout
    
    initial_sum = np.sum(corrected_indices[:initial_check_frames])
    
    if initial_sum > corruption_threshold:
        # Find the start of monotonically increasing frames
        valid_start_idx = 0
        
        for start_idx in range(total_frames - 10):  # Need at least 10 frames to check
            # Check if next 10 frames are monotonically increasing with small increments
            window = corrected_indices[start_idx:start_idx + 10]
            diffs = np.diff(window)
            
            # Check if all differences are positive and smaller than 100
            if np.all(diffs > 0) and np.all(diffs < 100):
                valid_start_idx = start_idx
                break
        
        if valid_start_idx > 0:
            corrected_indices[:valid_start_idx] = 0
            corrections_made += valid_start_idx
            if verbose:
                print(f"   🔧 Detected corrupted initial frames: sum={initial_sum:,.0f} > {corruption_threshold:,}")
                print(f"   🔧 Found valid monotonic start at frame {valid_start_idx}, set first {valid_start_idx} frames to 0")
        elif verbose:
            print(f"   ⚠️ High initial sum detected ({initial_sum:,.0f}) but no valid monotonic start found")
    
    # Expected sum calculation for reference (frames_eye=100, startsec=3, optiframesrate=240, eyeframerate=50):
    # expected_sum ≈ 100000, so threshold of 150000 provides 50% safety margin
    
    # STEP 1: Set max bit values to 0 (19-bit max = 524287)
    max_bit_value = 2**19 - 1  # 524287
    max_bit_mask = corrected_indices >= max_bit_value
    max_bit_count = np.sum(max_bit_mask)
    
    if max_bit_count > 0:
        corrected_indices[max_bit_mask] = 0
        corrections_made += max_bit_count
        if verbose:
            print(f"   🔧 Set {max_bit_count} max bit values ({max_bit_value:,}) to 0")
    
    # STEP 2: Fix backwards jumps AFTER corruption cleanup
    valid_mask = corrected_indices > 0
    if np.sum(valid_mask) <= 1:
        return corrected_indices.astype(int).tolist()
    
    backwards_fixes = 0
    
    # Sequential processing for backwards jumps
    for i in range(1, total_frames):
        current_idx = corrected_indices[i]
        prev_idx = corrected_indices[i-1]
        
        # Skip frames with zero index (invalid)
        if current_idx <= 0 or prev_idx <= 0:
            continue
        
        delta = current_idx - prev_idx
        
        # Check for backwards jump or identical frames  
        if delta <= 0:
            # Set frame equal to previous frame's index
            corrected_indices[i] = prev_idx
            backwards_fixes += 1
            corrections_made += 1
    
    if verbose and corrections_made > 0:
        print(f"   🔧 Fixed {backwards_fixes} backwards jumps, {corrections_made} total corrections")
    
    # Final verification
    final_valid_mask = corrected_indices > 0
    final_valid_indices = corrected_indices[final_valid_mask]
    
    if len(final_valid_indices) > 1:
        final_diffs = np.diff(final_valid_indices)
        remaining_backwards = np.sum(final_diffs < 0)
        if verbose and remaining_backwards == 0 and corrections_made > 0:
            print(f"   ✅ All backwards jumps eliminated - sequence is now monotonic")
    
    return corrected_indices.astype(int).tolist()


# =============================================================================
# OCR-based digit extraction for eye cameras with text overlay
# =============================================================================

def _ocr_to_int_list(proc_nums, missing_value=0):
    """Convert OCR string outputs to integer list, handling None and invalid values."""
    out = []
    for v in proc_nums:
        if v is None:
            out.append(missing_value)
            continue
        s = str(v).strip()
        if s == "":
            out.append(missing_value)
            continue
        # keep only digits (handles stray characters)
        digits = ''.join(ch for ch in s if ch.isdigit())
        if digits == "":
            out.append(missing_value)
            continue
        try:
            out.append(int(digits))
        except Exception:
            out.append(missing_value)
    return np.array(out, dtype=int)


def fix_ocr_frames(frame_indices, L_max=3, verbose=True, log_path=None):
    """
    Fix OCR errors in frame indices extracted from eye camera videos with text overlay.
    
    The eye camera (~50 fps) displays OptiTrack frame numbers (~240 Hz) as text.
    Expected diff pattern: [5,5,5,5,4] repeating (240/50 = 4.8 avg).
    
    NOTE: These rates are empirically validated from data fits:
      - Eye camera: 50.00 Hz (derived from linear fit slope)
      - OptiTrack: ~240 Hz  
      - Ratio: 4.8, which produces the 5,5,5,5,4 pattern (avg=4.8)
    
    Correction steps:
      1a. Replace leading NaNs with 0 and protect first valid frame
      1b. Replace leading repeating values with 0
      2. Detect repeated sequences of length <= L_max, zero the first occurrence
      3. Fix single-frame repeats using previous 5/4 pattern (LOCAL inference)
      4. Fix bad diffs (not 4 or 5) only if next diff also invalid, using pattern
      5. Handle 4,4,6 -> 4,5,5 pattern
      
    Parameters
    ----------
    frame_indices : list
        Raw frame indices from OCR (strings or ints, None for failures)
    L_max : int
        Max sequence length for duplicate detection (default: 3)
    verbose : bool
        Print progress to stdout
    log_path : str or Path, optional
        Path to write detailed CSV log file
        
    Returns
    -------
    list : Corrected frame indices
    list : Log entries for all corrections
    """
    from datetime import datetime
    
    # Initialize log entries list
    log_entries = []

    def log(correction_type, eye_frame_idx, original_value, corrected_value,
            diff_before=None, diff_after=None, details=""):
        entry = {
            'eye_frame_idx': eye_frame_idx,
            'correction_type': correction_type,
            'original_value': original_value,
            'corrected_value': corrected_value,
            'diff_before': diff_before,
            'diff_after': diff_after,
            'details': details
        }
        log_entries.append(entry)
        if verbose:
            print(f"[{correction_type}] frame {eye_frame_idx}: "
                  f"{original_value} -> {corrected_value} | {details}")

    def expected_from_local(valid_diffs):
        tail = valid_diffs[-4:]
        if len(tail) == 4 and all(d == 5 for d in tail):
            return 4
        return 5

    frame_indices_int = _ocr_to_int_list(frame_indices)
    frames = np.array(frame_indices_int, dtype=float)
    n = len(frames)

    if n == 0:
        return [], []

    skip_fix = np.zeros(n, dtype=bool)

    # ---------- Step 1a: Replace leading NaNs with 0 and PROTECT ----------
    first_valid = 0
    while first_valid < n and (not np.isfinite(frames[first_valid]) or frames[first_valid] == 0):
        first_valid += 1

    if first_valid > 0:
        first_valid_val = int(frames[first_valid]) if first_valid < n else 0
        for idx in range(first_valid):
            log(
                'LEADING_ZERO',
                idx,
                frame_indices_int[idx],
                0,
                details=(
                    f"frame[{idx}]={frame_indices_int[idx]}->0, "
                    f"first_valid@{first_valid}={first_valid_val}, "
                    f"gap={first_valid}frames"
                )
            )
        frames[:first_valid] = 0
        skip_fix[:first_valid] = True  # PROTECT

    if verbose and first_valid < n:
        print(f"Protected first valid frame at index {first_valid} (value={frames[first_valid]})")

    # ---------- Step 1b: Handle leading repeated numbers and PROTECT ----------
    if first_valid < n:
        val = frames[first_valid]
        j = first_valid + 1
        while j < n and frames[j] == val:
            j += 1

        repeat_len = j - first_valid
        if repeat_len > 1:
            for idx in range(first_valid, j - 1):
                log(
                    'LEADING_REPEAT',
                    idx,
                    int(frames[idx]),
                    0,
                    details=f"leading repeat of {int(val)}, kept last at index {j-1}"
                )
                frames[idx] = 0
                skip_fix[idx] = True  # PROTECT

            if verbose:
                print(
                    f"Leading repeats detected: value={int(val)}, "
                    f"count={repeat_len}, kept index {j-1}"
                )

    # ---------- Step 2: Detect repeated sequences ----------
    for L in range(2, L_max + 1):
        seen = {}
        for i in range(first_valid, n - L + 1):
            seq = tuple(frames[i:i+L])
            if not all(np.isfinite(seq)) or 0 in seq:
                continue
            if seq in seen:
                start = seen[seq]
                if not any(skip_fix[start:start+L]):
                    gap_frames = i - start
                    for idx in range(start, start + L):
                        pos = idx - start
                        log(
                            'DUPLICATE_SEQ',
                            idx,
                            int(frames[idx]),
                            0,
                            details=(
                                f"seq[{pos}]={int(frames[idx])}->0, "
                                f"L={L}, first@{start}, dup@{i}, gap={gap_frames}frames"
                            )
                        )
                    frames[start:start+L] = 0
                    skip_fix[start:start+L] = True
            else:
                seen[seq] = i

    # ---------- Step 3: Fix diffs using LOCAL pattern ----------
    valid_diffs = []
    i = first_valid

    while i < n - 1:
        if skip_fix[i]:
            i += 1
            continue

        prev = frames[i]
        curr = frames[i + 1]

        if not np.isfinite(prev) or not np.isfinite(curr):
            i += 1
            continue

        diff = curr - prev

        # --- Handle 4,4,6 -> 4,5,5 ---
        if diff == 6 and len(valid_diffs) >= 2 and valid_diffs[-2:] == [4, 4]:
            if (
                i - 1 >= first_valid and
                not skip_fix[i - 1] and
                not skip_fix[i] and
                not skip_fix[i + 1]
            ):
                old_i, old_i1 = frames[i], frames[i + 1]
                frames[i] = frames[i - 1] + 5
                frames[i + 1] = frames[i] + 5
                valid_diffs[-1] = 5
                valid_diffs.append(5)

                log(
                    'PATTERN_446',
                    i,
                    int(old_i),
                    int(frames[i]),
                    diff_before=4,
                    diff_after=5,
                    details="4,4,6 -> 4,5,5 correction"
                )
                log(
                    'PATTERN_446',
                    i + 1,
                    int(old_i1),
                    int(frames[i + 1]),
                    diff_before=6,
                    diff_after=5,
                    details="4,4,6 -> 4,5,5 correction"
                )
            i += 1
            continue

        if diff in (4, 5):
            valid_diffs.append(int(diff))
            valid_diffs = valid_diffs[-5:]
            i += 1
            continue

        # --- BAD_DIFF: fix only if next diff is also bad ---
        next_bad = True
        if i + 2 < n and not skip_fix[i + 1]:
            nxt_diff = frames[i + 2] - frames[i + 1]
            next_bad = nxt_diff not in (4, 5)

        if next_bad:
            expected = expected_from_local(valid_diffs)
            old_val = frames[i + 1]
            frames[i + 1] = prev + expected
            valid_diffs.append(expected)
            valid_diffs = valid_diffs[-5:]

            log(
                'BAD_DIFF',
                i + 1,
                int(old_val),
                int(frames[i + 1]),
                diff_before=int(old_val - prev),
                diff_after=expected,
                details="local pattern inference"
            )

        i += 1

    # ---------- Final interpolation ----------
    corrected, interp_log = interpolate_frame_indices(
        frames.astype(int).tolist(),
        avg_step=4.8,
        verbose=verbose
    )

    # ---------- Write log ----------
    if log_path is not None:
        log_path = Path(log_path)
        all_entries = log_entries + interp_log

        with open(log_path, 'w') as f:
            f.write("# Frame Correction Log (OCR-based)\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Total frames: {n}\n")
            f.write(f"# OCR corrections: {len(log_entries)}\n")
            f.write(f"# Interpolated frames: {len(interp_log)}\n")
            f.write(f"# L_max: {L_max}\n\n")
            f.write(
                "eye_frame_idx,correction_type,original_value,"
                "corrected_value,diff_before,diff_after,details\n"
            )
            for e in all_entries:
                f.write(
                    f"{e['eye_frame_idx']},{e['correction_type']},"
                    f"{e['original_value']},{e['corrected_value']},"
                    f"{'' if e['diff_before'] is None else e['diff_before']},"
                    f"{'' if e['diff_after'] is None else e['diff_after']},"
                    f"\"{e['details']}\"\n"
                )

        if verbose:
            print(f"\n📝 Log written to: {log_path}")

    return corrected, log_entries + interp_log


def interpolate_frame_indices(frame_indices, avg_step=4.8, verbose=True):
    """
    Fill zero/invalid frame indices via linear interpolation.
    
    After OCR correction, some frames may still have index=0 (unfixable).
    This function interpolates those gaps using neighboring valid values,
    ensuring every frame gets a unique, monotonically increasing index
    for proper timestamp mapping.
    
    Parameters
    ----------
    frame_indices : list or array
        Frame indices where 0 indicates invalid/missing
    avg_step : float
        Expected average step between frames. Default 4.8 is derived from
        OptiTrack rate (~240Hz) / eye camera rate (~50Hz) = 4.8.
        This is only used when no valid frames exist (synthetic fallback).
    verbose : bool
        Print interpolation statistics
        
    Returns
    -------
    tuple : (interpolated frame indices as list, log entries as list of dicts)
    """
    frames = np.array(frame_indices, dtype=float)
    original = frames.copy()
    frames[frames == 0] = np.nan
    n = len(frames)
    log_entries = []
    
    if n == 0:
        return [], []
    
    valid_idx = np.where(np.isfinite(frames))[0]
    invalid_idx = np.where(~np.isfinite(frames))[0]
    n_invalid = len(invalid_idx)
    
    # Edge case: no valid frames - generate synthetic sequence
    if len(valid_idx) == 0:
        if verbose:
            print(f"   ⚠️ No valid frames - generating synthetic sequence (step={avg_step})")
        result = np.round(np.arange(n) * avg_step + 1).astype(int)
        for i in range(n):
            diff_after = int(result[i] - result[i-1]) if i > 0 else None
            log_entries.append({
                'eye_frame_idx': i,
                'correction_type': 'INTERPOLATED_SYNTHETIC',
                'original_value': int(original[i]),
                'corrected_value': int(result[i]),
                'diff_before': None,
                'diff_after': diff_after,
                'details': f"frame[{i}]:{int(original[i])}->{int(result[i])}, synthetic, n_total={n}, step={avg_step}"
            })
        return result.tolist(), log_entries
    
    # Edge case: all valid - nothing to interpolate
    if len(valid_idx) == n:
        return frames.astype(int).tolist(), []
    
    # Interpolate invalid positions using valid neighbors
    filled = np.interp(np.arange(n), valid_idx, frames[valid_idx])
    
    # Extrapolate leading invalid frames (before first valid)
    if valid_idx[0] > 0:
        for i in range(valid_idx[0] - 1, -1, -1):
            filled[i] = max(1, filled[i + 1] - avg_step)
    
    # Extrapolate trailing invalid frames (after last valid)
    if valid_idx[-1] < n - 1:
        for i in range(valid_idx[-1] + 1, n):
            filled[i] = filled[i - 1] + avg_step
    
    # Ensure monotonically increasing and convert to int
    filled = np.maximum.accumulate(filled)
    result = np.round(filled).astype(int)
    
    # Final pass: ensure strict monotonicity (no duplicates)
    for i in range(1, n):
        if result[i] <= result[i - 1]:
            result[i] = result[i - 1] + 1
    
    # Log all interpolated frames
    for i in invalid_idx:
        interp_type = 'INTERPOLATED'
        if i < valid_idx[0]:
            interp_type = 'EXTRAPOLATED_LEADING'
        elif i > valid_idx[-1]:
            interp_type = 'EXTRAPOLATED_TRAILING'
        
        # Calculate diff_after (step from previous frame)
        diff_after = int(result[i] - result[i-1]) if i > 0 else None
        
        # Find neighbor info for context
        prev_valid_idx = valid_idx[valid_idx < i][-1] if np.any(valid_idx < i) else None
        next_valid_idx = valid_idx[valid_idx > i][0] if np.any(valid_idx > i) else None
        prev_valid_val = int(frames[prev_valid_idx]) if prev_valid_idx is not None else None
        next_valid_val = int(frames[next_valid_idx]) if next_valid_idx is not None else None
        
        # Calculate gap size
        if prev_valid_idx is not None and next_valid_idx is not None:
            gap_size = next_valid_idx - prev_valid_idx - 1
            gap_info = f"gap={gap_size}frames"
        elif prev_valid_idx is not None:
            gap_info = f"trailing_gap={n - prev_valid_idx - 1}frames"
        else:
            gap_info = f"leading_gap={next_valid_idx}frames"
        
        log_entries.append({
            'eye_frame_idx': i,
            'correction_type': interp_type,
            'original_value': int(original[i]),
            'corrected_value': int(result[i]),
            'diff_before': None,
            'diff_after': diff_after,
            'details': f"frame[{i}]:{int(original[i])}->{int(result[i])}, prev@{prev_valid_idx}={prev_valid_val}, next@{next_valid_idx}={next_valid_val}, {gap_info}"
        })
    
    if verbose and n_invalid > 0:
        print(f"   🔧 Interpolated {n_invalid} invalid frames ({100*n_invalid/n:.1f}%)")
    
    return result.tolist(), log_entries


def correct_timestamps_for_drops(timestamps, frame_drops, frame_period_ms=20.0):
    """
    Correct linear fit timestamps for detected frame drops.
    
    Each confirmed drop shifts all subsequent timestamps by one frame period
    to account for the missing frame(s).
    
    Parameters
    ----------
    timestamps : np.ndarray
        Timestamps from linear fit (before drop correction)
    frame_drops : list of dict
        Frame drop info from fit_ocr_frames_robust_linear(), each with:
        - frame_idx: where drop occurred
        - estimated_drops: how many frames dropped
        - confidence: 'high' or 'low'
    frame_period_ms : float
        Expected frame period in milliseconds (default: 20ms for 50Hz)
        
    Returns
    -------
    np.ndarray
        Corrected timestamps with drop offsets applied
    list
        Applied corrections [(frame_idx, offset_ms), ...]
    """
    ts_corrected = timestamps.copy()
    corrections = []
    
    # Only apply corrections for high-confidence drops
    high_conf_drops = sorted(
        [d for d in frame_drops if d['confidence'] == 'high'],
        key=lambda x: x['frame_idx']
    )
    
    cumulative_offset = 0.0
    for drop in high_conf_drops:
        idx = drop['frame_idx']
        n_dropped = drop['estimated_drops']
        offset_ms = n_dropped * frame_period_ms
        
        # All frames after this drop need offset correction
        ts_corrected[idx+1:] += offset_ms / 1000.0
        cumulative_offset += offset_ms
        corrections.append((idx, offset_ms))
    
    return ts_corrected, corrections


def fit_ocr_frames_robust_linear(raw_ocr_frames, optitrack_timestamps, 
                                  residual_threshold=0.003, min_inlier_ratio=0.7,
                                  correct_drops=False, verbose=True,
                                  prefilter=True, prefilter_min_step=3.0,
                                  prefilter_max_step=6.5):
    """
    Use robust linear regression to derive eye camera timestamps from OCR readings.
    
    Since both cameras run at constant clock rates, the relationship between
    eye frame index and aux timestamp must be linear:
    
        aux_time[i] = slope × i + offset
    
    This function uses RANSAC to robustly fit this linear model, automatically
    excluding OCR errors as outliers. The result is continuous timestamps
    directly in the aux timebase - no integer OptiTrack indices needed.
    
    Parameters
    ----------
    raw_ocr_frames : array-like
        Raw OCR readings (OptiTrack frame numbers, 1-based). Can contain 0s/NaNs
        for failed readings - these will be excluded from fitting.
    optitrack_timestamps : array-like  
        OptiTrack event timestamps from database (aux timebase, seconds).
        Index i corresponds to OptiTrack frame i+1.
    residual_threshold : float, optional
        Maximum residual (seconds) to consider a point an inlier.
        Default 0.003s (~3ms). Empirically, inlier residual std is ~2.2ms.
    min_inlier_ratio : float, optional
        Minimum fraction of points to use for initial RANSAC fit.
        Default 0.7 (70%). Empirically, >98% of valid OCR readings are inliers.
    correct_drops : bool, optional
        If True (default), automatically correct timestamps for detected frame
        drops. High-confidence drops shift subsequent timestamps by ~20ms per
        dropped frame. Set False to get raw linear fit timestamps.
    verbose : bool
        Print fit statistics (default: True)
    prefilter : bool
        If True, prefilter valid OCR points using local step consistency
        before running RANSAC.
    prefilter_min_step : float
        Minimum allowed OCR step per eye frame (default: 3.0).
    prefilter_max_step : float
        Maximum allowed OCR step per eye frame (default: 6.5).
        
    Returns
    -------
    timestamps : np.ndarray
        Predicted timestamps for each eye frame (in aux timebase, seconds).
        All frames get a timestamp via linear extrapolation.
    log_entries : list of dict
        Entries for EventInterpolation logging. Each entry contains:
        - eye_frame_idx, correction_type, original_value, corrected_value
        - diff_before (residual in ms), diff_after (predicted step in ms)
        - details (fit info, outlier flag, neighbor context)
    fit_info : dict
        Fit quality metrics:
        - slope, offset: linear model parameters
        - expected_eye_hz: inferred eye camera rate (1/slope)
        - inlier_ratio: fraction of valid OCR points used
        - n_outliers: number of OCR errors detected
        - residual_std_ms: standard deviation of inlier residuals (ms)
        - max_residual_ms: maximum inlier residual (ms)
    """
    from sklearn.linear_model import RANSACRegressor, LinearRegression
    
    raw_ocr = np.array(raw_ocr_frames, dtype=float)
    opti_ts = np.array(optitrack_timestamps, dtype=float)
    n_frames = len(raw_ocr)
    n_opti = len(opti_ts)
    
    if n_frames == 0:
        return np.array([]), [], {'error': 'No frames to process'}
    
    # Eye frame indices (0-based)
    eye_idx = np.arange(n_frames)
    
    # Convert raw OCR readings to timestamps via OptiTrack lookup
    # OCR value is 1-based OptiTrack frame number
    raw_timestamps = np.full(n_frames, np.nan)
    for i, opti_frame in enumerate(raw_ocr):
        # Valid OCR reading: positive integer within OptiTrack range
        if opti_frame >= 1 and opti_frame <= n_opti:
            raw_timestamps[i] = opti_ts[int(opti_frame) - 1]  # Convert to 0-based
    
    # Identify valid readings for fitting
    valid_mask_raw = np.isfinite(raw_timestamps)
    n_valid_raw = np.sum(valid_mask_raw)

    # Optional prefilter: keep points with locally consistent OCR step
    if prefilter and n_valid_raw > 1:
        valid_idx = np.where(valid_mask_raw)[0]
        step_mask = np.zeros(n_frames, dtype=bool)
        for j, idx in enumerate(valid_idx):
            ok = False
            if j > 0:
                prev_idx = valid_idx[j - 1]
                step_prev = (raw_ocr[idx] - raw_ocr[prev_idx]) / (idx - prev_idx)
                if prefilter_min_step <= step_prev <= prefilter_max_step:
                    ok = True
            if not ok and j + 1 < len(valid_idx):
                next_idx = valid_idx[j + 1]
                step_next = (raw_ocr[next_idx] - raw_ocr[idx]) / (next_idx - idx)
                if prefilter_min_step <= step_next <= prefilter_max_step:
                    ok = True
            step_mask[idx] = ok
        valid_mask = valid_mask_raw & step_mask
    else:
        valid_mask = valid_mask_raw

    n_valid = np.sum(valid_mask)
    
    if n_valid < 10:
        # Fallback: not enough valid points for robust fit
        if verbose:
            print(f"   ⚠️ Only {n_valid} valid OCR readings - using simple linear interpolation")
        # Use average expected rate as fallback (empirically validated: eye cameras run at ~50Hz)
        avg_step = 1.0 / 50.0  # 20ms per frame = 50Hz (derived from data, not assumed)
        if n_valid >= 2:
            # If we have at least 2 valid points, derive actual rate from data
            valid_idx = np.where(valid_mask)[0]
            valid_ts = raw_timestamps[valid_mask]
            avg_step = (valid_ts[-1] - valid_ts[0]) / (valid_idx[-1] - valid_idx[0])
        timestamps = np.arange(n_frames, dtype=float) * avg_step
        if n_valid > 0:
            # Anchor to first valid point
            first_valid_idx = np.where(valid_mask)[0][0]
            timestamps = timestamps - timestamps[first_valid_idx] + raw_timestamps[first_valid_idx]
        # CRITICAL: Verify all frames have finite timestamps
        assert len(timestamps) == n_frames, f"Timestamp count mismatch: {len(timestamps)} != {n_frames}"
        assert np.all(np.isfinite(timestamps)), "Fallback interpolation produced non-finite timestamps"
        return timestamps, [], {'error': 'Insufficient valid points', 'n_valid': n_valid, 'n_frames': n_frames}
    
    # Prepare data for RANSAC
    X_valid = eye_idx[valid_mask].reshape(-1, 1)
    y_valid = raw_timestamps[valid_mask]
    
    # Robust linear fit using RANSAC
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=max(10, int(min_inlier_ratio * n_valid)),
        residual_threshold=residual_threshold,
        random_state=42,
        max_trials=1000
    )
    
    try:
        ransac.fit(X_valid, y_valid)
    except Exception as e:
        if verbose:
            print(f"   ⚠️ RANSAC fit failed: {e}")
        # Fallback to simple linear regression
        lr = LinearRegression()
        lr.fit(X_valid, y_valid)
        timestamps = lr.predict(eye_idx.reshape(-1, 1))
        # CRITICAL: Verify all frames have finite timestamps
        assert len(timestamps) == n_frames, f"Timestamp count mismatch: {len(timestamps)} != {n_frames}"
        assert np.all(np.isfinite(timestamps)), "LR fallback produced non-finite timestamps"
        return timestamps, [], {'error': f'RANSAC failed: {e}', 'n_frames': n_frames}
    
    # Extract fit parameters
    slope = ransac.estimator_.coef_[0]
    offset = ransac.estimator_.intercept_
    
    # Predict timestamps for ALL eye frames (linear model guarantees finite output)
    timestamps = ransac.predict(eye_idx.reshape(-1, 1))
    
    # CRITICAL: Verify every frame has a finite timestamp
    assert len(timestamps) == n_frames, f"Timestamp count mismatch: {len(timestamps)} != {n_frames}"
    assert np.all(np.isfinite(timestamps)), "Linear fit produced non-finite timestamps"
    
    # Compute residuals for valid points
    inlier_mask_valid = ransac.inlier_mask_  # Relative to valid points only
    predicted_valid = ransac.predict(X_valid)
    residuals_valid = y_valid - predicted_valid  # seconds
    
    # Map inlier mask back to full frame array
    inlier_mask_full = np.zeros(n_frames, dtype=bool)
    inlier_mask_full[valid_mask] = inlier_mask_valid
    
    residuals_full = np.full(n_frames, np.nan)
    residuals_full[valid_mask] = residuals_valid
    
    # Fit quality metrics
    inlier_residuals = residuals_valid[inlier_mask_valid]
    fit_info = {
        'n_frames': n_frames,  # Total eye camera frames - ALL get timestamps
        'slope': slope,
        'offset': offset,
        'expected_eye_hz': 1.0 / slope if slope > 0 else np.nan,
        'inlier_ratio': np.mean(inlier_mask_valid),
        'n_valid': n_valid,
        'n_valid_raw': n_valid_raw,
        'n_prefiltered': int(n_valid_raw - n_valid),
        'prefilter_min_step': prefilter_min_step if prefilter else None,
        'prefilter_max_step': prefilter_max_step if prefilter else None,
        'n_inliers': np.sum(inlier_mask_valid),
        'n_outliers': np.sum(~inlier_mask_valid),
        'residual_std_ms': np.std(inlier_residuals) * 1000 if len(inlier_residuals) > 0 else np.nan,
        'max_residual_ms': np.max(np.abs(inlier_residuals)) * 1000 if len(inlier_residuals) > 0 else np.nan,
    }
    
    # ---------- Generate Log Entries for EventInterpolation ----------
    log_entries = []
    
    # Compute frame period for use in details
    frame_period_ms = slope * 1000  # ~20ms for 50Hz
    
    # Log RANSAC outliers (OCR errors detected by robust fit)
    outlier_indices = np.where(valid_mask)[0][~inlier_mask_valid]
    for idx in outlier_indices:
        residual_ms = residuals_full[idx] * 1000 if np.isfinite(residuals_full[idx]) else None
        ocr_val = int(raw_ocr[idx]) if np.isfinite(raw_ocr[idx]) else 0
        
        # Build verbose details string (ASCII-safe, max 256 chars)
        details_parts = [
            f"frame={idx}",
            f"ocr={ocr_val}",
            f"resid={residual_ms:.1f}ms" if residual_ms else "resid=N/A",
            f"thresh={residual_threshold*1000:.1f}ms",
            f"period={frame_period_ms:.2f}ms",
            f"eye_hz={fit_info['expected_eye_hz']:.2f}"
        ]
        details = ", ".join(details_parts)[:256]
        
        log_entries.append({
            'eye_frame_idx': int(idx),
            'correction_type': 'RANSAC_OUTLIER',
            'original_value': ocr_val,
            'corrected_value': ocr_val,  # Not corrected, just flagged
            'diff_before': int(round(residual_ms)) if residual_ms is not None else None,
            'diff_after': int(round(frame_period_ms)),  # Expected frame period in ms
            'details': details
        })
    
    # ---------- Frame Drop Detection ----------
    # Detect potential frame drops by analyzing OCR jump patterns
    # A real drop shows: large OCR jump (9-10+) but normal ~20ms linear interval
    
    frame_drops = []
    raw_ocr_clean = raw_ocr.copy()
    raw_ocr_clean[~np.isfinite(raw_ocr_clean)] = 0
    
    if len(raw_ocr_clean) > 1:
        ocr_diffs = np.diff(raw_ocr_clean)
        
        for i in range(len(ocr_diffs)):
            # Look for large OCR jumps (normal is 4-5, drop shows 9-10+)
            if ocr_diffs[i] > 8:
                # Estimate how many frames were dropped
                estimated_drops = round((ocr_diffs[i] - 4.8) / 4.8)
                
                if estimated_drops >= 1:
                    # Cross-validate: check if linear fit shows normal interval
                    # (which it will, since linear fit ignores the OCR value)
                    linear_interval_ms = slope * 1000  # expected ~20ms
                    
                    # Check residual pattern: does the offset shift after this point?
                    if i + 50 < n_frames and i >= 50:
                        residuals_before = residuals_full[max(0, i-50):i]
                        residuals_after = residuals_full[i+1:min(n_frames, i+51)]
                        
                        median_before = np.nanmedian(residuals_before) * 1000  # ms
                        median_after = np.nanmedian(residuals_after) * 1000    # ms
                        offset_shift = median_after - median_before
                        
                        # Large offset shift confirms a drop
                        confidence = 'high' if abs(offset_shift) > 15 else 'low'
                    else:
                        offset_shift = np.nan
                        confidence = 'low'
                    
                    frame_drops.append({
                        'frame_idx': i,
                        'ocr_jump': int(ocr_diffs[i]),
                        'estimated_drops': estimated_drops,
                        'offset_shift_ms': offset_shift,
                        'confidence': confidence
                    })
                    
                    # Log frame drop for EventInterpolation
                    correction_type = 'RANSAC_FRAME_DROP_HIGH' if confidence == 'high' else 'RANSAC_FRAME_DROP_LOW'
                    
                    # Calculate correction magnitude that would be applied
                    correction_ms = estimated_drops * frame_period_ms
                    
                    # Build verbose details string (ASCII-safe, max 256 chars)
                    ocr_before = int(raw_ocr_clean[i])
                    ocr_after = int(raw_ocr_clean[i+1]) if i+1 < len(raw_ocr_clean) else 0
                    
                    details_parts = [
                        f"frame={i}",
                        f"ocr:{ocr_before}->{ocr_after}",
                        f"jump={int(ocr_diffs[i])}",
                        f"drops={estimated_drops}",
                        f"offset_shift={offset_shift:.1f}ms" if np.isfinite(offset_shift) else "offset_shift=N/A",
                        f"correction={correction_ms:.1f}ms",
                        f"period={frame_period_ms:.2f}ms",
                        f"conf={confidence}"
                    ]
                    details = ", ".join(details_parts)[:256]
                    
                    log_entries.append({
                        'eye_frame_idx': i,
                        'correction_type': correction_type,
                        'original_value': ocr_before,
                        'corrected_value': ocr_after,
                        'diff_before': int(ocr_diffs[i]),  # OCR jump
                        'diff_after': estimated_drops,     # Estimated dropped frames
                        'details': details
                    })
    
    fit_info['frame_drops'] = frame_drops
    fit_info['n_drops_detected'] = len([d for d in frame_drops if d['confidence'] == 'high'])
    
    if verbose:
        if prefilter:
            print(f"      Prefilter: {n_valid}/{n_valid_raw} valid OCR points (step {prefilter_min_step:.1f}-{prefilter_max_step:.1f})")
        print(f"   📈 Robust linear fit: slope={slope*1000:.4f} ms/frame ({fit_info['expected_eye_hz']:.2f} Hz)")
        print(f"      Frames: {n_frames} total, all with timestamps")
        print(f"      Inliers: {fit_info['n_inliers']}/{n_valid} ({100*fit_info['inlier_ratio']:.1f}%)")
        print(f"      Outliers (OCR errors): {fit_info['n_outliers']}")
        print(f"      Residual std: {fit_info['residual_std_ms']:.3f} ms, max: {fit_info['max_residual_ms']:.3f} ms")
        
        if frame_drops:
            high_conf = [d for d in frame_drops if d['confidence'] == 'high']
            low_conf = [d for d in frame_drops if d['confidence'] == 'low']
            if high_conf:
                print(f"      ⚠️ FRAME DROPS DETECTED: {len(high_conf)} high-confidence")
                for d in high_conf[:5]:  # Show first 5
                    print(f"         Frame {d['frame_idx']}: OCR jumped {d['ocr_jump']}, ~{d['estimated_drops']} dropped, offset shift {d['offset_shift_ms']:.1f}ms")
            if low_conf:
                print(f"      ❓ Potential drops (low confidence): {len(low_conf)}")
        else:
            print(f"      ✅ No frame drops detected")
    
    # ---------- Apply Frame Drop Correction ----------
    drop_corrections = []
    if correct_drops and fit_info['n_drops_detected'] > 0:
        frame_period_ms = slope * 1000  # Use actual fitted frame period
        timestamps, drop_corrections = correct_timestamps_for_drops(
            timestamps, frame_drops, frame_period_ms=frame_period_ms
        )
        fit_info['drop_corrections_applied'] = drop_corrections
        fit_info['timestamps_corrected'] = True
        if verbose:
            total_offset = sum(c[1] for c in drop_corrections)
            print(f"      🔧 Applied {len(drop_corrections)} drop corrections (total offset: {total_offset:.1f}ms)")
    else:
        fit_info['drop_corrections_applied'] = []
        fit_info['timestamps_corrected'] = False
    
    return timestamps, log_entries, fit_info


def _infer_camera_role(camera_type, video_path):
    label = str(camera_type).lower() if camera_type is not None else ""
    path_label = str(video_path).lower()
    combined = f"{label} {path_label}"
    if "eye" in combined:
        if "left" in combined:
            return "eye_left"
        if "right" in combined:
            return "eye_right"
        return "eye"
    if "top" in combined or "world" in combined or "ego" in combined or "worldcam" in combined:
        return "world"
    return "unknown"


def _split_eye_world_summary(camera_data):
    eye_summary = {}
    world_summary = {}
    for camera_type, data in camera_data.items():
        summary = {
            'n_frames': data['n_frames'],
            'n_valid': data['n_valid'],
            'log_path': data['log_path']
        }
        role = (data.get('camera_role') or '').lower()
        if role == "world":
            world_summary[camera_type] = summary
        elif role.startswith("eye"):
            eye_summary[camera_type] = summary
        else:
            eye_summary[camera_type] = summary
    return eye_summary, world_summary


def _safe_video_fps(video_path):
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    except Exception:
        return None
    if fps is None or not np.isfinite(fps) or fps <= 0:
        return None
    return float(fps)


def ensure_deinterlaced_video(video_path, camera_type=None, input_fps=25.0,
                              output_suffix="deinterlaced", verbose=True):
    """
    Ensure eye/worldcam videos are deinterlaced (PAL default 25 Hz -> 50 Hz).
    Set input_fps=29.97 for NTSC sources if needed.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return {'error': f'Video file not found: {video_path}'}

    label = f"{camera_type or ''} {video_path.name}".lower()
    needs_deinterlace = ("eye" in label) or ("worldcam" in label) or ("world_cam" in label)
    if not needs_deinterlace:
        return {'success': True, 'path': video_path, 'skipped': True, 'reason': 'not_eye_or_worldcam'}

    name_has_deint = "deinterlaced" in video_path.name.lower()
    fps = _safe_video_fps(video_path)
    expected_deint_fps = float(input_fps) * 2.0 if input_fps else None

    if expected_deint_fps and fps and fps >= expected_deint_fps * 0.9:
        if verbose and not name_has_deint:
            print(f"   ℹ️ {video_path.name}: fps={fps:.2f} Hz suggests deinterlaced (using as-is)")
        return {'success': True, 'path': video_path, 'skipped': True, 'reason': 'fps_deinterlaced'}

    if name_has_deint:
        if verbose and expected_deint_fps and fps and fps < expected_deint_fps * 0.9:
            print(f"   ⚠️ {video_path.name}: name says deinterlaced but fps={fps:.2f} Hz")
        return {'success': True, 'path': video_path, 'skipped': True, 'reason': 'name_deinterlaced'}

    output_path = video_path.with_name(f"{video_path.stem}_{output_suffix}{video_path.suffix}")
    if output_path.exists():
        out_fps = _safe_video_fps(output_path)
        if expected_deint_fps and out_fps and out_fps >= expected_deint_fps * 0.9:
            if verbose:
                print(f"   ✅ Using existing deinterlaced file: {output_path.name}")
            return {'success': True, 'path': output_path, 'skipped': True, 'reason': 'existing_deinterlaced'}
        if verbose:
            print(f"   ⚠️ Existing deinterlaced file has fps={out_fps}; reprocessing")

    command = [
        'ffmpeg',
        '-y',
        '-r', str(input_fps),
        '-i', str(video_path),
        '-vf', 'bwdif=mode=1',
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '20',
        '-c:a', 'copy',
        str(output_path),
    ]

    if verbose:
        print(f"   🔧 Deinterlacing {video_path.name} -> {output_path.name}")

    try:
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    except Exception as exc:
        err = getattr(exc, 'stderr', None) or (proc.stderr if 'proc' in locals() else str(exc))
        return {'error': f'FFmpeg deinterlace failed: {err}'}

    if not output_path.exists():
        return {'error': f'Deinterlaced output not found: {output_path}'}

    out_fps = _safe_video_fps(output_path)
    if expected_deint_fps and out_fps and out_fps < expected_deint_fps * 0.9:
        if verbose:
            print(f"   ⚠️ Deinterlaced fps={out_fps:.2f} Hz (expected ~{expected_deint_fps:.2f} Hz)")

    return {'success': True, 'path': output_path, 'skipped': False, 'reason': 'deinterlaced'}


def _prefer_deinterlaced_video_files(video_files):
    if not video_files:
        return []

    grouped = {}
    for info in video_files:
        rec_id = info.get('recording_id') or info.get('camera') or info.get('file_path')
        key = (info.get('session_id'), info.get('scan_id'), rec_id)
        grouped.setdefault(key, []).append(info)

    selected = []
    for group in grouped.values():
        deint = [
            info for info in group
            if 'deinterlaced' in str(info.get('file_path', '')).lower()
        ]
        candidates = deint or group
        chosen = min(
            candidates,
            key=lambda info: int(info.get('file_id') or 0)
        )
        selected.append(chosen)

    return selected


def _register_deinterlaced_video_file(video_info, deinterlaced_path, verbose=False):
    if not deinterlaced_path:
        return None

    session_id = video_info.get('session_id')
    scan_id = video_info.get('scan_id')
    recording_id = video_info.get('recording_id')
    camera_type = video_info.get('camera')

    if not recording_id and session_id and scan_id and camera_type:
        try:
            recording_id = (model.VideoRecordingNew & {
                'session_id': session_id,
                'scan_id': scan_id,
                'camera': camera_type
            }).fetch1('recording_id')
        except Exception as exc:
            if verbose:
                print(f"   Warning: could not resolve recording_id for {camera_type}: {exc}")
            return None

    if not (session_id and scan_id and recording_id):
        if verbose:
            print("   Warning: missing identifiers; skipping VideoRecordingNew.File insert")
        return None

    file_key = {
        'session_id': session_id,
        'scan_id': scan_id,
        'recording_id': recording_id
    }

    existing = (model.VideoRecordingNew.File & file_key).fetch(
        'file_id', 'file_path', as_dict=True
    )
    deint_path_str = str(deinterlaced_path)
    for row in existing:
        if str(row.get('file_path')) == deint_path_str:
            return {'status': 'exists', 'file_id': row.get('file_id')}

    next_id = max([int(row.get('file_id', -1)) for row in existing], default=-1) + 1
    insert_key = {
        **file_key,
        'file_id': int(next_id),
        'file_path': deint_path_str
    }

    try:
        model.VideoRecordingNew.File.insert1(
            insert_key,
            ignore_extra_fields=True,
            skip_duplicates=True
        )
    except Exception as exc:
        if verbose:
            print(f"   Warning: failed to insert deinterlaced file ({exc})")
        return {'status': 'error', 'error': str(exc)}

    if verbose:
        print(f"   Registered deinterlaced file (file_id={next_id})")

    return {'status': 'inserted', 'file_id': next_id}


def _resolve_deinterlaced_video_path(video_info, input_fps=25.0, verbose=False):
    video_path = Path(video_info['file_path'])
    camera_type = video_info.get('camera')
    deint_result = ensure_deinterlaced_video(
        video_path,
        camera_type=camera_type,
        input_fps=input_fps,
        verbose=verbose,
    )
    if 'error' in deint_result:
        return deint_result, video_path

    resolved_path = Path(deint_result.get('path', video_path))
    if str(resolved_path) != str(video_path):
        _register_deinterlaced_video_file(
            video_info,
            resolved_path,
            verbose=verbose
        )

    return deint_result, resolved_path


def _resolve_scan_dir(scan_key, verbose=False):
    try:
        scan_path_rel = (scan.ScanPath & scan_key).fetch1('path')
    except Exception as exc:
        return None, f'Failed to fetch ScanPath: {exc}'

    scan_dir = Path(scan_path_rel)
    if scan_dir.exists():
        return scan_dir, None

    roots = get_experiment_root_data_dir() or []
    if not isinstance(roots, (list, tuple)):
        roots = [roots]

    if roots:
        try:
            scan_dir = find_full_path(roots, scan_path_rel)
        except Exception as exc:
            if verbose:
                print(f"Warning: could not resolve scan path via roots ({exc})")

    if scan_dir is not None and Path(scan_dir).exists():
        return Path(scan_dir), None

    return Path(scan_path_rel), f"Scan directory not found: {scan_path_rel}"


def _camera_video_patterns(camera_type):
    label = str(camera_type).lower()
    if "eye_left" in label:
        return [
            "*left_eye*video*.mp4*",
            "*eye_left*video*.mp4*",
            "*left*eye*video*.mp4*",
        ]
    if "eye_right" in label:
        return [
            "*right_eye*video*.mp4*",
            "*eye_right*video*.mp4*",
            "*right*eye*video*.mp4*",
        ]
    if "worldcam" in label or "world" in label:
        return [
            "*worldcam*video*.mp4*",
            "*headcam*worldcam*video*.mp4*",
        ]
    return [
        f"*{camera_type}*video*.mp4*",
        f"*{camera_type}*.mp4*",
    ]


def register_missing_video_recordings(session_key, scan_key, camera_types, verbose=True):
    """
    Register missing VideoRecordingNew entries by scanning the scan directory for mp4 files.
    """
    if isinstance(session_key, dict):
        session_id = session_key.get('session_id')
    else:
        session_id = session_key

    if isinstance(scan_key, dict):
        scan_id = scan_key.get('scan_id')
        session_id = session_id or scan_key.get('session_id')
    else:
        scan_id = scan_key

    if not session_id or not scan_id:
        return {'success': False, 'error': 'Missing session_id or scan_id', 'results': {}}

    key = {'session_id': session_id, 'scan_id': scan_id}
    scan_dir, scan_err = _resolve_scan_dir(key, verbose=verbose)
    if scan_err:
        return {'success': False, 'error': scan_err, 'results': {}}

    results = {}
    seen = set()
    camera_list = []
    for cam in camera_types or []:
        if cam not in seen:
            seen.add(cam)
            camera_list.append(cam)

    for camera_type in camera_list:
        rec_id = f"{scan_id}_{camera_type}"
        result = {'recording_id': rec_id, 'status': None}

        existing = (model.VideoRecordingNew & key & f'camera="{camera_type}"')
        if len(existing) > 0:
            file_count = len((model.VideoRecordingNew.File & existing))
            result.update(status='exists', file_count=file_count)
            results[camera_type] = result
            continue

        matches = []
        for pattern in _camera_video_patterns(camera_type):
            matches = sorted(Path(scan_dir).glob(pattern))
            if matches:
                break

        if not matches:
            result.update(status='missing', error=f'No video file found in {scan_dir}')
            results[camera_type] = result
            if verbose:
                print(f"   ⚠️ {camera_type}: no mp4 found in {scan_dir}")
            continue

        if len(matches) > 1 and verbose:
            print(f"   ⚠️ Multiple {camera_type} videos found, using {matches[0].name}")

        video_path = matches[0]
        try:
            insert_key = {**key, 'recording_id': rec_id, 'camera': camera_type}
            model.VideoRecordingNew.insert1(insert_key, skip_duplicates=True)
            model.VideoRecordingNew.File.insert1(
                {**insert_key, 'file_id': 0, 'file_path': str(video_path)},
                ignore_extra_fields=True,
                skip_duplicates=True
            )
            result.update(status='inserted', file_path=str(video_path))
        except Exception as exc:
            result.update(status='error', error=str(exc))
            if verbose:
                print(f"   ⚠️ {camera_type}: registration failed ({exc})")

        results[camera_type] = result

    success = all(r.get('status') in ('exists', 'inserted') for r in results.values())
    return {
        'success': success,
        'scan_dir': str(scan_dir),
        'results': results
    }


def _eye_ocr_config(side, is_high_res):
    if side == "left":
        if is_high_res:
            width, height, y = 10 * 2, 12 * 2, 255 * 2
            x_coords = [17 * 2, 29 * 2, 41 * 2, 53 * 2, 65 * 2, 77 * 2]
            threshold = 0.6
        else:
            width, height, y = 10, 12, 255
            x_coords = [17, 29, 41, 53, 65, 77]
            threshold = 0.7
    else:  # right eye
        if is_high_res:
            width, height, y = 10 * 2, 12 * 2, 255 * 2
            x_coords = [19 * 2, 31 * 2, 43 * 2, 55 * 2, 67 * 2, 79 * 2]
            threshold = 0.6
        else:
            width, height, y = 10, 12, 255
            x_coords = [19, 31, 43, 55, 67, 79]
            threshold = 0.7

    return {
        'x_coords': x_coords,
        'y': y,
        'width': width,
        'height': height,
        'threshold': threshold,
        'resize_back_to_orig': is_high_res,
        'orig_width': 10,
        'orig_height': 12,
        'label': f"eye_{side}_{'high' if is_high_res else 'std'}"
    }


def _scaled_ocr_config(side, scale, frame_width, frame_height, label=None):
    base_x = [17, 29, 41, 53, 65, 77] if side == "left" else [19, 31, 43, 55, 67, 79]
    width = max(1, int(round(10 * scale)))
    height = max(1, int(round(12 * scale)))
    y = int(round(255 * scale))
    x_coords = [int(round(x * scale)) for x in base_x]

    threshold = 0.6 if scale >= 1.5 else 0.7
    resize_back_to_orig = scale >= 1.5

    if frame_height is not None:
        y = min(max(0, y), max(0, frame_height - height))
    if frame_width is not None:
        x_coords = [min(max(0, x), max(0, frame_width - width)) for x in x_coords]

    return {
        'x_coords': x_coords,
        'y': y,
        'width': width,
        'height': height,
        'threshold': threshold,
        'resize_back_to_orig': resize_back_to_orig,
        'orig_width': 10,
        'orig_height': 12,
        'label': label or f"{side}_scale_{scale:.2f}"
    }


def _candidate_ocr_configs(camera_role, video_path, frame_width, frame_height, is_high_res):
    if camera_role in ("eye_left", "eye_right", "eye"):
        side = "left"
        if camera_role == "eye_right":
            side = "right"
        elif camera_role == "eye":
            path_label = str(video_path).lower()
            side = "right" if "right" in path_label else "left"
        return [_eye_ocr_config(side, is_high_res)], side

    base_height = 288.0
    scale = frame_height / base_height if frame_height else 1.0
    scales = [scale] if abs(scale - 1.0) < 0.05 else [scale, 1.0]
    configs = []
    for side in ("left", "right"):
        for s in scales:
            label = f"{side}_scale_{s:.2f}"
            configs.append(_scaled_ocr_config(side, s, frame_width, frame_height, label=label))
    return configs, None


def _ocr_digits_from_frame(frame, config, logreg_model):
    digits = []
    x_coords = config['x_coords']
    y = config['y']
    width = config['width']
    height = config['height']
    threshold = config['threshold']
    resize_back_to_orig = config.get('resize_back_to_orig', False)
    orig_width = config.get('orig_width', width)
    orig_height = config.get('orig_height', height)

    for x_start in x_coords:
        crop = frame[y:y+height, x_start:x_start+width]

        if crop.size == 0:
            digits.append(None)
            continue

        if resize_back_to_orig:
            try:
                crop = cv2.resize(crop, (orig_width, orig_height),
                                  interpolation=cv2.INTER_AREA)
            except Exception:
                digits.append(None)
                continue
        else:
            if crop.shape[:2] != (height, width):
                crop = cv2.resize(crop, (width, height),
                                  interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        feat = gray.reshape(1, -1).astype(np.float32)

        probs = logreg_model.predict_proba(feat)
        max_p = float(probs.max())
        pred_class = int(probs.argmax())

        if pred_class == 10 or max_p < threshold:
            pred = None
        else:
            pred = pred_class

        digits.append(pred)

    accepted_digits = [str(d) for d in digits if d is not None]
    if accepted_digits:
        return ''.join(accepted_digits)
    return None


def _probe_ocr_configs(cap, configs, logreg_model, n_probe=200):
    counts = [0 for _ in configs]
    frames_checked = 0

    while frames_checked < n_probe:
        ret, frame = cap.read()
        if not ret:
            break
        for idx, config in enumerate(configs):
            if _ocr_digits_from_frame(frame, config, logreg_model) is not None:
                counts[idx] += 1
        frames_checked += 1

    return counts, frames_checked


def extract_eye_camera_frames_ocr(video_path, model_path=None, verbose=True, write_log=True,
                                  apply_corrections=True, camera_type=None, crop_config=None):
    """
    Extract OptiTrack frame indices from eye camera video using OCR digit recognition.
    
    This function is for eye cameras that display frame numbers as text overlay,
    as opposed to binary LED encoding used in extract_eye_camera_frames().
    
    Parameters
    ----------
    video_path : str or Path
        Path to eye camera video file
    model_path : str or Path, optional
        Path to digit recognition model (joblib file).
        If None, uses default path: adamacs/user_data/other models/digit_model.joblib
    verbose : bool
        Print progress information (default: True)
    write_log : bool
        Write detailed correction log to CSV file in same folder as video (default: True).
        Ignored when apply_corrections=False.
    apply_corrections : bool
        If True, run heuristic OCR corrections and generate correction logs.
        If False, return raw OCR indices only (no correction log).
    camera_type : str, optional
        Camera identifier used to select OCR crop defaults (e.g., eye/top/world).
    crop_config : dict, optional
        Explicit OCR crop configuration to override auto-detection.
    
    Returns
    -------
    dict : Processing results including:
        - frame_indices_raw: Raw OCR-extracted frame numbers (strings)
        - frame_indices: Corrected frame indices (integers)
        - n_frames: Total number of video frames
        - n_valid: Number of frames with at least one digit recognized
        - log_path: Path to correction log file (if write_log=True)
        - error: Error message if processing failed
    """
    from datetime import datetime
    from joblib import load
    
    video_path = Path(video_path)
    if not video_path.exists():
        return {'error': f'Video file not found: {video_path}'}
    
    # Load digit recognition model
    if model_path is None:
        # Default path relative to adamacs package
        adamacs_root = Path(__file__).parent.parent.parent
        model_path = adamacs_root / "user_data" / "other models" / "digit_model.joblib"
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        return {'error': f'Digit model not found at: {model_path}'}
    
    try:
        logreg_model = load(model_path)
    except Exception as e:
        return {'error': f'Failed to load digit model: {e}'}
    
    if verbose:
        print(f"🎬 Loading video: {video_path.name}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {'error': f'Cannot open video: {video_path}'}
    
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if verbose:
        print(f"   📊 Video info: {n_frames} frames, {frame_width}x{frame_height}, {fps:.1f} FPS")
    
    # Auto-detect parameters based on video resolution and camera type
    is_high_res = frame_height > 300
    if verbose:
        print(f"   📍 Using {'high-res' if is_high_res else 'standard'} parameters")

    camera_role = _infer_camera_role(camera_type, video_path)
    side = None

    if crop_config is not None:
        configs = [crop_config]
        selected_config = crop_config
        config_label = crop_config.get('label', 'custom') if isinstance(crop_config, dict) else 'custom'
    else:
        configs, side = _candidate_ocr_configs(
            camera_role, video_path, frame_width, frame_height, is_high_res
        )
        selected_config = configs[0]
        config_label = selected_config.get('label', 'auto')

        if len(configs) > 1:
            if verbose:
                print("   Selecting OCR crop config...")
            counts, frames_checked = _probe_ocr_configs(cap, configs, logreg_model, n_probe=200)
            if frames_checked > 0:
                best_idx = int(np.argmax(counts))
                selected_config = configs[best_idx]
                config_label = selected_config.get('label', 'auto')
                if verbose:
                    print(f"   OCR config selected: {config_label} (probe {counts[best_idx]}/{frames_checked})")

            if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                cap.release()
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    return {'error': f'Cannot reopen video: {video_path}'}

    if verbose:
        if side in ("left", "right"):
            print(f"   👁 Eye side detected: {side}")
        else:
            print(f"   OCR config: {config_label}")

    x_coords = selected_config['x_coords']
    y = selected_config['y']
    width = selected_config['width']
    height = selected_config['height']
    threshold = selected_config['threshold']
    resize_back_to_orig = selected_config.get('resize_back_to_orig', False)
    orig_width = selected_config.get('orig_width', 10)
    orig_height = selected_config.get('orig_height', 12)

    # Process frames
    n_digits = len(x_coords)
    processed_numbers = []
    frames_with_any = 0
    frames_all_rejected = 0

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        digits = []

        for j, x_start in enumerate(x_coords):
            crop = frame[y:y+height, x_start:x_start+width]

            if crop.size == 0:
                digits.append(None)
                continue

            # Resize for high-res videos
            if resize_back_to_orig:
                try:
                    crop = cv2.resize(crop, (orig_width, orig_height),
                                      interpolation=cv2.INTER_AREA)
                except:
                    digits.append(None)
                    continue
            else:
                if crop.shape[:2] != (height, width):
                    crop = cv2.resize(crop, (width, height),
                                      interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            feat = gray.reshape(1, -1).astype(np.float32)

            probs = logreg_model.predict_proba(feat)
            max_p = float(probs.max())
            pred_class = int(probs.argmax())

            if pred_class == 10 or max_p < threshold:
                pred = None
            else:
                pred = pred_class

            digits.append(pred)

        accepted_digits = [str(d) for d in digits if d is not None]
        if accepted_digits:
            processed_numbers.append(''.join(accepted_digits))
            frames_with_any += 1
        else:
            processed_numbers.append(None)
            frames_all_rejected += 1

    cap.release()
    
    if verbose:
        print(f"   ✅ OCR complete: {frames_with_any}/{n_frames} frames with digits")

    log_path = None
    correction_log = []

    if apply_corrections:
        # Generate log path
        if write_log:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_stem = video_path.stem
            log_filename = f"{video_stem}_frame_correction_log_{timestamp}.csv"
            log_path = video_path.parent / log_filename

        # Apply frame correction
        frame_indices_corrected, correction_log = fix_ocr_frames(
            processed_numbers, verbose=verbose, log_path=log_path
        )
    else:
        frame_indices_corrected = _ocr_to_int_list(processed_numbers, missing_value=0).tolist()

    return {
        'frame_indices_raw': processed_numbers,
        'frame_indices': frame_indices_corrected,
        'correction_log': correction_log,  # Log entries for EventInterpolation table
        'n_frames': n_frames,
        'n_valid': frames_with_any,
        'n_rejected': frames_all_rejected,
        'video_path': str(video_path),
        'log_path': str(log_path) if log_path else None,
        'fps': fps,
        'resolution': (frame_width, frame_height),
        'eye_side': side,
        'camera_type': camera_type,
        'camera_role': camera_role,
        'ocr_config_label': config_label
    }


def ingest_egocams_ocr(session_key, scan_key, eye_cameras=['mini2p1_eye_left', 'mini2p1_eye_right'],
                       model_path=None, verbose=True, write_log=True,
                       apply_corrections=True, deinterlace_input_fps=25.0):
    """
    Ingest eye camera synchronization data using OCR digit extraction.
    Also process worldcam videos detected by 'worldcam' in the filename.
    
    This function is for eye cameras that display OptiTrack frame numbers as text overlay.
    It uses a trained digit recognition model to extract frame indices, then synchronizes
    with OptiTrack timestamps.
    
    Parameters
    ----------
    session_key : str or dict
        Session identifier
    scan_key : str or dict
        Scan identifier
    eye_cameras : list, optional
        List of eye camera types to process (default: left and right eye)
    model_path : str or Path, optional
        Path to digit recognition model. If None, uses default location.
    verbose : bool
        Print detailed progress information (default: True)
    write_log : bool
        Write correction logs for each video (default: True)
    apply_corrections : bool
        Apply heuristic OCR corrections before syncing (default: True).
    deinterlace_input_fps : float
        Input FPS for deinterlacing interlaced PAL/NTSC sources (default: 25.0).
        Set to 29.97 for NTSC if needed.
        
    Returns
    -------
    dict : Processing results and statistics
    """
    if verbose:
        print(f"🎥 Starting OCR-based eye camera ingestion for session {session_key}, scan {scan_key}")
    
    # Step 1: Get eye camera video files from database
    try:
        eye_video_key = (model.VideoRecordingNew & {'session_id': session_key, 'scan_id': scan_key} &
                         f'camera IN {tuple(eye_cameras)}').fetch('KEY')
        eye_video_files = (model.VideoRecordingNew * model.VideoRecordingNew.File & eye_video_key).fetch(as_dict=True)
    except Exception as e:
        return {'error': f'Failed to fetch video files from database: {e}'}

    eye_video_files = _prefer_deinterlaced_video_files(eye_video_files)

    if len(eye_video_files) == 0:
        if verbose:
            print("❌ No eye camera videos found in database")
            all_cameras = (model.VideoRecordingNew & {'session_id': session_key, 'scan_id': scan_key}).fetch('camera')
            if len(all_cameras) > 0:
                print("Available cameras:")
                for cam in set(all_cameras):
                    print(f"   - {cam}")
        return {'error': 'No eye camera videos found'}

    # Find worldcam videos by filename
    worldcam_files = []
    try:
        all_video_files = (model.VideoRecordingNew * model.VideoRecordingNew.File &
                          {'session_id': session_key, 'scan_id': scan_key}).fetch(as_dict=True)
        all_video_files = _prefer_deinterlaced_video_files(all_video_files)
        worldcam_files = [
            vf for vf in all_video_files
            if 'worldcam' in str(vf.get('file_path', '')).lower()
        ]
    except Exception as e:
        if verbose:
            print(f"Warning: failed to fetch worldcam videos: {e}")

    video_by_path = {vf['file_path']: vf for vf in eye_video_files}
    for vf in worldcam_files:
        video_by_path.setdefault(vf['file_path'], vf)
    video_files = list(video_by_path.values())
    worldcam_paths = {str(vf['file_path']).lower() for vf in worldcam_files}

    if len(video_files) == 0:
        return {'error': 'No camera videos found'}

    if verbose:
        print(f"📹 Found {len(video_files)} camera videos (eyes={len(eye_video_files)}, worldcam={len(worldcam_files)})")

    # Step 2: Extract frame indices using OCR
    eye_video_data = {}
    camera_errors = {}

    for video_info in video_files:
        camera_type = video_info['camera']
        file_path = Path(video_info['file_path'])
        is_worldcam = (
            'worldcam' in str(camera_type).lower()
            or 'worldcam' in str(file_path).lower()
            or str(video_info['file_path']).lower() in worldcam_paths
        )

        deint_result, file_path = _resolve_deinterlaced_video_path(
            video_info,
            input_fps=deinterlace_input_fps,
            verbose=verbose,
        )
        if 'error' in deint_result:
            if is_worldcam:
                camera_errors[file_path.name] = deint_result['error']
                if verbose:
                    print(f"   Warning: skipping worldcam {file_path.name}: {deint_result['error']}")
                continue
            return {'error': f"Failed to deinterlace {camera_type}: {deint_result['error']}"}

        if verbose:
            print(f"\n🔄 Processing {camera_type}: {file_path.name}")

        result = extract_eye_camera_frames_ocr(
            file_path,
            model_path=model_path,
            verbose=verbose,
            write_log=write_log,
            apply_corrections=apply_corrections,
            camera_type=camera_type
        )

        if 'error' not in result:
            eye_video_data[camera_type] = result
        else:
            if is_worldcam:
                camera_errors[file_path.name] = result['error']
                if verbose:
                    print(f"   Warning: skipping worldcam {file_path.name}: {result['error']}")
                continue
            if verbose:
                print(f"   ❌ Error processing {camera_type}: {result['error']}")
            return {'error': f"Failed to process {camera_type}: {result['error']}"}

    if not eye_video_data:
        return {'error': 'No camera data extracted successfully'}
    
    # Step 3: Load OptiTrack event timestamps
    try:
        optitrack_events = (event.Event & {'session_id': session_key, 'scan_id': scan_key} & 
                           'event_type="optitrack_frames"').fetch(
            'event_start_time', order_by='event_start_time'
        ).astype(float)
        
        if len(optitrack_events) == 0:
            if verbose:
                print("❌ No OptiTrack events found!")
                available_events = (event.Event & {'session_id': session_key, 'scan_id': scan_key}).fetch('event_type')
                if len(available_events) > 0:
                    print("Available event types:")
                    for event_type in set(available_events):
                        count = len((event.Event & {'session_id': session_key, 'scan_id': scan_key} & 
                                   f'event_type="{event_type}"'))
                        print(f"   - {event_type}: {count} events")
            return {'error': 'No OptiTrack events found'}
        
        if verbose:
            print(f"\n✅ Loaded {len(optitrack_events)} OptiTrack events")
            
    except Exception as e:
        return {'error': f'Failed to load OptiTrack events: {e}'}
    
    # Step 4: Synchronize eye camera frames with OptiTrack timing
    eye_timestamps = {}
    
    for camera_type, data in eye_video_data.items():
        frame_indices = np.array(data['frame_indices'])
        # NOTE: Zero-checking is redundant since interpolate_frame_indices() already
        # fills all zeros with interpolated values before we reach this point.
        # valid_mask = frame_indices > 0
        # valid_indices = frame_indices[valid_mask]
        # 
        # if len(valid_indices) == 0:
        #     if verbose:
        #         print(f"   ⚠️ No valid frame indices for {camera_type}")
        #     continue
        
        # Map frame indices to OptiTrack timestamps
        # Frame indices are 1-based OptiTrack frame numbers
        timestamps = []
        for idx in frame_indices:
            # idx must be >= 1 (1-based indexing) and within bounds
            # NOTE: After interpolate_frame_indices(), all indices should be >= 1,
            # but we keep the check for safety against edge cases
            if idx >= 1 and idx <= len(optitrack_events):
                timestamps.append(optitrack_events[idx - 1])  # Convert to 0-based indexing
            else:
                timestamps.append(np.nan)
        
        timestamps = np.array(timestamps)
        valid_timestamps = timestamps[np.isfinite(timestamps)]
        
        eye_timestamps[camera_type] = {
            'timestamps': timestamps,
            'valid_count': len(valid_timestamps),
            'total_frames': len(frame_indices)
        }
        
        if verbose:
            print(f"   📍 {camera_type}: {len(valid_timestamps)}/{len(frame_indices)} frames synchronized")
    
    # Step 5: Insert synchronized timestamps as events
    total_inserted = 0
    
    for camera_type, sync_data in eye_timestamps.items():
        timestamps = sync_data['timestamps']
        valid_mask = np.isfinite(timestamps)
        
        if np.sum(valid_mask) == 0:
            continue
            
        # Register event type
        event_type_name = f"{camera_type}_frames"
        event.EventType.insert1({
            'event_type': event_type_name,
            'event_type_description': f'Synchronized {camera_type} camera frames from OptiTrack timing (OCR extraction)'
        }, skip_duplicates=True)
        
        # Get session and scan IDs
        if isinstance(session_key, dict):
            session_id = session_key['session_id']
        else:
            session_id = session_key
            
        if isinstance(scan_key, dict):
            scan_id = scan_key['scan_id']
        else:
            scan_id = scan_key
        
        # Prepare event entries (only for valid timestamps)
        events_to_insert = []
        for i, ts in enumerate(timestamps):
            if np.isfinite(ts):
                events_to_insert.append([session_id, scan_id, event_type_name, ts, ts + 0.001])
        
        # Insert into Event table
        if len(events_to_insert) > 0:
            event.Event.insert(events_to_insert, skip_duplicates=True, allow_direct_insert=True)
            total_inserted += len(events_to_insert)
            
            if verbose:
                print(f"✅ Inserted {len(events_to_insert)} {camera_type} frame events")
        
        # Step 6: Insert interpolation log entries into EventInterpolation table
        correction_log = eye_video_data[camera_type].get('correction_log', [])
        frame_indices = np.array(eye_video_data[camera_type]['frame_indices'])
        
        if correction_log:
            interpolation_entries = []
            for log_entry in correction_log:
                # Find the corresponding event timestamp for this frame
                frame_idx = log_entry['eye_frame_idx']
                if frame_idx < len(timestamps) and np.isfinite(timestamps[frame_idx]):
                    ts = timestamps[frame_idx]
                    
                    # Get the corrected OptiTrack index for this frame
                    opti_idx = int(frame_indices[frame_idx]) if frame_idx < len(frame_indices) else None
                    
                    # Enhance details with OptiTrack index and timestamp info
                    base_details = log_entry['details'] if log_entry['details'] else ''
                    enhanced_details = f"{base_details} | opti_idx={opti_idx}, ts={ts:.4f}s"
                    
                    interpolation_entries.append({
                        'session_id': session_id,
                        'scan_id': scan_id,
                        'event_type': event_type_name,
                        'event_start_time': ts,
                        'frame_idx': frame_idx,
                        'interpolation_type': log_entry['correction_type'],
                        'original_value': log_entry['original_value'],
                        'corrected_value': log_entry['corrected_value'],
                        'diff_before': log_entry['diff_before'],
                        'diff_after': log_entry['diff_after'],
                        'details': enhanced_details[:256]
                    })
            
            if interpolation_entries:
                event.EventInterpolation.insert(interpolation_entries, skip_duplicates=True)
                if verbose:
                    print(f"📝 Logged {len(interpolation_entries)} interpolation entries for {camera_type}")
    
    eye_summary, world_summary = _split_eye_world_summary(eye_video_data)

    # Return summary
    result = {
        'cameras_processed': len(eye_video_data),
        'total_events_inserted': total_inserted,
        'optitrack_events': len(optitrack_events),
        'eye_data': eye_summary,
        'world_data': world_summary,
        'camera_errors': camera_errors,
        'sync_data': eye_timestamps
    }
    
    if verbose:
        print(f"\n🎉 OCR-based eye camera ingestion complete!")
        print(f"   📊 Cameras processed: {result['cameras_processed']}")
        print(f"   📊 Total events inserted: {result['total_events_inserted']}")
        if camera_errors:
            print(f"   Skipped cameras: {len(camera_errors)}")
    
    return result


def ingest_egocams_ocr_linear(session_key, scan_key, eye_cameras=['mini2p1_eye_left', 'mini2p1_eye_right'],
                               model_path=None, verbose=True, write_log=True,
                               residual_threshold=0.01, drop_trailing_invalid=True,
                               prefilter_ocr=True, prefilter_min_step=2.5,
                               prefilter_max_step=6.5, deinterlace_input_fps=25.0):
    """
    Ingest eye camera synchronization using robust linear fit (physics-based approach).
    Also processes worldcam videos detected by 'worldcam' in the filename.
    
    This function uses RANSAC to fit a linear model between eye frame indices and
    aux timestamps, leveraging the physical constraint that both cameras run at
    constant clock rates. OCR errors are automatically detected as outliers.
    
    Key advantages over heuristic correction (ingest_egocams_ocr):
    - No ad-hoc pattern matching (5,5,5,5,4 pattern)
    - Direct timestamp output (no integer rounding)
    - Principled outlier detection via robust regression
    - Sub-millisecond precision for inlier frames
    
    Events are inserted with "_lin" suffix (e.g., "mini2p1_eye_left_frames_lin").
    
    Parameters
    ----------
    session_key : str or dict
        Session identifier
    scan_key : str or dict
        Scan identifier
    eye_cameras : list, optional
        List of eye camera types to process (default: left and right eye)
    model_path : str or Path, optional
        Path to digit recognition model. If None, uses default location.
    verbose : bool
        Print detailed progress information (default: True)
    write_log : bool
        Write correction logs for each video (default: True)
    residual_threshold : float, optional
        Maximum residual (seconds) for RANSAC inlier classification.
        Default 0.005s (~5ms). Points with larger residuals are OCR errors.
    drop_trailing_invalid : bool, optional
        If True, discard trailing eye camera frames whose OCR values are
        out of OptiTrack range (e.g., camera kept recording after OptiTrack stopped).
        This avoids extrapolating timestamps beyond the last valid OptiTrack frame.
    prefilter_ocr : bool, optional
        If True, prefilter valid OCR points using local step consistency before fit.
    prefilter_min_step : float, optional
        Minimum allowed OCR step per eye frame (default: 3.0).
    prefilter_max_step : float, optional
        Maximum allowed OCR step per eye frame (default: 6.5).
    deinterlace_input_fps : float
        Input FPS for deinterlacing interlaced PAL/NTSC sources (default: 25.0).
        Set to 29.97 for NTSC if needed.
        
    Returns
    -------
    dict : Processing results including fit statistics and quality metrics
    """
    if verbose:
        print(f"🎥 Starting LINEAR FIT eye camera ingestion for session {session_key}, scan {scan_key}")
        print(f"   Using robust linear regression (RANSAC) with {residual_threshold*1000:.1f}ms threshold")
    
    # Step 1: Get eye camera video files from database
    try:
        eye_video_key = (model.VideoRecordingNew & {'session_id': session_key, 'scan_id': scan_key} &
                         f'camera IN {tuple(eye_cameras)}').fetch('KEY')
        eye_video_files = (model.VideoRecordingNew * model.VideoRecordingNew.File & eye_video_key).fetch(as_dict=True)
    except Exception as e:
        return {'error': f'Failed to fetch video files from database: {e}'}

    eye_video_files = _prefer_deinterlaced_video_files(eye_video_files)

    if len(eye_video_files) == 0:
        if verbose:
            print("❌ No eye camera videos found in database")
            all_cameras = (model.VideoRecordingNew & {'session_id': session_key, 'scan_id': scan_key}).fetch('camera')
            if len(all_cameras) > 0:
                print("Available cameras:")
                for cam in set(all_cameras):
                    print(f"   - {cam}")
        return {'error': 'No eye camera videos found'}

    # Find worldcam videos by filename
    worldcam_files = []
    try:
        all_video_files = (model.VideoRecordingNew * model.VideoRecordingNew.File &
                          {'session_id': session_key, 'scan_id': scan_key}).fetch(as_dict=True)
        all_video_files = _prefer_deinterlaced_video_files(all_video_files)
        worldcam_files = [
            vf for vf in all_video_files
            if 'worldcam' in str(vf.get('file_path', '')).lower()
        ]
    except Exception as e:
        if verbose:
            print(f"Warning: failed to fetch worldcam videos: {e}")

    video_by_path = {vf['file_path']: vf for vf in eye_video_files}
    for vf in worldcam_files:
        video_by_path.setdefault(vf['file_path'], vf)
    video_files = list(video_by_path.values())
    worldcam_paths = {str(vf['file_path']).lower() for vf in worldcam_files}

    if len(video_files) == 0:
        return {'error': 'No camera videos found'}

    if verbose:
        print(f"📹 Found {len(video_files)} camera videos (eyes={len(eye_video_files)}, worldcam={len(worldcam_files)})")

    # Step 2: Extract RAW frame indices using OCR (no correction)
    eye_video_data = {}
    camera_errors = {}

    for video_info in video_files:
        camera_type = video_info['camera']
        file_path = Path(video_info['file_path'])
        is_worldcam = (
            'worldcam' in str(camera_type).lower()
            or 'worldcam' in str(file_path).lower()
            or str(video_info['file_path']).lower() in worldcam_paths
        )

        deint_result, file_path = _resolve_deinterlaced_video_path(
            video_info,
            input_fps=deinterlace_input_fps,
            verbose=verbose,
        )
        if 'error' in deint_result:
            if is_worldcam:
                camera_errors[file_path.name] = deint_result['error']
                if verbose:
                    print(f"   Warning: skipping worldcam {file_path.name}: {deint_result['error']}")
                continue
            return {'error': f"Failed to deinterlace {camera_type}: {deint_result['error']}"}

        if verbose:
            print(f"\n🔄 Processing {camera_type}: {file_path.name}")

        # Extract OCR but we only need raw values for linear fit
        result = extract_eye_camera_frames_ocr(
            file_path,
            model_path=model_path,
            verbose=verbose,
            write_log=write_log,
            apply_corrections=False,
            camera_type=camera_type
        )

        if 'error' not in result:
            eye_video_data[camera_type] = result
        else:
            if is_worldcam:
                camera_errors[file_path.name] = result['error']
                if verbose:
                    print(f"   Warning: skipping worldcam {file_path.name}: {result['error']}")
                continue
            if verbose:
                print(f"   ❌ Error processing {camera_type}: {result['error']}")
            return {'error': f"Failed to process {camera_type}: {result['error']}"}

    if not eye_video_data:
        return {'error': 'No camera data extracted successfully'}
    
    # Step 3: Load OptiTrack event timestamps
    try:
        optitrack_events = (event.Event & {'session_id': session_key, 'scan_id': scan_key} & 
                           'event_type="optitrack_frames"').fetch(
            'event_start_time', order_by='event_start_time'
        ).astype(float)
        
        if len(optitrack_events) == 0:
            if verbose:
                print("❌ No OptiTrack events found!")
                available_events = (event.Event & {'session_id': session_key, 'scan_id': scan_key}).fetch('event_type')
                if len(available_events) > 0:
                    print("Available event types:")
                    for event_type in set(available_events):
                        count = len((event.Event & {'session_id': session_key, 'scan_id': scan_key} & 
                                   f'event_type="{event_type}"'))
                        print(f"   - {event_type}: {count} events")
            return {'error': 'No OptiTrack events found'}
        
        if verbose:
            print(f"\n✅ Loaded {len(optitrack_events)} OptiTrack events")
            
    except Exception as e:
        return {'error': f'Failed to load OptiTrack events: {e}'}
    
    # Step 4: Apply robust linear fit to derive timestamps
    eye_timestamps = {}
    fit_results = {}
    
    for camera_type, data in eye_video_data.items():
        if verbose:
            print(f"\n📈 Fitting linear model for {camera_type}...")
        
        # Get RAW OCR values (before correction)
        raw_ocr = _ocr_to_int_list(data['frame_indices_raw'])
        valid_ocr_mask = (raw_ocr >= 1) & (raw_ocr <= len(optitrack_events))
        last_valid_idx = np.where(valid_ocr_mask)[0][-1] if np.any(valid_ocr_mask) else None
        
        # Apply robust linear fit (returns timestamps, log entries, and fit info)
        timestamps, log_entries, fit_info = fit_ocr_frames_robust_linear(
            raw_ocr_frames=raw_ocr,
            optitrack_timestamps=optitrack_events,
            residual_threshold=residual_threshold,
            verbose=verbose,
            prefilter=prefilter_ocr,
            prefilter_min_step=prefilter_min_step,
            prefilter_max_step=prefilter_max_step
        )

        dropped_frame_indices = []
        last_valid_timestamp = None
        if drop_trailing_invalid and last_valid_idx is not None and last_valid_idx < (len(raw_ocr) - 1):
            dropped_frame_indices = list(range(last_valid_idx + 1, len(raw_ocr)))
            timestamps = timestamps[:last_valid_idx + 1]
            if len(timestamps) > 0 and np.isfinite(timestamps[-1]):
                last_valid_timestamp = float(timestamps[-1])
            if verbose:
                n_dropped = len(raw_ocr) - (last_valid_idx + 1)
                print(f"   ⚠️ Dropping {n_dropped} trailing frames with invalid OCR (beyond OptiTrack range)")
            for frame_idx in dropped_frame_indices:
                ocr_val = int(raw_ocr[frame_idx]) if frame_idx < len(raw_ocr) else 0
                details = f"frame={frame_idx}, ocr={ocr_val}, reason=trailing_invalid_ocr_no_optitrack"
                log_entries.append({
                    'eye_frame_idx': int(frame_idx),
                    'correction_type': 'TRAILING_INVALID_OCR',
                    'original_value': ocr_val,
                    'corrected_value': 0,
                    'diff_before': None,
                    'diff_after': None,
                    'details': details[:256]
                })
        
        if 'error' in fit_info:
            if verbose:
                print(f"   ⚠️ Fit warning: {fit_info['error']}")
        
        eye_timestamps[camera_type] = {
            'timestamps': timestamps,
            'valid_count': len(timestamps),
            'total_frames': len(raw_ocr),
            'used_frames': len(timestamps),
            'log_entries': log_entries,  # Store for later insertion
            'last_valid_timestamp': last_valid_timestamp
        }
        fit_results[camera_type] = fit_info
    
    # Step 5: Insert synchronized timestamps as events (with _lin suffix)
    total_inserted = 0
    
    # Get session and scan IDs
    if isinstance(session_key, dict):
        session_id = session_key['session_id']
    else:
        session_id = session_key
        
    if isinstance(scan_key, dict):
        scan_id = scan_key['scan_id']
    else:
        scan_id = scan_key
    
    for camera_type, sync_data in eye_timestamps.items():
        timestamps = sync_data['timestamps']
        n_frames_total = sync_data['total_frames']
        n_frames_used = sync_data.get('used_frames', len(timestamps))
        
        if len(timestamps) == 0:
            continue
        
        # CRITICAL: Verify we have a timestamp for every frame
        if len(timestamps) != n_frames_used:
            raise ValueError(f"{camera_type}: timestamp count ({len(timestamps)}) != used frame count ({n_frames_used})")
        
        if not np.all(np.isfinite(timestamps)):
            n_nan = np.sum(~np.isfinite(timestamps))
            raise ValueError(f"{camera_type}: {n_nan} frames have non-finite timestamps")
            
        # Register event type with _lin suffix
        event_type_name = f"{camera_type}_frames_lin"
        event.EventType.insert1({
            'event_type': event_type_name,
            'event_type_description': f'Synchronized {camera_type} camera frames using robust linear fit (RANSAC)'
        }, skip_duplicates=True)
        
        # Prepare event entries - ALL frames get inserted (timestamps are guaranteed finite)
        events_to_insert = []
        for i, ts in enumerate(timestamps):
            events_to_insert.append([session_id, scan_id, event_type_name, float(ts), float(ts) + 0.001])
        
        # Insert into Event table
        event.Event.insert(events_to_insert, skip_duplicates=True, allow_direct_insert=True)
        total_inserted += len(events_to_insert)
        
        if verbose:
            suffix = f" (trimmed from {n_frames_total})" if n_frames_used != n_frames_total else ""
            print(f"✅ Inserted {len(events_to_insert)} {camera_type} frame events (linear fit) - {n_frames_used} frames{suffix}")
        
        # Step 6: Insert interpolation/correction log entries into EventInterpolation table
        log_entries = sync_data.get('log_entries', [])
        last_valid_timestamp = sync_data.get('last_valid_timestamp', None)
        if log_entries:
            # Register new InterpolationType values if needed
            new_types = [
                ('RANSAC_OUTLIER', 'OCR error detected by RANSAC robust linear fit'),
                ('RANSAC_FRAME_DROP_HIGH', 'High-confidence frame drop detected by linear fit (offset shift >15ms)'),
                ('RANSAC_FRAME_DROP_LOW', 'Potential frame drop detected by linear fit (low confidence)'),
                ('TRAILING_INVALID_OCR', 'Trailing eye camera frames with OCR beyond OptiTrack range (dropped)')
            ]
            for type_name, type_desc in new_types:
                event.InterpolationType.insert1({
                    'interpolation_type': type_name,
                    'interpolation_description': type_desc
                }, skip_duplicates=True)
            
            # Prepare interpolation entries
            interpolation_entries = []
            for log_entry in log_entries:
                frame_idx = log_entry['eye_frame_idx']
                ts = None
                if frame_idx < len(timestamps) and np.isfinite(timestamps[frame_idx]):
                    ts = float(timestamps[frame_idx])
                elif log_entry.get('correction_type') == 'TRAILING_INVALID_OCR' and last_valid_timestamp is not None:
                    ts = float(last_valid_timestamp)
                if ts is not None:
                    
                    interpolation_entries.append({
                        'session_id': session_id,
                        'scan_id': scan_id,
                        'event_type': event_type_name,
                        'event_start_time': ts,
                        'frame_idx': frame_idx,
                        'interpolation_type': log_entry['correction_type'],
                        'original_value': log_entry['original_value'] or 0,
                        'corrected_value': log_entry['corrected_value'] or 0,
                        'diff_before': log_entry['diff_before'],
                        'diff_after': log_entry['diff_after'],
                        'details': (log_entry['details'] or '')[:256]
                    })
            
            if interpolation_entries:
                event.EventInterpolation.insert(interpolation_entries, skip_duplicates=True)
                if verbose:
                    print(f"📝 Logged {len(interpolation_entries)} interpolation entries for {camera_type} (outliers/drops)")
    
    eye_summary, world_summary = _split_eye_world_summary(eye_video_data)

    # Return summary with fit statistics
    result = {
        'method': 'robust_linear_fit',
        'cameras_processed': len(eye_video_data),
        'total_events_inserted': total_inserted,
        'optitrack_events': len(optitrack_events),
        'eye_data': eye_summary,
        'world_data': world_summary,
        'camera_errors': camera_errors,
        'sync_data': eye_timestamps,
        'fit_results': fit_results
    }
    
    if verbose:
        print(f"\n🎉 Linear fit eye camera ingestion complete!")
        print(f"   📊 Cameras processed: {result['cameras_processed']}")
        print(f"   📊 Total events inserted: {result['total_events_inserted']}")
        if camera_errors:
            print(f"   Skipped cameras: {len(camera_errors)}")
        print(f"\n📈 Fit Summary:")
        for cam, fit in fit_results.items():
            if 'error' not in fit:
                print(f"   {cam}:")
                print(f"      Eye camera rate: {fit.get('expected_eye_hz', 0):.2f} Hz")
                print(f"      Inliers: {fit.get('n_inliers', 0)}/{fit.get('n_valid', 0)} ({100*fit.get('inlier_ratio', 0):.1f}%)")
                print(f"      OCR errors (outliers): {fit.get('n_outliers', 0)}")
                print(f"      Residual std: {fit.get('residual_std_ms', 0):.3f} ms")
    
    return result


def extract_eye_camera_frames(video_path, crop_positions=None, threshold=20, verbose=True):
    """
    Extract binary-encoded OptiTrack frame indices from eye camera video
    Uses cv2.VideoCapture for robust video loading and optimized binary detection
    
    Parameters:
    -----------
    video_path : str or Path
        Path to eye camera video file
    crop_positions : list of tuples
        List of (x, y, width, height) for binary dot positions
        If None, uses default positions based on camera type
    threshold : int
        Threshold for binary dot detection (default: 20)
    verbose : bool
        Print progress information (default: True)
        
    Returns:
    --------
    dict : Processing results including frame_indices and metadata
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return {'error': f'Video file not found: {video_path}'}
    
    if verbose:
        print(f"🎬 Loading video: {video_path.name}")
    
    try:
        # Open video with cv2.VideoCapture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {'error': f'Cannot open video: {video_path}'}
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if verbose:
            print(f"   📊 Video info: {frame_count} frames, {width}x{height}, {fps:.1f} FPS")
        
    except Exception as e:
        return {'error': f'Cannot read video properties: {e}'}
    
    # Default crop positions for binary dots (camera-specific)
    if crop_positions is None:
        # Determine if this is a high-res (deinterlaced) video
        is_high_res = height > 300  # Assume >300px height is high-res
        
        if verbose:
            print(f"   📍 Using {'high-res' if is_high_res else 'standard'} parameters")
        
        # Determine camera type from video path
        camera_type = "left"  # Default
        if 'left' in str(video_path).lower():
            camera_type = "left"
        elif 'right' in str(video_path).lower():
            camera_type = "right"
        
        # Parameters based on camera type and resolution
        if camera_type == "left":
            if is_high_res:
                # High-res left eye parameters (deinterlaced)
                x_start = 126 * 2  # Starting x coordinate  
                y_start = 275 * 2  # Stable y coordinate
                width_crop, height_crop = 14, 14  # Crop size
                step = 24  # Step between crops
                threshold = 125 # threshold * 3  # Adjust threshold for larger crops
            else:
                # Standard resolution left eye parameters
                x_start = 126
                y_start = 275  
                width_crop, height_crop = 7, 7
                step = 12
        else:  # right eye
            if is_high_res:
                # High-res right eye parameters (deinterlaced)
                x_start = 128 * 2
                y_start = 275 * 2
                width_crop, height_crop = 14, 14
                step = 24
                threshold = 125 #   threshold * 3  # Adjust threshold for larger crops
            else:
                # Standard resolution right eye parameters  
                x_start = 128
                y_start = 275
                width_crop, height_crop = 7, 7
                step = 12
        
        crop_positions = [(x_start + i * step, y_start, width_crop, height_crop) for i in range(19)]
        
        if verbose:
            print(f"   📹 Camera: {camera_type}, Resolution: {'high' if is_high_res else 'standard'}")
            print(f"   📐 Crop area: {width_crop}x{height_crop} starting at ({x_start}, {y_start}) with {step}px steps")
            print(f"   📐 Pixel threshold: {threshold}")
    
    if verbose:
        print(f"   📍 Using {len(crop_positions)} binary dot positions")
    
    # Process all frames
    binary_code = []
    frame_idx = 0
    
    if verbose:
        print(f"   🔄 Processing {frame_count} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_binary = []
        
        # Extract binary pattern from each crop position
        for x, y, w, h in crop_positions:
            # Crop the region
            cropped = frame[y:y+h, x:x+w]
            
            # Convert to grayscale
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            
            # Count bright pixels and determine if dot is present
            dot_present = np.count_nonzero(binary) > threshold
            
            # Add 1 (dot present) or 0 (dot absent) to binary code
            frame_binary.append(1 if dot_present else 0)
        
        binary_code.append(frame_binary)
        frame_idx += 1
        
        # Progress indicator
        if verbose and frame_idx % 1000 == 0:
            print(f"      Processed {frame_idx}/{frame_count} frames...")
    
    cap.release()
    
    # Convert binary codes to decimal frame indices
    optitrack_frame_indices = []
    for frame_code in binary_code:
        # Convert binary list to string, then to decimal
        binary_str = ''.join(str(bit) for bit in frame_code)
        decimal_value = int(binary_str, 2)
        optitrack_frame_indices.append(decimal_value)
    
    # Apply frame index correction for temporal consistency
    optitrack_frame_indices = _fix_frame_indices_new(optitrack_frame_indices, verbose=verbose)
    # print('deactivated _fix_frame_indices')
    
    # Calculate statistics
    valid_frames = [f for f in optitrack_frame_indices if f > 0]
    
    if verbose:
        print(f"✅ Processing complete:")
        print(f"   📊 Total frames: {frame_idx}")
        print(f"   📊 Valid frames: {len(valid_frames)} ({len(valid_frames)/frame_idx*100:.1f}%)")
        
        if len(valid_frames) > 0:
            print(f"   📊 Frame index range: {min(valid_frames)} - {max(valid_frames)}")
    
    return {
        'frame_indices': optitrack_frame_indices,
        'total_frames': frame_idx,
        'file_path': str(video_path),
        'crop_positions': crop_positions,
        'binary_codes': binary_code  # Include raw binary codes for debugging
    }


def eye_timestamps_optilocked(decoded_codes, tick_times):
    """
    Simplified eye camera timestamp assignment using direct OptiTrack locking.
    
    Each eye camera frame is assigned to exactly the corresponding OptiTrack timestamp.
    For duplicate OptiTrack indices (consecutive frames with same index), the duplicates
    are spaced at the eye camera's natural frame rate (~20ms for 50Hz).
    
    This is a much simpler approach than sub-millisecond interpolation.

    Parameters
    ----------
    decoded_codes : array-like
        Frame indices from binary encoding (OptiTrack frame numbers)
    tick_times : array-like
        OptiTrack event timestamps in seconds

    Returns
    -------
    np.ndarray
        One timestamp per eye camera frame, monotonically increasing
    """
    K = np.asarray(decoded_codes, dtype=int)
    T = np.asarray(tick_times, dtype=float)
    N = len(K)
    
    if N == 0:
        return np.array([], dtype=float)
    
    # Bounds checking
    if K.min() < 0:
        raise ValueError("decoded_codes contains negative indices.")
    if K.max() >= len(T):
        raise ValueError("tick_times does not cover the largest decoded code.")
    
    # Estimate eye camera frame rate from the data itself
    # Count frame transitions to estimate natural eye camera frequency
    if N > 1:
        # Count unique consecutive OptiTrack index changes
        index_changes = np.diff(K)
        nonzero_changes = index_changes[index_changes != 0]
        
        if len(nonzero_changes) > 10:  # Need reasonable sample size
            # Estimate how many eye frames per OptiTrack transition
            frames_per_opti_transition = len(K) / len(nonzero_changes)
            opti_rate = 250.0  # Hz - OptiTrack frequency
            estimated_eye_fps = opti_rate / frames_per_opti_transition
            
            # Clamp to reasonable range (30-60 Hz for eye cameras)
            estimated_eye_fps = np.clip(estimated_eye_fps, 30.0, 60.0)
        else:
            estimated_eye_fps = 50.0  # Default fallback
            
        eye_frame_interval = 1.0 / estimated_eye_fps
    else:
        eye_frame_interval = 0.020  # 50Hz default (20ms)
    
    # Create output timestamps
    timestamps = np.empty(N, dtype=float)
    
    # Track duplicate counting for each OptiTrack index
    duplicate_counter = {}
    
    for i in range(N):
        opti_idx = K[i]
        base_time = T[opti_idx]
        
        # Count how many times we've seen this OptiTrack index
        if opti_idx not in duplicate_counter:
            duplicate_counter[opti_idx] = 0
        else:
            duplicate_counter[opti_idx] += 1
        
        # For first occurrence: use exact OptiTrack timestamp
        # For duplicates: add eye camera frame intervals
        duplicate_offset = duplicate_counter[opti_idx] * eye_frame_interval
        timestamps[i] = base_time + duplicate_offset
    
    # Ensure monotonic increasing (safety check)
    for i in range(1, N):
        if timestamps[i] <= timestamps[i-1]:
            timestamps[i] = timestamps[i-1] + eye_frame_interval
    
    return timestamps


def eye_timestamps_subms(decoded_codes, tick_times, phase_delta=0.0):
    """
    Generate sub-millisecond precise timestamps by placing each video frame strictly
    BETWEEN its 250 Hz tick edge T[k] and the next edge T[k+1]. Frames that share
    the same tick code are evenly spaced within that interval. A constant phase
    (phase_delta, in seconds) is applied and safely clipped to the interval.

    Parameters
    ----------
    decoded_codes : array-like
        K_i per frame (int) - frame indices from binary encoding (tick index shown in the frame)
    tick_times : array-like
        T_k at 250 Hz (seconds) - precise tick edge times extracted from the 20 kHz recorder
        (should cover up to max(decoded_codes)+1)
    phase_delta : float
        Constant phase (seconds) to shift from tick edge toward mid-exposure (default: 0.0).
        Applied per-interval and clipped so frames never cross into the next interval.

    Returns
    -------
    np.ndarray
        One timestamp (seconds) per frame, strictly increasing and sub-ms precise.
    """
    K = np.asarray(decoded_codes, dtype=int)
    T = np.asarray(tick_times, dtype=float)
    N = len(K)
    if N == 0:
        return np.array([], dtype=float)

    # Basic guard: we need T[k] for all used k, and preferably T[k+1].
    # If T[k+1] is missing for some k (near the end), we will synthesize it
    # from the median tick period (robust to small jitter).
    if K.min() < 0:
        raise ValueError("decoded_codes contains negative indices.")
    if K.max() >= len(T):  # we at least require T[max_k]
        raise ValueError("tick_times does not cover the largest decoded code.")

    dT = np.diff(T)
    median_period = float(np.median(dT)) if dT.size else 1/250.0  # fallback

    # Find runs of identical codes (consecutive frames sharing the same tick index)
    run_starts = np.flatnonzero(np.r_[True, K[1:] != K[:-1]])
    run_ends   = np.r_[run_starts[1:] - 1, N - 1]

    t = np.empty(N, dtype=float)
    # Small margin to keep positions strictly inside (0,1) even after phase shift
    eps_pos = 1e-6

    for a, b in zip(run_starts, run_ends):
        k = K[a]
        n = b - a + 1
        T0 = T[k]
        # If we don't have T[k+1], synthesize it from median period
        if k + 1 < len(T):
            T1 = T[k + 1]
        else:
            T1 = T0 + median_period

        width = T1 - T0
        if width <= 0:
            # Pathological case (shouldn't happen with real tick data); fall back to median
            width = median_period
            T1 = T0 + width

        # Evenly space n frames strictly inside (0,1) using centers if n>1; use 0.5 if n==1
        if n == 1:
            base_pos = np.array([0.5], dtype=float)  # middle of the interval
        else:
            base_pos = (np.arange(1, n + 1, dtype=float)) / (n + 1)  # (0,1) strictly increasing

        # Convert phase_delta (seconds) to a fractional shift within this interval
        shift = phase_delta / width
        pos = base_pos + shift

        # Clip to stay strictly inside the interval (no wrap across ticks)
        pos = np.clip(pos, eps_pos, 1.0 - eps_pos)

        # Map back to absolute time
        t[a:b+1] = T0 + pos * width

    # Enforce strict monotonicity to handle occasional backwards tick jumps
    # This is needed when the binary decoding occasionally produces decreasing indices
    # due to camera timing jitter or data corruption
    for i in range(1, N):
        if t[i] <= t[i-1]:
            # If timestamp would go backwards, place it just after the previous one
            # Use a minimal increment based on the typical video frame rate
            min_increment = 1.0 / 60.0  # ~16.7ms (60 FPS upper bound)
            t[i] = t[i-1] + min_increment

    return t


def synchronize_eye_camera_timestamps(eye_video_data, optitrack_events, verbose=True):
    """
    Generate sub-millisecond precise timestamps for eye camera frames using anchor-based interpolation
    with robust bounds checking to handle out-of-bounds frame indices
    
    Parameters:
    -----------
    eye_video_data : dict
        Dictionary containing eye camera frame indices for each camera
    optitrack_events : np.array
        Array of OptiTrack event timestamps
    verbose : bool
        Print progress information (default: True)
        
    Returns:
    --------
    dict : Synchronized timestamp data for each camera
    """
    
    if len(optitrack_events) == 0:
        return {'error': 'No OptiTrack events available for synchronization'}
    
    synchronized_data = {}
    max_valid_index = len(optitrack_events) - 1
    
    for camera_type, video_data in eye_video_data.items():
        if verbose:
            print(f"\n🕐 Synchronizing {camera_type} with sub-ms precision...")
        
        frame_indices = np.array(video_data['frame_indices'])
        total_frames = len(frame_indices)
        
        if verbose:
            print(f"   📊 Total video frames: {total_frames}")
        
        if total_frames == 0:
            if verbose:
                print(f"   ❌ No frames found for {camera_type}")
            synchronized_data[camera_type] = {
                'timestamps': np.array([]),
                'frame_numbers': np.array([]),
                'opti_indices': np.array([]),
                'error': 'No frames found'
            }
            continue
        
        # Filter out-of-bounds indices BEFORE processing
        valid_mask = (frame_indices >= 0) & (frame_indices <= max_valid_index)
        valid_frame_indices = frame_indices[valid_mask]
        valid_frame_positions = np.where(valid_mask)[0]
        
        out_of_bounds_count = np.sum(~valid_mask)
        
        if verbose:
            print(f"   📊 Out-of-bounds indices: {out_of_bounds_count}")
            print(f"   📊 Valid frames for sync: {len(valid_frame_indices)}")
            if len(valid_frame_indices) > 0:
                print(f"   📊 Valid index range: {valid_frame_indices.min()} to {valid_frame_indices.max()}")
        
        if len(valid_frame_indices) == 0:
            if verbose:
                print(f"   ❌ No valid frames for synchronization")
            synchronized_data[camera_type] = {
                'timestamps': np.array([]),
                'frame_numbers': np.array([]),
                'opti_indices': np.array([]),
                'error': 'No valid frames after bounds checking'
            }
            continue
        
        # Use the simplified OptiLocked timestamp algorithm with ONLY valid indices
        try:
            valid_timestamps = eye_timestamps_optilocked(valid_frame_indices, optitrack_events)
            
            # These should match 1:1 now
            final_timestamps = valid_timestamps
            final_frame_numbers = valid_frame_positions
            final_indices = valid_frame_indices
            
            # Filter out NaN timestamps
            nan_mask = ~np.isnan(final_timestamps)
            final_timestamps = final_timestamps[nan_mask]
            final_frame_numbers = final_frame_numbers[nan_mask]
            final_indices = final_indices[nan_mask]
            
            if verbose:
                print(f"   📊 Final synchronized frames: {len(final_timestamps)}")
            
            if len(final_timestamps) == 0:
                if verbose:
                    print(f"   ❌ No valid timestamps generated for {camera_type}")
                synchronized_data[camera_type] = {
                    'timestamps': np.array([]),
                    'frame_numbers': np.array([]),
                    'opti_indices': np.array([]),
                    'error': 'No valid timestamps generated'
                }
                continue
            
            if verbose:
                print(f"   📊 Timestamp range: {final_timestamps.min():.6f} - {final_timestamps.max():.6f} sec")
                print(f"   📊 Duration: {final_timestamps.max() - final_timestamps.min():.1f} sec")
            
            # Calculate frame rate
            if len(final_timestamps) > 1:
                intervals = np.diff(final_timestamps)
                median_interval = np.median(intervals[intervals > 0])
                estimated_fps = 1 / median_interval if median_interval > 0 else 0
                if verbose:
                    print(f"   📊 Estimated camera FPS: {estimated_fps:.1f} Hz")
                    print(f"   📊 Median interval: {median_interval*1000:.3f} ms")
            else:
                estimated_fps = 0
            
            # Store synchronized data
            synchronized_data[camera_type] = {
                'timestamps': final_timestamps,
                'frame_numbers': final_frame_numbers,
                'opti_indices': final_indices - 1,  # Convert to 0-based for consistency
                'total_frames': total_frames,
                'valid_frames': len(final_timestamps),
                'estimated_fps': estimated_fps,
                'out_of_bounds_count': out_of_bounds_count
            }
            
            if verbose:
                print(f"   ✅ Synchronized {len(final_timestamps)} frames with sub-ms precision")
                
        except Exception as e:
            if verbose:
                print(f"   ❌ Error during synchronization for {camera_type}: {e}")
            synchronized_data[camera_type] = {
                'timestamps': np.array([]),
                'frame_numbers': np.array([]),
                'opti_indices': np.array([]),
                'error': f'Synchronization failed: {e}'
            }
    
    return synchronized_data


def prepare_eye_camera_timestamps(eye_timestamps, session_key, scan_key, camera_type):
    """
    Prepare synchronized eye camera timestamps for insertion into event.Event table
    Creates instantaneous events with 1ms duration for each frame
    
    Parameters:
    -----------
    eye_timestamps : np.array
        Array of synchronized eye camera timestamps
    session_key : str
        Session identifier
    scan_key : str  
        Scan identifier
    camera_type : str
        Camera type (e.g., 'mini2p1_eye_left', 'mini2p1_eye_right')
        
    Returns:
    --------
    list : List of tuples ready for insertion into event.Event
    """
    
    if len(eye_timestamps) == 0:
        return []
    
    # Eye camera frames are instantaneous - create 1ms duration events
    ts_start = eye_timestamps
    ts_end = eye_timestamps + 0.001  # 1ms duration
    
    event_type = f"{camera_type}_frames"
    
    # Prepare for insertion: [session_key, scan_key, event_type, start_time, end_time]
    to_insert = [[session_key, scan_key, event_type, start, end] 
                 for start, end in zip(ts_start, ts_end)]
    
    return to_insert


def ingest_egocams(session_key, scan_key, eye_cameras=['mini2p1_eye_left', 'mini2p1_eye_right'], 
                   verbose=False, deinterlace_input_fps=25.0):
    """
    Ingest eye camera synchronization data as events in the ADAMACS pipeline
    
    This function:
    1. Finds eye camera videos from the database
    2. Extracts binary frame indices from videos  
    3. Loads OptiTrack event timestamps
    4. Synchronizes eye camera frames with OptiTrack timing
    5. Inserts synchronized timestamps as events
    
    Parameters:
    -----------
    session_key : str
        Session identifier
    scan_key : str
        Scan identifier  
    eye_cameras : list, optional
        List of eye camera types to process (default: left and right eye)
    verbose : bool, optional
        Print detailed progress information (default: True)
    deinterlace_input_fps : float
        Input FPS for deinterlacing interlaced PAL/NTSC sources (default: 25.0).
        Set to 29.97 for NTSC if needed.
        
    Returns:
    --------
    dict : Processing results and statistics
    """
    
    if verbose:
        print(f"🎥 Starting eye camera ingestion for session {session_key}, scan {scan_key}")
    
    # Step 1: Get eye camera video files from database
    try:
        video_file_key = (model.VideoRecordingNew & {'session_id': session_key, 'scan_id': scan_key} & 
                         f'camera IN {tuple(eye_cameras)}').fetch('KEY')
        video_files = (model.VideoRecordingNew * model.VideoRecordingNew.File & video_file_key).fetch(as_dict=True)
    except Exception as e:
        return {'error': f'Failed to fetch video files from database: {e}'}

    video_files = _prefer_deinterlaced_video_files(video_files)
    
    if len(video_files) == 0:
        if verbose:
            print("❌ No eye camera videos found in database")
            # Show available cameras for debugging
            all_cameras = (model.VideoRecordingNew & {'session_id': session_key, 'scan_id': scan_key}).fetch('camera')
            if len(all_cameras) > 0:
                print("Available cameras:")
                for cam in set(all_cameras):
                    print(f"   - {cam}")
        return {'error': 'No eye camera videos found'}
    
    if verbose:
        print(f"📹 Found {len(video_files)} eye camera videos")
    
    # Step 2: Extract binary frame indices from each video
    eye_video_data = {}
    
    for video_info in video_files:
        camera_type = video_info['camera']
        file_path = Path(video_info['file_path'])
        deint_result, file_path = _resolve_deinterlaced_video_path(
            video_info,
            input_fps=deinterlace_input_fps,
            verbose=verbose,
        )
        if 'error' in deint_result:
            return {'error': f"Failed to deinterlace {camera_type}: {deint_result['error']}"}
        
        if verbose:
            print(f"\n🔄 Processing {camera_type}: {file_path}")
        
        # Extract binary frame indices - let function automatically detect crop positions 
        # based on video resolution and camera type from file path
        result = extract_eye_camera_frames(file_path, crop_positions=None, verbose=verbose)
        
        if 'error' not in result:
            eye_video_data[camera_type] = result
        else:
            if verbose:
                print(f"   ❌ Error processing {camera_type}: {result['error']}")
            return {'error': f"Failed to process {camera_type}: {result['error']}"}
    
    if not eye_video_data:
        return {'error': 'No eye camera data extracted successfully'}
    
    # Step 3: Load OptiTrack event timestamps
    try:
        optitrack_events = (event.Event & {'session_id': session_key, 'scan_id': scan_key} & 
                           'event_type="optitrack_frames"').fetch(
            'event_start_time', order_by='event_start_time'
        ).astype(float)
        
        if len(optitrack_events) == 0:
            if verbose:
                print("❌ No OptiTrack events found!")
                # Show available event types for debugging
                available_events = (event.Event & {'session_id': session_key, 'scan_id': scan_key}).fetch('event_type')
                if len(available_events) > 0:
                    print("Available event types:")
                    for event_type in set(available_events):
                        count = len((event.Event & {'session_id': session_key, 'scan_id': scan_key} & 
                                   f'event_type="{event_type}"'))
                        print(f"   - {event_type}: {count} events")
            return {'error': 'No OptiTrack events found'}
        
        if verbose:
            print(f"✅ Loaded {len(optitrack_events)} OptiTrack events")
            
    except Exception as e:
        return {'error': f'Failed to load OptiTrack events: {e}'}
    
    # Step 4: Synchronize eye camera frames with OptiTrack timing
    eye_timestamps = synchronize_eye_camera_timestamps(eye_video_data, optitrack_events, verbose=verbose)
    
    if 'error' in eye_timestamps:
        return eye_timestamps
    
    # Step 5: Insert synchronized timestamps as events
    total_inserted = 0
    
    for camera_type, sync_data in eye_timestamps.items():
        if len(sync_data['timestamps']) == 0:
            continue
            
        # Register event type
        event_type = f"{camera_type}_frames"
        event.EventType.insert1({
            'event_type': event_type,
            'event_type_description': f'Synchronized {camera_type} camera frames from OptiTrack timing'
        }, skip_duplicates=True)
        
        # Prepare timestamps for insertion
        to_insert = prepare_eye_camera_timestamps(
            sync_data['timestamps'], session_key, scan_key, camera_type
        )
        
        # Insert into Event table
        if len(to_insert) > 0:
            event.Event.insert(to_insert, skip_duplicates=True, allow_direct_insert=True)
            total_inserted += len(to_insert)
            
            if verbose:
                print(f"✅ Inserted {len(to_insert)} {camera_type} frame events")
    
    # Return summary
    result = {
        'cameras_processed': len(eye_video_data),
        'total_events_inserted': total_inserted,
        'optitrack_events': len(optitrack_events),
        'sync_data': eye_timestamps
    }
    
    if verbose:
        print(f"\n🎉 Eye camera ingestion complete!")
        print(f"   📊 Cameras processed: {result['cameras_processed']}")
        print(f"   📊 Total events inserted: {result['total_events_inserted']}")
    
    return result


def ingest_basler_cams(session_key, scan_key, camera_type='mini2p1_top', expected_fps=None, verbose=False):
    """
    Ingest Basler camera synchronization data as events in the ADAMACS pipeline.
    Supports mini2p1_top and mini2p1_bottom cameras.

    This function:
    1. Extracts OptiTrack event timestamps from database
    2. Loads Basler camera video file and CSV metadata
    3. Validates frame count consistency between video and CSV
    4. Synchronizes Basler camera frames with OptiTrack timing by subsampling
    5. Inserts synchronized timestamps as events in the database

    Parameters:
    -----------
    session_key : str or dict
        Session identifier or session key dictionary
    scan_key : str or dict
        Scan identifier or scan key dictionary
    camera_type : str
        Basler camera identifier (e.g., 'mini2p1_top', 'mini2p1_bottom')
    expected_fps : float, optional
        Override expected camera FPS. If None, uses CSV-derived FPS when available,
        otherwise uses video FPS.
    verbose : bool, optional
        Print detailed progress information (default: False)

    Returns:
    --------
    dict : Processing results and statistics including:
        - success: bool indicating if operation completed successfully
        - frames_inserted: number of camera events inserted
        - optitrack_frames: number of OptiTrack frames available
        - sync_quality: timing accuracy percentage
        - frame_validation: status of video/CSV frame count comparison
    """

    if verbose:
        print(f"🎥 Starting Basler camera ingestion for {camera_type} (session {session_key}, scan {scan_key})")
    
    try:
        # Step 1: Extract OptiTrack frames from database
        if verbose:
            print("📡 Extracting OptiTrack frames from database...")
            
        optitrack_events = (event.Event & session_key & scan_key & 
                           'event_type="optitrack_frames"').fetch(
            'event_start_time', order_by='event_start_time'
        ).astype(float)
        
        if len(optitrack_events) == 0:
            return {'success': False, 'error': 'No OptiTrack events found in database'}
            
        if verbose:
            print(f"✅ Found {len(optitrack_events)} OptiTrack frames")
            
        # Calculate OptiTrack frequency
        opti_intervals = np.diff(optitrack_events)
        opti_fps = 1.0 / np.mean(opti_intervals)
        
        if verbose:
            print(f"   OptiTrack frequency: {opti_fps:.1f} Hz")
            
    except Exception as e:
        return {'success': False, 'error': f'Failed to load OptiTrack events: {e}'}
    
    try:
        # Step 2: Get Basler camera video file and CSV metadata
        if verbose:
            print(f"🎥 Loading {camera_type} data...")

        basler_files = (model.VideoRecordingNew * model.VideoRecordingNew.File &
                        session_key & scan_key &
                        f'camera="{camera_type}"').fetch(as_dict=True)

        if len(basler_files) == 0:
            return {'success': False, 'error': f'No {camera_type} video found in database'}

        if len(basler_files) > 1 and verbose:
            print(f"⚠️ Multiple {camera_type} videos found, using first: {basler_files[0].get('file_path')}")

        video_path = Path(basler_files[0]['file_path'])

        if verbose:
            print(f"📂 Video file: {video_path.name}")

        csv_path = find_basler_cam_csv_timestamp_file(video_path.parent, camera_type, verbose=verbose)
        if csv_path is None:
            return {'success': False, 'error': f'No CSV file found for {camera_type}'}

        csv_result = extract_topcam_csv_timestamps(csv_path, verbose=verbose)
        if 'error' in csv_result:
            return {'success': False, 'error': f'Failed to parse CSV for {camera_type}: {csv_result["error"]}'}

        csv_frame_count = csv_result['n_frames']
        csv_fps = csv_result['frame_rate_hz']
        csv_path = Path(csv_result.get('csv_path', csv_path))

        if verbose:
            print(f"📊 CSV file: {csv_path.name} ({csv_frame_count:,} entries)")
            
    except Exception as e:
        return {'success': False, 'error': f'Failed to load top camera files: {e}'}
    
    try:
        # Step 3: Extract video properties and validate frame counts
        if not video_path.exists():
            return {'success': False, 'error': f'Video file not found: {video_path}'}
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {'success': False, 'error': 'Cannot open video file'}
            
        fps_from_video = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Validate frame count consistency
        frame_difference = abs(frame_count - csv_frame_count)
        
        if verbose:
            print(f"📹 Video: {frame_count:,} frames at {fps_from_video:.1f} FPS")
            print(f"📊 CSV: {csv_frame_count:,} entries")
            print(f"🔍 Frame difference: {frame_difference:,}")
            
        # Determine validation status
        if frame_difference == 0:
            frame_validation = "perfect_match"
        elif frame_difference == 1:
            frame_validation = "nearly_perfect" 
        elif frame_difference < 10:
            frame_validation = "minor_discrepancy"
        elif frame_difference < 100:
            frame_validation = "moderate_discrepancy"
        else:
            frame_validation = "major_discrepancy"
            if verbose:
                print(f"⚠️ Large frame count difference may indicate data quality issues")
        
        expected_fps_source = 'video'
        if expected_fps is None:
            if np.isfinite(csv_fps) and csv_fps > 0:
                expected_fps = csv_fps
                expected_fps_source = 'csv'
            else:
                expected_fps = fps_from_video
        else:
            expected_fps_source = 'override'

        if expected_fps is None or not np.isfinite(expected_fps) or expected_fps <= 0:
            return {'success': False, 'error': f'Invalid expected FPS for {camera_type}'}
        
    except Exception as e:
        return {'success': False, 'error': f'Failed to analyze video properties: {e}'}
    
    try:
        # Step 4: Synchronize Basler camera with OptiTrack frames
        if verbose:
            print("🎯 Synchronizing with OptiTrack timing...")
            
        # Calculate sampling ratio
        sampling_ratio = opti_fps / expected_fps
        step_size = max(1, int(round(sampling_ratio)))
        if sampling_ratio < 1.0 and verbose:
            print(f"   ⚠️ Camera FPS ({expected_fps:.2f} Hz) exceeds OptiTrack FPS ({opti_fps:.2f} Hz); using step_size=1")
        
        if verbose:
            print(f"   Sampling every {step_size} OptiTrack frames ({sampling_ratio:.2f} ratio)")
            
        # Subsample OptiTrack frames to match camera framerate
        camera_synced_indices = np.arange(0, len(optitrack_events), step_size)
        
        # Limit to actual number of camera frames recorded
        max_camera_frames = min(frame_count, len(camera_synced_indices))
        camera_synced_indices = camera_synced_indices[:max_camera_frames]
        
        # Get corresponding OptiTrack timestamps
        camera_timestamps = optitrack_events[camera_synced_indices]
        
        # Calculate synchronization quality
        camera_intervals = np.diff(camera_timestamps)
        actual_fps = 1.0 / np.mean(camera_intervals)
        sync_quality = abs(actual_fps - expected_fps) / expected_fps * 100
        
        if verbose:
            print(f"🎬 Synchronized {len(camera_timestamps):,} frames")
            print(f"   Timing accuracy: {100-sync_quality:.1f}% ({sync_quality:.2f}% error)")
            
    except Exception as e:
        return {'success': False, 'error': f'Failed during synchronization: {e}'}
    
    try:
        # Step 5: Insert synchronized events into database
        if verbose:
            print("💾 Inserting events into database...")
            
        # Prepare event data
        ts_start = camera_timestamps
        ts_end = camera_timestamps + 0.001  # 1ms duration
        event_type = f"{camera_type}_frames"
        
        # Get session and scan IDs from keys
        if isinstance(session_key, dict):
            session_id = session_key['session_id']
        else:
            session_id = session_key
            
        if isinstance(scan_key, dict):
            scan_id = scan_key['scan_id']
        else:
            scan_id = scan_key
        
        events_to_insert = [[session_id, scan_id, event_type, start, end] 
                           for start, end in zip(ts_start, ts_end)]
        
        # Register event type
        event.EventType.insert1({
            'event_type': event_type,
            'event_type_description': f'Synchronized {camera_type} camera frames from OptiTrack timing at {expected_fps:.1f} Hz'
        }, skip_duplicates=True)
        
        # # Delete existing events if present
        # existing_events = len((event.Event & session_key & scan_key & f'event_type="{event_type}"'))
        # if existing_events > 0:
        #     if verbose:
        #         print(f"🗑️ Deleting {existing_events} existing events")
        #     (event.Event & session_key & scan_key & f'event_type="{event_type}"').delete()
        
        # Insert new events
        event.Event.insert(events_to_insert, allow_direct_insert=True, skip_duplicates=True)
        
        # Verify insertion
        inserted_count = len((event.Event & session_key & scan_key & f'event_type="{event_type}"'))
        
        if verbose:
            print(f"✅ Successfully inserted {inserted_count:,} events")
            
        return {
            'success': True,
            'frames_inserted': inserted_count,
            'optitrack_frames': len(optitrack_events),
            'optitrack_frames_used': len(camera_timestamps),
            'optitrack_frames_discarded': len(optitrack_events) - len(camera_timestamps),
            'sync_quality': 100 - sync_quality,  # Return as percentage accuracy
            'timing_error_percent': sync_quality,
            'frame_validation': frame_validation,
            'frame_difference': frame_difference,
            'video_frames': frame_count,
            'csv_frames': csv_frame_count,
            'expected_fps': expected_fps,
            'expected_fps_source': expected_fps_source,
            'actual_fps': actual_fps,
            'optitrack_fps': opti_fps,
            'csv_path': str(csv_path)
        }
        
    except Exception as e:
        return {'success': False, 'error': f'Failed to insert events: {e}'}


def ingest_topcam(session_key, scan_key, verbose=False):
    """
    Deprecated wrapper for ingest_basler_cams (top camera only).
    """
    return ingest_basler_cams(session_key, scan_key, camera_type='mini2p1_top', verbose=verbose)


def incest_basler_cams(*args, **kwargs):
    return ingest_basler_cams(*args, **kwargs)


# =============================================================================
# CSV Timestamp Extraction for Eye Cameras (Bonsai-RX)
# =============================================================================

def extract_csv_timestamps(csv_path, col = -1, verbose=True):
    """
    Extract high-precision timestamps from Bonsai-RX camera timestamp CSV files.
    
    The CSV format contains frame metadata with ISO 8601 timestamps in the last column.
    Example line:
    16,640,480,False,U8,3,640,480,1920,6302859408,0,0,0,640,480,False,2024-08-16T14:39:14.6379904+02:00
    
    The timestamp has sub-millisecond precision and includes timezone information.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to the Bonsai-RX video timestamp CSV file.
        Filename pattern: {scan_id}_headcam_mini2p1_{side}_eye{n}_video_timestamps_{datetime}.csv
    verbose : bool
        Print progress information (default: True)
        
    Returns
    -------
    dict : Extraction results including:
        - timestamps_iso: List of original ISO 8601 timestamp strings
        - timestamps_seconds: numpy array of timestamps in seconds (relative to first frame)
        - timestamps_absolute: numpy array of absolute timestamps (seconds since epoch)
        - first_datetime: datetime object for first timestamp
        - n_frames: number of frames
        - frame_rate_hz: estimated frame rate
        - duration_sec: total duration in seconds
        - error: error message if extraction failed
    """
    from datetime import datetime
    from dateutil import parser as date_parser
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {'error': f'CSV file not found: {csv_path}'}
    
    if verbose:
        print(f"📄 Loading CSV timestamps: {csv_path.name}")
    
    try:
        # Read CSV file - no header, timestamps in last column
        timestamps_iso = []
        
        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split by comma and get last column (ISO 8601 timestamp)
                parts = line.split(',')
                if len(parts) >= 17:  # Expected format has at least 17 columns
                    ts_str = parts[col].strip()
                    timestamps_iso.append(ts_str)
        
        if len(timestamps_iso) == 0:
            return {'error': 'No timestamps found in CSV file'}
        
        if verbose:
            print(f"   📊 Found {len(timestamps_iso):,} timestamp entries")
        
        # Parse ISO 8601 timestamps to datetime objects
        datetimes = []
        parse_errors = 0
        
        for ts_str in timestamps_iso:
            try:
                dt = date_parser.isoparse(ts_str)
                datetimes.append(dt)
            except Exception:
                parse_errors += 1
                datetimes.append(None)
        
        if parse_errors > 0 and verbose:
            print(f"   ⚠️ {parse_errors} timestamps failed to parse")
        
        # Filter out None values
        valid_datetimes = [dt for dt in datetimes if dt is not None]
        
        if len(valid_datetimes) == 0:
            return {'error': 'All timestamps failed to parse'}
        
        # Get first datetime as reference
        first_datetime = valid_datetimes[0]
        
        # Convert to absolute seconds (since epoch) and relative seconds
        timestamps_absolute = np.array([dt.timestamp() if dt else np.nan for dt in datetimes])
        
        # Relative timestamps (seconds from first frame)
        first_ts = timestamps_absolute[np.isfinite(timestamps_absolute)][0]
        timestamps_seconds = timestamps_absolute - first_ts
        
        # Calculate frame rate and duration
        valid_ts = timestamps_seconds[np.isfinite(timestamps_seconds)]
        
        if len(valid_ts) > 1:
            intervals = np.diff(valid_ts)
            # Filter out outliers (more than 3x median)
            median_interval = np.median(intervals)
            good_intervals = intervals[(intervals > 0) & (intervals < median_interval * 3)]
            
            if len(good_intervals) > 0:
                frame_rate_hz = 1.0 / np.median(good_intervals)
            else:
                frame_rate_hz = np.nan
            
            duration_sec = valid_ts[-1] - valid_ts[0]
        else:
            frame_rate_hz = np.nan
            duration_sec = 0.0
        
        if verbose:
            print(f"   📊 Frame rate: {frame_rate_hz:.2f} Hz")
            print(f"   📊 Duration: {duration_sec:.2f} seconds")
            print(f"   📊 First timestamp: {first_datetime.isoformat()}")
        
        return {
            'timestamps_iso': timestamps_iso,
            'timestamps_seconds': timestamps_seconds,
            'timestamps_absolute': timestamps_absolute,
            'first_datetime': first_datetime,
            'n_frames': len(timestamps_iso),
            'n_valid': len(valid_ts),
            'frame_rate_hz': frame_rate_hz,
            'duration_sec': duration_sec,
            'parse_errors': parse_errors,
            'csv_path': str(csv_path)
        }
        
    except Exception as e:
        return {'error': f'Failed to parse CSV: {e}'}


# =============================================================================
# CSV Timestamp Extraction for Top Camera (Bonsai)
# =============================================================================

def extract_topcam_csv_timestamps(csv_path, verbose=True):
    """
    Extract timestamps from top camera Bonsai CSV files, supporting both ISO 8601 and Basler hardware timestamps.

    If a column of large integers (Basler hardware timestamps, e.g. column 1) is present, use it as 'timestamps_absolute' (converted to seconds if needed).
    Otherwise, use ISO 8601 timestamps as 'timestamps_absolute'.
    Always provide 'timestamps_iso' if possible.
    """
    import numpy as np
    import pandas as pd
    from dateutil import parser as date_parser
    from pathlib import Path
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {'error': f'CSV file not found: {csv_path}'}

    if verbose:
        print(f"\U0001F4C4 Loading topcam CSV timestamps: {csv_path.name}")

    try:
        df = pd.read_csv(csv_path, header=None, dtype=str, keep_default_na=False)
        if df.empty:
            return {'error': 'No timestamps found in CSV file'}

        # Detect ISO 8601 timestamp column
        sample = df.head(200)
        iso_col = None
        iso_ratio = 0.0
        for col in sample.columns:
            values = sample[col].astype(str).str.strip().values
            total = 0
            ok = 0
            for v in values:
                if not v:
                    continue
                total += 1
                if 'T' not in v:
                    continue
                try:
                    date_parser.isoparse(v)
                    ok += 1
                except Exception:
                    pass
            if total > 0:
                ratio = ok / total
                if ratio > iso_ratio:
                    iso_ratio = ratio
                    iso_col = col

        # Detect Basler hardware timestamp column (large integers, e.g. col 1)
        basler_col = None
        for col in df.columns:
            try:
                vals = pd.to_numeric(df[col], errors='coerce')
                # Heuristic: at least 90% are large positive integers (e.g. >1e9)
                valid = vals[vals > 1e9]
                if len(valid) > 0 and len(valid) / len(vals) > 0.9:
                    basler_col = col
                    break
            except Exception:
                continue

        # Extract ISO timestamps if available
        timestamps_iso = None
        datetimes = None
        parse_errors = 0
        if iso_col is not None and iso_ratio > 0.3:
            timestamps_iso = df[iso_col].astype(str).str.strip().tolist()
            datetimes = []
            for ts_str in timestamps_iso:
                if not ts_str:
                    datetimes.append(None)
                    parse_errors += 1
                    continue
                try:
                    dt = date_parser.isoparse(ts_str)
                    datetimes.append(dt)
                except Exception:
                    datetimes.append(None)
                    parse_errors += 1

            # Remove leading non-timestamp rows
            first_valid_idx = next((i for i, dt in enumerate(datetimes) if dt is not None), None)
            if first_valid_idx is not None and first_valid_idx > 0:
                if verbose:
                    print(f"   ⚠️ Skipping {first_valid_idx} non-timestamp rows (likely header)")
                timestamps_iso = timestamps_iso[first_valid_idx:]
                datetimes = datetimes[first_valid_idx:]
                parse_errors = sum(1 for dt in datetimes if dt is None)
        else:
            timestamps_iso = None
            datetimes = None

        # Extract Basler hardware timestamps if available
        timestamps_absolute = None
        absolute_source = None
        if basler_col is not None:
            vals = pd.to_numeric(df[basler_col], errors='coerce')
            # Heuristic: Basler timestamps are in 100ns units (common for hardware counters)
            # Convert to seconds: 1e7 = 1s
            # If values are too large, try dividing by 1e7 or 1e6
            median_val = np.nanmedian(vals)
            if median_val > 1e13:
                timestamps_absolute = vals / 1e9
            elif median_val > 1e9:
                timestamps_absolute = vals / 1e6
            elif median_val > 1e13:
                timestamps_absolute = vals / 1e7
            else:
                timestamps_absolute = vals
            absolute_source = 'basler'
        elif datetimes is not None:
            timestamps_absolute = np.array([dt.timestamp() if dt else np.nan for dt in datetimes])
            absolute_source = 'iso'

        # Compute relative seconds
        if timestamps_absolute is not None and np.isfinite(timestamps_absolute).sum() > 0:
            first_ts = timestamps_absolute[np.isfinite(timestamps_absolute)][0]
            timestamps_seconds = timestamps_absolute - first_ts
            valid_ts = timestamps_seconds[np.isfinite(timestamps_seconds)]
        else:
            timestamps_seconds = None
            valid_ts = []

        # Frame rate and duration
        if timestamps_seconds is not None and len(valid_ts) > 1:
            intervals = np.diff(valid_ts)
            median_interval = np.median(intervals)
            good_intervals = intervals[(intervals > 0) & (intervals < median_interval * 3)]
            if len(good_intervals) > 0:
                frame_rate_hz = 1.0 / np.median(good_intervals)
            else:
                frame_rate_hz = np.nan
            duration_sec = valid_ts.iloc[-1] - valid_ts.iloc[0]
        else:
            frame_rate_hz = np.nan
            duration_sec = 0.0

        if verbose:
            n_iso = len(timestamps_iso) if timestamps_iso is not None else 0
            print(f"   📊 Found {n_iso:,} ISO timestamp entries" if n_iso else "   📊 No ISO timestamps detected")
            if absolute_source == 'basler':
                print(f"   📊 Using Basler hardware timestamps as absolute time (column {basler_col})")
            elif absolute_source == 'iso':
                print(f"   📊 Using ISO timestamps as absolute time")
            print(f"   📊 Frame rate: {frame_rate_hz:.2f} Hz")
            print(f"   📊 Duration: {duration_sec:.2f} seconds")
            if datetimes is not None and any(dt is not None for dt in datetimes):
                first_datetime = next(dt for dt in datetimes if dt is not None)
                print(f"   📊 First ISO timestamp: {first_datetime.isoformat()}")

        return {
            'timestamps_iso': timestamps_iso,
            'timestamps_seconds': timestamps_seconds,
            'timestamps_absolute': np.array(timestamps_absolute) if timestamps_absolute is not None else None,
            'first_datetime': next((dt for dt in datetimes if dt is not None), None) if datetimes is not None else None,
            'n_frames': len(df),
            'n_valid': len(valid_ts),
            'frame_rate_hz': frame_rate_hz,
            'duration_sec': duration_sec,
            'parse_errors': parse_errors,
            'csv_path': str(csv_path),
            'absolute_source': absolute_source
        }

    except Exception as e:
        return {'error': f'Failed to parse CSV: {e}'}
def find_csv_timestamp_file(session_dir, camera_type, verbose=True):
    """
    Find the Bonsai-RX timestamp CSV file for a given eye camera.
    
    Parameters
    ----------
    session_dir : str or Path
        Session directory containing camera files
    camera_type : str
        Camera type identifier (e.g., 'mini2p1_eye_left', 'mini2p1_eye_right')
    verbose : bool
        Print progress information
        
    Returns
    -------
    Path or None : Path to the CSV file if found, None otherwise
    """
    session_dir = Path(session_dir)
    
    # Map camera type to file pattern
    # Pattern: {scan_id}_headcam_mini2p1_{side}_eye{n}_video_timestamps_{datetime}.csv
    if 'left' in camera_type:
        patterns = ['*_headcam_mini2p1_left_eye*_video_timestamps_*.csv',
                    '*_left_eye*_video_timestamps*.csv',
                    '*left*timestamps*.csv']
    elif 'right' in camera_type:
        patterns = ['*_headcam_mini2p1_right_eye*_video_timestamps_*.csv',
                    '*_right_eye*_video_timestamps*.csv',
                    '*right*timestamps*.csv']
    else:
        patterns = [
            f'*{camera_type}*video_timestamps*.csv',
            '*worldcam*video_timestamps*.csv',
            '*worldcam*timestamps*.csv'
        ]
    
    # Search for CSV files matching patterns
    for pattern in patterns:
        matches = list(session_dir.glob(pattern))
        if matches:
            if len(matches) > 1 and verbose:
                print(f"   ⚠️ Multiple CSV files found for {camera_type}, using: {matches[0].name}")
            return matches[0]
    
    if verbose:
        print(f"   ⚠️ No CSV timestamp file found for {camera_type}")
    
    return None


def find_basler_cam_csv_timestamp_file(session_dir, camera_type, verbose=True):
    """
    Find the Bonsai Basler camera timestamp CSV file for a given session directory.

    Parameters
    ----------
    session_dir : str or Path
        Session directory containing camera files
    camera_type : str
        Basler camera identifier (e.g., 'mini2p1_top', 'mini2p1_bottom')
    verbose : bool
        Print progress information

    Returns
    -------
    Path or None : Path to the CSV file if found, None otherwise
    """
    session_dir = Path(session_dir)

    patterns = [
        f'*_{camera_type}_video_timestamps*.csv',
        f'*{camera_type}*video_timestamps*.csv',
        f'*{camera_type}*timestamps*.csv'
    ]

    for pattern in patterns:
        matches = sorted(session_dir.glob(pattern))
        if matches:
            if len(matches) > 1 and verbose:
                print(f"   ⚠️ Multiple {camera_type} CSV files found, using: {matches[0].name}")
            return matches[0]

    if verbose:
        print(f"   ⚠️ No {camera_type} CSV timestamp file found")

    return None


def find_topcam_csv_timestamp_file(session_dir, verbose=True):
    """
    Deprecated wrapper for find_basler_cam_csv_timestamp_file (top camera only).
    """
    return find_basler_cam_csv_timestamp_file(session_dir, camera_type='mini2p1_top', verbose=verbose)


def _first_iso_datetime(values):
    from dateutil import parser as date_parser

    if values is None:
        return None
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        try:
            return date_parser.isoparse(text)
        except Exception:
            continue
    return None


def _fetch_first_event_time(key, event_type):
    from adamacs.pipeline import event

    query = (event.Event & key & f'event_type="{event_type}"')
    if len(query) == 0:
        return None
    times = query.fetch('event_start_time', order_by='event_start_time').astype(float)
    if len(times) == 0:
        return None
    return float(times[0])


def _prepare_eye_csv_timestamps(csv_ts, verbose=False):
    csv_ts = np.array(csv_ts, dtype=float)
    csv_ts = csv_ts[np.isfinite(csv_ts)]
    if len(csv_ts) == 0:
        return {'error': 'No valid CSV timestamps'}

    intervals = np.diff(csv_ts)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if len(intervals) == 0:
        return {
            'timestamps': csv_ts,
            'frame_rate_hz': np.nan,
            'mean_interval': np.nan,
            'deinterlaced': False,
            'n_original': len(csv_ts),
            'n_synthetic': 0
        }

    median_interval = np.median(intervals)
    good_intervals = intervals[(intervals > 0) & (intervals < median_interval * 3)]
    if len(good_intervals) == 0:
        good_intervals = intervals

    mean_interval = float(np.mean(good_intervals))
    frame_rate_hz = 1.0 / mean_interval if mean_interval > 0 else np.nan

    deinterlace = mean_interval > 0.03
    if deinterlace:
        half = mean_interval / 2.0
        expanded = np.empty(len(csv_ts) * 2, dtype=float)
        expanded[0::2] = csv_ts
        expanded[1::2] = csv_ts + half
        expanded = expanded[np.isfinite(expanded)]
        expanded.sort()
        if verbose and np.isfinite(frame_rate_hz):
            print(f"  deinterlace: {frame_rate_hz:.2f} Hz -> {2*frame_rate_hz:.2f} Hz (mean {mean_interval*1000:.2f} ms)")
        return {
            'timestamps': expanded,
            'frame_rate_hz': frame_rate_hz,
            'mean_interval': mean_interval,
            'deinterlaced': True,
            'n_original': len(csv_ts),
            'n_synthetic': len(expanded) - len(csv_ts)
        }

    if verbose and np.isfinite(frame_rate_hz):
        print(f"  csv rate: {frame_rate_hz:.2f} Hz (no deinterlace)")

    return {
        'timestamps': csv_ts,
        'frame_rate_hz': frame_rate_hz,
        'mean_interval': mean_interval,
        'deinterlaced': False,
        'n_original': len(csv_ts),
        'n_synthetic': 0
    }


def align_eye_csv_to_event_using_topcam(session_id, scan_id, camera_type,
                                        topcam_camera_type='mini2p1_top',
                                        delete_existing=False, verbose=True):
    from adamacs.pipeline import event

    key = {'session_id': session_id, 'scan_id': scan_id}
    topcam_event_type = f"{topcam_camera_type}_frames"
    eye_event_type = f"{camera_type}_frames"

    try:
        top_raw = (event.CameraTimestamps & key & f'event_type="{topcam_event_type}"').fetch1()
    except Exception as exc:
        if verbose:
            print(f"{camera_type}: missing topcam CameraTimestamps ({exc})")
        return {'success': False, 'error': f'Topcam CameraTimestamps missing: {exc}'}

    try:
        eye_raw = (event.CameraTimestamps & key & f'event_type="{eye_event_type}"').fetch1()
    except Exception as exc:
        if verbose:
            print(f"{camera_type}: missing eye CameraTimestamps ({exc})")
        return {'success': False, 'error': f'Eye CameraTimestamps missing: {exc}'}

    top_event_time0 = _fetch_first_event_time(key, topcam_event_type)
    if top_event_time0 is None:
        return {'success': False, 'error': f'No Event timestamps for {topcam_event_type}'}

    top_iso = _first_iso_datetime(top_raw.get('csv_timestamps_iso'))
    if top_iso is None:
        top_iso = top_raw.get('csv_start_datetime')
    eye_iso = _first_iso_datetime(eye_raw.get('csv_timestamps_iso'))
    if eye_iso is None:
        eye_iso = eye_raw.get('csv_start_datetime')

    if top_iso is None or eye_iso is None:
        return {'success': False, 'error': 'Missing ISO timestamps for topcam/eye'}

    delta_sec = (eye_iso - top_iso).total_seconds()

    prep = _prepare_eye_csv_timestamps(eye_raw.get('csv_timestamps', []), verbose=verbose)
    if 'error' in prep:
        return {'success': False, 'error': prep['error']}

    csv_ts = prep['timestamps']
    aligned = top_event_time0 + delta_sec + csv_ts
    aligned = aligned[np.isfinite(aligned)]

    if len(aligned) == 0:
        return {'success': False, 'error': 'No aligned timestamps (all non-finite)'}

    event_type_new = f"{camera_type}_frames_bonsaicsv"
    event.EventType.insert1({
        'event_type': event_type_new,
        'event_type_description': f'Bonsai CSV timestamps aligned via topcam anchor for {camera_type}'
    }, skip_duplicates=True)

    if delete_existing:
        (event.Event & key & f'event_type="{event_type_new}"').delete()

    events_to_insert = [
        [session_id, scan_id, event_type_new, float(ts), float(ts) + 0.005]
        for ts in aligned
    ]

    event.Event.insert(events_to_insert, allow_direct_insert=True, skip_duplicates=True)

    return {
        'success': True,
        'n_inserted': len(events_to_insert),
        'delta_sec': delta_sec,
        'top_event_time0': float(top_event_time0),
        'csv_rate_hz': prep.get('frame_rate_hz', np.nan),
        'deinterlaced': prep.get('deinterlaced', False),
        'n_synthetic': prep.get('n_synthetic', 0)
    }


def ingest_camera_timestamps(session_key, scan_key, 
                             eye_cameras=['mini2p1_eye_left', 'mini2p1_eye_right', 'mini2p1_worldcam'],
                             model_path=None, verbose=True,
                             align_bonsai_csv_to_event=False,
                             topcam_camera_type='mini2p1_top',
                             delete_existing_bonsai_csv=False,
                             deinterlace_input_fps=25.0):
    """
    Ingest raw camera timestamps into the CameraTimestamps table for sanity checking.
    
    This function extracts and stores:
    1. Raw OCR frame indices (uncorrected OptiTrack frame numbers from video overlay)
    2. CSV timestamps (high-precision Bonsai-RX timestamps)
    
    The data serves as a fallback and sanity check for the corrected timestamps
    stored in the Event table.
    
    Parameters
    ----------
    session_key : str or dict
        Session identifier
    scan_key : str or dict
        Scan identifier
    eye_cameras : list, optional
        List of eye/world camera types to process (default: left, right, worldcam)
    model_path : str or Path, optional
        Path to digit recognition model for OCR extraction.
        If None, uses default location.
    verbose : bool
        Print detailed progress information (default: True)
    align_bonsai_csv_to_event : bool
        Align eye CSV timestamps to the Event timebase via topcam anchor.
    topcam_camera_type : str
        Top camera identifier (default: 'mini2p1_top')
    delete_existing_bonsai_csv : bool
        Delete existing *_frames_bonsaicsv events before inserting.
    deinterlace_input_fps : float
        Input FPS for deinterlacing interlaced PAL/NTSC sources (default: 25.0).
        Set to 29.97 for NTSC if needed.
        
    Returns
    -------
    dict : Processing results and statistics
    """
    from element_interface.utils import find_full_path
    from adamacs.paths import get_experiment_root_data_dir
    from adamacs.pipeline import event, session
    
    if verbose:
        print(f"📊 Starting CameraTimestamps ingestion for session {session_key}, scan {scan_key}")
    
    # Get session and scan IDs
    if isinstance(session_key, dict):
        session_id = session_key['session_id']
    else:
        session_id = session_key
        
    if isinstance(scan_key, dict):
        scan_id = scan_key['scan_id']
    else:
        scan_id = scan_key
    
    key = {'session_id': session_id, 'scan_id': scan_id}
    
    # Get session directory
    try:
        session_dir_rel = (session.SessionDirectory & key).fetch1('session_dir')
        session_dir = find_full_path(get_experiment_root_data_dir(), session_dir_rel)
        if verbose:
            print(f"📁 Session directory: {session_dir}")
    except Exception as e:
        return {'error': f'Failed to get session directory: {e}'}
    
    # Get video files from database
    try:
        video_file_key = (model.VideoRecordingNew & key & 
                         f'camera IN {tuple(eye_cameras)}').fetch('KEY')
        video_files = (model.VideoRecordingNew * model.VideoRecordingNew.File & video_file_key).fetch(as_dict=True)
    except Exception as e:
        return {'error': f'Failed to fetch video files from database: {e}'}

    video_files = _prefer_deinterlaced_video_files(video_files)

    if len(video_files) == 0:
        if verbose:
            print("❌ No eye camera videos found in database")
        return {'error': 'No eye camera videos found'}
    
    if verbose:
        print(f"📹 Found {len(video_files)} eye camera videos")
    
    results = {}
    
    for video_info in video_files:
        camera_type = video_info['camera']
        video_path = Path(video_info['file_path'])
        is_worldcam = (
            'worldcam' in str(camera_type).lower()
            or 'worldcam' in str(video_path).lower()
        )

        deint_result, video_path = _resolve_deinterlaced_video_path(
            video_info,
            input_fps=deinterlace_input_fps,
            verbose=verbose,
        )
        if 'error' in deint_result:
            if is_worldcam:
                if verbose:
                    print(f"   ⚠️ Skipping worldcam {video_path.name}: {deint_result['error']}")
                continue
            return {'error': f'Failed to deinterlace {camera_type}: {deint_result["error"]}'}
        
        if verbose:
            print(f"\n🔄 Processing {camera_type}: {video_path.name}")
        
        # Determine event type name
        event_type_name = f"{camera_type}_frames"
        
        # ===== Step 1: Extract raw OCR frame indices =====
        if verbose:
            print("   📋 Extracting raw OCR frame indices...")
        
        ocr_result = extract_eye_camera_frames_ocr(
            video_path, 
            model_path=model_path,
            verbose=False,  # Suppress detailed OCR output
            write_log=False,  # Don't write log file
            apply_corrections=False,
            camera_type=camera_type
        )
        
        if 'error' in ocr_result:
            if verbose:
                print(f"   ❌ OCR extraction failed: {ocr_result['error']}")
            raw_ocr_indices = np.array([], dtype=int)
            n_valid_ocr = 0
            n_frames = 0
        else:
            # Get RAW frame indices (before correction) - convert strings to integers
            raw_values = ocr_result['frame_indices_raw']
            raw_ocr_indices = _ocr_to_int_list(raw_values, missing_value=0)
            n_frames = ocr_result['n_frames']
            n_valid_ocr = ocr_result['n_valid']
            
            if verbose:
                print(f"   ✅ OCR: {n_valid_ocr}/{n_frames} frames with valid readings")
        
        # ===== Step 2: Extract CSV timestamps =====
        if verbose:
            print("   📋 Extracting CSV timestamps...")
        
        csv_file = find_csv_timestamp_file(session_dir, camera_type, verbose=verbose)
        
        if csv_file is None:
            # Also check in video file's directory
            csv_file = find_csv_timestamp_file(video_path.parent, camera_type, verbose=False)
        
        if csv_file is not None:
            csv_result = extract_csv_timestamps(csv_file, verbose=False)
            
            if 'error' in csv_result:
                if verbose:
                    print(f"   ❌ CSV extraction failed: {csv_result['error']}")
                csv_timestamps = np.array([], dtype=float)
                csv_timestamps_iso = np.array([], dtype=object)
                csv_start_datetime = None
                n_csv_timestamps = 0
                frame_rate_csv = np.nan
                csv_duration = 0.0
                notes = f"CSV extraction failed: {csv_result['error']}"
            else:
                csv_timestamps = csv_result['timestamps_seconds']
                csv_timestamps_iso = np.array(csv_result['timestamps_iso'], dtype=object)
                csv_start_datetime = csv_result['first_datetime']
                n_csv_timestamps = csv_result['n_frames']
                frame_rate_csv = csv_result['frame_rate_hz']
                csv_duration = csv_result['duration_sec']
                notes = ""
                
                if verbose:
                    print(f"   ✅ CSV: {n_csv_timestamps} timestamps @ {frame_rate_csv:.2f} Hz, duration {csv_duration:.1f}s")
        else:
            csv_timestamps = np.array([], dtype=float)
            csv_timestamps_iso = np.array([], dtype=object)
            csv_start_datetime = None
            n_csv_timestamps = 0
            frame_rate_csv = np.nan
            csv_duration = 0.0
            notes = "No CSV timestamp file found"
            
            if verbose:
                print(f"   ⚠️ No CSV timestamp file found")
        
        # ===== Step 3: Validate and add notes =====
        if n_frames > 0 and n_csv_timestamps > 0:
            frame_diff = abs(n_frames - n_csv_timestamps)
            if frame_diff > 0:
                notes += f"; Frame count mismatch: OCR={n_frames}, CSV={n_csv_timestamps}"
                if verbose:
                    print(f"   ⚠️ Frame count mismatch: OCR={n_frames}, CSV={n_csv_timestamps}")
        
        # ===== Step 4: Register event type and insert data =====
        # Register event type
        event.EventType.insert1({
            'event_type': event_type_name,
            'event_type_description': f'{camera_type} camera frame timestamps'
        }, skip_duplicates=True)
        
        # Prepare entry for CameraTimestamps table
        entry = {
            'session_id': session_id,
            'scan_id': scan_id,
            'event_type': event_type_name,
            'raw_ocr_frame_indices': raw_ocr_indices,
            'csv_timestamps': csv_timestamps,
            'csv_timestamps_iso': csv_timestamps_iso,
            'csv_start_datetime': csv_start_datetime,
            'n_frames': max(n_frames, n_csv_timestamps),
            'n_valid_ocr': n_valid_ocr,
            'n_csv_timestamps': n_csv_timestamps,
            'frame_rate_csv_hz': frame_rate_csv if np.isfinite(frame_rate_csv) else 0.0,
            'csv_duration_sec': csv_duration,
            'notes': notes[:1024] if notes else ''
        }
        
        # Insert into CameraTimestamps table
        try:
            event.CameraTimestamps.insert1(entry, skip_duplicates=True)
            if verbose:
                print(f"   ✅ Inserted CameraTimestamps entry for {camera_type}")
        except Exception as e:
            if verbose:
                print(f"   ❌ Failed to insert: {e}")
            return {'error': f'Failed to insert CameraTimestamps for {camera_type}: {e}'}
        
        results[camera_type] = {
            'n_frames': n_frames,
            'n_valid_ocr': n_valid_ocr,
            'n_csv_timestamps': n_csv_timestamps,
            'frame_rate_csv': frame_rate_csv,
            'csv_duration': csv_duration,
            'notes': notes
        }
    
    if verbose:
        print(f"\n🎉 CameraTimestamps ingestion complete!")
        for cam, data in results.items():
            print(f"   {cam}: OCR={data['n_valid_ocr']}/{data['n_frames']}, CSV={data['n_csv_timestamps']}")

    align_results = {}
    if align_bonsai_csv_to_event:
        for camera_type in results.keys():
            align_results[camera_type] = align_eye_csv_to_event_using_topcam(
                session_id=session_id,
                scan_id=scan_id,
                camera_type=camera_type,
                topcam_camera_type=topcam_camera_type,
                delete_existing=delete_existing_bonsai_csv,
                verbose=verbose
            )
        if verbose:
            n_ok = sum(1 for res in align_results.values() if res.get('success'))
            print(f"✅ Bonsai CSV alignment: {n_ok}/{len(align_results)} cameras")

    payload = {
        'success': True,
        'cameras_processed': len(results),
        'results': results
    }
    if align_results:
        payload['bonsai_csv_alignment'] = align_results
    return payload


def ingest_basler_cam_csv_timestamps(session_key, scan_key, camera_type='mini2p1_top', verbose=True):
    """
    Ingest Basler camera CSV timestamps into the CameraTimestamps table (no OCR).

    Parameters
    ----------
    session_key : str or dict
        Session identifier
    scan_key : str or dict
        Scan identifier
    camera_type : str, optional
        Basler camera identifier (default: 'mini2p1_top')
    verbose : bool
        Print detailed progress information (default: True)

    Returns
    -------
    dict : Processing results and statistics
    """
    from element_interface.utils import find_full_path
    from adamacs.paths import get_experiment_root_data_dir
    from adamacs.pipeline import event, session

    if verbose:
        print(f"📊 Starting Basler CSV ingestion for {camera_type} (session {session_key}, scan {scan_key})")

    if isinstance(session_key, dict):
        session_id = session_key['session_id']
    else:
        session_id = session_key

    if isinstance(scan_key, dict):
        scan_id = scan_key['scan_id']
    else:
        scan_id = scan_key

    key = {'session_id': session_id, 'scan_id': scan_id}

    session_dir = None
    try:
        session_dir_rel = (session.SessionDirectory & key).fetch1('session_dir')
        session_dir = find_full_path(get_experiment_root_data_dir(), session_dir_rel)
        if verbose:
            print(f"📁 Session directory: {session_dir}")
    except Exception as e:
        if verbose:
            print(f"⚠️ Failed to get session directory: {e}")

    video_path = None
    try:
        video_key = (model.VideoRecordingNew & key & f'camera="{camera_type}"').fetch('KEY')
        video_files = (model.VideoRecordingNew * model.VideoRecordingNew.File & video_key).fetch(as_dict=True)
        if len(video_files) > 0:
            video_path = Path(video_files[0]['file_path'])
            if verbose:
                print(f"📹 Video file: {video_path.name}")
    except Exception as e:
        if verbose:
            print(f"⚠️ Failed to fetch topcam video file: {e}")

    csv_file = None
    search_dirs = []
    if session_dir is not None:
        search_dirs.append(session_dir)
    if video_path is not None:
        search_dirs.append(video_path.parent)

    for search_dir in search_dirs:
        csv_file = find_basler_cam_csv_timestamp_file(search_dir, camera_type=camera_type, verbose=verbose)
        if csv_file is not None:
            break

    if csv_file is None:
        return {'error': f'No {camera_type} CSV timestamp file found'}

    csv_result = extract_topcam_csv_timestamps(csv_file, verbose=False)
    if 'error' in csv_result:
        return {'error': f"{camera_type} CSV extraction failed: {csv_result['error']}"}

    csv_timestamps = csv_result['timestamps_seconds']
    if csv_timestamps is None:
        return {'error': f'{camera_type} CSV extraction failed: no timestamps'}
    csv_timestamps_array = np.asarray(csv_timestamps, dtype="float64")
    csv_timestamps_iso = np.array(csv_result['timestamps_iso'], dtype=object)
    csv_start_datetime = csv_result['first_datetime']
    n_csv_timestamps = csv_result['n_frames']
    frame_rate_csv = csv_result['frame_rate_hz']
    csv_duration = csv_result['duration_sec']
    csv_timestamps_source = csv_result.get("absolute_source", "unknown")

    event_type_name = f"{camera_type}_frames"
    event.EventType.insert1({
        'event_type': event_type_name,
        'event_type_description': f'{camera_type} camera frame timestamps (CSV only)'
    }, skip_duplicates=True)

    entry = {
        'session_id': session_id,
        'scan_id': scan_id,
        'event_type': event_type_name,
        'raw_ocr_frame_indices': np.array([], dtype=int),
        'csv_timestamps': csv_timestamps_array,
        'csv_timestamps_iso': csv_timestamps_iso,
        'csv_start_datetime': csv_start_datetime,
        'n_frames': n_csv_timestamps,
        'n_valid_ocr': 0,
        'n_csv_timestamps': n_csv_timestamps,
        'frame_rate_csv_hz': frame_rate_csv if np.isfinite(frame_rate_csv) else 0.0,
        'csv_duration_sec': csv_duration,
        'notes': f'{camera_type} CSV only (no OCR) - csv_timestamps source: {csv_timestamps_source}'
    }

    try:
        event.CameraTimestamps.insert1(entry, skip_duplicates=True)
        if verbose:
            print(f"✅ Inserted CameraTimestamps entry for {camera_type}")
    except Exception as e:
        return {'error': f'Failed to insert CameraTimestamps for {camera_type}: {e}'}

    return {
        'success': True,
        'camera': camera_type,
        'n_csv_timestamps': n_csv_timestamps,
        'frame_rate_csv': frame_rate_csv,
        'csv_duration': csv_duration,
        'csv_path': csv_result.get('csv_path', '')
    }


def ingest_topcam_csv_timestamps(session_key, scan_key, camera_type='mini2p1_top', verbose=True):
    """
    Deprecated wrapper for ingest_basler_cam_csv_timestamps (top camera only).
    """
    return ingest_basler_cam_csv_timestamps(
        session_key=session_key,
        scan_key=scan_key,
        camera_type=camera_type,
        verbose=verbose
    )

def align_basler_cam_csv_to_event_using_basler_cam(session_id, scan_id, camera_type='mini2p1_top',
                                                   event_name=None, delete_existing=False, verbose=True):
    """
    Align Basler CSV timestamps to the first camera event timestamp in the Event table.
    This is analogous to align_eye_csv_to_event_using_topcam, but for Basler cameras.

    Parameters
    ----------
    session_id : str or int
        Session identifier
    scan_id : str or int
        Scan identifier
    camera_type : str, optional
        Basler camera identifier (default: 'mini2p1_top')
    delete_existing : bool, optional
        Delete existing aligned events before inserting (default: False)
    verbose : bool, optional
        Print progress information (default: True)
    event_name : str, optional
        Name of the event to align to (default: None)

    Returns
    -------
    dict : Alignment results and statistics
    """
    import numpy as np
    from adamacs.pipeline import event

    key = {'session_id': session_id, 'scan_id': scan_id}
    
    if event_name is None:
        event_name = f"{camera_type}_frames"
    event_type = f"{camera_type}_frames"

    aligned_event_type = f"{camera_type}_frames_bonsaicsv"

    # Fetch first event timestamp for camera
    query = (event.Event & key & f'event_type="{event_name}"')
    if len(query) == 0:
        if verbose:
            print(f"No Event timestamps for {event_name}")
        return {'success': False, 'error': f'No Event timestamps for {event_name}'}
    times = query.fetch('event_start_time', order_by='event_start_time').astype(float)
    if len(times) == 0:
        if verbose:
            print(f"No Event timestamps for {event_name}")
        return {'success': False, 'error': f'No Event timestamps for {event_name}'}
    event_time0 = float(times[0])

    # Fetch CameraTimestamps for camera
    try:
        cam_ts = (event.CameraTimestamps & key & f'event_type="{event_type}"').fetch1()
    except Exception as exc:
        if verbose:
            print(f"Missing CameraTimestamps for {event_type}: {exc}")
        return {'success': False, 'error': f'Missing CameraTimestamps: {exc}'}

    csv_ts = np.array(cam_ts.get('csv_timestamps', []), dtype=float)
    csv_ts = csv_ts[np.isfinite(csv_ts)]
    if len(csv_ts) == 0:
        return {'success': False, 'error': 'No valid CSV timestamps'}


    # Fetch last event timestamp for topcam
    event_timeN = float(times[-1])
    csv_tsN = csv_ts[-1]

    # Scale CSV timestamps so that first and last frame match event timebase
    # t_aligned = event_time0 + (csv_ts - csv_ts0) * scale
    if csv_tsN == csv_ts[0]:
        scale = 1.0
    else:
        scale = (event_timeN - event_time0) / (csv_tsN - csv_ts[0])
    aligned = event_time0 + (csv_ts - csv_ts[0]) * scale
    aligned = aligned[np.isfinite(aligned)]

    if len(aligned) == 0:
        return {'success': False, 'error': 'No aligned timestamps (all non-finite)'}

    # Register new event type
    event.EventType.insert1({
        'event_type': aligned_event_type,
        'event_type_description': f'Bonsai CSV timestamps aligned to first {event_name} event for {camera_type}'
    }, skip_duplicates=True)

    if delete_existing:
        (event.Event & key & f'event_type="{aligned_event_type}"').delete()

    # Insert aligned events (1ms duration)
    events_to_insert = [
        [session_id, scan_id, aligned_event_type, float(ts), float(ts) + 0.001]
        for ts in aligned
    ]
    event.Event.insert(events_to_insert, allow_direct_insert=True, skip_duplicates=True)

    if verbose:
        print(f"Inserted {len(events_to_insert)} aligned Basler CSV events for {camera_type}")
        print(f"Alignment scale factor: {scale:.8f}")

    return {
        'success': True,
        'n_inserted': len(events_to_insert),
        'event_time0': float(event_time0),
        'event_timeN': float(event_timeN),
        'csv_ts0': float(csv_ts[0]),
        'csv_tsN': float(csv_tsN),
        'scale': scale,
        'csv_rate_hz': 1.0 / np.median(np.diff(csv_ts)) if len(csv_ts) > 1 else np.nan
    }


def align_topcam_csv_to_event_using_topcam(session_id, scan_id, camera_type='mini2p1_top',
                                           event_name=None, delete_existing=False, verbose=True):
    """
    Deprecated wrapper for align_basler_cam_csv_to_event_using_basler_cam (top camera only).
    """
    return align_basler_cam_csv_to_event_using_basler_cam(
        session_id=session_id,
        scan_id=scan_id,
        camera_type=camera_type,
        event_name=event_name,
        delete_existing=delete_existing,
        verbose=verbose
    )
