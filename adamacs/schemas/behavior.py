"""Tables related to behavioral data.

During some recordings behavioral data is recorded.
This module organizes the different types of behavioral
raw data and relates them to the Recording.
"""

import datajoint as dj
from ..pipeline import session, event, db_prefix
from ..paths import get_experiment_root_data_dir, get_user_from_dlc_root_data_dir
from ..ingest.harp import HarpLoader, HarpLoader_sync, HarpLoader_CSV
from element_interface.utils import find_full_path
from pywavesurfer import ws
import numpy as np

schema = dj.schema(db_prefix + "behavior")

__all__ = ["session", "db_prefix", "HarpDevice", "HarpRecording", "TreadmillDevice", "TreadmillRecording"]

# -------------- Table declarations --------------

# NOTE: Previous tables depreciated with the use of element-event


@schema
class HarpDevice(dj.Lookup):
    definition = """
    harp_device_id: int
    ---
    harp_device_name: varchar(36)
    harp_device_description='': varchar(1000)
    """

    contents = [(1, "HARP Wear IMU", "9doF IMU MPU-9250 Bonsai device")]


@schema
class HarpRecording(dj.Imported):
    definition = """
    -> event.BehaviorRecording
    -> HarpDevice
    """

    class Channel(dj.Part):
        definition = """
        -> master
        channel_name: varchar(36)
        ---
        data=null : longblob  # 1d array of acquired data for this channel
        time=null : longblob  # 1d array of timestamps for this channel 
        """

    def make(self, key):
                
        # try:
        aux_path_relative = (event.BehaviorRecording.File & key &  "filepath LIKE '%.h5%'").fetch("filepath")[0]

        # exchange my old tobiasr folder with the actual root users
        aux_path_relative = aux_path_relative.replace("tobiasr", get_user_from_dlc_root_data_dir())
        
        harp_paths = list(find_full_path(
                get_experiment_root_data_dir(), aux_path_relative
            ).parent.glob("*IMU_harp_weardata_20*.csv"))
        usebinary = False
        
        if not harp_paths:
            harp_paths = list(find_full_path(
                get_experiment_root_data_dir(), aux_path_relative
            ).parent.glob("*IMU_harp*.bin"))
            usebinary = True

        if not harp_paths:
            return

        for p in harp_paths:
            print(p)            
            event.BehaviorRecording.File.insert1([key['session_id'], key['scan_id'], p], skip_duplicates=False)

        if not usebinary:
            # assert len(harp_paths) == 1, f"Found less or more than one harp CSV file\n\t{harp_paths}"
            IMU_data = HarpLoader_CSV(harp_paths[0]).data_for_insert()
        if usebinary:
            # assert len(harp_paths) == 3, f"Found less or more than three harp BIN files\n\t{harp_paths}"
            IMU_data = HarpLoader(harp_paths).data_for_insert()
        
        harp_frame_paths = list(find_full_path(
            get_experiment_root_data_dir(), aux_path_relative
        ).parent.glob("*2PFrames*.csv"))
        if not harp_frame_paths:
            harp_frame_paths = list(find_full_path(
                get_experiment_root_data_dir(), aux_path_relative
            ).parent.glob("*2Pframes*.csv"))
            # assert len(harp_frame_paths) == 1, f"Found less or more than one harp 2P Frame file\n\t{harp_frame_paths}"        

        IMU_sync_data = HarpLoader_sync(harp_frame_paths[0]).data_for_insert()
        
        insert_data = IMU_data + IMU_sync_data
        self.insert1(key, skip_duplicates=True)
        self.Channel.insert(
            [
                {**key, **channel} 
                for channel in insert_data
            ], skip_duplicates=True
        )
        # except:
            # print(f'Failed to ingest HARP IMU data for {key}')    

@schema
class TreadmillDevice(dj.Lookup):
    definition = """
    treadmill_device_id: int
    ---
    treadmill_device_name: varchar(36)
    treadmill_device_description='': varchar(1000)
    """

    contents = [(1, "rotary encoder", "BPod rotary encoder on running wheel")]


@schema
class TreadmillRecording(dj.Imported):
    definition = """
    -> event.BehaviorRecording
    -> TreadmillDevice
    """

    class Channel(dj.Part):
        definition = """
        -> master
        channel_name: varchar(36)
        ---
        data=null : longblob  # 1d array of acquired data for this channel
        time=null : longblob  # 1d array of timestamps for this channel 
        """

    def make(self, key):
        aux_path_relative = (event.BehaviorRecording.File & key &  "filepath LIKE '%.h5%'").fetch("filepath")[0]
        treadmill_chan = 1

        curr_aux = ws.loadDataFile(filename=aux_path_relative, format_string='double' )
        sweep = [x for x in curr_aux.keys() if 'sweep' in x][0]
        sr = curr_aux['header']['AcquisitionSampleRate'][0][0]
        
        timebase = np.arange(curr_aux[sweep]['analogScans'].shape[1]) / sr
        bpod_wheel_data = curr_aux[sweep]['analogScans'][treadmill_chan]
        
        downsample = 10
        timebase = timebase[::downsample]
        bpod_wheel_data = bpod_wheel_data[::downsample]
        
        self.insert1(key)

        channeldata =  {
                "channel_name": 'wheel_pos',
                "data": bpod_wheel_data,
                "time": timebase
                }
        self.Channel.insert([{**key, **channeldata}])

@schema
class CamSyncDevice(dj.Lookup):
    definition = """
    camsync_device_id: int
    ---
    camsync_device_name: varchar(36)
    camsync_device_description='': varchar(1000)
    """

    contents = [(1, "analog command signal", "Data on wafesurfer channel corresponding to the camera sync signal (analog waveform with pseudo-random modulation)")]


@schema
class CamSyncRecording(dj.Imported):
    definition = """
    -> event.BehaviorRecording
    -> CamSyncDevice
    """

    class Channel(dj.Part):
        definition = """
        -> master
        channel_name: varchar(36)
        ---
        data=null : longblob  # 1d array of acquired data for this channel
        time=null : longblob  # 1d array of timestamps for this channel 
        """

    def make(self, key):
        aux_path_relative = (event.BehaviorRecording.File & key &  "filepath LIKE '%.h5%'").fetch("filepath")[0]
        CamSyncChan = 4
        curr_aux = ws.loadDataFile(filename=aux_path_relative, format_string='double' )
        sweep = [x for x in curr_aux.keys() if 'sweep' in x][0]
        sr = curr_aux['header']['AcquisitionSampleRate'][0][0]
        
        timebase = np.arange(curr_aux[sweep]['analogScans'].shape[1]) / sr
        camsync_data = curr_aux[sweep]['analogScans'][CamSyncChan]
        
        downsample = 10
        timebase = timebase[::downsample]
        camsync_data = camsync_data[::downsample]
        
        self.insert1(key)

        channeldata =  {
                "channel_name": 'cam_sync',
                "data": camsync_data,
                "time": timebase
                }
        self.Channel.insert([{**key, **channeldata}])

