import numpy as np
from element_interface.utils import find_full_path, find_root_directory
from adamacs.paths import get_experiment_root_data_dir
from pathlib import Path
import pandas as pd
from datetime import datetime

class HarpLoader:
    def __init__(self, harp_path):
        self.harp_path = Path(harp_path)
        if not self.harp_path.exists():
            self._harp_path_full = find_full_path(
                get_experiment_root_data_dir(), harp_path
            )
        else:
            self._harp_path_full = Path(harp_path)
        self._harp_path_relative = self._harp_path_full.relative_to(
            find_root_directory(get_experiment_root_data_dir(), self._harp_path_full)
        )

        self._harp_file = pd.read_csv(self._harp_path_full, header=None)


    def data(self):
        return self._data_dict

    def data_for_insert(self):
        return [
            {
                "channel_name": "IMU accelerometer 1",
                "data": self._harp_file[0].to_numpy(),
                "time": [],
            },
            {
                "channel_name": "IMU accelerometer 2",
                "data": self._harp_file[1].to_numpy(),
                "time": [],
            },            
            {
                "channel_name": "IMU accelerometer 3",
                "data": self._harp_file[2].to_numpy(),
                "time": [],
            },            
            {
                "channel_name": "IMU gyroscope 1",
                "data": self._harp_file[3].to_numpy(),
                "time": [],
            },            
            {
                "channel_name": "IMU gyroscope 2",
                "data": self._harp_file[4].to_numpy(),
                "time": [],
            },            
            {
                "channel_name": "IMU gyroscope 3",
                "data": self._harp_file[5].to_numpy(),
                "time": [],
            },            
            {
                "channel_name": "IMU magnetometer 1",
                "data": self._harp_file[6].to_numpy(),
                "time": [],
            },            
            {
                "channel_name": "IMU magnetometer 2",
                "data": self._harp_file[7].to_numpy(),
                "time": [],
            },
            {
                "channel_name": "IMU magnetometer 3",
                "data": self._harp_file[8].to_numpy(),
                "time": [],
            },
        ]


# ------------ Helper functions ------------

class HarpLoader_sync:
    def __init__(self, harp_sync_path):
        self.harp_sync_path = Path(harp_sync_path)
        if not self.harp_sync_path.exists():
            self._harp_sync_path_full = find_full_path(
                get_experiment_root_data_dir(), harp_sync_path
            )
        else:
            self._harp_sync_path_full = Path(harp_sync_path)
        self._harp_sync_path_relative = self._harp_sync_path_full.relative_to(
            find_root_directory(get_experiment_root_data_dir(), self._harp_sync_path_full)
        )

        self._harp_sync_file = pd.read_csv(self._harp_sync_path_full, header=None)


    def data(self):
        return self._data_dict

    def data_for_insert(self):
        
        # Convert the string to a datetime
        dt = pd.to_datetime(self._harp_sync_file[1].to_numpy())

        # Convert the datetime to UTC
        dt_utc = dt.tz_convert('UTC')

        # Get the Unix timestamp in microseconds
        unix_timestamp_ms = dt_utc.astype(int) // 10**3
        time = (unix_timestamp_ms - unix_timestamp_ms[0])/1000   # subtract by AUX event HARP gate
             
        return [
            {
                "channel_name": "2p sync",
                "data": self._harp_sync_file[0].to_numpy(),
                "time": time.to_numpy(),
            },
        ]
class CamLoader_sync:
    def __init__(self, cam_sync_path):
        self.cam_sync_path = Path(cam_sync_path)
        if not self.cam_sync_path.exists():
            self._cam_sync_path_full = find_full_path(
                get_experiment_root_data_dir(), cam_sync_path
            )
        else:
            self._cam_sync_path_full = Path(cam_sync_path)
        self._cam_sync_path_relative = self._cam_sync_path_full.relative_to(
            find_root_directory(get_experiment_root_data_dir(), self._cam_sync_path_full)
        )

        self._cam_sync_file = pd.read_csv(self._cam_sync_path_full, header=None)


    def data(self):
        return self._data_dict

    def data_for_insert(self):
        
        # Convert the string to a datetime
        dt = pd.to_datetime(self._cam_sync_file[0].to_numpy())

        # Convert the datetime to UTC
        dt_utc = dt.tz_convert('UTC')

        # Get the Unix timestamp in microseconds
        unix_timestamp_us = dt_utc.astype(int) // 10**3
        time = (unix_timestamp_us - unix_timestamp_us[0]) / 1000   # subtract by AUX event HARP gate
             
        return [
            {
                "channel_name": "2p sync",
                "data": self._cam_sync_file[0].to_numpy(),
                "time": time.to_numpy(),
            },
        ]

def set_bit(value, bit):
    return value | (1 << bit)


def clear_bit(value, bit):
    return value & ~(1 << bit)


def bitget(value, bit_no):
    return (value >> bit_no) & 1

