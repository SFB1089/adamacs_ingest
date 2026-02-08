import datajoint as dj
from . import subject
#from adamacs.schemas import subject
from element_deeplabcut.model import PoseEstimationNew
from ..pipeline import session, scan, model, db_prefix

from adamacs.schemas.mocap import MotionCapture, MotionCaptureTask

import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as Rscipy


schema = dj.schema(db_prefix + 'virtual_markers_optitrack')


@schema
class EyeNoseCamPosCalib(dj.Manual):
  """local relation to head markers in optitrack"""
  definition = """
  -> subject.Subject
  ---
  eye_left:  longblob # 3D position of the left eye in local headset coordinates
  eye_right:  longblob # 3D position of the right eye in local headset coordinates
  nose:  longblob # 3D position of the nose in local headset coordinates
  world_cam:  longblob # 3D position of the world camera in local headset coordinates
  """

  # EyeNoseCamPosCalib.insert1({'subject': 1, 'eye_left': (x,y,z), 'eye_right': (x,y,z), 'nose': (x,y,z), 'world_cam': (x,y,z)})

@schema
class RigidMouseTracking(dj.Computed):
    """"""
    definition = """
    -> scan.Scan
    -> EyeNoseCamPosCalib
    -> MotionCaptureTask
    ---
    pivot_eye_center_global: longblob # 3D position of the eye pivot point in world coordinates
    yaw:  longblob #
    pitch:  longblob #
    roll:  longblob #
    q_x:  longblob #
    q_y:  longblob #
    q_z:  longblob #
    q_w:  longblob #
    eye_left_global: longblob # 3D position of the left eye in world coordinates
    eye_right_global: longblob # 3D position of the right eye in world coordinates
    nose_global: longblob # 3D position of the nose in world coordinates
    world_cam_global: longblob # 3D position of the world camera in world coordinates
    """


    def define_rigid_body_frame(self, p1, p2, p3):
        origin = p1
        x_axis = p2 - p1
        x_axis /= np.linalg.norm(x_axis)

        temp_vec = p3 - p1
        z_axis = np.cross(x_axis, temp_vec)
        z_axis /= np.linalg.norm(z_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        t = origin
        return R, t
    

    def process_array(self, arr, max_linear=10, max_cubic=60):
        """
        Process a (n_frames, 3) array:
        - Optional Y/Z swap
        - Low-pass filter (ignoring NaNs)
        - Fill gaps with interpolation
        Returns: (n_frames, 3) numpy array
        """
        # --- Parameters ---
        fs = 240      # Sampling rate (Hz)
        cutoff = 15   # Low-pass cutoff (Hz)
        order = 2     # Filter order

        # --- Filter design ---
        b, a = butter(order, cutoff / (0.5 * fs), btype="low")

        # Define transformation matrix
        M = np.array([
            [1, 0, 0],  
            [ 0, 0, -1], 
            [ 0, 1, 0]   
        ])
        arr_transformed = arr @ M.T

        df = pd.DataFrame(arr_transformed, columns=["x", "y", "z"])

        # Apply low-pass filter on valid segments
        for c in df.columns:
            series = df[c].to_numpy()
            mask = ~np.isnan(series)
            if mask.sum() > 3:  # filtfilt needs some valid points
                series[mask] = filtfilt(b, a, series[mask])
            df[c] = series

        # Fill gaps
        df = df.interpolate(method="linear", limit=max_linear, limit_direction="both")
        df = df.interpolate(method="cubic", limit=max_cubic, limit_direction="both")

        return df.to_numpy()
      
    def make(self, key):

        eye_left_local = (EyeNoseCamPosCalib & key).fetch1('eye_left')   # shape: (n_frames, 3)
        eye_right_local = (EyeNoseCamPosCalib & key).fetch1('eye_right')
        nose_local = (EyeNoseCamPosCalib & key).fetch1('nose')
        world_cam_local = (EyeNoseCamPosCalib & key).fetch1('world_cam')

        # --- Here assume mocap is raw data without axis swapping yet, i.e. y is up ---
        head_1_pos_xyz = (MotionCapture.TrackingPosition & key & 'tracking_id="Mouse_trackers:Marker 001"').fetch1('x_pos', 'y_pos', 'z_pos')
        head_1_pos = np.stack(head_1_pos_xyz, axis=-1).squeeze()
        head_2_pos_xyz = (MotionCapture.TrackingPosition & key & 'tracking_id="Mouse_trackers:Marker 002"').fetch1('x_pos', 'y_pos', 'z_pos')
        head_2_pos = np.stack(head_2_pos_xyz, axis=-1).squeeze()
        head_3_pos_xyz = (MotionCapture.TrackingPosition & key & 'tracking_id="Mouse_trackers:Marker 003"').fetch1('x_pos', 'y_pos', 'z_pos')
        head_3_pos = np.stack(head_3_pos_xyz, axis=-1).squeeze() ## shape: (n_frames, 3)

        # --- Step 1: Process data (changing axises to correspond to the common orientation, filtering and gap filling) ---
        head_1_pos_processed = self.process_array(head_1_pos)
        head_2_pos_processed = self.process_array(head_2_pos)
        head_3_pos_processed = self.process_array(head_3_pos)

        # --- Step 2: Reconstruct missing markers ---
        eye_left_global = []
        eye_right_global = []
        nose_global = []
        world_cam_global  = []

        for frame in range(head_1_pos_processed.shape[0]):
            # Extract the three reference headset markers
            p1 = head_1_pos_processed[frame]
            p2 = head_2_pos_processed[frame]
            p3 = head_3_pos_processed[frame]

            # Compute rigid body transform
            R, t = self.define_rigid_body_frame(p1, p2, p3)

            # Reconstruct all other markers
            eye_left_global.append(R @ eye_left_local + t)
            eye_right_global.append(R @ eye_right_local + t)
            nose_global.append(R @ nose_local + t)
            world_cam_global.append(R @ world_cam_local + t)

        # Step 3: Compute eye center (position)
        pivot_eye_center_pos = (np.array(eye_left_global) + np.array(eye_right_global)) / 2 # shape: (n_frames, 3)
 
        # Step 4: Compute forward vector (eye_center → nose) Y-axis
        forward = np.array(nose_global) - pivot_eye_center_pos
        norm_forward = np.linalg.norm(forward, axis=1, keepdims=True)
        forward = np.divide(forward, norm_forward, out=np.zeros_like(forward), where=norm_forward != 0)

        # Step 5: Compute right vector (eye_center → right eye) X-axis
        right = np.array(eye_right_global) - pivot_eye_center_pos
        norm_right = np.linalg.norm(right, axis=1, keepdims=True)
        right = np.divide(right, norm_right, out=np.zeros_like(right), where=norm_right != 0)

        # Step 6: Compute up vector (orthogonal to forward & right) Z-axis
        up = np.cross(right, forward)
        norm_up = np.linalg.norm(up, axis=1, keepdims=True)
        up = np.divide(up, norm_up, out=np.zeros_like(up), where=norm_up != 0)

        # Step 7: Build rotation matrices and convert to quaternions
        rotation_matrices = np.stack([right, forward, up], axis=-1)  # (n_frames, 3, 3)

        # Handle NaNs (rows with NaNs will cause invalid quaternions)
        valid_rows = ~np.isnan(rotation_matrices).any(axis=(1, 2))
        quaternions = np.full((rotation_matrices.shape[0], 4), np.nan)
        quaternions[valid_rows] = Rscipy.from_matrix(rotation_matrices[valid_rows]).as_quat()  # (x, y, z, w)

        q_x = quaternions[:, 0]
        q_y = quaternions[:, 1]
        q_z = quaternions[:, 2]
        q_w = quaternions[:, 3]

        # Step 8: Convert quaternions to Euler angles (roll, pitch, yaw)
        yaw_roll_pitch = Rscipy.from_matrix(rotation_matrices[valid_rows]).as_euler('ZYX', degrees=False)   # shape: (n_valid, 3)
        roll_pitch_yaw = Rscipy.from_matrix(rotation_matrices[valid_rows]).as_euler('XYZ', degrees=False)   # shape: (n_valid, 3)

        # Allocate full arrays (NaN-filled)
        yaw   = np.full(rotation_matrices.shape[0], np.nan)
        pitch = np.full_like(yaw, np.nan)
        roll  = np.full_like(yaw, np.nan)

        yaw2   = np.full(rotation_matrices.shape[0], np.nan)
        pitch2 = np.full_like(yaw, np.nan)
        roll2  = np.full_like(yaw, np.nan)

        # Fill with radians
        yaw[valid_rows]   = yaw_roll_pitch[:, 0]   # rotation about Z (up)
        roll[valid_rows] = yaw_roll_pitch[:, 1]   # rotation about Y (forward)
        pitch[valid_rows]  = yaw_roll_pitch[:, 2]   # rotation about X (right)

        self.insert1({**key, 'pivot_eye_center_global': pivot_eye_center_pos, 'yaw': yaw, 'pitch': pitch, 'roll': roll, 'q_x': q_x, 'q_y': q_y, 'q_z': q_z, 'q_w': q_w, 'eye_left_global': np.array(eye_left_global), 'eye_right_global': np.array(eye_right_global), 'nose_global': np.array(nose_global), 'world_cam_global': np.array(world_cam_global)})
