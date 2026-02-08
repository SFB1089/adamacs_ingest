"""
DISK (Deep Imputation of SKeleton data) Integration Schema

This module provides DataJoint tables for DISK-based imputation of:
1. DeepLabCut (DLC) 2D pose estimation data  
2. OptiTrack/Motive 3D motion capture data

Reference: https://github.com/bozeklab/DISK

Table hierarchy:
    DISKModel (Lookup) - Registry of trained DISK models
    
    DLCImputationTask (Manual) -> DLCImputation (Computed)
        └── BodyPartPositionDISK (Part)
    
    MocapImputationTask (Manual) -> MocapImputation (Computed)
        ├── TrackingPositionDISK (Part)
        └── RigidBodyPositionDISK (Part)
"""

import datajoint as dj
import numpy as np
from datetime import datetime
from pathlib import Path

from element_interface.utils import dict_to_uuid

# Import required modules from adamacs pipeline
from adamacs.pipeline import model
from adamacs.schemas import mocap

# Get schema prefix from configuration
db_prefix = dj.config['custom']['database.prefix']
schema = dj.schema(db_prefix + 'disk')


@schema
class DISKModel(dj.Lookup):
    """Registry of trained DISK models for marker/keypoint imputation.
    
    Models are trained using the DISK framework from https://github.com/bozeklab/DISK
    Each model is trained for a specific data type, number of keypoints, and sequence length.
    
    Attributes
    ----------
    disk_model_name : str
        Unique model identifier
    model_path : str
        Path to trained checkpoint folder (.hydra config must be present)
    model_description : str
        Description of model and training data
    data_type : str
        Type of data model was trained on ('dlc_2d', 'dlc_3d', or 'mocap_3d')
    n_keypoints : int
        Number of keypoints model expects
    seq_length : int
        Sequence length model was trained on
    network_type : str
        Network architecture used ('transformer', 'GRU', 'BiGRU', 'ST_GCN', 'TCN')
    """
    definition = """
    disk_model_name         : varchar(64)   # unique model identifier
    ---
    model_path              : varchar(512)  # path to trained checkpoint folder (.hydra config must be present)
    model_description=''    : varchar(512)  # description of model and training data
    data_type               : enum('dlc_2d', 'dlc_3d', 'mocap_3d')  # type of data model was trained on
    n_keypoints             : smallint      # number of keypoints model expects
    seq_length              : smallint      # sequence length model was trained on
    network_type            : enum('transformer', 'GRU', 'BiGRU', 'ST_GCN', 'TCN')  # network architecture
    """


# ============================================================================
# DeepLabCut DISK Imputation Tables
# ============================================================================

@schema
class DLCImputationTask(dj.Manual):
    """Staging table for DeepLabCut DISK imputation tasks.
    
    Links a PoseEstimationNew entry with a trained DISK model.
    Insert entries here to queue imputation jobs.
    
    Attributes
    ----------
    task_mode : str
        'load': load pre-computed results, 'trigger': run DISK inference
    output_dir : str
        Directory for saving/loading results
    disk_params : blob
        Optional override parameters for DISK inference
    task_description : str
        Description of this task
    """
    definition = """
    -> model.PoseEstimationNew      # inherits session_id, scan_id, camera, model_name keys
    -> DISKModel                    # which DISK model to use
    ---
    task_mode               : enum('load', 'trigger')  # 'load': load existing, 'trigger': run inference
    output_dir=''           : varchar(512)  # where to save/load results
    disk_params=null        : longblob      # optional override parameters for DISK inference
    task_description=''     : varchar(255)  # description of this task
    """


@schema
class DLCImputation(dj.Computed):
    """DISK-imputed DeepLabCut pose estimation results.
    
    Stores imputed body part positions where missing values have been filled in.
    
    Attributes
    ----------
    imputation_time : datetime
        When imputation was performed
    n_frames_total : int
        Total number of frames
    n_frames_imputed : int
        Number of frames with at least one imputed value
    n_values_imputed : int
        Total number of individual coordinate values imputed
    """
    definition = """
    -> DLCImputationTask
    ---
    imputation_time         : datetime      # when imputation was performed
    n_frames_total          : int           # total number of frames
    n_frames_imputed        : int           # number of frames with at least one imputed value
    n_values_imputed        : int           # total number of individual coordinate values imputed
    """
    
    class BodyPartPositionDISK(dj.Part):
        """DISK-imputed body part positions.
        
        Same structure as PoseEstimationNew.BodyPartPosition but with imputation info.
        
        Attributes
        ----------
        frame_index : blob
            Frame indices (same as original)
        x_pos : blob
            X positions (with imputed values)
        y_pos : blob
            Y positions (with imputed values)
        z_pos : blob
            Z positions (if 3D tracking)
        uncertainty : blob
            DISK uncertainty estimate per frame
        imputed_mask : blob
            Boolean array: True where values were imputed
        original_likelihood : blob
            Original DLC likelihood (for reference)
        """
        definition = """
        -> master
        -> model.Model.BodyPart
        ---
        frame_index         : longblob      # frame indices (same as original)
        x_pos               : longblob      # x positions (with imputed values)
        y_pos               : longblob      # y positions (with imputed values)
        z_pos=null          : longblob      # z positions (if 3D tracking)
        uncertainty         : longblob      # DISK uncertainty estimate per frame
        imputed_mask        : longblob      # boolean: True where values were imputed
        original_likelihood : longblob      # original DLC likelihood (for reference)
        """
    
    def make(self, key):
        """Populate method: run DISK imputation on the pose estimation data."""
        
        # Get task parameters
        task_mode, output_dir, disk_params = (DLCImputationTask & key).fetch1(
            'task_mode', 'output_dir', 'disk_params'
        )
        
        # Get DISK model info
        disk_model_name, model_path, data_type, n_keypoints, seq_length, network_type = (
            DISKModel & key
        ).fetch1('disk_model_name', 'model_path', 'data_type', 'n_keypoints', 
                 'seq_length', 'network_type')
        
        # Get original pose estimation data
        pose_key = {k: key[k] for k in model.PoseEstimationNew.primary_key}
        body_parts_data = (model.PoseEstimationNew.BodyPartPosition & pose_key).fetch(as_dict=True)
        
        if task_mode == 'trigger':
            # Prepare data for DISK
            data, keypoints, mask = self._prepare_dlc_data(body_parts_data)
            
            # Run DISK imputation
            imputed_data, uncertainty, imputed_mask = self._run_disk_inference(
                data, mask, model_path, seq_length
            )
            
            # Calculate statistics
            n_frames_total = data.shape[0]
            n_frames_imputed = np.sum(np.any(imputed_mask, axis=(1, 2)))
            n_values_imputed = np.sum(imputed_mask)
            
            # Insert main entry
            self.insert1({
                **key,
                'imputation_time': datetime.now(),
                'n_frames_total': n_frames_total,
                'n_frames_imputed': n_frames_imputed,
                'n_values_imputed': n_values_imputed,
            })
            
            # Insert part table entries
            for i, bp in enumerate(body_parts_data):
                self.BodyPartPositionDISK.insert1({
                    **key,
                    'body_part': bp['body_part'],
                    'frame_index': bp.get('frame_index', np.arange(n_frames_total)),
                    'x_pos': imputed_data[:, i, 0],
                    'y_pos': imputed_data[:, i, 1],
                    'z_pos': imputed_data[:, i, 2] if data.shape[2] > 2 else None,
                    'uncertainty': uncertainty[:, i],
                    'imputed_mask': imputed_mask[:, i, 0] | imputed_mask[:, i, 1],
                    'original_likelihood': bp['likelihood'],
                })
            
        elif task_mode == 'load':
            # Load pre-computed results from output_dir
            raise NotImplementedError("DISK load mode not yet implemented")
    
    @staticmethod
    def _prepare_dlc_data(body_parts_data, likelihood_threshold=0.5):
        """Convert DLC body part data to DISK-compatible format."""
        keypoints = [bp['body_part'] for bp in body_parts_data]
        n_frames = len(body_parts_data[0]['x_pos'])
        n_keypoints = len(keypoints)
        has_z = body_parts_data[0].get('z_pos') is not None
        n_dim = 3 if has_z else 2
        
        data = np.zeros((n_frames, n_keypoints, n_dim))
        mask = np.ones((n_frames, n_keypoints), dtype=bool)
        
        for i, bp in enumerate(body_parts_data):
            x = np.array(bp['x_pos'], dtype=float)
            y = np.array(bp['y_pos'], dtype=float)
            likelihood = np.array(bp['likelihood'], dtype=float)
            
            data[:, i, 0] = x
            data[:, i, 1] = y
            if has_z:
                data[:, i, 2] = np.array(bp['z_pos'], dtype=float)
            
            missing = np.isnan(x) | np.isnan(y) | (likelihood < likelihood_threshold)
            mask[:, i] = ~missing
            data[missing, i, :] = np.nan
        
        return data, keypoints, mask
    
    @staticmethod
    def _run_disk_inference(data, mask, model_path, seq_length):
        """Run DISK imputation. Requires DISK to be installed."""
        try:
            import torch
            from glob import glob
            from omegaconf import OmegaConf
            
            # DISK imports - uncomment when DISK is installed
            # from DISK.utils.utils import read_constant_file, load_checkpoint
            # from DISK.utils.transforms import init_transforms
            # from DISK.utils.train_fillmissing import construct_NN_model, feed_forward
            # from DISK.utils.dataset_utils import ImputeDataset
            # from torch.utils.data import DataLoader
            
            raise NotImplementedError(
                "DISK inference requires DISK to be installed. "
                "Install with: pip install -e /path/to/DISK"
            )
            
        except ImportError as e:
            raise ImportError(
                f"DISK not installed. Install from https://github.com/bozeklab/DISK\n"
                f"Error: {e}"
            )


# ============================================================================
# Motion Capture DISK Imputation Tables (for OptiTrack/Motive data)
# ============================================================================

@schema
class MocapImputationTask(dj.Manual):
    """Staging table for motion capture DISK imputation tasks.
    
    Links a MotionCapture entry with a trained DISK model.
    
    Attributes
    ----------
    task_mode : str
        'load': load pre-computed results, 'trigger': run DISK inference
    output_dir : str
        Directory for saving/loading results
    disk_params : blob
        Optional override parameters for DISK inference
    task_description : str
        Description of this task
    """
    definition = """
    -> mocap.MotionCapture          # inherits session_id, scan_id, camera, mocap_name keys  
    -> DISKModel                    # which DISK model to use
    ---
    task_mode               : enum('load', 'trigger')
    output_dir=''           : varchar(512)
    disk_params=null        : longblob      # optional DISK parameters
    task_description=''     : varchar(255)
    """


@schema 
class MocapImputation(dj.Computed):
    """DISK-imputed motion capture results.
    
    Attributes
    ----------
    imputation_time : datetime
        When imputation was performed
    n_frames_total : int
        Total number of frames
    n_frames_imputed : int
        Number of frames with at least one imputed value
    n_values_imputed : int
        Total number of individual coordinate values imputed
    """
    definition = """
    -> MocapImputationTask
    ---
    imputation_time         : datetime
    n_frames_total          : int
    n_frames_imputed        : int
    n_values_imputed        : int
    """
    
    class TrackingPositionDISK(dj.Part):
        """DISK-imputed marker tracking positions.
        
        Attributes
        ----------
        frame_index : blob
            Frame indices
        timestamps : blob
            Time stamps for each frame
        x_pos, y_pos, z_pos : blob
            Imputed 3D positions
        uncertainty : blob
            DISK uncertainty estimate
        imputed_mask : blob
            Boolean array: True where values were imputed
        """
        definition = """
        -> master
        -> mocap.Mocap.TrackingId
        ---
        frame_index         : longblob
        timestamps          : longblob
        x_pos               : longblob
        y_pos               : longblob
        z_pos               : longblob
        uncertainty         : longblob      # DISK uncertainty estimate
        imputed_mask        : longblob      # True where values were imputed
        """
    
    class RigidBodyPositionDISK(dj.Part):
        """DISK-imputed rigid body positions.
        
        Note: Rigid body quaternions/euler angles may need separate handling
        since DISK is designed for marker positions, not rotations.
        
        Attributes
        ----------
        frame_index : blob
            Frame indices
        timestamps : blob
            Time stamps
        x_pos, y_pos, z_pos : blob
            Imputed 3D positions
        qw, qx, qy, qz : blob
            Quaternion components (may need special handling)
        roll, pitch, yaw : blob
            Euler angles derived from imputed quaternions
        uncertainty : blob
            DISK uncertainty estimate
        imputed_mask : blob
            Boolean array: True where values were imputed
        """
        definition = """
        -> master
        -> mocap.Mocap.TrackingId
        rigid_body          : varchar(64)
        ---
        frame_index         : longblob
        timestamps          : longblob
        x_pos               : longblob
        y_pos               : longblob
        z_pos               : longblob
        qw                  : longblob      # quaternion may need special handling
        qx                  : longblob
        qy                  : longblob
        qz                  : longblob
        roll                : longblob      # derived from imputed quaternions
        pitch               : longblob
        yaw                 : longblob
        uncertainty         : longblob
        imputed_mask        : longblob
        """
    
    def make(self, key):
        """Populate method: run DISK imputation on motion capture data."""
        
        task_mode, output_dir, disk_params = (MocapImputationTask & key).fetch1(
            'task_mode', 'output_dir', 'disk_params'
        )
        
        disk_model_name, model_path = (DISKModel & key).fetch1('disk_model_name', 'model_path')
        
        # Get original motion capture data
        mocap_key = {k: key[k] for k in mocap.MotionCapture.primary_key}
        markers_data = (mocap.MotionCapture.TrackingPosition & mocap_key).fetch(as_dict=True)
        rb_data = (mocap.MotionCapture.RigidBodyPosition & mocap_key).fetch(as_dict=True)
        
        if task_mode == 'trigger':
            # Prepare data for DISK
            data, markers, mask, timestamps = self._prepare_mocap_data(markers_data)
            
            # Run DISK imputation
            imputed_data, uncertainty, imputed_mask = self._run_disk_inference(
                data, mask, model_path
            )
            
            # Calculate statistics
            n_frames_total = data.shape[0]
            n_frames_imputed = np.sum(np.any(imputed_mask, axis=(1, 2)))
            n_values_imputed = np.sum(imputed_mask)
            
            # Insert main entry
            self.insert1({
                **key,
                'imputation_time': datetime.now(),
                'n_frames_total': n_frames_total,
                'n_frames_imputed': n_frames_imputed,
                'n_values_imputed': n_values_imputed,
            })
            
            # Insert tracking position part table entries
            for i, td in enumerate(markers_data):
                self.TrackingPositionDISK.insert1({
                    **key,
                    'tracking_id': td['tracking_id'],
                    'frame_index': np.arange(n_frames_total),
                    'timestamps': timestamps,
                    'x_pos': imputed_data[:, i, 0],
                    'y_pos': imputed_data[:, i, 1],
                    'z_pos': imputed_data[:, i, 2],
                    'uncertainty': uncertainty[:, i],
                    'imputed_mask': imputed_mask[:, i, 0] | imputed_mask[:, i, 1] | imputed_mask[:, i, 2],
                })
            
            # TODO: Handle rigid body data separately (quaternion imputation)
            
        elif task_mode == 'load':
            raise NotImplementedError("DISK load mode for mocap not yet implemented")
    
    @staticmethod
    def _prepare_mocap_data(tracking_data):
        """Convert motion capture tracking data to DISK-compatible format."""
        markers = [td['tracking_id'] for td in tracking_data]
        n_frames = len(tracking_data[0]['x_pos'])
        n_markers = len(markers)
        timestamps = tracking_data[0]['timestamps']
        
        data = np.zeros((n_frames, n_markers, 3))
        mask = np.ones((n_frames, n_markers), dtype=bool)
        
        for i, td in enumerate(tracking_data):
            x = np.array(td['x_pos'], dtype=float)
            y = np.array(td['y_pos'], dtype=float)
            z = np.array(td['z_pos'], dtype=float)
            
            data[:, i, 0] = x
            data[:, i, 1] = y
            data[:, i, 2] = z
            
            missing = np.isnan(x) | np.isnan(y) | np.isnan(z)
            mask[:, i] = ~missing
            data[missing, i, :] = np.nan
        
        return data, markers, mask, timestamps
    
    @staticmethod
    def _run_disk_inference(data, mask, model_path):
        """Run DISK imputation. Requires DISK to be installed."""
        raise NotImplementedError(
            "DISK inference requires DISK to be installed. "
            "Install with: pip install -e /path/to/DISK"
        )


# ============================================================================
# Utility Functions
# ============================================================================

def analyze_missing_data(pose_key, likelihood_threshold=0.5):
    """Analyze missing/low-confidence data in pose estimation results.
    
    Parameters
    ----------
    pose_key : dict
        DataJoint key for PoseEstimationNew entry
    likelihood_threshold : float
        DLC points below this likelihood are treated as missing
    
    Returns
    -------
    dict
        Analysis results with missing data statistics per body part
    """
    import pandas as pd
    
    body_parts = (model.PoseEstimationNew.BodyPartPosition & pose_key).fetch(as_dict=True)
    
    if not body_parts:
        return None
    
    analysis = {
        'body_part': [],
        'n_frames': [],
        'n_nan': [],
        'n_low_likelihood': [],
        'pct_missing': [],
        'likelihood_threshold': likelihood_threshold
    }
    
    for bp in body_parts:
        x = np.array(bp['x_pos'])
        y = np.array(bp['y_pos'])
        likelihood = np.array(bp['likelihood'])
        
        n_frames = len(x)
        n_nan = np.sum(np.isnan(x) | np.isnan(y))
        n_low_conf = np.sum(likelihood < likelihood_threshold)
        pct_missing = 100 * (n_nan + n_low_conf) / n_frames if n_frames > 0 else 0
        
        analysis['body_part'].append(bp['body_part'])
        analysis['n_frames'].append(n_frames)
        analysis['n_nan'].append(n_nan)
        analysis['n_low_likelihood'].append(n_low_conf)
        analysis['pct_missing'].append(pct_missing)
    
    return pd.DataFrame(analysis)
