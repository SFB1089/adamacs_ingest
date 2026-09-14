"""
Widefield SVD Ingestion Functions

Functions to ingest MATLAB-processed SVD results into WfieldSvd table.
Uses WORKING functions from tongest.py (list_mat, load_shifts) that handle both HDF5 and old MATLAB formats.

Usage:
    from adamacs.helpers.wfield_ingest import ingest_svd_from_matlab
    
    ingest_svd_from_matlab(
        scan_id='scan9FW84TYB',
        session_id='sess9FW84TYB',
        wfield_folder='/home/aeltona/wfield_data'
    )

Author: Aelton Araújo
Date: October 10, 2025
"""

import numpy as np
import os
from pathlib import Path
from datetime import datetime
from glob import glob


def pjoin(*args):
    """Path join helper"""
    return os.path.join(*args)


def list_mat(folder, key=None):
    """
    Load MATLAB .mat files - handles both HDF5 (v7.3) and older formats.
    
    COPIED FROM TONGEST.PY - THIS IS THE WORKING VERSION!
    
    Parameters
    ----------
    folder : str
        Folder containing .mat files
    key : str, optional
        Substring to match in filename (e.g., 'Vc' for '*Vc*.mat')
    
    Returns
    -------
    dict
        Dictionary with MATLAB variables
    """
    import scipy.io
    import h5py
    
    if key:
        mat_files = glob(pjoin(folder, f'*{key}*.mat'))
    else:
        mat_files = glob(pjoin(folder, '*.mat'))
    
    if len(mat_files) > 1:
        print("Multiple .mat files found:")
        for i, mat_file in enumerate(mat_files):
            print(f"{i + 1}. {os.path.basename(mat_file)}")
        
        # Auto-select most recent file (non-interactive)
        mat_files_sorted = sorted(mat_files, key=lambda p: os.path.getmtime(p), reverse=True)
        mat_path = mat_files_sorted[0]
        print(f"Auto-selected most recent: {os.path.basename(mat_path)}")
    elif len(mat_files) == 1:
        mat_path = mat_files[0]
        print("Mat file found")
        print(f"Loading... {mat_path}")
    else:
        raise FileNotFoundError(f"No .mat files found in {folder}")

    try:
        # Try HDF5 format first (MATLAB v7.3)
        with h5py.File(mat_path, 'r') as file:
            print("Selected", os.path.basename(mat_path), "(HDF5 format)")
            return {key: file[key][()] for key in file.keys()}
    except OSError:
        # Fallback to scipy.io.loadmat for older MATLAB versions
        try:
            mat_data = scipy.io.loadmat(mat_path)
            print("Selected", os.path.basename(mat_path))
            return mat_data
        except Exception as err:
            print("Failed to load .mat file.")
            print(err)
            return None


def ingest_svd_from_matlab(scan_id, session_id, wfield_folder,
                           processing_id=0,
                           normalization_method='global_mean',
                           svd_method='matlab_fsvd', hemo_correction_method='linear_regression_SVD',
                           curation_note='', manual_curation=False):
    """
    Ingest MATLAB-processed SVD results into Svd table.
    
    Parameters
    ----------
    scan_id : str
        Scan identifier (e.g., 'scan9FW84TYB')
    session_id : str
        Session identifier (e.g., 'sess9FW84TYB')
    wfield_folder : str, optional
        Full path to the scan's wfield output folder.
        If None, reads from dj.config and searches for scan_id.
    processing_id : int
        ProcessingTask ID (default 0)
    normalization_method : str, default='global_mean'
    svd_method : str, default='matlab_fsvd'
    hemo_correction_method : str, default='linear_regression_SVD'
    curation_note : str, optional
    manual_curation : bool, default=False
        
    Returns
    -------
    dict
        The inserted key
    """
    from adamacs.pipeline import wfield
    from adamacs.helpers.wfield_helpers import check_longblob_safe

    print(f"Loading SVD data from: {wfield_folder}")
    
    SM_data = list_mat(wfield_folder, 'Vc')
    
    if SM_data is None:
        raise IOError(f"Failed to load SVD data from {wfield_folder}")
    
    try:
        U_raw = SM_data['U']
        newVc_raw = SM_data['newVc']
        
        # Transpose to match expected shapes: U (H,W,k), SVT (k,T)
        U = U_raw.T
        SVT = newVc_raw.T
        
        blueV = SM_data['blueV']
        hemoV = SM_data['hemoV']
        
    except KeyError as e:
        raise KeyError(f"Missing expected array in SVD output: {e}\nAvailable keys: {list(SM_data.keys())}")
    
    frame_height, frame_width, n_components = U.shape
    n_components_check, n_frames = SVT.shape
    
    assert n_components == n_components_check, f"Component mismatch: U has {n_components}, SVT has {n_components_check}"
    
    print(f"  SVD dimensions: {frame_height}x{frame_width}, {n_components} components, {n_frames} frames")
    print(f"  U: {U.nbytes / 1e6:.1f} MB, SVT: {SVT.nbytes / 1e6:.1f} MB")

    # Extract variance explained from MATLAB output
    variance_explained_arr = SM_data['variance_explained_per_component']
    hemo_var_global = float(SM_data['hemoVar_global'])
    check_longblob_safe(variance_explained_arr, 'variance_explained')
    
    svd_key = {
        'scan_id': scan_id,
        'session_id': session_id,
        'processing_id': processing_id,
        'n_components': int(n_components),
        'n_frames': int(n_frames),
        'frame_height': int(frame_height),
        'frame_width': int(frame_width),
        'svd_method': svd_method,
        'hemo_correction_method': hemo_correction_method,
        'normalization_method': normalization_method,
        'svd_output_folder': wfield_folder,
        'processing_timestamp': datetime.now(),
        'ingestion_timestamp': datetime.now(),
        'curation_time': datetime.now(),
        'manual_curation': manual_curation,
        'curation_note': curation_note,
        'variance_explained': variance_explained_arr,
        'hemo_variance_explained': hemo_var_global,
        'u': U,
        'new_vc': SVT,
        'blue_v': blueV,
        'hemo_v': hemoV,
    }
    
    print(f"Inserting into Svd table...")
    try:
        wfield.Svd.insert1(svd_key)
        print(f"  Successfully ingested: {scan_id}, processing_id={processing_id}")
        return svd_key
    except Exception as e:
        print(f"  Ingestion failed: {e}")
        raise


def ingest_svd_from_python(svd_result, scan_id, session_id,
                           svd_output_folder,
                           processing_id=0,
                           normalization_method='global_mean',
                           svd_method='python_fsvd',
                           hemo_correction_method='linear_regression_SVD',
                           curation_note='', manual_curation=False):
    """
    Ingest Python-processed SVD results directly from run_svd_python() output dict
    into the Svd table.
    
    Parameters
    ----------
    svd_result : dict
        Return value from wfield_helpers.run_svd_python() containing:
        'U', 'newVc', 'blueV', 'hemoV', 'variance_explained_percent',
        'hemoVar_global', 'variance_explained_per_component'
    scan_id : str
    session_id : str
    svd_output_folder : str
    processing_id : int
        ProcessingTask ID
    normalization_method : str
    svd_method : str
    hemo_correction_method : str
    curation_note : str
    manual_curation : bool
    
    Returns
    -------
    dict : The inserted key
    """
    from adamacs.pipeline import wfield
    from adamacs.helpers.wfield_helpers import check_longblob_safe

    U = svd_result['U']          # (H, W, k)
    newVc = svd_result['newVc']  # (k, T)
    blueV = svd_result['blueV'] # (k, T)
    hemoV = svd_result['hemoV'] # (k, T)
    
    frame_height, frame_width, n_components = U.shape
    n_components_check, n_frames = newVc.shape
    
    assert n_components == n_components_check, \
        f"Component mismatch: U has {n_components}, newVc has {n_components_check}"
    
    print(f"  Python SVD: {frame_height}x{frame_width}, {n_components} components, {n_frames} frames")
    print(f"  U: {U.nbytes / 1e6:.1f} MB, newVc: {newVc.nbytes / 1e6:.1f} MB")

    # Extract variance explained — direct access, no defaults
    variance_explained_arr = svd_result['variance_explained_per_component']
    hemo_var_global = float(svd_result['hemoVar_global'])
    check_longblob_safe(variance_explained_arr, 'variance_explained')
    
    svd_key = {
        'scan_id': scan_id,
        'session_id': session_id,
        'processing_id': processing_id,
        'n_components': int(n_components),
        'n_frames': int(n_frames),
        'frame_height': int(frame_height),
        'frame_width': int(frame_width),
        'svd_method': svd_method,
        'hemo_correction_method': hemo_correction_method,
        'normalization_method': normalization_method,
        'svd_output_folder': svd_output_folder,
        'processing_timestamp': datetime.now(),
        'ingestion_timestamp': datetime.now(),
        'curation_time': datetime.now(),
        'manual_curation': manual_curation,
        'curation_note': curation_note,
        'variance_explained': variance_explained_arr,
        'hemo_variance_explained': hemo_var_global,
        'u': U,
        'new_vc': newVc,
        'blue_v': blueV,
        'hemo_v': hemoV,
    }
    
    print(f"Inserting into Svd table...")
    try:
        wfield.Svd.insert1(svd_key)
        print(f"  Successfully ingested: {scan_id}, processing_id={processing_id}")
        return svd_key
    except Exception as e:
        print(f"  Ingestion failed: {e}")
        raise


def verify_svd_ingestion(scan_id, session_id, processing_id=0):
    """
    Verify that SVD data was correctly ingested.
    
    Parameters
    ----------
    scan_id : str
    session_id : str
    processing_id : int, default=0
        
    Returns
    -------
    dict
        Fetched data for verification
    """
    from adamacs.pipeline import wfield
    
    key = {
        'scan_id': scan_id,
        'session_id': session_id,
        'processing_id': processing_id
    }
    
    # Check if entry exists
    query = wfield.Svd & key
    
    if len(query) == 0:
        print(f"No SVD data found for {scan_id}, processing_id={processing_id}")
        return None
    
    # Fetch data
    svd_data = query.fetch1()
    
    print(f"SVD data found for {scan_id}, processing_id={processing_id}")
    print(f"   Components: {svd_data['n_components']}")
    print(f"   Frames: {svd_data['n_frames']}")
    print(f"   Frame size: {svd_data['frame_height']} x {svd_data['frame_width']}")
    print(f"   Method: {svd_data['svd_method']}")
    print(f"   Hemodynamic correction: {svd_data['hemo_correction_method']}")
    print(f"   Normalization: {svd_data['normalization_method']}")
    print(f"   Processing time: {svd_data['processing_timestamp']}")
    print(f"   U shape: {svd_data['u'].shape}")
    print(f"   new_vc shape: {svd_data['new_vc'].shape}")
    print(f"   Variance explained shape: {svd_data['variance_explained'].shape}")
    print(f"   Hemo variance explained: {svd_data['hemo_variance_explained']:.2f}%")
    
    return svd_data


def list_all_processing_runs(scan_id):
    """List all processing runs for a scan."""
    from adamacs.pipeline import wfield
    
    runs = (wfield.Svd & {'scan_id': scan_id}).fetch(
        'processing_id', 'n_components', 'normalization_method', 
        'manual_curation', 'curation_note', 
        order_by='processing_id'
    )
    
    if len(runs[0]) == 0:
        print(f"No SVD data found for {scan_id}")
        return
    
    print(f"Processing runs for {scan_id}:")
    for i in range(len(runs[0])):
        pid = runs[0][i]
        n_comp = runs[1][i]
        method = runs[2][i]
        manual = runs[3][i]
        note = runs[4][i]
        
        flag = "manual" if manual else "auto"
        print(f"  [{flag}] processing_id={pid}: {n_comp} components, {method}")
        if note:
            print(f"     Note: {note}")
