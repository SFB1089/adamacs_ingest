"""
Widefield SVD Processing and Ingestion Helpers

This module provides helper functions for:
1. Detecting imaging files (.tif/.dat) in scan folders
2. Checking wfield_data folder structure
3. Loading and validating SVD output files
4. Ingesting SVD results into WFieldSVD schema
5. Python-based SVD processing (when MATLAB outputs don't exist)

Author: Aelton Araújo (with GitHub Copilot)
Date: October 28, 2025
"""

import os
import json
from pathlib import Path
from glob import glob
from typing import Tuple, Optional, Dict, List
import pandas as pd
import json
import os
from functools import partial
from skimage.transform import SimilarityTransform
import warnings
import numpy as np
from scipy.io import loadmat
import datajoint as dj
from sklearn.decomposition import IncrementalPCA
from scipy.sparse import load_npz, issparse,csr_matrix
from adamacs.pipeline import subject, session, scan, wfield


# ==============================================================================
# LONGBLOB SAFETY CHECK
# ==============================================================================

def check_longblob_safe(array, field_name, threshold=0.8):
    """
    Check if a numpy array is safe to insert as a MySQL longblob.
    
    Queries @@max_allowed_packet and compares against the estimated
    serialized size of the array. Raises RuntimeError if the array
    exceeds threshold * max_allowed_packet.
    
    Args:
        array: numpy ndarray to check
        field_name: Name of the field (for error messages)
        threshold: Fraction of max_allowed_packet to treat as limit (default 0.8)
    
    Returns:
        True if safe
    
    Raises:
        RuntimeError: If array would exceed the safety threshold
    """
    import datajoint as dj
    # Estimate serialized size (raw bytes + small overhead)
    estimated_bytes = array.nbytes + 256

    # Query MySQL max_allowed_packet
    
    max_packet = dj.conn().query("SELECT @@max_allowed_packet").fetchone()[0]
    # except Exception:
    #     # If we can't query, assume conservative 64MB default
    #     max_packet = 64 * 1024 * 1024

    safe_limit = int(max_packet * threshold)

    if estimated_bytes > safe_limit:
        raise RuntimeError(
            f"Array '{field_name}' ({estimated_bytes / 1e6:.1f} MB) exceeds "
            f"{threshold*100:.0f}% of max_allowed_packet "
            f"({max_packet / 1e6:.1f} MB, limit={safe_limit / 1e6:.1f} MB). "
            f"Shape: {array.shape}, dtype: {array.dtype}. "
            f"Increase max_allowed_packet or use external storage."
        )

    return True


# ==============================================================================
# FILE DISCOVERY HELPERS
# ==============================================================================
def pjoin(*args):
    return os.path.join(*args)

def find_imaging_file(scan_folder):
    """
    Find the imaging file (.dat or .tif) in the scan folder.
    
    Args:
        scan_folder: Path to scan folder (str or Path)
    
    Returns:
        tuple: (file_path, file_type) or (None, None) if not found
    
    Examples:
        >>> imaging_file, file_type = find_imaging_file('/path/to/scan')
        >>> if imaging_file:
        ...     print(f"Found {file_type} file: {imaging_file}")
    """
    from pathlib import Path
    from glob import glob
    
    scan_folder = Path(scan_folder)
    
    # First, try to find .dat file
    dat_files = list(scan_folder.glob('*.dat'))
    if dat_files:
        return dat_files[0], 'dat'
    
    # If no .dat, look for .tif or .tiff
    tif_files = list(scan_folder.glob('*.tif'))
    tif_files.extend(list(scan_folder.glob('*.tiff')))
    if tif_files:
        return tif_files[0], 'tif'
    
    return None, None


# ==============================================================================
# MATLAB SCRIPT EXECUTION
# ==============================================================================

# ==============================================================================
# MOTION CORRECTION / REGISTRATION
# ==============================================================================

def compute_registration(imaging_file, output_dir, scan_id, proc_id='',
                         method='phaseCorrelate', apply=False):
    """
    Compute motion correction registration shifts for blue AND violet channels
    separately using Python (cv2.phaseCorrelate).
    
    Each channel gets its own reference frame (median of subsampled frames).
    Shifts are computed per-channel and stored in a single .npz file.
    
    When apply=True, registered data is written to a memmap temp file under
    the svds folder so downstream SVD can use it directly.
    
    Args:
        imaging_file: Path to imaging file (.dat or .tif)
        output_dir: Directory to save shifts file (typically session_dir/svds)
        scan_id: Scan identifier
        proc_id: Processing ID (alphanumeric from generate_proc_id())
        method: Registration method (default: 'phaseCorrelate')
        apply: If True, apply shifts to data and save registered memmap
    
    Returns:
        dict with keys:
            - shifts_file: Path to saved shifts .npz
            - shifts_blue: ndarray (n_per_channel, 2) [dy, dx] for blue
            - shifts_violet: ndarray (n_per_channel, 2) [dy, dx] for violet
            - reference_frame_blue: ndarray (H, W)
            - reference_frame_violet: ndarray (H, W)
            - n_frames: Total interleaved frame count
            - mean_shift_x_blue, mean_shift_y_blue, max_shift_x_blue, max_shift_y_blue
            - mean_shift_x_violet, mean_shift_y_violet, max_shift_x_violet, max_shift_y_violet
            - processing_time_sec: Time elapsed
    """
    import numpy as np
    import cv2
    from tqdm import tqdm
    import time
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine expected output filename
    if proc_id:
        shifts_filename = f"{scan_id}_{proc_id}_shifts.npz"
    else:
        shifts_filename = f"{scan_id}_shifts.npz"
    shifts_file = output_dir / shifts_filename
    
    # Check if registration already exists
    if shifts_file.exists():
        print(f"Registration already exists: {shifts_file}")
        print(f"   Loading existing registration data...")
        try:
            sd = np.load(shifts_file)
            result = {
                'shifts_file': str(shifts_file),
                'shifts_blue': sd['shifts_blue'],
                'shifts_violet': sd['shifts_violet'],
                'reference_frame_blue': sd['reference_frame_blue'],
                'reference_frame_violet': sd['reference_frame_violet'],
                'n_frames': int(sd['n_frames']),
                'mean_shift_x_blue': float(sd['mean_shift_x_blue']),
                'mean_shift_y_blue': float(sd['mean_shift_y_blue']),
                'max_shift_x_blue': float(sd['max_shift_x_blue']),
                'max_shift_y_blue': float(sd['max_shift_y_blue']),
                'mean_shift_x_violet': float(sd['mean_shift_x_violet']),
                'mean_shift_y_violet': float(sd['mean_shift_y_violet']),
                'max_shift_x_violet': float(sd['max_shift_x_violet']),
                'max_shift_y_violet': float(sd['max_shift_y_violet']),
                'processing_time_sec': float(sd['processing_time_sec']),
                'method': str(sd['method']),
            }
            print(f"   Loaded {result['n_frames']} total frames")
            return result
        except Exception as e:
            print(f"Failed to load existing registration file: {e}")
            print(f"   Recomputing registration...")
    
    print(f"Starting motion correction for {scan_id}...")
    start_time = time.time()
    
    # --- Load imaging data as memmap ---
    file_type = 'dat' if imaging_file.endswith('.dat') else 'tif'
    
    if file_type == 'dat':
        import json as _json
        meta_path = glob(pjoin(Path(imaging_file).parent, '*meta.json*'))[0]
        with open(meta_path) as f:
            meta = _json.load(f)
        shape = (meta['height'], meta['width'], meta['n_channels'], meta['n_frames'])
        raw = np.memmap(imaging_file, dtype='uint16', mode='r', shape=shape, order='F')
        # Allocate (allFrames, H, W) memmap and fill frame-by-frame
        # instead of np.transpose() which copies the entire file into RAM
        _tmp_transposed = output_dir / f'_tmp_{scan_id}_transposed.dat'
        dat_array = np.memmap(str(_tmp_transposed), dtype='uint16', mode='w+',
                              shape=(meta['n_frames'], meta['height'], meta['width']))
        for _i in tqdm(range(meta['n_frames']), desc='Transposing dat', unit='frame'):
            dat_array[_i] = raw[:, :, 0, _i]
        dat_array.flush()
        del raw
    else:
        from tifffile import TiffFile
        print(f"  Loading TIFF file: {imaging_file}")
        with TiffFile(imaging_file) as tif:
            dat_array = tif.asarray(out="memmap")
        _tmp_transposed = None

    n_frames = dat_array.shape[0]
    height, width = dat_array.shape[1], dat_array.shape[2]
    n_per_channel = n_frames // 2

    # De-interleave indices: violet=0,2,4,... blue=1,3,5,...
    violet_frames = dat_array[0::2]
    blue_frames = dat_array[1::2]
    print(f"  Loaded {n_frames} frames ({n_per_channel} per channel), {height}x{width}")

    # --- Compute separate reference frames ---
    print("  Computing reference frames (median of subsampled frames)...")
    n_subsample = min(1000, n_per_channel)
    sub_idx = np.linspace(0, n_per_channel - 1, n_subsample, dtype=int)
    
    ref_blue = np.median(blue_frames[sub_idx].astype(np.float32), axis=0).astype(np.float32)
    ref_violet = np.median(violet_frames[sub_idx].astype(np.float32), axis=0).astype(np.float32)
    del blue_frames, violet_frames  # Free views
    
    # --- Compute shifts per channel ---
    print(f"  Computing registration shifts using {method} (blue + violet separately)...")
    shifts_blue = []
    shifts_violet = []
    
    for i in tqdm(range(n_per_channel), desc="Computing shifts", unit="pair"):
        # Violet frame (even index)
        vframe = dat_array[2 * i].astype(np.float32)
        vblurred = cv2.GaussianBlur(vframe, ksize=(3, 3), sigmaX=1, borderType=cv2.BORDER_REFLECT)
        vshift, _ = cv2.phaseCorrelate(ref_violet, vblurred)
        shifts_violet.append([vshift[1], vshift[0]])  # [dy, dx]
        
        # Blue frame (odd index)
        bframe = dat_array[2 * i + 1].astype(np.float32)
        bblurred = cv2.GaussianBlur(bframe, ksize=(3, 3), sigmaX=1, borderType=cv2.BORDER_REFLECT)
        bshift, _ = cv2.phaseCorrelate(ref_blue, bblurred)
        shifts_blue.append([bshift[1], bshift[0]])  # [dy, dx]
    
    shifts_blue = np.array(shifts_blue, dtype=np.float32)    # (n_per_channel, 2)
    shifts_violet = np.array(shifts_violet, dtype=np.float32)  # (n_per_channel, 2)
    
    elapsed_time = time.time() - start_time
    print(f"  Registration complete in {elapsed_time:.1f}s ({n_frames / elapsed_time:.1f} fps)")
    
    # --- Compute per-channel statistics ---
    stats = {}
    for name, arr in [('blue', shifts_blue), ('violet', shifts_violet)]:
        stats[f'mean_shift_y_{name}'] = float(np.mean(arr[:, 0]))
        stats[f'mean_shift_x_{name}'] = float(np.mean(arr[:, 1]))
        stats[f'max_shift_y_{name}'] = float(np.max(np.abs(arr[:, 0])))
        stats[f'max_shift_x_{name}'] = float(np.max(np.abs(arr[:, 1])))
    
    print(f"  Blue  — mean: ({stats['mean_shift_x_blue']:.2f}, {stats['mean_shift_y_blue']:.2f}), "
          f"max: ({stats['max_shift_x_blue']:.2f}, {stats['max_shift_y_blue']:.2f})")
    print(f"  Violet — mean: ({stats['mean_shift_x_violet']:.2f}, {stats['mean_shift_y_violet']:.2f}), "
          f"max: ({stats['max_shift_x_violet']:.2f}, {stats['max_shift_y_violet']:.2f})")
    
    # --- Save shifts ---
    save_dict = dict(
        shifts_blue=shifts_blue,
        shifts_violet=shifts_violet,
        reference_frame_blue=ref_blue,
        reference_frame_violet=ref_violet,
        scan_id=scan_id,
        proc_id=proc_id,
        n_frames=n_frames,
        method=method,
        processing_time_sec=elapsed_time,
        **stats,
    )
    np.savez(shifts_file, **save_dict)
    print(f"  Saved shifts to: {shifts_file}")
    
    # Also save MATLAB-compatible .mat
    from scipy.io import savemat
    mat_file = output_dir / shifts_filename.replace('.npz', '.mat')
    # Interleave shifts back for MATLAB compatibility (allFrames order)
    all_shifts = np.empty((n_frames, 2), dtype=np.float32)
    all_shifts[0::2] = shifts_violet
    all_shifts[1::2] = shifts_blue
    savemat(str(mat_file), {
        'row_shift': all_shifts[:, 0],
        'col_shift': all_shifts[:, 1],
        'diffphase': np.zeros(n_frames, dtype=np.float32),
        'blueMean': ref_blue,
        'violetMean': ref_violet,
        'scan_id': scan_id,
        'n_frames': n_frames,
    })
    print(f"  Saved MATLAB-compatible: {mat_file}")
    
    # --- Optionally apply shifts to data ---
    registered_file = None
    if apply:
        print(f"  Applying Fourier shifts to data (memmap)...")
        t_apply = time.time()
        reg_file_path = output_dir / f"{scan_id}_{proc_id}_registered.dat"
        registered = np.memmap(
            str(reg_file_path), dtype='float32', mode='w+',
            shape=(n_frames, height, width),
        )
        for i in tqdm(range(n_per_channel), desc="Applying shifts", unit="pair"):
            # Violet
            vf = dat_array[2 * i].astype(np.float32)
            dy_v, dx_v = shifts_violet[i]
            registered[2 * i] = _apply_fourier_shift(vf, dy_v, dx_v)
            # Blue
            bf = dat_array[2 * i + 1].astype(np.float32)
            dy_b, dx_b = shifts_blue[i]
            registered[2 * i + 1] = _apply_fourier_shift(bf, dy_b, dx_b)
        registered.flush()
        registered_file = str(reg_file_path)
        print(f"  Applied shifts in {time.time() - t_apply:.1f}s -> {reg_file_path}")
    
    result = {
        'shifts_file': str(shifts_file),
        'shifts_blue': shifts_blue,
        'shifts_violet': shifts_violet,
        'reference_frame_blue': ref_blue,
        'reference_frame_violet': ref_violet,
        'n_frames': n_frames,
        'processing_time_sec': elapsed_time,
        'method': method,
        **stats,
    }
    if registered_file:
        result['registered_file'] = registered_file
    # Return transposed file path so the SVD step can reuse it directly
    # instead of re-loading and re-transposing the raw Fortran-order .dat
    if _tmp_transposed is not None:
        result['transposed_file'] = str(_tmp_transposed)

    return result


def ingest_registration(key, registration_result):
    """
    Ingest registration results into Registration table (separate blue/violet).
    
    Stores actual per-frame shift arrays as longblobs plus reference frames.
    
    Args:
        key: Full primary key dict from Processing.make() — must contain
             session_id, scan_id, processing_id
        registration_result: Dict returned by compute_registration()
    
    Raises:
        RuntimeError: If insert fails
    """
    from ..schemas import wfield
    from datetime import datetime
    
    # Safety check on all longblob arrays
    shifts_blue = registration_result['shifts_blue']    # (n_per_channel, 2)
    shifts_violet = registration_result['shifts_violet']  # (n_per_channel, 2)
    ref_blue = registration_result['reference_frame_blue']
    ref_violet = registration_result['reference_frame_violet']
    
    check_longblob_safe(shifts_blue, 'shifts_blue')
    check_longblob_safe(shifts_violet, 'shifts_violet')
    check_longblob_safe(ref_blue, 'reference_frame_blue')
    check_longblob_safe(ref_violet, 'reference_frame_violet')
    
    reg_key = {
        # Primary key fields — taken directly from Processing key (no secondary queries)
        'session_id': key['session_id'],
        'scan_id': key['scan_id'],
        'processing_id': key['processing_id'],
        'shifts_file_path': registration_result['shifts_file'],
        'n_frames_registered': registration_result['n_frames'],
        # Actual per-frame shift arrays
        'shifts_x_blue': shifts_blue[:, 1],       # column = x
        'shifts_y_blue': shifts_blue[:, 0],       # row = y
        'shifts_x_violet': shifts_violet[:, 1],   # column = x
        'shifts_y_violet': shifts_violet[:, 0],   # row = y
        # Reference frames
        'reference_frame_blue': ref_blue,
        'reference_frame_violet': ref_violet,
        'registration_method': registration_result['method'],
        'processing_time_sec': registration_result['processing_time_sec'],
        'registration_timestamp': datetime.now(),
    }
    
    wfield.Registration.insert1(reg_key, skip_duplicates=True)
    print(f"  Registration ingested (session_id={key['session_id']}, scan_id={key['scan_id']}, processing_id={key['processing_id']})")


def run_matlab_script_realtime(script_path, matlab_vars=None):
    """
    Execute a MATLAB script with real-time output streaming.
    
    This function runs a MATLAB script in batch mode (-batch flag) and streams
    the output in real-time so you can monitor progress.
    
    Args:
        script_path: Path to the MATLAB .m script file
        matlab_vars: Dictionary of variable name -> value pairs to pass to MATLAB
                    Strings are automatically quoted, bools converted to true/false
    
    Returns:
        None (raises on failure)
    
    Examples:
        >>> run_matlab_script_realtime(
        ...     '/path/to/run_SVD.m',
        ...     {'fullFilePath': '/data/scan.tif', 'dimCnt': 500}
        ... )
    
    Notes:
        - MATLAB must be installed and accessible at /home/aeltona/MATLAB/R2023a/bin/matlab
        - Script output is printed to console in real-time
        - Raises RuntimeError if MATLAB exits with non-zero code
    """
    import subprocess

    matlab_bin = "/home/aeltona/MATLAB/R2023a/bin/matlab"
    
    # Convert Python variables into MATLAB variable assignments
    if matlab_vars is None:
        matlab_vars = {}

    matlab_var_defs = []
    for var, val in matlab_vars.items():
        if isinstance(val, str):
            val_str = f"'{val}'"  # wrap strings in single quotes
        elif isinstance(val, bool):
            val_str = 'true' if val else 'false'
        else:
            val_str = str(val)
        matlab_var_defs.append(f"{var} = {val_str};")
    
    var_string = ' '.join(matlab_var_defs)
    run_string = f"{var_string} addpath('/home/aeltona/adamacs/adamacs/helpers/matlab'); run('{script_path}')"

    command = [matlab_bin, "-batch", run_string]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    print("=== MATLAB LOG START ===")
    for line in process.stdout:
        print(line, end='')  # stream output line-by-line

    process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"MATLAB exited with error code: {return_code}")
    else:
        print("\n=== MATLAB SCRIPT COMPLETED SUCCESSFULLY ===")


# ==============================================================================
# TODO 1: IMAGING FILE DETECTION
# ==============================================================================

def check_imaging_exists(scan_folder: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if imaging file (.tif or .dat) exists in scan folder.
    
    This is the unified detection function that gates all imaging ingestion.
    Uses existing wf.find_imaging_file() as base but adds validation.
    
    Args:
        scan_folder: Path to scan folder (can be string or Path object)
    
    Returns:
        Tuple of:
            has_imaging (bool): True if imaging file found
            file_path (str or None): Full path to imaging file
            file_type (str or None): 'tif' or 'dat' or None
    
    Examples:
        >>> has_img, path, ftype = check_imaging_exists('/path/to/scan/')
        >>> if has_img:
        ...     print(f"Found {ftype} file: {path}")
        >>> else:
        ...     print("Behavior-only experiment")
    """
    scan_folder = Path(scan_folder)
    
    if not scan_folder.exists():
        warnings.warn(f"Scan folder does not exist: {scan_folder}")
        return False, None, None
    
    # Use find_imaging_file() for detection
    file_path, file_type = find_imaging_file(str(scan_folder))
    
    if file_path is None:
        # No imaging file found - this is a behavior-only experiment
        return False, None, None
    
    # Validate file exists and is readable
    file_path = Path(file_path)
    if not file_path.exists():
        warnings.warn(f"Imaging file detected but not accessible: {file_path}")
        return False, None, None
    
    # Check file size (basic sanity check)
    file_size_gb = file_path.stat().st_size / (1024**3)
    if file_size_gb < 0.001:  # Less than 1 MB
        warnings.warn(f"Imaging file suspiciously small: {file_size_gb:.4f} GB")
    
    return True, str(file_path), file_type


# ==============================================================================
# TODO 2: WFIELD_DATA FOLDER CHECKING
# ==============================================================================

def check_wfield_folder(scan_id: str, wfield_root: Optional[str] = None) -> Dict[str, any]:
    """
    Check if wfield_data/{scan_id}/ folder exists and inventory its contents.
    
    This function checks for required SVD output files without blocking ingestion
    if they don't exist. It provides a status report for downstream decisions.
    
    Args:
        scan_id: Scan identifier (e.g., 'scan9FW5OVL6')
        wfield_root: Root directory for wfield data. If None, reads from DJ config.
    
    Returns:
        Dictionary with status information:
        {
            'folder_exists': bool,
            'folder_path': Path or None,
            'has_svd_metadata': bool,
            'has_vc_components': bool,
            'has_brain_mask': bool,
            'has_ccf_transforms': bool,
            'has_vessels_transform': bool,
            'missing_files': List[str],  # Names of expected but missing files
            'extra_files': List[str],    # Unexpected files found
            'ready_for_ingestion': bool  # All required files present
        }
    
    Examples:
        >>> status = check_wfield_folder('scan9FW5OVL6')
        >>> if status['ready_for_ingestion']:
        ...     ingest_wfield(scan_id, status['folder_path'])
        >>> else:
        ...     print(f"Missing files: {status['missing_files']}")
    """
    # Get wfield root from config or use provided
    if wfield_root is None:
        try:
            wfield_root = dj.config['custom'].get('wfield_root_data_dir')
            if not wfield_root:
                wfield_root = dj.config['custom'].get('wfield_processing_dir', '/mnt/data/aeltona/wfield_data')
        except:
            wfield_root = '/mnt/data/aeltona/wfield_data'
    
    wfield_root = Path(wfield_root)
    folder_path = wfield_root / scan_id
    
    # Initialize status dictionary
    status = {
        'folder_exists': False,
        'folder_path': None,
        'has_svd_metadata': False,
        'has_vc_components': False,
        'has_brain_mask': False,
        'has_ccf_transforms': False,
        'has_vessels_transform': False,
        'missing_files': [],
        'found_files': [],  # Track which files were actually found
        'extra_files': [],
        'ready_for_ingestion': False
    }
    
    if not folder_path.exists():
        status['missing_files'] = ['entire wfield_data folder']
        return status
    
    status['folder_exists'] = True
    status['folder_path'] = folder_path
    
    # Define expected files (common patterns from your notebooks)
    expected_files = {
        'svd_metadata': [
            f'{scan_id}_SVD_meta.mat',
            f'{scan_id}_SVD_metadata.mat',
            '*_SVD_meta.mat'  # Wildcard pattern
        ],
        'vc_components': [
            f'{scan_id}_Vc_hemocorrected.mat',
            f'{scan_id}_Vc.mat',
            f'Vc_{scan_id}.mat',
            '*_Vc*.mat'  # Wildcard pattern
        ],
        'brain_mask': [
            f'{scan_id}_brain_mask.npy',
            f'{scan_id}_mask.npy',
            '*_brain_mask.npy'
        ],
        'ccf_transforms': [
            f'{scan_id}_transform_landmarks.json',
            f'{scan_id}_ccf_landmarks.json',
            '*_transform_landmarks.json'
        ],
        'vessels_transform': [
            f'{scan_id}_vessels_transform.mat',
            f'{scan_id}_vessels.mat',
            '*_vessels_transform.mat'
        ]
    }
    
    # Check each file category
    for category, patterns in expected_files.items():
        found = False
        for pattern in patterns:
            matches = list(folder_path.glob(pattern))
            if matches:
                status[f'has_{category}'] = True
                found = True
                # Track the actual file that was found
                status['found_files'].append(matches[0].name)
                break
        
        if not found:
            status['missing_files'].append(category)
    
    # List all files in folder for debugging
    all_files = [f.name for f in folder_path.iterdir() if f.is_file()]
    
    # Identify unexpected files (not matching any expected pattern)
    expected_patterns = [p for patterns in expected_files.values() for p in patterns]
    for file in all_files:
        matches_expected = any(
            file == pattern or 
            (('*' in pattern) and pattern.replace('*', '') in file)
            for pattern in expected_patterns
        )
        if not matches_expected:
            status['extra_files'].append(file)
    
    # Determine if ready for ingestion (minimum required: SVD metadata + Vc)
    status['ready_for_ingestion'] = (
        status['has_svd_metadata'] and 
        status['has_vc_components']
    )
    
    return status


# ==============================================================================
# TODO 3: SVD FILE LOADING AND VALIDATION
# ==============================================================================

def load_svd_outputs(folder_path: Path, scan_id: str) -> Dict[str, any]:
    """
    Load SVD output files from wfield_data folder with validation.
    
    This function loads all available SVD-related files and performs
    basic consistency checks (dimensions, data types, etc.).
    
    Args:
        folder_path: Path to wfield_data/{scan_id}/ folder
        scan_id: Scan identifier for filename matching
    
    Returns:
        Dictionary containing:
        {
            'U': np.ndarray or None,          # Spatial components (H, W, n_comp)
            'Vc': np.ndarray or None,         # Temporal components (n_comp, n_frames)
            'metadata': dict,                  # SVD metadata
            'brain_mask': np.ndarray or None,  # Brain mask (H, W)
            'ccf_landmarks': dict or None,     # CCF transform landmarks
            'vessels': np.ndarray or None,     # Vessel image/transform
            'errors': List[str],               # Any loading errors encountered
            'warnings': List[str]              # Any warnings
        }
    
    Raises:
        FileNotFoundError: If critical files are missing
        ValueError: If data dimensions are inconsistent
    """
    result = {
        'U': None,
        'Vc': None,
        'metadata': {},
        'brain_mask': None,
        'ccf_landmarks': None,
        'vessels': None,
        'errors': [],
        'warnings': []
    }
    
    folder_path = Path(folder_path)
    
    # === 1. LOAD SVD METADATA ===
    metadata_patterns = [
        f'{scan_id}_SVD_meta.mat',
        f'{scan_id}_SVD_metadata.mat',
        '*_SVD_meta.mat'
    ]
    
    metadata_file = None
    for pattern in metadata_patterns:
        matches = list(folder_path.glob(pattern))
        if matches:
            metadata_file = matches[0]
            break
    
    if metadata_file is None:
        result['errors'].append("SVD metadata file not found")
    else:
        try:
            mat_data = loadmat(str(metadata_file), simplify_cells=True)
            result['metadata'] = {
                'n_components': int(mat_data.get('n_components', 200)),
                'variance_explained': float(mat_data.get('hemoVar_global', 0)),
                'variance_per_pixel': mat_data.get('hemoVar_per_pixel', None),
                'sampling_rate': float(mat_data.get('sample_rate', 15.13)),
                'raw_data': mat_data  # Keep raw data for debugging
            }
        except Exception as e:
            result['errors'].append(f"Failed to load SVD metadata: {e}")
    
    # === 2. LOAD Vc COMPONENTS ===
    vc_patterns = [
        f'{scan_id}_Vc_hemocorrected.mat',
        f'{scan_id}_Vc.mat',
        f'Vc_{scan_id}.mat',
        '*_Vc*.mat'
    ]
    
    vc_file = None
    for pattern in vc_patterns:
        matches = list(folder_path.glob(pattern))
        if matches:
            vc_file = matches[0]
            break
    
    if vc_file is None:
        result['errors'].append("Vc components file not found")
    else:
        try:
            mat_data = loadmat(str(vc_file), simplify_cells=True)
            
            # Try common variable names from your MATLAB code
            vc_var_names = ['newVc', 'Vc', 'vc', 'V', 'temporal_components']
            Vc = None
            for var_name in vc_var_names:
                if var_name in mat_data:
                    Vc = mat_data[var_name]
                    break
            
            if Vc is None:
                result['errors'].append(f"Vc variable not found in {vc_file.name}")
                result['warnings'].append(f"Available variables: {list(mat_data.keys())}")
            else:
                # Validate and transpose if needed (MATLAB often saves as frames x components)
                if Vc.ndim != 2:
                    result['errors'].append(f"Vc must be 2D, got shape {Vc.shape}")
                else:
                    # Check if transposition needed
                    if Vc.shape[0] > Vc.shape[1]:
                        # Likely (n_frames, n_components) - need to transpose
                        result['warnings'].append(
                            f"Transposing Vc from {Vc.shape} to {Vc.T.shape}"
                        )
                        Vc = Vc.T
                    
                    result['Vc'] = Vc
                    result['metadata']['n_components_actual'] = Vc.shape[0]
                    result['metadata']['n_frames'] = Vc.shape[1]
        
        except Exception as e:
            result['errors'].append(f"Failed to load Vc components: {e}")
    
    # === 3. LOAD U SPATIAL COMPONENTS ===
    # Often stored in the same metadata file
    if metadata_file:
        try:
            mat_data = loadmat(str(metadata_file), simplify_cells=True)
            u_var_names = ['U', 'u', 'spatial_components']
            for var_name in u_var_names:
                if var_name in mat_data:
                    U = mat_data[var_name]
                    # Validate dimensions
                    if U.ndim == 3:
                        result['U'] = U
                        result['metadata']['spatial_shape'] = U.shape[:2]
                        break
        except Exception as e:
            result['warnings'].append(f"Could not load U from metadata file: {e}")
    
    # === 4. LOAD BRAIN MASK ===
    mask_patterns = [
        f'{scan_id}_brain_mask.npy',
        f'{scan_id}_mask.npy',
        '*_brain_mask.npy'
    ]
    
    mask_file = None
    for pattern in mask_patterns:
        matches = list(folder_path.glob(pattern))
        if matches:
            mask_file = matches[0]
            break
    
    if mask_file:
        try:
            mask = np.load(mask_file)
            if mask.ndim != 2:
                result['warnings'].append(f"Brain mask should be 2D, got shape {mask.shape}")
            else:
                result['brain_mask'] = mask.astype(bool)
                result['metadata']['brain_pixel_fraction'] = mask.sum() / mask.size
        except Exception as e:
            result['warnings'].append(f"Failed to load brain mask: {e}")
    
    # === 5. LOAD CCF LANDMARKS ===
    landmarks_patterns = [
        f'{scan_id}_transform_landmarks.json',
        f'{scan_id}_ccf_landmarks.json',
        '*_transform_landmarks.json'
    ]
    
    landmarks_file = None
    for pattern in landmarks_patterns:
        matches = list(folder_path.glob(pattern))
        if matches:
            landmarks_file = matches[0]
            break
    
    if landmarks_file:
        try:
            with open(landmarks_file, 'r') as f:
                result['ccf_landmarks'] = json.load(f)
        except Exception as e:
            result['warnings'].append(f"Failed to load CCF landmarks: {e}")
    
    # === 6. LOAD VESSELS TRANSFORM ===
    vessels_patterns = [
        f'{scan_id}_vessels_transform.mat',
        f'{scan_id}_vessels.mat',
        '*_vessels_transform.mat'
    ]
    
    vessels_file = None
    for pattern in vessels_patterns:
        matches = list(folder_path.glob(pattern))
        if matches:
            vessels_file = matches[0]
            break
    
    if vessels_file:
        try:
            mat_data = loadmat(str(vessels_file), simplify_cells=True)
            # Look for common variable names
            vessel_vars = ['vessels', 'vessel_image', 'bloodvessels', 'transform']
            for var in vessel_vars:
                if var in mat_data:
                    result['vessels'] = mat_data[var]
                    break
        except Exception as e:
            result['warnings'].append(f"Failed to load vessels: {e}")
    
    # === CONSISTENCY CHECKS ===
    if result['U'] is not None and result['Vc'] is not None:
        if result['U'].shape[2] != result['Vc'].shape[0]:
            result['errors'].append(
                f"Component mismatch: U has {result['U'].shape[2]} components, "
                f"Vc has {result['Vc'].shape[0]} components"
            )
    
    if result['brain_mask'] is not None and result['U'] is not None:
        if result['brain_mask'].shape != result['U'].shape[:2]:
            result['errors'].append(
                f"Spatial dimension mismatch: brain_mask {result['brain_mask'].shape}, "
                f"U {result['U'].shape[:2]}"
            )
    
    return result


# ==============================================================================
# TODO 4: WFIELD SVD INGESTION FUNCTION
# ==============================================================================
def generate_proc_id():
    """
    Generate Base-36 timestamp ID with 'proc' prefix.
    Matches MATLAB's userfunction_getfileID pattern.
    
    Returns:
        str: 'proc' + 8-character Base-36 ID (e.g., 'proc9FXECJI5')
    """
    from datetime import datetime
    import numpy as np
    
    # MATLAB's now() is days since Jan 1, 0000
    # Python equivalent: (datetime.now() - datetime(1, 1, 1)).total_seconds() / 86400
    matlab_epoch = datetime(1, 1, 1)
    now = datetime.now()
    days_since_epoch = (now - matlab_epoch).total_seconds() / 86400
    
    # Scale to match MATLAB precision (10^6)
    time_serial = int(days_since_epoch * 1e6)
    
    # Convert to Base-36 (0-9, A-Z)
    def to_base36(num):
        chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        result = ''
        while num > 0:
            result = chars[num % 36] + result
            num //= 36
        return result.zfill(8)  # Pad to 8 characters
    
    return 'proc' + to_base36(time_serial)

def ingest_wfield(scan_id: str,
                  key: dict = None,
                  gpu_disk_path: Optional[str] = None,
                  skip_existing: bool = True) -> Dict[str, any]:
    """
    Ingest widefield SVD data into database schema.
    
    This is the main ingestion function that:
    1. Loads SVD metadata and components
    2. Loads brain mask if available
    3. Loads CCF transform landmarks if available
    4. Loads vessels transform if available
    5. Inserts into appropriate WFieldSVD tables
    6. Handles retinotopic map (separate subject-level folder)
    
    Args:
        scan_id: Scan identifier (e.g., 'scan9FW5OVL6')
        gpu_disk_path: Optional path to wfield_data folder. If None, uses config.
        skip_existing: If True, skip scans already in database
    
    Returns:
        Dictionary with ingestion status:
        {
            'success': bool,
            'scan_id': str,
            'inserted_tables': List[str],
            'errors': List[str],
            'warnings': List[str]
        }
    
    Examples:
        >>> result = ingest_wfield('scan9FW5OVL6')
        >>> if result['success']:
        ...     print(f"Ingested into: {result['inserted_tables']}")
        >>> else:
        ...     print(f"Errors: {result['errors']}")
    
    Notes:
        - Requires adamacs.schemas.wfield schema to be imported
        - Does NOT process raw imaging data - only ingests existing SVD outputs
        - For retinotopic maps, searches parent data folder for subject-specific folders
    """
    from ..schemas import wfield
    from ..pipeline import scan as scan_module
    
    result = {
        'success': False,
        'scan_id': scan_id,
        'inserted_tables': [],
        'errors': [],
        'warnings': []
    }
    
    # Check if already in database
    if skip_existing:
        existing = wfield.WfieldSvd & {'scan_id': scan_id}
        if len(existing) > 0:
            result['warnings'].append(f"Scan {scan_id} already in WfieldSvd table")
            result['success'] = True
            return result
    
    # === 2. CHECK WFIELD FOLDER ===
    folder_status = check_wfield_folder(scan_id, gpu_disk_path)
    
    if not folder_status['ready_for_ingestion']:
        result['errors'].append(
            f"Missing required files: {folder_status['missing_files']}"
        )
        return result
    
    folder_path = folder_status['folder_path']
    
    # === 3. LOAD SVD OUTPUTS ===
    try:
        svd_data = load_svd_outputs(folder_path, scan_id)
    except Exception as e:
        result['errors'].append(f"Failed to load SVD outputs: {e}")
        return result
    
    # Check for critical errors
    if svd_data['errors']:
        result['errors'].extend(svd_data['errors'])
        return result
    
    result['warnings'].extend(svd_data['warnings'])
    
    # === 4. PREPARE DATABASE ENTRY ===
    try:
        # Get scan key from database
        scan_key = (scan_module.Scan & {'scan_id': scan_id}).fetch1('KEY')
        
        # Prepare main WFieldSVD entry
        wfield_entry = {
            **scan_key,
            'processing_id': key['processing_id'],
            'n_components': svd_data['metadata'].get('n_components_actual', 200),
            'variance_explained': svd_data['metadata'].get('variance_explained', 0.0),
            'n_frames': svd_data['metadata'].get('n_frames', svd_data['Vc'].shape[1]),
            'has_brain_mask': svd_data['brain_mask'] is not None,
            'has_ccf_alignment': svd_data['ccf_landmarks'] is not None,
            'processing_complete': True,
            'svd_output_folder': str(folder_path)
        }
        
        # Insert main entry
        # NOTE: This function uses an older schema structure. Use ingest_svd_from_matlab() instead.
        wfield.WfieldSvd.insert1(wfield_entry, skip_duplicates=True)
        result['inserted_tables'].append('WfieldSvd')
        
        # === 5. INSERT COMPONENTS (if schema supports it) ===
        if hasattr(wfield, 'SvdSpatialComponent') and svd_data['U'] is not None:
            spatial_entry = {
                **scan_key,
                'spatial_components': svd_data['U']  # (height, width, n_components)
            }
            wfield.SvdSpatialComponent.insert1(spatial_entry, skip_duplicates=True)
            result['inserted_tables'].append('SvdSpatialComponent')
        
        if hasattr(wfield, 'SvdTemporalComponent') and svd_data['Vc'] is not None:
            temporal_entry = {
                **scan_key,
                'temporal_components': svd_data['Vc']  # (n_components, n_frames)
            }
            wfield.SvdTemporalComponent.insert1(temporal_entry, skip_duplicates=True)
            result['inserted_tables'].append('SvdTemporalComponent')
        
        # === 6. INSERT BRAIN MASK ===
        if hasattr(wfield, 'BrainMask') and svd_data['brain_mask'] is not None:
            mask_entry = {
                **scan_key,
                'mask_idx': 0,
                'brain_mask': svd_data['brain_mask'],
                'mask_method': 'svd_threshold',
                'mask_params': {'source': 'matlab_svd'},
                'creation_time': datetime.now(),
                'notes': 'Imported from MATLAB SVD processing'
            }
            wfield.BrainMask.insert1(mask_entry, skip_duplicates=True)
            result['inserted_tables'].append('BrainMask')
        
        # === 7. INSERT CCF ALIGNMENT (session-level) ===
        if hasattr(wfield, 'AtlasAlignment') and svd_data['ccf_landmarks'] is not None:
            # Need to get session_id from scan_id
            from ..pipeline import session as session_module
            scan_info = scan_module.ScanInfo & scan_key
            if len(scan_info) > 0:
                session_id = scan_info.fetch1('session_id')
                ccf_entry = {
                    'session_id': session_id,
                    'curation_idx': 0,  # Default curation
                    'registration_method': 'manual_landmarks',
                    'reference_scan_id': scan_id,
                    'transform_matrix': svd_data['ccf_landmarks'].get('transform_matrix', np.eye(3)),
                    'bregma_offset_x': svd_data['ccf_landmarks'].get('bregma_x', 0.0),
                    'bregma_offset_y': svd_data['ccf_landmarks'].get('bregma_y', 0.0),
                    'resolution_um': svd_data['metadata'].get('pixel_size_um', 10.0),
                    'registration_time': datetime.now(),
                    'notes': 'Imported from MATLAB processing'
                }
                wfield.AtlasAlignment.insert1(ccf_entry, skip_duplicates=True)
                result['inserted_tables'].append('AtlasAlignment')
        
        # === 8. INSERT RETINOTOPIC MAP (subject-level, one-time) ===
        if hasattr(wfield, 'WFieldRetinotopicMap'):
            try:
                # Get subject_id from scan following proper hierarchy
                subject_id = get_subject_from_scan(scan_id)
                
                # Check if retinotopic map already exists for this subject
                if not (wfield.WFieldRetinotopicMap & {'subject': subject_id}):
                    # Find and load retinotopic data (subject-specific folder)
                    retino_folder = find_retinotopic_folder(subject_id)
                    
                    if retino_folder:
                        retino_data = load_retinotopic_data(retino_folder, subject_id)
                        
                        if retino_data:
                            retino_entry = {
                                'subject': subject_id,
                                'retinotopic_session_id': retino_data['retinotopic_session_id'],
                                'bloodvessels_image': retino_data['bloodvessels_image'],
                                'vfs_map': retino_data['vfs_map'],
                                'retinotopic_folder': retino_data['retinotopic_folder']
                            }
                            wfield.WFieldRetinotopicMap.insert1(retino_entry, skip_duplicates=True)
                            result['inserted_tables'].append('WFieldRetinotopicMap')
                            print(f"✅ Retinotopic map ingested for subject {subject_id}")
                        else:
                            result['warnings'].append(f"Retinotopic data incomplete for {subject_id}")
                    else:
                        result['warnings'].append(f"No retinotopic folder found for {subject_id}")
                else:
                    print(f"ℹ️ Retinotopic map already exists for {subject_id}, skipping")
                            
            except Exception as e:
                result['warnings'].append(f"Retinotopic map ingestion failed: {e}")
                # Don't fail entire ingestion if retinotopic fails
        
        result['success'] = True
        
    except Exception as e:
        result['errors'].append(f"Database insertion failed: {e}")
        import traceback
        result['errors'].append(traceback.format_exc())
    
    return result


# ==============================================================================
# DATABASE HIERARCHY UTILITIES
# ==============================================================================

def get_subject_from_scan(scan_id: str) -> str:
    """
    Get subject_id from scan_id following database hierarchy.
    
    Hierarchy: scan_id -> ScanInfo -> session_id -> Session -> subject
    
    Args:
        scan_id: Scan identifier (e.g., 'scan9FW5OVL6')
    
    Returns:
        subject_id: Subject identifier (e.g., 'ROS-2156')
    
    Raises:
        DataJointError: If scan or session not found in database
    
    Examples:
        >>> subject_id = get_subject_from_scan('scan9FW5OVL6')
        >>> print(f"Scan belongs to subject: {subject_id}")
    
    Notes:
        - This respects the scan -> session -> subject hierarchy
        - Critical for multi-scan sessions where multiple scans share one subject
        - See SCAN_SESSION_HIERARCHY_REFERENCE.md for detailed documentation
    """
    from ..pipeline import scan as scan_module, session as session_module
    
    scan_key = {'scan_id': scan_id}
    scan_info = scan_module.ScanInfo & scan_key
    
    if len(scan_info) == 0:
        raise ValueError(f"Scan {scan_id} not found in ScanInfo table")
    
    session_id = scan_info.fetch1('session_id')
    session_data = session_module.Session & {'session_id': session_id}
    
    if len(session_data) == 0:
        raise ValueError(f"Session {session_id} not found in Session table")
    
    subject_id = session_data.fetch1('subject')
    return subject_id


# ==============================================================================
# RETINOTOPIC MAP LOADING (TODO 6 - Complete)
# ==============================================================================

def find_retinotopic_folder(subject_id: str, 
                           data_folder: Optional[str] = None) -> Optional[Path]:
    """
    Search for subject-specific retinotopic map folder.
    
    Pattern: Based on AA_ccf_align.ipynb, searches for folders matching:
    - `*{subject_id}_25*` (date-based naming like ROS-2156_251003)
    
    Args:
        subject_id: Subject identifier from subject.Subject() table (e.g., 'ROS-2156')
        data_folder: Root data folder to search. If None, reads from DJ config.
    
    Returns:
        Path to retinotopic folder or None if not found
    
    Examples:
        >>> folder = find_retinotopic_folder('ROS-2156')
        >>> if folder:
        ...     bloodvessels = load_retinotopic_data(folder, 'ROS-2156')
    """
    # Get data folder from config if not provided
    if data_folder is None:
        try:
            import datajoint as dj
            data_folder = dj.config['custom']['imaging_root_data_dir'][0]
        except:
            data_folder = '/datajoint-data/data/aeltona/'
    
    data_folder = Path(data_folder)
    
    if not data_folder.exists():
        warnings.warn(f"Data folder does not exist: {data_folder}")
        return None
    
    # Search pattern from AA_ccf_align.ipynb: *{subject_id}_25*
    # This matches folders like: ROS-2156_251003, ROS-2139_250915, etc.
    pattern = f'*{subject_id}_25*'
    
    matches = list(data_folder.glob(pattern))
    
    # Filter to directories only
    matches = [m for m in matches if m.is_dir()]
    
    if not matches:
        return None
    
    if len(matches) > 1:
        warnings.warn(
            f"Multiple retinotopic folders found for {subject_id}: {[m.name for m in matches]}\n"
            f"Using first match: {matches[0].name}"
        )
    
    return matches[0]


def load_retinotopic_data(retinotopic_folder: Path, 
                          subject_id: str) -> Optional[Dict]:
    """
    Load retinotopic mapping data from folder.
    
    Based on AA_ccf_align.ipynb workflow:
    - Loads BrainSkullView.tiff for bloodvessels image
    - Uses wf.load_vfs() to load VFS map from Block*/Wave_maps_Block*.mat files
    
    Args:
        retinotopic_folder: Path to retinotopic data folder
        subject_id: Subject identifier (for determining which blocks to use)
    
    Returns:
        Dictionary with retinotopic data or None if files not found:
        {
            'subject_id': str,
            'retinotopic_session_id': str,
            'bloodvessels_image': ndarray,
            'vfs_map': ndarray,
            'retinotopic_folder': str
        }
    
    Examples:
        >>> folder = find_retinotopic_folder('ROS-2156')
        >>> data = load_retinotopic_data(folder, 'ROS-2156')
        >>> if data:
        ...     print(f"VFS map shape: {data['vfs_map'].shape}")
    """
    retinotopic_folder = Path(retinotopic_folder)
    
    if not retinotopic_folder.exists():
        warnings.warn(f"Retinotopic folder does not exist: {retinotopic_folder}")
        return None
    
    result = {
        'subject_id': subject_id,
        'retinotopic_folder': str(retinotopic_folder)
    }
    
    # === 1. EXTRACT SESSION ID FROM FOLDER NAME ===
    # Pattern: ROS-2156_251003 (date-based), or ROS-2156_sess9ABC123 (session-based)
    import re
    folder_name = retinotopic_folder.name
    session_match = re.search(r'sess([A-Z0-9]{8})', folder_name)
    
    if session_match:
        result['retinotopic_session_id'] = f'sess{session_match.group(1)}'
    else:
        # Fallback: use folder name (e.g., ROS-2156_251003)
        result['retinotopic_session_id'] = folder_name
    
    # === 2. LOAD BLOOD VESSELS IMAGE (BrainSkullView.tiff) ===
    bloodvessels_file = retinotopic_folder / 'BrainSkullView.tiff'
    
    if not bloodvessels_file.exists():
        # Try alternative names
        alt_names = ['BrainSkullView.tif', 'bloodvessels.tiff', 'bloodvessels.tif']
        for alt_name in alt_names:
            alt_file = retinotopic_folder / alt_name
            if alt_file.exists():
                bloodvessels_file = alt_file
                break
        else:
            warnings.warn(f"No BrainSkullView.tiff found in {retinotopic_folder}")
            return None
    
    try:
        from tifffile import imread
        result['bloodvessels_image'] = imread(str(bloodvessels_file))
    except Exception as e:
        warnings.warn(f"Failed to load bloodvessels image: {e}")
        return None
    
    # === 3. LOAD VFS MAP using wf.load_vfs() ===
    # Determine which blocks to use based on subject_id
    if '2139' in subject_id:
        blocks = [1]
    elif '2156' in subject_id:
        blocks = [5]
    else:
        blocks = [1, 2, 3, 4, 5]  # Default: all blocks
    
    try:
        # NOTE: Requires wf.py module from old_main_backup for load_vfs() function
        # TODO: Extract load_vfs() into wfield_helpers.py to remove dependency
        import wf
        result['vfs_map'] = wf.load_vfs(str(retinotopic_folder), blocks=blocks)
    except Exception as e:
        warnings.warn(f"Failed to load VFS map: {e}")
        return None
    
    # === 4. VALIDATE DATA ===
    if 'bloodvessels_image' not in result or 'vfs_map' not in result:
        warnings.warn(f"Incomplete retinotopic data in {retinotopic_folder}")
        return None
    
    print(f"✅ Loaded retinotopic data for {subject_id}")
    print(f"   Blood vessels: {result['bloodvessels_image'].shape}")
    print(f"   VFS map: {result['vfs_map'].shape}")
    print(f"   Used blocks: {blocks}")
    
    return result


# ==============================================================================
# REGISTRATION UTILITIES (TODO 8 - Part 1)
# ==============================================================================

def check_registration(scan_id: str, wfield_folder: Optional[str] = None) -> Dict[str, any]:
    """
    Check if registration (motion correction) has been computed for a scan.
    
    Args:
        scan_id: Scan identifier
        wfield_folder: Root wfield_data folder. If None, reads from DJ config.
    
    Returns:
        Dictionary with registration status:
        {
            'registered': bool,
            'shifts_file': Path or None,
            'n_frames': int or None,
            'mean_shifts': tuple or None,  # (mean_x, mean_y)
            'max_shifts': tuple or None     # (max_x, max_y)
        }
    
    Examples:
        >>> status = check_registration('scan9FW5OVL6')
        >>> if status['registered']:
        ...     print(f"Shifts file: {status['shifts_file']}")
        >>> else:
        ...     print("Need to compute registration")
    """
    # Get wfield folder from config if not provided
    if wfield_folder is None:
        try:
            import datajoint as dj
            wfield_folder = dj.config['custom']['wfield_processing_dir']
        except:
            wfield_folder = '/home/aeltona/wfield_data'
    
    scan_folder = Path(wfield_folder) / scan_id
    
    status = {
        'registered': False,
        'shifts_file': None,
        'n_frames': None,
        'mean_shifts': None,
        'max_shifts': None
    }
    
    if not scan_folder.exists():
        return status
    
    # Look for shifts file (matches wf.registration() output pattern)
    shifts_files = list(scan_folder.glob('*shifts.npz')) + list(scan_folder.glob('*shifts.mat'))
    
    if not shifts_files:
        return status
    
    shifts_file = shifts_files[0]
    status['registered'] = True
    status['shifts_file'] = shifts_file
    
    # Load and analyze shifts
    try:
        if shifts_file.suffix == '.npz':
            import numpy as np
            shifts_data = np.load(shifts_file)
            shifts = shifts_data['shifts']  # (n_frames, 2) array of (dy, dx)
        elif shifts_file.suffix == '.mat':
            from scipy.io import loadmat
            mat_data = loadmat(shifts_file)
            shifts = mat_data['shifts']  # (n_frames, 2) array
        else:
            return status
        
        status['n_frames'] = shifts.shape[0]
        status['mean_shifts'] = (float(np.mean(shifts[:, 1])), float(np.mean(shifts[:, 0])))  # (x, y)
        status['max_shifts'] = (float(np.max(np.abs(shifts[:, 1]))), float(np.max(np.abs(shifts[:, 0]))))
        
    except Exception as e:
        warnings.warn(f"Could not load shifts file: {e}")
    
    return status


# ==============================================================================
# CCF ALIGNMENT AND ROI EXTRACTION HELPERS (from wf.py)
# ==============================================================================
def _create_wfield_folder(wfield_dir):
    if not os.path.isdir(wfield_dir):
        print('Created {0}'.format(wfield_dir))
        os.makedirs(wfield_dir)
    # try first from the shared folder 
    modulepath = pjoin(__file__.split('lib')[0],'share','wfield')
    refpath = pjoin(modulepath, 'references')
    if os.path.exists(refpath):
        reference_files = [pjoin(refpath,r) for r in os.listdir(refpath)]
        from shutil import copyfile
        for f in reference_files:
            if os.path.isfile(f):
                copyfile(f,f.replace(refpath,wfield_dir))
    else:
        try:
            import requests
        except Exception as err:
            print(err)
            raise(OSError('Could not import the requests package, please install it "pip install requests"'))
        from shutil import copyfileobj
        webpath = 'https://raw.githubusercontent.com/jcouto/wfield/master/references/{0}'
        files = ['dorsal_cortex_ccf_labels.json',
                 'dorsal_cortex_landmarks.json',
                 'dorsal_cortex_outline.npy',
                 'dorsal_cortex_projection.npy',
                 'vis_ccf_labels.json',
                 'vis_outline.npy',
                 'vis_projection.npy']
        # download the files
        for f in files:
            print('    Downloading {0}'.format(f))
            if '.json' in f: # because of the encodings.
                with open(pjoin(wfield_dir,f),'w') as fid:
                    res = requests.get(webpath.format(f))
                    fid.write(res.text)
                    del res
            else:
                with open(pjoin(wfield_dir,f),'wb') as fid:
                    raw = requests.get(webpath.format(f),stream=True)
                    copyfileobj(raw.raw, fid)
                    del raw

def allen_load_reference(reference_name, annotation_dir):
    '''
Load allen areas to use as reference.

Example:
    ccf_regions,proj,brain_outline = allen_load_reference('dorsal_cortex')

    Joao Couto - wfield, 2020
    '''
    if annotation_dir == pjoin(os.path.expanduser('~'),'.wfield'):
        # then it is the reference folder, download if not there
        if not os.path.exists(pjoin(
                annotation_dir,
                '{0}_ccf_labels.json'.format(reference_name))):
            from .utils import _create_wfield_folder
            _create_wfield_folder(annotation_dir)
    from pandas import read_json
    ccf_regions = read_json(pjoin(
        annotation_dir,'{0}_ccf_labels.json'.format(reference_name)))
    proj = np.load(pjoin(annotation_dir,
                         '{0}_projection.npy'.format(reference_name),))
    brain_outline = np.load(pjoin(annotation_dir,
                                  '{0}_outline.npy'.format(reference_name)))
    return ccf_regions,proj,brain_outline

def save_allen_landmarks(landmarks, annotation_dir,
                         filename = None,
                         resolution = None,
                         landmarks_match = None,
                         bregma_offset = None,
                         transform = None,
                         transform_inverse = None,
                         transform_type = 'similarity',
                         **kwargs):
    '''
    landmarks need to be pandas dataframes.

    default is dorsal_cortex_landmarks.json in the ~/.wfield directory.

    '''
    lmarks = dict(landmarks=landmarks.to_dict(orient='list'))
    if not resolution is None:
        lmarks['resolution'] = resolution    
    if not landmarks_match is None:
        lmarks['landmarks_match'] = landmarks_match.to_dict(orient='list')
    if not bregma_offset is None:
        if isinstance(bregma_offset,np.ndarray):
            bregma_offset = bregma_offset.tolist()
        lmarks['bregma_offset'] = bregma_offset
    if not transform is None:
        from skimage.transform import SimilarityTransform,AffineTransform
        if isinstance(transform,SimilarityTransform) or isinstance(transform,AffineTransform):
            lmarks['transform'] = transform.params.tolist()
        elif isinstance(transform,np.ndarray):
            lmarks['transform'] = transform.tolist()
        else:
            lmarks['transform'] = transform
    if not transform_inverse is None:
        from skimage.transform import SimilarityTransform,AffineTransform
        if isinstance(transform,SimilarityTransform) or isinstance(transform,AffineTransform):
            lmarks['transform_inverse'] = transform_inverse.params.tolist()
        elif isinstance(transform,np.ndarray):
            lmarks['transform_inverse'] = transform.tolist()
        else:
            lmarks['transform_inverse'] = transform_inverse
    if 'bregma_offset' in lmarks.keys() and 'resolution' in lmarks.keys():
        lmarks['landmarks_im'] = allen_landmarks_to_image_space(
            landmarks.copy(), 
            lmarks['bregma_offset'],
            lmarks['resolution']).to_dict(orient='list')
    if filename is None:
        filename = pjoin(annotation_dir,'dorsal_cortex_landmarks.json')
    with open(filename,'w') as fd:
        import json
        json.dump(lmarks,fd, sort_keys = True, indent = 4)

def align_scan_to_ccf(lmark_wid, resolution, bregma_offset, scan_key, image):
    landmarks = pd.DataFrame(lmark_wid.data)[['x','y','name','color']] # landmarks in allen_coords

    if len(wfield.CcfAlignment & scan_key) > 0:
        print('Using CCF alignment from database')
        bregma_offset = (wfield.CcfAlignment & scan_key).fetch1('bregma_offset')
        resolution = (wfield.CcfAlignment & scan_key).fetch1('resolution')
        landmarks_match = (wfield.CcfAlignment & scan_key).fetch1('landmarks_match')
        landmarks_im = (wfield.CcfAlignment & scan_key).fetch1('landmarks_im')
        wid,lmark_wid,_ = adjust_img_landmarks(image,landmarks,
                                                        landmarks_match = landmarks_match,
                                                        bregma_offset = bregma_offset,
                                                        resolution = resolution)
    else:
        print('Place the 4 markers based on atlas landmarks chosen in the above cell')
        landmarks_match = None
        wid,lmark_wid,landmarks_im = adjust_img_landmarks(image,landmarks,
                                                        landmarks_match = landmarks_match,
                                                        bregma_offset = bregma_offset,
                                                        resolution = resolution)

    
    return wid, lmark_wid, landmarks_im

def load_allen_landmarks(filename, annotation_dir, reference = 'dorsal_cortex'):
    '''
    lmarks = load_allen_landmarks(filename, reference = 'dorsal_cortex'):
    
    Loads an allen landmark file (json) and returns the transform objects if present.
    Joao Couto - wfield 2020
    '''
    if filename is None:
        filename = pjoin(annotation_dir,reference + '_landmarks.json')
    if not os.path.exists(filename):
        if '.wfield' in filename:
            from .utils import _create_wfield_folder
            _create_wfield_folder()
        else:
            raise(OSError('Could not find the reference file {0}.'.format(filename)))

    with open(filename,'r') as fd:
        import json
        lmarks = json.load(fd)
    for k in ['landmarks_im','landmarks','landmarks_match']:
        if k in lmarks.keys():
            from pandas import DataFrame
            lmarks[k] = DataFrame(lmarks[k])[['x','y','name','color']]
    if 'transform' in lmarks.keys():
        if not 'transform_type' in lmarks.keys():
            lmarks['transform_type'] = 'euclidian'
        if lmarks['transform_type'] == 'affine':
            from skimage.transform import AffineTransform
            lmarks['transform'] = AffineTransform(
                np.array(lmarks['transform']))
        else: # use similarity
            from skimage.transform import SimilarityTransform
            lmarks['transform'] = SimilarityTransform(
                np.array(lmarks['transform']))
        if 'transform_inverse' in lmarks.keys():
            if lmarks['transform_type'] == 'affine':
                from skimage.transform import AffineTransform
                lmarks['transform_inverse'] = AffineTransform(
                    np.array(lmarks['transform_inverse']))
            else: # use similarity
                from skimage.transform import SimilarityTransform
                lmarks['transform_inverse'] = SimilarityTransform(
                    np.array(lmarks['transform_inverse']))
    return lmarks


def estimate_similarity_transform(ref,points):
    '''
    
    ref = np.vstack([landmarks_im['x'],landmarks_im['y']]).T
    match = point_stream.data    
    cor = np.vstack([match['x'],match['y']]).T
    
    M = estimate_similarity_transform(ref, cor)
    
    Joao Couto - wfield (2020)
    '''
    from skimage.transform import SimilarityTransform
    M = SimilarityTransform()
    M.estimate(ref,points)
    return M

def estimate_similarity_transform(ref,points):
    '''
    
    ref = np.vstack([landmarks_im['x'],landmarks_im['y']]).T
    match = point_stream.data    
    cor = np.vstack([match['x'],match['y']]).T
    
    M = estimate_similarity_transform(ref, cor)
    
    Joao Couto - wfield (2020)
    '''
    from skimage.transform import SimilarityTransform
    M = SimilarityTransform()
    M.estimate(ref,points)
    return M

def allen_transform_from_landmarks(landmarks_im,match):
    '''
    Compute the similarity transform from annotated landmarks. 
    
    transform = allen_transform_from_landmarks(landmarks_im,match)
    
    '''
    ref = np.vstack([landmarks_im['x'],landmarks_im['y']]).T
    cor = np.vstack([match['x'],match['y']]).T
    return estimate_similarity_transform(ref, cor)

def plot_frame_with_allen_overlay(frame, ccf_regions, M,
                                  resolution=1,
                                  bregma_offset=np.array([0, 0]),
                                  side_selection='both',
                                  ax=None):
    import matplotlib.pyplot as plt
    import numpy as np
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    fig.set_facecolor('white')
    # Show the image
    ax.imshow(frame, cmap='gray', origin='upper')  # origin='upper' = (0,0) at top-left

    # Plot transformed atlas contours
    for _, c in ccf_regions.iterrows():
        if side_selection in ['right', 'both']:
            coords = np.vstack([
                np.array(c.right_x) / resolution + bregma_offset[0],
                np.array(c.right_y) / resolution + bregma_offset[1]
            ]).T
            transformed = M(coords)
            ax.plot(transformed[:, 0], transformed[:, 1], color='white', linewidth=1)
        if side_selection in ['left', 'both']:
            coords = np.vstack([
                np.array(c.left_x) / resolution + bregma_offset[0],
                np.array(c.left_y) / resolution + bregma_offset[1]
            ]).T
            transformed = M(coords)
            ax.plot(transformed[:, 0], transformed[:, 1], color='white', linewidth=1)

    ax.set_xlim(0, frame.shape[1])
    ax.set_ylim(frame.shape[0], 0)  # invert y-axis to match image coordinates
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Frame with transformed atlas overlay")
    return ax


def make_region_mask(shape, ccf_regions, ccf_regions_im, M,
                     resolution=1,
                     bregma_offset=np.array([0, 0]),
                     exclude=None):
    '''
    Build a boolean mask covering all CCF atlas regions in image pixel space.

    Uses the same coordinate transform as plot_frame_with_allen_overlay
    (divide by resolution, add bregma_offset, then apply M).

    Parameters
    ----------
    shape : tuple (H, W)
        Output mask shape.
    ccf_regions : pd.DataFrame
        From allen_load_reference; must have left_x/y, right_x/y columns.
    M : SimilarityTransform
        CCF-to-image transform (same as used for plotting).
    resolution : float
        mm per pixel (same as used for plotting).
    bregma_offset : array-like (2,)
        Bregma position in pixels (same as used for plotting).
    exclude : list of str, optional
        Region acronyms to skip. Defaults to ['MOB'].

    Returns
    -------
    region_mask : bool ndarray (H, W)
    '''
    from matplotlib.path import Path

    if exclude is None:
        exclude = ['MOB']

    region_polygons = []
    for i, r in ccf_regions_im.iterrows():
        if r['acronym'] != 'MOB':
            for side in ['left', 'right']:
                poly_coords = np.column_stack([r[side + '_x'], r[side + '_y']])
                if len(poly_coords) >= 3:
                    region_polygons.append(poly_coords)
    region_mask = np.zeros(shape, dtype=bool)
    for poly in region_polygons:
        path = Path(poly)
        y, x = np.mgrid[0:shape[0], 0:shape[1]]
        points = np.column_stack((x.ravel(), y.ravel()))
        mask = path.contains_points(points).reshape(shape)
        region_mask |= mask

    return region_mask


def plot_vfs_masked(image, fov_mask, ccf_regions,ccf_regions_im, M,
                    scan_key,
                    resolution=1,
                    bregma_offset=np.array([0, 0]),
                    exclude=None,
                    show_labels=True,
                    ax=None):
    '''
    Plot image (e.g. vfs_vessels) with a combined FOV+region mask applied,
    CCF region boundaries clipped to the FOV drawn on top, and the FOV
    outer contour drawn in black.

    Parameters
    ----------
    image : ndarray (H, W) or (H, W, 3)
        Frame to display (e.g. vfs_vessels from plot_aligned_overlay).
    fov_mask : bool ndarray (H, W)
        FOV / brain mask in image pixel space (e.g. brain_mask_smooth).
    ccf_regions : pd.DataFrame
        From allen_load_reference.
    M : SimilarityTransform
        CCF-to-image transform.
    resolution : float
        mm per pixel.
    bregma_offset : array-like (2,)
        Bregma position in pixels.
    exclude : list of str, optional
        Region acronyms to skip. Defaults to ['MOB'].
    show_labels : bool
        Whether to add region acronym labels at centroids.
    ax : matplotlib axis, optional

    Returns
    -------
    ax : matplotlib axis

    Usage
    -----
    reload(wfh);
    wfh.plot_vfs_masked(
        image=vfs_vessels,
        fov_mask=brain_mask_smooth,
        ccf_regions=ccf_regions,
        M=M,
        resolution=resolution,
        bregma_offset=bregma_offset,
    )
    plt.axis('off'); plt.show()
    '''
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours

    if exclude is None:
        exclude = ['MOB']

    
    if len(wfield.RegionMask & scan_key) ==0:
        print('Generating region_mask')
    # --- Combined mask ---
        region_mask = make_region_mask(
            shape=image.shape[:2],
            ccf_regions=ccf_regions,
            ccf_regions_im=ccf_regions_im,
            M=M,
            resolution=resolution,
            bregma_offset=bregma_offset,
            exclude=exclude,
        )
    else:
        print('Loading region_mask from database')
        region_mask = (wfield.RegionMask & scan_key).fetch1('region_mask')
    if len(image.shape) == 2:
        masked_image= np.where(fov_mask, image, np.nan)
    elif len(image.shape) == 3:
        masked_image = np.zeros_like(image)
        for color in range(image.shape[-1]):
            test = np.where(fov_mask, image[:, :, color], 255)
            masked_image[:, :, color] = test
    

    for i, r in ccf_regions_im.iterrows():
        if r['acronym'] != 'MOB':
            for side in ['left', 'right']:
                area_x = np.array(r[side + '_x'])
                area_y = np.array(r[side + '_y'])
                # Convert to integer coordinates and clip to image bounds
                xx = np.clip(np.round(area_x).astype(int), 0, fov_mask.shape[1] - 1)
                yy = np.clip(np.round(area_y).astype(int), 0, fov_mask.shape[0] - 1)
                # Create a mask for points inside fov_mask
                in_mask = fov_mask[yy, xx].astype(bool)
                # Apply the mask to coordinates
                x_masked = np.where(in_mask, area_x, np.nan)
                y_masked = np.where(in_mask, area_y, np.nan)
                # Plot only the in-mask segments (NaNs will break lines)
                plt.plot(x_masked, y_masked, 'k', lw=1.5)
                
                # Plot acronym label only if the centroid is in the FOV
                if side == 'right':
                    cx = int(np.round(np.mean(area_x)))
                    cy = int(np.round(np.mean(area_y)))
                    if fov_mask[np.clip(cy, 0, fov_mask.shape[0]-1),
                                np.clip(cx, 0, fov_mask.shape[1]-1)]:
                        plt.text(np.mean(area_x), np.mean(area_y), r['acronym'],
                                color='white', fontsize=8, ha='center', va='center')
    if len(image.shape) == 2:
        masked_image = np.where(region_mask, masked_image,np.nan)
        mask = ~np.isnan(masked_image)
    elif len(image.shape) == 3:
        for color in range(masked_image.shape[-1]):
            masked_image[:, :, color] = np.where(region_mask, masked_image[:, :, color], 255)
        mask = masked_image[:,:,0]!=255  # assume white background for RGB
    plt.imshow(masked_image);
    # --- FOV outer contour ---
      # True inside the data, False outside
    contours = find_contours(mask.astype(float), level=0.5)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], color='black', lw=3)
    plt.axis('off')
    plt.show()
    return None



def hv_plot_allen_regions(ccf_regions,
                          resolution = 1,
                          bregma_offset = np.array([0,0]),
                          side_selection='both'):
    '''
    Example: 

    import holoviews as hv
    hv.extension('bokeh')

    ccf_regions,proj,brain_outline = allen_load_reference('dorsal_cortex')
    hv_plot_allen_regions(ccf_regions).options({'Curve': {'color':'black', 'width': 600}})
        
    '''
    import holoviews as hv
    regs = []
    for p in ccf_regions.iterrows():
        c = p[1]
        if side_selection in ['right','both']:
            regs.append(hv.Curve(np.vstack([np.array(c.right_x)/resolution + bregma_offset[0],
                                            np.array(c.right_y)/resolution + bregma_offset[1]]).T))
        if side_selection in ['left','both']:
            regs.append(hv.Curve(np.vstack([np.array(c.left_x)/resolution + bregma_offset[0],
                                            np.array(c.left_y)/resolution + bregma_offset[1]]).T))
    # return a plot
    plot = regs[0]
    for i in range(1,len(regs)):
        plot *= regs[i]
    return plot.opts(invert_yaxis=True).options(width = 600)

def plot_aligned_overlay(img_fixed, img_to_transform, M,
                         clim=None,
                         alpha1=0.5, alpha2=0.5,
                         cmap1='gray', cmap2='Reds',
                         title='Overlay after transformation',
                         ax=None,
                         return_array=False):
    """
    Warp img_to_transform with M and overlay it transparently on img_fixed.

    Parameters
    ----------
    img_fixed : 2D array
        Target/reference image (e.g. frames_avg_flipped).
    img_to_transform : 2D array
        Image to warp and overlay (e.g. bloodvessels).
    M : callable
        SimilarityTransform object.
    alpha1, alpha2 : float
        Transparencies for base and overlay image.
    cmap1, cmap2 : str
        Matplotlib colormaps.
    title : str
        Title for the plot.
    ax : matplotlib axis or None
        Axis to draw in.
    return_array : bool
        If True, returns RGB image array (uint8).

    Returns
    -------
    ax : matplotlib axis
        Axis with plot.ax=plt.gca()
    overlay_rgb : np.ndarray (H, W, 3) if return_array=True
        Composite RGB image.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import warp
    from matplotlib.cm import get_cmap

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    if clim is None:
        # Normalize and warp the image
        img1 = (img_fixed - img_fixed.min()) / (img_fixed.max() - img_fixed.min())
        img2 = (img_to_transform - img_to_transform.min()) / (img_to_transform.max() - img_to_transform.min())
    else:
        vmin, vmax = clim
        img1 = np.clip((img_fixed - vmin) / (vmax - vmin), 0, 1)
        img2 = (img_to_transform - img_to_transform.min()) / (img_to_transform.max() - img_to_transform.min())

    img2_warped = warp(img2, inverse_map=M.inverse, output_shape=img_fixed.shape)

    # Convert grayscale to RGB using colormaps
    cmap_func1 = get_cmap(cmap1)
    cmap_func2 = get_cmap(cmap2)

    rgb1 = cmap_func1(img1)[..., :3] * alpha1
    rgb2 = cmap_func2(img2_warped)[..., :3] * alpha2
    overlay_rgb = np.clip(rgb1 + rgb2, 0, 1)

    # Plot
    ax.imshow(overlay_rgb)
    ax.set_xlim(0, img_fixed.shape[1])
    ax.set_ylim(img_fixed.shape[0], 0)
    ax.set_title(title)
    ax.set_axis_off()

    if return_array:
        return ax, (overlay_rgb * 255).astype(np.uint8)
    return ax

def allen_transform_regions(M,ccf_regions,resolution = 1,bregma_offset = [0.,0.]):
    ''' This transforms regions from the reference to image coordinates.
    Usage:

    lmarks = load_allen_landmarks('dorsal_cortex_landmarks.json')
    refregions = allen_load_reference('dorsal_cortex')
    refregions_image = allen_transform_regions(lmarks['transform'],refregions,
                                        resolution = lmarks['resolution'],
                                        bregma_offset = lmarks['bregma_offset'])

    The output: refregions_image is the same as refregions but in image space.
 '''
    nccf = ccf_regions.copy()
    for i,c in nccf.iterrows():
        for side in ['left','right']:
            x,y = apply_affine_to_points(np.array(c[side+'_x'])/resolution + bregma_offset[0],
                                         np.array(c[side+'_y'])/resolution + bregma_offset[1], M)
            nccf.at[i,side+'_x'] = x.tolist()
            nccf.at[i,side+'_y'] = y.tolist()
            x,y = apply_affine_to_points(c[side+'_center'][0]/resolution + bregma_offset[0],
                                         c[side+'_center'][1]/resolution + bregma_offset[1], M)
            nccf.at[i,side+'_center'] = [x[0],y[0]]
    return nccf

def apply_affine_to_points(x,y,M):
    '''
    x,y are vectors of points
    M is a similarity transform
    '''
    # Apply affine transform to points
    coords = np.vstack([x,y]).T
    new_coords = M(coords)
    return new_coords[:,0],new_coords[:,1]

def adjust_img_landmarks(image,landmarks,
                              landmarks_match = None,
                              bregma_offset = None,
                              resolution = 0.0194,
                              msize = 40):
    '''
    TODO: merge part of this with the one for the landmarks
    landmarks are in allen reference space
    '''
    import holoviews as hv
    from holoviews import opts, streams
    from holoviews.plotting.links import DataLink
    # h,w = image.shape
    if image.ndim == 2:
        h,w = image.shape
    elif image.ndim == 3:
        h, w = image.shape[:2]

    if bregma_offset is None:
        # then it is the center of the image
        bregma_offset = np.array([int(w/2),int(h/2)]) # place bregma in the center of the image
        
    landmarks_im = allen_landmarks_to_image_space(landmarks.copy(),bregma_offset,resolution)
    if landmarks_match is None:
        landmarks_match = landmarks_im


    # bounds = np.array([0,0,w,h])
    # im = hv.Image(image[::-1,:],
    #              bounds =tuple(bounds.tolist())).opts(
    #     invert_yaxis = True,cmap = 'gray')
    bounds = (0, 0, w, h)
    if image.ndim == 2:
        im = hv.Image(image[::-1, :], bounds=bounds).opts(cmap='gray', invert_yaxis=True)
    elif image.ndim == 3 and image.shape[2] in [3, 4]:
        im = hv.RGB(image[::-1, :, :], bounds=bounds).opts(invert_yaxis=True)
    else:
        raise ValueError("Image must be 2D grayscale or 3-channel RGB(A).")


    points = hv.Points(landmarks_match,vdims='color').opts(marker='+',size=msize)
    point_stream = streams.PointDraw(data=points.columns(), 
                                     add = False,num_objects=4, 
                                     source=points, empty_value='black')
    table = hv.Table(points, ['x', 'y','name'], 'color').opts(title='Annotation location')
    DataLink(points, table)

    from bokeh.models import HoverTool
    hoverpts = HoverTool(tooltips=[("i", "$index")])

    widget = (im*points + table).opts(
        opts.Layout(merge_tools=False),
        opts.Points(invert_yaxis=True,active_tools=['point_draw'], 
                    color='color',
                    tools=[hoverpts], 
                    width=int(w),
                    height=int(h)),
        opts.Table(editable=True))
    return widget,point_stream,landmarks_im

def hv_adjust_reference_landmarks(landmarks,ccf_regions,msize=40):
    '''
    landmarks = {'x': [-1.95, 0, 1.95, 0],
                 'y': [-3.45, -3.45, -3.45, 3.2],
                 'name': ['OB_left', 'OB_center', 'OB_right', 'RSP_base'],
                 'color': ['#fc9d03', '#0367fc', '#fc9d03', '#fc4103']}
    landmarks = pd.DataFrame(landmarks)
    # adjust landmarks
    wid,landmark_wid = hv_adjust_reference_landmarks(landmarks,ccf_regions)
    wid # to display
    # use the following to retrieve (on another cell) 
    landmarks = pd.DataFrame(landmark_wid.data)[['x','y','name','color']]
    '''
    import holoviews as hv
    from holoviews import opts, streams
    from holoviews.plotting.links import DataLink
    hv.extension('bokeh')

    referenceplt = hv_plot_allen_regions(ccf_regions).options(
        {'Curve': {'color':'black', 'width': 600}})

    points = hv.Points(landmarks,vdims='color').opts(marker='+',size=msize)
    point_stream = streams.PointDraw(data=points.columns(), 
                                     add = False,num_objects=4, 
                                     source=points, empty_value='black')
    table = hv.Table(points, ['x', 'y','name'], 'color').opts(title='Landmarks location')
    DataLink(points, table)
    widget = (referenceplt*points + table).opts(
        opts.Layout(merge_tools=False),
        opts.Points(invert_yaxis=True,
                    active_tools=['point_draw'],
                    color='color', height=500,
                    tools=['hover'], width=500),
        opts.Table(editable=True))
    return widget,point_stream

def get_ref_lmarks(image, landmarks, msize=40):
    """
    Adapted version of hv_adjust_reference_landmarks to allow selecting landmarks
    on an arbitrary image instead of the Allen atlas.

    Parameters
    ----------
    image : 2D numpy array
        The image to annotate (e.g., bloodvessels.tiff).
    landmarks : pd.DataFrame
        DataFrame with columns ['x', 'y', 'name', 'color'].
    msize : int
        Marker size for point display.

    Returns
    -------
    widget : holoviews layout
        The interactive widget layout.
    point_stream : holoviews.streams.PointDraw
        The point draw stream to retrieve adjusted landmarks.
    """
    import holoviews as hv
    from holoviews import opts, streams
    from holoviews.plotting.links import DataLink
    from bokeh.models import HoverTool
    hv.extension('bokeh')
    h, w = image.shape
    bounds = (0, 0, w, h)

    im = hv.Image(image[::-1, :], bounds=bounds).opts(
        cmap='gray',
        # clim=(-1,1),
        invert_yaxis=True
    )

    points = hv.Points(landmarks, vdims='color').opts(marker='+', size=msize)
    point_stream = streams.PointDraw(data=points.columns(),
                                     add=False, num_objects=len(landmarks),
                                     source=points, empty_value='black')

    table = hv.Table(points, ['x', 'y', 'name'], 'color').opts(title='Landmarks location')
    DataLink(points, table)

    widget = (im * points + table).opts(
        opts.Layout(merge_tools=False),
        opts.Points(invert_yaxis=True,
                    active_tools=['point_draw'],
                    color='color', height=h, width=w,
                    tools=[HoverTool(tooltips=[('name', '@name')])]),
        opts.Table(editable=True)
    )
    return widget, point_stream

def hv_adjust_exp_landmarks(image,landmarks,
                              landmarks_match = None,
                              bregma_offset = None,
                              resolution = 0.0194,
                              msize = 40):
    '''
    TODO: merge part of this with the one for the landmarks
    landmarks are in allen reference space
    '''
    h,w = image.shape
    if bregma_offset is None:
        # then it is the center of the image
        bregma_offset = np.array([int(w/2),int(h/2)]) # place bregma in the center of the image
        
    landmarks_im = allen_landmarks_to_image_space(landmarks.copy(),bregma_offset,resolution)
    if landmarks_match is None:
        landmarks_match = landmarks_im
    import holoviews as hv
    from holoviews import opts, streams
    from holoviews.plotting.links import DataLink

    bounds = np.array([0,0,w,h])
    im = hv.Image(image[::-1,:],
                 bounds =tuple(bounds.tolist())).opts(
        invert_yaxis = True,cmap = 'gray')

    points = hv.Points(landmarks_match,vdims='color').opts(marker='+',size=msize)
    point_stream = streams.PointDraw(data=points.columns(), 
                                     add = False,num_objects=4, 
                                     source=points, empty_value='black')
    table = hv.Table(points, ['x', 'y','name'], 'color').opts(title='Annotation location')
    DataLink(points, table)

    from bokeh.models import HoverTool
    hoverpts = HoverTool(tooltips=[("i", "$index")])

    widget = (im*points + table).opts(
        opts.Layout(merge_tools=False),
        opts.Points(invert_yaxis=True,active_tools=['point_draw'], 
                    color='color',
                    tools=[hoverpts], 
                    width=int(w),
                    height=int(h)),
        opts.Table(editable=True))
    return widget,point_stream,landmarks_im


def hv_adjust_exp_landmarks_v2(image, landmarks,
                                ref_image, ref_landmarks,
                                landmarks_match=None,
                                bregma_offset=None,
                                resolution=0.0194,
                                msize=40):
    '''
    Side-by-side version of hv_adjust_exp_landmarks.

    Shows ref_image (e.g. bloodvessels) with static ref_landmarks on the left,
    and image (e.g. blueRef) with interactive editable landmarks on the right.
    This helps mark the same anatomical points across both images.

    Parameters
    ----------
    image : 2D numpy array
        The experimental image to annotate (e.g. blueRef).
    landmarks : pd.DataFrame or dict
        Starting landmark positions for the editable panel (pixel coords, same
        format as hv_adjust_exp_landmarks).
    ref_image : 2D numpy array
        The reference image shown as a static background (e.g. bloodvessels).
    ref_landmarks : pd.DataFrame
        DataFrame with columns [x, y, name, color] already in pixel space of
        ref_image (e.g. vessels_lmarks_ref from get_ref_lmarks).
    landmarks_match : pd.DataFrame, optional
        Override starting positions for the editable points.
    bregma_offset : array-like, optional
        Bregma offset in pixels; defaults to center of image.
    resolution : float
        Resolution (same units as landmarks x/y).
    msize : int
        Marker size for both panels.

    Returns
    -------
    widget : holoviews Layout
        Side-by-side layout: [ref panel | edit panel | table].
    point_stream : holoviews.streams.PointDraw
        Stream to retrieve adjusted landmark positions.
    landmarks_im : pd.DataFrame
        Landmarks converted to image (pixel) space.

    Usage
    -----
    scan_wid, lmark_wid_im, lmarks_im_vessels = wfh.hv_adjust_exp_landmarks_v2(
        image=blueRef,
        landmarks=vessels_lmarks_ref,
        ref_image=bloodvessels,
        ref_landmarks=vessels_lmarks_ref,
        bregma_offset=bregma_offset,
        resolution=resolution)
    scan_wid
    # retrieve on a later cell:
    landmarks_vessels_match = pd.DataFrame(lmark_wid_im.data)
    '''
    import holoviews as hv
    from holoviews import opts, streams
    from holoviews.plotting.links import DataLink
    from bokeh.models import HoverTool
    hv.extension('bokeh')

    # --- Editable panel (right): experimental image (e.g. blueRef) ---
    h, w = image.shape
    if bregma_offset is None:
        bregma_offset = np.array([int(w / 2), int(h / 2)])

    landmarks_im = allen_landmarks_to_image_space(landmarks.copy(), bregma_offset, resolution)
    if landmarks_match is None:
        landmarks_match = landmarks_im

    def _norm(arr):
        lo, hi = np.nanpercentile(arr, (1, 99))
        return np.clip((arr.astype(float) - lo) / (hi - lo + 1e-12), 0, 1)

    image_norm = _norm(image)
    ref_image_norm = _norm(ref_image)

    bounds_edit = tuple(np.array([0, 0, w, h]).tolist())
    im_edit = hv.Image(image_norm[::-1, :], bounds=bounds_edit).opts(
        invert_yaxis=True, cmap='gray', clim=(0, 1),
        width=int(w), height=int(h))

    hover_edit = HoverTool(tooltips=[('name', '@name'), ('i', '$index')])
    pts_edit = hv.Points(landmarks_match, vdims='color').opts(
        marker='+', size=msize,
        invert_yaxis=True,
        active_tools=['point_draw'],
        color='color',
        tools=[hover_edit],
        width=int(w), height=int(h))

    point_stream = streams.PointDraw(data=pts_edit.columns(),
                                     add=False, num_objects=4,
                                     source=pts_edit, empty_value='black')
    table = hv.Table(pts_edit, ['x', 'y', 'name'], 'color').opts(title='Annotation location')
    DataLink(pts_edit, table)

    # --- Static reference panel (left): reference image (e.g. bloodvessels) ---
    h_ref, w_ref = ref_image.shape
    bounds_ref = (0, 0, w_ref, h_ref)
    im_ref = hv.Image(ref_image_norm[::-1, :], bounds=bounds_ref).opts(
        invert_yaxis=True, cmap='gray', clim=(0, 1),
        width=int(w_ref), height=int(h_ref))

    hover_ref = HoverTool(tooltips=[('name', '@name')])
    pts_ref = hv.Points(ref_landmarks, vdims='color').opts(
        marker='+', size=msize,
        invert_yaxis=True,
        active_tools=['wheel_zoom'],
        color='color',
        tools=[hover_ref],
        width=int(w_ref), height=int(h_ref))

    widget = (im_ref * pts_ref + im_edit * pts_edit + table).opts(
        opts.Layout(merge_tools=False),
        opts.Table(editable=True))

    return widget, point_stream, landmarks_im


def allen_landmarks_to_image_space(landmarks,
                                   bregma_offset = np.array([0,0]),
                                   resolution = 0.01):
    '''
    Convert landmarks from allen to "image" space.
    Basically just divides by the resolution and adds the bregma offset.

    Warning this does the operation in place, pass .copy()

    landmarks = allen_landmarks_to_image_space(landmarks.copy(),
                                   bregma_offset = np.array([0,0]),
                                   resolution = 0.01)
    '''
    landmarks['x'] = landmarks['x']/resolution + bregma_offset[0]
    landmarks['y'] = landmarks['y']/resolution + bregma_offset[1]
    return landmarks


def load_vfs(expt_dir, blocks=[1,2,3,4,5], save_dir=None):
    import scipy
    variable_names = [
    'PhaseMapHor1', 'PhaseMapHor2', 'PhaseMapVer1', 'PhaseMapVer2', 'PhaseMapHor', 'PhaseMapVer',
    'ScaledPhaseMapHor', 'ScaledPhaseMapVer', 'ScaledPhaseMapHorBlack', 'ScaledPhaseMapVerBlack',
    'HorRetinotopy', 'VerRetinotopy', 'HorVerRetinotopy',
    'HorRetinotopyBVoverlay', 'VerRetinotopyBVoverlay', 'HorVerRetinotopyBVoverlay',
    'HorRetinotopyBLoverlay', 'VerRetinotopyBLoverlay', 'HorVerRetinotopyBLoverlay',
    'ElePowerMap', 'AziPowerMap', 'PhaseHor', 'PhaseVer', 'VFS'
    ]
        # Load data from the first block to get the proper shapes
    first_block = 1
    first_mat_file_path = os.path.join(expt_dir, f'Block{first_block}', f'Wave_maps_Block{first_block}.mat')
    first_mat_data = scipy.io.loadmat(first_mat_file_path)

    # Initialize all_data based on the shape of the data from the first block
    all_data = {variable: np.zeros_like(first_mat_data[variable]) for variable in variable_names}

    # num_blocks = int(input('How many blocks do you want to average?'))
    num_blocks = len(blocks)
    # for block in range(1, num_blocks + 1):
    for block in blocks:
        mat_file_path = os.path.join(expt_dir,f'Block{block}',f'Wave_maps_Block{block}.mat')
        mat_data = scipy.io.loadmat(mat_file_path)

        # Accumulate data across blocks
        for key in all_data.keys():
            if all_data[key].shape == mat_data[key].shape:
                all_data[key] += mat_data[key]
            else:
                # Handle broadcasting for different shapes
                all_data[key] += mat_data[key][:, :, None]

        print(f'Analyzed Block {block}')
    # Average data across blocks
    avg_data = {key: value / num_blocks for key, value in all_data.items()}

    # Save the averaged data to Wave_maps.mat
    
    try:
        output_file_path = os.path.join(expt_dir, 'Wave_maps.mat')
        if not os.path.exists(output_file_path):
            
            scipy.io.savemat(output_file_path, avg_data)
            print(f'Averaged data saved to {output_file_path}')
        else:
            mat_data = scipy.io.loadmat(output_file_path)
            print('Loaded from file')
    except Exception as e:
        os.makedirs(save_dir, exist_ok=True)
        
        output_file_path = os.path.join(save_dir, 'Wave_maps.mat')
        if not os.path.exists(output_file_path):
            
            scipy.io.savemat(output_file_path, avg_data)
            print(f'Averaged data saved to {output_file_path}')
        else:
            mat_data = scipy.io.loadmat(output_file_path)
            print('Loaded from file')
    return mat_data['VFS']

def traces_by_region(ccf_regions_im, stack, fov_mask, thr=0.3):
    """
    Extract calcium traces for all visible atlas regions based on their contours and FOV mask.

    Parameters
    ----------
    ccf_regions_im : DataFrame
        Transformed atlas contours with columns ['acronym', 'left_x', 'left_y', 'right_x', 'right_y'].
    stack : object
        Widefield data object that implements `get_timecourse(np.where(...))` and has shape `stack.shape[1:]`.
    fov_mask : 2D boolean array
        Binary mask indicating valid FOV pixels.
    thr : float
        Percentage of a given region present in the fov to be included

    Returns
    -------
    ca_data : pd.DataFrame
        DataFrame with timecourses. Columns are region names (e.g. 'VISp_left'), rows = frames.
    """
    import numpy as np
    import pandas as pd
    from skimage.draw import polygon

    def mask_from_contour(region_row, shape, side='left'):
        xs = np.array(region_row[f'{side}_x'])
        ys = np.array(region_row[f'{side}_y'])
        mask = np.zeros(shape, dtype=bool)
        rr, cc = polygon(ys, xs, shape=shape)
        mask[rr, cc] = True
        return mask

    H, W = stack.shape[1:]
    traces = {}
    fractions = {}
    pixels = {}
    for _, row in ccf_regions_im.iterrows():
        acronym = row['acronym']
        for side in ['left', 'right']:
            if side == 'left':
                side_ = 'l'
            elif side == 'right':
                side_ = 'r'
            mask_region = mask_from_contour(row, shape=(H, W), side=side)
            mask_region_fov = mask_region & fov_mask
            pixel = np.count_nonzero(mask_region_fov) 
            fraction_in_fov = pixel/ np.count_nonzero(mask_region)

            if fraction_in_fov >= thr:  # or 0.3, depending on your tolerance
                trace = stack.get_timecourse(np.where(mask_region_fov))
                traces[f"{acronym}_{side_}"] = trace.mean(axis=0)
                fractions[f"{acronym}_{side_}"] = fraction_in_fov
                pixels[f"{acronym}_{side_}"] = pixel
                # print(f"{acronym}_{side_} loaded")
            else:
                print(f"{acronym}_{side_} not present")

    ca_data = pd.DataFrame(traces)
    return ca_data, fractions, pixels

def apply_pixelwise_svd(U,SVT,func, nchunks = 1024):
    '''
    Map a function to every pixel by reconstructing the SVD 

    Usage:
        variance_map = apply_pixelwise_svd(U, SVT, partial(np.nanvar,axis=1), nchunks = 1024)
        
        mean_map = apply_pixelwise_svd(U, SVT, partial(np.nanmean,axis=1), nchunks = 1024)

    Joao Couto - wfield, 2021
    '''
    dims = U.shape

    U = U.reshape([-1,dims[-1]])
    npix = U.shape[0]

    idx = np.array_split(np.arange(0,npix),nchunks)

    res = runpar(_apply_function_single_pix,
                 [U[ind,:] for ind in idx],
                 SVT=SVT,
                 func = func)
    res = np.hstack(res).astype('float32')
    U = U.reshape(dims)
    return res.reshape(dims[:2])

def reconstruct(u, svt, dims=None):
    '''
    Reconstruct a decomposed signal (e.g. decomposed with SVD).
    
    Args:
        u: spatial components - shape (pixels, components) when flattened
        svt: temporal components - shape (components,) for single frame OR (components, n_frames) for multiple
        dims: tuple of spatial dimensions (height, width)
    
    Returns:
        Single frame: (height, width)
        Multiple frames: (n_frames, height, width)
    
    Joao Couto - wfield, 2020 
    AA 2025 (modified for ADAMACS compatibility)
    '''
    if issparse(u):
        if dims is None:
            raise ValueError('Supply dims = [H,W] when using sparse arrays')
    else:
        if dims is None:
            dims = u.shape[:2]
    
    # Matrix multiplication: (pixels, components) @ (components, [n_frames])
    # Result: (pixels,) for single frame OR (pixels, n_frames) for multiple frames
    reconstructed = u @ svt
    
    # Determine if single frame or multiple frames based on SVT dimensions
    if svt.ndim == 1:
        # Single frame case
        # (pixels,) → (height, width)
        return reconstructed.reshape(dims)
    else:
        # Multiple frames case
        # (pixels, n_frames) → (height, width, n_frames) → (n_frames, height, width)
        n_frames = svt.shape[1]
        return reconstructed.reshape((*dims, n_frames)).transpose(2, 0, 1)

def im_apply_transform(im,M,dims = None):
    '''
    Applies an affine transform M to an image.
    nim = im_apply_transform(im,M)

    Joao Couto - wfield, 2020
    '''
    from skimage.transform import warp

    if issparse(im):
        # then reshape before
        if dims is None:
            raise ValueError('Provide dims when warping sparse matrices.')
        shape = im.shape
        tmp  = np.asarray(im.todense()).reshape(dims)
        tmp = warp(tmp,M,
                   order = 1,
                   mode='constant',
                   cval = 0,
                   clip = True,
                   preserve_range = True)
        return csr_matrix(tmp.reshape(shape))
    else:    
        return warp(im,M,
                    order = 1,
                    mode='constant',
                    cval = 0,
                    clip=True,
                    preserve_range=True)
def runpar(f,X,nprocesses = None,**kwargs):
    ''' 
    res = runpar(function,          # function to execute
                 data,              # data to be passed to the function
                 nprocesses = None, # defaults to the number of cores on the machine
                 **kwargs)          # additional arguments passed to the function (dictionary)

    Joao Couto - wfield, 2020

    '''
    from multiprocessing import context, reduction as reducer
    from multiprocessing.context import (
        AuthenticationError as AuthenticationError,
        BufferTooShort as BufferTooShort,
        Process as Process,
        ProcessError as ProcessError,
        TimeoutError as TimeoutError,
    )
    from multiprocessing.process import (
        active_children as active_children,
        current_process as current_process,
        parent_process as parent_process,
    )
    from multiprocessing import context, reduction as reducer

    cpu_count = context._default_context.cpu_count
    Pool = context._default_context.Pool

    if nprocesses is None:
        nprocesses = cpu_count()
    with Pool(initializer = parinit, processes=nprocesses) as pool:
        res = pool.map(partial(f,**kwargs),X)
    pool.join()
    return res         
def _apply_function_single_pix(U,SVT,func):
    '''
    Apply a function to a single pixel (helper to run in parallel)

    Joao Couto - wfield, 2021
    '''
    return func(U@SVT)

def parinit():
    import os
    os.environ['MKL_NUM_THREADS'] = "1"
    os.environ['OMP_NUM_THREADS'] = "1"

class SVDStack(object):
    def __init__(self, U, SVT, dims = None,
                 warped = None,
                 M = None, dtype = 'float32',nchunks = 1054):
        '''
stack = SVDStack(U,SVT)

Treat a decomposed dataset like a numpy array.
Args:
        - U: spatial components # (height, width, components) 
        - SVT: temporal components # (components, frames) 
        - dims: dimensions of the dataset [H,W] (for loading sparse matrices)
        - warped: warped spatial components (e.g. aligned to a reference atlas)
        - M: transform to warp spatial components
        - dtype: cast to this datatype
        - nchunks: number of chunks for pixelwise analysis

        Joao Couto - wfield, 2020
        '''
        self.U = U.astype(dtype)
        self.SVT = SVT.astype(dtype)
        self.nchunks = nchunks
        self.issparse = False
        if issparse(U):
            self.issparse = True
            if dims is None:
                raise ValueError('Supply dims = [H,W] when using sparse arrays')
            self.Uflat = self.U
        else:
            if dims is None:
                dims = U.shape[:2]
            self.Uflat = self.U.reshape(-1,self.U.shape[-1])
        self.U_warped = warped
        self.warped = False
        self.M = M
        self.shape = [SVT.shape[1],*dims]
        self.dtype = dtype
        self.originalU = None
        
    def set_warped(self,value,M = None):
        ''' Apply affine transform to the spatial components '''
        if not M is None:
            self.M = M
        if self.originalU is None:
            self.originalU = self.U.copy()
        if not value:
            self.U = self.originalU
            self.warped = False
        else:
            if self.U_warped is None:
                if not self.M is None:
                    if not self.issparse:
                        if self.originalU is None:
                            self.originalU = self.U.copy()
                        self.U_warped = self.originalU.copy()
                        self.U_warped[:,0,:] = 1e-10
                        self.U_warped[0,:,:] = 1e-10
                        self.U_warped[-1,:,:] = 1e-10
                        self.U_warped[:,-1,:] = 1e-10
                        self.U_warped = np.stack(runpar(im_apply_transform,
                                                        self.U_warped.transpose([2,0,1]),
                                                        M = self.M)).transpose([1,2,0]).astype(np.float32)
            if not self.U_warped is None:
                self.U = self.U_warped
                self.warped = True
        self.Uflat = self.U.reshape(-1,self.U.shape[-1])

    def mean(self):
        '''Pixelwise mean of the stack '''
        return apply_pixelwise_svd(self.U, self.SVT, partial(np.nanmean,axis=1), nchunks = self.nchunks)

    def var(self):
        '''Pixelwise variance of the stack '''
        return apply_pixelwise_svd(self.U, self.SVT, partial(np.nanvar,axis=1), nchunks = self.nchunks)

    def std(self):
        '''Pixelwise standard deviation of the stack '''
        return apply_pixelwise_svd(self.U, self.SVT, partial(np.nanstd,axis=1), nchunks = self.nchunks)
    
    def __len__(self):
        return self.SVT.shape[1]
    
    def __getitem__(self,*args):
        ndims  = len(args)
        if type(args[0]) is slice:
            idxz = range(*args[0].indices(self.shape[0]))
        else:
            idxz = args[0]      
        return reconstruct(self.Uflat,self.SVT[:,idxz],dims = self.shape[1:])
    
    def get_timecourse(self,xy):
        ''' Get a timecourse for the specified indices. 
        Index are in xy, like what np.where(mask) returns
        timecourse = get_timecourse([x,y])

        or 

        timecourse = np.nanmean(get_timecourse([x,y]),axis = 1)
        '''
        x = np.array(np.clip(xy[0],0,self.shape[1]),dtype=int)
        y = np.array(np.clip(xy[1],0,self.shape[2]),dtype=int)
        idx = np.ravel_multi_index((x,y),self.shape[1:])
        t = self.Uflat[idx,:]@self.SVT
        return t

def get_retinotopic_sessid(retinotopic_folder):
    """
    Convert datetime string (YYMMDD_HHMMSS) to base36 timestamp.
    
    Args:
        date_str: String in format 'YYMMDD_HHMMSS' (e.g., '250819_154750')
    
    Returns:
        str: 8-character Base-36 ID
    """
    from datetime import datetime
    
    parts = retinotopic_folder.split('_')
    date_str = f"{parts[-2]}_{parts[-1]}"
    # Parse the datetime string
    dt = datetime.strptime(date_str, '%y%m%d_%H%M%S')
    
    # Calculate days since MATLAB epoch (Jan 1, 0000)
    matlab_epoch = datetime(1, 1, 1)
    days_since_epoch = (dt - matlab_epoch).total_seconds() / 86400
    
    # Scale to match MATLAB precision (10^6)
    time_serial = int(days_since_epoch * 1e6)
    
    # Convert to Base-36
    def to_base36(num):
        chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        result = ''
        while num > 0:
            result = chars[num % 36] + result
            num //= 36
        return result.zfill(8)
    
    return f'sess{to_base36(time_serial)}'


# ==============================================================================
# FOV MASK CREATION
# ==============================================================================

def make_fov_mask(U, comp=0, threshold=0.4, sigma=40, op='greater'):
    """Create FOV brain mask from SVD spatial component.

    Args:
        U: Spatial components (height, width, n_components)
        comp: Component index (default: 0)
        threshold: Normalized intensity threshold (default: 0.8)
        sigma: Standard deviation for Gaussian filter (default: 40)
        op: Comparison operator ('greater' or 'less') (default: 'greater')

    Returns:
        bool ndarray (height, width)
    """
    from scipy.ndimage import binary_fill_holes, gaussian_filter
    from scipy.ndimage import binary_closing, binary_opening
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_erosion, remove_small_objects
    import matplotlib.pyplot as plt
    U_norm = (U[:,:,comp] - U[:,:,comp].min()) / (U[:,:,comp].max() - U[:,:,comp].min())
    
    if op == 'greater':
        mask_ = U_norm > threshold
    elif op == 'less':
        mask_ = U_norm < threshold
    else:
        raise ValueError("op must be 'greater' or 'less'")  
    plt.imshow(mask_, cmap='gray');
    plt.title(f'Initial mask (comp={comp}, thr={threshold})');
    plt.show()
    mask_smooth = binary_closing(mask_, iterations=3)
    mask_smooth = binary_opening(mask_smooth, iterations=2)
    mask_smooth = remove_small_objects(mask_smooth, min_size=100)
    mask_eroded = binary_erosion(mask_smooth)
    contour = mask_smooth & ~mask_eroded
    labeled = label(contour)
    regions = regionprops(labeled)
    largest_region = max(regions, key=lambda x: x.area)
    contour_clean = labeled == largest_region.label
    brain_mask = binary_fill_holes(contour_clean)
    brain_mask_smooth = gaussian_filter(brain_mask.astype(float), sigma=sigma) > 0.5
    brain_masked = np.where(brain_mask_smooth, U_norm, np.nan)
    plt.imshow(brain_masked, cmap='gray');
    plt.title(f'Final FOV mask (sigma={sigma})');
    plt.show()
    return brain_mask_smooth


# ==============================================================================
# DATABASE FETCH AND PLOT
# ==============================================================================

def fetch_plot_vfs_masked(scan_key, allen_ref_dir):
    """Fetch masks and alignment from database, plot VFS with CCF overlay.

    Reproduces the final plot_vfs_masked call using only data stored in
    wfield schema tables: FOVMask, VesselsTransform, CcfAlignment.

    Args:
        scan_key: dict with scan primary key (session_id, scan_id)
        allen_ref_dir: Path to Allen reference files directory
    """
    from ..schemas import wfield as wf_schema
    from skimage.transform import SimilarityTransform

    fov_mask = (wf_schema.FOVMask & scan_key).fetch1('fov_mask')
    vessels_data = (wf_schema.VesselsTransform & scan_key).fetch1()
    ccf_data = (wf_schema.CcfAlignment & scan_key).fetch1()

    vfs_vessels = vessels_data['vfs_vessels']
    M = SimilarityTransform(np.array(ccf_data['transform_params']))
    resolution = ccf_data['resolution']
    bregma_offset = np.array(ccf_data['bregma_offset'])
    ccf_regions_im = pd.DataFrame(ccf_data['ccf_regions_im'])

    reference_name = (wf_schema.CcfLandmarks &
                      f'ccf_landmarks_id={ccf_data["ccf_landmarks_id"]}').fetch1('reference_name')
    ccf_regions, _, _ = allen_load_reference(reference_name, annotation_dir=allen_ref_dir)

    plot_vfs_masked(
        image=vfs_vessels,
        fov_mask=fov_mask,
        ccf_regions=ccf_regions,
        ccf_regions_im=ccf_regions_im,
        M=M,
        resolution=resolution,
        bregma_offset=bregma_offset,
    )


def pjoin(*args):
    """Alias for os.path.join."""
    return os.path.join(*args)


# ==============================================================================
# PYTHON SVD PROCESSING (matches MATLAB run_SVD.m exactly)
# ==============================================================================

def _matlab_fftfreq(n):
    """
    Construct FFT frequency vector matching MATLAB convention:
      [0:floor(n/2), -floor((n-1)/2):-1] / n

    Differs from np.fft.fftfreq at the Nyquist bin for even n
    (MATLAB: +0.5, numpy: -0.5).
    """
    pos = np.arange(0, n // 2 + 1)
    neg = np.arange(-((n - 1) // 2), 0)
    return np.concatenate([pos, neg]).astype(np.float32) / n


def _fsvd(A, k, i=1, use_power_method=False):
    """
    Fast Randomized SVD — exact port of MATLAB fsvd.m (Halko et al. 2010).

    Uses the Krylov method by default (not power iteration), matching
    MATLAB's default ``i = 1`` with ``usePowerMethod = false``.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Input matrix (typically pixels × timepoints).
    k : int
        Number of singular components to keep.
    i : int
        Number of Krylov / power iterations (default 1).
    use_power_method : bool
        If True, use power iteration instead of Krylov (matches MATLAB
        ``usePowerMethod = true``).

    Returns
    -------
    U : ndarray (m, k)
    S : ndarray (k, k)  — diagonal matrix of singular values
    V : ndarray (n, k)
    """
    m, n = A.shape
    l = k + 2

    G = np.random.randn(n, l).astype(A.dtype)

    if use_power_method:
        H = A @ G
        for _ in range(i):
            H = A @ (A.T @ H)
    else:
        H_list = [A @ G]
        for _ in range(i):
            H_list.append(A @ (A.T @ H_list[-1]))
        H = np.concatenate(H_list, axis=1)

    Q, _ = np.linalg.qr(H, mode='reduced')
    T = A.T @ Q

    # MATLAB: [Vt, St, W] = svd(T, 'econ')  → T = Vt * St * W'
    # numpy:  U_s, s, Vh   = svd(T, False)   → T = U_s * diag(s) * Vh
    U_s, s, Vh = np.linalg.svd(T, full_matrices=False)

    Ut = Q @ Vh.T          # Q * W  in MATLAB notation
    U = Ut[:, :k]
    V = U_s[:, :k]          # Vt(:,1:k)
    S = np.diag(s[:k])

    return U, S, V


def _array_shrink(data, mask):
    """
    Port of MATLAB arrayShrink.m (merge mode only).

    Removes pixels marked True in *mask* from the first two spatial
    dimensions.  ``data`` can be (H, W) or (H, W, ...) and *mask* is
    (H, W) boolean where True = pixel to REMOVE.

    Returns array with first dimension = number of kept pixels.
    """
    kept = ~mask.ravel()
    extra_shape = data.shape[2:]
    flat = data.reshape(-1, *extra_shape)
    return flat[kept]


def _smooth_widefield(data, frame_cnt, sRate, high_cut):
    """
    Port of the ``smoothWidefield`` nested function inside
    ``cc_SvdHemoCorrect.m``.

    Applies a low-pass Butterworth filter (order 4) per trial segment
    with 10 sample edge padding to reduce ringing.

    Parameters
    ----------
    data : ndarray (nFrames, nComponents)
    frame_cnt : int
        Total number of frames (single trial assumed).
    sRate : float
        Per-channel sampling rate (Hz).
    high_cut : float
        Low-pass cutoff frequency (Hz).

    Returns
    -------
    data : ndarray with same shape, filtered in-place.
    """
    from scipy.signal import butter, filtfilt

    b, a = butter(4, high_cut / sRate, btype='low')

    # MATLAB treats frameCnt as a vector of trial lengths.
    # We support only a single segment (the common case).
    if np.isscalar(frame_cnt):
        frame_cnt = [frame_cnt]

    idx_start = 0
    for trial_len in frame_cnt:
        idx_end = idx_start + trial_len
        c_idx = slice(idx_start, idx_end)
        c_data = data[c_idx].copy()

        # Identify valid (non-NaN) rows
        nan_mask = ~np.isnan(c_data[:, 0])
        c_valid = c_data[nan_mask]

        if c_valid.shape[0] < 2:
            idx_start = idx_end
            continue

        # Pad 10 samples on each end (replicate edge), matching MATLAB
        padded = np.concatenate(
            [np.tile(c_valid[0:1], (10, 1)),
             c_valid,
             np.tile(c_valid[-1:], (10, 1))],
            axis=0,
        )

        filtered = filtfilt(b, a, padded.astype(np.float64), axis=0).astype(np.float32)
        data[c_idx][nan_mask] = filtered[10:-10]
        idx_start = idx_end

    return data


def _apply_fourier_shift(frame, dy, dx, diffphase=0.0):
    """
    Apply sub-pixel shift in Fourier domain, matching MATLAB run_SVD.m
    "apply saved shifts" branch exactly.

    After shifting, applies Gaussian blur (sigma=1, 3×3) and clips to
    [0, 65535].
    """
    import cv2

    frame = frame.astype(np.float32)
    nr, nc = frame.shape

    frame_fft = np.fft.fft2(frame)

    Nr = _matlab_fftfreq(nr)
    Nc = _matlab_fftfreq(nc)
    Nc_grid, Nr_grid = np.meshgrid(Nc, Nr)

    phase_ramp = np.exp(1j * 2 * np.pi * (dy * Nr_grid + dx * Nc_grid))
    Greg = frame_fft * phase_ramp * np.exp(1j * diffphase)

    registered = np.real(np.fft.ifft2(Greg)).astype(np.float32)
    registered = cv2.GaussianBlur(
        registered, ksize=(3, 3), sigmaX=1,
        borderType=cv2.BORDER_REFLECT,
    )
    return np.clip(registered, 0, 65535)


def run_svd_python(
    imaging_file: str,
    output_dir: str,
    scan_id: str,
    sRate: float,
    dimCnt: int = 200,
    dType: str = 'uint16',
    normalization_method: str = 'global_mean',
    proc_id: str = None,
    registered_file: str = None,
    transposed_file: str = None,
    mask_file: str = None,
    window_sec: float = None,
    percentile_val: int = None,
    trials_csv: str = None,
    random_seed: int = 0,
    enable_highpass: bool = True,
    enable_lowpass: bool = True,
    lowCut: float = 0.1,
    highCut: float = 10.0,
    smooth_blue: bool = False,
    max_frames: int = None,
    brain_mask: 'np.ndarray | None' = None,
):
    """
    Full Python SVD pipeline matching MATLAB ``run_SVD.m`` exactly.

    Replicates every step of the MATLAB pipeline:
      1. Load raw imaging data (.dat / .tif)
      2. Apply registration shifts (if ``*_shifts.npz`` or ``*_shifts.mat``
         found in *output_dir*)
      3. De-interleave into blue (neural) and violet (hemodynamic) channels
      4. Normalise (dF/F₀) using the specified method
      5. Apply brain mask (optional)
      6. Concatenate blue + hemo → randomised SVD (``fsvd``)
      7. Project temporal components (``pinv(U) @ data``)
      8. Hemodynamic correction (``cc_SvdHemoCorrect``)
      9. Save results (.npz and MATLAB v7.3 .mat)

    Parameters
    ----------
    imaging_file : str
        Path to the raw imaging file (.dat or .tif).
    output_dir : str
        Directory where shifts files live and results will be saved.
    scan_id : str
        Scan identifier (e.g. 'scan9FW5OVL6').
    sRate : float
        **Total** acquisition rate in Hz (both channels combined,
        e.g. 30 Hz for 15 Hz per channel).  Matches the ``sRate``
        variable passed to MATLAB.
    dimCnt : int
        Number of SVD components to keep (default 200).
    dType : str
        Data type of the raw file (default 'uint16').
    normalization_method : str
        One of 'global_mean', 'moving_avg', 'percentile', 'trial_based'.
    proc_id : str or None
        8-char processing ID.  Generated automatically if None.
    mask_file : str or None
        Path to brain mask (.npy or .mat with 'brain_mask' key).
    window_sec : float or None
        Window length in seconds (for 'moving_avg' / 'percentile').
    percentile_val : int or None
        Percentile for the 'percentile' method.
    trials_csv : str or None
        Path to trials CSV with 'onset_idx' and 'offset_idx' columns
        (for 'trial_based' normalisation).
    random_seed : int
        Seed for the randomised SVD (default 0 for reproducibility).
    enable_highpass : bool
        Apply high-pass filter in hemodynamic correction.
    enable_lowpass : bool
        Apply low-pass filter to violet channel in hemo correction.
    lowCut : float
        High-pass cutoff (Hz).
    highCut : float
        Low-pass cutoff (Hz).
    smooth_blue : bool
        Whether to also low-pass filter the blue channel.

    Returns
    -------
    dict with keys:
        'U'               : ndarray (height, width, nComponents)
        'newVc'           : ndarray (nComponents, nFrames)
        'blueV'           : ndarray (nComponents, nFrames)
        'hemoV'           : ndarray (nComponents, nFrames)
        'regC'            : ndarray (nBrainPixels,)
        'T'               : ndarray (nComponents, nComponents)
        'hemoVar_global'  : float  (% variance explained by hemo)
        'hemoVar_per_pixel': ndarray (nBrainPixels,)
        'variance_explained_percent' : float
        'output_file'     : str (path to saved .npz)
    """
    import time
    import cv2
    from scipy.signal import butter, filtfilt
    from scipy.io import loadmat, savemat
    from tqdm import tqdm

    t_start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if proc_id is None:
        proc_id = generate_proc_id()

    sRate_chn = sRate / 2.0  # per-channel rate

    print(f"\n{'='*60}")
    print(f"  PYTHON WIDEFIELD SVD PROCESSING")
    print(f"{'='*60}")
    print(f"  File       : {imaging_file}")
    print(f"  scan_id    : {scan_id}")
    print(f"  proc_id    : {proc_id}")
    print(f"  sRate      : {sRate} Hz total, {sRate_chn} Hz per channel")
    print(f"  dimCnt     : {dimCnt}")
    print(f"  norm       : {normalization_method}")
    print(f"  random_seed: {random_seed}")
    print(f"{'='*60}\n")

    # ================================================================
    # 1. LOAD RAW IMAGING DATA
    # ================================================================
    print("=== [1/9] Loading imaging data ===")
    t0 = time.time()
    imaging_file = str(imaging_file)
    file_ext = Path(imaging_file).suffix.lower()

    if registered_file and os.path.exists(registered_file):
        # ---- Load pre-registered memmap (float32, C-order) ----
        # Still need height/width from the raw file metadata
        if file_ext == '.dat':
            meta_candidates = glob(
                pjoin(str(Path(imaging_file).parent), '*meta.json*')
            )
            if meta_candidates:
                with open(meta_candidates[0]) as f:
                    meta = json.load(f)
                height, width = meta['height'], meta['width']
            else:
                import re as _re
                m = _re.search(r'_(\d+)_(\d+)_(\d+)_uint16\.dat$', imaging_file)
                if m is None:
                    raise ValueError(
                        f"Cannot parse dimensions from filename: {imaging_file}")
                height, width = int(m[1]), int(m[2])
        elif file_ext in ('.tif', '.tiff'):
            from tifffile import TiffFile
            with TiffFile(imaging_file) as tif:
                page = tif.pages[0]
                height, width = page.shape[0], page.shape[1]
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        file_bytes = os.path.getsize(registered_file)
        all_frames = file_bytes // (4 * height * width)  # float32 = 4 bytes
        all_data = np.memmap(registered_file, dtype='float32', mode='r',
                             shape=(all_frames, height, width))
        print(f"  Loaded pre-registered data: {registered_file}")
        print(f"  Shape: {all_frames} frames, {height}x{width}")
    elif transposed_file and os.path.exists(transposed_file):
        # ---- Load pre-transposed memmap (uint16, C-order, (n_total, H, W)) ----
        # Created by compute_registration(); avoids re-reading Fortran-order .dat.
        # Shifts will be applied in Step 2 as normal.
        if file_ext == '.dat':
            meta_candidates = glob(
                pjoin(str(Path(imaging_file).parent), '*meta.json*')
            )
            if meta_candidates:
                with open(meta_candidates[0]) as f:
                    meta = json.load(f)
                height, width = meta['height'], meta['width']
            else:
                import re as _re
                m = _re.search(r'_(\d+)_(\d+)_\d+_uint16\.dat$', imaging_file)
                if m is None:
                    raise ValueError(
                        f"Cannot parse dimensions from filename: {imaging_file}")
                height, width = int(m[1]), int(m[2])
        elif file_ext in ('.tif', '.tiff'):
            from tifffile import TiffFile
            with TiffFile(imaging_file) as tif:
                page = tif.pages[0]
                height, width = page.shape[0], page.shape[1]
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        file_bytes = os.path.getsize(transposed_file)
        n_total = file_bytes // (2 * height * width)  # uint16 = 2 bytes
        raw_transposed = np.memmap(transposed_file, dtype='uint16', mode='r',
                                   shape=(n_total, height, width))
        all_data = np.memmap(str(tmp_dir / 'all_data.dat'), dtype=np.float32,
                             mode='w+', shape=(n_total, height, width))
        for _i in tqdm(range(n_total), desc="Loading transposed to float32", unit="frame"):
            all_data[_i] = raw_transposed[_i].astype(np.float32)
        all_data.flush()
        del raw_transposed
        # Free transposed file immediately — all_data.dat holds the float32 copy
        os.remove(transposed_file)
        print(f"  Loaded pre-transposed data, deleted: {Path(transposed_file).name}")
        print(f"  Shape: {n_total} frames, {height}x{width}")
    else:
        # ---- Load raw imaging data ----
        if file_ext == '.dat':
            # Try JSON meta first, then fall back to filename parsing
            meta_candidates = glob(
                pjoin(str(Path(imaging_file).parent), '*meta.json*')
            )
            if meta_candidates:
                with open(meta_candidates[0]) as f:
                    meta = json.load(f)
                height, width = meta['height'], meta['width']
                n_channels = meta['n_channels']
                shape = (height, width, n_channels, meta['n_frames'])
                raw = np.memmap(imaging_file, dtype=dType, mode='r',
                                shape=shape, order='F')
            else:
                # Parse from filename: *_HEIGHT_WIDTH_NCHANNELS_uint16.dat
                import re
                m = re.search(r'_(\d+)_(\d+)_(\d+)_uint16\.dat$', imaging_file)
                if m is None:
                    raise ValueError(
                        f"Cannot parse dimensions from filename: {imaging_file}")
                height, width, n_channels = int(m[1]), int(m[2]), int(m[3])
                file_bytes = os.path.getsize(imaging_file)
                n_frames = file_bytes // (2 * height * width * n_channels)
                shape = (height, width, n_channels, n_frames)
                raw = np.memmap(imaging_file, dtype=dType, mode='r',
                                shape=shape, order='F')

            # Allocate (allFrames, H, W) memmap and fill frame-by-frame.
            # The .dat is Fortran-order (H, W, channels, frames) so we can't
            # mmap it directly as (frames, H, W) — a transposition copy is
            # unavoidable. We clamp to max_frames here so the MaskSvd path
            # never copies more than it needs.
            if n_channels == 1:
                n_total = shape[3]
                n_total_load = min(n_total, max_frames) if max_frames else n_total
                all_data = np.memmap(str(tmp_dir / 'all_data.dat'), dtype=np.float32,
                                     mode='w+', shape=(n_total_load, height, width))
                for _i in tqdm(range(n_total_load), desc="Loading .dat to memmap", unit="frame"):
                    all_data[_i] = raw[:, :, 0, _i].astype(np.float32)
                all_data.flush()
            else:
                # Separate channels → interleave into (allFrames, H, W)
                n_per_ch = raw.shape[3]
                n_per_ch_load = min(n_per_ch, max_frames) if max_frames else n_per_ch
                all_data = np.memmap(str(tmp_dir / 'all_data.dat'), dtype=np.float32,
                                     mode='w+', shape=(n_per_ch_load * 2, height, width))
                for _i in tqdm(range(n_per_ch_load), desc="Loading .dat to memmap", unit="frame"):
                    all_data[2 * _i] = raw[:, :, 0, _i].astype(np.float32)      # violet
                    all_data[2 * _i + 1] = raw[:, :, 1, _i].astype(np.float32)  # blue
                all_data.flush()
            del raw

        elif file_ext in ('.tif', '.tiff'):
            from tifffile import TiffFile
            with TiffFile(imaging_file) as tif:
                all_data = tif.asarray(out="memmap")
            # TIFF stack: (allFrames, H, W) — already interleaved
            if all_data.ndim == 3:
                height, width = all_data.shape[1], all_data.shape[2]
            else:
                raise ValueError(f"Unexpected TIFF shape: {all_data.shape}")
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    all_frames = all_data.shape[0]
    n_frames = all_frames // 2  # per-channel frame count
    print(f"  Loaded: {height}x{width}, {all_frames} interleaved frames "
          f"({n_frames} per channel)")
    print(f"  [{time.time() - t0:.1f}s]")

    # Truncate to first max_frames per channel.
    # For the raw .dat path this is already handled above (loop is clamped).
    # For transposed_file / TIFF paths the full data is loaded so we trim here.
    if max_frames is not None and max_frames * 2 < all_frames:
        limit = max_frames * 2
        all_data = all_data[:limit]
        all_frames = limit
        n_frames = all_frames // 2
        print(f"  Truncated to first {max_frames} per-channel frames ({limit} interleaved)")

    # ================================================================
    # 2. APPLY REGISTRATION SHIFTS (if available)
    # ================================================================
    print("\n=== [2/9] Registration ===")
    t0 = time.time()
    shifts_applied = False

    if registered_file and os.path.exists(registered_file):
        # Data was already registered by compute_registration(apply=True)
        print("  Using pre-registered data — shifts already applied")
        shifts_applied = True
    else:
        # Search for shifts file matching this proc_id first, then any
        if proc_id:
            npz_files = sorted(glob(pjoin(str(output_dir), f'*{proc_id}*_shifts.npz')),
                                key=os.path.getmtime, reverse=True)
            mat_shift_files = sorted(glob(pjoin(str(output_dir), f'*{proc_id}*_shifts.mat')),
                                     key=os.path.getmtime, reverse=True)
        else:
            npz_files = []
            mat_shift_files = []

        # Fall back to any shifts file if proc_id-specific not found
        if not npz_files:
            npz_files = sorted(glob(pjoin(str(output_dir), '*_shifts.npz')),
                                key=os.path.getmtime, reverse=True)
        if not mat_shift_files:
            mat_shift_files = sorted(glob(pjoin(str(output_dir), '*_shifts.mat')),
                                     key=os.path.getmtime, reverse=True)

        if npz_files:
            print(f"  Found shifts file: {Path(npz_files[0]).name}")
            shifts_data = np.load(npz_files[0])

            # Handle new format (shifts_blue / shifts_violet) or legacy (shifts)
            if 'shifts_blue' in shifts_data:
                shifts_blue = shifts_data['shifts_blue']    # (n_per_channel, 2) [dy, dx]
                shifts_violet = shifts_data['shifts_violet']  # (n_per_channel, 2) [dy, dx]
                # Interleave back to allFrames order
                dy_shifts = np.empty(all_frames, dtype=np.float32)
                dx_shifts = np.empty(all_frames, dtype=np.float32)
                dy_shifts[0::2] = shifts_violet[:, 0]
                dy_shifts[1::2] = shifts_blue[:, 0]
                dx_shifts[0::2] = shifts_violet[:, 1]
                dx_shifts[1::2] = shifts_blue[:, 1]
            else:
                # Legacy format: single 'shifts' array (allFrames, 2)
                dy_shifts = shifts_data['shifts'][:, 0].astype(np.float32)
                dx_shifts = shifts_data['shifts'][:, 1].astype(np.float32)

            if len(dy_shifts) != all_frames:
                raise ValueError(
                    f"Shifts file has {len(dy_shifts)} frames, "
                    f"data has {all_frames}")

            print(f"  Applying Fourier-domain shifts to {all_frames} frames...")
            for i in tqdm(range(all_frames), desc="Registering", unit="frame"):
                all_data[i] = _apply_fourier_shift(
                    all_data[i].astype(np.float32), dy_shifts[i], dx_shifts[i])
            shifts_applied = True

        elif mat_shift_files:
            print(f"  Found MATLAB shifts file: {Path(mat_shift_files[0]).name}")
            sdata = loadmat(mat_shift_files[0])
            dy_shifts = sdata['row_shift'].ravel().astype(np.float32)
            dx_shifts = sdata['col_shift'].ravel().astype(np.float32)

            if len(dy_shifts) != all_frames:
                raise ValueError(
                    f"Shifts file has {len(dy_shifts)} frames, "
                    f"data has {all_frames}")

            print(f"  Applying Fourier-domain shifts to {all_frames} frames...")
            for i in tqdm(range(all_frames), desc="Registering", unit="frame"):
                all_data[i] = _apply_fourier_shift(
                    all_data[i].astype(np.float32), dy_shifts[i], dx_shifts[i])
            shifts_applied = True
        else:
            print("  No shifts file found — skipping registration")

    print(f"  [{time.time() - t0:.1f}s]")

    # ================================================================
    # 3. DE-INTERLEAVE
    # ================================================================
    print("\n=== [3/9] De-interleaving ===")
    t0 = time.time()
    # Matches MATLAB: hemoData = cData(:,:,1:2:end), blueData = cData(:,:,2:2:end)
    # Allocate memmap files for each channel instead of .copy() RAM arrays
    hemo_data = np.memmap(str(tmp_dir / 'hemo.dat'), dtype='float32', mode='w+',
                          shape=(n_frames, height, width))
    blue_data = np.memmap(str(tmp_dir / 'blue.dat'), dtype='float32', mode='w+',
                          shape=(n_frames, height, width))
    hemo_data[:] = all_data[0::2]   # (nFrames, H, W) — violet/hemo
    blue_data[:] = all_data[1::2]   # (nFrames, H, W) — blue/neural
    hemo_data.flush()
    blue_data.flush()
    del all_data
    # Free disk: remove all_data tmp file (no longer needed)
    _tmp_all = tmp_dir / 'all_data.dat'
    if _tmp_all.exists():
        os.remove(str(_tmp_all))

    # Trim to equal lengths
    n_frames = min(hemo_data.shape[0], blue_data.shape[0])
    hemo_data = hemo_data[:n_frames]
    blue_data = blue_data[:n_frames]
    print(f"  Blue: {blue_data.shape}, Hemo: {hemo_data.shape}")
    print(f"  [{time.time() - t0:.1f}s]")

    # ================================================================
    # 4. LOAD BRAIN MASK
    # ================================================================
    print("\n=== [4/9] Brain mask ===")
    t0 = time.time()
    if brain_mask is not None:
        # Array passed directly from the database — no file I/O needed
        brain_mask = np.asarray(brain_mask, dtype=bool)
        if brain_mask.shape != (height, width):
            raise ValueError(
                f"Mask shape {brain_mask.shape} != data ({height}, {width})")
        pct = 100 * brain_mask.sum() / brain_mask.size
        print(f"  Using provided mask: {pct:.1f}% brain pixels")
    elif mask_file and os.path.isfile(mask_file):
        if mask_file.endswith('.npy'):
            brain_mask = np.load(mask_file).astype(bool)
        elif mask_file.endswith('.mat'):
            mdata = loadmat(mask_file)
            brain_mask = mdata['brain_mask'].astype(bool)
        else:
            print(f"  Unknown mask format: {mask_file}, using all pixels")
            brain_mask = np.ones((height, width), dtype=bool)
        if brain_mask.shape != (height, width):
            raise ValueError(
                f"Mask shape {brain_mask.shape} != data ({height}, {width})")
        pct = 100 * brain_mask.sum() / brain_mask.size
        print(f"  Loaded mask: {pct:.1f}% brain pixels")
    else:
        brain_mask = np.ones((height, width), dtype=bool)
        print("  No mask provided — using all pixels")
    print(f"  [{time.time() - t0:.1f}s]")

    # ================================================================
    # 5. NORMALISE (dF / F₀)
    # ================================================================
    print(f"\n=== [5/9] Normalisation ({normalization_method}) ===")
    t0 = time.time()

    if normalization_method == 'global_mean':
        # Per-pixel mean across time: mean(F, axis=0) -> (H,W)
        # dF/F = (F - mean) / mean   — in-place on memmap
        blue_mean = np.nanmean(blue_data, axis=0)   # (H,W)
        blue_data -= blue_mean
        blue_data /= blue_mean
        blue_data.flush()
        hemo_mean = np.nanmean(hemo_data, axis=0)
        hemo_data -= hemo_mean
        hemo_data /= hemo_mean
        hemo_data.flush()

    elif normalization_method == 'global_percentile':
        # Per-pixel Nth percentile across ALL time as F0 (fast, single pass).
        # Better than global_mean for calcium imaging: activity events push the
        # mean up, whereas the bottom-N% captures the resting baseline only.
        # dF/F = (F - F0) / F0   where F0 = percentile(F, pct, axis=0)
        if percentile_val is None:
            raise ValueError("percentile_val required for global_percentile")
        print(f"  Global {percentile_val}th-percentile baseline per pixel")
        blue_f0 = np.percentile(blue_data, percentile_val, axis=0).astype(np.float32)  # (H,W)
        blue_data -= blue_f0
        blue_data /= blue_f0
        blue_data.flush()
        hemo_f0 = np.percentile(hemo_data, percentile_val, axis=0).astype(np.float32)
        hemo_data -= hemo_f0
        hemo_data /= hemo_f0
        hemo_data.flush()

    elif normalization_method == 'moving_avg':
        if window_sec is None:
            raise ValueError("window_sec required for moving_avg")
        window_size = round(window_sec * sRate_chn)
        print(f"  Window: {window_size} frames ({window_sec}s)")
        blue_data, hemo_data = _moving_avg_dfof(
            blue_data, hemo_data, window_size)

    elif normalization_method == 'percentile':
        if window_sec is None or percentile_val is None:
            raise ValueError(
                "window_sec and percentile_val required for percentile")
        print(f'Using sRate_chn={sRate_chn} Hz for window size calculation')
        window_size = round(window_sec * sRate_chn)  # matches MATLAB
        print(f"  Percentile: {percentile_val}%, window: {window_size} frames")
        blue_data, hemo_data = _percentile_dfof(
            blue_data, hemo_data, percentile_val, window_size)

    elif normalization_method == 'trial_based':
        if trials_csv is None:
            raise ValueError("trials_csv required for trial_based")
        trials_df = pd.read_csv(trials_csv)
        trials = trials_df[['onset_idx', 'offset_idx']].values + 1  # 0→1-based
        n_trials = len(trials)
        print(f"  {n_trials} trials from {trials_csv}")
        for start_f, end_f in trials:
            if start_f < 1 or end_f > n_frames:
                continue
            sl = slice(start_f - 1, end_f)  # back to 0-based
            b_mean = np.nanmean(blue_data[sl], axis=0, keepdims=True)
            blue_data[sl] = (blue_data[sl] - b_mean) / b_mean
            h_mean = np.nanmean(hemo_data[sl], axis=0, keepdims=True)
            hemo_data[sl] = (hemo_data[sl] - h_mean) / h_mean
    else:
        raise ValueError(f"Unknown normalisation: {normalization_method}")

    print(f"  [{time.time() - t0:.1f}s]")

    # ================================================================
    # 6. APPLY MASK (zero out non-brain pixels)
    # ================================================================
    print("\n=== [6/9] Applying brain mask ===")
    t0 = time.time()
    outside = ~brain_mask.ravel()  # flatten (H*W,): True = non-brain
    n_outside = int(outside.sum())
    n_brain = height * width - n_outside
    if n_outside > 0:
        # Reshape to pixel space (T, H*W) — no copy since data is C-contiguous.
        # Process in time-chunks so writes are sequential (cache-friendly on memmap)
        # and tqdm gives per-chunk progress.
        blue_2d = blue_data.reshape(n_frames, height * width)
        hemo_2d = hemo_data.reshape(n_frames, height * width)
        _chunk_t = 500
        for _s in tqdm(range(0, n_frames, _chunk_t),
                       desc="Applying brain mask", unit="chunk"):
            _e = min(_s + _chunk_t, n_frames)
            blue_2d[_s:_e, outside] = 0.0
            hemo_2d[_s:_e, outside] = 0.0
        blue_data.flush()
        hemo_data.flush()
        print(f"  Zeroed {n_outside:,} background pixels "
              f"({100 * n_outside / (height * width):.1f}% of frame), "
              f"{n_brain:,} brain pixels retained")
    else:
        print("  Mask is all-brain — nothing to zero")
    print(f"  [{time.time() - t0:.1f}s]")

    # Save a diagnostic image: one random frame from each channel after masking
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend, safe for workers
        import matplotlib.pyplot as plt
        rng_idx = np.random.randint(0, n_frames)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(blue_data[rng_idx], cmap='gray', aspect='equal')
        axes[0].set_title(f'Blue channel — frame {rng_idx}')
        axes[0].axis('off')
        axes[1].imshow(hemo_data[rng_idx], cmap='gray', aspect='equal')
        axes[1].set_title(f'Hemo channel — frame {rng_idx}')
        axes[1].axis('off')
        fig.suptitle(f'{scan_id}  |  post-mask diagnostic', fontsize=10)
        fig.tight_layout()
        diag_path = output_dir / f'{scan_id}_{proc_id}_masked_frame{rng_idx}.png'
        fig.savefig(str(diag_path), dpi=100)
        plt.close(fig)
        print(f"  Saved masked-frame diagnostic: {diag_path.name}")
    except Exception as _e:
        print(f"  Warning: could not save diagnostic image ({_e})")

    # ================================================================
    # 7. SVD  (matches MATLAB: cat blue+hemo → fsvd → U = cU*cS)
    # ================================================================
    print(f"\n=== [7/9] SVD ({dimCnt} components) ===")
    t0 = time.time()
    n_pixels = height * width

    # Write blue + hemo into a (2T, H*W) C-order memmap on disk.
    # Transposing to (H*W, 2T) gives a non-contiguous view that numpy/BLAS
    # handles correctly (gemm with TransA) — no extra RAM copy needed.
    combined_flat = np.memmap(str(tmp_dir / 'combined.dat'), dtype='float32',
                              mode='w+', shape=(2 * n_frames, n_pixels))
    combined_flat[:n_frames] = blue_data.reshape(n_frames, n_pixels)
    combined_flat[n_frames:] = hemo_data.reshape(n_frames, n_pixels)
    combined_flat.flush()
    combined_2d = combined_flat.T   # (H*W, 2T) — non-contiguous view, no copy

    print(f"  SVD input: {combined_2d.shape[0]} pixels x "
          f"{combined_2d.shape[1]} timepoints")

    np.random.seed(random_seed)
    k = min(dimCnt, *combined_2d.shape)
    cU, cS, _ = _fsvd(combined_2d, k)

    # U = cU * cS  (absorb singular values into spatial components)
    U_flat = (cU @ cS).astype(np.float32)          # (H*W, k)
    U = U_flat.reshape(height, width, -1)           # (H, W, k)
    del combined_2d, combined_flat, cU, cS
    _comb_path = tmp_dir / 'combined.dat'
    if _comb_path.exists():
        os.remove(str(_comb_path))

    print(f"  U shape: {U.shape}")
    print(f"  [{time.time() - t0:.1f}s]")

    # ================================================================
    # 7b. VARIANCE EXPLAINED (blue channel only, matching MATLAB)
    # ================================================================
    print("\n=== [7b/9] Variance explained ===")
    t0 = time.time()
    # pinv_U is (k, H*W).  blue_data is a memmap (T, H, W).
    # We compute blueV = pinv_U @ blue_flat as (blue_2d @ pinv_U.T).T
    # where blue_2d = blue_data.reshape(T, H*W) is a C-contiguous view — no copy.
    pinv_U = np.linalg.pinv(U_flat)                          # (k, H*W)
    blue_2d = blue_data.reshape(n_frames, n_pixels)          # (T, H*W) view
    blueV = (blue_2d @ pinv_U.T).T.astype(np.float32)       # (k, T)

    # Total variance: accumulate in chunks — avoids squaring the full (H*W,T) array
    _CHUNK_V = 200
    total_var_blue = 0.0
    for _s in range(0, n_frames, _CHUNK_V):
        _e = min(_s + _CHUNK_V, n_frames)
        total_var_blue += float(np.sum(blue_2d[_s:_e].astype(np.float64) ** 2))
    total_var_blue /= n_frames
    del blue_2d

    # Reconstruction variance via U^T U — avoids materialising (H*W, T)
    # ||U_flat @ blueV||_F^2 = trace(blueV^T @ (U_flat^T @ U_flat) @ blueV)
    #                        = sum(blueV * ((U_flat^T @ U_flat) @ blueV))
    UU      = U_flat.T @ U_flat               # (k, k) — small
    UU_bV   = UU @ blueV                      # (k, T) — small
    recon_var_blue = float(np.sum(blueV * UU_bV)) / n_frames
    variance_explained_pct = min(100.0 * recon_var_blue / total_var_blue, 100.0)
    del UU, UU_bV

    # Per-component variance: ||u_i||^2 * ||v_i||^2 / (T * total_var)
    variance_per_component = (
        np.sum(U_flat ** 2, axis=0) * np.sum(blueV ** 2, axis=1)
        / n_frames / total_var_blue * 100.0
    ).astype(np.float32)

    print(f"  Variance explained: {variance_explained_pct:.2f}% "
          f"({k} components)")
    print(f"  Top-5 per-component: {variance_per_component[:5]}")
    print(f"  [{time.time() - t0:.1f}s]")

    # ================================================================
    # 8. TEMPORAL COMPONENTS  (pinv(U) @ channel_data)
    # ================================================================
    print("\n=== [8/9] Temporal projection + Hemodynamic correction ===")
    t0 = time.time()

    # blueV already computed above: (k, T)
    # Same trick: (hemo_2d @ pinv_U.T).T avoids transposing the large memmap
    hemo_2d = hemo_data.reshape(n_frames, n_pixels)          # (T, H*W) view
    hemoV = (hemo_2d @ pinv_U.T).T.astype(np.float32)       # (k, T)
    del hemo_2d, pinv_U, blue_data, hemo_data

    blueV = blueV.astype(np.float32)
    hemoV = hemoV.astype(np.float32)

    print(f"  blueV: {blueV.shape}, hemoV: {hemoV.shape}")

    # ================================================================
    # 8b. HEMODYNAMIC CORRECTION  (port of cc_SvdHemoCorrect.m)
    # ================================================================
    # -- Transpose to (T, k) as MATLAB does internally --
    bV = blueV.T.copy()     # (T, k)
    hV = hemoV.T.copy()     # (T, k)

    # Subtract means
    bV -= np.nanmean(bV, axis=0, keepdims=True)
    hV -= np.nanmean(hV, axis=0, keepdims=True)

    # High-pass filter  (Butterworth order 2, cutoff lowCut Hz)
    if enable_highpass:
        print(f"  High-pass filter: {lowCut} Hz, order 2")
        b_hp, a_hp = butter(2, 2 * lowCut / sRate_chn, btype='high')
        valid = ~np.isnan(bV[:, 0])
        bV[valid] = filtfilt(
            b_hp, a_hp, bV[valid].astype(np.float64),
            axis=0).astype(np.float32)
        hV[valid] = filtfilt(
            b_hp, a_hp, hV[valid].astype(np.float64),
            axis=0).astype(np.float32)

    # Apply brain mask to U → remove non-brain pixels
    # MATLAB: U = arrayShrink(U, ~brain_mask, 'merge')
    mask_remove = ~brain_mask  # True = remove
    if mask_remove.any():
        U_masked = _array_shrink(U, mask_remove)   # (nBrainPix, k)
    else:
        U_masked = U_flat.copy()                    # (H*W, k)

    n_brain_pix = U_masked.shape[0]
    print(f"  U after mask: {U_masked.shape}")

    # Low-pass filter on violet (and optionally blue)
    if enable_lowpass and sRate_chn > highCut:
        print(f"  Low-pass filter: {highCut} Hz, order 4 (violet)")
        hV = _smooth_widefield(hV, n_frames, sRate_chn, highCut)
        if smooth_blue:
            bV = _smooth_widefield(bV, n_frames, sRate_chn, highCut)

    # Pixel-wise regression in chunks of 500 (matching MATLAB exactly)
    regC = np.zeros(n_brain_pix, dtype=np.float32)
    blue_var = np.zeros(n_brain_pix, dtype=np.float32)
    new_var = np.zeros(n_brain_pix, dtype=np.float32)
    chunk = 500

    for start in range(0, n_brain_pix, chunk):
        end = min(start + chunk, n_brain_pix)
        # a = pixel traces from blue: (chunk_pix, T)
        a = U_masked[start:end] @ bV.T
        # b = pixel traces from hemo: (chunk_pix, T)
        b = U_masked[start:end] @ hV.T

        # regC = sum(a*b, axis=1) / sum(b*b, axis=1)
        regC[start:end] = (
            np.nansum(a * b, axis=1) / np.nansum(b * b, axis=1))
        blue_var[start:end] = np.var(a, axis=1)
        new_var[start:end] = np.var(
            a - b * regC[start:end, None], axis=1)
    del a, b

    # Transformation matrix:  T_mat = pinv(U_masked) @ (regC[:,None] * U_masked)
    # Zero out invalid/NaN regression coefficients to prevent blowing up the pseudo-inverse
    regC[~np.isfinite(regC)] = 0.0
    T_mat = np.linalg.pinv(U_masked) @ (regC[:, None] * U_masked)  # (k, k)

    # Corrected temporal components
    newVc = bV - hV @ T_mat.T                         # (T, k)
    newVc -= np.nanmean(newVc, axis=0, keepdims=True)  # subtract mean

    # Variance explained by hemodynamics - Calculate robustly in PIXEL space
    # (Component-space sums of squares get dominated by noise amplified in high-index components)
    # The actual physical variance explained across the image is the sum of pixel variances.
    sum_blue_var = np.nansum(blue_var)
    sum_new_var = np.nansum(new_var)
    if sum_blue_var > 0:
        hemoVar_global = 100.0 * (1.0 - sum_new_var / sum_blue_var)
    else:
        hemoVar_global = 0.0

    hemoVar_per_pixel = 100.0 * (1.0 - new_var / blue_var)

    print(f"  Hemo variance explained: {hemoVar_global:.2f}%")

    # Transpose back to (k, T) convention
    newVc = newVc.T.astype(np.float32)   # (k, T)

    print(f"  newVc: {newVc.shape}")
    print(f"  [{time.time() - t0:.1f}s]")

    # ================================================================
    # 9. SAVE RESULTS
    # ================================================================
    print("\n=== [9/9] Saving ===")
    t0 = time.time()

    # Build output filenames matching MATLAB naming convention
    if normalization_method == 'moving_avg':
        tag = f"movavg{window_sec}"
    elif normalization_method == 'percentile':
        tag = f"pct{percentile_val}_win{window_sec}"
    elif normalization_method == 'global_percentile':
        tag = f"globalpct{percentile_val}"
    elif normalization_method == 'trial_based':
        tag = "trial_based"
    else:
        tag = "globalmean"

    base = f"{scan_id}_Vc_{proc_id}_{tag}"
    npz_path = output_dir / f"{base}.npz"
    mat_path = output_dir / f"{base}.mat"

    # Save Python .npz
    np.savez(
        npz_path,
        U=U,                      # (H, W, k)
        newVc=newVc,               # (k, T)
        blueV=blueV,               # (k, T)
        hemoV=hemoV,               # (k, T)
        regC=regC,                 # (nBrainPix,)
        T=T_mat,                   # (k, k)
        hemoVar_global=hemoVar_global,
        hemoVar_per_pixel=hemoVar_per_pixel,
        variance_explained_percent=variance_explained_pct,
        variance_explained_per_component=variance_per_component,
        scan_id=scan_id,
        proc_id=proc_id,
        sRate=sRate,
        dimCnt=dimCnt,
        normalization_method=normalization_method,
    )
    print(f"  Saved: {npz_path.name} "
          f"({npz_path.stat().st_size / 1e6:.1f} MB)")

    # Save MATLAB-compatible .mat (v7.3 via h5py, or scipy for smaller)
    try:
        import hdf5storage
        hdf5storage.savemat(
            str(mat_path),
            {
                'U': U,
                'newVc': newVc,
                'blueV': blueV,
                'hemoV': hemoV,
                'regC': regC,
                'T': T_mat,
                'hemoVar_global': hemoVar_global,
                'hemoVar_per_pixel': hemoVar_per_pixel,
            },
            format='7.3',
        )
        print(f"  Saved: {mat_path.name} (HDF5/v7.3)")
    except ImportError:
        savemat(
            str(mat_path),
            {
                'U': U,
                'newVc': newVc,
                'blueV': blueV,
                'hemoV': hemoV,
                'regC': regC,
                'T': T_mat,
                'hemoVar_global': hemoVar_global,
                'hemoVar_per_pixel': hemoVar_per_pixel,
            },
        )
        print(f"  Saved: {mat_path.name} (scipy v5)")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  DONE — total time: {elapsed:.1f}s")
    print(f"  Variance explained (blue): {variance_explained_pct:.2f}%")
    print(f"  Hemo correction         : {hemoVar_global:.2f}%")
    print(f"{'='*60}\n")

    # Cleanup: delete temp registered memmap file if it was used
    if registered_file and os.path.exists(registered_file):
        os.remove(registered_file)
        print(f"  Deleted temp registered file: {Path(registered_file).name}")

    # Cleanup: delete all tmp memmap files
    import shutil
    if tmp_dir.exists():
        shutil.rmtree(str(tmp_dir))
        print(f"  Cleaned up tmp dir: {tmp_dir}")

    return {
        'U': U,
        'newVc': newVc,
        'blueV': blueV,
        'hemoV': hemoV,
        'regC': regC,
        'T': T_mat,
        'hemoVar_global': float(hemoVar_global),
        'hemoVar_per_pixel': hemoVar_per_pixel,
        'variance_explained_percent': float(variance_explained_pct),
        'variance_explained_per_component': variance_per_component,
        'output_file': str(npz_path),
        'output_file_mat': str(mat_path),
    }


def _moving_avg_dfof(blue_data, hemo_data, window_size):
    """
    Moving-average dF/F₀ — port of MATLAB ``movingAvg_dFoF_cpu.m``.

    Uses edge-replicated padding before computing the moving average,
    exactly matching the MATLAB implementation.

    Parameters
    ----------
    blue_data, hemo_data : ndarray (T, H, W)
    window_size : int (frames)

    Returns
    -------
    blue_norm, hemo_norm : ndarray (T, H, W), float32
    """
    T, H, W = blue_data.shape
    n_pix = H * W
    half_w = window_size // 2

    for data in [blue_data, hemo_data]:
        flat = data.reshape(T, n_pix).T.astype(np.float64)  # (pix, T)

        # Pad edges by replicating first/last column
        padded = np.concatenate(
            [np.tile(flat[:, :1], (1, half_w)),
             flat,
             np.tile(flat[:, -1:], (1, half_w))],
            axis=1,
        )

        # Moving mean along time axis
        kernel = np.ones(window_size) / window_size
        from scipy.ndimage import uniform_filter1d
        f0_padded = uniform_filter1d(padded, size=window_size, axis=1,
                                     mode='nearest')
        f0 = f0_padded[:, half_w:half_w + T]

        normed = ((flat - f0) / f0).astype(np.float32)
        data[:] = normed.T.reshape(T, H, W)

    return blue_data, hemo_data


def _percentile_dfof(blue_data, hemo_data, pct, window_size):
    """
    Percentile-based dF/F₀ — port of MATLAB ``percentile_dFoF_cpu.m``.

    Processes pixels in auto-sized chunks (tuned to ~40 GB peak RAM) using
    numpy sliding_window_view — no Python frame loop, no T×n_pix×window_size
    materialisation all at once.

    Parameters
    ----------
    blue_data, hemo_data : ndarray (T, H, W), writable (memmap or ndarray)
    pct : int/float   e.g. 10 for 10th percentile
    window_size : int  sliding window in frames

    Returns
    -------
    blue_data, hemo_data  — modified in-place, same objects returned
    """
    from numpy.lib.stride_tricks import sliding_window_view
    from tqdm import tqdm

    T, H, W = blue_data.shape
    n_pix   = H * W

    # Padding so that exactly T windows emerge from sliding_window_view:
    #   pad_left + T + pad_right - window_size + 1 == T
    #   => pad_left + pad_right == window_size - 1
    pad_left  = window_size // 2
    pad_right = window_size - pad_left - 1

    # ── auto chunk size ───────────────────────────────────────────────────────
    # Peak RAM per chunk: chunk_size * T * window_size * 4 bytes (float32)
    # Budget: 40 GB (conservative on a 500 GB system)
    _BUDGET_BYTES = 40 * 1024 ** 3
    chunk_size = max(1, int(_BUDGET_BYTES / (T * window_size * 4)))
    chunk_size = min(chunk_size, n_pix)
    n_chunks   = (n_pix + chunk_size - 1) // chunk_size

    print(f"  Percentile dF/F: T={T}, n_pix={n_pix:,}, window={window_size}, "
          f"chunk={chunk_size:,} px ({n_chunks} chunks), "
          f"peak ~{chunk_size*T*window_size*4/1e9:.1f} GB per chunk")

    for ch_name, data in [('blue', blue_data), ('hemo', hemo_data)]:
        # (n_pix, T) — contiguous float32 working buffer
        flat = np.ascontiguousarray(
            data.reshape(T, n_pix).T, dtype=np.float32)  # (n_pix, T)

        out = np.empty_like(flat)  # (n_pix, T) float32

        for s in tqdm(range(0, n_pix, chunk_size), total=n_chunks,
                      desc=f'Percentile dF/F ({ch_name})', unit='chunk'):
            e = min(s + chunk_size, n_pix)
            chunk = flat[s:e]                             # (cs, T) float32 — uint16 fits exactly

            # Edge-pad: (cs, T + window_size - 1)
            padded = np.pad(chunk, ((0, 0), (pad_left, pad_right)), mode='edge')

            # Sliding window view: (cs, T, window_size) — strides only, no RAM copy
            windows = sliding_window_view(padded, window_shape=window_size, axis=1)

            # Rolling Nth percentile baseline: (cs, T)
            f0 = np.percentile(windows, pct, axis=2).astype(np.float32)

            out[s:e] = (chunk - f0) / f0

        # Write back into memmap (T, H, W)
        data[:] = out.T.reshape(T, H, W)
        del flat, out

    return blue_data, hemo_data
