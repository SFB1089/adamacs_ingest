"""
Widefield SVD Processing and Ingestion Schema

This module defines tables for widefield calcium imaging SVD decomposition results.

**Primary Table: WfieldSvd**
- Stores SVD components (U, newVc, blueV, hemoV) as external blobs
- Primary keys: scan, session, area (brain region)
- Follows element-calcium-imaging patterns for consistency

**Lookup Table: Area**
- Brain regions from CCF (Common Coordinate Framework)
- Links to anatomical locations

Author: Aelton Araújo 
Date: October 10, 2025
"""

import datajoint as dj
from ..pipeline import subject, session, scan, event, trial, model, db_prefix
from ..paths import get_experiment_root_data_dir
from element_interface.utils import find_full_path

import numpy as np
from pathlib import Path
from datetime import datetime
# import traceback

# Import required schemas - these must be activated first in pipeline.py
# Don't import from ..pipeline to avoid circular dependencies
schema = dj.schema(db_prefix + "wfield")



# def activate(wfield_schema_name, *, create_schema=False, create_tables=True, linking_module=None):
#     """
#     Activate the wfield schema.
    
#     :param wfield_schema_name: schema name on the database server
#     :param create_schema: when True (default), create schema if it doesn't exist
#     :param create_tables: when True (default), create tables if they don't exist
#     :param linking_module: module containing required dependencies:
#         - scan: Scan table
#         - session: Session table
#         - subject: Subject table
#         - Location: AnatomicalLocation table
#     """
#     import inspect
#     import importlib
    
#     if isinstance(linking_module, str):
#         linking_module = importlib.import_module(linking_module)
#     assert inspect.ismodule(linking_module), \
#         "The argument 'linking_module' must be a module name or a module"
    
#     global _linking_module
#     _linking_module = linking_module
    
#     schema.activate(wfield_schema_name,
#                     create_schema=create_schema,
#                     create_tables=create_tables,
#                     add_objects=_linking_module.__dict__)


# # ============================================================================
# LOOKUP TABLES
# ============================================================================

@schema
class Area(dj.Lookup):
    definition = """
    # Brain regions/areas for widefield imaging (CCF-based)
    area: varchar(16)  # Region acronym (e.g., 'VISp', 'VISl', 'RSP')
    ---
    area_name: varchar(128)  # Full region name
    hemisphere: enum('left', 'right', 'bilateral')  # Brain hemisphere
    """
    
    contents = [
        # Olfactory
        ('MOB', 'Main olfactory bulb', 'bilateral'),
        ('MOB_l', 'Main olfactory bulb', 'left'),
        ('MOB_r', 'Main olfactory bulb', 'right'),
        
        # Prefrontal/Frontal cortex
        ('FRP', 'Frontal pole, cerebral cortex', 'bilateral'),
        ('FRP_l', 'Frontal pole, cerebral cortex', 'left'),
        ('FRP_r', 'Frontal pole, cerebral cortex', 'right'),
        
        ('PL', 'Prelimbic area', 'bilateral'),
        ('PL_l', 'Prelimbic area', 'left'),
        ('PL_r', 'Prelimbic area', 'right'),
        
        # Motor cortex
        ('MOp', 'Primary motor area', 'bilateral'),
        ('MOp_l', 'Primary motor area', 'left'),
        ('MOp_r', 'Primary motor area', 'right'),

        ('MOs', 'Secondary motor area', 'bilateral'),
        ('MOs_l', 'Secondary motor area', 'left'),
        ('MOs_r', 'Secondary motor area', 'right'),
        
        # Somatosensory cortex
        ('SSp', 'Primary somatosensory area', 'bilateral'),  # Kept based on DB
        ('SSp-n', 'Primary somatosensory area, nose', 'bilateral'),
        ('SSp-n_l', 'Primary somatosensory area, nose', 'left'),
        ('SSp-n_r', 'Primary somatosensory area, nose', 'right'),

        ('SSp-m', 'Primary somatosensory area, mouth', 'bilateral'),
        ('SSp-m_l', 'Primary somatosensory area, mouth', 'left'),
        ('SSp-m_r', 'Primary somatosensory area, mouth', 'right'),

        ('SSp-un', 'Primary somatosensory area, unassigned', 'bilateral'),
        ('SSp-un_l', 'Primary somatosensory area, unassigned', 'left'),
        ('SSp-un_r', 'Primary somatosensory area, unassigned', 'right'),

        ('SSp-bfd', 'Primary somatosensory area, barrel field', 'bilateral'),
        ('SSp-bfd_l', 'Primary somatosensory area, barrel field', 'left'),
        ('SSp-bfd_r', 'Primary somatosensory area, barrel field', 'right'),

        ('SSp-tr', 'Primary somatosensory area, trunk', 'bilateral'),
        ('SSp-tr_l', 'Primary somatosensory area, trunk', 'left'),
        ('SSp-tr_r', 'Primary somatosensory area, trunk', 'right'),

        ('SSp-ll', 'Primary somatosensory area, lower limb', 'bilateral'),
        ('SSp-ll_l', 'Primary somatosensory area, lower limb', 'left'),
        ('SSp-ll_r', 'Primary somatosensory area, lower limb', 'right'),

        ('SSp-ul', 'Primary somatosensory area, upper limb', 'bilateral'),
        ('SSp-ul_l', 'Primary somatosensory area, upper limb', 'left'),
        ('SSp-ul_r', 'Primary somatosensory area, upper limb', 'right'),

        ('SSs', 'Supplemental somatosensory area', 'bilateral'),
        ('SSs_l', 'Supplemental somatosensory area', 'left'),
        ('SSs_r', 'Supplemental somatosensory area', 'right'),
        
        # Cingulate cortex
        ('ACAd', 'Anterior cingulate area, dorsal part', 'bilateral'),
        ('ACAd_l', 'Anterior cingulate area, dorsal part', 'left'),
        ('ACAd_r', 'Anterior cingulate area, dorsal part', 'right'),
        
        # Retrosplenial cortex
        ('RSP', 'Retrosplenial area', 'bilateral'), # Kept based on DB
        ('RSPv', 'Retrosplenial area, ventral part', 'bilateral'),
        ('RSPv_l', 'Retrosplenial area, ventral part', 'left'),
        ('RSPv_r', 'Retrosplenial area, ventral part', 'right'),

        ('RSPd', 'Retrosplenial area, dorsal part', 'bilateral'),
        ('RSPd_l', 'Retrosplenial area, dorsal part', 'left'),
        ('RSPd_r', 'Retrosplenial area, dorsal part', 'right'),

        ('RSPagl', 'Retrosplenial area, lateral agranular part', 'bilateral'),
        ('RSPagl_l', 'Retrosplenial area, lateral agranular part', 'left'),
        ('RSPagl_r', 'Retrosplenial area, lateral agranular part', 'right'),
        
        # Visceral area
        ('VISC', 'Visceral area', 'bilateral'),
        ('VISC_l', 'Visceral area', 'left'),
        ('VISC_r', 'Visceral area', 'right'),
        
        # Temporal association
        ('TEa', 'Temporal association areas', 'bilateral'),
        ('TEa_l', 'Temporal association areas', 'left'),
        ('TEa_r', 'Temporal association areas', 'right'),
        
        # Auditory cortex
        ('AUDd', 'Dorsal auditory area', 'bilateral'),
        ('AUDd_l', 'Dorsal auditory area', 'left'),
        ('AUDd_r', 'Dorsal auditory area', 'right'),

        ('AUDp', 'Primary auditory area', 'bilateral'),
        ('AUDp_l', 'Primary auditory area', 'left'),
        ('AUDp_r', 'Primary auditory area', 'right'),

        ('AUDpo', 'Posterior auditory area', 'bilateral'),
        ('AUDpo_l', 'Posterior auditory area', 'left'),
        ('AUDpo_r', 'Posterior auditory area', 'right'),

        ('AUDv', 'Ventral auditory area', 'bilateral'),
        ('AUDv_l', 'Ventral auditory area', 'left'),
        ('AUDv_r', 'Ventral auditory area', 'right'),
        
        # Visual cortex
        ('VISp', 'Primary visual area', 'bilateral'),
        ('VISp_l', 'Primary visual area', 'left'),
        ('VISp_r', 'Primary visual area', 'right'),

        ('VISl', 'Lateral visual area', 'bilateral'),
        ('VISl_l', 'Lateral visual area', 'left'),
        ('VISl_r', 'Lateral visual area', 'right'),

        ('VISal', 'Anterolateral visual area', 'bilateral'),
        ('VISal_l', 'Anterolateral visual area', 'left'),
        ('VISal_r', 'Anterolateral visual area', 'right'),

        ('VISrl', 'Rostrolateral visual area', 'bilateral'),
        ('VISrl_l', 'Rostrolateral visual area', 'left'),
        ('VISrl_r', 'Rostrolateral visual area', 'right'),

        ('VISa', 'Anterior area', 'bilateral'),
        ('VISa_l', 'Anterior area', 'left'),
        ('VISa_r', 'Anterior area', 'right'),

        ('VISam', 'Anteromedial visual area', 'bilateral'),
        ('VISam_l', 'Anteromedial visual area', 'left'),
        ('VISam_r', 'Anteromedial visual area', 'right'),

        ('VISpm', 'Posteromedial visual area', 'bilateral'),
        ('VISpm_l', 'Posteromedial visual area', 'left'),
        ('VISpm_r', 'Posteromedial visual area', 'right'),

        ('VISpl', 'Posterolateral visual area', 'bilateral'),
        ('VISpl_l', 'Posterolateral visual area', 'left'),
        ('VISpl_r', 'Posterolateral visual area', 'right'),

        ('VISli', 'Laterointermediate area', 'bilateral'),
        ('VISli_l', 'Laterointermediate area', 'left'),
        ('VISli_r', 'Laterointermediate area', 'right'),

        ('VISpor', 'Postrhinal area', 'bilateral'),
        ('VISpor_l', 'Postrhinal area', 'left'),
        ('VISpor_r', 'Postrhinal area', 'right'),
    ]


@schema
class SvdMethod(dj.Lookup):
    definition = """
    # SVD computation method
    svd_method: varchar(16)
    ---
    svd_method_description: varchar(255)
    """
    
    contents = [
        ('matlab_fsvd', 'MATLAB fast SVD implementation (run_SVD.m)'),
        ('python_fsvd', 'Python fast randomized SVD (run_svd_python)'),
        ('python_sklearn', 'Python scikit-learn TruncatedSVD'),
        ('python_numpy', 'Python numpy.linalg.svd'),
    ]


@schema
class NormalizationMethod(dj.Lookup):
    definition = """
    # Normalization method
    normalization_method: varchar(32)
    ---
    method_description: varchar(255)
    """
    
    contents = [
        ('global_mean', 'Global mean subtraction'),
        ('none', 'No hemodynamic correction applied'),
        ('pixel_based', 'Per-pixel regression'),
        ('rolling_window', 'Rolling window percentile'),
        ('trial_based', 'Trial-based regression')
        
    ]
    
@schema
class HemoCorrectionMethod(dj.Lookup):
    definition = """
    # Hemodynamic correction method
    hemo_correction_method: varchar(32)
    ---
    method_description: varchar(255)
    """
    
    contents = [
        ('none', 'No hemodynamic correction'),
        ('linear_regression_SVD', 'Pixel-wise linear regression (SVD)'),
        ('linear_regression', 'Pixel-wise linear regression')
    ]

# ============================================================================
# PROCESSING AUTOMATION (ParamSets & Tasks)
# ============================================================================


@schema
class ParamSet(dj.Lookup):
    definition = """
    # Parameters for widefield SVD processing (supports MATLAB and Python engines)
    paramset_idx: smallint
    ---
    processing_engine: enum('matlab', 'python')  # Which processing engine to use
    -> SvdMethod
    paramset_desc: varchar(128)
    params: longblob  # dictionary of processing parameters
    """

    @classmethod
    def insert_default(cls):
        """Insert default parameters for both MATLAB and Python engines"""
        cls.insert([
            # --- MATLAB defaults ---
            (0, 'matlab', 'matlab_fsvd',
             'Matlab SVD 500 components, global mean', {
                'dimCnt': 500,
                'normalization_method': 'global_mean',
                'dType': 'uint16',
                'window_sec': 0,
                'percentile': 0,
                'mov_avg': False,
                'offset_thr': 'none'
            }),
            (1, 'matlab', 'matlab_fsvd',
             'Matlab SVD 500 components, 60s rolling window 10th percentile', {
                'dimCnt': 500,
                'normalization_method': 'percentile',
                'dType': 'uint16',
                'window_sec': 60,
                'percentile': 10,
                'mov_avg': False,
                'offset_thr': 'none'
            }),
            # --- Python defaults ---
            (10, 'python', 'python_fsvd',
             'Python SVD 200 components, global mean', {
                'dimCnt': 200,
                'normalization_method': 'global_mean',
                'dType': 'uint16',
                'window_sec': 0,
                'percentile_val': 0,
                'enable_highpass': True,
                'enable_lowpass': True,
                'lowCut': 0.01,
                'highCut': 10.0,
                'smooth_blue': False,
                'random_seed': 0,
            }),
            (11, 'python', 'python_fsvd',
             'Python SVD 500 components, 60s rolling window 10th percentile', {
                'dimCnt': 500,
                'normalization_method': 'percentile',
                'dType': 'uint16',
                'window_sec': 60,
                'percentile_val': 10,
                'enable_highpass': True,
                'enable_lowpass': True,
                'lowCut': 0.01,
                'highCut': 10.0,
                'smooth_blue': False,
                'random_seed': 0,
            }),
            (12, 'python', 'python_fsvd',
             'Python SVD 500 components, global mean', {
                'dimCnt': 500,
                'normalization_method': 'global_mean',
                'dType': 'uint16',
                'window_sec': 0,
                'percentile_val': 0,
                'enable_highpass': True,
                'enable_lowpass': True,
                'lowCut': 0.1,
                'highCut': 10.0,
                'smooth_blue': False,
                'random_seed': 0,
            }),
        ], skip_duplicates=True)


@schema
class ProcessingTask(dj.Manual):
    definition = """
    # Manual table to define a processing task for a scan
    -> scan.Scan
    processing_id: smallint  # ID to allow multiple processing runs per scan
    ---
    -> ParamSet
    task_mode='trigger': enum('load', 'trigger') # 'trigger' runs analysis, 'load' just ingests output
    processing_output_dir='': varchar(255) # Custom output directory (optional)
    """


@schema
class ProcessingRun(dj.Manual):
    definition = """
    # Maps integer processing_id to alphanumeric proc_id (ties Registration + SVD files together)
    -> ProcessingTask
    ---
    proc_id: varchar(16)  # Alphanumeric timestamp ID e.g. 'proc9FSWCF3E'
    run_timestamp: timestamp  # When this processing run was initiated
    """


# ============================================================================
# MASK SVD — truncated quick SVD to generate spatial components for FOV masking
# ============================================================================

@schema
class MaskSvd(dj.Computed):
    definition = """
    # Truncated SVD on first N frames (all pixels) — used to create wfield.FOVMask.
    # Worker populates this automatically when a ProcessingTask exists.
    # User then inspects `u[:,:,0]`, adjusts thresholds, and inserts into FOVMask.
    -> scan.Scan
    ---
    n_components: int            # Number of SVD components computed
    n_frames: int                # Per-channel frames used (e.g. 1000)
    frame_height: int
    frame_width: int
    u: blob@external-raw                  # Spatial components (H x W x n_components)
    mask_svd_timestamp: timestamp
    """

    @property
    def key_source(self):
        # Only process scans that have at least one ProcessingTask
        return scan.Scan & ProcessingTask

    def make(self, key):
        from ..helpers import wfield_helpers
        from pathlib import Path

        print(f"\nMaskSvd.make() for {key}")

        # Fetch scan info for frame rate
        scan_info_query = scan.ScanInfo & key
        if not len(scan_info_query):
            raise ValueError(f"No ScanInfo found for {key}")
        scan_info = scan_info_query.fetch1()
        scan_key = scan_info_query.fetch('KEY')[0]

        # Get session directory and imaging file
        session_dir = (session.SessionDirectory & scan_key).fetch1('session_dir')
        imaging_file, _ = wfield_helpers.find_imaging_file(str(session_dir))
        if imaging_file is None:
            raise FileNotFoundError(f"No imaging file found in {session_dir}")

        # Output dir for mask SVD tmp files
        output_dir = Path(session_dir) / 'svds' / 'mask_svd'
        output_dir.mkdir(parents=True, exist_ok=True)

        per_channel_fps = scan_info['fps']
        total_srate = per_channel_fps * 2
        proc_id = wfield_helpers.generate_proc_id()
        
        # Fetch dimCnt from the associated ProcessingTask ParamSet (same logic as Processing.make())
        task = (ProcessingTask & key).fetch(order_by='processing_id', limit=1, as_dict=True)[0]
        paramset = (ParamSet & {'paramset_idx': task['paramset_idx']}).fetch1()
        params = paramset['params']
        dim_cnt = 50 # keep this small since it's just for masking — we only care about the first few comps
        print(f"  Running truncated SVD on first 1000 frames (per channel), dimCnt={dim_cnt}...")
        svd_result = wfield_helpers.run_svd_python(
            imaging_file=str(imaging_file),
            output_dir=str(output_dir),
            scan_id=key['scan_id'],
            sRate=total_srate,
            dimCnt=dim_cnt, # keep this small since it's just for masking — we only care about the first few comps
            proc_id=proc_id,
            max_frames=1000,
        )

        U = svd_result['U']
        height, width, n_comp = U.shape
        n_frames_used = svd_result['newVc'].shape[1]

        self.insert1({
            **key,
            'n_components': n_comp,
            'n_frames': n_frames_used,
            'frame_height': height,
            'frame_width': width,
            'u': U,
            'mask_svd_timestamp': datetime.now(),
        })
        print(f"  MaskSvd stored for {key['scan_id']}: U{U.shape}")


@schema
class FOVMask(dj.Manual):
    definition = """
    # Field of view (FOV) mask for each scan
    -> scan.Scan
    ---
    fov_mask: longblob  # Binary mask of valid FOV pixels (height × width)
    mask_method: varchar(64)  # Method used to generate the mask
    mask_notes='': varchar(500)  # Additional notes on mask quality, issues, etc.
    """


@schema
class Processing(dj.Computed):
    definition = """
    # Automated widefield SVD processing — only runs once FOVMask exists for the scan.
    -> ProcessingTask
    -> FOVMask
    ---
    processing_time: datetime
    execution_duration: float  # (seconds)
    """
    
    def make(self, key):
        """
        Run widefield SVD processing (MATLAB or Python) based on ParamSet.processing_engine.
        
        1. Generate proc_id IMMEDIATELY and create ProcessingRun entry
        2. Fetch params from ParamSet (engine, svd_method, params dict)
        3. Construct output directory
        4. Run registration (always Python, separate blue/violet)
        5. Run SVD (MATLAB or Python depending on engine)
        6. Ingest results into Svd table
        """
        import time 
        from ..helpers import wfield_helpers
        
        # 0. Resolve proc_id — must already exist in ProcessingRun (inserted BEFORE populate()).
        #    DataJoint wraps make() in a transaction: any insert here is rolled back on failure,
        #    so ProcessingRun is populated externally (worker / manual script) to guarantee it
        #    persists even when processing fails.
        if len(ProcessingRun & key):
            proc_id = (ProcessingRun & key).fetch1('proc_id')
        else:
            # Fallback for direct make() calls — proc_id will be rolled back on failure,
            # but at least it works for successful runs without a pre-insert step.
            proc_id = wfield_helpers.generate_proc_id()
            ProcessingRun.insert1({
                **key,
                'proc_id': proc_id,
                'run_timestamp': datetime.now(),
            }, skip_duplicates=True)

        # 1. Fetch parameters from ParamSet
        task = (ProcessingTask & key).fetch1()
        paramset_idx = task['paramset_idx']
        paramset = (ParamSet & {'paramset_idx': paramset_idx}).fetch1()
        processing_engine = paramset['processing_engine']
        svd_method = paramset['svd_method']
        params = paramset['params']
        
        print(f"\n{'='*60}")
        print(f"  proc_id: {proc_id}")
        print(f"  Processing engine: {processing_engine.upper()}")
        print(f"  SVD method: {svd_method}")
        print(f"  ParamSet: {paramset_idx} — {paramset['paramset_desc']}")
        print(f"{'='*60}\n")
        
        # 2. Determine file paths using SessionDirectory
        try:
            scan_info_query = scan.ScanInfo & f'scan_id="{key["scan_id"]}"'
            n_records = len(scan_info_query)
            
            if n_records == 0:
                raise ValueError(f"No ScanInfo found for key: {key}")
            elif n_records > 1:
                raise ValueError(f"Multiple ScanInfo tuples found for key: {key}")
            
            scan_key = scan_info_query.fetch('KEY')[0]
            scan_info = scan_info_query.fetch1()
        except Exception as e:
            print(f"\nERROR fetching ScanInfo: {key} — {e}")
            raise
        
        # Get the session directory
        session_dir = (session.SessionDirectory & scan_key).fetch1('session_dir')
        
        # Find imaging file
        imaging_file, _ = wfield_helpers.find_imaging_file(str(session_dir))
        if imaging_file is None:
            raise FileNotFoundError(f"No imaging file found in {session_dir}")
        
        # Get/Create output directory: <session_dir>/svds/
        if task['processing_output_dir']:
            output_dir = Path(task['processing_output_dir'])
        else:
            output_dir = Path(session_dir) / 'svds'

        output_dir.mkdir(parents=True, exist_ok=True)
            
        # 3. Trigger Processing (if mode is 'trigger')
        start_time = time.time()
        svd_result = None  # Will hold Python SVD output dict
        
        if task['task_mode'] == 'trigger':
            # Step A: Compute registration (always Python, separate blue/violet)
            print(f"Step 1/2: Computing motion correction registration...")
            apply_registration = False  # shifts are applied in run_svd_python Step 2
            registration_result = wfield_helpers.compute_registration(
                imaging_file=str(imaging_file),
                output_dir=str(output_dir),
                scan_id=key['scan_id'],
                proc_id=proc_id,
                method='phaseCorrelate',
                apply=apply_registration,
            )
            
            wfield_helpers.ingest_registration(
                key=key,
                registration_result=registration_result,
            )
            
            # FPS: ScanInfo.fps stores per-channel rate; total camera rate = fps * 2
            per_channel_fps = scan_info['fps']
            total_srate = per_channel_fps * 2
            
            # Step B: Run SVD — branch on engine
            if processing_engine == 'matlab':
                print(f"Step 2/2: Starting MATLAB SVD processing...")
                matlab_vars = {
                    'fullFilePath': str(imaging_file),
                    'outputPath': str(output_dir) + '/',
                    'sRate': total_srate,
                    'proc_id': proc_id,
                    **params
                }
                wfield_helpers.run_matlab_script_realtime(
                    script_path='/home/aeltona/adamacs/adamacs/helpers/matlab/run_SVD.m',
                    matlab_vars=matlab_vars
                )
                
            elif processing_engine == 'python':
                print(f"Step 2/2: Starting Python SVD processing...")
                # Pass transposed file so run_svd_python skips re-reading the F-order .dat.
                # Shifts are applied in run_svd_python Step 2 from the saved .npz shifts file.
                transposed_file = registration_result.get('transposed_file')

                # Fetch FOV mask array directly from the database — no temp file needed.
                # Non-brain pixels are zeroed before SVD (after registration + normalisation).
                fov_mask = (FOVMask & scan_key).fetch1('fov_mask')
                print(f"  FOV mask: {int(fov_mask.sum())} brain pixels "
                      f"({100 * fov_mask.mean():.1f}% of frame)")

                python_kwargs = {
                    'imaging_file': str(imaging_file),
                    'output_dir': str(output_dir),
                    'scan_id': key['scan_id'],
                    'sRate': total_srate,
                    'proc_id': proc_id,
                    'transposed_file': transposed_file,
                    'dimCnt': params['dimCnt'],
                    'dType': params['dType'],
                    'normalization_method': params['normalization_method'],
                    'random_seed': params['random_seed'],
                    'enable_highpass': params['enable_highpass'],
                    'enable_lowpass': params['enable_lowpass'],
                    'lowCut': params['lowCut'],
                    'highCut': params['highCut'],
                    'smooth_blue': params['smooth_blue'],
                    'brain_mask': fov_mask,
                }
                if 'window_sec' in params:
                    python_kwargs['window_sec'] = params['window_sec']
                if 'percentile_val' in params:
                    python_kwargs['percentile_val'] = params['percentile_val']

                svd_result = wfield_helpers.run_svd_python(**python_kwargs)
            else:
                raise ValueError(f"Unknown processing engine: {processing_engine}")
        
        # 4. Find output file (for MATLAB or load mode)
        if task['task_mode'] == 'trigger':
            search_pattern = f"{key['scan_id']}_Vc_{proc_id}_*.mat"
        else:
            search_pattern = f"{key['scan_id']}_Vc*.mat"
        
        found_files = list(output_dir.glob(search_pattern))
        
        if not found_files and svd_result is None:
            raise FileNotFoundError(
                f"Could not find SVD output file pattern {search_pattern} in {output_dir}")
        
        if found_files:
            print(f"Found output file: {found_files[0]}")
            
        duration = time.time() - start_time
        
        # 5. Insert Processing record
        self.insert1({
            **key,
            'processing_time': datetime.now(),
            'execution_duration': duration
        })
        
        # 6. Ingest results into Svd table
        normalization_method = params['normalization_method']
        curation_note = f"Auto-ingested from Processing task {key['processing_id']} ({processing_engine})"
        
        if processing_engine == 'python' and svd_result is not None:
            from ..helpers.wfield_ingest import ingest_svd_from_python
            ingest_svd_from_python(
                svd_result=svd_result,
                scan_id=key['scan_id'],
                session_id=key['session_id'],
                processing_id=key['processing_id'],
                svd_output_folder=str(output_dir),
                normalization_method=normalization_method,
                svd_method=svd_method,
                curation_note=curation_note,
            )
        else:
            from ..helpers.wfield_ingest import ingest_svd_from_matlab
            ingest_svd_from_matlab(
                scan_id=key['scan_id'],
                session_id=key['session_id'],
                processing_id=key['processing_id'],
                wfield_folder=str(output_dir),
                normalization_method=normalization_method,
                svd_method=svd_method,
                curation_note=curation_note,
            )

# ============================================================================
# PRIMARY TABLE - WIDEFIELD SVD RESULTS
# ============================================================================

# Line ~170 (replace WfieldSvd class)
@schema
class Svd(dj.Manual):
    definition = """
    # Widefield SVD decomposition results
    -> scan.Scan
    -> session.Session
    processing_id: smallint  # Same as ProcessingTask ID
    ---
    # Metadata
    n_components: int  # Number of SVD components (e.g., 200)
    n_frames: int  # Total frames in decomposition
    frame_height: int  # Imaging frame height (pixels)
    frame_width: int  # Imaging frame width (pixels)
    -> SvdMethod
    -> HemoCorrectionMethod
    normalization_method: varchar(32)
    svd_output_folder: varchar(255)
    processing_timestamp: timestamp
    ingestion_timestamp: timestamp
    curation_time: datetime
    manual_curation: bool
    curation_note='': varchar(2000)
    # --- Variance Explained ---
    variance_explained: longblob  # Per-component variance explained ratio (n_components,)
    hemo_variance_explained: float  # Global hemodynamic variance explained (0-100 pct)
    # --- SVD Component Arrays (stored in external-raw file store) ---
    u: blob@external-raw  # Spatial components (height x width x n_components)
    new_vc: blob@external-raw  # Temporal corrected (n_components x n_frames)
    blue_v: blob@external-raw  # Temporal raw blue (n_components x n_frames)
    hemo_v: blob@external-raw  # Temporal violet/hemo (n_components x n_frames)
    """


# ============================================================================
# REGISTRATION AND ALIGNMENT TABLES
# ============================================================================

@schema  
class Registration(dj.Manual):
    definition = """
    # Motion correction registration shifts and reference frames (separate blue/violet)
    -> scan.Scan
    processing_id: smallint  # Same as ProcessingTask ID
    ---
    shifts_file_path: varchar(255)  # Path to _shifts.npz file
    n_frames_registered: int  # Total number of interleaved frames registered
    # Per-frame shift arrays 
    shifts_x_blue: blob@external-raw  # X shifts for blue channel (n_per_channel,) float32
    shifts_y_blue: blob@external-raw  # Y shifts for blue channel (n_per_channel,) float32
    shifts_x_violet: blob@external-raw  # X shifts for violet channel (n_per_channel,) float32
    shifts_y_violet: blob@external-raw  # Y shifts for violet channel (n_per_channel,) float32
    # Reference frames
    reference_frame_blue: blob@external-raw  # Reference frame for blue channel (H x W)
    reference_frame_violet: blob@external-raw  # Reference frame for violet channel (H x W)
    registration_method='phaseCorrelate': varchar(32)  # Registration algorithm used
    processing_time_sec: float  # Time to compute registration (seconds)
    registration_timestamp: timestamp  # When registration was computed
    registration_notes='': varchar(500)  # Additional notes
    """



# ============================================================================
# CCF ALIGNMENT AND LANDMARK TABLES
# ============================================================================

@schema
class CcfLandmarks(dj.Lookup):
    definition = """
    # CCF atlas reference landmarks for alignment (shared across users/scans)
    ccf_landmarks_id: smallint
    ---
    reference_name: varchar(32)  # Atlas reference name (e.g., 'dorsal_cortex')
    landmarks: longblob  # dict with keys x, y, name, color (lists)
    description: varchar(255)
    """

    @classmethod
    def insert_default(cls):
        """Insert default CCF landmarks"""
        cls.insert([
            (0, 'dorsal_cortex',
             {'x': [-2.40, -3.979, 2.40, 4],
              'y': [2.34, 2.42, 2.34, 2.45],
              'name': ['VISp-VISam_l', 'VISal-apex_l', 'VISp-VISam_r', 'VISal-apex_r'],
              'color': ['#fc9d03', '#0367fc', '#fc9d03', '#fc4103']},
             'Default visual cortex landmarks (VISp-VISam and VISal-apex, bilateral)')
        ], skip_duplicates=True)

@schema
class CcfReference(dj.Lookup):
    definition = """
    # CCF atlas reference data (shared across users/scans)
    reference_name: varchar(32)  # Atlas reference name (e.g., 'dorsal_cortex')
    ---
    
    ccf_regions_ref: longblob  # dict with keys 'acronym', 'name', 'reference', 'resolution', 'label', 'allen_id', 'allen_rgb', 'left_center', 'right_center', 'left_x', 'left_y', 'right_x', 'right_y' (lists)
    description: varchar(255)
    """


@schema
class RetinotopicRef(dj.Manual):
    definition = """
    # Retinotopic reference data (per subject)
    -> subject.Subject
    retinotopic_session_id: varchar(32)
    ---
    bloodvessels_image: longblob  # Blood vessels reference image (H, W)
    vfs_map: longblob  # Visual field sign map (H, W)
    vessels_landmarks_ref: longblob  # Reference landmarks dict (x, y, name, color)
    retinotopic_folder: varchar(255)
    vfs_blocks: longblob  # List of block indices used for VFS computation
    """


@schema
class VesselsTransform(dj.Manual):
    definition = """
    # Blood vessel transform: retinotopic session -> experimental scan
    -> scan.Scan
    ---
    -> RetinotopicRef
    transform_params: longblob  # SimilarityTransform .params (3x3 ndarray)
    landmarks_match: longblob  # Matched landmarks dict (x, y, name, color)
    resolution: float  # Resolution used for transform
    bregma_offset: longblob  # Bregma offset array
    vfs_vessels: longblob  # VFS overlayed on scan (H, W, 3) uint8 RGB
    transform_timestamp: timestamp
    """


@schema
class CcfAlignment(dj.Manual):
    definition = """
    # CCF alignment transform for a scan
    -> scan.Scan
    ---
    -> CcfLandmarks
    transform_params: longblob  # SimilarityTransform .params (3x3 ndarray)
    landmarks_match: longblob  # Matched landmarks dict (x, y, name, color)
    landmarks_im: longblob  # Landmarks in image space dict (x, y, name, color)
    resolution: float  # Resolution (mm/pixel)
    bregma_offset: longblob  # Bregma offset array
    ccf_regions_im: longblob  # Transformed CCF regions (DataFrame.to_dict())
    alignment_timestamp: timestamp
    """


@schema
class RegionMask(dj.Manual):
    definition = """
    # CCF region mask for scan (from atlas alignment + FOV mask)
    -> scan.Scan
    ---
    region_mask: longblob  # Boolean mask (H, W) covering all CCF regions in FOV
    mask_timestamp: timestamp
    """




# ============================================================================
# ROI TRACE EXTRACTION TABLES
# ============================================================================

@schema
class RegionActivity(dj.Manual):
    definition = """
    # ROI-averaged calcium traces per brain region
    -> scan.Scan
    -> Area
    ---
    corrected: longblob  # Mean calcium trace (corrected blue channel)
    blue: longblob  # Raw blue channel trace
    violet: longblob  # Raw violet channel trace (hemodynamics)
    n_pixels: int  # Number of pixels in ROI
    fraction_in_fov: float  # Fraction of region visible in FOV
    extraction_timestamp: timestamp  # When traces were extracted
    """

