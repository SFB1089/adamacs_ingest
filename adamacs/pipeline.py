from . import db_prefix

__all__ = ['subject', 'surgery', 'session', 'behavior', 'equipment', 'scan', 'imaging',
           'train', 'model', 'trial', 'event', 'analysis', 'mocap', 'virtual_markers_optitrack', 
           'pupil_tracking', 'denoising', 'disk',
           'Equipment', 'Location', 'Subject', 'Project', 'Lab', 'User', 
           'Session',
           'get_session_dir', 'get_bpod_root_data_dir', 'get_dlc_root_data_dir',
           'get_imaging_root_data_dir', 'get_scan_image_files', 'get_scan_box_files']

# Import items for "session" -------------------------------------------
from adamacs.schemas import subject
from adamacs.schemas.subject import Subject, Project, Lab, User
from adamacs.schemas import surgery
from element_session import session_with_id as session

# Activate "session" ---------------------------------------------------
Experimenter = User
session.activate(db_prefix + 'session', linking_module=__name__)
SessionUser = session.SessionUser

# Import remaining schemas ----------------------------------------------
from adamacs.schemas import equipment
from adamacs.schemas.equipment import Equipment
from adamacs.schemas.surgery import AnatomicalLocation as Location
from element_calcium_imaging import scan, imaging
from element_deeplabcut import train, model
from element_event import event, trial
from .paths import get_session_dir, get_experiment_root_data_dir, get_dlc_root_data_dir
from .paths import get_imaging_root_data_dir, get_scan_image_files, get_scan_box_files
get_bpod_root_data_dir = get_experiment_root_data_dir

# Activate "scan" ---------------------------------------------------
scan.activate(db_prefix + 'scan', linking_module=__name__)


# Rename items for foreign key references -------------------------------
Session = session.Session
Scan = scan.Scan
Device = Equipment  # Inherited in model.VideoRecording

# Activate "train" and "model" schema -----------------------------------

train.activate(db_prefix + 'train', linking_module=__name__)  # OPTIONAL, model training
model.activate(db_prefix + 'model', linking_module=__name__)

# Activate "imaging" and "scan" schema ----------------------------------

# requires Location activated above
imaging.activate(db_prefix + 'imaging',
                 db_prefix + 'scan',
                 linking_module=__name__)

trial.activate(db_prefix + 'trial',
               db_prefix + 'event',
               linking_module=__name__)

# Activate "analysis" schema ------------------------------------------ NOT IMPLEMENTED TR24

# analysis.activate(db_prefix + "analysis", linking_module=__name__)

# Behavior downstream of event
from adamacs.schemas import behavior

# Optional analysis extension (shim lives in adamacs.schemas.analysis)
from adamacs.schemas import analysis

# Analysis table
from adamacs.schemas import pupil_tracking

# Optional denoising extension (requires torch stack)
try:
    from adamacs.schemas import denoising
except ModuleNotFoundError:
    denoising = None

# MotionCapture table
from adamacs.schemas import mocap


# VirtualMarker  table
from adamacs.schemas import virtual_markers_optitrack

# DISK (Deep Imputation of SKeleton data) table
from adamacs.schemas import disk

#Denoising table
# from adamacs.schemas import behavior
