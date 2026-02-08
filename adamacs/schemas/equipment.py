"""Tables related to equipment.

We use a variety of different equipment 
"""
import datajoint as dj
from .. import db_prefix
schema = dj.schema(db_prefix + 'equipment')

__all__ = ['db_prefix', 'Equipment']

# -------------- Table declarations --------------


@schema
class Equipment(dj.Manual):
    definition = """
    scanner                        : varchar(32)
    ---
    scanner_description=''         : varchar(255)
    """

equipment_data = [
    {'scanner': 'mini2p_01', 'scanner_description': 'the oldest mini2p (USABLE - v3)'},
    {'scanner': 'mini2p_02', 'scanner_description': 'the first last gen mini2p (REPAIR - fast MEMS broken)'},
    {'scanner': 'mini2p_03', 'scanner_description': 'the second last gen mini2p (REPAIR - utlens broken, z-offset not best at 0V)'},
    {'scanner': 'mini2p_04', 'scanner_description': 'MEMS#: AA010 - the third last gen mini2p (USABLE - scanfield offset / dichroic or MEMS misaligned)'},
    {'scanner': 'mini2p_05', 'scanner_description': 'MEMS#:AA082 - the fourth last gen mini2p (the one you are currently finishing)'},
    {'scanner': 'bench2p', 'scanner_description': 'INSS RGG two-photon'},
    {'scanner': 'macroscope', 'scanner_description': 'Macroscope dual excitation'},
    {'scanner': 'dummy', 'scanner_description': 'dummy'}
]
Equipment.insert(equipment_data, skip_duplicates=True) 

@schema
class SetupRestraint(dj.Manual):
    definition = """
    setup_restraint                        : varchar(32)
    ---
    setup_restraint_description=''         : varchar(255)
    """

@schema
class Device(dj.Manual):
    definition = """
    camera                        : varchar(32)
    ---
    camera_description=''         : varchar(255)
    """



device_data = [
    {'camera': 'mini2p1_top', 'camera_description': 'Basler a2A1920-160umBAS, Xx objective'},
    {'camera': 'mini2p1_bottom', 'camera_description': 'Basler acA2500-60um, Xx objective'},
    {'camera': 'mini2p1_side1', 'camera_description': 'Basler a2A1920-160umPro, Xx objective'},
    {'camera': 'mini2p1_side2', 'camera_description': 'Basler a2A1920-160umPro, Xx objective'},
    {'camera': 'mini2p1_side3', 'camera_description': 'Basler a2A1920-160umPro, Xx objective'},
    {'camera': 'mini2p1_eye_ipsi', 'camera_description': 'USB cam, Xx objective'},
    {'camera': 'mini2p1_eye_contra', 'camera_description': 'USB cam, Xx objective'},
    {'camera': 'mini2p1_worldcam', 'camera_description': 'USB cam, Xx objective'},
    {'camera': 'mini2p2_top', 'camera_description': 'Basler a2A1920-160umBAS, Xx objective'},
    {'camera': 'mini2p2_bottom', 'camera_description': 'Basler acA2500-60um, Xx objective'},
    {'camera': 'bench2p_face', 'camera_description': 'Basler a2A1920-160umPro, Xx objective'},
    {'camera': 'bench2p_body', 'camera_description': 'Basler a2A1920-160umPro, Xx objective'},
    {'camera': 'bench2p_back', 'camera_description': 'Logitech C270 HD-webcam'},
    {'camera': 'macroscope_eye', 'camera_description': 'Basler a2A1920-160umPro, Xx objective'},
    {'camera': 'macroscope_back', 'camera_description': 'Logitech C270 HD-webcam'},
    {'camera': 'mini2p2_scope', 'camera_description': 'Thorlabs, Xx objective'},
    {'camera': 'mini2p1_scope', 'camera_description': 'XXX, Xx objective'}
]
Device.insert(device_data, skip_duplicates=True) 

restraint_data = [
    {'setup_restraint': 'mini2p1_headfixed',
    'setup_restraint_description': 'headfixed recording at mini2p1 setup',
    },
    {'setup_restraint': 'mini2p1_openfield',
    'setup_restraint_description': 'freely moving recording at mini2p1 setup',
    },
    {'setup_restraint': 'mini2p2_openfield',
    'setup_restraint_description': 'freely moving recording at mini2p2 setup',
    },
    {'setup_restraint': 'mini2p2_headfixed',
    'setup_restraint_description': 'headfixed recording at mini2p2 setup',
    },
    {'setup_restraint': 'bench2p_headfixed',
    'setup_restraint_description': 'headfixed recording at bench2p setup',
    },
    {'setup_restraint': 'macroscope_headfixed',
    'setup_restraint_description': 'freely moving recording at macroscope setup',
    }
]
SetupRestraint.insert(restraint_data, skip_duplicates=True)
