import sys
import types

if "datajoint" not in sys.modules:
    dj_stub = types.ModuleType("datajoint")
    dj_stub.config = {}
    sys.modules["datajoint"] = dj_stub

from adamacs.helpers.user_defaults_manager import UserDefaultsManager


def test_save_and_load_user_defaults(tmp_path):
    manager = UserDefaultsManager(config_dir=str(tmp_path))
    payload = {
        "project_idx": 1,
        "location_idx": 2,
        "equipment_idx": 3,
        "s2p_param_idx": 4,
        "dlc1_indices": [1, 2],
        "dlc2_indices": 5,
        "dlc3_indices": 6,
        "camera1": "cam_a",
        "camera2": "cam_b",
        "camera3": "cam_c",
        "aux_setup_type": "mini2p1_openfield",
        "recording_notes": ["", "baseline"],
    }

    assert manager.save_user_defaults("ZZ", payload)

    loaded = manager.load_user_defaults("ZZ")
    assert loaded["project_idx"] == 1
    assert loaded["camera1"] == "cam_a"
    assert loaded["aux_setup_type"] == "mini2p1_openfield"
