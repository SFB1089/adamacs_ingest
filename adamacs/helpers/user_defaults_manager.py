#!/usr/bin/env python3
"""
User Defaults Manager for ADAMACS

Manages user-specific configuration settings stored in INI files.
Provides methods to load, save, and manage user defaults for GUI widgets.

Author: System
Date: 2025-08-13
"""

import os
import configparser
from typing import List, Dict, Any, Union, Optional
import ast


class UserDefaultsManager:
    """
    Manages user-specific defaults stored in INI configuration files.
    
    Handles loading, saving, and converting user preferences for:
    - Project/location/equipment indices
    - DLC model selections (supporting multiple models per camera)
    - Camera configurations
    - AUX setup types
    - Recording notes options
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the UserDefaultsManager.
        
        Args:
            config_dir: Directory containing user INI files. 
                       If None, auto-detects based on script location.
        """
        if config_dir:
            self.config_dir = config_dir
        else:
            # Auto-detect config directory relative to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up two levels: helpers -> adamacs -> root, then add user_configs
            self.config_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'user_configs')
        
        # Ensure config directory exists
        os.makedirs(self.config_dir, exist_ok=True)
    
    def _parse_list_value(self, value: str) -> Union[List, int, str]:
        """
        Parse a string value that might be a list, integer, or string.
        
        Args:
            value: String value from INI file
            
        Returns:
            Parsed value as appropriate type
        """
        value = value.strip()
        
        # Handle list format: [1, 2, 3] or [12, 14, 15]
        if value.startswith('[') and value.endswith(']'):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # Fallback: split by comma and convert to integers
                try:
                    inner = value[1:-1].strip()
                    if not inner:
                        return []
                    return [int(x.strip()) for x in inner.split(',')]
                except ValueError:
                    return []
        
        # Handle comma-separated values without brackets
        if ',' in value:
            try:
                return [item.strip() for item in value.split(',')]
            except:
                return [value]
        
        # Try to parse as integer
        try:
            return int(value)
        except ValueError:
            # Return as string
            return value
    
    def _format_list_value(self, value: Union[List, int, str]) -> str:
        """
        Format a value for writing to INI file.
        
        Args:
            value: Value to format
            
        Returns:
            String representation for INI file
        """
        if isinstance(value, list):
            if len(value) == 0:
                return '[]'
            elif all(isinstance(x, int) for x in value):
                return str(value)  # [1, 2, 3]
            else:
                return ', '.join(str(x) for x in value)
        else:
            return str(value)
    
    def get_user_ini_path(self, user_initials: str) -> str:
        """Get the full path to a user's INI file."""
        return os.path.join(self.config_dir, f"{user_initials}.ini")
    
    def load_user_defaults(self, user_initials: str) -> Dict[str, Any]:
        """
        Load user defaults from the appropriate INI file.
        
        Args:
            user_initials: Two-letter user initials (e.g., 'TR', 'NK')
            
        Returns:
            Dictionary containing user defaults with all necessary keys.
            Returns hardcoded fallbacks if file is missing or corrupted.
        """
        config_file = os.path.join(self.config_dir, f'{user_initials}.ini')
        
        # Hardcoded fallback defaults
        fallback_defaults = {
            'project_idx': 0,
            'location_idx': 0, 
            'equipment_idx': 0,
            's2p_param_idx': 0,
            'dlc1_indices': [],
            'dlc2_indices': [],
            'dlc3_indices': [],
            'camera1': 'camera_1',
            'camera2': 'camera_2', 
            'camera3': 'camera_3',
            'aux_setup_type': 'behavior_box',
            'use_dlc_cropping': True,  # Default to True for dynamic cropping
            'recording_notes': ['', 'no comment', 'shaping', 'probe', 'recall', 'extinction', 'reversal', 'baseline', 'training', 'test session', 'habituation']
        }
        
        # If config file doesn't exist, return fallback defaults
        if not os.path.exists(config_file):
            return fallback_defaults
            
        try:
            config = configparser.ConfigParser()
            config.read(config_file)
            
            # Parse defaults section with robust error handling
            defaults = {}
            if config.has_section('defaults'):
                defaults_section = config['defaults']
                defaults.update({
                    'project_idx': defaults_section.getint('project_idx', fallback=fallback_defaults['project_idx']),
                    'location_idx': defaults_section.getint('location_idx', fallback=fallback_defaults['location_idx']),
                    'equipment_idx': defaults_section.getint('equipment_idx', fallback=fallback_defaults['equipment_idx']),
                    's2p_param_idx': defaults_section.getint('s2p_param_idx', fallback=fallback_defaults['s2p_param_idx']),
                    'dlc1_indices': self._parse_list_value(defaults_section.get('dlc1_indices', str(fallback_defaults['dlc1_indices']))),
                    'dlc2_indices': self._parse_list_value(defaults_section.get('dlc2_indices', str(fallback_defaults['dlc2_indices']))),
                    'dlc3_indices': self._parse_list_value(defaults_section.get('dlc3_indices', str(fallback_defaults['dlc3_indices']))),
                    'use_dlc_cropping': defaults_section.getboolean('use_dlc_cropping', fallback=fallback_defaults['use_dlc_cropping'])
                })
            
            # Parse cameras section  
            if config.has_section('cameras'):
                cameras_section = config['cameras']
                defaults.update({
                    'camera1': cameras_section.get('camera1', fallback=fallback_defaults['camera1']),
                    'camera2': cameras_section.get('camera2', fallback=fallback_defaults['camera2']),
                    'camera3': cameras_section.get('camera3', fallback=fallback_defaults['camera3'])
                })
            
            # Parse setup section
            if config.has_section('setup'):
                setup_section = config['setup']
                defaults.update({
                    'aux_setup_type': setup_section.get('aux_setup_type', fallback=fallback_defaults['aux_setup_type'])
                })
            
            # Parse recording notes section
            if config.has_section('recording_notes'):
                notes_section = config['recording_notes']
                recording_notes = []
                for key in sorted(notes_section.keys()):
                    if key.startswith('note'):
                        recording_notes.append(notes_section[key])
                defaults['recording_notes'] = recording_notes if recording_notes else fallback_defaults['recording_notes']
            
            # Fill in any missing keys with fallback values
            for key, fallback_value in fallback_defaults.items():
                if key not in defaults:
                    defaults[key] = fallback_value
                    
            return defaults
            
        except Exception as e:
            print(f"Warning: Could not parse config file {config_file}: {e}")
            return fallback_defaults
    
    def save_user_defaults(self, user_initials: str, defaults: Dict[str, Any]) -> bool:
        """
        Save user defaults to INI file.
        
        Args:
            user_initials: User initials
            defaults: Dictionary of defaults to save
            
        Returns:
            True if successful, False otherwise
        """
        ini_path = self.get_user_ini_path(user_initials)
        
        try:
            config = configparser.ConfigParser()
            
            # Create sections
            config['defaults'] = {}
            config['cameras'] = {}
            config['setup'] = {}
            config['recording_notes'] = {}
            
            # Save default values
            for key in ['project_idx', 'location_idx', 'equipment_idx', 's2p_param_idx', 
                       'dlc1_indices', 'dlc2_indices', 'dlc3_indices']:
                if key in defaults:
                    config['defaults'][key] = self._format_list_value(defaults[key])
            
            # Save boolean DLC cropping setting
            if 'use_dlc_cropping' in defaults:
                config['defaults']['use_dlc_cropping'] = str(defaults['use_dlc_cropping'])
            
            # Save camera settings
            for i, key in enumerate(['camera1', 'camera2', 'camera3'], 1):
                if key in defaults:
                    config['cameras'][f'camera{i}'] = str(defaults[key])
            
            # Save setup
            if 'aux_setup_type' in defaults:
                config['setup']['aux_setup_type'] = str(defaults['aux_setup_type'])
            
            # Save recording notes if available
            if 'recording_notes' in defaults:
                notes = defaults['recording_notes']
                if isinstance(notes, list):
                    for i, note in enumerate(notes):
                        config['recording_notes'][f'note{i}'] = str(note)
            
            # Write to file
            with open(ini_path, 'w') as f:
                config.write(f)
            
            return True
            
        except Exception as e:
            print(f"Error saving user defaults for {user_initials}: {e}")
            return False
    
    def get_user_cameras(self, user_initials: str) -> List[str]:
        """
        Get user camera configuration.
        
        Args:
            user_initials: User initials
            
        Returns:
            List of camera names
        """
        defaults = self.load_user_defaults(user_initials)
        return [defaults['camera1'], defaults['camera2'], defaults['camera3']]
    
    def get_user_array_format(self, user_initials: str) -> List:
        """
        Get user defaults in the legacy array format for backward compatibility.
        
        Args:
            user_initials: User initials
            
        Returns:
            Array in format [project, location, equipment, s2p_param, dlc1, dlc2, dlc3]
        """
        defaults = self.load_user_defaults(user_initials)
        
        return [
            defaults['project_idx'],
            defaults['location_idx'],
            defaults['equipment_idx'],
            defaults['s2p_param_idx'],
            defaults['dlc1_indices'],
            defaults['dlc2_indices'],
            defaults['dlc3_indices']
        ]
    
    def get_user_aux_setup(self, user_initials: str) -> str:
        """
        Get user AUX setup type.
        
        Args:
            user_initials: User initials
            
        Returns:
            AUX setup type string
        """
        defaults = self.load_user_defaults(user_initials)
        return defaults.get('aux_setup_type', 'bench2p')
    
    def get_user_recording_notes(self, user_initials: str) -> List[str]:
        """
        Get user recording notes options.
        
        Args:
            user_initials: User initials
            
        Returns:
            List of recording note options
        """
        ini_path = self.get_user_ini_path(user_initials)
        
        # Default fallback recording notes
        default_notes = ['', 'no comment', 'baseline', 'training', 'test session']
        
        if not os.path.exists(ini_path):
            return default_notes
        
        try:
            config = configparser.ConfigParser()
            config.read(ini_path)
            
            # Check if recording_notes section exists
            if 'recording_notes' in config:
                notes = []
                # Get all note entries (note0, note1, note2, etc.)
                for key in sorted(config['recording_notes'].keys()):
                    if key.startswith('note'):
                        notes.append(config['recording_notes'][key].strip())
                
                # Return notes if we found any, otherwise return defaults
                return notes if notes else default_notes
            
            return default_notes
            
        except Exception as e:
            print(f"Error loading recording notes for {user_initials}: {e}")
            return default_notes
    
    def get_user_dlc_cropping(self, user_initials: str) -> bool:
        """
        Get DLC cropping setting for a user.
        
        Args:
            user_initials: User initials
            
        Returns:
            Boolean indicating if DLC cropping should be used (default: True)
        """
        try:
            defaults = self.load_user_defaults(user_initials)
            return defaults.get('use_dlc_cropping', True)
        except Exception:
            return True  # Default to True if error
    
    def list_users(self) -> List[str]:
        """
        List all available user configurations.
        
        Returns:
            List of user initials that have INI files
        """
        if not os.path.exists(self.config_dir):
            return []
        
        users = []
        for filename in os.listdir(self.config_dir):
            if filename.endswith('.ini'):
                users.append(filename[:-4])  # Remove .ini extension
        
        return sorted(users)
    
    def convert_legacy_format(self, user_initials: str, legacy_array: List, 
                            camera_array: List[str], aux_setup: str, 
                            recording_notes: Optional[List[str]] = None) -> bool:
        """
        Convert legacy hardcoded format to INI file.
        
        Args:
            user_initials: User initials
            legacy_array: Array in format [project, location, equipment, s2p_param, dlc1, dlc2, dlc3]
            camera_array: List of camera names
            aux_setup: AUX setup type
            recording_notes: List of recording note options
            
        Returns:
            True if successful, False otherwise
        """
        defaults = {
            'project_idx': legacy_array[0],
            'location_idx': legacy_array[1], 
            'equipment_idx': legacy_array[2],
            's2p_param_idx': legacy_array[3],
            'dlc1_indices': legacy_array[4],
            'dlc2_indices': legacy_array[5],
            'dlc3_indices': legacy_array[6],
            'camera1': camera_array[0] if len(camera_array) > 0 else 'camera1',
            'camera2': camera_array[1] if len(camera_array) > 1 else 'camera2',
            'camera3': camera_array[2] if len(camera_array) > 2 else 'camera3',
            'aux_setup_type': aux_setup
        }
        
        if recording_notes:
            defaults['recording_notes'] = recording_notes
        
        return self.save_user_defaults(user_initials, defaults)


# For backward compatibility and testing
if __name__ == "__main__":
    # Test the UserDefaultsManager
    manager = UserDefaultsManager()
    
    # Test loading defaults
    test_defaults = manager.load_user_defaults('TEST')
    print("Test defaults:", test_defaults)
    
    # Test saving defaults
    test_save = {
        'project_idx': 1,
        'location_idx': 2,
        'equipment_idx': 0,
        's2p_param_idx': 5,
        'dlc1_indices': [12, 14, 15],
        'dlc2_indices': 8,
        'dlc3_indices': 8,
        'camera1': 'mini2p1_top',
        'camera2': 'mini2p1_eye_left', 
        'camera3': 'mini2p1_eye_right',
        'aux_setup_type': 'mini2p1_openfield'
    }
    
    success = manager.save_user_defaults('TEST', test_save)
    print("Save successful:", success)
    
    # Test loading back
    loaded = manager.load_user_defaults('TEST')
    print("Loaded back:", loaded)
