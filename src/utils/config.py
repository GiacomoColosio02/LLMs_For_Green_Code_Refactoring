"""
Configuration loader for measurement settings.
"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_config(config_name: str = 'measurement_config.yaml') -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_name: Name of the config file in configs/ directory
        
    Returns:
        Dictionary with configuration parameters
    """
    config_path = get_project_root() / 'configs' / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_grid_intensities() -> Dict[str, Any]:
    """
    Load carbon grid intensities by country.
    
    Returns:
        Dictionary with grid intensities (gCO2e/kWh) per country
    """
    grid_path = get_project_root() / 'configs' / 'grid_intensities.json'
    
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid intensities file not found: {grid_path}")
    
    with open(grid_path, 'r') as f:
        data = json.load(f)
    
    return data


def get_grid_intensity(country_code: str = None) -> float:
    """
    Get carbon intensity for a specific country.
    
    Args:
        country_code: ISO 3-letter country code (e.g., 'ESP', 'ITA')
                      If None, returns default from config
        
    Returns:
        Carbon intensity in gCO2e/kWh
    """
    grid_data = load_grid_intensities()
    config = load_config()
    
    if country_code is None:
        # Use default from config
        return config['carbon']['grid_intensity_default']
    
    intensities = grid_data['intensities']
    
    if country_code not in intensities:
        print(f"Warning: Country '{country_code}' not found. Using default.")
        return config['carbon']['grid_intensity_default']
    
    return intensities[country_code]


if __name__ == "__main__":
    # Test the config loader
    print("Testing config loader...")
    
    config = load_config()
    print(f"Config loaded: {config['measurement']['repetitions']} repetitions")
    
    grid_data = load_grid_intensities()
    print(f"Grid data loaded: {len(grid_data['intensities'])} countries")
    
    intensity_esp = get_grid_intensity('ESP')
    print(f"Spain intensity: {intensity_esp} gCO2e/kWh")
    
    print("\n All tests passed!")