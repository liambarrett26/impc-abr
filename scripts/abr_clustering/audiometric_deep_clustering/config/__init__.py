"""
Configuration module for ContrastiveVAE-DEC audiometric phenotype clustering.

This module provides configuration management for the deep learning clustering
approach to discover novel hearing loss phenotypes in mouse genetic data.
"""

import yaml
from pathlib import Path

def load_config(config_name: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_name: Name of config file (without .yaml extension)

    Returns:
        Dictionary containing configuration parameters
    """
    config_path = Path(__file__).parent / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def load_model_config() -> dict:
    """Load model architecture configuration."""
    return load_config('model_config')

def load_training_config() -> dict:
    """Load training process configuration."""
    return load_config('training_config')

def get_config_dir() -> Path:
    """Get the configuration directory path."""
    return Path(__file__).parent

__all__ = ['load_config', 'load_model_config', 'load_training_config', 'get_config_dir']
