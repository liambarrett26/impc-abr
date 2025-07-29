#!/usr/bin/env python3
"""
Extract original ABR data from metadata and create original_data.csv files
for existing clustering results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_original_data_files():
    """
    Extract original ABR data from shared metadata and create original_data.csv
    files for all existing clustering results.
    """
    # Load metadata with original ABR data
    metadata_path = Path("shared_data/metadata.csv")
    if not metadata_path.exists():
        raise FileNotFoundError("shared_data/metadata.csv not found")
    
    logger.info("Loading metadata with original ABR data")
    metadata_df = pd.read_csv(metadata_path)
    
    # ABR columns in original dB SPL scale
    abr_columns = [
        '6kHz-evoked ABR Threshold',
        '12kHz-evoked ABR Threshold', 
        '18kHz-evoked ABR Threshold',
        '24kHz-evoked ABR Threshold',
        '30kHz-evoked ABR Threshold'
    ]
    
    # Extract original ABR data
    original_data = metadata_df[abr_columns].copy()
    logger.info(f"Extracted original ABR data: {original_data.shape} samples")
    logger.info(f"Value range: {original_data.min().min():.1f} - {original_data.max().max():.1f} dB SPL")
    
    # Find all results directories (including nested ones)
    results_dirs = []
    results_base = Path("results")
    
    if results_base.exists():
        # Look for directories containing cluster_labels.npy recursively
        for cluster_file in results_base.rglob("cluster_labels.npy"):
            results_dir = cluster_file.parent
            results_dirs.append(results_dir)
    
    logger.info(f"Found {len(results_dirs)} clustering result directories")
    
    # Create original_data.csv for each results directory
    for results_dir in results_dirs:
        original_data_path = results_dir / "original_data.csv"
        original_data.to_csv(original_data_path, index=False)
        logger.info(f"Created: {original_data_path}")
    
    logger.info("All original_data.csv files created successfully!")
    return len(results_dirs)


if __name__ == "__main__":
    try:
        n_created = create_original_data_files()
        print(f"✅ Successfully created original_data.csv for {n_created} result directories")
        print("You can now regenerate visualizations with proper dB SPL scales using:")
        print("  python visualize_results.py results/gmm_k11_full/")
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)