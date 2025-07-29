"""
Test script to verify that the relaxed GMM loader now matches the Bayes loader
in terms of mouse count (59,145 mice).
"""

import sys
import pandas as pd
import numpy as np

# Add paths for imports
sys.path.append('/home/liamb/impc-abr/scripts/abr_analysis')
sys.path.append('/home/liamb/impc-abr/scripts/abr_clustering/utils')

from abr_analysis.data.loader import ABRDataLoader as BayesLoader
from relaxed_gmm_loader import RelaxedGMMLoader


def compare_relaxed_loaders(data_path):
    """Compare the Bayes loader with the relaxed GMM loader."""
    print("=" * 60)
    print("Comparing Bayes and Relaxed GMM Data Loaders")
    print(f"Data file: {data_path}")
    print("=" * 60)
    
    # Load with Bayes loader
    print("\n=== Bayes Analysis Loader ===")
    bayes_loader = BayesLoader(data_path)
    bayes_data = bayes_loader.load_data()
    print(f"Total rows loaded: {len(bayes_data)}")
    
    if 'specimen_id' in bayes_data.columns:
        bayes_mice = bayes_data['specimen_id'].nunique()
        print(f"Unique mice: {bayes_mice}")
    
    # Load with relaxed GMM loader
    print("\n=== Relaxed GMM Loader ===")
    gmm_loader = RelaxedGMMLoader(data_path)
    raw_data = gmm_loader.load_raw_data()
    gmm_data = gmm_loader.apply_quality_filters(raw_data)
    print(f"Total rows after relaxed filtering: {len(gmm_data)}")
    
    if 'specimen_id' in gmm_data.columns:
        gmm_mice = gmm_data['specimen_id'].nunique()
        print(f"Unique mice: {gmm_mice}")
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    if 'specimen_id' in bayes_data.columns and 'specimen_id' in gmm_data.columns:
        print(f"\nMouse counts match: {bayes_mice == gmm_mice}")
        print(f"  Bayes loader:        {bayes_mice:,} mice")
        print(f"  Relaxed GMM loader:  {gmm_mice:,} mice")
        
        if bayes_mice == gmm_mice:
            print("\n✓ SUCCESS: Both loaders now return the same number of mice!")
        else:
            print(f"\n✗ MISMATCH: Difference of {abs(bayes_mice - gmm_mice)} mice")
    
    # Check data transformations
    print("\n=== Data Transformations Applied ===")
    
    # Check for rounded values in GMM data
    for col in gmm_loader.abr_columns:
        if col in gmm_data.columns:
            # Check if all non-null values are multiples of 5
            non_null_values = gmm_data[col].dropna()
            if len(non_null_values) > 0:
                rounded_count = (non_null_values % 5 == 0).sum()
                print(f"\n{col}:")
                print(f"  Values rounded to 5 dB: {rounded_count}/{len(non_null_values)}")
                print(f"  Range: [{non_null_values.min():.1f}, {non_null_values.max():.1f}]")
    
    # Compare missing data patterns
    print("\n=== Missing Data Comparison ===")
    freq_cols = [col for col in bayes_data.columns if 'kHz-evoked ABR Threshold' in col]
    
    for col in freq_cols:
        bayes_missing = bayes_data[col].isnull().sum()
        gmm_missing = gmm_data[col].isnull().sum()
        print(f"{col}: Bayes={bayes_missing}, GMM={gmm_missing}")
    
    return bayes_data, gmm_data


if __name__ == "__main__":
    # Default data path
    data_path = "/home/liamb/impc-abr/data/processed/abr_full_data.csv"
    
    # Allow override via command line
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    # Run comparison
    bayes_data, gmm_data = compare_relaxed_loaders(data_path)