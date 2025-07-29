"""
Test script to compare the number of mice loaded by the Bayes analysis loader 
and the GMM clustering loader.

This script loads data using both approaches and reports the differences in
mouse counts and filtering criteria.
"""

import sys
import pandas as pd
from pathlib import Path

# Add paths for imports
sys.path.append('/home/liamb/impc-abr/scripts/abr_analysis')
sys.path.append('/home/liamb/impc-abr/scripts/abr_clustering/gmm')

from abr_analysis.data.loader import ABRDataLoader as BayesLoader
from loader import IMPCABRLoader as GMMLoader


def test_bayes_loader(data_path):
    """Test the Bayes analysis data loader."""
    print("\n=== Testing Bayes Analysis Loader ===")
    
    loader = BayesLoader(data_path)
    data = loader.load_data()
    
    print(f"Total rows loaded: {len(data)}")
    print(f"Total columns: {len(data.columns)}")
    
    # Check for unique mice (assuming specimen_id is the mouse identifier)
    if 'specimen_id' in data.columns:
        n_mice = data['specimen_id'].nunique()
        print(f"Unique mice (by specimen_id): {n_mice}")
    else:
        print("No specimen_id column found")
    
    # Check frequencies available
    freq_cols = loader.get_frequencies()
    print(f"Frequency columns found: {len(freq_cols)}")
    print(f"Frequencies: {freq_cols}")
    
    # Check for missing data in frequency columns
    if freq_cols:
        missing_data = data[freq_cols].isnull().sum()
        print("\nMissing data per frequency:")
        for freq, count in missing_data.items():
            print(f"  {freq}: {count} missing values")
    
    return data


def test_gmm_loader(data_path):
    """Test the GMM clustering data loader."""
    print("\n=== Testing GMM Clustering Loader ===")
    
    loader = GMMLoader(data_path)
    
    # Load raw data
    raw_data = loader.load_raw_data()
    print(f"Raw data loaded: {len(raw_data)} rows")
    
    # Apply quality filters
    filtered_data = loader.apply_quality_filters(raw_data)
    print(f"After quality filters: {len(filtered_data)} rows")
    
    # Check unique mice
    if 'specimen_id' in filtered_data.columns:
        n_mice = filtered_data['specimen_id'].nunique()
        print(f"Unique mice after filtering: {n_mice}")
    
    # Create experimental groups
    groups = loader.create_experimental_groups(filtered_data)
    print(f"\nExperimental groups created: {len(groups)} groups")
    
    # Count total mice across all groups (groups is a dict of DataFrames)
    total_mice_in_groups = sum(len(group) for group in groups.values())
    print(f"Total mice in experimental groups: {total_mice_in_groups}")
    
    # Show filtering statistics
    print("\nFiltering statistics:")
    print(f"  Rows removed by quality filters: {len(raw_data) - len(filtered_data)}")
    if 'specimen_id' in raw_data.columns and 'specimen_id' in filtered_data.columns:
        mice_removed = raw_data['specimen_id'].nunique() - filtered_data['specimen_id'].nunique()
        print(f"  Unique mice removed: {mice_removed}")
    
    return filtered_data, groups


def compare_loaders(data_path):
    """Compare the two loaders and report differences."""
    print("=" * 60)
    print("Comparing Bayes and GMM Data Loaders")
    print(f"Data file: {data_path}")
    print("=" * 60)
    
    # Test Bayes loader
    bayes_data = test_bayes_loader(data_path)
    
    # Test GMM loader
    gmm_data, gmm_groups = test_gmm_loader(data_path)
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    if 'specimen_id' in bayes_data.columns and 'specimen_id' in gmm_data.columns:
        bayes_mice = bayes_data['specimen_id'].nunique()
        gmm_mice = gmm_data['specimen_id'].nunique()
        
        print(f"\nUnique mice counts:")
        print(f"  Bayes loader: {bayes_mice:,} mice")
        print(f"  GMM loader:   {gmm_mice:,} mice (after filtering)")
        print(f"  Difference:   {bayes_mice - gmm_mice:,} mice")
        print(f"  GMM retains:  {gmm_mice/bayes_mice*100:.1f}% of Bayes mice")
        
        # Find mice that are in Bayes but not in GMM
        bayes_ids = set(bayes_data['specimen_id'].unique())
        gmm_ids = set(gmm_data['specimen_id'].unique())
        removed_ids = bayes_ids - gmm_ids
        
        print(f"\nMice removed by GMM filtering: {len(removed_ids):,}")
        
        # Analyze why mice were removed (sample first few)
        if removed_ids:
            sample_removed = list(removed_ids)[:5]
            print(f"\nAnalyzing why first {len(sample_removed)} mice were removed:")
            for mouse_id in sample_removed:
                mouse_data = bayes_data[bayes_data['specimen_id'] == mouse_id].iloc[0]
                print(f"\n  Mouse {mouse_id}:")
                
                # Check each ABR frequency
                freq_cols = [col for col in bayes_data.columns if 'kHz-evoked ABR Threshold' in col]
                for freq in freq_cols:
                    value = mouse_data[freq]
                    print(f"    {freq}: {value}")
                
                # Check age
                if 'age_in_weeks' in mouse_data:
                    print(f"    Age: {mouse_data['age_in_weeks']} weeks")
                
                # Check other relevant fields
                if 'biological_sample_group' in mouse_data:
                    print(f"    Group: {mouse_data['biological_sample_group']}")


if __name__ == "__main__":
    # Default data path
    data_path = "/home/liamb/impc-abr/data/processed/abr_full_data.csv"
    
    # Allow override via command line
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    # Run comparison
    compare_loaders(data_path)