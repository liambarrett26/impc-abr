"""
Test script to verify the updated GMM loader.py now loads all 59,145 mice.
"""

import sys
sys.path.append('/home/liamb/impc-abr/scripts/abr_clustering/gmm')

from loader import IMPCABRLoader, load_impc_data


def test_updated_loader():
    """Test the updated GMM loader."""
    data_path = "/home/liamb/impc-abr/data/processed/abr_full_data.csv"
    
    print("Testing updated GMM loader.py")
    print("=" * 60)
    
    # Test using the loader class directly
    loader = IMPCABRLoader(data_path)
    
    # Load raw data
    raw_data = loader.load_raw_data()
    print(f"Raw data loaded: {len(raw_data)} rows")
    
    # Apply quality filters
    filtered_data = loader.apply_quality_filters(raw_data)
    print(f"After quality filters: {len(filtered_data)} rows")
    
    # Check unique mice
    if 'specimen_id' in filtered_data.columns:
        n_mice = filtered_data['specimen_id'].nunique()
        print(f"Unique mice: {n_mice}")
        
        if n_mice == 59145:
            print("\n✓ SUCCESS: Loader now retains all 59,145 mice!")
        else:
            print(f"\n✗ MISMATCH: Expected 59,145 mice, got {n_mice}")
    
    # Test convenience function
    print("\n" + "=" * 60)
    print("Testing convenience function load_impc_data()")
    df, groups = load_impc_data(data_path)
    print(f"Loaded {len(df)} mice in {len(groups)} experimental groups")
    
    # Check value ranges
    print("\nValue ranges after filtering:")
    for col in loader.abr_columns:
        if col in df.columns:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                print(f"  {col}: [{non_null.min():.1f}, {non_null.max():.1f}]")


if __name__ == "__main__":
    test_updated_loader()