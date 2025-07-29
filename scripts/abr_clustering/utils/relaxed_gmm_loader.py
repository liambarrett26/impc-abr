"""
Modified GMM loader with relaxed filtering to retain all mice from Bayes analysis.

Changes:
1. No age filtering
2. Round out-of-range values to nearest 5 dB interval
3. Keep mice with missing data (using imputation if needed)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import sys

# Add path for imports
sys.path.append('/home/liamb/impc-abr/scripts/abr_clustering/gmm')
from loader import IMPCABRLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RelaxedGMMLoader(IMPCABRLoader):
    """
    Modified IMPC ABR loader with relaxed filtering criteria.
    
    Removes age filtering and rounds out-of-range values to retain all mice.
    """
    
    def round_to_nearest_5(self, value: float) -> float:
        """Round a value to the nearest 5 dB interval."""
        return round(value / 5) * 5
    
    def apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply relaxed quality control filters to the dataset.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Filtered dataframe with all mice retained
        """
        logger.info("Applying relaxed quality control filters")
        initial_count = len(df)
        
        # First, handle missing data in ABR columns
        # Instead of removing rows with missing data, we'll impute or handle them
        for col in self.abr_columns:
            if col in df.columns:
                # Count missing values
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    logger.info(f"{col}: {missing_count} missing values")
                    # For now, we'll keep the missing values as NaN
                    # Could implement imputation here if needed
        
        # Round out-of-range values to nearest 5 dB interval
        for col in self.abr_columns:
            if col in df.columns:
                # Find values outside 0-100 range
                out_of_range = df[(df[col] < 0) | (df[col] > 100)][col]
                if len(out_of_range) > 0:
                    logger.info(f"{col}: {len(out_of_range)} values outside 0-100 dB range")
                    
                    # Round to nearest 5 dB and clip to 0-100 range
                    df.loc[df[col] < 0, col] = df.loc[df[col] < 0, col].apply(
                        lambda x: max(0, self.round_to_nearest_5(x))
                    )
                    df.loc[df[col] > 100, col] = df.loc[df[col] > 100, col].apply(
                        lambda x: min(100, self.round_to_nearest_5(x))
                    )
        
        # NO AGE FILTERING - removed the age filter that was in original
        
        # Log final statistics
        logger.info(f"Final filtered dataset: {len(df)} rows")
        logger.info(f"Retained all {initial_count} mice")
        
        return df
    
    def create_experimental_groups(self, df: pd.DataFrame,
                                 min_mutants: int = 3,
                                 min_controls: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Create experimental groups but with more relaxed criteria.
        
        Args:
            df: Filtered dataframe
            min_mutants: Minimum mutants per group (relaxed)
            min_controls: Minimum controls per group (relaxed)
            
        Returns:
            Dictionary of experimental groups
        """
        logger.info("Creating experimental groups with relaxed criteria")
        
        if 'biological_sample_group' not in df.columns:
            logger.warning("No biological_sample_group column found")
            return {}
        
        # Separate controls and mutants
        controls = df[df['biological_sample_group'] == 'control'].copy()
        mutants = df[df['biological_sample_group'] == 'experimental'].copy()
        
        logger.info(f"Found {len(controls)} controls and {len(mutants)} mutants")
        
        # Group by gene, center, background
        grouping_cols = []
        if 'gene_symbol' in mutants.columns:
            grouping_cols.append('gene_symbol')
        if 'phenotyping_center' in mutants.columns:
            grouping_cols.append('phenotyping_center')
        if 'genetic_background' in mutants.columns:
            grouping_cols.append('genetic_background')
        
        if not grouping_cols:
            logger.warning("No grouping columns available")
            return {}
        
        experimental_groups = {}
        
        for group_key, group_data in mutants.groupby(grouping_cols):
            if len(group_data) < min_mutants:
                continue
                
            # Create group identifier
            if isinstance(group_key, tuple):
                group_name = '_'.join(str(k) for k in group_key)
            else:
                group_name = str(group_key)
            
            # Find matched controls with more relaxed matching
            if 'phenotyping_center' in grouping_cols and 'genetic_background' in grouping_cols:
                idx = grouping_cols.index('phenotyping_center')
                center = group_key[idx] if isinstance(group_key, tuple) else group_key
                
                idx_bg = grouping_cols.index('genetic_background')
                background = group_key[idx_bg] if isinstance(group_key, tuple) else group_key
                
                # Try exact match first
                matched_controls = controls[
                    (controls['phenotyping_center'] == center) &
                    (controls['genetic_background'] == background)
                ]
                
                # If not enough controls, just use center
                if len(matched_controls) < min_controls:
                    matched_controls = controls[controls['phenotyping_center'] == center]
                
                # If still not enough, use all controls from that center
                if len(matched_controls) < min_controls:
                    logger.debug(f"Using all controls from center {center} for group {group_name}")
            else:
                # Just use all controls if we can't match properly
                matched_controls = controls
            
            if len(matched_controls) >= min_controls:
                # Combine mutants and controls
                combined_group = pd.concat([group_data, matched_controls])
                experimental_groups[group_name] = combined_group
        
        logger.info(f"Created {len(experimental_groups)} experimental groups")
        return experimental_groups


def test_relaxed_loader(data_path: str):
    """Test the relaxed GMM loader."""
    print("\n=== Testing Relaxed GMM Loader ===")
    
    loader = RelaxedGMMLoader(data_path)
    
    # Load raw data
    raw_data = loader.load_raw_data()
    print(f"Raw data loaded: {len(raw_data)} rows")
    
    # Apply relaxed quality filters
    filtered_data = loader.apply_quality_filters(raw_data)
    print(f"After relaxed filters: {len(filtered_data)} rows")
    
    # Check unique mice
    if 'specimen_id' in filtered_data.columns:
        n_mice = filtered_data['specimen_id'].nunique()
        print(f"Unique mice after filtering: {n_mice}")
    
    # Check for any remaining missing values
    print("\nRemaining missing values per frequency:")
    for col in loader.abr_columns:
        if col in filtered_data.columns:
            missing = filtered_data[col].isnull().sum()
            if missing > 0:
                print(f"  {col}: {missing} missing values")
    
    # Check value ranges after rounding
    print("\nValue ranges after rounding:")
    for col in loader.abr_columns:
        if col in filtered_data.columns:
            min_val = filtered_data[col].min()
            max_val = filtered_data[col].max()
            print(f"  {col}: [{min_val:.1f}, {max_val:.1f}]")
    
    return filtered_data


if __name__ == "__main__":
    # Default data path
    data_path = "/home/liamb/impc-abr/data/processed/abr_full_data.csv"
    
    # Allow override via command line
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    # Test the relaxed loader
    filtered_data = test_relaxed_loader(data_path)