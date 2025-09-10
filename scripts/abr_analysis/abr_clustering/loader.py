"""
Data loader for IMPC ABR clustering analysis.

This module handles loading and initial filtering of IMPC ABR data for
audiometric phenotype discovery using Gaussian Mixture Models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IMPCABRLoader:
    """
    Loader for IMPC ABR data with filtering and quality control.

    Handles loading of raw IMPC data and applies initial filtering criteria
    including minimum sample sizes and data completeness requirements.
    """

    def __init__(self, data_path: Path):
        """
        Initialize the loader.

        Args:
            data_path: Path to the IMPC ABR dataset file
        """
        self.data_path = Path(data_path)
        self.abr_columns = [
            '6kHz-evoked ABR Threshold',
            '12kHz-evoked ABR Threshold',
            '18kHz-evoked ABR Threshold',
            '24kHz-evoked ABR Threshold',
            '30kHz-evoked ABR Threshold'
        ]
        self.metadata_columns = [
            'specimen_id', 'gene_symbol', 'allele_symbol', 'zygosity',
            'sex', 'age_in_weeks', 'weight', 'phenotyping_center',
            'pipeline_name', 'genetic_background', 'biological_sample_group',
            'metadata_Equipment manufacturer', 'metadata_Equipment model'
        ]

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw IMPC ABR data from file.

        Returns:
            Raw dataframe with all IMPC ABR data

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If required columns are missing
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        logger.info(f"Loading data from {self.data_path}")

        # Determine file type and load accordingly
        if self.data_path.suffix.lower() == '.csv':
            df = pd.read_csv(self.data_path)
        elif self.data_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(self.data_path)
        elif self.data_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(self.data_path)
        else:
            # Try CSV as default
            df = pd.read_csv(self.data_path)

        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

        # Verify required columns exist
        missing_abr = [col for col in self.abr_columns if col not in df.columns]
        missing_meta = [col for col in self.metadata_columns if col not in df.columns]

        if missing_abr:
            raise ValueError(f"Missing required ABR columns: {missing_abr}")
        if missing_meta:
            logger.warning(f"Missing metadata columns: {missing_meta}")
            # Keep only available metadata columns
            self.metadata_columns = [col for col in self.metadata_columns if col in df.columns]

        return df

    def apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply quality control filters to the dataset.

        Args:
            df: Raw dataframe

        Returns:
            Filtered dataframe meeting quality criteria
        """
        logger.info("Applying quality control filters")
        initial_count = len(df)

        # Filter 1: Complete ABR data (all 5 frequencies)
        complete_abr = df[self.abr_columns].notna().all(axis=1)
        df = df[complete_abr].copy()
        logger.info(f"After complete ABR filter: {len(df)} rows ({initial_count - len(df)} removed)")

        # Filter 2: Physiologically plausible thresholds (0-100 dB SPL)
        valid_thresholds = True
        for col in self.abr_columns:
            valid_range = (df[col] >= 0) & (df[col] <= 100)
            valid_thresholds = valid_thresholds & valid_range

        df = df[valid_thresholds].copy()
        logger.info(f"After threshold range filter: {len(df)} rows")

        # Filter 3: Age filter (typical testing age around 14 weeks)
        if 'age_in_weeks' in df.columns:
            age_filter = (df['age_in_weeks'] >= 10) & (df['age_in_weeks'] <= 20)
            df = df[age_filter].copy()
            logger.info(f"After age filter (10-20 weeks): {len(df)} rows")

        # Filter 4: Remove rows with missing critical metadata
        critical_cols = ['sex', 'phenotyping_center', 'genetic_background']
        available_critical = [col for col in critical_cols if col in df.columns]

        for col in available_critical:
            df = df[df[col].notna()].copy()

        logger.info(f"Final filtered dataset: {len(df)} rows")
        return df

    def create_experimental_groups(self, df: pd.DataFrame,
                                 min_mutants: int = 3,
                                 min_controls: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Create experimental groups with matched controls.

        Args:
            df: Filtered dataframe
            min_mutants: Minimum number of mutant mice required
            min_controls: Minimum number of control mice required

        Returns:
            Dictionary mapping group IDs to dataframes with mutants and matched controls
        """
        logger.info("Creating experimental groups with matched controls")

        # Identify control mice (wild-type)
        if 'biological_sample_group' in df.columns:
            controls = df[df['biological_sample_group'] == 'control'].copy()
        else:
            # Fallback: assume baseline/wild-type based on gene_symbol
            controls = df[df['gene_symbol'].isna()].copy()

        # Identify mutant groups
        mutant_mice = df[df['biological_sample_group'] != 'control'].copy() if 'biological_sample_group' in df.columns else df[df['gene_symbol'].notna()].copy()

        experimental_groups = {}

        # Group mutants by gene, center, pipeline, and equipment
        grouping_cols = ['gene_symbol', 'phenotyping_center', 'pipeline_name', 'genetic_background']
        available_grouping = [col for col in grouping_cols if col in mutant_mice.columns]

        if 'metadata_Equipment manufacturer' in mutant_mice.columns:
            available_grouping.append('metadata_Equipment manufacturer')
        if 'metadata_Equipment model' in mutant_mice.columns:
            available_grouping.append('metadata_Equipment model')

        for group_id, mutant_group in mutant_mice.groupby(available_grouping):
            if len(mutant_group) < min_mutants:
                continue

            # Create matching criteria for controls
            match_dict = dict(zip(available_grouping, group_id if isinstance(group_id, tuple) else [group_id]))

            # Find matching controls
            matched_controls = controls.copy()
            for col, value in match_dict.items():
                if col != 'gene_symbol':  # Don't match on gene for controls
                    matched_controls = matched_controls[matched_controls[col] == value]

            if len(matched_controls) < min_controls:
                continue

            # Combine mutants and matched controls
            group_data = pd.concat([mutant_group, matched_controls], ignore_index=True)
            group_key = f"{match_dict['gene_symbol']}_{match_dict['phenotyping_center']}"
            experimental_groups[group_key] = group_data

        logger.info(f"Created {len(experimental_groups)} experimental groups")
        return experimental_groups

    def load_and_prepare(self, min_mutants: int = 3,
                        min_controls: int = 20) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Complete loading and preparation pipeline.

        Args:
            min_mutants: Minimum number of mutant mice per group
            min_controls: Minimum number of matched controls per group

        Returns:
            Tuple of (complete_filtered_dataset, experimental_groups_dict)
        """
        # Load raw data
        df = self.load_raw_data()

        # Apply quality filters
        df = self.apply_quality_filters(df)

        # Create experimental groups
        experimental_groups = self.create_experimental_groups(df, min_mutants, min_controls)

        # Log summary statistics
        self._log_summary_stats(df, experimental_groups)

        return df, experimental_groups

    def _log_summary_stats(self, df: pd.DataFrame, experimental_groups: Dict[str, pd.DataFrame]):
        """Log summary statistics about the loaded data."""
        logger.info("=== Data Loading Summary ===")
        logger.info(f"Total mice: {len(df)}")

        if 'biological_sample_group' in df.columns:
            sample_counts = df['biological_sample_group'].value_counts()
            logger.info(f"Sample groups: {dict(sample_counts)}")

        if 'sex' in df.columns:
            sex_counts = df['sex'].value_counts()
            logger.info(f"Sex distribution: {dict(sex_counts)}")

        if 'phenotyping_center' in df.columns:
            center_counts = df['phenotyping_center'].value_counts()
            logger.info(f"Centers: {len(center_counts)} centers with mice")

        if experimental_groups:
            group_sizes = [len(group) for group in experimental_groups.values()]
            logger.info(f"Experimental groups: {len(experimental_groups)} groups")
            logger.info(f"Group size range: {min(group_sizes)}-{max(group_sizes)} mice")

        logger.info("=== End Summary ===")


def load_impc_data(data_path: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Convenience function to load IMPC ABR data.

    Args:
        data_path: Path to IMPC data file
        **kwargs: Additional arguments passed to load_and_prepare

    Returns:
        Tuple of (complete_dataset, experimental_groups)
    """
    loader = IMPCABRLoader(data_path)
    return loader.load_and_prepare(**kwargs)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python loader.py <data_path>")
        sys.exit(1)

    data_path = sys.argv[1]
    df, groups = load_impc_data(data_path)

    print(f"Loaded {len(df)} mice in {len(groups)} experimental groups")
    print("First few rows of ABR data:")
    abr_cols = ['6kHz-evoked ABR Threshold', '12kHz-evoked ABR Threshold',
                '18kHz-evoked ABR Threshold', '24kHz-evoked ABR Threshold',
                '30kHz-evoked ABR Threshold']
    print(df[abr_cols].head())