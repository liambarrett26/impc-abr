"""
Preprocessing module for IMPC ABR data normalization and standardization.

Implements the two-stage normalization approach:
1. Frequency-specific z-score normalization within testing center/equipment groups
2. Global min-max scaling to [0,1] range
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing parameters."""
    abr_columns: List[str]
    grouping_columns: List[str]
    target_range: Tuple[float, float] = (0.0, 1.0)
    center_threshold: int = 5  # Minimum mice per center/equipment group


class ABRPreprocessor:
    """
    Preprocessor for ABR audiometric data with batch effect correction.

    Implements frequency-specific normalization within technical groups
    followed by global scaling to ensure equal feature weighting.
    """

    def __init__(self, config: PreprocessingConfig):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Preprocessing configuration parameters
        """
        self.config = config
        self.group_scalers: Dict[str, Dict[str, StandardScaler]] = {}
        self.global_scaler: Optional[MinMaxScaler] = None
        self.is_fitted = False

    def _create_technical_groups(self, df: pd.DataFrame) -> pd.Series:
        """
        Create technical grouping variable from center/equipment metadata.

        Args:
            df: Input dataframe with metadata

        Returns:
            Series with technical group labels
        """
        # Combine available grouping columns
        available_cols = [col for col in self.config.grouping_columns if col in df.columns]

        if not available_cols:
            logger.warning("No grouping columns found, using single group")
            return pd.Series('default_group', index=df.index)

        # Create combined grouping variable
        group_parts = []
        for col in available_cols:
            group_parts.append(df[col].astype(str))

        # Fix: Use pd.Series.str.cat() instead of '_'.join() for pandas Series
        if len(group_parts) > 1:
            technical_groups = group_parts[0].str.cat(group_parts[1:], sep='_', na_rep='missing')
        else:
            technical_groups = group_parts[0].fillna('missing')

        return technical_groups

    def _validate_group_sizes(self, df: pd.DataFrame, technical_groups: pd.Series) -> pd.Series:
        """
        Filter out technical groups with insufficient sample sizes.

        Args:
            df: Input dataframe
            technical_groups: Technical group assignments

        Returns:
            Validated technical groups with small groups merged
        """
        group_counts = technical_groups.value_counts()
        small_groups = group_counts[group_counts < self.config.center_threshold].index

        if len(small_groups) > 0:
            logger.info(f"Merging {len(small_groups)} small technical groups into 'other'")
            technical_groups = technical_groups.replace(small_groups, 'other_combined')

        final_counts = technical_groups.value_counts()
        logger.info(f"Technical groups: {len(final_counts)} groups, sizes: {dict(final_counts)}")

        return technical_groups

    def fit(self, df: pd.DataFrame) -> 'ABRPreprocessor':
        """
        Fit preprocessing transformations on training data.

        Args:
            df: Training dataframe with ABR measurements

        Returns:
            Self for method chaining
        """
        logger.info("Fitting ABR preprocessing transformations")

        # Validate input data
        missing_cols = [col for col in self.config.abr_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing ABR columns: {missing_cols}")

        # Create technical groups
        technical_groups = self._create_technical_groups(df)
        technical_groups = self._validate_group_sizes(df, technical_groups)

        # Stage 1: Fit frequency-specific scalers within technical groups
        self.group_scalers = {}
        z_scored_data = df[self.config.abr_columns].copy()

        for group_name in technical_groups.unique():
            group_mask = technical_groups == group_name
            group_data = df.loc[group_mask, self.config.abr_columns]

            if len(group_data) == 0:
                continue

            # Fit scaler for each frequency within this technical group
            self.group_scalers[group_name] = {}

            for freq_col in self.config.abr_columns:
                scaler = StandardScaler()
                freq_data = group_data[freq_col].values.reshape(-1, 1)

                # Handle edge case of zero variance
                if np.std(freq_data) > 1e-8:
                    scaler.fit(freq_data)
                    z_scored_data.loc[group_mask, freq_col] = scaler.transform(freq_data).flatten()
                else:
                    logger.warning(f"Zero variance in {freq_col} for group {group_name}")
                    # Use identity transformation for zero variance
                    scaler.mean_ = np.array([np.mean(freq_data)])
                    scaler.scale_ = np.array([1.0])

                self.group_scalers[group_name][freq_col] = scaler

        # Stage 2: Fit global min-max scaler
        self.global_scaler = MinMaxScaler(feature_range=self.config.target_range)
        self.global_scaler.fit(z_scored_data.values)

        self.is_fitted = True
        self._log_fit_summary(df, technical_groups)

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform ABR data using fitted preprocessing.

        Args:
            df: Dataframe to transform

        Returns:
            Normalized ABR data as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Create technical groups
        technical_groups = self._create_technical_groups(df)
        technical_groups = self._validate_group_sizes(df, technical_groups)

        # Stage 1: Apply frequency-specific z-score normalization
        z_scored_data = df[self.config.abr_columns].copy()

        for group_name in technical_groups.unique():
            group_mask = technical_groups == group_name

            # Use fitted scaler if available, otherwise use 'other_combined'
            if group_name in self.group_scalers:
                group_scalers = self.group_scalers[group_name]
            elif 'other_combined' in self.group_scalers:
                group_scalers = self.group_scalers['other_combined']
                logger.info(f"Using 'other_combined' scaler for unseen group: {group_name}")
            else:
                # Fallback: use first available group's scalers
                fallback_group = list(self.group_scalers.keys())[0]
                group_scalers = self.group_scalers[fallback_group]
                logger.warning(f"Using fallback scaler '{fallback_group}' for group: {group_name}")

            group_data = df.loc[group_mask, self.config.abr_columns]

            for freq_col in self.config.abr_columns:
                if freq_col in group_scalers:
                    freq_data = group_data[freq_col].values.reshape(-1, 1)
                    z_scored_data.loc[group_mask, freq_col] = group_scalers[freq_col].transform(freq_data).flatten()

        # Stage 2: Apply global min-max scaling
        normalized_data = self.global_scaler.transform(z_scored_data.values)

        return normalized_data

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            df: Input dataframe

        Returns:
            Normalized ABR data
        """
        return self.fit(df).transform(df)

    def inverse_transform(self, normalized_data: np.ndarray,
                         technical_group: str = None) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.

        Args:
            normalized_data: Normalized ABR data
            technical_group: Technical group for inverse z-score transform

        Returns:
            Data in original scale
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse transform")

        # Stage 1: Inverse global scaling
        z_scored_data = self.global_scaler.inverse_transform(normalized_data)

        # Stage 2: Inverse z-score (requires specifying technical group)
        if technical_group and technical_group in self.group_scalers:
            group_scalers = self.group_scalers[technical_group]
            original_data = np.zeros_like(z_scored_data)

            for i, freq_col in enumerate(self.config.abr_columns):
                if freq_col in group_scalers:
                    freq_data = z_scored_data[:, i].reshape(-1, 1)
                    original_data[:, i] = group_scalers[freq_col].inverse_transform(freq_data).flatten()
                else:
                    original_data[:, i] = z_scored_data[:, i]
        else:
            logger.warning("Cannot perform inverse z-score without technical group info")
            original_data = z_scored_data

        return original_data

    def get_feature_names(self) -> List[str]:
        """Get standardized feature names for normalized data."""
        return [f"normalized_{col.replace('kHz-evoked ABR Threshold', 'kHz')}"
                for col in self.config.abr_columns]

    def _log_fit_summary(self, df: pd.DataFrame, technical_groups: pd.Series):
        """Log summary of preprocessing fit."""
        logger.info("=== Preprocessing Fit Summary ===")
        logger.info(f"Input shape: {df[self.config.abr_columns].shape}")
        logger.info(f"Technical groups: {len(technical_groups.unique())}")

        # Log scaling statistics
        for group_name, group_scalers in self.group_scalers.items():
            group_size = (technical_groups == group_name).sum()
            logger.info(f"Group '{group_name}': {group_size} mice, {len(group_scalers)} frequencies scaled")

        # Log global scaling range
        if self.global_scaler:
            data_min = self.global_scaler.data_min_
            data_max = self.global_scaler.data_max_
            logger.info(f"Global scaling - Min: {data_min}, Max: {data_max}")

        logger.info("=== End Preprocessing Summary ===")


def create_default_config(abr_columns: List[str] = None,
                         grouping_columns: List[str] = None) -> PreprocessingConfig:
    """
    Create default preprocessing configuration.

    Args:
        abr_columns: ABR threshold column names
        grouping_columns: Technical grouping column names

    Returns:
        Default preprocessing configuration
    """
    if abr_columns is None:
        abr_columns = [
            '6kHz-evoked ABR Threshold',
            '12kHz-evoked ABR Threshold',
            '18kHz-evoked ABR Threshold',
            '24kHz-evoked ABR Threshold',
            '30kHz-evoked ABR Threshold'
        ]

    if grouping_columns is None:
        grouping_columns = [
            'phenotyping_center',
            'pipeline_name',
            'metadata_Equipment manufacturer',
            'metadata_Equipment model'
        ]

    return PreprocessingConfig(
        abr_columns=abr_columns,
        grouping_columns=grouping_columns,
        target_range=(0.0, 1.0),
        center_threshold=5
    )


def preprocess_abr_data(df: pd.DataFrame,
                       config: PreprocessingConfig = None) -> Tuple[np.ndarray, ABRPreprocessor]:
    """
    Convenience function to preprocess ABR data with default settings.

    Args:
        df: Input dataframe with ABR measurements
        config: Preprocessing configuration (uses default if None)

    Returns:
        Tuple of (normalized_data, fitted_preprocessor)
    """
    if config is None:
        config = create_default_config()

    preprocessor = ABRPreprocessor(config)
    normalized_data = preprocessor.fit_transform(df)

    return normalized_data, preprocessor


if __name__ == "__main__":
    # Example usage and testing
    import sys
    from loader import load_impc_data

    if len(sys.argv) < 2:
        print("Usage: python preproc.py <data_path>")
        sys.exit(1)

    # Load sample data
    data_path = sys.argv[1]
    df, _ = load_impc_data(data_path)

    # Test preprocessing
    config = create_default_config()
    normalized_data, preprocessor = preprocess_abr_data(df, config)

    print(f"Original data shape: {df[config.abr_columns].shape}")
    print(f"Normalized data shape: {normalized_data.shape}")
    print(f"Normalized data range: [{normalized_data.min():.3f}, {normalized_data.max():.3f}]")
    print("Feature names:", preprocessor.get_feature_names())

    # Test inverse transform (using first technical group)
    if preprocessor.group_scalers:
        first_group = list(preprocessor.group_scalers.keys())[0]
        restored_data = preprocessor.inverse_transform(normalized_data[:10], first_group)
        print(f"Inverse transform test shape: {restored_data.shape}")