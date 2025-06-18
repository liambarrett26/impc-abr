"""
PyTorch-optimized data loader for Neural ADMIXTURE phenotypic analysis.
Handles mixed data types, missing values, and torch tensor conversion.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AdmixtureDataLoader:
    """
    Streamlined data loader for Neural ADMIXTURE with phenotypic data.
    Handles mixed data types and converts to PyTorch tensors.
    """

    def __init__(self):
        self.feature_columns = {
            # Continuous ABR features (highest priority)
            'abr_thresholds': [
                '6kHz-evoked ABR Threshold',
                '12kHz-evoked ABR Threshold',
                '18kHz-evoked ABR Threshold',
                '24kHz-evoked ABR Threshold',
                '30kHz-evoked ABR Threshold'
            ],
            'click_abr': ['Click-evoked ABR threshold'],

            # Continuous metadata
            'continuous_meta': ['weight'],

            # Categorical features
            'categorical': ['sex', 'zygosity', 'genetic_background', 'phenotyping_center'],

            # Datetime features (for age calculation)
            'datetime': ['date_of_birth', 'date_of_experiment']
        }

        self.required_columns = (
            self.feature_columns['abr_thresholds'] +
            self.feature_columns['click_abr'] +
            self.feature_columns['continuous_meta'] +
            self.feature_columns['categorical'] +
            self.feature_columns['datetime'] +
            ['gene_symbol']  # For gene association
        )

    def load_data(self, data_path: str, min_samples_per_gene: int = 3) -> Tuple[torch.Tensor, Dict]:
        """
        Load and preprocess data for Neural ADMIXTURE.

        Args:
            data_path: Path to IMPC data file
            min_samples_per_gene: Minimum samples per gene to include

        Returns:
            Tuple of (features_tensor, metadata_dict)
        """
        logger.info(f"Loading data from {data_path}")

        # Load raw data
        df = self._load_raw_data(data_path)

        # Quality control and filtering
        df = self._apply_quality_filters(df)

        # Gene filtering
        df = self._filter_genes(df, min_samples_per_gene)

        # Feature engineering
        features_df = self._engineer_features(df)

        # Convert to tensor
        features_tensor = torch.tensor(features_df.values, dtype=torch.float32)

        # Create metadata
        metadata = self._create_metadata(df, features_df)

        logger.info(f"Loaded {len(features_tensor)} samples with {features_tensor.shape[1]} features")
        return features_tensor, metadata

    def _load_raw_data(self, data_path: str) -> pd.DataFrame:
        """Load raw data file."""
        path = Path(data_path)

        if path.suffix.lower() == '.csv':
            df = pd.read_csv(path)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        elif path.suffix.lower() == '.parquet':
            df = pd.read_parquet(path)
        else:
            # Try CSV as default
            df = pd.read_csv(path)

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def _apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality control filters."""
        initial_count = len(df)

        # Check required columns exist
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Remove rows with missing ABR data (critical features) - FIRST
        critical_features = (
            self.feature_columns['abr_thresholds'] +
            self.feature_columns['click_abr']
        )

        for col in critical_features:
            df = df.dropna(subset=[col])

        logger.info(f"After removing missing ABR data: {len(df)} rows")

        # Remove rows with invalid ABR thresholds (0-100 dB SPL)
        for col in critical_features:
            df = df[(df[col] >= 0) & (df[col] <= 100)]

        logger.info(f"After ABR threshold validation: {len(df)} rows")

        # Remove rows with missing critical categorical data
        critical_categoricals = ['sex', 'zygosity', 'genetic_background']
        available_critical = [col for col in critical_categoricals if col in self.feature_columns['categorical']]

        for col in available_critical:
            df = df.dropna(subset=[col])

        logger.info(f"After removing missing critical categoricals: {len(df)} rows")

        # Remove rows with missing datetime data
        for col in self.feature_columns['datetime']:
            df = df.dropna(subset=[col])

        logger.info(f"After removing missing datetime data: {len(df)} rows")

        # Convert datetime columns with error handling
        for col in self.feature_columns['datetime']:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                # Remove rows where datetime conversion failed
                df = df.dropna(subset=[col])
            except Exception as e:
                logger.warning(f"Error converting {col} to datetime: {e}")

        # Handle weight data - allow NaN values but log them
        if 'weight' in df.columns:
            weight_nan_count = df['weight'].isna().sum()
            if weight_nan_count > 0:
                logger.info(f"Found {weight_nan_count} missing weight values - will be imputed during preprocessing")

        # Optional: Remove rows with missing phenotyping_center (if it affects matching)
        if 'phenotyping_center' in df.columns:
            df = df.dropna(subset=['phenotyping_center'])
            logger.info(f"After removing missing phenotyping_center: {len(df)} rows")

        logger.info(f"Quality filtering complete: {len(df)} rows remaining ({initial_count - len(df)} removed)")
        return df.reset_index(drop=True)

    def _filter_genes(self, df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
        """Filter genes with insufficient samples."""
        if 'gene_symbol' not in df.columns:
            logger.warning("No gene_symbol column found, skipping gene filtering")
            return df

        # Count samples per gene
        gene_counts = df['gene_symbol'].value_counts()
        valid_genes = gene_counts[gene_counts >= min_samples].index

        # Filter dataframe
        df_filtered = df[df['gene_symbol'].isin(valid_genes)].copy()

        logger.info(f"Gene filtering: {len(valid_genes)} genes with ≥{min_samples} samples")
        logger.info(f"Samples retained: {len(df_filtered)} ({len(df) - len(df_filtered)} removed)")

        return df_filtered.reset_index(drop=True)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for Neural ADMIXTURE."""
        features_list = []
        feature_names = []

        # 1. ABR thresholds (normalized 0-1) - highest weight
        abr_data = df[self.feature_columns['abr_thresholds']].values
        abr_min, abr_max = abr_data.min(), abr_data.max()
        if abr_max > abr_min:
            abr_normalized = (abr_data - abr_min) / (abr_max - abr_min)
        else:
            abr_normalized = np.zeros_like(abr_data)
        features_list.append(abr_normalized)
        feature_names.extend([f"norm_{col}" for col in self.feature_columns['abr_thresholds']])

        # 2. Click ABR (normalized 0-1) - medium weight
        click_data = df[self.feature_columns['click_abr']].values
        click_min, click_max = click_data.min(), click_data.max()
        if click_max > click_min:
            click_normalized = (click_data - click_min) / (click_max - click_min)
        else:
            click_normalized = np.zeros_like(click_data)
        features_list.append(click_normalized * 0.8)  # Slight downweighting
        feature_names.extend([f"norm_{col}" for col in self.feature_columns['click_abr']])

        # 3. Age calculation (days between dates)
        age_days = (df['date_of_experiment'] - df['date_of_birth']).dt.days.values
        age_min, age_max = age_days.min(), age_days.max()
        if age_max > age_min:
            age_normalized = (age_days - age_min) / (age_max - age_min)
        else:
            age_normalized = np.zeros_like(age_days, dtype=np.float32)
        features_list.append(age_normalized.reshape(-1, 1) * 0.5)  # Lower weight
        feature_names.append('age_days_norm')

        # 4. Weight (normalized) - lower weight, handle NaN values
        weight_data = df[self.feature_columns['continuous_meta']].values
        # Handle NaN values in weight
        weight_mean = np.nanmean(weight_data)
        weight_data = np.where(np.isnan(weight_data), weight_mean, weight_data)

        weight_min, weight_max = weight_data.min(), weight_data.max()
        if weight_max > weight_min:
            weight_normalized = (weight_data - weight_min) / (weight_max - weight_min)
        else:
            weight_normalized = np.zeros_like(weight_data)
        features_list.append(weight_normalized * 0.5)  # Lower weight
        feature_names.extend([f"norm_{col}" for col in self.feature_columns['continuous_meta']])

        # 5. One-hot encoded categoricals - lower weight
        for col in self.feature_columns['categorical']:
            # Handle any potential NaN values in categoricals
            col_data = df[col].fillna('unknown')
            encoded = pd.get_dummies(col_data, prefix=col, dummy_na=False).values.astype(np.float32)
            features_list.append(encoded * 0.4)  # Lowest weight
            feature_names.extend([f"{col}_{cat}" for cat in pd.get_dummies(col_data, prefix=col, dummy_na=False).columns])

        # Combine all features
        features_array = np.hstack(features_list)

        # Final check for any NaN values
        if np.any(np.isnan(features_array)):
            logger.warning("NaN values found in feature array, replacing with 0")
            features_array = np.where(np.isnan(features_array), 0.0, features_array)

        features_df = pd.DataFrame(features_array, columns=feature_names)

        logger.info(f"Feature engineering complete: {len(feature_names)} features created")
        logger.info(f"Feature array shape: {features_array.shape}")
        logger.info(f"Feature range: [{features_array.min():.3f}, {features_array.max():.3f}]")

        return features_df

    def _create_metadata(self, original_df: pd.DataFrame, features_df: pd.DataFrame) -> Dict:
        """Create metadata dictionary for downstream analysis."""
        metadata = {
            'gene_symbols': original_df['gene_symbol'].values,
            'feature_names': features_df.columns.tolist(),
            'n_samples': len(features_df),
            'n_features': len(features_df.columns),
            'feature_groups': {
                'abr_features': [i for i, name in enumerate(features_df.columns) if 'kHz-evoked' in name],
                'click_features': [i for i, name in enumerate(features_df.columns) if 'Click-evoked' in name],
                'age_features': [i for i, name in enumerate(features_df.columns) if 'age_days' in name],
                'weight_features': [i for i, name in enumerate(features_df.columns) if 'weight' in name],
                'categorical_features': [i for i, name in enumerate(features_df.columns) if any(cat in name for cat in self.feature_columns['categorical'])]
            },
            'unique_genes': original_df['gene_symbol'].nunique(),
            'samples_per_gene': original_df['gene_symbol'].value_counts().to_dict()
        }

        return metadata


def load_admixture_data(data_path: str, min_samples_per_gene: int = 3) -> Tuple[torch.Tensor, Dict]:
    """
    Convenience function to load data for Neural ADMIXTURE.

    Args:
        data_path: Path to IMPC data file
        min_samples_per_gene: Minimum samples per gene

    Returns:
        Tuple of (features_tensor, metadata_dict)
    """
    loader = AdmixtureDataLoader()
    return loader.load_data(data_path, min_samples_per_gene)


if __name__ == "__main__":
    # Test the loader
    import sys

    if len(sys.argv) < 2:
        print("Usage: python admixture_loader.py <data_path>")
        sys.exit(1)

    data_path = sys.argv[1]
    features, metadata = load_admixture_data(data_path)

    print(f"Loaded features shape: {features.shape}")
    print(f"Feature groups: {metadata['feature_groups']}")
    print(f"Unique genes: {metadata['unique_genes']}")