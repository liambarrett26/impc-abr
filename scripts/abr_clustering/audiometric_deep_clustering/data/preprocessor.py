"""
Feature preprocessing pipeline for ContrastiveVAE-DEC model.

This module provides preprocessing tailored for deep learning on audiometric data,
including normalization, PCA feature extraction, and categorical encoding.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class ABRFeaturePreprocessor:
    """
    Comprehensive feature preprocessor for ABR audiometric data.

    Handles multiple feature types and creates the full 18-dimensional feature vector
    required by the ContrastiveVAE-DEC model.
    """

    def __init__(self, config: Dict):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Preprocessing configuration dictionary
        """
        self.config = config
        self.abr_columns = config.get('abr_columns', [
            '6kHz-evoked ABR Threshold',
            '12kHz-evoked ABR Threshold',
            '18kHz-evoked ABR Threshold',
            '24kHz-evoked ABR Threshold',
            '30kHz-evoked ABR Threshold',
            'Click-evoked ABR threshold'
        ])

        self.metadata_columns = config.get('metadata_columns', [
            'age_in_weeks', 'weight', 'sex', 'zygosity',
            'genetic_background', 'phenotyping_center',
            'pipeline_name', 'metadata_Equipment manufacturer'
        ])

        # Initialize components
        self.abr_scaler = StandardScaler()
        self.metadata_scaler = StandardScaler()
        self.pca = PCA(n_components=2, random_state=42)
        self.label_encoders = {}
        self.column_transformer = None
        self.is_fitted = False

    def _prepare_abr_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare and normalize ABR threshold features.

        Args:
            df: Input dataframe

        Returns:
            Normalized ABR features
        """
        # Extract available ABR columns
        available_abr = [col for col in self.abr_columns if col in df.columns]

        if len(available_abr) == 0:
            raise ValueError("No ABR threshold columns found in data")

        # Handle missing columns by filling with median
        abr_data = df[available_abr].copy()

        # Fill missing values with column medians
        abr_data = abr_data.fillna(abr_data.median())

        # Log any data quality issues
        if abr_data.isnull().any().any():
            logger.warning("Found NaN values in ABR data after median imputation")

        return abr_data.values

    def _prepare_metadata_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare metadata features (continuous and categorical).

        Args:
            df: Input dataframe

        Returns:
            Tuple of (continuous_features, categorical_features)
        """
        # Continuous features
        continuous_cols = ['age_in_weeks', 'weight']
        available_continuous = [col for col in continuous_cols if col in df.columns]

        if available_continuous:
            continuous_data = df[available_continuous].copy()
            continuous_data = continuous_data.fillna(continuous_data.median())
            continuous_features = continuous_data.values
        else:
            continuous_features = np.empty((len(df), 0))

        # Categorical features
        categorical_cols = ['sex', 'zygosity', 'genetic_background',
                          'phenotyping_center', 'pipeline_name',
                          'metadata_Equipment manufacturer']
        available_categorical = [col for col in categorical_cols if col in df.columns]

        if available_categorical:
            categorical_data = df[available_categorical].copy()
            # Fill missing values with 'unknown'
            categorical_data = categorical_data.fillna('unknown')
            categorical_features = categorical_data.values
        else:
            categorical_features = np.empty((len(df), 0))

        return continuous_features, categorical_features

    def _create_column_transformer(self, continuous_features: np.ndarray,
                                 categorical_features: np.ndarray) -> ColumnTransformer:
        """
        Create sklearn ColumnTransformer for mixed data types.

        Args:
            continuous_features: Continuous feature matrix
            categorical_features: Categorical feature matrix

        Returns:
            Configured ColumnTransformer
        """
        transformers = []

        # Continuous features - standardize
        if continuous_features.shape[1] > 0:
            continuous_indices = list(range(continuous_features.shape[1]))
            transformers.append(
                ('continuous', StandardScaler(), continuous_indices)
            )

        # Categorical features - one-hot encode with limited categories
        if categorical_features.shape[1] > 0:
            categorical_indices = list(range(
                continuous_features.shape[1],
                continuous_features.shape[1] + categorical_features.shape[1]
            ))

            # Use OneHotEncoder with handling for unknown categories
            transformers.append(
                ('categorical',
                 OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=10),
                 categorical_indices)
            )

        return ColumnTransformer(transformers, remainder='passthrough')

    def fit(self, df: pd.DataFrame) -> 'ABRFeaturePreprocessor':
        """
        Fit preprocessing pipeline on training data.

        Args:
            df: Training dataframe

        Returns:
            Self for method chaining
        """
        logger.info("Fitting ABR feature preprocessor")

        # Prepare ABR features
        abr_features = self._prepare_abr_features(df)

        # Fit ABR scaler
        self.abr_scaler.fit(abr_features)

        # Prepare metadata features
        continuous_features, categorical_features = self._prepare_metadata_features(df)

        # Combine for column transformer
        if continuous_features.shape[1] > 0 or categorical_features.shape[1] > 0:
            combined_metadata = np.hstack([
                continuous_features,
                categorical_features
            ]) if continuous_features.shape[1] > 0 and categorical_features.shape[1] > 0 else (
                continuous_features if continuous_features.shape[1] > 0 else categorical_features
            )

            # Create and fit column transformer
            self.column_transformer = self._create_column_transformer(
                continuous_features, categorical_features
            )
            self.column_transformer.fit(combined_metadata)

        # Fit PCA on normalized ABR features
        normalized_abr = self.abr_scaler.transform(abr_features)
        self.pca.fit(normalized_abr)

        self.is_fitted = True
        self._log_fit_summary(abr_features, continuous_features, categorical_features)

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessing pipeline.

        Args:
            df: Dataframe to transform

        Returns:
            Preprocessed feature matrix (n_samples, 18)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Process ABR features
        abr_features = self._prepare_abr_features(df)
        normalized_abr = self.abr_scaler.transform(abr_features)

        # Process metadata features
        continuous_features, categorical_features = self._prepare_metadata_features(df)

        if self.column_transformer is not None:
            if continuous_features.shape[1] > 0 or categorical_features.shape[1] > 0:
                combined_metadata = np.hstack([
                    continuous_features,
                    categorical_features
                ]) if continuous_features.shape[1] > 0 and categorical_features.shape[1] > 0 else (
                    continuous_features if continuous_features.shape[1] > 0 else categorical_features
                )
                processed_metadata = self.column_transformer.transform(combined_metadata)
            else:
                processed_metadata = np.empty((len(df), 0))
        else:
            processed_metadata = np.empty((len(df), 0))

        # Generate PCA features from ABR data
        pca_features = self.pca.transform(normalized_abr)

        # Combine all features: ABR (6) + metadata (10) + PCA (2) = 18 total
        # Pad or truncate metadata to ensure exactly 10 dimensions
        if processed_metadata.shape[1] > 10:
            processed_metadata = processed_metadata[:, :10]
        elif processed_metadata.shape[1] < 10:
            padding = np.zeros((processed_metadata.shape[0], 10 - processed_metadata.shape[1]))
            processed_metadata = np.hstack([processed_metadata, padding])

        # Combine all features
        final_features = np.hstack([
            normalized_abr,           # 6 ABR features
            processed_metadata,       # 10 metadata features
            pca_features             # 2 PCA features
        ])

        assert final_features.shape[1] == 18, f"Expected 18 features, got {final_features.shape[1]}"

        return final_features

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            df: Input dataframe

        Returns:
            Preprocessed features
        """
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        """
        Get names of output features.

        Returns:
            List of feature names
        """
        feature_names = []

        # ABR feature names
        available_abr = [col for col in self.abr_columns if col in self.abr_columns]
        feature_names.extend([f"abr_{col.split('-')[0]}" for col in available_abr])

        # Metadata feature names (padded to 10)
        metadata_names = ['metadata_' + str(i) for i in range(10)]
        feature_names.extend(metadata_names)

        # PCA feature names
        feature_names.extend(['pca_1', 'pca_2'])

        return feature_names

    def save(self, path: str):
        """Save fitted preprocessor to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")

        save_dict = {
            'config': self.config,
            'abr_scaler': self.abr_scaler,
            'column_transformer': self.column_transformer,
            'pca': self.pca,
            'is_fitted': self.is_fitted
        }

        joblib.dump(save_dict, path)
        logger.info(f"Saved preprocessor to {path}")

    @classmethod
    def load(cls, path: str) -> 'ABRFeaturePreprocessor':
        """Load fitted preprocessor from disk."""
        save_dict = joblib.load(path)

        preprocessor = cls(save_dict['config'])
        preprocessor.abr_scaler = save_dict['abr_scaler']
        preprocessor.column_transformer = save_dict['column_transformer']
        preprocessor.pca = save_dict['pca']
        preprocessor.is_fitted = save_dict['is_fitted']

        logger.info(f"Loaded preprocessor from {path}")
        return preprocessor

    def _log_fit_summary(self, abr_features: np.ndarray,
                        continuous_features: np.ndarray,
                        categorical_features: np.ndarray):
        """Log summary of preprocessing fit."""
        logger.info("=== Preprocessing Fit Summary ===")
        logger.info(f"ABR features: {abr_features.shape}")
        logger.info(f"Continuous metadata: {continuous_features.shape}")
        logger.info(f"Categorical metadata: {categorical_features.shape}")

        if self.column_transformer:
            try:
                transformed_meta = self.column_transformer.transform(
                    np.hstack([continuous_features, categorical_features])[:1]
                )
                logger.info(f"Transformed metadata shape: {transformed_meta.shape}")
            except:
                logger.info("Could not determine transformed metadata shape")

        logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_}")
        logger.info("=== End Preprocessing Summary ===")


def create_default_config() -> Dict:
    """
    Create default preprocessing configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        'abr_columns': [
            '6kHz-evoked ABR Threshold',
            '12kHz-evoked ABR Threshold',
            '18kHz-evoked ABR Threshold',
            '24kHz-evoked ABR Threshold',
            '30kHz-evoked ABR Threshold',
            'Click-evoked ABR threshold'
        ],
        'metadata_columns': [
            'age_in_weeks', 'weight', 'sex', 'zygosity',
            'genetic_background', 'phenotyping_center',
            'pipeline_name', 'metadata_Equipment manufacturer'
        ],
        'target_dim': 18
    }


def create_preprocessor(config: Optional[Dict] = None) -> ABRFeaturePreprocessor:
    """
    Factory function to create preprocessor with default configuration.

    Args:
        config: Optional configuration override

    Returns:
        Configured preprocessor
    """
    if config is None:
        config = create_default_config()

    return ABRFeaturePreprocessor(config)