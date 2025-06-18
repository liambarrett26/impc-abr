"""
Enhanced feature engineering pipeline for Neural ADMIXTURE adaptation.

This module bridges the gap between ABR preprocessing and Neural ADMIXTURE input requirements,
handling categorical encoding, feature weighting, and data format conversion.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline."""
    abr_columns: List[str]
    click_column: str
    categorical_columns: List[str]
    continuous_columns: List[str]
    datetime_columns: List[str]

    # Feature weights
    abr_weight: float = 1.0
    click_weight: float = 0.7
    continuous_weight: float = 0.5
    categorical_weight: float = 0.5

    # Preprocessing parameters
    target_range: Tuple[float, float] = (0.0, 1.0)
    age_scaling: str = 'standard'  # 'standard', 'minmax', or 'none'


class NeuralAdmixtureFeatureEngineer:
    """
    Feature engineering pipeline specifically designed for Neural ADMIXTURE input.

    Bridges the gap between ABR preprocessing and neural network requirements.
    """

    def __init__(self, config: FeatureConfig):
        """
        Initialize feature engineer with configuration.

        Args:
            config: FeatureConfig with column specifications and weights
        """
        self.config = config
        self.categorical_encoders: Dict[str, LabelEncoder] = {}
        self.continuous_scalers: Dict[str, StandardScaler] = {}
        self.feature_names: List[str] = []
        self.is_fitted = False

    def _calculate_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate age-related features from datetime columns.

        Args:
            df: Input dataframe with datetime columns

        Returns:
            DataFrame with calculated age features
        """
        age_features = pd.DataFrame(index=df.index)

        # Convert datetime columns if they're not already datetime
        date_birth = pd.to_datetime(df['date_of_birth'])
        date_experiment = pd.to_datetime(df['date_of_experiment'])

        # Calculate age at experiment in days
        age_at_experiment = (date_experiment - date_birth).dt.days
        age_features['age_at_experiment_days'] = age_at_experiment

        # Calculate age in weeks (alternative measure)
        age_features['age_at_experiment_weeks'] = age_at_experiment / 7.0

        # Extract temporal features that might be relevant
        age_features['experiment_day_of_year'] = date_experiment.dt.dayofyear
        age_features['experiment_month'] = date_experiment.dt.month

        return age_features

    def _encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        One-hot encode categorical features.

        Args:
            df: Input dataframe
            fit: Whether to fit encoders (True for training, False for inference)

        Returns:
            DataFrame with one-hot encoded categorical features
        """
        categorical_features = pd.DataFrame(index=df.index)

        for col in self.config.categorical_columns:
            if col not in df.columns:
                logger.warning(f"Categorical column {col} not found in dataframe")
                continue

            if fit:
                # Create dummy variables and store the column names
                dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                self.categorical_encoders[col] = list(dummies.columns)
                categorical_features = pd.concat([categorical_features, dummies], axis=1)
            else:
                # Use stored column names to ensure consistent encoding
                if col in self.categorical_encoders:
                    dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                    # Ensure all expected columns are present
                    for expected_col in self.categorical_encoders[col]:
                        if expected_col not in dummies.columns:
                            dummies[expected_col] = 0
                    # Keep only expected columns in correct order
                    dummies = dummies[self.categorical_encoders[col]]
                    categorical_features = pd.concat([categorical_features, dummies], axis=1)
                else:
                    logger.warning(f"No encoder found for categorical column {col}")

        return categorical_features

    def _scale_continuous_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale continuous features (excluding ABR data which is pre-normalized).

        Args:
            df: Input dataframe
            fit: Whether to fit scalers

        Returns:
            DataFrame with scaled continuous features
        """
        continuous_features = pd.DataFrame(index=df.index)

        for col in self.config.continuous_columns:
            if col not in df.columns:
                logger.warning(f"Continuous column {col} not found in dataframe")
                continue

            values = df[col].values.reshape(-1, 1)

            if fit:
                if self.config.age_scaling == 'standard':
                    scaler = StandardScaler()
                elif self.config.age_scaling == 'minmax':
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler(feature_range=self.config.target_range)
                else:
                    scaler = None

                if scaler is not None:
                    scaler.fit(values)
                    self.continuous_scalers[col] = scaler
                    continuous_features[col] = scaler.transform(values).flatten()
                else:
                    continuous_features[col] = values.flatten()
            else:
                if col in self.continuous_scalers:
                    continuous_features[col] = self.continuous_scalers[col].transform(values).flatten()
                else:
                    continuous_features[col] = values.flatten()

        return continuous_features

    def _apply_feature_weights(self, feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply feature weights and combine all features.

        Args:
            feature_dict: Dictionary of feature arrays by category

        Returns:
            Combined weighted feature array
        """
        weighted_features = []
        feature_names = []

        # ABR features (highest weight)
        if 'abr' in feature_dict:
            weighted_abr = feature_dict['abr'] * self.config.abr_weight
            weighted_features.append(weighted_abr)
            feature_names.extend([f"abr_{i}" for i in range(weighted_abr.shape[1])])

        # Click ABR feature (medium weight)
        if 'click' in feature_dict:
            weighted_click = feature_dict['click'] * self.config.click_weight
            weighted_features.append(weighted_click)
            feature_names.extend(['click_abr'])

        # Continuous features (lower weight)
        if 'continuous' in feature_dict:
            weighted_continuous = feature_dict['continuous'] * self.config.continuous_weight
            weighted_features.append(weighted_continuous)
            feature_names.extend([f"continuous_{i}" for i in range(weighted_continuous.shape[1])])

        # Categorical features (lower weight)
        if 'categorical' in feature_dict:
            weighted_categorical = feature_dict['categorical'] * self.config.categorical_weight
            weighted_features.append(weighted_categorical)
            feature_names.extend([f"categorical_{i}" for i in range(weighted_categorical.shape[1])])

        if not weighted_features:
            raise ValueError("No features were processed")

        combined_features = np.hstack(weighted_features)

        if not self.is_fitted:
            self.feature_names = feature_names

        return combined_features

    def fit_transform(self, df: pd.DataFrame, abr_normalized: np.ndarray) -> np.ndarray:
        """
        Fit the feature engineering pipeline and transform data.

        Args:
            df: Input dataframe with all columns
            abr_normalized: Pre-normalized ABR data from ABRPreprocessor

        Returns:
            Transformed feature array ready for Neural ADMIXTURE
        """
        logger.info("Fitting feature engineering pipeline")

        feature_dict = {}

        # ABR features (already normalized by ABRPreprocessor)
        feature_dict['abr'] = abr_normalized

        # Click ABR feature
        if self.config.click_column in df.columns:
            click_values = df[self.config.click_column].values.reshape(-1, 1)
            # Normalize click ABR to same range as other ABR data
            click_normalized = (click_values - click_values.min()) / (click_values.max() - click_values.min())
            feature_dict['click'] = click_normalized
        else:
            logger.warning(f"Click column {self.config.click_column} not found")

        # Calculate age features
        age_features = self._calculate_age_features(df)

        # Combine age features with other continuous features
        continuous_cols = self.config.continuous_columns + list(age_features.columns)
        extended_df = pd.concat([df, age_features], axis=1)

        # Scale continuous features
        continuous_features = self._scale_continuous_features(extended_df[continuous_cols], fit=True)
        if len(continuous_features.columns) > 0:
            feature_dict['continuous'] = continuous_features.values

        # Encode categorical features
        categorical_features = self._encode_categorical_features(df, fit=True)
        if len(categorical_features.columns) > 0:
            feature_dict['categorical'] = categorical_features.values

        # Apply weights and combine
        combined_features = self._apply_feature_weights(feature_dict)

        self.is_fitted = True

        logger.info(f"Feature engineering complete: {combined_features.shape[1]} features from {combined_features.shape[0]} samples")
        logger.info(f"Feature breakdown: ABR({abr_normalized.shape[1]}), " +
                   f"Click({1 if 'click' in feature_dict else 0}), " +
                   f"Continuous({continuous_features.shape[1] if 'continuous' in feature_dict else 0}), " +
                   f"Categorical({categorical_features.shape[1] if 'categorical' in feature_dict else 0})")

        return combined_features

    def transform(self, df: pd.DataFrame, abr_normalized: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted pipeline.

        Args:
            df: Input dataframe
            abr_normalized: Pre-normalized ABR data

        Returns:
            Transformed feature array
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transform")

        feature_dict = {}

        # ABR features
        feature_dict['abr'] = abr_normalized

        # Click ABR feature
        if self.config.click_column in df.columns:
            click_values = df[self.config.click_column].values.reshape(-1, 1)
            # Use same normalization as training
            click_normalized = (click_values - click_values.min()) / (click_values.max() - click_values.min())
            feature_dict['click'] = click_normalized

        # Age features
        age_features = self._calculate_age_features(df)
        continuous_cols = self.config.continuous_columns + list(age_features.columns)
        extended_df = pd.concat([df, age_features], axis=1)

        # Scale continuous features
        continuous_features = self._scale_continuous_features(extended_df[continuous_cols], fit=False)
        if len(continuous_features.columns) > 0:
            feature_dict['continuous'] = continuous_features.values

        # Encode categorical features
        categorical_features = self._encode_categorical_features(df, fit=False)
        if len(categorical_features.columns) > 0:
            feature_dict['categorical'] = categorical_features.values

        # Apply weights and combine
        combined_features = self._apply_feature_weights(feature_dict)

        return combined_features

    def get_feature_names(self) -> List[str]:
        """Get names of all engineered features."""
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted first")
        return self.feature_names


def create_default_feature_config() -> FeatureConfig:
    """Create default feature configuration for IMPC ABR data."""
    return FeatureConfig(
        abr_columns=[
            '6kHz-evoked ABR Threshold',
            '12kHz-evoked ABR Threshold',
            '18kHz-evoked ABR Threshold',
            '24kHz-evoked ABR Threshold',
            '30kHz-evoked ABR Threshold'
        ],
        click_column='Click-evoked ABR threshold',
        categorical_columns=[
            'sex',
            'zygosity',
            'genetic_background',
            'phenotyping_center'
        ],
        continuous_columns=[
            'weight',
            'age_in_days',
            'age_in_weeks'
        ],
        datetime_columns=[
            'date_of_birth',
            'date_of_experiment'
        ]
    )
