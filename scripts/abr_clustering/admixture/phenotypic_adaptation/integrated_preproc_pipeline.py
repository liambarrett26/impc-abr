"""
Integrated preprocessing pipeline combining ABR normalization with Neural ADMIXTURE preparation.

This module provides a complete pipeline from raw IMPC data to Neural ADMIXTURE-ready tensors.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import logging

# Import from your existing modules
from preproc import ABRPreprocessor, create_default_config as create_abr_config
from loader import IMPCABRLoader

# Import new modules
from feature_engineering import NeuralAdmixtureFeatureEngineer, create_default_feature_config
from phenotypic_data_loader import load_clean_impc_data_for_neural_admixture, EnhancedPhenotypicDataProcessor

logger = logging.getLogger(__name__)


class IntegratedPhenotypicPipeline:
    """
    Complete pipeline integrating ABR preprocessing with Neural ADMIXTURE preparation.

    This class orchestrates the entire data flow from raw IMPC data to Neural ADMIXTURE input.
    """

    def __init__(self,
                 abr_config=None,
                 feature_config=None,
                 device: torch.device = None,
                 min_mutants: int = 3,
                 min_controls: int = 20):
        """
        Initialize integrated pipeline.

        Args:
            abr_config: Configuration for ABR preprocessing
            feature_config: Configuration for feature engineering
            device: Target device for tensors
            min_mutants: Minimum mutants per experimental group
            min_controls: Minimum controls per experimental group
        """
        self.abr_config = abr_config or create_abr_config()
        self.feature_config = feature_config or create_default_feature_config()
        self.device = device or torch.device('cpu')
        self.min_mutants = min_mutants
        self.min_controls = min_controls

        # Initialize components
        self.abr_preprocessor = ABRPreprocessor(self.abr_config)
        self.feature_engineer = NeuralAdmixtureFeatureEngineer(self.feature_config)
        self.data_processor = PhenotypicDataProcessor(self.device)

        # State tracking
        self.is_fitted = False
        self.feature_names = []
        self.data_stats = {}

    def load_and_preprocess(self, data_path: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load raw IMPC data using existing loader.

        Args:
            data_path: Path to IMPC data file

        Returns:
            Tuple of (complete_dataset, experimental_groups)
        """
        logger.info("Loading IMPC data...")
        loader = IMPCABRLoader(data_path)
        df, experimental_groups = loader.load_and_prepare(
            min_mutants=self.min_mutants,
            min_controls=self.min_controls
        )

        # Validate required columns are present
        self._validate_required_columns(df)

        return df, experimental_groups

    def _validate_required_columns(self, df: pd.DataFrame):
        """Validate that all required columns are present in the dataset."""
        required_columns = (
            self.feature_config.abr_columns +
            [self.feature_config.click_column] +
            self.feature_config.categorical_columns +
            self.feature_config.continuous_columns +
            self.feature_config.datetime_columns
        )

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")

            # Check for alternative column names
            alternative_names = {
                'Click-evoked ABR threshold': ['click_abr', 'click_threshold', 'Click ABR'],
                'date_of_birth': ['birth_date', 'dob'],
                'date_of_experiment': ['experiment_date', 'test_date']
            }

            for missing_col in missing_columns:
                if missing_col in alternative_names:
                    alternatives = [alt for alt in alternative_names[missing_col] if alt in df.columns]
                    if alternatives:
                        logger.info(f"Using {alternatives[0]} for {missing_col}")
                        df[missing_col] = df[alternatives[0]]
                        missing_columns.remove(missing_col)

            if missing_columns:
                raise ValueError(f"Critical columns missing: {missing_columns}")

    def fit_preprocessing(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit ABR preprocessing and feature engineering pipelines.

        Args:
            df: Input dataframe

        Returns:
            Tuple of (normalized_abr_data, engineered_features)
        """
        logger.info("Fitting preprocessing pipelines...")

        # Step 1: ABR preprocessing (your existing pipeline)
        logger.info("Step 1: ABR normalization...")
        abr_normalized = self.abr_preprocessor.fit_transform(df)

        # Step 2: Feature engineering
        logger.info("Step 2: Feature engineering...")
        engineered_features = self.feature_engineer.fit_transform(df, abr_normalized)

        # Store feature names and statistics
        self.feature_names = self.feature_engineer.get_feature_names()
        self._calculate_data_stats(engineered_features)

        self.is_fitted = True
        logger.info("Preprocessing pipelines fitted successfully")

        return abr_normalized, engineered_features

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform new data using fitted pipelines.

        Args:
            df: Input dataframe

        Returns:
            Tuple of (normalized_abr_data, engineered_features)
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")

        logger.info("Transforming data...")

        # ABR preprocessing
        abr_normalized = self.abr_preprocessor.transform(df)

        # Feature engineering
        engineered_features = self.feature_engineer.transform(df, abr_normalized)

        return abr_normalized, engineered_features

    def prepare_for_neural_admixture(self,
                                   df: pd.DataFrame,
                                   engineered_features: np.ndarray) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Prepare data for Neural ADMIXTURE input.

        Args:
            df: Original dataframe with metadata
            engineered_features: Output from feature engineering

        Returns:
            Tuple of (features, gene_labels, metadata) for Neural ADMIXTURE
        """
        logger.info("Preparing data for Neural ADMIXTURE...")

        # Extract gene information
        gene_symbols = df['gene_symbol'].values if 'gene_symbol' in df.columns else None
        center_info = df['phenotyping_center'].values if 'phenotyping_center' in df.columns else None

        # Convert to tensors
        features, gene_labels, metadata = self.data_processor.prepare_data_for_neural_admixture(
            engineered_features, gene_symbols, center_info
        )

        # Add pipeline metadata
        metadata.update({
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'data_stats': self.data_stats,
            'abr_config': self.abr_config.__dict__,
            'feature_config': self.feature_config.__dict__
        })

        return features, gene_labels, metadata

    def fit_transform_complete(self, data_path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any], pd.DataFrame]:
        """
        Complete pipeline: load, fit, and prepare data for Neural ADMIXTURE.

        Args:
            data_path: Path to IMPC data

        Returns:
            Tuple of (features, gene_labels, metadata, original_dataframe)
        """
        # Load data
        df, experimental_groups = self.load_and_preprocess(data_path)

        # Fit and transform
        abr_normalized, engineered_features = self.fit_preprocessing(df)

        # Prepare for Neural ADMIXTURE
        features, gene_labels, metadata = self.prepare_for_neural_admixture(df, engineered_features)

        # Add experimental group information
        metadata['experimental_groups'] = experimental_groups
        metadata['total_groups'] = len(experimental_groups)

        logger.info(f"Pipeline complete: {features.shape[0]} samples, {features.shape[1]} features")

        return features, gene_labels, metadata, df

    def _calculate_data_stats(self, features: np.ndarray):
        """Calculate and store data statistics."""
        self.data_stats = {
            'n_samples': features.shape[0],
            'n_features': features.shape[1],
            'feature_means': np.mean(features, axis=0).tolist(),
            'feature_stds': np.std(features, axis=0).tolist(),
            'feature_mins': np.min(features, axis=0).tolist(),
            'feature_maxs': np.max(features, axis=0).tolist(),
            'missing_rate': np.sum(np.isnan(features)) / features.size,
            'constant_features': np.sum(np.std(features, axis=0) < 1e-8)
        }

    def save_preprocessing_state(self, save_dir: str, name: str = "phenotypic_pipeline"):
        """
        Save preprocessing state for later use.

        Args:
            save_dir: Directory to save state
            name: Name prefix for saved files
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save ABR preprocessor
        import pickle
        with open(save_path / f"{name}_abr_preprocessor.pkl", 'wb') as f:
            pickle.dump(self.abr_preprocessor, f)

        # Save feature engineer
        with open(save_path / f"{name}_feature_engineer.pkl", 'wb') as f:
            pickle.dump(self.feature_engineer, f)

        # Save pipeline state
        state = {
            'feature_names': self.feature_names,
            'data_stats': self.data_stats,
            'abr_config': self.abr_config.__dict__,
            'feature_config': self.feature_config.__dict__,
            'device': str(self.device)
        }

        import json
        with open(save_path / f"{name}_pipeline_state.json", 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Pipeline state saved to {save_path}")

    def load_preprocessing_state(self, save_dir: str, name: str = "phenotypic_pipeline"):
        """
        Load preprocessing state from saved files.

        Args:
            save_dir: Directory containing saved state
            name: Name prefix for saved files
        """
        save_path = Path(save_dir)

        # Load ABR preprocessor
        import pickle
        with open(save_path / f"{name}_abr_preprocessor.pkl", 'rb') as f:
            self.abr_preprocessor = pickle.load(f)

        # Load feature engineer
        with open(save_path / f"{name}_feature_engineer.pkl", 'rb') as f:
            self.feature_engineer = pickle.load(f)

        # Load pipeline state
        import json
        with open(save_path / f"{name}_pipeline_state.json", 'r') as f:
            state = json.load(f)

        self.feature_names = state['feature_names']
        self.data_stats = state['data_stats']
        self.is_fitted = True

        logger.info(f"Pipeline state loaded from {save_path}")


def create_neural_admixture_dataset(data_path: str,
                                  device: torch.device = None,
                                  save_preprocessing: bool = True,
                                  save_dir: str = "preprocessing_outputs") -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """
    Convenience function to create Neural ADMIXTURE dataset from IMPC data.

    Args:
        data_path: Path to IMPC data
        device: Target device
        save_preprocessing: Whether to save preprocessing state
        save_dir: Directory to save preprocessing outputs

    Returns:
        Tuple of (features, gene_labels, metadata) ready for Neural ADMIXTURE
    """
    # Initialize pipeline
    pipeline = IntegratedPhenotypicPipeline(device=device)

    # Run complete pipeline
    features, gene_labels, metadata, df = pipeline.fit_transform_complete(data_path)

    # Save preprocessing state if requested
    if save_preprocessing:
        pipeline.save_preprocessing_state(save_dir)

    # Log summary
    logger.info("=" * 60)
    logger.info("NEURAL ADMIXTURE DATASET PREPARATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Samples: {features.shape[0]}")
    logger.info(f"Features: {features.shape[1]}")
    logger.info(f"Feature breakdown:")

    feature_config = pipeline.feature_config
    logger.info(f"  - ABR thresholds: {len(feature_config.abr_columns)} (weight: {feature_config.abr_weight})")
    logger.info(f"  - Click ABR: 1 (weight: {feature_config.click_weight})")
    logger.info(f"  - Continuous: ~{len(feature_config.continuous_columns)+4} (weight: {feature_config.continuous_weight})")  # +4 for age features
    logger.info(f"  - Categorical: ~{len(feature_config.categorical_columns)*3} (weight: {feature_config.categorical_weight})")  # Estimated one-hot size

    if gene_labels is not None:
        unique_genes = torch.unique(gene_labels)
        logger.info(f"Unique genes: {len(unique_genes)}")

    logger.info(f"Device: {features.device}")
    logger.info("=" * 60)

    return features, gene_labels, metadata


# Validation function
def validate_neural_admixture_input(features: torch.Tensor,
                                  gene_labels: Optional[torch.Tensor] = None,
                                  metadata: Dict[str, Any] = None) -> bool:
    """
    Validate that data is properly formatted for Neural ADMIXTURE.

    Args:
        features: Feature tensor
        gene_labels: Optional gene label tensor
        metadata: Metadata dictionary

    Returns:
        True if validation passes
    """
    logger.info("Validating Neural ADMIXTURE input format...")

    # Check feature tensor
    if not isinstance(features, torch.Tensor):
        raise TypeError("Features must be torch.Tensor")

    if len(features.shape) != 2:
        raise ValueError(f"Features must be 2D tensor, got shape {features.shape}")

    if features.dtype != torch.float32:
        logger.warning(f"Features dtype is {features.dtype}, expected torch.float32")

    # Check value ranges (should be roughly [0, 1] after preprocessing)
    min_val, max_val = features.min().item(), features.max().item()
    if min_val < -0.1 or max_val > 1.1:
        logger.warning(f"Feature values outside expected range [0,1]: [{min_val:.3f}, {max_val:.3f}]")

    # Check for NaN/inf
    if torch.any(torch.isnan(features)):
        logger.error("Features contain NaN values")
        return False

    if torch.any(torch.isinf(features)):
        logger.error("Features contain infinite values")
        return False

    # Check gene labels if provided
    if gene_labels is not None:
        if not isinstance(gene_labels, torch.Tensor):
            raise TypeError("Gene labels must be torch.Tensor")

        if gene_labels.size(0) != features.size(0):
            raise ValueError(f"Gene labels size mismatch: {gene_labels.size(0)} vs {features.size(0)}")

        if gene_labels.dtype != torch.long:
            logger.warning(f"Gene labels dtype is {gene_labels.dtype}, expected torch.long")

    # Check metadata
    if metadata is not None:
        required_keys = ['feature_names', 'n_features']
        missing_keys = [key for key in required_keys if key not in metadata]
        if missing_keys:
            logger.warning(f"Missing metadata keys: {missing_keys}")

    logger.info("Input validation passed")
    return True