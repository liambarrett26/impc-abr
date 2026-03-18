"""
Preprocessing for Neural ADMIXTURE phenotypic data.
Handles standardization, normalization, and tensor preparation.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


class AdmixturePreprocessor:
    """
    Preprocessing pipeline for Neural ADMIXTURE phenotypic clustering.
    Handles mixed data types with appropriate scaling strategies.
    """

    def __init__(self,
                 scaling_method: str = 'standard',
                 final_range: Tuple[float, float] = (0.0, 1.0),
                 eps: float = 1e-8):
        """
        Initialize preprocessor.

        Args:
            scaling_method: 'standard', 'robust', or 'minmax'
            final_range: Final range for all features
            eps: Small value to prevent division by zero
        """
        self.scaling_method = scaling_method
        self.final_range = final_range
        self.eps = eps

        # Scalers for different feature types
        self.scalers = {}
        self.feature_stats = {}
        self.is_fitted = False

    def fit(self, features: torch.Tensor, metadata: Dict) -> 'AdmixturePreprocessor':
        """
        Fit preprocessing transformations.

        Args:
            features: Feature tensor (n_samples, n_features)
            metadata: Metadata dictionary with feature groups

        Returns:
            Self for method chaining
        """
        logger.info("Fitting preprocessing transformations")

        features_np = features.numpy() if isinstance(features, torch.Tensor) else features

        # Fit scalers for each feature group
        for group_name, feature_indices in metadata['feature_groups'].items():
            if not feature_indices:
                continue

            group_data = features_np[:, feature_indices]
            scaler = self._create_scaler(group_name)
            scaler.fit(group_data)

            self.scalers[group_name] = scaler
            self.feature_stats[group_name] = {
                'indices': feature_indices,
                'n_features': len(feature_indices),
                'mean': np.mean(group_data, axis=0),
                'std': np.std(group_data, axis=0),
                'min': np.min(group_data, axis=0),
                'max': np.max(group_data, axis=0)
            }

            logger.info(f"Fitted scaler for {group_name}: {len(feature_indices)} features")

        self.is_fitted = True
        return self

    def transform(self, features: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """
        Transform features using fitted preprocessing.

        Args:
            features: Feature tensor to transform
            metadata: Metadata dictionary

        Returns:
            Transformed feature tensor
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        features_np = features.numpy() if isinstance(features, torch.Tensor) else features
        transformed_features = features_np.copy()

        # Apply group-specific scaling
        for group_name, scaler in self.scalers.items():
            indices = self.feature_stats[group_name]['indices']
            if indices:
                group_data = features_np[:, indices]
                transformed_group = scaler.transform(group_data)
                transformed_features[:, indices] = transformed_group

        # Final range scaling to [final_range]
        transformed_features = self._apply_final_scaling(transformed_features)

        # Convert back to tensor
        return torch.tensor(transformed_features, dtype=torch.float32)

    def fit_transform(self, features: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """Fit and transform in one step."""
        return self.fit(features, metadata).transform(features, metadata)

    def _create_scaler(self, group_name: str):
        """Create appropriate scaler for feature group."""
        if self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'robust':
            return RobustScaler()
        elif self.scaling_method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")

    def _apply_final_scaling(self, features: np.ndarray) -> np.ndarray:
        """Apply final range scaling to all features."""
        # Global min-max scaling to final range
        feature_min = np.min(features, axis=0, keepdims=True)
        feature_max = np.max(features, axis=0, keepdims=True)

        # Avoid division by zero
        feature_range = feature_max - feature_min
        feature_range = np.where(feature_range < self.eps, 1.0, feature_range)

        # Scale to [0, 1]
        normalized = (features - feature_min) / feature_range

        # Scale to final range
        target_min, target_max = self.final_range
        scaled = normalized * (target_max - target_min) + target_min

        return scaled

    def get_feature_importance_weights(self, metadata: Dict) -> torch.Tensor:
        """
        Get feature importance weights based on feature groups.

        Args:
            metadata: Metadata dictionary

        Returns:
            Weight tensor for each feature
        """
        n_features = metadata['n_features']
        weights = torch.ones(n_features)

        # Weight scheme based on biological importance
        weight_map = {
            'abr_features': 1.0,      # Highest - core audiometric data
            'click_features': 0.8,    # High - complementary audiometric
            'age_features': 0.5,      # Medium - biological context
            'weight_features': 0.5,   # Medium - biological context
            'categorical_features': 0.4  # Lower - technical factors
        }

        for group_name, indices in metadata['feature_groups'].items():
            if indices and group_name in weight_map:
                weights[indices] = weight_map[group_name]

        return weights

    def inverse_transform(self, features: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """
        Inverse transform features back to original scale.

        Args:
            features: Transformed feature tensor
            metadata: Metadata dictionary

        Returns:
            Features in original scale
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse transform")

        features_np = features.numpy() if isinstance(features, torch.Tensor) else features

        # Inverse final scaling
        target_min, target_max = self.final_range
        normalized = (features_np - target_min) / (target_max - target_min)

        # Reconstruct using stored statistics
        inverse_features = features_np.copy()

        for group_name, scaler in self.scalers.items():
            indices = self.feature_stats[group_name]['indices']
            if indices:
                # This is approximate - exact inverse depends on scaling method
                group_data = normalized[:, indices]
                stats = self.feature_stats[group_name]

                # Approximate inverse using stored min/max
                restored = group_data * (stats['max'] - stats['min']) + stats['min']
                inverse_features[:, indices] = restored

        return torch.tensor(inverse_features, dtype=torch.float32)

    def summary(self) -> Dict:
        """Get preprocessing summary statistics."""
        if not self.is_fitted:
            return {"status": "not_fitted"}

        summary = {
            "status": "fitted",
            "scaling_method": self.scaling_method,
            "final_range": self.final_range,
            "feature_groups": {}
        }

        for group_name, stats in self.feature_stats.items():
            summary["feature_groups"][group_name] = {
                "n_features": stats['n_features'],
                "mean_range": [float(np.min(stats['mean'])), float(np.max(stats['mean']))],
                "std_range": [float(np.min(stats['std'])), float(np.max(stats['std']))]
            }

        return summary


def preprocess_for_admixture(features: torch.Tensor,
                           metadata: Dict,
                           scaling_method: str = 'standard') -> Tuple[torch.Tensor, AdmixturePreprocessor]:
    """
    Convenience function for preprocessing features for Neural ADMIXTURE.

    Args:
        features: Raw feature tensor
        metadata: Metadata dictionary
        scaling_method: Scaling method to use

    Returns:
        Tuple of (preprocessed_features, fitted_preprocessor)
    """
    preprocessor = AdmixturePreprocessor(scaling_method=scaling_method)
    preprocessed_features = preprocessor.fit_transform(features, metadata)

    logger.info(f"Preprocessing complete: {preprocessed_features.shape}")
    logger.info(f"Feature range: [{preprocessed_features.min():.3f}, {preprocessed_features.max():.3f}]")

    return preprocessed_features, preprocessor


if __name__ == "__main__":
    # Test preprocessing
    from admixture_loader import load_admixture_data
    import sys

    if len(sys.argv) < 2:
        print("Usage: python admixture_preproc.py <data_path>")
        sys.exit(1)

    # Load data
    data_path = sys.argv[1]
    features, metadata = load_admixture_data(data_path)

    # Test preprocessing
    processed_features, preprocessor = preprocess_for_admixture(features, metadata)

    print(f"Original shape: {features.shape}")
    print(f"Processed shape: {processed_features.shape}")
    print(f"Processed range: [{processed_features.min():.3f}, {processed_features.max():.3f}]")
    print(f"Preprocessing summary: {preprocessor.summary()}")