"""
Data augmentations for contrastive learning on audiometric data.

This module provides biologically-informed augmentations for ABR threshold data
that preserve meaningful hearing patterns while creating diverse positive pairs
for contrastive learning.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseAugmentation(ABC):
    """Base class for audiometric data augmentations."""

    def __init__(self, probability: float = 0.5):
        """
        Initialize augmentation.

        Args:
            probability: Probability of applying this augmentation
        """
        self.probability = probability

    @abstractmethod
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to features."""
        pass


class GaussianNoise(BaseAugmentation):
    """
    Add Gaussian noise to features.

    Simulates measurement noise and biological variability in ABR recordings.
    """

    def __init__(self, noise_std: float = 0.05, probability: float = 0.8):
        """
        Initialize Gaussian noise augmentation.

        Args:
            noise_std: Standard deviation of noise relative to feature scale
            probability: Probability of applying noise
        """
        super().__init__(probability)
        self.noise_std = noise_std

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise to features."""
        if torch.rand(1).item() > self.probability:
            return features

        # Add noise scaled by feature magnitude
        noise = torch.randn_like(features) * self.noise_std
        return features + noise


class ABRFrequencySmoothing(BaseAugmentation):
    """
    Apply smoothing across ABR frequencies.

    Simulates slight variations in frequency response patterns while
    preserving overall audiogram shape.
    """

    def __init__(self, smoothing_strength: float = 0.1, probability: float = 0.6):
        """
        Initialize frequency smoothing.

        Args:
            smoothing_strength: Strength of smoothing operation
            probability: Probability of applying smoothing
        """
        super().__init__(probability)
        self.smoothing_strength = smoothing_strength

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Apply frequency smoothing to ABR features (first 6 dimensions)."""
        if torch.rand(1).item() > self.probability:
            return features

        augmented = features.clone()

        # Apply smoothing only to ABR features (indices 0-5)
        abr_features = features[:6]

        # Simple moving average smoothing
        if len(abr_features) >= 3:
            smoothed = abr_features.clone()
            for i in range(1, len(abr_features) - 1):
                smoothed[i] = (abr_features[i-1] + abr_features[i] + abr_features[i+1]) / 3.0

            # Blend with original
            augmented[:6] = (1 - self.smoothing_strength) * abr_features + \
                           self.smoothing_strength * smoothed

        return augmented


class FeatureDropout(BaseAugmentation):
    """
    Randomly set some features to zero.

    Simulates missing measurements or technical failures in specific channels.
    """

    def __init__(self, dropout_prob: float = 0.1, probability: float = 0.4):
        """
        Initialize feature dropout.

        Args:
            dropout_prob: Probability of dropping each feature
            probability: Probability of applying dropout
        """
        super().__init__(probability)
        self.dropout_prob = dropout_prob

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Apply feature dropout."""
        if torch.rand(1).item() > self.probability:
            return features

        # Create dropout mask
        mask = torch.rand_like(features) > self.dropout_prob
        return features * mask.float()


class ABRShift(BaseAugmentation):
    """
    Apply small systematic shifts to ABR thresholds.

    Simulates calibration differences or slight hearing threshold variations
    that preserve relative frequency patterns.
    """

    def __init__(self, max_shift: float = 0.1, probability: float = 0.7):
        """
        Initialize ABR threshold shifting.

        Args:
            max_shift: Maximum shift magnitude (relative to feature scale)
            probability: Probability of applying shift
        """
        super().__init__(probability)
        self.max_shift = max_shift

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Apply threshold shift to ABR features."""
        if torch.rand(1).item() > self.probability:
            return features

        augmented = features.clone()

        # Apply shift only to ABR features (indices 0-5)
        shift = torch.zeros(6).uniform_(-self.max_shift, self.max_shift)
        if augmented.dim() == 1:
            # Single sample
            augmented[:6] += shift
        else:
            # Batch of samples
            augmented[:, :6] += shift

        return augmented


class MetadataJitter(BaseAugmentation):
    """
    Add small perturbations to continuous metadata features.

    Simulates measurement uncertainties in age and weight measurements.
    """

    def __init__(self, jitter_strength: float = 0.02, probability: float = 0.5):
        """
        Initialize metadata jittering.

        Args:
            jitter_strength: Strength of jittering relative to feature scale
            probability: Probability of applying jitter
        """
        super().__init__(probability)
        self.jitter_strength = jitter_strength

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Apply jitter to continuous metadata features."""
        if torch.rand(1).item() > self.probability:
            return features

        augmented = features.clone()

        # Apply jitter to continuous metadata (indices 6-7: age, weight)
        # and PCA features (indices 16-17)
        continuous_indices = [6, 7, 16, 17]

        for idx in continuous_indices:
            if idx < len(features):
                jitter = torch.randn(1) * self.jitter_strength
                augmented[idx] += jitter.item()

        return augmented


class FrequencyMasking(BaseAugmentation):
    """
    Mask specific frequency bands in ABR data.

    Simulates frequency-specific hearing loss or technical issues
    at particular frequencies.
    """

    def __init__(self, max_mask_size: int = 2, probability: float = 0.3):
        """
        Initialize frequency masking.

        Args:
            max_mask_size: Maximum number of consecutive frequencies to mask
            probability: Probability of applying masking
        """
        super().__init__(probability)
        self.max_mask_size = max_mask_size

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking to ABR features."""
        if torch.rand(1).item() > self.probability:
            return features

        augmented = features.clone()

        # Mask random frequency band in ABR features (indices 0-5)
        abr_length = 6
        mask_size = torch.randint(1, min(self.max_mask_size + 1, abr_length), (1,)).item()
        start_idx = torch.randint(0, abr_length - mask_size + 1, (1,)).item()

        # Set masked frequencies to mean of remaining frequencies
        remaining_indices = [i for i in range(6) if i < start_idx or i >= start_idx + mask_size]
        if remaining_indices:
            mean_value = augmented[remaining_indices].mean()
            augmented[start_idx:start_idx + mask_size] = mean_value

        return augmented


class CompositeAugmentation:
    """
    Composite augmentation that applies multiple augmentations in sequence.

    Allows for complex, realistic data augmentations that combine multiple
    biologically-plausible variations.
    """

    def __init__(self, augmentations: List[BaseAugmentation],
                 apply_probability: float = 1.0):
        """
        Initialize composite augmentation.

        Args:
            augmentations: List of augmentation functions to apply
            apply_probability: Probability of applying the entire sequence
        """
        self.augmentations = augmentations
        self.apply_probability = apply_probability

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Apply sequence of augmentations."""
        if torch.rand(1).item() > self.apply_probability:
            return features

        augmented = features.clone()

        for augmentation in self.augmentations:
            augmented = augmentation(augmented)

        return augmented


class ContrastiveAugmentationPipeline:
    """
    Complete augmentation pipeline for contrastive learning.

    Provides carefully tuned augmentations that preserve biological meaning
    while creating sufficient diversity for effective contrastive learning.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize augmentation pipeline.

        Args:
            config: Configuration dictionary for augmentation parameters
        """
        if config is None:
            config = self._default_config()

        self.config = config

        # Create individual augmentations
        self.augmentations = self._create_augmentations()

        # Create composite pipeline
        self.pipeline = CompositeAugmentation(
            self.augmentations,
            apply_probability=config.get('pipeline_probability', 0.9)
        )

        logger.info(f"Initialized contrastive augmentation pipeline with "
                   f"{len(self.augmentations)} augmentations")

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Apply augmentation pipeline."""
        return self.pipeline(features)

    def _create_augmentations(self) -> List[BaseAugmentation]:
        """Create list of augmentations based on configuration."""
        augs = []

        # Gaussian noise (most common)
        augs.append(GaussianNoise(
            noise_std=self.config.get('noise_std', 0.05),
            probability=self.config.get('noise_probability', 0.8)
        ))

        # ABR-specific augmentations
        augs.append(ABRFrequencySmoothing(
            smoothing_strength=self.config.get('smoothing_strength', 0.1),
            probability=self.config.get('smoothing_probability', 0.6)
        ))

        augs.append(ABRShift(
            max_shift=self.config.get('shift_magnitude', 0.1),
            probability=self.config.get('shift_probability', 0.7)
        ))

        # Feature-level augmentations
        augs.append(FeatureDropout(
            dropout_prob=self.config.get('dropout_prob', 0.1),
            probability=self.config.get('dropout_probability', 0.4)
        ))

        # Metadata augmentations
        augs.append(MetadataJitter(
            jitter_strength=self.config.get('jitter_strength', 0.02),
            probability=self.config.get('jitter_probability', 0.5)
        ))

        # Frequency masking (less common, more aggressive)
        augs.append(FrequencyMasking(
            max_mask_size=self.config.get('mask_size', 2),
            probability=self.config.get('mask_probability', 0.3)
        ))

        return augs

    def _default_config(self) -> Dict:
        """Create default augmentation configuration."""
        return {
            'noise_std': 0.05,
            'noise_probability': 0.8,
            'smoothing_strength': 0.1,
            'smoothing_probability': 0.6,
            'shift_magnitude': 0.1,
            'shift_probability': 0.7,
            'dropout_prob': 0.1,
            'dropout_probability': 0.4,
            'jitter_strength': 0.02,
            'jitter_probability': 0.5,
            'mask_size': 2,
            'mask_probability': 0.3,
            'pipeline_probability': 0.9
        }

    def get_augmentation_info(self) -> Dict[str, int]:
        """Get information about the augmentation pipeline."""
        return {
            'num_augmentations': len(self.augmentations),
            'augmentation_types': [type(aug).__name__ for aug in self.augmentations],
            'config': self.config
        }


def create_augmentation_pipeline(config: Optional[Dict] = None) -> ContrastiveAugmentationPipeline:
    """
    Factory function to create augmentation pipeline.

    Args:
        config: Optional configuration override

    Returns:
        Configured augmentation pipeline
    """
    return ContrastiveAugmentationPipeline(config)


def create_light_augmentation() -> ContrastiveAugmentationPipeline:
    """Create a light augmentation pipeline for validation/testing."""
    light_config = {
        'noise_std': 0.02,
        'noise_probability': 0.5,
        'smoothing_strength': 0.05,
        'smoothing_probability': 0.3,
        'shift_magnitude': 0.05,
        'shift_probability': 0.4,
        'dropout_prob': 0.05,
        'dropout_probability': 0.2,
        'jitter_strength': 0.01,
        'jitter_probability': 0.3,
        'mask_size': 1,
        'mask_probability': 0.1,
        'pipeline_probability': 0.6
    }

    return ContrastiveAugmentationPipeline(light_config)


def create_strong_augmentation() -> ContrastiveAugmentationPipeline:
    """Create a strong augmentation pipeline for robust training."""
    strong_config = {
        'noise_std': 0.1,
        'noise_probability': 0.9,
        'smoothing_strength': 0.2,
        'smoothing_probability': 0.8,
        'shift_magnitude': 0.15,
        'shift_probability': 0.8,
        'dropout_prob': 0.15,
        'dropout_probability': 0.6,
        'jitter_strength': 0.03,
        'jitter_probability': 0.7,
        'mask_size': 3,
        'mask_probability': 0.5,
        'pipeline_probability': 0.95
    }

    return ContrastiveAugmentationPipeline(strong_config)