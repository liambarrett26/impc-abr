"""
Reconstruction losses for ContrastiveVAE-DEC model.

This module provides specialized reconstruction losses for different feature types
in audiometric data, with emphasis on preserving meaningful hearing patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureWeightedMSE(nn.Module):
    """
    Feature-weighted MSE loss for different audiometric feature types.

    Applies different weights to ABR, metadata, and PCA features to reflect
    their relative importance in phenotype characterization.
    """

    def __init__(self, feature_weights: Dict[str, float]):
        """
        Initialize weighted MSE loss.

        Args:
            feature_weights: Dictionary of weights for each feature type
        """
        super().__init__()
        self.feature_weights = feature_weights
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, reconstruction: torch.Tensor,
                target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute feature-weighted MSE loss.

        Args:
            reconstruction: Reconstructed features (batch_size, 18)
            target: Original features (batch_size, 18)

        Returns:
            Dictionary of loss components
        """
        # Split features by type
        abr_recon = reconstruction[:, :6]
        abr_target = target[:, :6]
        metadata_recon = reconstruction[:, 6:16]
        metadata_target = target[:, 6:16]
        pca_recon = reconstruction[:, 16:18]
        pca_target = target[:, 16:18]

        # Compute individual losses
        abr_loss = self.mse_loss(abr_recon, abr_target).mean(dim=1)
        metadata_loss = self.mse_loss(metadata_recon, metadata_target).mean(dim=1)
        pca_loss = self.mse_loss(pca_recon, pca_target).mean(dim=1)

        # Apply feature weights
        weighted_abr = abr_loss * self.feature_weights.get('abr', 1.0)
        weighted_metadata = metadata_loss * self.feature_weights.get('metadata', 1.0)
        weighted_pca = pca_loss * self.feature_weights.get('pca', 1.0)

        # Total weighted loss
        total_loss = weighted_abr + weighted_metadata + weighted_pca

        return {
            'total_loss': total_loss,
            'abr_loss': abr_loss,
            'metadata_loss': metadata_loss,
            'pca_loss': pca_loss,
            'weighted_abr': weighted_abr,
            'weighted_metadata': weighted_metadata,
            'weighted_pca': weighted_pca
        }


class ABRPatternLoss(nn.Module):
    """
    Specialized loss for preserving ABR frequency patterns.

    Focuses on maintaining realistic audiogram shapes and frequency relationships
    beyond simple point-wise reconstruction error.
    """

    def __init__(self, pattern_weight: float = 0.1):
        """
        Initialize ABR pattern loss.

        Args:
            pattern_weight: Weight for pattern preservation term
        """
        super().__init__()
        self.pattern_weight = pattern_weight
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, abr_reconstruction: torch.Tensor,
                abr_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute ABR pattern-aware reconstruction loss.

        Args:
            abr_reconstruction: Reconstructed ABR thresholds (batch_size, 6)
            abr_target: Original ABR thresholds (batch_size, 6)

        Returns:
            Dictionary of loss components
        """
        # Basic MSE loss
        mse_loss = self.mse_loss(abr_reconstruction, abr_target).mean(dim=1)

        # Frequency gradient preservation
        gradient_loss = self._compute_gradient_loss(abr_reconstruction, abr_target)

        # Audiogram shape preservation
        shape_loss = self._compute_shape_loss(abr_reconstruction, abr_target)

        # Range constraint loss (ensure physiological plausibility)
        range_loss = self._compute_range_loss(abr_reconstruction)

        # Combined pattern loss
        pattern_loss = gradient_loss + shape_loss + range_loss

        # Total loss
        total_loss = mse_loss + self.pattern_weight * pattern_loss

        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'gradient_loss': gradient_loss,
            'shape_loss': shape_loss,
            'range_loss': range_loss,
            'pattern_loss': pattern_loss
        }

    def _compute_gradient_loss(self, reconstruction: torch.Tensor,
                              target: torch.Tensor) -> torch.Tensor:
        """Preserve frequency-to-frequency gradient patterns."""
        # Compute gradients across frequencies
        recon_grad = torch.diff(reconstruction, dim=1)
        target_grad = torch.diff(target, dim=1)

        # MSE loss on gradients
        gradient_loss = F.mse_loss(recon_grad, target_grad, reduction='none').mean(dim=1)

        return gradient_loss

    def _compute_shape_loss(self, reconstruction: torch.Tensor,
                           target: torch.Tensor) -> torch.Tensor:
        """Preserve overall audiogram shape characteristics."""
        # Compute relative patterns (normalize by mean)
        recon_mean = reconstruction.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)

        recon_normalized = reconstruction / (recon_mean + 1e-8)
        target_normalized = target / (target_mean + 1e-8)

        # Shape loss
        shape_loss = F.mse_loss(recon_normalized, target_normalized, reduction='none').mean(dim=1)

        return shape_loss

    def _compute_range_loss(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """Penalize reconstructions outside physiological range."""
        # ABR thresholds should be between 0-100 dB SPL
        min_threshold = 0.0
        max_threshold = 100.0

        # Penalty for values outside range
        below_min = F.relu(min_threshold - reconstruction)
        above_max = F.relu(reconstruction - max_threshold)

        range_loss = (below_min + above_max).mean(dim=1)

        return range_loss


class PerceptualABRLoss(nn.Module):
    """
    Perceptually-motivated loss for ABR reconstruction.

    Uses auditory perception principles to weight frequency importance
    and emphasize clinically relevant threshold differences.
    """

    def __init__(self, clinical_threshold: float = 25.0):
        """
        Initialize perceptual ABR loss.

        Args:
            clinical_threshold: Clinical threshold for hearing loss (dB SPL)
        """
        super().__init__()
        self.clinical_threshold = clinical_threshold

        # Frequency importance
        self.register_buffer('freq_weights', torch.tensor([
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.8   # Click - lower importance
        ]))

    def forward(self, abr_reconstruction: torch.Tensor,
                abr_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute perceptually-weighted ABR loss.

        Args:
            abr_reconstruction: Reconstructed ABR thresholds (batch_size, 6)
            abr_target: Original ABR thresholds (batch_size, 6)

        Returns:
            Dictionary of loss components
        """
        # Frequency-weighted MSE
        mse_per_freq = F.mse_loss(abr_reconstruction, abr_target, reduction='none')
        weighted_mse = mse_per_freq * self.freq_weights.unsqueeze(0)
        freq_weighted_loss = weighted_mse.mean(dim=1)

        # Clinical threshold crossing loss
        clinical_loss = self._compute_clinical_loss(abr_reconstruction, abr_target)

        # Hearing loss severity matching
        severity_loss = self._compute_severity_loss(abr_reconstruction, abr_target)

        # Total perceptual loss
        total_loss = freq_weighted_loss + clinical_loss + severity_loss

        return {
            'total_loss': total_loss,
            'freq_weighted_loss': freq_weighted_loss,
            'clinical_loss': clinical_loss,
            'severity_loss': severity_loss
        }

    def _compute_clinical_loss(self, reconstruction: torch.Tensor,
                              target: torch.Tensor) -> torch.Tensor:
        """Emphasize accurate reconstruction around clinical thresholds."""
        # Distance from clinical threshold
        recon_dist = torch.abs(reconstruction - self.clinical_threshold)
        target_dist = torch.abs(target - self.clinical_threshold)

        # Higher penalty for disagreement near clinical threshold
        clinical_weights = torch.exp(-recon_dist / 10.0) + torch.exp(-target_dist / 10.0)

        mse_loss = F.mse_loss(reconstruction, target, reduction='none')
        clinical_weighted = mse_loss * clinical_weights

        return clinical_weighted.mean(dim=1)

    def _compute_severity_loss(self, reconstruction: torch.Tensor,
                              target: torch.Tensor) -> torch.Tensor:
        """Match hearing loss severity categories."""
        # Define severity bins (normal, mild, moderate, severe)
        severity_bounds = torch.tensor([25.0, 40.0, 70.0], device=reconstruction.device)

        # Compute severity categories
        recon_severity = torch.searchsorted(severity_bounds, reconstruction)
        target_severity = torch.searchsorted(severity_bounds, target)

        # Categorical loss (penalize severity mismatches)
        severity_mismatch = (recon_severity != target_severity).float()

        return severity_mismatch.mean(dim=1)


class AdaptiveReconstructionLoss(nn.Module):
    """
    Adaptive reconstruction loss that adjusts based on training progress.

    Starts with simple MSE and gradually incorporates more sophisticated
    pattern-based losses as training progresses.
    """

    def __init__(self, config: Dict):
        """Initialize adaptive reconstruction loss."""
        super().__init__()
        self.config = config

        # Component losses
        self.weighted_mse = FeatureWeightedMSE(
            config['loss_weights']['reconstruction_weights']
        )
        self.pattern_loss = ABRPatternLoss(pattern_weight=0.1)
        self.perceptual_loss = PerceptualABRLoss()

        # Adaptation parameters
        self.warmup_epochs = config.get('warmup_epochs', 50)
        self.current_epoch = 0

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor,
                epoch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive reconstruction loss.

        Args:
            reconstruction: Reconstructed features (batch_size, 18)
            target: Original features (batch_size, 18)
            epoch: Current training epoch (for adaptation)

        Returns:
            Dictionary of loss components
        """
        if epoch is not None:
            self.current_epoch = epoch

        # Compute base weighted MSE loss
        mse_losses = self.weighted_mse(reconstruction, target)

        # Extract ABR components for pattern losses
        abr_recon = reconstruction[:, :6]
        abr_target = target[:, :6]

        # Compute pattern and perceptual losses
        pattern_losses = self.pattern_loss(abr_recon, abr_target)
        perceptual_losses = self.perceptual_loss(abr_recon, abr_target)

        # Adaptive weighting based on training progress
        progress = min(self.current_epoch / self.warmup_epochs, 1.0)

        # Start with simple MSE, gradually add complexity
        mse_weight = 1.0
        pattern_weight = 0.2 * progress
        perceptual_weight = 0.1 * progress

        # Combined adaptive loss
        total_loss = (
            mse_weight * mse_losses['total_loss'] +
            pattern_weight * pattern_losses['total_loss'] +
            perceptual_weight * perceptual_losses['total_loss']
        )

        # Combine all components
        combined_losses = {
            'total_loss': total_loss,
            'mse_component': mse_losses['total_loss'],
            'pattern_component': pattern_losses['total_loss'],
            'perceptual_component': perceptual_losses['total_loss'],
            'mse_weight': mse_weight,
            'pattern_weight': pattern_weight,
            'perceptual_weight': perceptual_weight,
            **{f'mse_{k}': v for k, v in mse_losses.items()},
            **{f'pattern_{k}': v for k, v in pattern_losses.items()},
            **{f'perceptual_{k}': v for k, v in perceptual_losses.items()}
        }

        return combined_losses

    def update_epoch(self, epoch: int):
        """Update current epoch for adaptive weighting."""
        self.current_epoch = epoch


def create_reconstruction_loss(config: Dict) -> nn.Module:
    """
    Factory function to create appropriate reconstruction loss.

    Args:
        config: Loss configuration dictionary

    Returns:
        Configured reconstruction loss module
    """
    loss_type = config.get('reconstruction_loss_type', 'adaptive')

    if loss_type == 'weighted_mse':
        return FeatureWeightedMSE(config['loss_weights']['reconstruction_weights'])
    elif loss_type == 'pattern':
        return ABRPatternLoss()
    elif loss_type == 'perceptual':
        return PerceptualABRLoss()
    elif loss_type == 'adaptive':
        return AdaptiveReconstructionLoss(config)
    else:
        raise ValueError(f"Unknown reconstruction loss type: {loss_type}")


class ReconstructionLossAnalyzer:
    """Utility class for analyzing reconstruction loss behavior."""

    def __init__(self):
        """Initialize reconstruction loss analyzer."""
        pass

    def analyze_frequency_errors(self, reconstruction: torch.Tensor,
                                target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze reconstruction errors by frequency.

        Args:
            reconstruction: Reconstructed ABR features (batch_size, 6)
            target: Original ABR features (batch_size, 6)

        Returns:
            Dictionary of frequency-specific error metrics
        """
        with torch.no_grad():
            # Per-frequency MSE
            freq_mse = F.mse_loss(reconstruction, target, reduction='none')

            # Per-frequency MAE
            freq_mae = F.l1_loss(reconstruction, target, reduction='none')

            # Mean errors across batch
            mean_mse = freq_mse.mean(dim=0)
            mean_mae = freq_mae.mean(dim=0)

            # Error statistics
            return {
                'freq_mse': mean_mse,
                'freq_mae': mean_mae,
                'total_mse': mean_mse.mean(),
                'total_mae': mean_mae.mean(),
                'worst_freq_mse': torch.argmax(mean_mse),
                'best_freq_mse': torch.argmin(mean_mse)
            }

    def compute_reconstruction_metrics(self, reconstruction: torch.Tensor,
                                     target: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive reconstruction quality metrics.

        Args:
            reconstruction: Reconstructed features
            target: Original features

        Returns:
            Dictionary of quality metrics
        """
        with torch.no_grad():
            # Basic metrics
            mse = F.mse_loss(reconstruction, target)
            mae = F.l1_loss(reconstruction, target)

            # Correlation
            recon_flat = reconstruction.view(-1)
            target_flat = target.view(-1)
            correlation = torch.corrcoef(torch.stack([recon_flat, target_flat]))[0, 1]

            # R² score
            ss_res = torch.sum((target_flat - recon_flat) ** 2)
            ss_tot = torch.sum((target_flat - target_flat.mean()) ** 2)
            r2_score = 1 - ss_res / ss_tot

            return {
                'mse': mse.item(),
                'mae': mae.item(),
                'rmse': torch.sqrt(mse).item(),
                'correlation': correlation.item(),
                'r2_score': r2_score.item()
            }