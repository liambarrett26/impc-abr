"""
Reconstruction decoder for ContrastiveVAE-DEC model.

This module implements the decoder component that reconstructs the original
18-dimensional audiometric features from the latent representation, with
specialized handling for different feature types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ABRFeatureDecoder(nn.Module):
    """
    Specialized decoder for ABR threshold features.

    Reconstructs the first 6 dimensions (ABR thresholds) with constraints
    that ensure physiologically plausible outputs.
    """

    def __init__(self, input_dim: int, config: Dict):
        """
        Initialize ABR feature decoder.

        Args:
            input_dim: Input dimension from main decoder
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.abr_dim = config['data']['abr_features']  # 6

        # ABR reconstruction layers
        self.abr_layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32) if config['decoder']['use_batch_norm'] else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config['decoder']['dropout_rate']),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16) if config['decoder']['use_batch_norm'] else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config['decoder']['dropout_rate']),
            nn.Linear(16, self.abr_dim)
        )

        # Output activation - ensure positive outputs for thresholds
        self.output_activation = nn.ReLU()  # ABR thresholds should be positive

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode ABR features.

        Args:
            x: Latent representation

        Returns:
            Reconstructed ABR thresholds (batch_size, 6)
        """
        abr_features = self.abr_layers(x)

        # Apply activation to ensure positive thresholds
        abr_features = self.output_activation(abr_features)

        # Optional: Add physiological constraints (0-100 dB SPL range)
        # This helps ensure reconstructed values are in realistic ranges
        abr_features = torch.clamp(abr_features, min=0.0, max=100.0)

        return abr_features


class MetadataDecoder(nn.Module):
    """
    Decoder for metadata and PCA features.

    Reconstructs the remaining 12 dimensions with appropriate handling
    for mixed continuous and categorical-derived features.
    """

    def __init__(self, input_dim: int, config: Dict):
        """
        Initialize metadata decoder.

        Args:
            input_dim: Input dimension from main decoder
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.metadata_dim = config['data']['metadata_features'] + config['data']['pca_features']  # 12

        # Metadata reconstruction layers
        self.metadata_layers = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.BatchNorm1d(24) if config['decoder']['use_batch_norm'] else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config['decoder']['dropout_rate']),
            nn.Linear(24, 16),
            nn.BatchNorm1d(16) if config['decoder']['use_batch_norm'] else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config['decoder']['dropout_rate']),
            nn.Linear(16, self.metadata_dim)
        )

        # Linear output for continuous features (normalized data)
        # No additional activation needed as features are normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode metadata features.

        Args:
            x: Latent representation

        Returns:
            Reconstructed metadata features (batch_size, 12)
        """
        metadata_features = self.metadata_layers(x)
        return metadata_features


class ContrastiveVAEDecoder(nn.Module):
    """
    Complete decoder for ContrastiveVAE-DEC model.

    Reconstructs the full 18-dimensional feature space from latent representations,
    with specialized sub-decoders for different feature types.
    """

    def __init__(self, config: Dict):
        """
        Initialize the complete decoder.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config

        # Decoder parameters
        self.latent_dim = config['latent']['latent_dim']  # 10
        self.hidden_dims = config['decoder']['hidden_dims']  # [16, 32, 64]
        self.dropout_rate = config['decoder']['dropout_rate']
        self.total_features = config['data']['total_features']  # 18

        # Build main decoder network
        self.decoder_layers = self._build_decoder_layers()

        # Feature-specific decoders
        decoder_output_dim = self.hidden_dims[-1]  # 64
        self.abr_decoder = ABRFeatureDecoder(decoder_output_dim, config)
        self.metadata_decoder = MetadataDecoder(decoder_output_dim, config)

        # Initialize weights
        self.apply(self._init_weights)

        logger.info(f"Initialized ContrastiveVAE decoder: "
                   f"latent_dim={self.latent_dim}, "
                   f"output_dim={self.total_features}, "
                   f"hidden_dims={self.hidden_dims}")

    def _build_decoder_layers(self) -> nn.ModuleList:
        """Build the main decoder network layers."""
        layers = nn.ModuleList()

        input_dim = self.latent_dim

        for hidden_dim in self.hidden_dims:
            layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if self.config['decoder']['use_batch_norm'] else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            )
            layers.append(layer)
            input_dim = hidden_dim

        return layers

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            if self.config['architecture']['weight_init'] == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            elif self.config['architecture']['weight_init'] == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the decoder.

        Args:
            z: Latent representation (batch_size, latent_dim)

        Returns:
            Dictionary containing:
                - reconstruction: Full reconstructed features (batch_size, 18)
                - abr_reconstruction: ABR features (batch_size, 6)
                - metadata_reconstruction: Metadata features (batch_size, 12)
                - decoder_features: Intermediate decoder features
        """
        # Pass through main decoder layers
        decoder_features = z
        for layer in self.decoder_layers:
            decoder_features = layer(decoder_features)

        # Decode feature-specific components
        abr_reconstruction = self.abr_decoder(decoder_features)
        metadata_reconstruction = self.metadata_decoder(decoder_features)

        # Combine reconstructions
        full_reconstruction = torch.cat([abr_reconstruction, metadata_reconstruction], dim=1)

        return {
            'reconstruction': full_reconstruction,
            'abr_reconstruction': abr_reconstruction,
            'metadata_reconstruction': metadata_reconstruction,
            'decoder_features': decoder_features
        }

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Simple decode function returning only the reconstruction.

        Args:
            z: Latent representation

        Returns:
            Reconstructed features
        """
        output = self.forward(z)
        return output['reconstruction']

    def decode_abr_only(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode only ABR features for audiogram analysis.

        Args:
            z: Latent representation

        Returns:
            Reconstructed ABR thresholds
        """
        decoder_features = z
        for layer in self.decoder_layers:
            decoder_features = layer(decoder_features)

        return self.abr_decoder(decoder_features)

    def get_feature_weights(self) -> Dict[str, torch.Tensor]:
        """
        Extract decoder weights for interpretability analysis.

        Returns:
            Dictionary of decoder weights for different feature types
        """
        weights = {}

        # Get final layer weights for each decoder
        if hasattr(self.abr_decoder.abr_layers[-1], 'weight'):
            weights['abr_weights'] = self.abr_decoder.abr_layers[-1].weight.data.clone()

        if hasattr(self.metadata_decoder.metadata_layers[-1], 'weight'):
            weights['metadata_weights'] = self.metadata_decoder.metadata_layers[-1].weight.data.clone()

        # Get main decoder weights
        main_weights = []
        for layer in self.decoder_layers:
            if hasattr(layer[0], 'weight'):  # First element should be Linear layer
                main_weights.append(layer[0].weight.data.clone())

        if main_weights:
            weights['main_decoder_weights'] = main_weights

        return weights

    def analyze_latent_to_feature_mapping(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze how latent dimensions map to output features.

        Args:
            z: Latent representation

        Returns:
            Dictionary of feature importance mappings
        """
        z.requires_grad_(True)

        # Forward pass
        output = self.forward(z)
        reconstruction = output['reconstruction']

        # Compute gradients for each output feature
        feature_gradients = []

        for i in range(reconstruction.shape[1]):
            # Compute gradient of feature i with respect to latent space
            feature_sum = reconstruction[:, i].sum()
            grad = torch.autograd.grad(feature_sum, z, retain_graph=True)[0]
            feature_gradients.append(grad.abs().mean(dim=0))

        # Stack gradients: (num_features, latent_dim)
        gradient_matrix = torch.stack(feature_gradients)

        return {
            'latent_to_feature_importance': gradient_matrix,
            'abr_importance': gradient_matrix[:6],  # First 6 features
            'metadata_importance': gradient_matrix[6:]  # Remaining features
        }


class ReconstructionValidator:
    """
    Utility class for validating reconstruction quality and constraints.
    """

    def __init__(self, config: Dict):
        """Initialize validator with configuration."""
        self.config = config
        self.abr_min = 0.0
        self.abr_max = 100.0  # dB SPL range

    def validate_reconstruction(self, original: torch.Tensor,
                              reconstructed: torch.Tensor) -> Dict[str, float]:
        """
        Validate reconstruction quality and constraints.

        Args:
            original: Original features
            reconstructed: Reconstructed features

        Returns:
            Dictionary of validation metrics
        """
        with torch.no_grad():
            # Overall reconstruction error
            mse_loss = F.mse_loss(reconstructed, original)
            mae_loss = F.l1_loss(reconstructed, original)

            # ABR-specific validation
            abr_original = original[:, :6]
            abr_reconstructed = reconstructed[:, :6]

            abr_mse = F.mse_loss(abr_reconstructed, abr_original)
            abr_mae = F.l1_loss(abr_reconstructed, abr_original)

            # Check ABR constraints
            abr_valid_range = ((abr_reconstructed >= self.abr_min) &
                              (abr_reconstructed <= self.abr_max)).float().mean()

            # Metadata validation
            metadata_original = original[:, 6:]
            metadata_reconstructed = reconstructed[:, 6:]

            metadata_mse = F.mse_loss(metadata_reconstructed, metadata_original)
            metadata_mae = F.l1_loss(metadata_reconstructed, metadata_original)

            return {
                'total_mse': mse_loss.item(),
                'total_mae': mae_loss.item(),
                'abr_mse': abr_mse.item(),
                'abr_mae': abr_mae.item(),
                'abr_valid_range': abr_valid_range.item(),
                'metadata_mse': metadata_mse.item(),
                'metadata_mae': metadata_mae.item()
            }

    def check_audiogram_plausibility(self, abr_features: torch.Tensor) -> Dict[str, float]:
        """
        Check if reconstructed audiograms are physiologically plausible.

        Args:
            abr_features: ABR threshold features (batch_size, 6)

        Returns:
            Dictionary of plausibility metrics
        """
        with torch.no_grad():
            # Check for reasonable threshold ranges
            in_range = ((abr_features >= self.abr_min) &
                       (abr_features <= self.abr_max)).float()
            range_validity = in_range.mean()

            # Check for reasonable frequency patterns
            # Compute gradient across frequencies
            freq_gradients = torch.diff(abr_features, dim=1)

            # Reasonable threshold differences (not too extreme)
            reasonable_changes = (torch.abs(freq_gradients) < 50.0).float()
            pattern_validity = reasonable_changes.mean()

            # Check for monotonicity violations (optional)
            # Most hearing loss patterns show some structure
            std_per_mouse = abr_features.std(dim=1)
            variability_score = (std_per_mouse > 1.0).float().mean()  # Some variability expected

            return {
                'range_validity': range_validity.item(),
                'pattern_validity': pattern_validity.item(),
                'variability_score': variability_score.item()
            }