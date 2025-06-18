"""
VAE components for ContrastiveVAE-DEC model.

This module provides Variational Autoencoder components including
latent space operations, KL divergence computation, and VAE-specific
utilities for audiometric phenotype modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from typing import Dict, Tuple, Optional
import logging
import math

logger = logging.getLogger(__name__)


class LatentSpace(nn.Module):
    """
    Latent space module for VAE with reparameterization and sampling utilities.

    Handles the probabilistic latent representation and provides various
    sampling and interpolation methods for analysis.
    """

    def __init__(self, config: Dict):
        """
        Initialize latent space.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.latent_dim = config['latent']['latent_dim']
        self.beta = config['latent']['beta']  # Beta-VAE parameter
        self.min_logvar = config['latent']['min_logvar']
        self.max_logvar = config['latent']['max_logvar']

        # Prior distribution (standard normal)
        self.register_buffer('prior_mean', torch.zeros(self.latent_dim))
        self.register_buffer('prior_logvar', torch.zeros(self.latent_dim))

        logger.info(f"Initialized latent space: dim={self.latent_dim}, beta={self.beta}")

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for differentiable sampling.

        Args:
            mu: Mean of posterior distribution (batch_size, latent_dim)
            logvar: Log variance of posterior distribution (batch_size, latent_dim)

        Returns:
            Sampled latent vectors (batch_size, latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # Use mean during inference
            return mu

    def sample_prior(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample from prior distribution.

        Args:
            batch_size: Number of samples
            device: Device to create samples on

        Returns:
            Samples from prior (batch_size, latent_dim)
        """
        return torch.randn(batch_size, self.latent_dim, device=device)

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior.

        Args:
            mu: Posterior mean
            logvar: Posterior log variance

        Returns:
            KL divergence (batch_size,)
        """
        # KL(q(z|x) || p(z)) for diagonal Gaussian
        # = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_div

    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor,
                   num_steps: int = 10) -> torch.Tensor:
        """
        Linear interpolation between two latent vectors.

        Args:
            z1: First latent vector
            z2: Second latent vector
            num_steps: Number of interpolation steps

        Returns:
            Interpolated latent vectors (num_steps, latent_dim)
        """
        alphas = torch.linspace(0, 1, num_steps, device=z1.device).unsqueeze(1)
        return alphas * z2 + (1 - alphas) * z1

    def spherical_interpolation(self, z1: torch.Tensor, z2: torch.Tensor,
                               num_steps: int = 10) -> torch.Tensor:
        """
        Spherical interpolation (slerp) between latent vectors.

        More appropriate for high-dimensional latent spaces.

        Args:
            z1: First latent vector (normalized)
            z2: Second latent vector (normalized)
            num_steps: Number of interpolation steps

        Returns:
            Interpolated latent vectors (num_steps, latent_dim)
        """
        # Normalize vectors
        z1_norm = F.normalize(z1, dim=-1)
        z2_norm = F.normalize(z2, dim=-1)

        # Compute angle between vectors
        dot_product = torch.sum(z1_norm * z2_norm, dim=-1, keepdim=True)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        omega = torch.acos(dot_product)

        # Handle parallel vectors
        sin_omega = torch.sin(omega)
        parallel_mask = sin_omega < 1e-6

        results = []
        for i in range(num_steps):
            t = i / (num_steps - 1)

            if torch.any(parallel_mask):
                # Use linear interpolation for parallel vectors
                interp = (1 - t) * z1_norm + t * z2_norm
            else:
                # Spherical interpolation
                sin_t_omega = torch.sin(t * omega)
                sin_1_t_omega = torch.sin((1 - t) * omega)

                interp = (sin_1_t_omega * z1_norm + sin_t_omega * z2_norm) / sin_omega

            results.append(interp)

        return torch.stack(results)


class VAELoss(nn.Module):
    """
    Complete VAE loss computation including reconstruction and KL terms.

    Supports beta-VAE weighting and different reconstruction loss types
    for different feature components.
    """

    def __init__(self, config: Dict):
        """
        Initialize VAE loss.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.beta = config['latent']['beta']
        self.reconstruction_weights = config['loss_weights']['reconstruction_weights']

        # Loss functions for different feature types
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')

    def reconstruction_loss(self, original: torch.Tensor,
                          reconstructed: torch.Tensor,
                          feature_weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute weighted reconstruction loss for different feature types.

        Args:
            original: Original features (batch_size, 18)
            reconstructed: Reconstructed features (batch_size, 18)
            feature_weights: Optional feature-specific weights

        Returns:
            Dictionary of reconstruction losses
        """
        if feature_weights is None:
            feature_weights = self.reconstruction_weights

        # Split features by type
        abr_original = original[:, :6]
        abr_reconstructed = reconstructed[:, :6]
        metadata_original = original[:, 6:16]  # Metadata features
        metadata_reconstructed = reconstructed[:, 6:16]
        pca_original = original[:, 16:18]  # PCA features
        pca_reconstructed = reconstructed[:, 16:18]

        # Compute losses for each feature type
        abr_loss = self.mse_loss(abr_reconstructed, abr_original).mean(dim=1)
        metadata_loss = self.mse_loss(metadata_reconstructed, metadata_original).mean(dim=1)
        pca_loss = self.mse_loss(pca_reconstructed, pca_original).mean(dim=1)

        # Apply feature-specific weights
        weighted_abr_loss = abr_loss * feature_weights.get('abr', 1.0)
        weighted_metadata_loss = metadata_loss * feature_weights.get('metadata', 1.0)
        weighted_pca_loss = pca_loss * feature_weights.get('pca', 1.0)

        # Total reconstruction loss
        total_reconstruction = weighted_abr_loss + weighted_metadata_loss + weighted_pca_loss

        return {
            'total_reconstruction': total_reconstruction,
            'abr_reconstruction': abr_loss,
            'metadata_reconstruction': metadata_loss,
            'pca_reconstruction': pca_loss,
            'weighted_abr': weighted_abr_loss,
            'weighted_metadata': weighted_metadata_loss,
            'weighted_pca': weighted_pca_loss
        }

    def kl_divergence_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss.

        Args:
            mu: Posterior mean
            logvar: Posterior log variance

        Returns:
            KL divergence loss (batch_size,)
        """
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_div

    def forward(self, original: torch.Tensor, reconstructed: torch.Tensor,
                mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute complete VAE loss.

        Args:
            original: Original features
            reconstructed: Reconstructed features
            mu: Latent mean
            logvar: Latent log variance

        Returns:
            Dictionary of loss components
        """
        # Reconstruction losses
        recon_losses = self.reconstruction_loss(original, reconstructed)

        # KL divergence loss
        kl_loss = self.kl_divergence_loss(mu, logvar)

        # Total VAE loss (ELBO)
        vae_loss = recon_losses['total_reconstruction'] + self.beta * kl_loss

        return {
            'vae_loss': vae_loss,
            'reconstruction_loss': recon_losses['total_reconstruction'],
            'kl_loss': kl_loss,
            'beta_kl_loss': self.beta * kl_loss,
            **recon_losses  # Include all reconstruction components
        }


class VAEAnalyzer:
    """
    Utility class for analyzing VAE behavior and latent space properties.
    """

    def __init__(self, config: Dict):
        """Initialize VAE analyzer."""
        self.config = config
        self.latent_dim = config['latent']['latent_dim']

    def compute_active_units(self, mu: torch.Tensor, threshold: float = 0.01) -> int:
        """
        Compute number of active latent units.

        Args:
            mu: Latent means from a batch
            threshold: Variance threshold for active units

        Returns:
            Number of active latent dimensions
        """
        with torch.no_grad():
            latent_variance = torch.var(mu, dim=0)
            active_units = (latent_variance > threshold).sum().item()
            return active_units

    def compute_mutual_information(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Estimate mutual information between latent dimensions.

        Args:
            mu: Latent means (batch_size, latent_dim)

        Returns:
            Mutual information matrix (latent_dim, latent_dim)
        """
        with torch.no_grad():
            # Compute correlation matrix as MI approximation
            mu_centered = mu - mu.mean(dim=0, keepdim=True)
            cov_matrix = torch.mm(mu_centered.t(), mu_centered) / (mu.shape[0] - 1)

            # Normalize to get correlation
            std_dev = torch.sqrt(torch.diag(cov_matrix))
            correlation = cov_matrix / (std_dev.unsqueeze(1) * std_dev.unsqueeze(0))

            # Convert correlation to MI estimate
            # MI ≈ -0.5 * log(1 - ρ²) for Gaussian variables
            correlation_squared = correlation.pow(2)
            mi_estimate = -0.5 * torch.log(1 - correlation_squared + 1e-8)

            return mi_estimate

    def analyze_latent_traversal(self, decoder: nn.Module, base_z: torch.Tensor,
                                dim_idx: int, num_steps: int = 10,
                                step_size: float = 2.0) -> torch.Tensor:
        """
        Analyze effect of traversing a single latent dimension.

        Args:
            decoder: Decoder model
            base_z: Base latent vector
            dim_idx: Dimension to traverse
            num_steps: Number of steps
            step_size: Step size for traversal

        Returns:
            Reconstructions along the traversal (num_steps, feature_dim)
        """
        device = base_z.device
        steps = torch.linspace(-step_size, step_size, num_steps, device=device)

        reconstructions = []

        with torch.no_grad():
            for step in steps:
                z_modified = base_z.clone()
                z_modified[dim_idx] = step

                # Decode
                reconstruction = decoder.decode(z_modified.unsqueeze(0))
                reconstructions.append(reconstruction.squeeze(0))

        return torch.stack(reconstructions)

    def compute_latent_statistics(self, mu: torch.Tensor,
                                 logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive latent space statistics.

        Args:
            mu: Latent means
            logvar: Latent log variances

        Returns:
            Dictionary of statistics
        """
        with torch.no_grad():
            std = torch.exp(0.5 * logvar)

            stats = {
                'mean_mu': torch.mean(mu, dim=0),
                'std_mu': torch.std(mu, dim=0),
                'mean_std': torch.mean(std, dim=0),
                'kl_per_dim': 0.5 * (mu.pow(2) + std.pow(2) - 1 - logvar).mean(dim=0),
                'total_kl': 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 1 - logvar, dim=1).mean(),
                'active_units': self.compute_active_units(mu),
                'capacity': self.compute_active_units(mu) / self.latent_dim
            }

            return stats


class BetaScheduler:
    """
    Scheduler for beta parameter in beta-VAE training.

    Allows for gradual increase of KL penalty during training.
    """

    def __init__(self, beta_max: float = 1.0, warmup_epochs: int = 10,
                 schedule_type: str = 'linear'):
        """
        Initialize beta scheduler.

        Args:
            beta_max: Maximum beta value
            warmup_epochs: Number of epochs for warmup
            schedule_type: Type of schedule ('linear', 'exponential', 'cosine')
        """
        self.beta_max = beta_max
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type
        self.current_epoch = 0

    def get_beta(self, epoch: int) -> float:
        """
        Get beta value for current epoch.

        Args:
            epoch: Current epoch

        Returns:
            Beta value
        """
        if epoch >= self.warmup_epochs:
            return self.beta_max

        progress = epoch / self.warmup_epochs

        if self.schedule_type == 'linear':
            return progress * self.beta_max
        elif self.schedule_type == 'exponential':
            return self.beta_max * (progress ** 2)
        elif self.schedule_type == 'cosine':
            return self.beta_max * (1 - math.cos(progress * math.pi / 2))
        else:
            return progress * self.beta_max

    def step(self) -> float:
        """
        Step the scheduler and return current beta.

        Returns:
            Current beta value
        """
        beta = self.get_beta(self.current_epoch)
        self.current_epoch += 1
        return beta