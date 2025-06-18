"""
VAE losses for ContrastiveVAE-DEC model.

This module provides KL divergence computation, ELBO calculation, and
beta-VAE scheduling for the variational autoencoder components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from typing import Dict, Tuple, Optional
import logging
import math

logger = logging.getLogger(__name__)


class KLDivergenceLoss(nn.Module):
    """
    KL divergence loss for VAE latent space regularization.

    Computes KL divergence between learned posterior and standard normal prior,
    with support for beta-VAE weighting and dimension-wise analysis.
    """

    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        """
        Initialize KL divergence loss.

        Args:
            beta: Beta parameter for beta-VAE (KL weight)
            reduction: How to reduce the loss ('mean', 'sum', 'none')
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute KL divergence loss.

        Args:
            mu: Posterior mean (batch_size, latent_dim)
            logvar: Posterior log variance (batch_size, latent_dim)

        Returns:
            Dictionary of KL loss components
        """
        # KL divergence for diagonal Gaussian: KL(q(z|x) || p(z))
        # = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        # Per-sample KL divergence
        kl_per_sample = torch.sum(kl_per_dim, dim=1)

        # Apply reduction
        if self.reduction == 'mean':
            kl_loss = torch.mean(kl_per_sample)
        elif self.reduction == 'sum':
            kl_loss = torch.sum(kl_per_sample)
        else:  # 'none'
            kl_loss = kl_per_sample

        # Beta-weighted KL loss
        beta_kl_loss = self.beta * kl_loss

        # Per-dimension KL for analysis
        kl_per_dim_mean = torch.mean(kl_per_dim, dim=0)

        return {
            'kl_loss': kl_loss,
            'beta_kl_loss': beta_kl_loss,
            'kl_per_sample': kl_per_sample,
            'kl_per_dim': kl_per_dim_mean,
            'total_kl': torch.sum(kl_per_dim_mean),
            'beta': self.beta
        }

    def update_beta(self, new_beta: float):
        """Update beta parameter for beta-VAE scheduling."""
        self.beta = new_beta


class ELBOLoss(nn.Module):
    """
    Evidence Lower Bound (ELBO) loss for VAE training.

    Combines reconstruction loss and KL divergence into the complete
    VAE objective function with flexible weighting and scheduling.
    """

    def __init__(self, reconstruction_loss_fn: nn.Module,
                 beta: float = 1.0,
                 free_bits: Optional[float] = None):
        """
        Initialize ELBO loss.

        Args:
            reconstruction_loss_fn: Reconstruction loss function
            beta: Beta parameter for KL weighting
            free_bits: Free bits constraint for KL (optional)
        """
        super().__init__()
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.kl_loss = KLDivergenceLoss(beta=beta)
        self.free_bits = free_bits

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor,
                mu: torch.Tensor, logvar: torch.Tensor,
                epoch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Compute ELBO loss.

        Args:
            reconstruction: Reconstructed features
            target: Original features
            mu: Latent posterior mean
            logvar: Latent posterior log variance
            epoch: Current epoch (for adaptive losses)

        Returns:
            Dictionary of ELBO loss components
        """
        # Compute reconstruction loss
        if hasattr(self.reconstruction_loss_fn, 'forward'):
            if epoch is not None and hasattr(self.reconstruction_loss_fn, 'update_epoch'):
                self.reconstruction_loss_fn.update_epoch(epoch)
            recon_losses = self.reconstruction_loss_fn(reconstruction, target, epoch)
            recon_loss = recon_losses['total_loss'].mean()
        else:
            recon_loss = self.reconstruction_loss_fn(reconstruction, target)
            recon_losses = {'total_loss': recon_loss}

        # Compute KL divergence
        kl_losses = self.kl_loss(mu, logvar)

        # Apply free bits constraint if specified
        if self.free_bits is not None:
            kl_per_dim = kl_losses['kl_per_dim']
            constrained_kl = torch.clamp(kl_per_dim, min=self.free_bits)
            kl_loss = torch.sum(constrained_kl)
        else:
            kl_loss = kl_losses['kl_loss']

        # ELBO = -reconstruction_log_likelihood + KL_divergence
        # We minimize this, so ELBO = reconstruction_loss + beta * KL_loss
        elbo_loss = recon_loss + kl_losses['beta_kl_loss']

        # Combine all components
        combined_losses = {
            'elbo_loss': elbo_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'beta_kl_loss': kl_losses['beta_kl_loss'],
            'beta': kl_losses['beta'],
            **kl_losses,
            **{f'recon_{k}': v for k, v in recon_losses.items()}
        }

        return combined_losses

    def update_beta(self, new_beta: float):
        """Update beta parameter."""
        self.kl_loss.update_beta(new_beta)


class BetaScheduler:
    """
    Scheduler for beta parameter in beta-VAE training.

    Provides various scheduling strategies for gradually increasing
    the KL divergence weight during training.
    """

    def __init__(self, beta_max: float = 1.0, warmup_epochs: int = 10,
                 schedule_type: str = 'linear', cycle_length: Optional[int] = None):
        """
        Initialize beta scheduler.

        Args:
            beta_max: Maximum beta value
            warmup_epochs: Number of epochs for beta warmup
            schedule_type: Scheduling strategy ('linear', 'exponential', 'cosine', 'cyclical')
            cycle_length: Length of cycles for cyclical scheduling
        """
        self.beta_max = beta_max
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type
        self.cycle_length = cycle_length
        self.current_epoch = 0

    def get_beta(self, epoch: Optional[int] = None) -> float:
        """
        Get beta value for current or specified epoch.

        Args:
            epoch: Epoch number (uses internal counter if None)

        Returns:
            Beta value for the epoch
        """
        if epoch is None:
            epoch = self.current_epoch

        if epoch >= self.warmup_epochs:
            if self.schedule_type == 'cyclical' and self.cycle_length:
                return self._cyclical_beta(epoch)
            else:
                return self.beta_max

        # Warmup phase
        progress = epoch / self.warmup_epochs

        if self.schedule_type == 'linear':
            return progress * self.beta_max
        elif self.schedule_type == 'exponential':
            return self.beta_max * (progress ** 2)
        elif self.schedule_type == 'cosine':
            return self.beta_max * (1 - math.cos(progress * math.pi / 2))
        else:
            return progress * self.beta_max

    def _cyclical_beta(self, epoch: int) -> float:
        """Compute cyclical beta value."""
        cycle_position = (epoch % self.cycle_length) / self.cycle_length
        # Cosine annealing within cycle
        return self.beta_max * (1 + math.cos(math.pi * cycle_position)) / 2

    def step(self) -> float:
        """Step the scheduler and return current beta."""
        beta = self.get_beta(self.current_epoch)
        self.current_epoch += 1
        return beta


class CapacityScheduler:
    """
    Capacity-based scheduler for controlling effective KL penalty.

    Gradually increases the target capacity (effective information content)
    of the latent space during training.
    """

    def __init__(self, max_capacity: float = 25.0, capacity_epochs: int = 100,
                 gamma: float = 1000.0):
        """
        Initialize capacity scheduler.

        Args:
            max_capacity: Maximum latent capacity (in nats)
            capacity_epochs: Epochs to reach max capacity
            gamma: Weight for capacity term
        """
        self.max_capacity = max_capacity
        self.capacity_epochs = capacity_epochs
        self.gamma = gamma
        self.current_epoch = 0

    def get_capacity_loss(self, kl_divergence: torch.Tensor, epoch: Optional[int] = None) -> torch.Tensor:
        """
        Compute capacity-constrained loss.

        Args:
            kl_divergence: KL divergence value
            epoch: Current epoch

        Returns:
            Capacity-constrained KL loss
        """
        if epoch is None:
            epoch = self.current_epoch

        # Current target capacity
        progress = min(epoch / self.capacity_epochs, 1.0)
        current_capacity = progress * self.max_capacity

        # Capacity loss: only penalize KL above target capacity
        capacity_loss = self.gamma * torch.abs(kl_divergence - current_capacity)

        return capacity_loss

    def step(self):
        """Step the capacity scheduler."""
        self.current_epoch += 1


class VAERegularizer:
    """
    Additional regularization techniques for VAE training.

    Provides various regularization methods to improve latent space
    quality and prevent common VAE training issues.
    """

    def __init__(self, config: Dict):
        """Initialize VAE regularizer with configuration."""
        self.config = config

    def spectral_normalization_loss(self, encoder_weights: List[torch.Tensor]) -> torch.Tensor:
        """
        Spectral normalization regularization for encoder stability.

        Args:
            encoder_weights: List of encoder weight matrices

        Returns:
            Spectral normalization loss
        """
        spectral_losses = []

        for weight in encoder_weights:
            if weight.dim() == 2:  # Only for linear layers
                # Compute spectral norm (largest singular value)
                u, s, v = torch.svd(weight)
                spectral_norm = s[0]

                # Penalty for large spectral norms
                spectral_loss = F.relu(spectral_norm - 1.0) ** 2
                spectral_losses.append(spectral_loss)

        if spectral_losses:
            return torch.stack(spectral_losses).mean()
        else:
            return torch.tensor(0.0)

    def total_correlation_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Total correlation regularization for disentanglement.

        Args:
            z: Latent samples (batch_size, latent_dim)

        Returns:
            Total correlation estimate
        """
        # Estimate total correlation using sample-based approximation
        batch_size, latent_dim = z.shape

        # Compute log densities
        log_qz = self._gaussian_log_density(z, z.mean(dim=0), z.var(dim=0))

        # Compute log density under factorized approximation
        log_qz_factorized = 0
        for i in range(latent_dim):
            log_qz_factorized += self._gaussian_log_density(
                z[:, i:i+1], z[:, i:i+1].mean(dim=0), z[:, i:i+1].var(dim=0)
            )

        # Total correlation approximation
        tc_loss = (log_qz - log_qz_factorized).mean()

        return tc_loss

    def _gaussian_log_density(self, x: torch.Tensor, mu: torch.Tensor,
                             var: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian log density."""
        return -0.5 * (torch.log(2 * math.pi * var) + (x - mu) ** 2 / var).sum(dim=-1)


def create_vae_loss(config: Dict, reconstruction_loss_fn: nn.Module) -> ELBOLoss:
    """
    Factory function to create VAE loss with configuration.

    Args:
        config: VAE loss configuration
        reconstruction_loss_fn: Reconstruction loss function

    Returns:
        Configured ELBO loss
    """
    beta = config['latent'].get('beta', 1.0)
    free_bits = config.get('free_bits', None)

    return ELBOLoss(
        reconstruction_loss_fn=reconstruction_loss_fn,
        beta=beta,
        free_bits=free_bits
    )


class VAELossAnalyzer:
    """Utility class for analyzing VAE loss behavior and diagnostics."""

    def __init__(self):
        """Initialize VAE loss analyzer."""
        pass

    def analyze_posterior_collapse(self, mu: torch.Tensor,
                                  logvar: torch.Tensor) -> Dict[str, float]:
        """
        Analyze signs of posterior collapse in VAE training.

        Args:
            mu: Posterior means
            logvar: Posterior log variances

        Returns:
            Dictionary of collapse indicators
        """
        with torch.no_grad():
            # Active units (dimensions with meaningful variance)
            posterior_var = torch.exp(logvar)
            active_units = (posterior_var.mean(dim=0) > 0.01).sum().item()

            # KL divergence per dimension
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)

            # Signs of collapse
            collapsed_dims = (kl_per_dim < 0.1).sum().item()
            mean_kl = kl_per_dim.mean().item()

            return {
                'active_units': active_units,
                'total_dims': mu.shape[1],
                'active_ratio': active_units / mu.shape[1],
                'collapsed_dims': collapsed_dims,
                'mean_kl_per_dim': mean_kl,
                'max_kl_per_dim': kl_per_dim.max().item(),
                'min_kl_per_dim': kl_per_dim.min().item()
            }

    def compute_reconstruction_vs_kl_tradeoff(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Analyze the reconstruction vs KL tradeoff.

        Args:
            losses: Dictionary of loss components

        Returns:
            Tradeoff analysis metrics
        """
        recon_loss = losses['reconstruction_loss']
        kl_loss = losses['kl_loss']

        # Relative magnitudes
        total_loss = recon_loss + kl_loss
        recon_ratio = recon_loss / total_loss
        kl_ratio = kl_loss / total_loss

        return {
            'reconstruction_ratio': recon_ratio.item(),
            'kl_ratio': kl_ratio.item(),
            'recon_kl_balance': (recon_loss / kl_loss).item(),
            'total_loss': total_loss.item()
        }