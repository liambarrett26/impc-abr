"""
Combined multi-objective loss for ContrastiveVAE-DEC model.

This module orchestrates all loss components and provides stage-aware
training with adaptive weighting for audiometric phenotype discovery.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Any
import logging

from .reconstruction import create_reconstruction_loss
from .vae_loss import create_vae_loss, BetaScheduler
from .clustering_loss import create_clustering_loss
from .contrastive import create_contrastive_loss

logger = logging.getLogger(__name__)


class MultiObjectiveLoss(nn.Module):
    """
    Multi-objective loss combining all training objectives.

    Orchestrates reconstruction, VAE, clustering, and contrastive losses
    with stage-aware weighting and adaptive scheduling.
    """

    def __init__(self, config: Dict):
        """
        Initialize multi-objective loss.

        Args:
            config: Complete model configuration
        """
        super().__init__()
        self.config = config
        self.loss_weights = config['loss_weights']

        # Initialize component losses
        self.reconstruction_loss = create_reconstruction_loss(config)
        self.vae_loss = create_vae_loss(config, self.reconstruction_loss)
        self.clustering_loss = create_clustering_loss(config)
        self.contrastive_loss = create_contrastive_loss(config)

        # Training stage management
        self.training_stage = 'pretrain'  # 'pretrain', 'cluster_init', 'joint'
        self.current_epoch = 0

        # Adaptive scheduling
        self.beta_scheduler = BetaScheduler(
            beta_max=config['latent']['beta'],
            warmup_epochs=config.get('beta_warmup_epochs', 10),
            schedule_type=config.get('beta_schedule', 'linear')
        )

        # Loss history for analysis
        self.loss_history = {
            'reconstruction': [],
            'kl': [],
            'clustering': [],
            'contrastive': [],
            'total': []
        }

        logger.info("Initialized multi-objective loss with all components")

    def forward(self, model_output: Dict[str, torch.Tensor],
                batch: Dict[str, torch.Tensor],
                epoch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss based on current training stage.

        Args:
            model_output: Output from ContrastiveVAE-DEC model
            batch: Input batch dictionary
            epoch: Current training epoch

        Returns:
            Dictionary of all loss components
        """
        if epoch is not None:
            self.current_epoch = epoch

        # Extract required components
        features = batch['features']
        reconstruction = model_output['reconstruction']
        latent_mu = model_output['latent_mu']
        latent_logvar = model_output['latent_logvar']
        latent_z = model_output['latent_z']

        # Initialize loss dictionary
        losses = {}

        # Stage-specific loss computation
        if self.training_stage == 'pretrain':
            losses = self._compute_pretrain_losses(
                features, reconstruction, latent_mu, latent_logvar, epoch
            )

        elif self.training_stage == 'cluster_init':
            losses = self._compute_cluster_init_losses(
                features, reconstruction, latent_mu, latent_logvar, epoch
            )

        elif self.training_stage == 'joint':
            losses = self._compute_joint_losses(
                model_output, batch, epoch
            )

        # Update loss history
        self._update_loss_history(losses)

        return losses

    def _compute_pretrain_losses(self, features: torch.Tensor,
                                reconstruction: torch.Tensor,
                                latent_mu: torch.Tensor,
                                latent_logvar: torch.Tensor,
                                epoch: Optional[int]) -> Dict[str, torch.Tensor]:
        """Compute losses for pretraining stage (VAE only)."""
        # Update beta scheduling
        current_beta = self.beta_scheduler.get_beta(epoch)
        self.vae_loss.update_beta(current_beta)

        # VAE loss (reconstruction + KL)
        vae_losses = self.vae_loss(
            reconstruction, features, latent_mu, latent_logvar, epoch
        )

        # Total loss is just ELBO
        total_loss = vae_losses['elbo_loss']

        return {
            'total_loss': total_loss,
            'reconstruction_loss': vae_losses['reconstruction_loss'],
            'kl_loss': vae_losses['kl_loss'],
            'beta_kl_loss': vae_losses['beta_kl_loss'],
            'elbo_loss': vae_losses['elbo_loss'],
            'current_beta': current_beta,
            'stage': 'pretrain',
            **vae_losses
        }

    def _compute_cluster_init_losses(self, features: torch.Tensor,
                                   reconstruction: torch.Tensor,
                                   latent_mu: torch.Tensor,
                                   latent_logvar: torch.Tensor,
                                   epoch: Optional[int]) -> Dict[str, torch.Tensor]:
        """Compute losses for cluster initialization stage."""
        # Similar to pretraining but may have different beta schedule
        current_beta = self.beta_scheduler.get_beta(epoch)
        self.vae_loss.update_beta(current_beta)

        vae_losses = self.vae_loss(
            reconstruction, features, latent_mu, latent_logvar, epoch
        )

        total_loss = vae_losses['elbo_loss']

        return {
            'total_loss': total_loss,
            'reconstruction_loss': vae_losses['reconstruction_loss'],
            'kl_loss': vae_losses['kl_loss'],
            'beta_kl_loss': vae_losses['beta_kl_loss'],
            'elbo_loss': vae_losses['elbo_loss'],
            'current_beta': current_beta,
            'stage': 'cluster_init',
            **vae_losses
        }

    def _compute_joint_losses(self, model_output: Dict[str, torch.Tensor],
                             batch: Dict[str, torch.Tensor],
                             epoch: Optional[int]) -> Dict[str, torch.Tensor]:
        """Compute losses for joint training stage (all objectives)."""
        features = batch['features']
        reconstruction = model_output['reconstruction']
        latent_mu = model_output['latent_mu']
        latent_logvar = model_output['latent_logvar']
        latent_z = model_output['latent_z']

        # VAE losses
        current_beta = self.beta_scheduler.get_beta(epoch)
        self.vae_loss.update_beta(current_beta)

        vae_losses = self.vae_loss(
            reconstruction, features, latent_mu, latent_logvar, epoch
        )

        # Clustering losses (if cluster assignments available)
        clustering_losses = {}
        if 'q' in model_output and 'cluster_centers' in model_output:
            q = model_output['q']
            cluster_centers = model_output['cluster_centers']

            # Compute target distribution
            p = self._compute_target_distribution(q)

            clustering_losses = self.clustering_loss(
                latent_z, q, p, cluster_centers, features[:, :6],  # ABR features
                batch.get('gene_label'), epoch
            )

        # Contrastive losses (if positive pairs available)
        contrastive_losses = {}
        if 'contrastive_features' in model_output:
            contrastive_features = model_output['contrastive_features']

            # Use augmented features as positives if available
            positive_features = batch.get('positive', batch.get('augmented'))

            if positive_features is not None:
                # Project positive features if needed
                if hasattr(self.contrastive_loss, 'infonce_loss'):
                    # Assume model has contrastive projection head
                    positive_projected = contrastive_features  # Already projected
                else:
                    positive_projected = positive_features

                contrastive_losses = self.contrastive_loss(
                    contrastive_features, positive_projected,
                    features[:, :6],  # ABR features
                    batch.get('gene_label'), epoch
                )

        # Apply loss weights
        weights = self.loss_weights
        weighted_reconstruction = weights['reconstruction'] * vae_losses['reconstruction_loss']
        weighted_kl = weights['kl_divergence'] * vae_losses['kl_loss']
        weighted_clustering = weights['clustering'] * clustering_losses.get('total_clustering_loss', torch.tensor(0.0))
        weighted_contrastive = weights['contrastive'] * contrastive_losses.get('total_contrastive_loss', torch.tensor(0.0))

        # Additional regularization terms
        phenotype_consistency_loss = torch.tensor(0.0, device=features.device)
        if 'gene_label' in batch:
            phenotype_consistency_loss = self._compute_phenotype_consistency(
                latent_z, batch['gene_label']
            )

        weighted_phenotype = weights.get('phenotype_consistency', 0.3) * phenotype_consistency_loss

        # Frequency smoothness regularization for ABR features
        frequency_smoothness_loss = self._compute_frequency_smoothness(
            features[:, :6], reconstruction[:, :6]
        )
        weighted_smoothness = weights.get('frequency_smoothness', 0.1) * frequency_smoothness_loss

        # Total combined loss
        total_loss = (
            weighted_reconstruction +
            weighted_kl +
            weighted_clustering +
            weighted_contrastive +
            weighted_phenotype +
            weighted_smoothness
        )

        # Combine all loss components
        combined_losses = {
            'total_loss': total_loss,
            'reconstruction_loss': vae_losses['reconstruction_loss'],
            'kl_loss': vae_losses['kl_loss'],
            'clustering_loss': clustering_losses.get('total_clustering_loss', torch.tensor(0.0)),
            'contrastive_loss': contrastive_losses.get('total_contrastive_loss', torch.tensor(0.0)),
            'phenotype_consistency_loss': phenotype_consistency_loss,
            'frequency_smoothness_loss': frequency_smoothness_loss,
            'weighted_reconstruction': weighted_reconstruction,
            'weighted_kl': weighted_kl,
            'weighted_clustering': weighted_clustering,
            'weighted_contrastive': weighted_contrastive,
            'weighted_phenotype': weighted_phenotype,
            'weighted_smoothness': weighted_smoothness,
            'current_beta': current_beta,
            'stage': 'joint',
            **vae_losses,
            **{f'clustering_{k}': v for k, v in clustering_losses.items()},
            **{f'contrastive_{k}': v for k, v in contrastive_losses.items()}
        }

        return combined_losses

    def _compute_target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """Compute target distribution for DEC clustering."""
        # Frequency of assignment to each cluster
        f_j = torch.sum(q, dim=0)

        # Compute p_ij = q_ij^2 / f_j / Σ_k (q_ik^2 / f_k)
        numerator = q ** 2 / f_j.unsqueeze(0)
        denominator = torch.sum(numerator, dim=1, keepdim=True)

        p = numerator / (denominator + 1e-8)

        return p

    def _compute_phenotype_consistency(self, latent_z: torch.Tensor,
                                     gene_labels: torch.Tensor) -> torch.Tensor:
        """Encourage similar latent representations for same gene."""
        unique_genes = torch.unique(gene_labels)
        consistency_loss = torch.tensor(0.0, device=latent_z.device)

        for gene in unique_genes:
            if gene == -1:  # Skip unknown genes
                continue

            gene_mask = gene_labels == gene
            gene_latents = latent_z[gene_mask]

            if len(gene_latents) > 1:
                # Compute pairwise distances within gene group
                gene_mean = torch.mean(gene_latents, dim=0, keepdim=True)
                distances = torch.norm(gene_latents - gene_mean, dim=1)
                consistency_loss += torch.mean(distances)

        return consistency_loss / len(unique_genes) if len(unique_genes) > 0 else consistency_loss

    def _compute_frequency_smoothness(self, original_abr: torch.Tensor,
                                    reconstructed_abr: torch.Tensor) -> torch.Tensor:
        """Encourage smooth frequency patterns in ABR reconstruction."""
        # Compute gradients across frequencies
        orig_grad = torch.diff(original_abr, dim=1)
        recon_grad = torch.diff(reconstructed_abr, dim=1)

        # MSE loss on gradients (frequency smoothness)
        smoothness_loss = nn.functional.mse_loss(recon_grad, orig_grad)

        return smoothness_loss

    def _update_loss_history(self, losses: Dict[str, torch.Tensor]):
        """Update loss history for monitoring."""
        if 'reconstruction_loss' in losses:
            self.loss_history['reconstruction'].append(losses['reconstruction_loss'].item())
        if 'kl_loss' in losses:
            self.loss_history['kl'].append(losses['kl_loss'].item())
        if 'clustering_loss' in losses:
            self.loss_history['clustering'].append(losses['clustering_loss'].item())
        if 'contrastive_loss' in losses:
            self.loss_history['contrastive'].append(losses['contrastive_loss'].item())
        if 'total_loss' in losses:
            self.loss_history['total'].append(losses['total_loss'].item())

    def set_training_stage(self, stage: str):
        """Set the current training stage."""
        valid_stages = ['pretrain', 'cluster_init', 'joint']
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage {stage}. Must be one of {valid_stages}")

        self.training_stage = stage
        logger.info(f"Training stage set to: {stage}")

    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return self.loss_weights.copy()

    def update_loss_weights(self, new_weights: Dict[str, float]):
        """Update loss weights during training."""
        self.loss_weights.update(new_weights)
        logger.info(f"Updated loss weights: {new_weights}")

    def get_loss_history(self) -> Dict[str, List[float]]:
        """Get loss history for analysis."""
        return self.loss_history.copy()


class LossScheduler:
    """
    Scheduler for dynamic loss weight adjustment during training.

    Provides curriculum learning and adaptive weighting strategies.
    """

    def __init__(self, config: Dict):
        """Initialize loss scheduler."""
        self.config = config
        self.initial_weights = config['loss_weights'].copy()
        self.current_epoch = 0

        # Scheduling parameters
        self.schedule_type = config.get('loss_schedule', 'static')
        self.warmup_epochs = config.get('loss_warmup_epochs', 50)

    def get_weights(self, epoch: int, training_stage: str) -> Dict[str, float]:
        """
        Get loss weights for current epoch and stage.

        Args:
            epoch: Current training epoch
            training_stage: Current training stage

        Returns:
            Dictionary of loss weights
        """
        if self.schedule_type == 'static':
            return self.initial_weights

        elif self.schedule_type == 'curriculum':
            return self._curriculum_weights(epoch, training_stage)

        elif self.schedule_type == 'adaptive':
            return self._adaptive_weights(epoch, training_stage)

        else:
            return self.initial_weights

    def _curriculum_weights(self, epoch: int, stage: str) -> Dict[str, float]:
        """Curriculum learning weight schedule."""
        weights = self.initial_weights.copy()

        if stage == 'joint':
            progress = min(epoch / self.warmup_epochs, 1.0)

            # Gradually increase complex loss weights
            weights['clustering'] *= progress
            weights['contrastive'] *= progress
            weights['phenotype_consistency'] *= progress

        return weights

    def _adaptive_weights(self, epoch: int, stage: str) -> Dict[str, float]:
        """Adaptive weight adjustment based on loss magnitudes."""
        # This would require access to recent loss values
        # Implementation would adapt weights based on relative loss scales
        return self.initial_weights


def create_combined_loss(config: Dict) -> MultiObjectiveLoss:
    """
    Factory function to create combined multi-objective loss.

    Args:
        config: Complete model configuration

    Returns:
        Configured multi-objective loss
    """
    return MultiObjectiveLoss(config)


class LossAnalyzer:
    """Comprehensive loss analysis and monitoring utilities."""

    def __init__(self):
        """Initialize loss analyzer."""
        pass

    def analyze_loss_balance(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Analyze balance between different loss components.

        Args:
            losses: Dictionary of loss components

        Returns:
            Balance analysis metrics
        """
        # Extract main loss components
        reconstruction = losses.get('reconstruction_loss', torch.tensor(0.0))
        kl = losses.get('kl_loss', torch.tensor(0.0))
        clustering = losses.get('clustering_loss', torch.tensor(0.0))
        contrastive = losses.get('contrastive_loss', torch.tensor(0.0))

        total = reconstruction + kl + clustering + contrastive

        if total > 0:
            return {
                'reconstruction_ratio': (reconstruction / total).item(),
                'kl_ratio': (kl / total).item(),
                'clustering_ratio': (clustering / total).item(),
                'contrastive_ratio': (contrastive / total).item(),
                'dominant_loss': self._get_dominant_loss(losses),
                'loss_entropy': self._compute_loss_entropy(losses)
            }
        else:
            return {'error': 'All losses are zero'}

    def _get_dominant_loss(self, losses: Dict[str, torch.Tensor]) -> str:
        """Identify the dominant loss component."""
        main_losses = {
            'reconstruction': losses.get('reconstruction_loss', torch.tensor(0.0)),
            'kl': losses.get('kl_loss', torch.tensor(0.0)),
            'clustering': losses.get('clustering_loss', torch.tensor(0.0)),
            'contrastive': losses.get('contrastive_loss', torch.tensor(0.0))
        }

        return max(main_losses.items(), key=lambda x: x[1].item())[0]

    def _compute_loss_entropy(self, losses: Dict[str, torch.Tensor]) -> float:
        """Compute entropy of loss distribution (measure of balance)."""
        main_losses = [
            losses.get('reconstruction_loss', torch.tensor(0.0)),
            losses.get('kl_loss', torch.tensor(0.0)),
            losses.get('clustering_loss', torch.tensor(0.0)),
            losses.get('contrastive_loss', torch.tensor(0.0))
        ]

        loss_values = torch.stack([loss.item() for loss in main_losses])
        total = loss_values.sum()

        if total > 0:
            probs = loss_values / total
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            return entropy.item()
        else:
            return 0.0