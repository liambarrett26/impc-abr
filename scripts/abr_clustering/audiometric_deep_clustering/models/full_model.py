"""
Complete ContrastiveVAE-DEC model for audiometric phenotype discovery.

Integrates all model components: encoder, decoder, VAE, clustering, and attention
mechanisms for end-to-end training and phenotype clustering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import logging

from .encoder import ContrastiveVAEEncoder
from .decoder import ContrastiveVAEDecoder
from .vae import LatentSpace, VAELoss
from .clustering_layer import ClusteringLayer
from .attention import FrequencyAttention

logger = logging.getLogger(__name__)


class ContrastiveVAEDEC(nn.Module):
    """
    Complete ContrastiveVAE-DEC model for audiometric phenotype clustering.

    Combines variational autoencoder with deep embedded clustering and
    contrastive learning for discovering novel hearing loss patterns.
    """

    def __init__(self, config: Dict):
        """
        Initialize the complete model.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.latent_dim = config['latent']['latent_dim']
        self.num_clusters = config['clustering']['num_clusters']

        # Core components
        self.encoder = ContrastiveVAEEncoder(config)
        self.decoder = ContrastiveVAEDecoder(config)
        self.latent_space = LatentSpace(config)
        self.clustering_layer = ClusteringLayer(config)

        # Loss components
        self.vae_loss = VAELoss(config)

        # Contrastive projection head
        self.contrastive_head = ContrastiveProjectionHead(
            input_dim=self.latent_dim,
            projection_dim=config['contrastive']['projection_dim'],
            temperature=config['contrastive']['temperature']
        )

        # Training stage tracking
        self.training_stage = 'pretrain'  # 'pretrain', 'cluster_init', 'joint'
        self.clusters_initialized = False
        
        # Memory optimization settings
        self.use_gradient_checkpointing = config.get('architecture', {}).get('gradient_checkpointing', False)

        logger.info(f"Initialized ContrastiveVAE-DEC: "
                   f"latent_dim={self.latent_dim}, clusters={self.num_clusters}")

    def forward(self, x: torch.Tensor,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.

        Args:
            x: Input features (batch_size, 18)
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing all model outputs
        """
        # Encode input
        encoder_output = self.encoder(x)

        # Extract latent variables
        mu = encoder_output['latent_mu']
        logvar = encoder_output['latent_logvar']
        z = encoder_output['latent_z']

        # Decode reconstruction
        decoder_output = self.decoder(z)
        reconstruction = decoder_output['reconstruction']

        # Clustering (if initialized)
        cluster_output = {}
        if self.clusters_initialized:
            cluster_output = self.clustering_layer(z)

        # Contrastive projection
        contrastive_features = self.contrastive_head.project(z)

        # Combine outputs
        output = {
            'reconstruction': reconstruction,
            'latent_mu': mu,
            'latent_logvar': logvar,
            'latent_z': z,
            'contrastive_features': contrastive_features,
            **decoder_output,
            **cluster_output
        }

        if return_attention and 'attention_weights' in encoder_output:
            output['attention_weights'] = encoder_output['attention_weights']

        return output

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (deterministic)."""
        return self.encoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder.decode(z)

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample from the learned prior distribution."""
        z = self.latent_space.sample_prior(num_samples, device)
        return self.decode(z)

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor,
                   num_steps: int = 10) -> torch.Tensor:
        """Interpolate between two inputs in latent space."""
        z1 = self.encode(x1)
        z2 = self.encode(x2)

        interpolated_z = self.latent_space.interpolate(z1, z2, num_steps)
        return self.decode(interpolated_z)

    def get_cluster_assignments(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cluster assignments for input data."""
        if not self.clusters_initialized:
            raise ValueError("Clusters must be initialized before getting assignments")

        z = self.encode(x)
        return self.clustering_layer.get_cluster_assignments(z)

    def initialize_clusters(self, dataloader, method: str = 'kmeans'):
        """Initialize cluster centers using provided data."""
        logger.info(f"Initializing clusters with {method}")

        # Collect latent representations
        latent_representations = []
        self.eval()

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x = batch['features']
                elif isinstance(batch, (list, tuple)):
                    x = batch[0]  # Extract tensor from tuple/list
                else:
                    x = batch

                # Move tensor to the same device as the model
                device = next(self.parameters()).device
                x = x.to(device)
                
                z = self.encode(x)
                latent_representations.append(z)

        # Concatenate all representations
        all_z = torch.cat(latent_representations, dim=0)

        # Initialize clustering layer
        self.clustering_layer.initialize_clusters(all_z, method=method)
        self.clusters_initialized = True

        logger.info("Cluster initialization completed")

    def set_training_stage(self, stage: str):
        """Set the current training stage."""
        valid_stages = ['pretrain', 'cluster_init', 'joint']
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage {stage}. Must be one of {valid_stages}")

        self.training_stage = stage
        logger.info(f"Training stage set to: {stage}")

    def compute_losses(self, batch: Dict[str, torch.Tensor],
                      return_individual: bool = False) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components based on current training stage.

        Args:
            batch: Input batch dictionary
            return_individual: Whether to return individual loss components

        Returns:
            Dictionary of computed losses
        """
        x = batch['features']
        output = self.forward(x)

        # VAE losses (always computed)
        vae_losses = self.vae_loss(
            x, output['reconstruction'],
            output['latent_mu'], output['latent_logvar']
        )

        losses = {
            'reconstruction_loss': vae_losses['reconstruction_loss'],
            'kl_loss': vae_losses['kl_loss'],
            'vae_loss': vae_losses['vae_loss']
        }

        # Stage-specific losses
        if self.training_stage == 'pretrain':
            total_loss = vae_losses['vae_loss']

        elif self.training_stage == 'joint':
            # Add clustering loss
            if self.clusters_initialized and 'q' in output:
                q = output['q']
                p = self.clustering_layer.compute_target_distribution(q)
                clustering_loss = self.clustering_layer.clustering_loss(q, p)
                losses['clustering_loss'] = clustering_loss
            else:
                clustering_loss = torch.tensor(0.0, device=x.device)
                losses['clustering_loss'] = clustering_loss

            # Add contrastive loss if positive pairs available - MEMORY OPTIMIZED
            contrastive_loss = torch.tensor(0.0, device=x.device)
            if 'positive' in batch:
                # Memory-efficient contrastive learning with gradient checkpointing
                if hasattr(self, 'use_gradient_checkpointing') and self.use_gradient_checkpointing:
                    # Use gradient checkpointing for memory efficiency
                    from torch.utils.checkpoint import checkpoint
                    positive_latent = checkpoint(self.encode, batch['positive'])
                else:
                    # Standard approach - encode positive samples to latent space only
                    positive_latent = self.encode(batch['positive'])
                
                batch['positive_latent'] = positive_latent
                
                # Pass latent_z instead of already-projected features
                contrastive_loss = self._compute_contrastive_loss(
                    output['latent_z'], batch
                )
                losses['contrastive_loss'] = contrastive_loss

            # Add phenotype consistency loss if gene labels available
            phenotype_loss = torch.tensor(0.0, device=x.device)
            if 'gene_label' in batch:
                phenotype_loss = self._compute_phenotype_consistency_loss(
                    output['latent_z'], batch['gene_label']
                )
                losses['phenotype_consistency_loss'] = phenotype_loss

            # Weighted total loss
            weights = self.config['loss_weights']
            total_loss = (
                weights['reconstruction'] * vae_losses['reconstruction_loss'] +
                weights['kl_divergence'] * vae_losses['kl_loss'] +
                weights['clustering'] * clustering_loss +
                weights['contrastive'] * contrastive_loss +
                weights.get('phenotype_consistency', 0.3) * phenotype_loss
            )

        else:  # cluster_init stage
            total_loss = vae_losses['vae_loss']

        losses['total_loss'] = total_loss

        if return_individual:
            losses.update(vae_losses)

        return losses

    def _compute_contrastive_loss(self, features: torch.Tensor,
                                 batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute contrastive loss using projection head."""
        return self.contrastive_head.compute_loss(features, batch)

    def _compute_phenotype_consistency_loss(self, latent_z: torch.Tensor,
                                          gene_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute phenotype consistency loss for same-gene samples.

        Encourages mice from the same gene knockout to have similar latent representations.
        """
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

    def analyze_model(self, dataloader) -> Dict[str, float]:
        """Comprehensive model analysis."""
        self.eval()

        all_latents = []
        all_reconstructions = []
        all_originals = []
        all_assignments = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    x = batch['features']
                elif isinstance(batch, (list, tuple)):
                    x = batch[0]  # Extract tensor from tuple/list
                else:
                    x = batch

                output = self.forward(x)

                all_latents.append(output['latent_z'])
                all_reconstructions.append(output['reconstruction'])
                all_originals.append(x)

                if 'q' in output:
                    all_assignments.append(output['q'])

        # Concatenate results
        latents = torch.cat(all_latents, dim=0)
        reconstructions = torch.cat(all_reconstructions, dim=0)
        originals = torch.cat(all_originals, dim=0)

        # Basic metrics
        recon_mse = F.mse_loss(reconstructions, originals)

        analysis = {
            'reconstruction_mse': recon_mse.item(),
            'latent_mean_norm': torch.norm(latents.mean(dim=0)).item(),
            'latent_std_mean': latents.std(dim=0).mean().item()
        }

        # Clustering metrics if available
        if all_assignments and self.clusters_initialized:
            assignments = torch.cat(all_assignments, dim=0)
            cluster_stats = self.clustering_layer.get_cluster_statistics(latents)

            analysis.update({
                'num_active_clusters': cluster_stats['cluster_sizes'].nonzero().size(0),
                'cluster_balance': cluster_stats['cluster_proportions'].min().item() / cluster_stats['cluster_proportions'].max().item(),
                'mean_assignment_entropy': cluster_stats['mean_entropy'],
                'silhouette_score': cluster_stats['silhouette_score'].item()
            })

        return analysis


class ContrastiveProjectionHead(nn.Module):
    """Projection head for contrastive learning."""

    def __init__(self, input_dim: int, projection_dim: int, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def project(self, features: torch.Tensor) -> torch.Tensor:
        """Project features to contrastive space."""
        return F.normalize(self.projection(features), dim=1)

    def compute_loss(self, features: torch.Tensor,
                    batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute InfoNCE contrastive loss.
        
        Args:
            features: Latent features (batch_size, latent_dim)
            batch: Dictionary containing positive pairs
        """
        if 'positive' not in batch:
            return torch.tensor(0.0, device=features.device)

        # Project anchor features (latent z)
        anchor_proj = self.project(features)
        
        # Get positive features - these are raw features that need encoding
        positive_features = batch['positive']
        
        # Positive features need to be encoded to latent space first
        # We'll need access to the encoder for this
        # For now, we'll assume positive latent features are provided
        if 'positive_latent' in batch:
            positive_latent = batch['positive_latent']
            positive_proj = self.project(positive_latent)
        else:
            # If positive features have same dim as anchor, assume they're already latent
            if positive_features.shape[-1] == features.shape[-1]:
                positive_proj = self.project(positive_features)
            else:
                # This shouldn't happen with proper data loading
                raise ValueError(f"Positive features have wrong shape: {positive_features.shape} vs {features.shape}")

        # Compute similarities
        pos_sim = torch.sum(anchor_proj * positive_proj, dim=1) / self.temperature

        # Create negative pairs (all other samples in batch)
        batch_size = anchor_proj.size(0)
        neg_sim = torch.mm(anchor_proj, anchor_proj.t()) / self.temperature

        # Mask out self-similarities
        mask = torch.eye(batch_size, device=anchor_proj.device, dtype=torch.bool)
        neg_sim = neg_sim.masked_fill(mask, float('-inf'))

        # Compute InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor_proj.device)

        return F.cross_entropy(logits, labels)


def create_model(config: Dict) -> ContrastiveVAEDEC:
    """Factory function to create the complete model."""
    return ContrastiveVAEDEC(config)