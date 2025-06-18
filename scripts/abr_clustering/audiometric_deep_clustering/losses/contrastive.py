"""
Contrastive learning losses for ContrastiveVAE-DEC model.

This module provides InfoNCE and other contrastive losses for learning
representations that group similar audiometric phenotypes together.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for representation learning.

    Encourages similar representations for positive pairs (same gene)
    while pushing apart negative pairs (different genes).
    """

    def __init__(self, temperature: float = 0.5, negative_mode: str = 'unpaired'):
        """
        Initialize InfoNCE loss.

        Args:
            temperature: Temperature parameter for softmax
            negative_mode: How to construct negatives ('unpaired', 'hard')
        """
        super().__init__()
        self.temperature = temperature
        self.negative_mode = negative_mode

    def forward(self, anchor_features: torch.Tensor,
                positive_features: torch.Tensor,
                negative_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute InfoNCE loss.

        Args:
            anchor_features: Anchor representations (batch_size, feature_dim)
            positive_features: Positive pair representations (batch_size, feature_dim)
            negative_features: Explicit negative representations (optional)

        Returns:
            Dictionary of contrastive loss components
        """
        batch_size = anchor_features.shape[0]

        # Normalize features
        anchor_norm = F.normalize(anchor_features, dim=1)
        positive_norm = F.normalize(positive_features, dim=1)

        # Positive similarities
        positive_sim = torch.sum(anchor_norm * positive_norm, dim=1) / self.temperature

        # Negative similarities
        if negative_features is not None:
            negative_norm = F.normalize(negative_features, dim=1)
            negative_sim = torch.mm(anchor_norm, negative_norm.t()) / self.temperature
        else:
            # Use all other samples in batch as negatives
            negative_sim = torch.mm(anchor_norm, anchor_norm.t()) / self.temperature
            # Mask out self-similarities and positive pairs
            mask = torch.eye(batch_size, device=anchor_features.device, dtype=torch.bool)
            negative_sim = negative_sim.masked_fill(mask, float('-inf'))

        # Combine positive and negative similarities
        if negative_features is not None:
            logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)
        else:
            logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)

        # Labels (positive is always first)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor_features.device)

        # InfoNCE loss
        infonce_loss = F.cross_entropy(logits, labels)

        # Additional metrics
        positive_sim_mean = positive_sim.mean()
        if negative_features is not None:
            negative_sim_mean = negative_sim.mean()
        else:
            negative_sim_mean = negative_sim[~mask.unsqueeze(1).expand_as(negative_sim)].mean()

        return {
            'infonce_loss': infonce_loss,
            'positive_similarity': positive_sim_mean,
            'negative_similarity': negative_sim_mean,
            'similarity_gap': positive_sim_mean - negative_sim_mean,
            'logits': logits,
            'labels': labels
        }


class PhenotypeSimilarityLoss(nn.Module):
    """
    Phenotype-aware similarity loss for audiometric data.

    Uses ABR pattern similarity to define positive and negative pairs
    beyond just gene labels.
    """

    def __init__(self, similarity_threshold: float = 0.8, temperature: float = 0.5):
        """
        Initialize phenotype similarity loss.

        Args:
            similarity_threshold: Threshold for considering samples similar
            temperature: Temperature for contrastive loss
        """
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.temperature = temperature

    def forward(self, latent_features: torch.Tensor,
                abr_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute phenotype-aware contrastive loss.

        Args:
            latent_features: Latent representations (batch_size, latent_dim)
            abr_features: Original ABR features (batch_size, 6)

        Returns:
            Dictionary of phenotype similarity losses
        """
        batch_size = latent_features.shape[0]

        # Compute ABR similarity matrix
        abr_normalized = F.normalize(abr_features, dim=1)
        abr_similarity = torch.mm(abr_normalized, abr_normalized.t())

        # Create positive/negative masks based on ABR similarity
        positive_mask = (abr_similarity > self.similarity_threshold) & \
                       (~torch.eye(batch_size, device=latent_features.device, dtype=torch.bool))

        # Compute latent feature similarities
        latent_normalized = F.normalize(latent_features, dim=1)
        latent_similarity = torch.mm(latent_normalized, latent_normalized.t()) / self.temperature

        # Contrastive loss: pull similar ABR patterns together in latent space
        positive_pairs = latent_similarity[positive_mask]

        if len(positive_pairs) > 0:
            # For each positive pair, contrast against all negatives
            phenotype_loss = 0.0
            num_positive_pairs = 0

            for i in range(batch_size):
                positive_indices = positive_mask[i].nonzero(as_tuple=True)[0]

                if len(positive_indices) > 0:
                    anchor = latent_normalized[i:i+1]
                    positives = latent_normalized[positive_indices]

                    # Positive similarities
                    pos_sim = torch.mm(anchor, positives.t()) / self.temperature

                    # All similarities (including negatives)
                    all_sim = latent_similarity[i:i+1]

                    # Remove self-similarity
                    all_sim = torch.cat([all_sim[:, :i], all_sim[:, i+1:]], dim=1)

                    # InfoNCE-style loss for each positive
                    for j, pos_sim_val in enumerate(pos_sim[0]):
                        logits = torch.cat([pos_sim_val.unsqueeze(0), all_sim[0]])
                        labels = torch.zeros(1, dtype=torch.long, device=latent_features.device)
                        phenotype_loss += F.cross_entropy(logits.unsqueeze(0), labels)
                        num_positive_pairs += 1

            if num_positive_pairs > 0:
                phenotype_loss /= num_positive_pairs
        else:
            phenotype_loss = torch.tensor(0.0, device=latent_features.device)

        # Additional metrics
        mean_abr_similarity = abr_similarity[positive_mask].mean() if len(positive_pairs) > 0 else torch.tensor(0.0)
        num_positive_pairs_found = positive_mask.sum().item()

        return {
            'phenotype_similarity_loss': phenotype_loss,
            'mean_abr_similarity': mean_abr_similarity,
            'num_positive_pairs': num_positive_pairs_found,
            'positive_pair_ratio': num_positive_pairs_found / (batch_size * (batch_size - 1))
        }


class GeneAwareContrastive(nn.Module):
    """
    Gene-aware contrastive learning for audiometric phenotypes.

    Uses gene labels to create positive pairs while considering
    phenotypic variation within genes.
    """

    def __init__(self, temperature: float = 0.5, within_gene_weight: float = 1.0,
                 cross_gene_weight: float = 0.5):
        """
        Initialize gene-aware contrastive loss.

        Args:
            temperature: Temperature parameter
            within_gene_weight: Weight for within-gene similarity
            cross_gene_weight: Weight for cross-gene dissimilarity
        """
        super().__init__()
        self.temperature = temperature
        self.within_gene_weight = within_gene_weight
        self.cross_gene_weight = cross_gene_weight

    def forward(self, features: torch.Tensor,
                gene_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute gene-aware contrastive loss.

        Args:
            features: Feature representations (batch_size, feature_dim)
            gene_labels: Gene labels (batch_size,)

        Returns:
            Dictionary of gene-aware contrastive losses
        """
        batch_size = features.shape[0]

        # Normalize features
        features_norm = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(features_norm, features_norm.t()) / self.temperature

        # Create gene similarity matrix
        gene_similarity = (gene_labels.unsqueeze(0) == gene_labels.unsqueeze(1)).float()

        # Mask out self-similarities
        mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)
        gene_similarity = gene_similarity.masked_fill(mask, 0)

        # Within-gene contrastive loss
        within_gene_pairs = gene_similarity.sum()
        if within_gene_pairs > 0:
            positive_sim = similarity_matrix[gene_similarity.bool()]

            # For each gene, compute contrastive loss
            within_gene_loss = 0.0
            unique_genes = torch.unique(gene_labels)

            for gene in unique_genes:
                if gene == -1:  # Skip unknown genes
                    continue

                gene_mask = gene_labels == gene
                gene_indices = gene_mask.nonzero(as_tuple=True)[0]

                if len(gene_indices) > 1:
                    # All pairs within this gene
                    gene_features = features_norm[gene_indices]
                    gene_sim = torch.mm(gene_features, gene_features.t()) / self.temperature

                    # Mask diagonal
                    gene_mask_matrix = ~torch.eye(len(gene_indices), device=features.device, dtype=torch.bool)

                    # Positive similarities within gene
                    pos_similarities = gene_sim[gene_mask_matrix]

                    # Negative similarities (all other samples)
                    other_indices = (~gene_mask).nonzero(as_tuple=True)[0]
                    if len(other_indices) > 0:
                        other_features = features_norm[other_indices]
                        neg_similarities = torch.mm(gene_features, other_features.t()) / self.temperature

                        # Contrastive loss for this gene
                        for i in range(len(gene_indices)):
                            for j in range(i + 1, len(gene_indices)):
                                pos_sim = gene_sim[i, j]
                                neg_sim = neg_similarities[i]

                                logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])
                                labels = torch.zeros(1, dtype=torch.long, device=features.device)
                                within_gene_loss += F.cross_entropy(logits.unsqueeze(0), labels)
        else:
            within_gene_loss = torch.tensor(0.0, device=features.device)

        # Cross-gene separation loss
        cross_gene_mask = ~gene_similarity.bool() & ~mask
        cross_gene_loss = torch.tensor(0.0, device=features.device)

        if cross_gene_mask.sum() > 0:
            # Penalize high similarity between different genes
            cross_gene_similarities = similarity_matrix[cross_gene_mask]
            # Use margin loss: max(0, sim - margin)
            margin = 0.1
            cross_gene_loss = F.relu(cross_gene_similarities - margin).mean()

        # Combined loss
        total_loss = (self.within_gene_weight * within_gene_loss +
                     self.cross_gene_weight * cross_gene_loss)

        return {
            'gene_contrastive_loss': total_loss,
            'within_gene_loss': within_gene_loss,
            'cross_gene_loss': cross_gene_loss,
            'num_within_gene_pairs': within_gene_pairs,
            'num_cross_gene_pairs': cross_gene_mask.sum().item()
        }


class AdaptiveContrastiveLoss(nn.Module):
    """
    Adaptive contrastive loss that adjusts strategy based on training progress.

    Starts with simple InfoNCE and gradually incorporates phenotype-aware
    and gene-aware objectives.
    """

    def __init__(self, config: Dict):
        """Initialize adaptive contrastive loss."""
        super().__init__()
        self.config = config
        self.temperature = config['contrastive']['temperature']

        # Component losses
        self.infonce_loss = InfoNCELoss(temperature=self.temperature)
        self.phenotype_loss = PhenotypeSimilarityLoss(
            similarity_threshold=0.8,
            temperature=self.temperature
        )
        self.gene_loss = GeneAwareContrastive(temperature=self.temperature)

        # Adaptation parameters
        self.warmup_epochs = config.get('contrastive_warmup_epochs', 50)
        self.current_epoch = 0

    def forward(self, anchor_features: torch.Tensor,
                positive_features: torch.Tensor,
                abr_features: torch.Tensor,
                gene_labels: Optional[torch.Tensor] = None,
                epoch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive contrastive loss.

        Args:
            anchor_features: Anchor representations
            positive_features: Positive pair representations
            abr_features: Original ABR features
            gene_labels: Gene labels (optional)
            epoch: Current epoch

        Returns:
            Dictionary of adaptive contrastive losses
        """
        if epoch is not None:
            self.current_epoch = epoch

        # Core InfoNCE loss (always active)
        infonce_losses = self.infonce_loss(anchor_features, positive_features)

        # Phenotype-aware loss
        phenotype_losses = self.phenotype_loss(anchor_features, abr_features)

        # Gene-aware loss (if gene labels available)
        gene_losses = {}
        if gene_labels is not None:
            gene_losses = self.gene_loss(anchor_features, gene_labels)

        # Adaptive weighting based on training progress
        progress = min(self.current_epoch / self.warmup_epochs, 1.0)

        # Progressive complexity introduction
        infonce_weight = 1.0
        phenotype_weight = 0.3 * progress
        gene_weight = 0.2 * progress if gene_labels is not None else 0.0

        # Combined contrastive loss
        total_contrastive_loss = infonce_weight * infonce_losses['infonce_loss']

        if 'phenotype_similarity_loss' in phenotype_losses:
            total_contrastive_loss += phenotype_weight * phenotype_losses['phenotype_similarity_loss']

        if gene_losses and 'gene_contrastive_loss' in gene_losses:
            total_contrastive_loss += gene_weight * gene_losses['gene_contrastive_loss']

        # Combine all components
        combined_losses = {
            'total_contrastive_loss': total_contrastive_loss,
            'infonce_component': infonce_losses['infonce_loss'],
            'phenotype_component': phenotype_losses.get('phenotype_similarity_loss', torch.tensor(0.0)),
            'gene_component': gene_losses.get('gene_contrastive_loss', torch.tensor(0.0)),
            'infonce_weight': infonce_weight,
            'phenotype_weight': phenotype_weight,
            'gene_weight': gene_weight,
            **{f'infonce_{k}': v for k, v in infonce_losses.items()},
            **{f'phenotype_{k}': v for k, v in phenotype_losses.items()},
            **{f'gene_{k}': v for k, v in gene_losses.items()}
        }

        return combined_losses

    def update_epoch(self, epoch: int):
        """Update current epoch for adaptive weighting."""
        self.current_epoch = epoch


def create_contrastive_loss(config: Dict) -> AdaptiveContrastiveLoss:
    """
    Factory function to create contrastive loss.

    Args:
        config: Contrastive loss configuration

    Returns:
        Configured contrastive loss module
    """
    return AdaptiveContrastiveLoss(config)


class ContrastiveLossAnalyzer:
    """Utility class for analyzing contrastive loss behavior."""

    def __init__(self):
        """Initialize contrastive loss analyzer."""
        pass

    def analyze_representation_quality(self, features: torch.Tensor,
                                     gene_labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Analyze quality of learned representations.

        Args:
            features: Feature representations
            gene_labels: Gene labels for analysis

        Returns:
            Dictionary of representation quality metrics
        """
        with torch.no_grad():
            # Feature statistics
            feature_norm = torch.norm(features, dim=1).mean()
            feature_std = features.std(dim=0).mean()

            # Pairwise similarities
            features_norm = F.normalize(features, dim=1)
            similarity_matrix = torch.mm(features_norm, features_norm.t())

            # Remove diagonal
            mask = ~torch.eye(len(features), dtype=torch.bool, device=features.device)
            similarities = similarity_matrix[mask]

            metrics = {
                'mean_feature_norm': feature_norm.item(),
                'mean_feature_std': feature_std.item(),
                'mean_pairwise_similarity': similarities.mean().item(),
                'similarity_std': similarities.std().item(),
                'max_similarity': similarities.max().item(),
                'min_similarity': similarities.min().item()
            }

            # Gene-specific analysis if available
            if gene_labels is not None:
                within_gene_sims = []
                between_gene_sims = []

                for i in range(len(features)):
                    for j in range(i + 1, len(features)):
                        sim = similarity_matrix[i, j].item()

                        if gene_labels[i] == gene_labels[j] and gene_labels[i] != -1:
                            within_gene_sims.append(sim)
                        elif gene_labels[i] != gene_labels[j] and gene_labels[i] != -1 and gene_labels[j] != -1:
                            between_gene_sims.append(sim)

                if within_gene_sims and between_gene_sims:
                    metrics.update({
                        'within_gene_similarity': sum(within_gene_sims) / len(within_gene_sims),
                        'between_gene_similarity': sum(between_gene_sims) / len(between_gene_sims),
                        'gene_separation': (sum(within_gene_sims) / len(within_gene_sims)) -
                                         (sum(between_gene_sims) / len(between_gene_sims))
                    })

            return metrics