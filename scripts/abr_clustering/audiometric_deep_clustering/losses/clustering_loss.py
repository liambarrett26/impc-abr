"""
Clustering losses for ContrastiveVAE-DEC model.

This module provides DEC clustering loss, auxiliary clustering objectives,
and phenotype-aware clustering penalties for audiometric data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class DECLoss(nn.Module):
    """
    Deep Embedded Clustering (DEC) loss function.

    Implements the core DEC objective that minimizes KL divergence between
    current cluster assignments (Q) and target distribution (P).
    """

    def __init__(self, alpha: float = 1.0, cluster_update_interval: int = 50):
        """
        Initialize DEC loss.

        Args:
            alpha: Degrees of freedom parameter for Student's t-distribution
            cluster_update_interval: How often to update target distribution
        """
        super().__init__()
        self.alpha = alpha
        self.cluster_update_interval = cluster_update_interval
        self.update_counter = 0

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute DEC clustering loss.

        Args:
            q: Current soft cluster assignments (batch_size, num_clusters)
            p: Target distribution (batch_size, num_clusters)

        Returns:
            Dictionary of clustering loss components
        """
        # KL divergence: KL(P || Q) = Σ p_ij * log(p_ij / q_ij)
        kl_div = torch.sum(p * torch.log(p / (q + 1e-8) + 1e-8), dim=1)
        dec_loss = kl_div.mean()

        # Additional metrics for monitoring
        entropy_q = -torch.sum(q * torch.log(q + 1e-8), dim=1).mean()
        entropy_p = -torch.sum(p * torch.log(p + 1e-8), dim=1).mean()

        # Cluster confidence (how peaked are the assignments)
        max_assignment = torch.max(q, dim=1)[0].mean()

        return {
            'dec_loss': dec_loss,
            'kl_divergence': kl_div,
            'entropy_q': entropy_q,
            'entropy_p': entropy_p,
            'cluster_confidence': max_assignment
        }


class AuxiliaryClustering(nn.Module):
    """
    Auxiliary clustering objectives to improve cluster quality.

    Provides additional clustering penalties including cluster balance,
    separation, and compactness objectives.
    """

    def __init__(self, num_clusters: int, balance_weight: float = 0.1,
                 separation_weight: float = 0.1, compactness_weight: float = 0.1):
        """
        Initialize auxiliary clustering loss.

        Args:
            num_clusters: Number of clusters
            balance_weight: Weight for cluster balance loss
            separation_weight: Weight for cluster separation loss
            compactness_weight: Weight for within-cluster compactness loss
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.balance_weight = balance_weight
        self.separation_weight = separation_weight
        self.compactness_weight = compactness_weight

    def forward(self, latent_z: torch.Tensor, cluster_assignments: torch.Tensor,
                cluster_centers: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary clustering losses.

        Args:
            latent_z: Latent representations (batch_size, latent_dim)
            cluster_assignments: Soft assignments (batch_size, num_clusters)
            cluster_centers: Cluster centers (num_clusters, latent_dim)

        Returns:
            Dictionary of auxiliary loss components
        """
        # Cluster balance loss - encourage balanced cluster sizes
        cluster_probs = torch.mean(cluster_assignments, dim=0)
        uniform_prob = 1.0 / self.num_clusters
        balance_loss = F.kl_div(
            torch.log(cluster_probs + 1e-8),
            torch.full_like(cluster_probs, uniform_prob),
            reduction='sum'
        )

        # Cluster separation loss - push cluster centers apart
        center_distances = torch.cdist(cluster_centers, cluster_centers)
        # Mask diagonal (self-distances)
        mask = ~torch.eye(self.num_clusters, dtype=torch.bool, device=center_distances.device)
        separation_loss = -torch.mean(center_distances[mask])

        # Compactness loss - pull points closer to their assigned centers
        hard_assignments = torch.argmax(cluster_assignments, dim=1)
        compactness_losses = []

        for k in range(self.num_clusters):
            cluster_mask = hard_assignments == k
            if torch.sum(cluster_mask) > 0:
                cluster_points = latent_z[cluster_mask]
                center = cluster_centers[k]
                distances = torch.norm(cluster_points - center, dim=1)
                compactness_losses.append(distances.mean())

        if compactness_losses:
            compactness_loss = torch.stack(compactness_losses).mean()
        else:
            compactness_loss = torch.tensor(0.0, device=latent_z.device)

        # Combined auxiliary loss
        auxiliary_loss = (
            self.balance_weight * balance_loss +
            self.separation_weight * separation_loss +
            self.compactness_weight * compactness_loss
        )

        return {
            'auxiliary_loss': auxiliary_loss,
            'balance_loss': balance_loss,
            'separation_loss': separation_loss,
            'compactness_loss': compactness_loss,
            'cluster_balance': cluster_probs.std()  # Lower is more balanced
        }


class PhenotypeClustering(nn.Module):
    """
    Phenotype-aware clustering loss for audiometric data.

    Incorporates domain knowledge about hearing phenotypes to guide
    clustering towards biologically meaningful groupings.
    """

    def __init__(self, phenotype_weight: float = 0.2):
        """
        Initialize phenotype clustering loss.

        Args:
            phenotype_weight: Weight for phenotype-specific terms
        """
        super().__init__()
        self.phenotype_weight = phenotype_weight

    def forward(self, latent_z: torch.Tensor, abr_features: torch.Tensor,
                cluster_assignments: torch.Tensor,
                gene_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute phenotype-aware clustering loss.

        Args:
            latent_z: Latent representations (batch_size, latent_dim)
            abr_features: Original ABR features (batch_size, 6)
            cluster_assignments: Soft cluster assignments (batch_size, num_clusters)
            gene_labels: Gene labels for consistency (optional)

        Returns:
            Dictionary of phenotype loss components
        """
        # ABR pattern consistency within clusters
        pattern_loss = self._compute_abr_pattern_consistency(
            abr_features, cluster_assignments
        )

        # Hearing loss severity coherence
        severity_loss = self._compute_severity_coherence(
            abr_features, cluster_assignments
        )

        # Gene coherence (if gene labels available)
        gene_coherence_loss = torch.tensor(0.0, device=latent_z.device)
        if gene_labels is not None:
            gene_coherence_loss = self._compute_gene_coherence(
                cluster_assignments, gene_labels
            )

        # Combined phenotype loss
        phenotype_loss = pattern_loss + severity_loss + gene_coherence_loss

        return {
            'phenotype_loss': phenotype_loss * self.phenotype_weight,
            'pattern_consistency': pattern_loss,
            'severity_coherence': severity_loss,
            'gene_coherence': gene_coherence_loss
        }

    def _compute_abr_pattern_consistency(self, abr_features: torch.Tensor,
                                        assignments: torch.Tensor) -> torch.Tensor:
        """Encourage similar ABR patterns within clusters."""
        num_clusters = assignments.shape[1]
        pattern_losses = []

        for k in range(num_clusters):
            # Weight samples by their assignment probability to cluster k
            weights = assignments[:, k]
            if weights.sum() > 1e-6:  # Avoid empty clusters
                # Weighted mean ABR pattern for cluster k
                weighted_mean = torch.sum(weights.unsqueeze(1) * abr_features, dim=0) / weights.sum()

                # Weighted variance from mean pattern
                diff = abr_features - weighted_mean.unsqueeze(0)
                weighted_var = torch.sum(weights.unsqueeze(1) * diff ** 2, dim=0) / weights.sum()
                pattern_losses.append(weighted_var.mean())

        if pattern_losses:
            return torch.stack(pattern_losses).mean()
        else:
            return torch.tensor(0.0, device=abr_features.device)

    def _compute_severity_coherence(self, abr_features: torch.Tensor,
                                   assignments: torch.Tensor) -> torch.Tensor:
        """Encourage coherent hearing loss severity within clusters."""
        # Compute average ABR threshold as severity proxy
        avg_thresholds = abr_features.mean(dim=1)

        num_clusters = assignments.shape[1]
        severity_losses = []

        for k in range(num_clusters):
            weights = assignments[:, k]
            if weights.sum() > 1e-6:
                # Weighted mean severity for cluster k
                weighted_severity = torch.sum(weights * avg_thresholds) / weights.sum()

                # Weighted variance in severity
                severity_diff = avg_thresholds - weighted_severity
                weighted_severity_var = torch.sum(weights * severity_diff ** 2) / weights.sum()
                severity_losses.append(weighted_severity_var)

        if severity_losses:
            return torch.stack(severity_losses).mean()
        else:
            return torch.tensor(0.0, device=abr_features.device)

    def _compute_gene_coherence(self, assignments: torch.Tensor,
                               gene_labels: torch.Tensor) -> torch.Tensor:
        """Encourage mice from same gene to cluster together."""
        unique_genes = torch.unique(gene_labels)
        coherence_losses = []

        for gene in unique_genes:
            if gene == -1:  # Skip unknown genes
                continue

            gene_mask = gene_labels == gene
            if torch.sum(gene_mask) > 1:
                # Assignments for mice with this gene
                gene_assignments = assignments[gene_mask]

                # Compute entropy of cluster distribution for this gene
                # Lower entropy means mice from same gene cluster together
                mean_assignment = gene_assignments.mean(dim=0)
                entropy = -torch.sum(mean_assignment * torch.log(mean_assignment + 1e-8))
                coherence_losses.append(entropy)

        if coherence_losses:
            return torch.stack(coherence_losses).mean()
        else:
            return torch.tensor(0.0, device=assignments.device)


class AdaptiveClusteringLoss(nn.Module):
    """
    Adaptive clustering loss that adjusts based on training progress.

    Starts with simple cluster formation and gradually introduces
    more sophisticated phenotype-aware objectives.
    """

    def __init__(self, config: Dict):
        """Initialize adaptive clustering loss."""
        super().__init__()
        self.config = config

        # Component losses
        self.dec_loss = DECLoss(
            alpha=config['clustering']['alpha'],
            cluster_update_interval=config['clustering']['update_interval']
        )

        self.auxiliary_loss = AuxiliaryClustering(
            num_clusters=config['clustering']['num_clusters'],
            balance_weight=0.1,
            separation_weight=0.1,
            compactness_weight=0.1
        )

        self.phenotype_loss = PhenotypeClustering(phenotype_weight=0.2)

        # Adaptation parameters
        self.warmup_epochs = config.get('clustering_warmup_epochs', 100)
        self.current_epoch = 0

    def forward(self, latent_z: torch.Tensor, q: torch.Tensor, p: torch.Tensor,
                cluster_centers: torch.Tensor, abr_features: torch.Tensor,
                gene_labels: Optional[torch.Tensor] = None,
                epoch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive clustering loss.

        Args:
            latent_z: Latent representations
            q: Current cluster assignments
            p: Target distribution
            cluster_centers: Cluster centers
            abr_features: Original ABR features
            gene_labels: Gene labels (optional)
            epoch: Current epoch

        Returns:
            Dictionary of adaptive clustering losses
        """
        if epoch is not None:
            self.current_epoch = epoch

        # Core DEC loss (always active)
        dec_losses = self.dec_loss(q, p)

        # Auxiliary clustering objectives
        aux_losses = self.auxiliary_loss(latent_z, q, cluster_centers)

        # Phenotype-aware objectives
        phenotype_losses = self.phenotype_loss(
            latent_z, abr_features, q, gene_labels
        )

        # Adaptive weighting based on training progress
        progress = min(self.current_epoch / self.warmup_epochs, 1.0)

        # Gradually introduce complexity
        dec_weight = 1.0
        aux_weight = 0.3 * progress
        phenotype_weight = 0.2 * progress

        # Combined adaptive loss
        total_clustering_loss = (
            dec_weight * dec_losses['dec_loss'] +
            aux_weight * aux_losses['auxiliary_loss'] +
            phenotype_weight * phenotype_losses['phenotype_loss']
        )

        # Combine all components
        combined_losses = {
            'total_clustering_loss': total_clustering_loss,
            'dec_component': dec_losses['dec_loss'],
            'auxiliary_component': aux_losses['auxiliary_loss'],
            'phenotype_component': phenotype_losses['phenotype_loss'],
            'dec_weight': dec_weight,
            'aux_weight': aux_weight,
            'phenotype_weight': phenotype_weight,
            **{f'dec_{k}': v for k, v in dec_losses.items()},
            **{f'aux_{k}': v for k, v in aux_losses.items()},
            **{f'phenotype_{k}': v for k, v in phenotype_losses.items()}
        }

        return combined_losses

    def update_epoch(self, epoch: int):
        """Update current epoch for adaptive weighting."""
        self.current_epoch = epoch


def create_clustering_loss(config: Dict) -> AdaptiveClusteringLoss:
    """
    Factory function to create clustering loss.

    Args:
        config: Clustering loss configuration

    Returns:
        Configured clustering loss module
    """
    return AdaptiveClusteringLoss(config)


class ClusteringLossAnalyzer:
    """Utility class for analyzing clustering loss behavior."""

    def __init__(self):
        """Initialize clustering loss analyzer."""
        pass

    def analyze_cluster_quality(self, q: torch.Tensor, latent_z: torch.Tensor,
                               cluster_centers: torch.Tensor) -> Dict[str, float]:
        """
        Analyze clustering quality metrics.

        Args:
            q: Soft cluster assignments
            latent_z: Latent representations
            cluster_centers: Cluster centers

        Returns:
            Dictionary of clustering quality metrics
        """
        with torch.no_grad():
            # Hard assignments
            hard_assignments = torch.argmax(q, dim=1)

            # Cluster sizes
            cluster_sizes = torch.bincount(hard_assignments, minlength=len(cluster_centers))

            # Assignment confidence
            max_probs = torch.max(q, dim=1)[0]
            mean_confidence = max_probs.mean()

            # Assignment entropy (lower = more confident)
            entropy = -torch.sum(q * torch.log(q + 1e-8), dim=1).mean()

            # Silhouette score approximation
            silhouette = self._compute_silhouette_approx(latent_z, hard_assignments, cluster_centers)

            return {
                'num_active_clusters': (cluster_sizes > 0).sum().item(),
                'cluster_balance': cluster_sizes.std().item() / cluster_sizes.mean().item(),
                'mean_confidence': mean_confidence.item(),
                'assignment_entropy': entropy.item(),
                'silhouette_score': silhouette,
                'largest_cluster_size': cluster_sizes.max().item(),
                'smallest_cluster_size': cluster_sizes.min().item()
            }

    def _compute_silhouette_approx(self, latent_z: torch.Tensor,
                                  assignments: torch.Tensor,
                                  cluster_centers: torch.Tensor) -> float:
        """Compute approximate silhouette score efficiently."""
        # Simplified silhouette using cluster centers
        intra_distances = []
        inter_distances = []

        for i, assignment in enumerate(assignments):
            point = latent_z[i]

            # Distance to own cluster center
            intra_dist = torch.norm(point - cluster_centers[assignment])
            intra_distances.append(intra_dist)

            # Distance to nearest other cluster center
            other_centers = torch.cat([
                cluster_centers[:assignment],
                cluster_centers[assignment+1:]
            ])
            if len(other_centers) > 0:
                inter_dist = torch.min(torch.norm(point.unsqueeze(0) - other_centers, dim=1))
                inter_distances.append(inter_dist)

        if inter_distances:
            intra_tensor = torch.stack(intra_distances)
            inter_tensor = torch.stack(inter_distances)
            silhouette = ((inter_tensor - intra_tensor) / torch.max(inter_tensor, intra_tensor)).mean()
            return silhouette.item()
        else:
            return 0.0