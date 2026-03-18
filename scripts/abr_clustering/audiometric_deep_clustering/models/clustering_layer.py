"""
Deep Embedded Clustering (DEC) layer for ContrastiveVAE-DEC model.

This module implements the clustering component that learns to assign
latent representations to phenotype clusters, with soft assignments
and iterative cluster center updates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import logging
import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class ClusteringLayer(nn.Module):
    """
    DEC clustering layer with learnable cluster centers.

    Implements soft assignment of latent vectors to clusters using
    Student's t-distribution and learns cluster centers through
    gradient descent.
    """

    def __init__(self, config: Dict):
        """
        Initialize clustering layer.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.latent_dim = config['latent']['latent_dim']
        self.num_clusters = config['clustering']['num_clusters']
        self.alpha = config['clustering']['alpha']  # Degrees of freedom for t-distribution

        # Learnable cluster centers
        self.cluster_centers = nn.Parameter(
            torch.randn(self.num_clusters, self.latent_dim)
        )

        # Initialize cluster centers with small random values
        nn.init.xavier_uniform_(self.cluster_centers)

        # Track cluster assignments for analysis
        self.register_buffer('assignment_history', torch.zeros(1000, self.num_clusters))
        self.history_idx = 0

        logger.info(f"Initialized clustering layer: {self.num_clusters} clusters, "
                   f"latent_dim={self.latent_dim}, alpha={self.alpha}")

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute cluster assignments for latent vectors.

        Args:
            z: Latent vectors (batch_size, latent_dim)

        Returns:
            Dictionary containing:
                - q: Soft cluster assignments (batch_size, num_clusters)
                - distances: Distances to cluster centers (batch_size, num_clusters)
                - hard_assignments: Hard cluster assignments (batch_size,)
        """
        # Compute squared distances to cluster centers
        # ||z_i - μ_j||^2 for all i, j
        distances_squared = torch.sum(
            (z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2,
            dim=2
        )  # (batch_size, num_clusters)

        # Compute soft assignments using Student's t-distribution
        # q_ij = (1 + ||z_i - μ_j||^2 / α)^(-(α+1)/2) / Σ_k (1 + ||z_i - μ_k||^2 / α)^(-(α+1)/2)
        numerator = (1.0 + distances_squared / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        q = numerator / torch.sum(numerator, dim=1, keepdim=True)

        # Hard assignments (for analysis)
        hard_assignments = torch.argmax(q, dim=1)

        # Update assignment history for monitoring
        if self.training:
            self._update_assignment_history(q)

        return {
            'q': q,
            'distances': torch.sqrt(distances_squared + 1e-8),
            'hard_assignments': hard_assignments,
            'cluster_centers': self.cluster_centers
        }

    def compute_target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute target distribution P from current assignments Q.

        This sharpens the cluster assignments to improve separation.

        Args:
            q: Current soft assignments (batch_size, num_clusters)

        Returns:
            Target distribution P (batch_size, num_clusters)
        """
        # Frequency of assignment to each cluster
        f_j = torch.sum(q, dim=0)  # (num_clusters,)

        # Compute p_ij = q_ij^2 / f_j / Σ_k (q_ik^2 / f_k)
        numerator = q ** 2 / f_j.unsqueeze(0)
        denominator = torch.sum(numerator, dim=1, keepdim=True)

        p = numerator / (denominator + 1e-8)

        return p

    def clustering_loss(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Compute DEC clustering loss (KL divergence between P and Q).

        Args:
            q: Current assignments
            p: Target distribution

        Returns:
            Clustering loss (scalar)
        """
        # KL divergence: Σ p_ij * log(p_ij / q_ij)
        kl_div = torch.sum(p * torch.log(p / (q + 1e-8) + 1e-8))
        return kl_div

    def initialize_clusters(self, data: torch.Tensor, method: str = 'kmeans') -> None:
        """
        Initialize cluster centers using the specified method.

        Args:
            data: Latent representations for initialization (n_samples, latent_dim)
            method: Initialization method ('kmeans', 'random', 'kmeans++')
        """
        logger.info(f"Initializing clusters using {method}")

        with torch.no_grad():
            if method == 'kmeans' or method == 'kmeans++':
                # Use scikit-learn KMeans for initialization
                data_np = data.cpu().numpy()
                kmeans = KMeans(
                    n_clusters=self.num_clusters,
                    init='k-means++' if method == 'kmeans++' else 'random',
                    n_init=10,
                    random_state=42
                )
                kmeans.fit(data_np)

                # Set cluster centers
                self.cluster_centers.data = torch.tensor(
                    kmeans.cluster_centers_,
                    dtype=torch.float32,
                    device=self.cluster_centers.device
                )

            elif method == 'random':
                # Random initialization from data distribution
                indices = torch.randperm(len(data))[:self.num_clusters]
                self.cluster_centers.data = data[indices].clone()

            elif method == 'spread':
                # Initialize centers spread across the data
                # Choose first center randomly
                centers = [data[torch.randint(len(data), (1,))]]

                # Choose remaining centers to maximize distance
                for _ in range(self.num_clusters - 1):
                    distances = torch.stack([
                        torch.min(torch.norm(data - center.unsqueeze(0), dim=1))
                        for center in centers
                    ])
                    farthest_idx = torch.argmax(torch.min(distances, dim=0)[0])
                    centers.append(data[farthest_idx:farthest_idx+1])

                self.cluster_centers.data = torch.cat(centers, dim=0)

        logger.info("Cluster initialization completed")

    def get_cluster_assignments(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cluster assignments for given latent vectors.

        Args:
            z: Latent vectors

        Returns:
            Tuple of (soft_assignments, hard_assignments)
        """
        with torch.no_grad():
            output = self.forward(z)
            return output['q'], output['hard_assignments']

    def _update_assignment_history(self, q: torch.Tensor) -> None:
        """Update the history of cluster assignments for monitoring."""
        # Compute cluster frequencies
        cluster_freq = torch.mean(q, dim=0)

        # Update circular buffer
        if self.history_idx < self.assignment_history.size(0):
            self.assignment_history[self.history_idx] = cluster_freq
            self.history_idx += 1
        else:
            # Shift buffer and add new entry
            self.assignment_history[:-1] = self.assignment_history[1:].clone()
            self.assignment_history[-1] = cluster_freq

    def get_cluster_statistics(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive cluster statistics.

        Args:
            z: Latent vectors

        Returns:
            Dictionary of cluster statistics
        """
        with torch.no_grad():
            output = self.forward(z)
            q = output['q']
            hard_assignments = output['hard_assignments']

            # Cluster sizes
            cluster_sizes = torch.bincount(hard_assignments, minlength=self.num_clusters)

            # Cluster confidence (entropy of assignments)
            entropy = -torch.sum(q * torch.log(q + 1e-8), dim=1)

            # Intra-cluster distances
            intra_cluster_distances = []
            for k in range(self.num_clusters):
                mask = hard_assignments == k
                if torch.sum(mask) > 1:
                    cluster_points = z[mask]
                    center = self.cluster_centers[k]
                    distances = torch.norm(cluster_points - center, dim=1)
                    intra_cluster_distances.append(distances.mean())
                else:
                    intra_cluster_distances.append(torch.tensor(0.0, device=z.device, dtype=z.dtype))

            # Inter-cluster distances
            inter_cluster_distances = torch.norm(
                self.cluster_centers.unsqueeze(1) - self.cluster_centers.unsqueeze(0),
                dim=2
            )

            return {
                'cluster_sizes': cluster_sizes,
                'cluster_proportions': cluster_sizes.float() / len(z),
                'mean_entropy': entropy.mean(),
                'max_entropy': entropy.max(),
                'min_entropy': entropy.min(),
                'intra_cluster_distances': torch.stack(intra_cluster_distances),
                'inter_cluster_distances': inter_cluster_distances,
                'silhouette_score': self._compute_silhouette_score(z, hard_assignments)
            }

    def _compute_silhouette_score(self, z: torch.Tensor,
                                 assignments: torch.Tensor) -> torch.Tensor:
        """
        Compute simplified silhouette score.

        Args:
            z: Latent vectors
            assignments: Hard cluster assignments

        Returns:
            Mean silhouette score
        """
        silhouette_scores = []

        for i in range(len(z)):
            # Current point and its cluster
            point = z[i]
            cluster = assignments[i]

            # Intra-cluster distance (a)
            same_cluster_mask = assignments == cluster
            same_cluster_points = z[same_cluster_mask]

            if torch.sum(same_cluster_mask) > 1:
                a = torch.mean(torch.norm(same_cluster_points - point, dim=1))
            else:
                a = torch.tensor(0.0)

            # Inter-cluster distance (b) - minimum distance to other clusters
            b_values = []
            for k in range(self.num_clusters):
                if k != cluster:
                    other_cluster_mask = assignments == k
                    if torch.sum(other_cluster_mask) > 0:
                        other_cluster_points = z[other_cluster_mask]
                        b_k = torch.mean(torch.norm(other_cluster_points - point, dim=1))
                        b_values.append(b_k)

            if b_values:
                b = torch.min(torch.stack(b_values))

                # Silhouette score for this point
                s = (b - a) / torch.max(a, b)
                silhouette_scores.append(s)

        if silhouette_scores:
            return torch.mean(torch.stack(silhouette_scores))
        else:
            return torch.tensor(0.0)


class ClusteringScheduler:
    """
    Scheduler for managing clustering updates during training.

    Controls when to update cluster centers and target distributions
    to ensure stable convergence.
    """

    def __init__(self, update_interval: int = 50, tolerance: float = 0.001):
        """
        Initialize clustering scheduler.

        Args:
            update_interval: Number of epochs between cluster updates
            tolerance: Convergence tolerance for cluster centers
        """
        self.update_interval = update_interval
        self.tolerance = tolerance
        self.epoch = 0
        self.last_centers = None

    def should_update(self, clustering_layer: ClusteringLayer) -> bool:
        """
        Determine if cluster centers should be updated.

        Args:
            clustering_layer: The clustering layer

        Returns:
            Whether to update clusters
        """
        if self.epoch % self.update_interval == 0:
            return True

        # Check for convergence
        if self.last_centers is not None:
            current_centers = clustering_layer.cluster_centers.data
            center_shift = torch.norm(current_centers - self.last_centers)

            if center_shift < self.tolerance:
                logger.info(f"Cluster centers converged (shift: {center_shift:.6f})")
                return False

        return False

    def update_centers(self, clustering_layer: ClusteringLayer) -> None:
        """
        Update cluster center tracking.

        Args:
            clustering_layer: The clustering layer
        """
        self.last_centers = clustering_layer.cluster_centers.data.clone()

    def step(self) -> None:
        """Step the scheduler."""
        self.epoch += 1


class ClusteringAnalyzer:
    """
    Utility class for analyzing clustering behavior and quality.
    """

    def __init__(self, config: Dict):
        """Initialize clustering analyzer."""
        self.config = config
        self.num_clusters = config['clustering']['num_clusters']

    def analyze_cluster_quality(self, z: torch.Tensor,
                               assignments: torch.Tensor,
                               gene_labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Comprehensive cluster quality analysis.

        Args:
            z: Latent representations
            assignments: Hard cluster assignments
            gene_labels: Optional gene labels for biological validation

        Returns:
            Dictionary of quality metrics
        """
        metrics = {}

        # Basic cluster statistics
        cluster_sizes = torch.bincount(assignments, minlength=self.num_clusters)
        metrics['num_non_empty_clusters'] = (cluster_sizes > 0).sum().item()
        metrics['cluster_balance'] = (cluster_sizes.min() / cluster_sizes.max()).item() if cluster_sizes.max() > 0 else 0.0

        # Cluster separation
        unique_clusters = torch.unique(assignments)
        if len(unique_clusters) > 1:
            inter_cluster_dist = self._compute_inter_cluster_distance(z, assignments)
            intra_cluster_dist = self._compute_intra_cluster_distance(z, assignments)
            metrics['separation_ratio'] = (inter_cluster_dist / (intra_cluster_dist + 1e-8)).item()

        # Biological validation if gene labels available
        if gene_labels is not None:
            metrics.update(self._compute_biological_metrics(assignments, gene_labels))

        return metrics

    def _compute_inter_cluster_distance(self, z: torch.Tensor,
                                       assignments: torch.Tensor) -> torch.Tensor:
        """Compute average distance between cluster centers."""
        cluster_centers = []
        for k in range(self.num_clusters):
            mask = assignments == k
            if torch.sum(mask) > 0:
                center = torch.mean(z[mask], dim=0)
                cluster_centers.append(center)

        if len(cluster_centers) > 1:
            centers = torch.stack(cluster_centers)
            distances = torch.pdist(centers)
            return torch.mean(distances)
        else:
            return torch.tensor(0.0)

    def _compute_intra_cluster_distance(self, z: torch.Tensor,
                                       assignments: torch.Tensor) -> torch.Tensor:
        """Compute average within-cluster distance."""
        total_distance = 0.0
        total_points = 0

        for k in range(self.num_clusters):
            mask = assignments == k
            cluster_points = z[mask]

            if len(cluster_points) > 1:
                center = torch.mean(cluster_points, dim=0)
                distances = torch.norm(cluster_points - center, dim=1)
                total_distance += torch.sum(distances)
                total_points += len(cluster_points)

        if total_points > 0:
            return total_distance / total_points
        else:
            return torch.tensor(0.0)

    def _compute_biological_metrics(self, assignments: torch.Tensor,
                                   gene_labels: torch.Tensor) -> Dict[str, float]:
        """Compute biological validation metrics."""
        metrics = {}

        # Gene purity: how many clusters are dominated by single genes
        gene_purities = []
        for k in range(self.num_clusters):
            mask = assignments == k
            if torch.sum(mask) > 0:
                cluster_genes = gene_labels[mask]
                most_common_gene = torch.mode(cluster_genes)[0]
                purity = (cluster_genes == most_common_gene).float().mean()
                gene_purities.append(purity.item())

        if gene_purities:
            metrics['mean_gene_purity'] = np.mean(gene_purities)
            metrics['max_gene_purity'] = np.max(gene_purities)

        return metrics