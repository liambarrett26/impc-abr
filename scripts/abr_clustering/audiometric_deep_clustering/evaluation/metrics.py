"""
Clustering evaluation metrics for ContrastiveVAE-DEC model.

This module provides comprehensive metrics for evaluating clustering quality,
including unsupervised metrics and biological validation measures specific
to audiometric phenotype discovery.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score
)
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import pandas as pd

logger = logging.getLogger(__name__)


class ClusteringMetrics:
    """
    Comprehensive clustering evaluation metrics.

    Provides both unsupervised clustering quality metrics and supervised
    metrics when ground truth labels are available.
    """

    def __init__(self):
        """Initialize clustering metrics calculator."""
        self.last_computed_metrics = {}

    def compute_unsupervised_metrics(self, embeddings: np.ndarray,
                                   cluster_labels: np.ndarray) -> Dict[str, float]:
        """
        Compute unsupervised clustering quality metrics.

        Args:
            embeddings: Data embeddings (n_samples, n_features)
            cluster_labels: Cluster assignments (n_samples,)

        Returns:
            Dictionary of unsupervised clustering metrics
        """
        metrics = {}

        # Check for valid clustering
        n_clusters = len(np.unique(cluster_labels))
        if n_clusters < 2:
            logger.warning("Less than 2 clusters found, metrics may be unreliable")
            return {'error': 'insufficient_clusters', 'n_clusters': n_clusters}

        try:
            # Silhouette Score (-1 to 1, higher is better)
            metrics['silhouette_score'] = silhouette_score(embeddings, cluster_labels)

            # Calinski-Harabasz Index (higher is better)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, cluster_labels)

            # Davies-Bouldin Index (lower is better)
            metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, cluster_labels)

            # Custom metrics
            metrics.update(self._compute_custom_unsupervised_metrics(embeddings, cluster_labels))

        except Exception as e:
            logger.error(f"Error computing unsupervised metrics: {e}")
            metrics['error'] = str(e)

        return metrics

    def compute_supervised_metrics(self, predicted_labels: np.ndarray,
                                 true_labels: np.ndarray) -> Dict[str, float]:
        """
        Compute supervised clustering metrics against ground truth.

        Args:
            predicted_labels: Predicted cluster assignments
            true_labels: Ground truth labels

        Returns:
            Dictionary of supervised clustering metrics
        """
        metrics = {}

        try:
            # Adjusted Rand Index (0 to 1, higher is better)
            metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, predicted_labels)

            # Normalized Mutual Information (0 to 1, higher is better)
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, predicted_labels)

            # Adjusted Mutual Information (0 to 1, higher is better)
            metrics['adjusted_mutual_info'] = adjusted_mutual_info_score(true_labels, predicted_labels)

            # Homogeneity, Completeness, V-measure (0 to 1, higher is better)
            metrics['homogeneity_score'] = homogeneity_score(true_labels, predicted_labels)
            metrics['completeness_score'] = completeness_score(true_labels, predicted_labels)
            metrics['v_measure_score'] = v_measure_score(true_labels, predicted_labels)

        except Exception as e:
            logger.error(f"Error computing supervised metrics: {e}")
            metrics['error'] = str(e)

        return metrics

    def _compute_custom_unsupervised_metrics(self, embeddings: np.ndarray,
                                           cluster_labels: np.ndarray) -> Dict[str, float]:
        """Compute custom unsupervised clustering metrics."""
        metrics = {}

        # Cluster balance (coefficient of variation of cluster sizes)
        cluster_sizes = np.bincount(cluster_labels)
        metrics['cluster_balance'] = np.std(cluster_sizes) / np.mean(cluster_sizes)

        # Intra-cluster cohesion (average within-cluster distance)
        intra_distances = []
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) > 1:
                cluster_points = embeddings[cluster_mask]
                distances = pdist(cluster_points)
                intra_distances.extend(distances)

        metrics['intra_cluster_distance'] = np.mean(intra_distances) if intra_distances else 0

        # Inter-cluster separation (average between-cluster distance)
        cluster_centers = []
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            center = np.mean(embeddings[cluster_mask], axis=0)
            cluster_centers.append(center)

        if len(cluster_centers) > 1:
            inter_distances = pdist(np.array(cluster_centers))
            metrics['inter_cluster_distance'] = np.mean(inter_distances)
        else:
            metrics['inter_cluster_distance'] = 0

        # Separation ratio (inter/intra distance ratio)
        if metrics['intra_cluster_distance'] > 0:
            metrics['separation_ratio'] = metrics['inter_cluster_distance'] / metrics['intra_cluster_distance']
        else:
            metrics['separation_ratio'] = float('inf')

        # Cluster entropy (measure of assignment uncertainty)
        cluster_probs = cluster_sizes / np.sum(cluster_sizes)
        metrics['cluster_entropy'] = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10))

        return metrics

    def compute_stability_metrics(self, embeddings_list: List[np.ndarray],
                                labels_list: List[np.ndarray]) -> Dict[str, float]:
        """
        Compute clustering stability metrics across multiple runs.

        Args:
            embeddings_list: List of embedding matrices from different runs
            labels_list: List of cluster label arrays from different runs

        Returns:
            Dictionary of stability metrics
        """
        if len(embeddings_list) < 2:
            return {'error': 'need_at_least_2_runs'}

        metrics = {}

        # Pairwise agreement between runs
        agreements = []
        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                ari = adjusted_rand_score(labels_list[i], labels_list[j])
                agreements.append(ari)

        metrics['mean_stability_ari'] = np.mean(agreements)
        metrics['std_stability_ari'] = np.std(agreements)

        # Silhouette score stability
        silhouette_scores = []
        for embeddings, labels in zip(embeddings_list, labels_list):
            try:
                score = silhouette_score(embeddings, labels)
                silhouette_scores.append(score)
            except:
                continue

        if silhouette_scores:
            metrics['mean_silhouette'] = np.mean(silhouette_scores)
            metrics['std_silhouette'] = np.std(silhouette_scores)

        # Number of clusters stability
        n_clusters = [len(np.unique(labels)) for labels in labels_list]
        metrics['mean_n_clusters'] = np.mean(n_clusters)
        metrics['std_n_clusters'] = np.std(n_clusters)

        return metrics


class BiologicalValidationMetrics:
    """
    Metrics for biological validation of audiometric phenotype clusters.

    Evaluates whether discovered clusters correspond to meaningful
    biological categories and hearing loss patterns.
    """

    def __init__(self):
        """Initialize biological validation metrics."""
        pass

    def compute_gene_coherence_metrics(self, cluster_labels: np.ndarray,
                                     gene_labels: np.ndarray) -> Dict[str, float]:
        """
        Compute metrics for gene-cluster coherence.

        Args:
            cluster_labels: Predicted cluster assignments
            gene_labels: Gene knockout labels

        Returns:
            Dictionary of gene coherence metrics
        """
        metrics = {}

        # Remove unknown genes (-1 labels)
        valid_mask = gene_labels != -1
        if np.sum(valid_mask) == 0:
            return {'error': 'no_valid_gene_labels'}

        valid_clusters = cluster_labels[valid_mask]
        valid_genes = gene_labels[valid_mask]

        # Gene purity: how well each cluster is dominated by single genes
        cluster_gene_purities = []
        for cluster_id in np.unique(valid_clusters):
            cluster_mask = valid_clusters == cluster_id
            cluster_genes = valid_genes[cluster_mask]

            if len(cluster_genes) > 0:
                # Most common gene in this cluster
                unique_genes, counts = np.unique(cluster_genes, return_counts=True)
                max_count = np.max(counts)
                purity = max_count / len(cluster_genes)
                cluster_gene_purities.append(purity)

        metrics['mean_gene_purity'] = np.mean(cluster_gene_purities) if cluster_gene_purities else 0
        metrics['std_gene_purity'] = np.std(cluster_gene_purities) if cluster_gene_purities else 0

        # Gene coverage: how many clusters each gene spans
        gene_cluster_spans = []
        for gene_id in np.unique(valid_genes):
            gene_mask = valid_genes == gene_id
            gene_clusters = valid_clusters[gene_mask]
            n_clusters_spanned = len(np.unique(gene_clusters))
            gene_cluster_spans.append(n_clusters_spanned)

        metrics['mean_gene_span'] = np.mean(gene_cluster_spans) if gene_cluster_spans else 0
        metrics['genes_in_single_cluster'] = np.sum(np.array(gene_cluster_spans) == 1) / len(gene_cluster_spans) if gene_cluster_spans else 0

        # Overall gene-cluster association strength
        if len(np.unique(valid_genes)) > 1 and len(np.unique(valid_clusters)) > 1:
            metrics['gene_cluster_nmi'] = normalized_mutual_info_score(valid_genes, valid_clusters)
            metrics['gene_cluster_ari'] = adjusted_rand_score(valid_genes, valid_clusters)

        return metrics

    def compute_phenotype_coherence_metrics(self, cluster_labels: np.ndarray,
                                          abr_features: np.ndarray) -> Dict[str, float]:
        """
        Compute metrics for audiometric phenotype coherence within clusters.

        Args:
            cluster_labels: Predicted cluster assignments
            abr_features: ABR threshold features (n_samples, n_frequencies)

        Returns:
            Dictionary of phenotype coherence metrics
        """
        metrics = {}

        # Within-cluster ABR pattern coherence
        cluster_coherences = []
        cluster_severities = []

        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_abr = abr_features[cluster_mask]

            if len(cluster_abr) > 1:
                # Pattern coherence: inverse of within-cluster variance
                cluster_var = np.mean(np.var(cluster_abr, axis=0))
                cluster_coherences.append(1 / (1 + cluster_var))

                # Severity coherence: consistency of average hearing loss
                avg_thresholds = np.mean(cluster_abr, axis=1)
                severity_var = np.var(avg_thresholds)
                cluster_severities.append(1 / (1 + severity_var))

        metrics['mean_pattern_coherence'] = np.mean(cluster_coherences) if cluster_coherences else 0
        metrics['mean_severity_coherence'] = np.mean(cluster_severities) if cluster_severities else 0

        # Frequency-specific analysis
        frequency_separations = []
        for freq_idx in range(abr_features.shape[1]):
            freq_thresholds = abr_features[:, freq_idx]

            # Between-cluster variance vs within-cluster variance for this frequency
            between_var = 0
            within_var = 0
            total_mean = np.mean(freq_thresholds)

            for cluster_id in np.unique(cluster_labels):
                cluster_mask = cluster_labels == cluster_id
                cluster_freq = freq_thresholds[cluster_mask]

                if len(cluster_freq) > 1:
                    cluster_mean = np.mean(cluster_freq)
                    between_var += len(cluster_freq) * (cluster_mean - total_mean) ** 2
                    within_var += np.sum((cluster_freq - cluster_mean) ** 2)

            if within_var > 0:
                separation = between_var / within_var
                frequency_separations.append(separation)

        metrics['mean_frequency_separation'] = np.mean(frequency_separations) if frequency_separations else 0

        # Hearing loss type identification
        metrics.update(self._analyze_hearing_loss_types(cluster_labels, abr_features))

        return metrics

    def _analyze_hearing_loss_types(self, cluster_labels: np.ndarray,
                                   abr_features: np.ndarray) -> Dict[str, float]:
        """Analyze hearing loss type patterns in clusters."""
        metrics = {}

        # Define frequency ranges (assuming 6, 12, 18, 24, 30 kHz, click)
        low_freq_idx = [0, 1]  # 6, 12 kHz
        mid_freq_idx = [1, 2, 3]  # 12, 18, 24 kHz
        high_freq_idx = [3, 4]  # 24, 30 kHz

        cluster_types = {'flat': 0, 'high_freq': 0, 'low_freq': 0, 'mixed': 0}

        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_abr = abr_features[cluster_mask]

            if len(cluster_abr) > 0:
                # Average pattern for this cluster
                mean_pattern = np.mean(cluster_abr, axis=0)

                # Classify hearing loss type
                low_avg = np.mean(mean_pattern[low_freq_idx])
                mid_avg = np.mean(mean_pattern[mid_freq_idx])
                high_avg = np.mean(mean_pattern[high_freq_idx])

                # Simple classification rules
                if abs(high_avg - low_avg) < 10:  # Relatively flat
                    cluster_types['flat'] += 1
                elif high_avg > low_avg + 15:  # High frequency loss
                    cluster_types['high_freq'] += 1
                elif low_avg > high_avg + 15:  # Low frequency loss
                    cluster_types['low_freq'] += 1
                else:
                    cluster_types['mixed'] += 1

        total_clusters = sum(cluster_types.values())
        if total_clusters > 0:
            for loss_type, count in cluster_types.items():
                metrics[f'proportion_{loss_type}_loss'] = count / total_clusters

        return metrics


class ModelPerformanceMetrics:
    """
    Metrics for evaluating overall model performance.

    Combines clustering quality with reconstruction quality and
    other model-specific metrics.
    """

    def __init__(self):
        """Initialize model performance metrics."""
        pass

    def compute_reconstruction_metrics(self, original: np.ndarray,
                                     reconstructed: np.ndarray) -> Dict[str, float]:
        """
        Compute reconstruction quality metrics.

        Args:
            original: Original input features
            reconstructed: Reconstructed features

        Returns:
            Dictionary of reconstruction metrics
        """
        metrics = {}

        # Overall reconstruction quality
        mse = np.mean((original - reconstructed) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(original - reconstructed))

        metrics['reconstruction_mse'] = mse
        metrics['reconstruction_rmse'] = rmse
        metrics['reconstruction_mae'] = mae

        # Feature-wise reconstruction (assuming first 6 are ABR features)
        if original.shape[1] >= 6:
            abr_original = original[:, :6]
            abr_reconstructed = reconstructed[:, :6]

            metrics['abr_reconstruction_mse'] = np.mean((abr_original - abr_reconstructed) ** 2)
            metrics['abr_reconstruction_mae'] = np.mean(np.abs(abr_original - abr_reconstructed))

        # Correlation between original and reconstructed
        correlations = []
        for i in range(original.shape[1]):
            if np.std(original[:, i]) > 0 and np.std(reconstructed[:, i]) > 0:
                corr, _ = pearsonr(original[:, i], reconstructed[:, i])
                correlations.append(corr)

        metrics['mean_feature_correlation'] = np.mean(correlations) if correlations else 0

        return metrics

    def compute_latent_space_metrics(self, latent_representations: np.ndarray) -> Dict[str, float]:
        """
        Compute latent space quality metrics.

        Args:
            latent_representations: Latent space embeddings

        Returns:
            Dictionary of latent space metrics
        """
        metrics = {}

        # Latent space utilization
        latent_stds = np.std(latent_representations, axis=0)
        metrics['active_dimensions'] = np.sum(latent_stds > 0.01)
        metrics['latent_capacity'] = metrics['active_dimensions'] / latent_representations.shape[1]

        # Latent space organization
        metrics['latent_mean_norm'] = np.mean(np.linalg.norm(latent_representations, axis=1))
        metrics['latent_mean_std'] = np.mean(latent_stds)

        # Latent space separability (average pairwise distance)
        if len(latent_representations) > 1:
            pairwise_dists = pdist(latent_representations)
            metrics['latent_mean_distance'] = np.mean(pairwise_dists)
            metrics['latent_distance_std'] = np.std(pairwise_dists)

        return metrics


def compute_comprehensive_metrics(embeddings: np.ndarray,
                                cluster_labels: np.ndarray,
                                original_features: Optional[np.ndarray] = None,
                                reconstructed_features: Optional[np.ndarray] = None,
                                gene_labels: Optional[np.ndarray] = None,
                                abr_features: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    """
    Compute comprehensive evaluation metrics for clustering results.

    Args:
        embeddings: Latent space embeddings
        cluster_labels: Predicted cluster assignments
        original_features: Original input features (optional)
        reconstructed_features: Reconstructed features (optional)
        gene_labels: Gene knockout labels (optional)
        abr_features: ABR threshold features (optional)

    Returns:
        Dictionary of metric categories and their values
    """
    results = {}

    # Clustering metrics
    clustering_metrics = ClusteringMetrics()
    results['clustering'] = clustering_metrics.compute_unsupervised_metrics(embeddings, cluster_labels)

    # Biological validation (if gene/ABR data available)
    if gene_labels is not None or abr_features is not None:
        bio_metrics = BiologicalValidationMetrics()

        if gene_labels is not None:
            results['gene_coherence'] = bio_metrics.compute_gene_coherence_metrics(cluster_labels, gene_labels)

        if abr_features is not None:
            results['phenotype_coherence'] = bio_metrics.compute_phenotype_coherence_metrics(cluster_labels, abr_features)

    # Model performance metrics
    if original_features is not None or reconstructed_features is not None:
        model_metrics = ModelPerformanceMetrics()

        if original_features is not None and reconstructed_features is not None:
            results['reconstruction'] = model_metrics.compute_reconstruction_metrics(original_features, reconstructed_features)

        results['latent_space'] = model_metrics.compute_latent_space_metrics(embeddings)

    return results


def summarize_metrics(metrics_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Create a summary of the most important metrics.

    Args:
        metrics_dict: Complete metrics dictionary

    Returns:
        Dictionary of key summary metrics
    """
    summary = {}

    # Clustering quality
    if 'clustering' in metrics_dict:
        clustering = metrics_dict['clustering']
        summary['silhouette_score'] = clustering.get('silhouette_score', 0)
        summary['separation_ratio'] = clustering.get('separation_ratio', 0)
        summary['cluster_balance'] = clustering.get('cluster_balance', 0)

    # Biological relevance
    if 'gene_coherence' in metrics_dict:
        gene = metrics_dict['gene_coherence']
        summary['gene_purity'] = gene.get('mean_gene_purity', 0)
        summary['gene_cluster_nmi'] = gene.get('gene_cluster_nmi', 0)

    if 'phenotype_coherence' in metrics_dict:
        phenotype = metrics_dict['phenotype_coherence']
        summary['pattern_coherence'] = phenotype.get('mean_pattern_coherence', 0)
        summary['frequency_separation'] = phenotype.get('mean_frequency_separation', 0)

    # Model performance
    if 'reconstruction' in metrics_dict:
        recon = metrics_dict['reconstruction']
        summary['reconstruction_quality'] = 1 / (1 + recon.get('reconstruction_mse', 1))
        summary['feature_correlation'] = recon.get('mean_feature_correlation', 0)

    if 'latent_space' in metrics_dict:
        latent = metrics_dict['latent_space']
        summary['latent_capacity'] = latent.get('latent_capacity', 0)

    return summary