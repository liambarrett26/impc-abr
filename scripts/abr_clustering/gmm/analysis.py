"""
Analysis module for GMM clustering results.

Provides comprehensive analysis and visualization of clustering results,
including cluster characterization, gene associations, and biological validation.
"""

import logging
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ClusterCharacteristics:
    """Container for cluster characteristics and statistics."""

    cluster_id: int
    size: int
    proportion: float
    mean_profile: np.ndarray
    std_profile: np.ndarray
    center_profile: np.ndarray
    pattern_type: str
    severity_score: float
    dominant_genes: List[str]
    gene_count: int


class AudiometricAnalyzer:
    """
    Comprehensive analyzer for GMM clustering results on audiometric data.

    Provides cluster characterization, visualization, and biological interpretation
    of discovered audiometric phenotypes.
    """

    def __init__(self, frequency_labels: List[str] = None):
        """
        Initialize analyzer.

        Args:
            frequency_labels: Labels for ABR frequencies (default: 6,12,18,24,30 kHz)
        """
        if frequency_labels is None:
            self.frequency_labels = ["6 kHz", "12 kHz", "18 kHz", "24 kHz", "30 kHz"]
        else:
            self.frequency_labels = frequency_labels

        self.cluster_characteristics: Dict[int, ClusterCharacteristics] = {}
        self.gene_cluster_associations: Dict[str, Dict[str, Any]] = {}

    def analyze_clusters(
        self,
        normalized_data: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_probabilities: np.ndarray,
        metadata: pd.DataFrame,
        original_data: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of clustering results.

        Args:
            normalized_data: Normalized ABR data used for clustering
            cluster_labels: Predicted cluster assignments
            cluster_probabilities: Cluster assignment probabilities
            metadata: Metadata including gene information
            original_data: Original ABR data for interpretation

        Returns:
            Dictionary containing comprehensive analysis results
        """
        logger.info("Starting comprehensive cluster analysis")

        n_clusters = len(np.unique(cluster_labels))
        results = {
            "n_clusters": n_clusters,
            "cluster_sizes": {},
            "cluster_characteristics": {},
            "gene_associations": {},
            "summary_statistics": {},
        }

        # Analyze each cluster
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = normalized_data[cluster_mask]
            cluster_meta = (
                metadata.iloc[cluster_mask]
                if len(metadata) == len(normalized_data)
                else None
            )

            # Calculate cluster characteristics
            characteristics = self._characterize_cluster(
                cluster_id,
                cluster_data,
                cluster_meta,
                original_data[cluster_mask] if original_data is not None else None,
            )

            self.cluster_characteristics[cluster_id] = characteristics
            results["cluster_characteristics"][cluster_id] = characteristics
            results["cluster_sizes"][cluster_id] = characteristics.size

        # Analyze gene-cluster associations
        if cluster_meta is not None and "gene_symbol" in metadata.columns:
            gene_associations = self._analyze_gene_associations(
                cluster_labels, cluster_probabilities, metadata
            )
            results["gene_associations"] = gene_associations
            self.gene_cluster_associations = gene_associations

        # Calculate summary statistics
        results["summary_statistics"] = self._calculate_summary_statistics(
            normalized_data, cluster_labels, cluster_probabilities
        )

        logger.info(f"Analysis complete: {n_clusters} clusters identified")
        return results

    def _characterize_cluster(
        self,
        cluster_id: int,
        cluster_data: np.ndarray,
        cluster_meta: pd.DataFrame = None,
        original_data: np.ndarray = None,
    ) -> ClusterCharacteristics:
        """
        Characterize a single cluster with audiometric and biological features.

        Args:
            cluster_id: Cluster identifier
            cluster_data: Normalized data for this cluster
            cluster_meta: Metadata for cluster members
            original_data: Original scale data for interpretation

        Returns:
            ClusterCharacteristics object
        """
        # Basic statistics
        size = len(cluster_data)
        mean_profile = np.mean(cluster_data, axis=0)
        std_profile = np.std(cluster_data, axis=0)

        # Use original data for interpretable center if available
        if original_data is not None:
            center_profile = np.mean(original_data, axis=0)
        else:
            center_profile = mean_profile

        # Classify audiometric pattern
        pattern_type = self._classify_audiometric_pattern(mean_profile)

        # Calculate severity score (higher normalized values = more severe hearing loss)
        severity_score = np.mean(mean_profile)

        # Analyze genes in this cluster
        dominant_genes = []
        gene_count = 0

        if cluster_meta is not None and "gene_symbol" in cluster_meta.columns:
            gene_counts = cluster_meta["gene_symbol"].value_counts()
            dominant_genes = gene_counts.head(5).index.tolist()
            gene_count = len(gene_counts)

        return ClusterCharacteristics(
            cluster_id=cluster_id,
            size=size,
            proportion=0.0,  # Will be set later
            mean_profile=mean_profile,
            std_profile=std_profile,
            center_profile=center_profile,
            pattern_type=pattern_type,
            severity_score=severity_score,
            dominant_genes=dominant_genes,
            gene_count=gene_count,
        )

    def _classify_audiometric_pattern(self, profile: np.ndarray) -> str:
        """
        Classify audiometric pattern based on profile shape.

        Args:
            profile: Mean audiometric profile

        Returns:
            String description of pattern type
        """
        if len(profile) != 5:
            return "unknown"

        # Calculate differences between adjacent frequencies
        diffs = np.diff(profile)

        # Threshold for considering differences significant
        diff_threshold = 0.1  # Adjusted for normalized data

        # Check for flat pattern (minimal variation)
        if np.all(np.abs(diffs) < diff_threshold):
            if np.mean(profile) > 0.7:
                return "severe_flat"
            elif np.mean(profile) > 0.4:
                return "moderate_flat"
            else:
                return "normal_flat"

        # Check for high-frequency loss (increasing thresholds at higher frequencies)
        high_freq_slope = np.mean(diffs[2:])  # Focus on 18-30 kHz
        if high_freq_slope > diff_threshold:
            return "high_frequency_loss"

        # Check for low-frequency loss (decreasing thresholds at higher frequencies)
        low_freq_slope = np.mean(diffs[:2])  # Focus on 6-18 kHz
        if low_freq_slope < -diff_threshold:
            return "low_frequency_loss"

        # Check for cookie-bite pattern (U-shaped)
        if (
            profile[0] < profile[2]
            and profile[4] < profile[2]
            and profile[1] < profile[2]
            and profile[3] < profile[2]
        ):
            return "cookie_bite"

        # Check for reverse cookie-bite (inverted U)
        if (
            profile[0] > profile[2]
            and profile[4] > profile[2]
            and profile[1] > profile[2]
            and profile[3] > profile[2]
        ):
            return "reverse_cookie_bite"

        # Default to mixed pattern
        if np.mean(profile) > 0.5:
            return "mixed_severe"
        else:
            return "mixed_mild"

    def _analyze_gene_associations(
        self,
        cluster_labels: np.ndarray,
        cluster_probabilities: np.ndarray,
        metadata: pd.DataFrame,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze gene-cluster associations and enrichment.

        Args:
            cluster_labels: Cluster assignments
            cluster_probabilities: Assignment probabilities
            metadata: Metadata with gene information

        Returns:
            Dictionary of gene association results
        """
        gene_associations = {}

        if "gene_symbol" in metadata.columns:
            # Get unique genes
            genes = metadata["gene_symbol"].dropna().unique()

            for gene in genes:
                gene_mask = metadata["gene_symbol"] == gene
                if gene_mask.sum() < 3:  # Skip genes with too few observations
                    continue

                gene_labels = cluster_labels[gene_mask]
                gene_probs = cluster_probabilities[gene_mask]

                # Calculate cluster distribution for this gene
                cluster_dist = np.bincount(
                    gene_labels, minlength=len(np.unique(cluster_labels))
                )
                cluster_dist_norm = cluster_dist / cluster_dist.sum()

                # Find dominant cluster
                dominant_cluster = np.argmax(cluster_dist)
                dominant_proportion = cluster_dist_norm[dominant_cluster]

                # Calculate average assignment confidence
                max_probs = gene_probs.max(axis=1)
                avg_confidence = np.mean(max_probs)

                # Statistical test for enrichment in dominant cluster
                overall_dist = np.bincount(cluster_labels)
                overall_dist_norm = overall_dist / overall_dist.sum()
                expected_in_cluster = overall_dist_norm[dominant_cluster] * len(
                    gene_labels
                )
                observed_in_cluster = cluster_dist[dominant_cluster]

                # Chi-square test
                try:
                    chi2_stat, p_value = stats.chisquare(
                        [observed_in_cluster, len(gene_labels) - observed_in_cluster],
                        [expected_in_cluster, len(gene_labels) - expected_in_cluster],
                    )
                except:
                    p_value = 1.0

                gene_associations[gene] = {
                    "sample_size": len(gene_labels),
                    "cluster_distribution": cluster_dist.tolist(),
                    "cluster_proportions": cluster_dist_norm.tolist(),
                    "dominant_cluster": int(dominant_cluster),
                    "dominant_proportion": float(dominant_proportion),
                    "avg_confidence": float(avg_confidence),
                    "enrichment_p_value": float(p_value),
                }

        return gene_associations

    def _calculate_summary_statistics(
        self,
        normalized_data: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_probabilities: np.ndarray,
    ) -> Dict[str, Any]:
        """Calculate overall clustering summary statistics."""
        n_clusters = len(np.unique(cluster_labels))
        cluster_sizes = np.bincount(cluster_labels)

        # Assignment confidence statistics
        max_probs = cluster_probabilities.max(axis=1)

        # Silhouette score (if sklearn available and multiple clusters)
        silhouette = None
        if n_clusters > 1:
            try:
                from sklearn.metrics import silhouette_score

                silhouette = silhouette_score(normalized_data, cluster_labels)
            except:
                pass

        return {
            "total_samples": len(normalized_data),
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_sizes.tolist(),
            "cluster_proportions": (cluster_sizes / len(normalized_data)).tolist(),
            "avg_assignment_confidence": float(np.mean(max_probs)),
            "min_assignment_confidence": float(np.min(max_probs)),
            "max_assignment_confidence": float(np.max(max_probs)),
            "silhouette_score": float(silhouette) if silhouette is not None else None,
        }

    def create_visualizations(
        self,
        normalized_data: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_probabilities: np.ndarray,
        original_data: np.ndarray = None,
        output_dir: str = "results",
        use_confidence_intervals: bool = True,
    ) -> Dict[str, str]:
        """
        Create comprehensive visualizations of clustering results.

        Args:
            normalized_data: Normalized ABR data used for clustering
            cluster_labels: Cluster assignments
            cluster_probabilities: Assignment probabilities
            original_data: Original scale data for interpretable plots
            output_dir: Directory to save visualization files

        Returns:
            Dictionary mapping visualization names to file paths
        """
        logger.info("Creating clustering visualizations")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        plot_files = {}

        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

        # 1. Cluster audiogram profiles
        fig_profiles = self._plot_cluster_profiles(
            original_data if original_data is not None else normalized_data,
            cluster_labels,
            use_original_scale=original_data is not None,
            use_confidence_intervals=use_confidence_intervals,
        )
        profile_path_png = output_path / "cluster_audiogram_profiles.png"
        profile_path_eps = output_path / "cluster_audiogram_profiles.eps"
        fig_profiles.savefig(profile_path_png, dpi=1200, bbox_inches="tight")
        fig_profiles.savefig(profile_path_eps, format="eps", bbox_inches="tight")
        plt.close(fig_profiles)
        plot_files["audiogram_profiles_png"] = str(profile_path_png)
        plot_files["audiogram_profiles_eps"] = str(profile_path_eps)

        # 2. PCA visualization - clusters
        fig_pca = self._plot_pca_clusters(
            normalized_data, cluster_labels, cluster_probabilities
        )
        pca_path_png = output_path / "cluster_pca_clusters.png"
        pca_path_eps = output_path / "cluster_pca_clusters.eps"
        fig_pca.savefig(pca_path_png, dpi=1200, bbox_inches="tight")
        fig_pca.savefig(pca_path_eps, format="eps", bbox_inches="tight")
        plt.close(fig_pca)
        plot_files["pca_clusters_png"] = str(pca_path_png)
        plot_files["pca_clusters_eps"] = str(pca_path_eps)

        # 3. PCA visualization - confidence
        fig_pca_conf = self._plot_pca_confidence(
            normalized_data, cluster_labels, cluster_probabilities
        )
        pca_conf_path_png = output_path / "cluster_pca_confidence.png"
        pca_conf_path_eps = output_path / "cluster_pca_confidence.eps"
        fig_pca_conf.savefig(pca_conf_path_png, dpi=1200, bbox_inches="tight")
        fig_pca_conf.savefig(pca_conf_path_eps, format="eps", bbox_inches="tight")
        plt.close(fig_pca_conf)
        plot_files["pca_confidence_png"] = str(pca_conf_path_png)
        plot_files["pca_confidence_eps"] = str(pca_conf_path_eps)

        # 4. Cluster size and confidence distributions
        fig_dist = self._plot_cluster_distributions(
            cluster_labels, cluster_probabilities
        )
        dist_path_png = output_path / "cluster_distributions.png"
        dist_path_eps = output_path / "cluster_distributions.eps"
        fig_dist.savefig(dist_path_png, dpi=1200, bbox_inches="tight")
        fig_dist.savefig(dist_path_eps, format="eps", bbox_inches="tight")
        plt.close(fig_dist)
        plot_files["cluster_distributions_png"] = str(dist_path_png)
        plot_files["cluster_distributions_eps"] = str(dist_path_eps)

        # 5. Uncertainty heatmap
        fig_uncertainty = self._plot_uncertainty_heatmap(cluster_probabilities)
        uncertainty_path_png = output_path / "assignment_uncertainty.png"
        uncertainty_path_eps = output_path / "assignment_uncertainty.eps"
        fig_uncertainty.savefig(uncertainty_path_png, dpi=1200, bbox_inches="tight")
        fig_uncertainty.savefig(uncertainty_path_eps, format="eps", bbox_inches="tight")
        plt.close(fig_uncertainty)
        plot_files["uncertainty_heatmap_png"] = str(uncertainty_path_png)
        plot_files["uncertainty_heatmap_eps"] = str(uncertainty_path_eps)

        logger.info(f"Created {len(plot_files)} visualization files in {output_dir}")
        return plot_files

    def _plot_cluster_profiles(
        self,
        data: np.ndarray,
        cluster_labels: np.ndarray,
        use_original_scale: bool = True,
        use_confidence_intervals: bool = True,
    ) -> plt.Figure:
        """Plot mean audiogram profiles for each cluster."""
        n_clusters = len(np.unique(cluster_labels))

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Individual cluster profiles
        colors = sns.color_palette("husl", n_clusters)

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = data[cluster_mask]

            if len(cluster_data) == 0:
                continue

            mean_profile = np.mean(cluster_data, axis=0)

            if use_confidence_intervals:
                # Calculate 95% confidence intervals
                import scipy.stats as stats

                n = len(cluster_data)
                std_error = np.std(cluster_data, axis=0) / np.sqrt(n)
                confidence_level = 0.95
                t_critical = stats.t.ppf((1 + confidence_level) / 2, n - 1)
                margin_of_error = t_critical * std_error
                lower_bound = mean_profile - margin_of_error
                upper_bound = mean_profile + margin_of_error
                band_label = "95% CI"
            else:
                # Use standard deviation
                std_profile = np.std(cluster_data, axis=0)
                lower_bound = mean_profile - std_profile
                upper_bound = mean_profile + std_profile
                band_label = "±1 SD"

            ax.plot(
                self.frequency_labels,
                mean_profile,
                color=colors[cluster_id],
                linewidth=2,
                marker="o",
                markersize=6,
                label=f"Cluster {cluster_id + 1} (n={len(cluster_data)})",
            )

            ax.fill_between(
                self.frequency_labels,
                lower_bound,
                upper_bound,
                color=colors[cluster_id],
                alpha=0.2,
            )

        ax.set_xlabel("Frequency")
        ax.set_ylabel(
            "ABR Threshold (dB SPL)" if use_original_scale else "Normalized Threshold"
        )
        ax.set_title("Cluster Audiogram Profiles")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if use_original_scale:
            ax.set_ylim(0, 100)  # Standard audiogram range: 0-100 dB SPL

        plt.tight_layout()
        return fig

    def _plot_pca_clusters(
        self,
        normalized_data: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_probabilities: np.ndarray,
    ) -> plt.Figure:
        """Plot clusters in PCA space."""
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(normalized_data)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        n_clusters = len(np.unique(cluster_labels))
        colors = sns.color_palette("husl", n_clusters)

        # Plot colored by cluster
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            ax.scatter(
                data_pca[cluster_mask, 0],
                data_pca[cluster_mask, 1],
                c=[colors[cluster_id]],
                alpha=0.6,
                s=30,
                label=f"Cluster {cluster_id + 1}",
            )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.set_title("Clusters in PCA Space")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_pca_confidence(
        self,
        normalized_data: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_probabilities: np.ndarray,
    ) -> plt.Figure:
        """Plot assignment confidence in PCA space."""
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(normalized_data)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Plot colored by assignment confidence
        max_probs = cluster_probabilities.max(axis=1)
        scatter = ax.scatter(
            data_pca[:, 0], data_pca[:, 1], c=max_probs, cmap="plasma", s=30, alpha=0.7
        )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.set_title("Assignment Confidence in PCA Space")
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Assignment Confidence")

        plt.tight_layout()
        return fig

    def _plot_cluster_distributions(
        self, cluster_labels: np.ndarray, cluster_probabilities: np.ndarray
    ) -> plt.Figure:
        """Plot cluster size distributions and confidence statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        n_clusters = len(np.unique(cluster_labels))
        cluster_sizes = np.bincount(cluster_labels)

        # Top left: Cluster sizes
        axes[0, 0].bar(range(n_clusters), cluster_sizes)
        axes[0, 0].set_xlabel("Cluster ID")
        axes[0, 0].set_ylabel("Number of Samples")
        axes[0, 0].set_title("Cluster Sizes")
        axes[0, 0].grid(True, alpha=0.3)

        # Top right: Cluster proportions (pie chart)
        axes[0, 1].pie(
            cluster_sizes,
            labels=[f"Cluster {i}" for i in range(n_clusters)],
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[0, 1].set_title("Cluster Proportions")

        # Bottom left: Assignment confidence distribution
        max_probs = cluster_probabilities.max(axis=1)
        axes[1, 0].hist(max_probs, bins=30, alpha=0.7, edgecolor="black")
        axes[1, 0].axvline(
            np.mean(max_probs),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(max_probs):.3f}",
        )
        axes[1, 0].set_xlabel("Assignment Confidence")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Assignment Confidence Distribution")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Bottom right: Confidence by cluster
        conf_by_cluster = [max_probs[cluster_labels == i] for i in range(n_clusters)]
        axes[1, 1].boxplot(
            conf_by_cluster, labels=[f"C{i+1}" for i in range(n_clusters)]
        )
        axes[1, 1].set_xlabel("Cluster")
        axes[1, 1].set_ylabel("Assignment Confidence")
        axes[1, 1].set_title("Confidence by Cluster")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_uncertainty_heatmap(
        self, cluster_probabilities: np.ndarray
    ) -> plt.Figure:
        """Plot heatmap showing assignment uncertainty patterns."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Sort samples by dominant cluster and confidence
        max_probs = cluster_probabilities.max(axis=1)
        dominant_clusters = cluster_probabilities.argmax(axis=1)

        # Create sorting index
        sort_idx = np.lexsort((max_probs, dominant_clusters))
        sorted_probs = cluster_probabilities[sort_idx]

        # Left plot: Full probability matrix
        im1 = axes[0].imshow(sorted_probs.T, aspect="auto", cmap="viridis")
        axes[0].set_xlabel("Samples (sorted by cluster and confidence)")
        axes[0].set_ylabel("Cluster")
        axes[0].set_title("Assignment Probability Matrix")
        plt.colorbar(im1, ax=axes[0], label="Probability")

        # Right plot: Uncertainty (entropy)
        entropy = -np.sum(sorted_probs * np.log(sorted_probs + 1e-10), axis=1)

        im2 = axes[1].scatter(
            range(len(entropy)),
            entropy,
            c=dominant_clusters[sort_idx],
            cmap="tab10",
            alpha=0.6,
        )
        axes[1].set_xlabel("Samples (sorted)")
        axes[1].set_ylabel("Assignment Entropy (Uncertainty)")
        axes[1].set_title("Assignment Uncertainty by Sample")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def generate_report(
        self,
        analysis_results: Dict[str, Any],
        output_file: str = "clustering_report.txt",
    ) -> str:
        """
        Generate comprehensive text report of clustering analysis.

        Args:
            analysis_results: Results from analyze_clusters()
            output_file: Output file path

        Returns:
            Path to generated report file
        """
        logger.info(f"Generating clustering report: {output_file}")

        with open(output_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("AUDIOMETRIC PHENOTYPE CLUSTERING ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Overview
            f.write("OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"Total samples analyzed: {analysis_results['summary_statistics']['total_samples']}\n"
            )
            f.write(
                f"Number of clusters identified: {analysis_results['n_clusters']}\n"
            )
            f.write(
                f"Average assignment confidence: {analysis_results['summary_statistics']['avg_assignment_confidence']:.3f}\n"
            )

            if analysis_results["summary_statistics"]["silhouette_score"] is not None:
                f.write(
                    f"Silhouette score: {analysis_results['summary_statistics']['silhouette_score']:.3f}\n"
                )

            f.write("\n")

            # Cluster characteristics
            f.write("CLUSTER CHARACTERISTICS\n")
            f.write("-" * 40 + "\n")

            for cluster_id, char in analysis_results["cluster_characteristics"].items():
                f.write(f"\nCluster {cluster_id + 1}:\n")
                f.write(
                    f"  Size: {char.size} samples ({char.size/analysis_results['summary_statistics']['total_samples']*100:.1f}%)\n"
                )
                f.write(f"  Pattern type: {char.pattern_type}\n")
                f.write(f"  Severity score: {char.severity_score:.3f}\n")
                f.write(f"  Unique genes: {char.gene_count}\n")

                if char.dominant_genes:
                    f.write(f"  Top genes: {', '.join(char.dominant_genes[:3])}\n")

                # Profile summary
                if hasattr(char, "center_profile") and len(char.center_profile) == 5:
                    f.write("  Mean profile: ")
                    for i, freq in enumerate(self.frequency_labels):
                        f.write(f"{freq}={char.center_profile[i]:.1f}")
                        if i < len(self.frequency_labels) - 1:
                            f.write(", ")
                    f.write("\n")

            # Gene associations (if available)
            if analysis_results["gene_associations"]:
                f.write("\n\nGENE-CLUSTER ASSOCIATIONS\n")
                f.write("-" * 40 + "\n")

                # Sort genes by sample size and significance
                gene_items = list(analysis_results["gene_associations"].items())
                gene_items.sort(
                    key=lambda x: (x[1]["sample_size"], -x[1]["enrichment_p_value"]),
                    reverse=True,
                )

                f.write("Top gene associations (by sample size and significance):\n\n")

                for gene, assoc in gene_items[:20]:  # Top 20 genes
                    f.write(f"{gene}:\n")
                    f.write(f"  Sample size: {assoc['sample_size']}\n")
                    f.write(
                        f"  Dominant cluster: {assoc['dominant_cluster']} ({assoc['dominant_proportion']*100:.1f}%)\n"
                    )
                    f.write(f"  Assignment confidence: {assoc['avg_confidence']:.3f}\n")
                    f.write(
                        f"  Enrichment p-value: {assoc['enrichment_p_value']:.2e}\n"
                    )
                    f.write("\n")

            # Summary recommendations
            f.write("\nINTERPRETATION AND RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")

            # Identify potentially interesting clusters
            interesting_clusters = []
            for cluster_id, char in analysis_results["cluster_characteristics"].items():
                if char.pattern_type != "normal_flat" and char.size > 10:
                    interesting_clusters.append((cluster_id, char))

            if interesting_clusters:
                f.write("Clusters of potential biological interest:\n")
                for cluster_id, char in interesting_clusters:
                    f.write(
                        f"  - Cluster {cluster_id + 1}: {char.pattern_type} pattern with {char.size} samples\n"
                    )

            f.write(
                f"\nTotal unique audiometric patterns identified: {len(set(char.pattern_type for char in analysis_results['cluster_characteristics'].values()))}\n"
            )

            # Quality assessment
            avg_confidence = analysis_results["summary_statistics"][
                "avg_assignment_confidence"
            ]
            if avg_confidence > 0.8:
                f.write(
                    "Quality assessment: High confidence clustering (>0.8 average confidence)\n"
                )
            elif avg_confidence > 0.6:
                f.write(
                    "Quality assessment: Moderate confidence clustering (0.6-0.8 average confidence)\n"
                )
            else:
                f.write(
                    "Quality assessment: Low confidence clustering (<0.6 average confidence)\n"
                )
                f.write(
                    "  Consider: Additional preprocessing, different cluster numbers, or data quality issues\n"
                )

            f.write("\n" + "=" * 80 + "\n")

        logger.info(f"Report generated: {output_file}")
        return output_file


def analyze_gmm_results(
    normalized_data: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_probabilities: np.ndarray,
    metadata: pd.DataFrame,
    original_data: np.ndarray = None,
    output_dir: str = "results",
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Convenience function for complete GMM results analysis.

    Args:
        normalized_data: Normalized ABR data
        cluster_labels: Cluster assignments
        cluster_probabilities: Assignment probabilities
        metadata: Sample metadata
        original_data: Original scale ABR data
        output_dir: Output directory for files

    Returns:
        Tuple of (analysis_results, visualization_files)
    """
    analyzer = AudiometricAnalyzer()

    # Perform analysis
    analysis_results = analyzer.analyze_clusters(
        normalized_data, cluster_labels, cluster_probabilities, metadata, original_data
    )

    # Create visualizations
    visualization_files = analyzer.create_visualizations(
        normalized_data,
        cluster_labels,
        cluster_probabilities,
        original_data,
        output_dir,
    )

    # Generate report
    report_path = Path(output_dir) / "clustering_report.txt"
    analyzer.generate_report(analysis_results, str(report_path))
    visualization_files["report"] = str(report_path)

    return analysis_results, visualization_files


if __name__ == "__main__":
    # Example usage
    import sys

    from gmm import AudiometricGMM, create_default_gmm_config
    from loader import load_impc_data
    from preproc import preprocess_abr_data

    if len(sys.argv) < 2:
        print("Usage: python analysis.py <data_path>")
        sys.exit(1)

    # Load and process data
    data_path = sys.argv[1]
    df, _ = load_impc_data(data_path)

    # Take subset for testing
    test_df = df.sample(n=min(500, len(df)), random_state=42)
    normalized_data, preprocessor = preprocess_abr_data(test_df)

    # Fit GMM
    config = create_default_gmm_config(n_components_range=(3, 6), n_bootstrap=10)
    gmm = AudiometricGMM(config)
    gmm.fit(normalized_data)

    # Get results
    labels = gmm.predict(normalized_data)
    probabilities = gmm.predict_proba(normalized_data)

    # Analyze results
    abr_cols = [
        "6kHz-evoked ABR Threshold",
        "12kHz-evoked ABR Threshold",
        "18kHz-evoked ABR Threshold",
        "24kHz-evoked ABR Threshold",
        "30kHz-evoked ABR Threshold",
    ]
    original_data = test_df[abr_cols].values

    results, plot_files = analyze_gmm_results(
        normalized_data, labels, probabilities, test_df, original_data
    )

    print(f"Analysis complete. Results saved to: {list(plot_files.values())}")
    print(f"Identified {results['n_clusters']} clusters")
    print(
        "Cluster patterns:",
        [char.pattern_type for char in results["cluster_characteristics"].values()],
    )
