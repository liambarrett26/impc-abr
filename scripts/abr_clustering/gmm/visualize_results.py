#!/usr/bin/env python3
"""
Standalone visualization script for GMM clustering results.

This script recreates all visualizations from saved pipeline outputs,
allowing for customization without re-running the modeling.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """Create visualizations from saved GMM clustering results."""

    def __init__(self, results_dir: Path, assignment_method: str = "probabilistic"):
        """
        Initialize visualizer with results directory.

        Args:
            results_dir: Directory containing pipeline outputs
            assignment_method: 'probabilistic' or 'euclidean' for cluster assignment
        """
        self.results_dir = Path(results_dir)
        self.assignment_method = assignment_method
        self.frequency_labels = ["6 kHz", "12 kHz", "18 kHz", "24 kHz", "30 kHz"]

        # Load saved data
        self._load_results()

    def _load_results(self):
        """Load all necessary data from saved files."""
        # Load analysis results
        results_path = self.results_dir / "analysis_results.json"
        with open(results_path, "r", encoding="utf-8") as f:
            self.analysis_results = json.load(f)

        if self.assignment_method == "euclidean":
            # Load original data for Euclidean distance calculation
            original_path = self.results_dir / "original_data.csv"
            if not original_path.exists():
                # Try alternate location
                original_path = Path(
                    "results/june_23_2025/gmm_k4_tied/original_data.csv"
                )
            if original_path.exists():
                self.original_data_df = pd.read_csv(original_path)
            else:
                raise FileNotFoundError(
                    "Original data required for Euclidean assignments"
                )

            # Compute Euclidean assignments
            self.cluster_labels, self.cluster_probabilities = (
                self._compute_euclidean_assignments()
            )
        else:
            # Load cluster assignments for probabilistic
            assignments_path = self.results_dir / "cluster_assignments.csv"
            if assignments_path.exists():
                self.assignments_df = pd.read_csv(assignments_path)
                self.cluster_labels = self.assignments_df["cluster_label"].values

                # Extract cluster probabilities
                prob_cols = [
                    col
                    for col in self.assignments_df.columns
                    if col.startswith("cluster_") and col.endswith("_prob")
                ]
                self.cluster_probabilities = self.assignments_df[prob_cols].values
            else:
                # Load from numpy files (older format)
                self.cluster_labels = np.load(self.results_dir / "cluster_labels.npy")
                self.cluster_probabilities = np.load(
                    self.results_dir / "cluster_probabilities.npy"
                )

        # Load normalized data
        normalized_path = self.results_dir / "normalized_data.csv"
        if normalized_path.exists():
            self.normalized_data = pd.read_csv(normalized_path).values
        else:
            # Try loading from shared data
            shared_path = Path("shared_data/normalized_data.npy")
            if shared_path.exists():
                self.normalized_data = np.load(shared_path)
            else:
                raise FileNotFoundError("Cannot find normalized data")

        # Check for original data (if saved separately)
        if self.assignment_method != "euclidean":
            original_path = self.results_dir / "original_data.csv"
            if original_path.exists():
                self.original_data = pd.read_csv(original_path).values
            else:
                self.original_data = None
                logger.warning(
                    "Original data not found. Some visualizations will use normalized scale."
                )
        else:
            # For Euclidean, extract ABR columns from the loaded DataFrame
            ABR_COLS = [
                "6kHz-evoked ABR Threshold",
                "12kHz-evoked ABR Threshold",
                "18kHz-evoked ABR Threshold",
                "24kHz-evoked ABR Threshold",
                "30kHz-evoked ABR Threshold",
            ]
            self.original_data = self.original_data_df[ABR_COLS].values

        self.n_clusters = len(np.unique(self.cluster_labels))
        logger.info(
            f"Loaded results ({self.assignment_method}): {self.n_clusters} clusters, {len(self.cluster_labels)} samples"
        )

    def _compute_euclidean_assignments(self):
        """Compute cluster assignments using Euclidean distance in original space."""
        # First load probabilistic labels to compute cluster means
        prob_labels_path = self.results_dir / "cluster_labels.npy"
        if not prob_labels_path.exists():
            prob_labels_path = Path(
                "results/june_23_2025/gmm_k4_tied/cluster_labels.npy"
            )
        prob_labels = np.load(prob_labels_path)

        # ABR columns
        ABR_COLS = [
            "6kHz-evoked ABR Threshold",
            "12kHz-evoked ABR Threshold",
            "18kHz-evoked ABR Threshold",
            "24kHz-evoked ABR Threshold",
            "30kHz-evoked ABR Threshold",
        ]

        # Calculate cluster means in original space
        cluster_means_original = {}
        for cluster_id in range(4):
            cluster_mask = prob_labels == cluster_id
            cluster_data = self.original_data_df.iloc[cluster_mask][ABR_COLS]
            cluster_means_original[cluster_id] = cluster_data.mean().values

        # Compute Euclidean distances and assignments
        n_samples = len(self.original_data_df)
        n_clusters = 4
        euclidean_labels = np.zeros(n_samples, dtype=int)
        euclidean_distances = np.zeros((n_samples, n_clusters))

        for i in range(n_samples):
            sample_abr = self.original_data_df.iloc[i][ABR_COLS].values
            for cluster_id in range(n_clusters):
                euclidean_distances[i, cluster_id] = np.linalg.norm(
                    sample_abr - cluster_means_original[cluster_id]
                )
            euclidean_labels[i] = np.argmin(euclidean_distances[i])

        # Convert distances to pseudo-probabilities using softmax
        # Use negative distances (closer = higher probability) with temperature scaling
        euclidean_probs = np.zeros((n_samples, n_clusters))

        # Temperature parameter controls sharpness of probability distribution
        # Lower temperature = sharper distinctions, higher confidence
        temperature = 10.0  # Tune this based on typical distance scales

        for i in range(n_samples):
            # Use negative distances so smaller distance = higher score
            neg_distances = -euclidean_distances[i] / temperature
            # Subtract max for numerical stability
            neg_distances = neg_distances - np.max(neg_distances)
            # Apply softmax
            exp_scores = np.exp(neg_distances)
            euclidean_probs[i] = exp_scores / exp_scores.sum()

        return euclidean_labels, euclidean_probs

    def create_all_visualizations(
        self, output_dir: Optional[Path] = None, dpi: int = 1200
    ):
        """
        Create all standard visualizations.

        Args:
            output_dir: Output directory (defaults to results_dir/figures)
            dpi: Resolution for saved figures
        """
        if output_dir is None:
            output_dir = self.results_dir / "figures"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        logger.info(f"Creating visualizations in {output_dir}")

        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

        # Create each visualization
        plot_files = {}

        # 1. Cluster audiogram profiles
        fig = self.plot_cluster_profiles()
        self._save_figure(fig, output_dir / "cluster_audiogram_profiles", dpi)
        plot_files["audiogram_profiles"] = str(
            output_dir / "cluster_audiogram_profiles.png"
        )

        # 2. PCA visualization - clusters
        fig = self.plot_pca_clusters()
        self._save_figure(fig, output_dir / "cluster_pca_clusters", dpi)
        plot_files["pca_clusters"] = str(output_dir / "cluster_pca_clusters.png")

        # 3. PCA visualization - confidence
        fig = self.plot_pca_confidence()
        self._save_figure(fig, output_dir / "cluster_pca_confidence", dpi)
        plot_files["pca_confidence"] = str(output_dir / "cluster_pca_confidence.png")

        # 4. Cluster distributions
        fig = self.plot_cluster_distributions()
        self._save_figure(fig, output_dir / "cluster_distributions", dpi)
        plot_files["cluster_distributions"] = str(
            output_dir / "cluster_distributions.png"
        )

        # 5. Assignment uncertainty
        fig = self.plot_uncertainty_heatmap()
        self._save_figure(fig, output_dir / "assignment_uncertainty", dpi)
        plot_files["uncertainty_heatmap"] = str(
            output_dir / "assignment_uncertainty.png"
        )

        # 6. Gene associations (if available)
        if (
            "gene_associations" in self.analysis_results
            and self.analysis_results["gene_associations"]
        ):
            fig = self.plot_gene_associations()
            self._save_figure(fig, output_dir / "gene_cluster_associations", dpi)
            plot_files["gene_associations"] = str(
                output_dir / "gene_cluster_associations.png"
            )

        logger.info(f"Created {len(plot_files)} visualizations")
        return plot_files

    def _save_figure(self, fig, base_path: Path, dpi: int = 1200):
        """Save figure in both PNG and EPS formats."""
        png_path = f"{base_path}.png"
        eps_path = f"{base_path}.eps"

        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(eps_path, format="eps", bbox_inches="tight")
        plt.close(fig)

    def plot_cluster_profiles(
        self, use_original_scale: bool = True, use_confidence_intervals: bool = True
    ) -> plt.Figure:
        """Plot mean audiogram profiles for each cluster."""
        data = (
            self.original_data
            if (use_original_scale and self.original_data is not None)
            else self.normalized_data
        )

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        colors = sns.color_palette("husl", self.n_clusters)

        # Individual cluster profiles
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
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

            # Get pattern type from analysis results
            pattern = self.analysis_results["cluster_characteristics"][str(cluster_id)][
                "pattern_type"
            ]

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
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        if use_original_scale and self.original_data is not None:
            ax.set_ylim(0, 100)  # Standard audiogram range: 0-100 dB SPL

        plt.tight_layout()
        return fig

    def plot_pca_clusters(self) -> plt.Figure:
        """Plot clusters in PCA space."""
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(self.normalized_data)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        colors = sns.color_palette("husl", self.n_clusters)

        # Plot colored by cluster
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            pattern = self.analysis_results["cluster_characteristics"][str(cluster_id)][
                "pattern_type"
            ]

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

    def plot_pca_confidence(self) -> plt.Figure:
        """Plot assignment confidence in PCA space."""
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(self.normalized_data)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Plot colored by assignment confidence
        max_probs = self.cluster_probabilities.max(axis=1)
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

    def plot_cluster_distributions(self) -> plt.Figure:
        """Plot cluster size distributions and confidence statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        cluster_sizes = np.bincount(self.cluster_labels)

        # Top left: Cluster sizes
        axes[0, 0].bar(range(self.n_clusters), cluster_sizes)
        axes[0, 0].set_xlabel("Cluster ID")
        axes[0, 0].set_ylabel("Number of Samples")
        axes[0, 0].set_title("Cluster Sizes")
        axes[0, 0].set_xticks(range(self.n_clusters))
        axes[0, 0].set_xticklabels(range(1, self.n_clusters + 1))

        # Pattern labels removed as requested

        # Top right: Cluster proportions pie chart
        patterns = [
            self.analysis_results["cluster_characteristics"][str(i)]["pattern_type"]
            for i in range(self.n_clusters)
        ]
        labels = [f"C{i+1}" for i in range(self.n_clusters)]
        axes[0, 1].pie(cluster_sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        axes[0, 1].set_title("Cluster Proportions")

        # Bottom left: Assignment confidence distribution
        max_probs = self.cluster_probabilities.max(axis=1)
        axes[1, 0].hist(max_probs, bins=30, edgecolor="black", alpha=0.7)
        axes[1, 0].axvline(
            np.mean(max_probs),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(max_probs):.3f}",
        )
        axes[1, 0].set_xlabel("Assignment Confidence")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Distribution of Assignment Confidence")
        axes[1, 0].legend()

        # Bottom right: Confidence by cluster
        confidence_by_cluster = [
            max_probs[self.cluster_labels == i] for i in range(self.n_clusters)
        ]
        axes[1, 1].boxplot(confidence_by_cluster)
        axes[1, 1].set_xlabel("Cluster ID")
        axes[1, 1].set_ylabel("Assignment Confidence")
        axes[1, 1].set_title("Assignment Confidence by Cluster")
        axes[1, 1].set_xticks(range(1, self.n_clusters + 1))
        axes[1, 1].set_xticklabels(range(1, self.n_clusters + 1))

        plt.tight_layout()
        return fig

    def plot_uncertainty_heatmap(self, n_samples: int = 100) -> plt.Figure:
        """Plot assignment uncertainty heatmap for subset of samples."""
        # Sort samples by maximum probability (ascending = most uncertain first)
        max_probs = self.cluster_probabilities.max(axis=1)
        uncertain_idx = np.argsort(max_probs)[:n_samples]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap of probabilities for most uncertain samples
        prob_subset = self.cluster_probabilities[uncertain_idx]

        im = ax.imshow(prob_subset.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

        ax.set_xlabel("Sample Index (sorted by uncertainty)")
        ax.set_ylabel("Cluster")
        ax.set_title(f"Assignment Probabilities for {n_samples} Most Uncertain Samples")
        ax.set_yticks(range(self.n_clusters))
        ax.set_yticklabels(range(1, self.n_clusters + 1))

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Assignment Probability")

        # Add grid
        ax.set_xticks(np.arange(0, n_samples, 10))
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        return fig

    def plot_gene_associations(self, top_n: int = 20) -> plt.Figure:
        """Plot top gene-cluster associations."""
        gene_data = self.analysis_results["gene_associations"]

        if not gene_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No gene association data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        # Extract genes with significant associations
        gene_info = []
        for gene, info in gene_data.items():
            if info["sample_size"] >= 3:  # Minimum sample size
                gene_info.append(
                    {
                        "gene": gene,
                        "dominant_cluster": info["dominant_cluster"],
                        "proportion": info["dominant_proportion"],
                        "p_value": info["enrichment_p_value"],
                        "n": info["sample_size"],
                    }
                )

        gene_df = pd.DataFrame(gene_info)

        # Sort by proportion in dominant cluster
        gene_df = gene_df.sort_values("proportion", ascending=False).head(top_n)

        fig, axes = plt.subplots(1, 2, figsize=(15, 8))

        # Left plot: Bar chart of dominant cluster proportions
        colors = sns.color_palette("husl", self.n_clusters)
        bar_colors = [colors[int(c)] for c in gene_df["dominant_cluster"]]

        axes[0].barh(range(len(gene_df)), gene_df["proportion"], color=bar_colors)
        axes[0].set_yticks(range(len(gene_df)))
        axes[0].set_yticklabels(
            [f"{g} (n={n})" for g, n in zip(gene_df["gene"], gene_df["n"])]
        )
        axes[0].set_xlabel("Proportion in Dominant Cluster")
        axes[0].set_title(f"Top {len(gene_df)} Gene-Cluster Associations")
        axes[0].set_xlim(0, 1)

        # Add cluster labels
        for i, (cluster, prop) in enumerate(
            zip(gene_df["dominant_cluster"], gene_df["proportion"])
        ):
            axes[0].text(prop + 0.01, i, f"C{int(cluster)}", va="center", fontsize=9)

        # Right plot: Heatmap of gene distributions across clusters
        gene_names = gene_df["gene"].head(15)  # Top 15 for readability
        gene_distributions = []

        for gene in gene_names:
            if gene in gene_data:
                gene_distributions.append(gene_data[gene]["cluster_proportions"])

        if gene_distributions:
            gene_distributions = np.array(gene_distributions)

            im = axes[1].imshow(
                gene_distributions, aspect="auto", cmap="Blues", vmin=0, vmax=1
            )
            axes[1].set_yticks(range(len(gene_names)))
            axes[1].set_yticklabels(gene_names)
            axes[1].set_xticks(range(self.n_clusters))
            axes[1].set_xticklabels([f"C{i+1}" for i in range(self.n_clusters)])
            axes[1].set_xlabel("Cluster")
            axes[1].set_title("Gene Distribution Across Clusters")

            cbar = plt.colorbar(im, ax=axes[1])
            cbar.set_label("Proportion")

        plt.tight_layout()
        return fig

    def create_custom_figure(self, figure_type: str, **kwargs) -> plt.Figure:
        """
        Create a specific visualization with custom parameters.

        Args:
            figure_type: Type of figure ('profiles', 'pca_clusters', 'pca_confidence', 'distributions', 'uncertainty', 'genes')
            **kwargs: Additional parameters for the specific plot

        Returns:
            Matplotlib figure
        """
        figure_map = {
            "profiles": self.plot_cluster_profiles,
            "pca_clusters": self.plot_pca_clusters,
            "pca_confidence": self.plot_pca_confidence,
            "distributions": self.plot_cluster_distributions,
            "uncertainty": self.plot_uncertainty_heatmap,
            "genes": self.plot_gene_associations,
        }

        if figure_type not in figure_map:
            raise ValueError(
                f"Unknown figure type: {figure_type}. Choose from {list(figure_map.keys())}"
            )

        return figure_map[figure_type](**kwargs)


def main():
    """Command-line interface for visualization script."""
    parser = argparse.ArgumentParser(
        description="Create visualizations from GMM clustering results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "results_dir", type=str, help="Directory containing pipeline results"
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for figures (default: results_dir/figures)",
    )

    parser.add_argument(
        "--dpi", type=int, default=1200, help="Figure resolution in DPI"
    )

    parser.add_argument(
        "--figure-type",
        type=str,
        choices=[
            "all",
            "profiles",
            "pca_clusters",
            "pca_confidence",
            "distributions",
            "uncertainty",
            "genes",
        ],
        default="all",
        help="Type of figure to create",
    )

    parser.add_argument(
        "--no-original-scale",
        action="store_true",
        help="Use normalized scale instead of original dB SPL scale",
    )

    parser.add_argument(
        "--use-std-dev",
        action="store_true",
        help="Use standard deviation bands instead of 95% confidence intervals (default: CI)",
    )

    parser.add_argument(
        "--assignment-method",
        type=str,
        default="probabilistic",
        choices=["probabilistic", "euclidean"],
        help="Cluster assignment method to use",
    )

    args = parser.parse_args()

    # Initialize visualizer
    visualizer = ResultsVisualizer(args.results_dir, args.assignment_method)

    # Create visualizations
    if args.figure_type == "all":
        plot_files = visualizer.create_all_visualizations(
            output_dir=args.output_dir, dpi=args.dpi
        )
        print(f"Created {len(plot_files)} visualizations:")
        for name, path in plot_files.items():
            print(f"  - {name}: {path}")
    else:
        # Create single figure type
        kwargs = {"use_original_scale": not args.no_original_scale}
        if args.figure_type == "profiles":
            kwargs["use_confidence_intervals"] = not args.use_std_dev

        fig = visualizer.create_custom_figure(args.figure_type, **kwargs)

        # Save figure
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            if args.assignment_method == "euclidean":
                output_dir = Path(args.results_dir) / "figures_euclidean"
            else:
                output_dir = Path(args.results_dir) / "figures"
        output_dir.mkdir(exist_ok=True)

        base_path = output_dir / args.figure_type
        visualizer._save_figure(fig, base_path, args.dpi)

        print(
            f"Created {args.figure_type} visualization: {base_path}.png and {base_path}.eps"
        )


if __name__ == "__main__":
    main()
