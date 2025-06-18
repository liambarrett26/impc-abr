"""
Visualization tools for ContrastiveVAE-DEC clustering results.

This module provides comprehensive visualization capabilities for analyzing
audiometric phenotype clusters, including dimensionality reduction plots,
cluster analysis, and biological interpretation visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path

# Dimensionality reduction
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.decomposition import PCA

# Plotting
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class DimensionalityReductionVisualizer:
    """
    Visualizer for high-dimensional embeddings using various reduction techniques.

    Provides t-SNE, UMAP, and PCA visualizations of latent representations
    with cluster and gene label overlays.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize dimensionality reduction visualizer.

        Args:
            random_state: Random state for reproducible results
        """
        self.random_state = random_state
        self.reduction_methods = {}

    def fit_reductions(self, embeddings: np.ndarray,
                      methods: List[str] = ['pca', 'tsne', 'umap']) -> Dict[str, np.ndarray]:
        """
        Fit dimensionality reduction methods on embeddings.

        Args:
            embeddings: High-dimensional embeddings (n_samples, n_features)
            methods: List of reduction methods to use

        Returns:
            Dictionary of method names to 2D embeddings
        """
        logger.info(f"Fitting dimensionality reduction methods: {methods}")

        reductions = {}

        if 'pca' in methods:
            pca = PCA(n_components=2, random_state=self.random_state)
            reductions['pca'] = pca.fit_transform(embeddings)
            self.reduction_methods['pca'] = pca

        if 'tsne' in methods:
            # Use PCA pre-processing for t-SNE if high dimensional
            if embeddings.shape[1] > 50:
                pca_50 = PCA(n_components=50, random_state=self.random_state)
                embeddings_reduced = pca_50.fit_transform(embeddings)
            else:
                embeddings_reduced = embeddings

            tsne = TSNE(n_components=2, random_state=self.random_state,
                       perplexity=min(30, len(embeddings) // 4))
            reductions['tsne'] = tsne.fit_transform(embeddings_reduced)
            self.reduction_methods['tsne'] = tsne

        if 'umap' in methods:
            umap_reducer = UMAP(n_components=2, random_state=self.random_state,
                               n_neighbors=min(15, len(embeddings) // 3))
            reductions['umap'] = umap_reducer.fit_transform(embeddings)
            self.reduction_methods['umap'] = umap_reducer

        return reductions

    def plot_cluster_embeddings(self, reductions: Dict[str, np.ndarray],
                               cluster_labels: np.ndarray,
                               gene_labels: Optional[np.ndarray] = None,
                               save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot cluster embeddings using multiple reduction methods.

        Args:
            reductions: Dictionary of reduction method results
            cluster_labels: Cluster assignments
            gene_labels: Gene labels for coloring (optional)
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        n_methods = len(reductions)
        n_plots = n_methods * 2 if gene_labels is not None else n_methods

        fig, axes = plt.subplots(2 if gene_labels is not None else 1, n_methods,
                                figsize=(5 * n_methods, 10 if gene_labels is not None else 5))

        if n_methods == 1:
            axes = [axes] if gene_labels is None else [[axes[0]], [axes[1]]]
        elif gene_labels is None:
            axes = [axes]

        # Color palettes
        n_clusters = len(np.unique(cluster_labels))
        cluster_colors = sns.color_palette("husl", n_clusters)

        for i, (method, embedding) in enumerate(reductions.items()):
            # Plot clusters
            ax_cluster = axes[0][i]

            for cluster_id in np.unique(cluster_labels):
                mask = cluster_labels == cluster_id
                ax_cluster.scatter(embedding[mask, 0], embedding[mask, 1],
                                 c=[cluster_colors[cluster_id]],
                                 label=f'Cluster {cluster_id}',
                                 alpha=0.7, s=30)

            ax_cluster.set_title(f'{method.upper()} - Clusters')
            ax_cluster.set_xlabel(f'{method.upper()} 1')
            ax_cluster.set_ylabel(f'{method.upper()} 2')
            ax_cluster.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Plot genes if available
            if gene_labels is not None:
                ax_gene = axes[1][i]

                # Color by gene (show top N most frequent genes)
                unique_genes, counts = np.unique(gene_labels[gene_labels != -1], return_counts=True)
                top_genes = unique_genes[np.argsort(counts)[-10:]]  # Top 10 genes

                gene_colors = sns.color_palette("tab10", len(top_genes))

                # Plot top genes
                for j, gene in enumerate(top_genes):
                    mask = gene_labels == gene
                    if np.sum(mask) > 0:
                        ax_gene.scatter(embedding[mask, 0], embedding[mask, 1],
                                      c=[gene_colors[j]], label=f'Gene {gene}',
                                      alpha=0.7, s=30)

                # Plot unknown genes in gray
                unknown_mask = gene_labels == -1
                if np.sum(unknown_mask) > 0:
                    ax_gene.scatter(embedding[unknown_mask, 0], embedding[unknown_mask, 1],
                                  c='lightgray', label='Unknown', alpha=0.3, s=20)

                ax_gene.set_title(f'{method.upper()} - Genes')
                ax_gene.set_xlabel(f'{method.upper()} 1')
                ax_gene.set_ylabel(f'{method.upper()} 2')
                ax_gene.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cluster embedding plot: {save_path}")

        return fig


class ClusterAnalysisVisualizer:
    """
    Visualizer for detailed cluster analysis and characterization.

    Provides plots for cluster statistics, quality metrics, and
    audiometric pattern analysis.
    """

    def __init__(self):
        """Initialize cluster analysis visualizer."""
        pass

    def plot_cluster_statistics(self, cluster_labels: np.ndarray,
                               embeddings: np.ndarray,
                               save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot basic cluster statistics and quality metrics.

        Args:
            cluster_labels: Cluster assignments
            embeddings: Latent embeddings
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Cluster size distribution
        cluster_sizes = np.bincount(cluster_labels)
        axes[0, 0].bar(range(len(cluster_sizes)), cluster_sizes)
        axes[0, 0].set_title('Cluster Size Distribution')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Number of Samples')

        # Cluster size pie chart
        axes[0, 1].pie(cluster_sizes, labels=[f'C{i}' for i in range(len(cluster_sizes))],
                      autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Cluster Size Proportions')

        # Within-cluster distances
        within_cluster_distances = []
        cluster_ids = []

        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            cluster_points = embeddings[mask]

            if len(cluster_points) > 1:
                center = np.mean(cluster_points, axis=0)
                distances = np.linalg.norm(cluster_points - center, axis=1)
                within_cluster_distances.extend(distances)
                cluster_ids.extend([cluster_id] * len(distances))

        if within_cluster_distances:
            cluster_distance_df = pd.DataFrame({
                'cluster': cluster_ids,
                'distance': within_cluster_distances
            })

            sns.boxplot(data=cluster_distance_df, x='cluster', y='distance', ax=axes[1, 0])
            axes[1, 0].set_title('Within-Cluster Distance Distributions')
            axes[1, 0].set_xlabel('Cluster ID')
            axes[1, 0].set_ylabel('Distance to Centroid')

        # Cluster compactness vs separation
        cluster_compactness = []
        cluster_separation = []

        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            cluster_points = embeddings[mask]

            if len(cluster_points) > 1:
                # Compactness: average within-cluster distance
                center = np.mean(cluster_points, axis=0)
                distances = np.linalg.norm(cluster_points - center, axis=1)
                compactness = np.mean(distances)
                cluster_compactness.append(compactness)

                # Separation: distance to nearest other cluster center
                other_centers = []
                for other_id in np.unique(cluster_labels):
                    if other_id != cluster_id:
                        other_mask = cluster_labels == other_id
                        other_center = np.mean(embeddings[other_mask], axis=0)
                        other_centers.append(other_center)

                if other_centers:
                    min_separation = min([np.linalg.norm(center - other_center)
                                        for other_center in other_centers])
                    cluster_separation.append(min_separation)
                else:
                    cluster_separation.append(0)

        if cluster_compactness and cluster_separation:
            axes[1, 1].scatter(cluster_compactness, cluster_separation)
            for i, cluster_id in enumerate(np.unique(cluster_labels)):
                if i < len(cluster_compactness):
                    axes[1, 1].annotate(f'C{cluster_id}',
                                       (cluster_compactness[i], cluster_separation[i]))

            axes[1, 1].set_xlabel('Compactness (lower is better)')
            axes[1, 1].set_ylabel('Separation (higher is better)')
            axes[1, 1].set_title('Cluster Compactness vs Separation')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cluster statistics plot: {save_path}")

        return fig

    def plot_audiogram_patterns(self, cluster_labels: np.ndarray,
                               abr_features: np.ndarray,
                               frequency_labels: List[str] = None,
                               save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot audiogram patterns for each cluster.

        Args:
            cluster_labels: Cluster assignments
            abr_features: ABR threshold features (n_samples, n_frequencies)
            frequency_labels: Labels for frequency axes
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        if frequency_labels is None:
            frequency_labels = ['6kHz', '12kHz', '18kHz', '24kHz', '30kHz', 'Click']

        n_clusters = len(np.unique(cluster_labels))
        n_cols = min(3, n_clusters)
        n_rows = (n_clusters + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_clusters == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]

        for i, cluster_id in enumerate(np.unique(cluster_labels)):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row][col] if n_rows > 1 else axes[col]

            # Get cluster data
            mask = cluster_labels == cluster_id
            cluster_abr = abr_features[mask]

            if len(cluster_abr) > 0:
                # Plot individual audiograms
                for audiogram in cluster_abr:
                    ax.plot(frequency_labels, audiogram, 'o-', alpha=0.3, color='lightblue')

                # Plot mean audiogram
                mean_audiogram = np.mean(cluster_abr, axis=0)
                std_audiogram = np.std(cluster_abr, axis=0)

                ax.plot(frequency_labels, mean_audiogram, 'o-', linewidth=3,
                       color='darkblue', label=f'Mean (n={len(cluster_abr)})')

                # Error bars
                ax.errorbar(frequency_labels, mean_audiogram, yerr=std_audiogram,
                           capsize=5, capthick=2, color='darkblue', alpha=0.7)

                ax.set_title(f'Cluster {cluster_id} Audiograms')
                ax.set_xlabel('Frequency')
                ax.set_ylabel('Threshold (dB SPL)')
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)
                ax.legend()

                # Invert y-axis (lower thresholds = better hearing)
                ax.invert_yaxis()

        # Hide empty subplots
        for i in range(n_clusters, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row][col].set_visible(False)
            elif n_cols > 1:
                axes[col].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved audiogram patterns plot: {save_path}")

        return fig


class InteractiveVisualizer:
    """
    Interactive visualizations using Plotly for web-based exploration.

    Provides interactive cluster exploration, hover information,
    and dynamic filtering capabilities.
    """

    def __init__(self):
        """Initialize interactive visualizer."""
        pass

    def create_interactive_cluster_plot(self, embeddings_2d: np.ndarray,
                                      cluster_labels: np.ndarray,
                                      gene_labels: Optional[np.ndarray] = None,
                                      abr_features: Optional[np.ndarray] = None,
                                      mouse_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create interactive cluster plot with hover information.

        Args:
            embeddings_2d: 2D embeddings for plotting
            cluster_labels: Cluster assignments
            gene_labels: Gene labels (optional)
            abr_features: ABR features for hover info (optional)
            mouse_ids: Mouse IDs for hover info (optional)

        Returns:
            Plotly figure
        """
        # Prepare hover information
        hover_data = []
        for i in range(len(embeddings_2d)):
            hover_info = f"Mouse: {mouse_ids[i] if mouse_ids else i}<br>"
            hover_info += f"Cluster: {cluster_labels[i]}<br>"

            if gene_labels is not None:
                hover_info += f"Gene: {gene_labels[i]}<br>"

            if abr_features is not None:
                hover_info += "ABR Thresholds:<br>"
                freqs = ['6kHz', '12kHz', '18kHz', '24kHz', '30kHz', 'Click']
                for j, freq in enumerate(freqs[:abr_features.shape[1]]):
                    hover_info += f"  {freq}: {abr_features[i, j]:.1f} dB<br>"

            hover_data.append(hover_info)

        # Create scatter plot
        fig = go.Figure()

        # Add points colored by cluster
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id

            fig.add_trace(go.Scatter(
                x=embeddings_2d[mask, 0],
                y=embeddings_2d[mask, 1],
                mode='markers',
                name=f'Cluster {cluster_id}',
                text=[hover_data[i] for i in np.where(mask)[0]],
                hovertemplate='%{text}<extra></extra>',
                marker=dict(
                    size=8,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                )
            ))

        fig.update_layout(
            title='Interactive Cluster Visualization',
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            hovermode='closest',
            height=600,
            showlegend=True
        )

        return fig

    def create_cluster_comparison_dashboard(self, cluster_labels: np.ndarray,
                                          abr_features: np.ndarray,
                                          gene_labels: Optional[np.ndarray] = None) -> go.Figure:
        """
        Create interactive dashboard comparing clusters.

        Args:
            cluster_labels: Cluster assignments
            abr_features: ABR features
            gene_labels: Gene labels (optional)

        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Cluster Audiograms', 'Cluster Sizes',
                          'Hearing Loss Severity', 'Gene Distribution'],
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"secondary_y": False}, {"type": "bar"}]]
        )

        # Audiogram comparison
        frequency_labels = ['6kHz', '12kHz', '18kHz', '24kHz', '30kHz', 'Click']

        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            cluster_abr = abr_features[mask]

            if len(cluster_abr) > 0:
                mean_audiogram = np.mean(cluster_abr, axis=0)

                fig.add_trace(
                    go.Scatter(
                        x=frequency_labels[:len(mean_audiogram)],
                        y=mean_audiogram,
                        mode='lines+markers',
                        name=f'Cluster {cluster_id}',
                        line=dict(width=3),
                        showlegend=True
                    ),
                    row=1, col=1
                )

        # Cluster sizes
        cluster_sizes = np.bincount(cluster_labels)
        fig.add_trace(
            go.Pie(
                labels=[f'Cluster {i}' for i in range(len(cluster_sizes))],
                values=cluster_sizes,
                name="Cluster Sizes",
                showlegend=False
            ),
            row=1, col=2
        )

        # Hearing loss severity
        severity_data = []
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            cluster_abr = abr_features[mask]

            if len(cluster_abr) > 0:
                avg_thresholds = np.mean(cluster_abr, axis=1)
                severity_data.extend([(cluster_id, threshold) for threshold in avg_thresholds])

        if severity_data:
            severity_df = pd.DataFrame(severity_data, columns=['Cluster', 'Avg_Threshold'])

            for cluster_id in np.unique(cluster_labels):
                cluster_severities = severity_df[severity_df['Cluster'] == cluster_id]['Avg_Threshold']

                fig.add_trace(
                    go.Box(
                        y=cluster_severities,
                        name=f'Cluster {cluster_id}',
                        showlegend=False
                    ),
                    row=2, col=1
                )

        # Gene distribution (if available)
        if gene_labels is not None:
            gene_cluster_counts = {}
            for cluster_id in np.unique(cluster_labels):
                mask = cluster_labels == cluster_id
                cluster_genes = gene_labels[mask]
                valid_genes = cluster_genes[cluster_genes != -1]

                if len(valid_genes) > 0:
                    unique_genes, counts = np.unique(valid_genes, return_counts=True)
                    gene_cluster_counts[cluster_id] = len(unique_genes)
                else:
                    gene_cluster_counts[cluster_id] = 0

            fig.add_trace(
                go.Bar(
                    x=list(gene_cluster_counts.keys()),
                    y=list(gene_cluster_counts.values()),
                    name="Unique Genes per Cluster",
                    showlegend=False
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title_text="Cluster Comparison Dashboard",
            height=800,
            showlegend=True
        )

        # Update y-axis for audiograms (invert for hearing thresholds)
        fig.update_yaxes(autorange="reversed", row=1, col=1)
        fig.update_xaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Threshold (dB SPL)", row=1, col=1)

        fig.update_yaxes(title_text="Average Threshold (dB SPL)", row=2, col=1)
        fig.update_xaxes(title_text="Cluster", row=2, col=1)

        fig.update_xaxes(title_text="Cluster", row=2, col=2)
        fig.update_yaxes(title_text="Number of Unique Genes", row=2, col=2)

        return fig


def create_comprehensive_visualization_report(embeddings: np.ndarray,
                                            cluster_labels: np.ndarray,
                                            abr_features: np.ndarray,
                                            gene_labels: Optional[np.ndarray] = None,
                                            mouse_ids: Optional[List[str]] = None,
                                            save_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Create comprehensive visualization report with all major plots.

    Args:
        embeddings: High-dimensional embeddings
        cluster_labels: Cluster assignments
        abr_features: ABR threshold features
        gene_labels: Gene labels (optional)
        mouse_ids: Mouse identifiers (optional)
        save_dir: Directory to save plots

    Returns:
        Dictionary of created figures and file paths
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    report = {}

    # Dimensionality reduction plots
    dim_reducer = DimensionalityReductionVisualizer()
    reductions = dim_reducer.fit_reductions(embeddings)

    cluster_embedding_fig = dim_reducer.plot_cluster_embeddings(
        reductions, cluster_labels, gene_labels,
        save_path=save_dir / 'cluster_embeddings.png' if save_dir else None
    )
    report['cluster_embeddings'] = cluster_embedding_fig

    # Cluster analysis plots
    cluster_analyzer = ClusterAnalysisVisualizer()

    cluster_stats_fig = cluster_analyzer.plot_cluster_statistics(
        cluster_labels, list(reductions.values())[0],  # Use first reduction method
        save_path=save_dir / 'cluster_statistics.png' if save_dir else None
    )
    report['cluster_statistics'] = cluster_stats_fig

    audiogram_fig = cluster_analyzer.plot_audiogram_patterns(
        cluster_labels, abr_features,
        save_path=save_dir / 'audiogram_patterns.png' if save_dir else None
    )
    report['audiogram_patterns'] = audiogram_fig

    # Interactive visualizations
    interactive_viz = InteractiveVisualizer()

    # Use t-SNE for interactive plot if available, otherwise use first method
    embedding_2d = reductions.get('tsne', list(reductions.values())[0])

    interactive_cluster_fig = interactive_viz.create_interactive_cluster_plot(
        embedding_2d, cluster_labels, gene_labels, abr_features, mouse_ids
    )
    report['interactive_clusters'] = interactive_cluster_fig

    if save_dir:
        interactive_cluster_fig.write_html(save_dir / 'interactive_clusters.html')

    dashboard_fig = interactive_viz.create_cluster_comparison_dashboard(
        cluster_labels, abr_features, gene_labels
    )
    report['cluster_dashboard'] = dashboard_fig

    if save_dir:
        dashboard_fig.write_html(save_dir / 'cluster_dashboard.html')

    logger.info(f"Created comprehensive visualization report with {len(report)} figures")

    return report