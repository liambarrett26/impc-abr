#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced ABR Analysis Pipeline with GMM Clustering

This script implements a comprehensive pipeline for PCA and GMM clustering analysis
of audiograms from the IMPC ABR dataset. It handles data loading, PCA computation,
GMM clustering, and comprehensive visualizations.

author: Liam Barrett
version: 1.1.0
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import from existing IMPC ABR modules
from abr_analysis.data.loader import ABRDataLoader
from abr_analysis.data.matcher import ControlMatcher

# Import analysis modules
from abr_clustering.dimensionality.pca import AudiogramPCA
from abr_clustering.clustering.gmm import AudiogramGMM, select_optimal_clusters


def create_output_dirs(output_dir):
    """Create necessary output directories."""
    base_dir = Path(output_dir)

    # Create subdirectories
    dirs = [
        base_dir,
        base_dir / 'pca_results',
        base_dir / 'cluster_results',
        base_dir / 'gene_results',
        base_dir / 'gene_results' / 'individual_reports',
        base_dir / 'figures'
    ]

    for d in dirs:
        d.mkdir(exist_ok=True, parents=True)

    return base_dir


def load_and_process_data(data_path, min_mutants=3, min_controls=20):
    """
    Load and process ABR data for analysis.

    Parameters:
        data_path (str): Path to the ABR data file.
        min_mutants (int): Minimum number of mutant mice required per gene.
        min_controls (int): Minimum number of control mice required.

    Returns:
        dict: Processed data, including mutant profiles, gene metadata, etc.
    """
    print(f"Loading data from {data_path}...")
    loader = ABRDataLoader(data_path)
    data = loader.load_data()
    matcher = ControlMatcher(data)
    freq_cols = loader.get_frequencies()

    # Identify experimental groups
    mutants = data[data['biological_sample_group'] == 'experimental']
    genes = mutants['gene_symbol'].unique()
    genes = genes[~pd.isna(genes)]  # Remove NaN values

    print(f"Found {len(genes)} unique genes with experimental data")

    # Initialize storage
    mutant_profiles = []
    gene_labels = []
    gene_metadata = []
    failed_genes = []

    # Process each gene
    print("Processing genes and extracting audiograms...")
    for gene in tqdm(genes):
        # Find all experimental groups for this gene
        exp_groups = matcher.find_experimental_groups(gene)

        if not exp_groups:
            failed_genes.append((gene, "No experimental groups found"))
            continue

        # Process each experimental group
        for group in exp_groups:
            try:
                # Find matching controls
                controls = matcher.find_matching_controls(group)

                # Extract profiles
                control_profiles = matcher.get_control_profiles(controls['all'], freq_cols)
                mutant_group_profiles = matcher.get_experimental_profiles(group, freq_cols)

                # Check sample sizes
                if len(mutant_group_profiles) < min_mutants:
                    failed_genes.append((gene, f"Too few mutants: {len(mutant_group_profiles)} < {min_mutants}"))
                    continue

                if len(control_profiles) < min_controls:
                    failed_genes.append((gene, f"Too few controls: {len(control_profiles)} < {min_controls}"))
                    continue

                # Remove any profiles with NaN values
                mutant_group_profiles = mutant_group_profiles[~np.isnan(mutant_group_profiles).any(axis=1)]

                if len(mutant_group_profiles) < min_mutants:
                    failed_genes.append((gene, f"Too few valid mutants after NaN removal: {len(mutant_group_profiles)} < {min_mutants}"))
                    continue

                # Add to the collection
                mutant_profiles.append(mutant_group_profiles)
                gene_labels.extend([gene] * len(mutant_group_profiles))

                # Store metadata for this group
                group_info = {
                    'gene_symbol': gene,
                    'center': group.get('phenotyping_center', 'Unknown'),
                    'zygosity': group.get('zygosity', 'Unknown'),
                    'n_mutants': len(mutant_group_profiles),
                    'n_controls': len(control_profiles)
                }
                gene_metadata.extend([group_info] * len(mutant_group_profiles))

            except Exception as e:
                failed_genes.append((gene, str(e)))

    # Combine all profiles
    if not mutant_profiles:
        raise ValueError("No valid mutant profiles found after processing")

    all_mutant_profiles = np.vstack(mutant_profiles)
    gene_labels = np.array(gene_labels)

    print(f"Successfully processed {len(set(gene_labels))} genes with {len(all_mutant_profiles)} total audiograms")
    print(f"Failed to process {len(failed_genes)} genes")

    return {
        'all_mutant_profiles': all_mutant_profiles,
        'gene_labels': gene_labels,
        'gene_metadata': gene_metadata,
        'freq_cols': freq_cols,
        'failed_genes': failed_genes
    }


def perform_pca_analysis(processed_data, n_components=5, output_dir=None, create_plots=True):
    """
    Perform PCA analysis on audiogram data.

    Parameters:
        processed_data (dict): Processed data from load_and_process_data.
        n_components (int): Number of principal components to compute.
        output_dir (Path): Output directory for figures.
        create_plots (bool): Whether to create visualization plots.

    Returns:
        dict: PCA results, including transformed data.
    """
    print(f"\nPerforming PCA analysis with {n_components} components...")

    # Extract data
    all_mutant_profiles = processed_data['all_mutant_profiles']
    freq_cols = processed_data['freq_cols']

    # Initialize and fit PCA
    pca = AudiogramPCA(n_components=n_components)
    pca_coords = pca.fit_transform(all_mutant_profiles, freq_cols)

    # Save the model
    if output_dir:
        model_path = output_dir / 'pca_results' / 'pca_model.pkl'
        pca.save_model(model_path)
        print(f"PCA model saved to {model_path}")

        # Save PCA coordinates
        coords_path = output_dir / 'pca_results' / 'pca_coordinates.csv'
        coords_df = pd.DataFrame(pca_coords, columns=[f'PC{i+1}' for i in range(n_components)])
        coords_df['gene_symbol'] = processed_data['gene_labels']
        coords_df.to_csv(coords_path, index=False)
        print(f"PCA coordinates saved to {coords_path}")

    # Generate visualizations if requested
    if create_plots and output_dir:
        fig_dir = output_dir / 'figures'

        # Create the requested plots
        print("Creating PCA visualization plots...")

        # 1. Explained variance plot
        pca.plot_explained_variance(save_path=fig_dir / 'explained_variance.png')

        # 2. Component loadings plot
        pca.plot_components(save_path=fig_dir / 'pca_components.png')

        # 3. Component audiogram effects plot
        pca.plot_component_audiograms(save_path=fig_dir / 'component_effects.png')

        # 4. PCA scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_coords[:, 0], pca_coords[:, 1], alpha=0.6, s=30)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio[1]:.2%} variance)')
        plt.title('PCA Scatter Plot of Audiograms')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / 'pca_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Extreme audiograms along each principal component
        for i in range(n_components):
            pca.plot_extreme_audiograms(all_mutant_profiles, pca_coords,
                                       component=i+1, n_examples=5,
                                       save_path=fig_dir / f'extreme_audiograms_pc{i+1}.png')

    return {
        'pca': pca,
        'pca_coords': pca_coords
    }


def perform_gmm_clustering(processed_data, n_clusters=None, max_clusters=10,
                          use_pca=False, pca_components=3, output_dir=None, create_plots=True):
    """
    Perform GMM clustering analysis on audiogram data.

    Parameters:
        processed_data (dict): Processed data from load_and_process_data.
        n_clusters (int, optional): Number of clusters. If None, will be determined automatically.
        max_clusters (int): Maximum number of clusters to test for optimal selection.
        use_pca (bool): Whether to use PCA preprocessing for clustering.
        pca_components (int): Number of PCA components if using PCA.
        output_dir (Path): Output directory for results.
        create_plots (bool): Whether to create visualization plots.

    Returns:
        dict: GMM clustering results.
    """
    print("\nPerforming GMM clustering analysis...")

    # Extract data
    all_mutant_profiles = processed_data['all_mutant_profiles']
    gene_labels = processed_data['gene_labels']
    freq_cols = processed_data['freq_cols']

    # Select optimal number of clusters if not specified
    if n_clusters is None:
        print("Determining optimal number of clusters...")
        optimal_n_clusters, scores_df = select_optimal_clusters(
            all_mutant_profiles, max_clusters=max_clusters, metric='bic'
        )
        print(f"Optimal number of clusters: {optimal_n_clusters}")
        n_clusters = optimal_n_clusters

        # Save cluster selection results
        if output_dir:
            scores_path = output_dir / 'cluster_results' / 'cluster_selection_scores.csv'
            scores_df.to_csv(scores_path, index=False)

            # Plot cluster selection metrics
            if create_plots:
                _, axes = plt.subplots(2, 2, figsize=(12, 10))

                # BIC
                axes[0, 0].plot(scores_df['n_clusters'], scores_df['bic'], 'b-o')
                axes[0, 0].set_title('BIC Score')
                axes[0, 0].set_xlabel('Number of Clusters')
                axes[0, 0].set_ylabel('BIC')
                axes[0, 0].grid(True, alpha=0.3)

                # AIC
                axes[0, 1].plot(scores_df['n_clusters'], scores_df['aic'], 'r-o')
                axes[0, 1].set_title('AIC Score')
                axes[0, 1].set_xlabel('Number of Clusters')
                axes[0, 1].set_ylabel('AIC')
                axes[0, 1].grid(True, alpha=0.3)

                # Silhouette Score
                axes[1, 0].plot(scores_df['n_clusters'], scores_df['silhouette_score'], 'g-o')
                axes[1, 0].set_title('Silhouette Score')
                axes[1, 0].set_xlabel('Number of Clusters')
                axes[1, 0].set_ylabel('Silhouette Score')
                axes[1, 0].grid(True, alpha=0.3)

                # Log Likelihood
                axes[1, 1].plot(scores_df['n_clusters'], scores_df['log_likelihood'], 'm-o')
                axes[1, 1].set_title('Log Likelihood')
                axes[1, 1].set_xlabel('Number of Clusters')
                axes[1, 1].set_ylabel('Log Likelihood')
                axes[1, 1].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(output_dir / 'figures' / 'cluster_selection_metrics.png',
                           dpi=300, bbox_inches='tight')
                plt.close()

    # Fit GMM model
    print(f"Fitting GMM with {n_clusters} clusters...")
    gmm = AudiogramGMM(
        n_clusters=n_clusters,
        use_pca=use_pca,
        pca_components=pca_components
    )
    gmm.fit(all_mutant_profiles, freq_cols)

    # Calculate clustering metrics
    metrics = gmm.calculate_metrics(all_mutant_profiles)
    print(f"Clustering metrics: Silhouette={metrics['silhouette_score']:.3f}, "
          f"BIC={metrics['bic']:.1f}")

    # Analyze cluster patterns
    cluster_analysis = gmm.analyze_cluster_patterns()
    print("\nCluster Analysis:")
    print(cluster_analysis[['cluster_id', 'cluster_size', 'pattern_type', 'severity']].to_string(index=False))

    # Get gene-cluster mapping
    gene_cluster_mapping = gmm.get_gene_cluster_mapping(gene_labels)

    # Save results
    if output_dir:
        # Save GMM model
        model_path = output_dir / 'cluster_results' / 'gmm_model.pkl'
        gmm.save_model(model_path)
        print(f"GMM model saved to {model_path}")

        # Save cluster analysis
        analysis_path = output_dir / 'cluster_results' / 'cluster_summary.csv'
        cluster_analysis.to_csv(analysis_path, index=False)

        # Save gene-cluster mapping
        mapping_path = output_dir / 'gene_results' / 'gene_cluster_mapping.csv'
        gene_cluster_mapping.to_csv(mapping_path, index=False)

        # Create gene summary with cluster information
        gene_summary = create_gene_summary(gene_cluster_mapping, cluster_analysis)
        summary_path = output_dir / 'gene_results' / 'gene_summary.csv'
        gene_summary.to_csv(summary_path, index=False)

    # Generate visualizations if requested
    if create_plots and output_dir:
        fig_dir = output_dir / 'figures'

        print("Creating GMM visualization plots...")

        # 1. Cluster audiograms
        gmm.plot_cluster_audiograms(save_path=fig_dir / 'cluster_audiograms.png')

        # 2. Cluster scatter plot
        gmm.plot_cluster_scatter(all_mutant_profiles, save_path=fig_dir / 'cluster_scatter.png')

        # 3. Pattern distributions
        plot_pattern_distributions(cluster_analysis,
                                 save_path=fig_dir / 'pattern_distributions.png')

        # 4. Gene distributions by cluster
        plot_gene_distributions(gene_cluster_mapping,
                              save_path=fig_dir / 'gene_cluster_distribution.png')

        # 5. Create individual gene reports for top genes by cluster
        create_individual_gene_reports(gene_cluster_mapping, all_mutant_profiles,
                                     freq_cols, output_dir)

    return {
        'gmm': gmm,
        'cluster_analysis': cluster_analysis,
        'gene_cluster_mapping': gene_cluster_mapping,
        'metrics': metrics
    }


def create_gene_summary(gene_cluster_mapping, cluster_analysis):
    """Create a comprehensive gene summary with cluster information."""

    # Group by gene and get cluster statistics
    gene_stats = gene_cluster_mapping.groupby('gene_symbol').agg({
        'cluster_id': ['count', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]],
        'max_probability': ['mean', 'min', 'max']
    }).round(3)

    # Flatten column names
    gene_stats.columns = ['_'.join(col).strip() for col in gene_stats.columns]
    gene_stats = gene_stats.rename(columns={
        'cluster_id_count': 'n_audiograms',
        'cluster_id_<lambda_0>': 'most_common_cluster',  # Fixed lambda name
        'max_probability_mean': 'mean_cluster_probability',
        'max_probability_min': 'min_cluster_probability',
        'max_probability_max': 'max_cluster_probability'
    })

    # Add cluster pattern information
    cluster_info = cluster_analysis.set_index('cluster_id')[['pattern_type', 'severity']]
    gene_stats = gene_stats.join(cluster_info, on='most_common_cluster')

    # Calculate cluster diversity (entropy)
    def calculate_cluster_entropy(gene_data):
        cluster_counts = gene_data['cluster_id'].value_counts(normalize=True)
        entropy = -np.sum(cluster_counts * np.log2(cluster_counts + 1e-10))
        return entropy

    entropy_scores = gene_cluster_mapping.groupby('gene_symbol').apply(calculate_cluster_entropy)
    gene_stats['cluster_entropy'] = entropy_scores

    # Reset index to make gene_symbol a column
    gene_stats = gene_stats.reset_index()

    # Sort by number of audiograms (descending)
    gene_stats = gene_stats.sort_values('n_audiograms', ascending=False)

    return gene_stats


def plot_pattern_distributions(cluster_analysis, save_path=None):
    """Plot distributions of hearing loss patterns."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Cluster sizes
    axes[0, 0].bar(cluster_analysis['cluster_id'], cluster_analysis['cluster_percentage'])
    axes[0, 0].set_title('Cluster Size Distribution')
    axes[0, 0].set_xlabel('Cluster ID')
    axes[0, 0].set_ylabel('Percentage of Audiograms')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Pattern types
    pattern_counts = cluster_analysis['pattern_type'].value_counts()
    axes[0, 1].pie(pattern_counts.values, labels=pattern_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Hearing Loss Pattern Types')

    # 3. Severity distribution
    severity_counts = cluster_analysis['severity'].value_counts()
    axes[1, 0].bar(severity_counts.index, severity_counts.values)
    axes[1, 0].set_title('Hearing Loss Severity Distribution')
    axes[1, 0].set_ylabel('Number of Clusters')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Mean threshold by cluster
    axes[1, 1].bar(cluster_analysis['cluster_id'], cluster_analysis['mean_threshold'])
    axes[1, 1].set_title('Mean Threshold by Cluster')
    axes[1, 1].set_xlabel('Cluster ID')
    axes[1, 1].set_ylabel('Mean Threshold (dB SPL)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return fig


def plot_gene_distributions(gene_cluster_mapping, save_path=None):
    """Plot gene distribution across clusters."""

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Number of genes per cluster
    genes_per_cluster = gene_cluster_mapping.groupby('cluster_id')['gene_symbol'].nunique()
    axes[0].bar(genes_per_cluster.index, genes_per_cluster.values)
    axes[0].set_title('Number of Genes per Cluster')
    axes[0].set_xlabel('Cluster ID')
    axes[0].set_ylabel('Number of Unique Genes')
    axes[0].grid(True, alpha=0.3)

    # 2. Gene cluster probability distribution
    axes[1].hist(gene_cluster_mapping['max_probability'], bins=20, alpha=0.7, edgecolor='black')
    axes[1].set_title('Distribution of Maximum Cluster Probabilities')
    axes[1].set_xlabel('Maximum Cluster Probability')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return fig


def create_individual_gene_reports(gene_cluster_mapping, all_mutant_profiles, freq_cols, output_dir):
    """Create individual reports for genes with the most audiograms in each cluster."""

    # Find top genes per cluster (by number of audiograms)
    top_genes_per_cluster = (gene_cluster_mapping.groupby(['cluster_id', 'gene_symbol'])
                           .size()
                           .reset_index(name='count')
                           .sort_values(['cluster_id', 'count'], ascending=[True, False])
                           .groupby('cluster_id')
                           .head(3))  # Top 3 genes per cluster

    report_dir = output_dir / 'gene_results' / 'individual_reports'

    print(f"Creating individual gene reports for {len(top_genes_per_cluster)} gene-cluster combinations...")

    for _, row in top_genes_per_cluster.iterrows():
        gene = row['gene_symbol']

        # Get gene data
        gene_mask = gene_cluster_mapping['gene_symbol'] == gene
        gene_data = gene_cluster_mapping[gene_mask]
        gene_audiograms = all_mutant_profiles[gene_mask]

        # Create gene-specific directory
        gene_dir = report_dir / gene
        gene_dir.mkdir(exist_ok=True)

        # Create visualizations
        create_gene_visualizations(gene, gene_data, gene_audiograms, freq_cols, gene_dir)

        # Create summary report
        create_gene_summary_report(gene, gene_data, gene_dir)


def create_gene_visualizations(gene, gene_data, gene_audiograms, freq_cols, gene_dir):
    """Create visualizations for a specific gene."""

    # 1. Audiogram plot
    _, ax = plt.subplots(figsize=(10, 6))

    # Plot individual audiograms
    for i, audiogram in enumerate(gene_audiograms):
        cluster_id = gene_data.iloc[i]['cluster_id']
        alpha = 0.3 + 0.4 * gene_data.iloc[i]['max_probability']  # Alpha based on probability
        ax.plot(freq_cols, audiogram, alpha=alpha, color=f'C{cluster_id}', linewidth=1)

    # Plot mean audiogram
    mean_audiogram = np.mean(gene_audiograms, axis=0)
    ax.plot(freq_cols, mean_audiogram, 'k-', linewidth=3, label='Mean')

    ax.set_xlabel('Frequency')
    ax.set_ylabel('Threshold (dB SPL)')
    ax.set_title(f'{gene} - Audiograms (n={len(gene_audiograms)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(gene_dir / f'{gene}_audiograms.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Cluster distribution
    _, ax = plt.subplots(figsize=(8, 6))
    cluster_counts = gene_data['cluster_id'].value_counts().sort_index()
    ax.bar(cluster_counts.index, cluster_counts.values)
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Number of Audiograms')
    ax.set_title(f'{gene} - Cluster Distribution')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(gene_dir / f'{gene}_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. PCA projection (if available)
    if len(gene_audiograms) > 1:
        scaler = StandardScaler()
        audiograms_scaled = scaler.fit_transform(gene_audiograms)
        pca = PCA(n_components=min(2, len(gene_audiograms)-1))
        gene_pca_coords = pca.fit_transform(audiograms_scaled)

        _, ax = plt.subplots(figsize=(8, 6))

        if gene_pca_coords.shape[1] >= 2:
            scatter = ax.scatter(gene_pca_coords[:, 0], gene_pca_coords[:, 1],
                               c=gene_data['cluster_id'], cmap='Set1', s=50, alpha=0.7)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.colorbar(scatter, label='Cluster ID')
        else:
            ax.scatter(gene_pca_coords[:, 0], np.zeros_like(gene_pca_coords[:, 0]),
                      c=gene_data['cluster_id'], cmap='Set1', s=50, alpha=0.7)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel('PC2')

        ax.set_title(f'{gene} - PCA Projection')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(gene_dir / f'{gene}_pca.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_gene_summary_report(gene, gene_data, gene_dir):
    """Create a text summary report for a gene."""

    report_path = gene_dir / f'{gene}_summary.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Gene Summary Report: {gene}\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Number of audiograms: {len(gene_data)}\n")
        f.write("Cluster distribution:\n")

        cluster_counts = gene_data['cluster_id'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            percentage = count / len(gene_data) * 100
            f.write(f"  Cluster {cluster_id}: {count} ({percentage:.1f}%)\n")

        f.write("\nCluster probability statistics:\n")
        f.write(f"  Mean max probability: {gene_data['max_probability'].mean():.3f}\n")
        f.write(f"  Min max probability: {gene_data['max_probability'].min():.3f}\n")
        f.write(f"  Max max probability: {gene_data['max_probability'].max():.3f}\n")


def main(data_path, output_dir, n_components=5, n_clusters=None, max_clusters=10,
         use_pca_for_clustering=False, pca_components=3, create_plots=True):
    """
    Run the enhanced ABR analysis pipeline with PCA and GMM clustering.

    Parameters:
        data_path (str): Path to the ABR data file.
        output_dir (str): Output directory for results.
        n_components (int): Number of principal components for PCA.
        n_clusters (int, optional): Number of clusters for GMM.
        max_clusters (int): Maximum clusters to test for optimal selection.
        use_pca_for_clustering (bool): Whether to use PCA preprocessing for clustering.
        pca_components (int): Number of PCA components for clustering.
        create_plots (bool): Whether to create visualization plots.
    """
    # Create output directories
    output_dir = create_output_dirs(output_dir)

    # Load and process data
    processed_data = load_and_process_data(data_path)

    # Perform PCA analysis
    pca_results = perform_pca_analysis(processed_data, n_components, output_dir, create_plots)

    # Perform GMM clustering analysis
    gmm_results = perform_gmm_clustering(
        processed_data,
        n_clusters=n_clusters,
        max_clusters=max_clusters,
        use_pca=use_pca_for_clustering,
        pca_components=pca_components,
        output_dir=output_dir,
        create_plots=create_plots
    )

    print(f"\nAnalysis complete. Results saved to {output_dir}")
    print(f"Found {gmm_results['cluster_analysis']['cluster_id'].nunique()} distinct hearing loss patterns")
    print(f"Processed {len(set(processed_data['gene_labels']))} unique genes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced ABR Analysis Pipeline with GMM Clustering")

    parser.add_argument("--data", "-d", type=str, required=True,
                       help="Path to the ABR data file")
    parser.add_argument("--output", "-o", type=str, default="./output",
                       help="Output directory for results")
    parser.add_argument("--pca-components", type=int, default=5,
                       help="Number of principal components for PCA")
    parser.add_argument("--clusters", "-c", type=int, default=None,
                       help="Number of clusters for GMM (auto-select if not specified)")
    parser.add_argument("--max-clusters", type=int, default=10,
                       help="Maximum number of clusters to test for auto-selection")
    parser.add_argument("--use-pca-clustering", action="store_true",
                       help="Use PCA preprocessing for clustering")
    parser.add_argument("--pca-clustering-components", type=int, default=3,
                       help="Number of PCA components for clustering preprocessing")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip creating visualization plots")

    args = parser.parse_args()

    main(
        args.data,
        args.output,
        args.pca_components,
        args.clusters,
        args.max_clusters,
        args.use_pca_clustering,
        args.pca_clustering_components,
        not args.no_plots
    )
