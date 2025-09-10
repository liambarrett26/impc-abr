#!/usr/bin/env python3
"""
Mean-based Euclidean distance cluster reassignment for gene/allele combinations.
This version calculates the mean profile for each gene/allele first, then assigns
clusters based on the distance between the mean profile and cluster centers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path("/home/liamb/impc-abr/scripts/abr_clustering/gmm")
OUTPUT_DIR = BASE_DIR / "results" / "euclidean_assignment_mean_based"
PLOT_DIR = OUTPUT_DIR / "cluster_assignment_plots"

# ABR frequency columns
ABR_COLS = [
    '6kHz-evoked ABR Threshold',
    '12kHz-evoked ABR Threshold',
    '18kHz-evoked ABR Threshold',
    '24kHz-evoked ABR Threshold',
    '30kHz-evoked ABR Threshold'
]

FREQ_LABELS = ['6 kHz', '12 kHz', '18 kHz', '24 kHz', '30 kHz']


def load_all_data():
    """Load all necessary data files."""
    print("Loading data files...")
    
    # Load metadata
    metadata = pd.read_csv(BASE_DIR / 'shared_data' / 'metadata.csv')
    
    # Load normalized data
    normalized_data = np.load(BASE_DIR / 'shared_data' / 'normalized_data.npy')
    
    # Load cluster centers from k4_tied model
    cluster_centers = np.load(BASE_DIR / 'results' / 'june_23_2025' / 'gmm_k4_tied' / 'cluster_centers.npy')
    
    # Load original ABR data
    original_data = pd.read_csv(BASE_DIR / 'results' / 'june_23_2025' / 'gmm_k4_tied' / 'original_data.csv')
    
    # Load original gene associations for comparison
    orig_associations = pd.read_csv(BASE_DIR / 'results' / 'gene_cluster_analysis_sex_specific' / 'gene_cluster_associations.csv')
    
    print(f"Loaded data for {len(metadata)} mice")
    return metadata, normalized_data, cluster_centers, original_data, orig_associations


def create_gene_associations_mean_based(metadata, normalized_data, cluster_centers, orig_associations):
    """Create gene cluster associations using mean-based Euclidean distance assignments."""
    
    # Add indices to metadata for easy lookup
    metadata = metadata.copy()
    metadata['data_index'] = range(len(metadata))
    
    mean_based_associations = []
    
    for _, gene_row in orig_associations.iterrows():
        # Extract gene identifiers
        gene_symbol = gene_row['gene_symbol']
        allele_symbol = gene_row['allele_symbol']
        zygosity = gene_row['zygosity']
        phenotyping_center = gene_row['center']
        analysis_type = gene_row['analysis_type']
        
        # Find matching mice
        mask = (
            (metadata['gene_symbol'] == gene_symbol) &
            (metadata['allele_symbol'] == allele_symbol) &
            (metadata['zygosity'] == zygosity) &
            (metadata['phenotyping_center'] == phenotyping_center)
        )
        
        # Apply sex filtering if needed
        if analysis_type == 'male':
            mask = mask & (metadata['sex'] == 'male')
        elif analysis_type == 'female':
            mask = mask & (metadata['sex'] == 'female')
        
        matching_mice = metadata[mask]
        
        if len(matching_mice) > 0:
            # Get normalized data for these mice
            mice_indices = matching_mice['data_index'].values
            mice_normalized = normalized_data[mice_indices]
            
            # Calculate mean normalized profile
            mean_profile = mice_normalized.mean(axis=0)
            
            # Calculate distances from mean profile to each cluster center
            distances_to_centers = np.zeros(len(cluster_centers))
            for i, center in enumerate(cluster_centers):
                distances_to_centers[i] = np.linalg.norm(mean_profile - center)
            
            # Assign to nearest cluster
            dominant_cluster = int(np.argmin(distances_to_centers))
            n_total = len(matching_mice)
            
            # Since we're using mean-based assignment, all mice get the same cluster
            consistency_score = 1.0
            
            # Calculate sex-specific metrics if 'all' analysis
            sex_specific_metrics = {}
            if analysis_type == 'all':
                for sex in ['male', 'female']:
                    sex_mice = matching_mice[matching_mice['sex'] == sex]
                    if len(sex_mice) > 0:
                        sex_specific_metrics.update({
                            f'{sex}_n_mice': len(sex_mice),
                            f'{sex}_dominant_cluster': dominant_cluster,
                            f'{sex}_consistency_score': 1.0,
                            f'{sex}_cluster_0': len(sex_mice) if dominant_cluster == 0 else 0,
                            f'{sex}_cluster_1': len(sex_mice) if dominant_cluster == 1 else 0,
                            f'{sex}_cluster_2': len(sex_mice) if dominant_cluster == 2 else 0,
                            f'{sex}_cluster_3': len(sex_mice) if dominant_cluster == 3 else 0,
                        })
            
            # Create cluster distribution
            cluster_dist = {f'cluster_{i}': n_total if i == dominant_cluster else 0 for i in range(4)}
            
            # Create entry
            entry = {
                'gene_symbol': gene_symbol,
                'allele_symbol': allele_symbol,
                'zygosity': zygosity,
                'center': phenotyping_center,
                'analysis_type': analysis_type,
                'analysis_key': gene_row['analysis_key'],
                'bayes_factor': gene_row['bayes_factor'],
                'p_hearing_loss': gene_row['p_hearing_loss'],
                'n_mutants_reported': gene_row['n_mutants_reported'],
                'n_total_mice': n_total,
                'dominant_cluster_mean_based': dominant_cluster,
                'n_dominant_mean_based': n_total,
                'consistency_score_mean_based': consistency_score,
                **cluster_dist,
                'cluster_distribution': str({i: n_total if i == dominant_cluster else 0 for i in range(4)}),
                'distances_to_clusters': str({i: float(d) for i, d in enumerate(distances_to_centers)}),
                'mean_profile': ','.join([f'{v:.4f}' for v in mean_profile]),
                **sex_specific_metrics,
                # Keep original assignments for comparison
                'dominant_cluster_original': gene_row['dominant_cluster'],
                'consistency_score_original': gene_row['consistency_score'],
                'dominant_cluster_euclidean_individual': gene_row.get('dominant_cluster_euclidean', gene_row['dominant_cluster']),
                'changed_from_original': dominant_cluster != gene_row['dominant_cluster'],
                'changed_from_euclidean': dominant_cluster != gene_row.get('dominant_cluster_euclidean', gene_row['dominant_cluster'])
            }
            
            mean_based_associations.append(entry)
    
    return pd.DataFrame(mean_based_associations)


def plot_gene_abr_profiles_with_distances(gene_data, metadata, original_data, cluster_centers_orig, output_path_base):
    """Create ABR profile plot showing gene mean, cluster centers, and distances."""
    
    # Extract gene info
    gene_symbol = gene_data['gene_symbol']
    allele_symbol = gene_data['allele_symbol']
    zygosity = gene_data['zygosity']
    phenotyping_center = gene_data['center']
    analysis_type = gene_data['analysis_type']
    
    # Find matching mice
    mask = (
        (metadata['gene_symbol'] == gene_symbol) &
        (metadata['allele_symbol'] == allele_symbol) &
        (metadata['zygosity'] == zygosity) &
        (metadata['phenotyping_center'] == phenotyping_center)
    )
    
    if analysis_type == 'male':
        mask = mask & (metadata['sex'] == 'male')
    elif analysis_type == 'female':
        mask = mask & (metadata['sex'] == 'female')
    
    matching_indices = metadata[mask].index
    
    if len(matching_indices) == 0:
        return
    
    # Get data for these mice
    mice_abr = original_data.iloc[matching_indices][ABR_COLS].values
    dominant_cluster = gene_data['dominant_cluster_mean_based']
    n_mice = len(mice_abr)
    
    # Parse distances
    distances_str = gene_data['distances_to_clusters']
    distances = eval(distances_str)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Set style to match visualize_results.py
    sns.set_palette("husl")
    cluster_colors = sns.color_palette("husl", 4)
    
    # Plot cluster averages (using original scale)
    for cluster_id in range(4):
        label = f'Cluster {cluster_id} (d={distances[cluster_id]:.3f})'
        if cluster_id == dominant_cluster:
            label += ' ✓'
            linewidth = 3
            alpha = 1.0
        else:
            linewidth = 1.5
            alpha = 0.5
        
        # We need to plot cluster centers in original scale
        # For now, using the cluster means from the plot
        ax.plot(FREQ_LABELS, cluster_centers_orig[cluster_id], 
                color=cluster_colors[cluster_id], linewidth=linewidth, 
                alpha=alpha, label=label, marker='o', markersize=6)
    
    # Plot individual mice in black dashed lines
    for i in range(len(mice_abr)):
        ax.plot(FREQ_LABELS, mice_abr[i], 
                color='black', alpha=0.7, 
                linewidth=1, linestyle='--', marker='s', markersize=3)
    
    # Add gene mean profile (thick black dashed line)
    gene_mean = mice_abr.mean(axis=0)
    ax.plot(FREQ_LABELS, gene_mean, color='black', linestyle='--', linewidth=3, 
            label=f'{gene_symbol} mean', marker='D', markersize=8)
    
    # Formatting
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('ABR Threshold (dB SPL)', fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Title with cluster assignment
    title = f'{gene_symbol} ({allele_symbol}) - {zygosity} - {phenotyping_center}'
    if analysis_type != 'all':
        title += f' - {analysis_type}'
    title += f'\\nCluster {dominant_cluster} (mean-based) - N: {n_mice}'
    
    ax.set_title(title, fontsize=12, pad=10)
    
    # Save in multiple formats
    plt.tight_layout()
    
    # Get base filename without extension
    base_path = Path(output_path_base).with_suffix('')
    
    # Save high-res PNG (1200 DPI)
    fig.savefig(f"{base_path}_highres.png", dpi=1200, bbox_inches='tight')
    
    # Save low-res PNG (150 DPI)
    fig.savefig(f"{base_path}.png", dpi=150, bbox_inches='tight')
    
    # Save EPS (vector format, high quality)
    fig.savefig(f"{base_path}.eps", format='eps', bbox_inches='tight', dpi=1200)
    
    plt.close(fig)


def calculate_cluster_averages_original(original_data, cluster_labels):
    """Calculate average ABR profiles for each cluster in original scale."""
    cluster_means = {}
    for cluster_id in range(4):
        cluster_mask = cluster_labels == cluster_id
        if cluster_mask.sum() > 0:
            cluster_data = original_data.iloc[cluster_mask][ABR_COLS]
            cluster_means[cluster_id] = cluster_data.mean().values
        else:
            # Use overall mean if no samples in cluster
            cluster_means[cluster_id] = original_data[ABR_COLS].mean().values
    return cluster_means


def main():
    """Main execution function."""
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    metadata, normalized_data, cluster_centers, original_data, orig_associations = load_all_data()
    
    print("Creating mean-based gene associations...")
    mean_based_associations_df = create_gene_associations_mean_based(
        metadata, normalized_data, cluster_centers, orig_associations)
    
    # Save associations CSV
    mean_based_associations_df.to_csv(OUTPUT_DIR / 'gene_cluster_associations_mean_based.csv', index=False)
    print(f"Saved associations to {OUTPUT_DIR / 'gene_cluster_associations_mean_based.csv'}")
    
    # Load previous euclidean results for comparison
    euclidean_df = pd.read_csv(BASE_DIR / 'results' / 'euclidean_assignment' / 'gene_cluster_associations_euclidean.csv')
    
    # Merge to get euclidean assignments
    mean_based_associations_df = mean_based_associations_df.merge(
        euclidean_df[['analysis_key', 'dominant_cluster_euclidean']], 
        on='analysis_key', 
        how='left',
        suffixes=('', '_prev')
    )
    
    # Update the changed_from_euclidean field
    mean_based_associations_df['changed_from_euclidean'] = (
        mean_based_associations_df['dominant_cluster_mean_based'] != 
        mean_based_associations_df['dominant_cluster_euclidean']
    )
    
    # Save updated associations
    mean_based_associations_df.to_csv(OUTPUT_DIR / 'gene_cluster_associations_mean_based.csv', index=False)
    
    # Get cluster means in original scale
    # First, assign all samples using individual euclidean for cluster means calculation
    euclidean_labels = np.load(BASE_DIR / 'results' / 'june_23_2025' / 'gmm_k4_tied' / 'cluster_labels.npy')
    cluster_means_original = calculate_cluster_averages_original(original_data, euclidean_labels)
    
    # Create plots
    print(f"Creating plots for {len(mean_based_associations_df)} gene/allele combinations...")
    
    # Specifically check the problem genes
    problem_genes = ['Bap1', 'Fbxo38', 'Kdm5a', 'Ragbatta', 'Lsm14a', 'Defb43', 'Nin']
    
    for idx, gene_data in mean_based_associations_df.iterrows():
        # Create filename
        gene_symbol = gene_data['gene_symbol']
        allele_short = gene_data['allele_symbol'].replace('<', '_').replace('>', '_')
        zygosity = gene_data['zygosity']
        center = gene_data['center']
        analysis_type = gene_data['analysis_type']
        
        filename_base = f"{gene_symbol}_{allele_short}_{zygosity}_{center}_{analysis_type}"
        output_path_base = PLOT_DIR / filename_base
        
        try:
            plot_gene_abr_profiles_with_distances(gene_data, metadata, original_data, 
                                                cluster_means_original, output_path_base)
            
            # Report on problem genes
            if gene_symbol in problem_genes:
                print(f"  {gene_symbol}: Cluster {gene_data['dominant_cluster_mean_based']} "
                      f"(was {gene_data['dominant_cluster_original']}, "
                      f"euclidean individual: {gene_data.get('dominant_cluster_euclidean', 'N/A')})")
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1} / {len(mean_based_associations_df)} plots...")
        except Exception as e:
            print(f"  Error plotting {gene_symbol}: {str(e)}")
    
    print(f"\\nCompleted! Results saved to {OUTPUT_DIR}")
    
    # Create summary report
    summary = {
        'total_genes_analyzed': len(mean_based_associations_df),
        'changed_from_original': int(mean_based_associations_df['changed_from_original'].sum()),
        'changed_from_euclidean': int(mean_based_associations_df['changed_from_euclidean'].sum()),
        'problem_genes_analysis': {}
    }
    
    # Analyze problem genes
    for gene in problem_genes:
        gene_data = mean_based_associations_df[mean_based_associations_df['gene_symbol'] == gene]
        if not gene_data.empty:
            row = gene_data.iloc[0]
            summary['problem_genes_analysis'][gene] = {
                'original_cluster': int(row['dominant_cluster_original']),
                'euclidean_individual': int(row.get('dominant_cluster_euclidean', row['dominant_cluster_original'])),
                'mean_based': int(row['dominant_cluster_mean_based']),
                'distances': eval(row['distances_to_clusters'])
            }
    
    with open(OUTPUT_DIR / 'mean_based_assignment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\\nSummary:")
    print(f"  Total genes analyzed: {summary['total_genes_analyzed']}")
    print(f"  Changed from original: {summary['changed_from_original']}")
    print(f"  Changed from euclidean individual: {summary['changed_from_euclidean']}")
    print("\\nProblem genes reassignment:")
    for gene, info in summary['problem_genes_analysis'].items():
        print(f"  {gene}: {info['original_cluster']} → {info['euclidean_individual']} → {info['mean_based']}")


if __name__ == "__main__":
    main()