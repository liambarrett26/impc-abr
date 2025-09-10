#!/usr/bin/env python3
"""
Euclidean distance-based cluster reassignment and visualization for gene/allele combinations.
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
OUTPUT_DIR = BASE_DIR / "results" / "euclidean_assignment"
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


def assign_clusters_euclidean(normalized_data, cluster_centers):
    """Assign clusters based on minimum Euclidean distance."""
    n_samples = normalized_data.shape[0]
    n_clusters = cluster_centers.shape[0]
    
    # Calculate distances to all cluster centers
    distances = np.zeros((n_samples, n_clusters))
    for i in range(n_clusters):
        distances[:, i] = np.linalg.norm(normalized_data - cluster_centers[i], axis=1)
    
    # Assign to nearest cluster
    euclidean_labels = np.argmin(distances, axis=1)
    
    return euclidean_labels, distances


def calculate_cluster_averages_original(original_data, cluster_labels):
    """Calculate average ABR profiles for each cluster in original scale."""
    cluster_means = {}
    for cluster_id in range(4):
        cluster_mask = cluster_labels == cluster_id
        cluster_data = original_data.iloc[cluster_mask][ABR_COLS]
        cluster_means[cluster_id] = cluster_data.mean().values
    return cluster_means


def create_gene_associations_euclidean(metadata, euclidean_labels, orig_associations):
    """Create gene cluster associations using Euclidean distance assignments."""
    
    # Add Euclidean cluster assignments to metadata
    metadata = metadata.copy()
    metadata['euclidean_cluster'] = euclidean_labels
    
    # Process each gene from original associations
    euclidean_associations = []
    
    for _, gene_row in orig_associations.iterrows():
        # Extract gene identifiers
        gene_symbol = gene_row['gene_symbol']
        allele_symbol = gene_row['allele_symbol']
        zygosity = gene_row['zygosity']
        center = gene_row['center']
        analysis_type = gene_row['analysis_type']
        
        # Find matching mice
        mask = (
            (metadata['gene_symbol'] == gene_symbol) &
            (metadata['allele_symbol'] == allele_symbol) &
            (metadata['zygosity'] == zygosity) &
            (metadata['phenotyping_center'] == center)
        )
        
        # Apply sex filtering if needed
        if analysis_type == 'male':
            mask = mask & (metadata['sex'] == 'male')
        elif analysis_type == 'female':
            mask = mask & (metadata['sex'] == 'female')
        
        matching_mice = metadata[mask]
        
        if len(matching_mice) > 0:
            # Get Euclidean cluster assignments
            euclidean_clusters = matching_mice['euclidean_cluster'].values
            cluster_counts = Counter(euclidean_clusters)
            
            # Find dominant cluster
            dominant_cluster = max(cluster_counts, key=cluster_counts.get)
            n_dominant = cluster_counts[dominant_cluster]
            n_total = len(euclidean_clusters)
            consistency_score = n_dominant / n_total
            
            # Calculate distribution
            cluster_dist = {f'cluster_{i}': cluster_counts.get(i, 0) for i in range(4)}
            
            # Calculate sex-specific metrics if 'all' analysis
            sex_specific_metrics = {}
            if analysis_type == 'all':
                for sex in ['male', 'female']:
                    sex_mice = matching_mice[matching_mice['sex'] == sex]
                    if len(sex_mice) > 0:
                        sex_clusters = sex_mice['euclidean_cluster'].values
                        sex_counts = Counter(sex_clusters)
                        sex_dominant = max(sex_counts, key=sex_counts.get)
                        sex_consistency = sex_counts[sex_dominant] / len(sex_clusters)
                        
                        sex_specific_metrics.update({
                            f'{sex}_n_mice': len(sex_mice),
                            f'{sex}_dominant_cluster': sex_dominant,
                            f'{sex}_consistency_score': sex_consistency,
                            f'{sex}_cluster_0': sex_counts.get(0, 0),
                            f'{sex}_cluster_1': sex_counts.get(1, 0),
                            f'{sex}_cluster_2': sex_counts.get(2, 0),
                            f'{sex}_cluster_3': sex_counts.get(3, 0),
                        })
            
            # Create entry (copying most fields from original)
            entry = {
                'gene_symbol': gene_symbol,
                'allele_symbol': allele_symbol,
                'zygosity': zygosity,
                'center': center,
                'analysis_type': analysis_type,
                'analysis_key': gene_row['analysis_key'],
                'bayes_factor': gene_row['bayes_factor'],
                'p_hearing_loss': gene_row['p_hearing_loss'],
                'n_mutants_reported': gene_row['n_mutants_reported'],
                'n_total_mice': n_total,
                'dominant_cluster_euclidean': dominant_cluster,
                'n_dominant_euclidean': n_dominant,
                'consistency_score_euclidean': consistency_score,
                **cluster_dist,
                'cluster_distribution': str({i: cluster_counts.get(i, 0) for i in range(4)}),
                **sex_specific_metrics,
                # Keep original assignments for comparison
                'dominant_cluster_original': gene_row['dominant_cluster'],
                'consistency_score_original': gene_row['consistency_score'],
                'changed_assignment': dominant_cluster != gene_row['dominant_cluster']
            }
            
            euclidean_associations.append(entry)
    
    return pd.DataFrame(euclidean_associations)


def plot_gene_abr_profiles(gene_data, metadata, original_data, euclidean_labels, cluster_means_original, output_path_base):
    """Create ABR profile plot for a specific gene/allele combination."""
    
    # Extract gene info
    gene_symbol = gene_data['gene_symbol']
    allele_symbol = gene_data['allele_symbol']
    zygosity = gene_data['zygosity']
    center = gene_data['center']
    analysis_type = gene_data['analysis_type']
    
    # Find matching mice
    mask = (
        (metadata['gene_symbol'] == gene_symbol) &
        (metadata['allele_symbol'] == allele_symbol) &
        (metadata['zygosity'] == zygosity) &
        (metadata['phenotyping_center'] == center)
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
    mice_clusters = euclidean_labels[matching_indices]
    dominant_cluster = gene_data['dominant_cluster_euclidean']
    n_mice = len(mice_abr)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Set style to match visualize_results.py
    sns.set_palette("husl")
    cluster_colors = sns.color_palette("husl", 4)
    
    # Plot cluster averages (using husl color palette)
    for cluster_id in range(4):
        label = f'Cluster {cluster_id} avg'
        if cluster_id == dominant_cluster:
            label += ' (dominant)'
            linewidth = 3
            alpha = 1.0
        else:
            linewidth = 1.5
            alpha = 0.5
        
        ax.plot(FREQ_LABELS, cluster_means_original[cluster_id], 
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
    
    # Simplified title with consistency and sample size
    euclidean_consistency = gene_data['consistency_score_euclidean']
    
    title = f'{gene_symbol} ({allele_symbol}) - {zygosity} - {center}'
    if analysis_type != 'all':
        title += f' - {analysis_type}'
    title += f'\nCluster {dominant_cluster} - Consistency: {euclidean_consistency:.2f} - N: {n_mice}'
    
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


def main(plots_only=False):
    """Main execution function."""
    
    # Load all data
    metadata, normalized_data, cluster_centers, original_data, orig_associations = load_all_data()
    
    if not plots_only:
        print("Performing Euclidean distance-based cluster assignment...")
        euclidean_labels, distances = assign_clusters_euclidean(normalized_data, cluster_centers)
        
        # Compare with original assignments
        orig_labels = np.load(BASE_DIR / 'results' / 'june_23_2025' / 'gmm_k4_tied' / 'cluster_labels.npy')
        changes = euclidean_labels != orig_labels
        print(f"Cluster assignments changed for {changes.sum()} / {len(changes)} mice ({changes.sum()/len(changes)*100:.1f}%)")
        
        print("Creating gene associations with Euclidean assignments...")
        euclidean_associations_df = create_gene_associations_euclidean(metadata, euclidean_labels, orig_associations)
        
        # Save associations CSV
        euclidean_associations_df.to_csv(OUTPUT_DIR / 'gene_cluster_associations_euclidean.csv', index=False)
        print(f"Saved associations to {OUTPUT_DIR / 'gene_cluster_associations_euclidean.csv'}")
    else:
        print("Loading existing Euclidean assignments...")
        euclidean_associations_df = pd.read_csv(OUTPUT_DIR / 'gene_cluster_associations_euclidean.csv')
        euclidean_labels, _ = assign_clusters_euclidean(normalized_data, cluster_centers)
    
    # Calculate cluster averages in original scale
    cluster_means_original = calculate_cluster_averages_original(original_data, euclidean_labels)
    
    # Create plots for each gene/allele combination
    print(f"Creating plots for {len(euclidean_associations_df)} gene/allele combinations...")
    
    for idx, gene_data in euclidean_associations_df.iterrows():
        # Create filename
        gene_symbol = gene_data['gene_symbol']
        allele_short = gene_data['allele_symbol'].replace('<', '_').replace('>', '_')
        zygosity = gene_data['zygosity']
        center = gene_data['center']
        analysis_type = gene_data['analysis_type']
        
        filename_base = f"{gene_symbol}_{allele_short}_{zygosity}_{center}_{analysis_type}"
        output_path_base = PLOT_DIR / filename_base
        
        try:
            plot_gene_abr_profiles(gene_data, metadata, original_data, euclidean_labels, 
                                 cluster_means_original, output_path_base)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1} / {len(euclidean_associations_df)} plots...")
        except Exception as e:
            print(f"  Error plotting {gene_symbol}: {str(e)}")
    
    print(f"\nCompleted! Results saved to {OUTPUT_DIR}")
    
    if not plots_only:
        # Create summary report
        summary = {
            'total_genes_analyzed': len(euclidean_associations_df),
            'assignments_changed': int(euclidean_associations_df['changed_assignment'].sum()),
            'mean_consistency_euclidean': float(euclidean_associations_df['consistency_score_euclidean'].mean()),
            'mean_consistency_original': float(euclidean_associations_df['consistency_score_original'].mean()),
            'genes_with_improved_consistency': int((euclidean_associations_df['consistency_score_euclidean'] > 
                                                   euclidean_associations_df['consistency_score_original']).sum())
        }
        
        with open(OUTPUT_DIR / 'euclidean_assignment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\nSummary:")
        print(f"  Total genes analyzed: {summary['total_genes_analyzed']}")
        print(f"  Genes with changed assignments: {summary['assignments_changed']}")
        print(f"  Mean consistency (Euclidean): {summary['mean_consistency_euclidean']:.3f}")
        print(f"  Mean consistency (Original): {summary['mean_consistency_original']:.3f}")
        print(f"  Genes with improved consistency: {summary['genes_with_improved_consistency']}")


if __name__ == "__main__":
    import sys
    plots_only = '--plots-only' in sys.argv
    main(plots_only=plots_only)