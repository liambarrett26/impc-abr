#!/usr/bin/env python3
"""
Generate missing plots for the original-space euclidean assignment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path("/home/liamb/impc-abr/scripts/abr_clustering/gmm")
OUTPUT_DIR = BASE_DIR / "results" / "euclidean_assignment_original_space"
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


def calculate_cluster_means_original_space(original_data, cluster_labels):
    """Calculate mean ABR profiles for each cluster in original dB SPL space."""
    cluster_means_original = {}
    
    for cluster_id in range(4):
        cluster_mask = cluster_labels == cluster_id
        if cluster_mask.sum() > 0:
            cluster_data = original_data.iloc[cluster_mask][ABR_COLS]
            cluster_means_original[cluster_id] = cluster_data.mean().values
        else:
            # Use overall mean if no samples in cluster
            cluster_means_original[cluster_id] = original_data[ABR_COLS].mean().values
    
    return cluster_means_original


def plot_gene_abr_profiles_original_space(gene_data, metadata, original_data, cluster_means_original, output_path_base):
    """Create ABR profile plot showing gene mean, cluster centers, and distances in original space."""
    
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
        print(f"  Warning: No mice found for {gene_symbol} {allele_symbol}")
        return False
    
    # Get data for these mice
    mice_abr = original_data.iloc[matching_indices][ABR_COLS].values
    dominant_cluster = gene_data['dominant_cluster_original_space']
    n_mice = len(mice_abr)
    
    # Parse distances
    distances_str = gene_data['distances_to_clusters_original_space']
    distances = eval(distances_str)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Set style to match visualize_results.py
    sns.set_palette("husl")
    cluster_colors = sns.color_palette("husl", 4)
    
    # Plot cluster averages in original scale
    for cluster_id in range(4):
        label = f'Cluster {cluster_id} (d={distances[cluster_id]:.1f} dB)'
        if cluster_id == dominant_cluster:
            label += ' ✓'
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
    
    # Title with cluster assignment
    title = f'{gene_symbol} ({allele_symbol}) - {zygosity} - {phenotyping_center}'
    if analysis_type != 'all':
        title += f' - {analysis_type}'
    title += f'\\nCluster {dominant_cluster} (original-space) - N: {n_mice}'
    
    ax.set_title(title, fontsize=12, pad=10)
    
    # Save in multiple formats
    plt.tight_layout()
    
    # Get base filename without extension
    base_path = Path(output_path_base).with_suffix('')
    
    try:
        # Save high-res PNG (1200 DPI)
        fig.savefig(f"{base_path}_highres.png", dpi=1200, bbox_inches='tight')
        
        # Save low-res PNG (150 DPI)
        fig.savefig(f"{base_path}.png", dpi=150, bbox_inches='tight')
        
        # Save EPS (vector format, high quality)
        fig.savefig(f"{base_path}.eps", format='eps', bbox_inches='tight', dpi=1200)
        
        plt.close(fig)
        return True
    
    except Exception as e:
        print(f"  Error saving plots for {gene_symbol}: {str(e)}")
        plt.close(fig)
        return False


def main():
    """Generate missing plots only."""
    
    print("Loading data...")
    
    # Load metadata
    metadata = pd.read_csv(BASE_DIR / 'shared_data' / 'metadata.csv')
    
    # Load original ABR data
    original_data = pd.read_csv(BASE_DIR / 'results' / 'june_23_2025' / 'gmm_k4_tied' / 'original_data.csv')
    
    # Load cluster labels to compute cluster means in original space
    cluster_labels = np.load(BASE_DIR / 'results' / 'june_23_2025' / 'gmm_k4_tied' / 'cluster_labels.npy')
    
    # Load associations
    associations_df = pd.read_csv(OUTPUT_DIR / 'gene_cluster_associations_original_space.csv')
    
    # Calculate cluster means in original space
    cluster_means_original = calculate_cluster_means_original_space(original_data, cluster_labels)
    
    # Find missing alleles
    missing_alleles = []
    for _, row in associations_df.iterrows():
        gene_symbol = row['gene_symbol']
        allele_short = row['allele_symbol'].replace('<', '_').replace('>', '_')
        zygosity = row['zygosity']
        center = row['center']
        analysis_type = row['analysis_type']
        
        # Clean up all special characters for safe filenames
        filename_base = f"{gene_symbol}_{allele_short}_{zygosity}_{center}_{analysis_type}"
        filename_base = filename_base.replace('(', '_').replace(')', '_').replace(' ', '_').replace('-', '_')
        highres_file = PLOT_DIR / (filename_base + '_highres.png')
        
        if not highres_file.exists():
            missing_alleles.append((row, filename_base))
    
    print(f"Found {len(missing_alleles)} missing alleles to plot")
    
    # Generate missing plots
    success_count = 0
    for gene_data, filename_base in missing_alleles:
        gene_symbol = gene_data['gene_symbol']
        print(f"  Generating plots for {gene_symbol}...")
        
        output_path_base = PLOT_DIR / filename_base
        
        success = plot_gene_abr_profiles_original_space(
            gene_data, metadata, original_data, cluster_means_original, output_path_base
        )
        
        if success:
            success_count += 1
    
    print(f"\\nCompleted! Successfully generated plots for {success_count}/{len(missing_alleles)} alleles")


if __name__ == "__main__":
    main()