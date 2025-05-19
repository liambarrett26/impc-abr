#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple ABR PCA Pipeline

This script implements a focused pipeline for PCA analysis of audiograms from the
IMPC ABR dataset. It handles data loading, PCA computation, and essential visualizations.

author: Liam Barrett
version: 1.0.0
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import from existing IMPC ABR modules
from abr_analysis.data.loader import ABRDataLoader
from abr_analysis.data.matcher import ControlMatcher

# Import the new PCA module
from abr_clustering.dimensionality.pca import AudiogramPCA


def create_output_dirs(output_dir):
    """Create necessary output directories."""
    base_dir = Path(output_dir)
    
    # Create subdirectories
    dirs = [
        base_dir,
        base_dir / 'pca_results',
        base_dir / 'figures'
    ]
    
    for d in dirs:
        d.mkdir(exist_ok=True, parents=True)
        
    return base_dir


def load_and_process_data(data_path, min_mutants=3, min_controls=20):
    """
    Load and process ABR data for PCA analysis.
    
    Parameters:
        data_path (str): Path to the ABR data file.
        min_mutants (int): Minimum number of mutant mice required per gene.
        min_controls (int): Minimum number of control mice required.
        
    Returns:
        tuple: Processed data, including mutant profiles, gene metadata, etc.
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
        print("Creating visualization plots...")
        
        # 1. Explained variance plot
        pca.plot_explained_variance(save_path=fig_dir / 'explained_variance.png')
        
        # 2. Component loadings plot
        pca.plot_components(save_path=fig_dir / 'pca_components.png')
        
        # 3. Component audiogram effects plot
        pca.plot_component_audiograms(save_path=fig_dir / 'component_effects.png')
        
        # 4. Extreme audiograms along each principal component
        for i in range(n_components):
            pca.plot_extreme_audiograms(all_mutant_profiles, pca_coords, 
                                       component=i+1, n_examples=5,
                                       save_path=fig_dir / f'extreme_audiograms_pc{i+1}.png')
    
    return {
        'pca': pca,
        'pca_coords': pca_coords
    }


def main(data_path, output_dir, n_components=5, create_plots=True):
    """
    Run the ABR PCA pipeline.
    
    Parameters:
        data_path (str): Path to the ABR data file.
        output_dir (str): Output directory for results.
        n_components (int): Number of principal components to compute.
        create_plots (bool): Whether to create visualization plots.
    """
    # Create output directories
    output_dir = create_output_dirs(output_dir)
    
    # Load and process data
    processed_data = load_and_process_data(data_path)
    
    # Perform PCA analysis
    pca_results = perform_pca_analysis(processed_data, n_components, output_dir, create_plots)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple ABR PCA Pipeline")
    
    parser.add_argument("--data", "-d", type=str, required=True,
                       help="Path to the ABR data file")
    parser.add_argument("--output", "-o", type=str, default="./output",
                       help="Output directory for results")
    parser.add_argument("--components", "-c", type=int, default=5,
                       help="Number of principal components to compute")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip creating visualization plots")
    
    args = parser.parse_args()
    
    main(args.data, args.output, args.components, not args.no_plots)