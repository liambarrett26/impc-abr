#!/usr/bin/env python
"""
Script to create a filtered dataset containing missing genes and all controls.
"""

import pandas as pd
import numpy as np

def main():
    print("Loading missing genes data...")
    # Load missing genes
    missing_genes_df = pd.read_csv('data/processed/missing_allMaleFemale_results_genes.csv', low_memory=False)
    missing_gene_symbols = set(missing_genes_df['gene_symbol'].unique())
    print(f"Number of unique missing genes: {len(missing_gene_symbols)}")

    print("Loading full ABR data...")
    # Load full ABR data
    abr_full_df = pd.read_csv('data/processed/abr_full_data.csv', low_memory=False)
    print(f"Full ABR data shape: {abr_full_df.shape}")

    # Extract gene symbols from allele_symbol
    def extract_gene_symbol(allele_symbol):
        if pd.isna(allele_symbol):
            return None
        return str(allele_symbol).split('<')[0]

    abr_full_df['gene_symbol'] = abr_full_df['allele_symbol'].apply(extract_gene_symbol)

    print("Filtering data...")
    # Filter for missing genes (experimental samples)
    missing_genes_data = abr_full_df[
        (abr_full_df['gene_symbol'].isin(missing_gene_symbols)) &
        (abr_full_df['biological_sample_group'] == 'experimental')
    ].copy()

    print(f"Missing genes experimental samples: {len(missing_genes_data)}")

    # Get all control samples
    control_data = abr_full_df[abr_full_df['biological_sample_group'] == 'control'].copy()
    print(f"Control samples: {len(control_data)}")

    # Combine missing genes data with all controls
    combined_data = pd.concat([missing_genes_data, control_data], ignore_index=True)
    print(f"Combined dataset shape: {combined_data.shape}")

    # Summary statistics
    print("\nDataset summary:")
    print(f"- Experimental samples (missing genes): {len(missing_genes_data)}")
    print(f"- Control samples: {len(control_data)}")
    print(f"- Total samples: {len(combined_data)}")
    print(f"- Unique missing genes found in data: {combined_data[combined_data['biological_sample_group'] == 'experimental']['gene_symbol'].nunique()}")

    # Save the combined dataset
    output_path = 'data/processed/abr_missing_genes_data_v6.csv'
    print(f"\nSaving combined dataset to {output_path}...")
    combined_data.to_csv(output_path, index=False)
    print("Dataset saved successfully!")

    # Show some samples from each group
    print("\nSample experimental data:")
    print(combined_data[combined_data['biological_sample_group'] == 'experimental'][['gene_symbol', 'allele_symbol', 'biological_sample_group']].head())

    print("\nSample control data:")
    print(combined_data[combined_data['biological_sample_group'] == 'control'][['gene_symbol', 'allele_symbol', 'biological_sample_group']].head())

if __name__ == "__main__":
    main()