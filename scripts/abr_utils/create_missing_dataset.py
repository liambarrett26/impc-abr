#!/usr/bin/env python3
"""
Create a new dataset containing only mice for genes with missing sex-specific results
plus their matching controls for reanalysis.
"""

import pandas as pd

def create_missing_dataset():
    # Load the original data and missing test files
    print("Loading data files...")
    abr_data = pd.read_csv('/home/liamb/impc-abr/data/processed/abr_full_data.csv')
    missing_male = pd.read_csv('/home/liamb/impc-abr/missing_male_tests_that_could_run.csv')
    missing_female = pd.read_csv('/home/liamb/impc-abr/missing_female_tests_that_could_run.csv')
    
    # Get unique gene symbols from both missing files
    male_genes = set(missing_male['gene'].unique())
    female_genes = set(missing_female['gene'].unique())
    all_missing_genes = male_genes.union(female_genes)
    
    print(f"Found {len(all_missing_genes)} unique genes with missing results:")
    for gene in sorted(all_missing_genes):
        print(f"  {gene}")
    
    # Filter data for these genes plus all controls
    filtered_data = abr_data[
        (abr_data['gene_symbol'].isin(all_missing_genes)) |
        (abr_data['biological_sample_group'] == 'control')
    ].copy()
    
    print(f"\nOriginal dataset size: {len(abr_data):,} rows")
    print(f"Filtered dataset size: {len(filtered_data):,} rows")
    
    # Show breakdown by sample group
    sample_group_counts = filtered_data['biological_sample_group'].value_counts()
    print(f"\nSample group breakdown:")
    print(f"  Control: {sample_group_counts.get('control', 0):,}")
    print(f"  Experimental: {sample_group_counts.get('experimental', 0):,}")
    
    # Show breakdown by gene for experimental mice
    exp_data = filtered_data[filtered_data['biological_sample_group'] == 'experimental']
    gene_counts = exp_data['gene_symbol'].value_counts()
    print(f"\nExperimental mice by gene:")
    for gene in sorted(all_missing_genes):
        count = gene_counts.get(gene, 0)
        print(f"  {gene}: {count}")
    
    # Save the filtered dataset
    output_file = '/home/liamb/impc-abr/data/processed/abr_missingMaleFemale_data.csv'
    filtered_data.to_csv(output_file, index=False)
    print(f"\nSaved filtered dataset to: {output_file}")
    
    # Create summary file with gene details
    summary_data = []
    for _, row in missing_male.iterrows():
        summary_data.append({
            'gene': row['gene'],
            'allele': row['allele'],
            'zygosity': row['zygosity'],
            'center': row['center'],
            'missing_sex': 'male',
            'n_exp': row['n_exp_male'],
            'n_controls': row['n_controls_male']
        })
    
    for _, row in missing_female.iterrows():
        summary_data.append({
            'gene': row['gene'],
            'allele': row['allele'],
            'zygosity': row['zygosity'],
            'center': row['center'],
            'missing_sex': 'female',
            'n_exp': row['n_exp_female'],
            'n_controls': row['n_controls_female']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = '/home/liamb/impc-abr/missing_tests_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary of missing tests to: {summary_file}")
    
    return filtered_data

if __name__ == "__main__":
    create_missing_dataset()