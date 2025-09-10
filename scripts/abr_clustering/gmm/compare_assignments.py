#!/usr/bin/env python3
"""
Compare assignments between euclidean individual and original-space approaches.
"""

import pandas as pd

def compare_assignments():
    # Load both datasets
    original_space = pd.read_csv('results/euclidean_assignment_original_space/gene_cluster_associations_original_space.csv')
    euclidean = pd.read_csv('results/euclidean_assignment/gene_cluster_associations_euclidean.csv')
    
    # Merge on analysis_key to compare
    merged = original_space.merge(euclidean[['analysis_key', 'dominant_cluster_euclidean']], 
                                 on='analysis_key', how='inner')
    
    # Find genes that changed between the two euclidean approaches
    changed = merged[merged['dominant_cluster_original_space'] != merged['dominant_cluster_euclidean']].copy()
    
    print(f'Genes changed between euclidean individual vs original-space approaches: {len(changed)} / {len(merged)}')
    print()
    print('Changed assignments:')
    print('Format: Gene (Allele) - Zygosity - Center: Individual → Original-space')
    print()
    
    # Sort by gene symbol for easier reading
    changed_sorted = changed.sort_values('gene_symbol')
    
    for _, row in changed_sorted.iterrows():
        gene = row['gene_symbol']
        allele = row['allele_symbol'].replace('<', '').replace('>', '')
        zyg = row['zygosity']
        center = row['center']
        old_cluster = row['dominant_cluster_euclidean']
        new_cluster = row['dominant_cluster_original_space']
        
        print(f'{gene} ({allele}) - {zyg} - {center}: {old_cluster} → {new_cluster}')
    
    # Summary by cluster changes
    print('\n' + '='*60)
    print('Summary of cluster transitions:')
    print('='*60)
    
    for old_cluster in sorted(changed['dominant_cluster_euclidean'].unique()):
        subset = changed[changed['dominant_cluster_euclidean'] == old_cluster]
        print(f'\nFrom Cluster {old_cluster}:')
        for new_cluster in sorted(subset['dominant_cluster_original_space'].unique()):
            count = len(subset[subset['dominant_cluster_original_space'] == new_cluster])
            genes = subset[subset['dominant_cluster_original_space'] == new_cluster]['gene_symbol'].tolist()
            print(f'  → Cluster {new_cluster}: {count} genes ({", ".join(genes[:5])}{"..." if len(genes) > 5 else ""})')

if __name__ == "__main__":
    compare_assignments()