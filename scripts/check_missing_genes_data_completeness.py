#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check data completeness for genes with missing results.

This script analyzes genes that have sufficient mutants but no results,
determining whether they're excluded due to incomplete ABR data or other issues.

Author: Liam Barrett
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the package directory to the path
package_dir = Path(__file__).parent.parent
sys.path.insert(0, str(package_dir))

from scripts.abr_analysis.abr_analysis.data.loader import ABRDataLoader
from scripts.abr_analysis.abr_analysis.data.matcher import ControlMatcher


def analyze_gene_data_completeness(gene_symbol, matcher, freq_cols):
    """Analyze data completeness for a specific gene."""
    
    # Find experimental groups for this gene
    experimental_groups = matcher.find_experimental_groups(gene_symbol)
    
    if not experimental_groups:
        return [{
            'gene_symbol': gene_symbol,
            'status': 'No experimental groups found',
            'total_mice': 0,
            'groups': 0,
            'mice_with_complete_data': 0,
            'control_count': 0,
            'control_status': 'N/A'
        }]
    
    all_results = []
    
    for group in experimental_groups:
        # Get experimental profiles
        exp_profiles = matcher.get_experimental_profiles(group, freq_cols)
        
        # Check for matching controls
        try:
            controls = matcher.find_matching_controls(group)
            control_count = len(controls['all'])
            control_status = 'OK'
        except ValueError as e:
            control_count = 0
            control_status = str(e)
        
        # Analyze NaN patterns
        total_mice = len(exp_profiles)
        mice_with_complete_data = np.sum(~np.any(np.isnan(exp_profiles), axis=1))
        
        # Check each frequency
        freq_data = {}
        for i, freq in enumerate(freq_cols):
            valid_count = np.sum(~np.isnan(exp_profiles[:, i]))
            valid_values = exp_profiles[:, i][~np.isnan(exp_profiles[:, i])]
            
            freq_data[freq] = {
                'valid_count': valid_count,
                'mean_threshold': np.mean(valid_values) if len(valid_values) > 0 else np.nan
            }
        
        result = {
            'gene_symbol': gene_symbol,
            'allele_symbol': group['allele_symbol'],
            'zygosity': group['zygosity'],
            'center': group['phenotyping_center'],
            'total_mice': total_mice,
            'mice_with_complete_data': mice_with_complete_data,
            'control_count': control_count,
            'control_status': control_status,
            **{f'{k}_count': v['valid_count'] for k, v in freq_data.items()},
            **{f'{k}_mean': v['mean_threshold'] for k, v in freq_data.items()}
        }
        
        all_results.append(result)
    
    return all_results


def main():
    """Main function to check all missing genes."""
    
    print("Loading missing genes list...")
    missing_genes_df = pd.read_csv('data/processed/missing_results_genes.csv')
    print(f"Found {len(missing_genes_df)} genes to check")
    
    print("\nLoading ABR data...")
    data_path = "data/processed/abr_missing_genes_data_v4.csv"
    loader = ABRDataLoader(data_path)
    data = loader.load_data()
    
    # Extract gene symbols if not present
    if 'gene_symbol' not in data.columns and 'allele_symbol' in data.columns:
        def extract_gene_symbol(allele_symbol):
            if pd.isna(allele_symbol):
                return None
            return str(allele_symbol).split('<')[0]
        
        data['gene_symbol'] = data['allele_symbol'].apply(extract_gene_symbol)
    
    # Get frequency columns
    freq_cols = loader.get_frequencies()
    print(f"Frequency columns: {freq_cols}")
    
    # Initialize matcher
    matcher = ControlMatcher(data)
    
    print("\nAnalyzing gene data completeness...")
    all_results = []
    
    # Process each gene
    for idx, row in missing_genes_df.iterrows():
        gene_symbol = row['gene_symbol']
        if pd.isna(gene_symbol):
            continue
        
        print(f"Checking {gene_symbol}... ({idx+1}/{len(missing_genes_df)})", end='\r')
        
        gene_results = analyze_gene_data_completeness(gene_symbol, matcher, freq_cols)
        if isinstance(gene_results, dict):
            all_results.append(gene_results)
        else:
            all_results.extend(gene_results)
    
    print("\n\nCreating summary report...")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Categorize results
    if len(results_df) > 0:
        # Genes with no experimental groups
        no_groups = []
        if 'status' in results_df.columns:
            no_groups = results_df[results_df['status'] == 'No experimental groups found']['gene_symbol'].unique()
            # Filter to genes with experimental groups
            with_groups = results_df[results_df['status'] != 'No experimental groups found'].copy()
        else:
            # All results have experimental groups
            with_groups = results_df.copy()
        
        if len(with_groups) > 0:
            # Genes with insufficient controls
            insufficient_controls = with_groups[with_groups['control_count'] < 20]
            
            # Genes with sufficient controls
            sufficient_controls = with_groups[with_groups['control_count'] >= 20]
            
            # Among those with sufficient controls, check data completeness
            # Require at least 3 mice with complete data
            complete_data_ok = sufficient_controls[sufficient_controls['mice_with_complete_data'] >= 3]
            incomplete_data = sufficient_controls[sufficient_controls['mice_with_complete_data'] < 3]
            
            # For incomplete data, check if any frequency subset is viable (>=3 mice)
            freq_count_cols = [col for col in results_df.columns if col.endswith('_count') and not col.startswith('control')]
            incomplete_data['max_freq_count'] = incomplete_data[freq_count_cols].max(axis=1)
            
            # Categorize incomplete data genes
            could_analyze_subset = incomplete_data[incomplete_data['max_freq_count'] >= 3]
            truly_insufficient = incomplete_data[incomplete_data['max_freq_count'] < 3]
            
            # Create summary tables
            print("\n=== SUMMARY REPORT ===\n")
            
            print(f"Total genes checked: {len(missing_genes_df)}")
            print(f"Genes with no experimental groups in data: {len(no_groups)}")
            print(f"Genes with experimental groups: {len(with_groups['gene_symbol'].unique())}")
            
            print("\n--- EXCLUSION REASONS ---")
            
            if len(insufficient_controls) > 0:
                print(f"\n1. INSUFFICIENT CONTROLS (<20): {len(insufficient_controls['gene_symbol'].unique())} genes")
                control_summary = insufficient_controls.groupby('gene_symbol').agg({
                    'control_count': 'max',
                    'total_mice': 'sum'
                }).sort_values('total_mice', ascending=False)
                print(control_summary.head(10))
            
            if len(truly_insufficient) > 0:
                print(f"\n2. INSUFFICIENT DATA (no frequency with >=3 mice): {len(truly_insufficient['gene_symbol'].unique())} genes")
                insuff_summary = truly_insufficient.groupby('gene_symbol').agg({
                    'total_mice': 'sum',
                    'max_freq_count': 'max'
                }).sort_values('total_mice', ascending=False)
                print(insuff_summary.head(10))
            
            if len(could_analyze_subset) > 0:
                print(f"\n3. INCOMPLETE DATA BUT ANALYZABLE (>=3 mice for some frequencies): {len(could_analyze_subset['gene_symbol'].unique())} genes")
                print("These genes have missing data (likely due to severe hearing loss) but could be analyzed using frequency subsets:")
                
                # Show detailed breakdown for these genes
                subset_detail = []
                for gene in could_analyze_subset['gene_symbol'].unique():
                    gene_data = could_analyze_subset[could_analyze_subset['gene_symbol'] == gene].iloc[0]
                    
                    # Find which frequencies have sufficient data
                    viable_freqs = []
                    for col in freq_count_cols:
                        freq_name = col.replace('_count', '')
                        if gene_data[col] >= 3:
                            mean_col = f'{freq_name}_mean'
                            if mean_col in gene_data:
                                mean_val = gene_data[mean_col]
                                viable_freqs.append(f"{freq_name}: {int(gene_data[col])} mice (mean {mean_val:.1f} dB)")
                    
                    subset_detail.append({
                        'gene': gene,
                        'total_mice': int(gene_data['total_mice']),
                        'complete_data': int(gene_data['mice_with_complete_data']),
                        'viable_frequencies': '; '.join(viable_freqs)
                    })
                
                subset_df = pd.DataFrame(subset_detail).sort_values('total_mice', ascending=False)
                print(subset_df.head(20).to_string(index=False))
            
            if len(complete_data_ok) > 0:
                print(f"\n4. SHOULD BE ANALYZABLE (sufficient controls and complete data): {len(complete_data_ok['gene_symbol'].unique())} genes")
                print("These genes have sufficient data but still produced no results - investigate further:")
                analyzable_summary = complete_data_ok.groupby('gene_symbol').agg({
                    'total_mice': 'sum',
                    'mice_with_complete_data': 'sum',
                    'control_count': 'max'
                }).sort_values('total_mice', ascending=False)
                print(analyzable_summary.head(20))
            
            # Save detailed results
            output_file = 'data/processed/missing_genes_data_completeness_check.csv'
            results_df.to_csv(output_file, index=False)
            print(f"\n\nDetailed results saved to: {output_file}")
            
            # Save summary
            summary_file = 'data/processed/missing_genes_categorized_summary.csv'
            summary_data = []
            
            for gene in missing_genes_df['gene_symbol']:
                if gene in no_groups:
                    category = 'No experimental groups'
                elif gene in insufficient_controls['gene_symbol'].values:
                    category = 'Insufficient controls'
                elif gene in truly_insufficient['gene_symbol'].values:
                    category = 'Insufficient data (all frequencies)'
                elif gene in could_analyze_subset['gene_symbol'].values:
                    category = 'Incomplete data (analyzable subset)'
                elif gene in complete_data_ok['gene_symbol'].values:
                    category = 'Should be analyzable'
                else:
                    category = 'Unknown'
                
                summary_data.append({'gene_symbol': gene, 'category': category})
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_file, index=False)
            print(f"Summary categorization saved to: {summary_file}")
            
            # Print category counts
            print("\n--- CATEGORY SUMMARY ---")
            print(summary_df['category'].value_counts())
    
    else:
        print("No results found - check if the data file contains the genes")


if __name__ == "__main__":
    main()