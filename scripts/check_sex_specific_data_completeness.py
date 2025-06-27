#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check data completeness for sex-specific ABR analysis.

This script analyzes the full ABR dataset to determine which genes have sufficient
sample sizes and data completeness for male-only and female-only analysis.

Based on the inclusion criteria from the ControlMatcher:
- Minimum 3 experimental mice per sex per group
- Minimum 20 matched controls per sex
- Complete ABR data across all frequencies

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


def analyze_sex_specific_data_completeness(gene_symbol, matcher, freq_cols):
    """Analyze sex-specific data completeness for a specific gene."""
    
    # Find experimental groups for this gene
    experimental_groups = matcher.find_experimental_groups(gene_symbol)
    
    if not experimental_groups:
        return [{
            'gene_symbol': gene_symbol,
            'status': 'No experimental groups found',
            'total_groups': 0,
            'male_analyzable_groups': 0,
            'female_analyzable_groups': 0,
            'both_sex_analyzable_groups': 0
        }]
    
    all_results = []
    
    for group in experimental_groups:
        # Get experimental data for this group
        group_data = group['data']
        
        # Separate by sex
        male_data = group_data[group_data['sex'] == 'male']
        female_data = group_data[group_data['sex'] == 'female']
        
        # Get profiles for each sex
        male_profiles = matcher.get_experimental_profiles(
            {'data': male_data}, freq_cols
        ) if len(male_data) > 0 else np.array([]).reshape(0, len(freq_cols))
        
        female_profiles = matcher.get_experimental_profiles(
            {'data': female_data}, freq_cols
        ) if len(female_data) > 0 else np.array([]).reshape(0, len(freq_cols))
        
        # Check for matching controls
        try:
            controls = matcher.find_matching_controls(group)
            male_controls = controls['male']
            female_controls = controls['female']
            total_controls = controls['all']
            
            male_control_count = len(male_controls)
            female_control_count = len(female_controls)
            total_control_count = len(total_controls)
            control_status = 'OK'
        except ValueError as e:
            male_control_count = 0
            female_control_count = 0
            total_control_count = 0
            control_status = str(e)
        
        # Analyze data completeness for each sex
        def analyze_profiles(profiles, sex_name):
            if len(profiles) == 0:
                return {
                    f'{sex_name}_total_mice': 0,
                    f'{sex_name}_complete_data_mice': 0,
                    f'{sex_name}_sufficient_sample': False,
                    f'{sex_name}_complete_data_sufficient': False
                }
            
            total_mice = len(profiles)
            complete_data_mice = np.sum(~np.any(np.isnan(profiles), axis=1))
            
            # Check frequency-specific completeness
            freq_completeness = {}
            for i, freq in enumerate(freq_cols):
                valid_count = np.sum(~np.isnan(profiles[:, i]))
                freq_completeness[f'{sex_name}_{freq}_count'] = valid_count
                freq_completeness[f'{sex_name}_{freq}_sufficient'] = valid_count >= 3
            
            return {
                f'{sex_name}_total_mice': total_mice,
                f'{sex_name}_complete_data_mice': complete_data_mice,
                f'{sex_name}_sufficient_sample': total_mice >= 3,
                f'{sex_name}_complete_data_sufficient': complete_data_mice >= 3,
                **freq_completeness
            }
        
        male_analysis = analyze_profiles(male_profiles, 'male')
        female_analysis = analyze_profiles(female_profiles, 'female')
        
        # Determine analyzability
        male_controls_sufficient = male_control_count >= 20
        female_controls_sufficient = female_control_count >= 20
        total_controls_sufficient = total_control_count >= 20
        
        male_analyzable = (male_analysis['male_sufficient_sample'] and 
                          male_analysis['male_complete_data_sufficient'] and 
                          male_controls_sufficient)
        
        female_analyzable = (female_analysis['female_sufficient_sample'] and 
                           female_analysis['female_complete_data_sufficient'] and 
                           female_controls_sufficient)
        
        both_analyzable = (male_analyzable and female_analyzable and 
                          total_controls_sufficient)
        
        result = {
            'gene_symbol': gene_symbol,
            'allele_symbol': group['allele_symbol'],
            'zygosity': group['zygosity'],
            'center': group['phenotyping_center'],
            'total_control_count': total_control_count,
            'male_control_count': male_control_count,
            'female_control_count': female_control_count,
            'control_status': control_status,
            'male_analyzable': male_analyzable,
            'female_analyzable': female_analyzable,
            'both_sex_analyzable': both_analyzable,
            **male_analysis,
            **female_analysis
        }
        
        all_results.append(result)
    
    return all_results


def main():
    """Main function to check sex-specific data completeness for all genes."""
    
    print("Loading full ABR dataset...")
    data_path = "data/processed/abr_sample_data.csv"
    
    if not Path(data_path).exists():
        print(f"Error: {data_path} not found")
        return
    
    loader = ABRDataLoader(data_path)
    data = loader.load_data()
    print(f"Loaded {len(data)} records")
    
    # Get frequency columns
    freq_cols = loader.get_frequencies()
    print(f"Frequency columns: {freq_cols}")
    
    # Initialize matcher
    matcher = ControlMatcher(data)
    
    # Extract gene symbols if not present (like in the original script)
    if 'gene_symbol' not in data.columns and 'allele_symbol' in data.columns:
        def extract_gene_symbol(allele_symbol):
            if pd.isna(allele_symbol):
                return None
            return str(allele_symbol).split('<')[0]
        
        data['gene_symbol'] = data['allele_symbol'].apply(extract_gene_symbol)
        print("Extracted gene symbols from allele symbols")
    
    # Get all unique genes with experimental data
    experimental_data = data[data['biological_sample_group'] == 'experimental']
    genes = experimental_data['gene_symbol'].dropna().unique()
    print(f"Found {len(genes)} unique genes to analyze")
    
    print("\nAnalyzing sex-specific data completeness...")
    all_results = []
    
    # Process each gene
    for idx, gene_symbol in enumerate(genes):
        print(f"Checking {gene_symbol}... ({idx+1}/{len(genes)})", end='\r')
        
        gene_results = analyze_sex_specific_data_completeness(gene_symbol, matcher, freq_cols)
        if isinstance(gene_results, dict):
            all_results.append(gene_results)
        else:
            all_results.extend(gene_results)
    
    print("\n\nCreating sex-specific analysis report...")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) > 0:
        # Gene-level summary (best group per gene)
        gene_summary = []
        
        for gene in genes:
            gene_data = results_df[results_df['gene_symbol'] == gene]
            
            # Find if any group is analyzable for each sex
            male_any_analyzable = gene_data['male_analyzable'].any()
            female_any_analyzable = gene_data['female_analyzable'].any()
            both_any_analyzable = gene_data['both_sex_analyzable'].any()
            
            # Get best stats per gene
            max_male_mice = gene_data['male_total_mice'].max()
            max_female_mice = gene_data['female_total_mice'].max()
            max_male_complete = gene_data['male_complete_data_mice'].max()
            max_female_complete = gene_data['female_complete_data_mice'].max()
            max_male_controls = gene_data['male_control_count'].max()
            max_female_controls = gene_data['female_control_count'].max()
            
            gene_summary.append({
                'gene_symbol': gene,
                'total_groups': len(gene_data),
                'male_analyzable': male_any_analyzable,
                'female_analyzable': female_any_analyzable,
                'both_sex_analyzable': both_any_analyzable,
                'max_male_mice': max_male_mice,
                'max_female_mice': max_female_mice,
                'max_male_complete_data': max_male_complete,
                'max_female_complete_data': max_female_complete,
                'max_male_controls': max_male_controls,
                'max_female_controls': max_female_controls
            })
        
        gene_summary_df = pd.DataFrame(gene_summary)
        
        # Create summary statistics
        print("\n=== SEX-SPECIFIC ANALYSIS SUMMARY ===\n")
        
        total_genes = len(gene_summary_df)
        male_only_genes = gene_summary_df[
            gene_summary_df['male_analyzable'] & ~gene_summary_df['female_analyzable']
        ]
        female_only_genes = gene_summary_df[
            ~gene_summary_df['male_analyzable'] & gene_summary_df['female_analyzable']
        ]
        both_sex_genes = gene_summary_df[gene_summary_df['both_sex_analyzable']]
        either_sex_genes = gene_summary_df[
            gene_summary_df['male_analyzable'] | gene_summary_df['female_analyzable']
        ]
        
        print(f"Total genes analyzed: {total_genes}")
        print(f"Genes with male-only analysis possible: {len(male_only_genes)}")
        print(f"Genes with female-only analysis possible: {len(female_only_genes)}")
        print(f"Genes with both sex analysis possible: {len(both_sex_genes)}")
        print(f"Genes with either sex analysis possible: {len(either_sex_genes)}")
        
        # Detailed breakdowns
        print("\n--- MALE-ONLY ANALYZABLE GENES ---")
        if len(male_only_genes) > 0:
            male_only_display = male_only_genes[['gene_symbol', 'max_male_mice', 'max_male_complete_data', 'max_male_controls']].sort_values('max_male_mice', ascending=False)
            print(male_only_display.head(20).to_string(index=False))
        else:
            print("No male-only analyzable genes found")
        
        print("\n--- FEMALE-ONLY ANALYZABLE GENES ---")
        if len(female_only_genes) > 0:
            female_only_display = female_only_genes[['gene_symbol', 'max_female_mice', 'max_female_complete_data', 'max_female_controls']].sort_values('max_female_mice', ascending=False)
            print(female_only_display.head(20).to_string(index=False))
        else:
            print("No female-only analyzable genes found")
        
        print("\n--- BOTH SEX ANALYZABLE GENES ---")
        if len(both_sex_genes) > 0:
            both_sex_display = both_sex_genes[['gene_symbol', 'max_male_mice', 'max_female_mice', 'max_male_complete_data', 'max_female_complete_data']].sort_values('max_male_mice', ascending=False)
            print(both_sex_display.head(20).to_string(index=False))
        else:
            print("No genes analyzable for both sexes found")
        
        # Sample size distribution
        print("\n--- SAMPLE SIZE DISTRIBUTIONS ---")
        print(f"\nMale sample sizes (experimental mice):")
        print(f"  Mean: {gene_summary_df['max_male_mice'].mean():.1f}")
        print(f"  Median: {gene_summary_df['max_male_mice'].median():.1f}")
        print(f"  Range: {gene_summary_df['max_male_mice'].min()}-{gene_summary_df['max_male_mice'].max()}")
        
        print(f"\nFemale sample sizes (experimental mice):")
        print(f"  Mean: {gene_summary_df['max_female_mice'].mean():.1f}")
        print(f"  Median: {gene_summary_df['max_female_mice'].median():.1f}")
        print(f"  Range: {gene_summary_df['max_female_mice'].min()}-{gene_summary_df['max_female_mice'].max()}")
        
        # Control availability
        insufficient_male_controls = gene_summary_df[gene_summary_df['max_male_controls'] < 20]
        insufficient_female_controls = gene_summary_df[gene_summary_df['max_female_controls'] < 20]
        
        print(f"\nGenes with insufficient male controls (<20): {len(insufficient_male_controls)}")
        print(f"Genes with insufficient female controls (<20): {len(insufficient_female_controls)}")
        
        # Save detailed results
        detailed_output = 'data/processed/sex_specific_data_completeness_detailed.csv'
        results_df.to_csv(detailed_output, index=False)
        print(f"\n\nDetailed group-level results saved to: {detailed_output}")
        
        # Save gene-level summary
        summary_output = 'data/processed/sex_specific_data_completeness_summary.csv'
        gene_summary_df.to_csv(summary_output, index=False)
        print(f"Gene-level summary saved to: {summary_output}")
        
        # Create analysis recommendation file
        recommendations = []
        
        for _, gene in gene_summary_df.iterrows():
            gene_name = gene['gene_symbol']
            
            if gene['both_sex_analyzable']:
                rec = 'Both male and female analysis recommended'
            elif gene['male_analyzable'] and not gene['female_analyzable']:
                rec = 'Male-only analysis recommended'
            elif gene['female_analyzable'] and not gene['male_analyzable']:
                rec = 'Female-only analysis recommended'
            else:
                rec = 'Insufficient data for sex-specific analysis'
            
            recommendations.append({
                'gene_symbol': gene_name,
                'sex_analysis_recommendation': rec,
                'male_sufficient': gene['male_analyzable'],
                'female_sufficient': gene['female_analyzable'],
                'max_male_mice': gene['max_male_mice'],
                'max_female_mice': gene['max_female_mice']
            })
        
        rec_df = pd.DataFrame(recommendations)
        rec_output = 'data/processed/sex_specific_analysis_recommendations.csv'
        rec_df.to_csv(rec_output, index=False)
        print(f"Analysis recommendations saved to: {rec_output}")
        
        # Final summary counts
        print("\n--- RECOMMENDATION SUMMARY ---")
        print(rec_df['sex_analysis_recommendation'].value_counts())
        
    else:
        print("No results found - check if the data file contains experimental data")


if __name__ == "__main__":
    main()