#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check sex-specific data completeness for genes in the v5 dataset.

This script focuses specifically on genes that were identified as analyzable
in the v5 dataset to determine sex-specific analysis possibilities.

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


def main():
    """Main function to check sex-specific data completeness for v5 genes."""
    
    print("Loading v5 ABR dataset...")
    data_path = "data/processed/abr_missing_genes_data_v5.csv"
    
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
    
    # Extract gene symbols if not present
    if 'gene_symbol' not in data.columns and 'allele_symbol' in data.columns:
        def extract_gene_symbol(allele_symbol):
            if pd.isna(allele_symbol):
                return None
            return str(allele_symbol).split('<')[0]
        
        data['gene_symbol'] = data['allele_symbol'].apply(extract_gene_symbol)
        print("Extracted gene symbols from allele symbols")
    
    # Get unique genes with experimental data
    experimental_data = data[data['biological_sample_group'] == 'experimental']
    genes = experimental_data['gene_symbol'].dropna().unique()
    print(f"Found {len(genes)} unique genes to analyze")
    
    print("\nAnalyzing sex-specific data completeness...")
    all_results = []
    
    # Process each gene
    for idx, gene_symbol in enumerate(genes):
        print(f"Checking {gene_symbol}... ({idx+1}/{len(genes)})")
        
        # Find experimental groups for this gene
        experimental_groups = matcher.find_experimental_groups(gene_symbol)
        
        if not experimental_groups:
            continue
        
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
                
                male_control_count = len(male_controls)
                female_control_count = len(female_controls)
                control_status = 'OK'
            except ValueError as e:
                male_control_count = 0
                female_control_count = 0
                control_status = str(e)
            
            # Analyze data completeness for each sex
            def analyze_sex_data(profiles, sex_name):
                if len(profiles) == 0:
                    return {
                        'total_mice': 0,
                        'complete_data_mice': 0,
                        'sufficient_sample': False,
                        'complete_data_sufficient': False
                    }
                
                total_mice = len(profiles)
                complete_data_mice = np.sum(~np.any(np.isnan(profiles), axis=1))
                
                return {
                    'total_mice': total_mice,
                    'complete_data_mice': complete_data_mice,
                    'sufficient_sample': total_mice >= 3,
                    'complete_data_sufficient': complete_data_mice >= 3
                }
            
            male_analysis = analyze_sex_data(male_profiles, 'male')
            female_analysis = analyze_sex_data(female_profiles, 'female')
            
            # Determine analyzability
            male_controls_sufficient = male_control_count >= 20
            female_controls_sufficient = female_control_count >= 20
            
            male_analyzable = (male_analysis['sufficient_sample'] and 
                              male_analysis['complete_data_sufficient'] and 
                              male_controls_sufficient)
            
            female_analyzable = (female_analysis['sufficient_sample'] and 
                               female_analysis['complete_data_sufficient'] and 
                               female_controls_sufficient)
            
            result = {
                'gene_symbol': gene_symbol,
                'allele_symbol': group['allele_symbol'],
                'zygosity': group['zygosity'],
                'center': group['phenotyping_center'],
                'male_total_mice': male_analysis['total_mice'],
                'male_complete_data_mice': male_analysis['complete_data_mice'],
                'male_control_count': male_control_count,
                'male_analyzable': male_analyzable,
                'female_total_mice': female_analysis['total_mice'],
                'female_complete_data_mice': female_analysis['complete_data_mice'],
                'female_control_count': female_control_count,
                'female_analyzable': female_analyzable,
                'control_status': control_status
            }
            
            all_results.append(result)
    
    print(f"\nProcessed {len(all_results)} experimental groups")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) > 0:
        # Gene-level summary (best group per gene)
        gene_summary = []
        
        for gene in genes:
            gene_data = results_df[results_df['gene_symbol'] == gene]
            
            if len(gene_data) == 0:
                continue
            
            # Find if any group is analyzable for each sex
            male_any_analyzable = gene_data['male_analyzable'].any()
            female_any_analyzable = gene_data['female_analyzable'].any()
            both_any_analyzable = male_any_analyzable and female_any_analyzable
            
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
        print("\n=== SEX-SPECIFIC ANALYSIS SUMMARY (V5 GENES) ===\n")
        
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
        
        # Show percentages
        print(f"\nPercentages of v5 genes suitable for sex-specific analysis:")
        print(f"Male-only: {len(male_only_genes)/total_genes*100:.1f}%")
        print(f"Female-only: {len(female_only_genes)/total_genes*100:.1f}%")
        print(f"Both sexes: {len(both_sex_genes)/total_genes*100:.1f}%")
        print(f"Either sex: {len(either_sex_genes)/total_genes*100:.1f}%")
        
        # Show top genes for each category
        print("\n--- TOP MALE-ONLY ANALYZABLE GENES ---")
        if len(male_only_genes) > 0:
            male_display = male_only_genes[['gene_symbol', 'max_male_mice', 'max_male_complete_data']].sort_values('max_male_mice', ascending=False)
            print(male_display.head(10).to_string(index=False))
        else:
            print("No male-only analyzable genes found")
        
        print("\n--- TOP FEMALE-ONLY ANALYZABLE GENES ---")
        if len(female_only_genes) > 0:
            female_display = female_only_genes[['gene_symbol', 'max_female_mice', 'max_female_complete_data']].sort_values('max_female_mice', ascending=False)
            print(female_display.head(10).to_string(index=False))
        else:
            print("No female-only analyzable genes found")
        
        print("\n--- TOP BOTH-SEX ANALYZABLE GENES ---")
        if len(both_sex_genes) > 0:
            both_display = both_sex_genes[['gene_symbol', 'max_male_mice', 'max_female_mice']].sort_values(['max_male_mice', 'max_female_mice'], ascending=False)
            print(both_display.head(10).to_string(index=False))
        else:
            print("No genes analyzable for both sexes found")
        
        # Save results
        detailed_output = 'data/processed/sex_specific_data_completeness_v5_detailed.csv'
        results_df.to_csv(detailed_output, index=False)
        print(f"\n\nDetailed results saved to: {detailed_output}")
        
        summary_output = 'data/processed/sex_specific_data_completeness_v5_summary.csv'
        gene_summary_df.to_csv(summary_output, index=False)
        print(f"Gene-level summary saved to: {summary_output}")
        
        # Create recommendations
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
                'female_sufficient': gene['female_analyzable']
            })
        
        rec_df = pd.DataFrame(recommendations)
        rec_output = 'data/processed/sex_specific_analysis_recommendations_v5.csv'
        rec_df.to_csv(rec_output, index=False)
        print(f"Analysis recommendations saved to: {rec_output}")
        
        print("\n--- RECOMMENDATION SUMMARY ---")
        print(rec_df['sex_analysis_recommendation'].value_counts())
        
    else:
        print("No results found - check if the data file contains experimental data")


if __name__ == "__main__":
    main()