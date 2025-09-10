#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis Report Generator for ABR Multivariate Results

This script generates a comprehensive analysis report from the detailed_results.csv file,
comparing findings with confirmed and candidate deafness genes.

Author: Liam Barrett
Version: 1.0.0
"""

import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
import sys

def load_gene_list(file_path):
    """Load gene list from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        genes = [line.strip() for line in f if line.strip()]
    return set(genes)

def format_gene_list(genes, columns=3):
    """Format a list of genes into columns for pretty printing."""
    genes = sorted(list(genes))
    rows = []
    for i in range(0, len(genes), columns):
        row = genes[i:i + columns]
        rows.append('\t'.join(str(g).ljust(20) for g in row))
    return '\n'.join(rows)

def get_significant_genes(df, analysis_type, threshold=0.001):
    """Get significant genes for a given analysis type."""
    if analysis_type == 'all':
        col = 'all_p_value'
    elif analysis_type == 'male':
        col = 'male_p_value'
    elif analysis_type == 'female':
        col = 'female_p_value'
    elif analysis_type == 'combined':
        # For combined analysis, get genes significant in ANY analysis
        significant = df[
            ((df['all_p_value'].notna()) & (df['all_p_value'] <= threshold)) |
            ((df['male_p_value'].notna()) & (df['male_p_value'] <= threshold)) |
            ((df['female_p_value'].notna()) & (df['female_p_value'] <= threshold))
        ]
        return set(significant['gene_symbol'].unique())
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    # Filter for significant genes
    significant = df[(df[col].notna()) & (df[col] <= threshold)]
    return set(significant['gene_symbol'].unique())

def analyze_gene_overlap(significant_genes, confirmed_genes, candidate_genes):
    """Analyze overlap between significant genes and reference sets."""
    found_in_confirmed = significant_genes & confirmed_genes
    found_in_candidate = significant_genes & candidate_genes
    novel = significant_genes - confirmed_genes - candidate_genes
    
    missed_confirmed = confirmed_genes - significant_genes
    missed_candidate = candidate_genes - significant_genes
    
    return {
        'found_in_confirmed': found_in_confirmed,
        'found_in_candidate': found_in_candidate,
        'novel': novel,
        'missed_confirmed': missed_confirmed,
        'missed_candidate': missed_candidate
    }

def get_significant_entries_detailed(df, analysis_type, threshold=0.001):
    """Get detailed significant entries (gene + allele info) for a given analysis type."""
    if analysis_type == 'all':
        col = 'all_p_value'
    elif analysis_type == 'male':
        col = 'male_p_value'
    elif analysis_type == 'female':
        col = 'female_p_value'
    elif analysis_type == 'combined':
        # For combined analysis, get entries significant in ANY analysis
        significant = df[
            ((df['all_p_value'].notna()) & (df['all_p_value'] <= threshold)) |
            ((df['male_p_value'].notna()) & (df['male_p_value'] <= threshold)) |
            ((df['female_p_value'].notna()) & (df['female_p_value'] <= threshold))
        ]
        
        # Create detailed entries with gene_symbol and allele_symbol
        entries = []
        for _, row in significant.iterrows():
            entry = f"{row['gene_symbol']} ({row['allele_symbol']})"
            entries.append(entry)
        
        return entries
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    # Filter for significant entries
    significant = df[(df[col].notna()) & (df[col] <= threshold)]
    
    # Create detailed entries with gene_symbol and allele_symbol
    entries = []
    for _, row in significant.iterrows():
        entry = f"{row['gene_symbol']} ({row['allele_symbol']})"
        entries.append(entry)
    
    return entries

def generate_report(results_path, confirmed_genes_path, candidate_genes_path, 
                   output_path=None, threshold=0.001):
    """Generate comprehensive analysis report."""
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(results_path)
    confirmed_genes = load_gene_list(confirmed_genes_path)
    candidate_genes = load_gene_list(candidate_genes_path)
    
    # Calculate statistics for each analysis type
    analysis_results = {}
    for analysis_type in ['combined', 'all', 'male', 'female']:
        significant_genes = get_significant_genes(df, analysis_type, threshold)
        significant_entries = get_significant_entries_detailed(df, analysis_type, threshold)
        overlap_analysis = analyze_gene_overlap(significant_genes, confirmed_genes, candidate_genes)
        
        analysis_results[analysis_type] = {
            'significant_genes': significant_genes,
            'significant_entries': significant_entries,
            'overlap': overlap_analysis
        }
    
    # Create output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"analysis_report_{timestamp}.txt"
    
    # Generate report
    print(f"Writing report to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("ABR Analysis Report - Detailed Results Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        f.write("Overall Statistics\n")
        f.write("-" * 18 + "\n")
        f.write(f"Total genes analyzed: {len(df['gene_symbol'].unique())}\n")
        f.write(f"Total gene/allele combinations: {len(df)}\n")
        f.write(f"Confirmed deafness genes: {len(confirmed_genes)}\n")
        f.write(f"Candidate deafness genes: {len(candidate_genes)}\n")
        f.write(f"Significance threshold (p-value): {threshold}\n\n")
        
        # Results for each analysis type
        for analysis_type in ['combined', 'all', 'male', 'female']:
            if analysis_type == 'combined':
                title = "COMBINED ALL + MALE + FEMALE ANALYSIS"
                f.write(f"\n{title}\n")
                f.write("-" * len(title) + "\n")
            else:
                f.write(f"\n{analysis_type.upper()} ANALYSIS\n")
                f.write("-" * (len(analysis_type) + 9) + "\n")
            
            results = analysis_results[analysis_type]
            overlap = results['overlap']
            
            # Summary statistics
            total_sig_genes = len(results['significant_genes'])
            total_sig_entries = len(results['significant_entries'])
            f.write(f"Total significant genes: {total_sig_genes}\n")
            f.write(f"Total significant gene/allele combinations: {total_sig_entries}\n")
            f.write(f"Found in confirmed deafness genes: {len(overlap['found_in_confirmed'])}\n")
            f.write(f"Found in candidate deafness genes: {len(overlap['found_in_candidate'])}\n")
            f.write(f"Novel candidates: {len(overlap['novel'])}\n")
            f.write(f"Missed confirmed genes: {len(overlap['missed_confirmed'])}\n")
            f.write(f"Missed candidate genes: {len(overlap['missed_candidate'])}\n\n")
            
            # Detailed listings
            f.write("ALL SIGNIFICANT GENE/ALLELE COMBINATIONS:\n")
            f.write(format_gene_list(results['significant_entries']))
            f.write("\n\n")
            
            f.write("Found in confirmed deafness genes:\n")
            f.write(format_gene_list(overlap['found_in_confirmed']))
            f.write("\n\n")
            
            f.write("Found in candidate deafness genes:\n")
            f.write(format_gene_list(overlap['found_in_candidate']))
            f.write("\n\n")
            
            f.write("Novel candidates:\n")
            f.write(format_gene_list(overlap['novel']))
            f.write("\n\n")
            
            f.write("Missed confirmed genes:\n")
            f.write(format_gene_list(overlap['missed_confirmed']))
            f.write("\n\n")
            
            f.write("Missed candidate genes:\n")
            f.write(format_gene_list(overlap['missed_candidate']))
            f.write("\n\n")
        
        # Sample size statistics
        f.write("\nSample Size Statistics\n")
        f.write("--------------------\n")
        for analysis_type in ['all', 'male', 'female']:
            f.write(f"\n{analysis_type.upper()}:\n")
            
            # Only calculate for non-empty values
            mutant_col = f'{analysis_type}_n_mutants'
            control_col = f'{analysis_type}_n_controls'
            
            mutant_data = df[df[mutant_col].notna()][mutant_col]
            control_data = df[df[control_col].notna()][control_col]
            
            if len(mutant_data) > 0:
                f.write(f"Mean mutants per gene/allele: {mutant_data.mean():.1f}\n")
            if len(control_data) > 0:
                f.write(f"Mean controls per gene/allele: {control_data.mean():.1f}\n")
        
        # Analysis-specific counts
        f.write("\n\nAnalysis-Specific Significant Genes\n")
        f.write("----------------------------------\n")
        
        # Count genes significant in specific analyses only
        all_only = analysis_results['all']['significant_genes'] - \
                  analysis_results['male']['significant_genes'] - \
                  analysis_results['female']['significant_genes']
        
        male_only = analysis_results['male']['significant_genes'] - \
                   analysis_results['all']['significant_genes'] - \
                   analysis_results['female']['significant_genes']
        
        female_only = analysis_results['female']['significant_genes'] - \
                     analysis_results['all']['significant_genes'] - \
                     analysis_results['male']['significant_genes']
        
        f.write(f"Significant only in ALL analysis: {len(all_only)}\n")
        f.write(f"Significant only in MALE analysis: {len(male_only)}\n")
        f.write(f"Significant only in FEMALE analysis: {len(female_only)}\n\n")
        
        if all_only:
            f.write("Genes significant only in ALL analysis:\n")
            f.write(format_gene_list(all_only))
            f.write("\n\n")
        
        if male_only:
            f.write("Genes significant only in MALE analysis:\n")
            f.write(format_gene_list(male_only))
            f.write("\n\n")
        
        if female_only:
            f.write("Genes significant only in FEMALE analysis:\n")
            f.write(format_gene_list(female_only))
            f.write("\n\n")

    print("Report generated successfully!")
    return output_path

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate ABR analysis report from detailed results')
    parser.add_argument('--results', '-r', 
                       default='results/multivariate/detailed_results.csv',
                       help='Path to detailed_results.csv file')
    parser.add_argument('--confirmed', '-c',
                       default='data/confirmed_deafness_genes.txt',
                       help='Path to confirmed deafness genes file')
    parser.add_argument('--candidate', '-a',
                       default='data/candidate_deafness_genes.txt', 
                       help='Path to candidate deafness genes file')
    parser.add_argument('--output', '-o',
                       help='Output file path (default: timestamped filename)')
    parser.add_argument('--threshold', '-t', type=float, default=0.001,
                       help='P-value significance threshold (default: 0.001)')
    
    args = parser.parse_args()
    
    # Check if files exist
    for file_path in [args.results, args.confirmed, args.candidate]:
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    
    try:
        output_path = generate_report(
            args.results, 
            args.confirmed, 
            args.candidate, 
            args.output,
            args.threshold
        )
        print(f"Analysis report saved to: {output_path}")
    except Exception as e:
        print(f"Error generating report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()