#!/bin/env python
# -*- coding: utf-8 -*-

"""
Test runner for the ABR multivariate batch analysis pipeline.

This module provides functionality to test the full batch processing pipeline
for Auditory Brainstem Response (ABR) data analysis. It executes the entire workflow
including data loading, statistical analysis, comparison with known gene sets,
and generation of visualisations and reports.

The module:
- Sets up the necessary environment for testing
- Executes the batch analysis process
- Compares results with confirmed and candidate deafness genes
- Generates comprehensive reports and visualisations
- Provides formatted output for easy inspection of results

This script can be run directly to perform a full test of the analysis pipeline
on a specified dataset, with results saved to a timestamped directory for review.

Author: Liam Barrett
Version: 1.0.1
"""

import sys
from pathlib import Path
from datetime import datetime

# Add the package directory to the path
PACKAGE_DIR = str(Path(__file__).parent.parent)
sys.path.insert(0, PACKAGE_DIR)
from abr_analysis.analysis.batch_processor import GeneBatchAnalyzer

def format_gene_list(genes, columns=3):
    """Format a list of genes into columns for pretty printing."""
    genes = sorted(list(genes))
    rows = []
    for i in range(0, len(genes), columns):
        row = genes[i:i + columns]
        rows.append('\t'.join(str(g).ljust(20) for g in row))
    return '\n'.join(rows)

def run_batch_analysis(data_path, output_dir="results/multivariate"):
    """Run full batch analysis and save results."""
    # Get paths to gene lists (two directories up in docs/)
    docs_dir = Path(__file__).parent.parent.parent.parent / "docs"
    confirmed_genes_path = docs_dir / "multivariate_confirmed_deafness_genes.txt"
    candidate_genes_path = docs_dir / "multivariate_candidate_deafness_genes.txt"

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir) / f"analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting batch analysis...")
    print(f"Results will be saved to: {output_dir}")

    # Initialize and run analyzer
    analyzer = GeneBatchAnalyzer(data_path)
    results = analyzer.analyze_all_genes()

    # Compare with known genes
    comparisons = analyzer.compare_with_known_genes(confirmed_genes_path, candidate_genes_path)

    # Generate report
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ABR Analysis Report\n")
        f.write("=================\n\n")

        # Overall statistics
        f.write("Overall Statistics\n")
        f.write("-----------------\n")
        f.write(f"Total genes analyzed: {len(results)}\n")

        # Count unique gene identifiers
        total_genes_analyzed = len(results['gene_symbol'].unique())
        f.write(f"Total unique genes analyzed: {total_genes_analyzed}\n")

        # Read the gene lists to get counts
        with open(confirmed_genes_path, 'r', encoding='utf-8') as cf:
            confirmed_genes = [line.strip() for line in cf if line.strip()]

        with open(candidate_genes_path, 'r', encoding='utf-8') as cf:
            candidate_genes = [line.strip() for line in cf if line.strip()]

        f.write(f"Confirmed deafness genes: {len(confirmed_genes)}\n")
        f.write(f"Candidate deafness genes: {len(candidate_genes)}\n\n")

        # Results for each analysis type
        for analysis_type in ['all', 'male', 'female']:
            f.write(f"\n{analysis_type.upper()} ANALYSIS\n")
            f.write("-" * (len(analysis_type) + 9) + "\n")

            comp = comparisons[analysis_type]

            total_sig = len(comp['found_in_confirmed']) + len(comp['found_in_candidate']) + len(comp['novel'])
            f.write(f"Total significant genes: {total_sig}\n")
            f.write(f"Found in confirmed deafness genes: {len(comp['found_in_confirmed'])}\n")
            f.write(f"Found in candidate deafness genes: {len(comp['found_in_candidate'])}\n")
            f.write(f"Novel candidates: {len(comp['novel'])}\n")
            f.write(f"Missed confirmed genes: {len(comp['missed_confirmed'])}\n")
            f.write(f"Missed candidate genes: {len(comp['missed_candidate'])}\n\n")

            f.write("Found in confirmed deafness genes:\n")
            f.write(format_gene_list(comp['found_in_confirmed']))
            f.write("\n\nFound in candidate deafness genes:\n")
            f.write(format_gene_list(comp['found_in_candidate']))
            f.write("\n\nNovel candidates:\n")
            f.write(format_gene_list(comp['novel']))
            f.write("\n\nMissed confirmed genes:\n")
            f.write(format_gene_list(comp['missed_confirmed']))
            f.write("\n\nMissed candidate genes:\n")
            f.write(format_gene_list(comp['missed_candidate']))
            f.write("\n\n")

        # Sample size statistics
        f.write("\nSample Size Statistics\n")
        f.write("--------------------\n")
        for analysis_type in ['all', 'male', 'female']:
            f.write(f"\n{analysis_type.upper()}:\n")
            f.write(f"Mean mutants per gene: {results[f'{analysis_type}_n_mutants'].mean():.1f}\n")
            f.write(f"Mean controls per gene: {results[f'{analysis_type}_n_controls'].mean():.1f}\n")

    # Save detailed results
    results.to_csv(output_dir / "detailed_results.csv", index=False)

    # Create visualizations
    print("\nGenerating visualizations...")
    analyzer.create_visualizations(output_dir)

    # Print summary to console
    print("\nAnalysis complete! Summary of findings:")
    print("-" * 40)
    for analysis_type in ['all', 'male', 'female']:
        comp = comparisons[analysis_type]
        total_sig = len(comp['found_in_confirmed']) + len(comp['found_in_candidate']) + len(comp['novel'])
        print(f"\n{analysis_type.upper()}:")
        print(f"Total significant: {total_sig}")
        print(f"Found in confirmed: {len(comp['found_in_confirmed'])}")
        print(f"Found in candidate: {len(comp['found_in_candidate'])}")
        print(f"Novel: {len(comp['novel'])}")

    print(f"\nFull results saved to: {output_dir}")
    return results, comparisons, output_dir

if __name__ == "__main__":
    # Path to your data
    DATA_PATH = "../../../data/processed/abr_full_data.csv"

    try:
        results, comparisons, output_dir = run_batch_analysis(DATA_PATH)
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nError during test: {e}")
        raise
