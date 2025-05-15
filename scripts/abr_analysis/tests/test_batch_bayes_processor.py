#!/bin/env python
# -*- coding: utf-8 -*-
# tests/test_batch_bayes_processor.py

"""
Test runner for the Bayesian ABR analysis pipeline.

This module provides functionality to test the Bayesian analysis pipeline for
Auditory Brainstem Response (ABR) data. It supports both full dataset analysis
and targeted testing of specific genes to verify algorithm performance. The module
includes tools for generating comprehensive reports, visualisations, and comparisons
with confirmed and candidate hearing loss genes.

The module offers two main testing approaches:
1. Full batch analysis of all genes in a dataset with statistical comparisons
2. Targeted analysis of a subset of genes for detailed validation

Results are saved with timestamps to facilitate comparison between runs and include:
- Statistical summaries and detailed result data
- Gene-specific visualisations with probability distributions
- Comprehensive text reports with gene categorisations
- Comparisons with known hearing loss genes

This script can be run directly to execute either a full analysis or a targeted
test on specified genes.

Author: Liam Barrett
Version: 1.0.1
"""

import sys
from pathlib import Path
from datetime import datetime
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the package directory to the path
PACKAGE_DIR = str(Path(__file__).parent.parent)
sys.path.insert(0, PACKAGE_DIR)

from abr_analysis.analysis.batch_bayes_processor import GeneBayesianAnalyzer

def format_gene_list(genes, columns=3):
    """Format a list of genes into columns for pretty printing."""
    genes = sorted(list(genes))
    rows = []
    for i in range(0, len(genes), columns):
        row = genes[i:i + columns]
        rows.append('\t'.join(str(g).ljust(20) for g in row))
    return '\n'.join(rows)

def run_batch_bayesian_analysis(data_path, output_dir="results/bayes", min_bf=3.0):
    """Run full batch Bayesian analysis and save results."""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir) / f"bayes_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting batch Bayesian analysis...")
    print(f"Results will be saved to: {output_dir}")

    # Initialize and run analyzer
    analyzer = GeneBayesianAnalyzer(data_path)
    results = analyzer.analyze_all_genes()

    # Get paths to confirmed and candidate gene lists
    # These are in the docs/ directory (two levels up from abr_analysis)
    docs_dir = Path(PACKAGE_DIR).parent / "docs"
    confirmed_genes_path = docs_dir / "multivariate_confirmed_deafness_genes.txt"
    candidate_genes_path = docs_dir / "multivariate_candidate_deafness_genes.txt"

    # Compare with confirmed and candidate genes
    if not confirmed_genes_path.exists() or not candidate_genes_path.exists():
        print(f"Warning: Gene list files not found at {docs_dir}")
        print("Expected files: multivariate_confirmed_deafness_genes.txt, multivariate_candidate_deafness_genes.txt")
        comparisons = None
    else:
        comparisons = analyzer.compare_with_known_genes(
            confirmed_genes_path,
            candidate_genes_path,
            min_bf=min_bf
        )

    # Generate report
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ABR Bayesian Analysis Report\n")
        f.write("==========================\n\n")

        # Overall statistics
        f.write("Overall Statistics\n")
        f.write("-----------------\n")
        f.write(f"Total genes analyzed: {len(results)}\n")

        # Count experimental groups analyzed
        group_count = 0
        for _, row in analyzer.results.iterrows():
            for analysis_type in ['all', 'male', 'female']:
                if not pd.isna(row.get(f'{analysis_type}_bayes_factor')):
                    group_count += 1
                    break

        f.write(f"Total experimental groups analyzed: {group_count}\n")
        f.write(f"Significance threshold: Bayes Factor > {min_bf}\n\n")

        # Gene set information
        confirmed_count = 0
        candidate_count = 0

        if comparisons:
            # Load gene lists to get counts
            with open(confirmed_genes_path, 'r', encoding='utf-8') as cf:
                confirmed_count = sum(1 for line in cf if line.strip())
            with open(candidate_genes_path, 'r', encoding='utf-8') as cf:
                candidate_count = sum(1 for line in cf if line.strip())

            f.write(f"Confirmed deafness genes: {confirmed_count}\n")
            f.write(f"Candidate deafness genes: {candidate_count}\n\n")

        # Resusts for each analysis type
        for analysis_type in ['all', 'male', 'female']:
            f.write(f"\n{analysis_type.upper()} ANALYSIS\n")
            f.write("-" * (len(analysis_type) + 9) + "\n")

            # Count significant genes
            sig_count = sum(results[f'{analysis_type}_bayes_factor'] > min_bf)
            f.write(f"Total significant genes: {sig_count}\n\n")

            if comparisons:
                comp = comparisons[analysis_type]

                # Write confirmed gene results
                f.write("CONFIRMED GENES:\n")
                f.write(f"Found: {len(comp['found_in_confirmed'])} of {confirmed_count} ({len(comp['found_in_confirmed'])/confirmed_count*100:.1f}%)\n")
                f.write(f"Missed: {len(comp['missed_confirmed'])} of {confirmed_count} ({len(comp['missed_confirmed'])/confirmed_count*100:.1f}%)\n\n")

                f.write("Found confirmed genes:\n")
                f.write(format_gene_list(comp['found_in_confirmed']))
                f.write("\n\nMissed confirmed genes:\n")
                f.write(format_gene_list(comp['missed_confirmed']))
                f.write("\n\n")

                # Write candidate gene results
                f.write("CANDIDATE GENES:\n")
                f.write(f"Found: {len(comp['found_in_candidate'])} of {candidate_count} ({len(comp['found_in_candidate'])/candidate_count*100:.1f}%)\n")
                f.write(f"Missed: {len(comp['missed_candidate'])} of {candidate_count} ({len(comp['missed_candidate'])/candidate_count*100:.1f}%)\n\n")

                f.write("Found candidate genes:\n")
                f.write(format_gene_list(comp['found_in_candidate']))
                f.write("\n\nMissed candidate genes:\n")
                f.write(format_gene_list(comp['missed_candidate']))
                f.write("\n\n")

                # Write novel gene results
                f.write("NOVEL GENES:\n")
                f.write(f"New potential hearing loss genes: {len(comp['novel'])}\n\n")

                f.write("Novel genes:\n")
                f.write("Novel genes:\n")
                f.write(format_gene_list(comp['novel']))
                f.write("\n\n")

        # Evidence level breakdown
        f.write("\nEvidence Level Breakdown\n")
        f.write("----------------------\n")
        evidence_counts = results['all_evidence'].value_counts()
        for evidence, count in evidence_counts.items():
            f.write(f"{evidence}: {count} genes ({count/len(results)*100:.1f}%)\n")

        # Sample size statistics
        f.write("\nSample Size Statistics\n")
        f.write("--------------------\n")
        for analysis_type in ['all', 'male', 'female']:
            f.write(f"\n{analysis_type.upper()}:\n")
            f.write(f"Mean mutants per gene: {results[f'{analysis_type}_n_mutants'].mean():.1f}\n")
            f.write(f"Mean controls per gene: {results[f'{analysis_type}_n_controls'].mean():.1f}\n")

        # Center distribution
        f.write("\nCenter Distribution\n")
        f.write("-----------------\n")
        center_counts = {}
        for _, row in results.iterrows():
            center = row.get('all_center')
            if not pd.isna(center):
                center_counts[center] = center_counts.get(center, 0) + 1

        for center, count in sorted(center_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                f.write(f"{center}: {count} genes\n")

    # Save detailed results
    results.to_csv(output_dir / "gene_summary.csv", index=False)
    analyzer.results.to_csv(output_dir / "detailed_group_results.csv", index=False)

    # Create visualizations
    print("\nGenerating visualizations...")
    analyzer.create_visualizations(output_dir, min_bf)

    # Print summary to console
    print("\nBayesian analysis complete! Summary of findings:")
    print("-" * 50)

    if comparisons:
        for analysis_type in ['all', 'male', 'female']:
            comp = comparisons[analysis_type]
            print(f"\n{analysis_type.upper()}:")
            print(f"Total significant: {sum(results[f'{analysis_type}_bayes_factor'] > min_bf)}")
            print(f"Found in confirmed: {len(comp['found_in_confirmed'])} of {confirmed_count} ({len(comp['found_in_confirmed'])/confirmed_count*100:.1f}%)")
            print(f"Found in candidate: {len(comp['found_in_candidate'])} of {candidate_count} ({len(comp['found_in_candidate'])/candidate_count*100:.1f}%)")
            print(f"Novel: {len(comp['novel'])}")

    # Print evidence breakdown
    print("\nEvidence Level Breakdown:")
    for evidence, count in evidence_counts.items():
        print(f"{evidence}: {count} genes ({count/len(results)*100:.1f}%)")

    print(f"\nFull results saved to: {output_dir}")
    return results, comparisons, output_dir

def test_on_sample_genes(data_path, gene_list=None, output_dir="results/bayes"):
    """Run Bayesian analysis on a small set of genes to verify the pipeline."""
    if gene_list is None:
        # Default to testing these three genes - known hearing loss (Adgrv1),
        # a gene with mild hearing loss (Nptn), and a gene with visually insufficient evidence (Abca2)
        gene_list = ['Adgrv1', 'Nptn', 'Abca2']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir) / f"sample_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning sample test on genes: {', '.join(gene_list)}")
    print(f"Results will be saved to: {output_dir}")

    # Initialize analyzer
    analyzer = GeneBayesianAnalyzer(data_path)

    # Process each gene
    results_list = []
    all_groups = []

    for gene in gene_list:
        print(f"\nProcessing gene: {gene}")

        # Find all experimental groups for this gene
        exp_groups = analyzer.matcher.find_experimental_groups(gene)

        if not exp_groups:
            print(f"No experimental groups found for gene {gene}. Skipping.")
            continue

        print(f"Found {len(exp_groups)} experimental groups for {gene}")
        all_groups.extend(exp_groups)

        # Analyze each experimental group
        for group in exp_groups:
            group_desc = f"{gene} ({group['allele_symbol']}, {group['zygosity']}, {group['phenotyping_center']})"
            print(f"  Analyzing group: {group_desc}")

            # Analyze just for 'all' data
            analysis = analyzer.analyze_experimental_group(group)

            if analysis is None:
                print(f"  Analysis failed for group {group_desc}. Skipping.")
                continue

            # Add to results
            results_list.append(analysis)

            # Get analysis key for visualization
            analysis_key = analysis['analysis_key']
            print(f"  BF: {analysis['bayes_factor']:.2f}, P(HL): {analysis['p_hearing_loss']:.3f}")

            # Create visualization for this gene
            gene_dir = output_dir / 'visuals' / gene
            gene_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Get control and mutant data
                control_data = analyzer.control_profiles[analysis_key]
                mutant_data = analyzer.mutant_profiles[analysis_key]
                bayesian_model = analyzer.bayesian_models[analysis_key]

                # Extract profiles
                control_profiles = analyzer.matcher.get_control_profiles(control_data, analyzer.freq_cols)
                group_info_for_extract = {'data': mutant_data}
                mutant_profiles = analyzer.matcher.get_experimental_profiles(
                    group_info_for_extract, analyzer.freq_cols
                )

                # Remove NaN values
                control_profiles = control_profiles[~np.any(np.isnan(control_profiles), axis=1)]
                mutant_profiles = mutant_profiles[~np.any(np.isnan(mutant_profiles), axis=1)]

                # Create visualization
                fig = bayesian_model.plot_results(control_profiles, mutant_profiles, gene)

                # Add group information to title
                fig.suptitle(f"{gene} - {group['allele_symbol']} ({group['zygosity']}) - {group['phenotyping_center']}",
                            fontsize=14)

                fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
                safe_filename = f"{gene}_{group['allele_symbol'].replace('<', '').replace('>', '')}" \
                               f"_{group['zygosity']}_{group['phenotyping_center']}_bayesian.png"
                fig.savefig(gene_dir / safe_filename)
                plt.close(fig)

            except (ValueError, IOError, KeyError, AttributeError,
                    TypeError, IndexError, RuntimeError) as e:
                print(f"  Error creating visualisation: {str(e)}")
                traceback.print_exc()

    # Save results to CSV
    if results_list:
        # Convert results to DataFrame
        results_df = pd.DataFrame([
            {k: v for k, v in r.items() if not callable(v) and k != 'effect_sizes'
                                       and not isinstance(v, np.ndarray)}
            for r in results_list
        ])

        # Add effect sizes columns
        for i, freq in enumerate(analyzer.freq_cols):
            freq_name = freq.split()[0]
            results_df[f'effect_{freq_name}'] = [r['effect_sizes'][i] if i < len(r['effect_sizes']) else np.nan for r in results_list]

        results_df.to_csv(output_dir / "sample_results.csv", index=False)
    else:
        results_df = pd.DataFrame()
        print("No results generated.")

    # Create simple report
    report_path = output_dir / "sample_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Sample Test Results\n")
        f.write("==================\n\n")

        f.write(f"Genes analyzed: {', '.join(gene_list)}\n")
        f.write(f"Total experimental groups: {len(all_groups)}\n")
        f.write(f"Successful analyses: {len(results_list)}\n\n")

        for gene in gene_list:
            f.write(f"Gene: {gene}\n")
            f.write("-" * (len(gene) + 6) + "\n")

            # Get all results for this gene
            gene_results = [r for r in results_list if r['gene_symbol'] == gene]

            if gene_results:
                f.write(f"Experimental groups: {len(gene_results)}\n\n")

                for i, result in enumerate(gene_results):
                    allele = result['allele_symbol']
                    zygosity = result['zygosity']
                    center = result['center']

                    f.write(f"Group {i+1}: {allele} ({zygosity}) - {center}\n")
                    f.write(f"  Bayes Factor: {result['bayes_factor']:.2f}\n")
                    f.write(f"  P(Hearing Loss): {result['p_hearing_loss']:.3f}\n")
                    f.write(f"  95% HDI: [{result['hdi_lower']:.3f}, {result['hdi_upper']:.3f}]\n")
                    f.write(f"  Sample size: {result['n_mutants']} mutants, {result['n_controls']} controls\n\n")

                    # Evidence interpretation
                    bf = result['bayes_factor']
                    if bf > 100:
                        evidence = "Extreme"
                    elif bf > 30:
                        evidence = "Very Strong"
                    elif bf > 10:
                        evidence = "Strong"
                    elif bf > 3:
                        evidence = "Substantial"
                    else:
                        evidence = "Weak/None"

                    f.write(f"  Evidence level: {evidence}\n\n")
            else:
                f.write("No results available\n\n")

    print(f"\nSample test complete! Results saved to: {output_dir}")
    return results_df, output_dir

if __name__ == "__main__":
    # Path to your data
    DATA_PATH = "../../../data/processed/abr_full_data.csv"

    # To test on just a few genes:
    test_on_sample_genes(DATA_PATH)

    # Or run the full analysis:
    #try:
    #    results, comparisons, output_dir = run_batch_bayesian_analysis(data_path)
    #    print("\nTest completed successfully!")
    #except Exception as e:
    #    print(f"\nError during test: {e}")
    #    raise
