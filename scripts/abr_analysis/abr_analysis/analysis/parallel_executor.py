#!/bin/env python
# -*- coding: utf-8 -*-
# abr_analysis/analysis/parallel_executor.py

"""
Parallel processing module for Bayesian ABR data analysis.

This module implements a distributed computing framework that enables efficient
parallel analysis of large ABR datasets across multiple CPU cores. It divides the
gene analysis workload into small, manageable chunks that can be processed
independently, significantly reducing the overall computation time.

Key features:
- Multi-process execution using Python's ProcessPoolExecutor
- Batch processing with intermediate result storage to prevent data loss
- Gene-level and experimental group-level parallelization options
- Progress tracking and comprehensive reporting
- Automated visualization generation for result interpretation
- Statistical comparison with known hearing loss genes

The module is designed for high-throughput analysis of IMPC ABR data, handling
thousands of genes and experimental groups efficiently while managing memory
constraints and providing detailed feedback on analysis progress.

Author: Liam Barrett
Version: 1.0.0
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
import traceback
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from .batch_bayes_processor import GeneBayesianAnalyzer

def process_experimental_group(group_info, data_path, save_dir=None):
    """Process a single experimental group and return the result."""
    # Create a new analyzer for this process
    analyzer = GeneBayesianAnalyzer(data_path)

    gene = group_info['gene_symbol']
    allele = group_info['allele_symbol']
    zygosity = group_info['zygosity']
    center = group_info['phenotyping_center']

    group_desc = f"{gene} ({allele}, {zygosity}, {center})"
    print(f"Processing group: {group_desc}")

    # Analyze experimental group
    try:
        # Analyze for all data, males only, and females only
        analyses = {
            'all': analyzer.analyze_experimental_group(group_info),
            'male': analyzer.analyze_experimental_group(group_info, 'male'),
            'female': analyzer.analyze_experimental_group(group_info, 'female')
        }

        # Create result dictionary
        result = {
            'gene_symbol': gene,
            'allele_symbol': allele,
            'zygosity': zygosity,
            'center': center
        }

        # Record results and save models/visualizations
        for analysis_type, analysis in analyses.items():
            if analysis:
                # Include main statistics
                result[f'{analysis_type}_bayes_factor'] = analysis['bayes_factor']
                result[f'{analysis_type}_p_hearing_loss'] = analysis['p_hearing_loss']
                result[f'{analysis_type}_hdi_lower'] = analysis['hdi_lower']
                result[f'{analysis_type}_hdi_upper'] = analysis['hdi_upper']
                result[f'{analysis_type}_n_mutants'] = analysis['n_mutants']
                result[f'{analysis_type}_n_controls'] = analysis['n_controls']
                result[f'{analysis_type}_analysis_key'] = analysis['analysis_key']

                # Include effect sizes for each frequency
                freq_cols = analyzer.freq_cols
                for i, freq in enumerate(freq_cols):
                    freq_name = freq.split()[0]
                    result[f'{analysis_type}_effect_{freq_name}'] = analysis['effect_sizes'][i]

                # Save model and create visualizations if save_dir is provided
                if save_dir:
                    # Find the model in the analyzer's cache
                    analysis_key = analysis['analysis_key']

                    # Create directories
                    model_dir = Path(save_dir) / "models" / gene
                    visuals_dir = Path(save_dir) / "visuals" / gene
                    model_dir.mkdir(parents=True, exist_ok=True)
                    visuals_dir.mkdir(parents=True, exist_ok=True)

                    # Sanitize allele name for filesystem
                    safe_allele = str(allele).replace('<', '').replace('>', '').replace('/', '_')
                    model_save_dir = model_dir / f"{safe_allele}_{zygosity}_{center}_{analysis_type}"

                    if analysis_key in analyzer.bayesian_models:
                        # DIRECT MODEL SAVING - Save immediately after creation
                        print(f"Saving model for {group_desc} ({analysis_type}) to {model_save_dir}")
                        try:
                            model_save_dir.mkdir(parents=True, exist_ok=True)
                            bayesian_model = analyzer.bayesian_models[analysis_key]
                            save_success = bayesian_model.save_model(model_save_dir)

                            if save_success:
                                print(f"Successfully saved model for {group_desc} ({analysis_type})")

                                # Also create visualization if significant (BF > 3)
                                if analysis['bayes_factor'] > 3 and analysis_type == 'all':
                                    try:
                                        # Get control and mutant data
                                        control_data = analyzer.control_profiles[analysis_key]
                                        mutant_data = analyzer.mutant_profiles[analysis_key]

                                        # Extract profiles
                                        control_profiles = analyzer.matcher.get_control_profiles(control_data, analyzer.freq_cols)
                                        group_info_for_extract = {'data': mutant_data}

                                        if analysis_type in ['male', 'female']:
                                            mutant_profiles = analyzer.matcher.get_experimental_profiles(
                                                group_info_for_extract, analyzer.freq_cols, analysis_type
                                            )
                                        else:
                                            mutant_profiles = analyzer.matcher.get_experimental_profiles(
                                                group_info_for_extract, analyzer.freq_cols
                                            )

                                        # Remove NaN values
                                        control_profiles = control_profiles[~np.any(np.isnan(control_profiles), axis=1)]
                                        mutant_profiles = mutant_profiles[~np.any(np.isnan(mutant_profiles), axis=1)]

                                        # Create visualization
                                        fig = bayesian_model.plot_results(control_profiles, mutant_profiles, gene)

                                        # Add group information to title
                                        fig.suptitle(f"{gene} - {allele} ({zygosity}) - {center} ({analysis_type})", fontsize=14)
                                        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle

                                        # Save figure
                                        safe_filename = f"{gene}_{safe_allele}_{zygosity}_{center}_{analysis_type}_bayesian.png"
                                        fig.savefig(visuals_dir / safe_filename)
                                        plt.close(fig)
                                        print(f"Successfully saved visualization for {group_desc} ({analysis_type})")
                                    except Exception as viz_err:
                                        print(f"Error creating visualization for {group_desc} ({analysis_type}): {str(viz_err)}")
                            else:
                                print(f"Failed to save model for {group_desc} ({analysis_type})")
                        except Exception as model_err:
                            print(f"Error saving model for {group_desc} ({analysis_type}): {str(model_err)}")
                            traceback.print_exc()
                    else:
                        print(f"Warning: Model for analysis key '{analysis_key}' not found in cache")
            else:
                # Set missing values for metrics
                metrics = ['bayes_factor', 'p_hearing_loss', 'hdi_lower', 'hdi_upper',
                          'n_mutants', 'n_controls', 'analysis_key']
                for metric in metrics:
                    result[f'{analysis_type}_{metric}'] = np.nan

                # Set missing values for effect sizes
                freq_cols = analyzer.freq_cols
                for freq in freq_cols:
                    freq_name = freq.split()[0]
                    result[f'{analysis_type}_effect_{freq_name}'] = np.nan

        return result

    except Exception as e:
        print(f"Error processing group {group_desc}: {str(e)}")
        traceback.print_exc()
        return None

def process_gene_groups(gene, data_path, save_dir=None):
    """Process all experimental groups for a gene and return the results."""
    # Create a new analyzer to find groups
    analyzer = GeneBayesianAnalyzer(data_path)

    # Find all experimental groups for this gene
    exp_groups = analyzer.matcher.find_experimental_groups(gene)

    if not exp_groups:
        print(f"No experimental groups found for gene {gene}. Skipping.")
        return []

    print(f"Found {len(exp_groups)} experimental groups for {gene}")

    # Process each experimental group
    results = []
    for group in exp_groups:
        result = process_experimental_group(group, data_path, save_dir)
        if result:
            results.append(result)

    return results

def run_parallel_analysis(data_path, output_dir="results/parallel_bayes", min_bf=3.0,
                          n_processes=None, batch_size=10):
    """Run full batch Bayesian analysis in parallel and save results."""
    # Create output directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir) / f"parallel_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    models_dir = output_dir / "models"
    visuals_dir = output_dir / "visuals"
    models_dir.mkdir(exist_ok=True)
    visuals_dir.mkdir(exist_ok=True)

    print("\nStarting parallel batch Bayesian analysis...")
    print(f"Results will be saved to: {output_dir}")

    # Initialize analyzer to get gene list
    print("Loading data and preparing gene list...")
    temp_analyzer = GeneBayesianAnalyzer(data_path)
    mutants = temp_analyzer.data[temp_analyzer.data['biological_sample_group'] == 'experimental']
    genes = mutants['gene_symbol'].unique()
    genes = genes[~pd.isna(genes)]  # Remove NaN values

    # Set up multiprocessing
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)

    print(f"Analyzing {len(genes)} genes using {n_processes} processes")

    # Process in batches to track progress and save intermediate results
    all_results = []
    start_time = time.time()

    # Split genes into batches
    gene_batches = [genes[i:i+batch_size] for i in range(0, len(genes), batch_size)]

    for batch_index, gene_batch in enumerate(gene_batches):
        batch_start = time.time()
        print(f"\nProcessing batch {batch_index+1}/{len(gene_batches)} ({len(gene_batch)} genes)...")

        batch_results = []
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            # Submit all genes in this batch
            futures = {executor.submit(process_gene_groups, gene, data_path, output_dir): gene for gene in gene_batch}

            # Process results as they complete
            for future in tqdm(futures, total=len(gene_batch), desc=f"Batch {batch_index+1}"):
                gene = futures[future]
                try:
                    results = future.result()
                    batch_results.extend(results)
                except (ValueError, KeyError, AttributeError, TypeError, IndexError, RuntimeError,
                        np.linalg.LinAlgError, MemoryError, IOError) as e:
                    print(f"Error processing gene {gene}: {str(e)}")
                    traceback.print_exc()

        all_results.extend(batch_results)
        batch_time = time.time() - batch_start
        print(f"Batch {batch_index+1} completed in {batch_time:.2f} seconds")

        # Save intermediate results
        if batch_results:
            interim_df = pd.DataFrame(batch_results)
            interim_df.to_csv(output_dir / f"interim_results_batch_{batch_index+1}.csv", index=False)
            print(f"Saved interim results ({len(interim_df)} experimental groups processed so far)")

    # Convert all results to DataFrame
    if not all_results:
        print("No results were generated. Check for errors in the analysis.")
        return None, None, output_dir

    results_df = pd.DataFrame(all_results)

    # Create a gene-level summary
    gene_summary = create_gene_summary(results_df, temp_analyzer.freq_cols)

    # Add classification based on Bayes factors
    for analysis_type in ['all', 'male', 'female']:
        bf_col = f'{analysis_type}_bayes_factor'
        if bf_col in gene_summary.columns:
            evidence_col = f'{analysis_type}_evidence'

            gene_summary[evidence_col] = 'Insufficient data'
            mask = ~gene_summary[bf_col].isna()

            # Classify based on Bayes factor
            gene_summary.loc[mask & (gene_summary[bf_col] > 100), evidence_col] = 'Extreme'
            gene_summary.loc[mask & (gene_summary[bf_col] <= 100) & (gene_summary[bf_col] > 30), evidence_col] = 'Very Strong'
            gene_summary.loc[mask & (gene_summary[bf_col] <= 30) & (gene_summary[bf_col] > 10), evidence_col] = 'Strong'
            gene_summary.loc[mask & (gene_summary[bf_col] <= 10) & (gene_summary[bf_col] > 3), evidence_col] = 'Substantial'
            gene_summary.loc[mask & (gene_summary[bf_col] <= 3), evidence_col] = 'Weak/None'

    # Save full results
    results_df.to_csv(output_dir / "detailed_group_results.csv", index=False)
    gene_summary.to_csv(output_dir / "gene_summary.csv", index=False)

    # Generate summary report with confirmed and candidate gene comparisons
    total_time = time.time() - start_time

    # Get paths to gene lists (two levels up from package_dir)
    package_dir = Path(__file__).parent.parent.parent
    data_dir = package_dir.parent / "data"
    confirmed_genes_path = data_dir / "multivariate_confirmed_deafness_genes.txt"
    candidate_genes_path = data_dir / "multivariate_candidate_deafness_genes.txt"

    # Compare with known genes if available
    if confirmed_genes_path.exists() and candidate_genes_path.exists():
        comparisons = compare_with_known_genes(
            gene_summary,
            confirmed_genes_path,
            candidate_genes_path,
            min_bf
        )

        # Generate summary report with comparisons
        write_summary_report(
            output_dir / "summary.txt",
            gene_summary,
            comparisons,
            total_time,
            n_processes,
            min_bf,
            confirmed_genes_path,
            candidate_genes_path
        )
    else:
        print(f"Warning: Gene list files not found at {data_dir}")
        write_summary_report(
            output_dir / "summary.txt",
            gene_summary,
            None,
            total_time,
            n_processes,
            min_bf
        )

    # Create summary visualizations
    create_visualizations(gene_summary, output_dir, confirmed_genes_path, candidate_genes_path, min_bf)

    print(f"\nAnalysis completed in {total_time:.2f} seconds")
    print(f"Found {len(gene_summary)} genes with results out of {len(genes)} genes")
    print(f"Results saved to {output_dir}")
    print(f"Models saved to {models_dir}")
    print(f"Visualizations saved to {visuals_dir}")

    # Check if models were saved
    models_dir = output_dir / "models"
    if models_dir.exists():
        model_count = sum(1 for _ in models_dir.glob("**/*.json"))
        print(f"Saved {model_count} model specification files")
    else:
        print("Warning: No models directory found")

    return results_df, gene_summary, output_dir

def create_gene_summary(results_df, freq_cols):
    """Create a gene-level summary from all experimental groups."""
    if len(results_df) == 0:
        return pd.DataFrame()

    # Get unique genes
    genes = results_df['gene_symbol'].unique()

    summary_data = []

    for gene in genes:
        gene_data = results_df[results_df['gene_symbol'] == gene]

        # Initialize summary row
        summary = {'gene_symbol': gene}

        # For each analysis type, find the experimental group with the highest Bayes factor
        for analysis_type in ['all', 'male', 'female']:
            bf_col = f'{analysis_type}_bayes_factor'

            # Check if we have valid data
            valid_data = gene_data[~gene_data[bf_col].isna()]

            if len(valid_data) > 0:
                # Find the row with the highest Bayes factor
                best_row = valid_data.loc[valid_data[bf_col].idxmax()]

                # Copy key statistics to summary
                metrics = ['bayes_factor', 'p_hearing_loss', 'hdi_lower', 'hdi_upper',
                          'n_mutants', 'n_controls', 'analysis_key']
                for metric in metrics:
                    summary[f'{analysis_type}_{metric}'] = best_row[f'{analysis_type}_{metric}']

                # Copy effect sizes
                for freq in freq_cols:
                    freq_name = freq.split()[0]
                    summary[f'{analysis_type}_effect_{freq_name}'] = best_row[f'{analysis_type}_effect_{freq_name}']

                # Add allele info from the best row
                summary[f'{analysis_type}_allele'] = best_row['allele_symbol']
                summary[f'{analysis_type}_zygosity'] = best_row['zygosity']
                summary[f'{analysis_type}_center'] = best_row['center']
            else:
                # Set missing values
                metrics = ['bayes_factor', 'p_hearing_loss', 'hdi_lower', 'hdi_upper',
                          'n_mutants', 'n_controls', 'analysis_key', 'allele', 'zygosity', 'center']
                for metric in metrics:
                    summary[f'{analysis_type}_{metric}'] = np.nan

                # Set missing values for effect sizes
                for freq in freq_cols:
                    freq_name = freq.split()[0]
                    summary[f'{analysis_type}_effect_{freq_name}'] = np.nan

        summary_data.append(summary)

    return pd.DataFrame(summary_data)

def compare_with_known_genes(results_df, confirmed_genes_path, candidate_genes_path, min_bf=3.0):
    """
    Compare results with confirmed and candidate deafness genes.

    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results from the analysis
    confirmed_genes_path : str or Path
        Path to the confirmed deafness genes file
    candidate_genes_path : str or Path
        Path to the candidate deafness genes file
    min_bf : float
        Minimum Bayes Factor threshold for significance

    Returns:
    --------
    dict
        Comparison results for each analysis type
     """
    # Load confirmed and candidate gene lists
    with open(confirmed_genes_path, 'r', encoding='utf-8') as f:
        confirmed_genes = set(line.strip() for line in f if line.strip())

    with open(candidate_genes_path, 'r', encoding='utf-8') as f:
        candidate_genes = set(line.strip() for line in f if line.strip())

    # Create sets of significant genes for each analysis type
    significant_genes = {}
    for analysis_type in ['all', 'male', 'female']:
        bf_col = f'{analysis_type}_bayes_factor'

        if bf_col in results_df.columns:
            # Get significant genes
            sig_genes = set(results_df[results_df[bf_col] > min_bf]['gene_symbol'])
            significant_genes[analysis_type] = sig_genes

    # Generate comparison results
    comparisons = {}
    for analysis_type, sig_genes in significant_genes.items():
        comparisons[analysis_type] = {
            'found_in_confirmed': confirmed_genes & sig_genes,
            'found_in_candidate': candidate_genes & sig_genes,
            'novel': sig_genes - confirmed_genes - candidate_genes,
            'missed_confirmed': confirmed_genes - sig_genes,
            'missed_candidate': candidate_genes - sig_genes
        }

    return comparisons

def format_gene_list(genes, columns=3):
    """Format a list of genes into columns for pretty printing."""
    genes = sorted(list(genes))
    rows = []
    for i in range(0, len(genes), columns):
        row = genes[i:i + columns]
        rows.append('\t'.join(str(g).ljust(20) for g in row))
    return '\n'.join(rows)

def write_summary_report(output_path, gene_summary, comparisons, total_time, n_processes, min_bf,
                         confirmed_genes_path=None, candidate_genes_path=None):
    """Write a summary report of the analysis results."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Parallel Bayesian Analysis Summary\n")
        f.write("================================\n\n")
        f.write(f"Analysis completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {total_time:.2f} seconds\n")
        f.write(f"Processes used: {n_processes}\n")
        f.write(f"Total genes analyzed: {len(gene_summary)}\n\n")

        # Count experimental groups analyzed
        group_count = 0
        for analysis_type in ['all', 'male', 'female']:
            group_key = f'{analysis_type}_analysis_key'
            if group_key in gene_summary.columns:
                group_count += sum(~gene_summary[group_key].isna())

        f.write(f"Total experimental groups successfully analyzed: {group_count}\n\n")

        # Gene set information
        if confirmed_genes_path and candidate_genes_path:
            # Load gene lists to get counts
            with open(confirmed_genes_path, 'r', encoding='utf-8') as cf:
                confirmed_count = sum(1 for line in cf if line.strip())
            with open(candidate_genes_path, 'r', encoding='utf-8') as cf:
                candidate_count = sum(1 for line in cf if line.strip())

            f.write(f"Confirmed deafness genes: {confirmed_count}\n")
            f.write(f"Candidate deafness genes: {candidate_count}\n\n")

        # Results for each analysis type
        for analysis_type in ['all', 'male', 'female']:
            f.write(f"\n{analysis_type.upper()} ANALYSIS\n")
            f.write("-" * (len(analysis_type) + 9) + "\n")

            # Count significant genes
            bf_col = f'{analysis_type}_bayes_factor'
            if bf_col in gene_summary.columns:
                sig_count = sum(gene_summary[bf_col] > min_bf)
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
                    f.write(format_gene_list(comp['novel']))
                    f.write("\n\n")
            else:
                f.write("No analysis results available.\n")

        # Evidence level breakdown
        f.write("\nEvidence Level Breakdown\n")
        f.write("----------------------\n")
        if 'all_evidence' in gene_summary.columns:
            evidence_counts = gene_summary['all_evidence'].value_counts()
            for evidence, count in evidence_counts.items():
                f.write(f"{evidence}: {count} genes ({count/len(gene_summary)*100:.1f}%)\n")

        # Sample size statistics
        f.write("\nSample Size Statistics\n")
        f.write("--------------------\n")
        for analysis_type in ['all', 'male', 'female']:
            mutant_col = f'{analysis_type}_n_mutants'
            control_col = f'{analysis_type}_n_controls'

            if mutant_col in gene_summary.columns and control_col in gene_summary.columns:
                f.write(f"\n{analysis_type.upper()}:\n")
                f.write(f"Mean mutants per gene: {gene_summary[mutant_col].mean():.1f}\n")
                f.write(f"Mean controls per gene: {gene_summary[control_col].mean():.1f}\n")

        # Center distribution
        f.write("\nCenter Distribution\n")
        f.write("-----------------\n")
        center_counts = {}
        for _, row in gene_summary.iterrows():
            center = row.get('all_center')
            if not pd.isna(center):
                center_counts[center] = center_counts.get(center, 0) + 1

        for center, count in sorted(center_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                f.write(f"{center}: {count} genes\n")
def create_visualizations(gene_summary, output_dir, confirmed_genes_path=None,
                          candidate_genes_path=None, min_bf=3.0):
    """
    Create summary visualizations from analysis results.
    """
    output_dir = Path(output_dir) / "visualizations"
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Bayes Factor Distribution
    plt.figure(figsize=(10, 6))
    bfs = gene_summary['all_bayes_factor'].replace([np.inf, -np.inf], np.nan).dropna()

    if not bfs.empty:
        # Log transform for better visualization
        log_bfs = np.log10(bfs + 0.1)  # Add small constant to handle zeros

        sns.histplot(log_bfs, kde=True)
        plt.axvline(x=np.log10(3), color='r', linestyle='--', label='BF=3')
        plt.axvline(x=np.log10(10), color='g', linestyle='--', label='BF=10')
        plt.axvline(x=np.log10(100), color='b', linestyle='--', label='BF=100')

        plt.xlabel('log10(Bayes Factor)')
        plt.ylabel('Count')
        plt.title('Distribution of Bayes Factors')
        plt.legend()
        plt.savefig(output_dir / 'bayes_factor_distribution.png')
    plt.close()

    # 2. Evidence Classification Pie Chart
    if 'all_evidence' in gene_summary.columns:
        plt.figure(figsize=(10, 6))
        evidence_counts = gene_summary['all_evidence'].value_counts()

        if not evidence_counts.empty:
            plt.pie(evidence_counts, labels=evidence_counts.index, autopct='%1.1f%%')
            plt.title('Distribution of Evidence Levels')
            plt.savefig(output_dir / 'evidence_classification.png')
        plt.close()

    # 3. Sex Comparison Plot
    plt.figure(figsize=(10, 6))

    # Check if we have male and female data
    has_male_data = 'male_bayes_factor' in gene_summary.columns and not gene_summary['male_bayes_factor'].isna().all()
    has_female_data = 'female_bayes_factor' in gene_summary.columns and not gene_summary['female_bayes_factor'].isna().all()

    if has_male_data and has_female_data:
        male_sig = gene_summary['male_bayes_factor'] > min_bf
        female_sig = gene_summary['female_bayes_factor'] > min_bf

        # Fill NaN values with False for boolean comparisons
        male_sig = male_sig.fillna(False)
        female_sig = female_sig.fillna(False)

        venn_data = {
            'Male Only': sum(male_sig & ~female_sig),
            'Female Only': sum(female_sig & ~male_sig),
            'Both': sum(male_sig & female_sig)
        }

        # Check if we have any data to plot
        if sum(venn_data.values()) > 0:
            plt.pie(venn_data.values(), labels=venn_data.keys(), autopct='%1.1f%%')
            plt.title(f'Sex-specific Genes (BF > {min_bf})')
            plt.savefig(output_dir / 'sex_comparison.png')
    plt.close()

    # 4. Effect Size Distribution
    plt.figure(figsize=(12, 6))
    sig_genes = gene_summary[gene_summary['all_bayes_factor'] > min_bf]

    # Calculate mean effect size across frequencies
    effect_cols = [col for col in sig_genes.columns if col.startswith('all_effect_')]

    if effect_cols and not sig_genes.empty:
        # Check if we have any effect data
        if not sig_genes[effect_cols].dropna(how='all').empty:
            sig_genes['mean_effect'] = sig_genes[effect_cols].mean(axis=1)

            # Filter out NaN mean effects
            sig_genes = sig_genes.dropna(subset=['mean_effect'])

            if not sig_genes.empty:
                sns.histplot(data=sig_genes, x='mean_effect', bins=20)
                plt.xlabel('Mean Effect Size (dB)')
                plt.title(f'Effect Size Distribution (BF > {min_bf})')
                plt.savefig(output_dir / 'effect_size_distribution.png')
    plt.close()

    # 5. Center Comparison
    plt.figure(figsize=(12, 6))

    # Make sure we have center data
    if 'all_center' in gene_summary.columns:
        sig_with_center = gene_summary[(gene_summary['all_bayes_factor'] > min_bf) &
                                       (~gene_summary['all_center'].isna())]

        if not sig_with_center.empty:
            center_counts = sig_with_center.groupby('all_center').size()

            if not center_counts.empty:
                center_counts.plot(kind='bar')
                plt.title(f'Genes with BF > {min_bf} by Center')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / 'center_comparison.png')
    plt.close()

    # 6. Confirmed and Candidate Gene Comparison
    if confirmed_genes_path and candidate_genes_path and Path(confirmed_genes_path).exists() and Path(candidate_genes_path).exists():
        # Load gene lists
        with open(confirmed_genes_path, 'r', encoding='utf-8') as f:
            confirmed_genes = set(line.strip() for line in f if line.strip())

        with open(candidate_genes_path, 'r', encoding='utf-8') as f:
            candidate_genes = set(line.strip() for line in f if line.strip())

        # Get significant genes
        if 'all_bayes_factor' in gene_summary.columns:
            significant = set(gene_summary[gene_summary['all_bayes_factor'] > min_bf]['gene_symbol'].dropna())

            if significant:
                # Calculate overlaps
                found_confirmed = len(significant & confirmed_genes)
                found_candidate = len(significant & candidate_genes)
                novel = len(significant - confirmed_genes - candidate_genes)

                if found_confirmed > 0 or found_candidate > 0 or novel > 0:
                    plt.figure(figsize=(10, 6))
                    categories = ['Found Confirmed', 'Found Candidate', 'Novel']
                    counts = [found_confirmed, found_candidate, novel]

                    plt.bar(categories, counts)
                    plt.title(f'Comparison with Known Genes (BF > {min_bf})')
                    plt.ylabel('Number of Genes')
                    plt.tight_layout()
                    plt.savefig(output_dir / 'gene_list_comparison.png')
                    plt.close()

                    # Also create a pie chart
                    plt.figure(figsize=(10, 6))
                    plt.pie(counts, labels=categories, autopct='%1.1f%%')
                    plt.title(f'Significant Genes Breakdown (BF > {min_bf})')
                    plt.savefig(output_dir / 'gene_list_pie.png')
                    plt.close()
