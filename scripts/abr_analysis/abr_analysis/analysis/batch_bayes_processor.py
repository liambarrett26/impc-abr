#!/bin/env python
# -*- coding: utf-8 -*-
# abr_analysis/analysis/batch_bayes_processor.py

"""
Batch processing module for Bayesian ABR data analysis.

This module implements a pipeline for processing multiple gene datasets
through Bayesian analysis models, calculating evidence for hearing loss,
and generating visualisations across the entire gene set. It handles all
aspects of data loading, control matching, model fitting, and result aggregation.

Author: Liam Barrett
Version: 1.0.1
"""

from pathlib import Path
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ..data.loader import ABRDataLoader
from ..data.matcher import ControlMatcher
from ..models.bayesian import BayesianABRAnalysis

class GeneBayesianAnalyzer:
    """Analyze all genes in dataset using Bayesian approach."""

    def __init__(self, data_path):
        self.loader = ABRDataLoader(data_path)
        self.data = self.loader.load_data()
        self.matcher = ControlMatcher(self.data)
        self.freq_cols = self.loader.get_frequencies()
        self.results = None
        self.mutant_profiles = {}
        self.control_profiles = {}
        self.gene_metadata = {}
        self.bayesian_models = {}
        self.group_results = {}  # Store results for each experimental group
        self.gene_summary = None

    def analyze_experimental_group(self, group_info, sex_filter=None):
        """Analyze a single experimental group using Bayesian methods."""
        try:
            # Get the group data
            group_data = group_info['data']

            # Apply sex filter if specified
            if sex_filter:
                group_data = group_data[group_data['sex'] == sex_filter]

            # Check if we have any data after filtering
            if len(group_data) < 3:  # Minimum required for analysis
                return None

            # Create a unique key for this analysis
            gene_symbol = group_info['gene_symbol']
            allele_symbol = group_info['allele_symbol']
            zygosity = group_info['zygosity']
            center = group_info['phenotyping_center']

            analysis_key = f"{gene_symbol}_{allele_symbol}_{zygosity}_{center}"
            if sex_filter:
                analysis_key += f"_{sex_filter}"

            # Store metadata for later use in visualizations
            self.gene_metadata[analysis_key] = group_info['metadata']

            # Find matching controls
            try:
                controls = self.matcher.find_matching_controls(group_info)

                # Select the appropriate control set based on sex filter
                if sex_filter:
                    control_data = controls[sex_filter]
                else:
                    control_data = controls['all']

                if len(control_data) < 20:  # Minimum control requirement
                    return None

            except ValueError:  # Not enough controls
                return None

            # Extract profiles
            control_profiles = self.matcher.get_control_profiles(control_data, self.freq_cols)
            mutant_profiles = self.matcher.get_experimental_profiles(group_info,
                                                                     self.freq_cols,
                                                                     sex_filter)

            # Store profiles for later visualization
            self.control_profiles[analysis_key] = control_data
            self.mutant_profiles[analysis_key] = group_data

            # Remove any profiles with NaN values
            control_profiles_clean = control_profiles[~np.any(np.isnan(control_profiles), axis=1)]
            mutant_profiles_clean = mutant_profiles[~np.any(np.isnan(mutant_profiles), axis=1)]

            # Check minimum sample sizes
            if len(mutant_profiles_clean) < 3 or len(control_profiles_clean) < 20:
                return None

            # Fit Bayesian model
            bayesian_model = BayesianABRAnalysis()
            bayesian_model.fit(control_profiles_clean, mutant_profiles_clean, self.freq_cols)

            # Store model for later visualization
            self.bayesian_models[analysis_key] = bayesian_model

            # Get summary statistics
            summary = bayesian_model.get_summary_statistics()
            bayes_factor = bayesian_model.calculate_bayes_factor()

            # Extract key statistics
            p_hearing_loss = summary.loc['p_hearing_loss', 'mean']
            hdi_lower = summary.loc['p_hearing_loss', 'hdi_3%']
            hdi_upper = summary.loc['p_hearing_loss', 'hdi_97%']

            # Get effect sizes
            effect_sizes = []
            for i in range(len(self.freq_cols)):
                effect_sizes.append(summary.loc[f'hl_shift[{i}]', 'mean'])

            return {
                'gene_symbol': gene_symbol,
                'allele_symbol': allele_symbol,
                'zygosity': zygosity,
                'center': center,
                'bayes_factor': bayes_factor,
                'p_hearing_loss': p_hearing_loss,
                'hdi_lower': hdi_lower,
                'hdi_upper': hdi_upper,
                'effect_sizes': effect_sizes,
                'n_mutants': len(mutant_profiles_clean),
                'n_controls': len(control_profiles_clean),
                'mutant_means': np.mean(mutant_profiles_clean, axis=0),
                'control_means': np.mean(control_profiles_clean, axis=0),
                'analysis_key': analysis_key,
                'sex_filter': sex_filter if sex_filter else 'all'
            }

        except (ValueError, IOError, RuntimeError) as e:
            group_desc = f"{group_info['gene_symbol']} ({group_info['allele_symbol']}, {group_info['zygosity']}, {group_info['phenotyping_center']})"
            print(f"Warning: Error analyzing group {group_desc}{f' ({sex_filter})' if sex_filter else ''}: {str(e)}")
            traceback.print_exc()
            return None

    def analyze_all_genes(self):
        """Analyze all genes in the dataset using Bayesian approach."""
        # Get all genes with experimental data
        mutants = self.data[self.data['biological_sample_group'] == 'experimental']
        genes = mutants['gene_symbol'].unique()
        genes = genes[~pd.isna(genes)]  # Remove NaN values

        all_results = []
        group_counter = 0
        self.group_results = {}  # Reset group results

        # Create progress bar
        pbar = tqdm(total=len(genes), desc="Analyzing genes", unit="gene")

        for gene in genes:
            pbar.set_postfix_str(f"Current gene: {gene}")

            # Find all experimental groups for this gene
            exp_groups = self.matcher.find_experimental_groups(gene)

            if not exp_groups:
                pbar.update(1)
                continue

            group_counter += len(exp_groups)

            # Analyze each experimental group
            for group in exp_groups:
                # Analyze for all data, males only, and females only
                analyses = {
                    'all': self.analyze_experimental_group(group),
                    'male': self.analyze_experimental_group(group, 'male'),
                    'female': self.analyze_experimental_group(group, 'female')
                }

                # Store detailed group results
                for analysis_type, analysis in analyses.items():
                    if analysis:
                        group_key = f"{analysis['analysis_key']}"
                        self.group_results[group_key] = analysis

                # Create a gene-level result entry
                result = {
                    'gene_symbol': gene,
                    'allele_symbol': group['allele_symbol'],
                    'zygosity': group['zygosity'],
                    'center': group['phenotyping_center']
                }

                # Record results for each analysis type
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
                        for i, freq in enumerate(self.freq_cols):
                            freq_name = freq.split()[0]
                            result[f'{analysis_type}_effect_{freq_name}'] = analysis['effect_sizes'][i]
                    else:
                        # Set missing values for metrics
                        metrics = ['bayes_factor', 'p_hearing_loss', 'hdi_lower', 'hdi_upper',
                                'n_mutants', 'n_controls', 'analysis_key']
                        for metric in metrics:
                            result[f'{analysis_type}_{metric}'] = np.nan

                        # Set missing values for effect sizes
                        for freq in self.freq_cols:
                            freq_name = freq.split()[0]
                            result[f'{analysis_type}_effect_{freq_name}'] = np.nan

                all_results.append(result)

            pbar.update(1)

        pbar.close()
        print(f"Analyzed {group_counter} experimental groups across {len(genes)} genes")

        self.results = pd.DataFrame(all_results)

        # Create gene-level summary (best evidence across groups)
        gene_summary = self._create_gene_summary()

        # Add classification based on Bayes factors
        for analysis_type in ['all', 'male', 'female']:
            bf_col = f'{analysis_type}_bayes_factor'
            evidence_col = f'{analysis_type}_evidence'

            gene_summary[evidence_col] = 'Insufficient data'
            mask = ~gene_summary[bf_col].isna()

            # Classify based on Bayes factor
            gene_summary.loc[mask & (gene_summary[bf_col] > 100), evidence_col] = 'Extreme'
            gene_summary.loc[mask & (gene_summary[bf_col] <= 100) & (gene_summary[bf_col] > 30), evidence_col] = 'Very Strong'
            gene_summary.loc[mask & (gene_summary[bf_col] <= 30) & (gene_summary[bf_col] > 10), evidence_col] = 'Strong'
            gene_summary.loc[mask & (gene_summary[bf_col] <= 10) & (gene_summary[bf_col] > 3), evidence_col] = 'Substantial'
            gene_summary.loc[mask & (gene_summary[bf_col] <= 3), evidence_col] = 'Weak/None'

        self.gene_summary = gene_summary
        return gene_summary

    def _create_gene_summary(self):
        """Create a gene-level summary from all experimental groups."""
        if self.results is None or len(self.results) == 0:
            return pd.DataFrame()

        # Get unique genes
        genes = self.results['gene_symbol'].unique()

        summary_data = []

        for gene in genes:
            gene_data = self.results[self.results['gene_symbol'] == gene]

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
                    for freq in self.freq_cols:
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
                    for freq in self.freq_cols:
                        freq_name = freq.split()[0]
                        summary[f'{analysis_type}_effect_{freq_name}'] = np.nan

            summary_data.append(summary)

        return pd.DataFrame(summary_data)

    def compare_with_known_genes(self, confirmed_genes_path, candidate_genes_path, min_bf=3.0):
        """
        Compare results with confirmed and candidate deafness genes.
        
        Parameters:
            confirmed_genes_path (str): Path to the confirmed deafness genes file
            candidate_genes_path (str): Path to the candidate deafness genes file
            min_bf (float): Minimum Bayes Factor threshold for significance
            
        Returns:
            dict: Comparison results for each analysis type
        """
        # Load confirmed and candidate gene lists
        with open(confirmed_genes_path, 'r', encoding='utf-8') as f:
            confirmed_genes = set(line.strip() for line in f if line.strip())

        with open(candidate_genes_path, 'r', encoding='utf-8') as f:
            candidate_genes = set(line.strip() for line in f if line.strip())

        # Create sets of significant genes for each analysis type
        significant_genes = {}
        for analysis_type in ['all', 'male', 'female']:
            # Get unique gene symbols with significant Bayes factors
            sig_genes = set(self.gene_summary[self.gene_summary[f'{analysis_type}_bayes_factor'] > min_bf]['gene_symbol'])
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

    def create_gene_visualization(self, gene, output_dir, analysis_type='all'):
        """Create visualizations for a specific gene."""
        gene_dir = output_dir / 'visuals' / gene
        gene_dir.mkdir(parents=True, exist_ok=True)

        # Get gene summary data
        if gene not in self.gene_summary['gene_symbol'].values:
            print(f"No data available for gene {gene}")
            return

        gene_row = self.gene_summary[self.gene_summary['gene_symbol'] == gene].iloc[0]

        # Get the analysis key for the best result
        analysis_key = gene_row.get(f'{analysis_type}_analysis_key')

        if pd.isna(analysis_key) or analysis_key not in self.bayesian_models:
            print(f"No valid analysis found for gene {gene} ({analysis_type})")
            return

        # Create full results visualization
        try:
            # Get control and mutant data
            control_data = self.control_profiles.get(analysis_key)
            mutant_data = self.mutant_profiles.get(analysis_key)
            bayesian_model = self.bayesian_models.get(analysis_key)

            if control_data is None or mutant_data is None or bayesian_model is None:
                print(f"Missing data for visualization of {gene} ({analysis_key})")
                return

            # Extract profiles
            control_profiles = self.matcher.get_control_profiles(control_data, self.freq_cols)
            group_info_for_extract = {'data': mutant_data}

            # For mutant profiles, check if we need to filter by sex
            if 'male' in analysis_key or 'female' in analysis_key:
                sex_filter = 'male' if 'male' in analysis_key else 'female'
                mutant_profiles = self.matcher.get_experimental_profiles(
                    group_info_for_extract, self.freq_cols, sex_filter
                )
            else:
                mutant_profiles = self.matcher.get_experimental_profiles(
                    group_info_for_extract, self.freq_cols
                )

            # Remove NaN values
            control_profiles = control_profiles[~np.any(np.isnan(control_profiles), axis=1)]
            mutant_profiles = mutant_profiles[~np.any(np.isnan(mutant_profiles), axis=1)]

            # Create full visualization
            fig = bayesian_model.plot_results(control_profiles, mutant_profiles, gene)

            # Add allele, zygosity, and center information to the title
            allele = gene_row.get(f'{analysis_type}_allele', '')
            zygosity = gene_row.get(f'{analysis_type}_zygosity', '')
            center = gene_row.get(f'{analysis_type}_center', '')

            fig.suptitle(f"{gene} - {allele} ({zygosity}) - {center}", fontsize=14)

            fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle

            # Sanitize filename
            safe_allele = str(allele).replace('<', '').replace('>', '')
            safe_filename = f"{gene}_{safe_allele}_{zygosity}_{center}_{analysis_type}_bayesian_analysis.png"

            fig.savefig(gene_dir / safe_filename)
            plt.close(fig)

        except (ValueError, IOError, KeyError, AttributeError, TypeError) as e:
            print(f"Error creating visualisation for {gene}: {str(e)}")
            traceback.print_exc()

    def create_visualizations(self, output_dir='.', min_bf=3.0):
        """Create visualizations of the results."""
        output_dir = Path(output_dir)
        visuals_dir = output_dir / 'visuals'
        visuals_dir.mkdir(exist_ok=True, parents=True)

        # 1. Global Bayes Factor Distribution
        plt.figure(figsize=(10, 6))
        bfs = self.gene_summary['all_bayes_factor'].replace([np.inf, -np.inf], np.nan).dropna()

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
        plt.savefig(visuals_dir / 'bayes_factor_distribution.png')
        plt.close()

        # 2. Evidence Classification Pie Chart
        plt.figure(figsize=(10, 6))
        evidence_counts = self.gene_summary['all_evidence'].value_counts()
        plt.pie(evidence_counts, labels=evidence_counts.index, autopct='%1.1f%%')
        plt.title('Distribution of Evidence Levels')
        plt.savefig(visuals_dir / 'evidence_classification.png')
        plt.close()

        # 3. Sex Comparison Plot
        plt.figure(figsize=(10, 6))
        male_sig = self.gene_summary['male_bayes_factor'] > min_bf
        female_sig = self.gene_summary['female_bayes_factor'] > min_bf

        venn_data = {
            'Male Only': sum(male_sig & ~female_sig),
            'Female Only': sum(female_sig & ~male_sig),
            'Both': sum(male_sig & female_sig)
        }

        plt.pie(venn_data.values(), labels=venn_data.keys(), autopct='%1.1f%%')
        plt.title(f'Sex-specific Genes (BF > {min_bf})')
        plt.savefig(visuals_dir / 'sex_comparison.png')
        plt.close()

        # 4. Effect Size Distribution
        plt.figure(figsize=(12, 6))
        sig_genes = self.gene_summary[self.gene_summary['all_bayes_factor'] > min_bf]

        # Calculate mean effect size across frequencies
        effect_cols = [col for col in sig_genes.columns if col.startswith('all_effect_')]
        sig_genes['mean_effect'] = sig_genes[effect_cols].mean(axis=1)

        sns.histplot(data=sig_genes, x='mean_effect', bins=20)
        plt.xlabel('Mean Effect Size (dB)')
        plt.title(f'Effect Size Distribution (BF > {min_bf})')
        plt.savefig(visuals_dir / 'effect_size_distribution.png')
        plt.close()

        # 5. Center Comparison
        plt.figure(figsize=(12, 6))
        # Group by center and count significant genes
        center_counts = {}
        for _, row in self.gene_summary[self.gene_summary['all_bayes_factor'] > min_bf].iterrows():
            center = row.get('all_center')
            if not pd.isna(center):
                center_counts[center] = center_counts.get(center, 0) + 1

        centers = list(center_counts.keys())
        counts = list(center_counts.values())

        plt.bar(centers, counts)
        plt.title(f'Genes with BF > {min_bf} by Center')
        plt.xticks(rotation=45)
        plt.ylabel('Number of Genes')
        plt.tight_layout()
        plt.savefig(visuals_dir / 'center_comparison.png')
        plt.close()

        # 6. Create gene-specific visualizations
        print("\nGenerating gene-specific visualizations...")
        significant_genes = self.gene_summary[self.gene_summary['all_bayes_factor'] > min_bf]['gene_symbol'].tolist()

        # Create visualizations for all significant genes
        for gene in tqdm(significant_genes, desc="Creating gene visualizations", unit="gene"):
            self.create_gene_visualization(gene, output_dir, 'all')

        # Also create visualizations for sex-specific significant genes
        male_sig_genes = self.gene_summary[self.gene_summary['male_bayes_factor'] > min_bf]['gene_symbol'].tolist()
        female_sig_genes = self.gene_summary[self.gene_summary['female_bayes_factor'] > min_bf]['gene_symbol'].tolist()

        for gene in tqdm(male_sig_genes, desc="Creating male-specific visualizations", unit="gene"):
            if gene not in significant_genes:  # Avoid duplicates
                self.create_gene_visualization(gene, output_dir, 'male')

        for gene in tqdm(female_sig_genes, desc="Creating female-specific visualizations", unit="gene"):
            if gene not in significant_genes and gene not in male_sig_genes:  # Avoid duplicates
                self.create_gene_visualization(gene, output_dir, 'female')
