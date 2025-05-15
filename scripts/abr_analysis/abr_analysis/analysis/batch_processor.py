#!/bin/env python
# -*- coding: utf-8 -*-
# abr_analysis/analysis/batch_processor.py

"""
Batch processing module for multivariate Auditory Brainstem Response (ABR) analysis.

This module implements a pipeline for analysing large sets of ABR data through
multivariate statistical methods. It handles data loading, control matching,
statistical testing, and result visualization for multiple genes simultaneously.
The analysis treats ABR profiles as multivariate observations and applies
robust multivariate Gaussian models to identify significant hearing phenotypes.

The module supports:
- Batch processing of all genes in a dataset
- Sex-specific analyses
- Multiple test correction using FDR
- Comparison with known and candidate hearing loss genes
- Automated visualization generation for significant results

Author: Liam Barrett
Version: 1.0.1
"""

from pathlib import Path

import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ..data.loader import ABRDataLoader
from ..data.matcher import ControlMatcher
from ..models.distribution import RobustMultivariateGaussian


class GeneBatchAnalyzer:
    """Analyze all genes in dataset using multivariate distribution approach."""

    def __init__(self, data_path):
        self.loader = ABRDataLoader(data_path)
        self.data = self.loader.load_data()
        self.matcher = ControlMatcher(self.data)
        self.freq_cols = self.loader.get_frequencies()
        self.results = None
        self.mutant_profiles = {}
        self.control_profiles = {}
        self.gene_metadata = {}

    def analyze_gene_group(self, group_info, sex_filter=None):
        """Analyze a single experimental group using our multivariate distribution model."""
        try:
            # Get the experimental data
            gene_data = group_info['data']

            # Apply sex filter if specified
            if sex_filter:
                gene_data = gene_data[gene_data['sex'] == sex_filter]

            # Check if we have any data after filtering
            if len(gene_data) < 3:  # Minimum of 3 mutant animals
                return None

            # Find matching controls
            try:
                control_groups = self.matcher.find_matching_controls(group_info)

                # Get appropriate controls based on sex filter
                if sex_filter:
                    controls = control_groups[sex_filter]
                else:
                    controls = control_groups['all']

                if len(controls) < 20:  # Minimum control requirement
                    return None
            except ValueError:  # Not enough controls
                return None

            # Store metadata for later use in visualizations
            gene_symbol = gene_data['gene_symbol'].iloc[0]
            allele_symbol = group_info['allele_symbol']
            zygosity = group_info['zygosity']
            center = group_info['phenotyping_center']

            group_key = f"{gene_symbol}_{allele_symbol}_{zygosity}_{center}_{sex_filter if sex_filter else 'all'}"
            self.gene_metadata[group_key] = group_info['metadata']

            # Extract profiles
            control_profiles = self.matcher.get_control_profiles(controls, self.freq_cols)
            mutant_profiles = gene_data[self.freq_cols].values.astype(float)

            # Store profiles for later visualization
            self.control_profiles[group_key] = controls
            self.mutant_profiles[group_key] = gene_data

            # Remove any profiles with NaN values
            control_profiles_clean = control_profiles[~np.any(np.isnan(control_profiles), axis=1)]
            mutant_profiles_clean = mutant_profiles[~np.any(np.isnan(mutant_profiles), axis=1)]

            # Check minimum sample sizes again after removing NaN values
            if len(mutant_profiles_clean) < 3 or len(control_profiles_clean) < 20:
                return None

            # Fit distribution model to controls
            model = RobustMultivariateGaussian()
            model.fit(control_profiles_clean)

            # Calculate log probabilities for mutants
            mutant_log_probs = []
            mutant_distances = []
            for profile in mutant_profiles_clean:
                log_prob = model.score(profile)
                distance = model.mahalanobis(profile)
                mutant_log_probs.append(log_prob)
                mutant_distances.append(distance)

            # Calculate control log probabilities for comparison
            control_log_probs = []
            control_distances = []
            for profile in control_profiles_clean:
                log_prob = model.score(profile)
                distance = model.mahalanobis(profile)
                control_log_probs.append(log_prob)
                control_distances.append(distance)

            # Calculate statistics
            stat, pval = stats.ttest_ind(mutant_log_probs, control_log_probs)

            return {
                'p_value': pval,
                'test_statistic': stat,
                'n_mutants': len(mutant_profiles_clean),
                'n_controls': len(control_profiles_clean),
                'mean_mutant_logprob': np.mean(mutant_log_probs),
                'std_mutant_logprob': np.std(mutant_log_probs),
                'mean_mutant_distance': np.mean(mutant_distances),
                'mean_control_logprob': np.mean(control_log_probs),
                'std_control_logprob': np.std(control_log_probs),
                'mean_control_distance': np.mean(control_distances),
                'mutant_log_probs': mutant_log_probs,
                'control_log_probs': control_log_probs,
                'mutant_distances': mutant_distances,
                'control_distances': control_distances,
                'allele_symbol': allele_symbol,
                'zygosity': zygosity,
                'center': center,
                'group_key': group_key
            }

        except (ValueError, KeyError, AttributeError, TypeError, IndexError, RuntimeError) as e:
            print(f"Warning: Error analysing gene group {gene_symbol} ({allele_symbol}, {zygosity}, {center}){f', {sex_filter}' if sex_filter else ''}: {str(e)}")
            return None

    def analyze_all_genes(self):
        """Analyze all genes in the dataset, grouping by allele+zygosity+center."""
        mutants = self.data[self.data['biological_sample_group'] == 'experimental']
        genes = mutants['gene_symbol'].unique()
        genes = genes[~pd.isna(genes)]  # Remove NaN values

        results = []

        # Create progress bar
        pbar = tqdm(genes, desc="Analyzing genes", unit="gene")

        for gene in pbar:
            pbar.set_postfix_str(f"Current gene: {gene}")

            # Find all experimental groups for this gene
            experimental_groups = self.matcher.find_experimental_groups(gene)

            if not experimental_groups:
                continue  # Skip if no valid groups

            # Analyze each experimental group
            for group_info in experimental_groups:
                # Analyze for all data, males only, and females only
                analyses = {
                    'all': self.analyze_gene_group(group_info),
                    'male': self.analyze_gene_group(group_info, 'male'),
                    'female': self.analyze_gene_group(group_info, 'female')
                }

                # Only create result if at least one analysis succeeded
                if not any(analyses.values()):
                    continue

                result = {
                    'gene_symbol': gene,
                    'allele_symbol': group_info['allele_symbol'],
                    'zygosity': group_info['zygosity'],
                    'center': group_info['phenotyping_center'],
                    'background': group_info['metadata']['genetic_background'],
                    'group_id': f"{gene}_{group_info['allele_symbol']}_{group_info['zygosity']}_{group_info['phenotyping_center']}"
                }

                # Record results
                for analysis_type, analysis in analyses.items():
                    if analysis:
                        for key, value in analysis.items():
                            # Skip storing large arrays and redundant info in the main results dataframe
                            if key not in ['mutant_log_probs', 'control_log_probs',
                                        'mutant_distances', 'control_distances',
                                        'allele_symbol', 'zygosity', 'center', 'group_key']:
                                result[f'{analysis_type}_{key}'] = value
                    else:
                        metrics = ['p_value', 'test_statistic', 'n_mutants', 'n_controls',
                                'mean_mutant_logprob', 'std_mutant_logprob',
                                'mean_mutant_distance', 'mean_control_logprob',
                                'std_control_logprob', 'mean_control_distance']
                        for metric in metrics:
                            result[f'{analysis_type}_{metric}'] = np.nan

                results.append(result)

        self.results = pd.DataFrame(results)

        # Apply FDR correction
        for analysis_type in ['all', 'male', 'female']:
            p_vals = self.results[f'{analysis_type}_p_value'].values
            mask = ~np.isnan(p_vals)
            if np.any(mask):
                _, q_vals, _, _ = multipletests(p_vals[mask], method='fdr_bh')
                self.results.loc[mask, f'{analysis_type}_q_value'] = q_vals

        return self.results

    def create_gene_visualization(self, gene, output_dir, analysis_type='all', q_threshold=0.01):
        """Create visualizations for a specific gene."""
        # Get all rows for this gene
        gene_rows = self.results[self.results['gene_symbol'] == gene]

        if gene_rows.empty:
            return

        # Create gene directory
        gene_dir = output_dir / 'visuals' / gene
        gene_dir.mkdir(parents=True, exist_ok=True)

        # Create a visualization for each group
        for _, gene_row in gene_rows.iterrows():
            allele = gene_row['allele_symbol']
            zygosity = gene_row['zygosity']
            center = gene_row['center']

            # Skip if no significant result for the requested analysis type
            if pd.isna(gene_row[f'{analysis_type}_q_value']) or gene_row[f'{analysis_type}_q_value'] >= q_threshold:
                continue

            # Construct the group key
            group_key = f"{gene}_{allele}_{zygosity}_{center}_{analysis_type}"

            # Skip if we don't have the required data
            if group_key not in self.mutant_profiles and f"{gene}_{allele}_{zygosity}_{center}_all" not in self.mutant_profiles:
                continue

            # Use the specific analysis type data if available, otherwise fall back to 'all'
            if group_key in self.mutant_profiles:
                mutant_data = self.mutant_profiles[group_key]
                control_data = self.control_profiles[group_key]
            else:
                fallback_key = f"{gene}_{allele}_{zygosity}_{center}_all"
                mutant_data = self.mutant_profiles[fallback_key]
                control_data = self.control_profiles[fallback_key]

                # Filter by sex if needed
                if analysis_type in ['male', 'female']:
                    mutant_data = mutant_data[mutant_data['sex'] == analysis_type]
                    control_data = control_data[control_data['sex'] == analysis_type]

                    # Skip if not enough data after filtering
                    if len(mutant_data) < 3 or len(control_data) < 20:
                        continue

            try:
                plt.figure(figsize=(12, 6))

                # Plot 1: Profile Comparison
                plt.subplot(1, 2, 1)
                x = np.arange(len(self.freq_cols))
                plt.errorbar(x, control_data[self.freq_cols].mean(),
                            yerr=control_data[self.freq_cols].std(),
                            label='Controls', fmt='o-')
                plt.errorbar(x, mutant_data[self.freq_cols].mean(),
                            yerr=mutant_data[self.freq_cols].std(),
                            label=gene, fmt='o-')
                plt.xticks(x, [col.split()[0] for col in self.freq_cols], rotation=45)
                plt.ylabel('ABR Threshold (dB SPL)')
                plt.title(f'{gene} ({allele}, {zygosity}, {center})')
                plt.legend()

                # Get clean profiles for log probability calculation
                mutant_profiles = mutant_data[self.freq_cols].values.astype(float)
                control_profiles = self.matcher.get_control_profiles(control_data, self.freq_cols)

                # Remove NaN values
                mutant_profiles = mutant_profiles[~np.any(np.isnan(mutant_profiles), axis=1)]
                control_profiles = control_profiles[~np.any(np.isnan(control_profiles), axis=1)]

                if len(mutant_profiles) >= 3 and len(control_profiles) >= 20:
                    # Fit model and calculate log probs directly
                    model = RobustMultivariateGaussian()
                    model.fit(control_profiles)

                    # Calculate log probabilities
                    mutant_log_probs = [model.score(profile) for profile in mutant_profiles]
                    control_log_probs = [model.score(profile) for profile in control_profiles]

                    # Plot 2: Log Probabilities
                    plt.subplot(1, 2, 2)
                    plt.hist(control_log_probs, bins=20, alpha=0.5,
                            label='Controls', density=True)
                    plt.hist(mutant_log_probs, bins=20, alpha=0.5,
                            label=gene, density=True)
                    plt.xlabel('Log Probability')
                    plt.ylabel('Density')
                    plt.title('Distribution of Log Probabilities')
                    plt.legend()

                plt.tight_layout()
                plt.savefig(gene_dir / f'{gene}_{allele}_{zygosity}_{center}_{analysis_type}.png')
                plt.close()
            except (ValueError, IOError, KeyError, AttributeError, TypeError, IndexError, RuntimeError) as e:
                print(f"Error creating visualisation for {gene} ({allele}, {zygosity}, {center}, {analysis_type}): {e}")

    def create_visualizations(self, output_dir='.', q_threshold=0.01):
        """Create visualizations of the results."""
        output_dir = Path(output_dir)
        visuals_dir = output_dir / 'visuals'
        visuals_dir.mkdir(exist_ok=True, parents=True)

        # 1. Global Multivariate Evidence Plot
        plt.figure(figsize=(10, 6))

        # Use test_statistic
        x_vals = -np.log10(self.results['all_p_value'].replace([np.inf, -np.inf], np.nan))
        y_vals = np.abs(self.results['all_test_statistic'])

        # Remove infinite values
        mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
        plt.scatter(x_vals[mask], y_vals[mask], alpha=0.5, label='Not Significant')

        # Plot significant points
        significant = self.results['all_q_value'] < q_threshold
        sig_mask = significant & mask
        if np.any(sig_mask):
            plt.scatter(
                x_vals[sig_mask],
                y_vals[sig_mask],
                color='red',
                alpha=0.5,
                label='Significant'
            )

        plt.xlabel('-log10(p-value)')
        plt.ylabel('|Test Statistic|')
        plt.title('Multivariate Evidence for Hearing Loss')
        plt.legend()
        plt.savefig(visuals_dir / 'multivariate_evidence.png')
        plt.close()

        # 2. Sex Comparison Plot
        plt.figure(figsize=(10, 6))
        male_sig = self.results['male_q_value'] < q_threshold
        female_sig = self.results['female_q_value'] < q_threshold

        venn_data = {
            'Male Only': sum(male_sig & ~female_sig),
            'Female Only': sum(female_sig & ~male_sig),
            'Both': sum(male_sig & female_sig)
        }

        plt.pie(venn_data.values(), labels=venn_data.keys(), autopct='%1.1f%%')
        plt.title('Sex-specific Significant Genes')
        plt.savefig(visuals_dir / 'sex_comparison.png')
        plt.close()

        # 3. Effect Size Distribution
        plt.figure(figsize=(10, 6))
        sig_genes = self.results[self.results['all_q_value'] < q_threshold]

        sns.histplot(data=sig_genes, x='all_mean_mutant_distance', bins=20)
        plt.xlabel('Mean Mahalanobis Distance')
        plt.title('Effect Size Distribution (Significant Genes)')
        plt.savefig(visuals_dir / 'effect_size_distribution.png')
        plt.close()

        # 4. Center Comparison
        plt.figure(figsize=(12, 6))
        # Count unique gene symbols for each center to avoid double-counting groups
        center_counts = self.results[significant].groupby(['center', 'gene_symbol']).size().reset_index()
        center_counts = center_counts.groupby('center').size()
        center_counts.plot(kind='bar')
        plt.title('Significant Genes by Center')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(visuals_dir / 'center_comparison.png')
        plt.close()

        # 5. Allele Comparison
        plt.figure(figsize=(12, 6))
        # Count unique gene symbols for each allele to avoid double-counting groups
        allele_counts = self.results[significant].groupby(['allele_symbol', 'gene_symbol']).size().reset_index()
        allele_counts = allele_counts.groupby('allele_symbol').size().sort_values(ascending=False).head(20)
        allele_counts.plot(kind='bar')
        plt.title('Top 20 Significant Alleles')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(visuals_dir / 'allele_comparison.png')
        plt.close()

        # 6. Create gene-specific visualizations
        print("\nGenerating gene-specific visualizations...")
        # Get unique genes with significant results (using any analysis type)
        significant_genes = set()
        for analysis_type in ['all', 'male', 'female']:
            sig_genes = self.results[self.results[f'{analysis_type}_q_value'] < q_threshold]['gene_symbol'].unique()
            significant_genes.update(sig_genes)

        # Create visualizations for all significant genes
        for gene in tqdm(significant_genes, desc="Creating gene visualizations", unit="gene"):
            # Create for each analysis type
            for analysis_type in ['all', 'male', 'female']:
                self.create_gene_visualization(gene, output_dir, analysis_type, q_threshold)

    def compare_with_bowl(self, bowl_genes, q_threshold=0.01):
        """Compare results with Bowl et al. genes."""
        # Create sets of significant genes for each analysis type
        significant_genes = {}
        for analysis_type in ['all', 'male', 'female']:
            # Get unique gene symbols with significant q-values
            sig_genes = set(self.results[self.results[f'{analysis_type}_q_value'] < q_threshold]['gene_symbol'])
            significant_genes[analysis_type] = sig_genes

        bowl_genes = set(bowl_genes)

        comparisons = {}
        for analysis_type, sig_genes in significant_genes.items():
            comparisons[analysis_type] = {
                'found_in_bowl': bowl_genes & sig_genes,
                'novel': sig_genes - bowl_genes,
                'missed_from_bowl': bowl_genes - sig_genes
            }

        return comparisons

    def compare_with_known_genes(self,
                                 confirmed_genes_path,
                                 candidate_genes_path,
                                 q_threshold=0.01):
        """
        Compare results with confirmed and candidate deafness genes.
        
        Parameters:
            confirmed_genes_path (str): Path to the confirmed deafness genes file
            candidate_genes_path (str): Path to the candidate deafness genes file
            q_threshold (float): Significance threshold for q-values
            
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
            # Get unique gene symbols with significant q-values
            sig_genes = set(self.results[self.results[f'{analysis_type}_q_value'] < q_threshold]['gene_symbol'])
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
