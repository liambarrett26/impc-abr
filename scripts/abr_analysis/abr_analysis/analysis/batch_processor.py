# abr_analysis/analysis/batch_processor.py

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

    def analyze_gene(self, gene_data, sex_filter=None):
        """Analyze a single gene using our multivariate distribution model."""
        try:
            # Apply sex filter if specified
            if sex_filter:
                gene_data = gene_data[gene_data['sex'] == sex_filter]

            # Check if we have any data after filtering
            if len(gene_data) == 0:
                return None

            # Get metadata for control matching
            ko_metadata = {
                'phenotyping_center': gene_data['phenotyping_center'].iloc[0],
                'genetic_background': gene_data['genetic_background'].iloc[0],
                'pipeline_name': gene_data['pipeline_name'].iloc[0],
                'metadata_Equipment manufacturer': gene_data['metadata_Equipment manufacturer'].iloc[0],
                'metadata_Equipment model': gene_data['metadata_Equipment model'].iloc[0]
            }

            # Store metadata for later use in visualizations
            gene_symbol = gene_data['gene_symbol'].iloc[0]
            analysis_key = f"{gene_symbol}_{sex_filter if sex_filter else 'all'}"
            self.gene_metadata[analysis_key] = ko_metadata

            # Find matching controls
            try:
                controls = self.matcher.find_matching_controls(ko_metadata)
            except ValueError:  # Not enough controls
                return None
  
            if sex_filter:
                controls = controls[controls['sex'] == sex_filter]
                if len(controls) < 20:  # Minimum control requirement
                    return None

            # Extract profiles
            control_profiles = self.matcher.get_control_profiles(controls, self.freq_cols)
            mutant_profiles = gene_data[self.freq_cols].values.astype(float)

            # Store profiles for later visualization
            self.control_profiles[analysis_key] = controls
            self.mutant_profiles[analysis_key] = gene_data

            # Remove any profiles with NaN values
            control_profiles_clean = control_profiles[~np.any(np.isnan(control_profiles), axis=1)]
            mutant_profiles_clean = mutant_profiles[~np.any(np.isnan(mutant_profiles), axis=1)]

            # Check minimum sample sizes
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
                'control_distances': control_distances
            }

        except Exception as e:
            print(f"Warning: Error analyzing gene{f' ({sex_filter})' if sex_filter else ''}: {str(e)}")
            return None

    def analyze_all_genes(self):
        """Analyze all genes in the dataset."""
        mutants = self.data[self.data['biological_sample_group'] == 'experimental']
        genes = mutants['gene_symbol'].unique()
        genes = genes[~pd.isna(genes)]  # Remove NaN values

        results = []

        # Create progress bar
        pbar = tqdm(genes, desc="Analyzing genes", unit="gene")

        for gene in pbar:
            pbar.set_postfix_str(f"Current gene: {gene}")
            gene_data = mutants[mutants['gene_symbol'] == gene]

            # Analyze for all data, males only, and females only
            analyses = {
                'all': self.analyze_gene(gene_data),
                'male': self.analyze_gene(gene_data, 'male'),
                'female': self.analyze_gene(gene_data, 'female')
            }

            result = {
                'gene_symbol': gene,
                'center': gene_data['phenotyping_center'].iloc[0],
                'background': gene_data['genetic_background'].iloc[0]
            }

            # Record results
            for analysis_type, analysis in analyses.items():
                if analysis:
                    for key, value in analysis.items():
                        # Skip storing large arrays in the main results dataframe
                        if key not in ['mutant_log_probs', 'control_log_probs', 
                                     'mutant_distances', 'control_distances']:
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
        gene_dir = output_dir / 'visuals' / gene
        gene_dir.mkdir(parents=True, exist_ok=True)
        
        # Get gene data
        gene_row = self.results[self.results['gene_symbol'] == gene].iloc[0]
        analysis_key = f"{gene}_{analysis_type}"
        
        # Skip if we don't have the required data
        if analysis_key not in self.mutant_profiles or analysis_key not in self.control_profiles:
            return
                
        mutant_data = self.mutant_profiles[analysis_key]
        control_data = self.control_profiles[analysis_key]
        
        # 1. Profile Comparison
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
            plt.title(f'ABR Profiles - {gene}')
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
            plt.savefig(gene_dir / f'{gene}_profiles_{analysis_type}.png')
            plt.close()
        except Exception as e:
            print(f"Error creating profile visualization for {gene}: {e}")

        # 2. Create gene-specific multivariate evidence plot
        try:
            plt.figure(figsize=(10, 6))
            
            # Plot all genes
            p_values = self.results['all_p_value'].replace([np.inf, -np.inf], np.nan)
            # Handle zero values
            p_values = np.maximum(p_values, 1e-300)  # Set minimum to a very small number instead of zero
            x_vals = -np.log10(p_values)
            y_vals = np.abs(self.results['all_test_statistic'])
            
            # Remove infinite values
            mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
            plt.scatter(x_vals[mask], y_vals[mask], alpha=0.3, color='gray', label='Other Genes')
            
            # Plot significant points
            significant = self.results['all_q_value'] < q_threshold
            sig_mask = significant & mask
            if np.any(sig_mask):
                plt.scatter(
                    x_vals[sig_mask],
                    y_vals[sig_mask],
                    color='blue',
                    alpha=0.5,
                    label='Significant Genes'
                )
            
            # Highlight this gene
            gene_idx = self.results[self.results['gene_symbol'] == gene].index[0]
            if mask[gene_idx]:
                plt.scatter(
                    x_vals[gene_idx],
                    y_vals[gene_idx],
                    color='red',
                    s=100,
                    label=gene
                )
                plt.annotate(
                    gene,
                    (x_vals[gene_idx], y_vals[gene_idx]),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=12,
                    fontweight='bold'
                )
            
            plt.xlabel('-log10(p-value)')
            plt.ylabel('|Test Statistic|')
            plt.title(f'Multivariate Evidence for Hearing Loss - {gene}')
            plt.legend()
            plt.savefig(gene_dir / f'{gene}_evidence.png')
            plt.close()
        except Exception as e:
            print(f"Error creating evidence visualization for {gene}: {e}")

    def create_visualizations(self, output_dir='.', q_threshold=0.01):
        """Create visualizations of the results."""
        output_dir = Path(output_dir)
        visuals_dir = output_dir / 'visuals'
        visuals_dir.mkdir(exist_ok=True, parents=True)

        # 1. Global Multivariate Evidence Plot
        plt.figure(figsize=(10, 6))

        # Use test_statistic instead of all_statistic
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
        center_counts = self.results[significant].groupby('center').size()
        center_counts.plot(kind='bar')
        plt.title('Significant Genes by Center')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(visuals_dir / 'center_comparison.png')
        plt.close()
        
        # 5. Create gene-specific visualizations
        print("\nGenerating gene-specific visualizations...")
        significant_genes = self.results[self.results['all_q_value'] < q_threshold]['gene_symbol'].tolist()
        
        # Create visualizations for all significant genes
        for gene in tqdm(significant_genes, desc="Creating gene visualizations", unit="gene"):
            self.create_gene_visualization(gene, output_dir, 'all', q_threshold)
            
        # Also create visualizations for sex-specific significant genes
        male_sig_genes = self.results[self.results['male_q_value'] < q_threshold]['gene_symbol'].tolist()
        female_sig_genes = self.results[self.results['female_q_value'] < q_threshold]['gene_symbol'].tolist()
        
        for gene in tqdm(male_sig_genes, desc="Creating male-specific visualizations", unit="gene"):
            if gene not in significant_genes:  # Avoid duplicates
                self.create_gene_visualization(gene, output_dir, 'male', q_threshold)
                
        for gene in tqdm(female_sig_genes, desc="Creating female-specific visualizations", unit="gene"):
            if gene not in significant_genes and gene not in male_sig_genes:  # Avoid duplicates
                self.create_gene_visualization(gene, output_dir, 'female', q_threshold)

    def compare_with_bowl(self, bowl_genes, q_threshold=0.01):
        """Compare results with Bowl et al. genes."""
        significant_genes = {
            'all': set(self.results[self.results['all_q_value'] < q_threshold]['gene_symbol']),
            'male': set(self.results[self.results['male_q_value'] < q_threshold]['gene_symbol']),
            'female': set(self.results[self.results['female_q_value'] < q_threshold]['gene_symbol'])
        }

        bowl_genes = set(bowl_genes)

        comparisons = {}
        for analysis_type, sig_genes in significant_genes.items():
            comparisons[analysis_type] = {
                'found_in_bowl': bowl_genes & sig_genes,
                'novel': sig_genes - bowl_genes,
                'missed_from_bowl': bowl_genes - sig_genes
            }

        return comparisons