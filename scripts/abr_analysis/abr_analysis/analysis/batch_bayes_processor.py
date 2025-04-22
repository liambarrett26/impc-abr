# abr_analysis/analysis/batch_bayes_processor.py

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
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

    def analyze_gene(self, gene_data, sex_filter=None):
        """Analyze a single gene using Bayesian methods."""
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
                'bayes_factor': bayes_factor,
                'p_hearing_loss': p_hearing_loss,
                'hdi_lower': hdi_lower,
                'hdi_upper': hdi_upper,
                'effect_sizes': effect_sizes,
                'n_mutants': len(mutant_profiles_clean),
                'n_controls': len(control_profiles_clean),
                'mutant_means': np.mean(mutant_profiles_clean, axis=0),
                'control_means': np.mean(control_profiles_clean, axis=0)
            }

        except Exception as e:
            print(f"Warning: Error analyzing gene{f' ({sex_filter})' if sex_filter else ''}: {str(e)}")
            return None

    def analyze_all_genes(self):
        """Analyze all genes in the dataset using Bayesian approach."""
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
                    # Include main statistics
                    result[f'{analysis_type}_bayes_factor'] = analysis['bayes_factor']
                    result[f'{analysis_type}_p_hearing_loss'] = analysis['p_hearing_loss']
                    result[f'{analysis_type}_hdi_lower'] = analysis['hdi_lower']
                    result[f'{analysis_type}_hdi_upper'] = analysis['hdi_upper']
                    result[f'{analysis_type}_n_mutants'] = analysis['n_mutants']
                    result[f'{analysis_type}_n_controls'] = analysis['n_controls']
                    
                    # Include effect sizes for each frequency
                    for i, freq in enumerate(self.freq_cols):
                        freq_name = freq.split()[0]
                        result[f'{analysis_type}_effect_{freq_name}'] = analysis['effect_sizes'][i]
                else:
                    # Set missing values for metrics
                    metrics = ['bayes_factor', 'p_hearing_loss', 'hdi_lower', 'hdi_upper', 
                              'n_mutants', 'n_controls']
                    for metric in metrics:
                        result[f'{analysis_type}_{metric}'] = np.nan
                    
                    # Set missing values for effect sizes
                    for freq in self.freq_cols:
                        freq_name = freq.split()[0]
                        result[f'{analysis_type}_effect_{freq_name}'] = np.nan

            results.append(result)

        self.results = pd.DataFrame(results)

        # Add classification based on Bayes factors
        for analysis_type in ['all', 'male', 'female']:
            bf_col = f'{analysis_type}_bayes_factor'
            evidence_col = f'{analysis_type}_evidence'
            
            self.results[evidence_col] = 'Insufficient data'
            mask = ~self.results[bf_col].isna()
            
            # Classify based on Bayes factor
            self.results.loc[mask & (self.results[bf_col] > 100), evidence_col] = 'Extreme'
            self.results.loc[mask & (self.results[bf_col] <= 100) & (self.results[bf_col] > 30), evidence_col] = 'Very Strong'
            self.results.loc[mask & (self.results[bf_col] <= 30) & (self.results[bf_col] > 10), evidence_col] = 'Strong'
            self.results.loc[mask & (self.results[bf_col] <= 10) & (self.results[bf_col] > 3), evidence_col] = 'Substantial'
            self.results.loc[mask & (self.results[bf_col] <= 3), evidence_col] = 'Weak/None'

        return self.results

    def create_gene_visualization(self, gene, output_dir, analysis_type='all'):
        """Create visualizations for a specific gene."""
        gene_dir = output_dir / 'visuals' / gene
        gene_dir.mkdir(parents=True, exist_ok=True)
        
        # Get gene data
        analysis_key = f"{gene}_{analysis_type}"
        
        # Skip if we don't have the required data or model
        if (analysis_key not in self.mutant_profiles or 
            analysis_key not in self.control_profiles or
            analysis_key not in self.bayesian_models):
            return
                
        mutant_data = self.mutant_profiles[analysis_key]
        control_data = self.control_profiles[analysis_key]
        bayesian_model = self.bayesian_models[analysis_key]
        
        # 1. Profile Comparison
        try:
            plt.figure(figsize=(10, 6))
            control_profiles = self.matcher.get_control_profiles(control_data, self.freq_cols)
            mutant_profiles = mutant_data[self.freq_cols].values.astype(float)
            
            # Remove NaN values
            control_profiles = control_profiles[~np.any(np.isnan(control_profiles), axis=1)]
            mutant_profiles = mutant_profiles[~np.any(np.isnan(mutant_profiles), axis=1)]
            
            ax = plt.gca()
            bayesian_model._plot_profiles(ax, control_profiles, mutant_profiles, gene)
            plt.tight_layout()
            plt.savefig(gene_dir / f'{gene}_profiles_{analysis_type}.png')
            plt.close()
        except Exception as e:
            print(f"Error creating profile visualization for {gene}: {e}")
            
        # 2. Posterior Distribution of Hearing Loss
        try:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
            bayesian_model._plot_posterior_distributions(ax)
            plt.tight_layout()
            plt.savefig(gene_dir / f'{gene}_posterior_{analysis_type}.png')
            plt.close()
        except Exception as e:
            print(f"Error creating posterior visualization for {gene}: {e}")
            
        # 3. Hearing Loss Effect Size
        try:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
            bayesian_model._plot_effect_size(ax)
            plt.tight_layout()
            plt.savefig(gene_dir / f'{gene}_effect_size_{analysis_type}.png')
            plt.close()
        except Exception as e:
            print(f"Error creating effect size visualization for {gene}: {e}")

    def create_visualizations(self, output_dir='.', min_bf=3.0):
        """Create visualizations of the results."""
        output_dir = Path(output_dir)
        visuals_dir = output_dir / 'visuals'
        visuals_dir.mkdir(exist_ok=True, parents=True)

        # 1. Global Bayes Factor Distribution
        plt.figure(figsize=(10, 6))
        bfs = self.results['all_bayes_factor'].replace([np.inf, -np.inf], np.nan).dropna()
        
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
        evidence_counts = self.results['all_evidence'].value_counts()
        plt.pie(evidence_counts, labels=evidence_counts.index, autopct='%1.1f%%')
        plt.title('Distribution of Evidence Levels')
        plt.savefig(visuals_dir / 'evidence_classification.png')
        plt.close()

        # 3. Sex Comparison Plot
        plt.figure(figsize=(10, 6))
        male_sig = self.results['male_bayes_factor'] > min_bf
        female_sig = self.results['female_bayes_factor'] > min_bf
        
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
        sig_genes = self.results[self.results['all_bayes_factor'] > min_bf]
        
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
        center_counts = self.results[self.results['all_bayes_factor'] > min_bf].groupby('center').size()
        center_counts.plot(kind='bar')
        plt.title(f'Genes with BF > {min_bf} by Center')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(visuals_dir / 'center_comparison.png')
        plt.close()
        
        # 6. Create gene-specific visualizations
        print("\nGenerating gene-specific visualizations...")
        significant_genes = self.results[self.results['all_bayes_factor'] > min_bf]['gene_symbol'].tolist()
        
        # Create visualizations for all significant genes
        for gene in tqdm(significant_genes, desc="Creating gene visualizations", unit="gene"):
            self.create_gene_visualization(gene, output_dir, 'all')
            
        # Also create visualizations for sex-specific significant genes
        male_sig_genes = self.results[self.results['male_bayes_factor'] > min_bf]['gene_symbol'].tolist()
        female_sig_genes = self.results[self.results['female_bayes_factor'] > min_bf]['gene_symbol'].tolist()
        
        for gene in tqdm(male_sig_genes, desc="Creating male-specific visualizations", unit="gene"):
            if gene not in significant_genes:  # Avoid duplicates
                self.create_gene_visualization(gene, output_dir, 'male')
                
        for gene in tqdm(female_sig_genes, desc="Creating female-specific visualizations", unit="gene"):
            if gene not in significant_genes and gene not in male_sig_genes:  # Avoid duplicates
                self.create_gene_visualization(gene, output_dir, 'female')

    def compare_with_bowl(self, bowl_genes, min_bf=3.0):
        """Compare results with Bowl et al. genes."""
        significant_genes = {
            'all': set(self.results[self.results['all_bayes_factor'] > min_bf]['gene_symbol']),
            'male': set(self.results[self.results['male_bayes_factor'] > min_bf]['gene_symbol']),
            'female': set(self.results[self.results['female_bayes_factor'] > min_bf]['gene_symbol'])
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