# abr_analysis/analysis/parallel_executor.py

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import os

from ..data.loader import ABRDataLoader
from ..models.bayesian import BayesianABRAnalysis
from .batch_bayes_processor import GeneBayesianAnalyzer

def process_single_gene(gene, data_path, save_dir=None):
    """Process a single gene and return the result."""
    # Create a new analyzer for this process
    analyzer = GeneBayesianAnalyzer(data_path)
    
    # Get gene data
    mutants = analyzer.data[analyzer.data['biological_sample_group'] == 'experimental']
    gene_data = mutants[mutants['gene_symbol'] == gene]
    
    if len(gene_data) == 0:
        return None
    
    # Create directories for model and visualizations if save_dir is provided
    if save_dir:
        save_dir = Path(save_dir)
        model_dir = save_dir / "models" / gene
        visuals_dir = save_dir / "visuals" / gene
        model_dir.mkdir(parents=True, exist_ok=True)
        visuals_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze gene
    try:
        # Get metadata for control matching
        ko_metadata = {
            'phenotyping_center': gene_data['phenotyping_center'].iloc[0],
            'genetic_background': gene_data['genetic_background'].iloc[0],
            'pipeline_name': gene_data['pipeline_name'].iloc[0],
            'metadata_Equipment manufacturer': gene_data['metadata_Equipment manufacturer'].iloc[0],
            'metadata_Equipment model': gene_data['metadata_Equipment model'].iloc[0]
        }
        
        # Get mutant and control data for visualization
        analyzer.gene_metadata[f"{gene}_all"] = ko_metadata
        analyzer.mutant_profiles[f"{gene}_all"] = gene_data
        
        # Analyze for all data, males only, and females only
        analyses = {
            'all': analyzer.analyze_gene(gene_data),
            'male': analyzer.analyze_gene(gene_data, 'male'),
            'female': analyzer.analyze_gene(gene_data, 'female')
        }
        
        # Create result dictionary
        result = {
            'gene_symbol': gene,
            'center': gene_data['phenotyping_center'].iloc[0] if len(gene_data) > 0 else None,
            'background': gene_data['genetic_background'].iloc[0] if len(gene_data) > 0 else None
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
                
                # Include effect sizes for each frequency
                freq_cols = analyzer.freq_cols
                for i, freq in enumerate(freq_cols):
                    freq_name = freq.split()[0]
                    result[f'{analysis_type}_effect_{freq_name}'] = analysis['effect_sizes'][i]
                
                # Save model and create visualizations if save_dir is provided
                if save_dir and analysis_type == 'all':  # Just save the 'all' model for simplicity
                    # Find the model in the analyzer's cache
                    model_key = f"{gene}_{analysis_type}"
                    if model_key in analyzer.bayesian_models:
                        # Save model
                        model_save_dir = model_dir
                        analyzer.bayesian_models[model_key].save_model(model_save_dir)
                        
                        # Create and save visualizations
                        try:
                            analyzer.create_gene_visualization(gene, save_dir, analysis_type)
                        except Exception as viz_err:
                            print(f"Error creating visualizations for {gene}: {str(viz_err)}")
            else:
                # Set missing values for metrics
                metrics = ['bayes_factor', 'p_hearing_loss', 'hdi_lower', 'hdi_upper', 
                          'n_mutants', 'n_controls']
                for metric in metrics:
                    result[f'{analysis_type}_{metric}'] = np.nan
                
                # Set missing values for effect sizes
                freq_cols = analyzer.freq_cols
                for freq in freq_cols:
                    freq_name = freq.split()[0]
                    result[f'{analysis_type}_effect_{freq_name}'] = np.nan
        
        return result
    
    except Exception as e:
        print(f"Error processing gene {gene}: {str(e)}")
        return None

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
    
    print(f"\nStarting parallel batch Bayesian analysis...")
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
        
        futures = []
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            # Submit all genes in this batch
            for gene in gene_batch:
                futures.append(executor.submit(process_single_gene, gene, data_path, output_dir))
            
            # Process results as they complete
            for future in tqdm(futures, total=len(gene_batch), desc=f"Batch {batch_index+1}"):
                result = future.result()
                if result:
                    all_results.append(result)
        
        batch_time = time.time() - batch_start
        print(f"Batch {batch_index+1} completed in {batch_time:.2f} seconds")
        
        # Save intermediate results
        if all_results:
            interim_df = pd.DataFrame(all_results)
            interim_df.to_csv(output_dir / f"interim_results_batch_{batch_index+1}.csv", index=False)
            print(f"Saved interim results ({len(interim_df)} genes processed so far)")
    
    # Convert all results to DataFrame
    if not all_results:
        print("No results were generated. Check for errors in the analysis.")
        return None, output_dir
        
    results_df = pd.DataFrame(all_results)
    
    # Add classification based on Bayes factors
    for analysis_type in ['all', 'male', 'female']:
        bf_col = f'{analysis_type}_bayes_factor'
        if bf_col in results_df.columns:
            evidence_col = f'{analysis_type}_evidence'
            
            results_df[evidence_col] = 'Insufficient data'
            mask = ~results_df[bf_col].isna()
            
            # Classify based on Bayes factor
            results_df.loc[mask & (results_df[bf_col] > 100), evidence_col] = 'Extreme'
            results_df.loc[mask & (results_df[bf_col] <= 100) & (results_df[bf_col] > 30), evidence_col] = 'Very Strong'
            results_df.loc[mask & (results_df[bf_col] <= 30) & (results_df[bf_col] > 10), evidence_col] = 'Strong'
            results_df.loc[mask & (results_df[bf_col] <= 10) & (results_df[bf_col] > 3), evidence_col] = 'Substantial'
            results_df.loc[mask & (results_df[bf_col] <= 3), evidence_col] = 'Weak/None'
    
    # Save full results
    results_df.to_csv(output_dir / "detailed_results.csv", index=False)
    
    # Generate summary report
    total_time = time.time() - start_time
    
    with open(output_dir / "summary.txt", 'w') as f:
        f.write("Parallel Bayesian Analysis Summary\n")
        f.write("================================\n\n")
        f.write(f"Analysis completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {total_time:.2f} seconds\n")
        f.write(f"Processes used: {n_processes}\n")
        f.write(f"Total genes analyzed: {len(genes)}\n")
        f.write(f"Successful analyses: {len(results_df)}\n\n")
        
        # Evidence level breakdown
        f.write("Evidence Breakdown:\n")
        if 'all_evidence' in results_df.columns:
            evidence_counts = results_df['all_evidence'].value_counts()
            for evidence, count in evidence_counts.items():
                f.write(f"  {evidence}: {count} genes ({count/len(results_df)*100:.1f}%)\n")
        
        # Add summary of significant genes
        significant_genes = results_df[results_df['all_bayes_factor'] > min_bf]['gene_symbol'].tolist()
        f.write(f"\nSignificant Genes (BF > {min_bf}): {len(significant_genes)}\n")
        if significant_genes:
            # Format genes in columns
            genes_per_line = 5
            for i in range(0, len(significant_genes), genes_per_line):
                gene_group = significant_genes[i:i+genes_per_line]
                f.write(f"  {', '.join(gene_group)}\n")
    
    # Create summary visualizations
    create_visualizations(results_df, output_dir, min_bf=min_bf)
    
    print(f"\nAnalysis completed in {total_time:.2f} seconds")
    print(f"Found {len(results_df)} genes with results out of {len(genes)} genes")
    print(f"Results saved to {output_dir}")
    print(f"Models saved to {models_dir}")
    print(f"Visualizations saved to {visuals_dir}")
    
    return results_df, output_dir

def compare_gene_lists(results_df, reference_genes, min_bf=3.0):
    """
    Compare results with a reference gene list (e.g., Bowl et al.).
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results from the analysis
    reference_genes : set or list
        Reference gene symbols
    min_bf : float
        Minimum Bayes factor for significance
        
    Returns:
    --------
    dict
        Comparison results
    """
    reference_genes = set(reference_genes)
    
    comparisons = {}
    for analysis_type in ['all', 'male', 'female']:
        bf_col = f'{analysis_type}_bayes_factor'
        
        if bf_col in results_df.columns:
            # Get significant genes
            significant = set(results_df[results_df[bf_col] > min_bf]['gene_symbol'])
            
            comparisons[analysis_type] = {
                'found_in_reference': significant & reference_genes,
                'novel': significant - reference_genes,
                'missed_from_reference': reference_genes - significant
            }
    
    return comparisons

def create_visualizations(results_df, output_dir, reference_genes=None, min_bf=3.0):
    """
    Create summary visualizations from analysis results.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results from the analysis
    output_dir : str or Path
        Directory to save visualizations
    reference_genes : set or list, optional
        Reference gene symbols for comparison
    min_bf : float
        Minimum Bayes factor for significance
    """
    output_dir = Path(output_dir) / "visualizations"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Bayes Factor Distribution
    plt.figure(figsize=(10, 6))
    bfs = results_df['all_bayes_factor'].replace([np.inf, -np.inf], np.nan).dropna()
    
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
    plt.figure(figsize=(10, 6))
    evidence_counts = results_df['all_evidence'].value_counts()
    plt.pie(evidence_counts, labels=evidence_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of Evidence Levels')
    plt.savefig(output_dir / 'evidence_classification.png')
    plt.close()

    # 3. Sex Comparison Plot
    plt.figure(figsize=(10, 6))
    male_sig = results_df['male_bayes_factor'] > min_bf
    female_sig = results_df['female_bayes_factor'] > min_bf
    
    venn_data = {
        'Male Only': sum(male_sig & ~female_sig),
        'Female Only': sum(female_sig & ~male_sig),
        'Both': sum(male_sig & female_sig)
    }

    plt.pie(venn_data.values(), labels=venn_data.keys(), autopct='%1.1f%%')
    plt.title(f'Sex-specific Genes (BF > {min_bf})')
    plt.savefig(output_dir / 'sex_comparison.png')
    plt.close()

    # 4. Effect Size Distribution
    plt.figure(figsize=(12, 6))
    sig_genes = results_df[results_df['all_bayes_factor'] > min_bf]
    
    # Calculate mean effect size across frequencies
    effect_cols = [col for col in sig_genes.columns if col.startswith('all_effect_')]
    if effect_cols:
        sig_genes['mean_effect'] = sig_genes[effect_cols].mean(axis=1)
        
        sns.histplot(data=sig_genes, x='mean_effect', bins=20)
        plt.xlabel('Mean Effect Size (dB)')
        plt.title(f'Effect Size Distribution (BF > {min_bf})')
        plt.savefig(output_dir / 'effect_size_distribution.png')
        plt.close()

    # 5. Center Comparison
    plt.figure(figsize=(12, 6))
    center_counts = results_df[results_df['all_bayes_factor'] > min_bf].groupby('center').size()
    if not center_counts.empty:
        center_counts.plot(kind='bar')
        plt.title(f'Genes with BF > {min_bf} by Center')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'center_comparison.png')
        plt.close()
        
    # 6. Reference Gene Comparison (if provided)
    if reference_genes:
        comparison = compare_gene_lists(results_df, reference_genes, min_bf)
        
        plt.figure(figsize=(12, 6))
        comparison_data = [
            len(comparison['all']['found_in_reference']),
            len(comparison['all']['novel']),
            len(comparison['all']['missed_from_reference'])
        ]
        
        plt.bar(['Found in Reference', 'Novel', 'Missed from Reference'], comparison_data)
        plt.ylabel('Number of Genes')
        plt.title('Comparison with Reference Gene List')
        plt.savefig(output_dir / 'reference_comparison.png')
        plt.close()