# tests/test_parallel_batch_bayes_processor.py

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# Add the package directory to the path
package_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, package_dir)

# Make sure the imports are working
try:
    from abr_analysis.data.loader import ABRDataLoader
    from abr_analysis.data.matcher import ControlMatcher
    from abr_analysis.models.bayesian import BayesianABRAnalysis
    from abr_analysis.analysis.batch_bayes_processor import GeneBayesianAnalyzer
    print("Imports successful!")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

def process_single_gene(gene, data_path):
    """Process a single gene and return the result."""
    # Load data for this specific process
    print(f"Processing gene: {gene}")
    analyzer = GeneBayesianAnalyzer(data_path)
    
    # Get gene data
    mutants = analyzer.data[analyzer.data['biological_sample_group'] == 'experimental']
    gene_data = mutants[mutants['gene_symbol'] == gene]
    
    # Analyze gene
    try:
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
                freq_cols = analyzer.freq_cols
                for i, freq in enumerate(freq_cols):
                    freq_name = freq.split()[0]
                    result[f'{analysis_type}_effect_{freq_name}'] = analysis['effect_sizes'][i]
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
                    
        print(f"Completed analysis for gene: {gene}")
        return result
    
    except Exception as e:
        print(f"Error processing gene {gene}: {str(e)}")
        return None

def format_gene_list(genes, columns=3):
    """Format a list of genes into columns for pretty printing."""
    genes = sorted(list(genes))
    rows = []
    for i in range(0, len(genes), columns):
        row = genes[i:i + columns]
        rows.append('\t'.join(str(g).ljust(20) for g in row))
    return '\n'.join(rows)

def compare_performance(data_path, n_genes=5):
    """Compare performance of parallel vs. sequential analysis."""
    print(f"\nComparing performance of parallel vs. sequential analysis on {n_genes} genes...")
    
    # Load and sample data
    loader = ABRDataLoader(data_path)
    data = loader.load_data()
    mutants = data[data['biological_sample_group'] == 'experimental']
    all_genes = mutants['gene_symbol'].unique()
    all_genes = all_genes[~pd.isna(all_genes)]
    
    # Select sample of genes
    np.random.seed(42)  # For reproducibility
    sample_genes = np.random.choice(all_genes, min(n_genes, len(all_genes)), replace=False)
    print(f"Selected genes: {', '.join(sample_genes)}")
    
    # Filter data to just these genes
    sample_data = data[data['gene_symbol'].isin(sample_genes) | 
                      (data['biological_sample_group'] == 'control')]
    
    # Save sample data to temp file
    sample_path = Path("temp_sample_data.csv")
    sample_data.to_csv(sample_path, index=False)
    print(f"Sample data saved to {sample_path} ({len(sample_data)} rows)")
    
    # Run sequential analysis
    print("\nRunning sequential analysis...")
    seq_start = time.time()
    
    seq_results = []
    for gene in sample_genes:
        result = process_single_gene(gene, str(sample_path))
        if result:
            seq_results.append(result)
    
    seq_time = time.time() - seq_start
    print(f"Sequential analysis completed in {seq_time:.2f} seconds")
    
    # Run parallel analysis using concurrent.futures
    print("\nRunning parallel analysis...")
    para_start = time.time()
    
    n_processes = min(2, mp.cpu_count())
    print(f"Using {n_processes} processes")
    
    # Use ProcessPoolExecutor which doesn't use daemonic processes by default
    para_results = []
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        # Submit all gene processing tasks
        futures = [executor.submit(process_single_gene, gene, str(sample_path)) 
                  for gene in sample_genes]
        
        # Collect results as they complete
        for future in futures:
            result = future.result()
            if result:
                para_results.append(result)
    
    para_time = time.time() - para_start
    print(f"Parallel analysis completed in {para_time:.2f} seconds")
    
    # Print results
    print("\nPerformance Comparison Results:")
    print(f"Sequential processing time: {seq_time:.2f} seconds")
    print(f"Parallel processing time: {para_time:.2f} seconds")
    print(f"Speedup: {seq_time/para_time:.2f}x")
    
    # Remove temp file
    if sample_path.exists():
        sample_path.unlink()
        print(f"Removed temporary file: {sample_path}")
    
    return seq_time, para_time

def run_parallel_analysis(data_path, output_dir="results/parallel_bayes", min_bf=3.0, n_processes=None):
    """Run full batch Bayesian analysis in parallel and save results."""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir) / f"parallel_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting parallel batch Bayesian analysis...")
    print(f"Results will be saved to: {output_dir}")
    
    # Initialize analyzer to get gene list
    analyzer = GeneBayesianAnalyzer(data_path)
    mutants = analyzer.data[analyzer.data['biological_sample_group'] == 'experimental']
    genes = mutants['gene_symbol'].unique()
    genes = genes[~pd.isna(genes)]  # Remove NaN values
    
    # Set up multiprocessing
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    print(f"Analyzing {len(genes)} genes using {n_processes} processes")
    
    # Run analyses in parallel
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        # Submit all gene processing tasks
        futures = [executor.submit(process_single_gene, gene, data_path) for gene in genes]
        
        # Use a progress bar
        from tqdm import tqdm
        for future in tqdm(futures, total=len(genes), desc="Processing genes"):
            result = future.result()
            if result:
                results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
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
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Save results
    results_df.to_csv(output_dir / "results.csv", index=False)
    
    print(f"Analysis completed in {total_time:.2f} seconds")
    print(f"Found {len(results_df)} genes with results")
    print(f"Results saved to {output_dir}")
    
    return results_df, output_dir

if __name__ == "__main__":
    # Path to your data
    data_path = "/Volumes/IMPC/abr_full_data.csv"
    
    # Compare performance
    compare_performance(data_path, n_genes=5)
    
    # Run full parallel analysis
    #run_parallel_analysis(data_path,
    #                      output_dir="results/parallel_bayes",
    #                      min_bf=3.0,
    #                      #n_processes=4
    #                      )