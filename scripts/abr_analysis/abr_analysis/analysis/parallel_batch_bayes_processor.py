# abr_analysis/analysis/parallel_batch_bayes_processor.py

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import os
import time

from ..data.loader import ABRDataLoader
from ..data.matcher import ControlMatcher
from ..models.bayesian import BayesianABRAnalysis
from .batch_bayes_processor import GeneBayesianAnalyzer

class ParallelGeneBayesianAnalyzer(GeneBayesianAnalyzer):
    """Analyze genes in dataset using Bayesian approach with parallelization."""

    def __init__(self, data_path, n_processes=None):
        """
        Initialize the parallel analyzer.
        
        Parameters:
        -----------
        data_path : str
            Path to the ABR data file
        n_processes : int, optional
            Number of processes to use. Defaults to number of CPU cores - 1.
        """
        super().__init__(data_path)
        self.n_processes = n_processes if n_processes else max(1, mp.cpu_count() - 1)
        print(f"Initializing parallel analysis with {self.n_processes} processes")

    def _process_gene_batch(self, genes_batch):
        """Process a batch of genes and return their results."""
        mutants = self.data[self.data['biological_sample_group'] == 'experimental']
        results = []
        
        for gene in genes_batch:
            gene_data = mutants[mutants['gene_symbol'] == gene]
            
            # Analyze for all data, males only, and females only
            analyses = {
                'all': self.analyze_gene(gene_data),
                'male': self.analyze_gene(gene_data, 'male'),
                'female': self.analyze_gene(gene_data, 'female')
            }

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
            
        return results

    def _initialize_shared_objects(self):
        """
        Initialize objects needed for all processes (called by each worker).
        This avoids serializing and passing large data between processes.
        """
        # Objects are already loaded in __init__, no further action needed
        pass

    def analyze_all_genes(self, chunk_size=None):
        """
        Analyze all genes in the dataset using Bayesian approach with parallelization.
        
        Parameters:
        -----------
        chunk_size : int, optional
            Size of gene batches to process in each worker. 
            Default is to automatically determine based on number of genes and processes.
            
        Returns:
        --------
        pandas.DataFrame
            Results dataframe with all gene analyses
        """
        mutants = self.data[self.data['biological_sample_group'] == 'experimental']
        genes = mutants['gene_symbol'].unique()
        genes = genes[~pd.isna(genes)]  # Remove NaN values
        
        # Determine chunk size if not provided
        if chunk_size is None:
            chunk_size = max(1, len(genes) // (self.n_processes * 2))
        
        # Split genes into chunks
        gene_chunks = [genes[i:i + chunk_size] for i in range(0, len(genes), chunk_size)]
        
        print(f"Analyzing {len(genes)} genes in {len(gene_chunks)} chunks of size ~{chunk_size}")
        start_time = time.time()
        
        # Create a multiprocessing pool
        with mp.Pool(processes=self.n_processes) as pool:
            # Use tqdm to show progress
            results_list = list(tqdm(
                pool.imap(self._process_gene_batch, gene_chunks),
                total=len(gene_chunks),
                desc="Processing gene batches"
            ))
        
        # Flatten the list of lists
        all_results = [item for sublist in results_list for item in sublist]
        
        # Convert to DataFrame
        self.results = pd.DataFrame(all_results)
        
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
        
        end_time = time.time()
        print(f"Analysis completed in {end_time - start_time:.2f} seconds")
        
        return self.results