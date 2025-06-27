#!/bin/env python
# -*- coding: utf-8 -*-
# scripts/run_parallel_analysis.py

"""
Command-line interface for parallel Bayesian ABR data analysis.

This script provides a convenient command-line utility for running parallel
Bayesian analysis on Auditory Brainstem Response (ABR) data. It serves as
the main entry point for executing high-throughput analysis of large ABR datasets
using multiple CPU cores to accelerate computation.

The script handles:
- Command-line argument parsing for analysis configuration
- Optional reference gene list integration for result comparison
- Parallel execution coordination and progress reporting
- Result visualisation and report generation
- Performance timing and resource management

Usage examples:
    python run_parallel_analysis.py --data path/to/abr_data.csv --processes 8
    python run_parallel_analysis.py --data path/to/abr_data.csv --min-bf 5 --batch-size 20
    python run_parallel_analysis.py --data path/to/abr_data.csv --reference-genes known_genes.txt

Author: Liam Barrett
Version: 1.0.1
"""

import os
import sys
from pathlib import Path
import argparse
import time

# Add the package directory to the path
script_dir = Path(__file__).parent
package_dir = script_dir.parent
sys.path.insert(0, str(package_dir))

from abr_analysis.analysis.parallel_executor import run_parallel_analysis, create_visualizations

def main():
    """
    Main entry point for the parallel ABR analysis command-line tool.
    
    This function:
    - Parses command-line arguments to configure the analysis
    - Handles optional reference gene list loading
    - Executes the parallel Bayesian analysis process
    - Generates visualisations and reports
    - Times the overall execution and provides feedback
    
    The function coordinates the entire analysis workflow from data loading through
    to result generation, managing resources and providing appropriate user feedback
    throughout the process. It serves as the central orchestrator for the
    command-line utility.
    
    Returns:
        None: Results are saved to the specified output directory
    """
    parser = argparse.ArgumentParser(description='Run parallel Bayesian analysis of ABR data')
    parser.add_argument('--data', required=True, help='Path to the ABR data file')
    parser.add_argument('--output', default='results/parallel_bayes', help='Output directory')
    parser.add_argument('--processes', type=int, default=None,
                        help='Number of processes to use (default: CPU count - 1)')
    parser.add_argument('--min-bf', type=float, default=3.0,
                        help='Minimum Bayes factor for significance (default: 3.0)')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Number of genes to process in each batch (default: 10)')
    parser.add_argument('--reference-genes', type=str, default=None,
                        help='Path to a file with reference gene symbols (one per line)')

    args = parser.parse_args()

    print(f"Starting parallel Bayesian analysis with {args.processes if args.processes else 'auto'} processes")
    start_time = time.time()

    results_df, _, output_dir = run_parallel_analysis(
        data_path=args.data,
        output_dir=args.output,
        min_bf=args.min_bf,
        n_processes=args.processes,
        batch_size=args.batch_size
    )

    # Create summary visualizations with reference genes if provided
    if results_df is not None:
        create_visualizations(results_df, output_dir, './data/multivariate_confirmed_deafness_genes.txt',
                              './data/multivariate_candidate_deafness_genes.txt', args.min_bf)

    total_time = time.time() - start_time
    print(f"Analysis completed in {total_time:.2f} seconds")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
