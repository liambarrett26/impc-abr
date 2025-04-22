# scripts/run_parallel_analysis.py

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
    parser = argparse.ArgumentParser(description='Run parallel Bayesian analysis of ABR data')
    parser.add_argument('--data', required=True, help='Path to the ABR data file')
    parser.add_argument('--output', default='results/parallel_bayes', help='Output directory')
    parser.add_argument('--processes', type=int, default=None, 
                        help='Number of processes to use (default: CPU count - 1)')
    parser.add_argument('--min-bf', type=float, default=3.0, 
                        help='Minimum Bayes factor for significance (default: 3.0)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of genes to process in each batch (default: 10)')
    parser.add_argument('--reference-genes', type=str, default=None,
                        help='Path to a file with reference gene symbols (one per line)')
    
    args = parser.parse_args()
    
    # Load reference genes if specified
    reference_genes = None
    if args.reference_genes and os.path.exists(args.reference_genes):
        with open(args.reference_genes, 'r') as f:
            reference_genes = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(reference_genes)} reference genes from {args.reference_genes}")
    
    print(f"Starting parallel Bayesian analysis with {args.processes if args.processes else 'auto'} processes")
    start_time = time.time()
    
    results_df, output_dir = run_parallel_analysis(
        data_path=args.data,
        output_dir=args.output,
        min_bf=args.min_bf,
        n_processes=args.processes,
        batch_size=args.batch_size
    )
    
    # Create summary visualizations with reference genes if provided
    if results_df is not None:
        create_visualizations(results_df, output_dir, reference_genes, args.min_bf)
    
    total_time = time.time() - start_time
    print(f"Analysis completed in {total_time:.2f} seconds")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()