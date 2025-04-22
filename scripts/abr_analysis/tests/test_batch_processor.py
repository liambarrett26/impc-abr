# tests/test_batch_processor.py
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the package directory to the path
package_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, package_dir)

from abr_analysis.analysis.batch_processor import GeneBatchAnalyzer

def format_gene_list(genes, columns=3):
    """Format a list of genes into columns for pretty printing."""
    genes = sorted(list(genes))
    rows = []
    for i in range(0, len(genes), columns):
        row = genes[i:i + columns]
        rows.append('\t'.join(str(g).ljust(20) for g in row))
    return '\n'.join(rows)

def run_batch_analysis(data_path, output_dir="results/multivariate"):
    """Run full batch analysis and save results."""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir) / f"analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting batch analysis...")
    print(f"Results will be saved to: {output_dir}")
    
    # Initialize and run analyzer
    analyzer = GeneBatchAnalyzer(data_path)
    results = analyzer.analyze_all_genes()
    
    # Define Bowl's genes
    bowl_genes = {
        'Minar2', 'Aak1', 'Acsl4', 'Acvr2a', 'Adgrb1', 'Adgrv1', 'Ahsg', 
        'Ankrd11', 'Ap3m2', 'Ap3s1', 'Atp2b1', 'Pramel17', 'Baiap2l2', 
        'Ccdc88c', 'Ccdc92', 'Cib2', 'Clrn1', 'Col9a2', 'Cyb5r2', 'Dnase1', 
        'Duoxa2', 'Elmod1', 'Emb', 'Eps8l1', 'Ewsr1', 'Gata2', 'Gga1', 'Gipc3', 
        'Gpr152', 'Gpr50', 'Ikzf5', 'Il1r2', 'Ildr1', 'Klc2', 'Klhl18', 'Marveld2', 
        'Med28', 'Mpdz', 'Myh1', 'Myo7a', 'Nedd4l', 'Nfatc3', 'Nin', 'Nisch', 
        'Nptn', 'Ocm', 'Cimap1d', 'Otoa', 'Phf6', 'Ppm1a', 'Sema3f', 'Slc4a10', 
        'Slc5a5', 'Spns2', 'Srrm4', 'Tmem30b', 'Tmtc4', 'Tox', 'Tprn', 'Tram2', 
        'Ube2b', 'Ube2g1', 'Ush1c', 'Vti1a', 'Wdtc1', 'Zcchc14', 'Zfp719'
    }
    
    # Compare with Bowl's genes
    comparisons = analyzer.compare_with_bowl(bowl_genes)
    
    # Generate report
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write("ABR Analysis Report\n")
        f.write("=================\n\n")
        
        # Overall statistics
        f.write("Overall Statistics\n")
        f.write("-----------------\n")
        f.write(f"Total genes analyzed: {len(results)}\n")
        f.write(f"Bowl et al. genes: {len(bowl_genes)}\n\n")
        
        # Results for each analysis type
        for analysis_type in ['all', 'male', 'female']:
            f.write(f"\n{analysis_type.upper()} ANALYSIS\n")
            f.write("-" * (len(analysis_type) + 9) + "\n")
            
            comp = comparisons[analysis_type]
            
            f.write(f"Total significant genes: {len(comp['found_in_bowl']) + len(comp['novel'])}\n")
            f.write(f"Found in Bowl et al.: {len(comp['found_in_bowl'])}\n")
            f.write(f"Novel candidates: {len(comp['novel'])}\n")
            f.write(f"Missed from Bowl et al.: {len(comp['missed_from_bowl'])}\n\n")
            
            f.write("Found in Bowl et al.:\n")
            f.write(format_gene_list(comp['found_in_bowl']))
            f.write("\n\nNovel candidates:\n")
            f.write(format_gene_list(comp['novel']))
            f.write("\n\nMissed from Bowl et al.:\n")
            f.write(format_gene_list(comp['missed_from_bowl']))
            f.write("\n\n")
            
        # Sample size statistics
        f.write("\nSample Size Statistics\n")
        f.write("--------------------\n")
        for analysis_type in ['all', 'male', 'female']:
            f.write(f"\n{analysis_type.upper()}:\n")
            f.write(f"Mean mutants per gene: {results[f'{analysis_type}_n_mutants'].mean():.1f}\n")
            f.write(f"Mean controls per gene: {results[f'{analysis_type}_n_controls'].mean():.1f}\n")
    
    # Save detailed results
    results.to_csv(output_dir / "detailed_results.csv", index=False)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    analyzer.create_visualizations(output_dir)
    
    # Print summary to console
    print("\nAnalysis complete! Summary of findings:")
    print("-" * 40)
    for analysis_type in ['all', 'male', 'female']:
        comp = comparisons[analysis_type]
        print(f"\n{analysis_type.upper()}:")
        print(f"Total significant: {len(comp['found_in_bowl']) + len(comp['novel'])}")
        print(f"Found in Bowl: {len(comp['found_in_bowl'])}")
        print(f"Novel: {len(comp['novel'])}")
    
    print(f"\nFull results saved to: {output_dir}")
    return results, comparisons, output_dir

if __name__ == "__main__":
    # Path to your data
    data_path = "/Volumes/IMPC/abr_processed_data_March2025.csv"
    
    try:
        results, comparisons, output_dir = run_batch_analysis(data_path)
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nError during test: {e}")
        raise