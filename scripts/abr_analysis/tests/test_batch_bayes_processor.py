# tests/test_batch_bayes_processor.py
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

from abr_analysis.analysis.batch_bayes_processor import GeneBayesianAnalyzer

def format_gene_list(genes, columns=3):
    """Format a list of genes into columns for pretty printing."""
    genes = sorted(list(genes))
    rows = []
    for i in range(0, len(genes), columns):
        row = genes[i:i + columns]
        rows.append('\t'.join(str(g).ljust(20) for g in row))
    return '\n'.join(rows)

def run_batch_bayesian_analysis(data_path, output_dir="results/bayes", min_bf=3.0):
    """Run full batch Bayesian analysis and save results."""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir) / f"bayes_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting batch Bayesian analysis...")
    print(f"Results will be saved to: {output_dir}")
    
    # Initialize and run analyzer
    analyzer = GeneBayesianAnalyzer(data_path)
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
    comparisons = analyzer.compare_with_bowl(bowl_genes, min_bf)
    
    # Generate report
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write("ABR Bayesian Analysis Report\n")
        f.write("==========================\n\n")
        
        # Overall statistics
        f.write("Overall Statistics\n")
        f.write("-----------------\n")
        f.write(f"Total genes analyzed: {len(results)}\n")
        f.write(f"Bowl et al. genes: {len(bowl_genes)}\n")
        f.write(f"Significance threshold: Bayes Factor > {min_bf}\n\n")
        
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
            
        # Evidence level breakdown
        f.write("\nEvidence Level Breakdown\n")
        f.write("----------------------\n")
        evidence_counts = results['all_evidence'].value_counts()
        for evidence, count in evidence_counts.items():
            f.write(f"{evidence}: {count} genes ({count/len(results)*100:.1f}%)\n")
        
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
    analyzer.create_visualizations(output_dir, min_bf)
    
    # Print summary to console
    print("\nBayesian analysis complete! Summary of findings:")
    print("-" * 50)
    for analysis_type in ['all', 'male', 'female']:
        comp = comparisons[analysis_type]
        print(f"\n{analysis_type.upper()}:")
        print(f"Total significant: {len(comp['found_in_bowl']) + len(comp['novel'])}")
        print(f"Found in Bowl: {len(comp['found_in_bowl'])}")
        print(f"Novel: {len(comp['novel'])}")
    
    # Print evidence breakdown
    print("\nEvidence Level Breakdown:")
    for evidence, count in evidence_counts.items():
        print(f"{evidence}: {count} genes ({count/len(results)*100:.1f}%)")
    
    print(f"\nFull results saved to: {output_dir}")
    return results, comparisons, output_dir

def test_on_sample_genes(data_path, gene_list=None, output_dir="results/bayes"):
    """Run Bayesian analysis on a small set of genes to verify the pipeline."""
    if gene_list is None:
        # Default to testing these three genes - known hearing loss (Adgrv1), 
        # a gene with mild hearing loss (Nptn), and a gene with visually insufficient evidence (Abca2)
        gene_list = ['Adgrv1', 'Nptn', 'Abca2']
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir) / f"sample_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRunning sample test on genes: {', '.join(gene_list)}")
    print(f"Results will be saved to: {output_dir}")
    
    # Initialize analyzer
    analyzer = GeneBayesianAnalyzer(data_path)
    
    # Load data
    data = analyzer.data
    mutants = data[data['biological_sample_group'] == 'experimental']
    
    # Filter to just the sample genes
    sample_data = mutants[mutants['gene_symbol'].isin(gene_list)]
    
    # Initialize results list
    results = []
    
    # Process each gene
    for gene in gene_list:
        print(f"\nProcessing gene: {gene}")
        gene_data = sample_data[sample_data['gene_symbol'] == gene]
        
        if len(gene_data) == 0:
            print(f"No data found for gene {gene}. Skipping.")
            continue
            
        # Analyze just for 'all' data
        analysis = analyzer.analyze_gene(gene_data)
        
        if analysis is None:
            print(f"Analysis failed for gene {gene}. Skipping.")
            continue
            
        # Add to results
        result = {
            'gene_symbol': gene,
            'center': gene_data['phenotyping_center'].iloc[0],
            'background': gene_data['genetic_background'].iloc[0]
        }
        
        # Include main statistics
        result['all_bayes_factor'] = analysis['bayes_factor']
        result['all_p_hearing_loss'] = analysis['p_hearing_loss']
        result['all_hdi_lower'] = analysis['hdi_lower']
        result['all_hdi_upper'] = analysis['hdi_upper']
        result['all_n_mutants'] = analysis['n_mutants']
        result['all_n_controls'] = analysis['n_controls']
        
        # Include effect sizes for each frequency
        for i, freq in enumerate(analyzer.freq_cols):
            freq_name = freq.split()[0]
            result[f'all_effect_{freq_name}'] = analysis['effect_sizes'][i]
            
        results.append(result)
        
        # Create visualization for this gene
        print(f"Creating visualizations for {gene}...")
        analyzer.create_gene_visualization(gene, output_dir, 'all')
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "sample_results.csv", index=False)
    
    # Create simple report
    report_path = output_dir / "sample_report.txt"
    with open(report_path, 'w') as f:
        f.write("Sample Test Results\n")
        f.write("==================\n\n")
        
        for gene in gene_list:
            if gene in results_df['gene_symbol'].values:
                gene_row = results_df[results_df['gene_symbol'] == gene].iloc[0]
                
                f.write(f"Gene: {gene}\n")
                f.write("-" * (len(gene) + 6) + "\n")
                f.write(f"Bayes Factor: {gene_row['all_bayes_factor']:.2f}\n")
                f.write(f"P(Hearing Loss): {gene_row['all_p_hearing_loss']:.3f}\n")
                f.write(f"95% HDI: [{gene_row['all_hdi_lower']:.3f}, {gene_row['all_hdi_upper']:.3f}]\n")
                f.write(f"Sample size: {gene_row['all_n_mutants']} mutants, {gene_row['all_n_controls']} controls\n\n")
                
                # Evidence interpretation
                bf = gene_row['all_bayes_factor']
                if bf > 100:
                    evidence = "Extreme"
                elif bf > 30:
                    evidence = "Very Strong"
                elif bf > 10:
                    evidence = "Strong"
                elif bf > 3:
                    evidence = "Substantial"
                else:
                    evidence = "Weak/None"
                    
                f.write(f"Evidence level: {evidence}\n\n")
            else:
                f.write(f"Gene: {gene}\n")
                f.write("-" * (len(gene) + 6) + "\n")
                f.write("No results available\n\n")
    
    print(f"\nSample test complete! Results saved to: {output_dir}")
    return results_df, output_dir

if __name__ == "__main__":
    # Path to your data
    data_path = "/Volumes/IMPC/abr_full_data.csv"
    
    # To test on just a few genes:
    test_on_sample_genes(data_path)
    
    # Or run the full analysis:
    #try:
    #    results, comparisons, output_dir = run_batch_bayesian_analysis(data_path)
    #    print("\nTest completed successfully!")
    #except Exception as e:
    #    print(f"\nError during test: {e}")
    #    raise