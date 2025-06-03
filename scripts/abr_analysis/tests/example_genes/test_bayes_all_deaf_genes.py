#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tests/test_bayes_all_deaf_genes.py

"""
Test analysis pipeline using multiple known hearing loss genes.
This script loads ABR data, finds experimental groups by allele+zygosity+center,
matches appropriate controls, and performs Bayesian analysis on each group.

Author: Liam Barrett
Version: 1.1.0
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

# Add the package directory to the path
package_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, package_dir)

from abr_analysis.data.loader import ABRDataLoader
from abr_analysis.data.matcher import ControlMatcher
from abr_analysis.models.bayesian import BayesianABRAnalysis

def analyze_gene(gene_symbol, data, matcher, output_dir, freq_cols):
    """Analyze a single gene and save results."""

    print(f"\n{'='*60}")
    print(f"ANALYZING GENE: {gene_symbol}")
    print(f"{'='*60}")

    # Find experimental groups for this gene
    experimental_groups = matcher.find_experimental_groups(gene_symbol)

    if not experimental_groups:
        print(f"No experimental groups found for {gene_symbol}")
        return {'gene': gene_symbol, 'status': 'no_groups', 'groups_analyzed': 0}

    print(f"Found {len(experimental_groups)} experimental groups for {gene_symbol}")

    # Create gene-specific output directory
    gene_output_dir = output_dir / gene_symbol
    gene_output_dir.mkdir(exist_ok=True)

    groups_analyzed = 0
    gene_results = []

    # Analyze each experimental group
    for i, group in enumerate(experimental_groups, 1):
        print(f"\n--- Processing Group {i}/{len(experimental_groups)} ---")
        print(f"Allele: {group['allele_symbol']}")
        print(f"Zygosity: {group['zygosity']}")
        print(f"Center: {group['phenotyping_center']}")
        print(f"Number of mice: {len(group['data'])}")

        try:
            # Find matching controls
            controls = matcher.find_matching_controls(group)
            print(f"Found {len(controls['all'])} matching controls")
            print(f"  Male controls: {len(controls['male'])}")
            print(f"  Female controls: {len(controls['female'])}")

            # Extract profiles
            control_profiles = matcher.get_control_profiles(controls['all'], freq_cols)
            exp_profiles = matcher.get_experimental_profiles(group, freq_cols)

            # Remove any profiles with NaN values
            control_profiles = control_profiles[~np.any(np.isnan(control_profiles), axis=1)]
            exp_profiles = exp_profiles[~np.any(np.isnan(exp_profiles), axis=1)]

            if len(control_profiles) < 20 or len(exp_profiles) < 3:
                print(f"Insufficient samples after NaN removal. Controls: {len(control_profiles)}, Mutants: {len(exp_profiles)}")
                continue

            # Print mean thresholds for reference
            print("\nMean thresholds:")
            print("Controls:", np.round(np.mean(control_profiles, axis=0), 1))
            print("Experimental:", np.round(np.mean(exp_profiles, axis=0), 1))

            # Create group identifier for output files
            group_id = f"{group['gene_symbol']}_{group['allele_symbol']}_{group['zygosity']}_{group['phenotyping_center']}"
            group_id = group_id.replace('/', '_').replace('<', '_').replace('>', '_')  # Clean filename

            # Perform Bayesian analysis
            print("\nStarting Bayesian analysis...")
            bayesian_analyzer = BayesianABRAnalysis()
            bayesian_analyzer.fit(control_profiles, exp_profiles, freq_cols)

            # Create visualization
            print("Generating visualizations...")
            fig = bayesian_analyzer.plot_results(
                control_profiles,
                exp_profiles,
                gene_name=f"{gene_symbol} ({group['allele_symbol']}, {group['zygosity']})"
            )

            # Save visualization
            output_file = gene_output_dir / f"{group_id}_bayesian_analysis.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved visualization to '{output_file}'")

            # Calculate and print Bayes factor
            bf = bayesian_analyzer.calculate_bayes_factor()
            print(f"\nBayes factor (hearing loss vs normal): {bf:.2f}")

            # Print interpretation
            if bf > 100:
                evidence = "Extreme evidence"
            elif bf > 30:
                evidence = "Very strong evidence"
            elif bf > 10:
                evidence = "Strong evidence"
            elif bf > 3:
                evidence = "Substantial evidence"
            else:
                evidence = "Weak evidence"

            print(f"Interpretation: {evidence} for hearing loss")

            # Save model specifications
            model_dir = gene_output_dir / group_id
            bayesian_analyzer.save_model(model_dir)
            print(f"Saved model to '{model_dir}'")

            # Store results for summary
            group_result = {
                'gene_symbol': gene_symbol,
                'allele_symbol': group['allele_symbol'],
                'zygosity': group['zygosity'],
                'phenotyping_center': group['phenotyping_center'],
                'n_controls': len(control_profiles),
                'n_mutants': len(exp_profiles),
                'bayes_factor': bf,
                'evidence_level': evidence,
                'control_mean_thresholds': np.mean(control_profiles, axis=0).tolist(),
                'mutant_mean_thresholds': np.mean(exp_profiles, axis=0).tolist()
            }
            gene_results.append(group_result)

            # Perform sex-specific analysis if enough samples
            for sex in ['male', 'female']:
                sex_exp_profiles = matcher.get_experimental_profiles(group, freq_cols, sex=sex)
                sex_control_profiles = matcher.get_control_profiles(controls[sex], freq_cols)

                # Remove NaN values
                sex_control_profiles = sex_control_profiles[~np.any(np.isnan(sex_control_profiles), axis=1)]
                sex_exp_profiles = sex_exp_profiles[~np.any(np.isnan(sex_exp_profiles), axis=1)]

                if len(sex_control_profiles) >= 20 and len(sex_exp_profiles) >= 3:
                    print(f"\nPerforming {sex}-specific analysis...")
                    print(f"  {sex.capitalize()} controls: {len(sex_control_profiles)}")
                    print(f"  {sex.capitalize()} mutants: {len(sex_exp_profiles)}")

                    sex_analyzer = BayesianABRAnalysis()
                    sex_analyzer.fit(sex_control_profiles, sex_exp_profiles, freq_cols)

                    # Create visualization
                    sex_fig = sex_analyzer.plot_results(
                        sex_control_profiles,
                        sex_exp_profiles,
                        gene_name=f"{gene_symbol} ({group['allele_symbol']}, {group['zygosity']}, {sex})"
                    )

                    # Save visualization
                    sex_output_file = gene_output_dir / f"{group_id}_{sex}_bayesian_analysis.png"
                    sex_fig.savefig(sex_output_file, dpi=300, bbox_inches='tight')
                    plt.close(sex_fig)

                    # Calculate and print Bayes factor
                    sex_bf = sex_analyzer.calculate_bayes_factor()
                    print(f"  {sex.capitalize()} Bayes factor: {sex_bf:.2f}")

                    # Save model
                    sex_model_dir = gene_output_dir / f"{group_id}_{sex}"
                    sex_analyzer.save_model(sex_model_dir)

                    # Add sex-specific results
                    sex_result = group_result.copy()
                    sex_result.update({
                        'sex': sex,
                        'n_controls': len(sex_control_profiles),
                        'n_mutants': len(sex_exp_profiles),
                        'bayes_factor': sex_bf,
                        'control_mean_thresholds': np.mean(sex_control_profiles, axis=0).tolist(),
                        'mutant_mean_thresholds': np.mean(sex_exp_profiles, axis=0).tolist()
                    })
                    gene_results.append(sex_result)
                else:
                    print(f"\nInsufficient samples for {sex}-specific analysis.")
                    print(f"  {sex.capitalize()} controls: {len(sex_control_profiles)}")
                    print(f"  {sex.capitalize()} mutants: {len(sex_exp_profiles)}")

            groups_analyzed += 1

        except ValueError as e:
            print(f"Error processing group: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            print(traceback.format_exc())
            continue

    # Save gene-specific results summary
    if gene_results:
        results_df = pd.DataFrame(gene_results)
        results_file = gene_output_dir / f"{gene_symbol}_analysis_summary.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nSaved gene summary to '{results_file}'")

    return {
        'gene': gene_symbol,
        'status': 'completed',
        'groups_analyzed': groups_analyzed,
        'results': gene_results
    }

def test_bayesian_all_deaf_genes():
    """Test Bayesian analysis pipeline on all specified hearing loss genes."""

    # List of genes to analyze
    genes_to_analyze = [
        'Gpr156', 'Marveld2', 'Nedd4l', 'Clrn2', 'Cimap1d', 'Slc17a8', 'Elmod1',
        'Grxcr1', 'Cep85l', 'Tprn', 'Pcyox1l', 'Otogl', 'Espnl', 'Slc5a5',
        'Gipc3', 'Triobp', 'Kcna7', 'Cib2', 'Tmco6', 'Ebf1', 'Rab11fip3',
        'Scamp2', 'Cabp2', 'Syce1l', 'Otog', 'Tox', 'Pjvk', 'Tacr3', 'Lcorl',
        'Clrn1', 'Tmtc4', 'Rnd2', 'Rab3ip', 'Hspa5', 'Cdc37l1', 'Eif2s1',
        'Nptn', 'Panx2', 'Kcnt2', 'Rrbp1', 'Mpdz', 'Adgrv1', 'Msi2'
    ]

    # Remove any NaN values (if present)
    genes_to_analyze = [gene for gene in genes_to_analyze if pd.notna(gene)]

    print(f"Starting analysis of {len(genes_to_analyze)} genes...")
    print("Genes to analyze:", genes_to_analyze)

    print("\nLoading and preparing data...")

    # Initialize data loader
    data_path = "/Users/liambarrett/github/impc-abr/data/processed/abr_full_data.csv"
    loader = ABRDataLoader(data_path)
    data = loader.load_data()

    # Get frequency columns
    freq_cols = loader.get_frequencies()
    print(f"Frequency columns: {freq_cols}")

    # Initialize control matcher
    matcher = ControlMatcher(data)

    # Create main output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"deaf_genes_analysis_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    print(f"\nResults will be saved to: {output_dir}")

    # Track overall results
    overall_results = []
    successful_genes = 0
    failed_genes = 0

    # Analyze each gene
    for i, gene_symbol in enumerate(genes_to_analyze, 1):
        print(f"\n{'#'*80}")
        print(f"PROCESSING GENE {i}/{len(genes_to_analyze)}: {gene_symbol}")
        print(f"{'#'*80}")

        try:
            result = analyze_gene(gene_symbol, data, matcher, output_dir, freq_cols)
            overall_results.append(result)

            if result['status'] == 'completed' and result['groups_analyzed'] > 0:
                successful_genes += 1
            else:
                failed_genes += 1

        except Exception as e:
            print(f"Critical error analyzing {gene_symbol}: {e}")
            print(traceback.format_exc())
            overall_results.append({
                'gene': gene_symbol,
                'status': 'failed',
                'groups_analyzed': 0,
                'error': str(e)
            })
            failed_genes += 1

    # Create overall summary
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Total genes processed: {len(genes_to_analyze)}")
    print(f"Successfully analyzed: {successful_genes}")
    print(f"Failed or no data: {failed_genes}")

    # Save overall summary
    summary_df = pd.DataFrame(overall_results)
    summary_file = output_dir / "overall_analysis_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nOverall summary saved to: {summary_file}")

    # Create detailed results summary if we have results
    all_detailed_results = []
    for result in overall_results:
        if 'results' in result and result['results']:
            all_detailed_results.extend(result['results'])

    if all_detailed_results:
        detailed_df = pd.DataFrame(all_detailed_results)
        detailed_file = output_dir / "detailed_results_summary.csv"
        detailed_df.to_csv(detailed_file, index=False)
        print(f"Detailed results saved to: {detailed_file}")

        # Print some summary statistics
        print(f"\nSummary Statistics:")
        print(f"Total experimental groups analyzed: {len(detailed_df)}")

        if 'bayes_factor' in detailed_df.columns:
            bf_stats = detailed_df['bayes_factor'].describe()
            print(f"Bayes Factor statistics:\n{bf_stats}")

            # Count evidence levels
            if 'evidence_level' in detailed_df.columns:
                evidence_counts = detailed_df['evidence_level'].value_counts()
                print(f"\nEvidence level distribution:\n{evidence_counts}")

    print(f"\nAll results saved to: {output_dir}")

if __name__ == "__main__":
    test_bayesian_all_deaf_genes()