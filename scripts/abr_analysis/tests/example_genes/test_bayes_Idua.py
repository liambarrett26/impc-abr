#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tests/test_bayes_Idua.py

"""
Test analysis pipeline using Idua as a known hearing loss gene.
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

# Add the package directory to the path
package_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, package_dir)

from abr_analysis.data.loader import ABRDataLoader
from abr_analysis.data.matcher import ControlMatcher
from abr_analysis.models.bayesian import BayesianABRAnalysis

def sanity_check_data(control_profiles, exp_profiles, freq_cols):
    """Perform sanity check on the data."""
    print("\n=== SANITY CHECK ===")
    control_mean = np.mean(control_profiles, axis=0)
    exp_mean = np.mean(exp_profiles, axis=0)
    shifts = exp_mean - control_mean

    print("Frequency-wise analysis:")
    for i, freq in enumerate(freq_cols):
        freq_name = freq.split()[0]
        print(f"{freq_name}: Control={control_mean[i]:.1f}, Mutant={exp_mean[i]:.1f}, Shift={shifts[i]:.1f} dB")

    print(f"\nMax shift: {np.max(shifts):.1f} dB")
    print(f"Min shift: {np.min(shifts):.1f} dB")
    print(f"All shifts > 40 dB: {np.all(shifts > 40)}")
    print(f"Mutant std dev: {np.std(exp_profiles):.1f}")
    print("==================")

# Add this call in your test script before the Bayesian analysis:


def test_bayesian_Idua_analysis():
    """Test Bayesian analysis pipeline using Idua as a known hearing loss gene."""

    print("Loading and preparing data...")

    # Initialize data loader
    data_path = "/home/liamb/impc-abr/data/processed/abr_missing_genes_data_v4.csv"
    loader = ABRDataLoader(data_path)
    data = loader.load_data()

    # Extract gene symbols from allele_symbol column
    def extract_gene_symbol(allele_symbol):
        if pd.isna(allele_symbol):
            return None
        return str(allele_symbol).split('<')[0]

    data['gene_symbol'] = data['allele_symbol'].apply(extract_gene_symbol)

    # Get frequency columns
    freq_cols = loader.get_frequencies()
    print("\nFrequency columns:", freq_cols)

    # Initialize control matcher
    matcher = ControlMatcher(data)

    # Find experimental groups for Idua
    gene_symbol = 'Idua'
    experimental_groups = matcher.find_experimental_groups(gene_symbol)

    if not experimental_groups:
        raise ValueError(f"No experimental groups found for {gene_symbol}")

    print(f"\nFound {len(experimental_groups)} experimental groups for {gene_symbol}:")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{gene_symbol}_analysis_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    # Analyze each experimental group
    for i, group in enumerate(experimental_groups, 1):
        print(f"\n--- Processing Group {i} ---")
        print(f"Allele: {group['allele_symbol']}")
        print(f"Zygosity: {group['zygosity']}")
        print(f"Center: {group['phenotyping_center']}")
        print(f"Number of mice: {len(group['data'])}")

        # Find matching controls
        try:
            controls = matcher.find_matching_controls(group)
            print(f"Found {len(controls['all'])} matching controls")
            print(f"  Male controls: {len(controls['male'])}")
            print(f"  Female controls: {len(controls['female'])}")

            # Extract profiles
            control_profiles = matcher.get_control_profiles(controls['all'], freq_cols)
            exp_profiles = matcher.get_experimental_profiles(group, freq_cols)
            
            # Diagnostic: Check NaN patterns before removal
            print("\nDiagnostic - Experimental mice ABR data:")
            print(f"Total experimental mice: {len(exp_profiles)}")
            for i, profile in enumerate(exp_profiles):
                nan_mask = np.isnan(profile)
                nan_count = np.sum(nan_mask)
                print(f"  Mouse {i+1}: {nan_count} NaN values out of {len(profile)} frequencies")
                if nan_count > 0:
                    missing_freqs = [freq_cols[j] for j in range(len(profile)) if nan_mask[j]]
                    print(f"    Missing: {', '.join(missing_freqs)}")
            
            # Check if we have any frequency with complete data
            freq_completeness = []
            for i, freq in enumerate(freq_cols):
                complete_count = np.sum(~np.isnan(exp_profiles[:, i]))
                freq_completeness.append((freq, complete_count))
                # Calculate mean threshold for mice with data
                valid_values = exp_profiles[:, i][~np.isnan(exp_profiles[:, i])]
                if len(valid_values) > 0:
                    mean_threshold = np.mean(valid_values)
                    print(f"\n{freq}: {complete_count}/{len(exp_profiles)} mice have data (mean: {mean_threshold:.1f} dB)")
                else:
                    print(f"\n{freq}: {complete_count}/{len(exp_profiles)} mice have data")

            # Remove any profiles with NaN values
            control_profiles_clean = control_profiles[~np.any(np.isnan(control_profiles), axis=1)]
            exp_profiles_clean = exp_profiles[~np.any(np.isnan(exp_profiles), axis=1)]

            if len(control_profiles_clean) < 20 or len(exp_profiles_clean) < 3:
                print(f"\nInsufficient samples after NaN removal. Controls: {len(control_profiles_clean)}, Mutants: {len(exp_profiles_clean)}")
                
                # Try to find if we can use a subset of frequencies
                print("\nChecking for frequency subsets with sufficient data...")
                # Find frequencies where at least 3 mutants have data
                viable_freqs = []
                for i, (freq, count) in enumerate(freq_completeness):
                    if count >= 3:
                        viable_freqs.append((i, freq, count))
                
                if len(viable_freqs) >= 2:  # Need at least 2 frequencies for meaningful analysis
                    print(f"Found {len(viable_freqs)} frequencies with sufficient data:")
                    for idx, freq, count in viable_freqs:
                        print(f"  {freq}: {count} mice")
                    print("\nConsider modifying the analysis to use only these frequencies.")
                
                continue

            # Print mean thresholds for reference
            print("\nMean thresholds:")
            print("Controls:")
            print(np.mean(control_profiles, axis=0))
            print("\nExperimental:")
            print(np.mean(exp_profiles, axis=0))

            # Create group identifier for output files
            group_id = f"{group['gene_symbol']}_{group['allele_symbol']}_{group['zygosity']}_{group['phenotyping_center']}"

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
            output_file = output_dir / f"{group_id}_bayesian_analysis.png"
            fig.savefig(output_file)
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
            model_dir = output_dir / group_id
            bayesian_analyzer.save_model(model_dir)
            print(f"Saved model to '{model_dir}'")

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
                    sex_output_file = output_dir / f"{group_id}_{sex}_bayesian_analysis.png"
                    sex_fig.savefig(sex_output_file)
                    plt.close(sex_fig)

                    # Calculate and print Bayes factor
                    sex_bf = sex_analyzer.calculate_bayes_factor()
                    print(f"  {sex.capitalize()} Bayes factor: {sex_bf:.2f}")

                    # Save model
                    sex_model_dir = output_dir / f"{group_id}_{sex}"
                    sex_analyzer.save_model(sex_model_dir)
                else:
                    print(f"\nInsufficient samples for {sex}-specific analysis.")
                    print(f"  {sex.capitalize()} controls: {len(sex_control_profiles)}")
                    print(f"  {sex.capitalize()} mutants: {len(sex_exp_profiles)}")

        except ValueError as e:
            print(f"Error processing group: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue

    print("\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    test_bayesian_Idua_analysis()