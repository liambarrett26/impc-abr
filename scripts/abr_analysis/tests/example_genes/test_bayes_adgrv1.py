# tests/test_bayes_adgrv1.py
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the package directory to the path
package_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, package_dir)

from abr_analysis.data.loader import ABRDataLoader
from abr_analysis.data.matcher import ControlMatcher
from abr_analysis.models.bayesian import BayesianABRAnalysis

def test_bayesian_adgrv1_analysis():
    """Test Bayesian analysis pipeline using Adgrv1 as a known hearing loss gene."""

    print("Loading and preparing data...")

    # Initialize data loader
    data_path = "/Volumes/IMPC/abr_full_data.csv"
    loader = ABRDataLoader(data_path)
    data = loader.load_data()

    # Get Adgrv1 data
    adgrv1_data = data[data['gene_symbol'] == 'Adgrv1']

    if len(adgrv1_data) == 0:
        raise ValueError("No data found for Adgrv1")

    print(f"Found {len(adgrv1_data)} Adgrv1 samples")

    # Get metadata from first Adgrv1 mouse for control matching
    example_adgrv1 = adgrv1_data.iloc[0]
    ko_metadata = {
        'phenotyping_center': example_adgrv1['phenotyping_center'],
        'genetic_background': example_adgrv1['genetic_background'],
        'pipeline_name': example_adgrv1['pipeline_name'],
        'metadata_Equipment manufacturer': example_adgrv1['metadata_Equipment manufacturer'],
        'metadata_Equipment model': example_adgrv1['metadata_Equipment model']
    }

    print("\nMetadata for control matching:")
    for key, value in ko_metadata.items():
        print(f"{key}: {value}")

    # Find matching controls
    matcher = ControlMatcher(data)
    controls = matcher.find_matching_controls(ko_metadata)

    print(f"\nFound {len(controls)} matching controls")

    # Get frequency columns
    freq_cols = loader.get_frequencies()
    print("\nFrequency columns:", freq_cols)

    # Print mean thresholds for reference
    print("\nMean thresholds:")
    print("Controls:")
    print(controls[freq_cols].mean())
    print("\nAdgrv1:")
    print(adgrv1_data[freq_cols].mean())

    # Extract profiles
    control_profiles = matcher.get_control_profiles(controls, freq_cols)
    adgrv1_profiles = adgrv1_data[freq_cols].values.astype(float)

    # Remove any profiles with NaN values
    control_profiles = control_profiles[~np.any(np.isnan(control_profiles), axis=1)]
    adgrv1_profiles = adgrv1_profiles[~np.any(np.isnan(adgrv1_profiles), axis=1)]

    print("\nStarting Bayesian analysis...")

    # Perform Bayesian analysis
    try:
        bayesian_analyzer = BayesianABRAnalysis()
        bayesian_analyzer.fit(control_profiles, adgrv1_profiles, freq_cols)

        # Create visualization
        print("\nGenerating visualizations...")
        fig = bayesian_analyzer.plot_results(
            control_profiles,
            adgrv1_profiles,
            gene_name='Adgrv1'
        )
        fig.savefig('adgrv1_bayesian_analysis.png')
        print("Saved visualization to 'adgrv1_bayesian_analysis.png'")

        # Calculate and print Bayes factor
        bf = bayesian_analyzer.calculate_bayes_factor()
        print(f"\nBayes factor (hearing loss vs normal): {bf:.2f}")

        # Print interpretation
        if bf > 100:
            evidence = "Extreme evidence"
        elif bf > 10:
            evidence = "Strong evidence"
        elif bf > 3:
            evidence = "Substantial evidence"
        else:
            evidence = "Weak evidence"

        print(f"Interpretation: {evidence} for hearing loss")

    except Exception as e:
        print(f"\nError in Bayesian analysis: {e}")
        raise

if __name__ == "__main__":
    test_bayesian_adgrv1_analysis()
