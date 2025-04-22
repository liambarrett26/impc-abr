# tests/test_bayes_Asf1b.py
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

def test_bayesian_Asf1b_analysis():
    """Test Bayesian analysis pipeline using Asf1b as a known hearing loss gene."""

    print("Loading and preparing data...")

    # Initialize data loader
    data_path = "/Volumes/IMPC/abr_full_data.csv"
    loader = ABRDataLoader(data_path)
    data = loader.load_data()

    # Get Asf1b data
    Asf1b_data = data[data['gene_symbol'] == 'Asf1b']

    if len(Asf1b_data) == 0:
        raise ValueError("No data found for Asf1b")

    print(f"Found {len(Asf1b_data)} Asf1b samples")

    # Get metadata from first Asf1b mouse for control matching
    example_Asf1b = Asf1b_data.iloc[0]
    ko_metadata = {
        'phenotyping_center': example_Asf1b['phenotyping_center'],
        'genetic_background': example_Asf1b['genetic_background'],
        'pipeline_name': example_Asf1b['pipeline_name'],
        'metadata_Equipment manufacturer': example_Asf1b['metadata_Equipment manufacturer'],
        'metadata_Equipment model': example_Asf1b['metadata_Equipment model']
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
    print("\nAsf1b:")
    print(Asf1b_data[freq_cols].mean())

    # Extract profiles
    control_profiles = matcher.get_control_profiles(controls, freq_cols)
    Asf1b_profiles = Asf1b_data[freq_cols].values.astype(float)

    # Remove any profiles with NaN values
    control_profiles = control_profiles[~np.any(np.isnan(control_profiles), axis=1)]
    Asf1b_profiles = Asf1b_profiles[~np.any(np.isnan(Asf1b_profiles), axis=1)]

    print("\nStarting Bayesian analysis...")

    # Perform Bayesian analysis
    try:
        bayesian_analyzer = BayesianABRAnalysis()
        bayesian_analyzer.fit(control_profiles, Asf1b_profiles, freq_cols)

        # Create visualization
        print("\nGenerating visualizations...")
        fig = bayesian_analyzer.plot_results(
            control_profiles,
            Asf1b_profiles,
            gene_name='Asf1b'
        )
        fig.savefig('Asf1b_bayesian_analysis.png')
        print("Saved visualization to 'Asf1b_bayesian_analysis.png'")

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
    test_bayesian_Asf1b_analysis()
