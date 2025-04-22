import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Add the package directory to the path
package_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, package_dir)

from abr_analysis.data.loader import ABRDataLoader
from abr_analysis.data.matcher import ControlMatcher
from abr_analysis.models.distribution import RobustMultivariateGaussian

def test_Asf1b_analysis():
    """Test analysis pipeline using Asf1b as a known hearing loss gene."""
    
    # Initialize data loader
    data_path = "/Volumes/IMPC/abr_processed_data_March2025.csv"
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
    
    # Print mean thresholds for visualization
    print("\nMean thresholds:")
    print("Controls:")
    print(controls[freq_cols].mean())
    print("\nAsf1b:")
    print(Asf1b_data[freq_cols].mean())
    
    # Extract profiles
    control_profiles = matcher.get_control_profiles(controls, freq_cols)
    
    # Fit distribution to control data
    model = RobustMultivariateGaussian()
    model.fit(control_profiles)
    
    # Calculate log probabilities for all Asf1b profiles
    Asf1b_profiles = Asf1b_data[freq_cols].values.astype(float)
    Asf1b_log_probs = []
    
    for profile in Asf1b_profiles:
        if not np.any(np.isnan(profile)):  # Skip profiles with NaN values
            log_prob = model.score(profile)
            Asf1b_log_probs.append(log_prob)
    
    # Calculate control log probabilities for comparison
    control_log_probs = []
    for profile in control_profiles:
        if not np.any(np.isnan(profile)):  # Skip profiles with NaN values
            log_prob = model.score(profile)
            control_log_probs.append(log_prob)

    # Calculate Mahalanobis distances
    control_distances = []
    for profile in control_profiles:
        if not np.any(np.isnan(profile)):
            dist = model.mahalanobis(profile)
            control_distances.append(dist)
    
    Asf1b_distances = []
    for profile in Asf1b_profiles:
        if not np.any(np.isnan(profile)):
            dist = model.mahalanobis(profile)
            Asf1b_distances.append(dist)
    
    print("\nMahalanobis distance statistics:")
    print(f"Controls: mean = {np.mean(control_distances):.2f}, std = {np.std(control_distances):.2f}")
    print(f"Asf1b: mean = {np.mean(Asf1b_distances):.2f}, std = {np.std(Asf1b_distances):.2f}")
    
    print("\nLog probability statistics:")
    print(f"Controls: mean = {np.mean(control_log_probs):.2f}, std = {np.std(control_log_probs):.2f}")
    print(f"Asf1b: mean = {np.mean(Asf1b_log_probs):.2f}, std = {np.std(Asf1b_log_probs):.2f}")
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Profile Comparison
    plt.subplot(1, 2, 1)
    x = np.arange(len(freq_cols))
    plt.errorbar(x, controls[freq_cols].mean(), 
                yerr=controls[freq_cols].std(), 
                label='Controls', fmt='o-')
    plt.errorbar(x, Asf1b_data[freq_cols].mean(), 
                yerr=Asf1b_data[freq_cols].std(), 
                label='Asf1b', fmt='o-')
    plt.xticks(x, [col.split()[0] for col in freq_cols], rotation=45)
    plt.ylabel('ABR Threshold (dB SPL)')
    plt.title('ABR Profiles')
    plt.legend()
    
    # Plot 2: Log Probabilities
    plt.subplot(1, 2, 2)
    plt.hist(control_log_probs, bins=20, alpha=0.5, 
            label='Controls', density=True)
    plt.hist(Asf1b_log_probs, bins=20, alpha=0.5, 
            label='Asf1b', density=True)
    plt.xlabel('Log Probability')
    plt.ylabel('Density')
    plt.title('Distribution of Log Probabilities')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('Asf1b_test_results.png')
    plt.close()
    
    # Calculate statistical significance
    stat, pval = stats.ttest_ind(control_log_probs, Asf1b_log_probs)
    
    print(f"\nStatistical test results:")
    print(f"t-statistic: {stat:.2f}")
    print(f"p-value: {pval:.2e}")

if __name__ == "__main__":
    test_Asf1b_analysis()