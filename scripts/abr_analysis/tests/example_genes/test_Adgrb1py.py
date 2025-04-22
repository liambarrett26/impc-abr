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

def test_Adgrb1_analysis():
    """Test analysis pipeline using Adgrb1 as a known hearing loss gene."""
    
    # Initialize data loader
    data_path = "/Volumes/IMPC/abr_processed_data_March2025.csv"
    loader = ABRDataLoader(data_path)
    data = loader.load_data()
    
    # Get Adgrb1 data
    Adgrb1_data = data[data['gene_symbol'] == 'Adgrb1']
    
    if len(Adgrb1_data) == 0:
        raise ValueError("No data found for Adgrb1")
    
    print(f"Found {len(Adgrb1_data)} Adgrb1 samples")
    
    # Get metadata from first Adgrb1 mouse for control matching
    example_Adgrb1 = Adgrb1_data.iloc[0]
    ko_metadata = {
        'phenotyping_center': example_Adgrb1['phenotyping_center'],
        'genetic_background': example_Adgrb1['genetic_background'],
        'pipeline_name': example_Adgrb1['pipeline_name'],
        'metadata_Equipment manufacturer': example_Adgrb1['metadata_Equipment manufacturer'],
        'metadata_Equipment model': example_Adgrb1['metadata_Equipment model']
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
    print("\nAdgrb1:")
    print(Adgrb1_data[freq_cols].mean())
    
    # Extract profiles
    control_profiles = matcher.get_control_profiles(controls, freq_cols)
    
    # Fit distribution to control data
    model = RobustMultivariateGaussian()
    model.fit(control_profiles)
    
    # Calculate log probabilities for all Adgrb1 profiles
    Adgrb1_profiles = Adgrb1_data[freq_cols].values.astype(float)
    Adgrb1_log_probs = []
    
    for profile in Adgrb1_profiles:
        if not np.any(np.isnan(profile)):  # Skip profiles with NaN values
            log_prob = model.score(profile)
            Adgrb1_log_probs.append(log_prob)
    
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
    
    Adgrb1_distances = []
    for profile in Adgrb1_profiles:
        if not np.any(np.isnan(profile)):
            dist = model.mahalanobis(profile)
            Adgrb1_distances.append(dist)
    
    print("\nMahalanobis distance statistics:")
    print(f"Controls: mean = {np.mean(control_distances):.2f}, std = {np.std(control_distances):.2f}")
    print(f"Adgrb1: mean = {np.mean(Adgrb1_distances):.2f}, std = {np.std(Adgrb1_distances):.2f}")
    
    print("\nLog probability statistics:")
    print(f"Controls: mean = {np.mean(control_log_probs):.2f}, std = {np.std(control_log_probs):.2f}")
    print(f"Adgrb1: mean = {np.mean(Adgrb1_log_probs):.2f}, std = {np.std(Adgrb1_log_probs):.2f}")
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Profile Comparison
    plt.subplot(1, 2, 1)
    x = np.arange(len(freq_cols))
    plt.errorbar(x, controls[freq_cols].mean(), 
                yerr=controls[freq_cols].std(), 
                label='Controls', fmt='o-')
    plt.errorbar(x, Adgrb1_data[freq_cols].mean(), 
                yerr=Adgrb1_data[freq_cols].std(), 
                label='Adgrb1', fmt='o-')
    plt.xticks(x, [col.split()[0] for col in freq_cols], rotation=45)
    plt.ylabel('ABR Threshold (dB SPL)')
    plt.title('ABR Profiles')
    plt.legend()
    
    # Plot 2: Log Probabilities
    plt.subplot(1, 2, 2)
    plt.hist(control_log_probs, bins=20, alpha=0.5, 
            label='Controls', density=True)
    plt.hist(Adgrb1_log_probs, bins=20, alpha=0.5, 
            label='Adgrb1', density=True)
    plt.xlabel('Log Probability')
    plt.ylabel('Density')
    plt.title('Distribution of Log Probabilities')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('Adgrb1_test_results.png')
    plt.close()
    
    # Calculate statistical significance
    stat, pval = stats.ttest_ind(control_log_probs, Adgrb1_log_probs)
    
    print(f"\nStatistical test results:")
    print(f"t-statistic: {stat:.2f}")
    print(f"p-value: {pval:.2e}")

if __name__ == "__main__":
    test_Adgrb1_analysis()