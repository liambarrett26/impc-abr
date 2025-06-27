#!/usr/bin/env python3
"""
Check for missing sex-specific Bayes factors in concatenated results
and verify if they could have been run based on data availability.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load the results and full dataset
results_path = Path('/home/liamb/impc-abr/results/concatenated_results_v5.csv')
data_path = Path('/home/liamb/impc-abr/data/processed/abr_full_data.csv')

print("Loading data files...")
results_df = pd.read_csv(results_path)
full_data = pd.read_csv(data_path)

# Identify frequency columns
freq_cols = ['6kHz-evoked ABR Threshold', '12kHz-evoked ABR Threshold', 
             '18kHz-evoked ABR Threshold', '24kHz-evoked ABR Threshold', 
             '30kHz-evoked ABR Threshold']

print(f"\nTotal rows in results: {len(results_df)}")
print(f"Total rows in full data: {len(full_data)}")

# Find rows with missing sex-specific Bayes factors
missing_male = results_df[results_df['male_bayes_factor'].isna()]
missing_female = results_df[results_df['female_bayes_factor'].isna()]

print(f"\nRows with missing male Bayes factor: {len(missing_male)}")
print(f"Rows with missing female Bayes factor: {len(missing_female)}")

# Function to check if a sex-specific analysis could have been run
def check_sex_specific_feasibility(row, sex, full_data, freq_cols):
    """Check if sex-specific analysis was feasible for a given row."""
    
    # Extract group identifiers
    gene = row['gene_symbol']
    allele = row['allele_symbol']
    zygosity = row['zygosity']
    center = row['center']
    
    # Get experimental mice for this group and sex
    exp_mice = full_data[
        (full_data['gene_symbol'] == gene) &
        (full_data['allele_symbol'] == allele) &
        (full_data['zygosity'] == zygosity) &
        (full_data['phenotyping_center'] == center) &
        (full_data['biological_sample_group'] == 'experimental') &
        (full_data['sex'] == sex)
    ]
    
    # Check for complete ABR data
    exp_complete = exp_mice.dropna(subset=freq_cols)
    n_exp_complete = len(exp_complete)
    
    # Get matching controls based on metadata
    metadata_cols = [
        'phenotyping_center',
        'genetic_background',
        'pipeline_name',
        'metadata_Equipment manufacturer',
        'metadata_Equipment model'
    ]
    
    # Get metadata from experimental group (use first mouse)
    if len(exp_mice) == 0:
        return False, 0, 0, "No experimental mice found"
    
    first_exp = exp_mice.iloc[0]
    
    # Find matching controls
    controls = full_data[full_data['biological_sample_group'] == 'control']
    
    # Apply metadata filters
    for col in metadata_cols:
        if col in controls.columns and col in exp_mice.columns:
            controls = controls[controls[col] == first_exp[col]]
    
    # Filter by sex
    controls = controls[controls['sex'] == sex]
    
    # Check for complete ABR data in controls
    controls_complete = controls.dropna(subset=freq_cols)
    n_controls_complete = len(controls_complete)
    
    # Check if criteria are met
    meets_criteria = n_exp_complete >= 3 and n_controls_complete >= 20
    
    reason = ""
    if n_exp_complete < 3:
        reason = f"Insufficient experimental mice with complete data ({n_exp_complete} < 3)"
    elif n_controls_complete < 20:
        reason = f"Insufficient controls with complete data ({n_controls_complete} < 20)"
    
    return meets_criteria, n_exp_complete, n_controls_complete, reason

# Check each missing case
print("\n" + "="*80)
print("CHECKING MISSING MALE BAYES FACTORS")
print("="*80)

potentially_runnable_male = []

for idx, row in missing_male.iterrows():
    feasible, n_exp, n_ctrl, reason = check_sex_specific_feasibility(row, 'male', full_data, freq_cols)
    
    if feasible:
        potentially_runnable_male.append({
            'gene_symbol': row['gene_symbol'],
            'allele_symbol': row['allele_symbol'],
            'zygosity': row['zygosity'],
            'center': row['center'],
            'n_male_exp_complete': n_exp,
            'n_male_ctrl_complete': n_ctrl
        })
        print(f"\n⚠️  COULD HAVE BEEN RUN: {row['gene_symbol']} - {row['allele_symbol']} - {row['zygosity']} - {row['center']}")
        print(f"   Male experimental mice with complete data: {n_exp}")
        print(f"   Male controls with complete data: {n_ctrl}")

print(f"\nTotal potentially runnable male analyses: {len(potentially_runnable_male)}")

print("\n" + "="*80)
print("CHECKING MISSING FEMALE BAYES FACTORS")
print("="*80)

potentially_runnable_female = []

for idx, row in missing_female.iterrows():
    feasible, n_exp, n_ctrl, reason = check_sex_specific_feasibility(row, 'female', full_data, freq_cols)
    
    if feasible:
        potentially_runnable_female.append({
            'gene_symbol': row['gene_symbol'],
            'allele_symbol': row['allele_symbol'],
            'zygosity': row['zygosity'],
            'center': row['center'],
            'n_female_exp_complete': n_exp,
            'n_female_ctrl_complete': n_ctrl
        })
        print(f"\n⚠️  COULD HAVE BEEN RUN: {row['gene_symbol']} - {row['allele_symbol']} - {row['zygosity']} - {row['center']}")
        print(f"   Female experimental mice with complete data: {n_exp}")
        print(f"   Female controls with complete data: {n_ctrl}")

print(f"\nTotal potentially runnable female analyses: {len(potentially_runnable_female)}")

# Save results
if potentially_runnable_male or potentially_runnable_female:
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    if potentially_runnable_male:
        male_df = pd.DataFrame(potentially_runnable_male)
        male_df.to_csv('/home/liamb/impc-abr/results/missing_male_analyses.csv', index=False)
        print(f"Saved {len(male_df)} potentially runnable male analyses to missing_male_analyses.csv")
    
    if potentially_runnable_female:
        female_df = pd.DataFrame(potentially_runnable_female)
        female_df.to_csv('/home/liamb/impc-abr/results/missing_female_analyses.csv', index=False)
        print(f"Saved {len(female_df)} potentially runnable female analyses to missing_female_analyses.csv")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Missing male Bayes factors that could have been run: {len(potentially_runnable_male)}")
print(f"Missing female Bayes factors that could have been run: {len(potentially_runnable_female)}")