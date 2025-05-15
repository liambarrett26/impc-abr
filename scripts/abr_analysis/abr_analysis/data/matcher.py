#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ABR Control Matcher Module

This module provides functionality for matching knockout mice with appropriate 
wild-type controls for Auditory Brainstem Response (ABR) analysis. It ensures 
valid statistical comparisons by matching experimental and control groups on 
important metadata factors that could influence ABR measurements.

The ControlMatcher class implements a robust approach to identify suitable 
control mice that match experimental mice on critical factors like facility, 
genetic background, and equipment.

author: Liam Barrett
version: 1.0.1
"""

class ControlMatcher:
    """
    Match knockout mice with appropriate wild-type controls based on metadata.
    
    This class handles the critical task of identifying suitable control mice
    for comparison with knockout mice. It ensures controls are matched on
    important experimental factors like center, genetic background, and
    equipment, which is essential for valid comparisons in ABR studies.
    
    Attributes:
        data (pandas.DataFrame): The complete dataset containing both control 
            and experimental mice.
        metadata_cols (list): Columns used for matching controls to knockouts.
    """

    def __init__(self, data):
        """
        Initialize the ControlMatcher with a dataset.
        
        Parameters:
            data (pandas.DataFrame): The complete dataset containing both control 
                and experimental mice with metadata columns.
        """
        self.data = data
        # Define columns that should match between knockouts and controls
        self.metadata_cols = [
            'phenotyping_center',      # Facility where testing was performed
            'genetic_background',      # Mouse strain background
            'pipeline_name',           # Processing pipeline used
            'metadata_Equipment manufacturer',  # ABR equipment manufacturer
            'metadata_Equipment model'          # ABR equipment model
        ]

        # Add sex as a separate column for sex-specific matching
        self.sex_col = 'sex'

    def find_experimental_groups(self, gene_symbol):
        """
        Find all experimental groups for a given gene based on allele+zygosity+center.
        
        Parameters:
            gene_symbol (str): The gene symbol to find experimental groups for.
                
        Returns:
            list: List of dictionaries, each containing group identifier and data subset.
        """
        # Get all experimental mice for this gene
        gene_data = self.data[(self.data['gene_symbol'] == gene_symbol) &
                             (self.data['biological_sample_group'] == 'experimental')]

        if len(gene_data) == 0:
            return []

        # Group by allele symbol, zygosity, and phenotyping center
        group_cols = ['allele_symbol', 'zygosity', 'phenotyping_center']
        groups = []

        # Get unique combinations
        unique_groups = gene_data[group_cols].drop_duplicates()

        for _, row in unique_groups.iterrows():
            # Create filter for this specific group
            group_filter = True
            for col in group_cols:
                group_filter = group_filter & (gene_data[col] == row[col])

            # Get data for this group
            group_data = gene_data[group_filter]

            # Only include group if it has enough samples
            if len(group_data) >= 3:  # Minimum 3 experimental mice
                group_info = {
                    'gene_symbol': gene_symbol,
                    'allele_symbol': row['allele_symbol'],
                    'zygosity': row['zygosity'],
                    'phenotyping_center': row['phenotyping_center'],
                    'data': group_data,
                    'metadata': self._extract_metadata(group_data)
                }
                groups.append(group_info)

        return groups

    def _extract_metadata(self, group_data):
        """
        Extract metadata for control matching from a group of experimental mice.
        
        Parameters:
            group_data (pandas.DataFrame): Data for a group of experimental mice.
                
        Returns:
            dict: Metadata values for control matching.
        """
        # Use the first row to extract metadata (values should be consistent within group)
        first_row = group_data.iloc[0]
        metadata = {}

        for col in self.metadata_cols:
            metadata[col] = first_row[col]

        return metadata

    def find_matching_controls(self, group_info, min_controls=20):
        """
        Find matching controls based on experimental group metadata.
        
        Parameters:
            group_info (dict): Dictionary containing experimental group information.
            min_controls (int, optional): Minimum number of controls required.
                Defaults to 20.
                
        Returns:
            dict: Dictionary containing control data for each sex and combined.
            
        Raises:
            ValueError: If fewer than min_controls matching controls are found.
        """
        # Start with all control mice
        all_controls = self.data[self.data['biological_sample_group'] == 'control']

        # Apply each metadata constraint sequentially
        filtered_controls = all_controls.copy()
        for col in self.metadata_cols:
            filtered_controls = filtered_controls[filtered_controls[col] == group_info['metadata'][col]]

        # Ensure sufficient controls for statistical power
        if len(filtered_controls) < min_controls:
            raise ValueError(
                f"Insufficient controls found: {len(filtered_controls)} < {min_controls}"
            )

        # Split controls by sex for sex-specific analyses
        male_controls = filtered_controls[filtered_controls[self.sex_col] == 'male']
        female_controls = filtered_controls[filtered_controls[self.sex_col] == 'female']

        return {
            'all': filtered_controls,
            'male': male_controls,
            'female': female_controls
        }

    def get_control_profiles(self, controls, freq_cols):
        """
        Extract ABR profiles from control data.
        
        Converts the raw ABR threshold data into a numpy array suitable
        for statistical analysis.
        
        Parameters:
            controls (pandas.DataFrame): Control mice data with ABR thresholds.
            freq_cols (list): Column names for ABR frequency thresholds.
            
        Returns:
            numpy.ndarray: Array of shape (n_controls, n_frequencies) containing
                the ABR thresholds for each control mouse.
        """
        # Extract and convert to numerical array
        return controls[freq_cols].values.astype(float)

    def get_experimental_profiles(self, group_info, freq_cols, sex=None):
        """
        Extract ABR profiles from experimental data, optionally filtered by sex.
        
        Parameters:
            group_info (dict): Dictionary containing experimental group information.
            freq_cols (list): Column names for ABR frequency thresholds.
            sex (str, optional): Filter by sex ('male', 'female', or None for all).
            
        Returns:
            numpy.ndarray: Array of ABR thresholds for experimental mice.
        """
        data = group_info['data']

        if sex is not None:
            data = data[data[self.sex_col] == sex]

        return data[freq_cols].values.astype(float)
