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
version: 1.0.0
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
    
    def find_matching_controls(self, ko_metadata, min_controls=20):
        """
        Find matching controls based on knockout metadata.
        
        Identifies control mice that match the experimental mice on all
        specified metadata columns. This ensures that differences in ABR
        measurements are due to genetic differences rather than experimental
        conditions.
        
        Parameters:
            ko_metadata (dict): Dictionary containing metadata values for the 
                knockout mice, with keys matching self.metadata_cols.
            min_controls (int, optional): Minimum number of controls required.
                Defaults to 20.
                
        Returns:
            pandas.DataFrame: Subset of control mice matching the metadata.
            
        Raises:
            ValueError: If fewer than min_controls matching controls are found.
        """
        # Start with all control mice
        controls = self.data[self.data['biological_sample_group'] == 'control']
        
        # Apply each metadata constraint sequentially
        for col in self.metadata_cols:
            controls = controls[controls[col] == ko_metadata[col]]
            
        # Ensure sufficient controls for statistical power
        if len(controls) < min_controls:
            raise ValueError(
                f"Insufficient controls found: {len(controls)} < {min_controls}"
            )
            
        return controls
    
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