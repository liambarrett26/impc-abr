import pandas as pd
import numpy as np
from pathlib import Path

class ABRDataLoader:
    """Load and preprocess ABR data from IMPC."""
    
    def __init__(self, data_path):
        """Initialize with path to ABR data file."""
        self.data_path = Path(data_path)
        self.data = None
        self.metadata_cols = [
            'phenotyping_center', 
            'sex', 
            'genetic_background',
            'pipeline_name',
            'metadata_Equipment manufacturer',
            'metadata_Equipment model'
        ]
        
    def load_data(self):
        """Load the ABR data file."""
        self.data = pd.read_csv(self.data_path, low_memory=False)
        return self.data
    
    def get_frequencies(self):
        """Return the frequency columns in the dataset."""
        freq_cols = [
            '6kHz-evoked ABR Threshold',
            '12kHz-evoked ABR Threshold',
            '18kHz-evoked ABR Threshold',
            '24kHz-evoked ABR Threshold',
            '30kHz-evoked ABR Threshold'
        ]
        return [col for col in freq_cols if col in self.data.columns]
    
    def get_abr_profile(self, row):
        """Extract ABR profile as numpy array from a row."""
        freq_cols = self.get_frequencies()
        return row[freq_cols].values.astype(float)
    
    def get_metadata(self, row):
        """Extract metadata from a row."""
        return {col: row[col] for col in self.metadata_cols}