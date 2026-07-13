#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ABR data loading utilities.

This module loads Auditory Brainstem Response (ABR) data exported from the
International Mouse Phenotyping Consortium (IMPC) and exposes helpers for
extracting the per-frequency threshold columns and the metadata fields used to
match knockouts to wild-type controls.

author: Liam Barrett
version: 1.0.1
"""

from pathlib import Path

import pandas as pd


class ABRDataLoader:
    """Load and preprocess ABR data from the IMPC.

    Attributes:
        data_path (pathlib.Path): Path to the ABR data CSV file.
        data (pandas.DataFrame | None): The loaded dataset, or ``None`` until
            :meth:`load_data` is called.
        metadata_cols (list[str]): Columns describing experimental conditions,
            used downstream for control matching.
    """

    def __init__(self, data_path):
        """Initialise the loader.

        Args:
            data_path (str | pathlib.Path): Path to the ABR data CSV file.
        """
        self.data_path = Path(data_path)
        self.data = None
        self.metadata_cols = [
            "phenotyping_center",
            "sex",
            "genetic_background",
            "pipeline_name",
            "metadata_Equipment manufacturer",
            "metadata_Equipment model",
        ]

    def load_data(self):
        """Load the ABR data file into memory.

        Returns:
            pandas.DataFrame: The loaded dataset (also stored on ``self.data``).
        """
        self.data = pd.read_csv(self.data_path, low_memory=False)
        return self.data

    def get_frequencies(self):
        """Return the ABR threshold frequency columns present in the dataset.

        Returns:
            list[str]: The 6, 12, 18, 24 and 30 kHz threshold column names that
            are actually present in the loaded data, in ascending order.
        """
        freq_cols = [
            "6kHz-evoked ABR Threshold",
            "12kHz-evoked ABR Threshold",
            "18kHz-evoked ABR Threshold",
            "24kHz-evoked ABR Threshold",
            "30kHz-evoked ABR Threshold",
        ]
        return [col for col in freq_cols if col in self.data.columns]

    def get_abr_profile(self, row):
        """Extract the 5-frequency ABR profile from a single row.

        Args:
            row (pandas.Series): A row from the loaded dataset.

        Returns:
            numpy.ndarray: The frequency thresholds as a float array.
        """
        freq_cols = self.get_frequencies()
        return row[freq_cols].values.astype(float)

    def get_metadata(self, row):
        """Extract the control-matching metadata from a single row.

        Args:
            row (pandas.Series): A row from the loaded dataset.

        Returns:
            dict[str, object]: Mapping of each metadata column to its value.
        """
        return {col: row[col] for col in self.metadata_cols}
