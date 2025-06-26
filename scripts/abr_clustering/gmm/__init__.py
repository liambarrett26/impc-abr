#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ABR Clustering Package

This package provides tools for analyzing patterns in audiogram data
from the International Mouse Phenotyping Consortium (IMPC) using
dimensionality reduction and clustering techniques.

Main modules:
- dimensionality: Tools for reducing audiogram data dimensionality
- clustering: Algorithms for clustering audiograms by pattern
- visualization: Functions for visualizing audiogram patterns
- metrics: Metrics for evaluating clusters and audiogram shapes
- utils: Utility functions for preprocessing and analysis

author: Liam Barrett
version: 1.0.0
"""

# Import key classes for easier access
from .dimensionality.pca import AudiogramPCA

# Package metadata
__version__ = '1.0.0'