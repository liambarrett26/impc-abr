# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IMPC ABR (International Mouse Phenotyping Consortium - Auditory Brainstem Response) is a scientific research codebase for analyzing hearing phenotypes in knockout mice. The project implements novel multivariate statistical approaches to identify genes associated with hearing loss by treating ABR profiles as unified multivariate observations rather than analyzing individual frequency measurements in isolation.

## Development Environment

### Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate impc_abr

# Install packages in development mode
cd scripts
pip install -e .
```

### Key Dependencies
- PyMC v5.20.0 - Bayesian modeling
- scikit-learn v1.6.1 - Machine learning
- pandas/numpy - Data manipulation
- matplotlib/seaborn - Visualization
- arviz - Bayesian diagnostics

## Core Commands

### Bayesian Analysis
```bash
# Run parallel Bayesian analysis on ABR data
python scripts/run_parallel_analysis.py --data path/to/abr_data.csv --processes 8

# With custom parameters
python scripts/run_parallel_analysis.py --data path/to/abr_data.csv --min-bf 5 --batch-size 20 --reference-genes data/confirmed_genes.txt
```

### Clustering Analysis
```bash
# Run GMM clustering pipeline
python scripts/abr_clustering/pipeline.py data.csv --output-dir results/clustering/

# With specific configuration
python scripts/abr_clustering/pipeline.py data.csv --n-components 5 --random-state 42
```

### Data Extraction
```bash
# Extract ABR data from IMPC database
python scripts/abr_extraction/fetch_abr_data.py --output data/raw/
```

### Testing
```bash
# Run tests (using pytest)
cd scripts
pytest abr_analysis/tests/

# Run specific test file
pytest abr_analysis/tests/test_batch_processor.py

# Run example gene tests
pytest abr_analysis/tests/example_genes/
```

## Code Architecture

### Package Structure
- **abr_analysis/**: Core Bayesian analysis framework
  - `models/bayesian.py`: PyMC model implementations
  - `analysis/parallel_executor.py`: Parallel processing orchestration
  - `data/loader.py`: Data loading and validation
  - `data/matcher.py`: Control matching algorithms
  - `visualization/`: Plotting utilities for results

- **abr_clustering/**: Unsupervised learning pipeline
  - `pipeline.py`: Main pipeline orchestrator
  - `gmm.py`: Gaussian Mixture Model implementation
  - `analysis.py`: Result analysis and phenotype characterization
  - `loader.py`: IMPC data loading utilities
  - `preproc.py`: Data preprocessing and normalization

- **abr_extraction/**: Data acquisition utilities

### Key Design Patterns

1. **Parallel Processing**: The Bayesian analysis uses multiprocessing to analyze multiple genes concurrently. Each gene analysis runs independently with its own model fitting.

2. **Model Specification**: Bayesian models are defined using PyMC's probabilistic programming interface. Models account for:
   - Technical center effects
   - Sex-specific variations
   - Hierarchical structure of the data
   - Multiple frequency measurements as multivariate outcomes

3. **Control Matching**: Implements sophisticated algorithms to match knockout mice with appropriate wild-type controls based on:
   - Testing center
   - Time period
   - Background strain
   - Sex (when performing sex-specific analyses)

4. **Result Persistence**: All analysis results are saved with comprehensive metadata:
   - Model specifications as JSON
   - MCMC traces as NetCDF files
   - Visualizations as PNG/PDF
   - Summary statistics as CSV

### Analysis Workflow

1. **Data Loading**: Raw ABR data is loaded and validated, ensuring required columns are present
2. **Control Matching**: For each knockout group, appropriate controls are selected
3. **Model Fitting**: Bayesian models are fit using MCMC sampling (typically 4 chains, 2000+ samples)
4. **Evidence Calculation**: Bayes factors are computed to quantify evidence for hearing loss
5. **Visualization**: Comprehensive plots including ABR profiles, posterior distributions, and forest plots
6. **Reporting**: Human-readable summaries with statistical interpretations

## Important Considerations

- **Memory Usage**: Bayesian MCMC sampling can be memory-intensive. Monitor RAM usage when running parallel analyses.
- **Computation Time**: Full dataset analysis may take several hours depending on the number of genes and CPU cores.
- **File Outputs**: Each gene analysis creates its own subdirectory with all results. Ensure adequate disk space.
- **Reproducibility**: Always set random seeds for consistent results across runs.
- **Sex-Specific Analysis**: The pipeline can perform combined or sex-specific analyses. Sex-specific analyses require sufficient sample sizes per group.