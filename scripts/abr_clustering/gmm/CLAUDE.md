# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a GMM (Gaussian Mixture Model) clustering pipeline for discovering audiometric phenotypes in IMPC (International Mouse Phenotyping Consortium) mice data. The system clusters mice based on their ABR (Auditory Brainstem Response) thresholds across 5 frequencies (6, 12, 18, 24, 30 kHz) to identify distinct hearing patterns associated with genetic knockouts.

## Commands

### Running the Complete Pipeline
```bash
python pipeline.py <data_path> [options]

# Common options:
--output-dir results       # Output directory for results
--min-mutants 3           # Minimum mutant mice per experimental group
--min-controls 20         # Minimum control mice per group
--min-components 3        # Min GMM components to test
--max-components 12       # Max GMM components to test
--n-bootstrap 100         # Bootstrap iterations for stability
--log-level INFO          # Logging verbosity
```

### Running Individual Components
```bash
# Test data loading
python loader.py <data_path>

# Test preprocessing
python preproc.py <data_path>

# Test GMM clustering
python gmm.py <data_path>

# Test analysis
python analysis.py <data_path>
```

## Architecture

### Data Flow
1. **Raw IMPC Data** → `loader.py` filters and creates experimental groups
2. **Filtered Data** → `preproc.py` applies two-stage normalization
3. **Normalized Data** → `gmm.py` performs model selection and clustering
4. **Cluster Results** → `analysis.py` characterizes patterns and gene associations
5. **Pipeline** → `pipeline.py` orchestrates the complete workflow

### Key Technical Decisions

**Two-Stage Normalization Strategy:**
- Stage 1: Z-score normalization within technical groups (center/equipment) to correct batch effects
- Stage 2: Global min-max scaling to [0,1] for equal feature weighting across frequencies

**GMM Model Selection:**
- Tests 3-12 components with full/tied covariance structures
- Multiple initialization strategies: random, k-means++, PCA-informed
- Combined selection criteria: BIC (40%), AIC (20%), silhouette (20%), stability (20%)
- Bootstrap stability assessment with 80% subsampling

**Audiometric Pattern Classification:**
The system identifies patterns based on normalized profile shapes:
- Flat patterns (normal, moderate, severe)
- High-frequency loss (increasing thresholds at higher frequencies)
- Low-frequency loss (decreasing thresholds at higher frequencies)
- Cookie-bite (U-shaped) and reverse cookie-bite patterns

### Critical Implementation Details

**Data Requirements:**
- Complete ABR measurements for all 5 frequencies
- Physiologically plausible thresholds (0-100 dB SPL)
- Age filter: 10-20 weeks
- Minimum group sizes enforced for statistical validity

**Technical Group Handling:**
- Groups formed by: phenotyping_center + pipeline_name + equipment_manufacturer + equipment_model
- Small groups (<5 mice) merged into 'other_combined' category
- Fallback scalers for unseen technical groups during prediction

**Output Structure:**
```
results/
├── pipeline_config.json          # Complete configuration
├── analysis_results.json         # Clustering metrics and statistics
├── cluster_assignments.csv       # Sample-level assignments with probabilities
├── normalized_data.csv          # Preprocessed features
├── clustering_report.txt        # Human-readable summary
├── cluster_audiogram_profiles.png
├── cluster_pca_visualization.png
├── cluster_distributions.png
├── assignment_uncertainty.png
├── preprocessor.pkl             # Fitted preprocessor model
└── gmm_model.pkl               # Fitted GMM model
```

## Important Considerations

- The pipeline expects IMPC ABR data format with specific column names for thresholds and metadata
- Memory usage scales with data size and number of GMM components tested
- Bootstrap stability assessment is computationally intensive (controlled by --n-bootstrap)
- Models are saved by default; use --no-save-models to skip model persistence
- All randomness is controlled via --random-state for reproducibility