# GMM Clustering for Audiometric Phenotype Discovery

This module is part of the IMPC ABR analysis pipeline and implements Gaussian Mixture Model (GMM) clustering to discover distinct audiometric phenotypes in mouse hearing data.

## Overview

The GMM clustering approach treats each mouse's ABR (Auditory Brainstem Response) profile as a multivariate observation across 5 frequencies (6, 12, 18, 24, 30 kHz), identifying groups of mice with similar hearing patterns. This unsupervised approach can reveal novel phenotypes beyond simple "hearing loss" vs "normal hearing" classifications.

## What It Does

1. **Data Loading**: Filters IMPC ABR data for quality and completeness
2. **Preprocessing**: Applies two-stage normalization to correct for technical batch effects
3. **Clustering**: Uses GMM to identify distinct audiometric patterns
4. **Analysis**: Characterizes discovered phenotypes and their gene associations
5. **Visualization**: Generates comprehensive plots and reports

## How It Works

### Data Preprocessing
- **Stage 1**: Z-score normalization within technical groups (phenotyping center + equipment) to remove batch effects
- **Stage 2**: Global min-max scaling to [0,1] range for equal feature weighting across frequencies

### Model Selection
- Tests multiple configurations: k=3-12 components with full/tied covariance structures
- Evaluates models using multiple criteria:
  - BIC (Bayesian Information Criterion) - 40% weight
  - AIC (Akaike Information Criterion) - 20% weight
  - Silhouette coefficient - 20% weight
  - Bootstrap stability - 20% weight
- Selects optimal model based on weighted combination of metrics

### Pattern Classification
The system identifies audiometric patterns such as:
- Flat patterns (normal, moderate, severe)
- High-frequency loss
- Low-frequency loss
- Cookie-bite (U-shaped) patterns
- Mixed patterns

## Usage

### Serial Execution (Original Pipeline)

For single-threaded execution with all models trained sequentially:

```bash
python pipeline.py /home/liamb/impc-abr/data/processed/abr_full_data.csv \
    --output-dir results_serial \
    --min-components 3 \
    --max-components 12 \
    --n-bootstrap 100 \
    --log-level INFO
```

### Parallel Execution (Recommended for Large Datasets)

For parallel execution across multiple CPU cores:

```bash
./run_parallel_gmm.sh -d /home/liamb/impc-abr/data/processed/abr_full_data.csv \
    --output-dir results_parallel \
    --min-k 3 \
    --max-k 12 \
    --max-jobs 8 \
    --n-bootstrap 100
```

## Parallel Pipeline Architecture

The parallel implementation consists of:

1. **Preprocessing Phase**: Run once, saves normalized data to shared directory
2. **Parallel Training Phase**: Each k × covariance combination trained independently
3. **Aggregation Phase**: Collects all results, performs model selection, generates final report

### Key Scripts

- `pipeline.py`: Original serial pipeline (all models in sequence)
- `pipeline_parallel.py`: Trains single model configuration (called by bash script)
- `run_parallel_gmm.sh`: Orchestrates parallel execution
- `preprocess_data.py`: Shared preprocessing step
- `aggregate_results.py`: Combines results from parallel runs

## Output Structure

```
results/
├── pipeline_config.json          # Configuration parameters
├── model_comparison.csv          # All model metrics (parallel only)
├── best_model.pkl               # Selected GMM model
├── best_cluster_labels.npy      # Final cluster assignments
├── cluster_audiogram_profiles.png   # Audiometric patterns by cluster
├── cluster_pca_visualization.png    # 2D visualization of clusters
├── model_comparison_metrics.png     # Model selection visualization (parallel only)
├── clustering_report.txt            # Human-readable summary
└── gmm_k*_*/                       # Individual model results (parallel only)
    ├── model.pkl
    ├── metrics.json
    └── analysis_results.json
```

## Why GMM?

Gaussian Mixture Models are particularly suitable for audiometric data because:

1. **Soft clustering**: Provides probability of cluster membership, capturing uncertainty
2. **Flexible shapes**: Different covariance structures can model various data distributions
3. **Statistical foundation**: Based on probabilistic model with well-understood properties
4. **Interpretability**: Cluster centers directly correspond to audiometric profiles

## Requirements

- Python 3.7+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (for visualizations)
- 8+ GB RAM for full dataset
- Multiple CPU cores recommended for parallel execution

## Notes

- This module is part of the larger IMPC ABR analysis project
- Input data must be in IMPC format with specific column names
- Parallel execution is recommended for datasets with >1000 samples
- Bootstrap stability assessment is computationally intensive; adjust `--n-bootstrap` based on available resources

## Troubleshooting

- **Memory errors**: Reduce `--max-k` or use fewer bootstrap iterations
- **Incomplete models**: Check individual model logs in `results/gmm_k*/`
- **Preprocessing failures**: Ensure input data has all required ABR threshold columns