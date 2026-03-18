# IMPC ABR: Multivariate Analysis of Auditory Brainstem Response Data

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyMC](https://img.shields.io/badge/PyMC-5.20-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

Codebase for the multivariate, Bayesian, and clustering analysis of auditory brainstem response (ABR) data from the [International Mouse Phenotyping Consortium](https://www.mousephenotype.org/) (IMPC).

## Background

Hearing loss is the most common sensory deficit globally, with up to 80% of prelingual cases having a genetic basis. The IMPC provides an unparalleled resource for auditory gene discovery, with ABR data available for over 6,700 gene knockouts across 59,000+ individual mice. However, the current IMPC analytical pipeline applies univariate, frequency-by-frequency statistical testing, which limits sensitivity to coordinated threshold shifts and can produce both false positives and false negatives.

This project develops complementary statistical approaches that treat each mouse's ABR profile as a unified multivariate observation (thresholds at 6, 12, 18, 24, and 30 kHz) rather than analysing each frequency in isolation.

## Analytical Approaches

### 1. Multivariate Distribution Analysis

Each audiogram is treated as a 5-dimensional observation. Wild-type audiogram distributions are modelled per matched control group, and the Mahalanobis distance of each mutant audiogram from this distribution is calculated. Significance is assessed via chi-squared testing with FDR correction. This captures coordinated threshold shifts across the hearing spectrum that frequency-independent analyses miss.

### 2. Bayesian Mixture Model

A Bayesian mixture model that encodes the prior expectation that most gene knockouts do not affect hearing ($\beta_{(1, 3)}$ prior). Hearing loss is modelled as a mixture of normal and affected populations, estimating both the probability and magnitude of hearing loss for each gene. Evidence is quantified using Bayes factors, providing continuous evidence measures rather than binary significance thresholds. Implemented using PyMC with MCMC sampling.

### 3. Gaussian Mixture Model Clustering

Unsupervised clustering of ABR profiles using Gaussian Mixture Models to identify data-driven audiometric phenotypes. Models are evaluated across 3-12 clusters with both full and tied covariance structures, using BIC, AIC, silhouette scores, and bootstrap stability (Adjusted Rand Index) for model selection. Cluster assignments are linked back to Bayesian gene-level evidence to characterise the hearing loss profile associated with each gene.

## Repository Structure

```
impc-abr/
├── data/
│   ├── processed/          # Processed ABR datasets
│   └── raw/                # Raw data examples
├── scripts/
│   ├── abr_extraction/     # Data retrieval from IMPC Solr API
│   ├── abr_analysis/       # Bayesian and multivariate analysis
│   │   ├── abr_analysis/
│   │   │   ├── data/       # Data loading and control matching
│   │   │   ├── models/     # Bayesian and distribution models
│   │   │   └── analysis/   # Batch processing and parallel execution
│   │   └── tests/          # Gene-specific test cases
│   └── abr_clustering/     # GMM clustering
│       ├── gmm/            # GMM pipeline, analysis, and visualisation
│       ├── clustering/     # Core GMM implementation
│       ├── dimensionality/ # PCA for audiogram feature reduction
│       └── utils/          # Data loading utilities
├── docs/                   # Project documentation
├── environment.yml         # Conda environment specification
└── README.md
```

### Key Modules

| Module | Description |
|---|---|
| `abr_analysis.data.loader` | Loads IMPC ABR data, extracts frequency columns and metadata |
| `abr_analysis.data.matcher` | Matches knockouts to wild-type controls by facility, genetic background, and equipment |
| `abr_analysis.models.bayesian` | PyMC Bayesian mixture model for hearing loss evidence quantification |
| `abr_analysis.models.distribution` | Robust multivariate Gaussian fitting for audiogram distributions |
| `abr_analysis.analysis.batch_bayes_processor` | Batch Bayesian analysis across all genes with sex-specific stratification |
| `abr_analysis.analysis.parallel_executor` | Parallelised batch processing with intermediate result saving |
| `abr_clustering.gmm.gmm` | GMM fitting with stability assessment and model validation |
| `abr_clustering.gmm.pipeline_parallel` | Parallel GMM training across cluster/covariance configurations |
| `abr_clustering.gmm.analyze_gene_cluster_associations` | Links GMM clusters to Bayesian hearing loss evidence per gene |

## Installation

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/liambarrett26/impc-abr.git
cd impc-abr

# Create conda environment
conda env create -f environment.yml
conda activate impc_abr

# Install analysis and clustering packages
pip install -e scripts/abr_analysis
pip install -e scripts/
```

### Requirements

- **Python** >= 3.9
- **PyMC** 5.20 (Bayesian modelling)
- **ArviZ** 0.20 (posterior analysis)
- **scikit-learn** 1.6 (GMM clustering)
- **NumPy** 1.26, **pandas** 2.2, **SciPy** 1.15
- **matplotlib** 3.10, **seaborn** 0.13 (visualisation)

See `environment.yml` for the full dependency specification.

## Data

ABR data is sourced from the [IMPC Data Portal](https://www.mousephenotype.org/). The dataset comprises ABR thresholds at five frequencies (6, 12, 18, 24, 30 kHz) collected from 11 phenotyping centres worldwide, with all centres following standardised operating procedures.

Control mice are matched to experimental animals on phenotyping centre, genetic background, pipeline, and equipment specifications. Inclusion criteria require a minimum of 3 mutant mice per line, at least 20 matched controls, and complete data at all five frequencies.

## Usage

### Bayesian Analysis

```bash
# Run parallelised Bayesian analysis across all genes
python scripts/run_parallel_analysis.py --data data/processed/abr_full_data.csv --output results/
```

### GMM Clustering

```bash
# Preprocess data and run parallel GMM across k=3..12 with full and tied covariance
cd scripts/abr_clustering/gmm
bash run_parallel_gmm.sh /path/to/abr_full_data.csv
```

## Citation

Upcoming. In the meantime please cite this GitHub Repo.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

This work uses data generated by the [International Mouse Phenotyping Consortium](https://www.mousephenotype.org/). We thank the IMPC and its member institutions for making phenotyping data publicly available.
