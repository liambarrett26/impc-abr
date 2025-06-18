# ContrastiveVAE-DEC: Deep Learning for Audiometric Phenotype Discovery

A novel deep learning clustering approach for discovering audiometric phenotypes in mouse genetic data from the International Mouse Phenotyping Consortium (IMPC).

## Project Overview

This project implements ContrastiveVAE-DEC (Contrastive Variational Deep Embedded Clustering), which combines:
- Variational Autoencoder (VAE) for robust latent representations
- Deep Embedded Clustering (DEC) for joint representation learning and clustering
- Contrastive Learning for better separation of phenotypes
- Domain-specific adaptations for audiometric data

### Key Innovation
The model learns to cluster mice by hearing patterns rather than just genetic similarity, potentially discovering subtypes within known hearing loss genes and identifying novel phenotypes that traditional univariate analysis misses.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup
```bash
# Clone the repository (if needed)
cd /path/to/audiometric_deep_clustering

# Install package in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

### 1. Unit Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_models.py -v
python -m pytest tests/test_data.py -v
python -m pytest tests/test_losses.py -v
```

### 2. Test Training Run (Fast)
```bash
# Complete 3-stage pipeline with minimal epochs (~2 minutes)
python scripts/train.py --config config/test_config.yaml --experiment-name "test_run"

# Individual stages
python scripts/train.py --experiment-name "test_pretrain" --epochs 3 --pretrain-only
python scripts/train.py --experiment-name "test_joint" --epochs 2 --joint-only
```

### 3. Full-Scale Training, Evaluation & Visualization
```bash
# Complete training pipeline (several hours)
python scripts/train.py --experiment-name "full_scale_training"

# Comprehensive evaluation
python scripts/evaluate.py \
    --checkpoint experiments/full_scale_training/{run_id}/checkpoints/joint/best_joint.pt \
    --save-embeddings --save-predictions

# Generate all visualizations
python scripts/visualize_results.py \
    --embeddings experiments/full_scale_training/{run_id}/evaluation/embeddings/latent_embeddings.npz \
    --results experiments/full_scale_training/{run_id}/evaluation/embeddings/cluster_predictions.csv
```

## Architecture

```text
Input (18 features)
↓
Frequency-Aware Encoder (with attention)
↓
Variational Bottleneck (μ, σ)
↓
Latent Space (z)
├─→ Contrastive Loss
├─→ Clustering Layer (DEC)
└─→ Decoder
↓
Reconstruction + Multiple Losses
```

## Dataset

- **Size**: 59,145 mice from 6,749 gene knockout lines
- **Features**:
  - 6 ABR thresholds (6, 12, 18, 24, 30 kHz + Click-evoked)
  - 10 metadata features (age, weight, sex, zygosity, etc.)
  - 2 PCA components from ABR data (added during preprocessing)
- **Target**: Discover ~12 distinct audiometric phenotype clusters

## Training Strategy

1. **Stage 1: VAE Pretraining** - Learn latent representations (reconstruction + KL loss)
2. **Stage 2: Cluster Initialization** - K-means on latent space
3. **Stage 3: Joint Optimization** - All objectives (VAE + clustering + contrastive + phenotype consistency)

## Output Organization

All outputs are organized in a unified experiment structure:

```text
experiments/
└── {experiment_name}/
    └── {run_id}/
        ├── experiment_metadata.json      # Complete experiment tracking
        ├── config/                       # Saved configurations
        ├── logs/                        # Training logs
        ├── checkpoints/                 # Model checkpoints
        │   ├── pretrain/               # VAE pretraining checkpoints
        │   └── joint/                  # Joint training checkpoints
        ├── metrics/                     # Training metrics
        ├── evaluation/                  # Evaluation outputs
        │   ├── embeddings/             # Latent embeddings and predictions
        │   ├── metrics/                # Clustering quality metrics
        │   ├── analysis/               # Phenotype and gene analysis
        │   ├── visualizations/         # Evaluation plots
        │   └── logs/                   # Evaluation logs
        └── visualizations/             # Additional visualization outputs
```

## Key Configuration

### Model Parameters
- **Latent dimensions**: 10
- **Hidden dimensions**: [64, 32, 16]
- **Number of clusters**: 12
- **Attention heads**: 2

### Training Parameters
- **Learning rate**: 1e-3 with cosine annealing
- **Optimizer**: Adam
- **Default epochs**: 50 pretrain + 150 joint
- **Batch size**: 512

## Dependencies

Core dependencies are automatically installed with:
```bash
pip install -e .
```

Main packages:
- PyTorch >= 2.0
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn, Plotly
- PyYAML, UMAP

## Project Structure

```text
audiometric_deep_clustering/
├── config/                 # Configuration files
├── data/                   # Data processing and loading
├── models/                 # Neural network architectures
├── losses/                 # Loss functions and objectives
├── training/               # Training loops and optimization
├── evaluation/             # Evaluation metrics and analysis
├── utils/                  # Utilities (logging, checkpoints, etc.)
├── scripts/                # Command-line interfaces
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
├── setup.py               # Package installation
└── README.md              # This file
```

## Contact
This project is part of audiometric phenotype discovery research for the IMPC dataset.
