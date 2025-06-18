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

## Architecture

```text
nput (16-18 features)
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

## Project Structure

```text
audiometric_deep_clustering/
│
├── config/                     ✅ COMPLETE
│   ├── init.py
│   ├── model_config.yaml       # Model hyperparameters
│   └── training_config.yaml    # Training settings
│
├── data/                       ✅ COMPLETE
│   ├── init.py
│   ├── dataset.py              # PyTorch Dataset for audiometric data
│   ├── augmentations.py        # Data augmentations for contrastive learning
│   ├── preprocessor.py         # Feature preprocessing pipeline
│   └── dataloader.py           # Custom dataloader with sampling strategies
│
├── models/                     ✅ COMPLETE
│   ├── init.py
│   ├── encoder.py              # Frequency-aware encoder with attention
│   ├── decoder.py              # Reconstruction decoder
│   ├── vae.py                  # VAE components
│   ├── clustering_layer.py     # DEC clustering layer
│   ├── attention.py            # Frequency attention module
│   └── full_model.py           # Complete ContrastiveVAE-DEC model
│
├── losses/                     ✅ COMPLETE
│   ├── init.py
│   ├── reconstruction.py       # Reconstruction losses
│   ├── vae_loss.py            # KL divergence and ELBO
│   ├── clustering_loss.py      # DEC and auxiliary clustering losses
│   ├── contrastive.py         # Contrastive learning losses
│   └── combined_loss.py        # Multi-objective loss combination
│
├── training/                   ✅ COMPLETE
│   ├── init.py
│   ├── trainer.py              # Main training loop
│   ├── pretrain.py             # VAE pretraining
│   ├── finetune.py             # Joint clustering fine-tuning
│   └── callbacks.py            # Training callbacks and monitoring
│
├── evaluation/                 🚧 IN PROGRESS
│   ├── init.py
│   ├── metrics.py              ✅ # Clustering metrics
│   ├── visualization.py        ✅ # t-SNE, UMAP, cluster plots
│   ├── phenotype_analysis.py   ❌ # Biological interpretation
│   └── gene_enrichment.py      ❌ # Gene-cluster association analysis
│
├── utils/                      ❌ NOT STARTED
│   ├── init.py
│   ├── seed.py                 # Reproducibility utilities
│   ├── logging.py              # Logging configuration
│   └── checkpoint.py           # Model checkpointing
│
├── scripts/                    ❌ NOT STARTED
│   ├── train.py                # Main training script
│   ├── evaluate.py             # Evaluation script
│   ├── infer.py                # Inference on new data
│   └── visualize_results.py    # Generate all visualizations
│
├── notebooks/                  ❌ NOT STARTED
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_analysis.ipynb
│   └── 03_biological_interpretation.ipynb
│
├── tests/                      ❌ NOT STARTED
│   ├── test_models.py
│   ├── test_losses.py
│   └── test_data.py
│
├── requirements.txt            ❌ NOT STARTED
├── setup.py                    ❌ NOT STARTED
└── README.md                   ✅ THIS FILE
```

## Next Steps for Implementation

### Current Task: `evaluation/phenotype_analysis.py`
This module should provide:
- Hearing loss pattern classification (flat, high-frequency, low-frequency, mixed)
- Clinical severity assessment
- Phenotype-to-gene mapping
- Statistical validation of discovered phenotypes

### Remaining Evaluation Module: `evaluation/gene_enrichment.py`
This module should implement:
- Gene-cluster association analysis
- Statistical enrichment tests
- Gene set analysis for clusters
- Validation against known hearing loss genes

### Utils Module Requirements
- **seed.py**: Reproducibility utilities for PyTorch, NumPy, and random
- **logging.py**: Structured logging configuration with file and console outputs
- **checkpoint.py**: Model saving/loading with state preservation

### Scripts Module Requirements
- **train.py**: CLI interface for full training pipeline
- **evaluate.py**: Post-training evaluation and metric computation
- **infer.py**: Apply trained model to new data
- **visualize_results.py**: Generate publication-ready figures

## Key Implementation Details

### Data Configuration
- **Data path**: `/home/liamb/impc-abr/data/processed/abr_full_data.csv`
- **Feature dimensions**: 18 (6 ABR + 10 metadata + 2 PCA)
- **Batch size**: 512
- **Train/Val/Test split**: 80/10/10

### Model Configuration
- **Latent dimensions**: 10
- **Hidden dimensions**: [64, 32]
- **Number of clusters**: 12
- **Attention heads**: 4

### Training Configuration
- **Learning rate**: 2e-4
- **Optimizer**: AdamW
- **Stage 1 epochs**: 100
- **Stage 2 epochs**: 50
- **Stage 3 epochs**: 200

## Dependencies
- PyTorch >= 2.0
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- PyYAML
- tqdm

## Notes for Continued Implementation

1. **Phenotype Analysis Focus**: The biological interpretation should leverage domain knowledge about hearing loss patterns and clinical thresholds

2. **Gene Enrichment**: Should handle the hierarchical nature of gene-phenotype relationships and account for multiple testing

3. **Consistency**: Maintain the established coding style and integration patterns from completed modules

4. **Testing**: Each new module should include comprehensive unit tests

5. **Documentation**: Continue inline documentation and docstring patterns established in existing code

## Contact
This project is part of audiometric phenotype discovery research for the IMPC dataset.

---

Next Steps for Trialing the Codebase

  1. Environment Setup (Required Dependencies)

  # Install missing dependencies
  pip install scikit-learn plotly PyYAML umap-learn statsmodels

  # Optional: Install in development mode
  pip install -e .

  2. Data Preparation

  # Ensure your ABR data is in the expected format:
  # CSV with columns: specimen_id, gene_symbol, abr_6kHz, abr_12kHz, etc.
  # Update data paths in config/training_config.yaml

  3. Quick Verification Test

  # Run basic import test
  python -c "
  import sys; sys.path.append('.')
  from models.full_model import create_model
  from data.dataset import create_abr_dataset
  print('✓ Core imports working')
  "

  # Run unit tests
  python -m pytest tests/ -v

  4. Training Pipeline Trial

  # Stage 1: VAE Pretraining (5-10 epochs)
  python scripts/train.py --config config/training_config.yaml --stage pretrain --epochs 10

  # Stage 2: Cluster Initialization
  python scripts/train.py --config config/training_config.yaml --stage initialize --resume

  # Stage 3: Joint Training (20-50 epochs)
  python scripts/train.py --config config/training_config.yaml --stage joint --epochs 30

  5. Evaluation and Analysis

  # Comprehensive evaluation
  python scripts/evaluate.py --checkpoint checkpoints/best_model.ckpt --data-path data/test_data.csv

  # Generate visualizations
  python scripts/visualize_results.py --checkpoint checkpoints/best_model.ckpt --output-dir results/

  # Inference on new data
  python scripts/infer.py --checkpoint checkpoints/best_model.ckpt --input data/new_samples.csv

  6. Configuration Adjustments

  - Model size: Adjust hidden_dims in model_config.yaml for your dataset size
  - Cluster count: Set num_clusters based on expected phenotype diversity
  - Loss weights: Fine-tune in training_config.yaml based on initial results
  - Data paths: Update file paths to match your data location

  7. Monitoring and Debugging

  - Check logs in logs/ directory for training progress
  - Use --debug flag for verbose output
  - Monitor GPU usage and adjust batch sizes accordingly
  - Validate data loading with small batches first

  The codebase is now complete and ready for experimentation! Start with the environment setup and run the verification test before proceeding to full training.
