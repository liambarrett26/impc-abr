# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing **ContrastiveVAE-DEC** (Contrastive Variational Deep Embedded Clustering) for discovering audiometric phenotypes in mouse genetic data from the International Mouse Phenotyping Consortium (IMPC). The core innovation is learning to cluster mice by hearing patterns rather than genetic similarity, potentially discovering novel hearing loss subtypes and phenotypes.

## Architecture & Key Concepts

The model combines three learning paradigms:
- **Variational Autoencoder (VAE)**: Learns robust latent representations with reconstruction + KL divergence loss
- **Deep Embedded Clustering (DEC)**: Performs joint representation learning and clustering using soft assignments
- **Contrastive Learning**: Improves phenotype separation using positive/negative pairs

### Training Strategy (3-Stage Curriculum)
1. **Stage 1: VAE Pretraining** - Learn latent representations (reconstruction + KL loss only)
2. **Stage 2: Cluster Initialization** - K-means initialization in latent space 
3. **Stage 3: Joint Optimization** - All objectives combined (VAE + clustering + contrastive + phenotype consistency)

### Data Pipeline
- **Input**: 18 features (6 ABR thresholds + 10 metadata + 2 PCA components)
- **Dataset**: 59,145 mice from 6,749 gene knockout lines
- **Target**: Discover ~12 distinct audiometric phenotype clusters

## Core Architecture Components

### Models (`models/`)
- **`full_model.py`**: Main `ContrastiveVAEDEC` class that orchestrates all components
- **`encoder.py`**: Frequency-aware encoder with attention mechanism for ABR features
- **`decoder.py`**: Reconstruction decoder 
- **`clustering_layer.py`**: DEC clustering with learnable cluster centers and soft assignments
- **`attention.py`**: Multi-head attention specialized for ABR frequency relationships
- **`vae.py`**: VAE components (latent space, reparameterization trick)

### Loss Functions (`losses/`)
- **`combined_loss.py`**: Multi-objective loss orchestrator with stage-aware weighting
- **`vae_loss.py`**: ELBO, KL divergence, beta-VAE scheduling
- **`clustering_loss.py`**: DEC loss, auxiliary clustering objectives
- **`contrastive.py`**: InfoNCE and other contrastive learning losses
- **`reconstruction.py`**: Feature-specific reconstruction losses (ABR vs metadata)

### Training (`training/`)
- **`trainer.py`**: Main training orchestrator handling 3-stage curriculum
- **`pretrain.py`**: VAE pretraining stage implementation
- **`finetune.py`**: Joint optimization stage implementation  
- **`callbacks.py`**: Training monitoring, early stopping, checkpointing

### Data Handling (`data/`)
- **`dataset.py`**: PyTorch datasets with contrastive pair generation
- **`preprocessor.py`**: Feature preprocessing pipeline
- **`augmentations.py`**: Data augmentations for contrastive learning
- **`dataloader.py`**: Custom dataloaders with sampling strategies

## Configuration System

Uses YAML-based configuration in `config/`:
- **`model_config.yaml`**: Model architecture, hyperparameters, loss weights
- **`training_config.yaml`**: Training process, optimization, data loading

Key configuration parameters:
- Data path: `/home/liamb/impc-abr/data/processed/abr_full_data.csv`
- Latent dimensions: 10, Hidden dimensions: [64, 32], Clusters: 12
- 3-stage training: 100 → 50 → 200 epochs
- Batch size: 512, Learning rate: 1e-3 with cosine annealing

## Development Workflow

### Model Training
The training process must follow the 3-stage curriculum:
1. Load configurations from YAML files
2. Initialize model with `ContrastiveVAEDEC(config)`
3. Stage 1: Set `training_stage='pretrain'`, train VAE only
4. Stage 2: Initialize clusters with `model.initialize_clusters(dataloader)`
5. Stage 3: Set `training_stage='joint'`, train all objectives

### Key Model States
- **`training_stage`**: Controls which losses are active ('pretrain', 'cluster_init', 'joint')
- **`clusters_initialized`**: Boolean flag for cluster center initialization
- Loss weights are stage-dependent and can be scheduled

### Evaluation Pipeline (`evaluation/`)
- **`metrics.py`**: Clustering quality metrics (silhouette, ARI, NMI)
- **`visualization.py`**: t-SNE/UMAP plots, audiogram visualizations, interactive dashboards
- **`phenotype_analysis.py`**: Biological interpretation (hearing loss classification, gene associations)

## Important Implementation Notes

### Multi-Objective Loss Handling
The `MultiObjectiveLoss` class in `combined_loss.py` orchestrates all loss components with stage-aware weighting. Always use this rather than individual loss functions directly.

### Attention Mechanism
The frequency attention in the encoder treats ABR thresholds as a sequence where each frequency is a token. This captures relationships between frequencies critical for hearing loss pattern recognition.

### Data Preprocessing
ABR features require specialized preprocessing due to their frequency-dependent nature. Metadata features are handled separately with appropriate normalization for mixed data types.

### Gene Label Handling
Gene labels use -1 for unknown/missing genes. The contrastive learning uses gene labels to create positive pairs (same gene knockout = positive pair).

### Cluster Initialization
Clusters must be initialized before joint training. Use K-means on the latent space from pretrained VAE. The model tracks initialization state with `clusters_initialized` flag.

### Memory and Performance
- Designed for single GPU training with batch size 512
- Uses mixed precision (precision: 16) for memory efficiency
- Gradient clipping (norm: 1.0) for training stability

## Testing & Validation

The project emphasizes biological validation:
- Statistical significance testing (ANOVA, Mann-Whitney U)
- Gene-cluster association analysis (Fisher's exact test, Chi-square)
- Validation against known hearing loss genes
- Clinical hearing loss pattern classification
- Functional impact assessment

Always validate clustering results both statistically and biologically using the evaluation pipeline.