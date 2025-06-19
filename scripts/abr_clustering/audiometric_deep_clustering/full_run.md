# Full-Scale Training Guide for ContrastiveVAE-DEC

## Command Line Execution

For the full-scale training run, execute the following command:

```bash
python scripts/train.py \
    --config config/training_config.yaml \
    --model-config config/model_config.yaml \
    --experiment-name "full_scale_audiometric_clustering" \
    --device auto \
    --seed 42
```

This command will:

- Use the production configuration files
- Automatically organize outputs under `experiments/full_scale_audiometric_clustering/{timestamp}_{hostname}/`
- Run the complete 3-stage training pipeline (50 pretrain + 150 joint epochs = 200 total)
- Use GPU if available (auto-detection)
- Ensure reproducibility with seed 42

The production setting for full-scale training are:

- Training epochs: 50 pretrain + 150 joint = 200 total epochs
- Batch size: 512 (optimal for GPU memory and contrastive learning)
- Learning rate: 1e-3 with cosine annealing to 1e-6
- Mixed precision: FP16 enabled for memory efficiency
- Early stopping: 20 epoch patience
- Hardware: GPU with cuDNN benchmarking enabled

## Model Architecture Overview

### ContrastiveVAE-DEC: A Multi-Component Deep Learning Framework

The model integrates three complementary learning paradigms:

1. **Variational Autoencoder (VAE)**: Learns robust, continuous latent representations
2. **Deep Embedded Clustering (DEC)**: Jointly optimizes representation learning and clustering
3. **Contrastive Learning**: Enhances phenotype separation using gene-based positive pairs

### Key Architectural Components

#### 1. **Specialized Input Processing**
- **ABR Features (6)**: Auditory thresholds at 6, 12, 18, 24, 30 kHz + click-evoked
  - Processed through frequency-aware attention mechanism
  - Captures inter-frequency relationships critical for hearing patterns
- **Metadata Features (10)**: Age, weight, sex, zygosity, genetic background, etc.
  - Categorical features use learned embeddings
  - Continuous features are normalized
- **PCA Features (2)**: Principal components capturing global ABR patterns

#### 2. **Encoder Architecture** (18 → 10 dimensions)
```
Input (18) → Split Processing:
├─ ABR (6) → FrequencyAttention → ABREncoder → 16-dim
└─ Metadata (12) → EmbeddingLayers → MetadataEncoder → 24-dim
                    ↓
            Concatenate (40-dim)
                    ↓
        Hidden Layers: [64, 32, 16]
                    ↓
        Latent: μ (10-dim), log σ² (10-dim)
```

#### 3. **Latent Space** (10-dimensional)
- Probabilistic representation with reparameterization trick
- Enables smooth interpolation between phenotypes
- Constrained by KL divergence for well-behaved distributions

#### 4. **Clustering Layer**
- 12 learnable cluster centers in latent space
- Student's t-distribution for soft assignments (robust to outliers)
- Iterative refinement through self-supervised learning

#### 5. **Decoder Architecture** (10 → 18 dimensions)
- Mirrors encoder structure with separate ABR/metadata paths
- ABR reconstruction constrained to physiological ranges (0-100 dB SPL)
- Validates audiogram plausibility

### Parameter Choices and Rationale

#### Model Dimensions
- **Latent dimension: 10** - Balances expressiveness with interpretability
- **Hidden layers: [64, 32, 16]** - Progressive compression for smooth manifold
- **Cluster count: 12** - Based on domain knowledge of hearing loss subtypes

#### Training Parameters
- **Batch size: 512** - Optimal for contrastive learning and GPU memory
- **Learning rate: 1e-3** with cosine annealing - Stable convergence
- **Mixed precision (FP16)** - 2x memory efficiency with minimal accuracy loss

#### Loss Weights
- **Reconstruction: 1.0** (ABR: 2.0, Metadata: 1.0, PCA: 1.5)
- **KL divergence: 1.0** (β=1.0 for standard VAE)
- **Clustering: 1.0** - Equal importance to representation learning
- **Contrastive: 0.5** - Supplementary objective
- **Phenotype consistency: 0.3** - Encourages same-gene mice similarity

## Training Pipeline

### Stage 1: VAE Pretraining (50 epochs)
- **Objectives**: Reconstruction + KL divergence only
- **Purpose**: Learn meaningful latent representations
- **Expected outcome**: Smooth latent space with good reconstruction

### Stage 2: Cluster Initialization
- **Method**: K-means on pretrained latent representations
- **Iterations**: 300 K-means iterations
- **Purpose**: Data-driven initialization prevents poor local minima

### Stage 3: Joint Optimization (150 epochs)
- **All objectives active**: VAE + DEC + Contrastive + Phenotype consistency
- **Warmup**: 10 epochs for gradual loss weight increase
- **Expected outcome**: Well-separated clusters representing distinct phenotypes

## Why This Approach is Appropriate

### 1. **Domain-Specific Design**
- Frequency attention captures audiometric relationships
- Physiological constraints ensure valid reconstructions
- Gene-based contrastive learning leverages biological structure

### 2. **Addresses Key Challenges**
- **High dimensionality**: VAE provides dimensionality reduction
- **Heterogeneous features**: Separate processing paths for different data types
- **Unlabeled data**: Self-supervised clustering discovers structure
- **Genetic complexity**: Contrastive learning groups similar phenotypes

### 3. **Advantages Over Traditional Methods**
- **vs. K-means/GMM**: Learns non-linear manifolds, handles mixed data types
- **vs. Standard VAE**: Clustering objective prevents mode collapse
- **vs. Pure clustering**: Reconstruction regularizes learned representations

## Expected Outcomes

### 1. **Discovered Phenotype Clusters**
- ~12 distinct audiometric patterns
- Biologically interpretable (e.g., high-frequency loss, progressive loss)
- Novel subtypes within known deafness genes

### 2. **Gene-Phenotype Associations**
- Clusters enriched for specific gene knockouts
- Discovery of genes with similar audiometric effects
- Potential therapeutic targets based on phenotype similarity

### 3. **Clinical Translation Potential**
- Mouse phenotypes may correspond to human hearing loss subtypes
- Clustering could inform precision medicine approaches
- Novel biomarkers for hearing loss classification

## Monitoring Training Progress

Key metrics to watch:
- **VAE Loss**: Should decrease smoothly during pretraining
- **Cluster Purity**: Gene enrichment within clusters
- **Silhouette Score**: Cluster separation quality
- **Reconstruction Quality**: Especially ABR threshold patterns

## Post-Training Evaluation

After training completes:
```bash
# Comprehensive evaluation
python scripts/evaluate.py
    --checkpoint experiments/full_scale_audiometric_clustering/{run_id}/checkpoints/joint/best_joint.pt \
    --save-embeddings \
    --save-predictions

# Generate visualizations
python scripts/visualize_results.py \
    --embeddings experiments/full_scale_audiometric_clustering/{run_id}/evaluation/embeddings/latent_embeddings.npz \
    --results experiments/full_scale_audiometric_clustering/{run_id}/evaluation/embeddings/cluster_predictions.csv
```

## Expected Training Duration

- **Stage 1 (Pretraining)**: ~1-2 hours
- **Stage 2 (Initialization)**: ~5 minutes
- **Stage 3 (Joint training)**: ~3-4 hours
- **Total**: 4-6 hours on a modern GPU (e.g., V100, A100)

## Hardware Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM recommended
- **RAM**: 32GB+ system memory
- **Storage**: ~5GB for checkpoints and outputs
- **CUDA**: Version 11.0 or higher

The model will automatically use mixed precision training to optimize memory usage and training speed.