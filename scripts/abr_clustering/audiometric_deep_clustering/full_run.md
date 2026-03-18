# Full-Scale Training Guide for ContrastiveVAE-DEC

## Command Line Execution

### 🚨 RECOMMENDED: Memory-Optimized Training (32GB GPU)

For systems with 32GB GPU memory (RTX 5090) that encounter out-of-memory errors:

```bash
# Memory-optimized training script with environment setup
./scripts/train_memory_optimized.sh
```

Or manually:
```bash
# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Run memory-optimized training
python scripts/train.py \
    --config config/training_config_memory_optimized.yaml \
    --model-config config/model_config_memory_optimized.yaml \
    --experiment-name "full_scale_audiometric_clustering_memory_optimized" \
    --device auto \
    --seed 42
```

### Original Full-Scale Training (Requires >40GB GPU)

For the full-scale training run with proper model capacity, contrastive learning, and automatic cluster optimization, execute the following command:

```bash
python scripts/train.py \
    --config config/training_config_contrastive.yaml \
    --model-config config/model_config_large_optimized.yaml \
    --experiment-name "full_scale_audiometric_clustering_optimized" \
    --device auto \
    --seed 42
```

This command will:

- Use the **large optimized model configuration** with ~500K-1M parameters
- Enable **contrastive learning** with gene-based positive pairs
- Use **contrastive sampling** to ensure positive pairs in each batch
- **Automatically find optimal number of clusters** after VAE pretraining
- Automatically organize outputs under `experiments/full_scale_audiometric_clustering_optimized/{timestamp}_{hostname}/`
- Run the complete **4-stage training pipeline**:
  1. Stage 1: VAE Pretraining (50 epochs)
  2. Stage 1.5: Cluster Number Optimization (automatic)
  3. Stage 2: Cluster Initialization (with optimal k)
  4. Stage 3: Joint Training (150 epochs)
- Use GPU if available (auto-detection)
- Ensure reproducibility with seed 42

### Alternative Commands:

**Skip cluster optimization** (use fixed k=12):
```bash
python scripts/train.py \
    --config config/training_config_contrastive.yaml \
    --model-config config/model_config_large_optimized.yaml \
    --experiment-name "full_scale_fixed_clusters" \
    --skip-cluster-optimization \
    --device auto \
    --seed 42
```

**Force cluster optimization with custom range**:
```bash
python scripts/train.py \
    --config config/training_config_contrastive.yaml \
    --model-config config/model_config_large_optimized.yaml \
    --experiment-name "full_scale_custom_k_range" \
    --optimize-clusters \
    --k-range 8 15 \
    --device auto \
    --seed 42
```

The production settings for full-scale training are:

- Training epochs: 50 pretrain + 150 joint = 200 total epochs
- Batch size: 512 (optimal for GPU memory and contrastive learning)
- Learning rate: 1e-3 with cosine annealing to 1e-6
- Mixed precision: FP16 enabled for memory efficiency
- Early stopping: 30 epoch patience
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

## Memory Optimization Strategy

### Problem Analysis
The original model configuration (~500K-1M parameters) with batch size 512 and contrastive learning requires approximately:
- Model parameters: ~2GB
- Gradients: ~2GB
- Optimizer state (Adam): ~4GB
- Activations: ~4-8GB
- Contrastive learning (doubles memory): +8-16GB
- **Total: ~20-32GB** (exceeds 32GB with safety margin)

### Optimization Techniques Implemented

#### 1. **Model Architecture Reduction**
- Hidden dimensions: [128, 64, 32] vs [256, 128, 64] (50% reduction)
- Attention: 4 heads × 16 dims vs 8 heads × 32 dims (75% reduction)
- Latent dimension: 24 vs 32 (25% reduction)
- Contrastive projection: 64 vs 128 dims (50% reduction)
- **Parameter reduction: ~172K vs ~500K** (65% reduction)

#### 2. **Training Strategy Optimization**
- Batch size: 256 vs 512 (50% reduction)
- Gradient accumulation: 2 steps (maintains effective batch size 512)
- Mixed precision: fp16 (50% memory reduction)
- Gradient checkpointing: Trade compute for memory

#### 3. **Memory-Efficient Contrastive Learning**
- Only encode positive samples to latent space (not full forward pass)
- Use gradient checkpointing for positive sample encoding
- Reduced contrastive projection dimension

#### 4. **Runtime Optimizations**
- Environment variables: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Disabled memory-intensive logging (embeddings, attention weights)
- More frequent GPU cache clearing
- Reduced evaluation frequency (every 10 vs 5 epochs)

### Expected Performance Impact
- **Memory usage**: ~15-20GB (fits comfortably in 32GB)
- **Training time**: +10-15% due to gradient checkpointing
- **Model capacity**: Reduced but still sufficient for the task
- **Final performance**: Expected <5% degradation vs full model

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

With the larger model and cluster optimization (~500K-1M parameters):

- **Stage 1 (Pretraining)**: ~2-3 hours
- **Stage 1.5 (Cluster Optimization)**: ~15-30 minutes
- **Stage 2 (Initialization)**: ~5-10 minutes
- **Stage 3 (Joint training)**: ~6-8 hours
- **Total**: 8.5-12.5 hours on a modern GPU (e.g., V100, A100)

Note: The increased training time is due to:
- ~25-50x more parameters than the previous model
- Cluster optimization testing k=6 to k=18 (13 different cluster numbers)
- Contrastive loss computation requiring positive pair processing
- Larger attention mechanism (8 heads vs. 2)
- More frequent cluster updates (every 10 epochs vs. 50)

## Hardware Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM recommended
- **RAM**: 32GB+ system memory
- **Storage**: ~5GB for checkpoints and outputs
- **CUDA**: Version 11.0 or higher

The model will automatically use mixed precision training to optimize memory usage and training speed.