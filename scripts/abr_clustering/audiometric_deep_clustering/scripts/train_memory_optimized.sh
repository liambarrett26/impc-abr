#!/bin/bash

# Memory-Optimized Training Script for ContrastiveVAE-DEC
# Optimized for 32GB GPU (RTX 5090) with reduced memory usage

echo "==================================="
echo "Memory-Optimized Training Launch"
echo "==================================="

# Set environment variables for memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export CUDA_LAUNCH_BLOCKING=0

# Clear GPU cache before starting
echo "Clearing GPU cache..."
python -c "import torch; torch.cuda.empty_cache()"

# Display memory information
echo "GPU Memory Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv

echo ""
echo "Starting memory-optimized training..."
echo "Configuration: Batch size 256, Gradient accumulation 2 (effective batch size 512)"
echo "Model: Reduced dimensions for memory efficiency"
echo ""

# Run training with memory-optimized configurations
python scripts/train.py \
    --config config/training_config_memory_optimized.yaml \
    --model-config config/model_config_memory_optimized.yaml \
    --experiment-name "full_scale_audiometric_clustering_memory_optimized" \
    --device auto \
    --seed 42

echo ""
echo "Training completed!"
echo "Check experiments/ directory for results"