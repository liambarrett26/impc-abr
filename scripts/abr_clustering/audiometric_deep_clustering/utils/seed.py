"""
Reproducibility utilities for ContrastiveVAE-DEC training.

This module provides comprehensive seed setting and random state management
to ensure reproducible results across PyTorch, NumPy, and Python's random module.
"""

import random
import numpy as np
import torch
from typing import Optional, Tuple, List, Any
import logging
import os

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms (slower but more reproducible)
    """
    # Python random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Set deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        
        # Set environment variable for deterministic operations
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        # Enable benchmark for better performance with consistent input sizes
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    logger.info(f"Set random seed to {seed} (deterministic={deterministic})")


def get_random_state() -> dict:
    """
    Get current random state for all random number generators.
    
    Returns:
        Dictionary containing random states
    """
    return {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_random': torch.get_rng_state(),
        'torch_cuda_random': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }


def set_random_state(state: dict) -> None:
    """
    Restore random state for all random number generators.
    
    Args:
        state: Dictionary containing random states from get_random_state()
    """
    random.setstate(state['python_random'])
    np.random.set_state(state['numpy_random'])
    torch.set_rng_state(state['torch_random'])
    
    if torch.cuda.is_available() and state['torch_cuda_random'] is not None:
        torch.cuda.set_rng_state(state['torch_cuda_random'])
    
    logger.info("Restored random state from checkpoint")


def create_reproducible_split(data_size: int, 
                            train_ratio: float = 0.8,
                            val_ratio: float = 0.1,
                            test_ratio: float = 0.1,
                            seed: Optional[int] = None) -> Tuple[List[int], List[int], List[int]]:
    """
    Create reproducible train/validation/test splits.
    
    Args:
        data_size: Total number of samples
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set  
        test_ratio: Proportion for test set
        seed: Random seed for split (uses current state if None)
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Set seed if provided
    if seed is not None:
        current_state = get_random_state()
        set_seed(seed)
    
    try:
        # Create shuffled indices
        indices = list(range(data_size))
        random.shuffle(indices)
        
        # Calculate split points
        train_end = int(data_size * train_ratio)
        val_end = train_end + int(data_size * val_ratio)
        
        # Split indices
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        logger.info(f"Created reproducible split: train={len(train_indices)}, "
                   f"val={len(val_indices)}, test={len(test_indices)}")
        
        return train_indices, val_indices, test_indices
        
    finally:
        # Restore previous state if we set a temporary seed
        if seed is not None:
            set_random_state(current_state)


def create_stratified_split(labels: np.ndarray,
                          train_ratio: float = 0.8,
                          val_ratio: float = 0.1, 
                          test_ratio: float = 0.1,
                          seed: Optional[int] = None) -> Tuple[List[int], List[int], List[int]]:
    """
    Create stratified train/validation/test splits maintaining label proportions.
    
    Args:
        labels: Array of labels for stratification
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for split
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    from sklearn.model_selection import train_test_split
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Create indices
    indices = np.arange(len(labels))
    
    # First split: separate test set
    if test_ratio > 0:
        temp_ratio = train_ratio + val_ratio
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_ratio, stratify=labels, random_state=seed
        )
        
        # Second split: separate train and validation
        if val_ratio > 0:
            adjusted_val_ratio = val_ratio / temp_ratio
            train_indices, val_indices = train_test_split(
                train_val_indices, 
                test_size=adjusted_val_ratio,
                stratify=labels[train_val_indices],
                random_state=seed
            )
        else:
            train_indices = train_val_indices
            val_indices = []
    else:
        test_indices = []
        if val_ratio > 0:
            train_indices, val_indices = train_test_split(
                indices, test_size=val_ratio, stratify=labels, random_state=seed
            )
        else:
            train_indices = indices
            val_indices = []
    
    logger.info(f"Created stratified split: train={len(train_indices)}, "
               f"val={len(val_indices)}, test={len(test_indices)}")
    
    return train_indices.tolist(), val_indices.tolist(), test_indices.tolist()


def ensure_reproducibility(seed: int = 42, deterministic: bool = False) -> None:
    """
    Comprehensive reproducibility setup for training.
    
    Args:
        seed: Random seed to use
        deterministic: Whether to enforce deterministic operations
    """
    # Set all random seeds
    set_seed(seed, deterministic=deterministic)
    
    # Additional PyTorch settings for reproducibility
    torch.backends.cudnn.enabled = True
    
    if deterministic:
        # Disable non-deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variables for deterministic CUDA operations
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Warning about performance impact
        logger.warning("Deterministic mode enabled - training will be slower but more reproducible")
    else:
        # Enable optimizations for better performance
        torch.backends.cudnn.benchmark = True
        logger.info("Benchmark mode enabled for better performance with consistent input sizes")
    
    logger.info(f"Reproducibility setup complete with seed={seed}")


class ReproducibleRandomState:
    """
    Context manager for temporarily setting random state.
    
    Useful for ensuring specific operations use a particular seed
    while preserving the global random state.
    """
    
    def __init__(self, seed: int):
        """
        Initialize context manager.
        
        Args:
            seed: Temporary seed to use within context
        """
        self.seed = seed
        self.saved_state = None
    
    def __enter__(self):
        """Save current state and set temporary seed."""
        self.saved_state = get_random_state()
        set_seed(self.seed)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous random state."""
        if self.saved_state is not None:
            set_random_state(self.saved_state)


def seed_worker(worker_id: int) -> None:
    """
    Seed function for PyTorch DataLoader workers.
    
    Ensures each worker has a different but reproducible seed.
    Use with DataLoader: DataLoader(..., worker_init_fn=seed_worker)
    
    Args:
        worker_id: Worker ID provided by DataLoader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_generator(seed: int) -> torch.Generator:
    """
    Create a PyTorch Generator with specified seed.
    
    Useful for reproducible data loading and sampling.
    
    Args:
        seed: Seed for the generator
        
    Returns:
        PyTorch Generator instance
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def validate_reproducibility(func: callable, *args, seed: int = 42, num_trials: int = 3, **kwargs) -> bool:
    """
    Validate that a function produces reproducible results.
    
    Args:
        func: Function to test for reproducibility
        *args: Arguments to pass to function
        seed: Seed to use for testing
        num_trials: Number of trials to run
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        True if function is reproducible, False otherwise
    """
    results = []
    
    for trial in range(num_trials):
        set_seed(seed)
        result = func(*args, **kwargs)
        
        # Convert to numpy if tensor
        if torch.is_tensor(result):
            result = result.detach().cpu().numpy()
        
        results.append(result)
    
    # Check if all results are identical
    for i in range(1, len(results)):
        if not np.allclose(results[0], results[i], rtol=1e-10, atol=1e-10):
            logger.warning(f"Reproducibility test failed: trial {i} differs from trial 0")
            return False
    
    logger.info(f"Reproducibility validated over {num_trials} trials")
    return True