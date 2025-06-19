#!/usr/bin/env python3
"""
Memory monitoring utilities for GPU memory management.
"""

import torch
import psutil
import gc
import logging
from typing import Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor and log GPU and CPU memory usage."""
    
    def __init__(self, log_frequency: int = 100):
        """Initialize memory monitor.
        
        Args:
            log_frequency: Log memory stats every N calls
        """
        self.log_frequency = log_frequency
        self.call_count = 0
        self.peak_memory = 0
        
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get GPU memory information in GB."""
        if not torch.cuda.is_available():
            return {}
            
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1e9  # Convert to GB
        reserved = torch.cuda.memory_reserved(device) / 1e9
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        free = total - reserved
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'total': total,
            'utilization': allocated / total * 100
        }
    
    def get_cpu_memory_info(self) -> Dict[str, float]:
        """Get CPU memory information in GB."""
        memory = psutil.virtual_memory()
        return {
            'used': memory.used / 1e9,
            'available': memory.available / 1e9,
            'total': memory.total / 1e9,
            'utilization': memory.percent
        }
    
    def log_memory_stats(self, context: str = ""):
        """Log current memory statistics."""
        gpu_info = self.get_gpu_memory_info()
        cpu_info = self.get_cpu_memory_info()
        
        if gpu_info:
            self.peak_memory = max(self.peak_memory, gpu_info['allocated'])
            logger.info(f"Memory [{context}] - GPU: {gpu_info['allocated']:.2f}GB/{gpu_info['total']:.2f}GB "
                       f"({gpu_info['utilization']:.1f}%), Peak: {self.peak_memory:.2f}GB")
        
        logger.info(f"Memory [{context}] - CPU: {cpu_info['used']:.2f}GB/{cpu_info['total']:.2f}GB "
                   f"({cpu_info['utilization']:.1f}%)")
    
    def check_and_log(self, context: str = ""):
        """Check memory and log if frequency threshold met."""
        self.call_count += 1
        if self.call_count % self.log_frequency == 0:
            self.log_memory_stats(context)
    
    def force_cleanup(self):
        """Force garbage collection and clear GPU cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def check_memory_threshold(self, threshold: float = 0.9) -> bool:
        """Check if GPU memory usage exceeds threshold.
        
        Args:
            threshold: Memory usage threshold (0.0 to 1.0)
            
        Returns:
            True if memory usage exceeds threshold
        """
        gpu_info = self.get_gpu_memory_info()
        if gpu_info and gpu_info['utilization'] / 100 > threshold:
            logger.warning(f"GPU memory usage {gpu_info['utilization']:.1f}% exceeds threshold {threshold*100:.0f}%")
            return True
        return False


@contextmanager
def memory_context(name: str, monitor: Optional[MemoryMonitor] = None, cleanup: bool = True):
    """Context manager for memory monitoring around code blocks.
    
    Args:
        name: Context name for logging
        monitor: Memory monitor instance (creates new if None)
        cleanup: Whether to force cleanup on exit
    """
    if monitor is None:
        monitor = MemoryMonitor()
    
    monitor.log_memory_stats(f"{name} - Start")
    
    try:
        yield monitor
    finally:
        if cleanup:
            monitor.force_cleanup()
        monitor.log_memory_stats(f"{name} - End")


def optimize_memory_settings():
    """Apply memory optimization settings."""
    # Set memory allocation strategy
    if torch.cuda.is_available():
        # Enable memory mapping for large tensors
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory fraction to avoid full allocation
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        logger.info("Applied CUDA memory optimizations")
    
    # Set multiprocessing settings for data loading
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    logger.info("Memory optimization settings applied")


def check_memory_requirements(batch_size: int, model_params: int, sequence_length: int = 1) -> Dict[str, float]:
    """Estimate memory requirements for training.
    
    Args:
        batch_size: Training batch size
        model_params: Number of model parameters
        sequence_length: Sequence length (for sequential models)
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Estimate memory usage (rough approximation)
    # 4 bytes per float32 parameter
    model_memory = model_params * 4 / 1e9
    
    # Gradients (same size as model)
    gradient_memory = model_memory
    
    # Optimizer states (Adam: 2x model size)
    optimizer_memory = model_memory * 2
    
    # Activation memory (depends on batch size and architecture)
    # Rough estimate: batch_size * hidden_dims * layers * 4 bytes
    activation_memory = batch_size * 256 * 10 * 4 / 1e9  # Rough estimate
    
    total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory
    
    return {
        'model': model_memory,
        'gradients': gradient_memory,
        'optimizer': optimizer_memory,
        'activations': activation_memory,
        'total': total_memory
    }


def suggest_batch_size(available_memory_gb: float, model_params: int) -> int:
    """Suggest optimal batch size based on available memory.
    
    Args:
        available_memory_gb: Available GPU memory in GB
        model_params: Number of model parameters
        
    Returns:
        Suggested batch size
    """
    # Leave 20% headroom for safety
    usable_memory = available_memory_gb * 0.8
    
    # Try different batch sizes
    for batch_size in [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
        memory_req = check_memory_requirements(batch_size, model_params)
        if memory_req['total'] <= usable_memory:
            logger.info(f"Suggested batch size: {batch_size} (estimated memory: {memory_req['total']:.2f}GB)")
            return batch_size
    
    logger.warning("Could not find suitable batch size for available memory")
    return 1


if __name__ == "__main__":
    # Test memory monitoring
    monitor = MemoryMonitor()
    monitor.log_memory_stats("Test")
    
    # Test memory context
    with memory_context("Test Context") as ctx:
        # Simulate some computation
        if torch.cuda.is_available():
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.matmul(x, x.t())
            del x, y
    
    # Test memory estimation
    estimates = check_memory_requirements(batch_size=256, model_params=500000)
    print("Memory estimates:", estimates)