#!/usr/bin/env python3
"""
PyTensor cache management for parallel ABR analysis.

This script provides utilities to prevent PyTensor cache corruption
during parallel processing by setting up process-specific cache directories.
"""

import os
import tempfile
import shutil
from pathlib import Path
import multiprocessing as mp

def setup_process_specific_cache():
    """
    Set up a unique PyTensor cache directory for each process.
    Call this at the start of each worker process.
    """
    process_id = os.getpid()
    temp_cache_dir = Path(tempfile.gettempdir()) / f"pytensor_cache_{process_id}"
    
    # Create the directory if it doesn't exist
    temp_cache_dir.mkdir(exist_ok=True, parents=True)
    
    # Set PyTensor to use this cache directory
    os.environ['PYTENSOR_FLAGS'] = f'base_compiledir={temp_cache_dir}'
    
    print(f"Process {process_id}: Using cache directory {temp_cache_dir}")
    return temp_cache_dir

def cleanup_process_cache():
    """
    Clean up the process-specific cache directory.
    Call this at the end of each worker process.
    """
    process_id = os.getpid()
    temp_cache_dir = Path(tempfile.gettempdir()) / f"pytensor_cache_{process_id}"
    
    if temp_cache_dir.exists():
        try:
            shutil.rmtree(temp_cache_dir)
            print(f"Process {process_id}: Cleaned up cache directory {temp_cache_dir}")
        except Exception as e:
            print(f"Process {process_id}: Warning - could not clean cache: {e}")

def clear_main_cache():
    """
    Clear the main PyTensor cache directory.
    Call this before starting parallel analysis.
    """
    cache_dir = Path.home() / ".pytensor"
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            print("Cleared main PyTensor cache directory")
        except Exception as e:
            print(f"Warning: Could not clear main cache: {e}")

def set_pytensor_flags_for_parallel():
    """
    Set PyTensor flags optimized for parallel processing.
    """
    flags = [
        'device=cpu',  # Force CPU to avoid GPU conflicts
        'force_device=True',
        'optimizer=fast_compile',  # Faster compilation, less optimization
        'exception_verbosity=low'  # Reduce debug output in parallel
    ]
    
    # Add to existing flags if any
    existing_flags = os.environ.get('PYTENSOR_FLAGS', '')
    if existing_flags:
        flags.append(existing_flags)
    
    os.environ['PYTENSOR_FLAGS'] = ','.join(flags)
    print(f"Set PyTensor flags: {os.environ['PYTENSOR_FLAGS']}")

if __name__ == "__main__":
    print("PyTensor cache management utilities")
    print("Use these functions in your parallel processing code")