# PyTensor Cache Management for IMPC-ABR

## Problem
Parallel PyMC/PyTensor processes can corrupt shared cache files, causing `_pickle.UnpicklingError: pickle data was truncated`.

## Solution Implemented
The parallel analysis now automatically:

1. **Clears main cache** before starting analysis
2. **Creates process-specific caches** in `/tmp/pytensor_cache_{pid}`
3. **Sets optimal PyTensor flags** for parallel processing

## Manual Cache Management (if needed)

### Clear cache before analysis:
```bash
rm -rf ~/.pytensor/
```

### Check cache size:
```bash
du -sh ~/.pytensor/
```

### Set environment flags manually:
```bash
export PYTENSOR_FLAGS="device=cpu,optimizer=fast_compile,base_compiledir=/tmp/pytensor_cache"
```

## Automatic Prevention
The modifications to `parallel_executor.py` ensure cache isolation is **automatically applied** to all future analysis runs.

No manual intervention required - the system is now robust against cache corruption.