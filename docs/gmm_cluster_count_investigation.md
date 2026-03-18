# GMM Cluster Count Investigation

Last updated: 2026-03-18

## Problem

The four GMM cluster sizes reported in the manuscript sum to **56,326**:

| Cluster | Description | N |
|---|---|---|
| 1 | Moderate hearing loss | 9,525 |
| 2 | Normal hearing | 44,332 |
| 3 | Severe hearing loss | 562 |
| 4 | High-frequency hearing loss | 1,907 |
| **Total** | | **56,326** |

This matches the old "met inclusion criteria" figure (56,326), which has since been corrected to **55,904** based on the current dataset and stated inclusion criteria.

The GMM was trained on evident-linux. The trained model results are not available locally (see `docs/data_completeness.md`). Before re-running, we need to verify whether the discrepancy is due to:

1. The GMM being trained on a different/older version of the dataset
2. The GMM pipeline applying different filtering than the Bayesian pipeline
3. Both

## Key Difference: GMM Loader vs Bayesian Pipeline Filtering

The GMM loader (`scripts/abr_clustering/gmm/loader.py`) applies an **additional filter** not present in the Bayesian pipeline:

| Filter | Bayesian pipeline | GMM pipeline |
|---|---|---|
| Complete 5-freq data | Yes | Yes |
| Threshold range 0-100 dB SPL | **No** | **Yes** (loader.py:109-115) |
| Missing critical metadata | **No** | **Yes** (loader.py:120-127) |
| >= 3 mutants per group | Yes | Yes |
| >= 20 matched controls | Yes | Yes |

Applying these filters to the current dataset:

| Step | Mice remaining |
|---|---|
| Raw data | 59,145 |
| Complete 5-freq | 56,676 |
| + Threshold range (0-100 dB) | 56,484 |
| Old manuscript figure | 56,326 |
| Gap | **-158** |

The 0-100 dB range filter removes 192 mice (141 with negative 6kHz thresholds, plus others). After this filter we get 56,484, which is still 158 more than 56,326. This remaining gap is likely from:
- The GMM experimental group formation filtering (different grouping than Bayesian — groups by gene+centre+pipeline+background+equipment)
- Or a previous version of `abr_full_data.csv`

**Important:** The GMM clusters ALL mice that pass filtering (both mutants and controls), not just those in valid experimental groups. This is by design — the clustering is unsupervised across the full dataset.

## Verification Steps (on evident-linux)

### Step 1: Check the preprocessing log

```bash
# On evident-linux, navigate to the GMM results directory
cd /path/to/gmm/results  # or wherever the run was performed

# Check the preprocessing log for the original input count
cat shared_data/preprocessing.log | grep -E "Loaded|After|Final|rows"

# Check preprocessing info
cat shared_data/preprocessing_info.json
cat shared_data/data_statistics.json
```

This will tell you exactly how many mice were in the training set and what filters were applied.

### Step 2: Check the normalised data shape

```bash
python3 -c "
import numpy as np
data = np.load('shared_data/normalized_data.npy')
print(f'Normalised data shape: {data.shape}')
print(f'Number of mice clustered: {data.shape[0]}')
"
```

If this returns 56,326, the GMM was trained on that exact dataset.

### Step 3: Check metadata for dataset version

```bash
python3 -c "
import pandas as pd
meta = pd.read_csv('shared_data/metadata.csv')
print(f'Metadata rows: {len(meta)}')
print(f'Unique genes: {meta[\"gene_symbol\"].nunique()}')
print(f'Sample groups: {meta[\"biological_sample_group\"].value_counts().to_dict()}')
"
```

### Step 4: Verify cluster assignments match manuscript

```bash
python3 -c "
import numpy as np
labels = np.load('results/gmm_k4_tied/cluster_labels.npy')  # adjust path if needed
unique, counts = np.unique(labels, return_counts=True)
print('Cluster sizes:')
for u, c in zip(unique, counts):
    print(f'  Cluster {u}: {c}')
print(f'Total: {sum(counts)}')
"
```

### Step 5: Compare with current data

```bash
python3 -c "
import pandas as pd
import numpy as np

# Load the metadata used for training
meta_old = pd.read_csv('shared_data/metadata.csv')

# Load current dataset
df_new = pd.read_csv('/path/to/data/processed/abr_full_data.csv', low_memory=False)

print(f'Old training metadata: {len(meta_old)} rows')
print(f'New full dataset: {len(df_new)} rows')

# Check if specimen_ids match
if 'specimen_id' in meta_old.columns:
    old_ids = set(meta_old['specimen_id'])
    new_ids = set(df_new['specimen_id'])
    print(f'Specimens in old but not new: {len(old_ids - new_ids)}')
    print(f'Specimens in new but not old: {len(new_ids - old_ids)}')
"
```

## If Re-run Is Needed

If the investigation confirms the GMM was trained on a different dataset, re-run the full pipeline from the `scripts/abr_clustering/gmm/` directory:

```bash
cd /path/to/impc-abr/scripts/abr_clustering/gmm

bash run_parallel_gmm.sh \
  -d /path/to/data/processed/abr_full_data.csv \
  -o results \
  -s shared_data \
  --min-k 3 \
  --max-k 12 \
  --min-mutants 3 \
  --min-controls 20 \
  --n-bootstrap 100 \
  --random-state 42 \
  --log-level INFO \
  -j 8
```

This will:

1. **Preprocess** the data (`preprocess_data.py`):
   - Load `abr_full_data.csv` via `IMPCABRLoader`
   - Apply quality filters (complete data, 0-100 dB range, critical metadata)
   - Create experimental groups (>= 3 mutants, >= 20 matched controls)
   - Apply two-stage normalisation (group Z-score + global min-max scaling)
   - Save to `shared_data/`: `normalized_data.npy`, `metadata.csv`, `preprocessor.pkl`, etc.

2. **Train 20 GMM models** in parallel (`pipeline_parallel.py`):
   - k = 3 to 12, each with full and tied covariance
   - 100 bootstrap iterations for stability assessment per model
   - Each saves to `results/gmm_k{n}_{cov}/`: model, labels, probabilities, metrics

3. **Aggregate and select best model** (`aggregate_results.py`):
   - Weighted scoring: BIC (40%), AIC (20%), silhouette (20%), stability (20%)
   - Copies best model artifacts to `results/`
   - Generates comparison plots

### Post-Run Steps

After re-running, update the manuscript cluster sizes:

```bash
python3 -c "
import numpy as np
labels = np.load('results/gmm_k4_tied/cluster_labels.npy')
unique, counts = np.unique(labels, return_counts=True)
for u, c in sorted(zip(unique, counts)):
    print(f'Cluster {u}: N = {c:,}')
print(f'Total: {sum(counts):,}')
"
```

Also verify the optimal k hasn't changed:

```bash
cat results/model_selection_report.txt
```

If k=4 tied is still selected as optimal, update the cluster Ns in the manuscript. If a different model is now optimal, this requires more significant manuscript revision.

### Important Notes

- The GMM clusters **all mice** that pass quality filters (mutants + controls), not just those in valid experimental groups. This is intentional — the clustering is unsupervised and discovers phenotype patterns across the entire dataset.
- The Bayesian analysis counts (55,904 mice in valid groups) will therefore differ from the GMM total (which includes mice in groups that are too small for statistical testing but still valid for clustering).
- The manuscript should make this distinction clear: the inclusion criteria (>= 3 mutants, >= 20 controls) apply to the statistical analyses, while the GMM uses all mice with complete, quality-filtered ABR data.
