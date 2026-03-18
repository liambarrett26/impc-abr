# Data Completeness Audit

Last updated: 2026-03-18

This document tracks what data files are referenced in the codebase and their availability across machines.

## Machines

| Machine | Description |
|---|---|
| **local (macOS)** | Development laptop |
| **evident-linux** | Linux compute server — GMM training was run here |
| **External drive** (`/Volumes/IMPC`) | USB drive with raw IMPC waveform data |

---

## Bayesian / Multivariate Analysis

### Input Data

| File | Referenced by | Available locally? |
|---|---|---|
| `data/processed/abr_full_data.csv` | `loader.py`, tests, `run_parallel_analysis.py` | Yes |
| `data/processed/abr_1000_lines.csv` | Tests (smaller test dataset) | Yes |
| `data/multivariate_confirmed_deafness_genes.txt` | `parallel_executor.py`, `batch_bayes_processor.py`, tests | Yes |
| `data/multivariate_candidate_deafness_genes.txt` | `parallel_executor.py`, `batch_bayes_processor.py`, tests | Yes |
| `data/confirmed_deafness_genes.txt` | Gene list comparisons | Yes |
| `data/candidate_deafness_genes.txt` | Gene list comparisons | Yes |
| `/Volumes/IMPC/impc_abr_data_with_waveforms_10k.csv` | `get_abr_data.py` (extraction script) | No — external drive not mounted. Only needed for initial data extraction, not analysis. |

### Output / Results

| File | Available locally? |
|---|---|
| `scripts/abr_analysis/results/analysis_20250204_170352/` | Yes |
| `scripts/abr_analysis/results/analysis_20250204_170647/` | Yes |
| `scripts/abr_analysis/results/analysis_20250204_170818/` | Yes |
| `scripts/abr_analysis/results/analysis_20250205_103144/` | Yes |
| `scripts/abr_analysis/results/analysis_20250205_103223/` | Yes |

### Additional Processed Data (locally available)

Gene-specific datasets and intermediate files in `data/processed/`:

- `abr_controls_only.csv`, `controls.csv`
- `abr_Adgrb1.csv`, `abr_Wdtc1.csv`, `abr_Pank2.csv`, `abr_mettl5.csv`, `Asf1b_with_control.csv`
- `abr_ocm_control_data.csv`, `abr_prkab1_control_data.csv`, `abr_prkab1.xlsx`
- Various missing gene tracking files (`missing_genes*.csv`, `abr_missing_genes_data*.csv`)
- Sex-specific analysis files (`sex_specific_*.csv`)

---

## GMM Clustering

### Input / Shared Data

| File | Referenced by | Available locally? |
|---|---|---|
| `gmm/shared_data/normalized_data.npy` | `pipeline_parallel.py`, `visualize_results.py`, `euclidean_reassignment.py` | Yes |
| `gmm/shared_data/metadata.csv` | `pipeline_parallel.py`, `analyze_gene_cluster_associations.py`, `euclidean_reassignment.py`, `generate_missing_plots.py` | Yes |
| `gmm/shared_data/preprocessor.pkl` | `pipeline_parallel.py` | Yes |
| `gmm/shared_data/preprocessing_info.json` | Reference / documentation | Yes |
| `gmm/shared_data/data_statistics.json` | Reference / documentation | Yes |
| `gmm/shared_data/concatenated_results_v6.csv` | `analyze_gene_cluster_associations.py` | Yes |

### Trained Model Results

These are all **missing locally** and reside on **evident-linux**. They are gitignored.

| Directory / File | Referenced by |
|---|---|
| `gmm/results/gmm_k{3..12}_{full,tied}/` | `aggregate_results.py`, `pipeline_parallel.py` |
| Each model dir contains: `model.pkl`, `cluster_labels.npy`, `cluster_probabilities.npy`, `cluster_centers.npy`, `metrics.json`, `analysis_results.json`, `completed.txt` | Various |
| `gmm/results/best_model.pkl` | `aggregate_results.py` (copied from best k/cov) |
| `gmm/results/best_cluster_labels.npy` | `aggregate_results.py` |
| `gmm/results/best_cluster_probabilities.npy` | `aggregate_results.py` |
| `gmm/results/best_model_metrics.json` | `aggregate_results.py` |
| `gmm/results/best_model_analysis.json` | `aggregate_results.py` |
| `gmm/results/model_comparison.csv` | `aggregate_results.py` |
| `gmm/results/model_selection_results.json` | `aggregate_results.py` |

### Hardcoded Path References (results from specific run)

Several scripts reference a specific results directory that does **not exist locally**:

| Hardcoded path | Referenced by |
|---|---|
| `results/june_23_2025/gmm_k4_tied/original_data.csv` | `visualize_results.py`, `analyze_gene_cluster_associations.py`, `euclidean_reassignment.py`, `generate_missing_plots.py` |
| `results/june_23_2025/gmm_k4_tied/cluster_labels.npy` | `visualize_results.py`, `analyze_gene_cluster_associations.py`, `euclidean_reassignment.py` |
| `results/june_23_2025/gmm_k4_tied/cluster_probabilities.npy` | `analyze_gene_cluster_associations.py`, `euclidean_reassignment.py` |
| `results/june_23_2025/gmm_k4_tied/cluster_centers.npy` | `euclidean_reassignment.py` |
| `results/june_23_2025/gmm_k4_tied/analysis_results.json` | `analyze_gene_cluster_associations.py` |
| `results/gene_cluster_analysis_sex_specific/gene_cluster_associations.csv` | `euclidean_reassignment.py` |
| `results/euclidean_assignment/gene_cluster_associations_euclidean.csv` | `compare_assignments.py` |
| `results/euclidean_assignment_original_space/gene_cluster_associations_original_space.csv` | `compare_assignments.py` |

### Post-Processing Outputs (also on evident-linux)

| File | Referenced by |
|---|---|
| `gene_cluster_associations.csv` | `analyze_gene_cluster_associations.py` |
| `gene_cluster_associations_euclidean.csv` | `euclidean_reassignment.py` |
| `gene_cluster_associations_original_space.csv` | `euclidean_reassignment_original_space.py` |
| Various `.png` / `.eps` visualisations | `analysis.py`, `visualize_results.py`, `aggregate_results.py` |

---

## Summary

| Component | Input data | Results / Models |
|---|---|---|
| **Bayesian analysis** | Complete locally | Complete locally |
| **GMM clustering** | Complete locally (shared_data/) | On evident-linux only |
| **Data extraction** | Needs `/Volumes/IMPC` external drive | N/A — already extracted |
