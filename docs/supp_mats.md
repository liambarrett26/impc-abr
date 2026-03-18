# Supplementary Materials

Description of the five supplementary files accompanying the manuscript.

---

## Supplementary File 1 — IMPC Genotype-Phenotype Assertions

IMPC statistical results for all ABR parameters across all tested alleles in Data Release 22.1.

- **Format:** CSV
- **Rows:** 44,862 (one per allele × ABR parameter × centre statistical test)
- **Key columns:**
  - `marker_symbol`, `allele_symbol`, `zygosity` — gene/allele identification
  - `parameter_stable_id`, `parameter_name` — ABR parameter tested (6 frequency-specific + click + ABR_001 parameters)
  - `phenotyping_center`, `pipeline_name`, `genetic_background` — experimental context
  - `classification_tag` — IMPC significance call (e.g. "Not significant", "With phenotype threshold value 1e-04 - significant in combined dataset only (High)")
  - `statistical_method` — method used by IMPC pipeline
  - `female_control_count`, `male_control_count`, `female_mutant_count`, `male_mutant_count` — sample sizes
  - `female_control_mean`, `male_control_mean`, `female_mutant_mean`, `male_mutant_mean` — group means
  - `effect_size`, `p_value` — overall effect size and p-value
  - Sex-specific p-values and effect sizes for directional tests
- **Source:** IMPC `statistical-result` Solr core, filtered to ABR procedures (IMPC_ABR_001, IMPC_ABR_002)
- **Location:** `/Volumes/IMPC/stat_processed_data.csv`

---

## Supplementary File 2 — Multivariate Analysis Results (All Alleles)

Full multivariate distribution analysis results for all alleles that met inclusion criteria.

- **Format:** CSV
- **Rows:** 6,984 (one per experimental group: allele + zygosity + centre)
- **Key columns:**
  - `gene_symbol`, `allele_symbol`, `zygosity`, `center`, `background` — group identification
  - Combined-sex results (`all_` prefix): `p_value`, `q_value` (FDR-corrected), `test_statistic`, `n_mutants`, `n_controls`, `mean_mutant_logprob`, `mean_mutant_distance`, `mean_control_logprob`, `mean_control_distance`
  - Male-only results (`male_` prefix): same metrics
  - Female-only results (`female_` prefix): same metrics
- **Notes:** The Mahalanobis distance measures how far each mutant audiogram deviates from the matched control distribution. P-values are derived from the chi-squared distribution with 5 degrees of freedom. Q-values are FDR-corrected across all tests.
- **Location:** OneDrive `results/multivariate/detailed_results.csv`

---

## Supplementary File 3 — Curated Hearing Loss Gene Lists

Two curated lists of genes with prior evidence of hearing loss, used as benchmarks for validating the analytical pipeline.

- **Format:** Text (one gene symbol per line)
- **Known hearing loss genes** (52 genes): Genes with published evidence of causality for hearing loss in mouse or human, which are also known to have a significant ABR phenotype as called by the IMPC.
- **Candidate hearing loss genes** (47 genes): Genes previously reported by Bowl et al. (2017) as candidate hearing loss genes, or associated with hearing impairment in a genome-wide association study (GWAS).
- **Location:** Repo `data/confirmed_deafness_genes.txt` and `data/candidate_deafness_genes.txt`; also OneDrive `data/Known.txt` and `data/Candidate.txt`

---

## Supplementary File 4 — Bayesian Analysis Results

Full Bayesian mixture model results for all experimental groups that met inclusion criteria.

- **Format:** CSV
- **Rows:** 6,984 (one per experimental group: allele + zygosity + centre)
- **Genes:** 6,549 unique genes
- **Key columns:**
  - `gene_symbol`, `allele_symbol`, `zygosity`, `center` — group identification
  - Combined-sex results (`all_` prefix): `bayes_factor`, `p_hearing_loss` (posterior mean probability of hearing loss), `hdi_lower` / `hdi_upper` (94% highest density interval), `n_mutants`, `n_controls`, `analysis_key`
  - Frequency-specific effect sizes (`all_effect_6kHz-evoked` through `all_effect_30kHz-evoked`) — posterior mean hearing loss shift in dB at each frequency
  - Male-only results (`male_` prefix): same metrics
  - Female-only results (`female_` prefix): same metrics
- **Notes:** Bayes factors quantify evidence for hearing loss as the ratio of posterior to prior odds. Evidence thresholds: Extreme (BF > 100), Very Strong (30 < BF ≤ 100), Strong (10 < BF ≤ 30), Substantial (3 < BF ≤ 10), Weak/None (BF ≤ 3). 139 alleles (133 unique genes) met the threshold for substantial evidence (BF ≥ 3).
- **Location:** `scripts/abr_clustering/gmm/shared_data/concatenated_results_v6.csv` (also on OneDrive)
- **Filtered version (significant only):** OneDrive `results/all_sig_results_annotated.csv` — 139 rows with an additional `gene_classification` column (Known / Candidate / Novel)

---

## Supplementary File 5 — GMM Cluster Assignments

Cluster assignments for all mice from the four-cluster Gaussian mixture model with tied covariance.

- **Format:** CSV
- **Rows:** One per mouse (total determined by GMM quality filters — see `docs/gmm_cluster_count_investigation.md`)
- **Key columns:**
  - Sample identification: `specimen_id`, `gene_symbol`, `allele_symbol`, `zygosity`, `sex`, `phenotyping_center`, `biological_sample_group`
  - `cluster_assignment` — assigned cluster (1–4)
  - `cluster_probability_1` through `cluster_probability_4` — posterior probability of membership in each cluster
  - Original ABR thresholds at 6, 12, 18, 24, 30 kHz
- **Cluster definitions:**
  - Cluster 1: Moderate hearing loss at all frequencies
  - Cluster 2: Normal hearing
  - Cluster 3: Severe hearing loss across all frequencies
  - Cluster 4: High-frequency hearing loss
- **Location:** To be assembled from GMM results on evident-linux (`cluster_labels.npy` + `metadata.csv`)
