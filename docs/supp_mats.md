# Supplementary Materials

Description of the supplementary files accompanying the manuscript.

---

## Supplementary File 1 — IMPC Genotype-Phenotype Assertions

**File:** `supplementary_file_1_impc_genotype_phenotype_assertions.csv`

IMPC statistical results for all ABR parameters across all tested alleles in Data Release 22.1. Contains the results of the IMPC's own univariate reference range statistical pipeline (P < 1×10⁻⁴ threshold) for each gene tested at each ABR parameter.

- **Rows:** 44,862 (one per allele × ABR parameter × centre statistical test)
- **Genes:** 6,749 unique genes tested
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

---

## Supplementary File 2 — Multivariate Analysis Results (All Alleles)

**File:** `supplementary_file_2_multivariate_results_all.csv`

Full multivariate distribution analysis results for all alleles that met inclusion criteria. Each mouse's audiogram is treated as a unified 5-dimensional observation, and the Mahalanobis distance from the matched control distribution is calculated.

- **Rows:** 6,984 (one per experimental group: allele + zygosity + centre)
- **Key columns:**
  - `gene_symbol`, `allele_symbol`, `zygosity`, `center`, `background` — group identification
  - Combined-sex results (`all_` prefix): `p_value`, `q_value` (FDR-corrected), `test_statistic`, `n_mutants`, `n_controls`, `mean_mutant_logprob`, `mean_mutant_distance`, `mean_control_logprob`, `mean_control_distance`
  - Male-only results (`male_` prefix): same metrics
  - Female-only results (`female_` prefix): same metrics
- **Notes:** P-values are derived from the chi-squared distribution with 5 degrees of freedom. Q-values are FDR-corrected across all tests.

---

## Supplementary File 3 — Curated Hearing Loss Gene Lists

**Files:** `supplementary_file_3a_known_hearing_loss_genes.txt`, `supplementary_file_3b_candidate_hearing_loss_genes.txt`

Two curated lists of genes with prior evidence of hearing loss, used as benchmarks for validating the analytical pipeline.

- **Format:** Text (one gene symbol per line)
- **Known hearing loss genes** (52 genes): Genes with published evidence of causality for hearing loss in mouse or human, which are also known to have a significant ABR phenotype as called by the IMPC.
- **Candidate hearing loss genes** (47 genes): Genes previously reported by Bowl et al. (2017) as candidate hearing loss genes, or associated with hearing impairment in a genome-wide association study (GWAS).

---

## Supplementary File 4 — Bayesian Analysis Results

**Files:** `supplementary_file_4_bayesian_results_all.csv`, `supplementary_file_4_bayesian_results_significant_annotated.csv`

Bayesian mixture model results. The model estimates whether hearing loss is present and characterises its magnitude, incorporating the prior expectation that most gene knockouts do not affect hearing (Beta(1, 3) prior).

### 4a — Full results (`supplementary_file_4_bayesian_results_all.csv`)

- **Rows:** 6,984 (one per experimental group: allele + zygosity + centre)
- **Genes:** 6,549 unique genes
- **Key columns:**
  - `gene_symbol`, `allele_symbol`, `zygosity`, `center` — group identification
  - Combined-sex results (`all_` prefix): `bayes_factor`, `p_hearing_loss` (posterior mean probability of hearing loss), `hdi_lower` / `hdi_upper` (94% highest density interval), `n_mutants`, `n_controls`
  - Frequency-specific effect sizes (`all_effect_6kHz-evoked` through `all_effect_30kHz-evoked`) — posterior mean hearing loss shift in dB at each frequency
  - Male-only results (`male_` prefix): same metrics
  - Female-only results (`female_` prefix): same metrics
- **Notes:** Bayes factors quantify evidence for hearing loss as the ratio of posterior to prior odds. Evidence thresholds: Extreme (BF > 100), Very Strong (30 < BF ≤ 100), Strong (10 < BF ≤ 30), Substantial (3 < BF ≤ 10), Weak/None (BF ≤ 3).

### 4b — Significant alleles, annotated (`supplementary_file_4_bayesian_results_significant_annotated.csv`)

- **Rows:** 139 alleles (133 unique genes) with BF ≥ 3 in one or both sexes
- **Additional column:** `gene_classification` — categorisation as Known (47), Candidate (26), or Novel (60)

---

## Supplementary File 5 — GMM Cluster Assignments

**File:** `supplementary_file_5_cluster_assignments.csv`

Cluster assignments for all 56,326 mice from the four-cluster Gaussian mixture model (k=4, tied covariance). Assignments are based on Euclidean distance from cluster mean profiles in the original dB SPL space.

- **Rows:** 56,326 (one per mouse)
- **Key columns:**
  - `specimen_id`, `gene_symbol`, `allele_symbol`, `zygosity`, `sex`, `phenotyping_center`, `genetic_background`, `biological_sample_group` — sample identification and metadata
  - `6kHz-evoked ABR Threshold` through `30kHz-evoked ABR Threshold` — original ABR thresholds in dB SPL
  - `cluster_assignment` — assigned cluster (0–3)
  - `cluster_description` — cluster phenotype label
- **Cluster definitions:**
  - Cluster 0 — Moderate hearing loss (N = 9,525): elevated thresholds across all frequencies (~55 dB mean)
  - Cluster 1 — Normal hearing (N = 44,332): typical control-range thresholds (~30–50 dB)
  - Cluster 2 — Severe hearing loss (N = 562): profoundly elevated thresholds across all frequencies (~85–93 dB)
  - Cluster 3 — High-frequency hearing loss (N = 1,907): normal low-frequency thresholds with elevated 24–30 kHz (~74–83 dB)
