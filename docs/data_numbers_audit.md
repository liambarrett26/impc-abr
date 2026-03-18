# Data Numbers Audit

Last updated: 2026-03-18

Audit of key numbers referenced in the manuscript against values derived from `data/processed/abr_full_data.csv`.

## Sample Flow — Full Funnel

Data from IMPC Release 22.1 and `data/total_number_of_lines_specimens_impc_221.tsv`.

| Stage | Total mice | Mutants | Controls | Genes | Lines/Alleles |
|---|---|---|---|---|---|
| IMPC release 22.1 — all phenotyping (12 centres) | 317,626 | 238,569 | 79,057 | 9,073 genes; 9,774 lines | — |
| 11 centres performing ABR — all phenotyping | 271,856 | 202,493 | 69,363 | 8,560 lines | — |
| ABR data available (our dataset) | **59,145** | 44,431 | 14,714 | 6,749 genes | 7,362 (A+Z) |
| IMPC stats pipeline ran on ABR data | — | — | — | 6,749 genes | — |
| IMPC significant (P < 1×10⁻⁴ at any parameter) | — | — | — | 337 genes (4.99%) | 341 alleles |
| Complete ABR at all 5 frequencies | **56,676** | 42,901 | 13,775 | 6,735 genes | 7,343 (A+Z) |
| Met our analysis inclusion criteria | **55,904** | 42,129 | 13,775 | 6,549 genes | 6,919 (A+Z) |

### What is lost at each stage

- **317,626 → 59,145**: The IMPC phenotypes mice across many pipelines (not just ABR). Of 9,774 mutant lines (9,073 unique genes), 6,749 genes have ABR data. BCM is the only centre that does not perform ABR testing.
- **59,145 → 56,676**: 2,469 mice removed due to incomplete audiograms (missing data at one or more of the 5 frequencies). 1,530 mutants + 939 controls.
- **56,676 → 55,904**: 772 mutant mice removed because their experimental group (allele+zygosity+centre) had \<3 mutants or <20 matched controls. No controls are lost at this step.

### IMPC statistical results (from `/Volumes/IMPC/stat_processed_data.csv`)

The IMPC ran 44,862 individual statistical tests across 6,749 genes (8 ABR parameters × genes tested per parameter). Of these, 722 tests were significant (P < 1×10⁻⁴), spanning 337 unique genes (341 unique alleles). Therefore 337 / 6,749 = **4.99% ≈ 5.0%**.

The old manuscript figure of 3.69% is incorrect — it does not correspond to any identifiable denominator.

| ABR parameter | Genes tested | Genes significant |
|---|---|---|
| IMPC_ABR_002_001 (click) | 4,580 | 67 |
| IMPC_ABR_004_001 (6 kHz) | 6,744 | 125 |
| IMPC_ABR_006_001 (12 kHz) | 6,745 | 120 |
| IMPC_ABR_008_001 (18 kHz) | 6,743 | 126 |
| IMPC_ABR_010_001 (24 kHz) | 6,740 | 120 |
| IMPC_ABR_012_001 (30 kHz) | 6,735 | 142 |

314 genes are significant at one or more of the 5 frequency-specific parameters; the remaining 23 are significant only at the click stimulus.

### Note on "9,774" vs "9,775"

The manuscript states "9,774 mutant lines had been created". The TSV sums to 9,775. This is a rounding or off-by-one discrepancy — verify against the IMPC portal. The corrected figure from IMPC is 9,774 lines encompassing 9,073 phenotyped genes.

## Confirmed Values

| Manuscript placeholder | Value | How derived |
|---|---|---|
| "6,7XX individual gene knockouts" | **6,749** | `gene_symbol.nunique()` on experimental mice |
| "337 genes (XX% of the mutants with ABR data)" | **4.99%** (~5.0%) | 337 / 6,749 |
| Total mice in dataset | **59,145** | Row count — matches manuscript |

## Dataset Breakdown

| Metric | Value |
|---|---|
| Total mice | 59,145 |
| Experimental (mutant) mice | 44,431 |
| Control mice | 14,714 |
| Unique gene symbols (mutants) | 6,749 |
| Unique allele symbols (mutants) | 6,887 |
| Phenotyping centres | 11 |

### Mice per Centre

| Centre | Count |
|---|---|
| JAX | 17,866 |
| CCP-IMG | 9,339 |
| HMGU | 7,043 |
| UC Davis | 6,674 |
| WTSI | 4,943 |
| ICS | 3,851 |
| MRC Harwell | 3,782 |
| TCP | 3,087 |
| MARC | 1,146 |
| KMPC | 863 |
| RBRC | 551 |

### Procedure Split

| Procedure | Mice |
|---|---|
| IMPC_ABR_002 | 53,281 |
| IMPC_ABR_001 | 5,864 |

#### IMPC_ABR_001 Breakdown

| Centre | Mutants | Controls |
|---|---|---|
| WTSI | 3,983 | 960 |
| MRC Harwell | 344 | 400 |
| HMGU | 97 | 80 |

892 unique genes tested under ABR_001.

## Cross-Check Inconsistencies

The multivariate results section of the manuscript states:

> "Of the 59,145 individual observations in IMPC release 22.1, 56,326 met inclusion criteria. This accounted for 7118 alleles, encompassing 6603 unique genes."

Values derived from the current dataset:

| Metric | Manuscript | Derived | Difference |
|---|---|---|---|
| Mice meeting inclusion (complete 5-freq data) | 56,326 | 56,676 | +350 |
| Alleles | 7,118 | 6,872 (allele_symbol) or 7,343 (allele+zygosity) | See note |
| Genes | 6,603 | 6,735 | +132 |

### Notes on Discrepancies

- **56,326 vs 56,676**: The 350-mouse difference suggests additional filtering beyond just complete 5-frequency data was applied when the manuscript numbers were calculated. This may reflect data quality filtering (e.g. aberrant measurements) or an earlier version of the dataset.
- **7,118 alleles**: This number falls between unique `allele_symbol` (6,872) and unique `allele_symbol + zygosity` combinations (7,343). It likely represents allele+zygosity pairs after some exclusion criteria are applied, but the exact definition needs clarifying.
- **6,603 genes**: Similar — 6,735 genes have at least one mouse with complete data, but 6,603 may reflect genes remaining after group-size or control-matching filters.

These discrepancies were flagged by Sherylanne in manuscript comments. A full pass through the batch processor pipeline with logging would resolve which filters produce the stated numbers.

## Inclusion/Exclusion Criteria

### As defined in the manuscript (Methods — Data Source and Experimental Design)

> 1. A minimum of three mutant mice per line
> 2. At least 20 matched wild-type controls
> 3. Complete ABR data at all five tested frequencies

Controls are matched on: genetic background (predominantly C57BL/6N sub-strains), phenotyping centre, pipeline, and equipment specifications.

### As implemented in code

The pipeline applies these criteria at multiple stages:

1. **`loader.py`** — loads all data from CSV, no filtering
2. **`matcher.py`** — defines experimental groups and matching:
   - Groups mutants by `allele_symbol + zygosity + phenotyping_center` (line 74)
   - Excludes groups with < 3 experimental mice (line 90)
   - Matches controls on 5 columns: `phenotyping_center`, `genetic_background`, `pipeline_name`, `metadata_Equipment manufacturer`, `metadata_Equipment model` (lines 45-51)
   - Requires >= 20 matched controls (line 122, `min_controls=20`)
3. **`batch_bayes_processor.py`** / **`batch_processor.py`** — additional NaN filtering:
   - Removes individual profiles with any NaN across the 5 frequencies (lines 99-100)
   - Re-checks >= 3 mutants and >= 20 controls after NaN removal (lines 103-104)

So the effective filtering chain is: raw data → complete 5-freq data (drop NaN rows) → group by allele+zygosity+centre → require >= 3 mutants per group → match controls on 5 metadata columns → require >= 20 matched controls.

**Key finding**: The code matches on `pipeline_name` in addition to the four factors stated in the manuscript. This is an additional constraint not mentioned in the Methods section that could contribute to the number discrepancies.

### Genes excluded by our criteria

| Criterion | Groups failing | Unique genes |
|---|---|---|
| < 20 matched controls (on 5 matching columns) | 8 | 6 |
| < 3 mutant mice with complete data | 436 | — |
| Either criterion fails | 441 | 345 |
| **No valid group exists at all** | — | **189** |

- **6 genes** have groups with zero matched controls at their centre
- **189 genes** have no experimental group that passes both the >=3 mutants AND >=20 controls thresholds

### Manuscript placeholder: "however X genes are currently assigned to a parameter with too few controls"

This sentence refers to the **IMPC's own pipeline** threshold (>=60 controls), not our analysis threshold (>=20). The IMPC requires >=60 controls assigned to the same parameter before performing their statistical analysis. This is a separate criterion from ours and cannot be derived from our local data alone — it requires querying the IMPC portal or their genotype-phenotype assertions data.

## Interpretation of "allele" in the manuscript

The manuscript uses "allele" inconsistently:

- The codebase defines an **experimental group** as `allele_symbol + zygosity + phenotyping_center` — this is the unit of analysis
- "7,118 alleles" likely refers to **allele+zygosity combinations** (we get 7,343 from raw complete data). After some filtering this could reduce to ~7,118
- "6,872" is the count of unique `allele_symbol` values — this doesn't match
- The Bayesian results refer to "139 alleles (133 unique genes)" — here "allele" means allele+zygosity combinations with BF >= 3

Recommendation: define "allele" explicitly in the manuscript Methods section (e.g. "unique combinations of allele symbol and zygosity") and ensure consistent usage throughout.

## Pipeline Step-by-Step Counts

| Step | Mice | Genes | Alleles (symbol) | Alleles (symbol+zygosity) |
|---|---|---|---|---|
| 1. Raw data (mutants only) | 44,431 | 6,749 | 6,887 | — |
| 2. Complete 5-freq data (all mice) | 56,676 | — | — | — |
| 2a. Complete 5-freq (mutants) | 42,901 | 6,735 | 6,872 | 7,343 |
| 2b. Complete 5-freq (controls) | 13,775 | — | — | — |
| 3. >= 3 mutants per group | — | 6,553 | 6,687 | 6,924 |
| 4. + >= 20 matched controls | — | 6,549 | 6,682 | 6,919 |
| 4a. Total mice in valid groups | 55,904 | — | — | — |
| 4b. Mutants in valid groups | 42,129 | — | — | — |
| 4c. Controls in valid groups | 13,775 | — | — | — |

### Comparison with Manuscript

| Metric | Manuscript | Step 2 (complete data) | Step 3 (>= 3 mutants) | Step 4 (+ >= 20 controls) |
|---|---|---|---|---|
| Mice | 56,326 | 56,676 (+350) | — | 55,904 (-422) |
| Alleles (A+Z) | 7,118 | 7,343 (+225) | 6,924 (-194) | 6,919 (-199) |
| Genes | 6,603 | 6,735 (+132) | 6,553 (-50) | 6,549 (-54) |

None of the manuscript numbers exactly match any single pipeline step. They fall between steps, suggesting they were calculated from an earlier run or with slightly different filtering.

## Recommended Manuscript Numbers

Based on the current dataset (`abr_full_data.csv`) and the three stated inclusion criteria ((1) >= 3 mutants, (2) >= 20 matched controls, (3) complete 5-freq data), the correct numbers are:

| Metric | Old (manuscript) | Corrected | Notes |
|---|---|---|---|
| Total mice meeting criteria | 56,326 | **55,904** | 42,129 mutants + 13,775 controls in valid groups |
| Alleles | 7,118 | **6,919** | Unique allele_symbol + zygosity combos, collapsed across centres |
| Genes | 6,603 | **6,549** | Unique gene_symbol in valid groups |

### Definition of "allele"

"Allele" in the manuscript means **allele symbol + zygosity** (e.g., `Nedd4l<tm1b> homozygote` is one allele). This is collapsed across phenotyping centres — the same allele+zygosity tested at multiple centres counts once. 30 allele+zygosity combinations were tested at more than one centre. 347 genes have more than one allele+zygosity combination (e.g. different constructs or het vs hom).

This definition is consistent with the Bayesian results ("139 alleles, 133 genes" = 139 allele+zygosity combos across 133 genes).

### Carry-forward

These numbers should be used consistently throughout the manuscript:
- Multivariate results section: "55,904 met inclusion criteria... 6,919 alleles... 6,549 unique genes"
- Bayesian results section: "Of the 6,919 alleles encompassing 6,549 unique genes, 139 alleles (133 unique genes)..."

## Bayesian vs GMM: Different Inclusion Criteria by Design

The Bayesian/multivariate analyses and the GMM clustering operate on **different subsets** of the data. This is by design but should be made explicit in the manuscript.

### Bayesian / Multivariate Pipeline (55,904 mice)

Inclusion criteria:
1. Complete 5-frequency data (no NaN)
2. Experimental group (allele + zygosity + centre) has >= 3 mutants
3. Matched controls (centre + background + pipeline + equipment) has >= 20 controls

Mice in groups that fail the group-size thresholds are **excluded entirely**. This is necessary because the statistical tests require sufficient sample sizes per group.

### GMM Clustering Pipeline (~56,484 mice on current data)

Inclusion criteria:
1. Complete 5-frequency data (no NaN)
2. Physiologically plausible thresholds: 0-100 dB SPL (removes 192 mice)
3. Critical metadata present (sex, centre, genetic background)

The GMM clusters **all mice** that pass these quality filters — both mutants and controls, regardless of experimental group size. A knockout line with only 2 mutant mice gets excluded from Bayesian analysis, but those 2 mice still get clustered by the GMM. This is the correct approach: the Bayesian analysis requires sufficient samples for per-group inference, while the GMM performs unsupervised pattern discovery across the whole dataset.

### Implication for the Manuscript

The manuscript currently uses a single "met inclusion criteria" count (56,326) for both analyses. This should be revised to state separate counts:
- Bayesian/multivariate: **55,904** mice in valid experimental groups (6,919 alleles, 6,549 genes)
- GMM clustering: total mice with complete, quality-filtered ABR data (to be confirmed after re-run on current dataset; see `docs/gmm_cluster_count_investigation.md`)

The cluster sizes (9,525 / 44,332 / 562 / 1,907) sum to 56,326, which was the GMM training set size. This needs verification on evident-linux to confirm whether it was run on the current or an older version of the dataset.

## Supplementary Data Cross-Check (2026-03-18)

Verified manuscript claims against the supplementary files in `OneDrive/.../impc/supp_mats/`.

### Confirmed matches

| Claim | Supp file | Status |
|---|---|---|
| 337 genes significant by IMPC | Supp 1 | MATCH |
| 4.99% of genes with ABR data | Supp 1 | MATCH |
| 6,919 alleles / 6,549 genes meeting inclusion | Supp 2 & 4 | MATCH |
| 139 alleles (133 genes) with BF >= 3 | Supp 4 sig | MATCH |
| 52 known hearing loss genes | Supp 3a | MATCH |
| Cluster sizes 9,525 / 44,332 / 562 / 1,907 | Supp 5 | MATCH |

### Outstanding mismatches

| Claim | Manuscript | Supp data | Issue |
|---|---|---|---|
| 765 significant alleles (multivariate, q < 0.001) | 765 | 547 (allele+zygosity with q < 0.001 in Supp 2) | Count method differs — 765 may be per experimental group (not collapsed by A+Z), or threshold/counting differed |
| 38 male-only / 39 female-only significant | 38 / 39 | 30 / 37 in Supp 2 | Likely same counting issue as above |
| 46 of 52 known genes retrieved | 46 | 43 in Supp 2 | 3-gene difference — may depend on how sex-specific results are included |
| Gene classification: 47 known / 26 candidate / 60 novel | 47/26/60 | 43/17/79 in Supp 4 sig `gene_classification` column | The `gene_classification` column in `all_sig_results_annotated.csv` uses different Known/Candidate list definitions than the manuscript. Lists need reconciling |

These mismatches relate to (a) how the 765 multivariate significant count is derived (per-group vs collapsed) and (b) the gene classification lists used to annotate Supp 4 sig differing from the manuscript's definitions. Both need resolving before submission.

## Still To Resolve

1. Determine the correct value for "X genes with too few controls" — this refers to the IMPC's own >=60 control threshold, needs querying against IMPC portal data
2. Check whether `pipeline_name` matching is intentional or an oversight — it is in the code but not explicitly named in the manuscript Methods (though "pipeline" is mentioned)
3. Verify GMM training dataset on evident-linux and re-run if needed (see `docs/gmm_cluster_count_investigation.md`)
4. Reconcile the 765 multivariate significant count — clarify whether this is per experimental group or collapsed by allele+zygosity, and ensure Supp 2 matches
5. Reconcile gene classification lists (Known/Candidate) used in Supp 4 sig annotation with the manuscript's definitions (Supp 3a/3b)
