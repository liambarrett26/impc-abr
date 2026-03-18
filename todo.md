# To-do's

## Manuscript: Data & Numbers

- [X] **Fill in missing count**: "6,7XX individual gene knockouts" in Introduction — get exact number from IMPC release 22.1
- [X] **Fill in missing percentage**: "337 genes (XX% of the mutants with ABR data)" in Introduction — calculate 337/total
- [ ] **Fill in missing count**: "however X genes are currently assigned to a parameter with too few controls" in Results (IMPC-Reported section)
- [X] **Cross-check numbers across manuscript** — flagged inconsistencies in several places (allele counts, gene counts). Do a full pass to ensure all stated numbers (e.g. 139 alleles / 133 genes / 60 novel) are consistent throughout
- [X] Consider whether we include total number of mice in release 22.1, number with ABR thresholds and the number who met inclusion criteria.

## Manuscript: Figures & Plots

- [X] **Control audiogram by centre plot** — "please could you plot me the control audiogram from each centre?" (referenced as FIGURE?? in Results)
    - Available under `/Users/liambarrett/Library/CloudStorage/OneDrive-UniversityCollegeLondon/evident/projects/impc/figures/legacy/control_data_by_center_and_sex.png`
- [X] **Supplementary figure caption**.
    - Added:
    >"Supplementary Figure 2, Bayesian mixture model analysis of IMPC ABR data for (a) Nedd4ltm1b(KOMP)Wtsi homozygous and (b) Tmem51tm1b(EUCOMM)Hmgu homozygous mice. For each gene, three panels are shown. Left: ABR threshold profiles (dB SPL) across the five tested frequencies (6, 12, 18, 24, and 30 kHz) for control (blue, n = 858 and n = 323 respectively) and mutant (red, n = 5 and n = 4 respectively) mice. Solid lines show group means; shaded regions represent 95% confidence intervals. Centre: Posterior estimates of the hearing loss effect size (dB) at each frequency, with error bars indicating 95% credible intervals from the Bayesian model. Right: Posterior distribution of the probability of hearing loss, estimated from the Bayesian mixture model. The solid vertical line indicates the posterior mean, and dashed lines indicate the 94% highest density interval (HDI). Both genes show elevated mutant thresholds across all frequencies, with posterior mean probabilities of hearing loss of 0.666 (HDI: 0.389–0.919) for Nedd4l and 0.626 (HDI: 0.335–0.914) for Tmem51, supporting their classification as strong candidates for auditory dysfunction."

## Manuscript: Data Queries from Sherylanne

- [X] **IMPC_ABR_001 mutant list** — "a total list of mutants with data collected under procedure_stable_id:IMPC_ABR_001 and the number of WTs on IMPC_ABR_001 per centre" — likely only Harwell & WTSI but needs checking
    - The list of genes, split by allele/zygosity/centre is available at `data/processed/impc_abr_001_mutant_list.csv`.
- [X] **Allele discrepancy examples** — cases where: tm1a significant but tm1b not (and vice versa); multi-centre observations of HL genes — do all centres confirm deafness? Specific genes to check: Gm5148 (Het sig/Hom not), Lrp1b (Het sig/Hom not), Phf3 (em2@MARC sig/em1@Harwell not), Mkrn2 (em1 not sig/tm1b sig), Tmprss2 (em1@CCP-IMG not sig/tm2b@ICS sig)

## Manuscript: Analytical Considerations

- [X] **Multivariate significance threshold sensitivity** — change the threshold for the 6 genes missed by multivariate (Asic3, Gjb3, Nrp1, Rest, Tbl1xr1, Gata2). Gata2 narrowly failed (P=0.0014). Consider running at relaxed threshold and documenting the effect on total significant alleles
    - Decided to simply argue the case that multivariate is non optimal and Bayesian should be preferred.
- [X] **Completeness check on sex-specific Bayesian analyses** — these need running for completeness, but cautioned about retrofitting. Ensure combined + male-only + female-only are all complete and consistent

## Codebase

- [ ] **GMM cluster counts — verify and likely re-run** (see `docs/gmm_cluster_count_investigation.md`)
  - Cluster sizes sum to 56,326 which doesn't match corrected inclusion count (55,904)
  - GMM pipeline applies extra filters (0-100 dB range) not in Bayesian pipeline, so totals will differ — but need to verify against current dataset
  - On evident-linux: check `shared_data/preprocessing.log` and `normalized_data.npy` shape to confirm what was clustered
  - If dataset mismatch confirmed: re-run `bash run_parallel_gmm.sh -d abr_full_data.csv` and update cluster Ns in manuscript
- [ ] **GMM results**: Copy trained model results from evident-linux to local / ensure they are archived properly (see `docs/data_completeness.md`)
- [ ] **Hardcoded paths in GMM scripts**: Several scripts reference `results/june_23_2025/gmm_k4_tied/` — make configurable or document expected directory structure
- [ ] **Clustering on mutant-only data** — clustering was done on all data including controls, which biases towards a strong "normal hearing" cluster. Consider running a sensitivity analysis on mutant-only data to see if more expressive HL subtypes emerge (e.g. mild HL, low-frequency HL)
