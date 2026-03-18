# To-do's

## Manuscript: Data & Numbers

- [ ] **Fill in missing count**: "6,7XX individual gene knockouts" in Introduction — get exact number from IMPC release 22.1
- [ ] **Fill in missing percentage**: "337 genes (XX% of the mutants with ABR data)" in Introduction — calculate 337/total
- [ ] **Fill in missing count**: "however X genes are currently assigned to a parameter with too few controls" in Results (IMPC-Reported section)
- [ ] **Cross-check numbers across manuscript** — flagged inconsistencies in several places (allele counts, gene counts). Do a full pass to ensure all stated numbers (e.g. 139 alleles / 133 genes / 60 novel) are consistent throughout

## Manuscript: Figures & Plots

- [ ] **Control audiogram by centre plot** — "please could you plot me the control audiogram from each centre?" (referenced as FIGURE?? in Results)
- [ ] **Supplementary figure caption** — add to a supplementary figure caption (likely the Eps8l1/Ikzf5 supplementary figure)

## Manuscript: Data Queries from Sherylanne

- [ ] **IMPC_ABR_001 mutant list** — "a total list of mutants with data collected under procedure_stable_id:IMPC_ABR_001 and the number of WTs on IMPC_ABR_001 per centre" — likely only Harwell & WTSI but needs checking
- [ ] **Allele discrepancy examples** — cases where: tm1a significant but tm1b not (and vice versa); multi-centre observations of HL genes — do all centres confirm deafness? Specific genes to check: Gm5148 (Het sig/Hom not), Lrp1b (Het sig/Hom not), Phf3 (em2@MARC sig/em1@Harwell not), Mkrn2 (em1 not sig/tm1b sig), Tmprss2 (em1@CCP-IMG not sig/tm2b@ICS sig)

## Manuscript: Analytical Considerations

- [ ] **Multivariate significance threshold sensitivity** — change the threshold for the 6 genes missed by multivariate (Asic3, Gjb3, Nrp1, Rest, Tbl1xr1, Gata2). Gata2 narrowly failed (P=0.0014). Consider running at relaxed threshold and documenting the effect on total significant alleles
- [ ] **Completeness check on sex-specific Bayesian analyses** — these need running for completeness, but cautioned about retrofitting. Ensure combined + male-only + female-only are all complete and consistent

## Codebase

- [ ] **GMM results**: Copy trained model results from evident-linux to local / ensure they are archived properly (see `docs/data_completeness.md`)
- [ ] **Hardcoded paths in GMM scripts**: Several scripts reference `results/june_23_2025/gmm_k4_tied/` — make configurable or document expected directory structure
- [ ] **Clustering on mutant-only data** — clustering was done on all data including controls, which biases towards a strong "normal hearing" cluster. Consider running a sensitivity analysis on mutant-only data to see if more expressive HL subtypes emerge (e.g. mild HL, low-frequency HL)
