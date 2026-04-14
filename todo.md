# To-do's

## Data

- [ ] **Fill in missing count**: "however X genes are currently assigned to a parameter with too few controls" in Results (IMPC-Reported section)

## Figures & Plots

- [ ] For the enrichment plots:

> "Abnormal ear morphology isn't really a hearing or vestibular term - more craniofacial if anything, it basically just describes if the pinna or visible outer ear looks wrong. "
> "Stereotypic behaviour on the other hand from what I've seen seems to be a now depreciated term which only describes headbobbing or circling - so that is vestibular."
> "basically anything that sounds like they might be head bobbing or circling"
> "I don't think your filtering is wrong, it was just the way they were classified in the plot haha. I think your filtering is rational."
> "Yes, abnormal motor capabilities/coordination/movement is rotorod - so that is vestibular. Might be best for those sorts of plots to leave the classifications Acoustic vs non-acoustic. Any maybe leave the Sankey uncoded."

## Analysis

- [ ] For the enrichment analyses:

> "Is this calculated per mouse or per genotype? [No.]"
> "that would be more impressive than what the last paper did"

## Codebase

- [ ] **GMM cluster counts — verify and likely re-run** (see `docs/gmm_cluster_count_investigation.md`)
  - Cluster sizes sum to 56,326 which doesn't match corrected inclusion count (55,904)
  - GMM pipeline applies extra filters (0-100 dB range) not in Bayesian pipeline, so totals will differ — but need to verify against current dataset
  - On evident-linux: check `shared_data/preprocessing.log` and `normalized_data.npy` shape to confirm what was clustered
  - If dataset mismatch confirmed: re-run `bash run_parallel_gmm.sh -d abr_full_data.csv` and update cluster Ns in manuscript
- [ ] **GMM results**: Copy trained model results from evident-linux to local / ensure they are archived properly (see `docs/data_completeness.md`)
- [ ] **Hardcoded paths in GMM scripts**: Several scripts reference `results/june_23_2025/gmm_k4_tied/` — make configurable or document expected directory structure
- [ ] **Clustering on mutant-only data** — clustering was done on all data including controls, which biases towards a strong "normal hearing" cluster. Consider running a sensitivity analysis on mutant-only data to see if more expressive HL subtypes emerge (e.g. mild HL, low-frequency HL)
