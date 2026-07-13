# Per-Mouse Enrichment Analysis

## What is it and how does it differ from per-gene?

### Per-gene enrichment (what we currently do)

The unit of observation is the **gene**. For each of ~6,549 genes, we have:
- A binary classification: hearing loss (BF >= 3) or not
- A set of MP term annotations: gene-level assertions from the IMPC statistical pipeline (e.g. "Aak1 has abnormal startle reflex")

The enrichment question is: *are hearing loss genes more likely than other genes to carry MP term X?*

This is the standard approach used by Vicencio-Jimenez et al. and in our current analysis. The limitation is that it treats each gene as a single unit — ignoring within-gene variation (some mice in a knockout line may have worse hearing or worse behavioral scores than others) and ignoring continuous information (the magnitude of hearing loss or the severity of the behavioral phenotype).

### Per-mouse enrichment (proposed)

The unit of observation is the **individual mouse**. For each of ~56,000+ mice with ABR data, we would ask:
- Does this specific mouse have elevated ABR thresholds?
- Does this same mouse also show abnormal values on other IMPC phenotyping tests?

The enrichment question becomes: *are individual mice with worse hearing more likely to individually show abnormal behavior, bone density, metabolism, etc.?*

This is more powerful because:

1. **Within-gene variation is captured.** Not all mice of a knockout line show the same phenotype. Some Aak1 knockouts may have severe hearing loss and behavioral issues, while others in the same line may be unaffected. Per-gene analysis averages over this.

2. **Continuous hearing data is used.** Instead of a binary BF >= 3 cutoff, we can use each mouse's actual ABR thresholds (or PTA — pure tone average). This preserves information about hearing loss severity.

3. **Direct co-occurrence within individuals.** Per-gene analysis links a gene's hearing phenotype to the gene's behavioral phenotype — but these may come from different mice within the line (though likely the same, given IMPC's pipeline). Per-mouse analysis ensures the hearing measurement and the behavioral measurement come from the same animal.

4. **Larger sample size.** ~56,000 mice vs ~6,549 genes. This gives substantially more statistical power, particularly for detecting subtle associations.

5. **Novelty.** Vicencio-Jimenez et al. performed per-gene enrichment. A per-mouse analysis would be genuinely novel and, as Sherri noted, "more impressive than what the last paper did."

---

## What does the data look like?

### Data we have (per-mouse ABR)

Our `abr_full_data.csv` contains one row per mouse with:
- `specimen_id` — unique mouse identifier
- `gene_symbol`, `allele_symbol`, `zygosity` — genotype
- `phenotyping_center`, `sex` — covariates
- ABR thresholds at 6, 12, 18, 24, 30 kHz — continuous hearing measurements
- `biological_sample_group` — experimental vs control

### Data we need (per-mouse non-ABR phenotyping)

The IMPC `experiment` Solr core contains individual mouse measurements across all phenotyping procedures. Each record includes:
- `specimen_id` — same identifier as in ABR data, allowing cross-procedure linkage
- `procedure_name`, `parameter_name` — what was measured
- `data_point` — the actual measurement value
- `observation_type` — unidimensional (continuous), categorical, etc.

Confirmed: the same `specimen_id` appears across multiple procedures — a mouse tested for ABR is typically also tested in Open Field, SHIRPA, grip strength, clinical chemistry, etc. (14 of 16 specimens for Aak1 appeared in multiple procedures.)

### What we would need to fetch

For all mice in our ABR dataset (~56,000 specimens), fetch their individual measurements from other IMPC procedures. Key procedures of interest:

| Procedure | What it measures | Observation type |
|---|---|---|
| Open Field | Locomotor activity, anxiety, exploration | Continuous |
| Acoustic Startle & PPI | Startle reflex, sensorimotor gating | Continuous |
| Combined SHIRPA | Reflexes, gait, posture, neurological screen | Categorical + continuous |
| Grip Strength | Forelimb/hindlimb strength | Continuous |
| Fear Conditioning | Learning, memory, freezing behavior | Continuous |
| Clinical Chemistry | Blood biochemistry (cholesterol, etc.) | Continuous |
| DEXA | Bone mineral density, body composition | Continuous |
| Hematology | Blood cell counts | Continuous |
| Rotarod | Motor coordination/balance | Continuous |

---

## Analytical approaches

### Approach 1: Per-mouse Fisher's exact test (binary × binary)

The most direct per-mouse analogue of our current per-gene analysis.

**Hearing status (binary):** Classify each mouse as "hearing impaired" or "normal hearing" based on ABR thresholds. Options:
- Use the GMM cluster assignment (cluster 1/3/4 = impaired, cluster 2 = normal)
- Use a PTA (pure tone average) threshold (e.g. > 40 dB = impaired)
- Use the Mahalanobis distance from the control distribution

**Phenotype status (binary):** For each non-ABR parameter, classify each mouse as "abnormal" or "normal". Options:
- Use z-score relative to centre-matched controls (e.g. |z| > 2 = abnormal)
- Use the IMPC's own reference range method

**Test:** For each parameter, construct a 2×2 table (hearing impaired × phenotype abnormal) and run Fisher's exact test. FDR correct across all parameters tested.

**Pros:** Direct parallel to per-gene analysis. Easy to interpret.
**Cons:** Requires choosing thresholds for both hearing and phenotype status. Discards continuous information.

### Approach 2: Per-mouse correlation (continuous × continuous)

More powerful — uses the full continuous data without binning.

**Hearing measure (continuous):** Each mouse's PTA or individual frequency thresholds.

**Phenotype measure (continuous):** Each mouse's raw measurement on each non-ABR parameter (e.g. distance travelled in open field, grip strength in grams, cholesterol in mg/dL).

**Test:** For each parameter, compute the Spearman correlation between ABR threshold and the parameter value, across all mice that have both measurements. Correct for centre effects (include centre as a covariate, or compute within-centre correlations).

This is similar to what Vicencio-Jimenez do (their Figure 3: ABR threshold vs number of MP abnormalities, Figure 5: ABR threshold vs number of behavioral abnormalities), but at the individual mouse level rather than the gene level.

**Pros:** Uses full continuous data. No arbitrary thresholds. Higher statistical power.
**Cons:** Confounded by genotype — mice from the same knockout line share both hearing loss and behavioral phenotypes because they share the same genetic perturbation. Need to account for this (see challenges below).

### Approach 3: Mixed-effects model (recommended)

The most rigorous approach, addressing the key confound of genotype clustering.

**Model:** For each non-ABR parameter Y:

```
Y_ij = β₀ + β₁ × ABR_ij + β₂ × sex_ij + β₃ × centre_ij + u_j + ε_ij
```

Where:
- `i` = individual mouse, `j` = gene/allele group
- `ABR_ij` = PTA or hearing measure for mouse i in gene group j
- `u_j` = random intercept for gene group (accounts for genotype clustering)
- `β₁` = the key parameter — does individual hearing predict individual phenotype, beyond gene-level effects?

**Why mixed effects?** Mice within the same knockout line are not independent — they share the same genetic perturbation. A naive correlation would be inflated because all mice from a deaf knockout line have both high ABR thresholds AND (likely) abnormal behavioral measures. The random effect for gene group absorbs this between-gene variation, and `β₁` captures whether *within a gene group*, mice with worse hearing also have worse behavioral scores.

This would be genuinely novel — no one has done within-genotype hearing-behavior correlations at this scale.

**Pros:** Addresses the genotype clustering confound. Most statistically rigorous. Novel.
**Cons:** Most complex to implement. Requires sufficient within-gene variation (which exists — our Bayesian analysis explicitly models the proportion of affected mice within each line). May have limited power for genes with very small sample sizes (typically 4-8 mice per line).

---

## Key challenges

### 1. Data volume

Fetching per-mouse data for ~56,000 specimens across ~15 procedures would be a large API query. The IMPC experiment core has millions of records. We would need to:
- Query by `specimen_id` for our specific mice
- Or query by `gene_symbol` for our genes and filter to matching specimens
- Cache aggressively

### 2. Genotype confounding (Approach 2)

A naive per-mouse correlation between ABR threshold and (say) open field activity would be dominated by between-genotype differences. All mice from a profoundly deaf knockout line have high ABR and (likely) abnormal startle — this is the gene effect, not an individual mouse effect. The mixed-effects approach (Approach 3) addresses this.

### 3. Centre effects

ABR thresholds vary by centre. So do behavioral measures. Any per-mouse analysis must account for centre, either by:
- Including centre as a fixed effect in the model
- Normalising measurements within centre before analysis
- Restricting to within-centre comparisons

### 4. Missing data

Not all mice have data for all procedures. The overlap between ABR and (say) Open Field may be smaller than the full ABR dataset. Need to document the effective sample size for each per-mouse comparison.

### 5. The acoustic circularity problem (same as per-gene)

Per-mouse startle reflex and PPI correlations with ABR are expected and circular — a deaf mouse can't hear the stimulus. The same acoustic filtering logic applies: focus on non-acoustic parameters (open field, grip strength, clinical chemistry, DEXA, fear conditioning).

### 6. Multiple testing

Testing many parameters across many procedures generates a large multiple testing burden. FDR correction across all parameters, or structure the analysis hierarchically (test at procedure level first, then drill into parameters for significant procedures).

---

## What would the outputs look like?

### Per-mouse Fisher's (Approach 1)
- Table: parameter name, n_mice_tested, OR, fold enrichment, p_adj
- Similar structure to our current per-gene results but with mouse-level sample sizes

### Per-mouse correlation (Approach 2)
- Table: parameter name, n_mice, Spearman rho, p_value, p_adj
- Scatter plots: ABR threshold vs parameter value for key significant associations

### Mixed-effects model (Approach 3)
- Table: parameter name, n_mice, n_genes, β₁ (ABR effect), SE, p_value, p_adj
- Interpretation: β₁ > 0 means individual mice with worse hearing have higher values of parameter Y, beyond what's explained by their genotype

---

## Relationship to per-gene analysis

The per-gene and per-mouse analyses answer complementary questions:

| | Per-gene | Per-mouse |
|---|---|---|
| **Unit** | Gene (genotype) | Individual mouse |
| **Question** | Are HL genes enriched for other phenotypes? | Do individual mice with worse hearing show worse other phenotypes? |
| **Hearing measure** | Binary (BF >= 3 or not) | Continuous (ABR thresholds) |
| **Phenotype measure** | Binary (MP term present or not) | Continuous (raw measurements) |
| **Sample size** | ~6,549 genes | ~56,000 mice |
| **Confound** | N/A (each gene is independent) | Genotype clustering (mice share genes) |
| **Novelty** | Standard (Vicencio-Jimenez did this) | Novel (no one has done this at IMPC scale) |

Both should be reported. The per-gene analysis provides the validated, comparable result. The per-mouse analysis provides the novel, more powerful extension.

---

## Implementation considerations

### Data to fetch
- All experiment-level (per-mouse) data from the IMPC experiment core for specimens in our ABR dataset
- Query: `specimen_id` in our specimen list, across all non-ABR procedures
- Cache to local JSON/CSV as with the genotype-phenotype data

### Code structure
- New module: `scripts/abr_analysis/enrichment_per_mouse/` (separate from per-gene `enrichment/`)
- Share acoustic filtering logic and configuration with per-gene pipeline
- Separate results directories

### Recommended starting point
- Start with Approach 1 (binary × binary Fisher's) for direct comparability with per-gene results
- Then add Approach 2 (continuous correlation) for the novel contribution
- Approach 3 (mixed effects) as a sensitivity/robustness check if time permits
