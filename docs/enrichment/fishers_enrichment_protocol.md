# Protocol A: Gene Set Enrichment via Fisher's Exact Test

## Overview

This protocol formalises Sherri's enrichment analysis workflow. The goal is to determine whether genes identified as auditory phenodeviants (from the Bayesian ABR analysis) are enriched for other phenotypic categories in the IMPC dataset — particularly behavioural, neurological, and skeletal phenotypes.

This uses the classical Fisher's exact test with FDR correction, consistent with standard gene set enrichment methodology (e.g., as used by g:Profiler, DAVID, etc.).

---

## Inputs

- **Auditory phenodeviant gene list**: Alleles with Bayes Factor ≥ 3 from the Bayesian mixture model analysis (whole-population analysis; sex not considered at this stage). The current count is 139 alleles.
- **Background gene set**: All genes analysed in the Bayesian pipeline (~7,000+ genes with complete ABR data and matched controls).
- **Phenotype annotations**: Mammalian Phenotype (MP) ontology terms associated with each gene, sourced from IMPC statistical results. These should be matched by centre — i.e., only include MP calls made at the same centre where the ABR data were collected for that allele.
- **MP ontology structure**: The MP ontology hierarchy (mp.obo or equivalent) to map specific terms to top-level categories.

---

## Pre-processing

### 1. Centre matching of phenotype calls

For each gene in the auditory phenodeviant list, restrict associated MP terms to those called at the same phenotyping centre where the ABR data were generated. This controls for centre-specific effects in non-ABR phenotyping.

**Note**: Sherri flagged this as a potential issue — if a phenotype was called at a different centre than the ABR data, it should not be associated.

**Decision**: Centre matching is run as a **sensitivity analysis**, not the primary analysis. The primary analysis uses all MP assertions regardless of centre. Rationale: 96.4% of assertions are centre-matched anyway (the IMPC pipeline performs most phenotyping at the same centre), and the matched/unmatched results are substantively identical. Enforcing strict matching as primary would exclude valid data from the small number of genes phenotyped across centres.

### 2. Exclusion of ABR-related MP terms

Remove all MP terms directly related to auditory brainstem response or hearing thresholds from the analysis. These are expected to be enriched by definition and would inflate results. This includes any terms under the top-level "hearing/vestibular/ear" category that relate specifically to ABR measurements.

**Decision**: Only **acoustic-dependent** terms are removed in the filtered analysis — those where the test requires the mouse to hear a stimulus (ABR, startle, PPI, pinna reflex). Vestibular terms (head bobbing, trunk curl, circling) are **retained but flagged separately** as `vestibular` in the classification. Rationale: vestibular phenotypes represent genuine biological co-morbidity (shared inner ear structures) rather than circular reasoning. They confirm cochlear/vestibular pathology co-occurrence, which is biologically informative. The three-way classification (acoustic-dependent / vestibular / independent) lets readers assess each category on its own terms.

### 3. Filtering of MP terms with zero overlap

Remove MP terms where none of the auditory phenodeviant alleles have any association. These contribute nothing to the analysis and inflate the multiple testing burden.

---

## Stage 1: Top-Level MP Term Enrichment

### Purpose

Identify which broad phenotypic domains are over-represented among auditory phenodeviant genes.

### Procedure

For each top-level MP term (e.g., "nervous system phenotype", "behavior/neurological phenotype", "skeleton phenotype", etc.):

1. Construct a 2×2 contingency table:

|  | Has top-level MP term | Does not have term |
|--|--|--|
| **Auditory phenodeviant (BF ≥ 3)** | a | b |
| **Not auditory phenodeviant** | c | d |

Where:
- `a` = number of auditory phenodeviant genes with this top-level MP term
- `b` = number of auditory phenodeviant genes without this term
- `c` = number of non-phenodeviant genes with this term
- `d` = number of non-phenodeviant genes without this term
- Row totals: `a + b` = total auditory phenodeviants; `c + d` = total non-phenodeviants
- Column totals: `a + c` = total genes with this MP term; `b + d` = total genes without

2. Run a **two-sided Fisher's exact test** on the table.

3. Compute the **odds ratio**: `OR = (a × d) / (b × c)`

4. Compute **fold enrichment**: `(a / (a + b)) / ((a + c) / (a + b + c + d))`
   i.e., observed proportion in phenodeviant set ÷ expected proportion from background.

5. Record: top-level MP term name, a, b, c, d, total calls in background, calls in auditory alleles, odds ratio, fold enrichment, raw p-value.

### Multiple testing correction

Apply **Benjamini-Hochberg FDR correction** across all top-level MP terms tested. Use a corrected significance threshold of p_adj < 0.05.

---

## Stage 2: Full MP Hierarchy Enrichment (Sub-Branch Analysis)

### Purpose

For domains found significant in Stage 1, drill down through the **entire MP hierarchy** — not just leaf terms — to identify which specific terms are driving the enrichment and, critically, to assess whether the enrichment is circular or genuinely informative.

**This is essential.** Top-level enrichment (e.g., "behavior/neurological phenotype" is enriched) tells us very little on its own. That category contains everything from "absent pinna reflex" (a trivial consequence of deafness) to "abnormal learning" (a potentially meaningful co-morbidity). Without traversing the sub-branches, we cannot distinguish signal from circularity.

### Procedure

1. Take the top-level MP categories that were significant in Stage 1 (e.g., "nervous system phenotype" and "behavior/neurological phenotype").

2. Using the MP ontology hierarchy (mp.obo), extract **all descendant terms** at every level — intermediate grouping terms and leaf terms alike. This means testing not just "decreased startle reflex" but also its parent "abnormal startle reflex" and any sibling/cousin terms at the same level.

3. Filter to retain only terms where at least one auditory phenodeviant gene has the term.

4. For each term at every level, repeat the same 2×2 Fisher's exact test procedure as Stage 1.

5. Apply **Benjamini-Hochberg FDR correction** across all terms tested within this stage.

6. **Classify each enriched term** into one of three categories (see Circularity Assessment below).

7. **Re-run the Stage 1 top-level test after excluding acoustic-dependent terms.** If top-level enrichment for "behavior/neurological phenotype" disappears after removing acoustic confound terms, the original finding was circular.

### Expected output

A table with an additional classification column:

| MP term | MP level | Parent term | Total calls | Calls in auditory alleles | Odds ratio | Fold enrichment | P.adj | Classification |
|--|--|--|--|--|--|--|--|--|

Where Classification is one of: `acoustic-dependent`, `vestibular`, `independent`, `unclear`.

---

## Circularity Assessment

### The problem

We identify hearing loss genes using ABR thresholds. Many MP terms classified under behavioral or nervous system categories are themselves dependent on auditory function. Finding that hearing loss genes are "enriched for behavioral phenotypes" may simply reflect that deaf mice fail acoustic-dependent tests — not that the gene has any independent effect on behavior. This is circular reasoning: the enrichment tells us nothing we didn't already know from the ABR analysis.

### The chain of circularity

```
Our analysis identifies gene as causing hearing loss (via ABR thresholds)
    → Mouse is deaf/hearing impaired
        → Mouse fails to respond to acoustic stimulus in behavioral test
            → IMPC assigns "absent pinna reflex" or "decreased startle reflex"
                → These sit under "nervous system" or "behavior/neurological" MP terms
                    → We find "enrichment" for nervous system / behavioral phenotypes
                        → But this is just hearing loss detected by a different test
```

### Classification of enriched terms — two-pass approach

A keyword-only classifier is insufficient because intermediate/parent MP terms (e.g. "abnormal reflex", "abnormal synaptic transmission") can have non-acoustic names but be driven almost entirely by acoustic test data. We therefore use a two-pass classification:

**Pass 1 — Keyword/ontology classification**: Classifies terms based on their name and position in the MP ontology hierarchy.

**Pass 2 — Procedure-based classification**: For terms classified as "independent" in Pass 1, checks what fraction of foreground genes' evidence for that term comes from acoustic-dependent procedures. If >= 80% of genes' evidence is from acoustic procedures (ACS, ABR, CSD pinna reflex) and N >= 5, reclassifies to acoustic-dependent.

Pass 2 catches terms like:
- "Abnormal reflex" (96% of foreground genes' evidence from startle/pinna procedures)
- "Abnormal synaptic transmission" (97% from PPI/acoustic startle)
- "Abnormal involuntary movement" (90% from startle-dominated procedures)
- "Limb grasping" (88% from CSD reflex assessment)

These terms would be misreported as independent neurological enrichments without the procedure-based check.

#### Classification categories

**Acoustic-dependent**: The test that generates this MP term uses an acoustic stimulus, and the phenotype would be expected in any deaf mouse regardless of other gene function, OR the term's evidence in our foreground genes is >= 80% derived from acoustic procedures. Examples:
- Absent pinna reflex (the Preyer reflex is a direct response to sound)
- Decreased startle reflex (acoustic startle requires hearing the stimulus)
- Abnormal startle reflex
- Decreased prepulse inhibition (PPI stimuli are acoustic)
- Abnormal reflex (96% from startle/pinna — caught by Pass 2)
- Abnormal synaptic transmission (97% from PPI — caught by Pass 2)
- Abnormal nervous system physiology (97% from PPI — caught by Pass 2)

**Vestibular**: The phenotype likely reflects inner ear vestibular dysfunction rather than CNS function. Many hearing loss genes affect both cochlear and vestibular structures. Examples:
- Head bobbing
- Circling
- Trunk curl
- Abnormal ear morphology

**Independent**: The phenotype is assessed via a non-acoustic test, does not have an obvious vestibular explanation, and is not procedure-driven by acoustic tests. These are the genuinely informative co-morbidities. Examples:
- Hyperactivity (open field — 0% acoustic)
- Decreased anxiety-related response (open field / light-dark — 0% acoustic)
- Abnormal associative learning (fear conditioning — 0% acoustic)
- Decreased grip strength (grip strength test — 0% acoustic)
- Decreased bone mineral density (DEXA — 0% acoustic)
- Abnormal cholesterol level (clinical chemistry — 0% acoustic)

**Unclear**: Does not match any of the above rules. Not triggered by any term in the current analysis.

### Filtered enrichment analysis — assertion-level acoustic removal

Rather than removing whole MP terms based on a classification threshold, we filter at the **individual assertion level** using `acoustic_filter.py`. This is more principled: terms that are partially acoustic-dependent retain their non-acoustic gene evidence, while terms that are entirely acoustic-derived naturally drop to zero foreground genes without any arbitrary cutoff.

Filtering rules:
1. **Fully acoustic procedures** (ACS, ABR, ESLIM_011): all assertions removed
2. **Mixed procedures** (CSD/SHIRPA): only assertions with acoustic-specific MP terms (pinna reflex, startle) removed; non-acoustic assertions (gait, coat morphology) retained
3. **All other procedures**: all assertions retained

After filtering, gene-to-MP mappings are rebuilt from scratch and both top-level and hierarchical enrichment are re-run. The analysis reports:

1. **Full results** (unfiltered, all assertions) — for comparability with Vicencio-Jimenez et al.
2. **Filtered results** (acoustic assertions removed at source) — the genuinely informative analysis
3. **The delta** between (1) and (2) — quantifying circularity per term, with gene count changes showing exactly where acoustic evidence was removed

The circularity labels from the classifier (Stage 3) are retained on the unfiltered results for interpretive purposes but do not determine what is removed in the filtered analysis — the filtering is purely procedure-based.

---

## Assessing the Statistical Quality of MP Term Assignments

### The problem

Each MP annotation in IMPC is the result of a statistical test comparing knockout mice to matched controls on a phenotyping measure. These underlying tests may suffer from the same limitations that motivated our advanced Bayesian framework for ABR data.

### What to check

For each enriched MP term, determine:

1. **What IMPC test/parameter generated this MP term?** The IMPC provides mappings from MP terms back to specific procedures and parameters (e.g., IMPC_ABR_002 for click-evoked ABR threshold, IMPC_ACS_001 for acoustic startle). Document these for all enriched terms.

2. **Is the underlying test univariate or multivariate?** Most IMPC statistical analyses test each parameter independently — the same frequency-by-frequency approach we argue is suboptimal for ABR. Phenotyping procedures that produce multiple related measurements (e.g., PPI at multiple prepulse intensities, open field activity across time bins) may be reduced to single summary statistics before testing, potentially missing coordinated patterns. Note which enriched terms come from univariate vs. potentially multivariate tests.

3. **How was the statistical call made?** IMPC uses reference range methods (similar to the Bowl et al. approach we replicate as our classical baseline) for some parameters, and mixed model approaches for others. Some calls involve operator judgment. Document the statistical method for each enriched term where possible.

4. **Is the test well-powered for the sample sizes available?** IMPC typically phenotypes ≥7 males and ≥7 females per line. For some tests this may be adequate; for others (especially those measuring subtle behavioral effects) it may not be.

### Why this matters

If an enriched MP term was assigned by a robust, well-powered, objective test (e.g., body composition by DEXA, blood chemistry), that enrichment is more credible than one assigned by a subjective or low-power test. Conversely, if we find that our Bayesian gene list shows different enrichment patterns than the Vicencio-Jimenez list (derived from IMPC's standard annotations), the difference may partly reflect how genes were identified — our multivariate Bayesian approach captures coordinated patterns that IMPC's univariate pipeline misses.

This is an opportunity for the manuscript: we can frame the enrichment analysis as not just "what do hearing loss genes do?" but "what do hearing loss genes — identified by a more principled statistical approach — tell us about broader phenotypic associations?"

---

## Interpretation Guidelines

### Acoustic confounds

Many behavioural MP terms involve responses to sound (startle reflex, pinna reflex, prepulse inhibition). Enrichment of these among hearing loss genes may reflect loss of acoustic sensitivity rather than genuine behavioural co-morbidity. Flag these in the results and interpret accordingly. See Circularity Assessment above for the full classification scheme.

### Vestibular confounds

Terms such as "head bobbing", "circling", "trunk curl", and "stereotypic behavior" (which maps to "head bobbing/circling" in the IMPC parameter space) likely reflect vestibular dysfunction. Many genes causing hearing loss also affect vestibular function due to shared inner ear structures. These should be interpreted as inner ear phenotypes, not cognitive or emotional co-morbidities.

### Skeleton phenotypes

Sherri noted she hadn't yet investigated what drives the skeleton enrichment. This should be examined — some hearing loss genes affect bone development (e.g., craniofacial, ossicle development) which could explain this association.

---

## Sensitivity Analyses

1. **Vary the BF threshold**: Repeat with BF ≥ 10 (strong evidence) and BF ≥ 30 (very strong evidence) to see if enrichment patterns are consistent or threshold-dependent.

2. **Sex-specific analysis**: Repeat using sex-specific BF results (male-only and female-only phenodeviants) to check for sex-specific co-morbidity patterns.

3. **Centre matching stringency**: Compare results with and without strict centre matching of MP term calls.

4. **Expand to full allele set**: Sherri noted the 129 alleles could be expanded. Consider including alleles with lower BF that still show some evidence.

---

## Implementation

### Code location

```
scripts/abr_analysis/enrichment/
    __init__.py
    config.py               # Paths, API URLs, thresholds, keyword lists
    fetch_data.py            # Data acquisition: API, OBO parsing, gene sets
    enrichment_test.py       # Fisher's exact tests + BH FDR correction
    circularity.py           # Acoustic/vestibular/independent term labelling
    acoustic_filter.py       # Assertion-level acoustic filtering
    centre_matching.py       # Centre-matched gene-to-MP mappings
    visualise.py             # Sankey, dot plot, circularity comparison charts
    run_enrichment.py        # Main orchestrator (entry point)
    data/                    # Cached downloads (gitignored)
        genotype_phenotype_all.json   # 67,350 IMPC assertions
        mp.obo                        # MP ontology file
    results/                 # Output CSVs
```

### Execution

```bash
cd scripts/abr_analysis
python -m enrichment.run_enrichment
```

### Data acquisition (`fetch_data.py`)

**IMPC genotype-phenotype assertions**:
- Source: IMPC Solr API `genotype-phenotype` core (`https://www.ebi.ac.uk/mi/impc/solr/genotype-phenotype/select`)
- Paginated fetch: 5,000 records per request, 0.5s delay between pages
- Total: 67,350 records (Data Release 22.1)
- Fields retrieved: `marker_symbol`, `allele_symbol`, `phenotyping_center`, `procedure_name`, `procedure_stable_id`, `parameter_name`, `parameter_stable_id`, `top_level_mp_term_id`, `top_level_mp_term_name`, `intermediate_mp_term_id`, `intermediate_mp_term_name`, `mp_term_id`, `mp_term_name`, `zygosity`, `p_value`, `effect_size`, `life_stage_name`, `sex`
- Cached to `data/genotype_phenotype_all.json` after first fetch
- Note: `top_level_mp_term_id`, `intermediate_mp_term_id` are multi-valued arrays in Solr — a single assertion can map to multiple top-level categories

**MP ontology**:
- Source: `http://purl.obolibrary.org/obo/mp.obo`
- Parsed with custom OBO parser (no `pronto` dependency): extracts `id`, `name`, `is_a` relationships, `is_obsolete` flag
- Builds parent-child graph: 14,695 terms, 19,487 parent-child relationships
- `MPOntology` class provides `get_all_descendants(term_id)` and `get_all_ancestors(term_id)` via BFS traversal
- Cached to `data/mp.obo`

**Gene sets**:
- Foreground: 133 unique genes from `supplementary_file_4_bayesian_results_significant_annotated.csv` (alleles with BF >= 3)
- Background: 6,549 unique genes from `supplementary_file_4_bayesian_results_all.csv` (all genes meeting ABR inclusion criteria)
- Gene-to-centre mapping extracted from background file `center` column for centre-matching

**Gene-to-MP mappings** (built from genotype-phenotype data, restricted to background genes):
- `gene_to_top_mp`: {gene_symbol: set of top-level MP term IDs}
- `gene_to_all_mp`: {gene_symbol: set of all MP term IDs (leaf + intermediate)}
- `gene_to_mp_by_centre`: {(gene_symbol, centre): set of MP term IDs}
- `mp_to_procedures`: {mp_term_id: set of procedure_stable_ids}

### Fisher's exact test (`enrichment_test.py`)

For each MP term, constructs a 2x2 contingency table:

|  | Has term | No term |
|--|--|--|
| **Foreground (BF >= 3)** | a | b |
| **Background-only** | c | d |

Where background-only = background - foreground (not double-counted).

- Test: `scipy.stats.fisher_exact(table, alternative='greater')` (one-sided, testing for enrichment)
- Odds ratio: `(a * d) / (b * c)`
- Fold enrichment: `(a / (a+b)) / ((a+c) / N)` where N = total background
- Minimum foreground count: terms with a < 1 are skipped to reduce testing burden

**FDR correction**: Custom Benjamini-Hochberg implementation (avoids `statsmodels` dependency). Applied across all top-level terms in Stage 2. In Stage 3, FDR is corrected **within each top-level branch** separately to avoid excessive penalty from testing hundreds of terms across unrelated domains.

### Circularity classification (`circularity.py`)

Two-pass classification of each enriched MP term:

**Pass 1 — Keyword/ontology** (`classify_term()`):

- **`acoustic_dependent`** if:
  - Term is under `MP:0005377` (hearing/vestibular/ear) branch AND comes from `IMPC_ABR_*` procedures, OR
  - Term name matches acoustic keywords: "pinna reflex", "preyer", "startle reflex", "acoustic startle", "auditory", "hearing", "cochlea", "deaf", "ABR", "click-evoked", OR
  - Term comes from acoustic startle procedure (`*_ACS_*`) and name contains "startle", "prepulse", or "ppi"
- **`vestibular`** if: term name matches vestibular keywords ("head bobbing", "circling", "trunk curl", etc.) or is under the hearing branch but not from ABR/acoustic procedures
- **`independent`** otherwise

**Pass 2 — Procedure-based** (`_check_procedure_based_circularity()`):

For each significant term classified as "independent" in Pass 1:
1. For every foreground gene that has this term, query the genotype-phenotype data to determine which procedures generated it
2. Classify each gene's evidence as acoustic (from ACS or ABR procedures, or from CSD when the specific assertion involves pinna/reflex terms) or non-acoustic
3. If >= 80% of genes' evidence is acoustic AND N >= 5 total genes, reclassify to `acoustic_dependent`

This caught 8 additional terms in the current analysis, including "abnormal synaptic transmission" (97% acoustic), "abnormal reflex" (96%), and "abnormal involuntary movement" (90%).

### Assertion-level acoustic filtering (`acoustic_filter.py`)

Instead of removing whole MP terms, this module filters individual genotype-phenotype assertions based on their generating procedure:

1. **Fully acoustic procedures** (`ACS`, `IMPC_ABR_`, `ESLIM_011`): all assertions removed (2,774 assertions)
2. **Mixed procedures** (`CSD`, `SHI`, `ESLIM_008`, `M-G-P_008`): only assertions whose MP term name matches acoustic keywords ("pinna reflex", "startle reflex", etc.) are removed (253 assertions); non-acoustic assertions are retained (3,728 assertions)
3. **All other procedures**: retained (60,595 assertions)

Total: 3,027 of 67,350 assertions removed (4.5%). Gene-to-MP mappings are rebuilt from the 64,323 remaining assertions and both top-level and hierarchical enrichment re-run. A detailed log of every removal decision is saved to `acoustic_filter_log.csv`.

### Centre matching (`centre_matching.py`)

For each gene in the background, only includes MP assertions from the same phenotyping centre that produced the gene's ABR data. Gene-to-centre mapping comes from the `center` column of the Bayesian results file. A gene tested at multiple centres has assertions from all its ABR centres included. Produces a sensitivity analysis comparing centre-matched vs. unmatched results.

### Output files

| File | Description | Rows |
|---|---|---|
| `top_level_enrichment.csv` | All 24 top-level MP terms with Fisher's test results | 24 |
| `top_level_enrichment_no_acoustic.csv` | Same, after removing acoustic-dependent assertions | 24 |
| `top_level_enrichment_centre_matched.csv` | Centre-matched variant | 24 |
| `hierarchical_enrichment.csv` | All descendant terms of significant top-level categories, with classification | 525 (157 significant) |
| `hierarchical_enrichment_no_acoustic.csv` | Hierarchical enrichment re-run on assertion-filtered data | 143 significant |
| `acoustic_filter_log.csv` | Log of every assertion removed/kept from acoustic/mixed procedures | — |
| `procedure_audit.csv` | Classification of all 253 IMPC procedures as acoustic/mixed/non-acoustic | 253 |
| `enrichment_analysis_log.txt` | Full log of classification decisions for every significant term | — |

Each CSV contains: `mp_term_id`, `mp_term_name`, `a`, `b`, `c`, `d`, `total_with_term`, `odds_ratio`, `fold_enrichment`, `p_value`, `p_adjusted`, `significant`. Hierarchical results additionally include `top_level_mp_id`, `top_level_mp_name`, `classification`, `classification_reason`, `procedures`.

The `classification_reason` column records how each term was classified: `keyword/ontology` (Pass 1), `procedure-based (N/M=X% acoustic)` (Pass 2), or `default` (not significant or not checked).

### Dependencies

All from existing project environment:
- `scipy.stats.fisher_exact` — Fisher's exact test
- `pandas` — data handling
- `numpy` — BH FDR correction
- `requests` — IMPC API fetching
- `json` — caching

No additional packages required. FDR correction uses a custom 10-line Benjamini-Hochberg implementation rather than importing `statsmodels`.
