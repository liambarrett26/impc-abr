# Gene Enrichment Analysis: Concept, Methods, and Application to IMPC ABR

## What is Gene Enrichment Analysis?

Gene enrichment analysis answers a simple question: **given a list of genes you care about, are certain biological categories over-represented in that list compared to what you'd expect by chance?**

The intuition is straightforward. Suppose you have 7,000 genes in your dataset, and 500 of them (across the whole dataset) are annotated with "nervous system phenotype." That's about 7% of all genes. Now suppose you have a special list of 139 genes — your auditory phenodeviants — and 25 of them have "nervous system phenotype." That's about 19%. Is 19% meaningfully higher than 7%, or could this have happened by chance just because you picked 129 genes at random?

Gene enrichment analysis formalises this comparison. It tells you whether the overlap between your gene list and a given category is larger than expected under random sampling, and quantifies how confident you should be in that conclusion.

---

## The Core Data Structure

Every enrichment analysis starts with the same ingredients:

### 1. A gene list of interest (the "foreground")

This is the set of genes you've identified through some analysis. In our project, this is the set of genes with evidence of causing hearing loss, identified by the Bayesian mixture model (e.g., all genes with Bayes Factor ≥ 3).

In the Vicencio-Jimenez et al. (2026) paper, their equivalent foreground list is all genes flagged by IMPC as having "abnormal auditory phenotypes" — 331 genes based on IMPC's standard statistical pipeline annotations from Data Release 23.0.

### 2. A background set (the "universe")

This is the full set of genes against which you're comparing. It must be the set of genes that *could have* ended up in your foreground list. This is critical — using the wrong background inflates or deflates your results.

In our project, the background is all ~7,000+ genes that were analysed in our Bayesian pipeline (i.e., genes with complete ABR data and sufficient matched controls). It is *not* the entire mouse genome, because most genes were never tested for hearing.

Vicencio-Jimenez et al. use 9,277 knockout lines from the IMPC Early Adult pipeline as their background.

### 3. Annotation categories (the "terms")

These are the biological categories you're testing for enrichment. In our context, these are Mammalian Phenotype (MP) ontology terms — standardised labels describing phenotypic outcomes (e.g., "decreased startle reflex", "abnormal locomotor behavior", "nervous system phenotype").

MP terms are organised hierarchically. "Decreased startle reflex" is a child of "abnormal startle reflex", which is a child of "behavior/neurological phenotype" (a top-level term). Enrichment can be tested at any level of this hierarchy.

---

## The 2×2 Table

For each annotation term, you construct a contingency table:

|  | Has the MP term | Does not have the MP term | Row total |
|--|--|--|--|
| **In foreground (auditory phenodeviants)** | a | b | a + b = n |
| **Not in foreground** | c | d | c + d |
| **Column total** | a + c = K | b + d | N |

Where:
- **N** = total genes in background
- **n** = total genes in foreground (your gene list)
- **K** = total genes in background with this MP term
- **a** = genes that are both in your foreground AND have this MP term (the overlap you're testing)

The enrichment question reduces to: **is `a` larger than expected?**

Under random sampling (no association), the expected value of `a` is:

```
E(a) = n × K / N
```

If the observed `a` is substantially larger than this expectation, the term is enriched.

---

## Key Quantities

### Odds Ratio (OR)

```
OR = (a × d) / (b × c)
```

An OR of 1 means no enrichment. OR > 1 means the term is more common in your gene list than in the background. OR = 3.13 (as Vicencio-Jimenez report for hearing × behavioral abnormalities) means genes with hearing phenotypes are about 3 times more likely to also have behavioral abnormalities.

### Fold Enrichment

```
Fold enrichment = (a / n) / (K / N)
```

This is the ratio of the observed proportion to the expected proportion. A fold enrichment of 7.2 means the term appears 7 times more often in your gene list than you'd expect from the background rate.

### Statistical Significance

Quantifies whether the observed enrichment could have arisen by chance. Different frameworks provide this differently:

- **Fisher's exact test** (frequentist): computes the exact probability of seeing an overlap ≥ `a` under the null hypothesis of no association. Returns a p-value.
- **Bayes Factor** (Bayesian): compares the probability of the data under an enrichment model vs. a no-enrichment model. Returns a ratio quantifying evidence strength.
- **Posterior probability** (Bayesian): gives the probability that the enrichment rate exceeds the background rate, given the observed data.

---

## Multiple Testing

When you test many terms (potentially hundreds of MP terms), some will appear enriched purely by chance. If you test 200 terms at p < 0.05, you'd expect ~10 false positives even if no real enrichment exists.

**Frequentist correction**: Benjamini-Hochberg False Discovery Rate (FDR) adjusts p-values upward to control the expected proportion of false discoveries. This is standard in gene enrichment analysis.

**Bayesian perspective**: Bayes Factors are less susceptible to this problem because they measure relative evidence rather than tail probabilities. However, when reporting many BFs, some caution is still needed — a BF of 3.5 among 200 tests is less compelling than a BF of 3.5 from a single pre-specified test. In practice, focusing interpretation on BF ≥ 10 when testing many terms provides a reasonable safeguard.

---

## How Vicencio-Jimenez et al. (2026) Apply Enrichment

Their paper performs enrichment analysis in two distinct ways, both relevant to our project:

### 1. Phenotype co-occurrence (Fisher's exact test)

They ask: are genes with auditory abnormalities more likely to also have behavioral, cognitive, or emotional abnormalities?

Their foreground: 331 genes flagged by IMPC as having abnormal hearing phenotypes.
Their background: 9,277 KO lines from the Early Adult pipeline.
Their terms: behavioral abnormalities (OR = 3.13), cognitive abnormalities (OR = 1.98), emotional abnormalities (OR = 2.11).

This is exactly the analysis Sherri has begun replicating with our Bayesian phenodeviants, and the analysis described in Protocols A and B.

### 2. Gene Ontology (GO) enrichment (g:Profiler)

They also perform a separate enrichment analysis using Gene Ontology terms rather than MP terms. GO terms describe molecular functions, biological processes, and cellular components (e.g., "protein binding", "sensory perception of sound", "neuron projection").

Their foreground: subsets of genes with auditory phenotypes, auditory + behavioral, auditory + cognitive, auditory + emotional.
Their background: the full mouse genome (standard for GO enrichment).
Their terms: GO Biological Process, Molecular Function, and Cellular Component categories.

Key findings: genes with both auditory and cognitive abnormalities were enriched for potassium ion transport regulation, transporter complexes, and excitatory synapses — pathways with known roles in both hearing and neurodegeneration.

This type of GO enrichment is complementary to the MP term enrichment and could be a future extension of our analysis, but is not the current focus.

---

## How Enrichment Applies to Our Project

### What we have

Our Bayesian mixture model produces, for each of ~7,000+ genes:
- A **Bayes Factor** quantifying evidence for hearing loss
- A **posterior probability** of hearing loss
- Estimated **hearing loss shift** at each frequency (characterising the pattern)

This is richer output than what Vicencio-Jimenez et al. start with (binary significant/not-significant from IMPC's standard pipeline).

### What we want to know

1. **Are hearing loss genes enriched for other phenotypic categories?** Particularly behavioral, neurological, and skeletal phenotypes — mirroring the Vicencio-Jimenez findings but starting from our Bayesian gene list.

2. **Do the enrichment patterns depend on the strength of hearing loss evidence?** Genes with extreme evidence (BF > 100) may show different co-morbidity profiles than genes with marginal evidence (BF 3–10).

3. **Are specific behavioral phenotypes enriched, and which ones are genuinely informative vs. acoustic/vestibular confounds?**

### Why this matters for the manuscript

The enrichment analysis serves two purposes in the paper:

**Validation**: If our Bayesian phenodeviants are enriched for nervous system and behavioral phenotypes (as Vicencio-Jimenez found for IMPC's standard calls), this provides independent biological validation that our gene list is capturing real biology, not statistical noise.

**Added value**: If our Bayesian approach identifies enrichment patterns that the standard IMPC approach misses — or provides more specific associations — this demonstrates practical utility of the multivariate/Bayesian framework beyond just "finding more genes."

### Why the Bayesian enrichment framework (Protocol B) matters

Using Bayes Factors for enrichment (rather than Fisher's exact) gives us:

1. **Methodological coherence**: The entire paper uses the same statistical language. Genes are identified with BFs; their phenotypic associations are quantified with BFs.

2. **Weighted analysis**: We can weight each gene by its posterior probability rather than applying a hard cutoff. A gene with BF = 150 contributes more to the enrichment signal than a gene with BF = 3.1. Fisher's exact cannot do this — it requires a binary gene list.

3. **Graduated evidence**: Instead of "significant at FDR < 0.05" vs. "not significant", we can say "strong evidence of enrichment" vs. "weak evidence" — matching the graduated language used throughout the rest of the analysis.

4. **No multiple testing correction needed**: Bayes Factors are inherently comparative and don't require the FDR correction step, simplifying interpretation (though caution is still warranted when testing many terms).

---

## Practical Considerations

### Centre matching

A subtlety specific to IMPC data: phenotype annotations are made at individual phenotyping centres, and not all centres run the same battery of tests. If a gene's ABR data come from Centre A but a behavioral annotation comes from Centre B, associating the two is problematic. Sherri has flagged this as a concern. The enrichment analysis should either restrict MP term associations to the same centre as the ABR data, or include a sensitivity analysis comparing centre-matched vs. unmatched results.

Vicencio-Jimenez et al. address this by checking whether pipeline-level testing differences correlate with sensory abnormality prevalence (they find no significant correlation), but they do not perform centre-level matching of individual gene annotations.

### The circularity problem: enrichment that tells us nothing new

This is the single most important interpretive issue in this analysis and must be understood before any results are taken at face value.

The risk of circularity arises because of the chain of dependencies between how we identify hearing loss genes and how MP terms are assigned:

1. We identify hearing loss genes using ABR thresholds (our Bayesian mixture model).
2. We then ask: are these genes enriched for other MP terms?
3. But many MP terms classified under "behavior/neurological" or "nervous system" are themselves dependent on auditory function.

**The pinna reflex example**: The pinna reflex (Preyer reflex) is a direct acoustic response — the ear flicks in response to a loud sound. A deaf mouse will have an absent pinna reflex simply because it cannot hear the stimulus. In the MP ontology, "absent pinna reflex" sits under "nervous system phenotype" or "behavior/neurological phenotype", not under "hearing/vestibular/ear." So when we find that hearing loss genes are "enriched for nervous system phenotypes," part of that enrichment signal may be driven entirely by absent pinna reflex — which is itself a consequence of the hearing loss we already identified. This is circular: hearing loss genes → absent response to sound → "nervous system phenotype" → enrichment. We have not learned anything new.

**The startle reflex example**: Similarly, the acoustic startle reflex requires the mouse to hear the stimulus. Decreased or abnormal startle reflex in deaf mice is expected, not informative. Prepulse inhibition (PPI) is more nuanced — it involves sensorimotor gating circuits beyond simple sound detection — but the stimulus is still acoustic, so a deaf mouse may show altered PPI simply due to not hearing the prepulse or the startle stimulus properly.

**Why sub-branch analysis is essential**: The Vicencio-Jimenez paper reports enrichment at the top-level MP category ("behavior/neurological phenotype", OR = 3.13). This looks impressive, but the top-level category bundles together dozens of specific phenotypes. When you drill into the sub-branches, you may find that the enrichment is almost entirely driven by acoustic-dependent terms (pinna reflex, startle reflex) rather than genuinely independent behavioral phenotypes. **We must test enrichment at every level of the MP hierarchy — top-level, intermediate, and leaf terms — and explicitly flag which leaf terms are acoustic-dependent.** Only enrichment that survives after removing acoustic-dependent terms is genuinely informative.

To be concrete, the analysis should:
1. Run enrichment across the full MP hierarchy including all sub-branches, not just top-level terms.
2. Categorise each enriched leaf-level MP term as: (a) acoustic-dependent, (b) vestibular, or (c) independent of auditory function.
3. Re-run enrichment after excluding category (a) terms. If the top-level "behavior/neurological" enrichment disappears, the original finding was circular.
4. Report both the full and filtered results transparently.

### Questioning the statistical basis of MP term assignments

A second critical consideration: **how were the MP terms themselves assigned?** Each MP annotation in the IMPC is the result of a statistical test comparing knockout mice to controls on some phenotyping measure. These tests may suffer from the same limitations that motivated our advanced statistical framework for ABR data.

Specifically:
- **Univariate testing**: IMPC's standard pipeline typically tests each parameter independently (e.g., startle amplitude at a single intensity, grip strength as a single value). This is the same frequency-by-frequency approach we argue is suboptimal for ABR analysis. Phenotyping tests that produce multivariate data (e.g., PPI across multiple prepulse intensities, open field activity over time) may be reduced to single summary statistics before testing, potentially missing coordinated patterns.
- **Subjective thresholds**: Some MP annotations involve operator judgment or rule-based calls (similar to how ABR thresholds were traditionally assigned by visual inspection). The reliability and consistency of these calls across centres is not always clear.
- **Binary classification**: IMPC assigns MP terms as present/absent based on statistical significance thresholds. A gene that narrowly misses significance gets no annotation, while one that barely reaches it gets a full MP call. This is the same binary thinking our Bayesian framework was designed to move beyond.

This means we should be cautious about treating MP annotations as ground truth. The enrichment analysis inherits whatever biases exist in the original phenotype calls. Where possible, we should note which IMPC tests underpin the enriched MP terms and whether those tests are robust or potentially problematic.

For the manuscript, this is actually an opportunity: if we find enrichment for MP terms assigned via robust, well-powered tests (e.g., body composition, blood chemistry), that's more convincing than enrichment for terms assigned via subjective or low-power tests. And if our Bayesian gene list shows different enrichment patterns than Vicencio-Jimenez's IMPC-annotated list, the difference may partly reflect how genes were identified in the first place — our multivariate Bayesian approach vs. IMPC's standard univariate calls.

### Acoustic and vestibular confounds

Many behavioral MP terms in the IMPC are assessed using acoustic stimuli (startle reflex, prepulse inhibition, pinna reflex). Enrichment of these among hearing loss genes may simply reflect that deaf mice don't respond to sound — not that the gene affects "behavior" in any meaningful cognitive or emotional sense.

Similarly, many hearing loss genes also affect vestibular function (shared inner ear structures), producing phenotypes like head bobbing, circling, and impaired righting response. These are genuine biological co-morbidities but reflect inner ear pathology rather than central nervous system dysfunction.

Both Sherri and Vicencio-Jimenez et al. flag these issues. The enrichment results should clearly separate:
- **Acoustic confounds**: absent pinna reflex, decreased startle reflex, abnormal startle reflex (response depends on hearing the stimulus)
- **Vestibular phenotypes**: head bobbing, circling, trunk curl, stereotypic behavior (likely inner ear, not CNS)
- **Potentially informative co-morbidities**: abnormal learning, decreased prepulse inhibition (though see caveats above), abnormal sleep behavior, hyperactivity

### MP term hierarchy issues

As Sherri notes, the MP ontology can be messy. "Absent pinna reflex" is bundled under startle-related terms in some contexts but not others. "Stereotypic behavior" actually maps to the IMPC parameter "head bobbing/circling." Some terms appear to be defunct or inconsistently grouped.

**This is precisely why testing at the sub-branch level matters.** Top-level enrichment can obscure what's actually driving the signal. A top-level term like "behavior/neurological phenotype" contains everything from "abnormal learning" (potentially very interesting) to "absent pinna reflex" (almost certainly circular). Without examining the sub-branches, you cannot tell the difference.

Any implementation should:
- Document the specific MP term mappings used and flag known inconsistencies.
- Traverse the full MP tree, not just test at one level.
- Map each leaf term back to the IMPC test/parameter that generated it, enabling assessment of whether the underlying test is acoustic-dependent.

---

## Summary

| Concept | Definition |
|--|--|
| **Gene enrichment** | Testing whether a gene list is over-represented in a biological category |
| **Foreground** | Your gene list of interest (our Bayesian phenodeviants) |
| **Background** | All genes that could have been in the foreground (all genes analysed) |
| **Annotation terms** | Categories being tested (MP terms, GO terms, etc.) |
| **Odds ratio** | How much more likely foreground genes are to have the term |
| **Fold enrichment** | Observed rate ÷ expected rate |
| **Fisher's exact** | Frequentist test of whether overlap exceeds chance (Protocol A) |
| **Bayes Factor** | Bayesian evidence measure comparing enrichment vs. no-enrichment models (Protocol B) |
| **FDR correction** | Adjustment for testing many terms simultaneously (frequentist only) |
| **Centre matching** | Restricting phenotype associations to the same centre as ABR data |
