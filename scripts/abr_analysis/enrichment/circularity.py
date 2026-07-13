"""
Circularity assessment: classify enriched MP terms as acoustic-dependent,
vestibular, or independent.

Two-pass classification:
1. Keyword/ontology-based: catches terms with acoustic names or under hearing branch
2. Procedure-based: catches parent/intermediate terms where >80% of foreground
   genes' evidence comes from acoustic-dependent procedures (ACS, CSD pinna reflex)
"""

import logging
from collections import defaultdict

import pandas as pd

from .config import ACOUSTIC_KEYWORDS, ACOUSTIC_PROCEDURES, VESTIBULAR_KEYWORDS

logger = logging.getLogger(__name__)

# MP branch for hearing/vestibular/ear
HEARING_TOP_LEVEL = "MP:0005377"

# Procedures that are acoustic-dependent
ACOUSTIC_PROC_PATTERNS = ["ACS", "IMPC_ABR_", "ESLIM_011"]
# CSD (SHIRPA) includes pinna reflex — partially acoustic
CSD_PROC_PATTERN = "CSD"

# Threshold: if this fraction of foreground genes' evidence for a term
# comes from acoustic procedures, classify as acoustic_dependent
ACOUSTIC_PROC_THRESHOLD = 0.80


def classify_term(mp_term_id, mp_term_name, procedures, ontology):
    """Classify an enriched MP term using keyword/ontology rules.

    This is the first-pass classifier. The second pass (procedure-based)
    is applied in classify_results().

    Returns:
        One of: "acoustic_dependent", "vestibular", "independent", "unclear"
    """
    name_lower = mp_term_name.lower() if mp_term_name else ""

    # Check if term is under the hearing/vestibular/ear branch
    ancestors = ontology.get_all_ancestors(mp_term_id)
    is_hearing_branch = (
        HEARING_TOP_LEVEL in ancestors or mp_term_id == HEARING_TOP_LEVEL
    )

    # Check if any generating procedure is ABR
    is_abr_procedure = any(
        any(proc.startswith(prefix) for prefix in ACOUSTIC_PROCEDURES)
        for proc in procedures
    )

    # Acoustic-dependent: hearing branch AND ABR procedure, OR acoustic keywords
    if is_hearing_branch and is_abr_procedure:
        return "acoustic_dependent"

    if any(kw in name_lower for kw in ACOUSTIC_KEYWORDS):
        return "acoustic_dependent"

    # Check if from acoustic startle procedure with startle/PPI keywords
    is_startle_procedure = any("ACS" in proc for proc in procedures)
    if is_startle_procedure and any(
        kw in name_lower for kw in ["startle", "prepulse", "ppi"]
    ):
        return "acoustic_dependent"

    # Vestibular
    if any(kw in name_lower for kw in VESTIBULAR_KEYWORDS):
        return "vestibular"

    # If under hearing branch but not ABR or acoustic keyword
    if is_hearing_branch:
        return "vestibular"

    return "independent"


def _is_acoustic_procedure(proc_id):
    """Check if a procedure ID is acoustic-dependent."""
    if not proc_id:
        return False
    for pattern in ACOUSTIC_PROC_PATTERNS:
        if pattern in proc_id:
            return True
    return False


def _check_procedure_based_circularity(
    term_id, foreground_genes, gene_to_all_mp, gp_df
):
    """Check what fraction of foreground genes' evidence for this term
    comes from acoustic-dependent procedures.

    Returns:
        (n_total, n_acoustic, n_non_acoustic, fraction_acoustic)
    """
    n_acoustic = 0
    n_non_acoustic = 0
    n_total = 0

    for gene in foreground_genes:
        gene_terms = gene_to_all_mp.get(gene, set())
        if term_id not in gene_terms:
            continue

        n_total += 1

        # Find which procedures generated this term for this gene
        gene_rows = gp_df[gp_df["marker_symbol"] == gene]
        gene_has_acoustic = False
        gene_has_non_acoustic = False

        for _, grow in gene_rows.iterrows():
            # Check if this row's terms include our target term
            leaf = grow.get("mp_term_id")
            int_terms = grow.get("intermediate_mp_term_id", [])
            if isinstance(int_terms, str):
                int_terms = [int_terms]
            if not isinstance(int_terms, list):
                int_terms = []

            all_terms = set(int_terms)
            if pd.notna(leaf):
                all_terms.add(leaf)

            if term_id not in all_terms:
                continue

            proc = grow.get("procedure_stable_id", "")
            if isinstance(proc, list):
                proc = proc[0] if proc else ""

            if _is_acoustic_procedure(proc):
                gene_has_acoustic = True
            # CSD includes pinna reflex — check if this specific CSD
            # assertion is for a pinna/reflex term
            elif CSD_PROC_PATTERN in str(proc):
                # CSD is mixed — pinna reflex is acoustic, other SHIRPA items aren't
                leaf_name = str(grow.get("mp_term_name", "")).lower()
                int_names = grow.get("intermediate_mp_term_name", [])
                if isinstance(int_names, str):
                    int_names = [int_names]
                if not isinstance(int_names, list):
                    int_names = []
                all_names = leaf_name + " " + " ".join(str(n) for n in int_names)
                if "pinna" in all_names or "reflex" in all_names.lower():
                    gene_has_acoustic = True
                else:
                    gene_has_non_acoustic = True
            else:
                gene_has_non_acoustic = True

        if gene_has_non_acoustic:
            n_non_acoustic += 1
        elif gene_has_acoustic:
            n_acoustic += 1

    frac = n_acoustic / n_total if n_total > 0 else 0.0
    return n_total, n_acoustic, n_non_acoustic, frac


def classify_results(
    hier_df,
    mp_to_procedures,
    ontology,
    foreground_genes=None,
    gene_to_all_mp=None,
    gp_df=None,
):
    """Add classification column to hierarchical enrichment results.

    Two-pass classification:
    1. Keyword/ontology-based (classify_term)
    2. Procedure-based: for terms classified as 'independent' in pass 1,
       check if >80% of foreground genes' evidence is from acoustic procedures

    Args:
        hier_df: DataFrame from run_hierarchical_enrichment
        mp_to_procedures: {mp_term_id: set of procedure_stable_ids}
        ontology: MPOntology instance
        foreground_genes: set of foreground gene symbols (for pass 2)
        gene_to_all_mp: {gene: set of MP term IDs} (for pass 2)
        gp_df: genotype-phenotype DataFrame (for pass 2)

    Returns:
        DataFrame with added 'classification', 'procedures',
        'classification_reason' columns
    """
    # Pass 1: keyword/ontology classification
    classifications = []
    reasons = []
    procedure_strs = []

    for _, row in hier_df.iterrows():
        term_id = row["mp_term_id"]
        term_name = row.get("mp_term_name", "")
        procs = mp_to_procedures.get(term_id, set())

        classification = classify_term(term_id, term_name, procs, ontology)
        classifications.append(classification)
        procedure_strs.append("; ".join(sorted(procs)))

        if classification == "acoustic_dependent":
            reasons.append("keyword/ontology")
        elif classification == "vestibular":
            reasons.append("keyword/ontology")
        else:
            reasons.append("default")

    hier_df = hier_df.copy()
    hier_df["classification"] = classifications
    hier_df["procedures"] = procedure_strs
    hier_df["classification_reason"] = reasons

    # Pass 2: procedure-based reclassification
    if (
        foreground_genes is not None
        and gene_to_all_mp is not None
        and gp_df is not None
    ):
        logger.info(
            "Pass 2: Procedure-based circularity check on 'independent' terms..."
        )

        reclassified = 0
        for idx, row in hier_df.iterrows():
            if row["classification"] != "independent":
                continue
            if not row["significant"]:
                continue

            term_id = row["mp_term_id"]
            term_name = row.get("mp_term_name", "")

            n_total, n_acoustic, n_non_acoustic, frac = (
                _check_procedure_based_circularity(
                    term_id, foreground_genes, gene_to_all_mp, gp_df
                )
            )

            if frac >= ACOUSTIC_PROC_THRESHOLD and n_total >= 5:
                hier_df.at[idx, "classification"] = "acoustic_dependent"
                hier_df.at[idx, "classification_reason"] = (
                    f"procedure-based ({n_acoustic}/{n_total}={frac:.0%} acoustic)"
                )
                reclassified += 1
                logger.info(
                    f"  RECLASSIFIED: {term_name} "
                    f"({n_acoustic}/{n_total} = {frac:.0%} from acoustic procs, "
                    f"{n_non_acoustic} non-acoustic)"
                )
            elif n_total >= 5:
                logger.info(
                    f"  KEPT independent: {term_name} "
                    f"({n_acoustic}/{n_total} = {frac:.0%} acoustic, "
                    f"{n_non_acoustic} non-acoustic)"
                )

        logger.info(
            f"Pass 2 reclassified {reclassified} terms from independent → acoustic_dependent"
        )

    # Log final summary
    counts = hier_df[hier_df["significant"]]["classification"].value_counts()
    logger.info(f"Final classifications (significant terms): {counts.to_dict()}")

    return hier_df


def get_acoustic_term_ids(hier_df):
    """Extract the set of MP term IDs classified as acoustic-dependent."""
    if "classification" not in hier_df.columns:
        return set()
    return set(hier_df[hier_df["classification"] == "acoustic_dependent"]["mp_term_id"])
