"""
Stage 4: Centre-matched enrichment analysis.

Only counts MP term assertions from the same phenotyping centre
where the gene's ABR data were collected.
"""

import logging
from collections import defaultdict

from .config import MIN_FOREGROUND_COUNT

logger = logging.getLogger(__name__)


def build_centre_matched_mappings(gp_df, gene_centres, term_level="top"):
    """Build gene-to-MP-term mappings restricted to centre-matched assertions.

    Args:
        gp_df: genotype-phenotype DataFrame from IMPC API
        gene_centres: {gene_symbol: set of ABR centre names}
        term_level: "top" for top-level MP terms, "all" for leaf/intermediate

    Returns:
        dict mapping gene_symbol -> set of MP term IDs (centre-matched only)
    """
    gene_to_terms = defaultdict(set)
    matched = 0
    unmatched = 0

    for _, row in gp_df.iterrows():
        gene = row.get("marker_symbol")
        centre = row.get("phenotyping_center", "")

        if gene not in gene_centres:
            continue

        # Check if this assertion's centre matches any of the gene's ABR centres
        if centre not in gene_centres.get(gene, set()):
            unmatched += 1
            continue

        matched += 1

        if term_level == "top":
            top_ids = row.get("top_level_mp_term_id", [])
            if isinstance(top_ids, str):
                top_ids = [top_ids]
            if isinstance(top_ids, list):
                gene_to_terms[gene].update(top_ids)
        else:
            # Leaf and intermediate terms
            import pandas as pd
            leaf_id = row.get("mp_term_id")
            if pd.notna(leaf_id):
                gene_to_terms[gene].add(leaf_id)

            int_ids = row.get("intermediate_mp_term_id", [])
            if isinstance(int_ids, str):
                int_ids = [int_ids]
            if isinstance(int_ids, list):
                gene_to_terms[gene].update(int_ids)

    logger.info(f"Centre matching: {matched:,} matched, {unmatched:,} unmatched assertions")
    logger.info(f"Genes with centre-matched MP terms: {len(gene_to_terms)}")

    return dict(gene_to_terms)
