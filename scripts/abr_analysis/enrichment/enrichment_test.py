"""
Stages 2 and 3: Fisher's exact test enrichment at all MP hierarchy levels.
"""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from .config import FDR_ALPHA, MIN_FOREGROUND_COUNT

logger = logging.getLogger(__name__)


def benjamini_hochberg(pvalues):
    """Apply Benjamini-Hochberg FDR correction.

    Returns array of adjusted p-values.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    if n == 0:
        return np.array([])

    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]
    adjusted = np.empty(n)

    # BH: p_adj[i] = min(p[i] * n / rank, 1.0), enforcing monotonicity
    adjusted[sorted_idx[-1]] = min(sorted_pvals[-1], 1.0)
    for i in range(n - 2, -1, -1):
        rank = i + 1
        raw_adj = sorted_pvals[i] * n / rank
        adjusted[sorted_idx[i]] = min(raw_adj, adjusted[sorted_idx[i + 1]], 1.0)

    return adjusted


def run_fishers_test(foreground, background, gene_to_terms, term_id):
    """Run Fisher's exact test for a single MP term.

    Args:
        foreground: set of foreground gene symbols
        background: set of all background gene symbols (includes foreground)
        gene_to_terms: {gene: set of MP term IDs}
        term_id: the MP term to test

    Returns:
        dict with a, b, c, d, odds_ratio, fold_enrichment, p_value
        or None if the term has no foreground overlap
    """
    n_fg = len(foreground)
    n_bg = len(background)
    n_bg_only = n_bg - n_fg

    # a = foreground genes with this term
    a = sum(1 for g in foreground if term_id in gene_to_terms.get(g, set()))
    if a < MIN_FOREGROUND_COUNT:
        return None

    # Total genes with this term (foreground + background-only)
    total_with_term = sum(
        1 for g in background if term_id in gene_to_terms.get(g, set())
    )
    c = total_with_term - a  # background-only genes with this term

    b = n_fg - a
    d = n_bg_only - c

    table = [[a, b], [c, d]]
    odds_ratio, p_value = fisher_exact(table, alternative="greater")

    # Fold enrichment
    expected = n_fg * total_with_term / n_bg if n_bg > 0 else 0
    fold_enrichment = a / expected if expected > 0 else float("inf")

    return {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "total_with_term": total_with_term,
        "odds_ratio": odds_ratio,
        "fold_enrichment": fold_enrichment,
        "p_value": p_value,
    }


def run_top_level_enrichment(foreground, background, gene_to_top_mp, top_level_names):
    """Run Fisher's exact test for each top-level MP term.

    Returns a DataFrame with one row per top-level term, FDR-corrected.
    """
    # Collect all top-level term IDs present in the data
    all_top_ids = set()
    for terms in gene_to_top_mp.values():
        all_top_ids.update(terms)

    results = []
    for term_id in sorted(all_top_ids):
        res = run_fishers_test(foreground, background, gene_to_top_mp, term_id)
        if res is None:
            continue
        res["mp_term_id"] = term_id
        res["mp_term_name"] = top_level_names.get(term_id, term_id)
        results.append(res)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["p_adjusted"] = benjamini_hochberg(df["p_value"].values)
    df["significant"] = df["p_adjusted"] < FDR_ALPHA
    df = df.sort_values("p_value").reset_index(drop=True)

    cols = [
        "mp_term_id",
        "mp_term_name",
        "a",
        "b",
        "c",
        "d",
        "total_with_term",
        "odds_ratio",
        "fold_enrichment",
        "p_value",
        "p_adjusted",
        "significant",
    ]
    return df[cols]


def run_hierarchical_enrichment(
    significant_top_ids, foreground, background, gene_to_all_mp, ontology, mp_term_names
):
    """Run Fisher's exact test for all descendants of significant top-level terms.

    Tests at every level of the MP hierarchy (intermediate and leaf terms).

    Returns a DataFrame with one row per tested term.
    """
    results = []

    for top_id in significant_top_ids:
        top_name = ontology.terms.get(top_id, mp_term_names.get(top_id, top_id))
        descendants = ontology.get_all_descendants(top_id)
        # Also include the top-level term itself
        test_terms = descendants | {top_id}

        # Filter to terms that exist in our data
        terms_in_data = set()
        for gene_terms in gene_to_all_mp.values():
            terms_in_data.update(gene_terms)
        test_terms = test_terms & terms_in_data

        logger.info(f"Testing {len(test_terms)} terms under {top_name} ({top_id})")

        for term_id in sorted(test_terms):
            res = run_fishers_test(foreground, background, gene_to_all_mp, term_id)
            if res is None:
                continue
            res["mp_term_id"] = term_id
            res["mp_term_name"] = mp_term_names.get(
                term_id, ontology.terms.get(term_id, term_id)
            )
            res["top_level_mp_id"] = top_id
            res["top_level_mp_name"] = top_name
            results.append(res)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # FDR correct within each top-level branch
    adjusted = np.ones(len(df))
    for top_id in df["top_level_mp_id"].unique():
        mask = df["top_level_mp_id"] == top_id
        if mask.sum() > 0:
            adjusted[mask.values] = benjamini_hochberg(df.loc[mask, "p_value"].values)
    df["p_adjusted"] = adjusted
    df["significant"] = df["p_adjusted"] < FDR_ALPHA
    df = df.sort_values(["top_level_mp_id", "p_value"]).reset_index(drop=True)

    return df


def filter_terms_from_mappings(gene_to_terms, terms_to_remove):
    """Remove specified MP term IDs from gene-to-term mappings.

    Returns a new dict with the terms removed.
    """
    filtered = {}
    for gene, terms in gene_to_terms.items():
        remaining = terms - terms_to_remove
        if remaining:
            filtered[gene] = remaining
    return filtered
