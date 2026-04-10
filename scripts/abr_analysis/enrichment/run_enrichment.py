#!/usr/bin/env python3
"""
Main orchestrator for MP term enrichment analysis.

Runs Fisher's exact test enrichment at top-level and hierarchical levels,
with circularity assessment and centre-matching sensitivity analysis.

Usage:
    python -m enrichment.run_enrichment
"""

import logging
import sys

import pandas as pd

from .config import CACHE_DIR, RESULTS_DIR
from .fetch_data import (
    fetch_genotype_phenotype,
    download_mp_ontology,
    load_gene_sets,
    build_gene_mp_mappings,
)
from .enrichment_test import (
    run_top_level_enrichment,
    run_hierarchical_enrichment,
    filter_terms_from_mappings,
)
from .circularity import classify_results, get_acoustic_term_ids
from .acoustic_filter import filter_acoustic_assertions
from .centre_matching import build_centre_matched_mappings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Data acquisition ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1: Data acquisition")
    logger.info("=" * 60)

    gp_df = fetch_genotype_phenotype()
    ontology = download_mp_ontology()
    fg_genes, bg_genes, gene_centres = load_gene_sets()

    (gene_to_top_mp, gene_to_all_mp, gene_to_mp_by_centre,
     mp_to_procedures, mp_term_names, top_level_names) = build_gene_mp_mappings(gp_df)

    # Restrict to genes in our background set
    gene_to_top_mp = {g: t for g, t in gene_to_top_mp.items() if g in bg_genes}
    gene_to_all_mp = {g: t for g, t in gene_to_all_mp.items() if g in bg_genes}

    logger.info(f"Genes in background with any MP assertion: "
                f"{len(gene_to_top_mp)} (top-level), {len(gene_to_all_mp)} (all levels)")

    # ── Stage 2: Top-level enrichment ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 2: Top-level MP term enrichment")
    logger.info("=" * 60)

    top_results = run_top_level_enrichment(
        fg_genes, bg_genes, gene_to_top_mp, top_level_names
    )
    top_results.to_csv(RESULTS_DIR / "top_level_enrichment.csv", index=False)

    sig_top = top_results[top_results["significant"]]
    logger.info(f"Significant top-level terms (FDR < 0.05): {len(sig_top)}")
    for _, row in sig_top.iterrows():
        logger.info(f"  {row['mp_term_name']}: OR={row['odds_ratio']:.2f}, "
                     f"p_adj={row['p_adjusted']:.2e}, a={row['a']}")

    # Print full table for review
    logger.info("\nFull top-level results:")
    for _, row in top_results.iterrows():
        sig_marker = "*" if row["significant"] else " "
        logger.info(f"  {sig_marker} {row['mp_term_name']:<45} "
                     f"OR={row['odds_ratio']:>6.2f}  "
                     f"a={row['a']:>3}  "
                     f"p_adj={row['p_adjusted']:.2e}")

    # ── Stage 3: Hierarchical enrichment ───────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 3: Hierarchical enrichment (sub-branches)")
    logger.info("=" * 60)

    sig_top_ids = sig_top["mp_term_id"].tolist()

    if sig_top_ids:
        hier_results = run_hierarchical_enrichment(
            sig_top_ids, fg_genes, bg_genes,
            gene_to_all_mp, ontology, mp_term_names
        )

        # Classify for circularity (two-pass: keyword + procedure-based)
        hier_results = classify_results(
            hier_results, mp_to_procedures, ontology,
            foreground_genes=fg_genes,
            gene_to_all_mp=gene_to_all_mp,
            gp_df=gp_df,
        )
        hier_results.to_csv(RESULTS_DIR / "hierarchical_enrichment.csv", index=False)

        # Log significant hierarchical results by classification
        sig_hier = hier_results[hier_results["significant"]]
        logger.info(f"\nSignificant sub-branch terms: {len(sig_hier)}")
        for classification in ["acoustic_dependent", "vestibular", "independent", "unclear"]:
            subset = sig_hier[sig_hier["classification"] == classification]
            if len(subset) > 0:
                logger.info(f"\n  [{classification.upper()}] ({len(subset)} terms):")
                for _, row in subset.iterrows():
                    logger.info(f"    {row['mp_term_name']:<50} "
                                 f"OR={row['odds_ratio']:>6.2f}  "
                                 f"a={row['a']:>3}  "
                                 f"p_adj={row['p_adjusted']:.2e}  "
                                 f"procs: {row.get('procedures', '')[:60]}")

        # ── Re-run with acoustic assertions removed at source ─────────
        logger.info("=" * 60)
        logger.info("STAGE 3b: Assertion-level acoustic filtering")
        logger.info("=" * 60)

        gp_filtered, removal_log = filter_acoustic_assertions(gp_df)
        removal_log.to_csv(RESULTS_DIR / "acoustic_filter_log.csv", index=False)

        # Rebuild all mappings from filtered data
        from .fetch_data import build_gene_mp_mappings as _build
        (gene_to_top_mp_filt, gene_to_all_mp_filt,
         _, _, mp_names_filt, top_names_filt) = _build(gp_filtered)
        gene_to_top_mp_filt = {g: t for g, t in gene_to_top_mp_filt.items() if g in bg_genes}
        gene_to_all_mp_filt = {g: t for g, t in gene_to_all_mp_filt.items() if g in bg_genes}

        # Top-level enrichment on filtered data
        top_filtered = run_top_level_enrichment(
            fg_genes, bg_genes, gene_to_top_mp_filt, top_names_filt
        )
        top_filtered.to_csv(
            RESULTS_DIR / "top_level_enrichment_no_acoustic.csv", index=False
        )

        logger.info("\nTop-level results AFTER assertion-level acoustic filtering:")
        for _, row in top_filtered.iterrows():
            sig_marker = "*" if row["significant"] else " "
            logger.info(f"  {sig_marker} {row['mp_term_name']:<45} "
                         f"OR={row['odds_ratio']:>6.2f}  "
                         f"a={row['a']:>3}  "
                         f"p_adj={row['p_adjusted']:.2e}")

        # Hierarchical enrichment on filtered data
        sig_top_filt_ids = top_filtered[top_filtered["significant"]]["mp_term_id"].tolist()
        if sig_top_filt_ids:
            hier_filtered = run_hierarchical_enrichment(
                sig_top_filt_ids, fg_genes, bg_genes,
                gene_to_all_mp_filt, ontology, mp_names_filt
            )
            hier_filtered.to_csv(
                RESULTS_DIR / "hierarchical_enrichment_no_acoustic.csv", index=False
            )

            sig_hier_filt = hier_filtered[hier_filtered["significant"]]
            logger.info(f"\nFiltered hierarchical: {len(sig_hier_filt)} significant terms")
            for _, row in sig_hier_filt.head(30).iterrows():
                logger.info(f"    {row['mp_term_name']:<50} "
                             f"OR={row['odds_ratio']:>6.2f}  "
                             f"a={row['a']:>3}  "
                             f"p_adj={row['p_adjusted']:.2e}")

        # ── Comparison: full vs filtered ───────────────────────────────
        _print_comparison(top_results, top_filtered)

    else:
        logger.info("No significant top-level terms — skipping hierarchical analysis.")
        hier_results = pd.DataFrame()

    # ── Stage 4: Centre-matched analysis ───────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 4: Centre-matched enrichment (sensitivity analysis)")
    logger.info("=" * 60)

    gene_to_top_matched = build_centre_matched_mappings(
        gp_df, gene_centres, term_level="top"
    )
    gene_to_top_matched = {g: t for g, t in gene_to_top_matched.items() if g in bg_genes}

    # Collect top-level names from matched data
    top_names_matched = {}
    for _, row in gp_df.iterrows():
        top_ids = row.get("top_level_mp_term_id", [])
        top_nms = row.get("top_level_mp_term_name", [])
        if isinstance(top_ids, str):
            top_ids = [top_ids]
        if isinstance(top_nms, str):
            top_nms = [top_nms]
        if isinstance(top_ids, list) and isinstance(top_nms, list):
            for tid, tnm in zip(top_ids, top_nms):
                top_names_matched[tid] = tnm

    top_matched = run_top_level_enrichment(
        fg_genes, bg_genes, gene_to_top_matched, top_names_matched
    )
    top_matched.to_csv(
        RESULTS_DIR / "top_level_enrichment_centre_matched.csv", index=False
    )

    logger.info("\nCentre-matched top-level results:")
    for _, row in top_matched.iterrows():
        sig_marker = "*" if row["significant"] else " "
        logger.info(f"  {sig_marker} {row['mp_term_name']:<45} "
                     f"OR={row['odds_ratio']:>6.2f}  "
                     f"a={row['a']:>3}  "
                     f"p_adj={row['p_adjusted']:.2e}")

    # ── Summary ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info(f"Output files:")
    for f in sorted(RESULTS_DIR.glob("*.csv")):
        logger.info(f"  {f.name}")


def _print_comparison(full_results, filtered_results):
    """Print comparison of full vs acoustic-filtered enrichment."""
    logger.info("\n--- Comparison: Full vs. Acoustic-Filtered ---")
    logger.info(f"{'MP Term':<45} {'Full OR':>8} {'Filt OR':>8} {'Full sig':>9} {'Filt sig':>9}")
    logger.info("-" * 80)

    merged = full_results.merge(
        filtered_results, on="mp_term_id", suffixes=("_full", "_filt"), how="outer"
    )
    for _, row in merged.iterrows():
        name = row.get("mp_term_name_full", row.get("mp_term_name_filt", "?"))
        full_or = f"{row.get('odds_ratio_full', 0):.2f}" if pd.notna(row.get("odds_ratio_full")) else "N/A"
        filt_or = f"{row.get('odds_ratio_filt', 0):.2f}" if pd.notna(row.get("odds_ratio_filt")) else "N/A"
        full_sig = "YES" if row.get("significant_full", False) else "no"
        filt_sig = "YES" if row.get("significant_filt", False) else "no"
        logger.info(f"  {name:<45} {full_or:>8} {filt_or:>8} {full_sig:>9} {filt_sig:>9}")


if __name__ == "__main__":
    main()
