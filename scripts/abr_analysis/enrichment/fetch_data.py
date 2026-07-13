"""
Stage 1: Data acquisition for enrichment analysis.

Fetches genotype-phenotype assertions from IMPC Solr API,
downloads and parses the MP ontology, and loads gene sets.
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

from .config import (
    BACKGROUND_FILE,
    CACHE_DIR,
    FOREGROUND_FILE,
    MP_OBO_URL,
    SOLR_DELAY,
    SOLR_FIELDS,
    SOLR_GP_URL,
    SOLR_PAGE_SIZE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IMPC genotype-phenotype API
# ---------------------------------------------------------------------------


def fetch_genotype_phenotype(cache_path=None):
    """Fetch all genotype-phenotype assertions from IMPC Solr API.

    Returns a DataFrame with one row per assertion. Caches to parquet.
    """
    if cache_path is None:
        cache_path = CACHE_DIR / "genotype_phenotype_all.json"
    cache_path = Path(cache_path)

    if cache_path.exists():
        logger.info(f"Loading cached genotype-phenotype data from {cache_path}")
        import json as _json

        with open(cache_path) as f:
            all_docs = _json.load(f)
        return pd.DataFrame(all_docs)

    logger.info("Fetching genotype-phenotype data from IMPC Solr API...")
    all_docs = []
    start = 0
    fl = ",".join(SOLR_FIELDS)

    # Get total count first
    resp = requests.get(
        SOLR_GP_URL, params={"q": "*:*", "rows": 0, "wt": "json"}, timeout=60
    )
    resp.raise_for_status()
    total = resp.json()["response"]["numFound"]
    logger.info(f"Total records to fetch: {total:,}")

    while start < total:
        params = {
            "q": "*:*",
            "rows": SOLR_PAGE_SIZE,
            "start": start,
            "wt": "json",
            "fl": fl,
        }
        resp = requests.get(SOLR_GP_URL, params=params, timeout=60)
        resp.raise_for_status()
        docs = resp.json()["response"]["docs"]
        all_docs.extend(docs)
        start += SOLR_PAGE_SIZE
        logger.info(f"  Fetched {min(start, total):,} / {total:,}")
        if start < total:
            time.sleep(SOLR_DELAY)

    logger.info(f"Fetched {len(all_docs):,} records total")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(all_docs, f)
    logger.info(f"Cached {len(all_docs)} records to {cache_path}")

    return pd.DataFrame(all_docs)


# ---------------------------------------------------------------------------
# MP ontology
# ---------------------------------------------------------------------------


class MPOntology:
    """Minimal MP ontology parsed from OBO format."""

    def __init__(self, terms, parents, children):
        self.terms = terms  # {id: name}
        self.parents = parents  # {id: set of parent ids}
        self.children = children  # {id: set of child ids}

    def get_all_descendants(self, term_id):
        """Return all descendant term IDs (BFS)."""
        descendants = set()
        queue = list(self.children.get(term_id, []))
        while queue:
            child = queue.pop()
            if child not in descendants:
                descendants.add(child)
                queue.extend(self.children.get(child, []))
        return descendants

    def get_all_ancestors(self, term_id):
        """Return all ancestor term IDs (BFS)."""
        ancestors = set()
        queue = list(self.parents.get(term_id, []))
        while queue:
            parent = queue.pop()
            if parent not in ancestors:
                ancestors.add(parent)
                queue.extend(self.parents.get(parent, []))
        return ancestors

    def get_depth(self, term_id):
        """Return depth from root (0 = root)."""
        ancestors = self.get_all_ancestors(term_id)
        if not ancestors:
            return 0
        # Depth = longest path to root
        return max(self.get_depth(a) for a in self.parents.get(term_id, [])) + 1


def download_mp_ontology(cache_path=None):
    """Download and parse the MP ontology OBO file.

    Returns an MPOntology instance.
    """
    if cache_path is None:
        cache_path = CACHE_DIR / "mp.obo"
    cache_path = Path(cache_path)

    if not cache_path.exists():
        logger.info(f"Downloading MP ontology from {MP_OBO_URL}...")
        resp = requests.get(MP_OBO_URL, timeout=120)
        resp.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(resp.content)
        logger.info(f"Saved to {cache_path}")
    else:
        logger.info(f"Loading cached MP ontology from {cache_path}")

    return _parse_obo(cache_path)


def _parse_obo(path):
    """Parse an OBO file into an MPOntology."""
    terms = {}
    parents = defaultdict(set)
    children = defaultdict(set)

    current_id = None
    current_name = None
    is_obsolete = False

    with open(path) as f:
        in_term = False
        for line in f:
            line = line.strip()
            if line == "[Term]":
                in_term = True
                current_id = None
                current_name = None
                is_obsolete = False
            elif line == "" or line.startswith("["):
                if in_term and current_id and not is_obsolete:
                    terms[current_id] = current_name or current_id
                in_term = line == "[Term]"
                if in_term:
                    current_id = None
                    current_name = None
                    is_obsolete = False
            elif in_term:
                if line.startswith("id: "):
                    current_id = line[4:]
                elif line.startswith("name: "):
                    current_name = line[6:]
                elif line.startswith("is_a: "):
                    parent_id = line[6:].split(" !")[0].strip()
                    if current_id:
                        parents[current_id].add(parent_id)
                        children[parent_id].add(current_id)
                elif line.startswith("is_obsolete: true"):
                    is_obsolete = True

        # Handle last term
        if in_term and current_id and not is_obsolete:
            terms[current_id] = current_name or current_id

    logger.info(
        f"Parsed {len(terms):,} MP terms with "
        f"{sum(len(v) for v in children.values()):,} parent-child relationships"
    )
    return MPOntology(terms, dict(parents), dict(children))


# ---------------------------------------------------------------------------
# Gene sets and mappings
# ---------------------------------------------------------------------------


def load_gene_sets():
    """Load foreground and background gene sets from supplementary files.

    Returns:
        foreground_genes: set of gene symbols (BF >= 3)
        background_genes: set of gene symbols (all analysed)
        gene_centres: dict mapping gene_symbol -> set of ABR centres
    """
    fg_df = pd.read_csv(FOREGROUND_FILE)
    bg_df = pd.read_csv(BACKGROUND_FILE)

    foreground_genes = set(fg_df["gene_symbol"].dropna().unique())
    background_genes = set(bg_df["gene_symbol"].dropna().unique())

    # Build gene -> ABR centre mapping (a gene may have been tested at multiple centres)
    gene_centres = defaultdict(set)
    for _, row in bg_df.iterrows():
        if pd.notna(row.get("gene_symbol")) and pd.notna(row.get("center")):
            gene_centres[row["gene_symbol"]].add(row["center"])

    logger.info(f"Foreground: {len(foreground_genes)} genes (BF >= 3)")
    logger.info(f"Background: {len(background_genes)} genes (all analysed)")

    return foreground_genes, background_genes, dict(gene_centres)


def build_gene_mp_mappings(gp_df):
    """Build gene-to-MP-term mappings from genotype-phenotype data.

    Returns:
        gene_to_top_mp: {gene: set of top-level MP term IDs}
        gene_to_leaf_mp: {gene: set of leaf MP term IDs}
        gene_to_mp_by_centre: {(gene, centre): set of leaf MP term IDs}
        mp_to_procedures: {mp_term_id: set of procedure_stable_id prefixes}
        mp_term_names: {mp_term_id: mp_term_name}
        top_level_mp_names: {mp_term_id: mp_term_name}
    """
    gene_to_top_mp = defaultdict(set)
    gene_to_leaf_mp = defaultdict(set)
    gene_to_mp_by_centre = defaultdict(set)
    mp_to_procedures = defaultdict(set)
    mp_term_names = {}
    top_level_mp_names = {}

    for _, row in gp_df.iterrows():
        gene = row.get("marker_symbol")
        if pd.isna(gene):
            continue

        centre = row.get("phenotyping_center", "")
        proc_id = row.get("procedure_stable_id", "")
        if isinstance(proc_id, list):
            proc_id = proc_id[0] if proc_id else ""

        # Top-level MP terms (multi-valued)
        top_ids = row.get("top_level_mp_term_id", [])
        top_names = row.get("top_level_mp_term_name", [])
        if isinstance(top_ids, str):
            top_ids = [top_ids]
        if isinstance(top_names, str):
            top_names = [top_names]
        if not isinstance(top_ids, list):
            top_ids = []
        if not isinstance(top_names, list):
            top_names = []

        for tid, tname in zip(top_ids, top_names):
            gene_to_top_mp[gene].add(tid)
            top_level_mp_names[tid] = tname

        # Leaf MP term
        leaf_id = row.get("mp_term_id")
        leaf_name = row.get("mp_term_name")
        if pd.notna(leaf_id):
            gene_to_leaf_mp[gene].add(leaf_id)
            gene_to_mp_by_centre[(gene, centre)].add(leaf_id)
            mp_to_procedures[leaf_id].add(proc_id)
            if pd.notna(leaf_name):
                mp_term_names[leaf_id] = leaf_name

        # Intermediate MP terms (multi-valued)
        int_ids = row.get("intermediate_mp_term_id", [])
        int_names = row.get("intermediate_mp_term_name", [])
        if isinstance(int_ids, str):
            int_ids = [int_ids]
        if isinstance(int_names, str):
            int_names = [int_names]
        if not isinstance(int_ids, list):
            int_ids = []
        if not isinstance(int_names, list):
            int_names = []

        for iid, iname in zip(int_ids, int_names):
            gene_to_leaf_mp[gene].add(iid)
            gene_to_mp_by_centre[(gene, centre)].add(iid)
            mp_to_procedures[iid].add(proc_id)
            if pd.notna(iname):
                mp_term_names[iid] = iname

    logger.info(
        f"Built mappings: {len(gene_to_top_mp)} genes with top-level MP, "
        f"{len(gene_to_leaf_mp)} genes with leaf/intermediate MP, "
        f"{len(mp_term_names)} unique MP terms"
    )

    return (
        dict(gene_to_top_mp),
        dict(gene_to_leaf_mp),
        dict(gene_to_mp_by_centre),
        dict(mp_to_procedures),
        mp_term_names,
        top_level_mp_names,
    )
