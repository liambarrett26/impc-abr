"""Configuration constants for the enrichment analysis."""

import os
from pathlib import Path

# API
SOLR_GP_URL = "https://www.ebi.ac.uk/mi/impc/solr/genotype-phenotype/select"
MP_OBO_URL = "http://purl.obolibrary.org/obo/mp.obo"
SOLR_PAGE_SIZE = 5000
SOLR_DELAY = 0.5  # seconds between API requests

# Paths
ENRICHMENT_DIR = Path(__file__).parent
CACHE_DIR = ENRICHMENT_DIR / "data"
RESULTS_DIR = ENRICHMENT_DIR / "results"

# Directory holding the Bayesian-results inputs for the enrichment foreground /
# background. Defaults to the repository ``data/processed`` directory; override
# with the ``IMPC_SUPP_DIR`` environment variable to point elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUPP_DIR = Path(os.environ.get("IMPC_SUPP_DIR", PROJECT_ROOT / "data" / "processed"))
FOREGROUND_FILE = (
    SUPP_DIR / "supplementary_file_4_bayesian_results_significant_annotated.csv"
)
BACKGROUND_FILE = SUPP_DIR / "supplementary_file_4_bayesian_results_all.csv"

# Analysis parameters
BF_THRESHOLD = 3.0
FDR_ALPHA = 0.05
MIN_FOREGROUND_COUNT = 1  # minimum foreground genes with a term to test it

# Solr fields to retrieve
SOLR_FIELDS = [
    "marker_symbol",
    "allele_symbol",
    "phenotyping_center",
    "procedure_name",
    "procedure_stable_id",
    "parameter_name",
    "parameter_stable_id",
    "top_level_mp_term_id",
    "top_level_mp_term_name",
    "intermediate_mp_term_id",
    "intermediate_mp_term_name",
    "mp_term_id",
    "mp_term_name",
    "zygosity",
    "p_value",
    "effect_size",
    "life_stage_name",
    "sex",
]

# Acoustic-dependent keyword patterns for circularity classification
ACOUSTIC_KEYWORDS = [
    "pinna reflex",
    "preyer",
    "startle reflex",
    "acoustic startle",
    "auditory",
    "hearing",
    "cochlea",
    "deaf",
    "ABR",
    "click-evoked",
]
ACOUSTIC_PROCEDURES = ["IMPC_ABR_"]  # procedure_stable_id prefix

VESTIBULAR_KEYWORDS = [
    "vestibular",
    "head bobbing",
    "circling",
    "trunk curl",
    "head tilt",
    "balance",
    "righting",
]
