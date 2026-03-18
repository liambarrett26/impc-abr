"""
Gene-cluster association analysis for ContrastiveVAE-DEC clustering results.

This module provides comprehensive gene set enrichment analysis and statistical
testing for gene-cluster associations in audiometric phenotype discovery.
Implements multiple testing correction, pathway analysis, and biological validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import logging
from pathlib import Path
from scipy import stats
from scipy.stats import fisher_exact, chi2_contingency, hypergeom, false_discovery_control
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings

logger = logging.getLogger(__name__)


class GeneSetEnrichmentAnalysis:
    """
    Gene Set Enrichment Analysis (GSEA) for cluster-specific gene associations.
    
    Performs enrichment analysis to identify gene sets that are significantly
    over-represented in specific phenotype clusters compared to the background.
    """
    
    def __init__(self, correction_method: str = 'fdr_bh', alpha: float = 0.05):
        """
        Initialize GSEA analyzer.
        
        Args:
            correction_method: Multiple testing correction method ('fdr_bh', 'bonferroni', 'holm')
            alpha: Significance level
        """
        self.correction_method = correction_method
        self.alpha = alpha
        self.gene_sets = {}
        self.enrichment_results = {}
        
    def load_gene_sets(self, gene_sets: Dict[str, List[str]]) -> None:
        """
        Load gene sets for enrichment analysis.
        
        Args:
            gene_sets: Dictionary mapping gene set names to lists of genes
        """
        self.gene_sets = gene_sets
        logger.info(f"Loaded {len(gene_sets)} gene sets for enrichment analysis")
    
    def perform_enrichment_analysis(self, cluster_labels: np.ndarray,
                                  gene_labels: np.ndarray,
                                  min_genes_per_set: int = 5,
                                  max_genes_per_set: int = 500) -> Dict[str, Any]:
        """
        Perform gene set enrichment analysis for each cluster.
        
        Args:
            cluster_labels: Cluster assignments
            gene_labels: Gene knockout labels
            min_genes_per_set: Minimum genes required in set for analysis
            max_genes_per_set: Maximum genes allowed in set for analysis
            
        Returns:
            Dictionary of enrichment results per cluster
        """
        # Filter valid genes (remove -1 unknown genes)
        valid_mask = gene_labels != -1
        valid_clusters = cluster_labels[valid_mask]
        valid_genes = gene_labels[valid_mask]
        
        # Get unique genes in dataset
        dataset_genes = set(valid_genes)
        
        # Filter gene sets to those with sufficient overlap
        filtered_gene_sets = {}
        for set_name, gene_set in self.gene_sets.items():
            overlap_genes = set(gene_set) & dataset_genes
            if min_genes_per_set <= len(overlap_genes) <= max_genes_per_set:
                filtered_gene_sets[set_name] = list(overlap_genes)
        
        logger.info(f"Using {len(filtered_gene_sets)} gene sets for enrichment analysis")
        
        cluster_enrichments = {}
        
        for cluster_id in np.unique(valid_clusters):
            cluster_mask = valid_clusters == cluster_id
            cluster_genes = set(valid_genes[cluster_mask])
            
            enrichments = self._test_cluster_enrichments(
                cluster_genes, dataset_genes, filtered_gene_sets
            )
            
            cluster_enrichments[cluster_id] = enrichments
        
        # Apply multiple testing correction
        self._apply_multiple_testing_correction(cluster_enrichments)
        
        self.enrichment_results = cluster_enrichments
        return cluster_enrichments
    
    def _test_cluster_enrichments(self, cluster_genes: Set[str],
                                 dataset_genes: Set[str],
                                 gene_sets: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Test enrichment for a single cluster against all gene sets."""
        enrichments = []
        
        n_cluster_genes = len(cluster_genes)
        n_dataset_genes = len(dataset_genes)
        
        for set_name, gene_set in gene_sets.items():
            set_genes = set(gene_set)
            
            # Overlap between cluster and gene set
            overlap_genes = cluster_genes & set_genes
            n_overlap = len(overlap_genes)
            
            # Hypergeometric test
            # Population: all genes in dataset
            # Successes in population: genes in gene set
            # Sample: genes in cluster
            # Successes in sample: overlap genes
            
            n_set_genes = len(set_genes & dataset_genes)  # Only count genes present in dataset
            
            # Hypergeometric test for over-representation
            p_value = hypergeom.sf(n_overlap - 1, n_dataset_genes, n_set_genes, n_cluster_genes)
            
            # Effect size measures
            expected_overlap = (n_cluster_genes * n_set_genes) / n_dataset_genes
            fold_enrichment = n_overlap / expected_overlap if expected_overlap > 0 else float('inf')
            
            # Jaccard similarity
            jaccard_similarity = n_overlap / len(cluster_genes | set_genes) if len(cluster_genes | set_genes) > 0 else 0
            
            enrichments.append({
                'gene_set': set_name,
                'cluster_genes_in_set': n_overlap,
                'total_cluster_genes': n_cluster_genes,
                'set_genes_in_dataset': n_set_genes,
                'total_dataset_genes': n_dataset_genes,
                'expected_overlap': expected_overlap,
                'fold_enrichment': fold_enrichment,
                'jaccard_similarity': jaccard_similarity,
                'p_value': p_value,
                'overlap_genes': list(overlap_genes)
            })
        
        # Sort by p-value
        enrichments.sort(key=lambda x: x['p_value'])
        return enrichments
    
    def _apply_multiple_testing_correction(self, cluster_enrichments: Dict[int, List[Dict]]) -> None:
        """Apply multiple testing correction across all tests."""
        # Collect all p-values
        all_p_values = []
        for enrichments in cluster_enrichments.values():
            all_p_values.extend([e['p_value'] for e in enrichments])
        
        # Apply correction
        if self.correction_method == 'fdr_bh':
            rejected, corrected_p_values = false_discovery_control(all_p_values, method='bh')
        else:
            rejected, corrected_p_values, _, _ = multipletests(
                all_p_values, alpha=self.alpha, method=self.correction_method
            )
        
        # Assign corrected p-values back
        p_idx = 0
        for cluster_enrichments_list in cluster_enrichments.values():
            for enrichment in cluster_enrichments_list:
                enrichment['corrected_p_value'] = corrected_p_values[p_idx]
                enrichment['significant'] = rejected[p_idx]
                p_idx += 1
    
    def get_significant_enrichments(self, cluster_id: Optional[int] = None) -> Dict[str, List[Dict]]:
        """
        Get significantly enriched gene sets.
        
        Args:
            cluster_id: Specific cluster to analyze (None for all clusters)
            
        Returns:
            Dictionary of significant enrichments
        """
        if not self.enrichment_results:
            raise ValueError("Must run enrichment analysis first")
        
        significant_enrichments = {}
        
        clusters_to_check = [cluster_id] if cluster_id is not None else self.enrichment_results.keys()
        
        for cid in clusters_to_check:
            if cid in self.enrichment_results:
                significant = [
                    e for e in self.enrichment_results[cid] 
                    if e['significant'] and e['fold_enrichment'] > 1.0
                ]
                if significant:
                    significant_enrichments[cid] = significant
        
        return significant_enrichments


class GeneClusterAssociationTester:
    """
    Statistical testing for gene-cluster associations.
    
    Provides comprehensive statistical tests to identify genes that show
    significant associations with specific clusters.
    """
    
    def __init__(self, min_samples_per_gene: int = 5):
        """
        Initialize association tester.
        
        Args:
            min_samples_per_gene: Minimum samples required per gene for testing
        """
        self.min_samples_per_gene = min_samples_per_gene
        self.association_results = {}
    
    def test_all_associations(self, cluster_labels: np.ndarray,
                            gene_labels: np.ndarray) -> Dict[str, Any]:
        """
        Test associations between all genes and clusters.
        
        Args:
            cluster_labels: Cluster assignments
            gene_labels: Gene knockout labels
            
        Returns:
            Dictionary of association test results
        """
        # Filter valid genes and sufficient sample sizes
        valid_mask = gene_labels != -1
        valid_clusters = cluster_labels[valid_mask]
        valid_genes = gene_labels[valid_mask]
        
        # Count samples per gene
        gene_counts = Counter(valid_genes)
        frequent_genes = [gene for gene, count in gene_counts.items() 
                         if count >= self.min_samples_per_gene]
        
        logger.info(f"Testing associations for {len(frequent_genes)} genes with sufficient samples")
        
        # Overall contingency table analysis
        overall_results = self._test_overall_independence(valid_clusters, valid_genes, frequent_genes)
        
        # Gene-specific association tests
        gene_results = self._test_gene_specific_associations(valid_clusters, valid_genes, frequent_genes)
        
        # Cluster-specific association tests
        cluster_results = self._test_cluster_specific_associations(valid_clusters, valid_genes, frequent_genes)
        
        self.association_results = {
            'overall': overall_results,
            'gene_specific': gene_results,
            'cluster_specific': cluster_results,
            'summary': self._generate_association_summary(overall_results, gene_results, cluster_results)
        }
        
        return self.association_results
    
    def _test_overall_independence(self, clusters: np.ndarray, genes: np.ndarray,
                                 frequent_genes: List[str]) -> Dict[str, Any]:
        """Test overall independence between genes and clusters."""
        # Filter to frequent genes
        frequent_mask = np.isin(genes, frequent_genes)
        filtered_clusters = clusters[frequent_mask]
        filtered_genes = genes[frequent_mask]
        
        # Create contingency table
        contingency_table = pd.crosstab(filtered_genes, filtered_clusters)
        
        # Chi-square test of independence
        chi2_stat, chi2_p, chi2_dof, chi2_expected = chi2_contingency(contingency_table)
        
        # Cramér's V (effect size)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
        
        return {
            'contingency_table': contingency_table,
            'chi2_statistic': chi2_stat,
            'chi2_p_value': chi2_p,
            'chi2_degrees_freedom': chi2_dof,
            'cramers_v': cramers_v,
            'significant': chi2_p < 0.05,
            'n_genes_tested': len(frequent_genes),
            'n_samples_tested': len(filtered_clusters)
        }
    
    def _test_gene_specific_associations(self, clusters: np.ndarray, genes: np.ndarray,
                                       frequent_genes: List[str]) -> Dict[str, Dict[str, Any]]:
        """Test each gene for cluster association."""
        gene_results = {}
        
        for gene in frequent_genes:
            gene_mask = genes == gene
            gene_clusters = clusters[gene_mask]
            
            # Test against uniform distribution
            cluster_counts = np.bincount(gene_clusters, minlength=len(np.unique(clusters)))
            expected_counts = len(gene_clusters) / len(np.unique(clusters))
            
            # Chi-square goodness of fit test
            chi2_stat, chi2_p = stats.chisquare(cluster_counts, expected_counts)
            
            # Find dominant cluster
            dominant_cluster = np.argmax(cluster_counts)
            dominant_proportion = cluster_counts[dominant_cluster] / len(gene_clusters)
            
            # Fisher's exact test for dominant cluster vs all others
            fisher_table = [
                [cluster_counts[dominant_cluster], 
                 np.sum(cluster_counts) - cluster_counts[dominant_cluster]],
                [np.sum(clusters == dominant_cluster) - cluster_counts[dominant_cluster],
                 len(clusters) - np.sum(clusters == dominant_cluster) - 
                 (np.sum(cluster_counts) - cluster_counts[dominant_cluster])]
            ]
            
            try:
                fisher_odds_ratio, fisher_p = fisher_exact(fisher_table, alternative='greater')
            except:
                fisher_odds_ratio, fisher_p = 1.0, 1.0
            
            # Calculate entropy (measure of clustering vs spreading)
            cluster_probs = cluster_counts / cluster_counts.sum()
            entropy = -np.sum(cluster_probs[cluster_probs > 0] * np.log2(cluster_probs[cluster_probs > 0]))
            max_entropy = np.log2(len(np.unique(clusters)))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            gene_results[gene] = {
                'n_samples': len(gene_clusters),
                'cluster_distribution': cluster_counts,
                'dominant_cluster': dominant_cluster,
                'dominant_proportion': dominant_proportion,
                'chi2_statistic': chi2_stat,
                'chi2_p_value': chi2_p,
                'fisher_odds_ratio': fisher_odds_ratio,
                'fisher_p_value': fisher_p,
                'entropy': entropy,
                'normalized_entropy': normalized_entropy,
                'clustering_strength': 1 - normalized_entropy,  # High = more clustered
                'significant_clustering': chi2_p < 0.05,
                'significant_enrichment': fisher_p < 0.05
            }
        
        return gene_results
    
    def _test_cluster_specific_associations(self, clusters: np.ndarray, genes: np.ndarray,
                                          frequent_genes: List[str]) -> Dict[int, Dict[str, Any]]:
        """Test each cluster for gene enrichments."""
        cluster_results = {}
        
        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            cluster_genes = genes[cluster_mask]
            
            # Filter to frequent genes
            frequent_cluster_genes = [g for g in cluster_genes if g in frequent_genes]
            
            if len(frequent_cluster_genes) == 0:
                continue
            
            # Count genes in this cluster
            gene_counts_in_cluster = Counter(frequent_cluster_genes)
            
            # Test each frequent gene for enrichment in this cluster
            gene_enrichments = {}
            for gene in frequent_genes:
                # 2x2 contingency table for Fisher's exact test
                gene_in_cluster = np.sum((genes == gene) & (clusters == cluster_id))
                gene_not_in_cluster = np.sum((genes == gene) & (clusters != cluster_id))
                not_gene_in_cluster = np.sum((genes != gene) & (clusters == cluster_id))
                not_gene_not_in_cluster = np.sum((genes != gene) & (clusters != cluster_id))
                
                fisher_table = [[gene_in_cluster, gene_not_in_cluster],
                              [not_gene_in_cluster, not_gene_not_in_cluster]]
                
                try:
                    odds_ratio, p_value = fisher_exact(fisher_table, alternative='greater')
                    
                    # Calculate enrichment fold change
                    total_gene = gene_in_cluster + gene_not_in_cluster
                    total_cluster = gene_in_cluster + not_gene_in_cluster
                    total_samples = np.sum([[gene_in_cluster, gene_not_in_cluster],
                                          [not_gene_in_cluster, not_gene_not_in_cluster]])
                    
                    expected = (total_gene * total_cluster) / total_samples
                    fold_enrichment = gene_in_cluster / expected if expected > 0 else float('inf')
                    
                    gene_enrichments[gene] = {
                        'count_in_cluster': gene_in_cluster,
                        'total_count': total_gene,
                        'cluster_size': total_cluster,
                        'odds_ratio': odds_ratio,
                        'p_value': p_value,
                        'fold_enrichment': fold_enrichment,
                        'significant': p_value < 0.05 and fold_enrichment > 1.0
                    }
                except:
                    continue
            
            # Sort by significance
            significant_enrichments = {
                gene: data for gene, data in gene_enrichments.items()
                if data['significant']
            }
            
            cluster_results[cluster_id] = {
                'cluster_size': np.sum(cluster_mask),
                'n_unique_genes': len(set(frequent_cluster_genes)),
                'most_common_gene': max(gene_counts_in_cluster, key=gene_counts_in_cluster.get) if gene_counts_in_cluster else None,
                'gene_enrichments': gene_enrichments,
                'significant_enrichments': significant_enrichments,
                'n_significant_enrichments': len(significant_enrichments)
            }
        
        return cluster_results
    
    def _generate_association_summary(self, overall_results: Dict, gene_results: Dict,
                                    cluster_results: Dict) -> Dict[str, Any]:
        """Generate summary of association test results."""
        summary = {}
        
        # Overall summary
        summary['overall_significant'] = overall_results['significant']
        summary['cramers_v'] = overall_results['cramers_v']
        summary['n_genes_tested'] = overall_results['n_genes_tested']
        
        # Gene-level summary
        significantly_clustered_genes = [
            gene for gene, data in gene_results.items()
            if data['significant_clustering']
        ]
        significantly_enriched_genes = [
            gene for gene, data in gene_results.items()
            if data['significant_enrichment']
        ]
        
        summary['n_significantly_clustered_genes'] = len(significantly_clustered_genes)
        summary['n_significantly_enriched_genes'] = len(significantly_enriched_genes)
        summary['proportion_clustered_genes'] = len(significantly_clustered_genes) / len(gene_results)
        summary['proportion_enriched_genes'] = len(significantly_enriched_genes) / len(gene_results)
        
        # Cluster-level summary
        clusters_with_enrichments = [
            cluster_id for cluster_id, data in cluster_results.items()
            if data['n_significant_enrichments'] > 0
        ]
        
        summary['n_clusters_with_enrichments'] = len(clusters_with_enrichments)
        summary['mean_enrichments_per_cluster'] = np.mean([
            data['n_significant_enrichments'] for data in cluster_results.values()
        ])
        
        return summary


class BiologicalPathwayAnalyzer:
    """
    Biological pathway analysis for discovered gene clusters.
    
    Integrates with pathway databases to identify enriched biological
    processes and molecular functions in each phenotype cluster.
    """
    
    def __init__(self):
        """Initialize pathway analyzer."""
        self.pathway_databases = {}
        self.pathway_results = {}
    
    def load_pathway_database(self, database_name: str, pathways: Dict[str, List[str]]) -> None:
        """
        Load pathway database for analysis.
        
        Args:
            database_name: Name of the pathway database
            pathways: Dictionary mapping pathway names to gene lists
        """
        self.pathway_databases[database_name] = pathways
        logger.info(f"Loaded {database_name} with {len(pathways)} pathways")
    
    def analyze_cluster_pathways(self, cluster_gene_associations: Dict[int, List[str]],
                               database_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze pathway enrichments for cluster-associated genes.
        
        Args:
            cluster_gene_associations: Dictionary mapping clusters to associated genes
            database_names: Pathway databases to use (None for all)
            
        Returns:
            Pathway analysis results
        """
        if database_names is None:
            database_names = list(self.pathway_databases.keys())
        
        results = {}
        
        for db_name in database_names:
            if db_name not in self.pathway_databases:
                logger.warning(f"Database {db_name} not loaded, skipping")
                continue
            
            pathways = self.pathway_databases[db_name]
            
            # Run GSEA for each cluster
            gsea = GeneSetEnrichmentAnalysis()
            gsea.load_gene_sets(pathways)
            
            # Convert cluster associations to format needed for GSEA
            all_genes = []
            all_clusters = []
            
            for cluster_id, genes in cluster_gene_associations.items():
                all_genes.extend(genes)
                all_clusters.extend([cluster_id] * len(genes))
            
            if len(all_genes) > 0:
                enrichment_results = gsea.perform_enrichment_analysis(
                    np.array(all_clusters), np.array(all_genes)
                )
                
                results[db_name] = {
                    'enrichment_results': enrichment_results,
                    'significant_enrichments': gsea.get_significant_enrichments()
                }
        
        self.pathway_results = results
        return results


def perform_comprehensive_gene_enrichment_analysis(
    cluster_labels: np.ndarray,
    gene_labels: np.ndarray,
    gene_sets: Optional[Dict[str, List[str]]] = None,
    pathway_databases: Optional[Dict[str, Dict[str, List[str]]]] = None,
    save_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive gene enrichment analysis.
    
    Args:
        cluster_labels: Cluster assignments
        gene_labels: Gene knockout labels
        gene_sets: Gene sets for enrichment analysis
        pathway_databases: Pathway databases for biological analysis
        save_dir: Directory to save results
        
    Returns:
        Complete gene enrichment analysis results
    """
    logger.info("Starting comprehensive gene enrichment analysis")
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Gene-cluster association testing
    logger.info("Testing gene-cluster associations")
    association_tester = GeneClusterAssociationTester()
    association_results = association_tester.test_all_associations(cluster_labels, gene_labels)
    results['associations'] = association_results
    
    # 2. Gene set enrichment analysis (if gene sets provided)
    if gene_sets:
        logger.info("Performing gene set enrichment analysis")
        gsea = GeneSetEnrichmentAnalysis()
        gsea.load_gene_sets(gene_sets)
        enrichment_results = gsea.perform_enrichment_analysis(cluster_labels, gene_labels)
        results['gene_set_enrichments'] = enrichment_results
        results['significant_enrichments'] = gsea.get_significant_enrichments()
    
    # 3. Biological pathway analysis (if pathway databases provided)
    if pathway_databases:
        logger.info("Analyzing biological pathways")
        pathway_analyzer = BiologicalPathwayAnalyzer()
        
        # Load pathway databases
        for db_name, pathways in pathway_databases.items():
            pathway_analyzer.load_pathway_database(db_name, pathways)
        
        # Extract cluster-gene associations from association results
        cluster_gene_associations = {}
        for gene, gene_data in association_results['gene_specific'].items():
            if gene_data['significant_enrichment']:
                dominant_cluster = gene_data['dominant_cluster']
                if dominant_cluster not in cluster_gene_associations:
                    cluster_gene_associations[dominant_cluster] = []
                cluster_gene_associations[dominant_cluster].append(gene)
        
        pathway_results = pathway_analyzer.analyze_cluster_pathways(cluster_gene_associations)
        results['pathway_analysis'] = pathway_results
    
    # 4. Generate summary
    results['summary'] = _generate_enrichment_summary(results)
    
    # Save results if directory specified
    if save_dir:
        _save_enrichment_results(results, save_dir)
    
    logger.info("Gene enrichment analysis completed")
    return results


def _generate_enrichment_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary of enrichment analysis results."""
    summary = {}
    
    # Association summary
    if 'associations' in results:
        assoc_summary = results['associations']['summary']
        summary['overall_gene_cluster_association'] = assoc_summary['overall_significant']
        summary['proportion_clustered_genes'] = assoc_summary['proportion_clustered_genes']
        summary['n_significantly_enriched_genes'] = assoc_summary['n_significantly_enriched_genes']
        summary['n_clusters_with_enrichments'] = assoc_summary['n_clusters_with_enrichments']
    
    # Gene set enrichment summary
    if 'significant_enrichments' in results:
        sig_enrichments = results['significant_enrichments']
        summary['n_clusters_with_gene_set_enrichments'] = len(sig_enrichments)
        summary['total_significant_gene_sets'] = sum(len(enrichments) for enrichments in sig_enrichments.values())
    
    # Pathway analysis summary
    if 'pathway_analysis' in results:
        pathway_results = results['pathway_analysis']
        summary['pathway_databases_analyzed'] = list(pathway_results.keys())
        summary['n_pathway_databases'] = len(pathway_results)
    
    return summary


def _save_enrichment_results(results: Dict[str, Any], save_dir: Path):
    """Save enrichment analysis results."""
    import json
    import pickle
    
    # Save summary as JSON
    summary_file = save_dir / 'gene_enrichment_summary.json'
    with open(summary_file, 'w') as f:
        # Convert numpy types for JSON serialization
        summary_json = {}
        for key, value in results['summary'].items():
            if isinstance(value, (np.integer, np.floating)):
                summary_json[key] = value.item()
            else:
                summary_json[key] = value
        json.dump(summary_json, f, indent=2)
    
    # Save complete results as pickle
    results_file = save_dir / 'gene_enrichment_complete.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Saved gene enrichment analysis results to {save_dir}")


# Example gene sets for hearing loss research
def get_default_hearing_gene_sets() -> Dict[str, List[str]]:
    """Get default gene sets relevant to hearing loss research."""
    return {
        'known_hearing_loss_genes': [
            'Cdh23', 'Pcdh15', 'Myo7a', 'Ush1c', 'Ush1g', 'Whrn',
            'Otof', 'Slc26a5', 'Gjb2', 'Gjb6', 'Tecta', 'Col11a2',
            'Tmprss3', 'Strc', 'Loxhd1', 'Tmc1', 'Cib2', 'Espn'
        ],
        'inner_ear_development': [
            'Pax2', 'Pax8', 'Eya1', 'Six1', 'Dlx5', 'Dlx6', 'Tbx1',
            'Hoxa1', 'Hoxa2', 'Hoxb1', 'Fgf3', 'Fgf10', 'Bmp4'
        ],
        'mechanotransduction': [
            'Tmc1', 'Tmc2', 'Tmhs', 'Lhfpl5', 'Cib2', 'Cdh23', 'Pcdh15',
            'Myo7a', 'Myo1c', 'Espn', 'Whrn', 'Sans', 'Ush1c'
        ],
        'synaptic_transmission': [
            'Otof', 'Cacna1d', 'Slc17a8', 'Ribeye', 'Bassoon', 'Piccolo',
            'Snap25', 'Stx1a', 'Syp', 'Sv2b'
        ]
    }


# Export main functions
__all__ = [
    'GeneSetEnrichmentAnalysis',
    'GeneClusterAssociationTester',
    'BiologicalPathwayAnalyzer',
    'perform_comprehensive_gene_enrichment_analysis',
    'get_default_hearing_gene_sets'
]