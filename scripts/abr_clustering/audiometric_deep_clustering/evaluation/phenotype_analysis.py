"""
Phenotype analysis and biological interpretation for ContrastiveVAE-DEC clustering results.

This module provides comprehensive biological interpretation of discovered clusters,
including hearing loss pattern classification, clinical severity assessment,
phenotype-to-gene mapping, and statistical validation of audiometric phenotypes.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

logger = logging.getLogger(__name__)


class HearingLossClassifier:
    """
    Classifier for audiometric hearing loss patterns.
    
    Categorizes ABR patterns into clinically meaningful hearing loss types:
    - Normal hearing
    - Flat hearing loss  
    - High-frequency hearing loss
    - Low-frequency hearing loss
    - Mixed hearing loss
    - Profound hearing loss
    """
    
    def __init__(self, thresholds: Dict[str, float] = None):
        """
        Initialize hearing loss classifier.
        
        Args:
            thresholds: Custom thresholds for hearing loss classification
        """
        # Clinical thresholds (dB SPL) - conservative for mice
        self.thresholds = thresholds or {
            'normal': 25,      # Below this is normal
            'mild': 40,        # 25-40 dB: mild loss
            'moderate': 60,    # 40-60 dB: moderate loss  
            'severe': 80,      # 60-80 dB: severe loss
            'profound': 80     # Above this is profound
        }
        
        # Frequency ranges for pattern analysis
        self.frequency_ranges = {
            'low': [0, 1],      # 6, 12 kHz
            'mid': [1, 2, 3],   # 12, 18, 24 kHz  
            'high': [3, 4]      # 24, 30 kHz
        }
        
        logger.info("Initialized hearing loss classifier with clinical thresholds")
    
    def classify_individual_pattern(self, abr_thresholds: np.ndarray) -> Dict[str, Any]:
        """
        Classify hearing loss pattern for a single mouse.
        
        Args:
            abr_thresholds: ABR thresholds (6 frequencies: 6, 12, 18, 24, 30 kHz, click)
            
        Returns:
            Dictionary with classification results
        """
        # Calculate average thresholds by frequency range
        low_avg = np.mean(abr_thresholds[self.frequency_ranges['low']])
        mid_avg = np.mean(abr_thresholds[self.frequency_ranges['mid']])  
        high_avg = np.mean(abr_thresholds[self.frequency_ranges['high']])
        overall_avg = np.mean(abr_thresholds[:5])  # Exclude click-evoked
        
        # Determine overall severity
        severity = self._classify_severity(overall_avg)
        
        # Determine pattern type
        pattern_type = self._classify_pattern_type(low_avg, mid_avg, high_avg)
        
        # Calculate pattern metrics
        slope = self._calculate_frequency_slope(abr_thresholds[:5])
        variability = np.std(abr_thresholds[:5])
        
        return {
            'severity': severity,
            'pattern_type': pattern_type,
            'overall_average': overall_avg,
            'low_freq_avg': low_avg,
            'mid_freq_avg': mid_avg,
            'high_freq_avg': high_avg,
            'frequency_slope': slope,
            'threshold_variability': variability,
            'is_asymmetric': abs(high_avg - low_avg) > 20,
            'worst_frequency': np.argmax(abr_thresholds[:5]),
            'best_frequency': np.argmin(abr_thresholds[:5])
        }
    
    def classify_cluster_patterns(self, cluster_labels: np.ndarray, 
                                abr_features: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        Classify hearing loss patterns for all clusters.
        
        Args:
            cluster_labels: Cluster assignments
            abr_features: ABR threshold features
            
        Returns:
            Dictionary mapping cluster IDs to pattern classifications
        """
        cluster_patterns = {}
        
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            cluster_abr = abr_features[mask]
            
            if len(cluster_abr) == 0:
                continue
                
            # Classify individual patterns within cluster
            individual_patterns = [
                self.classify_individual_pattern(abr) for abr in cluster_abr
            ]
            
            # Aggregate cluster-level statistics
            cluster_patterns[cluster_id] = self._aggregate_cluster_patterns(
                individual_patterns, cluster_abr
            )
            
        return cluster_patterns
    
    def _classify_severity(self, avg_threshold: float) -> str:
        """Classify hearing loss severity based on average threshold."""
        if avg_threshold <= self.thresholds['normal']:
            return 'normal'
        elif avg_threshold <= self.thresholds['mild']:
            return 'mild'
        elif avg_threshold <= self.thresholds['moderate']:
            return 'moderate'
        elif avg_threshold <= self.thresholds['severe']:
            return 'severe'
        else:
            return 'profound'
    
    def _classify_pattern_type(self, low_avg: float, mid_avg: float, high_avg: float) -> str:
        """Classify hearing loss pattern type based on frequency-specific averages."""
        # Calculate differences between frequency ranges
        high_low_diff = high_avg - low_avg
        mid_low_diff = mid_avg - low_avg
        high_mid_diff = high_avg - mid_avg
        
        # Classification rules (conservative thresholds for mice)
        if abs(high_low_diff) < 15 and abs(mid_low_diff) < 15:
            return 'flat'
        elif high_low_diff > 20 and high_mid_diff > 10:
            return 'high_frequency'
        elif high_low_diff < -20 and mid_low_diff < -10:
            return 'low_frequency'
        elif abs(mid_low_diff) > 15 and abs(high_mid_diff) > 15:
            return 'mixed'
        else:
            return 'irregular'
    
    def _calculate_frequency_slope(self, abr_thresholds: np.ndarray) -> float:
        """Calculate the slope of hearing loss across frequencies."""
        frequencies = np.array([6, 12, 18, 24, 30])  # kHz
        slope, _, _, _, _ = stats.linregress(frequencies, abr_thresholds)
        return slope
    
    def _aggregate_cluster_patterns(self, individual_patterns: List[Dict], 
                                  cluster_abr: np.ndarray) -> Dict[str, Any]:
        """Aggregate individual pattern classifications into cluster-level summary."""
        # Count pattern types and severities
        pattern_counts = {}
        severity_counts = {}
        
        for pattern in individual_patterns:
            pattern_type = pattern['pattern_type']
            severity = pattern['severity']
            
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Determine dominant patterns
        dominant_pattern = max(pattern_counts.items(), key=lambda x: x[1])[0]
        dominant_severity = max(severity_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate cluster-level metrics
        mean_audiogram = np.mean(cluster_abr, axis=0)
        cluster_classification = self.classify_individual_pattern(mean_audiogram)
        
        # Pattern homogeneity
        pattern_homogeneity = max(pattern_counts.values()) / len(individual_patterns)
        severity_homogeneity = max(severity_counts.values()) / len(individual_patterns)
        
        return {
            'size': len(cluster_abr),
            'dominant_pattern': dominant_pattern,
            'dominant_severity': dominant_severity,
            'pattern_distribution': pattern_counts,
            'severity_distribution': severity_counts,
            'pattern_homogeneity': pattern_homogeneity,
            'severity_homogeneity': severity_homogeneity,
            'mean_classification': cluster_classification,
            'mean_audiogram': mean_audiogram,
            'individual_patterns': individual_patterns
        }


class ClinicalSeverityAssessment:
    """
    Clinical assessment of hearing loss severity and functional impact.
    
    Provides clinically relevant metrics for evaluating the severity
    and functional implications of discovered hearing loss phenotypes.
    """
    
    def __init__(self):
        """Initialize clinical severity assessment."""
        # Functional hearing ranges (critical for mouse communication/behavior)
        self.functional_ranges = {
            'communication': [6, 12, 18],    # Mouse communication frequencies
            'environmental': [18, 24, 30],   # Environmental sound detection
            'ultrasonic': [24, 30]           # Ultrasonic range
        }
        
        # Clinical severity scales
        self.severity_weights = {
            'normal': 0,
            'mild': 1,
            'moderate': 2,
            'severe': 3,
            'profound': 4
        }
        
    def assess_functional_impact(self, cluster_patterns: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Assess functional impact of hearing loss for each cluster.
        
        Args:
            cluster_patterns: Cluster pattern classifications
            
        Returns:
            Dictionary of functional impact assessments per cluster
        """
        functional_assessments = {}
        
        for cluster_id, pattern_data in cluster_patterns.items():
            mean_audiogram = pattern_data['mean_audiogram']
            
            # Calculate functional hearing scores
            functional_scores = {}
            for func_name, freq_indices in self.functional_ranges.items():
                avg_threshold = np.mean(mean_audiogram[freq_indices])
                functional_scores[f'{func_name}_threshold'] = avg_threshold
                functional_scores[f'{func_name}_impairment'] = self._calculate_impairment_score(avg_threshold)
            
            # Overall functional impact
            overall_impact = np.mean(list(functional_scores.values())[1::2])  # Impairment scores only
            
            # Communication-specific assessment
            comm_threshold = functional_scores['communication_threshold']
            comm_impact = self._classify_communication_impact(comm_threshold)
            
            functional_assessments[cluster_id] = {
                'functional_scores': functional_scores,
                'overall_functional_impact': overall_impact,
                'communication_impact': comm_impact,
                'environmental_detection_ability': 1 - functional_scores['environmental_impairment'],
                'ultrasonic_sensitivity': 1 - functional_scores['ultrasonic_impairment'],
                'clinical_severity_score': self._calculate_clinical_severity_score(pattern_data)
            }
            
        return functional_assessments
    
    def compare_cluster_severities(self, cluster_patterns: Dict[int, Dict[str, Any]], 
                                 functional_assessments: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare severity profiles across clusters.
        
        Args:
            cluster_patterns: Cluster pattern classifications
            functional_assessments: Functional impact assessments
            
        Returns:
            Comparative severity analysis
        """
        # Extract severity metrics for comparison
        severity_data = []
        for cluster_id in cluster_patterns.keys():
            pattern_data = cluster_patterns[cluster_id]
            func_data = functional_assessments[cluster_id]
            
            severity_data.append({
                'cluster_id': cluster_id,
                'size': pattern_data['size'],
                'dominant_severity': pattern_data['dominant_severity'],
                'severity_weight': self.severity_weights[pattern_data['dominant_severity']],
                'overall_avg': pattern_data['mean_classification']['overall_average'],
                'functional_impact': func_data['overall_functional_impact'],
                'clinical_score': func_data['clinical_severity_score'],
                'pattern_homogeneity': pattern_data['pattern_homogeneity']
            })
        
        severity_df = pd.DataFrame(severity_data)
        
        # Statistical comparisons
        comparisons = {
            'severity_distribution': severity_df['dominant_severity'].value_counts().to_dict(),
            'mean_severity_weight': severity_df['severity_weight'].mean(),
            'severity_range': (severity_df['severity_weight'].min(), severity_df['severity_weight'].max()),
            'most_severe_clusters': severity_df.nlargest(3, 'clinical_score')[['cluster_id', 'clinical_score']].to_dict('records'),
            'mildest_clusters': severity_df.nsmallest(3, 'clinical_score')[['cluster_id', 'clinical_score']].to_dict('records'),
            'high_impact_clusters': severity_df[severity_df['functional_impact'] > 0.7]['cluster_id'].tolist(),
            'homogeneous_clusters': severity_df[severity_df['pattern_homogeneity'] > 0.8]['cluster_id'].tolist()
        }
        
        return comparisons
    
    def _calculate_impairment_score(self, threshold: float) -> float:
        """Calculate functional impairment score (0-1, higher = more impaired)."""
        if threshold <= 25:
            return 0.0
        elif threshold <= 40:
            return 0.2
        elif threshold <= 60:
            return 0.5
        elif threshold <= 80:
            return 0.8
        else:
            return 1.0
    
    def _classify_communication_impact(self, comm_threshold: float) -> str:
        """Classify impact on communication abilities."""
        if comm_threshold <= 30:
            return 'minimal'
        elif comm_threshold <= 50:
            return 'moderate'
        elif comm_threshold <= 70:
            return 'significant'
        else:
            return 'severe'
    
    def _calculate_clinical_severity_score(self, pattern_data: Dict[str, Any]) -> float:
        """Calculate composite clinical severity score."""
        # Weight components: severity (40%), pattern complexity (30%), homogeneity (30%)
        severity_weight = self.severity_weights[pattern_data['dominant_severity']]
        
        # Pattern complexity (more complex patterns are clinically more significant)
        pattern_complexity = len(pattern_data['pattern_distribution']) / 5.0  # Normalized
        
        # Homogeneity (less homogeneous = more clinically complex)
        homogeneity_penalty = 1 - pattern_data['pattern_homogeneity']
        
        clinical_score = (
            0.4 * severity_weight / 4.0 +  # Normalized to 0-1
            0.3 * pattern_complexity +
            0.3 * homogeneity_penalty
        )
        
        return clinical_score


class PhenotypeGeneMapper:
    """
    Maps discovered phenotype clusters to gene knockout effects.
    
    Analyzes gene-cluster associations and identifies genes that
    consistently produce specific hearing loss patterns.
    """
    
    def __init__(self):
        """Initialize phenotype-gene mapper."""
        self.gene_cluster_associations = {}
        self.significant_associations = {}
        
    def analyze_gene_cluster_associations(self, cluster_labels: np.ndarray,
                                        gene_labels: np.ndarray,
                                        min_samples_per_gene: int = 5) -> Dict[str, Any]:
        """
        Analyze associations between genes and clusters.
        
        Args:
            cluster_labels: Cluster assignments
            gene_labels: Gene knockout labels
            min_samples_per_gene: Minimum samples required per gene for analysis
            
        Returns:
            Dictionary of gene-cluster association results
        """
        # Filter out unknown genes and low-sample genes
        valid_mask = gene_labels != -1
        valid_clusters = cluster_labels[valid_mask]
        valid_genes = gene_labels[valid_mask]
        
        # Count samples per gene
        gene_counts = pd.Series(valid_genes).value_counts()
        frequent_genes = gene_counts[gene_counts >= min_samples_per_gene].index
        
        # Filter to frequent genes only
        frequent_mask = np.isin(valid_genes, frequent_genes)
        analysis_clusters = valid_clusters[frequent_mask]
        analysis_genes = valid_genes[frequent_mask]
        
        # Create contingency table
        contingency_table = pd.crosstab(analysis_genes, analysis_clusters)
        
        # Statistical testing
        chi2_stat, chi2_p, chi2_dof, chi2_expected = chi2_contingency(contingency_table)
        
        # Gene-specific associations
        gene_associations = {}
        for gene in frequent_genes:
            gene_mask = analysis_genes == gene
            gene_cluster_dist = analysis_clusters[gene_mask]
            
            # Calculate gene's cluster preference
            cluster_probs = np.bincount(gene_cluster_dist, minlength=len(np.unique(cluster_labels)))
            cluster_probs = cluster_probs / cluster_probs.sum()
            
            # Find dominant cluster(s)
            dominant_cluster = np.argmax(cluster_probs)
            dominant_prob = cluster_probs[dominant_cluster]
            
            # Calculate enrichment vs expected
            expected_prob = np.bincount(analysis_clusters)[dominant_cluster] / len(analysis_clusters)
            enrichment = dominant_prob / expected_prob if expected_prob > 0 else float('inf')
            
            # Statistical significance (Fisher's exact test for dominant cluster)
            gene_in_cluster = np.sum((analysis_genes == gene) & (analysis_clusters == dominant_cluster))
            gene_not_in_cluster = np.sum((analysis_genes == gene) & (analysis_clusters != dominant_cluster))
            not_gene_in_cluster = np.sum((analysis_genes != gene) & (analysis_clusters == dominant_cluster))
            not_gene_not_in_cluster = np.sum((analysis_genes != gene) & (analysis_clusters != dominant_cluster))
            
            fisher_table = [[gene_in_cluster, gene_not_in_cluster],
                          [not_gene_in_cluster, not_gene_not_in_cluster]]
            
            try:
                _, fisher_p = fisher_exact(fisher_table, alternative='greater')
            except:
                fisher_p = 1.0
            
            gene_associations[gene] = {
                'sample_count': len(gene_cluster_dist),
                'cluster_distribution': cluster_probs,
                'dominant_cluster': dominant_cluster,
                'dominant_probability': dominant_prob,
                'enrichment_factor': enrichment,
                'fisher_p_value': fisher_p,
                'is_significant': fisher_p < 0.05,
                'cluster_entropy': -np.sum(cluster_probs[cluster_probs > 0] * np.log2(cluster_probs[cluster_probs > 0]))
            }
        
        return {
            'contingency_table': contingency_table,
            'chi2_stat': chi2_stat,
            'chi2_p_value': chi2_p,
            'overall_significant': chi2_p < 0.05,
            'gene_associations': gene_associations,
            'significant_genes': [gene for gene, assoc in gene_associations.items() if assoc['is_significant']],
            'num_genes_analyzed': len(frequent_genes),
            'total_samples_analyzed': len(analysis_clusters)
        }
    
    def identify_phenotype_specific_genes(self, gene_associations: Dict[str, Any],
                                        cluster_patterns: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Identify genes specifically associated with each phenotype cluster.
        
        Args:
            gene_associations: Gene-cluster association results
            cluster_patterns: Cluster pattern classifications
            
        Returns:
            Dictionary mapping clusters to their associated genes and patterns
        """
        cluster_gene_map = {}
        gene_data = gene_associations['gene_associations']
        
        for cluster_id in cluster_patterns.keys():
            # Find genes significantly enriched in this cluster
            enriched_genes = []
            for gene, assoc in gene_data.items():
                if (assoc['dominant_cluster'] == cluster_id and 
                    assoc['is_significant'] and 
                    assoc['enrichment_factor'] > 1.5):
                    enriched_genes.append({
                        'gene': gene,
                        'enrichment': assoc['enrichment_factor'],
                        'p_value': assoc['fisher_p_value'],
                        'sample_count': assoc['sample_count'],
                        'cluster_probability': assoc['dominant_probability']
                    })
            
            # Sort by enrichment factor
            enriched_genes.sort(key=lambda x: x['enrichment'], reverse=True)
            
            cluster_gene_map[cluster_id] = {
                'enriched_genes': enriched_genes,
                'num_enriched_genes': len(enriched_genes),
                'pattern_type': cluster_patterns[cluster_id]['dominant_pattern'],
                'severity': cluster_patterns[cluster_id]['dominant_severity'],
                'top_genes': enriched_genes[:5] if enriched_genes else []
            }
        
        return cluster_gene_map
    
    def validate_known_hearing_genes(self, gene_associations: Dict[str, Any],
                                   known_hearing_genes: List[str] = None) -> Dict[str, Any]:
        """
        Validate clustering results against known hearing loss genes.
        
        Args:
            gene_associations: Gene-cluster association results  
            known_hearing_genes: List of known hearing loss genes
            
        Returns:
            Validation results
        """
        if known_hearing_genes is None:
            # Common mouse hearing loss genes (partial list)
            known_hearing_genes = [
                'Cdh23', 'Pcdh15', 'Myo7a', 'Ush1c', 'Ush1g', 'Whrn',
                'Otof', 'Slc26a5', 'Gjb2', 'Gjb6', 'Tecta', 'Col11a2',
                'Tmprss3', 'Strc', 'Loxhd1', 'Tmc1', 'Cib2', 'Espn'
            ]
        
        gene_data = gene_associations['gene_associations']
        
        # Check which known genes are in our data
        known_genes_present = [gene for gene in known_hearing_genes if gene in gene_data]
        
        # Analyze clustering behavior of known genes
        known_gene_analysis = {}
        for gene in known_genes_present:
            assoc = gene_data[gene]
            known_gene_analysis[gene] = {
                'clustered_significantly': assoc['is_significant'],
                'dominant_cluster': assoc['dominant_cluster'],
                'cluster_specificity': assoc['dominant_probability'],
                'enrichment': assoc['enrichment_factor'],
                'sample_count': assoc['sample_count']
            }
        
        # Summary statistics
        known_genes_clustered = sum(1 for data in known_gene_analysis.values() if data['clustered_significantly'])
        validation_rate = known_genes_clustered / len(known_genes_present) if known_genes_present else 0
        
        return {
            'known_genes_in_data': known_genes_present,
            'known_genes_analysis': known_gene_analysis,
            'validation_rate': validation_rate,
            'genes_clustered_significantly': known_genes_clustered,
            'mean_specificity': np.mean([data['cluster_specificity'] for data in known_gene_analysis.values()]),
            'mean_enrichment': np.mean([data['enrichment'] for data in known_gene_analysis.values() if data['clustered_significantly']])
        }


class StatisticalValidator:
    """
    Statistical validation of discovered phenotype clusters.
    
    Provides rigorous statistical tests to validate the biological
    significance and reproducibility of discovered clusters.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical validator.
        
        Args:
            alpha: Significance level for statistical tests
        """
        self.alpha = alpha
        
    def validate_cluster_separation(self, cluster_labels: np.ndarray,
                                  abr_features: np.ndarray) -> Dict[str, Any]:
        """
        Validate statistical separation between clusters based on ABR features.
        
        Args:
            cluster_labels: Cluster assignments
            abr_features: ABR threshold features
            
        Returns:
            Statistical validation results
        """
        validation_results = {}
        
        # ANOVA for each frequency
        frequency_names = ['6kHz', '12kHz', '18kHz', '24kHz', '30kHz', 'Click']
        
        anova_results = {}
        for i, freq_name in enumerate(frequency_names):
            if i < abr_features.shape[1]:
                freq_data = abr_features[:, i]
                
                # Group data by cluster
                cluster_groups = [freq_data[cluster_labels == cid] for cid in np.unique(cluster_labels)]
                
                # Perform ANOVA
                f_stat, p_value = stats.f_oneway(*cluster_groups)
                
                anova_results[freq_name] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                }
        
        validation_results['frequency_anova'] = anova_results
        
        # Overall multivariate test (MANOVA approximation using Pillai's trace)
        try:
            from scipy.stats import multivariate_normal
            # Simplified multivariate analysis
            cluster_means = []
            for cid in np.unique(cluster_labels):
                cluster_data = abr_features[cluster_labels == cid]
                cluster_means.append(np.mean(cluster_data, axis=0))
            
            # Calculate between-cluster variance
            overall_mean = np.mean(abr_features, axis=0)
            between_cluster_var = np.var(cluster_means, axis=0)
            within_cluster_var = np.var(abr_features, axis=0) - between_cluster_var
            
            # F-ratio for multivariate test
            multivariate_f = np.mean(between_cluster_var / (within_cluster_var + 1e-8))
            
            validation_results['multivariate_separation'] = {
                'f_ratio': multivariate_f,
                'separation_strength': 'strong' if multivariate_f > 5 else 'moderate' if multivariate_f > 2 else 'weak'
            }
            
        except Exception as e:
            logger.warning(f"Could not compute multivariate test: {e}")
        
        # Pairwise cluster comparisons
        pairwise_comparisons = {}
        unique_clusters = np.unique(cluster_labels)
        
        for i, cluster1 in enumerate(unique_clusters):
            for cluster2 in unique_clusters[i+1:]:
                mask1 = cluster_labels == cluster1
                mask2 = cluster_labels == cluster2
                
                # Mann-Whitney U test for each frequency
                freq_comparisons = {}
                for freq_idx, freq_name in enumerate(frequency_names):
                    if freq_idx < abr_features.shape[1]:
                        data1 = abr_features[mask1, freq_idx]
                        data2 = abr_features[mask2, freq_idx]
                        
                        u_stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                        
                        freq_comparisons[freq_name] = {
                            'u_statistic': u_stat,
                            'p_value': p_value,
                            'significant': p_value < self.alpha,
                            'effect_size': abs(np.mean(data1) - np.mean(data2)) / np.sqrt((np.var(data1) + np.var(data2)) / 2)
                        }
                
                pairwise_comparisons[f'cluster_{cluster1}_vs_{cluster2}'] = freq_comparisons
        
        validation_results['pairwise_comparisons'] = pairwise_comparisons
        
        # Summary metrics
        significant_frequencies = sum(1 for result in anova_results.values() if result['significant'])
        total_frequencies = len(anova_results)
        
        validation_results['summary'] = {
            'significant_frequencies': significant_frequencies,
            'total_frequencies': total_frequencies,
            'proportion_significant': significant_frequencies / total_frequencies,
            'overall_separation_valid': significant_frequencies >= total_frequencies * 0.6
        }
        
        return validation_results
    
    def assess_cluster_stability(self, cluster_labels_list: List[np.ndarray],
                               method: str = 'ari') -> Dict[str, float]:
        """
        Assess stability of clustering across multiple runs.
        
        Args:
            cluster_labels_list: List of cluster label arrays from different runs
            method: Stability metric ('ari', 'nmi', 'jaccard')
            
        Returns:
            Stability assessment results
        """
        if len(cluster_labels_list) < 2:
            return {'error': 'need_multiple_runs'}
        
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        # Compute pairwise stability
        stabilities = []
        for i in range(len(cluster_labels_list)):
            for j in range(i + 1, len(cluster_labels_list)):
                labels1 = cluster_labels_list[i]
                labels2 = cluster_labels_list[j]
                
                if method == 'ari':
                    stability = adjusted_rand_score(labels1, labels2)
                elif method == 'nmi':
                    stability = normalized_mutual_info_score(labels1, labels2)
                else:
                    stability = adjusted_rand_score(labels1, labels2)  # Default
                
                stabilities.append(stability)
        
        return {
            'mean_stability': np.mean(stabilities),
            'std_stability': np.std(stabilities),
            'min_stability': np.min(stabilities),
            'max_stability': np.max(stabilities),
            'stability_scores': stabilities,
            'is_stable': np.mean(stabilities) > 0.7  # Threshold for good stability
        }


def perform_comprehensive_phenotype_analysis(cluster_labels: np.ndarray,
                                           abr_features: np.ndarray,
                                           gene_labels: Optional[np.ndarray] = None,
                                           known_hearing_genes: Optional[List[str]] = None,
                                           save_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Perform comprehensive phenotype analysis of clustering results.
    
    Args:
        cluster_labels: Cluster assignments
        abr_features: ABR threshold features
        gene_labels: Gene knockout labels (optional)
        known_hearing_genes: List of known hearing loss genes (optional)
        save_dir: Directory to save analysis results
        
    Returns:
        Complete phenotype analysis results
    """
    logger.info("Starting comprehensive phenotype analysis")
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_results = {}
    
    # 1. Hearing loss pattern classification
    logger.info("Classifying hearing loss patterns")
    classifier = HearingLossClassifier()
    cluster_patterns = classifier.classify_cluster_patterns(cluster_labels, abr_features)
    analysis_results['cluster_patterns'] = cluster_patterns
    
    # 2. Clinical severity assessment
    logger.info("Assessing clinical severity")
    severity_assessor = ClinicalSeverityAssessment()
    functional_assessments = severity_assessor.assess_functional_impact(cluster_patterns)
    severity_comparisons = severity_assessor.compare_cluster_severities(cluster_patterns, functional_assessments)
    
    analysis_results['functional_assessments'] = functional_assessments
    analysis_results['severity_comparisons'] = severity_comparisons
    
    # 3. Gene-phenotype mapping (if gene labels available)
    if gene_labels is not None:
        logger.info("Analyzing gene-phenotype associations")
        gene_mapper = PhenotypeGeneMapper()
        gene_associations = gene_mapper.analyze_gene_cluster_associations(cluster_labels, gene_labels)
        cluster_gene_map = gene_mapper.identify_phenotype_specific_genes(gene_associations, cluster_patterns)
        
        analysis_results['gene_associations'] = gene_associations
        analysis_results['cluster_gene_map'] = cluster_gene_map
        
        # Validate against known hearing genes
        if known_hearing_genes:
            validation_results = gene_mapper.validate_known_hearing_genes(gene_associations, known_hearing_genes)
            analysis_results['known_gene_validation'] = validation_results
    
    # 4. Statistical validation
    logger.info("Performing statistical validation")
    validator = StatisticalValidator()
    cluster_separation = validator.validate_cluster_separation(cluster_labels, abr_features)
    analysis_results['statistical_validation'] = cluster_separation
    
    # 5. Generate summary report
    analysis_results['summary'] = _generate_analysis_summary(analysis_results)
    
    # Save results if directory specified
    if save_dir:
        _save_analysis_results(analysis_results, save_dir)
    
    logger.info("Comprehensive phenotype analysis completed")
    return analysis_results


def _generate_analysis_summary(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate high-level summary of phenotype analysis."""
    summary = {}
    
    # Cluster pattern summary
    if 'cluster_patterns' in analysis_results:
        patterns = analysis_results['cluster_patterns']
        pattern_types = [p['dominant_pattern'] for p in patterns.values()]
        severity_types = [p['dominant_severity'] for p in patterns.values()]
        
        summary['pattern_diversity'] = len(set(pattern_types))
        summary['severity_diversity'] = len(set(severity_types))
        summary['most_common_pattern'] = max(set(pattern_types), key=pattern_types.count)
        summary['most_common_severity'] = max(set(severity_types), key=severity_types.count)
        summary['num_clusters'] = len(patterns)
    
    # Gene association summary
    if 'gene_associations' in analysis_results:
        gene_assoc = analysis_results['gene_associations']
        summary['significant_gene_associations'] = gene_assoc.get('overall_significant', False)
        summary['num_significant_genes'] = len(gene_assoc.get('significant_genes', []))
        
        if 'known_gene_validation' in analysis_results:
            validation = analysis_results['known_gene_validation']
            summary['known_gene_validation_rate'] = validation.get('validation_rate', 0)
    
    # Statistical validation summary
    if 'statistical_validation' in analysis_results:
        stat_val = analysis_results['statistical_validation']
        summary['cluster_separation_valid'] = stat_val.get('summary', {}).get('overall_separation_valid', False)
        summary['proportion_significant_frequencies'] = stat_val.get('summary', {}).get('proportion_significant', 0)
    
    # Clinical significance summary
    if 'severity_comparisons' in analysis_results:
        severity = analysis_results['severity_comparisons']
        summary['severity_range'] = severity.get('severity_range', (0, 0))
        summary['num_high_impact_clusters'] = len(severity.get('high_impact_clusters', []))
        summary['num_homogeneous_clusters'] = len(severity.get('homogeneous_clusters', []))
    
    return summary


def _save_analysis_results(analysis_results: Dict[str, Any], save_dir: Path):
    """Save analysis results to files."""
    import json
    import pickle
    
    # Save summary as JSON
    summary_file = save_dir / 'phenotype_analysis_summary.json'
    with open(summary_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        summary_json = {}
        for key, value in analysis_results['summary'].items():
            if isinstance(value, (np.integer, np.floating)):
                summary_json[key] = value.item()
            elif isinstance(value, tuple):
                summary_json[key] = list(value)
            else:
                summary_json[key] = value
        json.dump(summary_json, f, indent=2)
    
    # Save complete results as pickle
    results_file = save_dir / 'phenotype_analysis_complete.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(analysis_results, f)
    
    logger.info(f"Saved phenotype analysis results to {save_dir}")


def create_phenotype_analysis_report(analysis_results: Dict[str, Any], 
                                   output_file: Optional[Path] = None) -> str:
    """
    Create a human-readable phenotype analysis report.
    
    Args:
        analysis_results: Complete analysis results
        output_file: Optional file to save the report
        
    Returns:
        Formatted text report
    """
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("AUDIOMETRIC PHENOTYPE ANALYSIS REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Summary section
    if 'summary' in analysis_results:
        summary = analysis_results['summary']
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Number of clusters discovered: {summary.get('num_clusters', 'N/A')}")
        report_lines.append(f"Pattern diversity: {summary.get('pattern_diversity', 'N/A')} distinct patterns")
        report_lines.append(f"Severity diversity: {summary.get('severity_diversity', 'N/A')} severity levels")
        report_lines.append(f"Most common pattern: {summary.get('most_common_pattern', 'N/A')}")
        report_lines.append(f"Most common severity: {summary.get('most_common_severity', 'N/A')}")
        
        if 'known_gene_validation_rate' in summary:
            report_lines.append(f"Known gene validation rate: {summary['known_gene_validation_rate']:.2%}")
        
        report_lines.append("")
    
    # Cluster patterns section
    if 'cluster_patterns' in analysis_results:
        report_lines.append("CLUSTER PATTERN ANALYSIS")
        report_lines.append("-" * 40)
        
        for cluster_id, pattern_data in analysis_results['cluster_patterns'].items():
            report_lines.append(f"\nCluster {cluster_id}:")
            report_lines.append(f"  Size: {pattern_data['size']} mice")
            report_lines.append(f"  Dominant pattern: {pattern_data['dominant_pattern']}")
            report_lines.append(f"  Dominant severity: {pattern_data['dominant_severity']}")
            report_lines.append(f"  Pattern homogeneity: {pattern_data['pattern_homogeneity']:.2%}")
            
            # Mean audiogram summary
            mean_class = pattern_data['mean_classification']
            report_lines.append(f"  Overall average threshold: {mean_class['overall_average']:.1f} dB SPL")
            report_lines.append(f"  Frequency slope: {mean_class['frequency_slope']:.2f} dB/kHz")
    
    # Gene associations section  
    if 'gene_associations' in analysis_results:
        gene_assoc = analysis_results['gene_associations']
        report_lines.append("\n\nGENE-PHENOTYPE ASSOCIATIONS")
        report_lines.append("-" * 40)
        report_lines.append(f"Genes analyzed: {gene_assoc['num_genes_analyzed']}")
        report_lines.append(f"Significant associations: {'Yes' if gene_assoc['overall_significant'] else 'No'}")
        report_lines.append(f"Significantly enriched genes: {len(gene_assoc['significant_genes'])}")
        
        if gene_assoc['significant_genes']:
            report_lines.append("\nTop significantly enriched genes:")
            for gene in gene_assoc['significant_genes'][:10]:  # Top 10
                gene_data = gene_assoc['gene_associations'][gene]
                report_lines.append(f"  {gene}: Cluster {gene_data['dominant_cluster']} "
                                  f"(enrichment: {gene_data['enrichment_factor']:.2f}x, "
                                  f"p={gene_data['fisher_p_value']:.2e})")
    
    # Statistical validation section
    if 'statistical_validation' in analysis_results:
        stat_val = analysis_results['statistical_validation']
        report_lines.append("\n\nSTATISTICAL VALIDATION")
        report_lines.append("-" * 40)
        
        if 'summary' in stat_val:
            summary_stats = stat_val['summary']
            report_lines.append(f"Frequencies with significant cluster separation: "
                              f"{summary_stats['significant_frequencies']}/{summary_stats['total_frequencies']}")
            report_lines.append(f"Overall separation validity: {'Valid' if summary_stats['overall_separation_valid'] else 'Invalid'}")
        
        # Frequency-specific results
        if 'frequency_anova' in stat_val:
            report_lines.append("\nFrequency-specific separation (ANOVA):")
            for freq, result in stat_val['frequency_anova'].items():
                significance = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "ns"
                report_lines.append(f"  {freq}: F={result['f_statistic']:.2f}, p={result['p_value']:.2e} {significance}")
    
    # Clinical significance section
    if 'severity_comparisons' in analysis_results:
        severity = analysis_results['severity_comparisons']
        report_lines.append("\n\nCLINICAL SIGNIFICANCE")
        report_lines.append("-" * 40)
        
        report_lines.append("Severity distribution:")
        for severity_level, count in severity['severity_distribution'].items():
            report_lines.append(f"  {severity_level}: {count} clusters")
        
        if 'most_severe_clusters' in severity:
            report_lines.append(f"\nMost clinically severe clusters:")
            for cluster_info in severity['most_severe_clusters']:
                report_lines.append(f"  Cluster {cluster_info['cluster_id']}: "
                                  f"Clinical score {cluster_info['clinical_score']:.3f}")
    
    report_lines.append("\n" + "="*80)
    
    # Join all lines
    report_text = "\n".join(report_lines)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        logger.info(f"Saved phenotype analysis report to {output_file}")
    
    return report_text


# Export main functions
__all__ = [
    'HearingLossClassifier',
    'ClinicalSeverityAssessment', 
    'PhenotypeGeneMapper',
    'StatisticalValidator',
    'perform_comprehensive_phenotype_analysis',
    'create_phenotype_analysis_report'
]