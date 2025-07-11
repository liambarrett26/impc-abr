#!/usr/bin/env python3
"""
Analyze associations between Bayesian-identified significant genes and GMM clusters.

This script maps significant gene/allele/zygosity/center combinations from Bayesian
analysis to their dominant GMM cluster assignments and calculates variability metrics.

Usage:
    python analyze_gene_cluster_associations.py [--mode MODE]
    
Modes:
    full (default): Run complete analysis with visualizations and report
    sex-specific: Show detailed sex-specific gene effects
    sex-differences: Show genes with sex-specific effects only
    demo-csv: Show structure and examples from the output CSV
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import argparse
import sys


def load_data():
    """Load all necessary data files."""
    # Load Bayesian significant results
    bayesian_df = pd.read_csv('shared_data/concatenated_results_v6.csv')
    
    # Load metadata with mouse information
    metadata_df = pd.read_csv('shared_data/metadata.csv')
    
    # Load cluster assignments
    cluster_labels = np.load('results/gmm_k4_tied/cluster_labels.npy')
    cluster_probs = np.load('results/gmm_k4_tied/cluster_probabilities.npy')
    
    # Load analysis results for cluster characteristics
    with open('results/gmm_k4_tied/analysis_results.json', 'r') as f:
        analysis_results = json.load(f)
    
    return bayesian_df, metadata_df, cluster_labels, cluster_probs, analysis_results


def filter_significant_genes(bayesian_df, bf_threshold=3.0, p_threshold=0.5):
    """Filter for significant genes based on Bayes Factor and probability of hearing loss.
    
    This function now captures ANY gene with a significant effect in any analysis type.
    Returns a DataFrame with all significant genes and their most significant analysis type.
    """
    # Create masks for each analysis type
    all_sig = (
        (bayesian_df['all_bayes_factor'] >= bf_threshold) & 
        (bayesian_df['all_p_hearing_loss'] >= p_threshold)
    )
    
    male_sig = (
        (bayesian_df['male_bayes_factor'].notna()) &
        (bayesian_df['male_bayes_factor'] >= bf_threshold) & 
        (bayesian_df['male_p_hearing_loss'] >= p_threshold)
    )
    
    female_sig = (
        (bayesian_df['female_bayes_factor'].notna()) &
        (bayesian_df['female_bayes_factor'] >= bf_threshold) & 
        (bayesian_df['female_p_hearing_loss'] >= p_threshold)
    )
    
    # Get all genes with ANY significant effect
    any_sig = all_sig | male_sig | female_sig
    sig_genes_df = bayesian_df[any_sig].copy()
    
    # Determine the primary analysis type for each gene (highest BF)
    def get_primary_analysis(row):
        bf_values = {
            'all': row['all_bayes_factor'] if pd.notna(row['all_bayes_factor']) else 0,
            'male': row['male_bayes_factor'] if pd.notna(row['male_bayes_factor']) else 0,
            'female': row['female_bayes_factor'] if pd.notna(row['female_bayes_factor']) else 0
        }
        # Only consider significant effects
        if not (row['all_bayes_factor'] >= bf_threshold and row['all_p_hearing_loss'] >= p_threshold):
            bf_values['all'] = 0
        if not (pd.notna(row['male_bayes_factor']) and row['male_bayes_factor'] >= bf_threshold and row['male_p_hearing_loss'] >= p_threshold):
            bf_values['male'] = 0
        if not (pd.notna(row['female_bayes_factor']) and row['female_bayes_factor'] >= bf_threshold and row['female_p_hearing_loss'] >= p_threshold):
            bf_values['female'] = 0
        
        return max(bf_values.items(), key=lambda x: x[1])[0]
    
    sig_genes_df['primary_analysis_type'] = sig_genes_df.apply(get_primary_analysis, axis=1)
    
    # Add flags for which analyses are significant
    sig_genes_df['sig_in_all'] = all_sig[any_sig]
    sig_genes_df['sig_in_male'] = male_sig[any_sig]
    sig_genes_df['sig_in_female'] = female_sig[any_sig]
    
    # Group by primary analysis type for processing
    significant_groups = []
    for analysis_type in ['all', 'male', 'female']:
        type_df = sig_genes_df[sig_genes_df['primary_analysis_type'] == analysis_type].copy()
        if len(type_df) > 0:
            type_df['analysis_type'] = analysis_type
            significant_groups.append((type_df, analysis_type))
    
    return significant_groups


def create_gene_mouse_mapping(sig_genes_df, metadata_df, cluster_labels, analysis_type='all'):
    """Map each significant gene group to its mice and their cluster assignments."""
    # Add cluster assignments to metadata
    metadata_df = metadata_df.copy()
    metadata_df['cluster'] = cluster_labels
    
    gene_cluster_mapping = []
    
    for _, gene_row in sig_genes_df.iterrows():
        # Extract gene group identifiers
        gene_symbol = gene_row['gene_symbol']
        allele_symbol = gene_row['allele_symbol']
        zygosity = gene_row['zygosity']
        center = gene_row['center']
        
        # Find matching mice in metadata
        mask = (
            (metadata_df['gene_symbol'] == gene_symbol) &
            (metadata_df['allele_symbol'] == allele_symbol) &
            (metadata_df['zygosity'] == zygosity) &
            (metadata_df['phenotyping_center'] == center)
        )
        
        # Apply sex-specific filtering if needed
        if analysis_type == 'male':
            mask = mask & (metadata_df['sex'] == 'male')
        elif analysis_type == 'female':
            mask = mask & (metadata_df['sex'] == 'female')
        
        matching_mice = metadata_df[mask]
        
        if len(matching_mice) > 0:
            # Get overall cluster assignments
            clusters = matching_mice['cluster'].values
            cluster_counts = Counter(clusters)
            
            # Find dominant cluster
            dominant_cluster = max(cluster_counts, key=cluster_counts.get)
            n_dominant = cluster_counts[dominant_cluster]
            n_total = len(clusters)
            consistency_score = n_dominant / n_total
            
            # Calculate distribution across all clusters
            cluster_dist = {f'cluster_{i}': cluster_counts.get(i, 0) for i in range(4)}
            
            # Calculate sex-specific metrics if this is an 'all' analysis
            sex_specific_metrics = {}
            if analysis_type == 'all':
                for sex in ['male', 'female']:
                    sex_mice = matching_mice[matching_mice['sex'] == sex]
                    if len(sex_mice) > 0:
                        sex_clusters = sex_mice['cluster'].values
                        sex_counts = Counter(sex_clusters)
                        sex_dominant = max(sex_counts, key=sex_counts.get)
                        sex_consistency = sex_counts[sex_dominant] / len(sex_clusters)
                        
                        sex_specific_metrics.update({
                            f'{sex}_n_mice': len(sex_mice),
                            f'{sex}_dominant_cluster': sex_dominant,
                            f'{sex}_consistency_score': sex_consistency,
                            f'{sex}_cluster_0': sex_counts.get(0, 0),
                            f'{sex}_cluster_1': sex_counts.get(1, 0),
                            f'{sex}_cluster_2': sex_counts.get(2, 0),
                            f'{sex}_cluster_3': sex_counts.get(3, 0),
                        })
                    else:
                        sex_specific_metrics.update({
                            f'{sex}_n_mice': 0,
                            f'{sex}_dominant_cluster': None,
                            f'{sex}_consistency_score': None,
                            f'{sex}_cluster_0': 0,
                            f'{sex}_cluster_1': 0,
                            f'{sex}_cluster_2': 0,
                            f'{sex}_cluster_3': 0,
                        })
            
            # Use appropriate columns based on analysis type
            bf_col = f'{analysis_type}_bayes_factor'
            p_col = f'{analysis_type}_p_hearing_loss'
            n_col = f'{analysis_type}_n_mutants'
            key_col = f'{analysis_type}_analysis_key'
            
            mapping_entry = {
                'gene_symbol': gene_symbol,
                'allele_symbol': allele_symbol,
                'zygosity': zygosity,
                'center': center,
                'analysis_type': analysis_type,
                'analysis_key': gene_row.get(key_col, f"{gene_symbol}_{allele_symbol}_{zygosity}_{center}_{analysis_type}"),
                'bayes_factor': gene_row[bf_col],
                'p_hearing_loss': gene_row[p_col],
                'n_mutants_reported': gene_row[n_col],
                'n_total_mice': n_total,
                'dominant_cluster': dominant_cluster,
                'n_dominant': n_dominant,
                'consistency_score': consistency_score,
                **cluster_dist,
                'cluster_distribution': dict(cluster_counts),
                **sex_specific_metrics,
                # Add significance flags
                'sig_in_all': gene_row.get('sig_in_all', False),
                'sig_in_male': gene_row.get('sig_in_male', False),
                'sig_in_female': gene_row.get('sig_in_female', False),
                # Add all BF values for comparison
                'all_bayes_factor': gene_row.get('all_bayes_factor', None),
                'male_bayes_factor': gene_row.get('male_bayes_factor', None),
                'female_bayes_factor': gene_row.get('female_bayes_factor', None)
            }
            
            gene_cluster_mapping.append(mapping_entry)
    
    return pd.DataFrame(gene_cluster_mapping)


def analyze_cluster_patterns(gene_cluster_df, analysis_results):
    """Analyze patterns in gene-cluster associations."""
    # Map cluster IDs to their characteristics
    cluster_patterns = {
        0: 'moderate_HL',
        1: 'control/normal',
        2: 'high_frequency_loss',
        3: 'severe_HL'
    }
    
    gene_cluster_df['cluster_pattern'] = gene_cluster_df['dominant_cluster'].map(cluster_patterns)
    
    # Summary statistics
    summary_stats = {
        'total_significant_genes': len(gene_cluster_df),
        'genes_per_cluster': gene_cluster_df['dominant_cluster'].value_counts().to_dict(),
        'mean_consistency_score': gene_cluster_df['consistency_score'].mean(),
        'consistency_by_cluster': gene_cluster_df.groupby('dominant_cluster')['consistency_score'].agg(['mean', 'std']).to_dict(),
        'high_consistency_genes': len(gene_cluster_df[gene_cluster_df['consistency_score'] >= 0.8]),
        'mixed_phenotype_genes': len(gene_cluster_df[gene_cluster_df['consistency_score'] < 0.6])
    }
    
    return gene_cluster_df, summary_stats


def create_visualizations(gene_cluster_df, output_dir):
    """Create visualizations of gene-cluster associations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    
    # 1. Distribution of genes across dominant clusters
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cluster_counts = gene_cluster_df['dominant_cluster'].value_counts().sort_index()
    cluster_labels = ['Moderate HL', 'Control/Normal', 'High-Freq Loss', 'Severe HL']
    
    bars = ax.bar(cluster_counts.index, cluster_counts.values)
    ax.set_xticks(range(4))
    ax.set_xticklabels(cluster_labels, rotation=45, ha='right')
    ax.set_xlabel('Dominant Cluster')
    ax.set_ylabel('Number of Significant Genes')
    ax.set_title('Distribution of Significant Genes Across GMM Clusters')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gene_cluster_distribution.png')
    plt.close()
    
    # 2. Consistency scores by cluster
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create violin plot
    data_for_plot = []
    for cluster in range(4):
        cluster_data = gene_cluster_df[gene_cluster_df['dominant_cluster'] == cluster]['consistency_score']
        data_for_plot.append(cluster_data)
    
    parts = ax.violinplot(data_for_plot, positions=range(4), showmeans=True, showmedians=True)
    
    ax.set_xticks(range(4))
    ax.set_xticklabels(cluster_labels, rotation=45, ha='right')
    ax.set_xlabel('Dominant Cluster')
    ax.set_ylabel('Consistency Score')
    ax.set_title('Phenotypic Consistency of Genes by Dominant Cluster')
    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='High consistency threshold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gene_consistency_by_cluster.png')
    plt.close()
    
    # 3. Heatmap of gene distribution across all clusters
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Prepare data for heatmap
    heatmap_data = gene_cluster_df[['gene_symbol', 'cluster_0', 'cluster_1', 'cluster_2', 'cluster_3']].copy()
    heatmap_data = heatmap_data.set_index('gene_symbol')
    
    # Normalize by row to show proportion
    heatmap_data_norm = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)
    
    # Select top genes by effect size for visualization
    top_genes = gene_cluster_df.nlargest(30, 'bayes_factor')
    heatmap_subset = heatmap_data_norm.loc[top_genes['gene_symbol']]
    
    sns.heatmap(heatmap_subset, 
                cmap='YlOrRd',
                xticklabels=cluster_labels,
                yticklabels=True,
                cbar_kws={'label': 'Proportion of Mice'},
                ax=ax)
    
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Gene Symbol')
    ax.set_title('Distribution of Top 30 Genes Across Clusters (by Bayes Factor)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gene_cluster_heatmap.png', dpi=200)
    plt.close()
    
    # 4. Scatter plot: Bayes Factor vs Consistency Score
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    colors = ['orange', 'green', 'blue', 'red']
    markers = {'all': 'o', 'male': '^', 'female': 's'}
    
    for cluster in range(4):
        for atype in ['all', 'male', 'female']:
            cluster_data = gene_cluster_df[
                (gene_cluster_df['dominant_cluster'] == cluster) & 
                (gene_cluster_df['analysis_type'] == atype)
            ]
            if len(cluster_data) > 0:
                ax.scatter(cluster_data['bayes_factor'], 
                          cluster_data['consistency_score'],
                          c=colors[cluster],
                          marker=markers[atype],
                          label=f'{cluster_labels[cluster]} ({atype})',
                          alpha=0.6,
                          s=50)
    
    ax.set_xlabel('Bayes Factor (log scale)')
    ax.set_ylabel('Consistency Score')
    ax.set_xscale('log')
    ax.set_title('Gene Effect Size vs Phenotypic Consistency by Sex')
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for cluster in range(4):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=colors[cluster], markersize=8,
                                     label=cluster_labels[cluster]))
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='gray', markersize=8, label='All'),
        Line2D([0], [0], marker='^', color='gray', markersize=8, label='Male'),
        Line2D([0], [0], marker='s', color='gray', markersize=8, label='Female')
    ])
    
    ax.legend(handles=legend_elements, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bayes_factor_vs_consistency.png')
    plt.close()
    
    # 5. Sex-specific cluster distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, atype in enumerate(['all', 'male', 'female']):
        ax = axes[idx]
        atype_data = gene_cluster_df[gene_cluster_df['analysis_type'] == atype]
        if len(atype_data) > 0:
            cluster_counts = atype_data['dominant_cluster'].value_counts().sort_index()
            bars = ax.bar(cluster_counts.index, cluster_counts.values, color=colors)
            ax.set_xticks(range(4))
            ax.set_xticklabels(cluster_labels, rotation=45, ha='right')
            ax.set_xlabel('Dominant Cluster')
            ax.set_ylabel('Number of Genes')
            ax.set_title(f'{atype.capitalize()} Analysis')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
    
    plt.suptitle('Gene Distribution Across Clusters by Analysis Type')
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_distribution_by_sex.png')
    plt.close()


def generate_report(gene_cluster_df, summary_stats, output_dir):
    """Generate a comprehensive text report."""
    output_dir = Path(output_dir)
    
    report = []
    report.append("="*80)
    report.append("GENE-CLUSTER ASSOCIATION ANALYSIS REPORT")
    report.append("GMM Clustering (k=4, tied covariance) vs Bayesian Significant Genes")
    report.append("="*80)
    report.append("")
    
    # Summary statistics
    report.append("SUMMARY STATISTICS")
    report.append("-"*40)
    report.append(f"Total significant genes analyzed: {summary_stats['total_significant_genes']}")
    report.append(f"Mean consistency score: {summary_stats['mean_consistency_score']:.3f}")
    report.append(f"Genes with high consistency (≥0.8): {summary_stats['high_consistency_genes']}")
    report.append(f"Genes with mixed phenotypes (<0.6): {summary_stats['mixed_phenotype_genes']}")
    report.append("")
    
    # Genes by analysis type
    if 'genes_by_analysis_type' in summary_stats:
        report.append("GENES BY ANALYSIS TYPE")
        report.append("-"*40)
        for atype, count in summary_stats['genes_by_analysis_type'].items():
            report.append(f"{atype.capitalize()}: {count} genes")
        report.append("")
    
    # Distribution across clusters
    report.append("GENE DISTRIBUTION ACROSS CLUSTERS")
    report.append("-"*40)
    cluster_names = {0: 'Moderate HL', 1: 'Control/Normal', 2: 'High-Freq Loss', 3: 'Severe HL'}
    for cluster, count in sorted(summary_stats['genes_per_cluster'].items()):
        pct = (count / summary_stats['total_significant_genes']) * 100
        report.append(f"Cluster {cluster} ({cluster_names[cluster]}): {count} genes ({pct:.1f}%)")
    report.append("")
    
    # Consistency by cluster
    report.append("PHENOTYPIC CONSISTENCY BY CLUSTER")
    report.append("-"*40)
    for cluster in range(4):
        if cluster in summary_stats['consistency_by_cluster']:
            stats = summary_stats['consistency_by_cluster'][cluster]
            report.append(f"Cluster {cluster} ({cluster_names[cluster]}):")
            report.append(f"  Mean consistency: {stats['mean']:.3f} ± {stats['std']:.3f}")
    report.append("")
    
    # Top genes by cluster
    report.append("TOP GENES BY DOMINANT CLUSTER (by Bayes Factor)")
    report.append("-"*40)
    for cluster in range(4):
        cluster_genes = gene_cluster_df[gene_cluster_df['dominant_cluster'] == cluster].nlargest(5, 'bayes_factor')
        report.append(f"\nCluster {cluster} ({cluster_names[cluster]}):")
        for _, gene in cluster_genes.iterrows():
            report.append(f"  {gene['gene_symbol']} - BF: {gene['bayes_factor']:.1f}, "
                         f"Consistency: {gene['consistency_score']:.2f}, "
                         f"Mice: {gene['n_total_mice']}")
    report.append("")
    
    # Genes with mixed phenotypes
    report.append("GENES WITH MIXED PHENOTYPES (Consistency < 0.6)")
    report.append("-"*40)
    mixed_genes = gene_cluster_df[gene_cluster_df['consistency_score'] < 0.6].nlargest(10, 'bayes_factor')
    for _, gene in mixed_genes.iterrows():
        dist = gene['cluster_distribution']
        dist_str = ', '.join([f"C{k}:{v}" for k, v in sorted(dist.items())])
        report.append(f"{gene['gene_symbol']} - Consistency: {gene['consistency_score']:.2f}, "
                     f"Distribution: [{dist_str}]")
    report.append("")
    
    # Sex-specific consistency analysis (for 'all' analysis genes)
    all_genes = gene_cluster_df[gene_cluster_df['analysis_type'] == 'all']
    if len(all_genes) > 0 and 'male_consistency_score' in all_genes.columns:
        report.append("SEX-SPECIFIC CONSISTENCY ANALYSIS")
        report.append("-"*40)
        
        # Find genes with different dominant clusters by sex
        sex_diff_genes = all_genes[
            (all_genes['male_dominant_cluster'].notna()) & 
            (all_genes['female_dominant_cluster'].notna()) &
            (all_genes['male_dominant_cluster'] != all_genes['female_dominant_cluster'])
        ]
        
        if len(sex_diff_genes) > 0:
            report.append("Genes with different dominant clusters by sex:")
            for _, gene in sex_diff_genes.nlargest(10, 'bayes_factor').iterrows():
                male_cluster = int(gene['male_dominant_cluster']) if pd.notna(gene['male_dominant_cluster']) else 'N/A'
                female_cluster = int(gene['female_dominant_cluster']) if pd.notna(gene['female_dominant_cluster']) else 'N/A'
                report.append(f"  {gene['gene_symbol']} - Male: Cluster {male_cluster}, Female: Cluster {female_cluster}")
        
        # Summary of sex-specific consistency
        male_consistency = all_genes['male_consistency_score'].dropna()
        female_consistency = all_genes['female_consistency_score'].dropna()
        if len(male_consistency) > 0 and len(female_consistency) > 0:
            report.append("")
            report.append(f"Mean male consistency: {male_consistency.mean():.3f}")
            report.append(f"Mean female consistency: {female_consistency.mean():.3f}")
    
    # Write report
    with open(output_dir / 'gene_cluster_association_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    # Also save detailed results as CSV
    gene_cluster_df.to_csv(output_dir / 'gene_cluster_associations.csv', index=False)


def show_sex_specific_genes(output_dir='results/gene_cluster_analysis_sex_specific'):
    """Display detailed sex-specific gene effects."""
    # Load the results
    df = pd.read_csv(Path(output_dir) / 'gene_cluster_associations.csv')
    
    # Filter for sex-specific genes
    sex_specific = df[df['analysis_type'] != 'all'].sort_values(['analysis_type', 'bayes_factor'], ascending=[True, False])
    
    print("SEX-SPECIFIC GENE EFFECTS")
    print("="*80)
    print()
    
    for analysis_type in ['male', 'female']:
        type_df = sex_specific[sex_specific['analysis_type'] == analysis_type]
        if len(type_df) > 0:
            print(f"{analysis_type.upper()}-SPECIFIC GENES:")
            print("-"*40)
            for _, gene in type_df.iterrows():
                cluster_names = {0: 'Moderate HL', 1: 'Control/Normal', 2: 'High-Freq Loss', 3: 'Severe HL'}
                print(f"  {gene['gene_symbol']} ({gene['center']})")
                print(f"    - Dominant cluster: {cluster_names[gene['dominant_cluster']]}")
                print(f"    - Consistency: {gene['consistency_score']:.2f}")
                print(f"    - Bayes Factor: {gene['bayes_factor']:.1f}")
                print(f"    - N mice: {gene['n_total_mice']}")
                print(f"    - Distribution: C0:{gene['cluster_0']}, C1:{gene['cluster_1']}, C2:{gene['cluster_2']}, C3:{gene['cluster_3']}")
            print()


def show_sex_differences_only(output_dir='results/gene_cluster_analysis_sex_specific'):
    """Show genes that are significant in sex-specific but not in combined analysis."""
    # Load the results
    df = pd.read_csv(Path(output_dir) / 'gene_cluster_associations.csv')
    
    # Show genes that are significant in sex-specific but not in combined analysis
    sex_specific_only = df[
        (df['analysis_type'].isin(['male', 'female'])) & 
        (~df['sig_in_all'])
    ]
    
    print("GENES WITH SEX-SPECIFIC EFFECTS ONLY")
    print("(Significant in male or female analysis but NOT in combined analysis)")
    print("="*80)
    print()
    
    if len(sex_specific_only) > 0:
        print(f"Found {len(sex_specific_only)} genes with sex-specific effects only:")
        print()
        
        for _, gene in sex_specific_only.iterrows():
            cluster_names = {0: 'Moderate HL', 1: 'Control/Normal', 2: 'High-Freq Loss', 3: 'Severe HL'}
            print(f"{gene['gene_symbol']} ({gene['center']}) - {gene['analysis_type'].upper()}-specific")
            print(f"  - Dominant cluster: {cluster_names[gene['dominant_cluster']]}")
            print(f"  - {gene['analysis_type'].capitalize()} BF: {gene['bayes_factor']:.1f}")
            print(f"  - All BF: {gene['all_bayes_factor']:.1f}")
            print(f"  - Mice analyzed: {gene['n_total_mice']}")
            print()
    else:
        print("No genes found that are significant only in sex-specific analyses.")
    
    # Show summary of significance patterns
    print("\nSUMMARY OF SIGNIFICANCE PATTERNS:")
    print("-"*40)
    
    # Create summary of significance combinations
    sig_patterns = df.groupby(['sig_in_all', 'sig_in_male', 'sig_in_female']).size().reset_index(name='count')
    sig_patterns = sig_patterns.sort_values('count', ascending=False)
    
    for _, pattern in sig_patterns.iterrows():
        sigs = []
        if pattern['sig_in_all']:
            sigs.append('All')
        if pattern['sig_in_male']:
            sigs.append('Male')
        if pattern['sig_in_female']:
            sigs.append('Female')
        
        print(f"Significant in {', '.join(sigs)}: {pattern['count']} genes")
    
    # Show genes significant in multiple analyses
    multi_sig = df[
        (df['sig_in_all'].astype(int) + df['sig_in_male'].astype(int) + df['sig_in_female'].astype(int)) > 1
    ]
    
    if len(multi_sig) > 0:
        print(f"\n{len(multi_sig)} genes are significant in multiple analyses")
        print("These may show sex-specific differences in effect size or phenotype")


def demo_csv_structure(output_dir='results/gene_cluster_analysis_sex_specific'):
    """Demonstrate the structure and content of the output CSV."""
    # Load the results
    df = pd.read_csv(Path(output_dir) / 'gene_cluster_associations.csv')
    
    print("EXAMPLES OF SEX-SPECIFIC CLUSTER INFORMATION IN CSV")
    print("="*80)
    print()
    
    # Filter for 'all' analysis genes that have sex-specific differences
    all_genes = df[df['analysis_type'] == 'all']
    sex_diff = all_genes[
        (all_genes['male_dominant_cluster'].notna()) & 
        (all_genes['female_dominant_cluster'].notna()) &
        (all_genes['male_dominant_cluster'] != all_genes['female_dominant_cluster'])
    ]
    
    # Show first 5 examples
    columns_to_show = [
        'gene_symbol', 'bayes_factor', 
        'dominant_cluster', 'consistency_score',
        'male_n_mice', 'male_dominant_cluster', 'male_consistency_score',
        'female_n_mice', 'female_dominant_cluster', 'female_consistency_score'
    ]
    
    print("Genes with different dominant clusters by sex:")
    print(sex_diff[columns_to_show].head().to_string(index=False))
    
    print("\n" + "="*80)
    print("\nSUMMARY OF CSV STRUCTURE:")
    print(f"- Total rows: {len(df)}")
    print(f"- Analysis types: {df['analysis_type'].value_counts().to_dict()}")
    print(f"- Sex-specific columns added for 'all' analysis genes:")
    print("  - male/female_n_mice: Number of mice per sex")
    print("  - male/female_dominant_cluster: Most common cluster for each sex")
    print("  - male/female_consistency_score: Proportion in dominant cluster by sex")
    print("  - male/female_cluster_0-3: Distribution across clusters by sex")
    
    print("\nColumn headers:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")


def main(mode='full'):
    """Main analysis pipeline."""
    output_dir = Path('results/gene_cluster_analysis_sex_specific')
    
    if mode == 'sex-specific':
        show_sex_specific_genes(output_dir)
        return
    elif mode == 'sex-differences':
        show_sex_differences_only(output_dir)
        return
    elif mode == 'demo-csv':
        demo_csv_structure(output_dir)
        return
    elif mode != 'full':
        print(f"Unknown mode: {mode}")
        sys.exit(1)
    
    # Full analysis mode
    print("Loading data...")
    bayesian_df, metadata_df, cluster_labels, cluster_probs, analysis_results = load_data()
    
    print("Filtering for significant genes...")
    significant_groups = filter_significant_genes(bayesian_df)
    
    all_mappings = []
    
    # Process each analysis type separately
    for sig_genes_df, analysis_type in significant_groups:
        print(f"\nProcessing {analysis_type} analysis...")
        print(f"Found {len(sig_genes_df)} significant gene combinations for {analysis_type}")
        
        # Create mapping for this analysis type
        mapping_df = create_gene_mouse_mapping(sig_genes_df, metadata_df, cluster_labels, analysis_type)
        print(f"Successfully mapped {len(mapping_df)} gene groups to clusters for {analysis_type}")
        
        all_mappings.append(mapping_df)
    
    # Combine all mappings
    gene_cluster_df = pd.concat(all_mappings, ignore_index=True) if all_mappings else pd.DataFrame()
    
    print(f"\nTotal unique gene groups mapped: {len(gene_cluster_df)}")
    
    print("Analyzing cluster patterns...")
    gene_cluster_df, summary_stats = analyze_cluster_patterns(gene_cluster_df, analysis_results)
    
    # Add sex-specific summary statistics
    summary_stats['genes_by_analysis_type'] = gene_cluster_df['analysis_type'].value_counts().to_dict()
    summary_stats['consistency_by_analysis_type'] = gene_cluster_df.groupby('analysis_type')['consistency_score'].agg(['mean', 'std']).to_dict()
    
    # Create output directory
    output_dir = Path('results/gene_cluster_analysis_sex_specific')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("Creating visualizations...")
    create_visualizations(gene_cluster_df, output_dir)
    
    print("Generating report...")
    generate_report(gene_cluster_df, summary_stats, output_dir)
    
    # Save summary statistics as JSON
    with open(output_dir / 'summary_statistics.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/")
    print(f"Total genes analyzed: {summary_stats['total_significant_genes']}")
    print(f"Mean consistency score: {summary_stats['mean_consistency_score']:.3f}")
    print("\nGenes by analysis type:")
    for atype, count in summary_stats['genes_by_analysis_type'].items():
        print(f"  {atype}: {count} genes")
    print("\nGene distribution across clusters:")
    for cluster, count in sorted(summary_stats['genes_per_cluster'].items()):
        print(f"  Cluster {cluster}: {count} genes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze gene-cluster associations from Bayesian analysis and GMM clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Modes:
  full (default)    Run complete analysis with visualizations and report
  sex-specific      Show detailed sex-specific gene effects  
  sex-differences   Show genes with sex-specific effects only
  demo-csv          Show structure and examples from the output CSV
        ''')
    
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'sex-specific', 'sex-differences', 'demo-csv'],
                        help='Analysis mode to run')
    
    args = parser.parse_args()
    main(args.mode)