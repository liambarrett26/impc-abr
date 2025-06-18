#!/usr/bin/env python3
"""
Visualization generation script for ContrastiveVAE-DEC results.

This script generates comprehensive visualizations including dimensionality reduction
plots, cluster analysis, audiogram patterns, and interactive dashboards.
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.logging import setup_logging
from evaluation.visualization import (
    DimensionalityReductionVisualizer,
    ClusterAnalysisVisualizer,
    InteractiveVisualizer,
    create_comprehensive_visualization_report
)
from evaluation.phenotype_analysis import HearingLossClassifier


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive visualizations for ContrastiveVAE-DEC results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input data
    parser.add_argument('--embeddings', type=str, required=True,
                       help='Path to latent embeddings file (.npz)')
    parser.add_argument('--results', type=str,
                       help='Path to inference results file (.csv)')
    parser.add_argument('--original-data', type=str,
                       help='Path to original ABR data file (.csv)')
    
    # Visualization options
    parser.add_argument('--reduction-methods', type=str, nargs='+',
                       default=['pca', 'tsne', 'umap'],
                       choices=['pca', 'tsne', 'umap'],
                       help='Dimensionality reduction methods to use')
    parser.add_argument('--skip-static-plots', action='store_true',
                       help='Skip static matplotlib plots')
    parser.add_argument('--skip-interactive-plots', action='store_true',
                       help='Skip interactive plotly plots')
    parser.add_argument('--skip-cluster-analysis', action='store_true',
                       help='Skip cluster analysis plots')
    parser.add_argument('--skip-audiogram-plots', action='store_true',
                       help='Skip audiogram pattern plots')
    
    # Plot customization
    parser.add_argument('--figure-format', type=str, default='png',
                       choices=['png', 'pdf', 'svg', 'jpg'],
                       help='Format for static figures')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for static figures')
    parser.add_argument('--figure-size', type=float, nargs=2, default=[12, 8],
                       help='Figure size (width, height) in inches')
    parser.add_argument('--color-palette', type=str, default='husl',
                       help='Seaborn color palette to use')
    
    # Data processing
    parser.add_argument('--max-samples', type=int,
                       help='Maximum number of samples to plot (for performance)')
    parser.add_argument('--cluster-column', type=str, default='cluster_assignment',
                       help='Column name for cluster assignments')
    parser.add_argument('--gene-column', type=str, default='gene_label',
                       help='Column name for gene labels')
    parser.add_argument('--mouse-id-column', type=str, default='mouse_id',
                       help='Column name for mouse IDs')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='visualization_outputs',
                       help='Directory for visualization outputs')
    parser.add_argument('--experiment-name', type=str,
                       help='Name of experiment (for organizing outputs)')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress console output')
    
    return parser.parse_args()


def setup_output_directories(args: argparse.Namespace) -> Dict[str, Path]:
    """Setup output directories for different types of visualizations."""
    base_dir = Path(args.output_dir)
    
    if args.experiment_name:
        base_dir = base_dir / args.experiment_name
    
    directories = {
        'base': base_dir,
        'static': base_dir / 'static_plots',
        'interactive': base_dir / 'interactive_plots',
        'cluster_analysis': base_dir / 'cluster_analysis',
        'audiograms': base_dir / 'audiogram_patterns',
        'reports': base_dir / 'reports'
    }
    
    # Create directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories


def load_data(args: argparse.Namespace) -> Dict[str, Any]:
    """Load all required data for visualization."""
    data = {}
    
    # Load embeddings (required)
    embeddings_data = np.load(args.embeddings)
    data['embeddings'] = embeddings_data['embeddings']
    
    # Try to load additional data from embeddings file
    if 'cluster_assignments' in embeddings_data:
        data['cluster_assignments'] = embeddings_data['cluster_assignments']
    if 'mouse_ids' in embeddings_data:
        data['mouse_ids'] = embeddings_data['mouse_ids']
    if 'gene_labels' in embeddings_data:
        data['gene_labels'] = embeddings_data['gene_labels']
    
    logging.info(f"Loaded embeddings: {data['embeddings'].shape}")
    
    # Load results file if provided
    if args.results:
        results_df = pd.read_csv(args.results)
        
        # Override with results file data if available
        if args.cluster_column in results_df.columns:
            data['cluster_assignments'] = results_df[args.cluster_column].values
        if args.gene_column in results_df.columns:
            data['gene_labels'] = results_df[args.gene_column].values
        if args.mouse_id_column in results_df.columns:
            data['mouse_ids'] = results_df[args.mouse_id_column].values
        
        logging.info(f"Loaded results: {len(results_df)} samples")
    
    # Load original ABR data if provided
    if args.original_data:
        original_df = pd.read_csv(args.original_data)
        
        # Try to extract ABR features (first 6 numeric columns by default)
        numeric_cols = original_df.select_dtypes(include=[np.number]).columns
        abr_columns = numeric_cols[:6].tolist()
        
        if len(abr_columns) >= 6:
            data['abr_features'] = original_df[abr_columns].values
            data['abr_column_names'] = abr_columns
            logging.info(f"Loaded ABR features: {data['abr_features'].shape}")
        else:
            logging.warning("Could not find sufficient ABR features in original data")
    
    # Validate data consistency
    n_samples = len(data['embeddings'])
    for key, values in data.items():
        if key != 'abr_column_names' and hasattr(values, '__len__'):
            if len(values) != n_samples:
                logging.warning(f"Data size mismatch for {key}: {len(values)} vs {n_samples}")
    
    return data


def subsample_data(data: Dict[str, Any], max_samples: Optional[int]) -> Dict[str, Any]:
    """Subsample data for performance if needed."""
    n_samples = len(data['embeddings'])
    
    if max_samples is None or n_samples <= max_samples:
        return data
    
    # Random subsample
    indices = np.random.choice(n_samples, size=max_samples, replace=False)
    indices = np.sort(indices)  # Keep order
    
    subsampled_data = {}
    for key, values in data.items():
        if key == 'abr_column_names':
            subsampled_data[key] = values
        elif hasattr(values, '__len__') and len(values) == n_samples:
            subsampled_data[key] = values[indices]
        else:
            subsampled_data[key] = values
    
    logging.info(f"Subsampled data from {n_samples} to {max_samples} samples")
    return subsampled_data


def setup_plotting_style(args: argparse.Namespace):
    """Setup matplotlib and seaborn plotting style."""
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette(args.color_palette)
    
    # Set default figure size
    plt.rcParams['figure.figsize'] = args.figure_size
    plt.rcParams['figure.dpi'] = args.dpi
    plt.rcParams['savefig.dpi'] = args.dpi
    plt.rcParams['savefig.format'] = args.figure_format
    
    # Font settings
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def generate_dimensionality_reduction_plots(data: Dict[str, Any], args: argparse.Namespace,
                                           output_dirs: Dict[str, Path]) -> Dict[str, Any]:
    """Generate dimensionality reduction visualizations."""
    logging.info("Generating dimensionality reduction plots")
    
    visualizer = DimensionalityReductionVisualizer()
    
    # Fit reduction methods
    reductions = visualizer.fit_reductions(
        data['embeddings'], 
        methods=args.reduction_methods
    )
    
    # Create plots
    fig = visualizer.plot_cluster_embeddings(
        reductions=reductions,
        cluster_labels=data.get('cluster_assignments'),
        gene_labels=data.get('gene_labels'),
        save_path=output_dirs['static'] / f'cluster_embeddings.{args.figure_format}'
    )
    
    plt.close(fig)  # Free memory
    
    return reductions


def generate_cluster_analysis_plots(data: Dict[str, Any], reductions: Dict[str, np.ndarray],
                                   args: argparse.Namespace, output_dirs: Dict[str, Path]):
    """Generate cluster analysis visualizations."""
    logging.info("Generating cluster analysis plots")
    
    if 'cluster_assignments' not in data:
        logging.warning("No cluster assignments available for cluster analysis")
        return
    
    analyzer = ClusterAnalysisVisualizer()
    
    # Use first reduction method for cluster statistics
    embedding_2d = list(reductions.values())[0]
    
    # Cluster statistics plot
    fig = analyzer.plot_cluster_statistics(
        cluster_labels=data['cluster_assignments'],
        embeddings=embedding_2d,
        save_path=output_dirs['cluster_analysis'] / f'cluster_statistics.{args.figure_format}'
    )
    plt.close(fig)


def generate_audiogram_plots(data: Dict[str, Any], args: argparse.Namespace,
                           output_dirs: Dict[str, Path]):
    """Generate audiogram pattern visualizations."""
    logging.info("Generating audiogram plots")
    
    if 'abr_features' not in data or 'cluster_assignments' not in data:
        logging.warning("No ABR features or cluster assignments available for audiogram plots")
        return
    
    analyzer = ClusterAnalysisVisualizer()
    
    # Determine frequency labels
    frequency_labels = None
    if 'abr_column_names' in data:
        frequency_labels = data['abr_column_names']
    
    # Audiogram patterns plot
    fig = analyzer.plot_audiogram_patterns(
        cluster_labels=data['cluster_assignments'],
        abr_features=data['abr_features'],
        frequency_labels=frequency_labels,
        save_path=output_dirs['audiograms'] / f'audiogram_patterns.{args.figure_format}'
    )
    plt.close(fig)


def generate_interactive_visualizations(data: Dict[str, Any], reductions: Dict[str, np.ndarray],
                                       args: argparse.Namespace, output_dirs: Dict[str, Path]):
    """Generate interactive plotly visualizations."""
    logging.info("Generating interactive visualizations")
    
    visualizer = InteractiveVisualizer()
    
    # Use t-SNE if available, otherwise first reduction method
    embedding_2d = reductions.get('tsne', list(reductions.values())[0])
    
    # Interactive cluster plot
    fig = visualizer.create_interactive_cluster_plot(
        embeddings_2d=embedding_2d,
        cluster_labels=data.get('cluster_assignments'),
        gene_labels=data.get('gene_labels'),
        abr_features=data.get('abr_features'),
        mouse_ids=data.get('mouse_ids')
    )
    
    # Save interactive plot
    interactive_file = output_dirs['interactive'] / 'interactive_clusters.html'
    fig.write_html(interactive_file)
    logging.info(f"Interactive cluster plot saved to {interactive_file}")
    
    # Cluster comparison dashboard
    if 'cluster_assignments' in data and 'abr_features' in data:
        dashboard_fig = visualizer.create_cluster_comparison_dashboard(
            cluster_labels=data['cluster_assignments'],
            abr_features=data['abr_features'],
            gene_labels=data.get('gene_labels')
        )
        
        dashboard_file = output_dirs['interactive'] / 'cluster_dashboard.html'
        dashboard_fig.write_html(dashboard_file)
        logging.info(f"Cluster dashboard saved to {dashboard_file}")


def generate_phenotype_summary_plots(data: Dict[str, Any], args: argparse.Namespace,
                                    output_dirs: Dict[str, Path]):
    """Generate phenotype classification summary plots."""
    if 'abr_features' not in data or 'cluster_assignments' not in data:
        logging.warning("Insufficient data for phenotype summary plots")
        return
    
    logging.info("Generating phenotype summary plots")
    
    # Classify hearing loss patterns
    classifier = HearingLossClassifier()
    cluster_patterns = classifier.classify_cluster_patterns(
        data['cluster_assignments'], 
        data['abr_features']
    )
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Pattern type distribution
    pattern_types = [info['dominant_pattern'] for info in cluster_patterns.values()]
    pattern_counts = pd.Series(pattern_types).value_counts()
    
    axes[0, 0].bar(pattern_counts.index, pattern_counts.values)
    axes[0, 0].set_title('Hearing Loss Pattern Distribution')
    axes[0, 0].set_xlabel('Pattern Type')
    axes[0, 0].set_ylabel('Number of Clusters')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Severity distribution
    severities = [info['dominant_severity'] for info in cluster_patterns.values()]
    severity_counts = pd.Series(severities).value_counts()
    
    axes[0, 1].bar(severity_counts.index, severity_counts.values)
    axes[0, 1].set_title('Hearing Loss Severity Distribution')
    axes[0, 1].set_xlabel('Severity Level')
    axes[0, 1].set_ylabel('Number of Clusters')
    
    # Pattern homogeneity
    cluster_ids = list(cluster_patterns.keys())
    homogeneities = [cluster_patterns[cid]['pattern_homogeneity'] for cid in cluster_ids]
    
    axes[1, 0].bar(cluster_ids, homogeneities)
    axes[1, 0].set_title('Cluster Pattern Homogeneity')
    axes[1, 0].set_xlabel('Cluster ID')
    axes[1, 0].set_ylabel('Homogeneity Score')
    
    # Cluster sizes
    cluster_sizes = [cluster_patterns[cid]['size'] for cid in cluster_ids]
    
    axes[1, 1].bar(cluster_ids, cluster_sizes)
    axes[1, 1].set_title('Cluster Sizes')
    axes[1, 1].set_xlabel('Cluster ID')
    axes[1, 1].set_ylabel('Number of Samples')
    
    plt.tight_layout()
    
    # Save plot
    phenotype_file = output_dirs['cluster_analysis'] / f'phenotype_summary.{args.figure_format}'
    plt.savefig(phenotype_file, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    
    logging.info(f"Phenotype summary saved to {phenotype_file}")


def generate_comprehensive_report(data: Dict[str, Any], reductions: Dict[str, np.ndarray],
                                output_dirs: Dict[str, Path]) -> str:
    """Generate comprehensive visualization report."""
    # Use existing comprehensive report function
    report = create_comprehensive_visualization_report(
        embeddings=data['embeddings'],
        cluster_labels=data.get('cluster_assignments'),
        abr_features=data.get('abr_features'),
        gene_labels=data.get('gene_labels'),
        mouse_ids=data.get('mouse_ids'),
        save_dir=output_dirs['reports']
    )
    
    return "Comprehensive visualization report generated"


def create_visualization_index(output_dirs: Dict[str, Path], args: argparse.Namespace) -> None:
    """Create an HTML index of all generated visualizations."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ContrastiveVAE-DEC Visualization Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #333; }}
            .section {{ margin: 20px 0; }}
            .file-list {{ margin-left: 20px; }}
            .file-list li {{ margin: 5px 0; }}
            a {{ color: #0066cc; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <h1>ContrastiveVAE-DEC Visualization Results</h1>
        
        <div class="section">
            <h2>Static Plots</h2>
            <ul class="file-list">
    """
    
    # Add static plots
    for plot_file in output_dirs['static'].glob(f'*.{args.figure_format}'):
        html_content += f'<li><a href="static_plots/{plot_file.name}">{plot_file.stem}</a></li>\n'
    
    html_content += """
            </ul>
        </div>
        
        <div class="section">
            <h2>Interactive Visualizations</h2>
            <ul class="file-list">
    """
    
    # Add interactive plots
    for html_file in output_dirs['interactive'].glob('*.html'):
        html_content += f'<li><a href="interactive_plots/{html_file.name}">{html_file.stem}</a></li>\n'
    
    html_content += """
            </ul>
        </div>
        
        <div class="section">
            <h2>Cluster Analysis</h2>
            <ul class="file-list">
    """
    
    # Add cluster analysis plots
    for plot_file in output_dirs['cluster_analysis'].glob(f'*.{args.figure_format}'):
        html_content += f'<li><a href="cluster_analysis/{plot_file.name}">{plot_file.stem}</a></li>\n'
    
    html_content += """
            </ul>
        </div>
        
        <div class="section">
            <h2>Audiogram Patterns</h2>
            <ul class="file-list">
    """
    
    # Add audiogram plots
    for plot_file in output_dirs['audiograms'].glob(f'*.{args.figure_format}'):
        html_content += f'<li><a href="audiogram_patterns/{plot_file.name}">{plot_file.stem}</a></li>\n'
    
    html_content += """
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Save index file
    index_file = output_dirs['base'] / 'index.html'
    with open(index_file, 'w') as f:
        f.write(html_content)
    
    logging.info(f"Visualization index created: {index_file}")


def main():
    """Main visualization function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup output directories
    output_dirs = setup_output_directories(args)
    
    # Setup logging
    logger = setup_logging(
        log_level=args.log_level,
        log_dir=output_dirs['base'],
        console_output=not args.quiet,
        file_output=True
    )
    
    # Setup plotting style
    setup_plotting_style(args)
    
    # Load data
    data = load_data(args)
    
    # Subsample if needed
    if args.max_samples:
        data = subsample_data(data, args.max_samples)
    
    # Generate dimensionality reduction plots
    reductions = generate_dimensionality_reduction_plots(data, args, output_dirs)
    
    # Generate cluster analysis plots
    if not args.skip_cluster_analysis:
        generate_cluster_analysis_plots(data, reductions, args, output_dirs)
    
    # Generate audiogram plots
    if not args.skip_audiogram_plots:
        generate_audiogram_plots(data, args, output_dirs)
        generate_phenotype_summary_plots(data, args, output_dirs)
    
    # Generate interactive visualizations
    if not args.skip_interactive_plots:
        generate_interactive_visualizations(data, reductions, args, output_dirs)
    
    # Generate comprehensive report
    logging.info("Generating comprehensive visualization report")
    report_result = generate_comprehensive_report(data, reductions, output_dirs)
    
    # Create visualization index
    create_visualization_index(output_dirs, args)
    
    logging.info(f"All visualizations completed! Results saved to {output_dirs['base']}")
    print(f"Open {output_dirs['base']}/index.html to view all generated visualizations")


if __name__ == '__main__':
    main()