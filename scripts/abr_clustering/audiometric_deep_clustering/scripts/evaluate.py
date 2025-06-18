#!/usr/bin/env python3
"""
Evaluation script for ContrastiveVAE-DEC model.

This script provides comprehensive evaluation including clustering metrics,
biological validation, gene enrichment analysis, and visualization generation.
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

import torch
import numpy as np
import pandas as pd

from utils.seed import ensure_reproducibility
from utils.logging import setup_logging
from utils.checkpoint import load_checkpoint
from models.full_model import create_model
from data.dataset import create_abr_dataset
from data.dataloader import create_dataloaders
from data.preprocessor import ABRPreprocessor
from evaluation.metrics import compute_comprehensive_metrics
from evaluation.visualization import create_comprehensive_visualization_report
from evaluation.phenotype_analysis import perform_comprehensive_phenotype_analysis
from evaluation.gene_enrichment import perform_comprehensive_gene_enrichment_analysis, get_default_hearing_gene_sets


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate ContrastiveVAE-DEC model for audiometric phenotype discovery',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and data
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--model-config', type=str, default='config/model_config.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--data-path', type=str,
                       help='Path to dataset (overrides config)')
    
    # Evaluation options
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test', 'all'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, etc.)')
    
    # Analysis components
    parser.add_argument('--skip-clustering-metrics', action='store_true',
                       help='Skip clustering quality metrics')
    parser.add_argument('--skip-phenotype-analysis', action='store_true',
                       help='Skip biological phenotype analysis')
    parser.add_argument('--skip-gene-enrichment', action='store_true',
                       help='Skip gene enrichment analysis')
    parser.add_argument('--skip-visualizations', action='store_true',
                       help='Skip visualization generation')
    
    # Gene analysis options
    parser.add_argument('--known-hearing-genes', type=str,
                       help='Path to file with known hearing loss genes (one per line)')
    parser.add_argument('--gene-sets', type=str,
                       help='Path to JSON file with gene sets for enrichment analysis')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='evaluation_outputs',
                       help='Directory for evaluation outputs')
    parser.add_argument('--experiment-name', type=str,
                       help='Name of experiment (for organizing outputs)')
    parser.add_argument('--save-embeddings', action='store_true',
                       help='Save latent embeddings to file')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save cluster predictions to file')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-dir', type=str,
                       help='Directory for log files')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def load_configs(args: argparse.Namespace) -> Dict[str, Any]:
    """Load configuration files."""
    with open(args.config, 'r') as f:
        training_config = yaml.safe_load(f)
    
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Apply command line overrides
    if args.data_path:
        training_config['dataset']['data_path'] = args.data_path
    
    return {
        'training': training_config,
        'model': model_config
    }


def setup_device(device_arg: str) -> torch.device:
    """Setup computation device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_arg)
    
    logging.info(f"Using device: {device}")
    return device


def setup_output_directories(args: argparse.Namespace) -> Dict[str, Path]:
    """Setup output directories."""
    base_dir = Path(args.output_dir)
    
    if args.experiment_name:
        base_dir = base_dir / args.experiment_name
    
    directories = {
        'base': base_dir,
        'metrics': base_dir / 'metrics',
        'visualizations': base_dir / 'visualizations',
        'analysis': base_dir / 'analysis',
        'embeddings': base_dir / 'embeddings',
        'logs': base_dir / 'logs'
    }
    
    # Create directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories


def load_model_and_data(args: argparse.Namespace, config: Dict[str, Any],
                       device: torch.device) -> Dict[str, Any]:
    """Load model and datasets."""
    # Create model
    model = create_model(config['model'])
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_data = load_checkpoint(
        model=model,
        filepath=args.checkpoint,
        map_location=device
    )
    
    model.eval()
    logging.info(f"Loaded model from checkpoint (epoch {checkpoint_data['epoch']})")
    
    # Setup preprocessor
    preprocessor = ABRPreprocessor(
        normalize=True,
        add_pca=True,
        n_pca_components=config['model']['data']['pca_features']
    )
    
    # Create dataset
    dataset = create_abr_dataset(
        data_path=config['training']['dataset']['data_path'],
        preprocessor=preprocessor,
        feature_columns=None,
        mode='eval'
    )
    
    # Create data loaders
    dataloaders = create_dataloaders(
        dataset=dataset,
        train_ratio=config['training']['dataset']['train_split'],
        val_ratio=config['training']['dataset']['val_split'],
        test_ratio=config['training']['dataset']['test_split'],
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        seed=config['training']['dataset']['random_seed']
    )
    
    return {
        'model': model,
        'dataloaders': dataloaders,
        'preprocessor': preprocessor,
        'checkpoint_data': checkpoint_data
    }


def extract_model_outputs(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                         device: torch.device) -> Dict[str, np.ndarray]:
    """Extract model outputs for evaluation."""
    model.eval()
    
    all_features = []
    all_embeddings = []
    all_reconstructions = []
    all_cluster_assignments = []
    all_gene_labels = []
    all_mouse_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                features = batch['features'].to(device)
                gene_labels = batch.get('gene_label', torch.full((len(features),), -1))
                mouse_ids = batch.get('mouse_id', list(range(len(features))))
            else:
                features = batch.to(device)
                gene_labels = torch.full((len(features),), -1)
                mouse_ids = list(range(len(features)))
            
            # Forward pass
            output = model(features)
            
            # Extract outputs
            embeddings = output['latent_z']
            reconstructions = output['reconstruction']
            
            # Get cluster assignments if available
            if 'q' in output:
                cluster_probs = output['q']
                cluster_assignments = torch.argmax(cluster_probs, dim=1)
            else:
                cluster_assignments = torch.zeros(len(features), dtype=torch.long)
            
            # Store outputs
            all_features.append(features.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())
            all_reconstructions.append(reconstructions.cpu().numpy())
            all_cluster_assignments.append(cluster_assignments.cpu().numpy())
            all_gene_labels.append(gene_labels.numpy())
            all_mouse_ids.extend(mouse_ids)
    
    return {
        'features': np.concatenate(all_features, axis=0),
        'embeddings': np.concatenate(all_embeddings, axis=0),
        'reconstructions': np.concatenate(all_reconstructions, axis=0),
        'cluster_assignments': np.concatenate(all_cluster_assignments, axis=0),
        'gene_labels': np.concatenate(all_gene_labels, axis=0),
        'mouse_ids': np.array(all_mouse_ids)
    }


def load_additional_data(args: argparse.Namespace) -> Dict[str, Any]:
    """Load additional data for analysis."""
    additional_data = {}
    
    # Load known hearing genes
    if args.known_hearing_genes:
        with open(args.known_hearing_genes, 'r') as f:
            known_genes = [line.strip() for line in f if line.strip()]
        additional_data['known_hearing_genes'] = known_genes
    
    # Load gene sets
    if args.gene_sets:
        with open(args.gene_sets, 'r') as f:
            gene_sets = json.load(f)
        additional_data['gene_sets'] = gene_sets
    else:
        # Use default hearing-related gene sets
        additional_data['gene_sets'] = get_default_hearing_gene_sets()
    
    return additional_data


def run_clustering_evaluation(outputs: Dict[str, np.ndarray],
                            output_dirs: Dict[str, Path]) -> Dict[str, Any]:
    """Run clustering quality evaluation."""
    logging.info("Computing clustering metrics")
    
    # Extract relevant data
    embeddings = outputs['embeddings']
    cluster_labels = outputs['cluster_assignments']
    features = outputs['features']
    reconstructions = outputs['reconstructions']
    gene_labels = outputs.get('gene_labels')
    abr_features = features[:, :6]  # First 6 features are ABR thresholds
    
    # Compute comprehensive metrics
    metrics = compute_comprehensive_metrics(
        embeddings=embeddings,
        cluster_labels=cluster_labels,
        original_features=features,
        reconstructed_features=reconstructions,
        gene_labels=gene_labels,
        abr_features=abr_features
    )
    
    # Save metrics
    metrics_file = output_dirs['metrics'] / 'clustering_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    logging.info(f"Clustering metrics saved to {metrics_file}")
    return metrics


def run_phenotype_analysis(outputs: Dict[str, np.ndarray],
                         additional_data: Dict[str, Any],
                         output_dirs: Dict[str, Path]) -> Dict[str, Any]:
    """Run biological phenotype analysis."""
    logging.info("Performing phenotype analysis")
    
    # Extract relevant data
    cluster_labels = outputs['cluster_assignments']
    abr_features = outputs['features'][:, :6]  # ABR thresholds
    gene_labels = outputs.get('gene_labels')
    known_genes = additional_data.get('known_hearing_genes')
    
    # Run comprehensive phenotype analysis
    phenotype_results = perform_comprehensive_phenotype_analysis(
        cluster_labels=cluster_labels,
        abr_features=abr_features,
        gene_labels=gene_labels,
        known_hearing_genes=known_genes,
        save_dir=output_dirs['analysis']
    )
    
    logging.info("Phenotype analysis completed")
    return phenotype_results


def run_gene_enrichment_analysis(outputs: Dict[str, np.ndarray],
                                additional_data: Dict[str, Any],
                                output_dirs: Dict[str, Path]) -> Dict[str, Any]:
    """Run gene set enrichment analysis."""
    logging.info("Performing gene enrichment analysis")
    
    # Extract relevant data
    cluster_labels = outputs['cluster_assignments']
    gene_labels = outputs.get('gene_labels')
    gene_sets = additional_data.get('gene_sets', {})
    
    if gene_labels is None or len(gene_sets) == 0:
        logging.warning("Skipping gene enrichment analysis - insufficient data")
        return {}
    
    # Run enrichment analysis
    enrichment_results = perform_comprehensive_gene_enrichment_analysis(
        cluster_labels=cluster_labels,
        gene_labels=gene_labels,
        gene_sets=gene_sets,
        save_dir=output_dirs['analysis']
    )
    
    logging.info("Gene enrichment analysis completed")
    return enrichment_results


def run_visualization_generation(outputs: Dict[str, np.ndarray],
                               output_dirs: Dict[str, Path]) -> Dict[str, Any]:
    """Generate comprehensive visualizations."""
    logging.info("Generating visualizations")
    
    # Extract relevant data
    embeddings = outputs['embeddings']
    cluster_labels = outputs['cluster_assignments']
    abr_features = outputs['features'][:, :6]
    gene_labels = outputs.get('gene_labels')
    mouse_ids = outputs.get('mouse_ids')
    
    # Generate visualization report
    visualization_report = create_comprehensive_visualization_report(
        embeddings=embeddings,
        cluster_labels=cluster_labels,
        abr_features=abr_features,
        gene_labels=gene_labels,
        mouse_ids=mouse_ids.tolist() if mouse_ids is not None else None,
        save_dir=output_dirs['visualizations']
    )
    
    logging.info(f"Visualizations saved to {output_dirs['visualizations']}")
    return visualization_report


def save_outputs(outputs: Dict[str, np.ndarray], args: argparse.Namespace,
                output_dirs: Dict[str, Path]) -> None:
    """Save model outputs to files."""
    if args.save_embeddings:
        embeddings_file = output_dirs['embeddings'] / 'latent_embeddings.npz'
        np.savez_compressed(
            embeddings_file,
            embeddings=outputs['embeddings'],
            cluster_assignments=outputs['cluster_assignments'],
            mouse_ids=outputs['mouse_ids'],
            gene_labels=outputs['gene_labels']
        )
        logging.info(f"Embeddings saved to {embeddings_file}")
    
    if args.save_predictions:
        predictions_file = output_dirs['embeddings'] / 'cluster_predictions.csv'
        predictions_df = pd.DataFrame({
            'mouse_id': outputs['mouse_ids'],
            'cluster_assignment': outputs['cluster_assignments'],
            'gene_label': outputs['gene_labels']
        })
        predictions_df.to_csv(predictions_file, index=False)
        logging.info(f"Predictions saved to {predictions_file}")


def generate_evaluation_report(clustering_metrics: Dict[str, Any],
                             phenotype_results: Dict[str, Any],
                             enrichment_results: Dict[str, Any],
                             output_dirs: Dict[str, Path]) -> None:
    """Generate comprehensive evaluation report."""
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("CONTRASTIVEVAE-DEC EVALUATION REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Clustering metrics summary
    if clustering_metrics:
        report_lines.append("CLUSTERING QUALITY METRICS")
        report_lines.append("-" * 40)
        
        if 'clustering' in clustering_metrics:
            clustering = clustering_metrics['clustering']
            report_lines.append(f"Silhouette Score: {clustering.get('silhouette_score', 'N/A'):.3f}")
            report_lines.append(f"Calinski-Harabasz Index: {clustering.get('calinski_harabasz_score', 'N/A'):.2f}")
            report_lines.append(f"Davies-Bouldin Index: {clustering.get('davies_bouldin_score', 'N/A'):.3f}")
        
        if 'reconstruction' in clustering_metrics:
            recon = clustering_metrics['reconstruction']
            report_lines.append(f"Reconstruction MSE: {recon.get('reconstruction_mse', 'N/A'):.4f}")
            report_lines.append(f"Feature Correlation: {recon.get('mean_feature_correlation', 'N/A'):.3f}")
        
        report_lines.append("")
    
    # Phenotype analysis summary
    if phenotype_results and 'summary' in phenotype_results:
        summary = phenotype_results['summary']
        report_lines.append("PHENOTYPE ANALYSIS")
        report_lines.append("-" * 40)
        report_lines.append(f"Number of clusters: {summary.get('num_clusters', 'N/A')}")
        report_lines.append(f"Pattern diversity: {summary.get('pattern_diversity', 'N/A')}")
        report_lines.append(f"Most common pattern: {summary.get('most_common_pattern', 'N/A')}")
        report_lines.append(f"Most common severity: {summary.get('most_common_severity', 'N/A')}")
        
        if 'known_gene_validation_rate' in summary:
            report_lines.append(f"Known gene validation rate: {summary['known_gene_validation_rate']:.2%}")
        
        report_lines.append("")
    
    # Gene enrichment summary
    if enrichment_results and 'summary' in enrichment_results:
        summary = enrichment_results['summary']
        report_lines.append("GENE ENRICHMENT ANALYSIS")
        report_lines.append("-" * 40)
        report_lines.append(f"Overall gene-cluster association: {'Yes' if summary.get('overall_gene_cluster_association', False) else 'No'}")
        report_lines.append(f"Proportion of clustered genes: {summary.get('proportion_clustered_genes', 0):.2%}")
        report_lines.append(f"Significantly enriched genes: {summary.get('n_significantly_enriched_genes', 0)}")
        report_lines.append(f"Clusters with enrichments: {summary.get('n_clusters_with_enrichments', 0)}")
        report_lines.append("")
    
    report_lines.append("="*80)
    
    # Save report
    report_file = output_dirs['base'] / 'evaluation_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logging.info(f"Evaluation report saved to {report_file}")


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup output directories
    output_dirs = setup_output_directories(args)
    
    # Setup logging
    log_dir = Path(args.log_dir) if args.log_dir else output_dirs['logs']
    logger = setup_logging(
        log_level=args.log_level,
        log_dir=log_dir,
        experiment_name=args.experiment_name,
        console_output=True,
        file_output=True
    )
    
    # Setup reproducibility
    ensure_reproducibility(args.seed)
    
    # Load configurations
    config = load_configs(args)
    
    # Setup device
    device = setup_device(args.device)
    
    # Load model and data
    model_data = load_model_and_data(args, config, device)
    model = model_data['model']
    dataloaders = model_data['dataloaders']
    
    # Select dataloader for evaluation
    if args.split == 'all':
        eval_dataloaders = dataloaders
    else:
        eval_dataloaders = {args.split: dataloaders[args.split]}
    
    # Load additional data
    additional_data = load_additional_data(args)
    
    # Initialize results storage
    all_results = {}
    
    # Process each split
    for split_name, dataloader in eval_dataloaders.items():
        logging.info(f"Evaluating on {split_name} split ({len(dataloader.dataset)} samples)")
        
        # Extract model outputs
        outputs = extract_model_outputs(model, dataloader, device)
        
        split_results = {}
        
        # Save outputs if requested
        save_outputs(outputs, args, output_dirs)
        
        # Run clustering evaluation
        if not args.skip_clustering_metrics:
            clustering_metrics = run_clustering_evaluation(outputs, output_dirs)
            split_results['clustering_metrics'] = clustering_metrics
        
        # Run phenotype analysis
        if not args.skip_phenotype_analysis:
            phenotype_results = run_phenotype_analysis(outputs, additional_data, output_dirs)
            split_results['phenotype_analysis'] = phenotype_results
        
        # Run gene enrichment analysis
        if not args.skip_gene_enrichment:
            enrichment_results = run_gene_enrichment_analysis(outputs, additional_data, output_dirs)
            split_results['gene_enrichment'] = enrichment_results
        
        # Generate visualizations
        if not args.skip_visualizations:
            visualization_report = run_visualization_generation(outputs, output_dirs)
            split_results['visualizations'] = 'generated'
        
        all_results[split_name] = split_results
    
    # Generate comprehensive evaluation report
    if len(all_results) == 1:
        # Single split evaluation
        split_results = list(all_results.values())[0]
        generate_evaluation_report(
            clustering_metrics=split_results.get('clustering_metrics', {}),
            phenotype_results=split_results.get('phenotype_analysis', {}),
            enrichment_results=split_results.get('gene_enrichment', {}),
            output_dirs=output_dirs
        )
    
    # Save complete results
    results_file = output_dirs['base'] / 'complete_evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logging.info(f"Evaluation completed! Results saved to {output_dirs['base']}")


if __name__ == '__main__':
    main()