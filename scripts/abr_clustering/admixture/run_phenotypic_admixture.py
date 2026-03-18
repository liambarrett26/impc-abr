#!/usr/bin/env python3
"""
Complete Phenotypic Neural ADMIXTURE Runner

This script provides a comprehensive command-line interface for training, testing,
and evaluating Phenotypic Neural ADMIXTURE models on IMPC ABR data.
"""

import argparse
import logging
import sys
import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add all module paths relative to the script location
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir / 'existing_gmm'))

# Now use absolute imports in phenotypic_adapter.py:
from neural_admixture_original.model.neural_admixture import NeuralAdmixture
from neural_admixture_original.model.train import train
from existing_gmm.preproc import ABRPreprocessor

sys.path.insert(0, str(script_dir / 'phenotypic_adaptation'))

# Import phenotypic adaptation modules
from phenotypic_adapter import (
    PhenotypicNeuralAdmixture,
    visualize_phenotypic_clusters,
    extract_gene_cluster_associations,
    _calculate_assignment_entropy
)

# Setup logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup comprehensive logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )

    # Reduce noise from other loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Phenotypic Neural ADMIXTURE for Audiometric Phenotype Discovery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to IMPC ABR dataset (CSV/parquet format)"
    )

    # Model parameters
    model_group = parser.add_argument_group("Model Parameters")
    model_group.add_argument(
        "--min-k", type=int, default=3,
        help="Minimum number of clusters"
    )
    model_group.add_argument(
        "--max-k", type=int, default=12,
        help="Maximum number of clusters"
    )
    model_group.add_argument(
        "--hidden-size", type=int, default=32,
        help="Hidden layer size (reduced for phenotypic data)"
    )

    # Training parameters
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument(
        "--epochs", type=int, default=500,
        help="Number of training epochs"
    )
    train_group.add_argument(
        "--batch-size", type=int, default=512,
        help="Batch size for training"
    )
    train_group.add_argument(
        "--learning-rate", type=float, default=1e-3,
        help="Learning rate"
    )
    train_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )

    # Data parameters
    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument(
        "--min-mutants", type=int, default=3,
        help="Minimum mutant mice per experimental group"
    )
    data_group.add_argument(
        "--min-controls", type=int, default=20,
        help="Minimum control mice per experimental group"
    )
    data_group.add_argument(
        "--validation-split", type=float, default=0.2,
        help="Fraction of data for validation"
    )

    # Output parameters
    output_group = parser.add_argument_group("Output Parameters")
    output_group.add_argument(
        "--output-dir", type=str, default="phenotypic_admixture_results",
        help="Output directory for results"
    )
    output_group.add_argument(
        "--name", type=str, default=None,
        help="Name prefix for output files (default: auto-generated)"
    )
    output_group.add_argument(
        "--save-preprocessing", action="store_true",
        help="Save preprocessing state for reuse"
    )

    # Analysis parameters
    analysis_group = parser.add_argument_group("Analysis Parameters")
    analysis_group.add_argument(
        "--confidence-threshold", type=float, default=0.7,
        help="Minimum confidence for gene-cluster associations"
    )
    analysis_group.add_argument(
        "--create-visualizations", action="store_true",
        help="Create and save visualization plots"
    )
    analysis_group.add_argument(
        "--analyze-all-k", action="store_true",
        help="Generate analysis for all K values (default: only optimal K)"
    )

    # Technical parameters
    tech_group = parser.add_argument_group("Technical Parameters")
    tech_group.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Computing device"
    )
    tech_group.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    tech_group.add_argument(
        "--no-save-model", action="store_true",
        help="Don't save trained model state"
    )

    return parser


def setup_device(device_arg: str) -> torch.device:
    """Setup computing device based on argument and availability."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
    else:
        device = torch.device(device_arg)
        logger.info(f"Using specified device: {device}")

    return device


def validate_data_path(data_path: str) -> Path:
    """Validate and return data path."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if not path.suffix.lower() in ['.csv', '.parquet', '.pkl', '.xlsx']:
        logger.warning(f"Unexpected file format: {path.suffix}")

    return path


def generate_run_name(args: argparse.Namespace) -> str:
    """Generate a descriptive name for this run."""
    if args.name:
        return args.name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"phenotypic_admixture_k{args.min_k}-{args.max_k}_{timestamp}"


def train_model(args: argparse.Namespace, device: torch.device) -> PhenotypicNeuralAdmixture:
    """Train the Phenotypic Neural ADMIXTURE model."""
    logger.info("="*60)
    logger.info("TRAINING PHENOTYPIC NEURAL ADMIXTURE")
    logger.info("="*60)

    start_time = time.time()

    # Create model
    model = PhenotypicNeuralAdmixture(
        min_k=args.min_k,
        max_k=args.max_k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        device=device,
        seed=args.seed
    )

    # Train model
    model.fit(
        data_path=str(args.data_path),
        save_preprocessing=args.save_preprocessing,
        save_dir=args.output_dir
    )

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    return model


def analyze_results(model: PhenotypicNeuralAdmixture, args: argparse.Namespace) -> Dict[str, Any]:
    """Perform comprehensive analysis of results."""
    logger.info("="*60)
    logger.info("ANALYZING RESULTS")
    logger.info("="*60)

    analysis_results = {}
    k_values = list(range(args.min_k, args.max_k + 1))

    # Calculate metrics for each K
    for k in k_values:
        assignments = model.get_cluster_assignments(k)
        centers = model.get_cluster_centers(k)

        # Calculate assignment statistics
        max_probs = np.max(assignments, axis=1)
        entropy = _calculate_assignment_entropy(assignments)

        # Calculate cluster statistics
        hard_assignments = np.argmax(assignments, axis=1)
        cluster_sizes = np.bincount(hard_assignments, minlength=k)

        # Calculate feature discrimination (variance across clusters)
        feature_variance = np.var(centers, axis=0)

        k_results = {
            'k': k,
            'n_samples': assignments.shape[0],
            'n_features': centers.shape[1],
            'assignment_stats': {
                'mean_confidence': float(np.mean(max_probs)),
                'std_confidence': float(np.std(max_probs)),
                'min_confidence': float(np.min(max_probs)),
                'max_confidence': float(np.max(max_probs)),
                'mean_entropy': float(entropy),
                'high_confidence_fraction': float(np.mean(max_probs >= args.confidence_threshold))
            },
            'cluster_stats': {
                'cluster_sizes': cluster_sizes.tolist(),
                'size_balance': float(np.std(cluster_sizes) / np.mean(cluster_sizes)),
                'min_cluster_size': int(np.min(cluster_sizes)),
                'max_cluster_size': int(np.max(cluster_sizes))
            },
            'feature_stats': {
                'total_discrimination': float(np.sum(feature_variance)),
                'mean_discrimination': float(np.mean(feature_variance)),
                'max_discrimination': float(np.max(feature_variance)),
                'discrimination_per_feature': feature_variance.tolist()
            }
        }

        analysis_results[k] = k_results

        # Log summary for this K
        logger.info(f"K={k}: Mean confidence={np.mean(max_probs):.3f}, "
                   f"Entropy={entropy:.3f}, "
                   f"Balance={k_results['cluster_stats']['size_balance']:.3f}")

    # Find optimal K (balance between confidence and cluster balance)
    optimal_k = _find_optimal_k(analysis_results)
    analysis_results['optimal_k'] = optimal_k

    logger.info(f"Optimal K selected: {optimal_k}")

    return analysis_results


def _find_optimal_k(results: Dict[str, Any]) -> int:
    """Find optimal K value based on multiple criteria."""
    k_values = [k for k in results.keys() if isinstance(k, int)]

    scores = []
    for k in k_values:
        r = results[k]

        # Composite score: balance confidence, entropy, and cluster balance
        confidence_score = r['assignment_stats']['mean_confidence']
        entropy_score = 1.0 - (r['assignment_stats']['mean_entropy'] / np.log(k))  # Normalized entropy
        balance_score = 1.0 / (1.0 + r['cluster_stats']['size_balance'])  # Lower imbalance is better
        discrimination_score = r['feature_stats']['mean_discrimination']

        # Weighted combination
        composite_score = (
            0.3 * confidence_score +
            0.2 * entropy_score +
            0.3 * balance_score +
            0.2 * discrimination_score
        )

        scores.append(composite_score)

    optimal_idx = np.argmax(scores)
    return k_values[optimal_idx]


def extract_gene_associations(model: PhenotypicNeuralAdmixture,
                            data_path: str,
                            args: argparse.Namespace) -> Dict[int, pd.DataFrame]:
    """Extract gene-cluster associations for analysis."""
    logger.info("Extracting gene-cluster associations...")

    # Load original data for gene information
    try:
        if str(data_path).endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
    except Exception as e:
        logger.warning(f"Could not load original data for gene associations: {e}")
        return {}

    if 'gene_symbol' not in df.columns:
        logger.warning("No gene_symbol column found - skipping gene association analysis")
        return {}

    gene_associations = {}
    k_values = list(range(args.min_k, args.max_k + 1))

    for k in k_values:
        try:
            associations_df = extract_gene_cluster_associations(
                model, df, k, args.confidence_threshold
            )
            gene_associations[k] = associations_df

            if not associations_df.empty:
                logger.info(f"K={k}: {len(associations_df)} high-confidence gene associations")
        except Exception as e:
            logger.warning(f"Failed to extract associations for K={k}: {e}")

    return gene_associations


def create_visualizations(model: PhenotypicNeuralAdmixture,
                        analysis_results: Dict[str, Any],
                        args: argparse.Namespace):
    """Create and save visualization plots."""
    if not args.create_visualizations:
        return

    logger.info("Creating visualizations...")

    try:
        optimal_k = analysis_results['optimal_k']

        # Get feature names from model metadata
        feature_names = None
        if hasattr(model, 'results') and 'metadata' in model.results:
            feature_names = model.results['metadata'].get('feature_names')

        # Create visualization for optimal K
        viz_path = Path(args.output_dir) / f"clusters_k{optimal_k}.png"
        visualize_phenotypic_clusters(
            model,
            k=optimal_k,
            feature_names=feature_names,
            save_path=str(viz_path)
        )

        # Create visualizations for all K if requested
        if args.analyze_all_k:
            for k in range(args.min_k, args.max_k + 1):
                if k != optimal_k:
                    viz_path = Path(args.output_dir) / f"clusters_k{k}.png"
                    visualize_phenotypic_clusters(
                        model,
                        k=k,
                        feature_names=feature_names,
                        save_path=str(viz_path)
                    )

        logger.info("Visualizations created successfully")

    except Exception as e:
        logger.error(f"Failed to create visualizations: {e}")


def save_results(model: PhenotypicNeuralAdmixture,
                analysis_results: Dict[str, Any],
                gene_associations: Dict[int, pd.DataFrame],
                args: argparse.Namespace,
                run_name: str):
    """Save all results to files."""
    logger.info("Saving results...")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save analysis results
    import json
    with open(output_path / f"{run_name}_analysis.json", 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)

    # Save cluster assignments and centers for all K
    for k in range(args.min_k, args.max_k + 1):
        assignments = model.get_cluster_assignments(k)
        centers = model.get_cluster_centers(k)

        np.save(output_path / f"{run_name}_assignments_k{k}.npy", assignments)
        np.save(output_path / f"{run_name}_centers_k{k}.npy", centers)

    # Save gene associations
    for k, associations_df in gene_associations.items():
        if not associations_df.empty:
            associations_df.to_csv(
                output_path / f"{run_name}_gene_associations_k{k}.csv",
                index=False
            )

    # Save summary report
    create_summary_report(model, analysis_results, gene_associations,
                         output_path / f"{run_name}_summary.txt", args)

    logger.info(f"All results saved to {output_path}")


def create_summary_report(model: PhenotypicNeuralAdmixture,
                         analysis_results: Dict[str, Any],
                         gene_associations: Dict[int, pd.DataFrame],
                         report_path: Path,
                         args: argparse.Namespace):
    """Create a human-readable summary report."""
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHENOTYPIC NEURAL ADMIXTURE ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Source: {args.data_path}\n")
        f.write(f"K Range: {args.min_k}-{args.max_k}\n")
        f.write(f"Optimal K: {analysis_results['optimal_k']}\n\n")

        # Model parameters
        f.write("MODEL PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Hidden Size: {args.hidden_size}\n")
        f.write(f"Device: {args.device}\n\n")

        # Results summary
        f.write("RESULTS SUMMARY\n")
        f.write("-" * 40 + "\n")

        for k in range(args.min_k, args.max_k + 1):
            if k in analysis_results:
                r = analysis_results[k]
                f.write(f"\nK = {k}:\n")
                f.write(f"  Samples: {r['n_samples']}\n")
                f.write(f"  Mean Confidence: {r['assignment_stats']['mean_confidence']:.3f}\n")
                f.write(f"  Mean Entropy: {r['assignment_stats']['mean_entropy']:.3f}\n")
                f.write(f"  High Confidence %: {r['assignment_stats']['high_confidence_fraction']*100:.1f}%\n")
                f.write(f"  Cluster Sizes: {r['cluster_stats']['cluster_sizes']}\n")

                if k in gene_associations and not gene_associations[k].empty:
                    f.write(f"  Gene Associations: {len(gene_associations[k])}\n")

        # Optimal K details
        optimal_k = analysis_results['optimal_k']
        if optimal_k in analysis_results:
            f.write(f"\nOPTIMAL K={optimal_k} DETAILS\n")
            f.write("-" * 40 + "\n")
            opt_results = analysis_results[optimal_k]

            f.write(f"Assignment Statistics:\n")
            for key, value in opt_results['assignment_stats'].items():
                f.write(f"  {key}: {value:.3f}\n")

            f.write(f"\nCluster Statistics:\n")
            for key, value in opt_results['cluster_stats'].items():
                if isinstance(value, list):
                    f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {key}: {value:.3f}\n")

        f.write("\n" + "="*80 + "\n")


def main():
    """Main execution function."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup
    args.data_path = validate_data_path(args.data_path)
    run_name = generate_run_name(args)

    # Create output directory and setup logging
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_file = output_path / f"{run_name}.log"
    setup_logging(args.log_level, str(log_file))

    logger.info("Starting Phenotypic Neural ADMIXTURE Analysis")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Arguments: {vars(args)}")

    try:
        # Setup device
        device = setup_device(args.device)

        # Train model
        model = train_model(args, device)

        # Analyze results
        analysis_results = analyze_results(model, args)

        # Extract gene associations
        gene_associations = extract_gene_associations(model, args.data_path, args)

        # Create visualizations
        create_visualizations(model, analysis_results, args)

        # Save all results
        save_results(model, analysis_results, gene_associations, args, run_name)

        # Final summary
        optimal_k = analysis_results['optimal_k']
        logger.info("="*60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*60)
        logger.info(f"Optimal K: {optimal_k}")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info(f"Summary report: {run_name}_summary.txt")

        if optimal_k in gene_associations and not gene_associations[optimal_k].empty:
            logger.info(f"Gene associations: {len(gene_associations[optimal_k])} high-confidence")

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
