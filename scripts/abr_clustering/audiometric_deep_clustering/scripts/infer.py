#!/usr/bin/env python3
"""
Inference script for ContrastiveVAE-DEC model.

This script provides inference capabilities for applying the trained model
to new audiometric data, generating cluster assignments and phenotype predictions.
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
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
from data.dataset import IMPCABRDataset
from data.preprocessor import ABRPreprocessor
from evaluation.phenotype_analysis import HearingLossClassifier


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Apply ContrastiveVAE-DEC model to new audiometric data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and configuration
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model-config', type=str, default='config/model_config.yaml',
                       help='Path to model configuration file')
    
    # Input data
    parser.add_argument('--input-data', type=str, required=True,
                       help='Path to input data file (CSV format)')
    parser.add_argument('--feature-columns', type=str, nargs='+',
                       help='Column names for ABR features (if not default)')
    parser.add_argument('--gene-column', type=str, default='gene_symbol',
                       help='Column name for gene labels (optional)')
    parser.add_argument('--mouse-id-column', type=str, default='specimen_id',
                       help='Column name for mouse IDs')
    
    # Processing options
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, etc.)')
    parser.add_argument('--preprocessor-path', type=str,
                       help='Path to saved preprocessor (if available)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='inference_outputs',
                       help='Directory for inference outputs')
    parser.add_argument('--output-format', type=str, default='csv',
                       choices=['csv', 'json', 'both'],
                       help='Output format for results')
    parser.add_argument('--save-embeddings', action='store_true',
                       help='Save latent embeddings')
    parser.add_argument('--save-probabilities', action='store_true',
                       help='Save cluster assignment probabilities')
    parser.add_argument('--save-reconstructions', action='store_true',
                       help='Save feature reconstructions')
    
    # Analysis options
    parser.add_argument('--phenotype-analysis', action='store_true',
                       help='Perform phenotype classification')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                       help='Minimum confidence for cluster assignments')
    parser.add_argument('--include-attention', action='store_true',
                       help='Include attention weights in output')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress console output')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def load_model_config(config_path: str) -> Dict[str, Any]:
    """Load model configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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


def load_input_data(file_path: str, feature_columns: Optional[List[str]] = None,
                   gene_column: str = 'gene_symbol',
                   mouse_id_column: str = 'specimen_id') -> Dict[str, Any]:
    """Load and validate input data."""
    # Load data
    data = pd.read_csv(file_path)
    logging.info(f"Loaded input data: {len(data)} samples")
    
    # Define default feature columns if not provided
    if feature_columns is None:
        # Look for common ABR feature column patterns
        abr_patterns = ['6kHz', '12kHz', '18kHz', '24kHz', '30kHz', 'Click', '_6_', '_12_', '_18_', '_24_', '_30_']
        feature_columns = []
        
        for col in data.columns:
            if any(pattern in col for pattern in abr_patterns):
                feature_columns.append(col)
        
        if len(feature_columns) < 6:
            # Fall back to first 6 numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            feature_columns = numeric_cols[:6].tolist()
        
        logging.info(f"Auto-detected feature columns: {feature_columns}")
    
    # Validate required columns
    missing_cols = [col for col in feature_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    
    # Extract features
    features = data[feature_columns].values
    
    # Extract optional columns
    gene_labels = None
    if gene_column in data.columns:
        gene_labels = data[gene_column].values
    
    mouse_ids = None
    if mouse_id_column in data.columns:
        mouse_ids = data[mouse_id_column].values
    else:
        mouse_ids = np.arange(len(data))
    
    # Check for missing values
    if np.isnan(features).any():
        logging.warning("Found missing values in features - will be filled with median")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        features = imputer.fit_transform(features)
    
    return {
        'features': features,
        'gene_labels': gene_labels,
        'mouse_ids': mouse_ids,
        'feature_columns': feature_columns,
        'raw_data': data
    }


def setup_preprocessor(config: Dict[str, Any], preprocessor_path: Optional[str] = None) -> ABRPreprocessor:
    """Setup data preprocessor."""
    if preprocessor_path and Path(preprocessor_path).exists():
        # Load saved preprocessor
        import pickle
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        logging.info(f"Loaded preprocessor from {preprocessor_path}")
    else:
        # Create new preprocessor
        preprocessor = ABRPreprocessor(
            normalize=True,
            add_pca=True,
            n_pca_components=config['data']['pca_features']
        )
        logging.info("Created new preprocessor")
    
    return preprocessor


def load_trained_model(checkpoint_path: str, config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_data = load_checkpoint(
        model=model,
        filepath=checkpoint_path,
        map_location=device
    )
    
    model.eval()
    logging.info(f"Loaded trained model from checkpoint (epoch {checkpoint_data['epoch']})")
    
    return model


def run_inference(model: torch.nn.Module, features: np.ndarray, gene_labels: Optional[np.ndarray],
                 mouse_ids: np.ndarray, batch_size: int, device: torch.device,
                 include_attention: bool = False) -> Dict[str, np.ndarray]:
    """Run model inference on input data."""
    model.eval()
    
    # Create dataset and dataloader
    dataset = IMPCABRDataset(
        data=None,
        features=features,
        gene_labels=gene_labels,
        mouse_ids=mouse_ids,
        mode='eval'
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == 'cuda'
    )
    
    # Run inference
    all_embeddings = []
    all_reconstructions = []
    all_cluster_probs = []
    all_cluster_assignments = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch_features = batch['features'].to(device)
            
            # Forward pass
            output = model(batch_features, return_attention=include_attention)
            
            # Extract outputs
            embeddings = output['latent_z'].cpu().numpy()
            reconstructions = output['reconstruction'].cpu().numpy()
            
            all_embeddings.append(embeddings)
            all_reconstructions.append(reconstructions)
            
            # Cluster assignments if available
            if 'q' in output:
                cluster_probs = output['q'].cpu().numpy()
                cluster_assignments = np.argmax(cluster_probs, axis=1)
                all_cluster_probs.append(cluster_probs)
                all_cluster_assignments.append(cluster_assignments)
            
            # Attention weights if requested
            if include_attention and 'attention_weights' in output:
                attention_weights = output['attention_weights'].cpu().numpy()
                all_attention_weights.append(attention_weights)
    
    # Concatenate results
    results = {
        'embeddings': np.concatenate(all_embeddings, axis=0),
        'reconstructions': np.concatenate(all_reconstructions, axis=0)
    }
    
    if all_cluster_probs:
        results['cluster_probabilities'] = np.concatenate(all_cluster_probs, axis=0)
        results['cluster_assignments'] = np.concatenate(all_cluster_assignments, axis=0)
    
    if all_attention_weights:
        results['attention_weights'] = np.concatenate(all_attention_weights, axis=0)
    
    return results


def perform_phenotype_classification(features: np.ndarray, cluster_assignments: np.ndarray) -> Dict[str, Any]:
    """Perform biological phenotype classification."""
    classifier = HearingLossClassifier()
    
    # Classify individual patterns
    individual_patterns = []
    for abr_features in features[:, :6]:  # First 6 features are ABR
        pattern = classifier.classify_individual_pattern(abr_features)
        individual_patterns.append(pattern)
    
    # Classify cluster patterns
    cluster_patterns = classifier.classify_cluster_patterns(cluster_assignments, features[:, :6])
    
    return {
        'individual_patterns': individual_patterns,
        'cluster_patterns': cluster_patterns
    }


def filter_low_confidence_predictions(results: Dict[str, np.ndarray],
                                    confidence_threshold: float) -> Dict[str, np.ndarray]:
    """Filter out low-confidence predictions."""
    if 'cluster_probabilities' not in results:
        logging.warning("No cluster probabilities available for confidence filtering")
        return results
    
    cluster_probs = results['cluster_probabilities']
    max_probs = np.max(cluster_probs, axis=1)
    high_confidence_mask = max_probs >= confidence_threshold
    
    logging.info(f"Filtered to {np.sum(high_confidence_mask)}/{len(high_confidence_mask)} "
                f"high-confidence predictions (>= {confidence_threshold})")
    
    # Apply filter to all results
    filtered_results = {}
    for key, values in results.items():
        if isinstance(values, np.ndarray) and len(values) == len(high_confidence_mask):
            filtered_results[key] = values[high_confidence_mask]
        else:
            filtered_results[key] = values
    
    filtered_results['confidence_mask'] = high_confidence_mask
    return filtered_results


def save_results(results: Dict[str, np.ndarray], input_data: Dict[str, Any],
                phenotype_results: Optional[Dict[str, Any]], args: argparse.Namespace,
                output_dir: Path) -> None:
    """Save inference results to files."""
    mouse_ids = input_data['mouse_ids']
    gene_labels = input_data['gene_labels']
    
    # Filter mouse IDs and gene labels if confidence filtering was applied
    if 'confidence_mask' in results:
        mask = results['confidence_mask']
        mouse_ids = mouse_ids[mask]
        if gene_labels is not None:
            gene_labels = gene_labels[mask]
    
    # Prepare main results DataFrame
    results_data = {
        'mouse_id': mouse_ids,
    }
    
    if gene_labels is not None:
        results_data['gene_label'] = gene_labels
    
    if 'cluster_assignments' in results:
        results_data['cluster_assignment'] = results['cluster_assignments']
    
    if 'cluster_probabilities' in results and args.save_probabilities:
        cluster_probs = results['cluster_probabilities']
        for i in range(cluster_probs.shape[1]):
            results_data[f'cluster_{i}_probability'] = cluster_probs[:, i]
        results_data['max_probability'] = np.max(cluster_probs, axis=1)
    
    # Add phenotype classifications if available
    if phenotype_results and 'individual_patterns' in phenotype_results:
        individual_patterns = phenotype_results['individual_patterns']
        if 'confidence_mask' in results:
            mask = results['confidence_mask']
            individual_patterns = [individual_patterns[i] for i in range(len(individual_patterns)) if mask[i]]
        
        results_data['phenotype_pattern'] = [p['pattern_type'] for p in individual_patterns]
        results_data['phenotype_severity'] = [p['severity'] for p in individual_patterns]
        results_data['overall_threshold'] = [p['overall_average'] for p in individual_patterns]
        results_data['frequency_slope'] = [p['frequency_slope'] for p in individual_patterns]
    
    # Create main results DataFrame
    results_df = pd.DataFrame(results_data)
    
    # Save main results
    if args.output_format in ['csv', 'both']:
        csv_file = output_dir / 'inference_results.csv'
        results_df.to_csv(csv_file, index=False)
        logging.info(f"Results saved to {csv_file}")
    
    if args.output_format in ['json', 'both']:
        json_file = output_dir / 'inference_results.json'
        results_df.to_json(json_file, orient='records', indent=2)
        logging.info(f"Results saved to {json_file}")
    
    # Save embeddings if requested
    if args.save_embeddings and 'embeddings' in results:
        embeddings_file = output_dir / 'latent_embeddings.npz'
        np.savez_compressed(
            embeddings_file,
            embeddings=results['embeddings'],
            mouse_ids=mouse_ids,
            cluster_assignments=results.get('cluster_assignments'),
        )
        logging.info(f"Embeddings saved to {embeddings_file}")
    
    # Save reconstructions if requested
    if args.save_reconstructions and 'reconstructions' in results:
        reconstructions_file = output_dir / 'feature_reconstructions.npz'
        np.savez_compressed(
            reconstructions_file,
            original_features=input_data['features'],
            reconstructed_features=results['reconstructions'],
            mouse_ids=mouse_ids,
            feature_columns=input_data['feature_columns']
        )
        logging.info(f"Reconstructions saved to {reconstructions_file}")
    
    # Save attention weights if available
    if 'attention_weights' in results:
        attention_file = output_dir / 'attention_weights.npz'
        np.savez_compressed(
            attention_file,
            attention_weights=results['attention_weights'],
            mouse_ids=mouse_ids
        )
        logging.info(f"Attention weights saved to {attention_file}")
    
    # Save phenotype analysis results if available
    if phenotype_results:
        phenotype_file = output_dir / 'phenotype_analysis.json'
        with open(phenotype_file, 'w') as f:
            json.dump(phenotype_results, f, indent=2, default=str)
        logging.info(f"Phenotype analysis saved to {phenotype_file}")


def generate_inference_summary(results: Dict[str, np.ndarray], input_data: Dict[str, Any],
                              phenotype_results: Optional[Dict[str, Any]]) -> str:
    """Generate summary of inference results."""
    summary_lines = []
    summary_lines.append("INFERENCE SUMMARY")
    summary_lines.append("="*50)
    summary_lines.append(f"Input samples: {len(input_data['mouse_ids'])}")
    
    if 'confidence_mask' in results:
        high_conf = np.sum(results['confidence_mask'])
        summary_lines.append(f"High-confidence predictions: {high_conf}")
    
    if 'cluster_assignments' in results:
        cluster_counts = np.bincount(results['cluster_assignments'])
        summary_lines.append(f"Number of clusters: {len(cluster_counts)}")
        summary_lines.append("Cluster distribution:")
        for i, count in enumerate(cluster_counts):
            summary_lines.append(f"  Cluster {i}: {count} samples")
    
    if phenotype_results and 'individual_patterns' in phenotype_results:
        individual_patterns = phenotype_results['individual_patterns']
        
        # Pattern type distribution
        pattern_types = [p['pattern_type'] for p in individual_patterns]
        pattern_counts = {}
        for pattern in pattern_types:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        summary_lines.append("\nPhenotype pattern distribution:")
        for pattern, count in pattern_counts.items():
            summary_lines.append(f"  {pattern}: {count} samples")
        
        # Severity distribution
        severities = [p['severity'] for p in individual_patterns]
        severity_counts = {}
        for severity in severities:
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        summary_lines.append("\nSeverity distribution:")
        for severity, count in severity_counts.items():
            summary_lines.append(f"  {severity}: {count} samples")
    
    return '\n'.join(summary_lines)


def main():
    """Main inference function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(
        log_level=args.log_level,
        log_dir=output_dir,
        console_output=not args.quiet,
        file_output=True
    )
    
    # Setup reproducibility
    ensure_reproducibility(args.seed)
    
    # Load model configuration
    model_config = load_model_config(args.model_config)
    
    # Setup device
    device = setup_device(args.device)
    
    # Load input data
    input_data = load_input_data(
        args.input_data,
        args.feature_columns,
        args.gene_column,
        args.mouse_id_column
    )
    
    # Setup preprocessor
    preprocessor = setup_preprocessor(model_config, args.preprocessor_path)
    
    # Preprocess features
    processed_features = preprocessor.fit_transform(input_data['features'])
    
    # Load trained model
    model = load_trained_model(args.checkpoint, model_config, device)
    
    # Run inference
    logging.info("Running model inference...")
    results = run_inference(
        model=model,
        features=processed_features,
        gene_labels=input_data['gene_labels'],
        mouse_ids=input_data['mouse_ids'],
        batch_size=args.batch_size,
        device=device,
        include_attention=args.include_attention
    )
    
    # Filter low-confidence predictions if requested
    if args.confidence_threshold > 0 and 'cluster_probabilities' in results:
        results = filter_low_confidence_predictions(results, args.confidence_threshold)
    
    # Perform phenotype analysis if requested
    phenotype_results = None
    if args.phenotype_analysis and 'cluster_assignments' in results:
        logging.info("Performing phenotype classification...")
        phenotype_results = perform_phenotype_classification(
            processed_features, results['cluster_assignments']
        )
    
    # Save results
    save_results(results, input_data, phenotype_results, args, output_dir)
    
    # Generate and save summary
    summary = generate_inference_summary(results, input_data, phenotype_results)
    print(summary)
    
    summary_file = output_dir / 'inference_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    logging.info(f"Inference completed! Results saved to {output_dir}")


if __name__ == '__main__':
    main()