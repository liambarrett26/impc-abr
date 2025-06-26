#!/usr/bin/env python3
"""
Preprocessing script for shared data generation.

This script loads and preprocesses IMPC ABR data once, saving the results
for use by all parallel GMM model training jobs.
"""

import numpy as np
import pandas as pd
import pickle
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Import preprocessing components
from loader import IMPCABRLoader, load_impc_data
from preproc import ABRPreprocessor, create_default_config, preprocess_abr_data


def setup_logging(output_dir: Path, log_level: str = "INFO"):
    """Setup logging for preprocessing."""
    log_file = output_dir / "preprocessing.log"
    
    # Create logger
    logger = logging.getLogger('PreprocessingPipeline')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def preprocess_and_save(data_path: str,
                       output_dir: str,
                       min_mutants: int = 3,
                       min_controls: int = 20,
                       log_level: str = "INFO") -> dict:
    """
    Load and preprocess data, saving results for parallel processing.
    
    Args:
        data_path: Path to IMPC ABR data file
        output_dir: Directory to save preprocessed data
        min_mutants: Minimum mutant mice per experimental group
        min_controls: Minimum control mice per group
        log_level: Logging level
        
    Returns:
        Dictionary with preprocessing summary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_path, log_level)
    logger.info("Starting data preprocessing for parallel GMM pipeline")
    
    # Record start time
    start_time = datetime.now()
    
    try:
        # Step 1: Load data
        logger.info(f"Loading data from {data_path}")
        loader = IMPCABRLoader(data_path)
        df, experimental_groups = loader.load_and_prepare(
            min_mutants=min_mutants,
            min_controls=min_controls
        )
        
        logger.info(f"Loaded {len(df)} samples from {len(experimental_groups) if experimental_groups else 0} groups")
        
        # Step 2: Preprocess data
        logger.info("Preprocessing ABR data")
        config = create_default_config()
        preprocessor = ABRPreprocessor(config)
        normalized_data = preprocessor.fit_transform(df)
        
        logger.info(f"Normalized data shape: {normalized_data.shape}")
        
        # Step 3: Save preprocessed data
        logger.info("Saving preprocessed data")
        
        # Save normalized data
        np.save(output_path / "normalized_data.npy", normalized_data)
        
        # Save metadata (keeping only necessary columns)
        metadata_cols = [
            'specimen_id', 'gene_symbol', 'allele_symbol', 'zygosity',
            'sex', 'age_in_weeks', 'weight', 'phenotyping_center',
            'pipeline_name', 'genetic_background', 'biological_sample_group'
        ]
        
        # Add ABR columns for original scale visualization
        abr_cols = [
            '6kHz-evoked ABR Threshold',
            '12kHz-evoked ABR Threshold',
            '18kHz-evoked ABR Threshold',
            '24kHz-evoked ABR Threshold',
            '30kHz-evoked ABR Threshold'
        ]
        
        # Keep available columns
        available_cols = [col for col in metadata_cols + abr_cols if col in df.columns]
        metadata = df[available_cols].copy()
        metadata.to_csv(output_path / "metadata.csv", index=False)
        
        # Save preprocessor for potential inverse transforms
        with open(output_path / "preprocessor.pkl", 'wb') as f:
            pickle.dump(preprocessor, f)
            
        # Save preprocessing configuration
        preproc_info = {
            'timestamp': datetime.now().isoformat(),
            'data_path': str(data_path),
            'n_samples': len(df),
            'n_features': normalized_data.shape[1],
            'n_experimental_groups': len(experimental_groups) if experimental_groups else 0,
            'min_mutants': min_mutants,
            'min_controls': min_controls,
            'preprocessing_config': {
                'abr_columns': config.abr_columns,
                'grouping_columns': config.grouping_columns,
                'target_range': config.target_range,
                'center_threshold': config.center_threshold
            },
            'technical_groups': len(preprocessor.group_scalers),
            'processing_time': (datetime.now() - start_time).total_seconds()
        }
        
        with open(output_path / "preprocessing_info.json", 'w') as f:
            json.dump(preproc_info, f, indent=2)
            
        # Save data statistics
        data_stats = {
            'normalized_data': {
                'mean': float(np.mean(normalized_data)),
                'std': float(np.std(normalized_data)),
                'min': float(np.min(normalized_data)),
                'max': float(np.max(normalized_data)),
                'shape': normalized_data.shape
            },
            'metadata': {
                'n_unique_genes': int(metadata['gene_symbol'].nunique()) if 'gene_symbol' in metadata else 0,
                'n_centers': int(metadata['phenotyping_center'].nunique()) if 'phenotyping_center' in metadata else 0,
                'sex_distribution': metadata['sex'].value_counts().to_dict() if 'sex' in metadata else {},
                'sample_group_distribution': metadata['biological_sample_group'].value_counts().to_dict() if 'biological_sample_group' in metadata else {}
            }
        }
        
        with open(output_path / "data_statistics.json", 'w') as f:
            json.dump(data_stats, f, indent=2)
            
        logger.info(f"Preprocessing completed in {preproc_info['processing_time']:.1f} seconds")
        logger.info(f"Results saved to {output_path}")
        
        # Print summary
        print("\nPreprocessing Summary:")
        print(f"  Total samples: {preproc_info['n_samples']}")
        print(f"  Normalized features: {preproc_info['n_features']}")
        print(f"  Technical groups: {preproc_info['technical_groups']}")
        print(f"  Output directory: {output_path}")
        
        return preproc_info
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


def main():
    """Main entry point for preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Preprocess IMPC ABR data for parallel GMM clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to IMPC ABR data file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='shared_data',
        help='Output directory for preprocessed data'
    )
    
    parser.add_argument(
        '--min-mutants',
        type=int,
        default=3,
        help='Minimum mutant mice per experimental group'
    )
    
    parser.add_argument(
        '--min-controls',
        type=int,
        default=20,
        help='Minimum control mice per experimental group'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Run preprocessing
    try:
        preprocess_and_save(
            data_path=args.data_path,
            output_dir=args.output_dir,
            min_mutants=args.min_mutants,
            min_controls=args.min_controls,
            log_level=args.log_level
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())