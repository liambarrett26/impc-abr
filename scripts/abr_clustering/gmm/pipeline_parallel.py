#!/usr/bin/env python3
"""
Parallel pipeline for single GMM model training.

This script trains a single GMM model with specific parameters, designed to be
called in parallel by a bash script for different parameter combinations.
"""

import numpy as np
import pandas as pd
import pickle
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys

# Import pipeline components
from loader import IMPCABRLoader, load_impc_data
from preproc import ABRPreprocessor, create_default_config, preprocess_abr_data
from gmm import AudiometricGMM, create_default_gmm_config, GMMConfig
from analysis import analyze_gmm_results, AudiometricAnalyzer


class ParallelGMMPipeline:
    """
    Pipeline for training a single GMM model configuration.
    
    Designed for parallel execution where preprocessing is done once
    and each model configuration is trained independently.
    """
    
    def __init__(self, 
                 n_components: int,
                 covariance_type: str,
                 output_dir: str = "results",
                 shared_data_dir: str = "shared_data",
                 log_level: str = "INFO"):
        """
        Initialize parallel pipeline for specific model configuration.
        
        Args:
            n_components: Number of GMM components for this model
            covariance_type: Covariance type ('full' or 'tied')
            output_dir: Base output directory
            shared_data_dir: Directory containing preprocessed data
            log_level: Logging level
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.shared_data_dir = Path(shared_data_dir)
        
        # Create model-specific output directory
        self.model_name = f"gmm_k{n_components}_{covariance_type}"
        self.output_dir = Path(output_dir) / self.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging(log_level)
        
        self.logger.info(f"Initialized parallel pipeline for {self.model_name}")
        
    def _setup_logging(self, log_level: str):
        """Setup model-specific logging."""
        log_file = self.output_dir / f"{self.model_name}.log"
        
        # Create logger
        self.logger = logging.getLogger(f'ParallelGMM_{self.model_name}')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
    def load_preprocessed_data(self) -> tuple:
        """
        Load preprocessed data from shared directory.
        
        Returns:
            Tuple of (normalized_data, metadata, preprocessor)
        """
        self.logger.info("Loading preprocessed data from shared directory")
        
        # Load normalized data
        normalized_data_path = self.shared_data_dir / "normalized_data.npy"
        if not normalized_data_path.exists():
            raise FileNotFoundError(f"Preprocessed data not found at {normalized_data_path}")
        
        normalized_data = np.load(normalized_data_path)
        
        # Load metadata
        metadata_path = self.shared_data_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        metadata = pd.read_csv(metadata_path)
        
        # Load preprocessor (for potential inverse transforms)
        preprocessor_path = self.shared_data_dir / "preprocessor.pkl"
        if preprocessor_path.exists():
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
        else:
            preprocessor = None
            
        self.logger.info(f"Loaded data: {normalized_data.shape} samples")
        
        return normalized_data, metadata, preprocessor
        
    def train_model(self, normalized_data: np.ndarray, 
                   n_bootstrap: int = 100,
                   random_state: int = 42) -> dict:
        """
        Train a single GMM model with specified parameters.
        
        Args:
            normalized_data: Preprocessed ABR data
            n_bootstrap: Number of bootstrap iterations for stability
            random_state: Random seed
            
        Returns:
            Dictionary containing model and metrics
        """
        self.logger.info(f"Training GMM with k={self.n_components}, "
                        f"covariance={self.covariance_type}")
        
        start_time = datetime.now()
        
        # Create GMM configuration for single model
        config = create_default_gmm_config(
            n_components_range=(self.n_components, self.n_components),
            covariance_types=[self.covariance_type],
            n_bootstrap=n_bootstrap,
            random_state=random_state
        )
        
        # Initialize and fit model
        gmm = AudiometricGMM(config)
        
        try:
            gmm.fit(normalized_data)
            
            # Get predictions and metrics
            cluster_labels = gmm.predict(normalized_data)
            cluster_probabilities = gmm.predict_proba(normalized_data)
            metrics = gmm.get_metrics()
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Model trained successfully in {training_time:.1f}s")
            self.logger.info(f"Metrics: BIC={metrics.bic:.2f}, "
                           f"Silhouette={metrics.silhouette:.3f}, "
                           f"Stability={metrics.stability_score:.3f}")
            
            # Save model and results immediately
            self._save_model_results(gmm, cluster_labels, cluster_probabilities, 
                                   metrics, training_time)
            
            return {
                'success': True,
                'model': gmm,
                'labels': cluster_labels,
                'probabilities': cluster_probabilities,
                'metrics': metrics,
                'training_time': training_time
            }
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            
            # Save failure information
            failure_info = {
                'success': False,
                'error': str(e),
                'n_components': self.n_components,
                'covariance_type': self.covariance_type,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.output_dir / "training_failed.json", 'w') as f:
                json.dump(failure_info, f, indent=2)
                
            return failure_info
            
    def _save_model_results(self, model, labels, probabilities, metrics, training_time):
        """Save all model results to disk."""
        # Save model
        model_path = self.output_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        # Save predictions
        np.save(self.output_dir / "cluster_labels.npy", labels)
        np.save(self.output_dir / "cluster_probabilities.npy", probabilities)
        
        # Save metrics
        metrics_dict = {
            'n_components': metrics.n_components,
            'covariance_type': metrics.covariance_type,
            'bic': float(metrics.bic),
            'aic': float(metrics.aic),
            'silhouette': float(metrics.silhouette),
            'log_likelihood': float(metrics.log_likelihood),
            'stability_score': float(metrics.stability_score),
            'training_time': training_time,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / "metrics.json", 'w') as f:
            json.dump(metrics_dict, f, indent=2)
            
        # Save cluster centers
        centers = model.get_cluster_centers()
        np.save(self.output_dir / "cluster_centers.npy", centers)
        
        self.logger.info(f"Model results saved to {self.output_dir}")
        
    def run_analysis(self, normalized_data: np.ndarray, metadata: pd.DataFrame,
                    labels: np.ndarray, probabilities: np.ndarray) -> dict:
        """
        Run analysis on the trained model.
        
        Args:
            normalized_data: Preprocessed data
            metadata: Sample metadata
            labels: Cluster assignments
            probabilities: Assignment probabilities
            
        Returns:
            Analysis results dictionary
        """
        self.logger.info("Running cluster analysis")
        
        # Get original data if available
        abr_columns = [
            '6kHz-evoked ABR Threshold',
            '12kHz-evoked ABR Threshold', 
            '18kHz-evoked ABR Threshold',
            '24kHz-evoked ABR Threshold',
            '30kHz-evoked ABR Threshold'
        ]
        
        if all(col in metadata.columns for col in abr_columns):
            original_data = metadata[abr_columns].values
        else:
            original_data = None
            
        # Perform analysis
        analysis_results, visualization_files = analyze_gmm_results(
            normalized_data=normalized_data,
            cluster_labels=labels,
            cluster_probabilities=probabilities,
            metadata=metadata,
            original_data=original_data,
            output_dir=str(self.output_dir)
        )
        
        # Save analysis results
        with open(self.output_dir / "analysis_results.json", 'w') as f:
            # Convert to JSON-serializable format
            serializable_results = self._make_json_serializable(analysis_results)
            json.dump(serializable_results, f, indent=2)
            
        self.logger.info(f"Analysis complete, {len(visualization_files)} visualizations created")
        
        return analysis_results
        
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and custom objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj
            
    def run(self, n_bootstrap: int = 100, random_state: int = 42,
            skip_analysis: bool = False) -> dict:
        """
        Run the complete parallel pipeline for this model configuration.
        
        Args:
            n_bootstrap: Bootstrap iterations for stability assessment
            random_state: Random seed
            skip_analysis: Skip analysis and visualization steps
            
        Returns:
            Dictionary with results or error information
        """
        try:
            # Load preprocessed data
            normalized_data, metadata, preprocessor = self.load_preprocessed_data()
            
            # Train model
            results = self.train_model(normalized_data, n_bootstrap, random_state)
            
            if results['success'] and not skip_analysis:
                # Run analysis
                analysis_results = self.run_analysis(
                    normalized_data, metadata,
                    results['labels'], results['probabilities']
                )
                results['analysis'] = analysis_results
                
            # Create completion marker
            with open(self.output_dir / "completed.txt", 'w') as f:
                f.write(f"Completed at {datetime.now().isoformat()}\n")
                
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            error_info = {
                'success': False,
                'error': str(e),
                'model_name': self.model_name,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.output_dir / "pipeline_failed.json", 'w') as f:
                json.dump(error_info, f, indent=2)
                
            return error_info


def main():
    """Main entry point for parallel pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Train a single GMM model configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        'n_components',
        type=int,
        help='Number of GMM components'
    )
    
    parser.add_argument(
        'covariance_type',
        type=str,
        choices=['full', 'tied'],
        help='GMM covariance type'
    )
    
    # Optional arguments
    parser.add_argument(
        '--shared-data-dir',
        type=str,
        default='shared_data',
        help='Directory containing preprocessed data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Base output directory'
    )
    
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=100,
        help='Number of bootstrap iterations for stability'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Skip analysis and visualization steps'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = ParallelGMMPipeline(
        n_components=args.n_components,
        covariance_type=args.covariance_type,
        output_dir=args.output_dir,
        shared_data_dir=args.shared_data_dir,
        log_level=args.log_level
    )
    
    results = pipeline.run(
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
        skip_analysis=args.skip_analysis
    )
    
    # Exit with appropriate code
    if results.get('success', False):
        print(f"Model {pipeline.model_name} completed successfully")
        sys.exit(0)
    else:
        print(f"Model {pipeline.model_name} failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()