"""
Complete pipeline for audiometric phenotype discovery using GMM clustering.

This module orchestrates the entire analysis pipeline from data loading
through final results generation and visualization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
import json
import pickle
from datetime import datetime
from typing import Dict, Any, Optional
import warnings

# Import pipeline components
from loader import IMPCABRLoader, load_impc_data
from preproc import ABRPreprocessor, create_default_config, preprocess_abr_data
from gmm import AudiometricGMM, create_default_gmm_config, GMMConfig
from analysis import analyze_gmm_results, AudiometricAnalyzer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class AudiometricPhenotypePipeline:
    """
    Complete pipeline for discovering audiometric phenotypes in IMPC data.

    Integrates data loading, preprocessing, clustering, and analysis into
    a single cohesive workflow with comprehensive logging and result tracking.
    """

    def __init__(self,
                 output_dir: str = "results",
                 log_level: str = "INFO",
                 random_state: int = 42):
        """
        Initialize the pipeline.

        Args:
            output_dir: Directory for all output files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            random_state: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.random_state = random_state

        # Setup logging
        self._setup_logging(log_level)

        # Pipeline components
        self.loader: Optional[IMPCABRLoader] = None
        self.preprocessor: Optional[ABRPreprocessor] = None
        self.gmm_model: Optional[AudiometricGMM] = None
        self.analyzer: Optional[AudiometricAnalyzer] = None

        # Data storage
        self.raw_data: Optional[pd.DataFrame] = None
        self.experimental_groups: Optional[Dict] = None
        self.normalized_data: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_probabilities: Optional[np.ndarray] = None

        # Results storage
        self.analysis_results: Optional[Dict[str, Any]] = None
        self.visualization_files: Optional[Dict[str, str]] = None

        # Pipeline configuration
        self.config = {
            'pipeline_version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'random_state': random_state,
            'output_dir': str(output_dir)
        }

        self.logger.info(f"Initialized AudiometricPhenotypePipeline in {output_dir}")

    def _setup_logging(self, log_level: str):
        """Setup comprehensive logging for the pipeline."""
        log_file = self.output_dir / "pipeline.log"

        # Create logger
        self.logger = logging.getLogger('AudiometricPipeline')
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)

        self.logger.info(f"Logging initialized - Level: {log_level}, Log file: {log_file}")

    def load_data(self,
                  data_path: str,
                  min_mutants: int = 3,
                  min_controls: int = 20) -> 'AudiometricPhenotypePipeline':
        """
        Load and filter IMPC ABR data.

        Args:
            data_path: Path to IMPC data file
            min_mutants: Minimum mutant mice per experimental group
            min_controls: Minimum control mice per group

        Returns:
            Self for method chaining
        """
        self.logger.info(f"Loading data from {data_path}")

        try:
            # Initialize loader
            self.loader = IMPCABRLoader(data_path)

            # Load and prepare data
            self.raw_data, self.experimental_groups = self.loader.load_and_prepare(
                min_mutants=min_mutants,
                min_controls=min_controls
            )

            # Update configuration
            self.config.update({
                'data_path': str(data_path),
                'min_mutants': min_mutants,
                'min_controls': min_controls,
                'total_mice': len(self.raw_data),
                'experimental_groups': len(self.experimental_groups) if self.experimental_groups else 0
            })

            self.logger.info(f"Data loading complete: {len(self.raw_data)} mice, "
                           f"{len(self.experimental_groups) if self.experimental_groups else 0} experimental groups")

        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            raise

        return self

    def preprocess_data(self,
                       preproc_config: Optional[Dict] = None) -> 'AudiometricPhenotypePipeline':
        """
        Preprocess ABR data with normalization and batch correction.

        Args:
            preproc_config: Custom preprocessing configuration

        Returns:
            Self for method chaining
        """
        if self.raw_data is None:
            raise ValueError("Data must be loaded before preprocessing")

        self.logger.info("Starting data preprocessing")

        try:
            # Create preprocessing configuration
            if preproc_config is None:
                config = create_default_config()
            else:
                config = create_default_config()
                for key, value in preproc_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

            # Initialize and fit preprocessor
            self.preprocessor = ABRPreprocessor(config)
            self.normalized_data = self.preprocessor.fit_transform(self.raw_data)

            # Update configuration
            self.config.update({
                'preprocessing': {
                    'normalization_range': config.target_range,
                    'grouping_columns': config.grouping_columns,
                    'center_threshold': config.center_threshold,
                    'output_shape': self.normalized_data.shape
                }
            })

            self.logger.info(f"Preprocessing complete: {self.normalized_data.shape} normalized features")

        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise

        return self

    def fit_clustering(self,
                      gmm_config: Optional[Dict] = None) -> 'AudiometricPhenotypePipeline':
        """
        Fit GMM clustering model with model selection.

        Args:
            gmm_config: Custom GMM configuration parameters

        Returns:
            Self for method chaining
        """
        if self.normalized_data is None:
            raise ValueError("Data must be preprocessed before clustering")

        self.logger.info("Starting GMM clustering")

        try:
            # Create GMM configuration
            if gmm_config is None:
                config = create_default_gmm_config(random_state=self.random_state)
            else:
                # Don't pass random_state if it's already in gmm_config
                if 'random_state' not in gmm_config:
                    gmm_config['random_state'] = self.random_state
                config = create_default_gmm_config(**gmm_config)

            # Initialize and fit GMM
            self.gmm_model = AudiometricGMM(config)
            self.gmm_model.fit(self.normalized_data)

            # Get predictions
            self.cluster_labels = self.gmm_model.predict(self.normalized_data)
            self.cluster_probabilities = self.gmm_model.predict_proba(self.normalized_data)

            # Get model metrics
            metrics = self.gmm_model.get_metrics()

            # Update configuration
            self.config.update({
                'clustering': {
                    'n_components_range': config.n_components_range,
                    'covariance_types': config.covariance_types,
                    'n_init': config.n_init,
                    'n_bootstrap': config.n_bootstrap,
                    'best_n_components': metrics.n_components,
                    'best_covariance_type': metrics.covariance_type,
                    'bic_score': metrics.bic,
                    'aic_score': metrics.aic,
                    'silhouette_score': metrics.silhouette,
                    'stability_score': metrics.stability_score
                }
            })

            self.logger.info(f"Clustering complete: {metrics.n_components} clusters identified")
            self.logger.info(f"Best model metrics: BIC={metrics.bic:.2f}, "
                           f"Silhouette={metrics.silhouette:.3f}, "
                           f"Stability={metrics.stability_score:.3f}")

        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            raise

        return self

    def analyze_results(self) -> 'AudiometricPhenotypePipeline':
        """
        Perform comprehensive analysis of clustering results.

        Returns:
            Self for method chaining
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering must be completed before analysis")

        self.logger.info("Starting results analysis")

        try:
            # Get original data for interpretable visualizations
            abr_columns = [
                '6kHz-evoked ABR Threshold',
                '12kHz-evoked ABR Threshold',
                '18kHz-evoked ABR Threshold',
                '24kHz-evoked ABR Threshold',
                '30kHz-evoked ABR Threshold'
            ]

            original_data = self.raw_data[abr_columns].values

            # Perform comprehensive analysis
            self.analysis_results, self.visualization_files = analyze_gmm_results(
                normalized_data=self.normalized_data,
                cluster_labels=self.cluster_labels,
                cluster_probabilities=self.cluster_probabilities,
                metadata=self.raw_data,
                original_data=original_data,
                output_dir=str(self.output_dir)
            )

            # Update configuration with analysis summary
            self.config.update({
                'analysis': {
                    'n_clusters_final': self.analysis_results['n_clusters'],
                    'total_samples_analyzed': self.analysis_results['summary_statistics']['total_samples'],
                    'avg_assignment_confidence': self.analysis_results['summary_statistics']['avg_assignment_confidence'],
                    'cluster_patterns': {
                        str(cid): char.pattern_type
                        for cid, char in self.analysis_results['cluster_characteristics'].items()
                    },
                    'visualization_files': list(self.visualization_files.keys())
                }
            })

            self.logger.info(f"Analysis complete: {len(self.visualization_files)} files generated")

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise

        return self

    def save_results(self, save_models: bool = True) -> Dict[str, str]:
        """
        Save all pipeline results and models.

        Args:
            save_models: Whether to save fitted models (preprocessor, GMM)

        Returns:
            Dictionary mapping result types to file paths
        """
        self.logger.info("Saving pipeline results")

        saved_files = {}

        try:
            # Save configuration
            config_path = self.output_dir / "pipeline_config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
            saved_files['config'] = str(config_path)

            # Save analysis results
            if self.analysis_results:
                results_path = self.output_dir / "analysis_results.json"
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = self._make_json_serializable(self.analysis_results)
                with open(results_path, 'w') as f:
                    json.dump(serializable_results, f, indent=2, default=str)
                saved_files['analysis_results'] = str(results_path)

            # Save cluster assignments
            if self.cluster_labels is not None:
                assignments_path = self.output_dir / "cluster_assignments.csv"
                assignments_df = pd.DataFrame({
                    'sample_index': range(len(self.cluster_labels)),
                    'cluster_label': self.cluster_labels,
                    'max_probability': self.cluster_probabilities.max(axis=1),
                    'assignment_confidence': self.cluster_probabilities.max(axis=1)
                })

                # Add individual cluster probabilities
                n_clusters = self.cluster_probabilities.shape[1]
                for i in range(n_clusters):
                    assignments_df[f'cluster_{i}_prob'] = self.cluster_probabilities[:, i]

                assignments_df.to_csv(assignments_path, index=False)
                saved_files['cluster_assignments'] = str(assignments_path)

            # Save processed data
            if self.normalized_data is not None:
                processed_data_path = self.output_dir / "normalized_data.csv"
                feature_names = [f'normalized_{freq}kHz' for freq in [6, 12, 18, 24, 30]]
                processed_df = pd.DataFrame(self.normalized_data, columns=feature_names)
                processed_df.to_csv(processed_data_path, index=False)
                saved_files['normalized_data'] = str(processed_data_path)

            # Save models if requested
            if save_models:
                if self.preprocessor:
                    preprocessor_path = self.output_dir / "preprocessor.pkl"
                    with open(preprocessor_path, 'wb') as f:
                        pickle.dump(self.preprocessor, f)
                    saved_files['preprocessor_model'] = str(preprocessor_path)

                if self.gmm_model:
                    gmm_path = self.output_dir / "gmm_model.pkl"
                    with open(gmm_path, 'wb') as f:
                        pickle.dump(self.gmm_model, f)
                    saved_files['gmm_model'] = str(gmm_path)

            # Add visualization files
            if self.visualization_files:
                saved_files.update(self.visualization_files)

            # Create summary file
            summary_path = self.output_dir / "pipeline_summary.txt"
            self._create_pipeline_summary(summary_path, saved_files)
            saved_files['pipeline_summary'] = str(summary_path)

            self.logger.info(f"Results saved: {len(saved_files)} files in {self.output_dir}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise

        return saved_files

    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            # Handle custom objects like ClusterCharacteristics
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj

    def _create_pipeline_summary(self, summary_path: Path, saved_files: Dict[str, str]):
        """Create a human-readable summary of the pipeline execution."""
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("AUDIOMETRIC PHENOTYPE DISCOVERY PIPELINE SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Execution Date: {self.config['created_at']}\n")
            f.write(f"Pipeline Version: {self.config['pipeline_version']}\n")
            f.write(f"Output Directory: {self.config['output_dir']}\n\n")

            # Data summary
            f.write("DATA LOADING\n")
            f.write("-" * 40 + "\n")
            f.write(f"Input file: {self.config.get('data_path', 'N/A')}\n")
            f.write(f"Total mice: {self.config.get('total_mice', 'N/A')}\n")
            f.write(f"Experimental groups: {self.config.get('experimental_groups', 'N/A')}\n")
            f.write(f"Min mutants per group: {self.config.get('min_mutants', 'N/A')}\n")
            f.write(f"Min controls per group: {self.config.get('min_controls', 'N/A')}\n\n")

            # Preprocessing summary
            if 'preprocessing' in self.config:
                f.write("PREPROCESSING\n")
                f.write("-" * 40 + "\n")
                prep = self.config['preprocessing']
                f.write(f"Normalization range: {prep.get('normalization_range', 'N/A')}\n")
                f.write(f"Technical grouping: {len(prep.get('grouping_columns', []))} variables\n")
                f.write(f"Output shape: {prep.get('output_shape', 'N/A')}\n\n")

            # Clustering summary
            if 'clustering' in self.config:
                f.write("CLUSTERING\n")
                f.write("-" * 40 + "\n")
                clust = self.config['clustering']
                f.write(f"Components tested: {clust.get('n_components_range', 'N/A')}\n")
                f.write(f"Covariance types: {clust.get('covariance_types', 'N/A')}\n")
                f.write(f"Best model: {clust.get('best_n_components', 'N/A')} components, ")
                f.write(f"{clust.get('best_covariance_type', 'N/A')} covariance\n")
                f.write(f"BIC score: {clust.get('bic_score', 'N/A'):.2f}\n")
                f.write(f"Silhouette score: {clust.get('silhouette_score', 'N/A'):.3f}\n")
                f.write(f"Stability score: {clust.get('stability_score', 'N/A'):.3f}\n\n")

            # Analysis summary
            if 'analysis' in self.config:
                f.write("ANALYSIS RESULTS\n")
                f.write("-" * 40 + "\n")
                anal = self.config['analysis']
                f.write(f"Final clusters: {anal.get('n_clusters_final', 'N/A')}\n")
                f.write(f"Samples analyzed: {anal.get('total_samples_analyzed', 'N/A')}\n")
                f.write(f"Avg assignment confidence: {anal.get('avg_assignment_confidence', 'N/A'):.3f}\n")

                patterns = anal.get('cluster_patterns', {})
                if patterns:
                    f.write("Identified patterns:\n")
                    for cluster_id, pattern in patterns.items():
                        f.write(f"  Cluster {cluster_id}: {pattern}\n")
                f.write("\n")

            # Output files
            f.write("OUTPUT FILES\n")
            f.write("-" * 40 + "\n")
            for file_type, file_path in saved_files.items():
                f.write(f"{file_type}: {file_path}\n")

            f.write("\n" + "=" * 80 + "\n")

    def run_complete_pipeline(self,
                            data_path: str,
                            min_mutants: int = 3,
                            min_controls: int = 20,
                            preproc_config: Optional[Dict] = None,
                            gmm_config: Optional[Dict] = None,
                            save_models: bool = True) -> Dict[str, str]:
        """
        Run the complete pipeline from data loading to results generation.

        Args:
            data_path: Path to IMPC data file
            min_mutants: Minimum mutant mice per experimental group
            min_controls: Minimum control mice per group
            preproc_config: Custom preprocessing configuration
            gmm_config: Custom GMM configuration
            save_models: Whether to save fitted models

        Returns:
            Dictionary mapping result types to file paths
        """
        self.logger.info("Starting complete audiometric phenotype discovery pipeline")

        try:
            # Execute pipeline steps
            (self.load_data(data_path, min_mutants, min_controls)
             .preprocess_data(preproc_config)
             .fit_clustering(gmm_config)
             .analyze_results())

            # Save all results
            saved_files = self.save_results(save_models)

            self.logger.info("Pipeline execution completed successfully")
            return saved_files

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise


def create_argument_parser():
    """Create command-line argument parser for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Audiometric Phenotype Discovery Pipeline",
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
        '--output-dir', '-o',
        type=str,
        default='results',
        help='Output directory for results'
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
        '--min-components',
        type=int,
        default=3,
        help='Minimum number of GMM components to test'
    )

    parser.add_argument(
        '--max-components',
        type=int,
        default=12,
        help='Maximum number of GMM components to test'
    )

    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=100,
        help='Number of bootstrap iterations for stability assessment'
    )

    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    parser.add_argument(
        '--no-save-models',
        action='store_true',
        help='Do not save fitted models (preprocessor, GMM)'
    )

    return parser


def main():
    """Main entry point for command-line execution."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Create GMM configuration from arguments
    gmm_config = {
        'n_components_range': (args.min_components, args.max_components),
        'n_bootstrap': args.n_bootstrap,
        'random_state': args.random_state
    }

    # Initialize and run pipeline
    pipeline = AudiometricPhenotypePipeline(
        output_dir=args.output_dir,
        log_level=args.log_level,
        random_state=args.random_state
    )

    try:
        saved_files = pipeline.run_complete_pipeline(
            data_path=args.data_path,
            min_mutants=args.min_mutants,
            min_controls=args.min_controls,
            gmm_config=gmm_config,
            save_models=not args.no_save_models
        )

        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results saved to: {args.output_dir}")
        print(f"Total files generated: {len(saved_files)}")
        print(f"See {args.output_dir}/pipeline_summary.txt for detailed results")

    except Exception as e:
        print(f"\nPipeline execution failed: {e}")
        print(f"Check {args.output_dir}/pipeline.log for detailed error information")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())