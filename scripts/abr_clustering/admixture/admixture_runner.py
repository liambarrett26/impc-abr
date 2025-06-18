"""
Runner for Neural ADMIXTURE phenotypic clustering.
Orchestrates data loading, preprocessing, training, and analysis.
"""

import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
import time
from datetime import datetime

from admixture_loader import load_admixture_data
from admixture_preproc import preprocess_for_admixture
from phenotypic_adapter import patch_train_function
from phenotypic_adapter import create_phenotypic_admixture_trainer

# Import Neural ADMIXTURE components
# Add all module paths relative to the script location
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
from neural_admixture_original.model.neural_admixture import NeuralAdmixture
from neural_admixture_original.model.train import train

logger = logging.getLogger(__name__)

class AdmixtureRunner:
    """
    Complete pipeline runner for Neural ADMIXTURE phenotypic analysis.
    """

    def __init__(self,
                 output_dir: str = "admixture_results",
                 device: str = "auto",
                 random_seed: int = 42):
        """
        Initialize runner.

        Args:
            output_dir: Directory for outputs
            device: Device for computation ('auto', 'cuda', 'cpu')
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)

        # Setup logging
        self._setup_logging()

        # Storage for pipeline components
        self.features = None
        self.metadata = None
        self.preprocessed_features = None
        self.preprocessor = None
        self.model_results = None
        self.trained_model = None

        logger.info(f"AdmixtureRunner initialized - Device: {self.device}")

    def _setup_logging(self):
        """Setup logging for the runner."""
        log_file = self.output_dir / "admixture_run.log"

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add to logger
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    def load_and_preprocess(self,
                          data_path: str,
                          min_samples_per_gene: int = 3,
                          scaling_method: str = 'standard') -> 'AdmixtureRunner':
        """
        Load and preprocess data.

        Args:
            data_path: Path to data file
            min_samples_per_gene: Minimum samples per gene
            scaling_method: Preprocessing scaling method

        Returns:
            Self for method chaining
        """
        logger.info(f"Loading data from {data_path}")

        # Load data
        self.features, self.metadata = load_admixture_data(
            data_path, min_samples_per_gene
        )

        # Preprocess (ensure data stays on CPU)
        self.preprocessed_features, self.preprocessor = preprocess_for_admixture(
            self.features, self.metadata, scaling_method
        )

        # Ensure preprocessed features are on CPU (Neural ADMIXTURE handles GPU transfers)
        if self.preprocessed_features.is_cuda:
            self.preprocessed_features = self.preprocessed_features.cpu()

        logger.info(f"Data loading complete: {self.preprocessed_features.shape[0]} samples, "
                   f"{self.preprocessed_features.shape[1]} features")

        return self

    def train_model(self,
                   min_k: int = 3,
                   max_k: int = 12,
                   epochs: int = 500,
                   batch_size: int = 512,
                   learning_rate: float = 1e-3,
                   hidden_size: int = 64) -> 'AdmixtureRunner':
        """
        Train Neural ADMIXTURE model using phenotypic adapter.

        Args:
            min_k: Minimum number of clusters
            max_k: Maximum number of clusters
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            hidden_size: Hidden layer size

        Returns:
            Self for method chaining
        """
        if self.preprocessed_features is None:
            raise ValueError("Data must be loaded and preprocessed before training")

        logger.info(f"Training Neural ADMIXTURE: K={min_k}-{max_k}, epochs={epochs}")
        logger.info(f"Data shape: {self.preprocessed_features.shape}, Device: {self.device}")

        # Training start time
        train_start = time.time()

        # Ensure data is in proper format for phenotypic training
        data_float = self.preprocessed_features.float()

        # Ensure data is in [0,1] range
        data_float = torch.clamp(data_float, 0.0, 1.0)

        # Move to device
        data_tensor = data_float.to(self.device)

        # Create phenotypic Neural ADMIXTURE trainer
        trainer = create_phenotypic_admixture_trainer(
            min_k=min_k,
            max_k=max_k,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            device=self.device,
            random_seed=self.random_seed
        )

        # Call phenotypic Neural ADMIXTURE training (NOT placeholder!)
        try:
            logger.info("Starting phenotypic Neural ADMIXTURE training...")
            Ps, Qs, trained_model = trainer.train_phenotypic_admixture(
                data_tensor,
                self.metadata
            )

            # Format results for analysis
            self.model_results = {}
            for i, k in enumerate(range(min_k, max_k + 1)):
                self.model_results[k] = {
                    'Q': Qs[i],  # Cluster assignments
                    'P': Ps[i],  # Cluster parameters
                    'log_likelihood': self._calculate_log_likelihood(data_tensor, Qs[i], Ps[i])
                }

            # Store the trained model
            self.trained_model = trained_model

            # Log training completion
            train_time = time.time() - train_start
            logger.info(f"Phenotypic Neural ADMIXTURE training completed in {train_time:.1f} seconds")

        except Exception as e:
            logger.error(f"Phenotypic Neural ADMIXTURE training failed: {str(e)}")
            raise

        return self

    def _calculate_log_likelihood(self, data: torch.Tensor, Q: np.ndarray, P: np.ndarray) -> float:
        """
        Calculate log-likelihood of the data given Q and P matrices.

        Args:
            data: Original data tensor (float32 in [0,1] range)
            Q: Cluster assignment probabilities (N x K)
            P: Cluster parameters (K x M)

        Returns:
            Log-likelihood value
        """
        try:
            # Ensure data is on CPU and in numpy format
            if isinstance(data, torch.Tensor):
                data_np = data.cpu().numpy()
            else:
                data_np = data

            # Ensure proper data types
            data_np = data_np.astype(np.float64)
            Q = Q.astype(np.float64)
            P = P.astype(np.float64)

            # Calculate log-likelihood: sum over samples of log(sum over clusters of Q * P^data * (1-P)^(1-data))
            # For continuous data, we approximate this as reconstruction error
            reconstruction = Q @ P  # (N x K) @ (K x M) = (N x M)
            mse = np.mean((data_np - reconstruction) ** 2)
            log_likelihood = -mse * data_np.shape[0] * data_np.shape[1]  # Scale by data size

            return float(log_likelihood)

        except Exception as e:
            logger.warning(f"Log-likelihood calculation failed: {e}, returning -inf")
            return float('-inf')

    def analyze_results(self) -> Dict:
        """Analyze clustering results and create visualizations."""
        if self.model_results is None:
            raise ValueError("Model must be trained before analysis")

        logger.info("Analyzing clustering results")

        analysis = {}

        # Model selection based on log-likelihood
        best_k = self._select_best_k()
        analysis['best_k'] = best_k
        analysis['best_results'] = self.model_results[best_k]

        # Cluster statistics
        analysis['cluster_stats'] = self._compute_cluster_statistics(best_k)

        # Gene associations
        analysis['gene_associations'] = self._analyze_gene_associations(best_k)

        # Create visualizations
        viz_files = self._create_visualizations(best_k)
        analysis['visualization_files'] = viz_files

        logger.info(f"Analysis complete: Best K={best_k}")
        return analysis

    def _select_best_k(self) -> int:
        """Select best K based on log-likelihood."""
        log_likelihoods = {k: results['log_likelihood']
                          for k, results in self.model_results.items()}

        best_k = max(log_likelihoods.keys(), key=lambda k: log_likelihoods[k])

        # Log model selection results
        logger.info("Model selection results:")
        for k in sorted(log_likelihoods.keys()):
            logger.info(f"  K={k}: log-likelihood = {log_likelihoods[k]:.2f}")

        return best_k

    def _compute_cluster_statistics(self, k: int) -> Dict:
        """Compute statistics for the best clustering."""
        Q = self.model_results[k]['Q']

        # Cluster assignments (hard assignment)
        cluster_assignments = np.argmax(Q, axis=1)

        # Cluster sizes
        cluster_sizes = np.bincount(cluster_assignments, minlength=k)

        # Assignment confidence (max probability)
        assignment_confidence = np.max(Q, axis=1)

        stats = {
            'cluster_sizes': cluster_sizes.tolist(),
            'cluster_proportions': (cluster_sizes / len(Q)).tolist(),
            'mean_confidence': float(np.mean(assignment_confidence)),
            'confidence_std': float(np.std(assignment_confidence)),
            'min_confidence': float(np.min(assignment_confidence)),
            'max_confidence': float(np.max(assignment_confidence))
        }

        return stats

    def _analyze_gene_associations(self, k: int) -> Dict:
        """Analyze gene-cluster associations."""
        if 'gene_symbols' not in self.metadata:
            return {}

        Q = self.model_results[k]['Q']
        genes = self.metadata['gene_symbols']
        cluster_assignments = np.argmax(Q, axis=1)

        # Count genes per cluster
        gene_cluster_map = {}
        for i, gene in enumerate(genes):
            if gene not in gene_cluster_map:
                gene_cluster_map[gene] = []
            gene_cluster_map[gene].append(cluster_assignments[i])

        # Analyze dominant cluster per gene
        gene_associations = {}
        for gene, clusters in gene_cluster_map.items():
            cluster_counts = np.bincount(clusters, minlength=k)
            dominant_cluster = np.argmax(cluster_counts)

            gene_associations[gene] = {
                'sample_count': len(clusters),
                'dominant_cluster': int(dominant_cluster),
                'cluster_distribution': cluster_counts.tolist(),
                'dominance_fraction': float(cluster_counts[dominant_cluster] / len(clusters))
            }

        return gene_associations

    def _create_visualizations(self, best_k: int) -> Dict[str, str]:
        """Create visualization plots."""
        viz_files = {}

        # 1. Cluster assignment probabilities heatmap
        fig1 = self._plot_assignment_heatmap(best_k)
        path1 = self.output_dir / "cluster_assignments.png"
        fig1.savefig(path1, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        viz_files['assignments'] = str(path1)

        # 2. Model selection plot
        fig2 = self._plot_model_selection()
        path2 = self.output_dir / "model_selection.png"
        fig2.savefig(path2, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        viz_files['model_selection'] = str(path2)

        # 3. Feature importance by cluster
        fig3 = self._plot_cluster_profiles(best_k)
        path3 = self.output_dir / "cluster_profiles.png"
        fig3.savefig(path3, dpi=300, bbox_inches='tight')
        plt.close(fig3)
        viz_files['profiles'] = str(path3)

        logger.info(f"Created {len(viz_files)} visualization files")
        return viz_files

    def _plot_assignment_heatmap(self, k: int) -> plt.Figure:
        """Plot cluster assignment probabilities as heatmap."""
        Q = self.model_results[k]['Q']

        # Sort by dominant cluster
        dominant_clusters = np.argmax(Q, axis=1)
        sort_idx = np.argsort(dominant_clusters)
        Q_sorted = Q[sort_idx]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(Q_sorted.T, aspect='auto', cmap='viridis')

        ax.set_xlabel('Samples (sorted by dominant cluster)')
        ax.set_ylabel('Clusters')
        ax.set_title(f'Cluster Assignment Probabilities (K={k})')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Assignment Probability')

        return fig

    def _plot_model_selection(self) -> plt.Figure:
        """Plot model selection results."""
        k_values = sorted(self.model_results.keys())
        log_likelihoods = [self.model_results[k]['log_likelihood'] for k in k_values]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_values, log_likelihoods, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (K)')
        ax.set_ylabel('Log-Likelihood')
        ax.set_title('Model Selection: Log-Likelihood vs K')
        ax.grid(True, alpha=0.3)

        # Mark best K
        best_k = max(k_values, key=lambda k: self.model_results[k]['log_likelihood'])
        best_ll = self.model_results[best_k]['log_likelihood']
        ax.axvline(best_k, color='red', linestyle='--', alpha=0.7)
        ax.text(best_k, best_ll, f'Best K={best_k}',
               verticalalignment='bottom', horizontalalignment='center')

        return fig

    def _plot_cluster_profiles(self, k: int) -> plt.Figure:
        """Plot mean feature profiles for each cluster."""
        P = self.model_results[k]['P']  # Cluster parameters
        feature_names = self.metadata['feature_names']

        # Focus on ABR features for interpretability
        abr_indices = self.metadata['feature_groups']['abr_features']
        abr_names = [feature_names[i] for i in abr_indices]
        abr_profiles = P[:, abr_indices]

        fig, ax = plt.subplots(figsize=(12, 8))

        for cluster_id in range(k):
            ax.plot(abr_names, abr_profiles[cluster_id],
                   marker='o', linewidth=2, label=f'Cluster {cluster_id}')

        ax.set_xlabel('ABR Frequencies')
        ax.set_ylabel('Mean Threshold (normalized)')
        ax.set_title(f'Audiometric Profiles by Cluster (K={k})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels for readability
        plt.xticks(rotation=45)

        return fig

    def save_results(self, analysis: Dict) -> Dict[str, str]:
        """Save all results to files."""
        saved_files = {}

        # Save analysis results as JSON
        analysis_file = self.output_dir / "analysis_results.json"
        with open(analysis_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_safe_analysis = self._make_json_serializable(analysis)
            json.dump(json_safe_analysis, f, indent=2)
        saved_files['analysis'] = str(analysis_file)

        # Save cluster assignments
        best_k = analysis['best_k']
        Q = self.model_results[best_k]['Q']
        assignments_df = pd.DataFrame({
            'sample_id': range(len(Q)),
            'gene_symbol': self.metadata['gene_symbols'],
            'dominant_cluster': np.argmax(Q, axis=1),
            'max_probability': np.max(Q, axis=1)
        })

        # Add individual cluster probabilities
        for i in range(best_k):
            assignments_df[f'cluster_{i}_prob'] = Q[:, i]

        assignments_file = self.output_dir / "cluster_assignments.csv"
        assignments_df.to_csv(assignments_file, index=False)
        saved_files['assignments'] = str(assignments_file)

        # Save preprocessing info
        preproc_file = self.output_dir / "preprocessing_summary.json"
        with open(preproc_file, 'w') as f:
            json.dump(self.preprocessor.summary(), f, indent=2)
        saved_files['preprocessing'] = str(preproc_file)

        # Save trained model
        if hasattr(self, 'trained_model') and self.trained_model is not None:
            model_file = self.output_dir / "trained_model.pt"
            torch.save(self.trained_model.state_dict(), model_file)
            saved_files['trained_model'] = str(model_file)

            # Save model configuration
            config_file = self.output_dir / "model_config.json"
            model_config = {
                'best_k': best_k,
                'n_features': self.preprocessed_features.shape[1],
                'hidden_size': getattr(self.trained_model, 'hidden_size', 64),
                'device': str(self.device),
                'feature_names': self.metadata['feature_names']
            }

            with open(config_file, 'w') as f:
                json.dump(model_config, f, indent=2)
            saved_files['model_config'] = str(config_file)

        # Add visualization files
        saved_files.update(analysis['visualization_files'])

        logger.info(f"Results saved: {len(saved_files)} files")
        return saved_files

    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other objects for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        else:
            return obj

    def run_complete_pipeline(self,
                            data_path: str,
                            min_samples_per_gene: int = 3,
                            min_k: int = 3,
                            max_k: int = 12,
                            epochs: int = 500,
                            batch_size: int = 512,
                            learning_rate: float = 1e-3) -> Dict[str, str]:
        """
        Run complete pipeline from data loading to results.

        Args:
            data_path: Path to data file
            min_samples_per_gene: Minimum samples per gene
            min_k: Minimum clusters
            max_k: Maximum clusters
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Dictionary of saved file paths
        """
        start_time = time.time()

        logger.info("Starting complete Neural ADMIXTURE pipeline")

        try:
            # Execute pipeline
            (self.load_and_preprocess(data_path, min_samples_per_gene)
             .train_model(min_k, max_k, epochs, batch_size, learning_rate))

            # Analyze results
            analysis = self.analyze_results()

            # Save everything
            saved_files = self.save_results(analysis)

            elapsed_time = time.time() - start_time
            logger.info(f"Pipeline completed successfully in {elapsed_time:.1f} seconds")

            return saved_files

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Command line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Neural ADMIXTURE Phenotypic Clustering")
    parser.add_argument('data_path', help='Path to IMPC data file')
    parser.add_argument('--output-dir', '-o', default='admixture_results',
                       help='Output directory')
    parser.add_argument('--min-k', type=int, default=3, help='Minimum clusters')
    parser.add_argument('--max-k', type=int, default=12, help='Maximum clusters')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Initialize and run pipeline
    runner = AdmixtureRunner(
        output_dir=args.output_dir,
        device=args.device,
        random_seed=args.seed
    )

    saved_files = runner.run_complete_pipeline(
        data_path=args.data_path,
        min_k=args.min_k,
        max_k=args.max_k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    print(f"\nPipeline completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Files created: {len(saved_files)}")
    for file_type, path in saved_files.items():
        print(f"  {file_type}: {path}")


if __name__ == "__main__":
    main()