"""
Phenotypic Neural ADMIXTURE Adapter

This module serves as the main entry point for adapting Neural ADMIXTURE to work with
phenotypic data instead of genomic data. It replaces the genomic-specific components
while maintaining the elegant multi-head architecture.
"""

import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

# Import from existing Neural ADMIXTURE
from neural_admixture_original.model.neural_admixture import NeuralAdmixture
from neural_admixture_original.model.train import train

# Import our new phenotypic components
from integrated_preproc_pipeline import IntegratedPhenotypicPipeline, create_neural_admixture_dataset
from phenotypic_data_loader import enhanced_dataloader_phenotypic, EnhancedPhenotypicDataProcessor
from feature_engineering import create_default_feature_config

logger = logging.getLogger(__name__)


class PhenotypicNeuralAdmixture:
    """
    Phenotypic adaptation of Neural ADMIXTURE for audiometric phenotype discovery.

    This class maintains the multi-head architecture and training approach of Neural ADMIXTURE
    while adapting it for phenotypic clustering instead of genomic ancestry inference.
    """

    def __init__(self,
                 min_k: int = 3,
                 max_k: int = 12,
                 epochs: int = 500,
                 batch_size: int = 512,
                 learning_rate: float = 1e-3,
                 hidden_size: int = 32,
                 device: torch.device = None,
                 seed: int = 42):
        """
        Initialize Phenotypic Neural ADMIXTURE.

        Args:
            min_k: Minimum number of clusters
            max_k: Maximum number of clusters
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            hidden_size: Hidden layer size (reduced from genomic default)
            device: Computing device
            seed: Random seed
        """
        self.min_k = min_k
        self.max_k = max_k
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.seed = seed

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        # Initialize components
        self.pipeline = IntegratedPhenotypicPipeline(device=self.device)
        self.is_fitted = False
        self.results = {}

        logger.info(f"Initialized Phenotypic Neural ADMIXTURE for K={min_k}-{max_k}")
        logger.info(f"Device: {self.device}")

    def fit(self, data_path: str,
            save_preprocessing: bool = True,
            save_dir: str = "phenotypic_admixture_outputs") -> 'PhenotypicNeuralAdmixture':
        """
        Fit the model on IMPC data.

        Args:
            data_path: Path to IMPC ABR data
            save_preprocessing: Whether to save preprocessing state
            save_dir: Directory for outputs

        Returns:
            Self for method chaining
        """
        logger.info("Starting Phenotypic Neural ADMIXTURE training...")

        # Create output directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Step 1: Prepare data
        logger.info("Step 1: Data preparation and preprocessing...")
        features, gene_labels, metadata = create_neural_admixture_dataset(
            data_path,
            device=self.device,
            save_preprocessing=save_preprocessing,
            save_dir=save_dir
        )

        # Step 2: Initialize Neural ADMIXTURE model for phenotypic data
        logger.info("Step 2: Initializing Phenotypic Neural ADMIXTURE...")
        model = self._create_phenotypic_model(features, metadata)

        # Step 3: Train model
        logger.info("Step 3: Training multi-head model...")
        Qs, Ps, trained_model = self._train_phenotypic_model(
            model, features, gene_labels, metadata
        )

        # Step 4: Process and store results
        logger.info("Step 4: Processing results...")
        self.results = self._process_results(Qs, Ps, trained_model, metadata)

        # Step 5: Save results
        if save_preprocessing:
            self._save_results(save_dir)

        self.is_fitted = True
        logger.info("Phenotypic Neural ADMIXTURE training complete!")

        return self

    def _create_phenotypic_model(self, features: torch.Tensor, metadata: Dict[str, Any]) -> NeuralAdmixture:
        """Create Neural ADMIXTURE model adapted for phenotypic data."""
        n_features = features.shape[1]

        # Create model with phenotypic parameters
        model = NeuralAdmixture(
            k=None,  # Use range instead
            min_k=self.min_k,
            max_k=self.max_k,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            device=self.device,
            seed=self.seed,
            num_gpus=1 if self.device.type == 'cuda' else 0,
            master=True,
            pack2bit=None  # No bit packing for phenotypic data
        )

        return model

    def _train_phenotypic_model(self,
                               model: NeuralAdmixture,
                               features: torch.Tensor,
                               gene_labels: Optional[torch.Tensor],
                               metadata: Dict[str, Any]) -> Tuple[List[np.ndarray], List[np.ndarray], Any]:
        """Train the phenotypic model."""

        # Initialize with phenotypic GMM instead of genomic PCA+GMM
        P_init, V_matrix = self._initialize_phenotypic_params(features, metadata)

        # Adapt the training call for phenotypic data
        Qs, Ps, trained_model = self._adapted_train(
            model, features, P_init, V_matrix, gene_labels
        )

        return Qs, Ps, trained_model

    def _initialize_phenotypic_params(self, features: torch.Tensor, metadata: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize parameters for phenotypic data using GMM instead of PCA.

        Since we don't have thousands of SNPs requiring PCA dimensionality reduction,
        we can work directly with the engineered features.
        """
        from sklearn.mixture import GaussianMixture

        features_np = features.cpu().numpy()
        n_features = features_np.shape[1]

        # Initialize P matrix using GMM for each K
        P_matrices = []

        for k in range(self.min_k, self.max_k + 1):
            logger.info(f"Initializing GMM for K={k}")

            gmm = GaussianMixture(
                n_components=k,
                n_init=5,
                init_params='k-means++',
                covariance_type='full',
                max_iter=100,
                random_state=self.seed
            )

            gmm.fit(features_np)

            # For phenotypic data, the "allele frequencies" are actually
            # the characteristic feature values for each cluster
            P_k = np.clip(gmm.means_, 0.0, 1.0)  # Ensure [0,1] range
            P_matrices.append(P_k)

        # Concatenate all P matrices
        P_combined = np.vstack(P_matrices)
        P_init = torch.as_tensor(P_combined, dtype=torch.float32, device=self.device)

        # Create identity matrix as "V" (no PCA needed for phenotypic data)
        V_matrix = torch.eye(n_features, dtype=torch.float32, device=self.device)

        logger.info(f"Initialized P matrix: {P_init.shape}")

        return P_init, V_matrix

    def _adapted_train(self,
                      model: NeuralAdmixture,
                      features: torch.Tensor,
                      P_init: torch.Tensor,
                      V_matrix: torch.Tensor,
                      gene_labels: Optional[torch.Tensor]) -> Tuple[List[np.ndarray], List[np.ndarray], Any]:
        """
        Adapted training function for phenotypic data.

        This replaces the genomic train() function with phenotypic data handling.
        """
        N, M = features.shape

        # Convert features to format expected by Neural ADMIXTURE
        # Note: Neural ADMIXTURE expects values in {0, 0.5, 1} range for genomic data
        # For phenotypic data, we keep the [0,1] range from preprocessing
        data_for_training = features.cpu()

        # Create phenotypic dataloader
        dataloader = dataloader_phenotypic(
            features=data_for_training,
            batch_size=self.batch_size,
            num_gpus=1 if self.device.type == 'cuda' else 0,
            seed=self.seed,
            generator=torch.Generator().manual_seed(self.seed),
            gene_labels=gene_labels,
            shuffle=True
        )

        # Initialize model with phenotypic parameters
        model.initialize_model(
            P=P_init,
            hidden_size=self.hidden_size,
            num_features=M,
            V=V_matrix,
            ks_list=list(range(self.min_k, self.max_k + 1))
        )

        # Training loop adapted for phenotypic data
        model.raw_model.train()
        optimizer = model.raw_model.create_custom_adam(device=self.device, lr=self.learning_rate)

        logger.info("Starting phenotypic training loop...")

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_features, batch_labels in dataloader:
                batch_features = batch_features.to(self.device)

                # Forward pass
                optimizer.zero_grad()

                # Adapt input for Neural ADMIXTURE forward pass
                # The model expects the input preprocessing done in Q_P.forward()
                recs, _ = model.model(batch_features * 2)  # Scale to [0,2] range expected by model

                # Calculate loss across all heads
                target = batch_features.to(self.device)
                loss = sum(torch.nn.functional.mse_loss(rec, target) for rec in recs[0])

                # Backward pass
                loss.backward()
                optimizer.step()

                # Restrict P matrix values to [0,1]
                model.raw_model.restrict_P()

                epoch_loss += loss.item()
                n_batches += 1

            if epoch % 50 == 0:
                avg_loss = epoch_loss / n_batches
                logger.info(f"Epoch {epoch}: Average loss = {avg_loss:.4f}")

        # Inference to get Q matrices
        logger.info("Running inference to get cluster assignments...")
        model.raw_model.eval()

        Qs = [torch.tensor([], device=self.device) for _ in range(len(model.ks_list))]

        with torch.no_grad():
            inference_dataloader = dataloader_phenotypic(
                features=data_for_training,
                batch_size=min(5000, N),
                num_gpus=0,
                seed=self.seed,
                generator=torch.Generator().manual_seed(self.seed),
                gene_labels=gene_labels,
                shuffle=False
            )

            for batch_features, _ in inference_dataloader:
                batch_features = batch_features.to(self.device)
                probs, _ = model.model(batch_features * 2)

                for i in range(len(model.ks_list)):
                    Qs[i] = torch.cat((Qs[i], probs[i]), dim=0)

        # Extract P matrices (decoder weights)
        Ps = [decoder.weight.data.detach().cpu().numpy()
              for decoder in model.raw_model.decoders.decoders]

        # Convert Q matrices to numpy
        Qs = [Q.cpu().numpy() for Q in Qs]

        return Qs, Ps, model.raw_model

    def _process_results(self,
                        Qs: List[np.ndarray],
                        Ps: List[np.ndarray],
                        model: Any,
                        metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process and organize results."""

        results = {
            'cluster_assignments': {},  # Q matrices by K
            'cluster_centers': {},      # P matrices by K
            'model_parameters': {},
            'metadata': metadata,
            'k_range': list(range(self.min_k, self.max_k + 1))
        }

        # Store results for each K
        for i, k in enumerate(range(self.min_k, self.max_k + 1)):
            results['cluster_assignments'][k] = Qs[i]
            results['cluster_centers'][k] = Ps[i]

        # Store model parameters
        results['model_parameters'] = {
            'hidden_size': self.hidden_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'seed': self.seed
        }

        return results

    def _save_results(self, save_dir: str):
        """Save all results to files."""
        save_path = Path(save_dir)

        # Save cluster assignments and centers
        for k in range(self.min_k, self.max_k + 1):
            np.save(save_path / f"cluster_assignments_k{k}.npy", self.results['cluster_assignments'][k])
            np.save(save_path / f"cluster_centers_k{k}.npy", self.results['cluster_centers'][k])

        # Save metadata
        import json
        metadata_saveable = {k: v for k, v in self.results['metadata'].items()
                           if isinstance(v, (str, int, float, list, dict))}

        with open(save_path / "analysis_metadata.json", 'w') as f:
            json.dump(metadata_saveable, f, indent=2)

        logger.info(f"Results saved to {save_path}")

    def get_cluster_assignments(self, k: int) -> np.ndarray:
        """Get cluster assignments for specific K."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if k not in self.results['cluster_assignments']:
            raise ValueError(f"K={k} not in trained range {self.min_k}-{self.max_k}")

        return self.results['cluster_assignments'][k]

    def get_cluster_centers(self, k: int) -> np.ndarray:
        """Get cluster centers for specific K."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if k not in self.results['cluster_centers']:
            raise ValueError(f"K={k} not in trained range {self.min_k}-{self.max_k}")

        return self.results['cluster_centers'][k]

    def predict(self, data_path: str) -> Dict[int, np.ndarray]:
        """
        Predict cluster assignments for new data.

        Args:
            data_path: Path to new IMPC data

        Returns:
            Dictionary mapping K values to cluster assignments
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Load and preprocess new data
        df, _ = self.pipeline.load_and_preprocess(data_path)
        _, engineered_features = self.pipeline.transform(df)

        # Convert to tensor
        features = torch.as_tensor(engineered_features, dtype=torch.float32, device=self.device)

        # TODO: Implement inference with trained model
        # This would require storing the trained model state

        raise NotImplementedError("Prediction on new data not yet implemented")


def create_phenotypic_neural_admixture(data_path: str,
                                     min_k: int = 3,
                                     max_k: int = 12,
                                     epochs: int = 500,
                                     batch_size: int = 512,
                                     learning_rate: float = 1e-3,
                                     hidden_size: int = 32,
                                     device: torch.device = None,
                                     seed: int = 42,
                                     save_dir: str = "phenotypic_admixture_outputs") -> PhenotypicNeuralAdmixture:
    """
    Convenience function to create and train Phenotypic Neural ADMIXTURE.

    Args:
        data_path: Path to IMPC ABR data
        min_k: Minimum number of clusters
        max_k: Maximum number of clusters
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        hidden_size: Hidden layer size
        device: Computing device
        seed: Random seed
        save_dir: Output directory

    Returns:
        Fitted PhenotypicNeuralAdmixture instance
    """
    model = PhenotypicNeuralAdmixture(
        min_k=min_k,
        max_k=max_k,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        device=device,
        seed=seed
    )

    return model.fit(data_path, save_dir=save_dir)


def compare_with_gmm_baseline(phenotypic_model: PhenotypicNeuralAdmixture,
                             gmm_results: Dict[str, Any],
                             k_values: List[int] = None) -> Dict[str, Any]:
    """
    Compare Phenotypic Neural ADMIXTURE results with GMM baseline.

    Args:
        phenotypic_model: Fitted PhenotypicNeuralAdmixture instance
        gmm_results: Results from your existing GMM analysis
        k_values: K values to compare (default: all available)

    Returns:
        Comparison results dictionary
    """
    if not phenotypic_model.is_fitted:
        raise ValueError("Phenotypic model must be fitted first")

    if k_values is None:
        k_values = list(range(phenotypic_model.min_k, phenotypic_model.max_k + 1))

    comparison_results = {}

    for k in k_values:
        if k not in phenotypic_model.results['cluster_assignments']:
            logger.warning(f"K={k} not available in phenotypic model")
            continue

        neural_assignments = phenotypic_model.get_cluster_assignments(k)

        # Calculate comparison metrics
        result = {
            'k': k,
            'neural_admixture': {
                'n_samples': neural_assignments.shape[0],
                'n_clusters': neural_assignments.shape[1],
                'entropy': _calculate_assignment_entropy(neural_assignments),
                'max_assignment_confidence': np.mean(np.max(neural_assignments, axis=1))
            }
        }

        # Add GMM comparison if available
        if f'k_{k}' in gmm_results:
            gmm_assignments = gmm_results[f'k_{k}']['assignments']

            # Calculate adjusted rand index
            from sklearn.metrics import adjusted_rand_score
            neural_hard = np.argmax(neural_assignments, axis=1)
            gmm_hard = np.argmax(gmm_assignments, axis=1)
            ari = adjusted_rand_score(neural_hard, gmm_hard)

            result['comparison'] = {
                'adjusted_rand_index': ari,
                'assignment_correlation': np.corrcoef(
                    neural_assignments.flatten(),
                    gmm_assignments.flatten()
                )[0, 1]
            }

        comparison_results[k] = result

    return comparison_results


def _calculate_assignment_entropy(assignments: np.ndarray) -> float:
    """Calculate average entropy of cluster assignments."""
    entropy_sum = 0.0
    for i in range(assignments.shape[0]):
        probs = assignments[i]
        probs = probs + 1e-10  # Avoid log(0)
        entropy = -np.sum(probs * np.log(probs))
        entropy_sum += entropy

    return entropy_sum / assignments.shape[0]


def extract_gene_cluster_associations(phenotypic_model: PhenotypicNeuralAdmixture,
                                    gene_metadata: pd.DataFrame,
                                    k: int,
                                    confidence_threshold: float = 0.7) -> pd.DataFrame:
    """
    Extract gene-cluster associations from trained model.

    Args:
        phenotypic_model: Fitted model
        gene_metadata: DataFrame with gene information
        k: Number of clusters to analyze
        confidence_threshold: Minimum confidence for assignment

    Returns:
        DataFrame with gene-cluster associations
    """
    if not phenotypic_model.is_fitted:
        raise ValueError("Model must be fitted first")

    assignments = phenotypic_model.get_cluster_assignments(k)

    # Get high-confidence assignments
    max_probs = np.max(assignments, axis=1)
    confident_mask = max_probs >= confidence_threshold

    if 'gene_symbol' not in gene_metadata.columns:
        raise ValueError("gene_metadata must contain 'gene_symbol' column")

    # Create results dataframe
    results = []

    for i, (_, row) in enumerate(gene_metadata.iterrows()):
        if not confident_mask[i]:
            continue

        gene_symbol = row['gene_symbol']
        cluster_id = np.argmax(assignments[i])
        confidence = max_probs[i]

        results.append({
            'gene_symbol': gene_symbol,
            'cluster_id': cluster_id,
            'confidence': confidence,
            'sample_index': i
        })

    results_df = pd.DataFrame(results)

    # Add cluster statistics
    if not results_df.empty:
        cluster_stats = results_df.groupby('cluster_id').agg({
            'gene_symbol': 'count',
            'confidence': ['mean', 'std']
        }).round(3)

        logger.info(f"Gene-cluster associations for K={k}:")
        logger.info(f"High-confidence assignments: {len(results_df)}/{len(gene_metadata)} ({len(results_df)/len(gene_metadata)*100:.1f}%)")
        logger.info(f"Cluster statistics:\n{cluster_stats}")

    return results_df


def visualize_phenotypic_clusters(phenotypic_model: PhenotypicNeuralAdmixture,
                                k: int,
                                feature_names: List[str] = None,
                                save_path: str = None) -> None:
    """
    Create visualizations of phenotypic clusters.

    Args:
        phenotypic_model: Fitted model
        k: Number of clusters to visualize
        feature_names: Names of features for labeling
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not phenotypic_model.is_fitted:
        raise ValueError("Model must be fitted first")

    cluster_centers = phenotypic_model.get_cluster_centers(k)
    assignments = phenotypic_model.get_cluster_assignments(k)

    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Phenotypic Neural ADMIXTURE Results (K={k})', fontsize=16)

    # Plot 1: Cluster centers heatmap
    ax1 = axes[0, 0]
    sns.heatmap(cluster_centers, annot=True, fmt='.2f', cmap='viridis', ax=ax1)
    ax1.set_title('Cluster Centers (Feature Profiles)')
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Clusters')

    if feature_names and len(feature_names) == cluster_centers.shape[1]:
        ax1.set_xticklabels(feature_names, rotation=45, ha='right')

    # Plot 2: Assignment confidence distribution
    ax2 = axes[0, 1]
    max_probs = np.max(assignments, axis=1)
    ax2.hist(max_probs, bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(max_probs), color='red', linestyle='--',
               label=f'Mean: {np.mean(max_probs):.3f}')
    ax2.set_xlabel('Assignment Confidence')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Assignment Confidence Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cluster size distribution
    ax3 = axes[1, 0]
    cluster_assignments = np.argmax(assignments, axis=1)
    cluster_sizes = np.bincount(cluster_assignments)
    ax3.bar(range(k), cluster_sizes)
    ax3.set_xlabel('Cluster ID')
    ax3.set_ylabel('Number of Samples')
    ax3.set_title('Cluster Size Distribution')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Feature importance (variance across clusters)
    ax4 = axes[1, 1]
    feature_variance = np.var(cluster_centers, axis=0)
    feature_indices = range(len(feature_variance))
    ax4.bar(feature_indices, feature_variance)
    ax4.set_xlabel('Feature Index')
    ax4.set_ylabel('Variance Across Clusters')
    ax4.set_title('Feature Importance (Cluster Discrimination)')

    if feature_names and len(feature_names) == len(feature_variance):
        ax4.set_xticks(feature_indices[::max(1, len(feature_indices)//10)])
        ax4.set_xticklabels([feature_names[i] for i in feature_indices[::max(1, len(feature_indices)//10)]],
                           rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")

    plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the Phenotypic Neural ADMIXTURE

    # Basic usage
    data_path = "path/to/impc_data.csv"

    # Create and train model
    model = create_phenotypic_neural_admixture(
        data_path=data_path,
        min_k=3,
        max_k=8,
        epochs=200,
        batch_size=256,
        learning_rate=1e-3,
        save_dir="phenotypic_results"
    )

    # Get results for specific K
    k = 5
    assignments = model.get_cluster_assignments(k)
    centers = model.get_cluster_centers(k)

    print(f"Cluster assignments shape: {assignments.shape}")
    print(f"Cluster centers shape: {centers.shape}")

    # Visualize results
    visualize_phenotypic_clusters(
        model,
        k=k,
        save_path="phenotypic_clusters_k5.png"
    )