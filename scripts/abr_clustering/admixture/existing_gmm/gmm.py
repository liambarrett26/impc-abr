"""
Gaussian Mixture Model implementation for audiometric phenotype discovery.

Implements GMM clustering with the parameters optimized for ABR data analysis,
including model selection, stability validation, and uncertainty quantification.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class GMMConfig:
    """Configuration parameters for GMM clustering."""
    n_components_range: Tuple[int, int] = (3, 12)
    covariance_types: List[str] = field(default_factory=lambda: ['full', 'tied'])
    n_init: int = 10
    max_iter: int = 1000
    tol: float = 1e-6
    reg_covar: float = 1e-6
    random_state: int = 42
    n_bootstrap: int = 100
    bootstrap_fraction: float = 0.8


class ClusteringMetrics:
    """Container for clustering evaluation metrics."""

    def __init__(self):
        self.bic: Optional[float] = None
        self.aic: Optional[float] = None
        self.silhouette: Optional[float] = None
        self.log_likelihood: Optional[float] = None
        self.n_components: Optional[int] = None
        self.covariance_type: Optional[str] = None
        self.stability_score: Optional[float] = None

    def __repr__(self):
        return (f"ClusteringMetrics(n_components={self.n_components}, "
                f"bic={self.bic:.2f}, aic={self.aic:.2f}, "
                f"silhouette={self.silhouette:.3f})")


class BaseClusteringModel(ABC):
    """Abstract base class for clustering models."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseClusteringModel':
        """Fit the clustering model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignments."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignment probabilities."""
        pass

    @abstractmethod
    def get_metrics(self) -> ClusteringMetrics:
        """Get clustering evaluation metrics."""
        pass


class AudiometricGMM(BaseClusteringModel):
    """
    Gaussian Mixture Model optimized for audiometric phenotype discovery.

    Implements model selection, stability assessment, and biological validation
    for clustering ABR audiometric profiles.
    """

    def __init__(self, config: GMMConfig = None):
        """
        Initialize the GMM with configuration.

        Args:
            config: GMM configuration parameters
        """
        self.config = config if config is not None else GMMConfig()
        self.best_model: Optional[GaussianMixture] = None
        self.best_metrics: Optional[ClusteringMetrics] = None
        self.model_selection_results: Dict[str, Any] = {}
        self.stability_results: Dict[str, float] = {}
        self.is_fitted = False

    def _initialize_with_kmeans(self, X: np.ndarray, n_components: int) -> np.ndarray:
        """
        Initialize GMM parameters using K-means++ clustering.

        Args:
            X: Input data
            n_components: Number of components

        Returns:
            Initial cluster centers
        """
        kmeans = KMeans(n_clusters=n_components,
                       init='k-means++',
                       n_init=10,
                       random_state=self.config.random_state)
        kmeans.fit(X)
        return kmeans.cluster_centers_

    def _initialize_with_pca(self, X: np.ndarray, n_components: int) -> np.ndarray:
        """
        Initialize GMM using PCA-informed initialization.

        Args:
            X: Input data
            n_components: Number of components

        Returns:
            Initial cluster centers based on PCA
        """
        pca = PCA(n_components=min(2, X.shape[1]))
        X_pca = pca.fit_transform(X)

        # Create initial centers in PCA space, then transform back
        if n_components <= 4:
            # For small n_components, use corners of PCA space
            pc1_range = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(),
                                   int(np.ceil(np.sqrt(n_components))))
            if X_pca.shape[1] > 1:
                pc2_range = np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(),
                                       int(np.ceil(n_components / len(pc1_range))))
                centers_pca = np.array([[pc1, pc2] for pc1 in pc1_range for pc2 in pc2_range])
            else:
                centers_pca = np.array([[pc1, 0] for pc1 in pc1_range])

            centers_pca = centers_pca[:n_components]
        else:
            # For larger n_components, use random sampling in PCA space
            np.random.seed(self.config.random_state)
            centers_pca = np.random.uniform(
                low=[X_pca[:, 0].min(), X_pca[:, 1].min() if X_pca.shape[1] > 1 else 0],
                high=[X_pca[:, 0].max(), X_pca[:, 1].max() if X_pca.shape[1] > 1 else 0],
                size=(n_components, 2)
            )

        # Transform back to original space
        if X_pca.shape[1] == 1:
            centers_pca = np.column_stack([centers_pca[:, 0], np.zeros(n_components)])

        # Pad with zeros for dimensions beyond PC2
        if X.shape[1] > 2:
            padding = np.zeros((n_components, X.shape[1] - 2))
            centers_original = np.column_stack([centers_pca, padding])
        else:
            centers_original = centers_pca

        return pca.inverse_transform(centers_original[:, :pca.n_components_])

    def _fit_single_model(self, X: np.ndarray, n_components: int,
                         covariance_type: str) -> Tuple[GaussianMixture, ClusteringMetrics]:
        """
        Fit a single GMM with specified parameters.

        Args:
            X: Input data
            n_components: Number of mixture components
            covariance_type: Type of covariance matrix

        Returns:
            Tuple of (fitted_model, metrics)
        """
        best_model = None
        best_score = -np.inf

        # Try different initialization strategies
        init_strategies = ['random', 'kmeans', 'pca']

        for init_strategy in init_strategies:
            models = []
            scores = []

            for _ in range(self.config.n_init):
                try:
                    # Initialize model
                    model = GaussianMixture(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        max_iter=self.config.max_iter,
                        tol=self.config.tol,
                        reg_covar=self.config.reg_covar,
                        random_state=None,  # Use different random seeds for each init
                        warm_start=False
                    )

                    # Set initialization
                    if init_strategy == 'kmeans':
                        means_init = self._initialize_with_kmeans(X, n_components)
                        model.means_init = means_init
                    elif init_strategy == 'pca':
                        means_init = self._initialize_with_pca(X, n_components)
                        model.means_init = means_init

                    # Fit model
                    model.fit(X)

                    if model.converged_:
                        models.append(model)
                        scores.append(model.score(X))

                except Exception as e:
                    logger.debug(f"Model fitting failed with {init_strategy}: {e}")
                    continue

            # Select best model from this initialization strategy
            if scores and max(scores) > best_score:
                best_idx = np.argmax(scores)
                best_model = models[best_idx]
                best_score = scores[best_idx]

        if best_model is None:
            raise ValueError(f"Failed to fit GMM with {n_components} components")

        # Calculate metrics
        metrics = self._calculate_metrics(best_model, X)

        return best_model, metrics

    def _calculate_metrics(self, model: GaussianMixture, X: np.ndarray) -> ClusteringMetrics:
        """
        Calculate evaluation metrics for a fitted model.

        Args:
            model: Fitted GMM model
            X: Input data

        Returns:
            ClusteringMetrics object
        """
        metrics = ClusteringMetrics()

        try:
            metrics.n_components = model.n_components
            metrics.covariance_type = model.covariance_type
            metrics.bic = model.bic(X)
            metrics.aic = model.aic(X)
            metrics.log_likelihood = model.score(X)

            # Calculate silhouette score
            if model.n_components > 1:
                labels = model.predict(X)
                if len(np.unique(labels)) > 1:  # Ensure multiple clusters
                    metrics.silhouette = silhouette_score(X, labels)
                else:
                    metrics.silhouette = -1.0
            else:
                metrics.silhouette = 0.0

        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            metrics.silhouette = -1.0

        return metrics

    def _assess_stability(self, X: np.ndarray, n_components: int,
                         covariance_type: str) -> float:
        """
        Assess cluster stability using bootstrap resampling.

        Args:
            X: Input data
            n_components: Number of components
            covariance_type: Covariance type

        Returns:
            Average stability score across bootstrap samples
        """
        n_samples = int(self.config.bootstrap_fraction * len(X))
        stability_scores = []

        # Fit reference model on full data
        try:
            reference_model, _ = self._fit_single_model(X, n_components, covariance_type)
            reference_labels = reference_model.predict(X)
        except:
            return 0.0

        # Bootstrap resampling
        np.random.seed(self.config.random_state)
        for i in range(self.config.n_bootstrap):
            try:
                # Sample subset
                idx = np.random.choice(len(X), size=n_samples, replace=False)
                X_bootstrap = X[idx]

                # Fit model on bootstrap sample
                boot_model, _ = self._fit_single_model(X_bootstrap, n_components, covariance_type)

                # Predict on full dataset
                boot_labels = boot_model.predict(X)

                # Calculate stability using adjusted rand index
                stability = adjusted_rand_score(reference_labels, boot_labels)
                stability_scores.append(max(0, stability))  # Ensure non-negative

            except Exception as e:
                logger.debug(f"Bootstrap iteration {i} failed: {e}")
                continue

        return np.mean(stability_scores) if stability_scores else 0.0

    def _select_best_model(self, X: np.ndarray) -> Tuple[GaussianMixture, ClusteringMetrics]:
        """
        Perform model selection across different configurations.

        Args:
            X: Input data

        Returns:
            Tuple of (best_model, best_metrics)
        """
        logger.info("Performing GMM model selection")

        results = []
        n_min, n_max = self.config.n_components_range

        for n_components in range(n_min, n_max + 1):
            for covariance_type in self.config.covariance_types:
                logger.info(f"Testing {n_components} components with {covariance_type} covariance")

                try:
                    # Fit model
                    model, metrics = self._fit_single_model(X, n_components, covariance_type)

                    # Assess stability
                    stability = self._assess_stability(X, n_components, covariance_type)
                    metrics.stability_score = stability

                    results.append((model, metrics))

                    logger.info(f"  BIC: {metrics.bic:.2f}, AIC: {metrics.aic:.2f}, "
                              f"Silhouette: {metrics.silhouette:.3f}, Stability: {stability:.3f}")

                except Exception as e:
                    logger.warning(f"Failed to fit model with {n_components} components "
                                 f"and {covariance_type} covariance: {e}")
                    continue

        if not results:
            raise ValueError("No valid models found during model selection")

        # Store all results
        self.model_selection_results = {
            'models': [r[0] for r in results],
            'metrics': [r[1] for r in results],
            'bic_scores': [r[1].bic for r in results],
            'aic_scores': [r[1].aic for r in results],
            'silhouette_scores': [r[1].silhouette for r in results],
            'stability_scores': [r[1].stability_score for r in results]
        }

        # Select best model using combined criteria
        best_idx = self._select_best_by_criteria(results)

        return results[best_idx]

    def _select_best_by_criteria(self, results: List[Tuple[GaussianMixture, ClusteringMetrics]]) -> int:
        """
        Select best model using multiple criteria with weighted combination.

        Args:
            results: List of (model, metrics) tuples

        Returns:
            Index of best model
        """
        if len(results) == 1:
            return 0

        # Extract metrics
        bic_scores = np.array([r[1].bic for r in results])
        aic_scores = np.array([r[1].aic for r in results])
        silhouette_scores = np.array([r[1].silhouette for r in results])
        stability_scores = np.array([r[1].stability_score for r in results])

        # Normalize metrics (lower is better for BIC/AIC, higher for silhouette/stability)
        bic_norm = 1 - (bic_scores - bic_scores.min()) / (bic_scores.max() - bic_scores.min() + 1e-8)
        aic_norm = 1 - (aic_scores - aic_scores.min()) / (aic_scores.max() - aic_scores.min() + 1e-8)
        sil_norm = (silhouette_scores - silhouette_scores.min()) / (silhouette_scores.max() - silhouette_scores.min() + 1e-8)
        stab_norm = (stability_scores - stability_scores.min()) / (stability_scores.max() - stability_scores.min() + 1e-8)

        # Weighted combination (emphasizing BIC and stability)
        weights = np.array([0.4, 0.2, 0.2, 0.2])  # BIC, AIC, Silhouette, Stability
        combined_scores = (weights[0] * bic_norm +
                          weights[1] * aic_norm +
                          weights[2] * sil_norm +
                          weights[3] * stab_norm)

        return np.argmax(combined_scores)

    def fit(self, X: np.ndarray) -> 'AudiometricGMM':
        """
        Fit the GMM with model selection and validation.

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting AudiometricGMM on data with shape {X.shape}")

        if X.shape[0] < 10:
            raise ValueError("Insufficient data for clustering")

        if X.shape[1] != 5:
            logger.warning(f"Expected 5 features (ABR frequencies), got {X.shape[1]}")

        # Perform model selection
        self.best_model, self.best_metrics = self._select_best_model(X)

        logger.info(f"Best model: {self.best_metrics.n_components} components, "
                   f"{self.best_metrics.covariance_type} covariance")
        logger.info(f"Best model metrics: {self.best_metrics}")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for new data.

        Args:
            X: Input data

        Returns:
            Cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.best_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignment probabilities.

        Args:
            X: Input data

        Returns:
            Cluster assignment probabilities (n_samples, n_clusters)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.best_model.predict_proba(X)

    def get_metrics(self) -> ClusteringMetrics:
        """Get metrics for the best fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting metrics")

        return self.best_metrics

    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers (means) from the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting cluster centers")

        return self.best_model.means_

    def get_cluster_covariances(self) -> np.ndarray:
        """Get cluster covariance matrices from the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting covariances")

        return self.best_model.covariances_

    def score(self, X: np.ndarray) -> float:
        """Calculate log-likelihood of data under the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        return self.best_model.score(X)


def create_default_gmm_config(**kwargs) -> GMMConfig:
    """
    Create default GMM configuration with optional overrides.

    Args:
        **kwargs: Configuration parameters to override

    Returns:
        GMMConfig with specified parameters
    """
    config = GMMConfig()

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown configuration parameter: {key}")

    return config


if __name__ == "__main__":
    # Example usage and testing
    import sys
    from loader import load_impc_data
    from preproc import preprocess_abr_data

    if len(sys.argv) < 2:
        print("Usage: python gmm.py <data_path>")
        sys.exit(1)

    # Load and preprocess data
    data_path = sys.argv[1]
    df, _ = load_impc_data(data_path)

    # Take a subset for testing
    test_df = df.sample(n=min(1000, len(df)), random_state=42)
    normalized_data, _ = preprocess_abr_data(test_df)

    # Test GMM
    config = create_default_gmm_config(n_components_range=(2, 6), n_bootstrap=20)
    gmm = AudiometricGMM(config)
    gmm.fit(normalized_data)

    # Get results
    labels = gmm.predict(normalized_data)
    probabilities = gmm.predict_proba(normalized_data)
    metrics = gmm.get_metrics()

    print(f"Best model: {metrics.n_components} clusters")
    print(f"Cluster distribution: {np.bincount(labels)}")
    print(f"Metrics: {metrics}")
    print(f"Average prediction confidence: {probabilities.max(axis=1).mean():.3f}")