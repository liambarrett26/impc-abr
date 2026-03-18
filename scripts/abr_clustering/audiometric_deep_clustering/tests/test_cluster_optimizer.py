"""
Unit tests for cluster optimization components.

Tests cluster number optimization, metrics computation, and integration with training pipeline.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.cluster_optimizer import ClusterOptimizer


class MockModel(nn.Module):
    """Mock model for testing that returns predictable latent features."""
    
    def __init__(self, latent_dim=10, n_clusters=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        
    def forward(self, x):
        """Return synthetic latent features with known cluster structure."""
        batch_size = x.size(0)
        
        # Create latent features with clear cluster structure
        # Each cluster centered at different locations
        cluster_centers = torch.tensor([
            [2.0, 2.0] + [0.0] * (self.latent_dim - 2),  # Cluster 1
            [-2.0, 2.0] + [0.0] * (self.latent_dim - 2), # Cluster 2
            [0.0, -2.0] + [0.0] * (self.latent_dim - 2)  # Cluster 3
        ], dtype=torch.float32)
        
        # Assign samples to clusters cyclically and add noise
        latent_z = torch.zeros(batch_size, self.latent_dim)
        for i in range(batch_size):
            cluster_id = i % self.n_clusters
            latent_z[i] = cluster_centers[cluster_id] + 0.3 * torch.randn(self.latent_dim)
        
        return {'latent_z': latent_z}
    
    def eval(self):
        """Mock eval method."""
        return self


class MockDataLoader:
    """Mock dataloader for testing."""
    
    def __init__(self, n_batches=5, batch_size=20, n_features=18):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.n_features = n_features
        
    def __iter__(self):
        for i in range(self.n_batches):
            batch = {
                'features': torch.randn(self.batch_size, self.n_features)
            }
            yield batch
    
    def __len__(self):
        return self.n_batches


class TestClusterOptimizer(unittest.TestCase):
    """Test cluster optimization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'k_range': [2, 5],
            'metrics': ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'elbow'],
            'consensus_method': 'median',
            'save_analysis': False
        }
        self.optimizer = ClusterOptimizer(self.config)
        
        # Create synthetic data with known cluster structure
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create 3 clear clusters in 2D space for easy testing
        cluster1 = np.random.normal([2, 2], 0.3, (30, 2))
        cluster2 = np.random.normal([-2, 2], 0.3, (30, 2))
        cluster3 = np.random.normal([0, -2], 0.3, (30, 2))
        
        # Pad with zeros to make it 10-dimensional
        self.test_features = np.column_stack([
            np.vstack([cluster1, cluster2, cluster3]),
            np.zeros((90, 8))  # Pad to 10 dimensions
        ])
        
        self.device = torch.device('cpu')
    
    def test_initialization(self):
        """Test ClusterOptimizer initialization."""
        self.assertEqual(self.optimizer.k_range, (2, 5))
        self.assertEqual(self.optimizer.consensus_method, 'median')
        self.assertFalse(self.optimizer.save_analysis)
        self.assertIn('silhouette', self.optimizer.metrics)
    
    def test_find_elbow_point(self):
        """Test elbow point finding algorithm."""
        # Create synthetic inertia values with clear elbow at index 2
        inertia_values = np.array([100, 60, 40, 38, 37, 36])
        
        elbow_idx = self.optimizer.find_elbow_point(inertia_values)
        
        # Should find elbow around index 2 (where decrease slows down)
        self.assertIsInstance(elbow_idx, (int, np.integer))
        self.assertGreaterEqual(elbow_idx, 0)
        self.assertLess(elbow_idx, len(inertia_values))
    
    def test_compute_clustering_metrics(self):
        """Test clustering metrics computation."""
        metrics_df = self.optimizer.compute_clustering_metrics(self.test_features)
        
        # Check DataFrame structure
        expected_columns = ['k', 'inertia', 'silhouette', 'davies_bouldin', 'calinski_harabasz']
        for col in expected_columns:
            self.assertIn(col, metrics_df.columns)
        
        # Check k range
        expected_k_values = list(range(2, 6))  # 2, 3, 4, 5
        self.assertListEqual(metrics_df['k'].tolist(), expected_k_values)
        
        # Check that metrics are reasonable
        for _, row in metrics_df.iterrows():
            if row['k'] > 1:
                self.assertGreater(row['silhouette'], -1)
                self.assertLess(row['silhouette'], 1)
                self.assertGreater(row['davies_bouldin'], 0)
                self.assertGreater(row['calinski_harabasz'], 0)
            self.assertGreater(row['inertia'], 0)
        
        # For our synthetic data with 3 clear clusters, k=3 should be optimal
        # Silhouette score should be highest at k=3
        k3_silhouette = metrics_df[metrics_df['k'] == 3]['silhouette'].iloc[0]
        k2_silhouette = metrics_df[metrics_df['k'] == 2]['silhouette'].iloc[0]
        k4_silhouette = metrics_df[metrics_df['k'] == 4]['silhouette'].iloc[0]
        
        # k=3 should have better (higher) silhouette than k=2 or k=4
        self.assertGreater(k3_silhouette, k2_silhouette)
        self.assertGreater(k3_silhouette, k4_silhouette)
    
    def test_get_optimal_k_per_metric(self):
        """Test optimal k selection for each metric."""
        # Create mock metrics DataFrame
        metrics_df = pd.DataFrame({
            'k': [2, 3, 4, 5],
            'inertia': [100, 60, 45, 42],  # Elbow at k=3
            'silhouette': [0.3, 0.8, 0.5, 0.4],  # Best at k=3
            'davies_bouldin': [2.0, 1.0, 1.5, 2.2],  # Best at k=3 (lowest)
            'calinski_harabasz': [10, 25, 20, 15]  # Best at k=3 (highest)
        })
        
        optimal_ks = self.optimizer.get_optimal_k_per_metric(metrics_df)
        
        # All metrics should point to k=3 for this synthetic data
        self.assertIn('silhouette', optimal_ks)
        self.assertIn('davies_bouldin', optimal_ks)
        self.assertIn('calinski_harabasz', optimal_ks)
        self.assertIn('elbow', optimal_ks)
        
        self.assertEqual(optimal_ks['silhouette'], 3)
        self.assertEqual(optimal_ks['davies_bouldin'], 3)
        self.assertEqual(optimal_ks['calinski_harabasz'], 3)
    
    def test_get_consensus_k(self):
        """Test consensus k calculation."""
        optimal_ks = {'metric1': 3, 'metric2': 4, 'metric3': 3, 'metric4': 5}
        
        # Test median consensus
        self.optimizer.consensus_method = 'median'
        consensus_k = self.optimizer.get_consensus_k(optimal_ks)
        self.assertEqual(consensus_k, 3)  # median of [3, 4, 3, 5] = 3.5 -> 3
        
        # Test mean consensus
        self.optimizer.consensus_method = 'mean'
        consensus_k = self.optimizer.get_consensus_k(optimal_ks)
        expected_mean = int(round(np.mean([3, 4, 3, 5])))
        self.assertEqual(consensus_k, expected_mean)
    
    def test_extract_latent_features(self):
        """Test latent feature extraction from model."""
        model = MockModel(latent_dim=10, n_clusters=3)
        dataloader = MockDataLoader(n_batches=3, batch_size=10)
        
        latent_features = self.optimizer.extract_latent_features(model, dataloader, self.device)
        
        # Check output shape
        expected_samples = 3 * 10  # 3 batches * 10 samples
        self.assertEqual(latent_features.shape, (expected_samples, 10))
        
        # Check that features are numpy array
        self.assertIsInstance(latent_features, np.ndarray)
        
        # Check that we get the expected cluster structure
        # The mock model creates 3 distinct clusters
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(latent_features)
        
        # Should find 3 distinct clusters
        unique_labels = np.unique(labels)
        self.assertEqual(len(unique_labels), 3)
    
    def test_plot_analysis(self):
        """Test plotting functionality."""
        # Create mock metrics DataFrame
        metrics_df = pd.DataFrame({
            'k': [2, 3, 4],
            'inertia': [100, 60, 45],
            'silhouette': [0.3, 0.8, 0.5],
            'davies_bouldin': [2.0, 1.0, 1.5],
            'calinski_harabasz': [10, 25, 20]
        })
        
        optimal_ks = {'silhouette': 3, 'davies_bouldin': 3, 'calinski_harabasz': 3, 'elbow': 3}
        consensus_k = 3
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / 'test_plot.png'
            
            # Test plotting without errors
            fig = self.optimizer.plot_analysis(metrics_df, optimal_ks, consensus_k, save_path)
            
            self.assertIsNotNone(fig)
            self.assertTrue(save_path.exists())
            
            # Clean up
            plt.close(fig)
    
    @patch('training.cluster_optimizer.logger')
    def test_optimize_end_to_end(self, mock_logger):
        """Test end-to-end optimization process."""
        model = MockModel(latent_dim=10, n_clusters=3)
        dataloader = MockDataLoader(n_batches=5, batch_size=20)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)
            
            # Run optimization
            optimal_k = self.optimizer.optimize(model, dataloader, self.device, save_dir)
            
            # Check that we get a reasonable result
            self.assertIsInstance(optimal_k, int)
            self.assertGreaterEqual(optimal_k, self.config['k_range'][0])
            self.assertLessEqual(optimal_k, self.config['k_range'][1])
            
            # For our synthetic data with 3 clear clusters, should find k=3
            self.assertEqual(optimal_k, 3)
            
            # Check that logger was called
            mock_logger.info.assert_called()
    
    def test_config_validation(self):
        """Test configuration validation and defaults."""
        # Test with minimal config
        minimal_config = {}
        optimizer = ClusterOptimizer(minimal_config)
        
        # Should use defaults
        self.assertEqual(optimizer.k_range, (6, 18))
        self.assertEqual(optimizer.consensus_method, 'median')
        self.assertTrue(optimizer.save_analysis)
        
        # Test with custom config
        custom_config = {
            'k_range': [3, 8],
            'metrics': ['silhouette'],
            'consensus_method': 'mean',
            'save_analysis': False
        }
        optimizer = ClusterOptimizer(custom_config)
        
        self.assertEqual(optimizer.k_range, (3, 8))
        self.assertEqual(optimizer.metrics, ['silhouette'])
        self.assertEqual(optimizer.consensus_method, 'mean')
        self.assertFalse(optimizer.save_analysis)


class TestClusterOptimizerIntegration(unittest.TestCase):
    """Test cluster optimizer integration with training pipeline."""
    
    def test_config_structure(self):
        """Test that cluster optimization config structure is correct."""
        # This would typically be in a training config
        config = {
            'enabled': True,
            'k_range': [6, 18],
            'metrics': ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'elbow'],
            'consensus_method': 'median',
            'save_analysis': True
        }
        
        optimizer = ClusterOptimizer(config)
        
        # Verify configuration is properly set
        self.assertTrue(config['enabled'])
        self.assertEqual(optimizer.k_range, (6, 18))
        self.assertEqual(len(optimizer.metrics), 4)
    
    def test_synthetic_data_optimization(self):
        """Test optimization with synthetic data that has known optimal k."""
        # Create data with 4 clear clusters
        np.random.seed(42)
        clusters = []
        cluster_centers = [(0, 0), (5, 0), (0, 5), (5, 5)]
        
        for center in cluster_centers:
            cluster_data = np.random.normal(center, 0.5, (25, 2))
            # Pad to higher dimension
            padded_data = np.column_stack([cluster_data, np.zeros((25, 8))])
            clusters.append(padded_data)
        
        synthetic_features = np.vstack(clusters)
        
        config = {
            'k_range': [2, 6],
            'metrics': ['silhouette', 'davies_bouldin', 'calinski_harabasz'],
            'consensus_method': 'median',
            'save_analysis': False
        }
        
        optimizer = ClusterOptimizer(config)
        metrics_df = optimizer.compute_clustering_metrics(synthetic_features)
        optimal_ks = optimizer.get_optimal_k_per_metric(metrics_df)
        consensus_k = optimizer.get_consensus_k(optimal_ks)
        
        # For 4 clear clusters, should find k=4
        self.assertEqual(consensus_k, 4)


if __name__ == '__main__':
    unittest.main()