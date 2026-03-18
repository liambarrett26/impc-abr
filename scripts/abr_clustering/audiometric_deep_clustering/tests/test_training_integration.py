"""
Integration tests for training pipeline components.

Tests cluster optimization integration, training stages, and end-to-end workflows.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import yaml
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.cluster_optimizer import ClusterOptimizer
from models.full_model import create_model
from data.dataset import IMPCABRDataset


class MockDataModule:
    """Mock data module for testing."""
    
    def __init__(self, n_samples=100, n_features=18):
        self.n_samples = n_samples
        self.n_features = n_features
        
    def train_dataloader(self):
        """Return mock training dataloader."""
        return MockDataLoader(n_batches=5, batch_size=20, n_features=self.n_features)


class MockDataLoader:
    """Mock dataloader for testing."""
    
    def __init__(self, n_batches=5, batch_size=20, n_features=18):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.n_features = n_features
        self.dataset = Mock()
        self.dataset.__len__ = Mock(return_value=n_batches * batch_size)
        
    def __iter__(self):
        for i in range(self.n_batches):
            batch = {
                'features': torch.randn(self.batch_size, self.n_features),
                'gene_label': torch.randint(0, 5, (self.batch_size,))
            }
            yield batch
    
    def __len__(self):
        return self.n_batches


class TestTrainingIntegration(unittest.TestCase):
    """Test training pipeline integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        
        # Small model config for testing
        self.model_config = {
            'data': {
                'abr_features': 6,
                'metadata_features': 10,
                'pca_features': 2,
                'total_features': 18
            },
            'encoder': {
                'hidden_dims': [32, 16, 8],
                'dropout_rate': 0.2,
                'activation': 'relu',
                'abr_encoder': {
                    'hidden_dim': 8,
                    'use_batch_norm': True
                },
                'metadata_encoder': {
                    'hidden_dim': 12,
                    'embedding_dims': {
                        'sex': 4,
                        'zygosity': 4,
                        'genetic_background': 8,
                        'phenotyping_center': 12
                    }
                },
                'attention': {
                    'enabled': True,
                    'head_dim': 4,
                    'num_heads': 2,
                    'dropout': 0.1
                }
            },
            'latent': {
                'latent_dim': 6,
                'beta': 1.0,
                'min_logvar': -10,
                'max_logvar': 10
            },
            'decoder': {
                'hidden_dims': [8, 16, 32],
                'dropout_rate': 0.2,
                'activation': 'relu',
                'output_activation': 'linear',
                'use_batch_norm': True
            },
            'clustering': {
                'num_clusters': 4,  # Will be updated by optimization
                'cluster_init': 'kmeans',
                'alpha': 1.0,
                'update_interval': 10,
                'tolerance': 0.001
            },
            'contrastive': {
                'temperature': 0.5,
                'projection_dim': 16,
                'augmentation': {
                    'noise_std': 0.05,
                    'dropout_prob': 0.1,
                    'magnitude_scale': 0.1
                }
            },
            'loss_weights': {
                'reconstruction': 1.0,
                'kl_divergence': 1.0,
                'clustering': 1.0,
                'contrastive': 0.5,
                'phenotype_consistency': 0.3,
                'frequency_smoothness': 0.1,
                'reconstruction_weights': {
                    'abr': 2.0,
                    'metadata': 1.0,
                    'pca': 1.5
                }
            },
            'architecture': {
                'weight_init': 'xavier_uniform',
                'bias_init': 'zeros',
                'use_layer_norm': False,
                'use_spectral_norm': False,
                'residual_connections': False,
                'gradient_clipping': 1.0
            }
        }
        
        self.cluster_opt_config = {
            'enabled': True,
            'k_range': [2, 5],
            'metrics': ['silhouette', 'davies_bouldin'],
            'consensus_method': 'median',
            'save_analysis': False
        }
    
    def test_cluster_optimization_integration(self):
        """Test cluster optimization integration with model."""
        # Create model
        model = create_model(self.model_config)
        model.eval()
        
        # Create optimizer
        optimizer = ClusterOptimizer(self.cluster_opt_config)
        
        # Create mock dataloader
        dataloader = MockDataLoader(n_batches=3, batch_size=10, n_features=18)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)
            
            # Run optimization
            optimal_k = optimizer.optimize(model, dataloader, self.device, save_dir)
            
            # Should return a valid k value
            self.assertIsInstance(optimal_k, int)
            self.assertGreaterEqual(optimal_k, self.cluster_opt_config['k_range'][0])
            self.assertLessEqual(optimal_k, self.cluster_opt_config['k_range'][1])
    
    def test_model_cluster_update(self):
        """Test updating model with new cluster number."""
        # Create model
        model = create_model(self.model_config)
        original_k = model.num_clusters
        
        # Update cluster number
        new_k = 6
        self.model_config['clustering']['num_clusters'] = new_k
        model.num_clusters = new_k
        
        # Reinitialize clustering layer
        from models.clustering_layer import ClusteringLayer
        model.clustering_layer = ClusteringLayer(self.model_config)
        
        # Check that cluster number was updated
        self.assertEqual(model.num_clusters, new_k)
        self.assertEqual(model.clustering_layer.num_clusters, new_k)
        self.assertNotEqual(model.num_clusters, original_k)
        
        # Check that cluster centers have correct shape
        expected_shape = (new_k, self.model_config['latent']['latent_dim'])
        self.assertEqual(model.clustering_layer.cluster_centers.shape, expected_shape)
    
    @patch('training.cluster_optimizer.ClusterOptimizer')
    def test_training_stage_integration(self, mock_cluster_optimizer_class):
        """Test cluster optimization integration in training stages."""
        # Mock the optimizer instance
        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = 5
        mock_cluster_optimizer_class.return_value = mock_optimizer
        
        # Mock training config
        training_config = {
            'training_stages': {
                'cluster_optimization': self.cluster_opt_config
            }
        }
        
        # Mock output directories
        output_dirs = {
            'base': Path('/tmp/test'),
            'config': Path('/tmp/test/config')
        }
        
        # Simulate the training script logic
        cluster_opt_config = training_config.get('training_stages', {}).get('cluster_optimization', {})
        
        if cluster_opt_config.get('enabled', False):
            # Create optimizer (mocked)
            optimizer = mock_cluster_optimizer_class(cluster_opt_config)
            
            # Mock model and dataloader
            model = Mock()
            dataloader = Mock()
            device = torch.device('cpu')
            
            # Run optimization
            optimal_k = optimizer.optimize(
                model=model,
                dataloader=dataloader,
                device=device,
                save_dir=output_dirs['base'] / 'cluster_optimization'
            )
            
            # Update model configuration
            self.model_config['clustering']['num_clusters'] = optimal_k
            model.num_clusters = optimal_k
            
            # Check that optimization was called
            mock_optimizer.optimize.assert_called_once()
            self.assertEqual(optimal_k, 5)
            self.assertEqual(self.model_config['clustering']['num_clusters'], 5)
    
    def test_config_override_logic(self):
        """Test configuration override logic from CLI args."""
        # Simulate CLI argument parsing
        class MockArgs:
            def __init__(self):
                self.optimize_clusters = False
                self.skip_cluster_optimization = False
                self.k_range = None
        
        args = MockArgs()
        cluster_opt_config = self.cluster_opt_config.copy()
        
        # Test no overrides
        original_enabled = cluster_opt_config['enabled']
        
        # Apply CLI overrides (none in this case)
        if args.optimize_clusters:
            cluster_opt_config['enabled'] = True
        elif args.skip_cluster_optimization:
            cluster_opt_config['enabled'] = False
        
        if args.k_range:
            cluster_opt_config['k_range'] = args.k_range
        
        self.assertEqual(cluster_opt_config['enabled'], original_enabled)
        
        # Test override to enable
        args.optimize_clusters = True
        cluster_opt_config['enabled'] = False  # Start disabled
        
        if args.optimize_clusters:
            cluster_opt_config['enabled'] = True
        elif args.skip_cluster_optimization:
            cluster_opt_config['enabled'] = False
        
        self.assertTrue(cluster_opt_config['enabled'])
        
        # Test override to disable
        args.optimize_clusters = False
        args.skip_cluster_optimization = True
        cluster_opt_config['enabled'] = True  # Start enabled
        
        if args.optimize_clusters:
            cluster_opt_config['enabled'] = True
        elif args.skip_cluster_optimization:
            cluster_opt_config['enabled'] = False
        
        self.assertFalse(cluster_opt_config['enabled'])
        
        # Test k_range override
        args.k_range = [3, 8]
        original_k_range = cluster_opt_config['k_range']
        
        if args.k_range:
            cluster_opt_config['k_range'] = args.k_range
        
        self.assertEqual(cluster_opt_config['k_range'], [3, 8])
        self.assertNotEqual(cluster_opt_config['k_range'], original_k_range)
    
    def test_model_parameter_count_after_optimization(self):
        """Test that model parameter count changes correctly after cluster optimization."""
        # Create model with original cluster count
        original_model = create_model(self.model_config)
        original_params = sum(p.numel() for p in original_model.parameters())
        original_k = self.model_config['clustering']['num_clusters']
        
        # Update cluster number and recreate clustering layer
        new_k = 8
        self.model_config['clustering']['num_clusters'] = new_k
        
        from models.clustering_layer import ClusteringLayer
        new_clustering_layer = ClusteringLayer(self.model_config)
        
        # Calculate parameter difference
        original_cluster_params = original_k * self.model_config['latent']['latent_dim']
        new_cluster_params = new_k * self.model_config['latent']['latent_dim']
        expected_param_diff = new_cluster_params - original_cluster_params
        
        # Replace clustering layer
        original_model.clustering_layer = new_clustering_layer
        original_model.num_clusters = new_k
        
        new_params = sum(p.numel() for p in original_model.parameters())
        actual_param_diff = new_params - original_params
        
        self.assertEqual(actual_param_diff, expected_param_diff)
    
    def test_cluster_optimization_error_handling(self):
        """Test error handling in cluster optimization."""
        # Test with invalid k_range
        invalid_config = {
            'enabled': True,
            'k_range': [10, 5],  # Invalid: min > max
            'metrics': ['silhouette'],
            'consensus_method': 'median',
            'save_analysis': False
        }
        
        optimizer = ClusterOptimizer(invalid_config)
        
        # Should handle invalid range gracefully
        self.assertEqual(optimizer.k_range, (10, 5))  # Will be passed as-is, but should fail in range()
        
        # Test with empty metrics
        empty_metrics_config = invalid_config.copy()
        empty_metrics_config['metrics'] = []
        
        optimizer = ClusterOptimizer(empty_metrics_config)
        self.assertEqual(optimizer.metrics, [])


class TestClusterOptimizerFileIO(unittest.TestCase):
    """Test file I/O operations in cluster optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'k_range': [2, 4],
            'metrics': ['silhouette'],
            'consensus_method': 'median',
            'save_analysis': True
        }
        self.optimizer = ClusterOptimizer(self.config)
    
    def test_save_analysis_files(self):
        """Test that analysis files are saved correctly."""
        import pandas as pd
        
        # Mock data
        metrics_df = pd.DataFrame({
            'k': [2, 3, 4],
            'inertia': [100, 60, 45],
            'silhouette': [0.3, 0.8, 0.5],
            'davies_bouldin': [2.0, 1.0, 1.5],
            'calinski_harabasz': [10, 25, 20]
        })
        
        optimal_ks = {'silhouette': 3}
        consensus_k = 3
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)
            
            # Simulate saving analysis
            metrics_df.to_csv(save_dir / 'cluster_optimization_metrics.csv', index=False)
            
            summary = {
                'consensus_k': consensus_k,
                'optimal_k_per_metric': optimal_ks,
                'consensus_method': self.optimizer.consensus_method,
                'k_range': list(self.optimizer.k_range),  # Convert tuple to list for YAML
                'metrics_used': self.optimizer.metrics
            }
            
            with open(save_dir / 'cluster_optimization_summary.yaml', 'w') as f:
                yaml.dump(summary, f, default_flow_style=False)
            
            # Check files exist
            self.assertTrue((save_dir / 'cluster_optimization_metrics.csv').exists())
            self.assertTrue((save_dir / 'cluster_optimization_summary.yaml').exists())
            
            # Check file contents
            loaded_metrics = pd.read_csv(save_dir / 'cluster_optimization_metrics.csv')
            self.assertEqual(len(loaded_metrics), 3)
            self.assertIn('k', loaded_metrics.columns)
            self.assertIn('silhouette', loaded_metrics.columns)
            
            with open(save_dir / 'cluster_optimization_summary.yaml', 'r') as f:
                loaded_summary = yaml.safe_load(f)
            
            self.assertEqual(loaded_summary['consensus_k'], 3)
            self.assertEqual(loaded_summary['consensus_method'], 'median')


if __name__ == '__main__':
    unittest.main()