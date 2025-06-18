"""
Unit tests for ContrastiveVAE-DEC model components.

Tests model architecture, forward passes, training stages, and integration.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.full_model import ContrastiveVAEDEC, create_model
from models.encoder import ContrastiveVAEEncoder
from models.decoder import ContrastiveVAEDecoder
from models.clustering_layer import ClusteringLayer
from models.attention import FrequencyAttention
from models.vae import LatentSpace


class TestModelArchitecture(unittest.TestCase):
    """Test model architecture and components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'data': {
                'abr_features': 6,
                'metadata_features': 10,
                'pca_features': 2,
                'total_features': 18
            },
            'encoder': {
                'hidden_dims': [64, 32, 16],
                'dropout_rate': 0.2,
                'activation': 'relu',
                'abr_encoder': {
                    'hidden_dim': 16,
                    'use_batch_norm': True
                },
                'metadata_encoder': {
                    'hidden_dim': 24
                },
                'attention': {
                    'enabled': True,
                    'head_dim': 8,
                    'num_heads': 2,
                    'dropout': 0.1
                }
            },
            'decoder': {
                'hidden_dims': [16, 32, 64],
                'dropout_rate': 0.2,
                'activation': 'relu',
                'output_activation': 'linear',
                'use_batch_norm': True
            },
            'latent': {
                'latent_dim': 10,
                'beta': 1.0,
                'min_logvar': -10,
                'max_logvar': 10
            },
            'clustering': {
                'num_clusters': 12,
                'alpha': 1.0
            },
            'contrastive': {
                'temperature': 0.5,
                'projection_dim': 64
            },
            'architecture': {
                'weight_init': 'xavier_uniform',
                'bias_init': 'zeros'
            }
        }
        
        self.batch_size = 32
        self.input_features = torch.randn(self.batch_size, 18)
    
    def test_frequency_attention(self):
        """Test frequency attention mechanism."""
        attention = FrequencyAttention(
            input_dim=6,
            head_dim=8,
            num_heads=2,
            dropout=0.1
        )
        
        # Test forward pass
        abr_input = torch.randn(self.batch_size, 6)
        output, attention_weights = attention(abr_input)
        
        # Check output shapes
        self.assertEqual(output.shape, (self.batch_size, 6))
        self.assertEqual(attention_weights.shape, (self.batch_size, 6, 6))
        
        # Check attention weights sum to 1
        attention_sums = torch.sum(attention_weights, dim=-1)
        self.assertTrue(torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=1e-6))
    
    def test_encoder(self):
        """Test encoder component."""
        encoder = ContrastiveVAEEncoder(self.config)
        
        # Test forward pass
        output = encoder(self.input_features)
        
        # Check required outputs
        required_keys = ['latent_mu', 'latent_logvar', 'latent_z']
        for key in required_keys:
            self.assertIn(key, output)
        
        # Check output shapes
        latent_dim = self.config['latent']['latent_dim']
        self.assertEqual(output['latent_mu'].shape, (self.batch_size, latent_dim))
        self.assertEqual(output['latent_logvar'].shape, (self.batch_size, latent_dim))
        self.assertEqual(output['latent_z'].shape, (self.batch_size, latent_dim))
        
        # Check attention weights if enabled
        if self.config['encoder']['attention']['enabled']:
            self.assertIn('attention_weights', output)
    
    def test_decoder(self):
        """Test decoder component."""
        decoder = ContrastiveVAEDecoder(self.config)
        
        # Test forward pass
        latent_input = torch.randn(self.batch_size, self.config['latent']['latent_dim'])
        output = decoder(latent_input)
        
        # Check output shape
        self.assertIn('reconstruction', output)
        self.assertEqual(output['reconstruction'].shape, (self.batch_size, 18))
    
    def test_clustering_layer(self):
        """Test clustering layer."""
        clustering_layer = ClusteringLayer(self.config)
        
        # Test forward pass
        latent_input = torch.randn(self.batch_size, self.config['latent']['latent_dim'])
        output = clustering_layer(latent_input)
        
        # Check outputs
        required_keys = ['q', 'distances', 'hard_assignments', 'cluster_centers']
        for key in required_keys:
            self.assertIn(key, output)
        
        # Check shapes
        num_clusters = self.config['clustering']['num_clusters']
        self.assertEqual(output['q'].shape, (self.batch_size, num_clusters))
        self.assertEqual(output['distances'].shape, (self.batch_size, num_clusters))
        self.assertEqual(output['hard_assignments'].shape, (self.batch_size,))
        
        # Check soft assignments sum to 1
        q_sums = torch.sum(output['q'], dim=1)
        self.assertTrue(torch.allclose(q_sums, torch.ones(self.batch_size), atol=1e-6))
    
    def test_full_model(self):
        """Test complete ContrastiveVAE-DEC model."""
        model = ContrastiveVAEDEC(self.config)
        
        # Test forward pass
        output = model(self.input_features)
        
        # Check required outputs
        required_keys = ['reconstruction', 'latent_mu', 'latent_logvar', 'latent_z', 'contrastive_features']
        for key in required_keys:
            self.assertIn(key, output)
        
        # Check shapes
        self.assertEqual(output['reconstruction'].shape, (self.batch_size, 18))
        self.assertEqual(output['latent_z'].shape, (self.batch_size, 10))
        self.assertEqual(output['contrastive_features'].shape, (self.batch_size, 64))
    
    def test_model_factory(self):
        """Test model factory function."""
        model = create_model(self.config)
        
        self.assertIsInstance(model, ContrastiveVAEDEC)
        
        # Test model can process input
        output = model(self.input_features)
        self.assertIn('reconstruction', output)


class TestModelTrainingStages(unittest.TestCase):
    """Test model training stages and state management."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use minimal config for faster tests
        self.config = {
            'data': {'abr_features': 6, 'metadata_features': 10, 'pca_features': 2, 'total_features': 18},
            'encoder': {
                'hidden_dims': [32, 16],
                'dropout_rate': 0.1,
                'abr_encoder': {'hidden_dim': 8, 'use_batch_norm': False},
                'metadata_encoder': {'hidden_dim': 12},
                'attention': {'enabled': False}
            },
            'decoder': {'hidden_dims': [16, 32], 'dropout_rate': 0.1, 'use_batch_norm': False},
            'latent': {'latent_dim': 8, 'beta': 1.0, 'min_logvar': -10, 'max_logvar': 10},
            'clustering': {'num_clusters': 4, 'alpha': 1.0},
            'contrastive': {'temperature': 0.5, 'projection_dim': 32},
            'architecture': {'weight_init': 'xavier_uniform'}
        }
        
        self.model = ContrastiveVAEDEC(self.config)
        self.batch_size = 16
        self.input_features = torch.randn(self.batch_size, 18)
    
    def test_training_stage_management(self):
        """Test training stage transitions."""
        # Test initial stage
        self.assertEqual(self.model.training_stage, 'pretrain')
        self.assertFalse(self.model.clusters_initialized)
        
        # Test stage setting
        self.model.set_training_stage('joint')
        self.assertEqual(self.model.training_stage, 'joint')
        
        # Test invalid stage
        with self.assertRaises(ValueError):
            self.model.set_training_stage('invalid_stage')
    
    def test_cluster_initialization(self):
        """Test cluster initialization."""
        # Create mock dataloader
        dataset = torch.utils.data.TensorDataset(self.input_features)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        
        # Initialize clusters
        self.model.initialize_clusters(dataloader)
        
        # Check initialization state
        self.assertTrue(self.model.clusters_initialized)
        
        # Test cluster assignments after initialization
        output = self.model(self.input_features)
        self.assertIn('q', output)
    
    def test_cluster_assignments(self):
        """Test cluster assignment methods."""
        # Initialize clusters first
        dataset = torch.utils.data.TensorDataset(self.input_features)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        self.model.initialize_clusters(dataloader)
        
        # Test cluster assignments
        soft_assignments, hard_assignments = self.model.get_cluster_assignments(self.input_features)
        
        # Check shapes
        self.assertEqual(soft_assignments.shape, (self.batch_size, 4))
        self.assertEqual(hard_assignments.shape, (self.batch_size,))
        
        # Check hard assignments are valid cluster indices
        self.assertTrue(torch.all(hard_assignments >= 0))
        self.assertTrue(torch.all(hard_assignments < 4))


class TestModelUtilities(unittest.TestCase):
    """Test model utility functions and methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'data': {'abr_features': 6, 'metadata_features': 10, 'pca_features': 2, 'total_features': 18},
            'encoder': {
                'hidden_dims': [32],
                'dropout_rate': 0.0,
                'abr_encoder': {'hidden_dim': 8, 'use_batch_norm': False},
                'metadata_encoder': {'hidden_dim': 12},
                'attention': {'enabled': False}
            },
            'decoder': {'hidden_dims': [32], 'dropout_rate': 0.0, 'use_batch_norm': False},
            'latent': {'latent_dim': 8, 'beta': 1.0, 'min_logvar': -10, 'max_logvar': 10},
            'clustering': {'num_clusters': 4, 'alpha': 1.0},
            'contrastive': {'temperature': 0.5, 'projection_dim': 16},
            'architecture': {'weight_init': 'xavier_uniform'}
        }
        
        self.model = ContrastiveVAEDEC(self.config)
        self.input_features = torch.randn(16, 18)
    
    def test_encode_decode(self):
        """Test encode and decode methods."""
        # Test encoding
        encoded = self.model.encode(self.input_features)
        self.assertEqual(encoded.shape, (16, 8))
        
        # Test decoding
        decoded = self.model.decode(encoded)
        self.assertEqual(decoded.shape, (16, 18))
    
    def test_sampling(self):
        """Test sampling from learned prior."""
        device = next(self.model.parameters()).device
        samples = self.model.sample(num_samples=5, device=device)
        self.assertEqual(samples.shape, (5, 18))
    
    def test_interpolation(self):
        """Test interpolation between inputs."""
        x1 = self.input_features[:1]  # First sample
        x2 = self.input_features[1:2]  # Second sample
        
        interpolated = self.model.interpolate(x1, x2, num_steps=5)
        self.assertEqual(interpolated.shape, (5, 18))
    
    def test_model_analysis(self):
        """Test model analysis functionality."""
        dataset = torch.utils.data.TensorDataset(self.input_features)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        
        # Initialize clusters for analysis
        self.model.initialize_clusters(dataloader)
        
        # Run analysis
        analysis = self.model.analyze_model(dataloader)
        
        # Check required metrics
        required_keys = ['reconstruction_mse', 'latent_mean_norm', 'latent_std_mean']
        for key in required_keys:
            self.assertIn(key, analysis)
            self.assertIsInstance(analysis[key], float)


class TestModelNumericalStability(unittest.TestCase):
    """Test model numerical stability and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'data': {'abr_features': 6, 'metadata_features': 10, 'pca_features': 2, 'total_features': 18},
            'encoder': {
                'hidden_dims': [16],
                'dropout_rate': 0.0,
                'abr_encoder': {'hidden_dim': 4, 'use_batch_norm': False},
                'metadata_encoder': {'hidden_dim': 8},
                'attention': {'enabled': False}
            },
            'decoder': {'hidden_dims': [16], 'dropout_rate': 0.0, 'use_batch_norm': False},
            'latent': {'latent_dim': 4, 'beta': 1.0, 'min_logvar': -10, 'max_logvar': 10},
            'clustering': {'num_clusters': 2, 'alpha': 1.0},
            'contrastive': {'temperature': 0.5, 'projection_dim': 8},
            'architecture': {'weight_init': 'xavier_uniform'}
        }
        
        self.model = ContrastiveVAEDEC(self.config)
    
    def test_extreme_inputs(self):
        """Test model with extreme input values."""
        # Test very large inputs
        large_input = torch.randn(4, 18) * 100
        output = self.model(large_input)
        self.assertTrue(torch.all(torch.isfinite(output['reconstruction'])))
        
        # Test very small inputs
        small_input = torch.randn(4, 18) * 0.001
        output = self.model(small_input)
        self.assertTrue(torch.all(torch.isfinite(output['reconstruction'])))
        
        # Test zero inputs
        zero_input = torch.zeros(4, 18)
        output = self.model(zero_input)
        self.assertTrue(torch.all(torch.isfinite(output['reconstruction'])))
    
    def test_single_sample_batch(self):
        """Test model with single sample batches."""
        single_input = torch.randn(1, 18)
        output = self.model(single_input)
        
        self.assertEqual(output['reconstruction'].shape, (1, 18))
        self.assertEqual(output['latent_z'].shape, (1, 4))
    
    def test_log_variance_clamping(self):
        """Test that log variance is properly clamped."""
        input_features = torch.randn(8, 18)
        output = self.model(input_features)
        
        logvar = output['latent_logvar']
        
        # Check clamping
        self.assertTrue(torch.all(logvar >= self.config['latent']['min_logvar']))
        self.assertTrue(torch.all(logvar <= self.config['latent']['max_logvar']))
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        input_features = torch.randn(4, 18, requires_grad=True)
        output = self.model(input_features)
        
        # Compute loss and backpropagate
        loss = torch.mean(output['reconstruction'])
        loss.backward()
        
        # Check that input gradients exist
        self.assertIsNotNone(input_features.grad)
        self.assertTrue(torch.any(input_features.grad != 0))


if __name__ == '__main__':
    # Set up test environment
    torch.manual_seed(42)
    
    # Run tests
    unittest.main(verbosity=2)