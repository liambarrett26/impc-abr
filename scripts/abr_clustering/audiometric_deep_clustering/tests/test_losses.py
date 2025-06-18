"""
Unit tests for ContrastiveVAE-DEC loss functions.

Tests individual loss components, multi-objective loss combination,
and loss computation during different training stages.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from losses.reconstruction import create_reconstruction_loss, MSELoss, HuberLoss, WeightedMSELoss
from losses.vae_loss import VAELoss, BetaScheduler, create_vae_loss
from losses.clustering_loss import DECLoss, AuxiliaryClusteringLoss, create_clustering_loss
from losses.contrastive import InfoNCELoss, TripletLoss, create_contrastive_loss
from losses.combined_loss import MultiObjectiveLoss, create_combined_loss


class TestReconstructionLosses(unittest.TestCase):
    """Test reconstruction loss functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 16
        self.feature_dim = 18
        self.original = torch.randn(self.batch_size, self.feature_dim)
        self.reconstructed = torch.randn(self.batch_size, self.feature_dim)
        
        self.config = {
            'loss_weights': {
                'reconstruction_weights': {
                    'abr': 2.0,
                    'metadata': 1.0,
                    'pca': 1.5
                }
            },
            'data': {
                'abr_features': 6,
                'metadata_features': 10,
                'pca_features': 2
            }
        }
    
    def test_mse_loss(self):
        """Test MSE reconstruction loss."""
        loss_fn = MSELoss()
        loss = loss_fn(self.reconstructed, self.original)
        
        # Check loss is scalar and positive
        self.assertEqual(loss.shape, ())
        self.assertGreaterEqual(loss.item(), 0)
        
        # Test perfect reconstruction gives zero loss
        perfect_recon = self.original.clone()
        zero_loss = loss_fn(perfect_recon, self.original)
        self.assertAlmostEqual(zero_loss.item(), 0, places=6)
    
    def test_huber_loss(self):
        """Test Huber reconstruction loss."""
        loss_fn = HuberLoss(delta=1.0)
        loss = loss_fn(self.reconstructed, self.original)
        
        self.assertEqual(loss.shape, ())
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_weighted_mse_loss(self):
        """Test weighted MSE loss."""
        weights = torch.tensor([2.0] * 6 + [1.0] * 10 + [1.5] * 2)  # ABR, metadata, PCA weights
        loss_fn = WeightedMSELoss(weights)
        
        loss = loss_fn(self.reconstructed, self.original)
        
        self.assertEqual(loss.shape, ())
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_reconstruction_loss_factory(self):
        """Test reconstruction loss factory function."""
        # Test MSE loss creation
        mse_loss = create_reconstruction_loss(self.config, loss_type='mse')
        self.assertIsInstance(mse_loss, MSELoss)
        
        # Test weighted MSE loss creation
        weighted_loss = create_reconstruction_loss(self.config, loss_type='weighted_mse')
        self.assertIsInstance(weighted_loss, WeightedMSELoss)


class TestVAELoss(unittest.TestCase):
    """Test VAE loss components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 16
        self.latent_dim = 10
        self.feature_dim = 18
        
        self.mu = torch.randn(self.batch_size, self.latent_dim)
        self.logvar = torch.randn(self.batch_size, self.latent_dim)
        self.original = torch.randn(self.batch_size, self.feature_dim)
        self.reconstructed = torch.randn(self.batch_size, self.feature_dim)
        
        self.config = {
            'latent': {'beta': 1.0},
            'loss_weights': {'reconstruction_weights': {'abr': 1.0, 'metadata': 1.0, 'pca': 1.0}},
            'data': {'abr_features': 6, 'metadata_features': 10, 'pca_features': 2}
        }
    
    def test_vae_loss_computation(self):
        """Test VAE loss computation."""
        reconstruction_loss = create_reconstruction_loss(self.config)
        vae_loss = VAELoss(self.config, reconstruction_loss)
        
        losses = vae_loss(self.reconstructed, self.original, self.mu, self.logvar)
        
        # Check required loss components
        required_keys = ['reconstruction_loss', 'kl_loss', 'beta_kl_loss', 'elbo_loss']
        for key in required_keys:
            self.assertIn(key, losses)
            self.assertIsInstance(losses[key], torch.Tensor)
            self.assertEqual(losses[key].shape, ())
    
    def test_kl_divergence_computation(self):
        """Test KL divergence computation."""
        reconstruction_loss = create_reconstruction_loss(self.config)
        vae_loss = VAELoss(self.config, reconstruction_loss)
        
        # Test KL divergence calculation
        kl_loss = vae_loss.compute_kl_divergence(self.mu, self.logvar)
        
        self.assertEqual(kl_loss.shape, ())
        self.assertGreaterEqual(kl_loss.item(), 0)  # KL divergence is non-negative
        
        # Test that standard normal distribution gives zero KL
        zero_mu = torch.zeros_like(self.mu)
        zero_logvar = torch.zeros_like(self.logvar)
        zero_kl = vae_loss.compute_kl_divergence(zero_mu, zero_logvar)
        self.assertAlmostEqual(zero_kl.item(), 0, places=5)
    
    def test_beta_scheduler(self):
        """Test beta parameter scheduling."""
        scheduler = BetaScheduler(beta_max=2.0, warmup_epochs=10, schedule_type='linear')
        
        # Test linear schedule
        self.assertAlmostEqual(scheduler.get_beta(0), 0.0, places=5)
        self.assertAlmostEqual(scheduler.get_beta(5), 1.0, places=5)
        self.assertAlmostEqual(scheduler.get_beta(10), 2.0, places=5)
        self.assertAlmostEqual(scheduler.get_beta(15), 2.0, places=5)  # Should clamp at max
        
        # Test cosine schedule
        cosine_scheduler = BetaScheduler(beta_max=1.0, warmup_epochs=10, schedule_type='cosine')
        beta_values = [cosine_scheduler.get_beta(epoch) for epoch in range(15)]
        
        # Check monotonic increase during warmup
        for i in range(1, 10):
            self.assertGreaterEqual(beta_values[i], beta_values[i-1])


class TestClusteringLoss(unittest.TestCase):
    """Test clustering loss functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 16
        self.latent_dim = 10
        self.num_clusters = 4
        
        self.latent_z = torch.randn(self.batch_size, self.latent_dim)
        self.q = torch.softmax(torch.randn(self.batch_size, self.num_clusters), dim=1)
        self.cluster_centers = torch.randn(self.num_clusters, self.latent_dim)
        
        self.config = {
            'clustering': {'num_clusters': self.num_clusters, 'alpha': 1.0},
            'loss_weights': {'clustering': 1.0}
        }
    
    def test_dec_loss(self):
        """Test DEC clustering loss."""
        dec_loss = DECLoss(self.config)
        
        # Compute target distribution
        p = dec_loss.compute_target_distribution(self.q)
        
        # Test target distribution properties
        self.assertEqual(p.shape, self.q.shape)
        
        # Check p sums to 1 for each sample
        p_sums = torch.sum(p, dim=1)
        self.assertTrue(torch.allclose(p_sums, torch.ones(self.batch_size), atol=1e-5))
        
        # Compute DEC loss
        loss = dec_loss.compute_dec_loss(self.q, p)
        self.assertEqual(loss.shape, ())
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_auxiliary_clustering_loss(self):
        """Test auxiliary clustering losses."""
        aux_loss = AuxiliaryClusteringLoss(self.config)
        
        # Test cluster separation loss
        sep_loss = aux_loss.cluster_separation_loss(self.latent_z, self.cluster_centers)
        self.assertEqual(sep_loss.shape, ())
        
        # Test cluster compactness loss
        comp_loss = aux_loss.cluster_compactness_loss(self.latent_z, self.q, self.cluster_centers)
        self.assertEqual(comp_loss.shape, ())
    
    def test_clustering_loss_factory(self):
        """Test clustering loss factory function."""
        clustering_loss = create_clustering_loss(self.config)
        
        # Test forward pass
        abr_features = torch.randn(self.batch_size, 6)
        gene_labels = torch.randint(0, 5, (self.batch_size,))
        
        losses = clustering_loss(
            self.latent_z, self.q, self.q, self.cluster_centers,
            abr_features, gene_labels, epoch=10
        )
        
        self.assertIn('total_clustering_loss', losses)


class TestContrastiveLoss(unittest.TestCase):
    """Test contrastive learning losses."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 16
        self.feature_dim = 64
        
        self.anchor_features = torch.randn(self.batch_size, self.feature_dim)
        self.positive_features = torch.randn(self.batch_size, self.feature_dim)
        
        self.config = {
            'contrastive': {'temperature': 0.5},
            'loss_weights': {'contrastive': 1.0}
        }
    
    def test_infonce_loss(self):
        """Test InfoNCE contrastive loss."""
        infonce_loss = InfoNCELoss(temperature=0.5)
        
        loss = infonce_loss(self.anchor_features, self.positive_features)
        
        self.assertEqual(loss.shape, ())
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_triplet_loss(self):
        """Test triplet contrastive loss."""
        negative_features = torch.randn(self.batch_size, self.feature_dim)
        triplet_loss = TripletLoss(margin=1.0)
        
        loss = triplet_loss(self.anchor_features, self.positive_features, negative_features)
        
        self.assertEqual(loss.shape, ())
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_contrastive_loss_factory(self):
        """Test contrastive loss factory function."""
        contrastive_loss = create_contrastive_loss(self.config)
        
        # Test forward pass
        abr_features = torch.randn(self.batch_size, 6)
        gene_labels = torch.randint(0, 5, (self.batch_size,))
        
        losses = contrastive_loss(
            self.anchor_features, self.positive_features,
            abr_features, gene_labels, epoch=10
        )
        
        self.assertIn('total_contrastive_loss', losses)


class TestMultiObjectiveLoss(unittest.TestCase):
    """Test multi-objective loss combination."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 16
        self.feature_dim = 18
        self.latent_dim = 10
        self.num_clusters = 4
        
        self.config = {
            'data': {'abr_features': 6, 'metadata_features': 10, 'pca_features': 2},
            'latent': {'beta': 1.0},
            'clustering': {'num_clusters': self.num_clusters, 'alpha': 1.0},
            'contrastive': {'temperature': 0.5},
            'loss_weights': {
                'reconstruction': 1.0,
                'kl_divergence': 1.0,
                'clustering': 1.0,
                'contrastive': 0.5,
                'phenotype_consistency': 0.3,
                'frequency_smoothness': 0.1,
                'reconstruction_weights': {'abr': 1.0, 'metadata': 1.0, 'pca': 1.0}
            }
        }
        
        # Create model outputs
        self.model_output = {
            'reconstruction': torch.randn(self.batch_size, self.feature_dim),
            'latent_mu': torch.randn(self.batch_size, self.latent_dim),
            'latent_logvar': torch.randn(self.batch_size, self.latent_dim),
            'latent_z': torch.randn(self.batch_size, self.latent_dim),
            'contrastive_features': torch.randn(self.batch_size, 64),
            'q': torch.softmax(torch.randn(self.batch_size, self.num_clusters), dim=1),
            'cluster_centers': torch.randn(self.num_clusters, self.latent_dim)
        }
        
        self.batch = {
            'features': torch.randn(self.batch_size, self.feature_dim),
            'gene_label': torch.randint(0, 5, (self.batch_size,)),
            'positive': torch.randn(self.batch_size, 64)
        }
    
    def test_multi_objective_loss_initialization(self):
        """Test multi-objective loss initialization."""
        loss_fn = MultiObjectiveLoss(self.config)
        
        # Check initialization
        self.assertEqual(loss_fn.training_stage, 'pretrain')
        self.assertIsNotNone(loss_fn.reconstruction_loss)
        self.assertIsNotNone(loss_fn.vae_loss)
        self.assertIsNotNone(loss_fn.clustering_loss)
        self.assertIsNotNone(loss_fn.contrastive_loss)
    
    def test_pretrain_stage_loss(self):
        """Test loss computation in pretraining stage."""
        loss_fn = MultiObjectiveLoss(self.config)
        loss_fn.set_training_stage('pretrain')
        
        losses = loss_fn(self.model_output, self.batch, epoch=5)
        
        # Check required losses for pretraining
        required_keys = ['total_loss', 'reconstruction_loss', 'kl_loss', 'elbo_loss']
        for key in required_keys:
            self.assertIn(key, losses)
            self.assertIsInstance(losses[key], torch.Tensor)
        
        # In pretraining, total loss should be VAE loss
        self.assertAlmostEqual(
            losses['total_loss'].item(),
            losses['elbo_loss'].item(),
            places=5
        )
    
    def test_joint_stage_loss(self):
        """Test loss computation in joint training stage."""
        loss_fn = MultiObjectiveLoss(self.config)
        loss_fn.set_training_stage('joint')
        
        losses = loss_fn(self.model_output, self.batch, epoch=10)
        
        # Check all loss components are present
        expected_keys = [
            'total_loss', 'reconstruction_loss', 'kl_loss',
            'clustering_loss', 'contrastive_loss', 'phenotype_consistency_loss'
        ]
        for key in expected_keys:
            self.assertIn(key, losses)
        
        # Check weighted losses
        weighted_keys = [
            'weighted_reconstruction', 'weighted_kl', 'weighted_clustering',
            'weighted_contrastive', 'weighted_phenotype'
        ]
        for key in weighted_keys:
            self.assertIn(key, losses)
    
    def test_loss_weight_management(self):
        """Test loss weight management."""
        loss_fn = MultiObjectiveLoss(self.config)
        
        # Test getting weights
        weights = loss_fn.get_loss_weights()
        self.assertIn('reconstruction', weights)
        
        # Test updating weights
        new_weights = {'clustering': 2.0}
        loss_fn.update_loss_weights(new_weights)
        updated_weights = loss_fn.get_loss_weights()
        self.assertEqual(updated_weights['clustering'], 2.0)
    
    def test_phenotype_consistency_loss(self):
        """Test phenotype consistency loss computation."""
        loss_fn = MultiObjectiveLoss(self.config)
        
        latent_z = torch.randn(self.batch_size, self.latent_dim)
        gene_labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1] + [3] * 8)  # Mixed gene labels
        
        consistency_loss = loss_fn._compute_phenotype_consistency(latent_z, gene_labels)
        
        self.assertEqual(consistency_loss.shape, ())
        self.assertGreaterEqual(consistency_loss.item(), 0)
    
    def test_frequency_smoothness_loss(self):
        """Test frequency smoothness loss computation."""
        loss_fn = MultiObjectiveLoss(self.config)
        
        original_abr = torch.randn(self.batch_size, 6)
        reconstructed_abr = torch.randn(self.batch_size, 6)
        
        smoothness_loss = loss_fn._compute_frequency_smoothness(original_abr, reconstructed_abr)
        
        self.assertEqual(smoothness_loss.shape, ())
        self.assertGreaterEqual(smoothness_loss.item(), 0)
    
    def test_combined_loss_factory(self):
        """Test combined loss factory function."""
        loss_fn = create_combined_loss(self.config)
        self.assertIsInstance(loss_fn, MultiObjectiveLoss)


class TestLossNumericalStability(unittest.TestCase):
    """Test numerical stability of loss functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'data': {'abr_features': 6, 'metadata_features': 10, 'pca_features': 2},
            'latent': {'beta': 1.0},
            'clustering': {'num_clusters': 2, 'alpha': 1.0},
            'contrastive': {'temperature': 0.5},
            'loss_weights': {
                'reconstruction': 1.0, 'kl_divergence': 1.0, 'clustering': 1.0,
                'contrastive': 1.0, 'reconstruction_weights': {'abr': 1.0, 'metadata': 1.0, 'pca': 1.0}
            }
        }
    
    def test_extreme_values(self):
        """Test loss computation with extreme values."""
        loss_fn = MultiObjectiveLoss(self.config)
        
        # Test with very large values
        large_output = {
            'reconstruction': torch.randn(4, 18) * 100,
            'latent_mu': torch.randn(4, 8) * 100,
            'latent_logvar': torch.randn(4, 8) * 10,
            'latent_z': torch.randn(4, 8),
            'contrastive_features': torch.randn(4, 32),
            'q': torch.softmax(torch.randn(4, 2), dim=1)
        }
        
        large_batch = {'features': torch.randn(4, 18) * 100}
        
        losses = loss_fn(large_output, large_batch)
        
        # Check all losses are finite
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.all(torch.isfinite(value)), f"Loss {key} is not finite")
    
    def test_zero_values(self):
        """Test loss computation with zero values."""
        loss_fn = MultiObjectiveLoss(self.config)
        
        zero_output = {
            'reconstruction': torch.zeros(4, 18),
            'latent_mu': torch.zeros(4, 8),
            'latent_logvar': torch.zeros(4, 8),
            'latent_z': torch.zeros(4, 8),
            'contrastive_features': torch.zeros(4, 32),
            'q': torch.ones(4, 2) / 2  # Uniform distribution
        }
        
        zero_batch = {'features': torch.zeros(4, 18)}
        
        losses = loss_fn(zero_output, zero_batch)
        
        # Check reconstruction loss is zero for perfect reconstruction
        self.assertAlmostEqual(losses['reconstruction_loss'].item(), 0, places=5)


if __name__ == '__main__':
    # Set up test environment
    torch.manual_seed(42)
    
    # Run tests
    unittest.main(verbosity=2)