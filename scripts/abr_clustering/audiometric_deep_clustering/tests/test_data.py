"""
Unit tests for data handling components.

Tests dataset creation, preprocessing, data loading, and augmentation functionality.
"""

import unittest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import IMPCABRDataset, ContrastiveABRDataset, create_abr_dataset
from data.preprocessor import ABRPreprocessor
from data.dataloader import create_dataloaders, BalancedSampler
from data.augmentations import ABRAugmentations


class TestABRPreprocessor(unittest.TestCase):
    """Test ABR data preprocessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic ABR data
        np.random.seed(42)
        self.n_samples = 100
        self.abr_features = np.random.randn(self.n_samples, 6) * 20 + 50  # ABR thresholds
        self.metadata_features = np.random.randn(self.n_samples, 8)  # Metadata
        
        # Combine features
        self.raw_features = np.concatenate([self.abr_features, self.metadata_features], axis=1)
    
    def test_basic_preprocessing(self):
        """Test basic preprocessing functionality."""
        preprocessor = ABRPreprocessor(normalize=True, add_pca=False)
        
        processed = preprocessor.fit_transform(self.raw_features)
        
        # Check output shape
        self.assertEqual(processed.shape, (self.n_samples, 14))
        
        # Check normalization (should have zero mean, unit variance)
        self.assertAlmostEqual(np.mean(processed), 0, places=1)
        self.assertAlmostEqual(np.std(processed), 1, places=1)
    
    def test_pca_addition(self):
        """Test PCA component addition."""
        preprocessor = ABRPreprocessor(
            normalize=True, 
            add_pca=True, 
            n_pca_components=2,
            pca_features='abr'
        )
        
        processed = preprocessor.fit_transform(self.raw_features)
        
        # Check output shape (original + PCA components)
        self.assertEqual(processed.shape, (self.n_samples, 16))
    
    def test_abr_specific_processing(self):
        """Test ABR-specific preprocessing."""
        preprocessor = ABRPreprocessor(
            normalize=True,
            log_transform_abr=True,
            clip_abr_outliers=True,
            abr_clip_percentiles=(5, 95)
        )
        
        processed = preprocessor.fit_transform(self.raw_features)
        
        # Check that ABR features are processed
        abr_processed = processed[:, :6]
        self.assertGreater(np.min(abr_processed), -10)  # Should be clipped
        self.assertLess(np.max(abr_processed), 10)
    
    def test_feature_selection(self):
        """Test feature selection functionality."""
        # Add some constant features to test selection
        constant_features = np.ones((self.n_samples, 2))
        features_with_constant = np.concatenate([self.raw_features, constant_features], axis=1)
        
        preprocessor = ABRPreprocessor(
            normalize=True,
            feature_selection=True,
            variance_threshold=0.1
        )
        
        processed = preprocessor.fit_transform(features_with_constant)
        
        # Should remove constant features
        self.assertLess(processed.shape[1], features_with_constant.shape[1])
    
    def test_missing_value_handling(self):
        """Test missing value imputation."""
        # Add missing values
        features_with_missing = self.raw_features.copy()
        features_with_missing[0, 0] = np.nan
        features_with_missing[1, 2] = np.nan
        
        preprocessor = ABRPreprocessor(
            normalize=True,
            handle_missing=True,
            missing_strategy='median'
        )
        
        processed = preprocessor.fit_transform(features_with_missing)
        
        # Check no missing values remain
        self.assertFalse(np.any(np.isnan(processed)))
    
    def test_transform_consistency(self):
        """Test that transform is consistent after fitting."""
        preprocessor = ABRPreprocessor(normalize=True, add_pca=True)
        
        # Fit on training data
        train_features = self.raw_features[:80]
        preprocessor.fit(train_features)
        
        # Transform test data
        test_features = self.raw_features[80:]
        test_processed = preprocessor.transform(test_features)
        
        # Check shape consistency
        self.assertEqual(test_processed.shape[1], preprocessor.fit_transform(train_features).shape[1])


class TestIMPCABRDataset(unittest.TestCase):
    """Test IMPC ABR dataset functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic data
        np.random.seed(42)
        self.n_samples = 50
        self.features = np.random.randn(self.n_samples, 18)
        self.gene_labels = np.random.randint(0, 5, self.n_samples)
        self.mouse_ids = np.arange(self.n_samples)
        
        # Create metadata DataFrame
        self.metadata = pd.DataFrame({
            'mouse_id': self.mouse_ids,
            'gene': self.gene_labels,
            'age': np.random.randint(8, 20, self.n_samples),
            'weight': np.random.normal(25, 5, self.n_samples)
        })
    
    def test_dataset_creation(self):
        """Test basic dataset creation."""
        dataset = IMPCABRDataset(
            data=None,
            features=self.features,
            gene_labels=self.gene_labels,
            mouse_ids=self.mouse_ids,
            mode='train'
        )
        
        # Check dataset length
        self.assertEqual(len(dataset), self.n_samples)
        
        # Check sample retrieval
        sample = dataset[0]
        self.assertIn('features', sample)
        self.assertIn('gene_label', sample)
        self.assertIn('index', sample)
        
        # Check sample shapes
        self.assertEqual(sample['features'].shape, (18,))
        self.assertIsInstance(sample['gene_label'], torch.Tensor)
    
    def test_contrastive_pairs(self):
        """Test contrastive pair generation."""
        dataset = IMPCABRDataset(
            data=None,
            features=self.features,
            gene_labels=self.gene_labels,
            mouse_ids=self.mouse_ids,
            mode='train',
            return_pairs=True
        )
        
        sample = dataset[0]
        
        # Check positive pair is included
        self.assertIn('positive', sample)
        self.assertEqual(sample['positive'].shape, (18,))
    
    def test_gene_distribution(self):
        """Test gene distribution computation."""
        dataset = IMPCABRDataset(
            data=None,
            features=self.features,
            gene_labels=self.gene_labels,
            mouse_ids=self.mouse_ids
        )
        
        gene_dist = dataset.get_gene_distribution()
        
        # Check distribution
        self.assertIsInstance(gene_dist, dict)
        self.assertEqual(sum(gene_dist.values()), self.n_samples)
    
    def test_split_by_gene(self):
        """Test gene-based data splitting."""
        dataset = IMPCABRDataset(
            data=None,
            features=self.features,
            gene_labels=self.gene_labels,
            mouse_ids=self.mouse_ids
        )
        
        train_dataset, val_dataset, test_dataset = dataset.split_by_gene(
            train_ratio=0.6, val_ratio=0.2
        )
        
        # Check split sizes
        total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
        self.assertEqual(total_samples, self.n_samples)
        
        # Check no gene overlap between splits
        train_genes = set(train_dataset.gene_labels)
        val_genes = set(val_dataset.gene_labels)
        test_genes = set(test_dataset.gene_labels)
        
        self.assertEqual(len(train_genes & val_genes), 0)
        self.assertEqual(len(train_genes & test_genes), 0)
        self.assertEqual(len(val_genes & test_genes), 0)
    
    def test_feature_statistics(self):
        """Test feature statistics computation."""
        dataset = IMPCABRDataset(
            data=None,
            features=self.features,
            gene_labels=self.gene_labels,
            mouse_ids=self.mouse_ids
        )
        
        stats = dataset.get_feature_stats()
        
        # Check statistics
        required_keys = ['mean', 'std', 'min', 'max', 'shape']
        for key in required_keys:
            self.assertIn(key, stats)


class TestContrastiveABRDataset(unittest.TestCase):
    """Test contrastive ABR dataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 30
        self.features = np.random.randn(self.n_samples, 18)
        self.gene_labels = np.random.randint(0, 3, self.n_samples)  # Fewer genes for better pairs
        self.mouse_ids = np.arange(self.n_samples)
    
    def test_contrastive_dataset_creation(self):
        """Test contrastive dataset creation."""
        dataset = ContrastiveABRDataset(
            data=None,
            features=self.features,
            gene_labels=self.gene_labels,
            mouse_ids=self.mouse_ids,
            mode='train'
        )
        
        # Check that return_pairs is automatically True
        self.assertTrue(dataset.return_pairs)
        
        # Check sample contains positive pairs
        sample = dataset[0]
        self.assertIn('positive', sample)
    
    def test_augmentation_integration(self):
        """Test augmentation integration."""
        augmentation_fn = ABRAugmentations(
            noise_std=0.1,
            dropout_prob=0.1
        )
        
        dataset = ContrastiveABRDataset(
            data=None,
            features=self.features,
            gene_labels=self.gene_labels,
            mouse_ids=self.mouse_ids,
            mode='train',
            augment_fn=augmentation_fn
        )
        
        sample = dataset[0]
        
        # Check augmented versions are present
        self.assertIn('positive', sample)
        self.assertIn('anchor_augmented', sample)


class TestDataLoaders(unittest.TestCase):
    """Test data loader creation and functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 100
        self.features = np.random.randn(self.n_samples, 18)
        self.gene_labels = np.random.randint(0, 5, self.n_samples)
        self.mouse_ids = np.arange(self.n_samples)
        
        self.dataset = IMPCABRDataset(
            data=None,
            features=self.features,
            gene_labels=self.gene_labels,
            mouse_ids=self.mouse_ids
        )
    
    def test_dataloader_creation(self):
        """Test basic data loader creation."""
        dataloaders = create_dataloaders(
            dataset=self.dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            batch_size=16,
            seed=42
        )
        
        # Check all splits are created
        required_keys = ['train', 'val', 'test']
        for key in required_keys:
            self.assertIn(key, dataloaders)
            self.assertIsInstance(dataloaders[key], torch.utils.data.DataLoader)
        
        # Check batch sizes
        for batch in dataloaders['train']:
            self.assertLessEqual(len(batch['features']), 16)
            break
    
    def test_balanced_sampling(self):
        """Test balanced sampling functionality."""
        # Create imbalanced gene distribution
        imbalanced_gene_labels = np.array([0] * 60 + [1] * 30 + [2] * 10)
        np.random.shuffle(imbalanced_gene_labels)
        
        imbalanced_dataset = IMPCABRDataset(
            data=None,
            features=self.features,
            gene_labels=imbalanced_gene_labels,
            mouse_ids=self.mouse_ids
        )
        
        sampler = BalancedSampler(imbalanced_gene_labels, samples_per_class=10)
        
        dataloader = torch.utils.data.DataLoader(
            imbalanced_dataset,
            batch_size=16,
            sampler=sampler
        )
        
        # Check that sampler works
        for batch in dataloader:
            self.assertIn('features', batch)
            break
    
    def test_data_consistency(self):
        """Test data consistency across splits."""
        dataloaders = create_dataloaders(
            dataset=self.dataset,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            batch_size=16,
            seed=42
        )
        
        # Collect all samples from all splits
        all_indices = set()
        
        for split_name, dataloader in dataloaders.items():
            for batch in dataloader:
                indices = batch['index'].numpy()
                all_indices.update(indices)
        
        # Check all original samples are included exactly once
        self.assertEqual(len(all_indices), self.n_samples)
        self.assertEqual(all_indices, set(range(self.n_samples)))


class TestABRAugmentations(unittest.TestCase):
    """Test ABR data augmentations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.features = torch.randn(16, 18)
        self.augmentations = ABRAugmentations(
            noise_std=0.1,
            dropout_prob=0.1,
            magnitude_scale=0.1,
            frequency_shift_range=0.1
        )
    
    def test_gaussian_noise(self):
        """Test Gaussian noise augmentation."""
        augmented = self.augmentations.add_gaussian_noise(self.features)
        
        # Check shape preservation
        self.assertEqual(augmented.shape, self.features.shape)
        
        # Check that augmentation actually changes the data
        self.assertFalse(torch.allclose(augmented, self.features))
    
    def test_feature_dropout(self):
        """Test feature dropout augmentation."""
        augmented = self.augmentations.apply_feature_dropout(self.features)
        
        # Check shape preservation
        self.assertEqual(augmented.shape, self.features.shape)
        
        # Check that some features are zeroed (with high probability)
        self.assertTrue(torch.any(augmented == 0))
    
    def test_magnitude_scaling(self):
        """Test magnitude scaling augmentation."""
        augmented = self.augmentations.apply_magnitude_scaling(self.features)
        
        # Check shape preservation
        self.assertEqual(augmented.shape, self.features.shape)
        
        # Check that scaling actually changes the data
        self.assertFalse(torch.allclose(augmented, self.features))
    
    def test_frequency_shift(self):
        """Test frequency shift augmentation for ABR features."""
        abr_features = self.features[:, :6]  # First 6 features are ABR
        augmented = self.augmentations.apply_frequency_shift(abr_features)
        
        # Check shape preservation
        self.assertEqual(augmented.shape, abr_features.shape)
    
    def test_combined_augmentation(self):
        """Test combined augmentation pipeline."""
        augmented = self.augmentations(self.features)
        
        # Check shape preservation
        self.assertEqual(augmented.shape, self.features.shape)
        
        # Check that augmentation changes the data
        self.assertFalse(torch.allclose(augmented, self.features, atol=1e-6))
    
    def test_augmentation_determinism(self):
        """Test augmentation reproducibility with seeds."""
        torch.manual_seed(42)
        augmented1 = self.augmentations(self.features)
        
        torch.manual_seed(42)
        augmented2 = self.augmentations(self.features)
        
        # Should be identical with same seed
        self.assertTrue(torch.allclose(augmented1, augmented2))


class TestDatasetFactory(unittest.TestCase):
    """Test dataset factory functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary CSV file
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, 'test_data.csv')
        
        # Create synthetic data
        np.random.seed(42)
        data = {
            'specimen_id': [f'mouse_{i}' for i in range(50)],
            'gene_symbol': np.random.choice(['Gene1', 'Gene2', 'Gene3'], 50),
            'abr_6kHz': np.random.normal(40, 15, 50),
            'abr_12kHz': np.random.normal(45, 15, 50),
            'abr_18kHz': np.random.normal(50, 15, 50),
            'abr_24kHz': np.random.normal(55, 15, 50),
            'abr_30kHz': np.random.normal(60, 15, 50),
            'abr_click': np.random.normal(35, 15, 50),
            'age': np.random.randint(8, 20, 50),
            'weight': np.random.normal(25, 5, 50)
        }
        
        self.df = pd.DataFrame(data)
        self.df.to_csv(self.csv_path, index=False)
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        os.rmdir(self.temp_dir)
    
    def test_create_abr_dataset(self):
        """Test ABR dataset creation from file."""
        preprocessor = ABRPreprocessor(normalize=True, add_pca=False)
        
        feature_columns = ['abr_6kHz', 'abr_12kHz', 'abr_18kHz', 'abr_24kHz', 'abr_30kHz', 'abr_click']
        
        dataset = create_abr_dataset(
            data_path=self.csv_path,
            preprocessor=preprocessor,
            feature_columns=feature_columns,
            gene_column='gene_symbol',
            mouse_id_column='specimen_id',
            mode='train'
        )
        
        # Check dataset creation
        self.assertEqual(len(dataset), 50)
        
        # Check sample structure
        sample = dataset[0]
        self.assertIn('features', sample)
        self.assertIn('gene_label', sample)
        self.assertIn('mouse_id', sample)
    
    def test_missing_columns_handling(self):
        """Test handling of missing columns."""
        preprocessor = ABRPreprocessor(normalize=True)
        
        # Try with non-existent column
        with self.assertRaises(KeyError):
            create_abr_dataset(
                data_path=self.csv_path,
                preprocessor=preprocessor,
                feature_columns=['non_existent_column'],
                mode='train'
            )


class TestDataConsistency(unittest.TestCase):
    """Test data consistency and edge cases."""
    
    def test_empty_dataset(self):
        """Test handling of empty datasets."""
        empty_features = np.empty((0, 18))
        empty_gene_labels = np.empty(0, dtype=int)
        empty_mouse_ids = np.empty(0, dtype=int)
        
        dataset = IMPCABRDataset(
            data=None,
            features=empty_features,
            gene_labels=empty_gene_labels,
            mouse_ids=empty_mouse_ids
        )
        
        # Check empty dataset length
        self.assertEqual(len(dataset), 0)
    
    def test_single_sample_dataset(self):
        """Test handling of single sample datasets."""
        single_features = np.random.randn(1, 18)
        single_gene_labels = np.array([0])
        single_mouse_ids = np.array([0])
        
        dataset = IMPCABRDataset(
            data=None,
            features=single_features,
            gene_labels=single_gene_labels,
            mouse_ids=single_mouse_ids
        )
        
        # Check single sample dataset
        self.assertEqual(len(dataset), 1)
        
        sample = dataset[0]
        self.assertEqual(sample['features'].shape, (18,))
    
    def test_inconsistent_data_sizes(self):
        """Test handling of inconsistent data sizes."""
        features = np.random.randn(10, 18)
        gene_labels = np.random.randint(0, 3, 5)  # Wrong size
        
        # Should handle inconsistent sizes gracefully
        dataset = IMPCABRDataset(
            data=None,
            features=features,
            gene_labels=gene_labels,
            mouse_ids=None
        )
        
        # Dataset should still be created but might have issues
        self.assertEqual(len(dataset), 10)


if __name__ == '__main__':
    # Set up test environment
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    unittest.main(verbosity=2)