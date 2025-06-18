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
from data.preprocessor import ABRFeaturePreprocessor, create_preprocessor, create_default_config
from data.dataloader import create_data_module, BalancedGeneSampler
from data.augmentations import ContrastiveAugmentationPipeline, create_augmentation_pipeline


class TestABRFeaturePreprocessor(unittest.TestCase):
    """Test ABR data preprocessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic ABR data
        np.random.seed(42)
        self.n_samples = 100
        
        # Create test DataFrame with proper column names
        abr_columns = [
            '6kHz-evoked ABR Threshold',
            '12kHz-evoked ABR Threshold',
            '18kHz-evoked ABR Threshold',
            '24kHz-evoked ABR Threshold',
            '30kHz-evoked ABR Threshold',
            'Click-evoked ABR threshold'
        ]
        
        metadata_columns = [
            'age_in_weeks', 'weight', 'sex', 'zygosity',
            'genetic_background', 'phenotyping_center',
            'pipeline_name', 'metadata_Equipment manufacturer'
        ]
        
        # Generate test data
        data = {}
        
        # ABR thresholds (continuous values 20-100 dB SPL)
        for col in abr_columns:
            data[col] = np.random.randn(self.n_samples) * 20 + 50
            
        # Continuous metadata
        data['age_in_weeks'] = np.random.randn(self.n_samples) * 5 + 12
        data['weight'] = np.random.randn(self.n_samples) * 5 + 25
        
        # Categorical metadata
        data['sex'] = np.random.choice(['Male', 'Female'], self.n_samples)
        data['zygosity'] = np.random.choice(['homozygote', 'heterozygote'], self.n_samples)
        data['genetic_background'] = np.random.choice(['C57BL/6N', 'C57BL/6J'], self.n_samples)
        data['phenotyping_center'] = np.random.choice(['JAX', 'TCP'], self.n_samples)
        data['pipeline_name'] = np.random.choice(['IMPC_001', 'IMPC_002'], self.n_samples)
        data['metadata_Equipment manufacturer'] = np.random.choice(['Bioseb', 'Other'], self.n_samples)
        
        self.test_df = pd.DataFrame(data)
    
    def test_basic_preprocessing(self):
        """Test basic preprocessing functionality."""
        config = create_default_config()
        preprocessor = ABRFeaturePreprocessor(config)
        
        processed = preprocessor.fit_transform(self.test_df)
        
        # Check output shape (6 ABR + 10 metadata + 2 PCA = 18)
        self.assertEqual(processed.shape, (self.n_samples, 18))
        
        # Check that we get reasonable values (not all zeros or NaN)
        self.assertFalse(np.isnan(processed).any())
        self.assertTrue(np.isfinite(processed).all())
    
    def test_pca_addition(self):
        """Test PCA component addition."""
        config = create_default_config()
        preprocessor = ABRFeaturePreprocessor(config)
        
        processed = preprocessor.fit_transform(self.test_df)
        
        # Check output shape includes PCA components (6 ABR + 10 metadata + 2 PCA = 18)
        self.assertEqual(processed.shape, (self.n_samples, 18))
        
        # PCA components should be in the last 2 columns
        pca_components = processed[:, -2:]
        self.assertEqual(pca_components.shape, (self.n_samples, 2))
    
    def test_abr_specific_processing(self):
        """Test ABR-specific preprocessing."""
        config = create_default_config()
        preprocessor = ABRFeaturePreprocessor(config)
        
        processed = preprocessor.fit_transform(self.test_df)
        
        # Check that ABR features are processed (first 6 columns)
        abr_processed = processed[:, :6]
        self.assertEqual(abr_processed.shape, (self.n_samples, 6))
        
        # Check that ABR values are reasonable after standardization
        self.assertTrue(np.isfinite(abr_processed).all())
    
    def test_feature_selection(self):
        """Test feature selection functionality."""
        config = create_default_config()
        preprocessor = ABRFeaturePreprocessor(config)
        
        processed = preprocessor.fit_transform(self.test_df)
        
        # Should produce expected number of features
        self.assertEqual(processed.shape[1], 18)  # 6 ABR + 10 metadata + 2 PCA
        
        # All features should have some variance (not constant)
        feature_stds = np.std(processed, axis=0)
        self.assertTrue(np.all(feature_stds > 0))
    
    def test_missing_value_handling(self):
        """Test missing value imputation."""
        # Add missing values to test DataFrame
        test_df_missing = self.test_df.copy()
        test_df_missing.iloc[0, 0] = np.nan  # Missing ABR value
        test_df_missing.iloc[1, 6] = np.nan  # Missing age
        test_df_missing.iloc[2, 8] = None    # Missing categorical value
        
        config = create_default_config()
        preprocessor = ABRFeaturePreprocessor(config)
        
        processed = preprocessor.fit_transform(test_df_missing)
        
        # Check no missing values remain
        self.assertFalse(np.any(np.isnan(processed)))
        self.assertTrue(np.isfinite(processed).all())
    
    def test_transform_consistency(self):
        """Test that transform is consistent after fitting."""
        config = create_default_config()
        preprocessor = ABRFeaturePreprocessor(config)
        
        # Fit on training data
        train_df = self.test_df[:80]
        preprocessor.fit(train_df)
        
        # Transform test data
        test_df = self.test_df[80:]
        test_processed = preprocessor.transform(test_df)
        
        # Check shape consistency
        expected_shape = preprocessor.fit_transform(train_df).shape[1]
        self.assertEqual(test_processed.shape[1], expected_shape)


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
        aug_config = {
            'noise_std': 0.1,
            'dropout_prob': 0.1,
            'pipeline_probability': 1.0
        }
        augmentation_fn = create_augmentation_pipeline(aug_config)
        
        dataset = ContrastiveABRDataset(
            data=None,
            features=self.features,
            gene_labels=self.gene_labels,
            mouse_ids=self.mouse_ids,
            mode='train',
            augment_fn=augmentation_fn
        )
        
        sample = dataset[0]
        
        # Check basic structure
        self.assertIn('features', sample)
        self.assertIsInstance(sample['features'], torch.Tensor)


class TestDataLoaders(unittest.TestCase):
    """Test data loader creation and functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_samples = 100
        
        # Create temporary directory and CSV file
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, 'test_data.csv')
        
        # Create test DataFrame similar to preprocessor test
        abr_columns = [
            '6kHz-evoked ABR Threshold',
            '12kHz-evoked ABR Threshold',
            '18kHz-evoked ABR Threshold',
            '24kHz-evoked ABR Threshold',
            '30kHz-evoked ABR Threshold',
            'Click-evoked ABR threshold'
        ]
        
        # Generate test data
        data = {}
        
        # ABR thresholds
        for col in abr_columns:
            data[col] = np.random.randn(self.n_samples) * 20 + 50
            
        # Metadata
        data['age_in_weeks'] = np.random.randn(self.n_samples) * 5 + 12
        data['weight'] = np.random.randn(self.n_samples) * 5 + 25
        data['sex'] = np.random.choice(['Male', 'Female'], self.n_samples)
        data['zygosity'] = np.random.choice(['homozygote', 'heterozygote'], self.n_samples)
        data['genetic_background'] = np.random.choice(['C57BL/6N', 'C57BL/6J'], self.n_samples)
        data['phenotyping_center'] = np.random.choice(['JAX', 'TCP'], self.n_samples)
        data['pipeline_name'] = np.random.choice(['IMPC_001', 'IMPC_002'], self.n_samples)
        data['metadata_Equipment manufacturer'] = np.random.choice(['Bioseb', 'Other'], self.n_samples)
        
        # Add gene labels and specimen IDs
        data['gene_symbol'] = np.random.choice(['Gene_A', 'Gene_B', 'Gene_C', 'Gene_D', 'Gene_E'], self.n_samples)
        data['specimen_id'] = [f'mouse_{i}' for i in range(self.n_samples)]
        
        # Save as CSV
        test_df = pd.DataFrame(data)
        test_df.to_csv(self.csv_path, index=False)
        
        # Create data module config
        self.config = {
            'batch_size': 16,
            'num_workers': 0,  # No multiprocessing for tests
            'pin_memory': False,
            'train_split': 0.7,
            'val_split': 0.15
        }
    
    def test_dataloader_creation(self):
        """Test basic data loader creation."""
        data_module = create_data_module(self.config)
        data_module.setup(self.csv_path)
        
        # Create dataloaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()
        
        # Check dataloaders are created
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        
        # Check batch sizes
        for batch in train_loader:
            self.assertIn('features', batch)
            features = batch['features']
            self.assertLessEqual(len(features), 16)
            self.assertEqual(features.shape[1], 18)  # Expected feature dimension
            break
    
    def test_balanced_sampling(self):
        """Test balanced sampling functionality."""
        data_module = create_data_module(self.config)
        data_module.setup(self.csv_path)
        
        # Test with balanced sampling
        train_loader = data_module.train_dataloader(use_balanced_sampling=True)
        
        # Check that dataloader works
        for batch in train_loader:
            self.assertIn('features', batch)
            features = batch['features']
            self.assertIsInstance(features, torch.Tensor)
            if 'gene_labels' in batch:
                gene_labels = batch['gene_labels']
                self.assertIsInstance(gene_labels, torch.Tensor)
            break
    
    def test_data_consistency(self):
        """Test data consistency across splits."""
        data_module = create_data_module(self.config)
        data_module.setup(self.csv_path)
        
        # Get dataset info
        info = data_module.get_dataset_info()
        
        # Check that splits add up correctly
        total_samples = info.get('train_size', 0) + info.get('val_size', 0) + info.get('test_size', 0)
        self.assertEqual(total_samples, self.n_samples)
        
        # Check feature dimension
        self.assertEqual(info.get('feature_dim'), 18)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class TestABRAugmentations(unittest.TestCase):
    """Test ABR data augmentations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.features = torch.randn(16, 18)
        
        # Create augmentation config
        self.aug_config = {
            'noise_std': 0.1,
            'noise_probability': 0.8,
            'dropout_prob': 0.1,
            'dropout_probability': 0.4,
            'shift_magnitude': 0.1,
            'shift_probability': 0.7,
            'pipeline_probability': 1.0  # Always apply for testing
        }
        
        self.augmentations = create_augmentation_pipeline(self.aug_config)
    
    def test_gaussian_noise(self):
        """Test Gaussian noise augmentation."""
        # Apply the augmentation pipeline
        augmented = self.augmentations(self.features)
        
        # Check shape preservation
        self.assertEqual(augmented.shape, self.features.shape)
        
        # Check that data is still finite
        self.assertTrue(torch.isfinite(augmented).all())
    
    def test_feature_dropout(self):
        """Test feature dropout augmentation."""
        # Apply pipeline multiple times to check dropout behavior
        augmented = self.augmentations(self.features)
        
        # Check shape preservation
        self.assertEqual(augmented.shape, self.features.shape)
        
        # Check that data is still finite
        self.assertTrue(torch.isfinite(augmented).all())
    
    def test_magnitude_scaling(self):
        """Test magnitude scaling augmentation."""
        augmented = self.augmentations(self.features)
        
        # Check shape preservation
        self.assertEqual(augmented.shape, self.features.shape)
        
        # Check that data is still finite
        self.assertTrue(torch.isfinite(augmented).all())
    
    def test_frequency_shift(self):
        """Test frequency shift augmentation for ABR features."""
        augmented = self.augmentations(self.features)
        
        # Check shape preservation
        self.assertEqual(augmented.shape, self.features.shape)
        
        # Check that data is still finite
        self.assertTrue(torch.isfinite(augmented).all())
    
    def test_combined_augmentation(self):
        """Test combined augmentation pipeline."""
        augmented = self.augmentations(self.features)
        
        # Check shape preservation
        self.assertEqual(augmented.shape, self.features.shape)
        
        # Check that data is still finite
        self.assertTrue(torch.isfinite(augmented).all())
    
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
        
        # Create synthetic data with correct column names
        np.random.seed(42)
        data = {
            'specimen_id': [f'mouse_{i}' for i in range(50)],
            'gene_symbol': np.random.choice(['Gene1', 'Gene2', 'Gene3'], 50),
            '6kHz-evoked ABR Threshold': np.random.normal(40, 15, 50),
            '12kHz-evoked ABR Threshold': np.random.normal(45, 15, 50),
            '18kHz-evoked ABR Threshold': np.random.normal(50, 15, 50),
            '24kHz-evoked ABR Threshold': np.random.normal(55, 15, 50),
            '30kHz-evoked ABR Threshold': np.random.normal(60, 15, 50),
            'Click-evoked ABR threshold': np.random.normal(35, 15, 50),
            'age_in_weeks': np.random.randint(8, 20, 50),
            'weight': np.random.normal(25, 5, 50),
            'sex': np.random.choice(['Male', 'Female'], 50),
            'zygosity': np.random.choice(['homozygote', 'heterozygote'], 50),
            'genetic_background': np.random.choice(['C57BL/6N', 'C57BL/6J'], 50),
            'phenotyping_center': np.random.choice(['JAX', 'TCP'], 50),
            'pipeline_name': np.random.choice(['IMPC_001', 'IMPC_002'], 50),
            'metadata_Equipment manufacturer': np.random.choice(['Bioseb', 'Other'], 50)
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
        config = create_default_config()
        preprocessor = create_preprocessor(config)
        
        feature_columns = [
            '6kHz-evoked ABR Threshold',
            '12kHz-evoked ABR Threshold', 
            '18kHz-evoked ABR Threshold',
            '24kHz-evoked ABR Threshold',
            '30kHz-evoked ABR Threshold',
            'Click-evoked ABR threshold'
        ]
        
        dataset = create_abr_dataset(
            data_path=self.csv_path,
            preprocessor=preprocessor,
            feature_columns=feature_columns,
            gene_column='gene_symbol',
            mouse_id_column='specimen_id',
            mode='train'
        )
        
        # Check dataset creation
        self.assertIsInstance(dataset, IMPCABRDataset)
        self.assertEqual(len(dataset), 50)
        
        # Check sample structure
        sample = dataset[0]
        self.assertIn('features', sample)
        self.assertIn('gene_label', sample)
        self.assertIn('mouse_id', sample)
    
    def test_missing_columns_handling(self):
        """Test handling of missing columns."""
        config = create_default_config()
        preprocessor = create_preprocessor(config)
        
        # Try with non-existent file
        with self.assertRaises(FileNotFoundError):
            create_abr_dataset(
                data_path='non_existent_file.csv',
                preprocessor=preprocessor,
                feature_columns=[
                    '6kHz-evoked ABR Threshold',
                    '12kHz-evoked ABR Threshold'
                ],
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