"""
PyTorch Dataset for IMPC ABR audiometric data.

This module provides PyTorch datasets for training the ContrastiveVAE-DEC model
on audiometric phenotype data with support for contrastive learning pairs.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class IMPCABRDataset(Dataset):
    """
    PyTorch Dataset for IMPC ABR audiometric data.

    Supports both standard training and contrastive learning with data augmentation.
    """

    def __init__(self,
                 data: Union[pd.DataFrame, np.ndarray],
                 features: np.ndarray,
                 gene_labels: Optional[np.ndarray] = None,
                 mouse_ids: Optional[np.ndarray] = None,
                 metadata: Optional[pd.DataFrame] = None,
                 mode: str = 'train',
                 return_pairs: bool = False,
                 augment_fn: Optional[callable] = None):
        """
        Initialize the dataset.

        Args:
            data: Raw data (DataFrame) or preprocessed features (ndarray)
            features: Preprocessed feature matrix (n_samples, n_features)
            gene_labels: Gene labels for each sample (optional)
            mouse_ids: Mouse specimen IDs (optional)
            metadata: Additional metadata (optional)
            mode: Dataset mode ('train', 'val', 'test')
            return_pairs: Whether to return contrastive pairs
            augment_fn: Data augmentation function for contrastive learning
        """
        self.features = torch.FloatTensor(features)
        self.gene_labels = gene_labels
        self.mouse_ids = mouse_ids
        self.metadata = metadata
        self.mode = mode
        self.return_pairs = return_pairs
        self.augment_fn = augment_fn

        # Store original data if provided
        if isinstance(data, pd.DataFrame):
            self.raw_data = data
        else:
            self.raw_data = None

        # Create gene-to-indices mapping for contrastive learning
        if gene_labels is not None:
            self.gene_to_indices = self._create_gene_mapping()
        else:
            self.gene_to_indices = {}

        logger.info(f"Initialized ABR dataset: {len(self)} samples, mode='{mode}', "
                   f"return_pairs={return_pairs}")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing sample data
        """
        # Get base features
        features = self.features[idx]

        # Prepare return dictionary
        sample = {
            'features': features,
            'index': torch.tensor(idx, dtype=torch.long)
        }

        # Add gene label if available
        if self.gene_labels is not None:
            sample['gene_label'] = torch.tensor(self.gene_labels[idx], dtype=torch.long)

        # Add mouse ID if available
        if self.mouse_ids is not None:
            sample['mouse_id'] = self.mouse_ids[idx]

        # Handle contrastive pairs
        if self.return_pairs and self.mode == 'train':
            positive_sample = self._get_positive_sample(idx)
            sample['positive'] = positive_sample

        # Apply augmentation if specified
        if self.augment_fn is not None and self.mode == 'train':
            sample['augmented'] = self.augment_fn(features)

        return sample

    def _create_gene_mapping(self) -> Dict[str, List[int]]:
        """Create mapping from gene labels to sample indices."""
        gene_mapping = {}
        for idx, gene_label in enumerate(self.gene_labels):
            if gene_label not in gene_mapping:
                gene_mapping[gene_label] = []
            gene_mapping[gene_label].append(idx)
        return gene_mapping

    def _get_positive_sample(self, idx: int) -> torch.Tensor:
        """
        Get a positive sample for contrastive learning.

        Positive samples are from the same gene knockout line.

        Args:
            idx: Index of anchor sample

        Returns:
            Features of positive sample
        """
        if self.gene_labels is None:
            # If no gene labels, return augmented version of same sample
            return self.augment_fn(self.features[idx]) if self.augment_fn else self.features[idx]

        # Get gene label for anchor sample
        gene_label = self.gene_labels[idx]

        # Get all samples from same gene
        same_gene_indices = self.gene_to_indices.get(gene_label, [idx])

        # Remove anchor sample from candidates
        candidates = [i for i in same_gene_indices if i != idx]

        if len(candidates) == 0:
            # If no other samples from same gene, return augmented anchor
            return self.augment_fn(self.features[idx]) if self.augment_fn else self.features[idx]

        # Randomly select a positive sample
        pos_idx = np.random.choice(candidates)
        positive_features = self.features[pos_idx]

        # Apply augmentation to positive sample
        if self.augment_fn is not None:
            positive_features = self.augment_fn(positive_features)

        return positive_features

    def get_gene_distribution(self) -> Dict[str, int]:
        """Get distribution of samples per gene."""
        if self.gene_labels is None:
            return {}

        unique, counts = np.unique(self.gene_labels, return_counts=True)
        return dict(zip(unique, counts))

    def get_feature_stats(self) -> Dict[str, float]:
        """Get basic statistics about features."""
        features_np = self.features.numpy()
        return {
            'mean': features_np.mean(),
            'std': features_np.std(),
            'min': features_np.min(),
            'max': features_np.max(),
            'shape': features_np.shape
        }

    def split_by_gene(self, train_ratio: float = 0.8,
                     val_ratio: float = 0.1) -> Tuple['IMPCABRDataset', 'IMPCABRDataset', 'IMPCABRDataset']:
        """
        Split dataset by genes to ensure no gene leakage between splits.

        Args:
            train_ratio: Proportion of genes for training
            val_ratio: Proportion of genes for validation

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if self.gene_labels is None:
            raise ValueError("Cannot split by gene without gene labels")

        # Get unique genes
        unique_genes = np.unique(self.gene_labels)
        n_genes = len(unique_genes)

        # Shuffle genes
        np.random.shuffle(unique_genes)

        # Split genes
        n_train = int(n_genes * train_ratio)
        n_val = int(n_genes * val_ratio)

        train_genes = set(unique_genes[:n_train])
        val_genes = set(unique_genes[n_train:n_train + n_val])
        test_genes = set(unique_genes[n_train + n_val:])

        # Create indices for each split
        train_indices = [i for i, gene in enumerate(self.gene_labels) if gene in train_genes]
        val_indices = [i for i, gene in enumerate(self.gene_labels) if gene in val_genes]
        test_indices = [i for i, gene in enumerate(self.gene_labels) if gene in test_genes]

        # Create datasets
        train_dataset = self._create_subset(train_indices, 'train')
        val_dataset = self._create_subset(val_indices, 'val')
        test_dataset = self._create_subset(test_indices, 'test')

        logger.info(f"Split dataset by genes: train={len(train_dataset)} ({len(train_genes)} genes), "
                   f"val={len(val_dataset)} ({len(val_genes)} genes), "
                   f"test={len(test_dataset)} ({len(test_genes)} genes)")

        return train_dataset, val_dataset, test_dataset

    def _create_subset(self, indices: List[int], mode: str) -> 'IMPCABRDataset':
        """Create a subset dataset with given indices."""
        subset_features = self.features[indices]
        subset_gene_labels = self.gene_labels[indices] if self.gene_labels is not None else None
        subset_mouse_ids = self.mouse_ids[indices] if self.mouse_ids is not None else None
        subset_metadata = self.metadata.iloc[indices] if self.metadata is not None else None

        return IMPCABRDataset(
            data=None,
            features=subset_features.numpy(),
            gene_labels=subset_gene_labels,
            mouse_ids=subset_mouse_ids,
            metadata=subset_metadata,
            mode=mode,
            return_pairs=self.return_pairs if mode == 'train' else False,
            augment_fn=self.augment_fn if mode == 'train' else None
        )


class ContrastiveABRDataset(IMPCABRDataset):
    """
    Specialized dataset for contrastive learning with ABR data.

    Always returns positive pairs and supports multiple augmentation strategies.
    """

    def __init__(self, *args, **kwargs):
        """Initialize contrastive dataset."""
        kwargs['return_pairs'] = True
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get contrastive sample pair.

        Returns:
            Dictionary with anchor, positive, and metadata
        """
        # Get base sample
        sample = super().__getitem__(idx)

        # Ensure we have positive sample
        if 'positive' not in sample:
            sample['positive'] = self._get_positive_sample(idx)

        # Apply augmentation to anchor as well
        if self.augment_fn is not None:
            sample['anchor_augmented'] = self.augment_fn(sample['features'])

        return sample


def create_abr_dataset(data_path: str,
                      preprocessor: callable,
                      feature_columns: List[str],
                      gene_column: str = 'gene_symbol',
                      mouse_id_column: str = 'specimen_id',
                      mode: str = 'train',
                      **kwargs) -> IMPCABRDataset:
    """
    Factory function to create ABR dataset from file.

    Args:
        data_path: Path to data file
        preprocessor: Preprocessing function/pipeline
        feature_columns: List of feature column names
        gene_column: Name of gene label column
        mouse_id_column: Name of mouse ID column
        mode: Dataset mode
        **kwargs: Additional arguments for dataset

    Returns:
        Configured ABR dataset
    """
    logger.info(f"Creating ABR dataset from {data_path}")

    # Load data
    data = pd.read_csv(data_path)

    # Preprocess features using the full DataFrame
    processed_features = preprocessor.fit_transform(data)

    # Extract gene labels if available
    gene_labels = None
    if gene_column in data.columns:
        # Convert gene symbols to numeric labels
        unique_genes = data[gene_column].unique()
        gene_to_id = {gene: i for i, gene in enumerate(unique_genes)}
        gene_labels = data[gene_column].map(gene_to_id).values

    # Extract mouse IDs
    mouse_ids = None
    if mouse_id_column in data.columns:
        mouse_ids = data[mouse_id_column].values

    # Create dataset
    dataset = IMPCABRDataset(
        data=data,
        features=processed_features,
        gene_labels=gene_labels,
        mouse_ids=mouse_ids,
        metadata=data,
        mode=mode,
        **kwargs
    )

    return dataset