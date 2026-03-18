"""
Custom dataloaders with specialized sampling strategies for ContrastiveVAE-DEC.

This module provides PyTorch dataloaders with balanced sampling and
gene-aware batching strategies for audiometric phenotype discovery.
"""

import torch
from torch.utils.data import DataLoader, Sampler
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Iterator, Union
import logging
from collections import defaultdict, Counter
import random

from .dataset import IMPCABRDataset, ContrastiveABRDataset
from .preprocessor import ABRFeaturePreprocessor, create_preprocessor

logger = logging.getLogger(__name__)


class BalancedGeneSampler(Sampler):
    """
    Sampler that ensures balanced representation of genes in each batch.

    Useful for contrastive learning where we want diverse gene representation
    to create meaningful positive/negative pairs.
    """

    def __init__(self, dataset: IMPCABRDataset,
                 batch_size: int = 32,
                 samples_per_gene: Optional[int] = None):
        """
        Initialize balanced gene sampler.

        Args:
            dataset: ABR dataset with gene labels
            batch_size: Desired batch size
            samples_per_gene: Maximum samples per gene per batch
        """
        self.dataset = dataset
        self.batch_size = batch_size

        if dataset.gene_labels is None:
            raise ValueError("BalancedGeneSampler requires gene labels")

        # Create gene to indices mapping
        self.gene_to_indices = defaultdict(list)
        for idx, gene_label in enumerate(dataset.gene_labels):
            self.gene_to_indices[gene_label].append(idx)

        self.genes = list(self.gene_to_indices.keys())
        self.n_genes = len(self.genes)

        # Set samples per gene per batch
        if samples_per_gene is None:
            self.samples_per_gene = max(1, batch_size // (self.n_genes + 1))
        else:
            self.samples_per_gene = samples_per_gene

        # Calculate effective batch size
        self.effective_batch_size = min(batch_size, self.samples_per_gene * self.n_genes)

        logger.info(f"BalancedGeneSampler: {self.n_genes} genes, "
                   f"{self.samples_per_gene} samples/gene, "
                   f"effective batch size: {self.effective_batch_size}")

    def __iter__(self) -> Iterator[int]:
        """Generate balanced batches."""
        # Shuffle genes and create batches
        n_batches = len(self.dataset) // self.effective_batch_size

        for _ in range(n_batches):
            batch_indices = []

            # Sample from each gene
            for gene in self.genes:
                gene_indices = self.gene_to_indices[gene]

                # Sample up to samples_per_gene from this gene
                n_samples = min(self.samples_per_gene, len(gene_indices))
                sampled_indices = random.sample(gene_indices, n_samples)
                batch_indices.extend(sampled_indices)

            # Shuffle batch and ensure correct size
            random.shuffle(batch_indices)
            batch_indices = batch_indices[:self.effective_batch_size]

            yield from batch_indices

    def __len__(self) -> int:
        """Return number of samples per epoch."""
        n_batches = len(self.dataset) // self.effective_batch_size
        return n_batches * self.effective_batch_size


class ContrastiveSampler(Sampler):
    """
    Sampler optimized for contrastive learning.

    Ensures each batch contains multiple samples from the same genes
    to create positive pairs, while maintaining diversity.
    """

    def __init__(self, dataset: IMPCABRDataset,
                 batch_size: int = 32,
                 min_samples_per_gene: int = 2):
        """
        Initialize contrastive sampler.

        Args:
            dataset: ABR dataset with gene labels
            batch_size: Batch size
            min_samples_per_gene: Minimum samples per gene in batch for contrastive learning
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_samples_per_gene = min_samples_per_gene

        if dataset.gene_labels is None:
            raise ValueError("ContrastiveSampler requires gene labels")

        # Create gene mappings
        self.gene_to_indices = defaultdict(list)
        for idx, gene_label in enumerate(dataset.gene_labels):
            self.gene_to_indices[gene_label].append(idx)

        # Filter genes with sufficient samples
        self.viable_genes = [
            gene for gene, indices in self.gene_to_indices.items()
            if len(indices) >= min_samples_per_gene
        ]

        logger.info(f"ContrastiveSampler: {len(self.viable_genes)} viable genes "
                   f"(min {min_samples_per_gene} samples each)")

    def __iter__(self) -> Iterator[int]:
        """Generate contrastive batches."""
        n_batches = len(self.dataset) // self.batch_size

        for _ in range(n_batches):
            batch_indices = []

            # Determine how many genes to include in this batch
            genes_per_batch = min(
                len(self.viable_genes),
                self.batch_size // self.min_samples_per_gene
            )

            # Randomly select genes for this batch
            selected_genes = random.sample(self.viable_genes, genes_per_batch)

            # Sample from each selected gene
            samples_per_gene = self.batch_size // genes_per_batch

            for gene in selected_genes:
                gene_indices = self.gene_to_indices[gene]
                n_samples = min(samples_per_gene, len(gene_indices))
                sampled_indices = random.sample(gene_indices, n_samples)
                batch_indices.extend(sampled_indices)

            # Fill remaining slots randomly if needed
            while len(batch_indices) < self.batch_size:
                remaining_indices = [
                    idx for idx in range(len(self.dataset))
                    if idx not in batch_indices
                ]
                if remaining_indices:
                    batch_indices.append(random.choice(remaining_indices))
                else:
                    break

            # Shuffle and ensure correct size
            random.shuffle(batch_indices)
            batch_indices = batch_indices[:self.batch_size]

            yield from batch_indices

    def __len__(self) -> int:
        """Return number of samples per epoch."""
        n_batches = len(self.dataset) // self.batch_size
        return n_batches * self.batch_size


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for ABR data.

    Handles variable-length metadata and optional fields.

    Args:
        batch: List of sample dictionaries

    Returns:
        Collated batch dictionary
    """
    # Stack standard features
    features = torch.stack([sample['features'] for sample in batch])
    indices = torch.stack([sample['index'] for sample in batch])

    collated = {
        'features': features,
        'index': indices
    }

    # Handle optional gene labels
    if 'gene_label' in batch[0]:
        gene_labels = torch.stack([sample['gene_label'] for sample in batch])
        collated['gene_label'] = gene_labels

    # Handle contrastive pairs
    if 'positive' in batch[0]:
        positives = torch.stack([sample['positive'] for sample in batch])
        collated['positive'] = positives

    # Handle augmented features
    if 'augmented' in batch[0]:
        augmented = torch.stack([sample['augmented'] for sample in batch])
        collated['augmented'] = augmented

    if 'anchor_augmented' in batch[0]:
        anchor_augmented = torch.stack([sample['anchor_augmented'] for sample in batch])
        collated['anchor_augmented'] = anchor_augmented

    # Handle mouse IDs (keep as list since they might be strings)
    if 'mouse_id' in batch[0]:
        mouse_ids = [sample['mouse_id'] for sample in batch]
        collated['mouse_id'] = mouse_ids

    return collated


class ABRDataModule:
    """
    Data module that handles loading, preprocessing, and dataloader creation.

    Provides a unified interface for data handling in the ContrastiveVAE-DEC pipeline.
    """

    def __init__(self, config: Dict):
        """
        Initialize data module.

        Args:
            config: Data configuration dictionary
        """
        self.config = config
        self.preprocessor = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, data_path: str, stage: Optional[str] = None):
        """
        Setup datasets and preprocessing.

        Args:
            data_path: Path to data file
            stage: Setup stage ('fit', 'test', or None for all)
        """
        logger.info(f"Setting up data module from {data_path}")

        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples")

        # Initialize and fit preprocessor
        self.preprocessor = create_preprocessor()
        processed_features = self.preprocessor.fit_transform(df)

        # Extract gene labels and mouse IDs
        gene_labels = None
        if 'gene_symbol' in df.columns:
            unique_genes = df['gene_symbol'].dropna().unique()
            gene_to_id = {gene: i for i, gene in enumerate(unique_genes)}
            gene_labels = df['gene_symbol'].map(gene_to_id).fillna(-1).astype(int).values

        mouse_ids = df['specimen_id'].values if 'specimen_id' in df.columns else None

        # Create full dataset
        full_dataset = IMPCABRDataset(
            data=df,
            features=processed_features,
            gene_labels=gene_labels,
            mouse_ids=mouse_ids,
            metadata=df
        )

        # Split datasets
        if stage == 'fit' or stage is None:
            if gene_labels is not None:
                # Split by genes to prevent leakage
                self.train_dataset, self.val_dataset, self.test_dataset = \
                    full_dataset.split_by_gene(
                        train_ratio=self.config.get('train_split', 0.8),
                        val_ratio=self.config.get('val_split', 0.1)
                    )
            else:
                # Random split if no gene labels
                n_train = int(len(full_dataset) * self.config.get('train_split', 0.8))
                n_val = int(len(full_dataset) * self.config.get('val_split', 0.1))

                indices = np.random.permutation(len(full_dataset))
                train_indices = indices[:n_train]
                val_indices = indices[n_train:n_train + n_val]
                test_indices = indices[n_train + n_val:]

                self.train_dataset = full_dataset._create_subset(train_indices, 'train')
                self.val_dataset = full_dataset._create_subset(val_indices, 'val')
                self.test_dataset = full_dataset._create_subset(test_indices, 'test')

        if stage == 'test':
            self.test_dataset = full_dataset

    def train_dataloader(self, use_contrastive: bool = False,
                        use_balanced_sampling: bool = False) -> DataLoader:
        """
        Create training dataloader.

        Args:
            use_contrastive: Whether to use contrastive learning setup
            use_balanced_sampling: Whether to use balanced gene sampling

        Returns:
            Training dataloader
        """
        if self.train_dataset is None:
            raise ValueError("Must call setup() before creating dataloaders")

        # Convert to contrastive dataset if needed
        if use_contrastive:
            dataset = ContrastiveABRDataset(
                data=self.train_dataset.raw_data,
                features=self.train_dataset.features.numpy(),
                gene_labels=self.train_dataset.gene_labels,
                mouse_ids=self.train_dataset.mouse_ids,
                metadata=self.train_dataset.metadata,
                mode='train'
            )
        else:
            dataset = self.train_dataset

        # Choose sampler
        sampler = None
        shuffle = True

        if use_balanced_sampling and dataset.gene_labels is not None:
            sampler = BalancedGeneSampler(dataset, self.config.get('batch_size', 512))
            shuffle = False
        elif use_contrastive and dataset.gene_labels is not None:
            sampler = ContrastiveSampler(dataset, self.config.get('batch_size', 512))
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 512),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', True),
            collate_fn=collate_fn,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        if self.val_dataset is None:
            raise ValueError("Must call setup() before creating dataloaders")

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.get('batch_size', 512),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', True),
            collate_fn=collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        if self.test_dataset is None:
            raise ValueError("Must call setup() before creating dataloaders")

        return DataLoader(
            self.test_dataset,
            batch_size=self.config.get('batch_size', 512),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', True),
            collate_fn=collate_fn
        )

    def get_dataset_info(self) -> Dict[str, Union[int, float]]:
        """Get information about the datasets."""
        info = {}

        if self.train_dataset:
            info['train_size'] = len(self.train_dataset)
            if hasattr(self.train_dataset, 'get_gene_distribution'):
                train_gene_dist = self.train_dataset.get_gene_distribution()
                info['train_genes'] = len(train_gene_dist)

        if self.val_dataset:
            info['val_size'] = len(self.val_dataset)

        if self.test_dataset:
            info['test_size'] = len(self.test_dataset)

        if self.preprocessor:
            info['feature_dim'] = 18  # Fixed for our architecture

        return info


def create_data_module(config: Dict) -> ABRDataModule:
    """
    Factory function to create data module.

    Args:
        config: Data configuration

    Returns:
        Configured data module
    """
    return ABRDataModule(config)