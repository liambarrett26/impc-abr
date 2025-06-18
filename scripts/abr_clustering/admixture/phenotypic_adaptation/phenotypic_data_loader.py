"""
Phenotypic data loader adapted for Neural ADMIXTURE.

This module replaces the genomic data loading components with phenotypic data handling,
maintaining compatibility with the Neural ADMIXTURE training pipeline.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PhenotypicDataset(Dataset):
    """
    Dataset for phenotypic data compatible with Neural ADMIXTURE.

    Replaces the genomic Dataset_admixture with phenotypic data handling.
    """

    def __init__(self, features: torch.Tensor, gene_labels: Optional[torch.Tensor] = None,
                 metadata: Optional[Dict[str, torch.Tensor]] = None):
        """
        Initialize phenotypic dataset.

        Args:
            features: Tensor of shape (n_samples, n_features) with engineered features
            gene_labels: Optional tensor of gene labels for supervised learning
            metadata: Optional dictionary with additional metadata tensors
        """
        self.features = features
        self.gene_labels = gene_labels if gene_labels is not None else torch.zeros(features.size(0))
        self.metadata = metadata or {}

        # Validate dimensions
        if len(self.features.shape) != 2:
            raise ValueError(f"Features must be 2D tensor, got shape {self.features.shape}")

        if self.gene_labels.size(0) != self.features.size(0):
            raise ValueError(f"Gene labels size {self.gene_labels.size(0)} doesn't match features {self.features.size(0)}")

    def __len__(self) -> int:
        """Return number of samples."""
        return self.features.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sample at index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features, gene_labels)
        """
        return self.features[idx], self.gene_labels[idx]

    def get_feature_dim(self) -> int:
        """Get number of features."""
        return self.features.size(1)

    def get_metadata(self, key: str) -> Optional[torch.Tensor]:
        """Get metadata tensor by key."""
        return self.metadata.get(key)


def dataloader_phenotypic(features: torch.Tensor, batch_size: int, num_gpus: int,
                         seed: int, generator: torch.Generator,
                         gene_labels: Optional[torch.Tensor] = None,
                         shuffle: bool = True,
                         metadata: Optional[Dict[str, torch.Tensor]] = None) -> DataLoader:
    """
    Create DataLoader for phenotypic data.

    Args:
        features: Feature tensor (n_samples, n_features)
        batch_size: Batch size
        num_gpus: Number of GPUs for distributed training
        seed: Random seed
        generator: Random generator
        gene_labels: Optional gene labels
        shuffle: Whether to shuffle data
        metadata: Optional metadata dictionary

    Returns:
        DataLoader for phenotypic data
    """
    dataset = PhenotypicDataset(features, gene_labels, metadata)

    if num_gpus > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed)
    else:
        if shuffle:
            sampler = RandomSampler(dataset, generator=generator)
        else:
            sampler = SequentialSampler(dataset)

    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size,
                       pin_memory=True if torch.cuda.is_available() else False)

    return loader


class PhenotypicDataProcessor:
    """
    Data processor for integrating with existing IMPC preprocessing pipeline.

    Bridges the gap between your ABR preprocessing and Neural ADMIXTURE requirements.
    """

    def __init__(self, device: torch.device = None):
        """
        Initialize data processor.

        Args:
            device: Target device for tensors
        """
        self.device = device or torch.device('cpu')

    def prepare_data_for_neural_admixture(self,
                                        features: np.ndarray,
                                        gene_symbols: Optional[np.ndarray] = None,
                                        center_info: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Convert numpy arrays to tensors compatible with Neural ADMIXTURE.

        Args:
            features: Engineered features from FeatureEngineer
            gene_symbols: Array of gene symbols for each sample
            center_info: Array of center information for each sample

        Returns:
            Tuple of (feature_tensor, gene_label_tensor, metadata_dict)
        """
        # Convert features to tensor
        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)

        # Handle gene labels
        gene_label_tensor = None
        if gene_symbols is not None:
            # Handle potential NaN/None values
            gene_symbols_clean = np.array([str(g) if g is not None and pd.notna(g) else 'unknown'
                                         for g in gene_symbols])

            # Convert gene symbols to numerical labels
            unique_genes = np.unique(gene_symbols_clean)
            gene_to_idx = {gene: idx for idx, gene in enumerate(unique_genes)}
            gene_indices = np.array([gene_to_idx[gene] for gene in gene_symbols_clean])
            gene_label_tensor = torch.as_tensor(gene_indices, dtype=torch.long, device=self.device)

        # Prepare metadata
        metadata = {}
        if center_info is not None:
            # Handle potential NaN/None values and convert to strings first
            center_info_clean = np.array([str(c) if c is not None and pd.notna(c) else 'unknown'
                                        for c in center_info])

            # Convert to numerical labels
            unique_centers = np.unique(center_info_clean)
            center_to_idx = {center: idx for idx, center in enumerate(unique_centers)}
            center_indices = np.array([center_to_idx[center] for center in center_info_clean])
            center_tensor = torch.as_tensor(center_indices, dtype=torch.long, device=self.device)
            metadata['center_info'] = center_tensor
            metadata['center_mapping'] = {str(k): v for k, v in center_to_idx.items()}

        logger.info(f"Prepared data: {feature_tensor.shape[0]} samples, {feature_tensor.shape[1]} features")
        if gene_label_tensor is not None:
            logger.info(f"Gene labels: {len(unique_genes)} unique genes")
        if center_info is not None:
            logger.info(f"Centers: {len(unique_centers)} unique centers")

        return feature_tensor, gene_label_tensor, metadata

    def create_validation_split(self, features: torch.Tensor,
                              gene_labels: Optional[torch.Tensor] = None,
                              val_fraction: float = 0.2,
                              seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Create train/validation split.

        Args:
            features: Feature tensor
            gene_labels: Optional gene labels
            val_fraction: Fraction for validation
            seed: Random seed

        Returns:
            Tuple of (train_features, val_features, train_labels, val_labels)
        """
        torch.manual_seed(seed)
        n_samples = features.size(0)
        indices = torch.randperm(n_samples)

        val_size = int(n_samples * val_fraction)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        train_features = features[train_indices]
        val_features = features[val_indices]

        train_labels = gene_labels[train_indices] if gene_labels is not None else None
        val_labels = gene_labels[val_indices] if gene_labels is not None else None

        logger.info(f"Created splits: train={len(train_indices)}, val={len(val_indices)}")

        return train_features, val_features, train_labels, val_labels


def validate_phenotypic_data(features: np.ndarray,
                           feature_names: list,
                           expected_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> bool:
    """
    Validate phenotypic data for Neural ADMIXTURE compatibility.

    Args:
        features: Feature array
        feature_names: List of feature names
        expected_ranges: Optional dictionary of expected value ranges

    Returns:
        True if validation passes
    """
    logger.info("Validating phenotypic data...")

    # Check basic properties
    if len(features.shape) != 2:
        raise ValueError(f"Features must be 2D array, got shape {features.shape}")

    if features.shape[1] != len(feature_names):
        raise ValueError(f"Feature count mismatch: {features.shape[1]} vs {len(feature_names)}")

    # Check for NaN or infinite values
    if np.any(np.isnan(features)):
        nan_count = np.sum(np.isnan(features))
        logger.warning(f"Found {nan_count} NaN values in features")

    if np.any(np.isinf(features)):
        inf_count = np.sum(np.isinf(features))
        logger.warning(f"Found {inf_count} infinite values in features")

    # Check value ranges
    for i, name in enumerate(feature_names):
        col_data = features[:, i]
        col_min, col_max = np.min(col_data), np.max(col_data)

        logger.debug(f"Feature {name}: range [{col_min:.3f}, {col_max:.3f}]")

        if expected_ranges and name in expected_ranges:
            expected_min, expected_max = expected_ranges[name]
            if col_min < expected_min or col_max > expected_max:
                logger.warning(f"Feature {name} outside expected range [{expected_min}, {expected_max}]")

    # Check for constant features
    constant_features = []
    for i, name in enumerate(feature_names):
        if np.std(features[:, i]) < 1e-8:
            constant_features.append(name)

    if constant_features:
        logger.warning(f"Found {len(constant_features)} constant features: {constant_features}")

    logger.info("Data validation complete")
    return True


# Integration function to replace genomic data loading
def load_phenotypic_data_for_neural_admixture(df: pd.DataFrame,
                                             abr_normalized: np.ndarray,
                                             feature_engineer,
                                             device: torch.device) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """
    Complete pipeline to load and prepare phenotypic data for Neural ADMIXTURE.

    Args:
        df: Raw dataframe from IMPC
        abr_normalized: Normalized ABR data from ABRPreprocessor
        feature_engineer: Fitted FeatureEngineer instance
        device: Target device

    Returns:
        Tuple of (features, gene_labels, metadata) ready for Neural ADMIXTURE
    """
    # Engineer features
    engineered_features = feature_engineer.transform(df, abr_normalized)

    # Validate data
    feature_names = feature_engineer.get_feature_names()
    validate_phenotypic_data(engineered_features, feature_names)

    # Prepare for Neural ADMIXTURE
    processor = PhenotypicDataProcessor(device)

    gene_symbols = df['gene_symbol'].values if 'gene_symbol' in df.columns else None
    center_info = df['phenotyping_center'].values if 'phenotyping_center' in df.columns else None

    features, gene_labels, metadata = processor.prepare_data_for_neural_admixture(
        engineered_features, gene_symbols, center_info
    )

    # Add feature names to metadata
    metadata['feature_names'] = feature_names
    metadata['n_features'] = len(feature_names)

    return features, gene_labels, metadata