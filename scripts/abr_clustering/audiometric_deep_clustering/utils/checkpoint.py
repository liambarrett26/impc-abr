"""
Model checkpointing utilities for ContrastiveVAE-DEC training.

This module provides comprehensive model saving and loading functionality
with state preservation, best model tracking, and recovery capabilities.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
import json
import shutil
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class ModelCheckpoint:
    """
    Comprehensive model checkpointing with automatic best model tracking.
    
    Handles saving model state, optimizer state, training metrics, and
    configuration for complete training state recovery.
    """
    
    def __init__(self,
                 checkpoint_dir: Union[str, Path],
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_top_k: int = 3,
                 save_last: bool = True,
                 save_every_n_epochs: Optional[int] = None,
                 filename_pattern: str = 'epoch_{epoch:03d}_{monitor:.4f}'):
        """
        Initialize model checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor for best model selection
            mode: 'min' or 'max' for best model selection
            save_top_k: Number of best checkpoints to keep
            save_last: Whether to always save the last epoch
            save_every_n_epochs: Save checkpoint every N epochs
            filename_pattern: Pattern for checkpoint filenames
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.save_every_n_epochs = save_every_n_epochs
        self.filename_pattern = filename_pattern
        
        # Track best models
        self.best_k_models = []  # List of (score, filepath) tuples
        self.best_model_score = float('inf') if mode == 'min' else float('-inf')
        self.best_model_path = None
        
        # State tracking
        self.epoch_count = 0
        self.last_checkpoint_path = None
        
        logger.info(f"ModelCheckpoint initialized: monitor={monitor}, mode={mode}, "
                   f"save_top_k={save_top_k}, dir={checkpoint_dir}")
    
    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       epoch: int = 0,
                       metrics: Optional[Dict[str, float]] = None,
                       config: Optional[Dict[str, Any]] = None,
                       extra_state: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """
        Save model checkpoint with complete training state.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state to save
            scheduler: Learning rate scheduler state (optional)
            epoch: Current epoch number
            metrics: Training/validation metrics
            config: Model configuration
            extra_state: Additional state to save
            
        Returns:
            Path to saved checkpoint (None if not saved)
        """
        self.epoch_count = epoch
        metrics = metrics or {}
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__
        }
        
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        if config is not None:
            checkpoint_data['config'] = config
        
        if extra_state is not None:
            checkpoint_data['extra_state'] = extra_state
        
        # Determine if we should save this checkpoint
        should_save = False
        monitor_value = metrics.get(self.monitor)
        
        # Check if this is a best model
        is_best = False
        if monitor_value is not None:
            if self.mode == 'min':
                is_best = monitor_value < self.best_model_score
            else:
                is_best = monitor_value > self.best_model_score
            
            if is_best:
                self.best_model_score = monitor_value
                should_save = True
        
        # Check other save conditions
        if self.save_every_n_epochs and epoch % self.save_every_n_epochs == 0:
            should_save = True
        
        if self.save_last:
            should_save = True
        
        if not should_save:
            return None
        
        # Generate filename
        if monitor_value is not None:
            filename = self.filename_pattern.format(
                epoch=epoch,
                monitor=monitor_value,
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            )
        else:
            filename = f'epoch_{epoch:03d}'
        
        filename += '.ckpt'
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Update best model tracking
            if monitor_value is not None:
                self._update_best_models(monitor_value, checkpoint_path)
            
            # Save as best model if applicable
            if is_best:
                self.best_model_path = checkpoint_path
                best_path = self.checkpoint_dir / 'best_model.ckpt'
                shutil.copy2(checkpoint_path, best_path)
                logger.info(f"New best model saved: {best_path}")
            
            # Save as last model
            if self.save_last:
                last_path = self.checkpoint_dir / 'last_model.ckpt'
                shutil.copy2(checkpoint_path, last_path)
                self.last_checkpoint_path = last_path
            
            # Clean up old checkpoints
            self._cleanup_checkpoints()
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None
    
    def _update_best_models(self, score: float, checkpoint_path: Path):
        """Update the list of best models."""
        self.best_k_models.append((score, checkpoint_path))
        
        # Sort by score
        if self.mode == 'min':
            self.best_k_models.sort(key=lambda x: x[0])
        else:
            self.best_k_models.sort(key=lambda x: x[0], reverse=True)
        
        # Keep only top k
        if len(self.best_k_models) > self.save_top_k:
            # Remove the worst checkpoint file
            _, worst_path = self.best_k_models.pop()
            if worst_path.exists() and worst_path != self.last_checkpoint_path:
                try:
                    worst_path.unlink()
                    logger.info(f"Removed old checkpoint: {worst_path}")
                except Exception as e:
                    logger.warning(f"Could not remove checkpoint {worst_path}: {e}")
    
    def _cleanup_checkpoints(self):
        """Clean up old checkpoint files."""
        # Get all checkpoint files
        checkpoint_files = list(self.checkpoint_dir.glob('epoch_*.ckpt'))
        
        # Keep best models and last model
        protected_files = set()
        protected_files.update([path for _, path in self.best_k_models])
        
        if self.last_checkpoint_path:
            protected_files.add(self.last_checkpoint_path)
        
        if self.best_model_path:
            protected_files.add(self.best_model_path)
        
        # Add special checkpoint names
        protected_files.add(self.checkpoint_dir / 'best_model.ckpt')
        protected_files.add(self.checkpoint_dir / 'last_model.ckpt')
        
        # Remove unprotected files
        for file_path in checkpoint_files:
            if file_path not in protected_files:
                try:
                    file_path.unlink()
                    logger.debug(f"Cleaned up old checkpoint: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not remove checkpoint {file_path}: {e}")
    
    def load_best_checkpoint(self) -> Optional[Path]:
        """
        Get path to best checkpoint.
        
        Returns:
            Path to best checkpoint file
        """
        best_path = self.checkpoint_dir / 'best_model.ckpt'
        if best_path.exists():
            return best_path
        elif self.best_model_path and self.best_model_path.exists():
            return self.best_model_path
        else:
            logger.warning("No best checkpoint found")
            return None
    
    def load_last_checkpoint(self) -> Optional[Path]:
        """
        Get path to last checkpoint.
        
        Returns:
            Path to last checkpoint file
        """
        last_path = self.checkpoint_dir / 'last_model.ckpt'
        if last_path.exists():
            return last_path
        elif self.last_checkpoint_path and self.last_checkpoint_path.exists():
            return self.last_checkpoint_path
        else:
            logger.warning("No last checkpoint found")
            return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with metadata.
        
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []
        
        for ckpt_file in self.checkpoint_dir.glob('*.ckpt'):
            try:
                # Load minimal metadata
                checkpoint = torch.load(ckpt_file, map_location='cpu')
                
                checkpoints.append({
                    'path': ckpt_file,
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'metrics': checkpoint.get('metrics', {}),
                    'timestamp': checkpoint.get('timestamp', 'unknown'),
                    'size_mb': ckpt_file.stat().st_size / (1024 * 1024)
                })
            except Exception as e:
                logger.warning(f"Could not read checkpoint {ckpt_file}: {e}")
        
        # Sort by epoch
        checkpoints.sort(key=lambda x: x['epoch'] if isinstance(x['epoch'], int) else 0)
        return checkpoints


def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   epoch: int = 0,
                   metrics: Optional[Dict[str, float]] = None,
                   config: Optional[Dict[str, Any]] = None,
                   filepath: Union[str, Path] = 'checkpoint.ckpt',
                   extra_state: Optional[Dict[str, Any]] = None) -> None:
    """
    Save a single checkpoint to specified path.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler (optional)
        epoch: Current epoch
        metrics: Training metrics
        config: Model configuration
        filepath: Path to save checkpoint
        extra_state: Additional state to save
    """
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics or {},
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__
    }
    
    if scheduler is not None:
        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
    
    if config is not None:
        checkpoint_data['config'] = config
    
    if extra_state is not None:
        checkpoint_data['extra_state'] = extra_state
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        torch.save(checkpoint_data, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {filepath}: {e}")
        raise


def load_checkpoint(model: nn.Module,
                   filepath: Union[str, Path],
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   map_location: Optional[Union[str, torch.device]] = None,
                   strict: bool = True) -> Dict[str, Any]:
    """
    Load checkpoint and restore model state.
    
    Args:
        model: PyTorch model to load state into
        filepath: Path to checkpoint file
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        map_location: Device mapping for loading
        strict: Whether to strictly enforce state dict matching
        
    Returns:
        Dictionary containing checkpoint metadata
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    try:
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint['model_state_dict'], strict=strict
            )
            
            if missing_keys and strict:
                logger.warning(f"Missing keys in model state dict: {missing_keys}")
            if unexpected_keys and strict:
                logger.warning(f"Unexpected keys in model state dict: {unexpected_keys}")
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Optimizer state loaded")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Scheduler state loaded")
            except Exception as e:
                logger.warning(f"Failed to load scheduler state: {e}")
        
        logger.info(f"Checkpoint loaded from {filepath}")
        
        # Return metadata
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', {}),
            'extra_state': checkpoint.get('extra_state', {}),
            'timestamp': checkpoint.get('timestamp'),
            'pytorch_version': checkpoint.get('pytorch_version')
        }
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {filepath}: {e}")
        raise


def create_checkpoint_callback(checkpoint_dir: Union[str, Path],
                             monitor: str = 'val_loss',
                             mode: str = 'min',
                             save_top_k: int = 3) -> ModelCheckpoint:
    """
    Create a ModelCheckpoint callback for training.
    
    Args:
        checkpoint_dir: Directory for checkpoints
        monitor: Metric to monitor
        mode: 'min' or 'max'
        save_top_k: Number of best models to keep
        
    Returns:
        Configured ModelCheckpoint instance
    """
    return ModelCheckpoint(
        checkpoint_dir=checkpoint_dir,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=True
    )


def find_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """
    Find the most recent checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory to search
        
    Returns:
        Path to latest checkpoint (None if not found)
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Look for last_model.ckpt first
    last_model = checkpoint_dir / 'last_model.ckpt'
    if last_model.exists():
        return last_model
    
    # Find the most recent checkpoint file
    checkpoint_files = list(checkpoint_dir.glob('*.ckpt'))
    if not checkpoint_files:
        return None
    
    # Sort by modification time
    latest_file = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    return latest_file


def cleanup_old_checkpoints(checkpoint_dir: Union[str, Path],
                          keep_best: int = 3,
                          keep_recent: int = 5) -> None:
    """
    Clean up old checkpoint files, keeping only the most important ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_best: Number of best models to keep
        keep_recent: Number of recent models to keep
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Get all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob('epoch_*.ckpt'))
    
    # Always protect special files
    protected_files = {
        checkpoint_dir / 'best_model.ckpt',
        checkpoint_dir / 'last_model.ckpt'
    }
    
    # Sort by modification time (most recent first)
    checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Protect recent files
    protected_files.update(checkpoint_files[:keep_recent])
    
    # Remove old files
    removed_count = 0
    for file_path in checkpoint_files[keep_recent:]:
        if file_path not in protected_files:
            try:
                file_path.unlink()
                removed_count += 1
                logger.info(f"Removed old checkpoint: {file_path}")
            except Exception as e:
                logger.warning(f"Could not remove checkpoint {file_path}: {e}")
    
    logger.info(f"Cleaned up {removed_count} old checkpoints")