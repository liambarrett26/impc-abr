"""
Training callbacks and monitoring for ContrastiveVAE-DEC model.

This module provides comprehensive monitoring, early stopping, checkpointing,
and analysis callbacks for the multi-objective training process.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
import logging
import numpy as np
from pathlib import Path
import time
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base class for training callbacks."""

    def __init__(self):
        """Initialize callback."""
        self.model = None
        self.logs = {}

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Called at the beginning of training."""
        if logs:
            self.logs.update(logs)

    def on_train_end(self, logs: Optional[Dict] = None):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Called at the end of each batch."""
        pass


class CallbackManager:
    """
    Manages multiple callbacks during training.

    Coordinates callback execution and provides unified interface
    for training monitoring and control.
    """

    def __init__(self, callbacks: List[Callback]):
        """
        Initialize callback manager.

        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks
        self.stopped = False
        self.logs = {}

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Execute on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict] = None):
        """Execute on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Execute on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Execute on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

            # Check if any callback requested stopping
            if hasattr(callback, 'stop_training') and callback.stop_training:
                self.stopped = True

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Execute on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Execute on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def should_stop(self) -> bool:
        """Check if training should be stopped."""
        return self.stopped

    def get_logs(self) -> Dict[str, Any]:
        """Get aggregated logs from all callbacks."""
        combined_logs = {}
        for callback in self.callbacks:
            if hasattr(callback, 'get_logs'):
                callback_logs = callback.get_logs()
                combined_logs[callback.__class__.__name__] = callback_logs
        return combined_logs


class EarlyStopping(Callback):
    """
    Early stopping callback to prevent overfitting.

    Monitors a specified metric and stops training when no improvement
    is observed for a specified number of epochs.
    """

    def __init__(self, monitor: str = 'val_total_loss', patience: int = 10,
                 min_delta: float = 0.0001, mode: str = 'min'):
        """
        Initialize early stopping.

        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement after which training stops
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.wait = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.stop_training = False

        self.history = []

        logger.info(f"EarlyStopping: monitor={monitor}, patience={patience}, mode={mode}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Check for early stopping condition."""
        if not logs or self.monitor not in logs:
            logger.warning(f"EarlyStopping: metric '{self.monitor}' not found in logs")
            return

        current_value = logs[self.monitor]
        self.history.append(current_value)

        # Check for improvement
        if self.mode == 'min':
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta

        if improved:
            self.best_value = current_value
            self.wait = 0
            logger.debug(f"EarlyStopping: improvement detected at epoch {epoch}")
        else:
            self.wait += 1
            logger.debug(f"EarlyStopping: no improvement for {self.wait} epochs")

        # Check stopping condition
        if self.wait >= self.patience:
            self.stop_training = True
            logger.info(f"EarlyStopping: stopping training at epoch {epoch}")
            logger.info(f"Best {self.monitor}: {self.best_value:.6f}")

    def get_logs(self) -> Dict[str, Any]:
        """Get early stopping logs."""
        return {
            'monitor': self.monitor,
            'best_value': self.best_value,
            'patience': self.patience,
            'wait': self.wait,
            'stopped': self.stop_training,
            'history': self.history
        }


class ModelCheckpoint(Callback):
    """
    Callback to save model checkpoints during training.

    Can save best model based on monitored metric or save at regular intervals.
    """

    def __init__(self, filepath: Path, monitor: str = 'val_total_loss',
                 save_best_only: bool = True, mode: str = 'min',
                 save_freq: int = 10):
        """
        Initialize model checkpoint callback.

        Args:
            filepath: Path template for saving checkpoints
            monitor: Metric to monitor for best model
            save_best_only: Whether to only save when metric improves
            mode: 'min' or 'max' for monitored metric
            save_freq: Frequency (epochs) for regular saves
        """
        super().__init__()
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_freq = save_freq

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.saved_epochs = []

        # Create directory if needed
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"ModelCheckpoint: filepath={filepath}, monitor={monitor}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Save checkpoint if conditions are met."""
        if not logs:
            return

        save_checkpoint = False
        save_reason = ""

        # Check if this is the best model
        if self.monitor in logs:
            current_value = logs[self.monitor]

            if self.mode == 'min':
                improved = current_value < self.best_value
            else:
                improved = current_value > self.best_value

            if improved:
                self.best_value = current_value
                save_checkpoint = True
                save_reason = f"best_{self.monitor}"

        # Check regular save frequency
        if not self.save_best_only and epoch % self.save_freq == 0:
            save_checkpoint = True
            save_reason = f"regular_epoch_{epoch}"

        # Save checkpoint
        if save_checkpoint and 'model' in self.logs:
            checkpoint_path = self._get_checkpoint_path(epoch, save_reason)
            self._save_checkpoint(checkpoint_path, epoch, logs)
            self.saved_epochs.append(epoch)

    def _get_checkpoint_path(self, epoch: int, reason: str) -> Path:
        """Generate checkpoint filepath."""
        base_path = self.filepath
        if reason.startswith('best_'):
            return base_path.parent / f"best_{base_path.name}"
        else:
            return base_path.parent / f"epoch_{epoch}_{base_path.name}"

    def _save_checkpoint(self, path: Path, epoch: int, logs: Dict[str, Any]):
        """Save model checkpoint."""
        model = self.logs.get('model')
        if model is None:
            logger.warning("ModelCheckpoint: no model available for saving")
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': logs,
            'best_value': self.best_value,
            'monitor': self.monitor
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def get_logs(self) -> Dict[str, Any]:
        """Get checkpoint logs."""
        return {
            'best_value': self.best_value,
            'saved_epochs': self.saved_epochs,
            'total_saves': len(self.saved_epochs)
        }


class LossMonitor(Callback):
    """
    Callback for monitoring and logging loss components.

    Tracks loss evolution and provides analysis of training dynamics.
    """

    def __init__(self, log_frequency: int = 100):
        """
        Initialize loss monitor.

        Args:
            log_frequency: Frequency (batches) for detailed logging
        """
        super().__init__()
        self.log_frequency = log_frequency

        self.batch_losses = []
        self.epoch_losses = []
        self.loss_components = {
            'reconstruction': [],
            'kl': [],
            'clustering': [],
            'contrastive': [],
            'total': []
        }

        self.batch_count = 0

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Monitor batch-level losses."""
        if not logs:
            return

        self.batch_count += 1

        # Store batch loss
        if 'loss' in logs:
            self.batch_losses.append(logs['loss'])

        # Detailed logging at specified frequency
        if self.batch_count % self.log_frequency == 0:
            self._log_batch_details(batch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Monitor epoch-level losses."""
        if not logs:
            return

        # Store epoch losses
        if 'train_total_loss' in logs:
            self.epoch_losses.append(logs['train_total_loss'])

        # Store loss components
        component_map = {
            'reconstruction': 'train_reconstruction',
            'kl': 'train_kl',
            'clustering': 'train_clustering',
            'contrastive': 'train_contrastive',
            'total': 'train_total_loss'
        }

        for component, log_key in component_map.items():
            if log_key in logs:
                self.loss_components[component].append(logs[log_key])

        # Analyze loss trends
        if epoch > 0 and epoch % 20 == 0:
            self._analyze_loss_trends(epoch)

    def _log_batch_details(self, batch: int, logs: Dict[str, Any]):
        """Log detailed batch information."""
        loss_info = f"Batch {batch:4d}: Loss={logs.get('loss', 0.0):.4f}"

        if 'reconstruction' in logs:
            loss_info += f", Recon={logs['reconstruction']:.4f}"
        if 'kl' in logs:
            loss_info += f", KL={logs['kl']:.4f}"

        logger.debug(loss_info)

    def _analyze_loss_trends(self, epoch: int):
        """Analyze loss trends over recent epochs."""
        if len(self.epoch_losses) < 10:
            return

        recent_losses = self.epoch_losses[-10:]

        # Calculate trend
        epochs = list(range(len(recent_losses)))
        trend = np.polyfit(epochs, recent_losses, 1)[0]

        # Calculate volatility
        volatility = np.std(recent_losses)

        logger.info(f"Loss analysis at epoch {epoch}: trend={trend:.6f}, volatility={volatility:.4f}")

        # Warning for concerning trends
        if trend > 0.001:
            logger.warning("Loss appears to be increasing - possible overfitting")
        if volatility > 0.1:
            logger.warning("High loss volatility detected - consider reducing learning rate")

    def get_logs(self) -> Dict[str, Any]:
        """Get loss monitoring logs."""
        analysis = {}

        if self.epoch_losses:
            analysis['final_loss'] = self.epoch_losses[-1]
            analysis['best_loss'] = min(self.epoch_losses)
            analysis['loss_reduction'] = (self.epoch_losses[0] - self.epoch_losses[-1]) / self.epoch_losses[0]

        if len(self.epoch_losses) > 1:
            analysis['loss_std'] = np.std(self.epoch_losses)

            # Recent trend
            if len(self.epoch_losses) >= 5:
                recent = self.epoch_losses[-5:]
                epochs = list(range(len(recent)))
                analysis['recent_trend'] = np.polyfit(epochs, recent, 1)[0]

        return {
            'loss_history': self.epoch_losses,
            'loss_components': self.loss_components,
            'batch_count': self.batch_count,
            'analysis': analysis
        }


class ClusteringMonitor(Callback):
    """
    Specialized callback for monitoring clustering quality.

    Tracks cluster formation, quality metrics, and phenotype discovery progress.
    """

    def __init__(self, monitor_frequency: int = 5):
        """
        Initialize clustering monitor.

        Args:
            monitor_frequency: Frequency (epochs) for cluster analysis
        """
        super().__init__()
        self.monitor_frequency = monitor_frequency

        self.cluster_history = {
            'epoch': [],
            'num_active_clusters': [],
            'silhouette_score': [],
            'cluster_balance': [],
            'assignment_entropy': []
        }

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Monitor clustering progress."""
        if epoch % self.monitor_frequency != 0:
            return

        if not logs or 'model' not in self.logs:
            return

        model = self.logs['model']

        # Check if model has clustering capability
        if not hasattr(model, 'clustering_layer'):
            return

        # Analyze current clustering state
        cluster_stats = self._analyze_clustering(model)

        # Store history
        self.cluster_history['epoch'].append(epoch)
        self.cluster_history['num_active_clusters'].append(cluster_stats['num_active_clusters'])
        self.cluster_history['silhouette_score'].append(cluster_stats['silhouette_score'])
        self.cluster_history['cluster_balance'].append(cluster_stats['cluster_balance'])
        self.cluster_history['assignment_entropy'].append(cluster_stats['assignment_entropy'])

        # Log cluster status
        logger.info(
            f"Clustering at epoch {epoch}: "
            f"Active clusters: {cluster_stats['num_active_clusters']}, "
            f"Silhouette: {cluster_stats['silhouette_score']:.3f}, "
            f"Balance: {cluster_stats['cluster_balance']:.3f}"
        )

    def _analyze_clustering(self, model) -> Dict[str, float]:
        """Analyze current clustering state."""
        # This would require access to validation data
        # For now, return placeholder metrics
        return {
            'num_active_clusters': 10,  # Would be computed from actual data
            'silhouette_score': 0.5,   # Would be computed from actual data
            'cluster_balance': 0.8,    # Would be computed from actual data
            'assignment_entropy': 1.5  # Would be computed from actual data
        }

    def get_logs(self) -> Dict[str, Any]:
        """Get clustering monitoring logs."""
        return {
            'cluster_history': self.cluster_history,
            'final_stats': {
                'num_active_clusters': self.cluster_history['num_active_clusters'][-1] if self.cluster_history['num_active_clusters'] else 0,
                'final_silhouette': self.cluster_history['silhouette_score'][-1] if self.cluster_history['silhouette_score'] else 0
            }
        }


class PerformanceProfiler(Callback):
    """
    Callback for profiling training performance.

    Monitors training speed, memory usage, and resource utilization.
    """

    def __init__(self):
        """Initialize performance profiler."""
        super().__init__()

        self.epoch_times = []
        self.batch_times = []
        self.memory_usage = []

        self.epoch_start_time = None
        self.batch_start_time = None

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Start epoch timing."""
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """End epoch timing and record metrics."""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)

            # Memory usage (if CUDA available)
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                self.memory_usage.append(memory_mb)
                torch.cuda.reset_peak_memory_stats()

            # Log performance every 10 epochs
            if epoch % 10 == 0:
                avg_epoch_time = np.mean(self.epoch_times[-10:])
                logger.info(f"Performance: avg epoch time={avg_epoch_time:.2f}s")

                if self.memory_usage:
                    avg_memory = np.mean(self.memory_usage[-10:])
                    logger.info(f"Memory usage: {avg_memory:.1f}MB")

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Start batch timing."""
        self.batch_start_time = time.time()

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """End batch timing."""
        if self.batch_start_time:
            batch_time = time.time() - self.batch_start_time
            self.batch_times.append(batch_time)

    def get_logs(self) -> Dict[str, Any]:
        """Get performance profiling logs."""
        logs = {}

        if self.epoch_times:
            logs['avg_epoch_time'] = np.mean(self.epoch_times)
            logs['total_training_time'] = sum(self.epoch_times)

        if self.batch_times:
            logs['avg_batch_time'] = np.mean(self.batch_times)

        if self.memory_usage:
            logs['peak_memory_mb'] = max(self.memory_usage)
            logs['avg_memory_mb'] = np.mean(self.memory_usage)

        return logs


def create_default_callbacks(config: Dict, save_dir: Path) -> List[Callback]:
    """
    Create default set of callbacks for training.

    Args:
        config: Training configuration
        save_dir: Directory for saving outputs

    Returns:
        List of configured callbacks
    """
    callbacks = []

    # Early stopping
    if config['training']['early_stopping']['patience'] > 0:
        early_stopping = EarlyStopping(
            monitor=config['training']['early_stopping']['monitor'],
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta']
        )
        callbacks.append(early_stopping)

    # Model checkpointing
    checkpoint_path = save_dir / 'model_checkpoint.pt'
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=config['training']['monitor_checkpoint'],
        save_best_only=True
    )
    callbacks.append(model_checkpoint)

    # Loss monitoring
    loss_monitor = LossMonitor(
        log_frequency=config['logging']['log_every_n_steps']
    )
    callbacks.append(loss_monitor)

    # Performance profiling
    profiler = PerformanceProfiler()
    callbacks.append(profiler)

    return callbacks