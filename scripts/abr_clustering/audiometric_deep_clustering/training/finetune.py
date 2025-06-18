"""
Joint training (fine-tuning) for ContrastiveVAE-DEC model.

This module handles the final training stage where all objectives
(VAE, clustering, contrastive) are jointly optimized for audiometric
phenotype discovery.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Any, List
import logging
from pathlib import Path
import time
import numpy as np

from .callbacks import CallbackManager, EarlyStopping, ModelCheckpoint, LossMonitor

logger = logging.getLogger(__name__)


class JointTrainer:
    """
    Handles joint training stage for ContrastiveVAE-DEC.

    Optimizes all objectives simultaneously: reconstruction, KL divergence,
    clustering, contrastive learning, and phenotype consistency.
    """

    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        """
        Initialize joint trainer.

        Args:
            model: ContrastiveVAE-DEC model (with initialized clusters)
            config: Training configuration
            device: Training device
        """
        self.model = model
        self.config = config
        self.device = device

        # Set model to joint training stage
        self.model.set_training_stage('joint')

        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training parameters
        self.joint_epochs = config['training_stages']['stage3_joint']['epochs']
        self.warmup_epochs = config['training_stages']['stage3_joint']['warmup_epochs']
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.starting_epoch = 0

        # Clustering update management
        self.cluster_update_interval = config['clustering']['update_interval']
        self.last_cluster_update = -1

        # Loss component tracking
        self.loss_history = {
            'epoch': [],
            'train_total_loss': [],
            'val_total_loss': [],
            'train_reconstruction': [],
            'train_kl': [],
            'train_clustering': [],
            'train_contrastive': [],
            'train_phenotype': [],
            'val_reconstruction': [],
            'val_kl': [],
            'val_clustering': [],
            'cluster_quality': [],
            'learning_rate': []
        }

        # Initialize callbacks
        self.callback_manager = self._create_callbacks()

        logger.info(f"Initialized joint trainer for {self.joint_epochs} epochs")
        logger.info(f"Warmup epochs: {self.warmup_epochs}")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for joint training."""
        optimizer_config = self.config['optimizer']

        # Potentially different learning rate for joint training
        joint_lr = optimizer_config.get('joint_lr', optimizer_config['lr'])

        if optimizer_config['type'].lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=joint_lr,
                betas=optimizer_config['betas'],
                eps=optimizer_config['eps'],
                weight_decay=optimizer_config['weight_decay']
            )
        elif optimizer_config['type'].lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=joint_lr,
                betas=optimizer_config['betas'],
                eps=optimizer_config['eps'],
                weight_decay=optimizer_config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['type']}")

    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler for joint training."""
        scheduler_config = self.config['scheduler']

        if scheduler_config['type'] == 'cosine_annealing':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.joint_epochs,
                eta_min=scheduler_config['eta_min']
            )
        elif scheduler_config['type'] == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif scheduler_config['type'] == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=scheduler_config['patience'],
                factor=scheduler_config['factor'],
                mode='min'
            )
        else:
            return None

    def _create_callbacks(self) -> CallbackManager:
        """Create callback manager with monitoring callbacks."""
        callbacks = []

        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config['training']['early_stopping']['patience'],
            min_delta=self.config['training']['early_stopping']['min_delta'],
            monitor='val_total_loss'
        )
        callbacks.append(early_stopping)

        # Loss monitoring
        loss_monitor = LossMonitor(
            log_frequency=self.config['logging']['log_every_n_steps']
        )
        callbacks.append(loss_monitor)

        return CallbackManager(callbacks)

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              save_dir: Path, starting_epoch: int = 0) -> Dict[str, Any]:
        """
        Execute joint training.

        Args:
            train_loader: Training data loader (with contrastive pairs)
            val_loader: Validation data loader
            save_dir: Directory to save checkpoints
            starting_epoch: Global epoch offset from previous stages

        Returns:
            Joint training results
        """
        logger.info("Starting joint training (all objectives)")
        save_dir.mkdir(parents=True, exist_ok=True)

        self.starting_epoch = starting_epoch
        start_time = time.time()

        # Initialize callbacks
        self.callback_manager.on_train_begin(logs={'model': self.model})

        for epoch in range(self.joint_epochs):
            self.current_epoch = epoch
            global_epoch = starting_epoch + epoch

            # Callback: epoch begin
            epoch_logs = {'epoch': global_epoch, 'global_epoch': global_epoch}
            self.callback_manager.on_epoch_begin(epoch, epoch_logs)

            # Update clusters periodically
            if self._should_update_clusters(epoch):
                self._update_cluster_assignments(train_loader)

            # Training step
            train_metrics = self._train_epoch(train_loader, global_epoch)

            # Validation step
            val_metrics = self._validate_epoch(val_loader, global_epoch)

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_total_loss'])
                else:
                    self.scheduler.step()

            # Update history
            self._update_history(train_metrics, val_metrics, global_epoch)

            # Callback: epoch end
            epoch_logs.update(train_metrics)
            epoch_logs.update(val_metrics)
            self.callback_manager.on_epoch_end(epoch, epoch_logs)

            # Logging
            if epoch % 5 == 0 or epoch == self.joint_epochs - 1:
                self._log_epoch(global_epoch, train_metrics, val_metrics)

            # Save best model
            if val_metrics['val_total_loss'] < self.best_loss:
                self.best_loss = val_metrics['val_total_loss']
                self._save_checkpoint(save_dir / 'best_joint.pt', global_epoch, val_metrics)

            # Regular checkpoint
            if epoch % 25 == 0:
                self._save_checkpoint(save_dir / f'joint_epoch_{global_epoch}.pt', global_epoch, val_metrics)

            # Check for early stopping
            if self.callback_manager.should_stop():
                logger.info(f"Early stopping triggered at epoch {global_epoch}")
                break

        # Save final checkpoint
        self._save_checkpoint(save_dir / 'final_joint.pt', global_epoch, val_metrics)

        # Finalize callbacks
        final_logs = {'best_loss': self.best_loss}
        self.callback_manager.on_train_end(final_logs)

        training_time = time.time() - start_time
        logger.info(f"Joint training completed in {training_time:.2f} seconds")

        return {
            'training_history': self.loss_history,
            'best_loss': self.best_loss,
            'final_epoch': epoch,
            'training_time': training_time,
            'callback_logs': self.callback_manager.get_logs()
        }

    def _should_update_clusters(self, epoch: int) -> bool:
        """Determine if clusters should be updated this epoch."""
        if epoch == 0:  # Always update on first epoch
            return True

        if epoch - self.last_cluster_update >= self.cluster_update_interval:
            return True

        return False

    def _update_cluster_assignments(self, train_loader: DataLoader):
        """Update cluster center assignments based on current model."""
        logger.info(f"Updating cluster assignments at epoch {self.current_epoch}")

        self.model.eval()

        # Collect current latent representations
        all_latents = []
        with torch.no_grad():
            for batch in train_loader:
                batch = self._batch_to_device(batch)
                z = self.model.encode(batch['features'])
                all_latents.append(z)

        # Concatenate all latents
        latents = torch.cat(all_latents, dim=0)

        # Re-initialize clusters with current representations
        self.model.clustering_layer.initialize_clusters(latents, method='kmeans')

        self.last_cluster_update = self.current_epoch
        self.model.train()

    def _train_epoch(self, train_loader: DataLoader, global_epoch: int) -> Dict[str, float]:
        """Train for one epoch with all objectives."""
        self.model.train()

        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'kl_loss': 0.0,
            'clustering_loss': 0.0,
            'contrastive_loss': 0.0,
            'phenotype_loss': 0.0
        }

        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = self._batch_to_device(batch)

            # Callback: batch begin
            batch_logs = {'batch': batch_idx, 'size': len(batch['features'])}
            self.callback_manager.on_batch_begin(batch_idx, batch_logs)

            # Forward pass
            self.optimizer.zero_grad()

            model_output = self.model(batch['features'], return_attention=True)
            losses = self.model.compute_losses(batch, return_individual=True)

            total_loss = losses['total_loss'].mean()

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            if self.config['training']['grad_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip_norm']
                )

            self.optimizer.step()

            # Accumulate losses
            epoch_losses['total_loss'] += total_loss.item()
            epoch_losses['reconstruction_loss'] += losses['reconstruction_loss'].mean().item()
            epoch_losses['kl_loss'] += losses['kl_loss'].mean().item()
            epoch_losses['clustering_loss'] += losses.get('clustering_loss', torch.tensor(0.0)).mean().item()
            epoch_losses['contrastive_loss'] += losses.get('contrastive_loss', torch.tensor(0.0)).mean().item()
            epoch_losses['phenotype_loss'] += losses.get('phenotype_consistency_loss', torch.tensor(0.0)).mean().item()

            num_batches += 1

            # Callback: batch end
            batch_logs.update({
                'loss': total_loss.item(),
                'reconstruction': losses['reconstruction_loss'].mean().item(),
                'kl': losses['kl_loss'].mean().item()
            })
            self.callback_manager.on_batch_end(batch_idx, batch_logs)

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return {
            'train_total_loss': epoch_losses['total_loss'],
            'train_reconstruction': epoch_losses['reconstruction_loss'],
            'train_kl': epoch_losses['kl_loss'],
            'train_clustering': epoch_losses['clustering_loss'],
            'train_contrastive': epoch_losses['contrastive_loss'],
            'train_phenotype': epoch_losses['phenotype_loss'],
            'num_batches': num_batches
        }

    def _validate_epoch(self, val_loader: DataLoader, global_epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()

        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'kl_loss': 0.0,
            'clustering_loss': 0.0
        }

        cluster_quality_metrics = []
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = self._batch_to_device(batch)

                model_output = self.model(batch['features'])
                losses = self.model.compute_losses(batch)

                epoch_losses['total_loss'] += losses['total_loss'].mean().item()
                epoch_losses['reconstruction_loss'] += losses['reconstruction_loss'].mean().item()
                epoch_losses['kl_loss'] += losses['kl_loss'].mean().item()
                epoch_losses['clustering_loss'] += losses.get('clustering_loss', torch.tensor(0.0)).mean().item()

                # Analyze cluster quality if clustering is active
                if 'q' in model_output and 'cluster_centers' in model_output:
                    cluster_stats = self.model.clustering_layer.get_cluster_statistics(
                        model_output['latent_z']
                    )
                    cluster_quality_metrics.append(cluster_stats['silhouette_score'].item())

                num_batches += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        # Average cluster quality
        avg_cluster_quality = np.mean(cluster_quality_metrics) if cluster_quality_metrics else 0.0

        return {
            'val_total_loss': epoch_losses['total_loss'],
            'val_reconstruction': epoch_losses['reconstruction_loss'],
            'val_kl': epoch_losses['kl_loss'],
            'val_clustering': epoch_losses['clustering_loss'],
            'cluster_quality': avg_cluster_quality
        }

    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to training device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

    def _update_history(self, train_metrics: Dict[str, float],
                       val_metrics: Dict[str, float], global_epoch: int):
        """Update training history."""
        self.loss_history['epoch'].append(global_epoch)
        self.loss_history['train_total_loss'].append(train_metrics['train_total_loss'])
        self.loss_history['val_total_loss'].append(val_metrics['val_total_loss'])
        self.loss_history['train_reconstruction'].append(train_metrics['train_reconstruction'])
        self.loss_history['train_kl'].append(train_metrics['train_kl'])
        self.loss_history['train_clustering'].append(train_metrics['train_clustering'])
        self.loss_history['train_contrastive'].append(train_metrics['train_contrastive'])
        self.loss_history['train_phenotype'].append(train_metrics['train_phenotype'])
        self.loss_history['val_reconstruction'].append(val_metrics['val_reconstruction'])
        self.loss_history['val_kl'].append(val_metrics['val_kl'])
        self.loss_history['val_clustering'].append(val_metrics['val_clustering'])
        self.loss_history['cluster_quality'].append(val_metrics['cluster_quality'])
        self.loss_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

    def _log_epoch(self, global_epoch: int, train_metrics: Dict[str, float],
                  val_metrics: Dict[str, float]):
        """Log epoch results."""
        lr = self.optimizer.param_groups[0]['lr']

        logger.info(
            f"Joint Epoch {global_epoch:3d}: "
            f"Train Loss: {train_metrics['train_total_loss']:.4f}, "
            f"Val Loss: {val_metrics['val_total_loss']:.4f}, "
            f"Recon: {train_metrics['train_reconstruction']:.4f}, "
            f"KL: {train_metrics['train_kl']:.4f}, "
            f"Cluster: {train_metrics['train_clustering']:.4f}, "
            f"Contrast: {train_metrics['train_contrastive']:.4f}, "
            f"Quality: {val_metrics['cluster_quality']:.3f}, "
            f"LR: {lr:.6f}"
        )

    def _save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'training_history': self.loss_history,
            'config': self.config,
            'clusters_initialized': self.model.clusters_initialized,
            'best_loss': self.best_loss
        }

        torch.save(checkpoint, path)
        logger.debug(f"Saved joint training checkpoint: {path}")

    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Load joint training checkpoint."""
        logger.info(f"Loading joint training checkpoint: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch'] - self.starting_epoch
        self.loss_history = checkpoint['training_history']
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

        return checkpoint

    def analyze_joint_training(self) -> Dict[str, Any]:
        """Analyze joint training results."""
        if not self.loss_history['epoch']:
            return {'error': 'No training history available'}

        # Convergence analysis
        recent_losses = self.loss_history['val_total_loss'][-10:]
        loss_std = np.std(recent_losses) if len(recent_losses) > 1 else 0.0

        # Loss component analysis
        final_components = {
            'reconstruction': self.loss_history['train_reconstruction'][-1],
            'kl': self.loss_history['train_kl'][-1],
            'clustering': self.loss_history['train_clustering'][-1],
            'contrastive': self.loss_history['train_contrastive'][-1],
            'phenotype': self.loss_history['train_phenotype'][-1]
        }

        # Cluster quality trend
        cluster_qualities = self.loss_history['cluster_quality']
        cluster_improvement = cluster_qualities[-1] - cluster_qualities[0] if len(cluster_qualities) > 1 else 0.0

        # Best performance
        best_epoch_idx = self.loss_history['val_total_loss'].index(min(self.loss_history['val_total_loss']))
        best_epoch = self.loss_history['epoch'][best_epoch_idx]

        return {
            'converged': loss_std < 0.001,
            'best_epoch': best_epoch,
            'best_val_loss': min(self.loss_history['val_total_loss']),
            'final_loss_components': final_components,
            'cluster_quality_improvement': cluster_improvement,
            'final_cluster_quality': cluster_qualities[-1] if cluster_qualities else 0.0,
            'loss_reduction': (self.loss_history['val_total_loss'][0] - self.loss_history['val_total_loss'][-1]) / self.loss_history['val_total_loss'][0],
            'training_stable': loss_std < 0.01
        }


def run_joint_training(model: nn.Module, train_loader: DataLoader,
                      val_loader: DataLoader, config: Dict,
                      device: torch.device, save_dir: Path,
                      starting_epoch: int = 0) -> Dict[str, Any]:
    """
    Convenience function to run joint training.

    Args:
        model: ContrastiveVAE-DEC model with initialized clusters
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Training device
        save_dir: Save directory
        starting_epoch: Global epoch offset

    Returns:
        Joint training results
    """
    trainer = JointTrainer(model, config, device)
    results = trainer.train(train_loader, val_loader, save_dir, starting_epoch)

    # Add analysis
    analysis = trainer.analyze_joint_training()
    results['analysis'] = analysis

    return results