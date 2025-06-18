"""
VAE pretraining for ContrastiveVAE-DEC model.

This module handles the first training stage where only the VAE components
(encoder, decoder, latent space) are trained to learn meaningful representations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class VAEPretrainer:
    """
    Handles VAE pretraining stage for ContrastiveVAE-DEC.

    Focuses on learning good latent representations through reconstruction
    and regularization before introducing clustering objectives.
    """

    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        """
        Initialize VAE pretrainer.

        Args:
            model: ContrastiveVAE-DEC model
            config: Training configuration
            device: Training device
        """
        self.model = model
        self.config = config
        self.device = device

        # Set model to pretrain stage
        self.model.set_training_stage('pretrain')

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Training parameters
        self.pretrain_epochs = config['training']['training_stages']['stage1_pretrain']['epochs']
        self.current_epoch = 0
        self.best_loss = float('inf')

        # Monitoring
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'beta': [],
            'learning_rate': []
        }

        logger.info(f"Initialized VAE pretrainer for {self.pretrain_epochs} epochs")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for pretraining."""
        optimizer_config = self.config['training']['optimizer']

        if optimizer_config['type'].lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=float(optimizer_config['lr']),
                betas=list(optimizer_config['betas']),
                eps=float(optimizer_config['eps']),
                weight_decay=float(optimizer_config['weight_decay'])
            )
        elif optimizer_config['type'].lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=float(optimizer_config['lr']),
                betas=list(optimizer_config['betas']),
                eps=float(optimizer_config['eps']),
                weight_decay=float(optimizer_config['weight_decay'])
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['type']}")

    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler."""
        scheduler_config = self.config['training']['scheduler']

        if scheduler_config['type'] == 'cosine_annealing':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(scheduler_config['T_max']),
                eta_min=float(scheduler_config['eta_min'])
            )
        elif scheduler_config['type'] == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(scheduler_config['step_size']),
                gamma=float(scheduler_config['gamma'])
            )
        elif scheduler_config['type'] == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=int(scheduler_config['patience']),
                factor=float(scheduler_config['factor']),
                mode='min'
            )
        else:
            return None

    def pretrain(self, train_loader: DataLoader,
                val_loader: DataLoader,
                save_dir: Path) -> Dict[str, Any]:
        """
        Execute VAE pretraining.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save checkpoints

        Returns:
            Training results dictionary
        """
        logger.info("Starting VAE pretraining")
        save_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        for epoch in range(self.pretrain_epochs):
            self.current_epoch = epoch

            # Training step
            train_metrics = self._train_epoch(train_loader)

            # Validation step
            val_metrics = self._validate_epoch(val_loader)

            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()

            # Update history
            self._update_history(train_metrics, val_metrics)

            # Logging
            if epoch % 10 == 0 or epoch == self.pretrain_epochs - 1:
                self._log_epoch(epoch, train_metrics, val_metrics)

            # Save checkpoint if best model
            if val_metrics['val_loss'] < self.best_loss:
                self.best_loss = val_metrics['val_loss']
                self._save_checkpoint(save_dir / 'best_pretrain.pt', epoch, val_metrics)

            # Save regular checkpoint
            if epoch % 20 == 0:
                self._save_checkpoint(save_dir / f'pretrain_epoch_{epoch}.pt', epoch, val_metrics)

        # Save final checkpoint
        self._save_checkpoint(save_dir / 'final_pretrain.pt', epoch, val_metrics)

        training_time = time.time() - start_time
        logger.info(f"VAE pretraining completed in {training_time:.2f} seconds")

        return {
            'training_history': self.training_history,
            'best_loss': self.best_loss,
            'final_epoch': epoch,
            'training_time': training_time
        }

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_reconstruction = 0.0
        total_kl = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = self._batch_to_device(batch)

            # Forward pass
            self.optimizer.zero_grad()

            model_output = self.model(batch['features'])
            losses = self.model.compute_losses(batch)

            total_loss_batch = losses['total_loss'].mean()

            # Backward pass
            total_loss_batch.backward()

            # Gradient clipping
            if self.config['training']['training']['grad_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['training']['grad_clip_norm']
                )

            self.optimizer.step()

            # Accumulate metrics
            total_loss += total_loss_batch.item()
            total_reconstruction += losses['reconstruction_loss'].mean().item()
            total_kl += losses['kl_loss'].mean().item()
            num_batches += 1

        return {
            'train_loss': total_loss / num_batches,
            'train_reconstruction': total_reconstruction / num_batches,
            'train_kl': total_kl / num_batches,
            'num_batches': num_batches
        }

    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()

        total_loss = 0.0
        total_reconstruction = 0.0
        total_kl = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = self._batch_to_device(batch)

                model_output = self.model(batch['features'])
                losses = self.model.compute_losses(batch)

                total_loss += losses['total_loss'].mean().item()
                total_reconstruction += losses['reconstruction_loss'].mean().item()
                total_kl += losses['kl_loss'].mean().item()
                num_batches += 1

        return {
            'val_loss': total_loss / num_batches,
            'val_reconstruction': total_reconstruction / num_batches,
            'val_kl': total_kl / num_batches
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
                       val_metrics: Dict[str, float]):
        """Update training history."""
        self.training_history['epoch'].append(self.current_epoch)
        self.training_history['train_loss'].append(train_metrics['train_loss'])
        self.training_history['val_loss'].append(val_metrics['val_loss'])
        self.training_history['reconstruction_loss'].append(train_metrics['train_reconstruction'])
        self.training_history['kl_loss'].append(train_metrics['train_kl'])
        self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

        # Get current beta from model if available
        if hasattr(self.model, 'compute_losses'):
            dummy_batch = {'features': torch.randn(1, 18, device=self.device)}
            dummy_output = self.model(dummy_batch['features'])
            dummy_losses = self.model.compute_losses(dummy_batch)
            self.training_history['beta'].append(dummy_losses.get('current_beta', 1.0))
        else:
            self.training_history['beta'].append(1.0)

    def _log_epoch(self, epoch: int, train_metrics: Dict[str, float],
                  val_metrics: Dict[str, float]):
        """Log epoch results."""
        lr = self.optimizer.param_groups[0]['lr']

        logger.info(
            f"Pretrain Epoch {epoch:3d}/{self.pretrain_epochs}: "
            f"Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Val Loss: {val_metrics['val_loss']:.4f}, "
            f"Recon: {train_metrics['train_reconstruction']:.4f}, "
            f"KL: {train_metrics['train_kl']:.4f}, "
            f"LR: {lr:.6f}"
        )

    def _save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'training_history': self.training_history,
            'config': self.config
        }

        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Load pretrained checkpoint."""
        logger.info(f"Loading pretrain checkpoint: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint['training_history']

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

        return checkpoint

    def analyze_pretraining(self) -> Dict[str, Any]:
        """Analyze pretraining results."""
        if not self.training_history['epoch']:
            return {'error': 'No training history available'}

        # Convergence analysis
        recent_losses = self.training_history['val_loss'][-10:]
        loss_std = torch.tensor(recent_losses).std().item() if len(recent_losses) > 1 else 0.0

        # Best performance
        best_epoch = self.training_history['val_loss'].index(min(self.training_history['val_loss']))

        # Loss component analysis
        final_reconstruction = self.training_history['reconstruction_loss'][-1]
        final_kl = self.training_history['kl_loss'][-1]

        return {
            'converged': loss_std < 0.001,
            'best_epoch': best_epoch,
            'best_val_loss': min(self.training_history['val_loss']),
            'final_reconstruction_loss': final_reconstruction,
            'final_kl_loss': final_kl,
            'reconstruction_kl_ratio': final_reconstruction / (final_kl + 1e-8),
            'loss_reduction': (self.training_history['val_loss'][0] - self.training_history['val_loss'][-1]) / self.training_history['val_loss'][0]
        }


def run_pretraining(model: nn.Module, train_loader: DataLoader,
                   val_loader: DataLoader, config: Dict,
                   device: torch.device, save_dir: Path) -> Dict[str, Any]:
    """
    Convenience function to run VAE pretraining.

    Args:
        model: ContrastiveVAE-DEC model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Training device
        save_dir: Save directory

    Returns:
        Pretraining results
    """
    pretrainer = VAEPretrainer(model, config, device)
    results = pretrainer.pretrain(train_loader, val_loader, save_dir)

    # Add analysis
    analysis = pretrainer.analyze_pretraining()
    results['analysis'] = analysis

    return results